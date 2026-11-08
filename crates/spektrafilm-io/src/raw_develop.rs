//! Scene-linear output from rawler, without its above-white color remapping.
use anyhow::{Result, ensure};
use rawler::{
    RawImage,
    imgop::{
        chromatic_adaption::bradford_adaption_matrix,
        develop::{Intermediate, ProcessingStep, RawDevelop},
        matrix::{IDENTITY_MATRIX_3, multiply, normalize, pseudo_inverse},
        raw::clip_negative,
        xyz::{Illuminant, SRGB_TO_XYZ_D65},
    },
    pixarray::Color2D,
};
use rayon::prelude::*;
use spektrafilm_math::{
    image::ImageBuf,
    precision::{Scalar, from_f64},
};

pub(super) fn develop(raw: &RawImage) -> Result<ImageBuf> {
    let mut developer = RawDevelop::default();
    // Calibrate bundles the camera matrix with clip_euclidean_norm_avg, whose
    // discontinuity at RGB=1 destroys highlight color before our film pipeline.
    // Upstream issue: https://github.com/dnglab/dnglab/issues/884
    // TODO: use upstream calibration again once #884 is fixed and the rawler
    // version we depend on includes the fix, preserving unbounded linear output.
    developer
        .steps
        .retain(|step| !matches!(step, ProcessingStep::Calibrate | ProcessingStep::SRgb));
    let camera = developer
        .develop_intermediate(raw)
        .map_err(|e| anyhow::anyhow!("RAW develop: {e:?}"))?;
    match camera {
        Intermediate::Monochrome(p) => Ok(ImageBuf::from_data(
            p.width as u32,
            p.height as u32,
            p.data
                .into_iter()
                .flat_map(|v| [from_f64(v as f64); 3])
                .collect(),
        )),
        Intermediate::ThreeColor(p) => color(p, raw),
        Intermediate::FourColor(p) => color(p, raw),
    }
}

fn camera_to_rgb(raw: &RawImage) -> Result<[[f32; 4]; 3]> {
    // Keep RawDevelop's matrix preference and normalization, including older
    // cameras whose calibration must be adapted to D65.
    let (illuminant, values) = raw
        .color_matrix_find_first([
            Illuminant::D65,
            Illuminant::A,
            Illuminant::B,
            Illuminant::C,
            Illuminant::D50,
            Illuminant::D55,
            Illuminant::D75,
            Illuminant::Daylight,
            Illuminant::Flash,
        ])
        .unwrap_or_else(|| {
            tracing::warn!("RAW has no supported color matrix; using identity calibration");
            (Illuminant::D65, IDENTITY_MATRIX_3.as_flattened().to_vec())
        });
    ensure!(
        matches!(values.len(), 9 | 12) && values.iter().all(|v| v.is_finite()),
        "RAW color matrix must contain 9 or 12 finite values"
    );
    let mut xyz_to_camera = [[0.; 3]; 4];
    for (row, values) in xyz_to_camera.iter_mut().zip(values.chunks_exact(3)) {
        row.copy_from_slice(values);
    }
    if illuminant != Illuminant::D65 {
        xyz_to_camera = multiply(
            &xyz_to_camera,
            &bradford_adaption_matrix(&illuminant, &Illuminant::D65),
        );
    }
    let matrix = pseudo_inverse(normalize(multiply(&xyz_to_camera, &SRGB_TO_XYZ_D65)));
    ensure!(
        matrix.iter().flatten().all(|v| v.is_finite()),
        "RAW color matrix is singular"
    );
    Ok(matrix)
}

fn color<const N: usize>(pixels: Color2D<f32, N>, raw: &RawImage) -> Result<ImageBuf> {
    let matrix = camera_to_rgb(raw)?;
    let wb = if raw.wb_coeffs[0].is_nan() {
        [1.; 4]
    } else {
        raw.wb_coeffs
    };
    ensure!(
        wb[..N].iter().all(|v| v.is_finite()),
        "RAW white balance contains non-finite gains"
    );
    let data: Vec<[Scalar; 3]> = pixels
        .data
        .into_par_iter()
        .map(|pixel| {
            let balanced: [f32; N] = std::array::from_fn(|c| pixel[c] * wb[c]);
            let rgb = std::array::from_fn(|c| {
                let mut value = matrix[c][0] * balanced[0]
                    + matrix[c][1] * balanced[1]
                    + matrix[c][2] * balanced[2];
                if N == 4 {
                    value += matrix[c][3] * balanced[3];
                }
                value
            });
            // Preserve rawler's negative-value handling; leave highlights unbounded.
            clip_negative(&rgb).map(|v| from_f64(v as f64))
        })
        .collect();
    Ok(ImageBuf::from_data(
        pixels.width as u32,
        pixels.height as u32,
        data.into_flattened(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rawler::{
        imgop::xyz::XYZ_TO_SRGB_D65,
        rawimage::{BlackLevel, RawImageData, RawPhotometricInterpretation, WhiteLevel},
    };
    use std::path::Path;

    fn linear_raw<const N: usize>(pixels: &[[f32; N]]) -> RawImage {
        let mut raw = rawler::decode_file(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/synthetic.dng"),
        )
        .unwrap();
        raw.width = pixels.len();
        raw.height = 1;
        raw.cpp = N;
        raw.photometric = RawPhotometricInterpretation::LinearRaw;
        raw.data = RawImageData::Float(pixels.iter().flatten().copied().collect());
        raw.blacklevel = BlackLevel::zero(1, 1, N);
        raw.whitelevel = WhiteLevel::new(vec![1; N]);
        raw.wb_coeffs = [1.; 4];
        raw.crop_area = None;
        raw.active_area = None;
        raw.color_matrix.clear();
        raw.color_matrix
            .insert(Illuminant::D65, XYZ_TO_SRGB_D65.as_flattened().to_vec());
        raw
    }

    fn stock(raw: &RawImage) -> ImageBuf {
        let mut developer = RawDevelop::default();
        developer.steps.retain(|s| *s != ProcessingStep::SRgb);
        let Intermediate::ThreeColor(p) = developer.develop_intermediate(raw).unwrap() else {
            panic!("expected RGB");
        };
        ImageBuf::from_data(
            p.width as u32,
            p.height as u32,
            p.data
                .into_flattened()
                .into_iter()
                .map(|v| from_f64(v as f64))
                .collect(),
        )
    }

    #[test]
    fn highlights_are_continuous_and_keep_exposure_and_color_ratios() {
        let raw = linear_raw(&[
            [0.9998, 0.2, 0.1],
            [1.0002, 0.2, 0.1],
            [0.5, 0.1, 0.05],
            [2., 0.4, 0.2],
        ]);
        let fixed = develop(&raw).unwrap();
        let old = stock(&raw);
        assert!(
            (old.get(1, 0)[1] - old.get(0, 0)[1]).abs() > 0.1,
            "fixture must expose the original discontinuity"
        );
        assert!(fixed.get(1, 0)[0] > 1.);
        for c in 0..3 {
            assert!((fixed.get(1, 0)[c] - fixed.get(0, 0)[c]).abs() < 0.001);
            assert!((fixed.get(3, 0)[c] - fixed.get(2, 0)[c] * 4.).abs() < 1e-5);
        }
        assert_eq!(
            fixed.get(0, 0),
            old.get(0, 0),
            "below-white development must be unchanged"
        );
    }

    #[test]
    fn ordinary_colors_and_white_balance_match_stock_with_d65_or_adapted_matrices() {
        for illuminant in [Illuminant::D65, Illuminant::A, Illuminant::D50] {
            let mut raw = linear_raw(&[[0.1, 0.1, 0.1], [0.25, 0.12, 0.06], [0.03, 0.2, 0.12]]);
            raw.color_matrix.clear();
            raw.color_matrix
                .insert(illuminant, XYZ_TO_SRGB_D65.as_flattened().to_vec());
            for wb in [[1.7, 1., 2.1, f32::NAN], [f32::NAN; 4]] {
                raw.wb_coeffs = wb;
                let expected = stock(&raw);
                assert!(
                    expected.data.iter().all(|v| *v < 0.9),
                    "fixture must stay below the threshold"
                );
                assert_eq!(
                    develop(&raw).unwrap().data,
                    expected.data,
                    "{illuminant:?}, {wb:?}"
                );
            }
        }
    }

    #[test]
    fn four_channel_cameras_include_the_fourth_channel_and_preserve_headroom() {
        let mut raw = linear_raw(&[
            [0.05, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.1],
            [1., 1., 1., 1.],
        ]);
        let mut matrix = XYZ_TO_SRGB_D65.as_flattened().to_vec();
        matrix.extend(XYZ_TO_SRGB_D65[1]);
        raw.color_matrix.insert(Illuminant::D65, matrix);
        raw.wb_coeffs = [1.7, 1., 2.1, 1.3];
        let fixed = develop(&raw).unwrap();
        let old = stock(&raw);
        assert_eq!(fixed.get(0, 0), old.get(0, 0));
        assert_eq!(fixed.get(1, 0), old.get(1, 0));
        assert_ne!(fixed.get(0, 0), fixed.get(1, 0));
        assert!(fixed.get(2, 0).iter().any(|v| *v > 1.));
    }

    #[test]
    fn monochrome_retains_values_above_white_without_color_calibration() {
        let mut raw = linear_raw(&[[0.25], [1.5], [3.]]);
        raw.color_matrix.clear();
        assert_eq!(
            develop(&raw).unwrap().data,
            vec![from_f64(0.25); 3]
                .into_iter()
                .chain([from_f64(1.5); 3])
                .chain([from_f64(3.); 3])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn malformed_calibration_returns_errors() {
        let mut raw = linear_raw(&[[0.1, 0.1, 0.1]]);
        for matrix in [vec![1.; 8], vec![0.; 9], vec![f32::NAN; 9]] {
            raw.color_matrix.insert(Illuminant::D65, matrix);
            assert!(develop(&raw).is_err());
        }
    }
}
