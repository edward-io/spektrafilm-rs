//! Scene-linear output from rawler, without its above-white color remapping.
use anyhow::{Result, ensure};
use rawler::{
    RawImage,
    imgop::{
        chromatic_adaption::bradford_adaption_matrix,
        develop::{Intermediate, ProcessingStep, RawDevelop},
        matrix::{IDENTITY_MATRIX_3, multiply, normalize, pseudo_inverse},
        raw::clip_negative,
        sensor::SensorType,
        xyz::{Illuminant, SRGB_TO_XYZ_D65},
    },
    pixarray::Color2D,
    rawimage::{RawImageData, RawPhotometricInterpretation},
};
use rayon::prelude::*;
use spektrafilm_math::{
    image::ImageBuf,
    precision::{Scalar, from_f64},
};

pub(super) fn develop(raw: &mut RawImage) -> Result<ImageBuf> {
    let mut developer = RawDevelop::default();
    // Calibrate bundles the camera matrix with clip_euclidean_norm_avg, whose
    // discontinuity at RGB=1 destroys highlight color before our film pipeline.
    // Upstream issue: https://github.com/dnglab/dnglab/issues/884
    // TODO: use upstream calibration again once our rawler version fixes #884
    // and we can preserve the highlight handling for each sensor layout.
    developer
        .steps
        .retain(|step| !matches!(step, ProcessingStep::Calibrate | ProcessingStep::SRgb));
    let wb = white_balance_gains(raw);
    let capped_before_demosaic = cap_bayer_highlights(raw, wb)?;
    if capped_before_demosaic {
        developer
            .steps
            .retain(|step| *step != ProcessingStep::Rescale);
    }
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
        Intermediate::ThreeColor(p) => color(p, raw, wb, capped_before_demosaic),
        Intermediate::FourColor(p) => color(p, raw, wb, capped_before_demosaic),
    }
}

fn cap_bayer_highlights(raw: &mut RawImage, wb: [f32; 4]) -> Result<bool> {
    let RawPhotometricInterpretation::Cfa(config) = &raw.photometric else {
        return Ok(false);
    };
    if raw.cpp != 1
        || config.sensor != SensorType::Bayer
        || !config.cfa.is_rgb()
        || raw.width < 2
        || raw.height < 2
        || !wb[..3].iter().all(|gain| gain.is_finite() && *gain > 0.)
    {
        return Ok(false);
    }
    let gains = std::array::from_fn::<_, 4, _>(|i| wb[config.cfa.color_at(i / 2, i % 2)]);
    let white = gains.into_iter().fold(f32::INFINITY, f32::min);
    let ceilings = gains.map(|gain| white / gain);
    // Normalize sensor white to 1, as RawDevelop normally does before demosaicing.
    raw.apply_scaling()
        .map_err(|e| anyhow::anyhow!("RAW rescale: {e:?}"))?;
    let RawImageData::Float(samples) = &mut raw.data else {
        anyhow::bail!("RAW rescale did not produce floating-point samples");
    };
    let width = raw.width;
    samples.par_chunks_exact_mut(width * 2).for_each(|rows| {
        let (top, bottom) = rows.split_at_mut(width);
        for (top, bottom) in top.chunks_exact_mut(2).zip(bottom.chunks_exact_mut(2)) {
            // A strong sample alone can be real color. Use all four sites
            // in the Bayer cell to decide how much of the cap to apply.
            let weakest = (top[0] * gains[0])
                .min(top[1] * gains[1])
                .min(bottom[0] * gains[2])
                .min(bottom[1] * gains[3]);
            if weakest <= 0.9 * white {
                continue;
            }
            let t = ((weakest / white - 0.9) / 0.1).min(1.);
            let weight = t * t * (3. - 2. * t);
            for (sample, ceiling) in top.iter_mut().chain(bottom).zip(ceilings) {
                *sample -= (*sample - ceiling).max(0.) * weight;
            }
        }
    });
    Ok(true)
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

fn white_balance_gains(raw: &RawImage) -> [f32; 4] {
    if raw.wb_coeffs[0].is_nan() {
        [1.; 4]
    } else {
        raw.wb_coeffs
    }
}

fn color<const N: usize>(
    pixels: Color2D<f32, N>,
    raw: &RawImage,
    wb: [f32; 4],
    capped_before_demosaic: bool,
) -> Result<ImageBuf> {
    let matrix = camera_to_rgb(raw)?;
    ensure!(
        wb[..N].iter().all(|v| v.is_finite()),
        "RAW white balance contains non-finite gains"
    );
    // Sensor white is 1 before white balance. The smallest gain sets the
    // highest level all channels can measure for a neutral highlight.
    let neutral_white = wb[..N].iter().copied().fold(f32::INFINITY, f32::min);
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
            let rgb = clip_negative(&rgb);
            let rgb = if capped_before_demosaic {
                rgb
            } else {
                neutralize_highlight(rgb, balanced, neutral_white)
            };
            rgb.map(|v| from_f64(v as f64))
        })
        .collect();
    Ok(ImageBuf::from_data(
        pixels.width as u32,
        pixels.height as u32,
        data.into_flattened(),
    ))
}

fn neutralize_highlight<const N: usize>(
    rgb: [f32; 3],
    balanced: [f32; N],
    neutral_white: f32,
) -> [f32; 3] {
    let weakest = balanced.into_iter().fold(f32::INFINITY, f32::min);
    let knee = neutral_white * 0.95;
    if neutral_white <= 0. || weakest <= knee {
        return rgb;
    }
    // Fade the unreliable chroma into gray while keeping scene-linear luminance.
    let t = ((weakest - knee) / (neutral_white - knee)).min(1.);
    let weight = t * t * (3. - 2. * t);
    let [r, g, b] = SRGB_TO_XYZ_D65[1];
    let luminance = rgb[0] * r + rgb[1] * g + rgb[2] * b;
    rgb.map(|channel| channel + (luminance - channel) * weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rawler::{
        CFA,
        imgop::xyz::XYZ_TO_SRGB_D65,
        rawimage::{BlackLevel, CFAConfig, WhiteLevel},
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
        let mut raw = linear_raw(&[
            [0.9998, 0.2, 0.1],
            [1.0002, 0.2, 0.1],
            [0.5, 0.1, 0.05],
            [2., 0.4, 0.2],
        ]);
        let fixed = develop(&mut raw).unwrap();
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
                    develop(&mut raw).unwrap().data,
                    expected.data,
                    "{illuminant:?}, {wb:?}"
                );
            }
        }
    }

    #[test]
    fn bayer_cap_uses_all_four_sites_and_preserves_color_below_the_knee() {
        let mut raw = linear_raw(&[[0.]]);
        raw.width = 7;
        raw.height = 3;
        raw.photometric = RawPhotometricInterpretation::Cfa(CFAConfig::new(
            &CFA::new("BGGR"),
            &raw.camera.plane_color,
        ));
        let values = [
            1.05, 1.05, 1.05, 0.95, 0.25, 1.05, 1.05, // B G B G B G
            1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05, // G R G R G R
            1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05,
        ];
        raw.data = RawImageData::Float(values.to_vec());
        assert!(cap_bayer_highlights(&mut raw, [2., 1., 1.5, f32::NAN]).unwrap());
        let samples = raw.data.as_f32();
        let expected = [
            1. / 1.5,
            1.,
            (1.05 + 1. / 1.5) / 2.,
            0.95,
            0.25,
            1.05,
            1.05,
            1.,
            0.5,
            1.025,
            0.775,
            1.05,
            1.05,
            1.05,
            1.05,
            1.05,
            1.05,
            1.05,
            1.05,
            1.05,
            1.05,
        ];
        for (sample, expected) in samples.iter().zip(expected) {
            assert!((sample - expected).abs() < 1e-6, "{sample} != {expected}");
        }
    }

    #[test]
    fn sensor_saturation_neutralizes_highlights_without_clipping_luminance() {
        let mut raw = linear_raw(&[
            [1.05, 1.05, 1.05],
            [0.7, 0.975, 0.9],
            [0.7, 0.95, 0.9],
            [1.05, 0.5, 0.3],
        ]);
        raw.wb_coeffs = [2.75, 1., 1.37, f32::NAN];
        let output = develop(&mut raw).unwrap();
        let [r, g, b] = output.get(0, 0);
        let [yr, yg, yb] = SRGB_TO_XYZ_D65[1];
        let expected_luminance = from_f64((1.05 * (2.75 * yr + yg + 1.37 * yb)) as f64);
        assert!(g > 1.);
        assert!((g - expected_luminance).abs() < 1e-6);
        assert!((r - g).abs() < 1e-6 && (b - g).abs() < 1e-6);
        let [r, g, b] = output.get(1, 0);
        assert!(r > g && r - g < 0.5);
        assert!(b > g);
        let [r, g, b] = output.get(2, 0);
        assert!(r - g > 0.9 && b > g);
        let [r, g, b] = output.get(3, 0);
        assert!((r - 1.05 * 2.75).abs() < 1e-6);
        assert!((g - 0.5).abs() < 1e-6 && (b - 0.3 * 1.37).abs() < 1e-6);
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
        let fixed = develop(&mut raw).unwrap();
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
            develop(&mut raw).unwrap().data,
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
            assert!(develop(&mut raw).is_err());
        }
    }
}
