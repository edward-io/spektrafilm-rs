//! Built-in Lensfun 0.3; the XML database is discovered or downloaded on first use.
//! https://github.com/lensfun/lensfun/blob/v0.3.4/include/lensfun/lensfun.h.in

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
use std::{
    ffi::{CString, c_void},
    path::Path,
};
type Ptr = *mut c_void;

#[derive(Debug)]
pub struct CorrectionFailure;
impl std::fmt::Display for CorrectionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Lens correction failed")
    }
}
impl std::error::Error for CorrectionFailure {}
/// Returns false when no camera/lens is in the database, matching Python.
pub fn correct(image: &mut ImageBuf, path: &Path, database: Option<&Path>) -> Result<bool> {
    correct_inner(image, path, database).context(CorrectionFailure)
}
fn correct_inner(image: &mut ImageBuf, path: &Path, database: Option<&Path>) -> Result<bool> {
    use little_exif::{exif_tag::ExifTag, metadata::Metadata};
    let metadata = Metadata::new_from_path(path).or_else(|_| {
        Metadata::new_from_vec(
            &std::fs::read(path)?,
            little_exif::filetype::FileExtension::TIFF,
        )
    });
    let Ok(metadata) = metadata else {
        return Ok(false);
    };
    let (mut make, mut model, mut lens_make, mut lens_model) =
        (String::new(), String::new(), String::new(), String::new());
    let (mut focal, mut aperture) = (0f32, 0f32);
    for tag in &metadata {
        match tag {
            ExifTag::Make(v) => make = v.clone(),
            ExifTag::Model(v) => model = v.clone(),
            ExifTag::LensMake(v) => lens_make = v.clone(),
            ExifTag::LensModel(v) => lens_model = v.clone(),
            ExifTag::FocalLength(v) => {
                if let Some(v) = v.first() {
                    focal = v.nominator as f32 / v.denominator as f32;
                }
            }
            ExifTag::FNumber(v) => {
                if let Some(v) = v.first() {
                    aperture = v.nominator as f32 / v.denominator as f32;
                }
            }
            _ => {}
        }
    }
    if lens_model.is_empty() || focal <= 0. {
        return Ok(false);
    }
    let api = super::lens_api::Api::new()?;
    // SAFETY: Api owns the selected implementation for the lifetime of all guards.
    unsafe {
        let (
            new,
            destroy,
            load,
            load_dir,
            cameras,
            lenses,
            free,
            modifier_new,
            modifier_destroy,
            initialize,
            color,
            geometry,
        ) = (
            api.new,
            api.destroy,
            api.load,
            api.load_dir,
            api.cameras,
            api.lenses,
            api.free,
            api.modifier_new,
            api.modifier_destroy,
            api.initialize,
            api.color,
            api.geometry,
        );
        let db = new();
        if db.is_null() {
            bail!("Lensfun database allocation failed");
        }
        struct Guard(Ptr, unsafe extern "C" fn(Ptr));
        impl Drop for Guard {
            fn drop(&mut self) {
                unsafe { (self.1)(self.0) }
            }
        }
        let _db = Guard(db, destroy);
        let selected = super::lens_database::discover(database)?;
        let load_path = |path: &Path| -> Result<bool> {
            let filename = CString::new(path.to_string_lossy().as_bytes())?;
            Ok(load_dir(db, filename.as_ptr()) != 0)
        };
        let loaded = if let Some(path) = selected {
            load_path(&path)?
        } else if load(db) == 0 {
            true
        } else {
            load_path(&super::lens_database::ensure_cached()?)?
        };
        if !loaded {
            bail!(
                "Lensfun could not load the database. Choose a compatible version_1 database or use the automatic database."
            );
        }
        let (make, model, lens_make, lens_model) = (
            CString::new(make)?,
            CString::new(model)?,
            CString::new(lens_make)?,
            CString::new(lens_model)?,
        );
        let cams = cameras(db, make.as_ptr(), model.as_ptr(), 1);
        if cams.is_null() {
            return Ok(false);
        }
        let _cams = Guard(cams.cast(), free);
        let camera = *cams;
        if camera.is_null() {
            return Ok(false);
        }
        let found = lenses(
            db,
            camera,
            if lens_make.as_bytes().is_empty() {
                std::ptr::null()
            } else {
                lens_make.as_ptr()
            },
            lens_model.as_ptr(),
            1,
        );
        if found.is_null() {
            return Ok(false);
        }
        let _found = Guard(found.cast(), free);
        let lens = *found;
        if lens.is_null() {
            return Ok(false);
        }
        let modifier = modifier_new(
            lens,
            (*camera).crop,
            image.width as i32,
            image.height as i32,
        );
        if modifier.is_null() {
            bail!("Lensfun modifier allocation failed");
        }
        let _modifier = Guard(modifier, modifier_destroy);
        let enabled = initialize(modifier, lens, 3, focal, aperture, 1000., 1., 1, -1, 0);
        if enabled == 0 {
            return Ok(false);
        }
        tracing::info!(camera=%model.to_string_lossy(),lens=%lens_model.to_string_lossy(),flags=enabled,"Applying Lensfun correction");
        let mut pixels: Vec<f32> = image.data.iter().map(|v| *v as f32).collect();
        color(
            modifier,
            pixels.as_mut_ptr().cast(),
            0.,
            0.,
            image.width as i32,
            image.height as i32,
            super::lens_api::RGB_COMPONENTS,
            (image.width * 3 * 4) as i32,
        );
        let mut coords = vec![0f32; image.pixel_count() * 6];
        if geometry(
            modifier,
            0.,
            0.,
            image.width as i32,
            image.height as i32,
            coords.as_mut_ptr(),
        ) != 0
        {
            resample_geometry(image, &pixels, &coords);
        } else {
            for (a, b) in image.data.iter_mut().zip(pixels) {
                *a = from_f64(b as f64);
            }
        }
        Ok(true)
    }
}

fn resample_geometry(image: &mut ImageBuf, pixels: &[f32], coords: &[f32]) {
    let (width, height) = (image.width, image.height);
    image.par_pixels_mut().enumerate().for_each(|(i, p)| {
        for c in 0..3 {
            let x = coords[i * 6 + c * 2].clamp(0., (width - 1) as f32);
            let y = coords[i * 6 + c * 2 + 1].clamp(0., (height - 1) as f32);
            let (x0, y0) = (x.floor() as u32, y.floor() as u32);
            let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
            let (fx, fy) = (x - x0 as f32, y - y0 as f32);
            let at = |x, y| pixels[((y * width + x) * 3) as usize + c];
            p[c] = from_f64(
                (at(x0, y0) * (1. - fx) * (1. - fy)
                    + at(x1, y0) * fx * (1. - fy)
                    + at(x0, y1) * (1. - fx) * fy
                    + at(x1, y1) * fx * fy) as f64,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_geometry_preserves_serial_samples_and_edge_clamping() {
        for (width, height) in [(1, 1), (1, 7), (9, 1), (37, 29)] {
            let count = width as usize * height as usize * 3;
            let pixels: Vec<f32> = (0..count)
                .map(|i| (i * 997 % 4093) as f32 / 512. - 1.)
                .collect();
            let coords: Vec<f32> = (0..count)
                .flat_map(|i| {
                    [
                        (i * 7 % (width + 4) as usize) as f32 - 2.25,
                        (i * 11 % (height + 4) as usize) as f32 - 1.5,
                    ]
                })
                .collect();
            let mut expected = ImageBuf::new(width, height);
            for (i, p) in expected.pixels_mut().enumerate() {
                for c in 0..3 {
                    let x = coords[i * 6 + c * 2].clamp(0., (width - 1) as f32);
                    let y = coords[i * 6 + c * 2 + 1].clamp(0., (height - 1) as f32);
                    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
                    let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
                    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
                    let at = |x, y| pixels[((y * width + x) * 3) as usize + c];
                    p[c] = from_f64(
                        (at(x0, y0) * (1. - fx) * (1. - fy)
                            + at(x1, y0) * fx * (1. - fy)
                            + at(x0, y1) * (1. - fx) * fy
                            + at(x1, y1) * fx * fy) as f64,
                    );
                }
            }
            for threads in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let mut actual = ImageBuf::new(width, height);
                pool.install(|| resample_geometry(&mut actual, &pixels, &coords));
                assert!(
                    actual
                        .data
                        .iter()
                        .zip(&expected.data)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{width}x{height}, {threads} threads changed a lens-corrected sample"
                );
            }
        }
    }

    #[test]
    fn installed_lensfun_corrects_a_known_lens() {
        if std::env::var_os("SPEKTRAFILM_LENSFUN_DATABASE").is_none()
            && std::env::var_os("SPEKTRAFILM_LENSFUN_LIBRARY").is_none()
        {
            return;
        }
        let Some(db) = std::env::var_os("SPEKTRAFILM_LENSFUN_DATABASE")
            .map(std::path::PathBuf::from)
            .or_else(crate::lens_database::cached)
        else {
            return;
        };
        use little_exif::{exif_tag::ExifTag, metadata::Metadata, rational::uR64};
        let path =
            std::env::temp_dir().join(format!("spektrafilm-lens-{}.jpg", std::process::id()));
        image::save_buffer(
            &path,
            &vec![128u8; 32 * 32 * 3],
            32,
            32,
            image::ColorType::Rgb8,
        )
        .unwrap();
        let mut tags = Metadata::new();
        tags.set_tag(ExifTag::Make("Canon".into()));
        tags.set_tag(ExifTag::Model("Canon EOS 5D Mark II".into()));
        tags.set_tag(ExifTag::LensMake("Canon".into()));
        tags.set_tag(ExifTag::LensModel("Canon EF 50mm f/1.8 STM".into()));
        tags.set_tag(ExifTag::FocalLength(vec![uR64 {
            nominator: 50,
            denominator: 1,
        }]));
        tags.set_tag(ExifTag::FNumber(vec![uR64 {
            nominator: 28,
            denominator: 10,
        }]));
        tags.write_to_file(&path).unwrap();
        let mut image = ImageBuf::from_data(32, 32, vec![from_f64(0.18); 32 * 32 * 3]);
        assert!(correct(&mut image, &path, Some(Path::new(&db))).unwrap());
        assert!(image.data.iter().all(|v| v.is_finite()));
        assert!(image.data.iter().any(|v| (*v as f64 - 0.18).abs() > 1e-5));
        let mut automatic = ImageBuf::from_data(32, 32, vec![from_f64(0.18); 32 * 32 * 3]);
        assert!(correct(&mut automatic, &path, None).unwrap());
        assert_eq!(
            automatic.data, image.data,
            "automatic discovery must use the same corrections"
        );
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(all(test, feature = "bundled-lensfun"))]
mod builtin_tests {
    use super::*;
    use little_exif::{exif_tag::ExifTag, metadata::Metadata, rational::uR64};
    const XML: &str = r#"<lensdatabase version="1">
    <mount><name>Canon EF</name></mount>
    <camera><maker>Canon</maker><model>Canon EOS 5D Mark II</model><mount>Canon EF</mount><cropfactor>1</cropfactor></camera>
    <lens><maker>Canon</maker><model>Fixture Lens</model><mount>Canon EF</mount><focal value="35"/><aperture min="2.8"/><cropfactor>1</cropfactor>
    <calibration><distortion model="poly3" focal="35" k1="-0.1"/><tca model="linear" focal="35" kr="1.0005" kb="0.9995"/><vignetting model="pa" focal="35" aperture="2.8" distance="1000" k1="-0.2" k2="-0.03" k3="-0.01"/></calibration>
    </lens></lensdatabase>"#;
    fn tags() -> Metadata {
        let mut tags = Metadata::new();
        tags.set_tag(ExifTag::Make("Canon".into()));
        tags.set_tag(ExifTag::Model("Canon EOS 5D Mark II".into()));
        tags.set_tag(ExifTag::LensMake("Canon".into()));
        tags.set_tag(ExifTag::LensModel("Fixture Lens".into()));
        tags.set_tag(ExifTag::FocalLength(vec![uR64 {
            nominator: 35,
            denominator: 1,
        }]));
        tags.set_tag(ExifTag::FNumber(vec![uR64 {
            nominator: 28,
            denominator: 10,
        }]));
        tags
    }
    #[test]
    fn builtin_library_applies_real_corrections_without_an_external_library() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("fixture.xml"), XML).unwrap();
        let image_path = root.path().join("fixture.jpg");
        image::save_buffer(
            &image_path,
            &vec![128u8; 37 * 29 * 3],
            37,
            29,
            image::ColorType::Rgb8,
        )
        .unwrap();
        tags().write_to_file(&image_path).unwrap();
        let mut pixels = ImageBuf::from_data(37, 29, vec![from_f64(0.18); 37 * 29 * 3]);
        assert!(correct(&mut pixels, &image_path, Some(root.path())).unwrap());
        assert!(pixels.data.iter().all(|v| v.is_finite()));
        assert!(pixels.data.iter().any(|v| (*v as f64 - 0.18).abs() > 1e-4));
        for pixel in pixels.data.chunks_exact(3) {
            assert!((pixel[0] as f64 - pixel[1] as f64).abs() < 1e-3);
            assert!((pixel[1] as f64 - pixel[2] as f64).abs() < 1e-3);
        }
    }
    #[test]
    fn lens_setup_failure_is_not_retried_or_mislabeled_as_a_raw_decoder_failure() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("fixture.dng");
        std::fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/synthetic.dng"),
            &input,
        )
        .unwrap();
        let mut buffer = std::fs::read(&input).unwrap();
        let mut metadata =
            Metadata::new_from_vec(&buffer, little_exif::filetype::FileExtension::TIFF).unwrap();
        for tag in &tags() {
            metadata.set_tag(tag.clone());
        }
        metadata
            .write_to_vec(&mut buffer, little_exif::filetype::FileExtension::TIFF)
            .unwrap();
        std::fs::write(&input, buffer).unwrap();
        // Prove the fixture itself still decodes after attaching lens EXIF.
        super::super::load(
            &input,
            &crate::RawOptions {
                decoder: "rawler".into(),
                ..Default::default()
            },
        )
        .unwrap();
        let error = super::super::load(
            &input,
            &crate::RawOptions {
                lens_correction: true,
                lens_database: Some(root.path().join("missing-database")),
                ..Default::default()
            },
        )
        .err()
        .expect("invalid database must fail");
        let message = format!("{error:#}");
        assert!(error.is::<CorrectionFailure>(), "{message}");
        assert!(message.starts_with("Lens correction failed:"), "{message}");
        assert!(!message.contains("RAW decoding failed"), "{message}");
        assert!(!message.contains("install a current LibRaw"), "{message}");
    }
    #[test]
    fn fallback_preserves_lens_errors_from_either_decoder() {
        let error = crate::raw_fallback_error(
            anyhow::anyhow!("database unavailable").context(CorrectionFailure),
            anyhow::anyhow!("LibRaw is not installed"),
        );
        assert!(error.is::<CorrectionFailure>());
        assert_eq!(
            format!("{error:#}"),
            "Lens correction failed: database unavailable"
        );
        let error = crate::raw_fallback_error(
            anyhow::anyhow!("unknown camera"),
            anyhow::anyhow!("database unavailable").context(CorrectionFailure),
        );
        assert!(error.is::<CorrectionFailure>());
        assert_eq!(
            format!("{error:#}"),
            "Lens correction failed: database unavailable"
        );
        let decode_error = crate::raw_fallback_error(
            anyhow::anyhow!("unknown camera"),
            anyhow::anyhow!("LibRaw is not installed"),
        );
        assert!(decode_error.to_string().starts_with("RAW decoding failed."));
    }
}
