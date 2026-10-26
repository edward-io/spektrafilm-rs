//! One immutable photo export, shared by the editor and the selection queue.
use super::*;
use crate::edits::PhotoEdit;

#[derive(Clone)]
pub(super) struct Request {
    pub path: PathBuf,
    pub edit: PhotoEdit,
    pub detect_input_color: bool,
}

impl Request {
    pub fn prepare(
        &self,
        source: SourceImage,
    ) -> Result<(Arc<ImageBuf>, RuntimeParams, spektrafilm_io::Metadata)> {
        let mut params = self.edit.params.clone();
        params.settings.preview_mode = false;
        if self.detect_input_color {
            params.io.input_color_space = source.color_space;
            params.io.input_cctf_decoding = source.encoded;
        }
        let crop = self.edit.crop.unwrap_or_else(|| {
            crop_ui::lightroom_crop(&self.edit.raw, &source.pixels, self.edit.rotation)
        });
        let mut raw = self.edit.raw.clone();
        raw.lightroom_geometry = raw.lightroom_geometry.map(|g| g.orientation_only());
        let metadata = source
            .metadata
            .with_edit_settings(&spektrafilm_io::EditSettings {
                film: &self.edit.film,
                paper: &self.edit.paper,
                params: &params,
                raw: &raw,
                rotation: self.edit.rotation,
                crop,
            })?;
        params.camera.film_format_mm *=
            crop.long_side_fraction(source.pixels.width as f64, source.pixels.height as f64) as f32;
        let image = crop_ui::cropped(&source.pixels, &crop).map_err(anyhow::Error::msg)?;
        Ok((image, params, metadata))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        output: &Path,
        options: &spektrafilm_io::SaveOptions,
        data: &Path,
        backend: &dyn ComputeBackend,
        cli: Option<&Path>,
        cancel: &AtomicBool,
        task: &Task,
        cached: Option<SourceImage>,
    ) -> Result<()> {
        task.check()?;
        if let Some(cli) = cli {
            let _cpu = task.cpu()?;
            let mut params = self.edit.params.clone();
            params.settings.preview_mode = false;
            return run_f64_export(
                cli,
                &self.path,
                output,
                &self.edit.film,
                &self.edit.paper,
                &params,
                data,
                cancel,
                &self.edit.raw,
                options,
                self.edit.rotation,
                self.edit.crop,
                self.detect_input_color,
            );
        }
        let cpu = task.cpu()?;
        let source = match cached {
            Some(source) => source,
            None => SourceImage::load(&self.path, &self.edit.raw, self.edit.rotation, task)
                .map_err(anyhow::Error::msg)?,
        };
        let (image, params, metadata) = self.prepare(source)?;
        drop(cpu);
        task.check()?;
        run_gpu_export(
            &image,
            output,
            &self.edit.film,
            &self.edit.paper,
            &params,
            data,
            backend,
            cancel,
            options,
            &metadata,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a built f64 helper; set SPEKTRAFILM_TEST_F64_CLI"]
    fn f64_helper_preserves_catalog_framing_and_per_photo_raw_settings() {
        let cli = PathBuf::from(
            std::env::var_os("SPEKTRAFILM_TEST_F64_CLI").expect("set SPEKTRAFILM_TEST_F64_CLI"),
        );
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let dir = tempfile::tempdir().unwrap();
        let mut edit = PhotoEdit::default();
        edit.raw.decoder = "rawler".into();
        edit.raw.white_balance_shift = [0.2, -0.1];
        let geometry = spektrafilm_io::lightroom::Geometry {
            orientation: 6,
            crop: [0.2, 0.1, 0.8, 0.7],
            angle: 3.,
        };
        edit.raw.lightroom_geometry = Some(geometry.clone());
        edit.rotation = 1;
        edit.params.io.input_color_space = "ProPhoto RGB".into();
        edit.params.debug.deactivate_spatial_effects = true;
        edit.params.debug.deactivate_stochastic_effects = true;
        let mut request = Request {
            path: root.join("crates/spektrafilm-io/tests/fixtures/synthetic.dng"),
            edit,
            detect_input_color: true,
        };
        let scheduler = Scheduler::default();
        let task = scheduler.task(Priority::Export);
        let cpu = spektrafilm_gpu::cpu_backend::CpuBackend;
        for explicit in [None, Some(Crop::FULL)] {
            request.edit.crop = explicit;
            let output = dir.path().join("output.tif");
            task.run(|| {
                request.run(
                    &output,
                    &Default::default(),
                    &root.join("data"),
                    &cpu,
                    Some(&cli),
                    &AtomicBool::new(false),
                    &task,
                    None,
                )
            })
            .unwrap();
            let result = spektrafilm_io::load(&output, &Default::default()).unwrap();
            let exif = result.metadata.native.as_ref().unwrap();
            let bytes = exif
                .get_tag_by_hex(0x9286, None)
                .next()
                .unwrap()
                .value_as_u8_vec(&exif.get_endian());
            let recipe: serde_json::Value = serde_json::from_slice(&bytes[8..]).unwrap();
            let expected =
                explicit.unwrap_or_else(|| Crop::from_catalog(Some(&geometry), 32, 32, 1));
            assert_eq!(
                recipe["settings"]["crop"],
                serde_json::to_value(expected).unwrap()
            );
            assert_eq!(
                recipe["settings"]["params"]["io"]["input_color_space"],
                "sRGB"
            );
            assert_eq!(
                recipe["settings"]["raw"]["white_balance_shift"],
                serde_json::json!([0.2, -0.1])
            );
        }
        let error = task
            .run(|| {
                request.run(
                    &dir.path().join("missing/output.tif"),
                    &Default::default(),
                    &root.join("data"),
                    &cpu,
                    Some(&cli),
                    &AtomicBool::new(false),
                    &task,
                    None,
                )
            })
            .unwrap_err();
        assert!(error.is::<spektrafilm_io::OutputIoError>(), "{error:#}");
    }

    #[test]
    fn preparation_resolves_unopened_photos_without_reusing_the_editors_input_color() {
        let mut edit = PhotoEdit::default();
        edit.params.camera.film_format_mm = 40.;
        edit.params.io.input_color_space = "ProPhoto RGB".into();
        edit.params.io.input_cctf_decoding = false;
        edit.rotation = 1;
        edit.raw.lightroom_geometry = Some(spektrafilm_io::lightroom::Geometry {
            orientation: 1,
            crop: [0.25, 0., 1., 0.5],
            angle: 0.,
        });
        let source = SourceImage {
            pixels: Arc::new(ImageBuf::new(24, 32)),
            metadata: Default::default(),
            color_space: "sRGB".into(),
            encoded: true,
            stamp: None,
        };
        let mut request = Request {
            path: "photo.png".into(),
            edit,
            detect_input_color: true,
        };
        let (image, params, metadata) = request.prepare(source.clone()).unwrap();
        assert_eq!((image.width, image.height), (12, 24));
        assert_eq!(params.camera.film_format_mm, 30.);
        assert_eq!(params.io.input_color_space, "sRGB");
        assert!(params.io.input_cctf_decoding);
        let settings = metadata.edit_settings.unwrap();
        assert_eq!(settings["params"]["camera"]["film_format_mm"], 40.);
        assert_eq!(
            settings["raw"]["lightroom_geometry"]["crop"],
            serde_json::json!([0., 0., 1., 1.])
        );
        request.edit.crop = Some(Crop::FULL);
        request.detect_input_color = false;
        let (image, params, _) = request.prepare(source).unwrap();
        assert_eq!((image.width, image.height), (24, 32));
        assert_eq!(params.camera.film_format_mm, 40.);
        assert_eq!(params.io.input_color_space, "ProPhoto RGB");
        assert!(!params.io.input_cctf_decoding);
    }

    #[test]
    fn queued_original_matches_single_export_with_crop_tone_and_color_conversion() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.png");
        image::RgbImage::from_fn(32, 24, |x, y| {
            image::Rgb([30 + x as u8 * 5, 20 + y as u8 * 7, 120])
        })
        .save(&input)
        .unwrap();
        let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        let mut edit = PhotoEdit::default();
        edit.params.debug.deactivate_spatial_effects = true;
        edit.params.debug.deactivate_stochastic_effects = true;
        edit.params.camera.auto_exposure = false;
        edit.params.camera.tone.shadows = 25.;
        edit.params.camera.exposure_compensation_ev = -0.5;
        edit.raw.lightroom_geometry = Some(spektrafilm_io::lightroom::Geometry {
            orientation: 6,
            crop: [0.2, 0.1, 0.8, 0.7],
            angle: 3.,
        });
        edit.rotation = 1;
        let request = Request {
            path: input,
            edit,
            detect_input_color: true,
        };
        let options = spektrafilm_io::SaveOptions {
            bit_depth: 32,
            color_space: "Display P3".into(),
            ..Default::default()
        };
        let scheduler = Scheduler::default();
        let mut backends: Vec<Box<dyn ComputeBackend>> =
            vec![Box::new(spektrafilm_gpu::cpu_backend::CpuBackend)];
        let gpu = spektrafilm_gpu::select_backend();
        if gpu.is_gpu() {
            backends.push(gpu);
        }
        for backend in backends {
            let task = scheduler.task(Priority::Export);
            task.run(|| {
                let source = SourceImage::load(
                    &request.path,
                    &request.edit.raw,
                    request.edit.rotation,
                    &task,
                )
                .unwrap();
                let reference = dir.path().join("single.tif");
                let actual = dir.path().join("queued.tif");
                request
                    .run(
                        &reference,
                        &options,
                        &data,
                        backend.as_ref(),
                        None,
                        &AtomicBool::new(false),
                        &task,
                        Some(source),
                    )
                    .unwrap();
                request
                    .run(
                        &actual,
                        &options,
                        &data,
                        backend.as_ref(),
                        None,
                        &AtomicBool::new(false),
                        &task,
                        None,
                    )
                    .unwrap();
                let reference = spektrafilm_io::load(&reference, &Default::default()).unwrap();
                let actual = spektrafilm_io::load(&actual, &Default::default()).unwrap();
                assert_eq!(
                    (actual.pixels.width, actual.pixels.height),
                    (reference.pixels.width, reference.pixels.height)
                );
                assert_eq!(
                    actual.pixels.data,
                    reference.pixels.data,
                    "{}",
                    backend.name()
                );
                let exif = actual.metadata.native.as_ref().unwrap();
                let comment = exif
                    .get_tag_by_hex(0x9286, None)
                    .next()
                    .unwrap()
                    .value_as_u8_vec(&exif.get_endian());
                let recipe: serde_json::Value = serde_json::from_slice(&comment[8..]).unwrap();
                assert_eq!(recipe["settings"]["rotation"], 1);
                assert_eq!(
                    recipe["settings"]["params"]["camera"]["tone"]["shadows"],
                    25.
                );
                assert_eq!(
                    recipe["settings"]["params"]["io"]["input_color_space"],
                    "sRGB"
                );
                assert_eq!(recipe["output"]["color_space"], "Display P3");
            });
        }
    }
}
