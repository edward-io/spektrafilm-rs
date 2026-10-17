//! The resident physical RMS grain kernel must produce the reference law and texture.
use spektrafilm_core::{params::RuntimeParams, pipeline::Pipeline, profile::load_profile_by_name};
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
fn stats(image: &ImageBuf, c: usize) -> (f64, f64) {
    let mean = image.pixels().map(|p| p[c] as f64).sum::<f64>() / image.pixel_count() as f64;
    let var = image
        .pixels()
        .map(|p| (p[c] as f64 - mean).powi(2))
        .sum::<f64>()
        / image.pixel_count() as f64;
    (mean, var)
}
#[test]
fn resident_rms_grain_is_repeatable_and_preserves_reference_statistics() {
    let Some(gpu) = spektrafilm_gpu::wgpu_backend::WgpuBackend::new() else {
        return;
    };
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let image = ImageBuf::from_data(256, 256, vec![from_f64(0.18); 256 * 256 * 3]);
    let mut params = RuntimeParams::default();
    params.camera.auto_exposure = false;
    params.camera.film_format_mm = 2.048;
    params.film_render.halation.active = false;
    params.film_render.dir_couplers.active = false;
    params.settings.apply_stock_specifics = false;
    params.print_render.glare.active = false;
    params.scanner.unsharp_mask = [0., 0.];
    for (rms, blur) in [(3., 0.), (6., 0.7), (12., 4.)] {
        params.film_render.grain.rms_granularity = [rms; 3];
        params.film_render.grain.blur = blur;
        let p = Pipeline::new_with_spectral(
            load_profile_by_name(&data, "kodak_portra_400").unwrap(),
            load_profile_by_name(&data, "kodak_portra_endura").unwrap(),
            params.clone(),
            &data,
        )
        .unwrap();
        let reference = p.process(image.clone(), &spektrafilm_gpu::cpu_backend::CpuBackend);
        let output = p.process_resident_borrowed(&image, &gpu).unwrap();
        let repeat = p.process_resident_borrowed(&image, &gpu).unwrap();
        assert_eq!(
            output.data, repeat.data,
            "physical GPU grain is not repeatable"
        );
        for c in 0..3 {
            let (a, va) = stats(&reference, c);
            let (b, vb) = stats(&output, c);
            eprintln!("RMS {rms}, blur {blur}, channel {c}: mean {a}/{b}, variance {va}/{vb}");
            assert!((a - b).abs() < 0.003, "physical RMS grain changed tone");
            assert!(
                vb > 1e-9,
                "physical RMS grain kernel did not write its texture"
            );
            if blur < 1. {
                assert!(
                    (vb / va - 1.).abs() < 0.06,
                    "physical RMS grain variance drift {}",
                    vb / va
                );
            }
        }
    }
}
