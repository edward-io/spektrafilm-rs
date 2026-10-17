use spektrafilm_core::{params::RuntimeParams, pipeline::Pipeline, profile::load_profile_by_name};
use spektrafilm_math::{image::ImageBuf, precision::from_f64, transfer::Transfer};

#[test]
fn autoexposure_uses_linear_light_on_cpu_and_gpu() {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let gpu = spektrafilm_gpu::wgpu_backend::WgpuBackend::new();
    let cpu = spektrafilm_gpu::cpu_backend::CpuBackend;
    for brightness in [0.03, 0.18, 0.5, 1.8] {
        let mut params = RuntimeParams::default();
        params.io.input_color_space = "sRGB".into();
        params.io.output_cctf_encoding = false;
        params.debug.deactivate_spatial_effects = true;
        params.debug.deactivate_stochastic_effects = true;
        let linear = ImageBuf::from_data(8, 8, vec![from_f64(brightness); 8 * 8 * 3]);
        let encoded = ImageBuf::from_data(
            8,
            8,
            vec![from_f64(Transfer::Srgb.encode(brightness)); 8 * 8 * 3],
        );
        let build = |p| {
            Pipeline::new_with_spectral(
                load_profile_by_name(&data, "kodak_portra_400").unwrap(),
                load_profile_by_name(&data, "kodak_portra_endura").unwrap(),
                p,
                &data,
            )
            .unwrap()
        };
        let reference = build(params.clone()).process(linear.clone(), &cpu);
        params.io.input_cctf_decoding = true;
        let p = build(params);
        let actual = p.process(encoded.clone(), &cpu);
        for (a, b) in actual.data.iter().zip(&reference.data) {
            assert!(
                (*a as f64 - *b as f64).abs() < 1e-5,
                "{brightness}: {a} / {b}"
            );
        }
        if let Some(gpu) = &gpu {
            let actual = p
                .process_resident_borrowed(&encoded, gpu)
                .expect("encoded autoexposure stays resident");
            for (a, b) in actual.data.iter().zip(&reference.data) {
                assert!(
                    (*a as f64 - *b as f64).abs() < 0.001,
                    "GPU {brightness}: {a} / {b}"
                );
            }
        }
    }
}
