//! Exercise real single-channel profiles through both grain dispatch paths.
use spektrafilm_core::{
    params::RuntimeParams, pipeline::Pipeline, profile::load_profile_by_name, stages::filming,
};
use spektrafilm_gpu::{ComputeBackend, cpu_backend::CpuBackend, wgpu_backend::WgpuBackend};
use spektrafilm_math::{image::ImageBuf, precision::from_f64};

fn pipeline(stock: &str, development_time: Option<f64>) -> Pipeline {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let mut params = RuntimeParams::default();
    params.camera.auto_exposure = false;
    params.camera.film_format_mm = 0.64;
    params.film_render.development_time = development_time;
    params.film_render.halation.active = false;
    params.film_render.dir_couplers.active = false;
    params.film_render.glare.active = false;
    params.print_render.glare.active = false;
    params.scanner.unsharp_mask = [0., 0.];
    Pipeline::new_with_spectral(
        load_profile_by_name(&data, stock).unwrap(),
        load_profile_by_name(&data, "kodak_2302").unwrap(),
        params,
        &data,
    )
    .unwrap()
}

fn check_develop(pipeline: &Pipeline, backend: &dyn ComputeBackend) {
    assert!(pipeline.params.film_render.grain.active);
    assert!(pipeline.params.film_render.grain.sublayers_active);
    assert!(pipeline.params.film_render.grain.monochrome);
    let exposure = pipeline.film.data.log_exposure[128];
    let input = ImageBuf::from_data(64, 48, vec![from_f64(exposure); 64 * 48 * 3]);
    let output = filming::develop(&input, &pipeline.film, &pipeline.params, backend);
    assert!(output.data.iter().all(|v| v.is_finite()));
    assert!(output.pixels().all(|p| p[0] == p[1] && p[1] == p[2]));
    let mut clean_params = pipeline.params.clone();
    clean_params.film_render.grain.active = false;
    let clean = filming::develop(&input, &pipeline.film, &clean_params, backend);
    assert!(
        output
            .data
            .iter()
            .zip(&clean.data)
            .any(|(a, b)| (*a - *b).abs() > from_f64(1e-6)),
        "physical grain was bypassed"
    );
}

#[test]
fn bw_physical_grain_develops_on_cpu_and_gpu() {
    let gpu = WgpuBackend::new();
    for (stock, time) in [
        ("kodak_doublex", None),
        ("kodak_doublex", Some(12.0)),
        ("kodak_trix", None),
    ] {
        let pipeline = pipeline(stock, time);
        check_develop(&pipeline, &CpuBackend);
        if let Some(gpu) = &gpu {
            check_develop(&pipeline, gpu);
        }
    }
}

#[test]
fn bw_physical_grain_renders_in_resident_gpu_and_cpu_pipelines() {
    let gpu = WgpuBackend::new();
    let input = ImageBuf::from_data(
        64,
        48,
        (0..64 * 48)
            .flat_map(|i| [from_f64(0.04 + 0.4 * (i % 64) as f64 / 63.); 3])
            .collect(),
    );
    for stock in ["kodak_doublex", "kodak_trix"] {
        let mut pipeline = pipeline(stock, None);
        for scan_film in [false, true] {
            pipeline.params.io.scan_film = scan_film;
            let cpu = pipeline.process(input.clone(), &CpuBackend);
            assert_eq!((cpu.width, cpu.height), (64, 48));
            assert!(cpu.data.iter().all(|v| v.is_finite()), "{stock}: CPU");
            if let Some(gpu) = &gpu {
                let output = pipeline
                    .process_resident_borrowed(&input, gpu)
                    .expect("physical B&W grain must stay on the GPU");
                assert_eq!((output.width, output.height), (64, 48));
                assert!(output.data.iter().all(|v| v.is_finite()), "{stock}: GPU");
                let repeat = pipeline.process_resident_borrowed(&input, gpu).unwrap();
                assert_eq!(output.data, repeat.data, "{stock}: nondeterministic grain");
            }
        }
    }
}
