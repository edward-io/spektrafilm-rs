//! Each selectable transport must keep its meaning on the resident GPU path.
#![cfg(feature = "precision-f64")]
#[path = "support/gpu_audit_cases.rs"]
mod audit_cases;
use spektrafilm_core::{params::RuntimeParams, pipeline::Pipeline, profile::load_profile_by_name};
use spektrafilm_math::{image::ImageBuf, transfer};
fn run(
    name: &str,
    input: &ImageBuf,
    params: RuntimeParams,
    gpu: &dyn spektrafilm_gpu::ComputeBackend,
) {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let pipeline = Pipeline::new_with_spectral(
        load_profile_by_name(&data, "kodak_portra_400").unwrap(),
        load_profile_by_name(&data, "kodak_portra_endura").unwrap(),
        params,
        &data,
    )
    .unwrap();
    let reference = pipeline.process(input.clone(), &spektrafilm_gpu::cpu_backend::CpuBackend);
    let actual = pipeline
        .process_resident_borrowed(input, gpu)
        .expect("color controls must stay resident");
    let mut max = 0f64;
    for (&a, &b) in reference.data.iter().zip(&actual.data) {
        // Some declared log transports have a limited mathematical domain.
        // Match the CPU's undefined output explicitly rather than clipping it.
        if !a.is_finite() {
            assert!(
                (a.is_nan() && b.is_nan()) || a == b,
                "{name}: nonfinite mismatch {a} / {b}"
            );
            continue;
        }
        assert!(
            b.is_finite(),
            "{name}: nonfinite GPU value for finite reference {a}"
        );
        max = max.max((a - b).abs());
    }
    eprintln!("resident color {name}: max {max}");
    assert!(max <= 0.001, "{name}: exceeded 0.1% full scale: {max}");
}
#[test]
fn all_named_input_and_output_transports_stay_resident() {
    let Some(gpu) = spektrafilm_gpu::wgpu_backend::WgpuBackend::new() else {
        return;
    };
    let mut params = audit_cases::cases().remove(0).1;
    params.debug.deactivate_spatial_effects = true;
    params.io.output_gamut_compress.algorithm = "off".into();
    let image = spektrafilm_core::resize::preview(&audit_cases::image(), 64);
    for name in transfer::names() {
        let space = transfer::get(name).unwrap();
        if !space.transfer.supported() {
            continue;
        }
        let mut p = params.clone();
        p.io.output_color_space = name.into();
        run(&format!("output {name}"), &image, p, &gpu);
        // Realistic encoded input: generate from a bounded positive linear ramp,
        // keeping every log curve within its declared domain.
        let encoded = ImageBuf::from_data(
            image.width,
            image.height,
            image
                .data
                .iter()
                .map(|v| space.transfer.encode(v.clamp(0., 1.)))
                .collect(),
        );
        let mut p = params.clone();
        p.io.input_color_space = name.into();
        p.io.input_cctf_decoding = true;
        run(&format!("input {name}"), &encoded, p, &gpu);
    }
}
#[test]
fn primaries_gamut_white_and_encoded_metering_stay_resident() {
    let Some(gpu) = spektrafilm_gpu::wgpu_backend::WgpuBackend::new() else {
        return;
    };
    let base = audit_cases::cases().remove(0).1;
    let image = audit_cases::image();
    for cs in [
        "sRGB",
        "Display P3",
        "ProPhoto RGB",
        "ACEScg",
        "ITU-R BT.2020",
    ] {
        for gamut in ["off", "aces_rgc", "oklch", "oklrab", "cam16ucs", "jzazbz"] {
            let mut p = base.clone();
            p.io.output_color_space = cs.into();
            p.io.output_gamut_compress.algorithm = gamut.into();
            run(&format!("{cs} {gamut}"), &image, p, &gpu);
        }
    }
    for method in [
        "average",
        "median",
        "center_weighted",
        "partial",
        "matrix",
        "multi_zone",
        "highlight_weighted",
    ] {
        let mut p = base.clone();
        p.camera.auto_exposure = true;
        p.camera.auto_exposure_method = method.into();
        p.io.input_cctf_decoding = true;
        p.io.output_color_space = "Display P3".into();
        run(&format!("encoded metering {method}"), &image, p, &gpu);
    }
}
