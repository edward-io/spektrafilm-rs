//! Independent Python simulator fixtures for native calibration-strip assembly.
use spektrafilm_core::{
    calibration::{self, CalibrationOptions},
    params::RuntimeParams,
    profile::load_profile_by_name,
};
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
#[test]
fn calibration_ramps_match_python_in_both_stack_directions() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/calibration.json")).unwrap();
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let film = load_profile_by_name(&data, "kodak_portra_400").unwrap();
    let paper = load_profile_by_name(&data, "kodak_portra_endura").unwrap();
    let params: RuntimeParams = serde_json::from_value(fixture["params"].clone()).unwrap();
    let input = ImageBuf::from_data(
        12,
        8,
        fixture["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| from_f64(v.as_f64().unwrap()))
            .collect(),
    );
    for case in fixture["cases"].as_array().unwrap() {
        let options: CalibrationOptions = serde_json::from_value(case.clone()).unwrap();
        let strip = calibration::render(
            &input,
            &film,
            &paper,
            &params,
            &options,
            &data,
            &spektrafilm_gpu::cpu_backend::CpuBackend,
        )
        .unwrap();
        assert_eq!(strip.image.width, case["shape"][1].as_u64().unwrap() as u32);
        assert_eq!(
            strip.image.height,
            case["shape"][0].as_u64().unwrap() as u32
        );
        let mut max = 0f64;
        for (&actual, expected) in strip
            .image
            .data
            .iter()
            .zip(case["expected"].as_array().unwrap())
        {
            max = max.max((actual as f64 - expected.as_f64().unwrap()).abs());
        }
        assert!(max < 2e-6, "calibration strip differs from Python: {max}");
        assert_eq!(strip.labels.len(), 3);
    }
}
