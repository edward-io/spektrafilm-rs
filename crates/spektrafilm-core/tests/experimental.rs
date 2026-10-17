use spektrafilm_core::{params::RuntimeParams, pipeline::Pipeline, profile::load_profile_by_name};
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
#[test]
fn experimental_runtime_matches_python_on_cpu_and_gpu() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/experimental.json")).unwrap();
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let cpu = spektrafilm_gpu::cpu_backend::CpuBackend;
    let gpu = spektrafilm_gpu::wgpu_backend::WgpuBackend::new();
    let mut failures = vec![];
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let params: RuntimeParams = serde_json::from_value(case["params"].clone()).unwrap();
        let pipeline = Pipeline::new_with_spectral(
            load_profile_by_name(&directory, case["film"].as_str().unwrap()).unwrap(),
            load_profile_by_name(&directory, case["print"].as_str().unwrap()).unwrap(),
            params,
            &directory,
        )
        .unwrap();
        let input = ImageBuf::from_data(
            case["width"].as_u64().unwrap() as u32,
            case["height"].as_u64().unwrap() as u32,
            case["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| from_f64(v.as_f64().unwrap()))
                .collect(),
        );
        for (backend, label, tolerance) in [
            (
                &cpu as &dyn spektrafilm_gpu::ComputeBackend,
                "CPU",
                if cfg!(feature = "precision-f64") {
                    2e-5
                } else {
                    0.002
                },
            ),
            (
                gpu.as_ref()
                    .map(|v| v as &dyn spektrafilm_gpu::ComputeBackend)
                    .unwrap_or(&cpu),
                "GPU",
                0.002,
            ),
        ] {
            let out = pipeline.process(input.clone(), backend);
            assert_eq!(
                (out.width as u64, out.height as u64),
                (
                    case["output_width"].as_u64().unwrap(),
                    case["output_height"].as_u64().unwrap()
                ),
                "{name}"
            );
            let max = out
                .data
                .iter()
                .zip(case["expected"].as_array().unwrap())
                .map(|(a, b)| (*a as f64 - b.as_f64().unwrap()).abs())
                .fold(0., f64::max);
            eprintln!("experimental {name} {label}: max {max:.8}");
            if max > tolerance || out.data.iter().any(|v| !v.is_finite()) {
                failures.push(format!("{name} {label}: {max}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn nonlinear_coupler_curve_inversion_matches_numpy() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/experimental.json")).unwrap();
    for detail in fixture["dir_details"].as_array().unwrap() {
        let stock = detail["film"].as_str().unwrap();
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["name"] == format!("dir_{stock}"))
            .unwrap();
        let p: RuntimeParams = serde_json::from_value(case["params"].clone()).unwrap();
        let dir = &p.film_render.dir_couplers;
        let curves: Vec<[f64; 3]> = serde_json::from_value(detail["curves"].clone()).unwrap();
        let axis: Vec<f64> = serde_json::from_value(detail["axis"].clone()).unwrap();
        let matrix = spektrafilm_model::couplers::compute_dir_couplers_matrix(
            dir.gamma_samelayer_rgb,
            dir.gamma_interlayer_r_to_gb,
            dir.gamma_interlayer_g_to_rb,
            dir.gamma_interlayer_b_to_rg,
            dir.inhibition_samelayer,
            dir.inhibition_interlayer,
        );
        let response = spektrafilm_model::couplers::CouplerResponse::new(
            &curves,
            matrix,
            dir.amount,
            dir.langmuir_donor_k_rgb,
            dir.langmuir_receiver_k_rgb,
            stock.contains("provia"),
        );
        let actual = response.curves_before(&curves, &axis);
        let expected: Vec<[f64; 3]> = serde_json::from_value(detail["before"].clone()).unwrap();
        let max = actual
            .iter()
            .flatten()
            .zip(expected.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        assert!(max < 1e-10, "{stock} curve inversion: {max}");
    }
}

#[test]
fn runtime_defaults_match_experimental_schema() {
    fn compare(actual: &serde_json::Value, expected: &serde_json::Value, path: &str) {
        if let Some(object) = expected.as_object() {
            for (key, value) in object {
                compare(&actual[key], value, &format!("{path}/{key}"));
            }
        } else if let Some(array) = expected.as_array() {
            for (i, value) in array.iter().enumerate() {
                compare(&actual[i], value, &format!("{path}/{i}"));
            }
        } else if let Some(value) = expected.as_f64() {
            assert!(
                (actual.as_f64().unwrap() - value).abs() < 1e-6 * (1. + value.abs()),
                "{path}: {actual} vs {expected}"
            );
        } else {
            assert_eq!(actual, expected, "{path}");
        }
    }
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/experimental.json")).unwrap();
    let p = RuntimeParams::default();
    let mut actual = serde_json::to_value(&p).unwrap();
    actual["workflow"]["route"] = p.route().into();
    compare(&actual, &fixture["extra"]["defaults"], "");
}

#[test]
fn septic_chemistry_preserves_total_and_physical_layers() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/experimental.json")).unwrap();
    for case in fixture["extra"]["sept"].as_array().unwrap() {
        let axis: Vec<f64> = serde_json::from_value(case["axis"].clone()).unwrap();
        let model: spektrafilm_core::profile::DensityCurvesModel =
            serde_json::from_value(case["model"].clone()).unwrap();
        let p = serde_json::from_value(case["params"].clone()).unwrap();
        let (total, layers) = spektrafilm_core::print_morph::morph_density_curves_with_layers(
            &axis,
            &model,
            &p,
            case["positive"].as_bool().unwrap(),
        )
        .unwrap();
        let expected: Vec<[f64; 3]> = serde_json::from_value(case["total"].clone()).unwrap();
        let expected_layers: Vec<Vec<Vec<f64>>> =
            serde_json::from_value(case["layers"].clone()).unwrap();
        for i in 0..axis.len() {
            for c in 0..3 {
                assert!((total[i][c] - expected[i][c]).abs() < 1e-9);
                for sl in 0..3 {
                    assert!((layers[i][sl][c] - expected_layers[i][sl][c]).abs() < 1e-9);
                }
                assert!((layers[i].iter().map(|r| r[c]).sum::<f64>() - total[i][c]).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn rms_particle_sizing_matches_python_realized_layer_peak() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../spektrafilm-gpu/tests/fixtures/grain_replay.json"
    ))
    .unwrap();
    let case = &fixture["cases"][0];
    let table: Vec<[f64; 4]> = serde_json::from_value(case["curves"].clone()).unwrap();
    let curves: Vec<[f64; 3]> = table
        .chunks_exact(3)
        .map(|r| [r[0][0], r[1][0], r[2][0]])
        .collect();
    let layers: Vec<Vec<Vec<f64>>> = table
        .chunks_exact(3)
        .map(|r| {
            (0..3)
                .map(|sl| (0..3).map(|ch| r[ch][sl + 1]).collect())
                .collect()
        })
        .collect();
    let p = spektrafilm_model::grain::LayeredGrainParams {
        rms_granularity: [6., 8., 12.],
        usm_sigma: 0.7,
        usm_amount: 1.5,
        layer_scale: [1., 0.4, 0.25],
        density_min: [0.03, 0.04, 0.02],
        uniformity: [0.97, 0.99, 0.96],
        blur: 0.89,
        dye_cloud_blur: 16.,
        micro_structure: [0., 0.],
        monochrome: false,
        fast_stats: true,
    };
    let actual = spektrafilm_model::grain::layered_gpu_params(5., &curves, &layers, false, &p);
    for sl in 0..3 {
        for ch in 0..3 {
            let expected = case["layers"][sl]["particles"][ch].as_f64().unwrap();
            assert!((actual.layers[sl].particles[ch] as f64 / expected - 1.).abs() < 2e-6);
        }
    }
}

#[test]
fn explicit_workflows_override_legacy_scan_flags() {
    let mut p = RuntimeParams::default();
    p.io.scan_film = true;
    p.workflow.route = "input > film > print > scan".into();
    p.taps.collect = Some("cmy_print".into());
    p.validate().unwrap();
    p.digest();
    assert!(!p.io.scan_film);
    p.workflow.route = "input > convert-film > scan".into();
    p.taps = Default::default();
    p.digest();
    assert!(p.io.scan_film && p.converts_film());
}
