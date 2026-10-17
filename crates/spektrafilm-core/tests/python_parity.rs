use spektrafilm_core::{params::RuntimeParams, pipeline::Pipeline, profile::load_profile_by_name};
use spektrafilm_math::{image::ImageBuf, precision::from_f64};

#[test]
fn python_reference_cases() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/python_parity.json")).unwrap();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let params: RuntimeParams = serde_json::from_value(case["params"].clone()).unwrap();
        let p = Pipeline::new_with_spectral(
            load_profile_by_name(&dir, "kodak_portra_400").unwrap(),
            load_profile_by_name(&dir, "kodak_portra_endura").unwrap(),
            params,
            &dir,
        )
        .unwrap();
        let image = ImageBuf::from_data(
            case["width"].as_u64().unwrap() as u32,
            case["height"].as_u64().unwrap() as u32,
            case["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| from_f64(v.as_f64().unwrap()))
                .collect(),
        );
        let out = p.process(image, &spektrafilm_gpu::cpu_backend::CpuBackend);
        assert_eq!(
            out.width as u64,
            case["output_width"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(
            out.height as u64,
            case["output_height"].as_u64().unwrap(),
            "{name}"
        );
        let tol = case["tolerance"]
            .as_f64()
            .unwrap()
            .max(if cfg!(feature = "precision-f64") {
                0.
            } else {
                0.003
            });
        for (i, (got, want)) in out
            .data
            .iter()
            .zip(case["expected"].as_array().unwrap())
            .enumerate()
        {
            let want = want.as_f64().unwrap();
            assert!(
                (*got as f64 - want).abs() < tol,
                "{name} [{i}]: got {got}, expected {want} (tolerance {tol})"
            );
        }
    }
}

#[test]
fn calibrated_lut_bake_is_deterministic() {
    use spektrafilm_core::lut_export::*;
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let opts = BakeOptions {
        film: "kodak_portra_400",
        paper: "kodak_portra_endura",
        size: 5,
        input: "sRGB",
        output: "sRGB",
        gain: Some(1.),
        params: RuntimeParams::default(),
    };
    let a = bake(&opts, &dir).unwrap();
    let b = bake(&opts, &dir).unwrap();
    assert_eq!(a.data, b.data);
    assert!(a.data.iter().all(|v| v.is_finite() && *v >= 0. && *v <= 1.));
    assert!(a.data.iter().any(|v| *v > 0.5));
}

#[test]
fn transport_curves_match_python() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/transfers.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let cs = spektrafilm_core::lut_export::space(name).unwrap();
        for (i, input) in case["input"].as_array().unwrap().iter().enumerate() {
            let x = input.as_f64().unwrap();
            for (actual, key) in [
                (cs.curve.encode(x), "encode"),
                (cs.curve.decode(x), "decode"),
            ] {
                let expected = case[key][i].as_f64().unwrap();
                assert!(
                    (actual - expected).abs() < 1e-7 * (1. + expected.abs()),
                    "{name} {key}({x}): {actual} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn exported_luts_match_python_bundle_bakes() {
    use spektrafilm_core::lut_export::*;
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/luts.json")).unwrap();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    for case in cases.as_array().unwrap() {
        let options = BakeOptions {
            film: "kodak_portra_400",
            paper: "kodak_portra_endura",
            size: 5,
            input: case["input"].as_str().unwrap(),
            output: case["output"].as_str().unwrap(),
            gain: case["exposure_ev"].as_f64().map(|ev| {
                space(case["input"].as_str().unwrap()).unwrap().auto_gain * 2f64.powf(ev)
            }),
            params: if case["params"].is_object() {
                serde_json::from_value(case["params"].clone()).unwrap()
            } else {
                RuntimeParams::default()
            },
        };
        let out = bake(&options, &dir).unwrap();
        let tolerance = if cfg!(feature = "precision-f64") {
            2e-5
        } else {
            0.003
        };
        for (i, (got, want)) in out
            .data
            .iter()
            .zip(case["expected"].as_array().unwrap())
            .enumerate()
        {
            let want = want.as_f64().unwrap();
            assert!(
                (*got as f64 - want).abs() < tolerance,
                "{} -> {} [{i}]: {got} vs {want}",
                options.input,
                options.output
            );
        }
    }
}

#[test]
fn tapped_chain_reconstructs_full_pipeline() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let mut params = RuntimeParams::default();
    params.debug.lut_mode = true;
    params.io.input_color_space = "sRGB".into();
    params.io.output_cctf_encoding = false;
    let base = Pipeline::new_with_spectral(
        load_profile_by_name(&dir, "kodak_portra_400").unwrap(),
        load_profile_by_name(&dir, "kodak_portra_endura").unwrap(),
        params,
        &dir,
    )
    .unwrap();
    let image = ImageBuf::from_data(2, 2, vec![from_f64(0.184); 12]);
    let cpu = spektrafilm_gpu::cpu_backend::CpuBackend;
    let full = base.process(image.clone(), &cpu);
    let mut stage = image;
    for (inject, collect) in [
        ("rgb_in", "log_e_film"),
        ("log_e_film", "cmy_film"),
        ("cmy_film", "log_e_print"),
        ("log_e_print", "rgb_out"),
    ] {
        let mut pipeline = base.clone();
        pipeline.params.taps.inject = Some(inject.into());
        pipeline.params.taps.collect = Some(collect.into());
        stage = pipeline.process(stage, &cpu);
    }
    assert_eq!(full.data, stage.data);
}

#[test]
fn bounded_preview_matches_skimage_reference() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/preview.json")).unwrap();
    for case in cases.as_array().unwrap() {
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
        let out = spektrafilm_core::resize::preview(&input, case["limit"].as_u64().unwrap() as u32);
        assert_eq!(out.width as u64, case["output_width"].as_u64().unwrap());
        assert_eq!(out.height as u64, case["output_height"].as_u64().unwrap());
        for (got, want) in out.data.iter().zip(case["expected"].as_array().unwrap()) {
            assert!(
                (*got as f64 - want.as_f64().unwrap()).abs() < 2e-6,
                "preview boundary drift: {got} vs {want}"
            );
        }
    }
}

#[test]
fn resident_layered_grain_preserves_reference_tone() {
    let Some(gpu) = spektrafilm_gpu::wgpu_backend::WgpuBackend::new() else {
        return;
    };
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let mut params = RuntimeParams::default();
    params.camera.auto_exposure = false;
    params.camera.film_format_mm = 5.12;
    params.io.input_color_space = "sRGB".into();
    params.io.input_cctf_decoding = false;
    params.film_render.halation.active = false;
    params.film_render.dir_couplers.active = false;
    params.print_render.glare.active = false;
    params.scanner.unsharp_mask = [0., 0.];
    let p = Pipeline::new_with_spectral(
        load_profile_by_name(&dir, "kodak_portra_400").unwrap(),
        load_profile_by_name(&dir, "kodak_portra_endura").unwrap(),
        params,
        &dir,
    )
    .unwrap();
    let input = ImageBuf::from_data(
        1024,
        128,
        (0..1024 * 128)
            .flat_map(|_| [0.18, 0.24, 0.31].map(from_f64))
            .collect(),
    );
    let reference = p.process(input.clone(), &spektrafilm_gpu::cpu_backend::CpuBackend);
    let output = p
        .process_resident_borrowed(&input, &gpu)
        .expect("layered grain left the resident GPU chain");
    for c in 0..3 {
        let mean = |image: &ImageBuf| {
            image.pixels().map(|p| p[c] as f64).sum::<f64>() / image.pixel_count() as f64
        };
        assert!(
            (mean(&reference) - mean(&output)).abs() < 0.003,
            "resident grain changed channel {c} tone"
        );
    }
    assert!(output.data.iter().all(|v| v.is_finite()));
}

#[test]
fn slide_lut_transform_is_independent_of_image_shape() {
    use spektrafilm_core::lut_export::*;
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let mut params = RuntimeParams::default();
    params.io.scan_film = true;
    // Deliberately amplify the optional native film-scanning glare. LUT mode
    // must exclude it along with the print glare and emulsion grain.
    params.film_render.glare.percent = 30.;
    let options = BakeOptions {
        film: "kodak_ektachrome_100",
        paper: "kodak_portra_endura",
        size: 5,
        input: "sRGB",
        output: "sRGB",
        gain: Some(1.),
        params,
    };
    let a = reference_image(
        &options,
        &dir,
        ImageBuf::from_data(1, 1, vec![from_f64(0.3); 3]),
    )
    .unwrap();
    let b = reference_image(
        &options,
        &dir,
        ImageBuf::from_data(2, 2, vec![from_f64(0.3); 12]),
    )
    .unwrap();
    for p in b.pixels() {
        for c in 0..3 {
            assert!((p[c] - a.data[c]).abs() < from_f64(1e-6));
        }
    }
}
