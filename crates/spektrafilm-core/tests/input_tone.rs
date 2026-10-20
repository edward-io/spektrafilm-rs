use spektrafilm_core::{
    lut_export::{self, BakeOptions},
    params::{RuntimeParams, ToneParams},
    pipeline::Pipeline,
    profile::load_profile_by_name,
};
use spektrafilm_gpu::cpu_backend::CpuBackend;
use spektrafilm_math::{
    image::ImageBuf,
    precision::{Scalar, from_f64},
    transfer::Transfer,
};
use std::path::{Path, PathBuf};

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}
fn params() -> RuntimeParams {
    let mut p = RuntimeParams::default();
    p.camera.auto_exposure = false;
    p.camera.tone = ToneParams {
        shadows: 70.,
        highlights: -80.,
        whites: 40.,
        blacks: -25.,
    };
    p.debug.deactivate_spatial_effects = true;
    p.debug.deactivate_stochastic_effects = true;
    p.io.output_cctf_encoding = false;
    p
}
fn build(film: &str, p: RuntimeParams) -> Pipeline {
    Pipeline::new_with_spectral(
        load_profile_by_name(&data(), film).unwrap(),
        load_profile_by_name(&data(), "kodak_portra_endura").unwrap(),
        p,
        &data(),
    )
    .unwrap()
}
/// A tonal ramp with checkerboard texture, so the regional base matters.
fn image() -> ImageBuf {
    ImageBuf::from_data(
        65,
        24,
        (0..65 * 24)
            .flat_map(|i| {
                let texture = if (i % 65 + i / 65) % 2 == 0 {
                    0.3
                } else {
                    -0.3
                };
                let y = 0.184 * 2f64.powf(-8. + 12. * (i % 65) as f64 / 64. + texture);
                [y, y * 0.8, y * 0.65].map(from_f64)
            })
            .collect(),
    )
}
fn difference(a: &ImageBuf, b: &ImageBuf) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height));
    a.data
        .iter()
        .zip(&b.data)
        .map(|(a, b)| {
            assert!(a.is_finite() && b.is_finite(), "{a}, {b}");
            (*a as f64 - *b as f64).abs()
        })
        .fold(0., f64::max)
}

#[test]
fn tone_defaults_partial_settings_and_validation() {
    let old: RuntimeParams =
        serde_json::from_value(serde_json::json!({"camera": {"auto_exposure":false}})).unwrap();
    assert!(old.camera.tone.is_identity());
    let partial: ToneParams = serde_json::from_value(serde_json::json!({"shadows":40})).unwrap();
    assert_eq!(partial.values(), [40., 0., 0., 0.]);
    let p = params();
    let restored: RuntimeParams =
        serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
    assert_eq!(restored.camera.tone.values(), p.camera.tone.values());
    for invalid in [-101., 101., f32::INFINITY, f32::NAN] {
        let mut p = params();
        p.camera.tone.shadows = invalid;
        assert!(p.validate().unwrap_err().contains("tone"));
    }
}

#[test]
fn tone_is_after_camera_gain_before_spectral_conversion_and_excluded_from_calibration() {
    let mut p = params();
    p.camera.exposure_compensation_ev = 1.25;
    p.taps.collect = Some("log_e_film".into());
    let pipeline = build("kodak_gold_200", p.clone());
    let input = image();
    let actual = pipeline.process(input.clone(), &CpuBackend);
    let mut prepared = input;
    let gain = 2f32.powf(p.camera.exposure_compensation_ev) as Scalar;
    prepared.data.iter_mut().for_each(|v| *v *= gain);
    p.camera
        .tone
        .curve(&p.io.input_color_space)
        .apply(&mut prepared);
    p.camera.tone = Default::default();
    let untoned = build("kodak_gold_200", p.clone());
    assert_eq!(
        pipeline.print_exposure_factor(),
        untoned.print_exposure_factor()
    );
    p.camera.exposure_compensation_ev = 0.;
    // Film exposure taps avoid the print normalization's separate EV behavior.
    let expected = untoned.with_params(p).process(prepared, &CpuBackend);
    assert!(difference(&actual, &expected) < 1e-6);
}

#[test]
fn lut_mode_bakes_the_uniform_region_curve() {
    let mut p = params();
    p.taps.collect = Some("log_e_film".into());
    p.debug.lut_mode = true;
    let input = image();
    let actual = build("kodak_gold_200", p.clone()).process(input.clone(), &CpuBackend);
    let mut prepared = input;
    p.camera
        .tone
        .curve(&p.io.input_color_space)
        .apply_pointwise(&mut prepared);
    p.camera.tone = Default::default();
    let expected = build("kodak_gold_200", p).process(prepared, &CpuBackend);
    assert!(difference(&actual, &expected) < 1e-6);
}

#[test]
fn bypass_routes_and_downstream_segments_do_not_apply_tone() {
    for route in [
        "input",
        "input > convert-film > print > scan",
        "input > convert-film > scan",
    ] {
        let mut p = params();
        p.workflow.route = route.into();
        let active = build("kodak_gold_200", p.clone());
        p.camera.tone = Default::default();
        let neutral = build("kodak_gold_200", p);
        let input = ImageBuf::from_data(
            4,
            4,
            (0..48)
                .map(|i| from_f64(0.05 + (i % 12) as f64 / 20.))
                .collect(),
        );
        assert_eq!(
            active.process(input.clone(), &CpuBackend).data,
            neutral.process(input, &CpuBackend).data,
            "{route}"
        );
    }
    let p = build("kodak_gold_200", params());
    let input = image();
    let complete = p.process(input.clone(), &CpuBackend);
    let mut first = p.clone();
    first.params.taps.collect = Some("log_e_film".into());
    let film_exposure = first.process(input, &CpuBackend);
    let mut last = p;
    last.params.taps.inject = Some("log_e_film".into());
    let split = last.process(film_exposure, &CpuBackend);
    assert!(
        difference(&complete, &split) < 1e-6,
        "a segmented transform must include tone once"
    );
}

#[test]
fn complete_look_luts_include_the_selected_tone() {
    let mut options = BakeOptions {
        film: "kodak_gold_200",
        paper: "kodak_portra_endura",
        size: 3,
        input: "srgb",
        output: "srgb",
        gain: Some(1.),
        params: params(),
    };
    let active = lut_export::bake(&options, &data()).unwrap();
    options.params.camera.tone = Default::default();
    let neutral = lut_export::bake(&options, &data()).unwrap();
    assert!(difference(&active, &neutral) > 1e-4);
}

#[test]
fn resident_tone_matches_cpu_for_encoded_metered_and_corrected_slide_inputs() {
    let Some(gpu) = spektrafilm_gpu::wgpu_backend::WgpuBackend::new() else {
        return;
    };
    for (film, scan, auto_exposure, encoded, ev) in [
        ("kodak_portra_400", false, false, false, 1.25),
        ("kodak_gold_200", false, true, true, -0.7),
        ("fujifilm_provia_100f", true, true, false, 0.6),
    ] {
        let mut p = params();
        p.workflow.route = if scan {
            "input > film > scan"
        } else {
            "input > film > print > scan"
        }
        .into();
        p.camera.auto_exposure = auto_exposure;
        p.camera.exposure_compensation_ev = ev;
        p.io.input_cctf_decoding = encoded;
        p.scanner.white_correction = scan;
        p.scanner.black_correction = scan;
        let mut input = image();
        if encoded {
            input
                .data
                .iter_mut()
                .for_each(|v| *v = from_f64(Transfer::Srgb.encode(*v as f64)));
        }
        let pipeline = build(film, p);
        let expected = pipeline.process(input.clone(), &CpuBackend);
        let actual = pipeline
            .process_resident_borrowed(&input, &gpu)
            .expect("tone must stay resident");
        let max = difference(&expected, &actual);
        eprintln!(
            "tone {film}, encoded={encoded}, auto={auto_exposure}: max CPU/GPU difference {max}"
        );
        if scan {
            let mut neutral = pipeline.params.clone();
            neutral.camera.tone = Default::default();
            let neutral = build(film, neutral);
            let cpu = neutral.process(input.clone(), &CpuBackend);
            let gpu = neutral.process_resident_borrowed(&input, &gpu).unwrap();
            assert!(
                difference(&cpu, &gpu) < 0.001,
                "corrected scans must also agree with tone bypassed"
            );
        }
        assert!(max < 0.001, "{film}: {max}");
    }
}
