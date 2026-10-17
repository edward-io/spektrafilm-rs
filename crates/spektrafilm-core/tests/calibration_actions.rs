use spektrafilm_core::{
    calibration,
    params::{BaseDensityParams, RuntimeParams},
    pipeline::Pipeline,
    profile::load_profile_by_name,
    stock,
};
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
use spektrafilm_model::convert::{FilmConverter, parse_calibration};
fn directory() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}
fn params() -> RuntimeParams {
    let mut p = RuntimeParams::default();
    p.camera.auto_exposure = false;
    p.io.input_color_space = "sRGB".into();
    p.io.input_cctf_decoding = false;
    p.io.output_cctf_encoding = false;
    p.scanner.unsharp_mask = [0.; 2];
    p.debug.deactivate_stochastic_effects = true;
    p.debug.deactivate_spatial_effects = true;
    p
}
#[test]
fn clear_film_fit_recovers_color_and_scan_level() {
    let p = params();
    let film = load_profile_by_name(&directory(), "kodak_portra_400").unwrap();
    let base = BaseDensityParams {
        cyan: 1.2,
        magenta: 0.85,
        yellow: 1.1,
        ..Default::default()
    };
    let dyes: Vec<_> = film
        .data
        .channel_density
        .iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    let illum = spektrafilm_math::illuminants::get("D55").unwrap();
    let truth = FilmConverter::new(
        &dyes,
        &stock::tune_base(&film.data.base_density, &base, true),
        illum,
        "sRGB",
        [3.; 3],
    )
    .forward_jac([0.; 3])
    .0;
    let image = ImageBuf::from_data(
        4,
        4,
        (0..16)
            .flat_map(|_| truth.map(|v| from_f64(v / 1.5)))
            .collect(),
    );
    let (fitted, ev) = calibration::detect_base(&image, film.clone(), &p).unwrap();
    let actual = FilmConverter::new(
        &dyes,
        &stock::tune_base(&film.data.base_density, &fitted, true),
        illum,
        "sRGB",
        [3.; 3],
    )
    .forward_jac([0.; 3])
    .0;
    let expected = truth.map(|v| v / 1.5 * 2f64.powf(ev));
    for c in 0..3 {
        assert!(
            (actual[c] - expected[c]).abs() < 0.003,
            "{actual:?} vs {expected:?}"
        );
    }
    assert!(ev.is_finite());
}
#[test]
fn blind_device_fit_preserves_an_ideal_colorimetric_scan() {
    let p = params();
    let film = load_profile_by_name(&directory(), "kodak_portra_400").unwrap();
    let prepared = stock::prepare(film.clone(), &p, true).unwrap();
    let model = spektrafilm_core::conversion::Conversion::new(&prepared, &p)
        .unwrap()
        .model;
    let image = ImageBuf::from_data(
        65,
        1,
        (0..65)
            .flat_map(|i| {
                let d = if i == 64 {
                    [0.; 3]
                } else {
                    std::array::from_fn(|c| 0.03 + 0.5 * ((i >> (c * 2)) & 3) as f64 / 3.)
                };
                model.forward_jac(d).0.map(from_f64)
            })
            .collect(),
    );
    let fitted = parse_calibration(&calibration::blind_calibration(&image, film, &p).unwrap());
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (fitted[i][j] - if i == j { 1. } else { 0. }).abs() < 0.01,
                "{fitted:?}"
            );
        }
    }
}
#[test]
fn printer_balance_neutralizes_a_tuned_film_base() {
    let data = directory();
    let mut p = params();
    p.film_render.base.cyan = 1.25;
    let film = load_profile_by_name(&data, "kodak_portra_400").unwrap();
    let paper = load_profile_by_name(&data, "kodak_portra_endura").unwrap();
    let [m, y] =
        calibration::neutral_print_balance(film.clone(), paper.clone(), &p, &data).unwrap();
    p.enlarger.m_filter_shift = m;
    p.enlarger.y_filter_shift = y;
    p.io.output_gamut_compress.algorithm = "off".into();
    let pipeline = Pipeline::new_with_spectral(film, paper, p, &data).unwrap();
    let out = pipeline.process(
        ImageBuf::from_data(2, 2, vec![from_f64(0.184); 12]),
        &spektrafilm_gpu::cpu_backend::CpuBackend,
    );
    let px = out.get(0, 0);
    assert!((px[0] / px[1] - 1.).abs() < 0.005, "{px:?}");
    assert!((px[2] / px[1] - 1.).abs() < 0.005, "{px:?}");
}
