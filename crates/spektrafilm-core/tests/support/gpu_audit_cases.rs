use spektrafilm_core::params::RuntimeParams;
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
pub fn image() -> ImageBuf {
    let (w, h) = (192, 128);
    let image = ImageBuf::from_data(
        w,
        h,
        (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    let t = x as f64 / (w - 1) as f64;
                    let u = y as f64 / (h - 1) as f64;
                    let rgb = if y < h / 4 {
                        [0.18 * 2f64.powf(-8. + 16. * t); 3]
                    } else if y < h / 2 {
                        [t, u, 1. - t]
                    } else if y < 3 * h / 4 {
                        let a = if (x / 5 + y / 7) % 2 == 0 { 0.01 } else { 0.9 };
                        [a, 0.7 * a, 0.5 * a]
                    } else if (x == 0 || x == w / 2) && (y == h - 1 || y == 7 * h / 8) {
                        [25., 12., 3.]
                    } else {
                        [0.02, 0.04, 0.08]
                    };
                    rgb.map(from_f64)
                })
            })
            .collect(),
    );
    image
}
pub fn cases() -> Vec<(String, RuntimeParams)> {
    let w = 192;
    let mut base = RuntimeParams::default();
    base.debug.deactivate_stochastic_effects = true;
    base.camera.film_format_mm = 35. * w as f32 / 2048.;
    base.camera.auto_exposure = false;
    base.io.input_color_space = "sRGB".into();
    let mut cases = vec![("default".into(), base.clone())];
    let mut p = base.clone();
    p.debug.deactivate_spatial_effects = true;
    cases.push(("no_spatial".into(), p));
    let mut p = base.clone();
    p.scanner.unsharp_mask = [0., 0.];
    cases.push(("no_unsharp".into(), p));
    let mut p = base.clone();
    p.film_render.halation.active = false;
    cases.push(("no_halation".into(), p));
    let mut p = base.clone();
    p.film_render.dir_couplers.active = false;
    cases.push(("no_couplers".into(), p));
    let mut p = base.clone();
    p.film_render.halation.boost_ev = 2.;
    cases.push(("highlight_boost".into(), p));
    let mut p = base.clone();
    p.film_render.dir_couplers.amount = 2.;
    p.film_render.dir_couplers.diffusion_size_um = 60.;
    p.film_render.dir_couplers.diffusion_tail_um = 300.;
    p.film_render.dir_couplers.diffusion_tail_weight = 0.25;
    cases.push(("dir_diffusion".into(), p));
    for blur in [0.7, 3., 32.] {
        let mut p = base.clone();
        p.debug.deactivate_stochastic_effects = false;
        p.film_render.grain.active = false;
        p.print_render.glare.active = true;
        p.print_render.glare.percent = 5.;
        p.print_render.glare.roughness = 0.;
        p.print_render.glare.blur = blur;
        cases.push((format!("deterministic_glare_{blur}"), p));
    }
    for family in ["black_pro_mist", "pro_mist", "glimmerglass", "cinebloom"] {
        for camera in [true, false] {
            let mut p = base.clone();
            let d = if camera {
                &mut p.camera.diffusion_filter
            } else {
                &mut p.enlarger.diffusion_filter
            };
            d.active = true;
            d.filter_family = family.into();
            d.strength = 1.;
            cases.push((
                format!("{}_{}", if camera { "camera" } else { "enlarger" }, family),
                p,
            ));
        }
    }
    let mut p = base.clone();
    p.camera.lens_blur_um = 20.;
    p.scanner.lens_blur = 1.3;
    p.scanner.unsharp_mask = [1.7, 1.2];
    cases.push(("lens_unsharp".into(), p));
    let mut p = base.clone();
    p.camera.lens_blur_um = 300.;
    p.scanner.lens_blur = 7.;
    p.scanner.unsharp_mask = [12., 1.2];
    cases.push(("wide_lens_unsharp".into(), p));
    let mut p = base.clone();
    p.camera.diffusion_filter.active = true;
    p.camera.diffusion_filter.strength = 1.2;
    p.camera.diffusion_filter.halo_warmth = -1.2;
    p.camera.diffusion_filter.core_intensity = 0.2;
    p.camera.diffusion_filter.bloom_size = 0.3;
    cases.push(("filter_advanced_controls".into(), p));
    let mut p = base.clone();
    p.camera.diffusion_filter.active = true;
    p.enlarger.diffusion_filter.active = true;
    p.enlarger.diffusion_filter.filter_family = "glimmerglass".into();
    cases.push(("both_diffusion_stages".into(), p));
    let mut p = base.clone();
    p.io.input_cctf_decoding = true;
    p.io.output_cctf_encoding = false;
    cases.push(("encoded_input_linear_output".into(), p));
    let mut p = base.clone();
    p.io.output_color_space = "Display P3".into();
    p.camera.diffusion_filter.active = true;
    p.enlarger.diffusion_filter.active = true;
    cases.push(("staged_color_diffusion".into(), p));
    let mut p = base.clone();
    p.io.input_cctf_decoding = true;
    p.camera.diffusion_filter.active = true;
    p.enlarger.diffusion_filter.active = true;
    cases.push(("encoded_input_diffusion".into(), p));
    for gamut in ["off", "aces_rgc", "oklch", "oklrab", "jzazbz"] {
        let mut p = base.clone();
        p.io.output_gamut_compress.algorithm = gamut.into();
        cases.push((format!("gamut_{gamut}"), p));
    }
    for cs in ["Display P3", "ProPhoto RGB", "ACEScg"] {
        let mut p = base.clone();
        p.io.output_color_space = cs.into();
        cases.push((format!("output_{cs}"), p));
    }
    for resolution in [3, 17, 33] {
        for scanner in [false, true] {
            let mut p = base.clone();
            p.settings.lut_resolution = resolution;
            p.settings.use_enlarger_lut = !scanner;
            p.settings.use_scanner_lut = scanner;
            p.enlarger.preflash_exposure = 0.1;
            cases.push((
                format!(
                    "spectral_lut_{resolution}_{}",
                    if scanner { "scanner" } else { "enlarger" }
                ),
                p,
            ));
        }
    }
    let mut p = base.clone();
    p.settings.use_enlarger_lut = true;
    p.settings.use_scanner_lut = true;
    p.camera.diffusion_filter.active = true;
    p.enlarger.diffusion_filter.active = true;
    cases.push(("spectral_luts_diffusion".into(), p));
    cases
}
