mod debug_compare;
pub mod filming;
pub mod printing;
pub mod scanning;

#[cfg(test)]
mod integration_tests {
    use crate::params::RuntimeParams;
    use crate::pipeline::Pipeline;
    use crate::profile;
    use spektrafilm_math::image::ImageBuf;
    use spektrafilm_math::precision::{Scalar, from_f64};
    use std::path::Path;

    fn data_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("data")
    }

    #[test]
    fn test_full_pipeline_portra_400_to_endura() {
        let dir = data_dir();
        let film = profile::load_profile_by_name(&dir, "kodak_portra_400").unwrap();
        let print = profile::load_profile_by_name(&dir, "kodak_portra_endura").unwrap();

        let mut params = RuntimeParams::default();
        params.film_render.grain.active = false;
        params.film_render.halation.active = false;
        params.film_render.dir_couplers.active = false;
        params.camera.auto_exposure = false;

        let backend = spektrafilm_gpu::cpu_backend::CpuBackend;
        let img = ImageBuf::from_data(8, 8, vec![from_f64(0.184); 8 * 8 * 3]);

        let pipeline = Pipeline::new(film, print, params);
        let result = pipeline.process(img, &backend);

        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
        let px = result.get(4, 4);
        for c in 0..3 {
            assert!(
                px[c] >= from_f64(0.0) && px[c] <= from_f64(1.0),
                "channel {c} out of range: {}",
                px[c]
            );
        }
        let mean: Scalar =
            result.data.iter().copied().sum::<Scalar>() / result.data.len() as Scalar;
        assert!(mean > from_f64(0.01), "output near-black: mean={mean}");
        assert!(mean < from_f64(0.99), "output near-white: mean={mean}");
    }

    #[test]
    fn test_full_pipeline_with_spectral_lut() {
        let dir = data_dir();
        let film = profile::load_profile_by_name(&dir, "kodak_portra_400").unwrap();
        let print = profile::load_profile_by_name(&dir, "kodak_portra_endura").unwrap();

        let mut params = RuntimeParams::default();
        params.film_render.grain.active = false;
        params.film_render.halation.active = false;
        params.film_render.dir_couplers.active = false;
        params.camera.auto_exposure = false;
        params.io.input_color_space = "sRGB".to_string();

        let backend = spektrafilm_gpu::cpu_backend::CpuBackend;
        let img = ImageBuf::from_data(8, 8, vec![from_f64(0.184); 8 * 8 * 3]);

        let pipeline = Pipeline::new_with_spectral(film, print, params, &dir);
        match pipeline {
            Ok(p) => {
                let result = p.process(img, &backend);
                let mean: Scalar =
                    result.data.iter().copied().sum::<Scalar>() / result.data.len() as Scalar;
                eprintln!("Spectral pipeline output mean: {mean}");
                let px = result.get(4, 4);
                eprintln!("Spectral pipeline pixel(4,4): {:?}", px);
                assert!(
                    mean > from_f64(0.01),
                    "spectral output near-black: mean={mean}"
                );
                assert!(
                    mean < from_f64(0.99),
                    "spectral output near-white: mean={mean}"
                );
            }
            Err(e) => {
                eprintln!("Spectral LUT not available: {e} — skipping test");
            }
        }
    }

    /// Experimental's fitted curves, gamut defaults, and spectral normalization.
    #[test]
    fn full_chain_matches_python_experimental_defaults() {
        let dir = data_dir();
        let film = profile::load_profile_by_name(&dir, "kodak_portra_400").unwrap();
        let print = profile::load_profile_by_name(&dir, "kodak_portra_endura").unwrap();

        let mut params = RuntimeParams::default();
        params.camera.auto_exposure = false;
        params.io.input_color_space = "sRGB".to_string();
        params.io.input_cctf_decoding = false;
        params.io.output_cctf_encoding = false;
        params.film_render.grain.active = false;
        params.film_render.halation.active = false;
        params.film_render.dir_couplers.active = false;
        params.print_render.glare.active = false;
        params.scanner.unsharp_mask = [0.0, 0.0];

        let backend = spektrafilm_gpu::cpu_backend::CpuBackend;
        let pipeline =
            Pipeline::new_with_spectral(film, print, params, &dir).expect("spectral LUT");

        // The f32 Scalar chain carries ~1e-4 of accumulated rounding against
        // the f64 Python reference; the f64 build tracks it tightly.
        #[cfg(feature = "precision-f64")]
        const TOL: f64 = 1e-6;
        #[cfg(not(feature = "precision-f64"))]
        const TOL: f64 = 3e-3;

        // 1×1 midgray.
        let img = ImageBuf::from_data(1, 1, vec![from_f64(0.184); 3]);
        let out = pipeline.process(img, &backend);
        let want: [f64; 3] =
            serde_json::from_value(crate::experimental_reference()["extra"]["gray"].clone())
                .unwrap();
        for c in 0..3 {
            let got = out.get(0, 0)[c] as f64;
            assert!(
                (got - want[c]).abs() < TOL,
                "midgray ch {c}: {got} vs {} (diff {:.3e})",
                want[c],
                (got - want[c]).abs()
            );
        }

        // 4×4 deterministic gradient (same integer formula on both sides).
        let (w, h) = (4u32, 4u32);
        let data: Vec<_> = (0..w * h * 3)
            .map(|i| from_f64(0.05 + 0.5 * ((i * 37) % 256) as f64 / 255.0))
            .collect();
        let img = ImageBuf::from_data(w, h, data);
        let out = pipeline.process(img, &backend);
        let want: Vec<[f64; 3]> =
            serde_json::from_value(crate::experimental_reference()["extra"]["ramp"].clone())
                .unwrap();
        let mut max_diff = 0.0f64;
        for (i, want_px) in want.iter().enumerate() {
            let px = out.get(i as u32 % w, i as u32 / w);
            for c in 0..3 {
                let d = (px[c] as f64 - want_px[c]).abs();
                max_diff = max_diff.max(d);
                assert!(
                    d < TOL,
                    "px {i} ch {c}: {} vs {} (diff {d:.3e})",
                    px[c],
                    want_px[c]
                );
            }
        }
        eprintln!("full-chain 0.3.4 defaults: max diff vs Python {max_diff:.3e}");
    }

    #[test]
    fn test_film_scan_pipeline() {
        let dir = data_dir();
        let film = profile::load_profile_by_name(&dir, "kodak_portra_400").unwrap();
        let print = film.clone();

        let mut params = RuntimeParams::default();
        params.io.scan_film = true;
        params.film_render.grain.active = false;
        params.film_render.halation.active = false;
        params.film_render.dir_couplers.active = false;
        params.camera.auto_exposure = false;

        let backend = spektrafilm_gpu::cpu_backend::CpuBackend;
        let img = ImageBuf::from_data(4, 4, vec![from_f64(0.184); 4 * 4 * 3]);

        let pipeline = Pipeline::new(film, print, params);
        let result = pipeline.process(img, &backend);
        let mean: Scalar =
            result.data.iter().copied().sum::<Scalar>() / result.data.len() as Scalar;
        assert!(mean > from_f64(0.01), "film scan near-black: mean={mean}");
        assert!(mean < from_f64(0.99), "film scan near-white: mean={mean}");
    }

    #[test]
    fn positive_slide_film_scan_stays_photographic() {
        let dir = data_dir();
        let film = profile::load_profile_by_name(&dir, "kodak_ektachrome_100").unwrap();
        let print = film.clone();

        let mut params = RuntimeParams::default();
        params.io.scan_film = true;
        params.camera.auto_exposure = false;

        let backend = spektrafilm_gpu::cpu_backend::CpuBackend;
        let mut data = Vec::with_capacity(64 * 64 * 3);
        for y in 0..64 {
            for x in 0..64 {
                data.push(from_f64(x as f64 / 63.0));
                data.push(from_f64(y as f64 / 63.0));
                data.push(from_f64(((x + y) as f64 / 126.0).powf(1.2)));
            }
        }
        let img = ImageBuf::from_data(64, 64, data);

        let pipeline = Pipeline::new_with_spectral(film, print, params.clone(), &dir)
            .unwrap()
            .with_params(params);
        assert_eq!(
            pipeline.params.film_render.dir_couplers.gamma_samelayer_rgb,
            [0.2291, 0.1029, 0.2651],
            "GUI-style with_params must preserve positive-film DIR parameters"
        );
        let result = pipeline.process(img, &backend);
        let mut clipped_low = 0usize;
        let mut clipped_high = 0usize;
        for c in 0..3 {
            let values = result.data.iter().skip(c).step_by(3);
            let (mut min, mut max, mut sum) = (Scalar::INFINITY, Scalar::NEG_INFINITY, 0.0);
            let mut n = 0usize;
            for &v in values {
                min = min.min(v);
                max = max.max(v);
                sum += v;
                if v <= from_f64(0.0) {
                    clipped_low += 1;
                }
                if v >= from_f64(1.0) {
                    clipped_high += 1;
                }
                n += 1;
            }
            let mean = sum / n as Scalar;
            assert!(
                mean > from_f64(0.2) && mean < from_f64(0.8),
                "slide scan channel {c} mean is implausible: {mean} (min={min}, max={max})"
            );
        }
        let total = result.data.len();
        assert!(
            clipped_low < total / 20,
            "slide scan has too many black-clipped samples: {clipped_low}/{total}"
        );
        assert!(
            clipped_high < total / 20,
            "slide scan has too many white-clipped samples: {clipped_high}/{total}"
        );
    }
}

#[cfg(test)]
mod debug_tests {
    use crate::params::RuntimeParams;
    use crate::profile;
    use crate::stages;
    use spektrafilm_math::image::ImageBuf;
    use spektrafilm_math::precision::from_f64;
    use std::path::Path;

    #[test]
    fn debug_pipeline_values() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("data");
        let film = profile::load_profile_by_name(&dir, "kodak_portra_400").unwrap();
        let print = profile::load_profile_by_name(&dir, "kodak_portra_endura").unwrap();

        let mut params = RuntimeParams::default();
        params.film_render.grain.active = false;
        params.film_render.halation.active = false;
        params.film_render.dir_couplers.active = false;
        params.camera.auto_exposure = false;
        params.io.input_color_space = "sRGB".to_string();

        let backend = spektrafilm_gpu::cpu_backend::CpuBackend;
        let gray = from_f64(0.184);
        let img = ImageBuf::from_data(1, 1, vec![gray, gray, gray]);
        eprintln!("Input: {:?}", img.get(0, 0));

        let ref_illuminant = stages::filming::select_illuminant(&film.info.reference_illuminant);
        let log_raw =
            stages::filming::expose(&img, &film, &params, &backend, None, ref_illuminant, 1.0);
        eprintln!("log_raw: {:?}", log_raw.get(0, 0));

        let density_cmy = stages::filming::develop(&log_raw, &film, &params, &backend);
        eprintln!("density_cmy: {:?}", density_cmy.get(0, 0));

        // Use simplified printing path for debug trace
        let printed = stages::printing::process(&density_cmy, &film, &print, &params, &backend);
        eprintln!("density_print: {:?}", printed.get(0, 0));
        let density_print = printed;
        let rgb_out = stages::scanning::scan(
            &density_print,
            &print,
            &params,
            &backend,
            &crate::color_reference::ColorReference::identity(),
            &crate::gamut_compression::OutputGamutCompress::identity(),
        );
        eprintln!("rgb_out: {:?}", rgb_out.get(0, 0));
    }
}
