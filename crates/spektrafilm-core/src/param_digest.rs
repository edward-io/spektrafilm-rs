use crate::params::*;

impl RuntimeParams {
    pub fn validate(&self) -> Result<(), String> {
        spektrafilm_math::illuminants::get(&self.enlarger.illuminant)?;
        for mut order in [
            self.film_render.channel_order,
            self.print_render.channel_order,
        ] {
            order.sort();
            if order != [0, 1, 2] {
                return Err("channel_order must be a permutation of [0, 1, 2]".into());
            }
        }
        for (name, encoded) in [
            (&self.io.input_color_space, self.io.input_cctf_decoding),
            (&self.io.output_color_space, self.io.output_cctf_encoding),
        ] {
            let cs = spektrafilm_math::transfer::get(name)?;
            if encoded && !cs.transfer.supported() {
                return Err(format!("transfer function for {name} is not supported"));
            }
        }
        if !["off", "xy", "oklch"].contains(&self.io.input_gamut_compress.algorithm.as_str()) {
            return Err("unknown input gamut compression algorithm".into());
        }
        if !["off", "aces_rgc", "oklch", "oklrab", "cam16ucs", "jzazbz"]
            .contains(&self.io.output_gamut_compress.algorithm.as_str())
        {
            return Err("unknown output gamut compression algorithm".into());
        }
        for knee in [
            self.io.input_gamut_compress.knee,
            self.io.output_gamut_compress.knee,
        ] {
            if !knee.iter().all(|v| v.is_finite())
                || !(0.0..1.0).contains(&knee[0])
                || knee[1] <= knee[0]
                || knee[2] <= 0.
            {
                return Err(
                    "gamut knee requires 0 <= threshold < 1, limit > threshold, and power > 0"
                        .into(),
                );
            }
        }
        if !self.io.upscale_factor.is_finite() || self.io.upscale_factor <= 0. {
            return Err("upscale_factor must be finite and positive".into());
        }
        if self.settings.lut_resolution < 2 {
            return Err("lut_resolution must be at least 2".into());
        }
        let g = &self.film_render.grain;
        if !g.agx_particle_area_um2.is_finite()
            || !g
                .agx_particle_scale
                .iter()
                .chain(g.agx_particle_scale_layers.iter())
                .all(|v| v.is_finite() && *v > 0.)
            || !g
                .uniformity
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            || !g.density_min.iter().all(|v| v.is_finite() && *v >= 0.)
            || ![
                g.blur,
                g.blur_dye_clouds_um,
                g.micro_structure[0],
                g.micro_structure[1],
            ]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.)
        {
            return Err("grain scales must be positive, blur/density/microstructure nonnegative, and uniformity in [0, 1]".into());
        }
        if self.film_render.grain.n_sub_layers == 0
            || self.film_render.grain.agx_particle_area_um2 <= 0.
        {
            return Err("grain needs positive particle area and sublayer count".into());
        }
        if self.io.crop
            && self
                .io
                .crop_size
                .iter()
                .any(|x| !x.is_finite() || *x <= 0. || *x > 1.)
        {
            return Err("crop_size must be in (0, 1]".into());
        }
        let names = [
            "rgb_in",
            "rgb_pre",
            "log_e_film",
            "cmy_film",
            "log_e_print",
            "cmy_print",
            "rgb_out",
        ];
        let index = |s: &str| {
            names
                .iter()
                .position(|x| *x == s)
                .ok_or_else(|| format!("unknown pipeline tap: {s}"))
        };
        let start = index(self.taps.inject.as_deref().unwrap_or("rgb_in"))?;
        let end = index(self.taps.collect.as_deref().unwrap_or("rgb_out"))?;
        if start > end {
            return Err("collect tap precedes inject tap".into());
        }
        if self.io.scan_film && ((start == 4 || start == 5) || (end == 4 || end == 5)) {
            return Err("print taps are unavailable in scan_film mode".into());
        }
        Ok(())
    }

    /// Apply the same preview/debug regime as Python's digest_params.
    pub fn digest(&mut self) {
        if self.settings.preview_mode {
            self.enlarger.lens_blur = 0.;
            self.film_render.dir_couplers.diffusion_size_um = 0.;
            self.film_render.grain.active = false;
            self.film_render.grain.blur = 0.;
            self.print_render.glare.blur = 0.;
            self.film_render.glare.blur = 0.;
            self.camera.lens_blur_um = 0.;
            self.scanner.lens_blur = 0.;
            self.scanner.unsharp_mask = [0., 0.];
        }
        if self.debug.lut_mode {
            self.debug.deactivate_spatial_effects = true;
            self.debug.deactivate_stochastic_effects = true;
            self.camera.auto_exposure = false;
            self.camera.exposure_compensation_ev = 0.;
            self.enlarger.print_exposure_compensation = false;
            self.enlarger.print_exposure = 1.;
            self.film_render.halation.boost_ev = 0.;
            self.scanner.white_correction = false;
            self.scanner.black_correction = false;
            self.scanner.unsharp_mask = [0., 0.];
        }
        if self.debug.deactivate_spatial_effects {
            self.film_render.halation.active = false;
            self.film_render.dir_couplers.diffusion_size_um = 0.;
            self.film_render.grain.blur = 0.;
            self.film_render.grain.blur_dye_clouds_um = 0.;
            self.print_render.glare.blur = 0.;
            self.film_render.glare.blur = 0.;
            self.camera.lens_blur_um = 0.;
            self.enlarger.lens_blur = 0.;
            self.camera.diffusion_filter.active = false;
            self.enlarger.diffusion_filter.active = false;
            self.scanner.lens_blur = 0.;
            self.scanner.unsharp_mask = [0., 0.];
        }
        if self.debug.deactivate_stochastic_effects {
            self.film_render.grain.active = false;
            self.print_render.glare.active = false;
            self.film_render.glare.active = false;
        }
    }
}
