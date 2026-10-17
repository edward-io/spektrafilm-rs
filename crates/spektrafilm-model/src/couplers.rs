// DIR (Development Inhibitor Release) coupler model.
// Handles same-layer and inter-layer inhibition with spatial diffusion.

use rayon::prelude::*;
use spektrafilm_gpu::ComputeBackend;
use spektrafilm_math::image::ImageBuf;
use spektrafilm_math::precision::from_f64;

/// DIR couplers matrix [3][3]. Row = donor layer, Column = receiver layer.
pub fn compute_dir_couplers_matrix(
    gamma_samelayer: [f64; 3],
    gamma_r_to_gb: [f64; 2],
    gamma_g_to_rb: [f64; 2],
    gamma_b_to_rg: [f64; 2],
    inhibition_samelayer: f64,
    inhibition_interlayer: f64,
) -> [[f64; 3]; 3] {
    let mut m = [[0.0f64; 3]; 3];
    m[0][0] = gamma_samelayer[0] * inhibition_samelayer;
    m[1][1] = gamma_samelayer[1] * inhibition_samelayer;
    m[2][2] = gamma_samelayer[2] * inhibition_samelayer;
    m[0][1] = gamma_r_to_gb[0] * inhibition_interlayer;
    m[0][2] = gamma_r_to_gb[1] * inhibition_interlayer;
    m[1][0] = gamma_g_to_rb[0] * inhibition_interlayer;
    m[1][2] = gamma_g_to_rb[1] * inhibition_interlayer;
    m[2][0] = gamma_b_to_rg[0] * inhibition_interlayer;
    m[2][1] = gamma_b_to_rg[1] * inhibition_interlayer;
    m
}

/// The saturation knees use the unit matrix, so changing amount leaves them fixed.
#[derive(Clone, Copy, Debug)]
pub struct CouplerResponse {
    pub density_max: [f64; 3],
    pub donor_k: [f64; 3],
    pub receiver_k: [f64; 3],
    pub receiver_ref: [f64; 3],
    pub matrix: [[f64; 3]; 3],
    pub positive: bool,
}

#[inline]
pub fn langmuir(density: f64, knee: f64, reference: f64) -> f64 {
    if !knee.is_finite() {
        density
    } else if knee + density == 0.0 {
        0.0
    } else {
        density * (knee + reference) / (knee + density)
    }
}

impl CouplerResponse {
    pub fn new(
        curves: &[[f64; 3]],
        matrix: [[f64; 3]; 3],
        amount: f64,
        donor_k: [f64; 3],
        receiver_k: [f64; 3],
        positive: bool,
    ) -> Self {
        let density_max = crate::density_curves::max_density_f64(curves);
        let receiver_ref =
            std::array::from_fn(|m| (0..3).map(|c| density_max[c] * 0.5 * matrix[c][m]).sum());
        Self {
            density_max,
            donor_k: std::array::from_fn(|c| donor_k[c] * density_max[c]),
            receiver_k: std::array::from_fn(|c| receiver_k[c] * 2.0 * receiver_ref[c]),
            receiver_ref,
            matrix: matrix.map(|row| row.map(|v| v * amount)),
            positive,
        }
    }
    fn released(&self, density: [f64; 3]) -> [f64; 3] {
        let donor: [f64; 3] = std::array::from_fn(|c| {
            if self.positive {
                self.density_max[c] - density[c]
            } else {
                langmuir(density[c], self.donor_k[c], self.density_max[c] * 0.5)
            }
        });
        std::array::from_fn(|m| (0..3).map(|c| donor[c] * self.matrix[c][m]).sum())
    }
    fn received(&self, correction: f64, c: usize) -> f64 {
        if self.positive {
            langmuir(correction, self.receiver_k[c], self.receiver_ref[c])
        } else {
            correction
        }
    }
    pub fn curves_before(&self, curves: &[[f64; 3]], axis: &[f64]) -> Vec<[f64; 3]> {
        let correction: Vec<_> = curves.iter().map(|&row| self.released(row)).collect();
        let mut result = vec![[0.; 3]; curves.len()];
        for c in 0..3 {
            let shifted: Vec<_> = axis
                .iter()
                .enumerate()
                .map(|(i, x)| x - self.received(correction[i][c], c))
                .collect();
            let values: Vec<_> = curves.iter().map(|r| r[c]).collect();
            let sampled = spektrafilm_math::interp::interp_batch_f64(&shifted, &values, axis);
            for (i, v) in sampled.iter().enumerate() {
                result[i][c] = *v;
            }
        }
        result
    }
}

#[allow(clippy::too_many_arguments)]
pub fn compute_exposure_correction(
    log_raw: &ImageBuf,
    density_cmy: &ImageBuf,
    response: &CouplerResponse,
    diffusion_size_pixel: f32,
    diffusion_tail_pixel: f32,
    diffusion_tail_weight: f64,
    backend: &dyn ComputeBackend,
) -> ImageBuf {
    let mut correction = ImageBuf::new(log_raw.width, log_raw.height);
    correction
        .data
        .par_chunks_exact_mut(3)
        .zip(density_cmy.data.par_chunks_exact(3))
        .for_each(|(out, px)| {
            let released = response.released([px[0] as f64, px[1] as f64, px[2] as f64]);
            for c in 0..3 {
                out[c] = from_f64(released[c]);
            }
        });
    if diffusion_size_pixel > 0.0 {
        if let Some(output) = backend.dir_diffusion(
            &correction,
            diffusion_size_pixel,
            diffusion_tail_pixel,
            diffusion_tail_weight as f32,
        ) {
            correction = output;
        } else {
            // Python: `(1 - w) * fast_gaussian_filter(corr, σ_size)
            //         + w * fast_exponential_filter(corr, σ_tail)`.
            // The tail kernel is EXPONENTIAL (3-Gaussian mixture), not
            // another Gaussian — using two Gaussians here was a structural
            // diff vs Python.
            use spektrafilm_math::gaussian::{exponential_filter_channel, gaussian_blur_channel};
            let w_img = correction.width;
            let h_img = correction.height;
            let n_pix = (w_img as usize) * (h_img as usize);
            let w = from_f64(diffusion_tail_weight);
            let one = from_f64(1.0);
            let mut blended_channels: [Vec<spektrafilm_math::precision::Scalar>; 3] =
                [vec![one; n_pix], vec![one; n_pix], vec![one; n_pix]];
            for c in 0..3 {
                let ch = correction.extract_channel(c);
                let g = gaussian_blur_channel(&ch, w_img, h_img, diffusion_size_pixel);
                let t = exponential_filter_channel(&ch, w_img, h_img, diffusion_tail_pixel);
                for i in 0..n_pix {
                    blended_channels[c][i] = (one - w) * g[i] + w * t[i];
                }
            }
            for c in 0..3 {
                correction.write_channel(c, &blended_channels[c]);
            }
        }
    }

    let mut result = log_raw.clone();
    result
        .data
        .par_chunks_exact_mut(3)
        .zip(correction.data.par_chunks_exact(3))
        .for_each(|(out, corr)| {
            for c in 0..3 {
                out[c] -= from_f64(response.received(corr[c] as f64, c));
            }
        });
    result
}

#[allow(clippy::too_many_arguments)]
pub fn apply_density_correction(
    density_cmy: &ImageBuf,
    log_raw: &ImageBuf,
    pixel_size_um: f32,
    log_exposure: &[f64],
    density_curves: &[[f64; 3]],
    couplers_matrix: &[[f64; 3]; 3],
    amount: f64,
    donor_k: [f64; 3],
    receiver_k: [f64; 3],
    diffusion_size_um: f64,
    diffusion_tail_um: f64,
    diffusion_tail_weight: f64,
    positive: bool,
    gamma_factor: f64,
    backend: &dyn ComputeBackend,
) -> ImageBuf {
    let curves = crate::density_curves::normalize_density_curves_f64(density_curves);
    let response = CouplerResponse::new(
        &curves,
        *couplers_matrix,
        amount,
        donor_k,
        receiver_k,
        positive,
    );
    let curves_before = response.curves_before(&curves, log_exposure);
    let corrected = compute_exposure_correction(
        log_raw,
        density_cmy,
        &response,
        (diffusion_size_um / pixel_size_um as f64) as f32,
        (diffusion_tail_um / pixel_size_um as f64) as f32,
        diffusion_tail_weight,
        backend,
    );
    backend.density_curve_interp(&corrected, log_exposure, &curves_before, gamma_factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn langmuir_preserves_operating_point_and_linear_limit() {
        assert_eq!(langmuir(1., 2., 1.), 1.);
        assert_eq!(langmuir(3., f64::INFINITY, 1.), 3.);
        assert_eq!(langmuir(0., 0., 0.), 0.);
        assert!(langmuir(3., 2., 1.) < 3.);
        let a = CouplerResponse::new(
            &[[0.; 3], [2.; 3]],
            [[0.2; 3]; 3],
            1.,
            [1.; 3],
            [1.; 3],
            true,
        );
        let b = CouplerResponse::new(
            &[[0.; 3], [2.; 3]],
            [[0.2; 3]; 3],
            4.,
            [1.; 3],
            [1.; 3],
            true,
        );
        assert_eq!(a.receiver_k, b.receiver_k);
        assert_eq!(a.received(a.receiver_ref[0], 0), a.receiver_ref[0]);
    }
}
