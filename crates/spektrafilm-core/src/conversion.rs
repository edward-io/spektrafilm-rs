//! Cached scan model. Exposure and device calibration remain outside the LUT.
use crate::{params::RuntimeParams, profile::Profile};
use rayon::prelude::*;
use spektrafilm_math::{image::ImageBuf, lut::TetrahedralLut, precision::from_f64};
use spektrafilm_model::convert::{FilmConverter, parse_calibration};
#[derive(Clone)]
pub struct Conversion {
    pub model: FilmConverter,
    pub lut: Option<TetrahedralLut>,
    pub bounds: ([f64; 3], [f64; 3]),
}
impl Conversion {
    pub fn new(film: &Profile, p: &RuntimeParams) -> Result<Self, String> {
        let dyes: Vec<[f64; 3]> = film
            .data
            .channel_density
            .iter()
            .map(|r| [r[0], r[1], r[2]])
            .collect();
        let maximum = spektrafilm_model::density_curves::max_density_f64(
            &spektrafilm_model::density_curves::normalize_density_curves_f64(
                &film.density_curves_f64(),
            ),
        );
        let model = FilmConverter::new(
            &dyes,
            &film.data.base_density,
            spektrafilm_math::illuminants::get(&p.film_render.convert.scan_illuminant)?,
            &p.io.input_color_space,
            maximum,
        );
        let bounds = model.bounds();
        let lut = if p.settings.use_convert_lut {
            let n = p.settings.convert_lut_resolution as usize;
            let values: Vec<[f64; 3]> = (0..n * n * n)
                .into_par_iter()
                .map(|i| {
                    let index = [i / (n * n), (i / n) % n, i % n];
                    model.invert(std::array::from_fn(|c| {
                        bounds.0[c] + (bounds.1[c] - bounds.0[c]) * index[c] as f64 / (n - 1) as f64
                    }))
                })
                .collect();
            Some(TetrahedralLut {
                size: n,
                data: values.into_iter().flatten().collect(),
            })
        } else {
            None
        };
        Ok(Self { model, lut, bounds })
    }
    pub fn process(
        &self,
        image: &ImageBuf,
        p: &RuntimeParams,
        backend: &dyn spektrafilm_gpu::ComputeBackend,
    ) -> ImageBuf {
        let calibration = parse_calibration(&p.film_render.convert.calibration);
        let gain = 2f64.powf(p.film_render.convert.exposure_compensation_ev);
        let Some(lut) = &self.lut else {
            return self.model.convert(image, calibration, gain);
        };
        let mut input = image.clone();
        for px in input.pixels_mut() {
            let row = [px[0], px[1], px[2]];
            for c in 0..3 {
                px[c] = from_f64(
                    (0..3)
                        .map(|k| row[k] as f64 * calibration[k][c])
                        .sum::<f64>()
                        * gain,
                );
            }
        }
        if let Some(out) = backend.tetrahedral3d(&input, lut, self.bounds.0, self.bounds.1) {
            return out;
        }
        input.data.par_chunks_exact_mut(3).for_each(|px| {
            let q: [f64; 3] = std::array::from_fn(|c| {
                (px[c] as f64 - self.bounds.0[c]) / (self.bounds.1[c] - self.bounds.0[c])
                    * (lut.size - 1) as f64
            });
            let out = lut.sample(q);
            for c in 0..3 {
                px[c] = from_f64(out[c]);
            }
        });
        input
    }
    pub fn gpu_params<'a>(&'a self, p: &RuntimeParams) -> spektrafilm_gpu::ConvertGpuParams<'a> {
        let gain = 2f64.powf(p.film_render.convert.exposure_compensation_ev);
        spektrafilm_gpu::ConvertGpuParams {
            bands: &self.model.bands,
            maximum: self.model.maximum,
            midpoint: self.model.midpoint,
            log_rgb: self.model.log_rgb,
            seed_inverse: self.model.seed_inverse,
            calibration: parse_calibration(&p.film_render.convert.calibration)
                .map(|r| r.map(|v| v * gain)),
            lut: self.lut.as_ref(),
            bounds: self.bounds,
        }
    }
}
