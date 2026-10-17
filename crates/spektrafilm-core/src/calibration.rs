//! Isolated-parameter calibration strips for exposure, grain, DIR and glare.
use crate::{params::RuntimeParams, pipeline::Pipeline, profile::Profile};
use serde::{Deserialize, Serialize};
use spektrafilm_gpu::ComputeBackend;
use spektrafilm_math::image::ImageBuf;
use std::path::Path;
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ramp {
    NegativeExposure,
    PrintExposure,
    #[serde(alias = "grain_particle_area")]
    GrainRmsGranularity,
    DirCouplers,
    GlarePercent,
}
impl Ramp {
    pub fn title(self) -> &'static str {
        match self {
            Self::NegativeExposure => "Negative exposure (EV)",
            Self::PrintExposure => "Print exposure",
            Self::GrainRmsGranularity => "Grain RMS granularity (48 µm aperture)",
            Self::DirCouplers => "DIR inhibition strength",
            Self::GlarePercent => "Glare light (%)",
        }
    }
    fn apply(self, p: &mut RuntimeParams, v: f64) {
        match self {
            Self::NegativeExposure => p.camera.exposure_compensation_ev = v as f32,
            Self::PrintExposure => p.enlarger.print_exposure = v as f32,
            Self::GrainRmsGranularity => p.film_render.grain.rms_granularity = [v; 3],
            Self::DirCouplers => p.film_render.dir_couplers.amount = v,
            Self::GlarePercent => p.print_render.glare.percent = v as f32,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CalibrationOptions {
    pub ramp: Ramp,
    pub values: Vec<f64>,
    pub crop_center: [f64; 2],
    pub crop_size: [f64; 2],
    pub vertical: bool,
    /// Clockwise rotation; 3 reproduces the Python helper's rotate=True.
    pub rotate_quarter_turns: u8,
}
impl Default for CalibrationOptions {
    fn default() -> Self {
        Self {
            ramp: Ramp::NegativeExposure,
            values: vec![-3., -2., -1., 0., 1., 2., 3.],
            crop_center: [0.5; 2],
            crop_size: [0.2, 1.],
            vertical: false,
            rotate_quarter_turns: 0,
        }
    }
}
pub struct CalibrationStrip {
    pub image: ImageBuf,
    pub labels: Vec<String>,
    pub title: String,
    pub vertical: bool,
}
/// Each condition owns its runtime settings. The Python draft's shallow copy
/// can accidentally share nested camera/enlarger settings between conditions.
pub fn render(
    image: &ImageBuf,
    film: &Profile,
    paper: &Profile,
    base: &RuntimeParams,
    options: &CalibrationOptions,
    data: &Path,
    backend: &dyn ComputeBackend,
) -> Result<CalibrationStrip, String> {
    if options.values.is_empty()
        || options.values.len() > 128
        || options.values.iter().any(|v| !v.is_finite())
        || options
            .crop_center
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || options
            .crop_size
            .iter()
            .any(|v| !v.is_finite() || *v <= 0. || *v > 1.)
    {
        return Err("calibration needs 1–128 finite values and a valid normalized crop".into());
    }
    let mut base = base.clone();
    if base.settings.apply_stock_specifics {
        crate::pipeline::apply_stock_defaults(film, &mut base);
    }
    let rotated = image.rotated_quarter_turns(options.rotate_quarter_turns);
    let mut sections = Vec::new();
    let mut labels = Vec::new();
    for &value in &options.values {
        let mut params = base.clone();
        params.io.crop = true;
        params.io.crop_center = options.crop_center.map(|v| v as f32);
        params.io.crop_size = options.crop_size.map(|v| v as f32);
        options.ramp.apply(&mut params, value);
        let pipeline = Pipeline::new_with_spectral(film.clone(), paper.clone(), params, data)?;
        sections.push(pipeline.process(rotated.clone(), backend));
        labels.push(match options.ramp {
            Ramp::NegativeExposure => format!("{value:.0}"),
            _ => format!("{value:.2}"),
        });
    }
    let (w, h) = (sections[0].width, sections[0].height);
    if sections
        .iter()
        .any(|section| section.width != w || section.height != h)
    {
        return Err("calibration conditions produced different image sizes".into());
    }
    let count = sections.len() as u32;
    let width = w
        .checked_mul(if options.vertical { 1 } else { count })
        .ok_or("calibration width overflow")?;
    let height = h
        .checked_mul(if options.vertical { count } else { 1 })
        .ok_or("calibration height overflow")?;
    let mut out = ImageBuf::new(width, height);
    for (i, section) in sections.iter().enumerate() {
        for y in 0..h as usize {
            let dest = if options.vertical {
                (i * h as usize + y) * width as usize * 3
            } else {
                (y * width as usize + i * w as usize) * 3
            };
            out.data[dest..dest + w as usize * 3]
                .copy_from_slice(&section.data[y * w as usize * 3..(y + 1) * w as usize * 3]);
        }
    }
    Ok(CalibrationStrip {
        image: out,
        labels,
        title: options.ramp.title().into(),
        vertical: options.vertical,
    })
}

/// Small damped least-squares solve for the two printer shifts or device/base fit.
fn least_squares(
    mut x: Vec<f64>,
    residual: impl Fn(&[f64]) -> Vec<f64>,
    max_evaluations: usize,
) -> Result<Vec<f64>, String> {
    fn cost(r: &[f64]) -> f64 {
        r.iter().map(|v| v * v).sum()
    }
    let mut r = residual(&x);
    let mut lambda = 1e-3;
    let mut evaluations = 1;
    if r.is_empty() || r.iter().any(|v| !v.is_finite()) {
        return Err("calibration has no finite observations".into());
    }
    while evaluations + x.len() + 1 <= max_evaluations {
        let n = x.len();
        let mut jac = vec![vec![0.; r.len()]; n];
        for j in 0..n {
            let step = 1e-4 * (1. + x[j].abs());
            let mut next = x.clone();
            next[j] += step;
            let perturbed = residual(&next);
            evaluations += 1;
            if perturbed.len() != r.len() {
                return Err("calibration observation count changed".into());
            }
            for i in 0..r.len() {
                jac[j][i] = (perturbed[i] - r[i]) / step;
            }
        }
        let mut system = vec![vec![0.; n + 1]; n];
        for i in 0..n {
            for j in 0..n {
                system[i][j] = jac[i].iter().zip(&jac[j]).map(|(a, b)| a * b).sum();
            }
            system[i][i] += lambda;
            system[i][n] = jac[i].iter().zip(&r).map(|(a, b)| a * b).sum();
        }
        for col in 0..n {
            let pivot = (col..n)
                .max_by(|a, b| system[*a][col].abs().total_cmp(&system[*b][col].abs()))
                .unwrap();
            system.swap(col, pivot);
            let divisor = system[col][col];
            if divisor.abs() < 1e-20 {
                return Err("calibration is singular".into());
            }
            for j in col..=n {
                system[col][j] /= divisor;
            }
            for i in 0..n {
                if i != col {
                    let v = system[i][col];
                    for j in col..=n {
                        system[i][j] -= v * system[col][j];
                    }
                }
            }
        }
        let next: Vec<_> = (0..n).map(|j| x[j] - system[j][n]).collect();
        let candidate = residual(&next);
        evaluations += 1;
        if candidate.iter().all(|v| v.is_finite()) && cost(&candidate) < cost(&r) {
            let improvement = cost(&r) - cost(&candidate);
            x = next;
            r = candidate;
            lambda = (lambda * 0.3).max(1e-10);
            if improvement < 1e-12 {
                break;
            }
        } else {
            lambda *= 10.;
            if lambda > 1e12 {
                break;
            }
        }
    }
    Ok(x)
}

fn linear_pixels(image: &ImageBuf, p: &RuntimeParams, limit: usize) -> Vec<[f64; 3]> {
    let transfer = spektrafilm_math::transfer::get(&p.io.input_color_space)
        .expect("validated input")
        .transfer;
    let finite = |px: &&[spektrafilm_math::precision::Scalar]| px.iter().all(|v| v.is_finite());
    let total = image.pixels().filter(finite).count();
    let count = total.min(limit);
    let mut samples = Vec::with_capacity(count);
    let mut target = 0;
    for (i, px) in image.pixels().filter(finite).enumerate() {
        if samples.len() == count {
            break;
        }
        if i != target {
            continue;
        }
        samples.push(std::array::from_fn(|c| {
            if p.io.input_cctf_decoding {
                transfer.decode(px[c] as f64)
            } else {
                px[c] as f64
            }
        }));
        target = samples.len() * (total - 1) / (count - 1).max(1);
    }
    samples
}
fn brightest(pixels: &[[f64; 3]], percentile: f64, weights: [f64; 3]) -> Result<[f64; 3], String> {
    if pixels.is_empty() {
        return Err("image has no finite pixels".into());
    }
    let luminance = |p: &[f64; 3]| (0..3).map(|c| p[c] * weights[c]).sum::<f64>();
    let mut values: Vec<_> = pixels.iter().map(luminance).collect();
    values.sort_by(f64::total_cmp);
    let q = percentile.clamp(0., 100.) / 100. * (values.len() - 1) as f64;
    let i = q.floor() as usize;
    let threshold =
        values[i] + (q - i as f64) * (values[(i + 1).min(values.len() - 1)] - values[i]);
    let selected: Vec<_> = pixels
        .iter()
        .filter(|p| luminance(p) >= threshold)
        .collect();
    Ok(std::array::from_fn(|c| {
        selected.iter().map(|p| p[c]).sum::<f64>() / selected.len() as f64
    }))
}

/// Estimate the clear film base and exposure from the brightest linear pixels.
pub fn detect_base(
    image: &ImageBuf,
    film: Profile,
    p: &RuntimeParams,
) -> Result<(crate::params::BaseDensityParams, f64), String> {
    p.validate()?;
    let target = brightest(
        &linear_pixels(image, p, image.pixel_count()),
        p.film_render.convert.base_percentile,
        [0.2126, 0.7152, 0.0722],
    )?;
    let mut untuned = p.clone();
    untuned.film_render.base.active = false;
    let film = crate::stock::prepare(film, &untuned, true)?;
    let dyes: Vec<_> = film
        .data
        .channel_density
        .iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    let illum = spektrafilm_math::illuminants::get(&p.film_render.convert.scan_illuminant)?;
    let forward = |x: &[f64]| {
        let mut base = p.film_render.base.clone();
        base.active = true;
        base.cyan = x[0];
        base.magenta = x[1];
        base.yellow = x[2];
        let spectrum = crate::stock::tune_base(&film.data.base_density, &base, true);
        spektrafilm_model::convert::FilmConverter::new(
            &dyes,
            &spectrum,
            illum,
            &p.io.input_color_space,
            [1.; 3],
        )
        .forward_jac([0.; 3])
        .0
    };
    let gain = |f: [f64; 3]| {
        (0..3).map(|c| f[c] * target[c]).sum::<f64>()
            / target.iter().map(|v| v * v).sum::<f64>().max(1e-30)
    };
    let residual = |x: &[f64]| {
        let f = forward(x);
        let g = gain(f);
        let mut r: Vec<_> = (0..3).map(|c| f[c] - g * target[c]).collect();
        r.extend(x.iter().map(|v| 0.02 * (v - 1.)));
        r
    };
    let x = least_squares(
        vec![
            p.film_render.base.cyan,
            p.film_render.base.magenta,
            p.film_render.base.yellow,
        ],
        residual,
        200,
    )?;
    let mut base = p.film_render.base.clone();
    base.active = true;
    base.cyan = x[0];
    base.magenta = x[1];
    base.yellow = x[2];
    if x.iter().any(|v| *v < 0.) {
        return Err(
            "base fit requires negative density factors; choose a brighter clear-film sample"
                .into(),
        );
    }
    Ok((base, gain(forward(&x)).max(1e-9).log2()))
}

/// Fit a device matrix to the film dye gamut, anchored by the observed clear base.
pub fn blind_calibration(
    image: &ImageBuf,
    film: Profile,
    p: &RuntimeParams,
) -> Result<String, String> {
    use rayon::prelude::*;
    p.validate()?;
    let film = crate::stock::prepare(film, p, true)?;
    let model = crate::conversion::Conversion::new(&film, &{
        let mut q = p.clone();
        q.settings.use_convert_lut = false;
        q
    })?
    .model;
    let samples: Vec<_> = linear_pixels(image, p, 1500)
        .into_iter()
        .map(|row| row.map(|v| v.max(0.)))
        .collect();
    if samples.is_empty() {
        return Err("image has no finite pixels".into());
    }
    let base = brightest(&samples, 99., [0.2, 0.7, 0.1])?;
    let truth = model.forward_jac([0.; 3]).0;
    let gain = 2f64.powf(p.film_render.convert.exposure_compensation_ev);
    let identity = vec![1., 0., 0., 0., 1., 0., 0., 0., 1.];
    let residual = |x: &[f64]| {
        let transform = |v: [f64; 3]| {
            std::array::from_fn::<_, 3, _>(|c| {
                (0..3).map(|k| v[k] * x[k * 3 + c]).sum::<f64>() * gain
            })
        };
        let rows: Vec<[f64; 3]> = samples
            .par_iter()
            .map(|v| {
                let target = transform(*v);
                let density = model.invert(target.map(|v| v.max(0.)));
                let fitted = model.forward_jac(density).0;
                std::array::from_fn(|c| target[c] - fitted[c])
            })
            .collect();
        let mut result: Vec<_> = rows.into_iter().flatten().collect();
        let observed = transform(base);
        result.extend((0..3).map(|c| 5. * (observed[c] - truth[c])));
        result.extend(x.iter().zip(&identity).map(|(a, b)| 0.03 * (a - b)));
        result
    };
    let result = least_squares(identity.clone(), residual, 300)?;
    Ok(result
        .iter()
        .map(|v| format!("{v:.5}"))
        .collect::<Vec<_>>()
        .join(" "))
}

/// Find M/Y enlarger shifts that print the current stock's midgray neutral.
pub fn neutral_print_balance(
    film: Profile,
    paper: Profile,
    p: &RuntimeParams,
    data_dir: &Path,
) -> Result<[f32; 2], String> {
    let mut q = p.clone();
    q.workflow.route = "input > film > print > scan".into();
    q.io.scan_film = false;
    q.settings.use_enlarger_lut = false;
    q.settings.use_scanner_lut = false;
    q.film_render.grain.active = false;
    q.print_render.glare.active = false;
    q.io.output_cctf_encoding = false;
    q.io.output_gamut_compress.algorithm = "off".into();
    q.io.output_gamut_compress.lightness_compression = None;
    q.io.input_cctf_decoding = false;
    q.io.crop = false;
    q.io.upscale_factor = 1.;
    let gray = ImageBuf::from_data(2, 2, vec![spektrafilm_math::precision::from_f64(0.184); 12]);
    // Validate once before the optimizer's numerical probes.
    Pipeline::new_with_spectral(film.clone(), paper.clone(), q.clone(), data_dir)?;
    let residual = |x: &[f64]| {
        let mut trial = q.clone();
        trial.enlarger.m_filter_shift = x[0] as f32;
        trial.enlarger.y_filter_shift = x[1] as f32;
        let pipeline = Pipeline::new_with_spectral(film.clone(), paper.clone(), trial, data_dir)
            .expect("validated calibration params");
        let output = pipeline.process(gray.clone(), &spektrafilm_gpu::cpu_backend::CpuBackend);
        let rgb: [f64; 3] =
            std::array::from_fn(|c| output.pixels().map(|px| px[c] as f64).sum::<f64>() / 4.);
        let green = rgb[1].max(1e-6);
        vec![(rgb[0] - rgb[1]) / green, (rgb[2] - rgb[1]) / green]
    };
    let result = least_squares(
        vec![
            q.enlarger.m_filter_shift as f64,
            q.enlarger.y_filter_shift as f64,
        ],
        residual,
        100,
    )?;
    Ok([result[0] as f32, result[1] as f32])
}

#[cfg(test)]
mod sampling_tests {
    #[test]
    fn calibration_decodes_only_evenly_spaced_finite_samples() {
        use super::*;
        let image = ImageBuf::from_data(
            2001,
            1,
            (0..2001)
                .flat_map(|i| [spektrafilm_math::precision::from_f64(i as f64); 3])
                .collect(),
        );
        let mut p = RuntimeParams::default();
        p.io.input_cctf_decoding = false;
        let actual = linear_pixels(&image, &p, 7);
        assert_eq!(
            actual.iter().map(|v| v[0] as usize).collect::<Vec<_>>(),
            vec![0, 333, 666, 1000, 1333, 1666, 2000]
        );
    }
}
