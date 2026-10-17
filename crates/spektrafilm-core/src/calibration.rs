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
    GrainParticleArea,
    DirCouplers,
    GlarePercent,
}
impl Ramp {
    pub fn title(self) -> &'static str {
        match self {
            Self::NegativeExposure => "Negative exposure (EV)",
            Self::PrintExposure => "Print exposure",
            Self::GrainParticleArea => "Grain particle area (µm²)",
            Self::DirCouplers => "DIR inhibition strength",
            Self::GlarePercent => "Glare light (%)",
        }
    }
    fn apply(self, p: &mut RuntimeParams, v: f64) {
        match self {
            Self::NegativeExposure => p.camera.exposure_compensation_ev = v as f32,
            Self::PrintExposure => p.enlarger.print_exposure = v as f32,
            Self::GrainParticleArea => p.film_render.grain.agx_particle_area_um2 = v,
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
