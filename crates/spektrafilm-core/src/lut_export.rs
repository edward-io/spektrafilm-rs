//! Deterministic, calibrated LUT baking. Transport encodings live outside the runtime.
use crate::{params::RuntimeParams, pipeline::Pipeline, profile::load_profile_by_name};
use spektrafilm_math::{image::ImageBuf, precision::from_f64, transfer::Transfer};
use std::{io::Write, path::Path};

pub struct LutSpace {
    pub name: &'static str,
    pub primaries: &'static str,
    pub tag: &'static str,
    pub input_role: bool,
    pub output_role: bool,
    pub curve: Transfer,
    pub auto_gain: f64,
    pub midgray_linear: f64,
    pub output_gain: f64,
}
impl LutSpace {
    pub fn decode_pixel(&self, p: [f64; 3]) -> [f64; 3] {
        if matches!(self.curve, Transfer::Hlg) {
            let scene = p.map(|v| {
                if v <= 0.5 {
                    v * v / 3.
                } else {
                    (((v - 0.55991073) / 0.17883277).exp() + 0.28466892) / 12.
                }
            });
            let y = 0.2627 * scene[0] + 0.6780 * scene[1] + 0.0593 * scene[2];
            scene.map(|v| 1000. * v * y.abs().powf(0.2))
        } else {
            p.map(|v| self.curve.decode(v))
        }
    }
    pub fn encode_pixel(&self, p: [f64; 3]) -> [f64; 3] {
        if matches!(self.curve, Transfer::Hlg) {
            let y = (0.2627 * p[0] + 0.6780 * p[1] + 0.0593 * p[2]) / 1000.;
            let scale = if y == 0. {
                0.
            } else {
                y.abs().powf(-0.2 / 1.2)
            };
            p.map(|v| {
                let e = v / 1000. * scale;
                if e <= 1. / 12. {
                    (3. * e).max(0.).sqrt()
                } else {
                    0.17883277 * (12. * e - 0.28466892).ln() + 0.55991073
                }
            })
        } else {
            p.map(|v| self.curve.encode(v))
        }
    }
}
pub fn decode_grid(image: &mut ImageBuf, input: &LutSpace, gain: f64) {
    for p in image.data.chunks_exact_mut(3) {
        let v = input.decode_pixel([p[0] as f64, p[1] as f64, p[2] as f64]);
        for c in 0..3 {
            p[c] = from_f64((v[c] * gain) as f32 as f64);
        }
    }
}
pub fn encode_grid(image: &mut ImageBuf, output: &LutSpace) {
    for p in image.data.chunks_exact_mut(3) {
        let v = output.encode_pixel([
            p[0] as f64 * output.output_gain,
            p[1] as f64 * output.output_gain,
            p[2] as f64 * output.output_gain,
        ]);
        for c in 0..3 {
            p[c] = from_f64(v[c].clamp(0., 1.));
        }
    }
}

pub fn space(name: &str) -> Result<&'static LutSpace, String> {
    crate::lut_spaces::SPACES
        .iter()
        .find(|s| s.name == name || s.tag == name)
        .ok_or_else(|| format!("unknown LUT color space: {name}"))
}
pub fn spaces() -> impl Iterator<Item = &'static LutSpace> {
    crate::lut_spaces::SPACES.iter()
}

pub struct BakeOptions<'a> {
    pub film: &'a str,
    pub paper: &'a str,
    pub size: u32,
    pub input: &'a str,
    pub output: &'a str,
    pub gain: Option<f64>,
    pub params: RuntimeParams,
}

pub fn grid(size: u32) -> ImageBuf {
    let mut data = Vec::with_capacity(size.pow(3) as usize * 3);
    for b in 0..size {
        for g in 0..size {
            for r in 0..size {
                for v in [r, g, b] {
                    data.push(from_f64(v as f64 / (size - 1) as f64));
                }
            }
        }
    }
    ImageBuf::from_data(size, size * size, data)
}

pub fn bake(options: &BakeOptions<'_>, data_dir: &Path) -> Result<ImageBuf, String> {
    if !(2..=129).contains(&options.size) {
        return Err("LUT size must be in 2..=129".into());
    }
    reference_image(options, data_dir, grid(options.size))
}

pub fn reference_image(
    options: &BakeOptions<'_>,
    data_dir: &Path,
    mut image: ImageBuf,
) -> Result<ImageBuf, String> {
    let input = space(options.input)?;
    let output = space(options.output)?;
    if !input.input_role {
        return Err(format!(
            "{} is not enabled as a LUT input in the reference registry",
            input.name
        ));
    }
    if !output.output_role {
        return Err(format!(
            "{} is not enabled as a LUT output in the reference registry",
            output.name
        ));
    }
    if options.gain.is_some_and(|g| !g.is_finite() || g <= 0.) {
        return Err("LUT input gain must be finite and positive".into());
    }
    let mut params = options.params.clone();
    params.debug.lut_mode = true;
    params.io.input_color_space = input.primaries.into();
    params.io.output_color_space = output.primaries.into();
    params.io.input_cctf_decoding = false;
    params.io.output_cctf_encoding = false;
    params.io.crop = false;
    params.io.upscale_factor = 1.;
    params.taps = Default::default();
    let film = load_profile_by_name(data_dir, options.film).map_err(|e| e.to_string())?;
    let paper = load_profile_by_name(data_dir, options.paper).map_err(|e| e.to_string())?;
    let pipeline = Pipeline::new_with_spectral(film, paper, params, data_dir)?;
    let gain = options.gain.unwrap_or(input.auto_gain);
    decode_grid(&mut image, input, gain);
    // CPU avoids GPU approximation and gives export the build's Scalar precision.
    let mut out = pipeline.process(image, &spektrafilm_gpu::cpu_backend::CpuBackend);
    encode_grid(&mut out, output);
    Ok(out)
}

pub fn write_lut(
    image: &ImageBuf,
    size: u32,
    path: &Path,
    format: &str,
    title: &str,
    input: &str,
) -> Result<(), String> {
    if image.pixel_count() != size.pow(3) as usize {
        return Err("LUT data length does not match resolution".into());
    }
    if format == "hald_png" {
        let level = (size as f64).sqrt() as u32;
        if level * level != size {
            return Err("Hald PNG requires a square cube resolution (4, 9, 16, 25, 36, 49, 64, 81, 100, 121)".into());
        }
        let side = level.pow(3);
        let data: Vec<u8> = image
            .data
            .iter()
            .map(|v| ((*v as f64).clamp(0., 1.) * 255.).round_ties_even() as u8)
            .collect();
        return image::save_buffer_with_format(
            path,
            &data,
            side,
            side,
            image::ColorType::Rgb8,
            image::ImageFormat::Png,
        )
        .map_err(|e| e.to_string());
    }
    let mut file = std::io::BufWriter::new(std::fs::File::create(path).map_err(|e| e.to_string())?);
    let result = (|| -> std::io::Result<()> {
        if format == "3dl" {
            writeln!(file, "# {title}")?;
            for i in 0..size {
                write!(
                    file,
                    "{}{}",
                    (i as f64 / (size - 1) as f64 * 1023.).round_ties_even() as u32,
                    if i + 1 == size { "\n" } else { " " }
                )?;
            }
            for p in image.pixels() {
                writeln!(
                    file,
                    "{} {} {}",
                    (p[0] as f64 * 1023.).round_ties_even().clamp(0., 1023.) as u32,
                    (p[1] as f64 * 1023.).round_ties_even().clamp(0., 1023.) as u32,
                    (p[2] as f64 * 1023.).round_ties_even().clamp(0., 1023.) as u32
                )?;
            }
        } else if format == "cube" || format == "lumix" {
            writeln!(file, "TITLE \"{}\"", title.replace(['\n', '\r', '"'], " "))?;
            if format == "lumix" {
                if ![17, 33, 65].contains(&size) {
                    return Err(std::io::Error::other(
                        "Lumix resolution must be 17, 33 or 65",
                    ));
                }
                let tag = match space(input).map(|s| s.tag) {
                    Ok("vlog") => "VLOG",
                    _ => {
                        return Err(std::io::Error::other("Lumix input must be Panasonic V-Log"));
                    }
                };
                writeln!(file, "#LUMIXPHOTOSTYLE {tag}")?;
            }
            writeln!(
                file,
                "LUT_3D_SIZE {size}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n"
            )?;
            for p in image.pixels() {
                if format == "lumix" {
                    writeln!(file, "{:.6} {:.6} {:.6}", p[0], p[1], p[2])?;
                } else {
                    writeln!(file, "{:.9} {:.9} {:.9}", p[0], p[1], p[2])?;
                }
            }
        } else {
            return Err(std::io::Error::other(format!(
                "unknown LUT format: {format}"
            )));
        }
        file.flush()
    })();
    result.map_err(|e| e.to_string())
}
