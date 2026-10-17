//! Native entry points for profile measurements and calibration strips.
use anyhow::{Context, Result};
use clap::Subcommand;
use spektrafilm_core::{
    calibration::{CalibrationOptions, Ramp},
    params::RuntimeParams,
    profile,
};
use std::{
    io::Cursor,
    path::{Path, PathBuf},
};
#[derive(Subcommand)]
pub enum Commands {
    /// Measure inverse gamma, local slopes and fitted density floors of a profile.
    MeasureProfile {
        profile: String,
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Write profile / toe-fit SVG figures to this directory.
        #[arg(long)]
        plot_dir: Option<PathBuf>,
        #[arg(long)]
        development_time: Option<f64>,
        #[arg(long, default_value = "0.25")]
        density_low: f64,
        #[arg(long, default_value = "1")]
        density_high: f64,
        #[arg(long, default_value = "0")]
        reference_exposure: f64,
        #[arg(long, default_value = "0.6020599913279624")]
        exposure_window: f64,
    },
    /// Render a labeled exposure, print, grain, DIR or glare calibration strip.
    CalibrationStrip {
        input: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long)]
        film: String,
        #[arg(long)]
        paper: Option<String>,
        #[arg(long,default_value="negative_exposure",value_parser=["negative_exposure","print_exposure","grain_particle_area","dir_couplers","glare_percent"])]
        kind: String,
        #[arg(long, value_delimiter = ',', allow_hyphen_values = true)]
        values: Option<Vec<f64>>,
        #[arg(long, default_value = "0.5,0.5", value_delimiter = ',', num_args = 1)]
        crop_center: Vec<f64>,
        #[arg(long, default_value = "0.2,1", value_delimiter = ',', num_args = 1)]
        crop_size: Vec<f64>,
        #[arg(long)]
        vertical: bool,
        #[arg(long)]
        line_dividers: bool,
        #[arg(long,default_value="0",value_parser=clap::value_parser!(u8).range(0..=3))]
        rotate_quarter_turns: u8,
        #[arg(long)]
        params: Option<PathBuf>,
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
    },
}
pub fn run(command: Commands) -> Result<()> {
    match command {
        Commands::MeasureProfile {
            profile: stock,
            data_dir,
            output,
            plot_dir,
            development_time,
            density_low,
            density_high,
            reference_exposure,
            exposure_window,
        } => {
            let p = profile::resolve_for_render(
                profile::load_profile_by_name(&data_dir, &stock)?,
                development_time,
            );
            let x = p.log_exposure_f64();
            let y = p.density_curves_f64();
            let gamma = spektrafilm_core::measure::measure_gamma(&x, &y, density_low, density_high);
            let slopes = spektrafilm_core::measure::measure_slopes_at_exposure(
                &x,
                &y,
                reference_exposure,
                exposure_window,
            );
            let toe = spektrafilm_core::measure::fit_density_toes(&x, &y, p.is_positive());
            if let Some(directory) = plot_dir {
                spektrafilm_core::qa_report::profile_measurement_plots(
                    &p,
                    toe.as_ref().ok(),
                    &directory,
                )
                .map_err(anyhow::Error::msg)?;
            }
            let report = serde_json::json!({"profile":stock,"development_time":development_time,
                "gamma":gamma.as_ref().ok(),"gamma_error":gamma.err(),"density_targets":[density_low,density_high],
                "slopes":slopes.as_ref().ok(),"slope_error":slopes.err(),
                "reference_log_exposure":reference_exposure,"log_exposure_window":exposure_window,
                "density_min":toe.as_ref().ok().map(|fits|fits.clone().map(|fit|fit.parameters[2])),
                "toe_fits":toe.as_ref().ok(),"toe_error":toe.err(),
                "log_exposure":x,"density_curves":y});
            let text = serde_json::to_string_pretty(&report)?;
            if let Some(output) = output {
                std::fs::write(output, text)?;
            } else {
                println!("{text}");
            }
        }
        Commands::CalibrationStrip {
            input,
            output,
            film,
            paper,
            kind,
            values,
            crop_center,
            crop_size,
            vertical,
            line_dividers,
            rotate_quarter_turns,
            params,
            data_dir,
        } => {
            let source = spektrafilm_io::load(&input, &Default::default())?;
            let mut runtime: RuntimeParams = if let Some(params) = &params {
                serde_json::from_reader(std::fs::File::open(params)?)?
            } else {
                RuntimeParams::default()
            };
            if params.is_none() || spektrafilm_io::is_raw(&input) {
                runtime.io.input_color_space = source.color_space;
                runtime.io.input_cctf_decoding = source.encoded;
            }
            let film = profile::load_profile_by_name(&data_dir, &film)?;
            let paper = paper
                .or_else(|| film.info.target_print.clone())
                .unwrap_or_else(|| "kodak_portra_endura".into());
            let paper = profile::load_profile_by_name(&data_dir, &paper)?;
            let ramp = match kind.as_str() {
                "negative_exposure" => Ramp::NegativeExposure,
                "print_exposure" => Ramp::PrintExposure,
                "grain_particle_area" => Ramp::GrainParticleArea,
                "dir_couplers" => Ramp::DirCouplers,
                _ => Ramp::GlarePercent,
            };
            let defaults = match ramp {
                Ramp::NegativeExposure => vec![-3., -2., -1., 0., 1., 2., 3.],
                Ramp::PrintExposure => vec![0.3, 0.5, 0.7, 1., 1.5, 2., 3.],
                Ramp::GrainParticleArea => vec![0.05, 0.1, 0.2, 0.4, 0.8, 1.6],
                Ramp::DirCouplers => vec![0., 0.5, 1., 1.5, 2.],
                Ramp::GlarePercent => vec![0.02, 0.05, 0.1, 0.2, 0.4],
            };
            let options = CalibrationOptions {
                ramp,
                values: values.unwrap_or(defaults),
                crop_center: crop_center
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("crop-center needs two numbers"))?,
                crop_size: crop_size
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("crop-size needs two numbers"))?,
                vertical,
                rotate_quarter_turns,
            };
            let backend = spektrafilm_gpu::select_backend();
            let strip = spektrafilm_core::calibration::render(
                &source.pixels,
                &film,
                &paper,
                &runtime,
                &options,
                &data_dir,
                backend.as_ref(),
            )
            .map_err(anyhow::Error::msg)?;
            let save = spektrafilm_io::SaveOptions {
                color_space: runtime.io.output_color_space.clone(),
                encoded: runtime.io.output_cctf_encoding,
                ..Default::default()
            };
            spektrafilm_io::save(&strip.image, &output, &save, &source.metadata, &data_dir)?;
            std::fs::write(
                output.with_extension("calibration.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"title":strip.title,"labels":strip.labels,"options":options,"params":runtime}),
                )?,
            )?;
            write_strip_svg(
                &strip,
                &runtime,
                &output.with_extension("svg"),
                line_dividers,
            )?;
            println!("{}", output.display());
        }
    }
    Ok(())
}
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
fn write_strip_svg(
    strip: &spektrafilm_core::calibration::CalibrationStrip,
    params: &RuntimeParams,
    path: &Path,
    lines: bool,
) -> Result<()> {
    use base64::Engine;
    let thumbnail = spektrafilm_core::resize::preview(&strip.image, 2048);
    let display = spektrafilm_io::convert_color(
        &thumbnail,
        &params.io.output_color_space,
        params.io.output_cctf_encoding,
        "sRGB",
        true,
    )?;
    let bytes: Vec<u8> = display
        .data
        .iter()
        .map(|v| (((*v as f64).clamp(0., 1.) * 255.).round()) as u8)
        .collect();
    let raster = image::RgbImage::from_raw(display.width, display.height, bytes)
        .context("invalid strip image shape")?;
    let mut png = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(raster).write_to(&mut png, image::ImageFormat::Png)?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(png.into_inner());
    let scale = (640. / display.width.max(display.height) as f64).clamp(1., 8.);
    let (w, h) = (display.width as f64 * scale, display.height as f64 * scale);
    let margin = if strip.vertical { 110. } else { 20. };
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\"><rect width=\"100%\" height=\"100%\" fill=\"white\"/><text x=\"{margin}\" y=\"24\" font-family=\"sans-serif\" font-size=\"16\">{}</text><image x=\"{margin}\" y=\"55\" width=\"{w}\" height=\"{h}\" href=\"data:image/png;base64,{encoded}\"/>",
        (w + margin + 20.).max(360.),
        h + 75.,
        (w + margin + 20.).max(360.),
        h + 75.,
        escape(&strip.title)
    );
    for (i, label) in strip.labels.iter().enumerate() {
        let f = (i as f64 + 0.5) / strip.labels.len() as f64;
        let (x, y) = if strip.vertical {
            (margin - 8., 55. + h * f)
        } else {
            (margin + w * f, 47.)
        };
        svg.push_str(&format!("<text x=\"{x}\" y=\"{y}\" text-anchor=\"{}\" font-family=\"sans-serif\" font-size=\"12\">{}</text>",if strip.vertical {"end"} else {"middle"},escape(label)));
        if lines && i > 0 {
            let f = i as f64 / strip.labels.len() as f64;
            let (x1, y1, x2, y2) = if strip.vertical {
                (margin, 55. + h * f, margin + w, 55. + h * f)
            } else {
                (margin + w * f, 55., margin + w * f, 55. + h)
            };
            svg.push_str(&format!(
                "<path d=\"M {x1} {y1} L {x2} {y2}\" stroke=\"white\" stroke-width=\".5\"/>"
            ));
        }
    }
    svg.push_str("</svg>");
    std::fs::write(path, svg)?;
    Ok(())
}
