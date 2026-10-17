mod bundle_spec;
mod diagnostics_cli;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use spektrafilm_core::params::RuntimeParams;
use spektrafilm_core::pipeline::Pipeline;
use spektrafilm_core::profile;

#[derive(Parser)]
#[command(
    name = "spektrafilm",
    about = "Physically-based spectral film emulation"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(flatten)]
    Diagnostics(diagnostics_cli::Commands),
    /// Build LUT bundles from a Python-compatible TOML or JSON BundleSpec.
    BuildLuts {
        #[arg(long = "from")]
        spec: Option<PathBuf>,
        #[command(flatten)]
        overrides: bundle_spec::Overrides,
        #[arg(short, long, alias = "out")]
        output: PathBuf,
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
    },
    /// List supported LUT input/output transports and their aliases.
    ListColorSpaces,

    /// Process an image through the film simulation pipeline.
    Process {
        /// Input image (TIFF, EXR, or PNG).
        input: PathBuf,
        /// Clockwise rotation applied after decoding, before crop / processing.
        #[arg(long, default_value="0", value_parser=clap::value_parser!(u8).range(0..=3))]
        rotate_quarter_turns: u8,
        /// Output image path.
        #[arg(short, long)]
        output: PathBuf,
        /// Film stock name (e.g. kodak_portra_400).
        #[arg(long)]
        film: String,
        /// Paper stock name (e.g. fujifilm_crystal_archive_typeii).
        /// If omitted and --scan-film is not set, uses the film's target_print.
        #[arg(long)]
        paper: Option<String>,
        /// Scan film directly (skip printing stage).
        #[arg(long)]
        scan_film: bool,
        /// Print per-stage timing information.
        #[arg(long)]
        timings: bool,
        /// Path to JSON params file for overrides.
        #[arg(long)]
        params: Option<PathBuf>,
        /// Dump the raw f64 output buffer (HxWx3, row-major, channel-interleaved)
        /// before sRGB encoding/clipping. Used for bit-exact parity comparison.
        #[arg(long)]
        raw_out: Option<PathBuf>,
        /// Run the pipeline N times in the same process. Each iteration is
        /// timed individually so you can see cold-start vs warm-cache speed.
        /// Default: 1 (no repetition).
        #[arg(long, default_value = "1")]
        iters: usize,
        /// Path to the data directory.
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        #[arg(long, default_value="16", value_parser=clap::value_parser!(u8).range(8..=32))]
        bit_depth: u8,
        #[arg(long,default_value="auto",value_parser=["auto","libraw","rawler"])]
        raw_decoder: String,
        #[arg(long)]
        lens_correction: bool,
        #[arg(long)]
        lens_database: Option<PathBuf>,
        #[arg(long, default_value="95", value_parser=clap::value_parser!(u8).range(1..=100))]
        jpeg_quality: u8,
        /// Convert the rendered output into this space before writing.
        #[arg(long)]
        saving_color_space: Option<String>,
        #[arg(long, num_args=0..=1, default_missing_value="true", action=clap::ArgAction::Set)]
        saving_cctf_encoding: Option<bool>,
        #[arg(long, default_value="as_shot", value_parser=["camera","as_shot","daylight","tungsten","custom"])]
        white_balance: String,
        #[arg(long, default_value = "5500")]
        temperature: f64,
        #[arg(long, default_value = "1", allow_hyphen_values = true)]
        tint: f64,
    },
    /// List available film and paper profiles.
    ListProfiles {
        /// Path to the data directory.
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
    },
    /// Export a 3D CUBE LUT for use in other software.
    ExportLut {
        /// Film stock name.
        #[arg(long)]
        film: String,
        /// Paper stock name.
        #[arg(long)]
        paper: Option<String>,
        /// LUT cube size (e.g. 33, 65).
        #[arg(long, default_value = "33")]
        size: u32,
        /// Output .cube file path.
        #[arg(short, long)]
        output: PathBuf,
        /// Path to the data directory.
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        #[arg(long, value_parser=["1lut","2lut","3lut","4lut"])]
        topology: Option<String>,
        #[arg(long)]
        include_combinations: bool,
        /// Include scientific QA diagnostics (requires --topology).
        #[arg(long)]
        qa: bool,
        /// Parameter overrides (LUT mode disables image-dependent effects).
        #[arg(long)]
        params: Option<PathBuf>,
        #[arg(long, default_value = "sRGB")]
        input_space: String,
        #[arg(long, default_value = "sRGB")]
        output_space: String,
        #[arg(long, default_value="cube", value_parser=["cube","3dl","lumix","hald_png"])]
        format: String,
        /// Source white exposure in stops above 0.18. Default follows the input registry.
        #[arg(long)]
        stops_above_midgray: Option<f64>,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Diagnostics(command) => diagnostics_cli::run(command)?,
        Commands::BuildLuts {
            spec,
            overrides,
            output,
            data_dir,
        } => {
            let path = bundle_spec::build(spec.as_deref(), &output, &data_dir, overrides)?;
            println!("{}", path.display());
        }
        Commands::ListColorSpaces => {
            for entry in spektrafilm_core::lut_export::spaces() {
                println!(
                    "{:<22} {} ({}, input={}, output={})",
                    entry.tag, entry.name, entry.primaries, entry.input_role, entry.output_role
                );
            }
        }

        Commands::Process {
            input,
            rotate_quarter_turns,
            output,
            film,
            paper,
            scan_film,
            timings,
            params: params_file,
            raw_out,
            iters,
            data_dir,
            bit_depth,
            jpeg_quality,
            saving_color_space,
            saving_cctf_encoding,
            white_balance,
            temperature,
            tint,
            raw_decoder,
            lens_correction,
            lens_database,
        } => {
            cmd_process(
                &input,
                &output,
                &film,
                paper.as_deref(),
                scan_film,
                timings,
                params_file.as_deref(),
                raw_out.as_deref(),
                iters,
                &data_dir,
                bit_depth,
                jpeg_quality,
                saving_color_space.as_deref(),
                saving_cctf_encoding,
                &white_balance,
                temperature,
                tint,
                &raw_decoder,
                lens_correction,
                lens_database.as_deref(),
                rotate_quarter_turns,
            )?;
        }
        Commands::ListProfiles { data_dir } => {
            cmd_list_profiles(&data_dir);
        }
        Commands::ExportLut {
            film,
            paper,
            size,
            output,
            data_dir,
            params,
            input_space,
            output_space,
            format,
            stops_above_midgray,
            topology,
            include_combinations,
            qa,
        } => {
            cmd_export_lut(
                &film,
                paper.as_deref(),
                size,
                &output,
                &data_dir,
                params.as_deref(),
                &input_space,
                &output_space,
                &format,
                stops_above_midgray,
                topology.as_deref(),
                include_combinations,
                qa,
            )?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_process(
    input: &Path,
    output: &Path,
    film_name: &str,
    paper_name: Option<&str>,
    scan_film: bool,
    show_timings: bool,
    params_file: Option<&Path>,
    raw_out: Option<&Path>,
    iters: usize,
    data_dir: &Path,
    bit_depth: u8,
    jpeg_quality: u8,
    saving_color_space: Option<&str>,
    saving_cctf_encoding: Option<bool>,
    white_balance: &str,
    temperature: f64,
    tint: f64,
    raw_decoder: &str,
    lens_correction: bool,
    lens_database: Option<&Path>,
    rotate_quarter_turns: u8,
) -> Result<()> {
    let total_start = Instant::now();

    // Load params (defaults + optional overrides)
    let mut params = if let Some(pf) = params_file {
        let f = std::fs::File::open(pf)
            .with_context(|| format!("opening params file: {}", pf.display()))?;
        serde_json::from_reader(std::io::BufReader::new(f))
            .with_context(|| "parsing params file")?
    } else {
        RuntimeParams::default()
    };
    params.io.scan_film |= scan_film;
    let scan_film = params.io.scan_film;

    // Load profiles
    let t = Instant::now();
    let film = profile::load_profile_by_name(data_dir, film_name)
        .with_context(|| format!("loading film profile: {film_name}"))?;

    let print_stock = if scan_film {
        film_name.to_string()
    } else if let Some(p) = paper_name {
        p.to_string()
    } else if let Some(ref target) = film.info.target_print {
        target.clone()
    } else {
        bail!("no paper specified and film has no target_print — use --paper or --scan-film");
    };

    let print = profile::load_profile_by_name(data_dir, &print_stock)
        .with_context(|| format!("loading print profile: {print_stock}"))?;
    eprintln!("Profiles loaded: {} ms", t.elapsed().as_millis());

    let t = Instant::now();
    let loaded = spektrafilm_io::load(
        input,
        &spektrafilm_io::RawOptions {
            decoder: raw_decoder.into(),
            white_balance: white_balance.into(),
            temperature,
            tint,
            lens_correction,
            lens_database: lens_database.map(Path::to_path_buf),
        },
    )?;
    if params_file.is_none() || spektrafilm_io::is_raw(input) {
        params.io.input_color_space = loaded.color_space;
        params.io.input_cctf_decoding = loaded.encoded;
    }
    let save_options = spektrafilm_io::SaveOptions {
        bit_depth,
        jpeg_quality,
        color_space: saving_color_space
            .unwrap_or(&params.io.output_color_space)
            .into(),
        encoded: saving_cctf_encoding.unwrap_or(params.io.output_cctf_encoding),
    };
    let image = if rotate_quarter_turns == 0 {
        loaded.pixels
    } else {
        loaded.pixels.rotated_quarter_turns(rotate_quarter_turns)
    };
    eprintln!(
        "Image loaded: {}x{} in {} ms",
        image.width,
        image.height,
        t.elapsed().as_millis()
    );

    // Select backend
    let backend = spektrafilm_gpu::select_backend();
    eprintln!("Backend: {}", backend.name());

    // Run pipeline (with full Hanatos2025 spectral upsampling)
    let t = Instant::now();
    let pipeline =
        Pipeline::new_with_spectral(film, print, params, data_dir).map_err(anyhow::Error::msg)?;
    let result = if iters > 1 {
        // Warm-cache bench: process the image `iters` times in the same backend
        // instance. The first iteration pays shader-compile cost; subsequent ones
        // hit the cache and reflect "GUI live preview" performance.
        let mut last = pipeline.process(image.clone(), backend.as_ref());
        let iter1_ms = t.elapsed().as_millis();
        eprintln!("Pipeline (iter 1): {} ms (cold)", iter1_ms);
        for i in 2..=iters {
            let ti = Instant::now();
            last = pipeline.process(image.clone(), backend.as_ref());
            eprintln!(
                "Pipeline (iter {}): {} ms (warm)",
                i,
                ti.elapsed().as_millis()
            );
        }
        last
    } else {
        let r = pipeline.process(image, backend.as_ref());
        eprintln!("Pipeline: {} ms", t.elapsed().as_millis());
        r
    };

    // Save output
    let t = Instant::now();
    spektrafilm_io::save_converted(
        &result,
        &pipeline.params.io.output_color_space,
        pipeline.params.io.output_cctf_encoding,
        output,
        &save_options,
        &loaded.metadata,
        data_dir,
    )?;
    eprintln!(
        "Saved: {} ({}x{}), {} ms",
        output.display(),
        result.width,
        result.height,
        t.elapsed().as_millis()
    );

    // Optional raw f64 dump for bit-exact parity comparison.
    if let Some(p) = raw_out {
        use std::io::Write;
        let buf: Vec<f64> = result.data.iter().map(|&v| v as f64).collect();
        let bytes = bytemuck_cast_f64_to_bytes(&buf);
        let mut f = std::fs::File::create(p)
            .with_context(|| format!("creating raw_out file: {}", p.display()))?;
        f.write_all(bytes)?;
        eprintln!("raw f64 buffer: {} ({} values)", p.display(), buf.len());
    }

    if show_timings {
        eprintln!("Total: {} ms", total_start.elapsed().as_millis());
    }

    Ok(())
}

fn bytemuck_cast_f64_to_bytes(v: &[f64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn cmd_list_profiles(data_dir: &Path) {
    let profiles_dir = data_dir.join("profiles");
    let mut names = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&profiles_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy().to_string();
            if name.ends_with(".json") {
                names.push(name.trim_end_matches(".json").to_string());
            }
        }
    } else {
        eprintln!("Profile directory not found: {}", profiles_dir.display());
        return;
    }
    names.sort();
    for name in &names {
        // Try to load and show the human-readable name
        if let Ok(p) = profile::load_profile_by_name(data_dir, name) {
            let display_name = p.info.name.as_deref().unwrap_or(name);
            let film_type = &p.info.film_type;
            let support = &p.info.support;
            println!("{name:<40} {display_name:<30} ({film_type}, {support})");
        } else {
            println!("{name}");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_export_lut(
    film_name: &str,
    paper_name: Option<&str>,
    size: u32,
    output: &Path,
    data_dir: &Path,
    params_file: Option<&Path>,
    input: &str,
    out: &str,
    format: &str,
    stops: Option<f64>,
    topology: Option<&str>,
    combinations: bool,
    qa: bool,
) -> Result<()> {
    use spektrafilm_core::lut_export::{self, BakeOptions};
    let film = profile::load_profile_by_name(data_dir, film_name)?;
    let print_stock = paper_name
        .map(str::to_owned)
        .or(film.info.target_print.clone())
        .context("specify --paper for a film without target_print")?;
    let params = if let Some(path) = params_file {
        serde_json::from_reader(std::fs::File::open(path)?)?
    } else {
        RuntimeParams::default()
    };
    let entry = lut_export::space(input).map_err(anyhow::Error::msg)?;
    let gain = stops.map(|v| 0.18 * 2f64.powf(v) / entry.curve.decode(1.));
    let options = BakeOptions {
        film: film_name,
        paper: &print_stock,
        size,
        input,
        output: out,
        gain,
        params,
    };
    if let Some(topology) = topology {
        let meta = spektrafilm_core::lut_bundle::write_bundle(
            &options,
            data_dir,
            output,
            topology,
            format,
            combinations,
        )
        .map_err(anyhow::Error::msg)?;
        if qa {
            let quality = spektrafilm_core::lut_quality::assess(&options, data_dir, &meta, output)
                .map_err(anyhow::Error::msg)?;
            quality.write_report(output).map_err(anyhow::Error::msg)?;
        }
        return Ok(());
    }
    if qa {
        bail!("--qa requires --topology");
    }
    let image = lut_export::bake(&options, data_dir).map_err(anyhow::Error::msg)?;
    let title = format!("spektrafilm {film_name} -> {print_stock} ({input} -> {out})");
    lut_export::write_lut(&image, size, output, format, &title, input)
        .map_err(anyhow::Error::msg)?;
    let metadata = serde_json::json!({"version":env!("CARGO_PKG_VERSION"),"film":film_name,"paper":print_stock,"input_color_space":input,"output_color_space":out,"resolution":size,"format":format,"input_gain":gain.unwrap_or(entry.auto_gain),"params":options.params,"precision":if cfg!(feature="precision-f64") {"f64"} else {"f32"}});
    std::fs::write(
        output.with_extension("json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    Ok(())
}
