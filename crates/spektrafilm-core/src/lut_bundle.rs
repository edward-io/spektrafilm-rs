//! Multi-stage LUT bundles with explicit, invertible boundary-wire encodings.
use crate::{
    lut_export::{self, BakeOptions},
    pipeline::Pipeline,
    profile::load_profile_by_name,
};
use serde::{Deserialize, Serialize};
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Serialize, Deserialize)]
pub struct Wire {
    pub min: [f64; 3],
    pub max: [f64; 3],
}
impl Wire {
    fn decode(&self, image: &mut ImageBuf) {
        for p in image.data.chunks_exact_mut(3) {
            for c in 0..3 {
                p[c] = from_f64(
                    (self.min[c] + p[c] as f64 * (self.max[c] - self.min[c])) as f32 as f64,
                );
            }
        }
    }
    fn encode(&self, image: &mut ImageBuf) {
        for p in image.data.chunks_exact_mut(3) {
            for c in 0..3 {
                p[c] = from_f64(
                    ((p[c] as f64 - self.min[c]) / (self.max[c] - self.min[c])).clamp(0., 1.),
                );
            }
        }
    }
}
#[derive(Serialize)]
pub struct Entry {
    pub file: String,
    pub inject: String,
    pub collect: String,
}
#[derive(Serialize)]
pub struct BundleMeta {
    pub version: u32,
    pub film: String,
    pub paper: String,
    pub input: String,
    pub output: String,
    pub resolution: u32,
    pub topology: String,
    pub wires: BTreeMap<String, Wire>,
    pub luts: Vec<Entry>,
    pub schema_version: u32,
    pub provenance: serde_json::Value,
    pub stocks: serde_json::Value,
    pub color_spaces: serde_json::Value,
    pub input_exposure: serde_json::Value,
    pub params_snapshot: serde_json::Value,
}

pub fn write_bundle(
    options: &BakeOptions<'_>,
    data_dir: &Path,
    destination: &Path,
    topology: &str,
    format: &str,
    combinations: bool,
) -> Result<BundleMeta, String> {
    lut_export::validate_format(options.size, format, options.input)?;
    if format == "lumix" && topology != "1lut" {
        return Err("Lumix delivery requires topology='1lut'".into());
    }
    let output = crate::bundle_output::OutputDirectory::new(destination)?;
    let meta = write_bundle_contents(
        options,
        data_dir,
        output.path(),
        topology,
        format,
        combinations,
    )?;
    output.commit()?;
    Ok(meta)
}

fn write_bundle_contents(
    options: &BakeOptions<'_>,
    data_dir: &Path,
    destination: &Path,
    topology: &str,
    format: &str,
    combinations: bool,
) -> Result<BundleMeta, String> {
    let taps: &[&str] = match topology {
        "1lut" => &["rgb_in", "rgb_out"],
        "2lut" => &["rgb_in", "cmy_film", "rgb_out"],
        "3lut" => &["rgb_in", "log_e_film", "cmy_film", "rgb_out"],
        "4lut" => &["rgb_in", "log_e_film", "cmy_film", "log_e_print", "rgb_out"],
        _ => return Err("topology must be 1lut, 2lut, 3lut, or 4lut".into()),
    };
    if !(2..=129).contains(&options.size) {
        return Err("LUT resolution must be in 2..=129".into());
    }
    if options.params.io.scan_film && topology == "4lut" {
        return Err("4lut requires a print stage".into());
    }
    let input = lut_export::space(options.input)?;
    let output = lut_export::space(options.output)?;
    if !input.input_role || !output.output_role {
        return Err(
            "LUT input/output space is not enabled for that role in the reference registry".into(),
        );
    }
    if options.gain.is_some_and(|g| !g.is_finite() || g <= 0.) {
        return Err("LUT input gain must be finite and positive".into());
    }
    let gain = options.gain.unwrap_or(input.auto_gain);
    let mut params = options.params.clone();
    params.debug.lut_mode = true;
    params.settings.preview_mode = false;
    params.workflow.route = "input > film > print > scan".into();
    params.io.scan_film = false;
    params.io.input_color_space = input.primaries.into();
    params.io.output_color_space = output.primaries.into();
    params.io.input_cctf_decoding = false;
    params.io.output_cctf_encoding = false;
    params.io.crop = false;
    params.io.upscale_factor = 1.;
    params.taps = Default::default();
    let film = load_profile_by_name(data_dir, options.film).map_err(|e| e.to_string())?;
    let paper = load_profile_by_name(data_dir, options.paper).map_err(|e| e.to_string())?;
    let base = Pipeline::new_with_spectral(film, paper, params, data_dir)?;
    let backend = spektrafilm_gpu::cpu_backend::CpuBackend;
    let mut probe = lut_export::grid(9);
    lut_export::decode_grid(&mut probe, input, gain);
    let mut wires = BTreeMap::new();
    for &tap in taps.iter().skip(1).take(taps.len() - 2) {
        let mut pipeline = base.clone();
        pipeline.params.taps.collect = Some(tap.into());
        let sample = pipeline.process(probe.clone(), &backend);
        let wire = if tap == "cmy_film" {
            let max = std::array::from_fn(|c| {
                (sample.pixels().map(|p| p[c] as f64).fold(0., f64::max) * 1.05 * 10000.).ceil()
                    / 10000.
            });
            Wire {
                min: [-0.2; 3],
                max,
            }
        } else {
            let min = ((sample
                .data
                .iter()
                .map(|v| *v as f64)
                .fold(f64::INFINITY, f64::min)
                - 0.1)
                * 10000.)
                .floor()
                / 10000.;
            let max = ((sample
                .data
                .iter()
                .map(|v| *v as f64)
                .fold(f64::NEG_INFINITY, f64::max)
                + 0.1)
                * 10000.)
                .ceil()
                / 10000.;
            Wire {
                min: [min; 3],
                max: [max; 3],
            }
        };
        wires.insert(tap.to_string(), wire);
    }
    std::fs::create_dir_all(destination).map_err(|e| e.to_string())?;
    let ext = match format {
        "hald_png" => "png",
        "3dl" => "3dl",
        _ => "cube",
    };
    let mut entries = Vec::new();
    for start in 0..taps.len() - 1 {
        for end in start + 1..taps.len() {
            if !combinations && end != start + 1 {
                continue;
            }
            let (inject, collect) = (taps[start], taps[end]);
            let mut grid = lut_export::grid(options.size);
            if inject == "rgb_in" {
                lut_export::decode_grid(&mut grid, input, gain);
            } else {
                wires[inject].decode(&mut grid);
            }
            let mut pipeline = base.clone();
            pipeline.params.taps.inject = Some(inject.into());
            pipeline.params.taps.collect = Some(collect.into());
            let mut image = pipeline.process(grid, &backend);
            if collect == "rgb_out" {
                lut_export::encode_grid(&mut image, output);
            } else {
                wires[collect].encode(&mut image);
            }
            let filename = format!("l{}_{}_{}.{}", start + 1, inject, collect, ext);
            let title = format!(
                "{} -> {}: {inject} -> {collect}",
                options.film, options.paper
            );
            lut_export::write_lut(
                &image,
                options.size,
                &destination.join(&filename),
                format,
                &title,
                options.input,
            )?;
            entries.push(Entry {
                file: filename,
                inject: inject.into(),
                collect: collect.into(),
            });
        }
    }
    let meta = BundleMeta {
        schema_version: 1,
        provenance: serde_json::json!({
            "implementation": "spektrafilm-rs", "implementation_version": env!("CARGO_PKG_VERSION"),
            "reference_version": "0.3.4", "reference_commit": "28bf883e1672e884307edc75852549376e13644e",
            "created_unix_seconds": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(|e| e.to_string())?.as_secs(),
            "precision": if cfg!(feature="precision-f64") {"f64"} else {"f32"},
            "license": "Profiles and derived LUTs: CC BY-SA 4.0; see SPEKTRAFILM_LICENSE.txt.",
            "citation": "Andrea Volpato, spektrafilm. See https://github.com/andreavolpato/spektrafilm and CITATION.cff.",
            "project_url": "https://github.com/andreavolpato/spektrafilm"
        }),
        stocks: serde_json::json!({"film": options.film, "prints": [options.paper]}),
        color_spaces: serde_json::json!({"input": {"name":input.name,"primaries":input.primaries,"cctf":!matches!(input.curve, spektrafilm_math::transfer::Transfer::Linear)},
            "output": {"name":output.name,"primaries":output.primaries,"cctf":!matches!(output.curve, spektrafilm_math::transfer::Transfer::Linear)}}),
        input_exposure: serde_json::json!({"gain":gain,"exposure_ev":(gain/input.auto_gain).log2(),"stops_above_midgray":(gain*input.curve.decode(1.)/0.18).log2(),"mode":if options.gain.is_none(){"auto"}else{"explicit"}}),
        params_snapshot: serde_json::json!({options.paper: {"runtime": base.params,"input_gain":gain,"output_gain":output.output_gain,"resolution":options.size,"topology":topology,"format":format}}),
        version: 1,
        film: options.film.into(),
        paper: options.paper.into(),
        input: options.input.into(),
        output: options.output.into(),
        resolution: options.size,
        topology: topology.into(),
        wires,
        luts: entries,
    };
    std::fs::write(
        destination.join("bundle.json"),
        serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let license = data_dir.join("licenses/SPEKTRAFILM_LICENSE.txt");
    if license.is_file() {
        std::fs::copy(license, destination.join("SPEKTRAFILM_LICENSE.txt"))
            .map_err(|e| e.to_string())?;
    }
    let canonical: Vec<_> = meta
        .luts
        .iter()
        .filter(|e| {
            let a = taps.iter().position(|x| *x == e.inject).unwrap();
            let b = taps.iter().position(|x| *x == e.collect).unwrap();
            b == a + 1
        })
        .collect();
    if format == "cube" || format == "lumix" {
        if let Some(ocio) = crate::ocio::config(options, &meta, taps) {
            std::fs::write(destination.join("config.ocio"), ocio).map_err(|e| e.to_string())?;
        }
    }
    let readme = format!(
        "# spektrafilm LUT bundle\n\n{} -> {}, {} -> {}.\n\nApply the canonical LUTs in this order:\n\n{}\n\nBoundary wire ranges and transport encodings are recorded in bundle.json. Spatial and stochastic effects, auto exposure, scanner corrections, and highlight boost are disabled while baking.\n\nThe LUTs contain a sampled approximation of the deterministic spectral model; image-space effects must be added separately.\n\nUse the declared input encoding before applying this chain. Intermediate boundary wires carry normalized data and are not display color spaces. Optional combined LUTs replace their corresponding stages; do not apply them in addition to those stages.\n\nProfiles and derived LUTs: CC BY-SA 4.0. See SPEKTRAFILM_LICENSE.txt. Please cite Andrea Volpato and the [spektrafilm project](https://github.com/andreavolpato/spektrafilm). Runtime configuration, exposure gain, precision and source versions are recorded in bundle.json.\n",
        options.film,
        options.paper,
        options.input,
        options.output,
        canonical
            .iter()
            .map(|e| format!("- {}", e.file))
            .collect::<Vec<_>>()
            .join("\n")
    );
    std::fs::write(destination.join("README.md"), readme).map_err(|e| e.to_string())?;
    Ok(meta)
}
