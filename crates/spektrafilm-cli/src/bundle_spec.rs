use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use spektrafilm_core::{
    lut_export::{self, BakeOptions},
    params::{InputGamutCompressParams, OutputGamutCompressParams, RuntimeParams},
};
use std::path::{Path, PathBuf};
#[derive(Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
struct Spec {
    film_profile: String,
    print_profiles: Vec<String>,
    input_color_space: String,
    output_color_space: String,
    name: String,
    topology: String,
    resolution: u32,
    target: Option<String>,
    container: String,
    qa: bool,
    qa_print_index: Option<usize>,
    input_gamut_compress: InputGamutCompressParams,
    output_gamut_compress: serde_json::Value,
    stops_above_midgray: serde_json::Value,
    ocio_config: bool,
    include_combinations: bool,
}
impl Default for Spec {
    fn default() -> Self {
        Self {
            film_profile: "kodak_portra_400".into(),
            print_profiles: vec!["kodak_portra_endura".into()],
            input_color_space: "sRGB".into(),
            output_color_space: "sRGB".into(),
            name: String::new(),
            topology: "1lut".into(),
            resolution: 33,
            target: None,
            container: "directory".into(),
            qa: false,
            qa_print_index: None,
            input_gamut_compress: Default::default(),
            output_gamut_compress: serde_json::to_value(OutputGamutCompressParams::default())
                .unwrap(),
            stops_above_midgray: serde_json::json!("auto"),
            ocio_config: false,
            include_combinations: false,
        }
    }
}

/// Values explicitly supplied on the command line replace matching spec fields.
#[derive(clap::Args, Default)]
pub struct Overrides {
    #[arg(long)]
    name: Option<String>,
    #[arg(long)]
    film: Option<String>,
    #[arg(long = "print", alias = "paper")]
    prints: Vec<String>,
    #[arg(long, alias = "input")]
    input_space: Option<String>,
    #[arg(long)]
    output_space: Option<String>,
    #[arg(long, value_parser = ["1lut", "2lut", "3lut", "4lut"])]
    topology: Option<String>,
    #[arg(long, alias = "size")]
    resolution: Option<u32>,
    #[arg(long)]
    target: Option<String>,
    #[arg(long, value_parser = ["directory", "zip"])]
    container: Option<String>,
    #[arg(
        long = "stops-above-gray",
        alias = "stops-above-midgray",
        allow_hyphen_values = true
    )]
    stops: Option<String>,
    #[arg(long)]
    qa: bool,
    #[arg(long)]
    qa_print_index: Option<usize>,
    #[arg(long)]
    ocio_config: bool,
    #[arg(long, alias = "include-combinations")]
    combinations: bool,
}
impl Overrides {
    fn apply(self, spec: &mut Spec) -> Result<()> {
        macro_rules! replace {
            ($field:ident, $value:ident) => {
                if let Some(value) = self.$value {
                    spec.$field = value;
                }
            };
        }
        replace!(name, name);
        replace!(film_profile, film);
        replace!(input_color_space, input_space);
        replace!(output_color_space, output_space);
        replace!(topology, topology);
        replace!(resolution, resolution);
        replace!(container, container);
        if !self.prints.is_empty() {
            spec.print_profiles = self.prints;
        }
        if let Some(value) = self.target {
            spec.target = Some(value);
        }
        if let Some(value) = self.qa_print_index {
            spec.qa_print_index = Some(value);
        }
        if self.qa {
            spec.qa = true;
        }
        if self.ocio_config {
            spec.ocio_config = true;
        }
        if self.combinations {
            spec.include_combinations = true;
        }
        if let Some(value) = self.stops {
            spec.stops_above_midgray = match value.as_str() {
                "auto" => serde_json::json!("auto"),
                "native" => serde_json::Value::Null,
                _ => {
                    let stops = value.parse::<f64>().map_err(|_| {
                        anyhow::anyhow!("stops must be a number, 'auto', or 'native'")
                    })?;
                    if !stops.is_finite() {
                        bail!("stops must be finite");
                    }
                    serde_json::json!(stops)
                }
            };
        }
        Ok(())
    }
}

pub fn build(
    spec_file: Option<&Path>,
    output: &Path,
    data_dir: &Path,
    overrides: Overrides,
) -> Result<PathBuf> {
    let mut spec: Spec = if let Some(spec_file) = spec_file {
        let text = std::fs::read_to_string(spec_file)?;
        if spec_file.extension().and_then(|s| s.to_str()) == Some("json") {
            serde_json::from_str(&text)?
        } else {
            toml::from_str(&text)?
        }
    } else {
        Spec::default()
    };
    overrides.apply(&mut spec)?;
    if spec.print_profiles.is_empty() {
        bail!("print_profiles must not be empty");
    }
    if spec
        .qa_print_index
        .is_some_and(|i| i >= spec.print_profiles.len())
    {
        bail!("qa_print_index is outside print_profiles");
    }
    if !["directory", "zip"].contains(&spec.container.as_str()) {
        bail!("container must be directory or zip");
    }
    let input = lut_export::space(&spec.input_color_space).map_err(anyhow::Error::msg)?;
    let out = lut_export::space(&spec.output_color_space).map_err(anyhow::Error::msg)?;
    let gain = if spec.stops_above_midgray.is_null() {
        Some(1.)
    } else if let Some(stops) = spec.stops_above_midgray.as_f64() {
        Some(0.18 * 2f64.powf(stops) / input.curve.decode(1.))
    } else if spec.stops_above_midgray == "auto" {
        None
    } else {
        bail!("stops_above_midgray must be a number, null (JSON), or 'auto'");
    };
    let format = if let Some(target) = &spec.target {
        if target != "lumix_realtime_vlog" {
            bail!("unknown delivery target: {target}");
        }
        if input.tag != "vlog" || !["srgb", "rec709", "rec2020"].contains(&out.tag) {
            bail!("lumix_realtime_vlog requires V-Log input and sRGB, Rec.709, or Rec.2020 output");
        }
        if spec.topology != "1lut" {
            bail!("Lumix delivery requires topology='1lut'");
        }
        "lumix"
    } else {
        "cube"
    };
    let mut params = RuntimeParams::default();
    params.io.input_gamut_compress = spec.input_gamut_compress.clone();
    params.io.output_gamut_compress = if let Some(algorithm) = spec.output_gamut_compress.as_str() {
        OutputGamutCompressParams {
            algorithm: algorithm.into(),
            ..Default::default()
        }
    } else {
        serde_json::from_value(spec.output_gamut_compress.clone())?
    };
    params.validate().map_err(anyhow::Error::msg)?;
    let destination = if spec.name.is_empty() {
        output.to_path_buf()
    } else {
        if spec.name.contains(['/', '\\']) || spec.name == ".." {
            bail!("bundle name must be a single directory name");
        }
        output.join(&spec.name)
    };
    let destination = std::path::absolute(destination)?;
    let staged = spektrafilm_core::bundle_output::OutputDirectory::new(&destination)
        .map_err(anyhow::Error::msg)?;
    let root = staged.path();
    for (index, paper) in spec.print_profiles.iter().enumerate() {
        if paper.contains(['/', '\\']) || paper == ".." {
            bail!("print profile must be a stock identifier");
        }
        let directory = if spec.print_profiles.len() == 1 {
            root.to_path_buf()
        } else {
            root.join(paper)
        };
        let options = BakeOptions {
            film: &spec.film_profile,
            paper,
            size: spec.resolution,
            input: &spec.input_color_space,
            output: &spec.output_color_space,
            gain,
            params: params.clone(),
        };
        let meta = spektrafilm_core::lut_bundle::write_bundle(
            &options,
            data_dir,
            &directory,
            &spec.topology,
            format,
            spec.include_combinations,
        )
        .map_err(anyhow::Error::msg)?;
        if !spec.ocio_config {
            let config = directory.join("config.ocio");
            if config.exists() {
                std::fs::remove_file(config)?;
            }
        }
        if spec.qa && spec.qa_print_index.is_none_or(|i| i == index) {
            let quality =
                spektrafilm_core::lut_quality::assess(&options, data_dir, &meta, &directory)
                    .map_err(anyhow::Error::msg)?;
            quality
                .write_report(&directory)
                .map_err(anyhow::Error::msg)?;
        }
    }
    std::fs::write(root.join("spec.json"), serde_json::to_vec_pretty(&spec)?)?;
    let license = data_dir.join("licenses/SPEKTRAFILM_LICENSE.txt");
    if license.exists() {
        std::fs::copy(license, root.join("SPEKTRAFILM_LICENSE.txt"))?;
    }
    if spec.container == "zip" {
        let archive = destination.with_extension("zip");
        let parent = archive
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let temporary = tempfile::NamedTempFile::new_in(parent)?;
        let file = temporary.reopen()?;
        let mut writer = zip::ZipWriter::new(file);
        archive_directory(&mut writer, root, root)?;
        writer.finish()?.sync_all()?;
        staged.commit().map_err(anyhow::Error::msg)?;
        temporary.persist(&archive)?;
        Ok(archive)
    } else {
        staged.commit().map_err(anyhow::Error::msg)
    }
}
fn archive_directory(
    writer: &mut zip::ZipWriter<std::fs::File>,
    root: &Path,
    directory: &Path,
) -> Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            archive_directory(writer, root, &path)?;
        } else if entry.file_type()?.is_file() {
            writer.start_file(
                path.strip_prefix(root)?
                    .to_string_lossy()
                    .replace('\\', "/"),
                zip::write::SimpleFileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated),
            )?;
            std::io::copy(&mut std::fs::File::open(path)?, writer)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_flags_override_loaded_values_and_preserve_unspecified_fields() {
        let mut spec = Spec {
            resolution: 17,
            qa: true,
            name: "original".into(),
            ..Default::default()
        };
        Overrides {
            resolution: Some(65),
            film: Some("kodak_gold_200".into()),
            prints: vec!["fujifilm_crystal_archive_typeii".into()],
            stops: Some("native".into()),
            ..Default::default()
        }
        .apply(&mut spec)
        .unwrap();
        assert_eq!(spec.resolution, 65);
        assert!(spec.qa);
        assert_eq!(spec.name, "original");
        assert_eq!(spec.film_profile, "kodak_gold_200");
        assert_eq!(spec.print_profiles, ["fujifilm_crystal_archive_typeii"]);
        assert!(spec.stops_above_midgray.is_null());
    }
}
