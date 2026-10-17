//! Portable project files and explicit startup defaults.
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use spektrafilm_core::params::RuntimeParams;
use std::path::{Path, PathBuf};
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DisplayParams {
    pub gray_18_canvas: bool,
    pub display_transform: bool,
    pub animate: bool,
    pub show_border: bool,
    pub show_paper_back: bool,
    pub show_input: bool,
    pub show_output: bool,
    pub input_opacity: f32,
    pub output_opacity: f32,
    pub compare_split: bool,
    pub split_position: f32,
    pub white_padding: f32,
    pub output_interpolation: String,
}
impl Default for DisplayParams {
    fn default() -> Self {
        Self {
            gray_18_canvas: true,
            display_transform: true,
            animate: true,
            show_border: true,
            show_paper_back: true,
            show_input: false,
            show_output: true,
            input_opacity: 1.,
            output_opacity: 1.,
            compare_split: false,
            split_position: 0.5,
            white_padding: 0.03,
            output_interpolation: "linear".into(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Project {
    pub version: u32,
    pub rotation: u8,
    pub auto_preview: bool,
    pub gpu_export: bool,
    pub lut: crate::lut_ui::Settings,
    pub display: DisplayParams,
    pub scan_for_print_restore: Option<[bool; 3]>,
    pub film: String,
    pub paper: String,
    pub params: RuntimeParams,
    pub image: Option<PathBuf>,
    pub zoom: f32,
    pub pan: [f32; 2],
    pub raw: spektrafilm_io::RawOptions,
    pub output: spektrafilm_io::SaveOptions,
}
impl Default for Project {
    fn default() -> Self {
        Self {
            version: 1,
            rotation: 0,
            auto_preview: true,
            gpu_export: true,
            lut: Default::default(),
            display: Default::default(),
            scan_for_print_restore: None,
            film: "kodak_gold_200".into(),
            paper: "kodak_portra_endura".into(),
            params: {
                let mut p = RuntimeParams::default();
                p.settings.preview_max_size = 2048;
                p
            },
            image: None,
            zoom: 1.,
            pan: [0., 0.],
            raw: Default::default(),
            output: Default::default(),
        }
    }
}
impl Project {
    pub fn load(path: &Path) -> Result<Self> {
        let source: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path)?).context("reading project")?;
        let mut p: Self = if source["input_image"].is_object() {
            python_project(&source)?
        } else if source["params"].is_object() {
            serde_json::from_value(source)?
        } else {
            bail!("not a spektrafilm project or Python GUI state file")
        };
        if p.version != 1 {
            bail!("unsupported project version: {}", p.version);
        }
        p.params.validate().map_err(anyhow::Error::msg)?;
        if !p.display.white_padding.is_finite() || !(0.0..=1.0).contains(&p.display.white_padding) {
            bail!("display white_padding must be in [0, 1]");
        }
        p.rotation %= 4;
        for value in [
            p.display.input_opacity,
            p.display.output_opacity,
            p.display.split_position,
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                bail!("display opacity / split must be in [0, 1]");
            }
        }
        spektrafilm_math::transfer::get(&p.output.color_space).map_err(anyhow::Error::msg)?;
        if let Some(image) = &mut p.image {
            if image.is_relative() {
                *image = path.parent().unwrap_or(Path::new(".")).join(&image);
            }
        }
        Ok(p)
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        self.params.validate().map_err(anyhow::Error::msg)?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut p = self.clone();
        if let Some(image) = &self.image {
            if let Ok(relative) = image.strip_prefix(path.parent().unwrap_or(Path::new("."))) {
                p.image = Some(relative.into());
            }
        }
        let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&temporary, serde_json::to_vec_pretty(&p)?)?;
        std::fs::rename(&temporary, path).context("saving project atomically")
    }
}
pub fn default_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA")
            .map(PathBuf::from)
            .unwrap_or(home)
    } else if cfg!(target_os = "macos") {
        home.join("Library/Application Support")
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or(home.join(".config"))
    };
    base.join("spektrafilm-rs/defaults.json")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn project_roundtrip() {
        let path =
            std::env::temp_dir().join(format!("spektrafilm-project-{}.json", std::process::id()));
        let mut p = Project::default();
        p.params.camera.exposure_compensation_ev = 1.5;
        p.raw.tint = 12.;
        p.gpu_export = false;
        p.rotation = 3;
        p.display.output_interpolation = "blackman".into();
        p.display.compare_split = true;
        p.display.input_opacity = 0.25;
        p.display.animate = false;
        p.lut.input = "Sony S-Log3".into();
        p.lut.qa = true;
        p.save(&path).unwrap();
        let loaded = Project::load(&path).unwrap();
        assert_eq!(loaded.params.camera.exposure_compensation_ev, 1.5);
        assert_eq!(loaded.raw.tint, 12.);
        assert!(!loaded.gpu_export);
        assert_eq!(loaded.rotation, 3);
        assert_eq!(loaded.display.output_interpolation, "blackman");
        assert!(loaded.display.compare_split);
        assert_eq!(loaded.display.input_opacity, 0.25);
        assert!(!loaded.display.animate);
        assert_eq!(loaded.lut.input, "Sony S-Log3");
        assert!(loaded.lut.qa);
        std::fs::remove_file(path).unwrap();
    }
}

fn set_path(root: &mut serde_json::Value, path: &str, value: serde_json::Value) {
    let mut current = root;
    let parts: Vec<_> = path.split('.').collect();
    for key in &parts[..parts.len() - 1] {
        if !current[*key].is_object() {
            current[*key] = serde_json::json!({});
        }
        current = &mut current[*key];
    }
    current[parts[parts.len() - 1]] = value;
}
fn merge_group(root: &mut serde_json::Value, path: &str, source: &serde_json::Value) {
    if let Some(fields) = source.as_object() {
        for (key, value) in fields {
            let key = if path == "params.film_render.grain" && key.starts_with("particle_") {
                format!("agx_{key}")
            } else {
                key.clone()
            };
            set_path(root, &format!("{path}.{key}"), value.clone());
        }
    }
}
fn python_project(source: &serde_json::Value) -> Result<Project> {
    let mut output = serde_json::to_value(Project::default())?;
    let bindings: std::collections::BTreeMap<String, String> =
        serde_json::from_str(include_str!("python_gui_bindings.json"))?;
    for (src, dst) in bindings {
        let mut value = source;
        for key in src.split('.') {
            value = &value[key];
        }
        if !value.is_null() {
            set_path(&mut output, &dst, value.clone());
        }
    }
    for (src, dst) in [
        ("grain", "film_render.grain"),
        ("halation", "film_render.halation"),
        ("couplers", "film_render.dir_couplers"),
        ("chemistry", "print_render.density_curves_morph"),
        ("camera", "camera"),
        ("camera_diffusion", "camera.diffusion_filter"),
        ("enlarger_diffusion", "enlarger.diffusion_filter"),
        ("glare", "print_render.glare"),
        ("scanner", "scanner"),
        ("input_gamut_compress", "io.input_gamut_compress"),
        ("output_gamut_compress", "io.output_gamut_compress"),
    ] {
        merge_group(&mut output, &format!("params.{dst}"), &source[src]);
    }
    for key in [
        "preflash_exposure",
        "preflash_y_filter_shift",
        "preflash_m_filter_shift",
    ] {
        if !source["preflashing"][key].is_null() {
            set_path(
                &mut output,
                &format!("params.enlarger.{key}"),
                source["preflashing"][key].clone(),
            );
        }
    }
    let raw = if source["load_raw"].is_object() {
        &source["load_raw"]
    } else {
        &source["gui_only"]["load_raw"]
    };
    merge_group(&mut output, "raw", raw);
    set_path(
        &mut output,
        "params.settings.apply_stock_specifics",
        serde_json::json!(false),
    );
    Ok(serde_json::from_value(output)?)
}

#[cfg(test)]
mod python_tests {
    use super::*;
    #[test]
    fn imports_python_gui_state() {
        let source: serde_json::Value =
            serde_json::from_str(include_str!("python_gui_state_fixture.json")).unwrap();
        let project = python_project(&source).unwrap();
        project.params.validate().unwrap();
        assert_eq!(project.params.camera.exposure_compensation_ev, 1.25);
        assert_eq!(project.params.film_render.grain.agx_particle_area_um2, 0.6);
        assert_eq!(project.params.film_render.channel_order, [0, 2, 1]);
        assert!(!project.auto_preview);
        assert_eq!(project.output.color_space, "Display P3");
        assert!(!project.output.encoded);
        assert!((project.display.white_padding - 0.07).abs() < 1e-6);
        assert!(!project.display.gray_18_canvas);
        assert_eq!(project.raw.white_balance, "tungsten");
        assert!(project.raw.lens_correction);
        assert_eq!(
            project.film,
            source["simulation"]["film_stock"].as_str().unwrap()
        );
    }
}
