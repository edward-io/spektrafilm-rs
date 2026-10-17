//! Native LUT export controls; the bundled f64 CLI owns deterministic baking.
use crate::{App, TempPath, locate_f64_cli, run_cli, tooltips};
use eframe::egui;
use serde::{Deserialize, Serialize};
use spektrafilm_core::lut_export;
use std::{path::PathBuf, process::Command};

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub exposure_ev: f64,
    pub input: String,
    pub output: String,
    pub format: String,
    pub size: u32,
    pub bundle: bool,
    pub topology: String,
    pub combinations: bool,
    pub qa: bool,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            exposure_ev: 0.,
            input: "sRGB".into(),
            output: "sRGB".into(),
            format: "cube".into(),
            size: 33,
            bundle: true,
            topology: "1lut".into(),
            combinations: false,
            qa: false,
        }
    }
}
impl App {
    pub(crate) fn lut_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::CollapsingHeader::new("LUT export")
            .show(ui, |ui| {
                ui.label(tooltips::LUT_SETTINGS_NOTE);
                for (id, label, selected, input_role) in [
                    ("lut_input", "Input", &mut self.lut.input, true),
                    ("lut_output", "Output", &mut self.lut.output, false),
                ] {
                    ui.label(label).on_hover_text(if input_role {
                        tooltips::LUT_INPUT
                    } else {
                        tooltips::LUT_OUTPUT
                    });
                    egui::ComboBox::from_id_salt(id)
                        .selected_text(selected.as_str())
                        .show_ui(ui, |ui| {
                            for space in lut_export::spaces().filter(|s| {
                                if input_role {
                                    s.input_role
                                } else {
                                    s.output_role
                                }
                            }) {
                                ui.selectable_value(selected, space.name.into(), space.name)
                                    .on_hover_text(tooltips::option(space.name));
                            }
                        });
                }
                tooltips::labeled(ui,egui::Slider::new(&mut self.lut.exposure_ev,-6. ..=6.),"LUT exposure EV","Re-expose the LUT input from its native midgray; zero preserves the source exposure.");
                ui.label("File format")
                    .on_hover_text(tooltips::MENU_LUT_FORMAT);
                egui::ComboBox::from_id_salt("lut_format")
                    .selected_text(&self.lut.format)
                    .show_ui(ui, |ui| {
                        for (id, name) in [
                            ("cube", "CUBE"),
                            ("3dl", "Autodesk 3DL (10-bit)"),
                            ("lumix", "Lumix V-Log CUBE"),
                            ("hald_png", "Hald PNG"),
                        ] {
                            if ui
                                .selectable_value(&mut self.lut.format, id.into(), name)
                                .on_hover_text(tooltips::option(id))
                                .changed()
                            {
                                if id == "hald_png" {
                                    self.lut.size = 64;
                                } else if ![17, 33, 65].contains(&self.lut.size) {
                                    self.lut.size = 33;
                                }
                                if id == "lumix" {
                                    self.lut.input = "Panasonic V-Log".into();
                                    self.lut.output = "sRGB".into();
                                    self.lut.topology = "1lut".into();
                                }
                            }
                        }
                    });
                tooltips::labeled(
                    ui,
                    egui::DragValue::new(&mut self.lut.size).range(2..=129),
                    "Cube resolution: ",
                    tooltips::CONTROL_CUBE_RESOLUTION,
                );
                if self.lut.format == "hald_png" {
                    ui.label("Hald requires a square resolution, such as 16 or 64.");
                }
                tooltips::checkbox(
                    ui,
                    &mut self.lut.bundle,
                    "Export a bundle with metadata and OCIO",
                    tooltips::CONTROL_EXPORT_A_BUNDLE_WITH_METADATA_AND_OCIO,
                );
                if self.lut.bundle {
                    ui.label("Stage layout")
                        .on_hover_text(tooltips::MENU_LUT_TOPOLOGY);
                    egui::ComboBox::from_id_salt("lut_topology")
                        .selected_text(&self.lut.topology)
                        .show_ui(ui, |ui| {
                            for value in ["1lut", "2lut", "3lut", "4lut"] {
                                ui.selectable_value(&mut self.lut.topology, value.into(), value)
                                    .on_hover_text(tooltips::option(value));
                            }
                        });
                    tooltips::checkbox(
                        ui,
                        &mut self.lut.combinations,
                        "Include combined stages",
                        tooltips::CONTROL_INCLUDE_COMBINED_STAGES,
                    );
                    tooltips::checkbox(
                        ui,
                        &mut self.lut.qa,
                        "Include diagnostic report",
                        tooltips::CONTROL_INCLUDE_DIAGNOSTIC_REPORT,
                    );
                }
                let available = self.export_job.is_none();
                if ui
                    .add_enabled(available, egui::Button::new("Export LUT…"))
                    .on_hover_text(tooltips::CONTROL_EXPORT_LUT)
                    .on_disabled_hover_text(tooltips::CONTROL_EXPORT_LUT)
                    .clicked()
                {
                    self.lut_export_dialog(ctx);
                }
                if ui
                    .add_enabled(available, egui::Button::new("Build from bundle spec…"))
                    .on_hover_text(tooltips::BUILD_BUNDLE_SPEC)
                    .clicked()
                {
                    self.lut_spec_dialog(ctx);
                }
            })
            .header_response
            .on_hover_text(tooltips::SECTION_LUT_EXPORT);
    }
    fn lut_export_dialog(&mut self, ctx: &egui::Context) {
        let cli = match locate_f64_cli() {
            Ok(path) => path,
            Err(e) => {
                self.status = format!("LUT export: {e}");
                return;
            }
        };
        let settings = self.lut.clone();
        let output = if settings.bundle {
            rfd::FileDialog::new()
                .set_title("Choose a parent folder for the LUT bundle")
                .pick_folder()
                .map(|p| {
                    p.join(format!(
                        "{}_{}_{}_lut",
                        self.film_name, self.print_name, settings.topology
                    ))
                })
        } else {
            let ext = match settings.format.as_str() {
                "hald_png" => "png",
                "3dl" => "3dl",
                _ => "cube",
            };
            rfd::FileDialog::new()
                .add_filter("LUT", &[ext])
                .set_file_name(format!("{}_{}.{}", self.film_name, self.print_name, ext))
                .save_file()
        };
        let Some(output) = output else {
            return;
        };
        let film = self.film_name.clone();
        let paper = self.print_name.clone();
        let params = self.params.clone();
        let data = self.data_dir.clone();
        self.start_export_job(ctx, "f64 LUT", output.clone(), move |cancel| {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos();
            let params_file = TempPath(std::env::temp_dir().join(format!(
                "spektrafilm-lut-{}-{nanos}.json",
                std::process::id()
            )));
            std::fs::write(&params_file.0, serde_json::to_vec(&params)?)?;
            let mut cmd = Command::new(cli);
            cmd.env("SPEKTRAFILM_BACKEND", "cpu")
                .arg("export-lut")
                .arg("--film")
                .arg(film)
                .arg("--paper")
                .arg(paper)
                .arg("--size")
                .arg(settings.size.to_string())
                .arg("--input-space")
                .arg(settings.input)
                .arg("--output-space")
                .arg(settings.output)
                .arg("--exposure-ev")
                .arg(settings.exposure_ev.to_string())
                .arg("--format")
                .arg(settings.format)
                .arg("--params")
                .arg(&params_file.0)
                .arg("--data-dir")
                .arg(data)
                .arg("--output")
                .arg(output);
            if settings.bundle {
                cmd.arg("--topology").arg(settings.topology);
                if settings.combinations {
                    cmd.arg("--include-combinations");
                }
                if settings.qa {
                    cmd.arg("--qa");
                }
            }
            run_cli(&mut cmd, cancel)
        });
    }
    fn lut_spec_dialog(&mut self, ctx: &egui::Context) {
        let Some(spec) = rfd::FileDialog::new()
            .add_filter("Bundle specification", &["json", "toml"])
            .pick_file()
        else {
            return;
        };
        let Some(output) = rfd::FileDialog::new()
            .set_title("Choose bundle output folder")
            .pick_folder()
        else {
            return;
        };
        let cli = match locate_f64_cli() {
            Ok(path) => path,
            Err(e) => {
                self.status = format!("LUT export: {e}");
                return;
            }
        };
        let data: PathBuf = self.data_dir.clone();
        self.start_export_job(ctx, "f64 LUT bundle", output.clone(), move |cancel| {
            run_cli(
                Command::new(cli)
                    .env("SPEKTRAFILM_BACKEND", "cpu")
                    .arg("build-luts")
                    .arg("--from")
                    .arg(spec)
                    .arg("--output")
                    .arg(output)
                    .arg("--data-dir")
                    .arg(data),
                cancel,
            )
        });
    }
}
