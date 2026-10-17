//! Window chrome around the inspector: menu bar and shortcuts, the sidebar
//! header (workflow, stocks, stage tabs), canvas toolbar and overlays, and
//! the status bar. Also owns undo/redo and the inspector's per-frame state.
use super::*;
use crate::inspector::{Stages, Tab};
use crate::{history, inspector, names, theme};
use egui::{Key, KeyboardShortcut, Modifiers, RichText};

const OPEN: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::O);
const OPEN_PROJECT: KeyboardShortcut =
    KeyboardShortcut::new(Modifiers::COMMAND.plus(Modifiers::SHIFT), Key::O);
const OPEN_CATALOG: KeyboardShortcut =
    KeyboardShortcut::new(Modifiers::COMMAND.plus(Modifiers::SHIFT), Key::L);
const SAVE_PROJECT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::S);
const EXPORT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::E);
const UNDO: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::Z);
const REDO: KeyboardShortcut =
    KeyboardShortcut::new(Modifiers::COMMAND.plus(Modifiers::SHIFT), Key::Z);
const REDO_ALT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::Y);
const FIT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::Num0);
const ACTUAL: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::Num1);
const ZOOM_IN: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::Equals);
const ZOOM_OUT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::Minus);
const ROTATE_LEFT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::L);
const ROTATE_RIGHT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::COMMAND, Key::R);
const SPLIT: KeyboardShortcut = KeyboardShortcut::new(Modifiers::NONE, Key::Y);
const FULL_SCREEN: KeyboardShortcut =
    KeyboardShortcut::new(Modifiers::COMMAND.plus(Modifiers::CTRL), Key::F);
/// Plain F, which is also what arrives for macOS fn/Globe+F: the fn key itself
/// never reaches the app.
const FULL_SCREEN_KEY: KeyboardShortcut = KeyboardShortcut::new(Modifiers::NONE, Key::F);
/// Held to show the unprocessed input in place of the render.
const BEFORE_KEY: Key = Key::Backslash;
/// Space reserved under the image for the floating canvas toolbar.
const TOOLBAR_CLEARANCE: f32 = 52.0;
const CANVAS_MARGIN: f32 = 28.0;

const IMAGE_EXTENSIONS: &[&str] = &[
    "png", "jpg", "jpeg", "tif", "tiff", "exr", // standard
    "dng", "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2", "raf", "orf", "rw2", "pef", "srw",
    "x3f", "iiq", "3fr", "crw", "rwl", "mrw", "mef", "kdc",
];

/// Display order of stock groups in the film / paper pickers.
const STOCK_GROUPS: [&str; 7] = [
    "Color negative",
    "Motion picture negative",
    "Black & white negative",
    "Slide (reversal)",
    "Black & white reversal",
    "Photographic paper",
    "Cinema print film",
];

pub(crate) fn stock_group(profile: &spektrafilm_core::profile::Profile) -> &'static str {
    let info = &profile.info;
    if info.stage == "printing" {
        if info.support == "paper" {
            "Photographic paper"
        } else {
            "Cinema print film"
        }
    } else if profile.is_positive() {
        if profile.is_bw() {
            "Black & white reversal"
        } else {
            "Slide (reversal)"
        }
    } else if profile.is_bw() {
        "Black & white negative"
    } else if info.usage == "cine" {
        "Motion picture negative"
    } else {
        "Color negative"
    }
}

impl App {
    // ── Actions ─────────────────────────────────────────────────────────

    pub(crate) fn open_image_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Image", IMAGE_EXTENSIONS)
            .pick_file()
        {
            self.load_image_from_path(&path);
        }
    }

    pub(crate) fn open_catalog_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Lightroom Classic catalog", &["lrcat"])
            .pick_file()
        {
            self.open_catalog(path);
        }
    }

    pub(crate) fn save_preview_enabled(&self) -> bool {
        self.output_image.is_some()
            && self.load_job.is_none()
            && self.render_job.is_none()
            && !self.dirty
            && !self.pending_dirty
    }

    fn export_enabled(&self) -> bool {
        self.image_path.is_some() && self.load_job.is_none() && self.export_job.is_none()
    }

    fn save_project_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Project", &["json"])
            .set_file_name("spektrafilm.json")
            .save_file()
        {
            self.status = match self.project_state().save(&path) {
                Ok(()) => format!("Saved project {}", path.display()),
                Err(e) => format!("Save project error: {e:#}"),
            };
        }
    }

    fn open_project_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Project", &["json"])
            .pick_file()
        {
            self.status = match self.restore_project(&path, true) {
                Ok(()) => format!("Opened project {}", path.display()),
                Err(e) => format!("Open project error: {e:#}"),
            };
            self.history.reset(self.snapshot());
        }
    }

    fn save_defaults(&mut self) {
        let mut state = self.project_state();
        state.image = None;
        state.raw = self.raw_options.clone();
        state.raw.lightroom_geometry = None;
        self.status = match state.save(&project::default_path()) {
            Ok(()) => "Saved startup defaults".into(),
            Err(e) => format!("Save defaults error: {e:#}"),
        };
    }

    fn factory_defaults(&mut self) {
        let input_color = self.params.io.input_color_space.clone();
        let input_encoded = self.params.io.input_cctf_decoding;
        self.params = RuntimeParams::default();
        if self.image.is_some() {
            self.params.io.input_color_space = input_color;
            self.params.io.input_cctf_decoding = input_encoded;
        }
        self.auto_preview = true;
        self.params.settings.preview_max_size = 2048;
        self.gpu_export = true;
        self.lut = Default::default();
        self.display = Default::default();
        self.scan_for_print_restore = None;
        if let Ok(film) = profile::load_profile_by_name(&self.data_dir, &self.film_name) {
            spektrafilm_core::pipeline::apply_stock_defaults(&film, &mut self.params);
        }
        self.raw_options = Default::default();
        self.save_options = Default::default();
        self.pipeline_cache = None;
        self.pipeline_cache_key = None;
        self.dirty = true;
        self.pending_dirty = self.render_job.is_some();
        match std::fs::remove_file(project::default_path()) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                self.status = format!("Reset defaults error: {e}");
            }
        }
    }

    fn zoom_fit(&mut self) {
        self.zoom = 1.0;
        self.pan = egui::Vec2::ZERO;
    }

    fn zoom_actual(&mut self) {
        if self.view_actual_zoom > 0.0 {
            self.pan *= self.view_actual_zoom / self.zoom;
            self.zoom = self.view_actual_zoom;
        }
    }

    /// Zoom about the canvas centre.
    fn zoom_by(&mut self, factor: f32) {
        let old = self.zoom;
        self.zoom = (self.zoom * factor).clamp(0.1, 32.0);
        self.pan *= self.zoom / old;
    }

    fn request_render(&mut self) {
        self.render_requested = true;
        self.dirty = true;
        self.dirty_since = None;
    }

    // ── Undo / inspector frame state ────────────────────────────────────

    pub(crate) fn snapshot(&self) -> history::Snapshot {
        history::Snapshot {
            params: self.params_json.clone(),
            film: self.film_name.clone(),
            print: self.print_name.clone(),
            scan_for_print_restore: self.scan_for_print_restore,
        }
    }

    fn apply_snapshot(&mut self, snapshot: history::Snapshot) {
        let params = match serde_json::from_value::<RuntimeParams>((*snapshot.params).clone()) {
            Ok(params) => params,
            Err(e) => {
                self.status = format!("Undo error: {e}");
                return;
            }
        };
        self.params = params;
        self.params_json = snapshot.params;
        if self.film_name != snapshot.film {
            self.film_name = snapshot.film;
            self.film_dev_times = profile_dev_times(&self.data_dir, &self.film_name);
        }
        if self.print_name != snapshot.print {
            self.print_name = snapshot.print;
            self.print_dev_times = profile_dev_times(&self.data_dir, &self.print_name);
        }
        self.scan_for_print_restore = snapshot.scan_for_print_restore;
        self.dirty = true;
        self.pending_dirty = self.render_job.is_some();
    }

    fn undo(&mut self) {
        match self.history.undo(self.snapshot()) {
            Some(s) => {
                self.apply_snapshot(s);
                self.status = "Undo".into();
            }
            None => self.status = "Nothing to undo".into(),
        }
    }

    fn redo(&mut self) {
        match self.history.redo(self.snapshot()) {
            Some(s) => {
                self.apply_snapshot(s);
                self.status = "Redo".into();
            }
            None => self.status = "Nothing to redo".into(),
        }
    }

    /// Stock defaults for the current film: what resets restore and where the
    /// slider ticks sit.
    pub(crate) fn stock_params(&mut self) -> Arc<RuntimeParams> {
        self.refresh_baseline();
        self.baseline.as_ref().expect("refreshed above").1.clone()
    }

    fn baseline_json(&mut self) -> Arc<serde_json::Value> {
        self.refresh_baseline();
        self.baseline.as_ref().expect("refreshed above").2.clone()
    }

    fn refresh_baseline(&mut self) {
        if self
            .baseline
            .as_ref()
            .is_some_and(|(film, ..)| *film == self.film_name)
        {
            return;
        }
        let mut params = RuntimeParams::default();
        params.settings.preview_max_size = 2048;
        if let Ok(film) = profile::load_profile_by_name(&self.data_dir, &self.film_name) {
            spektrafilm_core::pipeline::apply_stock_defaults(&film, &mut params);
        }
        let value = Arc::new(serde_json::to_value(&params).unwrap_or_default());
        self.baseline = Some((self.film_name.clone(), Arc::new(params), value));
    }

    /// Serialize the params once per frame for undo, change markers and resets,
    /// and commit an undo step once the user has let go of the controls.
    pub(crate) fn begin_frame_state(&mut self, ctx: &egui::Context) {
        self.params_json = Arc::new(serde_json::to_value(&self.params).unwrap_or_default());
        let idle = !ctx.input(|i| i.pointer.any_down()) && !ctx.wants_keyboard_input();
        if idle {
            self.history.observe(self.snapshot());
        }
        let baseline = self.baseline_json();
        inspector::begin_frame(
            self.tab,
            self.display.show_advanced,
            self.params_json.clone(),
            baseline,
            self.learned_rows.clone(),
        );
    }

    /// Learn which params a changed row controls, then apply any resets.
    pub(crate) fn end_frame_state(&mut self) {
        let out = inspector::end_frame();
        if out.changed.is_empty() && out.resets.is_empty() {
            return;
        }
        let mut after = serde_json::to_value(&self.params).unwrap_or_default();
        if let [id] = out.changed[..] {
            let pointers = inspector::diff_pointers(&self.params_json, &after);
            if !pointers.is_empty() {
                Arc::make_mut(&mut self.learned_rows).insert(id, pointers);
            }
        }
        if out.resets.is_empty() {
            return;
        }
        let baseline = self.baseline_json();
        for pointers in &out.resets {
            inspector::reset_pointers(&mut after, &baseline, pointers);
        }
        match serde_json::from_value::<RuntimeParams>(after) {
            Ok(params) => {
                self.params = params;
                self.dirty = true;
                self.status = "Reset to stock defaults".into();
            }
            Err(e) => self.status = format!("Reset error: {e}"),
        }
    }

    // ── Shortcuts and menu bar ──────────────────────────────────────────

    pub(crate) fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        let typing = ctx.wants_keyboard_input();
        let pressed = |s: &KeyboardShortcut| ctx.input_mut(|i| i.consume_shortcut(s));
        // Shifted variants first: shortcut matching ignores extra Shift.
        if pressed(&OPEN_CATALOG) {
            self.open_catalog_dialog();
        }
        if pressed(&OPEN_PROJECT) {
            self.open_project_dialog();
        } else if pressed(&OPEN) {
            self.open_image_dialog();
        }
        if pressed(&SAVE_PROJECT) {
            self.save_project_dialog();
        }
        if pressed(&EXPORT) && self.export_enabled() {
            self.export_dialog(ctx);
        }
        if !typing {
            if pressed(&REDO) || pressed(&REDO_ALT) {
                self.redo();
            } else if pressed(&UNDO) {
                self.undo();
            }
        }
        if pressed(&FIT) {
            self.zoom_fit();
        }
        if pressed(&ACTUAL) {
            self.zoom_actual();
        }
        if pressed(&ZOOM_IN) {
            self.zoom_by(1.25);
        }
        if pressed(&ZOOM_OUT) {
            self.zoom_by(0.8);
        }
        if pressed(&ROTATE_LEFT) {
            self.rotate_input(3);
        }
        if pressed(&ROTATE_RIGHT) {
            self.rotate_input(1);
        }
        if pressed(&FULL_SCREEN) {
            toggle_full_screen(ctx);
        }
        if !typing {
            if pressed(&FULL_SCREEN_KEY) {
                toggle_full_screen(ctx);
            }
            let full_screen = ctx.input(|i| i.viewport().fullscreen.unwrap_or(false));
            if full_screen
                && !ctx.memory(|m| m.any_popup_open())
                && ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Escape))
            {
                ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
            }
            if pressed(&SPLIT) {
                self.display.compare_split = !self.display.compare_split;
            }
        }
        let key_before = !typing && ctx.input(|i| i.key_down(BEFORE_KEY));
        self.compare_hold = key_before || self.before_button_held;
    }

    pub(crate) fn menu_bar(&mut self, ctx: &egui::Context) {
        let shortcut = |s: &KeyboardShortcut| ctx.format_shortcut(s);
        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    ui.set_min_width(220.0);
                    if ui
                        .add(egui::Button::new("Open image…").shortcut_text(shortcut(&OPEN)))
                        .on_hover_text(tooltips::CONTROL_OPEN)
                        .clicked()
                    {
                        ui.close_menu();
                        self.open_image_dialog();
                    }
                    if ui
                        .add(
                            egui::Button::new("Open Lightroom catalog…")
                                .shortcut_text(shortcut(&OPEN_CATALOG)),
                        )
                        .on_hover_text("Browse a Lightroom Classic catalog read-only")
                        .clicked()
                    {
                        ui.close_menu();
                        self.open_catalog_dialog();
                    }
                    if ui
                        .add(
                            egui::Button::new("Open project…")
                                .shortcut_text(shortcut(&OPEN_PROJECT)),
                        )
                        .on_hover_text(tooltips::CONTROL_OPEN_PROJECT)
                        .clicked()
                    {
                        ui.close_menu();
                        self.open_project_dialog();
                    }
                    if ui
                        .add(
                            egui::Button::new("Save project…")
                                .shortcut_text(shortcut(&SAVE_PROJECT)),
                        )
                        .on_hover_text(tooltips::CONTROL_SAVE_PROJECT)
                        .clicked()
                    {
                        ui.close_menu();
                        self.save_project_dialog();
                    }
                    ui.separator();
                    if ui
                        .add_enabled(
                            self.save_preview_enabled(),
                            egui::Button::new("Save preview…"),
                        )
                        .on_hover_text(tooltips::CONTROL_SAVE_PREVIEW)
                        .on_disabled_hover_text("Render a preview first")
                        .clicked()
                    {
                        ui.close_menu();
                        self.save_dialog();
                    }
                    if self.export_job.is_some() {
                        if ui
                            .button("Cancel export")
                            .on_hover_text(tooltips::CANCEL_EXPORT)
                            .clicked()
                        {
                            ui.close_menu();
                            self.cancel_export();
                        }
                    } else if ui
                        .add_enabled(
                            self.export_enabled(),
                            egui::Button::new("Export image…").shortcut_text(shortcut(&EXPORT)),
                        )
                        .on_hover_text(tooltips::EXPORT_IMAGE)
                        .on_disabled_hover_text("Load an image first")
                        .clicked()
                    {
                        ui.close_menu();
                        self.export_dialog(ctx);
                    }
                    if ui
                        .button("LUT export settings…")
                        .on_hover_text(tooltips::SECTION_LUT_EXPORT)
                        .clicked()
                    {
                        ui.close_menu();
                        self.tab = Tab::Output;
                    }
                    ui.separator();
                    if ui
                        .button("Save as startup defaults")
                        .on_hover_text(tooltips::CONTROL_SAVE_DEFAULTS)
                        .clicked()
                    {
                        ui.close_menu();
                        self.save_defaults();
                    }
                    if ui
                        .button("Reset to factory defaults")
                        .on_hover_text(tooltips::CONTROL_FACTORY_DEFAULTS)
                        .clicked()
                    {
                        ui.close_menu();
                        self.factory_defaults();
                    }
                });
                ui.menu_button("Edit", |ui| {
                    ui.set_min_width(200.0);
                    if ui
                        .add_enabled(
                            self.history.can_undo(),
                            egui::Button::new("Undo").shortcut_text(shortcut(&UNDO)),
                        )
                        .clicked()
                    {
                        ui.close_menu();
                        self.undo();
                    }
                    if ui
                        .add_enabled(
                            self.history.can_redo(),
                            egui::Button::new("Redo").shortcut_text(shortcut(&REDO)),
                        )
                        .clicked()
                    {
                        ui.close_menu();
                        self.redo();
                    }
                });
                ui.menu_button("View", |ui| {
                    ui.set_min_width(240.0);
                    if ui
                        .add(egui::Button::new("Fit to window").shortcut_text(shortcut(&FIT)))
                        .clicked()
                    {
                        ui.close_menu();
                        self.zoom_fit();
                    }
                    if ui
                        .add(egui::Button::new("Actual pixels").shortcut_text(shortcut(&ACTUAL)))
                        .clicked()
                    {
                        ui.close_menu();
                        self.zoom_actual();
                    }
                    if ui
                        .add(egui::Button::new("Zoom in").shortcut_text(shortcut(&ZOOM_IN)))
                        .clicked()
                    {
                        self.zoom_by(1.25);
                    }
                    if ui
                        .add(egui::Button::new("Zoom out").shortcut_text(shortcut(&ZOOM_OUT)))
                        .clicked()
                    {
                        self.zoom_by(0.8);
                    }
                    let full_screen = ctx.input(|i| i.viewport().fullscreen.unwrap_or(false));
                    if ui
                        .add(
                            egui::Button::new(if full_screen {
                                "Exit Full Screen"
                            } else {
                                "Enter Full Screen"
                            })
                            .shortcut_text(shortcut(&FULL_SCREEN_KEY)),
                        )
                        .on_hover_text(format!(
                            "Also fn+F or {}. Esc leaves full screen.",
                            shortcut(&FULL_SCREEN)
                        ))
                        .clicked()
                    {
                        ui.close_menu();
                        toggle_full_screen(ctx);
                    }
                    ui.separator();
                    ui.add(egui::Checkbox::new(
                        &mut self.display.compare_split,
                        "Before / after split",
                    ))
                    .on_hover_text(format!(
                        "{}\nShortcut: {}. Hold \\ to see the input alone.",
                        tooltips::CONTROL_COMPARE_INPUT_OUTPUT,
                        shortcut(&SPLIT)
                    ));
                    if ui
                        .add_enabled(
                            self.image.is_some(),
                            egui::Button::new("Rotate left").shortcut_text(shortcut(&ROTATE_LEFT)),
                        )
                        .clicked()
                    {
                        ui.close_menu();
                        self.rotate_input(3);
                    }
                    if ui
                        .add_enabled(
                            self.image.is_some(),
                            egui::Button::new("Rotate right")
                                .shortcut_text(shortcut(&ROTATE_RIGHT)),
                        )
                        .clicked()
                    {
                        ui.close_menu();
                        self.rotate_input(1);
                    }
                    ui.separator();
                    ui.checkbox(&mut self.auto_preview, "Auto preview")
                        .on_hover_text(tooltips::CONTROL_AUTO_PREVIEW);
                    ui.checkbox(&mut self.display.show_advanced, "Show advanced controls")
                        .on_hover_text(
                            "Show expert parameters: emulsion physics, inhibition matrices, \
                             spectral LUTs and other parity controls.",
                        );
                    ui.separator();
                    ui.checkbox(&mut self.display.show_border, "Print border")
                        .on_hover_text(tooltips::CONTROL_WHITE_BORDER_LAYER);
                    ui.checkbox(&mut self.display.show_paper_back, "Paper texture")
                        .on_hover_text(tooltips::CONTROL_PAPER_BACK_LAYER);
                    ui.checkbox(&mut self.display.gray_18_canvas, "18% gray canvas")
                        .on_hover_text(tooltips::CONTROL_18_GRAY_CANVAS);
                    ui.checkbox(&mut self.display.animate, "Paper development animation")
                        .on_hover_text(tooltips::CONTROL_PAPER_DEVELOPMENT_CROSSFADE);
                });
            });
        });
    }

    // ── Sidebar header: workflow, stocks, stage tabs ────────────────────

    pub(crate) fn sidebar_header(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        let route = self.params.route().to_owned();
        let mut selected = route.clone();
        tooltips::combo(
            ui,
            egui::ComboBox::from_id_salt("workflow").selected_text(names::workflow(&route)),
            "Workflow",
            "Choose whether to simulate film, convert a scanned negative, or pass the input through.",
            |ui| {
                let mut group = "";
                for (heading, value, name) in names::WORKFLOWS {
                    if heading != group {
                        group = heading;
                        ui.label(RichText::new(heading).small().color(theme::TEXT_WEAK));
                    }
                    ui.selectable_value(&mut selected, value.to_owned(), name)
                        .on_hover_text(match value {
                            "input" => "Convert input to the output color space without film simulation.",
                            "input > film > scan" => "Expose and develop film, then scan it directly.",
                            "input > film > print > scan" => {
                                "Expose film, print it onto paper, then scan the print."
                            }
                            "input > convert-film > print > scan" => {
                                "Recover dye densities from a scanned negative and print them."
                            }
                            "input > convert-film > scan-minus-base" => {
                                "Recover film dye densities and scan them without the film base."
                            }
                            _ => "Recover film dye densities and scan them under the selected scan illuminant.",
                        });
                }
            },
        );
        if selected != route {
            self.params.workflow.route = selected;
            self.params.digest();
            self.dirty = true;
        }
        ui.add_space(4.0);

        let film_changed = stock_picker(ui, "film", "Film", &self.films, &mut self.film_name, None);
        if film_changed {
            self.film_changed();
        }
        // B&W stocks are profiled at several development times — pick one
        // (longer = more contrast). Hidden for the single-time / colour case.
        if self.film_dev_times.len() > 1
            && dev_time_combo(
                ui,
                "film_dev_time",
                "Development time",
                &self.film_dev_times.clone(),
                &mut self.params.film_render.chemistry.development_time,
            )
        {
            self.dirty = true;
        }
        // Slide films are scanned directly, so the print paper is unused —
        // disable the picker and say why rather than silently affecting nothing.
        let paper_note = self
            .params
            .scans_film()
            .then_some("Not used: the film is scanned directly");
        let paper_changed = ui
            .add_enabled_ui(!self.params.scans_film(), |ui| {
                stock_picker(
                    ui,
                    "paper",
                    "Paper",
                    &self.papers,
                    &mut self.print_name,
                    paper_note,
                )
            })
            .inner;
        if paper_changed {
            self.print_dev_times = profile_dev_times(&self.data_dir, &self.print_name);
            self.params.print_render.chemistry.development_time = None;
            self.dirty = true;
        }
        if !self.params.scans_film()
            && self.print_dev_times.len() > 1
            && dev_time_combo(
                ui,
                "print_dev_time",
                "Print development time",
                &self.print_dev_times.clone(),
                &mut self.params.print_render.chemistry.development_time,
            )
        {
            self.dirty = true;
        }
        ui.add_space(8.0);
        self.stage_tabs(ui);
        ui.add_space(2.0);
    }

    /// The pipeline as tabs: Input › Film › Print › Scan, then Output. Stages the
    /// workflow skips stay clickable but are drawn faint.
    fn stage_tabs(&mut self, ui: &mut egui::Ui) {
        let stages = Stages::of(&self.params);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            for (i, tab) in Tab::ALL.into_iter().enumerate() {
                if i > 0 {
                    let sep = if tab == Tab::Output { "  " } else { "›" };
                    ui.label(RichText::new(sep).color(theme::TEXT_FAINT));
                }
                let used = stages.tab_used(tab);
                let selected = self.tab == tab;
                let mut text = RichText::new(tab.name());
                if selected {
                    text = text.family(egui::FontFamily::Name(theme::SEMIBOLD.into()));
                } else if !used {
                    text = text.color(theme::TEXT_FAINT);
                }
                let response = ui.selectable_label(selected, text).on_hover_text(if used {
                    tab_help(tab)
                } else {
                    "Not used by the current workflow."
                });
                if response.clicked() {
                    self.tab = tab;
                }
            }
        });
    }

    // ── Canvas ──────────────────────────────────────────────────────────

    /// The canvas rect left for the image once margins and the toolbar are
    /// reserved, so a fitted print never sits under the toolbar.
    pub(crate) fn image_area(rect: egui::Rect) -> egui::Rect {
        let mut area = rect.shrink(CANVAS_MARGIN);
        area.max.y = (area.max.y - TOOLBAR_CLEARANCE + CANVAS_MARGIN).max(area.min.y + 1.0);
        area
    }

    pub(crate) fn empty_canvas(&mut self, ui: &mut egui::Ui) {
        let loading = self.load_job.is_some();
        ui.vertical_centered(|ui| {
            ui.add_space((ui.available_height() * 0.5 - 80.0).max(0.0));
            if loading {
                ui.spinner();
                ui.label(RichText::new("Loading image…").color(theme::TEXT_WEAK));
                return;
            }
            if self.image.is_some() {
                ui.label(theme_heading("Ready to render"));
                ui.add_space(8.0);
                if ui
                    .add(
                        egui::Button::new(RichText::new("Render preview").size(15.0))
                            .min_size(egui::vec2(160.0, 32.0)),
                    )
                    .on_hover_text(tooltips::CONTROL_RENDER_PREVIEW)
                    .clicked()
                {
                    self.request_render();
                }
                return;
            }
            ui.label(theme_heading("Open an image to begin"));
            ui.add_space(10.0);
            if ui
                .add(
                    egui::Button::new(RichText::new("Open image…").size(15.0))
                        .min_size(egui::vec2(160.0, 32.0)),
                )
                .on_hover_text(tooltips::CONTROL_OPEN)
                .clicked()
            {
                self.open_image_dialog();
            }
            ui.add_space(8.0);
            ui.label(
                RichText::new("or drop a file here · RAW, TIFF, PNG, JPEG, EXR")
                    .color(theme::TEXT_WEAK),
            );
        });
    }

    /// Floating toolbar, progress indicator and drop target over the canvas.
    pub(crate) fn canvas_overlays(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let ctx = ui.ctx().clone();
        let painter = ui.painter_at(rect);

        // Render / load progress, top-right, once a job has run long enough to notice.
        let busy = match (&self.load_job, &self.render_job) {
            (Some(job), _) => Some(("Loading", job.started)),
            (None, Some(job)) => Some(("Rendering", job.started)),
            _ => None,
        };
        if let Some((what, started)) = busy
            && self.has_canvas_image()
        {
            if started.elapsed() > Duration::from_millis(150) {
                egui::Area::new(egui::Id::new("canvas-progress"))
                    .order(egui::Order::Middle)
                    .fixed_pos(rect.right_top() + egui::vec2(-12.0, 12.0))
                    .pivot(egui::Align2::RIGHT_TOP)
                    .interactable(false)
                    .show(&ctx, |ui| {
                        overlay_frame().show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.add(egui::Spinner::new().size(12.0));
                                ui.label(RichText::new(what).small());
                            });
                        });
                    });
            } else {
                ctx.request_repaint_after(Duration::from_millis(150));
            }
        }

        if self.has_canvas_image() {
            egui::Area::new(egui::Id::new("canvas-toolbar"))
                .order(egui::Order::Middle)
                .fixed_pos(rect.center_bottom() - egui::vec2(0.0, 12.0))
                .pivot(egui::Align2::CENTER_BOTTOM)
                .show(&ctx, |ui| {
                    overlay_frame().show(ui, |ui| {
                        ui.horizontal(|ui| self.toolbar(ui));
                    });
                });
        }

        // Drop target while a file is dragged over the window.
        if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
            let target = rect.shrink(10.0);
            painter.rect_filled(target, 8.0, theme::ACCENT.gamma_multiply(0.12));
            painter.rect_stroke(
                target,
                8.0,
                egui::Stroke::new(2.0, theme::ACCENT),
                egui::StrokeKind::Inside,
            );
            painter.text(
                target.center(),
                egui::Align2::CENTER_CENTER,
                "Drop to open",
                egui::FontId::new(18.0, egui::FontFamily::Name(theme::SEMIBOLD.into())),
                theme::TEXT,
            );
        }
    }

    pub(crate) fn has_canvas_image(&self) -> bool {
        self.canvas_input.is_some()
            || self.input_tex.is_some()
            || self.canvas_output.is_some()
            || self.output_tex.is_some()
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.spacing_mut().item_spacing.x = 4.0;
        ui.spacing_mut().button_padding = egui::vec2(7.0, 3.0);
        let fit = (self.zoom - 1.0).abs() < 1e-3 && self.pan.length_sq() < 1.0;
        if ui
            .selectable_label(fit, "Fit")
            .on_hover_text(format!(
                "Fit the print to the window ({})",
                ui.ctx().format_shortcut(&FIT)
            ))
            .clicked()
        {
            self.zoom_fit();
        }
        let actual =
            self.view_actual_zoom > 0.0 && (self.zoom / self.view_actual_zoom - 1.0).abs() < 1e-3;
        if ui
            .selectable_label(actual, "100%")
            .on_hover_text(format!(
                "One image pixel per screen pixel ({}). The preview is rendered at {} px on the long edge; raise it in View for more detail.",
                ui.ctx().format_shortcut(&ACTUAL),
                self.rendered_preview_limit
            ))
            .clicked()
        {
            self.zoom_actual();
        }
        if ui.button("−").on_hover_text("Zoom out").clicked() {
            self.zoom_by(0.8);
        }
        let pct = if self.view_actual_zoom > 0.0 {
            (self.zoom / self.view_actual_zoom * 100.0).round()
        } else {
            100.0
        };
        ui.add_sized(
            [44.0, ui.spacing().interact_size.y],
            egui::Label::new(RichText::new(format!("{pct:.0}%")).monospace()),
        )
        .on_hover_text("Zoom relative to the source image. Scroll or pinch to zoom, drag to pan, double-click to fit.");
        if ui.button("+").on_hover_text("Zoom in").clicked() {
            self.zoom_by(1.25);
        }
        toolbar_divider(ui);

        let before = ui
            .add(egui::Button::new("Before").selected(self.compare_hold))
            .on_hover_text("Hold to show the unprocessed input (or hold \\).");
        self.before_button_held = before.is_pointer_button_down_on();
        ui.toggle_value(&mut self.display.compare_split, "Split")
            .on_hover_text(format!(
                "{} Drag the divider on the canvas to move it. ({})",
                tooltips::CONTROL_COMPARE_INPUT_OUTPUT,
                ui.ctx().format_shortcut(&SPLIT)
            ));
        toolbar_divider(ui);

        let has_image = self.image.is_some();
        if ui
            .add_enabled(has_image, icons::RotateButton::new(false).compact())
            .on_hover_text(tooltips::CONTROL_ROTATE)
            .clicked()
        {
            self.rotate_input(3);
        }
        if ui
            .add_enabled(has_image, icons::RotateButton::new(true).compact())
            .on_hover_text(tooltips::CONTROL_ROTATE_2)
            .clicked()
        {
            self.rotate_input(1);
        }
        ui.toggle_value(&mut self.display.show_border, "Border")
            .on_hover_text(tooltips::CONTROL_WHITE_BORDER_LAYER);
        toolbar_divider(ui);
        ui.menu_button("View…", |ui| self.view_menu(ui))
            .response
            .on_hover_text(tooltips::SECTION_DISPLAY);
        if !self.auto_preview {
            toolbar_divider(ui);
            let stale = self.dirty || self.pending_dirty;
            if ui
                .add_enabled(
                    has_image && self.render_job.is_none(),
                    egui::Button::new(RichText::new("Render").color(if stale {
                        theme::ACCENT
                    } else {
                        theme::TEXT
                    })),
                )
                .on_hover_text(tooltips::CONTROL_RENDER_PREVIEW)
                .clicked()
            {
                self.request_render();
            }
        }
    }

    /// Display settings: how the render is shown, never what it contains
    /// (apart from the preview resolution, which only bounds the preview).
    fn view_menu(&mut self, ui: &mut egui::Ui) {
        ui.set_min_width(300.0);
        if tooltips::labeled(
            ui,
            egui::Slider::new(&mut self.params.settings.preview_max_size, 128..=8192).step_by(128.),
            "Preview long edge (px)",
            tooltips::CONTROL_PREVIEW_LONG_EDGE_PX,
        )
        .changed()
        {
            self.dirty = true;
        }
        if tooltips::checkbox(
            ui,
            &mut self.display.display_transform,
            "Display color transform",
            tooltips::CONTROL_DISPLAY_COLOR_TRANSFORM,
        )
        .changed()
        {
            self.dirty = true;
        }
        ui.separator();
        tooltips::checkbox(
            ui,
            &mut self.display.show_border,
            "White border layer",
            tooltips::CONTROL_WHITE_BORDER_LAYER,
        );
        if self.display.show_border {
            tooltips::labeled(
                ui,
                egui::Slider::new(&mut self.display.white_padding, 0.0..=0.2),
                "White border",
                tooltips::CONTROL_WHITE_BORDER,
            );
        }
        tooltips::checkbox(
            ui,
            &mut self.display.show_paper_back,
            "Paper-back layer",
            tooltips::CONTROL_PAPER_BACK_LAYER,
        );
        tooltips::checkbox(
            ui,
            &mut self.display.show_input,
            "Input layer",
            tooltips::CONTROL_INPUT_LAYER,
        );
        if self.display.show_input {
            tooltips::labeled(
                ui,
                egui::Slider::new(&mut self.display.input_opacity, 0.0..=1.0),
                "Input opacity",
                tooltips::CONTROL_INPUT_OPACITY,
            );
        }
        tooltips::checkbox(
            ui,
            &mut self.display.show_output,
            "Output layer",
            tooltips::CONTROL_OUTPUT_LAYER,
        );
        if self.display.show_output {
            tooltips::labeled(
                ui,
                egui::Slider::new(&mut self.display.output_opacity, 0.0..=1.0),
                "Output opacity",
                tooltips::CONTROL_OUTPUT_OPACITY,
            );
        }
        if self.display.compare_split {
            tooltips::labeled(
                ui,
                egui::Slider::new(&mut self.display.split_position, 0.0..=1.0),
                "Comparison split",
                tooltips::CONTROL_COMPARISON_SPLIT,
            );
        }
        ui.separator();
        tooltips::checkbox(
            ui,
            &mut self.display.gray_18_canvas,
            "18% gray canvas",
            tooltips::CONTROL_18_GRAY_CANVAS,
        );
        tooltips::checkbox(
            ui,
            &mut self.display.animate,
            "Paper development / crossfade",
            tooltips::CONTROL_PAPER_DEVELOPMENT_CROSSFADE,
        );
        if ui
            .add_enabled(
                self.canvas_output.is_some(),
                egui::Button::new("Replay paper development"),
            )
            .on_hover_text(tooltips::CONTROL_REPLAY_PAPER_DEVELOPMENT)
            .on_disabled_hover_text(tooltips::CONTROL_REPLAY_PAPER_DEVELOPMENT)
            .clicked()
        {
            self.canvas_previous = None;
            self.animation = Some((Instant::now(), true));
            self.ctx.request_repaint();
        }
        tooltips::combo(
            ui,
            egui::ComboBox::from_id_salt("canvas_interpolation")
                .selected_text(names::label(&self.display.output_interpolation)),
            "Preview sampling",
            tooltips::MENU_CANVAS_INTERPOLATION,
            |ui| {
                for mode in view::INTERPOLATIONS {
                    if ui
                        .selectable_value(
                            &mut self.display.output_interpolation,
                            mode.into(),
                            names::label(mode),
                        )
                        .on_hover_text(tooltips::option(mode))
                        .changed()
                        && self.canvas.is_none()
                    {
                        self.dirty = true;
                    }
                }
            },
        );
        if self.canvas.is_none() {
            ui.label(
                RichText::new(tooltips::GPU_CANVAS_NOTE)
                    .small()
                    .color(theme::TEXT_WEAK),
            );
        }
    }

    // ── Status bar and window title ─────────────────────────────────────

    pub(crate) fn status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status")
            .frame(
                egui::Frame::side_top_panel(&ctx.style())
                    .inner_margin(egui::Margin::symmetric(10, 4)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 10.0;
                    let weak = |s: String| RichText::new(s).small().color(theme::TEXT_WEAK);
                    // Right-hand facts first so the status message takes the rest.
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if self.last_worker_total_ms > 0.0 {
                            ui.label(weak(format!(
                                "{} · {:.0} ms",
                                self.render_path, self.last_worker_total_ms
                            )))
                            .on_hover_text(self.timing_report());
                        } else {
                            ui.label(weak(self.backend.name().to_string()));
                        }
                        if let Some(out) = &self.output_image {
                            ui.label(weak(format!("preview {} × {}", out.width, out.height)));
                        }
                        if let Some(image) = &self.image {
                            ui.label(weak(format!(
                                "{} × {} ({:.1} MP)",
                                image.width,
                                image.height,
                                image.pixel_count() as f64 / 1e6
                            )));
                        }
                        if let Some(name) = self.image_path.as_ref().and_then(|p| p.file_name()) {
                            ui.label(
                                RichText::new(name.to_string_lossy())
                                    .small()
                                    .color(theme::TEXT),
                            );
                        }
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            if self.load_job.is_some()
                                || self.render_job.is_some()
                                || self.export_job.is_some()
                                || self.calibration_job.is_some()
                            {
                                ui.add(egui::Spinner::new().size(12.0));
                            }
                            if self.export_job.is_some()
                                && ui
                                    .small_button("Cancel")
                                    .on_hover_text(tooltips::CANCEL_EXPORT)
                                    .clicked()
                            {
                                self.cancel_export();
                            }
                            let color = if status_is_error(&self.status) {
                                ui.visuals().error_fg_color
                            } else {
                                theme::TEXT_CAPTION
                            };
                            ui.add(
                                egui::Label::new(RichText::new(&self.status).small().color(color))
                                    .truncate(),
                            )
                            .on_hover_text(&self.status);
                        });
                    });
                });
            });
    }

    fn timing_report(&self) -> String {
        format!(
            "Render {:.1} ms ({})\nPipeline build {:.1} ms\nClone / scale {:.1} / {:.1} ms\nPreview pack {:.1} ms\nWorker total {:.1} ms\nBackend: {}\nExecution: {}",
            self.last_render_ms,
            fps_label(self.last_render_ms),
            self.last_pipeline_build_ms,
            self.last_input_clone_ms,
            self.last_scale_ms,
            self.last_preview_ms,
            self.last_worker_total_ms,
            self.backend.name(),
            self.render_path,
        )
    }

    pub(crate) fn update_title(&mut self, ctx: &egui::Context) {
        let title = match self.image_path.as_ref().and_then(|p| p.file_name()) {
            Some(name) => format!("{} — spektrafilm", name.to_string_lossy()),
            None => "spektrafilm".to_string(),
        };
        if title != self.window_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.window_title = title;
        }
    }
}

/// Searchable, grouped stock picker showing the selected stock prominently.
/// Returns true when the selection changed.
fn stock_picker(
    ui: &mut egui::Ui,
    salt: &str,
    caption: &str,
    entries: &[ProfileEntry],
    selected: &mut String,
    note: Option<&str>,
) -> bool {
    let help = if salt == "film" {
        tooltips::FILM_PROFILE
    } else {
        tooltips::PAPER_PROFILE
    };
    let current = entries.iter().find(|e| &e.stock == selected);
    let name = current.map_or(selected.as_str(), |e| e.display.as_str());
    let subtitle = note.unwrap_or(current.map_or("", |e| e.group));
    let popup_id = ui.make_persistent_id(("stock-picker", salt));
    let search_id = popup_id.with("search");

    let width = ui.available_width();
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, 44.0), egui::Sense::click());
    let response = response.on_hover_text(help);
    if ui.is_rect_visible(rect) {
        let open = ui.memory(|m| m.is_popup_open(popup_id));
        let visuals = ui.style().interact_selectable(&response, open);
        ui.painter().rect_filled(rect, 5.0, visuals.weak_bg_fill);
        let text_color = if ui.is_enabled() {
            theme::TEXT
        } else {
            theme::TEXT_FAINT
        };
        ui.painter().text(
            rect.left_top() + egui::vec2(10.0, 5.0),
            egui::Align2::LEFT_TOP,
            caption.to_uppercase(),
            egui::FontId::proportional(9.5),
            theme::TEXT_FAINT,
        );
        ui.painter().text(
            rect.right_top() + egui::vec2(-10.0, 5.0),
            egui::Align2::RIGHT_TOP,
            subtitle,
            egui::FontId::proportional(9.5),
            theme::TEXT_WEAK,
        );
        let name_galley = ui.painter().layout(
            name.to_string(),
            egui::FontId::new(15.0, egui::FontFamily::Name(theme::SEMIBOLD.into())),
            text_color,
            rect.width() - 36.0,
        );
        ui.painter().galley(
            rect.left_bottom() + egui::vec2(10.0, -6.0 - name_galley.size().y),
            name_galley,
            text_color,
        );
        // Drawn rather than typed: the UI fonts lack a small down-triangle glyph.
        let c = rect.right_bottom() + egui::vec2(-16.0, -16.0);
        ui.painter().add(egui::Shape::convex_polygon(
            vec![
                c + egui::vec2(-4.5, -2.5),
                c + egui::vec2(4.5, -2.5),
                c + egui::vec2(0.0, 3.0),
            ],
            theme::TEXT_WEAK,
            egui::Stroke::NONE,
        ));
    }
    if response.clicked() {
        ui.memory_mut(|m| m.toggle_popup(popup_id));
        ui.data_mut(|d| d.insert_temp(search_id, String::new()));
        ui.memory_mut(|m| m.request_focus(search_id));
    }

    let mut changed = false;
    egui::popup_below_widget(
        ui,
        popup_id,
        &response,
        egui::PopupCloseBehavior::CloseOnClickOutside,
        |ui| {
            ui.set_width(width - 12.0);
            let mut query: String = ui.data_mut(|d| d.get_temp(search_id).unwrap_or_default());
            let search = ui.add(
                egui::TextEdit::singleline(&mut query)
                    .id(search_id)
                    .hint_text("Search stocks…")
                    .desired_width(f32::INFINITY),
            );
            ui.data_mut(|d| d.insert_temp(search_id, query.clone()));
            let needle = query.to_lowercase();
            let matches: Vec<&ProfileEntry> = entries
                .iter()
                .filter(|e| {
                    needle.is_empty()
                        || e.display.to_lowercase().contains(&needle)
                        || e.stock.contains(&needle)
                        || e.group.to_lowercase().contains(&needle)
                })
                .collect();
            if search.lost_focus()
                && ui.input(|i| i.key_pressed(Key::Enter))
                && let Some(first) = matches.first()
            {
                changed |= *selected != first.stock;
                *selected = first.stock.clone();
                ui.memory_mut(|m| m.close_popup());
            }
            if ui.input(|i| i.key_pressed(Key::Escape)) {
                ui.memory_mut(|m| m.close_popup());
            }
            egui::ScrollArea::vertical()
                .max_height(360.0)
                .show(ui, |ui| {
                    for group in STOCK_GROUPS {
                        let members: Vec<&&ProfileEntry> =
                            matches.iter().filter(|e| e.group == group).collect();
                        if members.is_empty() {
                            continue;
                        }
                        ui.add_space(4.0);
                        ui.label(RichText::new(group).small().color(theme::TEXT_WEAK));
                        for entry in members {
                            if ui
                                .add_sized(
                                    [ui.available_width(), 22.0],
                                    egui::SelectableLabel::new(
                                        *selected == entry.stock,
                                        &entry.display,
                                    ),
                                )
                                .clicked()
                            {
                                changed |= *selected != entry.stock;
                                *selected = entry.stock.clone();
                                ui.memory_mut(|m| m.close_popup());
                            }
                        }
                    }
                    if matches.is_empty() {
                        ui.label(RichText::new("No matching stock").color(theme::TEXT_WEAK));
                    }
                });
        },
    );
    ui.add_space(4.0);
    changed
}

fn toggle_full_screen(ctx: &egui::Context) {
    let full_screen = ctx.input(|i| i.viewport().fullscreen.unwrap_or(false));
    ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(!full_screen));
}

fn tab_help(tab: Tab) -> &'static str {
    match tab {
        Tab::Input => {
            "Camera exposure, lens and filters, RAW decoding, and scanned-negative conversion."
        }
        Tab::Film => "Film emulsion: halation, DIR couplers, grain, chemistry and base.",
        Tab::Print => "Enlarger, print paper chemistry and base, and glare.",
        Tab::Scan => "The scanner that digitizes the film or print.",
        Tab::Output => "Color management, output encoding, file and LUT export.",
    }
}

fn theme_heading(text: &str) -> RichText {
    inspector::title(text).size(18.0).color(theme::TEXT)
}

fn overlay_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(egui::Color32::from_rgba_unmultiplied(24, 24, 24, 228))
        .stroke(egui::Stroke::new(1.0, egui::Color32::from_gray(58)))
        .corner_radius(8.0)
        .inner_margin(egui::Margin::symmetric(8, 5))
        .shadow(egui::epaint::Shadow {
            offset: [0, 3],
            blur: 10,
            spread: 0,
            color: egui::Color32::from_black_alpha(90),
        })
}

fn toolbar_divider(ui: &mut egui::Ui) {
    ui.add_space(2.0);
    let height = ui.spacing().interact_size.y - 6.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(1.0, height), egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, 0.0, egui::Color32::from_gray(64));
    ui.add_space(2.0);
}

fn status_is_error(status: &str) -> bool {
    let s = status.to_ascii_lowercase();
    s.contains("error") || s.contains("failed") || s.contains("could not")
}
