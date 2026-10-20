//! Short explanations shared by dropdown controls and their menu items, plus
//! the inspector row helpers that lay controls out as label | value columns.

use crate::{inspector, theme};
use eframe::egui::{
    self, Align, Label, Layout, Response, RichText, Sense, TextWrapMode, Ui, Widget, WidgetText,
};

/// Width of the caption column, so every control's value starts on one edge.
fn caption_width(ui: &Ui) -> f32 {
    (ui.available_width() * 0.42).clamp(96.0, 176.0)
}

/// Split a caption such as `Lens blur (µm)` or `Tail weight: ` into the name
/// and a unit hint drawn in a weaker tone.
fn caption_parts(text: &str) -> (&str, Option<&str>) {
    let text = text.trim_end().trim_end_matches(':').trim_end();
    match text.rfind(" (") {
        Some(at) if text.ends_with(')') => (&text[..at], Some(&text[at + 2..text.len() - 1])),
        _ => (text, None),
    }
}

/// The caption cell: a change-marker gutter plus the name.
struct Caption {
    response: Response,
    marker: egui::Pos2,
}

impl Caption {
    fn paint_marker(&self, ui: &Ui, modified: bool) {
        if modified {
            ui.painter().circle_filled(self.marker, 2.5, theme::ACCENT);
        }
    }
}

fn caption(ui: &mut Ui, text: &str, width: f32, sense: Sense) -> Caption {
    let (name, unit) = caption_parts(text);
    let height = ui.spacing().interact_size.y;
    ui.allocate_ui_with_layout(
        egui::vec2(width, height),
        Layout::left_to_right(Align::Center),
        |ui| {
            ui.set_min_width(width);
            ui.spacing_mut().item_spacing.x = 4.0;
            let (dot, _) = ui.allocate_exact_size(egui::vec2(6.0, height), Sense::hover());
            let mut job = egui::text::LayoutJob::default();
            let body = egui::TextStyle::Body.resolve(ui.style());
            job.append(
                name,
                0.0,
                egui::TextFormat::simple(body.clone(), theme::TEXT_CAPTION),
            );
            if let Some(unit) = unit {
                job.append(unit, 5.0, egui::TextFormat::simple(body, theme::TEXT_FAINT));
            }
            let response = ui.add(Label::new(job).selectable(false).truncate().sense(sense));
            Caption {
                response,
                marker: dot.center(),
            }
        },
    )
    .inner
}

/// The caption's context menu offers "Reset to stock default"; true when chosen.
fn reset_requested(label: &Response) -> bool {
    let mut chosen = false;
    label.context_menu(|ui| {
        if ui.button("Reset to stock default").clicked() {
            chosen = true;
            ui.close_menu();
        }
    });
    chosen
}

/// Hover help, with the reset gesture spelled out once the row has changed.
fn attach_help(label: Response, help: &str, modified: bool, double_click_reset: bool) {
    if modified {
        let hint = if double_click_reset {
            "Changed from the stock default. Double-click the name to reset."
        } else {
            "Changed from the stock default. Right-click the name to reset."
        };
        label
            .on_hover_text(format!("{help}\n\n{hint}"))
            .on_disabled_hover_text(help);
    } else {
        label.on_hover_text(help).on_disabled_hover_text(help);
    }
}

/// One inspector row: caption on the left, the value editor filling the rest.
/// Help lives on the caption, never on the value editor. Double-clicking the
/// caption restores the stock value once the row is known to differ from it.
pub fn labeled(
    ui: &mut Ui,
    widget: impl Widget,
    text: impl Into<WidgetText>,
    help: &str,
) -> Response {
    let text = text.into();
    let id = inspector::row_id(text.text());
    let learned_modified = inspector::row_modified(id);
    ui.horizontal(|ui| {
        // DragValue (including the slider's numeric editor) has its own hover
        // instructions; keep those off the editable area as well.
        ui.style_mut().explanation_tooltips = false;
        let width = caption_width(ui);
        let label = caption(ui, text.text(), width, Sense::click());
        let reset = label.response.double_clicked() || reset_requested(&label.response);
        inspector::begin_row(reset);
        if reset && inspector::row_stock_modified().is_none() {
            inspector::request_row_reset(id);
        }
        // Sliders size from `slider_width`; leave room for their value box.
        let spacing = ui.spacing().clone();
        ui.spacing_mut().slider_width =
            (ui.available_width() - spacing.interact_size.x - spacing.item_spacing.x - 2.0)
                .max(48.0);
        let response = ui.add(widget).labelled_by(label.response.id);
        let modified = inspector::row_stock_modified().unwrap_or(learned_modified);
        label.paint_marker(ui, modified);
        attach_help(label.response, help, modified, true);
        if response.changed() {
            inspector::note_changed(id, text.text());
        }
        response
    })
    .inner
}

/// A full-width checkbox row. Clicking the caption still toggles the box, but
/// only the caption shows help.
pub fn checkbox(
    ui: &mut Ui,
    value: &mut bool,
    text: impl Into<WidgetText>,
    help: &str,
) -> Response {
    let text = text.into();
    let caption = text.text().to_owned();
    let id = inspector::row_id(&caption);
    let modified = inspector::row_modified(id);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        let height = ui.spacing().interact_size.y;
        let (dot, _) = ui.allocate_exact_size(egui::vec2(6.0, height), Sense::hover());
        if modified {
            ui.painter().circle_filled(dot.center(), 2.5, theme::ACCENT);
        }
        let mut response = ui.checkbox(value, "");
        let label = ui.add(
            Label::new(text)
                .selectable(false)
                .sense(Sense::click())
                .wrap_mode(TextWrapMode::Extend),
        );
        if label.clicked() {
            *value = !*value;
            response.mark_changed();
            ui.ctx().request_repaint();
        }
        let response = response.labelled_by(label.id).union(label.clone());
        if reset_requested(&label) {
            inspector::request_row_reset(id);
        }
        attach_help(label, help, modified, false);
        if response.changed() {
            inspector::note_changed(id, &caption);
        }
        response
    })
    .inner
}

/// A dropdown row: caption on the left, the combo filling the value column.
pub fn combo<R>(
    ui: &mut Ui,
    combo: egui::ComboBox,
    text: impl Into<WidgetText>,
    help: &str,
    contents: impl FnOnce(&mut Ui) -> R,
) -> egui::InnerResponse<Option<R>> {
    let text = text.into();
    ui.horizontal(|ui| {
        let width = caption_width(ui);
        let label = caption(ui, text.text(), width, Sense::hover()).response;
        ui.spacing_mut().combo_width = ui.available_width().max(60.0);
        let mut response = combo.show_ui(ui, contents);
        response.response = response.response.labelled_by(label.id);
        label.on_hover_text(help).on_disabled_hover_text(help);
        response
    })
    .inner
}

/// Several values of one parameter on a single row — per-channel R/G/B,
/// C/M/Y, X/Y, or sublayer values — each as a compact drag field.
pub fn channels<T: egui::emath::Numeric, const N: usize>(
    ui: &mut Ui,
    text: impl Into<WidgetText>,
    help: &str,
    names: [&str; N],
    values: [&mut T; N],
    range: std::ops::RangeInclusive<T>,
    speed: f64,
) -> Response {
    let text = text.into();
    let id = inspector::row_id(text.text());
    let modified = inspector::row_modified(id);
    ui.horizontal(|ui| {
        ui.style_mut().explanation_tooltips = false;
        let width = caption_width(ui);
        let label = caption(ui, text.text(), width, Sense::click());
        label.paint_marker(ui, modified);
        if modified && label.response.double_clicked() || reset_requested(&label.response) {
            inspector::request_row_reset(id);
        }
        let label = label.response;
        ui.spacing_mut().item_spacing.x = 3.0;
        let gap = ui.spacing().item_spacing.x;
        let tag = 10.0;
        let lead = 4.0;
        // Each channel is [lead space][tag][field], every item followed by `gap`.
        let per_channel = lead + tag + 3.0 * gap;
        let field =
            ((ui.available_width() - per_channel * N as f32) / N as f32 - 1.0).clamp(28.0, 72.0);
        // DragValue's minimum width is `interact_size.x`; let it fit the column.
        ui.spacing_mut().interact_size.x = field;
        let mut response: Option<Response> = None;
        for (name, value) in names.into_iter().zip(values) {
            ui.add_space(lead);
            ui.add_sized(
                [tag, ui.spacing().interact_size.y],
                Label::new(RichText::new(name).small().color(channel_color(name)))
                    .selectable(false),
            );
            let r = ui.add_sized(
                [field, ui.spacing().interact_size.y],
                egui::DragValue::new(value)
                    .range(range.clone())
                    .speed(speed),
            );
            response = Some(match response {
                Some(prev) => prev.union(r),
                None => r,
            });
        }
        let response = response
            .expect("at least one channel")
            .labelled_by(label.id);
        attach_help(label, help, modified, true);
        if response.changed() {
            inspector::note_changed(id, text.text());
        }
        response
    })
    .inner
}

/// A slider that knows its stock default: it draws a tick at that value,
/// reports whether the current value differs, and restores it when the row's
/// caption asks for a reset. Built like `egui::Slider`.
pub struct StockSlider<'a, N: egui::emath::Numeric> {
    value: &'a mut N,
    range: std::ops::RangeInclusive<N>,
    stock: Option<N>,
    logarithmic: bool,
    step: Option<f64>,
}

pub fn slider<N: egui::emath::Numeric>(
    value: &mut N,
    range: std::ops::RangeInclusive<N>,
) -> StockSlider<'_, N> {
    StockSlider {
        value,
        range,
        stock: None,
        logarithmic: false,
        step: None,
    }
}

impl<N: egui::emath::Numeric> StockSlider<'_, N> {
    pub fn stock(mut self, stock: N) -> Self {
        self.stock = Some(stock);
        self
    }
    pub fn logarithmic(mut self, logarithmic: bool) -> Self {
        self.logarithmic = logarithmic;
        self
    }
    pub fn step_by(mut self, step: f64) -> Self {
        self.step = Some(step);
        self
    }
    /// Where `value` sits along the track, 0–1, matching `egui::Slider`.
    fn normalized(&self, value: f64) -> f32 {
        let (min, max) = (self.range.start().to_f64(), self.range.end().to_f64());
        let t = if self.logarithmic && min > 0.0 && max > min {
            (value.max(min).ln() - min.ln()) / (max.ln() - min.ln())
        } else if max > min {
            (value - min) / (max - min)
        } else {
            0.0
        };
        t.clamp(0.0, 1.0) as f32
    }
}

impl<N: egui::emath::Numeric> Widget for StockSlider<'_, N> {
    fn ui(self, ui: &mut Ui) -> Response {
        let reset = inspector::take_row_reset()
            && self
                .stock
                .is_some_and(|stock| stock.to_f64() != self.value.to_f64());
        if reset {
            *self.value = self.stock.expect("checked above");
        }
        let mut inner =
            egui::Slider::new(&mut *self.value, self.range.clone()).logarithmic(self.logarithmic);
        if let Some(step) = self.step {
            inner = inner.step_by(step);
        }
        let track_width = ui.spacing().slider_width;
        // Reserve a slot beneath the slider so the tick sits behind the handle:
        // hidden at the stock value, visible once the value moves away.
        let tick_slot = ui.painter().add(egui::Shape::Noop);
        let mut response = ui.add(inner);
        if reset {
            response.mark_changed();
        }
        let Some(stock) = self.stock else {
            return response;
        };
        let (value, stock) = (self.value.to_f64(), stock.to_f64());
        let tolerance = 1e-9 * stock.abs().max(1.0);
        inspector::set_row_modified((value - stock).abs() > tolerance);
        if ui.is_rect_visible(response.rect) {
            // Same geometry as egui's slider: the handle travels within the
            // track inset by its radius (track height / 2.5).
            let height = ui
                .text_style_height(&egui::TextStyle::Body)
                .max(ui.spacing().interact_size.y);
            let radius = height / 2.5;
            let left = response.rect.left() + radius;
            let x = left + self.normalized(stock) * (track_width - 2.0 * radius);
            let y = response.rect.center().y;
            ui.painter().set(
                tick_slot,
                egui::Shape::line_segment(
                    [egui::pos2(x, y - 6.0), egui::pos2(x, y + 6.0)],
                    egui::Stroke::new(2.0, theme::TEXT_WEAK),
                ),
            );
        }
        response
    }
}

pub const RGB: [&str; 3] = ["R", "G", "B"];
pub const CMY: [&str; 3] = ["C", "M", "Y"];

fn channel_color(name: &str) -> egui::Color32 {
    match name {
        "R" => egui::Color32::from_rgb(232, 120, 112),
        "G" => egui::Color32::from_rgb(132, 204, 120),
        "B" => egui::Color32::from_rgb(120, 160, 236),
        "C" => egui::Color32::from_rgb(110, 200, 214),
        "M" => egui::Color32::from_rgb(214, 120, 196),
        "Y" => egui::Color32::from_rgb(222, 200, 100),
        _ => theme::TEXT_WEAK,
    }
}

pub fn inhibition_receiver(donor: &str, receiver: usize) -> &'static str {
    match (donor, receiver) {
        ("Red → green / blue", 0) => "Green",
        ("Red → green / blue", _) | ("Green → red / blue", _) if receiver == 1 => "Blue",
        ("Green → red / blue", _) | ("Blue → red / green", 0) => "Red",
        _ => "Green",
    }
}

pub fn option(name: &str) -> &'static str {
    match name {
        "auto" => "Try rawler first and fall back to LibRaw if decoding fails.",
        "libraw" => "Decode with an installed LibRaw library, matching the Python RAW workflow.",
        "rawler" => "Decode with the built-in Rust RAW decoder without an external RAW library.",
        "as_shot" | "camera" => "Use the white balance recorded by the camera.",
        "daylight" => "Use a fixed daylight white balance.",
        "tungsten" => "Use a white balance for warm tungsten lighting.",
        "custom" => "Set RAW white balance using the temperature and tint controls.",
        "average" => "Meter exposure using the average image brightness.",
        "median" => "Meter exposure using the median image brightness to reduce outlier influence.",
        "center_weighted" => "Meter the whole image while giving more weight to its center.",
        "partial" => "Meter exposure from a small area around the image center.",
        "matrix" => "Meter a 5×5 grid with more weight near the center.",
        "multi_zone" => {
            "Meter three concentric regions, giving the central spot the greatest weight."
        }
        "highlight_weighted" => "Favor bright areas when metering to help preserve highlights.",
        "off" => "Disable this processing step.",
        "xy" => "Compress extreme input colors using CIE xy chromaticity coordinates.",
        "oklch" => "Compress out-of-gamut chroma in OkLCh while preserving hue.",
        "oklrab" => "Compress out-of-gamut colors in the lightness-remapped OkLab space.",
        "cam16ucs" => "Compress out-of-gamut colors using the CAM16 uniform appearance space.",
        "jzazbz" => "Compress out-of-gamut colors using a perceptual space designed for HDR.",
        "aces_rgc" => "Use ACES Reference Gamut Compression to tame extreme RGB values.",
        "hanatos2025" => {
            "Reconstruct spectral exposure from RGB using the Hanatos 2025 lookup basis."
        }
        "arctic2026alpha02" => {
            "Reconstruct spectral exposure using the Arctic memory-color reflectance lookup basis."
        }
        "mallett2019" => {
            "Relight the Mallett 2019 reflectance LUT and integrate it against film sensitivity."
        }
        "arctic2026beta04" => {
            "Recover reflectance with Arctic beta04, relight it, and normalize its neutral per film channel."
        }
        "jakob2019" => {
            "Recover reflectance using the Jakob 2019 basis and relight it for this film."
        }
        "otsu2018" => "Recover reflectance using the Otsu 2018 basis and relight it for this film.",
        "gauss-lasers" => {
            "Use narrow Gaussian spectra as a spectral-reconstruction comparison method."
        }
        "black_pro_mist" => {
            "Use the Black Pro-Mist model's balance of core softness, halos and bloom."
        }
        "glimmerglass" => "Use the Glimmerglass model's balance of core softness, halos and bloom.",
        "pro_mist" => "Use the Pro-Mist model's balance of core softness, halos and bloom.",
        "cinebloom" => "Use the CineBloom model's balance of core softness, halos and bloom.",
        "nearest" => "Display the closest image pixel without smoothing.",
        "linear" | "bilinear" => {
            "Blend neighboring pixels for smooth, inexpensive display scaling."
        }
        "cubic" | "bicubic" => "Use cubic interpolation for smoother display scaling.",
        "catrom" => {
            "Use Catmull-Rom interpolation for crisp display scaling with possible ringing."
        }
        "mitchell" => "Use Mitchell interpolation to balance display sharpness and ringing.",
        "lanczos" | "lanczos3" => {
            "Use a windowed sinc filter for sharp display scaling with possible ringing."
        }
        "spline16" => "Use a compact spline filter for smooth display scaling.",
        "spline36" => "Use a wider spline filter for detailed, smooth display scaling.",
        "blackman" => "Use a Blackman-windowed sinc filter for smooth display scaling.",
        "cube" => "Write a standard 3D CUBE LUT for compatible grading software.",
        "3dl" => "Write an Autodesk 3DL LUT with 10-bit sample values.",
        "lumix" => "Write a Lumix-compatible V-Log CUBE at a supported camera resolution.",
        "hald_png" => "Encode the LUT as a Hald PNG image for compatible image editors.",
        "1lut" => "Bake the complete film, print and scan transform into one LUT.",
        "2lut" => "Split the transform at film dye density into two LUTs.",
        "3lut" => "Separate film exposure, film development and print/scan into three LUTs.",
        "4lut" => {
            "Separate film exposure, film development, print exposure and print/scan into four LUTs."
        }
        "sRGB" | "srgb" => "Use the standard RGB space for web images and most displays.",
        "Display P3" => "Use the wide-gamut RGB space common on modern Apple displays.",
        "DCI-P3" => "Use cinema P3 primaries and the selected cinema encoding.",
        "ProPhoto RGB" => "Use a very wide RGB gamut suited to high-bit-depth editing and export.",
        "Adobe RGB" | "Adobe RGB (1998)" => {
            "Use a wider green/cyan gamut than sRGB for print-oriented workflows."
        }
        "ITU-R BT.709" | "Rec.709" => "Use the standard HDTV RGB primaries and signal encoding.",
        "ITU-R BT.2020" | "Rec.2020" => "Use the wide RGB gamut defined for UHD television.",
        "ACEScg" => "Use linear ACES AP1 primaries for rendering and compositing.",
        "ACES2065-1" => "Use linear ACES AP0 primaries for ACES interchange.",
        "Rec.709 Linear" | "sRGB Linear" => {
            "Use sRGB/Rec.709 primaries with linear-light RGB values."
        }
        "Rec.2020 Linear" => "Use wide Rec.2020 primaries with linear-light RGB values.",
        "ProPhoto Linear" => "Use ProPhoto primaries with linear-light RGB values.",
        "P3-D65 Linear" => "Use P3 primaries and a D65 white point with linear-light RGB values.",
        "ACEScc" => "Use the logarithmic ACEScc encoding for scene-referred color grading.",
        "ACEScct" => "Use the ACEScct grading encoding, which adds a linear toe to its log curve.",
        "ARRI LogC3 (EI800)" => "Use ARRI LogC3 at EI800 with ARRI Wide Gamut 3 primaries.",
        "ARRI LogC4" => "Use ARRI LogC4 with ARRI Wide Gamut 4 primaries.",
        "Sony S-Log3" => "Use Sony S-Log3 with S-Gamut3 primaries.",
        "Sony S-Log3 (S-Gamut3.Cine)" => {
            "Use Sony S-Log3 with the cinema-oriented S-Gamut3.Cine primaries."
        }
        "Panasonic V-Log" => "Use Panasonic V-Log with V-Gamut primaries for camera log footage.",
        "Fujifilm F-Log" => "Use Fujifilm F-Log encoding for camera log footage.",
        "Fujifilm F-Log2" => "Use Fujifilm F-Log2 encoding for camera log footage.",
        "Canon Log 3" => "Use Canon Log 3 encoding for camera log footage.",
        "RED Log3G10" => "Use RED Log3G10 encoding for camera log footage.",
        "DaVinci Intermediate" => {
            "Use DaVinci Intermediate encoding for a scene-referred grading workflow."
        }
        "Apple Log" => "Use Apple Log encoding for recorded camera footage.",
        "Blackmagic Film Gen 5" => {
            "Use Blackmagic Film Generation 5 encoding for camera log footage."
        }
        "DJI D-Log" => "Use DJI D-Log encoding for camera log footage.",
        "Nikon N-Log" => "Use Nikon N-Log with Rec.2020 primaries for camera log footage.",
        "Rec.2100 PQ" => "Use PQ encoding and Rec.2020 primaries for HDR output or input.",
        "Rec.2100 HLG" => "Use HLG encoding and Rec.2020 primaries for HDR output or input.",
        "P3-D65 PQ" => "Use PQ encoding with P3-D65 primaries for HDR output or input.",
        "TH-KG3" => "Illuminate the print with the filtered tungsten-halogen enlarger spectrum.",
        "TH-KG3-L" => "Use the alternate filtered tungsten-halogen enlarger spectrum.",
        "D50" => "Illuminate the print with a daylight spectrum near 5000 K.",
        "D55" => "Illuminate the print with a daylight spectrum near 5500 K.",
        "D65" => "Illuminate the print with a daylight spectrum near 6500 K.",
        "BB3200" => "Illuminate the print with a 3200 K blackbody spectrum.",
        _ => "Use this color space and its associated signal encoding.",
    }
}

pub fn inhibition(donor: &str, receiver: usize) -> &'static str {
    match (donor, receiver) {
        ("Red → green / blue", 0) => {
            "Set how strongly red-layer development inhibits the green layer."
        }
        ("Red → green / blue", _) => {
            "Set how strongly red-layer development inhibits the blue layer."
        }
        ("Green → red / blue", 0) => {
            "Set how strongly green-layer development inhibits the red layer."
        }
        ("Green → red / blue", _) => {
            "Set how strongly green-layer development inhibits the blue layer."
        }
        ("Blue → red / green", 0) => {
            "Set how strongly blue-layer development inhibits the red layer."
        }
        _ => "Set how strongly blue-layer development inhibits the green layer.",
    }
}

pub const CONTROL_OPEN: &str = "Open a camera RAW file or a JPEG, PNG, TIFF or EXR image.";
pub const CONTROL_SAVE_PREVIEW: &str =
    "Save the current rendered preview at its preview resolution.";
pub const CONTROL_RENDER_PREVIEW: &str =
    "Render a fresh preview using the current processing settings.";
pub const CONTROL_AUTO_PREVIEW: &str =
    "Refresh the preview automatically when processing settings change.";
pub const CONTROL_SAVE_PROJECT: &str =
    "Save the image path, processing settings and display preferences in a project file.";
pub const CONTROL_OPEN_PROJECT: &str =
    "Restore a saved project or import a compatible Python GUI state file.";
pub const CONTROL_SAVE_DEFAULTS: &str =
    "Use the current settings for future launches without reopening this image.";
pub const CONTROL_FACTORY_DEFAULTS: &str =
    "Reset processing and display settings and remove saved startup defaults.";
pub const CONTROL_AUTO_EXPOSURE: &str =
    "Estimate film exposure from image brightness before applying EV compensation.";
pub const CONTROL_EV_COMPENSATION: &str =
    "Adjust film exposure in stops, with +1 EV doubling the exposure.";
pub const SECTION_TONE: &str =
    "Balance the photograph's tonal ranges after exposure and before film simulation.";
pub const CONTROL_HIGHLIGHTS: &str = "Darken or brighten highlight detail before film exposure. Negative values restrain bright detail; the brightest tail is controlled by Whites.";
pub const CONTROL_SHADOWS: &str =
    "Lift or deepen shadow detail before film exposure, keeping middle gray and true black fixed.";
pub const CONTROL_WHITES: &str = "Reduce or increase the brightest exposures before film simulation, preserving highlight headroom.";
pub const CONTROL_BLACKS: &str = "Negative values deepen and clip the darkest tones. Positive values raise a neutral black floor, reducing saturation near black.";
pub const CONTROL_FILM_FORMAT_MM: &str = "Set the film’s long edge in millimeters, which determines the physical scale of grain and optical effects.";
pub const CONTROL_LENS_BLUR_UM: &str =
    "Soften the camera image using a Gaussian blur measured in film-plane micrometers.";
pub const CONTROL_HALATION_AMOUNT: &str =
    "Scale the glow caused by light reflecting back through the film emulsion.";
pub const CONTROL_HALATION_SCALE: &str =
    "Change how far reflected halation light spreads around bright areas.";
pub const CONTROL_SCATTER_AMOUNT: &str =
    "Set how much light spreads within the emulsion before film development.";
pub const CONTROL_SCATTER_SCALE: &str =
    "Scale the distance that light scattering spreads within the emulsion.";
pub const CONTROL_BOUNCES: &str =
    "Set how many reflected-light passes contribute to the halation halo.";
pub const CONTROL_BOUNCE_DECAY: &str =
    "Set how much strength each successive halation bounce retains.";
pub const CONTROL_RENORMALIZE: &str =
    "Preserve midgray brightness by compensating for the light added by halation.";
pub const CONTROL_SCATTER_CORE_UM: &str =
    "Set the scatter core’s Gaussian standard deviation for this color channel, in micrometers.";
pub const CONTROL_SCATTER_TAIL_UM: &str =
    "Set the exponential scatter tail’s decay length for this color channel, in micrometers.";
pub const CONTROL_TAIL_WEIGHT: &str =
    "Set the fraction of scattering assigned to the wider exponential tail.";
pub const CONTROL_HALATION_STRENGTH: &str =
    "Set the reflected-light contribution for this emulsion color channel.";
pub const CONTROL_FIRST_BOUNCE_UM: &str =
    "Set the first halation bounce’s Gaussian standard deviation for this channel, in micrometers.";
pub const CONTROL_HIGHLIGHT_BOOST_EV: &str =
    "Amplify bright film exposures by up to this many stops before optical spreading.";
pub const CONTROL_BOOST_RAMP_0_1: &str =
    "Control how gradually the highlight boost grows above its protected range.";
pub const CONTROL_PROTECT_EV: &str =
    "Protect tones below this many stops above middle gray from highlight boosting.";
pub const CONTROL_AMOUNT: &str =
    "Scale DIR development inhibition to change color separation and local edge contrast.";
pub const CONTROL_DIFFUSION_SIZE_UM: &str =
    "Set the Gaussian σ of inhibitor diffusion in micrometers, controlling local edge contrast.";
pub const CONTROL_DIFFUSION_TAIL_UM: &str =
    "Set the inhibitor diffusion tail’s exponential decay length in micrometers.";
pub const CONTROL_TAIL_WEIGHT_2: &str =
    "Blend the narrow inhibitor diffusion with its wider exponential tail.";
pub const CONTROL_SAME_LAYER_SCALE: &str =
    "Scale development inhibition acting within each color-sensitive layer.";
pub const CONTROL_INTER_LAYER_SCALE: &str =
    "Scale development inhibition passing between different color-sensitive layers.";
pub const CONTROL_STRENGTH: &str =
    "Set the modeled diffusion-filter grade, where 0.25 corresponds to a 1/4-strength filter.";
pub const CONTROL_SPATIAL_SCALE: &str =
    "Scale the overall size of the diffusion filter’s softness, halos and bloom.";
pub const CONTROL_HALO_WARMTH: &str =
    "Positive values warm the outer halo and cool its inner region; negative values reverse this.";
pub const CONTROL_CORE_INTENSITY: &str =
    "Scale the diffusion core’s contribution relative to the halo and bloom.";
pub const CONTROL_HALO_INTENSITY: &str =
    "Scale the diffusion halo’s contribution relative to the core and bloom.";
pub const CONTROL_BLOOM_INTENSITY: &str =
    "Scale the diffusion bloom’s contribution relative to the core and halo.";
pub const CONTROL_HALO_SIZE: &str =
    "Scale the distance that the medium-range diffusion halo spreads.";
pub const CONTROL_BLOOM_SIZE: &str = "Scale the distance that the broad diffusion bloom spreads.";
pub const CONTROL_POST_BLUR_SIGMA: &str = "Soften the generated grain with Gaussian blur whose standard deviation σ is measured in pixels.";
pub const CONTROL_PERCENT: &str = "Set the percentage of glare light added to the scanned image.";
pub const CONTROL_ROUGHNESS: &str = "Set the spatial variation in the glare field.";
pub const CONTROL_BLUR_SIGMA_PX: &str =
    "Smooth the glare field with a Gaussian standard deviation measured in pixels.";
pub const CONTROL_GAMMA: &str =
    "Scale the overall contrast of the print’s parametric density curves.";
pub const CONTROL_GAMMA_FAST: &str =
    "Scale the contrast contribution of the faster print emulsion component.";
pub const CONTROL_GAMMA_SLOW: &str =
    "Scale contrast in both the middle and slow print emulsion sublayers.";
pub const CONTROL_DEVELOPER_EXHAUSTION: &str =
    "Change the paper’s highlight shoulder while preserving its midgray density.";
pub const CONTROL_LENS_BLUR_SIGMA_PX: &str =
    "Soften the scanned image with a Gaussian standard deviation measured in pixels.";
pub const CONTROL_UNSHARP_SIGMA_PX: &str =
    "Set the Gaussian standard deviation used to isolate detail for sharpening.";
pub const CONTROL_UNSHARP_AMOUNT: &str =
    "Set how strongly unsharp masking increases local edge contrast.";
pub const CONTROL_WHITE_CORRECTION: &str =
    "Normalize the scanner’s white reference to the chosen white level.";
pub const CONTROL_WHITE_LEVEL: &str =
    "Set the target brightness of the scanner’s corrected white reference.";
pub const CONTROL_BLACK_CORRECTION: &str =
    "Normalize the scanner’s black reference to the chosen black level.";
pub const CONTROL_BLACK_LEVEL: &str =
    "Set the target brightness of the scanner’s corrected black reference.";
pub const CONTROL_CAT16_INPUT_ADAPTATION: &str =
    "Use CAT16 instead of CAT02 to adapt input colors to the film’s reference white.";
pub const CONTROL_PRINT_EXPOSURE: &str =
    "Scale the light exposure of the print paper under the enlarger.";
pub const CONTROL_MAGENTA_FILTER_SHIFT: &str =
    "Adjust magenta filtration relative to the enlarger’s neutral color balance.";
pub const CONTROL_YELLOW_FILTER_SHIFT: &str =
    "Adjust yellow filtration relative to the enlarger’s neutral color balance.";
pub const CONTROL_PREFLASH_MAGENTA_SHIFT: &str =
    "Adjust the magenta filtration of the uniform paper preflash.";
pub const CONTROL_PREFLASH_YELLOW_SHIFT: &str =
    "Adjust the yellow filtration of the uniform paper preflash.";
pub const CONTROL_ENCODE_OUTPUT_TRANSFER_FUNCTION: &str =
    "Encode processed RGB values with the output color space’s transfer curve.";
pub const CONTROL_INPUT_PRIMARIES: &str =
    "Choose the RGB primaries and white point used to interpret the source image.";
pub const CONTROL_OUTPUT_PRIMARIES: &str =
    "Choose the RGB primaries and white point produced by the simulation.";
pub const CONTROL_DECODE_INPUT_TRANSFER_FUNCTION: &str =
    "Convert encoded source RGB values to linear light before film exposure.";
pub const CONTROL_ENLARGER_SPECTRAL_LUT: &str =
    "Approximate the enlarger’s spectral integration with an interpolated lookup table.";
pub const CONTROL_SCANNER_SPECTRAL_LUT: &str =
    "Approximate the scanner’s spectral integration with an interpolated lookup table.";
pub const CONTROL_SPECTRAL_LUT_RESOLUTION: &str =
    "Set samples per axis for spectral lookup tables, trading setup cost for accuracy.";
pub const CONTROL_APPROXIMATE_PHYSICAL_GRAIN_STATISTICS: &str =
    "Use faster approximate grain statistics instead of the exact particle model.";
pub const CONTROL_OUTPUT_GAMUT_COMPRESSION: &str =
    "Map extreme colors into the chosen output gamut using a perceptual compression method.";
pub const CONTROL_INPUT_GAMUT_COMPRESSION: &str =
    "Compress extreme source colors before reconstructing spectral film exposure.";
pub const CONTROL_CROP_BEFORE_PROCESSING: &str =
    "Process only the selected rectangle of the source image.";
pub const CONTROL_DYE_CLOUD_BLUR_UM: &str =
    "Scale the particle-dependent Gaussian blur used to form grain dye clouds.";
pub const CONTROL_SENSITIVITY_ADAPTATION_SURFACE: &str =
    "Apply the Hanatos illuminant-dependent polynomial correction when reconstructing spectra.";
pub const CONTROL_SPECTRAL_BLUR: &str = "Smooth reconstructed spectra along the wavelength axis.";
pub const CONTROL_UV_FILTER_AMOUNT: &str =
    "Set how strongly the camera filter attenuates ultraviolet light.";
pub const CONTROL_IR_FILTER_AMOUNT: &str =
    "Set how strongly the camera filter attenuates infrared light.";
pub const CONTROL_SIMPLIFIED_PREVIEW_MODE: &str =
    "Skip grain, lens blur and sharpening while retaining halation and optical diffusion.";
pub const CONTROL_PREVIEW_LONG_EDGE_PX: &str =
    "Limit the interactive preview’s long edge in pixels; export starts from the original image.";
pub const CONTROL_TEMPERATURE_K: &str =
    "Set the illuminant temperature used for custom RAW white balance.";
pub const CONTROL_TINT: &str = "Adjust green–magenta balance for custom RAW white balance.";
pub const CONTROL_LENS_CORRECTION_LENSFUN: &str =
    "Correct supported lens distortion, color fringing and vignetting using Lensfun profiles.";
pub const CONTROL_USE_AUTOMATIC_LENSFUN_DATABASE: &str =
    "Clear the chosen folder and discover a local or automatically downloaded lens database.";
pub const CONTROL_ROTATE: &str =
    "Rotate the source image 90 degrees counterclockwise before processing and export.";
pub const CONTROL_ROTATE_2: &str =
    "Rotate the source image 90 degrees clockwise before processing and export.";
pub const CONTROL_DISPLAY_COLOR_TRANSFORM: &str =
    "Convert preview colors to sRGB for display without changing the exported color space.";
pub const CONTROL_WHITE_BORDER_LAYER: &str =
    "Show or hide the white border around the preview image.";
pub const CONTROL_PAPER_BACK_LAYER: &str =
    "Show or hide the textured paper backing in the preview.";
pub const CONTROL_INPUT_LAYER: &str = "Show or hide the original source image in the preview.";
pub const CONTROL_OUTPUT_LAYER: &str = "Show or hide the processed image in the preview.";
pub const CONTROL_INPUT_OPACITY: &str =
    "Set how strongly the source image contributes to the preview display.";
pub const CONTROL_OUTPUT_OPACITY: &str =
    "Set how strongly the processed image contributes to the preview display.";
pub const CONTROL_COMPARE_INPUT_OUTPUT: &str =
    "Show the source and processed image on opposite sides of a movable divider.";
pub const CONTROL_COMPARISON_SPLIT: &str =
    "Move the divider between the source and processed preview.";
pub const CONTROL_18_GRAY_CANVAS: &str = "Use a middle-gray background around the preview image.";
pub const CONTROL_WHITE_BORDER: &str =
    "Set preview border width as a fraction of the image’s long edge without affecting export.";
pub const CONTROL_PAPER_DEVELOPMENT_CROSSFADE: &str =
    "Animate paper development and transitions between rendered previews.";
pub const CONTROL_REPLAY_PAPER_DEVELOPMENT: &str =
    "Replay the paper-development animation for the current processed preview.";
pub const CONTROL_ENCODE_SAVING_TRANSFER_FUNCTION: &str =
    "Apply the saving color space’s transfer curve when encoding the output file.";
pub const CONTROL_JPEG_QUALITY: &str =
    "Trade JPEG file size against compression artifacts when saving an image.";
pub const CONTROL_CUBE_RESOLUTION: &str =
    "Set the LUT’s samples per axis, with larger cubes using more space and finer sampling.";
pub const CONTROL_EXPORT_A_BUNDLE_WITH_METADATA_AND_OCIO: &str =
    "Write stage LUTs and metadata together, with OCIO for supported color transports.";
pub const CONTROL_INCLUDE_COMBINED_STAGES: &str =
    "Also bake contiguous stage combinations for simpler use in grading software.";
pub const CONTROL_INCLUDE_DIAGNOSTIC_REPORT: &str =
    "Generate a report measuring the baked LUT’s color behavior and interpolation accuracy.";
pub const CONTROL_EXPORT_LUT: &str =
    "Bake the selected film and paper transform into a LUT file or stage bundle.";
pub const SECTION_EXPOSURE: &str = "Control how much source light reaches the simulated film.";
pub const SECTION_HALATION: &str =
    "Simulate light scattering inside the emulsion and reflecting back through the film.";
pub const SECTION_PHYSICAL_EMULSION_PARAMETERS: &str =
    "Adjust channel-specific scattering and reflected-light behavior inside the film.";
pub const SECTION_DIR_COUPLERS: &str =
    "Development-inhibitor-releasing couplers alter color separation and local edge contrast.";
pub const SECTION_INHIBITION_MATRIX: &str =
    "Control how development in each color layer inhibits the same layer and other layers.";
pub const SECTION_DIFFUSION_FILTER_LENS: &str =
    "Simulate a lens-mounted diffusion filter that softens detail and spreads highlight light.";
pub const SECTION_GRAIN: &str =
    "Control the size, softness and emulsion structure of simulated film grain.";
pub const SECTION_GLARE: &str =
    "Add spatially varying veiling light during film or print scanning.";
pub const SECTION_SCANNER: &str =
    "Control scan softness, sharpening and white/black reference levels.";
pub const SECTION_COLOR_MANAGEMENT: &str =
    "Choose spectral reconstruction, chromatic adaptation and gamut-compression methods.";
pub const SECTION_ENLARGER: &str =
    "Control the light, color filtration and preflash used to expose the print paper.";
pub const SECTION_OUTPUT: &str =
    "Choose direct film scanning, processing resolution and output signal encoding.";
pub const SECTION_COLOR_AND_PROCESSING: &str =
    "Adjust color spaces, spectral processing, grain structure and source cropping.";
pub const SECTION_RAW_DEVELOPMENT: &str = "Set how camera RAW data is decoded before the film simulation. Changes reload the RAW automatically.";
pub const SECTION_DISPLAY: &str =
    "Control preview layers, comparison, borders, scaling and display animations.";
pub const SECTION_FILE_EXPORT: &str =
    "Choose full-resolution export precision, file color encoding and image quality.";
pub const SECTION_LUT_EXPORT: &str =
    "Bake a reusable color transform from the current film, paper and processing settings.";
pub const MENU_METERING: &str = "Choose which image regions determine automatic film exposure.";
pub const MENU_FAMILY: &str =
    "Choose the diffusion model’s balance of core softness, halos and bloom.";
pub const MENU_RGB_RAW_UPSAMPLING: &str =
    "Choose how input RGB colors are reconstructed into spectral film exposure.";
pub const MENU_ENLARGER_ILLUMINANT: &str =
    "Choose the spectral light source used to expose the print paper.";
pub const MENU_RAW_DECODER: &str =
    "Choose the engine that decodes camera RAW data before film processing.";
pub const MENU_RAW_WHITE_BALANCE: &str =
    "Choose the white-balance method used when decoding the RAW file.";
pub const MENU_CANVAS_INTERPOLATION: &str =
    "Choose how image pixels are resampled for preview zooming without changing export.";
pub const MENU_SAVING_COLOR_SPACE: &str =
    "Convert rendered colors to this RGB color space before writing the image file.";
pub const MENU_EXPORT_FORMAT: &str =
    "Choose the default file type for Save preview and Export image.";
pub const MENU_EXPORT_DEPTH: &str =
    "Choose TIFF integer or floating-point precision, or EXR floating-point precision.";
pub const MENU_LUT_FORMAT: &str =
    "Choose the LUT file format supported by the target camera or editing application.";
pub const MENU_LUT_TOPOLOGY: &str =
    "Choose where the film, print and scan transform is split into separate LUT stages.";
pub const CANCEL_EXPORT: &str =
    "Cancel the current export after any active GPU dispatch completes.";
pub const EXPORT_IMAGE: &str =
    "Render the original image at full resolution using the selected export precision.";
pub const SCAN_FOR_PRINT: &str = "Enable scanner white/black correction and disable glare, restoring prior settings when turned off.";
pub const ENABLE_HALATION: &str =
    "Enable scattering and reflected-light halation in the film emulsion.";
pub const ENABLE_DIR: &str =
    "Enable development-inhibitor couplers between the film’s color layers.";
pub const DIR_SELF_INHIBITION: &str =
    "Set development inhibition within this color-sensitive layer.";
pub const DIR_DONOR: &str =
    "Set how much this donor layer inhibits each of the two named receiving layers.";
pub const ENABLE_DIFFUSION: &str = "Enable the lens diffusion filter before film exposure.";
pub const ENABLE_GRAIN: &str = "Enable simulated silver-halide grain in film development.";
pub const ENABLE_GLARE: &str = "Enable veiling glare light in the film or print scan.";
pub const MORPH_PRINT_CURVES: &str = "Rebuild paper density curves from its parametric model to apply the contrast and developer controls.";
pub const PREFLASH: &str = "Add a uniform paper preflash through the film base to adjust shadow density and lower print contrast.";
pub const WORKING_RESOLUTION: &str =
    "Scale the image before processing, trading preview and export resolution for speed.";
pub const DEVELOPMENT_VARIANT: &str =
    "Use this measured development-time variant, which changes the profile’s tonal response.";
pub const DEVELOPMENT_TIME: &str =
    "Choose a measured development-time variant that controls the profile’s contrast.";
pub const FILM_PROFILE: &str =
    "Use this film stock’s measured sensitivities, density curves and stock defaults.";
pub const PAPER_PROFILE: &str = "Use this print paper’s measured sensitivities and density curves.";
pub const DYE_ORDER: &str = "Reorder the three dye-density channels before spectral rendering.";
pub const AUTOMATIC_NEUTRAL_FILTERS: &str =
    "Use the calibrated preset for the selected film, paper and illuminant.";
pub const MANUAL_NEUTRAL_FILTER: &str =
    "Set this enlarger filter’s neutral strength for manual print color balance.";
pub const CROP_CENTER: &str = "Place the crop center along this image axis, from 0 to 1.";
pub const CROP_SIZE: &str =
    "Set crop width or height as a fraction of the source image’s long edge.";
pub const LAYER_PARTICLE_SCALE: &str =
    "Scale the particle area in this emulsion sublayer to change its grain strength.";
pub const LENS_DATABASE_OVERRIDE: &str =
    "Use this chosen Lensfun database folder instead of automatic discovery.";
pub const CHOOSE_LENS_DATABASE: &str =
    "Optional override; no folder selection is needed for automatic setup.";
pub const EXPORT_PRECISION: &str = "Render full-resolution exports on the GPU at f32 precision, or disable for f64 CPU processing.";
pub const BUILD_BUNDLE_SPEC: &str =
    "Choose a JSON or TOML specification for multi-paper, ZIP and delivery-target workflows.";
pub const TIFF_FLOAT: &str =
    "Store floating-point TIFF values, preserving negative values and highlights above 1.";
pub const TIFF_INTEGER: &str =
    "Store integer TIFF values clipped to the 0–1 range at this bit depth.";
pub const LUT_INPUT: &str = "Choose the RGB encoding expected at the LUT input.";
pub const LUT_OUTPUT: &str = "Choose the RGB encoding produced by the LUT.";
pub const GPU_CANVAS_NOTE: &str = "Advanced sampling and animation use the wgpu canvas. Restart with SPEKTRAFILM_GUI_RENDERER=wgpu to enable them.";
pub const LUT_SETTINGS_NOTE: &str = "Uses the selected film, paper and processing settings. Spatial effects and grain are excluded from LUTs.";

#[cfg(test)]
mod tests {
    use super::*;

    const CAPTION: &str = "Control caption";
    const HELP: &str = "Label-only help is visible.";

    #[derive(Clone, Copy, Debug)]
    enum Editor {
        Slider,
        Number,
        Checkbox,
        Combo,
    }

    fn paint(
        ctx: &egui::Context,
        editor: Editor,
        time: f64,
        events: Vec<egui::Event>,
        checked: &mut bool,
    ) -> (egui::FullOutput, egui::Rect, bool) {
        let mut rect = egui::Rect::NOTHING;
        let mut changed = false;
        let output = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(640.0, 480.0),
                )),
                time: Some(time),
                events,
                ..Default::default()
            },
            |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    let mut value = 0.5f32;
                    let response = match editor {
                        Editor::Slider => {
                            labeled(ui, egui::Slider::new(&mut value, 0.0..=1.0), CAPTION, HELP)
                        }
                        Editor::Number => {
                            labeled(ui, egui::DragValue::new(&mut value), CAPTION, HELP)
                        }
                        Editor::Checkbox => checkbox(ui, checked, CAPTION, HELP),
                        Editor::Combo => {
                            combo(
                                ui,
                                egui::ComboBox::from_id_salt("test_combo")
                                    .selected_text("Selection"),
                                CAPTION,
                                HELP,
                                |_| {},
                            )
                            .response
                        }
                    };
                    rect = response.rect;
                    changed = response.changed();
                });
            },
        );
        (output, rect, changed)
    }

    fn text_rect(output: &egui::FullOutput, text: &str) -> Option<egui::Rect> {
        fn find(shape: &egui::Shape, text: &str) -> Option<egui::Rect> {
            match shape {
                egui::Shape::Text(shape) if shape.galley.text() == text => {
                    Some(egui::Rect::from_min_size(shape.pos, shape.galley.size()))
                }
                egui::Shape::Vec(shapes) => shapes.iter().find_map(|s| find(s, text)),
                _ => None,
            }
        }
        output.shapes.iter().find_map(|s| find(&s.shape, text))
    }

    fn visible_text(output: &egui::FullOutput) -> Vec<String> {
        fn collect(shape: &egui::Shape, text: &mut Vec<String>) {
            match shape {
                egui::Shape::Text(shape) => text.push(shape.galley.text().to_string()),
                egui::Shape::Vec(shapes) => {
                    for shape in shapes {
                        collect(shape, text);
                    }
                }
                _ => {}
            }
        }
        let mut text = vec![];
        for shape in &output.shapes {
            collect(&shape.shape, &mut text);
        }
        text
    }

    #[test]
    fn help_appears_over_captions_and_not_slider_tracks_values_or_checkboxes() {
        for editor in [
            Editor::Slider,
            Editor::Number,
            Editor::Checkbox,
            Editor::Combo,
        ] {
            for target in ["label", "control_left", "control_right"] {
                if matches!(editor, Editor::Checkbox) && target == "control_right" {
                    // The checkbox response includes its clickable caption.
                    continue;
                }
                let hover_label = target == "label";
                let ctx = egui::Context::default();
                ctx.style_mut(|s| s.interaction.tooltip_delay = 0.0);
                let mut checked = false;
                let (output, control, _) = paint(&ctx, editor, 0.0, vec![], &mut checked);
                let baseline_text = visible_text(&output);
                let caption = text_rect(&output, CAPTION).expect("caption must be painted");
                let position = if hover_label {
                    caption.center()
                } else if target == "control_left" {
                    egui::pos2(control.left() + 5.0, control.center().y)
                } else {
                    egui::pos2(control.right() - 5.0, control.center().y)
                };
                let mut shown = false;
                for frame in 1..=5 {
                    let (output, _, _) = paint(
                        &ctx,
                        editor,
                        frame as f64,
                        vec![egui::Event::PointerMoved(position)],
                        &mut checked,
                    );
                    shown |= text_rect(&output, HELP).is_some();
                    if hover_label {
                        assert_ne!(output.platform_output.cursor_icon, egui::CursorIcon::Text);
                    }
                    if !hover_label {
                        for text in visible_text(&output) {
                            assert!(
                                baseline_text.contains(&text),
                                "{editor:?} {target}: unexpected hover text {text:?}"
                            );
                        }
                    }
                }
                assert_eq!(shown, hover_label, "{editor:?}, label={hover_label}");
            }
        }
    }

    #[test]
    fn double_clicking_a_changed_slider_caption_restores_its_stock_value() {
        const STOCK: f32 = 0.25;
        let ctx = egui::Context::default();
        let mut value = 0.75f32;
        let frame = |time: f64, events: Vec<egui::Event>, value: &mut f32| {
            let mut changed = false;
            let output = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(640.0, 480.0),
                    )),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        changed = labeled(ui, slider(value, 0.0..=1.0).stock(STOCK), CAPTION, HELP)
                            .changed();
                    });
                },
            );
            (output, changed)
        };
        let (output, _) = frame(0.0, vec![], &mut value);
        let caption = text_rect(&output, CAPTION).expect("caption must be painted");
        let click = |pressed| egui::Event::PointerButton {
            pos: caption.center(),
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        frame(
            0.1,
            vec![egui::Event::PointerMoved(caption.center())],
            &mut value,
        );
        frame(0.2, vec![click(true), click(false)], &mut value);
        let (_, changed) = frame(0.3, vec![click(true), click(false)], &mut value);
        assert_eq!(value, STOCK);
        assert!(
            changed,
            "the reset must be reported so the preview re-renders"
        );
    }

    #[test]
    fn stock_tick_follows_slider_scale() {
        let mut v = 1.0f64;
        let linear = slider(&mut v, 0.0..=4.0);
        assert_eq!(linear.normalized(1.0), 0.25);
        let mut v = 1.0f64;
        let log = slider(&mut v, 0.1..=10.0).logarithmic(true);
        assert!((log.normalized(1.0) - 0.5).abs() < 1e-6);
        let mut v = 1.0f64;
        assert_eq!(slider(&mut v, 0.0..=4.0).normalized(9.0), 1.0);
    }

    #[test]
    fn clicking_a_checkbox_caption_still_toggles_and_reports_a_change() {
        let ctx = egui::Context::default();
        let mut checked = false;
        let (output, _, _) = paint(&ctx, Editor::Checkbox, 0.0, vec![], &mut checked);
        let position = text_rect(&output, CAPTION).unwrap().center();
        for (frame, pressed) in [(1, true), (2, false)] {
            let (_, _, changed) = paint(
                &ctx,
                Editor::Checkbox,
                frame as f64 * 0.05,
                vec![
                    egui::Event::PointerMoved(position),
                    egui::Event::PointerButton {
                        pos: position,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    },
                ],
                &mut checked,
            );
            if !pressed {
                assert!(checked);
                assert!(changed);
            }
        }
    }
}
