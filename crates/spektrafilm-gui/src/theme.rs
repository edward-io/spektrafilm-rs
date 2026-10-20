//! Visual theme: neutral greys so the chrome never biases colour judgement,
//! one muted safelight-amber accent, and bundled Inter / JetBrains Mono.
use eframe::egui::{
    self, Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Stroke, TextStyle,
    Theme,
};
use std::sync::Arc;

pub const ACCENT: Color32 = Color32::from_rgb(214, 160, 92);
pub const ACCENT_DIM: Color32 = Color32::from_rgb(112, 82, 46);
pub const PANEL: Color32 = Color32::from_gray(30);
pub const PANEL_RAISED: Color32 = Color32::from_gray(38);
pub const WELL: Color32 = Color32::from_gray(20);
pub const TEXT: Color32 = Color32::from_gray(214);
pub const TEXT_CAPTION: Color32 = Color32::from_gray(190);
pub const TEXT_WEAK: Color32 = Color32::from_gray(150);
pub const TEXT_FAINT: Color32 = Color32::from_gray(105);
/// Header and stock-name weight; egui has no font weights, so this is a family.
pub const SEMIBOLD: &str = "semibold";

pub fn apply(ctx: &egui::Context) {
    ctx.set_fonts(fonts());
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.style_mut_of(Theme::Dark, style);
}

fn fonts() -> FontDefinitions {
    let mut fonts = FontDefinitions::default();
    for (name, bytes) in [
        (
            "Inter",
            &include_bytes!("../assets/fonts/Inter-Regular.ttf")[..],
        ),
        (
            "Inter-SemiBold",
            &include_bytes!("../assets/fonts/Inter-SemiBold.ttf")[..],
        ),
        (
            "JetBrainsMono",
            &include_bytes!("../assets/fonts/JetBrainsMono-Regular.ttf")[..],
        ),
    ] {
        fonts
            .font_data
            .insert(name.into(), Arc::new(FontData::from_static(bytes)));
    }
    // Keep egui's bundled fonts behind ours as glyph fallbacks (symbols, emoji).
    let proportional = fonts.families.entry(FontFamily::Proportional).or_default();
    proportional.insert(0, "Inter".into());
    let fallbacks = proportional[1..].to_vec();
    // Icons get their own family: Inter has glyphs of its own at some of the
    // private-use code points Phosphor uses.
    fonts.font_data.insert(
        "phosphor".into(),
        Arc::new(egui_phosphor::Variant::Regular.font_data()),
    );
    fonts.families.insert(
        FontFamily::Name(crate::icons::FAMILY.into()),
        vec!["phosphor".into()],
    );
    let mono = fonts.families.entry(FontFamily::Monospace).or_default();
    mono.insert(0, "JetBrainsMono".into());
    fonts.families.insert(
        FontFamily::Name(SEMIBOLD.into()),
        std::iter::once("Inter-SemiBold".to_string())
            .chain(fallbacks)
            .collect(),
    );
    fonts
}

fn style(style: &mut egui::Style) {
    style.text_styles = [
        (TextStyle::Small, FontId::proportional(11.0)),
        (TextStyle::Body, FontId::proportional(13.0)),
        (TextStyle::Button, FontId::proportional(13.0)),
        (
            TextStyle::Heading,
            FontId::new(15.0, FontFamily::Name(SEMIBOLD.into())),
        ),
        (TextStyle::Monospace, FontId::monospace(12.0)),
    ]
    .into();

    let s = &mut style.spacing;
    s.item_spacing = egui::vec2(8.0, 6.0);
    s.button_padding = egui::vec2(8.0, 3.0);
    s.interact_size = egui::vec2(52.0, 22.0);
    s.indent = 14.0;
    s.combo_width = 140.0;
    s.menu_margin = egui::Margin::same(6);
    s.scroll = egui::style::ScrollStyle::solid();

    let v = &mut style.visuals;
    let radius = CornerRadius::same(4);
    v.dark_mode = true;
    v.panel_fill = PANEL;
    v.window_fill = PANEL_RAISED;
    v.window_stroke = Stroke::new(1.0, Color32::from_gray(52));
    v.window_corner_radius = CornerRadius::same(6);
    v.menu_corner_radius = CornerRadius::same(6);
    v.extreme_bg_color = WELL;
    v.faint_bg_color = PANEL_RAISED;
    v.code_bg_color = WELL;
    v.hyperlink_color = ACCENT;
    v.warn_fg_color = Color32::from_rgb(232, 184, 96);
    v.error_fg_color = Color32::from_rgb(240, 110, 100);
    v.slider_trailing_fill = true;
    v.indent_has_left_vline = false;
    v.collapsing_header_frame = false;
    v.selection.bg_fill = ACCENT_DIM;
    v.selection.stroke = Stroke::new(1.0, Color32::from_rgb(255, 226, 184));
    v.popup_shadow = egui::epaint::Shadow {
        offset: [0, 4],
        blur: 12,
        spread: 0,
        color: Color32::from_black_alpha(110),
    };
    v.window_shadow = v.popup_shadow;

    let w = &mut v.widgets;
    w.noninteractive.bg_fill = PANEL;
    w.noninteractive.weak_bg_fill = PANEL;
    w.noninteractive.bg_stroke = Stroke::new(1.0, Color32::from_gray(48));
    w.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
    for (visuals, bg, fg) in [
        (&mut w.inactive, 44, 220),
        (&mut w.hovered, 56, 240),
        (&mut w.active, 66, 255),
        (&mut w.open, 50, 230),
    ] {
        visuals.bg_fill = Color32::from_gray(bg);
        visuals.weak_bg_fill = Color32::from_gray(bg);
        visuals.fg_stroke = Stroke::new(1.0, Color32::from_gray(fg));
        visuals.bg_stroke = Stroke::NONE;
        visuals.corner_radius = radius;
        visuals.expansion = 0.0;
    }
    w.noninteractive.corner_radius = radius;
    w.hovered.bg_stroke = Stroke::new(1.0, Color32::from_gray(80));
    w.active.bg_stroke = Stroke::new(1.0, ACCENT);
}
