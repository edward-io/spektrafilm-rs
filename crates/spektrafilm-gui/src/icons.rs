//! Icons from Phosphor (MIT), drawn from its icon font so they follow the
//! text colour and scale like text.
use eframe::egui::{FontFamily, FontId, RichText};
pub use egui_phosphor::regular::*;

/// The font family prioritizing icons, with text fallbacks (see `theme`).
pub const FAMILY: &str = "icons";

pub fn font(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(FAMILY.into()))
}

/// An icon sized for toolbar buttons.
pub fn icon(glyph: &str) -> RichText {
    RichText::new(glyph).font(font(15.0))
}
