//! Small UI icons drawn as geometry instead of font-dependent glyphs.
use eframe::egui::{self, Response, Ui, Widget};

pub struct RotateButton {
    clockwise: bool,
}

impl RotateButton {
    pub fn new(clockwise: bool) -> Self {
        Self { clockwise }
    }
}

impl Widget for RotateButton {
    fn ui(self, ui: &mut Ui) -> Response {
        let padding = ui.spacing().button_padding;
        let icon_size = ui.spacing().icon_width;
        let gap = ui.spacing().item_spacing.x;
        let text = ui.painter().layout_no_wrap(
            "Rotate".into(),
            egui::TextStyle::Button.resolve(ui.style()),
            ui.visuals().text_color(),
        );
        let size = egui::vec2(
            text.size().x + icon_size + gap + padding.x * 2.0,
            text.size().y.max(icon_size) + padding.y * 2.0,
        );
        let response = ui.add(egui::Button::new("").min_size(size));
        response.widget_info(|| {
            egui::WidgetInfo::labeled(
                egui::WidgetType::Button,
                ui.is_enabled(),
                if self.clockwise {
                    "Rotate clockwise"
                } else {
                    "Rotate counterclockwise"
                },
            )
        });
        if ui.is_rect_visible(response.rect) {
            let color = ui.style().interact(&response).fg_stroke.color;
            ui.painter().galley(
                egui::pos2(
                    response.rect.left() + padding.x,
                    response.rect.center().y - text.size().y / 2.0,
                ),
                text,
                color,
            );
            let center = egui::pos2(
                response.rect.right() - padding.x - icon_size / 2.0,
                response.rect.center().y,
            );
            let radius = icon_size * 0.34;
            let mirror = if self.clockwise { 1.0 } else { -1.0 };
            let points: Vec<_> = (0..=24)
                .map(|i| {
                    let angle = std::f32::consts::PI * (-1.75 + 1.5 * i as f32 / 24.0);
                    center + egui::vec2(mirror * angle.cos(), angle.sin()) * radius
                })
                .collect();
            let tip = points[24];
            let tangent = (tip - points[23]).normalized();
            let normal = egui::vec2(-tangent.y, tangent.x);
            let base = tip - tangent * (icon_size * 0.25);
            let wing = normal * (icon_size * 0.17);
            let stroke = egui::Stroke::new(1.5, color);
            ui.painter().line(points, stroke);
            ui.painter()
                .line(vec![base + wing, tip, base - wing], stroke);
        }
        response
    }
}
