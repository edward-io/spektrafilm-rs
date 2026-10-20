//! Crop & straighten (R), as in Lightroom: a level crop box over the whole
//! frame, with the image turning beneath it. Handles resize the box, dragging
//! inside moves the image, and dragging outside straightens it. The box never
//! leaves the image, so there are no blank corners.
//!
//! The crop is applied to the pipeline input, so straightening doesn't soften
//! grain, and grain and other spatial effects keep the size they have in the
//! whole frame. While the tool is open the whole frame is rendered; the crop
//! is applied (one re-render) when it closes, and recorded as one history step.
use super::*;
use crate::{chrome::toolbar_divider, icons, theme};
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use spektrafilm_io::crop::MAX_ANGLE;

/// Aspect ratio presets, long side ÷ short side; `None` is the frame's own.
const PRESETS: [(&str, Option<f64>); 6] = [
    ("Original", None),
    ("1 : 1", Some(1.)),
    ("4 : 5", Some(5. / 4.)),
    ("5 : 7", Some(7. / 5.)),
    ("2 : 3", Some(3. / 2.)),
    ("16 : 9", Some(16. / 9.)),
];

/// Smallest crop side, as a fraction of the frame's long side.
const MIN_SIDE: f64 = 0.02;

pub struct CropTool {
    /// The crop when the tool opened: restored by Cancel, and the start of
    /// the history step recorded when the crop is applied.
    pub original: Crop,
    /// Locked width ÷ height of the crop, in frame pixels; `None` is free.
    ratio: Option<f64>,
    drag: Option<Drag>,
    /// The frame point (pixels) kept at the canvas centre while resizing, so
    /// the box moves rather than the image; otherwise the crop centre.
    anchor: Option<[f64; 2]>,
    /// The crop when the angle slider was grabbed, so straightening and then
    /// levelling again restores the size it had.
    angle_start: Option<Crop>,
}

#[derive(Clone, Copy)]
enum Drag {
    Move(Crop),
    /// Which sides move: -1 (left / top), 0 or 1 (right / bottom).
    Resize(Crop, [i8; 2]),
    /// The pointer's angle about the box centre when the drag started.
    Rotate(Crop, f32),
}

impl CropTool {
    fn new(crop: Crop, frame: Frame) -> Self {
        let [w, h] = frame.size(&crop);
        Self {
            original: crop,
            ratio: Some(w / h),
            drag: None,
            anchor: None,
            angle_start: None,
        }
    }

    /// Keep up with a quarter turn of the frame.
    pub fn rotate(&mut self, turns: u8) {
        self.original = self.original.rotated_quarter_turns(turns);
        if turns % 2 == 1 {
            self.ratio = self.ratio.map(f64::recip);
        }
        self.drag = None;
        self.anchor = None;
        self.angle_start = None;
    }
}

/// The whole frame's size in pixels, for crop geometry.
#[derive(Clone, Copy)]
struct Frame {
    w: f64,
    h: f64,
}

impl Frame {
    fn of(image: &ImageBuf) -> Self {
        Self {
            w: image.width as f64,
            h: image.height as f64,
        }
    }
    fn centre(&self, crop: &Crop) -> [f64; 2] {
        [crop.center[0] * self.w, crop.center[1] * self.h]
    }
    fn size(&self, crop: &Crop) -> [f64; 2] {
        crop.pixels(self.w, self.h)
    }
    fn crop(&self, centre: [f64; 2], size: [f64; 2], angle: f64) -> Crop {
        Crop {
            center: [centre[0] / self.w, centre[1] / self.h],
            size: [size[0] / self.w, size[1] / self.h],
            angle,
        }
    }
    /// Half extents of the crop's bounding box in the frame.
    fn extents(&self, crop: &Crop) -> [f64; 2] {
        let [w, h] = self.size(crop);
        let (sin, cos) = crop.angle.to_radians().sin_cos();
        let (sin, cos) = (sin.abs(), cos.abs());
        [(w * cos + h * sin) / 2., (w * sin + h * cos) / 2.]
    }
    /// Whether all four corners lie inside the frame.
    fn fits(&self, crop: &Crop) -> bool {
        let [x, y] = self.centre(crop);
        let [ex, ey] = self.extents(crop);
        let slack = 1e-6 * self.w.max(self.h);
        x - ex >= -slack && x + ex <= self.w + slack && y - ey >= -slack && y + ey <= self.h + slack
    }
    /// The crop shrunk about its centre until it fits.
    fn shrink_to_fit(&self, crop: Crop) -> Crop {
        let [x, y] = self.centre(&crop);
        let [ex, ey] = self.extents(&crop);
        let scale = (x.min(self.w - x) / ex)
            .min(y.min(self.h - y) / ey)
            .clamp(0., 1.);
        let [w, h] = self.size(&crop);
        self.crop([x, y], [w * scale, h * scale], crop.angle)
    }
    /// Frame vector for an offset along the crop's own axes.
    fn along(&self, crop: &Crop, [u, v]: [f64; 2]) -> [f64; 2] {
        let (sin, cos) = crop.angle.to_radians().sin_cos();
        [u * cos + v * sin, -u * sin + v * cos]
    }
    fn min_side(&self) -> f64 {
        MIN_SIDE * self.w.max(self.h)
    }
}

/// Where the frame appears on screen: `anchor` (frame pixels) at `centre`,
/// `scale` screen points per frame pixel, turned clockwise by `angle`.
struct View {
    centre: Pos2,
    anchor: [f64; 2],
    scale: f64,
    angle: f64,
}

impl View {
    fn to_screen(&self, [x, y]: [f64; 2]) -> Pos2 {
        let (sin, cos) = self.angle.to_radians().sin_cos();
        let (dx, dy) = (x - self.anchor[0], y - self.anchor[1]);
        self.centre
            + Vec2::new(
                ((dx * cos - dy * sin) * self.scale) as f32,
                ((dx * sin + dy * cos) * self.scale) as f32,
            )
    }
    fn to_frame(&self, p: Pos2) -> [f64; 2] {
        let (sin, cos) = self.angle.to_radians().sin_cos();
        let d = p - self.centre;
        let (dx, dy) = (d.x as f64 / self.scale, d.y as f64 / self.scale);
        [
            self.anchor[0] + dx * cos + dy * sin,
            self.anchor[1] - dx * sin + dy * cos,
        ]
    }
}

/// Lightroom's crop of a catalog photo, in the frame after `rotation` quarter
/// turns; the whole frame for other photos.
pub fn lightroom_crop(
    options: &spektrafilm_io::RawOptions,
    frame: &ImageBuf,
    rotation: u8,
) -> Crop {
    let Some(geometry) = &options.lightroom_geometry else {
        return Crop::FULL;
    };
    let (w, h) = if rotation % 2 == 1 {
        (frame.height, frame.width)
    } else {
        (frame.width, frame.height)
    };
    let crop = Crop::from_lightroom(geometry, w as f64, h as f64).rotated_quarter_turns(rotation);
    // Rounding can put a corner a hair outside the frame.
    let crop = Frame::of(frame).shrink_to_fit(crop);
    if crop.validate().is_ok() {
        crop
    } else {
        Crop::FULL
    }
}

/// `frame` with `crop` applied, sharing the pixels when nothing is cropped.
pub fn cropped(frame: &Arc<ImageBuf>, crop: &Crop) -> Result<Arc<ImageBuf>, String> {
    if crop.is_full() {
        return Ok(frame.clone());
    }
    crop.apply(frame)
        .map(Arc::new)
        .map_err(|e| format!("crop: {e:#}"))
}

/// History label for a crop change.
pub fn describe(old: &Crop, new: &Crop) -> String {
    if new.is_full() {
        "Reset crop".into()
    } else if old.angle != new.angle {
        format!("Straighten {:.1}°", new.angle)
    } else {
        "Crop".into()
    }
}

impl App {
    /// The crop the pipeline input should have: the whole frame while the
    /// tool is open. `None` until a catalog photo's Lightroom crop is read.
    pub(crate) fn wanted_crop(&self) -> Option<Crop> {
        if self.crop_tool.is_some() {
            Some(Crop::FULL)
        } else {
            self.crop
        }
    }

    /// The crop as last applied (not the one being edited).
    pub(crate) fn committed_crop(&self) -> Crop {
        self.committed_crop_option().unwrap_or(Crop::FULL)
    }

    pub(crate) fn committed_crop_option(&self) -> Option<Crop> {
        self.crop_tool
            .as_ref()
            .map(|tool| tool.original)
            .or(self.crop)
    }

    /// Params for rendering an input cropped with `crop`: the film format
    /// shrinks with it, so grain keeps the size it has in the whole frame.
    pub(crate) fn pipeline_params(&self, crop: Crop) -> RuntimeParams {
        let mut params = self.params.clone();
        if let Some(frame) = &self.uncropped {
            params.camera.film_format_mm *=
                crop.long_side_fraction(frame.width as f64, frame.height as f64) as f32;
        }
        params
    }

    /// Rebuild the input when the wanted crop differs from the applied one
    /// (closing the tool, undo, a pasted or restored crop).
    pub(crate) fn sync_crop(&mut self) {
        let Some(crop) = self.wanted_crop() else {
            return;
        };
        if crop.validate().is_err() {
            self.crop = Some(Crop::FULL);
            return;
        }
        if crop != self.applied_crop && self.uncropped.is_some() && !self.image_loading() {
            self.reframe(0, crop);
        }
    }

    pub(crate) fn can_crop(&self) -> bool {
        self.uncropped.is_some() && self.crop.is_some()
    }

    pub(crate) fn toggle_crop_tool(&mut self) {
        if self.crop_tool.is_some() {
            self.close_crop_tool(true);
        } else {
            self.open_crop_tool();
        }
    }

    pub(crate) fn open_crop_tool(&mut self) {
        let (Some(frame), Some(crop)) = (&self.uncropped, self.crop) else {
            return;
        };
        self.crop_tool = Some(CropTool::new(crop, Frame::of(frame)));
        self.zoom_fit();
        self.display.compare_split = false;
    }

    /// Close the tool, keeping the crop (`apply`) or restoring the original.
    pub(crate) fn close_crop_tool(&mut self, apply: bool) {
        let Some(tool) = self.crop_tool.take() else {
            return;
        };
        if !apply {
            self.crop = Some(tool.original);
        }
        self.record_step();
    }

    /// Swap the crop between portrait and landscape (X).
    pub(crate) fn swap_crop_orientation(&mut self) {
        let (Some(tool), Some(frame), Some(crop)) =
            (self.crop_tool.as_mut(), &self.uncropped, self.crop)
        else {
            return;
        };
        let frame = Frame::of(frame);
        let [w, h] = frame.size(&crop);
        tool.ratio = tool.ratio.map(f64::recip);
        tool.anchor = None;
        self.crop = Some(frame.shrink_to_fit(frame.crop(frame.centre(&crop), [h, w], crop.angle)));
    }

    /// The canvas while cropping: the whole frame turned under a level box.
    pub(crate) fn crop_canvas(&mut self, ui: &mut egui::Ui) {
        let (Some(mut tool), Some(frame_image), Some(mut crop)) =
            (self.crop_tool.take(), self.uncropped.clone(), self.crop)
        else {
            return;
        };
        let frame = Frame::of(&frame_image);
        let response = ui.allocate_response(ui.available_size(), egui::Sense::click_and_drag());
        if let Some(path) = &self.image_path
            && let Some(error) = file_menu::show(&response, path)
        {
            self.status = error;
        }
        let rect = response.rect;
        let area = Self::image_area(rect).shrink(12.);
        let scale = (area.width() as f64 / frame.w).min(area.height() as f64 / frame.h);
        let view = |crop: &Crop, anchor: Option<[f64; 2]>| View {
            centre: area.center(),
            anchor: anchor.unwrap_or(frame.centre(crop)),
            scale,
            angle: crop.angle,
        };

        // Interaction, against the box as drawn last frame.
        let current = view(&crop, tool.anchor);
        let box_rect = |view: &View, crop: &Crop| {
            let [w, h] = frame.size(crop);
            Rect::from_center_size(
                view.to_screen(frame.centre(crop)),
                Vec2::new((w * view.scale) as f32, (h * view.scale) as f32),
            )
        };
        let bounds = box_rect(&current, &crop);
        let hit = |p: Pos2| -> Option<[i8; 2]> {
            let side = |v: f32, lo: f32, hi: f32| {
                if (v - lo).abs() < 10. {
                    -1
                } else if (v - hi).abs() < 10. {
                    1
                } else {
                    0
                }
            };
            let grown = bounds.expand(10.);
            if !grown.contains(p) {
                return None;
            }
            Some([
                side(p.x, bounds.left(), bounds.right()),
                side(p.y, bounds.top(), bounds.bottom()),
            ])
        };
        // The straightening cursor (egui has none) shows only while turning.
        let mut rotate_cursor = None;
        if let Some(p) = response.hover_pos().filter(|_| tool.drag.is_none()) {
            ui.ctx().set_cursor_icon(match hit(p) {
                Some([0, 0]) => egui::CursorIcon::Grab,
                Some([x, y]) => resize_cursor([x, y]),
                None => egui::CursorIcon::Default,
            });
        }
        if response.drag_started_by(egui::PointerButton::Primary)
            && let Some(origin) = ui.input(|i| i.pointer.press_origin())
        {
            tool.drag = Some(match hit(origin) {
                Some([0, 0]) => Drag::Move(crop),
                Some(sides) => {
                    tool.anchor = Some(current.anchor);
                    Drag::Resize(crop, sides)
                }
                None => Drag::Rotate(crop, pointer_angle(bounds.center(), origin)),
            });
        }
        if let (Some(drag), Some(origin), Some(pointer)) = (
            tool.drag,
            ui.input(|i| i.pointer.press_origin()),
            response.interact_pointer_pos(),
        ) && response.dragged_by(egui::PointerButton::Primary)
        {
            let delta = pointer - origin;
            crop = match drag {
                Drag::Move(start) => moved(frame, start, delta, scale),
                Drag::Resize(start, sides) => {
                    resized(frame, start, sides, delta, scale, tool.ratio)
                }
                Drag::Rotate(start, at) => {
                    let turn = (pointer_angle(bounds.center(), pointer) - at).to_degrees() as f64;
                    straightened(frame, start, start.angle + turn)
                }
            };
            ui.ctx().set_cursor_icon(match drag {
                Drag::Move(_) => egui::CursorIcon::Grabbing,
                Drag::Resize(_, sides) => resize_cursor(sides),
                Drag::Rotate(..) => {
                    rotate_cursor = Some(pointer);
                    egui::CursorIcon::None
                }
            });
        }
        let rotating = matches!(tool.drag, Some(Drag::Rotate(..))) || tool.angle_start.is_some();
        let dragging = tool.drag.is_some();
        if response.drag_stopped_by(egui::PointerButton::Primary) {
            tool.drag = None;
            tool.anchor = None;
        }
        // Double-click applies, as in Lightroom.
        let apply = response.double_clicked()
            && hit(response.interact_pointer_pos().unwrap_or_default()).is_some();

        // Drawing, with the updated crop.
        let view = view(&crop, tool.anchor);
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0., theme::WELL);
        self.paint_frame(&painter, &view, frame);
        let bounds = box_rect(&view, &crop);
        let shade = Color32::from_black_alpha(150);
        for outside in [
            Rect::from_min_max(rect.min, Pos2::new(rect.max.x, bounds.top())),
            Rect::from_min_max(Pos2::new(rect.min.x, bounds.bottom()), rect.max),
            Rect::from_min_max(
                Pos2::new(rect.min.x, bounds.top()),
                Pos2::new(bounds.left(), bounds.bottom()),
            ),
            Rect::from_min_max(
                Pos2::new(bounds.right(), bounds.top()),
                Pos2::new(rect.max.x, bounds.bottom()),
            ),
        ] {
            painter.rect_filled(outside, 0., shade);
        }
        let lines = if rotating { 9 } else { 3 };
        let grid = Stroke::new(
            1.,
            Color32::from_white_alpha(if dragging { 110 } else { 60 }),
        );
        for i in 1..lines {
            let t = i as f32 / lines as f32;
            let x = bounds.left() + bounds.width() * t;
            let y = bounds.top() + bounds.height() * t;
            painter.line_segment(
                [Pos2::new(x, bounds.top()), Pos2::new(x, bounds.bottom())],
                grid,
            );
            painter.line_segment(
                [Pos2::new(bounds.left(), y), Pos2::new(bounds.right(), y)],
                grid,
            );
        }
        painter.rect_stroke(
            bounds,
            0.,
            Stroke::new(1., Color32::WHITE),
            egui::StrokeKind::Outside,
        );
        paint_handles(&painter, bounds);
        if rotating {
            canvas_tag(
                &painter,
                bounds.center_top() - Vec2::new(0., 10.),
                egui::Align2::CENTER_BOTTOM,
                &format!("{:.1}°", crop.angle),
            );
        }

        if let Some(p) = rotate_cursor {
            let layer = egui::LayerId::new(egui::Order::Tooltip, response.id.with("rotate-cursor"));
            paint_rotate_cursor(&ui.ctx().layer_painter(layer), p);
        }

        self.crop = Some(crop);
        self.crop_tool = Some(tool);
        if apply {
            self.close_crop_tool(true);
        }
    }

    /// The whole frame through `view`: the render if it is of the whole
    /// frame already, otherwise the unprocessed input.
    fn paint_frame(&self, painter: &egui::Painter, view: &View, frame: Frame) {
        let matches = |size: [usize; 2]| {
            let aspect = size[0] as f64 / size[1] as f64;
            (aspect / (frame.w / frame.h) - 1.).abs() < 0.01
        };
        if let Some(canvas) = &self.canvas {
            let image = [&self.canvas_output, &self.canvas_input]
                .into_iter()
                .flatten()
                .find(|image| matches(image.size));
            if let Some(image) = image {
                let texels = image.size[0] as f64 / frame.w;
                canvas.paint_mapped(painter, image, &self.display.output_interpolation, |p| {
                    let [x, y] = view.to_frame(p);
                    Pos2::new((x * texels) as f32, (y * texels) as f32)
                });
            }
            return;
        }
        let texture = [&self.output_tex, &self.input_tex]
            .into_iter()
            .flatten()
            .find(|t| matches(t.size()));
        if let Some(texture) = texture {
            let mut mesh = egui::Mesh::with_texture(texture.id());
            let corners = [[0., 0.], [frame.w, 0.], [frame.w, frame.h], [0., frame.h]];
            let uvs = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
            for (corner, uv) in corners.into_iter().zip(uvs) {
                mesh.vertices.push(egui::epaint::Vertex {
                    pos: view.to_screen(corner),
                    uv: Pos2::new(uv[0], uv[1]),
                    color: Color32::WHITE,
                });
            }
            mesh.indices.extend([0, 1, 2, 0, 2, 3]);
            painter.add(mesh);
        }
    }

    /// Replaces the canvas toolbar while cropping.
    pub(crate) fn crop_toolbar(&mut self, ui: &mut egui::Ui) {
        let (Some(mut tool), Some(frame_image), Some(mut crop)) =
            (self.crop_tool.take(), self.uncropped.clone(), self.crop)
        else {
            return;
        };
        let frame = Frame::of(&frame_image);
        let [w, h] = frame.size(&crop);
        let landscape = w >= h;
        let preset_ratio = |long_short: Option<f64>| {
            let r = long_short.unwrap_or((frame.w / frame.h).max(frame.h / frame.w));
            if landscape { r } else { 1. / r }
        };
        let selected = match tool.ratio {
            None => "Free".to_owned(),
            Some(r) => PRESETS
                .iter()
                .find(|(_, preset)| (preset_ratio(*preset) / r - 1.).abs() < 1e-3)
                .map_or("Custom".to_owned(), |(name, _)| (*name).to_owned()),
        };
        egui::ComboBox::from_id_salt("crop-aspect")
            .selected_text(selected)
            .width(86.)
            .show_ui(ui, |ui| {
                for (name, preset) in PRESETS {
                    if ui.selectable_label(false, name).clicked() {
                        let ratio = preset_ratio(preset);
                        tool.ratio = Some(ratio);
                        // The largest box of that shape inside the current one.
                        let width = w.min(h * ratio);
                        crop = frame.crop(frame.centre(&crop), [width, width / ratio], crop.angle);
                    }
                }
                if ui.selectable_label(tool.ratio.is_none(), "Free").clicked() {
                    tool.ratio = None;
                }
            })
            .response
            .on_hover_text("Aspect ratio of the crop. Press X to swap portrait and landscape.");
        let locked = tool.ratio.is_some();
        if ui
            .add(
                egui::Button::new(icons::icon(if locked {
                    icons::LOCK_SIMPLE
                } else {
                    icons::LOCK_SIMPLE_OPEN
                }))
                .selected(locked),
            )
            .on_hover_text(if locked {
                "Aspect ratio locked; click to crop freely"
            } else {
                "Lock the current aspect ratio"
            })
            .clicked()
        {
            tool.ratio = if locked { None } else { Some(w / h) };
        }
        toolbar_divider(ui);
        ui.label(egui::RichText::new("Angle").small().color(theme::TEXT_WEAK));
        let mut angle = crop.angle;
        ui.spacing_mut().slider_width = 120.;
        let slider = ui
            .add(
                egui::Slider::new(&mut angle, -MAX_ANGLE..=MAX_ANGLE)
                    .step_by(0.1)
                    .suffix("°")
                    .fixed_decimals(1),
            )
            .on_hover_text(
                "Straighten. You can also drag outside the crop. Double-click to level.",
            );
        if slider.double_clicked() {
            angle = 0.;
        }
        if angle != crop.angle {
            let start = *tool.angle_start.get_or_insert(crop);
            crop = straightened(frame, start, angle);
        }
        if !slider.dragged() && !slider.has_focus() {
            tool.angle_start = None;
        }
        if ui
            .button("Reset")
            .on_hover_text("Remove the crop and straightening")
            .clicked()
        {
            crop = Crop::FULL;
            if tool.ratio.is_some() {
                tool.ratio = Some(frame.w / frame.h);
            }
        }
        toolbar_divider(ui);
        let cancel = ui.button("Cancel").on_hover_text("Esc").clicked();
        let done = ui
            .add(egui::Button::new(
                egui::RichText::new("Done").color(theme::ACCENT),
            ))
            .on_hover_text("Apply the crop (Enter or R)")
            .clicked();
        self.crop = Some(crop);
        self.crop_tool = Some(tool);
        if cancel || done {
            self.close_crop_tool(done);
        }
    }
}

fn pointer_angle(centre: Pos2, p: Pos2) -> f32 {
    let d = p - centre;
    d.y.atan2(d.x)
}

/// The straightening cursor (egui has none): Phosphor's rotate arrows,
/// white with a dark halo so it shows on any image.
fn paint_rotate_cursor(painter: &egui::Painter, p: Pos2) {
    let font = icons::font(22.);
    let halo = Color32::from_black_alpha(170);
    for i in 0..8 {
        let offset = Vec2::angled(i as f32 * std::f32::consts::FRAC_PI_4) * 1.5;
        painter.text(
            p + offset,
            egui::Align2::CENTER_CENTER,
            icons::ARROWS_CLOCKWISE,
            font.clone(),
            halo,
        );
    }
    painter.text(
        p,
        egui::Align2::CENTER_CENTER,
        icons::ARROWS_CLOCKWISE,
        font,
        Color32::WHITE,
    );
}

fn resize_cursor(sides: [i8; 2]) -> egui::CursorIcon {
    match sides {
        [0, _] => egui::CursorIcon::ResizeVertical,
        [_, 0] => egui::CursorIcon::ResizeHorizontal,
        [x, y] if x == y => egui::CursorIcon::ResizeNwSe,
        _ => egui::CursorIcon::ResizeNeSw,
    }
}

/// Dragging inside: the image follows the pointer under the box, stopping at
/// the frame's edges.
fn moved(frame: Frame, start: Crop, delta: Vec2, scale: f64) -> Crop {
    let [dx, dy] = frame.along(&start, [delta.x as f64 / scale, delta.y as f64 / scale]);
    let [x, y] = frame.centre(&start);
    let [ex, ey] = frame.extents(&start);
    let centre = [
        (x - dx).clamp(ex.min(frame.w / 2.), (frame.w - ex).max(frame.w / 2.)),
        (y - dy).clamp(ey.min(frame.h / 2.), (frame.h - ey).max(frame.h / 2.)),
    ];
    frame.crop(centre, frame.size(&start), start.angle)
}

/// Dragging a handle: the opposite side stays put; a locked ratio keeps its
/// shape. The drag stops where the box would leave the frame.
fn resized(
    frame: Frame,
    start: Crop,
    sides: [i8; 2],
    delta: Vec2,
    scale: f64,
    ratio: Option<f64>,
) -> Crop {
    let candidate = |tu: f64, tv: f64| {
        let [w, h] = frame.size(&start);
        let (mut x0, mut x1, mut y0, mut y1) = (-w / 2., w / 2., -h / 2., h / 2.);
        let (du, dv) = (delta.x as f64 / scale * tu, delta.y as f64 / scale * tv);
        let min = frame.min_side();
        match sides[0] {
            1 => x1 = (x1 + du).max(x0 + min),
            -1 => x0 = (x0 + du).min(x1 - min),
            _ => {}
        }
        match sides[1] {
            1 => y1 = (y1 + dv).max(y0 + min),
            -1 => y0 = (y0 + dv).min(y1 - min),
            _ => {}
        }
        if let Some(ratio) = ratio {
            let (mut w, mut h) = (x1 - x0, y1 - y0);
            match sides {
                [_, 0] => h = w / ratio,
                [0, _] => w = h * ratio,
                _ if w / h > ratio => h = w / ratio,
                _ => w = h * ratio,
            }
            // Anchor the opposite side (or keep centred on an unmoved axis).
            match sides[0] {
                1 => x1 = x0 + w,
                -1 => x0 = x1 - w,
                _ => (x0, x1) = (-w / 2., w / 2.),
            }
            match sides[1] {
                1 => y1 = y0 + h,
                -1 => y0 = y1 - h,
                _ => (y0, y1) = (-h / 2., h / 2.),
            }
        }
        let [cx, cy] = frame.centre(&start);
        let [ox, oy] = frame.along(&start, [(x0 + x1) / 2., (y0 + y1) / 2.]);
        frame.crop([cx + ox, cy + oy], [x1 - x0, y1 - y0], start.angle)
    };
    let full = candidate(1., 1.);
    if frame.fits(&full) {
        return full;
    }
    // The furthest point along the drag that still fits.
    let furthest = |fits: &dyn Fn(f64) -> bool| {
        let (mut lo, mut hi) = (0., 1.);
        for _ in 0..24 {
            let mid = (lo + hi) / 2.;
            if fits(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    if ratio.is_some() {
        let t = furthest(&|t| frame.fits(&candidate(t, t)));
        return candidate(t, t);
    }
    // A free box stops at each edge separately, so it slides along the other.
    let tu = furthest(&|t| frame.fits(&candidate(t, 0.)));
    let tv = furthest(&|t| frame.fits(&candidate(tu, t)));
    candidate(tu, tv)
}

/// The crop at `angle`, shrunk about its centre to stay inside the image.
/// Always derived from where straightening started, so levelling again gives
/// the original size back.
fn straightened(frame: Frame, start: Crop, angle: f64) -> Crop {
    frame.shrink_to_fit(Crop {
        angle: angle.clamp(-MAX_ANGLE, MAX_ANGLE),
        ..start
    })
}

fn paint_handles(painter: &egui::Painter, bounds: Rect) {
    let stroke = Stroke::new(3., Color32::WHITE);
    let arm = 16f32.min(bounds.width() / 3.).min(bounds.height() / 3.);
    for (corner, dx, dy) in [
        (bounds.left_top(), 1., 1.),
        (bounds.right_top(), -1., 1.),
        (bounds.right_bottom(), -1., -1.),
        (bounds.left_bottom(), 1., -1.),
    ] {
        painter.line(
            vec![
                corner + Vec2::new(dx * arm, 0.),
                corner,
                corner + Vec2::new(0., dy * arm),
            ],
            stroke,
        );
    }
    let half = arm / 2.;
    for (mid, horizontal) in [
        (bounds.center_top(), true),
        (bounds.center_bottom(), true),
        (bounds.left_center(), false),
        (bounds.right_center(), false),
    ] {
        let d = if horizontal {
            Vec2::new(half, 0.)
        } else {
            Vec2::new(0., half)
        };
        painter.line_segment([mid - d, mid + d], stroke);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: Frame = Frame { w: 3000., h: 2000. };

    #[test]
    fn straightening_shrinks_to_stay_inside_and_levelling_restores_it() {
        let tilted = straightened(FRAME, Crop::FULL, 10.);
        assert!(FRAME.fits(&tilted));
        assert!(tilted.size[0] < 1. && (tilted.size[0] / tilted.size[1] - 1.).abs() < 1e-9);
        assert_eq!(straightened(FRAME, Crop::FULL, 0.), Crop::FULL);
        assert_eq!(straightened(FRAME, Crop::FULL, 80.).angle, MAX_ANGLE);
    }

    #[test]
    fn moving_stops_at_the_frame_edge() {
        let half = FRAME.crop([1500., 1000.], [1500., 1000.], 0.);
        // The image follows the pointer, so the crop moves the other way.
        let moved = moved(FRAME, half, Vec2::new(10_000., 0.), 1.);
        assert_eq!(FRAME.centre(&moved), [750., 1000.]);
        assert!(FRAME.fits(&moved));
    }

    #[test]
    fn resizing_keeps_the_opposite_side_and_the_ratio() {
        let start = FRAME.crop([1500., 1000.], [1500., 1000.], 0.);
        // Drag the right edge out by 300 px with the ratio locked at 3:2.
        let wider = resized(FRAME, start, [1, 0], Vec2::new(300., 0.), 1., Some(1.5));
        let [w, h] = FRAME.size(&wider);
        assert!((w - 1800.).abs() < 1e-6 && (h - 1200.).abs() < 1e-6);
        let [x, _] = FRAME.centre(&wider);
        assert!((x - w / 2. - 750.).abs() < 1e-6, "left edge stays");
        // A drag far past the frame stops at its edge.
        let edge = resized(FRAME, start, [1, 1], Vec2::new(1e5, 1e5), 1., None);
        assert!(FRAME.fits(&edge));
        assert!((FRAME.centre(&edge)[0] + FRAME.size(&edge)[0] / 2. - 3000.).abs() < 1.);
        // Never smaller than the minimum.
        let tiny = resized(FRAME, start, [1, 0], Vec2::new(-1e5, 0.), 1., None);
        assert!((FRAME.size(&tiny)[0] - FRAME.min_side()).abs() < 1e-6);
    }

    #[test]
    fn screen_mapping_round_trips_and_keeps_the_box_level() {
        let crop = Crop {
            center: [0.4, 0.55],
            size: [0.5, 0.4],
            angle: 12.,
        };
        let view = View {
            centre: Pos2::new(400., 300.),
            anchor: FRAME.centre(&crop),
            scale: 0.2,
            angle: crop.angle,
        };
        let p = [1234., 567.];
        let back = view.to_frame(view.to_screen(p));
        assert!((back[0] - p[0]).abs() < 1e-3 && (back[1] - p[1]).abs() < 1e-3);
        // The crop's corners land on a level rectangle.
        let [w, h] = FRAME.size(&crop);
        let corner = |u: f64, v: f64| {
            let [ox, oy] = FRAME.along(&crop, [u * w / 2., v * h / 2.]);
            let [cx, cy] = FRAME.centre(&crop);
            view.to_screen([cx + ox, cy + oy])
        };
        assert!((corner(-1., -1.).y - corner(1., -1.).y).abs() < 1e-3);
        assert!((corner(1., -1.).x - corner(1., 1.).x).abs() < 1e-3);
    }

    #[test]
    fn history_labels() {
        let tilted = Crop {
            angle: 2.5,
            ..Crop::FULL
        };
        assert_eq!(describe(&Crop::FULL, &tilted), "Straighten 2.5°");
        assert_eq!(describe(&tilted, &Crop::FULL), "Reset crop");
        let smaller = Crop {
            size: [0.5, 0.5],
            ..Crop::FULL
        };
        assert_eq!(describe(&Crop::FULL, &smaller), "Crop");
    }

    fn finish(app: &mut App) {
        while let Some(job) = &mut app.load_job {
            if let Some(handle) = job.handle.take() {
                handle.join().unwrap();
            }
            app.poll_load_job();
        }
        assert!(app.image.is_some(), "{}", app.status);
    }

    fn loaded_app(dir: &std::path::Path) -> App {
        let path = dir.join("frame.png");
        image::RgbImage::from_fn(60, 40, |x, y| image::Rgb([x as u8 * 4, y as u8 * 6, 90]))
            .save(&path)
            .unwrap();
        let cc = eframe::CreationContext::_new_kittest(egui::Context::default());
        let mut app = App::new(
            &cc,
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            None,
        );
        app.load_image_from_path(&path);
        finish(&mut app);
        app
    }

    #[test]
    fn project_reopen_restores_the_saved_crop_after_rotation() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = loaded_app(dir.path());
        let saved = Crop {
            center: [0.25, 0.5],
            size: [0.5, 0.5],
            angle: 0.,
        };
        app.crop = Some(saved);
        app.sync_crop();
        finish(&mut app);
        let expected = app.image.clone().unwrap();
        let project = dir.path().join("saved.json");
        app.project_state().save(&project).unwrap();

        for turns in 1..4 {
            app.rotate_input(turns);
            finish(&mut app);
            app.restore_project(&project, true).unwrap();
            finish(&mut app);
            app.sync_crop();
            finish(&mut app);
            assert_eq!(app.rotation, 0);
            assert_eq!(app.crop, Some(saved));
            assert_eq!(app.applied_crop, saved);
            assert_eq!(app.image.as_ref().unwrap().data, expected.data);
        }
    }

    #[test]
    fn project_reopen_replaces_the_active_crop_session() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = loaded_app(dir.path());
        let left = Crop {
            center: [0.25, 0.5],
            size: [0.5, 0.5],
            angle: 0.,
        };
        app.crop = Some(left);
        app.sync_crop();
        finish(&mut app);
        let right = Crop {
            center: [0.75, 0.5],
            ..left
        };
        let mut state = app.project_state();
        state.crop = Some(right);
        let project = dir.path().join("saved.json");
        state.save(&project).unwrap();

        app.open_crop_tool();
        app.crop = Some(Crop::FULL);
        app.sync_crop();
        finish(&mut app);
        assert_eq!(app.applied_crop, Crop::FULL);
        assert_eq!(app.committed_crop(), left);

        app.restore_project(&project, true).unwrap();
        finish(&mut app);
        assert!(app.crop_tool.is_none());
        assert_eq!(app.applied_crop, right);
        assert_eq!(app.committed_crop(), right);
        assert_eq!(app.project_state().crop, Some(right));
        let expected = right.apply(app.uncropped.as_ref().unwrap()).unwrap();
        assert_eq!(app.image.as_ref().unwrap().data, expected.data);
    }

    #[test]
    fn cropping_is_one_undoable_step_that_reframes_without_decoding() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = loaded_app(dir.path());
        assert_eq!(app.crop, Some(Crop::FULL));
        let frame = app.uncropped.clone().unwrap();
        let format = app.params.camera.film_format_mm;

        // Edits inside the tool are one step, recorded when it closes.
        app.open_crop_tool();
        let half = Crop {
            center: [0.25, 0.5],
            size: [0.5, 0.5],
            angle: 0.,
        };
        app.crop = Some(half);
        assert!(!app.record_step(), "nothing recorded while cropping");
        app.close_crop_tool(true);
        assert_eq!(app.history.undo_label(), Some("Crop"));
        app.sync_crop();
        finish(&mut app);
        assert!(
            Arc::ptr_eq(app.uncropped.as_ref().unwrap(), &frame),
            "not decoded again"
        );
        let image = app.image.clone().unwrap();
        assert_eq!((image.width, image.height), (30, 20));
        assert_eq!(app.applied_crop, half);
        assert_eq!(
            app.pipeline_params(app.applied_crop).camera.film_format_mm,
            format * 0.5
        );

        // Reopening shows the whole frame; cancelling keeps the applied crop.
        app.open_crop_tool();
        app.sync_crop();
        finish(&mut app);
        assert_eq!(app.image.as_ref().unwrap().width, 60);
        app.crop = Some(Crop::FULL);
        app.close_crop_tool(false);
        assert_eq!(app.crop, Some(half));
        app.sync_crop();
        finish(&mut app);
        assert_eq!(app.image.as_ref().unwrap().width, 30);

        // Quarter turns carry the crop round with the frame, and aren't steps.
        app.rotate_input(1);
        finish(&mut app);
        let image = app.image.clone().unwrap();
        assert_eq!((image.width, image.height), (20, 30));
        assert_eq!(app.crop, Some(half.rotated_quarter_turns(1)));
        assert!(!app.record_step());
        assert_eq!(app.current_edit().crop, app.crop, "saved with its rotation");

        // Undo returns to the whole frame.
        app.rotate_input(3);
        finish(&mut app);
        app.undo();
        assert_eq!(app.crop, Some(Crop::FULL));
        app.sync_crop();
        finish(&mut app);
        assert_eq!(app.image.as_ref().unwrap().width, 60);
    }
}
