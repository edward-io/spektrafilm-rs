//! Read-only Lightroom Classic catalog browser: a thumbnail grid in the left
//! panel that feeds the selected photo, with its crop, into the editor.
use crate::{inspector, theme};
use eframe::egui::{self, RichText};
use spektrafilm_gpu::scheduler::{Priority, Scheduler, Task};
use spektrafilm_io::lightroom::{Catalog, Photo, THUMBNAIL_PX};
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::mpsc,
};

pub enum Action {
    Select(Photo),
    Empty,
    Close,
}

pub struct Browser {
    scheduler: Scheduler,
    thumbnail_task: Task,
    path: PathBuf,
    catalog: Option<Catalog>,
    loading: Option<mpsc::Receiver<Result<Catalog, String>>>,
    error: Option<String>,
    accepted_only: bool,
    visible: Vec<usize>,
    pub selected: Option<usize>,
    scroll_to_selected: bool,
    thumbnails: HashMap<i64, Option<egui::TextureHandle>>,
    pending_thumbnails: HashSet<i64>,
    thumbnail_tx: mpsc::SyncSender<(i64, PathBuf, u16)>,
    thumbnail_rx: mpsc::Receiver<(i64, Option<image::RgbaImage>)>,
    /// Photos with settings saved in Spektrafilm, marked in the list.
    pub edited: HashSet<i64>,
}

impl Drop for Browser {
    fn drop(&mut self) {
        self.thumbnail_task.cancel();
    }
}

impl Browser {
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub(super) fn source_thumbnail(
        &mut self,
        ctx: &egui::Context,
        path: &std::path::Path,
        preview: &super::PreviewTextureData,
    ) {
        let Some(photo) = self
            .selected
            .and_then(|i| self.catalog.as_ref()?.photos.get(i))
        else {
            return;
        };
        if photo.path != path || self.thumbnails.get(&photo.id).is_some_and(Option::is_some) {
            return;
        }
        let Some(image) = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
            preview.size[0] as u32,
            preview.size[1] as u32,
            preview.rgba.as_slice(),
        ) else {
            return;
        };
        let scale = (THUMBNAIL_PX as f32 / preview.size[0].max(preview.size[1]) as f32).min(1.);
        let thumbnail = image::imageops::thumbnail(
            &image,
            (preview.size[0] as f32 * scale).round().max(1.) as u32,
            (preview.size[1] as f32 * scale).round().max(1.) as u32,
        );
        self.thumbnails.insert(
            photo.id,
            Some(ctx.load_texture(
                format!("catalog-{}", photo.id),
                egui::ColorImage::from_rgba_unmultiplied(
                    [thumbnail.width() as usize, thumbnail.height() as usize],
                    thumbnail.as_raw(),
                ),
                egui::TextureOptions::LINEAR,
            )),
        );
    }

    pub fn open(path: PathBuf, ctx: &egui::Context, scheduler: &Scheduler) -> Self {
        let (thumbnail_tx, requests) = mpsc::sync_channel::<(i64, PathBuf, u16)>(24);
        let (results, thumbnail_rx) = mpsc::channel();
        let repaint = ctx.clone();
        let thumbnail_task = scheduler.task(Priority::Background);
        let task = thumbnail_task.clone();
        std::thread::spawn(move || {
            let _scope = task.enter();
            while let Ok((id, path, orientation)) = requests.recv() {
                let image = {
                    let Ok(_cpu) = task.cpu() else { break };
                    std::panic::catch_unwind(|| {
                        spektrafilm_io::lightroom::thumbnail(&path, orientation)
                    })
                    .ok()
                    .and_then(Result::ok)
                };
                if results.send((id, image)).is_err() {
                    break;
                }
                repaint.request_repaint();
            }
        });
        let mut browser = Self {
            scheduler: scheduler.clone(),
            thumbnail_task,
            path,
            catalog: None,
            loading: None,
            error: None,
            accepted_only: false,
            visible: Vec::new(),
            selected: None,
            scroll_to_selected: false,
            thumbnails: HashMap::new(),
            pending_thumbnails: HashSet::new(),
            thumbnail_tx,
            thumbnail_rx,
            edited: HashSet::new(),
        };
        browser.refresh(ctx);
        browser
    }

    fn refresh(&mut self, ctx: &egui::Context) {
        if self.loading.is_some() {
            return;
        }
        let path = self.path.clone();
        let ctx = ctx.clone();
        let (tx, rx) = mpsc::channel();
        let task = self.scheduler.task(Priority::Interactive);
        std::thread::spawn(move || {
            let result = task
                .cpu()
                .map_err(|e| e.to_string())
                .and_then(|_cpu| Catalog::open(&path).map_err(|e| format!("{e:#}")));
            let _ = tx.send(result);
            ctx.request_repaint();
        });
        self.loading = Some(rx);
        self.error = None;
    }

    fn filter(&mut self) {
        self.visible = self
            .catalog
            .as_ref()
            .map(|catalog| {
                catalog
                    .photos
                    .iter()
                    .enumerate()
                    .filter_map(|(i, photo)| {
                        (photo.available && (!self.accepted_only || photo.flag == 1)).then_some(i)
                    })
                    .collect()
            })
            .unwrap_or_default();
    }

    fn select(&mut self, position: usize) -> Option<Action> {
        let index = *self.visible.get(position)?;
        self.selected = Some(index);
        self.scroll_to_selected = true;
        Some(Action::Select(self.catalog.as_ref()?.photos[index].clone()))
    }

    pub fn show(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) -> Option<Action> {
        let mut action = None;
        if let Some(rx) = &self.loading {
            match rx.try_recv() {
                Ok(result) => {
                    self.loading = None;
                    match result {
                        Ok(catalog) => {
                            let selected_id = self
                                .selected
                                .and_then(|i| self.catalog.as_ref()?.photos.get(i))
                                .map(|p| p.id);
                            self.selected = selected_id
                                .and_then(|id| catalog.photos.iter().position(|p| p.id == id));
                            self.catalog = Some(catalog);
                            self.filter();
                            let position = self
                                .selected
                                .and_then(|index| self.visible.iter().position(|i| *i == index))
                                .unwrap_or(0);
                            action = self.select(position).or(Some(Action::Empty));
                        }
                        Err(error) => self.error = Some(error),
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.loading = None;
                    self.error = Some("Catalog reader stopped unexpectedly".into());
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        while let Ok((id, image)) = self.thumbnail_rx.try_recv() {
            self.pending_thumbnails.remove(&id);
            if image.is_none() && self.thumbnails.get(&id).is_some_and(Option::is_some) {
                continue;
            }
            if self.thumbnails.len() >= 256 {
                if let Some(id) = self.thumbnails.keys().next().copied() {
                    self.thumbnails.remove(&id);
                }
            }
            self.thumbnails.insert(
                id,
                image.map(|image| {
                    ctx.load_texture(
                        format!("catalog-{id}"),
                        egui::ColorImage::from_rgba_unmultiplied(
                            [image.width() as usize, image.height() as usize],
                            image.as_raw(),
                        ),
                        egui::TextureOptions::LINEAR,
                    )
                }),
            );
        }
        // ── Header: catalog name, refresh, close ────────────────────────
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(
                inspector::title(&self.path.file_stem().unwrap_or_default().to_string_lossy())
                    .size(14.0),
            )
            .on_hover_text(self.path.display().to_string());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("×")
                    .on_hover_text("Close the catalog")
                    .clicked()
                {
                    action = Some(Action::Close);
                }
                if ui
                    .add_enabled(self.loading.is_none(), egui::Button::new("⟳").small())
                    .on_hover_text("Re-read the catalog, e.g. after reconnecting a drive")
                    .clicked()
                {
                    self.refresh(ctx);
                }
            });
        });
        let catalog_stats = self.catalog.as_ref().map(|catalog| {
            let picks = catalog
                .photos
                .iter()
                .filter(|p| p.flag == 1 && p.available)
                .count();
            let available = catalog.photos.iter().filter(|p| p.available).count();
            let unavailable = catalog.photos.len() - available;
            (available, picks, unavailable)
        });
        ui.label(
            RichText::new(match catalog_stats {
                Some((available, _, _)) => {
                    format!("Lightroom Classic · {available} photos · read-only")
                }
                None => "Lightroom Classic · read-only".into(),
            })
            .small()
            .color(theme::TEXT_WEAK),
        );
        ui.add_space(4.0);

        // ── Filter ──────────────────────────────────────────────────────
        if let Some((available, picks, _)) = catalog_stats {
            let mut accepted_only = self.accepted_only;
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 2.0;
                let width = (ui.available_width() - 2.0) / 2.0;
                for (value, text) in [
                    (false, format!("All  {available}")),
                    (true, format!("Picks  {picks}")),
                ] {
                    let response = ui.add_sized(
                        [width, 22.0],
                        egui::SelectableLabel::new(accepted_only == value, text),
                    );
                    if response.clicked() {
                        response.surrender_focus();
                        accepted_only = value;
                    }
                }
            });
            if accepted_only != self.accepted_only {
                self.accepted_only = accepted_only;
                self.filter();
                if !self.selected.is_some_and(|i| self.visible.contains(&i)) {
                    self.selected = None;
                    action = self.select(0).or(Some(Action::Empty));
                }
            }
        }

        // ── Notices ─────────────────────────────────────────────────────
        if self.loading.is_some() {
            ui.horizontal(|ui| {
                ui.add(egui::Spinner::new().size(12.0));
                ui.label(
                    RichText::new("Reading catalog…")
                        .small()
                        .color(theme::TEXT_WEAK),
                );
            });
        }
        if let Some(error) = &self.error {
            notice(ui, ui.visuals().error_fg_color, error);
        }
        let Some(catalog) = &self.catalog else {
            return action;
        };
        if let Some(warning) = catalog.compatibility_warning() {
            notice(ui, ui.visuals().warn_fg_color, &warning);
        }
        let unavailable = catalog_stats.map_or(0, |(_, _, u)| u);
        if unavailable > 0 {
            notice(
                ui,
                theme::TEXT_WEAK,
                &format!(
                    "{unavailable} photo{} skipped: original not found. Connect the drive, then ⟳.",
                    if unavailable == 1 { "" } else { "s" }
                ),
            );
        }
        if let Some(photo) = self.selected.and_then(|i| catalog.photos.get(i))
            && photo.framing_warning
        {
            notice(
                ui,
                ui.visuals().warn_fg_color,
                "Lightroom lens and perspective corrections aren't imported, so framing may differ.",
            );
        }
        ui.add_space(2.0);

        // ── Navigation ──────────────────────────────────────────────────
        let position = self
            .selected
            .and_then(|i| self.visible.iter().position(|v| *v == i));
        // The list lives inside a scroll area whose bar takes width of its own;
        // rows sized to the full panel overflow it and egui widens the panel.
        let list_width = ui.available_width() - ui.spacing().scroll.allocated_width();
        let mut next_position = None;
        if !ctx.wants_keyboard_input() {
            let step =
                |key: egui::Key| ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, key));
            let last = self.visible.len().saturating_sub(1);
            if step(egui::Key::ArrowLeft) || step(egui::Key::ArrowUp) {
                next_position = position.and_then(|p| p.checked_sub(1));
            }
            if step(egui::Key::ArrowRight) || step(egui::Key::ArrowDown) {
                next_position = position.map(|p| p + 1).filter(|p| *p <= last);
            }
        }

        // ── Footer (drawn first so the grid can take the remaining height) ─
        egui::TopBottomPanel::bottom("catalog-footer")
            .frame(egui::Frame::new().inner_margin(egui::Margin::symmetric(0, 6)))
            .show_separator_line(true)
            .show_inside(ui, |ui| {
                ui.horizontal(|ui| {
                    let weak = |s: String| RichText::new(s).small().color(theme::TEXT_WEAK);
                    ui.label(weak(match position {
                        Some(p) => format!("{} of {}", p + 1, self.visible.len()),
                        None => format!("{} photos", self.visible.len()),
                    }));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(weak("↑ ↓ to browse".into()));
                    });
                });
            });

        // ── Thumbnail grid ──────────────────────────────────────────────
        if self.visible.is_empty() {
            ui.add_space(24.0);
            ui.vertical_centered(|ui| {
                ui.label(
                    RichText::new(if unavailable == catalog.photos.len() && unavailable > 0 {
                        "Originals are unavailable.\nConnect the photo drive, then ⟳."
                    } else if self.accepted_only {
                        "No picks in this catalog."
                    } else {
                        "No photos in this catalog."
                    })
                    .color(theme::TEXT_WEAK),
                );
            });
            if let Some(position) = next_position {
                action = self.select(position);
            }
            return action;
        }
        let gap = 6.0;
        let frame = egui::vec2(list_width.floor(), (list_width * 2.0 / 3.0).floor());
        let caption_height = 16.0;
        let row_height = frame.y + caption_height;
        let mut scroll = egui::ScrollArea::vertical()
            .id_salt("catalog-photos")
            .auto_shrink([false; 2]);
        if self.scroll_to_selected {
            if let Some(position) = position {
                // Keep the selection one row below the top so its neighbour shows.
                let offset = (position as f32 - 1.0).max(0.0) * (row_height + gap);
                scroll = scroll.vertical_scroll_offset(offset);
            }
            self.scroll_to_selected = false;
        }
        ui.spacing_mut().item_spacing.y = gap;
        scroll.show_rows(ui, row_height, self.visible.len(), |ui, range| {
            for position in range {
                let index = self.visible[position];
                let photo = &catalog.photos[index];
                if !self.thumbnails.contains_key(&photo.id)
                    && !self.pending_thumbnails.contains(&photo.id)
                {
                    let orientation = photo.geometry.as_ref().map(|g| g.orientation).unwrap_or(1);
                    if self
                        .thumbnail_tx
                        .try_send((photo.id, photo.path.clone(), orientation))
                        .is_ok()
                    {
                        self.pending_thumbnails.insert(photo.id);
                    }
                }
                let texture = self.thumbnails.get(&photo.id).and_then(Option::as_ref);
                let response = thumbnail_cell(
                    ui,
                    photo,
                    texture,
                    frame,
                    caption_height,
                    self.selected == Some(index),
                    self.edited.contains(&photo.id),
                );
                if response.clicked() {
                    response.surrender_focus();
                    next_position = Some(position);
                }
            }
        });
        if let Some(position) = next_position {
            action = self.select(position);
        }
        action
    }
}

/// A compact one-line notice with a coloured marker.
fn notice(ui: &mut egui::Ui, color: egui::Color32, text: &str) {
    ui.horizontal_top(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.label(RichText::new("●").size(8.0).color(color));
        ui.add(egui::Label::new(RichText::new(text).small().color(theme::TEXT_CAPTION)).wrap());
    });
}

/// One list row: the thumbnail fitted into a 3:2 frame, its pick flag as a
/// corner badge, rejects dimmed as in Lightroom, and the filename beneath.
fn thumbnail_cell(
    ui: &mut egui::Ui,
    photo: &Photo,
    texture: Option<&egui::TextureHandle>,
    frame_size: egui::Vec2,
    caption_height: f32,
    selected: bool,
    edited: bool,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(
        frame_size + egui::vec2(0.0, caption_height),
        egui::Sense::click(),
    );
    let name = photo.path.file_name().unwrap_or_default().to_string_lossy();
    let mut hover = photo.path.display().to_string();
    if !photo.copy_name.is_empty() {
        hover += &format!("\nVirtual copy: {}", photo.copy_name);
    }
    if edited {
        hover += "\nEdited in Spektrafilm";
    }
    let response = response.on_hover_text(hover);
    if !ui.is_rect_visible(rect) {
        return response;
    }
    let painter = ui.painter();
    let frame = egui::Rect::from_min_size(rect.min, frame_size);
    let fill = if response.hovered() {
        egui::Color32::from_gray(44)
    } else {
        theme::WELL
    };
    painter.rect_filled(frame, 4.0, fill);
    let inner = frame.shrink(4.0);
    match texture {
        Some(texture) => {
            let size = texture.size_vec2();
            let scale = (inner.width() / size.x).min(inner.height() / size.y);
            let image_rect = egui::Rect::from_center_size(inner.center(), size * scale);
            let tint = if photo.flag == -1 {
                egui::Color32::from_white_alpha(90)
            } else {
                egui::Color32::WHITE
            };
            painter.image(
                texture.id(),
                image_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                tint,
            );
        }
        None => {
            painter.rect_filled(
                inner.shrink(inner.width() * 0.2),
                2.0,
                egui::Color32::from_gray(34),
            );
        }
    }
    if selected {
        painter.rect_stroke(
            frame,
            4.0,
            egui::Stroke::new(2.0, theme::ACCENT),
            egui::StrokeKind::Inside,
        );
    }
    let badge = match photo.flag {
        1 => Some(egui::Color32::from_rgb(120, 200, 140)),
        -1 => Some(egui::Color32::from_rgb(230, 110, 100)),
        _ => None,
    };
    if let Some(color) = badge {
        let c = frame.left_top() + egui::vec2(10.0, 10.0);
        painter.circle(
            c,
            4.5,
            color,
            egui::Stroke::new(1.5, egui::Color32::from_black_alpha(160)),
        );
    }
    if edited {
        // Accent dot, matching the inspector's "changed" marker.
        let c = frame.right_bottom() + egui::vec2(-10.0, -10.0);
        painter.circle(
            c,
            4.5,
            theme::ACCENT,
            egui::Stroke::new(1.5, egui::Color32::from_black_alpha(160)),
        );
    }
    if !photo.copy_name.is_empty() {
        painter.text(
            frame.right_top() + egui::vec2(-6.0, 4.0),
            egui::Align2::RIGHT_TOP,
            "VC",
            egui::FontId::proportional(9.0),
            theme::TEXT_CAPTION,
        );
    }
    let caption = egui::Rect::from_min_max(
        egui::pos2(rect.left(), frame.bottom() + 2.0),
        rect.right_bottom(),
    );
    let color = if selected {
        theme::TEXT
    } else {
        theme::TEXT_WEAK
    };
    let mut job = egui::text::LayoutJob::simple_singleline(
        name.into_owned(),
        egui::FontId::proportional(10.5),
        color,
    );
    job.wrap = egui::text::TextWrapping::truncate_at_width(caption.width());
    painter.galley(caption.left_top(), painter.layout_job(job), color);
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    fn photos(count: usize) -> Vec<Photo> {
        (0..count)
            .map(|i| Photo {
                id: i as i64,
                path: PathBuf::from(format!("/roll/DSC_{i:04}.ARW")),
                available: true,
                copy_name: String::new(),
                flag: 0,
                geometry: Ok(spektrafilm_io::lightroom::Geometry {
                    orientation: 1,
                    crop: [0., 0., 1., 1.],
                    angle: 0.,
                }),
                framing_warning: false,
            })
            .collect()
    }

    /// The panel must keep the width the user chose: list rows that overflow
    /// the scroll area (e.g. by ignoring its scrollbar) make egui widen the
    /// panel again on the next frame.
    #[test]
    fn catalog_panel_keeps_a_narrow_width_with_a_scrolling_list() {
        for width in [210.0, 260.0, 333.0, 480.0] {
            let ctx = egui::Context::default();
            crate::theme::apply(&ctx);
            let mut browser = Browser::open(
                PathBuf::from("/missing/catalog.lrcat"),
                &ctx,
                &Scheduler::default(),
            );
            browser.loading = None;
            browser.catalog = Some(Catalog {
                path: PathBuf::new(),
                photos: photos(60),
                schema_version: Some("9999999".into()),
            });
            browser.filter();
            let mut panel_width = 0.0;
            for frame in 0..6 {
                let _ = ctx.run(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(1200.0, 700.0),
                        )),
                        time: Some(frame as f64),
                        ..Default::default()
                    },
                    |ctx| {
                        panel_width = egui::SidePanel::left("catalog")
                            .exact_width(width)
                            .show(ctx, |ui| browser.show(ui, ctx))
                            .response
                            .rect
                            .width();
                    },
                );
            }
            assert!(
                panel_width <= width + 0.5,
                "panel set to {width} grew to {panel_width}"
            );
        }
    }
    #[test]
    fn accepted_filter_and_navigation_use_catalog_flags() {
        let ctx = egui::Context::default();
        let mut browser = Browser::open(
            PathBuf::from("/missing/catalog.lrcat"),
            &ctx,
            &Scheduler::default(),
        );
        browser.catalog = Some(Catalog {
            path: PathBuf::new(),
            schema_version: Some("1504001".into()),
            photos: [1, 0, -1, 1, 1]
                .into_iter()
                .enumerate()
                .map(|(i, flag)| Photo {
                    id: i as i64,
                    path: PathBuf::new(),
                    available: i != 4,
                    copy_name: String::new(),
                    flag,
                    geometry: Ok(spektrafilm_io::lightroom::Geometry {
                        orientation: 1,
                        crop: [0., 0., 1., 1.],
                        angle: 0.,
                    }),
                    framing_warning: false,
                })
                .collect(),
        });
        browser.accepted_only = true;
        browser.filter();
        assert_eq!(browser.visible, [0, 3]);
        assert!(matches!(
            browser.select(1),
            Some(Action::Select(Photo { id: 3, .. }))
        ));
        assert!(browser.select(2).is_none());
        browser.accepted_only = false;
        browser.filter();
        assert_eq!(browser.visible, [0, 1, 2, 3]);
        browser.source_thumbnail(
            &ctx,
            std::path::Path::new(""),
            &crate::PreviewTextureData {
                size: [4, 2],
                rgba: vec![255; 32],
            },
        );
        let [width, height] = browser.thumbnails[&3].as_ref().unwrap().size();
        assert_eq!(width, 2 * height);
        assert!(width <= THUMBNAIL_PX as usize);
    }
}
