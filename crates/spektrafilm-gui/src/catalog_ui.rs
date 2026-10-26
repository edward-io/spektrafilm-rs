//! Read-only Lightroom Classic catalog browser: a thumbnail grid in the left
//! panel that feeds the selected photo, with its crop, into the editor.
use crate::{file_menu, icons, inspector, theme};
use eframe::egui::{self, RichText};
use spektrafilm_gpu::scheduler::{Priority, Scheduler, Task};
use spektrafilm_io::lightroom::{Catalog, Photo, THUMBNAIL_PX};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::PathBuf,
    sync::mpsc,
};

const THUMBNAIL_CACHE_LIMIT: usize = 256;

pub enum Action {
    Select(Photo),
    Export(Vec<Photo>),
    Empty,
}

#[derive(Default)]
struct Selection {
    ids: HashSet<i64>,
    anchor: Option<i64>,
}

impl Selection {
    fn single(&mut self, id: i64) {
        self.ids.clear();
        self.ids.insert(id);
        self.anchor = Some(id);
    }

    fn range(&mut self, id: i64, order: &[i64]) {
        let Some(end) = order.iter().position(|&item| item == id) else {
            return;
        };
        let start = self
            .anchor
            .and_then(|anchor| order.iter().position(|&item| item == anchor))
            .or_else(|| order.iter().position(|item| self.ids.contains(item)))
            .unwrap_or(end);
        self.anchor = Some(order[start]);
        self.ids = order[start.min(end)..=start.max(end)]
            .iter()
            .copied()
            .collect();
    }

    fn retain(&mut self, visible: &HashSet<i64>) {
        self.ids.retain(|id| visible.contains(id));
        if self.anchor.is_some_and(|id| !visible.contains(&id)) {
            self.anchor = None;
        }
    }
}

pub struct Browser {
    scheduler: Scheduler,
    thumbnail_task: Task,
    path: PathBuf,
    catalog: Option<Catalog>,
    loading: Option<mpsc::Receiver<Result<Catalog, String>>>,
    error: Option<String>,
    accepted_only: bool,
    /// Catalog notices are folded into one line until expanded.
    show_notices: bool,
    visible: Vec<usize>,
    pub selected: Option<usize>,
    selection: Selection,
    scroll_to_selected: bool,
    thumbnails: HashMap<i64, Option<egui::TextureHandle>>,
    rendered: HashMap<i64, (String, u32)>,
    preview_status: HashMap<i64, String>,
    viewport: Vec<usize>,
    /// Only read a camera preview after the edited disk cache has missed.
    fallback_allowed: HashSet<i64>,
    thumbnail_order: VecDeque<i64>,
    thumbnail_generation: u64,
    pending_thumbnails: HashSet<i64>,
    thumbnail_tx: mpsc::SyncSender<(u64, i64, PathBuf, u16)>,
    thumbnail_rx: mpsc::Receiver<(u64, i64, Option<image::RgbaImage>)>,
    /// Photos with settings saved in Spektrafilm, marked in the list.
    pub edited: HashSet<i64>,
}

impl Drop for Browser {
    fn drop(&mut self) {
        self.thumbnail_task.cancel();
    }
}

impl Browser {
    pub(super) fn preview_context(&self) -> (PathBuf, u64) {
        (self.path.clone(), self.thumbnail_generation)
    }

    pub(super) fn viewport_photos(&self) -> Vec<Photo> {
        self.catalog
            .as_ref()
            .map(|catalog| {
                self.viewport
                    .iter()
                    .filter_map(|&i| catalog.photos.get(i).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub(super) fn active_photo(&self) -> Option<Photo> {
        self.catalog.as_ref()?.photos.get(self.selected?).cloned()
    }

    pub(super) fn rendered_key(&self, id: i64) -> Option<&str> {
        self.rendered.get(&id).map(|(key, _)| key.as_str())
    }

    pub(super) fn allow_source_fallback(&mut self, id: i64) {
        self.fallback_allowed.insert(id);
    }

    pub(super) fn preview_status(&mut self, id: i64, status: Option<String>) {
        if let Some(status) = status {
            self.preview_status.insert(id, status);
        } else {
            self.preview_status.remove(&id);
        }
    }

    pub(super) fn rendered_thumbnail(
        &mut self,
        ctx: &egui::Context,
        id: i64,
        key: String,
        quality: u32,
        image: &image::RgbaImage,
    ) {
        if self
            .rendered
            .get(&id)
            .is_some_and(|(old, size)| old == &key && *size > quality)
        {
            return;
        }
        let pixels = egui::ColorImage::from_rgba_unmultiplied(
            [image.width() as usize, image.height() as usize],
            image.as_raw(),
        );
        let texture = if let Some(Some(texture)) = self.thumbnails.get_mut(&id) {
            texture.set(pixels, egui::TextureOptions::LINEAR);
            texture.clone()
        } else {
            ctx.load_texture(
                format!("catalog-{id}"),
                pixels,
                egui::TextureOptions::LINEAR,
            )
        };
        self.store_thumbnail(id, Some(texture));
        self.rendered.insert(id, (key, quality));
        self.preview_status.remove(&id);
        ctx.request_repaint();
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    #[cfg(test)]
    pub(super) fn test_catalog(
        path: PathBuf,
        photos: Vec<Photo>,
        selected: usize,
        ctx: &egui::Context,
        scheduler: &Scheduler,
    ) -> Self {
        let mut browser = Self::open(path.clone(), ctx, scheduler);
        browser.loading = None;
        browser.catalog = Some(Catalog {
            path,
            photos,
            schema_version: Some("1504001".into()),
        });
        browser.filter();
        browser.selected = Some(selected);
        if let Some(photo) = browser
            .catalog
            .as_ref()
            .and_then(|c| c.photos.get(selected))
        {
            browser.selection.single(photo.id);
        }
        browser
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
        let id = photo.id;
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
        self.cache_thumbnail(
            id,
            Some(ctx.load_texture(
                format!("catalog-{id}"),
                egui::ColorImage::from_rgba_unmultiplied(
                    [thumbnail.width() as usize, thumbnail.height() as usize],
                    thumbnail.as_raw(),
                ),
                egui::TextureOptions::LINEAR,
            )),
        );
    }

    fn cache_thumbnail(&mut self, id: i64, texture: Option<egui::TextureHandle>) {
        // Late camera previews must never replace an edited preview, even while stale.
        if self.rendered.contains_key(&id) {
            return;
        }
        self.store_thumbnail(id, texture);
    }

    fn store_thumbnail(&mut self, id: i64, texture: Option<egui::TextureHandle>) {
        self.fallback_allowed.remove(&id);
        if texture.is_none() && self.thumbnails.get(&id).is_some_and(Option::is_some) {
            return;
        }
        self.thumbnails.insert(id, texture);
        self.thumbnail_order.retain(|cached| *cached != id);
        self.thumbnail_order.push_back(id);
        if self.thumbnail_order.len() > THUMBNAIL_CACHE_LIMIT {
            let oldest = self.thumbnail_order.pop_front().unwrap();
            self.thumbnails.remove(&oldest);
            self.rendered.remove(&oldest);
            self.preview_status.remove(&oldest);
        }
    }

    pub fn open(path: PathBuf, ctx: &egui::Context, scheduler: &Scheduler) -> Self {
        let (thumbnail_tx, requests) = mpsc::sync_channel::<(u64, i64, PathBuf, u16)>(24);
        let (results, thumbnail_rx) = mpsc::channel();
        let repaint = ctx.clone();
        let thumbnail_task = scheduler.task(Priority::Visible);
        let task = thumbnail_task.clone();
        std::thread::spawn(move || {
            while let Ok((generation, id, path, orientation)) = requests.recv() {
                let Ok(_cpu) = task.cpu() else { break };
                let image = task
                    .run(|| {
                        std::panic::catch_unwind(|| {
                            spektrafilm_io::lightroom::thumbnail(&path, orientation)
                        })
                    })
                    .ok()
                    .and_then(Result::ok);
                if results.send((generation, id, image)).is_err() {
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
            show_notices: false,
            visible: Vec::new(),
            selected: None,
            selection: Selection::default(),
            scroll_to_selected: false,
            thumbnails: HashMap::new(),
            rendered: HashMap::new(),
            preview_status: HashMap::new(),
            viewport: Vec::new(),
            fallback_allowed: HashSet::new(),
            thumbnail_order: VecDeque::new(),
            thumbnail_generation: 0,
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
        let visible = self
            .catalog
            .as_ref()
            .map(|catalog| self.visible.iter().map(|&i| catalog.photos[i].id).collect())
            .unwrap_or_default();
        self.selection.retain(&visible);
    }

    fn select(&mut self, position: usize) -> Option<Action> {
        let index = *self.visible.get(position)?;
        self.selection
            .single(self.catalog.as_ref()?.photos[index].id);
        self.activate(position)
    }

    fn activate(&mut self, position: usize) -> Option<Action> {
        let index = *self.visible.get(position)?;
        self.selected = Some(index);
        self.scroll_to_selected = true;
        Some(Action::Select(self.catalog.as_ref()?.photos[index].clone()))
    }

    pub fn clear_selection(&mut self) {
        self.selected = None;
        self.selection = Selection::default();
    }

    pub(super) fn preload_photos(&self, radius: u8) -> Vec<Photo> {
        let Some(catalog) = &self.catalog else {
            return vec![];
        };
        let Some(position) = self
            .selected
            .and_then(|selected| self.visible.iter().position(|&index| index == selected))
        else {
            return vec![];
        };
        let eligible = |index: &usize| {
            let photo = &catalog.photos[*index];
            photo.available && photo.geometry.is_ok() && spektrafilm_io::is_raw(&photo.path)
        };
        let mut next = self.visible[position + 1..]
            .iter()
            .copied()
            .filter(eligible)
            .take(radius as usize);
        let mut previous = self.visible[..position]
            .iter()
            .rev()
            .copied()
            .filter(eligible)
            .take(radius as usize);
        let mut photos = Vec::new();
        for _ in 0..radius {
            for index in [next.next(), previous.next()].into_iter().flatten() {
                photos.push(catalog.photos[index].clone());
            }
        }
        photos
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        export_ready: bool,
    ) -> Option<Action> {
        let mut action = None;
        self.viewport.clear();
        if let Some(rx) = &self.loading {
            match rx.try_recv() {
                Ok(result) => {
                    self.loading = None;
                    match result {
                        Ok(catalog) => {
                            // A refresh can change paths/orientation and must retry failed reads.
                            self.thumbnail_generation = self.thumbnail_generation.wrapping_add(1);
                            let ids: HashSet<_> = catalog.photos.iter().map(|p| p.id).collect();
                            self.rendered.retain(|id, _| ids.contains(id));
                            self.thumbnails
                                .retain(|id, _| self.rendered.contains_key(id));
                            self.thumbnail_order
                                .retain(|id| self.thumbnails.contains_key(id));
                            self.preview_status.clear();
                            for &id in self.rendered.keys() {
                                self.preview_status
                                    .insert(id, "Checking edited preview…".into());
                            }
                            self.pending_thumbnails.clear();
                            self.fallback_allowed.clear();
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
                            action = if self.selection.ids.is_empty() {
                                self.select(position)
                            } else {
                                self.activate(position)
                            }
                            .or(Some(Action::Empty));
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
        while let Ok((generation, id, image)) = self.thumbnail_rx.try_recv() {
            if generation != self.thumbnail_generation {
                continue;
            }
            self.pending_thumbnails.remove(&id);
            self.cache_thumbnail(
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
        // ── Header: catalog name and refresh ───────────────────────────
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(
                inspector::title(&self.path.file_stem().unwrap_or_default().to_string_lossy())
                    .size(14.0),
            )
            .on_hover_text(self.path.display().to_string());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(self.loading.is_none(), egui::Button::new("⟳").small())
                    .on_hover_text("Re-read the catalog, e.g. after reconnecting a drive")
                    .clicked()
                {
                    self.refresh(ctx);
                }
            });
        });
        let export_ready = export_ready && self.loading.is_none();
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
                    let position = self
                        .catalog
                        .as_ref()
                        .and_then(|c| {
                            self.visible
                                .iter()
                                .position(|&i| self.selection.ids.contains(&c.photos[i].id))
                        })
                        .unwrap_or(0);
                    action = if self.selection.ids.is_empty() {
                        self.select(position)
                    } else {
                        self.activate(position)
                    }
                    .or(Some(Action::Empty));
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
        let warn = ui.visuals().warn_fg_color;
        let mut notices = Vec::new();
        if let Some(warning) = catalog.compatibility_warning() {
            notices.push((warn, warning));
        }
        let unavailable = catalog_stats.map_or(0, |(_, _, u)| u);
        if unavailable > 0 {
            notices.push((
                theme::TEXT_WEAK,
                format!(
                    "{unavailable} photo{} skipped: original not found. Connect the drive, then ⟳.",
                    if unavailable == 1 { "" } else { "s" }
                ),
            ));
        }
        if let Some(photo) = self.selected.and_then(|i| catalog.photos.get(i))
            && photo.framing_warning
        {
            notices.push((
                warn,
                "Lightroom lens and perspective corrections aren't imported, so framing may differ."
                    .into(),
            ));
        }
        if !notices.is_empty() {
            let color = if notices.iter().any(|(c, _)| *c == warn) {
                warn
            } else {
                theme::TEXT_WEAK
            };
            let summary = format!(
                "{} notice{}",
                notices.len(),
                if notices.len() == 1 { "" } else { "s" }
            );
            // Phosphor glyphs need their own family; Inter maps those code points too.
            let caret = if self.show_notices {
                icons::CARET_DOWN
            } else {
                icons::CARET_RIGHT
            };
            let response = ui
                .horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    ui.label(RichText::new("●").size(8.0).color(color));
                    ui.label(
                        RichText::new(caret)
                            .font(icons::font(11.0))
                            .color(theme::TEXT_CAPTION),
                    );
                    ui.label(RichText::new(summary).small().color(theme::TEXT_CAPTION));
                })
                .response
                .interact(egui::Sense::click())
                .on_hover_cursor(egui::CursorIcon::PointingHand);
            let response = if self.show_notices {
                response
            } else {
                let all = notices.iter().map(|(_, t)| t.as_str()).collect::<Vec<_>>();
                response.on_hover_text(all.join("\n\n"))
            };
            if response.clicked() {
                self.show_notices = !self.show_notices;
            }
            if self.show_notices {
                for (color, text) in &notices {
                    notice(ui, *color, text);
                }
            }
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
                    ui.label(weak(if self.selection.ids.len() > 1 {
                        format!("{} selected", self.selection.ids.len())
                    } else {
                        match position {
                            Some(p) => format!("{} of {}", p + 1, self.visible.len()),
                            None => format!("{} photos", self.visible.len()),
                        }
                    }));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(weak("Shift-click to select".into()));
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
            self.viewport = range
                .clone()
                .map(|position| self.visible[position])
                .collect();
            for position in range {
                let index = self.visible[position];
                let photo = &catalog.photos[index];
                if self.thumbnails.contains_key(&photo.id) {
                    // Keep on-screen images ahead of older cached thumbnails.
                    self.thumbnail_order.retain(|cached| *cached != photo.id);
                    self.thumbnail_order.push_back(photo.id);
                } else if !self.pending_thumbnails.contains(&photo.id)
                    && self.fallback_allowed.contains(&photo.id)
                {
                    let orientation = photo.geometry.as_ref().map(|g| g.orientation).unwrap_or(1);
                    if self
                        .thumbnail_tx
                        .try_send((
                            self.thumbnail_generation,
                            photo.id,
                            photo.path.clone(),
                            orientation,
                        ))
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
                    self.selection.ids.contains(&photo.id),
                    self.edited.contains(&photo.id),
                    self.preview_status.get(&photo.id).map(String::as_str),
                );
                if response.secondary_clicked() && !self.selection.ids.contains(&photo.id) {
                    self.selection.single(photo.id);
                    next_position = Some(position);
                }
                if let Some(error) = file_menu::show_with(&response, &photo.path, |ui| {
                    let count = self.selection.ids.len();
                    let label = if count == 1 {
                        "Export image…".to_owned()
                    } else {
                        format!("Export {count} images…")
                    };
                    if ui
                        .add_enabled(export_ready && count > 0, egui::Button::new(label))
                        .on_disabled_hover_text(
                            "Finish the crop or image reload, and wait for the current export.",
                        )
                        .clicked()
                    {
                        let photos = self
                            .visible
                            .iter()
                            .map(|&i| &catalog.photos[i])
                            .filter(|p| self.selection.ids.contains(&p.id))
                            .cloned()
                            .collect();
                        action = Some(Action::Export(photos));
                        ui.close_menu();
                    }
                    ui.separator();
                }) {
                    self.error = Some(error);
                }
                if response.clicked() {
                    response.surrender_focus();
                    if ctx.input(|i| i.modifiers.shift) {
                        let order: Vec<_> =
                            self.visible.iter().map(|&i| catalog.photos[i].id).collect();
                        self.selection.range(photo.id, &order);
                    } else {
                        next_position = Some(position);
                    }
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
    active: bool,
    selected: bool,
    edited: bool,
    preview_status: Option<&str>,
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
    if let Some(status) = preview_status {
        hover += &format!("\n{status}");
    }
    let response = response.on_hover_text(hover);
    if !ui.is_rect_visible(rect) {
        return response;
    }
    let painter = ui.painter();
    let frame = egui::Rect::from_min_size(rect.min, frame_size);
    let fill = if selected {
        theme::ACCENT.gamma_multiply(0.35)
    } else if response.hovered() {
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
        painter.rect_filled(inner, 2.0, theme::ACCENT.gamma_multiply(0.12));
    }
    if active || selected {
        painter.rect_stroke(
            frame,
            4.0,
            egui::Stroke::new(
                if active { 3.0 } else { 1.0 },
                if active {
                    theme::ACCENT
                } else {
                    theme::TEXT_WEAK
                },
            ),
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
    if preview_status.is_some() {
        painter.circle_stroke(
            frame.left_bottom() + egui::vec2(10., -10.),
            4.,
            egui::Stroke::new(1.5, theme::ACCENT),
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

    #[test]
    fn ranges_keep_the_anchor_and_follow_filtered_order() {
        let mut selection = Selection::default();
        let order = [90, 7, 23, 4, 51];
        selection.single(7);
        selection.range(51, &order);
        assert_eq!(selection.ids, HashSet::from([7, 23, 4, 51]));
        selection.range(23, &order);
        assert_eq!(selection.ids, HashSet::from([7, 23]));
        selection.range(90, &order);
        assert_eq!(selection.ids, HashSet::from([90, 7]));
        selection.retain(&HashSet::from([90, 23, 51]));
        assert_eq!(selection.ids, HashSet::from([90]));
        selection.range(51, &[90, 23, 51]);
        assert_eq!(selection.ids, HashSet::from([90, 23, 51]));
        selection.retain(&HashSet::new());
        selection.range(23, &order);
        assert_eq!(selection.ids, HashSet::from([23]));
    }

    #[test]
    fn shift_click_and_context_menu_export_a_range_without_opening_its_raws() {
        let ctx = egui::Context::default();
        crate::theme::apply(&ctx);
        let mut browser = Browser::test_catalog(
            "/test.lrcat".into(),
            photos(6),
            0,
            &ctx,
            &Scheduler::default(),
        );
        let frame = |browser: &mut Browser, events, modifiers| {
            let mut action = None;
            let output = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(270., 1800.),
                    )),
                    events,
                    modifiers,
                    ..Default::default()
                },
                |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        action = browser.show(ui, ctx, true);
                    });
                },
            );
            (output, action)
        };
        let click = |position, button, modifiers| {
            vec![
                egui::Event::PointerMoved(position),
                egui::Event::PointerButton {
                    pos: position,
                    button,
                    pressed: true,
                    modifiers,
                },
                egui::Event::PointerButton {
                    pos: position,
                    button,
                    pressed: false,
                    modifiers,
                },
            ]
        };
        let (output, _) = frame(&mut browser, vec![], egui::Modifiers::NONE);
        let caption = |output: &egui::FullOutput, name: &str| {
            output
                .shapes
                .iter()
                .find_map(|shape| match &shape.shape {
                    egui::epaint::Shape::Text(text) if text.galley.job.text == name => {
                        Some(text.pos + text.galley.rect.center().to_vec2())
                    }
                    _ => None,
                })
                .unwrap_or_else(|| panic!("missing {name}"))
        };
        let third = caption(&output, "DSC_0003.ARW");
        let (_, action) = frame(
            &mut browser,
            click(third, egui::PointerButton::Primary, egui::Modifiers::SHIFT),
            egui::Modifiers::SHIFT,
        );
        assert!(action.is_none(), "range selection must not open a RAW");
        assert_eq!(browser.selected, Some(0));
        assert_eq!(browser.selection.ids, HashSet::from([0, 1, 2, 3]));
        frame(
            &mut browser,
            click(third, egui::PointerButton::Secondary, egui::Modifiers::NONE),
            egui::Modifiers::NONE,
        );
        let (menu, _) = frame(&mut browser, vec![], egui::Modifiers::NONE);
        assert_eq!(
            browser.selection.ids.len(),
            4,
            "right-click discarded the range"
        );
        let export = caption(&menu, "Export 4 images…");
        let (_, action) = frame(
            &mut browser,
            click(export, egui::PointerButton::Primary, egui::Modifiers::NONE),
            egui::Modifiers::NONE,
        );
        let Some(Action::Export(photos)) = action else {
            panic!("missing export action")
        };
        assert_eq!(
            photos.iter().map(|p| p.id).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(browser.selected, Some(0));
        let (output, _) = frame(&mut browser, vec![], egui::Modifiers::NONE);
        let fifth = caption(&output, "DSC_0005.ARW");
        let (_, action) = frame(
            &mut browser,
            click(fifth, egui::PointerButton::Secondary, egui::Modifiers::NONE),
            egui::Modifiers::NONE,
        );
        assert!(matches!(action, Some(Action::Select(photo)) if photo.id == 5));
        assert_eq!(browser.selection.ids, HashSet::from([5]));
    }

    #[test]
    fn edited_previews_keep_their_texture_and_cannot_be_replaced_by_camera_results() {
        let ctx = egui::Context::default();
        let mut browser = Browser::test_catalog(
            "/test.lrcat".into(),
            photos(THUMBNAIL_CACHE_LIMIT + 1),
            0,
            &ctx,
            &Scheduler::default(),
        );
        let original = image::RgbaImage::new(16, 10);
        browser.rendered_thumbnail(&ctx, 0, "original".into(), 2048, &original);
        let texture = browser.thumbnails[&0].as_ref().unwrap().id();
        browser.preview_status(0, Some("Updating edited preview…".into()));
        browser.cache_thumbnail(0, None);
        browser.cache_thumbnail(
            0,
            Some(ctx.load_texture(
                "late camera result",
                egui::ColorImage::new([2, 2], egui::Color32::WHITE),
                egui::TextureOptions::LINEAR,
            )),
        );
        assert_eq!(browser.thumbnails[&0].as_ref().unwrap().id(), texture);
        assert!(browser.preview_status.contains_key(&0));
        let cropped = image::RgbaImage::new(8, 10);
        browser.rendered_thumbnail(&ctx, 0, "cropped".into(), 2048, &cropped);
        browser.rendered_thumbnail(&ctx, 0, "cropped".into(), 640, &original);
        let current = browser.thumbnails[&0].as_ref().unwrap();
        assert_eq!(current.id(), texture);
        assert_eq!(current.size(), [8, 10]);
        assert!(!browser.preview_status.contains_key(&0));
        for id in 1..=THUMBNAIL_CACHE_LIMIT as i64 {
            browser.rendered_thumbnail(&ctx, id, "current".into(), 2048, &original);
        }
        assert_eq!(browser.rendered.len(), THUMBNAIL_CACHE_LIMIT);
        assert_eq!(browser.thumbnails.len(), THUMBNAIL_CACHE_LIMIT);
        assert!(!browser.rendered.contains_key(&0));
    }

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

    #[test]
    fn visible_thumbnails_survive_cache_turnover_without_reloading() {
        let ctx = egui::Context::default();
        crate::theme::apply(&ctx);
        let mut browser = Browser::test_catalog(
            "/test.lrcat".into(),
            photos(THUMBNAIL_CACHE_LIMIT + 32),
            0,
            &ctx,
            &Scheduler::default(),
        );
        let (requests_tx, requests_rx) = mpsc::sync_channel(24);
        let (results_tx, results_rx) = mpsc::channel();
        browser.thumbnail_tx = requests_tx;
        browser.thumbnail_rx = results_rx;
        let preview = crate::PreviewTextureData {
            size: [4, 2],
            rgba: vec![255; 32],
        };
        for index in 0..THUMBNAIL_CACHE_LIMIT {
            browser.selected = Some(index);
            let path = browser.catalog.as_ref().unwrap().photos[index].path.clone();
            browser.source_thumbnail(&ctx, &path, &preview);
        }
        // Put the old policy's next victim on screen, regardless of hash order.
        let visible_id = *browser.thumbnails.keys().next().unwrap();
        browser
            .catalog
            .as_mut()
            .unwrap()
            .photos
            .swap(0, visible_id as usize);
        browser.selected = None;
        let texture_id = browser.thumbnails[&visible_id].as_ref().unwrap().id();
        for frame in 0..33 {
            if frame > 0 {
                results_tx
                    .send((
                        browser.thumbnail_generation,
                        (THUMBNAIL_CACHE_LIMIT + frame - 1) as i64,
                        Some(image::RgbaImage::new(4, 2)),
                    ))
                    .unwrap();
            }
            let width = if frame < 16 {
                260.
            } else {
                [210., 480., 333.][frame % 3]
            };
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1200., 700.),
                    )),
                    time: Some(frame as f64 / 60.),
                    ..Default::default()
                },
                |ctx| {
                    egui::SidePanel::left("catalog")
                        .exact_width(width)
                        .show(ctx, |ui| browser.show(ui, ctx, true));
                },
            );
            assert_eq!(
                browser
                    .thumbnails
                    .get(&visible_id)
                    .and_then(Option::as_ref)
                    .map(egui::TextureHandle::id),
                Some(texture_id),
                "visible thumbnail disappeared on frame {frame}"
            );
            assert!(
                requests_rx.try_iter().all(|(_, id, _, _)| id != visible_id),
                "visible thumbnail was requeued on frame {frame}"
            );
        }
        assert_eq!(browser.thumbnails.len(), THUMBNAIL_CACHE_LIMIT);
    }

    #[test]
    fn refresh_invalidates_thumbnails_and_ignores_late_results_from_the_old_catalog() {
        let ctx = egui::Context::default();
        crate::theme::apply(&ctx);
        let mut browser = Browser::test_catalog(
            "/test.lrcat".into(),
            photos(2),
            0,
            &ctx,
            &Scheduler::default(),
        );
        let (requests_tx, requests_rx) = mpsc::sync_channel(24);
        let (results_tx, results_rx) = mpsc::channel();
        browser.thumbnail_tx = requests_tx;
        browser.thumbnail_rx = results_rx;
        browser.cache_thumbnail(
            0,
            Some(ctx.load_texture(
                "old",
                egui::ColorImage::new([1, 1], egui::Color32::WHITE),
                egui::TextureOptions::LINEAR,
            )),
        );
        browser.cache_thumbnail(1, None);
        browser.selection.range(1, &[0, 1]);
        browser.pending_thumbnails.insert(0);
        let generation = browser.thumbnail_generation;
        let (tx, rx) = mpsc::channel();
        browser.loading = Some(rx);
        let mut updated = photos(2);
        updated[0].geometry.as_mut().unwrap().orientation = 6;
        updated[0].path = "/moved/photo.ARW".into();
        updated.swap(0, 1);
        tx.send(Ok(Catalog {
            path: browser.path.clone(),
            photos: updated,
            schema_version: None,
        }))
        .unwrap();
        results_tx
            .send((generation, 0, Some(image::RgbaImage::new(1, 1))))
            .unwrap();
        results_tx.send((generation, 1, None)).unwrap();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| browser.show(ui, ctx, true));
        });
        assert_eq!(browser.thumbnail_generation, generation + 1);
        assert_eq!(browser.selected, Some(1));
        assert_eq!(browser.selection.ids, HashSet::from([0, 1]));
        assert_eq!(browser.selection.anchor, Some(0));
        assert!(browser.thumbnails.is_empty());
        assert!(browser.thumbnail_order.is_empty());
        // The edited disk cache gets the first chance after a refresh.
        assert!(requests_rx.try_recv().is_err());
        browser.allow_source_fallback(0);
        browser.allow_source_fallback(1);
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| browser.show(ui, ctx, true));
        });
        let requests: Vec<_> = requests_rx.try_iter().collect();
        assert!(!requests.is_empty());
        assert!(requests.iter().all(|(g, _, _, _)| *g == generation + 1));
        assert!(requests.iter().any(|(_, id, path, orientation)| *id == 0
            && *path == PathBuf::from("/moved/photo.ARW")
            && *orientation == 6));
    }

    #[test]
    fn thumbnail_cache_updates_preserve_other_entries_and_failed_loads_preserve_good_images() {
        let ctx = egui::Context::default();
        let mut browser = Browser::test_catalog(
            "/test.lrcat".into(),
            photos(1),
            0,
            &ctx,
            &Scheduler::default(),
        );
        let texture = ctx.load_texture(
            "thumbnail",
            egui::ColorImage::new([1, 1], egui::Color32::WHITE),
            egui::TextureOptions::LINEAR,
        );
        for id in 0..THUMBNAIL_CACHE_LIMIT as i64 {
            browser.cache_thumbnail(id, Some(texture.clone()));
        }
        let last = THUMBNAIL_CACHE_LIMIT as i64 - 1;
        browser.cache_thumbnail(last, Some(texture.clone()));
        browser.cache_thumbnail(last, None);
        assert_eq!(browser.thumbnails.len(), THUMBNAIL_CACHE_LIMIT);
        assert!(browser.thumbnails.contains_key(&0));
        assert_eq!(
            browser.thumbnails[&last].as_ref().unwrap().id(),
            texture.id()
        );

        browser.cache_thumbnail(last + 1, None);
        assert_eq!(browser.thumbnails.len(), THUMBNAIL_CACHE_LIMIT);
        assert_eq!(browser.thumbnail_order.len(), THUMBNAIL_CACHE_LIMIT);
        assert!(!browser.thumbnails.contains_key(&0));
        assert!(browser.thumbnails[&(last + 1)].is_none());
    }

    #[test]
    fn preload_order_follows_the_filter_and_skips_non_raw_and_missing_files() {
        let ctx = egui::Context::default();
        let mut photos = photos(9);
        photos[5].path.set_extension("jpg");
        photos[7].available = false;
        photos[1].flag = -1;
        for i in [0, 2, 4, 6, 8] {
            photos[i].flag = 1;
        }
        let mut browser =
            Browser::test_catalog("/test.lrcat".into(), photos, 4, &ctx, &Scheduler::default());
        let ids = |photos: Vec<Photo>| photos.into_iter().map(|p| p.id).collect::<Vec<_>>();
        assert_eq!(ids(browser.preload_photos(2)), [6, 3, 8, 2]);
        browser.accepted_only = true;
        browser.filter();
        assert_eq!(ids(browser.preload_photos(2)), [6, 2, 8, 0]);
        browser.selected = Some(0);
        assert_eq!(ids(browser.preload_photos(2)), [2, 4]);
        assert!(browser.preload_photos(0).is_empty());
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
                            .show(ctx, |ui| browser.show(ui, ctx, true))
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
