use eframe::egui;
use spektrafilm_io::lightroom::{Catalog, Photo};
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
}

impl Browser {
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
        let scale = (96. / preview.size[0].max(preview.size[1]) as f32).min(1.);
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

    pub fn open(path: PathBuf, ctx: &egui::Context) -> Self {
        let (thumbnail_tx, requests) = mpsc::sync_channel::<(i64, PathBuf, u16)>(24);
        let (results, thumbnail_rx) = mpsc::channel();
        let repaint = ctx.clone();
        std::thread::spawn(move || {
            while let Ok((id, path, orientation)) = requests.recv() {
                let image = std::panic::catch_unwind(|| {
                    spektrafilm_io::lightroom::thumbnail(&path, orientation)
                })
                .ok()
                .and_then(Result::ok);
                if results.send((id, image)).is_err() {
                    break;
                }
                repaint.request_repaint();
            }
        });
        let mut browser = Self {
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
        std::thread::spawn(move || {
            let result = Catalog::open(&path).map_err(|e| format!("{e:#}"));
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
        ui.heading("Lightroom catalog");
        ui.label(self.path.file_stem().unwrap_or_default().to_string_lossy())
            .on_hover_text(self.path.display().to_string());
        ui.horizontal(|ui| {
            if ui
                .add_enabled(self.loading.is_none(), egui::Button::new("Refresh"))
                .clicked()
            {
                self.refresh(ctx);
            }
            if ui.button("Close").clicked() {
                action = Some(Action::Close);
            }
        });
        if self.loading.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Reading catalog…");
            });
        }
        if let Some(error) = &self.error {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
        let response = ui.checkbox(&mut self.accepted_only, "Accepted only");
        if response.changed() {
            response.surrender_focus();
            self.filter();
            if !self.selected.is_some_and(|i| self.visible.contains(&i)) {
                self.selected = None;
                action = self.select(0).or(Some(Action::Empty));
            }
        }
        let Some(catalog) = &self.catalog else {
            return action;
        };
        let accepted = catalog.photos.iter().filter(|p| p.flag == 1).count();
        let rejected = catalog.photos.iter().filter(|p| p.flag == -1).count();
        let unavailable = catalog.photos.iter().filter(|p| !p.available).count();
        ui.small(format!(
            "{} photos · {accepted} accepted · {rejected} rejected",
            catalog.photos.len()
        ));
        if unavailable > 0 {
            ui.small(format!("{unavailable} unavailable originals skipped"));
        }
        ui.small("Read-only · RAW development in Spektrafilm");
        let position = self
            .selected
            .and_then(|i| self.visible.iter().position(|v| *v == i));
        let mut next_position = None;
        ui.horizontal(|ui| {
            let previous = ui.add_enabled(
                position.is_some_and(|p| p > 0),
                egui::Button::new("Previous"),
            );
            if previous.clicked() {
                previous.surrender_focus();
                next_position = position.map(|p| p - 1);
            }
            let next = ui.add_enabled(
                position.is_some_and(|p| p + 1 < self.visible.len()),
                egui::Button::new("Next"),
            );
            if next.clicked() {
                next.surrender_focus();
                next_position = position.map(|p| p + 1);
            }
        });
        if !ctx.wants_keyboard_input() {
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft)) {
                next_position = position.and_then(|p| p.checked_sub(1));
            }
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight)) {
                next_position = position.map(|p| p + 1).filter(|p| *p < self.visible.len());
            }
        }
        ui.small(format!(
            "{} shown · Left / Right keys to browse",
            self.visible.len()
        ));
        ui.separator();
        if self.visible.is_empty() {
            ui.label(if unavailable == catalog.photos.len() && unavailable > 0 {
                "Originals are unavailable. Connect the photo drive, then Refresh."
            } else {
                "No available photos match this filter."
            });
        }
        if let Some(index) = self.selected {
            let photo = &catalog.photos[index];
            if photo.framing_warning {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    "Adobe lens/perspective corrections are not imported; framing may differ.",
                );
            }
        }
        let row_height = 64.;
        let mut scroll = egui::ScrollArea::vertical().id_salt("catalog-photos");
        if self.scroll_to_selected {
            if let Some(position) = position {
                scroll = scroll.vertical_scroll_offset(
                    position as f32 * (row_height + ui.spacing().item_spacing.y),
                );
            }
            self.scroll_to_selected = false;
        }
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
                let (flag, color) = match photo.flag {
                    1 => ("Accepted", egui::Color32::from_rgb(160, 232, 176)),
                    -1 => ("Rejected", egui::Color32::from_rgb(255, 180, 170)),
                    _ => ("Unflagged", ui.visuals().text_color()),
                };
                let name = photo.path.file_name().unwrap_or_default().to_string_lossy();
                let copy = if photo.copy_name.is_empty() {
                    String::new()
                } else {
                    format!(" · {}", photo.copy_name)
                };
                let text = egui::RichText::new(format!("{flag}\n{name}{copy}")).color(color);
                let button = match self.thumbnails.get(&photo.id).and_then(Option::as_ref) {
                    Some(texture) => egui::Button::image_and_text(
                        egui::Image::new((texture.id(), texture.size_vec2()))
                            .fit_to_exact_size(egui::vec2(56., 56.))
                            .maintain_aspect_ratio(true),
                        text,
                    ),
                    None => egui::Button::new(text),
                }
                .selected(self.selected == Some(index));
                let response = ui
                    .add_sized([ui.available_width(), row_height], button)
                    .on_hover_text(photo.path.display().to_string());
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn accepted_filter_and_navigation_use_catalog_flags() {
        let ctx = egui::Context::default();
        let mut browser = Browser::open(PathBuf::from("/missing/catalog.lrcat"), &ctx);
        browser.catalog = Some(Catalog {
            path: PathBuf::new(),
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
        assert!(width <= 96);
    }
}
