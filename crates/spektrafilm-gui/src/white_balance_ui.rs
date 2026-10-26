use super::*;

impl App {
    pub(super) fn reset_white_balance(&mut self) {
        self.raw_options.white_balance = "as_shot".into();
        self.raw_options.temperature = 5500.;
        self.raw_options.tint = 1.;
        self.raw_options.white_balance_shift = [0.; 2];
        self.pending_label = Some("White balance: As Shot".into());
    }

    pub(super) fn white_balance_controls(&mut self, ui: &mut egui::Ui) {
        use inspector::{Section, Tab};
        let raw = self
            .image_path
            .as_deref()
            .is_some_and(spektrafilm_io::is_raw);
        Section::new("White balance", Tab::Input, tooltips::SECTION_WHITE_BALANCE).show(ui, |ui| {
            let base = self.metadata.white_balance;
            let ready = raw
                && base.is_some()
                && self.loaded_raw_options.as_ref().is_some_and(|o| {
                    o.decoder == self.raw_options.decoder
                        && o.white_balance == self.raw_options.white_balance
                        && o.temperature == self.raw_options.temperature
                        && o.tint == self.raw_options.tint
                });
            let mut reset = false;
            ui.add_enabled_ui(ready, |ui| {
                let base = base.unwrap_or(spektrafilm_io::WhiteBalance {
                    temperature: 5500.,
                    tint: 0.,
                });
                let mut value = if self.raw_options.white_balance_shift == [0.; 2] {
                    base
                } else {
                    base.shifted(self.raw_options.white_balance_shift)
                };
                let mut changed = tooltips::labeled_with_reset(
                    ui,
                    |ui: &mut egui::Ui| {
                        colored_slider(
                            ui,
                            egui::Slider::new(&mut value.temperature, 1667.0..=25000.0)
                                .logarithmic(true)
                                .clamping(egui::SliderClamping::Never)
                                .max_decimals(0),
                            [
                                egui::Color32::from_rgb(55, 75, 98),
                                egui::Color32::from_gray(56),
                                egui::Color32::from_rgb(107, 83, 56),
                            ],
                        )
                    },
                    "Temperature (K)",
                    tooltips::CONTROL_TEMPERATURE_K,
                    "Reset to As Shot",
                )
                .changed();
                reset |= inspector::take_row_reset();
                changed |= tooltips::labeled_with_reset(
                    ui,
                    |ui: &mut egui::Ui| {
                        colored_slider(
                            ui,
                            egui::Slider::new(
                                &mut value.tint,
                                (base.tint - 150.)..=(base.tint + 150.),
                            )
                            .clamping(egui::SliderClamping::Never)
                            .max_decimals(1),
                            [
                                egui::Color32::from_rgb(56, 88, 64),
                                egui::Color32::from_gray(56),
                                egui::Color32::from_rgb(98, 61, 88),
                            ],
                        )
                    },
                    "Tint",
                    tooltips::CONTROL_TINT,
                    "Reset to As Shot",
                )
                .changed();
                reset |= inspector::take_row_reset();
                if changed && !reset {
                    value.temperature = value.temperature.clamp(1667., 25000.);
                    value.tint = value.tint.clamp(base.tint - 150., base.tint + 150.);
                    self.raw_options.white_balance_shift = base.shift_to(value);
                }
            });
            if reset {
                self.reset_white_balance();
            }
            if !raw {
                ui.weak("White balance is available for RAW photos.");
            } else if base.is_none() {
                ui.weak("Camera white balance data is unavailable.");
            }
        });
    }
}

/// Keep the native slider's interaction and handle, with a subtly colored rounded rail.
fn colored_slider(
    ui: &mut egui::Ui,
    slider: egui::Slider<'_>,
    colors: [egui::Color32; 3],
) -> egui::Response {
    let width = ui.spacing().slider_width;
    let height = ui.spacing().slider_rail_height;
    let slot = ui.painter().add(egui::Shape::Noop);
    let response = ui
        .scope(|ui| {
            ui.spacing_mut().slider_rail_height = 0.;
            ui.add(slider.trailing_fill(false))
        })
        .inner;
    if ui.is_rect_visible(response.rect) {
        let id = egui::Id::new(("white-balance-rail-gradient", colors));
        let texture = ui
            .ctx()
            .data_mut(|data| data.get_temp::<egui::TextureHandle>(id))
            .unwrap_or_else(|| {
                let texture = ui.ctx().load_texture(
                    "white balance rail",
                    egui::ColorImage {
                        size: [3, 1],
                        pixels: colors.to_vec(),
                    },
                    egui::TextureOptions::LINEAR,
                );
                ui.ctx()
                    .data_mut(|data| data.insert_temp(id, texture.clone()));
                texture
            });
        let rail = egui::Rect::from_center_size(
            egui::pos2(response.rect.left() + width / 2., response.rect.center().y),
            egui::vec2(width, height),
        );
        ui.painter().set(
            slot,
            egui::epaint::RectShape::filled(
                rail,
                ui.visuals().widgets.inactive.corner_radius,
                egui::Color32::WHITE,
            )
            .with_texture(
                texture.id(),
                egui::Rect::from_min_max(egui::pos2(1. / 6., 0.), egui::pos2(5. / 6., 1.)),
            ),
        );
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> App {
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx);
        let mut app = App::new(
            &cc,
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            None,
        );
        app.raw_options.decoder = "rawler".into();
        let input = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../spektrafilm-io/tests/fixtures/synthetic.dng");
        app.load_image_from_path(&input);
        finish(&mut app);
        app
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

    #[test]
    fn white_balance_reset_and_history_restore_as_shot_without_clearing_preview() {
        let mut app = app();
        let original = app.image.clone().unwrap();
        let input_tex = app.input_tex.clone().unwrap();
        app.output_tex = Some(input_tex.clone());
        app.output_image = Some((*original).clone());
        let previous = app.raw_options.clone();
        app.raw_options.white_balance_shift = [-20., 12.];
        app.record_step();
        app.reload_raw_if_changed(&previous);
        assert!(!app.save_preview_enabled());
        finish(&mut app);
        assert_eq!(app.output_tex.as_ref().unwrap().id(), input_tex.id());
        assert!(
            app.output_image.is_none(),
            "old pixels must not be saveable"
        );
        assert!(!app.develop_next);
        assert_ne!(app.image.as_ref().unwrap().data, original.data);
        assert_eq!(app.export_raw_options().white_balance_shift, [-20., 12.]);
        app.undo();
        finish(&mut app);
        assert_eq!(app.raw_options.white_balance_shift, [0.; 2]);
        assert_eq!(app.image.as_ref().unwrap().data, original.data);
        app.redo();
        finish(&mut app);
        assert_eq!(app.raw_options.white_balance_shift, [-20., 12.]);
        let previous = app.raw_options.clone();
        app.reset_white_balance();
        app.record_step();
        app.reload_raw_if_changed(&previous);
        finish(&mut app);
        assert_eq!(app.image.as_ref().unwrap().data, original.data);
        app.undo();
        finish(&mut app);
        assert_eq!(app.raw_options.white_balance_shift, [-20., 12.]);
    }

    #[test]
    fn white_balance_sliders_are_visible_for_as_shot_and_custom_without_dropdown() {
        let mut app = app();
        let ctx = app.ctx.clone();
        for mode in ["as_shot", "custom"] {
            app.raw_options.white_balance = mode.into();
            app.loaded_raw_options = Some(app.raw_options.clone());
            app.begin_frame_state(&ctx);
            let output = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(480., 600.),
                    )),
                    ..Default::default()
                },
                |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| app.white_balance_controls(ui));
                },
            );
            let texts: Vec<_> = output
                .shapes
                .iter()
                .filter_map(|s| match &s.shape {
                    egui::Shape::Text(text) => Some(text.galley.text()),
                    _ => None,
                })
                .collect();
            assert!(texts.contains(&"TemperatureK"), "{texts:?}");
            assert!(texts.contains(&"Tint"), "{texts:?}");
            assert!(
                !texts.contains(&"Reset to As Shot")
                    && !texts.contains(&"As Shot")
                    && !texts.contains(&"Custom"),
                "{texts:?}"
            );
            let rails: Vec<_> = output
                .shapes
                .iter()
                .filter_map(|s| match &s.shape {
                    egui::Shape::Rect(rect) if rect.brush.is_some() => Some(rect),
                    _ => None,
                })
                .collect();
            assert_eq!(
                rails.len(),
                2,
                "Both white balance sliders should be colored"
            );
            for rail in &rails {
                assert_eq!(
                    rail.corner_radius,
                    ctx.style().visuals.widgets.inactive.corner_radius
                );
                assert_eq!(rail.rect.height(), ctx.style().spacing.slider_rail_height);
            }
            assert_ne!(
                rails[0].fill_texture_id(),
                rails[1].fill_texture_id(),
                "Temperature and Tint need distinct gradients"
            );
            assert!(!texts.contains(&"Daylight") && !texts.contains(&"Tungsten"));
        }
    }
    #[test]
    fn double_clicking_white_balance_caption_resets_to_as_shot() {
        let mut app = app();
        app.raw_options.white_balance_shift = [-20., 10.];
        let ctx = app.ctx.clone();
        let frame = |app: &mut App, time, events| {
            app.begin_frame_state(&ctx);
            ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(480., 600.),
                    )),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| app.white_balance_controls(ui));
                },
            )
        };
        let output = frame(&mut app, 0., vec![]);
        let position = output
            .shapes
            .iter()
            .find_map(|s| match &s.shape {
                egui::Shape::Text(text) if text.galley.text() == "TemperatureK" => {
                    Some(text.pos + text.galley.rect.center().to_vec2())
                }
                _ => None,
            })
            .unwrap();
        for (index, pressed) in [true, false, true, false].into_iter().enumerate() {
            frame(
                &mut app,
                0.1 + index as f64 * 0.05,
                vec![
                    egui::Event::PointerMoved(position),
                    egui::Event::PointerButton {
                        pos: position,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::default(),
                    },
                ],
            );
        }
        assert_eq!(app.raw_options.white_balance, "as_shot");
        assert_eq!(app.raw_options.white_balance_shift, [0.; 2]);
    }
}
