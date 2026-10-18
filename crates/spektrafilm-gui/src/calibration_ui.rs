//! Image calibration runs off the UI thread and applies only to its input snapshot.
use super::*;
#[derive(Clone, Copy)]
enum Action {
    Base,
    Device,
    Printer,
}
enum Output {
    Base(spektrafilm_core::params::BaseDensityParams, f64),
    Device(String),
    Printer([f32; 2]),
}
pub(super) struct Job {
    key: String,
    rx: mpsc::Receiver<Result<Output, String>>,
}
impl App {
    fn calibration_key(&self) -> String {
        serde_json::json!([
            self.film_name,
            self.print_name,
            self.params,
            self.image.as_ref().map(|i| Arc::as_ptr(i) as usize)
        ])
        .to_string()
    }
    /// Scanned-negative calibrations, shown with the scan conversion controls.
    pub(super) fn calibration_controls(&mut self, ui: &mut egui::Ui) {
        self.calibration_buttons(ui, &[
            ("Detect base",Action::Base,"Fit the film base and scan exposure from the brightest clear-film pixels."),
            ("Blind calibration",Action::Device,"Fit a device matrix using the film dye gamut; a colorful frame gives the strongest fit."),
        ]);
    }
    /// Enlarger filtration solve, shown with the enlarger controls.
    pub(super) fn print_balance_controls(&mut self, ui: &mut egui::Ui) {
        self.calibration_buttons(
            ui,
            &[(
                "Neutral print balance",
                Action::Printer,
                "Solve enlarger magenta and yellow shifts so this film's midgray prints neutral.",
            )],
        );
    }
    fn calibration_buttons(&mut self, ui: &mut egui::Ui, actions: &[(&str, Action, &str)]) {
        ui.horizontal_wrapped(|ui| {
            for &(label, action, help) in actions {
                let enabled = self.calibration_job.is_none()
                    && (matches!(action, Action::Printer) || self.image.is_some());
                if ui
                    .add_enabled(enabled, egui::Button::new(label))
                    .on_hover_text(help)
                    .clicked()
                {
                    self.start_calibration(action);
                }
            }
        });
        if self.calibration_job.is_some() {
            ui.label("Calibrating…");
        }
    }
    fn start_calibration(&mut self, action: Action) {
        let key = self.calibration_key();
        let image = self.image.clone();
        let p = self.params.clone();
        let data = self.data_dir.clone();
        let film = self.film_name.clone();
        let paper = self.print_name.clone();
        let (tx, rx) = mpsc::channel();
        let task = self.scheduler.task(Priority::Export);
        std::thread::Builder::new()
            .name("spektrafilm-calibration".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || -> Result<Output, String> {
                        let _scope = task.enter();
                        let _cpu = task.cpu().map_err(|e| e.to_string())?;
                        let film = profile::load_profile_by_name(&data, &film)
                            .map_err(|e| e.to_string())?;
                        match action {
                            Action::Base => {
                                let (base, ev) = spektrafilm_core::calibration::detect_base(
                                    image.as_deref().ok_or("load an image first")?,
                                    film,
                                    &p,
                                )?;
                                Ok(Output::Base(base, ev))
                            }
                            Action::Device => Ok(Output::Device(
                                spektrafilm_core::calibration::blind_calibration(
                                    image.as_deref().ok_or("load an image first")?,
                                    film,
                                    &p,
                                )?,
                            )),
                            Action::Printer => {
                                let paper = profile::load_profile_by_name(&data, &paper)
                                    .map_err(|e| e.to_string())?;
                                Ok(Output::Printer(
                                    spektrafilm_core::calibration::neutral_print_balance(
                                        film, paper, &p, &data,
                                    )?,
                                ))
                            }
                        }
                    },
                ))
                .unwrap_or_else(|panic| Err(panic_message(panic)));
                let _ = tx.send(result);
            })
            .map_err(|e| self.status = format!("Calibration could not start: {e}"))
            .ok()
            .map(|_| {
                self.calibration_job = Some(Job { key, rx });
                self.status = "Calibrating…".into();
            });
    }
    pub(super) fn poll_calibration(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.calibration_job else {
            return;
        };
        let result = match job.rx.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => {
                ctx.request_repaint_after(Duration::from_millis(50));
                return;
            }
            Err(mpsc::TryRecvError::Disconnected) => Err("calibration worker stopped".into()),
        };
        let job = self.calibration_job.take().unwrap();
        if job.key != self.calibration_key() {
            self.status = "Calibration discarded because its inputs changed.".into();
            return;
        }
        match result {
            Ok(Output::Base(base, ev)) => {
                self.params.film_render.base = base;
                self.params.film_render.convert.exposure_compensation_ev = ev;
                self.status = format!("Film base fitted; scan exposure {ev:+.2} EV");
                self.dirty = true;
            }
            Ok(Output::Device(matrix)) => {
                self.params.film_render.convert.calibration = matrix;
                self.status = "Device calibration fitted".into();
                self.dirty = true;
            }
            Ok(Output::Printer([m, y])) => {
                self.params.enlarger.m_filter_shift = m;
                self.params.enlarger.y_filter_shift = y;
                self.status = format!("Neutral print balance: M {m:+.2}, Y {y:+.2}");
                self.dirty = true;
            }
            Err(error) => self.status = format!("Calibration failed: {error}"),
        }
    }
}
