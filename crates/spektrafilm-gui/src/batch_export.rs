//! Snapshot a selection, then export one original at a time without driving the editor.
use super::*;
use crate::{edits::PhotoEdit, photo_export::Request};
use spektrafilm_io::lightroom::Photo;
use std::sync::Mutex;

struct Item {
    path: PathBuf,
    label: String,
    request: Result<Request, String>,
}

#[derive(Default)]
struct Progress {
    total: usize,
    done: usize,
    exported: usize,
    skipped: usize,
    failed: usize,
    current: String,
    issues: Vec<(String, String)>,
    fatal: Option<String>,
    cancelled: bool,
}

pub(super) struct Job {
    progress: Arc<Mutex<Progress>>,
    cancel: Arc<AtomicBool>,
    task: Task,
    handle: Option<JoinHandle<()>>,
    folder: PathBuf,
}

impl Job {
    pub fn running(&self) -> bool {
        self.handle
            .as_ref()
            .is_some_and(|handle| !handle.is_finished())
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
        self.task.cancel();
    }

    fn poll(&mut self) {
        if self.handle.as_ref().is_some_and(JoinHandle::is_finished) {
            let _ = self.handle.take().unwrap().join();
        }
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        self.cancel();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn filename(request: &Request) -> String {
    let stem = request
        .path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let name = format!("{stem}_{}_spektra", request.edit.film);
    let mut name: String = name
        .chars()
        .map(|c| {
            if c.is_control() || "/\\<>:\"|?*".contains(c) {
                '_'
            } else {
                c
            }
        })
        .collect();
    let mut length = name.len().min(180);
    while !name.is_char_boundary(length) {
        length -= 1;
    }
    name.truncate(length);
    name.trim_matches([' ', '.']).to_owned()
}

fn publish_unique(
    mut temporary: tempfile::TempPath,
    folder: &Path,
    stem: &str,
    extension: &str,
    cancel: &AtomicBool,
) -> Result<PathBuf> {
    for suffix in 1..=u32::MAX {
        anyhow::ensure!(!cancel.load(Ordering::SeqCst), "cancelled");
        let name = if suffix == 1 {
            format!("{stem}.{extension}")
        } else {
            format!("{stem}_{suffix}.{extension}")
        };
        let output = folder.join(name);
        match temporary.persist_noclobber(&output) {
            Ok(()) => return Ok(output),
            Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
                temporary = error.path
            }
            Err(error) => return Err(spektrafilm_io::OutputIoError(error.error.into()).into()),
        }
    }
    Err(spektrafilm_io::OutputIoError(anyhow::anyhow!("No unused export filename remains")).into())
}

fn run_items(
    items: Vec<Item>,
    folder: &Path,
    extension: &str,
    cancel: &AtomicBool,
    progress: &Mutex<Progress>,
    mut render: impl FnMut(&Request, &Path) -> Result<()>,
) {
    for item in items {
        if cancel.load(Ordering::SeqCst) {
            break;
        }
        progress.lock().unwrap().current = item.label.clone();
        let missing = matches!(std::fs::metadata(&item.path), Err(error) if error.kind() == std::io::ErrorKind::NotFound);
        if missing {
            let mut state = progress.lock().unwrap();
            state.skipped += 1;
            state.done += 1;
            state
                .issues
                .push((item.label, "Original is unavailable; skipped.".into()));
            continue;
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let request = item.request.map_err(anyhow::Error::msg)?;
            let temporary = tempfile::Builder::new()
                .prefix(".spektrafilm-export-")
                .suffix(&format!(".{extension}"))
                .tempfile_in(folder)
                .map_err(|e| spektrafilm_io::OutputIoError(e.into()))?
                .into_temp_path();
            render(&request, &temporary)?;
            anyhow::ensure!(!cancel.load(Ordering::SeqCst), "cancelled");
            publish_unique(temporary, folder, &filename(&request), extension, cancel)
        }))
        .unwrap_or_else(|panic| Err(anyhow::anyhow!(panic_message(panic))));
        let mut state = progress.lock().unwrap();
        match result {
            Ok(_) => {
                state.exported += 1;
                state.done += 1;
            }
            Err(_) if cancel.load(Ordering::SeqCst) => break,
            Err(error) => {
                state.failed += 1;
                state.done += 1;
                state.issues.push((item.label, format!("{error:#}")));
                if error.is::<spektrafilm_io::OutputIoError>() {
                    state.fatal = Some(format!("Export stopped: {error:#}"));
                    break;
                }
            }
        }
    }
    let mut state = progress.lock().unwrap();
    state.cancelled = cancel.load(Ordering::SeqCst) && state.done < state.total;
    state.current.clear();
}

fn snapshot(photo: Photo, edit: Result<PhotoEdit, String>, detect_input_color: bool) -> Item {
    let path = photo.path;
    let mut label = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    if !photo.copy_name.is_empty() {
        label += &format!(" ({})", photo.copy_name);
    }
    let request = edit.and_then(|mut edit| {
        edit.raw.lightroom_geometry = Some(photo.geometry?);
        edit.params.io.crop = false;
        Ok(Request {
            path: path.clone(),
            edit,
            detect_input_color,
        })
    });
    Item {
        path,
        label,
        request,
    }
}

impl App {
    pub(super) fn export_busy(&self) -> bool {
        self.export_job.is_some() || self.batch_export.as_ref().is_some_and(Job::running)
    }

    pub(super) fn selection_export_ready(&self) -> bool {
        !self.export_busy() && !self.image_loading() && self.crop_tool.is_none()
    }

    fn export_items(&mut self, photos: Vec<Photo>) -> Vec<Item> {
        let catalog = self.catalog.as_ref().unwrap().path().to_path_buf();
        let current_id = self
            .current_photo
            .as_ref()
            .filter(|(path, _)| path == &catalog)
            .map(|(_, id)| *id);
        let current = self.current_edit();
        let current_loaded = self.image.is_some();
        let defaults = self.default_edit();
        let store = self.edit_store();
        photos
            .into_iter()
            .map(|photo| {
                let active = current_id == Some(photo.id);
                let edit = if active {
                    Ok(current.clone())
                } else {
                    store
                        .ok_or_else(|| "Couldn't open the saved photo edits.".to_owned())
                        .and_then(|store| {
                            store
                                .load(&catalog, photo.id)
                                .map_err(|e| format!("Couldn't read saved edits: {e:#}"))
                        })
                        .map(|edit| edit.unwrap_or_else(|| defaults.clone()))
                };
                snapshot(photo, edit, !(active && current_loaded))
            })
            .collect()
    }

    pub(super) fn export_selection_dialog(&mut self, ctx: &egui::Context, photos: Vec<Photo>) {
        if photos.is_empty() || !self.selection_export_ready() {
            return;
        }
        let cli = if self.gpu_export {
            None
        } else {
            match locate_f64_cli() {
                Ok(path) => Some(path),
                Err(error) => {
                    self.status = format!("Export: {error}");
                    return;
                }
            }
        };
        let Some(folder) = rfd::FileDialog::new()
            .set_title(format!("Export {} photos to folder", photos.len()))
            .pick_folder()
        else {
            return;
        };
        self.commit_photo_edit();
        let items = self.export_items(photos);
        let progress = Arc::new(Mutex::new(Progress {
            total: items.len(),
            ..Default::default()
        }));
        let cancel = Arc::new(AtomicBool::new(false));
        let task = self.scheduler.task(Priority::Export);
        let worker_progress = progress.clone();
        let worker_cancel = cancel.clone();
        let worker_task = task.clone();
        let worker_folder = folder.clone();
        let options = self.current_save_options();
        let extension = self.export_format.extensions()[0];
        let backend = self.backend.clone();
        let data = self.data_dir.clone();
        let repaint = ctx.clone();
        let handle = std::thread::Builder::new()
            .name("spektrafilm-batch-export".into())
            .spawn(move || {
                worker_task.run(|| {
                    run_items(
                        items,
                        &worker_folder,
                        extension,
                        &worker_cancel,
                        &worker_progress,
                        |request, output| {
                            repaint.request_repaint();
                            request.run(
                                output,
                                &options,
                                &data,
                                backend.as_ref(),
                                cli.as_deref(),
                                &worker_cancel,
                                &worker_task,
                                None,
                            )
                        },
                    )
                });
                repaint.request_repaint();
            });
        match handle {
            Ok(handle) => {
                self.batch_export = Some(Job {
                    progress,
                    cancel,
                    task,
                    handle: Some(handle),
                    folder,
                })
            }
            Err(error) => self.status = format!("Couldn't start export: {error}"),
        }
    }

    pub(super) fn batch_progress_ui(&mut self, ui: &mut egui::Ui) {
        let Some(job) = &mut self.batch_export else {
            return;
        };
        job.poll();
        let running = job.running();
        let state = job.progress.lock().unwrap();
        let mut dismiss = false;
        ui.horizontal(|ui| {
            if running {
                ui.spinner();
                let text = if job.cancel.load(Ordering::SeqCst) {
                    "Cancelling export…".to_owned()
                } else {
                    format!(
                        "Exporting {} of {} · {}",
                        (state.done + 1).min(state.total),
                        state.total,
                        state.current
                    )
                };
                ui.add(egui::Label::new(text).truncate())
                    .on_hover_text(job.folder.display().to_string());
                if ui.small_button("Cancel").clicked() {
                    job.cancel();
                }
                ui.ctx().request_repaint_after(Duration::from_millis(200));
            } else {
                let prefix = if state.cancelled {
                    "Export cancelled"
                } else if state.fatal.is_some() {
                    "Export stopped"
                } else {
                    "Export finished"
                };
                ui.label(format!(
                    "{prefix}: {} exported · {} skipped · {} failed",
                    state.exported, state.skipped, state.failed
                ))
                .on_hover_text(format!(
                    "{} remaining\n{}",
                    state.total - state.done,
                    job.folder.display()
                ));
                dismiss = ui.small_button("Dismiss").clicked();
            }
        });
        if let Some(error) = &state.fatal {
            ui.colored_label(ui.visuals().error_fg_color, error);
        }
        if !state.issues.is_empty() {
            ui.collapsing(format!("{} export issues", state.issues.len()), |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("batch-export-errors")
                    .max_height(150.)
                    .show(ui, |ui| {
                        for (name, error) in &state.issues {
                            ui.label(format!("{name}: {error}"));
                        }
                    });
            });
        }
        ui.separator();
        drop(state);
        if dismiss {
            self.batch_export = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn photo(path: &Path, id: i64) -> Photo {
        Photo {
            id,
            path: path.into(),
            available: true,
            copy_name: format!("copy {id}"),
            flag: 1,
            geometry: Ok(spektrafilm_io::lightroom::Geometry {
                orientation: 1,
                crop: [0., 0., 1., 1.],
                angle: 0.,
            }),
            framing_warning: false,
        }
    }

    fn item(path: &Path, id: i64) -> Item {
        let mut edit = PhotoEdit::default();
        edit.params.camera.exposure_compensation_ev = id as f32;
        snapshot(photo(path, id), Ok(edit), true)
    }

    #[test]
    fn snapshots_use_each_saved_edit_and_the_active_edit_without_mutating_the_editor() {
        let dir = tempfile::tempdir().unwrap();
        let catalog = dir.path().join("catalog.lrcat");
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut app = App::new(
            &cc,
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            None,
        );
        let photos: Vec<_> = (0..3)
            .map(|id| photo(&dir.path().join("same.ARW"), id))
            .collect();
        app.catalog = Some(catalog_ui::Browser::test_catalog(
            catalog.clone(),
            photos.clone(),
            0,
            &ctx,
            &app.scheduler,
        ));
        let store = edits::EditStore::open(&dir.path().join("edits.sqlite")).unwrap();
        let mut saved = PhotoEdit::default();
        saved.params.camera.exposure_compensation_ev = -2.;
        store.save(&catalog, 0, &saved, 0).unwrap();
        saved.params.camera.exposure_compensation_ev = -1.;
        saved.raw.white_balance = "custom".into();
        saved.raw.temperature = 4200.;
        saved.rotation = 2;
        saved.crop = Some(Crop {
            size: [0.5, 0.5],
            ..Default::default()
        });
        store.save(&catalog, 1, &saved, 0).unwrap();
        app.edit_store = Some(store);
        app.current_photo = Some((catalog, 0));
        app.params.camera.exposure_compensation_ev = 0.5;
        app.params.camera.tone.shadows = 34.;
        app.image = Some(Arc::new(ImageBuf::new(4, 2)));
        let defaults = app.default_edit();
        let items = app.export_items(photos);
        app.params.camera.exposure_compensation_ev = 3.;
        let requests: Vec<_> = items
            .iter()
            .map(|item| item.request.as_ref().unwrap())
            .collect();
        assert_eq!(requests[0].edit.params.camera.exposure_compensation_ev, 0.5);
        assert_eq!(requests[0].edit.params.camera.tone.shadows, 34.);
        assert!(!requests[0].detect_input_color);
        assert_eq!(requests[1].edit.params.camera.exposure_compensation_ev, -1.);
        assert_eq!(requests[1].edit.raw.temperature, 4200.);
        assert_eq!(requests[1].edit.rotation, 2);
        assert_eq!(requests[1].edit.crop, saved.crop);
        assert!(requests[1].detect_input_color);
        assert_eq!(
            serde_json::to_value(&requests[2].edit.params).unwrap(),
            serde_json::to_value(&defaults.params).unwrap()
        );
        assert_eq!(app.catalog.as_ref().unwrap().selected, Some(0));
        assert_eq!(app.params.camera.exposure_compensation_ev, 3.);
    }

    #[test]
    fn queue_preserves_existing_files_and_virtual_copies_and_continues_after_missing_or_bad_images()
    {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("photo.ARW");
        std::fs::write(&source, b"original").unwrap();
        let name = filename(item(&source, 0).request.as_ref().unwrap());
        let existing = dir.path().join(format!("{name}.png"));
        std::fs::write(&existing, b"existing export").unwrap();
        let mut broken = item(&source, 3);
        broken.request = Err("Invalid saved settings".into());
        let items = vec![
            item(&source, 0),
            item(&dir.path().join("missing.ARW"), 1),
            broken,
            item(&source, 2),
        ];
        let progress = Mutex::new(Progress {
            total: items.len(),
            ..Default::default()
        });
        run_items(
            items,
            dir.path(),
            "png",
            &AtomicBool::new(false),
            &progress,
            |request, output| {
                std::fs::write(
                    output,
                    request
                        .edit
                        .params
                        .camera
                        .exposure_compensation_ev
                        .to_string(),
                )?;
                Ok(())
            },
        );
        let state = progress.lock().unwrap();
        assert_eq!(
            (state.exported, state.skipped, state.failed, state.done),
            (2, 1, 1, 4)
        );
        assert_eq!(state.issues.len(), 2);
        assert!(state.fatal.is_none());
        assert_eq!(std::fs::read(&source).unwrap(), b"original");
        assert_eq!(std::fs::read(&existing).unwrap(), b"existing export");
        assert_eq!(
            std::fs::read(dir.path().join(format!("{name}_2.png"))).unwrap(),
            b"0"
        );
        assert_eq!(
            std::fs::read(dir.path().join(format!("{name}_3.png"))).unwrap(),
            b"2"
        );
        assert!(!std::fs::read_dir(dir.path()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".spektrafilm-export-")
        }));
    }

    #[test]
    fn cancelling_keeps_completed_exports_and_removes_the_unpublished_image() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("photo.ARW");
        std::fs::write(&source, b"original").unwrap();
        let progress = Mutex::new(Progress {
            total: 3,
            ..Default::default()
        });
        let cancel = AtomicBool::new(false);
        let mut calls = 0;
        run_items(
            (0..3).map(|id| item(&source, id)).collect(),
            dir.path(),
            "png",
            &cancel,
            &progress,
            |_, output| {
                calls += 1;
                std::fs::write(output, b"rendered")?;
                if calls == 2 {
                    cancel.store(true, Ordering::SeqCst);
                }
                Ok(())
            },
        );
        let state = progress.lock().unwrap();
        assert_eq!(calls, 2);
        assert_eq!((state.exported, state.failed, state.done), (1, 0, 1));
        assert!(state.cancelled);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
    }

    #[test]
    fn destination_failure_stops_the_queue_after_preserving_finished_files() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("photo.ARW");
        std::fs::write(&source, b"original").unwrap();
        let progress = Mutex::new(Progress {
            total: 3,
            ..Default::default()
        });
        let mut calls = 0;
        run_items(
            (0..3).map(|id| item(&source, id)).collect(),
            dir.path(),
            "png",
            &AtomicBool::new(false),
            &progress,
            |_, output| {
                calls += 1;
                if calls == 2 {
                    return Err(
                        spektrafilm_io::OutputIoError(anyhow::anyhow!("Disk is full")).into(),
                    );
                }
                std::fs::write(output, b"rendered")?;
                Ok(())
            },
        );
        let state = progress.lock().unwrap();
        assert_eq!(
            (calls, state.exported, state.failed, state.done),
            (2, 1, 1, 2)
        );
        assert!(state.fatal.as_ref().unwrap().contains("Disk is full"));
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
    }

    #[test]
    fn concurrent_publication_uses_unique_names_without_overwriting() {
        let dir = tempfile::tempdir().unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let outputs = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|id| {
                    let barrier = barrier.clone();
                    let folder = dir.path();
                    scope.spawn(move || {
                        let temporary = tempfile::NamedTempFile::new_in(folder)
                            .unwrap()
                            .into_temp_path();
                        std::fs::write(&temporary, [id]).unwrap();
                        barrier.wait();
                        let path = publish_unique(
                            temporary,
                            folder,
                            "same",
                            "png",
                            &AtomicBool::new(false),
                        )
                        .unwrap();
                        assert_eq!(std::fs::read(&path).unwrap(), [id]);
                        path
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect::<HashSet<_>>()
        });
        assert_eq!(outputs.len(), 8);
    }
}
