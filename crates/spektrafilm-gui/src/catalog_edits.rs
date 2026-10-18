//! Per-photo settings and history for catalog photos.
//!
//! Opening a photo restores its saved look and history. A photo without saved
//! settings starts from the startup defaults (never from whatever was on screen
//! before), and only gets rows once it is edited; "Apply previous settings"
//! (P) copies the last photo's look on purpose. Every recorded history step is
//! saved, together with the photo's look and its current history position.
//! Standalone images get the same history in memory.
use super::*;
use crate::edits::{EditStore, PhotoEdit, StoredStep};
use crate::history::{History, Recorded, Snapshot, Step};

impl App {
    /// The look the current photo would be saved with.
    pub(crate) fn current_edit(&self) -> PhotoEdit {
        let mut raw = self.export_raw_options().clone();
        raw.lightroom_geometry = None;
        PhotoEdit {
            film: self.film_name.clone(),
            paper: self.print_name.clone(),
            params: self.params.clone(),
            scan_for_print_restore: self.scan_for_print_restore,
            rotation: self.rotation,
            raw,
        }
    }

    /// Open the edit store on first use. A store that can't be opened is
    /// reported once; the app keeps working without saving edits.
    fn edit_store(&mut self) -> Option<&EditStore> {
        if self.edit_store.is_none() && !self.edit_store_failed {
            match EditStore::open(&EditStore::default_path()) {
                Ok(store) => self.edit_store = Some(store),
                Err(e) => {
                    self.edit_store_failed = true;
                    self.status = format!("Photo edits won't be saved: {e:#}");
                }
            }
        }
        self.edit_store.as_ref()
    }

    // ── History ─────────────────────────────────────────────────────────

    /// Record a history step if the look changed since the current step, and
    /// save it for catalog photos. Returns true when a step was recorded.
    pub(crate) fn record_step(&mut self) -> bool {
        self.params_json = Arc::new(serde_json::to_value(&self.params).unwrap_or_default());
        let current = self.snapshot();
        let Some(previous) = self.history.pending(&current) else {
            self.pending_label = None;
            return false;
        };
        let label = match self.pending_label.take() {
            Some(label) => label,
            None => self.describe_change(&previous, &current),
        };
        let recorded = self.history.commit(current, label);
        self.persist_step(&recorded);
        true
    }

    fn persist_step(&mut self, recorded: &Recorded) {
        let Some((catalog, id)) = self.current_photo.clone() else {
            return;
        };
        let steps = self.history.steps().to_vec();
        let full_rewrite = recorded.folded || !self.history_persisted;
        let Some(store) = self.edit_store() else {
            return;
        };
        let result = if full_rewrite {
            let stored: Vec<StoredStep> = steps.iter().map(stored_step).collect();
            store.replace_history(&catalog, id, &stored)
        } else {
            let index = recorded.index;
            let previous = snapshot_state(&steps[index - 1].snapshot);
            store.record_step(
                &catalog,
                id,
                index,
                Some(&previous),
                &stored_step(&steps[index]),
            )
        };
        match result {
            Ok(()) => {
                self.history_persisted = true;
                self.save_photo_edit();
            }
            Err(e) => self.status = format!("Saving photo history failed: {e:#}"),
        }
    }

    /// Move through history (undo, redo or a click in the History panel),
    /// recording any pending edit first.
    pub(crate) fn go_to_step(&mut self, step: impl FnOnce(&mut History) -> Option<Snapshot>) {
        self.record_step();
        if let Some(snapshot) = step(&mut self.history) {
            self.apply_snapshot(snapshot);
            self.save_photo_edit();
        }
    }

    /// Start a fresh history from the current state, so undo never reaches
    /// into a previous photo.
    pub(crate) fn rebase_history(&mut self) {
        self.params_json = Arc::new(serde_json::to_value(&self.params).unwrap_or_default());
        self.history.reset(self.snapshot(), "Start");
        self.history_persisted = false;
    }

    /// Keep the history but let the current step absorb a state change that
    /// isn't an edit, such as a newly loaded file's colour encoding.
    pub(crate) fn amend_history(&mut self) {
        self.params_json = Arc::new(serde_json::to_value(&self.params).unwrap_or_default());
        self.history.amend_current(self.snapshot());
    }

    /// Restore the photo's saved history, or start a new one.
    fn load_photo_history(&mut self, catalog: &Path, id: i64) {
        let saved = match self
            .edit_store()
            .map(|store| store.load_history(catalog, id))
        {
            Some(Ok(saved)) => saved,
            Some(Err(e)) => {
                self.status = format!("Couldn't read this photo's history: {e:#}");
                None
            }
            None => None,
        };
        let steps: Option<Vec<Step>> = saved.as_ref().and_then(|(steps, _)| {
            steps
                .iter()
                .map(|s| {
                    Some(Step {
                        snapshot: state_snapshot(&s.state)?,
                        label: s.label.clone(),
                        at: s.at,
                    })
                })
                .collect()
        });
        match (steps, saved) {
            (Some(steps), Some((_, position))) => {
                self.history.restore(steps, position);
                self.history_persisted = true;
                // The photo's own look (saved alongside) is what's current.
                self.amend_history();
            }
            _ => self.rebase_history(),
        }
    }

    /// Forget the history, keeping the current look as the new start.
    pub(crate) fn clear_history(&mut self) {
        self.rebase_history();
        let Some((catalog, id)) = self.current_photo.clone() else {
            return;
        };
        let base = stored_step(&self.history.steps()[0]);
        let result = match self.edit_store() {
            Some(store) => store.replace_history(&catalog, id, &[base]),
            None => return,
        };
        match result {
            Ok(()) => {
                self.history_persisted = true;
                self.save_photo_edit();
            }
            Err(e) => self.status = format!("Clearing history failed: {e:#}"),
        }
    }

    /// A readable label for the change between two steps, e.g.
    /// "Halation amount 1.00 → 2.00" or "Film: Kodak Portra 400".
    pub(crate) fn describe_change(&self, previous: &Snapshot, current: &Snapshot) -> String {
        let stock = |entries: &[ProfileEntry], stock: &str| {
            entries
                .iter()
                .find(|e| e.stock == stock)
                .map_or(stock.to_owned(), |e| e.display.clone())
        };
        if previous.film != current.film {
            return format!("Film: {}", stock(&self.films, &current.film));
        }
        if previous.print != current.print {
            return format!("Paper: {}", stock(&self.papers, &current.print));
        }
        if previous.scan_for_print_restore.is_some() != current.scan_for_print_restore.is_some() {
            return format!(
                "Scan for print: {}",
                on_off(current.scan_for_print_restore.is_some())
            );
        }
        let pointers = inspector::diff_pointers(&previous.params, &current.params);
        let mut names: Vec<String> = Vec::new();
        for pointer in &pointers {
            let name = match self.pointer_captions.get(pointer) {
                Some(caption) => caption_name(caption),
                None => pointer_name(pointer),
            };
            if !names.contains(&name) {
                names.push(name);
            }
        }
        match (names.as_slice(), pointers.as_slice()) {
            ([], _) => "Edit".into(),
            ([name], [pointer]) => {
                let old = previous.params.pointer(pointer);
                let new = current.params.pointer(pointer);
                match (old, new) {
                    (_, Some(serde_json::Value::Bool(on))) => format!("{name}: {}", on_off(*on)),
                    (Some(old), Some(new)) => {
                        let (old, new) = change_text(old, new);
                        format!("{name} {old} → {new}")
                    }
                    _ => name.clone(),
                }
            }
            ([name], _) => name.clone(),
            ([first, rest @ ..], _) => format!("{first} and {} more", rest.len()),
        }
    }

    // ── Saved look ──────────────────────────────────────────────────────

    /// Persist the current photo's look and history position, if a catalog
    /// photo is open.
    pub(crate) fn save_photo_edit(&mut self) {
        let Some((catalog, id)) = self.current_photo.clone() else {
            return;
        };
        let edit = self.current_edit();
        let position = self.history.position();
        let Some(store) = self.edit_store() else {
            return;
        };
        match store.save(&catalog, id, &edit, position) {
            Ok(()) => {
                if let Some(browser) = &mut self.catalog {
                    browser.edited.insert(id);
                }
            }
            Err(e) => self.status = format!("Saving photo edit failed: {e:#}"),
        }
    }

    /// Record any edit still pending (e.g. before switching photos).
    pub(crate) fn commit_photo_edit(&mut self) {
        self.record_step();
    }

    /// Open a catalog photo's saved look and history. Returns its saved
    /// rotation; a photo without saved settings starts from the defaults.
    pub(crate) fn open_photo_edit(&mut self, catalog: &Path, id: i64) -> u8 {
        let saved = match self.edit_store().map(|store| store.load(catalog, id)) {
            Some(Ok(saved)) => saved,
            Some(Err(e)) => {
                self.status = format!("Couldn't read this photo's saved edit: {e:#}");
                None
            }
            None => None,
        };
        let edit = saved.unwrap_or_else(|| self.default_edit());
        let rotation = edit.rotation % 4;
        self.apply_edit(edit, true);
        self.load_photo_history(catalog, id);
        rotation
    }

    /// Apply a look. The Lightroom crop and the image's own colour encoding
    /// are kept: those come from the catalog and the file.
    pub(crate) fn apply_edit(&mut self, edit: PhotoEdit, include_raw: bool) {
        let input_color = self.params.io.input_color_space.clone();
        let input_encoded = self.params.io.input_cctf_decoding;
        if self.film_name != edit.film {
            self.film_name = edit.film;
            self.film_dev_times = profile_dev_times(&self.data_dir, &self.film_name);
        }
        if self.print_name != edit.paper {
            self.print_name = edit.paper;
            self.print_dev_times = profile_dev_times(&self.data_dir, &self.print_name);
        }
        self.params = edit.params;
        self.params.io.input_color_space = input_color;
        self.params.io.input_cctf_decoding = input_encoded;
        self.scan_for_print_restore = edit.scan_for_print_restore;
        if include_raw {
            let geometry = self.raw_options.lightroom_geometry.take();
            self.raw_options = edit.raw;
            self.raw_options.lightroom_geometry = geometry;
        }
        self.dirty = true;
        self.pending_dirty = self.render_job.is_some();
    }

    /// Startup defaults if saved, otherwise factory settings with the stock's
    /// own defaults: what "Reset to defaults" restores.
    fn default_edit(&self) -> PhotoEdit {
        if let Ok(project) = project::Project::load(&project::default_path()) {
            return PhotoEdit {
                film: project.film,
                paper: project.paper,
                params: project.params,
                scan_for_print_restore: project.scan_for_print_restore,
                rotation: 0,
                raw: project.raw,
            };
        }
        // Factory settings: the default stocks, never whatever is on screen.
        let mut edit = PhotoEdit::default();
        if let Ok(film) = profile::load_profile_by_name(&self.data_dir, &edit.film) {
            edit.params.io.scan_film = film.is_positive();
            spektrafilm_core::pipeline::apply_stock_defaults(&film, &mut edit.params);
        }
        edit
    }

    /// Return the photo to the defaults, as an undoable history step.
    pub(crate) fn reset_photo_edit(&mut self) {
        let defaults = self.default_edit();
        self.apply_edit(defaults, false);
        self.pending_label = Some("Reset to defaults".into());
        self.record_step();
    }

    /// Apply the look of the photo viewed before this one (P), as one step.
    pub(crate) fn apply_previous_edit(&mut self) {
        let Some(edit) = self.previous_edit.clone() else {
            return;
        };
        self.apply_edit(edit, false);
        self.pending_label = Some("Apply previous settings".into());
        self.record_step();
    }

    pub(crate) fn copy_photo_edit(&mut self) {
        self.copied_edit = Some(self.current_edit());
        self.status = "Settings copied".into();
    }

    /// Paste the copied look (not rotation or RAW options) as one history step.
    pub(crate) fn paste_photo_edit(&mut self) {
        let Some(edit) = self.copied_edit.clone() else {
            return;
        };
        self.apply_edit(edit, false);
        self.pending_label = Some("Paste settings".into());
        self.record_step();
    }

    /// Leave the current catalog photo (closing the catalog, opening another
    /// file or project): save it, and stop saving later edits to it.
    pub(crate) fn leave_photo(&mut self) {
        self.commit_photo_edit();
        self.current_photo = None;
        self.history_persisted = false;
    }

    /// Mark the browser's edited photos for a newly opened catalog.
    pub(crate) fn load_edited_markers(&mut self) {
        let Some(path) = self.catalog.as_ref().map(|c| c.path().to_path_buf()) else {
            return;
        };
        let edited = match self.edit_store().map(|store| store.edited(&path)) {
            Some(Ok(edited)) => edited,
            Some(Err(e)) => {
                self.status = format!("Couldn't read saved photo edits: {e:#}");
                return;
            }
            None => return,
        };
        if let Some(browser) = &mut self.catalog {
            browser.edited = edited;
        }
    }
}

/// The look part of a history step, as stored.
fn snapshot_state(snapshot: &Snapshot) -> serde_json::Value {
    serde_json::json!({
        "film": snapshot.film,
        "paper": snapshot.print,
        "params": *snapshot.params,
        "scan_for_print_restore": snapshot.scan_for_print_restore,
    })
}

fn state_snapshot(state: &serde_json::Value) -> Option<Snapshot> {
    Some(Snapshot {
        film: state.get("film")?.as_str()?.to_owned(),
        print: state.get("paper")?.as_str()?.to_owned(),
        params: Arc::new(state.get("params")?.clone()),
        scan_for_print_restore: serde_json::from_value(
            state
                .get("scan_for_print_restore")
                .cloned()
                .unwrap_or_default(),
        )
        .ok()?,
    })
}

fn stored_step(step: &Step) -> StoredStep {
    StoredStep {
        label: step.label.clone(),
        at: step.at,
        state: snapshot_state(&step.snapshot),
    }
}

fn on_off(on: bool) -> &'static str {
    if on { "on" } else { "off" }
}

/// An inspector caption without its unit, e.g. "Lens blur (µm)" → "Lens blur".
fn caption_name(caption: &str) -> String {
    let caption = caption.trim_end().trim_end_matches(':').trim_end();
    match caption.rfind(" (") {
        Some(at) if caption.ends_with(')') => caption[..at].to_owned(),
        _ => caption.to_owned(),
    }
}

/// A name for a parameter no inspector row has claimed yet, e.g.
/// "/film_render/halation/active" → "Halation".
fn pointer_name(pointer: &str) -> String {
    let parts: Vec<&str> = pointer
        .split('/')
        .filter(|p| !p.is_empty() && p.parse::<usize>().is_err())
        .collect();
    let part = match parts.as_slice() {
        [.., parent, "active"] => parent,
        [.., last] => last,
        [] => return "Edit".into(),
    };
    names::label(part).into_owned()
}

/// Both sides of a change at the same precision: "1.00 → 1.60", "3 → 4".
fn change_text(old: &serde_json::Value, new: &serde_json::Value) -> (String, String) {
    let whole = |v: &serde_json::Value| v.as_f64().is_some_and(|v| v.fract() == 0.0);
    match (old.as_f64(), new.as_f64()) {
        (Some(a), Some(b)) if whole(old) && whole(new) => (format!("{a:.0}"), format!("{b:.0}")),
        (Some(a), Some(b)) => (format!("{a:.2}"), format!("{b:.2}")),
        _ => (value_text(old), value_text(new)),
    }
}

fn value_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Number(n) => format!("{:.2}", n.as_f64().unwrap_or_default()),
        serde_json::Value::String(s) => names::label(s).into_owned(),
        serde_json::Value::Bool(on) => on_off(*on).into(),
        _ => "…".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spektrafilm_io::lightroom::{Geometry, Photo};

    fn finish(app: &mut App) {
        while let Some(job) = &mut app.load_job {
            if let Some(handle) = job.handle.take() {
                handle.join().unwrap();
            }
            app.poll_load_job();
        }
        assert!(app.image.is_some(), "{}", app.status);
    }

    fn new_app(store: &Path) -> App {
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx);
        let mut app = App::new(
            &cc,
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            None,
        );
        app.raw_options.decoder = "rawler".into();
        app.edit_store = Some(EditStore::open(store).unwrap());
        app
    }

    /// A minimal Lightroom catalog on disk, so the test can check it's never written.
    fn write_catalog(dir: &Path) -> PathBuf {
        let path = dir.join("test.lrcat");
        let db = rusqlite::Connection::open(&path).unwrap();
        db.execute_batch(
            "CREATE TABLE Adobe_images(id_local INTEGER, rootFile INTEGER, copyName TEXT, pick REAL, orientation TEXT, fileFormat TEXT, captureTime TEXT);
             CREATE TABLE AgLibraryFile(id_local INTEGER, folder INTEGER, idx_filename TEXT);
             CREATE TABLE AgLibraryFolder(id_local INTEGER, rootFolder INTEGER, pathFromRoot TEXT);
             CREATE TABLE AgLibraryRootFolder(id_local INTEGER, absolutePath TEXT);
             CREATE TABLE Adobe_imageDevelopSettings(image INTEGER, text TEXT);",
        )
        .unwrap();
        drop(db);
        path
    }

    fn photo(dir: &Path, id: i64) -> Photo {
        let path = dir.join(format!("{id}.dng"));
        std::fs::copy(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../spektrafilm-io/tests/fixtures/synthetic.dng"),
            &path,
        )
        .unwrap();
        Photo {
            id,
            path,
            available: true,
            copy_name: String::new(),
            flag: 0,
            geometry: Ok(Geometry {
                orientation: 1,
                crop: [0., 0., 1., 1.],
                angle: 0.,
            }),
            framing_warning: false,
        }
    }

    fn open(app: &mut App, catalog: &Path, photo: &Photo) {
        app.open_catalog(catalog.to_path_buf());
        app.select_catalog_photo(photo.clone());
        finish(app);
    }

    fn select(app: &mut App, photo: &Photo) {
        app.select_catalog_photo(photo.clone());
        finish(app);
    }

    fn ev(app: &App) -> f64 {
        app.params.camera.exposure_compensation_ev as f64
    }

    #[test]
    fn photos_keep_their_own_settings_across_switches_and_reopening() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("edits.sqlite");
        let catalog = write_catalog(dir.path());
        let catalog_bytes = std::fs::read(&catalog).unwrap();
        let (a, b, c) = (
            photo(dir.path(), 1),
            photo(dir.path(), 2),
            photo(dir.path(), 3),
        );

        let mut app = new_app(&store);
        open(&mut app, &catalog, &a);
        let default_ev = ev(&app);
        app.params.camera.exposure_compensation_ev = 1.25;
        app.commit_photo_edit();

        // B has no saved settings, so it starts from the defaults, not A's look.
        select(&mut app, &b);
        assert_eq!(ev(&app), default_ev);
        app.params.camera.exposure_compensation_ev = -0.5;
        app.commit_photo_edit();

        select(&mut app, &a);
        assert_eq!(ev(&app), 1.25, "A keeps its own settings");
        select(&mut app, &b);
        assert_eq!(ev(&app), -0.5, "B keeps its own settings");
        // Viewing C without editing it shows the defaults and saves nothing for C.
        select(&mut app, &c);
        assert_eq!(ev(&app), default_ev);
        select(&mut app, &a);
        app.leave_photo();
        drop(app);

        // Reopening the catalog restores each photo's own settings.
        let mut app = new_app(&store);
        open(&mut app, &catalog, &b);
        assert_eq!(ev(&app), -0.5);
        select(&mut app, &a);
        assert_eq!(ev(&app), 1.25);
        let edited = app.edit_store.as_ref().unwrap().edited(&catalog).unwrap();
        assert_eq!(edited, std::collections::HashSet::from([1, 2]));

        assert_eq!(
            std::fs::read(&catalog).unwrap(),
            catalog_bytes,
            "the Lightroom catalog is never written"
        );
    }

    #[test]
    fn apply_previous_copies_the_last_photos_look_as_one_saved_step() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("edits.sqlite");
        let catalog = write_catalog(dir.path());
        let (a, b) = (photo(dir.path(), 1), photo(dir.path(), 2));
        let mut app = new_app(&store);
        open(&mut app, &catalog, &a);
        assert!(
            app.previous_edit.is_none(),
            "nothing to apply on the first photo"
        );
        app.params.camera.exposure_compensation_ev = 1.25;
        app.film_name = "kodak_portra_400".into();
        app.commit_photo_edit();

        select(&mut app, &b);
        let default_ev = ev(&app);
        app.apply_previous_edit();
        assert_eq!(ev(&app), 1.25);
        assert_eq!(app.film_name, "kodak_portra_400");
        assert_eq!(app.history.undo_label(), Some("Apply previous settings"));
        app.undo();
        assert_eq!(
            ev(&app),
            default_ev,
            "applying the previous look is undoable"
        );
        app.redo();
        app.leave_photo();
        drop(app);

        let mut app = new_app(&store);
        open(&mut app, &catalog, &b);
        assert_eq!(ev(&app), 1.25, "the applied look was saved for B");
    }

    #[test]
    fn switching_photos_saves_a_pending_edit_and_clears_undo() {
        let dir = tempfile::tempdir().unwrap();
        let catalog = write_catalog(dir.path());
        let (a, b) = (photo(dir.path(), 1), photo(dir.path(), 2));
        let mut app = new_app(&dir.path().join("edits.sqlite"));
        open(&mut app, &catalog, &a);

        // An edit made but not yet committed (e.g. mid-gesture) is saved on switch.
        app.params.camera.exposure_compensation_ev = 0.75;
        select(&mut app, &b);
        assert!(
            !app.history.can_undo(),
            "undo can't reach into the previous photo"
        );
        select(&mut app, &a);
        assert_eq!(ev(&app), 0.75);
    }

    #[test]
    fn virtual_copies_are_independent_and_reset_is_an_undoable_step() {
        let dir = tempfile::tempdir().unwrap();
        let catalog = write_catalog(dir.path());
        let original = photo(dir.path(), 1);
        let copy = Photo {
            id: 2,
            copy_name: "Copy 1".into(),
            ..original.clone()
        };
        let mut app = new_app(&dir.path().join("edits.sqlite"));
        open(&mut app, &catalog, &original);
        app.params.camera.exposure_compensation_ev = 1.0;
        app.commit_photo_edit();
        select(&mut app, &copy);
        app.params.camera.exposure_compensation_ev = -1.0;
        app.commit_photo_edit();

        app.reset_photo_edit();
        assert_ne!(ev(&app), -1.0);
        assert_eq!(app.history.undo_label(), Some("Reset to defaults"));
        app.undo();
        assert_eq!(ev(&app), -1.0, "reset can be undone");
        select(&mut app, &original);
        assert_eq!(ev(&app), 1.0, "the copy's edits leave the original alone");
    }

    #[test]
    fn history_survives_reopening_and_a_new_edit_after_jumping_back_drops_later_steps() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("edits.sqlite");
        let catalog = write_catalog(dir.path());
        let a = photo(dir.path(), 1);

        let mut app = new_app(&store);
        open(&mut app, &catalog, &a);
        for value in [0.5, 1.0, 1.5] {
            app.params.camera.exposure_compensation_ev = value;
            app.commit_photo_edit();
        }
        app.film_name = "kodak_portra_400".into();
        app.commit_photo_edit();
        let labels: Vec<_> = app
            .history
            .steps()
            .iter()
            .map(|s| s.label.clone())
            .collect();
        assert_eq!(labels[0], "Start");
        assert!(
            labels[1].starts_with("Exposure compensation ev 0.00 → 0.50"),
            "{labels:?}"
        );
        assert!(labels[4].starts_with("Film: "), "{labels:?}");
        // Go back two steps, as if clicking in the History panel.
        app.go_to_step(|h| h.jump(2));
        assert_eq!(ev(&app), 1.0);
        app.leave_photo();
        drop(app);

        // After reopening: same steps, same position, and undo still works.
        let mut app = new_app(&store);
        open(&mut app, &catalog, &a);
        assert_eq!(app.history.steps().len(), 5);
        assert_eq!(app.history.position(), 2);
        assert_eq!(ev(&app), 1.0);
        assert!(app.history.can_redo());
        app.undo();
        assert_eq!(ev(&app), 0.5);

        // A new edit here discards the undone steps, in memory and on disk.
        app.params.camera.exposure_compensation_ev = -2.0;
        app.commit_photo_edit();
        assert_eq!(app.history.steps().len(), 3);
        app.leave_photo();
        drop(app);
        let mut app = new_app(&store);
        open(&mut app, &catalog, &a);
        assert_eq!(app.history.steps().len(), 3);
        assert_eq!(ev(&app), -2.0);
        assert!(!app.history.can_redo());
    }

    #[test]
    fn steps_are_labelled_with_the_inspector_caption_when_known() {
        let dir = tempfile::tempdir().unwrap();
        let app = new_app(&dir.path().join("edits.sqlite"));
        let snapshot = |ev: f64, halation: bool| Snapshot {
            params: Arc::new(serde_json::json!({
                "camera": {"exposure_compensation_ev": ev},
                "film_render": {"halation": {"active": halation}}
            })),
            film: "kodak_gold_200".into(),
            print: "kodak_portra_endura".into(),
            scan_for_print_restore: None,
        };
        let mut app = app;
        assert_eq!(
            app.describe_change(&snapshot(0.0, true), &snapshot(0.0, false)),
            "Halation: off"
        );
        app.pointer_captions.insert(
            "/camera/exposure_compensation_ev".into(),
            "EV compensation".into(),
        );
        assert_eq!(
            app.describe_change(&snapshot(0.0, true), &snapshot(1.25, true)),
            "EV compensation 0.00 → 1.25"
        );
        assert_eq!(
            app.describe_change(&snapshot(0.0, true), &snapshot(1.0, false)),
            "EV compensation and 1 more"
        );
        assert_eq!(
            app.describe_change(&snapshot(1.0, true), &snapshot(1.6, true)),
            "EV compensation 1.00 → 1.60"
        );
    }
}
