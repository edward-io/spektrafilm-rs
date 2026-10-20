//! Bounded, nearest-first RAW prefetching. Only decoded originals are cached;
//! the selected photo's display preparation and film render stay foreground.
use super::*;
use spektrafilm_io::{RawOptions, lightroom::Geometry};
use std::collections::VecDeque;

pub(super) const DEFAULT_RADIUS: u8 = 2;
const PRELOAD_BYTES: usize = 2 * 1024 * 1024 * 1024;
const REVISIT_BYTES: usize = 512 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
struct Key {
    path: PathBuf,
    rotation: u8,
    options: RawOptions,
}

impl Key {
    fn matches(&self, request: &LoadRequest) -> bool {
        self.path == request.path
            && self.rotation == request.rotation
            && self.options == request.raw_options
    }
}

struct Entry {
    key: Key,
    source: SourceImage,
}

#[derive(PartialEq)]
struct Window {
    catalog: PathBuf,
    radius: u8,
    photos: Vec<(i64, PathBuf, Geometry)>,
}

struct Job {
    key: Key,
    task: Task,
    started: Arc<AtomicBool>,
    rx: mpsc::Receiver<Result<SourceImage, String>>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for Job {
    fn drop(&mut self) {
        self.task.cancel();
    }
}

impl Job {
    fn start(key: Key, scheduler: &Scheduler, ctx: &egui::Context) -> std::io::Result<Self> {
        let task = scheduler.task(Priority::Background);
        let worker_task = task.clone();
        let worker_key = key.clone();
        let started = Arc::new(AtomicBool::new(false));
        let worker_started = started.clone();
        let repaint = ctx.clone();
        let (tx, rx) = mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("spektrafilm-preload".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    worker_task.run(|| {
                        let _cpu = worker_task.cpu().map_err(|e| e.to_string())?;
                        worker_started.store(true, Ordering::Release);
                        SourceImage::load(
                            &worker_key.path,
                            &worker_key.options,
                            worker_key.rotation,
                            &worker_task,
                        )
                    })
                }))
                .unwrap_or_else(|panic| Err(panic_message(panic)));
                let _ = tx.send(result);
                repaint.request_repaint();
            })?;
        Ok(Self {
            key,
            task,
            started,
            rx,
            handle: Some(handle),
        })
    }
}

pub(super) struct Preloader {
    entries: VecDeque<Entry>,
    current: Option<Key>,
    window: Option<Window>,
    wanted: Vec<Key>,
    pending: VecDeque<Key>,
    job: Option<Job>,
    waiting: Option<LoadRequest>,
    budget: usize,
}

impl Default for Preloader {
    fn default() -> Self {
        Self {
            entries: VecDeque::new(),
            current: None,
            window: None,
            wanted: vec![],
            pending: VecDeque::new(),
            job: None,
            waiting: None,
            budget: REVISIT_BYTES,
        }
    }
}

impl Preloader {
    pub(super) fn cached(
        &mut self,
        path: &Path,
        rotation: u8,
        options: &RawOptions,
    ) -> Option<SourceImage> {
        let index = self.entries.iter().position(|entry| {
            entry.key.path == path
                && entry.key.rotation == rotation
                && &entry.key.options == options
        })?;
        let entry = self.entries.remove(index)?;
        let source = entry.source.clone();
        self.entries.push_back(entry);
        Some(source)
    }

    pub(super) fn remember_current(
        &mut self,
        path: PathBuf,
        rotation: u8,
        options: RawOptions,
        source: SourceImage,
    ) {
        let key = Key {
            path,
            rotation,
            options,
        };
        self.current = Some(key.clone());
        self.insert(key, source);
    }

    fn insert(&mut self, key: Key, source: SourceImage) {
        self.entries.retain(|entry| entry.key != key);
        if self.current.as_ref() != Some(&key)
            && std::mem::size_of_val(source.pixels.data.as_slice()) > self.budget
        {
            return;
        }
        self.entries.push_back(Entry { key, source });
        self.trim();
    }

    fn bytes(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| Some(&entry.key) != self.current.as_ref())
            .map(|entry| std::mem::size_of_val(entry.source.pixels.data.as_slice()))
            .sum()
    }

    fn trim(&mut self) {
        while self.bytes() > self.budget {
            // Evict old revisits, then the farthest neighbour; keep the selected original.
            let index = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| Some(&entry.key) != self.current.as_ref())
                .max_by_key(|(index, entry)| {
                    (
                        self.wanted
                            .iter()
                            .position(|key| key == &entry.key)
                            .unwrap_or(usize::MAX),
                        usize::MAX - index,
                    )
                })
                .map(|(index, _)| index);
            let Some(index) = index else { break };
            self.entries.remove(index);
        }
    }

    pub(super) fn waiting(&self) -> Option<&LoadRequest> {
        self.waiting.as_ref()
    }
    pub(super) fn abandon_wait(&mut self) {
        if let Some(request) = &self.waiting
            && let Some(job) = &self.job
            && job.key.matches(request)
            && !self.wanted.contains(&job.key)
        {
            job.task.cancel();
        }
        self.waiting = None;
    }

    pub(super) fn stop(&mut self) {
        if let Some(job) = &self.job {
            job.task.cancel();
        }
        self.waiting = None;
        self.pending.clear();
        self.wanted.clear();
        self.window = None;
        self.budget = REVISIT_BYTES;
        self.trim();
    }

    pub(super) fn invalidate_window(&mut self) {
        self.window = None;
    }

    pub(super) fn forget_current(&mut self) {
        self.current = None;
        self.trim();
    }
}

impl App {
    pub(super) fn image_loading(&self) -> bool {
        self.load_job.is_some() || self.preload.waiting.is_some()
    }

    pub(super) fn wait_for_preload(&mut self, request: LoadRequest) -> Option<LoadRequest> {
        let already_cached = self
            .preload
            .entries
            .iter()
            .any(|entry| entry.key.matches(&request));
        if !already_cached
            && let Some(job) = &self.preload.job
            && job.key.matches(&request)
            && job.task.check().is_ok()
        {
            if job.started.load(Ordering::Acquire) {
                // The decoder is already running. Reuse its result, then prepare the display
                // with a foreground task, without queuing another decode of the same RAW.
                self.status = format!(
                    "Finishing background decode of {}…",
                    request
                        .path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                );
                self.preload.waiting = Some(request);
                return None;
            }
            // A queued preload must not make the selected image wait behind other work.
            job.task.cancel();
        }
        Some(request)
    }

    pub(super) fn poll_preload(&mut self) {
        let Some(job) = self.preload.job.as_ref() else {
            return;
        };
        let result = match job.rx.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("preload worker stopped unexpectedly".into())
            }
        };
        let mut job = self.preload.job.take().unwrap();
        if let Some(handle) = job.handle.take() {
            let _ = handle.join();
        }
        let waiting = self.preload.waiting.take();
        let needed = self.preload.wanted.contains(&job.key)
            || waiting
                .as_ref()
                .is_some_and(|request| job.key.matches(request));
        if needed && job.task.check().is_ok() {
            match result {
                Ok(source) if source.stamp.is_some() => {
                    if waiting
                        .as_ref()
                        .is_some_and(|request| job.key.matches(request))
                    {
                        self.preload.current = Some(job.key.clone());
                    }
                    self.preload.insert(job.key.clone(), source)
                }
                Err(error) => {
                    tracing::debug!(path = %job.key.path.display(), %error, "preload skipped")
                }
                _ => {}
            }
        }
        if let Some(request) = waiting {
            // A failed or invalidated preload falls back to normal foreground loading.
            self.start_image_load(request);
        }
    }

    pub(super) fn update_preloading(&mut self) {
        let Some(browser) = self.catalog.as_ref() else {
            if self.preload.window.is_some() {
                self.preload.stop();
            }
            return;
        };
        let window = Window {
            catalog: browser.path().to_path_buf(),
            radius: self.preload_radius.min(8),
            photos: browser
                .preload_photos(self.preload_radius.min(8))
                .into_iter()
                .filter_map(|photo| {
                    Some((
                        photo.id,
                        std::path::absolute(photo.path).ok()?,
                        photo.geometry.ok()?,
                    ))
                })
                .collect(),
        };
        if self.preload.window.as_ref() != Some(&window) {
            let defaults = self.default_edit().raw;
            let wanted: Vec<Key> = window.photos.iter().filter_map(|(id,path,geometry)| {
                let edit = match self.edit_store.as_ref().map(|store| store.load(&window.catalog,*id)) {
                    Some(Ok(edit)) => edit,
                    Some(Err(error)) => {
                        tracing::debug!(%id, %error, "preload could not read saved RAW settings");
                        return None;
                    }
                    None => None,
                };
                let (mut options, rotation) = edit.map(|edit| (edit.raw, edit.rotation % 4))
                    .unwrap_or_else(|| (defaults.clone(), 0));
                options.lightroom_geometry = Some(geometry.clone());
                Some(Key { path: path.clone(), rotation, options })
            }).collect();
            if let Some(job) = &self.preload.job
                && !wanted.contains(&job.key)
                && !self
                    .preload
                    .waiting
                    .as_ref()
                    .is_some_and(|request| job.key.matches(request))
            {
                job.task.cancel();
            }
            self.preload.pending = wanted
                .iter()
                .filter(|key| {
                    !self.preload.entries.iter().any(|entry| &entry.key == *key)
                        && !self
                            .preload
                            .job
                            .as_ref()
                            .is_some_and(|job| &job.key == *key && job.task.check().is_ok())
                })
                .cloned()
                .collect();
            self.preload.wanted = wanted;
            self.preload.window = Some(window);
            self.preload.budget = if self.preload_radius == 0 {
                REVISIT_BYTES
            } else {
                PRELOAD_BYTES
            };
            self.preload.trim();
        }
        if self.preload.job.is_none()
            && self.preload.waiting.is_none()
            && self.load_job.is_none()
            && self.render_job.is_none()
            && (!self.dirty || (!self.auto_preview && !self.render_requested))
            && self.image.is_some()
        {
            while let Some(key) = self.preload.pending.pop_front() {
                if self.preload.entries.iter().any(|entry| entry.key == key) {
                    continue;
                }
                // Reserve using the largest decode seen in this window; keep the nearest neighbours first.
                let rank = self
                    .preload
                    .wanted
                    .iter()
                    .position(|wanted| wanted == &key)
                    .unwrap_or(usize::MAX);
                let nearer: Vec<&Entry> = self
                    .preload
                    .entries
                    .iter()
                    .filter(|entry| {
                        self.preload
                            .wanted
                            .iter()
                            .position(|wanted| wanted == &entry.key)
                            .is_some_and(|i| i < rank)
                    })
                    .collect();
                let bytes =
                    |source: &SourceImage| std::mem::size_of_val(source.pixels.data.as_slice());
                let estimate = nearer
                    .iter()
                    .map(|entry| bytes(&entry.source))
                    .chain(
                        self.image
                            .iter()
                            .map(|image| std::mem::size_of_val(image.data.as_slice())),
                    )
                    .max()
                    .unwrap_or(0);
                if nearer
                    .iter()
                    .map(|entry| bytes(&entry.source))
                    .sum::<usize>()
                    + estimate
                    > self.preload.budget
                {
                    self.preload.pending.clear();
                    break;
                }
                match Job::start(key, &self.scheduler, &self.ctx) {
                    Ok(job) => self.preload.job = Some(job),
                    Err(error) => {
                        tracing::warn!(%error, "could not start RAW preload");
                        self.preload.pending.clear();
                    }
                }
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spektrafilm_io::lightroom::Photo;

    fn app(backend: Arc<dyn ComputeBackend>) -> App {
        let ctx = egui::Context::default();
        App::new(&eframe::CreationContext::_new_kittest(ctx), backend, None)
    }

    fn photos(dir: &Path, count: usize) -> Vec<Photo> {
        (0..count)
            .map(|id| {
                let path = dir.join(format!("{id}.dng"));
                std::fs::copy(
                    Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../spektrafilm-io/tests/fixtures/synthetic.dng"),
                    &path,
                )
                .unwrap();
                Photo {
                    id: id as i64,
                    path,
                    flag: 1,
                    available: true,
                    copy_name: String::new(),
                    framing_warning: false,
                    geometry: Ok(Geometry {
                        orientation: 1,
                        crop: [0., 0., 1., 1.],
                        angle: 0.,
                    }),
                }
            })
            .collect()
    }

    fn select(app: &mut App, photo: &Photo) {
        if let Some(browser) = &mut app.catalog {
            browser.selected = Some(photo.id as usize);
        }
        app.select_catalog_photo(photo.clone());
    }

    fn finish_load(app: &mut App) {
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            app.poll_preload();
            if let Some(job) = &mut app.load_job {
                job.handle.take().unwrap().join().unwrap();
                app.poll_load_job();
            }
            if app.load_job.is_none() && app.preload.waiting.is_none() {
                break;
            }
            assert!(Instant::now() < deadline, "foreground load did not finish");
            std::thread::yield_now();
        }
    }

    fn finish_window(app: &mut App) {
        // Measure and test decoded-source prefetch, independently of film rendering.
        app.dirty = false;
        app.update_preloading();
        let deadline = Instant::now() + Duration::from_secs(30);
        while app.preload.job.is_some() || !app.preload.pending.is_empty() {
            assert!(Instant::now() < deadline, "preload window did not finish");
            if let Some(handle) = app.preload.job.as_mut().and_then(|job| job.handle.take()) {
                handle.join().unwrap();
            }
            app.poll_preload();
            app.update_preloading();
        }
    }

    #[test]
    fn cache_budget_keeps_current_and_nearest_images() {
        let mut cache = Preloader::default();
        let source = || SourceImage {
            pixels: Arc::new(ImageBuf::new(2, 1)),
            metadata: Default::default(),
            color_space: "sRGB".into(),
            encoded: false,
            stamp: None,
        };
        let key = |id| Key {
            path: PathBuf::from(format!("{id}.dng")),
            rotation: 0,
            options: Default::default(),
        };
        let size = std::mem::size_of_val(source().pixels.data.as_slice());
        cache.budget = size * 2;
        cache.wanted = (1..=4).map(key).collect();
        cache.remember_current(key(0).path, 0, Default::default(), source());
        for id in 1..=4 {
            cache.insert(key(id), source());
        }
        assert_eq!(cache.bytes(), size * 2);
        assert_eq!(
            cache
                .entries
                .iter()
                .map(|entry| entry.key.path.clone())
                .collect::<Vec<_>>(),
            [key(0).path, key(1).path, key(2).path]
        );
        cache.wanted.clear();
        assert!(cache.cached(&key(1).path, 0, &Default::default()).is_some());
        cache.insert(key(5), source());
        assert!(!cache.entries.iter().any(|entry| entry.key == key(2)));
        assert!(cache.entries.iter().any(|entry| entry.key == key(1)));
        cache.insert(key(1), source());
        assert_eq!(
            cache.entries.len(),
            3,
            "replacing a key must not evict another image"
        );
        cache.insert(
            key(6),
            SourceImage {
                pixels: Arc::new(ImageBuf::new(20, 20)),
                ..source()
            },
        );
        assert!(cache.bytes() <= cache.budget);
        assert!(!cache.entries.iter().any(|entry| entry.key == key(6)));
        assert_eq!(
            cache.entries.len(),
            3,
            "an oversized preload must not flush useful cached originals"
        );
        cache.budget = 0;
        cache.trim();
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.entries.front().unwrap().key, key(0));
        cache.forget_current();
        assert!(
            cache.entries.is_empty(),
            "a closed photo must no longer bypass the budget"
        );
    }

    #[test]
    fn reframing_does_not_mark_old_pixels_as_a_fresh_decode_of_a_changed_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let photo = photos(dir.path(), 1).remove(0);
        let mut app = app(Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend));
        app.load_image_with_state(&photo.path, 0, None, None);
        finish_load(&mut app);
        let original_stamp = image_stamp(&photo.path).unwrap();
        std::fs::OpenOptions::new()
            .append(true)
            .open(&photo.path)
            .unwrap()
            .write_all(b"new version")
            .unwrap();
        app.rotate_input(1);
        finish_load(&mut app);
        let options = app.export_raw_options().clone();
        let rotated = app.preload.cached(&photo.path, 1, &options).unwrap();
        assert_eq!(
            rotated.stamp,
            Some(original_stamp),
            "rotating old pixels must preserve their source fingerprint"
        );
        app.load_image_with_state(&photo.path, 1, None, None);
        finish_load(&mut app);
        assert!(!Arc::ptr_eq(
            app.uncropped.as_ref().unwrap(),
            &rotated.pixels
        ));
    }

    #[test]
    fn preloads_saved_raw_settings_geometry_and_rotation_then_invalidates_changed_files() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let mut photos = photos(dir.path(), 5);
        photos[3].geometry.as_mut().unwrap().orientation = 6;
        photos[3].geometry.as_mut().unwrap().crop = [0.25, 0., 1., 0.5];
        let catalog_path = dir.path().join("test.lrcat");
        let mut app = app(Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend));
        app.edit_store = Some(edits::EditStore::open(&dir.path().join("edits.sqlite")).unwrap());
        let mut saved = edits::PhotoEdit::default();
        saved.rotation = 1;
        saved.raw.white_balance = "custom".into();
        saved.raw.tint = 1.4;
        app.edit_store
            .as_ref()
            .unwrap()
            .save(&catalog_path, 3, &saved, 0)
            .unwrap();
        app.catalog = Some(catalog_ui::Browser::test_catalog(
            catalog_path,
            photos.clone(),
            2,
            &app.ctx,
            &app.scheduler,
        ));
        select(&mut app, &photos[2]);
        finish_load(&mut app);
        app.dirty = false;
        app.update_preloading();
        assert_eq!(
            app.preload.job.as_ref().unwrap().task.priority(),
            Priority::Background
        );
        finish_window(&mut app);
        let mut options = saved.raw.clone();
        options.lightroom_geometry = photos[3].geometry.clone().ok();
        let cached = app.preload.cached(&photos[3].path, 1, &options).unwrap();
        select(&mut app, &photos[3]);
        finish_load(&mut app);
        // The preloaded whole frame is used; Lightroom's crop is applied to it.
        assert!(Arc::ptr_eq(app.uncropped.as_ref().unwrap(), &cached.pixels));
        let expected = spektrafilm_io::load(&photos[3].path, &options)
            .unwrap()
            .pixels
            .rotated_quarter_turns(1);
        let image = app.image.as_ref().unwrap();
        assert_eq!(
            (image.width, image.height),
            (expected.width, expected.height)
        );
        assert!(
            image
                .data
                .iter()
                .zip(&expected.data)
                .all(|(a, b)| (*a as f64 - *b as f64).abs() < 1e-5)
        );

        // Appending bytes changes the fingerprint without changing this TIFF's pixels.
        std::fs::OpenOptions::new()
            .append(true)
            .open(&photos[3].path)
            .unwrap()
            .write_all(b"changed")
            .unwrap();
        select(&mut app, &photos[3]);
        finish_load(&mut app);
        assert!(!Arc::ptr_eq(app.image.as_ref().unwrap(), &cached.pixels));
        let previous = app.raw_options.clone();
        app.raw_options.tint = 1.7;
        app.reload_raw_if_changed(&previous);
        finish_load(&mut app);
        assert_eq!(app.export_raw_options().tint, 1.7);
        assert!(!Arc::ptr_eq(app.image.as_ref().unwrap(), &cached.pixels));
        let last_good = app.image.as_ref().unwrap().clone();
        std::fs::remove_file(&photos[3].path).unwrap();
        select(&mut app, &photos[3]);
        finish_load(&mut app);
        assert!(app.status.starts_with("Load error:"));
        assert!(Arc::ptr_eq(app.image.as_ref().unwrap(), &last_good));
    }

    #[test]
    fn selecting_an_active_preload_reuses_it_and_later_navigation_cannot_be_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let photos = photos(dir.path(), 2);
        let mut app = app(Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend));
        let task = app.scheduler.task(Priority::Interactive);
        let source = task
            .run(|| SourceImage::load(&photos[0].path, &RawOptions::default(), 0, &task))
            .unwrap();
        let request = || LoadRequest {
            path: photos[0].path.clone(),
            rotation: 0,
            raw_options: Default::default(),
            color_override: None,
            view_override: None,
            crop: None,
        };
        let key = Key {
            path: photos[0].path.clone(),
            rotation: 0,
            options: Default::default(),
        };
        let install = |app: &mut App| {
            let (tx, rx) = mpsc::channel();
            app.preload.job = Some(Job {
                key: key.clone(),
                task: app.scheduler.task(Priority::Background),
                started: Arc::new(AtomicBool::new(true)),
                rx,
                handle: None,
            });
            tx
        };
        let tx = install(&mut app);
        app.start_image_load(request());
        assert!(app.load_job.is_none() && app.preload.waiting.is_some());
        tx.send(Ok(source.clone())).ok().unwrap();
        finish_load(&mut app);
        assert!(Arc::ptr_eq(app.image.as_ref().unwrap(), &source.pixels));

        app.preload.entries.clear();
        let tx = install(&mut app);
        app.start_image_load(request());
        app.load_image_with_state(&photos[1].path, 0, None, None);
        tx.send(Ok(source.clone())).ok().unwrap();
        finish_load(&mut app);
        assert_eq!(app.image_path.as_ref(), Some(&photos[1].path));

        app.preload.entries.clear();
        let tx = install(&mut app);
        app.start_image_load(request());
        let previous = app.raw_options.clone();
        app.raw_options.white_balance = "custom".into();
        app.raw_options.tint = 1.2;
        app.reload_raw_if_changed(&previous);
        assert!(app.preload.waiting.is_none());
        tx.send(Ok(source.clone())).ok().unwrap();
        finish_load(&mut app);
        assert_eq!(app.export_raw_options().tint, 1.2);
        assert!(!Arc::ptr_eq(app.image.as_ref().unwrap(), &source.pixels));

        app.preload.entries.clear();
        let tx = install(&mut app);
        app.start_image_load(request());
        let mut invalid = photos[1].clone();
        invalid.geometry = Err("invalid catalog crop".into());
        app.select_catalog_photo(invalid);
        tx.send(Ok(source)).ok().unwrap();
        finish_load(&mut app);
        assert!(app.image.is_none() && app.preload.waiting.is_none());
    }

    #[test]
    fn queued_preloads_do_not_block_foreground_and_disabling_cancels_them() {
        let dir = tempfile::tempdir().unwrap();
        let photos = photos(dir.path(), 3);
        let mut app = app(Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend));
        app.catalog = Some(catalog_ui::Browser::test_catalog(
            dir.path().join("test.lrcat"),
            photos.clone(),
            1,
            &app.ctx,
            &app.scheduler,
        ));
        select(&mut app, &photos[1]);
        finish_load(&mut app);
        app.auto_preview = false;
        assert!(app.dirty);
        let foreground = app.scheduler.task(Priority::Interactive).cpu().unwrap();
        let ctx = app.ctx.clone();
        let mut frame = eframe::Frame::_new_kittest();
        let _ = ctx.run(Default::default(), |ctx| {
            eframe::App::update(&mut app, ctx, &mut frame)
        });
        assert!(
            !app.preload
                .job
                .as_ref()
                .unwrap()
                .started
                .load(Ordering::Acquire)
        );
        select(&mut app, &photos[2]);
        assert!(app.load_job.is_some() && app.preload.waiting.is_none());
        assert!(app.preload.job.as_ref().unwrap().task.check().is_err());
        drop(foreground);
        finish_load(&mut app);
        app.preload_radius = 0;
        app.dirty = false;
        app.update_preloading();
        assert!(app.preload.pending.is_empty() && app.preload.wanted.is_empty());
        app.preload.stop();
        assert!(app.preload.waiting.is_none());
    }

    #[test]
    #[ignore = "release benchmark requiring SPEKTRAFILM_BENCH_LIGHTROOM_CATALOG"]
    fn benchmark_preloaded_navigation() {
        let path = std::env::var_os("SPEKTRAFILM_BENCH_LIGHTROOM_CATALOG").expect("catalog path");
        let catalog = spektrafilm_io::lightroom::Catalog::open(Path::new(&path)).unwrap();
        let mut photos: Vec<Photo> = catalog
            .photos
            .into_iter()
            .filter(|photo| photo.available && spektrafilm_io::is_raw(&photo.path))
            .take(5)
            .collect();
        assert_eq!(photos.len(), 5);
        for (id, photo) in photos.iter_mut().enumerate() {
            photo.id = id as i64;
        }
        let backend: Arc<dyn ComputeBackend> = Arc::from(spektrafilm_gpu::select_backend());
        let mut app = app(backend);
        app.catalog = Some(catalog_ui::Browser::test_catalog(
            PathBuf::from(path),
            photos.clone(),
            2,
            &app.ctx,
            &app.scheduler,
        ));
        select(&mut app, &photos[2]);
        finish_load(&mut app);
        let start = Instant::now();
        finish_window(&mut app);
        println!(
            "PRELOAD window_ms={:.3} cached_bytes={}",
            start.elapsed().as_secs_f64() * 1000.,
            app.preload.bytes()
        );
        let warmed: Vec<_> = app
            .preload
            .entries
            .iter()
            .filter(|entry| Some(&entry.key) != app.preload.current.as_ref())
            .map(|entry| (entry.key.path.clone(), entry.source.pixels.clone()))
            .collect();
        for round in 0..3 {
            for (path, pixels) in &warmed {
                let photo = photos.iter().find(|photo| &photo.path == path).unwrap();
                let start = Instant::now();
                select(&mut app, photo);
                finish_load(&mut app);
                assert!(Arc::ptr_eq(app.image.as_ref().unwrap(), pixels));
                println!(
                    "NAV warm=true round={round} file={} source_ms={:.3}",
                    path.file_name().unwrap().to_string_lossy(),
                    start.elapsed().as_secs_f64() * 1000.
                );
            }
        }
        app.preload.stop();
        for photo in photos.iter().filter(|photo| photo.id != 2) {
            app.preload.entries.clear();
            let start = Instant::now();
            select(&mut app, photo);
            finish_load(&mut app);
            println!(
                "NAV warm=false file={} source_ms={:.3}",
                photo.path.file_name().unwrap().to_string_lossy(),
                start.elapsed().as_secs_f64() * 1000.
            );
        }
    }
}
