//! Edited reel previews. Disk caching is independent of the authoritative photo edits.
use super::*;
use crate::edits::PhotoEdit;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Cursor;

const DISK_LIMIT: u64 = 256 * 1024 * 1024;
const IO_CAPACITY: usize = 16;
const CHECK_INTERVAL: Duration = Duration::from_secs(3);
const THUMBNAIL_PX: u32 = spektrafilm_io::lightroom::THUMBNAIL_PX;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct PhotoKey {
    pub catalog: PathBuf,
    pub id: i64,
}

#[derive(Clone)]
pub(super) struct Spec {
    pub photo: PhotoKey,
    pub path: PathBuf,
    pub edit: PhotoEdit,
    pub detect_input_color: bool,
    pub data: PathBuf,
    pub build: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct FrameInfo {
    pub size: [u32; 2],
    pub color: String,
    pub encoded: bool,
    pub stamp: Option<(u64, std::time::SystemTime)>,
}

pub(super) struct Rendered {
    pub spec: Spec,
    pub frame: FrameInfo,
    pub fingerprint: String,
    pub quality: u32,
    pub image: image::RgbaImage,
}

impl Spec {
    fn identity(&self) -> String {
        serde_json::json!({"path":self.path,"edit":self.edit,"auto_input":self.detect_input_color,"data":self.data,"build":self.build}).to_string()
    }

    fn resolve(&self, frame: &FrameInfo) -> (RuntimeParams, Crop) {
        let mut params = self.edit.params.clone();
        if self.detect_input_color {
            params.io.input_color_space = frame.color.clone();
            params.io.input_cctf_decoding = frame.encoded;
        }
        let crop = self.edit.crop.unwrap_or_else(|| {
            Crop::from_catalog(
                self.edit.raw.lightroom_geometry.as_ref(),
                frame.size[0],
                frame.size[1],
                self.edit.rotation,
            )
        });
        params.camera.film_format_mm *=
            crop.long_side_fraction(frame.size[0] as f64, frame.size[1] as f64) as f32;
        (params, crop)
    }

    fn lens_sources(&self) -> Vec<(PathBuf, Option<(u64, std::time::SystemTime)>)> {
        let mut sources = vec![];
        if self.edit.raw.lens_correction {
            if let Ok(Some(directory)) =
                spektrafilm_io::lens_database::discover(self.edit.raw.lens_database.as_deref())
            {
                if let Ok(entries) = std::fs::read_dir(&directory) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        sources.push((path.clone(), image_stamp(&path)));
                    }
                }
                sources.push((directory.clone(), image_stamp(&directory)));
            }
        }
        sources.sort_by(|a, b| a.0.cmp(&b.0));
        sources
    }

    pub fn fingerprint(&self, frame: &FrameInfo) -> String {
        let (params, crop) = self.resolve(frame);
        let mut raw = self.edit.raw.clone();
        raw.lightroom_geometry = raw.lightroom_geometry.map(|g| g.orientation_only());
        serde_json::json!({
            "version":2,"build":self.build,"source":self.path,"stamp":frame.stamp,
            "size":frame.size,"film":self.edit.film,"paper":self.edit.paper,
            "params":params,"raw":raw,"rotation":self.edit.rotation,"crop":crop,
            "profiles":preview_pipeline_cache_key(&self.data,&self.edit.film,&self.edit.paper,&params),
            "lenses":self.lens_sources(),
        }).to_string()
    }

    fn revision(&self) -> String {
        format!(
            "{}:{:?}:{}:{:?}",
            self.identity(),
            image_stamp(&self.path),
            preview_pipeline_cache_key(
                &self.data,
                &self.edit.film,
                &self.edit.paper,
                &self.edit.params
            ),
            self.lens_sources()
        )
    }
}

pub(super) fn shrink(preview: &PreviewTextureData) -> image::RgbaImage {
    let image = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
        preview.size[0] as u32,
        preview.size[1] as u32,
        preview.rgba.as_slice(),
    )
    .expect("valid preview dimensions");
    let (width, height) = image.dimensions();
    let scale = (THUMBNAIL_PX as f32 / width.max(height) as f32).min(1.);
    image::imageops::thumbnail(
        &image,
        (width as f32 * scale).round().max(1.) as u32,
        (height as f32 * scale).round().max(1.) as u32,
    )
}

struct Probe {
    spec: Spec,
    ticket: u64,
    memory_key: Option<String>,
    frame: Option<FrameInfo>,
}

enum Command {
    Probe(Probe),
    Store(Rendered),
    Stop(mpsc::Sender<()>),
}

struct Reply {
    photo: PhotoKey,
    ticket: u64,
    revision: String,
    hit: Option<(String, u32, Option<image::RgbaImage>)>,
}

struct IoWorker {
    tx: mpsc::SyncSender<Command>,
    rx: mpsc::Receiver<Reply>,
}

impl IoWorker {
    fn start(ctx: egui::Context, path: PathBuf) -> Self {
        let (tx, commands) = mpsc::sync_channel(IO_CAPACITY);
        let (results, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name("spektrafilm-thumbnail-cache".into())
            .spawn(move || {
                let mut cache = match DiskCache::open_recovering(&path, DISK_LIMIT) {
                    Ok(cache) => Some(cache),
                    Err(error) => {
                        eprintln!("[spektrafilm] thumbnail cache: {error:#}");
                        None
                    }
                };
                let mut reported_error = false;
                while let Ok(command) = commands.recv() {
                    match command {
                        Command::Probe(probe) => {
                            let revision = probe.spec.revision();
                            let memory_hit = probe.frame.as_ref().and_then(|frame| {
                                let mut frame = frame.clone();
                                frame.stamp = image_stamp(&probe.spec.path);
                                let key = probe.spec.fingerprint(&frame);
                                (frame.stamp.is_some() && probe.memory_key.as_ref() == Some(&key))
                                    .then_some((key, 0, None))
                            });
                            if let (Some((key, _, _)), Some(cache)) = (&memory_hit, &cache) {
                                let _ = cache.touch(&probe.spec.photo, key);
                            }
                            let hit = memory_hit.or_else(|| {
                                cache.as_ref().and_then(|cache| {
                                    cache
                                        .get(&probe.spec, probe.memory_key.as_deref())
                                        .ok()
                                        .flatten()
                                })
                            });
                            if results
                                .send(Reply {
                                    photo: probe.spec.photo,
                                    ticket: probe.ticket,
                                    revision,
                                    hit,
                                })
                                .is_err()
                            {
                                break;
                            }
                            ctx.request_repaint();
                        }
                        Command::Store(image) => {
                            if let Some(cache) = &mut cache {
                                if let Err(error) = cache.put(&image) {
                                    if !reported_error {
                                        eprintln!("[spektrafilm] thumbnail cache write: {error:#}");
                                        reported_error = true;
                                    }
                                }
                            }
                        }
                        Command::Stop(done) => {
                            let _ = done.send(());
                            break;
                        }
                    }
                }
            })
            .expect("thumbnail cache worker");
        Self { tx, rx }
    }
}

struct Wanted {
    spec: Spec,
    identity: String,
    frame: Option<FrameInfo>,
    ticket: u64,
    checked: Instant,
    inflight: bool,
    rendering: bool,
    current: bool,
    edited: bool,
    /// A missing texture after a successful result should bypass the polling delay.
    had_texture: bool,
    failed_revision: Option<String>,
}

struct Pending {
    photo: PhotoKey,
    ticket: u64,
    revision: String,
}
struct RenderJob {
    pending: Pending,
    task: Task,
    rx: mpsc::Receiver<Result<Rendered, String>>,
}

pub(super) struct Manager {
    io: Option<IoWorker>,
    context: Option<(PathBuf, u64)>,
    wanted: HashMap<PhotoKey, Wanted>,
    snapshot_failures: HashMap<PhotoKey, Instant>,
    pending: VecDeque<Pending>,
    writes: VecDeque<Rendered>,
    render: Option<RenderJob>,
    serial: u64,
    pub build: String,
}

impl Default for Manager {
    fn default() -> Self {
        Self {
            io: None,
            context: None,
            wanted: HashMap::new(),
            snapshot_failures: HashMap::new(),
            pending: VecDeque::new(),
            writes: VecDeque::new(),
            render: None,
            serial: 0,
            build: format!(
                "{}:{:?}",
                env!("CARGO_PKG_VERSION"),
                std::env::current_exe().ok().and_then(|p| image_stamp(&p))
            ),
        }
    }
}

impl Drop for Manager {
    fn drop(&mut self) {
        if let Some(job) = &self.render {
            job.task.cancel();
        }
    }
}

impl Manager {
    fn ensure_io(&mut self, ctx: &egui::Context) {
        if self.io.is_none() {
            self.io = Some(IoWorker::start(
                ctx.clone(),
                edits::EditStore::default_path().with_file_name("previews.sqlite"),
            ));
        }
    }

    fn context(&mut self, context: Option<(PathBuf, u64)>) {
        if self.context == context {
            return;
        }
        if let Some(job) = &self.render {
            job.task.cancel();
        }
        self.context = context;
        self.wanted.clear();
        self.snapshot_failures.clear();
        self.pending.clear();
    }

    fn needs_snapshot(&self, photo: &PhotoKey, has_texture: bool) -> bool {
        if self
            .snapshot_failures
            .get(photo)
            .is_some_and(|at| at.elapsed() < CHECK_INTERVAL)
        {
            return false;
        }
        self.wanted.get(photo).is_none_or(|w| {
            !w.inflight
                && !w.rendering
                && ((!has_texture && w.had_texture) || w.checked.elapsed() >= CHECK_INTERVAL)
        })
    }

    fn observe(
        &mut self,
        spec: Spec,
        frame: Option<FrameInfo>,
        current: bool,
        edited: bool,
        memory: Option<String>,
        ctx: &egui::Context,
    ) -> bool {
        self.ensure_io(ctx);
        let identity = format!("{}:{:?}:{current}", spec.identity(), frame);
        let photo = spec.photo.clone();
        if self
            .wanted
            .get(&photo)
            .is_none_or(|w| w.identity != identity)
        {
            self.serial = self.serial.wrapping_add(1);
            self.wanted.insert(
                photo.clone(),
                Wanted {
                    spec,
                    frame,
                    identity,
                    ticket: self.serial,
                    checked: Instant::now()
                        .checked_sub(CHECK_INTERVAL)
                        .unwrap_or_else(Instant::now),
                    inflight: false,
                    rendering: false,
                    current,
                    edited,
                    had_texture: false,
                    failed_revision: None,
                },
            );
        }
        let wanted = self.wanted.get_mut(&photo).unwrap();
        if wanted.inflight
            || wanted.rendering
            || (wanted.checked.elapsed() < CHECK_INTERVAL
                && (!wanted.had_texture || memory.is_some()))
        {
            return false;
        }
        let probe = Probe {
            spec: wanted.spec.clone(),
            ticket: wanted.ticket,
            memory_key: memory,
            frame: wanted.frame.clone(),
        };
        if self
            .io
            .as_ref()
            .unwrap()
            .tx
            .try_send(Command::Probe(probe))
            .is_ok()
        {
            wanted.inflight = true;
            wanted.checked = Instant::now();
            true
        } else {
            false
        }
    }

    fn enqueue_write(&mut self, image: Rendered) {
        self.writes.retain(|old| old.spec.photo != image.spec.photo);
        self.writes.push_back(image);
        if self.writes.len() > 256 {
            self.writes.pop_front();
        }
    }

    fn flush_writes(&mut self) {
        let Some(io) = &self.io else { return };
        while let Some(image) = self.writes.pop_front() {
            match io.tx.try_send(Command::Store(image)) {
                Ok(()) => {}
                Err(mpsc::TrySendError::Full(Command::Store(image))) => {
                    self.writes.push_front(image);
                    break;
                }
                Err(_) => break,
            }
        }
    }

    fn foreground(
        &mut self,
        result: Rendered,
        persist: bool,
        browser: &mut catalog_ui::Browser,
        ctx: &egui::Context,
    ) {
        self.ensure_io(ctx);
        if browser.path() != result.spec.photo.catalog {
            return;
        }
        browser.rendered_thumbnail(
            ctx,
            result.spec.photo.id,
            result.fingerprint.clone(),
            result.quality,
            &result.image,
        );
        self.serial = self.serial.wrapping_add(1);
        let identity = format!("{}:{:?}:true", result.spec.identity(), Some(&result.frame));
        self.wanted.insert(
            result.spec.photo.clone(),
            Wanted {
                spec: result.spec.clone(),
                frame: Some(result.frame.clone()),
                identity,
                ticket: self.serial,
                checked: Instant::now(),
                inflight: false,
                rendering: false,
                current: true,
                edited: true,
                had_texture: true,
                failed_revision: None,
            },
        );
        if persist {
            self.enqueue_write(result);
            self.flush_writes();
        }
    }

    fn poll(&mut self, browser: &mut catalog_ui::Browser, ctx: &egui::Context) {
        self.flush_writes();
        if let Some(io) = &self.io {
            while let Ok(reply) = io.rx.try_recv() {
                let Some(wanted) = self
                    .wanted
                    .get_mut(&reply.photo)
                    .filter(|w| w.ticket == reply.ticket)
                else {
                    continue;
                };
                wanted.inflight = false;
                ctx.request_repaint();
                if reply.hit.is_none() {
                    wanted.had_texture = false;
                    browser.allow_source_fallback(reply.photo.id);
                }
                match reply.hit {
                    Some((key, quality, image)) => {
                        if let Some(image) = image {
                            browser.rendered_thumbnail(ctx, reply.photo.id, key, quality, &image);
                        } else if browser.rendered_key(reply.photo.id) != Some(key.as_str()) {
                            // The cache lookup raced with texture eviction. Request pixels next frame.
                            wanted.checked = Instant::now() - CHECK_INTERVAL;
                            continue;
                        }
                        wanted.had_texture = true;
                        browser.preview_status(reply.photo.id, None);
                        wanted.failed_revision = None;
                    }
                    None if wanted.failed_revision.as_ref() == Some(&reply.revision) => {}
                    None => {
                        browser.preview_status(
                            reply.photo.id,
                            Some("Updating edited preview…".into()),
                        );
                        if !wanted.current {
                            wanted.rendering = true;
                            self.pending.push_back(Pending {
                                photo: reply.photo,
                                ticket: reply.ticket,
                                revision: reply.revision,
                            });
                        }
                    }
                }
            }
        }
        if let Some(job) = &self.render {
            let valid = self
                .wanted
                .get(&job.pending.photo)
                .is_some_and(|w| w.ticket == job.pending.ticket && !w.current);
            if !valid {
                job.task.cancel();
            }
            match job.rx.try_recv() {
                Err(mpsc::TryRecvError::Empty) => {}
                result => {
                    ctx.request_repaint();
                    let job = self.render.take().unwrap();
                    if let Some(wanted) = self
                        .wanted
                        .get_mut(&job.pending.photo)
                        .filter(|w| w.ticket == job.pending.ticket && !w.current)
                    {
                        wanted.rendering = false;
                        wanted.checked = Instant::now();
                        match result.unwrap_or_else(|_| Err("Thumbnail worker stopped".into())) {
                            Ok(image) => {
                                wanted.frame = Some(image.frame.clone());
                                wanted.had_texture = true;
                                browser.rendered_thumbnail(
                                    ctx,
                                    image.spec.photo.id,
                                    image.fingerprint.clone(),
                                    image.quality,
                                    &image.image,
                                );
                                self.enqueue_write(image);
                            }
                            Err(error) => {
                                wanted.failed_revision = Some(job.pending.revision);
                                browser.preview_status(
                                    job.pending.photo.id,
                                    Some(format!("Preview couldn't be updated: {error}")),
                                );
                            }
                        }
                    }
                }
            }
        }
        self.flush_writes();
    }

    fn start_render(
        &mut self,
        scheduler: &Scheduler,
        backend: Arc<dyn ComputeBackend>,
        preload: &mut preload::Preloader,
        ctx: &egui::Context,
        busy: bool,
    ) {
        self.pending.retain(|p| {
            self.wanted
                .get(&p.photo)
                .is_some_and(|w| w.ticket == p.ticket && !w.current)
        });
        if self.render.is_some() || busy {
            return;
        }
        let position = self
            .pending
            .iter()
            .position(|p| self.wanted.get(&p.photo).is_some_and(|w| w.edited))
            .unwrap_or(0);
        let Some(pending) = self.pending.remove(position) else {
            return;
        };
        let spec = self.wanted[&pending.photo].spec.clone();
        let cached = preload.cached(&spec.path, spec.edit.rotation, &spec.edit.raw);
        let task = scheduler.task(Priority::Background);
        let worker_task = task.clone();
        let repaint = ctx.clone();
        let (tx, rx) = mpsc::channel();
        let result = std::thread::Builder::new()
            .name("spektrafilm-edited-thumbnail".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    worker_task.run(|| render(&spec, cached, &worker_task, backend.as_ref()))
                }))
                .unwrap_or_else(|panic| Err(anyhow::anyhow!(panic_message(panic))))
                .map_err(|e| format!("{e:#}"));
                let _ = tx.send(result);
                repaint.request_repaint();
            });
        if let Err(error) = result {
            if let Some(wanted) = self.wanted.get_mut(&pending.photo) {
                wanted.rendering = false;
                wanted.failed_revision = Some(pending.revision);
            }
            eprintln!("[spektrafilm] thumbnail worker: {error}");
        } else {
            self.render = Some(RenderJob { pending, task, rx });
        }
    }

    pub fn shutdown(&mut self) {
        if let Some(job) = &self.render {
            job.task.cancel();
        }
        // Cache writes are best effort; a slow/unplugged source must not hold up exit.
        self.flush_writes();
        if let Some(io) = self.io.take() {
            let (tx, rx) = mpsc::channel();
            if io.tx.try_send(Command::Stop(tx)).is_ok() {
                let _ = rx.recv_timeout(Duration::from_millis(500));
            }
        }
    }
}

impl App {
    pub(super) fn thumbnail_frame(&self) -> Option<FrameInfo> {
        let frame = self.uncropped.as_ref()?;
        let (color, encoded) = self.native_input_color.as_ref()?;
        Some(FrameInfo {
            size: [frame.width, frame.height],
            color: color.clone(),
            encoded: *encoded,
            stamp: self.source_stamp,
        })
    }

    pub(super) fn current_thumbnail_spec(&self) -> Option<Spec> {
        if self.image_loading() || self.crop_tool.is_some() {
            return None;
        }
        let (catalog, id) = self.current_photo.as_ref()?;
        let path = self.image_path.as_ref()?;
        let mut edit = self.current_edit();
        edit.params.settings.preview_max_size = self.preview_max_size;
        edit.raw = self.export_raw_options().clone();
        edit.crop = Some(self.applied_crop);
        Some(Spec {
            photo: PhotoKey {
                catalog: catalog.clone(),
                id: *id,
            },
            path: path.clone(),
            edit,
            detect_input_color: false,
            data: self.data_dir.clone(),
            build: self.thumbnails.build.clone(),
        })
    }

    fn observed_thumbnail_spec(&self) -> Option<Spec> {
        if let Some(spec) = self.current_thumbnail_spec() {
            return Some(spec);
        }
        let (catalog, id) = self.current_photo.as_ref()?;
        let photo = self.catalog.as_ref()?.active_photo()?;
        if photo.id != *id {
            return None;
        }
        let mut edit = self.current_edit();
        edit.params.settings.preview_max_size = self.preview_max_size;
        edit.raw = self.raw_options.clone();
        edit.raw.lightroom_geometry = Some(photo.geometry.ok()?);
        let incoming = self
            .pending_load
            .as_ref()
            .or_else(|| self.preload.waiting())
            .filter(|request| request.path == photo.path)
            .map(|request| (request.rotation, request.crop))
            .or_else(|| {
                self.load_job
                    .as_ref()
                    .filter(|job| !job.discard && job.path == photo.path)
                    .map(|job| (job.rotation, job.crop))
            });
        if let Some((rotation, crop)) = incoming {
            edit.crop = if self.crop_tool.is_some() {
                edit.crop
                    .map(|crop| crop.rotated_quarter_turns((rotation + 4 - self.rotation) % 4))
            } else {
                crop
            };
            edit.rotation = rotation;
        }
        Some(Spec {
            photo: PhotoKey {
                catalog: catalog.clone(),
                id: *id,
            },
            path: photo.path,
            edit,
            detect_input_color: self.image_loading() || self.native_input_color.is_none(),
            data: self.data_dir.clone(),
            build: self.thumbnails.build.clone(),
        })
    }

    pub(super) fn accept_rendered_thumbnail(
        &mut self,
        result: Rendered,
        persist: bool,
        ctx: &egui::Context,
    ) {
        let Some(current) = self.current_thumbnail_spec() else {
            return;
        };
        if current.photo != result.spec.photo
            || current.identity() != result.spec.identity()
            || self.source_stamp != result.frame.stamp
        {
            return;
        }
        if let Some(browser) = &mut self.catalog {
            self.thumbnails.foreground(result, persist, browser, ctx);
        }
    }

    pub(super) fn update_edited_thumbnails(&mut self, ctx: &egui::Context) {
        let context = self
            .catalog
            .as_ref()
            .map(catalog_ui::Browser::preview_context);
        self.thumbnails.context(context.clone());
        let Some((catalog, _)) = context else { return };
        let photos = self.catalog.as_ref().unwrap().viewport_photos();
        let active = self
            .current_photo
            .as_ref()
            .filter(|(path, _)| *path == catalog)
            .map(|(_, id)| *id);
        let mut keep: HashSet<_> = photos
            .iter()
            .map(|p| PhotoKey {
                catalog: catalog.clone(),
                id: p.id,
            })
            .collect();
        if let Some(id) = active {
            keep.insert(PhotoKey {
                catalog: catalog.clone(),
                id,
            });
        }
        self.thumbnails
            .wanted
            .retain(|photo, _| keep.contains(photo));
        self.thumbnails
            .snapshot_failures
            .retain(|photo, _| keep.contains(photo));
        if let Some(spec) = self.observed_thumbnail_spec() {
            let memory = self
                .catalog
                .as_ref()
                .unwrap()
                .rendered_key(spec.photo.id)
                .map(str::to_owned);
            let frame = if self.image_loading() {
                None
            } else {
                self.thumbnail_frame()
            };
            self.thumbnails
                .observe(spec, frame, true, true, memory, ctx);
        } else if let Some(id) = active {
            self.thumbnails.wanted.remove(&PhotoKey {
                catalog: catalog.clone(),
                id,
            });
        }
        let mut defaults = None;
        for photo in photos {
            let key = PhotoKey {
                catalog: catalog.clone(),
                id: photo.id,
            };
            if active == Some(photo.id) {
                if self.image_loading() {
                    self.catalog
                        .as_mut()
                        .unwrap()
                        .preview_status(photo.id, Some("Waiting for the current edit…".into()));
                }
                continue;
            }
            let has_texture = self
                .catalog
                .as_ref()
                .unwrap()
                .rendered_key(photo.id)
                .is_some();
            if !self.thumbnails.needs_snapshot(&key, has_texture) {
                continue;
            }
            let edit = self
                .edit_store()
                .ok_or_else(|| "Photo edits are unavailable".to_owned())
                .and_then(|store| {
                    store
                        .load(&catalog, photo.id)
                        .map_err(|error| format!("{error:#}"))
                })
                .map(|edit| {
                    edit.unwrap_or_else(|| {
                        defaults.get_or_insert_with(|| self.default_edit()).clone()
                    })
                });
            let spec = edit.and_then(|mut edit| {
                edit.raw.lightroom_geometry = Some(photo.geometry.clone()?);
                edit.params.io.crop = false;
                edit.params.settings.preview_max_size = self.preview_max_size;
                Ok(Spec {
                    photo: key.clone(),
                    path: photo.path.clone(),
                    edit,
                    detect_input_color: true,
                    data: self.data_dir.clone(),
                    build: self.thumbnails.build.clone(),
                })
            });
            match spec {
                Ok(spec) => {
                    let browser = self.catalog.as_ref().unwrap();
                    let memory = browser.rendered_key(photo.id).map(str::to_owned);
                    let edited = browser.edited.contains(&photo.id)
                        || photo
                            .geometry
                            .as_ref()
                            .is_ok_and(|g| g.crop != [0., 0., 1., 1.] || g.angle != 0.);
                    self.thumbnails
                        .observe(spec, None, false, edited, memory, ctx);
                    self.thumbnails.snapshot_failures.remove(&key);
                }
                Err(error) => {
                    self.catalog
                        .as_mut()
                        .unwrap()
                        .allow_source_fallback(photo.id);
                    self.thumbnails
                        .snapshot_failures
                        .insert(key, Instant::now());
                    self.catalog.as_mut().unwrap().preview_status(
                        photo.id,
                        Some(format!("Preview couldn't be updated: {error}")),
                    );
                }
            }
        }
        self.thumbnails.poll(self.catalog.as_mut().unwrap(), ctx);
        if let Some(id) = active
            && !self.image_loading()
            && self.image.is_none()
        {
            let browser = self.catalog.as_mut().unwrap();
            browser.allow_source_fallback(id);
            browser.preview_status(
                id,
                Some(format!("Edited preview unavailable: {}", self.status)),
            );
        }
        let busy = self.image_loading()
            || self.render_job.is_some()
            || ((self.dirty || self.pending_dirty) && (self.auto_preview || self.render_requested))
            || ctx.is_using_pointer();
        self.thumbnails.start_render(
            &self.scheduler,
            self.backend.clone(),
            &mut self.preload,
            ctx,
            busy,
        );
        ctx.request_repaint_after(
            if self.thumbnails.render.is_some() || !self.thumbnails.pending.is_empty() {
                Duration::from_millis(200)
            } else {
                CHECK_INTERVAL
            },
        );
    }
}

fn render(
    spec: &Spec,
    cached: Option<SourceImage>,
    task: &Task,
    backend: &dyn ComputeBackend,
) -> Result<Rendered> {
    let cpu = task.cpu()?;
    let source = match cached
        .filter(|source| source.stamp.is_some() && source.stamp == image_stamp(&spec.path))
    {
        Some(source) => source,
        None => SourceImage::load(&spec.path, &spec.edit.raw, spec.edit.rotation, task)
            .map_err(anyhow::Error::msg)?,
    };
    let frame = FrameInfo {
        size: [source.pixels.width, source.pixels.height],
        color: source.color_space,
        encoded: source.encoded,
        stamp: source.stamp,
    };
    let (params, crop) = spec.resolve(&frame);
    params.validate().map_err(anyhow::Error::msg)?;
    let image = crop_ui::cropped(&source.pixels, &crop).map_err(anyhow::Error::msg)?;
    let quality = if params.settings.preview_max_size == 0 {
        image.width.max(image.height)
    } else {
        params.settings.preview_max_size
    };
    let image = if image.width.max(image.height) > quality {
        Arc::new(spektrafilm_core::resize::preview(&image, quality))
    } else {
        image
    };
    let fingerprint = spec.fingerprint(&frame);
    let film = profile::load_profile_by_name(&spec.data, &spec.edit.film)?;
    let paper = profile::load_profile_by_name(
        &spec.data,
        if params.scans_film() {
            &spec.edit.film
        } else {
            &spec.edit.paper
        },
    )?;
    let mut pipeline =
        Pipeline::new_with_spectral(film, paper, params, &spec.data).map_err(anyhow::Error::msg)?;
    drop(cpu);
    task.check()?;
    let image = if let Some(color) = pipeline.input_denoise_color() {
        let prepared = super::denoise_preview(
            &image,
            color,
            pipeline.params.camera.chroma_noise_reduction / 100.,
            &mut Vec::new(),
            backend,
            task,
        )?;
        pipeline.params.camera.chroma_noise_reduction = 0.;
        prepared
    } else {
        image
    };
    task.check()?;
    let cpu = if backend.is_gpu() {
        None
    } else {
        Some(task.cpu()?)
    };
    let (output, _) = render_preview_image(&pipeline, &image, backend);
    drop(cpu);
    task.check()?;
    let _cpu = task.cpu()?;
    let image = shrink(&prepare_preview_texture_data(
        &output,
        &pipeline.params.io,
        true,
        backend,
    ));
    anyhow::ensure!(
        image_stamp(&spec.path) == frame.stamp && spec.fingerprint(&frame) == fingerprint,
        "Source or profiles changed while rendering"
    );
    Ok(Rendered {
        spec: spec.clone(),
        frame,
        fingerprint,
        quality,
        image,
    })
}

struct DiskCache {
    connection: Connection,
    budget: u64,
}

impl DiskCache {
    fn open_recovering(path: &Path, limit: u64) -> Result<Self> {
        match Self::open(path, limit) {
            Ok(cache) => Ok(cache),
            Err(error) => {
                let corrupt = error.chain().any(|cause| matches!(
                    cause.downcast_ref::<rusqlite::Error>(),
                    Some(rusqlite::Error::SqliteFailure(code, _)) if matches!(
                        code.code, rusqlite::ErrorCode::DatabaseCorrupt | rusqlite::ErrorCode::NotADatabase
                    )
                ));
                if !corrupt {
                    return Err(error);
                }
                // Only this disposable cache is removed; photo edits live in another database.
                std::fs::remove_file(path)?;
                Self::open(path, limit)
            }
        }
    }

    fn open(path: &Path, limit: u64) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let connection = Connection::open(path)?;
        connection.busy_timeout(Duration::from_millis(100))?;
        connection.execute_batch(
            "PRAGMA auto_vacuum=FULL; PRAGMA journal_mode=DELETE; PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS previews (
                catalog TEXT NOT NULL, photo INTEGER NOT NULL, fingerprint TEXT NOT NULL,
                frame TEXT NOT NULL, quality INTEGER NOT NULL, png BLOB NOT NULL,
                bytes INTEGER NOT NULL, used INTEGER NOT NULL, PRIMARY KEY(catalog, photo));
            CREATE INDEX IF NOT EXISTS previews_lru ON previews(used);",
        )?;
        let page_size: u64 = connection.pragma_query_value(None, "page_size", |r| r.get(0))?;
        connection.pragma_update(None, "max_page_count", (limit / page_size).max(8))?;
        Ok(Self {
            connection,
            budget: limit * 3 / 4,
        })
    }

    fn remove(&self, photo: &PhotoKey) -> Result<()> {
        self.connection.execute(
            "DELETE FROM previews WHERE catalog=?1 AND photo=?2",
            params![photo.catalog.to_string_lossy(), photo.id],
        )?;
        Ok(())
    }

    fn touch(&self, photo: &PhotoKey, key: &str) -> Result<()> {
        self.connection.execute(
            "UPDATE previews SET used=(SELECT COALESCE(MAX(used),0)+1 FROM previews)
             WHERE catalog=?1 AND photo=?2 AND fingerprint=?3",
            params![photo.catalog.to_string_lossy(), photo.id, key],
        )?;
        Ok(())
    }

    fn get(
        &self,
        spec: &Spec,
        memory_key: Option<&str>,
    ) -> Result<Option<(String, u32, Option<image::RgbaImage>)>> {
        let row = self
            .connection
            .query_row(
                "SELECT fingerprint,frame,quality,png FROM previews WHERE catalog=?1 AND photo=?2",
                params![spec.photo.catalog.to_string_lossy(), spec.photo.id],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, u32>(2)?,
                        r.get::<_, Vec<u8>>(3)?,
                    ))
                },
            )
            .optional()?;
        let Some((key, frame, quality, bytes)) = row else {
            return Ok(None);
        };
        let Ok(mut frame) = serde_json::from_str::<FrameInfo>(&frame) else {
            self.remove(&spec.photo)?;
            return Ok(None);
        };
        if frame.size.contains(&0) {
            self.remove(&spec.photo)?;
            return Ok(None);
        }
        frame.stamp = image_stamp(&spec.path);
        if frame.stamp.is_none() || spec.fingerprint(&frame) != key {
            return Ok(None);
        }
        let image = if memory_key == Some(key.as_str()) {
            None
        } else {
            let decoded = (|| -> Result<image::RgbaImage> {
                anyhow::ensure!(
                    bytes.len() <= (THUMBNAIL_PX * THUMBNAIL_PX * 4) as usize + 65536,
                    "oversized thumbnail"
                );
                let decoder =
                    image::ImageReader::with_format(Cursor::new(&bytes), image::ImageFormat::Png)
                        .into_decoder()?;
                use image::ImageDecoder;
                let (w, h) = decoder.dimensions();
                anyhow::ensure!(
                    w > 0 && h > 0 && w <= THUMBNAIL_PX && h <= THUMBNAIL_PX,
                    "invalid thumbnail dimensions"
                );
                Ok(image::DynamicImage::from_decoder(decoder)?.to_rgba8())
            })();
            match decoded {
                Ok(image) => Some(image),
                Err(_) => {
                    self.remove(&spec.photo)?;
                    return Ok(None);
                }
            }
        };
        self.touch(&spec.photo, &key)?;
        Ok(Some((key, quality, image)))
    }

    fn put(&mut self, result: &Rendered) -> Result<()> {
        let mut frame = result.frame.clone();
        frame.stamp = image_stamp(&result.spec.path);
        if frame.stamp.is_none() || result.spec.fingerprint(&frame) != result.fingerprint {
            return Ok(());
        }
        let mut png = Cursor::new(Vec::new());
        result.image.write_to(&mut png, image::ImageFormat::Png)?;
        let png = png.into_inner();
        let frame = serde_json::to_string(&result.frame)?;
        let bytes = (png.len() + frame.len() + result.fingerprint.len()) as u64;
        if bytes > self.budget {
            return Ok(());
        }
        let transaction = self.connection.transaction()?;
        let existing: Option<(String, u32)> = transaction
            .query_row(
                "SELECT fingerprint,quality FROM previews WHERE catalog=?1 AND photo=?2",
                params![
                    result.spec.photo.catalog.to_string_lossy(),
                    result.spec.photo.id
                ],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        if existing
            .is_some_and(|(key, quality)| key == result.fingerprint && quality > result.quality)
        {
            return Ok(());
        }
        transaction.execute(
            "DELETE FROM previews WHERE catalog=?1 AND photo=?2",
            params![
                result.spec.photo.catalog.to_string_lossy(),
                result.spec.photo.id
            ],
        )?;
        loop {
            let used: u64 =
                transaction.query_row("SELECT COALESCE(SUM(bytes),0) FROM previews", [], |r| {
                    r.get(0)
                })?;
            if used + bytes <= self.budget {
                break;
            }
            transaction.execute("DELETE FROM previews WHERE rowid=(SELECT rowid FROM previews ORDER BY used LIMIT 1)",[])?;
        }
        transaction.execute(
            "INSERT INTO previews(catalog,photo,fingerprint,frame,quality,png,bytes,used)
            VALUES(?1,?2,?3,?4,?5,?6,?7,(SELECT COALESCE(MAX(used),0)+1 FROM previews))",
            params![
                result.spec.photo.catalog.to_string_lossy(),
                result.spec.photo.id,
                result.fingerprint,
                frame,
                result.quality,
                png,
                bytes
            ],
        )?;
        transaction.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spektrafilm_io::lightroom::{Geometry, Photo};

    #[test]
    fn shrink_preserves_preview_proportions_without_enlarging_small_images() {
        for (size, expected) in [
            ([400, 250], (160, 100)),
            ([250, 400], (100, 160)),
            ([400, 1], (160, 1)),
            ([40, 25], (40, 25)),
        ] {
            let preview = PreviewTextureData {
                size,
                rgba: vec![255; size[0] * size[1] * 4],
            };
            assert_eq!(shrink(&preview).dimensions(), expected);
        }
    }

    fn fixture(dir: &Path, id: i64) -> Rendered {
        let path = dir.join("original.png");
        if !path.exists() {
            image::RgbaImage::from_pixel(64, 40, image::Rgba([120, 80, 50, 255]))
                .save(&path)
                .unwrap();
        }
        let mut edit = PhotoEdit::default();
        edit.params.settings.preview_max_size = 64;
        edit.raw.lightroom_geometry = Some(Geometry {
            orientation: 1,
            crop: [0.25, 0., 0.75, 1.],
            angle: 0.,
        });
        let spec = Spec {
            photo: PhotoKey {
                catalog: dir.join("catalog.lrcat"),
                id,
            },
            path,
            edit,
            detect_input_color: true,
            data: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data"),
            build: "test-build".into(),
        };
        let frame = FrameInfo {
            size: [64, 40],
            color: "sRGB".into(),
            encoded: true,
            stamp: image_stamp(&spec.path),
        };
        let fingerprint = spec.fingerprint(&frame);
        Rendered {
            spec,
            frame,
            fingerprint,
            quality: 64,
            image: image::RgbaImage::from_pixel(32, 40, image::Rgba([150, 90, 65, 255])),
        }
    }

    fn noise() -> image::RgbaImage {
        image::RgbaImage::from_fn(160, 100, |x, y| {
            let mut v = (y * 160 + x + 1).wrapping_mul(0x9e3779b9);
            v ^= v >> 16;
            v = v.wrapping_mul(0x85ebca6b);
            image::Rgba([v as u8, (v >> 8) as u8, (v >> 16) as u8, 255])
        })
    }

    fn photo(spec: &Spec) -> Photo {
        Photo {
            id: spec.photo.id,
            path: spec.path.clone(),
            available: true,
            copy_name: String::new(),
            flag: 1,
            framing_warning: false,
            geometry: Ok(spec.edit.raw.lightroom_geometry.clone().unwrap()),
        }
    }

    fn browser(spec: &Spec, ctx: &egui::Context) -> catalog_ui::Browser {
        catalog_ui::Browser::test_catalog(
            spec.photo.catalog.clone(),
            vec![photo(spec)],
            0,
            ctx,
            &Scheduler::default(),
        )
    }

    fn fake_io() -> (IoWorker, mpsc::Receiver<Command>, mpsc::Sender<Reply>) {
        let (tx, commands) = mpsc::sync_channel(IO_CAPACITY);
        let (replies, rx) = mpsc::channel();
        (IoWorker { tx, rx }, commands, replies)
    }

    fn finish_load(app: &mut App) {
        while let Some(job) = &mut app.load_job {
            if let Some(handle) = job.handle.take() {
                handle.join().unwrap();
            }
            app.poll_load_job();
        }
    }

    #[test]
    fn loading_and_reframing_use_incoming_rotation_and_committed_crop() {
        let dir = tempfile::tempdir().unwrap();
        let mut result = fixture(dir.path(), 1);
        result.spec.edit.rotation = 1;
        result.spec.edit.crop = Some(Crop {
            size: [0.5, 0.75],
            ..Crop::FULL
        });
        result.spec.edit.params.io.crop = false;
        result.frame.size = [40, 64];
        result.fingerprint = result.spec.fingerprint(&result.frame);
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut app = App::new(
            &cc,
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            None,
        );
        app.data_dir = result.spec.data.clone();
        app.preview_max_size = 64;
        app.thumbnails.build = result.spec.build.clone();
        app.catalog = Some(browser(&result.spec, &ctx));
        let store = edits::EditStore::open(&dir.path().join("edits.sqlite")).unwrap();
        store
            .save(&result.spec.photo.catalog, 1, &result.spec.edit, 0)
            .unwrap();
        app.edit_store = Some(store);
        let mut cache = DiskCache::open(&dir.path().join("previews.sqlite"), DISK_LIMIT).unwrap();
        cache.put(&result).unwrap();
        app.select_catalog_photo(photo(&result.spec));
        assert!(app.image_loading());
        assert_eq!(app.rotation, 0);
        let observed = app.observed_thumbnail_spec().unwrap();
        assert_eq!(observed.edit.rotation, 1);
        assert!(
            cache.get(&observed, None).unwrap().is_some(),
            "cache must hit before the decode finishes"
        );
        finish_load(&mut app);
        assert!(
            cache
                .get(&app.current_thumbnail_spec().unwrap(), None)
                .unwrap()
                .is_some()
        );

        let crop = app.crop.unwrap().rotated_quarter_turns(1);
        app.rotate_input(1);
        let observed = app.observed_thumbnail_spec().unwrap();
        assert_eq!(observed.edit.rotation, 2);
        assert_eq!(observed.edit.crop, Some(crop));
        finish_load(&mut app);

        // The crop tool loads the whole frame, but the reel keeps the committed crop.
        app.open_crop_tool();
        app.reframe(1, Crop::FULL);
        let observed = app.observed_thumbnail_spec().unwrap();
        assert_eq!(observed.edit.rotation, 3);
        assert_eq!(observed.edit.crop, Some(crop.rotated_quarter_turns(1)));
        finish_load(&mut app);
    }

    #[test]
    fn queued_load_framing_takes_precedence_over_the_discarded_load() {
        let dir = tempfile::tempdir().unwrap();
        let result = fixture(dir.path(), 1);
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut app = App::new(
            &cc,
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            None,
        );
        app.catalog = Some(browser(&result.spec, &ctx));
        app.current_photo = Some((result.spec.photo.catalog.clone(), 1));
        app.load_image_with_state(&result.spec.path, 1, None, None);
        let crop = Crop {
            size: [0.75, 0.5],
            ..Crop::FULL
        };
        app.crop = Some(crop);
        app.load_image_with_state(&result.spec.path, 2, None, None);
        assert!(app.load_job.as_ref().unwrap().discard);
        assert!(app.pending_load.is_some());
        let observed = app.observed_thumbnail_spec().unwrap();
        assert_eq!(observed.edit.rotation, 2);
        assert_eq!(observed.edit.crop, Some(crop));
        finish_load(&mut app);
    }

    #[test]
    fn texture_eviction_reloads_pixels_immediately_before_or_after_a_cache_hit() {
        let dir = tempfile::tempdir().unwrap();
        let result = fixture(dir.path(), 1);
        for current in [false, true] {
            for evict_before_reply in [false, true] {
                let ctx = egui::Context::default();
                let mut browser = browser(&result.spec, &ctx);
                browser.rendered_thumbnail(&ctx, 1, result.fingerprint.clone(), 64, &result.image);
                let (io, commands, replies) = fake_io();
                let mut manager = Manager::default();
                manager.io = Some(io);
                let frame = current.then(|| result.frame.clone());
                assert!(manager.observe(
                    result.spec.clone(),
                    frame.clone(),
                    current,
                    true,
                    Some(result.fingerprint.clone()),
                    &ctx
                ));
                let ticket = manager.wanted[&result.spec.photo].ticket;
                assert!(matches!(commands.try_recv().unwrap(), Command::Probe(_)));
                let evict = |browser: &mut catalog_ui::Browser| {
                    for id in 2..=257 {
                        browser.rendered_thumbnail(&ctx, id, "other".into(), 64, &result.image);
                    }
                    assert!(browser.rendered_key(1).is_none());
                };
                if evict_before_reply {
                    evict(&mut browser);
                }
                replies
                    .send(Reply {
                        photo: result.spec.photo.clone(),
                        ticket,
                        revision: "same".into(),
                        hit: Some((result.fingerprint.clone(), 0, None)),
                    })
                    .unwrap();
                manager.poll(&mut browser, &ctx);
                if !evict_before_reply {
                    evict(&mut browser);
                }
                assert!(manager.needs_snapshot(&result.spec.photo, false));
                assert!(manager.observe(result.spec.clone(), frame, current, true, None, &ctx));
                let Command::Probe(probe) = commands.try_recv().unwrap() else {
                    panic!("expected pixel reload");
                };
                assert!(probe.memory_key.is_none());
                replies
                    .send(Reply {
                        photo: probe.spec.photo,
                        ticket: probe.ticket,
                        revision: "same".into(),
                        hit: Some((result.fingerprint.clone(), 64, Some(result.image.clone()))),
                    })
                    .unwrap();
                manager.poll(&mut browser, &ctx);
                assert_eq!(browser.rendered_key(1), Some(result.fingerprint.as_str()));
                assert!(!manager.needs_snapshot(&result.spec.photo, true));
            }
        }
    }

    #[test]
    fn a_cache_miss_without_a_prior_texture_does_not_spin() {
        let dir = tempfile::tempdir().unwrap();
        let result = fixture(dir.path(), 1);
        let ctx = egui::Context::default();
        let mut browser = browser(&result.spec, &ctx);
        let (io, _commands, replies) = fake_io();
        let mut manager = Manager::default();
        manager.io = Some(io);
        manager.observe(result.spec.clone(), None, true, true, None, &ctx);
        let ticket = manager.wanted[&result.spec.photo].ticket;
        replies
            .send(Reply {
                photo: result.spec.photo.clone(),
                ticket,
                revision: "same".into(),
                hit: None,
            })
            .unwrap();
        manager.poll(&mut browser, &ctx);
        assert!(!manager.needs_snapshot(&result.spec.photo, false));
        assert!(!manager.observe(result.spec, None, true, true, None, &ctx));
    }

    #[test]
    fn background_job_reuses_the_matching_preloaded_frame_without_decoding() {
        let dir = tempfile::tempdir().unwrap();
        let mut result = fixture(dir.path(), 1);
        result.spec.path = dir.path().join("unreadable.ARW");
        // Metadata is readable; reopening this original as a RAW would fail.
        std::fs::write(&result.spec.path, b"not a decodable RAW").unwrap();
        result.spec.edit.raw.decoder = "rawler".into();
        result.spec.edit.raw.white_balance_shift = [0.2, -0.1];
        result.spec.edit.rotation = 1;
        result.spec.edit.params.debug.deactivate_stochastic_effects = true;
        result.spec.edit.params.debug.deactivate_spatial_effects = true;
        let mut pixels = ImageBuf::new(10, 16);
        pixels.data.fill(0.25);
        let source = SourceImage {
            pixels: Arc::new(pixels),
            metadata: Default::default(),
            color_space: "sRGB".into(),
            encoded: true,
            stamp: image_stamp(&result.spec.path),
        };
        let mut preload = preload::Preloader::default();
        preload.remember_current(
            result.spec.path.clone(),
            result.spec.edit.rotation,
            result.spec.edit.raw.clone(),
            source,
        );
        let ctx = egui::Context::default();
        let mut browser = browser(&result.spec, &ctx);
        let (io, _commands, replies) = fake_io();
        let mut manager = Manager::default();
        manager.io = Some(io);
        manager.observe(result.spec.clone(), None, false, true, None, &ctx);
        let ticket = manager.wanted[&result.spec.photo].ticket;
        replies
            .send(Reply {
                photo: result.spec.photo.clone(),
                ticket,
                revision: "cold".into(),
                hit: None,
            })
            .unwrap();
        manager.poll(&mut browser, &ctx);
        manager.start_render(
            &Scheduler::default(),
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            &mut preload,
            &ctx,
            false,
        );
        assert!(manager.render.is_some());
        let deadline = Instant::now() + Duration::from_secs(10);
        while manager.render.is_some() && Instant::now() < deadline {
            manager.poll(&mut browser, &ctx);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(manager.render.is_none(), "background render did not finish");
        assert!(
            manager.wanted[&result.spec.photo].failed_revision.is_none(),
            "cached frame was decoded again"
        );
        assert_eq!(
            manager.wanted[&result.spec.photo]
                .frame
                .as_ref()
                .unwrap()
                .size,
            [10, 16]
        );
        assert!(browser.rendered_key(1).is_some());
    }

    #[test]
    fn stale_preloaded_pixels_are_redecoded_and_match_a_cold_render() {
        let dir = tempfile::tempdir().unwrap();
        let mut result = fixture(dir.path(), 1);
        result.spec.edit.params.debug.deactivate_stochastic_effects = true;
        result.spec.edit.params.debug.deactivate_spatial_effects = true;
        let task = Scheduler::default().task(Priority::Background);
        let source = task
            .run(|| SourceImage::load(&result.spec.path, &result.spec.edit.raw, 0, &task))
            .unwrap();
        let replacement = dir.path().join("replacement.png");
        image::RgbaImage::from_pixel(72, 48, image::Rgba([30, 160, 80, 255]))
            .save(&replacement)
            .unwrap();
        std::fs::rename(replacement, &result.spec.path).unwrap();
        assert_ne!(source.stamp, image_stamp(&result.spec.path));
        let backend = spektrafilm_gpu::cpu_backend::CpuBackend;
        let actual = task
            .run(|| render(&result.spec, Some(source), &task, &backend))
            .unwrap();
        let expected = task
            .run(|| render(&result.spec, None, &task, &backend))
            .unwrap();
        assert_eq!(actual.frame.size, [72, 48]);
        assert_eq!(actual.frame.stamp, image_stamp(&result.spec.path));
        assert_eq!(actual.fingerprint, expected.fingerprint);
        assert_eq!(actual.image, expected.image);
    }

    #[test]
    fn restart_reuses_pixels_and_normalizes_catalog_crop_and_native_color() {
        let dir = tempfile::tempdir().unwrap();
        let result = fixture(dir.path(), 1);
        let path = dir.path().join("previews.sqlite");
        let mut cache = DiskCache::open(&path, DISK_LIMIT).unwrap();
        cache.put(&result).unwrap();
        drop(cache);
        let cache = DiskCache::open(&path, DISK_LIMIT).unwrap();
        let (_, quality, pixels) = cache.get(&result.spec, None).unwrap().unwrap();
        assert_eq!(quality, 64);
        assert_eq!(pixels.unwrap(), result.image);
        let mut active = result.spec.clone();
        active.detect_input_color = false;
        active.edit.params.io.input_color_space = result.frame.color.clone();
        active.edit.params.io.input_cctf_decoding = result.frame.encoded;
        active.edit.crop = Some(result.spec.resolve(&result.frame).1);
        assert_eq!(active.fingerprint(&result.frame), result.fingerprint);
        assert!(
            cache
                .get(&active, Some(&result.fingerprint))
                .unwrap()
                .unwrap()
                .2
                .is_none()
        );
    }

    #[test]
    fn edits_sources_profiles_and_builds_invalidate_but_undo_reuses_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let mut result = fixture(dir.path(), 1);
        result.spec.data = dir.path().join("data");
        let profiles = result.spec.data.join("profiles");
        std::fs::create_dir_all(&profiles).unwrap();
        let profile = profiles.join(format!("{}.json", result.spec.edit.film));
        std::fs::write(&profile, b"first").unwrap();
        result.fingerprint = result.spec.fingerprint(&result.frame);
        let mut cache = DiskCache::open(&dir.path().join("previews.sqlite"), DISK_LIMIT).unwrap();
        cache.put(&result).unwrap();
        let changes: Vec<Box<dyn Fn(&mut Spec)>> = vec![
            Box::new(|s| s.edit.params.camera.exposure_compensation_ev += 1.),
            Box::new(|s| s.edit.params.camera.tone.shadows = 30.),
            Box::new(|s| s.edit.raw.white_balance_shift = [0.1, -0.2]),
            Box::new(|s| s.edit.rotation = 1),
            Box::new(|s| s.edit.crop = Some(Crop::FULL)),
            Box::new(|s| s.edit.raw.lightroom_geometry.as_mut().unwrap().angle = 3.),
            Box::new(|s| s.edit.params.settings.preview_max_size = 128),
            Box::new(|s| s.build += "new"),
        ];
        for change in changes {
            let mut changed = result.spec.clone();
            change(&mut changed);
            assert!(cache.get(&changed, None).unwrap().is_none());
            assert!(cache.get(&result.spec, None).unwrap().is_some());
        }
        std::fs::write(&profile, b"second").unwrap();
        assert!(cache.get(&result.spec, None).unwrap().is_none());
        result.fingerprint = result.spec.fingerprint(&result.frame);
        cache.put(&result).unwrap();
        let original = std::fs::File::open(&result.spec.path).unwrap();
        let modified = original.metadata().unwrap().modified().unwrap();
        let replacement = dir.path().join("replacement");
        std::fs::copy(&result.spec.path, &replacement).unwrap();
        std::fs::File::open(&replacement)
            .unwrap()
            .set_modified(modified)
            .unwrap();
        std::fs::rename(&replacement, &result.spec.path).unwrap();
        assert!(cache.get(&result.spec, None).unwrap().is_none());
        // A delayed write from that old decode must also be rejected.
        cache.remove(&result.spec.photo).unwrap();
        cache.put(&result).unwrap();
        assert!(cache.get(&result.spec, None).unwrap().is_none());
    }

    #[test]
    fn virtual_copies_and_catalogs_have_independent_cache_entries() {
        let dir = tempfile::tempdir().unwrap();
        let first = fixture(dir.path(), 1);
        let mut copy = fixture(dir.path(), 2);
        copy.spec.edit.crop = Some(Crop::FULL);
        copy.fingerprint = copy.spec.fingerprint(&copy.frame);
        copy.image = noise();
        let mut other = fixture(dir.path(), 1);
        other.spec.photo.catalog = dir.path().join("other.lrcat");
        let mut cache = DiskCache::open(&dir.path().join("previews.sqlite"), DISK_LIMIT).unwrap();
        for result in [&first, &copy, &other] {
            cache.put(result).unwrap();
        }
        for result in [&first, &copy, &other] {
            assert_eq!(
                cache.get(&result.spec, None).unwrap().unwrap().2.unwrap(),
                result.image
            );
        }
        assert_eq!(
            cache
                .connection
                .query_row("SELECT COUNT(*) FROM previews", [], |r| r.get::<_, u32>(0))
                .unwrap(),
            3
        );
    }

    #[test]
    fn disk_cache_evicts_least_recently_used_and_stays_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("previews.sqlite");
        let limit = 512 * 1024;
        let mut cache = DiskCache::open(&path, limit).unwrap();
        let mut result = fixture(dir.path(), 1);
        result.image = noise();
        cache.put(&result).unwrap();
        let entry_bytes: u64 = cache
            .connection
            .query_row("SELECT bytes FROM previews", [], |r| r.get(0))
            .unwrap();
        cache.budget = entry_bytes * 3;
        for id in 2..=3 {
            result.spec.photo.id = id;
            cache.put(&result).unwrap();
        }
        let first = fixture(dir.path(), 1).spec;
        cache.get(&first, None).unwrap().unwrap();
        result.spec.photo.id = 4;
        cache.put(&result).unwrap();
        assert!(
            cache
                .get(&fixture(dir.path(), 2).spec, None)
                .unwrap()
                .is_none()
        );
        assert!(cache.get(&first, None).unwrap().is_some());
        for id in 5..=40 {
            result.spec.photo.id = id;
            cache.put(&result).unwrap();
        }
        assert!(std::fs::metadata(&path).unwrap().len() <= limit);
        assert_eq!(
            cache
                .connection
                .query_row("SELECT COUNT(*) FROM previews", [], |r| r.get::<_, u32>(0))
                .unwrap(),
            3
        );
        // An entry that cannot fit must not evict anything.
        cache.budget = 1;
        result.spec.photo.id = 41;
        cache.put(&result).unwrap();
        assert_eq!(
            cache
                .connection
                .query_row("SELECT COUNT(*) FROM previews", [], |r| r.get::<_, u32>(0))
                .unwrap(),
            3
        );
        // Repeated versions replace one row; a late drag frame can't lower its quality.
        cache.budget = limit * 3 / 4;
        result.spec.photo.id = 40;
        result.quality = 2048;
        cache.put(&result).unwrap();
        result.quality = 640;
        result.image = image::RgbaImage::new(1, 1);
        cache.put(&result).unwrap();
        assert_eq!(cache.get(&result.spec, None).unwrap().unwrap().1, 2048);
        result.spec.edit.params.camera.exposure_compensation_ev = 1.;
        result.fingerprint = result.spec.fingerprint(&result.frame);
        cache.put(&result).unwrap();
        assert_eq!(
            cache
                .connection
                .query_row("SELECT COUNT(*) FROM previews WHERE photo=40", [], |r| r
                    .get::<_, u32>(0))
                .unwrap(),
            1
        );
    }

    #[test]
    fn corrupt_images_and_disposable_database_recover() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("previews.sqlite");
        let result = fixture(dir.path(), 1);
        let mut cache = DiskCache::open(&path, DISK_LIMIT).unwrap();
        for sql in [
            "UPDATE previews SET png=X'001122'",
            "UPDATE previews SET frame='broken json'",
        ] {
            cache.put(&result).unwrap();
            cache.connection.execute_batch(sql).unwrap();
            assert!(cache.get(&result.spec, None).unwrap().is_none());
            assert_eq!(
                cache
                    .connection
                    .query_row("SELECT COUNT(*) FROM previews", [], |r| r.get::<_, u32>(0))
                    .unwrap(),
                0
            );
        }
        drop(cache);
        std::fs::write(&path, b"broken database").unwrap();
        let edits = dir.path().join("edits.sqlite");
        std::fs::write(&edits, b"authoritative edits").unwrap();
        let mut cache = DiskCache::open_recovering(&path, DISK_LIMIT).unwrap();
        cache.put(&result).unwrap();
        assert!(cache.get(&result.spec, None).unwrap().is_some());
        assert_eq!(std::fs::read(&edits).unwrap(), b"authoritative edits");
    }

    #[test]
    fn restart_probe_returns_cached_image_without_starting_a_render() {
        let dir = tempfile::tempdir().unwrap();
        let result = fixture(dir.path(), 1);
        let path = dir.path().join("previews.sqlite");
        DiskCache::open(&path, DISK_LIMIT)
            .unwrap()
            .put(&result)
            .unwrap();
        let ctx = egui::Context::default();
        let mut browser = browser(&result.spec, &ctx);
        let mut manager = Manager::default();
        manager.io = Some(IoWorker::start(ctx.clone(), path));
        // No decoded frame exists yet, as when opening a catalog after a restart.
        manager.observe(result.spec.clone(), None, true, true, None, &ctx);
        let deadline = Instant::now() + Duration::from_secs(5);
        while browser.rendered_key(1).is_none() && Instant::now() < deadline {
            manager.poll(&mut browser, &ctx);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(browser.rendered_key(1), Some(result.fingerprint.as_str()));
        assert!(manager.pending.is_empty() && manager.render.is_none());
        manager.shutdown();
    }

    #[test]
    fn obsolete_disk_and_background_results_cannot_replace_newer_edits() {
        let dir = tempfile::tempdir().unwrap();
        let old = fixture(dir.path(), 1);
        let ctx = egui::Context::default();
        let mut browser = browser(&old.spec, &ctx);
        let (io, _commands, replies) = fake_io();
        let mut manager = Manager::default();
        manager.io = Some(io);
        manager.observe(old.spec.clone(), None, false, true, None, &ctx);
        let old_ticket = manager.wanted[&old.spec.photo].ticket;
        let mut current = fixture(dir.path(), 1);
        current.spec.edit.params.camera.exposure_compensation_ev = 1.;
        current.fingerprint = current.spec.fingerprint(&current.frame);
        manager.observe(current.spec.clone(), None, false, true, None, &ctx);
        let ticket = manager.wanted[&current.spec.photo].ticket;
        replies
            .send(Reply {
                photo: current.spec.photo.clone(),
                ticket,
                revision: "new".into(),
                hit: Some((current.fingerprint.clone(), 64, Some(current.image))),
            })
            .unwrap();
        replies
            .send(Reply {
                photo: old.spec.photo.clone(),
                ticket: old_ticket,
                revision: "old".into(),
                hit: Some((old.fingerprint.clone(), 64, Some(old.image.clone()))),
            })
            .unwrap();
        let (tx, rx) = mpsc::channel();
        manager.render = Some(RenderJob {
            pending: Pending {
                photo: old.spec.photo.clone(),
                ticket: old_ticket,
                revision: "old".into(),
            },
            task: Scheduler::default().task(Priority::Background),
            rx,
        });
        tx.send(Ok(old)).unwrap();
        manager.poll(&mut browser, &ctx);
        assert_eq!(browser.rendered_key(1), Some(current.fingerprint.as_str()));
        assert!(manager.render.is_none() && manager.writes.is_empty());
    }

    #[test]
    fn failed_background_render_does_not_retry_forever_and_scroll_prunes_work() {
        let dir = tempfile::tempdir().unwrap();
        let result = fixture(dir.path(), 1);
        let ctx = egui::Context::default();
        let mut browser = browser(&result.spec, &ctx);
        let (io, _commands, replies) = fake_io();
        let mut manager = Manager::default();
        manager.io = Some(io);
        manager.observe(result.spec.clone(), None, false, true, None, &ctx);
        let ticket = manager.wanted[&result.spec.photo].ticket;
        let pending = || Pending {
            photo: result.spec.photo.clone(),
            ticket,
            revision: "same".into(),
        };
        let (tx, rx) = mpsc::channel();
        manager.render = Some(RenderJob {
            pending: pending(),
            task: Scheduler::default().task(Priority::Background),
            rx,
        });
        tx.send(Err("unavailable original".into())).unwrap();
        manager.poll(&mut browser, &ctx);
        replies
            .send(Reply {
                photo: result.spec.photo.clone(),
                ticket,
                revision: "same".into(),
                hit: None,
            })
            .unwrap();
        manager.poll(&mut browser, &ctx);
        assert!(manager.pending.is_empty());
        replies
            .send(Reply {
                photo: result.spec.photo.clone(),
                ticket,
                revision: "changed".into(),
                hit: None,
            })
            .unwrap();
        manager.poll(&mut browser, &ctx);
        assert_eq!(manager.pending.len(), 1);
        manager.wanted.clear();
        manager.start_render(
            &Scheduler::default(),
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            &mut preload::Preloader::default(),
            &ctx,
            true,
        );
        assert!(manager.pending.is_empty());
    }

    #[test]
    fn background_matches_editor_for_raw_white_balance_crop_and_display_conversion() {
        compare_render_paths(Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend));
    }

    #[test]
    fn gpu_background_matches_editor() {
        let Some(backend) = spektrafilm_gpu::wgpu_backend::WgpuBackend::new() else {
            return;
        };
        compare_render_paths(Arc::new(backend));
    }

    fn compare_render_paths(backend: Arc<dyn ComputeBackend>) {
        let dir = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut app = App::new(&cc, backend.clone(), None);
        app.preview_max_size = 24;
        app.params.settings.preview_max_size = 128; // Legacy per-photo value must be ignored.
        app.params.debug.deactivate_stochastic_effects = true;
        app.params.debug.deactivate_spatial_effects = true;
        app.params.io.crop = false;
        app.params.camera.exposure_compensation_ev = 0.75;
        app.params.camera.tone.shadows = 20.;
        app.raw_options.decoder = "rawler".into();
        app.raw_options.white_balance_shift = [0.1, -0.05];
        app.raw_options.lightroom_geometry = Some(Geometry {
            orientation: 6,
            crop: [0.1, 0.2, 0.9, 0.8],
            angle: 3.,
        });
        let input = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../spektrafilm-io/tests/fixtures/synthetic.dng");
        let catalog = dir.path().join("test.lrcat");
        app.current_photo = Some((catalog.clone(), 1));
        app.load_image_with_state(&input, 1, None, None);
        while let Some(job) = &mut app.load_job {
            if let Some(handle) = job.handle.take() {
                handle.join().unwrap();
            }
            app.poll_load_job();
        }
        assert!(app.image.is_some(), "{}", app.status);
        let spec = app.current_thumbnail_spec().unwrap();
        app.catalog = Some(browser(&spec, &ctx));
        app.thumbnails.io = Some(fake_io().0);
        let task = app.scheduler.task(Priority::Background);
        let expected = task
            .run(|| render(&spec, None, &task, backend.as_ref()))
            .unwrap();
        for transform in [true, false] {
            app.display.display_transform = transform;
            app.dispatch_render(&ctx);
            let mut job = app.render_job.take().unwrap();
            job.handle.take().unwrap().join().unwrap();
            let result = job.rx.recv().unwrap().unwrap();
            let thumbnail = result.thumbnail.unwrap();
            assert_eq!(thumbnail.fingerprint, expected.fingerprint);
            assert_eq!(thumbnail.quality, 24);
            assert_eq!(thumbnail.image, expected.image);
            assert_eq!(
                thumbnail.image.width() as f64 / thumbnail.image.height() as f64,
                expected.image.width() as f64 / expected.image.height() as f64
            );
            app.accept_rendered_thumbnail(thumbnail, false, &ctx);
            assert_eq!(
                app.catalog.as_ref().unwrap().rendered_key(1),
                Some(expected.fingerprint.as_str())
            );
        }
        // Neither a completed render from another virtual copy nor a pre-undo edit can land.
        let mut stale = task
            .run(|| render(&spec, None, &task, backend.as_ref()))
            .unwrap();
        stale.spec.photo.id = 2;
        app.accept_rendered_thumbnail(stale, false, &ctx);
        assert!(app.catalog.as_ref().unwrap().rendered_key(2).is_none());
        app.params.camera.exposure_compensation_ev = -1.;
        let before = app.thumbnails.serial;
        app.accept_rendered_thumbnail(expected, true, &ctx);
        assert_eq!(app.thumbnails.serial, before);
        app.open_crop_tool();
        assert!(app.current_thumbnail_spec().is_none());
        assert!(app.observed_thumbnail_spec().unwrap().edit.crop.is_some());
    }
}
