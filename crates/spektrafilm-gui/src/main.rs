// Native editor for spektrafilm.
//
// Loads an image, exposes the most impactful runtime parameters as
// egui controls, and re-renders the full GPU-resident pipeline whenever
// a control changes. The output texture is uploaded to egui once per
// render; the panel scales it to fit.

mod calibration_ui;
mod catalog_edits;
mod catalog_ui;
mod chrome;
mod crop_ui;
mod edits;
mod file_menu;
mod history;
mod icons;
mod inspector;
mod lut_ui;
mod names;
mod paper_back;
mod preload;
mod project;
mod settings;
mod theme;
mod tooltips;
mod view;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use eframe::egui;
use project::ExportFormat;
use rayon::prelude::*;
use spektrafilm_core::params::RuntimeParams;
use spektrafilm_core::pipeline::Pipeline;
use spektrafilm_core::profile;
use spektrafilm_gpu::ComputeBackend;
use spektrafilm_gpu::scheduler::{self, Priority, Scheduler, Task};
use spektrafilm_io::crop::Crop;
use spektrafilm_math::image::ImageBuf;
use spektrafilm_math::precision::to_f32;

const PREVIEW_DEBOUNCE: Duration = Duration::from_millis(40);
const INTERACTIVE_DEBOUNCE: Duration = Duration::from_millis(16);
const IN_FLIGHT_REPAINT: Duration = Duration::from_millis(16);
const PREVIEW_TEXTURE_MAX_DIM: usize = 8192;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    if let Err(err) = embedded_bundle::setup() {
        eprintln!("[spektrafilm] embedded bundle setup failed: {err:#}");
    }
    let backend = spektrafilm_gpu::select_backend();
    let backend: Arc<dyn ComputeBackend> = Arc::from(backend);
    let initial_image = std::env::args().nth(1).map(PathBuf::from);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([900.0, 560.0]),
        renderer: gui_renderer(),
        // Window geometry, panel widths, section state and app preferences.
        persistence_path: Some(settings::storage_file()),
        // egui's default device descriptor hardcodes
        // `max_texture_dimension_2d: 8192`, so uploading the preview of an
        // image wider or taller than 8192 px (a single texture) panics.
        // Raise the cap to the adapter's real limit (16384 on Apple
        // Silicon Metal, higher on discrete GPUs), floored at egui's 8192
        // so we never regress. This is eframe's OWN render device — the
        // compute pipeline's wgpu device (WgpuBackend::new) is separate
        // and already lifts its storage-buffer limits.
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            wgpu_setup: eframe::egui_wgpu::WgpuSetupCreateNew {
                device_descriptor: Arc::new(|adapter: &eframe::wgpu::Adapter| {
                    let base_limits = if adapter.get_info().backend == eframe::wgpu::Backend::Gl {
                        eframe::wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        eframe::wgpu::Limits::default()
                    };
                    eframe::wgpu::DeviceDescriptor {
                        label: Some("spektrafilm egui wgpu device"),
                        required_features: eframe::wgpu::Features::default(),
                        required_limits: eframe::wgpu::Limits {
                            max_texture_dimension_2d: adapter
                                .limits()
                                .max_texture_dimension_2d
                                .max(8192),
                            ..base_limits
                        },
                        memory_hints: eframe::wgpu::MemoryHints::default(),
                    }
                }),
                ..Default::default()
            }
            .into(),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::run_native(
        "spektrafilm",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc, backend, initial_image)))),
    )
}

#[cfg(embed_bundle)]
mod embedded_bundle {
    use std::io::Write;
    use std::path::{Path, PathBuf};

    use anyhow::{Context, Result};

    include!(env!("SPEKTRAFILM_EMBED_MANIFEST"));

    pub fn setup() -> Result<()> {
        let root = cache_root().join(BUNDLE_ID);
        let data_dir = root.join("data");
        let f64_cli = root.join(f64_cli_name());
        let marker = root.join(".ready");

        if !marker.is_file() || !data_dir.join("profiles").is_dir() || !f64_cli.is_file() {
            std::fs::create_dir_all(&data_dir)
                .with_context(|| format!("creating {}", data_dir.display()))?;
            for (rel, bytes) in DATA_FILES {
                write_file(&data_dir.join(rel), bytes)?;
            }
            write_file(&f64_cli, F64_EXE)?;
            write_file(&marker, BUNDLE_ID.as_bytes())?;
        }

        // This runs before any worker threads are spawned.
        unsafe {
            std::env::set_var("SPEKTRAFILM_DATA_DIR", &data_dir);
            std::env::set_var("SPEKTRAFILM_F64_CLI", &f64_cli);
        }
        Ok(())
    }

    fn cache_root() -> PathBuf {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("spektrafilm")
            .join("embedded")
    }

    fn f64_cli_name() -> &'static str {
        if cfg!(windows) {
            "spektrafilm-f64.exe"
        } else {
            "spektrafilm-f64"
        }
    }

    fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let mut file =
            std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
        file.write_all(bytes)
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }
}

#[cfg(not(embed_bundle))]
mod embedded_bundle {
    use anyhow::Result;

    pub fn setup() -> Result<()> {
        Ok(())
    }
}

fn gui_renderer() -> eframe::Renderer {
    match std::env::var("SPEKTRAFILM_GUI_RENDERER") {
        Ok(v) if v.eq_ignore_ascii_case("wgpu") => eframe::Renderer::Wgpu,
        Ok(v) if v.eq_ignore_ascii_case("glow") => eframe::Renderer::Glow,
        _ => eframe::Renderer::Wgpu,
    }
}

/// One entry in the film / paper combo box. `stock` is the filename
/// stem (the unique key the profile loader expects); `display` is the
/// human-readable label from the profile's `info.name`, falling back
/// to the stock id when missing.
#[derive(Debug, Clone)]
struct ProfileEntry {
    stock: String,
    display: String,
    /// Picker group such as "Color negative" or "Photographic paper".
    group: &'static str,
}

struct LensDatabaseStatus {
    label: &'static str,
    detail: String,
}

impl LensDatabaseStatus {
    fn detect() -> Self {
        use spektrafilm_io::lens_database;
        match lens_database::discover(None) {
            Ok(Some(path)) => Self {
                label: if lens_database::cached().as_ref() == Some(&path) {
                    "Database: downloaded"
                } else {
                    "Database: installed locally"
                },
                detail: path.display().to_string(),
            },
            Ok(None) => Self {
                label: "Database: not downloaded yet",
                detail: "Downloads automatically when lens correction is first applied.".into(),
            },
            Err(error) => Self {
                label: "Database: unavailable",
                detail: format!("{error:#}"),
            },
        }
    }
}

struct App {
    ctx: egui::Context,
    scheduler: Scheduler,
    load_job: Option<LoadJob>,
    pending_load: Option<LoadRequest>,
    preload: preload::Preloader,
    preload_radius: u8,
    catalog: Option<catalog_ui::Browser>,
    preview_input: Vec<(u32, Arc<ImageBuf>)>,
    rendered_preview_limit: u32,
    render_path: String,
    auto_preview: bool,
    gpu_export: bool,
    export_format: ExportFormat,
    lut: lut_ui::Settings,
    render_requested: bool,
    display: project::DisplayParams,
    scan_for_print_restore: Option<[bool; 3]>,
    backend: Arc<dyn ComputeBackend>,
    data_dir: PathBuf,
    /// Profiles where `info.support == "film"`.
    films: Vec<ProfileEntry>,
    /// Profiles where `info.support == "paper"` (or other print-stage
    /// supports).
    papers: Vec<ProfileEntry>,
    film_name: String,
    print_name: String,
    /// Development-time family (minutes) of the selected film / paper —
    /// non-trivial only for B&W stocks. Drives the dev-time pickers.
    film_dev_times: Vec<f64>,
    print_dev_times: Vec<f64>,
    params: RuntimeParams,
    metadata: spektrafilm_io::Metadata,
    raw_options: spektrafilm_io::RawOptions,
    /// Settings that produced the current pixels while automatic RAW reloads run.
    loaded_raw_options: Option<spektrafilm_io::RawOptions>,
    lens_database_status: LensDatabaseStatus,
    save_options: spektrafilm_io::SaveOptions,
    image_path: Option<PathBuf>,
    /// The pixels the pipeline sees: `uncropped` with `applied_crop`.
    image: Option<Arc<ImageBuf>>,
    /// The whole frame, after orientation and quarter turns.
    uncropped: Option<Arc<ImageBuf>>,
    source_stamp: Option<(u64, std::time::SystemTime)>,
    /// The photo's crop; `None` until a catalog photo's Lightroom crop is read.
    crop: Option<Crop>,
    /// The crop `image` was made with (the whole frame while cropping).
    applied_crop: Crop,
    /// The crop tool, while it is open.
    crop_tool: Option<crop_ui::CropTool>,
    /// The next rendered frame develops from blank paper (a newly opened photo).
    develop_next: bool,
    /// Last rendered preview in the selected output color space. Retained
    /// so the Save button can write it without re-running the pipeline.
    output_image: Option<ImageBuf>,
    output_tex: Option<egui::TextureHandle>,
    input_tex: Option<egui::TextureHandle>,
    source_display_key: Option<(String, bool, bool)>,
    rotation: u8,
    canvas: Option<Arc<view::Renderer>>,
    canvas_input: Option<Arc<view::Image>>,
    canvas_output: Option<Arc<view::Image>>,
    canvas_previous: Option<Arc<view::Image>>,
    canvas_paper: Option<Arc<view::Image>>,
    paper_tex: Option<egui::TextureHandle>,
    paper_size: Option<[usize; 2]>,
    animation: Option<(Instant, bool)>,
    calibration_job: Option<calibration_ui::Job>,
    pipeline_cache_key: Option<String>,
    pipeline_cache: Option<Pipeline>,
    last_render_ms: f32,
    last_pipeline_build_ms: f32,
    last_input_clone_ms: f32,
    last_scale_ms: f32,
    last_preview_ms: f32,
    last_worker_total_ms: f32,
    status: String,
    /// Set when any control change should trigger a re-render. The
    /// next `update()` tick dispatches the render onto a worker
    /// thread so the UI stays responsive while the pipeline (up to
    /// ~500 ms on 6 MP) runs in the background.
    dirty: bool,
    /// Marked when the user changes a param while a previous render
    /// is still in flight. After that render finishes we re-arm
    /// `dirty` so the latest state gets a fresh pass instead of
    /// dropping the user's mid-render edits on the floor.
    pending_dirty: bool,
    dirty_since: Option<Instant>,
    /// Current preview zoom multiplier on top of the fit-to-panel
    /// scale (1.0 = fit). Mouse wheel adjusts it around the cursor,
    /// drag pans, double-click resets.
    zoom: f32,
    /// Pan offset (screen pixels) added to the preview centre.
    pan: egui::Vec2,
    /// macOS-only: tag the wgpu CAMetalLayer's `colorspace` as sRGB on
    /// the first `update()` tick (it's not yet wired up at
    /// `App::new` time). Set to `true` once the call succeeds so we
    /// don't retry every frame.
    #[cfg(target_os = "macos")]
    metal_colorspace_tagged: bool,
    /// In-flight pipeline render. `Some` while the worker thread is
    /// running; main thread polls the receiver each frame and
    /// uploads the resulting texture once it lands. Decoupling the
    /// render from the UI thread is what keeps sliders responsive —
    /// the 250–500 ms pipeline used to block input handling.
    render_job: Option<RenderJob>,
    /// In-flight image or LUT export. The `update()` loop polls the receiver and
    /// surfaces success/failure in `status` when the worker thread
    /// completes. Joined eagerly to release the thread.
    export_job: Option<ExportJob>,
    /// Inspector stage tab shown in the sidebar.
    tab: inspector::Tab,
    history: history::History,
    /// `params` serialized at the start of the frame (undo, change markers).
    params_json: Arc<serde_json::Value>,
    /// Stock-default params for the named film, typed and as JSON.
    baseline: Option<(String, Arc<RuntimeParams>, Arc<serde_json::Value>)>,
    /// Inspector rows mapped to the param pointers they edit, learned on change.
    learned_rows: Arc<std::collections::HashMap<egui::Id, Vec<String>>>,
    /// Show the unprocessed input instead of the render (held key or button).
    compare_hold: bool,
    before_button_held: bool,
    /// `zoom` value at which one source pixel maps to one screen pixel.
    view_actual_zoom: f32,
    window_title: String,
    /// Per-photo settings database, opened on first use.
    edit_store: Option<edits::EditStore>,
    edit_store_failed: bool,
    /// The open catalog photo (catalog path, Lightroom photo id) that edits save to.
    current_photo: Option<(PathBuf, i64)>,
    /// Settings captured by Photo → Copy settings (C).
    copied_edit: Option<edits::PhotoEdit>,
    /// Whether the current photo's history already has rows in the store.
    history_persisted: bool,
    /// Label for the next history step when it isn't a control change.
    pending_label: Option<String>,
    /// Inspector captions for param pointers, learned as rows change; used to
    /// label history steps.
    pointer_captions: std::collections::HashMap<String, String>,
    /// The catalog to reopen at the next launch.
    last_catalog: Option<PathBuf>,
}

struct LoadJob {
    discard: bool,
    task: Task,
    path: PathBuf,
    started: Instant,
    color_override: Option<(String, bool)>,
    rotation: u8,
    raw_options: spektrafilm_io::RawOptions,
    view_override: Option<(f32, egui::Vec2)>,
    rx: mpsc::Receiver<Result<LoadedPreview, String>>,
    handle: Option<JoinHandle<()>>,
}

struct LoadedPreview {
    source: SourceImage,
    /// `source` cropped with `crop`.
    image: Arc<ImageBuf>,
    crop: Crop,
    input: Arc<ImageBuf>,
    preview: PreviewTextureData,
    paper: Arc<PreviewTextureData>,
    source_key: (String, bool, bool),
}

#[derive(Clone)]
struct SourceImage {
    pixels: Arc<ImageBuf>,
    metadata: spektrafilm_io::Metadata,
    color_space: String,
    encoded: bool,
    stamp: Option<(u64, std::time::SystemTime)>,
}

#[derive(Clone)]
struct LoadRequest {
    path: PathBuf,
    rotation: u8,
    color_override: Option<(String, bool)>,
    view_override: Option<(f32, egui::Vec2)>,
    raw_options: spektrafilm_io::RawOptions,
    /// `None` reads the Lightroom crop, if any, once the frame size is known.
    crop: Option<Crop>,
}

impl SourceImage {
    fn load(
        path: &Path,
        options: &spektrafilm_io::RawOptions,
        rotation: u8,
        task: &Task,
    ) -> Result<Self, String> {
        task.check().map_err(|e| e.to_string())?;
        let stamp = image_stamp(path);
        // The whole frame; the editor's own crop starts from Lightroom's.
        let mut options = options.clone();
        options.lightroom_geometry = options
            .lightroom_geometry
            .map(|geometry| geometry.orientation_only());
        let mut loaded = spektrafilm_io::load(path, &options).map_err(|e| format!("{e:#}"))?;
        task.check().map_err(|e| e.to_string())?;
        if rotation % 4 != 0 {
            loaded.pixels = loaded.pixels.rotated_quarter_turns(rotation);
        }
        task.check().map_err(|e| e.to_string())?;
        if stamp != image_stamp(path) {
            return Err(
                "Original changed while it was being decoded; select it again to reload.".into(),
            );
        }
        Ok(Self {
            pixels: Arc::new(loaded.pixels),
            metadata: loaded.metadata,
            color_space: loaded.color_space,
            encoded: loaded.encoded,
            stamp,
        })
    }
}

fn image_stamp(path: &Path) -> Option<(u64, std::time::SystemTime)> {
    use std::hash::{Hash, Hasher};
    let metadata = std::fs::metadata(path).ok()?;
    let mut identity = std::collections::hash_map::DefaultHasher::new();
    metadata.len().hash(&mut identity);
    metadata.created().ok().hash(&mut identity);
    // Replacing or editing a file can preserve its size and modification time.
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        (
            metadata.dev(),
            metadata.ino(),
            metadata.ctime(),
            metadata.ctime_nsec(),
        )
            .hash(&mut identity);
    }
    Some((identity.finish(), metadata.modified().ok()?))
}

/// One in-flight preview render. The worker owns a Pipeline + the
/// ImageBuf clone and, when it finishes, sends back the output buffer
/// plus the two timings the status bar shows.
struct RenderJob {
    discard: bool,
    task: Task,
    started: Instant,
    /// A fast low-resolution frame requested while a control was being dragged.
    interactive: bool,
    rx: mpsc::Receiver<Result<RenderResult, String>>,
    handle: Option<JoinHandle<()>>,
}

struct RenderResult {
    output: ImageBuf,
    input_preview: Arc<ImageBuf>,
    preview_limit: u32,
    path: String,
    preview: PreviewTextureData,
    source_preview: Option<((String, bool, bool), PreviewTextureData)>,
    paper: Arc<PreviewTextureData>,
    input_clone_ms: f32,
    scale_ms: f32,
    pipeline_build_ms: f32,
    render_ms: f32,
    preview_ms: f32,
    worker_total_ms: f32,
    pipeline_cache: (String, Pipeline),
}

impl LoadJob {
    fn discard(&mut self) {
        self.discard = true;
        self.task.cancel();
    }
}

impl RenderJob {
    fn discard(&mut self) {
        self.discard = true;
        self.task.cancel();
    }
}

struct PreviewTextureData {
    size: [usize; 2],
    rgba: Vec<u8>,
}

/// An export worker with cooperative GPU cancellation or cancellable CLI subprocess.
/// The worker is joined at completion and when the app exits.
struct ExportJob {
    rx: mpsc::Receiver<Result<(f32, String), String>>,
    handle: Option<JoinHandle<()>>,
    cancel: Arc<AtomicBool>,
    task: Task,
    started_at: Instant,
    label: String,
}

impl App {
    fn new(
        cc: &eframe::CreationContext<'_>,
        backend: Arc<dyn ComputeBackend>,
        initial_image: Option<PathBuf>,
    ) -> Self {
        theme::apply(&cc.egui_ctx);

        let data_dir = pick_data_dir();
        let (films, papers) = scan_profiles(&data_dir);
        let film_name = pick_default_stock(&films, "kodak_gold_200");
        // Load the default film once to derive two start-up choices:
        //   - the paired print paper (`target_print`): each profile is
        //     tuned against a specific paper, so a generic default gives a
        //     colour-shifted preview (Kodak Gold pairs with Portra Endura,
        //     not Fujifilm Crystal Archive).
        //   - whether to scan the film directly: positive/slide stocks have
        //     no print paper and must be scanned (see the film-change
        //     handler in `controls_panel`).
        let film_profile = profile::load_profile_by_name(&data_dir, &film_name).ok();
        let print_name = film_profile
            .as_ref()
            .and_then(|f| f.info.target_print.clone())
            .filter(|t| papers.iter().any(|p| &p.stock == t))
            .unwrap_or_else(|| pick_default_stock(&papers, "fujifilm_crystal_archive_typeii"));
        let mut params = RuntimeParams::default();
        params.settings.preview_max_size = 2048;
        params.io.scan_film = film_profile.as_ref().is_some_and(|f| f.is_positive());
        if let Some(film) = &film_profile {
            spektrafilm_core::pipeline::apply_stock_defaults(film, &mut params);
        }
        let film_dev_times = film_profile
            .as_ref()
            .map(|f| f.data.development_time.clone())
            .unwrap_or_default();
        let print_dev_times = profile_dev_times(&data_dir, &print_name);

        let mut app = Self {
            ctx: cc.egui_ctx.clone(),
            scheduler: Scheduler::default(),
            load_job: None,
            pending_load: None,
            preload: Default::default(),
            preload_radius: preload::DEFAULT_RADIUS,
            catalog: None,
            preview_input: Vec::new(),
            rendered_preview_limit: 0,
            render_path: "Awaiting render".into(),
            auto_preview: true,
            gpu_export: true,
            export_format: Default::default(),
            lut: Default::default(),
            render_requested: false,
            display: Default::default(),
            scan_for_print_restore: None,
            backend,
            data_dir,
            films,
            papers,
            film_name,
            print_name,
            film_dev_times,
            print_dev_times,
            params,
            metadata: Default::default(),
            raw_options: Default::default(),
            loaded_raw_options: None,
            lens_database_status: LensDatabaseStatus::detect(),
            save_options: Default::default(),
            image_path: None,
            image: None,
            uncropped: None,
            source_stamp: None,
            crop: None,
            applied_crop: Crop::FULL,
            crop_tool: None,
            develop_next: false,
            output_image: None,
            output_tex: None,
            input_tex: None,
            source_display_key: None,
            rotation: 0,
            canvas: cc.wgpu_render_state.as_ref().map(view::Renderer::new),
            canvas_input: None,
            canvas_output: None,
            canvas_previous: None,
            animation: None,
            canvas_paper: None,
            paper_tex: None,
            paper_size: None,
            calibration_job: None,
            pipeline_cache_key: None,
            pipeline_cache: None,
            last_render_ms: 0.0,
            last_pipeline_build_ms: 0.0,
            last_input_clone_ms: 0.0,
            last_scale_ms: 0.0,
            last_preview_ms: 0.0,
            last_worker_total_ms: 0.0,
            pending_dirty: false,
            dirty_since: None,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            render_job: None,
            status: String::new(),
            dirty: false,
            #[cfg(target_os = "macos")]
            metal_colorspace_tagged: false,
            export_job: None,
            tab: inspector::Tab::default(),
            history: Default::default(),
            params_json: Default::default(),
            baseline: None,
            learned_rows: Default::default(),
            compare_hold: false,
            before_button_held: false,
            view_actual_zoom: 0.0,
            window_title: String::new(),
            edit_store: None,
            edit_store_failed: false,
            current_photo: None,
            copied_edit: None,
            history_persisted: false,
            pending_label: None,
            pointer_captions: Default::default(),
            last_catalog: None,
        };
        if project::default_path().exists() {
            if let Err(e) = app.restore_project(&project::default_path(), false) {
                app.status = format!("Startup defaults: {e:#}");
            }
        }
        if let Some(storage) = cc.storage {
            app.restore_settings(storage, initial_image.is_some());
        }
        if let Some(p) = initial_image {
            app.load_image_from_path(&p);
        }
        app
    }

    fn load_image_from_path(&mut self, path: &Path) {
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("lrcat"))
        {
            self.open_catalog(path.to_path_buf());
            return;
        }
        self.leave_photo();
        self.preload.stop();
        self.raw_options.lightroom_geometry = None;
        self.crop = None;
        if let Some(catalog) = &mut self.catalog {
            catalog.selected = None;
        }
        self.load_image_with_state(path, 0, None, None);
    }

    fn clear_image(&mut self) {
        self.preload.forget_current();
        self.image = None;
        self.uncropped = None;
        self.source_stamp = None;
        self.applied_crop = Crop::FULL;
        self.crop_tool = None;
        self.image_path = None;
        self.loaded_raw_options = None;
        self.output_image = None;
        self.output_tex = None;
        self.input_tex = None;
        self.canvas_input = None;
        self.canvas_output = None;
        self.canvas_previous = None;
        self.animation = None;
        self.preview_input.clear();
        self.dirty = false;
        self.pending_dirty = false;
    }

    fn open_catalog(&mut self, path: PathBuf) {
        self.preload.stop();
        self.leave_photo();
        if let Some(job) = &mut self.load_job {
            job.discard();
        }
        if let Some(job) = &mut self.render_job {
            job.discard();
        }
        self.pending_load = None;
        self.clear_image();
        self.raw_options.lightroom_geometry = None;
        self.last_catalog = Some(path.clone());
        self.catalog = Some(catalog_ui::Browser::open(path, &self.ctx, &self.scheduler));
        self.status = "Reading Lightroom catalog…".into();
        self.load_edited_markers();
    }

    fn select_catalog_photo(&mut self, photo: spektrafilm_io::lightroom::Photo) {
        self.preload.abandon_wait();
        // Save the outgoing photo, then open the incoming one with its own
        // settings (or the defaults, if it has none).
        self.commit_photo_edit();
        self.current_photo = self
            .catalog
            .as_ref()
            .map(|browser| (browser.path().to_path_buf(), photo.id));
        self.history_persisted = false;
        let rotation = match self.current_photo.clone() {
            Some((catalog, id)) => self.open_photo_edit(&catalog, id),
            None => {
                self.rebase_history();
                0
            }
        };
        match photo.geometry {
            Ok(geometry) => {
                self.raw_options.lightroom_geometry = Some(geometry);
                self.params.io.crop = false;
                self.load_image_with_state(&photo.path, rotation, None, None);
            }
            Err(error) => {
                if let Some(job) = &mut self.load_job {
                    job.discard();
                }
                if let Some(job) = &mut self.render_job {
                    job.discard();
                }
                self.pending_load = None;
                self.clear_image();
                self.status = format!("Load error: catalog crop/rotation: {error}");
            }
        }
    }
    fn load_image_with_state(
        &mut self,
        path: &Path,
        rotation: u8,
        color_override: Option<(String, bool)>,
        view_override: Option<(f32, egui::Vec2)>,
    ) {
        self.preload.abandon_wait();
        if let Some(render) = &mut self.render_job {
            render.discard();
        }
        let path = match std::path::absolute(path) {
            Ok(path) => path,
            Err(e) => {
                self.status = format!("Load error: could not resolve image path: {e}");
                return;
            }
        };
        let request = LoadRequest {
            path,
            rotation,
            color_override,
            view_override,
            raw_options: self.raw_options.clone(),
            crop: self.wanted_crop(),
        };
        if self.image_path.as_ref() != Some(&request.path)
            || self.export_raw_options().lightroom_geometry
                != request.raw_options.lightroom_geometry
        {
            self.clear_image();
        }
        if let Some(job) = &mut self.load_job {
            job.discard();
            self.status = format!(
                "Waiting to load {}…",
                request
                    .path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            );
            self.pending_load = Some(request);
            return;
        }
        self.start_image_load(request);
    }

    fn reload_raw_if_changed(&mut self, previous: &spektrafilm_io::RawOptions) {
        if previous == &self.raw_options {
            return;
        }
        let (path, rotation, view, options) = if let Some(request) = self
            .pending_load
            .as_ref()
            .or_else(|| self.preload.waiting())
        {
            (
                &request.path,
                request.rotation,
                request.view_override,
                &request.raw_options,
            )
        } else if let Some(job) = &self.load_job {
            if job.discard {
                return;
            }
            (&job.path, job.rotation, job.view_override, &job.raw_options)
        } else if let Some(path) = &self.image_path {
            (
                path,
                self.rotation,
                Some((self.zoom, self.pan)),
                self.export_raw_options(),
            )
        } else {
            return;
        };
        if !spektrafilm_io::is_raw(path) || options == &self.raw_options {
            return;
        }
        let path = path.clone();
        self.load_image_with_state(&path, rotation, None, view);
    }

    fn start_image_load(&mut self, request: LoadRequest) {
        let Some(request) = self.wait_for_preload(request) else {
            return;
        };
        let LoadRequest {
            path,
            rotation,
            color_override,
            view_override,
            raw_options: options,
            crop,
        } = request;
        let display_transform = self.display.display_transform;
        let backend = self.backend.clone();
        let preview_color = color_override.clone();
        let cached = self.preload.cached(&path, rotation, &options);
        let worker_path = path.clone();
        let loaded_options = options.clone();
        let task = self.scheduler.task(Priority::Interactive);
        let worker_task = task.clone();
        let ctx = self.ctx.clone();
        let (tx, rx) = mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("spektrafilm-load".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _scope = worker_task.enter();
                    let cpu = worker_task.cpu().map_err(|e| e.to_string())?;
                    let started = Instant::now();
                    let stamp = image_stamp(&worker_path);
                    let source =
                        match cached.filter(|source| stamp.is_some() && source.stamp == stamp) {
                            Some(source) => source,
                            None => {
                                SourceImage::load(&worker_path, &options, rotation, &worker_task)?
                            }
                        };
                    let crop = crop.unwrap_or_else(|| {
                        crop_ui::lightroom_crop(&options, &source.pixels, rotation)
                    });
                    let image = crop_ui::cropped(&source.pixels, &crop)?;
                    drop(cpu);
                    let loaded = started.elapsed();
                    let cpu = worker_task.cpu().map_err(|e| e.to_string())?;
                    let input = Arc::new(spektrafilm_core::resize::preview(&image, 2048));
                    drop(cpu);
                    let resized = started.elapsed();
                    let (color, encoded) =
                        preview_color.unwrap_or((source.color_space.clone(), source.encoded));
                    let mut io = spektrafilm_core::params::IoParams::default();
                    io.output_color_space = color;
                    io.output_cctf_encoding = encoded;
                    let source_key = (
                        io.output_color_space.clone(),
                        io.output_cctf_encoding,
                        display_transform,
                    );
                    let cpu = worker_task.cpu().map_err(|e| e.to_string())?;
                    let preview = prepare_preview_texture_data(&input, &io, display_transform, backend.as_ref());
                    drop(cpu);
                    let displayed = started.elapsed();
                    let _cpu = worker_task.cpu().map_err(|e| e.to_string())?;
                    let paper = paper_back::cached(preview.size[0], preview.size[1]);
                    eprintln!("[spektrafilm] load {}: decode {:.1} ms, resize {:.1} ms, display {:.1} ms, paper {:.1} ms", worker_path.display(), loaded.as_secs_f64() * 1000., (resized - loaded).as_secs_f64() * 1000., (displayed - resized).as_secs_f64() * 1000., (started.elapsed() - displayed).as_secs_f64() * 1000.);
                    Ok(LoadedPreview {
                        source,
                        image,
                        crop,
                        input,
                        preview,
                        paper,
                        source_key,
                    })
                }))
                .unwrap_or_else(|panic| Err(panic_message(panic)));
                ctx.request_repaint();
                let _ = tx.send(result);
            });
        match handle {
            Ok(handle) => {
                self.status = format!(
                    "Loading {}…",
                    path.file_name().unwrap_or_default().to_string_lossy()
                );
                self.load_job = Some(LoadJob {
                    discard: false,
                    task,
                    path,
                    started: Instant::now(),
                    color_override,
                    rotation,
                    raw_options: loaded_options,
                    view_override,
                    rx,
                    handle: Some(handle),
                });
                self.ctx.request_repaint();
            }
            Err(e) => self.status = format!("Load error: could not start decoder: {e}"),
        }
    }

    fn poll_load_job(&mut self) {
        let Some(job) = self.load_job.as_mut() else {
            return;
        };
        let result = match job.rx.try_recv() {
            Ok(r) => r,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                Err("decoder worker stopped unexpectedly".into())
            }
        };
        let mut job = self.load_job.take().unwrap();
        if let Some(handle) = job.handle.take() {
            let _ = handle.join();
        }
        if job.discard {
            if let Some(request) = self.pending_load.take() {
                self.start_image_load(request);
            }
            return;
        }
        // Refresh after a load (including failures after a successful download),
        // keeping filesystem discovery out of the UI paint loop.
        self.lens_database_status = LensDatabaseStatus::detect();
        match result {
            Ok(display) => {
                let LoadedPreview {
                    source,
                    image,
                    crop,
                    input,
                    preview,
                    paper,
                    source_key,
                } = display;
                if let Some(catalog) = &mut self.catalog {
                    catalog.source_thumbnail(&self.ctx, &job.path, &preview);
                }
                self.preload.remember_current(
                    job.path.clone(),
                    job.rotation,
                    job.raw_options.clone(),
                    source.clone(),
                );
                let new_image = self.image_path.as_ref() != Some(&job.path);
                // Rotating, cropping or redeveloping the same photo swaps in place.
                self.develop_next = new_image;
                self.loaded_raw_options = Some(job.raw_options);
                self.metadata = source.metadata;
                let (color, encoded) = job
                    .color_override
                    .unwrap_or((source.color_space, source.encoded));
                self.params.io.input_color_space = color;
                self.params.io.input_cctf_decoding = encoded;
                if !new_image {
                    // A quarter turn of the same photo turns its crop too.
                    let turns = (job.rotation + 4 - self.rotation) % 4;
                    self.crop = self.crop.map(|crop| crop.rotated_quarter_turns(turns));
                    if let Some(tool) = &mut self.crop_tool {
                        tool.rotate(turns);
                    }
                }
                self.rotation = job.rotation;
                self.source_stamp = source.stamp;
                self.uncropped = Some(source.pixels);
                self.applied_crop = crop;
                // A catalog photo's Lightroom crop, read now that its size is known.
                self.crop.get_or_insert(crop);
                let img = image;
                self.status = format!(
                    "Loaded {} × {} ({:.1} MP) in {:.0} ms",
                    img.width,
                    img.height,
                    img.pixel_count() as f64 / 1e6,
                    job.started.elapsed().as_secs_f32() * 1000.
                );
                self.image = Some(img);
                self.image_path = Some(job.path);
                self.output_image = None;
                self.output_tex = None;
                self.input_tex = None;
                self.canvas_input = None;
                self.canvas_output = None;
                self.canvas_previous = None;
                self.animation = None;
                self.preview_input.clear();
                self.preview_input.push((2048, input));
                self.source_display_key = Some(source_key);
                if let Some(canvas) = &self.canvas {
                    self.canvas_input = Some(canvas.upload(&preview));
                    self.canvas_paper = Some(canvas.upload(&paper));
                } else {
                    self.input_tex = Some(make_texture(
                        &self.ctx,
                        preview,
                        self.display.output_interpolation == "nearest",
                    ));
                    self.paper_tex = Some(make_texture(
                        &self.ctx,
                        PreviewTextureData {
                            size: paper.size,
                            rgba: paper.rgba.clone(),
                        },
                        false,
                    ));
                }
                self.paper_size = Some(paper.size);
                self.rendered_preview_limit = 0;
                let (zoom, pan) = job.view_override.unwrap_or((1., egui::Vec2::ZERO));
                self.zoom = zoom;
                self.pan = pan;
                if let Some(render) = &mut self.render_job {
                    render.discard();
                }
                self.dirty = true;
                if new_image {
                    // The file's own colour encoding was just applied; that's not
                    // an edit. A catalog photo keeps its (restored) history.
                    if self.current_photo.is_some() {
                        self.amend_history();
                    } else {
                        self.rebase_history();
                    }
                } else {
                    // Rotation or a RAW reload of the same photo.
                    self.save_photo_edit();
                }
            }
            Err(e) => {
                self.status = format!("Load error: {e}");
                eprintln!("[spektrafilm] {}", self.status);
            }
        }
    }

    fn rotate_input(&mut self, turns: u8) {
        if self.uncropped.is_none() || self.image_loading() {
            return;
        }
        // The crop turns with the frame (`crop` follows when the job lands).
        let crop = self.wanted_crop().unwrap_or(Crop::FULL);
        self.reframe(turns, crop.rotated_quarter_turns(turns));
    }

    /// Rebuild the pipeline input from the whole frame: turn it clockwise by
    /// `turns` quarter turns, then apply `crop`. The photo isn't decoded again.
    fn reframe(&mut self, turns: u8, crop: Crop) {
        let (Some(frame), Some(path)) = (self.uncropped.clone(), self.image_path.clone()) else {
            return;
        };
        if self.image_loading() {
            return;
        }
        let rotation = (self.rotation + turns) % 4;
        let metadata = self.metadata.clone();
        let color = self.params.io.input_color_space.clone();
        let encoded = self.params.io.input_cctf_decoding;
        let display_transform = self.display.display_transform;
        let backend = self.backend.clone();
        let ctx = self.ctx.clone();
        let (tx, rx) = mpsc::channel();
        let worker_color = color.clone();
        let source_stamp = self.source_stamp;
        let task = self.scheduler.task(Priority::Interactive);
        let worker_task = task.clone();
        let handle = std::thread::Builder::new()
            .name("spektrafilm-reframe".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _scope = worker_task.enter();
                    let _cpu = worker_task.cpu().map_err(|e| e.to_string())?;
                    let pixels = if turns % 4 == 0 {
                        frame
                    } else {
                        Arc::new(frame.rotated_quarter_turns(turns))
                    };
                    worker_task.check().map_err(|e| e.to_string())?;
                    let image = crop_ui::cropped(&pixels, &crop)?;
                    worker_task.check().map_err(|e| e.to_string())?;
                    let input = Arc::new(spektrafilm_core::resize::preview(&image, 2048));
                    let mut io = spektrafilm_core::params::IoParams::default();
                    io.output_color_space = worker_color.clone();
                    io.output_cctf_encoding = encoded;
                    let source_key = (
                        io.output_color_space.clone(),
                        io.output_cctf_encoding,
                        display_transform,
                    );
                    let preview = prepare_preview_texture_data(
                        &input,
                        &io,
                        display_transform,
                        backend.as_ref(),
                    );
                    let paper = paper_back::cached(preview.size[0], preview.size[1]);
                    Ok(LoadedPreview {
                        source: SourceImage {
                            pixels,
                            metadata,
                            color_space: worker_color,
                            encoded,
                            stamp: source_stamp,
                        },
                        image,
                        crop,
                        input,
                        preview,
                        paper,
                        source_key,
                    })
                }))
                .unwrap_or_else(|panic| Err(panic_message(panic)));
                let _ = tx.send(result);
                ctx.request_repaint();
            });
        match handle {
            Ok(handle) => {
                self.load_job = Some(LoadJob {
                    discard: false,
                    task,
                    path,
                    started: Instant::now(),
                    color_override: Some((color, encoded)),
                    rotation,
                    raw_options: self.export_raw_options().clone(),
                    view_override: None,
                    rx,
                    handle: Some(handle),
                });
                if let Some(job) = &mut self.render_job {
                    job.discard();
                }
                self.status = if turns % 4 == 0 {
                    "Cropping…"
                } else {
                    "Rotating input…"
                }
                .into();
                self.ctx.request_repaint();
            }
            Err(e) => self.status = format!("Rotate: {e}"),
        }
    }

    /// Spawn a render on a worker thread. The UI stays interactive
    /// during the 250–500 ms pipeline run (previously this blocked
    /// the main thread, dropping mid-drag slider events). If a job
    /// is already in flight, mark `pending_dirty` so the latest
    /// params get re-rendered as soon as the in-flight one returns.
    fn dispatch_render(&mut self, ctx: &egui::Context) {
        if self.render_job.is_some() {
            self.pending_dirty = true;
            return;
        }
        let t_clone = Instant::now();
        let Some(image) = self.image.as_ref().cloned() else {
            return;
        };
        let input_clone_ms = t_clone.elapsed().as_secs_f32() * 1000.0;
        let film_name = self.film_name.clone();
        let print_name = self.print_name.clone();
        let params = self.pipeline_params(self.applied_crop);
        let pipeline_template = self
            .pipeline_cache_key
            .clone()
            .zip(self.pipeline_cache.clone());
        let data_dir = self.data_dir.clone();
        let configured_limit = if params.settings.preview_max_size == 0 {
            image.width.max(image.height)
        } else {
            params.settings.preview_max_size
        };
        let preview_limit = if ctx.is_using_pointer() && self.auto_preview {
            configured_limit.min(640)
        } else {
            configured_limit
        };
        let interactive = preview_limit < configured_limit;
        let source_key = (
            params.io.input_color_space.clone(),
            params.io.input_cctf_decoding,
            self.display.display_transform,
        );
        let prepare_source = self.source_display_key.as_ref() != Some(&source_key);
        let source_image = self
            .preview_input
            .iter()
            .find(|(limit, _)| *limit == 2048)
            .map(|(_, i)| i.clone())
            .unwrap_or_else(|| image.clone());
        let image = self
            .preview_input
            .iter()
            .find(|(limit, _)| *limit == preview_limit)
            .map(|(_, image)| image.clone())
            .unwrap_or(image);
        let backend = self.backend.clone();
        let display_transform = self.display.display_transform;
        self.render_requested = false;
        self.status = format!("Rendering preview (maximum {preview_limit} px)…");
        let (tx, rx) = mpsc::channel();
        let ctx_for_worker = ctx.clone();
        // Drag frames go ahead of everything, including the last refine.
        let task = self.scheduler.task(if interactive {
            Priority::Drag
        } else {
            Priority::Interactive
        });
        let worker_task = task.clone();
        let handle = std::thread::Builder::new()
            .name("spektrafilm-render".into())
            .spawn(move || {
                // A panic inside the pipeline must still produce a channel
                // message — otherwise the receiver only sees a disconnect
                // and the user gets a uselessly vague status line.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _scope = worker_task.enter();
                    let cpu = worker_task.cpu().map_err(|e| e.to_string())?;
                    params
                        .validate()
                        .map_err(|e| format!("processing settings: {e}"))?;
                    let t_total = Instant::now();
                    let t_scale = Instant::now();
                    // simulate_preview always bounds input size. Preview mode
                    // independently selects simplified effects.
                    let image =
                        if image.width.max(image.height) > preview_limit && preview_limit > 0 {
                            Arc::new(spektrafilm_core::resize::preview(&image, preview_limit))
                        } else {
                            image
                        };
                    let scale_ms = t_scale.elapsed().as_secs_f32() * 1000.;
                    let input_preview = image.clone();
                    let t_build = Instant::now();
                    let cache_key =
                        preview_pipeline_cache_key(&data_dir, &film_name, &print_name, &params);
                    let pipeline = match pipeline_template {
                        Some((key, pipeline)) if key == cache_key => pipeline.with_params(params),
                        _ => {
                            let film = profile::load_profile_by_name(&data_dir, &film_name)
                                .map_err(|e| format!("film profile: {e}"))?;
                            let print = profile::load_profile_by_name(
                                &data_dir,
                                if params.scans_film() {
                                    &film_name
                                } else {
                                    &print_name
                                },
                            )
                            .map_err(|e| format!("print profile: {e}"))?;
                            Pipeline::new_with_spectral(film, print, params, &data_dir)
                                .map_err(|e| format!("pipeline build: {e}"))?
                        }
                    };
                    let pipeline_build_ms = t_build.elapsed().as_secs_f32() * 1000.;
                    drop(cpu);
                    worker_task.check().map_err(|e| e.to_string())?;
                    let cpu_render = if backend.is_gpu() {
                        None
                    } else {
                        Some(worker_task.cpu().map_err(|e| e.to_string())?)
                    };
                    let t = Instant::now();
                    let mut path = if backend.is_gpu() {
                        "GPU stages"
                    } else {
                        "CPU"
                    }
                    .to_string();
                    if backend.is_gpu()
                        && pipeline.params.film_render.grain.active
                        && !pipeline.params.converts_film()
                        && pipeline.params.film_render.grain.sublayers_active
                        && !backend.supports_layered_grain()
                    {
                        path.push_str(" + CPU sublayer grain");
                    }
                    let output = match () {
                        () if pipeline.params.io.crop
                            || (pipeline.params.io.upscale_factor - 1.).abs() > 1e-6
                            || (pipeline.params.scans_film() && pipeline.film.is_positive()) =>
                        {
                            pipeline.process((*image).clone(), backend.as_ref())
                        }
                        () => {
                            if let Some(output) =
                                pipeline.process_resident_borrowed(image.as_ref(), backend.as_ref())
                            {
                                path = "GPU resident".into();
                                output
                            } else {
                                pipeline.process((*image).clone(), backend.as_ref())
                            }
                        }
                    };
                    let render_ms = t.elapsed().as_secs_f32() * 1000.0;
                    drop(cpu_render);
                    let _cpu = worker_task.cpu().map_err(|e| e.to_string())?;
                    let t_preview = Instant::now();
                    let preview = prepare_preview_texture_data(
                        &output,
                        &pipeline.params.io,
                        display_transform,
                        backend.as_ref(),
                    );
                    let source_preview = if prepare_source {
                        let source = if source_image.width.max(source_image.height) > 2048 {
                            Arc::new(spektrafilm_core::resize::preview(&source_image, 2048))
                        } else {
                            source_image
                        };
                        let mut source_io = pipeline.params.io.clone();
                        source_io.output_color_space = source_key.0.clone();
                        source_io.output_cctf_encoding = source_key.1;
                        Some((
                            source_key,
                            prepare_preview_texture_data(
                                &source,
                                &source_io,
                                display_transform,
                                backend.as_ref(),
                            ),
                        ))
                    } else {
                        None
                    };
                    let paper = paper_back::cached(
                        input_preview.width as usize,
                        input_preview.height as usize,
                    );
                    let preview_ms = t_preview.elapsed().as_secs_f32() * 1000.0;
                    let worker_total_ms = t_total.elapsed().as_secs_f32() * 1000.0;
                    eprintln!(
                        "[spektrafilm] preview {}x{}: resize {:.1} ms, render {:.1} ms, {path}",
                        output.width, output.height, scale_ms, render_ms
                    );
                    Ok(RenderResult {
                        output,
                        input_preview,
                        preview_limit,
                        path,
                        preview,
                        source_preview,
                        paper,
                        input_clone_ms,
                        scale_ms,
                        pipeline_build_ms,
                        render_ms,
                        preview_ms,
                        worker_total_ms,
                        pipeline_cache: (cache_key, pipeline),
                    })
                }))
                .unwrap_or_else(|panic| Err(panic_message(panic)));
                ctx_for_worker.request_repaint();
                let _ = tx.send(result);
            })
            .expect("OS thread spawn");
        self.render_job = Some(RenderJob {
            discard: false,
            task,
            started: Instant::now(),
            interactive,
            rx,
            handle: Some(handle),
        });
    }

    /// Central-panel preview with mouse-wheel zoom (centred on the
    /// cursor) and click-drag pan. Double-click resets to fit.
    /// `self.zoom = 1.0` means fit-to-panel; values >1 zoom in.
    fn draw_preview_with_zoom(&mut self, ui: &mut egui::Ui) {
        if self.crop_tool.is_some() {
            self.crop_canvas(ui);
            return;
        }
        // Holding "before" shows the unprocessed input in place of the render.
        let held_display = self.compare_hold.then(|| {
            let saved = self.display.clone();
            self.display.show_input = true;
            self.display.input_opacity = 1.0;
            self.display.show_output = false;
            self.display.compare_split = false;
            saved
        });
        let avail = ui.available_size();
        let source_size = self
            .canvas_input
            .as_ref()
            .map(|i| egui::vec2(i.size[0] as f32, i.size[1] as f32))
            .or_else(|| self.input_tex.as_ref().map(|t| t.size_vec2()));
        let output_size = self
            .canvas_output
            .as_ref()
            .map(|i| egui::vec2(i.size[0] as f32, i.size[1] as f32))
            .or_else(|| self.output_tex.as_ref().map(|t| t.size_vec2()));
        let img_size = source_size.or(output_size).unwrap_or(egui::Vec2::splat(1.));
        let border = img_size.x.max(img_size.y) * self.display.white_padding;
        let paper_size = img_size + egui::Vec2::splat(2. * border);
        // Fit inside the margins and above the floating toolbar.
        let area = Self::image_area(egui::Rect::from_min_size(egui::Pos2::ZERO, avail));
        let fit_scale = (area.width() / paper_size.x)
            .min(area.height() / paper_size.y)
            .min(1.0);
        let scale = fit_scale * self.zoom;
        let drawn_size = img_size * scale;
        if let Some(image) = &self.image {
            let ppp = ui.ctx().pixels_per_point();
            self.view_actual_zoom = image.width.max(image.height) as f32
                / (ppp * img_size.x.max(img_size.y) * fit_scale);
        }

        let response = ui.allocate_response(avail, egui::Sense::click_and_drag());
        if let Some(path) = &self.image_path
            && let Some(error) = file_menu::show(&response, path)
        {
            self.status = error;
        }
        let rect = response.rect;
        let fit_centre = rect.min + area.center().to_vec2();

        // Two zoom sources, never combined:
        //   1. trackpad pinch (`zoom_delta` — already a multiplier
        //      around 1.0, smoothed by egui per frame)
        //   2. mouse wheel scroll (`smooth_scroll_delta.y` — egui's
        //      per-frame integrated scroll). Don't add `raw_scroll_delta`
        //      on top; `smooth` already accumulates the raw events.
        // Both are anchored on the cursor so the pixel under the
        // pointer stays put.
        let (pinch, scroll) = ui.input(|i| {
            if response.hovered() {
                (i.zoom_delta(), i.smooth_scroll_delta.y)
            } else {
                (1.0, 0.0)
            }
        });
        // Scroll → small log-zoom step per notch. 0.0015 keeps a single
        // trackpad swipe feeling like a smooth zoom rather than a leap.
        let scroll_factor = if scroll.abs() > 0.01 {
            (scroll * 0.0015).exp()
        } else {
            1.0
        };
        let factor = pinch * scroll_factor;
        if (factor - 1.0).abs() > 1e-4 {
            let cursor = ui.input(|i| i.pointer.hover_pos());
            let old_zoom = self.zoom;
            self.zoom = (self.zoom * factor).clamp(0.1, 32.0);
            if let Some(cur) = cursor {
                // Keep the image point under the cursor anchored.
                let centre = fit_centre + self.pan;
                let offset_from_centre = cur - centre;
                let scale_ratio = self.zoom / old_zoom;
                self.pan += offset_from_centre * (1.0 - scale_ratio);
            }
        }

        // In split view, dragging near the divider moves it instead of panning.
        let split_rect = egui::Rect::from_center_size(fit_centre + self.pan, drawn_size);
        let split_x = split_rect.left() + split_rect.width() * self.display.split_position;
        let near_split =
            |p: egui::Pos2| (p.x - split_x).abs() < 8.0 && split_rect.y_range().contains(p.y);
        let split_id = response.id.with("split-drag");
        let mut dragging_split = self.display.compare_split
            && response.dragged_by(egui::PointerButton::Primary)
            && ui.data(|d| d.get_temp::<bool>(split_id)).unwrap_or(false);
        if self.display.compare_split && response.drag_started_by(egui::PointerButton::Primary) {
            dragging_split = ui
                .input(|i| i.pointer.press_origin())
                .is_some_and(near_split);
        }
        if self.display.compare_split
            && (dragging_split || response.hover_pos().is_some_and(near_split))
        {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        }
        ui.data_mut(|d| d.insert_temp(split_id, dragging_split));
        if dragging_split {
            if let Some(p) = response.interact_pointer_pos() {
                self.display.split_position =
                    ((p.x - split_rect.left()) / split_rect.width()).clamp(0.0, 1.0);
            }
        } else if response.dragged_by(egui::PointerButton::Primary) {
            self.pan += response.drag_delta();
        }
        if response.double_clicked() {
            self.zoom = 1.0;
            self.pan = egui::Vec2::ZERO;
        }

        let centre = fit_centre + self.pan;
        let source_rect = egui::Rect::from_center_size(centre, drawn_size);
        let out_size = output_size.unwrap_or(img_size);
        let out_scale = (drawn_size.x / out_size.x).min(drawn_size.y / out_size.y);
        let draw_rect = egui::Rect::from_center_size(centre, out_size * out_scale);
        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        // Clip to the central panel so panned-off pixels don't leak.
        let painter = ui.painter_at(rect);
        let paper_rect = if self.display.show_border {
            source_rect.expand(border * scale)
        } else {
            source_rect
        };
        // Lift the print off the canvas.
        painter.add(
            egui::epaint::Shadow {
                offset: [0, 6],
                blur: 28,
                spread: 0,
                color: egui::Color32::from_black_alpha(110),
            }
            .as_shape(paper_rect, 0.0),
        );
        if self.display.show_border && border > 0. {
            painter.rect_filled(paper_rect, 0., egui::Color32::WHITE);
        }
        if self.display.show_paper_back {
            if let (Some(canvas), Some(paper)) = (&self.canvas, &self.canvas_paper) {
                canvas.paint(&painter, source_rect, paper, None, 1., "spline36", None, 1.);
            } else if let Some(paper) = &self.paper_tex {
                painter.image(paper.id(), source_rect, uv, egui::Color32::WHITE);
            }
        }
        if self.display.show_input || self.display.compare_split {
            let source_painter = if self.display.compare_split {
                painter.with_clip_rect(
                    egui::Rect::from_min_max(
                        rect.min,
                        egui::pos2(
                            source_rect.left() + source_rect.width() * self.display.split_position,
                            rect.bottom(),
                        ),
                    )
                    .intersect(rect),
                )
            } else {
                painter.clone()
            };
            if let (Some(canvas), Some(input)) = (&self.canvas, &self.canvas_input) {
                canvas.paint(
                    &source_painter,
                    source_rect,
                    input,
                    None,
                    self.display.input_opacity,
                    &self.display.output_interpolation,
                    None,
                    1.,
                );
            } else if let Some(source) = &self.input_tex {
                source_painter.image(
                    source.id(),
                    source_rect,
                    uv,
                    egui::Color32::from_white_alpha(
                        (255. * self.display.input_opacity).round() as u8
                    ),
                );
            }
        }
        if output_size.is_some() && (self.display.show_output || self.display.compare_split) {
            let output_painter = if self.display.compare_split {
                painter.with_clip_rect(
                    egui::Rect::from_min_max(
                        egui::pos2(
                            source_rect.left() + source_rect.width() * self.display.split_position,
                            rect.top(),
                        ),
                        rect.max,
                    )
                    .intersect(rect),
                )
            } else {
                painter.clone()
            };
            if let (Some(canvas), Some(output)) = (&self.canvas, &self.canvas_output) {
                let (development, fade) = match self.animation {
                    Some((start, true)) => (
                        Some((start.elapsed().as_secs_f32() / 1.6).clamp(0., 1.)),
                        1.,
                    ),
                    Some((start, false)) => {
                        (None, (start.elapsed().as_secs_f32() / 0.32).clamp(0., 1.))
                    }
                    None => (None, 1.),
                };
                canvas.paint(
                    &output_painter,
                    draw_rect,
                    output,
                    self.canvas_previous.as_ref(),
                    self.display.output_opacity,
                    &self.display.output_interpolation,
                    development,
                    fade,
                );
            } else if let Some(tex) = &self.output_tex {
                output_painter.image(
                    tex.id(),
                    draw_rect,
                    uv,
                    egui::Color32::from_white_alpha(
                        (255. * self.display.output_opacity).round() as u8
                    ),
                );
            }
        }
        if self.display.compare_split {
            let x = source_rect.left() + source_rect.width() * self.display.split_position;
            painter.line_segment(
                [
                    egui::pos2(x, draw_rect.top()),
                    egui::pos2(x, draw_rect.bottom()),
                ],
                egui::Stroke::new(1., egui::Color32::WHITE),
            );
            painter.circle(
                egui::pos2(x, draw_rect.center().y),
                7.0,
                egui::Color32::from_black_alpha(140),
                egui::Stroke::new(1.5, egui::Color32::WHITE),
            );
            let top = draw_rect.top().max(rect.top()) + 10.0;
            canvas_tag(
                &painter,
                egui::pos2(x - 8.0, top),
                egui::Align2::RIGHT_TOP,
                "Before",
            );
            canvas_tag(
                &painter,
                egui::pos2(x + 8.0, top),
                egui::Align2::LEFT_TOP,
                "After",
            );
        }
        if let Some(saved) = held_display {
            self.display = saved;
            let top = source_rect.top().max(rect.top()) + 10.0;
            canvas_tag(
                &painter,
                egui::pos2(source_rect.left().max(rect.left()) + 10.0, top),
                egui::Align2::LEFT_TOP,
                "Before",
            );
        }
    }

    /// Called once per `update()`. If the in-flight render finished,
    /// upload the texture and unblock the next pass. If more changes
    /// arrived during the render, re-arm `dirty`.
    fn poll_render_job(&mut self, ctx: &egui::Context) {
        let Some(job) = self.render_job.as_mut() else {
            return;
        };
        let result = match job.rx.try_recv() {
            Ok(r) => r,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.render_job = None;
                self.output_image = None;
                self.status = "Render error: worker thread vanished".into();
                return;
            }
        };
        if let Some(h) = job.handle.take() {
            let _ = h.join();
        }
        let discard = job.discard;
        let interactive = job.interactive;
        self.render_job = None;
        if discard {
            self.dirty = true;
            self.pending_dirty = false;
            return;
        }
        match result {
            Ok(r) => {
                self.pipeline_cache_key = Some(r.pipeline_cache.0);
                self.pipeline_cache = Some(r.pipeline_cache.1);
                self.rendered_preview_limit = r.preview_limit;
                self.preview_input
                    .retain(|(limit, _)| *limit != r.preview_limit);
                self.preview_input.push((r.preview_limit, r.input_preview));
                if self.preview_input.len() > 2 {
                    self.preview_input.remove(0);
                }
                // Keep low-resolution frames flowing while dragging. Drop
                // obsolete high-resolution refinements and schedule the newest
                // controls; the source-image discard flag is handled above.
                if self.pending_dirty && r.preview_limit > 640 {
                    self.pending_dirty = false;
                    self.dirty = true;
                    self.dirty_since = Instant::now().checked_sub(PREVIEW_DEBOUNCE);
                    ctx.request_repaint();
                    return;
                }
                self.render_path = r.path;
                self.last_input_clone_ms = r.input_clone_ms;
                self.last_scale_ms = r.scale_ms;
                self.last_pipeline_build_ms = r.pipeline_build_ms;
                self.last_render_ms = r.render_ms;
                self.last_preview_ms = r.preview_ms;
                self.last_worker_total_ms = r.worker_total_ms;
                if let Some(canvas) = &self.canvas {
                    let first_frame =
                        self.canvas_output.is_none() && std::mem::take(&mut self.develop_next);
                    self.canvas_previous = self
                        .canvas_output
                        .take()
                        .filter(|old| old.size == r.preview.size);
                    self.animation = canvas_transition(
                        self.display.animate,
                        r.preview.size,
                        interactive || ctx.is_using_pointer(),
                        first_frame,
                        self.canvas_previous.is_some(),
                    )
                    .map(|develop| (Instant::now(), develop));
                    if let Some((key, source)) = &r.source_preview {
                        self.canvas_input = Some(canvas.upload(source));
                        self.source_display_key = Some(key.clone());
                    }
                    self.canvas_output = Some(canvas.upload(&r.preview));
                }
                if self.paper_size != Some(r.paper.size) {
                    if let Some(canvas) = &self.canvas {
                        self.canvas_paper = Some(canvas.upload(&r.paper));
                    } else {
                        self.paper_tex = Some(make_texture(
                            ctx,
                            PreviewTextureData {
                                size: r.paper.size,
                                rgba: r.paper.rgba.clone(),
                            },
                            false,
                        ));
                    }
                    self.paper_size = Some(r.paper.size);
                }
                // Upload a single canvas texture; egui textures are only needed
                // for the explicitly selected legacy Glow compositor.
                if self.canvas.is_none() {
                    if let Some((key, source)) = r.source_preview {
                        self.input_tex = Some(make_texture(
                            ctx,
                            source,
                            self.display.output_interpolation == "nearest",
                        ));
                        self.source_display_key = Some(key);
                    }
                    self.output_tex = Some(make_texture(
                        ctx,
                        r.preview,
                        self.display.output_interpolation == "nearest",
                    ));
                }
                if self.status.starts_with("Rendering preview") {
                    self.status.clear();
                }
                self.output_image = Some(r.output);
            }
            Err(msg) => {
                // The previous pixels do not represent the requested settings.
                // Invalidate the saveable result and stop the idle refinement
                // condition from retrying a failed frame on every UI update.
                self.output_image = None;
                eprintln!("[spektrafilm] render error: {msg}");
                self.status = format!("Render error: {msg}");
            }
        }
        if self.pending_dirty {
            self.pending_dirty = false;
            self.dirty = true;
            // The latest edit already waited for the in-flight frame. Do not
            // add another debounce interval before servicing it.
            self.dirty_since = Instant::now().checked_sub(PREVIEW_DEBOUNCE);
            ctx.request_repaint();
        }
    }

    fn image_save_name(&self) -> String {
        let extension = self.export_format.extensions()[0];
        match &self.image_path {
            Some(p) => {
                let stem = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("spektrafilm");
                format!("{stem}_{}_spektra.{extension}", self.film_name)
            }
            None => format!("spektrafilm.{extension}"),
        }
    }

    fn image_save_dialog(&self) -> Option<PathBuf> {
        rfd::FileDialog::new()
            .add_filter(self.export_format.label(), self.export_format.extensions())
            .add_filter("All images", &["jpg", "jpeg", "tif", "tiff", "png", "exr"])
            .set_file_name(self.image_save_name())
            .save_file()
    }

    fn export_metadata(
        &self,
        params: &RuntimeParams,
        crop: Crop,
    ) -> Result<spektrafilm_io::Metadata> {
        let mut raw = self.export_raw_options().clone();
        raw.lightroom_geometry = raw
            .lightroom_geometry
            .map(|geometry| geometry.orientation_only());
        self.metadata
            .with_edit_settings(&spektrafilm_io::EditSettings {
                film: &self.film_name,
                paper: &self.print_name,
                params,
                raw: &raw,
                rotation: self.rotation,
                crop,
            })
    }

    /// Save the rendered preview using the selected file format.
    fn save_dialog(&mut self) {
        let Some(out) = self.output_image.clone() else {
            self.status = "Nothing to save yet — load an image first.".into();
            return;
        };
        let Some(path) = self.image_save_dialog() else {
            return;
        };
        let t = Instant::now();
        let mut params = self.params.clone();
        params.settings.preview_max_size = self.rendered_preview_limit;
        let result = self
            .export_metadata(&params, self.applied_crop)
            .and_then(|metadata| {
                spektrafilm_io::save_converted(
                    &out,
                    &self.params.io.output_color_space,
                    self.params.io.output_cctf_encoding,
                    &path,
                    &self.current_save_options(),
                    &metadata,
                    &self.data_dir,
                )
            });
        match result {
            Ok(()) => {
                self.status = format!(
                    "Saved {} in {:.0} ms",
                    path.file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("(file)"),
                    t.elapsed().as_secs_f32() * 1000.0
                );
            }
            Err(e) => {
                self.status = format!("Save error: {e:#}");
            }
        }
    }

    /// Export the original image at full resolution. GPU is the default;
    /// the precision option selects the bundled f64 CPU CLI instead.
    fn export_dialog(&mut self, ctx: &egui::Context) {
        let Some(input_path) = self.image_path.clone() else {
            self.status = "Load an image before exporting.".into();
            return;
        };
        let cli_path = if self.gpu_export {
            None
        } else {
            match locate_f64_cli() {
                Ok(p) => Some(p),
                Err(e) => {
                    self.status = format!("Export: {e}");
                    return;
                }
            }
        };
        let Some(out_path) = self.image_save_dialog() else {
            return;
        };

        let film = self.film_name.clone();
        let paper = self.print_name.clone();
        let crop = self.committed_crop();
        let mut params = self.params.clone();
        params.settings.preview_mode = false;
        // The CLI scales grain for the crop itself.
        let cli_params = params.clone();
        params.camera.film_format_mm = self.pipeline_params(crop).camera.film_format_mm;
        let data_dir = self.data_dir.clone();
        let raw_options = self.export_raw_options().clone();
        let rotation = self.rotation;
        let save_options = self.current_save_options();
        let metadata = self.export_metadata(&cli_params, crop);
        // While cropping, `image` is the whole frame; export the committed crop.
        let image = if crop == self.applied_crop {
            self.image.clone().map(Ok)
        } else {
            self.uncropped.clone().map(Err)
        };
        let backend = self.backend.clone();
        let label = if cli_path.is_some() {
            "f64 CPU"
        } else if backend.is_gpu() {
            "GPU"
        } else {
            "CPU"
        };
        self.start_export_job(ctx, label, out_path.clone(), move |cancel| {
            if let Some(cli_path) = cli_path {
                return run_f64_export(
                    &cli_path,
                    &input_path,
                    &out_path,
                    &film,
                    &paper,
                    &cli_params,
                    &data_dir,
                    cancel,
                    &raw_options,
                    &save_options,
                    rotation,
                    crop,
                );
            }
            let metadata = metadata?;
            let image = match image.context("source image is still loading")? {
                Ok(image) => image,
                Err(frame) => Arc::new(crop.apply(&frame)?),
            };
            run_gpu_export(
                image.as_ref(),
                &out_path,
                &film,
                &paper,
                &params,
                &data_dir,
                backend.as_ref(),
                cancel,
                &save_options,
                &metadata,
            )
        });
    }

    fn export_raw_options(&self) -> &spektrafilm_io::RawOptions {
        self.loaded_raw_options
            .as_ref()
            .unwrap_or(&self.raw_options)
    }

    fn start_export_job<F>(&mut self, ctx: &egui::Context, label: &str, output: PathBuf, run: F)
    where
        F: FnOnce(&AtomicBool) -> Result<()> + Send + 'static,
    {
        if self.export_job.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let task = self.scheduler.task(Priority::Export);
        let worker_task = task.clone();
        let started_at = Instant::now();
        let handle = std::thread::Builder::new()
            .name("spektrafilm-export".into())
            .spawn(move || {
                let result = worker_task
                    .run(|| {
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            run(&worker_cancel)
                        }))
                    })
                    .unwrap_or_else(|panic| Err(anyhow::anyhow!(panic_message(panic))));
                let message = result
                    .map(|_| {
                        (
                            started_at.elapsed().as_secs_f32(),
                            output.display().to_string(),
                        )
                    })
                    .map_err(|e| format!("{e:#}"));
                ctx.request_repaint();
                let _ = tx.send(message);
            });
        match handle {
            Ok(handle) => {
                self.status = format!("Exporting ({label})…");
                self.export_job = Some(ExportJob {
                    rx,
                    handle: Some(handle),
                    cancel,
                    task,
                    started_at,
                    label: label.into(),
                });
            }
            Err(e) => self.status = format!("Export: could not start worker: {e}"),
        }
    }

    /// Request cancellation of the in-flight export. The worker thread
    /// observes the flag in its `try_wait` poll loop and SIGKILLs the
    /// child; `poll_export_job` then drains the resulting error message
    /// on the next `update()` tick.
    fn cancel_export(&mut self) {
        let Some(job) = self.export_job.as_ref() else {
            return;
        };
        if !job.cancel.swap(true, Ordering::SeqCst) {
            job.task.cancel();
            self.status = "Cancelling export…".into();
        }
    }

    /// Called once per `update()`. If the export is still in-flight,
    /// refreshes the status with elapsed time and requests a repaint a
    /// second from now (so the timer ticks without us busy-looping).
    /// If the job finished, drains the result into the status bar and
    /// joins the worker thread.
    fn poll_export_job(&mut self, ctx: &egui::Context) {
        let Some(job) = self.export_job.as_mut() else {
            return;
        };
        let msg = match job.rx.try_recv() {
            Ok(m) => m,
            Err(mpsc::TryRecvError::Empty) => {
                let secs = job.started_at.elapsed().as_secs();
                self.status = if job.cancel.load(Ordering::SeqCst) {
                    format!("Cancelling export… ({secs}s)")
                } else {
                    format!("Exporting ({})… ({secs}s)", job.label)
                };
                ctx.request_repaint_after(Duration::from_secs(1));
                return;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                self.status = "Export error: worker thread vanished".into();
                self.export_job = None;
                return;
            }
        };
        if let Some(h) = job.handle.take() {
            let _ = h.join();
        }
        let label = job.label.clone();
        self.export_job = None;
        self.lens_database_status = LensDatabaseStatus::detect();
        self.status = match msg {
            Ok((secs, name)) => format!("Exported ({label}) {name} in {secs:.1} s"),
            Err(e) if e.contains("cancelled") => "Export cancelled.".into(),
            Err(e) => format!("Export error: {e}"),
        };
    }

    /// Apply a new film stock: its stock defaults, its paired paper, and the
    /// route its type implies (slides scan directly, negatives print).
    fn film_changed(&mut self) {
        self.film_dev_times = Vec::new();
        self.params.film_render.chemistry.development_time = None;
        if let Ok(film) = profile::load_profile_by_name(&self.data_dir, &self.film_name) {
            // Slide/positive stocks have no print paper — they are scanned
            // directly. Negatives print onto their paired `target_print`.
            // Auto-follow the film type so switching stocks doesn't leave a
            // wrong (or, for a paperless slide, a failed) render.
            if !self.params.converts_film() && self.params.route() != "input" {
                self.params.io.scan_film = film.is_positive();
                self.params.workflow.route = if film.is_positive() {
                    "input > film > scan"
                } else {
                    "input > film > print > scan"
                }
                .into();
            }
            spektrafilm_core::pipeline::apply_stock_defaults(&film, &mut self.params);
            self.film_dev_times = film.data.development_time.clone();
            if let Some(target) = film.info.target_print.as_deref()
                && self.papers.iter().any(|p| p.stock == target)
                && self.print_name != target
            {
                self.print_name = target.to_string();
                self.print_dev_times = profile_dev_times(&self.data_dir, &self.print_name);
                self.params.print_render.chemistry.development_time = None;
            }
        }
        self.dirty = true;
    }

    fn controls_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        use inspector::{Section, Tab};
        // Control captions are hover targets, not selectable text; the separate
        // status bar remains selectable for copying errors.
        ui.style_mut().interaction.selectable_labels = false;
        let stages = inspector::Stages::of(&self.params);
        let stock = self.stock_params();
        let not_camera = if self.params.converts_film() {
            "Not used: this workflow starts from a scanned negative."
        } else {
            "Not used: this workflow bypasses film simulation."
        };
        const NOT_FILM: &str = "Not used: this workflow bypasses film simulation.";
        const NOT_PRINT: &str = "Not used: this workflow does not print.";
        const NOT_SCAN: &str = "Not used: this workflow bypasses the scanner.";

        // ── Exposure ────────────────────────────────────────────────────
        Section::new("Exposure", Tab::Input, tooltips::SECTION_EXPOSURE)
            .used(stages.camera, not_camera)
            .pointers(&[
                "/camera/auto_exposure",
                "/camera/auto_exposure_method",
                "/camera/exposure_compensation_ev",
                "/camera/film_format_mm",
                "/camera/lens_blur_um",
            ])
            .show(ui, |ui| {
                let mut changed = false;
                changed |= tooltips::checkbox(
                    ui,
                    &mut self.params.camera.auto_exposure,
                    "Auto exposure",
                    tooltips::CONTROL_AUTO_EXPOSURE,
                )
                .changed();
                if self.params.camera.auto_exposure {
                    let method = &mut self.params.camera.auto_exposure_method;
                    tooltips::combo(
                        ui,
                        egui::ComboBox::from_id_salt("Metering")
                            .selected_text(names::label(method)),
                        "Metering",
                        tooltips::MENU_METERING,
                        |ui| {
                            for opt in [
                                "average",
                                "median",
                                "center_weighted",
                                "partial",
                                "matrix",
                                "multi_zone",
                                "highlight_weighted",
                            ] {
                                changed |= ui
                                    .selectable_value(method, opt.to_string(), names::label(opt))
                                    .on_hover_text(tooltips::option(opt))
                                    .changed();
                            }
                        },
                    );
                }
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut self.params.camera.exposure_compensation_ev, -5.0..=5.0)
                        .stock(stock.camera.exposure_compensation_ev)
                        .step_by(0.1),
                    "EV compensation",
                    tooltips::CONTROL_EV_COMPENSATION,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut self.params.camera.film_format_mm, 4.0..=120.0)
                        .stock(stock.camera.film_format_mm)
                        .logarithmic(true),
                    "Film format (mm)",
                    tooltips::CONTROL_FILM_FORMAT_MM,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut self.params.camera.lens_blur_um, 0.0..=100.0)
                        .stock(stock.camera.lens_blur_um),
                    "Lens blur (µm)",
                    tooltips::CONTROL_LENS_BLUR_UM,
                )
                .changed();
                if changed {
                    self.dirty = true;
                }
            });

        Section::new("Tone", Tab::Input, tooltips::SECTION_TONE)
            .used(stages.camera, not_camera)
            .pointers(&["/camera/tone"])
            .show(ui, |ui| {
                let t = &mut self.params.camera.tone;
                let defaults = &stock.camera.tone;
                let mut changed = false;
                for (label, value, default, help) in [
                    (
                        "Highlights",
                        &mut t.highlights,
                        defaults.highlights,
                        tooltips::CONTROL_HIGHLIGHTS,
                    ),
                    (
                        "Shadows",
                        &mut t.shadows,
                        defaults.shadows,
                        tooltips::CONTROL_SHADOWS,
                    ),
                    (
                        "Whites",
                        &mut t.whites,
                        defaults.whites,
                        tooltips::CONTROL_WHITES,
                    ),
                    (
                        "Blacks",
                        &mut t.blacks,
                        defaults.blacks,
                        tooltips::CONTROL_BLACKS,
                    ),
                ] {
                    changed |= tooltips::labeled(
                        ui,
                        tooltips::slider(value, -100. ..=100.)
                            .stock(default)
                            .step_by(1.),
                        label,
                        help,
                    )
                    .changed();
                }
                if changed {
                    self.dirty = true;
                }
            });

        // ── Diffusion filter (lens) ─────────────────────────────────────
        let toggled = Section::new(
            "Diffusion filter (lens)",
            Tab::Input,
            tooltips::SECTION_DIFFUSION_FILTER_LENS,
        )
        .active(
            self.params.camera.diffusion_filter.active,
            tooltips::ENABLE_DIFFUSION,
        )
        .used(stages.camera, not_camera)
        .pointers(&["/camera/diffusion_filter"])
        .show(ui, |ui| {
            let df = &mut self.params.camera.diffusion_filter;
            let mut changed = false;
            tooltips::combo(
                ui,
                egui::ComboBox::from_id_salt("Family")
                    .selected_text(names::label(&df.filter_family)),
                "Family",
                tooltips::MENU_FAMILY,
                |ui| {
                    for fam in ["black_pro_mist", "glimmerglass", "pro_mist", "cinebloom"] {
                        changed |= ui
                            .selectable_value(
                                &mut df.filter_family,
                                fam.to_string(),
                                names::label(fam),
                            )
                            .on_hover_text(tooltips::option(fam))
                            .changed();
                    }
                },
            );
            changed |= tooltips::labeled(
                ui,
                tooltips::slider(&mut df.strength, 0.0..=2.0)
                    .stock(stock.camera.diffusion_filter.strength),
                "Strength",
                tooltips::CONTROL_STRENGTH,
            )
            .changed();
            changed |= tooltips::labeled(
                ui,
                tooltips::slider(&mut df.spatial_scale, 0.1..=3.0)
                    .stock(stock.camera.diffusion_filter.spatial_scale),
                "Spatial scale",
                tooltips::CONTROL_SPATIAL_SCALE,
            )
            .changed();
            changed |= tooltips::labeled(
                ui,
                tooltips::slider(&mut df.halo_warmth, -1.5..=1.5)
                    .stock(stock.camera.diffusion_filter.halo_warmth),
                "Halo warmth",
                tooltips::CONTROL_HALO_WARMTH,
            )
            .changed();
            changed |= tooltips::labeled(
                ui,
                tooltips::slider(&mut df.core_intensity, 0.0..=2.0)
                    .stock(stock.camera.diffusion_filter.core_intensity),
                "Core intensity",
                tooltips::CONTROL_CORE_INTENSITY,
            )
            .changed();
            changed |= tooltips::labeled(
                ui,
                tooltips::slider(&mut df.halo_intensity, 0.0..=2.0)
                    .stock(stock.camera.diffusion_filter.halo_intensity),
                "Halo intensity",
                tooltips::CONTROL_HALO_INTENSITY,
            )
            .changed();
            changed |= tooltips::labeled(
                ui,
                tooltips::slider(&mut df.bloom_intensity, 0.0..=2.0)
                    .stock(stock.camera.diffusion_filter.bloom_intensity),
                "Bloom intensity",
                tooltips::CONTROL_BLOOM_INTENSITY,
            )
            .changed();
            changed |= tooltips::labeled(
                ui,
                tooltips::slider(&mut df.halo_size, 0.1..=3.0)
                    .stock(stock.camera.diffusion_filter.halo_size),
                "Halo size",
                tooltips::CONTROL_HALO_SIZE,
            )
            .changed();
            changed |= tooltips::labeled(
                ui,
                tooltips::slider(&mut df.bloom_size, 0.1..=3.0)
                    .stock(stock.camera.diffusion_filter.bloom_size),
                "Bloom size",
                tooltips::CONTROL_BLOOM_SIZE,
            )
            .changed();
            if changed {
                self.dirty = true;
            }
        });
        if toggled {
            self.params.camera.diffusion_filter.active ^= true;
            self.dirty = true;
        }

        Section::new(
            "Taking filter",
            Tab::Input,
            "Measured transmission of a filter placed in front of the taking lens.",
        )
        .closed()
        .used(stages.camera, not_camera)
        .pointers(&["/camera/color_filter"])
        .show(ui, |ui| {
            let mut changed = false;
            tooltips::combo(
                ui,
                egui::ComboBox::from_id_salt("taking_filter")
                    .selected_text(names::label(&self.params.camera.color_filter)),
                "Color filter",
                "Measured transmission of a filter placed in front of the taking lens.",
                |ui| {
                    for name in [
                        "none", "hoya_x0", "hoya_x1", "hoya_y2", "hoya_ya3", "hoya_r1",
                    ] {
                        changed |= ui
                            .selectable_value(
                                &mut self.params.camera.color_filter,
                                name.into(),
                                names::label(name),
                            )
                            .changed();
                    }
                },
            );
            if changed {
                self.dirty = true;
            }
        });

        Section::new(
            "Scan conversion",
            Tab::Input,
            "Recover film dye densities from a scanned negative before printing or rescanning.",
        )
        .used(stages.convert, "Used only by the scanned-negative workflows.")
        .pointers(&["/film_render/convert"])
        .show(ui, |ui| {
            let p = &mut self.params.film_render.convert;
            let mut changed = false;
            tooltips::combo(
                ui,
                egui::ComboBox::from_id_salt("scan_illuminant").selected_text(&p.scan_illuminant),
                "Scan illuminant",
                "The physical light used to capture the scanned negative.",
                |ui| {
                    for name in spektrafilm_math::illuminants::names() {
                        changed |= ui
                            .selectable_value(&mut p.scan_illuminant, name.into(), name)
                            .changed();
                    }
                },
            );
            changed |= tooltips::labeled(
                ui,
                tooltips::slider(&mut p.exposure_compensation_ev, -8. ..=8.).stock(stock.film_render.convert.exposure_compensation_ev),
                "Scan exposure EV",
                "Align the scanned input's level to the physical scan model before inversion.",
            )
            .changed();
            changed |= tooltips::labeled(
                ui,
                tooltips::slider(&mut p.base_percentile, 90. ..=100.).stock(stock.film_render.convert.base_percentile),
                "Base percentile",
                "Use the brightest pixels above this luminance percentile to estimate clear film base.",
            )
            .changed();
            changed |= tooltips::labeled(
                ui,
                egui::TextEdit::singleline(&mut p.calibration).desired_width(f32::INFINITY),
                "Device matrix",
                "Nine row-major numbers correcting the scanner or camera color response before inversion.",
            )
            .changed();
            changed |= tooltips::checkbox(
                ui,
                &mut self.params.settings.use_convert_lut,
                "Conversion LUT",
                "Cache the inverse scan model in a 3D tetrahedral LUT for faster repeated previews.",
            )
            .changed();
            if self.params.settings.use_convert_lut {
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut self.params.settings.convert_lut_resolution, 9..=65).stock(stock.settings.convert_lut_resolution),
                    "Conversion LUT resolution",
                    "Samples per RGB axis of the cached inverse scan model.",
                )
                .changed();
            }
            if changed {
                self.dirty = true;
            }
            self.calibration_controls(ui);
        });

        self.raw_controls(ui);

        // ── Halation ────────────────────────────────────────────────────
        let toggled = Section::new("Halation", Tab::Film, tooltips::SECTION_HALATION)
            .active(
                self.params.film_render.halation.active,
                tooltips::ENABLE_HALATION,
            )
            .used(stages.camera, not_camera)
            .pointers(&["/film_render/halation"])
            .show(ui, |ui| {
                let h = &mut self.params.film_render.halation;
                let mut changed = false;
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut h.halation_amount, 0.0..=3.0)
                        .stock(stock.film_render.halation.halation_amount),
                    "Halation amount",
                    tooltips::CONTROL_HALATION_AMOUNT,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut h.halation_spatial_scale, 0.1..=5.0)
                        .stock(stock.film_render.halation.halation_spatial_scale),
                    "Halation scale",
                    tooltips::CONTROL_HALATION_SCALE,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut h.scatter_amount, 0.0..=3.0)
                        .stock(stock.film_render.halation.scatter_amount),
                    "Scatter amount",
                    tooltips::CONTROL_SCATTER_AMOUNT,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut h.scatter_spatial_scale, 0.1..=5.0)
                        .stock(stock.film_render.halation.scatter_spatial_scale),
                    "Scatter scale",
                    tooltips::CONTROL_SCATTER_SCALE,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut h.halation_n_bounces, 1..=5)
                        .stock(stock.film_render.halation.halation_n_bounces),
                    "Bounces",
                    tooltips::CONTROL_BOUNCES,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut h.halation_bounce_decay, 0.0..=1.0)
                        .stock(stock.film_render.halation.halation_bounce_decay),
                    "Bounce decay",
                    tooltips::CONTROL_BOUNCE_DECAY,
                )
                .changed();
                changed |= tooltips::checkbox(
                    ui,
                    &mut h.halation_renormalize,
                    "Renormalize",
                    tooltips::CONTROL_RENORMALIZE,
                )
                .changed();
                Section::nested(
                    "Physical emulsion parameters",
                    tooltips::SECTION_PHYSICAL_EMULSION_PARAMETERS,
                )
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new(
                            "Values for the red-, green- and blue-sensitive layers.",
                        )
                        .small()
                        .color(theme::TEXT_WEAK),
                    );
                    changed |= tooltips::channels(
                        ui,
                        "Scatter core (µm)",
                        tooltips::CONTROL_SCATTER_CORE_UM,
                        tooltips::RGB,
                        h.scatter_core_um.each_mut(),
                        0.0..=1000.0,
                        0.1,
                    )
                    .changed();
                    changed |= tooltips::channels(
                        ui,
                        "Scatter tail (µm)",
                        tooltips::CONTROL_SCATTER_TAIL_UM,
                        tooltips::RGB,
                        h.scatter_tail_um.each_mut(),
                        0.0..=2000.0,
                        0.5,
                    )
                    .changed();
                    changed |= tooltips::channels(
                        ui,
                        "Tail weight",
                        tooltips::CONTROL_TAIL_WEIGHT,
                        tooltips::RGB,
                        h.scatter_tail_weight.each_mut(),
                        0.0..=1.0,
                        0.01,
                    )
                    .changed();
                    changed |= tooltips::channels(
                        ui,
                        "Halation strength",
                        tooltips::CONTROL_HALATION_STRENGTH,
                        tooltips::RGB,
                        h.halation_strength.each_mut(),
                        0.0..=3.0,
                        0.001,
                    )
                    .changed();
                    changed |= tooltips::channels(
                        ui,
                        "First bounce (µm)",
                        tooltips::CONTROL_FIRST_BOUNCE_UM,
                        tooltips::RGB,
                        h.halation_first_sigma_um.each_mut(),
                        0.0..=2000.0,
                        0.5,
                    )
                    .changed();
                    changed |= tooltips::labeled(
                        ui,
                        egui::DragValue::new(&mut h.boost_ev)
                            .range(0.0..=8.0)
                            .speed(0.01),
                        "Highlight boost EV",
                        tooltips::CONTROL_HIGHLIGHT_BOOST_EV,
                    )
                    .changed();
                    changed |= tooltips::labeled(
                        ui,
                        egui::DragValue::new(&mut h.boost_range)
                            .range(0.0..=1.0)
                            .speed(0.01),
                        "Boost ramp (0–1)",
                        tooltips::CONTROL_BOOST_RAMP_0_1,
                    )
                    .changed();
                    changed |= tooltips::labeled(
                        ui,
                        egui::DragValue::new(&mut h.protect_ev)
                            .range(0.0..=16.0)
                            .speed(0.01),
                        "Protect EV",
                        tooltips::CONTROL_PROTECT_EV,
                    )
                    .changed();
                });
                if changed {
                    self.dirty = true;
                }
            });
        if toggled {
            self.params.film_render.halation.active ^= true;
            self.dirty = true;
        }

        // ── DIR couplers ────────────────────────────────────────────────
        let toggled = Section::new("DIR couplers", Tab::Film, tooltips::SECTION_DIR_COUPLERS)
            .active(self.params.film_render.dir_couplers.active, tooltips::ENABLE_DIR)
            .used(stages.camera, not_camera)
            .pointers(&["/film_render/dir_couplers"])
            .show(ui, |ui| {
                let d = &mut self.params.film_render.dir_couplers;
                let mut changed = false;
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut d.amount, 0.0..=2.0).stock(stock.film_render.dir_couplers.amount),
                    "Amount",
                    tooltips::CONTROL_AMOUNT,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut d.diffusion_size_um, 0.0..=100.0).stock(stock.film_render.dir_couplers.diffusion_size_um),
                    "Diffusion size (µm)",
                    tooltips::CONTROL_DIFFUSION_SIZE_UM,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut d.diffusion_tail_um, 0.0..=400.0).stock(stock.film_render.dir_couplers.diffusion_tail_um),
                    "Diffusion tail (µm)",
                    tooltips::CONTROL_DIFFUSION_TAIL_UM,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut d.diffusion_tail_weight, 0.0..=1.0).stock(stock.film_render.dir_couplers.diffusion_tail_weight),
                    "Tail weight",
                    tooltips::CONTROL_TAIL_WEIGHT_2,
                )
                .changed();
                changed |= tooltips::channels(
                    ui,
                    "Donor knee",
                    "Negative-film inhibitor release knee as a fraction of maximum density; larger values approach a linear donor.",
                    tooltips::RGB,
                    d.langmuir_donor_k_rgb.each_mut(),
                    0.01..=20.,
                    0.01,
                )
                .changed();
                changed |= tooltips::channels(
                    ui,
                    "Receiver knee",
                    "Reversal-film receiver saturation after diffusion, relative to its inhibitor ceiling.",
                    tooltips::RGB,
                    d.langmuir_receiver_k_rgb.each_mut(),
                    0.01..=20.,
                    0.01,
                )
                .changed();
                Section::nested("Inhibition matrix", tooltips::SECTION_INHIBITION_MATRIX)
                    .show(ui, |ui| {
                        changed |= tooltips::labeled(
                            ui,
                            egui::DragValue::new(&mut d.inhibition_samelayer)
                                .range(0.0..=4.0)
                                .speed(0.01),
                            "Same-layer scale",
                            tooltips::CONTROL_SAME_LAYER_SCALE,
                        )
                        .changed();
                        changed |= tooltips::labeled(
                            ui,
                            egui::DragValue::new(&mut d.inhibition_interlayer)
                                .range(0.0..=4.0)
                                .speed(0.01),
                            "Inter-layer scale",
                            tooltips::CONTROL_INTER_LAYER_SCALE,
                        )
                        .changed();
                        changed |= tooltips::channels(
                            ui,
                            "Self inhibition",
                            tooltips::DIR_SELF_INHIBITION,
                            tooltips::RGB,
                            d.gamma_samelayer_rgb.each_mut(),
                            0.0..=4.0,
                            0.001,
                        )
                        .changed();
                        for (name, values) in [
                            ("Red → green / blue", &mut d.gamma_interlayer_r_to_gb),
                            ("Green → red / blue", &mut d.gamma_interlayer_g_to_rb),
                            ("Blue → red / green", &mut d.gamma_interlayer_b_to_rg),
                        ] {
                            let [a, b] = values.each_mut();
                            let receivers = [
                                &tooltips::inhibition_receiver(name, 0)[..1],
                                &tooltips::inhibition_receiver(name, 1)[..1],
                            ];
                            let help = format!(
                                "{}\n\n{}\n{}",
                                tooltips::DIR_DONOR,
                                tooltips::inhibition(name, 0),
                                tooltips::inhibition(name, 1)
                            );
                            changed |= tooltips::channels(
                                ui,
                                name,
                                &help,
                                receivers,
                                [a, b],
                                0.0..=4.0,
                                0.001,
                            )
                            .changed();
                        }
                    });
                if changed {
                    self.dirty = true;
                }
            });
        if toggled {
            self.params.film_render.dir_couplers.active ^= true;
            self.dirty = true;
        }

        // ── Grain ───────────────────────────────────────────────────────
        let toggled = Section::new("Grain", Tab::Film, tooltips::SECTION_GRAIN)
            .active(self.params.film_render.grain.active, tooltips::ENABLE_GRAIN)
            .used(stages.camera, not_camera)
            .pointers(&["/film_render/grain"])
            .show(ui, |ui| {
                let g = &mut self.params.film_render.grain;
                let mut changed = false;
                changed |= tooltips::channels(
                    ui,
                    "RMS granularity",
                    "Peak density noise in thousandths, measured with a 48 µm aperture; higher values give coarser grain.",
                    tooltips::RGB,
                    g.rms_granularity.each_mut(),
                    0.1..=60.,
                    0.1,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut g.mult_usm_sigma, 0. ..=3.).stock(stock.film_render.grain.mult_usm_sigma),
                    "Grain sharpening σ",
                    "Blur radius in pixels used by the multiplicative grain sharpen.",
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut g.mult_usm_amount, 0. ..=3.).stock(stock.film_render.grain.mult_usm_amount),
                    "Grain sharpening amount",
                    "Sharpen absolute grain density while preserving each channel's mean density.",
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut g.blur, 0.0..=3.0).stock(stock.film_render.grain.blur),
                    "Post-blur σ",
                    tooltips::CONTROL_POST_BLUR_SIGMA,
                )
                .changed();
                changed |= tooltips::channels(
                    ui,
                    "Layer particle scale",
                    tooltips::LAYER_PARTICLE_SCALE,
                    ["1", "2", "3"],
                    g.particle_scale_sublayers.each_mut(),
                    0.1..=5.0,
                    0.01,
                )
                .changed();
                if changed {
                    self.dirty = true;
                }
            });
        if toggled {
            self.params.film_render.grain.active ^= true;
            self.dirty = true;
        }

        // ── Print curves (s023 morph) ───────────────────────────────────
        for (title, tab, m, used) in [
            (
                "Film chemistry",
                Tab::Film,
                &mut self.params.film_render.chemistry,
                stages.film,
            ),
            (
                "Print chemistry",
                Tab::Print,
                &mut self.params.print_render.chemistry,
                stages.print,
            ),
        ] {
            let sm = if tab == Tab::Film {
                &stock.film_render.chemistry
            } else {
                &stock.print_render.chemistry
            };
            let pointers: &[&str] = if tab == Tab::Film {
                &[
                    "/film_render/chemistry/active",
                    "/film_render/chemistry/gamma_factor",
                    "/film_render/chemistry/gamma_factor_fast",
                    "/film_render/chemistry/gamma_factor_slow",
                    "/film_render/chemistry/gamma_factor_red",
                    "/film_render/chemistry/gamma_factor_green",
                    "/film_render/chemistry/gamma_factor_blue",
                    "/film_render/chemistry/developer_exhaustion",
                ]
            } else {
                &[
                    "/print_render/chemistry/active",
                    "/print_render/chemistry/gamma_factor",
                    "/print_render/chemistry/gamma_factor_fast",
                    "/print_render/chemistry/gamma_factor_slow",
                    "/print_render/chemistry/gamma_factor_red",
                    "/print_render/chemistry/gamma_factor_green",
                    "/print_render/chemistry/gamma_factor_blue",
                    "/print_render/chemistry/developer_exhaustion",
                ]
            };
            let toggled = Section::new(
                title,
                tab,
                "Morph the fitted development curves and physical emulsion layers together.",
            )
            .closed()
            .active(m.active, tooltips::MORPH_PRINT_CURVES)
            .used(
                used,
                if tab == Tab::Film {
                    NOT_FILM
                } else {
                    NOT_PRINT
                },
            )
            .pointers(pointers)
            .show(ui, |ui| {
                let mut changed = false;
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut m.gamma_factor, 0.5..=2.0).stock(sm.gamma_factor),
                    "Gamma",
                    tooltips::CONTROL_GAMMA,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut m.gamma_factor_fast, 0.5..=2.0)
                        .stock(sm.gamma_factor_fast),
                    "Gamma fast",
                    tooltips::CONTROL_GAMMA_FAST,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut m.gamma_factor_slow, 0.5..=2.0)
                        .stock(sm.gamma_factor_slow),
                    "Gamma slow",
                    tooltips::CONTROL_GAMMA_SLOW,
                )
                .changed();
                changed |= tooltips::channels(
                    ui,
                    "Gamma per channel",
                    "Scale contrast in the red-, green- and blue-sensitive density curves.",
                    tooltips::RGB,
                    [
                        &mut m.gamma_factor_red,
                        &mut m.gamma_factor_green,
                        &mut m.gamma_factor_blue,
                    ],
                    0.5..=2.0,
                    0.005,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut m.developer_exhaustion, 0.0..=1.0)
                        .stock(sm.developer_exhaustion),
                    "Developer exhaustion",
                    tooltips::CONTROL_DEVELOPER_EXHAUSTION,
                )
                .changed();
                if changed {
                    self.dirty = true;
                }
            });
            if toggled {
                m.active ^= true;
                self.dirty = true;
            }
        }

        for (title, tab, base, film, used) in [
            (
                "Film base",
                Tab::Film,
                &mut self.params.film_render.base,
                true,
                stages.film,
            ),
            (
                "Print base",
                Tab::Print,
                &mut self.params.print_render.base,
                false,
                stages.print,
            ),
        ] {
            let sb = if film {
                &stock.film_render.base
            } else {
                &stock.print_render.base
            };
            let toggled = Section::new(title, tab, "Tune the stock's base and fog density.")
                .closed()
                .active(
                    base.active,
                    "Enable changes to the stock's base and fog density.",
                )
                .used(used, if film { NOT_FILM } else { NOT_PRINT })
                .pointers(if film {
                    &["/film_render/base"]
                } else {
                    &["/print_render/base"]
                })
                .show(ui, |ui| {
                    let mut changed = tooltips::labeled(
                        ui,
                        tooltips::slider(&mut base.scale, 0. ..=3.).stock(sb.scale),
                        "Base scale",
                        "Scale the stock's spectral base density.",
                    )
                    .changed();
                    changed |= tooltips::channels(
                        ui,
                        "Base color scale",
                        "Scale each color component of the spectral base density.",
                        tooltips::CMY,
                        [&mut base.cyan, &mut base.magenta, &mut base.yellow],
                        0. ..=3.,
                        0.005,
                    )
                    .changed();
                    if film {
                        changed |= tooltips::labeled(
                            ui,
                            tooltips::slider(&mut base.tilt, -1. ..=1.).stock(sb.tilt),
                            "Base tilt",
                            "Tilt the film base density across wavelength around 555 nm.",
                        )
                        .changed();
                    }
                    if changed {
                        self.dirty = true;
                    }
                });
            if toggled {
                base.active ^= true;
                self.dirty = true;
            }
        }

        // ── Enlarger ────────────────────────────────────────────────────
        Section::new("Enlarger", Tab::Print, tooltips::SECTION_ENLARGER)
            .used(stages.print, NOT_PRINT)
            .pointers(&[
                "/enlarger/print_exposure",
                "/enlarger/m_filter_shift",
                "/enlarger/y_filter_shift",
                "/enlarger/preflash_exposure",
                "/enlarger/preflash_m_filter_shift",
                "/enlarger/preflash_y_filter_shift",
            ])
            .show(ui, |ui| {
                let e = &mut self.params.enlarger;
                let mut changed = false;
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut e.print_exposure, 0.1..=5.0)
                        .stock(stock.enlarger.print_exposure)
                        .logarithmic(true),
                    "Print exposure",
                    tooltips::CONTROL_PRINT_EXPOSURE,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut e.m_filter_shift, -50.0..=50.0)
                        .stock(stock.enlarger.m_filter_shift),
                    "Magenta filter shift",
                    tooltips::CONTROL_MAGENTA_FILTER_SHIFT,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut e.y_filter_shift, -50.0..=50.0)
                        .stock(stock.enlarger.y_filter_shift),
                    "Yellow filter shift",
                    tooltips::CONTROL_YELLOW_FILTER_SHIFT,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut e.preflash_exposure, 0.0..=0.5)
                        .stock(stock.enlarger.preflash_exposure),
                    "Preflash exposure",
                    tooltips::PREFLASH,
                )
                .changed();
                if e.preflash_exposure > 0.0 {
                    changed |= tooltips::labeled(
                        ui,
                        tooltips::slider(&mut e.preflash_m_filter_shift, -50.0..=50.0)
                            .stock(stock.enlarger.preflash_m_filter_shift),
                        "Preflash magenta shift",
                        tooltips::CONTROL_PREFLASH_MAGENTA_SHIFT,
                    )
                    .changed();
                    changed |= tooltips::labeled(
                        ui,
                        tooltips::slider(&mut e.preflash_y_filter_shift, -50.0..=50.0)
                            .stock(stock.enlarger.preflash_y_filter_shift),
                        "Preflash yellow shift",
                        tooltips::CONTROL_PREFLASH_YELLOW_SHIFT,
                    )
                    .changed();
                }
                if changed {
                    self.dirty = true;
                }
                self.print_balance_controls(ui);
            });

        // ── Glare ───────────────────────────────────────────────────────
        let toggled = Section::new("Glare", Tab::Print, tooltips::SECTION_GLARE)
            .active(
                self.params.print_render.glare.active,
                tooltips::ENABLE_GLARE,
            )
            .used(stages.print, NOT_PRINT)
            .pointers(&["/print_render/glare"])
            .show(ui, |ui| {
                let g = &mut self.params.print_render.glare;
                let mut changed = false;
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut g.percent, 0.0..=0.2)
                        .stock(stock.print_render.glare.percent),
                    "Percent",
                    tooltips::CONTROL_PERCENT,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut g.roughness, 0.0..=2.0)
                        .stock(stock.print_render.glare.roughness),
                    "Roughness",
                    tooltips::CONTROL_ROUGHNESS,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut g.blur, 0.0..=5.0).stock(stock.print_render.glare.blur),
                    "Blur σ (px)",
                    tooltips::CONTROL_BLUR_SIGMA_PX,
                )
                .changed();
                if changed {
                    self.dirty = true;
                }
            });
        if toggled {
            self.params.print_render.glare.active ^= true;
            self.dirty = true;
        }

        // ── Scanner ─────────────────────────────────────────────────────
        if self.tab == Tab::Scan {
            ui.add_space(4.0);
            let mut scan_for_print = self.scan_for_print_restore.is_some();
            if tooltips::checkbox(
                ui,
                &mut scan_for_print,
                "Scan for print",
                tooltips::SCAN_FOR_PRINT,
            )
            .changed()
            {
                if scan_for_print {
                    self.scan_for_print_restore = Some([
                        self.params.scanner.white_correction,
                        self.params.scanner.black_correction,
                        self.params.print_render.glare.active,
                    ]);
                    self.params.scanner.white_correction = true;
                    self.params.scanner.black_correction = true;
                    self.params.print_render.glare.active = false;
                } else if let Some([white, black, glare]) = self.scan_for_print_restore.take() {
                    self.params.scanner.white_correction = white;
                    self.params.scanner.black_correction = black;
                    self.params.print_render.glare.active = glare;
                }
                self.dirty = true;
            }
        }
        Section::new("Scanner", Tab::Scan, tooltips::SECTION_SCANNER)
            .used(stages.scan, NOT_SCAN)
            .pointers(&["/scanner"])
            .show(ui, |ui| {
                let s = &mut self.params.scanner;
                let mut changed = false;
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut s.lens_blur, 0.0..=5.0).stock(stock.scanner.lens_blur),
                    "Lens blur σ (px)",
                    tooltips::CONTROL_LENS_BLUR_SIGMA_PX,
                )
                .changed();
                let [mut sigma, mut amount] = s.unsharp_mask;
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut sigma, 0.0..=3.0).stock(stock.scanner.unsharp_mask[0]),
                    "Unsharp σ (px)",
                    tooltips::CONTROL_UNSHARP_SIGMA_PX,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut amount, 0.0..=2.0).stock(stock.scanner.unsharp_mask[1]),
                    "Unsharp amount",
                    tooltips::CONTROL_UNSHARP_AMOUNT,
                )
                .changed();
                if changed {
                    s.unsharp_mask = [sigma, amount];
                }
                changed |= tooltips::checkbox(
                    ui,
                    &mut s.white_correction,
                    "White correction",
                    tooltips::CONTROL_WHITE_CORRECTION,
                )
                .changed();
                if s.white_correction {
                    changed |= tooltips::labeled(
                        ui,
                        tooltips::slider(&mut s.white_level, 0.5..=1.0)
                            .stock(stock.scanner.white_level),
                        "White level",
                        tooltips::CONTROL_WHITE_LEVEL,
                    )
                    .changed();
                }
                changed |= tooltips::checkbox(
                    ui,
                    &mut s.black_correction,
                    "Black correction",
                    tooltips::CONTROL_BLACK_CORRECTION,
                )
                .changed();
                if s.black_correction {
                    changed |= tooltips::labeled(
                        ui,
                        tooltips::slider(&mut s.black_level, 0.0..=0.5)
                            .stock(stock.scanner.black_level),
                        "Black level",
                        tooltips::CONTROL_BLACK_LEVEL,
                    )
                    .changed();
                }
                if changed {
                    self.dirty = true;
                }
            });

        // ── Color management ────────────────────────────────────────────
        Section::new("Color management", Tab::Output, tooltips::SECTION_COLOR_MANAGEMENT)
            .pointers(&[
                "/io/output_gamut_compress",
                "/io/input_gamut_compress",
                "/settings/use_cat16",
                "/settings/rgb_to_raw_method",
            ])
            .show(ui, |ui| {
                let algo = &mut self.params.io.output_gamut_compress.algorithm;
                let mut changed = false;
                // Only the ported modes are offered; cam16ucs is upstream's
                // default, oklch the lighter perceptual option. Enabling either
                // drops the preview off the GPU-resident fast path.
                tooltips::combo(
                    ui,
                    egui::ComboBox::from_id_salt("Gamut compression").selected_text(names::label(algo)),
                    "Gamut compression",
                    tooltips::CONTROL_OUTPUT_GAMUT_COMPRESSION,
                    |ui| {
                        for opt in ["off", "oklch", "oklrab", "cam16ucs", "aces_rgc"] {
                            changed |= ui
                                .selectable_value(algo, opt.to_string(), names::label(opt))
                                .on_hover_text(tooltips::option(opt))
                                .changed();
                        }
                    },
                );
                // Input gamut compression — baked into the tc_lut at build time,
                // so changing it rebuilds the LUT on the next pass. "xy" is the
                // ACES-RGC-style radial compression toward the spectral locus.
                let in_algo = &mut self.params.io.input_gamut_compress.algorithm;
                tooltips::combo(
                    ui,
                    egui::ComboBox::from_id_salt("Input gamut compression")
                        .selected_text(names::label(in_algo)),
                    "Input gamut compression",
                    tooltips::CONTROL_INPUT_GAMUT_COMPRESSION,
                    |ui| {
                        for opt in ["off", "xy"] {
                            changed |= ui
                                .selectable_value(in_algo, opt.to_string(), names::label(opt))
                                .on_hover_text(tooltips::option(opt))
                                .changed();
                        }
                    },
                );
                if self.params.io.input_gamut_compress.algorithm != "off" {
                    let gamut = &mut self.params.io.input_gamut_compress;
                    tooltips::combo(
                        ui,
                        egui::ComboBox::from_id_salt("input_boundary")
                            .selected_text(names::label(&gamut.boundary)),
                        "Input boundary",
                        "Compress toward the spectral locus or its smooth inscribed hull.",
                        |ui| {
                            for name in ["inscribed_hull", "locus"] {
                                changed |= ui
                                    .selectable_value(&mut gamut.boundary, name.into(), names::label(name))
                                    .changed();
                            }
                        },
                    );
                    changed |= tooltips::labeled(
                        ui,
                        tooltips::slider(&mut gamut.hull_detail, 1. ..=30.).stock(stock.io.input_gamut_compress.hull_detail),
                        "Hull detail",
                        "Number of Fourier modes retained in the inscribed gamut boundary; higher values follow the locus more closely.",
                    )
                    .changed();
                }
                // CAT16 (vs CAT02) for the input chromatic adaptation feeding
                // Hanatos — better blue/violet behavior; matches upstream >=0.3.3.
                changed |= tooltips::checkbox(
                    ui,
                    &mut self.params.settings.use_cat16,
                    "CAT16 input adaptation",
                    tooltips::CONTROL_CAT16_INPUT_ADAPTATION,
                )
                .changed();
                // RGB → film raw spectral upsampler. hanatos2025 is the default
                // spectral LUT; arctic2026alpha02 is the new memory-color
                // reflectance LUT; mallett2019 is a faster matrix basis.
                let method = &mut self.params.settings.rgb_to_raw_method;
                tooltips::combo(
                    ui,
                    egui::ComboBox::from_id_salt("RGB→raw upsampling")
                        .selected_text(names::label(method)),
                    "RGB→raw upsampling",
                    tooltips::MENU_RGB_RAW_UPSAMPLING,
                    |ui| {
                        for opt in ["hanatos2025", "arctic2026beta04", "mallett2019", "jakob2019", "otsu2018", "gauss-lasers", "arctic2026alpha02"] {
                            changed |= ui
                                .selectable_value(method, opt.to_string(), names::label(opt))
                                .on_hover_text(tooltips::option(opt))
                                .changed();
                        }
                    },
                );
                if changed {
                    self.dirty = true;
                }
            });

        // ── Output ──────────────────────────────────────────────────────
        Section::new("Output", Tab::Output, tooltips::SECTION_OUTPUT)
            .pointers(&["/io/upscale_factor", "/io/output_cctf_encoding"])
            .show(ui, |ui| {
                let io = &mut self.params.io;
                let mut changed = false;
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut io.upscale_factor, 0.1..=1.0)
                        .stock(stock.io.upscale_factor),
                    "Working resolution",
                    tooltips::WORKING_RESOLUTION,
                )
                .changed();
                changed |= tooltips::checkbox(
                    ui,
                    &mut io.output_cctf_encoding,
                    "Encode output transfer function",
                    tooltips::CONTROL_ENCODE_OUTPUT_TRANSFER_FUNCTION,
                )
                .changed();
                if changed {
                    self.dirty = true;
                }
            });

        self.parity_controls(ui);
        if self.tab == Tab::Output {
            self.lut_controls(ui, ctx);
        }
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        self.store_settings(storage);
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // First-paint hook: tag the wgpu Metal layer's colorspace as sRGB.
        // Has to happen here (not in `App::new`) because the CAMetalLayer
        // isn't wired up at construction time.
        #[cfg(target_os = "macos")]
        if !self.metal_colorspace_tagged {
            match tag_metal_layer_srgb(frame) {
                Ok(()) => {
                    eprintln!("[spektrafilm] CAMetalLayer.colorspace = sRGB — tagged OK");
                    self.metal_colorspace_tagged = true;
                }
                Err(e) => {
                    eprintln!("[spektrafilm] colorspace tag attempt: {e}");
                }
            }
        }
        let _ = frame;

        self.poll_preload();
        self.poll_load_job();
        self.poll_calibration(ctx);
        // Refine an interactive 640 px frame after the slider is released.
        let configured_limit = if self.params.settings.preview_max_size == 0 {
            self.image
                .as_ref()
                .map(|i| i.width.max(i.height))
                .unwrap_or(0)
        } else {
            self.params.settings.preview_max_size
        };
        if !ctx.is_using_pointer()
            && self.render_job.is_none()
            && self.output_image.is_some()
            && self.rendered_preview_limit < configured_limit
        {
            self.dirty = true;
        }
        if self.dirty
            && (self.auto_preview || self.render_requested)
            && self.image.is_some()
            && !self.image_loading()
        {
            let now = Instant::now();
            let dirty_since = *self.dirty_since.get_or_insert(now);
            if self.render_job.is_some() {
                self.pending_dirty = true;
                self.dirty = false;
                self.dirty_since = None;
            } else {
                let elapsed = now.saturating_duration_since(dirty_since);
                let debounce = if ctx.is_using_pointer() && self.auto_preview {
                    INTERACTIVE_DEBOUNCE
                } else {
                    PREVIEW_DEBOUNCE
                };
                if elapsed >= debounce {
                    self.dispatch_render(ctx);
                    self.dirty = false;
                    self.dirty_since = None;
                } else {
                    ctx.request_repaint_after(debounce - elapsed);
                }
            }
        }
        if self.render_job.is_some() || self.image_loading() {
            ctx.request_repaint_after(IN_FLIGHT_REPAINT);
        }
        self.poll_render_job(ctx);
        if let Some((start, developing)) = self.animation {
            let duration = if developing { 1.6 } else { 0.32 };
            if !self.display.animate || start.elapsed().as_secs_f32() >= duration {
                self.animation = None;
                self.canvas_previous = None;
            } else {
                ctx.request_repaint_after(Duration::from_millis(16));
            }
        }
        self.poll_export_job(ctx);
        let raw_before = self.raw_options.clone();
        self.handle_shortcuts(ctx);
        self.begin_frame_state(ctx);
        self.update_title(ctx);
        self.menu_bar(ctx);
        self.status_bar(ctx);
        egui::SidePanel::right("controls")
            .resizable(true)
            .default_width(380.0)
            .width_range(330.0..=560.0)
            .frame(
                egui::Frame::side_top_panel(&ctx.style()).inner_margin(egui::Margin {
                    left: 12,
                    right: 4,
                    top: 4,
                    bottom: 0,
                }),
            )
            .show(ctx, |ui| {
                ui.style_mut().interaction.selectable_labels = false;
                egui::Frame::new()
                    .inner_margin(egui::Margin {
                        left: 0,
                        right: 8,
                        top: 0,
                        bottom: 0,
                    })
                    .show(ui, |ui| self.sidebar_header(ui));
                ui.separator();
                egui::TopBottomPanel::bottom("history")
                    .frame(egui::Frame::new().inner_margin(egui::Margin {
                        left: 0,
                        right: 8,
                        top: 4,
                        bottom: 6,
                    }))
                    .show_separator_line(true)
                    .show_inside(ui, |ui| self.history_panel(ui));
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
                    .show(ui, |ui| {
                        ui.add_space(2.0);
                        egui::Frame::new()
                            .inner_margin(egui::Margin {
                                left: 0,
                                right: 10,
                                top: 0,
                                bottom: 12,
                            })
                            .show(ui, |ui| self.controls_panel(ui, ctx));
                    });
            });
        let preload_radius = &mut self.preload_radius;
        let action = self.catalog.as_mut().and_then(|catalog| {
            egui::SidePanel::left("catalog")
                .default_width(236.)
                .width_range(200.0..=480.0)
                .frame(
                    egui::Frame::side_top_panel(&ctx.style()).inner_margin(egui::Margin {
                        left: 10,
                        right: 10,
                        top: 4,
                        bottom: 0,
                    }),
                )
                .show(ctx, |ui| {
                    ui.style_mut().interaction.selectable_labels = false;
                    catalog.show(ui, ctx, preload_radius)
                })
                .inner
        });
        match action {
            Some(catalog_ui::Action::Close) => {
                let request = self.preload.waiting().cloned();
                self.preload.stop();
                self.leave_photo();
                self.catalog = None;
                self.last_catalog = None;
                if let Some(request) = request {
                    self.start_image_load(request);
                }
            }
            Some(catalog_ui::Action::Select(photo)) => self.select_catalog_photo(photo),
            Some(catalog_ui::Action::Empty) => {
                self.preload.stop();
                self.leave_photo();
                if let Some(job) = &mut self.load_job {
                    job.discard();
                }
                if let Some(job) = &mut self.render_job {
                    job.discard();
                }
                self.pending_load = None;
                self.clear_image();
                self.status = "Catalog loaded. No available photos match the filter.".into();
            }
            None => {}
        }
        self.update_preloading();
        let canvas = if self.display.gray_18_canvas {
            egui::Color32::from_gray(118)
        } else {
            theme::WELL
        };
        egui::CentralPanel::default()
            .frame(
                egui::Frame::central_panel(&ctx.style())
                    .fill(canvas)
                    .inner_margin(0.0),
            )
            .show(ctx, |ui| {
                let rect = ui.max_rect();
                if self.has_canvas_image() {
                    self.draw_preview_with_zoom(ui);
                } else {
                    self.empty_canvas(ui);
                }
                self.canvas_overlays(ui, rect);
            });
        self.end_frame_state();

        // Accept drag-and-dropped image files.
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if let Some(path) = dropped.into_iter().next() {
            self.load_image_from_path(&path);
        }
        self.reload_raw_if_changed(&raw_before);
        self.sync_crop();
    }

    /// Called once when the window is closing. If an export is still
    /// in-flight, set the cancel flag and join the worker thread so
    /// the child process is SIGKILLed before the GUI exits. Without
    /// this the f64 CLI orphans into its own process group and keeps
    /// hammering the CPU long after the window is gone.
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.preload.stop();
        self.leave_photo();
        let Some(job) = self.export_job.take() else {
            return;
        };
        job.cancel.store(true, Ordering::SeqCst);
        job.task.cancel();
        if let Some(h) = job.handle {
            let _ = h.join();
        }
    }
}

/// Extract the human-readable payload from a caught panic. Panics carry
/// either a `&str` (literal messages) or a `String` (formatted ones);
/// anything else gets a generic label.
fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    let panic = panic.as_ref();
    if let Some(s) = panic.downcast_ref::<&str>() {
        format!("panic: {s}")
    } else if let Some(s) = panic.downcast_ref::<String>() {
        format!("panic: {s}")
    } else {
        "panic (no message)".to_string()
    }
}

/// Find the data directory. Probes, in order:
///   1. `$SPEKTRAFILM_DATA_DIR` — explicit override.
///   2. `<exe>/../Resources/data` — macOS `.app` bundle layout
///      (`Foo.app/Contents/MacOS/<exe>` → `Contents/Resources/data`).
///   3. `<exe>/data` — data shipped next to the binary.
///   4. `./data` — current working directory.
///   5. `<CARGO_MANIFEST_DIR>/../../data` — workspace root, for
///      `cargo run -p spektrafilm-gui`.
///
/// Finder launches apps with `cwd=/`, so the exe-relative probes (2–3)
/// matter for a double-clicked `.app`. Falls back to `./data` so the
/// downstream loader surfaces a clear "profiles not found" error.
fn pick_data_dir() -> PathBuf {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(dir) = std::env::var_os("SPEKTRAFILM_DATA_DIR") {
        candidates.push(PathBuf::from(dir));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        candidates.push(dir.join("..").join("Resources").join("data"));
        candidates.push(dir.join("data"));
    }
    candidates.push(PathBuf::from("data"));
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        candidates.push(PathBuf::from(manifest).join("..").join("..").join("data"));
    }
    candidates
        .into_iter()
        .find(|p| p.is_dir())
        .unwrap_or_else(|| PathBuf::from("data"))
}

/// Scan `<data_dir>/profiles/*.json`, parse each profile's `info`, and
/// bucket the results by `info.support`. Films go into the first vec,
/// papers (and any other print-stage supports) into the second.
/// Each entry carries the filename stem (the unique loader key) plus a
/// human-readable display label.
fn scan_profiles(data_dir: &Path) -> (Vec<ProfileEntry>, Vec<ProfileEntry>) {
    let mut films = Vec::new();
    let mut papers = Vec::new();
    let dir = data_dir.join("profiles");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return (films, papers);
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_lossy = name.to_string_lossy().to_string();
        let Some(stem) = name_lossy.strip_suffix(".json") else {
            continue;
        };
        let stock = stem.to_string();
        // Cheap probe: just deserialize the file's `info` field. We
        // could skip the rest of the profile but `Profile` already does
        // the right thing — and we pay this once at startup.
        let (display, is_paper, group) = match profile::load_profile_by_name(data_dir, &stock) {
            Ok(p) => {
                let display = p.info.name.clone().unwrap_or_else(|| stock.clone());
                let is_paper = p.info.support == "paper" || p.info.stage == "printing";
                (display, is_paper, chrome::stock_group(&p))
            }
            Err(_) => (stock.clone(), false, "Color negative"),
        };
        let entry = ProfileEntry {
            stock,
            display,
            group,
        };
        if is_paper {
            papers.push(entry);
        } else {
            films.push(entry);
        }
    }
    films.sort_by(|a, b| a.display.cmp(&b.display));
    papers.sort_by(|a, b| a.display.cmp(&b.display));
    (films, papers)
}

/// Development-time family of a profile (empty for colour stocks or on
/// load failure — the picker hides itself for ≤1 entries either way).
fn profile_dev_times(data_dir: &Path, stock: &str) -> Vec<f64> {
    profile::load_profile_by_name(data_dir, stock)
        .map(|p| p.data.development_time)
        .unwrap_or_default()
}

/// Development-time picker for a B&W development-time family. `selection`
/// of `None` means the profile's default (the floor-middle entry, matching
/// upstream's `select_development_time`). Returns true when changed.
fn dev_time_combo(
    ui: &mut egui::Ui,
    salt: &str,
    label: &str,
    times: &[f64],
    selection: &mut Option<f64>,
) -> bool {
    // Same resolution as the render path, so the combo always highlights
    // exactly the entry the pipeline will use.
    let current_idx = profile::development_time_index(times, *selection);
    let mut changed = false;
    tooltips::combo(
        ui,
        egui::ComboBox::from_id_salt(salt).selected_text(format!("{} min", times[current_idx])),
        label,
        tooltips::DEVELOPMENT_TIME,
        |ui| {
            for (i, t) in times.iter().enumerate() {
                if ui
                    .selectable_label(i == current_idx, format!("{t} min"))
                    .on_hover_text(tooltips::DEVELOPMENT_VARIANT)
                    .clicked()
                    && i != current_idx
                {
                    *selection = Some(*t);
                    changed = true;
                }
            }
        },
    );
    changed
}

/// How a freshly rendered preview replaces the previous one: `Some(true)`
/// develops it from blank paper, `Some(false)` crossfades, `None` swaps
/// immediately.
///
/// Only the first frame of a newly loaded image develops. Frames rendered for
/// a drag never animate, even when they land after the control is released,
/// and a resolution change (drag preview to full preview) swaps rather than
/// re-developing an image already on screen.
fn canvas_transition(
    animate: bool,
    size: [usize; 2],
    interactive: bool,
    first_frame: bool,
    previous_same_size: bool,
) -> Option<bool> {
    let small_enough = size[0] * size[1] <= 1_500_000;
    if !animate || !small_enough || interactive {
        None
    } else if first_frame {
        Some(true)
    } else if previous_same_size {
        Some(false)
    } else {
        None
    }
}

/// A small dark label pill drawn over the image.
fn canvas_tag(painter: &egui::Painter, pos: egui::Pos2, anchor: egui::Align2, text: &str) {
    let galley = painter.layout_no_wrap(
        text.to_owned(),
        egui::FontId::proportional(11.0),
        egui::Color32::WHITE,
    );
    let rect = anchor.anchor_size(pos, galley.size() + egui::vec2(12.0, 6.0));
    painter.rect_filled(rect, 4.0, egui::Color32::from_black_alpha(150));
    painter.galley(
        rect.min + egui::vec2(6.0, 3.0),
        galley,
        egui::Color32::WHITE,
    );
}

fn fps_label(ms: f32) -> &'static str {
    if ms < 16.0 {
        "60 fps"
    } else if ms < 33.0 {
        "30 fps"
    } else if ms < 67.0 {
        "15 fps"
    } else if ms < 200.0 {
        "5 fps"
    } else if ms < 400.0 {
        "2 fps"
    } else {
        ""
    }
}

fn preview_pipeline_cache_key(
    data_dir: &Path,
    film_name: &str,
    print_name: &str,
    params: &RuntimeParams,
) -> String {
    let params_key = serde_json::json!({
        "film": film_name,
        "print": print_name,
        "workflow":params.route(),
        "film_chemistry":params.film_render.chemistry,"print_chemistry":params.print_render.chemistry,
        "print_gamma":params.print_render.density_curve_gamma,
        "film_base":params.film_render.base,"print_base":params.print_render.base,
        "scan_illuminant":params.film_render.convert.scan_illuminant,
        "convert_lut":params.settings.use_convert_lut,"convert_resolution":params.settings.convert_lut_resolution,
        "taking_filter":params.camera.color_filter,"auto_exposure":params.camera.auto_exposure,"auto_exposure_method":params.camera.auto_exposure_method,
        "film_dev": params.film_render.development_time,
        "film_gamma":params.film_render.density_curve_gamma,
        "debug":params.debug,
        "print_dev": params.print_render.development_time,
        "scan_film": params.scans_film(),
        "film_channel_order":params.film_render.channel_order,"print_channel_order":params.print_render.channel_order,
        "input_color_space": params.io.input_color_space,
        "input_cctf_decoding": params.io.input_cctf_decoding,
        "filter_uv": params.camera.filter_uv,"filter_ir": params.camera.filter_ir,
        "input_gamut": params.io.input_gamut_compress,
        "output_color_space": params.io.output_color_space,
        "output_gamut": params.io.output_gamut_compress,
        "settings": {
            "apply_stock_specifics":params.settings.apply_stock_specifics,
            "rgb_to_raw_method": params.settings.rgb_to_raw_method,
            "apply_hanatos2025_adaptation_window": params.settings.apply_hanatos2025_adaptation_window,
            "apply_hanatos2025_adaptation_surface": params.settings.apply_hanatos2025_adaptation_surface,
            "spectral_gaussian_blur": params.settings.spectral_gaussian_blur,
            "lut_resolution": params.settings.lut_resolution,
            "neutral_print_filters_from_database": params.settings.neutral_print_filters_from_database,
            "use_cat16": params.settings.use_cat16,
        },
        "enlarger": {
            "illuminant": params.enlarger.illuminant,
            "c_filter_neutral": params.enlarger.c_filter_neutral,
            "m_filter_neutral": params.enlarger.m_filter_neutral,
            "y_filter_neutral": params.enlarger.y_filter_neutral,
            "m_filter_shift": params.enlarger.m_filter_shift,
            "y_filter_shift": params.enlarger.y_filter_shift,
            "preflash_exposure": params.enlarger.preflash_exposure,
            "preflash_m_filter_shift": params.enlarger.preflash_m_filter_shift,
            "preflash_y_filter_shift": params.enlarger.preflash_y_filter_shift,
            "normalize_print_exposure": params.enlarger.normalize_print_exposure,
            "print_exposure_compensation": params.enlarger.print_exposure_compensation,
        },
        "exposure_compensation_ev": if params.enlarger.print_exposure_compensation {
            params.camera.exposure_compensation_ev
        } else {
            0.0
        },
    })
    .to_string();
    let mut files: Vec<_> = [
        film_name,
        if params.scans_film() {
            film_name
        } else {
            print_name
        },
    ]
    .map(|stock| data_dir.join("profiles").join(format!("{stock}.json")))
    .into();
    files.push(data_dir.join("filters/neutral_print_filters.json"));
    let spectral = data_dir.join("luts/spectral_upsampling");
    if let Ok(entries) = std::fs::read_dir(&spectral) {
        files.extend(entries.filter_map(Result::ok).map(|entry| entry.path()));
    }
    files.push(spectral.join("arctic2026alpha02/reflectance_xy_tc.npy"));
    files.sort();
    files.dedup();
    let sources: Vec<_> = files.iter().map(|path| (path, image_stamp(path))).collect();
    format!("{params_key}:{:?}", (data_dir, sources))
}

fn pick_default_stock(entries: &[ProfileEntry], preferred: &str) -> String {
    if entries.iter().any(|e| e.stock == preferred) {
        preferred.to_string()
    } else {
        entries
            .first()
            .map(|e| e.stock.clone())
            .unwrap_or_else(|| preferred.to_string())
    }
}

fn prepare_preview_texture_data(
    out: &ImageBuf,
    io: &spektrafilm_core::params::IoParams,
    display_transform: bool,
    backend: &dyn ComputeBackend,
) -> PreviewTextureData {
    // Small previews and already encoded sRGB are cheaper to pack on the CPU.
    if out.pixel_count() >= 512 * 1024
        && display_transform
        && !(io.output_color_space == "sRGB" && io.output_cctf_encoding)
    {
        if let Some(preview) = backend.display_preview(
            out,
            &io.output_color_space,
            io.output_cctf_encoding,
            display_transform,
            PREVIEW_TEXTURE_MAX_DIM,
        ) {
            return PreviewTextureData {
                size: preview.size,
                rgba: preview.rgba,
            };
        }
    }
    make_preview_texture_data(out, io, display_transform)
}

/// Convert the selected output color transport to an sRGB RGBA display preview.
/// The full output remains available for saving, but the interactive texture is
/// capped so a large RAW does not force a full-size upload into egui every pass.
fn make_preview_texture_data(
    out: &ImageBuf,
    io: &spektrafilm_core::params::IoParams,
    display_transform: bool,
) -> PreviewTextureData {
    let converted;
    let out = if !display_transform || (io.output_color_space == "sRGB" && io.output_cctf_encoding)
    {
        out
    } else {
        let mut display = out.clone();
        let curve = spektrafilm_math::transfer::get(&io.output_color_space)
            .expect("validated output")
            .transfer;
        let matrix = spektrafilm_math::transfer::rgb_matrix(&io.output_color_space, "sRGB");
        display.data.par_chunks_exact_mut(3).for_each(|px| {
            let linear = std::array::from_fn::<_, 3, _>(|c| {
                if io.output_cctf_encoding {
                    curve.decode(px[c] as f64)
                } else {
                    px[c] as f64
                }
            });
            for c in 0..3 {
                px[c] = spektrafilm_math::precision::from_f64(
                    spektrafilm_math::transfer::Transfer::Srgb.encode(
                        matrix[c][0] * linear[0]
                            + matrix[c][1] * linear[1]
                            + matrix[c][2] * linear[2],
                    ),
                );
            }
        });
        converted = display;
        &converted
    };
    let src_w = out.width as usize;
    let src_h = out.height as usize;
    let max_src_dim = src_w.max(src_h).max(1);
    let scale_num = PREVIEW_TEXTURE_MAX_DIM.min(max_src_dim);
    let w = ((src_w * scale_num + max_src_dim / 2) / max_src_dim).max(1);
    let h = ((src_h * scale_num + max_src_dim / 2) / max_src_dim).max(1);
    let mut rgba = vec![0u8; w * h * 4];
    rgba.par_chunks_exact_mut(w * 4)
        .enumerate()
        .for_each(|(y, row)| {
            let gy = if h == src_h {
                y as f32
            } else {
                (y as f32 + 0.5) * (src_h as f32 / h as f32) - 0.5
            };
            let y0 = gy.floor().clamp(0.0, (src_h - 1) as f32) as usize;
            let y1 = (y0 + 1).min(src_h - 1);
            let ty = (gy - y0 as f32).clamp(0.0, 1.0);

            for x in 0..w {
                let gx = if w == src_w {
                    x as f32
                } else {
                    (x as f32 + 0.5) * (src_w as f32 / w as f32) - 0.5
                };
                let x0 = gx.floor().clamp(0.0, (src_w - 1) as f32) as usize;
                let x1 = (x0 + 1).min(src_w - 1);
                let tx = (gx - x0 as f32).clamp(0.0, 1.0);
                let p00 = (y0 * src_w + x0) * 3;
                let p10 = (y0 * src_w + x1) * 3;
                let p01 = (y1 * src_w + x0) * 3;
                let p11 = (y1 * src_w + x1) * 3;
                let dst = x * 4;

                for c in 0..3 {
                    let v00 = to_f32(out.data[p00 + c]);
                    let v10 = to_f32(out.data[p10 + c]);
                    let v01 = to_f32(out.data[p01 + c]);
                    let v11 = to_f32(out.data[p11 + c]);
                    let top = v00 + (v10 - v00) * tx;
                    let bot = v01 + (v11 - v01) * tx;
                    row[dst + c] = ((top + (bot - top) * ty).clamp(0.0, 1.0) * 255.0).round() as u8;
                }
                row[dst + 3] = 255;
            }
        });
    PreviewTextureData { size: [w, h], rgba }
}

/// Upload an sRGB display preview prepared independently of saved image pixels.
fn make_texture(
    ctx: &egui::Context,
    preview: PreviewTextureData,
    nearest: bool,
) -> egui::TextureHandle {
    let color_image = egui::ColorImage::from_rgba_unmultiplied(preview.size, &preview.rgba);
    ctx.load_texture(
        "spektrafilm-output",
        color_image,
        if nearest {
            egui::TextureOptions::NEAREST
        } else {
            egui::TextureOptions::LINEAR
        },
    )
}

/// Locate the f64-built `spektrafilm` CLI binary. Search order:
///   1. `$SPEKTRAFILM_F64_CLI` — explicit override, full path.
///   2. `spektrafilm-f64` next to the running GUI executable (release
///      builds: both binaries live in `target/release/`).
///   3. `spektrafilm-f64` on `$PATH`.
///   4. `target/release/spektrafilm-f64` relative to `CARGO_MANIFEST_DIR`
///      (handy when running via `cargo run`).
///
/// Returns a path that exists and is executable, or an explanatory
/// error pointing to the build command.
fn locate_f64_cli() -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("SPEKTRAFILM_F64_CLI") {
        let path = PathBuf::from(&p);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "SPEKTRAFILM_F64_CLI points at {p} but no such file"
        ));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        for name in f64_cli_names() {
            let p = dir.join(name);
            if p.is_file() {
                return Ok(p);
            }
        }
    }
    for name in f64_cli_names() {
        if let Some(p) = which_on_path(name) {
            return Ok(p);
        }
    }
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        for name in f64_cli_names() {
            let p = PathBuf::from(&manifest)
                .join("..")
                .join("..")
                .join("target")
                .join("release")
                .join(name);
            if p.is_file() {
                return Ok(p);
            }
        }
    }
    Err("no f64 CLI found. Build it with \
         `cargo build --release --features precision-f64 -p spektrafilm-cli`, \
         rename `target/release/spektrafilm` to `spektrafilm-f64`, \
         and either put it on PATH or set $SPEKTRAFILM_F64_CLI."
        .into())
}

fn f64_cli_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["spektrafilm-f64.exe", "spektrafilm-f64"]
    } else {
        &["spektrafilm-f64"]
    }
}

/// Minimal PATH lookup so we don't pull in the `which` crate for one call.
fn which_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// RAII guard so the params tempfile is unlinked on every code path
/// (early `?` return, panic during `Command::output`, normal exit).
/// Without this, a failed `serde_json::to_writer` or a panic mid-spawn
/// would leak user-state JSON into the OS temp dir.
struct TempPath(PathBuf);

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Run the f64 CLI as a child process. Writes the current
/// `RuntimeParams` to a temp JSON file, invokes
/// `spektrafilm-f64 process …`, polls for completion while watching
/// `cancel`, and surfaces the CLI's stderr verbatim on failure. If
/// `cancel` is set the child is SIGKILLed and `Err("cancelled")` is
/// returned — this prevents the orphan-process / system-freeze bug
/// where closing the GUI mid-export left the CPU pipeline running.
fn run_f64_export(
    cli_path: &Path,
    input: &Path,
    output: &Path,
    film: &str,
    paper: &str,
    params: &spektrafilm_core::params::RuntimeParams,
    data_dir: &Path,
    cancel: &AtomicBool,
    raw_options: &spektrafilm_io::RawOptions,
    save_options: &spektrafilm_io::SaveOptions,
    rotation: u8,
    crop: Crop,
) -> Result<()> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let temp = TempPath(std::env::temp_dir().join(format!(
        "spektrafilm-export-{}-{nanos}.json",
        std::process::id()
    )));
    {
        let f = std::fs::File::create(&temp.0)
            .with_context(|| format!("creating params tempfile {}", temp.0.display()))?;
        serde_json::to_writer(std::io::BufWriter::new(f), params)
            .context("serializing params to JSON")?;
    }

    let mut cmd = std::process::Command::new(cli_path);
    // Force the CPU backend: the wgpu shaders run f32 even in a
    // precision-f64 build (WGSL has no f64). Letting the child default
    // to GPU would silently demote the export back to f32 math.
    cmd.env("SPEKTRAFILM_BACKEND", "cpu")
        .arg("process")
        .arg("--rotate-quarter-turns")
        .arg(rotation.to_string())
        .arg("--bit-depth")
        .arg(save_options.bit_depth.to_string())
        .arg("--jpeg-quality")
        .arg(save_options.jpeg_quality.to_string())
        .arg("--saving-color-space")
        .arg(&save_options.color_space)
        .arg("--saving-cctf-encoding")
        .arg(save_options.encoded.to_string())
        .arg("--raw-decoder")
        .arg(&raw_options.decoder)
        .arg("--white-balance")
        .arg(&raw_options.white_balance)
        .arg("--temperature")
        .arg(raw_options.temperature.to_string())
        .arg("--tint")
        .arg(raw_options.tint.to_string())
        .arg(input)
        .arg("-o")
        .arg(output)
        .arg("--film")
        .arg(film)
        .arg("--paper")
        .arg(paper)
        .arg("--params")
        .arg(&temp.0)
        .arg("--data-dir")
        .arg(data_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    if params.scans_film() {
        cmd.arg("--scan-film");
    }

    if raw_options.lens_correction {
        cmd.arg("--lens-correction");
    }
    if let Some(path) = &raw_options.lens_database {
        cmd.arg("--lens-database").arg(path);
    }
    // Lightroom's crop is already part of `crop`.
    if let Some(geometry) = &raw_options.lightroom_geometry {
        cmd.arg("--lightroom-geometry")
            .arg(serde_json::to_string(&geometry.orientation_only())?);
    }
    if !crop.is_full() {
        cmd.arg("--crop").arg(serde_json::to_string(&crop)?);
    }
    run_cli(&mut cmd, cancel)
}

/// Full-resolution export shares the same pipeline and saving conversion as the CLI.
#[allow(clippy::too_many_arguments)]
fn run_gpu_export(
    image: &ImageBuf,
    output: &Path,
    film_name: &str,
    paper_name: &str,
    params: &RuntimeParams,
    data_dir: &Path,
    backend: &dyn ComputeBackend,
    cancel: &AtomicBool,
    options: &spektrafilm_io::SaveOptions,
    metadata: &spektrafilm_io::Metadata,
) -> Result<()> {
    let cpu = scheduler::cpu_permit()?;
    let film = profile::load_profile_by_name(data_dir, film_name)?;
    let paper = profile::load_profile_by_name(
        data_dir,
        if params.scans_film() {
            film_name
        } else {
            paper_name
        },
    )?;
    let mut params = params.clone();
    params.settings.preview_mode = false;
    let pipeline = Pipeline::new_with_spectral(film, paper, params.clone(), data_dir)
        .map_err(anyhow::Error::msg)?;
    drop(cpu);
    if cancel.load(Ordering::SeqCst) {
        anyhow::bail!("cancelled");
    }
    let cpu_render = if backend.is_gpu() {
        None
    } else {
        scheduler::cpu_permit()?
    };
    let output_image = pipeline
        .process_resident_borrowed(image, backend)
        .unwrap_or_else(|| pipeline.process(image.clone(), backend));
    drop(cpu_render);
    if cancel.load(Ordering::SeqCst) {
        anyhow::bail!("cancelled");
    }
    let _cpu = scheduler::cpu_permit()?;
    spektrafilm_io::save_converted(
        &output_image,
        &params.io.output_color_space,
        params.io.output_cctf_encoding,
        output,
        options,
        metadata,
        data_dir,
    )
}

/// Drain both child pipes while polling, so verbose jobs cannot block on a full pipe.
fn run_cli(cmd: &mut std::process::Command, cancel: &AtomicBool) -> Result<()> {
    use std::io::Read;
    cmd.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn().context("starting export CLI")?;
    let mut stderr = child.stderr.take().context("opening export diagnostics")?;
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text);
        text
    });
    let result = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if !cancel.load(Ordering::SeqCst) => {
                std::thread::sleep(Duration::from_millis(100))
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(anyhow::anyhow!("cancelled"));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(e.into());
            }
        }
    };
    let stderr = reader.join().unwrap_or_default();
    let status = result?;
    if !status.success() {
        anyhow::bail!("export CLI exited {status}: {}", stderr.trim());
    }
    Ok(())
}

/// macOS only: walk from the eframe `RawWindowHandle` down to the
/// `CAMetalLayer` and tag its colorspace as sRGB. Without this the
/// metal layer's `colorspace` is null and macOS treats the framebuffer
/// pixels as raw display primaries — sRGB content rendered into a
/// wide-gamut panel comes out oversaturated. Setting the layer's
/// colorspace makes the OS gamut-map exactly like it does for a
/// regular sRGB-tagged PNG opened in Preview, so what you see in the
/// GUI matches what you export.
#[cfg(target_os = "macos")]
fn tag_metal_layer_srgb<H: raw_window_handle::HasWindowHandle>(h: &H) -> Result<(), String> {
    use core_graphics::color_space::{CGColorSpace, kCGColorSpaceSRGB};
    use foreign_types::ForeignType;
    use objc2::msg_send;
    use objc2::runtime::AnyObject;
    use raw_window_handle::RawWindowHandle;

    let handle = h
        .window_handle()
        .map_err(|e| format!("no window handle: {e}"))?;
    let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
        return Err("not an AppKit window".into());
    };
    let ns_view: *mut AnyObject = appkit.ns_view.as_ptr().cast();
    if ns_view.is_null() {
        return Err("ns_view is null".into());
    }

    // `kCGColorSpaceSRGB` is an extern static CFStringRef, accessing it
    // is unsafe in the 2024 edition's stricter model.
    let srgb_cs = unsafe { CGColorSpace::create_with_name(kCGColorSpaceSRGB) }
        .ok_or_else(|| "CGColorSpaceCreateWithName(sRGB) returned null".to_string())?;
    // `foreign_types::ForeignType::as_ptr` returns the opaque
    // `CGColorSpaceRef` (i.e. `*mut sys::CGColorSpace`). `setColorspace:`
    // is declared as taking a `CGColorSpaceRef`, whose ObjC type encoding
    // is `^{CGColorSpace=}`. A bare `*mut c_void` encodes as `^v`, which
    // objc2's debug-build argument-encoding verification rejects (the
    // release build skips that check, which is why only debug crashed).
    // Cast through a zero-field opaque struct whose `Encode` matches the
    // expected `^{CGColorSpace=}` so both builds pass the same pointer.
    #[repr(C)]
    struct CGColorSpaceOpaque {
        _private: [u8; 0],
    }
    // A pointer to this type encodes as `^{CGColorSpace=}` — what
    // `setColorspace:` expects. `RefEncode` (not `Encode`) is the trait
    // objc2 consults for a `*const T` argument.
    unsafe impl objc2::RefEncode for CGColorSpaceOpaque {
        const ENCODING_REF: objc2::Encoding =
            objc2::Encoding::Pointer(&objc2::Encoding::Struct("CGColorSpace", &[]));
    }
    let cs_ref = srgb_cs.as_ptr() as *const CGColorSpaceOpaque;

    // Only `CAMetalLayer` has a `colorspace` property; calling
    // `setColorspace:` on a plain `CALayer` is an unrecognized selector
    // (crash), so we always gate on `respondsToSelector:`.
    let selector = objc2::sel!(setColorspace:);
    unsafe {
        let root_layer: *mut AnyObject = msg_send![ns_view, layer];
        if root_layer.is_null() {
            return Err("ns_view.layer is null".into());
        }
        // Find the CAMetalLayer. wgpu-hal 24 (the raw-window-metal
        // approach) does NOT replace the view's layer: when the root
        // layer isn't already a CAMetalLayer — the default for an
        // eframe/winit NSView — it installs the metal layer as a
        // *sublayer*. So the root layer is a plain CALayer with no
        // `colorspace`; the layer we must tag is the metal sublayer.
        // (Older eframe set the view's own layer to the CAMetalLayer, so
        // check the root first and fall back to scanning sublayers.)
        let metal_layer = if msg_send![root_layer, respondsToSelector: selector] {
            root_layer
        } else {
            let sublayers: *mut AnyObject = msg_send![root_layer, sublayers];
            let mut found: *mut AnyObject = std::ptr::null_mut();
            if !sublayers.is_null() {
                let count: usize = msg_send![sublayers, count];
                for i in 0..count {
                    let sub: *mut AnyObject = msg_send![sublayers, objectAtIndex: i];
                    if !sub.is_null() && msg_send![sub, respondsToSelector: selector] {
                        found = sub;
                        break;
                    }
                }
            }
            found
        };
        if metal_layer.is_null() {
            // The metal sublayer may not exist yet on the first frame;
            // the caller retries until this succeeds.
            return Err("no CAMetalLayer on the view or its sublayers yet".into());
        }
        // CAMetalLayer.colorspace is `retain`-strong, so ObjC takes its
        // own reference; we can let `srgb_cs` drop after the call.
        let _: () = msg_send![metal_layer, setColorspace: cs_ref];
        tracing::info!("tagged CAMetalLayer.colorspace = sRGB");
    }
    Ok(())
}

impl App {
    fn current_save_options(&self) -> spektrafilm_io::SaveOptions {
        self.save_options.clone()
    }
    fn project_state(&self) -> project::Project {
        project::Project {
            version: 1,
            rotation: self.rotation,
            auto_preview: self.auto_preview,
            gpu_export: self.gpu_export,
            export_format: self.export_format,
            lut: self.lut.clone(),
            display: self.display.clone(),
            scan_for_print_restore: self.scan_for_print_restore,
            film: self.film_name.clone(),
            paper: self.print_name.clone(),
            params: self.params.clone(),
            image: self.image_path.clone(),
            zoom: self.zoom,
            pan: [self.pan.x, self.pan.y],
            raw: self.export_raw_options().clone(),
            crop: self.committed_crop_option(),
            output: self.current_save_options(),
        }
    }
    fn restore_project(&mut self, path: &Path, load_image: bool) -> Result<()> {
        let state = project::Project::load(path)?;
        if load_image {
            self.leave_photo();
        }
        profile::load_profile_by_name(&self.data_dir, &state.film)?;
        profile::load_profile_by_name(&self.data_dir, &state.paper)?;
        self.auto_preview = state.auto_preview;
        self.gpu_export = state.gpu_export;
        self.export_format = state.export_format;
        self.lut = state.lut;
        self.display = state.display;
        self.scan_for_print_restore = state.scan_for_print_restore;
        self.raw_options = state.raw;
        self.save_options = state.output;
        if load_image {
            if let Some(catalog) = &mut self.catalog {
                catalog.selected = None;
            }
            if let Some(image) = state.image {
                // Project crops already use the saved rotation, even for this photo.
                self.clear_image();
                self.crop = state.crop;
                self.load_image_with_state(
                    &image,
                    state.rotation,
                    Some((
                        state.params.io.input_color_space.clone(),
                        state.params.io.input_cctf_decoding,
                    )),
                    Some((state.zoom, egui::vec2(state.pan[0], state.pan[1]))),
                );
            }
        }
        self.params = state.params;
        if self.params.settings.apply_stock_specifics {
            let film = profile::load_profile_by_name(&self.data_dir, &state.film)?;
            spektrafilm_core::pipeline::apply_stock_defaults(&film, &mut self.params);
        }
        self.film_name = state.film;
        self.print_name = state.paper;
        self.film_dev_times = profile_dev_times(&self.data_dir, &self.film_name);
        self.print_dev_times = profile_dev_times(&self.data_dir, &self.print_name);
        self.zoom = state.zoom;
        self.pan = egui::vec2(state.pan[0], state.pan[1]);
        self.pipeline_cache = None;
        self.pipeline_cache_key = None;
        self.dirty = true;
        self.pending_dirty = self.render_job.is_some();
        Ok(())
    }
    fn parity_controls(&mut self, ui: &mut egui::Ui) {
        use inspector::{Section, Tab};
        let stock = self.stock_params();
        Section::new(
            "Color and processing",
            Tab::Output,
            tooltips::SECTION_COLOR_AND_PROCESSING,
        )
        .closed()
        .show(ui, |ui| {
            let mut changed = false;
            let spaces = [
                "sRGB",
                "ProPhoto RGB",
                "ITU-R BT.2020",
                "ITU-R BT.709",
                "Adobe RGB (1998)",
                "Display P3",
                "DCI-P3",
                "ACEScg",
                "ACES2065-1",
            ];
            tooltips::combo(
                ui,
                egui::ComboBox::from_id_salt("input_primaries")
                    .selected_text(&self.params.io.input_color_space),
                "Input primaries",
                tooltips::CONTROL_INPUT_PRIMARIES,
                |ui| {
                    for name in spaces {
                        changed |= ui
                            .selectable_value(
                                &mut self.params.io.input_color_space,
                                name.into(),
                                name,
                            )
                            .on_hover_text(tooltips::option(name))
                            .changed();
                    }
                },
            );
            changed |= tooltips::checkbox(
                ui,
                &mut self.params.io.input_cctf_decoding,
                "Decode input transfer function",
                tooltips::CONTROL_DECODE_INPUT_TRANSFER_FUNCTION,
            )
            .changed();
            tooltips::combo(
                ui,
                egui::ComboBox::from_id_salt("output_primaries")
                    .selected_text(&self.params.io.output_color_space),
                "Output primaries",
                tooltips::CONTROL_OUTPUT_PRIMARIES,
                |ui| {
                    for name in spaces {
                        changed |= ui
                            .selectable_value(
                                &mut self.params.io.output_color_space,
                                name.into(),
                                name,
                            )
                            .on_hover_text(tooltips::option(name))
                            .changed();
                    }
                },
            );
            for (label, order) in [
                ("Film dye order", &mut self.params.film_render.channel_order),
                (
                    "Print dye order",
                    &mut self.params.print_render.channel_order,
                ),
            ] {
                tooltips::combo(
                    ui,
                    egui::ComboBox::from_id_salt(label).selected_text(format!("{order:?}")),
                    label,
                    tooltips::DYE_ORDER,
                    |ui| {
                        for value in [
                            [0, 1, 2],
                            [0, 2, 1],
                            [1, 0, 2],
                            [1, 2, 0],
                            [2, 0, 1],
                            [2, 1, 0],
                        ] {
                            changed |= ui
                                .selectable_value(order, value, format!("{value:?}"))
                                .on_hover_text(tooltips::DYE_ORDER)
                                .changed();
                        }
                    },
                );
            }
            tooltips::combo(
                ui,
                egui::ComboBox::from_id_salt("enlarger_illuminant")
                    .selected_text(&self.params.enlarger.illuminant),
                "Enlarger illuminant",
                tooltips::MENU_ENLARGER_ILLUMINANT,
                |ui| {
                    for name in ["TH-KG3", "TH-KG3-L", "D50", "D55", "D65", "BB3200"] {
                        changed |= ui
                            .selectable_value(
                                &mut self.params.enlarger.illuminant,
                                name.into(),
                                name,
                            )
                            .on_hover_text(tooltips::option(name))
                            .changed();
                    }
                },
            );
            let automatic_before = self.params.settings.neutral_print_filters_from_database;
            let neutral_changed = tooltips::checkbox(
                ui,
                &mut self.params.settings.neutral_print_filters_from_database,
                "Automatic neutral print filters",
                tooltips::AUTOMATIC_NEUTRAL_FILTERS,
            )
            .changed();
            changed |= neutral_changed;
            if neutral_changed && automatic_before {
                let db = spektrafilm_core::neutral_filters::NeutralFilters::load(&self.data_dir);
                if let Some([c, m, y]) = db.lookup(
                    &self.print_name,
                    &self.params.enlarger.illuminant,
                    &self.film_name,
                ) {
                    self.params.enlarger.c_filter_neutral = c as f32;
                    self.params.enlarger.m_filter_neutral = m as f32;
                    self.params.enlarger.y_filter_neutral = y as f32;
                }
            }
            if !self.params.settings.neutral_print_filters_from_database {
                let e = &mut self.params.enlarger;
                changed |= tooltips::channels(
                    ui,
                    "Neutral filters",
                    tooltips::MANUAL_NEUTRAL_FILTER,
                    tooltips::CMY,
                    [
                        &mut e.c_filter_neutral,
                        &mut e.m_filter_neutral,
                        &mut e.y_filter_neutral,
                    ],
                    0.0..=200.0,
                    0.1,
                )
                .changed();
            }
            changed |= tooltips::checkbox(
                ui,
                &mut self.params.settings.use_enlarger_lut,
                "Enlarger spectral LUT",
                tooltips::CONTROL_ENLARGER_SPECTRAL_LUT,
            )
            .changed();
            changed |= tooltips::checkbox(
                ui,
                &mut self.params.settings.use_scanner_lut,
                "Scanner spectral LUT",
                tooltips::CONTROL_SCANNER_SPECTRAL_LUT,
            )
            .changed();
            if self.params.settings.use_enlarger_lut || self.params.settings.use_scanner_lut {
                changed |= tooltips::labeled(
                    ui,
                    tooltips::slider(&mut self.params.settings.lut_resolution, 3..=65)
                        .stock(stock.settings.lut_resolution),
                    "Spectral LUT resolution",
                    tooltips::CONTROL_SPECTRAL_LUT_RESOLUTION,
                )
                .changed();
            }
            changed |= tooltips::checkbox(
                ui,
                &mut self.params.settings.use_fast_stats,
                "Approximate physical-grain statistics",
                tooltips::CONTROL_APPROXIMATE_PHYSICAL_GRAIN_STATISTICS,
            )
            .changed();
            tooltips::combo(
                ui,
                egui::ComboBox::from_id_salt("gamut_algorithm").selected_text(names::label(
                    &self.params.io.output_gamut_compress.algorithm,
                )),
                "Output gamut compression",
                tooltips::CONTROL_OUTPUT_GAMUT_COMPRESSION,
                |ui| {
                    for name in ["off", "cam16ucs", "oklch", "oklrab", "jzazbz", "aces_rgc"] {
                        changed |= ui
                            .selectable_value(
                                &mut self.params.io.output_gamut_compress.algorithm,
                                name.into(),
                                names::label(name),
                            )
                            .on_hover_text(tooltips::option(name))
                            .changed();
                    }
                },
            );
            changed |= tooltips::checkbox(
                ui,
                &mut self.params.io.input_gamut_compress.active,
                "Input gamut compression",
                tooltips::CONTROL_INPUT_GAMUT_COMPRESSION,
            )
            .changed();
            changed |= tooltips::checkbox(
                ui,
                &mut self.params.io.crop,
                "Crop before processing",
                tooltips::CONTROL_CROP_BEFORE_PROCESSING,
            )
            .changed();
            if self.params.io.crop {
                let io = &mut self.params.io;
                let [cx, cy] = io.crop_center.each_mut();
                changed |= tooltips::channels(
                    ui,
                    "Crop center",
                    tooltips::CROP_CENTER,
                    ["X", "Y"],
                    [cx, cy],
                    0.0..=1.0,
                    0.002,
                )
                .changed();
                let [w, h] = io.crop_size.each_mut();
                changed |= tooltips::channels(
                    ui,
                    "Crop size",
                    tooltips::CROP_SIZE,
                    ["W", "H"],
                    [w, h],
                    0.01..=1.0,
                    0.002,
                )
                .changed();
            }
            changed |= tooltips::labeled(
                ui,
                tooltips::slider(
                    &mut self.params.film_render.grain.blur_dye_clouds_um,
                    0.0..=5.0,
                )
                .stock(stock.film_render.grain.blur_dye_clouds_um),
                "Dye cloud blur (µm)",
                tooltips::CONTROL_DYE_CLOUD_BLUR_UM,
            )
            .changed();
            changed |= tooltips::checkbox(
                ui,
                &mut self.params.settings.apply_hanatos2025_adaptation_surface,
                "Sensitivity adaptation surface",
                tooltips::CONTROL_SENSITIVITY_ADAPTATION_SURFACE,
            )
            .changed();
            changed |= tooltips::labeled(
                ui,
                tooltips::slider(&mut self.params.settings.spectral_gaussian_blur, 0.0..=10.0)
                    .stock(stock.settings.spectral_gaussian_blur),
                "Spectral blur",
                tooltips::CONTROL_SPECTRAL_BLUR,
            )
            .changed();
            changed |= tooltips::labeled(
                ui,
                tooltips::slider(&mut self.params.camera.filter_uv[0], 0.0..=1.0)
                    .stock(stock.camera.filter_uv[0]),
                "UV filter amount",
                tooltips::CONTROL_UV_FILTER_AMOUNT,
            )
            .changed();
            changed |= tooltips::labeled(
                ui,
                tooltips::slider(&mut self.params.camera.filter_ir[0], 0.0..=1.0)
                    .stock(stock.camera.filter_ir[0]),
                "IR filter amount",
                tooltips::CONTROL_IR_FILTER_AMOUNT,
            )
            .changed();
            changed |= tooltips::checkbox(
                ui,
                &mut self.params.settings.preview_mode,
                "Simplified preview mode",
                tooltips::CONTROL_SIMPLIFIED_PREVIEW_MODE,
            )
            .changed();
            if changed {
                self.dirty = true;
            }
        });
        Section::new("File export", Tab::Output, tooltips::SECTION_FILE_EXPORT)
            .closed()
            .show(ui, |ui| {
                tooltips::combo(
                    ui,
                    egui::ComboBox::from_id_salt("export_format")
                        .selected_text(self.export_format.label()),
                    "File format",
                    tooltips::MENU_EXPORT_FORMAT,
                    |ui| {
                        for format in [
                            ExportFormat::Jpeg,
                            ExportFormat::Tiff,
                            ExportFormat::Png,
                            ExportFormat::Exr,
                        ] {
                            ui.selectable_value(&mut self.export_format, format, format.label());
                        }
                    },
                );
                if self.export_format == ExportFormat::Exr && self.save_options.bit_depth == 8 {
                    self.save_options.bit_depth = 16;
                }
                tooltips::checkbox(
                    ui,
                    &mut self.gpu_export,
                    "Use GPU for full-resolution export",
                    tooltips::EXPORT_PRECISION,
                );
                tooltips::combo(
                    ui,
                    egui::ComboBox::from_id_salt("saving_color_space")
                        .selected_text(&self.save_options.color_space),
                    "Saving color space",
                    tooltips::MENU_SAVING_COLOR_SPACE,
                    |ui| {
                        for name in [
                            "sRGB",
                            "Display P3",
                            "ProPhoto RGB",
                            "Adobe RGB (1998)",
                            "ITU-R BT.2020",
                            "ACES2065-1",
                            "ACEScg",
                        ] {
                            ui.selectable_value(
                                &mut self.save_options.color_space,
                                name.into(),
                                name,
                            )
                            .on_hover_text(tooltips::option(name));
                        }
                    },
                );
                tooltips::checkbox(
                    ui,
                    &mut self.save_options.encoded,
                    "Encode saving transfer function",
                    tooltips::CONTROL_ENCODE_SAVING_TRANSFER_FUNCTION,
                );
                if matches!(self.export_format, ExportFormat::Tiff | ExportFormat::Exr) {
                    tooltips::combo(
                        ui,
                        egui::ComboBox::from_id_salt("export_depth")
                            .selected_text(format!("{} bit", self.save_options.bit_depth)),
                        if self.export_format == ExportFormat::Exr {
                            "EXR bit depth"
                        } else {
                            "TIFF bit depth"
                        },
                        tooltips::MENU_EXPORT_DEPTH,
                        |ui| {
                            let depths: &[u8] = if self.export_format == ExportFormat::Exr {
                                &[16, 32]
                            } else {
                                &[8, 16, 32]
                            };
                            for &depth in depths {
                                ui.selectable_value(
                                    &mut self.save_options.bit_depth,
                                    depth,
                                    format!("{depth} bit"),
                                )
                                .on_hover_text(
                                    if self.export_format == ExportFormat::Exr {
                                        "Floating-point precision for linear EXR pixels."
                                    } else if depth == 32 {
                                        tooltips::TIFF_FLOAT
                                    } else {
                                        tooltips::TIFF_INTEGER
                                    },
                                );
                            }
                        },
                    );
                    ui.label(
                        egui::RichText::new("Float TIFF and EXR preserve values outside 0–1.")
                            .small()
                            .color(theme::TEXT_WEAK),
                    );
                }
                if self.export_format == ExportFormat::Jpeg {
                    tooltips::labeled(
                        ui,
                        egui::Slider::new(&mut self.save_options.jpeg_quality, 1..=100),
                        "JPEG quality",
                        tooltips::CONTROL_JPEG_QUALITY,
                    );
                }
            });
    }

    fn raw_controls(&mut self, ui: &mut egui::Ui) {
        use inspector::{Section, Tab};
        Section::new(
            "RAW development",
            Tab::Input,
            tooltips::SECTION_RAW_DEVELOPMENT,
        )
        .closed()
        .show(ui, |ui| {
            tooltips::combo(
                ui,
                egui::ComboBox::from_id_salt("raw_decoder")
                    .selected_text(names::label(&self.raw_options.decoder)),
                "RAW decoder",
                tooltips::MENU_RAW_DECODER,
                |ui| {
                    for name in ["auto", "libraw", "rawler"] {
                        ui.selectable_value(
                            &mut self.raw_options.decoder,
                            name.into(),
                            names::label(name),
                        )
                        .on_hover_text(tooltips::option(name));
                    }
                },
            );
            tooltips::combo(
                ui,
                egui::ComboBox::from_id_salt("raw_white_balance")
                    .selected_text(names::label(&self.raw_options.white_balance)),
                "White balance",
                tooltips::MENU_RAW_WHITE_BALANCE,
                |ui| {
                    for name in ["as_shot", "daylight", "tungsten", "custom"] {
                        ui.selectable_value(
                            &mut self.raw_options.white_balance,
                            name.into(),
                            names::label(name),
                        )
                        .on_hover_text(tooltips::option(name));
                    }
                },
            );
            if self.raw_options.white_balance == "custom" {
                tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut self.raw_options.temperature, 2000.0..=15000.0),
                    "Temperature (K)",
                    tooltips::CONTROL_TEMPERATURE_K,
                );
                tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut self.raw_options.tint, 0.1..=3.0),
                    "Tint",
                    tooltips::CONTROL_TINT,
                );
            }
            tooltips::checkbox(
                ui,
                &mut self.raw_options.lens_correction,
                "Lens correction (Lensfun)",
                tooltips::CONTROL_LENS_CORRECTION_LENSFUN,
            );
            if self.raw_options.lens_correction {
                if let Some(path) = &self.raw_options.lens_database {
                    ui.label(
                        egui::RichText::new(format!("Database override: {}", path.display()))
                            .small()
                            .color(theme::TEXT_WEAK),
                    )
                    .on_hover_text(tooltips::LENS_DATABASE_OVERRIDE);
                    if ui
                        .button("Use automatic Lensfun database")
                        .on_hover_text(tooltips::CONTROL_USE_AUTOMATIC_LENSFUN_DATABASE)
                        .clicked()
                    {
                        self.raw_options.lens_database = None;
                        self.lens_database_status = LensDatabaseStatus::detect();
                    }
                } else {
                    ui.label(
                        egui::RichText::new(self.lens_database_status.label)
                            .small()
                            .color(theme::TEXT_WEAK),
                    )
                    .on_hover_text(&self.lens_database_status.detail);
                }
                if ui
                    .button("Choose Lensfun database…")
                    .on_hover_text(tooltips::CHOOSE_LENS_DATABASE)
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        self.raw_options.lens_database = Some(path);
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod raw_reload_tests {
    use super::*;

    fn new_app() -> App {
        let cc = eframe::CreationContext::_new_kittest(egui::Context::default());
        let mut app = App::new(
            &cc,
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            None,
        );
        app.raw_options = spektrafilm_io::RawOptions {
            decoder: "rawler".into(),
            ..Default::default()
        };
        app
    }

    fn input() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../spektrafilm-io/tests/fixtures/synthetic.dng")
    }

    fn finish(app: &mut App) {
        while let Some(job) = &mut app.load_job {
            if let Some(handle) = job.handle.take() {
                handle.join().unwrap();
            }
            app.poll_load_job();
        }
    }

    #[test]
    fn raw_changes_reload_pixels_and_preserve_geometry_rotation_and_view() {
        let mut app = new_app();
        app.raw_options.lightroom_geometry = Some(spektrafilm_io::lightroom::Geometry {
            orientation: 6,
            crop: [0.1, 0.1, 0.9, 0.9],
            angle: 0.,
        });
        let view = (2., egui::vec2(15., -8.));
        app.load_image_with_state(&input(), 1, None, Some(view));
        finish(&mut app);
        assert!(app.image.is_some(), "{}", app.status);
        let original = app.image.clone().unwrap();
        let previous = app.raw_options.clone();
        app.raw_options.white_balance = "custom".into();
        app.raw_options.temperature = 3000.;
        app.raw_options.tint = 1.3;
        app.reload_raw_if_changed(&previous);
        assert!(app.load_job.is_some());
        assert_eq!(app.export_raw_options(), &previous);
        finish(&mut app);
        assert!(app.status.starts_with("Loaded "), "{}", app.status);
        let expected = spektrafilm_io::load(&input(), &app.raw_options)
            .unwrap()
            .pixels
            .rotated_quarter_turns(1);
        // The editor applies Lightroom's crop itself, on the whole frame.
        let image = app.image.as_ref().unwrap();
        assert_eq!(
            (image.width, image.height),
            (expected.width, expected.height)
        );
        let error = image
            .data
            .iter()
            .zip(&expected.data)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0., f64::max);
        assert!(error < 1e-5, "{error}");
        let kept = app.image.as_ref().unwrap().clone();
        assert_ne!(app.image.as_ref().unwrap().data, original.data);
        assert_eq!(app.rotation, 1);
        assert_eq!((app.zoom, app.pan), view);
        assert_eq!(app.project_state().raw, app.raw_options);

        let applied = app.raw_options.clone();
        app.raw_options.decoder = "invalid".into();
        app.reload_raw_if_changed(&applied);
        finish(&mut app);
        assert!(app.status.starts_with("Load error:"), "{}", app.status);
        assert_eq!(app.export_raw_options(), &applied);
        assert!(
            Arc::ptr_eq(app.image.as_ref().unwrap(), &kept),
            "a failed reload keeps the pixels"
        );
        app.reload_raw_if_changed(&app.raw_options.clone());
        assert!(
            app.load_job.is_none(),
            "failed reload was retried without an edit"
        );
    }

    #[test]
    fn rapid_raw_changes_keep_only_the_latest_selected_photo_and_settings() {
        let mut app = new_app();
        app.load_image_from_path(&input());
        let dir = tempfile::tempdir().unwrap();
        let selected = dir.path().join("selected.dng");
        std::fs::copy(input(), &selected).unwrap();
        let view = (1.5, egui::vec2(-4., 12.));
        app.load_image_with_state(&selected, 3, None, Some(view));
        for tint in [1.1, 1.2, 1.3] {
            let previous = app.raw_options.clone();
            app.raw_options.white_balance = "custom".into();
            app.raw_options.tint = tint;
            app.reload_raw_if_changed(&previous);
            assert!(app.load_job.as_ref().unwrap().discard);
            let pending = app.pending_load.as_ref().unwrap();
            assert_eq!(pending.path, selected);
            assert_eq!(pending.rotation, 3);
            assert_eq!(pending.view_override, Some(view));
            assert_eq!(pending.raw_options, app.raw_options);
        }
        finish(&mut app);
        assert!(app.status.starts_with("Loaded "), "{}", app.status);
        assert_eq!(app.image_path, Some(selected));
        assert_eq!(app.export_raw_options(), &app.raw_options);
        assert_eq!(app.rotation, 3);
        assert_eq!((app.zoom, app.pan), view);
        assert!(app.pending_load.is_none());
        app.reload_raw_if_changed(&app.raw_options.clone());
        assert!(
            app.load_job.is_none(),
            "unchanged settings started another load"
        );
    }

    #[test]
    fn raw_changes_without_a_raw_photo_do_not_start_a_load() {
        let mut app = new_app();
        let previous = app.raw_options.clone();
        app.raw_options.tint = 1.2;
        app.reload_raw_if_changed(&previous);
        assert!(app.load_job.is_none());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("image.png");
        image::RgbImage::new(8, 4).save(&path).unwrap();
        app.load_image_from_path(&path);
        finish(&mut app);
        assert!(app.image.is_some(), "{}", app.status);
        let previous = app.raw_options.clone();
        app.raw_options.tint = 1.3;
        app.reload_raw_if_changed(&previous);
        assert!(app.load_job.is_none());
    }
}

#[cfg(test)]
mod transition_tests {
    use super::canvas_transition;

    const SMALL: [usize; 2] = [1024, 683];

    #[test]
    fn only_a_new_image_develops_from_blank_paper() {
        assert_eq!(
            canvas_transition(true, SMALL, false, true, false),
            Some(true)
        );
        assert_eq!(
            canvas_transition(true, SMALL, false, false, true),
            Some(false)
        );
    }

    #[test]
    fn editing_a_control_never_redevelops_the_image() {
        // Drag preview landing after release, after a full-size frame.
        assert_eq!(
            canvas_transition(true, [640, 427], true, false, false),
            None
        );
        // Full-size frame replacing a drag preview of a different size.
        assert_eq!(canvas_transition(true, SMALL, false, false, false), None);
        // Drag previews never animate, even back to back at the same size.
        assert_eq!(canvas_transition(true, [640, 427], true, false, true), None);
    }

    #[test]
    fn animation_respects_the_toggle_and_size_limit() {
        assert_eq!(canvas_transition(false, SMALL, false, true, false), None);
        assert_eq!(
            canvas_transition(true, [2048, 1365], false, true, false),
            None
        );
    }
}

#[cfg(all(test, unix))]
mod export_tests {
    use super::*;

    #[test]
    fn export_recipe_uses_loaded_raw_settings_and_keeps_editor_values_before_crop_scaling() {
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx);
        let mut app = App::new(
            &cc,
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            None,
        );
        app.params.camera.tone.shadows = 42.;
        app.params.camera.film_format_mm = 35.;
        app.rotation = 1;
        app.raw_options.temperature = 9000.;
        app.loaded_raw_options = Some(spektrafilm_io::RawOptions {
            decoder: "rawler".into(),
            white_balance: "custom".into(),
            temperature: 4200.,
            lightroom_geometry: Some(spektrafilm_io::lightroom::Geometry {
                orientation: 6,
                crop: [0.25, 0., 1., 0.5],
                angle: 2.,
            }),
            ..Default::default()
        });
        let crop = Crop {
            size: [0.5, 0.75],
            angle: 2.,
            ..Default::default()
        };
        let metadata = app.export_metadata(&app.params, crop).unwrap();
        let settings = metadata.edit_settings.unwrap();
        assert_eq!(settings["film"], app.film_name);
        assert_eq!(settings["paper"], app.print_name);
        assert_eq!(
            settings["params"],
            serde_json::to_value(&app.params).unwrap()
        );
        assert_eq!(settings["raw"]["temperature"], 4200.);
        assert_eq!(settings["raw"]["lightroom_geometry"]["orientation"], 6);
        assert_eq!(
            settings["raw"]["lightroom_geometry"]["crop"],
            serde_json::json!([0., 0., 1., 1.])
        );
        assert_eq!(settings["raw"]["lightroom_geometry"]["angle"], 0.);
        assert_eq!(settings["rotation"], 1);
        assert_eq!(settings["crop"], serde_json::to_value(crop).unwrap());
    }

    #[test]
    fn cancelling_an_export_waiting_for_capacity_does_not_run_or_write_it() {
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut app = App::new(
            &cc,
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            None,
        );
        let first = app.scheduler.task(Priority::Interactive).cpu().unwrap();
        let second = app.scheduler.task(Priority::Interactive).cpu().unwrap();
        let ran = Arc::new(AtomicBool::new(false));
        let visited = ran.clone();
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("cancelled.png");
        app.start_export_job(&ctx, "test", output.clone(), move |_| {
            let _cpu = scheduler::cpu_permit()?;
            visited.store(true, Ordering::Relaxed);
            Ok(())
        });
        app.cancel_export();
        let job = app.export_job.as_mut().unwrap();
        let result = job.rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(result.unwrap_err().contains("cancelled"));
        job.handle.take().unwrap().join().unwrap();
        assert!(!ran.load(Ordering::Relaxed));
        assert!(!output.exists());
        drop((first, second));
    }

    #[test]
    fn verbose_child_does_not_block_export() {
        let mut cmd = std::process::Command::new("sh");
        cmd.arg("-c")
            .arg("i=0; while [ $i -lt 10000 ]; do echo diagnostic-line >&2; i=$((i+1)); done");
        run_cli(&mut cmd, &AtomicBool::new(false)).unwrap();
    }
    #[test]
    fn cancelled_export_reaps_child() {
        let mut cmd = std::process::Command::new("sleep");
        cmd.arg("30");
        let start = Instant::now();
        let error = run_cli(&mut cmd, &AtomicBool::new(true)).unwrap_err();
        assert!(error.to_string().contains("cancelled"));
        assert!(start.elapsed() < Duration::from_secs(3));
    }
}

#[cfg(test)]
mod panic_tests {
    use super::panic_message;

    #[test]
    fn worker_panics_preserve_literal_and_formatted_messages() {
        let literal = std::panic::catch_unwind(|| panic!("literal failure")).unwrap_err();
        assert_eq!(panic_message(literal), "panic: literal failure");
        let index = 1;
        let formatted = std::panic::catch_unwind(|| panic!("invalid layer {index}")).unwrap_err();
        assert_eq!(panic_message(formatted), "panic: invalid layer 1");
        let other = std::panic::catch_unwind(|| std::panic::panic_any(42u32)).unwrap_err();
        assert_eq!(panic_message(other), "panic (no message)");
    }
}

#[cfg(test)]
mod preview_failure_tests {
    use super::*;

    #[test]
    fn project_reopens_with_applied_raw_settings() {
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx);
        let mut app = App::new(
            &cc,
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            None,
        );
        app.raw_options = spektrafilm_io::RawOptions {
            decoder: "rawler".into(),
            ..Default::default()
        };
        let input = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../spektrafilm-io/tests/fixtures/synthetic.dng");
        let finish = |app: &mut App| {
            app.load_job
                .as_mut()
                .unwrap()
                .handle
                .take()
                .unwrap()
                .join()
                .unwrap();
            app.poll_load_job();
            assert!(app.status.starts_with("Loaded "), "{}", app.status);
        };
        app.load_image_from_path(&input);
        finish(&mut app);
        let original = app.image.as_ref().unwrap().clone();
        let applied = app.export_raw_options().clone();
        app.raw_options.white_balance = "custom".into();
        app.raw_options.temperature = 3000.;
        app.raw_options.tint = 1.3;
        let state = app.project_state();
        assert_eq!(state.raw, applied);
        let path = std::env::temp_dir().join(format!(
            "spektrafilm-applied-raw-project-{}.json",
            std::process::id()
        ));
        state.save(&path).unwrap();
        app.restore_project(&path, true).unwrap();
        finish(&mut app);
        assert_eq!(app.export_raw_options(), &applied);
        assert_eq!(app.image.as_ref().unwrap().data, original.data);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn export_keeps_applied_raw_settings_through_edits_and_rotation_until_reload() {
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx);
        let mut app = App::new(
            &cc,
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            None,
        );
        app.raw_options = spektrafilm_io::RawOptions {
            decoder: "rawler".into(),
            ..Default::default()
        };
        let original = app.raw_options.clone();
        let input = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../spektrafilm-io/tests/fixtures/synthetic.dng");
        let finish_load = |app: &mut App| {
            app.load_job
                .as_mut()
                .unwrap()
                .handle
                .take()
                .unwrap()
                .join()
                .unwrap();
            app.poll_load_job();
            assert!(
                app.load_job.is_none(),
                "load did not finish: {}",
                app.status
            );
            assert!(app.image.is_some(), "load failed: {}", app.status);
            assert!(app.status.starts_with("Loaded "), "{}", app.status);
        };
        app.load_image_from_path(&input);
        // Edits can arrive while decoding; they apply only after a reload.
        app.raw_options.white_balance = "custom".into();
        app.raw_options.tint = 1.3;
        finish_load(&mut app);
        assert_eq!(app.export_raw_options(), &original);
        app.rotate_input(1);
        finish_load(&mut app);
        assert_eq!(app.export_raw_options(), &original);
        app.load_image_from_path(&input);
        finish_load(&mut app);
        assert_eq!(app.export_raw_options(), &app.raw_options);
    }

    #[test]
    fn failed_refinement_does_not_restart_or_leave_saveable_stale_output() {
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut app = App::new(
            &cc,
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            None,
        );
        app.params = RuntimeParams::default();
        app.params.settings.preview_max_size = 2048;
        app.auto_preview = true;
        app.image = Some(Arc::new(ImageBuf::new(64, 32)));
        app.output_image = Some(ImageBuf::new(64, 32));
        app.rendered_preview_limit = 640;
        app.dirty = false;
        app.pending_dirty = false;
        app.dirty_since = Instant::now().checked_sub(PREVIEW_DEBOUNCE);
        #[cfg(target_os = "macos")]
        {
            app.metal_colorspace_tagged = true;
        }
        let (tx, rx) = mpsc::channel();
        tx.send(Err("test refinement failure".into())).unwrap();
        app.render_job = Some(RenderJob {
            started: Instant::now(),
            discard: false,
            task: app.scheduler.task(Priority::Interactive),
            interactive: false,
            rx,
            handle: None,
        });
        app.poll_render_job(&ctx);
        assert!(
            app.output_image.is_none(),
            "failed settings still expose the old preview for saving"
        );
        let mut frame = eframe::Frame::_new_kittest();
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            eframe::App::update(&mut app, ctx, &mut frame);
        });
        assert!(
            !app.dirty && app.render_job.is_none(),
            "failed refinement automatically retried"
        );
        assert!(app.status.contains("test refinement failure"));
    }
}

#[cfg(test)]
mod preview_texture_tests {
    use super::*;

    #[test]
    fn accelerated_display_preserves_source_and_cpu_fallback() {
        let image = ImageBuf::from_data(
            1025,
            683,
            (0..1025 * 683 * 3)
                .map(|i| spektrafilm_math::precision::from_f64((i * 73 % 1021) as f64 / 511. - 0.1))
                .collect(),
        );
        let before = image.data.clone();
        let io = spektrafilm_core::params::IoParams {
            output_color_space: "ACES2065-1".into(),
            output_cctf_encoding: false,
            ..Default::default()
        };
        let expected = make_preview_texture_data(&image, &io, true);
        let cpu = prepare_preview_texture_data(
            &image,
            &io,
            true,
            &spektrafilm_gpu::cpu_backend::CpuBackend,
        );
        assert_eq!(cpu.rgba, expected.rgba);
        if let Some(backend) = spektrafilm_gpu::wgpu_backend::WgpuBackend::new() {
            let actual = prepare_preview_texture_data(&image, &io, true, &backend);
            assert_eq!(actual.size, expected.size);
            assert!(
                actual
                    .rgba
                    .iter()
                    .zip(&expected.rgba)
                    .all(|(a, b)| a.abs_diff(*b) <= 1)
            );
        }
        assert_eq!(image.data, before);
    }

    #[test]
    fn gpu_preview_preparation_matches_cpu() {
        let Some(backend) = spektrafilm_gpu::wgpu_backend::WgpuBackend::new() else {
            return;
        };
        if backend
            .display_preview(
                &ImageBuf::new(1, 1),
                "sRGB",
                true,
                false,
                PREVIEW_TEXTURE_MAX_DIM,
            )
            .is_none()
        {
            return;
        }
        for color in spektrafilm_math::transfer::names() {
            let curve = spektrafilm_math::transfer::get(color).unwrap().transfer;
            if !curve.supported() {
                continue;
            }
            for encoded in [false, true] {
                let image = ImageBuf::from_data(
                    31,
                    17,
                    (0..31 * 17 * 3)
                        .map(|i| {
                            let v = (i * 73 % 1021) as f64 / 511.;
                            spektrafilm_math::precision::from_f64(if encoded {
                                curve.encode(v)
                            } else {
                                v
                            })
                        })
                        .collect(),
                );
                let mut io = spektrafilm_core::params::IoParams::default();
                io.output_color_space = color.into();
                io.output_cctf_encoding = encoded;
                for transform in [false, true] {
                    let actual = backend
                        .display_preview(&image, color, encoded, transform, PREVIEW_TEXTURE_MAX_DIM)
                        .unwrap();
                    let expected = make_preview_texture_data(&image, &io, transform);
                    assert_eq!(actual.size, expected.size);
                    let error = actual
                        .rgba
                        .iter()
                        .zip(&expected.rgba)
                        .map(|(a, b)| a.abs_diff(*b))
                        .max()
                        .unwrap();
                    assert!(
                        error <= 1,
                        "{color}, encoded={encoded}, transform={transform}: {error} byte error"
                    );
                }
            }
        }
        // Exercise the display texture cap and its transform-before-interpolation order.
        let width = PREVIEW_TEXTURE_MAX_DIM as u32 + 17;
        let image = ImageBuf::from_data(
            width,
            3,
            (0..width * 3 * 3)
                .map(|i| spektrafilm_math::precision::from_f64((i * 73 % 1021) as f64 / 511. - 0.1))
                .collect(),
        );
        for transform in [false, true] {
            let io = spektrafilm_core::params::IoParams {
                output_color_space: "ProPhoto RGB".into(),
                output_cctf_encoding: true,
                ..Default::default()
            };
            let expected = make_preview_texture_data(&image, &io, transform);
            let actual = backend
                .display_preview(
                    &image,
                    "ProPhoto RGB",
                    true,
                    transform,
                    PREVIEW_TEXTURE_MAX_DIM,
                )
                .unwrap();
            assert_eq!(actual.size, expected.size);
            let error = actual
                .rgba
                .iter()
                .zip(&expected.rgba)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            assert!(
                error <= 1,
                "capped preview, transform={transform}: {error} byte error"
            );
        }
    }

    #[test]
    #[ignore = "release-mode A/B benchmark; set SPEKTRAFILM_BENCH_LIGHTROOM_CATALOG to a Lightroom catalog"]
    fn preview_gpu_benchmark() {
        use std::hint::black_box;
        let catalog =
            std::env::var_os("SPEKTRAFILM_BENCH_LIGHTROOM_CATALOG").expect("catalog path");
        let catalog = spektrafilm_io::lightroom::Catalog::open(Path::new(&catalog)).unwrap();
        let backend = spektrafilm_gpu::wgpu_backend::WgpuBackend::new().expect("GPU");
        let scheduler = Scheduler::default();
        let task = scheduler.task(Priority::Interactive);
        let _scope = task.enter();
        for photo in catalog
            .photos
            .iter()
            .filter(|p| p.available && spektrafilm_io::is_raw(&p.path))
            .take(6)
        {
            let started = Instant::now();
            let source = spektrafilm_io::load(
                &photo.path,
                &spektrafilm_io::RawOptions {
                    decoder: "rawler".into(),
                    lightroom_geometry: photo.geometry.clone().ok(),
                    ..Default::default()
                },
            )
            .unwrap();
            let decode_ms = started.elapsed().as_secs_f64() * 1000.;
            let input = spektrafilm_core::resize::preview(&source.pixels, 2048);
            let mut io = spektrafilm_core::params::IoParams::default();
            io.output_color_space = source.color_space.clone();
            io.output_cctf_encoding = source.encoded;
            // Warm compilation before paired, alternating measurements.
            let cold_start = Instant::now();
            let first = backend
                .display_preview(
                    &input,
                    &source.color_space,
                    source.encoded,
                    true,
                    PREVIEW_TEXTURE_MAX_DIM,
                )
                .unwrap();
            let cold_ms = cold_start.elapsed().as_secs_f64() * 1000.;
            let expected = make_preview_texture_data(&input, &io, true);
            let error = first
                .rgba
                .iter()
                .zip(&expected.rgba)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            assert!(
                error <= 1,
                "real RAW source display differs by {error} byte levels"
            );
            println!(
                "FIRST file={} gpu_display_ms={cold_ms:.3} max_byte_error={error}",
                photo.path.file_name().unwrap().to_string_lossy()
            );
            for round in 0..7 {
                let mut times = [0.; 5];
                for i in if round % 2 == 0 {
                    [0, 1, 2, 3, 4]
                } else {
                    [4, 3, 2, 1, 0]
                } {
                    let start = Instant::now();
                    match i {
                        0 => {
                            black_box(spektrafilm_core::resize::preview(&source.pixels, 2048));
                        }
                        1 => {
                            black_box(make_preview_texture_data(&input, &io, true));
                        }
                        2 => {
                            black_box(
                                backend
                                    .display_preview(
                                        &input,
                                        &source.color_space,
                                        source.encoded,
                                        true,
                                        PREVIEW_TEXTURE_MAX_DIM,
                                    )
                                    .unwrap(),
                            );
                        }
                        3 => {
                            let input = spektrafilm_core::resize::preview(&source.pixels, 2048);
                            black_box(make_preview_texture_data(&input, &io, true));
                        }
                        _ => {
                            let input = spektrafilm_core::resize::preview(&source.pixels, 2048);
                            black_box(prepare_preview_texture_data(&input, &io, true, &backend));
                        }
                    }
                    times[i] = start.elapsed().as_secs_f64() * 1000.;
                }
                println!(
                    "PREP file={} round={round} decode_ms={decode_ms:.3} cpu_resize_ms={:.3} cpu_display_ms={:.3} gpu_display_ms={:.3} cpu_prepare_ms={:.3} gpu_prepare_ms={:.3}",
                    photo.path.file_name().unwrap().to_string_lossy(),
                    times[0],
                    times[1],
                    times[2],
                    times[3],
                    times[4]
                );
            }
        }
        for width in [256, 512, 1024, 2048] {
            let height = width * 2 / 3;
            let image = ImageBuf::from_data(
                width,
                height,
                (0..width * height * 3)
                    .map(|i| spektrafilm_math::precision::from_f64((i * 73 % 1021) as f64 / 1021.))
                    .collect(),
            );
            for encoded in [false, true] {
                let io = spektrafilm_core::params::IoParams {
                    output_cctf_encoding: encoded,
                    ..Default::default()
                };
                for round in 0..5 {
                    let mut times = [0.; 2];
                    for i in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                        let start = Instant::now();
                        if i == 0 {
                            black_box(make_preview_texture_data(&image, &io, true));
                        } else {
                            black_box(
                                backend
                                    .display_preview(
                                        &image,
                                        "sRGB",
                                        encoded,
                                        true,
                                        PREVIEW_TEXTURE_MAX_DIM,
                                    )
                                    .unwrap(),
                            );
                        }
                        times[i] = start.elapsed().as_secs_f64() * 1000.;
                    }
                    println!(
                        "DISPLAY width={width} encoded={encoded} cpu_ms={:.3} gpu_ms={:.3}",
                        times[0], times[1]
                    );
                }
            }
        }
    }

    #[test]
    fn parallel_display_conversion_matches_scalar_reference() {
        let image = ImageBuf::from_data(
            31,
            17,
            (0..31 * 17 * 3)
                .map(|i| spektrafilm_math::precision::from_f64((i % 127) as f64 / 100. - 0.1))
                .collect(),
        );
        for (color, encoded) in [
            ("ACES2065-1", false),
            ("sRGB", false),
            ("sRGB", true),
            ("ProPhoto RGB", true),
        ] {
            let mut io = spektrafilm_core::params::IoParams::default();
            io.output_color_space = color.into();
            io.output_cctf_encoding = encoded;
            for transform in [false, true] {
                let actual = make_preview_texture_data(&image, &io, transform);
                let curve = spektrafilm_math::transfer::get(color).unwrap().transfer;
                let matrix = spektrafilm_math::transfer::rgb_matrix(color, "sRGB");
                let expected: Vec<u8> = image
                    .pixels()
                    .flat_map(|pixel| {
                        let mut rgb = [pixel[0], pixel[1], pixel[2]];
                        if transform && !(color == "sRGB" && encoded) {
                            let linear = rgb.map(|v| {
                                if encoded {
                                    curve.decode(v as f64)
                                } else {
                                    v as f64
                                }
                            });
                            rgb = std::array::from_fn(|c| {
                                spektrafilm_math::precision::from_f64(
                                    spektrafilm_math::transfer::Transfer::Srgb.encode(
                                        matrix[c][0] * linear[0]
                                            + matrix[c][1] * linear[1]
                                            + matrix[c][2] * linear[2],
                                    ),
                                )
                            });
                        }
                        let rgb = rgb.map(|v| (to_f32(v).clamp(0., 1.) * 255.).round() as u8);
                        [rgb[0], rgb[1], rgb[2], 255]
                    })
                    .collect();
                assert_eq!(
                    actual.rgba, expected,
                    "{color}, encoded={encoded}, transform={transform}"
                );
            }
        }
    }
}

#[cfg(test)]
mod catalog_loading_tests {
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

    #[test]
    #[ignore = "requires SPEKTRAFILM_BENCH_LIGHTROOM_CATALOG and local originals"]
    fn benchmark_catalog_load() {
        tracing_subscriber::fmt()
            .with_env_filter("spektrafilm_io=debug")
            .try_init()
            .ok();
        let path = std::env::var_os("SPEKTRAFILM_BENCH_LIGHTROOM_CATALOG").expect("catalog path");
        let catalog = spektrafilm_io::lightroom::Catalog::open(Path::new(&path)).unwrap();
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut app = App::new(&cc, Arc::from(spektrafilm_gpu::select_backend()), None);
        if let Ok(decoder) = std::env::var("SPEKTRAFILM_BENCH_RAW_DECODER") {
            app.raw_options.decoder = decoder;
        }
        for (index, photo) in catalog
            .photos
            .into_iter()
            .filter(|p| p.available && spektrafilm_io::is_raw(&p.path))
            .take(6)
            .enumerate()
        {
            let started = Instant::now();
            app.select_catalog_photo(photo);
            finish(&mut app);
            eprintln!("[benchmark] source: {}", app.status);
            app.dispatch_render(&ctx);
            while let Some(job) = &mut app.render_job {
                if let Some(handle) = job.handle.take() {
                    handle.join().unwrap();
                }
                app.poll_render_job(&ctx);
            }
            assert!(app.output_image.is_some(), "{}", app.status);
            eprintln!(
                "[benchmark] click-to-render {:.1} ms: {}",
                started.elapsed().as_secs_f64() * 1000.,
                app.status
            );
            if index == 0 || index == 3 {
                let mut io = app.params.io.clone();
                io.output_color_space = io.input_color_space.clone();
                io.output_cctf_encoding = io.input_cctf_decoding;
                let source = &app
                    .preview_input
                    .iter()
                    .find(|(limit, _)| *limit == 2048)
                    .unwrap()
                    .1;
                let preview = make_preview_texture_data(source, &io, true);
                let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../target/catalog-load-benchmark");
                std::fs::create_dir_all(&dir).unwrap();
                image::save_buffer(
                    dir.join(format!("{}-{index}.png", app.raw_options.decoder)),
                    &preview.rgba,
                    preview.size[0] as u32,
                    preview.size[1] as u32,
                    image::ColorType::Rgba8,
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn rapid_navigation_keeps_latest_geometry_and_revisits_use_decoded_cache() {
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx);
        let mut app = App::new(
            &cc,
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            None,
        );
        app.raw_options = spektrafilm_io::RawOptions {
            decoder: "rawler".into(),
            ..Default::default()
        };
        app.params.camera.exposure_compensation_ev = 1.25;
        app.params.workflow.route = "input > convert-film > print > scan".into();
        app.params.camera.color_filter = "hoya_y2".into();
        app.params.film_render.chemistry.active = true;
        app.params.film_render.chemistry.gamma_factor = 1.18;
        app.params.print_render.chemistry.active = true;
        app.params.print_render.chemistry.developer_exhaustion = 0.3;
        app.params.film_render.base.active = true;
        app.params.film_render.base.cyan = 1.2;
        app.params.film_render.convert.calibration = "1 .01 0 0 1 0 0 0 1".into();
        let workflow_controls = |p: &RuntimeParams| {
            serde_json::json!([
                p.workflow,
                p.camera.color_filter,
                p.film_render.chemistry,
                p.print_render.chemistry,
                p.film_render.base,
                p.film_render.convert
            ])
        };
        let expected_controls = workflow_controls(&app.params);
        let dir = tempfile::tempdir().unwrap();
        let photo = |id, orientation, crop| {
            let path = dir.path().join(format!("{id}.dng"));
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
                flag: 1,
                geometry: Ok(Geometry {
                    orientation,
                    crop,
                    angle: 0.,
                }),
                framing_warning: false,
            }
        };
        let first = photo(1, 1, [0., 0., 1., 1.]);
        let second = photo(2, 6, [0.25, 0., 1., 0.5]);
        let latest = photo(3, 8, [0., 0., 0.5, 1.]);
        app.select_catalog_photo(first.clone());
        app.select_catalog_photo(second.clone());
        app.select_catalog_photo(latest.clone());
        assert!(app.pending_load.is_some());
        finish(&mut app);
        assert_eq!(app.image_path.as_ref(), Some(&latest.path));
        assert_eq!(
            (
                app.image.as_ref().unwrap().width,
                app.image.as_ref().unwrap().height
            ),
            (32, 16)
        );
        assert_eq!(app.params.camera.exposure_compensation_ev, 1.25);
        assert_eq!(workflow_controls(&app.params), expected_controls);
        let latest_frame = app.uncropped.as_ref().unwrap().clone();
        let latest_pixels = app.image.as_ref().unwrap().clone();
        app.select_catalog_photo(first);
        finish(&mut app);
        app.select_catalog_photo(latest.clone());
        finish(&mut app);
        // The decoded frame is reused; the crop is applied to it again.
        assert!(Arc::ptr_eq(app.uncropped.as_ref().unwrap(), &latest_frame));
        assert_eq!(app.image.as_ref().unwrap().data, latest_pixels.data);
        let path = dir.path().join("project.json");
        app.project_state().save(&path).unwrap();
        let mut reopened = App::new(
            &cc,
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            None,
        );
        reopened.restore_project(&path, true).unwrap();
        finish(&mut reopened);
        assert_eq!(reopened.image.as_ref().unwrap().data, latest_pixels.data);
        assert_eq!(workflow_controls(&reopened.params), expected_controls);
        assert_eq!(
            reopened.export_raw_options().lightroom_geometry,
            latest.geometry.ok()
        );
        app.open_catalog(dir.path().join("missing.lrcat"));
        assert!(app.image.is_none() && app.image_path.is_none());
        assert!(app.pending_load.is_none());
    }
}

#[cfg(test)]
mod preview_cache_tests {
    use super::*;

    #[test]
    #[ignore = "requires SPEKTRAFILM_BENCH_IMAGE and local originals"]
    fn benchmark_setting_changes() {
        let path = std::env::var_os("SPEKTRAFILM_BENCH_IMAGE").expect("image path");
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut app = App::new(&cc, Arc::from(spektrafilm_gpu::select_backend()), None);
        app.params = RuntimeParams::default();
        app.film_name = "kodak_gold_200".into();
        app.print_name = "kodak_portra_endura".into();
        app.raw_options = spektrafilm_io::RawOptions {
            decoder: "rawler".into(),
            ..Default::default()
        };
        app.load_image_from_path(Path::new(&path));
        while let Some(job) = &mut app.load_job {
            if let Some(handle) = job.handle.take() {
                handle.join().unwrap();
            }
            app.poll_load_job();
        }
        assert!(app.image.is_some(), "{}", app.status);
        for limit in [640, 2048] {
            app.params.settings.preview_max_size = limit;
            for setting in ["exposure", "print_exposure", "halation"] {
                for step in 0..5 {
                    match setting {
                        "exposure" => app.params.camera.exposure_compensation_ev = step as f32 / 4.,
                        "print_exposure" => {
                            app.params.enlarger.print_exposure = 1. + step as f32 / 4.
                        }
                        "halation" => app.params.film_render.halation.boost_ev = step as f32 / 4.,
                        _ => unreachable!(),
                    }
                    let start = Instant::now();
                    app.dispatch_render(&ctx);
                    let dispatch_ms = start.elapsed().as_secs_f64() * 1000.;
                    while let Some(job) = &mut app.render_job {
                        if let Some(handle) = job.handle.take() {
                            handle.join().unwrap();
                        }
                        app.poll_render_job(&ctx);
                    }
                    assert!(app.output_image.is_some(), "{}", app.status);
                    eprintln!(
                        "[settings-benchmark] limit={limit} setting={setting} step={step} dispatch={dispatch_ms:.3} build={:.3} resize={:.3} render={:.3} display={:.3} worker={:.3} total={:.3}",
                        app.last_pipeline_build_ms,
                        app.last_scale_ms,
                        app.last_render_ms,
                        app.last_preview_ms,
                        app.last_worker_total_ms,
                        start.elapsed().as_secs_f64() * 1000.
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(unix)]
    fn file_stamp_invalidates_replacements_with_the_same_size_and_modification_time() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("original");
        let replacement = dir.path().join("replacement");
        std::fs::write(&path, b"aaaa").unwrap();
        let before = image_stamp(&path).unwrap();
        let old = std::fs::File::open(&path).unwrap();
        let modified = old.metadata().unwrap().modified().unwrap();
        std::fs::write(&replacement, b"bbbb").unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&replacement)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            modified
        );
        assert_ne!(image_stamp(&path).unwrap(), before);
    }
    #[test]
    fn pipeline_cache_tracks_selected_profile_files_and_data_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("profiles")).unwrap();
        let film = dir.path().join("profiles/film.json");
        let paper = dir.path().join("profiles/paper.json");
        std::fs::write(&film, b"film v1").unwrap();
        std::fs::write(&paper, b"paper v1").unwrap();
        let mut params = RuntimeParams::default();
        let key = |p: &RuntimeParams| preview_pipeline_cache_key(dir.path(), "film", "paper", p);
        let first = key(&params);
        std::fs::write(dir.path().join("profiles/unused.json"), b"unrelated").unwrap();
        assert_eq!(first, key(&params));
        std::fs::write(&film, b"film version two").unwrap();
        let second = key(&params);
        assert_ne!(first, second);
        std::fs::write(&paper, b"paper version two").unwrap();
        assert_ne!(second, key(&params));
        assert_ne!(
            key(&params),
            preview_pipeline_cache_key(&dir.path().join("other"), "film", "paper", &params)
        );

        let spectral = dir.path().join("luts/spectral_upsampling");
        std::fs::create_dir_all(&spectral).unwrap();
        std::fs::write(spectral.join("method.toml"), b"descriptor").unwrap();
        std::fs::write(spectral.join("method.npy"), b"spectra").unwrap();
        let before = key(&params);
        std::fs::write(spectral.join("method.npy"), b"new spectra").unwrap();
        assert_ne!(before, key(&params));
        let before = key(&params);
        std::fs::write(spectral.join("method.toml"), b"new descriptor").unwrap();
        assert_ne!(before, key(&params));
        std::fs::create_dir(dir.path().join("filters")).unwrap();
        let before = key(&params);
        std::fs::write(
            dir.path().join("filters/neutral_print_filters.json"),
            b"filters",
        )
        .unwrap();
        assert_ne!(before, key(&params));

        params.workflow.route = "input > film > scan".into();
        let scan_key = key(&params);
        std::fs::remove_file(&paper).unwrap();
        assert_eq!(
            scan_key,
            key(&params),
            "an unused paper file must not invalidate a film scan"
        );
        std::fs::remove_file(&film).unwrap();
        assert_ne!(scan_key, key(&params));
    }

    #[test]
    fn cached_previews_match_rebuilt_pipelines_after_control_changes() {
        use serde_json::json;
        let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        let build = |film: &str, paper: &str, params: RuntimeParams| {
            let print = if params.scans_film() { film } else { paper };
            Pipeline::new_with_spectral(
                profile::load_profile_by_name(&data, film).unwrap(),
                profile::load_profile_by_name(&data, print).unwrap(),
                params,
                &data,
            )
            .unwrap()
        };
        let changes = [
            ("/camera/tone/highlights", json!(-80.0)),
            ("/camera/tone/shadows", json!(65.0)),
            ("/camera/tone/whites", json!(40.0)),
            ("/camera/tone/blacks", json!(-35.0)),
            ("/camera/exposure_compensation_ev", json!(1.0)),
            ("/camera/auto_exposure", json!(false)),
            ("/camera/auto_exposure_method", json!("matrix")),
            ("/camera/film_format_mm", json!(24.0)),
            ("/camera/lens_blur_um", json!(12.0)),
            ("/camera/filter_uv", json!([0.5, 410.0, 8.0])),
            ("/camera/filter_ir", json!([0.5, 675.0, 15.0])),
            ("/camera/diffusion_filter/active", json!(true)),
            ("/enlarger/print_exposure", json!(1.3)),
            ("/enlarger/print_exposure_compensation", json!(false)),
            ("/enlarger/normalize_print_exposure", json!(false)),
            ("/enlarger/m_filter_shift", json!(8.0)),
            ("/enlarger/preflash_exposure", json!(0.08)),
            ("/enlarger/diffusion_filter/active", json!(true)),
            ("/film_render/density_curve_gamma", json!(1.15)),
            ("/film_render/channel_order", json!([2, 1, 0])),
            ("/film_render/development_time", json!(12.0)),
            ("/film_render/dir_couplers/amount", json!(1.4)),
            ("/film_render/halation/boost_ev", json!(2.0)),
            ("/print_render/density_curve_gamma", json!(1.15)),
            ("/print_render/development_time", json!(12.0)),
            (
                "/print_render/chemistry",
                json!({"active":true,"gamma_factor":1.2,"developer_exhaustion":0.25}),
            ),
            ("/io/scan_film", json!(true)),
            ("/io/input_color_space", json!("ACEScg")),
            ("/io/output_color_space", json!("Display P3")),
            ("/io/output_cctf_encoding", json!(false)),
            ("/settings/preview_mode", json!(true)),
            ("/settings/use_cat16", json!(false)),
            ("/settings/spectral_gaussian_blur", json!(1.0)),
            (
                "/settings/apply_hanatos2025_adaptation_surface",
                json!(true),
            ),
            ("/settings/rgb_to_raw_method", json!("mallett2019")),
            ("/camera/color_filter", json!("hoya_y2")),
            ("/film_render/chemistry/gamma_factor", json!(1.18)),
            ("/film_render/chemistry/developer_exhaustion", json!(0.3)),
            ("/film_render/base/cyan", json!(1.2)),
            ("/print_render/base/magenta", json!(0.85)),
            ("/io/input_gamut_compress/hull_detail", json!(8.0)),
            ("/film_render/convert/exposure_compensation_ev", json!(0.3)),
            (
                "/film_render/convert/calibration",
                json!("1 .01 0 0 1 0 0 0 1"),
            ),
            ("/film_render/convert/scan_illuminant", json!("D65")),
            ("/workflow/route", json!("input > convert-film > scan")),
            (
                "/settings/neutral_print_filters_from_database",
                json!(false),
            ),
            ("/scanner/white_correction", json!(true)),
            ("/scanner/black_correction", json!(true)),
        ];
        let image = ImageBuf::from_data(
            32,
            16,
            (0..32 * 16 * 3)
                .map(|i| spektrafilm_math::precision::from_f64(0.03 + 0.5 * (i % 32) as f64 / 31.))
                .collect(),
        );
        for (film, paper) in [
            ("kodak_gold_200", "kodak_portra_endura"),
            ("kodak_doublex", "kodak_2302"),
        ] {
            let mut params = RuntimeParams::default();
            params.debug.deactivate_stochastic_effects = true;
            params.camera.auto_exposure = true;
            let template = build(film, paper, params.clone());
            let original_key = preview_pipeline_cache_key(&data, film, paper, &params);
            for (path, value) in &changes {
                let mut edited = serde_json::to_value(&params).unwrap();
                *edited.pointer_mut(path).unwrap() = value.clone();
                let changed: RuntimeParams = serde_json::from_value(edited).unwrap();
                if path.starts_with("/camera/tone/") {
                    assert_eq!(
                        preview_pipeline_cache_key(&data, film, paper, &changed),
                        original_key,
                        "tone changes must reuse spectral calibration"
                    );
                }
                let expected = build(film, paper, changed.clone());
                let actual =
                    if preview_pipeline_cache_key(&data, film, paper, &changed) == original_key {
                        template.clone().with_params(changed)
                    } else {
                        build(film, paper, changed)
                    };
                let expected =
                    expected.process(image.clone(), &spektrafilm_gpu::cpu_backend::CpuBackend);
                let actual =
                    actual.process(image.clone(), &spektrafilm_gpu::cpu_backend::CpuBackend);
                assert!(
                    expected
                        .data
                        .iter()
                        .chain(&actual.data)
                        .all(|v| v.is_finite()),
                    "{film} {path}: nonfinite output"
                );
                let max = expected
                    .data
                    .iter()
                    .zip(&actual.data)
                    .map(|(a, b)| (*a as f64 - *b as f64).abs())
                    .fold(0., f64::max);
                assert!(
                    max < 1e-6,
                    "{film} {path}: cached preview differs from a fresh export by {max}"
                );
            }
        }
    }
}

#[cfg(test)]
mod gpu_export_tests {
    use super::*;

    #[test]
    fn scheduled_exports_match_interactive_gpu_processing() {
        let backend = spektrafilm_gpu::select_backend();
        if !backend.is_gpu() {
            return;
        }
        let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        let image = ImageBuf::from_data(
            63,
            31,
            (0..63 * 31 * 3)
                .map(|i| spektrafilm_math::precision::from_f64(0.03 + 0.6 * (i % 63) as f64 / 62.))
                .collect(),
        );
        let scheduler = Scheduler::default();
        for route in [
            "input > film > print > scan",
            "input > convert-film > print > scan",
        ] {
            let mut params = RuntimeParams::default();
            params.workflow.route = route.into();
            params.debug.deactivate_stochastic_effects = true;
            let pipeline = Pipeline::new_with_spectral(
                profile::load_profile_by_name(&data, "kodak_gold_200").unwrap(),
                profile::load_profile_by_name(&data, "kodak_portra_endura").unwrap(),
                params,
                &data,
            )
            .unwrap();
            let render = |priority| {
                let task = scheduler.task(priority);
                let _scope = task.enter();
                pipeline
                    .process_resident_borrowed(&image, backend.as_ref())
                    .expect("GPU resident path")
            };
            let expected = render(Priority::Interactive);
            let actual = render(Priority::Export);
            assert_eq!(
                (actual.width, actual.height),
                (expected.width, expected.height)
            );
            assert_eq!(
                actual.data, expected.data,
                "scheduled export changed {route}"
            );
        }
    }

    #[test]
    fn gpu_export_uses_original_and_honors_working_scale_and_saving_color() {
        let backend = spektrafilm_gpu::select_backend();
        let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        let image = ImageBuf::from_data(
            64,
            32,
            (0..64 * 32 * 3)
                .map(|i| spektrafilm_math::precision::from_f64(0.04 + 0.4 * (i % 64) as f64 / 63.))
                .collect(),
        );
        let mut params = RuntimeParams::default();
        params.settings.preview_max_size = 16;
        params.debug.deactivate_stochastic_effects = true;
        let options = spektrafilm_io::SaveOptions {
            color_space: "Display P3".into(),
            ..Default::default()
        };
        let path =
            std::env::temp_dir().join(format!("spektrafilm-gpu-export-{}.png", std::process::id()));
        for (scale, dimensions) in [(1., (64, 32)), (0.5, (32, 16))] {
            params.io.upscale_factor = scale;
            let metadata = spektrafilm_io::Metadata::default()
                .with_edit_settings(&spektrafilm_io::EditSettings {
                    film: "kodak_gold_200",
                    paper: "kodak_portra_endura",
                    params: &params,
                    raw: &Default::default(),
                    rotation: 0,
                    crop: Crop::FULL,
                })
                .unwrap();
            run_gpu_export(
                &image,
                &path,
                "kodak_gold_200",
                "kodak_portra_endura",
                &params,
                &data,
                backend.as_ref(),
                &AtomicBool::new(false),
                &options,
                &metadata,
            )
            .unwrap();
            let loaded = spektrafilm_io::load(&path, &Default::default()).unwrap();
            assert_eq!((loaded.pixels.width, loaded.pixels.height), dimensions);
            assert_eq!(loaded.color_space, "Display P3");
            assert!(loaded.encoded);
            let tags = loaded.metadata.native.as_ref().unwrap();
            let comment = tags
                .get_tag_by_hex(0x9286, None)
                .next()
                .unwrap()
                .value_as_u8_vec(&tags.get_endian());
            let recipe: serde_json::Value =
                serde_json::from_slice(comment.strip_prefix(b"ASCII\0\0\0").unwrap()).unwrap();
            assert_eq!(recipe["settings"]["params"]["io"]["upscale_factor"], scale);
            assert_eq!(recipe["output"]["color_space"], "Display P3");
        }
        std::fs::remove_file(path).unwrap();
    }
}
