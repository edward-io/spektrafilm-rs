// Native editor for spektrafilm.
//
// Loads an image, exposes the most impactful runtime parameters as
// egui controls, and re-renders the full GPU-resident pipeline whenever
// a control changes. The output texture is uploaded to egui once per
// render; the panel scales it to fit.

mod calibration_ui;
mod catalog_ui;
mod icons;
mod lut_ui;
mod paper_back;
mod project;
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
use rayon::prelude::*;
use spektrafilm_core::params::RuntimeParams;
use spektrafilm_core::pipeline::Pipeline;
use spektrafilm_core::profile;
use spektrafilm_gpu::ComputeBackend;
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
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 900.0]),
        renderer: gui_renderer(),
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
    load_job: Option<LoadJob>,
    pending_load: Option<LoadRequest>,
    image_cache: std::collections::VecDeque<CachedImage>,
    catalog: Option<catalog_ui::Browser>,
    preview_input: Vec<(u32, Arc<ImageBuf>)>,
    rendered_preview_limit: u32,
    render_path: String,
    auto_preview: bool,
    gpu_export: bool,
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
    /// Settings that produced the current pixels; controls can contain edits
    /// waiting for an explicit RAW reload.
    loaded_raw_options: Option<spektrafilm_io::RawOptions>,
    lens_database_status: LensDatabaseStatus,
    save_options: spektrafilm_io::SaveOptions,
    image_path: Option<PathBuf>,
    image: Option<Arc<ImageBuf>>,
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
}

struct LoadJob {
    discard: bool,
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

struct LoadRequest {
    path: PathBuf,
    rotation: u8,
    color_override: Option<(String, bool)>,
    view_override: Option<(f32, egui::Vec2)>,
    raw_options: spektrafilm_io::RawOptions,
}

struct CachedImage {
    path: PathBuf,
    rotation: u8,
    options: spektrafilm_io::RawOptions,
    source: SourceImage,
}

fn image_stamp(path: &Path) -> Option<(u64, std::time::SystemTime)> {
    let metadata = std::fs::metadata(path).ok()?;
    Some((metadata.len(), metadata.modified().ok()?))
}

/// One in-flight preview render. The worker owns a Pipeline + the
/// ImageBuf clone and, when it finishes, sends back the output buffer
/// plus the two timings the status bar shows.
struct RenderJob {
    discard: bool,
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
    started_at: Instant,
    label: String,
}

impl App {
    fn new(
        cc: &eframe::CreationContext<'_>,
        backend: Arc<dyn ComputeBackend>,
        initial_image: Option<PathBuf>,
    ) -> Self {
        let _ = cc;

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
            load_job: None,
            pending_load: None,
            image_cache: Default::default(),
            catalog: None,
            preview_input: Vec::new(),
            rendered_preview_limit: 0,
            render_path: "Awaiting render".into(),
            auto_preview: true,
            gpu_export: true,
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
            status: String::from("Load an image to start."),
            dirty: false,
            #[cfg(target_os = "macos")]
            metal_colorspace_tagged: false,
            export_job: None,
        };
        if project::default_path().exists() {
            if let Err(e) = app.restore_project(&project::default_path(), false) {
                app.status = format!("Startup defaults: {e:#}");
            }
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
        self.raw_options.lightroom_geometry = None;
        if let Some(catalog) = &mut self.catalog {
            catalog.selected = None;
        }
        self.load_image_with_state(path, 0, None, None);
    }

    fn clear_image(&mut self) {
        self.image = None;
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
        if let Some(job) = &mut self.load_job {
            job.discard = true;
        }
        if let Some(job) = &mut self.render_job {
            job.discard = true;
        }
        self.pending_load = None;
        self.clear_image();
        self.raw_options.lightroom_geometry = None;
        self.catalog = Some(catalog_ui::Browser::open(path, &self.ctx));
        self.status = "Reading Lightroom catalog…".into();
    }

    fn select_catalog_photo(&mut self, photo: spektrafilm_io::lightroom::Photo) {
        match photo.geometry {
            Ok(geometry) => {
                self.raw_options.lightroom_geometry = Some(geometry);
                self.params.io.crop = false;
                self.load_image_with_state(&photo.path, 0, None, None);
            }
            Err(error) => {
                if let Some(job) = &mut self.load_job {
                    job.discard = true;
                }
                if let Some(job) = &mut self.render_job {
                    job.discard = true;
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
        if let Some(render) = &mut self.render_job {
            render.discard = true;
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
        };
        if self.image_path.as_ref() != Some(&request.path)
            || self.export_raw_options().lightroom_geometry
                != request.raw_options.lightroom_geometry
        {
            self.clear_image();
        }
        if let Some(job) = &mut self.load_job {
            job.discard = true;
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

    fn start_image_load(&mut self, request: LoadRequest) {
        let LoadRequest {
            path,
            rotation,
            color_override,
            view_override,
            raw_options: options,
        } = request;
        let display_transform = self.display.display_transform;
        let preview_color = color_override.clone();
        let cached = self
            .image_cache
            .iter()
            .find(|cached| {
                cached.path == path && cached.rotation == rotation && cached.options == options
            })
            .map(|cached| cached.source.clone());
        let worker_path = path.clone();
        let loaded_options = options.clone();
        let ctx = self.ctx.clone();
        let (tx, rx) = mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("spektrafilm-load".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let started = Instant::now();
                    let stamp = image_stamp(&worker_path);
                    let source =
                        match cached.filter(|source| stamp.is_some() && source.stamp == stamp) {
                            Some(source) => source,
                            None => {
                                let mut loaded = spektrafilm_io::load(&worker_path, &options)
                                    .map_err(|e| format!("{e:#}"))?;
                                if rotation % 4 != 0 {
                                    loaded.pixels = loaded.pixels.rotated_quarter_turns(rotation);
                                }
                                SourceImage {
                                    pixels: Arc::new(loaded.pixels),
                                    metadata: loaded.metadata,
                                    color_space: loaded.color_space,
                                    encoded: loaded.encoded,
                                    stamp,
                                }
                            }
                        };
                    let loaded = started.elapsed();
                    let input = Arc::new(spektrafilm_core::resize::preview(&source.pixels, 2048));
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
                    let preview = make_preview_texture_data(&input, &io, display_transform);
                    let displayed = started.elapsed();
                    let paper = paper_back::cached(preview.size[0], preview.size[1]);
                    eprintln!("[spektrafilm] load {}: decode {:.1} ms, resize {:.1} ms, display {:.1} ms, paper {:.1} ms", worker_path.display(), loaded.as_secs_f64() * 1000., (resized - loaded).as_secs_f64() * 1000., (displayed - resized).as_secs_f64() * 1000., (started.elapsed() - displayed).as_secs_f64() * 1000.);
                    Ok(LoadedPreview {
                        source,
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
                    input,
                    preview,
                    paper,
                    source_key,
                } = display;
                if let Some(catalog) = &mut self.catalog {
                    catalog.source_thumbnail(&self.ctx, &job.path, &preview);
                }
                self.image_cache.retain(|cached| {
                    cached.path != job.path
                        || cached.options != job.raw_options
                        || cached.rotation != job.rotation
                });
                self.image_cache.push_back(CachedImage {
                    path: job.path.clone(),
                    rotation: job.rotation,
                    options: job.raw_options.clone(),
                    source: source.clone(),
                });
                // Keep the current source plus at most 512 MiB of previously developed images.
                let mut previous_bytes: usize = self
                    .image_cache
                    .iter()
                    .rev()
                    .skip(1)
                    .map(|cached| std::mem::size_of_val(cached.source.pixels.data.as_slice()))
                    .sum();
                while previous_bytes > 512 * 1024 * 1024 {
                    if let Some(cached) = self.image_cache.pop_front() {
                        previous_bytes -=
                            std::mem::size_of_val(cached.source.pixels.data.as_slice());
                    }
                }
                self.loaded_raw_options = Some(job.raw_options);
                self.metadata = source.metadata;
                let (color, encoded) = job
                    .color_override
                    .unwrap_or((source.color_space, source.encoded));
                self.params.io.input_color_space = color;
                self.params.io.input_cctf_decoding = encoded;
                self.rotation = job.rotation;
                let img = source.pixels;
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
                    render.discard = true;
                }
                self.dirty = true;
            }
            Err(e) => {
                self.status = format!("Load error: {e}");
                eprintln!("[spektrafilm] {}", self.status);
            }
        }
    }

    fn rotate_input(&mut self, turns: u8) {
        let (Some(image), Some(path)) = (self.image.clone(), self.image_path.clone()) else {
            return;
        };
        if self.load_job.is_some() {
            return;
        }
        let rotation = (self.rotation + turns) % 4;
        let metadata = self.metadata.clone();
        let color = self.params.io.input_color_space.clone();
        let encoded = self.params.io.input_cctf_decoding;
        let display_transform = self.display.display_transform;
        let ctx = self.ctx.clone();
        let (tx, rx) = mpsc::channel();
        let worker_color = color.clone();
        let worker_path = path.clone();
        let handle = std::thread::Builder::new()
            .name("spektrafilm-rotate".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let pixels = image.rotated_quarter_turns(turns);
                    let input = Arc::new(spektrafilm_core::resize::preview(&pixels, 2048));
                    let mut io = spektrafilm_core::params::IoParams::default();
                    io.output_color_space = worker_color.clone();
                    io.output_cctf_encoding = encoded;
                    let source_key = (
                        io.output_color_space.clone(),
                        io.output_cctf_encoding,
                        display_transform,
                    );
                    let preview = make_preview_texture_data(&input, &io, display_transform);
                    let paper = paper_back::cached(preview.size[0], preview.size[1]);
                    Ok(LoadedPreview {
                        source: SourceImage {
                            pixels: Arc::new(pixels),
                            metadata,
                            color_space: worker_color,
                            encoded,
                            stamp: image_stamp(&worker_path),
                        },
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
                    job.discard = true;
                }
                self.status = "Rotating input…".into();
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
        let params = self.params.clone();
        let cache_key = preview_pipeline_cache_key(&film_name, &print_name, &params);
        let pipeline_template = if self.pipeline_cache_key.as_deref() == Some(cache_key.as_str()) {
            self.pipeline_cache.clone()
        } else {
            None
        };
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
        let handle = std::thread::Builder::new()
            .name("spektrafilm-render".into())
            .spawn(move || {
                // A panic inside the pipeline must still produce a channel
                // message — otherwise the receiver only sees a disconnect
                // and the user gets a uselessly vague status line.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
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
                    let pipeline = match pipeline_template {
                        Some(pipeline) => pipeline.with_params(params),
                        None => {
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
                    let t_preview = Instant::now();
                    let preview =
                        make_preview_texture_data(&output, &pipeline.params.io, display_transform);
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
                            make_preview_texture_data(&source, &source_io, display_transform),
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
            rx,
            handle: Some(handle),
        });
    }

    /// Central-panel preview with mouse-wheel zoom (centred on the
    /// cursor) and click-drag pan. Double-click resets to fit.
    /// `self.zoom = 1.0` means fit-to-panel; values >1 zoom in.
    fn draw_preview_with_zoom(&mut self, ui: &mut egui::Ui) {
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
        let fit_scale = (avail.x / paper_size.x)
            .min(avail.y / paper_size.y)
            .min(1.0);
        let scale = fit_scale * self.zoom;
        let drawn_size = img_size * scale;

        let response = ui.allocate_response(avail, egui::Sense::click_and_drag());
        let rect = response.rect;

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
                let centre = rect.center() + self.pan;
                let offset_from_centre = cur - centre;
                let scale_ratio = self.zoom / old_zoom;
                self.pan += offset_from_centre * (1.0 - scale_ratio);
            }
        }

        if response.dragged() {
            self.pan += response.drag_delta();
        }
        if response.double_clicked() {
            self.zoom = 1.0;
            self.pan = egui::Vec2::ZERO;
        }

        let centre = rect.center() + self.pan;
        let source_rect = egui::Rect::from_center_size(centre, drawn_size);
        let out_size = output_size.unwrap_or(img_size);
        let out_scale = (drawn_size.x / out_size.x).min(drawn_size.y / out_size.y);
        let draw_rect = egui::Rect::from_center_size(centre, out_size * out_scale);
        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        // Clip to the central panel so panned-off pixels don't leak.
        let painter = ui.painter_at(rect);
        if self.display.show_border && border > 0. {
            painter.rect_filled(source_rect.expand(border * scale), 0., egui::Color32::WHITE);
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
        }

        // Zoom indicator + reset chip in the corner.
        let pct = (self.zoom * fit_scale * 100.0).round() as i32;
        let text = if (self.zoom - 1.0).abs() < 0.001 && self.pan.length_sq() < 1.0 {
            "fit".to_string()
        } else {
            format!("{pct} %  (double-click to reset)")
        };
        let pos = rect.left_top() + egui::vec2(8.0, 8.0);
        painter.text(
            pos,
            egui::Align2::LEFT_TOP,
            text,
            egui::FontId::monospace(12.0),
            egui::Color32::from_rgba_premultiplied(220, 220, 220, 200),
        );
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
                    self.canvas_previous = self
                        .canvas_output
                        .take()
                        .filter(|old| old.size == r.preview.size);
                    self.animation = if self.display.animate
                        && r.preview.size[0] * r.preview.size[1] <= 1_500_000
                        && !ctx.is_using_pointer()
                    {
                        Some((Instant::now(), self.canvas_previous.is_none()))
                    } else {
                        None
                    };
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
                self.status = format!(
                    "Rendered {} × {} ({:.1} MP)",
                    r.output.width,
                    r.output.height,
                    r.output.pixel_count() as f64 / 1e6
                );
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

    /// Open a save dialog and write the most recent rendered output to
    /// disk. Suggested filename is the input stem + the chosen film
    /// stock + the chosen extension; default extension is PNG (8-bit
    /// sRGB-encoded, matching what's on screen).
    fn save_dialog(&mut self) {
        let Some(out) = self.output_image.clone() else {
            self.status = "Nothing to save yet — load an image first.".into();
            return;
        };
        let default_name = match &self.image_path {
            Some(p) => {
                let stem = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("spektrafilm");
                format!("{stem}_{}_spektra.png", self.film_name)
            }
            None => "spektrafilm.png".into(),
        };
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Image", &["png", "jpg", "jpeg", "tif", "tiff", "exr"])
            .set_file_name(&default_name)
            .save_file()
        else {
            return;
        };
        let t = Instant::now();
        match spektrafilm_io::save_converted(
            &out,
            &self.params.io.output_color_space,
            self.params.io.output_cctf_encoding,
            &path,
            &self.current_save_options(),
            &self.metadata,
            &self.data_dir,
        ) {
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
        let stem = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("spektrafilm");
        let default_name = format!("{stem}_{}_spektra.png", self.film_name);
        let Some(out_path) = rfd::FileDialog::new()
            .add_filter("Image", &["png", "tif", "tiff", "jpg", "jpeg", "exr"])
            .set_file_name(&default_name)
            .save_file()
        else {
            return;
        };

        let film = self.film_name.clone();
        let paper = self.print_name.clone();
        let mut params = self.params.clone();
        params.settings.preview_mode = false;
        let data_dir = self.data_dir.clone();
        let raw_options = self.export_raw_options().clone();
        let rotation = self.rotation;
        let save_options = self.current_save_options();
        let metadata = self.metadata.clone();
        let image = self.image.clone();
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
                    &params,
                    &data_dir,
                    cancel,
                    &raw_options,
                    &save_options,
                    rotation,
                );
            }
            run_gpu_export(
                image.context("source image is still loading")?.as_ref(),
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
        let started_at = Instant::now();
        let handle = std::thread::Builder::new()
            .name("spektrafilm-export".into())
            .spawn(move || {
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&worker_cancel)))
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

    fn controls_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        // Control captions are hover targets, not selectable text; the separate
        // status bar remains selectable for copying errors.
        ui.style_mut().interaction.selectable_labels = false;
        ui.heading("spektrafilm");
        ui.add_space(6.0);

        // ── File ────────────────────────────────────────────────────────
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Open…")
                .on_hover_text(tooltips::CONTROL_OPEN)
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter(
                        "Image",
                        &[
                            "png", "jpg", "jpeg", "tif", "tiff", "exr", // standard
                            "dng", "cr2", "cr3", "nef", "nrw", "arw", "srf", "sr2", "raf", "orf",
                            "rw2", "pef", "srw", "x3f", "iiq", "3fr", "crw", "rwl", "mrw", "mef",
                            "kdc",
                        ],
                    )
                    .pick_file()
                {
                    self.load_image_from_path(&path);
                }
            }
            if ui
                .button("Catalog…")
                .on_hover_text("Browse a Lightroom Classic catalog read-only")
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Lightroom Classic catalog", &["lrcat"])
                    .pick_file()
                {
                    self.open_catalog(path);
                }
            }
            let save_enabled = self.output_image.is_some()
                && self.load_job.is_none()
                && self.render_job.is_none()
                && !self.dirty
                && !self.pending_dirty;
            if ui
                .add_enabled(save_enabled, egui::Button::new("Save preview…"))
                .on_hover_text(tooltips::CONTROL_SAVE_PREVIEW)
                .on_disabled_hover_text("Render a preview first")
                .clicked()
            {
                self.save_dialog();
            }
            let export_busy = self.export_job.is_some();
            if export_busy {
                if ui
                    .button("Cancel")
                    .on_hover_text(tooltips::CANCEL_EXPORT)
                    .clicked()
                {
                    self.cancel_export();
                }
            } else {
                let export_enabled = self.image_path.is_some() && self.load_job.is_none();
                if ui
                    .add_enabled(export_enabled, egui::Button::new("Export…"))
                    .on_hover_text(tooltips::EXPORT_IMAGE)
                    .on_disabled_hover_text("Load an image first")
                    .clicked()
                {
                    self.export_dialog(ctx);
                }
            }
        });
        if let Some(p) = &self.image_path {
            ui.label(
                egui::RichText::new(p.file_name().and_then(|s| s.to_str()).unwrap_or("")).small(),
            );
        }
        ui.add_space(4.0);

        self.project_controls(ui);
        self.lut_controls(ui, ctx);
        ui.horizontal(|ui| {
            tooltips::checkbox(
                ui,
                &mut self.auto_preview,
                "Auto preview",
                tooltips::CONTROL_AUTO_PREVIEW,
            );
            if ui
                .add_enabled(self.image.is_some(), egui::Button::new("Render preview"))
                .on_hover_text(tooltips::CONTROL_RENDER_PREVIEW)
                .on_disabled_hover_text(tooltips::CONTROL_RENDER_PREVIEW)
                .clicked()
            {
                self.render_requested = true;
                self.dirty = true;
                self.dirty_since = None;
            }
        });
        let route = self.params.route().to_owned();
        let mut selected = route.clone();
        tooltips::combo(
            ui,
            egui::ComboBox::from_id_salt("workflow").selected_text(match selected.as_str() {
                "input" => "Input only",
                "input > film > scan" => "Film → scan",
                "input > film > print > scan" => "Film → print → scan",
                "input > convert-film > print > scan" => "Negative scan → print",
                "input > convert-film > scan-minus-base" => "Negative scan → dyes",
                _ => "Negative scan → scan",
            }),
            "Workflow",
            "Choose whether to simulate film, convert a scanned negative, or pass the input through.",
            |ui| {
                for route in [
                    "input",
                    "input > film > scan",
                    "input > film > print > scan",
                    "input > convert-film > print > scan",
                    "input > convert-film > scan-minus-base",
                    "input > convert-film > scan",
                ] {
                    ui.selectable_value(&mut selected,route.to_owned(),route).on_hover_text(match route {
                    "input"=>"Convert input to the output color space without film simulation.",
                    "input > film > scan"=>"Expose and develop film, then scan it directly.",
                    "input > film > print > scan"=>"Expose film, print it onto paper, then scan the print.",
                    "input > convert-film > print > scan"=>"Recover dye densities from a scanned negative and print them.",
                    "input > convert-film > scan-minus-base"=>"Recover film dye densities and scan them without the film base.",
                    _=>"Recover film dye densities and scan them under the selected scan illuminant.",
                });
                }
            },
        );
        if selected != route {
            self.params.workflow.route = selected;
            self.params.digest();
            self.dirty = true;
        }
        // ── Profiles ────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Profiles")
            .default_open(true)
            .show(ui, |ui| {
                let film_changed =
                    profile_combo(ui, "film", "Film stock", &self.films, &mut self.film_name);
                if film_changed {
                    self.film_dev_times = Vec::new();
                    self.params.film_render.chemistry.development_time = None;
                    if let Ok(film) = profile::load_profile_by_name(&self.data_dir, &self.film_name)
                    {
                        // Slide/positive stocks have no print paper — they
                        // are scanned directly. Negatives print onto their
                        // paired `target_print`. Auto-follow the film type
                        // so switching stocks doesn't leave a wrong (or, for
                        // a paperless slide, a failed) render.
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
                            self.print_dev_times =
                                profile_dev_times(&self.data_dir, &self.print_name);
                            self.params.print_render.chemistry.development_time = None;
                        }
                    }
                    self.dirty = true;
                }
                // B&W stocks are profiled at several development times —
                // pick one (longer = more contrast). Hidden for the
                // single-time / colour case.
                if self.film_dev_times.len() > 1
                    && dev_time_combo(
                        ui,
                        "film_dev_time",
                        "Development time",
                        &self.film_dev_times.clone(),
                        &mut self.params.film_render.chemistry.development_time,
                    )
                {
                    self.dirty = true;
                }
                // Slide films are scanned directly, so the print paper is
                // unused — disable the picker and say why rather than
                // letting it silently affect nothing.
                if self.params.scans_film() {
                    ui.label(
                        egui::RichText::new("Film scanned directly (no print paper).")
                            .italics()
                            .small(),
                    );
                }
                let paper_changed = ui
                    .add_enabled_ui(!self.params.scans_film(), |ui| {
                        profile_combo(
                            ui,
                            "paper",
                            "Print paper",
                            &self.papers,
                            &mut self.print_name,
                        )
                    })
                    .inner;
                if paper_changed {
                    self.print_dev_times = profile_dev_times(&self.data_dir, &self.print_name);
                    self.params.print_render.chemistry.development_time = None;
                    self.dirty = true;
                }
                if !self.params.scans_film()
                    && self.print_dev_times.len() > 1
                    && dev_time_combo(
                        ui,
                        "print_dev_time",
                        "Print development time",
                        &self.print_dev_times.clone(),
                        &mut self.params.print_render.chemistry.development_time,
                    )
                {
                    self.dirty = true;
                }
            })
            .header_response
            .on_hover_text(tooltips::SECTION_PROFILES);

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

        // ── Exposure ────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Exposure")
            .default_open(true)
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
                        egui::ComboBox::from_id_salt("Metering").selected_text(method.clone()),
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
                                    .selectable_value(method, opt.to_string(), opt)
                                    .on_hover_text(tooltips::option(opt))
                                    .changed();
                            }
                        },
                    );
                }
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut self.params.camera.exposure_compensation_ev, -5.0..=5.0)
                        .step_by(0.1),
                    "EV compensation",
                    tooltips::CONTROL_EV_COMPENSATION,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut self.params.camera.film_format_mm, 4.0..=120.0)
                        .logarithmic(true),
                    "Film format (mm)",
                    tooltips::CONTROL_FILM_FORMAT_MM,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut self.params.camera.lens_blur_um, 0.0..=100.0),
                    "Lens blur (µm)",
                    tooltips::CONTROL_LENS_BLUR_UM,
                )
                .changed();
                if changed {
                    self.dirty = true;
                }
            })
            .header_response
            .on_hover_text(tooltips::SECTION_EXPOSURE);

        // ── Halation ────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Halation")
            .default_open(true)
            .show(ui, |ui| {
                let h = &mut self.params.film_render.halation;
                let mut changed = false;
                changed |=
                    tooltips::checkbox(ui, &mut h.active, "Active", tooltips::ENABLE_HALATION)
                        .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut h.halation_amount, 0.0..=3.0),
                    "Halation amount",
                    tooltips::CONTROL_HALATION_AMOUNT,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut h.halation_spatial_scale, 0.1..=5.0),
                    "Halation scale",
                    tooltips::CONTROL_HALATION_SCALE,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut h.scatter_amount, 0.0..=3.0),
                    "Scatter amount",
                    tooltips::CONTROL_SCATTER_AMOUNT,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut h.scatter_spatial_scale, 0.1..=5.0),
                    "Scatter scale",
                    tooltips::CONTROL_SCATTER_SCALE,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut h.halation_n_bounces, 1..=5),
                    "Bounces",
                    tooltips::CONTROL_BOUNCES,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut h.halation_bounce_decay, 0.0..=1.0),
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
                egui::CollapsingHeader::new("Physical emulsion parameters")
                    .show(ui, |ui| {
                        for c in 0..3 {
                            let channel = ["Red", "Green", "Blue"][c];
                            ui.label(channel).on_hover_text(tooltips::EMULSION_CHANNEL);
                            changed |= tooltips::labeled(
                                ui,
                                egui::DragValue::new(&mut h.scatter_core_um[c])
                                    .range(0.0..=1000.0)
                                    .speed(0.1),
                                "Scatter core µm: ",
                                tooltips::CONTROL_SCATTER_CORE_UM,
                            )
                            .changed();
                            changed |= tooltips::labeled(
                                ui,
                                egui::DragValue::new(&mut h.scatter_tail_um[c])
                                    .range(0.0..=2000.0)
                                    .speed(0.5),
                                "Scatter tail µm: ",
                                tooltips::CONTROL_SCATTER_TAIL_UM,
                            )
                            .changed();
                            changed |= tooltips::labeled(
                                ui,
                                egui::DragValue::new(&mut h.scatter_tail_weight[c])
                                    .range(0.0..=1.0)
                                    .speed(0.01),
                                "Tail weight: ",
                                tooltips::CONTROL_TAIL_WEIGHT,
                            )
                            .changed();
                            changed |= tooltips::labeled(
                                ui,
                                egui::DragValue::new(&mut h.halation_strength[c])
                                    .range(0.0..=3.0)
                                    .speed(0.001),
                                "Halation strength: ",
                                tooltips::CONTROL_HALATION_STRENGTH,
                            )
                            .changed();
                            changed |= tooltips::labeled(
                                ui,
                                egui::DragValue::new(&mut h.halation_first_sigma_um[c])
                                    .range(0.0..=2000.0)
                                    .speed(0.5),
                                "First bounce µm: ",
                                tooltips::CONTROL_FIRST_BOUNCE_UM,
                            )
                            .changed();
                        }
                        changed |= tooltips::labeled(
                            ui,
                            egui::DragValue::new(&mut h.boost_ev)
                                .range(0.0..=8.0)
                                .speed(0.01),
                            "Highlight boost EV: ",
                            tooltips::CONTROL_HIGHLIGHT_BOOST_EV,
                        )
                        .changed();
                        changed |= tooltips::labeled(
                            ui,
                            egui::DragValue::new(&mut h.boost_range)
                                .range(0.0..=1.0)
                                .speed(0.01),
                            "Boost ramp (0–1): ",
                            tooltips::CONTROL_BOOST_RAMP_0_1,
                        )
                        .changed();
                        changed |= tooltips::labeled(
                            ui,
                            egui::DragValue::new(&mut h.protect_ev)
                                .range(0.0..=16.0)
                                .speed(0.01),
                            "Protect EV: ",
                            tooltips::CONTROL_PROTECT_EV,
                        )
                        .changed();
                    })
                    .header_response
                    .on_hover_text(tooltips::SECTION_PHYSICAL_EMULSION_PARAMETERS);
                if changed {
                    self.dirty = true;
                }
            })
            .header_response
            .on_hover_text(tooltips::SECTION_HALATION);

        // ── DIR couplers ────────────────────────────────────────────────
        egui::CollapsingHeader::new("DIR couplers")
            .default_open(true)
            .show(ui, |ui| {
                let d = &mut self.params.film_render.dir_couplers;
                let mut changed = false;
                changed |=
                    tooltips::checkbox(ui, &mut d.active, "Active", tooltips::ENABLE_DIR).changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut d.amount, 0.0..=2.0),
                    "Amount",
                    tooltips::CONTROL_AMOUNT,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut d.diffusion_size_um, 0.0..=100.0),
                    "Diffusion size (µm)",
                    tooltips::CONTROL_DIFFUSION_SIZE_UM,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut d.diffusion_tail_um, 0.0..=400.0),
                    "Diffusion tail (µm)",
                    tooltips::CONTROL_DIFFUSION_TAIL_UM,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut d.diffusion_tail_weight, 0.0..=1.0),
                    "Tail weight",
                    tooltips::CONTROL_TAIL_WEIGHT_2,
                )
                .changed();
                for c in 0..3 {
                    changed|=tooltips::labeled(ui,egui::Slider::new(&mut d.langmuir_donor_k_rgb[c],0.01..=20.).logarithmic(true),format!("Donor knee {}",["R","G","B"][c]),"Negative-film inhibitor release knee as a fraction of maximum density; larger values approach a linear donor.").changed();
                    changed|=tooltips::labeled(ui,egui::Slider::new(&mut d.langmuir_receiver_k_rgb[c],0.01..=20.).logarithmic(true),format!("Receiver knee {}",["R","G","B"][c]),"Reversal-film receiver saturation after diffusion, relative to its inhibitor ceiling.").changed();
                }
                egui::CollapsingHeader::new("Inhibition matrix")
                    .show(ui, |ui| {
                        changed |= tooltips::labeled(
                            ui,
                            egui::DragValue::new(&mut d.inhibition_samelayer)
                                .range(0.0..=4.0)
                                .speed(0.01),
                            "Same-layer scale: ",
                            tooltips::CONTROL_SAME_LAYER_SCALE,
                        )
                        .changed();
                        changed |= tooltips::labeled(
                            ui,
                            egui::DragValue::new(&mut d.inhibition_interlayer)
                                .range(0.0..=4.0)
                                .speed(0.01),
                            "Inter-layer scale: ",
                            tooltips::CONTROL_INTER_LAYER_SCALE,
                        )
                        .changed();
                        for c in 0..3 {
                            changed |= tooltips::labeled(
                                ui,
                                egui::DragValue::new(&mut d.gamma_samelayer_rgb[c])
                                    .range(0.0..=4.0)
                                    .speed(0.001),
                                format!("{} self: ", ["Red", "Green", "Blue"][c]),
                                tooltips::DIR_SELF_INHIBITION,
                            )
                            .changed();
                        }
                        for (name, values) in [
                            ("Red → green / blue", &mut d.gamma_interlayer_r_to_gb),
                            ("Green → red / blue", &mut d.gamma_interlayer_g_to_rb),
                            ("Blue → red / green", &mut d.gamma_interlayer_b_to_rg),
                        ] {
                            ui.label(name).on_hover_text(tooltips::DIR_DONOR);
                            ui.horizontal(|ui| {
                                for (receiver, v) in values.iter_mut().enumerate() {
                                    changed |= tooltips::labeled(
                                        ui,
                                        egui::DragValue::new(v).range(0.0..=4.0).speed(0.001),
                                        tooltips::inhibition_receiver(name, receiver),
                                        tooltips::inhibition(name, receiver),
                                    )
                                    .changed();
                                }
                            });
                        }
                    })
                    .header_response
                    .on_hover_text(tooltips::SECTION_INHIBITION_MATRIX);
                if changed {
                    self.dirty = true;
                }
            })
            .header_response
            .on_hover_text(tooltips::SECTION_DIR_COUPLERS);

        // ── Diffusion filter (lens) ─────────────────────────────────────
        egui::CollapsingHeader::new("Diffusion filter (lens)")
            .default_open(true)
            .show(ui, |ui| {
                let df = &mut self.params.camera.diffusion_filter;
                let mut changed = false;
                changed |=
                    tooltips::checkbox(ui, &mut df.active, "Active", tooltips::ENABLE_DIFFUSION)
                        .changed();
                tooltips::combo(
                    ui,
                    egui::ComboBox::from_id_salt("Family").selected_text(df.filter_family.clone()),
                    "Family",
                    tooltips::MENU_FAMILY,
                    |ui| {
                        for fam in ["black_pro_mist", "glimmerglass", "pro_mist", "cinebloom"] {
                            changed |= ui
                                .selectable_value(&mut df.filter_family, fam.to_string(), fam)
                                .on_hover_text(tooltips::option(fam))
                                .changed();
                        }
                    },
                );
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut df.strength, 0.0..=2.0),
                    "Strength",
                    tooltips::CONTROL_STRENGTH,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut df.spatial_scale, 0.1..=3.0),
                    "Spatial scale",
                    tooltips::CONTROL_SPATIAL_SCALE,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut df.halo_warmth, -1.5..=1.5),
                    "Halo warmth",
                    tooltips::CONTROL_HALO_WARMTH,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut df.core_intensity, 0.0..=2.0),
                    "Core intensity",
                    tooltips::CONTROL_CORE_INTENSITY,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut df.halo_intensity, 0.0..=2.0),
                    "Halo intensity",
                    tooltips::CONTROL_HALO_INTENSITY,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut df.bloom_intensity, 0.0..=2.0),
                    "Bloom intensity",
                    tooltips::CONTROL_BLOOM_INTENSITY,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut df.halo_size, 0.1..=3.0),
                    "Halo size",
                    tooltips::CONTROL_HALO_SIZE,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut df.bloom_size, 0.1..=3.0),
                    "Bloom size",
                    tooltips::CONTROL_BLOOM_SIZE,
                )
                .changed();
                if changed {
                    self.dirty = true;
                }
            })
            .header_response
            .on_hover_text(tooltips::SECTION_DIFFUSION_FILTER_LENS);

        egui::CollapsingHeader::new("Taking filter").show(ui, |ui| {
            let mut changed = false;
            tooltips::combo(
                ui,
                egui::ComboBox::from_id_salt("taking_filter")
                    .selected_text(&self.params.camera.color_filter),
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
                                name,
                            )
                            .changed();
                    }
                },
            );
            if changed {
                self.dirty = true;
            }
        });

        // ── Grain ───────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Grain")
            .default_open(true)
            .show(ui, |ui| {
                let g = &mut self.params.film_render.grain;
                let mut changed = false;
                changed |= tooltips::checkbox(ui, &mut g.active, "Active", tooltips::ENABLE_GRAIN)
                    .changed();
                for c in 0..3 {
                    changed |= tooltips::labeled(ui,egui::Slider::new(&mut g.rms_granularity[c],0.1..=60.),format!("RMS granularity {}",["R","G","B"][c]),"Peak density noise in thousandths, measured with a 48 µm aperture; higher values give coarser grain.").changed();
                }
                changed |= tooltips::labeled(ui,egui::Slider::new(&mut g.mult_usm_sigma,0. ..=3.),"Grain sharpening σ","Blur radius in pixels used by the multiplicative grain sharpen.").changed();
                changed |= tooltips::labeled(ui,egui::Slider::new(&mut g.mult_usm_amount,0. ..=3.),"Grain sharpening amount","Sharpen absolute grain density while preserving each channel's mean density.").changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut g.blur, 0.0..=3.0),
                    "Post-blur σ",
                    tooltips::CONTROL_POST_BLUR_SIGMA,
                )
                .changed();
                if changed {
                    self.dirty = true;
                }
            })
            .header_response
            .on_hover_text(tooltips::SECTION_GRAIN);

        // ── Glare ───────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Glare")
            .default_open(false)
            .show(ui, |ui| {
                let g = &mut self.params.print_render.glare;
                let mut changed = false;
                changed |= tooltips::checkbox(ui, &mut g.active, "Active", tooltips::ENABLE_GLARE)
                    .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut g.percent, 0.0..=0.2),
                    "Percent",
                    tooltips::CONTROL_PERCENT,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut g.roughness, 0.0..=2.0),
                    "Roughness",
                    tooltips::CONTROL_ROUGHNESS,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut g.blur, 0.0..=5.0),
                    "Blur σ (px)",
                    tooltips::CONTROL_BLUR_SIGMA_PX,
                )
                .changed();
                if changed {
                    self.dirty = true;
                }
            })
            .header_response
            .on_hover_text(tooltips::SECTION_GLARE);

        // ── Print curves (s023 morph) ───────────────────────────────────
        for (title, m) in [
            ("Film chemistry", &mut self.params.film_render.chemistry),
            ("Print chemistry", &mut self.params.print_render.chemistry),
        ] {
            egui::CollapsingHeader::new(title)
                .default_open(false)
                .show(ui, |ui| {
                    let mut changed = false;
                    changed |= tooltips::checkbox(
                        ui,
                        &mut m.active,
                        "Active chemistry",
                        tooltips::MORPH_PRINT_CURVES,
                    )
                    .changed();
                    if m.active {
                        changed |= tooltips::labeled(
                            ui,
                            egui::Slider::new(&mut m.gamma_factor, 0.5..=2.0),
                            "Gamma",
                            tooltips::CONTROL_GAMMA,
                        )
                        .changed();
                        changed |= tooltips::labeled(
                            ui,
                            egui::Slider::new(&mut m.gamma_factor_fast, 0.5..=2.0),
                            "Gamma fast",
                            tooltips::CONTROL_GAMMA_FAST,
                        )
                        .changed();
                        changed |= tooltips::labeled(
                            ui,
                            egui::Slider::new(&mut m.gamma_factor_slow, 0.5..=2.0),
                            "Gamma slow",
                            tooltips::CONTROL_GAMMA_SLOW,
                        )
                        .changed();
                        changed |= tooltips::labeled(
                            ui,
                            egui::Slider::new(&mut m.gamma_factor_red, 0.5..=2.0),
                            "Gamma R",
                            tooltips::CONTROL_GAMMA_R,
                        )
                        .changed();
                        changed |= tooltips::labeled(
                            ui,
                            egui::Slider::new(&mut m.gamma_factor_green, 0.5..=2.0),
                            "Gamma G",
                            tooltips::CONTROL_GAMMA_G,
                        )
                        .changed();
                        changed |= tooltips::labeled(
                            ui,
                            egui::Slider::new(&mut m.gamma_factor_blue, 0.5..=2.0),
                            "Gamma B",
                            tooltips::CONTROL_GAMMA_B,
                        )
                        .changed();
                        changed |= tooltips::labeled(
                            ui,
                            egui::Slider::new(&mut m.developer_exhaustion, 0.0..=1.0),
                            "Developer exhaustion",
                            tooltips::CONTROL_DEVELOPER_EXHAUSTION,
                        )
                        .changed();
                    }
                    if changed {
                        self.dirty = true;
                    }
                })
                .header_response
                .on_hover_text(
                    "Morph the fitted development curves and physical emulsion layers together.",
                );
        }

        for (title, base, film) in [
            ("Film base", &mut self.params.film_render.base, true),
            ("Print base", &mut self.params.print_render.base, false),
        ] {
            egui::CollapsingHeader::new(title).show(ui, |ui| {
                let mut changed = tooltips::checkbox(
                    ui,
                    &mut base.active,
                    "Active base tuning",
                    "Enable changes to the stock's base and fog density.",
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut base.scale, 0. ..=3.),
                    "Base scale",
                    "Scale the stock's spectral base density.",
                )
                .changed();
                for (name, value) in [
                    ("Cyan", &mut base.cyan),
                    ("Magenta", &mut base.magenta),
                    ("Yellow", &mut base.yellow),
                ] {
                    changed |= tooltips::labeled(
                        ui,
                        egui::Slider::new(value, 0. ..=3.),
                        name,
                        "Scale this color component of the spectral base density.",
                    )
                    .changed();
                }
                if film {
                    changed |= tooltips::labeled(
                        ui,
                        egui::Slider::new(&mut base.tilt, -1. ..=1.),
                        "Base tilt",
                        "Tilt the film base density across wavelength around 555 nm.",
                    )
                    .changed();
                }
                if changed {
                    self.dirty = true;
                }
            });
        }
        egui::CollapsingHeader::new("Scan conversion").show(ui,|ui| {
            let p=&mut self.params.film_render.convert;let mut changed=false;
            tooltips::combo(ui,egui::ComboBox::from_id_salt("scan_illuminant").selected_text(&p.scan_illuminant),"Scan illuminant","The physical light used to capture the scanned negative.",|ui| {
                for name in spektrafilm_math::illuminants::names() {changed|=ui.selectable_value(&mut p.scan_illuminant,name.into(),name).changed();}
            });
            changed|=tooltips::labeled(ui,egui::Slider::new(&mut p.exposure_compensation_ev,-8. ..=8.),"Scan exposure EV","Align the scanned input's level to the physical scan model before inversion.").changed();
            changed|=tooltips::labeled(ui,egui::Slider::new(&mut p.base_percentile,90. ..=100.),"Base percentile","Use the brightest pixels above this luminance percentile to estimate clear film base.").changed();
            changed|=tooltips::labeled(ui,egui::TextEdit::singleline(&mut p.calibration).desired_width(170.),"Device matrix","Nine row-major numbers correcting the scanner or camera color response before inversion.").changed();
            changed|=tooltips::checkbox(ui,&mut self.params.settings.use_convert_lut,"Conversion LUT","Cache the inverse scan model in a 3D tetrahedral LUT for faster repeated previews.").changed();
            if self.params.settings.use_convert_lut {changed|=tooltips::labeled(ui,egui::Slider::new(&mut self.params.settings.convert_lut_resolution,9..=65),"Conversion LUT resolution","Samples per RGB axis of the cached inverse scan model.").changed();}
            if changed {self.dirty=true;}
            self.calibration_controls(ui);
        });

        // ── Scanner ─────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Scanner")
            .default_open(false)
            .show(ui, |ui| {
                let s = &mut self.params.scanner;
                let mut changed = false;
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut s.lens_blur, 0.0..=5.0),
                    "Lens blur σ (px)",
                    tooltips::CONTROL_LENS_BLUR_SIGMA_PX,
                )
                .changed();
                let [mut sigma, mut amount] = s.unsharp_mask;
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut sigma, 0.0..=3.0),
                    "Unsharp σ (px)",
                    tooltips::CONTROL_UNSHARP_SIGMA_PX,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut amount, 0.0..=2.0),
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
                        egui::Slider::new(&mut s.white_level, 0.5..=1.0),
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
                        egui::Slider::new(&mut s.black_level, 0.0..=0.5),
                        "Black level",
                        tooltips::CONTROL_BLACK_LEVEL,
                    )
                    .changed();
                }
                if changed {
                    self.dirty = true;
                }
            })
            .header_response
            .on_hover_text(tooltips::SECTION_SCANNER);

        // ── Color management ────────────────────────────────────────────
        egui::CollapsingHeader::new("Color management")
            .default_open(false)
            .show(ui, |ui| {
                let algo = &mut self.params.io.output_gamut_compress.algorithm;
                let mut changed = false;
                // Only the ported modes are offered; cam16ucs is upstream's
                // default, oklch the lighter perceptual option. Enabling either
                // drops the preview off the GPU-resident fast path.
                tooltips::combo(
                    ui,
                    egui::ComboBox::from_id_salt("Gamut compression").selected_text(algo.clone()),
                    "Gamut compression",
                    tooltips::CONTROL_OUTPUT_GAMUT_COMPRESSION,
                    |ui| {
                        for opt in ["off", "oklch", "oklrab", "cam16ucs", "aces_rgc"] {
                            changed |= ui
                                .selectable_value(algo, opt.to_string(), opt)
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
                        .selected_text(in_algo.clone()),
                    "Input gamut compression",
                    tooltips::CONTROL_INPUT_GAMUT_COMPRESSION,
                    |ui| {
                        for opt in ["off", "xy"] {
                            changed |= ui
                                .selectable_value(in_algo, opt.to_string(), opt)
                                .on_hover_text(tooltips::option(opt))
                                .changed();
                        }
                    },
                );
                tooltips::combo(ui,egui::ComboBox::from_id_salt("input_boundary").selected_text(&self.params.io.input_gamut_compress.boundary),"Input boundary","Compress toward the spectral locus or its smooth inscribed hull.",|ui| {
                    for name in ["inscribed_hull","locus"] {changed|=ui.selectable_value(&mut self.params.io.input_gamut_compress.boundary,name.into(),name).changed();}
                });
                changed|=tooltips::labeled(ui,egui::Slider::new(&mut self.params.io.input_gamut_compress.hull_detail,1. ..=30.),"Hull detail","Number of Fourier modes retained in the inscribed gamut boundary; higher values follow the locus more closely.").changed();
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
                        .selected_text(method.clone()),
                    "RGB→raw upsampling",
                    tooltips::MENU_RGB_RAW_UPSAMPLING,
                    |ui| {
                        for opt in ["hanatos2025", "arctic2026beta04", "mallett2019", "jakob2019", "otsu2018", "gauss-lasers", "arctic2026alpha02"] {
                            changed |= ui
                                .selectable_value(method, opt.to_string(), opt)
                                .on_hover_text(tooltips::option(opt))
                                .changed();
                        }
                    },
                );
                if changed {
                    self.dirty = true;
                }
            })
            .header_response
            .on_hover_text(tooltips::SECTION_COLOR_MANAGEMENT);

        // ── Enlarger ────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Enlarger")
            .default_open(false)
            .show(ui, |ui| {
                let e = &mut self.params.enlarger;
                let mut changed = false;
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut e.print_exposure, 0.1..=5.0).logarithmic(true),
                    "Print exposure",
                    tooltips::CONTROL_PRINT_EXPOSURE,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut e.m_filter_shift, -50.0..=50.0),
                    "Magenta filter shift",
                    tooltips::CONTROL_MAGENTA_FILTER_SHIFT,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut e.y_filter_shift, -50.0..=50.0),
                    "Yellow filter shift",
                    tooltips::CONTROL_YELLOW_FILTER_SHIFT,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut e.preflash_exposure, 0.0..=0.5),
                    "Preflash exposure",
                    tooltips::PREFLASH,
                )
                .changed();
                if e.preflash_exposure > 0.0 {
                    changed |= tooltips::labeled(
                        ui,
                        egui::Slider::new(&mut e.preflash_m_filter_shift, -50.0..=50.0),
                        "Preflash magenta shift",
                        tooltips::CONTROL_PREFLASH_MAGENTA_SHIFT,
                    )
                    .changed();
                    changed |= tooltips::labeled(
                        ui,
                        egui::Slider::new(&mut e.preflash_y_filter_shift, -50.0..=50.0),
                        "Preflash yellow shift",
                        tooltips::CONTROL_PREFLASH_YELLOW_SHIFT,
                    )
                    .changed();
                }
                if changed {
                    self.dirty = true;
                }
            })
            .header_response
            .on_hover_text(tooltips::SECTION_ENLARGER);

        // ── Output ──────────────────────────────────────────────────────
        egui::CollapsingHeader::new("Output")
            .default_open(false)
            .show(ui, |ui| {
                let io = &mut self.params.io;
                let mut changed = false;
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut io.upscale_factor, 0.1..=1.0),
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
            })
            .header_response
            .on_hover_text(tooltips::SECTION_OUTPUT);

        self.parity_controls(ui);
        // ── Metrics ─────────────────────────────────────────────────────
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(6.0);
        ui.monospace(format!(
            "render:        {:>6.1} ms   {}",
            self.last_render_ms,
            fps_label(self.last_render_ms)
        ));
        ui.monospace(format!(
            "pipeline build:{:>6.1} ms",
            self.last_pipeline_build_ms
        ));
        ui.monospace(format!(
            "clone/scale:   {:>6.1} / {:>5.1} ms",
            self.last_input_clone_ms, self.last_scale_ms
        ));
        ui.monospace(format!("preview pack:  {:>6.1} ms", self.last_preview_ms));
        ui.monospace(format!(
            "worker total:  {:>6.1} ms",
            self.last_worker_total_ms
        ));
        ui.monospace(format!("backend: {}", self.backend.name()));
        ui.label(format!("Execution: {}", self.render_path));
        ui.label(egui::RichText::new(&self.status).small());
    }
}

impl eframe::App for App {
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
            && self.load_job.is_none()
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
        if self.render_job.is_some() || self.load_job.is_some() {
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
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if self.load_job.is_some() || self.render_job.is_some() {
                    ui.spinner();
                }
                if self.status.starts_with("Load error:")
                    || self.status.starts_with("Render error:")
                {
                    ui.colored_label(ui.visuals().error_fg_color, &self.status);
                } else {
                    ui.label(&self.status);
                }
            });
        });
        egui::SidePanel::right("controls")
            .resizable(false)
            .exact_width(340.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.controls_panel(ui, ctx));
            });
        let action = self.catalog.as_mut().and_then(|catalog| {
            egui::SidePanel::left("catalog")
                .default_width(270.)
                .min_width(220.)
                .max_width(420.)
                .show(ctx, |ui| catalog.show(ui, ctx))
                .inner
        });
        match action {
            Some(catalog_ui::Action::Close) => self.catalog = None,
            Some(catalog_ui::Action::Select(photo)) => self.select_catalog_photo(photo),
            Some(catalog_ui::Action::Empty) => {
                if let Some(job) = &mut self.load_job {
                    job.discard = true;
                }
                if let Some(job) = &mut self.render_job {
                    job.discard = true;
                }
                self.pending_load = None;
                self.clear_image();
                self.status = "Catalog loaded. No available photos match the filter.".into();
            }
            None => {}
        }
        let canvas = if self.display.gray_18_canvas {
            egui::Color32::from_gray(118)
        } else {
            ctx.style().visuals.panel_fill
        };
        egui::CentralPanel::default()
            .frame(egui::Frame::central_panel(&ctx.style()).fill(canvas))
            .show(ctx, |ui| {
                if self.canvas_input.is_some()
                    || self.input_tex.is_some()
                    || self.canvas_output.is_some()
                    || self.output_tex.is_some()
                {
                    self.draw_preview_with_zoom(ui);
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            egui::RichText::new(if self.image.is_some() {
                                "Click Render preview to process this image."
                            } else {
                                "Drop or open an image to start."
                            })
                            .size(18.0),
                        );
                    });
                }
            });

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
    }

    /// Called once when the window is closing. If an export is still
    /// in-flight, set the cancel flag and join the worker thread so
    /// the child process is SIGKILLed before the GUI exits. Without
    /// this the f64 CLI orphans into its own process group and keeps
    /// hammering the CPU long after the window is gone.
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let Some(job) = self.export_job.take() else {
            return;
        };
        job.cancel.store(true, Ordering::SeqCst);
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
        let (display, is_paper) = match profile::load_profile_by_name(data_dir, &stock) {
            Ok(p) => {
                let display = p.info.name.clone().unwrap_or_else(|| stock.clone());
                let is_paper = p.info.support == "paper" || p.info.stage == "printing";
                (display, is_paper)
            }
            Err(_) => (stock.clone(), false),
        };
        let entry = ProfileEntry { stock, display };
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

/// Combo box that picks one of `entries` by its `stock` id (the
/// underlying file stem) while showing `display` (the human-readable
/// name) as the label. Falls back to showing the raw stock id if no
/// entry with the current `selected_stock` exists.
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
    let hint = tooltips::DEVELOPMENT_TIME;
    ui.label(label)
        .on_hover_text(hint)
        .on_disabled_hover_text(hint);
    egui::ComboBox::from_id_salt(salt)
        .selected_text(format!("{} min", times[current_idx]))
        .width(ui.available_width().min(280.0))
        .show_ui(ui, |ui| {
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
        });
    changed
}

fn profile_combo(
    ui: &mut egui::Ui,
    salt: &str,
    label: &str,
    entries: &[ProfileEntry],
    selected_stock: &mut String,
) -> bool {
    let hint = if salt == "film" {
        tooltips::FILM_PROFILE
    } else {
        tooltips::PAPER_PROFILE
    };
    ui.label(label)
        .on_hover_text(hint)
        .on_disabled_hover_text(hint);
    let display = entries
        .iter()
        .find(|e| &e.stock == selected_stock)
        .map(|e| e.display.clone())
        .unwrap_or_else(|| selected_stock.clone());
    let prev = selected_stock.clone();
    egui::ComboBox::from_id_salt(salt)
        .selected_text(&display)
        .width(ui.available_width().min(280.0))
        .show_ui(ui, |ui| {
            for entry in entries {
                ui.selectable_value(selected_stock, entry.stock.clone(), &entry.display)
                    .on_hover_text(hint)
                    .on_disabled_hover_text(hint);
            }
        });
    prev != *selected_stock
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

fn preview_pipeline_cache_key(film_name: &str, print_name: &str, params: &RuntimeParams) -> String {
    serde_json::json!({
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
    .to_string()
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
    if let Some(geometry) = &raw_options.lightroom_geometry {
        cmd.arg("--lightroom-geometry")
            .arg(serde_json::to_string(geometry)?);
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
    if cancel.load(Ordering::SeqCst) {
        anyhow::bail!("cancelled");
    }
    let output_image = pipeline
        .process_resident_borrowed(image, backend)
        .unwrap_or_else(|| pipeline.process(image.clone(), backend));
    if cancel.load(Ordering::SeqCst) {
        anyhow::bail!("cancelled");
    }
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
            output: self.current_save_options(),
        }
    }
    fn restore_project(&mut self, path: &Path, load_image: bool) -> Result<()> {
        let state = project::Project::load(path)?;
        profile::load_profile_by_name(&self.data_dir, &state.film)?;
        profile::load_profile_by_name(&self.data_dir, &state.paper)?;
        self.auto_preview = state.auto_preview;
        self.gpu_export = state.gpu_export;
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
    fn project_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Save project…")
                .on_hover_text(tooltips::CONTROL_SAVE_PROJECT)
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Project", &["json"])
                    .set_file_name("spektrafilm.json")
                    .save_file()
                {
                    self.status = match self.project_state().save(&path) {
                        Ok(()) => format!("Saved project {}", path.display()),
                        Err(e) => format!("Save project: {e:#}"),
                    };
                }
            }
            if ui
                .button("Open project…")
                .on_hover_text(tooltips::CONTROL_OPEN_PROJECT)
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Project", &["json"])
                    .pick_file()
                {
                    self.status = match self.restore_project(&path, true) {
                        Ok(()) => format!("Opened project {}", path.display()),
                        Err(e) => format!("Open project: {e:#}"),
                    };
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("Save defaults")
                .on_hover_text(tooltips::CONTROL_SAVE_DEFAULTS)
                .clicked()
            {
                let mut state = self.project_state();
                state.image = None;
                state.raw = self.raw_options.clone();
                state.raw.lightroom_geometry = None;
                self.status = match state.save(&project::default_path()) {
                    Ok(()) => "Saved startup defaults".into(),
                    Err(e) => format!("Save defaults: {e:#}"),
                };
            }
            if ui
                .button("Factory defaults")
                .on_hover_text(tooltips::CONTROL_FACTORY_DEFAULTS)
                .clicked()
            {
                let input_color = self.params.io.input_color_space.clone();
                let input_encoded = self.params.io.input_cctf_decoding;
                self.params = RuntimeParams::default();
                if self.image.is_some() {
                    self.params.io.input_color_space = input_color;
                    self.params.io.input_cctf_decoding = input_encoded;
                }
                self.auto_preview = true;
                self.params.settings.preview_max_size = 2048;
                self.gpu_export = true;
                self.lut = Default::default();
                self.display = Default::default();
                self.scan_for_print_restore = None;
                if let Ok(film) = profile::load_profile_by_name(&self.data_dir, &self.film_name) {
                    spektrafilm_core::pipeline::apply_stock_defaults(&film, &mut self.params);
                }
                self.raw_options = Default::default();
                self.save_options = Default::default();
                self.pipeline_cache = None;
                self.pipeline_cache_key = None;
                self.dirty = true;
                self.pending_dirty = self.render_job.is_some();
                match std::fs::remove_file(project::default_path()) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => {
                        self.status = format!("Reset defaults: {e}");
                    }
                }
            }
        });
    }
    fn parity_controls(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Color and processing")
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
                ui.label("Input primaries")
                    .on_hover_text(tooltips::CONTROL_INPUT_PRIMARIES);
                egui::ComboBox::from_id_salt("input_primaries")
                    .selected_text(&self.params.io.input_color_space)
                    .show_ui(ui, |ui| {
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
                    });
                changed |= tooltips::checkbox(
                    ui,
                    &mut self.params.io.input_cctf_decoding,
                    "Decode input transfer function",
                    tooltips::CONTROL_DECODE_INPUT_TRANSFER_FUNCTION,
                )
                .changed();
                ui.label("Output primaries")
                    .on_hover_text(tooltips::CONTROL_OUTPUT_PRIMARIES);
                egui::ComboBox::from_id_salt("output_primaries")
                    .selected_text(&self.params.io.output_color_space)
                    .show_ui(ui, |ui| {
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
                    });
                for (label, order) in [
                    ("Film dye order", &mut self.params.film_render.channel_order),
                    (
                        "Print dye order",
                        &mut self.params.print_render.channel_order,
                    ),
                ] {
                    ui.label(label).on_hover_text(tooltips::DYE_ORDER);
                    egui::ComboBox::from_id_salt(label)
                        .selected_text(format!("{order:?}"))
                        .show_ui(ui, |ui| {
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
                        });
                }
                ui.label("Enlarger illuminant")
                    .on_hover_text(tooltips::MENU_ENLARGER_ILLUMINANT);
                egui::ComboBox::from_id_salt("enlarger_illuminant")
                    .selected_text(&self.params.enlarger.illuminant)
                    .show_ui(ui, |ui| {
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
                    });
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
                    let db =
                        spektrafilm_core::neutral_filters::NeutralFilters::load(&self.data_dir);
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
                    for (label, value) in [
                        ("Neutral cyan", &mut self.params.enlarger.c_filter_neutral),
                        (
                            "Neutral magenta",
                            &mut self.params.enlarger.m_filter_neutral,
                        ),
                        ("Neutral yellow", &mut self.params.enlarger.y_filter_neutral),
                    ] {
                        changed |= tooltips::labeled(
                            ui,
                            egui::DragValue::new(value).speed(0.1).range(0.0..=200.0),
                            format!("{label}: "),
                            tooltips::MANUAL_NEUTRAL_FILTER,
                        )
                        .changed();
                    }
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
                        egui::Slider::new(&mut self.params.settings.lut_resolution, 3..=65),
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
                ui.label("Output gamut compression")
                    .on_hover_text(tooltips::CONTROL_OUTPUT_GAMUT_COMPRESSION);
                egui::ComboBox::from_id_salt("gamut_algorithm")
                    .selected_text(&self.params.io.output_gamut_compress.algorithm)
                    .show_ui(ui, |ui| {
                        for name in ["off", "cam16ucs", "oklch", "oklrab", "jzazbz", "aces_rgc"] {
                            changed |= ui
                                .selectable_value(
                                    &mut self.params.io.output_gamut_compress.algorithm,
                                    name.into(),
                                    name,
                                )
                                .on_hover_text(tooltips::option(name))
                                .changed();
                        }
                    });
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
                    for c in 0..2 {
                        changed |= tooltips::labeled(
                            ui,
                            egui::Slider::new(&mut self.params.io.crop_center[c], 0.0..=1.0),
                            if c == 0 {
                                "Crop center X"
                            } else {
                                "Crop center Y"
                            },
                            tooltips::CROP_CENTER,
                        )
                        .changed();
                        changed |= tooltips::labeled(
                            ui,
                            egui::Slider::new(&mut self.params.io.crop_size[c], 0.01..=1.0),
                            if c == 0 { "Crop width" } else { "Crop height" },
                            tooltips::CROP_SIZE,
                        )
                        .changed();
                    }
                }
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(
                        &mut self.params.film_render.grain.blur_dye_clouds_um,
                        0.0..=5.0,
                    ),
                    "Dye cloud blur (µm)",
                    tooltips::CONTROL_DYE_CLOUD_BLUR_UM,
                )
                .changed();
                for i in 0..3 {
                    changed |= tooltips::labeled(
                        ui,
                        egui::Slider::new(
                            &mut self.params.film_render.grain.particle_scale_sublayers[i],
                            0.1..=5.0,
                        ),
                        format!("Layer {} particle scale", i + 1),
                        tooltips::LAYER_PARTICLE_SCALE,
                    )
                    .changed();
                }
                changed |= tooltips::checkbox(
                    ui,
                    &mut self.params.settings.apply_hanatos2025_adaptation_surface,
                    "Sensitivity adaptation surface",
                    tooltips::CONTROL_SENSITIVITY_ADAPTATION_SURFACE,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut self.params.settings.spectral_gaussian_blur, 0.0..=10.0),
                    "Spectral blur",
                    tooltips::CONTROL_SPECTRAL_BLUR,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut self.params.camera.filter_uv[0], 0.0..=1.0),
                    "UV filter amount",
                    tooltips::CONTROL_UV_FILTER_AMOUNT,
                )
                .changed();
                changed |= tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut self.params.camera.filter_ir[0], 0.0..=1.0),
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
            })
            .header_response
            .on_hover_text(tooltips::SECTION_COLOR_AND_PROCESSING);
        if tooltips::labeled(
            ui,
            egui::Slider::new(&mut self.params.settings.preview_max_size, 128..=8192).step_by(128.),
            "Preview long edge (px)",
            tooltips::CONTROL_PREVIEW_LONG_EDGE_PX,
        )
        .changed()
        {
            self.dirty = true;
        }
        egui::CollapsingHeader::new("RAW development")
            .show(ui, |ui| {
                ui.label("RAW decoder")
                    .on_hover_text(tooltips::MENU_RAW_DECODER);
                egui::ComboBox::from_id_salt("raw_decoder")
                    .selected_text(&self.raw_options.decoder)
                    .show_ui(ui, |ui| {
                        for name in ["auto", "libraw", "rawler"] {
                            ui.selectable_value(&mut self.raw_options.decoder, name.into(), name)
                                .on_hover_text(tooltips::option(name));
                        }
                    });
                ui.label("White balance")
                    .on_hover_text(tooltips::MENU_RAW_WHITE_BALANCE);
                egui::ComboBox::from_id_salt("raw_white_balance")
                    .selected_text(&self.raw_options.white_balance)
                    .show_ui(ui, |ui| {
                        for name in ["as_shot", "daylight", "tungsten", "custom"] {
                            ui.selectable_value(
                                &mut self.raw_options.white_balance,
                                name.into(),
                                name,
                            )
                            .on_hover_text(tooltips::option(name));
                        }
                    });
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
                        ui.label(format!("Database override: {}", path.display()))
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
                        ui.label(self.lens_database_status.label)
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
                if ui
                    .button("Reload RAW with these settings")
                    .on_hover_text(tooltips::CONTROL_RELOAD_RAW_WITH_THESE_SETTINGS)
                    .clicked()
                {
                    if let Some(path) = self.image_path.clone() {
                        let rotation = self.rotation;
                        self.load_image_with_state(&path, rotation, None, None);
                    }
                }
            })
            .header_response
            .on_hover_text(tooltips::SECTION_RAW_DEVELOPMENT);
        egui::CollapsingHeader::new("Display")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(self.image.is_some(), icons::RotateButton::new(false))
                        .on_hover_text(tooltips::CONTROL_ROTATE)
                        .on_disabled_hover_text(tooltips::CONTROL_ROTATE)
                        .clicked()
                    {
                        self.rotate_input(3);
                    }
                    if ui
                        .add_enabled(self.image.is_some(), icons::RotateButton::new(true))
                        .on_hover_text(tooltips::CONTROL_ROTATE_2)
                        .on_disabled_hover_text(tooltips::CONTROL_ROTATE_2)
                        .clicked()
                    {
                        self.rotate_input(1);
                    }
                });
                if tooltips::checkbox(
                    ui,
                    &mut self.display.display_transform,
                    "Display color transform",
                    tooltips::CONTROL_DISPLAY_COLOR_TRANSFORM,
                )
                .changed()
                {
                    self.dirty = true;
                }
                tooltips::checkbox(
                    ui,
                    &mut self.display.show_border,
                    "White border layer",
                    tooltips::CONTROL_WHITE_BORDER_LAYER,
                );
                tooltips::checkbox(
                    ui,
                    &mut self.display.show_paper_back,
                    "Paper-back layer",
                    tooltips::CONTROL_PAPER_BACK_LAYER,
                );
                tooltips::checkbox(
                    ui,
                    &mut self.display.show_input,
                    "Input layer",
                    tooltips::CONTROL_INPUT_LAYER,
                );
                if self.display.show_input {
                    tooltips::labeled(
                        ui,
                        egui::Slider::new(&mut self.display.input_opacity, 0.0..=1.0),
                        "Input opacity",
                        tooltips::CONTROL_INPUT_OPACITY,
                    );
                }
                tooltips::checkbox(
                    ui,
                    &mut self.display.show_output,
                    "Output layer",
                    tooltips::CONTROL_OUTPUT_LAYER,
                );
                if self.display.show_output {
                    tooltips::labeled(
                        ui,
                        egui::Slider::new(&mut self.display.output_opacity, 0.0..=1.0),
                        "Output opacity",
                        tooltips::CONTROL_OUTPUT_OPACITY,
                    );
                }
                tooltips::checkbox(
                    ui,
                    &mut self.display.compare_split,
                    "Compare input / output",
                    tooltips::CONTROL_COMPARE_INPUT_OUTPUT,
                );
                if self.display.compare_split {
                    tooltips::labeled(
                        ui,
                        egui::Slider::new(&mut self.display.split_position, 0.0..=1.0),
                        "Comparison split",
                        tooltips::CONTROL_COMPARISON_SPLIT,
                    );
                }
                tooltips::checkbox(
                    ui,
                    &mut self.display.gray_18_canvas,
                    "18% gray canvas",
                    tooltips::CONTROL_18_GRAY_CANVAS,
                );
                tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut self.display.white_padding, 0.0..=0.2),
                    "White border",
                    tooltips::CONTROL_WHITE_BORDER,
                );
                tooltips::checkbox(
                    ui,
                    &mut self.display.animate,
                    "Paper development / crossfade",
                    tooltips::CONTROL_PAPER_DEVELOPMENT_CROSSFADE,
                );
                if ui
                    .add_enabled(
                        self.canvas_output.is_some(),
                        egui::Button::new("Replay paper development"),
                    )
                    .on_hover_text(tooltips::CONTROL_REPLAY_PAPER_DEVELOPMENT)
                    .on_disabled_hover_text(tooltips::CONTROL_REPLAY_PAPER_DEVELOPMENT)
                    .clicked()
                {
                    self.canvas_previous = None;
                    self.animation = Some((Instant::now(), true));
                    self.ctx.request_repaint();
                }
                ui.label("Preview sampling")
                    .on_hover_text(tooltips::MENU_CANVAS_INTERPOLATION);
                egui::ComboBox::from_id_salt("canvas_interpolation")
                    .selected_text(&self.display.output_interpolation)
                    .show_ui(ui, |ui| {
                        for mode in view::INTERPOLATIONS {
                            if ui
                                .selectable_value(
                                    &mut self.display.output_interpolation,
                                    mode.into(),
                                    mode,
                                )
                                .on_hover_text(tooltips::option(mode))
                                .changed()
                            {
                                if self.canvas.is_none() {
                                    self.dirty = true;
                                }
                            }
                        }
                    });
                if self.canvas.is_none() {
                    ui.label(tooltips::GPU_CANVAS_NOTE);
                }
            })
            .header_response
            .on_hover_text(tooltips::SECTION_DISPLAY);
        egui::CollapsingHeader::new("File export")
            .show(ui, |ui| {
                tooltips::checkbox(
                    ui,
                    &mut self.gpu_export,
                    "Use GPU for full-resolution export",
                    tooltips::EXPORT_PRECISION,
                );
                ui.label("Saving color space")
                    .on_hover_text(tooltips::MENU_SAVING_COLOR_SPACE);
                egui::ComboBox::from_id_salt("saving_color_space")
                    .selected_text(&self.save_options.color_space)
                    .show_ui(ui, |ui| {
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
                    });
                tooltips::checkbox(
                    ui,
                    &mut self.save_options.encoded,
                    "Encode saving transfer function",
                    tooltips::CONTROL_ENCODE_SAVING_TRANSFER_FUNCTION,
                );
                ui.label("TIFF bit depth")
                    .on_hover_text(tooltips::MENU_EXPORT_DEPTH);
                egui::ComboBox::from_id_salt("export_depth")
                    .selected_text(format!("{} bit TIFF", self.save_options.bit_depth))
                    .show_ui(ui, |ui| {
                        for depth in [8, 16, 32] {
                            ui.selectable_value(
                                &mut self.save_options.bit_depth,
                                depth,
                                format!("{depth} bit TIFF"),
                            )
                            .on_hover_text(if depth == 32 {
                                tooltips::TIFF_FLOAT
                            } else {
                                tooltips::TIFF_INTEGER
                            });
                        }
                    });
                tooltips::labeled(
                    ui,
                    egui::Slider::new(&mut self.save_options.jpeg_quality, 1..=100),
                    "JPEG quality",
                    tooltips::CONTROL_JPEG_QUALITY,
                );
                ui.label("Float TIFF and EXR preserve values outside 0–1.");
            })
            .header_response
            .on_hover_text(tooltips::SECTION_FILE_EXPORT);
    }
}

#[cfg(all(test, unix))]
mod export_tests {
    use super::*;
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
            discard: false,
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
        let latest_pixels = app.image.as_ref().unwrap().clone();
        app.select_catalog_photo(first);
        finish(&mut app);
        app.select_catalog_photo(latest.clone());
        finish(&mut app);
        assert!(Arc::ptr_eq(app.image.as_ref().unwrap(), &latest_pixels));
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
            let original_key = preview_pipeline_cache_key(film, paper, &params);
            for (path, value) in &changes {
                let mut edited = serde_json::to_value(&params).unwrap();
                *edited.pointer_mut(path).unwrap() = value.clone();
                let changed: RuntimeParams = serde_json::from_value(edited).unwrap();
                let expected = build(film, paper, changed.clone());
                let actual = if preview_pipeline_cache_key(film, paper, &changed) == original_key {
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
                &Default::default(),
            )
            .unwrap();
            let loaded = spektrafilm_io::load(&path, &Default::default()).unwrap();
            assert_eq!((loaded.pixels.width, loaded.pixels.height), dimensions);
            assert_eq!(loaded.color_space, "Display P3");
            assert!(loaded.encoded);
        }
        std::fs::remove_file(path).unwrap();
    }
}
