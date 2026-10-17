#[cfg(feature = "wgpu-backend")]
#[path = "color_transfer.rs"]
mod color_transfer;
#[cfg(feature = "wgpu-backend")]
#[path = "fft_diffusion.rs"]
mod fft_diffusion;
#[cfg(feature = "wgpu-backend")]
#[path = "layered_grain.rs"]
mod layered_grain;
#[cfg(feature = "wgpu-backend")]
#[path = "pchip.rs"]
mod pchip;
#[cfg(feature = "wgpu-backend")]
#[path = "spatial.rs"]
mod spatial;
/// wgpu compute backend — dispatches WGSL shaders on GPU via Metal/Vulkan/DX12.

#[cfg(feature = "wgpu-backend")]
use std::borrow::Cow;

#[cfg(feature = "wgpu-backend")]
use spektrafilm_math::image::ImageBuf;

#[cfg(feature = "wgpu-backend")]
use crate::{ComputeBackend, Lut3D, cpu_backend};

/// Borrow a `&[Scalar]` as `&[f32]` for GPU upload.
/// Zero-copy in f32 mode; allocates a converted Vec in f64 mode.
#[cfg(all(feature = "wgpu-backend", not(feature = "precision-f64")))]
#[inline]
fn scalars_to_f32(v: &[spektrafilm_math::precision::Scalar]) -> std::borrow::Cow<'_, [f32]> {
    std::borrow::Cow::Borrowed(v)
}

#[cfg(all(feature = "wgpu-backend", feature = "precision-f64"))]
#[inline]
fn scalars_to_f32(v: &[spektrafilm_math::precision::Scalar]) -> std::borrow::Cow<'static, [f32]> {
    std::borrow::Cow::Owned(v.iter().map(|&s| s as f32).collect())
}

/// Convert a `Vec<f32>` from GPU readback into `Vec<Scalar>`.
/// Zero-copy (identity) in f32 mode; allocates in f64 mode.
#[cfg(all(feature = "wgpu-backend", not(feature = "precision-f64")))]
#[inline]
fn f32_to_scalars(v: Vec<f32>) -> Vec<spektrafilm_math::precision::Scalar> {
    v
}

/// Sanitize NaN values in spectral inputs before GPU upload.
///
/// **Why this exists**: profile JSON has NaN values for `channel_density` /
/// `base_density` at UV/IR wavelengths where the dye/base density isn't
/// measured. Python's `density_to_light` zeros NaN values out (correctly
/// excluding those wavelengths from the einsum). WGSL on Metal compiles with
/// fast-math semantics and optimizes away `x != x` NaN checks, so we cannot
/// rely on in-shader NaN handling. We sanitize CPU-side instead:
///
/// - `channel_density` NaN → 0
/// - `base_density` NaN OR any of its row in channel_density is NaN → +1000
///   (so `pow(10, -1000) ≈ 0` zeros the whole wavelength's contribution)
///
/// This preserves Python's wavelength-skipping semantics exactly.
#[cfg(feature = "wgpu-backend")]
fn sanitize_spectral_inputs(
    channel_density: &[[f64; 3]],
    base_density: &[f64],
    n_wl: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut cd = Vec::with_capacity(n_wl * 3);
    let mut bd = Vec::with_capacity(n_wl);
    for wl in 0..n_wl {
        let r = channel_density[wl][0];
        let g = channel_density[wl][1];
        let b = channel_density[wl][2];
        let base = if wl < base_density.len() {
            base_density[wl]
        } else {
            f64::NAN
        };
        let row_has_nan = r.is_nan() || g.is_nan() || b.is_nan() || base.is_nan();
        cd.push(if r.is_nan() { 0.0 } else { r as f32 });
        cd.push(if g.is_nan() { 0.0 } else { g as f32 });
        cd.push(if b.is_nan() { 0.0 } else { b as f32 });
        bd.push(if row_has_nan { 1000.0 } else { base as f32 });
    }
    (cd, bd)
}

#[cfg(all(feature = "wgpu-backend", feature = "precision-f64"))]
#[inline]
fn f32_to_scalars(v: Vec<f32>) -> Vec<spektrafilm_math::precision::Scalar> {
    v.into_iter().map(|x| x as f64).collect()
}

#[cfg(feature = "wgpu-backend")]
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Cache compiled compute pipelines keyed by shader source pointer.
    /// `&'static str` is fine because all our shader sources come from `include_str!`.
    pipeline_cache: std::sync::Mutex<std::collections::HashMap<usize, CachedPipeline>>,
    diffusion_cache: std::sync::Mutex<fft_diffusion::Cache>,
}

#[cfg(feature = "wgpu-backend")]
struct CachedPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

#[cfg(feature = "wgpu-backend")]
impl WgpuBackend {
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        // The default `Limits` cap storage buffer bindings at 128 MB,
        // which a 6-channel ≥ 14 MP image exceeds (image_bytes =
        // width × height × 3 × 4). Bump every relevant limit up to the
        // adapter's hardware ceiling so we can render arbitrary
        // megapixel counts (within RAM).
        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        limits.max_buffer_size = adapter_limits.max_buffer_size;
        limits.max_compute_workgroups_per_dimension =
            adapter_limits.max_compute_workgroups_per_dimension;
        limits.max_bind_groups = adapter_limits.max_bind_groups.max(limits.max_bind_groups);
        // Our per-pixel shaders use `@workgroup_size(1024)` so the
        // dispatch grid stays under the 65535-per-dimension limit even
        // for 30+ MP images. The default Limits cap workgroup
        // invocations at 256, so we have to lift that here too.
        limits.max_compute_invocations_per_workgroup =
            adapter_limits.max_compute_invocations_per_workgroup;
        limits.max_compute_workgroup_size_x = adapter_limits.max_compute_workgroup_size_x;
        limits.max_compute_workgroup_size_y = adapter_limits.max_compute_workgroup_size_y;
        limits.max_compute_workgroup_size_z = adapter_limits.max_compute_workgroup_size_z;
        // MAPPABLE_PRIMARY_BUFFERS lets the input/output STORAGE buffers also
        // be MAP_WRITE / MAP_READ, so they map directly for a zero-copy
        // upload/readback. On unified-memory GPUs this skips the slow
        // Private↔Shared staging blits (~0.5 GB/s) that otherwise dominate the
        // per-frame cost. No-op (falls back to the blit path) if unsupported.
        let opt_feats = if adapter_info.device_type == wgpu::DeviceType::IntegratedGpu {
            wgpu::Features::MAPPABLE_PRIMARY_BUFFERS & adapter.features()
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("spektrafilm"),
                required_features: opt_feats,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        tracing::info!(
            adapter = adapter_info.name,
            backend = ?adapter_info.backend,
            "wgpu backend initialized"
        );

        Some(Self {
            device,
            queue,
            pipeline_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            diffusion_cache: Default::default(),
        })
    }

    /// Get or compile + cache a pipeline keyed by shader source pointer.
    /// All shader sources come from `include_str!` so the pointer is stable.
    fn get_or_compile_pipeline<F>(
        &self,
        shader_source: &'static str,
        layout_entries_fn: F,
    ) -> std::sync::MutexGuard<'_, std::collections::HashMap<usize, CachedPipeline>>
    where
        F: FnOnce() -> Vec<wgpu::BindGroupLayoutEntry>,
    {
        let key = shader_source.as_ptr() as usize;
        let mut cache = self.pipeline_cache.lock().unwrap();
        if !cache.contains_key(&key) {
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("compute_shader"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
                });
            let entries = layout_entries_fn();
            let bind_group_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("compute_layout"),
                        entries: &entries,
                    });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("compute_pipeline_layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("compute_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            cache.insert(
                key,
                CachedPipeline {
                    bind_group_layout,
                    pipeline,
                },
            );
        }
        cache
    }

    /// Generic GPU compute dispatch helper.
    ///
    /// `shader_source` must be a `'static str` (typically from `include_str!`) so
    /// the cache can key by pointer identity.
    fn dispatch_compute(
        &self,
        shader_source: &'static str,
        bindings: &[GpuBuffer],
        n_pixels: u32,
        output_idx: usize,
    ) -> Vec<f32> {
        self.dispatch_compute_grid(
            shader_source,
            bindings,
            [n_pixels.div_ceil(1024), 1, 1],
            output_idx,
        )
    }
    fn dispatch_compute_grid(
        &self,
        shader_source: &'static str,
        bindings: &[GpuBuffer],
        groups: [u32; 3],
        output_idx: usize,
    ) -> Vec<f32> {
        let t_start = std::time::Instant::now();
        let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: b.binding_type,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let entries_for_compile = layout_entries.clone();
        let cache = self.get_or_compile_pipeline(shader_source, || entries_for_compile);
        let key = shader_source.as_ptr() as usize;
        let cached = cache.get(&key).expect("pipeline just inserted");
        let pipeline = &cached.pipeline;
        let bind_group_layout = &cached.bind_group_layout;
        let t_compile = t_start.elapsed();

        // Create GPU buffers
        let gpu_buffers: Vec<wgpu::Buffer> = bindings
            .iter()
            .map(|b| {
                use wgpu::util::DeviceExt;
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("buffer"),
                        contents: &b.data,
                        usage: b.usage,
                    })
            })
            .collect();

        let bind_entries: Vec<wgpu::BindGroupEntry> = gpu_buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bind_group"),
            layout: &bind_group_layout,
            entries: &bind_entries,
        });

        // Readback buffer
        let output_size = bindings[output_idx].data.len() as u64;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dispatch
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
        }
        encoder.copy_buffer_to_buffer(&gpu_buffers[output_idx], 0, &readback, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        // Read back
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback.unmap();
        let _t_total = t_start.elapsed();
        let _ = t_compile;
        result
    }

    /// GPU separable Gaussian blur via two FIR passes (horizontal then vertical).
    /// Kernel weights are computed on CPU and uploaded as a storage buffer.
    /// Two ping-pong image buffers minimize allocations.
    pub fn gaussian_blur_gpu(&self, img: &ImageBuf, sigma: f32) -> ImageBuf {
        spatial::gaussian_multi(self, img, &[sigma]).remove(0)
    }

    /// Blur `img` with every sigma in `sigmas`, all within a single command
    /// buffer. One upload, N pairs of H/V dispatches, one submit, one
    /// readback that fans out into N output `ImageBuf`s.
    ///
    /// Used by halation (multi-bounce blurs of the same source) and any
    /// caller that needs the same input at several radii.
    pub fn gaussian_blur_multi_gpu(&self, img: &ImageBuf, sigmas: &[f32]) -> Vec<ImageBuf> {
        spatial::gaussian_multi(self, img, sigmas)
    }

    /// GPU-resident pipeline: runs the front pass (hanatos LUT lookup or
    /// mallett matmul), highlight boost, camera diffusion/lens blur,
    /// halation, density curves, DIR, grain, print spectral, enlarger
    /// diffusion, scan spectral, glare, gamut compression, scanner lens blur,
    /// and unsharp as a single command buffer with ping-pong image storage.
    /// Only one upload at the start and one readback at the end.
    pub fn run_film_chain(&self, p: &crate::FilmChainParams<'_>) -> ImageBuf {
        use wgpu::util::DeviceExt;
        let t_start = std::time::Instant::now();
        // Pull all references into locals so the existing body below
        // doesn't need a rewrite — only the param sources change.
        let image = p.image;
        let film_log_exposure = p.film_log_exposure;
        let film_density_curves_normalized = p.film_density_curves_normalized;
        let film_gamma = p.film_gamma;
        let film_channel_density = p.film_channel_density;
        let film_base_density = p.film_base_density;
        let print_illuminant = p.print_illuminant;
        let print_sensitivity = p.print_sensitivity;
        let print_normalization_factor = p.print_normalization_factor;
        let print_log_exposure = p.print_log_exposure;
        let print_density_curves = p.print_density_curves;
        let print_gamma = p.print_gamma;
        let preflash = p.preflash;
        let print_channel_density = p.print_channel_density;
        let print_base_density = p.print_base_density;
        let viewing_illuminant = p.viewing_illuminant;
        let scan_normalization = p.scan_normalization;
        let scan_xyz_to_rgb = p.scan_xyz_to_rgb;

        let n_pixels = image.pixel_count() as u32;
        let img_bytes = n_pixels as usize * 3 * 4;

        // Ping-pong image buffers (each holds H*W*3 f32 values).
        let make_img_buf = |label: &str| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: img_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let mappable = self
            .device
            .features()
            .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);

        // buf_a is the input/ping-pong buffer. The dominant per-frame cost was
        // the input upload: `queue.write_buffer` stages CPU→GPU through a slow
        // (~0.5 GB/s) blit on this unified-memory GPU. With
        // MAPPABLE_PRIMARY_BUFFERS we create buf_a already mapped (MAP_WRITE)
        // and memcpy the input straight into its (shared) memory — no staging.
        let input_f32 = scalars_to_f32(&image.data);
        let buf_a = if mappable {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("img_a"),
                size: img_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::MAP_WRITE,
                mapped_at_creation: true,
            });
            buf.slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::cast_slice(&input_f32));
            buf.unmap();
            buf
        } else {
            let buf = make_img_buf("img_a");
            self.queue
                .write_buffer(&buf, 0, bytemuck::cast_slice(&input_f32));
            buf
        };

        // buf_b holds the final RGB. With MAPPABLE_PRIMARY_BUFFERS we add
        // MAP_READ so it can be mapped directly for a zero-copy readback,
        // skipping the slow Private→Shared blit on unified memory.
        let buf_b = {
            let mut usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC;
            if mappable {
                usage |= wgpu::BufferUsages::MAP_READ;
            }
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("img_b"),
                size: img_bytes as u64,
                usage,
                mapped_at_creation: false,
            })
        };

        // Static (LUT) buffers — uploaded once.
        let mk_storage = |label: &str, bytes: &[u8]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let mk_uniform = |label: &str, bytes: &[u8]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        };

        // ── Pre-compute every static GPU-side buffer ─────────────────────
        // Filming density curves — already normalized by caller.
        let film_log_exp_f32: Vec<f32> = film_log_exposure.iter().map(|&v| v as f32).collect();
        let film_curves_f32: Vec<f32> = film_density_curves_normalized
            .iter()
            .flat_map(|r| r.iter().map(|&v| if v.is_nan() { 0.0 } else { v as f32 }))
            .collect();
        let film_log_exp_buf = mk_storage("film_log_exp", bytemuck::cast_slice(&film_log_exp_f32));
        let film_curves_buf = mk_storage("film_curves", bytemuck::cast_slice(&film_curves_f32));

        // Film spectral data (for printing pass).
        let (film_cd_f32, mut film_bd_f32) = sanitize_spectral_inputs(
            film_channel_density,
            film_base_density,
            film_channel_density.len(),
        );
        film_bd_f32.resize(film_channel_density.len(), 0.0);
        let film_cd_buf = mk_storage("film_cd", bytemuck::cast_slice(&film_cd_f32));
        let film_bd_buf = mk_storage("film_bd", bytemuck::cast_slice(&film_bd_f32));

        let print_illu_f32: Vec<f32> = print_illuminant.iter().map(|&v| v as f32).collect();
        let print_sens_f32: Vec<f32> = print_sensitivity
            .iter()
            .flat_map(|r| r.iter().map(|&v| if v.is_nan() { 0.0 } else { v as f32 }))
            .collect();
        let print_illu_buf = mk_storage("print_illu", bytemuck::cast_slice(&print_illu_f32));
        let print_sens_buf = mk_storage("print_sens", bytemuck::cast_slice(&print_sens_f32));

        // Print density curves (RAW, no normalization for print path).
        let print_log_exp_f32: Vec<f32> = print_log_exposure.iter().map(|&v| v as f32).collect();
        let print_curves_f32: Vec<f32> = print_density_curves
            .iter()
            .flat_map(|r| r.iter().map(|&v| if v.is_nan() { 0.0 } else { v as f32 }))
            .collect();
        let print_log_exp_buf =
            mk_storage("print_log_exp", bytemuck::cast_slice(&print_log_exp_f32));
        let print_curves_buf = mk_storage("print_curves", bytemuck::cast_slice(&print_curves_f32));

        // Spectral dye-density data for the scanning pass. For scan_film we
        // scan the developed film directly, so feed the film's spectral
        // densities; otherwise the print's. (print_spectral uses
        // sensitivity, not these — this buffer feeds only scan_spectral.)
        let (scan_cd_src, scan_bd_src): (&[[f64; 3]], &[f64]) = if p.scan_film {
            (film_channel_density, film_base_density)
        } else {
            (print_channel_density, print_base_density)
        };
        let (scan_cd_f32, mut scan_bd_f32) =
            sanitize_spectral_inputs(scan_cd_src, scan_bd_src, scan_cd_src.len());
        scan_bd_f32.resize(scan_cd_src.len(), 0.0);
        let scan_cd_buf = mk_storage("scan_cd", bytemuck::cast_slice(&scan_cd_f32));
        let scan_bd_buf = mk_storage("scan_bd", bytemuck::cast_slice(&scan_bd_f32));

        let view_illu_f32: Vec<f32> = viewing_illuminant.iter().map(|&v| v as f32).collect();
        let view_illu_buf = mk_storage("view_illu", bytemuck::cast_slice(&view_illu_f32));
        let cmf_x_buf = mk_storage(
            "cmf_x",
            bytemuck::cast_slice(&spektrafilm_math::spectral::CMF_X),
        );
        let cmf_y_buf = mk_storage(
            "cmf_y",
            bytemuck::cast_slice(&spektrafilm_math::spectral::CMF_Y),
        );
        let cmf_z_buf = mk_storage(
            "cmf_z",
            bytemuck::cast_slice(&spektrafilm_math::spectral::CMF_Z),
        );

        // ── Param structs ────────────────────────────────────────────────
        // Front pass (hanatos LUT lookup or mallett matmul). Both shaders
        // share this uniform layout; `lut_size` is hanatos-only (0 for
        // mallett, where the field is a pad).
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct FrontParams {
            width: u32,
            height: u32,
            lut_size: u32,
            _pad: u32,
            col0: [f32; 4],
            col1: [f32; 4],
            col2: [f32; 4],
        }
        let (m, front_lut_size) = match &p.front {
            crate::FrontPass::Hanatos2025 {
                tc_lut,
                rgb_to_adapted_xyz,
            } => (rgb_to_adapted_xyz, tc_lut.size as u32),
            crate::FrontPass::Mallett2019 { matrix } => (matrix, 0u32),
        };
        let front_params = FrontParams {
            width: image.width,
            height: image.height,
            lut_size: front_lut_size,
            _pad: 0,
            col0: [m[0][0] as f32, m[1][0] as f32, m[2][0] as f32, 0.0],
            col1: [m[0][1] as f32, m[1][1] as f32, m[2][1] as f32, 0.0],
            col2: [m[0][2] as f32, m[1][2] as f32, m[2][2] as f32, 0.0],
        };
        let front_params_buf = mk_uniform("front_params", bytemuck::bytes_of(&front_params));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct DensityParams {
            width: u32,
            height: u32,
            k: u32,
            uniform_grid: u32,
            gamma_inv: [f32; 3],
            _pad: f32,
        }
        let film_density_params = DensityParams {
            width: image.width,
            height: image.height,
            k: film_log_exposure.len() as u32,
            uniform_grid: if is_uniform(film_log_exposure) { 1 } else { 0 },
            gamma_inv: [(1.0 / film_gamma) as f32; 3],
            _pad: 0.0,
        };
        let film_density_params_buf =
            mk_uniform("film_dens_params", bytemuck::bytes_of(&film_density_params));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct PrintParams {
            width: u32,
            height: u32,
            n_wavelengths: u32,
            normalization_factor: f32,
            preflash: [f32; 3],
            _pad: f32,
        }
        let has_enlarger_diffusion = p.enlarger_diffusion.is_some();
        let print_exposure_scale = p.print_exposure_scale as f32;
        let print_norm_for_shader = if has_enlarger_diffusion && print_exposure_scale != 0.0 {
            (print_normalization_factor / p.print_exposure_scale) as f32
        } else {
            print_normalization_factor as f32
        };
        let print_params = PrintParams {
            width: image.width,
            height: image.height,
            n_wavelengths: film_channel_density.len() as u32,
            normalization_factor: print_norm_for_shader,
            preflash: [preflash[0] as f32, preflash[1] as f32, preflash[2] as f32],
            _pad: 0.0,
        };
        let print_params_buf = mk_uniform("print_params", bytemuck::bytes_of(&print_params));

        let print_density_params = DensityParams {
            width: image.width,
            height: image.height,
            k: print_log_exposure.len() as u32,
            uniform_grid: if is_uniform(print_log_exposure) { 1 } else { 0 },
            gamma_inv: [(1.0 / print_gamma) as f32; 3],
            _pad: 0.0,
        };
        let print_density_params_buf = mk_uniform(
            "print_dens_params",
            bytemuck::bytes_of(&print_density_params),
        );

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ScanParams {
            width: u32,
            height: u32,
            n_wavelengths: u32,
            normalization: f32,
            col0: [f32; 4],
            col1: [f32; 4],
            col2: [f32; 4],
            bw: [f32; 4],
        }
        let s = scan_xyz_to_rgb;
        // B&W/slide luminance remap (m, q); z=1 enables it in the shader.
        // w=1 skips the shader's [0,1] clamp — required when the gamut
        // compression pass follows (it needs the out-of-gamut values).
        let skip_clamp = 1.0;
        let bw = match p.bw_xyz_remap {
            Some((m, q)) => [m as f32, q as f32, 1.0, skip_clamp],
            None => [1.0, 0.0, 0.0, skip_clamp],
        };
        let scan_params = ScanParams {
            width: image.width,
            height: image.height,
            n_wavelengths: scan_cd_src.len() as u32,
            normalization: scan_normalization as f32,
            col0: [s[0][0] as f32, s[1][0] as f32, s[2][0] as f32, 0.0],
            col1: [s[0][1] as f32, s[1][1] as f32, s[2][1] as f32, 0.0],
            col2: [s[0][2] as f32, s[1][2] as f32, s[2][2] as f32, 0.0],
            bw,
        };
        let scan_params_buf = mk_uniform("scan_params", bytemuck::bytes_of(&scan_params));

        // ── Pre-compile pipelines (cached after first call) ──────────────
        // Each shader's bindings layout is fixed and known here.
        let density_pipe = self.cached_pipeline(
            include_str!("../../spektrafilm-shaders/wgsl/density_curve_interp.wgsl"),
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let print_pipe = self.cached_pipeline(
            include_str!("../../spektrafilm-shaders/wgsl/print_spectral.wgsl"),
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );
        let scan_pipe = self.cached_pipeline(
            include_str!("../../spektrafilm-shaders/wgsl/scan_spectral.wgsl"),
            &[
                wgpu::BufferBindingType::Uniform,
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: true },
                wgpu::BufferBindingType::Storage { read_only: false },
            ],
        );

        // ── Build bind groups (per dispatch, but no buffer creation) ─────
        let workgroup_size = 1024u32;

        // Front pass: hanatos (params + rgb_in + tc_lut + raw_out) or mallett
        // (params + rgb_in + raw_out). The tc_lut buffer rides along in the
        // tuple to outlive the encoder on the hanatos arm.
        let (front_pipe, bg_front, _front_tc_lut) = match &p.front {
            crate::FrontPass::Hanatos2025 { tc_lut, .. } => {
                let tc_lut_f32: Vec<f32> = tc_lut.data.iter().map(|&v| v as f32).collect();
                let tc_lut_buf = mk_storage("tc_lut", bytemuck::cast_slice(&tc_lut_f32));
                let pipe = self.cached_pipeline(
                    include_str!("../../spektrafilm-shaders/wgsl/hanatos2025_rgb_to_raw.wgsl"),
                    &[
                        wgpu::BufferBindingType::Uniform,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        wgpu::BufferBindingType::Storage { read_only: true },
                        wgpu::BufferBindingType::Storage { read_only: false },
                    ],
                );
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bg_hanatos"),
                    layout: &pipe.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: front_params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buf_a.as_entire_binding(),
                        }, // rgb_in
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: tc_lut_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: buf_b.as_entire_binding(),
                        }, // raw_out
                    ],
                });
                (pipe, bg, Some(tc_lut_buf))
            }
            crate::FrontPass::Mallett2019 { .. } => {
                let pipe = self.cached_pipeline(
                    include_str!("../../spektrafilm-shaders/wgsl/mallett_rgb_to_raw.wgsl"),
                    &[
                        wgpu::BufferBindingType::Uniform,
                        wgpu::BufferBindingType::Storage { read_only: true },
                        wgpu::BufferBindingType::Storage { read_only: false },
                    ],
                );
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bg_mallett"),
                    layout: &pipe.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: front_params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buf_a.as_entire_binding(),
                        }, // rgb_in
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: buf_b.as_entire_binding(),
                        }, // raw_out
                    ],
                });
                (pipe, bg, None)
            }
        };
        // After the front pass: log10 + density curve interp into normalized
        // film curves. The front pass outputs raw (not log_raw), so a small
        // log10 shader transforms buf_b in-place, then density_curve_interp
        // reads buf_b → buf_a.

        let bg_log10 = {
            let pipe = self.cached_pipeline(
                include_str!("../../spektrafilm-shaders/wgsl/log10_inplace.wgsl"),
                &[
                    wgpu::BufferBindingType::Uniform,
                    wgpu::BufferBindingType::Storage { read_only: false },
                ],
            );
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct Log10Params {
                // WGSL struct: `n: u32 + _pad: vec3<u32>`. vec3 has 16-byte alignment,
                // so the struct is 32 bytes total. We pad on the Rust side accordingly.
                n_pixels: u32,
                _pad: [u32; 7],
            }
            let log10_params_buf = mk_uniform(
                "log10_params",
                bytemuck::bytes_of(&Log10Params {
                    n_pixels,
                    _pad: [0; 7],
                }),
            );
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_log10"),
                layout: &pipe.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: log10_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_b.as_entire_binding(),
                    },
                ],
            });
            // Return both the buffer (to keep it alive) and the bind group.
            (pipe, bg, log10_params_buf)
        };

        let bg_density_film = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_density_film"),
            layout: &density_pipe.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: film_density_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                }, // log_raw
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: film_log_exp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: film_curves_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_a.as_entire_binding(),
                }, // density_cmy
            ],
        });
        let bg_print = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_print"),
            layout: &print_pipe.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: print_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_a.as_entire_binding(),
                }, // density_cmy
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: film_cd_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: film_bd_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: print_illu_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: print_sens_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_b.as_entire_binding(),
                }, // log_raw_print
            ],
        });
        let bg_density_print = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_density_print"),
            layout: &density_pipe.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: print_density_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: print_log_exp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: print_curves_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_a.as_entire_binding(),
                },
            ],
        });
        let bg_scan = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_scan"),
            layout: &scan_pipe.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scan_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_a.as_entire_binding(),
                }, // density_print
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scan_cd_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scan_bd_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: view_illu_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cmf_x_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: cmf_y_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: cmf_z_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buf_b.as_entire_binding(),
                }, // final rgb
            ],
        });

        // Readback buffer (for the final image only) — only needed when we
        // can't map buf_b directly. Skipping it on the mappable path also
        // avoids allocating a second full-image buffer per frame.
        let readback = (!mappable).then(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: img_bytes as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // ── Halation auxiliary buffers + bind groups (only if active) ────
        // Allocated up-front so they live for the encoder. The two ping-pong
        // buffers `buf_a` / `buf_b` are reused as blur input/mid; the new
        // `buf_c` and `buf_d` hold the scatter outputs and the halation
        // accumulator. None of this is touched when `p.halation` is `None`.
        let halation_state = p.halation.as_ref().map(|hp| {
            build_halation_state(
                &self.device,
                hp,
                image.width,
                image.height,
                &buf_a,
                &buf_b,
                self,
            )
        });

        // ── Highlight boost state ────────────────────────────────────────
        // Runs on raw film exposure immediately after the front pass.
        let highlight_state = p
            .highlight_boost
            .as_ref()
            .map(|hp| build_highlight_boost_state(&self.device, hp, n_pixels, &buf_b, self));

        // ── Camera diffusion filter state ────────────────────────────────
        // Applied on the raw film exposure (buf_b) right after hanatos and
        // before halation, matching the CPU filming order. Owns its own
        // downsampled scratch buffers.
        let diffusion_state = p.diffusion.as_ref().map(|plan| {
            build_diffusion_state(&self.device, plan, image.width, image.height, &buf_b, self)
        });

        let camera_lens_blur_state = p.camera_lens_blur_px.and_then(|sigma| {
            (sigma > 0.0).then(|| {
                build_simple_blur_state(
                    &self.device,
                    sigma,
                    image.width,
                    image.height,
                    &buf_b,
                    &buf_a,
                    "camera_lens",
                    self,
                )
            })
        });

        // ── Unsharp mask state ───────────────────────────────────────────
        // Last pass before readback. Blurs buf_b → buf_c (via buf_a mid),
        // then combines: buf_b_out = (1+amount)*buf_b - amount*buf_c.
        // Since the combine writes back to buf_b in-place we need to
        // route via a temporary buffer to satisfy wgpu aliasing rules.
        let unsharp_state = p.unsharp.as_ref().map(|up| {
            build_unsharp_state(
                &self.device,
                up,
                image.width,
                image.height,
                &buf_a,
                &buf_b,
                self,
            )
        });

        // ── Output gamut compression state ───────────────────────────────
        // Single per-pixel dispatch in place on buf_b, after glare and
        // before unsharp (the CPU scanning order). The C_max table is
        // baked once at pipeline construction; only the upload happens here.
        let gamut_state = p
            .gamut
            .as_ref()
            .map(|gp| build_gamut_state(&self.device, gp, &buf_b, n_pixels, self));

        // ── Glare state ──────────────────────────────────────────────────
        // Applied in place on buf_b (the scan_spectral output) just before
        // readback. Generates per-pixel lognormal noise into a scratch
        // buffer, optionally blurs it, then adds `g * rgb_offset[c]` to
        // the image. Uses buf_a (free after scan_spectral consumed
        // density_cmy) and one fresh scratch buffer.
        let glare_state = p.glare.as_ref().map(|gp| {
            build_glare_state(
                &self.device,
                gp,
                image.width,
                image.height,
                &buf_a,
                &buf_b,
                self,
            )
        });

        let scanner_lens_blur_state = p.scanner_lens_blur_px.and_then(|sigma| {
            (sigma > 0.0).then(|| {
                build_simple_blur_state(
                    &self.device,
                    sigma,
                    image.width,
                    image.height,
                    &buf_b,
                    &buf_a,
                    "scanner_lens",
                    self,
                )
            })
        });

        // ── Grain state ──────────────────────────────────────────────────
        // Applied in place on buf_a (density_cmy) after DIR couplers,
        // before print_spectral. Optionally followed by a Gaussian
        // post-blur (uses buf_b as mid, since log_raw_corrected in buf_b
        // is no longer needed after DIR's final density_curve_0).
        let grain_state = p.grain.as_ref().map(|gp| {
            build_grain_state(
                &self.device,
                gp,
                image.width,
                image.height,
                &buf_a,
                &buf_b,
                self,
            )
        });

        let layered_grain_state = p
            .layered_grain
            .as_ref()
            .map(|gp| layered_grain::build(self, gp, [image.width, image.height], &buf_a, &buf_b));

        // ── DIR couplers state ────────────────────────────────────────────
        // Allocated lazily when the DIR stage is active. Reads buf_a
        // (density_cmy from film density curve) and buf_b (log_raw),
        // produces a corrected buf_a via re-interpolation of
        // `density_curves_0` against `log_raw - correction`. Uses three
        // owned scratch buffers (correction, mid, accumulator). buf_a is
        // reused as blur mid since density_cmy is no longer needed once
        // the matmul has consumed it (the final density_curve_interp
        // overwrites buf_a anyway).
        let dir_state = p.dir_couplers.as_ref().map(|dp| {
            build_dir_state(
                &self.device,
                dp,
                image.width,
                image.height,
                &buf_a,
                &buf_b,
                self,
            )
        });

        let enlarger_log_to_linear_state = p.enlarger_diffusion.as_ref().map(|_| {
            build_log_to_linear_state(&self.device, n_pixels, print_exposure_scale, &buf_b, self)
        });
        let enlarger_diffusion_state = p.enlarger_diffusion.as_ref().map(|plan| {
            build_diffusion_state(&self.device, plan, image.width, image.height, &buf_b, self)
        });
        let enlarger_linear_to_log_state = p
            .enlarger_diffusion
            .as_ref()
            .map(|_| build_linear_to_log_state(&self.device, n_pixels, &buf_b, self));
        let input_transfer_state = if !matches!(
            p.input_transfer,
            spektrafilm_math::transfer::Transfer::Linear
        ) {
            Some(color_transfer::build(
                self,
                n_pixels,
                p.input_transfer,
                false,
                1.,
                &buf_a,
            ))
        } else {
            None
        };
        let post_scan_state = p
            .output_cctf_encoding
            .then(|| color_transfer::build(self, n_pixels, p.output_transfer, true, 1., &buf_b));

        // ── Single command buffer chaining everything ────────────────────
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let dispatch = |encoder: &mut wgpu::CommandEncoder,
                        pipe: &wgpu::ComputePipeline,
                        bg: &wgpu::BindGroup,
                        n: u32| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups((n + workgroup_size - 1) / workgroup_size, 1, 1);
        };

        if let Some(state) = &input_transfer_state {
            state.encode_passes(&mut encoder, n_pixels);
        }
        // 1. Front pass (hanatos or mallett): buf_a (rgb in) → buf_b (raw)
        dispatch(&mut encoder, &front_pipe.pipeline, &bg_front, n_pixels);
        // 1a. Highlight boost on raw, before optical scatter.
        if let Some(hs) = highlight_state.as_ref() {
            hs.encode_passes(&mut encoder);
        }
        // 1a. Camera diffusion filter in-place on buf_b (raw), before
        //     halation — mirrors the CPU filming order.
        if let Some(ds) = diffusion_state.as_ref() {
            ds.encode_passes(&mut encoder, &buf_b);
        }
        // 1b. Camera lens blur on raw, after camera diffusion and before halation.
        if let Some(bs) = camera_lens_blur_state.as_ref() {
            bs.encode(&mut encoder);
        }
        // 1b. Halation in-place on buf_b. Uses buf_a as blur scratch (the
        //     input RGB image is no longer needed), buf_c / buf_d for the
        //     scatter outputs and halation accumulator.
        if let Some(hs) = halation_state.as_ref() {
            hs.encode(&mut encoder);
        }
        // 2. log10 in-place on buf_b: raw → log_raw (3 channels per thread).
        let (log10_pipe, log10_bg, _keepalive) = &bg_log10;
        dispatch(&mut encoder, &log10_pipe.pipeline, log10_bg, n_pixels);
        // 3. Density curve (film, normalized): buf_b (log_raw) → buf_a (density_cmy)
        dispatch(
            &mut encoder,
            &density_pipe.pipeline,
            &bg_density_film,
            n_pixels,
        );
        // 3b. DIR couplers (operates on buf_a, mutates buf_b → log_raw_corrected,
        //     re-interps density curve back into buf_a).
        if let Some(ds) = dir_state.as_ref() {
            let wg_xy = (image.width.div_ceil(16), image.height.div_ceil(16));
            ds.encode_passes(&mut encoder, n_pixels, workgroup_size, wg_xy);
        }
        // 3c. Grain (in-place on buf_a, optional post-blur via buf_b mid).
        if let Some(gs) = grain_state.as_ref() {
            let wg_xy = (image.width.div_ceil(16), image.height.div_ceil(16));
            gs.encode_passes(&mut encoder, n_pixels, workgroup_size, wg_xy);
        }
        if let Some(state) = &layered_grain_state {
            state.encode(&mut encoder, n_pixels);
        }
        // 4 + 5. Printing: print_spectral (buf_a → buf_b) then the print
        //     density curve (buf_b → buf_a). Skipped for scan_film — the
        //     scan pass consumes the film density already in buf_a.
        if !p.scan_film {
            // 4. Print spectral: buf_a → buf_b (log_raw_print)
            dispatch(&mut encoder, &print_pipe.pipeline, &bg_print, n_pixels);
            if let (Some(l2l), Some(ds), Some(l2g)) = (
                enlarger_log_to_linear_state.as_ref(),
                enlarger_diffusion_state.as_ref(),
                enlarger_linear_to_log_state.as_ref(),
            ) {
                l2l.encode_passes(&mut encoder, n_pixels);
                ds.encode_passes(&mut encoder, &buf_b);
                l2g.encode_passes(&mut encoder, n_pixels);
            }
            // 5. Density curve (print, raw curves): buf_b → buf_a (density_print)
            dispatch(
                &mut encoder,
                &density_pipe.pipeline,
                &bg_density_print,
                n_pixels,
            );
        }
        // 6. Scan spectral: buf_a → buf_b (unclipped linear RGB)
        dispatch(&mut encoder, &scan_pipe.pipeline, &bg_scan, n_pixels);
        // 6b. Glare (in place on buf_b).
        if let Some(gs) = glare_state.as_ref() {
            let wg_xy = (image.width.div_ceil(16), image.height.div_ceil(16));
            gs.encode_passes(&mut encoder, n_pixels, workgroup_size, wg_xy);
        }
        // 6c. Output gamut compression (in place on buf_b) — after glare,
        //     before unsharp, mirroring the CPU scanning order.
        if let Some(gs) = gamut_state.as_ref() {
            gs.encode_passes(&mut encoder, n_pixels, workgroup_size);
        }
        // 6d. Scanner lens blur — after glare/gamut, before unsharp.
        if let Some(bs) = scanner_lens_blur_state.as_ref() {
            bs.encode(&mut encoder);
        }
        // 6d. Unsharp mask — last in-flight pass. Writes the final image
        //     back to buf_b so the readback path below is unchanged.
        if let Some(us) = unsharp_state.as_ref() {
            let wg_xy = (image.width.div_ceil(16), image.height.div_ceil(16));
            us.encode_passes(
                &mut encoder,
                n_pixels,
                workgroup_size,
                wg_xy,
                &buf_b,
                img_bytes as u64,
            );
        }
        // 6e. Final optional sRGB encode. This mirrors
        // Pipeline::apply_post_scan so the resident WGSL path can return
        // display-ready pixels without a CPU full-frame post pass.
        if let Some(state) = &post_scan_state {
            state.encode_passes(&mut encoder, n_pixels);
        }

        // Zero-copy path: when buf_b is mappable, skip the blit and map it
        // directly below. Otherwise stage it into the MAP_READ readback buffer.
        if let Some(rb) = readback.as_ref() {
            encoder.copy_buffer_to_buffer(&buf_b, 0, rb, 0, img_bytes as u64);
        }
        // Everything since function entry: CPU-side param prep, buffer
        // creation/uploads, and command encoding.
        let cpu_setup_ms = t_start.elapsed().as_secs_f32() * 1000.0;
        self.queue.submit(Some(encoder.finish()));

        // Single sync point at the end. Map buf_b directly on the zero-copy
        // path, or the staging buffer otherwise.
        let map_target = readback.as_ref().unwrap_or(&buf_b);
        let slice = map_target.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let gpu_wait_ms = t_start.elapsed().as_secs_f32() * 1000.0 - cpu_setup_ms;
        let data = slice.get_mapped_range();
        let out_f32: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        map_target.unmap();

        let out = ImageBuf::from_data(image.width, image.height, f32_to_scalars(out_f32));
        tracing::debug!(
            cpu_setup_ms = format!("{cpu_setup_ms:.1}"),
            gpu_wait_ms = format!("{gpu_wait_ms:.1}"),
            readback_ms = format!(
                "{:.1}",
                t_start.elapsed().as_secs_f32() * 1000.0 - cpu_setup_ms - gpu_wait_ms
            ),
            "film chain timings"
        );
        out
    }

    /// Get-or-compile a pipeline by shader source + binding layout. Cached by
    /// shader source pointer.
    fn cached_pipeline(
        &self,
        shader_source: &'static str,
        binding_types: &[wgpu::BufferBindingType],
    ) -> CachedPipelineRef {
        let entries: Vec<wgpu::BindGroupLayoutEntry> = binding_types
            .iter()
            .enumerate()
            .map(|(i, &ty)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let entries_for_compile = entries;
        let mut cache = self.pipeline_cache.lock().unwrap();
        let key = shader_source.as_ptr() as usize;
        if !cache.contains_key(&key) {
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("compute_shader"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_source)),
                });
            let bind_group_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("compute_layout"),
                        entries: &entries_for_compile,
                    });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("compute_pipeline_layout"),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("compute_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            cache.insert(
                key,
                CachedPipeline {
                    bind_group_layout,
                    pipeline,
                },
            );
        }
        // Drop the guard but the underlying Arc keeps the entries alive.
        // Return cloned handles.
        let cached = cache.get(&key).unwrap();
        CachedPipelineRef {
            pipeline: cached.pipeline.clone(),
            layout: cached.bind_group_layout.clone(),
        }
    }
}

#[cfg(feature = "wgpu-backend")]
#[derive(Clone)]
struct CachedPipelineRef {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

#[cfg(feature = "wgpu-backend")]
struct GpuBuffer {
    data: Vec<u8>,
    binding_type: wgpu::BufferBindingType,
    usage: wgpu::BufferUsages,
}

#[cfg(feature = "wgpu-backend")]
impl GpuBuffer {
    fn uniform(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
            binding_type: wgpu::BufferBindingType::Uniform,
            usage: wgpu::BufferUsages::UNIFORM,
        }
    }
    fn storage_ro(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
            binding_type: wgpu::BufferBindingType::Storage { read_only: true },
            usage: wgpu::BufferUsages::STORAGE,
        }
    }
    fn storage_rw(data: &[u8]) -> Self {
        Self {
            data: data.to_vec(),
            binding_type: wgpu::BufferBindingType::Storage { read_only: false },
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        }
    }
}

#[cfg(feature = "wgpu-backend")]
impl ComputeBackend for WgpuBackend {
    fn colorspace_convert(&self, img: &ImageBuf, matrix: &[[f32; 3]; 3]) -> ImageBuf {
        cpu_backend::CpuBackend.colorspace_convert(img, matrix)
    }
    fn cctf_encode_srgb(&self, img: &ImageBuf) -> ImageBuf {
        cpu_backend::CpuBackend.cctf_encode_srgb(img)
    }
    fn cctf_decode_srgb(&self, img: &ImageBuf) -> ImageBuf {
        cpu_backend::CpuBackend.cctf_decode_srgb(img)
    }
    fn gaussian_blur(&self, img: &ImageBuf, sigma: f32) -> ImageBuf {
        if sigma <= 0.0 {
            return img.clone();
        }
        // For very small sigmas the FIR overhead dominates; CPU path is fine.
        // For practical halation/glare sigmas (1-40 pixels) the GPU is much faster.
        self.gaussian_blur_gpu(img, sigma)
    }
    fn gaussian_blur_multi(&self, img: &ImageBuf, sigmas: &[f32]) -> Vec<ImageBuf> {
        if sigmas.is_empty() {
            return Vec::new();
        }
        self.gaussian_blur_multi_gpu(img, sigmas)
    }
    fn table_lookup(&self, img: &ImageBuf, table_x: &[f32], table_y: &[[f32; 3]]) -> ImageBuf {
        cpu_backend::CpuBackend.table_lookup(img, table_x, table_y)
    }
    fn lut3d_interp(&self, img: &ImageBuf, lut: &Lut3D) -> ImageBuf {
        cpu_backend::CpuBackend.lut3d_interp(img, lut)
    }

    fn scan_spectral(
        &self,
        density_cmy: &ImageBuf,
        channel_density: &[[f64; 3]],
        base_density: &[f64],
        illuminant: &[f64],
        normalization: f64,
        cat: &[[f64; 3]; 3],
        xyz_to_rgb: &[[f64; 3]; 3],
    ) -> ImageBuf {
        // GPU live-preview path collapses CAT and XYZ→RGB into a single
        // matrix — small precision drop acceptable for preview, matches
        // the same trade-off as `hanatos2025_rgb_to_raw`.
        let combined: [[f64; 3]; 3] = {
            let mut out = [[0.0f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    out[i][j] = xyz_to_rgb[i][0] * cat[0][j]
                        + xyz_to_rgb[i][1] * cat[1][j]
                        + xyz_to_rgb[i][2] * cat[2][j];
                }
            }
            out
        };
        let xyz_to_rgb = &combined;
        let n_wl = channel_density.len();
        let n_pixels = density_cmy.pixel_count() as u32;

        // Pack uniform params — mat3x3 in WGSL std140 is 3 columns of vec4.
        // GPU shaders are f32-only: cast precision-preserved inputs at the shader boundary.
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            width: u32,
            height: u32,
            n_wavelengths: u32,
            normalization: f32,
            col0: [f32; 4],
            col1: [f32; 4],
            col2: [f32; 4],
            bw: [f32; 4],
        }

        let params = Params {
            width: density_cmy.width,
            height: density_cmy.height,
            n_wavelengths: n_wl as u32,
            normalization: normalization as f32,
            col0: [
                xyz_to_rgb[0][0] as f32,
                xyz_to_rgb[1][0] as f32,
                xyz_to_rgb[2][0] as f32,
                0.0,
            ],
            col1: [
                xyz_to_rgb[0][1] as f32,
                xyz_to_rgb[1][1] as f32,
                xyz_to_rgb[2][1] as f32,
                0.0,
            ],
            col2: [
                xyz_to_rgb[0][2] as f32,
                xyz_to_rgb[1][2] as f32,
                xyz_to_rgb[2][2] as f32,
                0.0,
            ],
            // Standalone scan path carries no B&W remap (the resident chain
            // is the only caller that sets it). Identity / disabled.
            // w=1: skip the shader clamp — the per-stage CPU callers
            // (glare, gamut compression, blur/unsharp) expect the
            // unclamped scan RGB, exactly like `scan_spectral_cpu` returns;
            // the final encode+clip happens at the end of the scanning stage.
            bw: [1.0, 0.0, 0.0, 1.0],
        };

        // GPU shaders are f32-only — narrow f64 inputs at the shader boundary.
        // Metal's compiler runs fast-math (no-NaN assumption), so we sanitize CPU-side:
        // any wavelength with a NaN input has its `base_density` bumped to +1000 so
        // `pow(10, -d) ≈ 0`, zeroing that wavelength's contribution exactly the same
        // way Python's `density_to_light` zeros NaN-bearing wavelengths.
        let (cd_flat, mut bd) = sanitize_spectral_inputs(channel_density, base_density, n_wl);
        let illu_f32: Vec<f32> = illuminant.iter().map(|&v| v as f32).collect();
        bd.resize(n_wl, 0.0);
        let input_f32 = scalars_to_f32(&density_cmy.data);
        let output_bytes = vec![0u8; n_pixels as usize * 3 * 4];

        let result = self.dispatch_compute(
            include_str!("../../spektrafilm-shaders/wgsl/scan_spectral.wgsl"),
            &[
                GpuBuffer::uniform(bytemuck::bytes_of(&params)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&input_f32)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&cd_flat)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&bd)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&illu_f32)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&spektrafilm_math::spectral::CMF_X)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&spektrafilm_math::spectral::CMF_Y)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&spektrafilm_math::spectral::CMF_Z)),
                GpuBuffer::storage_rw(&output_bytes),
            ],
            n_pixels,
            8, // output buffer index
        );

        ImageBuf::from_data(
            density_cmy.width,
            density_cmy.height,
            f32_to_scalars(result),
        )
    }

    fn print_spectral(
        &self,
        density_cmy: &ImageBuf,
        channel_density: &[[f64; 3]],
        base_density: &[f64],
        illuminant: &[f64],
        sensitivity: &[[f64; 3]],
        normalization_factor: f64,
        preflash: [f64; 3],
    ) -> ImageBuf {
        let n_wl = channel_density.len();
        let n_pixels = density_cmy.pixel_count() as u32;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            width: u32,
            height: u32,
            n_wavelengths: u32,
            normalization_factor: f32,
            preflash: [f32; 3],
            _pad: f32,
        }

        let params = Params {
            width: density_cmy.width,
            height: density_cmy.height,
            n_wavelengths: n_wl as u32,
            normalization_factor: normalization_factor as f32,
            preflash: [preflash[0] as f32, preflash[1] as f32, preflash[2] as f32],
            _pad: 0.0,
        };

        // GPU shaders are f32-only — narrow f64 inputs at the shader boundary.
        // See `sanitize_spectral_inputs` for the NaN-handling story (Metal fast-math).
        let (cd_flat, mut bd) = sanitize_spectral_inputs(channel_density, base_density, n_wl);
        let sens_flat: Vec<f32> = sensitivity
            .iter()
            .flat_map(|r| r.iter().map(|&v| if v.is_nan() { 0.0 } else { v as f32 }))
            .collect();
        bd.resize(n_wl, 0.0);
        let illu_f32: Vec<f32> = illuminant.iter().map(|&v| v as f32).collect();
        let input_f32 = scalars_to_f32(&density_cmy.data);
        let output_bytes = vec![0u8; n_pixels as usize * 3 * 4];

        let result = self.dispatch_compute(
            include_str!("../../spektrafilm-shaders/wgsl/print_spectral.wgsl"),
            &[
                GpuBuffer::uniform(bytemuck::bytes_of(&params)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&input_f32)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&cd_flat)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&bd)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&illu_f32)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&sens_flat)),
                GpuBuffer::storage_rw(&output_bytes),
            ],
            n_pixels,
            6, // output buffer index
        );

        ImageBuf::from_data(
            density_cmy.width,
            density_cmy.height,
            f32_to_scalars(result),
        )
    }

    fn hanatos2025_rgb_to_raw(
        &self,
        image: &ImageBuf,
        tc_lut: &spektrafilm_math::spectral::TcLut,
        color_space: &str,
        ref_illuminant: &[f32],
        cat16: bool,
    ) -> ImageBuf {
        // GPU live-preview path collapses the two-step CAT02 adaptation
        // into a single matmul — small visible-spectrum precision drop
        // that's acceptable for preview. The CPU path keeps the two-step
        // for export bit-parity with Python.
        let rgb_to_adapted_xyz = spektrafilm_math::spectral::build_rgb_to_adapted_xyz(
            color_space,
            ref_illuminant,
            cat16,
        );
        let rgb_to_adapted_xyz = &rgb_to_adapted_xyz;
        let n_pixels = image.pixel_count() as u32;
        let lut_size = tc_lut.size as u32;

        // Pack uniform: mat3x3 in std140 = 3 vec4 columns. f32 only on GPU.
        // WGSL columns are read as columns of the matrix; storing row-major as
        // [col[0], col[1], col[2]] where col[j][i] = mat[i][j].
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            width: u32,
            height: u32,
            lut_size: u32,
            _pad: u32,
            col0: [f32; 4],
            col1: [f32; 4],
            col2: [f32; 4],
        }

        let m = rgb_to_adapted_xyz;
        let params = Params {
            width: image.width,
            height: image.height,
            lut_size,
            _pad: 0,
            col0: [m[0][0] as f32, m[1][0] as f32, m[2][0] as f32, 0.0],
            col1: [m[0][1] as f32, m[1][1] as f32, m[2][1] as f32, 0.0],
            col2: [m[0][2] as f32, m[1][2] as f32, m[2][2] as f32, 0.0],
        };

        // TC LUT data is f64 on disk; narrow to f32 at the GPU boundary.
        let tc_lut_f32: Vec<f32> = tc_lut.data.iter().map(|&v| v as f32).collect();
        let input_f32 = scalars_to_f32(&image.data);
        let output_bytes = vec![0u8; n_pixels as usize * 3 * 4];

        let result = self.dispatch_compute(
            include_str!("../../spektrafilm-shaders/wgsl/hanatos2025_rgb_to_raw.wgsl"),
            &[
                GpuBuffer::uniform(bytemuck::bytes_of(&params)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&input_f32)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&tc_lut_f32)),
                GpuBuffer::storage_rw(&output_bytes),
            ],
            n_pixels,
            3, // output buffer index
        );

        ImageBuf::from_data(image.width, image.height, f32_to_scalars(result))
    }

    fn density_curve_interp(
        &self,
        log_raw: &ImageBuf,
        log_exposure: &[f64],
        density_curves: &[[f64; 3]],
        gamma_factor: f64,
    ) -> ImageBuf {
        let n_pixels = log_raw.pixel_count() as u32;
        let k = log_exposure.len() as u32;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            width: u32,
            height: u32,
            k: u32,
            uniform_grid: u32,
            gamma_inv: [f32; 3],
            _pad: f32,
        }
        // Detect uniformly-spaced log_exposure (typical case) to use the fast path.
        let uniform = is_uniform(log_exposure);
        let gamma_inv = if gamma_factor.abs() > 1e-12 {
            (1.0 / gamma_factor) as f32
        } else {
            1.0
        };
        let params = Params {
            width: log_raw.width,
            height: log_raw.height,
            k,
            uniform_grid: if uniform { 1 } else { 0 },
            gamma_inv: [gamma_inv; 3],
            _pad: 0.0,
        };

        let log_exp_f32: Vec<f32> = log_exposure.iter().map(|&v| v as f32).collect();
        let curves_f32: Vec<f32> = density_curves
            .iter()
            .flat_map(|r| r.iter().map(|&v| if v.is_nan() { 0.0 } else { v as f32 }))
            .collect();
        let input_f32 = scalars_to_f32(&log_raw.data);
        let output_bytes = vec![0u8; n_pixels as usize * 3 * 4];

        let result = self.dispatch_compute(
            include_str!("../../spektrafilm-shaders/wgsl/density_curve_interp.wgsl"),
            &[
                GpuBuffer::uniform(bytemuck::bytes_of(&params)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&input_f32)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&log_exp_f32)),
                GpuBuffer::storage_ro(bytemuck::cast_slice(&curves_f32)),
                GpuBuffer::storage_rw(&output_bytes),
            ],
            n_pixels,
            4, // output buffer index
        );

        ImageBuf::from_data(log_raw.width, log_raw.height, f32_to_scalars(result))
    }

    fn try_run_film_chain(&self, params: &crate::FilmChainParams<'_>) -> Option<ImageBuf> {
        for plan in [
            params.diffusion.as_ref(),
            params.enlarger_diffusion.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if fft_diffusion::dimensions(
                self,
                [params.image.width, params.image.height],
                (plan.psf_side - 1) / 2,
            )
            .is_none()
            {
                return None;
            }
        }

        Some(self.run_film_chain(params))
    }

    fn pchip3d(
        &self,
        image: &ImageBuf,
        prepared: &spektrafilm_math::pchip3d::PreparedPchip3d,
        min: [f64; 3],
        max: [f64; 3],
    ) -> Option<ImageBuf> {
        Some(pchip::apply(self, image, prepared, min, max))
    }
    fn supports_extended_color(&self) -> bool {
        true
    }
    fn resident_chain_applies_post_scan(&self) -> bool {
        true
    }

    fn halation(&self, image: &ImageBuf, params: &crate::HalationGpuParams) -> Option<ImageBuf> {
        Some(spatial::apply(self, image, |src, mid, size| {
            spatial::halation(self, params, size, src, mid)
        }))
    }
    fn dir_diffusion(
        &self,
        image: &ImageBuf,
        core: f32,
        tail: f32,
        weight: f32,
    ) -> Option<ImageBuf> {
        Some(spatial::dir_diffusion(self, image, core, tail, weight))
    }
    fn diffusion_filter(
        &self,
        image: &ImageBuf,
        plan: &crate::DiffusionGpuPlan,
    ) -> Option<ImageBuf> {
        fft_diffusion::dimensions(self, [image.width, image.height], (plan.psf_side - 1) / 2)?;
        Some(fft_diffusion::apply(self, image, plan))
    }
    fn supports_layered_grain(&self) -> bool {
        true
    }
    fn layered_grain(
        &self,
        image: &ImageBuf,
        p: &crate::LayeredGrainGpuParams,
    ) -> Option<ImageBuf> {
        Some(layered_grain::apply(self, image, p))
    }
    fn is_gpu(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "wgpu (GPU)"
    }
}

/// Pre-built halation passes — owns the per-sigma kernel buffers, params
/// buffers, and bind groups so they live long enough for the encoder.
///
/// The CPU equivalent is `spektrafilm_model::diffusion::apply_halation_um`.
/// All passes operate on the resident image buffer (passed in as `buf_b`),
/// using `buf_a` as blur intermediate and two newly-allocated buffers
/// `buf_c` / `buf_d` for scatter outputs and the halation accumulator.
#[cfg(feature = "wgpu-backend")]
type HalationState = spatial::State;
#[cfg(feature = "wgpu-backend")]
struct DispatchJob {
    _params_buf: wgpu::Buffer,
    pipeline: CachedPipelineRef,
    bg: wgpu::BindGroup,
}
#[cfg(feature = "wgpu-backend")]
fn build_halation_state(
    _device: &wgpu::Device,
    hp: &crate::HalationGpuParams,
    width: u32,
    height: u32,
    buf_a: &wgpu::Buffer,
    buf_b: &wgpu::Buffer,
    backend: &WgpuBackend,
) -> HalationState {
    spatial::halation(backend, hp, [width, height], buf_b, buf_a)
}

/// Exact reflected-PSF FFT diffusion on resident buffers.
#[cfg(feature = "wgpu-backend")]
type DiffusionState = fft_diffusion::State;
#[cfg(feature = "wgpu-backend")]
fn build_diffusion_state(
    _device: &wgpu::Device,
    plan: &crate::DiffusionGpuPlan,
    width: u32,
    height: u32,
    image: &wgpu::Buffer,
    backend: &WgpuBackend,
) -> DiffusionState {
    fft_diffusion::build(backend, plan, [width, height], image)
}

#[cfg(feature = "wgpu-backend")]
type SimpleBlurState = spatial::State;
#[cfg(feature = "wgpu-backend")]
fn build_simple_blur_state(
    _device: &wgpu::Device,
    sigma: f32,
    width: u32,
    height: u32,
    src: &wgpu::Buffer,
    mid: &wgpu::Buffer,
    _label: &str,
    backend: &WgpuBackend,
) -> SimpleBlurState {
    let mut state = spatial::State::new();
    state.gaussian(backend, [width, height], [sigma; 3], src, mid, src);
    state
}
#[cfg(feature = "wgpu-backend")]
struct HighlightBoostState {
    _buffers: Vec<wgpu::Buffer>,
    reductions: Vec<(DispatchJob, u32)>,
    boost: DispatchJob,
    n_values: u32,
}

#[cfg(feature = "wgpu-backend")]
fn build_highlight_boost_state(
    device: &wgpu::Device,
    hp: &crate::HighlightBoostGpuParams,
    n_pixels: u32,
    img: &wgpu::Buffer,
    backend: &WgpuBackend,
) -> HighlightBoostState {
    use wgpu::util::DeviceExt;
    let n_values = n_pixels * 3;
    let mut counts = Vec::new();
    let mut values = n_values;
    loop {
        let blocks = values.div_ceil(2048).max(1);
        counts.push(blocks);
        if blocks == 1 {
            break;
        }
        values = blocks;
    }

    let reduce_pipe = backend.cached_pipeline(
        include_str!("../../spektrafilm-shaders/wgsl/max_reduce.wgsl"),
        &[
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::BufferBindingType::Storage { read_only: false },
        ],
    );
    let boost_pipe = backend.cached_pipeline(
        include_str!("../../spektrafilm-shaders/wgsl/highlight_boost.wgsl"),
        &[
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::BufferBindingType::Storage { read_only: false },
        ],
    );

    let buffers: Vec<wgpu::Buffer> = counts
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("highlight_reduce_{i}")),
                size: (c as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        })
        .collect();

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReduceParams {
        n_values: u32,
        _pad: [u32; 7],
    }

    let mut reductions = Vec::with_capacity(counts.len());
    for (i, &blocks) in counts.iter().enumerate() {
        let pass_values = if i == 0 { n_values } else { counts[i - 1] };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("highlight_reduce_params_{i}")),
            contents: bytemuck::bytes_of(&ReduceParams {
                n_values: pass_values,
                _pad: [0; 7],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let input = if i == 0 { img } else { &buffers[i - 1] };
        let output = &buffers[i];
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("highlight_reduce_bg_{i}")),
            layout: &reduce_pipe.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        reductions.push((
            DispatchJob {
                _params_buf: params_buf,
                pipeline: reduce_pipe.clone(),
                bg,
            },
            blocks,
        ));
    }

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct BoostParams {
        n_values: u32,
        boost_ev: f32,
        boost_range: f32,
        protect_ev: f32,
    }
    let boost_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("highlight_boost_params"),
        contents: bytemuck::bytes_of(&BoostParams {
            n_values,
            boost_ev: hp.boost_ev,
            boost_range: hp.boost_range,
            protect_ev: hp.protect_ev,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let final_max = buffers.last().expect("highlight reduction has output");
    let boost_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("highlight_boost_bg"),
        layout: &boost_pipe.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: boost_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: final_max.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: img.as_entire_binding(),
            },
        ],
    });

    HighlightBoostState {
        _buffers: buffers,
        reductions,
        boost: DispatchJob {
            _params_buf: boost_params,
            pipeline: boost_pipe,
            bg: boost_bg,
        },
        n_values,
    }
}

#[cfg(feature = "wgpu-backend")]
impl HighlightBoostState {
    fn encode_passes(&self, encoder: &mut wgpu::CommandEncoder) {
        for (job, blocks) in &self.reductions {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("highlight_reduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&job.pipeline.pipeline);
            pass.set_bind_group(0, &job.bg, &[]);
            let y = blocks.div_ceil(65535);
            pass.dispatch_workgroups(blocks.div_ceil(y), y, 1);
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("highlight_boost"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.boost.pipeline.pipeline);
        pass.set_bind_group(0, &self.boost.bg, &[]);
        let groups = self.n_values.div_ceil(1024);
        let y = groups.div_ceil(65535);
        pass.dispatch_workgroups(groups.div_ceil(y), y, 1);
    }
}

#[cfg(feature = "wgpu-backend")]
struct InplaceUnaryState {
    job: DispatchJob,
}

#[cfg(feature = "wgpu-backend")]
impl InplaceUnaryState {
    fn encode_passes(&self, encoder: &mut wgpu::CommandEncoder, n_pixels: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("inplace_unary"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.job.pipeline.pipeline);
        pass.set_bind_group(0, &self.job.bg, &[]);
        pass.dispatch_workgroups(n_pixels.div_ceil(1024), 1, 1);
    }
}

#[cfg(feature = "wgpu-backend")]
fn build_log_to_linear_state(
    device: &wgpu::Device,
    n_pixels: u32,
    scale: f32,
    img: &wgpu::Buffer,
    backend: &WgpuBackend,
) -> InplaceUnaryState {
    use wgpu::util::DeviceExt;
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n_pixels: u32,
        scale: f32,
        _pad: [u32; 2],
    }
    let pipe = backend.cached_pipeline(
        include_str!("../../spektrafilm-shaders/wgsl/log_to_linear_scaled.wgsl"),
        &[
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: false },
        ],
    );
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("print_log_to_linear_params"),
        contents: bytemuck::bytes_of(&Params {
            n_pixels,
            scale,
            _pad: [0; 2],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("print_log_to_linear_bg"),
        layout: &pipe.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: img.as_entire_binding(),
            },
        ],
    });
    InplaceUnaryState {
        job: DispatchJob {
            _params_buf: params_buf,
            pipeline: pipe,
            bg,
        },
    }
}

#[cfg(feature = "wgpu-backend")]
fn build_linear_to_log_state(
    device: &wgpu::Device,
    n_pixels: u32,
    img: &wgpu::Buffer,
    backend: &WgpuBackend,
) -> InplaceUnaryState {
    use wgpu::util::DeviceExt;
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n_pixels: u32,
        _pad: [u32; 7],
    }
    let pipe = backend.cached_pipeline(
        include_str!("../../spektrafilm-shaders/wgsl/linear_to_log10_outer.wgsl"),
        &[
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: false },
        ],
    );
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("print_linear_to_log_params"),
        contents: bytemuck::bytes_of(&Params {
            n_pixels,
            _pad: [0; 7],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("print_linear_to_log_bg"),
        layout: &pipe.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: img.as_entire_binding(),
            },
        ],
    });
    InplaceUnaryState {
        job: DispatchJob {
            _params_buf: params_buf,
            pipeline: pipe,
            bg,
        },
    }
}

/// Pre-built DIR coupler passes — owns scratch buffers, kernel buffers,
/// and bind groups so they live for the encoder.
///
/// CPU equivalent: `spektrafilm_model::couplers::apply_density_correction`.
/// Encodes per-pixel matmul → two Gaussian blurs of the correction →
/// weighted lerp via `add_scaled` → in-place subtract from log_raw via
/// `add_scaled(scale=-1)` → re-interpolation of `density_curves_0`.
#[cfg(feature = "wgpu-backend")]
struct DirState {
    _buf_correction: wgpu::Buffer, // matmul output → tail_part (after blur2)
    _buf_gaussian: wgpu::Buffer,   // gaussian_part output (after blur1)
    _buf_mix: wgpu::Buffer,        // weighted lerp output → final correction
    matmul: DispatchJob,
    blur: spatial::State,
    _buf_tail: wgpu::Buffer,
    lerp_clear: DispatchJob,      // buf_mix = (1-w) * buf_gaussian
    lerp_accumulate: DispatchJob, // buf_mix += w * buf_correction (tail_part)
    subtract: DispatchJob,        // buf_b -= buf_mix
    density_curve_0: DispatchJob, // re-interp density curves
    /// Buffers we own that the bind groups reference (log_exp, curves) —
    /// kept alive for the encoder's lifetime.
    _owned: Vec<wgpu::Buffer>,
}

#[cfg(feature = "wgpu-backend")]
fn build_dir_state(
    device: &wgpu::Device,
    dp: &crate::DirCouplersGpuParams<'_>,
    width: u32,
    height: u32,
    buf_a: &wgpu::Buffer,
    buf_b: &wgpu::Buffer,
    backend: &WgpuBackend,
) -> DirState {
    use wgpu::util::DeviceExt;
    let n_pixels = (width as usize) * (height as usize);
    let img_bytes = (n_pixels * 3 * 4) as u64;

    let mk_buf = |label: &str| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: img_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    // buf_correction holds the matmul output, then is overwritten by the
    // tail blur (in-place via the V pass of a separable blur — H and V
    // are separate compute passes so wgpu does NOT see the input/output
    // as aliasing). buf_gaussian holds the first blur's output.
    // buf_mix holds the weighted lerp result that ultimately gets
    // subtracted from log_raw.
    let buf_correction = mk_buf("dir_correction");
    let buf_gaussian = mk_buf("dir_gaussian");
    let buf_mix = mk_buf("dir_mix");

    // ── 1. dir_matmul ─────────────────────────────────────────────────
    let matmul_pipe = backend.cached_pipeline(
        include_str!("../../spektrafilm-shaders/wgsl/dir_matmul.wgsl"),
        &[
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::BufferBindingType::Storage { read_only: false },
        ],
    );
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct DirMatmulParams {
        n_pixels: u32,
        positive: u32,
        _pad0: u32,
        _pad1: u32,
        density_max: [f32; 4],
        m_row0: [f32; 4],
        m_row1: [f32; 4],
        m_row2: [f32; 4],
    }
    let m = &dp.couplers_matrix_scaled;
    let matmul_params = DirMatmulParams {
        n_pixels: n_pixels as u32,
        positive: if dp.is_positive { 1 } else { 0 },
        _pad0: 0,
        _pad1: 0,
        density_max: [dp.density_max[0], dp.density_max[1], dp.density_max[2], 0.0],
        m_row0: [m[0][0], m[0][1], m[0][2], 0.0],
        m_row1: [m[1][0], m[1][1], m[1][2], 0.0],
        m_row2: [m[2][0], m[2][1], m[2][2], 0.0],
    };
    let matmul_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dir_matmul_params"),
        contents: bytemuck::bytes_of(&matmul_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let matmul_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dir_matmul_bg"),
        layout: &matmul_pipe.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: matmul_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_correction.as_entire_binding(),
            },
        ],
    });
    let matmul = DispatchJob {
        _params_buf: matmul_params_buf,
        pipeline: matmul_pipe,
        bg: matmul_bg,
    };

    let size = [width, height];
    let buf_tail = spatial::buffer(backend, size);
    let mut blur = spatial::State::new();
    if dp.diffusion_size_px > 0. {
        blur.gaussian(
            backend,
            size,
            [dp.diffusion_size_px; 3],
            &buf_correction,
            buf_a,
            &buf_gaussian,
        );
        blur.exponential(
            backend,
            size,
            [dp.diffusion_tail_px; 3],
            &buf_correction,
            buf_a,
            &buf_tail,
        );
    } else {
        // Zero spatial size disables the entire correction diffusion, including its tail.
        blur.mix(
            backend,
            size,
            &buf_correction,
            &buf_gaussian,
            [1.; 3],
            [0.; 3],
        );
    }

    // ── 3. Weighted lerp via two add_scaled passes ────────────────────
    // buf_mix = (1-w) * buf_gaussian   (clear_first writes scale*src into dst)
    // buf_mix += w * buf_correction  (tail_part)
    let add_scaled_pipe = backend.cached_pipeline(
        include_str!("../../spektrafilm-shaders/wgsl/add_scaled.wgsl"),
        &[
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::BufferBindingType::Storage { read_only: false },
        ],
    );
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct AddScaledParams {
        n_pixels: u32,
        scale: f32,
        clear_first: u32,
        _pad: u32,
    }
    let w = if dp.diffusion_size_px > 0. {
        dp.diffusion_tail_weight
    } else {
        0.
    };
    let lerp_clear_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dir_lerp_clear_params"),
        contents: bytemuck::bytes_of(&AddScaledParams {
            n_pixels: n_pixels as u32,
            scale: 1.0 - w,
            clear_first: 1,
            _pad: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    // src = buf_gaussian (read), dst = buf_mix (write). With clear_first=1
    // this is `buf_mix = (1-w) * buf_gaussian`. Different buffers, no
    // aliasing.
    let lerp_clear_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dir_lerp_clear_bg"),
        layout: &add_scaled_pipe.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lerp_clear_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_gaussian.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_mix.as_entire_binding(),
            },
        ],
    });
    let lerp_clear = DispatchJob {
        _params_buf: lerp_clear_params,
        pipeline: add_scaled_pipe.clone(),
        bg: lerp_clear_bg,
    };

    let lerp_acc_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dir_lerp_acc_params"),
        contents: bytemuck::bytes_of(&AddScaledParams {
            n_pixels: n_pixels as u32,
            scale: w,
            clear_first: 0,
            _pad: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let lerp_acc_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dir_lerp_acc_bg"),
        layout: &add_scaled_pipe.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lerp_acc_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_tail.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_mix.as_entire_binding(),
            },
        ],
    });
    let lerp_accumulate = DispatchJob {
        _params_buf: lerp_acc_params,
        pipeline: add_scaled_pipe.clone(),
        bg: lerp_acc_bg,
    };

    // ── 4. Subtract: buf_b -= buf_mix ────────────────────────────────
    let subtract_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dir_subtract_params"),
        contents: bytemuck::bytes_of(&AddScaledParams {
            n_pixels: n_pixels as u32,
            scale: -1.0,
            clear_first: 0,
            _pad: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let subtract_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dir_subtract_bg"),
        layout: &add_scaled_pipe.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: subtract_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_mix.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_b.as_entire_binding(),
            },
        ],
    });
    let subtract = DispatchJob {
        _params_buf: subtract_params,
        pipeline: add_scaled_pipe,
        bg: subtract_bg,
    };

    // ── 5. Final density curve re-interp using density_curves_0 ───────
    let density_pipe = backend.cached_pipeline(
        include_str!("../../spektrafilm-shaders/wgsl/density_curve_interp.wgsl"),
        &[
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::BufferBindingType::Storage { read_only: false },
        ],
    );
    let log_exp_f32: Vec<f32> = dp.log_exposure.iter().map(|&v| v as f32).collect();
    let curves_f32: Vec<f32> = dp
        .density_curves_0
        .iter()
        .flat_map(|r| r.iter().map(|&v| if v.is_nan() { 0.0 } else { v as f32 }))
        .collect();
    let log_exp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dir_density_log_exp"),
        contents: bytemuck::cast_slice(&log_exp_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let curves_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dir_density_curves_0"),
        contents: bytemuck::cast_slice(&curves_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct DensityParams {
        width: u32,
        height: u32,
        k: u32,
        uniform_grid: u32,
        gamma_inv: [f32; 3],
        _pad: f32,
    }
    let density_params = DensityParams {
        width,
        height,
        k: dp.log_exposure.len() as u32,
        uniform_grid: if is_uniform(dp.log_exposure) { 1 } else { 0 },
        gamma_inv: [(1.0 / dp.gamma_factor) as f32; 3],
        _pad: 0.0,
    };
    let density_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dir_density_params"),
        contents: bytemuck::bytes_of(&density_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let density_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dir_density_bg"),
        layout: &density_pipe.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: density_params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_b.as_entire_binding(),
            }, // input: corrected log_raw
            wgpu::BindGroupEntry {
                binding: 2,
                resource: log_exp_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: curves_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf_a.as_entire_binding(),
            }, // output: corrected density_cmy
        ],
    });
    // Keep log_exp_buf / curves_buf alive via the DispatchJob's
    // _params_buf isn't ideal — pack them into the state instead.
    let density_curve_0 = DispatchJob {
        _params_buf: density_params_buf,
        pipeline: density_pipe,
        bg: density_bg,
    };

    // The two storage buffers (log_exp, curves) need to outlive the
    // encoder. Stuff them somewhere — extend DirState with a small
    // owner Vec.
    DirState {
        _buf_correction: buf_correction,
        _buf_gaussian: buf_gaussian,
        _buf_mix: buf_mix,
        matmul,
        blur,
        _buf_tail: buf_tail,
        lerp_clear,
        lerp_accumulate,
        subtract,
        density_curve_0,

        _owned: vec![log_exp_buf, curves_buf],
    }
}

#[cfg(feature = "wgpu-backend")]
impl DirState {
    fn encode_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n_pixels: u32,
        workgroup_size: u32,
        _wg_xy: (u32, u32),
    ) {
        let dispatch_linear = |enc: &mut wgpu::CommandEncoder, job: &DispatchJob| {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dir_linear"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&job.pipeline.pipeline);
            pass.set_bind_group(0, &job.bg, &[]);
            pass.dispatch_workgroups((n_pixels + workgroup_size - 1) / workgroup_size, 1, 1);
        };

        dispatch_linear(encoder, &self.matmul);
        self.blur.encode(encoder);
        dispatch_linear(encoder, &self.lerp_clear); // buf_acc *= (1-w)
        dispatch_linear(encoder, &self.lerp_accumulate); // buf_acc += w * buf_correction
        dispatch_linear(encoder, &self.subtract); // buf_b -= buf_acc
        dispatch_linear(encoder, &self.density_curve_0); // buf_b → buf_a (corrected)
    }
}

/// Pre-built unsharp mask pass — owns the blur and the combine dispatch.
///
/// CPU equivalent: `spektrafilm_model::diffusion::apply_unsharp_mask`.
/// The combine shader writes to a temp buffer (avoids same-pass
/// aliasing); `encode_passes` then `copy_buffer_to_buffer`s the result
/// back into buf_b so the rest of the pipeline doesn't need to know
/// about the temporary.
#[cfg(feature = "wgpu-backend")]
struct UnsharpState {
    blur: spatial::State,
    combine: DispatchJob,
    _blur_dst: wgpu::Buffer,
    out_buf: wgpu::Buffer, // combine target; copied back to buf_b at the end
}

#[cfg(feature = "wgpu-backend")]
fn build_unsharp_state(
    device: &wgpu::Device,
    up: &crate::UnsharpGpuParams,
    width: u32,
    height: u32,
    buf_a: &wgpu::Buffer,
    buf_b: &wgpu::Buffer,
    backend: &WgpuBackend,
) -> UnsharpState {
    use wgpu::util::DeviceExt;
    let n_pixels = (width as usize) * (height as usize);
    let img_bytes = (n_pixels * 3 * 4) as u64;

    // Output buffer for the combine pass. Cleared each render — fine,
    // it's only used between the combine dispatch and the copy_buffer.
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("unsharp_out"),
        size: img_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let blur_dst = spatial::buffer(backend, [width, height]);
    let mut blur = spatial::State::new();
    blur.gaussian(
        backend,
        [width, height],
        [up.sigma_px; 3],
        buf_b,
        buf_a,
        &blur_dst,
    );

    // ── Combine: out = (1+amount)*orig - amount*blurred
    let combine_pipe = backend.cached_pipeline(
        include_str!("../../spektrafilm-shaders/wgsl/unsharp_combine.wgsl"),
        &[
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::BufferBindingType::Storage { read_only: false },
        ],
    );
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct CombineParams {
        n_pixels: u32,
        amount: f32,
        _pad0: u32,
        _pad1: u32,
    }
    let combine_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("unsharp_combine_params"),
        contents: bytemuck::bytes_of(&CombineParams {
            n_pixels: n_pixels as u32,
            amount: up.amount,
            _pad0: 0,
            _pad1: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let combine_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("unsharp_combine_bg"),
        layout: &combine_pipe.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: combine_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_b.as_entire_binding(),
            }, // a = original
            wgpu::BindGroupEntry {
                binding: 2,
                resource: blur_dst.as_entire_binding(),
            }, // b = blurred
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            }, // out = sharpened
        ],
    });
    let combine = DispatchJob {
        _params_buf: combine_params,
        pipeline: combine_pipe,
        bg: combine_bg,
    };

    UnsharpState {
        blur,
        combine,
        _blur_dst: blur_dst,
        out_buf,
    }
}

#[cfg(feature = "wgpu-backend")]
impl UnsharpState {
    fn encode_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n_pixels: u32,
        workgroup_size: u32,
        _wg_xy: (u32, u32),
        buf_b: &wgpu::Buffer,
        img_bytes: u64,
    ) {
        self.blur.encode(encoder);
        // Combine.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("unsharp_combine"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.combine.pipeline.pipeline);
            pass.set_bind_group(0, &self.combine.bg, &[]);
            pass.dispatch_workgroups((n_pixels + workgroup_size - 1) / workgroup_size, 1, 1);
        }
        // Copy sharpened result back into buf_b so the downstream
        // readback sees it without needing to know we used a temp.
        encoder.copy_buffer_to_buffer(&self.out_buf, 0, buf_b, 0, img_bytes);
    }
}

/// Pre-built glare pass — owns the noise-gen dispatch, optional blur,
/// and the additive RGB apply.
///
/// CPU equivalent:
/// `spektrafilm_model::glare::{compute_random_glare_amount,
/// add_glare_with_amount}`. Operates on buf_b (the RGB output of
/// scan_spectral) in place; the lognormal noise is held in a fresh
/// scratch buffer, blurred via the existing 3-channel separable
/// Gaussian.
#[cfg(feature = "wgpu-backend")]
struct GlareState {
    _scratch: wgpu::Buffer,
    gen_dispatch: DispatchJob,
    blur: spatial::State,
    apply_dispatch: DispatchJob,
}

#[cfg(feature = "wgpu-backend")]
fn build_glare_state(
    device: &wgpu::Device,
    gp: &crate::GlareGpuParams,
    width: u32,
    height: u32,
    buf_a: &wgpu::Buffer,
    buf_b: &wgpu::Buffer,
    backend: &WgpuBackend,
) -> GlareState {
    use wgpu::util::DeviceExt;
    let n_pixels = (width as usize) * (height as usize);
    let img_bytes = (n_pixels * 3 * 4) as u64;

    let scratch = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("glare_scratch"),
        size: img_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // ── Noise generation ──────────────────────────────────────────────
    let gen_pipe = backend.cached_pipeline(
        include_str!("../../spektrafilm-shaders/wgsl/glare_gen.wgsl"),
        &[
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: false },
        ],
    );
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct GenParams {
        n_pixels: u32,
        base_seed: u32,
        mu: f32,
        sigma: f32,
    }
    let gen_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("glare_gen_params"),
        contents: bytemuck::bytes_of(&GenParams {
            n_pixels: n_pixels as u32,
            base_seed: gp.base_seed,
            mu: gp.mu,
            sigma: gp.sigma,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let gen_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("glare_gen_bg"),
        layout: &gen_pipe.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gen_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scratch.as_entire_binding(),
            },
        ],
    });
    let gen_dispatch = DispatchJob {
        _params_buf: gen_params,
        pipeline: gen_pipe,
        bg: gen_bg,
    };

    // Share the same reference FIR / compensated IIR as every physical blur.
    let mut blur = spatial::State::new();
    blur.gaussian(
        backend,
        [width, height],
        [gp.blur_px; 3],
        &scratch,
        buf_a,
        &scratch,
    );

    // ── Apply: image[c] += glare_amount * rgb_offset[c] ───────────────
    let apply_pipe = backend.cached_pipeline(
        include_str!("../../spektrafilm-shaders/wgsl/glare_apply.wgsl"),
        &[
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::BufferBindingType::Storage { read_only: false },
        ],
    );
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct ApplyParams {
        n_pixels: u32,
        _pad0: u32,
        _pad1: u32,
        _pad2: u32,
        offset: [f32; 4],
    }
    let apply_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("glare_apply_params"),
        contents: bytemuck::bytes_of(&ApplyParams {
            n_pixels: n_pixels as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            offset: [gp.rgb_offset[0], gp.rgb_offset[1], gp.rgb_offset[2], 0.0],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let apply_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("glare_apply_bg"),
        layout: &apply_pipe.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: apply_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: scratch.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_b.as_entire_binding(),
            },
        ],
    });
    let apply_dispatch = DispatchJob {
        _params_buf: apply_params,
        pipeline: apply_pipe,
        bg: apply_bg,
    };

    GlareState {
        _scratch: scratch,
        gen_dispatch,
        blur,
        apply_dispatch,
    }
}

#[cfg(feature = "wgpu-backend")]
impl GlareState {
    fn encode_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n_pixels: u32,
        workgroup_size: u32,
        _wg_xy: (u32, u32),
    ) {
        // 1. Noise generation.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("glare_gen"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gen_dispatch.pipeline.pipeline);
            pass.set_bind_group(0, &self.gen_dispatch.bg, &[]);
            pass.dispatch_workgroups((n_pixels + workgroup_size - 1) / workgroup_size, 1, 1);
        }
        self.blur.encode(encoder);
        // 3. Apply onto buf_b.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("glare_apply"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.apply_dispatch.pipeline.pipeline);
            pass.set_bind_group(0, &self.apply_dispatch.bg, &[]);
            pass.dispatch_workgroups((n_pixels + workgroup_size - 1) / workgroup_size, 1, 1);
        }
    }
}

/// Pre-built output gamut compression pass — a single per-pixel dispatch
/// in place on `buf_b` (the scan output), between glare and unsharp,
/// matching the CPU scanning order. Owns the uniform + `C_max` table
/// buffers so they outlive the encoder.
///
/// CPU equivalent: `OutputGamutCompress::compress`.
#[cfg(feature = "wgpu-backend")]
struct GamutState {
    _cmax_buf: wgpu::Buffer,
    dispatch: DispatchJob,
}

#[cfg(feature = "wgpu-backend")]
fn build_gamut_state(
    device: &wgpu::Device,
    gp: &crate::GamutGpuParams<'_>,
    buf_b: &wgpu::Buffer,
    n_pixels: u32,
    backend: &WgpuBackend,
) -> GamutState {
    use wgpu::util::DeviceExt;
    let pipe = backend.cached_pipeline(
        include_str!("../../spektrafilm-shaders/wgsl/gamut_compress.wgsl"),
        &[
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::BufferBindingType::Storage { read_only: false },
        ],
    );
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct GamutParams {
        n_pixels: u32,
        mode: u32,
        n_l: u32,
        n_h: u32,
        knee: [f32; 4],
        lightness: [f32; 4],
        lspace: [f32; 4],
        rgb_to_xyz: [[f32; 4]; 3],
        xyz_to_rgb: [[f32; 4]; 3],
        cam: [f32; 4],
    }
    let lightness = match gp.lightness {
        Some([t, l, p]) => [t, l, p, 1.0],
        None => [0.0, 1.0, 1.0, 0.0],
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gamut_params"),
        contents: bytemuck::bytes_of(&GamutParams {
            n_pixels,
            mode: gp.mode,
            n_l: gp.n_l,
            n_h: gp.n_h,
            knee: [gp.knee[0], gp.knee[1], gp.knee[2], 0.0],
            lightness,
            lspace: [gp.l_white, gp.l_min, gp.l_max, 0.0],
            rgb_to_xyz: gp.rgb_to_xyz.map(|r| [r[0], r[1], r[2], 0.]),
            xyz_to_rgb: gp.xyz_to_rgb.map(|r| [r[0], r[1], r[2], 0.]),
            cam: [gp.cam_gain[0], gp.cam_gain[1], gp.cam_gain[2], gp.cam_aw],
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    // aces_rgc carries no table — bind a 1-element placeholder (the shader
    // never reads it on that mode; wgpu requires a non-empty binding).
    let cmax_f32: Vec<f32> = if gp.cmax.is_empty() {
        vec![0.0]
    } else {
        gp.cmax.iter().map(|&v| v as f32).collect()
    };
    let cmax_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gamut_cmax"),
        contents: bytemuck::cast_slice(&cmax_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gamut_bg"),
        layout: &pipe.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: cmax_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_b.as_entire_binding(),
            },
        ],
    });
    GamutState {
        _cmax_buf: cmax_buf,
        dispatch: DispatchJob {
            _params_buf: params_buf,
            pipeline: pipe,
            bg,
        },
    }
}

#[cfg(feature = "wgpu-backend")]
impl GamutState {
    fn encode_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n_pixels: u32,
        workgroup_size: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gamut_compress"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.dispatch.pipeline.pipeline);
        pass.set_bind_group(0, &self.dispatch.bg, &[]);
        pass.dispatch_workgroups((n_pixels + workgroup_size - 1) / workgroup_size, 1, 1);
    }
}

/// Pre-built grain pass — owns the grain compute bind group plus the
/// optional post-blur pipeline state.
///
/// CPU equivalent: `spektrafilm_model::grain::apply_grain_to_density`.
/// Operates in place on `buf_a` (density_cmy). When `grain_blur > 0.4`,
/// follows up with a separable Gaussian blur that uses `buf_b` as mid
/// (free at this point — DIR's `density_curve_0` already overwrote
/// buf_b's role).
#[cfg(feature = "wgpu-backend")]
struct GrainState {
    grain_dispatch: DispatchJob,
    blur: Option<spatial::State>,
    _blur_mid: Option<wgpu::Buffer>,
}

#[cfg(feature = "wgpu-backend")]
fn build_grain_state(
    device: &wgpu::Device,
    gp: &crate::GrainGpuParams,
    width: u32,
    height: u32,
    buf_a: &wgpu::Buffer,
    buf_b: &wgpu::Buffer,
    backend: &WgpuBackend,
) -> GrainState {
    use wgpu::util::DeviceExt;
    let n_pixels = (width as usize) * (height as usize);

    let grain_pipe = backend.cached_pipeline(
        concat!(
            include_str!("../../spektrafilm-shaders/wgsl/particle_sampling.wgsl"),
            include_str!("../../spektrafilm-shaders/wgsl/grain.wgsl")
        ),
        &[
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: false },
        ],
    );
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct GrainParams {
        n_pixels: u32,
        base_seed: u32,
        n_sub_layers: u32,
        monochrome: u32,
        density_min: [f32; 4],
        density_max: [f32; 4],
        n_particles_per_pixel: [f32; 4],
        grain_uniformity: [f32; 4],
    }
    let params = GrainParams {
        n_pixels: n_pixels as u32,
        base_seed: gp.base_seed,
        n_sub_layers: gp.n_sub_layers.max(1),
        monochrome: gp.monochrome as u32,
        density_min: [gp.density_min[0], gp.density_min[1], gp.density_min[2], 0.0],
        density_max: [gp.density_max[0], gp.density_max[1], gp.density_max[2], 0.0],
        n_particles_per_pixel: [
            gp.n_particles_per_pixel[0],
            gp.n_particles_per_pixel[1],
            gp.n_particles_per_pixel[2],
            0.0,
        ],
        grain_uniformity: [
            gp.grain_uniformity[0],
            gp.grain_uniformity[1],
            gp.grain_uniformity[2],
            0.0,
        ],
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grain_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("grain_bg"),
        layout: &grain_pipe.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_a.as_entire_binding(),
            },
        ],
    });
    let grain_dispatch = DispatchJob {
        _params_buf: params_buf,
        pipeline: grain_pipe,
        bg,
    };

    let (blur, mid) = if gp.grain_blur > 0.4 {
        let mid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simple grain blur scratch"),
            size: n_pixels as u64 * 12,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut state = spatial::State::new();
        state.gaussian(
            backend,
            [width, height],
            [gp.grain_blur; 3],
            buf_a,
            buf_b,
            &mid,
        );
        // gaussian writes its output to mid, then return to the film buffer.
        state.copy(&mid, buf_a, n_pixels as u64 * 12);
        (Some(state), Some(mid))
    } else {
        (None, None)
    };
    GrainState {
        grain_dispatch,
        blur,
        _blur_mid: mid,
    }
}
#[cfg(feature = "wgpu-backend")]
impl GrainState {
    fn encode_passes(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n_pixels: u32,
        workgroup_size: u32,
        _wg_xy: (u32, u32),
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("exact simple grain"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.grain_dispatch.pipeline.pipeline);
            pass.set_bind_group(0, &self.grain_dispatch.bg, &[]);
            let groups = n_pixels.div_ceil(workgroup_size);
            let y = groups.div_ceil(65535).max(1);
            pass.dispatch_workgroups(groups.div_ceil(y), y, 1);
        }
        if let Some(blur) = &self.blur {
            blur.encode(encoder);
        }
    }
}

/// Detects whether a 1D sequence is uniformly spaced (within tight tolerance).
#[cfg(feature = "wgpu-backend")]
fn is_uniform(xs: &[f64]) -> bool {
    if xs.len() < 3 {
        return true;
    }
    let step = (xs[xs.len() - 1] - xs[0]) / (xs.len() as f64 - 1.0);
    let tol = step.abs() * 1e-9 + 1e-12;
    for i in 1..xs.len() {
        let expected = xs[0] + (i as f64) * step;
        if (xs[i] - expected).abs() > tol {
            return false;
        }
    }
    true
}
