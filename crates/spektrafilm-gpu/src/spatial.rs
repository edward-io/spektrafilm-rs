//! Resident, channel-aware filters sharing the CPU reference's FIR/IIR model.
use super::*;
use wgpu::util::DeviceExt;
const READ: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: true };
const WRITE: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: false };
enum Step {
    Compute(DispatchJob, [u32; 3]),
    Copy(wgpu::Buffer, wgpu::Buffer, u64),
}
pub(super) struct State {
    jobs: Vec<Step>,
    buffers: Vec<wgpu::Buffer>,
    lo: Option<std::sync::Arc<wgpu::Buffer>>,
    pixel_grid_limit: u32,
}
impl State {
    pub(super) fn new() -> Self {
        Self {
            jobs: vec![],
            buffers: vec![],
            lo: None,
            pixel_grid_limit: 65535,
        }
    }
    pub(super) fn copy(&mut self, src: &wgpu::Buffer, dst: &wgpu::Buffer, bytes: u64) {
        self.jobs.push(Step::Copy(src.clone(), dst.clone(), bytes));
    }
    pub(super) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        for step in &self.jobs {
            match step {
                Step::Compute(job, groups) => {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("reference spatial filter"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&job.pipeline.pipeline);
                    pass.set_bind_group(0, &job.bg, &[]);
                    pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                }
                Step::Copy(src, dst, bytes) => {
                    encoder.copy_buffer_to_buffer(src, 0, dst, 0, *bytes)
                }
            }
        }
    }

    fn pixel_groups(&self, size: [u32; 2]) -> [u32; 3] {
        let groups = (size[0] * size[1]).div_ceil(256);
        let y = groups.div_ceil(self.pixel_grid_limit).max(1);
        [groups.div_ceil(y), y, 1]
    }
    fn dispatch<T: bytemuck::Pod>(
        &mut self,
        b: &WgpuBackend,
        shader: &'static str,
        params: &T,
        bindings: &[(&wgpu::Buffer, wgpu::BufferBindingType)],
        groups: [u32; 3],
    ) {
        let mut layout = vec![wgpu::BufferBindingType::Uniform];
        layout.extend(bindings.iter().map(|(_, ty)| *ty));
        let pipeline = b.cached_pipeline(shader, &layout);
        let uniform = b
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spatial params"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }];
        entries.extend(
            bindings
                .iter()
                .enumerate()
                .map(|(i, (buffer, _))| wgpu::BindGroupEntry {
                    binding: i as u32 + 1,
                    resource: buffer.as_entire_binding(),
                }),
        );
        let bg = b.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spatial"),
            layout: &pipeline.layout,
            entries: &entries,
        });
        self.jobs.push(Step::Compute(
            DispatchJob {
                _params_buf: uniform,
                pipeline,
                bg,
            },
            groups,
        ));
    }
    pub(super) fn gaussian(
        &mut self,
        b: &WgpuBackend,
        size: [u32; 2],
        sigma: [f32; 3],
        src: &wgpu::Buffer,
        mid: &wgpu::Buffer,
        dst: &wgpu::Buffer,
    ) {
        let radius = sigma.map(|s| {
            if s < 3. {
                (3. * s.max(0.) as f64 + 0.5) as u32
            } else {
                0
            }
        });
        if sigma.iter().all(|s| *s < 3.) && radius == [0; 3] {
            if src != dst {
                self.jobs.push(Step::Copy(
                    src.clone(),
                    dst.clone(),
                    size[0] as u64 * size[1] as u64 * 12,
                ));
            }
            return;
        }
        let stride = 2 * radius.into_iter().max().unwrap() + 1;
        let mut kernel = vec![0f32; stride as usize * 3];
        let mut high = [[0.; 4]; 3];
        let mut low = high;
        for c in 0..3 {
            if sigma[c] < 3. {
                let r = radius[c] as i32;
                let weights: Vec<_> = (-r..=r)
                    .map(|k| {
                        if r == 0 {
                            1.
                        } else {
                            (-0.5 * (k as f64 / sigma[c] as f64).powi(2)).exp()
                        }
                    })
                    .collect();
                let sum = weights.iter().sum::<f64>();
                for (i, v) in weights.into_iter().enumerate() {
                    kernel[c * stride as usize + i] = (v / sum) as f32;
                }
            } else {
                let s = sigma[c] as f64;
                let q = 0.98711 * s - 0.96330;
                let q2 = q * q;
                let q3 = q2 * q;
                let b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3;
                let b1 = (2.44413 * q + 2.85619 * q2 + 1.26661 * q3) / b0;
                let b2 = -(1.4281 * q2 + 1.26661 * q3) / b0;
                let b3 = 0.422205 * q3 / b0;
                for (i, value) in [1. - b1 - b2 - b3, b1, b2, b3].into_iter().enumerate() {
                    high[c][i] = value as f32;
                    low[c][i] = (value - high[c][i] as f64) as f32;
                }
            }
        }
        let kernel = b
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spatial FIR kernel"),
                contents: bytemuck::cast_slice(&kernel),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let lo = self
            .lo
            .get_or_insert_with(|| std::sync::Arc::new(buffer(b, size)))
            .clone();
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            shape: [u32; 4],
            modes: [u32; 4],
            radius: [u32; 4],
            high: [[f32; 4]; 3],
            low: [[f32; 4]; 3],
        }
        let modes = [
            u32::from(sigma[0] >= 3.),
            u32::from(sigma[1] >= 3.),
            u32::from(sigma[2] >= 3.),
            0,
        ];
        for axis in 0..2 {
            let (input, output) = if axis == 0 { (src, mid) } else { (mid, dst) };
            let p = Params {
                shape: [size[0], size[1], axis, stride],
                modes,
                radius: [radius[0], radius[1], radius[2], 0],
                high,
                low,
            };
            let bindings = [
                (input, READ),
                (output, WRITE),
                (&kernel, READ),
                (lo.as_ref(), WRITE),
            ];
            if modes[..3].contains(&0) {
                self.dispatch(
                    b,
                    include_str!("../../spektrafilm-shaders/wgsl/spatial_gaussian_fir.wgsl"),
                    &p,
                    &bindings,
                    self.pixel_groups(size),
                );
            }
            if modes[..3].iter().any(|v| *v > 0) {
                let lines = if axis == 0 { size[1] } else { size[0] };
                self.dispatch(
                    b,
                    include_str!("../../spektrafilm-shaders/wgsl/spatial_gaussian_iir.wgsl"),
                    &p,
                    &bindings,
                    [(lines * 3).div_ceil(64), 1, 1],
                );
            }
        }
        self.buffers.push(kernel);
    }
    pub(super) fn mix(
        &mut self,
        b: &WgpuBackend,
        size: [u32; 2],
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        source: [f32; 3],
        destination: [f32; 3],
    ) {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            header: [u32; 4],
            source: [f32; 4],
            destination: [f32; 4],
        }
        self.dispatch(
            b,
            include_str!("../../spektrafilm-shaders/wgsl/spatial_mix.wgsl"),
            &Params {
                header: [size[0] * size[1], 0, 0, 0],
                source: [source[0], source[1], source[2], 0.],
                destination: [destination[0], destination[1], destination[2], 0.],
            },
            &[(src, READ), (dst, WRITE)],
            self.pixel_groups(size),
        );
    }
    pub(super) fn exponential(
        &mut self,
        b: &WgpuBackend,
        size: [u32; 2],
        sigma: [f32; 3],
        src: &wgpu::Buffer,
        mid: &wgpu::Buffer,
        dst: &wgpu::Buffer,
    ) {
        let component = buffer(b, size);
        for (i, (weight, ratio)) in [(0.1633, 0.5360), (0.6496, 1.5236), (0.1870, 2.7684)]
            .into_iter()
            .enumerate()
        {
            self.gaussian(
                b,
                size,
                sigma.map(|s| (s as f64 * ratio) as f32),
                src,
                mid,
                &component,
            );
            self.mix(
                b,
                size,
                &component,
                dst,
                [weight as f32; 3],
                [if i == 0 { 0. } else { 1. }; 3],
            );
        }
        self.buffers.push(component);
    }
}
pub(super) fn buffer(b: &WgpuBackend, size: [u32; 2]) -> wgpu::Buffer {
    b.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("spatial scratch"),
        size: size[0] as u64 * size[1] as u64 * 12,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
pub(super) fn halation(
    b: &WgpuBackend,
    p: &crate::HalationGpuParams,
    size: [u32; 2],
    src: &wgpu::Buffer,
    mid: &wgpu::Buffer,
) -> State {
    let mut state = State::new();
    let core = buffer(b, size);
    let tail = buffer(b, size);
    let sum = buffer(b, size);
    if p.scatter_amount > 0. {
        state.gaussian(b, size, p.scatter_core_px_rgb, src, mid, &core);
        state.exponential(b, size, p.scatter_tail_px_rgb, src, mid, &tail);
        state.mix(
            b,
            size,
            &core,
            &sum,
            p.scatter_tail_weight.map(|w| 1. - w),
            [0.; 3],
        );
        state.mix(b, size, &tail, &sum, p.scatter_tail_weight, [1.; 3]);
        let scatter = std::array::from_fn(|c| {
            if p.scatter_core_px_rgb[c] > 0. || p.scatter_tail_px_rgb[c] > 0. {
                p.scatter_amount
            } else {
                0.
            }
        });
        state.mix(b, size, &sum, src, scatter, scatter.map(|v| 1. - v));
    }
    if p.halation_n_bounces > 0 && p.halation_a_tot.iter().any(|v| *v > 0.) {
        let norm = (0..p.halation_n_bounces)
            .map(|i| (p.halation_bounce_decay as f64).powi(i as i32))
            .sum::<f64>();
        for i in 0..p.halation_n_bounces {
            state.gaussian(
                b,
                size,
                p.halation_first_sigma_px_rgb
                    .map(|s| (s as f64 * ((i + 1) as f64).sqrt()) as f32),
                src,
                mid,
                &core,
            );
            let weight = ((p.halation_bounce_decay as f64).powi(i as i32) / norm) as f32;
            state.mix(
                b,
                size,
                &core,
                &sum,
                [weight; 3],
                [if i == 0 { 0. } else { 1. }; 3],
            );
        }
        // Combine with a separate source to avoid binding the image for both read and write.
        state.mix(b, size, src, &tail, [1.; 3], [0.; 3]);
        state.mix(
            b,
            size,
            &sum,
            &tail,
            std::array::from_fn(|c| {
                if p.halation_first_sigma_px_rgb[c] > 0. {
                    p.halation_a_tot[c]
                } else {
                    0.
                }
            }),
            [1.; 3],
        );
        let inv = p.halation_a_tot.map(|v| {
            if p.halation_renormalize {
                1. / (1. + v)
            } else {
                1.
            }
        });
        state.mix(b, size, &tail, src, inv, [0.; 3]);
    }
    state.buffers.extend([core, tail, sum]);
    state
}

pub(super) fn gaussian_multi(b: &WgpuBackend, image: &ImageBuf, sigmas: &[f32]) -> Vec<ImageBuf> {
    let _gpu = crate::scheduler::gpu_permit();
    if sigmas.is_empty() {
        return vec![];
    }
    let size = [image.width, image.height];
    let bytes = image.pixel_count() as u64 * 12;
    let src = buffer(b, size);
    let mid = buffer(b, size);
    let values = scalars_to_f32(&image.data);
    b.queue
        .write_buffer(&src, 0, bytemuck::cast_slice(values.as_ref()));
    let outputs: Vec<_> = sigmas.iter().map(|_| buffer(b, size)).collect();
    let mut state = State::new();
    for (&s, dst) in sigmas.iter().zip(&outputs) {
        state.gaussian(b, size, [s; 3], &src, &mid, dst);
    }
    let readback = b.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("spatial readback"),
        size: bytes * sigmas.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = b.device.create_command_encoder(&Default::default());
    state.encode(&mut encoder);
    for (i, dst) in outputs.iter().enumerate() {
        encoder.copy_buffer_to_buffer(dst, 0, &readback, bytes * i as u64, bytes);
    }
    b.queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    b.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let values = slice.get_mapped_range();
    let data: &[f32] = bytemuck::cast_slice(&values);
    let result = data
        .chunks_exact(image.pixel_count() * 3)
        .map(|v| ImageBuf::from_data(image.width, image.height, f32_to_scalars(v.to_vec())))
        .collect();
    drop(values);
    readback.unmap();
    result
}
pub(super) fn apply(
    b: &WgpuBackend,
    image: &ImageBuf,
    build: impl FnOnce(&wgpu::Buffer, &wgpu::Buffer, [u32; 2]) -> State,
) -> ImageBuf {
    let _gpu = crate::scheduler::gpu_permit();
    let size = [image.width, image.height];
    let bytes = image.pixel_count() as u64 * 12;
    let src = buffer(b, size);
    let mid = buffer(b, size);
    let values = scalars_to_f32(&image.data);
    b.queue
        .write_buffer(&src, 0, bytemuck::cast_slice(values.as_ref()));
    let state = build(&src, &mid, size);
    let readback = b.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("spatial effect readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = b.device.create_command_encoder(&Default::default());
    state.encode(&mut encoder);
    encoder.copy_buffer_to_buffer(&src, 0, &readback, 0, bytes);
    b.queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    b.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let data = f32_to_scalars(bytemuck::cast_slice::<u8, f32>(&mapped).to_vec());
    drop(mapped);
    readback.unmap();
    ImageBuf::from_data(image.width, image.height, data)
}
pub(super) fn dir_diffusion(
    b: &WgpuBackend,
    image: &ImageBuf,
    core_sigma: f32,
    tail_lambda: f32,
    weight: f32,
) -> ImageBuf {
    apply(b, image, |src, mid, size| {
        let core = buffer(b, size);
        let tail = buffer(b, size);
        let mut state = State::new();
        state.gaussian(b, size, [core_sigma; 3], src, mid, &core);
        state.exponential(b, size, [tail_lambda; 3], src, mid, &tail);
        state.mix(b, size, &core, src, [1. - weight; 3], [0.; 3]);
        state.mix(b, size, &tail, src, [weight; 3], [1.; 3]);
        state.buffers.extend([core, tail]);
        state
    })
}
#[cfg(all(test, feature = "precision-f64"))]
mod tests {
    use super::*;
    use spektrafilm_math::precision::from_f64;
    mod python_gaussian {
        include!("../../spektrafilm-math/tests/support/gaussian_fixture.rs");
    }
    #[test]
    fn spatial_filters_match_independent_python_edges() {
        let Some(gpu) = WgpuBackend::new() else {
            return;
        };
        let image = python_gaussian::image();
        let fixture = python_gaussian::fixture();
        let sigmas: Vec<_> = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["sigma"].as_f64().unwrap() as f32)
            .collect();
        let outputs = gaussian_multi(&gpu, &image, &sigmas);
        for (output, case) in outputs.iter().zip(fixture["cases"].as_array().unwrap()) {
            let mut max = 0f64;
            for (point, expected) in fixture["points"]
                .as_array()
                .unwrap()
                .iter()
                .zip(case["expected"].as_array().unwrap())
            {
                let actual = output.get(
                    point[0].as_u64().unwrap() as u32,
                    point[1].as_u64().unwrap() as u32,
                );
                for c in 0..3 {
                    max = max.max((actual[c] - expected[c].as_f64().unwrap()).abs());
                }
            }
            eprintln!("GPU vs Python Gaussian sigma {}: max {max}", case["sigma"]);
            assert!(max < 2e-6, "GPU Gaussian differs from Python: {max}");
        }
    }
    #[test]
    fn spatial_pixels_cover_wrapped_dispatch_rows() {
        let Some(gpu) = WgpuBackend::new() else {
            return;
        };
        let image = ImageBuf::from_data(
            93,
            61,
            (0..93 * 61 * 3)
                .map(|i| from_f64((i % 233) as f64 / 233.))
                .collect(),
        );
        let output = apply(&gpu, &image, |src, mid, size| {
            let mut state = State::new();
            state.pixel_grid_limit = 1;
            state.gaussian(&gpu, size, [1.7; 3], src, mid, src);
            let scaled = buffer(&gpu, size);
            state.mix(&gpu, size, src, &scaled, [0.5; 3], [0.; 3]);
            state.mix(&gpu, size, &scaled, src, [2.; 3], [0.; 3]);
            state.buffers.push(scaled);
            state
        });
        let reference = spektrafilm_math::gaussian::gaussian_blur(&image, 1.7);
        let max = reference
            .data
            .iter()
            .zip(&output.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        assert!(max < 1e-6, "wrapped rows were skipped / repeated: {max}");
    }
    #[test]
    #[cfg(feature = "precision-f64")]
    fn fir_iir_match_cpu_on_edges_and_hdr() {
        let Some(gpu) = WgpuBackend::new() else {
            return;
        };
        let (w, h) = (192, 128);
        let image = ImageBuf::from_data(
            w,
            h,
            (0..h)
                .flat_map(|y| {
                    (0..w).flat_map(move |x| {
                        let v = if x == 0 || y == 0 {
                            1.
                        } else if x > 80 && y > 60 {
                            0.9
                        } else {
                            0.01
                        };
                        [v, v * 0.4, v * 0.1].map(from_f64)
                    })
                })
                .collect(),
        );
        let sigmas = [0., 0.7, 1.7, 2.99, 3., 4., 5.99, 10., 17.5, 32., 100.];
        let outputs = gaussian_multi(&gpu, &image, &sigmas);
        let mut worst = 0f64;
        for (&s, out) in sigmas.iter().zip(outputs) {
            let reference = spektrafilm_math::gaussian::gaussian_blur(&image, s);
            let max = reference
                .data
                .iter()
                .zip(&out.data)
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .fold(0f64, f64::max);
            eprintln!("sigma {s} max {max}");
            worst = worst.max(max);
        }
        assert!(worst < 0.000001, "drift {worst}");
    }
}
