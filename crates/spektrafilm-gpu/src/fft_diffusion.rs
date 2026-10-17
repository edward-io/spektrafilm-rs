//! Exact discrete PSF convolution on the GPU using reflected overlap-save FFTs.
//! Three complex work buffers are reused across channels and image tiles.
use super::*;
use wgpu::util::DeviceExt;
const READ: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: true };
const WRITE: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: false };
/// Bounded LRU of immutable, already-submitted kernel spectra. Exposure and
/// film settings do not enter the key: only the physical PSF and FFT geometry.
#[derive(Default)]
pub(super) struct Cache {
    entries: std::collections::VecDeque<(Vec<u32>, Vec<wgpu::Buffer>)>,
    hits: usize,
}
impl Cache {
    fn get(&mut self, key: &[u32]) -> Option<Vec<wgpu::Buffer>> {
        let i = self.entries.iter().position(|(k, _)| k == key)?;
        let entry = self.entries.remove(i)?;
        let buffers = entry.1.clone();
        self.entries.push_back(entry);
        self.hits += 1;
        Some(buffers)
    }
    fn insert(&mut self, key: Vec<u32>, buffers: Vec<wgpu::Buffer>) {
        const LIMIT: u64 = 1024 * 1024 * 1024;
        if buffers.iter().map(wgpu::Buffer::size).sum::<u64>() > LIMIT {
            return;
        }
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push_back((key, buffers));
        while self.entries.len() > 8
            || self
                .entries
                .iter()
                .flat_map(|(_, v)| v)
                .map(wgpu::Buffer::size)
                .sum::<u64>()
                > LIMIT
        {
            self.entries.pop_front();
        }
    }
}
enum Command {
    Compute(DispatchJob, [u32; 3]),
    Copy(usize, usize),
}
pub(super) struct State {
    commands: Vec<Command>,
    buffers: Vec<wgpu::Buffer>,
    output: wgpu::Buffer,
    bytes: u64,
}
impl State {
    fn encode_commands(&self, encoder: &mut wgpu::CommandEncoder) {
        for command in &self.commands {
            match command {
                Command::Compute(job, groups) => {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("accurate diffusion FFT"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&job.pipeline.pipeline);
                    pass.set_bind_group(0, &job.bg, &[]);
                    pass.dispatch_workgroups(groups[0], groups[1], 1);
                }
                Command::Copy(src, dst) => encoder.copy_buffer_to_buffer(
                    &self.buffers[*src],
                    0,
                    &self.buffers[*dst],
                    0,
                    self.buffers[*src].size(),
                ),
            }
        }
    }
    pub(super) fn encode_passes(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::Buffer) {
        self.encode_commands(encoder);
        encoder.copy_buffer_to_buffer(&self.output, 0, target, 0, self.bytes);
    }
    fn dispatch<T: bytemuck::Pod>(
        &mut self,
        b: &WgpuBackend,
        shader: &'static str,
        params: &T,
        bindings: &[(&wgpu::Buffer, wgpu::BufferBindingType)],
        n: u32,
    ) {
        let mut layout = vec![wgpu::BufferBindingType::Uniform];
        layout.extend(bindings.iter().map(|(_, ty)| *ty));
        let pipeline = b.cached_pipeline(shader, &layout);
        let uniform = b
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FFT params"),
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
            label: Some("FFT bindings"),
            layout: &pipeline.layout,
            entries: &entries,
        });
        let groups = n.div_ceil(256);
        let x = groups.min(32768);
        self.commands.push(Command::Compute(
            DispatchJob {
                _params_buf: uniform,
                pipeline,
                bg,
            },
            [x, groups.div_ceil(x), 1],
        ));
    }
    fn transform(
        &mut self,
        b: &WgpuBackend,
        shape: [u32; 2],
        ping: &wgpu::Buffer,
        pong: &wgpu::Buffer,
        twiddles: &wgpu::Buffer,
        mut current: usize,
        inverse: bool,
    ) -> usize {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            shape: [u32; 4],
            direction: [u32; 4],
        }
        let max = shape[0].max(shape[1]);
        for axis in 0..2 {
            let mut span = 2;
            while span <= shape[axis] {
                let (src, dst) = if current == 0 {
                    (ping, pong)
                } else {
                    (pong, ping)
                };
                self.dispatch(
                    b,
                    include_str!("../../spektrafilm-shaders/wgsl/diffusion_fft_stage.wgsl"),
                    &Params {
                        shape: [shape[0], shape[1], span, max],
                        direction: [axis as u32, u32::from(inverse), 0, 0],
                    },
                    &[(src, READ), (dst, WRITE), (twiddles, READ)],
                    shape[0] * shape[1] / 2,
                );
                current = 1 - current;
                span *= 2;
            }
        }
        current
    }
}
/// Working FFT dimensions and output tile sizes. Fall back accurately if the
/// adapter cannot bind the transform needed by this physical kernel radius.
pub(super) fn dimensions(b: &WgpuBackend, size: [u32; 2], radius: u32) -> Option<[u32; 2]> {
    let min = (2 * radius + 1).next_power_of_two();
    let mut shape = size.map(|n| (n + 2 * radius).next_power_of_two().min(8192).max(min));
    let limit = b.device.limits().max_storage_buffer_binding_size as u64;
    while shape[0] as u64 * shape[1] as u64 * 8 > limit {
        let axis = usize::from(shape[1] > shape[0]);
        if shape[axis] / 2 < min {
            return None;
        }
        shape[axis] /= 2;
    }
    Some(shape)
}
pub(super) fn build(
    b: &WgpuBackend,
    plan: &crate::DiffusionGpuPlan,
    size: [u32; 2],
    image: &wgpu::Buffer,
) -> State {
    let shape = dimensions(b, size, (plan.psf_side - 1) / 2).expect("validated FFT dimensions");
    build_at_shape(b, plan, size, image, shape)
}
fn build_at_shape(
    b: &WgpuBackend,
    plan: &crate::DiffusionGpuPlan,
    size: [u32; 2],
    image: &wgpu::Buffer,
    shape: [u32; 2],
) -> State {
    let r = (plan.psf_side - 1) / 2;
    let n = shape[0] * shape[1];
    let bytes = n as u64 * 8;
    let make = || {
        b.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT work"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let ping = make();
    let pong = make();
    let mut key = vec![shape[0], shape[1], plan.psf_side];
    key.extend(plan.radial_psf.iter().flatten().map(|v| v.to_bits()));
    let cached = b.diffusion_cache.lock().unwrap().get(&key);
    let spectra = cached
        .clone()
        .unwrap_or_else(|| (0..3).map(|_| make()).collect());
    let output = spatial::buffer(b, size);
    let max = shape[0].max(shape[1]);
    let twiddles: Vec<[f32; 2]> = (0..max / 2)
        .map(|i| {
            let theta = -std::f64::consts::TAU * i as f64 / max as f64;
            [theta.cos() as f32, theta.sin() as f32]
        })
        .collect();
    let twiddles = b
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFT twiddles"),
            contents: bytemuck::cast_slice(&twiddles),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let psf = b
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("exact diffusion PSF"),
            contents: bytemuck::cast_slice(&plan.radial_psf),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let mut state = State {
        commands: vec![],
        buffers: vec![],
        output: spatial::buffer(b, [1, 1]),
        bytes: size[0] as u64 * size[1] as u64 * 12,
    };
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Init {
        shape: [u32; 4],
        frame: [u32; 4],
        tile: [u32; 4],
    }
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Extract {
        shape: [u32; 4],
        frame: [u32; 4],
        tile: [u32; 4],
        mix: [f32; 4],
    }
    let header = [shape[0], shape[1], shape[0].ilog2(), shape[1].ilog2()];
    let tile = [shape[0] - 2 * r, shape[1] - 2 * r];
    if cached.is_none() {
        for c in 0..3 {
            state.dispatch(
                b,
                include_str!("../../spektrafilm-shaders/wgsl/diffusion_fft_init.wgsl"),
                &Init {
                    shape: header,
                    frame: [size[0], size[1], r, c],
                    tile: [0, 0, plan.radial_psf.len() as u32, 1],
                },
                &[(&psf, READ), (&ping, WRITE)],
                n,
            );
            let result = state.transform(b, shape, &ping, &pong, &twiddles, 0, false);
            state.commands.push(Command::Copy(result, 2 + c as usize));
        }
        state.buffers = vec![ping.clone(), pong.clone()];
        state.buffers.extend(spectra.iter().cloned());
        let mut encoder = b.device.create_command_encoder(&Default::default());
        state.encode_commands(&mut encoder);
        // Submit initialization before publishing it. Other render workers can
        // use the cache immediately: this device queue preserves submit order.
        b.queue.submit(Some(encoder.finish()));
        state.commands.clear();
        b.diffusion_cache
            .lock()
            .unwrap()
            .insert(key, spectra.clone());
    }
    for c in 0..3 {
        let spectrum = &spectra[c as usize];
        for y in (0..size[1]).step_by(tile[1] as usize) {
            for x in (0..size[0]).step_by(tile[0] as usize) {
                state.dispatch(
                    b,
                    include_str!("../../spektrafilm-shaders/wgsl/diffusion_fft_init.wgsl"),
                    &Init {
                        shape: header,
                        frame: [size[0], size[1], r, c],
                        tile: [x, y, 0, 0],
                    },
                    &[(image, READ), (&ping, WRITE)],
                    n,
                );
                let result = state.transform(b, shape, &ping, &pong, &twiddles, 0, false);
                let (input, product) = if result == 0 {
                    (&ping, &pong)
                } else {
                    (&pong, &ping)
                };
                state.dispatch(
                    b,
                    include_str!("../../spektrafilm-shaders/wgsl/diffusion_fft_multiply.wgsl"),
                    &header,
                    &[(input, READ), (spectrum, READ), (product, WRITE)],
                    n,
                );
                let result = state.transform(b, shape, &ping, &pong, &twiddles, 1 - result, true);
                let convolution = if result == 0 { &ping } else { &pong };
                state.dispatch(
                    b,
                    include_str!("../../spektrafilm-shaders/wgsl/diffusion_fft_extract.wgsl"),
                    &Extract {
                        shape: header,
                        frame: [size[0], size[1], r, c],
                        tile: [x, y, tile[0].min(size[0] - x), tile[1].min(size[1] - y)],
                        mix: [plan.p_s, 0., 0., 0.],
                    },
                    &[(convolution, READ), (image, READ), (&output, WRITE)],
                    tile[0].min(size[0] - x) * tile[1].min(size[1] - y),
                );
            }
        }
    }
    state.buffers = vec![ping, pong, twiddles, psf];
    state.buffers.extend(spectra);
    state.output = output;
    state
}

pub(super) fn apply(b: &WgpuBackend, image: &ImageBuf, plan: &crate::DiffusionGpuPlan) -> ImageBuf {
    let size = [image.width, image.height];
    let bytes = image.pixel_count() as u64 * 12;
    let source = spatial::buffer(b, size);
    let values = scalars_to_f32(&image.data);
    b.queue
        .write_buffer(&source, 0, bytemuck::cast_slice(values.as_ref()));
    let state = build(b, plan, size, &source);
    let readback = b.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("diffusion readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = b.device.create_command_encoder(&Default::default());
    state.encode_passes(&mut encoder, &source);
    encoder.copy_buffer_to_buffer(&source, 0, &readback, 0, bytes);
    b.queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    b.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let data = f32_to_scalars(bytemuck::cast_slice::<_, f32>(&mapped).to_vec());
    drop(mapped);
    readback.unmap();
    ImageBuf::from_data(image.width, image.height, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlap_save_matches_reflected_fft_across_tile_boundaries() {
        let Some(gpu) = WgpuBackend::new() else {
            return;
        };
        let size = [93, 61];
        let k = 15usize;
        let components = vec![
            [0.7, 0.4, 0.1, 0.05],
            [4., 0.2, 0.6, 0.2],
            [20., 0.4, 0.3, 0.75],
        ];
        let plan = crate::DiffusionGpuPlan {
            psf_side: k as u32,
            radial_psf: components.clone(),
            d: 1,
            small_w: size[0],
            small_h: size[1],
            p_s: 0.6,
            sigmas: vec![],
            coeffs: vec![],
        };
        let image: Vec<f32> = (0..size[1])
            .flat_map(|y| {
                (0..size[0]).flat_map(move |x| {
                    if x == 0 || y == 0 {
                        [4., 1., 0.2]
                    } else if (x / 5 + y / 7) % 2 == 0 {
                        [0.02, 0.04, 0.1]
                    } else {
                        [0.8, 0.2, 0.6]
                    }
                })
            })
            .collect();
        let buffer = spatial::buffer(&gpu, size);
        gpu.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(&image));
        let first = build_at_shape(&gpu, &plan, size, &buffer, [64, 64]);
        let state = build_at_shape(&gpu, &plan, size, &buffer, [64, 64]);
        assert_eq!(gpu.diffusion_cache.lock().unwrap().hits, 1);
        drop(first);
        let bytes = image.len() as u64 * 4;
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        state.encode_passes(&mut encoder, &buffer);
        encoder.copy_buffer_to_buffer(&buffer, 0, &readback, 0, bytes);
        gpu.queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let output: &[f32] = bytemuck::cast_slice(&mapped);
        let mut worst = 0f64;
        for c in 0..3 {
            let mut psf: Vec<f64> = (0..k * k)
                .map(|i| {
                    let x = (i % k) as f64 - (k / 2) as f64;
                    let y = (i / k) as f64 - (k / 2) as f64;
                    let r = x.hypot(y);
                    components
                        .iter()
                        .map(|p| {
                            p[c + 1] as f64 * (-r / p[0] as f64).exp()
                                / (std::f64::consts::TAU * (p[0] as f64).powi(2))
                        })
                        .sum()
                })
                .collect();
            let sum = psf.iter().sum::<f64>();
            for v in &mut psf {
                *v /= sum;
            }
            let source: Vec<_> = image.chunks_exact(3).map(|p| p[c] as f64).collect();
            let reference = spektrafilm_math::fft_conv::convolve2d_reflect(
                &source,
                size[1] as usize,
                size[0] as usize,
                &psf,
                k,
            );
            for (i, blurred) in reference.iter().enumerate() {
                let expected = (1. - plan.p_s as f64) * source[i] + plan.p_s as f64 * blurred;
                worst = worst.max((output[i * 3 + c] as f64 - expected).abs());
            }
        }
        eprintln!("overlap-save max error {worst}");
        assert!(
            worst < 0.00001,
            "FFT seam / reflection / normalization error {worst}"
        );
    }
}
