//! GPU schedule for the scene-linear chroma filter used by the editor preview.
use super::*;
use spektrafilm_math::transfer::Transfer;
use wgpu::{Buffer, BufferBindingType, BufferUsages, util::DeviceExt};

const SHADER: &str = include_str!("../../spektrafilm-shaders/wgsl/chroma_preview.wgsl");
const READ: BufferBindingType = BufferBindingType::Storage { read_only: true };
const WRITE: BufferBindingType = BufferBindingType::Storage { read_only: false };
const X: [i32; 2] = [1, 0];
const Y: [i32; 2] = [0, 1];

#[repr(u32)]
#[derive(Clone, Copy)]
enum Pass {
    Luminance,
    Gaussian,
    Normalize,
    Residual,
    Median,
    Guide,
    Bilateral,
    Edge,
    Score,
    DilateSquare,
    Protect,
    Repair,
    Reconstruct,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    size: [u32; 2],
    operation: u32,
    radius: i32,
    direction: [i32; 2],
    step: i32,
    _pad: u32,
    values: [f32; 4],
    weights: [[f32; 4]; 9],
}

impl Params {
    fn kernel(sigma: f32, radius: i32, step: i32, normalize: bool) -> Self {
        let mut p = Self {
            radius,
            step,
            ..Self::default()
        };
        let weights = p.weights.as_flattened_mut();
        let count = (2 * radius / step + 1) as usize;
        for (i, weight) in weights[..count].iter_mut().enumerate() {
            let offset = i as i32 * step - radius;
            *weight = (-0.5 * (offset as f32 / sigma).powi(2)).exp();
        }
        if normalize {
            let sum: f32 = weights[..count].iter().sum();
            weights[..count].iter_mut().for_each(|w| *w /= sum);
        }
        p
    }
}

struct Schedule<'a> {
    backend: &'a WgpuBackend,
    jobs: spatial::State,
    size: [u32; 2],
    rgb: &'a Buffer,
    unused_read: &'a Buffer,
    unused_write: &'a Buffer,
    scratch: &'a Buffer,
}

impl Schedule<'_> {
    fn pass(&mut self, op: Pass, mut p: Params, input: &[&Buffer], output: Option<&Buffer>) {
        p.size = self.size;
        p.operation = op as u32;
        let reads: [&Buffer; 3] =
            std::array::from_fn(|i| input.get(i).copied().unwrap_or(self.unused_read));
        self.jobs.dispatch(
            self.backend,
            SHADER,
            &p,
            &[
                (reads[0], READ),
                (reads[1], READ),
                (reads[2], READ),
                (output.unwrap_or(self.unused_write), WRITE),
                (self.rgb, WRITE),
            ],
            [self.size[0].div_ceil(16), self.size[1].div_ceil(8), 1],
        );
    }

    fn point(&mut self, op: Pass, values: [f32; 4], input: &[&Buffer], output: Option<&Buffer>) {
        self.pass(
            op,
            Params {
                values,
                ..Params::default()
            },
            input,
            output,
        );
    }

    fn separable(
        &mut self,
        op: Pass,
        mut p: Params,
        input: &Buffer,
        guide: Option<&Buffer>,
        output: &Buffer,
    ) {
        for (direction, src, dst) in [(X, input, self.scratch), (Y, self.scratch, output)] {
            p.direction = direction;
            self.pass(op, p, &[src, guide.unwrap_or(self.unused_read)], Some(dst));
        }
    }

    fn gaussian(&mut self, src: &Buffer, dst: &Buffer, sigma: f32) {
        self.separable(
            Pass::Gaussian,
            Params::kernel(sigma, (4. * sigma).round() as i32, 1, true),
            src,
            None,
            dst,
        );
    }

    fn bilateral(&mut self, src: &Buffer, guide: &Buffer, dst: &Buffer, sigma: f32, step: i32) {
        let radius = (2.6 * sigma / step as f32).ceil() as i32 * step;
        self.separable(
            Pass::Bilateral,
            Params::kernel(sigma, radius, step, false),
            src,
            Some(guide),
            dst,
        );
    }

    fn encode(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.jobs.encode(encoder);
        self.jobs = spatial::State::new();
    }
}

impl WgpuBackend {
    pub(super) fn chroma_preview(
        &self,
        image: &ImageBuf,
        luma: [f64; 3],
        transfer: Transfer,
    ) -> Option<ImageBuf> {
        let pixels = image.pixel_count();
        let plane_bytes = pixels as u64 * 16;
        let limits = self.device.limits();
        if cfg!(feature = "precision-f64")
            || image.width < 2
            || image.height < 2
            || pixels > 4 * 1024 * 1024
            || plane_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || plane_bytes > limits.max_buffer_size
            || limits.max_storage_buffers_per_shader_stage < 5
            || image.width.div_ceil(16) > limits.max_compute_workgroups_per_dimension
            || image.height.div_ceil(8) > limits.max_compute_workgroups_per_dimension
            || luma[1] <= 0.
        {
            return None;
        }
        let sum: f64 = luma.iter().sum();
        if !sum.is_finite() || sum <= 0. {
            return None;
        }
        let luma = luma.map(|v| (v / sum) as f32);
        let color = [luma[0], luma[1], luma[2], 0.];
        let original = scalars_to_f32(&image.data);
        let linear = if transfer == Transfer::Linear {
            Cow::Borrowed(original.as_ref())
        } else {
            use rayon::prelude::*;
            let _cpu = crate::scheduler::cpu_permit().ok()?;
            Cow::Owned(
                original
                    .par_iter()
                    .map(|&v| transfer.decode(f64::from(v)) as f32)
                    .collect(),
            )
        };
        // Shader fast math cannot be relied on to handle invalid or extreme samples.
        if !linear.iter().all(|v| v.abs() <= 1e6) {
            return None;
        }

        let mut values = self.filter_linear_chroma([image.width, image.height], &linear, color)?;
        if transfer != Transfer::Linear {
            use rayon::prelude::*;
            let _cpu = crate::scheduler::cpu_permit().ok()?;
            values
                .par_iter_mut()
                .zip(linear.par_iter())
                .zip(original.par_iter())
                .for_each(|((out, &input), &encoded)| {
                    *out = if *out == input {
                        encoded
                    } else {
                        transfer.encode(f64::from(*out)) as f32
                    };
                });
        }
        Some(ImageBuf::from_data(
            image.width,
            image.height,
            f32_to_scalars(values),
        ))
    }

    fn filter_linear_chroma(
        &self,
        size: [u32; 2],
        linear: &[f32],
        color: [f32; 4],
    ) -> Option<Vec<f32>> {
        let plane_bytes = u64::from(size[0]) * u64::from(size[1]) * 16;
        let mut turn = GpuTurn::new(crate::scheduler::gpu_permit_cancellable().ok()?);
        let buffer = |label, size, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let plane = |label| buffer(label, plane_bytes, BufferUsages::STORAGE);
        let bytes = linear.len() as u64 * 4;
        let direct_map = self
            .device
            .features()
            .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        let rgb = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("chroma preview RGB"),
                contents: bytemuck::cast_slice(linear),
                usage: BufferUsages::STORAGE
                    | if direct_map {
                        BufferUsages::MAP_READ
                    } else {
                        BufferUsages::COPY_SRC
                    },
            });
        let unused_read = buffer("chroma unused read", 16, BufferUsages::STORAGE);
        let unused_write = buffer("chroma unused write", 16, BufferUsages::STORAGE);
        let scratch = plane("chroma separable scratch");
        let lum = plane("chroma luminance");
        let local = plane("chroma local luminance");
        let chroma = plane("chroma normalized");
        let noise = plane("chroma noise");
        let fine = plane("chroma fine");
        let mid = plane("chroma middle");
        let protected = plane("chroma protection");
        let guide = plane("chroma guide");
        let a = plane("chroma scratch A");
        let b = plane("chroma scratch B");
        let mut schedule = Schedule {
            backend: self,
            jobs: spatial::State::new(),
            size,
            rgb: &rgb,
            unused_read: &unused_read,
            unused_write: &unused_write,
            scratch: &scratch,
        };
        let mut encoder = self.device.create_command_encoder(&Default::default());
        schedule.point(Pass::Luminance, color, &[], Some(&lum));
        schedule.gaussian(&lum, &local, 0.95);
        schedule.point(Pass::Normalize, [0.; 4], &[&lum, &local], Some(&chroma));
        schedule.gaussian(&chroma, &a, 0.9);
        schedule.point(Pass::Residual, [0.; 4], &[&chroma, &a], Some(&b));
        schedule.separable(Pass::Median, Params::default(), &b, None, &a);
        schedule.gaussian(&a, &noise, 4.);
        schedule.encode(&mut encoder);
        self.yield_gpu(&mut encoder, &mut turn).ok()?;

        schedule.gaussian(&chroma, &a, 1.35);
        schedule.point(
            Pass::Guide,
            [0.34, 0.20, 0., 0.],
            &[&local, &a],
            Some(&guide),
        );
        schedule.bilateral(&chroma, &guide, &fine, 2.5, 1);
        schedule.gaussian(&lum, &a, 2.2);
        schedule.gaussian(&fine, &b, 1.5);
        schedule.point(Pass::Guide, [0.40, 0.24, 0., 0.], &[&a, &b], Some(&guide));
        schedule.bilateral(&fine, &guide, &mid, 11., 3);
        schedule.encode(&mut encoder);
        self.yield_gpu(&mut encoder, &mut turn).ok()?;

        for (i, [dx, dy]) in [
            [1, 0],
            [0, 1],
            [1, 1],
            [-1, 1],
            [2, 1],
            [-2, 1],
            [1, 2],
            [-1, 2],
        ]
        .into_iter()
        .enumerate()
        {
            let length = (dx as f32).hypot(dy as f32);
            let mut tangent =
                Params::kernel(5.2 / length, (2.5 * 5.2 / length).ceil() as i32, 1, true);
            tangent.direction = [dx, dy];
            schedule.pass(Pass::Gaussian, tangent, &[&chroma], Some(&a));
            schedule.point(
                Pass::Edge,
                [-dy as f32 / length, dx as f32 / length, 0., 0.],
                &[&a],
                Some(&b),
            );
            schedule.gaussian(&b, &a, 3.);
            schedule.point(Pass::Score, [0.; 4], &[&b, &a, &noise], Some(&guide));
            schedule.point(Pass::DilateSquare, [0.; 4], &[&guide], Some(&a));
            schedule.pass(
                Pass::Protect,
                Params {
                    radius: (8. / length).ceil() as i32,
                    direction: [-dy, dx],
                    ..Params::default()
                },
                &[&a, &b],
                Some(&protected),
            );
            if i % 2 == 1 {
                schedule.encode(&mut encoder);
                self.yield_gpu(&mut encoder, &mut turn).ok()?;
            }
        }

        schedule.gaussian(&lum, &a, 3.);
        schedule.gaussian(&fine, &b, 4.);
        schedule.point(
            Pass::Guide,
            [0.72, 0.53, 1., 0.],
            &[&a, &b, &protected],
            Some(&guide),
        );
        schedule.bilateral(&mid, &guide, &b, 32., 7);
        schedule.point(Pass::Repair, [0.; 4], &[&mid, &b, &protected], Some(&a));
        schedule.point(Pass::Reconstruct, color, &[&a, &local, &noise], None);
        schedule.encode(&mut encoder);

        let staging = (!direct_map).then(|| {
            let out = buffer(
                "chroma preview readback",
                bytes,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            );
            encoder.copy_buffer_to_buffer(&rgb, 0, &out, 0, bytes);
            out
        });
        let readback = staging.as_ref().unwrap_or(&rgb);
        self.queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().ok()?.ok()?;
        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        readback.unmap();
        Some(values)
    }
}

#[cfg(all(test, not(feature = "precision-f64")))]
mod tests {
    use super::*;

    #[test]
    fn gpu_matches_chroma_reference_and_retains_hdr() {
        let Some(gpu) = WgpuBackend::new() else {
            return;
        };
        let luma = [0.2126, 0.7152, 0.0722];
        let (w, h) = (97, 89);
        let mut seed = 217u32;
        let mut pixels = Vec::new();
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let base = if (0..4).contains(&(x - y)) {
                    [0.165, 0.083, 0.138]
                } else {
                    [0.13, 0.10, 0.075]
                };
                for v in base {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    pixels.push(v + ((seed >> 8) as f32 / 16777216. - 0.5) * 0.04);
                }
            }
        }
        pixels[0..6].copy_from_slice(&[2.4, 1.7, 1.3, -0.02, 0.12, 0.15]);
        for transfer in [Transfer::Linear, Transfer::Srgb] {
            let image = ImageBuf::from_data(
                w,
                h,
                pixels
                    .iter()
                    .map(|&v| transfer.encode(f64::from(v)) as f32)
                    .collect(),
            );
            let cpu = spektrafilm_math::chroma::filter_full(&image, luma, transfer);
            let actual = gpu
                .chroma_preview(&image, luma, transfer)
                .expect("WGPU denoise");
            assert_ne!(cpu.data, image.data);
            assert_eq!(&actual.data[3..6], &image.data[3..6]);
            assert!(actual.data[0] > 1.);
            let max_error = actual
                .data
                .iter()
                .zip(&cpu.data)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0., f32::max);
            assert!(max_error < 2e-5, "max error {max_error} in {transfer:?}");
        }
    }
}
