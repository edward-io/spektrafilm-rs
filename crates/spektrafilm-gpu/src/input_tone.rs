//! Resident camera gain and tone preparation before spectral conversion.
use super::*;
use crate::InputToneGpuParams;
use spektrafilm_math::tone;

const SHADER: &str = include_str!("../../spektrafilm-shaders/wgsl/input_tone.wgsl");
/// Stage 5 of the shader: the per-pixel curve without a regional base.
const POINTWISE: u32 = 5;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_pixels: u32,
    width: u32,
    height: u32,
    radius: u32,
    stage: u32,
    pad: [u32; 3],
    controls: [f32; 4],
    luma_gain: [f32; 4],
    constants: [f32; 4],
}

impl Params {
    fn new(size: [u32; 2], tone: &InputToneGpuParams, stage: u32) -> Self {
        let luma = tone.curve.luma;
        Self {
            n_pixels: size[0] * size[1],
            width: size[0],
            height: size[1],
            radius: tone::region_radius(size[0], size[1]),
            stage,
            pad: [0; 3],
            controls: tone.curve.controls.map(|v| v as f32),
            luma_gain: [
                luma[0] as f32,
                luma[1] as f32,
                luma[2] as f32,
                tone.camera_gain as f32,
            ],
            constants: [
                tone::HIGHLIGHT_STOPS as f32,
                tone::DETAIL_EPS as f32,
                tone::FLOOR_STOPS as f32,
                0.,
            ],
        }
    }
}

pub(super) struct State {
    jobs: Vec<DispatchJob>,
    n_pixels: u32,
    _pairs: wgpu::Buffer,
}

impl State {
    pub(super) fn encode_passes(&self, encoder: &mut wgpu::CommandEncoder) {
        // One pass per stage: each reads what the previous stage wrote.
        for job in &self.jobs {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("input tone"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&job.pipeline.pipeline);
            pass.set_bind_group(0, &job.bg, &[]);
            pass.dispatch_workgroups(self.n_pixels.div_ceil(1024), 1, 1);
        }
    }
}

/// `scratch` must hold three floats per pixel and be unused until the front
/// pass; the resident chain lends its output buffer.
pub(super) fn build(
    b: &WgpuBackend,
    size: [u32; 2],
    tone: &InputToneGpuParams,
    img: &wgpu::Buffer,
    scratch: &wgpu::Buffer,
) -> State {
    let n_pixels = size[0] * size[1];
    let pairs = b.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("input tone pairs"),
        size: n_pixels as u64 * 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let stages = if tone.regional && tone.curve.is_regional() {
        0..5
    } else {
        POINTWISE..POINTWISE + 1
    };
    let storage = wgpu::BufferBindingType::Storage { read_only: false };
    let jobs = stages
        .map(|stage| {
            storage_job(
                b,
                SHADER,
                &Params::new(size, tone, stage),
                &[(img, storage), (scratch, storage), (&pairs, storage)],
            )
        })
        .collect();
    State {
        jobs,
        n_pixels,
        _pairs: pairs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spektrafilm_math::{image::ImageBuf, precision::Scalar, tone::ToneCurve, transfer};

    fn run(b: &WgpuBackend, image: &ImageBuf, tone: &InputToneGpuParams) -> Vec<f32> {
        let size = [image.width, image.height];
        let bytes = image.pixel_count() as u64 * 12;
        let img = spatial::buffer(b, size);
        let scratch = spatial::buffer(b, size);
        b.queue.write_buffer(
            &img,
            0,
            bytemuck::cast_slice(scalars_to_f32(&image.data).as_ref()),
        );
        let state = build(b, size, tone, &img, &scratch);
        let readback = b.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("input tone readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = b.device.create_command_encoder(&Default::default());
        state.encode_passes(&mut encoder);
        encoder.copy_buffer_to_buffer(&img, 0, &readback, 0, bytes);
        b.queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        b.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        readback.unmap();
        out
    }

    #[test]
    fn input_tone_shader_matches_cpu_with_camera_gain_and_primaries() {
        let Some(b) = WgpuBackend::new() else { return };
        // Gradient, checkerboard texture, a hard edge, colored and signed pixels.
        let (w, h) = (97u32, 61u32);
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let stops = -9.
                    + 13. * x as f32 / w as f32
                    + if (x + y) % 2 == 0 { 0.3 } else { -0.3 }
                    + if y > h / 2 { 2. } else { 0. };
                let v = 0.184 * stops.exp2();
                data.extend(match (x * 7 + y) % 23 {
                    0 => [0.02, 0.03, 0.6],
                    1 => [-0.1, 0.3, 0.6],
                    2 => [0.; 3],
                    _ => [v, v * 0.8, v * 0.65],
                });
            }
        }
        let image = ImageBuf::from_data(w, h, data.iter().map(|&v| v as Scalar).collect());
        for space in ["sRGB", "ACEScg", "ProPhoto RGB"] {
            for (controls, regional) in [
                ([100., 0., 0., 0.], true),
                ([0., -100., 100., 0.], true),
                ([0., 100., -100., 0.], true),
                ([100., 0., 0., -100.], true),
                ([60., -80., 40., -25.], false),
                ([0., 0., 0., 100.], true),
            ] {
                let tone = InputToneGpuParams {
                    curve: ToneCurve::new(controls, transfer::get(space).unwrap().to_xyz[1]),
                    camera_gain: 2.5,
                    regional,
                };
                let actual = run(&b, &image, &tone);
                let mut expected = image.clone();
                expected
                    .data
                    .iter_mut()
                    .for_each(|v| *v *= tone.camera_gain as Scalar);
                if regional {
                    tone.curve.apply(&mut expected);
                } else {
                    tone.curve.apply_pointwise(&mut expected);
                }
                for (i, (&a, &e)) in actual.iter().zip(&expected.data).enumerate() {
                    // Box sums accumulate in f32 on the GPU and f64 on the CPU.
                    let tolerance = 1e-6 + 2e-4 * (e as f64).abs();
                    assert!(
                        a.is_finite() && (a as f64 - e as f64).abs() <= tolerance,
                        "{space} {controls:?} value {i}: {a} vs {e}"
                    );
                }
            }
        }
    }
}
