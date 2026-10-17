//! GPU-resident physical emulsion layers, dye clouds and microstructure.
use super::*;
use crate::LayeredGrainGpuParams;
use wgpu::util::DeviceExt;
const READ: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: true };
const WRITE: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: false };
pub(super) struct State {
    jobs: Vec<Step>,
    _buffers: Vec<wgpu::Buffer>,
}
enum Step {
    Compute(DispatchJob),
    Spatial(spatial::State),
}
impl State {
    pub(super) fn encode(&self, encoder: &mut wgpu::CommandEncoder, n: u32) {
        for step in &self.jobs {
            match step {
                Step::Spatial(state) => state.encode(encoder),
                Step::Compute(job) => {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("layered_grain"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&job.pipeline.pipeline);
                    pass.set_bind_group(0, &job.bg, &[]);
                    pass.dispatch_workgroups(n.div_ceil(1024), 1, 1);
                }
            }
        }
    }
}
fn job<T: bytemuck::Pod>(
    b: &WgpuBackend,
    shader: &'static str,
    params: &T,
    bindings: &[(&wgpu::Buffer, wgpu::BufferBindingType)],
) -> Step {
    let mut layout = vec![wgpu::BufferBindingType::Uniform];
    layout.extend(bindings.iter().map(|(_, ty)| *ty));
    let pipeline = b.cached_pipeline(shader, &layout);
    let uniform = b
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("layered_grain_params"),
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
        label: Some("layered_grain_bg"),
        layout: &pipeline.layout,
        entries: &entries,
    });
    Step::Compute(DispatchJob {
        _params_buf: uniform,
        pipeline,
        bg,
    })
}
fn buffer(b: &WgpuBackend, bytes: u64) -> wgpu::Buffer {
    b.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layered_grain_scratch"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn vec4(v: [f32; 3]) -> [f32; 4] {
    [v[0], v[1], v[2], 0.]
}
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SampleParams {
    header: [u32; 4],
    min: [f32; 4],
    max: [f32; 4],
    particles: [f32; 4],
    uniformity: [f32; 4],
}
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FinishParams {
    header: [u32; 4],
    value: [f32; 4],
}
fn blur(
    state: &mut State,
    b: &WgpuBackend,
    size: [u32; 2],
    sigmas: [f32; 3],
    src: &wgpu::Buffer,
    mid: &wgpu::Buffer,
) {
    // The same FIR / compensated IIR recurrence as the CPU model, including
    // broad clouds. A truncated FIR changed the grain spectrum at sigma >= 3.
    let mut filter = spatial::State::new();
    filter.gaussian(b, size, sigmas, src, mid, src);
    state.jobs.push(Step::Spatial(filter));
}
pub(super) fn build(
    b: &WgpuBackend,
    p: &LayeredGrainGpuParams,
    size: [u32; 2],
    density: &wgpu::Buffer,
    scratch: &wgpu::Buffer,
) -> State {
    build_with_draws(b, p, size, density, scratch, None)
}
// Private replay path: identical sampled counts / clumping can be supplied by
// a reference fixture to separate numerical error from different RNG streams.
fn build_with_draws(
    b: &WgpuBackend,
    p: &LayeredGrainGpuParams,
    size: [u32; 2],
    density: &wgpu::Buffer,
    scratch: &wgpu::Buffer,
    draws: Option<&wgpu::Buffer>,
) -> State {
    let n = size[0] * size[1];
    let bytes = n as u64 * 12;
    let layer = buffer(b, bytes);
    let sum = buffer(b, bytes);
    let curves = b
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("layered_grain_curves"),
            contents: bytemuck::cast_slice(&p.curves),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let mut state = State {
        jobs: vec![],
        _buffers: vec![],
    };
    for sl in 0..3 {
        let l = &p.layers[sl];
        let params = SampleParams {
            header: [
                n,
                p.curves.len() as u32 / 3,
                sl as u32,
                p.positive as u32
                    + 2 * p.monochrome as u32
                    + 4 * p.fast_stats as u32
                    + 8 * draws.is_some() as u32,
            ],
            min: vec4(l.density_min),
            max: vec4(l.density_max),
            particles: vec4(l.particles),
            uniformity: vec4(p.uniformity),
        };
        state.jobs.push(job(
            b,
            concat!(
                include_str!("../../spektrafilm-shaders/wgsl/particle_sampling.wgsl"),
                include_str!("../../spektrafilm-shaders/wgsl/layered_grain.wgsl")
            ),
            &params,
            &[
                (density, READ),
                (&curves, READ),
                (&layer, WRITE),
                (draws.unwrap_or(&curves), READ),
            ],
        ));
        if l.cloud_sigma.iter().any(|s| *s > 0.) {
            blur(&mut state, b, size, l.cloud_sigma, &layer, scratch);
        }
        state.jobs.push(job(
            b,
            include_str!("../../spektrafilm-shaders/wgsl/add_scaled_per_channel.wgsl"),
            &FinishParams {
                header: [n, 0, 0, 0],
                value: [1., 1., 1., 0.],
            },
            &[(&layer, READ), (&sum, WRITE)],
        ));
    }
    // Match the reference physical model's explicit activation cutoffs.
    let micro = p.micro_sigma > 0.05;
    if micro {
        let variance = (p.micro_sigma as f64).powi(2).ln_1p() as f32;
        state.jobs.push(job(
            b,
            include_str!("../../spektrafilm-shaders/wgsl/grain_microstructure.wgsl"),
            &FinishParams {
                header: [n, p.monochrome as u32, draws.is_some() as u32, n * 9],
                value: [variance, 0., 0., 0.],
            },
            &[(&layer, WRITE), (draws.unwrap_or(&curves), READ)],
        ));
        if p.micro_blur > 0.4 {
            blur(&mut state, b, size, [p.micro_blur; 3], &layer, scratch);
        }
    }
    state.jobs.push(job(
        b,
        include_str!("../../spektrafilm-shaders/wgsl/grain_finish.wgsl"),
        &FinishParams {
            header: [n, micro as u32, 0, 0],
            value: vec4(p.density_min),
        },
        &[(&sum, READ), (&layer, READ), (density, WRITE)],
    ));
    if p.blur > 0. {
        blur(&mut state, b, size, [p.blur; 3], density, scratch);
    }
    state._buffers.extend([layer, sum, curves]);
    state
}
pub(super) fn apply(b: &WgpuBackend, image: &ImageBuf, p: &LayeredGrainGpuParams) -> ImageBuf {
    apply_with_draws(b, image, p, None)
}
fn apply_with_draws(
    b: &WgpuBackend,
    image: &ImageBuf,
    p: &LayeredGrainGpuParams,
    draws: Option<&wgpu::Buffer>,
) -> ImageBuf {
    let input = scalars_to_f32(&image.data);
    let n = image.pixel_count() as u32;
    let bytes = n as u64 * 12;
    let density = b
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("layered_grain_input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let scratch = buffer(b, bytes);
    let state = build_with_draws(b, p, [image.width, image.height], &density, &scratch, draws);
    let readback = b.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("layered_grain_readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = b.device.create_command_encoder(&Default::default());
    state.encode(&mut encoder, n);
    encoder.copy_buffer_to_buffer(&density, 0, &readback, 0, bytes);
    b.queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    b.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let output = f32_to_scalars(bytemuck::cast_slice::<u8, f32>(&data).to_vec());
    drop(data);
    readback.unmap();
    ImageBuf::from_data(image.width, image.height, output)
}

#[cfg(all(test, feature = "precision-f64"))]
mod tests {
    use super::*;
    #[test]
    fn coupled_draws_match_unmodified_python_grain_processing() {
        let Some(gpu) = WgpuBackend::new() else {
            return;
        };
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/grain_replay.json")).unwrap();
        let size: [u32; 2] = serde_json::from_value(fixture["size"].clone()).unwrap();
        let density: Vec<f64> = serde_json::from_value(fixture["density"].clone()).unwrap();
        let image = ImageBuf::from_data(size[0], size[1], density);
        let draws: Vec<f32> = serde_json::from_value(fixture["draws"].clone()).unwrap();
        let samples = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("coupled reference grain draws"),
                contents: bytemuck::cast_slice(&draws),
                usage: wgpu::BufferUsages::STORAGE,
            });
        for case in fixture["cases"].as_array().unwrap() {
            let a3 = |v: serde_json::Value| serde_json::from_value::<[f32; 3]>(v).unwrap();
            let p = LayeredGrainGpuParams {
                curves: serde_json::from_value(case["curves"].clone()).unwrap(),
                layers: std::array::from_fn(|i| {
                    let l = &case["layers"][i];
                    crate::LayerParticleGpuParams {
                        density_min: a3(l["density_min"].clone()),
                        density_max: a3(l["density_max"].clone()),
                        particles: a3(l["particles"].clone()),
                        cloud_sigma: a3(l["cloud_sigma"].clone()),
                    }
                }),
                uniformity: a3(case["uniformity"].clone()),
                density_min: a3(case["density_min"].clone()),
                blur: case["blur"].as_f64().unwrap() as f32,
                micro_sigma: case["micro_sigma"].as_f64().unwrap() as f32,
                micro_blur: case["micro_blur"].as_f64().unwrap() as f32,
                positive: case["positive"].as_bool().unwrap(),
                monochrome: false,
                fast_stats: false,
            };
            let output = apply_with_draws(&gpu, &image, &p, Some(&samples));
            let expected: Vec<f64> = serde_json::from_value(case["expected"].clone()).unwrap();
            let max = expected
                .iter()
                .zip(&output.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0f64, f64::max);
            eprintln!(
                "coupled grain positive {}, cloud {:?}, micro blur {}, post {}: max {max}",
                p.positive, p.layers[0].cloud_sigma, p.micro_blur, p.blur
            );
            assert!(
                max <= 0.001,
                "coupled grain exceeded 0.1% full scale: {max}"
            );
        }
    }
    #[test]
    fn exact_particle_sampling_preserves_poisson_law_across_preview_scales() {
        let Some(gpu) = WgpuBackend::new() else {
            return;
        };
        let image = ImageBuf::from_data(512, 512, vec![0.3; 512 * 512 * 3]);
        for npp in [1., 10., 100., 10_000., 1_000_000_000.] {
            let p = crate::LayeredGrainGpuParams {
                curves: vec![
                    [0.; 4],
                    [0.; 4],
                    [0.; 4],
                    [1., 1., 0., 0.],
                    [1., 1., 0., 0.],
                    [1., 1., 0., 0.],
                ],
                layers: std::array::from_fn(|l| crate::LayerParticleGpuParams {
                    density_min: [0.; 3],
                    density_max: [if l == 0 { 1. } else { 0. }; 3],
                    particles: [if l == 0 { npp } else { 0. }; 3],
                    cloud_sigma: [0.; 3],
                }),
                uniformity: [0.; 3],
                density_min: [0.; 3],
                blur: 0.,
                micro_sigma: 0.,
                micro_blur: 0.,
                positive: false,
                monochrome: false,
                fast_stats: false,
            };
            let output = apply(&gpu, &image, &p);
            let n = output.data.len() as f64;
            let mean = output.data.iter().sum::<f64>() / n;
            let variance = output.data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            let expected_mean = 0.3;
            let expected_variance = 0.3 / npp as f64;
            let mean_z = (mean - expected_mean).abs() / (expected_variance / n).sqrt();
            let variance_z = (variance / expected_variance - 1.).abs()
                / ((2. + 1. / (npp as f64 * 0.3)) / n).sqrt();
            eprintln!("Poisson thinning npp {npp}: mean z {mean_z:.2}, variance z {variance_z:.2}");
            // These are finite-sample confidence bounds, not relaxed numerical
            // tolerances. No normal approximation is used for exact sampling.
            assert!(
                mean_z < 7. && variance_z < 7.,
                "particle law changed at npp {npp}"
            );
        }
    }
}
