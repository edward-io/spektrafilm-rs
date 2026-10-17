//! Resident conversion front-end; both solver and LUT keep the image on GPU.
use super::*;
use wgpu::util::DeviceExt;
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    header: [u32; 4],
    maximum: [f32; 4],
    midpoint: [f32; 4],
    log_rgb: [f32; 4],
    seed: [[f32; 4]; 3],
    calibration: [[f32; 4]; 3],
}
pub(super) struct State {
    job: DispatchJob,
    lut: Option<tetrahedral::State>,
    _bands: wgpu::Buffer,
}
fn v4(v: [f64; 3]) -> [f32; 4] {
    [v[0] as f32, v[1] as f32, v[2] as f32, 0.]
}
pub(super) fn build(
    b: &WgpuBackend,
    p: &crate::ConvertGpuParams<'_>,
    n: u32,
    input: &wgpu::Buffer,
    intermediate: &wgpu::Buffer,
) -> State {
    let bands: Vec<[[f32; 4]; 2]> = p
        .bands
        .iter()
        .map(|v| [v4([v[0], v[1], v[2]]), v4([v[3], v[4], v[5]])])
        .collect();
    let bands = b
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("conversion spectra"),
            contents: bytemuck::cast_slice(&bands),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let seed = std::array::from_fn(|c| v4(std::array::from_fn(|r| p.seed_inverse[r][c])));
    let params = Params {
        header: [n, p.bands.len() as u32, p.lut.is_some() as u32, 0],
        maximum: v4(p.maximum),
        midpoint: v4(p.midpoint),
        log_rgb: v4(p.log_rgb),
        seed,
        calibration: p.calibration.map(v4),
    };
    let read = wgpu::BufferBindingType::Storage { read_only: true };
    let write = wgpu::BufferBindingType::Storage { read_only: false };
    let job = storage_job(
        b,
        include_str!("../../spektrafilm-shaders/wgsl/convert_film.wgsl"),
        &params,
        &[(input, read), (&bands, read), (intermediate, write)],
    );
    let lut = p
        .lut
        .map(|lut| tetrahedral::build(b, n, lut, p.bounds.0, p.bounds.1, intermediate, input));
    State {
        job,
        lut,
        _bands: bands,
    }
}
impl State {
    pub(super) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n: u32,
        input: &wgpu::Buffer,
        intermediate: &wgpu::Buffer,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("convert film"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.job.pipeline.pipeline);
            pass.set_bind_group(0, &self.job.bg, &[]);
            let groups = n.div_ceil(256);
            let y = groups.div_ceil(65535).max(1);
            pass.dispatch_workgroups(groups.div_ceil(y), y, 1);
        }
        if let Some(lut) = &self.lut {
            lut.encode(encoder, n);
        } else {
            encoder.copy_buffer_to_buffer(intermediate, 0, input, 0, n as u64 * 12);
        }
    }
}
