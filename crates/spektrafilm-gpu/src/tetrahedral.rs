//! Shared four-tap runtime LUT, including resident scan conversion.
use super::*;
use spektrafilm_math::lut::TetrahedralLut;
use wgpu::util::DeviceExt;
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    header: [u32; 4],
    min: [f32; 4],
    inv: [f32; 4],
}
fn params(n: u32, lut: &TetrahedralLut, min: [f64; 3], max: [f64; 3]) -> Params {
    let s = (lut.size - 1) as f64;
    Params {
        header: [n, lut.size as u32, 0, 0],
        min: [min[0] as f32, min[1] as f32, min[2] as f32, 0.],
        inv: [
            (s / (max[0] - min[0])) as f32,
            (s / (max[1] - min[1])) as f32,
            (s / (max[2] - min[2])) as f32,
            0.,
        ],
    }
}
pub(super) struct State {
    job: DispatchJob,
    _nodes: wgpu::Buffer,
}
impl State {
    pub(super) fn encode(&self, encoder: &mut wgpu::CommandEncoder, n: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("runtime tetrahedral LUT"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.job.pipeline.pipeline);
        pass.set_bind_group(0, &self.job.bg, &[]);
        let groups = n.div_ceil(256);
        let y = groups.div_ceil(65535).max(1);
        pass.dispatch_workgroups(groups.div_ceil(y), y, 1);
    }
}
pub(super) fn build(
    b: &WgpuBackend,
    n: u32,
    lut: &TetrahedralLut,
    min: [f64; 3],
    max: [f64; 3],
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
) -> State {
    let nodes: Vec<f32> = lut.data.iter().map(|v| *v as f32).collect();
    let nodes = b
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("runtime LUT"),
            contents: bytemuck::cast_slice(&nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let read = wgpu::BufferBindingType::Storage { read_only: true };
    let write = wgpu::BufferBindingType::Storage { read_only: false };
    State {
        job: storage_job(
            b,
            include_str!("../../spektrafilm-shaders/wgsl/tetrahedral3d.wgsl"),
            &params(n, lut, min, max),
            &[(input, read), (&nodes, read), (output, write)],
        ),
        _nodes: nodes,
    }
}
pub(super) fn apply(
    b: &WgpuBackend,
    image: &ImageBuf,
    lut: &TetrahedralLut,
    min: [f64; 3],
    max: [f64; 3],
) -> ImageBuf {
    let nodes: Vec<f32> = lut.data.iter().map(|v| *v as f32).collect();
    let bindings = [
        GpuBuffer::uniform(bytemuck::bytes_of(&params(
            image.pixel_count() as u32,
            lut,
            min,
            max,
        ))),
        GpuBuffer::storage_ro(bytemuck::cast_slice(&scalars_to_f32(&image.data))),
        GpuBuffer::storage_ro(bytemuck::cast_slice(&nodes)),
        GpuBuffer::storage_rw(&vec![0; image.data.len() * 4]),
    ];
    let groups = (image.pixel_count() as u32).div_ceil(256);
    let y = groups.div_ceil(65535).max(1);
    let out = b.dispatch_compute_grid(
        include_str!("../../spektrafilm-shaders/wgsl/tetrahedral3d.wgsl"),
        &bindings,
        [groups.div_ceil(y), y, 1],
        3,
    );
    ImageBuf::from_data(image.width, image.height, f32_to_scalars(out))
}
