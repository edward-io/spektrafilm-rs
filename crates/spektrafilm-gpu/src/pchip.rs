//! Monotone PCHIP interpolation for the optional enlarger/scanner LUTs.
use super::*;
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_pixels: u32,
    size: u32,
    pad: [u32; 2],
    data_min: [f32; 4],
    inv: [f32; 4],
}
pub(super) fn apply(
    b: &WgpuBackend,
    image: &ImageBuf,
    prepared: &spektrafilm_math::pchip3d::PreparedPchip3d,
    min: [f64; 3],
    max: [f64; 3],
) -> ImageBuf {
    let s = (prepared.size - 1) as f64;
    let p = Params {
        n_pixels: image.pixel_count() as u32,
        size: prepared.size as u32,
        pad: [0; 2],
        data_min: [min[0] as f32, min[1] as f32, min[2] as f32, 0.],
        inv: [
            (s / (max[0] - min[0])) as f32,
            (s / (max[1] - min[1])) as f32,
            (s / (max[2] - min[2])) as f32,
            0.,
        ],
    };
    let nodes: Vec<[f32; 4]> = (0..prepared.lut.len())
        .map(|i| {
            [
                prepared.lut[i] as f32,
                prepared.slope_x[i] as f32,
                prepared.slope_y[i] as f32,
                prepared.slope_z[i] as f32,
            ]
        })
        .collect();
    let read = wgpu::BufferBindingType::Storage { read_only: true };
    let bindings = [
        GpuBuffer {
            data: bytemuck::bytes_of(&p).to_vec(),
            binding_type: wgpu::BufferBindingType::Uniform,
            usage: wgpu::BufferUsages::UNIFORM,
        },
        GpuBuffer {
            data: bytemuck::cast_slice(&scalars_to_f32(&image.data)).to_vec(),
            binding_type: read,
            usage: wgpu::BufferUsages::STORAGE,
        },
        GpuBuffer {
            data: bytemuck::cast_slice(&nodes).to_vec(),
            binding_type: read,
            usage: wgpu::BufferUsages::STORAGE,
        },
        GpuBuffer {
            data: vec![0; image.data.len() * 4],
            binding_type: wgpu::BufferBindingType::Storage { read_only: false },
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        },
    ];
    // A 256-thread group keeps the eight-corner Hermite kernel within Metal's
    // per-thread register budget; 2D dispatch also covers full-resolution RAW.
    let groups = p.n_pixels.div_ceil(256);
    let y = groups.div_ceil(65535).max(1);
    let result = b.dispatch_compute_grid(
        include_str!("../../spektrafilm-shaders/wgsl/pchip3d.wgsl"),
        &bindings,
        [groups.div_ceil(y), y, 1],
        3,
    );
    ImageBuf::from_data(image.width, image.height, f32_to_scalars(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tensor_hermite_matches_reference_between_nodes_and_outside_bounds() {
        let Some(gpu) = WgpuBackend::new() else {
            return;
        };
        let n = 9;
        let mut lut = vec![];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let x = i as f64 / 8.;
                    let y = j as f64 / 8.;
                    let z = k as f64 / 8.;
                    lut.extend([
                        (x + y * z + 0.1).ln(),
                        x * y + z * z,
                        (y * 4.).sin() + x * z,
                    ]);
                }
            }
        }
        let prepared = spektrafilm_math::pchip3d::prepare_pchip_3d(lut, n);
        let input = ImageBuf::from_data(
            37,
            29,
            (0..37 * 29 * 3)
                .map(|i| {
                    spektrafilm_math::precision::from_f64(
                        -0.2 + 1.5 * ((i * 179) % 4093) as f64 / 4093.,
                    )
                })
                .collect(),
        );
        let output = apply(&gpu, &input, &prepared, [0.; 3], [1.; 3]);
        let mut max = 0f64;
        for (src, actual) in input.pixels().zip(output.pixels()) {
            let expected = spektrafilm_math::pchip3d::pchip_interp(
                &prepared,
                src[0] as f64 * 8.,
                src[1] as f64 * 8.,
                src[2] as f64 * 8.,
            );
            for c in 0..3 {
                max = max.max((actual[c] as f64 - expected[c]).abs());
            }
        }
        assert!(max < 2e-6, "GPU PCHIP error {max}");
    }
}
