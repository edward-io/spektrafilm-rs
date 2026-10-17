//! Shared unclipped named CCTFs for resident input and output passes.
use super::*;
use spektrafilm_math::transfer::Transfer;
use wgpu::util::DeviceExt;
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_pixels: u32,
    mode: u32,
    encode: u32,
    pad: u32,
    scale: f32,
    padding: [f32; 3],
    controls: [f32; 4],
}
fn coefficients(t: Transfer) -> (u32, [f32; 4]) {
    use Transfer::*;
    let mode = match t {
        Linear => 0,
        Srgb => 1,
        ProPhoto => 2,
        Bt709 => 3,
        Gamma(_) => 4,
        Acescc => 5,
        Acescct => 6,
        Vlog => 7,
        Slog3 => 8,
        Logc3 => 9,
        Logc4 => 10,
        DaVinci => 11,
        Blackmagic5 => 12,
        Flog => 13,
        Flog2 => 14,
        Canonlog3 => 15,
        Log3g10 => 16,
        Applelog => 17,
        Dlog => 18,
        Nlog => 19,
        Pq => 20,
        Hlg => 21,
        Acesproxy => 22,
        Dcdm => 23,
        Cineon => 24,
        Erimm => 25,
        Tlog => 26,
        Protune => 27,
        Rimm => 28,
        Slog2 => 29,
        Smpte240 => 30,
        Lab => 31,
        Unsupported => panic!("validate transfer before dispatch"),
    };
    let controls = match t {
        Gamma(g) => [g as f32, 0., 0., 0.],
        Tlog => {
            let b = 1. / (0.7107 + 1.2359 * (128f64 * 16.).ln());
            let gs = 16. / (1. - 0.075);
            let c = b / gs;
            let a = 1. - b * (128. + c).ln();
            let y0 = a + b * c.ln();
            let scale = (1. - 0.075) / (1. - y0);
            [
                (1. + (a - 1.) * scale) as f32,
                (b * scale) as f32,
                (gs * scale) as f32,
                c as f32,
            ]
        }
        _ => [0.; 4],
    };
    (mode, controls)
}
pub(super) fn build(
    b: &WgpuBackend,
    n_pixels: u32,
    transfer: Transfer,
    encode: bool,
    scale: f64,
    img: &wgpu::Buffer,
) -> InplaceUnaryState {
    let (mode, controls) = coefficients(transfer);
    let pipeline = b.cached_pipeline(
        include_str!("../../spektrafilm-shaders/wgsl/color_transfer.wgsl"),
        &[
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: false },
        ],
    );
    let uniform = b
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("named transfer"),
            contents: bytemuck::bytes_of(&Params {
                n_pixels,
                mode,
                encode: encode as u32,
                pad: 0,
                scale: scale as f32,
                padding: [0.; 3],
                controls,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bg = b.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("named transfer"),
        layout: &pipeline.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: img.as_entire_binding(),
            },
        ],
    });
    InplaceUnaryState {
        job: DispatchJob {
            _params_buf: uniform,
            pipeline,
            bg,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn named_transfers_match_cpu_across_the_registry() {
        let Some(b) = WgpuBackend::new() else { return };
        for name in spektrafilm_math::transfer::names() {
            let t = spektrafilm_math::transfer::get(name).unwrap().transfer;
            if !t.supported() {
                continue;
            }
            for encode in [true, false] {
                let values: Vec<f64> = (0..2048)
                    .map(|i| {
                        if encode {
                            if i < 16 {
                                -(i as f64) * 0.005
                            } else {
                                2f64.powf(-20. + 27. * (i - 16) as f64 / 2031.)
                            }
                        } else {
                            -0.05 + 1.10 * i as f64 / 2047.
                        }
                    })
                    .collect();
                let input: Vec<f32> = values.iter().map(|x| *x as f32).collect();
                let (mode, controls) = coefficients(t);
                let params = Params {
                    n_pixels: input.len() as u32 / 3,
                    mode,
                    encode: encode as u32,
                    pad: 0,
                    scale: 1.,
                    padding: [0.; 3],
                    controls,
                };
                let n = params.n_pixels as usize * 3;
                let mut bytes = bytemuck::bytes_of(&params).to_vec();
                let buffers = vec![
                    GpuBuffer {
                        data: std::mem::take(&mut bytes),
                        binding_type: wgpu::BufferBindingType::Uniform,
                        usage: wgpu::BufferUsages::UNIFORM,
                    },
                    GpuBuffer {
                        data: bytemuck::cast_slice(&input[..n]).to_vec(),
                        binding_type: wgpu::BufferBindingType::Storage { read_only: false },
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    },
                ];
                let output = b.dispatch_compute(
                    include_str!("../../spektrafilm-shaders/wgsl/color_transfer.wgsl"),
                    &buffers,
                    params.n_pixels,
                    1,
                );

                for (i, &actual) in output.iter().enumerate() {
                    let reference = if encode {
                        t.encode(input[i] as f64)
                    } else {
                        t.decode(input[i] as f64)
                    };
                    if !reference.is_finite() {
                        continue;
                    }
                    let tolerance = if encode {
                        0.001
                    } else {
                        0.0001 * reference.abs().max(1.)
                    };
                    assert!(
                        actual.is_finite() && (actual as f64 - reference).abs() <= tolerance,
                        "{name}, encode {encode}, input {}: {actual} vs {reference} (tol {tolerance})",
                        input[i]
                    );
                }
            }
        }
    }
}
