//! GPU canvas filters and paper development; texture uploads happen once per render.
use eframe::{egui, egui_wgpu, wgpu};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub const INTERPOLATIONS: [&str; 7] = [
    "nearest", "linear", "cubic", "spline16", "spline36", "lanczos", "blackman",
];
pub struct Image {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    pub maxima: [f32; 4],
    pub chroma_max: [f32; 4],
    pub size: [usize; 2],
}
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    linear_framebuffer: bool,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    uv: [f32; 4],
    options: [f32; 4],
    maxima: [f32; 4],
    chroma_max: [f32; 4],
    coefficients0: [f32; 4],
    coefficients1: [f32; 4],
    flags: [f32; 4],
    transform0: [f32; 4],
    transform1: [f32; 4],
}
impl Renderer {
    pub fn new(state: &egui_wgpu::RenderState) -> Arc<Self> {
        Self::with_device(
            state.device.clone(),
            state.queue.clone(),
            state.target_format,
        )
    }
    fn with_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        format: wgpu::TextureFormat,
    ) -> Arc<Self> {
        let texture = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("canvas layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: texture,
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: texture,
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("canvas sampling and development"),
            source: wgpu::ShaderSource::Wgsl(include_str!("view.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("canvas"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("canvas"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: if format == wgpu::TextureFormat::Rgba32Float {
                        None
                    } else {
                        Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING)
                    },
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        Arc::new(Self {
            device,
            queue,
            layout,
            pipeline,
            linear_framebuffer: format.is_srgb(),
        })
    }
    pub fn upload(&self, data: &super::PreviewTextureData) -> Arc<Image> {
        let [w, h] = data.size;
        let size = wgpu::Extent3d {
            width: w as u32,
            height: h as u32,
            depth_or_array_layers: 1,
        };
        // Unorm is intentional: the source contains display-encoded samples.
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("canvas image"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w as u32 * 4),
                rows_per_image: Some(h as u32),
            },
            size,
        );
        let (maxima, chroma_max) = density_maxima(&data.rgba);
        Arc::new(Image {
            view: texture.create_view(&Default::default()),
            _texture: texture,
            maxima,
            chroma_max,
            size: data.size,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn paint(
        self: &Arc<Self>,
        painter: &egui::Painter,
        rect: egui::Rect,
        image: &Arc<Image>,
        previous: Option<&Arc<Image>>,
        opacity: f32,
        interpolation: &str,
        development: Option<f32>,
        fade: f32,
    ) {
        let visible = rect.intersect(painter.clip_rect());
        if !visible.is_positive() || opacity <= 0. {
            return;
        }
        let uv0 = (visible.min - rect.min) / rect.size();
        let uv1 = (visible.max - rect.min) / rect.size();
        let params = self.params(image, opacity, interpolation, development, fade);
        let params = Params {
            uv: [uv0.x, uv0.y, uv1.x, uv1.y],
            ..params
        };
        self.draw(painter, visible, params, image, previous);
    }

    /// Paint `image` over the whole clip rect through `to_image`, which maps a
    /// screen position to image pixels (e.g. the image rotated under the crop
    /// tool). Screen pixels outside the image stay transparent.
    pub fn paint_mapped(
        self: &Arc<Self>,
        painter: &egui::Painter,
        image: &Arc<Image>,
        interpolation: &str,
        to_image: impl Fn(egui::Pos2) -> egui::Pos2,
    ) {
        let visible = painter.clip_rect();
        if !visible.is_positive() {
            return;
        }
        let size = egui::vec2(image.size[0] as f32, image.size[1] as f32);
        // Affine: image uv from viewport uv, from three mapped corners.
        let uv = |p: egui::Pos2| (to_image(p).to_vec2() / size).to_pos2();
        let origin = uv(visible.left_top());
        let x = uv(visible.right_top()) - origin;
        let y = uv(visible.left_bottom()) - origin;
        let params = Params {
            uv: [0., 0., 1., 1.],
            transform0: [x.x, y.x, x.y, y.y],
            transform1: [origin.x, origin.y, 1., 0.],
            ..self.params(image, 1., interpolation, None, 1.)
        };
        self.draw(painter, visible, params, image, None);
    }

    fn params(
        &self,
        image: &Image,
        opacity: f32,
        interpolation: &str,
        development: Option<f32>,
        fade: f32,
    ) -> Params {
        let coeff = coefficients(development.unwrap_or(1.));
        Params {
            uv: [0., 0., 1., 1.],
            options: [
                opacity,
                INTERPOLATIONS
                    .iter()
                    .position(|m| *m == interpolation)
                    .unwrap_or(1) as f32,
                development.unwrap_or(-1.),
                fade,
            ],
            maxima: image.maxima,
            chroma_max: image.chroma_max,
            coefficients0: coeff[..4].try_into().unwrap(),
            coefficients1: coeff[4..8].try_into().unwrap(),
            flags: [coeff[8], self.linear_framebuffer as u8 as f32, 0., 0.],
            transform0: [1., 0., 0., 1.],
            transform1: [0.; 4],
        }
    }

    fn draw(
        self: &Arc<Self>,
        painter: &egui::Painter,
        visible: egui::Rect,
        params: Params,
        image: &Arc<Image>,
        previous: Option<&Arc<Image>>,
    ) {
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("canvas frame"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let previous = previous
            .filter(|old| old.size == image.size)
            .unwrap_or(image);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("canvas image"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&image.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&previous.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        painter.add(egui_wgpu::Callback::new_paint_callback(
            visible,
            Paint {
                renderer: self.clone(),
                bg,
                _uniform: uniform,
                _image: image.clone(),
                _previous: previous.clone(),
            },
        ));
    }
}
struct Paint {
    renderer: Arc<Renderer>,
    bg: wgpu::BindGroup,
    _uniform: wgpu::Buffer,
    _image: Arc<Image>,
    _previous: Arc<Image>,
}
impl egui_wgpu::CallbackTrait for Paint {
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        _resources: &egui_wgpu::CallbackResources,
    ) {
        pass.set_pipeline(&self.renderer.pipeline);
        pass.set_bind_group(0, &self.bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
pub fn density_maxima(rgba: &[u8]) -> ([f32; 4], [f32; 4]) {
    let mut maxima = [0f32; 4];
    let mut chroma_max = [0f32; 4];
    // A 256-entry LUT avoids three logarithms per image pixel on the CPU.
    let lut: [f32; 256] =
        std::array::from_fn(|i| (-2.2 * (i as f32 / 255.).max(0.0001).ln()).clamp(0., 6.));
    for p in rgba.chunks_exact(4) {
        let d = [lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize]];
        let n = d.into_iter().fold(f32::INFINITY, f32::min);
        maxima[3] = maxima[3].max(n);
        for c in 0..3 {
            maxima[c] = maxima[c].max(d[c]);
            chroma_max[c] = chroma_max[c].max(d[c] - n);
        }
    }
    (maxima, chroma_max)
}
/// Source-equivalent nine-layer development schedule, evaluated once per frame.
pub fn coefficients(t: f32) -> [f32; 9] {
    fn sigmoid(t: f32, c: f32, s: f32) -> f32 {
        let expit = |x: f32| 1. / (1. + (-x.clamp(-60., 60.)).exp());
        let start = expit(-c * s);
        let end = expit((1. - c) * s);
        ((expit((t - c) * s) - start) / (end - start + 0.0001)).clamp(0., 1.)
    }
    fn rise(t: f32, c: f32, s: f32, p: f32) -> f32 {
        let r = sigmoid(t, c, s);
        r + (1. - r) * t.powf(p)
    }
    fn pulse(t: f32, on: f32, ons: f32, off: f32, offs: f32, p: f32) -> f32 {
        sigmoid(t, on, ons) * (1. - sigmoid(t, off, offs)) * (1. - t).powf(p)
    }
    let t = t.clamp(0., 1.);
    let body = rise(t, 0.19, 7.4, 3.8);
    [
        body * rise(t, 0.12, 8., 2.8),
        body * rise(t, 0.22, 11., 3.),
        body * rise(t, 0.31, 10.2, 3.4),
        body * rise(t, 0.52, 9.4, 5.4),
        1.08 * (1. - t).powf(0.46),
        pulse(t, 0.05, 32., 0.44, 8., 0.24),
        1.22 * pulse(t, 0.10, 54., 0.28, 28., 0.18),
        0.96 * pulse(t, 0.12, 20., 0.80, 6.5, 0.34),
        0.68 * pulse(t, 0.03, 24., 0.98, 4., 0.08),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gpu_development_matches_python_frames() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/development.json")).unwrap();
        let instance = wgpu::Instance::new(&Default::default());
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default()))
        else {
            return;
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
                .unwrap();
        let renderer = Renderer::with_device(
            device.clone(),
            queue.clone(),
            wgpu::TextureFormat::Rgba32Float,
        );
        let data = super::super::PreviewTextureData {
            size: [16, 8],
            rgba: serde_json::from_value(fixture["rgba"].clone()).unwrap(),
        };
        let image = renderer.upload(&data);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("canvas test output"),
            size: wgpu::Extent3d {
                width: 16,
                height: 8,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("canvas test readback"),
            size: 16 * 8 * 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut frames = fixture["frames"].as_array().unwrap().clone();
        let interpolation: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/interpolation.json")).unwrap();
        frames.extend(interpolation["frames"].as_array().unwrap().clone());
        for frame in &frames {
            let t = frame["t"].as_f64().unwrap_or(-1.) as f32;
            let mode = frame["mode"].as_u64().unwrap_or(0) as f32;
            let coeff = coefficients(t);
            if !frame["coefficients"].is_null() {
                let expected: Vec<f32> =
                    serde_json::from_value(frame["coefficients"].clone()).unwrap();
                for i in 0..9 {
                    assert!(
                        (coeff[i] - expected[i]).abs() < 1e-6,
                        "coefficient {i} t {t}"
                    );
                }
            }
            let params = Params {
                uv: if frame["uv"].is_null() {
                    [0., 0., 1., 1.]
                } else {
                    serde_json::from_value(frame["uv"].clone()).unwrap()
                },
                options: [1., mode, t, 1.],
                maxima: image.maxima,
                chroma_max: image.chroma_max,
                coefficients0: coeff[..4].try_into().unwrap(),
                coefficients1: coeff[4..8].try_into().unwrap(),
                flags: [coeff[8], 0., 0., 0.],
                transform0: [1., 0., 0., 1.],
                transform1: [0.; 4],
            };
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &renderer.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&image.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&image.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            let target_view = target.create_view(&Default::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&renderer.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &target,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(8),
                    },
                },
                wgpu::Extent3d {
                    width: 16,
                    height: 8,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit(Some(encoder.finish()));
            let slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let mapped = slice.get_mapped_range();
            let actual: &[f32] = bytemuck::cast_slice(&mapped);
            let expected: Vec<f32> = serde_json::from_value(frame["rgb"].clone()).unwrap();
            let max = actual
                .chunks_exact(4)
                .zip(expected.chunks_exact(3))
                .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs()))
                .fold(0f32, f32::max);
            assert!(
                max < 2e-5,
                "canvas differs from Python t {t}, mode {mode}: max {max}"
            );
            drop(mapped);
            readback.unmap();
        }
    }
}
