//! Interactive preparation with one upload and one compact readback per operation.
use super::*;
const READ: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: true };
const WRITE: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: false };

fn groups(n: u32) -> [u32; 3] {
    let count = n.div_ceil(256);
    let y = count.div_ceil(65535).max(1);
    [count.div_ceil(y), y, 1]
}

#[derive(Default)]
pub(super) struct Buffers {
    input: Option<wgpu::Buffer>,
    output: Option<wgpu::Buffer>,
}

fn map(b: &WgpuBackend, buffer: &wgpu::Buffer, bytes: u64, mode: wgpu::MapMode) -> Option<()> {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..bytes).map_async(mode, move |result| {
        let _ = tx.send(result);
    });
    b.device.poll(wgpu::Maintain::Wait);
    rx.recv().ok()?.ok()?;
    Some(())
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn reused_preview_buffers_refresh_pixels_when_images_grow_and_shrink() {
        let Some(gpu) = WgpuBackend::new() else {
            return;
        };
        if !gpu
            .device
            .features()
            .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS)
        {
            return;
        }
        let mut largest = 0;
        for (width, height, rgb, bytes) in [
            (128, 64, [1., 0., 0.], [255, 0, 0, 255]),
            (8, 4, [0., 1., 0.], [0, 255, 0, 255]),
            (256, 128, [0., 0., 1.], [0, 0, 255, 255]),
            (7, 3, [0.25, 0.5, 0.75], [64, 128, 191, 255]),
        ] {
            let image = ImageBuf::from_data(
                width,
                height,
                (0..width * height)
                    .flat_map(|_| rgb.map(spektrafilm_math::precision::from_f64))
                    .collect(),
            );
            let preview = gpu
                .display_preview_gpu(&image, "sRGB", true, false, 256)
                .unwrap();
            assert_eq!(preview.size, [width as usize, height as usize]);
            assert!(preview.rgba.chunks_exact(4).all(|pixel| pixel == bytes));
            largest = largest.max(width as u64 * height as u64);
            let buffers = gpu.preview_buffers.lock().unwrap();
            assert_eq!(buffers.input.as_ref().unwrap().size(), largest * 12);
            assert_eq!(buffers.output.as_ref().unwrap().size(), largest * 4);
        }
    }
}

impl WgpuBackend {
    fn preview_buffer(&self, bytes: u64, mapping: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("display preview"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | mapping,
            mapped_at_creation: false,
        })
    }

    pub(super) fn display_preview_gpu(
        &self,
        image: &ImageBuf,
        color: &str,
        encoded: bool,
        transform: bool,
        max_size: usize,
    ) -> Option<crate::DisplayPreview> {
        // Shared mapped buffers avoid allocating and staging a float image on every edit.
        if !self
            .device
            .features()
            .contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS)
        {
            return None;
        }
        let long = image.width.max(image.height).max(1) as usize;
        let scale = max_size.min(long);
        let [nw, nh] = [image.width, image.height]
            .map(|v| ((v as usize * scale + long / 2) / long).max(1) as u32);
        let input_bytes = image.pixel_count() as u64 * 12;
        let output_bytes = nw as u64 * nh as u64 * 4;
        let limit = self.device.limits().max_storage_buffer_binding_size as u64;
        if input_bytes == 0 || input_bytes > limit || output_bytes > limit {
            return None;
        }
        let convert = transform && !(color == "sRGB" && encoded);
        if convert
            && encoded
            && (image.pixel_count() as u32).div_ceil(1024)
                > self.device.limits().max_compute_workgroups_per_dimension
        {
            return None;
        }
        let _gpu = crate::scheduler::gpu_permit();
        let mut buffers = self.preview_buffers.lock().unwrap();
        let input = buffers
            .input
            .get_or_insert_with(|| self.preview_buffer(input_bytes, wgpu::BufferUsages::MAP_WRITE));
        if input.size() < input_bytes {
            *input = self.preview_buffer(input_bytes, wgpu::BufferUsages::MAP_WRITE);
        }
        map(self, input, input_bytes, wgpu::MapMode::Write)?;
        input
            .slice(..input_bytes)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&scalars_to_f32(&image.data)));
        input.unmap();
        let output = buffers
            .output
            .get_or_insert_with(|| self.preview_buffer(output_bytes, wgpu::BufferUsages::MAP_READ));
        if output.size() < output_bytes {
            *output = self.preview_buffer(output_bytes, wgpu::BufferUsages::MAP_READ);
        }
        let Buffers {
            input: Some(input),
            output: Some(output),
        } = &*buffers
        else {
            unreachable!()
        };
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if convert && encoded {
            let transfer = spektrafilm_math::transfer::get(color).ok()?.transfer;
            if !transfer.supported() {
                return None;
            }
            color_transfer::build(
                self,
                image.pixel_count() as u32,
                transfer,
                false,
                1.,
                &input,
            )
            .encode_passes(&mut encoder, image.pixel_count() as u32);
        }
        let matrix = if convert {
            spektrafilm_math::transfer::rgb_matrix(color, "sRGB")
        } else {
            [[0.; 3]; 3]
        };
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            source: [u32; 2],
            destination: [u32; 2],
            matrix: [[f32; 4]; 3],
            convert: u32,
            pad: [u32; 3],
        }
        let mut state = spatial::State::new();
        state.dispatch(
            self,
            include_str!("../../spektrafilm-shaders/wgsl/display_preview.wgsl"),
            &Params {
                source: [image.width, image.height],
                destination: [nw, nh],
                matrix: matrix.map(|r| [r[0] as f32, r[1] as f32, r[2] as f32, 0.]),
                convert: convert as u32,
                pad: [0; 3],
            },
            &[(&input, READ), (&output, WRITE)],
            groups(nw * nh),
        );
        state.encode(&mut encoder);
        Some(crate::DisplayPreview {
            size: [nw as usize, nh as usize],
            rgba: {
                self.queue.submit(Some(encoder.finish()));
                map(self, output, output_bytes, wgpu::MapMode::Read)?;
                let bytes = output.slice(..output_bytes).get_mapped_range().to_vec();
                output.unmap();
                bytes
            },
        })
    }
}
