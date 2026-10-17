//! LibRaw's public C API, without depending on its version-specific private data layout.
//! Optional: set SPEKTRAFILM_LIBRAW_LIBRARY or install a system LibRaw library.
use super::{LoadedImage, Metadata, RawOptions};
use anyhow::{Result, bail};
use rayon::prelude::*;
use spektrafilm_math::{colorspace, image::ImageBuf, precision::from_f64, transfer};
use std::{
    ffi::{CString, c_char, c_void},
    path::Path,
    time::Instant,
};
type Ptr = *mut c_void;
#[repr(C)]
struct Processed {
    kind: i32,
    height: u16,
    width: u16,
    colors: u16,
    bits: u16,
    data_size: u32,
    data: [u8; 1],
}

pub fn library() -> Result<libloading::Library> {
    let names = std::env::var_os("SPEKTRAFILM_LIBRAW_LIBRARY")
        .map(|s| vec![s.to_string_lossy().to_string()])
        .unwrap_or_else(|| {
            vec![
                "libraw_r.dylib".into(),
                "/opt/homebrew/lib/libraw_r.dylib".into(),
                "/usr/local/lib/libraw_r.dylib".into(),
                "libraw_r.so.25".into(),
                "libraw_r.so.23".into(),
                "libraw_r.so.24".into(),
                "libraw_r.so".into(),
                "raw_r.dll".into(),
                "libraw.dll".into(),
            ]
        });
    names
        .iter()
        .find_map(|s| unsafe { libloading::Library::new(s).ok() })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "LibRaw is unavailable; install LibRaw or set SPEKTRAFILM_LIBRAW_LIBRARY{}",
                std::env::var_os("SPEKTRAFILM_LIBRAW_LIBRARY")
                    .map(|s| format!(" (could not load {})", s.to_string_lossy()))
                    .unwrap_or_default()
            )
        })
}
pub fn load(path: &Path, options: &RawOptions, lib: libloading::Library) -> Result<LoadedImage> {
    let started = Instant::now();
    // SAFETY: only documented C API functions and libraw_processed_image_t's
    // stable layout are used. Guards free both allocations before unloading.
    unsafe {
        let strerror =
            lib.get::<unsafe extern "C" fn(i32) -> *const c_char>(b"libraw_strerror\0")?;
        let init = lib.get::<unsafe extern "C" fn(u32) -> Ptr>(b"libraw_init\0")?;
        let close = lib.get::<unsafe extern "C" fn(Ptr)>(b"libraw_close\0")?;
        let open =
            lib.get::<unsafe extern "C" fn(Ptr, *const c_char) -> i32>(b"libraw_open_file\0")?;
        let unpack = lib.get::<unsafe extern "C" fn(Ptr) -> i32>(b"libraw_unpack\0")?;
        let process = lib.get::<unsafe extern "C" fn(Ptr) -> i32>(b"libraw_dcraw_process\0")?;
        let output_color =
            lib.get::<unsafe extern "C" fn(Ptr, i32)>(b"libraw_set_output_color\0")?;
        let bps = lib.get::<unsafe extern "C" fn(Ptr, i32)>(b"libraw_set_output_bps\0")?;
        let no_bright =
            lib.get::<unsafe extern "C" fn(Ptr, i32)>(b"libraw_set_no_auto_bright\0")?;
        let gamma = lib.get::<unsafe extern "C" fn(Ptr, i32, f32)>(b"libraw_set_gamma\0")?;
        let cam_mul = lib.get::<unsafe extern "C" fn(Ptr, i32) -> f32>(b"libraw_get_cam_mul\0")?;
        let user_mul = lib.get::<unsafe extern "C" fn(Ptr, i32, f32)>(b"libraw_set_user_mul\0")?;
        let mem = lib.get::<unsafe extern "C" fn(Ptr, *mut i32) -> *mut Processed>(
            b"libraw_dcraw_make_mem_image\0",
        )?;
        let free = lib.get::<unsafe extern "C" fn(Ptr)>(b"libraw_dcraw_clear_mem\0")?;
        let raw = init(0);
        if raw.is_null() {
            bail!("LibRaw allocation failed");
        }
        struct Guard(Ptr, unsafe extern "C" fn(Ptr));
        impl Drop for Guard {
            fn drop(&mut self) {
                unsafe { (self.1)(self.0) }
            }
        }
        let _raw = Guard(raw, *close);
        let filename = CString::new(path.to_string_lossy().as_bytes())?;
        let check = |code: i32, stage: &str| {
            if code == 0 {
                Ok(())
            } else {
                Err(anyhow::anyhow!(
                    "LibRaw {stage} failed: {} ({code})",
                    std::ffi::CStr::from_ptr(strerror(code)).to_string_lossy()
                ))
            }
        };
        check(open(raw, filename.as_ptr()), "open")?;
        let opened = started.elapsed();
        check(unpack(raw), "unpack")?;
        let unpacked = started.elapsed();
        // LibRaw 0.22 exposes the vendor crop through its stable C API.
        // Apply it after unpack (masked-area black levels are already read)
        // and before processing; otherwise newer Sony ARWs include padding.
        // Older libraries have no C entry point and keep their native crop.
        if let Ok(adjust_crop) = lib
            .get::<unsafe extern "C" fn(Ptr, u32, f32) -> i32>(b"libraw_adjust_to_raw_inset_crop\0")
        {
            adjust_crop(raw, 1, 0.55);
        }
        output_color(raw, 6);
        bps(raw, 16);
        no_bright(raw, 1);
        gamma(raw, 0, 1.);
        gamma(raw, 1, 1.);
        if matches!(options.white_balance.as_str(), "camera" | "as_shot") && cam_mul(raw, 0) > 0. {
            for c in 0..4 {
                let v = cam_mul(raw, c);
                user_mul(raw, c, if v > 0. { v } else { cam_mul(raw, 1) });
            }
        }
        check(process(raw), "process")?;
        let processed = started.elapsed();
        let mut error = 0;
        let image = mem(raw, &mut error);
        check(error, "image buffer")?;
        if image.is_null() {
            bail!("LibRaw returned an empty image");
        }
        let _image = Guard(image.cast(), *free);
        let copied = started.elapsed();
        let header = &*image;
        if header.kind != 2
            || header.bits != 16
            || header.colors < 3
            || header.width == 0
            || header.height == 0
        {
            bail!("LibRaw returned an unsupported bitmap layout");
        }
        let count = header.width as usize * header.height as usize * header.colors as usize;
        if count.checked_mul(2) != Some(header.data_size as usize) {
            bail!("LibRaw bitmap size mismatch");
        }
        let bytes = std::slice::from_raw_parts(header.data.as_ptr(), header.data_size as usize);
        let mut pixels = ImageBuf::new(header.width as u32, header.height as u32);
        copy_bitmap(&mut pixels, bytes, header.colors as usize);
        if options.lens_correction {
            super::lens::correct(&mut pixels, path, options.lens_database.as_deref())?;
        }
        if matches!(options.white_balance.as_str(), "custom" | "tungsten") {
            let temperature = if options.white_balance == "tungsten" {
                2850.
            } else {
                options.temperature
            };
            let matrix = colorspace::chromatic_adaptation_matrix_f64(
                white_xyz(temperature),
                white_xyz(6504.),
            );
            let cs = transfer::get("ACES2065-1").unwrap();
            for p in pixels.data.chunks_exact_mut(3) {
                let rgb = [p[0] as f64, p[1] as f64, p[2] as f64];
                let mul = |m: &[[f64; 3]; 3], v: [f64; 3]| {
                    std::array::from_fn::<_, 3, _>(|i| {
                        m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2]
                    })
                };
                let rgb = mul(&cs.from_xyz, mul(&matrix, mul(&cs.to_xyz, rgb)));
                for c in 0..3 {
                    p[c] = from_f64(rgb[c] as f32 as f64);
                }
                if options.white_balance == "custom" {
                    p[1] = from_f64((p[1] as f32 * options.tint as f32) as f64);
                }
            }
        }
        let converted = started.elapsed();
        let native = little_exif::metadata::Metadata::new_from_path(path)
            .or_else(|_| {
                little_exif::metadata::Metadata::new_from_vec(
                    &std::fs::read(path).unwrap_or_default(),
                    little_exif::filetype::FileExtension::TIFF,
                )
            })
            .ok();
        let source_orientation = native
            .as_ref()
            .and_then(|metadata| {
                metadata.into_iter().find_map(|tag| match tag {
                    little_exif::exif_tag::ExifTag::Orientation(value) => value.first().copied(),
                    _ => None,
                })
            })
            .unwrap_or(1);
        tracing::debug!(
            open_ms = opened.as_secs_f64() * 1000.,
            unpack_ms = (unpacked - opened).as_secs_f64() * 1000.,
            develop_ms = (processed - unpacked).as_secs_f64() * 1000.,
            copy_ms = (copied - processed).as_secs_f64() * 1000.,
            convert_ms = (converted - copied).as_secs_f64() * 1000.,
            metadata_ms = (started.elapsed() - converted).as_secs_f64() * 1000.,
            "LibRaw load"
        );
        Ok(LoadedImage {
            pixels,
            metadata: Metadata {
                source_orientation,
                native,
                ..Default::default()
            },
            color_space: "ACES2065-1".into(),
            encoded: false,
        })
    }
}

fn copy_bitmap(pixels: &mut ImageBuf, bytes: &[u8], colors: usize) {
    let stride = pixels.width as usize * colors * 2;
    pixels
        .par_rows_mut()
        .zip(bytes.par_chunks_exact(stride))
        .for_each(|(row, source)| {
            for (dst, src) in row.chunks_exact_mut(3).zip(source.chunks_exact(colors * 2)) {
                for c in 0..3 {
                    let v = u16::from_ne_bytes([src[c * 2], src[c * 2 + 1]]) as f32 / 65535f32;
                    dst[c] = from_f64(v as f64);
                }
            }
        });
}

use spektrafilm_math::illuminants::white_xyz;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_bitmap_copy_preserves_rgb_samples_and_ignores_extra_channels() {
        for colors in [3, 4] {
            let samples: Vec<u16> = (0..35 * 19 * colors).map(|i| (i * 997) as u16).collect();
            let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let mut pixels = ImageBuf::new(35, 19);
            copy_bitmap(&mut pixels, &bytes, colors);
            for (actual, source) in pixels.pixels().zip(samples.chunks_exact(colors)) {
                for c in 0..3 {
                    assert_eq!(actual[c], from_f64((source[c] as f32 / 65535f32) as f64));
                }
            }
        }
    }
}
