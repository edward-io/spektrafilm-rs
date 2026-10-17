//! Shared application image I/O. Quantization and clipping happen only here.
mod lens;
mod lens_api;
pub mod lens_database;
mod raw_libraw;
mod xmp;
use anyhow::{Context, Result, bail};
use image::{ImageDecoder, ImageEncoder};
use serde::{Deserialize, Serialize};
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
use std::{fs::File, io::BufWriter, path::Path};

#[derive(Clone, Default)]
pub struct Metadata {
    pub native: Option<little_exif::metadata::Metadata>,
    pub jpeg_segments: Vec<Vec<u8>>,
    pub exif: Option<Vec<u8>>,
    pub xmp: Option<Vec<u8>>,
    pub iptc: Option<Vec<u8>>,
}
pub struct LoadedImage {
    pub pixels: ImageBuf,
    pub metadata: Metadata,
    pub color_space: String,
    pub encoded: bool,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RawOptions {
    pub decoder: String,
    pub white_balance: String,
    pub temperature: f64,
    pub tint: f64,
    pub lens_correction: bool,
    pub lens_database: Option<std::path::PathBuf>,
}
impl Default for RawOptions {
    fn default() -> Self {
        Self {
            decoder: "auto".into(),
            white_balance: "as_shot".into(),
            temperature: 5500.,
            tint: 1.,
            lens_correction: false,
            lens_database: None,
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SaveOptions {
    pub bit_depth: u8,
    pub jpeg_quality: u8,
    pub color_space: String,
    pub encoded: bool,
}
impl Default for SaveOptions {
    fn default() -> Self {
        Self {
            bit_depth: 16,
            jpeg_quality: 95,
            color_space: "sRGB".into(),
            encoded: true,
        }
    }
}
fn extension(path: &Path) -> String {
    path.extension()
        .and_then(|v| v.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}
pub fn is_raw(path: &Path) -> bool {
    matches!(
        extension(path).as_str(),
        "dng"
            | "cr2"
            | "cr3"
            | "nef"
            | "nrw"
            | "arw"
            | "srf"
            | "sr2"
            | "raf"
            | "orf"
            | "rw2"
            | "pef"
            | "srw"
            | "x3f"
            | "iiq"
            | "3fr"
            | "crw"
            | "rwl"
            | "mrw"
            | "mef"
            | "kdc"
            | "ari"
            | "bay"
            | "dcr"
            | "drf"
            | "erf"
            | "fff"
            | "k25"
            | "mos"
            | "ptx"
    )
}

pub fn load(path: &Path, raw: &RawOptions) -> Result<LoadedImage> {
    if is_raw(path) {
        if !["auto", "libraw", "rawler"].contains(&raw.decoder.as_str()) {
            bail!("unknown RAW decoder: {}", raw.decoder);
        }
        if !["camera", "as_shot", "daylight", "tungsten", "custom"]
            .contains(&raw.white_balance.as_str())
        {
            bail!("unknown RAW white balance mode: {}", raw.white_balance);
        }
        if !raw.temperature.is_finite()
            || raw.temperature <= 0.
            || !raw.tint.is_finite()
            || raw.tint <= 0.
        {
            bail!("RAW temperature and tint must be finite and positive");
        }
        if raw.decoder == "rawler" {
            return load_raw(path, raw);
        }
        let libraw_result =
            raw_libraw::library().and_then(|library| raw_libraw::load(path, raw, library));
        match libraw_result {
            Ok(image) => return Ok(image),
            Err(e) if raw.decoder == "libraw" || e.is::<lens::CorrectionFailure>() => {
                return Err(e);
            }
            Err(libraw_error) => {
                return load_raw(path, raw)
                    .map_err(|rawler_error| raw_fallback_error(rawler_error, libraw_error));
            }
        }
    }
    if extension(path) == "exr" {
        let img = exr::prelude::read_first_rgba_layer_from_file(
            path,
            |size, _| ImageBuf::new(size.width() as u32, size.height() as u32),
            |image, pos, (r, g, b, _): (f32, f32, f32, f32)| {
                let i = (pos.y() * image.width as usize + pos.x()) * 3;
                image.data[i..i + 3].copy_from_slice(&[
                    from_f64(r as f64),
                    from_f64(g as f64),
                    from_f64(b as f64),
                ]);
            },
        )?;
        return Ok(LoadedImage {
            pixels: img.layer_data.channel_data.pixels,
            metadata: Metadata::default(),
            color_space: "ProPhoto RGB".into(),
            encoded: false,
        });
    }
    let mut reader = image::ImageReader::open(path)?.with_guessed_format()?;
    reader.no_limits();
    let mut decoder = reader.into_decoder()?;
    // image's TIFF adapter limits generic tag decoding to the pixel-buffer
    // size. A small image can have a larger ICC / XMP packet; use the TIFF
    // decoder's metadata budget instead so valid metadata does not fail load.
    let (tiff_xmp, tiff_icc) = if matches!(extension(path).as_str(), "tif" | "tiff") {
        let mut tags = tiff::decoder::Decoder::new(File::open(path)?)?;
        (
            tags.get_tag_u8_vec(tiff::tags::Tag::Unknown(700)).ok(),
            tags.get_tag_u8_vec(tiff::tags::Tag::Unknown(34675)).ok(),
        )
    } else {
        (None, None)
    };
    let metadata = Metadata {
        native: little_exif::metadata::Metadata::new_from_path(path).ok(),
        jpeg_segments: jpeg_metadata_segments(path),
        exif: decoder.exif_metadata()?,
        xmp: if matches!(extension(path).as_str(), "tif" | "tiff") {
            tiff_xmp
        } else {
            decoder.xmp_metadata()?
        },
        iptc: decoder.iptc_metadata()?,
    };
    let embedded_icc = decoder.icc_profile()?.or(tiff_icc);
    let orientation = decoder.orientation()?;
    let mut image = image::DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    let default_encoded = matches!(extension(path).as_str(), "png" | "jpg" | "jpeg");
    let tagged = embedded_icc.as_deref().and_then(identify_icc);
    let (color_space, encoded) = tagged.unwrap_or((
        if default_encoded {
            "sRGB"
        } else {
            "ProPhoto RGB"
        },
        default_encoded,
    ));
    let rgb = image.to_rgb32f();
    let (w, h) = (rgb.width(), rgb.height());
    // Standard raster input stays encoded, so a caller's input-space selection
    // is respected and the core decodes exactly once.
    Ok(LoadedImage {
        pixels: ImageBuf::from_data(
            w,
            h,
            rgb.into_raw()
                .into_iter()
                .map(|v| from_f64(v as f64))
                .collect(),
        ),
        metadata,
        color_space: color_space.into(),
        encoded,
    })
}

fn raw_fallback_error(rawler_error: anyhow::Error, libraw_error: anyhow::Error) -> anyhow::Error {
    if rawler_error.is::<lens::CorrectionFailure>() {
        return rawler_error;
    }
    anyhow::anyhow!(
        "RAW decoding failed. rawler: {rawler_error:#}. LibRaw: {libraw_error:#}. \
         For newer cameras, install a current LibRaw and select Auto, or set SPEKTRAFILM_LIBRAW_LIBRARY to its library path."
    )
}
fn identify_icc(profile: &[u8]) -> Option<(&'static str, bool)> {
    let known: &[(&[u8], &str, bool)] = &[
        (
            include_bytes!("../../../data/icc/ellelstone/sRGB-elle-V2-srgbtrc.icc").as_slice(),
            "sRGB",
            true,
        ),
        (
            include_bytes!("../../../data/icc/ellelstone/sRGB-elle-V2-g10.icc").as_slice(),
            "sRGB",
            false,
        ),
        (
            include_bytes!("../../../data/icc/ellelstone/LargeRGB-elle-V2-g18.icc").as_slice(),
            "ProPhoto RGB",
            true,
        ),
        (
            include_bytes!("../../../data/icc/ellelstone/LargeRGB-elle-V2-g10.icc").as_slice(),
            "ProPhoto RGB",
            false,
        ),
        (
            include_bytes!("../../../data/icc/ellelstone/ClayRGB-elle-V2-g22.icc").as_slice(),
            "Adobe RGB (1998)",
            true,
        ),
        (
            include_bytes!("../../../data/icc/ellelstone/ClayRGB-elle-V2-g10.icc").as_slice(),
            "Adobe RGB (1998)",
            false,
        ),
        (
            include_bytes!("../../../data/icc/ellelstone/Rec2020-elle-V2-rec709.icc").as_slice(),
            "ITU-R BT.2020",
            true,
        ),
        (
            include_bytes!("../../../data/icc/ellelstone/Rec2020-elle-V2-g10.icc").as_slice(),
            "ITU-R BT.2020",
            false,
        ),
        (
            include_bytes!("../../../data/icc/ellelstone/Rec709-elle-V2-rec709.icc").as_slice(),
            "ITU-R BT.709",
            true,
        ),
        (
            include_bytes!("../../../data/icc/ellelstone/ACES-elle-V2-g10.icc").as_slice(),
            "ACES2065-1",
            false,
        ),
        (
            include_bytes!("../../../data/icc/ellelstone/ACEScg-elle-V2-g10.icc").as_slice(),
            "ACEScg",
            false,
        ),
        (
            include_bytes!("../../../data/icc/saucecontrol/DisplayP3-v2-micro.icc").as_slice(),
            "Display P3",
            true,
        ),
        (
            include_bytes!("../../../data/icc/saucecontrol/DCI-P3-v4.icc").as_slice(),
            "DCI-P3",
            true,
        ),
    ];
    known
        .iter()
        .find(|(bytes, _, _)| *bytes == profile)
        .map(|(_, name, encoded)| (*name, *encoded))
}

fn load_raw(path: &Path, options: &RawOptions) -> Result<LoadedImage> {
    use rawler::imgop::develop::{Intermediate, ProcessingStep, RawDevelop};
    let mut raw = rawler::decode_file(path).map_err(|e| anyhow::anyhow!("RAW decode: {e:?}"))?;
    if !["camera", "as_shot", "daylight", "tungsten", "custom"]
        .contains(&options.white_balance.as_str())
    {
        bail!("unknown RAW white balance mode: {}", options.white_balance);
    }
    if options.white_balance != "camera" && options.white_balance != "as_shot" {
        // A camera-space neutral from the selected Planckian white point.
        let t = if options.white_balance == "daylight" {
            6504.
        } else if options.white_balance == "tungsten" {
            2850.
        } else {
            options.temperature
        }
        .clamp(1667., 25000.);
        let x = if t <= 4000. {
            -0.2661239e9 / t.powi(3) - 0.2343589e6 / t.powi(2) + 0.8776956e3 / t + 0.179910
        } else {
            -3.0258469e9 / t.powi(3) + 2.1070379e6 / t.powi(2) + 0.2226347e3 / t + 0.240390
        };
        let y = if t <= 2222. {
            -1.1063814 * x.powi(3) - 1.34811020 * x * x + 2.18555832 * x - 0.20219683
        } else if t <= 4000. {
            -0.9549476 * x.powi(3) - 1.37418593 * x * x + 2.09137015 * x - 0.16748867
        } else {
            3.0817580 * x.powi(3) - 5.87338670 * x * x + 3.75112997 * x - 0.37001483
        };
        let xyz = [x / y, 1., (1. - x - y) / y];
        for c in 0..4 {
            let neutral: f64 = (0..3).map(|i| raw.xyz_to_cam[c][i] as f64 * xyz[i]).sum();
            raw.wb_coeffs[c] = if neutral > 1e-9 {
                (1. / neutral) as f32
            } else {
                1.
            };
        }
        let g = raw.wb_coeffs[1];
        for v in &mut raw.wb_coeffs {
            *v /= g;
        }
    }
    let mut dev = RawDevelop::default();
    dev.steps.retain(|s| !matches!(s, ProcessingStep::SRgb));
    let developed = dev
        .develop_intermediate(&raw)
        .map_err(|e| anyhow::anyhow!("RAW develop: {e:?}"))?;
    let (w, h, data) = match developed {
        Intermediate::Monochrome(p) => (
            p.width,
            p.height,
            p.data.into_iter().flat_map(|v| [v; 3]).collect::<Vec<_>>(),
        ),
        Intermediate::ThreeColor(p) => (p.width, p.height, p.data.into_iter().flatten().collect()),
        Intermediate::FourColor(p) => (
            p.width,
            p.height,
            p.data
                .into_iter()
                .flat_map(|v| [v[0], v[1], v[2]])
                .collect(),
        ),
    };
    let mut pixels = ImageBuf::from_data(
        w as u32,
        h as u32,
        data.into_iter().map(|v| from_f64(v as f64)).collect(),
    );
    if options.tint != 1. {
        for p in pixels.data.chunks_exact_mut(3) {
            p[1] *= from_f64(options.tint);
        }
    }
    if options.lens_correction {
        lens::correct(&mut pixels, path, options.lens_database.as_deref())?;
    }
    Ok(LoadedImage {
        pixels,
        metadata: Metadata {
            native: little_exif::metadata::Metadata::new_from_path(path)
                .or_else(|_| {
                    little_exif::metadata::Metadata::new_from_vec(
                        &std::fs::read(path).unwrap_or_default(),
                        little_exif::filetype::FileExtension::TIFF,
                    )
                })
                .ok(),
            ..Default::default()
        },
        color_space: "sRGB".into(),
        encoded: false,
    })
}

fn icc(data_dir: &Path, options: &SaveOptions) -> Option<Vec<u8>> {
    if options.encoded {
        let compact = match options.color_space.as_str() {
            "Display P3" => Some("DisplayP3-v2-micro.icc"),
            "DCI-P3" => Some("DCI-P3-v4.icc"),
            _ => None,
        };
        if let Some(name) = compact {
            return std::fs::read(data_dir.join("icc/saucecontrol").join(name)).ok();
        }
    }
    let stem = match options.color_space.as_str() {
        "sRGB" => "sRGB",
        "ProPhoto RGB" => "LargeRGB",
        "Adobe RGB (1998)" => "ClayRGB",
        "ITU-R BT.2020" => "Rec2020",
        "ACES2065-1" => "ACES",
        "ACEScg" => "ACEScg",
        "ITU-R BT.709" => "Rec709",
        _ => return None,
    };
    let curve = if !options.encoded {
        "g10"
    } else {
        match stem {
            "sRGB" => "srgbtrc",
            "LargeRGB" => "g18",
            "ClayRGB" => "g22",
            "Rec2020" | "Rec709" => "rec709",
            _ => "g10",
        }
    };
    std::fs::read(
        data_dir
            .join("icc/ellelstone")
            .join(format!("{stem}-elle-V2-{curve}.icc")),
    )
    .ok()
}

fn save_pixels(
    image: &ImageBuf,
    path: &Path,
    options: &SaveOptions,
    metadata: &Metadata,
    data_dir: &Path,
) -> Result<()> {
    let ext = extension(path);
    let icc = icc(data_dir, options);
    let w = image.width;
    let h = image.height;
    if ext == "exr" {
        // Float EXR deliberately keeps negative values and highlights above 1.
        if options.bit_depth == 16 {
            exr::prelude::write_rgb_file(path, w as usize, h as usize, |x, y| {
                let i = (y * w as usize + x) * 3;
                (
                    exr::prelude::f16::from_f32(image.data[i] as f32),
                    exr::prelude::f16::from_f32(image.data[i + 1] as f32),
                    exr::prelude::f16::from_f32(image.data[i + 2] as f32),
                )
            })?;
        } else if options.bit_depth == 32 {
            exr::prelude::write_rgb_file(path, w as usize, h as usize, |x, y| {
                let i = (y * w as usize + x) * 3;
                (
                    image.data[i] as f32,
                    image.data[i + 1] as f32,
                    image.data[i + 2] as f32,
                )
            })?;
        } else {
            bail!("EXR bit depth must be 16 or 32");
        }
        return Ok(());
    }
    if ext == "tiff" || ext == "tif" {
        let mut encoder = tiff::encoder::TiffEncoder::new(BufWriter::new(File::create(path)?))?;
        macro_rules! write_tiff {
            ($ty:ty,$data:expr) => {{
                let mut im = encoder.new_image::<$ty>(w, h)?;
                if let Some(icc) = icc {
                    im.encoder()
                        .write_tag(tiff::tags::Tag::Unknown(34675), icc.as_slice())?;
                }
                if let Some(xmp) = metadata.xmp.as_ref() {
                    im.encoder()
                        .write_tag(tiff::tags::Tag::Unknown(700), xmp.as_slice())?;
                }
                im.write_data(&$data)?;
            }};
        }
        match options.bit_depth {
            32 => write_tiff!(
                tiff::encoder::colortype::RGB32Float,
                image.data.iter().map(|v| *v as f32).collect::<Vec<_>>()
            ),
            16 => write_tiff!(
                tiff::encoder::colortype::RGB16,
                image
                    .data
                    .iter()
                    .map(|v| ((*v as f64).clamp(0., 1.) * 65535.).round_ties_even() as u16)
                    .collect::<Vec<_>>()
            ),
            8 => write_tiff!(
                tiff::encoder::colortype::RGB8,
                image
                    .data
                    .iter()
                    .map(|v| ((*v as f64).clamp(0., 1.) * 255.).round_ties_even() as u8)
                    .collect::<Vec<_>>()
            ),
            _ => bail!("TIFF bit depth must be 8, 16, or 32"),
        }
        return Ok(());
    }
    let bytes: Vec<u8> = image
        .data
        .iter()
        .map(|v| ((*v as f64).clamp(0., 1.) * 255.).round_ties_even() as u8)
        .collect();
    let mut writer =
        BufWriter::new(File::create(path).with_context(|| format!("creating {}", path.display()))?);
    macro_rules! encode {
        ($enc:expr) => {{
            let mut enc = $enc;
            if let Some(icc) = icc {
                enc.set_icc_profile(icc)?;
            }
            if let Some(exif) = metadata.exif.clone() {
                enc.set_exif_metadata(exif)?;
            }
            enc.write_image(&bytes, w, h, image::ExtendedColorType::Rgb8)?;
        }};
    }
    match ext.as_str() {
        "png" => encode!(image::codecs::png::PngEncoder::new(&mut writer)),
        "jpg" | "jpeg" => encode!(image::codecs::jpeg::JpegEncoder::new_with_quality(
            &mut writer,
            options.jpeg_quality
        )),
        _ => bail!("unsupported output format: {ext}"),
    }
    Ok(())
}

/// Convert application output to the independently selected saving space.
/// Values remain unbounded until the chosen file format quantizes them.
pub fn convert_color(
    image: &ImageBuf,
    source: &str,
    source_encoded: bool,
    destination: &str,
    destination_encoded: bool,
) -> Result<ImageBuf> {
    use rayon::prelude::*;
    let src = spektrafilm_math::transfer::get(source).map_err(anyhow::Error::msg)?;
    let dst = spektrafilm_math::transfer::get(destination).map_err(anyhow::Error::msg)?;
    if (source_encoded && !src.transfer.supported())
        || (destination_encoded && !dst.transfer.supported())
    {
        bail!("unsupported saving transfer function");
    }
    if source == destination && source_encoded == destination_encoded {
        return Ok(image.clone());
    }
    let matrix = spektrafilm_math::transfer::rgb_matrix(source, destination);
    let mut out = image.clone();
    out.data.par_chunks_exact_mut(3).for_each(|p| {
        let linear = std::array::from_fn::<_, 3, _>(|c| {
            if source_encoded {
                src.transfer.decode(p[c] as f64)
            } else {
                p[c] as f64
            }
        });
        for c in 0..3 {
            let v = matrix[c][0] * linear[0] + matrix[c][1] * linear[1] + matrix[c][2] * linear[2];
            p[c] = from_f64(if destination_encoded {
                dst.transfer.encode(v)
            } else {
                v
            });
        }
    });
    Ok(out)
}
pub fn save_converted(
    image: &ImageBuf,
    source: &str,
    source_encoded: bool,
    path: &Path,
    options: &SaveOptions,
    metadata: &Metadata,
    data_dir: &Path,
) -> Result<()> {
    if source == options.color_space && source_encoded == options.encoded {
        return save(image, path, options, metadata, data_dir);
    }
    let image = convert_color(
        image,
        source,
        source_encoded,
        &options.color_space,
        options.encoded,
    )?;
    save(&image, path, options, metadata, data_dir)
}

/// Save pixels, then merge source descriptive tags into the output's own IFDs.
/// Structural TIFF/thumbnail offsets are never transplanted from the source.
pub fn save(
    image: &ImageBuf,
    path: &Path,
    options: &SaveOptions,
    metadata: &Metadata,
    data_dir: &Path,
) -> Result<()> {
    let mut metadata = metadata.clone();
    if extension(path) != "exr" {
        metadata.xmp = Some(xmp::saving_profile(
            metadata.xmp.as_deref(),
            &options.color_space,
            options.encoded,
        )?);
        if matches!(extension(path).as_str(), "jpg" | "jpeg") {
            metadata.jpeg_segments.retain(|s| {
                !s.get(4..)
                    .is_some_and(|p| p.starts_with(b"http://ns.adobe.com/xap/1.0/"))
            });
            let mut payload = b"http://ns.adobe.com/xap/1.0/\0".to_vec();
            payload.extend_from_slice(metadata.xmp.as_ref().unwrap());
            let length = u16::try_from(payload.len() + 2)
                .context("XMP packet exceeds standard JPEG segment size")?;
            let mut segment = vec![0xff, 0xe1];
            segment.extend_from_slice(&length.to_be_bytes());
            segment.extend(payload);
            metadata.jpeg_segments.push(segment);
        }
    }
    save_pixels(image, path, options, &metadata, data_dir)?;
    if extension(path) == "exr" {
        return Ok(());
    }
    use little_exif::{exif_tag::ExifTag, metadata::Metadata as Exif};
    // TIFF's newly encoded structural tags must be retained. JPEG/PNG without
    // source EXIF legitimately have no EXIF block yet; avoid asking the parser
    // to decode one (and logging a spurious error) before creating it.
    let mut output =
        if matches!(extension(path).as_str(), "tif" | "tiff") || metadata.exif.is_some() {
            Exif::new_from_path(path).unwrap_or_else(|_| Exif::new())
        } else {
            Exif::new()
        };
    if let Some(source) = &metadata.native {
        for tag in source {
            if !matches!(
                tag.as_u16(),
                0x0100
                    | 0x0101
                    | 0x0102
                    | 0x0103
                    | 0x0106
                    | 0x0111
                    | 0x0115
                    | 0x0116
                    | 0x0117
                    | 0x011c
                    | 0x013d
                    | 0x0140
                    | 0x0142
                    | 0x0143
                    | 0x014a
                    | 0x0152
                    | 0x0153
                    | 0x0144
                    | 0x0145
                    | 0x0201
                    | 0x0202
                    | 0x8769
                    | 0x8825
                    | 0xa005
                    | 0x8773
                    | 0x02bc
            ) {
                output.set_tag(tag.clone());
            }
        }
    }
    output.set_tag(ExifTag::Orientation(vec![1]));
    output.set_tag(ExifTag::Software("spektrafilm-rs".into()));
    output.set_tag(ExifTag::ExifImageWidth(vec![image.width]));
    output.set_tag(ExifTag::ExifImageHeight(vec![image.height]));
    output.set_tag(ExifTag::ColorSpace(vec![
        if options.color_space == "sRGB" && options.encoded {
            1
        } else {
            65535
        },
    ]));
    output.remove_tag(ExifTag::InteroperabilityIndex(String::new()));
    if options.encoded {
        match options.color_space.as_str() {
            "sRGB" => output.set_tag(ExifTag::InteroperabilityIndex("R98".into())),
            "Adobe RGB (1998)" => output.set_tag(ExifTag::InteroperabilityIndex("R03".into())),
            _ => {}
        }
    }
    output
        .write_to_file(path)
        .context("writing output metadata")?;
    if matches!(extension(path).as_str(), "jpg" | "jpeg") && !metadata.jpeg_segments.is_empty() {
        let bytes = std::fs::read(path)?;
        let mut result = Vec::with_capacity(
            bytes.len() + metadata.jpeg_segments.iter().map(Vec::len).sum::<usize>(),
        );
        // Keep the output's EXIF APP1 ahead of XMP: some EXIF readers
        // mistakenly interpret the first APP1 as TIFF regardless of its ID.
        let mut offset = 2usize;
        while offset + 4 <= bytes.len() && bytes[offset] == 0xff {
            let marker = bytes[offset + 1];
            if marker == 0xda || marker == 0xd9 {
                break;
            }
            let length = u16::from_be_bytes([bytes[offset + 2], bytes[offset + 3]]) as usize;
            if length < 2 || offset + 2 + length > bytes.len() {
                break;
            }
            offset += 2 + length;
        }
        result.extend_from_slice(&bytes[..offset]);
        for segment in &metadata.jpeg_segments {
            result.extend_from_slice(segment);
        }
        result.extend_from_slice(&bytes[offset..]);
        std::fs::write(path, result)?;
    }
    if extension(path) == "png" {
        if let Some(xmp) = &metadata.xmp {
            let bytes = std::fs::read(path)?;
            let mut payload = b"XML:com.adobe.xmp\0\0\0\0\0".to_vec();
            payload.extend_from_slice(xmp);
            let mut chunk = Vec::new();
            chunk.extend_from_slice(&(payload.len() as u32).to_be_bytes());
            chunk.extend_from_slice(b"iTXt");
            chunk.extend_from_slice(&payload);
            chunk.extend_from_slice(&crc32fast::hash(&chunk[4..]).to_be_bytes());
            // Metadata readers inspect PNG headers before decoding IDAT.
            // Put XMP before the first image-data chunk, not after it.
            let mut offset = 8usize;
            while offset + 12 <= bytes.len() {
                let length =
                    u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
                if &bytes[offset + 4..offset + 8] == b"IDAT" {
                    break;
                }
                offset = offset
                    .checked_add(length + 12)
                    .context("invalid PNG chunk length")?;
            }
            anyhow::ensure!(offset + 12 <= bytes.len(), "PNG has no image-data chunk");
            let mut out = bytes[..offset].to_vec();
            out.extend(chunk);
            out.extend_from_slice(&bytes[offset..]);
            std::fs::write(path, out)?;
        }
    }
    Ok(())
}
fn jpeg_metadata_segments(path: &Path) -> Vec<Vec<u8>> {
    if !matches!(extension(path).as_str(), "jpg" | "jpeg") {
        return Vec::new();
    }
    let Ok(data) = std::fs::read(path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut i = 2;
    while i + 4 <= data.len() && data[i] == 0xff {
        let marker = data[i + 1];
        if marker == 0xda || marker == 0xd9 {
            break;
        }
        let len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
        if len < 2 || i + 2 + len > data.len() {
            break;
        }
        let payload = &data[i + 4..i + 2 + len];
        if marker == 0xed || (marker == 0xe1 && payload.starts_with(b"http://ns.adobe.com/")) {
            out.push(data[i..i + 2 + len].to_vec());
        }
        i += 2 + len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn float_exports_preserve_highlights_and_metadata() {
        use little_exif::{exif_tag::ExifTag, metadata::Metadata as Exif};
        let dir = std::env::temp_dir().join(format!("spektrafilm-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let image = ImageBuf::from_data(
            2,
            1,
            vec![
                from_f64(-0.25),
                from_f64(2.),
                from_f64(0.125),
                from_f64(0.5),
                from_f64(1.),
                from_f64(4.),
            ],
        );
        let options = SaveOptions {
            bit_depth: 32,
            encoded: false,
            ..Default::default()
        };
        let mut exif = Exif::new();
        exif.set_tag(ExifTag::ImageDescription("preserve me".into()));
        let metadata = Metadata {
            native: Some(exif),
            ..Default::default()
        };
        for ext in ["tif", "exr"] {
            let path = dir.join(format!("float.{ext}"));
            save(&image, &path, &options, &metadata, &dir).unwrap();
            let loaded = load(&path, &RawOptions::default()).unwrap();
            assert_eq!(loaded.pixels.data, image.data, "{ext} lost float values");
            if ext == "tif" {
                let tags = Exif::new_from_path(&path).unwrap();
                assert!(
                    tags.into_iter()
                        .any(|tag| matches!(tag,ExifTag::ImageDescription(s) if s=="preserve me"))
                );
            }
        }
        // Source pixel layout must not override the new float TIFF layout.
        let source = dir.join("source.tif");
        save(
            &image,
            &source,
            &SaveOptions {
                bit_depth: 8,
                ..Default::default()
            },
            &Metadata::default(),
            &dir,
        )
        .unwrap();
        let source = load(&source, &RawOptions::default()).unwrap();
        let converted = dir.join("converted.tif");
        save(&image, &converted, &options, &source.metadata, &dir).unwrap();
        assert_eq!(
            load(&converted, &RawOptions::default())
                .unwrap()
                .pixels
                .data,
            image.data
        );
        let png = dir.join("metadata.png");
        save(&image, &png, &SaveOptions::default(), &metadata, &dir).unwrap();
        let loaded = Exif::new_from_path(&png).unwrap();
        assert!(
            loaded
                .into_iter()
                .any(|tag| matches!(tag,ExifTag::Orientation(v) if v==&vec![1]))
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
