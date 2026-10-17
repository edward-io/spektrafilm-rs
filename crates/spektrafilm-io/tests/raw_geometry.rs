use spektrafilm_io::{RawOptions, load};
use std::path::PathBuf;

// Append a replacement IFD so every original strip/data offset stays valid.
// The source fixture is a little-endian, uncompressed 32x32 Bayer DNG.
fn fixture(name: &str, orientation: u16, crop: bool) -> PathBuf {
    let mut bytes = include_bytes!("fixtures/synthetic.dng").to_vec();
    assert_eq!(&bytes[..2], b"II");
    let offset = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let count = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()) as usize;
    let mut entries: Vec<[u8; 12]> = bytes[offset + 2..offset + 2 + count * 12]
        .chunks_exact(12)
        .map(|entry| entry.try_into().unwrap())
        .collect();
    let mut tag = |id: u16, kind: u16, values: &[u32]| {
        let mut entry = [0u8; 12];
        entry[..2].copy_from_slice(&id.to_le_bytes());
        entry[2..4].copy_from_slice(&kind.to_le_bytes());
        entry[4..8].copy_from_slice(&(values.len() as u32).to_le_bytes());
        let value = if values.len() == 1 {
            values[0]
        } else {
            let position = bytes.len() as u32;
            for v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            position
        };
        entry[8..12].copy_from_slice(&value.to_le_bytes());
        entries.retain(|entry| u16::from_le_bytes(entry[..2].try_into().unwrap()) != id);
        entries.push(entry);
    };
    tag(0x0112, 3, &[orientation as u32]);
    if crop {
        tag(50719, 4, &[2, 4]); // DefaultCropOrigin
        tag(50720, 4, &[28, 24]); // DefaultCropSize
    }
    entries.sort_by_key(|entry| u16::from_le_bytes(entry[..2].try_into().unwrap()));
    let directory = bytes.len() as u32;
    bytes[4..8].copy_from_slice(&directory.to_le_bytes());
    bytes.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for entry in entries {
        bytes.extend_from_slice(&entry);
    }
    bytes.extend_from_slice(&0u32.to_le_bytes());
    let path = std::env::temp_dir().join(format!("spektrafilm-{name}-{}.dng", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn rawler_applies_all_exif_orientations_before_clearing_export_orientation() {
    let baseline_path = fixture("orientation-normal", 1, true);
    let options = RawOptions {
        decoder: "rawler".into(),
        ..Default::default()
    };
    let baseline = load(&baseline_path, &options).unwrap().pixels;
    std::fs::remove_file(baseline_path).unwrap();
    for orientation in 2..=8 {
        let path = fixture(&format!("orientation-{orientation}"), orientation, true);
        let actual = load(&path, &options).unwrap().pixels;
        let (w, h) = (baseline.width, baseline.height);
        let dimensions = if orientation >= 5 { (h, w) } else { (w, h) };
        assert_eq!((actual.width, actual.height), dimensions);
        for y in 0..actual.height {
            for x in 0..actual.width {
                let (sx, sy) = match orientation {
                    2 => (w - 1 - x, y),
                    3 => (w - 1 - x, h - 1 - y),
                    4 => (x, h - 1 - y),
                    5 => (y, x),
                    6 => (y, h - 1 - x),
                    7 => (w - 1 - y, h - 1 - x),
                    8 => (w - 1 - y, x),
                    _ => unreachable!(),
                };
                assert_eq!(
                    actual.get(x, y),
                    baseline.get(sx, sy),
                    "orientation {orientation}, {x},{y}"
                );
            }
        }
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn catalog_geometry_overrides_camera_orientation_without_applying_it_twice() {
    let normal_path = fixture("catalog-normal", 1, true);
    let rotated_path = fixture("catalog-rotated", 6, true);
    let options = RawOptions {
        decoder: "rawler".into(),
        ..Default::default()
    };
    let baseline = load(&normal_path, &options).unwrap().pixels;
    let geometry = spektrafilm_io::lightroom::Geometry {
        orientation: 8,
        crop: [0.25, 0., 0.75, 0.5],
        angle: 0.,
    };
    let expected = geometry.apply(baseline, 1).unwrap();
    let loaded = load(
        &rotated_path,
        &RawOptions {
            lightroom_geometry: Some(geometry),
            ..options
        },
    )
    .unwrap();
    assert_eq!(loaded.metadata.source_orientation, 6);
    assert_eq!(
        (loaded.pixels.width, loaded.pixels.height),
        (expected.width, expected.height)
    );
    assert_eq!(loaded.pixels.data, expected.data);
    std::fs::remove_file(normal_path).unwrap();
    std::fs::remove_file(rotated_path).unwrap();
}

#[test]
fn inactive_custom_tint_does_not_change_rawler_camera_white_balance() {
    let path = fixture("inactive-tint", 1, false);
    for mode in ["as_shot", "camera", "daylight", "tungsten", "custom"] {
        let options = RawOptions {
            decoder: "rawler".into(),
            white_balance: mode.into(),
            ..Default::default()
        };
        let baseline = load(&path, &options).unwrap().pixels;
        let changed = load(
            &path,
            &RawOptions {
                tint: 1.3,
                ..options
            },
        )
        .unwrap()
        .pixels;
        if mode == "custom" {
            for (a, b) in baseline.pixels().zip(changed.pixels()) {
                assert_eq!(a[0], b[0]);
                assert!((a[1] as f64 * 1.3 - b[1] as f64).abs() < 1e-6);
                assert_eq!(a[2], b[2]);
            }
        } else {
            assert!(
                baseline.data == changed.data,
                "custom tint leaked into {mode} mode"
            );
        }
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn auto_prefers_rawler_and_preserves_its_geometry_and_pixels() {
    let path = fixture("auto-rawler", 6, true);
    let expected = load(
        &path,
        &RawOptions {
            decoder: "rawler".into(),
            ..Default::default()
        },
    )
    .unwrap();
    let actual = load(&path, &RawOptions::default()).unwrap();
    assert_eq!(actual.color_space, "sRGB");
    assert_eq!(actual.encoded, expected.encoded);
    assert_eq!(
        actual.metadata.source_orientation,
        expected.metadata.source_orientation
    );
    assert_eq!(
        (actual.pixels.width, actual.pixels.height),
        (expected.pixels.width, expected.pixels.height)
    );
    assert_eq!(actual.pixels.data, expected.pixels.data);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn auto_reports_both_decoder_failures_and_explicit_rawler_does_not_fall_back() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("missing.dng");
    let message = format!("{:#}", load(&path, &RawOptions::default()).err().unwrap());
    assert!(
        message.contains("RAW decoding failed. rawler:"),
        "{message}"
    );
    assert!(message.contains("LibRaw:"), "{message}");
    let message = format!(
        "{:#}",
        load(
            &path,
            &RawOptions {
                decoder: "rawler".into(),
                ..Default::default()
            }
        )
        .err()
        .unwrap()
    );
    assert!(!message.contains("LibRaw"), "{message}");
}

#[test]
fn libraw_uses_vendor_crop_before_postprocessing() {
    // This regression requires the C crop API added in LibRaw 0.22.
    if std::env::var_os("SPEKTRAFILM_LIBRAW_LIBRARY").is_none() {
        return;
    }
    let path = fixture("vendor-crop", 1, true);
    let loaded = load(
        &path,
        &RawOptions {
            decoder: "libraw".into(),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!((loaded.pixels.width, loaded.pixels.height), (28, 24));
    std::fs::remove_file(path).unwrap();
}
