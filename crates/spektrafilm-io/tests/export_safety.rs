use spektrafilm_io::{Metadata, RawOptions, SaveOptions, load, save, save_converted};
use spektrafilm_math::{image::ImageBuf, precision::from_f64, transfer};
use std::{fs, path::Path};

fn data() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}
fn image() -> ImageBuf {
    ImageBuf::from_data(2, 1, [-0.125, 0.18, 2., 0.5, 1., 0.].map(from_f64).to_vec())
}

#[test]
fn failed_exports_preserve_existing_files_and_remove_temporary_files() {
    let dir = tempfile::tempdir().unwrap();
    for (name, depth, metadata) in [
        ("existing.tif", 24, Metadata::default()),
        ("existing.exr", 8, Metadata::default()),
        ("existing.unsupported", 16, Metadata::default()),
        (
            "existing.png",
            16,
            Metadata {
                xmp: Some(b"broken XML".to_vec()),
                ..Default::default()
            },
        ),
    ] {
        let path = dir.path().join(name);
        fs::write(&path, b"previous good export").unwrap();
        let options = SaveOptions {
            bit_depth: depth,
            ..Default::default()
        };
        assert!(save(&image(), &path, &options, &metadata, &data()).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"previous good export");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
        fs::remove_file(path).unwrap();
    }
}

#[test]
fn exr_round_trips_linear_color_and_unbounded_pixels() {
    let dir = tempfile::tempdir().unwrap();
    for name in [
        "sRGB",
        "ProPhoto RGB",
        "Adobe RGB (1998)",
        "ITU-R BT.2020",
        "ACES2065-1",
        "ACEScg",
        "Display P3",
        "DCI-P3",
    ] {
        for depth in [16, 32] {
            let path = dir.path().join("export.exr");
            let options = SaveOptions {
                color_space: name.into(),
                encoded: true,
                bit_depth: depth,
                ..Default::default()
            };
            save_converted(
                &image(),
                name,
                false,
                &path,
                &options,
                &Metadata::default(),
                &data(),
            )
            .unwrap();
            let actual = load(&path, &RawOptions::default()).unwrap();
            assert_eq!(actual.color_space, name);
            assert!(!actual.encoded);
            for (a, b) in actual.pixels.data.iter().zip(image().data) {
                assert!(
                    (*a as f64 - b as f64).abs() < if depth == 16 { 0.001 } else { 1e-6 },
                    "{name}: {a} / {b}"
                );
            }
        }
    }
    let path = dir.path().join("encoded.exr");
    let encoded = ImageBuf::from_data(
        2,
        1,
        image()
            .data
            .iter()
            .map(|v| from_f64(transfer::Transfer::Srgb.encode(*v as f64)))
            .collect(),
    );
    save(
        &encoded,
        &path,
        &SaveOptions {
            bit_depth: 32,
            ..Default::default()
        },
        &Metadata::default(),
        &data(),
    )
    .unwrap();
    let actual = load(&path, &RawOptions::default()).unwrap();
    for (a, b) in actual.pixels.data.iter().zip(image().data) {
        assert!((*a as f64 - b as f64).abs() < 2e-6);
    }
}

#[test]
fn untagged_exr_uses_standard_linear_srgb_and_unknown_primaries_fail() {
    use exr::{meta::attribute::Chromaticities, prelude::*};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("external.exr");
    write_rgb_file(&path, 2, 1, |_, _| (0.1f32, 0.2f32, 0.3f32)).unwrap();
    assert_eq!(
        load(&path, &RawOptions::default()).unwrap().color_space,
        "sRGB"
    );
    let mut exr = Image::from_channels((2, 1), SpecificChannels::rgb(|_| (0.1f32, 0.2f32, 0.3f32)));
    exr.attributes.chromaticities = Some(Chromaticities {
        red: Vec2(0.61, 0.31),
        green: Vec2(0.21, 0.71),
        blue: Vec2(0.1, 0.02),
        white: Vec2(0.3127, 0.329),
    });
    exr.write().to_file(&path).unwrap();
    assert!(
        load(&path, &RawOptions::default())
            .err()
            .unwrap()
            .to_string()
            .contains("unsupported EXR chromaticities")
    );
}

fn tagged_tiff(path: &Path, profile: &[u8], values: &[f32]) {
    let mut enc = tiff::encoder::TiffEncoder::new(fs::File::create(path).unwrap()).unwrap();
    let mut image = enc
        .new_image::<tiff::encoder::colortype::RGB32Float>(2, 1)
        .unwrap();
    image
        .encoder()
        .write_tag(tiff::tags::Tag::Unknown(34675), profile)
        .unwrap();
    image.write_data(values).unwrap();
}

#[test]
fn valid_alternate_icc_profiles_are_converted_and_bad_profiles_are_reported() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tagged.tif");
    let encoded = [0.25, 0.5, 0.75, 0.1, 0.3, 0.6];
    let profile = fs::read(data().join("icc/ellelstone/sRGB-elle-V4-srgbtrc.icc")).unwrap();
    tagged_tiff(&path, &profile, &encoded);
    let actual = load(&path, &RawOptions::default()).unwrap();
    assert_eq!(actual.color_space, "sRGB");
    assert!(!actual.encoded);
    for (a, b) in actual.pixels.data.iter().zip(encoded) {
        let expected = transfer::Transfer::Srgb.decode(b as f64);
        assert!((*a as f64 - expected).abs() < 0.0003, "{a} / {expected}");
    }
    let profile = fs::read(data().join("icc/ellelstone/sRGB-elle-V4-g10.icc")).unwrap();
    let values = [-0.125, 0.18, 2., 0.5, 1., 0.];
    tagged_tiff(&path, &profile, &values);
    let actual = load(&path, &RawOptions::default()).unwrap();
    // The ICC's fixed-point colorants differ slightly from the canonical matrix.
    for (a, b) in actual.pixels.data.iter().zip(values) {
        assert!(
            (*a as f64 - b as f64).abs() < 0.0005 * (b as f64).abs().max(1.),
            "{a} / {b}"
        );
    }
    tagged_tiff(&path, b"not an ICC profile", &encoded);
    assert!(
        load(&path, &RawOptions::default())
            .err()
            .unwrap()
            .to_string()
            .contains("ICC profile")
    );
}
