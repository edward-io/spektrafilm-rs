use little_exif::{exif_tag::ExifTag, metadata::Metadata as Exif};
use spektrafilm_io::{Metadata, RawOptions, SaveOptions, load, save};
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
use std::path::Path;

fn data() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

fn image() -> ImageBuf {
    ImageBuf::from_data(4, 2, vec![from_f64(0.18); 4 * 2 * 3])
}

fn user_comment(metadata: &Metadata) -> Vec<u8> {
    metadata
        .native
        .as_ref()
        .unwrap()
        .into_iter()
        .find_map(|tag| match tag {
            ExifTag::UserComment(value) => Some(value.clone()),
            _ => None,
        })
        .unwrap()
}

fn recipe(metadata: &Metadata) -> serde_json::Value {
    let comment = user_comment(metadata);
    let json = comment.strip_prefix(b"ASCII\0\0\0").unwrap();
    assert!(json.is_ascii());
    serde_json::from_slice(json).unwrap()
}

#[test]
fn exports_omit_makernotes_from_both_parsed_and_encoded_source_exif() {
    use little_exif::rational::uR64;
    let dir = tempfile::tempdir().unwrap();
    let mut exif = Exif::new();
    exif.set_tag(ExifTag::Make("SONY".into()));
    exif.set_tag(ExifTag::Model("Test camera".into()));
    exif.set_tag(ExifTag::LensModel("Test lens".into()));
    exif.set_tag(ExifTag::DateTimeOriginal("2026:03:15 13:02:47".into()));
    exif.set_tag(ExifTag::ExposureTime(vec![uR64 {
        nominator: 1,
        denominator: 800,
    }]));
    exif.set_tag(ExifTag::GPSLatitudeRef("N".into()));
    exif.set_tag(ExifTag::GPSLatitude(vec![
        uR64 {
            nominator: 1,
            denominator: 1
        };
        3
    ]));
    exif.set_tag(ExifTag::UserComment(
        b"ASCII\0\0\0Original comment".to_vec(),
    ));
    exif.set_tag(ExifTag::MakerNote(b"opaque offsets into the RAW".to_vec()));
    let metadata = Metadata {
        exif: Some(exif.encode().unwrap()),
        native: Some(exif),
        ..Default::default()
    };
    for ext in ["jpg", "png", "tif"] {
        let path = dir.path().join(format!("output.{ext}"));
        save(&image(), &path, &SaveOptions::default(), &metadata, &data()).unwrap();
        let loaded = load(&path, &RawOptions::default()).unwrap();
        let tags = loaded.metadata.native.as_ref().unwrap();
        assert!(!tags.into_iter().any(|tag| tag.as_u16() == 0x927c), "{ext}");
        for expected in metadata.native.as_ref().unwrap() {
            if expected.as_u16() != 0x927c {
                assert!(
                    tags.into_iter().any(|tag| tag == expected),
                    "{ext}: {expected:?}"
                );
            }
        }
    }
}

#[test]
fn export_recipes_preserve_unicode_and_original_comments_without_nesting() {
    let dir = tempfile::tempdir().unwrap();
    let original = b"UNICODE\0\xff\xfeA\0\0\0".to_vec();
    let mut exif = Exif::new();
    exif.set_tag(ExifTag::UserComment(original.clone()));
    let source = Metadata {
        native: Some(exif),
        ..Default::default()
    };
    let raw = RawOptions {
        decoder: "rawler".into(),
        temperature: 4200.,
        ..Default::default()
    };
    let crop = spektrafilm_io::crop::Crop {
        size: [0.5, 0.75],
        angle: 2.,
        ..Default::default()
    };
    let params = serde_json::json!({"camera":{"tone":{"shadows":42}},"label":"日本語 🌈"});
    let settings = spektrafilm_io::EditSettings {
        film: "kodak_portra_400",
        paper: "kodak_portra_endura",
        params: &params,
        raw: &raw,
        rotation: 3,
        crop,
    };
    let metadata = source.with_edit_settings(&settings).unwrap();
    let options = SaveOptions {
        color_space: "Display P3".into(),
        ..Default::default()
    };
    for ext in ["jpg", "png", "tif"] {
        let path = dir.path().join(format!("output.{ext}"));
        save(&image(), &path, &options, &metadata, &data()).unwrap();
        let first = load(&path, &RawOptions::default()).unwrap();
        let document = recipe(&first.metadata);
        assert_eq!(document["application"], "spektrafilm");
        assert_eq!(document["schema_version"], 1);
        assert_eq!(document["application_version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(
            document["settings"],
            serde_json::to_value(&settings).unwrap()
        );
        assert_eq!(document["output"], serde_json::to_value(&options).unwrap());
        assert_eq!(document["format"], ext);
        assert_eq!(document["dimensions"], serde_json::json!([4, 2]));
        assert_eq!(
            document["original_user_comment"],
            serde_json::json!(original)
        );
        let next = serde_json::json!({"film":"kodak_gold_200","params":{"camera":{"tone":{"shadows":-10}}}});
        let updated = first.metadata.with_edit_settings(&next).unwrap();
        let second = dir.path().join(format!("second.{ext}"));
        save(&image(), &second, &options, &updated, &data()).unwrap();
        let second = load(&second, &RawOptions::default()).unwrap();
        let document = recipe(&second.metadata);
        assert_eq!(document["settings"], next);
        assert_eq!(
            document["original_user_comment"],
            serde_json::json!(original)
        );
        assert!(!document.to_string().contains("kodak_portra_400"));
    }
}

#[test]
fn oversized_jpeg_recipe_returns_an_error_without_overwriting_the_output() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("existing.jpg");
    std::fs::write(&path, b"existing photo").unwrap();
    let metadata = Metadata::default()
        .with_edit_settings(&"x".repeat(70_000))
        .unwrap();
    let error = save(&image(), &path, &SaveOptions::default(), &metadata, &data()).unwrap_err();
    assert!(
        error.to_string().contains("JPEG's 64 KiB limit"),
        "{error:#}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), b"existing photo");
}

#[test]
fn exr_exports_include_the_same_recipe_as_text_at_both_bit_depths() {
    use exr::{meta::MetaData, prelude::AttributeValue};
    let dir = tempfile::tempdir().unwrap();
    let settings = serde_json::json!({"film":"kodak_gold_200","label":"日本語 🌈"});
    let metadata = Metadata::default().with_edit_settings(&settings).unwrap();
    for bit_depth in [16, 32] {
        let path = dir.path().join(format!("output-{bit_depth}.exr"));
        let options = SaveOptions {
            bit_depth,
            ..Default::default()
        };
        save(&image(), &path, &options, &metadata, &data()).unwrap();
        let meta = MetaData::read_from_file(&path, false).unwrap();
        let value = meta.headers[0]
            .own_attributes
            .other
            .iter()
            .find_map(|(name, value)| {
                if name == "spektrafilm" {
                    Some(value)
                } else {
                    None
                }
            })
            .unwrap();
        let AttributeValue::Text(value) = value else {
            panic!("expected text recipe")
        };
        let document: serde_json::Value = serde_json::from_str(&value.to_string()).unwrap();
        assert_eq!(document["settings"], settings);
        assert_eq!(document["output"]["bit_depth"], bit_depth);
        assert_eq!(document["output"]["encoded"], false);
    }
}

#[test]
fn converted_exports_relabel_xmp_and_interoperability_without_losing_descriptions() {
    let dir = std::env::temp_dir().join(format!("spektrafilm-saving-label-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let image = ImageBuf::from_data(4, 2, vec![from_f64(0.18); 4 * 2 * 3]);
    let mut exif = Exif::new();
    exif.set_tag(ExifTag::InteroperabilityIndex("R98".into()));
    let metadata=Metadata {native:Some(exif),xmp:Some(br#"<r:RDF xmlns:r="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><r:Description xmlns:p="http://ns.adobe.com/photoshop/1.0/" p:ICCProfile="Old space" xmlns:dc="http://purl.org/dc/elements/1.1/" dc:title="Original title"/></r:RDF>"#.to_vec()),..Default::default()};
    for ext in ["png", "jpg", "tif"] {
        for (space, encoded, index) in [
            ("sRGB", true, Some("R98")),
            ("Adobe RGB (1998)", true, Some("R03")),
            ("Display P3", false, None),
        ] {
            let path = dir.join(format!("{space}-{encoded}.{ext}"));
            save(
                &image,
                &path,
                &SaveOptions {
                    color_space: space.into(),
                    encoded,
                    bit_depth: if ext == "jpg" { 8 } else { 16 },
                    ..Default::default()
                },
                &metadata,
                &data,
            )
            .unwrap();
            let loaded = load(&path, &RawOptions::default()).unwrap();
            let xmp = String::from_utf8(
                loaded
                    .metadata
                    .xmp
                    .unwrap_or_else(|| panic!("missing XMP in {ext} {space}")),
            )
            .unwrap();
            assert!(xmp.contains("Original title"));
            assert!(!xmp.contains("Old space"));
            assert!(xmp.contains(&if encoded {
                space.to_owned()
            } else {
                format!("{space} (linear)")
            }));
            let tags = Exif::new_from_path(&path).unwrap();
            let actual = tags.into_iter().find_map(|tag| {
                if let ExifTag::InteroperabilityIndex(v) = tag {
                    Some(v.trim_end_matches('\0').to_owned())
                } else {
                    None
                }
            });
            assert_eq!(actual.as_deref(), index, "{ext} {space}");
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}
