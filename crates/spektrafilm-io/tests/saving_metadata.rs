use little_exif::{exif_tag::ExifTag, metadata::Metadata as Exif};
use spektrafilm_io::{Metadata, RawOptions, SaveOptions, load, save};
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
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
            let xmp = String::from_utf8(loaded.metadata.xmp.unwrap_or_else(||panic!("missing XMP in {ext} {space}"))).unwrap();
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
