use spektrafilm_io::*;
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
#[test]
fn saving_transport_matches_colour_and_preserves_icc() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/saving.json")).unwrap();
    let dir = std::env::temp_dir().join(format!("spektrafilm-color-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    for (i, case) in fixtures.as_array().unwrap().iter().enumerate() {
        let input = ImageBuf::from_data(
            2,
            1,
            case["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| from_f64(v.as_f64().unwrap()))
                .collect(),
        );
        let image = convert_color(
            &input,
            case["source"].as_str().unwrap(),
            case["source_encoded"].as_bool().unwrap(),
            case["destination"].as_str().unwrap(),
            case["destination_encoded"].as_bool().unwrap(),
        )
        .unwrap();
        for (got, want) in image.data.iter().zip(case["expected"].as_array().unwrap()) {
            assert!((*got as f64 - want.as_f64().unwrap()).abs() < 2e-6);
        }
        let options = SaveOptions {
            bit_depth: 32,
            color_space: case["destination"].as_str().unwrap().into(),
            encoded: case["destination_encoded"].as_bool().unwrap(),
            ..Default::default()
        };
        let path = dir.join(format!("{i}.tif"));
        save_converted(
            &input,
            case["source"].as_str().unwrap(),
            case["source_encoded"].as_bool().unwrap(),
            &path,
            &options,
            &Metadata::default(),
            &data,
        )
        .unwrap();
        let loaded = load(&path, &RawOptions::default()).unwrap();
        assert_eq!(loaded.color_space, options.color_space);
        assert_eq!(loaded.encoded, options.encoded);
        for (got, want) in loaded.pixels.data.iter().zip(image.data.iter()) {
            assert!(
                (*got as f64 - *want as f64).abs() < 2e-6,
                "float conversion was clipped"
            );
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}
