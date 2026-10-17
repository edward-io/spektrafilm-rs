use spektrafilm_core::{resize, stages::filming};
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
#[test]
fn metering_matches_python_thumbnail_then_decoding_order() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/metering.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let w = case["size"][0].as_u64().unwrap() as u32;
        let h = case["size"][1].as_u64().unwrap() as u32;
        let checker = case["pattern"].as_str().unwrap() == "checker";
        let image = ImageBuf::from_data(
            w,
            h,
            (0..h)
                .flat_map(|y| {
                    (0..w).flat_map(move |x| {
                        let values = if checker {
                            let v = if (x + y) % 2 == 0 { 0.95 } else { 0.02 };
                            [v, 0.8 * v + 0.04, 0.7 * v + 0.1]
                        } else {
                            std::array::from_fn(|c| {
                                ((x * 7 + y * 13 + c as u32 * 29) % 100) as f64 / 100.
                            })
                        };
                        values.map(|v| from_f64(v as f32 as f64))
                    })
                })
                .collect(),
        );
        let space = case["color_space"].as_str().unwrap();
        let encoded = case["encoded"].as_bool().unwrap();
        let thumbnail = resize::metering_preview(&image, 256);
        assert_eq!(
            (thumbnail.width, thumbnail.height),
            (
                case["thumbnail_size"][0].as_u64().unwrap() as u32,
                case["thumbnail_size"][1].as_u64().unwrap() as u32
            )
        );
        let matrix = filming::input_colorspace_to_xyz(space);
        let transfer = encoded.then(|| spektrafilm_math::transfer::get(space).unwrap().transfer);
        for (method, expected) in case["expected"].as_object().unwrap() {
            let actual =
                filming::measure_autoexposure_ev_with_transfer(&image, &matrix, method, transfer)
                    as f64;
            let error = (actual - expected.as_f64().unwrap()).abs();
            assert!(
                error < 2e-6,
                "{} {space} encoded {encoded}, {method}: {actual}, expected {expected}, error {error}",
                case["pattern"]
            );
        }
    }
}
