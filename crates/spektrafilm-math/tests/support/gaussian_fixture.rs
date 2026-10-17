use spektrafilm_math::{image::ImageBuf, precision::from_f64};
pub fn image() -> ImageBuf {
    ImageBuf::from_data(
        257,
        193,
        (0..193)
            .flat_map(|y| {
                (0..257).flat_map(move |x| {
                    let v = if x == 0 && y == 192 {
                        25.
                    } else {
                        if (x / 5 + y / 7) % 2 == 0 { 0.9 } else { 0.01 }
                    };
                    let v = if x == 0 && y == 192 {
                        v
                    } else {
                        v + 0.0002 * x as f64 + 0.0001 * y as f64
                    };
                    [1., 0.4, 0.1].map(|c| from_f64((v * c) as f32 as f64))
                })
            })
            .collect(),
    )
}
pub fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!("../fixtures/gaussian.json")).unwrap()
}
