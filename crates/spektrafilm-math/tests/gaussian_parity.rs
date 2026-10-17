#[path = "support/gaussian_fixture.rs"]
mod reference;
#[test]
fn gaussian_edges_match_the_python_runtime() {
    let source = reference::image();
    let fixture = reference::fixture();
    for case in fixture["cases"].as_array().unwrap() {
        let sigma = case["sigma"].as_f64().unwrap() as f32;
        let output = spektrafilm_math::gaussian::gaussian_blur(&source, sigma);
        let mut max = 0f64;
        for (point, expected) in fixture["points"]
            .as_array()
            .unwrap()
            .iter()
            .zip(case["expected"].as_array().unwrap())
        {
            let actual = output.get(
                point[0].as_u64().unwrap() as u32,
                point[1].as_u64().unwrap() as u32,
            );
            for c in 0..3 {
                max = max.max((actual[c] as f64 - expected[c].as_f64().unwrap()).abs());
            }
        }
        eprintln!("CPU Gaussian sigma {sigma}: max {max}");
        let tolerance = if cfg!(feature = "precision-f64") {
            2e-6
        } else {
            0.001
        };
        assert!(
            max <= tolerance,
            "Gaussian differs from Python at sigma {sigma}: {max}"
        );
    }
}
