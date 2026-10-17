//! Verify the gradient's color geometry before applying its seeded noise field.
use spektrafilm_core::{lut_export, qa_patterns};
#[test]
fn perceptual_gradient_matches_unmodified_python_for_sdr_log_and_hdr() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/noise_gradient.json")).unwrap();
    for case in f.as_array().unwrap() {
        let name = case["space"].as_str().unwrap();
        let input = lut_export::space(name).unwrap();
        let actual = qa_patterns::noise_gradient_input(input, None, 32, 24);
        let mut max = 0f64;
        for (i, p) in actual.encoded.iter().enumerate() {
            for c in 0..3 {
                max = max.max((p[c] - case["input"][i][c].as_f64().unwrap()).abs());
            }
        }
        eprintln!(
            "{name} noise gradient max {max}, in gamut {}",
            actual.in_gamut_fraction
        );
        assert!(max < 2e-6, "{name}: gradient error {max}");
        assert!(
            (actual.in_gamut_fraction - case["stats"]["in_gamut_fraction"].as_f64().unwrap()).abs()
                < 1e-10
        );
    }
}
