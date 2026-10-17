//! Wide-gamut stimuli must preserve their negative components into the runtime.
use spektrafilm_core::{lut_export, qa_style};
#[test]
fn target_stress_gradients_match_source_cat16_and_gamut_statistics() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/edge_stress.json")).unwrap();
    for case in f.as_array().unwrap() {
        let input = lut_export::space(case["input_space"].as_str().unwrap()).unwrap();
        let target = lut_export::space(case["target"].as_str().unwrap()).unwrap();
        let actual = qa_style::edge_input(target, input, 24, 16).unwrap();
        let mut max = 0f64;
        for (i, p) in actual.linear.iter().enumerate() {
            for c in 0..3 {
                max = max.max((p[c] - case["input"][i][c].as_f64().unwrap()).abs());
            }
        }
        assert!(
            max < 2e-6,
            "{} → {} stress stimulus {max}",
            target.name,
            input.name
        );
        for (name, value) in [
            ("oog_fraction_total", actual.oog_fraction_total),
            (
                "oog_fraction_saturated_row",
                actual.oog_fraction_saturated_row,
            ),
        ] {
            assert!(
                (value - case["stats"][name].as_f64().unwrap()).abs() < 1e-10,
                "{} → {} {name}: {value} vs {}",
                target.name,
                input.name,
                case["stats"][name]
            );
        }
    }
}
