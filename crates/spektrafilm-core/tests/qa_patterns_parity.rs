//! Stimuli must match the source suite before comparing diagnostic metrics.
use spektrafilm_core::{lut_export, qa_patterns};
#[test]
fn daylight_planckian_stimuli_match_upstream_transports() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qa_patterns.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let name = case["space"].as_str().unwrap();
        let space = lut_export::space(name).unwrap();
        let (samples, cct) =
            qa_patterns::planckian_sweep(space, 16, case["gain"].as_f64()).unwrap();
        let mut max = 0f64;
        for (i, sample) in samples.iter().enumerate() {
            assert!((cct[i] - case["cct"][i].as_f64().unwrap()).abs() < 1e-10);
            for c in 0..3 {
                max = max.max((sample[c] - case["samples"][i][c].as_f64().unwrap()).abs());
            }
        }
        let dynamic = qa_patterns::dynamic_range_ramp(space).unwrap();
        for i in 0..257 {
            assert!(
                (dynamic.stops[i] - case["dynamic"]["stops"][i].as_f64().unwrap()).abs() < 1e-10,
                "{name}: dynamic exposure endpoint"
            );
            assert_eq!(
                dynamic.clipped[i],
                case["dynamic"]["clipped"][i].as_bool().unwrap()
            );
            for c in 0..3 {
                assert!(
                    (dynamic.encoded[i][c] - case["dynamic"]["encoded"][i][c].as_f64().unwrap())
                        .abs()
                        < 2e-6
                );
            }
        }
        eprintln!("{name} Planckian stimulus error {max}");
        assert!(max < 2e-6, "{name}: QA stimulus drift {max}");
    }
}
