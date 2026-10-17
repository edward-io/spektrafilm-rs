//! Match the source's density-pinned channel sweeps and system diagnostics.
use spektrafilm_core::qa_patterns;
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
#[test]
fn multi_pin_characteristic_probe_matches_unmodified_python() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/characteristic.json")).unwrap();
    let n = f["n"].as_u64().unwrap() as usize;
    let table = ImageBuf::from_data(
        n as u32,
        (n * n) as u32,
        f["table"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| from_f64(v.as_f64().unwrap()))
            .collect(),
    );
    let probe = qa_patterns::characteristic_probe(&table, n).unwrap();
    let tolerance = if cfg!(feature = "precision-f64") {
        1e-10
    } else {
        1e-6
    };
    let close = |a: f64, b: f64| assert!((a - b).abs() < tolerance, "characteristic {a} vs {b}");
    for i in 0..5 {
        close(probe.pin_codes[i], f["pin_codes"][i].as_f64().unwrap());
    }
    close(
        probe.system_gamma_at_mid,
        f["summary"]["system_gamma_at_mid"].as_f64().unwrap(),
    );
    close(
        probe.max_channel_density_spread,
        f["summary"]["max_channel_density_spread"].as_f64().unwrap(),
    );
    for axis in 0..3 {
        for pin in 0..5 {
            for sample in 0..65 {
                for c in 0..3 {
                    close(
                        probe.sweeps[axis][pin][sample][c],
                        f["sweeps"][axis][pin][sample][c].as_f64().unwrap(),
                    );
                }
            }
        }
    }
}
