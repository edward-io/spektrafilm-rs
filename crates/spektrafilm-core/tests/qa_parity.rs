use spektrafilm_core::{lut_export, lut_quality, qa_metrics as q};
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-5 * (1. + b.abs()), "{a} vs {b}");
}
#[test]
fn scientific_metrics_match_python() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/qa.json")).unwrap();
    let n = fixture["n"].as_u64().unwrap() as usize;
    let table = ImageBuf::from_data(
        n as u32,
        (n * n) as u32,
        fixture["table"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| from_f64(v.as_f64().unwrap()))
            .collect(),
    );
    let row = |v: &serde_json::Value| std::array::from_fn::<_, 3, _>(|i| v[i].as_f64().unwrap());
    for case in fixture["colors"].as_array().unwrap() {
        let space = lut_export::space(case["space"].as_str().unwrap()).unwrap();
        for i in 0..case["a"].as_array().unwrap().len() {
            let d = q::differences(row(&case["a"][i]), row(&case["b"][i]), space);
            for (c, key) in ["de2000", "itp", "oklab"].iter().enumerate() {
                close(d[c], case[*key][i].as_f64().unwrap());
            }
        }
    }
    for (i, p) in fixture["samples"].as_array().unwrap().iter().enumerate() {
        for (value, key) in [
            (lut_quality::trilinear(&table, n, row(p)), "trilinear"),
            (lut_quality::tetrahedral(&table, n, row(p)), "tetrahedral"),
        ] {
            for c in 0..3 {
                close(value[c], fixture[key][i][c].as_f64().unwrap());
            }
        }
    }
    let space = lut_export::space("sRGB").unwrap();
    for (i, p) in fixture["samples"].as_array().unwrap().iter().enumerate() {
        let noise = q::noise(row(p), &|p| lut_quality::trilinear(&table, n, p), space);
        for (value, key) in [
            (noise.sigma1, "sigma1"),
            (noise.anisotropy, "anisotropy"),
            (noise.sigma_l, "sigma_L"),
            (noise.sigma_ab, "sigma_ab"),
        ] {
            close(value, fixture["noise"][key][i].as_f64().unwrap());
        }
        for r in 0..3 {
            for c in 0..3 {
                close(
                    noise.jacobian[r][c],
                    fixture["noise"]["jacobian"][i][r][c].as_f64().unwrap(),
                );
                close(
                    noise.covariance[r][c],
                    fixture["noise"]["cov_oklab"][i][r][c].as_f64().unwrap(),
                );
            }
        }
    }
    let structure = q::structure(&table, n);
    assert_eq!(
        structure.monotonicity_violations as u64,
        fixture["monotonicity"]["violations"].as_u64().unwrap()
    );
    close(
        structure.worst_negative_diff,
        fixture["monotonicity"]["worst_negative_diff"]
            .as_f64()
            .unwrap(),
    );
    for (i, key) in ["tv_r", "tv_g", "tv_b"].iter().enumerate() {
        close(
            structure.total_variation[i],
            fixture["tv"][*key].as_f64().unwrap(),
        );
    }
    for (i, key) in [
        "axial_highband_ratio_r",
        "axial_highband_ratio_g",
        "axial_highband_ratio_b",
    ]
    .iter()
    .enumerate()
    {
        close(
            structure.fft_highband_ratio[i],
            fixture["fft"][*key].as_f64().unwrap(),
        );
    }
    close(
        structure.jacobian_log_condition.max,
        fixture["condition"]["max"].as_f64().unwrap(),
    );
    close(
        structure.jacobian_log_condition.mean,
        fixture["condition"]["mean"].as_f64().unwrap(),
    );
    assert_eq!(
        structure.surface_flips as u64,
        fixture["surface"]["flips"].as_u64().unwrap()
    );
    let space = lut_export::space("sRGB").unwrap();
    let input = lut_export::grid(n as u32);
    let a: Vec<_> = input
        .pixels()
        .map(|p| q::oklab(q::xyz([p[0] as f64, p[1] as f64, p[2] as f64], space)))
        .collect();
    let b: Vec<_> = table
        .pixels()
        .map(|p| q::oklab(q::xyz([p[0] as f64, p[1] as f64, p[2] as f64], space)))
        .collect();
    close(
        q::hull_volume(&a),
        fixture["hull"]["input_hull_volume"].as_f64().unwrap(),
    );
    close(
        q::hull_volume(&b),
        fixture["hull"]["output_hull_volume"].as_f64().unwrap(),
    );
    let hue = q::hue_rotation(&a, &b);
    for (key, value) in fixture["hue"].as_object().unwrap() {
        close(hue[key].as_f64().unwrap(), value.as_f64().unwrap());
    }
    let floats = |key: &str| {
        fixture[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>()
    };
    let clipped: Vec<_> = fixture["clipped"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();
    let dynamic = q::dynamic_range(&floats("stops"), &floats("y"), &clipped);
    for (key, value) in fixture["dynamic"].as_object().unwrap() {
        close(dynamic[key].as_f64().unwrap(), value.as_f64().unwrap());
    }
}
