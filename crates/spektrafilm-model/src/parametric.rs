//! Smooth toe / straight-line / shoulder model from `model/parametric.py`.
/// Evaluate RGB density curves. The softplus form remains finite for large
/// exposures where the reference's intermediate `10**x` would overflow.
pub fn parametric_density_curves_model(
    log_exposure: &[f64],
    gamma: [f64; 3],
    log_exposure_0: [f64; 3],
    density_max: [f64; 3],
    toe_size: [f64; 3],
    shoulder_size: [f64; 3],
) -> Result<Vec<[f64; 3]>, String> {
    if !log_exposure.iter().all(|v| v.is_finite())
        || !(0..3).all(|c| {
            gamma[c].is_finite()
                && gamma[c] != 0.
                && log_exposure_0[c].is_finite()
                && density_max[c].is_finite()
                && toe_size[c].is_finite()
                && toe_size[c] > 0.
                && shoulder_size[c].is_finite()
                && shoulder_size[c] > 0.
        })
    {
        return Err("density model needs finite exposures / parameters, nonzero gamma and positive toe / shoulder widths".into());
    }
    let log_exp = |x: f64| {
        let z = x * std::f64::consts::LN_10;
        (z.max(0.) + (-z.abs()).exp().ln_1p()) / std::f64::consts::LN_10
    };
    Ok(log_exposure
        .iter()
        .map(|&exposure| {
            std::array::from_fn(|c| {
                let e = exposure - log_exposure_0[c];
                gamma[c] * toe_size[c] * log_exp(e / toe_size[c])
                    - gamma[c]
                        * shoulder_size[c]
                        * log_exp((e - density_max[c] / gamma[c]) / shoulder_size[c])
            })
        })
        .collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_numpy_parametric_curves_and_extends_to_hdr() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/parametric.json")).unwrap();
        let exposure: Vec<f64> = serde_json::from_value(fixture["log_exposure"].clone()).unwrap();
        let array = |key: &str| serde_json::from_value(fixture[key].clone()).unwrap();
        let result = parametric_density_curves_model(
            &exposure,
            array("gamma"),
            array("log_exposure_0"),
            array("density_max"),
            array("toe_size"),
            array("shoulder_size"),
        )
        .unwrap();
        let expected: Vec<[f64; 3]> = serde_json::from_value(fixture["density"].clone()).unwrap();
        for (a, b) in result.iter().zip(&expected) {
            for c in 0..3 {
                assert!((a[c] - b[c]).abs() < 1e-12);
            }
        }
        let extremes = parametric_density_curves_model(
            &[-1000., 1000.],
            [0.65; 3],
            [0.; 3],
            [2.8; 3],
            [0.2; 3],
            [0.4; 3],
        )
        .unwrap();
        assert!(extremes.iter().flatten().all(|v| v.is_finite()));
        assert!(extremes[0][0].abs() < 1e-12);
        assert!((extremes[1][0] - 2.8).abs() < 1e-10);
    }
}
