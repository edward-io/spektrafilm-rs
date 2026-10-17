//! Experimental film and print chemistry; total curves and grain layers share one fit.
use crate::{params::PrintCurvesMorphParams, profile::DensityCurvesModel};
const SIGMA_FLOOR: f64 = 0.05;
fn norm_cdf(a: f64) -> f64 {
    let x = a * std::f64::consts::FRAC_1_SQRT_2;
    if x.abs() < std::f64::consts::FRAC_1_SQRT_2 {
        0.5 + 0.5 * libm::erf(x)
    } else {
        let y = 0.5 * libm::erfc(x.abs());
        if x > 0. { 1. - y } else { y }
    }
}
fn cdf(z: f64, positive: bool, kind: &str, alpha: f64, mix: f64) -> f64 {
    let z = if positive { -z } else { z };
    let value = if kind == "sept_norm_cdfs" {
        let u = (z / 5.8013 + 0.5).clamp(0., 1.);
        let v = (u + alpha * u * (1. - u) * (2. * u - 1.).powi(2)).clamp(0., 1.);
        v.powi(4) * (35. + v * (-84. + v * (70. - 20. * v)))
    } else {
        norm_cdf(z)
    };
    if mix > 0. {
        let location = -2.0f64.ln().ln();
        let width = 0.5 * 2.0f64.ln() * (2. * std::f64::consts::PI).sqrt();
        (1. - mix) * value + mix * (-(-(z / width + location)).exp()).exp()
    } else {
        value
    }
}
pub fn morph_density_curves(
    log_exposure: &[f64],
    model: &DensityCurvesModel,
    params: &PrintCurvesMorphParams,
    positive: bool,
) -> Result<Vec<[f64; 3]>, String> {
    morph_density_curves_with_layers(log_exposure, model, params, positive).map(|(total, _)| total)
}
pub fn morph_density_curves_with_layers(
    log_exposure: &[f64],
    model: &DensityCurvesModel,
    p: &PrintCurvesMorphParams,
    positive: bool,
) -> Result<(Vec<[f64; 3]>, Vec<Vec<Vec<f64>>>), String> {
    if !["cdfs", "norm_cdfs", "sept_norm_cdfs"].contains(&model.model_type.as_str()) {
        return Err(format!(
            "unsupported density_curves_model type {:?}",
            model.model_type
        ));
    }
    let n = model.n_layers();
    if n == 0 || n > 3 || model.n_channels() != 3 {
        return Err("density model requires 3 channels with 1 to 3 layers".into());
    }
    for (name, rows) in [
        ("centers", &model.centers),
        ("amplitudes", &model.amplitudes),
        ("sigmas", &model.sigmas),
    ] {
        if rows.len() != 3
            || rows
                .iter()
                .any(|r| r.len() != n || r.iter().any(|v| !v.is_finite()))
        {
            return Err(format!("density_curves_model.{name} must be finite 3×{n}"));
        }
    }
    if model.sigmas.iter().flatten().any(|v| *v <= 0.) {
        return Err("density model sigmas must be positive".into());
    }
    if let Some(a) = &model.alphas {
        if a.len() != 3
            || a.iter()
                .any(|r| r.len() != n || r.iter().any(|v| !v.is_finite()))
        {
            return Err("invalid density model alphas".into());
        }
    }
    if p.active
        && (![
            p.gamma_factor,
            p.gamma_factor_fast,
            p.gamma_factor_slow,
            p.gamma_factor_red,
            p.gamma_factor_green,
            p.gamma_factor_blue,
        ]
        .iter()
        .all(|v| v.is_finite() && *v > 0.)
            || !p.developer_exhaustion.is_finite()
            || !(0. ..=1.).contains(&p.developer_exhaustion))
    {
        return Err("chemistry gamma must be positive and developer exhaustion in [0,1]".into());
    }
    let mut total = vec![[0.; 3]; log_exposure.len()];
    let mut layers = vec![vec![vec![0.; 3]; 3]; log_exposure.len()];
    for ch in 0..3 {
        let mut centers = model.centers[ch].clone();
        let mut sigmas = model.sigmas[ch].clone();
        let amplitudes = &model.amplitudes[ch];
        let alpha = |l: usize| model.alphas.as_ref().map_or(0., |a| a[ch][l]);
        let mix = if p.active { p.developer_exhaustion } else { 0. };
        if p.active {
            let mut order: Vec<_> = (0..n).collect();
            order.sort_by(|a, b| centers[*a].total_cmp(&centers[*b]));
            let base = p.gamma_factor
                * [
                    p.gamma_factor_red,
                    p.gamma_factor_green,
                    p.gamma_factor_blue,
                ][ch];
            for (l, g) in [
                (order[0], base * p.gamma_factor_fast),
                (order[n / 2], base * p.gamma_factor_slow),
                (order[n - 1], base * p.gamma_factor_slow),
            ] {
                centers[l] /= g;
                sigmas[l] = (sigmas[l] / g).max(SIGMA_FLOOR);
            }
            if mix > 1e-8 {
                let target: f64 = (0..n)
                    .map(|l| {
                        amplitudes[l]
                            * cdf(
                                -centers[l] / sigmas[l],
                                positive,
                                &model.model_type,
                                alpha(l),
                                0.,
                            )
                    })
                    .sum();
                let residual = |offset: f64| {
                    (0..n)
                        .map(|l| {
                            amplitudes[l]
                                * cdf(
                                    -(centers[l] + offset) / sigmas[l],
                                    positive,
                                    &model.model_type,
                                    alpha(l),
                                    mix,
                                )
                        })
                        .sum::<f64>()
                        - target
                };
                let mut lo = -0.25;
                let mut hi = 0.25;
                for _ in 0..12 {
                    if residual(lo) * residual(hi) <= 0. {
                        break;
                    }
                    lo *= 2.;
                    hi *= 2.;
                }
                if residual(lo) * residual(hi) > 0. {
                    return Err("could not bracket chemistry midgray correction".into());
                }
                let offset = brentq(&residual, lo, hi, 1e-10);
                for center in &mut centers {
                    *center += offset;
                }
            }
        }
        for (i, &x) in log_exposure.iter().enumerate() {
            for l in 0..n {
                let v = amplitudes[l]
                    * cdf(
                        (x - centers[l]) / sigmas[l],
                        positive,
                        &model.model_type,
                        alpha(l),
                        mix,
                    );
                layers[i][l][ch] = v;
                total[i][ch] += v;
            }
        }
    }
    Ok((total, layers))
}

fn brentq(f: &dyn Fn(f64) -> f64, xa: f64, xb: f64, xtol: f64) -> f64 {
    const RTOL: f64 = 4.0 * f64::EPSILON;
    const MAXITER: usize = 100;

    let (mut xpre, mut xcur) = (xa, xb);
    let mut xblk = 0.0;
    let mut fpre = f(xpre);
    let mut fcur = f(xcur);
    let mut fblk = 0.0;
    let mut spre = 0.0;
    let mut scur = 0.0;

    if fpre * fcur > 0.0 {
        return 0.0;
    }
    if fpre == 0.0 {
        return xpre;
    }
    if fcur == 0.0 {
        return xcur;
    }

    for _ in 0..MAXITER {
        if fpre * fcur < 0.0 {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = xcur - xpre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        let delta = (xtol + RTOL * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return xcur;
        }

        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                // interpolate
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // extrapolate
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                // good short step
                spre = scur;
                scur = stry;
            } else {
                // bisect
                spre = sbis;
                scur = sbis;
            }
        } else {
            // bisect
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }
        fcur = f(xcur);
    }
    xcur
}

#[cfg(test)]
mod parity_tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn data_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("data")
    }

    /// s023 print-curve morph regression guard. Morphed density curves for the
    /// kodak_portra_endura fitted model are cross-checked against a faithful
    /// transcription of upstream `morph_curves.py` (scipy `norm.cdf` + `brentq`)
    /// with a non-identity setting that exercises every control, including the
    /// developer-exhaustion offset solve.
    #[test]
    fn morph_matches_python_reference() {
        let dir = data_dir();
        let print = crate::profile::load_profile_by_name(&dir, "kodak_portra_endura").unwrap();
        let model = print.data.density_curves_model.as_ref().unwrap();
        let log_exposure = print.log_exposure_f64();
        let positive = print.is_positive();
        assert!(!positive, "endura is a negative-type print paper");

        let p = PrintCurvesMorphParams {
            development_time: None,
            active: true,
            gamma_factor: 1.1,
            gamma_factor_fast: 0.9,
            gamma_factor_slow: 1.2,
            gamma_factor_red: 1.05,
            gamma_factor_green: 0.95,
            gamma_factor_blue: 1.0,
            developer_exhaustion: 0.3,
        };
        let morphed = morph_density_curves(&log_exposure, model, &p, positive).unwrap();

        let reference = crate::experimental_reference();
        let expected: Vec<[f64; 3]> =
            serde_json::from_value(reference["extra"]["morph"].clone()).unwrap();
        let expect = expected.into_iter().enumerate();
        for (i, want) in expect {
            for c in 0..3 {
                let got = morphed[i][c];
                assert!(
                    (got - want[c]).abs() <= 1e-9 + 1e-9 * want[c].abs(),
                    "row {i} ch {c}: {got} vs {}",
                    want[c]
                );
            }
        }
    }

    /// Identity params reproduce the fitted model (the morph is a no-op at
    /// defaults except `active`).
    #[test]
    fn morph_identity_reproduces_model() {
        let dir = data_dir();
        let print = crate::profile::load_profile_by_name(&dir, "kodak_portra_endura").unwrap();
        let model = print.data.density_curves_model.as_ref().unwrap();
        let log_exposure = print.log_exposure_f64();
        let positive = print.is_positive();

        let identity = PrintCurvesMorphParams {
            active: true,
            ..Default::default()
        };
        let morphed = morph_density_curves(&log_exposure, model, &identity, positive).unwrap();

        // Direct evaluation of the unmorphed model.
        for (i, &x) in log_exposure.iter().enumerate() {
            for c in 0..3 {
                let mut d = 0.0;
                for l in 0..model.n_layers() {
                    let z = (x - model.centers[c][l]) / model.sigmas[c][l];
                    d += model.amplitudes[c][l] * norm_cdf(if positive { -z } else { z });
                }
                assert!((morphed[i][c] - d).abs() < 1e-15, "row {i} ch {c}");
            }
        }
    }

    /// The port only implements the Gaussian `cdfs` model; any other
    /// `model_type` must be refused rather than evaluated as Gaussian.
    #[test]
    fn morph_rejects_unknown_model_type() {
        let model = DensityCurvesModel {
            alphas: None,
            model_type: "unknown".into(),
            centers: vec![vec![0.0; 3]; 3],
            amplitudes: vec![vec![1.0; 3]; 3],
            sigmas: vec![vec![1.0; 3]; 3],
        };
        let p = PrintCurvesMorphParams {
            development_time: None,
            active: true,
            ..Default::default()
        };
        let err = morph_density_curves(&[0.0, 1.0], &model, &p, false).unwrap_err();
        assert!(err.contains("unsupported"), "{err}");
    }
}
