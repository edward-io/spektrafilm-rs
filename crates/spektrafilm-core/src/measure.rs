//! Profile-analysis helpers corresponding to `spektrafilm.utils.measure`.
//! All calculations retain f64 independently of render storage precision.
use serde::Serialize;

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
    let n = b.len();
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .unwrap();
        if a[pivot][k].abs() < 1e-24 {
            return Err("singular measurement fit".into());
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        for i in k + 1..n {
            let factor = a[i][k] / a[k][k];
            for j in k + 1..n {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }
    let mut x = vec![0.; n];
    for i in (0..n).rev() {
        x[i] = (b[i] - (i + 1..n).map(|j| a[i][j] * x[j]).sum::<f64>()) / a[i][i];
    }
    Ok(x)
}
struct Spline {
    x: Vec<f64>,
    y: Vec<f64>,
    second: Vec<f64>,
}
impl Spline {
    fn new(mut points: Vec<(f64, f64)>) -> Result<Self, String> {
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        if points.len() < 2
            || points.iter().any(|(x, y)| !x.is_finite() || !y.is_finite())
            || points.windows(2).any(|w| w[1].0 <= w[0].0)
        {
            return Err("cubic measurement needs at least two finite, distinct knots".into());
        }
        let (x, y): (Vec<_>, Vec<_>) = points.into_iter().unzip();
        let n = x.len();
        let second = if n == 2 {
            vec![0.; 2]
        } else if n == 3 {
            vec![
                2. * ((y[2] - y[1]) / (x[2] - x[1]) - (y[1] - y[0]) / (x[1] - x[0])) / (x[2] - x[0]);
                3
            ]
        } else {
            let h: Vec<_> = x.windows(2).map(|w| w[1] - w[0]).collect();
            let mut a = vec![vec![0.; n]; n];
            let mut rhs = vec![0.; n];
            a[0][0] = -h[1];
            a[0][1] = h[0] + h[1];
            a[0][2] = -h[0];
            for i in 1..n - 1 {
                a[i][i - 1] = h[i - 1];
                a[i][i] = 2. * (h[i - 1] + h[i]);
                a[i][i + 1] = h[i];
                rhs[i] = 6. * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
            }
            a[n - 1][n - 3] = -h[n - 2];
            a[n - 1][n - 2] = h[n - 3] + h[n - 2];
            a[n - 1][n - 1] = -h[n - 3];
            solve(a, rhs)?
        };
        Ok(Self { x, y, second })
    }
    fn evaluate(&self, x: f64) -> f64 {
        let i = self
            .x
            .partition_point(|v| *v <= x)
            .saturating_sub(1)
            .min(self.x.len() - 2);
        let h = self.x[i + 1] - self.x[i];
        let a = (self.x[i + 1] - x) / h;
        let b = (x - self.x[i]) / h;
        a * self.y[i]
            + b * self.y[i + 1]
            + ((a * a * a - a) * self.second[i] + (b * b * b - b) * self.second[i + 1]) * h * h / 6.
    }
}
fn shape(exposure: &[f64], density: &[[f64; 3]]) -> Result<(), String> {
    if exposure.len() != density.len()
        || exposure.len() < 2
        || exposure.iter().any(|v| !v.is_finite())
    {
        return Err(
            "measurement needs matching exposure/density arrays and finite exposures".into(),
        );
    }
    Ok(())
}
/// Average density/log-exposure slope between two densities, using the
/// reference's inverse not-a-knot cubic interpolation (default 0.25 to 1.0).
pub fn measure_gamma(
    exposure: &[f64],
    density: &[[f64; 3]],
    density_0: f64,
    density_1: f64,
) -> Result<[f64; 3], String> {
    shape(exposure, density)?;
    if !density_0.is_finite() || !density_1.is_finite() || density_0 == density_1 {
        return Err("gamma requires distinct finite density targets".into());
    }
    let mut out = [0.; 3];
    for c in 0..3 {
        let spline = Spline::new(
            exposure
                .iter()
                .zip(density)
                .map(|(&x, d)| (d[c], x))
                .collect(),
        )?;
        if density_0.min(density_1) < spline.x[0]
            || density_0.max(density_1) > *spline.x.last().unwrap()
        {
            return Err(format!(
                "channel {c}: target density outside measured curve"
            ));
        }
        out[c] =
            (density_1 - density_0) / (spline.evaluate(density_1) - spline.evaluate(density_0));
    }
    Ok(out)
}
/// Average local slope in a log-exposure window; unlike inverse gamma,
/// forward splines omit NaN density samples and extrapolate at either end.
pub fn measure_slopes_at_exposure(
    exposure: &[f64],
    density: &[[f64; 3]],
    reference: f64,
    range: f64,
) -> Result<[f64; 3], String> {
    shape(exposure, density)?;
    if !reference.is_finite() || !range.is_finite() || range <= 0. {
        return Err("slope window must be finite and positive".into());
    }
    let mut out = [0.; 3];
    for c in 0..3 {
        let spline = Spline::new(
            exposure
                .iter()
                .zip(density)
                .filter(|(_, d)| !d[c].is_nan())
                .map(|(&x, d)| (x, d[c]))
                .collect(),
        )?;
        out[c] = (spline.evaluate(reference + range / 2.)
            - spline.evaluate(reference - range / 2.))
            / range;
    }
    Ok(out)
}
#[derive(Clone, Debug, Serialize)]
pub struct ToeFit {
    /// d_max, exposure midpoint, density floor, erf steepness.
    pub parameters: [f64; 4],
    pub samples: usize,
    pub rms_residual: f64,
}
impl ToeFit {
    pub fn density(&self, exposure: f64, positive: bool) -> f64 {
        toe(exposure, self.parameters, positive).0
    }
}
fn toe(x: f64, p: [f64; 4], positive: bool) -> (f64, [f64; 4]) {
    let sign = if positive { -1. } else { 1. };
    let u = sign * p[3] * (x - p[1]);
    let h = 0.5 * (1. + libm::erf(u));
    let d = p[0] / std::f64::consts::PI.sqrt() * (-u * u).exp();
    (
        p[0] * h + p[2],
        [h, -d * sign * p[3], 1., d * sign * (x - p[1])],
    )
}
/// Bounded four-parameter erf fit to the lowest 10% of negative density
/// (4% for positive), matching the reference's model, starting point and bounds.
/// Returned fit data also supports diagnostic plot exports without Matplotlib.
pub fn fit_density_toes(
    exposure: &[f64],
    density: &[[f64; 3]],
    positive: bool,
) -> Result<[ToeFit; 3], String> {
    shape(exposure, density)?;
    let lower = [if positive { 1.5 } else { 1. }, -2., 0., 0.];
    let upper = [4., 2., if positive { 0.5 } else { 1. }, 4.];
    let mut result = Vec::new();
    for c in 0..3 {
        let min = density
            .iter()
            .map(|d| d[c])
            .filter(|v| v.is_finite())
            .fold(f64::INFINITY, f64::min);
        let max = density
            .iter()
            .map(|d| d[c])
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        let ceiling = min + (max - min) * if positive { 0.04 } else { 0.1 };
        let points: Vec<_> = exposure
            .iter()
            .zip(density)
            .filter(|(_, d)| d[c].is_finite() && d[c] <= ceiling)
            .map(|(&x, d)| (x, d[c]))
            .collect();
        if points.is_empty() {
            return Err(format!("channel {c}: no finite toe samples"));
        }
        let cost = |p: [f64; 4]| {
            points
                .iter()
                .map(|&(x, y)| (y - toe(x, p, positive).0).powi(2))
                .sum::<f64>()
        };
        let mut p = [2., 0., 0.05, 2.];
        let mut best = cost(p);
        let mut lambda = 1e-3;
        for _ in 0..1000 {
            let mut h = vec![vec![0.; 4]; 4];
            let mut g = vec![0.; 4];
            for &(x, y) in &points {
                let (prediction, j) = toe(x, p, positive);
                let residual = y - prediction;
                for i in 0..4 {
                    g[i] += j[i] * residual;
                    for k in 0..4 {
                        h[i][k] += j[i] * j[k];
                    }
                }
            }
            if g.iter().all(|v| v.abs() < 1e-12) {
                break;
            }
            for i in 0..4 {
                h[i][i] += lambda * h[i][i].max(1e-12);
            }
            let step = solve(h, g)?;
            let candidate = std::array::from_fn(|i| (p[i] + step[i]).clamp(lower[i], upper[i]));
            let next = cost(candidate);
            if next < best {
                let improvement = best - next;
                p = candidate;
                best = next;
                lambda = (lambda / 3.).max(1e-15);
                if improvement < 1e-24 {
                    break;
                }
            } else {
                lambda *= 5.;
                if lambda > 1e20 {
                    break;
                }
            }
        }
        result.push(ToeFit {
            parameters: p,
            samples: points.len(),
            rms_residual: (best / points.len() as f64).sqrt(),
        });
    }
    Ok(result.try_into().unwrap())
}
pub fn measure_density_min(
    exposure: &[f64],
    density: &[[f64; 3]],
    positive: bool,
) -> Result<[f64; 3], String> {
    Ok(fit_density_toes(exposure, density, positive)?.map(|fit| fit.parameters[2]))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_independent_scipy_profile_measurements() {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/measurement.json")).unwrap();
        for case in f["cases"].as_array().unwrap() {
            let x: Vec<f64> = serde_json::from_value(case["log_exposure"].clone()).unwrap();
            let y: Vec<[f64; 3]> = serde_json::from_value(case["density_curves"].clone()).unwrap();
            for (key, actual, tolerance) in [
                ("gamma", measure_gamma(&x, &y, 0.25, 1.).unwrap(), 1e-10),
                (
                    "slopes",
                    measure_slopes_at_exposure(&x, &y, 0., 4f64.ln() / 10f64.ln()).unwrap(),
                    1e-10,
                ),
                (
                    "density_min",
                    measure_density_min(&x, &y, case["positive"].as_bool().unwrap()).unwrap(),
                    1e-5,
                ),
            ] {
                let expected: [f64; 3] = serde_json::from_value(case[key].clone()).unwrap();
                for c in 0..3 {
                    assert!(
                        (actual[c] - expected[c]).abs() < tolerance,
                        "{key} channel {c}: {} versus {}",
                        actual[c],
                        expected[c]
                    );
                }
            }
        }
    }
}
