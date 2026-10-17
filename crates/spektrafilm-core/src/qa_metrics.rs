//! Scientific LUT diagnostics; conventions match the Python QA module.
use crate::lut_export::LutSpace;
use serde::Serialize;
use spektrafilm_math::{image::ImageBuf, transfer};
pub fn mul(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| m[i][j] * v[j]).sum())
}
pub fn xyz(rgb: [f64; 3], space: &LutSpace) -> [f64; 3] {
    mul(
        transfer::get(space.primaries).unwrap().to_xyz,
        space.decode_pixel(rgb).map(|v| v / space.output_gain),
    )
}
pub fn oklab(v: [f64; 3]) -> [f64; 3] {
    let lms = mul(
        [
            [0.8189330101, 0.3618667424, -0.1288597137],
            [0.0329845436, 0.9293118715, 0.0361456387],
            [0.0482003018, 0.2643662691, 0.633851707],
        ],
        v,
    )
    .map(f64::cbrt);
    mul(
        [
            [0.2104542553, 0.793617785, -0.0040720468],
            [1.9779984951, -2.428592205, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.808675766],
        ],
        lms,
    )
}
pub fn lab(v: [f64; 3]) -> [f64; 3] {
    let white = [0.3127 / 0.329, 1., (1. - 0.3127 - 0.329) / 0.329];
    let f = std::array::from_fn::<_, 3, _>(|c| {
        let t = v[c] / white[c];
        if t > (6f64 / 29.).powi(3) {
            t.cbrt()
        } else {
            t / (3. * (6f64 / 29.).powi(2)) + 4. / 29.
        }
    });
    [
        116. * f[1] - 16.,
        500. * (f[0] - f[1]),
        200. * (f[1] - f[2]),
    ]
}
pub fn de2000(a: [f64; 3], b: [f64; 3]) -> f64 {
    let c1 = a[1].hypot(a[2]);
    let c2 = b[1].hypot(b[2]);
    let cm = (c1 + c2) / 2.;
    let g = 0.5 * (1. - (cm.powi(7) / (cm.powi(7) + 25f64.powi(7))).sqrt());
    let ap1 = (1. + g) * a[1];
    let ap2 = (1. + g) * b[1];
    let cp1 = ap1.hypot(a[2]);
    let cp2 = ap2.hypot(b[2]);
    let hp = |y: f64, x: f64| y.atan2(x).to_degrees().rem_euclid(360.);
    let h1 = hp(a[2], ap1);
    let h2 = hp(b[2], ap2);
    let dl = b[0] - a[0];
    let dc = cp2 - cp1;
    let mut dh = h2 - h1;
    if cp1 * cp2 == 0. {
        dh = 0.;
    } else if dh > 180. {
        dh -= 360.;
    } else if dh < -180. {
        dh += 360.;
    }
    let d_h = 2. * (cp1 * cp2).sqrt() * (dh.to_radians() / 2.).sin();
    let lm = (a[0] + b[0]) / 2.;
    let cp = (cp1 + cp2) / 2.;
    let hm = if cp1 * cp2 == 0. {
        h1 + h2
    } else if (h1 - h2).abs() <= 180. {
        (h1 + h2) / 2.
    } else if h1 + h2 < 360. {
        (h1 + h2 + 360.) / 2.
    } else {
        (h1 + h2 - 360.) / 2.
    };
    let cos = |d: f64| d.to_radians().cos();
    let t = 1. - 0.17 * cos(hm - 30.) + 0.24 * cos(2. * hm) + 0.32 * cos(3. * hm + 6.)
        - 0.20 * cos(4. * hm - 63.);
    let sl = 1. + 0.015 * (lm - 50.).powi(2) / (20. + (lm - 50.).powi(2)).sqrt();
    let sc = 1. + 0.045 * cp;
    let sh = 1. + 0.015 * cp * t;
    let rotation = -2.
        * (cp.powi(7) / (cp.powi(7) + 25f64.powi(7))).sqrt()
        * (60. * (-((hm - 275.) / 25.).powi(2)).exp())
            .to_radians()
            .sin();
    ((dl / sl).powi(2) + (dc / sc).powi(2) + (d_h / sh).powi(2) + rotation * (dc / sc) * (d_h / sh))
        .max(0.)
        .sqrt()
}
pub fn ictcp(v: [f64; 3]) -> [f64; 3] {
    let rgb = mul(transfer::get("ITU-R BT.2020").unwrap().from_xyz, v);
    let lms = mul(
        [
            [1688. / 4096., 2146. / 4096., 262. / 4096.],
            [683. / 4096., 2951. / 4096., 462. / 4096.],
            [99. / 4096., 309. / 4096., 3688. / 4096.],
        ],
        rgb,
    )
    .map(|v| {
        // colour-science extends ST 2084 by signed powers for diagnostic
        // values outside the physical domain. Preserve that convention.
        let spow = |v: f64, p: f64| v.signum() * v.abs().powf(p);
        let y = spow(v / 10000., 2610. / 16384.);
        spow(
            (3424. / 4096. + 2413. / 128. * y) / (1. + 2392. / 128. * y),
            2523. / 32.,
        )
    });
    mul(
        [
            [0.5, 0.5, 0.],
            [6610. / 4096., -13613. / 4096., 7003. / 4096.],
            [17933. / 4096., -17390. / 4096., -543. / 4096.],
        ],
        lms,
    )
}
pub fn differences(a: [f64; 3], b: [f64; 3], space: &LutSpace) -> [f64; 3] {
    let x = xyz(a, space);
    let y = xyz(b, space);
    let ia = ictcp(x);
    let ib = ictcp(y);
    let oa = oklab(x);
    let ob = oklab(y);
    [
        de2000(lab(x), lab(y)),
        720. * ((ia[0] - ib[0]).powi(2) + 0.25 * (ia[1] - ib[1]).powi(2) + (ia[2] - ib[2]).powi(2))
            .sqrt(),
        (0..3).map(|c| (oa[c] - ob[c]).powi(2)).sum::<f64>().sqrt(),
    ]
}
#[derive(Serialize)]
pub struct Summary {
    pub max: f64,
    pub p99: f64,
    pub p95: f64,
    pub p50: f64,
    pub mean: f64,
}
pub fn summary(values: &[f64]) -> Summary {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n == 0 {
        return Summary {
            max: 0.,
            p99: 0.,
            p95: 0.,
            p50: 0.,
            mean: 0.,
        };
    }
    let percentile = |p: f64| {
        let q = p * (n - 1) as f64;
        let lo = q.floor() as usize;
        let hi = q.ceil() as usize;
        sorted[lo] + (q - lo as f64) * (sorted[hi] - sorted[lo])
    };
    Summary {
        max: sorted[n - 1],
        p99: percentile(0.99),
        p95: percentile(0.95),
        p50: percentile(0.5),
        mean: values.iter().sum::<f64>() / n as f64,
    }
}
pub fn singular_values(j: [[f64; 3]; 3]) -> [f64; 3] {
    let mut a = std::array::from_fn::<_, 3, _>(|r| {
        std::array::from_fn::<_, 3, _>(|c| (0..3).map(|k| j[k][r] * j[k][c]).sum::<f64>())
    });
    for _ in 0..16 {
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-30 {
                continue;
            }
            let tau = (a[q][q] - a[p][p]) / (2. * a[p][q]);
            let t = if tau >= 0. { 1. } else { -1. } / (tau.abs() + (1. + tau * tau).sqrt());
            let c = 1. / (1. + t * t).sqrt();
            let s = t * c;
            let ap = a[p][p];
            let aq = a[q][q];
            let off = a[p][q];
            a[p][p] = ap - t * off;
            a[q][q] = aq + t * off;
            a[p][q] = 0.;
            a[q][p] = 0.;
            for k in 0..3 {
                if k != p && k != q {
                    let kp = a[k][p];
                    let kq = a[k][q];
                    a[k][p] = c * kp - s * kq;
                    a[p][k] = a[k][p];
                    a[k][q] = s * kp + c * kq;
                    a[q][k] = a[k][q];
                }
            }
        }
    }
    let mut sv = std::array::from_fn(|i| a[i][i].max(0.).sqrt());
    sv.sort_by(|a, b| b.total_cmp(a));
    sv
}
#[derive(Serialize)]
pub struct Structure {
    pub monotonicity_violations: usize,
    pub worst_negative_diff: f64,
    pub total_variation: [f64; 3],
    pub jacobian_log_condition: Summary,
    #[serde(skip)]
    pub jacobian_log_condition_values: Vec<f64>,
    pub surface_flips: usize,
    pub surface_cells: usize,
    pub fft_highband_ratio: [f64; 3],
}
pub fn structure(table: &ImageBuf, n: usize) -> Structure {
    let at = |p: [usize; 3]| {
        let i = ((p[2] * n + p[1]) * n + p[0]) * 3;
        [
            table.data[i] as f64,
            table.data[i + 1] as f64,
            table.data[i + 2] as f64,
        ]
    };
    let (mut violations, mut worst, mut tv) = (0, 0f64, [0.; 3]);
    let mut cond = vec![];
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                let p = [r, g, b];
                let v = at(p);
                for axis in 0..3 {
                    if p[axis] + 1 < n {
                        let mut q = p;
                        q[axis] += 1;
                        let next = at(q);
                        let d = next[axis] - v[axis];
                        if d < 0. {
                            violations += 1;
                            worst = worst.min(d);
                        }
                        tv[axis] += (0..3).map(|c| (next[c] - v[c]).abs()).sum::<f64>();
                    }
                }
                if p.iter().all(|v| *v > 0 && *v + 1 < n) {
                    let mut j = [[0.; 3]; 3];
                    for axis in 0..3 {
                        let mut lo = p;
                        let mut hi = p;
                        lo[axis] -= 1;
                        hi[axis] += 1;
                        let a = at(lo);
                        let b = at(hi);
                        for c in 0..3 {
                            j[c][axis] = b[c] - a[c];
                        }
                    }
                    let sv = singular_values(j);
                    cond.push((sv[0] / sv[2].max(1e-12)).max(1.).log10());
                }
            }
        }
    }
    let count = (n * n * (n - 1) * 3) as f64;
    for v in &mut tv {
        *v /= count;
    }
    let (mut flips, mut cells) = (0, 0);
    for fixed in 0..if n >= 3 { 3 } else { 0 } {
        for side in [0, n - 1] {
            let free: Vec<_> = (0..3).filter(|a| *a != fixed).collect();
            let keep: Vec<_> = (0..3).filter(|a| *a != fixed).collect();
            let (mut negative, mut positive) = (0, 0);
            for u in 0..n - 1 {
                for v in 0..n - 1 {
                    let mut p = [0; 3];
                    p[2 - fixed] = side;
                    p[2 - free[0]] = u;
                    p[2 - free[1]] = v;
                    let a = at(p);
                    let mut pu = p;
                    pu[2 - free[0]] += 1;
                    let bu = at(pu);
                    let mut pv = p;
                    pv[2 - free[1]] += 1;
                    let bv = at(pv);
                    let det = (bu[keep[0]] - a[keep[0]]) * (bv[keep[1]] - a[keep[1]])
                        - (bu[keep[1]] - a[keep[1]]) * (bv[keep[0]] - a[keep[0]]);
                    negative += usize::from(det < 0.);
                    positive += usize::from(det > 0.);
                    cells += 1;
                }
            }
            flips += negative.min(positive);
        }
    }
    let mut ratios = [0.; 3];
    let mut planner = rustfft::FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    for spatial in 0..3 {
        let axis = 2 - spatial;
        let free: Vec<_> = (0..3).filter(|a| *a != axis).collect();
        let (mut total, mut high) = (0., 0.);
        let band = (n / 4).max(1);
        for a in 0..n {
            for b in 0..n {
                for c in 0..3 {
                    let mut line = vec![rustfft::num_complex::Complex::new(0., 0.); n];
                    for i in 0..n {
                        let mut p = [0; 3];
                        p[axis] = i;
                        p[free[0]] = a;
                        p[free[1]] = b;
                        line[i].re = at(p)[c];
                    }
                    fft.process(&mut line);
                    for i in 0..=n / 2 {
                        let energy = line[i].norm();
                        total += energy;
                        if i >= band {
                            high += energy;
                        }
                    }
                }
            }
        }
        ratios[spatial] = high / total.max(1e-12);
    }
    Structure {
        monotonicity_violations: violations,
        worst_negative_diff: worst,
        total_variation: tv,
        jacobian_log_condition: summary(&cond),
        jacobian_log_condition_values: cond,
        surface_flips: flips,
        surface_cells: cells,
        fft_highband_ratio: ratios,
    }
}
pub fn dynamic_range(stops: &[f64], y: &[f64], clipped: &[bool]) -> serde_json::Value {
    let encoded: Vec<_> = clipped
        .iter()
        .enumerate()
        .filter_map(|(i, c)| if !c { Some(i) } else { None })
        .collect();
    if encoded.len() < 2 {
        return serde_json::json!({"encoded_range_stops":0.,"active_range_stops":0.,"toe_collapsed_stops":0.,"shoulder_collapsed_stops":0.});
    }
    let lo = encoded[0];
    let hi = *encoded.last().unwrap();
    let step = (stops[stops.len() - 1] - stops[0]) / (stops.len() - 1) as f64;
    let active: Vec<_> = (lo..hi)
        .map(|i| {
            (y[i + 1].max(1e-4).log10() - y[i].max(1e-4).log10()).abs()
                / (stops[i + 1] - stops[i]).abs().max(1e-9)
                > 0.1
        })
        .collect();
    serde_json::json!({"encoded_range_stops":stops[hi]-stops[lo],"active_range_stops":active.iter().filter(|v|**v).count() as f64*step,"toe_collapsed_stops":active.iter().take_while(|v|!**v).count() as f64*step,"shoulder_collapsed_stops":active.iter().rev().take_while(|v|!**v).count() as f64*step})
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|c| a[c] - b[c])
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|c| a[c] * b[c]).sum()
}
/// Incremental 3D convex hull. Degenerate point sets have zero volume.
pub fn hull_volume(points: &[[f64; 3]]) -> f64 {
    if points.len() < 4 {
        return 0.;
    }
    let first = (0..points.len())
        .min_by(|&a, &b| points[a][0].total_cmp(&points[b][0]))
        .unwrap();
    let second = (0..points.len())
        .max_by(|&a, &b| {
            dot(sub(points[a], points[first]), sub(points[a], points[first])).total_cmp(&dot(
                sub(points[b], points[first]),
                sub(points[b], points[first]),
            ))
        })
        .unwrap();
    let line = sub(points[second], points[first]);
    let third = (0..points.len())
        .max_by(|&a, &b| {
            let x = cross(line, sub(points[a], points[first]));
            let y = cross(line, sub(points[b], points[first]));
            dot(x, x).total_cmp(&dot(y, y))
        })
        .unwrap();
    let normal = cross(line, sub(points[third], points[first]));
    let fourth = (0..points.len())
        .max_by(|&a, &b| {
            dot(normal, sub(points[a], points[first]))
                .abs()
                .total_cmp(&dot(normal, sub(points[b], points[first])).abs())
        })
        .unwrap();
    if dot(normal, sub(points[fourth], points[first])).abs() < 1e-18 {
        return 0.;
    }
    let inside = std::array::from_fn(|c| {
        (points[first][c] + points[second][c] + points[third][c] + points[fourth][c]) / 4.
    });
    let orient = |mut f: [usize; 3]| {
        if dot(
            cross(
                sub(points[f[1]], points[f[0]]),
                sub(points[f[2]], points[f[0]]),
            ),
            sub(inside, points[f[0]]),
        ) > 0.
        {
            f.swap(1, 2);
        }
        f
    };
    let mut faces = vec![
        orient([first, second, third]),
        orient([first, fourth, second]),
        orient([first, third, fourth]),
        orient([second, fourth, third]),
    ];
    for (i, p) in points.iter().enumerate() {
        let visible: Vec<_> = faces
            .iter()
            .map(|f| {
                let n = cross(
                    sub(points[f[1]], points[f[0]]),
                    sub(points[f[2]], points[f[0]]),
                );
                dot(n, sub(*p, points[f[0]])) > 1e-11 * dot(n, n).sqrt()
            })
            .collect();
        if !visible.iter().any(|v| *v) {
            continue;
        }
        let mut horizon = std::collections::BTreeMap::<(usize, usize), usize>::new();
        for (f, v) in faces.iter().zip(&visible) {
            if *v {
                for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                    *horizon.entry((a.min(b), a.max(b))).or_default() += 1;
                }
            }
        }
        faces = faces
            .into_iter()
            .zip(visible)
            .filter_map(|(f, v)| if v { None } else { Some(f) })
            .collect();
        for ((a, b), count) in horizon {
            if count == 1 {
                faces.push(orient([a, b, i]));
            }
        }
    }
    faces
        .iter()
        .map(|f| dot(points[f[0]], cross(points[f[1]], points[f[2]])) / 6.)
        .sum::<f64>()
        .abs()
}
pub fn hue_rotation(input: &[[f64; 3]], output: &[[f64; 3]]) -> serde_json::Value {
    let max = input.iter().map(|p| p[1].hypot(p[2])).fold(0f64, f64::max);
    let mut values = serde_json::Map::new();
    let mut overall = 0f64;
    for (lo, hi) in [(0.20, 0.35), (0.35, 0.50), (0.50, 0.70), (0.70, 1.001)] {
        let angles: Vec<_> = input
            .iter()
            .zip(output)
            .filter(|(a, _)| {
                let c = a[1].hypot(a[2]);
                max > 0. && c >= lo * max && c < hi * max
            })
            .map(|(a, b)| {
                let d = (b[2].atan2(b[1]) - a[2].atan2(a[1]))
                    .to_degrees()
                    .rem_euclid(360.);
                d.min(360. - d)
            })
            .collect();
        if !angles.is_empty() {
            let worst = angles.into_iter().fold(0f64, f64::max);
            overall = overall.max(worst);
            values.insert(
                format!("max_rotation_chroma_{lo:.2}_{hi:.2}"),
                serde_json::json!(worst),
            );
        }
    }
    values.insert("max_hue_rotation_deg".into(), serde_json::json!(overall));
    values.into()
}
#[derive(Serialize)]
pub struct Noise {
    pub sigma1: f64,
    pub anisotropy: f64,
    pub sigma_l: f64,
    pub sigma_ab: f64,
    pub ellipse_angle_degrees: f64,
    pub jacobian: [[f64; 3]; 3],
    pub covariance: [[f64; 3]; 3],
}
pub fn noise<F: Fn([f64; 3]) -> [f64; 3]>(
    sample: [f64; 3],
    evaluate: &F,
    space: &LutSpace,
) -> Noise {
    let mut j = [[0.; 3]; 3];
    for axis in 0..3 {
        let mut a = sample;
        let mut b = sample;
        a[axis] += 0.005;
        b[axis] -= 0.005;
        let a = oklab(xyz(evaluate(a), space));
        let b = oklab(xyz(evaluate(b), space));
        for c in 0..3 {
            j[c][axis] = (a[c] - b[c]) / 0.01;
        }
    }
    let covariance = std::array::from_fn::<_, 3, _>(|a| {
        std::array::from_fn(|b| 0.005f64.powi(2) * (0..3).map(|k| j[a][k] * j[b][k]).sum::<f64>())
    });
    let sv = singular_values(j);
    Noise {
        sigma1: sv[0],
        anisotropy: sv[0] / sv[2].max(1e-9),
        sigma_l: covariance[0][0].max(0.).sqrt(),
        sigma_ab: (covariance[1][1] + covariance[2][2]).max(0.).sqrt(),
        ellipse_angle_degrees: (2. * covariance[1][2])
            .atan2(covariance[1][1] - covariance[2][2])
            .to_degrees()
            / 2.,
        jacobian: j,
        covariance,
    }
}
pub fn second_derivative(curve: &[f64]) -> f64 {
    curve
        .windows(3)
        .map(|p| (p[2] - 2. * p[1] + p[0]).abs())
        .fold(0f64, f64::max)
}
