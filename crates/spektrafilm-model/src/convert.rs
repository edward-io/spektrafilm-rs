//! Inverse spectral scan map, shared by exact conversion and LUT baking.
use rayon::prelude::*;
use spektrafilm_math::{image::ImageBuf, precision::from_f64, spectral};

pub fn parse_calibration(text: &str) -> [[f64; 3]; 3] {
    let values: Result<Vec<f64>, _> = text
        .split(|c: char| c.is_whitespace() || ",;[]()".contains(c))
        .filter(|s| !s.is_empty())
        .map(str::parse)
        .collect();
    match values {
        Ok(v) if v.len() == 9 && v.iter().all(|v| v.is_finite()) => {
            std::array::from_fn(|i| std::array::from_fn(|j| v[3 * i + j]))
        }
        _ => [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    }
}
pub fn inverse(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c = [
        m[1][1] * m[2][2] - m[1][2] * m[2][1],
        m[1][2] * m[2][0] - m[1][0] * m[2][2],
        m[1][0] * m[2][1] - m[1][1] * m[2][0],
    ];
    let determinant = (0..3).map(|i| m[0][i] * c[i]).sum::<f64>();
    if determinant.abs() < 1e-30 || !determinant.is_finite() {
        return None;
    }
    Some(
        [
            [
                c[0],
                m[0][2] * m[2][1] - m[0][1] * m[2][2],
                m[0][1] * m[1][2] - m[0][2] * m[1][1],
            ],
            [
                c[1],
                m[0][0] * m[2][2] - m[0][2] * m[2][0],
                m[0][2] * m[1][0] - m[0][0] * m[1][2],
            ],
            [
                c[2],
                m[0][1] * m[2][0] - m[0][0] * m[2][1],
                m[0][0] * m[1][1] - m[0][1] * m[1][0],
            ],
        ]
        .map(|row| row.map(|v| v / determinant)),
    )
}
fn matvec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    m.map(|r| (0..3).map(|i| r[i] * v[i]).sum())
}

#[derive(Clone, Debug)]
pub struct FilmConverter {
    /// Dye densities and illuminant/base-weighted RGB observer, one row per band.
    pub bands: Vec<[f64; 6]>,
    pub maximum: [f64; 3],
    pub midpoint: [f64; 3],
    pub log_rgb: [f64; 3],
    pub seed_inverse: [[f64; 3]; 3],
}
impl FilmConverter {
    pub fn new(
        dyes: &[[f64; 3]],
        base: &[f64],
        illuminant: &[f64],
        space: &str,
        maximum: [f64; 3],
    ) -> Self {
        let norm: f64 = illuminant
            .iter()
            .enumerate()
            .map(|(w, e)| e * spectral::CMF_Y_F64[w])
            .sum();
        let white: [f64; 3] = std::array::from_fn(|c| {
            illuminant
                .iter()
                .enumerate()
                .map(|(w, e)| {
                    e * [
                        spectral::CMF_X_F64[w],
                        spectral::CMF_Y_F64[w],
                        spectral::CMF_Z_F64[w],
                    ][c]
                })
                .sum::<f64>()
                / norm
        });
        let cat = spektrafilm_math::colorspace::chromatic_adaptation_matrix_f64(
            white,
            spectral::colorspace_white_xyz_f64(space),
        );
        let rgb = spektrafilm_math::transfer::get(space)
            .expect("validated color space")
            .from_xyz;
        let bands = dyes
            .iter()
            .enumerate()
            .map(|(w, d)| {
                if !d.iter().all(|v| v.is_finite()) || !base[w].is_finite() {
                    return [0.; 6];
                }
                let observer = matvec(
                    rgb,
                    matvec(
                        cat,
                        [
                            spectral::CMF_X_F64[w],
                            spectral::CMF_Y_F64[w],
                            spectral::CMF_Z_F64[w],
                        ],
                    ),
                );
                let light = illuminant[w] * 10f64.powf(-base[w]) / norm;
                [
                    d[0],
                    d[1],
                    d[2],
                    observer[0] * light,
                    observer[1] * light,
                    observer[2] * light,
                ]
            })
            .collect();
        let midpoint = maximum.map(|v| 0.5 * v);
        let mut result = Self {
            bands,
            maximum,
            midpoint,
            log_rgb: [0.; 3],
            seed_inverse: [[0.; 3]; 3],
        };
        let (f, j) = result.forward_jac(midpoint);
        result.log_rgb = f.map(|v| v.max(1e-12).log10());
        let g: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|c| -j[i][c] / (f[i].max(1e-12) * std::f64::consts::LN_10))
        });
        // A damped pseudoinverse also handles a single BW emulsion broadcast over RGB.
        let normal: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|c| {
                (0..3).map(|k| g[k][i] * g[k][c]).sum::<f64>() + if i == c { 1e-12 } else { 0. }
            })
        });
        if let Some(inv) = inverse(normal) {
            result.seed_inverse = std::array::from_fn(|i| {
                std::array::from_fn(|c| (0..3).map(|k| inv[i][k] * g[c][k]).sum())
            });
        }
        result
    }
    pub fn forward_jac(&self, density: [f64; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
        let mut rgb = [0.; 3];
        let mut jac = [[0.; 3]; 3];
        for b in &self.bands {
            let light = (-std::f64::consts::LN_10
                * (b[0] * density[0] + b[1] * density[1] + b[2] * density[2]))
                .exp();
            for c in 0..3 {
                let v = light * b[3 + c];
                rgb[c] += v;
                for k in 0..3 {
                    jac[c][k] -= std::f64::consts::LN_10 * v * b[k];
                }
            }
        }
        (rgb, jac)
    }
    pub fn invert(&self, rgb: [f64; 3]) -> [f64; 3] {
        let delta = matvec(
            self.seed_inverse,
            std::array::from_fn(|c| self.log_rgb[c] - rgb[c].max(1e-12).log10()),
        );
        let mut x: [f64; 3] =
            std::array::from_fn(|c| (self.midpoint[c] + delta[c]).clamp(0., self.maximum[c]));
        for _ in 0..8 {
            let (f, j) = self.forward_jac(x);
            let residual: [f64; 3] = std::array::from_fn(|c| f[c] - rgb[c]);
            if residual.iter().all(|v| v.abs() < 1e-6) {
                break;
            }
            let normal = std::array::from_fn(|i| {
                std::array::from_fn(|c| {
                    (0..3).map(|k| j[k][i] * j[k][c]).sum::<f64>() + if i == c { 1e-7 } else { 0. }
                })
            });
            let gradient = std::array::from_fn(|i| (0..3).map(|k| j[k][i] * residual[k]).sum());
            let Some(inv) = inverse(normal) else {
                break;
            };
            let step = matvec(inv, gradient);
            for c in 0..3 {
                x[c] = (x[c] - step[c]).clamp(0., self.maximum[c]);
            }
        }
        x
    }
    pub fn convert(&self, image: &ImageBuf, calibration: [[f64; 3]; 3], gain: f64) -> ImageBuf {
        let mut out = image.clone();
        out.data.par_chunks_exact_mut(3).for_each(|p| {
            let y = std::array::from_fn(|c| {
                (0..3).map(|k| p[k] as f64 * calibration[k][c]).sum::<f64>() * gain
            });
            let x = self.invert(y);
            for c in 0..3 {
                p[c] = from_f64(x[c]);
            }
        });
        out
    }
    pub fn bounds(&self) -> ([f64; 3], [f64; 3]) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for i in 0..12 {
            for j in 0..12 {
                for k in 0..12 {
                    let d = std::array::from_fn(|c| [i, j, k][c] as f64 / 11. * self.maximum[c]);
                    let (rgb, _) = self.forward_jac(d);
                    for c in 0..3 {
                        lo[c] = lo[c].min(rgb[c]);
                        hi[c] = hi[c].max(rgb[c]);
                    }
                }
            }
        }
        let margin: [f64; 3] = std::array::from_fn(|c| 0.02 * (hi[c] - lo[c]));
        (
            std::array::from_fn(|c| lo[c] - margin[c]),
            std::array::from_fn(|c| hi[c] + margin[c]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calibration_parser_handles_gui_pastes_and_invalid_text() {
        assert_eq!(
            parse_calibration("[[1,0,0]; [0,1,0]; [0,0,1]]"),
            parse_calibration("invalid")
        );
        assert_eq!(parse_calibration("1 2 3 4 5 6 7 8 9")[2], [7., 8., 9.]);
    }
    #[test]
    fn inverse_solves_general_matrix() {
        let m = [[2., 1., 3.], [0., 3., 2.], [5., 1., 1.]];
        let v = [3., 4., 5.];
        let actual = matvec(inverse(m).unwrap(), matvec(m, v));
        for c in 0..3 {
            assert!((actual[c] - v[c]).abs() < 1e-12);
        }
    }
}
