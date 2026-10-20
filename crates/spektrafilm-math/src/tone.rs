//! Input luminance controls, applied before spectral reconstruction.
//!
//! Shadows, Highlights and Whites choose their exposure change from a regional
//! base: a self-guided filter of log luminance. Texture inside a region keeps
//! its contrast; uniform regions follow the scalar curve exactly. Blacks is a
//! per-pixel adjustment near zero.
use crate::{image::ImageBuf, precision::Scalar};
use rayon::prelude::*;

pub const HIGHLIGHT_STOPS: f64 = 0.6495;
/// Box radius of the guided filter as a fraction of the long image edge, so
/// previews and full-resolution exports select the same regions.
pub const REGION_FRACTION: f64 = 0.02;
/// Guided-filter regularization in stops squared. Variation well below
/// `sqrt(DETAIL_EPS)` stops counts as detail; larger steps count as edges.
pub const DETAIL_EPS: f64 = 8.;
/// Luminance floor for the log coordinate, 12 stops below middle gray.
pub const FLOOR_STOPS: f64 = -12.;
const GRAY: Scalar = 0.184;
const TOE: Scalar = GRAY / 8.;
const FLOOR: Scalar = TOE / 3.;

/// Guided-filter box radius for an image, shared by CPU and GPU.
pub fn region_radius(width: u32, height: u32) -> u32 {
    ((width.max(height) as f64 * REGION_FRACTION).round() as u32).max(1)
}

#[derive(Debug, Clone, Copy)]
pub struct ToneCurve {
    /// Normalized Shadows, Highlights, Whites, Blacks, in [-1, 1].
    pub controls: [Scalar; 4],
    pub luma: [Scalar; 3],
}

fn bell(x: Scalar) -> Scalar {
    let t = x.clamp(0., 1.);
    16. * t * t * (1. - t) * (1. - t)
}

fn smooth(x: Scalar) -> Scalar {
    let t = x.clamp(0., 1.);
    (t * t * t * (10. + t * (-15. + 6. * t))).clamp(0., 1.)
}

impl ToneCurve {
    /// Sliders are validated percentages; weights are the native XYZ Y row.
    pub fn new(sliders: [f64; 4], weights: [f64; 3]) -> Self {
        let sum: f64 = weights.iter().sum();
        Self {
            controls: sliders.map(|v| (v / 100.) as Scalar),
            luma: weights.map(|v| (v / sum) as Scalar),
        }
    }

    pub fn is_identity(&self) -> bool {
        self.controls == [0.; 4]
    }

    /// Whether any control needs the regional base (all but Blacks).
    pub fn is_regional(&self) -> bool {
        self.controls[..3] != [0.; 3]
    }

    fn luminance(&self, rgb: [Scalar; 3]) -> Scalar {
        (0..3).map(|c| rgb[c] * self.luma[c]).sum()
    }

    /// Stops relative to middle gray, with a floor so zero stays finite.
    fn stops(y: Scalar) -> Scalar {
        let floor = GRAY * (FLOOR_STOPS as Scalar).exp2();
        (y.abs().max(floor) / GRAY).log2()
    }

    /// Combined Shadows, Highlights and Whites change in stops at `e`.
    fn delta(&self, e: Scalar) -> Scalar {
        let [s, h, w, _] = self.controls;
        let ds = s * bell((e + 5.) / 5.);
        let e1 = e + ds;
        let dh = HIGHLIGHT_STOPS as Scalar * h * bell(e1 / 3.);
        ds + dh + 1.2 * w * smooth((e1 + dh - 0.5) / 3.5)
    }

    /// Apply an exposure change, then Blacks on the pixel's own luminance.
    fn finish(&self, rgb: [Scalar; 3], y: Scalar, delta: Scalar) -> [Scalar; 3] {
        let gain = delta.exp2();
        let rgb = rgb.map(|v| v * gain);
        let b = self.controls[3];
        if b == 0. {
            return rgb;
        }
        let a = (y * gain).abs();
        if a >= TOE {
            return rgb;
        }
        let offset = b * FLOOR * (1. - smooth(a / TOE));
        if b > 0. {
            // A neutral floor can lift zero without an unbounded RGB gain.
            rgb.map(|v| v + offset)
        } else {
            let scale = if a == 0. {
                0.
            } else {
                (a + offset).max(0.) / a
            };
            rgb.map(|v| v * scale)
        }
    }

    /// The scalar curve: the response of a uniform region.
    pub fn apply_pixel(&self, rgb: [Scalar; 3]) -> [Scalar; 3] {
        if self.is_identity() {
            return rgb;
        }
        let y = self.luminance(rgb);
        if !y.is_finite() {
            return rgb;
        }
        self.finish(rgb, y, self.delta(Self::stops(y)))
    }

    /// Per-pixel application, for LUT bakes whose lattice has no image space.
    pub fn apply_pointwise(&self, image: &mut ImageBuf) {
        if self.is_identity() {
            return;
        }
        image.data.par_chunks_exact_mut(3).for_each(|px| {
            px.copy_from_slice(&self.apply_pixel([px[0], px[1], px[2]]));
        });
    }

    /// Regional application to a photograph.
    pub fn apply(&self, image: &mut ImageBuf) {
        if !self.is_regional() {
            return self.apply_pointwise(image);
        }
        let (w, h) = (image.width as usize, image.height as usize);
        let r = region_radius(image.width, image.height) as usize;
        // Invalid pixels stay as they are and act as middle gray in the base.
        let e: Vec<Scalar> = image
            .data
            .par_chunks_exact(3)
            .map(|p| {
                let y = self.luminance([p[0], p[1], p[2]]);
                if y.is_finite() { Self::stops(y) } else { 0. }
            })
            .collect();
        let stats: Vec<[Scalar; 2]> = e.par_iter().map(|&v| [v, v * v]).collect();
        let eps = DETAIL_EPS as Scalar;
        let coefficients: Vec<[Scalar; 2]> = box_mean(&stats, w, h, r)
            .into_par_iter()
            .map(|[mean, mean_sq]| {
                let a = (mean_sq - mean * mean).max(0.);
                let a = a / (a + eps);
                [a, mean * (1. - a)]
            })
            .collect();
        let coefficients = box_mean(&coefficients, w, h, r);
        image
            .data
            .par_chunks_exact_mut(3)
            .zip(e.par_iter().zip(&coefficients))
            .for_each(|(px, (&e, &[a, b]))| {
                let rgb = [px[0], px[1], px[2]];
                let y = self.luminance(rgb);
                if y.is_finite() {
                    px.copy_from_slice(&self.finish(rgb, y, self.delta(a * e + b)));
                }
            });
    }
}

/// Mean of each pair over the in-bounds part of a `(2r+1)²` window.
fn box_mean(src: &[[Scalar; 2]], w: usize, h: usize, r: usize) -> Vec<[Scalar; 2]> {
    let rows = box_rows(src, w, h, r);
    transpose(&box_rows(&transpose(&rows, w, h), h, w, r), h, w)
}

/// Horizontal running mean with f64 accumulators.
fn box_rows(src: &[[Scalar; 2]], w: usize, h: usize, r: usize) -> Vec<[Scalar; 2]> {
    let mut out = vec![[0.; 2]; w * h];
    out.par_chunks_exact_mut(w)
        .zip(src.par_chunks_exact(w))
        .for_each(|(out, row)| {
            let mut sum = [0f64; 2];
            for v in &row[..(r + 1).min(w)] {
                sum = [sum[0] + v[0] as f64, sum[1] + v[1] as f64];
            }
            for x in 0..w {
                let count = ((x + r).min(w - 1) - x.saturating_sub(r) + 1) as f64;
                out[x] = sum.map(|s| (s / count) as Scalar);
                if let Some(v) = row.get(x + r + 1) {
                    sum = [sum[0] + v[0] as f64, sum[1] + v[1] as f64];
                }
                if x >= r {
                    let v = row[x - r];
                    sum = [sum[0] - v[0] as f64, sum[1] - v[1] as f64];
                }
            }
        });
    out
}

fn transpose(src: &[[Scalar; 2]], w: usize, h: usize) -> Vec<[Scalar; 2]> {
    let mut out = vec![[0.; 2]; w * h];
    out.par_chunks_exact_mut(h)
        .enumerate()
        .for_each(|(x, col)| {
            for (y, v) in col.iter_mut().enumerate() {
                *v = src[y * w + x];
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(values: [f64; 4]) -> ToneCurve {
        ToneCurve::new(values, [0.2126, 0.7152, 0.0722])
    }

    fn gray_image(w: u32, h: u32, stops: impl Fn(u32, u32) -> f64) -> ImageBuf {
        let data = (0..w * h)
            .flat_map(|i| {
                let v = (0.184 * stops(i % w, i / w).exp2()) as Scalar;
                [v; 3]
            })
            .collect();
        ImageBuf::from_data(w, h, data)
    }

    fn stops_at(image: &ImageBuf, x: u32, y: u32) -> f64 {
        (image.data[((y * image.width + x) * 3) as usize] as f64 / 0.184).log2()
    }

    #[test]
    fn zero_controls_preserve_every_bit() {
        let pixels = [-0., 0., -0.2, 16., Scalar::MIN_POSITIVE, 0.184];
        let mut image = ImageBuf::from_data(2, 1, pixels.to_vec());
        curve([0.; 4]).apply(&mut image);
        for (a, b) in image.data.iter().zip(pixels) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn reference_values_and_rgb_behavior() {
        let tolerance = if cfg!(feature = "precision-f64") {
            2e-12
        } else {
            2e-6
        };
        for (values, input, expected) in [
            ([100., 0., 0., 0.], 0.03252691193458119, 0.06505382386916238),
            ([0., 100., 0., 0.], 0.520430590953299, 0.816360153539059),
            ([0., 0., 100., 0.], 1., 1.6501316827555212),
            ([100., 0., 0., -100.], 0.0115, 0.01363800559376751),
            (
                [0., -100., 100., 0.],
                0.520430590953299,
                0.33415546666025664,
            ),
            ([0., 0., 0., 100.], 0., 0.184 / 24.),
            ([0., 0., 0., -100.], 0.003833, 0.),
        ] {
            let actual = curve(values).apply_pixel([input as Scalar; 3]);
            for v in actual {
                assert!(
                    (v as f64 - expected).abs() < tolerance,
                    "{values:?}: {v} vs {expected}"
                );
            }
        }
        let input = [0.02, 0.03, 0.6];
        let output = curve([100., 0., 0., 0.]).apply_pixel(input);
        assert!((output[0] / input[0] - output[2] / input[2]).abs() < 2e-6);
        assert!(output[2] > 0.94 && output[2] < 0.96);
        assert_eq!(curve([0., 0., 0., -100.]).apply_pixel([0.; 3]), [0.; 3]);
        assert!(curve([0., 0., 0., 100.]).apply_pixel([0.001, 0., 0.])[1] > 0.);
    }

    #[test]
    fn combined_curves_preserve_signed_luminance_order_and_gray() {
        let levels = [-100., -50., 0., 50., 100.];
        let mut inputs: Vec<Scalar> = (-400i32..=400)
            .map(|i| {
                if i == 0 {
                    0.
                } else {
                    (i as Scalar).signum() * GRAY * (i.abs() as Scalar / 20. - 12.).exp2()
                }
            })
            .collect();
        inputs.sort_by(|a, b| a.total_cmp(b));
        for s in levels {
            for h in levels {
                for w in levels {
                    for b in levels {
                        let c = curve([s, h, w, b]);
                        let mut previous = Scalar::NEG_INFINITY;
                        for &v in &inputs {
                            let out = c.apply_pixel([v; 3])[0];
                            assert!(
                                out.is_finite() && out >= previous - 2e-6,
                                "{:?}: {v} -> {out}, previous {previous}",
                                c.controls
                            );
                            previous = out;
                        }
                        assert!((c.apply_pixel([GRAY; 3])[0] - GRAY).abs() < 1e-7);
                    }
                }
            }
        }
        for rgb in [[1., -0.3, 0.], [-0.2, 0.01, 0.], [Scalar::MIN_POSITIVE; 3]] {
            assert!(
                curve([100., -100., 100., -100.])
                    .apply_pixel(rgb)
                    .iter()
                    .all(|v| v.is_finite())
            );
        }
    }

    #[test]
    fn contrast_floor_uses_the_right_coordinate() {
        let max_bell_derivative = 16. * 3f64.sqrt() / 9.;
        for floor in [
            1. - max_bell_derivative / 5.,
            1. - HIGHLIGHT_STOPS / 3. * max_bell_derivative,
            1. - 1.2 / 3.5 * 15. / 8.,
            1. - (1. / 3.) * 15. / 8.,
        ] {
            assert!(floor >= 1. / 3., "{floor}");
        }
        let y = GRAY / 256.;
        let c = curve([0., 0., 0., 100.]);
        let out = c.apply_pixel([y; 3])[0];
        let t = y / TOE;
        let linear_slope = 1. - (1. / 3.) * 30. * t * t * (1. - t) * (1. - t);
        let stop_slope = y / out * linear_slope;
        assert!((stop_slope as f64 - 0.08495133630889873).abs() < 1e-6);
    }

    #[test]
    fn uniform_regions_follow_the_scalar_curve() {
        for controls in [
            [100., 0., 0., 0.],
            [-100., 100., -100., 0.],
            [60., -80., 40., -25.],
        ] {
            let c = curve(controls);
            for stops in [-8., -2.5, -1., 0., 1.5, 3., 6.] {
                let mut image = gray_image(37, 23, |_, _| stops);
                c.apply(&mut image);
                let expected = c.apply_pixel([(0.184 * f64::exp2(stops)) as Scalar; 3])[0];
                for &v in &image.data {
                    assert!(
                        (v - expected).abs() <= 1e-5 * expected.abs() + 1e-9,
                        "{controls:?} at {stops}: {v} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn regional_lift_keeps_texture_the_scalar_curve_flattens() {
        // Half-stop checkerboard one stop below gray, where the scalar
        // Shadows curve has its strongest compression.
        let pattern = |x: u32, y: u32| {
            if (x + y).is_multiple_of(2) {
                -1.25
            } else {
                -0.75
            }
        };
        let c = curve([100., 0., 0., 0.]);
        let mut regional = gray_image(64, 64, pattern);
        c.apply(&mut regional);
        let mut scalar = gray_image(64, 64, pattern);
        c.apply_pointwise(&mut scalar);
        let contrast = |image: &ImageBuf| stops_at(image, 31, 32) - stops_at(image, 32, 32);
        assert!(contrast(&regional) > 0.95 * 0.5, "{}", contrast(&regional));
        assert!(contrast(&scalar) < 0.75 * 0.5, "{}", contrast(&scalar));
        assert!(
            stops_at(&regional, 32, 32) > -1.25 + 0.2,
            "the region is lifted"
        );
    }

    #[test]
    fn edges_bound_regions_and_keep_their_order() {
        let (w, h) = (200, 40);
        let r = region_radius(w, h);
        let c = curve([100., -100., 50., 0.]);
        let mut image = gray_image(w, h, |x, _| if x < w / 2 { -4. } else { 1.5 });
        c.apply(&mut image);
        let dark = c.apply_pixel([(0.184 * f64::exp2(-4.)) as Scalar; 3])[0] as f64;
        let bright = c.apply_pixel([(0.184 * f64::exp2(1.5)) as Scalar; 3])[0] as f64;
        for x in (0..w / 2 - 2 * r).chain(w / 2 + 2 * r..w) {
            let expected = if x < w / 2 { dark } else { bright };
            let v = image.data[(20 * w + x) as usize * 3] as f64;
            assert!((v / expected - 1.).abs() < 1e-5, "x={x}: {v} vs {expected}");
        }
        let row: Vec<f64> = (0..w)
            .map(|x| image.data[(20 * w + x) as usize * 3] as f64)
            .collect();
        assert!(
            row[..w as usize / 2]
                .iter()
                .all(|&v| v < row[w as usize / 2])
        );
    }

    #[test]
    fn blacks_only_and_invalid_pixels_stay_per_pixel() {
        let c = curve([0., 0., 0., -60.]);
        let mut image = gray_image(16, 9, |x, y| -6. + (x * 7 + y * 3) as f64 % 5.);
        let mut expected = image.clone();
        c.apply(&mut image);
        c.apply_pointwise(&mut expected);
        assert_eq!(image.data, expected.data);

        let mut image = gray_image(16, 9, |_, _| -2.);
        image.data[30] = Scalar::NAN;
        curve([100., 0., 0., 0.]).apply(&mut image);
        assert!(image.data[30].is_nan());
        assert!(
            image
                .data
                .iter()
                .enumerate()
                .all(|(i, v)| i / 3 == 10 || v.is_finite())
        );
    }
}
