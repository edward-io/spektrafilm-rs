//! Source-equivalent static paper back. Built on a worker and cached by aspect.
use super::PreviewTextureData;
use rayon::prelude::*;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone)]
struct Plane {
    w: usize,
    h: usize,
    p: Vec<f32>,
}
impl Plane {
    fn sample(&self, y: f64, x: f64) -> f32 {
        if x < 0. || y < 0. || x > (self.w - 1) as f64 || y > (self.h - 1) as f64 {
            return 0.;
        }
        let ix = x.floor() as usize;
        let iy = y.floor() as usize;
        let tx = x - ix as f64;
        let ty = y - iy as f64;
        let a = self.p[iy * self.w + ix] as f64;
        let b = self.p[iy * self.w + (ix + 1).min(self.w - 1)] as f64;
        let c = self.p[(iy + 1).min(self.h - 1) * self.w + ix] as f64;
        let d = self.p[(iy + 1).min(self.h - 1) * self.w + (ix + 1).min(self.w - 1)] as f64;
        ((a * (1. - tx) + b * tx) * (1. - ty) + (c * (1. - tx) + d * tx) * ty) as f32
    }
    fn gaussian(&self, sigma: f64) -> Self {
        if sigma <= 0. {
            return self.clone();
        }
        let r = (4. * sigma + 0.5).floor() as isize;
        let mut kernel: Vec<_> = (-r..=r)
            .map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp())
            .collect();
        let sum = kernel.iter().sum::<f64>();
        kernel.iter_mut().for_each(|v| *v /= sum);
        let mut result = self.clone();
        // scipy.ndimage's default y-then-x ordering, nearest extension.
        for axis in 0..2 {
            let src = result.p;
            let mut dst = vec![0f32; self.p.len()];
            dst.par_chunks_exact_mut(self.w)
                .enumerate()
                .for_each(|(y, row)| {
                    for (x, value) in row.iter_mut().enumerate() {
                        *value = kernel
                            .iter()
                            .enumerate()
                            .map(|(i, k)| {
                                let delta = i as isize - r;
                                let (xx, yy) = if axis == 0 {
                                    (
                                        x,
                                        (y as isize + delta).clamp(0, self.h as isize - 1) as usize,
                                    )
                                } else {
                                    (
                                        (x as isize + delta).clamp(0, self.w as isize - 1) as usize,
                                        y,
                                    )
                                };
                                *k * src[yy * self.w + xx] as f64
                            })
                            .sum::<f64>() as f32;
                    }
                });
            result.p = dst;
        }
        result
    }
}
fn stamp(long_edge: usize) -> Plane {
    static LOGO: OnceLock<Plane> = OnceLock::new();
    let source = LOGO.get_or_init(|| {
        let image = image::load_from_memory(include_bytes!(
            "../../../data/assets/virtual_paper_watermark.png"
        ))
        .expect("bundled paper-back logo")
        .to_rgba8();
        Plane {
            w: image.width() as usize,
            h: image.height() as usize,
            p: image.pixels().map(|p| p[3] as f32 / 255.).collect(),
        }
    });
    let scale = 0.22 * long_edge as f64 / source.w.max(source.h) as f64;
    let source = source.gaussian((0.5 / scale - 0.5).max(0.));
    let w = (source.w as f64 * scale).round_ties_even().max(1.) as usize;
    let h = (source.h as f64 * scale).round_ties_even().max(1.) as usize;
    let mut resized = Plane {
        w,
        h,
        p: vec![0.; w * h],
    };
    resized
        .p
        .par_chunks_exact_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, v) in row.iter_mut().enumerate() {
                *v = source.sample(
                    y as f64 * (source.h - 1) as f64 / (h - 1).max(1) as f64,
                    x as f64 * (source.w - 1) as f64 / (w - 1).max(1) as f64,
                );
            }
        });
    let (s, c) = 16f64.to_radians().sin_cos();
    let rw = (c * w as f64 + s * h as f64 + 0.5) as usize;
    let rh = (c * h as f64 + s * w as f64 + 0.5) as usize;
    let oy = (h - 1) as f64 / 2. - (c * (rh - 1) as f64 + s * (rw - 1) as f64) / 2.;
    let ox = (w - 1) as f64 / 2. - (-s * (rh - 1) as f64 + c * (rw - 1) as f64) / 2.;
    let mut rotated = Plane {
        w: rw,
        h: rh,
        p: vec![0.; rw * rh],
    };
    rotated
        .p
        .par_chunks_exact_mut(rw)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, v) in row.iter_mut().enumerate() {
                *v = resized
                    .sample(
                        c * y as f64 + s * x as f64 + oy,
                        -s * y as f64 + c * x as f64 + ox,
                    )
                    .clamp(0., 1.);
            }
        });
    let mut x0 = rw;
    let mut x1 = 0;
    let mut y0 = rh;
    let mut y1 = 0;
    for (i, &v) in rotated.p.iter().enumerate() {
        if v > 1e-4 {
            x0 = x0.min(i % rw);
            x1 = x1.max(i % rw + 1);
            y0 = y0.min(i / rw);
            y1 = y1.max(i / rw + 1);
        }
    }
    Plane {
        w: x1 - x0,
        h: y1 - y0,
        p: (y0..y1)
            .flat_map(|y| rotated.p[y * rw + x0..y * rw + x1].iter().copied())
            .collect(),
    }
}
/// NumPy default_rng(0)'s PCG64 initial state and float32 stream. This keeps the
/// fixed seeded paper texture identical without carrying a Python runtime.
struct Pcg {
    state: u128,
    cache: Option<u32>,
}
impl Pcg {
    fn new() -> Self {
        Self {
            state: 35399562948360463058890781895381311971,
            cache: None,
        }
    }
    fn uniform(&mut self) -> f32 {
        let bits = if let Some(v) = self.cache.take() {
            v
        } else {
            self.state = self
                .state
                .wrapping_mul(47026247687942121848144207491837523525)
                .wrapping_add(87136372517582989555478159403783844777);
            let word = ((self.state >> 64) as u64 ^ self.state as u64)
                .rotate_right((self.state >> 122) as u32);
            self.cache = Some((word >> 32) as u32);
            word as u32
        };
        (bits >> 8) as f32 * (1. / 16777216.)
    }
}
fn build(w: usize, h: usize) -> PreviewTextureData {
    let tile = stamp(w.max(h));
    let long = w.max(h) as f32;
    // The rhombic basis and center arithmetic are float32 in the reference.
    let half = 36f64.to_radians();
    let a = 16f64.to_radians();
    let distance = 0.25 * long;
    let b0 = [distance * half.cos() as f32, distance * half.sin() as f32];
    let b1 = [b0[0], -b0[1]];
    let rotate = |b: [f32; 2]| {
        [
            b[0] * a.cos() as f32 + b[1] * a.sin() as f32,
            -b[0] * a.sin() as f32 + b[1] * a.cos() as f32,
        ]
    };
    let b0 = rotate(b0);
    let b1 = rotate(b1);
    let radius = 0.5 * (tile.w as f64).hypot(tile.h as f64);
    let limit = (((w as f64).hypot(h as f64) + 2. * radius) / (distance as f64)).ceil() as i32 + 2;
    let mut centers = vec![];
    for i in -limit..=limit {
        for j in -limit..=limit {
            // NumPy promotes integer mesh coordinates to float64 here.
            let x = w as f64 * 0.5 + i as f64 * b0[0] as f64 + j as f64 * b1[0] as f64;
            let y = h as f64 * 0.5 + i as f64 * b0[1] as f64 + j as f64 * b1[1] as f64;
            if x >= -radius && x <= w as f64 + radius && y >= -radius && y <= h as f64 + radius {
                centers.push([x, y]);
            }
        }
    }
    centers.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let mut trans = vec![1f32; w * h];
    for [x, y] in centers {
        let x0 = (x - 0.5 * tile.w as f64).round_ties_even() as isize;
        let y0 = (y - 0.5 * tile.h as f64).round_ties_even() as isize;
        for yy in y0.max(0)..(y0 + tile.h as isize).min(h as isize) {
            for xx in x0.max(0)..(x0 + tile.w as isize).min(w as isize) {
                trans[yy as usize * w + xx as usize] *=
                    1. - tile.p[(yy - y0) as usize * tile.w + (xx - x0) as usize];
            }
        }
    }
    let mut rng = Pcg::new();
    let a: Vec<_> = (0..w * h).map(|_| rng.uniform()).collect();
    let glare: Vec<_> = a
        .into_iter()
        .map(|a| 0.88 * a + 0.12 * rng.uniform().powf(10.))
        .collect();
    let mut rgba = vec![255u8; w * h * 4];
    rgba.par_chunks_exact_mut(4)
        .enumerate()
        .for_each(|(i, out)| {
            let t = trans[i];
            let g = glare[i];
            let paper = 0.7 * t + 0.3;
            let variation = 1. + (2. * g - 1.) * paper * (0.07 * 0.25);
            let lift = g.powf(6.) * paper * (0.42 * 0.25);
            let mut v = t * (242. / 255. - 215. / 255.) + 215. / 255.;
            v *= variation;
            v += (1. - v) * lift;
            let v = (v.clamp(0., 1.) * 255.).round_ties_even() as u8;
            out[..3].fill(v);
        });
    PreviewTextureData { size: [w, h], rgba }
}
pub fn cached(w: usize, h: usize) -> Arc<PreviewTextureData> {
    type Entry = ([usize; 2], Arc<PreviewTextureData>);
    static CACHE: OnceLock<Mutex<Vec<Entry>>> = OnceLock::new();
    let long = w.max(h).max(1);
    let w = ((w as f64 * 1024. / long as f64).round_ties_even() as usize).max(1);
    let h = ((h as f64 * 1024. / long as f64).round_ties_even() as usize).max(1);
    let cache = CACHE.get_or_init(Default::default);
    if let Some((_, image)) = cache
        .lock()
        .unwrap()
        .iter()
        .find(|(size, _)| *size == [w, h])
    {
        return image.clone();
    }
    let result = Arc::new(build(w, h));
    let mut entries = cache.lock().unwrap();
    if entries.len() >= 8 {
        entries.remove(0);
    }
    entries.push(([w, h], result.clone()));
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paper_back_matches_python_default_texture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/paper_back.json")).unwrap();
        let size: [usize; 2] = serde_json::from_value(fixture["size"].clone()).unwrap();
        let expected: Vec<u8> = serde_json::from_value(fixture["rgba"].clone()).unwrap();
        let actual = build(size[0], size[1]);
        let max = actual
            .rgba
            .iter()
            .zip(&expected)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        let different = actual
            .rgba
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            max <= 1 && different < 100,
            "paper back drift: max {max}/255, changed {different}"
        );
    }
}
