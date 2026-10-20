//! Input gamut compression — faithful port of upstream
//! `spektrafilm/utils/gamut_compression.py` (input side).
//!
//! The compression is baked into the tc_lut once at build time via
//! remap-resample: `new_lut[xy] = old_lut[compress(xy)]`. The per-pixel
//! runtime path (RGB → CIE xy → tc → bilinear LUT sample) therefore stays
//! compression-agnostic. The production `"xy"` radial compressor and inspection `"oklch"` variant
//! are both available.

use rayon::prelude::*;
use spektrafilm_math::spectral::{self, CMF_X_F64, CMF_Y_F64, CMF_Z_F64, TcLut};
use std::sync::{Arc, Mutex, OnceLock};

use crate::gamut_compression::reinhard_knee;

/// CIE 1931 2° visible spectral locus as a closed xy polygon, sampled at 5 nm
/// from 380 to 700 nm (65 vertices + the first repeated). Mirrors upstream
/// `spectral_locus_xy()`; the CMFs are the same colour-science values.
pub fn spectral_locus_xy() -> Vec<[f64; 2]> {
    // Wavelengths 380..=700 step 5 → indices 0..65 of the 380..780 grid.
    const N: usize = 65;
    let mut poly = Vec::with_capacity(N + 1);
    for i in 0..N {
        let (x, y, z) = (CMF_X_F64[i], CMF_Y_F64[i], CMF_Z_F64[i]);
        let total = (x + y + z).max(1e-12);
        poly.push([x / total, y / total]);
    }
    poly.push(poly[0]);
    poly
}

/// Distance from `origin` along unit `direction` to the first intersection
/// with the closed polygon, via parametric segment intersection. Returns
/// `f64::INFINITY` for rays that miss (should not happen for an interior
/// origin and the visible locus). Mirrors upstream `_ray_polygon_distance`.
fn ray_polygon_distance(origin: [f64; 2], direction: [f64; 2], polygon: &[[f64; 2]]) -> f64 {
    let mut t_min = f64::INFINITY;
    let (dx, dy) = (direction[0], direction[1]);
    for seg in polygon.windows(2) {
        let (a, b) = (seg[0], seg[1]);
        let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
        let denom = dx * ey - dy * ex;
        if denom.abs() <= 1e-12 {
            continue;
        }
        let (ox, oy) = (origin[0] - a[0], origin[1] - a[1]);
        // origin + t·direction = a + s·edge
        let t = (-ox * ey + oy * ex) / denom;
        let s = (-ox * dy + oy * dx) / denom;
        if t > 1e-9 && (-1e-9..=1.0 + 1e-9).contains(&s) && t < t_min {
            t_min = t;
        }
    }
    t_min
}

/// ACES-RGC-style radial compression of a single CIE xy toward the spectral
/// locus, around `white_xy`. Hue (dominant wavelength) is preserved by
/// construction. Mirrors upstream `compress_xy_radial`.
fn compress_xy_radial(
    xy: [f64; 2],
    white_xy: [f64; 2],
    knee: (f64, f64, f64),
    locus: &[[f64; 2]],
) -> [f64; 2] {
    let delta = [xy[0] - white_xy[0], xy[1] - white_xy[1]];
    let dist = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
    if dist < 1e-9 {
        return xy;
    }
    let safe_dist = dist.max(1e-12);
    let direction = [delta[0] / safe_dist, delta[1] / safe_dist];
    let boundary = ray_polygon_distance(white_xy, direction, locus);
    let d_norm = dist / boundary.max(1e-12);
    let (t, l, p) = knee;
    let d_compressed = reinhard_knee(d_norm, t, l, p);
    let scaled = d_compressed * boundary;
    [
        white_xy[0] + direction[0] * scaled,
        white_xy[1] + direction[1] * scaled,
    ]
}

/// Configured input gamut compressor. `build` resolves the algorithm string;
/// `is_active` reports whether a non-identity remap will be applied.
pub struct InputGamutCompress {
    active: bool,
    oklch: bool,
    knee: (f64, f64, f64),
    locus: Vec<[f64; 2]>,
    oklch_table: OnceLock<Arc<Vec<f64>>>,
}

impl InputGamutCompress {
    /// Resolve from the param algorithm string. `"off"` → inactive (identity);
    /// `"xy"` → active radial compression. Any other value errors. The knee is
    /// validated as in upstream `InputGamutCompressSpec.__post_init__`.
    pub fn build(algorithm: &str, knee: [f32; 3]) -> Result<Self, String> {
        let active = match algorithm {
            "off" => false,
            "xy" | "oklch" => true,
            other => return Err(format!("unknown input gamut algorithm: {other}")),
        };
        let [t, l, p] = knee;
        if !(0.0..1.0).contains(&t) {
            return Err(format!("knee threshold must be in [0, 1), got {t}"));
        }
        if l <= 0.0 {
            return Err(format!("knee limit must be > 0, got {l}"));
        }
        if p <= 0.0 {
            return Err(format!("knee power must be > 0, got {p}"));
        }
        Ok(Self {
            active,
            oklch: algorithm == "oklch",
            knee: (knee[0] as f64, knee[1] as f64, knee[2] as f64),
            locus: if active {
                spectral_locus_xy()
            } else {
                Vec::new()
            },
            oklch_table: OnceLock::new(),
        })
    }

    pub fn set_boundary(
        &mut self,
        boundary: &str,
        detail: f64,
        white: [f64; 2],
    ) -> Result<(), String> {
        if !detail.is_finite() || detail <= 0. {
            return Err("input gamut hull detail must be positive".into());
        }
        if !self.active {
            return Ok(());
        }
        if boundary == "locus" {
            self.locus = spectral_locus_xy();
            self.oklch_table.take();
            return Ok(());
        }
        if boundary != "inscribed_hull" {
            return Err(format!("unknown input gamut boundary: {boundary}"));
        }
        use rustfft::{FftPlanner, num_complex::Complex};
        let n = 1024usize;
        let directions: Vec<_> = (0..n)
            .map(|i| {
                let t = i as f64 * std::f64::consts::TAU / n as f64;
                [t.cos(), t.sin()]
            })
            .collect();
        let locus = spectral_locus_xy();
        let reach: Vec<_> = directions
            .iter()
            .map(|&d| ray_polygon_distance(white, d, &locus))
            .collect();
        if reach.iter().any(|v| !v.is_finite()) {
            return Err("input gamut hull has no finite boundary".into());
        }
        let mut frequencies: Vec<_> = reach.iter().map(|r| Complex::new(*r, 0.)).collect();
        let mut planner = FftPlanner::<f64>::new();
        planner.plan_fft_forward(n).process(&mut frequencies);
        for (i, f) in frequencies.iter_mut().enumerate() {
            let k = i.min(n - i) as f64;
            *f *= (-0.5 * (k / detail).powi(2)).exp();
        }
        planner.plan_fft_inverse(n).process(&mut frequencies);
        let radii: Vec<_> = frequencies
            .iter()
            .map(|f| (f.re / n as f64).max(1e-12))
            .collect();
        let inscribe = reach
            .iter()
            .zip(&radii)
            .map(|(a, b)| a / b)
            .fold(f64::INFINITY, f64::min);
        self.locus = directions
            .iter()
            .zip(radii)
            .map(|(d, r)| {
                [
                    white[0] + inscribe * r * d[0],
                    white[1] + inscribe * r * d[1],
                ]
            })
            .collect();
        self.locus.push(self.locus[0]);
        self.oklch_table.take();
        Ok(())
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn compress_xy(&self, xy: [f64; 2], reference_white: [f64; 2]) -> [f64; 2] {
        if !self.active {
            xy
        } else if self.oklch {
            compress_oklch(
                xy,
                self.knee,
                self.oklch_table.get_or_init(|| oklch_table(&self.locus)),
            )
        } else {
            compress_xy_radial(xy, reference_white, self.knee, &self.locus)
        }
    }

    fn xy_remap_coordinates(&self, size: usize, white: [f64; 2]) -> Arc<Vec<[f64; 2]>> {
        #[derive(PartialEq, Eq)]
        struct Key {
            size: usize,
            knee: [u64; 3],
            white: [u64; 2],
            locus: Vec<[u64; 2]>,
        }
        // Keep only the latest geometry; LUT values are always sampled afresh.
        static CACHE: Mutex<Option<(Key, Arc<Vec<[f64; 2]>>)>> = Mutex::new(None);
        let key = Key {
            size,
            knee: [self.knee.0, self.knee.1, self.knee.2].map(f64::to_bits),
            white: white.map(f64::to_bits),
            locus: self.locus.iter().map(|p| p.map(f64::to_bits)).collect(),
        };
        if let Some((cached_key, coordinates)) = CACHE.lock().unwrap().as_ref()
            && *cached_key == key
        {
            return Arc::clone(coordinates);
        }
        let inv = 1.0 / (size as f64 - 1.0);
        let coordinates = Arc::new(
            (0..size * size)
                .into_par_iter()
                .map(|index| {
                    let (x, y) = spectral::tc_to_xy(
                        (index / size) as f64 * inv,
                        (index % size) as f64 * inv,
                    );
                    let xy = compress_xy_radial([x, y], white, self.knee, &self.locus);
                    let (tx, ty) = spectral::xy_to_tc(xy[0], xy[1]);
                    [tx * (size as f64 - 1.0), ty * (size as f64 - 1.0)]
                })
                .collect(),
        );
        *CACHE.lock().unwrap() = Some((key, Arc::clone(&coordinates)));
        coordinates
    }

    /// Bake the compression into a freshly resampled copy of `lut`. For each
    /// LUT cell we decode its tc index to CIE xy, compress, re-encode to tc,
    /// and bilinearly sample the original LUT there (clamp-to-edge boundary,
    /// matching `scipy.ndimage.map_coordinates(order=1, mode="nearest")`).
    /// `ref_xy` is the film reference illuminant chromaticity — the
    /// compression's achromatic axis. Returns `lut` unchanged when inactive.
    pub fn remap(&self, lut: &TcLut, ref_xy: [f64; 2]) -> TcLut {
        if !self.active {
            return lut.clone();
        }
        let size = lut.size;
        let ch = lut.channels;
        let inv = 1.0 / (size as f64 - 1.0);
        let mut data = vec![0.0f64; size * size * ch];
        let coordinates = (!self.oklch).then(|| self.xy_remap_coordinates(size, ref_xy));
        let row_len = (size * ch).max(1);
        let write_row = |(i, row): (usize, &mut [f64])| {
            for (j, cell) in row.chunks_exact_mut(ch).enumerate() {
                let [ci, cj] = if let Some(coordinates) = &coordinates {
                    coordinates[i * size + j]
                } else {
                    let (x, y) = spectral::tc_to_xy(i as f64 * inv, j as f64 * inv);
                    let cxy = self.compress_xy([x, y], ref_xy);
                    let (tx, ty) = spectral::xy_to_tc(cxy[0], cxy[1]);
                    [tx * (size as f64 - 1.0), ty * (size as f64 - 1.0)]
                };
                let sample = bilinear_sample(lut, ci, cj);
                cell.copy_from_slice(&sample[..ch]);
            }
        };
        if self.oklch {
            // OKLCH lazily builds its own parallel lookup table.
            data.chunks_exact_mut(row_len)
                .enumerate()
                .for_each(write_row);
        } else {
            data.par_chunks_exact_mut(row_len)
                .enumerate()
                .for_each(write_row);
        }
        TcLut {
            size,
            channels: ch,
            data,
        }
    }
}

/// Bilinear sample of all channels at fractional grid coordinate `(ci, cj)`,
/// clamping fetch indices to the edge (scipy `mode="nearest"`).
fn bilinear_sample(lut: &TcLut, ci: f64, cj: f64) -> [f64; 3] {
    let size = lut.size;
    let ch = lut.channels;
    let max_idx = size as isize - 1;
    let clamp = |v: isize| v.clamp(0, max_idx) as usize;

    let bi = ci.floor();
    let bj = cj.floor();
    let fi = ci - bi;
    let fj = cj - bj;
    let (i0, i1) = (clamp(bi as isize), clamp(bi as isize + 1));
    let (j0, j1) = (clamp(bj as isize), clamp(bj as isize + 1));

    let cell = |i: usize, j: usize, c: usize| lut.data[(i * size + j) * ch + c];
    let mut out = [0.0f64; 3];
    for (c, slot) in out.iter_mut().enumerate().take(ch) {
        let v00 = cell(i0, j0, c);
        let v01 = cell(i0, j1, c);
        let v10 = cell(i1, j0, c);
        let v11 = cell(i1, j1, c);
        *slot = v00 * (1.0 - fi) * (1.0 - fj)
            + v01 * (1.0 - fi) * fj
            + v10 * fi * (1.0 - fj)
            + v11 * fi * fj;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic synthetic LUT, mirrored bit-for-bit in the Python
    /// reference (`tmp/igc_ref.py`): cell (i, j, c) = `((i² + 3j + 7c) % 97) / 97`.
    fn synthetic_lut(size: usize) -> TcLut {
        let mut data = vec![0.0f64; size * size * 3];
        for i in 0..size {
            for j in 0..size {
                for c in 0..3 {
                    let v = ((i * i + j * 3 + c * 7) % 97) as f64 / 97.0;
                    data[(i * size + j) * 3 + c] = v;
                }
            }
        }
        TcLut {
            size,
            channels: 3,
            data,
        }
    }

    fn serial_remap(spec: &InputGamutCompress, lut: &TcLut, white: [f64; 2]) -> Vec<f64> {
        let inv = 1. / (lut.size as f64 - 1.);
        let mut expected = Vec::with_capacity(lut.data.len());
        for i in 0..lut.size {
            for j in 0..lut.size {
                let xy = spectral::tc_to_xy(i as f64 * inv, j as f64 * inv);
                let xy = spec.compress_xy([xy.0, xy.1], white);
                let (tx, ty) = spectral::xy_to_tc(xy[0], xy[1]);
                let sample = bilinear_sample(
                    lut,
                    tx * (lut.size as f64 - 1.),
                    ty * (lut.size as f64 - 1.),
                );
                expected.extend_from_slice(&sample[..lut.channels]);
            }
        }
        expected
    }

    /// The baked tc_lut remap matches the upstream remap-resample
    /// (`remap_tc_lut_for_compression` with the `"xy"` algorithm) against the
    /// CIE 1931 2° locus, sampled with scipy `map_coordinates(order=1,
    /// mode="nearest")`.
    #[test]
    fn remap_matches_python_reference() {
        let spec = InputGamutCompress::build("xy", [0.0, 1.0, 6.0]).unwrap();
        let lut = synthetic_lut(64);
        let out = spec.remap(&lut, [0.3127, 0.3290]);

        // (i, j) → [r, g, b] from the Python reference.
        let cells = [
            (0usize, 0usize, [0.4137658005, 0.1877218965, 0.2598868450]),
            (10, 20, [0.7521338372, 0.8242987857, 0.8964637341]),
            (32, 32, [0.5463917966, 0.6185567450, 0.6907215587]),
            (50, 5, [0.5807479510, 0.0691153288, 0.1412802772]),
            (63, 63, [0.6891891640, 0.7613541125, 0.8335190609]),
            (5, 60, [0.0995187055, 0.1716836540, 0.2438486024]),
        ];
        for (i, j, expected) in cells {
            let base = (i * 64 + j) * 3;
            for c in 0..3 {
                assert!(
                    (out.data[base + c] - expected[c]).abs() < 1e-7,
                    "cell ({i},{j}) ch {c}: got {}, expected {}",
                    out.data[base + c],
                    expected[c],
                );
            }
        }

        let checksum: f64 = out.data.iter().sum();
        assert!(
            (checksum - 5846.5426164588).abs() < 1e-6,
            "checksum {checksum}",
        );
    }

    /// Worker count does not change the serial per-cell arithmetic.
    #[test]
    fn parallel_remap_is_bit_identical_to_serial_reference() {
        let lut = synthetic_lut(31);
        let white = [0.3127, 0.3290];
        for boundary in ["locus", "inscribed_hull"] {
            let mut spec = InputGamutCompress::build("xy", [0.815, 1., 1.2]).unwrap();
            spec.set_boundary(boundary, 5., white).unwrap();
            let expected = serial_remap(&spec, &lut, white);
            for threads in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let output = pool.install(|| spec.remap(&lut, white));
                assert_eq!(output.data.len(), expected.len());
                assert!(
                    output
                        .data
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{boundary}, {threads} threads changed a LUT sample"
                );
            }
        }
    }

    #[test]
    fn cached_xy_remap_tracks_geometry_and_samples_current_lut() {
        let white = [0.3127, 0.3290];
        for (knee, boundary, detail, ref_xy, size) in [
            ([0.815, 1., 1.2], "locus", 5., white, 17),
            ([0.815, 1., 1.2], "locus", 5., [0.3457, 0.3585], 17),
            ([0.7, 1., 1.2], "locus", 5., [0.3457, 0.3585], 17),
            ([0.7, 1.1, 1.2], "locus", 5., [0.3457, 0.3585], 17),
            ([0.7, 1.1, 2.], "locus", 5., [0.3457, 0.3585], 17),
            ([0.7, 1.1, 2.], "locus", 5., [0.3457, 0.3585], 19),
            ([0.7, 1.1, 2.], "inscribed_hull", 5., [0.3457, 0.3585], 19),
            ([0.7, 1.1, 2.], "inscribed_hull", 6., [0.3457, 0.3585], 19),
        ] {
            let mut spec = InputGamutCompress::build("xy", knee).unwrap();
            spec.set_boundary(boundary, detail, white).unwrap();
            for (channels, offset) in [(3, 0.), (3, 2.), (1, -0.5)] {
                let lut = TcLut {
                    size,
                    channels,
                    data: synthetic_lut(size)
                        .data
                        .chunks_exact(3)
                        .flat_map(|p| p[..channels].iter().map(|v| v + offset))
                        .collect(),
                };
                let expected = serial_remap(&spec, &lut, ref_xy);
                let output = spec.remap(&lut, ref_xy);
                assert_eq!(output.data.len(), expected.len());
                assert!(
                    output
                        .data
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "stale mapping or LUT values: {knee:?}, {boundary}, {detail}, {ref_xy:?}, {size}, {channels}, {offset}"
                );
            }
        }
    }

    #[test]
    fn oklch_tables_follow_each_compressors_boundary() {
        let mut wide = InputGamutCompress::build("oklch", [0.7, 1., 1.2]).unwrap();
        let mut narrow = InputGamutCompress::build("oklch", [0.7, 1., 1.2]).unwrap();
        let square = |radius| {
            vec![
                [0.33 - radius, 0.33 - radius],
                [0.33 + radius, 0.33 - radius],
                [0.33 + radius, 0.33 + radius],
                [0.33 - radius, 0.33 + radius],
                [0.33 - radius, 0.33 - radius],
            ]
        };
        wide.locus = square(0.2);
        narrow.locus = square(0.05);
        let xy = [0.7, 0.2];
        let a = wide.compress_xy(xy, [0.3127, 0.329]);
        let b = narrow.compress_xy(xy, [0.3127, 0.329]);
        assert_ne!(a, b, "a different boundary must not reuse the first table");
        assert_eq!(wide.compress_xy(xy, [0.3127, 0.329]), a);

        narrow.set_boundary("locus", 5., [0.3127, 0.329]).unwrap();
        let fresh = InputGamutCompress::build("oklch", [0.7, 1., 1.2]).unwrap();
        assert_eq!(
            narrow.compress_xy(xy, [0.3127, 0.329]),
            fresh.compress_xy(xy, [0.3127, 0.329])
        );
    }

    #[test]
    fn changing_boundary_replaces_instead_of_repeatedly_smoothing_it() {
        let mut changed = InputGamutCompress::build("xy", [0.7, 1., 1.2]).unwrap();
        let mut fresh = InputGamutCompress::build("xy", [0.7, 1., 1.2]).unwrap();
        let white = [0.3127, 0.329];
        changed.set_boundary("inscribed_hull", 5., white).unwrap();
        changed.set_boundary("inscribed_hull", 8., white).unwrap();
        fresh.set_boundary("inscribed_hull", 8., white).unwrap();
        assert_eq!(changed.locus, fresh.locus);
        changed.set_boundary("locus", 5., white).unwrap();
        assert_eq!(changed.locus, spectral_locus_xy());
    }

    /// An inactive ("off") compressor returns the LUT unchanged.
    #[test]
    fn off_is_identity() {
        let spec = InputGamutCompress::build("off", [0.0, 1.0, 6.0]).unwrap();
        assert!(!spec.is_active());
        let lut = synthetic_lut(16);
        let out = spec.remap(&lut, [0.3127, 0.3290]);
        assert_eq!(out.data, lut.data);
    }
}

fn contains(p: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let mut inside = false;
    for pair in polygon.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if (a[1] > p[1]) != (b[1] > p[1])
            && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            inside = !inside;
        }
    }
    inside
}
const NL: usize = 64;
const NH: usize = 720;

fn oklch_table(locus: &[[f64; 2]]) -> Arc<Vec<f64>> {
    use crate::gamut_compression::oklab_to_xyz;
    // Keep one shared boundary; each compressor retains its own immutable table.
    static CACHE: Mutex<Option<(Vec<[u64; 2]>, Arc<Vec<f64>>)>> = Mutex::new(None);
    let key = locus
        .iter()
        .map(|p| p.map(f64::to_bits))
        .collect::<Vec<_>>();
    if let Some((cached_key, table)) = CACHE.lock().unwrap().as_ref()
        && *cached_key == key
    {
        return Arc::clone(table);
    }
    let table = Arc::new(
        (0..NL * NH)
            .into_par_iter()
            .map(|i| {
                let l = 0.05 + 0.95 * (i / NH) as f64 / (NL - 1) as f64;
                let h =
                    -std::f64::consts::PI + 2. * std::f64::consts::PI * (i % NH) as f64 / NH as f64;
                let (mut lo, mut hi) = (0., 0.5);
                for _ in 0..18 {
                    let mid = (lo + hi) / 2.;
                    let xyz = oklab_to_xyz([l, mid * h.cos(), mid * h.sin()]);
                    let sum: f64 = xyz.iter().sum();
                    if contains([xyz[0] / sum, xyz[1] / sum], locus) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                lo
            })
            .collect(),
    );
    *CACHE.lock().unwrap() = Some((key, Arc::clone(&table)));
    table
}

fn compress_oklch(xy: [f64; 2], knee: (f64, f64, f64), table: &[f64]) -> [f64; 2] {
    use crate::gamut_compression::{oklab_to_xyz, xyz_to_oklab};
    let y = xy[1].max(1e-9);
    let lab = xyz_to_oklab([xy[0] / y, 1., (1. - xy[0] - xy[1]) / y]);
    let h = lab[2].atan2(lab[1]);
    let c = lab[1].hypot(lab[2]);
    let li = (lab[0].clamp(0.05, 1.) - 0.05) / 0.95 * (NL - 1) as f64;
    let l0 = (li.floor() as usize).min(NL - 2);
    let lf = li - l0 as f64;
    let hj = (h + std::f64::consts::PI) / (2. * std::f64::consts::PI) * NH as f64;
    let h0 = (hj.floor() as usize) % NH;
    let h1 = (h0 + 1) % NH;
    let hf = hj - hj.floor();
    let cm = (table[l0 * NH + h0] * (1. - lf) * (1. - hf)
        + table[l0 * NH + h1] * (1. - lf) * hf
        + table[(l0 + 1) * NH + h0] * lf * (1. - hf)
        + table[(l0 + 1) * NH + h1] * lf * hf)
        .max(1e-9);
    let nc = reinhard_knee(c / cm, knee.0, knee.1, knee.2) * cm;
    let xyz = oklab_to_xyz([lab[0], nc * h.cos(), nc * h.sin()]);
    let sum: f64 = xyz.iter().sum();
    [xyz[0] / sum, xyz[1] / sum]
}
