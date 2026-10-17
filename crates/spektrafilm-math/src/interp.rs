use crate::image::ImageBuf;
use crate::precision::{Scalar, from_f32, from_f64};
/// Fast 1D linear interpolation with binary search.
///
/// Port of Python `fast_interp.py`. Supports both common x-axis (same for all channels)
/// and per-channel x-axes.
use rayon::prelude::*;

/// Interpolate a single value from sorted arrays.
#[inline]
pub fn interp_1d(x: &[f32], y: &[f32], xq: f32) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    debug_assert!(!x.is_empty());

    let n = x.len();
    if n == 1 {
        return y[0];
    }
    if xq <= x[0] {
        return y[0];
    }
    if xq >= x[n - 1] {
        return y[n - 1];
    }

    let i = match x.binary_search_by(|v| v.partial_cmp(&xq).unwrap()) {
        Ok(i) => return y[i],
        Err(i) => i - 1,
    };

    let t = (xq - x[i]) / (x[i + 1] - x[i]);
    y[i] + t * (y[i + 1] - y[i])
}

/// NumPy-compatible right-biased interpolation, including repeated knots.
pub fn interp_1d_f64(x: &[f64], y: &[f64], query: f64) -> f64 {
    debug_assert!(!x.is_empty() && x.len() == y.len());
    if query < x[0] {
        return y[0];
    }
    let upper = x.partition_point(|v| *v <= query);
    if upper >= x.len() {
        return y[y.len() - 1];
    }
    if upper == 0 {
        return y[0];
    }
    let i = upper - 1;
    y[i] + (query - x[i]) / (x[upper] - x[i]) * (y[upper] - y[i])
}

/// NumPy's batch search order matters when strong DIR inhibition folds the axis.
/// Search semantics follow NumPy compiled_base.c (BSD-3-Clause, data/licenses).
pub fn interp_batch_f64(axis: &[f64], values: &[f64], queries: &[f64]) -> Vec<f64> {
    let n = axis.len();
    let mut guess = 0isize;
    queries
        .iter()
        .map(|&q| {
            if q.is_nan() {
                return q;
            }
            if q > axis[n - 1] {
                guess = n as isize;
                return values[n - 1];
            }
            if q < axis[0] {
                guess = -1;
                return values[0];
            }
            let mut lo = 0usize;
            let mut hi = n;
            let mut found = None;
            if n <= 4 {
                lo = 1;
                while lo < n && q >= axis[lo] {
                    lo += 1;
                }
                found = Some(lo - 1);
            } else {
                let g = guess.clamp(1, (n - 3) as isize) as usize;
                if q < axis[g] {
                    if q < axis[g - 1] {
                        hi = g - 1;
                        if g > 8 && q >= axis[g - 8] {
                            lo = g - 8;
                        }
                    } else {
                        found = Some(g - 1);
                    }
                } else if q < axis[g + 1] {
                    found = Some(g);
                } else if q < axis[g + 2] {
                    found = Some(g + 1);
                } else {
                    lo = g + 2;
                    if g + 9 < n && q < axis[g + 8] {
                        hi = g + 8;
                    }
                }
            }
            let index = found.unwrap_or_else(|| {
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if q >= axis[mid] {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                lo.saturating_sub(1)
            });
            guess = index as isize;
            if index == n - 1 || axis[index] == q {
                values[index]
            } else {
                (values[index + 1] - values[index]) / (axis[index + 1] - axis[index])
                    * (q - axis[index])
                    + values[index]
            }
        })
        .collect()
}

/// Interpolate a uniformly-spaced table.
#[inline]
pub fn interp_uniform(x_min: f32, x_max: f32, y: &[f32], xq: f32) -> f32 {
    let n = y.len();
    if n == 1 {
        return y[0];
    }

    let step = (x_max - x_min) / (n as f32 - 1.0);
    let t = (xq - x_min) / step;

    if t <= 0.0 {
        return y[0];
    }
    if t >= (n - 1) as f32 {
        return y[n - 1];
    }

    let i = t as usize;
    let frac = t - i as f32;
    y[i] + frac * (y[i + 1] - y[i])
}

/// Batch interpolation: uniform table for 3 channels.
#[inline]
pub fn interp_uniform_3ch(x_min: f32, x_max: f32, table: &[[f32; 3]], xq: f32) -> [f32; 3] {
    let n = table.len();
    if n == 1 {
        return table[0];
    }

    let step = (x_max - x_min) / (n as f32 - 1.0);
    let t = (xq - x_min) / step;

    if t <= 0.0 {
        return table[0];
    }
    if t >= (n - 1) as f32 {
        return table[n - 1];
    }

    let i = t as usize;
    let frac = t - i as f32;
    [
        table[i][0] + frac * (table[i + 1][0] - table[i][0]),
        table[i][1] + frac * (table[i + 1][1] - table[i][1]),
        table[i][2] + frac * (table[i + 1][2] - table[i][2]),
    ]
}

/// Fast image interpolation: for each pixel (HxWx3), look up each channel
/// independently in a common x-axis → y-values table.
///
/// This is the direct Rust port of the Python `fast_interp` Numba kernel.
/// `x_axis`: sorted 1D array of length K.
/// `y_vals`: [K][3] array of y-values per channel.
pub fn fast_interp_image(img: &ImageBuf, x_axis: &[f32], y_vals: &[[f32; 3]]) -> ImageBuf {
    let k = x_axis.len();
    assert_eq!(y_vals.len(), k);

    // Promote constant data to Scalar once (no-op in f32 mode, widening in f64 mode).
    let xa: Vec<Scalar> = x_axis.iter().map(|&v| from_f32(v)).collect();
    let ya: Vec<[Scalar; 3]> = y_vals
        .iter()
        .map(|r| [from_f32(r[0]), from_f32(r[1]), from_f32(r[2])])
        .collect();
    let inv_dx: Vec<Scalar> = (0..k - 1)
        .map(|i| {
            let dx = xa[i + 1] - xa[i];
            if dx != 0.0 { 1.0 / dx } else { 0.0 }
        })
        .collect();

    let mut out = img.clone();
    out.par_pixels_mut().for_each(|px| {
        for c in 0..3 {
            let x = px[c];
            if x <= xa[0] {
                px[c] = ya[0][c];
            } else if x >= xa[k - 1] {
                px[c] = ya[k - 1][c];
            } else {
                let idx = xa.partition_point(|&v| v < x);
                let low = if idx > 0 { idx - 1 } else { 0 };
                let t = (x - xa[low]) * inv_dx[low];
                px[c] = ya[low][c] + t * (ya[low + 1][c] - ya[low][c]);
            }
        }
    });
    out
}

/// f64 variant of fast_interp_image — accepts f64 axes/curves directly
/// to preserve precision (no f32 truncation at the JSON boundary).
pub fn fast_interp_image_f64(img: &ImageBuf, x_axis: &[f64], y_vals: &[[f64; 3]]) -> ImageBuf {
    let k = x_axis.len();
    assert_eq!(y_vals.len(), k);

    // Promote to Scalar once (no-op in f64 mode, narrowing in f32 mode).
    let xa: Vec<Scalar> = x_axis.iter().map(|&v| from_f64(v)).collect();
    let ya: Vec<[Scalar; 3]> = y_vals
        .iter()
        .map(|r| [from_f64(r[0]), from_f64(r[1]), from_f64(r[2])])
        .collect();
    let inv_dx: Vec<Scalar> = (0..k - 1)
        .map(|i| {
            let dx = xa[i + 1] - xa[i];
            if dx != 0.0 { 1.0 / dx } else { 0.0 }
        })
        .collect();

    let mut out = img.clone();
    out.par_pixels_mut().for_each(|px| {
        for c in 0..3 {
            let x = px[c];
            if x <= xa[0] {
                px[c] = ya[0][c];
            } else if x >= xa[k - 1] {
                px[c] = ya[k - 1][c];
            } else {
                // Match numpy's searchsorted(side='right') - 1: include equal values on the left side.
                let idx = xa.partition_point(|&v| v <= x);
                let low = if idx > 0 { idx - 1 } else { 0 };
                let t = (x - xa[low]) * inv_dx[low];
                px[c] = ya[low][c] + t * (ya[low + 1][c] - ya[low][c]);
            }
        }
    });
    out
}

/// f64 variant with per-channel x-axes.
pub fn fast_interp_image_perchannel_f64(
    img: &ImageBuf,
    x_axes: &[[f64; 3]],
    y_vals: &[[f64; 3]],
) -> ImageBuf {
    let k = x_axes.len();
    assert_eq!(y_vals.len(), k);

    let xax: Vec<[Scalar; 3]> = x_axes
        .iter()
        .map(|r| [from_f64(r[0]), from_f64(r[1]), from_f64(r[2])])
        .collect();
    let ya: Vec<[Scalar; 3]> = y_vals
        .iter()
        .map(|r| [from_f64(r[0]), from_f64(r[1]), from_f64(r[2])])
        .collect();
    let per_chan_axes: [Vec<Scalar>; 3] = [
        xax.iter().map(|r| r[0]).collect(),
        xax.iter().map(|r| r[1]).collect(),
        xax.iter().map(|r| r[2]).collect(),
    ];
    let inv_dx: Vec<[Scalar; 3]> = (0..k - 1)
        .map(|i| {
            let mut id = [Scalar::default(); 3];
            for c in 0..3 {
                let dx = xax[i + 1][c] - xax[i][c];
                id[c] = if dx != 0.0 { 1.0 / dx } else { 0.0 };
            }
            id
        })
        .collect();

    let mut out = img.clone();
    out.par_pixels_mut().for_each(|px| {
        for c in 0..3 {
            let x = px[c];
            let x0 = xax[0][c];
            let xn = xax[k - 1][c];
            if x <= x0 {
                px[c] = ya[0][c];
            } else if x >= xn {
                px[c] = ya[k - 1][c];
            } else {
                let xa = &per_chan_axes[c];
                let idx = xa.partition_point(|&v| v <= x);
                let low = if idx > 0 { idx - 1 } else { 0 };
                let t = (x - xax[low][c]) * inv_dx[low][c];
                px[c] = ya[low][c] + t * (ya[low + 1][c] - ya[low][c]);
            }
        }
    });
    out
}

/// Fast image interpolation with per-channel x-axes.
///
/// `x_axes`: [K][3] — different x-axis per channel (e.g. log_exposure / gamma_factor).
/// `y_vals`: [K][3] — y-values per channel.
pub fn fast_interp_image_perchannel(
    img: &ImageBuf,
    x_axes: &[[f32; 3]],
    y_vals: &[[f32; 3]],
) -> ImageBuf {
    let k = x_axes.len();
    assert_eq!(y_vals.len(), k);

    // Promote constant data to Scalar once. Pre-extract per-channel axes for the binary
    // search inside the parallel loop (avoids per-pixel allocation).
    let xax: Vec<[Scalar; 3]> = x_axes
        .iter()
        .map(|r| [from_f32(r[0]), from_f32(r[1]), from_f32(r[2])])
        .collect();
    let ya: Vec<[Scalar; 3]> = y_vals
        .iter()
        .map(|r| [from_f32(r[0]), from_f32(r[1]), from_f32(r[2])])
        .collect();
    let per_chan_axes: [Vec<Scalar>; 3] = [
        xax.iter().map(|r| r[0]).collect(),
        xax.iter().map(|r| r[1]).collect(),
        xax.iter().map(|r| r[2]).collect(),
    ];
    let inv_dx: Vec<[Scalar; 3]> = (0..k - 1)
        .map(|i| {
            let mut id = [Scalar::default(); 3];
            for c in 0..3 {
                let dx = xax[i + 1][c] - xax[i][c];
                id[c] = if dx != 0.0 { 1.0 / dx } else { 0.0 };
            }
            id
        })
        .collect();

    let mut out = img.clone();
    out.par_pixels_mut().for_each(|px| {
        for c in 0..3 {
            let x = px[c];
            let x0 = xax[0][c];
            let xn = xax[k - 1][c];
            if x <= x0 {
                px[c] = ya[0][c];
            } else if x >= xn {
                px[c] = ya[k - 1][c];
            } else {
                let xa = &per_chan_axes[c];
                let idx = xa.partition_point(|&v| v < x);
                let low = if idx > 0 { idx - 1 } else { 0 };
                let t = (x - xax[low][c]) * inv_dx[low][c];
                px[c] = ya[low][c] + t * (ya[low + 1][c] - ya[low][c]);
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interp_1d_exact() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 10.0, 20.0, 30.0];
        assert_eq!(interp_1d(&x, &y, 1.0), 10.0);
        assert_eq!(interp_1d(&x, &y, 2.0), 20.0);
    }

    #[test]
    fn test_interp_1d_midpoint() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 10.0, 20.0];
        let v = interp_1d(&x, &y, 0.5);
        assert!((v - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_interp_1d_clamp() {
        let x = vec![0.0, 1.0];
        let y = vec![5.0, 15.0];
        assert_eq!(interp_1d(&x, &y, -1.0), 5.0);
        assert_eq!(interp_1d(&x, &y, 2.0), 15.0);
    }

    #[test]
    fn test_interp_uniform() {
        let y = vec![0.0, 10.0, 20.0, 30.0];
        let v = interp_uniform(0.0, 3.0, &y, 1.5);
        assert!((v - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_interp_uniform_3ch() {
        let table = vec![[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]];
        let v = interp_uniform_3ch(0.0, 1.0, &table, 0.5);
        assert!((v[0] - 5.0).abs() < 1e-6);
        assert!((v[1] - 10.0).abs() < 1e-6);
        assert!((v[2] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_fast_interp_image() {
        use crate::precision::from_f64;
        let x_axis = vec![0.0_f32, 0.5, 1.0];
        let y_vals = vec![[0.0_f32, 0.0, 0.0], [5.0, 10.0, 15.0], [10.0, 20.0, 30.0]];
        let img = ImageBuf::from_data(
            2,
            1,
            vec![
                from_f64(0.25),
                from_f64(0.25),
                from_f64(0.25),
                from_f64(0.75),
                from_f64(0.75),
                from_f64(0.75),
            ],
        );
        let out = fast_interp_image(&img, &x_axis, &y_vals);
        let p0 = out.get(0, 0);
        assert!((p0[0] - from_f64(2.5)).abs() < from_f64(1e-4));
        assert!((p0[1] - from_f64(5.0)).abs() < from_f64(1e-4));
        assert!((p0[2] - from_f64(7.5)).abs() < from_f64(1e-4));
    }
}
