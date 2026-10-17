use spektrafilm_math::image::ImageBuf;

/// Crop sizes are fractions of the original long edge, with NumPy's ties-even rounding.
pub fn crop(image: &ImageBuf, center: [f32; 2], size: [f32; 2]) -> ImageBuf {
    let long = image.width.max(image.height) as f64;
    let w = (long * size[0] as f64)
        .round_ties_even()
        .max(1.)
        .min(image.width as f64) as u32;
    let h = (long * size[1] as f64)
        .round_ties_even()
        .max(1.)
        .min(image.height as f64) as u32;
    let x = ((image.width as f64 * center[0] as f64).round_ties_even() - w as f64 / 2.)
        .round_ties_even()
        .clamp(0., (image.width - w) as f64) as usize;
    let y = ((image.height as f64 * center[1] as f64).round_ties_even() - h as f64 / 2.)
        .round_ties_even()
        .clamp(0., (image.height - h) as f64) as usize;
    let mut data = Vec::with_capacity(w as usize * h as usize * 3);
    for row in y..y + h as usize {
        let start = (row * image.width as usize + x) * 3;
        data.extend_from_slice(&image.data[start..start + w as usize * 3]);
    }
    ImageBuf::from_data(w, h, data)
}

/// skimage.rescale(order=3, mode="reflect") with anti-aliasing, B-spline
/// prefiltering, pixel-center coordinates, and global range preservation.
pub fn rescale(image: &ImageBuf, factor: f64) -> ImageBuf {
    if (factor - 1.).abs() < 1e-12 {
        return image.clone();
    }
    let w = image.width as usize;
    let h = image.height as usize;
    let nw = (w as f64 * factor).round_ties_even().max(1.) as usize;
    let nh = (h as f64 * factor).round_ties_even().max(1.) as usize;
    resize(image, nw, nh, 3)
}

/// Python's anti-aliased order-1 preview resize, evaluated only at the
/// destination sample locations. A vertical intermediate replaces two
/// full-resolution f64 filter buffers.
pub fn preview(image: &ImageBuf, max_size: u32) -> ImageBuf {
    let long = image.width.max(image.height);
    if long <= max_size || max_size == 0 {
        return image.clone();
    }
    let factor = max_size as f64 / long as f64;
    let (w, h) = (image.width as usize, image.height as usize);
    let (nw, nh) = (
        (w as f64 * factor).floor().max(1.) as usize,
        (h as f64 * factor).floor().max(1.) as usize,
    );
    sampled_preview(image, nw, nh, 1)
}

/// Python's metering thumbnail: ties-even output size, anti-aliasing in
/// encoded sample space, then nearest-neighbor sampling (rescale order=0).
pub fn metering_preview(image: &ImageBuf, max_size: u32) -> ImageBuf {
    let long = image.width.max(image.height);
    if long <= max_size || max_size == 0 {
        return image.clone();
    }
    let scale = max_size as f64 / long as f64;
    let nw = (image.width as f64 * scale).round_ties_even().max(1.) as usize;
    let nh = (image.height as f64 * scale).round_ties_even().max(1.) as usize;
    sampled_preview(image, nw, nh, 0)
}
fn sampled_preview(image: &ImageBuf, nw: usize, nh: usize, order: u8) -> ImageBuf {
    use rayon::prelude::*;
    let (w, h) = (image.width as usize, image.height as usize);
    let taps = |source: usize, destination: usize| -> Vec<Vec<(usize, f64)>> {
        let scale = source as f64 / destination as f64;
        let sigma = ((scale - 1.) / 2.).max(0.);
        let radius = (4. * sigma + 0.5) as isize;
        let kernel: Vec<f64> = (-radius..=radius)
            .map(|i| {
                if sigma > 0. {
                    (-0.5 * (i as f64 / sigma).powi(2)).exp()
                } else {
                    1.
                }
            })
            .collect();
        let norm = kernel.iter().sum::<f64>();
        (0..destination)
            .map(|i| {
                let pos = (i as f64 + 0.5) * scale - 0.5;
                let lo = pos.floor() as isize;
                let fraction = pos - lo as f64;
                let mut weights = std::collections::BTreeMap::<usize, f64>::new();
                let samples = if order == 0 {
                    vec![(pos.round() as isize, 1.)]
                } else {
                    vec![(lo, 1. - fraction), (lo + 1, fraction)]
                };
                for (base, weight) in samples {
                    let base = mirror(base, source) as isize;
                    for (j, &k) in kernel.iter().enumerate() {
                        *weights
                            .entry(mirror(base + j as isize - radius, source))
                            .or_default() += weight * k / norm;
                    }
                }
                weights.into_iter().collect()
            })
            .collect()
    };
    let vertical = taps(h, nh);
    let horizontal = taps(w, nw);
    let mut mid = vec![0f64; nh * w * 3];
    mid.par_chunks_exact_mut(w * 3)
        .enumerate()
        .for_each(|(y, row)| {
            for &(source, weight) in &vertical[y] {
                let input = &image.data[source * w * 3..(source + 1) * w * 3];
                for (dst, &v) in row.iter_mut().zip(input) {
                    *dst += v as f64 * weight;
                }
            }
        });
    let (min, max) = if order == 0 {
        (f64::NEG_INFINITY, f64::INFINITY)
    } else {
        image
            .data
            .par_iter()
            .map(|v| (*v as f64, *v as f64))
            .reduce(
                || (f64::INFINITY, f64::NEG_INFINITY),
                |(a, b), (c, d)| (a.min(c), b.max(d)),
            )
    };
    let mut out = ImageBuf::new(nw as u32, nh as u32);
    out.data
        .par_chunks_exact_mut(nw * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let source = &mid[y * w * 3..(y + 1) * w * 3];
            for (x, pixel) in row.chunks_exact_mut(3).enumerate() {
                let mut value = [0.; 3];
                for &(sample, weight) in &horizontal[x] {
                    for c in 0..3 {
                        value[c] += source[sample * 3 + c] * weight;
                    }
                }
                for c in 0..3 {
                    pixel[c] = spektrafilm_math::precision::from_f64(value[c].clamp(min, max));
                }
            }
        });
    out
}

fn resize(image: &ImageBuf, nw: usize, nh: usize, order: u8) -> ImageBuf {
    let w = image.width as usize;
    let h = image.height as usize;
    let mut samples: Vec<f64> = image.data.iter().map(|v| *v as f64).collect();
    let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for (horizontal, scale) in [(false, h as f64 / nh as f64), (true, w as f64 / nw as f64)] {
        let sigma = ((scale - 1.) / 2.).max(0.);
        if sigma > 0. {
            samples = filter_axis(&samples, w, h, horizontal, |line| {
                let r = (4. * sigma + 0.5) as isize;
                let weights: Vec<_> = (-r..=r)
                    .map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp())
                    .collect();
                let norm: f64 = weights.iter().sum();
                (0..line.len())
                    .map(|i| {
                        weights
                            .iter()
                            .enumerate()
                            .map(|(j, k)| {
                                line[mirror(i as isize + j as isize - r, line.len())] * k / norm
                            })
                            .sum()
                    })
                    .collect()
            });
        }
    }
    if order == 3 {
        for horizontal in [false, true] {
            samples = filter_axis(&samples, w, h, horizontal, spline_coefficients);
        }
    }
    let mut data = vec![spektrafilm_math::precision::from_f64(0.); nw * nh * 3];
    for y in 0..nh {
        let sy = (y as f64 + 0.5) * h as f64 / nh as f64 - 0.5;
        let iy = sy.floor() as isize;
        for x in 0..nw {
            let sx = (x as f64 + 0.5) * w as f64 / nw as f64 - 0.5;
            let ix = sx.floor() as isize;
            for c in 0..3 {
                let mut value = 0.;
                let range = if order == 3 { -1..=2 } else { 0..=1 };
                for dy in range.clone() {
                    for dx in range.clone() {
                        value += samples[(mirror(iy + dy, h) * w + mirror(ix + dx, w)) * 3 + c]
                            * weight(sy - (iy + dy) as f64, order)
                            * weight(sx - (ix + dx) as f64, order);
                    }
                }
                data[(y * nw + x) * 3 + c] =
                    spektrafilm_math::precision::from_f64(value.clamp(min, max));
            }
        }
    }
    ImageBuf::from_data(nw as u32, nh as u32, data)
}
fn mirror(i: isize, n: usize) -> usize {
    if n < 2 {
        0
    } else {
        let period = 2 * (n - 1) as isize;
        let j = i.rem_euclid(period);
        if j >= n as isize {
            (period - j) as usize
        } else {
            j as usize
        }
    }
}
fn cubic(d: f64) -> f64 {
    let d = d.abs();
    if d < 1. {
        (4. - 6. * d * d + 3. * d * d * d) / 6.
    } else if d < 2. {
        (2. - d).powi(3) / 6.
    } else {
        0.
    }
}
fn filter_axis(
    data: &[f64],
    w: usize,
    h: usize,
    horizontal: bool,
    filter: impl Fn(&[f64]) -> Vec<f64>,
) -> Vec<f64> {
    let mut out = data.to_vec();
    let n = if horizontal { w } else { h };
    let lines = if horizontal { h } else { w };
    for line in 0..lines {
        for c in 0..3 {
            let index = |i| {
                if horizontal {
                    (line * w + i) * 3 + c
                } else {
                    (i * w + line) * 3 + c
                }
            };
            let input: Vec<_> = (0..n).map(|i| data[index(i)]).collect();
            let result = filter(&input);
            for i in 0..n {
                out[index(i)] = result[i];
            }
        }
    }
    out
}
fn spline_coefficients(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n < 2 {
        return data.to_vec();
    }
    // Mirror endpoint conditions: 4*c0+2*c1=6*x0, 2*c[n-2]+4*c[n-1]=6*x[n-1].
    let mut diag = vec![4.; n];
    let mut rhs: Vec<_> = data.iter().map(|v| 6. * v).collect();
    for i in 1..n {
        let lower = if i + 1 == n { 2. } else { 1. };
        let upper = if i == 1 { 2. } else { 1. };
        let f = lower / diag[i - 1];
        diag[i] -= f * upper;
        rhs[i] -= f * rhs[i - 1];
    }
    rhs[n - 1] /= diag[n - 1];
    for i in (0..n - 1).rev() {
        rhs[i] = (rhs[i] - (if i == 0 { 2. } else { 1. }) * rhs[i + 1]) / diag[i];
    }
    rhs
}

fn weight(d: f64, order: u8) -> f64 {
    if order == 3 {
        cubic(d)
    } else {
        (1. - d.abs()).max(0.)
    }
}
