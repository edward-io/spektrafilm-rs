//! Scene-linear chroma denoising with noise-adaptive, directional edge protection.
use crate::{
    image::ImageBuf,
    precision::{Scalar, from_f64, to_f64},
    transfer::Transfer,
};
use rayon::prelude::*;
use std::ops::Range;

const PREVIEW_TILE_WIDTH: usize = 1024;
const PREVIEW_TILE_HEIGHT: usize = 256;
// Covers the support of the widest pass and the filters that feed it.
const HALO: usize = 144;
type Chroma = [f32; 2];

/// Apply the filter at the supplied image resolution. Amount is a linear mix
/// with the original chroma. Luminance and linear highlight headroom are retained.
pub fn apply(input: &ImageBuf, luma: [f64; 3], transfer: Transfer, amount: f32) -> ImageBuf {
    if amount <= 0. {
        return input.clone();
    }
    mix(input, filter_full(input, luma, transfer), transfer, amount)
}

/// Prepare the full-strength endpoint once for repeated slider changes.
pub fn filter_full(input: &ImageBuf, luma: [f64; 3], transfer: Transfer) -> ImageBuf {
    if input.width < 2 || input.height < 2 || luma[1] <= 0. {
        return input.clone();
    }
    let sum: f64 = luma.iter().sum();
    let luma = luma.map(|v| v / sum);
    let w = input.width as usize;
    let h = input.height as usize;
    // Shorter bands expose more parallel work on previews. Larger frames
    // already have more bands, so use square tiles to limit halo overlap.
    let (tile_width, tile_height) = if h <= 2048 {
        (PREVIEW_TILE_WIDTH, PREVIEW_TILE_HEIGHT)
    } else {
        (512, 512)
    };
    let mut output = input.clone();
    output
        .data
        .par_chunks_mut(w * 3 * tile_height)
        .enumerate()
        .for_each(|(band, output)| {
            let top = band * tile_height;
            let bottom = (top + tile_height).min(h);
            for left in (0..w).step_by(tile_width) {
                let right = (left + tile_width).min(w);
                let x0 = left.saturating_sub(HALO);
                let y0 = top.saturating_sub(HALO);
                let size = Grid {
                    w: (right + HALO).min(w) - x0,
                    h: (bottom + HALO).min(h) - y0,
                };
                let mut ys = Vec::with_capacity(size.len());
                let mut deltas = Vec::with_capacity(size.len());
                for y in y0..y0 + size.h {
                    for x in x0..x0 + size.w {
                        let rgb = linear_rgb(&input.data, y * w + x, transfer);
                        let lum = dot(rgb, luma);
                        if rgb.into_iter().all(f64::is_finite) && lum > 0. {
                            ys.push([lum as f32]);
                            deltas.push([(rgb[0] - lum) as f32, (rgb[2] - lum) as f32]);
                        } else {
                            ys.push([0.]);
                            deltas.push([0.; 2]);
                        }
                    }
                }
                let Some((filtered, weights)) = filter(
                    size,
                    &ys,
                    &deltas,
                    left - x0..right - x0,
                    top - y0..bottom - y0,
                ) else {
                    continue;
                };
                for y in top..bottom {
                    for x in left..right {
                        let i = (y - y0) * size.w + x - x0;
                        let mix = f64::from(weights[i]);
                        if mix <= 0. {
                            continue;
                        }
                        let rgb = linear_rgb(&input.data, y * w + x, transfer);
                        // Preserve signed wide-gamut values and unusual linear inputs.
                        if !rgb.into_iter().all(|v| v.is_finite() && v >= 0.) {
                            continue;
                        }
                        let lum = dot(rgb, luma);
                        let red = (rgb[0] - lum) * (1. - mix) + f64::from(filtered[i][0]) * mix;
                        let blue = (rgb[2] - lum) * (1. - mix) + f64::from(filtered[i][1]) * mix;
                        let green = -(luma[0] * red + luma[2] * blue) / luma[1];
                        // Compress chroma to the lower RGB boundary without clipping luminance.
                        let bound = (lum / (-red.min(green).min(blue)).max(1e-10)).min(1.);
                        let start = 3 * ((y - top) * w + x);
                        for (out, delta) in
                            output[start..start + 3].iter_mut().zip([red, green, blue])
                        {
                            *out = from_f64(transfer.encode(lum + delta * bound));
                        }
                    }
                }
            }
        });
    output
}

/// Blend the full-strength endpoint with its source in linear light.
pub fn mix(input: &ImageBuf, mut full: ImageBuf, transfer: Transfer, amount: f32) -> ImageBuf {
    if amount >= 1. {
        return full;
    }
    if amount <= 0. {
        return input.clone();
    }
    let amount = f64::from(amount);
    full.data
        .par_iter_mut()
        .zip(&input.data)
        .for_each(|(clean, &source)| {
            if *clean != source && source.is_finite() {
                let a = transfer.decode(to_f64(source));
                let b = transfer.decode(to_f64(*clean));
                *clean = from_f64(transfer.encode(a + amount * (b - a)));
            }
        });
    full
}

fn linear_rgb(data: &[Scalar], i: usize, transfer: Transfer) -> [f64; 3] {
    std::array::from_fn(|c| transfer.decode(to_f64(data[3 * i + c])))
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[derive(Clone, Copy)]
struct Grid {
    w: usize,
    h: usize,
}

impl Grid {
    fn len(self) -> usize {
        self.w * self.h
    }
}

fn reflect(x: isize, len: usize) -> usize {
    if (0..len as isize).contains(&x) {
        return x as usize;
    }
    let period = (2 * len) as isize;
    let x = x.rem_euclid(period);
    x.min(period - 1 - x) as usize
}

fn smooth(value: f32, low: f32, high: f32) -> f32 {
    let t = ((value - low) / (high - low)).clamp(0., 1.);
    t * t * (3. - 2. * t)
}

fn kernel(sigma: f32, radius: isize) -> Vec<f32> {
    let mut kernel: Vec<_> = (-radius..=radius)
        .map(|i| (-0.5 * (i as f32 / sigma).powi(2)).exp())
        .collect();
    let sum: f32 = kernel.iter().sum();
    kernel.iter_mut().for_each(|v| *v /= sum);
    kernel
}

// Finite Gaussian support lets tile interiors agree across tile boundaries.
fn gaussian<const N: usize>(size: Grid, input: &[[f32; N]], sigma: f32) -> Vec<[f32; N]> {
    let radius = (4. * sigma).round() as isize;
    let weights = kernel(sigma, radius);
    let mut horizontal = vec![[0.; N]; size.len()];
    let mut padded = vec![[0.; N]; size.w + 2 * radius as usize];
    let indices: Vec<_> = (-radius..size.w as isize + radius)
        .map(|x| reflect(x, size.w))
        .collect();
    for (out, row) in horizontal
        .chunks_exact_mut(size.w)
        .zip(input.chunks_exact(size.w))
    {
        for (dst, &x) in padded.iter_mut().zip(&indices) {
            *dst = row[x];
        }
        for (offset, &weight) in weights.iter().enumerate() {
            accumulate(out, &padded[offset..offset + size.w], weight);
        }
    }

    let mut output = vec![[0.; N]; size.len()];
    for (y, row) in output.chunks_exact_mut(size.w).enumerate() {
        for (offset, &weight) in (-radius..=radius).zip(&weights) {
            let start = reflect(y as isize + offset, size.h) * size.w;
            accumulate(row, &horizontal[start..start + size.w], weight);
        }
    }
    output
}

fn accumulate<const N: usize>(out: &mut [[f32; N]], row: &[[f32; N]], weight: f32) {
    for (out, pixel) in out.iter_mut().zip(row) {
        for c in 0..N {
            out[c] += weight * pixel[c];
        }
    }
}

fn neighbor_indices(len: usize, radius: isize, step: usize) -> Vec<Vec<usize>> {
    (0..len)
        .map(|i| {
            (-radius..=radius)
                .step_by(step)
                .map(|d| reflect(i as isize + d, len))
                .collect()
        })
        .collect()
}

fn axis_position(size: Grid, i: usize, horizontal: bool) -> (usize, usize, usize) {
    let x = i % size.w;
    let y = i / size.w;
    if horizontal {
        (x, y * size.w, 1)
    } else {
        (y, x, size.w)
    }
}

fn median_noise(size: Grid, input: &[[f32; 1]]) -> Vec<[f32; 1]> {
    let pass = |input: &[[f32; 1]], horizontal: bool| {
        let len = if horizontal { size.w } else { size.h };
        let indices = neighbor_indices(len, 5, 1);
        (0..size.len())
            .map(|i| {
                let (line, offset, stride) = axis_position(size, i, horizontal);
                let mut window: [f32; 11] =
                    std::array::from_fn(|j| input[offset + indices[line][j] * stride][0]);
                let (_, median, _) = window.select_nth_unstable_by(5, f32::total_cmp);
                [*median]
            })
            .collect::<Vec<_>>()
    };
    pass(&pass(input, true), false)
}

fn guide(ys: &[[f32; 1]], color: &[Chroma], luma_range: f32, color_range: f32) -> Vec<[f32; 3]> {
    ys.iter()
        .zip(color)
        .map(|(&[y], &[r, b])| {
            [
                y.max(0.002).log2() / luma_range,
                r / color_range,
                b / color_range,
            ]
        })
        .collect()
}

fn edge_filter(
    size: Grid,
    input: &[Chroma],
    guide: &[[f32; 3]],
    sigma: f32,
    step: usize,
) -> Vec<Chroma> {
    let radius = ((2.6 * sigma / step as f32).ceil() as usize * step) as isize;
    let spatial: Vec<_> = (-radius..=radius)
        .step_by(step)
        .map(|i| (-0.5 * (i as f32 / sigma).powi(2)).exp())
        .collect();
    let pass = |input: &[Chroma], horizontal: bool| {
        let len = if horizontal { size.w } else { size.h };
        let indices = neighbor_indices(len, radius, step);
        (0..size.len())
            .map(|i| {
                let (line, offset, stride) = axis_position(size, i, horizontal);
                let [y, r, b] = guide[i];
                let mut total = 0.;
                let mut sum = [0.; 2];
                for (&index, &spatial) in indices[line].iter().zip(&spatial) {
                    let j = offset + index * stride;
                    let [ny, nr, nb] = guide[j];
                    let distance = (ny - y).powi(2) + (nr - r).powi(2) + (nb - b).powi(2);
                    let weight = spatial * (-0.5 * distance).exp();
                    total += weight;
                    sum[0] += weight * input[j][0];
                    sum[1] += weight * input[j][1];
                }
                sum.map(|v| v / total)
            })
            .collect::<Vec<_>>()
    };
    pass(&pass(input, true), false)
}

fn fractional_shift(size: Grid, data: &[Chroma], i: usize, dx: f32, dy: f32) -> Chroma {
    let x = dx.floor() as isize;
    let y = dy.floor() as isize;
    let tx = dx - x as f32;
    let ty = dy - y as f32;
    let x = (i % size.w) as isize + x;
    let y = (i / size.w) as isize + y;
    let left = reflect(x, size.w);
    let right = reflect(x + 1, size.w);
    let top = reflect(y, size.h) * size.w;
    let bottom = reflect(y + 1, size.h) * size.w;
    let a = data[top + left];
    let b = data[top + right];
    let c = data[bottom + left];
    let d = data[bottom + right];
    std::array::from_fn(|k| {
        (a[k] * (1. - tx) + b[k] * tx) * (1. - ty) + (c[k] * (1. - tx) + d[k] * tx) * ty
    })
}

fn protect_edges(size: Grid, chroma: &[Chroma], noise: &[[f32; 1]]) -> (Vec<f32>, Vec<Chroma>) {
    let mut confidence = vec![0.; size.len()];
    let mut tangent = vec![[0.; 2]; size.len()];
    // Pad once so directional convolution can use linear offsets.
    const PAD: isize = 14;
    let stride = size.w + 2 * PAD as usize;
    let padded: Vec<_> = (-PAD..size.h as isize + PAD)
        .flat_map(|y| {
            let row = reflect(y, size.h) * size.w;
            (-PAD..size.w as isize + PAD).map(move |x| chroma[row + reflect(x, size.w)])
        })
        .collect();
    for (dy, dx) in [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
        (1, 2),
        (1, -2),
        (2, 1),
        (2, -1),
    ] {
        let length = (dy as f32).hypot(dx as f32);
        let radius = (2.5 * 5.2 / length).ceil() as isize;
        let weights = kernel(5.2 / length, radius);
        let step = dy * stride as isize + dx;
        let taps: Vec<_> = (-radius..=radius)
            .zip(weights)
            .map(|(k, weight)| (k * step, weight))
            .collect();
        let detail: Vec<Chroma> = (0..size.h)
            .flat_map(|y| {
                let start = ((y + PAD as usize) * stride) as isize + PAD;
                let taps = &taps;
                let padded = &padded;
                (start..start + size.w as isize).map(move |center| {
                    let mut sum = [0.; 2];
                    for &(offset, weight) in taps {
                        let [r, b] = padded[(center + offset) as usize];
                        sum[0] += r * weight;
                        sum[1] += b * weight;
                    }
                    sum
                })
            })
            .collect();
        let nx = -dy as f32 / length;
        let ny = dx as f32 / length;
        let edge: Vec<Chroma> = (0..size.len())
            .map(|i| {
                let before = fractional_shift(size, &detail, i, nx, ny);
                let after = fractional_shift(size, &detail, i, -nx, -ny);
                std::array::from_fn(|c| 0.5 * (after[c] - before[c]))
            })
            .collect();
        let broad = gaussian(size, &edge, 3.);
        let score: Vec<_> = edge
            .into_iter()
            .zip(broad)
            .zip(noise)
            .map(|(([r, b], [br, bb]), &[noise])| {
                let signal = ((r - br).powi(2) + (b - bb).powi(2)).sqrt();
                smooth(signal / (0.02 + noise), 1.3, 2.15)
            })
            .collect();
        if score.iter().all(|&v| v == 0.) {
            continue;
        }
        let span = (8. / length).ceil() as isize;
        let mut offsets = Vec::new();
        for k in -span..=span {
            for y in -1..=1 {
                for x in -1..=1 {
                    offsets.push((-k * dy + x, k * dx + y));
                }
            }
        }
        offsets.sort_unstable();
        offsets.dedup();
        for (i, best) in dilate_edges(size, &score, &offsets).into_iter().enumerate() {
            if best > confidence[i] {
                confidence[i] = best;
                let before = fractional_shift(size, &detail, i, nx, ny);
                let after = fractional_shift(size, &detail, i, -nx, -ny);
                tangent[i] =
                    std::array::from_fn(|c| 0.82 * detail[i][c] + 0.09 * (before[c] + after[c]));
            }
        }
    }
    (confidence, tangent)
}

// Scatter only nonzero scores. Visiting the reflected border as well makes
// this equivalent to gathering the maximum over the same footprint.
fn dilate_edges(size: Grid, score: &[f32], offsets: &[(isize, isize)]) -> Vec<f32> {
    let width = size.w as isize;
    let height = size.h as isize;
    let margin_x = offsets.iter().map(|&(x, _)| x.abs()).max().unwrap();
    let margin_y = offsets.iter().map(|&(_, y)| y.abs()).max().unwrap();
    let mut output = vec![0.0f32; size.len()];
    for y in -margin_y..height + margin_y {
        let row = reflect(y, size.h) * size.w;
        for x in -margin_x..width + margin_x {
            let value = score[row + reflect(x, size.w)];
            if value == 0. {
                continue;
            }
            for &(dx, dy) in offsets {
                let target_y = y - dy;
                let target_x = x - dx;
                if (0..height).contains(&target_y) && (0..width).contains(&target_x) {
                    let i = target_y as usize * size.w + target_x as usize;
                    output[i] = output[i].max(value);
                }
            }
        }
    }
    output
}

fn filter(
    size: Grid,
    ys: &[[f32; 1]],
    deltas: &[Chroma],
    x: Range<usize>,
    y: Range<usize>,
) -> Option<(Vec<Chroma>, Vec<f32>)> {
    let local_y = gaussian(size, ys, 0.95);
    let chroma: Vec<Chroma> = deltas
        .iter()
        .zip(&local_y)
        .map(|(&delta, &[y])| delta.map(|v| v / y.max(0.002)))
        .collect();
    let fine_noise = gaussian(size, &chroma, 0.9);
    let residual: Vec<_> = chroma
        .iter()
        .zip(fine_noise)
        .map(|(&[r, b], [sr, sb])| [((r - sr).powi(2) * 0.5 + (b - sb).powi(2) * 0.5).sqrt()])
        .collect();
    let mut noise = gaussian(size, &median_noise(size, &residual), 4.);
    noise.iter_mut().for_each(|v| v[0] *= 1.2);
    let strength: Vec<_> = noise.iter().map(|&[n]| smooth(n, 0.030, 0.075)).collect();
    // Noise in the halo alone cannot change any output pixel.
    if y.map(|y| &strength[y * size.w + x.start..y * size.w + x.end])
        .flatten()
        .all(|&v| v == 0.)
    {
        return None;
    }
    let fine = edge_filter(
        size,
        &chroma,
        &guide(&local_y, &gaussian(size, &chroma, 1.35), 0.34, 0.20),
        2.5,
        1,
    );
    let mid = edge_filter(
        size,
        &fine,
        &guide(
            &gaussian(size, ys, 2.2),
            &gaussian(size, &fine, 1.5),
            0.40,
            0.24,
        ),
        11.,
        3,
    );
    let (confidence, tangent) = protect_edges(size, &chroma, &noise);
    let mut guide_chroma = gaussian(size, &fine, 4.);
    for (i, color) in guide_chroma.iter_mut().enumerate() {
        for c in 0..2 {
            color[c] += confidence[i] * (tangent[i][c] - color[c]);
        }
    }
    let mut wide = edge_filter(
        size,
        &mid,
        &guide(&gaussian(size, ys, 3.), &guide_chroma, 0.72, 0.53),
        32.,
        7,
    );
    for (i, color) in wide.iter_mut().enumerate() {
        for c in 0..2 {
            let mixed = mid[i][c] + 0.96 * (color[c] - mid[i][c]);
            color[c] = (mixed + confidence[i] * (tangent[i][c] - mixed)) * local_y[i][0].max(0.002);
        }
    }
    Some((wide, strength))
}

#[cfg(test)]
mod tests {
    use super::*;

    const LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];

    #[test]
    fn output_agrees_across_tile_boundaries() {
        let w = PREVIEW_TILE_WIDTH + 16;
        let h = PREVIEW_TILE_HEIGHT + 16;
        let mut seed = 217u32;
        let data = (0..w * h * 3)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                from_f64(0.1 + (f64::from(seed >> 8) / 16777216. - 0.5) * 0.06)
            })
            .collect();
        let input = ImageBuf::from_data(w as u32, h as u32, data);
        let tiled = filter_full(&input, LUMA, Transfer::Linear);

        let (x0, y0) = (
            PREVIEW_TILE_WIDTH - HALO - 8,
            PREVIEW_TILE_HEIGHT - HALO - 8,
        );
        let corner = ImageBuf::from_data(
            (w - x0) as u32,
            (h - y0) as u32,
            input
                .data
                .chunks_exact(w * 3)
                .skip(y0)
                .flat_map(|row| &row[x0 * 3..])
                .copied()
                .collect(),
        );
        let single = filter_full(&corner, LUMA, Transfer::Linear);
        assert_ne!(single.data, corner.data);
        for y in PREVIEW_TILE_HEIGHT - 4..h {
            let tiled_start = (y * w + PREVIEW_TILE_WIDTH - 4) * 3;
            let single_start = ((y - y0) * (w - x0) + PREVIEW_TILE_WIDTH - 4 - x0) * 3;
            let len = (w - PREVIEW_TILE_WIDTH + 4) * 3;
            assert_eq!(
                tiled.data[tiled_start..tiled_start + len],
                single.data[single_start..single_start + len]
            );
        }
    }

    #[test]
    fn retains_luminance_and_thin_diagonal_color_detail() {
        let base = [0.13, 0.10, 0.075];
        let tint = [0.165, 0.083, 0.138];
        let gain = dot(base, LUMA) / dot(tint, LUMA);
        let mut image = ImageBuf::new(96, 96);
        let mut seed = 217u32;
        for (i, pixel) in image.data.chunks_exact_mut(3).enumerate() {
            let stripe = (0..4).contains(&(i as i32 % 96 - i as i32 / 96));
            for (c, v) in pixel.iter_mut().enumerate() {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let noise = (f64::from(seed >> 8) / 16777216. - 0.5) * 0.04;
                let value = if stripe { tint[c] * gain } else { base[c] };
                *v = from_f64(value + noise);
            }
        }
        let filtered = apply(&image, LUMA, Transfer::Linear, 1.);
        let mut blue = [0.; 2];
        let mut count = [0.; 2];
        for (i, pixel) in filtered.data.chunks_exact(3).enumerate() {
            let old = linear_rgb(&image.data, i, Transfer::Linear);
            let new = linear_rgb(&filtered.data, i, Transfer::Linear);
            assert!((dot(old, LUMA) - dot(new, LUMA)).abs() < 1e-7);
            let x = (i % 96) as i32;
            let y = (i / 96) as i32;
            if (20..76).contains(&x) && (20..76).contains(&y) {
                let offset = x - y;
                for (group, range) in [(0, -10..-5), (1, 0..4)] {
                    if range.contains(&offset) {
                        blue[group] += to_f64(pixel[2]) - dot(new, LUMA);
                        count[group] += 1.;
                    }
                }
            }
        }
        assert_ne!(filtered.data, image.data);
        assert!(blue[1] / count[1] - blue[0] / count[0] > 0.7 * (tint[2] * gain - base[2]));
        assert_eq!(apply(&image, LUMA, Transfer::Linear, 0.).data, image.data);
    }
}
