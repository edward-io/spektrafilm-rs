//! Recover saturated Bayer sites before rawler interpolates their color.
use anyhow::{Result, bail};
use rawler::{
    RawImage,
    imgop::sensor::SensorType,
    rawimage::{RawImageData, RawPhotometricInterpretation},
};
use rayon::prelude::*;
use spektrafilm_math::{
    gaussian::gaussian_blur,
    image::ImageBuf,
    precision::{from_f32, to_f32},
};

const RECOVERY_START: f32 = 0.94;
const GUIDE_MARGIN: usize = 96;

/// Returns true when the mosaic has been normalized and can skip rawler's rescale.
pub(super) fn reconstruct(raw: &mut RawImage, wb: [f32; 4]) -> Result<bool> {
    let RawPhotometricInterpretation::Cfa(config) = &raw.photometric else {
        return Ok(false);
    };
    if raw.cpp != 1
        || config.sensor != SensorType::Bayer
        || !config.cfa.is_rgb()
        || raw.width < 2
        || raw.height < 2
        || !wb[..3].iter().all(|gain| gain.is_finite() && *gain > 0.)
    {
        return Ok(false);
    }
    let channels = std::array::from_fn(|i| config.cfa.color_at(i / 2, i % 2));
    raw.apply_scaling()
        .map_err(|e| anyhow::anyhow!("RAW rescale: {e:?}"))?;
    let RawImageData::Float(samples) = &mut raw.data else {
        bail!("RAW rescale did not produce floating-point samples");
    };
    reconstruct_mosaic(samples, raw.width, channels, wb);
    Ok(true)
}

fn reconstruct_mosaic(samples: &mut [f32], width: usize, channels: [usize; 4], wb: [f32; 4]) {
    let cell_width = width / 2;
    let cell_height = samples.len() / (width * 2);
    let clipped = samples
        .par_chunks_exact(width * 2)
        .enumerate()
        .filter_map(|(y, rows)| {
            let mut xs = (0..cell_width).filter(|&x| {
                read_cell(rows, width, x)
                    .into_iter()
                    .any(|v| v > RECOVERY_START)
            });
            let first = xs.next()?;
            Some([first, y, xs.next_back().unwrap_or(first), y])
        })
        .reduce_with(|a, b| {
            [
                a[0].min(b[0]),
                a[1].min(b[1]),
                a[2].max(b[2]),
                a[3].max(b[3]),
            ]
        });
    let Some([left, top, right, bottom]) = clipped else {
        return;
    };
    let x0 = left.saturating_sub(GUIDE_MARGIN);
    let y0 = top.saturating_sub(GUIDE_MARGIN);
    let w = (right + GUIDE_MARGIN + 1).min(cell_width) - x0;
    let h = (bottom + GUIDE_MARGIN + 1).min(cell_height) - y0;

    let mut guides = vec![[0.; 3]; w * h];
    guides
        .par_chunks_exact_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            let rows = &samples[(y0 + y) * width * 2..][..width * 2];
            for (x, guide) in row.iter_mut().enumerate() {
                let cell = read_cell(rows, width, x0 + x);
                let peak = cell.into_iter().fold(0., f32::max);
                if !(0.35 < peak && peak < 0.93) {
                    continue;
                }
                let camera = camera_rgb(cell, channels);
                if camera.into_iter().all(|v| v > 0.035) {
                    let max = camera.into_iter().fold(0., f32::max);
                    *guide = camera.map(|v| v / max);
                }
            }
        });
    let ratios = nearest_ratios(&guides, w, h);
    drop(guides);
    let ratios = ratios.map(|field| gaussian_blur(&field, 4.));
    let gains = channels.map(|c| wb[c]);
    samples
        .par_chunks_exact_mut(width * 2)
        .enumerate()
        .skip(top)
        .take(bottom - top + 1)
        .for_each(|(y, rows)| {
            for x in left..=right {
                let cell = read_cell(rows, width, x);
                if !cell.into_iter().any(|v| v > RECOVERY_START) {
                    continue;
                }
                let ratio = ratios.as_ref().map(|field| {
                    field
                        .get((x - x0) as u32, (y - y0) as u32)
                        .map(|v| to_f32(v).max(0.05))
                });
                let [a, b, c, d] = recover_cell(cell, channels, gains, ratio);
                let x = 2 * x;
                rows[x] = a;
                rows[x + 1] = b;
                rows[width + x] = c;
                rows[width + x + 1] = d;
            }
        });
}

fn read_cell(rows: &[f32], width: usize, x: usize) -> [f32; 4] {
    let x = 2 * x;
    [rows[x], rows[x + 1], rows[width + x], rows[width + x + 1]]
}

fn camera_rgb(cell: [f32; 4], channels: [usize; 4]) -> [f32; 3] {
    let mut rgb = [0.; 3];
    for (sample, channel) in cell.into_iter().zip(channels) {
        rgb[channel] += sample;
    }
    rgb[1] *= 0.5;
    rgb
}

fn smoothstep(value: f32, start: f32) -> f32 {
    let t = ((value - start) / (1. - start)).clamp(0., 1.);
    t * t * (3. - 2. * t)
}

fn recover_cell(
    cell: [f32; 4],
    channels: [usize; 4],
    gains: [f32; 4],
    ratio: Option<[f32; 3]>,
) -> [f32; 4] {
    if !cell.into_iter().all(f32::is_finite) {
        return cell;
    }
    let estimate = ratio.map_or([0.; 3], |ratio| {
        let rgb = camera_rgb(cell, channels);
        let mut sum = 0.;
        let mut total_weight = 0.;
        for (sample, ratio) in rgb.into_iter().zip(ratio) {
            let weight = (1. - smoothstep(sample, 0.9)) * sample.clamp(0., 1.).powi(2);
            sum += weight * sample / ratio;
            total_weight += weight;
        }
        ratio.map(|r| (sum / total_weight.max(1e-9) * r).min(4.))
    });
    // With no surviving channel, use the strongest measured signal as the
    // brightness of a neutral highlight in white-balanced space.
    let neutral = cell
        .into_iter()
        .zip(gains)
        .map(|(v, g)| v * g)
        .fold(0., f32::max);
    let lost = smoothstep(cell.into_iter().fold(f32::INFINITY, f32::min), 0.92);
    std::array::from_fn(|i| {
        let sample = cell[i];
        let recovered =
            sample + smoothstep(sample, RECOVERY_START) * (estimate[channels[i]] - sample).max(0.);
        recovered + lost * (neutral / gains[i] - recovered)
    })
}

// First find the closest reference vertically in each column. Across a row,
// squared distances to those candidates form parabolas with equal curvature.
// Their lower envelope gives the nearest reference at every x in linear time.
fn nearest_ratios(guides: &[[f32; 3]], w: usize, h: usize) -> Option<ImageBuf> {
    const MISSING: u32 = u32::MAX;
    if guides.iter().all(|guide| guide[0] == 0.) {
        return None;
    }
    let mut vertical = vec![MISSING; w * h];
    let mut closest = vec![MISSING; w];
    for (y, row) in vertical.chunks_exact_mut(w).enumerate() {
        for (x, nearest) in row.iter_mut().enumerate() {
            if guides[y * w + x][0] > 0. {
                closest[x] = y as u32;
            }
            *nearest = closest[x];
        }
    }
    closest.fill(MISSING);
    for (y, row) in vertical.chunks_exact_mut(w).enumerate().rev() {
        for (x, nearest) in row.iter_mut().enumerate() {
            if guides[y * w + x][0] > 0. {
                closest[x] = y as u32;
            }
            if closest[x] != MISSING
                && (*nearest == MISSING || closest[x] as usize - y < y - *nearest as usize)
            {
                *nearest = closest[x];
            }
        }
    }
    let mut field = ImageBuf::new(w as u32, h as u32);
    field
        .data
        .par_chunks_exact_mut(w * 3)
        .zip(vertical.par_chunks_exact(w))
        .enumerate()
        .for_each_init(Vec::<(i64, i64)>::new, |envelope, (y, (out, columns))| {
            envelope.clear();
            let cost = |x: i64| (i64::from(columns[x as usize]) - y as i64).pow(2) + x * x;
            for (x, _) in columns
                .iter()
                .enumerate()
                .filter(|&(_, &row)| row != MISSING)
            {
                let x = x as i64;
                let mut start = 0;
                while let Some(&(previous, previous_start)) = envelope.last() {
                    // First integer coordinate where the new candidate is closer.
                    start = (cost(x) - cost(previous)).div_euclid(2 * (x - previous)) + 1;
                    if start > previous_start {
                        break;
                    }
                    envelope.pop();
                    start = 0;
                }
                envelope.push((x, start));
            }
            let mut site = 0;
            for (x, pixel) in out.chunks_exact_mut(3).enumerate() {
                while envelope
                    .get(site + 1)
                    .is_some_and(|&(_, start)| start <= x as i64)
                {
                    site += 1;
                }
                let col = envelope[site].0 as usize;
                let guide = guides[columns[col] as usize * w + col];
                pixel.copy_from_slice(&guide.map(from_f32));
            }
        });
    Some(field)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bayer_recovery_uses_local_color_preserves_valid_sites_and_neutralizes_a_clipped_core() {
        let original = [
            [0.4, 0.6, 0.8, 1., 1.05, 1.05, 0.9],
            [0.6, 0.2, 1., 0.4, 1.05, 1.05, 1.08],
            [1.05; 7],
        ];
        let mut samples = original.as_flattened().to_vec();
        reconstruct_mosaic(&mut samples, 7, [0, 1, 1, 2], [2., 1., 1.5, f32::NAN]);
        let expected = [
            [0.4, 0.6, 0.8, 1.2, 1.05, 2.1, 0.9],
            [0.6, 0.2, 1.2, 0.4, 2.1, 1.4, 1.08],
            [1.05; 7],
        ];
        for (&actual, &expected) in samples.iter().zip(expected.as_flattened()) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn nearest_reference_matches_squared_distance_and_handles_no_guides() {
        let (w, h) = (9, 8);
        let positions = [(0usize, 6usize), (2, 2), (2, 7), (5, 3), (8, 0), (8, 6)];
        let mut guides = vec![[0.; 3]; w * h];
        assert!(nearest_ratios(&guides, w, h).is_none());
        for (i, &(x, y)) in positions.iter().enumerate() {
            guides[y * w + x] = [(i + 1) as f32; 3];
        }
        let actual = nearest_ratios(&guides, w, h).unwrap();
        for y in 0..h {
            for x in 0..w {
                let distance =
                    |(gx, gy): (usize, usize)| x.abs_diff(gx).pow(2) + y.abs_diff(gy).pow(2);
                let i = actual.get(x as u32, y as u32)[0] as usize - 1;
                assert_eq!(
                    distance(positions[i]),
                    positions.into_iter().map(distance).min().unwrap()
                );
            }
        }
    }
}
