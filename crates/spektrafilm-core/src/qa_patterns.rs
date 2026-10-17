//! Reference QA stimulus patterns, independent of rendering / plotting.
use crate::{lut_export::LutSpace, qa_metrics};
use rayon::prelude::*;
use spektrafilm_math::transfer;
/// CIE D daylight locus above 4000 K; Kang 2002 Planckian fit below it.
pub fn white_xy(t: f64) -> Result<[f64; 2], String> {
    if !t.is_finite() || !(1667.0..=25000.0).contains(&t) {
        return Err("QA CCT must be in 1667–25000 K".into());
    }
    let (x, y) = if t >= 4000. {
        let x = if t <= 7000. {
            -4.607e9 / t.powi(3) + 2.9678e6 / t.powi(2) + 99.11 / t + 0.244063
        } else {
            -2.0064e9 / t.powi(3) + 1.9018e6 / t.powi(2) + 247.48 / t + 0.23704
        };
        (x, -3. * x * x + 2.87 * x - 0.275)
    } else {
        let x = -0.2661239e9 / t.powi(3) - 0.2343589e6 / t.powi(2) + 877.6956 / t + 0.179910;
        let y = if t <= 2222. {
            -1.1063814 * x.powi(3) - 1.34811020 * x * x + 2.18555832 * x - 0.20219683
        } else {
            -0.9549476 * x.powi(3) - 1.37418593 * x * x + 2.09137015 * x - 0.16748867
        };
        (x, y)
    };
    Ok([x, y])
}
pub fn planckian_sweep(
    input: &LutSpace,
    n: usize,
    gain: Option<f64>,
) -> Result<(Vec<[f64; 3]>, Vec<f64>), String> {
    if n < 2 {
        return Err("CCT sweep needs at least two samples".into());
    }
    let matrix = transfer::get(input.primaries)?.from_xyz;
    let temperatures: Vec<_> = (0..n)
        .map(|i| 2700. + 7300. * i as f64 / (n - 1) as f64)
        .collect();
    let mut samples = Vec::new();
    for &t in &temperatures {
        let [x, y] = white_xy(t)?;
        let rgb = qa_metrics::mul(matrix, [x / y, 1., (1. - x - y) / y]);
        let peak = rgb.into_iter().fold(1e-6, f64::max);
        let linear = rgb.map(|v| (v / peak).clamp(0., 1.) / gain.unwrap_or(input.auto_gain));
        samples.push(input.encode_pixel(linear).map(|v| (v as f32) as f64));
    }
    Ok((samples, temperatures))
}
pub fn max_bend_angle(points: &[[f64; 2]]) -> f64 {
    let differences: Vec<_> = points
        .windows(2)
        .map(|p| [p[1][0] - p[0][0], p[1][1] - p[0][1]])
        .collect();
    differences
        .windows(2)
        .map(|p| {
            let a = p[0][0].hypot(p[0][1]) + 1e-12;
            let b = p[1][0].hypot(p[1][1]) + 1e-12;
            ((p[0][0] * p[1][0] + p[0][1] * p[1][1]) / (a * b))
                .clamp(-1., 1.)
                .acos()
                .to_degrees()
        })
        .fold(0., f64::max)
}

pub struct CharacteristicProbe {
    pub axis: Vec<f64>,
    pub pin_densities: [f64; 5],
    pub pin_codes: [f64; 5],
    /// Swept input channel → pinned density → encoded RGB outputs.
    pub sweeps: [Vec<Vec<[f64; 3]>>; 3],
    pub neutral: Vec<[f64; 3]>,
    pub system_gamma_at_mid: f64,
    pub max_channel_density_spread: f64,
}
pub fn characteristic_probe(
    table: &spektrafilm_math::image::ImageBuf,
    n: usize,
) -> Result<CharacteristicProbe, String> {
    if n < 2 || table.data.len() != n * n * n * 3 {
        return Err("invalid diagnostic cube shape".into());
    }
    let evaluate = |p| crate::lut_quality::trilinear(table, n, p);
    let density = |rgb: [f64; 3]| -((rgb.iter().sum::<f64>() / 3.).clamp(1e-4, 1.)).log10();
    let probe: Vec<_> = (0..257).map(|i| i as f64 / 256.).collect();
    let d: Vec<_> = probe
        .iter()
        .rev()
        .map(|&v| density(evaluate([v; 3])))
        .collect();
    let codes: Vec<_> = probe.iter().rev().copied().collect();
    let pin_densities = [0.2, 0.4, 0.6, 0.8, 1.];
    let pin_codes = pin_densities.map(|target| {
        if target < d[0] {
            return codes[0];
        }
        if target >= d[d.len() - 1] {
            return codes[codes.len() - 1];
        }
        let i = d.partition_point(|v| *v <= target).clamp(1, d.len() - 1);
        let f = (target - d[i - 1]) / (d[i] - d[i - 1]);
        codes[i - 1] + f * (codes[i] - codes[i - 1])
    });
    let axis: Vec<_> = (0..65).map(|i| i as f64 / 64.).collect();
    let sweeps = std::array::from_fn(|c| {
        pin_codes
            .iter()
            .map(|&pin| {
                axis.iter()
                    .map(|&x| {
                        let mut rgb = [pin; 3];
                        rgb[c] = x;
                        evaluate(rgb.map(|v| v as f32 as f64))
                    })
                    .collect()
            })
            .collect()
    });
    let neutral = axis.iter().map(|&v| evaluate([v; 3])).collect();
    let grid_neutral: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let base = ((i * n + i) * n + i) * 3;
            std::array::from_fn(|c| table.data[base + c] as f64)
        })
        .collect();
    let (lo, hi) = if n >= 3 {
        (n / 2 - 1, n / 2 + 1)
    } else {
        (0, 1)
    };
    let log_in = |i: usize| (1e-6 + (1. - 1e-6) * i as f64 / (n - 1) as f64).log10();
    let gamma =
        (-density(grid_neutral[hi]) + density(grid_neutral[lo])) / (log_in(hi) - log_in(lo));
    let spread = grid_neutral
        .iter()
        .map(|p| {
            let d = p.map(|v| -v.clamp(1e-4, 1.).log10());
            d.into_iter().fold(f64::NEG_INFINITY, f64::max)
                - d.into_iter().fold(f64::INFINITY, f64::min)
        })
        .fold(0., f64::max);
    Ok(CharacteristicProbe {
        axis,
        pin_densities,
        pin_codes,
        sweeps,
        neutral,
        system_gamma_at_mid: gamma,
        max_channel_density_spread: spread,
    })
}

pub struct DynamicRamp {
    pub stops: Vec<f64>,
    pub encoded: Vec<[f64; 3]>,
    pub clipped: Vec<bool>,
}
pub fn dynamic_range_ramp(input: &LutSpace) -> Result<DynamicRamp, String> {
    // The source probes in 0.01-stop steps instead of assuming a perfect
    // analytical inverse CCTF. Preserve those endpoints and its half-stop pad.
    let ceiling = (0..=3200)
        .rev()
        .map(|i| -12. + 32. * i as f64 / 3200.)
        .find(|stop| {
            input
                .encode_pixel([input.midgray_linear * 2f64.powf(*stop); 3])
                .iter()
                .all(|v| *v <= 1.)
        })
        .ok_or("input encoding clips at every probed exposure")?;
    let stops: Vec<_> = (0..257)
        .map(|i| -8. + (ceiling + 0.5 + 8.) * i as f64 / 256.)
        .collect();
    let raw: Vec<_> = stops
        .iter()
        .map(|s| input.encode_pixel([input.midgray_linear * 2f64.powf(*s); 3]))
        .collect();
    let clipped = raw
        .iter()
        .map(|p| p.iter().any(|v| *v < 0. || *v > 1.))
        .collect();
    let encoded = raw
        .iter()
        .map(|p| p.map(|v| v.clamp(0., 1.) as f32 as f64))
        .collect();
    Ok(DynamicRamp {
        stops,
        encoded,
        clipped,
    })
}

pub const MIDGRAY_OKLAB_L: f64 = 0.5646225971435698;
fn oklab_to_input(lab: [f64; 3], input: &LutSpace) -> [f64; 3] {
    qa_metrics::mul(
        transfer::get(input.primaries).unwrap().from_xyz,
        crate::gamut_compression::oklab_to_xyz(lab),
    )
}
fn in_rgb_gamut(p: [f64; 3]) -> bool {
    p.iter().all(|v| *v >= 0. && *v <= 1.)
}
/// Source 18-iteration constant-lightness gamut boundary, with adaptive ceiling.
pub fn gamut_slice(input: &LutSpace, lightness: f64, n_hues: usize) -> Vec<[f64; 3]> {
    (0..n_hues)
        .map(|i| {
            let angle = 2. * std::f64::consts::PI * i as f64 / n_hues as f64;
            let (sin, cos) = angle.sin_cos();
            let valid = |c: f64| in_rgb_gamut(oklab_to_input([lightness, c * cos, c * sin], input));
            let mut hi = 0.45;
            let mut lo = 0.;
            for _ in 0..3 {
                if valid(hi) {
                    hi *= 1.5;
                }
            }
            for _ in 0..18 {
                let mid = (lo + hi) * 0.5;
                if valid(mid) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            [lightness, lo * cos, lo * sin]
        })
        .collect()
}
pub struct NoiseGradientInput {
    pub encoded: Vec<[f64; 3]>,
    pub in_gamut_fraction: f64,
}
pub fn noise_gradient_input(
    input: &LutSpace,
    gain: Option<f64>,
    width: usize,
    height: usize,
) -> NoiseGradientInput {
    let rows: Vec<_> =
        (0..height)
            .into_par_iter()
            .map(|y| {
                let mut encoded = Vec::with_capacity(width);
                let mut valid = 0usize;
                let v = y as f64 / height.saturating_sub(1).max(1) as f64;
                let lightness = if v <= 0.5 {
                    v / 0.5 * MIDGRAY_OKLAB_L
                } else {
                    MIDGRAY_OKLAB_L + (v - 0.5) / 0.5 * (1. - MIDGRAY_OKLAB_L)
                };
                let saturation = 1. - (2. * v - 1.).abs();
                let outline = gamut_slice(input, lightness, width);
                for (x, lab) in outline.into_iter().enumerate() {
                    let hue = 2. * std::f64::consts::PI * x as f64 / width as f64;
                    let chroma = 0.96 * saturation * lab[1].hypot(lab[2]);
                    let rgb =
                        oklab_to_input([lightness, chroma * hue.cos(), chroma * hue.sin()], input);
                    valid += usize::from(in_rgb_gamut(rgb));
                    encoded.push(input.encode_pixel(
                        rgb.map(|v| v.clamp(0., 1.) / gain.unwrap_or(input.auto_gain)),
                    ));
                }
                (encoded, valid)
            })
            .collect();
    let valid = rows.iter().map(|(_, count)| *count).sum::<usize>();
    let encoded = rows.into_iter().flat_map(|(row, _)| row).collect();
    NoiseGradientInput {
        encoded,
        in_gamut_fraction: valid as f64 / (width * height) as f64,
    }
}

/// Bounded cache for the expensive, stock-independent gamut boundary raster.
pub(crate) fn cached_noise_gradient(
    input: &LutSpace,
    gain: Option<f64>,
) -> std::sync::Arc<NoiseGradientInput> {
    type Key = (String, u64);
    type Cache = std::collections::VecDeque<(Key, std::sync::Arc<NoiseGradientInput>)>;
    static CACHE: std::sync::OnceLock<std::sync::Mutex<Cache>> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key = (
        input.name.to_owned(),
        gain.unwrap_or(input.auto_gain).to_bits(),
    );
    {
        let mut entries = cache.lock().unwrap();
        if let Some(index) = entries.iter().position(|(k, _)| *k == key) {
            let entry = entries.remove(index).unwrap();
            let result = entry.1.clone();
            entries.push_back(entry);
            return result;
        }
    }
    let result = std::sync::Arc::new(noise_gradient_input(input, gain, 512, 512));
    let mut entries = cache.lock().unwrap();
    if entries.len() >= 4 {
        entries.pop_front();
    }
    entries.push_back((key, result.clone()));
    result
}

pub struct NoisePolarSample {
    pub encoded: [f64; 3],
    pub hue_deg: f64,
    pub chroma: f64,
    pub ring: i32,
}
pub fn noise_polar_samples(input: &LutSpace, gain: Option<f64>) -> Vec<NoisePolarSample> {
    let mut requested = vec![([MIDGRAY_OKLAB_L, 0., 0.], 0., 0., -1)];
    for (ring, chroma) in [0.07, 0.14, 0.21].into_iter().enumerate() {
        for i in 0..16 {
            let h = 2. * std::f64::consts::PI * i as f64 / 16.;
            requested.push((
                [MIDGRAY_OKLAB_L, chroma * h.cos(), chroma * h.sin()],
                h.to_degrees(),
                chroma,
                ring as i32,
            ));
        }
    }
    requested
        .into_iter()
        .filter_map(|(lab, hue_deg, chroma, ring)| {
            let linear = oklab_to_input(lab, input);
            if !in_rgb_gamut(linear) {
                return None;
            }
            let encoded = input.encode_pixel(linear.map(|v| v / gain.unwrap_or(input.auto_gain)));
            encoded
                .iter()
                .all(|v| *v > 0.006 && *v < 0.994)
                .then_some(NoisePolarSample {
                    encoded,
                    hue_deg,
                    chroma,
                    ring,
                })
        })
        .collect()
}
pub(crate) fn input_linear_from_oklab(lab: [f64; 3], input: &LutSpace) -> [f64; 3] {
    oklab_to_input(lab, input)
}

pub(crate) fn hue_rgb(hue: f64) -> [f64; 3] {
    let h = hue.rem_euclid(1.) * 6.;
    let f = h - h.floor();
    match h.floor() as usize % 6 {
        0 => [1., f, 0.],
        1 => [1. - f, 1., 0.],
        2 => [0., 1., f],
        3 => [0., 1. - f, 1.],
        4 => [f, 0., 1.],
        _ => [1., 0., 1. - f],
    }
}
