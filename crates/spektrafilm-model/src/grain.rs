// Stochastic grain generation.
// Poisson-binomial particle model with dye cloud blur and lognormal micro-structure.

use rayon::prelude::*;
use spektrafilm_gpu::ComputeBackend;
use spektrafilm_math::gaussian;
use spektrafilm_math::image::ImageBuf;
use spektrafilm_math::precision::{Scalar, ZERO, from_f64};
use std::time::Instant;

fn stage_timings_enabled() -> bool {
    std::env::var_os("SPEKTRAFILM_STAGE_TIMINGS").is_some()
}

fn print_stage_timing(enabled: bool, stage: &str, start: Instant) {
    if enabled {
        eprintln!("stage {stage}: {} ms", start.elapsed().as_millis());
    }
}

/// Apply the Poisson-binomial grain particle model to a single-channel density image.
///
/// Port of Python `layer_particle_model`. Density values flow through in `Scalar`
/// (full f64 precision in `precision-f64` mode); RNG sampling uses f64 directly.
pub fn layer_particle_model(
    density: &[Scalar],
    width: u32,
    height: u32,
    density_max: f64,
    n_particles_per_pixel: f64,
    grain_uniformity: f64,
    seed: u64,
    blur_particle: f32,
) -> Vec<Scalar> {
    // Inputs are f64 to match Python's `layer_particle_model` —
    // density_max/uniformity/agx-area come from JSON profiles at full
    // f64 precision in Python. Passing them as f32 in Rust truncated
    // ~7-decimal noise into `od_particle` and `saturation` which then
    // shifted every Poisson lambda by ~5e-8, producing a different
    // RNG stream and thus visibly different grain patterns.
    let dmax = density_max;
    let od_particle = dmax / n_particles_per_pixel;
    let gu = grain_uniformity;
    let npp = n_particles_per_pixel;

    // Python-bit-exact RNG: numpy's MT19937 + RandomState distribution
    // algorithms (`rk_poisson` / `rk_binomial`). Numpy's seed-to-stream
    // takes a `u32`, so truncate the 64-bit seed — for the CPU export
    // path we never need more than 24 distinct streams (3 channels × N
    // sub-layers). Iteration is single-threaded so the RNG state advances
    // in row-major order exactly like `numpy.random.RandomState(seed)
    // .poisson(lambda_array)` does internally.
    //
    // CRITICAL — match Python's call order: it does
    //   seeds = np.random.poisson(lam_array)   # whole array, then
    //   grain = np.random.binomial(seeds, p)   # whole array.
    // So the RNG must draw ALL Poisson samples first, THEN ALL Binomial
    // samples. Interleaving Poisson/Binomial per pixel (what we used to
    // do) gives a different RNG state at each draw and produces a
    // completely different grain pattern.
    let n = density.len();
    let mut rng = rand_mt::Mt::new(seed as u32);
    let mut p_arr = Vec::with_capacity(n);
    let mut sat_arr = Vec::with_capacity(n);
    for &d in density.iter() {
        let p = ((d as f64) / dmax).clamp(1e-6, 1.0 - 1e-6);
        let saturation = 1.0 - p * gu * (1.0 - 1e-6);
        p_arr.push(p);
        sat_arr.push(saturation);
    }
    let mut seeds = Vec::with_capacity(n);
    for sat in &sat_arr {
        seeds.push(spektrafilm_math::numpy_rng::rk_poisson(&mut rng, npp / sat));
    }
    let mut grain = vec![ZERO; n];
    for (i, slot) in grain.iter_mut().enumerate() {
        let developed = spektrafilm_math::numpy_rng::rk_binomial(&mut rng, seeds[i], p_arr[i]);
        *slot = from_f64((developed as f64) * od_particle * sat_arr[i]);
    }

    if blur_particle > 0.0 {
        // Match Python: `sigma = blur_particle * np.sqrt(od_particle)`
        // — keep the multiplication in f64 and narrow at the end.
        let sigma = (blur_particle as f64 * od_particle.sqrt()) as f32;
        if sigma > 0.0 {
            let mut img = ImageBuf::from_data(
                width,
                height,
                grain.iter().flat_map(|&v| [v, v, v]).collect(),
            );
            img = gaussian::gaussian_blur(&img, sigma);
            grain = img.extract_channel(0);
        }
    }

    grain
}

/// Three physical emulsion layers per dye channel (Python's default grain model).
pub struct LayeredGrainParams {
    pub rms_granularity: [f64; 3],
    pub usm_sigma: f32,
    pub usm_amount: f32,
    pub layer_scale: [f64; 3],
    pub density_min: [f64; 3],
    pub uniformity: [f64; 3],
    pub blur: f32,
    pub dye_cloud_blur: f32,
    pub micro_structure: [f32; 2],
    pub monochrome: bool,
    pub fast_stats: bool,
}

/// Size particles from the realized variance peak of the complete physical layers.
pub fn coarsest_particle_area(layers: &[Vec<Vec<f64>>], p: &LayeredGrainParams) -> [f64; 3] {
    std::array::from_fn(|c| {
        let maxima: [f64; 3] =
            std::array::from_fn(|sl| layers.iter().map(|r| r[sl][c]).fold(0., f64::max));
        let total: f64 = maxima.iter().sum();
        let fractions = maxima.map(|v| v / total.max(1e-30));
        let peak = layers
            .iter()
            .map(|row| {
                (0..3)
                    .filter(|sl| fractions[*sl] > 0.)
                    .map(|sl| {
                        let floor = fractions[sl] * p.density_min[c];
                        let d = row[sl][c] + floor;
                        p.layer_scale[sl] / fractions[sl]
                            * d
                            * (maxima[sl] + floor - p.uniformity[c] * d)
                    })
                    .sum::<f64>()
            })
            .fold(0., f64::max);
        (p.rms_granularity[c] / 1000.).powi(2) * std::f64::consts::PI * 24f64.powi(2)
            / peak.max(1e-9)
    })
}

pub fn multiplicative_unsharp_mask(image: &ImageBuf, sigma: f32, amount: f32) -> ImageBuf {
    if sigma <= 0. || amount <= 0. {
        return image.clone();
    }
    let mut input = image.clone();
    for d in &mut input.data {
        *d = d.max(ZERO);
    }
    let blurred = gaussian::gaussian_blur(&input, sigma);
    let mut out = input.clone();
    for ((v, d), b) in out.data.iter_mut().zip(&input.data).zip(&blurred.data) {
        *v = if *d <= ZERO {
            ZERO
        } else {
            from_f64(
                (*d as f64).max(1e-6).powf(1. + amount as f64)
                    / (*b as f64).max(1e-6).powf(amount as f64),
            )
        };
    }
    for c in 0..3 {
        let before: f64 = input.pixels().map(|p| p[c] as f64).sum();
        let after: f64 = out.pixels().map(|p| p[c] as f64).sum();
        if after > 0. {
            for p in out.pixels_mut() {
                p[c] *= from_f64(before / after);
            }
        }
    }
    out
}

pub fn layered_gpu_params(
    pixel_size_um: f32,
    curves: &[[f64; 3]],
    layers: &[Vec<Vec<f64>>],
    positive: bool,
    p: &LayeredGrainParams,
) -> spektrafilm_gpu::LayeredGrainGpuParams {
    let maxima: [[f64; 3]; 3] = std::array::from_fn(|sl| {
        std::array::from_fn(|ch| {
            layers
                .iter()
                .map(|r| r[sl][ch])
                .fold(f64::NEG_INFINITY, f64::max)
        })
    });
    let total: [f64; 3] = std::array::from_fn(|ch| maxima.iter().map(|r| r[ch]).sum());
    let areas = coarsest_particle_area(layers, p);
    let layer_params = std::array::from_fn(|sl| {
        let fraction: [f64; 3] = std::array::from_fn(|ch| {
            if total[ch] > 0. && maxima[sl][ch].is_finite() {
                maxima[sl][ch] / total[ch]
            } else {
                0.
            }
        });
        let min: [f32; 3] = std::array::from_fn(|ch| (fraction[ch] * p.density_min[ch]) as f32);
        let max: [f32; 3] = std::array::from_fn(|ch| (maxima[sl][ch] + min[ch] as f64) as f32);
        let particles: [f32; 3] = std::array::from_fn(|ch| {
            ((pixel_size_um as f64).powi(2) * fraction[ch] / (areas[ch] * p.layer_scale[sl])) as f32
        });
        let cloud_sigma = std::array::from_fn(|ch| {
            if particles[ch] > 0. {
                p.dye_cloud_blur * (max[ch] / particles[ch]).sqrt()
            } else {
                0.
            }
        });
        spektrafilm_gpu::LayerParticleGpuParams {
            density_min: min,
            density_max: max,
            particles,
            cloud_sigma,
        }
    });
    let mut table = Vec::with_capacity(curves.len() * 3);
    for (i, row) in curves.iter().enumerate() {
        for ch in 0..3 {
            table.push([
                if positive { -row[ch] } else { row[ch] } as f32,
                layers[i][0][ch] as f32,
                layers[i][1][ch] as f32,
                layers[i][2][ch] as f32,
            ]);
        }
    }
    spektrafilm_gpu::LayeredGrainGpuParams {
        curves: table,
        layers: layer_params,
        uniformity: p.uniformity.map(|v| v as f32),
        density_min: p.density_min.map(|v| v as f32),
        blur: p.blur,
        usm_sigma: p.usm_sigma,
        usm_amount: p.usm_amount,
        micro_sigma: p.micro_structure[1] * 0.001 / pixel_size_um,
        micro_blur: p.micro_structure[0] / pixel_size_um,
        fast_stats: p.fast_stats,
        positive,
        monochrome: p.monochrome,
    }
}

pub fn apply_grain_to_density_layers(
    density: &ImageBuf,
    pixel_size_um: f32,
    curves: &[[f64; 3]],
    layers: &[Vec<Vec<f64>>],
    positive: bool,
    p: &LayeredGrainParams,
    backend: &dyn ComputeBackend,
) -> ImageBuf {
    if backend.supports_layered_grain() {
        if let Some(out) = backend.layered_grain(
            density,
            &layered_gpu_params(pixel_size_um, curves, layers, positive, p),
        ) {
            return out;
        }
    }
    use spektrafilm_math::numpy_rng::{GaussRng, rk_binomial, rk_poisson};
    let mut out = ImageBuf::new(density.width, density.height);
    let area = (pixel_size_um as f64).powi(2);
    let areas = coarsest_particle_area(layers, p);
    let channels: Vec<Vec<Scalar>> = (0..3)
        .into_par_iter()
        .map(|ch| {
            let axis: Vec<f64> = curves
                .iter()
                .map(|r| if positive { -r[ch] } else { r[ch] })
                .collect();
            let maxima: [f64; 3] = std::array::from_fn(|l| {
                layers
                    .iter()
                    .map(|r| r[l][ch])
                    .fold(f64::NEG_INFINITY, f64::max)
            });
            let total: f64 = maxima.iter().sum();
            let mut sum = vec![ZERO; density.pixel_count()];
            for sl in 0..3 {
                let fraction = maxima[sl] / total;
                let dmin = fraction * p.density_min[ch];
                let dmax = maxima[sl] + dmin;
                if fraction <= 0. || !fraction.is_finite() {
                    continue;
                }
                let npp = area * fraction / (areas[ch] * p.layer_scale[sl]);
                let od = dmax / npp;
                let ys: Vec<f64> = layers.iter().map(|r| r[sl][ch]).collect();
                let mut rng =
                    GaussRng::new(if p.monochrome { 0 } else { ch as u32 } + sl as u32 * 10);
                let mut probabilities = Vec::with_capacity(sum.len());
                let mut saturations = Vec::with_capacity(sum.len());
                for px in density.pixels() {
                    let q = if positive {
                        -(px[ch] as f64)
                    } else {
                        px[ch] as f64
                    };
                    // np.interp's right-biased exact matches, including plateaus.
                    let upper = axis.partition_point(|v| *v <= q);
                    let v = if q < axis[0] {
                        ys[0]
                    } else if upper >= axis.len() {
                        ys[ys.len() - 1]
                    } else {
                        let lo = upper - 1;
                        ys[lo] + (q - axis[lo]) / (axis[upper] - axis[lo]) * (ys[upper] - ys[lo])
                    };
                    let prob = ((v + dmin) / dmax).clamp(1e-6, 1. - 1e-6);
                    probabilities.push(prob);
                    saturations.push(1. - prob * p.uniformity[ch] * (1. - 1e-6));
                }
                let seeds: Vec<u64> = saturations
                    .iter()
                    .map(|sat| {
                        let lam = npp / sat;
                        if p.fast_stats && lam >= 30. {
                            (lam + lam.sqrt() * rng.gauss()).round_ties_even().max(0.) as u64
                        } else {
                            rk_poisson(&mut rng.rng, lam)
                        }
                    })
                    .collect();
                let mut sampled: Vec<Scalar> = seeds
                    .iter()
                    .enumerate()
                    .map(|(i, &n)| {
                        let prob = probabilities[i];
                        let mean = n as f64 * prob;
                        let var = mean * (1. - prob);
                        let count = if p.fast_stats && var > 10. {
                            (mean + var.sqrt() * rng.gauss())
                                .round_ties_even()
                                .clamp(0., n as f64) as u64
                        } else {
                            rk_binomial(&mut rng.rng, n, prob)
                        };
                        from_f64(count as f64 * od * saturations[i])
                    })
                    .collect();
                if p.dye_cloud_blur > 0. {
                    sampled = gaussian::gaussian_blur_channel(
                        &sampled,
                        density.width,
                        density.height,
                        (p.dye_cloud_blur as f64 * od.sqrt()) as f32,
                    );
                }
                for (a, b) in sum.iter_mut().zip(sampled) {
                    *a += b;
                }
            }
            sum
        })
        .collect();
    for (ch, values) in channels.iter().enumerate() {
        out.write_channel(ch, values);
    }
    // Clumping is stochastic in the Python numba path as well. Give this
    // independent field a stable seed for reproducible exports.
    let sigma = p.micro_structure[1] as f64 * 0.001 / pixel_size_um as f64;
    if sigma > 0.05 {
        let variance = (1. + sigma * sigma).ln();
        let mut rng = GaussRng::new(777);
        let field_data = if p.monochrome {
            // The native monochrome mode represents one emulsion. Broadcast
            // one clumping draw per pixel across its RGB storage channels.
            (0..out.pixel_count())
                .flat_map(|_| {
                    let value = from_f64((-variance / 2. + variance.sqrt() * rng.gauss()).exp());
                    [value; 3]
                })
                .collect()
        } else {
            (0..out.data.len())
                .map(|_| from_f64((-variance / 2. + variance.sqrt() * rng.gauss()).exp()))
                .collect()
        };
        let mut field = ImageBuf::from_data(density.width, density.height, field_data);
        let blur = p.micro_structure[0] / pixel_size_um;
        if blur > 0.4 {
            field = gaussian::gaussian_blur(&field, blur);
        }
        for (v, f) in out.data.iter_mut().zip(field.data) {
            *v *= f;
        }
    }
    if p.blur > 0. {
        out = gaussian::gaussian_blur(&out, p.blur);
    }
    out = multiplicative_unsharp_mask(&out, p.usm_sigma, p.usm_amount);
    for px in out.data.chunks_exact_mut(3) {
        for c in 0..3 {
            px[c] -= from_f64(p.density_min[c]);
        }
    }
    out
}

/// Apply grain to a CMY density image.
///
/// Port of Python `apply_grain_to_density`.
#[allow(clippy::too_many_arguments)]
pub fn apply_grain_to_density(
    density_cmy: &ImageBuf,
    pixel_size_um: f32,
    agx_particle_area_um2: f64,
    agx_particle_scale: [f64; 3],
    density_min: [f64; 3],
    density_max_curves: [f64; 3],
    grain_uniformity: [f64; 3],
    grain_blur: f32,
    n_sub_layers: u32,
    monochrome: bool,
    backend: &dyn ComputeBackend,
) -> ImageBuf {
    let stage_timings = stage_timings_enabled();
    let w = density_cmy.width;
    let h = density_cmy.height;
    let pixel_area = (pixel_size_um as f64) * (pixel_size_um as f64);
    let density_max: [f64; 3] = [
        density_max_curves[0] + density_min[0],
        density_max_curves[1] + density_min[1],
        density_max_curves[2] + density_min[2],
    ];

    let channels: Vec<(usize, Vec<Scalar>)> = (0..3)
        .into_par_iter()
        .map(|ch| {
            let t_ch = Instant::now();
            let particle_area = agx_particle_area_um2 * agx_particle_scale[ch];
            let mut n_particles = pixel_area / particle_area;
            if n_sub_layers > 1 {
                n_particles /= n_sub_layers as f64;
            }

            // Add density_min to input (kept in Scalar precision)
            let dmin_s = from_f64(density_min[ch]);
            let t = Instant::now();
            let density_ch: Vec<Scalar> = density_cmy.pixels().map(|px| px[ch] + dmin_s).collect();
            print_stage_timing(stage_timings, "grain.extract_channel", t);

            let mut grain_sum = vec![ZERO; density_ch.len()];

            for sl in 0..n_sub_layers {
                let t = Instant::now();
                // Monochrome (B&W): one shared noise field — every lane draws
                // from channel 0's RNG stream for the native monochrome
                // profile extension, seed ch=0.
                let seed_ch = if monochrome { 0 } else { ch as u64 };
                let seed = seed_ch + (sl as u64) * 10;
                let g = layer_particle_model(
                    &density_ch,
                    w,
                    h,
                    density_max[ch],
                    n_particles,
                    grain_uniformity[ch],
                    seed,
                    0.0,
                );
                print_stage_timing(stage_timings, "grain.layer_particle_model", t);
                for (s, &v) in grain_sum.iter_mut().zip(g.iter()) {
                    *s += v;
                }
            }

            // Average sub-layers
            let scale = from_f64(1.0 / n_sub_layers as f64);
            for v in &mut grain_sum {
                *v = *v * scale - dmin_s;
            }

            print_stage_timing(stage_timings, "grain.channel_total", t_ch);
            (ch, grain_sum)
        })
        .collect();

    let mut out = ImageBuf::new(w, h);
    for (ch, grain_sum) in channels {
        out.write_channel(ch, &grain_sum);
    }

    // Final blur — typically a few px sigma at 1–6 MP, big enough that the
    // GPU separable kernel wins.
    if grain_blur > 0.4 {
        let t = Instant::now();
        out = backend.gaussian_blur(&out, grain_blur);
        print_stage_timing(stage_timings, "grain.final_blur", t);
    }

    out
}
