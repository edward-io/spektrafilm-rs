//! Shared stock preparation for preview, export and LUT baking.
use crate::{
    params::{BaseDensityParams, RuntimeParams},
    profile::Profile,
};
use serde_json::Value;
use std::sync::LazyLock;

static GRAIN: LazyLock<Value> = LazyLock::new(|| {
    toml::from_str(include_str!("../../../data/presets/grain.toml")).expect("bundled grain presets")
});
static COUPLERS: LazyLock<Value> = LazyLock::new(|| {
    toml::from_str(include_str!("../../../data/presets/couplers.toml"))
        .expect("bundled coupler presets")
});

pub fn apply_presets(film: &Profile, params: &mut RuntimeParams) {
    fn apply<T: serde::Serialize + serde::de::DeserializeOwned>(
        target: &mut T,
        library: &Value,
        film: &Profile,
    ) {
        let channel = if film.is_bw() { "bw" } else { "color" };
        let polarity = if film.is_positive() {
            "positive"
        } else {
            "negative"
        };
        let defaults = &library["defaults"][channel][polarity];
        let cine = if film.info.usage == "cine" && !film.is_positive() && !film.is_bw() {
            &defaults["cine"][if film
                .info
                .reference_illuminant
                .to_uppercase()
                .starts_with('T')
            {
                "tungsten"
            } else {
                "daylight"
            }]
        } else {
            &Value::Null
        };
        let mut value = serde_json::to_value(&*target).expect("serializable stock params");
        for source in [
            defaults,
            cine,
            &library[film.info.stock.as_deref().unwrap_or("")],
        ] {
            if let Some(fields) = source.as_object() {
                for (key, v) in fields {
                    if value.get(key).is_some() && !v.is_object() {
                        value[key] = v.clone();
                    }
                }
            }
        }
        *target = serde_json::from_value(value).expect("valid bundled stock preset");
    }
    apply(&mut params.film_render.grain, &GRAIN, film);
    apply(&mut params.film_render.dir_couplers, &COUPLERS, film);
}

fn smooth(values: &[f64], sigma: f64) -> Vec<f64> {
    let radius = (4. * sigma + 0.5) as isize;
    let mut kernel: Vec<_> = (-radius..=radius)
        .map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp())
        .collect();
    let sum: f64 = kernel.iter().sum();
    for k in &mut kernel {
        *k /= sum;
    }
    let n = values.len() as isize;
    (0..values.len())
        .map(|i| {
            kernel
                .iter()
                .enumerate()
                .map(|(k, w)| {
                    let j = (i as isize + k as isize - radius).rem_euclid(2 * n);
                    values[if j < n { j } else { 2 * n - 1 - j } as usize] * w
                })
                .sum()
        })
        .collect()
}
fn interp(x: f64, axis: [f64; 3], v: [f64; 3]) -> f64 {
    if x <= axis[0] {
        v[0]
    } else if x >= axis[2] {
        v[2]
    } else {
        let i = if x < axis[1] { 0 } else { 1 };
        v[i] + (v[i + 1] - v[i]) * (x - axis[i]) / (axis[i + 1] - axis[i])
    }
}
pub fn tune_base(base: &[f64], p: &BaseDensityParams, film: bool) -> Vec<f64> {
    if !p.active {
        return base.to_vec();
    }
    if [p.cyan, p.magenta, p.yellow] == [1.; 3] && (!film || p.tilt == 0.) {
        return base.iter().map(|v| v * p.scale).collect();
    }
    if film {
        let scales = smooth(
            &(0..base.len())
                .map(|i| {
                    interp(
                        380. + 5. * i as f64,
                        [460., 555., 650.],
                        [p.yellow, p.magenta, p.cyan],
                    )
                })
                .collect::<Vec<_>>(),
            7.,
        );
        let tilt = smooth(
            &(0..base.len())
                .map(|i| (1. + p.tilt * (380. + 5. * i as f64 - 555.) / 95.).max(0.))
                .collect::<Vec<_>>(),
            7.,
        );
        base.iter()
            .zip(scales)
            .zip(tilt)
            .map(|((b, s), t)| b * p.scale * s * t * p.scale)
            .collect()
    } else {
        let peaks = [445., 530., 610.];
        let sample = |x: f64| {
            let f = (x - 380.) / 5.;
            let i = f.floor() as usize;
            let j = (i + 1).min(base.len() - 1);
            base[i] + (f - i as f64) * (base[j] - base[i])
        };
        let v = [
            sample(peaks[0]) * p.yellow,
            sample(peaks[1]) * p.magenta,
            sample(peaks[2]) * p.cyan,
        ];
        smooth(
            &(0..base.len())
                .map(|i| interp(380. + 5. * i as f64, peaks, v))
                .collect::<Vec<_>>(),
            4.,
        )
        .into_iter()
        .map(|v| v * p.scale)
        .collect()
    }
}

pub fn prepare(
    mut profile: Profile,
    params: &RuntimeParams,
    film: bool,
) -> Result<Profile, String> {
    let chemistry = if film {
        &params.film_render.chemistry
    } else {
        &params.print_render.chemistry
    };
    let time = chemistry.development_time.or(if film {
        params.film_render.development_time
    } else {
        params.print_render.development_time
    });
    profile = crate::profile::resolve_for_render(profile, time);
    if let Some(model) = &profile.data.density_curves_model {
        let mut p = chemistry.clone();
        // Printing applies its chemistry when developing the print, after exposure.
        if !film {
            p.active = false;
        }
        let (total, layers) = crate::print_morph::morph_density_curves_with_layers(
            &profile.data.log_exposure,
            model,
            &p,
            profile.is_positive(),
        )?;
        profile.data.density_curves = total.into_iter().map(|r| r.to_vec()).collect();
        profile.data.density_curves_layers = if model.n_layers() == 1 {
            let normalized = spektrafilm_model::density_curves::normalize_density_curves_f64(
                &profile.density_curves_f64(),
            );
            normalized
                .into_iter()
                .map(|row| vec![row.to_vec(), vec![0.; 3], vec![0.; 3]])
                .collect()
        } else {
            layers
        };
    }
    profile.data.base_density = tune_base(
        &profile.data.base_density,
        if film {
            &params.film_render.base
        } else {
            &params.print_render.base
        },
        film,
    );
    Ok(profile)
}
