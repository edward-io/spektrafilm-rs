//! Standard illuminants and normalized Planck spectra on the runtime wavelength grid.
#[path = "illuminant_registry.rs"]
mod registry;
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
pub fn get(name: &str) -> Result<&'static [f64; 81], String> {
    if let Some((_, values)) = registry::ILLUMINANTS.iter().find(|(key, _)| *key == name) {
        return Ok(values);
    }
    if let Some(temperature) = name
        .strip_prefix("BB")
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|t| t.is_finite() && *t > 0.)
    {
        static CACHE: OnceLock<Mutex<HashMap<String, &'static [f64; 81]>>> = OnceLock::new();
        let mut cache = CACHE
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap();
        if let Some(values) = cache.get(name) {
            return Ok(*values);
        }
        let mut spectrum = std::array::from_fn(|i| {
            let wl = (380. + 5. * i as f64) * 1e-9;
            1. / (wl.powi(5) * (0.014388 / (wl * temperature)).exp_m1())
        });
        let mean = spectrum.iter().sum::<f64>() / 81.;
        for v in &mut spectrum {
            *v /= mean;
        }
        if spectrum.iter().any(|v| !v.is_finite()) {
            return Err("blackbody spectrum is outside the supported numerical range".into());
        }
        let values: &'static [f64; 81] = Box::leak(Box::new(spectrum));
        cache.insert(name.into(), values);
        return Ok(values);
    }
    Err(format!("unknown illuminant: {name}"))
}
pub fn get_f32(name: &str) -> Result<&'static [f32; 81], String> {
    static CACHE: OnceLock<Mutex<HashMap<String, &'static [f32; 81]>>> = OnceLock::new();
    let mut cache = CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    if let Some(v) = cache.get(name) {
        return Ok(*v);
    }
    let values: &'static [f32; 81] = Box::leak(Box::new(get(name)?.map(|v| v as f32)));
    cache.insert(name.into(), values);
    Ok(values)
}

/// CIE D-series white at >=4000 K; Kang 2002 Planckian approximation below.
pub fn white_xyz(t: f64) -> [f64; 3] {
    let t = t.clamp(1667., 25000.);
    let x = if t >= 4000. {
        if t <= 7000. {
            -4.6070e9 / t.powi(3) + 2.9678e6 / t.powi(2) + 0.09911e3 / t + 0.244063
        } else {
            -2.0064e9 / t.powi(3) + 1.9018e6 / t.powi(2) + 0.24748e3 / t + 0.237040
        }
    } else {
        -0.2661239e9 / t.powi(3) - 0.2343589e6 / t.powi(2) + 0.8776956e3 / t + 0.179910
    };
    let y = if t >= 4000. {
        -3. * x * x + 2.87 * x - 0.275
    } else if t <= 2222. {
        -1.1063814 * x.powi(3) - 1.34811020 * x * x + 2.18555832 * x - 0.20219683
    } else {
        -0.9549476 * x.powi(3) - 1.37418593 * x * x + 2.09137015 * x - 0.16748867
    };
    [x / y, 1., (1. - x - y) / y]
}
