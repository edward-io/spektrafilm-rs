//! Standard illuminants and normalized Planck spectra on the runtime wavelength grid.
#[path = "illuminant_registry.rs"]
mod registry;
// These 81-sample arrays are cheap to return by value; arbitrary BB temperatures
// must not be interned into leaked allocations for the lifetime of the process.
pub fn get(name: &str) -> Result<[f64; 81], String> {
    if let Some((_, values)) = registry::ILLUMINANTS.iter().find(|(key, _)| *key == name) {
        return Ok(*values);
    }
    if let Some(temperature) = name
        .strip_prefix("BB")
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|t| t.is_finite() && *t > 0.)
    {
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
        return Ok(spectrum);
    }
    Err(format!("unknown illuminant: {name}"))
}
pub fn get_f32(name: &str) -> Result<[f32; 81], String> {
    Ok(get(name)?.map(|v| v as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectra_are_owned_and_keep_their_values_across_temperature_changes() {
        let first = get("BB3200").unwrap();
        for i in 0..2048 {
            let name = format!("BB{}", 2000. + i as f64 / 4.);
            let spectrum = get(&name).unwrap();
            assert!(spectrum.iter().all(|v| v.is_finite() && *v >= 0.));
            assert!((spectrum.iter().sum::<f64>() / 81. - 1.).abs() < 1e-12);
            assert_eq!(get_f32(&name).unwrap(), spectrum.map(|v| v as f32));
        }
        assert_eq!(first, get("BB3.2e3").unwrap());
        for name in names() {
            assert_eq!(get_f32(name).unwrap(), get(name).unwrap().map(|v| v as f32));
        }
        for name in ["BB0", "BB-1", "BBNaN", "BBinf", "BB1", "unknown"] {
            assert!(get(name).is_err(), "{name}");
            assert!(get_f32(name).is_err(), "{name}");
        }
    }
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

pub fn names() -> impl Iterator<Item = &'static str> {
    registry::ILLUMINANTS.iter().map(|(name, _)| *name)
}
