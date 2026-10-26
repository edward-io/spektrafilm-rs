//! White balance controls relative to the decoder's existing rendering.
use crate::{LoadedImage, RawOptions};
use anyhow::{Result, bail};
use rayon::prelude::*;
use spektrafilm_math::{colorspace, precision::from_f64, transfer};

/// Estimated illuminant coordinates, not camera-vendor or Lightroom calibration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WhiteBalance {
    pub temperature: f64,
    pub tint: f64,
}

pub(crate) fn planckian_xyz(temperature: f64) -> [f64; 3] {
    let t = temperature.clamp(1667., 25000.);
    let x = if t <= 4000. {
        -0.2661239e9 / t.powi(3) - 0.2343589e6 / t.powi(2) + 0.8776956e3 / t + 0.179910
    } else {
        -3.0258469e9 / t.powi(3) + 2.1070379e6 / t.powi(2) + 0.2226347e3 / t + 0.240390
    };
    let y = if t <= 2222. {
        -1.1063814 * x.powi(3) - 1.34811020 * x * x + 2.18555832 * x - 0.20219683
    } else if t <= 4000. {
        -0.9549476 * x.powi(3) - 1.37418593 * x * x + 2.09137015 * x - 0.16748867
    } else {
        3.0817580 * x.powi(3) - 5.87338670 * x * x + 3.75112997 * x - 0.37001483
    };
    [x / y, 1., (1. - x - y) / y]
}

impl WhiteBalance {
    pub(crate) fn from_xyz(xyz: [f64; 3]) -> Option<Self> {
        if xyz.iter().any(|v| !v.is_finite() || *v <= 0.) {
            return None;
        }
        let ratio = xyz[0] / xyz[2];
        let (mut lo, mut hi) = (1667., 25000.);
        for _ in 0..40 {
            let mid = (lo + hi) / 2.;
            let white = planckian_xyz(mid);
            if white[0] / white[2] > ratio {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let temperature = (lo + hi) / 2.;
        let tint = 1000. * (planckian_xyz(temperature)[0] * xyz[1] / xyz[0]).ln();
        Some(Self { temperature, tint })
    }

    pub(crate) fn for_options(as_shot: Option<Self>, options: &RawOptions) -> Option<Self> {
        match options.white_balance.as_str() {
            "camera" | "as_shot" => as_shot,
            "daylight" => Some(Self {
                temperature: 6504.,
                tint: 0.,
            }),
            "tungsten" => Some(Self {
                temperature: 2850.,
                tint: 0.,
            }),
            "custom" => Some(Self {
                temperature: options.temperature,
                tint: -1000. * options.tint.ln(),
            }),
            _ => None,
        }
    }

    pub fn shifted(self, shift: [f64; 2]) -> Self {
        Self {
            temperature: 1e6 / (1e6 / self.temperature + shift[0]).clamp(1e6 / 25000., 1e6 / 1667.),
            tint: self.tint + shift[1],
        }
    }

    pub fn shift_to(self, target: Self) -> [f64; 2] {
        [
            1e6 / target.temperature - 1e6 / self.temperature,
            target.tint - self.tint,
        ]
    }

    fn xyz(self) -> [f64; 3] {
        let white = planckian_xyz(self.temperature);
        let green = (self.tint / 1000.).exp();
        [white[0] / green, 1., white[2] / green]
    }
}

pub(crate) fn apply(image: &mut LoadedImage, shift: [f64; 2]) -> Result<()> {
    if shift.iter().any(|v| !v.is_finite()) || shift[0].abs() > 1000. || shift[1].abs() > 300. {
        bail!("RAW white balance shifts must be finite and within 1000 mired / 300 tint units");
    }
    if shift == [0.; 2] {
        return Ok(());
    }
    let Some(base) = image.metadata.white_balance else {
        bail!("This RAW does not provide a usable camera white balance");
    };
    let target = base.shifted(shift);
    let cs = transfer::get(&image.color_space).map_err(anyhow::Error::msg)?;
    // Preserve the decoder's exact baseline; only apply the requested difference.
    let matrix = transfer::matmul(
        cs.from_xyz,
        transfer::matmul(
            colorspace::chromatic_adaptation_matrix_f64(target.xyz(), base.xyz()),
            cs.to_xyz,
        ),
    );
    image.pixels.data.par_chunks_exact_mut(3).for_each(|p| {
        let rgb = [p[0] as f64, p[1] as f64, p[2] as f64];
        for c in 0..3 {
            p[c] = from_f64((0..3).map(|j| matrix[c][j] * rgb[j]).sum());
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn illuminant_coordinates_round_trip_and_shifts_are_continuous() {
        for temperature in [2000., 3000., 4500., 6500., 10000., 20000.] {
            for tint in [-50., 0., 50.] {
                let wb = WhiteBalance { temperature, tint };
                let recovered = WhiteBalance::from_xyz(wb.xyz()).unwrap();
                assert!((recovered.temperature - temperature).abs() < 1e-5);
                assert!((recovered.tint - tint).abs() < 1e-7);
                let target = WhiteBalance {
                    temperature: temperature + 1.,
                    tint: tint + 0.01,
                };
                let shifted = wb.shifted(wb.shift_to(target));
                assert!((shifted.temperature - target.temperature).abs() < 1e-9);
                assert!((shifted.tint - target.tint).abs() < 1e-9);
            }
        }
        assert!(WhiteBalance::from_xyz([0.; 3]).is_none());
        assert!(WhiteBalance::from_xyz([f64::NAN, 1., 1.]).is_none());
    }
}
