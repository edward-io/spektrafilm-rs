//! Named RGB primaries and transport curves. Pixel values are never clipped here.
#[path = "color_registry.rs"]
mod registry;

#[derive(Clone, Copy, Debug)]
pub enum Transfer {
    Linear,
    Srgb,
    ProPhoto,
    Bt709,
    Gamma(f64),
    Acescc,
    Acescct,
    Vlog,
    Slog3,
    Logc3,
    Logc4,
    DaVinci,
    Blackmagic5,
    Flog,
    Flog2,
    Canonlog3,
    Log3g10,
    Applelog,
    Dlog,
    Nlog,
    Pq,
    Hlg,
    Acesproxy,
    Dcdm,
    Cineon,
    Erimm,
    Tlog,
    Protune,
    Rimm,
    Slog2,
    Smpte240,
    Lab,
    Unsupported,
}

#[derive(Clone, Copy, Debug)]
pub struct ColorSpace {
    pub name: &'static str,
    pub to_xyz: [[f64; 3]; 3],
    pub from_xyz: [[f64; 3]; 3],
    pub white: [f64; 3],
    pub transfer: Transfer,
}

pub fn get(name: &str) -> Result<&'static ColorSpace, String> {
    let name = match name {
        "Rec. 2020" | "Rec2020" => "ITU-R BT.2020",
        "Rec.709" => "ITU-R BT.709",
        n => n,
    };
    registry::SPACES
        .iter()
        .find(|s| s.name == name)
        .ok_or_else(|| format!("unknown RGB color space: {name}"))
}

pub fn names() -> impl Iterator<Item = &'static str> {
    registry::SPACES.iter().map(|s| s.name)
}

impl Transfer {
    pub fn supported(self) -> bool {
        !matches!(self, Self::Unsupported)
    }
    pub fn encode(self, x: f64) -> f64 {
        match self {
            Self::Linear => x,
            Self::Srgb => {
                if x <= 0.0031308 {
                    12.92 * x
                } else {
                    1.055 * x.powf(1. / 2.4) - 0.055
                }
            }
            Self::ProPhoto => {
                if x < 1. / 512. {
                    16. * x
                } else {
                    x.powf(1. / 1.8)
                }
            }
            Self::Bt709 => {
                if x < 0.018 {
                    4.5 * x
                } else {
                    1.099 * x.powf(0.45) - 0.099
                }
            }
            Self::Gamma(g) => x.signum() * x.abs().powf(1. / g),
            Self::Acescc => {
                if x <= 0. {
                    (-16. + 9.72) / 17.52
                } else if x < 2f64.powi(-15) {
                    ((2f64.powi(-16) + x * 0.5).log2() + 9.72) / 17.52
                } else {
                    (x.log2() + 9.72) / 17.52
                }
            }
            Self::Acescct => {
                if x <= 0.0078125 {
                    10.5402377416545 * x + 0.0729055341958355
                } else {
                    (x.log2() + 9.72) / 17.52
                }
            }
            Self::Vlog => {
                if x < 0.01 {
                    5.6 * x + 0.125
                } else {
                    0.241514 * (x + 0.00873).log10() + 0.598206
                }
            }
            Self::Slog3 => {
                if x >= 0.01125 {
                    (420. + (x + 0.01).log10().mul_add(261.5, -0.19f64.log10() * 261.5)) / 1023.
                } else {
                    (x * (171.2102946929 - 95.) / 0.01125 + 95.) / 1023.
                }
            }
            Self::Logc3 => {
                if x > 0.010591 {
                    0.247190 * (5.555556 * x + 0.052272).log10() + 0.385537
                } else {
                    5.367655 * x + 0.092809
                }
            }
            Self::Logc4 => {
                let a = (2f64.powi(18) - 16.) / 117.45;
                let b = (1023. - 95.) / 1023.;
                let c = 95. / 1023.;
                let s = (7. * std::f64::consts::LN_2 * 2f64.powf(7. - 14. * c / b)) / (a * b);
                let t = (2f64.powf(6. - 14. * c / b) - 64.) / a;
                if x < t {
                    (x - t) / s
                } else {
                    ((a * x + 64.).log2() - 6.) / 14. * b + c
                }
            }
            Self::DaVinci => {
                if x <= 0.00262409 {
                    x * 10.44426855
                } else {
                    ((x + 0.0075).log2() + 7.) * 0.07329248
                }
            }
            Self::Blackmagic5 => {
                if x < 0.005 {
                    8.283605932402494 * x + 0.09246575342465753
                } else {
                    0.08692876065491224 * (x + 0.005494072432257808).ln() + 0.5300133392291939
                }
            }
            Self::Flog | Self::Flog2 => {
                let (cut, a, b, c, d, e, f) = if matches!(self, Self::Flog) {
                    (
                        0.00089, 0.555556, 0.009468, 0.344676, 0.790453, 8.735631, 0.092864,
                    )
                } else {
                    (
                        0.000889, 5.555556, 0.064829, 0.245281, 0.384316, 8.799461, 0.092864,
                    )
                };
                if x < cut {
                    e * x + f
                } else {
                    c * (a * x + b).log10() + d
                }
            }
            Self::Canonlog3 => {
                let x = x / 0.9;
                if x < -0.014 {
                    -0.36726845 * (-x * 14.98325 + 1.).log10() + 0.12783901
                } else if x <= 0.014 {
                    1.9754798 * x + 0.12512219
                } else {
                    0.36726845 * (x * 14.98325 + 1.).log10() + 0.12240537
                }
            }
            Self::Log3g10 => {
                let v = x + 0.01;
                if v < 0. {
                    v * 15.1927
                } else {
                    0.224282 * (v * 155.975327 + 1.).log10()
                }
            }
            Self::Applelog => {
                if x >= 0.01 {
                    0.08550479 * (x + 0.00964052).log2() + 0.69336945
                } else if x >= -0.05641088 {
                    47.28711236 * (x + 0.05641088).powi(2)
                } else {
                    0.
                }
            }
            Self::Dlog => {
                if x <= 0.0078 {
                    6.025 * x + 0.0929
                } else {
                    (x * 0.9892 + 0.0108).log10() * 0.256663 + 0.584555
                }
            }
            Self::Nlog => {
                if x < 0.328 {
                    0.635386119257087 * (x + 0.0075).cbrt()
                } else {
                    0.1466275659824047 * x.ln() + 0.6050830889540567
                }
            }
            Self::Pq => {
                let y = (x / 10000.).max(0.).powf(2610. / 16384.);
                ((3424. / 4096. + 2413. / 128. * y) / (1. + 2392. / 128. * y)).powf(2523. / 32.)
            }
            Self::Hlg => {
                let e = x.signum() * (x.abs() / 1000.).powf(1. / 1.2);
                if e <= 1. / 12. {
                    (3. * e).max(0.).sqrt()
                } else {
                    0.17883277 * (12. * e - 0.28466892).ln() + 0.559910729529562
                }
            }
            Self::Acesproxy => {
                if x > 2f64.powf(-9.72) {
                    ((x.log2() + 2.5) * 50. + 425.)
                        .round_ties_even()
                        .clamp(64., 940.)
                        / 1023.
                } else {
                    64. / 1023.
                }
            }
            Self::Dcdm => (x / 52.37).signum() * (x.abs() / 52.37).powf(1. / 2.6),
            Self::Cineon => {
                let black = 10f64.powf((95. - 685.) / 300.);
                (685. + 300. * (x * (1. - black) + black).log10()) / 1023.
            }
            Self::Erimm => {
                let span = (316.2f64 / 0.001).ln();
                let toe = std::f64::consts::E * 0.001;
                if x < 0. {
                    0.
                } else if x <= toe {
                    x / (toe * span)
                } else {
                    (x / 0.001).ln() / span
                }
            }
            Self::Tlog => {
                let (a, b, c, g) = tlog_parameters();
                if x < 0. {
                    g * x + 0.075
                } else {
                    (x + c).ln() * b + a
                }
            }
            Self::Protune => (112. * x).ln_1p() / 113f64.ln(),
            Self::Rimm => Self::Bt709.encode(x.max(0.)) / (1.099 * 2f64.powf(0.45) - 0.099),
            Self::Slog2 => {
                let v = x * 155. / 219. / 0.9;
                let y = if v >= 0. {
                    0.432699 * (v + 0.037584).log10() + 0.616596 + 0.03
                } else {
                    v * 5. + 0.030001222851889303
                };
                (y * 876. + 64.) / 1023.
            }
            Self::Smpte240 => {
                if x < 0.0228 {
                    4. * x
                } else {
                    1.1115 * x.powf(0.45) - 0.1115
                }
            }
            Self::Lab => {
                if x > 216. / 24389. {
                    (116. * x.cbrt() - 16.) / 100.
                } else {
                    (24389. / 27.) * x / 100.
                }
            }
            Self::Unsupported => {
                panic!("unsupported transfer function; validate at pipeline construction")
            }
        }
    }
    pub fn decode(self, x: f64) -> f64 {
        match self {
            Self::Linear => x,
            Self::Srgb => {
                if x <= 0.04045 {
                    x / 12.92
                } else {
                    ((x + 0.055) / 1.055).powf(2.4)
                }
            }
            Self::ProPhoto => {
                if x < 0.03125 {
                    x / 16.
                } else {
                    x.powf(1.8)
                }
            }
            Self::Bt709 => {
                if x < 0.081 {
                    x / 4.5
                } else {
                    ((x + 0.099) / 1.099).powf(1. / 0.45)
                }
            }
            Self::Gamma(g) => x.signum() * x.abs().powf(g),
            Self::Acescc => {
                if x < (-15. + 9.72) / 17.52 {
                    (2f64.powf(x * 17.52 - 9.72) - 2f64.powi(-16)) * 2.
                } else {
                    2f64.powf(x * 17.52 - 9.72).min(65504.)
                }
            }
            Self::Acescct => {
                if x <= 0.155251141552511 {
                    (x - 0.0729055341958355) / 10.5402377416545
                } else {
                    2f64.powf(x * 17.52 - 9.72)
                }
            }
            Self::Vlog => {
                if x < 0.181 {
                    (x - 0.125) / 5.6
                } else {
                    10f64.powf((x - 0.598206) / 0.241514) - 0.00873
                }
            }
            Self::Slog3 => {
                if x * 1023. >= 171.2102946929 {
                    10f64.powf((x * 1023. - 420.) / 261.5) * 0.19 - 0.01
                } else {
                    (x * 1023. - 95.) * 0.01125 / (171.2102946929 - 95.)
                }
            }
            Self::Logc3 => {
                if x > 5.367655 * 0.010591 + 0.092809 {
                    (10f64.powf((x - 0.385537) / 0.247190) - 0.052272) / 5.555556
                } else {
                    (x - 0.092809) / 5.367655
                }
            }
            Self::Logc4 => {
                let a = (2f64.powi(18) - 16.) / 117.45;
                let b = (1023. - 95.) / 1023.;
                let c = 95. / 1023.;
                let s = (7. * std::f64::consts::LN_2 * 2f64.powf(7. - 14. * c / b)) / (a * b);
                let t = (2f64.powf(6. - 14. * c / b) - 64.) / a;
                if x < 0. {
                    x * s + t
                } else {
                    (2f64.powf(14. * (x - c) / b + 6.) - 64.) / a
                }
            }
            Self::DaVinci => {
                if x <= 0.02740668 {
                    x / 10.44426855
                } else {
                    2f64.powf(x / 0.07329248 - 7.) - 0.0075
                }
            }
            Self::Blackmagic5 => {
                if x < 0.005 * 8.283605932402494 + 0.09246575342465753 {
                    (x - 0.09246575342465753) / 8.283605932402494
                } else {
                    ((x - 0.5300133392291939) / 0.08692876065491224).exp() - 0.005494072432257808
                }
            }
            Self::Flog | Self::Flog2 => {
                let (cut, a, b, c, d, e, f) = if matches!(self, Self::Flog) {
                    (
                        0.100537775223865,
                        0.555556,
                        0.009468,
                        0.344676,
                        0.790453,
                        8.735631,
                        0.092864,
                    )
                } else {
                    (
                        0.100686685370811,
                        5.555556,
                        0.064829,
                        0.245281,
                        0.384316,
                        8.799461,
                        0.092864,
                    )
                };
                if x < cut {
                    (x - f) / e
                } else {
                    (10f64.powf((x - d) / c) - b) / a
                }
            }
            Self::Canonlog3 => {
                0.9 * if x < 0.097465473 {
                    -(10f64.powf((0.12783901 - x) / 0.36726845) - 1.) / 14.98325
                } else if x <= 0.15277891 {
                    (x - 0.12512219) / 1.9754798
                } else {
                    (10f64.powf((x - 0.12240537) / 0.36726845) - 1.) / 14.98325
                }
            }
            Self::Log3g10 => {
                if x < 0. {
                    x / 15.1927 - 0.01
                } else {
                    (10f64.powf(x / 0.224282) - 1.) / 155.975327 - 0.01
                }
            }
            Self::Applelog => {
                if x >= 47.28711236 * (0.01 + 0.05641088f64).powi(2) {
                    2f64.powf((x - 0.69336945) / 0.08550479) - 0.00964052
                } else if x >= 0. {
                    (x / 47.28711236).sqrt() - 0.05641088
                } else {
                    -0.05641088
                }
            }
            Self::Dlog => {
                if x <= 0.14 {
                    (x - 0.0929) / 6.025
                } else {
                    (10f64.powf(3.89616 * x - 2.27752) - 0.0108) / 0.9892
                }
            }
            Self::Nlog => {
                if x < 0.4418377321603128 {
                    (x / 0.635386119257087).powi(3) - 0.0075
                } else {
                    ((x - 0.6050830889540567) / 0.1466275659824047).exp()
                }
            }
            Self::Pq => {
                let p = x.max(0.).powf(32. / 2523.);
                10000.
                    * ((p - 3424. / 4096.).max(0.) / (2413. / 128. - 2392. / 128. * p))
                        .powf(16384. / 2610.)
            }
            Self::Hlg => {
                let e = if x <= 0.5 {
                    x * x / 3.
                } else {
                    (((x - 0.559910729529562) / 0.17883277).exp() + 0.28466892) / 12.
                };
                1000. * e.powf(1.2)
            }
            Self::Acesproxy => 2f64.powf((x * 1023. - 425.) / 50. - 2.5),
            Self::Dcdm => 52.37 * x.signum() * x.abs().powf(2.6),
            Self::Cineon => {
                let black = 10f64.powf((95. - 685.) / 300.);
                (10f64.powf((1023. * x - 685.) / 300.) - black) / (1. - black)
            }
            Self::Erimm => {
                let span = (316.2f64 / 0.001).ln();
                if x <= 1. / span {
                    span * x * std::f64::consts::E * 0.001
                } else {
                    (x * span + 0.001f64.ln()).exp()
                }
            }
            Self::Tlog => {
                let (a, b, c, g) = tlog_parameters();
                if x < 0.075 {
                    (x - 0.075) / g
                } else {
                    ((x - a) / b).exp() - c
                }
            }
            Self::Protune => (113f64.powf(x) - 1.) / 112.,
            Self::Rimm => {
                let v = x * (1.099 * 2f64.powf(0.45) - 0.099);
                if x < Self::Rimm.encode(0.018) {
                    v / 4.5
                } else {
                    ((v + 0.099) / 1.099).powf(1. / 0.45)
                }
            }
            Self::Slog2 => {
                let v = (x * 1023. - 64.) / 876.;
                let y = if x >= Self::Slog2.encode(0.) {
                    10f64.powf((v - 0.616596 - 0.03) / 0.432699) - 0.037584
                } else {
                    (v - 0.030001222851889303) / 5.
                };
                219. * y * 0.9 / 155.
            }
            Self::Smpte240 => {
                if x < Self::Smpte240.encode(0.0228) {
                    x / 4.
                } else {
                    ((x + 0.1115) / 1.1115).powf(1. / 0.45)
                }
            }
            Self::Lab => {
                if x > 0.08 {
                    ((x * 100. + 16.) / 116.).powi(3)
                } else {
                    x * 100. / (24389. / 27.)
                }
            }
            Self::Unsupported => {
                panic!("unsupported transfer function; validate at pipeline construction")
            }
        }
    }
}

pub fn matmul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

pub fn rgb_matrix(src: &str, dst: &str) -> [[f64; 3]; 3] {
    let a = get(src).expect("validated input primaries");
    let b = get(dst).expect("validated output primaries");
    matmul(
        b.from_xyz,
        matmul(
            crate::colorspace::chromatic_adaptation_matrix_f64(a.white, b.white),
            a.to_xyz,
        ),
    )
}

fn tlog_parameters() -> (f64, f64, f64, f64) {
    let b = 1. / (0.7107 + 1.2359 * (128f64 * 16.).ln());
    let gs = 16. / (1. - 0.075);
    let c = b / gs;
    let a = 1. - b * (128. + c).ln();
    let y0 = a + b * c.ln();
    let scale = (1. - 0.075) / (1. - y0);
    (1. + (a - 1.) * scale, b * scale, c, gs * scale)
}
