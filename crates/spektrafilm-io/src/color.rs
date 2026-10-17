use anyhow::{Context, Result, bail};
use exr::{meta::attribute::Chromaticities, prelude::Vec2};
use moxcms::{
    ColorProfile, DataColorSpace, Layout, RenderingIntent, ToneReprCurve, TransformOptions,
};
use spektrafilm_math::transfer;

pub(crate) fn apply_icc(
    image: &mut image::Rgb32FImage,
    bytes: &[u8],
) -> Result<(&'static str, bool)> {
    if let Some(tagged) = super::identify_icc(bytes) {
        return Ok(tagged);
    }
    let mut source = ColorProfile::new_from_slice(bytes).context("reading embedded ICC profile")?;
    if source.color_space != DataColorSpace::Rgb {
        bail!(
            "unsupported embedded ICC color space: {:?}; convert the image to RGB first",
            source.color_space
        );
    }
    // Express gamma 1 as identity so scene-linear negatives survive the CMS.
    for curve in [
        &mut source.red_trc,
        &mut source.green_trc,
        &mut source.blue_trc,
    ] {
        let identity = match curve {
            Some(ToneReprCurve::Lut(values)) => values.as_slice() == [256],
            Some(ToneReprCurve::Parametric(values)) => values.as_slice() == [1.],
            None => false,
        };
        if identity {
            *curve = Some(ToneReprCurve::Lut(Vec::new()));
        }
    }
    let mut destination = ColorProfile::new_srgb();
    let linear = Some(ToneReprCurve::Lut(Vec::new()));
    destination.red_trc = linear.clone();
    destination.green_trc = linear.clone();
    destination.blue_trc = linear;
    destination.cicp = None;
    let transform = source
        .create_transform_f32(
            Layout::Rgb,
            &destination,
            Layout::Rgb,
            TransformOptions {
                rendering_intent: RenderingIntent::RelativeColorimetric,
                prefer_fixed_point: false,
                allow_extended_range_rgb_xyz: true,
                ..Default::default()
            },
        )
        .context("unsupported embedded ICC transform")?;
    let mut pixels = vec![0.; image.as_raw().len()];
    transform
        .transform(image.as_raw(), &mut pixels)
        .context("converting embedded ICC profile")?;
    *image = image::Rgb32FImage::from_raw(image.width(), image.height(), pixels)
        .context("invalid ICC output dimensions")?;
    Ok(("sRGB", false))
}

pub(crate) fn exr_chromaticities(name: &str) -> Result<Chromaticities> {
    let space = transfer::get(name).map_err(anyhow::Error::msg)?;
    let xy = |xyz: [f64; 3]| {
        let sum = xyz.iter().sum::<f64>();
        Vec2((xyz[0] / sum) as f32, (xyz[1] / sum) as f32)
    };
    let primary = |c| xy(std::array::from_fn(|row| space.to_xyz[row][c]));
    Ok(Chromaticities {
        red: primary(0),
        green: primary(1),
        blue: primary(2),
        white: xy(space.white),
    })
}

pub(crate) fn exr_color_space(chromaticities: Option<Chromaticities>) -> Result<&'static str> {
    let Some(actual) = chromaticities else {
        return Ok("sRGB");
    };
    let canonical = [
        "sRGB",
        "ProPhoto RGB",
        "Adobe RGB (1998)",
        "ITU-R BT.2020",
        "ACES2065-1",
        "ACEScg",
        "Display P3",
        "DCI-P3",
    ];
    for name in canonical.into_iter().chain(transfer::names()) {
        let expected = exr_chromaticities(name)?;
        let pairs = [
            (actual.red, expected.red),
            (actual.green, expected.green),
            (actual.blue, expected.blue),
            (actual.white, expected.white),
        ];
        if pairs
            .iter()
            .all(|(a, b)| (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5)
        {
            return Ok(name);
        }
    }
    bail!("unsupported EXR chromaticities; convert the image to a named RGB color space first")
}
