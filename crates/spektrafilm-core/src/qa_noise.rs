//! Source-equivalent polar noise propagation and seeded perceptual gradients.
use crate::{
    lut_export::{self, LutSpace},
    lut_quality::trilinear,
    qa_metrics as m, qa_patterns,
};
use spektrafilm_math::{image::ImageBuf, transfer};
use std::path::Path;
struct Point {
    sample: qa_patterns::NoisePolarSample,
    noise: m::Noise,
    input_lab: [f64; 3],
    output_lab: [f64; 3],
    output: [f64; 3],
}
struct Analysis {
    points: Vec<Point>,
    summary: serde_json::Value,
}
fn analyze(
    table: &ImageBuf,
    n: usize,
    input: &LutSpace,
    output: &LutSpace,
    gain: Option<f64>,
) -> Analysis {
    let samples = qa_patterns::noise_polar_samples(input, gain);
    let evaluate = |p| trilinear(table, n, p);
    let points: Vec<_> = samples
        .into_iter()
        .map(|sample| {
            let output_encoded = evaluate(sample.encoded);
            let noise = m::noise(sample.encoded, &evaluate, output);
            let input_lab = m::oklab(m::xyz(sample.encoded, input));
            let output_lab = m::oklab(m::xyz(output_encoded, output));
            Point {
                sample,
                noise,
                input_lab,
                output_lab,
                output: output_encoded,
            }
        })
        .collect();
    let sigma = m::summary(&points.iter().map(|p| p.noise.sigma1).collect::<Vec<_>>());
    let anisotropy = m::summary(
        &points
            .iter()
            .map(|p| p.noise.anisotropy)
            .collect::<Vec<_>>(),
    );
    let mut worst: Option<(f64, f64)> = None;
    for p in &points {
        if p.input_lab[1].hypot(p.input_lab[2]) > 1e-3 {
            let h_in = p.input_lab[2].atan2(p.input_lab[1]).to_degrees();
            let h_out = p.output_lab[2].atan2(p.output_lab[1]).to_degrees();
            let d = ((h_out - h_in + 180.).rem_euclid(360.) - 180.).abs();
            if worst.is_none_or(|(v, _)| d > v) {
                worst = Some((d, h_in));
            }
        }
    }
    let (rotation, hue) = worst.unwrap_or((0., 0.));
    let summary = serde_json::json!({"L_slice":qa_patterns::MIDGRAY_OKLAB_L,"sigma_in_encoded":0.005,
        "n_input_samples":points.len(),"max_sigma1":sigma.max,"p99_sigma1":sigma.p99,"p50_sigma1":sigma.p50,
        "max_anisotropy":anisotropy.max,"p99_anisotropy":anisotropy.p99,"max_hue_rotation_deg":rotation,
        "worst_hue_deg":hue,"ellipse_display_scale":1.5});
    Analysis { points, summary }
}
fn color(value: [f64; 3], space: &LutSpace) -> String {
    let rgb = m::mul(
        transfer::get("sRGB").unwrap().from_xyz,
        m::xyz(value, space),
    )
    .map(|v| (transfer::Transfer::Srgb.encode(v).clamp(0., 1.) * 255.).round() as u8);
    format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
}
fn outline(lab: &[[f64; 3]], offset: f64, extent: f64) -> String {
    lab.iter()
        .chain(lab.first())
        .map(|p| {
            format!(
                "{:.2},{:.2}",
                offset + 300. + p[1] / extent * 230.,
                310. - p[2] / extent * 230.
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}
pub(crate) fn sensitivity(
    table: &ImageBuf,
    n: usize,
    input: &LutSpace,
    output: &LutSpace,
    gain: Option<f64>,
    assets: &Path,
) -> Result<serde_json::Value, String> {
    use base64::Engine;
    let analysis = analyze(table, n, input, output, gain);
    let in_outline = qa_patterns::gamut_slice(input, qa_patterns::MIDGRAY_OKLAB_L, 181);
    let out_outline = qa_patterns::gamut_slice(output, qa_patterns::MIDGRAY_OKLAB_L, 181);
    let extent = (in_outline
        .iter()
        .chain(&out_outline)
        .map(|p| p[1].abs().max(p[2].abs()))
        .fold(0., f64::max)
        * 1.15)
        .max(0.3);
    let mut heat = vec![None; 96 * 96];
    for y in 0..96 {
        for x in 0..96 {
            let a = -extent + 2. * extent * x as f64 / 95.;
            let b = -extent + 2. * extent * y as f64 / 95.;
            let linear =
                qa_patterns::input_linear_from_oklab([qa_patterns::MIDGRAY_OKLAB_L, a, b], input);
            if linear.iter().all(|v| *v >= 0. && *v <= 1.) {
                let encoded = input
                    .encode_pixel(linear.map(|v| v.clamp(0., 1.) / gain.unwrap_or(input.auto_gain)))
                    .map(|v| v.clamp(0.006, 0.994));
                heat[y * 96 + x] =
                    Some(m::noise(encoded, &|p| trilinear(table, n, p), output).sigma1);
            }
        }
    }
    let max = heat.iter().flatten().copied().fold(0., f64::max).max(1e-9);
    let mut png = image::RgbImage::from_pixel(96, 96, image::Rgb([16, 24, 39]));
    for y in 0..96 {
        for x in 0..96 {
            if let Some(value) = heat[y * 96 + x] {
                let t = (value / max).clamp(0., 1.);
                let rgb = [
                    (0.15 + 0.8 * t) * 255.,
                    (0.07 + 0.85 * t.sqrt()) * 255.,
                    (0.45 + 0.4 * (1. - 2. * t).max(0.) - 0.3 * t) * 255.,
                ]
                .map(|v| v.clamp(0., 255.) as u8);
                png.put_pixel(x as u32, 95 - y as u32, image::Rgb(rgb));
            }
        }
    }
    png.save(assets.join("noise-gain-heatmap.png"))
        .map_err(|e| e.to_string())?;
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(std::fs::read(assets.join("noise-gain-heatmap.png")).map_err(|e| e.to_string())?);
    let mut s = String::from(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1800\" height=\"620\" viewBox=\"0 0 1800 620\"><rect width=\"1800\" height=\"620\" fill=\"#101827\"/><defs><clipPath id=\"ellipse-clip\"><rect x=\"50\" y=\"80\" width=\"500\" height=\"460\"/></clipPath></defs><g fill=\"#e5e7eb\" font-family=\"sans-serif\"><text x=\"35\" y=\"30\" font-size=\"20\">Noise propagation in OkLab</text><text x=\"35\" y=\"62\">Covariance ellipses, 2σ × 1.5; input → output</text><text x=\"635\" y=\"62\">Worst-direction gain σ₁(J), at 18%-gray lightness</text><text x=\"1235\" y=\"62\">Middle chroma ring: noise versus input hue</text>",
    );
    s.push_str(&format!("<image x=\"670\" y=\"80\" width=\"460\" height=\"460\" href=\"data:image/png;base64,{encoded}\"/><text x=\"670\" y=\"570\">a* / b*: ±{extent:.3}; gain scale 0–{max:.3}</text>"));
    for offset in [0., 600.] {
        for step in -2..=2 {
            let x = offset + 300. + step as f64 * 115.;
            let y = 310. - step as f64 * 115.;
            s.push_str(&format!(
                "<path d=\"M {x} 80 V 540 M {} {y} H {}\" stroke=\"#374151\" fill=\"none\"/>",
                offset + 70.,
                offset + 530.
            ));
        }
        s.push_str(&format!("<polyline points=\"{}\" fill=\"none\" stroke=\"#fbbf24\"/><polyline points=\"{}\" fill=\"none\" stroke=\"#e5e7eb\" stroke-dasharray=\"4 3\"/>",outline(&in_outline,offset,extent),outline(&out_outline,offset,extent)));
    }
    s.push_str("<g clip-path=\"url(#ellipse-clip)\">");
    for point in &analysis.points {
        let (a, b) = (point.output_lab[1], point.output_lab[2]);
        let x = 300. + a / extent * 230.;
        let y = 310. - b / extent * 230.;
        let covariance = point.noise.covariance;
        let (aa, ab, bb) = (covariance[1][1], covariance[1][2], covariance[2][2]);
        let discriminant = ((aa - bb).powi(2) + 4. * ab * ab).sqrt();
        let rx = 3. * ((aa + bb + discriminant) * 0.5).max(0.).sqrt() / extent * 230.;
        let ry = 3. * ((aa + bb - discriminant) * 0.5).max(0.).sqrt() / extent * 230.;
        let angle = -0.5 * (2. * ab).atan2(aa - bb).to_degrees();
        let ix = 300. + point.input_lab[1] / extent * 230.;
        let iy = 310. - point.input_lab[2] / extent * 230.;
        s.push_str(&format!("<path d=\"M {ix} {iy} L {x} {y}\" stroke=\"#94a3b8\" stroke-opacity=\".5\"/><ellipse cx=\"{x}\" cy=\"{y}\" rx=\"{}\" ry=\"{}\" transform=\"rotate({angle} {x} {y})\" fill=\"{}\" fill-opacity=\".7\" stroke=\"#e5e7eb\" stroke-width=\".5\"/>",rx.max(0.5),ry.max(0.5),color(point.output,output)));
    }
    s.push_str("</g>");
    let ring: Vec<_> = analysis
        .points
        .iter()
        .filter(|p| p.sample.ring == 1)
        .collect();
    let radial = ring
        .iter()
        .map(|p| p.noise.sigma_l.max(p.noise.sigma_ab))
        .fold(0., f64::max)
        .max(1e-9)
        * 1.1;
    for k in 1..=4 {
        let r = k as f64 * 50.;
        s.push_str(&format!("<circle cx=\"1500\" cy=\"310\" r=\"{r}\" fill=\"none\" stroke=\"#374151\"/><text x=\"{}\" y=\"305\" font-size=\"11\">{:.4}</text>",1505.+r,radial*k as f64/4.));
    }
    for (ch, color) in [(0, "#93c5fd"), (1, "#fbbf24")] {
        let points = ring
            .iter()
            .copied()
            .chain(ring.first().copied())
            .map(|p| {
                let angle = p.sample.hue_deg.to_radians();
                let radius = if ch == 0 {
                    p.noise.sigma_l
                } else {
                    p.noise.sigma_ab
                } / radial
                    * 200.;
                format!(
                    "{:.2},{:.2}",
                    1500. + radius * angle.cos(),
                    310. - radius * angle.sin()
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        s.push_str(&format!(
            "<polyline points=\"{points}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2\"/>"
        ));
    }
    s.push_str("<text x=\"1715\" y=\"315\">0°</text><text x=\"1490\" y=\"95\">90°</text><text x=\"1250\" y=\"315\">180°</text><text x=\"1490\" y=\"535\">270°</text><text x=\"1310\" y=\"570\" fill=\"#93c5fd\">Luminance σL</text><text x=\"1510\" y=\"570\" fill=\"#fbbf24\">Chroma σab</text><text x=\"35\" y=\"600\" font-size=\"13\">Input noise σ = 0.005 encoded RGB. Solid yellow: input gamut slice; dashed white: output gamut slice.</text></g></svg>");
    std::fs::write(assets.join("noise.svg"), s).map_err(|e| e.to_string())?;
    Ok(analysis.summary)
}

pub(crate) fn gradient(
    input: &lut_export::LutSpace,
    output: &lut_export::LutSpace,
    gain: Option<f64>,
    evaluate: &dyn Fn([f64; 3]) -> [f64; 3],
    assets: &Path,
) -> Result<serde_json::Value, String> {
    use base64::Engine;
    let gradient = crate::qa_patterns::cached_noise_gradient(input, gain);
    let noise = crate::qa_rng::normal_values(512 * 512 * 3);
    let display_matrix = transfer::get("sRGB").unwrap().from_xyz;
    let display = |p: [f64; 3]| {
        m::mul(display_matrix, m::xyz(p, output))
            .map(|v| (transfer::Transfer::Srgb.encode(v).clamp(0., 1.) * 255.).round() as u8)
    };
    let mut png = image::RgbImage::new(1024, 512);
    let mut delta = Vec::with_capacity(512 * 512 * 3);
    for (i, &clean) in gradient.encoded.iter().enumerate() {
        let noisy = std::array::from_fn(|c| (clean[c] + 0.02 * noise[i * 3 + c]).clamp(0., 1.));
        let clean_out = evaluate(clean);
        let noisy_out = evaluate(noisy);
        delta.extend((0..3).map(|c| (clean_out[c] - noisy_out[c]).abs()));
        let (x, y) = ((i % 512) as u32, (i / 512) as u32);
        png.put_pixel(x, y, image::Rgb(display(clean_out)));
        png.put_pixel(x + 512, y, image::Rgb(display(noisy_out)));
    }
    png.save(assets.join("noise-gradient.png"))
        .map_err(|e| e.to_string())?;
    let encoded = base64::engine::general_purpose::STANDARD
        .encode(std::fs::read(assets.join("noise-gradient.png")).map_err(|e| e.to_string())?);
    let svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1064\" height=\"610\" viewBox=\"0 0 1064 610\"><rect width=\"1064\" height=\"610\" fill=\"#101827\"/><g fill=\"#e5e7eb\" font-family=\"sans-serif\"><text x=\"20\" y=\"26\" font-size=\"18\">Continuous OkLab noise gradient</text><text x=\"20\" y=\"50\">Clean LUT output</text><text x=\"532\" y=\"50\">Noisy input → LUT (σ = 0.02 encoded RGB)</text><image x=\"20\" y=\"60\" width=\"1024\" height=\"512\" href=\"data:image/png;base64,{encoded}\"/><text x=\"20\" y=\"598\" font-size=\"13\">Hue cycles horizontally; black → 18%-gray lightness → white vertically. Fixed NumPy seed 0.</text></g></svg>"
    );
    std::fs::write(assets.join("noise-gradient.svg"), svg).map_err(|e| e.to_string())?;
    let stats = m::summary(&delta);
    Ok(
        serde_json::json!({"image_side":512,"sigma_in_encoded":0.02,"mid_L":crate::qa_patterns::MIDGRAY_OKLAB_L,
        "rim_scale":0.96,"input_in_gamut_fraction":gradient.in_gamut_fraction,"mean_abs_output_delta":stats.mean,"p99_abs_output_delta":stats.p99}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn polar_samples_and_complete_noise_summaries_match_python() {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/noise_polar.json")).unwrap();
        let n = f["n"].as_u64().unwrap() as usize;
        let table = ImageBuf::from_data(
            n as u32,
            (n * n) as u32,
            f["table"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| spektrafilm_math::precision::from_f64(v.as_f64().unwrap()))
                .collect(),
        );
        let output = lut_export::space("sRGB").unwrap();
        for case in f["cases"].as_array().unwrap() {
            let name = case["space"].as_str().unwrap();
            let input = lut_export::space(name).unwrap();
            let analysis = analyze(&table, n, input, output, None);
            assert_eq!(
                analysis.points.len(),
                case["samples"]["input_encoded"].as_array().unwrap().len()
            );
            for (i, p) in analysis.points.iter().enumerate() {
                for c in 0..3 {
                    assert!(
                        (p.sample.encoded[c]
                            - case["samples"]["input_encoded"][i][c].as_f64().unwrap())
                        .abs()
                            < 1e-10
                    );
                }
            }
            for (key, expected) in case["summary"].as_object().unwrap() {
                let expected = expected.as_f64().unwrap();
                let actual = analysis.summary[key].as_f64().unwrap();
                let difference = if key == "worst_hue_deg" {
                    ((actual - expected + 180.).rem_euclid(360.) - 180.).abs()
                } else {
                    (actual - expected).abs()
                };
                assert!(
                    difference < 2e-5 * (1. + expected.abs()),
                    "{name} {key}: {actual} versus {expected}"
                );
            }
        }
    }
}
