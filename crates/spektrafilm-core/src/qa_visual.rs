//! Bounded raster plots with vector axes; all samples contribute to the raster.
use crate::{lut_export::LutSpace, qa_metrics as m};
use base64::Engine;
use spektrafilm_math::transfer;
use std::path::Path;
pub struct Cloud {
    width: usize,
    height: usize,
    bounds: [f64; 4],
    pixels: Vec<[f32; 3]>,
}
impl Cloud {
    pub fn new(width: usize, height: usize, bounds: [f64; 4]) -> Self {
        Self {
            width,
            height,
            bounds,
            pixels: vec![[16. / 255., 24. / 255., 39. / 255.]; width * height],
        }
    }
    pub fn point(&mut self, p: [f64; 2], color: [f64; 3], opacity: f64) {
        if !p.iter().all(|v| v.is_finite()) {
            return;
        }
        let [xmin, xmax, ymin, ymax] = self.bounds;
        let x = (p[0] - xmin) / (xmax - xmin) * (self.width - 1) as f64;
        let y = (ymax - p[1]) / (ymax - ymin) * (self.height - 1) as f64;
        let (ix, iy) = (x.floor() as isize, y.floor() as isize);
        let (tx, ty) = (x - x.floor(), y - y.floor());
        for dy in 0..2 {
            for dx in 0..2 {
                let (xx, yy) = (ix + dx, iy + dy);
                if xx < 0 || yy < 0 || xx >= self.width as isize || yy >= self.height as isize {
                    continue;
                }
                let alpha = (opacity
                    * if dx == 0 { 1. - tx } else { tx }
                    * if dy == 0 { 1. - ty } else { ty }) as f32;
                let dst = &mut self.pixels[yy as usize * self.width + xx as usize];
                for c in 0..3 {
                    dst[c] = dst[c] * (1. - alpha) + color[c].clamp(0., 1.) as f32 * alpha;
                }
            }
        }
    }
    pub(crate) fn image(&self) -> image::RgbImage {
        let bytes: Vec<_> = self
            .pixels
            .iter()
            .flatten()
            .map(|v| (v.clamp(0., 1.) * 255.).round() as u8)
            .collect();
        image::RgbImage::from_raw(self.width as u32, self.height as u32, bytes).unwrap()
    }
}
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
fn png_encoded(png: &image::RgbImage) -> Result<String, String> {
    let mut bytes = std::io::Cursor::new(Vec::new());
    png.write_to(&mut bytes, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes.into_inner()))
}
pub fn image_plot(
    png: &image::RgbImage,
    title: &str,
    labels: [&str; 2],
    bounds: [f64; 4],
    caption: &str,
) -> Result<String, String> {
    let (w, h) = (png.width() as f64, png.height() as f64);
    let [xmin, xmax, ymin, ymax] = bounds;
    let encoded = png_encoded(png)?;
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\"><rect width=\"100%\" height=\"100%\" fill=\"#101827\"/><g fill=\"#e5e7eb\" font-family=\"sans-serif\"><text x=\"65\" y=\"28\" font-size=\"16\">{}</text><image x=\"65\" y=\"60\" width=\"{w}\" height=\"{h}\" href=\"data:image/png;base64,{encoded}\"/>",
        w + 130.,
        h + 150.,
        w + 130.,
        h + 150.,
        escape(title)
    );
    for i in 0..=4 {
        let x = 65. + w * i as f64 / 4.;
        let y = 60. + h * (1. - i as f64 / 4.);
        svg.push_str(&format!("<path d=\"M {x} 60 V {} M 65 {y} H {}\" stroke=\"#cbd5e1\" stroke-opacity=\".12\" fill=\"none\"/><text x=\"{x}\" y=\"{}\" text-anchor=\"middle\" font-size=\"11\">{:.3}</text><text x=\"56\" y=\"{}\" text-anchor=\"end\" font-size=\"11\">{:.3}</text>",60.+h,65.+w,79.+h,xmin+(xmax-xmin)*i as f64/4.,y+4.,ymin+(ymax-ymin)*i as f64/4.));
    }
    svg.push_str(&format!("<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"13\">{}</text><text x=\"65\" y=\"48\" font-size=\"12\">{}</text><text x=\"65\" y=\"{}\" font-size=\"11\">{}</text></g></svg>",65.+w/2.,105.+h,escape(labels[0]),escape(labels[1]),132.+h,escape(caption)));
    Ok(svg)
}
pub fn combine(
    panels: &[String],
    columns: usize,
    width: usize,
    height: usize,
    path: &Path,
) -> Result<(), String> {
    let rows = panels.len().div_ceil(columns);
    let mut s = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">",
        columns * width,
        rows * height,
        columns * width,
        rows * height
    );
    for (i, panel) in panels.iter().enumerate() {
        let inner = &panel[panel.find('>').ok_or("invalid SVG panel")? + 1
            ..panel.rfind("</svg>").ok_or("invalid SVG panel")?];
        s.push_str(&format!(
            "<g transform=\"translate({},{})\">{inner}</g>",
            i % columns * width,
            i / columns * height
        ));
    }
    s.push_str("</svg>");
    std::fs::write(path, s).map_err(|e| e.to_string())
}
fn heat_color(t: f64) -> [u8; 3] {
    let t = t.clamp(0., 1.);
    [
        (0.1 + 0.85 * t) * 255.,
        (0.05 + 0.9 * t.sqrt()) * 255.,
        (0.5 + 0.25 * (1. - 2. * t).max(0.) - 0.4 * t) * 255.,
    ]
    .map(|v| v.clamp(0., 255.) as u8)
}
pub fn off_grid(
    samples: &[[f64; 3]],
    tri: &[f64],
    tet: &[f64],
    assets: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(assets).map_err(|e| e.to_string())?;
    let max = tri.iter().chain(tet).copied().fold(0., f64::max).max(1e-9);
    let mut panels = Vec::new();
    for (a, b, label) in [(0, 1, "R / G"), (1, 2, "G / B"), (0, 2, "R / B")] {
        let mut bins = vec![0f64; 128 * 128];
        for (i, p) in samples.iter().enumerate() {
            let x = (p[a].clamp(0., 1.) * 127.) as usize;
            let y = (p[b].clamp(0., 1.) * 127.) as usize;
            bins[y * 128 + x] = bins[y * 128 + x].max(tri[i].max(tet[i]));
        }
        let mut png = image::RgbImage::new(128, 128);
        for y in 0..128 {
            for x in 0..128 {
                png.put_pixel(
                    x,
                    127 - y,
                    image::Rgb(heat_color(bins[y as usize * 128 + x as usize] / max)),
                );
            }
        }
        let png = image::imageops::resize(&png, 384, 384, image::imageops::FilterType::Nearest);
        panels.push(image_plot(
            &png,
            &format!("Off-grid ΔE2000: {label}"),
            [
                ["Input R", "Input G", "Input B"][a],
                ["Input R", "Input G", "Input B"][b],
            ],
            [0., 1., 0., 1.],
            &format!(
                "{} samples; bin maximum of both interpolators; color scale 0–{max:.3}",
                samples.len()
            ),
        )?);
    }
    combine(&panels, 3, 514, 534, &assets.join("off-grid.svg"))
}
pub fn conditioning(
    values: &[f64],
    side: usize,
    maximum: f64,
    assets: &Path,
) -> Result<(), String> {
    let max = maximum.max(1e-9);
    let mut panels = Vec::new();
    for (axis, label) in ["R", "G", "B"].into_iter().enumerate() {
        let mut png = image::RgbImage::from_pixel(
            side.max(1) as u32,
            side.max(1) as u32,
            image::Rgb([16, 24, 39]),
        );
        if side > 0 {
            for y in 0..side {
                for x in 0..side {
                    let free: Vec<_> = (0..3).filter(|v| *v != axis).collect();
                    let mut p = [side / 2; 3];
                    p[free[0]] = x;
                    p[free[1]] = y;
                    let value = values[(p[2] * side + p[1]) * side + p[0]];
                    png.put_pixel(
                        x as u32,
                        (side - 1 - y) as u32,
                        image::Rgb(heat_color(value / max)),
                    );
                }
            }
        }
        let png = image::imageops::resize(&png, 384, 384, image::imageops::FilterType::Nearest);
        panels.push(image_plot(
            &png,
            &format!("Jacobian log10(condition), {label} pinned"),
            match axis {
                0 => ["Input G", "Input B"],
                1 => ["Input R", "Input B"],
                _ => ["Input R", "Input G"],
            },
            [0., 1., 0., 1.],
            &format!("Interior cells; color scale 0–{max:.3}; brighter = near-singular"),
        )?);
    }
    combine(&panels, 3, 514, 534, &assets.join("conditioning.svg"))
}
pub fn variation(tv: [f64; 3], fft: [f64; 3], assets: &Path) -> Result<(), String> {
    let mut svg = String::from(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"470\" viewBox=\"0 0 900 470\"><rect width=\"900\" height=\"470\" fill=\"#101827\"/><g fill=\"#e5e7eb\" font-family=\"sans-serif\"><text x=\"45\" y=\"30\" font-size=\"18\">LUT variation and high-frequency structure</text>",
    );
    for (panel, values, title) in [
        (0, tv, "Mean absolute neighbor RGB step"),
        (1, fft, "Axial FFT high-band energy fraction"),
    ] {
        let x = 45 + panel * 440;
        let max = values.into_iter().fold(0., f64::max).max(1e-12);
        svg.push_str(&format!(
            "<text x=\"{x}\" y=\"65\" font-size=\"14\">{title}</text>"
        ));
        for c in 0..3 {
            let y = 115 + c * 80;
            let width = values[c] / max * 280.;
            svg.push_str(&format!("<text x=\"{x}\" y=\"{}\">{}</text><rect x=\"{}\" y=\"{y}\" width=\"{width}\" height=\"28\" fill=\"{}\"/><text x=\"{}\" y=\"{}\" font-size=\"12\">{:.6}</text>",y+18,["R axis","G axis","B axis"][c],x+60,["#fb7185","#6ee7b7","#93c5fd"][c],x+60,y+48,values[c]));
        }
    }
    svg.push_str("<text x=\"45\" y=\"445\" font-size=\"13\">Informational smoothness metrics; high-band energy reveals structure beyond the low-frequency tone shape.</text></g></svg>");
    std::fs::write(assets.join("variation.svg"), svg).map_err(|e| e.to_string())
}
pub fn hue_twist(before: &[[f64; 3]], after: &[[f64; 3]], assets: &Path) -> Result<(), String> {
    let mut cloud = Cloud::new(720, 500, [0., 0.6, 0., 180.]);
    for (a, b) in before.iter().zip(after) {
        let chroma = a[1].hypot(a[2]);
        if chroma <= 1e-3 {
            continue;
        }
        let h = a[2].atan2(a[1]);
        let out = b[2].atan2(b[1]);
        let d = ((out - h + std::f64::consts::PI).rem_euclid(2. * std::f64::consts::PI)
            - std::f64::consts::PI)
            .abs()
            .to_degrees();
        let color = crate::qa_patterns::hue_rgb(h / (2. * std::f64::consts::PI));
        cloud.point([chroma, d], color, 0.35);
    }
    let svg = image_plot(
        &cloud.image(),
        "Hue rotation versus input chroma",
        ["Input OkLab chroma", "Absolute hue rotation (degrees)"],
        [0., 0.6, 0., 180.],
        "Every chromatic in-locus cube sample; colored by input hue. Stock-dependent, informational.",
    )?;
    std::fs::write(assets.join("hue-twist.svg"), svg).map_err(|e| e.to_string())
}

#[path = "qa_primaries.rs"]
mod primary_data;
pub(crate) fn primaries(space: &LutSpace) -> [[f64; 2]; 3] {
    primary_data::PRIMARIES
        .iter()
        .find(|(name, _)| *name == space.primaries)
        .map(|(_, xy)| *xy)
        .unwrap_or_else(|| {
            let matrix = transfer::get(space.primaries).unwrap().to_xyz;
            std::array::from_fn(|c| {
                let sum = matrix.iter().map(|row| row[c]).sum::<f64>();
                [matrix[0][c] / sum, matrix[1][c] / sum]
            })
        })
}
/// Crossing test used by the source plotting path, including its edge convention.
pub(crate) fn contains(point: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let mut inside = false;
    for edge in polygon.windows(2) {
        let (a, b) = (edge[0], edge[1]);
        let high_a = a[1] >= point[1];
        let high_b = b[1] >= point[1];
        if high_a != high_b
            && (((b[1] - point[1]) * (a[0] - b[0]) >= (b[0] - point[0]) * (a[1] - b[1])) == high_b)
        {
            inside = !inside;
        }
    }
    inside
}
pub(crate) fn line_svg(
    points: &[[f64; 2]],
    color: &str,
    dashed: bool,
    bounds: [f64; 4],
    width: f64,
    height: f64,
) -> String {
    let [xmin, xmax, ymin, ymax] = bounds;
    let points = points
        .iter()
        .map(|p| {
            format!(
                "{:.3},{:.3}",
                65. + (p[0] - xmin) / (xmax - xmin) * width,
                60. + (ymax - p[1]) / (ymax - ymin) * height
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "<polyline points=\"{points}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"1.3\" {} />",
        if dashed {
            "stroke-dasharray=\"5 3\""
        } else {
            ""
        }
    )
}
pub(crate) fn xy_cloud_svg(
    cloud: &Cloud,
    title: &str,
    caption: &str,
    lines: &[(&[[f64; 2]], &str, bool)],
) -> Result<String, String> {
    let mut svg = image_plot(
        &cloud.image(),
        title,
        ["CIE x", "CIE y"],
        cloud.bounds,
        caption,
    )?;
    let extras = lines
        .iter()
        .map(|(p, c, d)| {
            line_svg(
                p,
                c,
                *d,
                cloud.bounds,
                cloud.width as f64,
                cloud.height as f64,
            )
        })
        .collect::<String>();
    let closing = svg.rfind("</g></svg>").ok_or("invalid cloud SVG")?;
    svg.insert_str(closing, &extras);
    Ok(svg)
}
pub(crate) fn spectral_cloud(
    cube: &[[f64; 3]],
    n: usize,
    output: &LutSpace,
    assets: &Path,
) -> Result<serde_json::Value, String> {
    let locus = crate::input_gamut::spectral_locus_xy();
    let primary = primaries(output);
    let triangle = [primary[0], primary[1], primary[2], primary[0]];
    let mut cloud = Cloud::new(720, 800, [-0.05, 0.85, -0.05, 0.95]);
    let (mut valid, mut inside, mut near, mut outside_locus) = (0usize, 0usize, 0usize, 0usize);
    for &encoded in cube {
        let xyz = m::xyz(encoded, output);
        let sum = xyz.iter().sum::<f64>();
        if xyz[1] <= 1e-4 || sum == 0. {
            continue;
        }
        let xy = [xyz[0] / sum, xyz[1] / sum];
        if !xy.iter().all(|v| v.is_finite()) {
            continue;
        }
        valid += 1;
        inside += usize::from(contains(xy, &triangle));
        outside_locus += usize::from(!contains(xy, &locus));
        near += usize::from(
            locus
                .iter()
                .any(|p| (p[0] - xy[0]).powi(2) + (p[1] - xy[1]).powi(2) < 0.02f64.powi(2)),
        );
        // Encoded RGB colors follow the source chromaticity-map convention.
        cloud.point(xy, encoded, 0.65);
    }
    let caption = format!(
        "{n}³ cells; {valid} valid at Y > 0.0001; {:.1}% inside output primaries",
        inside as f64 / valid.max(1) as f64 * 100.
    );
    let svg = xy_cloud_svg(
        &cloud,
        "LUT chromaticity map: every valid cube cell",
        &caption,
        &[(&locus, "#94a3b8", false), (&triangle, "#e5e7eb", false)],
    )?;
    std::fs::write(assets.join("spectral-envelope.svg"), svg).map_err(|e| e.to_string())?;
    Ok(
        serde_json::json!({"cube_resolution":n,"cube_cells":cube.len(),"valid_cells":valid,
        "inside_output_gamut_fraction":inside as f64/valid.max(1) as f64,"near_locus_fraction":near as f64/valid.max(1) as f64,
        "outside_spectral_locus":outside_locus}),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn full_cube_envelope_matches_original_python_measurements() {
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/qa_envelope.json")).unwrap();
        let path = std::env::temp_dir().join(format!(
            "spektrafilm-envelope-fixture-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        for case in cases.as_array().unwrap() {
            let table: Vec<[f64; 3]> = serde_json::from_value(case["table"].clone()).unwrap();
            let space = crate::lut_export::space(case["output"].as_str().unwrap()).unwrap();
            let actual = super::spectral_cloud(&table, 5, space, &path).unwrap();
            for (key, expected) in case["summary"].as_object().unwrap() {
                assert!(
                    (actual[key].as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-12,
                    "{} {key}: {} versus {expected}",
                    space.name,
                    actual[key]
                );
            }
        }
        std::fs::remove_dir_all(path).unwrap();
    }
}
