//! Native image-style diagnostics with source-equivalent target stimuli.
use crate::{
    lut_export::{self, BakeOptions, LutSpace},
    pipeline::Pipeline,
    profile::load_profile_by_name,
    qa_metrics as m,
};
use base64::Engine;
use spektrafilm_math::{image::ImageBuf, precision::from_f64, transfer};
use std::path::Path;
pub(crate) fn matrix_cat16(source: &str, destination: &str) -> Result<[[f64; 3]; 3], String> {
    let a = transfer::get(source)?;
    let b = transfer::get(destination)?;
    Ok(transfer::matmul(
        b.from_xyz,
        transfer::matmul(
            spektrafilm_math::colorspace::chromatic_adaptation_matrix_cat16_f64(a.white, b.white),
            a.to_xyz,
        ),
    ))
}
fn png_data(png: &image::RgbImage) -> Result<String, String> {
    let mut bytes = std::io::Cursor::new(Vec::new());
    png.write_to(&mut bytes, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes.into_inner()))
}
fn rgb8(linear: [f64; 3]) -> [u8; 3] {
    linear.map(|v| (transfer::Transfer::Srgb.encode(v.clamp(0., 1.)) * 255.).round() as u8)
}
pub struct EdgeInput {
    pub linear: Vec<[f64; 3]>,
    pub oog_fraction_total: f64,
    pub oog_fraction_saturated_row: f64,
}
pub fn edge_input(
    target: &LutSpace,
    input: &LutSpace,
    width: usize,
    height: usize,
) -> Result<EdgeInput, String> {
    let matrix = matrix_cat16(target.primaries, input.primaries)?;
    let mut linear = Vec::with_capacity(width * height);
    let (mut outside, mut middle_out, mut middle_count) = (0usize, 0usize, 0usize);
    let start = height / 2 - height / 16;
    let end = height / 2 + height / 16;
    for y in 0..height {
        let v = y as f64 / height.saturating_sub(1).max(1) as f64;
        let white = (1. - 2. * v).max(0.);
        let sat = 1. - (2. * v - 1.).abs();
        for x in 0..width {
            let t = 6. * x as f64 / width as f64;
            let f = t - t.floor();
            let hue = match t.floor() as usize % 6 {
                0 => [1., f, 0.],
                1 => [1. - f, 1., 0.],
                2 => [0., 1., f],
                3 => [0., 1. - f, 1.],
                4 => [f, 0., 1.],
                _ => [1., 0., 1. - f],
            };
            let encoded = hue.map(|c| white + sat * c);
            let p = m::mul(matrix, target.decode_pixel(encoded));
            let oog = p.iter().any(|v| *v < 0. || *v > 1.);
            outside += usize::from(oog);
            if (start..end).contains(&y) {
                middle_count += 1;
                middle_out += usize::from(oog);
            }
            linear.push(p.map(|v| v as f32 as f64));
        }
    }
    Ok(EdgeInput {
        linear,
        oog_fraction_total: outside as f64 / (width * height) as f64,
        oog_fraction_saturated_row: middle_out as f64 / middle_count.max(1) as f64,
    })
}
pub(crate) fn edge_stress(
    options: &BakeOptions<'_>,
    data: &Path,
    assets: &Path,
) -> Result<serde_json::Value, String> {
    let input = lut_export::space(options.input)?;
    let output = lut_export::space(options.output)?;
    let mut params = options.params.clone();
    params.debug.lut_mode = true;
    params.io.input_color_space = input.primaries.into();
    params.io.output_color_space = output.primaries.into();
    params.io.input_cctf_decoding = false;
    params.io.output_cctf_encoding = false;
    let pipeline = Pipeline::new_with_spectral(
        load_profile_by_name(data, options.film).map_err(|e| e.to_string())?,
        load_profile_by_name(data, options.paper).map_err(|e| e.to_string())?,
        params,
        data,
    )?;
    let display = matrix_cat16(output.primaries, "sRGB")?;
    let mut svg = String::from(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"808\" height=\"980\" viewBox=\"0 0 808 980\"><rect width=\"808\" height=\"980\" fill=\"#101827\"/><g fill=\"#e5e7eb\" font-family=\"sans-serif\"><text x=\"20\" y=\"28\" font-size=\"18\">Wide-gamut edge stress through the spectral runtime</text>",
    );
    let mut summary = serde_json::Map::new();
    for (index, name) in ["Rec.709", "DCI-P3", "Rec.2020"].into_iter().enumerate() {
        let target = lut_export::space(name)?;
        let source = edge_input(target, input, 768, 256)?;
        let image = ImageBuf::from_data(
            768 * 256,
            1,
            source
                .linear
                .iter()
                .flatten()
                .map(|v| from_f64(*v))
                .collect(),
        );
        let rendered = pipeline.process(image, &spektrafilm_gpu::cpu_backend::CpuBackend);
        let mut png = image::RgbImage::new(768, 256);
        for (i, p) in rendered.pixels().enumerate() {
            png.put_pixel(
                (i % 768) as u32,
                (i / 768) as u32,
                image::Rgb(rgb8(m::mul(
                    display,
                    [p[0] as f64, p[1] as f64, p[2] as f64],
                ))),
            );
        }
        png.save(assets.join(format!("gamut-edge-{index}.png")))
            .map_err(|e| e.to_string())?;
        let encoded = png_data(&png)?;
        let title_y = 62 + index * 296;
        let image_y = 75 + index * 296;
        svg.push_str(&format!("<text x=\"20\" y=\"{title_y}\">{name}: input-gamut escape in the saturated band {:.2}%</text><image x=\"20\" y=\"{image_y}\" width=\"768\" height=\"256\" href=\"data:image/png;base64,{encoded}\"/>",source.oog_fraction_saturated_row*100.));
        summary.insert(
            format!("{name}_oog_fraction_saturated_row"),
            serde_json::json!(source.oog_fraction_saturated_row),
        );
        summary.insert(
            format!("{name}_oog_fraction_total"),
            serde_json::json!(source.oog_fraction_total),
        );
    }
    svg.push_str("<text x=\"20\" y=\"963\" font-size=\"12\">White → saturated RGB-cube hue cycle → black. Target → input and output → display use CAT16.</text></g></svg>");
    std::fs::write(assets.join("gamut-edges.svg"), svg).map_err(|e| e.to_string())?;
    Ok(serde_json::Value::Object(summary))
}
pub(crate) fn slices(
    table: &ImageBuf,
    n: usize,
    output: &LutSpace,
    assets: &Path,
) -> Result<serde_json::Value, String> {
    let count = n.min(9);
    let rows = count.div_ceil(3);
    let display = matrix_cat16(output.primaries, "sRGB")?;
    let mut composite =
        image::RgbImage::from_pixel(3 * 256, rows as u32 * 256, image::Rgb([16, 24, 39]));
    let height = 80 + rows * 280;
    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"{height}\" viewBox=\"0 0 900 {height}\"><rect width=\"900\" height=\"{height}\" fill=\"#101827\"/><g fill=\"#e5e7eb\" font-family=\"sans-serif\"><text x=\"35\" y=\"28\" font-size=\"18\">R–G cube slices at varying B, converted to sRGB</text>"
    );
    let mut pins = Vec::new();
    for panel in 0..count {
        let b = panel * (n - 1) / (count - 1).max(1);
        pins.push(b as f64 / (n - 1) as f64);
        let mut png = image::RgbImage::new(n as u32, n as u32);
        for g in 0..n {
            for r in 0..n {
                let i = ((b * n + g) * n + r) * 3;
                let encoded = std::array::from_fn(|c| table.data[i + c] as f64);
                let linear = output.decode_pixel(encoded).map(|v| v / output.output_gain);
                png.put_pixel(
                    r as u32,
                    (n - 1 - g) as u32,
                    image::Rgb(rgb8(m::mul(display, linear))),
                );
            }
        }
        let enlarged =
            image::imageops::resize(&png, 256, 256, image::imageops::FilterType::Triangle);
        image::imageops::replace(
            &mut composite,
            &enlarged,
            (panel % 3 * 256) as i64,
            (panel / 3 * 256) as i64,
        );
        let encoded = png_data(&png)?;
        let x = 45 + panel % 3 * 290;
        let y = 70 + panel / 3 * 280;
        svg.push_str(&format!("<text x=\"{}\" y=\"{}\">B = {:.2}</text><image x=\"{x}\" y=\"{y}\" width=\"230\" height=\"230\" href=\"data:image/png;base64,{encoded}\"/><text x=\"{}\" y=\"{}\" font-size=\"12\">R input →</text><text x=\"{}\" y=\"{}\" font-size=\"12\">G ↑</text>",x+75,y-12,pins[panel],x+70,y+251,x-32,y+120));
    }
    svg.push_str("</g></svg>");
    std::fs::write(assets.join("slices.svg"), svg).map_err(|e| e.to_string())?;
    composite
        .save(assets.join("slices.png"))
        .map_err(|e| e.to_string())?;
    Ok(
        serde_json::json!({"n_slices":count,"cube_resolution":n,"blue_pins":pins,"display":"sRGB","adaptation":"CAT16"}),
    )
}
