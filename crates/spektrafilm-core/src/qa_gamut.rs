//! Source-equivalent gamut probes, with portable native figures.
use crate::{
    gamut_compression::OutputGamutCompress,
    input_gamut::InputGamutCompress,
    lut_export::{self, BakeOptions, LutSpace},
    pipeline::Pipeline,
    profile::load_profile_by_name,
    qa_metrics as m,
    qa_visual::{self as v, Cloud},
};
use spektrafilm_math::{image::ImageBuf, precision::from_f64, transfer};
use std::path::Path;

// QA uses the standard observer's tabulated white, as does colour.CCS_ILLUMINANTS.
// The physical spectral runtime retains its integrated illuminant white.
fn white(name: &str) -> Result<[f64; 2], String> {
    Ok(match name {
        "D50" => [0.3457, 0.3585],
        "D55" => [0.33243, 0.34744],
        "D65" => [0.3127, 0.3290],
        "D75" => [0.29903, 0.31488],
        "D60" => [0.32161670970526801, 0.33761991655081702],
        "A" => [0.44758, 0.40745],
        "E" => [1. / 3., 1. / 3.],
        _ => {
            let (x, y) = spektrafilm_math::spectral::illuminant_to_xy(
                spektrafilm_math::illuminants::get_f32(name)?,
            );
            [x, y]
        }
    })
}
fn projection(
    cube: &[[f64; 3]],
    input: &LutSpace,
    reference: [f64; 2],
) -> Result<Vec<([f64; 2], f64)>, String> {
    let cs = transfer::get(input.primaries)?;
    let dst = [
        reference[0] / reference[1],
        1.,
        (1. - reference[0] - reference[1]) / reference[1],
    ];
    let matrix = transfer::matmul(
        spektrafilm_math::colorspace::chromatic_adaptation_matrix_cat16_f64(cs.white, dst),
        cs.to_xyz,
    );
    Ok(cube
        .iter()
        .map(|p| {
            let xyz = m::mul(matrix, input.decode_pixel(*p));
            let b = xyz.iter().sum::<f64>();
            let safe = if b > 1e-12 { b } else { 1. };
            ([xyz[0] / safe, xyz[1] / safe], b)
        })
        .collect())
}
fn hsv(h: f64) -> [f64; 3] {
    let t = h.rem_euclid(1.) * 6.;
    let f = t - t.floor();
    match t.floor() as usize {
        0 => [1., f, 0.],
        1 => [1. - f, 1., 0.],
        2 => [0., 1., f],
        3 => [0., 1. - f, 1.],
        4 => [f, 0., 1.],
        _ => [1., 0., 1. - f],
    }
}
fn triangle(space: &LutSpace) -> [[f64; 2]; 4] {
    let p = v::primaries(space);
    [p[0], p[1], p[2], p[0]]
}
fn marker(white: [f64; 2]) -> [[f64; 2]; 5] {
    let [x, y] = white;
    [
        [x - 0.008, y],
        [x, y + 0.008],
        [x + 0.008, y],
        [x, y - 0.008],
        [x - 0.008, y],
    ]
}
fn ring(
    reference: [f64; 2],
    compressor: &InputGamutCompress,
) -> (Vec<[f64; 2]>, Vec<[f64; 2]>, m::Summary) {
    let before: Vec<_> = (0..720)
        .map(|i| {
            let a = i as f64 * std::f64::consts::TAU / 720.;
            [reference[0] + 0.3 * a.cos(), reference[1] + 0.3 * a.sin()]
        })
        .collect();
    let after: Vec<_> = before
        .iter()
        .map(|p| compressor.compress_xy(*p, reference))
        .collect();
    let steps: Vec<_> = after
        .windows(2)
        .map(|p| (p[1][0] - p[0][0]).hypot(p[1][1] - p[0][1]))
        .collect();
    (before, after, m::summary(&steps))
}
pub(crate) fn input(
    cube: &[[f64; 3]],
    input: &LutSpace,
    spec: &crate::params::InputGamutCompressParams,
    reference_name: &str,
    assets: &Path,
) -> Result<(serde_json::Value, serde_json::Value), String> {
    let reference = white(reference_name)?;
    let locus = crate::input_gamut::spectral_locus_xy();
    let tri = triangle(input);
    let mut compressor =
        InputGamutCompress::build(if spec.active { &spec.algorithm } else { "off" }, spec.knee)?;
    compressor.set_boundary(
        &spec.boundary,
        spec.hull_detail as f64,
        [reference[0], reference[1]],
    )?;
    let points = projection(cube, input, reference)?;
    let mut cloud = Cloud::new(720, 800, [-0.05, 0.85, -0.05, 0.95]);
    let (mut valid, mut oog, mut bright) = (0usize, 0usize, 0usize);
    let mut arrows = Vec::new();
    let mut moves = Vec::new();
    for &(p, b) in &points {
        if b <= 1e-4 {
            continue;
        }
        valid += 1;
        let inside = v::contains(p, &locus);
        if !inside {
            oog += 1;
            bright += usize::from(b > 1e-2);
        }
        cloud.point(
            p,
            if inside {
                [0.4, 0.8, 0.6]
            } else {
                [1., 0.4, 0.4]
            },
            if inside { 0.25 } else { 0.45 },
        );
    }
    for &(p, b) in &points {
        if b <= 1e-2 {
            continue;
        }
        let dest = compressor.compress_xy(p, reference);
        let d = (dest[0] - p[0]).hypot(dest[1] - p[1]);
        let outside = !v::contains(p, &locus);
        if spec.active && (outside || d > 1e-3) {
            cloud.point(dest, [0.4, 0.8, 1.], 0.85);
            arrows.push((p, dest));
        }
        if outside {
            moves.push(d);
        }
    }
    let mut svg = v::xy_cloud_svg(
        &cloud,
        "Input gamut: original and compressed chromaticities",
        &format!(
            "{}; {} / {valid} outside locus; {bright} bright; red = original, cyan = compressed",
            spec.algorithm, oog
        ),
        &[
            (&locus, "#cbd5e1", false),
            (&tri, "#ffee66", false),
            (&marker(reference), "#ffffff", false),
        ],
    )?;
    // Every sample contributes to the cloud; displacement arrows are bounded for readability.
    let mut lines = String::new();
    for &(a, b) in arrows.iter().step_by(arrows.len().div_ceil(800).max(1)) {
        lines.push_str(&v::line_svg(
            &[a, b],
            "#fb923c",
            false,
            [-0.05, 0.85, -0.05, 0.95],
            720.,
            800.,
        ));
    }
    let end = svg.rfind("</g></svg>").unwrap();
    svg.insert_str(end, &lines);
    std::fs::write(assets.join("input-gamut.svg"), svg).map_err(|e| e.to_string())?;
    let preview = serde_json::json!({"active":spec.active,"algorithm":spec.algorithm,"knee_threshold":spec.knee[0],"knee_limit":spec.knee[1],"knee_power":spec.knee[2],
        "oog_fraction":oog as f64/valid.max(1) as f64,"n_oog_samples":oog,"n_oog_bright":bright,"reference_illuminant":reference_name,"bright_displacement":m::summary(&moves)});
    let (before, after, steps) = ring(reference, &compressor);
    let mut cloud = Cloud::new(720, 800, [-0.05, 0.85, -0.05, 0.95]);
    for (i, p) in after.iter().enumerate() {
        // Slightly thicken samples without SVG marker overhead.
        let color = hsv(i as f64 / 720.);
        for d in [
            [0., 0.],
            [0.0008, 0.],
            [0., 0.0008],
            [-0.0008, 0.],
            [0., -0.0008],
        ] {
            cloud.point([p[0] + d[0], p[1] + d[1]], color, 0.95);
        }
    }
    let svg = v::xy_cloud_svg(
        &cloud,
        "Circumferential compression: color tracks input angle",
        &format!(
            "Radius 0.30 around {reference_name}; 720 points; worst / median step {:0.3}",
            steps.max / steps.p50.max(1e-9)
        ),
        &[
            (&locus, "#cbd5e1", false),
            (&tri, "#ffee66", false),
            (&before, "#6b7280", true),
            (&marker(reference), "#ffffff", false),
        ],
    )?;
    std::fs::write(assets.join("input-gamut-smoothness.svg"), svg).map_err(|e| e.to_string())?;
    let smooth = serde_json::json!({"active":spec.active,"algorithm":spec.algorithm,"knee_threshold":spec.knee[0],"knee_limit":spec.knee[1],"knee_power":spec.knee[2],
        "probe_radius":0.3,"probe_samples":720,"worst_step":steps.max,"median_step":steps.p50,"worst_over_median_step":steps.max/steps.p50.max(1e-9),"reference_illuminant":reference_name});
    Ok((preview, smooth))
}
fn edges(n: usize) -> Vec<[f64; 3]> {
    let mut out = Vec::with_capacity(12 * n);
    for axis in 0..3 {
        for a in [0., 1.] {
            for b in [0., 1.] {
                for i in 0..n {
                    let mut p = [0.; 3];
                    let free: Vec<_> = (0..3).filter(|c| *c != axis).collect();
                    p[axis] = i as f64 / (n - 1) as f64;
                    p[free[0]] = a;
                    p[free[1]] = b;
                    out.push(p);
                }
            }
        }
    }
    out
}
fn xy(xyz: [f64; 3]) -> [f64; 2] {
    let s = xyz.iter().sum::<f64>();
    if s.abs() > 1e-12 {
        [xyz[0] / s, xyz[1] / s]
    } else {
        [0., 0.]
    }
}
pub(crate) fn output(
    options: &BakeOptions<'_>,
    data: &Path,
    cube: &[[f64; 3]],
    assets: &Path,
) -> Result<serde_json::Value, String> {
    let input = lut_export::space(options.input)?;
    let output = lut_export::space(options.output)?;
    let samples = edges(96);
    let mut params = options.params.clone();
    params.debug.lut_mode = true;
    params.io.input_color_space = input.primaries.into();
    params.io.output_color_space = output.primaries.into();
    params.io.input_cctf_decoding = false;
    params.io.output_cctf_encoding = false;
    params.io.output_gamut_compress.algorithm = "off".into();
    params.io.crop = false;
    params.io.upscale_factor = 1.;
    params.taps = Default::default();
    let pipeline = Pipeline::new_with_spectral(
        load_profile_by_name(data, options.film).map_err(|e| e.to_string())?,
        load_profile_by_name(data, options.paper).map_err(|e| e.to_string())?,
        params,
        data,
    )?;
    let linear: Vec<_> = samples
        .iter()
        .flat_map(|p| input.decode_pixel(*p).map(from_f64))
        .collect();
    let rendered = pipeline.process(
        ImageBuf::from_data(samples.len() as u32, 1, linear),
        &spektrafilm_gpu::cpu_backend::CpuBackend,
    );
    let compressor =
        OutputGamutCompress::build(&options.params.io.output_gamut_compress, output.primaries);
    let unbounded: Vec<_> = rendered
        .pixels()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();
    let compressed: Vec<_> = unbounded.iter().map(|p| compressor.compress(*p)).collect();
    let cs = transfer::get(output.primaries)?;
    let locus = crate::input_gamut::spectral_locus_xy();
    let tri = triangle(output);
    let mut ab = Cloud::new(640, 640, [-0.5, 0.5, -0.5, 0.5]);
    let mut chroma = Cloud::new(720, 800, [-0.05, 0.85, -0.05, 0.95]);
    for p in cube {
        let xyz = m::xyz(*p, output);
        let lab = m::oklab(xyz);
        ab.point([lab[1], lab[2]], *p, 0.12);
        chroma.point(xy(xyz), *p, 0.06);
    }
    let mut ab_lines = Vec::new();
    let mut xy_lines = Vec::new();
    let mut bright = 0usize;
    let mut displacements = Vec::new();
    for i in 0..samples.len() {
        let p = unbounded[i];
        let ach = p.into_iter().fold(f64::NEG_INFINITY, f64::max);
        if ach > 1e-2 {
            bright += 1;
            let max_d = p
                .iter()
                .map(|c| (ach - c) / ach.max(1e-6))
                .fold(0., f64::max);
            if max_d > 1. {
                displacements.push(
                    (0..3)
                        .map(|c| (p[c] - compressed[i][c]).powi(2))
                        .sum::<f64>()
                        .sqrt(),
                );
            }
        }
    }
    // Segment colors follow the encoded input hue. Four shorter pieces per cube edge
    // retain the hue progression while distinguishing solid before / dashed after.
    for segment in 0..12 {
        for part in 0..4 {
            let start = segment * 96 + part * 24;
            let end = (start + 25).min((segment + 1) * 96);
            let source = samples[(start + end - 1) / 2];
            let max = source.into_iter().fold(f64::NEG_INFINITY, f64::max);
            let min = source.into_iter().fold(f64::INFINITY, f64::min);
            let d = max - min;
            let h = if d < 1e-12 {
                0.
            } else if source[0] == max {
                ((source[1] - source[2]) / d) / 6.
            } else if source[1] == max {
                ((source[2] - source[0]) / d + 2.) / 6.
            } else {
                ((source[0] - source[1]) / d + 4.) / 6.
            };
            let color = hsv(h).map(|v| (v * 255.) as u8);
            let color = format!("#{:02x}{:02x}{:02x}", color[0], color[1], color[2]);
            for (points, dashed) in [(&unbounded, false), (&compressed, true)] {
                let xyz: Vec<_> = points[start..end]
                    .iter()
                    .map(|p| m::mul(cs.to_xyz, *p))
                    .collect();
                let a: Vec<_> = xyz
                    .iter()
                    .map(|p| {
                        let l = m::oklab(*p);
                        [l[1], l[2]]
                    })
                    .collect();
                let b: Vec<_> = xyz.into_iter().map(xy).collect();
                ab_lines.push((a, color.clone(), dashed));
                xy_lines.push((b, color.clone(), dashed));
            }
        }
    }
    let mut left = v::image_plot(
        &ab.image(),
        "Output gamut rim in OkLab chroma",
        ["OkLab a", "OkLab b"],
        [-0.5, 0.5, -0.5, 0.5],
        "Solid = unbounded runtime; dashed = compressed; cloud = LUT cells",
    )?;
    let at = left.rfind("</g></svg>").unwrap();
    let lines = ab_lines
        .iter()
        .map(|(p, c, d)| v::line_svg(p, c, *d, [-0.5, 0.5, -0.5, 0.5], 640., 640.))
        .collect::<String>();
    left.insert_str(at, &lines);
    let refs: Vec<_> = std::iter::once((locus.as_slice(), "#cbd5e1", false))
        .chain(std::iter::once((tri.as_slice(), "#ffffff", false)))
        .chain(
            xy_lines
                .iter()
                .map(|(p, c, d)| (p.as_slice(), c.as_str(), *d)),
        )
        .collect();
    let right = v::xy_cloud_svg(
        &chroma,
        "Output gamut rim in chromaticity",
        "12 encoded cube edges × 96 points; same compression event",
        &refs,
    )?;
    // Independent viewBoxes keep both coordinate systems legible in one report figure.
    let svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1620\" height=\"950\" viewBox=\"0 0 1620 950\"><rect width=\"1620\" height=\"950\" fill=\"#101827\"/><svg x=\"0\" y=\"0\" width=\"770\" height=\"790\" viewBox=\"0 0 770 790\">{}</svg><svg x=\"770\" y=\"0\" width=\"850\" height=\"950\" viewBox=\"0 0 850 950\">{}</svg></svg>",
        &left[left.find('>').unwrap() + 1..left.rfind("</svg>").unwrap()],
        &right[right.find('>').unwrap() + 1..right.rfind("</svg>").unwrap()]
    );
    std::fs::write(assets.join("output-gamut.svg"), svg).map_err(|e| e.to_string())?;
    let stats = m::summary(&displacements);
    Ok(
        serde_json::json!({"compression_algorithm":options.params.io.output_gamut_compress.algorithm,"rim_samples":samples.len(),"rim_oog_samples":displacements.len(),"rim_oog_fraction":displacements.len() as f64/bright.max(1) as f64,"rim_max_displacement":stats.max,"rim_mean_displacement":stats.mean}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn input_gamut_projection_and_complete_summaries_match_source_qa() {
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/qa_gamut.json")).unwrap();
        let directory =
            std::env::temp_dir().join(format!("spektrafilm-gamut-fixture-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        for case in cases.as_array().unwrap() {
            let space = lut_export::space(case["input"].as_str().unwrap()).unwrap();
            let grid: Vec<[f64; 3]> = serde_json::from_value(case["grid"].clone()).unwrap();
            let expected_xy: Vec<[f64; 2]> = serde_json::from_value(case["xy"].clone()).unwrap();
            let expected_b: Vec<f64> = serde_json::from_value(case["b"].clone()).unwrap();
            let expected_inside: Vec<bool> =
                serde_json::from_value(case["inside"].clone()).unwrap();
            let locus = crate::input_gamut::spectral_locus_xy();
            for (i, (xy, b)) in projection(&grid, space, white("D55").unwrap())
                .unwrap()
                .iter()
                .enumerate()
            {
                assert!((b - expected_b[i]).abs() < 1e-10);
                assert!(
                    (xy[0] - expected_xy[i][0]).abs() < 1e-11
                        && (xy[1] - expected_xy[i][1]).abs() < 1e-11
                );
                assert_eq!(
                    v::contains(*xy, &locus),
                    expected_inside[i],
                    "{} sample {i}",
                    space.name
                );
            }
            let spec = crate::params::InputGamutCompressParams {
                active: case["active"].as_bool().unwrap(),
                algorithm: case["algorithm"].as_str().unwrap().into(),
                knee: [0., 1., 6.],
                boundary: "locus".into(),
                hull_detail: 5.,
            };
            let (preview, smooth) = input(&grid, space, &spec, "D55", &directory).unwrap();
            for (actual, key) in [(&preview, "preview"), (&smooth, "smooth")] {
                for (name, expected) in case[key].as_object().unwrap() {
                    let got = &actual[name];
                    if let Some(value) = expected.as_f64() {
                        assert!(
                            (got.as_f64().unwrap() - value).abs() < 1e-7,
                            "{key} {name}: {got} vs {value}"
                        );
                    } else {
                        assert_eq!(got, expected, "{key} {name}");
                    }
                }
            }
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
}
