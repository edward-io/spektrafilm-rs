//! Native scientific QA report and portable SVG/PNG diagnostics.
use crate::{
    lut_export::{self, BakeOptions},
    lut_quality::{tetrahedral, trilinear},
    qa_metrics as m,
};
use serde::Serialize;
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
use std::path::Path;
#[derive(Serialize)]
pub struct Case {
    pub name: &'static str,
    pub category: &'static str,
    pub summary: serde_json::Value,
    pub passed: Option<bool>,
    pub figure: String,
}
#[derive(Serialize)]
pub struct Report {
    pub cases: Vec<Case>,
    pub trilinear_de2000: m::Summary,
    pub tetrahedral_de2000: m::Summary,
    pub trilinear_itp: m::Summary,
    pub tetrahedral_itp: m::Summary,
    pub oklab_distance: m::Summary,
}
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
fn svg_start(title: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\" height=\"460\" viewBox=\"0 0 800 460\"><rect width=\"800\" height=\"460\" fill=\"#101827\"/><g font-family=\"sans-serif\" fill=\"#e5e7eb\"><text x=\"45\" y=\"30\" font-size=\"18\">{}</text>",
        escape(title)
    )
}
fn curve(
    path: &Path,
    title: &str,
    xlabel: &str,
    ylabel: &str,
    x: &[f64],
    lines: &[(&str, Vec<f64>, &str)],
) -> Result<(), String> {
    let styled: Vec<_> = lines
        .iter()
        .map(|(name, values, color)| (*name, values.clone(), *color, 1.))
        .collect();
    curve_styled(path, title, xlabel, ylabel, x, &styled)
}
fn curve_styled(
    path: &Path,
    title: &str,
    xlabel: &str,
    ylabel: &str,
    x: &[f64],
    lines: &[(&str, Vec<f64>, &str, f64)],
) -> Result<(), String> {
    let lo = x.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = lines
        .iter()
        .flat_map(|(_, v, _, _)| v.iter())
        .copied()
        .fold(0f64, f64::min);
    let max = lines
        .iter()
        .flat_map(|(_, v, _, _)| v.iter())
        .copied()
        .fold(1e-9, f64::max);
    let mut s = svg_start(title);
    for k in 0..=4 {
        let y = 70. + k as f64 * 75.;
        let value = max - k as f64 / 4. * (max - min);
        s.push_str(&format!("<path d=\"M 72 {y} H 755\" stroke=\"#374151\"/><text x=\"10\" y=\"{}\" font-size=\"12\">{value:.3}</text>",y+4.));
        let xv = 72. + k as f64 / 4. * 683.;
        s.push_str(&format!(
            "<text x=\"{xv}\" y=\"392\" font-size=\"12\" text-anchor=\"middle\">{:.2}</text>",
            lo + k as f64 / 4. * (hi - lo)
        ));
    }
    for (i, (label, values, color, opacity)) in lines.iter().enumerate() {
        let points = x
            .iter()
            .zip(values)
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .map(|(x, y)| {
                format!(
                    "{:.2},{:.2}",
                    72. + (x - lo) / (hi - lo).max(1e-12) * 683.,
                    370. - (y - min) / (max - min).max(1e-12) * 300.
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        s.push_str(&format!("<polyline points=\"{points}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2\" stroke-opacity=\"{opacity}\"/><text x=\"{}\" y=\"55\" font-size=\"12\" fill=\"{color}\">{}</text>",72+i*170,escape(label)));
    }
    s.push_str(&format!("<text x=\"350\" y=\"430\" font-size=\"13\">{}</text><text x=\"75\" y=\"450\" font-size=\"12\">{}</text></g></svg>",escape(xlabel),escape(ylabel)));
    std::fs::write(path, s).map_err(|e| e.to_string())
}
fn scatter(
    path: &Path,
    title: &str,
    points: &[[f64; 2]],
    outline: &[[f64; 2]],
    arrows: Option<&[[f64; 2]]>,
) -> Result<(), String> {
    let mut s = svg_start(title);
    let xy = |p: [f64; 2]| [80. + p[0] / 0.8 * 650., 380. - p[1] / 0.9 * 300.];
    for k in 0..=4 {
        let x = 80. + k as f64 * 162.5;
        let y = 380. - k as f64 * 75.;
        s.push_str(&format!("<path d=\"M {x} 80 V 380 M 80 {y} H 730\" stroke=\"#374151\"/><text x=\"{x}\" y=\"400\" font-size=\"12\">{:.2}</text>",k as f64*0.2));
    }
    let poly = outline
        .iter()
        .map(|p| {
            let p = xy(*p);
            format!("{},{}", p[0], p[1])
        })
        .collect::<Vec<_>>()
        .join(" ");
    s.push_str(&format!(
        "<polyline points=\"{poly}\" fill=\"none\" stroke=\"#d1d5db\" stroke-width=\"1.5\"/>"
    ));
    for (i, p) in points.iter().enumerate() {
        if p.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let a = xy(*p);
        s.push_str(&format!(
            "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"2\" fill=\"#6ee7b7\" fill-opacity=\".6\"/>",
            a[0], a[1]
        ));
        if let Some(target) = arrows.and_then(|v| v.get(i)) {
            let b = xy(*target);
            s.push_str(&format!(
                "<path d=\"M {} {} L {} {}\" stroke=\"#fb923c\" stroke-opacity=\".6\"/>",
                a[0], a[1], b[0], b[1]
            ));
        }
    }
    s.push_str(
        "<text x=\"390\" y=\"430\">CIE x</text><text x=\"30\" y=\"80\">CIE y</text></g></svg>",
    );
    std::fs::write(path, s).map_err(|e| e.to_string())
}
fn chroma(v: [f64; 3]) -> [f64; 2] {
    let t = v.iter().sum::<f64>().max(1e-12);
    [v[0] / t, v[1] / t]
}
fn image(samples: &[[f64; 3]]) -> ImageBuf {
    ImageBuf::from_data(
        samples.len() as u32,
        1,
        samples.iter().flatten().map(|v| from_f64(*v)).collect(),
    )
}
fn rows(image: &ImageBuf) -> Vec<[f64; 3]> {
    image
        .pixels()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect()
}
fn channels(v: &[[f64; 3]]) -> [Vec<f64>; 3] {
    std::array::from_fn(|c| v.iter().map(|p| p[c]).collect())
}
fn rgb_lines(v: &[[f64; 3]]) -> Vec<(&'static str, Vec<f64>, &'static str)> {
    let [r, g, b] = channels(v);
    vec![
        ("R", r, "#fb7185"),
        ("G", g, "#6ee7b7"),
        ("B", b, "#93c5fd"),
    ]
}
fn case(
    name: &'static str,
    category: &'static str,
    summary: serde_json::Value,
    passed: Option<bool>,
    figure: &str,
) -> Case {
    Case {
        name,
        category,
        summary,
        passed,
        figure: figure.into(),
    }
}
pub fn build(
    options: &BakeOptions<'_>,
    data: &Path,
    chain: &[ImageBuf],
    samples: &[[f64; 3]],
    reference: &ImageBuf,
    directory: &Path,
) -> Result<Report, String> {
    let n = options.size as usize;
    let output = lut_export::space(options.output)?;
    let input = lut_export::space(options.input)?;
    // Upstream QA resamples the canonical chain at cube vertices, then treats
    // that effective cube as one LUT for its probes and interpolators.
    let cube = lut_export::grid(options.size);
    let cube_input = rows(&cube);
    let cube_out: Vec<_> = cube_input
        .iter()
        .map(|p| chain.iter().fold(*p, |p, t| trilinear(t, n, p)))
        .collect();
    let effective = image(&cube_out);
    let evaluate = |p: [f64; 3]| trilinear(&effective, n, p);
    let eval_tet = |p: [f64; 3]| tetrahedral(&effective, n, p);
    let mut errors = [vec![], vec![], vec![], vec![], vec![]];
    for (p, expected) in samples.iter().zip(reference.pixels()) {
        let expected = [expected[0] as f64, expected[1] as f64, expected[2] as f64];
        let a = m::differences(evaluate(*p), expected, output);
        let b = m::differences(eval_tet(*p), expected, output);
        for (c, v) in [a[0], b[0], a[1], b[1], a[2]].into_iter().enumerate() {
            errors[c].push(v);
        }
    }
    crate::qa_visual::off_grid(samples, &errors[0], &errors[1], &directory.join("qa"))?;
    let [tri, tet, itp, tet_itp, oklab] = errors.map(|v| m::summary(&v));
    let passed = tri.max <= 2. && tri.p99 <= 1. && tet.max <= 2. && tet.p99 <= 1.;
    let structure = m::structure(&effective, n);
    let assets = directory.join("qa");
    std::fs::create_dir_all(&assets).map_err(|e| e.to_string())?;
    let mut cases = vec![];
    let axis: Vec<_> = (0..65).map(|i| i as f64 / 64.).collect();
    let neutral: Vec<_> = axis.iter().map(|t| evaluate([*t; 3])).collect();
    let neutral_ref = rows(&lut_export::reference_image(
        options,
        data,
        image(&axis.iter().map(|v| [*v; 3]).collect::<Vec<_>>()),
    )?);
    curve(
        &assets.join("neutral.svg"),
        "Neutral transfer: sampled LUT and spectral reference",
        "Encoded input",
        "Encoded output",
        &axis,
        &[
            (
                "LUT mean",
                neutral.iter().map(|p| p.iter().sum::<f64>() / 3.).collect(),
                "#93c5fd",
            ),
            (
                "Reference mean",
                neutral_ref
                    .iter()
                    .map(|p| p.iter().sum::<f64>() / 3.)
                    .collect(),
                "#fbbf24",
            ),
        ],
    )?;
    cases.push(case("off_grid_identity","LUT fidelity",serde_json::json!({"trilinear_dE2000":tri,"tetrahedral_dE2000":tet,"trilinear_dITP":itp,"tetrahedral_dITP":tet_itp,"targets":{"max_dE2000":2.,"p99_dE2000":1.}}),Some(passed),"qa/off-grid.svg"));
    let pin = input.encode_pixel([0.18 / options.gain.unwrap_or(input.auto_gain); 3])[0];
    let mut per_axis = vec![];
    for c in 0..3 {
        per_axis.push((
            match c {
                0 => "R→R",
                1 => "G→G",
                _ => "B→B",
            },
            axis.iter()
                .map(|v| {
                    let mut p = [pin; 3];
                    p[c] = *v;
                    evaluate(p)[c]
                })
                .collect(),
            ["#fb7185", "#6ee7b7", "#93c5fd"][c],
        ));
    }
    curve(
        &assets.join("monotonicity.svg"),
        "Channel transfer curves through midgray",
        "Encoded input",
        "Matching output channel",
        &axis,
        &per_axis,
    )?;
    cases.push(case("monotonicity","LUT fidelity",serde_json::json!({"violations":structure.monotonicity_violations,"worst_negative_diff":structure.worst_negative_diff}),Some(structure.monotonicity_violations==0),"qa/monotonicity.svg"));
    crate::qa_visual::conditioning(
        &structure.jacobian_log_condition_values,
        n.saturating_sub(2),
        structure.jacobian_log_condition.max,
        &assets,
    )?;
    crate::qa_visual::variation(
        structure.total_variation,
        structure.fft_highband_ratio,
        &assets,
    )?;
    cases.push(case(
        "jacobian_condition",
        "LUT fidelity",
        serde_json::to_value(&structure.jacobian_log_condition).unwrap(),
        None,
        "qa/conditioning.svg",
    ));
    cases.push(case("total_variation","LUT fidelity",serde_json::json!({"tv_r":structure.total_variation[0],"tv_g":structure.total_variation[1],"tv_b":structure.total_variation[2],"tv":structure.total_variation.iter().sum::<f64>(),"axial_fft_highband_ratio":structure.fft_highband_ratio}),None,"qa/variation.svg"));
    let characteristic = crate::qa_patterns::characteristic_probe(&effective, n)?;
    let mut panels = Vec::new();
    for c in 0..3 {
        let mut lines = Vec::new();
        for pin in 0..5 {
            for channel in 0..3 {
                lines.push((
                    if pin == 0 {
                        ["Red", "Green", "Blue"][channel]
                    } else {
                        ""
                    },
                    characteristic.sweeps[c][pin]
                        .iter()
                        .map(|p| -p[channel].clamp(1e-4, 1.).log10())
                        .collect(),
                    ["#fb7185", "#6ee7b7", "#93c5fd"][channel],
                    [0.3, 0.6, 1., 0.6, 0.3][pin],
                ));
            }
        }
        let filename = format!("characteristic-{}.svg", ["red", "green", "blue"][c]);
        curve_styled(
            &assets.join(&filename),
            &format!(
                "{} input; other channels pinned at 0.2–1.0 density",
                ["Red", "Green", "Blue"][c]
            ),
            "Encoded input",
            "Optical density −log10(output)",
            &characteristic.axis,
            &lines,
        )?;
        panels.push(filename);
    }
    let density: Vec<_> = characteristic
        .neutral
        .iter()
        .map(|p| p.map(|v| -v.clamp(1e-4, 1.).log10()))
        .collect();
    curve(
        &assets.join("characteristic-neutral.svg"),
        "Neutral characteristic curve",
        "Encoded neutral input",
        "Optical density −log10(output)",
        &characteristic.axis,
        &rgb_lines(&density),
    )?;
    panels.push("characteristic-neutral.svg".into());
    let mut combined = String::from(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1600\" height=\"920\" viewBox=\"0 0 1600 920\">",
    );
    for (i, file) in panels.iter().enumerate() {
        let panel = std::fs::read_to_string(assets.join(file)).map_err(|e| e.to_string())?;
        let inner = &panel[panel.find('>').unwrap() + 1..panel.rfind("</svg>").unwrap()];
        combined.push_str(&format!(
            "<g transform=\"translate({},{})\">{inner}</g>",
            i % 2 * 800,
            i / 2 * 460
        ));
    }
    combined.push_str("</svg>");
    std::fs::write(assets.join("characteristic.svg"), combined).map_err(|e| e.to_string())?;
    cases.push(case("characteristic_curve","Model diagnostic",serde_json::json!({"system_gamma_at_mid":characteristic.system_gamma_at_mid,"max_channel_density_spread":characteristic.max_channel_density_spread,"pin_densities":characteristic.pin_densities,"pin_input_codes":characteristic.pin_codes}),None,"qa/characteristic.svg"));
    let dynamic_ramp = crate::qa_patterns::dynamic_range_ramp(input)?;
    let stops = dynamic_ramp.stops;
    let clipped = dynamic_ramp.clipped;
    let ramp = dynamic_ramp.encoded;
    let luminance: Vec<_> = ramp
        .iter()
        .map(|p| m::xyz(evaluate(*p), output)[1].max(1e-6))
        .collect();
    let mut dynamic = m::dynamic_range(&stops, &luminance, &clipped);
    let i = stops.partition_point(|v| *v < 0.).clamp(1, stops.len() - 1);
    let mid = luminance[i - 1]
        + (0. - stops[i - 1]) / (stops[i] - stops[i - 1]) * (luminance[i] - luminance[i - 1]);
    dynamic["midgray_output_y"] = serde_json::json!(mid);
    dynamic["midgray_offset_stops"] = serde_json::json!((mid / 0.18).log2());
    curve(
        &assets.join("dynamic-range.svg"),
        "Dynamic range in camera exposure stops",
        "Input stops above native midgray",
        "Output density",
        &stops,
        &[(
            "Density",
            luminance.iter().map(|v| -v.log10()).collect(),
            "#93c5fd",
        )],
    )?;
    cases.push(case(
        "dynamic_range_usage",
        "Model diagnostic",
        dynamic,
        None,
        "qa/dynamic-range.svg",
    ));
    let (planck, temps) = crate::qa_patterns::planckian_sweep(input, 16, options.gain)?;
    let planck_out: Vec<_> = planck.iter().map(|p| evaluate(*p)).collect();
    let planck_xy: Vec<_> = planck_out
        .iter()
        .map(|p| chroma(m::xyz(*p, output)))
        .collect();
    let bend = crate::qa_patterns::max_bend_angle(&planck_xy);
    let links: Vec<_> = planck_xy
        .iter()
        .skip(1)
        .copied()
        .chain(planck_xy.last().copied())
        .collect();
    scatter(
        &assets.join("planckian.svg"),
        "Daylight / Planckian white-balance sweep",
        &planck_xy,
        &crate::input_gamut::spectral_locus_xy(),
        Some(&links),
    )?;
    cases.push(case("planckian_sweep","Model diagnostic",serde_json::json!({"max_bend_angle_deg":bend,"cct_range_k":"2700-10000","temperatures":temps,"output_xy":planck_xy,"reference_max_bend_angle_deg":30.}),Some(bend<=30.),"qa/planckian.svg"));
    let indices = crate::qa_rng::hull_indices(cube_input.len());
    let in_lab: Vec<_> = indices
        .iter()
        .map(|&i| m::oklab(m::xyz(cube_input[i], output)))
        .collect();
    let out_lab: Vec<_> = indices
        .iter()
        .map(|&i| m::oklab(m::xyz(cube_out[i], output)))
        .collect();
    let volume_in = m::hull_volume(&in_lab);
    let volume_out = m::hull_volume(&out_lab);
    let mut output_gamut = crate::qa_gamut::output(options, data, &cube_out, &assets)?;
    output_gamut["input_hull_volume"] = serde_json::json!(volume_in);
    output_gamut["output_hull_volume"] = serde_json::json!(volume_out);
    output_gamut["compression_ratio"] = serde_json::json!(volume_out / volume_in.max(1e-12));
    output_gamut["fold_triangles"] = serde_json::json!(structure.surface_flips);
    output_gamut["fold_fraction"] =
        serde_json::json!(structure.surface_flips as f64 / structure.surface_cells.max(1) as f64);
    cases.push(case(
        "output_gamut_compression",
        "LUT fidelity",
        output_gamut,
        Some(
            structure.surface_flips == 0
                && (0.05..=1.05).contains(&(volume_out / volume_in.max(1e-12))),
        ),
        "qa/output-gamut.svg",
    ));
    // Hull comparison deliberately uses the output frame for both clouds;
    // hue comparison uses each transport's own primaries and excludes the
    // unphysical chroma extremes, matching the reference diagnostic.
    let hue_pairs: Vec<_> = cube_input
        .iter()
        .zip(&cube_out)
        .filter_map(|(before, after)| {
            let lab = m::oklab(m::xyz(*before, input));
            (lab[1].hypot(lab[2]) <= 0.6).then(|| (lab, m::oklab(m::xyz(*after, output))))
        })
        .collect();
    let hue_in: Vec<_> = hue_pairs.iter().map(|p| p.0).collect();
    let hue_out: Vec<_> = hue_pairs.iter().map(|p| p.1).collect();
    let mut hue = m::hue_rotation(&hue_in, &hue_out);
    hue["samples_in_locus"] = serde_json::json!(hue_in.len());
    hue["samples_filtered_out_of_locus"] = serde_json::json!(cube_input.len() - hue_in.len());
    crate::qa_visual::hue_twist(&hue_in, &hue_out, &assets)?;
    cases.push(case(
        "hue_twist_oklab",
        "Model diagnostic",
        hue,
        None,
        "qa/hue-twist.svg",
    ));
    let envelope_summary = crate::qa_visual::spectral_cloud(&cube_out, n, output, &assets)?;
    cases.push(case(
        "spectral_locus_envelope",
        "Model diagnostic",
        envelope_summary,
        None,
        "qa/spectral-envelope.svg",
    ));
    let film =
        crate::profile::load_profile_by_name(data, options.film).map_err(|e| e.to_string())?;
    let (preview, smoothness) = crate::qa_gamut::input(
        &cube_input,
        input,
        &options.params.io.input_gamut_compress,
        &film.info.reference_illuminant,
        &assets,
    )?;
    cases.push(case(
        "input_gamut_compression_preview",
        "Gamut compression",
        preview,
        None,
        "qa/input-gamut.svg",
    ));
    cases.push(case(
        "input_gamut_compression_smoothness",
        "Gamut compression",
        smoothness,
        None,
        "qa/input-gamut-smoothness.svg",
    ));
    let noise_summary =
        crate::qa_noise::sensitivity(&effective, n, input, output, options.gain, &assets)?;
    cases.push(case(
        "noise_sensitivity",
        "Picture style",
        noise_summary,
        None,
        "qa/noise.svg",
    ));
    let gradient_summary =
        crate::qa_noise::gradient(input, output, options.gain, &evaluate, &assets)?;
    cases.push(case(
        "noise_gradient",
        "Picture style",
        gradient_summary,
        None,
        "qa/noise-gradient.svg",
    ));
    let edge_summary = crate::qa_style::edge_stress(options, data, &assets)?;
    cases.push(case(
        "output_gamut_edge_stress",
        "Picture style",
        edge_summary,
        None,
        "qa/gamut-edges.svg",
    ));
    let slice_summary = crate::qa_style::slices(&effective, n, output, &assets)?;
    cases.push(case(
        "rg_plane_slices",
        "Picture style",
        slice_summary,
        None,
        "qa/slices.svg",
    ));
    let report = Report {
        cases,
        trilinear_de2000: tri,
        tetrahedral_de2000: tet,
        trilinear_itp: itp,
        tetrahedral_itp: tet_itp,
        oklab_distance: oklab,
    };
    write_html(&report, directory)?;
    Ok(report)
}
fn write_html(report: &Report, directory: &Path) -> Result<(), String> {
    let mut html = String::from(
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\"><title>Spektrafilm LUT diagnostics</title><style>body{font:16px system-ui;max-width:1100px;margin:50px auto;padding:0 20px;background:#101827;color:#e5e7eb;line-height:1.6}section{margin:30px 0;border-top:1px solid #374151}img{max-width:100%;height:auto}pre{white-space:pre-wrap;background:#172033;padding:20px}a{color:#93c5fd}.pass{color:#6ee7b7}.fail{color:#fb7185}</style></head><body><h1>LUT diagnostics</h1><p>Native Rust analysis of the exported canonical LUT chain. Scientific diagnostics use a resampled effective cube, matching the upstream suite; qa.json also records the actual sequential chain's absolute error. Off-grid predictions use 50,000 NumPy-seeded samples and the deterministic spectral reference. Model and style diagnostics describe the transform; they do not certify a film stock or imply that every rendering decision must be monotone.</p><nav>",
    );
    for case in &report.cases {
        html.push_str(&format!("<a href=\"#{}\">{}</a> · ", case.name, case.name));
    }
    html.push_str("</nav>");
    for case in &report.cases {
        let status = match case.passed {
            Some(true) => "<span class=\"pass\">PASS</span>",
            Some(false) => "<span class=\"fail\">Outside reference target</span>",
            None => "Informational",
        };
        html.push_str(&format!("<section id=\"{}\"><h2>{}</h2><p>{} · {status}</p><img src=\"{}\" alt=\"{} diagnostic\"><details><summary>Measurements</summary><pre>{}</pre></details></section>",case.name,case.name,case.category,case.figure,case.name,escape(&serde_json::to_string_pretty(&case.summary).unwrap())));
    }
    html.push_str("<p>GPU grain and spatial effects are outside the deterministic LUT transform. All diagnostic images are local SVG/PNG files; no Python runtime is required.</p></body></html>");
    std::fs::write(directory.join("qa.html"), html).map_err(|e| e.to_string())
}

/// Native profile figures for the measurement CLI: sensitivity, characteristic
/// curves, dye spectra, and each channel's bounded toe fit / inferred floor.
pub fn profile_measurement_plots(
    profile: &crate::profile::Profile,
    fits: Option<&[crate::measure::ToeFit; 3]>,
    directory: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    let x = profile.log_exposure_f64();
    let y = profile.density_curves_f64();
    curve(
        &directory.join("density-curves.svg"),
        "Profile density curves",
        "Log exposure",
        "Optical density",
        &x,
        &rgb_lines(&y),
    )?;
    curve(
        &directory.join("sensitivity.svg"),
        "Spectral sensitivity",
        "Wavelength (nm)",
        "Log sensitivity",
        &profile.data.wavelengths,
        &rgb_lines(&profile.log_sensitivity_f64()),
    )?;
    let dyes: Vec<[f64; 3]> = profile
        .data
        .channel_density
        .iter()
        .map(|row| std::array::from_fn(|c| row.get(c).copied().unwrap_or(0.)))
        .collect();
    curve(
        &directory.join("dye-density.svg"),
        "Spectral dye density",
        "Wavelength (nm)",
        "Diffuse density",
        &profile.data.wavelengths,
        &rgb_lines(&dyes),
    )?;
    if let Some(fits) = fits {
        for c in 0..3 {
            let fitted: Vec<_> = x
                .iter()
                .map(|v| fits[c].density(*v, profile.is_positive()))
                .collect();
            let floor = vec![fits[c].parameters[2]; x.len()];
            curve(
                &directory.join(format!("toe-{}.svg", ["red", "green", "blue"][c])),
                &format!("{} toe and density floor", ["Red", "Green", "Blue"][c]),
                "Log exposure",
                "Optical density",
                &x,
                &[
                    (
                        "Measured curve",
                        y.iter().map(|p| p[c]).collect(),
                        "#93c5fd",
                    ),
                    ("Erf fit", fitted, "#fbbf24"),
                    ("Density floor", floor, "#f87171"),
                ],
            )?;
        }
    }
    Ok(())
}
