//! Standalone OCIO configs use a linear AP0 scene reference, matching Python.
use crate::{
    lut_bundle::BundleMeta,
    lut_export::{self, BakeOptions},
};
use std::collections::BTreeMap;
fn quoted(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}
fn builtin_lines(steps: &[(String, String)]) -> Vec<String> {
    steps
        .iter()
        .map(|(style, direction)| {
            format!(
                "        - !<BuiltinTransform> {{style: {}, direction: {direction}}}",
                quoted(style)
            )
        })
        .collect()
}
fn color_space(name: &str, description: &str, steps: Vec<String>) -> String {
    let mut text = format!(
        "  - !<ColorSpace>\n    name: {}\n    family: spektrafilm\n    description: {}\n    bitdepth: 32f\n    isdata: false\n    allocation: uniform\n",
        quoted(name),
        quoted(description)
    );
    if !steps.is_empty() {
        text.push_str("    from_scene_reference: !<GroupTransform>\n      children:\n");
        text.push_str(&steps.join("\n"));
        text.push('\n');
    }
    text
}
pub fn config(options: &BakeOptions<'_>, bundle: &BundleMeta, taps: &[&str]) -> Option<String> {
    let catalog: BTreeMap<String, Vec<(String, String)>> =
        serde_json::from_str(include_str!("ocio_builtins.json")).ok()?;
    let input = lut_export::space(options.input).ok()?.name;
    let output = lut_export::space(options.output).ok()?.name;
    let input_steps = catalog.get(input)?;
    catalog.get(output)?;
    let final_name = format!("spektrafilm_{}_{}", options.film, options.paper);
    let mut config = format!(
        "ocio_profile_version: 2.4\nsearch_path: .\nstrictparsing: true\nroles:\n  default: ACES2065-1\n  scene_linear: ACES2065-1\n  aces_interchange: ACES2065-1\n  color_timing: ACES2065-1\n  compositing_log: ACES2065-1\ndisplays:\n  {}:\n    - !<View> {{name: Film, colorspace: {}}}\n    - !<View> {{name: Raw, colorspace: {}}}\nactive_displays: [{}]\nactive_views: [Film, Raw]\ncolorspaces:\n",
        quoted(output),
        quoted(&final_name),
        quoted(output),
        quoted(output)
    );
    config.push_str(&color_space(
        "ACES2065-1",
        "Linear AP0 scene reference",
        vec![],
    ));
    for name in [input, output] {
        if name == "ACES2065-1" || (name == output && input == output) {
            continue;
        }
        config.push_str(&color_space(
            name,
            "Native transport encoding",
            builtin_lines(catalog.get(name)?),
        ));
    }
    // When input and output match, emit the transport once.
    if input == output && input != "ACES2065-1" {
        config.push_str(&color_space(
            input,
            "Native transport encoding",
            builtin_lines(input_steps),
        ));
    }
    let mut chain = builtin_lines(input_steps);
    for pair in taps.windows(2) {
        let entry = bundle
            .luts
            .iter()
            .find(|e| e.inject == pair[0] && e.collect == pair[1])?;
        chain.push(format!(
            "        - !<FileTransform> {{src: {}, interpolation: tetrahedral}}",
            quoted(&entry.file)
        ));
        if pair[1] != "rgb_out" {
            let name = format!("{}_{}", pair[1], options.film);
            let description = serde_json::to_string(bundle.wires.get(pair[1])?).ok()?;
            config.push_str(&color_space(&name,&format!("Normalized boundary wire {description}. Forward transform only; LUT inversion is undefined."),chain.clone()));
        }
    }
    config.push_str(&color_space(
        &final_name,
        "Deterministic film and print simulation",
        chain,
    ));
    Some(config)
}
