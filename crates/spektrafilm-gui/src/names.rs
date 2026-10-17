//! Display names for option identifiers stored in params (`center_weighted`,
//! `black_pro_mist`, …). Stored values never change; only what users read does.
use std::borrow::Cow;

pub fn label(id: &str) -> Cow<'_, str> {
    let known = match id {
        "off" => "Off",
        "none" => "None",
        "auto" => "Automatic",
        "xy" => "CIE xy",
        "oklch" => "OkLCh",
        "oklrab" => "OkLrab",
        "cam16ucs" => "CAM16-UCS",
        "jzazbz" => "JzAzBz",
        "aces_rgc" => "ACES RGC",
        "hanatos2025" => "Hanatos 2025",
        "arctic2026beta04" => "Arctic 2026 β04",
        "arctic2026alpha02" => "Arctic 2026 α02",
        "mallett2019" => "Mallett 2019",
        "jakob2019" => "Jakob 2019",
        "otsu2018" => "Otsu 2018",
        "gauss-lasers" => "Gaussian lasers",
        "libraw" => "LibRaw",
        "rawler" => "rawler (built-in)",
        "as_shot" => "As shot",
        "black_pro_mist" => "Black Pro-Mist",
        "pro_mist" => "Pro-Mist",
        "glimmerglass" => "Glimmerglass",
        "cinebloom" => "CineBloom",
        "inscribed_hull" => "Inscribed hull",
        "locus" => "Spectral locus",
        "hoya_x0" => "Hoya X0 (yellow-green)",
        "hoya_x1" => "Hoya X1 (green)",
        "hoya_y2" => "Hoya Y2 (yellow)",
        "hoya_ya3" => "Hoya YA3 (orange)",
        "hoya_r1" => "Hoya R1 (red)",
        "cube" => "CUBE",
        "3dl" => "Autodesk 3DL (10-bit)",
        "lumix" => "Lumix V-Log CUBE",
        "hald_png" => "Hald PNG",
        "1lut" => "1 LUT",
        "2lut" => "2 LUTs",
        "3lut" => "3 LUTs",
        "4lut" => "4 LUTs",
        "spline16" => "Spline 16",
        "spline36" => "Spline 36",
        "catrom" => "Catmull-Rom",
        "lanczos3" => "Lanczos 3",
        "multi_zone" => "Multi-zone",
        "center_weighted" => "Center-weighted",
        "highlight_weighted" => "Highlight-weighted",
        _ => return humanize(id),
    };
    Cow::Borrowed(known)
}

/// Fallback for lower-case identifiers: `foo_bar` → `Foo bar`. Values that
/// already read as names (`sRGB`, `TH-KG3`, `D65`) pass through unchanged.
fn humanize(id: &str) -> Cow<'_, str> {
    if id.is_empty() || id.chars().any(|c| c.is_ascii_uppercase() || c == ' ') {
        return Cow::Borrowed(id);
    }
    let spaced = id.replace('_', " ");
    let mut chars = spaced.chars();
    match chars.next() {
        Some(first) => Cow::Owned(first.to_uppercase().chain(chars).collect()),
        None => Cow::Borrowed(id),
    }
}

/// Workflow routes, grouped by intent, with the label shown for each.
pub const WORKFLOWS: [(&str, &str, &str); 6] = [
    (
        "Simulate",
        "input > film > print > scan",
        "Film → print → scan",
    ),
    ("Simulate", "input > film > scan", "Film → scan"),
    (
        "Convert a scanned negative",
        "input > convert-film > print > scan",
        "Negative → print → scan",
    ),
    (
        "Convert a scanned negative",
        "input > convert-film > scan",
        "Negative → rescan",
    ),
    (
        "Convert a scanned negative",
        "input > convert-film > scan-minus-base",
        "Negative → rescan without base",
    ),
    ("Bypass", "input", "Color conversion only"),
];

pub fn workflow(route: &str) -> &str {
    WORKFLOWS
        .iter()
        .find(|(_, r, _)| *r == route)
        .map_or(route, |(_, _, name)| name)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn readable_names() {
        assert_eq!(label("center_weighted"), "Center-weighted");
        assert_eq!(label("as_shot"), "As shot");
        assert_eq!(label("daylight"), "Daylight");
        assert_eq!(label("sRGB"), "sRGB");
        assert_eq!(label("TH-KG3"), "TH-KG3");
        assert_eq!(workflow("input > film > scan"), "Film → scan");
    }
}
