//! Reproducible encoded-RGB LUT fidelity measurements against the spectral reference.
use crate::{
    lut_bundle::BundleMeta,
    lut_export::{self, BakeOptions},
};
use serde::Serialize;
use spektrafilm_math::{image::ImageBuf, precision::from_f64};
use std::path::Path;
#[derive(Serialize)]
pub struct Quality {
    pub samples: usize,
    pub interpolation: &'static str,
    pub max_absolute_error: f64,
    pub mean_absolute_error: f64,
    pub rms_error: f64,
    pub scientific: crate::qa_report::Report,
}
pub fn trilinear(table: &ImageBuf, n: usize, p: [f64; 3]) -> [f64; 3] {
    let q = p.map(|v| v.clamp(0., 1.) * (n - 1) as f64);
    let lower = q.map(|v| (v.floor() as usize).min(n - 2));
    let frac = std::array::from_fn::<_, 3, _>(|c| q[c] - lower[c] as f64);
    let mut value = [0.; 3];
    for b in 0..2 {
        for g in 0..2 {
            for r in 0..2 {
                let k = [r, g, b];
                let weight = (0..3)
                    .map(|c| if k[c] == 0 { 1. - frac[c] } else { frac[c] })
                    .product::<f64>();
                let i = (((lower[2] + b) * n + lower[1] + g) * n + lower[0] + r) * 3;
                for c in 0..3 {
                    value[c] += table.data[i + c] as f64 * weight;
                }
            }
        }
    }
    value
}
pub fn tetrahedral(table: &ImageBuf, n: usize, p: [f64; 3]) -> [f64; 3] {
    let q = p.map(|v| v.clamp(0., 1.) * (n - 1) as f64);
    let lo = q.map(|v| (v.floor() as usize).min(n - 2));
    let f = std::array::from_fn::<_, 3, _>(|c| q[c] - lo[c] as f64);
    let mut order = [0, 1, 2];
    order.sort_by(|a, b| f[*b].total_cmp(&f[*a]));
    let at = |offset: [usize; 3]| {
        let i = (((lo[2] + offset[2]) * n + lo[1] + offset[1]) * n + lo[0] + offset[0]) * 3;
        [
            table.data[i] as f64,
            table.data[i + 1] as f64,
            table.data[i + 2] as f64,
        ]
    };
    let a = at([0; 3]);
    let mut offset = [0; 3];
    offset[order[0]] = 1;
    let b = at(offset);
    offset[order[1]] = 1;
    let c = at(offset);
    let d = at([1; 3]);
    std::array::from_fn(|i| {
        (1. - f[order[0]]) * a[i]
            + (f[order[0]] - f[order[1]]) * b[i]
            + (f[order[1]] - f[order[2]]) * c[i]
            + f[order[2]] * d[i]
    })
}
fn read(path: &Path, n: u32) -> Result<ImageBuf, String> {
    let ext = path.extension().and_then(|v| v.to_str()).unwrap_or("");
    if ext == "png" {
        let rgb = image::open(path).map_err(|e| e.to_string())?.to_rgb32f();
        return Ok(ImageBuf::from_data(
            n,
            n * n,
            rgb.into_raw()
                .into_iter()
                .map(|v| from_f64(v as f64))
                .collect(),
        ));
    }
    let source = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    let mut shape = ext == "3dl";
    for line in source.lines() {
        if line
            .split_whitespace()
            .next()
            .is_none_or(|v| v.parse::<f64>().is_err())
        {
            continue;
        }
        let values: Vec<f64> = line
            .split_whitespace()
            .filter_map(|v| v.parse().ok())
            .collect();
        if shape && values.len() == n as usize {
            shape = false;
            continue;
        }
        if values.len() == 3 {
            rows.extend(
                values
                    .into_iter()
                    .map(|v| from_f64(if ext == "3dl" { v / 1023. } else { v })),
            );
        }
    }
    if rows.len() != n.pow(3) as usize * 3 {
        return Err(format!("{} has an invalid LUT data length", path.display()));
    }
    Ok(ImageBuf::from_data(n, n * n, rows))
}
pub fn assess(
    options: &BakeOptions<'_>,
    data_dir: &Path,
    bundle: &BundleMeta,
    directory: &Path,
) -> Result<Quality, String> {
    let names = ["rgb_in", "log_e_film", "cmy_film", "log_e_print", "rgb_out"];
    let mut chain = Vec::new();
    let mut tap = "rgb_in";
    while tap != "rgb_out" {
        let entry = bundle
            .luts
            .iter()
            .filter(|e| e.inject == tap)
            .min_by_key(|e| names.iter().position(|n| *n == e.collect).unwrap())
            .ok_or("broken LUT chain")?;
        chain.push(read(&directory.join(&entry.file), options.size)?);
        tap = &entry.collect;
    }
    let count = 50_000;
    let samples = crate::qa_rng::reference_samples(count);
    let image = ImageBuf::from_data(
        count as u32,
        1,
        samples.iter().flatten().map(|v| from_f64(*v)).collect(),
    );
    let expected = lut_export::reference_image(options, data_dir, image)?;
    let (mut max, mut sum, mut sum2) = (0f64, 0f64, 0f64);
    for (i, p) in samples.iter().copied().enumerate() {
        let result = chain
            .iter()
            .fold(p, |v, t| trilinear(t, options.size as usize, v));
        for c in 0..3 {
            let error = (result[c] - expected.data[i * 3 + c] as f64).abs();
            max = max.max(error);
            sum += error;
            sum2 += error * error;
        }
    }
    let scientific =
        crate::qa_report::build(options, data_dir, &chain, &samples, &expected, directory)?;
    Ok(Quality {
        scientific,
        samples: count,
        interpolation: "trilinear",
        max_absolute_error: max,
        mean_absolute_error: sum / (count * 3) as f64,
        rms_error: (sum2 / (count * 3) as f64).sqrt(),
    })
}

#[cfg(test)]
mod sampling_tests {
    #[test]
    fn qa_stimulus_matches_numpy_default_rng_seed_20260515() {
        let expected = [
            [0.7962043881416321, 0.06747125089168549, 0.42187580466270447],
            [0.8735185265541077, 0.5697402358055115, 0.8870167136192322],
            [
                0.38475680351257324,
                0.40927159786224365,
                0.04420004412531853,
            ],
            [
                0.36833682656288147,
                0.08512622863054276,
                0.24373841285705566,
            ],
            [
                0.24570029973983765,
                0.012508844025433064,
                0.8927443027496338,
            ],
        ];
        assert_eq!(crate::qa_rng::reference_samples(5), expected);
    }
}

impl Quality {
    /// Attach machine-readable QA and a quick quality table to an existing bundle.
    pub fn write_report(&self, directory: &Path) -> Result<(), String> {
        std::fs::write(
            directory.join("qa.json"),
            serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let count = |status| {
            self.scientific
                .cases
                .iter()
                .filter(|c| c.passed == status)
                .count()
        };
        let mut section = format!(
            "\n<!-- native-quality -->\n## Quality\n\n{} pass · {} fail · {} informational. [Full diagnostic report](qa.html), [measurements](qa.json).\n\n50,000 off-grid samples; sequential-chain maximum encoded RGB error {:.6}, RMS {:.6}. These measure LUT sampling error, independently of the GPU numerical tolerance.\n\n| Test | Status | Measurements |\n| --- | --- | --- |\n",
            count(Some(true)),
            count(Some(false)),
            count(None),
            self.max_absolute_error,
            self.rms_error
        );
        for case in &self.scientific.cases {
            let status = match case.passed {
                Some(true) => "✓ PASS",
                Some(false) => "✗ FAIL",
                None => "· INFO",
            };
            let fields = case
                .summary
                .as_object()
                .map(|o| {
                    o.iter()
                        .filter(|(_, v)| v.is_number() || v.is_boolean() || v.is_string())
                        .take(4)
                        .map(|(k, v)| format!("{k}: {v}"))
                        .collect::<Vec<_>>()
                        .join("; ")
                })
                .unwrap_or_default();
            section.push_str(&format!(
                "| [{}](qa.html#{}) | {status} | {} |\n",
                case.name,
                case.name,
                fields.replace('|', " / ")
            ));
        }
        section.push_str("\n<!-- /native-quality -->\n");
        let readme = directory.join("README.md");
        let old = std::fs::read_to_string(&readme).map_err(|e| e.to_string())?;
        let old = old
            .split("\n<!-- native-quality -->")
            .next()
            .unwrap_or(&old);
        std::fs::write(readme, format!("{old}{section}")).map_err(|e| e.to_string())?;
        let file = directory.join("bundle.json");
        let mut meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&file).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        meta["quality"] = serde_json::json!({"samples":self.samples,"report":"qa.html","measurements":"qa.json","pass":count(Some(true)),"fail":count(Some(false)),"informational":count(None),"max_absolute_error":self.max_absolute_error,"rms_error":self.rms_error});
        std::fs::write(
            file,
            serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    }
}
