//! Deterministic CPU-f64/GPU audit. Run with --features precision-f64.
#[path = "../../spektrafilm-core/tests/support/gpu_audit_cases.rs"]
mod audit_cases;
use serde::Serialize;
use spektrafilm_core::{pipeline::Pipeline, profile::load_profile_by_name};
use std::{path::Path, time::Instant};
#[derive(Serialize)]
struct Measurement {
    case: String,
    resident: bool,
    max_absolute: f64,
    mean_absolute: f64,
    rms: f64,
    p99: f64,
    over_0_1_percent: usize,
    values: usize,
    cpu_ms: f64,
    gpu_ms: f64,
}
fn main() -> anyhow::Result<()> {
    anyhow::ensure!(
        cfg!(feature = "precision-f64"),
        "build this audit with --features precision-f64"
    );
    let gpu = spektrafilm_gpu::wgpu_backend::WgpuBackend::new()
        .ok_or_else(|| anyhow::anyhow!("GPU unavailable"))?;
    let cpu = spektrafilm_gpu::cpu_backend::CpuBackend;
    let args: Vec<_> = std::env::args().collect();
    let loaded = args
        .get(2)
        .map(|p| spektrafilm_io::load(Path::new(p), &Default::default()))
        .transpose()?;
    let image = if let Some(image) = &loaded {
        spektrafilm_core::resize::preview(
            &image.pixels,
            args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(640),
        )
    } else {
        audit_cases::image()
    };
    let (w, _h) = (image.width, image.height);
    let mut cases = audit_cases::cases();
    if let Some(loaded) = &loaded {
        for (_, params) in &mut cases {
            params.io.input_color_space = loaded.color_space.clone();
            params.io.input_cctf_decoding = loaded.encoded;
            params.camera.film_format_mm = 35.;
        }
    }

    let data = Path::new("data");
    let mut results = vec![];
    for (name, params) in cases {
        let pipeline = Pipeline::new_with_spectral(
            load_profile_by_name(data, "kodak_portra_400")?,
            load_profile_by_name(data, "kodak_portra_endura")?,
            params,
            data,
        )
        .map_err(anyhow::Error::msg)?;
        let t = Instant::now();
        let reference = pipeline.process(image.clone(), &cpu);
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.;
        let t = Instant::now();
        let resident = pipeline.process_resident_borrowed(&image, &gpu);
        let was_resident = resident.is_some();
        let output = resident.unwrap_or_else(|| pipeline.process(image.clone(), &gpu));
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.;
        let worst = (0..reference.data.len())
            .max_by(|a, b| {
                ((reference.data[*a] - output.data[*a]).abs())
                    .total_cmp(&(reference.data[*b] - output.data[*b]).abs())
            })
            .unwrap();
        eprintln!(
            "  worst pixel ({},{}) channel {}: CPU {} GPU {}",
            (worst / 3) % w as usize,
            (worst / 3) / w as usize,
            worst % 3,
            reference.data[worst],
            output.data[worst]
        );
        let mut errors: Vec<_> = reference
            .data
            .iter()
            .zip(&output.data)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .collect();
        anyhow::ensure!(
            errors.iter().all(|v| v.is_finite()),
            "{name}: nonfinite output"
        );
        errors.sort_by(f64::total_cmp);
        let row = Measurement {
            case: name,
            resident: was_resident,
            max_absolute: *errors.last().unwrap(),
            mean_absolute: errors.iter().sum::<f64>() / errors.len() as f64,
            rms: (errors.iter().map(|v| v * v).sum::<f64>() / errors.len() as f64).sqrt(),
            p99: errors[errors.len() * 99 / 100],
            over_0_1_percent: errors.iter().filter(|v| **v > 0.001).count(),
            values: errors.len(),
            cpu_ms,
            gpu_ms,
        };
        eprintln!(
            "{}: max {:.6}, p99 {:.6}, RMS {:.6}, >0.1% {} / {}, resident {}, CPU {:.1} ms GPU {:.1} ms",
            row.case,
            row.max_absolute,
            row.p99,
            row.rms,
            row.over_0_1_percent,
            row.values,
            row.resident,
            row.cpu_ms,
            row.gpu_ms
        );
        results.push(row);
    }
    let path = std::env::args()
        .nth(1)
        .unwrap_or("/tmp/spektrafilm-gpu-accuracy.json".into());
    std::fs::write(path, serde_json::to_vec_pretty(&results)?)?;
    anyhow::ensure!(
        results.iter().all(|r| r.max_absolute <= 0.001),
        "GPU error exceeded 0.1% full scale; see the JSON report"
    );
    Ok(())
}
