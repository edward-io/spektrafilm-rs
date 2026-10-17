//! Deterministic 0.1%-of-full-scale gate against the CPU f64 path.
#![cfg(feature = "precision-f64")]
#[path = "support/gpu_audit_cases.rs"]
mod audit_cases;
use spektrafilm_core::{pipeline::Pipeline, profile::load_profile_by_name};
#[test]
fn hdr_edges_optical_filters_and_color_stay_within_0_1_percent() {
    let Some(gpu) = spektrafilm_gpu::wgpu_backend::WgpuBackend::new() else {
        return;
    };
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let image = audit_cases::image();
    for (name, params) in audit_cases::cases() {
        let pipeline = Pipeline::new_with_spectral(
            load_profile_by_name(&data, "kodak_portra_400").unwrap(),
            load_profile_by_name(&data, "kodak_portra_endura").unwrap(),
            params,
            &data,
        )
        .unwrap();
        let reference = pipeline.process(image.clone(), &spektrafilm_gpu::cpu_backend::CpuBackend);
        let output = pipeline.process(image.clone(), &gpu);
        let max = reference
            .data
            .iter()
            .zip(&output.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        eprintln!("{name}: max {max}");
        assert!(output.data.iter().all(|v| v.is_finite()));
        assert!(max <= 0.001, "{name}: exceeded 0.1% full scale ({max})");
    }
}

#[test]
fn bundled_stocks_and_papers_stay_within_0_1_percent() {
    let Some(gpu) = spektrafilm_gpu::wgpu_backend::WgpuBackend::new() else {
        return;
    };
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let mut profiles: Vec<_> = std::fs::read_dir(data.join("profiles"))
        .unwrap()
        .map(|p| p.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    profiles.sort();
    let image = audit_cases::image();
    let mut cases = vec![];
    for path in profiles {
        let stock = path.file_stem().unwrap().to_str().unwrap();
        let p = load_profile_by_name(&data, stock).unwrap();
        if p.info.support == "film" && p.info.stage != "printing" {
            let paper = p
                .info
                .target_print
                .clone()
                .unwrap_or_else(|| "kodak_portra_endura".into());
            cases.push((stock.to_owned(), paper.clone(), p.is_positive(), None));
            if !p.is_positive() {
                cases.push((stock.to_owned(), paper.clone(), true, None));
            }
            if p.data.development_time.len() > 1 {
                for &time in [
                    p.data.development_time.first().unwrap(),
                    p.data.development_time.last().unwrap(),
                ] {
                    cases.push((stock.to_owned(), paper.clone(), false, Some(time)));
                }
            }
        } else {
            cases.push(("kodak_portra_400".into(), stock.to_owned(), false, None));
        }
    }
    let mut failures = vec![];
    for (film, paper, scan_film, development_time) in cases {
        let mut params = audit_cases::cases().remove(0).1;
        params.io.scan_film = scan_film;
        params.film_render.development_time = development_time;
        let pipeline = Pipeline::new_with_spectral(
            load_profile_by_name(&data, &film).unwrap(),
            load_profile_by_name(&data, &paper).unwrap(),
            params,
            &data,
        )
        .unwrap();
        let reference = pipeline.process(image.clone(), &spektrafilm_gpu::cpu_backend::CpuBackend);
        let output = pipeline.process(image.clone(), &gpu);
        let max = reference
            .data
            .iter()
            .zip(&output.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        let finite = output.data.iter().all(|v| v.is_finite());
        eprintln!(
            "stock {film} -> {paper}, scan {scan_film}, development {development_time:?}: max {max}"
        );
        if !finite || max > 0.001 {
            failures.push((film, paper, scan_film, development_time, max));
        }
    }
    assert!(
        failures.is_empty(),
        "bundled-stock GPU errors: {failures:?}"
    );
}
