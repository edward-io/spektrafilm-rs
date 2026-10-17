//! Dump the same stress cases from independently built f64 CPU / f32 GPU frontends.
#[path = "../../spektrafilm-core/tests/support/gpu_audit_cases.rs"]
mod audit_cases;
use spektrafilm_core::{pipeline::Pipeline, profile::load_profile_by_name};
use std::{path::Path, sync::Arc};
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 3,
        "usage: frontend_accuracy OUTPUT_DIR cpu|gpu [IMAGE] [LONG_EDGE]"
    );
    let directory = Path::new(&args[1]);
    std::fs::create_dir_all(directory)?;
    let backend: Arc<dyn spektrafilm_gpu::ComputeBackend> = if args[2] == "cpu" {
        Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend)
    } else {
        anyhow::ensure!(args[2] == "gpu", "backend must be cpu or gpu");
        Arc::new(
            spektrafilm_gpu::wgpu_backend::WgpuBackend::new()
                .ok_or_else(|| anyhow::anyhow!("GPU unavailable"))?,
        )
    };
    let loaded = args
        .get(3)
        .map(|p| spektrafilm_io::load(Path::new(p), &Default::default()))
        .transpose()?;
    let image = if let Some(loaded) = &loaded {
        spektrafilm_core::resize::preview(
            &loaded.pixels,
            args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(640),
        )
    } else {
        audit_cases::image()
    };
    let data = Path::new("data");
    let mut results = vec![];
    for (index, (name, mut params)) in audit_cases::cases().into_iter().enumerate() {
        if let Some(loaded) = &loaded {
            params.io.input_color_space = loaded.color_space.clone();
            params.io.input_cctf_decoding = loaded.encoded;
            params.camera.film_format_mm = 35.;
        }
        let pipeline = Pipeline::new_with_spectral(
            load_profile_by_name(data, "kodak_portra_400")?,
            load_profile_by_name(data, "kodak_portra_endura")?,
            params,
            data,
        )
        .map_err(anyhow::Error::msg)?;
        let output = pipeline.process(image.clone(), backend.as_ref());
        let file = format!("{index:02}.f64le");
        let bytes: Vec<u8> = output
            .data
            .iter()
            .flat_map(|&v| (v as f64).to_le_bytes())
            .collect();
        std::fs::write(directory.join(&file), bytes)?;
        results.push(serde_json::json!({"case":name,"file":file,"width":output.width,"height":output.height}));
    }
    std::fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"precision":if cfg!(feature="precision-f64") {"f64"} else {"f32"},"backend":args[2],"cases":results}),
        )?,
    )?;
    Ok(())
}
