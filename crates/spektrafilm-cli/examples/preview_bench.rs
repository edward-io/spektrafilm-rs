//! Profile preview stages on a real image. Set SPEKTRAFILM_STAGE_TIMINGS=1.
use spektrafilm_core::{params::RuntimeParams, pipeline::Pipeline, profile::load_profile_by_name};
use std::{path::Path, time::Instant};
fn main() -> anyhow::Result<()> {
    let _=tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_|"warn".into())).try_init();
    let args: Vec<_> = std::env::args().collect();
    let path = Path::new(args.get(1).expect("preview_bench <image> [long-edge=2048]"));
    let size: u32 = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(2048);
    let iterations: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(5);
    let t = Instant::now();
    let loaded = spektrafilm_io::load(path, &Default::default())?;
    eprintln!("Decode: {:.1} ms", t.elapsed().as_secs_f64() * 1000.);
    let t = Instant::now();
    let input = spektrafilm_core::resize::preview(&loaded.pixels, size);
    eprintln!(
        "Resize: {:.1} ms, {}x{}",
        t.elapsed().as_secs_f64() * 1000.,
        input.width,
        input.height
    );
    let backend = spektrafilm_gpu::select_backend();
    eprintln!("Backend: {}", backend.name());
    let mut params = RuntimeParams::default();
    params.io.input_color_space = loaded.color_space;
    params.io.input_cctf_decoding = loaded.encoded;
    let dir = Path::new("data");
    let t = Instant::now();
    let pipeline = Pipeline::new_with_spectral(
        load_profile_by_name(dir, "kodak_gold_200")?,
        load_profile_by_name(dir, "kodak_portra_endura")?,
        params.clone(),
        dir,
    )
    .map_err(anyhow::Error::msg)?;
    eprintln!("Build: {:.1} ms", t.elapsed().as_secs_f64() * 1000.);
    for mode in ["layered", "simple", "off"] {
        params.film_render.grain.active = mode != "off";
        params.film_render.grain.sublayers_active = mode == "layered";
        let pipeline = pipeline.clone().with_params(params.clone());
        for iteration in 0..iterations {
            let t = Instant::now();
            let output = pipeline.process(input.clone(), backend.as_ref());
            eprintln!(
                "BENCH {mode} {iteration}: {:.1} ms, {}x{}, finite={}",
                t.elapsed().as_secs_f64() * 1000.,
                output.width,
                output.height,
                output.data.iter().all(|x| x.is_finite())
            );
        }
    }
    Ok(())
}
