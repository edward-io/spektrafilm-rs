//! Exercise the GUI/CLI RAW-loading path with Lensfun enabled and no override.
fn main() -> anyhow::Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .ok_or_else(|| {
            anyhow::anyhow!("usage: check_lensfun_raw IMAGE.RAW [auto|rawler|libraw]")
        })?;
    let decoder = std::env::args().nth(2).unwrap_or_else(|| "auto".into());
    let start = std::time::Instant::now();
    let loaded = spektrafilm_io::load(
        &path,
        &spektrafilm_io::RawOptions {
            decoder: decoder.clone(),
            lens_correction: true,
            ..Default::default()
        },
    )?;
    anyhow::ensure!(
        loaded.pixels.data.iter().all(|v| v.is_finite()),
        "nonfinite decoded pixels"
    );
    println!(
        "{decoder}: loaded {}×{} {} with lens correction enabled in {:.2}s",
        loaded.pixels.width,
        loaded.pixels.height,
        loaded.color_space,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
