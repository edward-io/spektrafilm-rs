//! Write the GUI's built-in RAW development as a linear, 32-bit TIFF.
use anyhow::{Result, anyhow};
use spektrafilm_io::{RawOptions, SaveOptions, load, save};
use std::path::Path;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args
        .next()
        .ok_or_else(|| anyhow!("usage: decode_raw_gui <input.raw> <output.tif>"))?;
    let output = args
        .next()
        .ok_or_else(|| anyhow!("usage: decode_raw_gui <input.raw> <output.tif>"))?;
    let loaded = load(
        Path::new(&input),
        &RawOptions {
            decoder: "rawler".into(),
            ..Default::default()
        },
    )?;
    let options = SaveOptions {
        color_space: loaded.color_space,
        encoded: false,
        bit_depth: 32,
        ..Default::default()
    };
    save(
        &loaded.pixels,
        Path::new(&output),
        &options,
        &loaded.metadata,
        Path::new("data"),
    )
}
