//! Optional real-camera regression: keep personal photographs out of the repository.
use spektrafilm_io::{RawOptions, load};
#[test]
fn camera_raw_opens_with_both_decoders() {
    let Some(path) = std::env::var_os("SPEKTRAFILM_RAW_TEST_IMAGE") else {
        return;
    };
    for decoder in ["rawler", "libraw", "auto"] {
        let loaded = load(
            std::path::Path::new(&path),
            &RawOptions {
                decoder: decoder.into(),
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("{decoder}: {e:#}"));
        assert!(!loaded.encoded);
        assert!(loaded.pixels.width > 0 && loaded.pixels.height > 0);
        assert!(loaded.pixels.data.iter().all(|v| v.is_finite()));
        assert!(loaded.pixels.data.iter().any(|v| *v > 0.));
        eprintln!(
            "{decoder}: {} × {}, linear {}",
            loaded.pixels.width, loaded.pixels.height, loaded.color_space
        );
    }
}
