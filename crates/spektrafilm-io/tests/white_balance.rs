use spektrafilm_io::{RawOptions, WhiteBalance, load};
use std::path::Path;

fn check(path: &Path, decoder: &str) {
    let mut options = RawOptions {
        decoder: decoder.into(),
        ..Default::default()
    };
    let original = load(path, &options).unwrap();
    let base = original
        .metadata
        .white_balance
        .expect("camera white balance coordinates");
    eprintln!("{} {decoder}: {base:?}", path.display());
    assert!((1667. ..=25000.).contains(&base.temperature));
    assert!(base.tint.is_finite());
    let mean = |image: &spektrafilm_math::image::ImageBuf, c| {
        image.pixels().map(|p| p[c] as f64).sum::<f64>() / image.pixel_count() as f64
    };
    let baseline_rb = mean(&original.pixels, 0) / mean(&original.pixels, 2);
    for delta in [-500., 1., 500.] {
        let target = WhiteBalance {
            temperature: base.temperature + delta,
            ..base
        };
        options.white_balance_shift = base.shift_to(target);
        let changed = load(path, &options).unwrap();
        let rb = mean(&changed.pixels, 0) / mean(&changed.pixels, 2);
        assert!(
            (rb - baseline_rb) * delta > 0.,
            "temperature direction: {rb} vs {baseline_rb}"
        );
        if delta == 1. {
            let error = changed
                .pixels
                .data
                .iter()
                .zip(&original.pixels.data)
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .fold(0., f64::max);
            assert!(error < 0.01, "first adjustment jumped: {error}");
        }
    }
    let baseline_green =
        mean(&original.pixels, 1) / (mean(&original.pixels, 0) + mean(&original.pixels, 2));
    for tint in [-10., 10.] {
        options.white_balance_shift = [0., tint];
        let changed = load(path, &options).unwrap();
        let green =
            mean(&changed.pixels, 1) / (mean(&changed.pixels, 0) + mean(&changed.pixels, 2));
        assert!((green - baseline_green) * tint < 0., "tint direction");
    }
    options.white_balance_shift = [0.; 2];
    let reset = load(path, &options).unwrap();
    assert_eq!(
        reset.pixels.data, original.pixels.data,
        "reset is exactly As Shot"
    );
    options.white_balance_shift = [f64::NAN, 0.];
    assert!(load(path, &options).is_err());
}

#[test]
fn sliders_preserve_as_shot_and_adjust_continuously() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/synthetic.dng");
    check(&path, "rawler");
    if std::env::var_os("SPEKTRAFILM_LIBRAW_LIBRARY").is_some() {
        check(&path, "libraw");
    }
}

#[test]
fn existing_modes_and_saved_options_keep_their_meaning() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/synthetic.dng");
    let old = r#"{"decoder":"rawler","white_balance":"custom","temperature":4300.0,"tint":1.1}"#;
    let mut options: RawOptions = serde_json::from_str(old).unwrap();
    assert_eq!(options.white_balance_shift, [0.; 2]);
    let baseline = load(&path, &options).unwrap();
    options.white_balance_shift = [-15., 8.];
    let encoded = serde_json::to_string(&options).unwrap();
    let restored = serde_json::from_str(&encoded).unwrap();
    assert_eq!(options, restored);
    assert_eq!(
        load(&path, &options).unwrap().pixels.data,
        load(&path, &restored).unwrap().pixels.data
    );
    options.white_balance_shift = [0.; 2];
    assert_eq!(
        load(&path, &options).unwrap().pixels.data,
        baseline.pixels.data
    );
}

#[test]
#[ignore = "requires SPEKTRAFILM_WB_IMAGE with a local RAW"]
fn camera_raw_white_balance() {
    let path = std::env::var_os("SPEKTRAFILM_WB_IMAGE").unwrap();
    check(Path::new(&path), "rawler");
    check(Path::new(&path), "libraw");
}
