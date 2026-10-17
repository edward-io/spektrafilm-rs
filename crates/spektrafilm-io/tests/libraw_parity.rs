use spektrafilm_io::{RawOptions, load};
#[test]
fn libraw_matches_python_rawpy() {
    // Opt in with an installed library; normal builds remain dependency-free.
    if std::env::var_os("SPEKTRAFILM_LIBRAW_LIBRARY").is_none() {
        return;
    }
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/libraw.json")).unwrap();
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/synthetic.dng");
    for case in fixture.as_array().unwrap() {
        let options = RawOptions {
            decoder: "libraw".into(),
            white_balance: case["mode"].as_str().unwrap().into(),
            temperature: 4500.,
            tint: 1.1,
            ..Default::default()
        };
        let loaded = load(&path, &options).unwrap();
        assert_eq!(loaded.color_space, "ACES2065-1");
        assert_eq!(loaded.pixels.width as u64, case["width"].as_u64().unwrap());
        assert_eq!(
            loaded.pixels.height as u64,
            case["height"].as_u64().unwrap()
        );
        for (i, (got, want)) in loaded
            .pixels
            .data
            .iter()
            .zip(case["expected"].as_array().unwrap())
            .enumerate()
        {
            let want = want.as_f64().unwrap();
            assert!(
                (*got as f64 - want).abs() < 2e-6,
                "{} [{i}]: {got} vs {want}",
                options.white_balance
            );
        }
    }
}
