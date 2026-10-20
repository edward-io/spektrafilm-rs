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

#[test]
fn adjusted_white_balance_is_bit_identical_to_serial_transform() {
    if std::env::var_os("SPEKTRAFILM_LIBRAW_LIBRARY").is_none() {
        return;
    }
    use spektrafilm_math::{colorspace, illuminants::white_xyz, precision::from_f64, transfer};
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/synthetic.dng");
    let baseline = load(
        &path,
        &RawOptions {
            decoder: "libraw".into(),
            white_balance: "daylight".into(),
            ..Default::default()
        },
    )
    .unwrap();
    let cs = transfer::get("ACES2065-1").unwrap();
    for mode in ["custom", "tungsten"] {
        for temperature in [2850., 4500., 8000.] {
            let options = RawOptions {
                decoder: "libraw".into(),
                white_balance: mode.into(),
                temperature,
                tint: 1.3,
                ..Default::default()
            };
            let matrix = colorspace::chromatic_adaptation_matrix_f64(
                white_xyz(if mode == "tungsten" {
                    2850.
                } else {
                    temperature
                }),
                white_xyz(6504.),
            );
            let mut expected = baseline.pixels.clone();
            for p in expected.pixels_mut() {
                let mul = |m: &[[f64; 3]; 3], v: [f64; 3]| {
                    std::array::from_fn::<_, 3, _>(|i| {
                        m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2]
                    })
                };
                let rgb = [p[0] as f64, p[1] as f64, p[2] as f64];
                let rgb = mul(&cs.from_xyz, mul(&matrix, mul(&cs.to_xyz, rgb)));
                for c in 0..3 {
                    p[c] = from_f64(rgb[c] as f32 as f64);
                }
                if mode == "custom" {
                    p[1] = from_f64((p[1] as f32 * options.tint as f32) as f64);
                }
            }
            let actual = load(&path, &options).unwrap().pixels;
            assert_eq!(
                (actual.width, actual.height),
                (expected.width, expected.height)
            );
            assert!(
                actual
                    .data
                    .iter()
                    .zip(&expected.data)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{mode}, {temperature} changed a sample"
            );
        }
    }
}
