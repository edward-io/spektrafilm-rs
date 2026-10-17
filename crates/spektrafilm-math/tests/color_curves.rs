#[test]
fn named_color_space_curves_match_colour_science() {
    let cases: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/color_curves.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let cs = spektrafilm_math::transfer::get(name).unwrap();
        assert!(cs.transfer.supported(), "{name}");
        for (i, input) in case["input"].as_array().unwrap().iter().enumerate() {
            let x = input.as_f64().unwrap();
            for (got, key) in [
                (cs.transfer.encode(x), "encode"),
                (cs.transfer.decode(x), "decode"),
            ] {
                let want = case[key][i].as_f64().unwrap();
                assert!(
                    (got - want).abs() < 1e-7 * (1. + want.abs()),
                    "{name} {key}({x}): {got} vs {want}"
                );
            }
        }
    }
}
