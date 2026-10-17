use spektrafilm_core::{params::RuntimeParams, pipeline::Pipeline, profile::load_profile_by_name};
use spektrafilm_io::{RawOptions, load};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_spektrafilm"))
        .args(args)
        .env("SPEKTRAFILM_BACKEND", "cpu")
        .current_dir(root())
        .output()
        .unwrap()
}
fn success(result: std::process::Output) {
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn process_exports_imported_lightroom_crop_and_rotation() {
    let dir = tempfile::tempdir().unwrap();
    let input = root().join("crates/spektrafilm-io/tests/fixtures/synthetic.dng");
    let output = dir.path().join("cropped.tif");
    let settings = dir.path().join("settings.json");
    let mut params = RuntimeParams::default();
    params.debug.deactivate_spatial_effects = true;
    params.debug.deactivate_stochastic_effects = true;
    fs::write(&settings, serde_json::to_vec(&params).unwrap()).unwrap();
    success(cli(&[
        "process",
        input.to_str().unwrap(),
        "--film",
        "kodak_gold_200",
        "--paper",
        "kodak_portra_endura",
        "--raw-decoder",
        "rawler",
        "--lightroom-geometry",
        r#"{"orientation":6,"crop":[0.25,0,1,0.5],"angle":0}"#,
        "--params",
        settings.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ]));
    let image = load(&output, &RawOptions::default()).unwrap().pixels;
    assert_eq!((image.width, image.height), (16, 24));
}

#[test]
fn raw_processing_honors_explicit_input_fields_and_detects_unspecified_fields() {
    let dir = tempfile::tempdir().unwrap();
    let data = root().join("data");
    let input = root().join("crates/spektrafilm-io/tests/fixtures/synthetic.dng");
    let loaded = load(
        &input,
        &RawOptions {
            decoder: "rawler".into(),
            ..Default::default()
        },
    )
    .unwrap();
    for (space_override, encoding_override) in
        [(true, true), (true, false), (false, true), (false, false)]
    {
        let mut params = RuntimeParams::default();
        params.camera.auto_exposure = false;
        params.debug.deactivate_spatial_effects = true;
        params.debug.deactivate_stochastic_effects = true;
        params.io.input_color_space = if space_override {
            "ITU-R BT.2020".into()
        } else {
            loaded.color_space.clone()
        };
        params.io.input_cctf_decoding = encoding_override;
        let mut settings = serde_json::to_value(&params).unwrap();
        if !space_override && !encoding_override {
            settings.as_object_mut().unwrap().remove("io");
        } else {
            if !space_override {
                settings["io"]
                    .as_object_mut()
                    .unwrap()
                    .remove("input_color_space");
            }
            if !encoding_override {
                settings["io"]
                    .as_object_mut()
                    .unwrap()
                    .remove("input_cctf_decoding");
            }
        }
        let config = dir.path().join("settings.json");
        fs::write(&config, serde_json::to_vec(&settings).unwrap()).unwrap();
        let raw = dir.path().join("output.f64");
        let output = dir.path().join("output.png");
        success(cli(&[
            "process",
            input.to_str().unwrap(),
            "--film",
            "kodak_portra_400",
            "--paper",
            "kodak_portra_endura",
            "--raw-decoder",
            "rawler",
            "--params",
            config.to_str().unwrap(),
            "--raw-out",
            raw.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ]));
        let reference = Pipeline::new_with_spectral(
            load_profile_by_name(&data, "kodak_portra_400").unwrap(),
            load_profile_by_name(&data, "kodak_portra_endura").unwrap(),
            params,
            &data,
        )
        .unwrap()
        .process(
            loaded.pixels.clone(),
            &spektrafilm_gpu::cpu_backend::CpuBackend,
        );
        let actual = fs::read(raw).unwrap();
        assert_eq!(actual.len(), reference.data.len() * 8);
        for (bytes, expected) in actual.chunks_exact(8).zip(reference.data) {
            let actual = f64::from_le_bytes(bytes.try_into().unwrap());
            assert!(
                (actual - expected as f64).abs() < 1e-7,
                "overrides ({space_override}, {encoding_override}): {actual} / {expected}"
            );
        }
    }
}

#[test]
fn rebuilding_bundle_zip_removes_stale_outputs_and_rejects_unrelated_material() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("bundle");
    let spec = dir.path().join("spec.json");
    let write_spec = |size| {
        fs::write(&spec,serde_json::to_vec(&serde_json::json!({"resolution":size,"input_color_space":"vlog","output_color_space":"srgb","container":"zip"})).unwrap()).unwrap()
    };
    write_spec(3);
    success(cli(&[
        "build-luts",
        "--from",
        spec.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ]));
    fs::write(output.join("qa.json"), b"old diagnostics").unwrap();
    fs::write(output.join("qa.html"), b"old report").unwrap();
    fs::create_dir(output.join("qa")).unwrap();
    fs::write(output.join("qa/old.svg"), b"old plot").unwrap();
    write_spec(5);
    success(
        Command::new(env!("CARGO_BIN_EXE_spektrafilm"))
            .args([
                "build-luts",
                "--from",
                spec.to_str().unwrap(),
                "-o",
                ".",
                "--data-dir",
                root().join("data").to_str().unwrap(),
            ])
            .env("SPEKTRAFILM_BACKEND", "cpu")
            .current_dir(&output)
            .output()
            .unwrap(),
    );
    assert!(!output.join("qa.json").exists());
    assert!(!output.join("qa.html").exists());
    assert!(!output.join("qa").exists());
    let archive = output.with_extension("zip");
    let zip = zip::ZipArchive::new(fs::File::open(&archive).unwrap()).unwrap();
    assert!(!zip.file_names().any(|name| name.starts_with("qa")));
    let old_archive = fs::read(&archive).unwrap();
    fs::write(output.join("unrelated.txt"), b"private notes").unwrap();
    let result = cli(&[
        "build-luts",
        "--from",
        spec.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
    ]);
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("unrelated"));
    assert_eq!(
        fs::read(output.join("unrelated.txt")).unwrap(),
        b"private notes"
    );
    assert_eq!(fs::read(archive).unwrap(), old_archive);
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 3);
}

#[test]
fn raw_comparison_helper_uses_gui_orientation_and_color() {
    let dir = tempfile::tempdir().unwrap();
    let mut raw = include_bytes!("../../spektrafilm-io/tests/fixtures/synthetic.dng").to_vec();
    let offset = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
    let count = u16::from_le_bytes(raw[offset..offset + 2].try_into().unwrap()) as usize;
    let mut entries: Vec<[u8; 12]> = raw[offset + 2..offset + 2 + count * 12]
        .chunks_exact(12)
        .map(|e| e.try_into().unwrap())
        .collect();
    for (tag, kind, values) in [
        (274u16, 3u16, vec![6u32]),
        (50719, 4, vec![2, 4]),
        (50720, 4, vec![28, 24]),
    ] {
        let mut entry = [0u8; 12];
        entry[..2].copy_from_slice(&tag.to_le_bytes());
        entry[2..4].copy_from_slice(&kind.to_le_bytes());
        entry[4..8].copy_from_slice(&(values.len() as u32).to_le_bytes());
        let value = if values.len() == 1 {
            values[0]
        } else {
            let pos = raw.len() as u32;
            for v in &values {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            pos
        };
        entry[8..12].copy_from_slice(&value.to_le_bytes());
        entries.retain(|e| u16::from_le_bytes(e[..2].try_into().unwrap()) != tag);
        entries.push(entry);
    }
    entries.sort_by_key(|e| u16::from_le_bytes(e[..2].try_into().unwrap()));
    let directory = raw.len() as u32;
    raw[4..8].copy_from_slice(&directory.to_le_bytes());
    raw.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for entry in entries {
        raw.extend_from_slice(&entry);
    }
    raw.extend_from_slice(&0u32.to_le_bytes());
    let input = dir.path().join("portrait.dng");
    fs::write(&input, raw).unwrap();
    let output = dir.path().join("decoded.tif");
    success(
        Command::new(env!("CARGO_BIN_EXE_decode_raw_gui"))
            .arg(&input)
            .arg(&output)
            .current_dir(root())
            .output()
            .unwrap(),
    );
    let reference = load(
        &input,
        &RawOptions {
            decoder: "rawler".into(),
            ..Default::default()
        },
    )
    .unwrap();
    let actual = load(&output, &RawOptions::default()).unwrap();
    assert_eq!((actual.pixels.width, actual.pixels.height), (24, 28));
    assert_eq!(actual.color_space, reference.color_space);
    assert!(!actual.encoded);
    assert_eq!(actual.pixels.data, reference.pixels.data);
}
