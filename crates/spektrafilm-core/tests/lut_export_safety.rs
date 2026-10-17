use spektrafilm_core::{
    bundle_output::OutputDirectory,
    lut_bundle::write_bundle,
    lut_export::{self, BakeOptions},
    params::RuntimeParams,
};
use std::{fs, path::Path};

#[test]
fn invalid_luts_do_not_touch_existing_output() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("existing.cube");
    for (size, format, input) in [
        (18, "lumix", "vlog"),
        (17, "lumix", "srgb"),
        (5, "hald_png", "srgb"),
        (3, "unknown", "srgb"),
    ] {
        fs::write(&path, b"previous good LUT").unwrap();
        assert!(
            lut_export::write_lut(&lut_export::grid(size), size, &path, format, "test", input)
                .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), b"previous good LUT");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }
    assert!(lut_export::write_lut(&lut_export::grid(2), 3, &path, "cube", "test", "srgb").is_err());
    assert_eq!(fs::read(&path).unwrap(), b"previous good LUT");
    lut_export::write_lut(&lut_export::grid(2), 2, &path, "cube", "test", "srgb").unwrap();
    assert!(fs::read_to_string(path).unwrap().contains("LUT_3D_SIZE 2"));
}

#[test]
fn lumix_bundles_require_a_single_complete_transform() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bundle");
    let options = BakeOptions {
        film: "kodak_portra_400",
        paper: "kodak_portra_endura",
        size: 17,
        input: "vlog",
        output: "srgb",
        gain: None,
        params: RuntimeParams::default(),
    };
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    for topology in ["2lut", "3lut", "4lut"] {
        assert!(
            write_bundle(&options, &data, &path, topology, "lumix", false)
                .err()
                .unwrap()
                .contains("1lut")
        );
        assert!(!path.exists());
    }
    write_bundle(&options, &data, &path, "1lut", "lumix", false).unwrap();
    let source = fs::read_to_string(path.join("l1_rgb_in_rgb_out.cube")).unwrap();
    assert!(source.contains("#LUMIXPHOTOSTYLE VLOG"));
}

#[test]
fn bundle_publication_replaces_only_owned_directories_and_restores_on_failure() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bundle");
    fs::create_dir(&path).unwrap();
    fs::write(
        path.join("bundle.json"),
        br#"{"luts":[{"file":"old.cube"}]}"#,
    )
    .unwrap();
    fs::write(path.join("old.cube"), b"old").unwrap();
    fs::write(path.join("qa.html"), b"stale").unwrap();
    let staged = OutputDirectory::new(&path).unwrap();
    fs::remove_dir_all(staged.path()).unwrap();
    assert!(staged.commit().is_err());
    assert_eq!(fs::read(path.join("old.cube")).unwrap(), b"old");
    let staged = OutputDirectory::new(&path).unwrap();
    fs::write(
        staged.path().join("bundle.json"),
        br#"{"luts":[{"file":"new.cube"}]}"#,
    )
    .unwrap();
    fs::write(staged.path().join("new.cube"), b"new").unwrap();
    staged.commit().unwrap();
    assert!(!path.join("old.cube").exists());
    assert!(!path.join("qa.html").exists());
    fs::write(path.join("private.txt"), b"unrelated").unwrap();
    assert!(OutputDirectory::new(&path).is_err());
    assert_eq!(fs::read(path.join("private.txt")).unwrap(), b"unrelated");
    assert_eq!(fs::read(path.join("new.cube")).unwrap(), b"new");
}

#[test]
fn multi_paper_replacement_checks_nested_files_and_manifest_paths() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bundle");
    let paper = path.join("paper_one");
    fs::create_dir_all(&paper).unwrap();
    fs::write(
        path.join("spec.json"),
        br#"{"print_profiles":["paper_one","paper_two"]}"#,
    )
    .unwrap();
    fs::write(
        paper.join("bundle.json"),
        br#"{"luts":[{"file":"old.cube"}]}"#,
    )
    .unwrap();
    fs::write(paper.join("old.cube"), b"old").unwrap();
    fs::write(paper.join("private.txt"), b"private notes").unwrap();
    assert!(OutputDirectory::new(&path).is_err());
    assert_eq!(
        fs::read(paper.join("private.txt")).unwrap(),
        b"private notes"
    );
    fs::remove_file(paper.join("private.txt")).unwrap();

    fs::write(
        paper.join("bundle.json"),
        br#"{"luts":[{"file":"../private.txt"}]}"#,
    )
    .unwrap();
    assert!(OutputDirectory::new(&path).is_err());
    fs::write(
        paper.join("bundle.json"),
        br#"{"luts":[{"file":"old.cube"}]}"#,
    )
    .unwrap();
    let staged = OutputDirectory::new(&path).unwrap();
    fs::write(staged.path().join("bundle.json"), br#"{"luts":[]}"#).unwrap();
    staged.commit().unwrap();
    assert!(!paper.exists());
    assert_eq!(fs::read_dir(&path).unwrap().count(), 1);
}
