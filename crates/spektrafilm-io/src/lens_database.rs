//! Automatic Lensfun 0.3 database discovery and first-use installation.
//! The upstream updater distributes versioned, backwards-compatible archives:
//! https://github.com/lensfun/lensfun/blob/master/apps/lensfun-update-data
use anyhow::{Context, Result, bail};
use quick_xml::{Reader, events::Event};
use std::{
    fs,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

pub const DOWNLOAD_URL: &str = "https://lensfun.github.io/db/version_1.tar.bz2";
const MAX_DOWNLOAD: u64 = 8 * 1024 * 1024;
const MAX_EXPANDED: u64 = 32 * 1024 * 1024;
const MAX_FILE: u64 = 2 * 1024 * 1024;
const CURRENT: &str = "current-v1.txt";
static INSTALL_LOCK: Mutex<()> = Mutex::new(());

pub fn cache_root() -> Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
    let base = if cfg!(target_os = "windows") {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .or_else(|| home.as_ref().map(|p| p.join("AppData/Local")))
    } else if cfg!(target_os = "macos") {
        home.as_ref().map(|p| p.join("Library/Application Support"))
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| home.as_ref().map(|p| p.join(".local/share")))
    }
    .context("Cannot locate the user data directory for the Lensfun database")?;
    Ok(base.join("spektrafilm-rs/lensfun"))
}
fn xml_version(bytes: &[u8], complete: bool) -> Result<usize> {
    let mut reader = Reader::from_reader(bytes);
    let (mut depth, mut roots, mut profiles) = (0usize, 0usize, 0usize);
    loop {
        match reader
            .read_event()
            .context("Malformed Lensfun database XML")?
        {
            Event::Start(e) | Event::Empty(e) if depth == 0 => {
                if roots != 0 || e.name().as_ref() != b"lensdatabase" {
                    bail!("Expected a Lensfun lensdatabase XML root");
                }
                roots += 1;
                let version = e
                    .try_get_attribute("version")?
                    .map(|a| a.unescape_value().map(|s| s.into_owned()))
                    .transpose()?
                    .unwrap_or_else(|| "0".into())
                    .parse::<u32>()?;
                if version > 1 {
                    bail!("Lensfun 0.3 requires database format 0 or 1; found {version}");
                }
                if !complete {
                    return Ok(0);
                }
                // A root must have contents; an empty root cannot supply profiles.
                depth = 1;
            }
            Event::Start(e) => {
                if matches!(e.name().as_ref(), b"camera" | b"lens") {
                    profiles += 1;
                }
                depth += 1;
            }
            Event::End(_) => {
                depth = depth
                    .checked_sub(1)
                    .context("Unexpected Lensfun XML closing element")?;
            }
            Event::DocType(_) => bail!("Unexpected document type in Lensfun database"),
            Event::Text(text) if depth == 0 && !text.unescape()?.trim().is_empty() => {
                bail!("Unexpected text outside the Lensfun database XML root");
            }
            Event::CData(_) if depth == 0 => bail!("Unexpected data outside the Lensfun XML root"),
            Event::Eof => break,
            _ => {}
        }
    }
    if roots != 1 || depth != 0 {
        bail!("Incomplete Lensfun database XML");
    }
    Ok(profiles)
}
/// Accept a folder containing XML or a standard parent with version_1 beneath it.
fn compatible_directory(path: &Path) -> Option<PathBuf> {
    for candidate in [path.to_path_buf(), path.join("version_1")] {
        let Ok(entries) = fs::read_dir(&candidate) else {
            continue;
        };
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|v| v == "xml") {
                let Ok(file) = fs::File::open(entry.path()) else {
                    continue;
                };
                let mut prefix = Vec::new();
                if file.take(4096).read_to_end(&mut prefix).is_ok()
                    && xml_version(&prefix, false).is_ok()
                {
                    return Some(candidate);
                }
            }
        }
    }
    None
}
fn cached_at(root: &Path) -> Option<PathBuf> {
    let name = fs::read_to_string(root.join(CURRENT)).ok()?;
    let name = name.trim();
    if name.is_empty()
        || !name.starts_with("version_1-")
        || Path::new(name).components().count() != 1
        || !matches!(
            Path::new(name).components().next(),
            Some(Component::Normal(_))
        )
    {
        return None;
    }
    let path = root.join(name);
    if !path.join(".complete").is_file() {
        return None;
    }
    compatible_directory(&path)
}
pub fn cached() -> Option<PathBuf> {
    cache_root().ok().and_then(|p| cached_at(&p))
}
/// Explicit and environment overrides are authoritative; an invalid override is
/// reported instead of silently replacing the user's database.
pub fn discover(explicit: Option<&Path>) -> Result<Option<PathBuf>> {
    let env = std::env::var_os("SPEKTRAFILM_LENSFUN_DATABASE").map(PathBuf::from);
    if let Some(path) = explicit.or(env.as_deref()) {
        return compatible_directory(path).map(Some).with_context(||format!("No Lensfun 0.3-compatible XML database in {}. Choose a version_1 folder or use the automatic database.",path.display()));
    }
    if let Some(path) = cached() {
        return Ok(Some(path));
    }
    let mut candidates = Vec::new();
    if let Some(home) = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
    {
        if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
            let xdg = PathBuf::from(xdg);
            candidates.push(xdg.join("lensfun/updates/version_1"));
            candidates.push(xdg.join("lensfun"));
        }
        for suffix in [
            ".local/share/lensfun/updates/version_1",
            ".local/share/lensfun/version_1",
            ".local/share/lensfun",
            "Library/Application Support/lensfun/updates/version_1",
            "Library/Application Support/lensfun",
            "AppData/Local/lensfun/updates/version_1",
            "AppData/Local/lensfun",
        ] {
            candidates.push(home.join(suffix));
        }
    }
    if let Some(program) = std::env::var_os("PROGRAMDATA") {
        candidates.push(PathBuf::from(program).join("Lensfun"));
    }
    candidates.extend(
        [
            "/opt/homebrew/share/lensfun",
            "/usr/local/share/lensfun",
            "/usr/share/lensfun",
            "/var/lib/lensfun-updates",
        ]
        .map(PathBuf::from),
    );
    Ok(candidates.iter().find_map(|p| compatible_directory(p)))
}
fn install_archive(root: &Path, compressed: &[u8]) -> Result<PathBuf> {
    if compressed.len() as u64 > MAX_DOWNLOAD {
        bail!("Lensfun database download exceeds 8 MiB");
    }
    fs::create_dir_all(root)
        .with_context(|| format!("Creating Lensfun cache {}", root.display()))?;
    // Every install gets an immutable generation. Only a validated snapshot is
    // published; a partial download or another process cannot break the old one.
    let stage = tempfile::Builder::new()
        .prefix("version_1-")
        .tempdir_in(root)?;
    let expanded = bzip2::read::BzDecoder::new(compressed).take(MAX_EXPANDED + 1);
    let mut archive = tar::Archive::new(expanded);
    let (mut total, mut files, mut profiles) = (0u64, 0usize, 0usize);
    for entry in archive
        .entries()
        .context("Invalid Lensfun database archive")?
    {
        let mut entry = entry.context("Reading Lensfun database archive")?;
        if entry.header().entry_type().is_dir() {
            continue;
        }
        if !entry.header().entry_type().is_file() {
            bail!("Unsupported link or special file in Lensfun database archive");
        }
        let name = entry.path()?.into_owned();
        let parts: Vec<_> = name
            .components()
            .filter(|p| !matches!(p, Component::CurDir))
            .collect();
        if parts.len() != 1 || !matches!(parts[0], Component::Normal(_)) {
            bail!("Unsafe path in Lensfun database archive");
        }
        let filename = name
            .file_name()
            .context("Missing Lensfun archive filename")?;
        if filename
            .to_str()
            .is_none_or(|v| v.starts_with('.') || v.contains(['\\', ':']))
        {
            bail!("Invalid Lensfun archive filename");
        }
        let size = entry.size();
        total = total
            .checked_add(size)
            .context("Lensfun archive size overflow")?;
        files += 1;
        if size > MAX_FILE || total > MAX_EXPANDED || files > 512 {
            bail!("Lensfun database archive exceeds extraction limits");
        }
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)?;
        if name.extension().is_some_and(|v| v == "xml") {
            profiles += xml_version(&bytes, true)?;
        } else if !matches!(
            filename.to_str(),
            Some(
                "timestamp.txt"
                    | "COPYING"
                    | "COPYING.CC_BY-SA_3.0"
                    | "README"
                    | "README.md"
                    | "README.txt"
            )
        ) {
            bail!("Unexpected file in Lensfun database archive");
        }
        fs::write(stage.path().join(filename), bytes)?;
    }
    if profiles == 0 {
        bail!("Downloaded Lensfun database contains no camera or lens profiles");
    }
    fs::write(
        stage.path().join("SOURCE.txt"),
        format!(
            "Lensfun database, format 1.\nDownloaded from {DOWNLOAD_URL}\nProject: https://lensfun.github.io/\nDatabase license: CC BY-SA 3.0\nhttps://github.com/lensfun/lensfun/blob/master/data/COPYING.CC_BY-SA_3.0\n"
        ),
    )?;
    fs::write(
        stage.path().join(".complete"),
        b"Lensfun database format 1\n",
    )?;
    let name = stage.path().file_name().unwrap().to_os_string();
    let mut pointer = tempfile::NamedTempFile::new_in(root)?;
    writeln!(
        pointer,
        "{}",
        name.to_str().context("Non-UTF8 Lensfun cache directory")?
    )?;
    pointer.as_file().sync_all()?;
    pointer
        .persist(root.join(CURRENT))
        .map_err(|e| e.error)
        .context("Publishing the downloaded Lensfun database")?;
    Ok(stage.keep())
}
/// Discover a compatible database, downloading a managed copy when necessary.
pub fn resolve(explicit: Option<&Path>) -> Result<PathBuf> {
    match discover(explicit)? {
        Some(path) => Ok(path),
        None => ensure_cached(),
    }
}
fn ensure_at(root: &Path, fetch: impl FnOnce() -> Result<Vec<u8>>) -> Result<PathBuf> {
    let _lock = INSTALL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(path) = cached_at(root) {
        return Ok(path);
    }
    install_archive(root, &fetch()?).context("Installing the downloaded Lensfun database")
}
/// Download once; later loads and exports reuse the cache without a network call.
/// Call from a worker thread, never the UI paint path.
pub fn ensure_cached() -> Result<PathBuf> {
    ensure_at(&cache_root()?, || {
        tracing::info!(url = DOWNLOAD_URL, "Downloading Lensfun database");
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .https_only(true)
            .timeout_global(Some(Duration::from_secs(30)))
            .timeout_connect(Some(Duration::from_secs(5)))
            .build()
            .into();
        agent.get(DOWNLOAD_URL)
            .header("User-Agent", concat!("spektrafilm-rs/", env!("CARGO_PKG_VERSION")))
            .call().context("Could not download the Lensfun database; check the internet connection, then reload the RAW to retry")?
            .body_mut().with_config().limit(MAX_DOWNLOAD).read_to_vec()
            .context("Reading Lensfun database download")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const XML: &str = r#"<lensdatabase version="1"><camera><maker>Canon</maker><model>Canon EOS 5D Mark II</model><mount>Canon EF</mount><cropfactor>1</cropfactor></camera></lensdatabase>"#;
    fn archive(files: &[(&str, &[u8], bool)]) -> Vec<u8> {
        let encoder = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::default());
        let mut builder = tar::Builder::new(encoder);
        for &(name, content, link) in files {
            let mut header = tar::Header::new_gnu();
            // Raw header paths intentionally include cases the safe tar builder
            // would refuse, exercising the installer's untrusted-input boundary.
            header.as_mut_bytes()[..name.len()].copy_from_slice(name.as_bytes());
            header.set_mode(0o644);
            header.set_size(if link { 0 } else { content.len() as u64 });
            if link {
                header.set_entry_type(tar::EntryType::Symlink);
                header.set_link_name("/tmp/unexpected").unwrap();
            }
            header.set_cksum();
            builder
                .append(&header, if link { &[][..] } else { content })
                .unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap()
    }
    #[test]
    fn installs_once_and_reuses_the_cache_offline() {
        let root = tempfile::tempdir().unwrap();
        let bytes = archive(&[("camera.xml", XML.as_bytes(), false)]);
        let path = ensure_at(root.path(), || Ok(bytes)).unwrap();
        assert_eq!(cached_at(root.path()).unwrap(), path);
        assert!(path.join("SOURCE.txt").is_file());
        let reused = ensure_at(root.path(), || bail!("offline: fetch must not run")).unwrap();
        assert_eq!(reused, path);
    }
    #[test]
    fn failed_publication_removes_the_unpublished_snapshot() {
        let root = tempfile::tempdir().unwrap();
        // A directory at CURRENT forces publication to fail without filling a disk.
        fs::create_dir(root.path().join(CURRENT)).unwrap();
        let bytes = archive(&[("camera.xml", XML.as_bytes(), false)]);
        assert!(install_archive(root.path(), &bytes).is_err());
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
        assert!(root.path().join(CURRENT).is_dir());
    }

    #[test]
    fn a_failed_download_can_be_retried() {
        let root = tempfile::tempdir().unwrap();
        assert!(ensure_at(root.path(), || bail!("offline")).is_err());
        assert!(cached_at(root.path()).is_none());
        assert!(
            ensure_at(root.path(), || Ok(archive(&[(
                "camera.xml",
                XML.as_bytes(),
                false
            )])))
            .is_ok()
        );
    }
    #[test]
    fn malformed_or_incompatible_archives_preserve_the_installed_snapshot() {
        let root = tempfile::tempdir().unwrap();
        let old = install_archive(
            root.path(),
            &archive(&[("camera.xml", XML.as_bytes(), false)]),
        )
        .unwrap();
        for bytes in [
            archive(&[(
                "camera.xml",
                br#"<lensdatabase version="2"></lensdatabase>"#,
                false,
            )]),
            archive(&[("camera.xml", b"<lensdatabase><camera>", false)]),
            archive(&[("camera.xml", b"<lensdatabase></lensdatabase>", false)]),
            b"not a bzip2 archive".to_vec(),
        ] {
            assert!(install_archive(root.path(), &bytes).is_err());
            assert_eq!(cached_at(root.path()).unwrap(), old);
        }
        let entries = fs::read_dir(root.path()).unwrap().count();
        assert_eq!(entries, 2, "failed staging directories must be removed");
    }
    #[test]
    fn archive_paths_and_links_cannot_escape_the_cache() {
        for (name, link) in [
            ("../escape.xml", false),
            ("/escape.xml", false),
            ("nested/camera.xml", false),
            (r"..\escape.xml", false),
            ("C:camera.xml", false),
            ("camera.xml", true),
        ] {
            let root = tempfile::tempdir().unwrap();
            assert!(
                install_archive(root.path(), &archive(&[(name, XML.as_bytes(), link)])).is_err(),
                "{name}"
            );
            assert!(cached_at(root.path()).is_none());
            assert_eq!(fs::read_dir(root.path()).unwrap().count(), 0);
        }
    }
    #[test]
    fn unsupported_override_is_reported_and_standard_parent_is_accepted() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("version_1")).unwrap();
        fs::write(root.path().join("version_1/camera.xml"), XML).unwrap();
        assert_eq!(
            discover(Some(root.path())).unwrap().unwrap(),
            root.path().join("version_1")
        );
        let invalid = tempfile::tempdir().unwrap();
        fs::write(
            invalid.path().join("camera.xml"),
            XML.replace(r#"version="1""#, r#"version="2""#),
        )
        .unwrap();
        assert!(
            discover(Some(invalid.path()))
                .unwrap_err()
                .to_string()
                .contains("version_1")
        );
    }
    #[test]
    fn oversized_files_and_snapshot_pointer_traversal_are_rejected() {
        let root = tempfile::tempdir().unwrap();
        let big = vec![b'x'; MAX_FILE as usize + 1];
        assert!(install_archive(root.path(), &archive(&[("camera.xml", &big, false)])).is_err());
        for name in [
            "../version_1-other",
            "/version_1-absolute",
            "version_1-a/../outside",
            "",
        ] {
            fs::write(root.path().join(CURRENT), name).unwrap();
            assert!(cached_at(root.path()).is_none());
        }
    }
    #[test]
    #[ignore = "downloads the official database into the user cache; requires internet"]
    fn downloads_official_database_and_reuses_it() {
        let installed = ensure_cached().unwrap();
        assert!(installed.join(".complete").is_file());
        assert_eq!(ensure_cached().unwrap(), installed);
        assert_eq!(cached().unwrap(), installed);
        assert!(
            fs::read_dir(installed)
                .unwrap()
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().is_some_and(|v| v == "xml"))
                .count()
                > 20
        );
    }
}
