//! Stage complete bundles and replace only directories known to belong to the exporter.
use std::{
    collections::HashSet,
    fs,
    path::{Component, Path, PathBuf},
};

pub struct OutputDirectory {
    stage: tempfile::TempDir,
    destination: PathBuf,
}

impl OutputDirectory {
    pub fn new(destination: &Path) -> Result<Self, String> {
        let destination = std::path::absolute(destination).map_err(|e| e.to_string())?;
        check_destination(&destination)?;
        let parent = parent(&destination);
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        let stage = tempfile::Builder::new()
            .prefix(".spektrafilm-build-")
            .tempdir_in(parent)
            .map_err(|e| e.to_string())?;
        Ok(Self { stage, destination })
    }

    pub fn path(&self) -> &Path {
        self.stage.path()
    }

    pub fn commit(self) -> Result<PathBuf, String> {
        check_destination(&self.destination)?;
        let backup = if self.destination.exists() {
            let backup = tempfile::Builder::new()
                .prefix(".spektrafilm-previous-")
                .tempdir_in(parent(&self.destination))
                .map_err(|e| e.to_string())?;
            fs::rename(&self.destination, backup.path().join("bundle"))
                .map_err(|e| e.to_string())?;
            Some(backup)
        } else {
            None
        };
        if let Err(error) = fs::rename(self.stage.path(), &self.destination) {
            if let Some(backup) = backup {
                if let Err(restore) = fs::rename(backup.path().join("bundle"), &self.destination) {
                    let kept = backup.keep();
                    return Err(format!(
                        "Publishing bundle failed: {error}; restore failed: {restore}. Previous bundle kept in {}",
                        kept.display()
                    ));
                }
            }
            return Err(format!("Publishing bundle failed: {error}"));
        }
        Ok(self.destination)
    }
}

fn parent(path: &Path) -> &Path {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

fn single_name(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    matches!(parts.next(), Some(Component::Normal(_)))
        && parts.next().is_none()
        && !name.contains(['\\', ':'])
}

fn check_destination(path: &Path) -> Result<(), String> {
    let meta = match fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.to_string()),
    };
    if !meta.is_dir() || meta.file_type().is_symlink() {
        return Err(format!(
            "Bundle destination must be a directory: {}",
            path.display()
        ));
    }
    let entries = fs::read_dir(path)
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    if entries.is_empty() {
        return Ok(());
    }
    let read = |name| -> Option<serde_json::Value> {
        serde_json::from_slice(&fs::read(path.join(name)).ok()?).ok()
    };
    let bundle = read("bundle.json");
    let spec = read("spec.json");
    if bundle.as_ref().is_none_or(|v| !v["luts"].is_array())
        && spec
            .as_ref()
            .is_none_or(|v| !v["print_profiles"].is_array())
    {
        return Err(format!(
            "Choose an empty folder or an existing spektrafilm bundle: {}",
            path.display()
        ));
    }
    let mut files: HashSet<String> = [
        "bundle.json",
        "spec.json",
        "README.md",
        "SPEKTRAFILM_LICENSE.txt",
        "config.ocio",
        "qa.json",
        "qa.html",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    if let Some(luts) = bundle.as_ref().and_then(|v| v["luts"].as_array()) {
        for lut in luts {
            let name = lut["file"]
                .as_str()
                .filter(|n| single_name(n))
                .ok_or("Invalid existing bundle LUT path")?;
            files.insert(name.into());
        }
    }
    let mut papers = HashSet::new();
    if let Some(prints) = spec.as_ref().and_then(|v| v["print_profiles"].as_array()) {
        if prints.len() > 1 {
            for print in prints {
                let name = print
                    .as_str()
                    .filter(|n| single_name(n))
                    .ok_or("Invalid existing print path")?;
                papers.insert(name.to_owned());
            }
        }
    }
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        let kind = entry.file_type().map_err(|e| e.to_string())?;
        if kind.is_file() && files.contains(&name) {
            continue;
        }
        if kind.is_dir() && name == "qa" && bundle.is_some() {
            continue;
        }
        if kind.is_dir() && papers.contains(&name) {
            check_destination(&entry.path())?;
            continue;
        }
        return Err(format!(
            "Bundle destination contains unrelated material: {}. Choose a separate output folder.",
            entry.path().display()
        ));
    }
    Ok(())
}
