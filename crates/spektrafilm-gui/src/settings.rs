//! App preferences saved automatically between launches: how things are
//! viewed and exported, and the catalog to reopen. eframe stores them with the
//! window geometry and panel / section state. The look itself lives in startup
//! defaults and per-photo edits, not here.
use super::*;
use serde::{Deserialize, Serialize};

const KEY: &str = "spektrafilm-settings";

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    pub display: project::DisplayParams,
    pub tab: inspector::Tab,
    pub auto_preview: bool,
    pub gpu_export: bool,
    pub lut: lut_ui::Settings,
    pub save_options: spektrafilm_io::SaveOptions,
    pub last_catalog: Option<PathBuf>,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            display: Default::default(),
            tab: Default::default(),
            auto_preview: true,
            gpu_export: true,
            lut: Default::default(),
            save_options: Default::default(),
            last_catalog: None,
        }
    }
}

/// eframe's state file, next to the startup defaults and edits. (eframe takes
/// a file path here, despite documenting a folder.)
pub fn storage_file() -> PathBuf {
    project::default_path().with_file_name("app.ron")
}

impl App {
    pub(crate) fn settings(&self) -> AppSettings {
        AppSettings {
            display: self.display.clone(),
            tab: self.tab,
            auto_preview: self.auto_preview,
            gpu_export: self.gpu_export,
            lut: self.lut.clone(),
            save_options: self.save_options.clone(),
            last_catalog: self.last_catalog.clone(),
        }
    }

    pub(crate) fn apply_settings(&mut self, settings: AppSettings) {
        self.display = settings.display;
        self.tab = settings.tab;
        self.auto_preview = settings.auto_preview;
        self.gpu_export = settings.gpu_export;
        self.lut = settings.lut;
        self.save_options = settings.save_options;
        self.last_catalog = settings.last_catalog;
    }

    /// Restore saved preferences, then reopen the last catalog unless an image
    /// was passed on the command line.
    pub(crate) fn restore_settings(&mut self, storage: &dyn eframe::Storage, opening_image: bool) {
        let Some(settings) = eframe::get_value::<AppSettings>(storage, KEY) else {
            return;
        };
        self.apply_settings(settings);
        if !opening_image
            && let Some(catalog) = self.last_catalog.clone().filter(|path| path.is_file())
        {
            self.open_catalog(catalog);
        }
    }

    pub(crate) fn store_settings(&self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, KEY, &self.settings());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct MemoryStorage(HashMap<String, String>);
    impl eframe::Storage for MemoryStorage {
        fn get_string(&self, key: &str) -> Option<String> {
            self.0.get(key).cloned()
        }
        fn set_string(&mut self, key: &str, value: String) {
            self.0.insert(key.into(), value);
        }
        fn flush(&mut self) {}
    }

    fn app() -> App {
        let ctx = egui::Context::default();
        let cc = eframe::CreationContext::_new_kittest(ctx);
        App::new(
            &cc,
            Arc::new(spektrafilm_gpu::cpu_backend::CpuBackend),
            None,
        )
    }

    #[test]
    fn preferences_survive_a_restart_and_reopen_the_last_catalog() {
        let dir = tempfile::tempdir().unwrap();
        let catalog = dir.path().join("test.lrcat");
        std::fs::write(&catalog, b"").unwrap();

        let mut storage = MemoryStorage::default();
        let mut first = app();
        first.display.show_border = false;
        first.display.gray_18_canvas = false;
        first.tab = inspector::Tab::Print;
        first.auto_preview = false;
        first.save_options.jpeg_quality = 77;
        first.last_catalog = Some(catalog.clone());
        first.store_settings(&mut storage);

        let mut second = app();
        second.restore_settings(&storage, false);
        assert!(!second.display.show_border);
        assert!(!second.display.gray_18_canvas);
        assert_eq!(second.tab, inspector::Tab::Print);
        assert!(!second.auto_preview);
        assert_eq!(second.save_options.jpeg_quality, 77);
        assert_eq!(
            second.catalog.as_ref().map(|c| c.path()),
            Some(catalog.as_path())
        );

        // An image on the command line wins over reopening the catalog.
        let mut third = app();
        third.restore_settings(&storage, true);
        assert!(third.catalog.is_none());

        // A catalog that has since gone away is skipped.
        std::fs::remove_file(&catalog).unwrap();
        let mut fourth = app();
        fourth.restore_settings(&storage, false);
        assert!(fourth.catalog.is_none());
    }
}
