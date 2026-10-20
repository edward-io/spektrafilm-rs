//! Per-photo settings for Lightroom catalog photos, kept in a local SQLite
//! database next to the startup defaults. Lightroom catalogs and originals are
//! never written.
//!
//! Each row holds one photo's edit as JSON plus a schema version. Saving merges
//! the new values over the stored document, so fields written by a newer build
//! survive a round trip through an older one; reading relies on serde defaults
//! for fields an older row lacks.
//!
//! The photo's history lives in `photo_history`: step 0 stores the full
//! starting state, later steps only the JSON pointers whose values changed.
use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use spektrafilm_core::params::RuntimeParams;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Version of the JSON document in `photo_edits.edit`. Bump it, and add a step
/// to [`migrate`], when a change can't be expressed through serde defaults.
const EDIT_VERSION: i64 = 1;
/// Version of the table layout, stored in SQLite's `user_version`.
const DATABASE_VERSION: i64 = 1;

/// Everything that makes up one photo's look in the editor.
#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct PhotoEdit {
    pub film: String,
    pub paper: String,
    pub params: RuntimeParams,
    pub scan_for_print_restore: Option<[bool; 3]>,
    /// Quarter turns applied in the editor on top of Lightroom's orientation.
    pub rotation: u8,
    /// RAW development options; the Lightroom crop is always re-read from the
    /// catalog instead.
    pub raw: spektrafilm_io::RawOptions,
    /// Crop and straightening; `None` (never cropped here) uses Lightroom's.
    pub crop: Option<spektrafilm_io::crop::Crop>,
}

impl Default for PhotoEdit {
    fn default() -> Self {
        let project = crate::project::Project::default();
        Self {
            film: project.film,
            paper: project.paper,
            params: project.params,
            scan_for_print_restore: None,
            rotation: 0,
            raw: Default::default(),
            crop: None,
        }
    }
}

pub struct EditStore {
    connection: Connection,
}

impl EditStore {
    /// The store beside the startup defaults. Tests use a throwaway location so
    /// they never touch a real user's edits.
    pub fn default_path() -> PathBuf {
        if cfg!(test) {
            return std::env::temp_dir()
                .join(format!("spektrafilm-test-edits-{}", std::process::id()))
                .join("edits.sqlite");
        }
        crate::project::default_path().with_file_name("edits.sqlite")
    }

    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let connection =
            Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        let version: i64 = connection.pragma_query_value(None, "user_version", |r| r.get(0))?;
        if version > DATABASE_VERSION {
            anyhow::bail!(
                "{} was written by a newer Spektrafilm (database version {version})",
                path.display()
            );
        }
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS photo_edits (
                 catalog TEXT NOT NULL,
                 photo_id INTEGER NOT NULL,
                 edit_version INTEGER NOT NULL,
                 updated_at INTEGER NOT NULL,
                 edit TEXT NOT NULL,
                 history_position INTEGER NOT NULL DEFAULT 0,
                 PRIMARY KEY (catalog, photo_id)
             ) WITHOUT ROWID;
             CREATE TABLE IF NOT EXISTS photo_history (
                 catalog TEXT NOT NULL,
                 photo_id INTEGER NOT NULL,
                 seq INTEGER NOT NULL,
                 created_at INTEGER NOT NULL,
                 label TEXT NOT NULL,
                 change TEXT NOT NULL,
                 PRIMARY KEY (catalog, photo_id, seq)
             ) WITHOUT ROWID;",
        )?;
        connection.pragma_update(None, "user_version", DATABASE_VERSION)?;
        Ok(Self { connection })
    }

    /// The photo's saved edit, if it has one. A row this build can't read is
    /// reported as an error rather than silently replaced.
    pub fn load(&self, catalog: &Path, photo_id: i64) -> Result<Option<PhotoEdit>> {
        let Some((version, edit)) = self.row(catalog, photo_id)? else {
            return Ok(None);
        };
        let document = migrate(version, serde_json::from_str(&edit)?)?;
        Ok(Some(serde_json::from_value(document)?))
    }

    /// Save the photo's edit and which history step it is at.
    pub fn save(
        &self,
        catalog: &Path,
        photo_id: i64,
        edit: &PhotoEdit,
        history_position: usize,
    ) -> Result<()> {
        let mut document = match self.row(catalog, photo_id)? {
            Some((version, stored)) => serde_json::from_str(&stored)
                .ok()
                .and_then(|value| migrate(version, value).ok())
                .unwrap_or(Value::Null),
            None => Value::Null,
        };
        merge(&mut document, serde_json::to_value(edit)?);
        self.connection.execute(
            "INSERT INTO photo_edits
                 (catalog, photo_id, edit_version, updated_at, edit, history_position)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT (catalog, photo_id) DO UPDATE SET
                 edit_version = excluded.edit_version,
                 updated_at = excluded.updated_at,
                 edit = excluded.edit,
                 history_position = excluded.history_position",
            params![
                catalog_key(catalog),
                photo_id,
                EDIT_VERSION.max(self.stored_version(catalog, photo_id)?),
                crate::history::now() as i64,
                document.to_string(),
                history_position as i64,
            ],
        )?;
        Ok(())
    }

    /// The photo's history with full states rebuilt, and its current position.
    pub fn load_history(
        &self,
        catalog: &Path,
        photo_id: i64,
    ) -> Result<Option<(Vec<StoredStep>, usize)>> {
        let key = catalog_key(catalog);
        let mut statement = self.connection.prepare(
            "SELECT created_at, label, change FROM photo_history
             WHERE catalog = ?1 AND photo_id = ?2 ORDER BY seq",
        )?;
        let rows = statement
            .query_map(params![key, photo_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        if rows.is_empty() {
            return Ok(None);
        }
        let mut steps = Vec::with_capacity(rows.len());
        let mut state = Value::Null;
        for (at, label, change) in rows {
            apply_change(&mut state, serde_json::from_str(&change)?);
            steps.push(StoredStep {
                label,
                at: at as u64,
                state: state.clone(),
            });
        }
        let position: i64 = self
            .connection
            .query_row(
                "SELECT history_position FROM photo_edits WHERE catalog = ?1 AND photo_id = ?2",
                params![key, photo_id],
                |row| row.get(0),
            )
            .optional()?
            .unwrap_or(steps.len() as i64 - 1);
        Ok(Some((steps, position as usize)))
    }

    /// Store step `index` (replacing it and any later steps): the full state
    /// for the first step, otherwise only what changed since `previous`.
    pub fn record_step(
        &self,
        catalog: &Path,
        photo_id: i64,
        index: usize,
        previous: Option<&Value>,
        step: &StoredStep,
    ) -> Result<()> {
        let key = catalog_key(catalog);
        self.connection.execute(
            "DELETE FROM photo_history WHERE catalog = ?1 AND photo_id = ?2 AND seq >= ?3",
            params![key, photo_id, index as i64],
        )?;
        let change = match previous {
            Some(previous) => diff(previous, &step.state),
            None => Value::Object([(String::new(), step.state.clone())].into_iter().collect()),
        };
        self.connection.execute(
            "INSERT INTO photo_history (catalog, photo_id, seq, created_at, label, change)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                key,
                photo_id,
                index as i64,
                step.at as i64,
                step.label,
                change.to_string()
            ],
        )?;
        Ok(())
    }

    /// Rewrite the whole history, e.g. after the oldest step was folded away
    /// or the history was cleared.
    pub fn replace_history(
        &self,
        catalog: &Path,
        photo_id: i64,
        steps: &[StoredStep],
    ) -> Result<()> {
        let transaction = self.connection.unchecked_transaction()?;
        self.connection.execute(
            "DELETE FROM photo_history WHERE catalog = ?1 AND photo_id = ?2",
            params![catalog_key(catalog), photo_id],
        )?;
        for (index, step) in steps.iter().enumerate() {
            let previous = index.checked_sub(1).map(|i| &steps[i].state);
            self.record_step(catalog, photo_id, index, previous, step)?;
        }
        transaction.commit()?;
        Ok(())
    }

    /// Photos in `catalog` that have saved edits, for the browser's markers.
    pub fn edited(&self, catalog: &Path) -> Result<HashSet<i64>> {
        let mut statement = self
            .connection
            .prepare("SELECT photo_id FROM photo_edits WHERE catalog = ?1")?;
        let ids = statement
            .query_map(params![catalog_key(catalog)], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(ids)
    }

    fn row(&self, catalog: &Path, photo_id: i64) -> Result<Option<(i64, String)>> {
        Ok(self
            .connection
            .query_row(
                "SELECT edit_version, edit FROM photo_edits WHERE catalog = ?1 AND photo_id = ?2",
                params![catalog_key(catalog), photo_id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?)
    }

    fn stored_version(&self, catalog: &Path, photo_id: i64) -> Result<i64> {
        Ok(self
            .row(catalog, photo_id)?
            .map_or(0, |(version, _)| version))
    }
}

/// One history step as stored: the full state is rebuilt on load.
#[derive(Clone, Debug)]
pub struct StoredStep {
    pub label: String,
    pub at: u64,
    pub state: Value,
}

/// JSON pointers (to leaves, or whole arrays) whose values differ in `new`.
fn diff(old: &Value, new: &Value) -> Value {
    fn walk(old: &Value, new: &Value, path: &mut String, out: &mut serde_json::Map<String, Value>) {
        match (old, new) {
            (Value::Object(o), Value::Object(n)) => {
                for (key, value) in n {
                    let len = path.len();
                    path.push('/');
                    path.push_str(&key.replace('~', "~0").replace('/', "~1"));
                    match o.get(key) {
                        Some(previous) => walk(previous, value, path, out),
                        None => {
                            out.insert(path.clone(), value.clone());
                        }
                    }
                    path.truncate(len);
                }
            }
            _ if old != new => {
                out.insert(path.clone(), new.clone());
            }
            _ => {}
        }
    }
    let mut out = serde_json::Map::new();
    walk(old, new, &mut String::new(), &mut out);
    Value::Object(out)
}

/// Apply a change produced by [`diff`] (or a full state under the "" pointer).
fn apply_change(state: &mut Value, change: Value) {
    let Value::Object(change) = change else {
        return;
    };
    for (pointer, value) in change {
        if pointer.is_empty() {
            *state = value;
            continue;
        }
        let mut slot = &mut *state;
        for part in pointer.split('/').skip(1) {
            let key = part.replace("~1", "/").replace("~0", "~");
            if !slot.is_object() {
                *slot = Value::Object(Default::default());
            }
            slot = slot
                .as_object_mut()
                .expect("made an object above")
                .entry(key)
                .or_insert(Value::Null);
        }
        *slot = value;
    }
}

/// Catalogs are identified by their absolute path.
fn catalog_key(catalog: &Path) -> String {
    std::path::absolute(catalog)
        .unwrap_or_else(|_| catalog.to_path_buf())
        .to_string_lossy()
        .into_owned()
}

/// Bring a stored document up to [`EDIT_VERSION`]. Newer documents are read
/// as-is: unknown fields are ignored and missing ones take their defaults.
fn migrate(version: i64, document: Value) -> Result<Value> {
    match version {
        1.. => Ok(document),
        _ => anyhow::bail!("unknown edit version {version}"),
    }
}

/// Overlay `update` on `base`, keeping keys only `base` has (fields from a
/// newer build that this one doesn't know about).
pub(crate) fn merge(base: &mut Value, update: Value) {
    match (base, update) {
        (Value::Object(base), Value::Object(update)) => {
            for (key, value) in update {
                merge(base.entry(key).or_insert(Value::Null), value);
            }
        }
        (base, update) => *base = update,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(edit: Option<PhotoEdit>) -> Option<Value> {
        edit.map(|edit| serde_json::to_value(edit).unwrap())
    }

    fn store() -> (tempfile::TempDir, EditStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = EditStore::open(&dir.path().join("edits.sqlite")).unwrap();
        (dir, store)
    }

    #[test]
    fn edits_round_trip_per_catalog_and_photo() {
        let (_dir, store) = store();
        let catalog = Path::new("/catalogs/a.lrcat");
        let mut edit = PhotoEdit::default();
        edit.film = "kodak_portra_400".into();
        edit.params.camera.exposure_compensation_ev = 1.5;
        edit.params.camera.tone.shadows = 75.;
        edit.params.camera.tone.highlights = -45.;
        edit.rotation = 3;
        store.save(catalog, 7, &edit, 0).unwrap();

        assert_eq!(
            json(store.load(catalog, 7).unwrap()),
            json(Some(edit.clone()))
        );
        assert!(store.load(catalog, 8).unwrap().is_none());
        assert!(
            store
                .load(Path::new("/catalogs/b.lrcat"), 7)
                .unwrap()
                .is_none()
        );
        assert_eq!(store.edited(catalog).unwrap(), HashSet::from([7]));

        edit.params.camera.exposure_compensation_ev = -0.5;
        store.save(catalog, 7, &edit, 0).unwrap();
        assert_eq!(
            store
                .load(catalog, 7)
                .unwrap()
                .unwrap()
                .params
                .camera
                .exposure_compensation_ev,
            -0.5
        );
    }

    #[test]
    fn fields_from_a_newer_build_survive_a_save() {
        let (_dir, store) = store();
        let catalog = Path::new("/catalogs/a.lrcat");
        store.save(catalog, 1, &PhotoEdit::default(), 0).unwrap();
        // A newer build added fields this one doesn't know.
        let (version, stored) = store.row(catalog, 1).unwrap().unwrap();
        let mut document: Value = serde_json::from_str(&stored).unwrap();
        document["future_setting"] = 42.into();
        document["params"]["camera"]["future_knob"] = "on".into();
        store
            .connection
            .execute(
                "UPDATE photo_edits SET edit = ?1, edit_version = ?2",
                params![document.to_string(), version + 1],
            )
            .unwrap();

        // This build still reads it, and its own save keeps the unknown fields.
        let mut edit = store.load(catalog, 1).unwrap().unwrap();
        edit.params.camera.exposure_compensation_ev = 2.0;
        store.save(catalog, 1, &edit, 0).unwrap();
        let (version_after, stored) = store.row(catalog, 1).unwrap().unwrap();
        let document: Value = serde_json::from_str(&stored).unwrap();
        assert_eq!(document["future_setting"], 42);
        assert_eq!(document["params"]["camera"]["future_knob"], "on");
        assert_eq!(
            document["params"]["camera"]["exposure_compensation_ev"],
            2.0
        );
        assert_eq!(version_after, version + 1, "the newer version is kept");
    }

    #[test]
    fn missing_fields_in_an_older_row_take_defaults() {
        let (_dir, store) = store();
        let catalog = Path::new("/catalogs/a.lrcat");
        store
            .connection
            .execute(
                "INSERT INTO photo_edits (catalog, photo_id, edit_version, updated_at, edit)
                 VALUES (?1, 3, 1, 0, ?2)",
                params![
                    catalog_key(catalog),
                    r#"{"film":"kodak_gold_200","params":{"camera":{"exposure_compensation_ev":0.7}}}"#
                ],
            )
            .unwrap();
        let edit = store.load(catalog, 3).unwrap().unwrap();
        assert_eq!(edit.film, "kodak_gold_200");
        assert_eq!(edit.params.camera.exposure_compensation_ev, 0.7);
        assert_eq!(edit.rotation, 0);
    }

    fn state(ev: f64, film: &str) -> Value {
        serde_json::json!({"film": film, "params": {"camera": {"exposure_compensation_ev": ev, "lens_blur_um": 0.0}}})
    }
    fn step(label: &str, state: Value) -> StoredStep {
        StoredStep {
            label: label.into(),
            at: 1,
            state,
        }
    }

    #[test]
    fn history_steps_store_only_what_changed_and_rebuild_in_full() {
        let (_dir, store) = store();
        let catalog = Path::new("/catalogs/a.lrcat");
        let states = [state(0.0, "gold"), state(1.0, "gold"), state(1.0, "portra")];
        for (i, s) in states.iter().enumerate() {
            let previous = i.checked_sub(1).map(|p| &states[p]);
            store
                .record_step(
                    catalog,
                    5,
                    i,
                    previous,
                    &step(&format!("step {i}"), s.clone()),
                )
                .unwrap();
        }
        store.save(catalog, 5, &PhotoEdit::default(), 1).unwrap();

        // Later steps hold only the changed pointer.
        let change: String = store
            .connection
            .query_row("SELECT change FROM photo_history WHERE seq = 2", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(change, r#"{"/film":"portra"}"#);

        let (steps, position) = store.load_history(catalog, 5).unwrap().unwrap();
        assert_eq!(position, 1);
        assert_eq!(
            steps.iter().map(|s| &s.state).collect::<Vec<_>>(),
            states.iter().collect::<Vec<_>>()
        );
        assert_eq!(steps[2].label, "step 2");
        assert!(store.load_history(catalog, 6).unwrap().is_none());
    }

    #[test]
    fn recording_after_a_jump_replaces_later_steps_and_history_can_be_rewritten() {
        let (_dir, store) = store();
        let catalog = Path::new("/catalogs/a.lrcat");
        let states = [state(0.0, "gold"), state(1.0, "gold"), state(2.0, "gold")];
        for (i, s) in states.iter().enumerate() {
            let previous = i.checked_sub(1).map(|p| &states[p]);
            store
                .record_step(catalog, 5, i, previous, &step("s", s.clone()))
                .unwrap();
        }
        // Jumped back to step 0, then edited: step 1 is replaced and step 2 dropped.
        store
            .record_step(
                catalog,
                5,
                1,
                Some(&states[0]),
                &step("new", state(-1.0, "gold")),
            )
            .unwrap();
        let (steps, _) = store.load_history(catalog, 5).unwrap().unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[1].state, state(-1.0, "gold"));

        store
            .replace_history(catalog, 5, &[step("Start", state(3.0, "portra"))])
            .unwrap();
        let (steps, _) = store.load_history(catalog, 5).unwrap().unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].state, state(3.0, "portra"));
    }

    #[test]
    fn an_unreadable_row_is_an_error_not_a_silent_reset() {
        let (_dir, store) = store();
        let catalog = Path::new("/catalogs/a.lrcat");
        store
            .connection
            .execute(
                "INSERT INTO photo_edits (catalog, photo_id, edit_version, updated_at, edit)
                 VALUES (?1, 4, 1, 0, 'not json')",
                params![catalog_key(catalog)],
            )
            .unwrap();
        assert!(store.load(catalog, 4).is_err());
    }
}
