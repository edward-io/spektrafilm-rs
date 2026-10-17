//! Undo / redo of the rendering state (params plus the chosen stocks).
//!
//! The app observes its state once per frame. A change is committed as one
//! step only after the pointer is released and no text field has focus, so a
//! whole slider drag or a typed matrix becomes a single undo step.
use serde_json::Value;
use std::sync::Arc;

const LIMIT: usize = 200;

#[derive(Clone, PartialEq, Debug)]
pub struct Snapshot {
    pub params: Arc<Value>,
    pub film: String,
    pub print: String,
    pub scan_for_print_restore: Option<[bool; 3]>,
}

#[derive(Default)]
pub struct History {
    committed: Option<Snapshot>,
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
}

impl History {
    /// Record `current` as the latest state if it differs from the last commit.
    pub fn observe(&mut self, current: Snapshot) {
        match &self.committed {
            Some(last) if *last == current => {}
            Some(_) => {
                let last = self.committed.replace(current).expect("checked above");
                self.undo.push(last);
                if self.undo.len() > LIMIT {
                    self.undo.remove(0);
                }
                self.redo.clear();
            }
            None => self.committed = Some(current),
        }
    }

    /// Forget the past, e.g. after opening a project.
    pub fn reset(&mut self, current: Snapshot) {
        self.committed = Some(current);
        self.undo.clear();
        self.redo.clear();
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// `current` is the live state, which may hold an edit not yet committed.
    pub fn undo(&mut self, current: Snapshot) -> Option<Snapshot> {
        self.observe(current);
        let target = self.undo.pop()?;
        let now = self
            .committed
            .replace(target.clone())
            .expect("observed above");
        self.redo.push(now);
        Some(target)
    }

    pub fn redo(&mut self, current: Snapshot) -> Option<Snapshot> {
        self.observe(current);
        let target = self.redo.pop()?;
        let now = self
            .committed
            .replace(target.clone())
            .expect("observed above");
        self.undo.push(now);
        Some(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn snap(v: i32) -> Snapshot {
        Snapshot {
            params: Arc::new(serde_json::json!(v)),
            film: "f".into(),
            print: "p".into(),
            scan_for_print_restore: None,
        }
    }
    #[test]
    fn undo_redo_round_trip() {
        let mut h = History::default();
        h.observe(snap(0));
        h.observe(snap(1));
        h.observe(snap(1));
        h.observe(snap(2));
        assert_eq!(h.undo(snap(2)).unwrap(), snap(1));
        assert_eq!(h.undo(snap(1)).unwrap(), snap(0));
        assert!(h.undo(snap(0)).is_none());
        assert_eq!(h.redo(snap(0)).unwrap(), snap(1));
        // A new edit drops the redo branch.
        h.observe(snap(5));
        assert!(!h.can_redo());
        assert_eq!(h.undo(snap(5)).unwrap(), snap(1));
    }
    #[test]
    fn uncommitted_edit_is_undone_first() {
        let mut h = History::default();
        h.observe(snap(0));
        assert_eq!(h.undo(snap(3)).unwrap(), snap(0));
        assert_eq!(h.redo(snap(0)).unwrap(), snap(3));
    }
}
