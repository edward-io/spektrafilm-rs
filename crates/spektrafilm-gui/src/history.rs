//! Edit history: a list of labelled states with a current position, shared by
//! undo / redo and the History panel (Lightroom-style).
//!
//! The app observes its state once per frame. A change is recorded as one step
//! only after the pointer is released and no text field has focus, so a whole
//! slider drag or a typed matrix becomes a single step. Jumping back and then
//! editing discards the steps after the current one.
use serde_json::Value;
use std::sync::Arc;

/// Steps kept per photo; older ones fold into the starting state.
pub const LIMIT: usize = 500;

#[derive(Clone, PartialEq, Debug)]
pub struct Snapshot {
    pub params: Arc<Value>,
    pub film: String,
    pub print: String,
    pub scan_for_print_restore: Option<[bool; 3]>,
}

#[derive(Clone, Debug)]
pub struct Step {
    pub snapshot: Snapshot,
    pub label: String,
    /// Unix seconds.
    pub at: u64,
}

/// What [`History::commit`] did, for persisting it.
pub struct Recorded {
    /// Index of the new step; any steps after it were discarded.
    pub index: usize,
    /// The oldest step was folded into the starting state (cap reached).
    pub folded: bool,
}

#[derive(Default)]
pub struct History {
    steps: Vec<Step>,
    position: usize,
}

pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

impl History {
    /// Start over from `current`.
    pub fn reset(&mut self, current: Snapshot, label: &str) {
        self.steps = vec![Step {
            snapshot: current,
            label: label.into(),
            at: now(),
        }];
        self.position = 0;
    }

    /// Restore a saved history.
    pub fn restore(&mut self, steps: Vec<Step>, position: usize) {
        self.position = position.min(steps.len().saturating_sub(1));
        self.steps = steps;
    }

    /// Replace the current step's state without recording a change, e.g. when a
    /// newly loaded file sets its own colour encoding.
    pub fn amend_current(&mut self, current: Snapshot) {
        match self.steps.get_mut(self.position) {
            Some(step) => step.snapshot = current,
            None => self.reset(current, "Start"),
        }
    }

    pub fn steps(&self) -> &[Step] {
        &self.steps
    }
    pub fn position(&self) -> usize {
        self.position
    }

    /// The current step's state, if `current` differs from it (a change worth
    /// recording). Starts the history if there is none yet.
    pub fn pending(&mut self, current: &Snapshot) -> Option<Snapshot> {
        match self.steps.get(self.position) {
            Some(step) if step.snapshot == *current => None,
            Some(step) => Some(step.snapshot.clone()),
            None => {
                self.reset(current.clone(), "Start");
                None
            }
        }
    }

    /// Record `current` as a new step after the current position.
    pub fn commit(&mut self, current: Snapshot, label: String) -> Recorded {
        self.steps.truncate(self.position + 1);
        self.steps.push(Step {
            snapshot: current,
            label,
            at: now(),
        });
        let folded = self.steps.len() > LIMIT;
        if folded {
            // The oldest step stops being reachable; the next one becomes the start.
            self.steps.remove(0);
        }
        self.position = self.steps.len() - 1;
        Recorded {
            index: self.position,
            folded,
        }
    }

    pub fn can_undo(&self) -> bool {
        self.position > 0
    }
    pub fn can_redo(&self) -> bool {
        self.position + 1 < self.steps.len()
    }
    pub fn undo_label(&self) -> Option<&str> {
        self.can_undo()
            .then(|| self.steps[self.position].label.as_str())
    }
    pub fn redo_label(&self) -> Option<&str> {
        self.steps.get(self.position + 1).map(|s| s.label.as_str())
    }

    /// Move to step `index`, returning its state.
    pub fn jump(&mut self, index: usize) -> Option<Snapshot> {
        let step = self.steps.get(index)?;
        self.position = index;
        Some(step.snapshot.clone())
    }
    pub fn undo(&mut self) -> Option<Snapshot> {
        self.can_undo()
            .then(|| self.position - 1)
            .and_then(|i| self.jump(i))
    }
    pub fn redo(&mut self) -> Option<Snapshot> {
        self.can_redo()
            .then(|| self.position + 1)
            .and_then(|i| self.jump(i))
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
    fn observe(h: &mut History, v: i32) -> bool {
        match h.pending(&snap(v)) {
            Some(_) => {
                h.commit(snap(v), format!("to {v}"));
                true
            }
            None => false,
        }
    }

    #[test]
    fn undo_redo_walk_the_steps() {
        let mut h = History::default();
        assert!(!observe(&mut h, 0), "the first state starts the history");
        assert!(observe(&mut h, 1));
        assert!(!observe(&mut h, 1));
        assert!(observe(&mut h, 2));
        assert_eq!(h.undo_label(), Some("to 2"));
        assert_eq!(h.undo().unwrap(), snap(1));
        assert_eq!(h.undo().unwrap(), snap(0));
        assert!(h.undo().is_none());
        assert_eq!(h.redo_label(), Some("to 1"));
        assert_eq!(h.redo().unwrap(), snap(1));
    }

    #[test]
    fn editing_after_jumping_back_discards_later_steps() {
        let mut h = History::default();
        for v in 0..4 {
            observe(&mut h, v);
        }
        assert_eq!(h.jump(1).unwrap(), snap(1));
        let recorded = h.commit(snap(9), "to 9".into());
        assert_eq!(recorded.index, 2);
        assert_eq!(h.steps().len(), 3);
        assert!(!h.can_redo());
        assert_eq!(h.undo().unwrap(), snap(1));
    }

    #[test]
    fn history_is_capped_by_folding_the_oldest_step() {
        let mut h = History::default();
        observe(&mut h, -1);
        let mut folded = false;
        for v in 0..LIMIT as i32 + 5 {
            observe(&mut h, v);
            folded |= h.steps().len() == LIMIT && h.steps()[0].snapshot != snap(-1);
        }
        assert!(folded);
        assert_eq!(h.steps().len(), LIMIT);
        assert_eq!(h.position(), LIMIT - 1);
    }

    #[test]
    fn amending_the_current_step_is_not_a_change() {
        let mut h = History::default();
        observe(&mut h, 0);
        h.amend_current(snap(5));
        assert!(!observe(&mut h, 5));
        assert!(!h.can_undo());
    }
}
