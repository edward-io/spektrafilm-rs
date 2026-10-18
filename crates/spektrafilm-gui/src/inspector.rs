//! Sidebar inspector plumbing: stage tabs, section headers, and stock-default
//! tracking for the parameter controls.
//!
//! Controls write straight into `RuntimeParams`, so the inspector observes the
//! serialized params instead of threading bindings through every widget. Each
//! frame the app publishes the current and stock-default JSON; sections compare
//! their JSON pointers directly, and individual rows learn their pointers the
//! first time the user changes them (the params diff of that frame).
use crate::theme;
use eframe::egui::{self, Id, RichText, Ui};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum Tab {
    #[default]
    Input,
    Film,
    Print,
    Scan,
    Output,
}

impl Tab {
    pub const ALL: [Tab; 5] = [Tab::Input, Tab::Film, Tab::Print, Tab::Scan, Tab::Output];
    pub fn name(self) -> &'static str {
        match self {
            Tab::Input => "Input",
            Tab::Film => "Film",
            Tab::Print => "Print",
            Tab::Scan => "Scan",
            Tab::Output => "Output",
        }
    }
}

/// Which pipeline stages the current workflow route runs.
#[derive(Clone, Copy)]
pub struct Stages {
    pub camera: bool,
    pub film: bool,
    pub convert: bool,
    pub print: bool,
    pub scan: bool,
}

impl Stages {
    pub fn of(params: &spektrafilm_core::params::RuntimeParams) -> Self {
        let route = params.route();
        let passthrough = route == "input";
        let convert = params.converts_film();
        Self {
            camera: !passthrough && !convert,
            film: !passthrough,
            convert,
            print: route.contains("print") && !params.scans_film(),
            scan: !passthrough,
        }
    }
    pub fn tab_used(self, tab: Tab) -> bool {
        match tab {
            Tab::Input | Tab::Output => true,
            Tab::Film => self.film,
            Tab::Print => self.print,
            Tab::Scan => self.scan,
        }
    }
}

/// Shared per-frame state between the app and the stateless widget helpers.
#[derive(Default)]
struct Frame {
    tab: Tab,
    current: Arc<Value>,
    baseline: Arc<Value>,
    learned: Arc<HashMap<Id, Vec<String>>>,
    section: String,
    changed: Vec<(Id, String)>,
    resets: Vec<Vec<String>>,
}

thread_local! {
    static FRAME: RefCell<Frame> = RefCell::default();
}

/// What the panel did this frame, for the app to apply after drawing.
pub struct FrameOutput {
    /// Rows changed this frame, with their captions.
    pub changed: Vec<(Id, String)>,
    pub resets: Vec<Vec<String>>,
}

pub fn begin_frame(
    tab: Tab,
    current: Arc<Value>,
    baseline: Arc<Value>,
    learned: Arc<HashMap<Id, Vec<String>>>,
) {
    FRAME.with_borrow_mut(|f| {
        *f = Frame {
            tab,
            current,
            baseline,
            learned,
            ..Default::default()
        }
    });
}

pub fn end_frame() -> FrameOutput {
    FRAME.with_borrow_mut(|f| FrameOutput {
        changed: std::mem::take(&mut f.changed),
        resets: std::mem::take(&mut f.resets),
    })
}

/// Stable identity of a control row: its section plus its caption.
pub fn row_id(caption: &str) -> Id {
    FRAME.with_borrow(|f| Id::new(("inspector-row", f.section.as_str(), caption)))
}

thread_local! {
    /// Set by a row caption (double-click / context menu) for the widget that
    /// follows it; the widget reports back whether it differs from stock.
    static ROW: std::cell::Cell<(bool, Option<bool>)> = const { std::cell::Cell::new((false, None)) };
}

/// Start a row: `reset` asks the row's value widget to restore its stock value.
pub fn begin_row(reset: bool) {
    ROW.set((reset, None));
}

/// Called by a stock-aware widget: take the pending reset request.
pub fn take_row_reset() -> bool {
    let (reset, modified) = ROW.get();
    ROW.set((false, modified));
    reset
}

/// Called by a stock-aware widget once it knows its value.
pub fn set_row_modified(modified: bool) {
    let (reset, _) = ROW.get();
    ROW.set((reset, Some(modified)));
}

/// What the row's widget reported, if it knows its stock value.
pub fn row_stock_modified() -> Option<bool> {
    ROW.get().1
}

pub fn note_changed(id: Id, caption: &str) {
    FRAME.with_borrow_mut(|f| f.changed.push((id, caption.to_owned())));
}

/// Whether a learned row differs from the stock default.
pub fn row_modified(id: Id) -> bool {
    FRAME.with_borrow(|f| {
        f.learned
            .get(&id)
            .is_some_and(|ptrs| differs(&f.current, &f.baseline, ptrs))
    })
}

pub fn request_row_reset(id: Id) {
    FRAME.with_borrow_mut(|f| {
        if let Some(ptrs) = f.learned.get(&id).cloned() {
            f.resets.push(ptrs);
        }
    });
}

fn differs(current: &Value, baseline: &Value, pointers: &[String]) -> bool {
    pointers
        .iter()
        .any(|p| match (current.pointer(p), baseline.pointer(p)) {
            (Some(a), Some(b)) => a != b,
            _ => false,
        })
}

/// Leaf JSON pointers whose values differ between `a` and `b`.
pub fn diff_pointers(a: &Value, b: &Value) -> Vec<String> {
    fn walk(a: &Value, b: &Value, path: &mut String, out: &mut Vec<String>) {
        match (a, b) {
            (Value::Object(x), Value::Object(y)) => {
                for (k, va) in x {
                    let len = path.len();
                    path.push('/');
                    path.push_str(&k.replace('~', "~0").replace('/', "~1"));
                    match y.get(k) {
                        Some(vb) => walk(va, vb, path, out),
                        None => out.push(path.clone()),
                    }
                    path.truncate(len);
                }
            }
            _ if a != b => out.push(path.clone()),
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(a, b, &mut String::new(), &mut out);
    out
}

/// Copy the stock-default value at each pointer into `target`.
pub fn reset_pointers(target: &mut Value, baseline: &Value, pointers: &[String]) {
    for p in pointers {
        if let (Some(slot), Some(value)) = (target.pointer_mut(p), baseline.pointer(p)) {
            *slot = value.clone();
        }
    }
}

/// Section header text in the semibold UI face.
pub fn title(text: &str) -> RichText {
    RichText::new(text).family(egui::FontFamily::Name(theme::SEMIBOLD.into()))
}

/// A collapsible inspector section. Filters itself by tab, hosts its enable
/// checkbox in the header, fades its body when the stage is off or unused by
/// the workflow, and offers a reset when any of its parameters differ from the
/// stock default.
pub struct Section<'a> {
    title: &'a str,
    tab: Option<Tab>,
    help: &'a str,
    default_open: bool,
    active: Option<(bool, &'a str)>,
    unused: Option<&'a str>,
    pointers: &'a [&'a str],
    nested: bool,
}

impl<'a> Section<'a> {
    pub fn new(title: &'a str, tab: Tab, help: &'a str) -> Self {
        Self {
            title,
            tab: Some(tab),
            help,
            default_open: true,
            active: None,
            unused: None,
            pointers: &[],
            nested: false,
        }
    }
    /// A subsection inside another section's body (no tab filter, lighter header).
    pub fn nested(title: &'a str, help: &'a str) -> Self {
        Self {
            tab: None,
            nested: true,
            default_open: false,
            ..Self::new(title, Tab::Input, help)
        }
    }
    pub fn closed(mut self) -> Self {
        self.default_open = false;
        self
    }
    /// Header checkbox for the stage's `active` flag. Starts collapsed when off.
    pub fn active(mut self, active: bool, help: &'a str) -> Self {
        self.active = Some((active, help));
        self
    }
    /// Grey the section out with `reason` when the workflow skips this stage.
    pub fn used(mut self, used: bool, reason: &'a str) -> Self {
        self.unused = (!used).then_some(reason);
        self
    }
    /// JSON pointers into `RuntimeParams` covered by this section's reset.
    pub fn pointers(mut self, pointers: &'a [&'a str]) -> Self {
        self.pointers = pointers;
        self
    }
    /// Draw the section. Returns `true` when the header checkbox was toggled;
    /// the caller flips its own `active` flag so the body closure can keep its
    /// mutable borrow of the same struct.
    pub fn show(self, ui: &mut Ui, add_body: impl FnOnce(&mut Ui)) -> bool {
        let tab = FRAME.with_borrow(|f| f.tab);
        if self.tab.is_some_and(|t| t != tab) {
            return false;
        }
        let pointers: Vec<String> = self.pointers.iter().map(|p| p.to_string()).collect();
        let modified = FRAME.with_borrow(|f| differs(&f.current, &f.baseline, &pointers));
        let id = ui.make_persistent_id(("inspector-section", self.title));
        let default_open =
            self.default_open && self.active.is_none_or(|(on, _)| on) && self.unused.is_none();
        let enabled = self.active.is_none_or(|(on, _)| on) && self.unused.is_none();
        let mut toggled = false;

        if !self.nested {
            ui.add_space(4.0);
        }
        let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            id,
            default_open,
        );
        let header = ui.horizontal(|ui| {
            state.show_toggle_button(ui, egui::collapsing_header::paint_default_icon);
            if let Some((mut on, help)) = self.active
                && ui.checkbox(&mut on, "").on_hover_text(help).changed()
            {
                toggled = true;
                if on {
                    state.set_open(true);
                }
            }
            let mut text = if self.nested {
                RichText::new(self.title)
            } else {
                title(self.title).size(14.0)
            };
            if !enabled {
                text = text.color(theme::TEXT_FAINT);
            }
            let label = ui
                .add(
                    egui::Label::new(text)
                        .sense(egui::Sense::click())
                        .selectable(false),
                )
                .on_hover_text(match self.unused {
                    Some(reason) => format!("{}\n\n{reason}", self.help),
                    None => self.help.to_owned(),
                });
            if label.clicked() {
                state.toggle(ui);
            }
            if modified {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("↺")
                        .on_hover_text("Reset this section to the stock defaults")
                        .clicked()
                    {
                        FRAME.with_borrow_mut(|f| f.resets.push(pointers.clone()));
                    }
                    ui.label(RichText::new("●").color(theme::ACCENT).size(8.0))
                        .on_hover_text("Changed from the stock defaults");
                });
            }
        });
        header.response.context_menu(|ui| {
            if ui
                .add_enabled(
                    modified,
                    egui::Button::new("Reset section to stock defaults"),
                )
                .clicked()
            {
                FRAME.with_borrow_mut(|f| f.resets.push(pointers.clone()));
                ui.close_menu();
            }
        });
        let previous =
            FRAME.with_borrow_mut(|f| std::mem::replace(&mut f.section, self.title.into()));
        state.show_body_indented(&header.response, ui, |ui| {
            if let Some(reason) = self.unused {
                ui.label(
                    RichText::new(reason)
                        .italics()
                        .small()
                        .color(theme::TEXT_WEAK),
                );
            }
            // Faded rather than disabled: values stay editable so a stage can be
            // set up before it is switched on or the workflow changes.
            if !enabled {
                ui.multiply_opacity(0.55);
            }
            add_body(ui);
        });
        FRAME.with_borrow_mut(|f| f.section = previous);
        if !self.nested {
            ui.add_space(2.0);
        }
        toggled
    }
}
