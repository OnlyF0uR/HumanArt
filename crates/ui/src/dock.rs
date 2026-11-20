//! Docking `[UI-05]`: dockable, tabbed, splittable panels with serialisable layouts and floating
//! panels inside the main window.
//!
//! A [`DockLayout`] is a tree: a [`DockNode::Split`] divides its area between two children along
//! an axis at a ratio, a [`DockNode::Tabs`] group shows one of its panels under a tab bar. Floating
//! panels are tab groups in their own rectangles above the docked ones. Panels are named by stable
//! ids (`"viewport.1"`, `"outliner"`); [`dock_area`] asks the application for a panel's title and
//! builds its content.
//!
//! **Pointer.** Drag a tab onto another group: onto its middle to add it as a tab, onto an edge to
//! split that group there (the drop zone is highlighted), outside every group to float it. Drag a
//! divider to resize; drag a floating panel's bar to move it and its corner to resize it.
//!
//! **Keyboard** `[UI-09]`. A tab bar is a Tab stop: Left / Right switch tabs, Shift+F10 or the
//! context-menu key opens the tab menu (split left / right / up / down, float, dock). F6 and
//! Shift+F6 jump between the panels' tab bars. A divider is a Tab stop: the arrows move it
//! (Shift: finely).
//!
//! **Files.** [`DockLayout::to_json`] / [`DockLayout::from_json`] save and restore a layout (one per
//! workspace, stored by the application):
//!
//! ```json
//! {"version": 1,
//!  "root": {"split": "row", "ratio": 0.2,
//!           "first": {"tabs": ["outliner"], "active": 0},
//!           "second": {"tabs": ["viewport.1", "viewport.2"], "active": 0}},
//!  "floating": [{"rect": [80, 60, 320, 240], "tabs": {"tabs": ["console"], "active": 0}}]}
//! ```
//!
//! Reading checks every count, depth and name length against the documented limits before using
//! it, drops panels the application does not know, rejects duplicates, and repairs ratios and
//! rectangles; a damaged file is an error and never a panic `[ROB-01]`.
//!
//! Multi-window: the reference application has a single window, so panels float inside the main
//! window and are not detached to a second OS window (the optional part of `[UI-05]`).

use humanart_io::json::{self, Node, Object, Value};

use crate::error::UiError;
use crate::geom::{Insets, Point, Rect};
use crate::input::{Key, Modifiers};
use crate::kit;
use crate::layout::{Align as BoxAlign, Dim, Direction, NodeId, Style};
use crate::paint::Layer;
use crate::popup::{MenuItem, context_menu};
use crate::text::Align;
use crate::ui::{Anchor, Area, CursorIcon, Id, Role, SemanticValue, Sense, Ui};

/// Most nodes in a layout file.
pub const MAX_DOCK_NODES: usize = 256;
/// Deepest nesting of splits (also enforced when docking).
pub const MAX_DOCK_DEPTH: usize = 16;
/// Longest panel name, bytes.
pub const MAX_PANEL_NAME: usize = 64;
/// Most panels in one tab group.
pub const MAX_TABS: usize = 64;
/// Most floating panels.
pub const MAX_FLOATING: usize = 16;
/// The smallest pane a divider leaves, logical pixels.
pub const MIN_PANE: f32 = 80.0;
/// The layout file format version.
pub const DOCK_FORMAT_VERSION: u64 = 1;
/// Thickness of a divider.
const DIVIDER: f32 = 6.0;
/// Height of a floating panel's tab bar and the side of its resize grip.
const GRIP: f32 = 14.0;

/// How a split divides its area.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitAxis {
    /// Side by side (a vertical divider).
    Row,
    /// One above the other (a horizontal divider).
    Column,
}

/// Where a dragged panel goes relative to a target group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropZone {
    /// Into the group as a tab.
    Center,
    /// A new group left of it.
    Left,
    /// A new group right of it.
    Right,
    /// A new group above it.
    Top,
    /// A new group below it.
    Bottom,
}

/// A node of the docking tree.
#[derive(Clone, Debug, PartialEq)]
pub enum DockNode {
    /// Two children sharing the area along `axis`; `ratio` (0.05 to 0.95) goes to `first`.
    Split {
        /// The direction of the division.
        axis: SplitAxis,
        /// The first child's share.
        ratio: f32,
        /// Left or top child.
        first: Box<DockNode>,
        /// Right or bottom child.
        second: Box<DockNode>,
    },
    /// A tab group showing `panels[active]`.
    Tabs {
        /// Panel ids in tab order.
        panels: Vec<String>,
        /// The shown tab.
        active: usize,
    },
}

impl DockNode {
    /// A tab group of `panels`.
    pub fn tabs(panels: &[&str]) -> Self {
        Self::Tabs { panels: panels.iter().map(|p| (*p).to_owned()).collect(), active: 0 }
    }

    /// A split of two nodes.
    pub fn split(axis: SplitAxis, ratio: f32, first: Self, second: Self) -> Self {
        Self::Split { axis, ratio: clean_ratio(ratio), first: Box::new(first), second: Box::new(second) }
    }

    fn is_empty_tabs(&self) -> bool {
        matches!(self, Self::Tabs { panels, .. } if panels.is_empty())
    }

    fn contains(&self, panel: &str) -> bool {
        match self {
            Self::Tabs { panels, .. } => panels.iter().any(|p| p == panel),
            Self::Split { first, second, .. } => first.contains(panel) || second.contains(panel),
        }
    }

    /// The tab group holding `panel` and its depth.
    fn group_mut(&mut self, panel: &str, depth: usize) -> Option<(&mut Self, usize)> {
        match self {
            Self::Tabs { panels, .. } => panels.iter().any(|p| p == panel).then_some((self, depth)),
            Self::Split { first, second, .. } => {
                if first.contains(panel) {
                    first.group_mut(panel, depth + 1)
                } else {
                    second.group_mut(panel, depth + 1)
                }
            }
        }
    }

    fn remove(&mut self, panel: &str) -> bool {
        match self {
            Self::Tabs { panels, active } => {
                let Some(i) = panels.iter().position(|p| p == panel) else {
                    return false;
                };
                panels.remove(i);
                if *active > i || *active >= panels.len() {
                    *active = active.saturating_sub(1);
                }
                true
            }
            Self::Split { first, second, .. } => first.remove(panel) || second.remove(panel),
        }
    }

    /// Replaces splits that have an empty side by the other side, repairs ratios and active tabs.
    fn collapse(&mut self) {
        if let Self::Split { first, second, ratio, .. } = self {
            first.collapse();
            second.collapse();
            *ratio = clean_ratio(*ratio);
            let keep = if first.is_empty_tabs() {
                Some(std::mem::replace(second.as_mut(), Self::Tabs { panels: Vec::new(), active: 0 }))
            } else if second.is_empty_tabs() {
                Some(std::mem::replace(first.as_mut(), Self::Tabs { panels: Vec::new(), active: 0 }))
            } else {
                None
            };
            if let Some(k) = keep {
                *self = k;
            }
        } else if let Self::Tabs { panels, active } = self {
            *active = (*active).min(panels.len().saturating_sub(1));
        }
    }

    fn for_each_panel<'a>(&'a self, f: &mut dyn FnMut(&'a str)) {
        match self {
            Self::Tabs { panels, .. } => panels.iter().for_each(|p| f(p)),
            Self::Split { first, second, .. } => {
                first.for_each_panel(f);
                second.for_each_panel(f);
            }
        }
    }

    fn first_panel(&self) -> Option<&str> {
        match self {
            Self::Tabs { panels, .. } => panels.first().map(String::as_str),
            Self::Split { first, second, .. } => first.first_panel().or_else(|| second.first_panel()),
        }
    }

    fn to_value(&self) -> Value {
        let mut o = Object::new();
        match self {
            Self::Tabs { panels, active } => {
                o.insert("tabs", Value::Array(panels.iter().map(|p| Value::from(p.as_str())).collect()));
                o.insert("active", *active);
            }
            Self::Split { axis, ratio, first, second } => {
                o.insert("split", if *axis == SplitAxis::Row { "row" } else { "column" });
                o.insert("ratio", f64::from((*ratio * 10_000.0).round() / 10_000.0));
                o.insert("first", first.to_value());
                o.insert("second", second.to_value());
            }
        }
        Value::from(o)
    }
}

/// A ratio inside 0.05..0.95 (0.5 when not finite).
fn clean_ratio(r: f32) -> f32 {
    if r.is_finite() { r.clamp(0.05, 0.95) } else { 0.5 }
}

/// A tab group floating in its own rectangle above the docked panels.
#[derive(Clone, Debug, PartialEq)]
pub struct FloatingPanel {
    /// Its rectangle (logical pixels in the window).
    pub rect: Rect,
    /// Its tab group (always [`DockNode::Tabs`]).
    pub group: DockNode,
}

/// A docking layout `[UI-05]`: the docked tree and the floating panels.
#[derive(Clone, Debug, PartialEq)]
pub struct DockLayout {
    root: DockNode,
    floating: Vec<FloatingPanel>,
}

impl DockLayout {
    /// One tab group of `panels`.
    pub fn new(panels: &[&str]) -> Self {
        Self { root: DockNode::tabs(panels), floating: Vec::new() }
    }

    /// A layout with `root` as its tree (no floating panels). Duplicate panels are removed.
    pub fn from_root(root: DockNode) -> Self {
        let mut layout = Self { root, floating: Vec::new() };
        layout.dedup();
        layout
    }

    /// The docked tree.
    pub fn root(&self) -> &DockNode {
        &self.root
    }

    /// The floating panels.
    pub fn floating(&self) -> &[FloatingPanel] {
        &self.floating
    }

    /// Whether `panel` is in the layout (docked or floating).
    pub fn contains(&self, panel: &str) -> bool {
        self.root.contains(panel) || self.floating.iter().any(|f| f.group.contains(panel))
    }

    /// Every panel, docked ones first.
    pub fn panels(&self) -> Vec<&str> {
        let mut out = Vec::new();
        self.root.for_each_panel(&mut |p| out.push(p));
        for f in &self.floating {
            f.group.for_each_panel(&mut |p| out.push(p));
        }
        out
    }

    fn dedup(&mut self) {
        let mut seen: Vec<String> = Vec::new();
        let mut dedup_node = |n: &mut DockNode| {
            fn walk(n: &mut DockNode, seen: &mut Vec<String>) {
                match n {
                    DockNode::Tabs { panels, .. } => panels.retain(|p| {
                        if seen.contains(p) || p.len() > MAX_PANEL_NAME || p.is_empty() {
                            false
                        } else {
                            seen.push(p.clone());
                            true
                        }
                    }),
                    DockNode::Split { first, second, .. } => {
                        walk(first, seen);
                        walk(second, seen);
                    }
                }
            }
            walk(n, &mut seen);
        };
        dedup_node(&mut self.root);
        for f in &mut self.floating {
            dedup_node(&mut f.group);
        }
        self.normalize();
    }

    fn normalize(&mut self) {
        self.root.collapse();
        for f in &mut self.floating {
            f.group.collapse();
        }
        self.floating.retain(|f| !f.group.is_empty_tabs() && matches!(f.group, DockNode::Tabs { .. }));
    }

    /// Adds `panel` as a tab of the first docked group when it is not in the layout yet.
    pub fn ensure(&mut self, panel: &str) {
        if self.contains(panel) || panel.is_empty() || panel.len() > MAX_PANEL_NAME {
            return;
        }
        let mut node = &mut self.root;
        loop {
            match node {
                DockNode::Split { first, .. } => node = first,
                DockNode::Tabs { panels, .. } => {
                    if panels.len() < MAX_TABS {
                        panels.push(panel.to_owned());
                    }
                    return;
                }
            }
        }
    }

    /// Removes `panel`; returns whether it was there. Empty groups disappear.
    pub fn remove(&mut self, panel: &str) -> bool {
        let mut found = self.root.remove(panel);
        for f in &mut self.floating {
            found |= f.group.remove(panel);
        }
        self.normalize();
        found
    }

    /// Shows `panel` in its group.
    pub fn activate(&mut self, panel: &str) {
        let group = match self.root.group_mut(panel, 0) {
            Some((g, _)) => Some(g),
            None => self.floating.iter_mut().find(|f| f.group.contains(panel)).map(|f| &mut f.group),
        };
        if let Some(DockNode::Tabs { panels, active }) = group
            && let Some(i) = panels.iter().position(|p| p == panel)
        {
            *active = i;
        }
    }

    /// The panel shown by the group holding `panel` (its active tab).
    pub fn active_of(&self, panel: &str) -> Option<&str> {
        active_in(&self.root, panel).or_else(|| self.floating.iter().find_map(|f| active_in(&f.group, panel)))
    }

    /// Moves `panel` next to (or into) the group holding `target` `[UI-05]`. A split deeper than
    /// [`MAX_DOCK_DEPTH`] becomes a tab instead. Errors when `target` is not docked or is `panel`
    /// alone in its group.
    pub fn dock(&mut self, panel: &str, target: &str, zone: DropZone) -> Result<(), UiError> {
        if panel.is_empty() || panel.len() > MAX_PANEL_NAME {
            return Err(UiError::DockLayout(format!("panel name of {} bytes", panel.len())));
        }
        if !self.root.contains(target) {
            return Err(UiError::DockLayout(format!("\"{target}\" is not docked")));
        }
        if panel == target || self.root.group_mut(target, 0).is_some_and(|(g, _)| g.contains(panel)) {
            let alone =
                matches!(self.root.group_mut(target, 0), Some((DockNode::Tabs { panels, .. }, _)) if panels.len() < 2);
            if alone {
                return Err(UiError::DockLayout(format!("\"{panel}\" cannot be docked beside itself")));
            }
            if zone == DropZone::Center {
                self.activate(panel);
                return Ok(());
            }
        }
        // Take the panel out (it may be floating), keep the target's group, then place it.
        self.root.remove(panel);
        for f in &mut self.floating {
            f.group.remove(panel);
        }
        self.normalize();
        let Some((group, depth)) = self.root.group_mut(target, 0) else {
            return Err(UiError::DockLayout(format!("\"{target}\" vanished while docking")));
        };
        let zone = if depth + 1 >= MAX_DOCK_DEPTH { DropZone::Center } else { zone };
        match zone {
            DropZone::Center => {
                if let DockNode::Tabs { panels, active } = group {
                    if panels.len() >= MAX_TABS {
                        return Err(UiError::DockLayout(format!("a group holds at most {MAX_TABS} tabs")));
                    }
                    panels.push(panel.to_owned());
                    *active = panels.len() - 1;
                }
            }
            side => {
                let old = std::mem::replace(group, DockNode::tabs(&[]));
                let new = DockNode::tabs(&[panel]);
                let (axis, new_first) = match side {
                    DropZone::Left => (SplitAxis::Row, true),
                    DropZone::Right => (SplitAxis::Row, false),
                    DropZone::Top => (SplitAxis::Column, true),
                    _ => (SplitAxis::Column, false),
                };
                *group =
                    if new_first { DockNode::split(axis, 0.5, new, old) } else { DockNode::split(axis, 0.5, old, new) };
            }
        }
        Ok(())
    }

    /// Docks `panel` (floating or not) as a tab of the first docked group, or as the only group
    /// when nothing is docked. Returns whether the layout changed.
    pub fn dock_to_main(&mut self, panel: &str) -> bool {
        if !self.contains(panel) {
            return false;
        }
        match self.root.first_panel().map(str::to_owned) {
            Some(first) if first != panel => self.dock(panel, &first, DropZone::Center).is_ok(),
            Some(_) => false,
            None => {
                for f in &mut self.floating {
                    f.group.remove(panel);
                }
                self.root = DockNode::tabs(&[panel]);
                self.normalize();
                true
            }
        }
    }

    /// Floats `panel` in `rect` (its own tab group above the docked panels).
    pub fn float(&mut self, panel: &str, rect: Rect) -> Result<(), UiError> {
        if !self.contains(panel) {
            return Err(UiError::DockLayout(format!("\"{panel}\" is not in the layout")));
        }
        if self.floating.len() >= MAX_FLOATING {
            return Err(UiError::DockLayout(format!("at most {MAX_FLOATING} floating panels")));
        }
        self.root.remove(panel);
        for f in &mut self.floating {
            f.group.remove(panel);
        }
        self.normalize();
        self.floating.push(FloatingPanel { rect: clean_rect(rect), group: DockNode::tabs(&[panel]) });
        Ok(())
    }

    /// The layout file (see the module documentation).
    pub fn to_json(&self) -> Result<String, UiError> {
        let mut o = Object::new();
        o.insert("version", DOCK_FORMAT_VERSION);
        o.insert("root", self.root.to_value());
        let floating: Vec<Value> = self
            .floating
            .iter()
            .map(|f| {
                let mut fo = Object::new();
                let r = f.rect;
                fo.insert("rect", Value::Array(vec![r.x.into(), r.y.into(), r.w.into(), r.h.into()]));
                fo.insert("tabs", f.group.to_value());
                Value::from(fo)
            })
            .collect();
        o.insert("floating", Value::Array(floating));
        json::to_string_pretty(&Value::from(o)).map_err(|e| UiError::DockLayout(format!("{e}")))
    }

    /// Reads a layout file, keeping only the panels in `known` (a saved layout may name a panel the
    /// application no longer has). Errors on a damaged or over-limit file.
    pub fn from_json(text: &str, known: &[&str]) -> Result<Self, UiError> {
        let err = |e: &dyn std::fmt::Display| UiError::DockLayout(format!("{e}"));
        let value = json::parse_str(text).map_err(|e| err(&e))?;
        let root = value.node();
        root.deny_unknown(&["version", "root", "floating"]).map_err(|e| err(&e))?;
        let version = root.get::<u64>("version").map_err(|e| err(&e))?;
        if version != DOCK_FORMAT_VERSION {
            return Err(UiError::DockLayout(format!("version {version}, expected {DOCK_FORMAT_VERSION}")));
        }
        let mut ctx = ParseCtx { known, count: 0, seen: Vec::new() };
        let tree = parse_node(&root.field("root").map_err(|e| err(&e))?, 0, &mut ctx)?;
        let mut floating = Vec::new();
        if let Some(list) = root.opt_field("floating").map_err(|e| err(&e))? {
            let n = list.array_len().map_err(|e| err(&e))?;
            if n > MAX_FLOATING {
                return Err(UiError::DockLayout(format!("{n} floating panels, at most {MAX_FLOATING}")));
            }
            for item in list.elements().map_err(|e| err(&e))? {
                item.deny_unknown(&["rect", "tabs"]).map_err(|e| err(&e))?;
                let mut r = [0.0_f32; 4];
                let rect = item.field("rect").map_err(|e| err(&e))?;
                if rect.array_len().map_err(|e| err(&e))? != 4 {
                    return Err(UiError::DockLayout("a floating rectangle needs 4 numbers".into()));
                }
                rect.read_into(&mut r).map_err(|e| err(&e))?;
                let group = parse_node(&item.field("tabs").map_err(|e| err(&e))?, 0, &mut ctx)?;
                if !matches!(group, DockNode::Tabs { .. }) {
                    return Err(UiError::DockLayout("a floating panel holds one tab group".into()));
                }
                floating.push(FloatingPanel { rect: clean_rect(Rect::new(r[0], r[1], r[2], r[3])), group });
            }
        }
        let mut layout = Self { root: tree, floating };
        layout.normalize();
        Ok(layout)
    }
}

/// The active tab of the group in `n` holding `panel`.
fn active_in<'a>(n: &'a DockNode, panel: &str) -> Option<&'a str> {
    match n {
        DockNode::Tabs { panels, active } => {
            if panels.iter().any(|p| p == panel) {
                panels.get(*active).map(String::as_str)
            } else {
                None
            }
        }
        DockNode::Split { first, second, .. } => active_in(first, panel).or_else(|| active_in(second, panel)),
    }
}

/// A floating rectangle with finite values and a usable size.
fn clean_rect(r: Rect) -> Rect {
    let f = |v: f32, lo: f32, hi: f32, d: f32| if v.is_finite() { v.clamp(lo, hi) } else { d };
    Rect::new(
        f(r.x, -10_000.0, 10_000.0, 40.0),
        f(r.y, -10_000.0, 10_000.0, 40.0),
        f(r.w, 160.0, 8000.0, 320.0),
        f(r.h, 100.0, 8000.0, 240.0),
    )
}

/// What a layout read keeps track of.
struct ParseCtx<'a> {
    known: &'a [&'a str],
    count: usize,
    seen: Vec<String>,
}

/// One node of a layout file; depth and node count are checked before descending.
fn parse_node(node: &Node<'_, '_>, depth: usize, ctx: &mut ParseCtx<'_>) -> Result<DockNode, UiError> {
    let err = |e: &dyn std::fmt::Display| UiError::DockLayout(format!("{e}"));
    if depth > MAX_DOCK_DEPTH {
        return Err(UiError::DockLayout(format!("splits nested deeper than {MAX_DOCK_DEPTH}")));
    }
    ctx.count += 1;
    if ctx.count > MAX_DOCK_NODES {
        return Err(UiError::DockLayout(format!("more than {MAX_DOCK_NODES} nodes")));
    }
    if node.has("tabs") {
        node.deny_unknown(&["tabs", "active"]).map_err(|e| err(&e))?;
        let list = node.field("tabs").map_err(|e| err(&e))?;
        let n = list.array_len().map_err(|e| err(&e))?;
        if n > MAX_TABS {
            return Err(UiError::DockLayout(format!("{n} tabs in a group, at most {MAX_TABS}")));
        }
        let mut panels = Vec::with_capacity(n);
        for item in list.elements().map_err(|e| err(&e))? {
            let name = item.str().map_err(|e| err(&e))?;
            if name.len() > MAX_PANEL_NAME {
                return Err(UiError::DockLayout(format!(
                    "a panel name of {} bytes, at most {MAX_PANEL_NAME}",
                    name.len()
                )));
            }
            if ctx.seen.iter().any(|s| s == name) {
                return Err(UiError::DockLayout(format!("panel \"{name}\" appears twice")));
            }
            ctx.seen.push(name.to_owned());
            if ctx.known.contains(&name) {
                panels.push(name.to_owned());
            }
        }
        let active = usize::try_from(node.get_or::<u64>("active", 0).map_err(|e| err(&e))?).unwrap_or(0);
        let active = active.min(panels.len().saturating_sub(1));
        return Ok(DockNode::Tabs { panels, active });
    }
    node.deny_unknown(&["split", "ratio", "first", "second"]).map_err(|e| err(&e))?;
    let axis = node
        .field("split")
        .map_err(|e| err(&e))?
        .choice(&[("row", SplitAxis::Row), ("column", SplitAxis::Column)])
        .map_err(|e| err(&e))?;
    let ratio = clean_ratio(node.get_or::<f32>("ratio", 0.5).map_err(|e| err(&e))?);
    let first = parse_node(&node.field("first").map_err(|e| err(&e))?, depth + 1, ctx)?;
    let second = parse_node(&node.field("second").map_err(|e| err(&e))?, depth + 1, ctx)?;
    Ok(DockNode::Split { axis, ratio, first: Box::new(first), second: Box::new(second) })
}

// ---------------------------------------------------------------------------------------------
// The docking area.

/// A tab being dragged and the tab bars of last frame (for F6), in [`Ui::memory`].
#[derive(Clone, Debug, Default)]
struct DockDrag {
    panel: Option<String>,
    bars: Vec<Id>,
}

/// A structural change found while building, applied after the tree is drawn.
enum Pending {
    Dock(String, String, DropZone),
    Float(String, Rect),
}

/// What a docking area did this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DockResponse {
    /// The layout changed (save it with the workspace).
    pub changed: bool,
}

/// The drop zone of `p` over a group occupying `r`: the middle third is the centre, the rest goes
/// to the nearest edge.
pub fn drop_zone(r: Rect, p: Point) -> DropZone {
    let (fx, fy) = ((p.x - r.x) / r.w.max(1.0), (p.y - r.y) / r.h.max(1.0));
    if (0.3..=0.7).contains(&fx) && (0.3..=0.7).contains(&fy) {
        return DropZone::Center;
    }
    let d = [(fx, DropZone::Left), (1.0 - fx, DropZone::Right), (fy, DropZone::Top), (1.0 - fy, DropZone::Bottom)];
    d.iter().min_by(|a, b| a.0.total_cmp(&b.0)).map_or(DropZone::Center, |x| x.1)
}

/// The part of `r` a zone would give the dropped panel (for the highlight).
fn zone_rect(r: Rect, z: DropZone) -> Rect {
    match z {
        DropZone::Center => r,
        DropZone::Left => Rect::new(r.x, r.y, r.w * 0.5, r.h),
        DropZone::Right => Rect::new(r.x + r.w * 0.5, r.y, r.w * 0.5, r.h),
        DropZone::Top => Rect::new(r.x, r.y, r.w, r.h * 0.5),
        DropZone::Bottom => Rect::new(r.x, r.y + r.h * 0.5, r.w, r.h * 0.5),
    }
}

/// The building context of one docking area.
struct Ctx<'a> {
    id: Id,
    title: &'a dyn Fn(&str) -> &str,
    content: &'a mut dyn FnMut(&mut Ui, &str),
    dragging: Option<&'a str>,
    drag_started: Option<String>,
    bars: Vec<Id>,
    pending: Option<Pending>,
    changed: bool,
}

/// The id of the content box of `panel` in the docking area `dock` (its rectangle as of last frame
/// is `ui.state(id).rect`: where the application renders a viewport panel).
pub fn panel_id(dock: Id, panel: &str) -> Id {
    dock.with("panel").with(panel)
}

/// The rectangle of `panel`'s content in the docking area labelled `id_label` (in the current id
/// scope) as of last frame, if it was shown.
pub fn panel_rect(ui: &Ui, id_label: &str, panel: &str) -> Option<Rect> {
    ui.state(panel_id(ui.id(id_label), panel)).map(|s| s.rect).filter(|r| r.w > 0.0 && r.h > 0.0)
}

/// Builds a docking area `[UI-05]` filling its parent: the docked tree, then the floating panels.
/// `title(panel)` gives a tab's text and `content(ui, panel)` builds the shown panels (inside a
/// clipped column, in an id scope of their own).
pub fn dock_area(
    ui: &mut Ui,
    id_label: &str,
    layout: &mut DockLayout,
    title: &dyn Fn(&str) -> &str,
    content: &mut dyn FnMut(&mut Ui, &str),
) -> DockResponse {
    let id = ui.id(id_label);
    let mut drag = std::mem::take(ui.memory::<DockDrag>(id));
    let released = ui.pointer_released() || !ui.pointer_down();
    let drop_at = if released { ui.pointer() } else { None };
    // F6 / Shift+F6 move the keyboard focus to the next / previous panel's tab bar [UI-09].
    if let Some(mods) = ui.consume_key_any(Key::F(6))
        && !drag.bars.is_empty()
    {
        let n = drag.bars.len();
        let at = ui.focused().and_then(|f| drag.bars.iter().position(|b| *b == f));
        let next = match (at, mods.shift) {
            (Some(i), false) => (i + 1) % n,
            (Some(i), true) => (i + n - 1) % n,
            (None, false) => 0,
            (None, true) => n - 1,
        };
        ui.request_focus(drag.bars[next]);
    }
    let mut bars = std::mem::take(&mut drag.bars);
    bars.clear();
    let mut ctx = Ctx {
        id,
        title,
        content,
        dragging: drag.panel.as_deref(),
        drag_started: None,
        bars,
        pending: None,
        changed: false,
    };
    let area =
        ui.begin(Style { width: Dim::Percent(100.0), height: Dim::Percent(100.0), grow: 1.0, ..Style::column() });
    let _ = area;
    let root_style = Style { grow: 1.0, basis: Dim::Px(0.0), ..Style::default() };
    show_node(ui, &mut ctx, &mut layout.root, id.with("root"), root_style, 0);
    ui.end();
    for (k, f) in layout.floating.iter_mut().enumerate() {
        show_floating(ui, &mut ctx, f, id.with("float").with_index(k as u64));
    }
    // The dragged tab: a ghost at the pointer, and the drop when the button comes up.
    if let (Some(panel), Some(p)) = (ctx.dragging, ui.pointer()) {
        let theme = *ui.theme();
        let (m, pal) = (theme.metrics, theme.palette);
        let text = (ctx.title)(panel);
        let w = ui.measure_text(text, theme.text.label).w + 2.0 * m.space_5;
        let ghost =
            ui.begin_overlay(Rect::new(p.x + 12.0, p.y + 8.0, w, m.row_height), Layer::Tooltip, Style::default());
        kit::card(ui, ghost, theme.shadows.menu, pal.surface, m.radius_sm, 0.92);
        kit::text(ui, ghost, Area::FULL, text, theme.text.label, pal.text, Align::Center);
        ui.end_overlay();
        ui.set_cursor(CursorIcon::Grabbing);
    }
    if let (Some(panel), Some(p)) = (ctx.dragging, drop_at) {
        let target = find_group(ui, layout, id, p);
        ctx.pending = Some(match target {
            Some((name, rect)) => Pending::Dock(panel.to_owned(), name, drop_zone(rect, p)),
            None => Pending::Float(panel.to_owned(), Rect::new(p.x - 40.0, p.y - 12.0, 360.0, 260.0)),
        });
    }
    let started = ctx.drag_started.take();
    let bars = std::mem::take(&mut ctx.bars);
    let pending = ctx.pending.take();
    let mut changed = ctx.changed;
    if drop_at.is_some() {
        drag.panel = None;
    }
    if let Some(s) = started {
        drag.panel = Some(s);
    }
    if let Some(op) = pending {
        let ok = match op {
            Pending::Dock(panel, target, zone) if target.is_empty() => {
                let _ = zone;
                layout.dock_to_main(&panel)
            }
            Pending::Dock(panel, target, zone) => layout.dock(&panel, &target, zone).is_ok(),
            Pending::Float(panel, rect) => {
                let vp = ui.viewport();
                let r = Rect::new(
                    rect.x.clamp(vp.x, (vp.right() - rect.w).max(vp.x)),
                    rect.y.clamp(vp.y, (vp.bottom() - rect.h).max(vp.y)),
                    rect.w,
                    rect.h,
                );
                layout.float(&panel, r).is_ok()
            }
        };
        changed |= ok;
    }
    drag.bars = bars;
    *ui.memory::<DockDrag>(id) = drag;
    DockResponse { changed }
}

/// The group (its first panel and rectangle as of last frame) under `p`: floating ones first.
fn find_group(ui: &Ui, layout: &DockLayout, id: Id, p: Point) -> Option<(String, Rect)> {
    for (k, f) in layout.floating.iter().enumerate().rev() {
        let gid = id.with("float").with_index(k as u64).with("group");
        if let Some(s) = ui.state(gid)
            && s.rect.contains(p)
        {
            return f.group.first_panel().map(|n| (n.to_owned(), s.rect));
        }
    }
    group_under(ui, &layout.root, id.with("root"), p, 0)
}

/// The docked group under `p` (its first panel and last frame's rectangle).
fn group_under(ui: &Ui, n: &DockNode, nid: Id, p: Point, depth: usize) -> Option<(String, Rect)> {
    if depth > MAX_DOCK_DEPTH {
        return None;
    }
    match n {
        DockNode::Tabs { panels, .. } => {
            let s = ui.state(nid.with("group"))?;
            (s.rect.contains(p)).then(|| panels.first().map(|f| (f.clone(), s.rect))).flatten()
        }
        DockNode::Split { first, second, .. } => group_under(ui, first, nid.with("a"), p, depth + 1)
            .or_else(|| group_under(ui, second, nid.with("b"), p, depth + 1)),
    }
}

/// Builds one node of the docked tree in a box of `style`.
fn show_node(ui: &mut Ui, ctx: &mut Ctx<'_>, node: &mut DockNode, nid: Id, style: Style, depth: usize) {
    if depth > MAX_DOCK_DEPTH {
        return;
    }
    match node {
        DockNode::Tabs { panels, active } => {
            show_group(ui, ctx, panels, active, nid, style, None);
        }
        DockNode::Split { axis, ratio, first, second } => {
            let theme = *ui.theme();
            let (m, p) = (theme.metrics, theme.palette);
            let direction = if *axis == SplitAxis::Row { Direction::Row } else { Direction::Column };
            let boxn = ui.begin(Style { direction, ..style });
            let br = ui.interact(nid, boxn, Sense::HOVER);
            let size = if *axis == SplitAxis::Row { br.rect.w } else { br.rect.h };
            let child = |share: f32| Style {
                grow: share,
                basis: Dim::Px(0.0),
                min_width: 0.0,
                min_height: 0.0,
                ..Style::default()
            };
            show_node(ui, ctx, first, nid.with("a"), child(*ratio), depth + 1);
            // The divider.
            let handle = ui.node(match axis {
                SplitAxis::Row => Style { width: Dim::Px(DIVIDER), shrink: 0.0, ..Style::default() },
                SplitAxis::Column => Style { height: Dim::Px(DIVIDER), shrink: 0.0, ..Style::default() },
            });
            let hr = ui.interact(nid.with("divider"), handle, Sense::DRAG);
            let usable = (size - DIVIDER).max(1.0);
            let mut r = *ratio;
            if hr.dragged {
                r += match axis {
                    SplitAxis::Row => hr.drag_delta.x,
                    SplitAxis::Column => hr.drag_delta.y,
                } / usable;
            }
            if hr.focused {
                let (less, more) = match axis {
                    SplitAxis::Row => (Key::Left, Key::Right),
                    SplitAxis::Column => (Key::Up, Key::Down),
                };
                while let Some(mods) = ui.consume_key_any(less) {
                    r -= if mods.shift { 0.005 } else { 0.02 };
                }
                while let Some(mods) = ui.consume_key_any(more) {
                    r += if mods.shift { 0.005 } else { 0.02 };
                }
            }
            if r != *ratio {
                let lo = if usable > 2.0 * MIN_PANE { MIN_PANE / usable } else { 0.05 };
                *ratio = clean_ratio(r.clamp(lo, 1.0 - lo));
                ctx.changed = true;
            }
            if hr.hovered || hr.dragged {
                ui.set_cursor(match axis {
                    SplitAxis::Row => CursorIcon::ResizeHorizontal,
                    SplitAxis::Column => CursorIcon::ResizeVertical,
                });
            }
            let t = hr.hover_t.max(hr.press_t);
            let (a, b) = match axis {
                SplitAxis::Row => {
                    ((Anchor::at(0.5, 0.0), Anchor::start(0.0)), (Anchor::at(0.5, 0.0), Anchor::end(0.0)))
                }
                SplitAxis::Column => {
                    ((Anchor::start(0.0), Anchor::at(0.5, 0.0)), (Anchor::end(0.0), Anchor::at(0.5, 0.0)))
                }
            };
            kit::line(ui, handle, a, b, m.border + t, p.border.mix(p.focus_ring, t));
            if hr.focus_visible {
                kit::ring(ui, handle, Area::FULL, 2.0);
            }
            let value = SemanticValue::Number { value: f64::from(*ratio), min: 0.05, max: 0.95 };
            ui.semantic(hr.id, Role::Custom("splitter"), "Panel divider", value);
            show_node(ui, ctx, second, nid.with("b"), child(1.0 - *ratio), depth + 1);
            ui.end();
        }
    }
}

/// Builds a tab group: its bar, its menu, its shown panel and the drop highlight while a tab is
/// dragged over it. `floating` is the group's index among the floating panels.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn show_group(
    ui: &mut Ui,
    ctx: &mut Ctx<'_>,
    panels: &mut [String],
    active: &mut usize,
    nid: Id,
    style: Style,
    floating: Option<&mut Rect>,
) -> NodeId {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let col = ui.begin(Style { direction: Direction::Column, ..style });
    let gid = nid.with("group");
    let gr = ui.interact(gid, col, Sense::HOVER);
    kit::fill(ui, col, Area::FULL, p.bg, 0.0);
    let n = panels.len();
    *active = (*active).min(n.saturating_sub(1));
    // The bar.
    let bar = ui.begin(Style {
        height: Dim::Px(m.row_height + 4.0),
        shrink: 0.0,
        align_items: BoxAlign::Stretch,
        ..Style::row()
    });
    kit::fill(ui, bar, Area::FULL, p.panel, 0.0);
    kit::line(
        ui,
        bar,
        (Anchor::start(0.0), Anchor::end(-0.5)),
        (Anchor::end(0.0), Anchor::end(-0.5)),
        m.border,
        p.border,
    );
    let mut br = ui.interact(nid.with("bar"), bar, Sense { click: false, drag: floating.is_some(), focus: true });
    ctx.bars.push(br.id);
    if br.focused && n > 0 {
        if ui.consume_key(Key::Right, Modifiers::NONE) {
            *active = (*active + 1) % n;
            ctx.changed = true;
        }
        if ui.consume_key(Key::Left, Modifiers::NONE) {
            *active = (*active + n - 1) % n;
            ctx.changed = true;
        }
    }
    for (i, panel) in panels.iter().enumerate() {
        let text = (ctx.title)(panel);
        let w = ui.measure_text(text, theme.text.label).w + 2.0 * m.space_5;
        let tab = ui.node(Style { width: Dim::Px(w), shrink: 0.0, ..Style::default() });
        let tr =
            ui.interact(nid.with("tab").with_index(i as u64), tab, Sense { click: true, drag: true, focus: false });
        if tr.pressed || tr.secondary_clicked {
            if *active != i {
                ctx.changed = true;
            }
            *active = i;
            ui.request_focus(br.id);
        }
        if tr.secondary_clicked {
            br.secondary_clicked = true;
        }
        if tr.drag_started && ctx.dragging.is_none() {
            ctx.drag_started = Some(panel.clone());
        }
        let on = i == *active;
        let dragged_away = ctx.dragging == Some(panel.as_str());
        kit::fill(ui, tab, kit::at(2.0, 3.0, w - 4.0, m.row_height - 2.0), p.hover.fade(tr.hover_t * 0.8), m.radius_sm);
        let ink = if dragged_away {
            p.text_faint
        } else if on {
            p.text
        } else {
            p.text_muted.mix(p.text, tr.hover_t)
        };
        kit::text(ui, tab, Area::FULL, text, theme.text.label, ink, Align::Center);
        if on {
            let under = Area {
                x0: Anchor::start(m.space_4),
                y0: Anchor::end(-2.0),
                x1: Anchor::end(-m.space_4),
                y1: Anchor::end(0.0),
            };
            kit::fill(ui, tab, under, p.accent, 1.0);
            if br.focus_visible {
                kit::ring(ui, tab, Area::inset(2.0), m.radius_sm);
            }
        }
        ui.semantic(tr.id, Role::Custom("tab"), text, SemanticValue::Toggle(on));
    }
    ui.node(Style { grow: 1.0, ..Style::default() });
    ui.end();
    // Dragging the bar's free space moves a floating panel.
    let is_floating = floating.is_some();
    if let Some(rect) = floating
        && br.dragged
    {
        rect.x += br.drag_delta.x;
        rect.y += br.drag_delta.y;
        ctx.changed = true;
    }
    let shown = panels.get(*active).map(String::as_str);
    let label = shown.map_or("", |s| (ctx.title)(s));
    ui.semantic(br.id, Role::Custom("tablist"), label, SemanticValue::None);
    // The tab menu (secondary click on a tab, Shift+F10 on the bar).
    let many = n > 1;
    let items = [
        MenuItem::action("Split left").enabled(many && !is_floating),
        MenuItem::action("Split right").enabled(many && !is_floating),
        MenuItem::action("Split up").enabled(many && !is_floating),
        MenuItem::action("Split down").enabled(many && !is_floating),
        MenuItem::Separator,
        if is_floating { MenuItem::action("Dock") } else { MenuItem::action("Float").enabled(n > 0) },
    ];
    if let Some(pick) = context_menu(ui, &br, &items)
        && let Some(panel) = shown
    {
        let other = panels.iter().find(|q| q.as_str() != panel).cloned();
        let zone = [DropZone::Left, DropZone::Right, DropZone::Top, DropZone::Bottom].get(pick).copied();
        ctx.pending = match (pick, zone, other) {
            (0..=3, Some(z), Some(o)) => Some(Pending::Dock(panel.to_owned(), o, z)),
            (5, _, _) if is_floating => Some(Pending::Dock(panel.to_owned(), String::new(), DropZone::Center)),
            (5, _, _) => {
                let r = gr.rect;
                Some(Pending::Float(
                    panel.to_owned(),
                    Rect::new(r.x + 40.0, r.y + 40.0, (r.w * 0.6).max(320.0), (r.h * 0.6).max(220.0)),
                ))
            }
            _ => None,
        };
    }
    // The shown panel.
    let body = ui.begin(Style { grow: 1.0, basis: Dim::Px(0.0), min_height: 0.0, ..Style::column() });
    if let Some(panel) = shown {
        ui.interact(panel_id(ctx.id, panel), body, Sense::HOVER);
        ui.push_clip(body, Area::FULL);
        ui.push_id(ctx.id.with(panel));
        (ctx.content)(ui, panel);
        ui.pop_id();
        ui.pop_clip();
    }
    ui.end();
    // The drop highlight.
    if let (Some(_), Some(ptr)) = (ctx.dragging, ui.pointer())
        && gr.rect.contains(ptr)
    {
        let z = zone_rect(gr.rect, drop_zone(gr.rect, ptr)).inset(Insets::all(4.0));
        let hl = ui.begin_overlay(z, Layer::Tooltip, Style::default());
        ui.paint(
            hl,
            crate::ui::Paint::Rect {
                area: Area::FULL,
                fill: p.accent.fade(0.14),
                radius: crate::paint::Corners::all(m.radius),
                border: 2.0,
                border_color: p.accent.fade(0.7),
            },
        );
        ui.end_overlay();
    }
    ui.end();
    col
}

/// Builds a floating panel: a card with its tab group, moved by its bar and sized by its corner.
fn show_floating(ui: &mut Ui, ctx: &mut Ctx<'_>, f: &mut FloatingPanel, fid: Id) {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let vp = ui.viewport();
    // Keep it inside the window when it fits, and at least its bar on screen when it does not.
    let fit = |pos: f32, size: f32, lo: f32, hi: f32, keep: f32| {
        if size <= hi - lo { pos.clamp(lo, hi - size) } else { pos.clamp(lo - size + keep, (hi - keep).max(lo)) }
    };
    f.rect.x = fit(f.rect.x, f.rect.w, vp.x, vp.right(), 60.0);
    f.rect.y = fit(f.rect.y, f.rect.h, vp.y, vp.bottom(), m.row_height);
    let card = ui.begin_overlay(f.rect, Layer::Popup, Style { padding: Insets::all(1.0), ..Style::column() });
    kit::card(ui, card, theme.shadows.modal, p.bg, m.radius, 1.0);
    ui.interact(fid.with("card"), card, Sense { click: true, drag: false, focus: false });
    let DockNode::Tabs { panels, active } = &mut f.group else {
        ui.end_overlay();
        return;
    };
    let style = Style { grow: 1.0, basis: Dim::Px(0.0), ..Style::default() };
    let mut rect = f.rect;
    show_group(ui, ctx, panels, active, fid, style, Some(&mut rect));
    f.rect = rect;
    // The resize grip.
    let grip = ui.node(Style {
        position: crate::layout::Position::Absolute { left: None, top: None, right: Some(0.0), bottom: Some(0.0) },
        width: Dim::Px(GRIP),
        height: Dim::Px(GRIP),
        ..Style::default()
    });
    let gr = ui.interact(fid.with("grip"), grip, Sense { click: false, drag: true, focus: false });
    if gr.dragged {
        f.rect.w = (f.rect.w + gr.drag_delta.x).clamp(160.0, 8000.0);
        f.rect.h = (f.rect.h + gr.drag_delta.y).clamp(100.0, 8000.0);
        ctx.changed = true;
    }
    if gr.hovered || gr.dragged {
        ui.set_cursor(CursorIcon::ResizeHorizontal);
    }
    kit::line(
        ui,
        grip,
        (Anchor::end(-3.0), Anchor::start(5.0)),
        (Anchor::start(5.0), Anchor::end(-3.0)),
        1.0,
        p.text_faint,
    );
    kit::line(
        ui,
        grip,
        (Anchor::end(-3.0), Anchor::start(9.0)),
        (Anchor::start(9.0), Anchor::end(-3.0)),
        1.0,
        p.text_faint,
    );
    ui.end_overlay();
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::input::{Event, PointerButton};
    use crate::tests::Driver;
    use crate::theme::Theme;

    const KNOWN: [&str; 6] = ["outliner", "viewport.1", "viewport.2", "viewport.3", "properties", "console"];

    fn title(panel: &str) -> &str {
        match panel {
            "outliner" => "Outliner",
            "viewport.1" => "Viewport 1",
            "viewport.2" => "Viewport 2",
            "viewport.3" => "Viewport 3",
            "properties" => "Properties",
            "console" => "Console",
            "viewport.4" => "Viewport 4",
            _ => "?",
        }
    }

    /// The editor's default arrangement in miniature: outliner | (viewports / console) | properties.
    fn workspace() -> DockLayout {
        DockLayout::from_root(DockNode::split(
            SplitAxis::Row,
            0.22,
            DockNode::tabs(&["outliner"]),
            DockNode::split(
                SplitAxis::Row,
                0.7,
                DockNode::split(
                    SplitAxis::Column,
                    0.72,
                    DockNode::split(
                        SplitAxis::Row,
                        0.5,
                        DockNode::tabs(&["viewport.1", "viewport.3"]),
                        DockNode::tabs(&["viewport.2"]),
                    ),
                    DockNode::tabs(&["console"]),
                ),
                DockNode::tabs(&["properties"]),
            ),
        ))
    }

    fn panel_content(ui: &mut Ui, panel: &str) {
        let theme = *ui.theme();
        let body = ui.begin(Style { grow: 1.0, padding: Insets::all(10.0), gap: 6.0, ..Style::column() });
        if panel.starts_with("viewport") {
            kit::fill(ui, body, Area::inset(4.0), crate::color::Color::MID_GREY, theme.metrics.radius_sm);
        }
        crate::widgets::label(ui, title(panel));
        ui.end();
    }

    struct Model {
        layout: DockLayout,
        changes: u32,
    }

    fn build(ui: &mut Ui, model: &mut Model) {
        let out = dock_area(ui, "Workspace", &mut model.layout, &title, &mut panel_content);
        if out.changed {
            model.changes += 1;
        }
    }

    fn drag(d: &mut Driver, b: &mut dyn FnMut(&mut Ui), from: Point, to: Point) {
        d.frame(vec![Event::PointerMoved(from)], b);
        d.frame(vec![Event::Button { button: PointerButton::Primary, pressed: true, pos: from }], b);
        d.frame(vec![Event::PointerMoved(Point::new(f32::midpoint(from.x, to.x), f32::midpoint(from.y, to.y)))], b);
        d.frame(vec![Event::PointerMoved(to)], b);
        d.frame(vec![], b);
        d.frame(vec![Event::Button { button: PointerButton::Primary, pressed: false, pos: to }], b);
        d.frame(vec![], b);
    }

    #[test]
    fn ui_05_layout_operations_split_tab_float_and_collapse() {
        let mut l = DockLayout::new(&["a", "b", "c"]);
        l.dock("b", "a", DropZone::Right).expect("split right");
        assert!(matches!(l.root(), DockNode::Split { axis: SplitAxis::Row, .. }));
        assert_eq!(l.panels(), ["a", "c", "b"]);
        l.dock("c", "b", DropZone::Bottom).expect("split below b");
        l.dock("c", "a", DropZone::Center).expect("back as a tab of a");
        assert_eq!(l.active_of("a"), Some("c"), "a docked tab becomes active");
        assert!(matches!(l.root(), DockNode::Split { .. }));
        // Alone in its group: cannot dock beside itself; with company it can.
        assert!(l.dock("b", "b", DropZone::Left).is_err());
        assert!(l.dock("a", "c", DropZone::Left).is_ok());
        // Floating and back.
        l.float("b", Rect::new(10.0, 10.0, 50.0, f32::NAN)).expect("float");
        assert_eq!(l.floating().len(), 1);
        assert_eq!(l.floating()[0].rect.w, 160.0, "repaired size");
        assert!(l.dock_to_main("b"));
        assert!(l.floating().is_empty());
        // Removing collapses empty groups and their splits.
        assert!(l.remove("a"));
        assert!(l.remove("b"));
        assert_eq!(*l.root(), DockNode::Tabs { panels: vec!["c".into()], active: 0 });
        assert!(!l.remove("zzz"));
        l.ensure("d");
        l.ensure("d");
        assert_eq!(l.panels(), ["c", "d"]);
        // Duplicates in a constructed tree are dropped.
        let dup = DockLayout::from_root(DockNode::split(
            SplitAxis::Row,
            2.0,
            DockNode::tabs(&["x"]),
            DockNode::tabs(&["x", "y"]),
        ));
        assert_eq!(dup.panels(), ["x", "y"]);
        // Splitting deeper than the limit becomes a tab.
        let mut deep = DockLayout::new(&["p0"]);
        for i in 1..40 {
            let name = format!("p{i}");
            let prev = format!("p{}", i - 1);
            deep.ensure(&name);
            deep.dock(&name, &prev, DropZone::Right).expect("dock");
        }
        let json = deep.to_json().expect("write");
        let known: Vec<String> = (0..40).map(|i| format!("p{i}")).collect();
        let known: Vec<&str> = known.iter().map(String::as_str).collect();
        assert_eq!(DockLayout::from_json(&json, &known).expect("read"), deep, "the depth limit holds");
    }

    #[test]
    fn ui_05_layout_files_round_trip_and_reject_damage() {
        let mut l = workspace();
        l.float("console", Rect::new(80.0, 60.0, 320.0, 240.0)).expect("float");
        let text = l.to_json().expect("write");
        assert_eq!(DockLayout::from_json(&text, &KNOWN).expect("read"), l);
        // Panels the application no longer has are dropped (and their groups collapse).
        let fewer =
            DockLayout::from_json(&text, &["outliner", "viewport.1", "viewport.2", "properties"]).expect("read");
        assert_eq!(fewer.panels(), ["outliner", "viewport.1", "viewport.2", "properties"]);
        assert!(fewer.floating().is_empty());
        let bad = [
            r#"{"version": 2, "root": {"tabs": []}}"#,
            r#"{"version": 1, "root": {"tabs": ["a", "a"]}}"#,
            r#"{"version": 1, "root": {"split": "diagonal", "first": {"tabs": []}, "second": {"tabs": []}}}"#,
            r#"{"version": 1, "root": {"tabs": [], "extra": 1}}"#,
            r#"{"version": 1, "root": {"tabs": ["a"]}, "floating": [{"rect": [1, 2, 3], "tabs": {"tabs": []}}]}"#,
            r#"{"version": 1, "root": {"split": "row", "first": {"tabs": []}}}"#,
            "[1, 2",
        ];
        for b in bad {
            assert!(DockLayout::from_json(b, &["a"]).is_err(), "{b}");
        }
        // Over the limits: nesting and node count.
        let mut deep = String::from(r#"{"version": 1, "root": "#);
        for _ in 0..(MAX_DOCK_DEPTH + 2) {
            deep.push_str(r#"{"split": "row", "first": {"tabs": []}, "second": "#);
        }
        deep.push_str(r#"{"tabs": []}"#);
        for _ in 0..(MAX_DOCK_DEPTH + 2) {
            deep.push('}');
        }
        deep.push('}');
        assert!(DockLayout::from_json(&deep, &["a"]).is_err());
        let long = format!(r#"{{"version": 1, "root": {{"tabs": ["{}"]}}}}"#, "n".repeat(MAX_PANEL_NAME + 1));
        assert!(DockLayout::from_json(&long, &["a"]).is_err());
        // A bad ratio is repaired, not rejected.
        let odd = r#"{"version": 1, "root": {"split": "column", "ratio": 7, "first": {"tabs": ["a"]}, "second": {"tabs": ["b"]}}}"#;
        let l = DockLayout::from_json(odd, &["a", "b"]).expect("read");
        assert!(matches!(l.root(), DockNode::Split { ratio, .. } if *ratio == 0.95));
    }

    #[test]
    fn rob_01_layout_file_mutation_fuzz() {
        // [ROB-01] Mutated layout files never panic.
        let mut l = workspace();
        l.float("console", Rect::new(80.0, 60.0, 320.0, 240.0)).expect("float");
        let corpus = vec![
            l.to_json().expect("write").into_bytes(),
            workspace().to_json().expect("write").into_bytes(),
            br#"{"version": 1, "root": {"tabs": ["outliner"], "active": 3}}"#.to_vec(),
        ];
        humanart_core::fuzz::run("dock-layout", &corpus, 20_000, 0xD0C_0001, std::time::Duration::from_secs(60), |b| {
            if let Ok(t) = std::str::from_utf8(b)
                && let Ok(layout) = DockLayout::from_json(t, &KNOWN)
            {
                let _ = layout.to_json();
            }
        });
    }

    #[test]
    fn ui_05_drag_a_tab_to_split_and_dividers_resize() {
        let model = RefCell::new(Model { layout: workspace(), changes: 0 });
        let mut b = |ui: &mut Ui| build(ui, &mut model.borrow_mut());
        let mut d = Driver::new(&Theme::dark().with_reduced_motion(true), 1000.0, 640.0);
        d.frame(vec![], &mut b);
        d.frame(vec![], &mut b);
        // Three viewports are docked; two are shown.
        let r1 = panel_rect(&d.ui, "Workspace", "viewport.1").expect("viewport 1 shown");
        let r2 = panel_rect(&d.ui, "Workspace", "viewport.2").expect("viewport 2 shown");
        assert!(panel_rect(&d.ui, "Workspace", "viewport.3").is_none(), "a hidden tab");
        assert!(r2.x >= r1.right() && (r1.y - r2.y).abs() < 0.5);
        // Drag "Viewport 3" onto the bottom edge of viewport 2's group.
        let tab = d.rect_of("Viewport 3").center();
        let target = Point::new(r2.center().x, r2.bottom() - 10.0);
        drag(&mut d, &mut b, tab, target);
        let l = model.borrow().layout.clone();
        assert!(l.active_of("viewport.1") == Some("viewport.1"), "the source group keeps its other tab");
        let r3 = panel_rect(&d.ui, "Workspace", "viewport.3").expect("viewport 3 shown");
        let r2b = panel_rect(&d.ui, "Workspace", "viewport.2").expect("viewport 2 still shown");
        assert!(r3.y > r2b.y && (r3.x - r2b.x).abs() < 0.5, "below viewport 2: {r3:?} {r2b:?}");
        assert!(model.borrow().changes >= 1);
        // The first divider follows the pointer, within limits.
        let out = d.rect_of("Outliner");
        let div =
            d.ui.semantics().iter().find(|s| s.role == Role::Custom("splitter")).map(|s| s.rect).expect("a divider");
        let start = div.center();
        drag(&mut d, &mut b, start, Point::new(start.x + 60.0, start.y));
        let out2 = d.rect_of("Outliner");
        let _ = (out, out2);
        let ratio = match model.borrow().layout.root() {
            DockNode::Split { ratio, .. } => *ratio,
            DockNode::Tabs { .. } => 0.0,
        };
        assert!((ratio - (0.22 + 60.0 / (1000.0 - DIVIDER))).abs() < 0.01, "{ratio}");
        drag(&mut d, &mut b, Point::new(start.x + 60.0, start.y), Point::new(-500.0, start.y));
        let ratio = match model.borrow().layout.root() {
            DockNode::Split { ratio, .. } => *ratio,
            DockNode::Tabs { .. } => 0.0,
        };
        assert!((ratio - MIN_PANE / (1000.0 - DIVIDER)).abs() < 0.002, "a pane keeps its minimum: {ratio}");
    }

    #[test]
    fn ui_09_tabs_and_docking_from_the_keyboard_and_floating_panels() {
        let model = RefCell::new(Model { layout: workspace(), changes: 0 });
        let mut b = |ui: &mut Ui| build(ui, &mut model.borrow_mut());
        let mut d = Driver::new(&Theme::dark().with_reduced_motion(true), 1000.0, 640.0);
        d.frame(vec![], &mut b);
        d.frame(vec![], &mut b);
        for _ in 0..30 {
            if d.focused_label() == "Viewport 1" {
                break;
            }
            d.key(Key::Tab, Modifiers::NONE, &mut b);
        }
        assert_eq!(d.focused_label(), "Viewport 1", "the tab bar is a Tab stop labelled by its shown panel");
        // F6 walks the panels: viewport 2, console, properties, outliner, back to viewport 1.
        let mut walk = Vec::new();
        for _ in 0..5 {
            d.key(Key::F(6), Modifiers::NONE, &mut b);
            walk.push(d.focused_label());
        }
        assert_eq!(walk, ["Viewport 2", "Console", "Properties", "Outliner", "Viewport 1"]);
        d.key(Key::F(6), Modifiers::SHIFT, &mut b);
        assert_eq!(d.focused_label(), "Outliner");
        d.key(Key::F(6), Modifiers::NONE, &mut b);
        d.frame(vec![Event::Modifiers(Modifiers::NONE)], &mut b);
        d.key(Key::Right, Modifiers::NONE, &mut b);
        assert_eq!(model.borrow().layout.active_of("viewport.1"), Some("viewport.3"));
        // Shift+F10, Down to "Split right", Enter: viewport 3 gets its own group.
        d.key(Key::F(10), Modifiers::SHIFT, &mut b);
        d.key(Key::Down, Modifiers::NONE, &mut b);
        d.key(Key::Enter, Modifiers::NONE, &mut b);
        d.frame(vec![], &mut b);
        assert!(
            panel_rect(&d.ui, "Workspace", "viewport.1").is_some()
                && panel_rect(&d.ui, "Workspace", "viewport.3").is_some()
        );
        // Float the console from its menu (secondary click on its tab), move and resize it.
        let tab = d.rect_of("Console").center();
        d.frame(vec![Event::PointerMoved(tab)], &mut b);
        d.frame(vec![Event::Button { button: PointerButton::Secondary, pressed: true, pos: tab }], &mut b);
        d.frame(vec![Event::Button { button: PointerButton::Secondary, pressed: false, pos: tab }], &mut b);
        d.click(d.rect_of("Float").center(), &mut b);
        d.frame(vec![], &mut b);
        assert_eq!(model.borrow().layout.floating().len(), 1);
        let before = model.borrow().layout.floating()[0].rect;
        let bar = Point::new(before.right() - 30.0, before.y + 12.0);
        drag(&mut d, &mut b, bar, Point::new(bar.x - 100.0, bar.y - 150.0));
        let moved = model.borrow().layout.floating()[0].rect;
        assert!(
            (moved.x - (before.x - 100.0)).abs() < 1.0 && (moved.y - (before.y - 150.0)).abs() < 1.0,
            "{before:?} -> {moved:?}"
        );
        let grip = Point::new(moved.right() - 5.0, moved.bottom() - 5.0);
        drag(&mut d, &mut b, grip, Point::new(grip.x + 40.0, grip.y + 30.0));
        let sized = model.borrow().layout.floating()[0].rect;
        assert!(
            (sized.w - moved.w - 40.0).abs() < 1.0 && (sized.h - moved.h - 30.0).abs() < 1.0,
            "{moved:?} -> {sized:?}"
        );
        let in_float = panel_rect(&d.ui, "Workspace", "console").expect("console shown");
        assert!(sized.contains(in_float.center()));
        // Back into the dock from its menu.
        let tab = d.rect_of("Console").center();
        d.frame(vec![Event::PointerMoved(tab)], &mut b);
        d.frame(vec![Event::Button { button: PointerButton::Secondary, pressed: true, pos: tab }], &mut b);
        d.frame(vec![Event::Button { button: PointerButton::Secondary, pressed: false, pos: tab }], &mut b);
        d.click(d.rect_of("Dock").center(), &mut b);
        assert!(model.borrow().layout.floating().is_empty());
        assert!(model.borrow().layout.contains("console"));
    }

    #[test]
    fn perf_r1_docking_area_does_not_allocate_in_steady_frames() {
        let model = RefCell::new(Model { layout: workspace(), changes: 0 });
        model.borrow_mut().layout.float("console", Rect::new(500.0, 300.0, 300.0, 200.0)).expect("float");
        let mut b = |ui: &mut Ui| build(ui, &mut model.borrow_mut());
        let mut d = Driver::new(&Theme::dark(), 1000.0, 640.0);
        for _ in 0..4 {
            d.frame(vec![], &mut b);
        }
        let stats = humanart_core::alloc::count_in(|| {
            for _ in 0..5 {
                d.frame(Vec::new(), &mut b);
            }
        });
        assert_eq!(stats.allocations + stats.reallocations, 0, "{stats:?}");
    }

    #[test]
    fn ac_wp12_2_docking_golden_at_100_150_200_percent() {
        // [UI-05] [AC-WP12-2] Outliner | three viewports (two shown) over the console |
        // properties, a floating panel, and a tab dragged over a group (drop highlight).
        for (theme, scale, name) in [
            (Theme::dark(), 1.0, "dock_100"),
            (Theme::dark(), 1.5, "dock_150"),
            (Theme::dark(), 2.0, "dock_200"),
            (Theme::light(), 1.0, "dock_light"),
        ] {
            let mut layout = workspace();
            layout.ensure("viewport.4");
            layout.float("viewport.4", Rect::new(560.0, 330.0, 260.0, 180.0)).expect("float");
            let model = RefCell::new(Model { layout, changes: 0 });
            let mut b = |ui: &mut Ui| build(ui, &mut model.borrow_mut());
            let mut d = Driver::new(&theme.with_reduced_motion(true), 1000.0, 640.0);
            d.scale = scale;
            d.frame(vec![], &mut b);
            d.frame(vec![], &mut b);
            let tab = d.rect_of("Properties").center();
            let over = panel_rect(&d.ui, "Workspace", "viewport.1")
                .map_or(Point::ZERO, |r| Point::new(r.x + 12.0, r.center().y));
            d.frame(vec![Event::PointerMoved(tab)], &mut b);
            d.frame(vec![Event::Button { button: PointerButton::Primary, pressed: true, pos: tab }], &mut b);
            d.frame(vec![Event::PointerMoved(Point::new(tab.x - 40.0, tab.y + 40.0))], &mut b);
            d.frame(vec![Event::PointerMoved(over)], &mut b);
            d.frame(vec![], &mut b);
            d.golden(name);
        }
    }
}
