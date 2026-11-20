//! Small controls `[UI-07]`: icon button (always with a tooltip `[UI-09]`), toggle switch,
//! checkbox, radio group, progress bar, collapsible section, search box, tabs, splitter, toolbar
//! and status bar. Each is operable from the keyboard and describes itself semantically; looks
//! follow the reference application (`icon_button`, `toggle`, `dropdown_frame`, section 16).

use crate::color::Color;
use crate::geom::Insets;
use crate::input::{Key, Modifiers, Shortcut};
use crate::kit;
use crate::layout::{Align as BoxAlign, Dim, NodeId, Style};
use crate::paint::Axis;
use crate::popup::{Tip, tooltip};
use crate::text::Align;
use crate::ui::{Anchor, Area, CursorIcon, Id, Response, Role, SemanticValue, Sense, Ui, ease};
use crate::widgets::{FieldResponse, TextField};

/// A square ghost button showing an icon `[UI-07]`. Icon-only controls always carry a tooltip
/// `[UI-09]`: `tip` is its tooltip, its semantic label and its id.
#[derive(Clone, Copy, Debug)]
pub struct IconButton<'a> {
    icon: char,
    tip: &'a str,
    shortcut: Option<Shortcut>,
    selected: bool,
    enabled: bool,
    size: Option<f32>,
}

impl<'a> IconButton<'a> {
    /// A button with `icon` and the tooltip (and label) `tip`.
    pub fn new(icon: char, tip: &'a str) -> Self {
        Self { icon, tip, shortcut: None, selected: false, enabled: true, size: None }
    }

    /// The shortcut shown in the tooltip.
    #[must_use]
    pub fn shortcut(mut self, shortcut: Shortcut) -> Self {
        self.shortcut = Some(shortcut);
        self
    }

    /// Drawn pressed in (a tool that is active).
    #[must_use]
    pub fn selected(mut self, selected: bool) -> Self {
        self.selected = selected;
        self
    }

    /// Whether it can be used.
    #[must_use]
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// The side of the square (default the theme's `icon_button`).
    #[must_use]
    pub fn size(mut self, size: f32) -> Self {
        self.size = Some(size);
        self
    }

    /// Adds the button; `clicked` on a click or Enter / Space while focused.
    pub fn show(self, ui: &mut Ui) -> Response {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let side = self.size.unwrap_or(m.icon_button);
        let node = ui.node(Style { width: Dim::Px(side), height: Dim::Px(side), ..Style::default() });
        let id = ui.id(self.tip);
        let mut r = ui.interact(id, node, if self.enabled { Sense::CLICK } else { Sense::HOVER });
        if !self.enabled {
            r.clicked = false;
        }
        let h = if self.enabled { r.hover_t } else { 0.0 };
        let fill =
            if self.selected { p.active.mix(p.hover, h * 0.5) } else { p.hover.fade(h).mix(p.active, r.press_t) };
        kit::fill(ui, node, Area::FULL, fill, m.radius_sm);
        let color = if !self.enabled {
            p.text_faint.fade(0.6)
        } else if self.selected {
            p.text
        } else {
            p.text_muted.mix(p.text, h)
        };
        let size = if side < m.icon_button { m.icon_sm } else { m.icon };
        kit::icon_centered(ui, node, self.icon, size, color);
        if r.focus_visible {
            kit::ring(ui, node, Area::FULL, m.radius_sm);
        }
        ui.semantic(id, Role::Button, self.tip, SemanticValue::Toggle(self.selected));
        if !self.enabled {
            ui.semantic_disabled();
        }
        let mut tip = Tip::new(self.tip);
        if let Some(s) = self.shortcut {
            tip = tip.shortcut(s);
        }
        tooltip(ui, &r, &tip);
        r
    }
}

/// An icon button with a tooltip.
pub fn icon_button(ui: &mut Ui, icon: char, tip: &str) -> Response {
    IconButton::new(icon, tip).show(ui)
}

/// A switch with a label after it `[UI-07]` (38 × 22, reference `toggle`). Click, Enter or Space
/// flips it.
pub fn toggle(ui: &mut Ui, label: &str, on: &mut bool) -> Response {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let text_w = if label.is_empty() { 0.0 } else { ui.measure_text(label, theme.text.body).w + m.space_4 };
    let node =
        ui.node(Style { width: Dim::Px(m.switch_w + text_w), height: Dim::Px(m.control_height), ..Style::default() });
    let id = ui.id(label);
    let mut r = ui.interact(id, node, Sense::CLICK);
    if r.clicked {
        *on = !*on;
        r.changed = true;
    }
    let slide = {
        let dt = ui.dt();
        let reduced = theme.motion.reduced;
        let s = ui.memory::<f32>(id);
        *s = ease(*s, if *on { 1.0 } else { 0.0 }, dt, theme.motion.ease_rate, reduced);
        *s
    };
    if slide > 0.0 && slide < 1.0 {
        ui.request_repaint();
    }
    let y0 = (m.control_height - m.switch_h) * 0.5;
    let track = kit::at(0.0, y0, m.switch_w, m.switch_h);
    let base = p.border_strong.mix(p.hover, r.hover_t * 0.5);
    kit::fill(ui, node, track, base.mix(p.accent, slide), m.switch_h * 0.5);
    let inset = (m.switch_h - m.switch_knob) * 0.5;
    let travel = m.switch_w - m.switch_h;
    let knob = kit::at(inset + travel * slide, y0 + inset, m.switch_knob, m.switch_knob);
    kit::fill(ui, node, knob, p.text_muted.mix(p.on_accent, slide), m.switch_knob * 0.5);
    kit::text(ui, node, kit::inset_x(m.switch_w + m.space_4, 0.0), label, theme.text.body, p.text, Align::Start);
    if r.focus_visible {
        kit::ring(ui, node, track, m.switch_h * 0.5);
    }
    ui.semantic(id, Role::Toggle, label, SemanticValue::Toggle(*on));
    r
}

/// A checkbox with a label `[UI-07]`. Click, Enter or Space flips it.
pub fn checkbox(ui: &mut Ui, label: &str, checked: &mut bool) -> Response {
    checkbox_impl(ui, label, checked, true)
}

/// A checkbox whose label is only its semantic name (a property grid shows the name beside it).
pub(crate) fn checkbox_unlabelled(ui: &mut Ui, label: &str, checked: &mut bool) -> Response {
    checkbox_impl(ui, label, checked, false)
}

fn checkbox_impl(ui: &mut Ui, label: &str, checked: &mut bool, show_label: bool) -> Response {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let side = 16.0;
    let text_w =
        if label.is_empty() || !show_label { 0.0 } else { ui.measure_text(label, theme.text.body).w + m.space_4 };
    let node = ui.node(Style { width: Dim::Px(side + text_w), height: Dim::Px(m.control_height), ..Style::default() });
    let id = ui.id(label);
    let mut r = ui.interact(id, node, Sense::CLICK);
    if r.clicked {
        *checked = !*checked;
        r.changed = true;
    }
    let y0 = (m.control_height - side) * 0.5;
    let boxa = kit::at(0.0, y0, side, side);
    if *checked {
        kit::fill(ui, node, boxa, p.accent.fade(1.0 - 0.12 * r.hover_t), 4.0);
        kit::icon(ui, node, Anchor::start(side * 0.5), Anchor::at(0.5, 0.0), crate::icons::CHECK, 12.0, p.on_accent);
    } else {
        kit::frame(
            ui,
            node,
            boxa,
            p.surface.mix(p.hover, r.hover_t),
            4.0,
            p.border_strong.mix(p.border_focus, r.hover_t),
        );
    }
    if show_label {
        kit::text(ui, node, kit::inset_x(side + m.space_4, 0.0), label, theme.text.body, p.text, Align::Start);
    }
    if r.focus_visible {
        kit::ring(ui, node, boxa, 4.0);
    }
    ui.semantic(id, Role::Custom("checkbox"), label, SemanticValue::Toggle(*checked));
    r
}

/// A group of radio buttons choosing one of `options` `[UI-07]`. The group is one stop in the Tab
/// order (the selected option); the arrow keys move the choice, as in native radio groups.
pub fn radio_group(ui: &mut Ui, label: &str, selected: &mut usize, options: &[&str]) -> Response {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let group = ui.id(label);
    ui.begin(Style { gap: m.space_1, ..Style::column() });
    ui.begin_semantic_group(group);
    ui.semantic(group, Role::Group, label, SemanticValue::None);
    ui.push_id(group);
    let mut out = Response { id: group, ..Response::default() };
    let before = *selected;
    let mut moved_to = None;
    let side = 16.0;
    for (i, option) in options.iter().enumerate() {
        let text_w = ui.measure_text(option, theme.text.body).w;
        let node = ui.node(Style {
            width: Dim::Px(side + m.space_4 + text_w),
            height: Dim::Px(m.control_height_sm),
            ..Style::default()
        });
        let id = Id::ROOT.with(label).with_index(i as u64);
        let is_sel = i == *selected;
        let sense = if is_sel { Sense::CLICK } else { Sense { click: true, drag: false, focus: false } };
        let r = ui.interact(id, node, sense);
        if r.clicked {
            *selected = i;
            if !is_sel {
                moved_to = Some(id);
            }
        }
        if r.focused {
            let n = options.len();
            if ui.consume_key(Key::Down, Modifiers::NONE) || ui.consume_key(Key::Right, Modifiers::NONE) {
                *selected = (i + 1) % n;
            } else if ui.consume_key(Key::Up, Modifiers::NONE) || ui.consume_key(Key::Left, Modifiers::NONE) {
                *selected = (i + n - 1) % n;
            }
            if *selected != i {
                moved_to = Some(Id::ROOT.with(label).with_index(*selected as u64));
            }
            out.focused = true;
        }
        let y0 = (m.control_height_sm - side) * 0.5;
        let dot = kit::at(0.0, y0, side, side);
        let on = i == *selected;
        kit::frame(
            ui,
            node,
            dot,
            p.surface.mix(p.hover, r.hover_t),
            side * 0.5,
            if on { p.accent } else { p.border_strong.mix(p.border_focus, r.hover_t) },
        );
        if on {
            kit::fill(ui, node, kit::at(4.0, y0 + 4.0, side - 8.0, side - 8.0), p.accent, (side - 8.0) * 0.5);
        }
        kit::text(ui, node, kit::inset_x(side + m.space_4, 0.0), option, theme.text.body, p.text, Align::Start);
        if r.focus_visible {
            kit::ring(ui, node, dot, side * 0.5);
        }
        ui.semantic(id, Role::Custom("radio"), option, SemanticValue::Toggle(on));
    }
    ui.pop_id();
    ui.end_semantic_group();
    ui.end();
    if let Some(id) = moved_to {
        ui.request_focus(id);
    }
    out.changed = *selected != before;
    out
}

/// A progress bar `[UI-07]`: `Some(fraction)` (0 to 1) or `None` for an indeterminate activity
/// (a sliding segment; a still centred segment under reduced motion).
pub fn progress_bar(ui: &mut Ui, label: &str, fraction: Option<f32>) -> NodeId {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let node = ui.node(Style { height: Dim::Px(6.0), min_width: 40.0, grow: kit::grow_x(ui), ..Style::default() });
    kit::fill(ui, node, Area::FULL, p.active, 3.0);
    let (a, b) = match fraction {
        Some(f) => (0.0, if f.is_finite() { f.clamp(0.0, 1.0) } else { 0.0 }),
        None if theme.motion.reduced => (0.35, 0.65),
        None => {
            ui.request_repaint();
            #[allow(clippy::cast_possible_truncation)] // A phase in 0..1.
            let t = (ui.time() * 0.8).fract() as f32;
            ((t * 1.3 - 0.3).max(0.0), (t * 1.3).min(1.0))
        }
    };
    if b > a {
        let area =
            Area { x0: Anchor::at(a, 0.0), y0: Anchor::start(0.0), x1: Anchor::at(b, 0.0), y1: Anchor::end(0.0) };
        kit::fill(ui, node, area, p.accent, 3.0);
    }
    let _ = m;
    let id = ui.id(label);
    let value = fraction.map_or(SemanticValue::None, |f| SemanticValue::Number {
        value: f64::from(f.clamp(0.0, 1.0)),
        min: 0.0,
        max: 1.0,
    });
    ui.semantic(id, Role::Custom("progress"), label, value);
    ui.semantic_node(node);
    node
}

/// The open state of a [`Collapsible`], in [`Ui::memory`].
#[derive(Clone, Copy, Debug, Default)]
struct CollapseState {
    known: bool,
    open: bool,
}

/// A collapsible section `[UI-07]`: a header row with a chevron that shows or hides its content.
/// Click, Enter or Space toggles; Right opens and Left closes.
#[derive(Clone, Copy, Debug)]
pub struct Collapsible<'a> {
    title: &'a str,
    default_open: bool,
}

impl<'a> Collapsible<'a> {
    /// A section titled (and identified by) `title`, closed at first.
    pub fn new(title: &'a str) -> Self {
        Self { title, default_open: false }
    }

    /// Whether it starts open.
    #[must_use]
    pub fn default_open(mut self, open: bool) -> Self {
        self.default_open = open;
        self
    }

    /// Adds the header and, when open, the content built by `content`. Returns the header's
    /// response (`changed` when it opened or closed) and whether it is open.
    pub fn show(self, ui: &mut Ui, content: impl FnOnce(&mut Ui)) -> (Response, bool) {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let id = ui.id(self.title);
        let mut st = *ui.memory::<CollapseState>(id);
        if !st.known {
            st = CollapseState { known: true, open: self.default_open };
        }
        let node = ui.node(Style { height: Dim::Px(m.row_height), ..Style::default() });
        let mut r = ui.interact(id, node, Sense::CLICK);
        if r.clicked {
            st.open = !st.open;
            r.changed = true;
        }
        if r.focused {
            if ui.consume_key(Key::Right, Modifiers::NONE) && !st.open {
                st.open = true;
                r.changed = true;
            }
            if ui.consume_key(Key::Left, Modifiers::NONE) && st.open {
                st.open = false;
                r.changed = true;
            }
        }
        *ui.memory::<CollapseState>(id) = st;
        kit::fill(ui, node, Area::FULL, p.hover.fade(r.hover_t), m.radius_sm);
        let chevron = if st.open { crate::icons::CHEVRON_DOWN } else { crate::icons::CHEVRON_RIGHT };
        kit::icon(ui, node, Anchor::start(m.space_6), Anchor::at(0.5, 0.0), chevron, m.icon_sm, p.text_muted);
        kit::text(ui, node, kit::inset_x(m.space_8, 0.0), self.title, theme.text.label, p.text, Align::Start);
        if r.focus_visible {
            kit::ring(ui, node, Area::FULL, m.radius_sm);
        }
        ui.semantic(id, Role::Custom("section"), self.title, SemanticValue::Toggle(st.open));
        if st.open {
            ui.begin(Style {
                padding: Insets { left: m.space_6, top: m.space_2, right: 0.0, bottom: m.space_4 },
                gap: m.space_3,
                ..Style::column()
            });
            ui.push_id(id);
            content(ui);
            ui.pop_id();
            ui.end();
        }
        (r, st.open)
    }
}

/// A search box `[UI-07]`: a text field with a search icon and a clear button; Escape clears the
/// text first and leaves the field on a second press.
pub fn search_box(ui: &mut Ui, id_label: &str, text: &mut String, placeholder: &str) -> FieldResponse {
    let theme = *ui.theme();
    let m = theme.metrics;
    ui.begin(Style { grow: kit::grow_x(ui), align_items: BoxAlign::Center, ..Style::row() });
    let field_id = ui.id(id_label);
    if ui.focused() == Some(field_id) && !text.is_empty() && ui.consume_key(Key::Escape, Modifiers::NONE) {
        text.clear();
    }
    let mut out = TextField::new(id_label, text).placeholder(placeholder).icon(crate::icons::SEARCH).show(ui);
    if !text.is_empty() {
        ui.push_id(field_id);
        let row = ui.begin(Style {
            position: crate::layout::Position::Absolute { left: None, top: Some(2.0), right: Some(2.0), bottom: None },
            ..Style::row()
        });
        let _ = row;
        if IconButton::new(crate::icons::X, "Clear search").size(m.control_height - 4.0).show(ui).clicked {
            text.clear();
            out.response.changed = true;
            ui.request_focus(field_id);
        }
        ui.end();
        ui.pop_id();
    }
    ui.end();
    out
}

/// A row of tabs choosing one of `labels` `[UI-07]` (underline style). The bar is one Tab stop;
/// Left / Right (and Home / End) move the selection.
pub fn tabs(ui: &mut Ui, id_label: &str, selected: &mut usize, labels: &[&str]) -> Response {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let id = ui.id(id_label);
    let bar = ui.begin(Style { height: Dim::Px(m.row_height + 2.0), gap: m.space_1, ..Style::row() });
    kit::line(
        ui,
        bar,
        (Anchor::start(0.0), Anchor::end(-0.5)),
        (Anchor::end(0.0), Anchor::end(-0.5)),
        m.border,
        p.border,
    );
    let mut r = ui.interact(id, bar, Sense { click: false, drag: false, focus: true });
    let before = *selected;
    let n = labels.len();
    if r.focused && n > 0 {
        if ui.consume_key(Key::Right, Modifiers::NONE) {
            *selected = (*selected + 1) % n;
        }
        if ui.consume_key(Key::Left, Modifiers::NONE) {
            *selected = (*selected + n - 1) % n;
        }
        if ui.consume_key(Key::Home, Modifiers::NONE) {
            *selected = 0;
        }
        if ui.consume_key(Key::End, Modifiers::NONE) {
            *selected = n - 1;
        }
    }
    ui.push_id(id);
    for (i, label) in labels.iter().enumerate() {
        let w = ui.measure_text(label, theme.text.label).w + 2.0 * m.space_5;
        let tab = ui.node(Style { width: Dim::Px(w), ..Style::default() });
        let tid = Id::ROOT.with(id_label).with_index(i as u64);
        let tr = ui.interact(tid, tab, Sense { click: true, drag: false, focus: false });
        if tr.clicked {
            *selected = i;
            ui.request_focus(id);
        }
        let on = i == *selected;
        kit::fill(ui, tab, kit::at(0.0, 3.0, w, m.row_height - 4.0), p.hover.fade(tr.hover_t * 0.8), m.radius_sm);
        let ink = if on { p.text } else { p.text_muted.mix(p.text, tr.hover_t) };
        kit::text(ui, tab, Area::FULL, label, theme.text.label, ink, Align::Center);
        if on {
            let under = Area {
                x0: Anchor::start(m.space_4),
                y0: Anchor::end(-2.0),
                x1: Anchor::end(-m.space_4),
                y1: Anchor::end(0.0),
            };
            kit::fill(ui, tab, under, p.accent, 1.0);
            if r.focus_visible {
                kit::ring(ui, tab, Area::inset(2.0), m.radius_sm);
            }
        }
        ui.semantic(tid, Role::Custom("tab"), label, SemanticValue::Toggle(on));
    }
    ui.pop_id();
    ui.end();
    r.changed = *selected != before;
    ui.semantic(id, Role::Custom("tablist"), id_label, SemanticValue::None);
    r
}

/// A draggable divider between two panes `[UI-07]` that sets `size` (the first pane's size in
/// logical pixels along `axis`) within `min..=max`. Keyboard: arrows move it 8 px (Shift: 1 px),
/// Home / End go to the limits.
pub fn splitter(ui: &mut Ui, id_label: &str, axis: Axis, size: &mut f32, min: f32, max: f32) -> Response {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let (min, max) = if min <= max { (min, max) } else { (max, min) };
    let thick = 6.0;
    let style = match axis {
        Axis::Horizontal => Style { width: Dim::Px(thick), ..Style::default() },
        Axis::Vertical => Style { height: Dim::Px(thick), ..Style::default() },
    };
    let node = ui.node(Style { shrink: 0.0, ..style });
    let id = ui.id(id_label);
    let mut r = ui.interact(id, node, Sense::DRAG);
    let before = *size;
    if r.dragged {
        *size += match axis {
            Axis::Horizontal => r.drag_delta.x,
            Axis::Vertical => r.drag_delta.y,
        };
    }
    if r.focused {
        let (less, more) = match axis {
            Axis::Horizontal => (Key::Left, Key::Right),
            Axis::Vertical => (Key::Up, Key::Down),
        };
        while let Some(mods) = ui.consume_key_any(less) {
            *size -= if mods.shift { 1.0 } else { 8.0 };
        }
        while let Some(mods) = ui.consume_key_any(more) {
            *size += if mods.shift { 1.0 } else { 8.0 };
        }
        if ui.consume_key(Key::Home, Modifiers::NONE) {
            *size = min;
        }
        if ui.consume_key(Key::End, Modifiers::NONE) {
            *size = max;
        }
    }
    *size = if size.is_finite() { size.clamp(min, max) } else { min };
    r.changed = (*size - before).abs() > 0.0;
    if r.hovered || r.dragged {
        ui.set_cursor(match axis {
            Axis::Horizontal => CursorIcon::ResizeHorizontal,
            Axis::Vertical => CursorIcon::ResizeVertical,
        });
    }
    let t = r.hover_t.max(r.press_t);
    let color = p.border.mix(p.focus_ring, t);
    let width = m.border + t;
    match axis {
        Axis::Horizontal => {
            kit::line(
                ui,
                node,
                (Anchor::at(0.5, 0.0), Anchor::start(0.0)),
                (Anchor::at(0.5, 0.0), Anchor::end(0.0)),
                width,
                color,
            );
        }
        Axis::Vertical => {
            kit::line(
                ui,
                node,
                (Anchor::start(0.0), Anchor::at(0.5, 0.0)),
                (Anchor::end(0.0), Anchor::at(0.5, 0.0)),
                width,
                color,
            );
        }
    }
    if r.focus_visible {
        kit::ring(ui, node, Area::FULL, 2.0);
    }
    ui.semantic(
        id,
        Role::Custom("splitter"),
        id_label,
        SemanticValue::Number { value: f64::from(*size), min: f64::from(min), max: f64::from(max) },
    );
    r
}

/// Opens a toolbar `[UI-07]`: a row of compact controls on the panel colour with a hairline under
/// it. Add icon buttons, separators ([`toolbar_separator`]) and other controls, then [`Ui::end`].
pub fn begin_toolbar(ui: &mut Ui) -> NodeId {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let node = ui.begin(Style {
        height: Dim::Px(m.icon_button + 2.0 * m.space_2 + m.border),
        padding: Insets { left: m.space_3, top: m.space_2, right: m.space_3, bottom: m.space_2 + m.border },
        gap: m.space_1,
        align_items: BoxAlign::Center,
        shrink: 0.0,
        ..Style::row()
    });
    kit::fill(ui, node, Area::FULL, p.panel, 0.0);
    kit::line(
        ui,
        node,
        (Anchor::start(0.0), Anchor::end(-0.5)),
        (Anchor::end(0.0), Anchor::end(-0.5)),
        m.border,
        p.border,
    );
    ui.semantic(ui.id("toolbar"), Role::Custom("toolbar"), "", SemanticValue::None);
    node
}

/// A vertical hairline between groups of toolbar controls.
pub fn toolbar_separator(ui: &mut Ui) {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let node = ui.node(Style { width: Dim::Px(m.space_5), height: Dim::Px(m.icon_button - 8.0), ..Style::default() });
    kit::line(
        ui,
        node,
        (Anchor::at(0.5, 0.0), Anchor::start(0.0)),
        (Anchor::at(0.5, 0.0), Anchor::end(0.0)),
        m.border,
        p.border,
    );
}

/// Opens a status bar `[UI-07]`: a short row of small text at the bottom of the window. Add
/// [`status_item`]s (and a growing spacer to push items right), then [`Ui::end`].
pub fn begin_status_bar(ui: &mut Ui) -> NodeId {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let node = ui.begin(Style {
        height: Dim::Px(m.control_height_sm),
        padding: Insets::xy(m.space_5, 0.0),
        gap: m.space_6,
        align_items: BoxAlign::Center,
        shrink: 0.0,
        ..Style::row()
    });
    kit::fill(ui, node, Area::FULL, p.panel, 0.0);
    kit::line(
        ui,
        node,
        (Anchor::start(0.0), Anchor::start(0.5)),
        (Anchor::end(0.0), Anchor::start(0.5)),
        m.border,
        p.border,
    );
    ui.semantic(ui.id("status"), Role::Custom("statusbar"), "", SemanticValue::None);
    node
}

/// One piece of status text, with an optional icon and colour.
pub fn status_item(ui: &mut Ui, icon: Option<char>, text: &str, color: Option<Color>) -> NodeId {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let style = theme.text.tiny;
    let icon_w = if icon.is_some() { m.icon_sm + m.space_2 } else { 0.0 };
    let w = ui.measure_text(text, style).w + icon_w;
    let node = ui.node(Style { width: Dim::Px(w), height: Dim::Px(m.control_height_sm), ..Style::default() });
    let ink = color.unwrap_or(p.text_muted);
    if let Some(i) = icon {
        kit::icon(ui, node, Anchor::start(m.icon_sm * 0.5), Anchor::at(0.5, 0.0), i, m.icon_sm, ink);
    }
    kit::text(ui, node, kit::inset_x(icon_w, 0.0), text, style, ink, Align::Start);
    let id = ui.id(text);
    ui.semantic(id, Role::Label, text, SemanticValue::None);
    ui.semantic_node(node);
    node
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::geom::Point;
    use crate::input::{Event, PointerButton};
    use crate::tests::Driver;
    use crate::theme::Theme;

    /// The state the controls gallery edits.
    struct Model {
        snap: bool,
        mirror: bool,
        quality: usize,
        tab: usize,
        split: f32,
        query: String,
        icon_clicks: u32,
    }

    impl Model {
        fn new() -> Self {
            Self { snap: true, mirror: false, quality: 0, tab: 0, split: 120.0, query: String::new(), icon_clicks: 0 }
        }
    }

    /// Every control of this module in a 360 × 460 window.
    fn gallery(ui: &mut Ui, model: &mut Model) {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        ui.begin(Style { width: Dim::Percent(100.0), height: Dim::Percent(100.0), ..Style::column() });
        begin_toolbar(ui);
        let play = IconButton::new(crate::icons::PLAY, "Play").shortcut(Shortcut::new(Key::Space, Modifiers::NONE));
        if play.show(ui).clicked {
            model.icon_clicks += 1;
        }
        IconButton::new(crate::icons::PIN, "Pin").selected(true).show(ui);
        toolbar_separator(ui);
        IconButton::new(crate::icons::TRASH, "Delete").enabled(false).show(ui);
        ui.end();
        let body = ui.begin(Style { padding: Insets::all(m.space_6), gap: m.space_4, grow: 1.0, ..Style::column() });
        kit::fill(ui, body, Area::FULL, p.panel, 0.0);
        toggle(ui, "Snap to grid", &mut model.snap);
        checkbox(ui, "Mirror X", &mut model.mirror);
        radio_group(ui, "Quality", &mut model.quality, &["Draft", "Final"]);
        tabs(ui, "Panels", &mut model.tab, &["Shape", "Skin", "Hair"]);
        ui.begin(Style { gap: m.space_4, align_items: BoxAlign::Center, height: Dim::Px(12.0), ..Style::row() });
        progress_bar(ui, "Baking", Some(0.6));
        progress_bar(ui, "Loading", None);
        ui.end();
        search_box(ui, "regions", &mut model.query, "Search regions");
        Collapsible::new("Advanced").default_open(true).show(ui, |ui| {
            crate::widgets::label(ui, "Inside the section");
        });
        ui.begin(Style { height: Dim::Px(40.0), ..Style::row() });
        let left = ui.node(Style { width: Dim::Px(model.split), ..Style::default() });
        kit::fill(ui, left, Area::FULL, p.surface, m.radius_sm);
        splitter(ui, "Divider", Axis::Horizontal, &mut model.split, 60.0, 200.0);
        let right = ui.node(Style { grow: 1.0, ..Style::default() });
        kit::fill(ui, right, Area::FULL, p.surface, m.radius_sm);
        ui.end();
        ui.end();
        begin_status_bar(ui);
        status_item(ui, Some(crate::icons::CHECK_CIRCLE), "Saved", Some(p.positive));
        ui.node(Style { grow: 1.0, ..Style::default() });
        status_item(ui, None, "60 fps", None);
        ui.end();
        ui.end();
    }

    /// Presses Tab until the widget labelled `label` has the focus.
    fn focus(d: &mut Driver, build: &mut dyn FnMut(&mut Ui), label: &str) {
        for _ in 0..40 {
            if d.focused_label() == label {
                return;
            }
            d.key(Key::Tab, Modifiers::NONE, build);
        }
        panic!("{label} never got the focus");
    }

    #[test]
    fn ui_07_controls_operate_by_pointer_and_keyboard() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = Driver::new(&Theme::dark(), 360.0, 460.0);
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        // Pointer: the switch flips, the icon button clicks, the disabled one is marked.
        d.click(d.rect_of("Snap to grid").center(), &mut build);
        assert!(!model.borrow().snap);
        d.click(d.rect_of("Play").center(), &mut build);
        assert_eq!(model.borrow().icon_clicks, 1);
        assert!(d.semantic("Delete").is_some_and(|s| s.disabled));
        // Keyboard: the checkbox with Space, the radio group with arrows (one Tab stop).
        focus(&mut d, &mut build, "Mirror X");
        d.key(Key::Space, Modifiers::NONE, &mut build);
        assert!(model.borrow().mirror);
        focus(&mut d, &mut build, "Draft");
        d.key(Key::Down, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().quality, 1);
        d.frame(vec![], &mut build);
        assert_eq!(d.focused_label(), "Final", "the focus follows the choice");
        d.key(Key::Tab, Modifiers::NONE, &mut build);
        assert_eq!(d.focused_label(), "Panels", "the group is a single Tab stop");
        d.key(Key::Right, Modifiers::NONE, &mut build);
        d.key(Key::End, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().tab, 2);
        // The search box: Escape clears first and leaves the focus where it is.
        focus(&mut d, &mut build, "Search regions");
        d.frame("lip".chars().map(Event::Text).collect(), &mut build);
        assert_eq!(model.borrow().query, "lip");
        d.key(Key::Escape, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().query, "");
        assert_eq!(d.focused_label(), "Search regions");
        // The collapsible section: Left closes, Enter opens.
        focus(&mut d, &mut build, "Advanced");
        assert!(d.semantic("Inside the section").is_some());
        d.key(Key::Left, Modifiers::NONE, &mut build);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Inside the section").is_none());
        d.key(Key::Enter, Modifiers::NONE, &mut build);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Inside the section").is_some());
        // The splitter: arrows move 8 px, Shift 1 px, End to the limit, dragging moves it.
        focus(&mut d, &mut build, "Divider");
        d.key(Key::Right, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().split, 128.0);
        d.key(Key::Left, Modifiers::SHIFT, &mut build);
        assert_eq!(model.borrow().split, 127.0);
        d.key(Key::End, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().split, 200.0);
        d.frame(vec![Event::Modifiers(Modifiers::NONE)], &mut build);
        d.frame(vec![], &mut build);
        let s = d.rect_of("Divider").center();
        let end = Point::new(s.x - 60.0, s.y);
        let drag = [
            vec![Event::PointerMoved(s), Event::Button { button: PointerButton::Primary, pressed: true, pos: s }],
            vec![Event::PointerMoved(Point::new(s.x - 30.0, s.y))],
            vec![Event::PointerMoved(end)],
            vec![Event::Button { button: PointerButton::Primary, pressed: false, pos: end }],
        ];
        for events in drag {
            d.frame(events, &mut build);
        }
        let split = model.borrow().split;
        assert!((split - 140.0).abs() < 8.5, "{split}");
    }

    #[test]
    fn ui_09_icon_buttons_show_their_tooltip_on_rest_and_on_focus() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = Driver::new(&Theme::dark(), 360.0, 460.0);
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        let tip_shown = |d: &Driver| d.ui.semantics().iter().any(|s| s.role == Role::Custom("tooltip"));
        // Every icon-only control in the gallery is labelled (its tooltip's title).
        for label in ["Play", "Pin", "Delete"] {
            assert!(d.semantic(label).is_some(), "{label}");
        }
        let play = d.rect_of("Play").center();
        d.frame(vec![Event::PointerMoved(play)], &mut build);
        assert!(!tip_shown(&d), "not before the delay");
        assert!(d.output.animating, "frames keep coming while the delay runs");
        for _ in 0..40 {
            d.frame(vec![], &mut build);
        }
        assert!(tip_shown(&d), "after resting {:.2} s", d.time);
        d.frame(vec![Event::PointerMoved(Point::new(200.0, 300.0))], &mut build);
        assert!(!tip_shown(&d));
        // Keyboard focus shows it at once.
        focus(&mut d, &mut build, "Play");
        assert!(tip_shown(&d));
    }

    #[test]
    fn ac_wp12_2_controls_golden_at_100_150_200_percent() {
        // [UI-07] [AC-WP12-2] Toolbar, switch, checkbox, radio, tabs, progress, search, section,
        // splitter and status bar; the keyboard focus on the checkbox (focus ring).
        for (scale, name) in [(1.0, "controls_100"), (1.5, "controls_150"), (2.0, "controls_200")] {
            let model = RefCell::new(Model::new());
            let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
            let mut d = Driver::new(&Theme::dark().with_reduced_motion(true), 360.0, 460.0);
            d.scale = scale;
            d.frame(vec![], &mut build);
            d.frame(vec![], &mut build);
            focus(&mut d, &mut build, "Mirror X");
            d.frame(vec![], &mut build);
            d.golden(name);
        }
        // [UI-09] The light and high-contrast themes, and the smallest UI scale (75 %).
        for (theme, scale, name) in [
            (Theme::light(), 1.0, "controls_light"),
            (Theme::high_contrast(), 1.0, "controls_high_contrast"),
            (Theme::dark(), 0.75, "controls_75"),
        ] {
            let model = RefCell::new(Model::new());
            let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
            let mut d = Driver::new(&theme.with_reduced_motion(true), 360.0, 460.0);
            d.scale = scale;
            d.frame(vec![], &mut build);
            d.frame(vec![], &mut build);
            focus(&mut d, &mut build, "Mirror X");
            d.frame(vec![], &mut build);
            d.golden(name);
        }
    }
}
