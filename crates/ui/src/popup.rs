//! Popups `[UI-07]`: tooltips with rich text, menus (menu buttons and context menus), dropdowns,
//! combo boxes with search, modal dialogs and toast notifications. They draw on
//! [`Layer::Popup`] / [`Layer::Tooltip`] through [`Ui::begin_overlay`], so they escape every clip
//! and lie above the panels; their looks follow the reference application (section 15: menus,
//! tooltips, modals, notices; section 16.4 shadows).
//!
//! **Keyboard** `[UI-09]`: an open list takes Up / Down / Home / End / Page Up / Page Down, Enter
//! or Space picks, Escape or Tab closes, a letter jumps to the next item starting with it (lists
//! without a search field). A modal dialog traps Tab inside itself and closes on Escape. Toasts'
//! close buttons are in the Tab order.

use crate::color::Color;
use crate::controls::IconButton;
use crate::geom::{Insets, Point, Rect};
use crate::input::{Key, Modifiers, Shortcut};
use crate::kit;
use crate::layout::{Dim, Style};
use crate::paint::Layer;
use crate::text::Align;
use crate::ui::{Anchor, Area, Id, Response, Role, SemanticValue, Sense, Ui, ease};
use crate::widgets::TextField;

/// The contents of a tooltip: a title (with an optional shortcut), wrapped text and name / value
/// rows: the "rich text" of `[UI-07]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tip<'a> {
    /// The first line, in the label style.
    pub title: &'a str,
    /// Wrapped body text.
    pub text: &'a str,
    /// Name / value rows (muted name, value flush right).
    pub rows: &'a [(&'a str, &'a str)],
    /// A shortcut shown after the title.
    pub shortcut: Option<Shortcut>,
}

impl<'a> Tip<'a> {
    /// A tooltip with just a title.
    pub fn new(title: &'a str) -> Self {
        Self { title, ..Self::default() }
    }

    /// Wrapped body text.
    #[must_use]
    pub fn text(mut self, text: &'a str) -> Self {
        self.text = text;
        self
    }

    /// Name / value rows.
    #[must_use]
    pub fn rows(mut self, rows: &'a [(&'a str, &'a str)]) -> Self {
        self.rows = rows;
        self
    }

    /// The shortcut of the action.
    #[must_use]
    pub fn shortcut(mut self, shortcut: Shortcut) -> Self {
        self.shortcut = Some(shortcut);
        self
    }
}

/// Shows `tip` for the widget of `r` once the pointer has rested on it (or at once while it has
/// the keyboard focus) `[UI-09]`. Returns whether it is shown.
pub fn tooltip(ui: &mut Ui, r: &Response, tip: &Tip<'_>) -> bool {
    if !ui.tip_ready(r) {
        return false;
    }
    show_tip(ui, r.rect, tip);
    true
}

/// Height of a name / value row of a tooltip (reference `draw_tip`).
const TIP_ROW: f32 = 22.0;

/// Draws `tip` below `anchor` (above it where there is no room), inside the viewport.
pub fn show_tip(ui: &mut Ui, anchor: Rect, tip: &Tip<'_>) {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let vp = ui.viewport();
    let pad = m.tooltip_pad;
    let max_w = m.tooltip_width.min(vp.w - 24.0).max(60.0);
    let max_inner = max_w - 2.0 * pad;
    let (title_style, small) = (theme.text.label, theme.text.small);
    let mut key = kit::Buf::<48>::new();
    if let Some(s) = tip.shortcut {
        let _ = std::fmt::write(&mut key, format_args!("{s}"));
    }
    let key_w =
        if key.as_str().is_empty() { 0.0 } else { ui.measure_text(key.as_str(), theme.text.mono).w + m.space_5 };
    let title_w = if tip.title.is_empty() { 0.0 } else { ui.measure_text(tip.title, title_style).w } + key_w;
    let text_w = if tip.text.is_empty() { 0.0 } else { ui.measure_text(tip.text, small).w };
    let rows_w = if tip.rows.is_empty() { 0.0 } else { max_inner };
    let inner = title_w.max(text_w).max(rows_w).min(max_inner).ceil();
    let title_h = if title_w > 0.0 { title_style.size * title_style.line_height } else { 0.0 };
    let text_h = if tip.text.is_empty() { 0.0 } else { ui.text_layout(tip.text, small, Some(inner)).size.h };
    #[allow(clippy::cast_precision_loss)] // A handful of rows.
    let rows_h = tip.rows.len() as f32 * TIP_ROW;
    let parts = [title_h, text_h, rows_h];
    let gaps = parts.iter().filter(|h| **h > 0.0).count().saturating_sub(1);
    #[allow(clippy::cast_precision_loss)]
    let height = 2.0 * pad + parts.iter().sum::<f32>() + gaps as f32 * m.space_3;
    let width = inner + 2.0 * pad;
    let below = anchor.bottom() + 6.0;
    let y = if below + height <= vp.bottom() - 8.0 { below } else { (anchor.y - 6.0 - height).max(vp.y + 8.0) };
    let x = anchor.x.min(vp.right() - width - 12.0).max(vp.x + 12.0);
    let rect = Rect::new(x.round(), y.round(), width, height);
    let node = ui.begin_overlay(rect, Layer::Tooltip, Style::column());
    kit::card(ui, node, theme.shadows.tooltip, p.surface, m.radius, 1.0);
    let mut top = pad;
    if title_h > 0.0 {
        kit::text(ui, node, kit::at(pad, top, inner, title_h), tip.title, title_style, p.text, Align::Start);
        kit::text(ui, node, kit::at(pad, top, inner, title_h), key.as_str(), theme.text.mono, p.text_faint, Align::End);
        top += title_h + m.space_3;
    }
    if text_h > 0.0 {
        let color = if title_h > 0.0 { p.text_muted } else { p.text };
        kit::wrapped(ui, node, kit::at(pad, top, inner, text_h), tip.text, small, color);
        top += text_h + m.space_3;
    }
    for (name, value) in tip.rows {
        kit::text(ui, node, kit::at(pad, top, inner, TIP_ROW), name, small, p.text_muted, Align::Start);
        kit::text(ui, node, kit::at(pad, top, inner, TIP_ROW), value, small, p.text, Align::End);
        top += TIP_ROW;
    }
    let id = Id::ROOT.with("tooltip");
    ui.semantic(id, Role::Custom("tooltip"), tip.title, SemanticValue::None);
    ui.end_overlay();
}

/// One entry of a menu or list popup.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MenuItem<'a> {
    /// Something to pick.
    Action {
        /// Its text.
        label: &'a str,
        /// An icon before the text.
        icon: Option<char>,
        /// The shortcut shown after the text.
        shortcut: Option<Shortcut>,
        /// A check mark (`Some(true)` ticked).
        checked: Option<bool>,
        /// Whether it can be picked.
        enabled: bool,
    },
    /// A hairline between groups.
    Separator,
}

impl<'a> MenuItem<'a> {
    /// An enabled action with a label.
    pub const fn action(label: &'a str) -> Self {
        Self::Action { label, icon: None, shortcut: None, checked: None, enabled: true }
    }

    /// The same item with an icon.
    #[must_use]
    pub const fn icon(self, glyph: char) -> Self {
        match self {
            Self::Action { label, shortcut, checked, enabled, .. } => {
                Self::Action { label, icon: Some(glyph), shortcut, checked, enabled }
            }
            Self::Separator => self,
        }
    }

    /// The same item with a shortcut.
    #[must_use]
    pub const fn shortcut(self, s: Shortcut) -> Self {
        match self {
            Self::Action { label, icon, checked, enabled, .. } => {
                Self::Action { label, icon, shortcut: Some(s), checked, enabled }
            }
            Self::Separator => self,
        }
    }

    /// The same item with a check mark state.
    #[must_use]
    pub const fn checked(self, on: bool) -> Self {
        match self {
            Self::Action { label, icon, shortcut, enabled, .. } => {
                Self::Action { label, icon, shortcut, checked: Some(on), enabled }
            }
            Self::Separator => self,
        }
    }

    /// The same item enabled or not.
    #[must_use]
    pub const fn enabled(self, on: bool) -> Self {
        match self {
            Self::Action { label, icon, shortcut, checked, .. } => {
                Self::Action { label, icon, shortcut, checked, enabled: on }
            }
            Self::Separator => self,
        }
    }

    fn pickable(&self) -> bool {
        matches!(self, Self::Action { enabled: true, .. })
    }

    fn label(&self) -> &'a str {
        match self {
            Self::Action { label, .. } => label,
            Self::Separator => "",
        }
    }
}

/// The state of an open (or closed) popup list, in [`Ui::memory`] under its owner's id.
#[derive(Clone, Debug, Default)]
pub struct PopupState {
    /// Whether the list is open.
    pub open: bool,
    anchor: Rect,
    highlight: Option<usize>,
    appear: f32,
    query: String,
    filtered: Vec<u32>,
    focus_search: bool,
    reveal: bool,
}

impl PopupState {
    /// Opens the list under (or above) `anchor` with `highlight` marked.
    pub fn open_at(&mut self, anchor: Rect, highlight: Option<usize>) {
        self.open = true;
        self.anchor = anchor;
        self.highlight = highlight;
        self.appear = 0.0;
        self.query.clear();
        self.focus_search = true;
        self.reveal = true;
    }
}

/// Row height of a menu (reference: 34 px).
const MENU_ROW: f32 = 34.0;
/// Height of a separator.
const MENU_SEPARATOR: f32 = 9.0;
/// Tallest list (reference: 440 px).
const MENU_MAX_H: f32 = 440.0;
/// Height of the search row of a combo box.
const SEARCH_H: f32 = 40.0;

fn row_h(item: &MenuItem<'_>) -> f32 {
    if matches!(item, MenuItem::Separator) { MENU_SEPARATOR } else { MENU_ROW }
}

/// Lower-case-insensitive "contains" without allocating.
pub fn contains_ignore_case(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    let lower = |c: char| c.to_lowercase().next().unwrap_or(c);
    let n: usize = needle.chars().count();
    let starts = haystack.char_indices().map(|(i, _)| i);
    for start in starts {
        let mut hs = haystack[start..].chars();
        let mut ok = true;
        let mut count = 0;
        for nc in needle.chars() {
            match hs.next() {
                Some(hc) if lower(hc) == lower(nc) => count += 1,
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && count == n {
            return true;
        }
    }
    false
}

/// What a popup list did this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ListOutcome {
    /// The item picked (an index into the caller's items).
    pub picked: Option<usize>,
    /// The list closed this frame (picked, Escape, Tab or a click outside).
    pub closed: bool,
}

/// Draws an open popup list of `count` items (`item(i)`) for the owner `owner` and handles its
/// input; with `search`, a search field filters the items by label. Does nothing when closed.
#[allow(clippy::too_many_lines)]
pub fn popup_list<'a>(
    ui: &mut Ui,
    owner: Id,
    width: f32,
    count: usize,
    item: &dyn Fn(usize) -> MenuItem<'a>,
    search: bool,
) -> ListOutcome {
    let mut st = std::mem::take(ui.memory::<PopupState>(owner));
    let mut out = ListOutcome::default();
    if !st.open {
        *ui.memory::<PopupState>(owner) = st;
        return out;
    }
    ui.begin_key_grab();
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    // The visible items: all, or those matching the query.
    st.filtered.clear();
    for i in 0..count {
        if !search || contains_ignore_case(item(i).label(), &st.query) {
            st.filtered.push(u32::try_from(i).unwrap_or(u32::MAX));
        }
    }
    let visible = st.filtered.len();
    let at = |st: &PopupState, k: usize| st.filtered.get(k).map_or(0, |&i| i as usize);
    if st.highlight.is_some_and(|h| h >= visible) {
        st.highlight = None;
    }
    if search && st.highlight.is_none() && visible > 0 {
        st.highlight = (0..visible).find(|&k| item(at(&st, k)).pickable());
    }

    // Keyboard.
    let step = |st: &PopupState, from: Option<usize>, dir: i64| -> Option<usize> {
        let n = i64::try_from(visible).unwrap_or(0);
        if n == 0 {
            return None;
        }
        let mut k = from.map_or(if dir > 0 { -1 } else { n }, |h| i64::try_from(h).unwrap_or(0));
        for _ in 0..n {
            k = (k + dir).rem_euclid(n);
            let ku = usize::try_from(k).unwrap_or(0);
            if item(at(st, ku)).pickable() {
                return Some(ku);
            }
        }
        from
    };
    let mut moved = false;
    if ui.consume_key(Key::Down, Modifiers::NONE) {
        st.highlight = step(&st, st.highlight, 1);
        moved = true;
    }
    if ui.consume_key(Key::Up, Modifiers::NONE) {
        st.highlight = step(&st, st.highlight, -1);
        moved = true;
    }
    if ui.consume_key(Key::PageDown, Modifiers::NONE) {
        for _ in 0..8 {
            st.highlight = step(&st, st.highlight, 1);
        }
        moved = true;
    }
    if ui.consume_key(Key::PageUp, Modifiers::NONE) {
        for _ in 0..8 {
            st.highlight = step(&st, st.highlight, -1);
        }
        moved = true;
    }
    if !search {
        if ui.consume_key(Key::Home, Modifiers::NONE) {
            st.highlight = step(&st, None, 1);
            moved = true;
        }
        if ui.consume_key(Key::End, Modifiers::NONE) {
            st.highlight = step(&st, None, -1);
            moved = true;
        }
        // A letter jumps to the next item starting with it.
        let typed = ui.typed_text().0.chars().next();
        if let Some(c) = typed.filter(|c| c.is_alphanumeric()) {
            let lower = c.to_lowercase().next().unwrap_or(c);
            let start = st.highlight.map_or(0, |h| h + 1);
            for d in 0..visible {
                let k = (start + d) % visible;
                let it = item(at(&st, k));
                let first = it.label().chars().next().map(|f| f.to_lowercase().next().unwrap_or(f));
                if it.pickable() && first == Some(lower) {
                    st.highlight = Some(k);
                    moved = true;
                    break;
                }
            }
        }
    }
    let pick_key =
        ui.consume_key(Key::Enter, Modifiers::NONE) || (!search && ui.consume_key(Key::Space, Modifiers::NONE));
    if pick_key && let Some(h) = st.highlight {
        out.picked = Some(at(&st, h));
    }
    if ui.consume_key(Key::Escape, Modifiers::NONE) || ui.consume_key_any(Key::Tab).is_some() {
        out.closed = true;
    }

    // Geometry: below the anchor, or above it when there is more room there.
    let vp = ui.viewport();
    let search_h = if search { SEARCH_H } else { 0.0 };
    let mut content = 0.0;
    for k in 0..visible {
        content += row_h(&item(at(&st, k)));
    }
    let room_below = vp.bottom() - st.anchor.bottom() - 16.0;
    let room_above = st.anchor.y - vp.y - 16.0;
    let want = (content + search_h + 12.0).min(MENU_MAX_H);
    let below = room_below >= want || room_below >= room_above;
    let room = if below { room_below } else { room_above };
    let height = want.min(room.max(120.0));
    let w = width.max(st.anchor.w).min(vp.w - 24.0);
    let x = st.anchor.x.min(vp.right() - w - 12.0).max(vp.x + 12.0);
    let y = if below { st.anchor.bottom() + 6.0 } else { st.anchor.y - 6.0 - height };
    let rect = Rect::new(x.round(), y.round(), w, height);

    // A click outside closes it (the catcher lies under the list on the popup layer).
    let catcher = ui.begin_overlay(vp, Layer::Popup, Style::default());
    let cr = ui.interact(owner.with("outside"), catcher, Sense { click: true, drag: false, focus: false });
    if cr.pressed || cr.secondary_clicked {
        out.closed = true;
    }
    ui.end_overlay();

    let dt = ui.dt();
    st.appear = ease(st.appear, 1.0, dt, theme.motion.ease_rate, theme.motion.reduced);
    if st.appear < 1.0 {
        ui.request_repaint();
    }
    let card = ui.begin_overlay(rect, Layer::Popup, Style { padding: Insets::all(6.0), ..Style::column() });
    kit::card(ui, card, theme.shadows.menu, p.surface, m.radius, st.appear);
    // Swallow pointer input on the card's padding.
    ui.interact(owner.with("card"), card, Sense { click: true, drag: false, focus: false });
    ui.begin_semantic_group(owner);
    ui.push_id(owner);
    if search {
        let row = ui.begin(Style { height: Dim::Px(SEARCH_H - 6.0), padding: Insets::xy(0.0, 0.0), ..Style::row() });
        let _ = row;
        let field = TextField::new("search", &mut st.query).placeholder("Search").icon(crate::icons::SEARCH).show(ui);
        if st.focus_search {
            ui.request_focus(field.response.id);
            st.focus_search = false;
        }
        if field.response.changed {
            st.highlight = None;
            moved = true;
        }
        ui.end();
    }
    let list_id = owner.with("list");
    let list = ui.begin_scroll(list_id, Style { grow: 1.0, ..Style::column() });
    let _ = list;
    let (offset, view) = ui.scroll_view(list_id);
    let view = if view > 0.0 { view } else { height };
    // Virtualised: rows outside the view are one spacer above and one below.
    let (mut y, mut before, mut after) = (0.0_f32, 0.0_f32, 0.0_f32);
    let mut started = false;
    let mut reveal: Option<(f32, f32)> = None;
    for k in 0..visible {
        let i = at(&st, k);
        let it = item(i);
        let h = row_h(&it);
        if st.highlight == Some(k) && (moved || st.reveal) {
            reveal = Some((y, h));
        }
        let shown = y + h >= offset - MENU_ROW && y <= offset + view + MENU_ROW;
        if !shown {
            if started {
                after += h;
            } else {
                before += h;
            }
            y += h;
            continue;
        }
        if !started {
            started = true;
            if before > 0.0 {
                ui.node(Style { height: Dim::Px(before), ..Style::default() });
            }
        }
        y += h;
        let node = ui.node(Style { height: Dim::Px(h), shrink: 0.0, ..Style::default() });
        match it {
            MenuItem::Separator => {
                kit::fill(ui, node, kit::at(4.0, 4.0, 0.0, m.border).with_right(-4.0), p.border, 0.0);
            }
            MenuItem::Action { label, icon, shortcut, checked, enabled } => {
                let rid = list_id.with_index(i as u64);
                let r = ui.interact(rid, node, Sense { click: enabled, drag: false, focus: false });
                if r.hovered && enabled && (ui.pointer_delta() != Point::ZERO || st.highlight.is_none()) {
                    st.highlight = Some(k);
                }
                if r.clicked && enabled {
                    out.picked = Some(i);
                }
                let lit = st.highlight == Some(k) && enabled;
                kit::fill(
                    ui,
                    node,
                    kit::at(0.0, 0.0, 0.0, h - 2.0).with_right(0.0),
                    p.hover.fade(if lit { 1.0 } else { 0.0 }),
                    m.radius_sm,
                );
                let ink = if enabled { p.text } else { p.text_faint };
                let mut left = 10.0;
                if let Some(g) = icon {
                    kit::icon(
                        ui,
                        node,
                        Anchor::start(left + 8.0),
                        Anchor::at(0.5, -1.0),
                        g,
                        m.icon,
                        if enabled { p.text_muted } else { p.text_faint },
                    );
                    left += 26.0;
                }
                let mut right = 10.0;
                if checked == Some(true) {
                    kit::icon(ui, node, Anchor::end(-16.0), Anchor::at(0.5, -1.0), crate::icons::CHECK, 14.0, p.text);
                    right += 24.0;
                }
                if let Some(s) = shortcut {
                    let text = kit::fmt64(format_args!("{s}"));
                    kit::text(
                        ui,
                        node,
                        kit::at(0.0, 0.0, 0.0, h - 2.0).with_left(left).with_right(-right),
                        text.as_str(),
                        theme.text.tiny,
                        p.text_faint,
                        Align::End,
                    );
                }
                kit::text(
                    ui,
                    node,
                    kit::at(0.0, 0.0, 0.0, h - 2.0).with_left(left).with_right(-right),
                    label,
                    theme.text.small,
                    ink,
                    Align::Start,
                );
                ui.semantic(
                    rid,
                    Role::Custom("menuitem"),
                    label,
                    checked.map_or(SemanticValue::None, SemanticValue::Toggle),
                );
                if !enabled {
                    ui.semantic_disabled();
                }
            }
        }
    }
    if after > 0.0 {
        ui.node(Style { height: Dim::Px(after), ..Style::default() });
    }
    ui.end_scroll(list_id);
    if let Some((ry, rh)) = reveal {
        if ry < offset {
            ui.scroll_to(list_id, ry);
        } else if ry + rh > offset + view {
            ui.scroll_to(list_id, ry + rh - view);
        }
        st.reveal = false;
    }
    if visible == 0 {
        let empty = ui.node(Style { height: Dim::Px(MENU_ROW), ..Style::default() });
        kit::text(ui, empty, Area::inset(10.0), "No matches", theme.text.small, p.text_faint, Align::Start);
    }
    ui.pop_id();
    ui.end_semantic_group();
    ui.end_overlay();
    if out.picked.is_some() {
        out.closed = true;
    }
    if out.closed {
        st.open = false;
        ui.request_focus(owner);
    }
    ui.end_key_grab(!out.closed);
    *ui.memory::<PopupState>(owner) = st;
    out
}

/// Area helpers for the popup rows.
trait AreaExt {
    fn with_right(self, px: f32) -> Self;
    fn with_left(self, px: f32) -> Self;
}

impl AreaExt for Area {
    fn with_right(mut self, px: f32) -> Self {
        self.x1 = Anchor::end(px);
        self
    }

    fn with_left(mut self, px: f32) -> Self {
        self.x0 = Anchor::start(px);
        self
    }
}

/// A button that opens a menu of `items` below it `[UI-07]` (toolbars: "File", "View"). Returns
/// the index of the item picked this frame.
pub fn menu_button(ui: &mut Ui, label: &str, items: &[MenuItem<'_>]) -> Option<usize> {
    let r = crate::widgets::Button::new(label).kind(crate::widgets::ButtonKind::Ghost).show(ui);
    let id = r.id;
    let open = ui.memory::<PopupState>(id).open;
    if (r.clicked && !open) || (r.focused && ui.consume_key(Key::Down, Modifiers::NONE)) {
        ui.memory::<PopupState>(id).open_at(r.rect, None);
    }
    popup_list(ui, id, 220.0, items.len(), &|i| items.get(i).copied().unwrap_or(MenuItem::Separator), false).picked
}

/// A context menu `[UI-07]` for the widget of `r`: opens on a secondary click (at the pointer) or
/// Shift+F10 / the context-menu key (under the widget). Returns the index of the item picked.
pub fn context_menu(ui: &mut Ui, r: &Response, items: &[MenuItem<'_>]) -> Option<usize> {
    let id = r.id.with("context");
    if r.secondary_clicked {
        let at = ui
            .pointer()
            .filter(|p| r.rect.contains(*p))
            .map_or(Rect::new(r.rect.x, r.rect.bottom(), 0.0, 0.0), |p| Rect::new(p.x, p.y - 6.0, 0.0, 0.0));
        let first = items.iter().position(MenuItem::pickable);
        ui.memory::<PopupState>(id).open_at(at, first);
    }
    let out = popup_list(ui, id, 220.0, items.len(), &|i| items.get(i).copied().unwrap_or(MenuItem::Separator), false);
    if out.closed {
        ui.request_focus(r.id);
    }
    out.picked
}

/// A dropdown choosing one of `options` `[UI-07]` (reference `dropdown_frame`): the choice with a
/// chevron; a click, Enter, Space or Alt+Down opens the list; Up / Down change the choice while
/// it is closed.
#[derive(Debug)]
pub struct Dropdown<'a> {
    id_label: &'a str,
    selected: &'a mut usize,
    options: &'a [&'a str],
    width: Option<f32>,
}

impl<'a> Dropdown<'a> {
    /// A dropdown identified by `id_label`.
    pub fn new(id_label: &'a str, selected: &'a mut usize, options: &'a [&'a str]) -> Self {
        Self { id_label, selected, options, width: None }
    }

    /// A fixed width (otherwise it grows).
    #[must_use]
    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width);
        self
    }

    /// Adds the dropdown; `changed` when the choice changed.
    pub fn show(self, ui: &mut Ui) -> Response {
        let n = self.options.len();
        let label = self.options.get(*self.selected).copied().unwrap_or("");
        let (mut r, open) = dropdown_frame(ui, self.id_label, label, self.width);
        let before = *self.selected;
        if r.focused && !open && n > 0 {
            if ui.consume_key(Key::Down, Modifiers::NONE) {
                *self.selected = (*self.selected + 1).min(n - 1);
            }
            if ui.consume_key(Key::Up, Modifiers::NONE) {
                *self.selected = self.selected.saturating_sub(1);
            }
        }
        let wants_open =
            r.clicked || (r.focused && ui.consume_key(Key::Down, Modifiers { alt: true, ..Modifiers::NONE }));
        if wants_open && !open {
            let sel = *self.selected;
            ui.memory::<PopupState>(r.id).open_at(r.rect, Some(sel));
        }
        let sel = *self.selected;
        let opts = self.options;
        let picked = popup_list(
            ui,
            r.id,
            r.rect.w,
            n,
            &|i| MenuItem::action(opts.get(i).copied().unwrap_or("")).checked(i == sel),
            false,
        )
        .picked;
        if let Some(i) = picked {
            *self.selected = i;
        }
        r.changed = *self.selected != before;
        r
    }
}

/// The closed face of a dropdown or combo box; returns its response and whether its list is open.
fn dropdown_frame(ui: &mut Ui, id_label: &str, label: &str, width: Option<f32>) -> (Response, bool) {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let node = ui.node(Style {
        width: width.map_or(Dim::Auto, Dim::Px),
        grow: if width.is_some() { 0.0 } else { kit::grow_x(ui) },
        min_width: 80.0,
        height: Dim::Px(m.control_height),
        ..Style::default()
    });
    let id = ui.id(id_label);
    let r = ui.interact(id, node, Sense::CLICK);
    let open = ui.memory::<PopupState>(id).open;
    let hover = if open { 1.0 } else { r.hover_t };
    kit::frame(
        ui,
        node,
        Area::FULL,
        p.bg.mix(p.hover, hover * 0.6),
        m.radius_sm,
        p.border_strong.mix(p.border_focus, hover),
    );
    kit::text(ui, node, kit::inset_x(m.control_pad_x, 30.0), label, theme.text.body, p.text, Align::Start);
    kit::icon(ui, node, Anchor::end(-17.0), Anchor::at(0.5, 0.0), crate::icons::CHEVRON_DOWN, 14.0, p.text_faint);
    if r.focus_visible {
        kit::ring(ui, node, Area::FULL, m.radius_sm);
    }
    let value = ui.intern(label);
    ui.semantic(id, Role::Custom("combobox"), id_label, SemanticValue::Text(value));
    (r, open)
}

/// A combo box with search `[UI-07]`: like [`Dropdown`], but the list has a search field that
/// filters the options as you type; `None` means nothing chosen.
#[derive(Debug)]
pub struct ComboBox<'a> {
    id_label: &'a str,
    selected: &'a mut Option<usize>,
    options: &'a [&'a str],
    placeholder: &'a str,
    width: Option<f32>,
}

impl<'a> ComboBox<'a> {
    /// A combo box identified by `id_label`.
    pub fn new(id_label: &'a str, selected: &'a mut Option<usize>, options: &'a [&'a str]) -> Self {
        Self { id_label, selected, options, placeholder: "Choose\u{2026}", width: None }
    }

    /// Text shown while nothing is chosen.
    #[must_use]
    pub fn placeholder(mut self, placeholder: &'a str) -> Self {
        self.placeholder = placeholder;
        self
    }

    /// A fixed width (otherwise it grows).
    #[must_use]
    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width);
        self
    }

    /// Adds the combo box; `changed` when the choice changed.
    pub fn show(self, ui: &mut Ui) -> Response {
        let label = self.selected.and_then(|i| self.options.get(i).copied()).unwrap_or(self.placeholder);
        let (mut r, open) = dropdown_frame(ui, self.id_label, label, self.width);
        let before = *self.selected;
        let wants_open = r.clicked || (r.focused && ui.consume_key(Key::Down, Modifiers::NONE));
        if wants_open && !open {
            ui.memory::<PopupState>(r.id).open_at(r.rect, None);
        }
        let sel = *self.selected;
        let opts = self.options;
        let picked = popup_list(
            ui,
            r.id,
            r.rect.w.max(240.0),
            opts.len(),
            &|i| MenuItem::action(opts.get(i).copied().unwrap_or("")).checked(Some(i) == sel),
            true,
        )
        .picked;
        if let Some(i) = picked {
            *self.selected = Some(i);
        }
        r.changed = *self.selected != before;
        r
    }
}

/// What a modal dialog did this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModalOutcome {
    /// Escape, the close button or a click outside asked to close it.
    pub close: bool,
}

/// A modal dialog `[UI-07]` (reference `widgets::modal`): dims the window with the scrim, draws a
/// centred card of at most the requested size with a title and a close button, traps the keyboard
/// focus inside it `[UI-09]` and asks to close on Escape, the close button or a click outside.
#[derive(Clone, Copy, Debug)]
pub struct Modal<'a> {
    title: &'a str,
    width: f32,
    height: f32,
}

impl<'a> Modal<'a> {
    /// A dialog titled (and identified by) `title`, `width × height` logical pixels at most.
    pub fn new(title: &'a str, width: f32, height: f32) -> Self {
        Self { title, width, height }
    }

    /// Adds the dialog with the content built by `content` (a column with padding).
    pub fn show<R>(self, ui: &mut Ui, content: impl FnOnce(&mut Ui) -> R) -> (R, ModalOutcome) {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let id = ui.id(self.title);
        let vp = ui.viewport();
        let dt = ui.dt();
        let appear = {
            let a = ui.memory::<f32>(id);
            *a = ease(*a, 1.0, dt, theme.motion.ease_rate, theme.motion.reduced);
            *a
        };
        if appear < 1.0 {
            ui.request_repaint();
        }
        let mut out = ModalOutcome::default();
        let scrim = ui.begin_overlay(vp, Layer::Popup, Style::default());
        kit::fill(ui, scrim, Area::FULL, p.scrim.fade(appear), 0.0);
        let sr = ui.interact(id.with("scrim"), scrim, Sense { click: true, drag: false, focus: false });
        if sr.clicked {
            out.close = true;
        }
        ui.end_overlay();
        let (w, h) = (self.width.min(vp.w - 48.0).max(0.0), self.height.min(vp.h - 64.0).max(0.0));
        let slide = if theme.motion.reduced { 0.0 } else { 10.0 * (1.0 - appear) };
        let rect = Rect::new((vp.x + (vp.w - w) * 0.5).round(), (vp.y + (vp.h - h) * 0.5 + slide).round(), w, h);
        let card = ui.begin_overlay(
            rect,
            Layer::Popup,
            Style { padding: Insets::all(m.space_6), gap: m.space_5, ..Style::column() },
        );
        kit::card(ui, card, theme.shadows.modal, p.bg, m.radius_lg, appear);
        ui.interact(id.with("card"), card, Sense { click: true, drag: false, focus: false });
        ui.begin_semantic_group(id);
        ui.semantic(id, Role::Window, self.title, SemanticValue::None);
        ui.push_id(id);
        ui.begin_focus_trap();
        let head = ui.begin(Style {
            height: Dim::Px(m.icon_button),
            align_items: crate::layout::Align::Center,
            ..Style::row()
        });
        let title = ui.node(Style { grow: 1.0, ..Style::default() });
        kit::text(ui, title, Area::FULL, self.title, theme.text.heading, p.text, Align::Start);
        let _ = head;
        if IconButton::new(crate::icons::X, "Close").show(ui).clicked {
            out.close = true;
        }
        ui.end();
        ui.begin(Style { grow: 1.0, gap: m.space_4, ..Style::column() });
        let result = content(ui);
        ui.end();
        ui.end_focus_trap();
        ui.pop_id();
        ui.end_semantic_group();
        ui.end_overlay();
        if ui.consume_key(Key::Escape, Modifiers::NONE) {
            out.close = true;
        }
        if out.close {
            *ui.memory::<f32>(id) = 0.0;
        }
        (result, out)
    }
}

/// The kind of a toast notification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToastKind {
    /// Information.
    Info,
    /// Something finished well.
    Success,
    /// Something needs attention.
    Warning,
    /// Something failed.
    Error,
}

/// One toast.
#[derive(Clone, Debug)]
struct Toast {
    id: u64,
    kind: ToastKind,
    text: String,
    created: Option<f64>,
    ttl: f32,
    hovered: bool,
}

/// Toast notifications `[UI-07]` (reference notices: up to three dismissible cards stacked at the
/// top right). The application owns the list, pushes toasts on events and calls [`Toasts::show`]
/// every frame; a toast disappears after its time unless hovered, or on its close button.
#[derive(Clone, Debug, Default)]
pub struct Toasts {
    items: Vec<Toast>,
    next: u64,
}

impl Toasts {
    /// Most toasts shown at once; older ones wait.
    pub const VISIBLE: usize = 3;
    /// Most toasts kept; the oldest is dropped beyond it.
    pub const MAX: usize = 32;

    /// No toasts.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a toast shown for `seconds` (0 or less: until dismissed); returns its id.
    pub fn push(&mut self, kind: ToastKind, text: &str, seconds: f32) -> u64 {
        if self.items.len() >= Self::MAX {
            self.items.remove(0);
        }
        self.next += 1;
        let ttl = if seconds.is_finite() { seconds } else { 0.0 };
        self.items.push(Toast { id: self.next, kind, text: text.to_owned(), created: None, ttl, hovered: false });
        self.next
    }

    /// Removes a toast.
    pub fn dismiss(&mut self, id: u64) {
        self.items.retain(|t| t.id != id);
    }

    /// How many toasts are waiting or shown.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Draws the toasts at the top right, `top` pixels below the window's top, and expires them.
    pub fn show(&mut self, ui: &mut Ui, top: f32) {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let now = ui.time();
        let mut dismiss = None;
        self.items.retain(|t| match t.created {
            Some(c) if t.ttl > 0.0 && !t.hovered => now - c < f64::from(t.ttl),
            _ => true,
        });
        let vp = ui.viewport();
        let width = 340.0_f32.min(vp.w - 24.0);
        let pad = m.space_5;
        let text_w = width - 2.0 * pad - 22.0 - m.icon_button;
        let mut y = top + m.space_5;
        for t in self.items.iter_mut().take(Self::VISIBLE) {
            if t.created.is_none() {
                t.created = Some(now);
            }
            if t.ttl > 0.0 {
                ui.request_repaint();
            }
            let text_h = ui.text_layout(&t.text, theme.text.small, Some(text_w)).size.h;
            let h = (text_h + 2.0 * pad).max(m.icon_button + 2.0 * m.space_3);
            let rect = Rect::new(vp.right() - width - m.space_5, y, width, h);
            y += h + m.space_4;
            let id = Id::ROOT.with("toast").with_index(t.id);
            let node =
                ui.begin_overlay(rect, Layer::Tooltip, Style { padding: Insets::all(m.space_3), ..Style::row() });
            kit::card(ui, node, theme.shadows.notice, p.surface, m.radius, 1.0);
            let r = ui.interact(id, node, Sense::HOVER);
            t.hovered = r.hovered;
            let (glyph, tint) = match t.kind {
                ToastKind::Info => (crate::icons::INFO, p.link),
                ToastKind::Success => (crate::icons::CHECK_CIRCLE, p.positive),
                ToastKind::Warning => (crate::icons::ALERT_TRIANGLE, p.warning),
                ToastKind::Error => (crate::icons::ALERT_CIRCLE, p.danger),
            };
            kit::icon(ui, node, Anchor::start(pad + 7.0), Anchor::start(pad + 9.0), glyph, m.icon, tint);
            kit::wrapped(ui, node, kit::at(pad + 22.0, pad, text_w, text_h), &t.text, theme.text.small, p.text);
            ui.push_id(id);
            let spacer = ui.node(Style { grow: 1.0, ..Style::default() });
            let _ = spacer;
            if IconButton::new(crate::icons::X, "Dismiss").size(m.control_height_sm).show(ui).clicked {
                dismiss = Some(t.id);
            }
            ui.pop_id();
            let role = match t.kind {
                ToastKind::Error => Role::Custom("alert"),
                _ => Role::Custom("status"),
            };
            ui.semantic(id, role, &t.text, SemanticValue::None);
            ui.end_overlay();
        }
        if let Some(d) = dismiss {
            self.dismiss(d);
        }
    }
}

/// The colour of a toast kind in the current theme (for status icons elsewhere).
pub fn toast_color(ui: &Ui, kind: ToastKind) -> Color {
    let p = ui.theme().palette;
    match kind {
        ToastKind::Info => p.link,
        ToastKind::Success => p.positive,
        ToastKind::Warning => p.warning,
        ToastKind::Error => p.danger,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::input::{Event, PointerButton};
    use crate::tests::Driver;
    use crate::theme::Theme;

    fn file_menu() -> [MenuItem<'static>; 6] {
        [
            MenuItem::action("New").icon(crate::icons::FILE).shortcut(Shortcut::new(Key::Letter('N'), Modifiers::CTRL)),
            MenuItem::action("Open"),
            MenuItem::Separator,
            MenuItem::action("Recent").enabled(false),
            MenuItem::action("Snap").checked(true),
            MenuItem::action("Quit"),
        ]
    }

    const UNITS: [&str; 3] = ["Metres", "Centimetres", "Inches"];
    const REGIONS: [&str; 6] = ["Brow", "Cheek", "Chin", "Ear", "Forehead", "Lips"];

    /// The state the popup gallery edits.
    struct Model {
        picked: Vec<usize>,
        context: Vec<usize>,
        unit: usize,
        region: Option<usize>,
        dialog: bool,
        closes: u32,
        toasts: Toasts,
    }

    impl Model {
        fn new() -> Self {
            Self {
                picked: Vec::new(),
                context: Vec::new(),
                unit: 0,
                region: None,
                dialog: false,
                closes: 0,
                toasts: Toasts::new(),
            }
        }
    }

    fn gallery(ui: &mut Ui, model: &mut Model) {
        let m = ui.theme().metrics;
        ui.begin(Style {
            padding: Insets::all(m.space_6),
            gap: m.space_4,
            width: Dim::Percent(100.0),
            ..Style::column()
        });
        ui.begin(Style { gap: m.space_4, ..Style::row() });
        if let Some(i) = menu_button(ui, "File", &file_menu()) {
            model.picked.push(i);
        }
        let canvas = crate::widgets::Button::new("Canvas").show(ui);
        let items =
            [MenuItem::action("Copy").icon(crate::icons::COPY), MenuItem::action("Delete").icon(crate::icons::TRASH)];
        if let Some(i) = context_menu(ui, &canvas, &items) {
            model.context.push(i);
        }
        ui.end();
        Dropdown::new("Units", &mut model.unit, &UNITS).width(180.0).show(ui);
        ComboBox::new("Region", &mut model.region, &REGIONS).width(180.0).show(ui);
        if crate::widgets::button(ui, "Open dialog").clicked {
            model.dialog = true;
        }
        ui.end();
        if model.dialog {
            let ((), out) = Modal::new("Export", 300.0, 180.0).show(ui, |ui| {
                crate::widgets::label(ui, "Export the character?");
                ui.begin(Style { gap: m.space_4, ..Style::row() });
                crate::widgets::button(ui, "Cancel");
                crate::widgets::Button::new("Export").kind(crate::widgets::ButtonKind::Primary).show(ui);
                ui.end();
            });
            if out.close {
                model.dialog = false;
                model.closes += 1;
            }
        }
        model.toasts.show(ui, 0.0);
    }

    fn setup(theme: &Theme) -> Driver {
        Driver::new(theme, 480.0, 420.0)
    }

    #[test]
    fn ignore_case_search_matches_anywhere_without_allocating() {
        assert!(contains_ignore_case("Forehead", "HEAD"));
        assert!(contains_ignore_case("L\u{e8}vres", "\u{c8}V"));
        assert!(contains_ignore_case("abc", ""));
        assert!(!contains_ignore_case("ab", "abc"));
        assert!(!contains_ignore_case("Cheek", "chin"));
    }

    #[test]
    fn ui_07_menu_opens_navigates_skips_and_picks() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        d.click(d.rect_of("File").center(), &mut build);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Quit").is_some(), "the menu is open");
        let file = d.rect_of("File");
        assert!(d.rect_of("New").y > file.bottom(), "below its button");
        // Down x3 from nothing: New, Open, (separator, disabled Recent skipped) Snap; Enter picks.
        for _ in 0..3 {
            d.key(Key::Down, Modifiers::NONE, &mut build);
        }
        d.key(Key::Enter, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().picked, [4]);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Quit").is_none(), "closed after the pick");
        assert_eq!(d.focused_label(), "File", "the focus returns to the button");
        // Reopen from the keyboard; a letter jumps; Escape closes without picking.
        d.key(Key::Down, Modifiers::NONE, &mut build);
        d.frame(vec![Event::Text('q')], &mut build);
        d.key(Key::Escape, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().picked, [4]);
        d.key(Key::Down, Modifiers::NONE, &mut build);
        d.frame(vec![Event::Text('q')], &mut build);
        d.key(Key::Space, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().picked, [4, 5]);
        // (Hit testing uses last frame's rectangles, where the closed list still lay.)
        d.frame(vec![], &mut build);
        // A click on a disabled item does nothing; a click outside closes.
        d.click(d.rect_of("File").center(), &mut build);
        d.frame(vec![], &mut build);
        d.click(d.rect_of("Recent").center(), &mut build);
        assert!(d.semantic("Recent").is_some_and(|s| s.disabled));
        d.click(Point::new(460.0, 400.0), &mut build);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Quit").is_none());
        assert_eq!(model.borrow().picked, [4, 5]);
        // The pointer picks by click.
        d.click(d.rect_of("File").center(), &mut build);
        d.frame(vec![], &mut build);
        d.click(d.rect_of("Open").center(), &mut build);
        assert_eq!(model.borrow().picked, [4, 5, 1]);
    }

    #[test]
    fn ui_07_context_menu_opens_on_secondary_click_and_shift_f10() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        let c = d.rect_of("Canvas").center();
        d.frame(
            vec![Event::PointerMoved(c), Event::Button { button: PointerButton::Secondary, pressed: true, pos: c }],
            &mut build,
        );
        d.frame(vec![Event::Button { button: PointerButton::Secondary, pressed: false, pos: c }], &mut build);
        let copy = d.rect_of("Copy");
        assert!((copy.x - c.x).abs() < 16.0 && copy.y >= c.y - 8.0, "at the pointer: {copy:?} vs {c:?}");
        d.key(Key::Down, Modifiers::NONE, &mut build);
        d.key(Key::Enter, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().context, [1]);
        // From the keyboard: focus the button, Shift+F10, the first item is highlighted.
        for _ in 0..20 {
            if d.focused_label() == "Canvas" {
                break;
            }
            d.key(Key::Tab, Modifiers::NONE, &mut build);
        }
        d.key(Key::F(10), Modifiers::SHIFT, &mut build);
        d.key(Key::Enter, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().context, [1, 0]);
        assert_eq!(d.focused_label(), "Canvas");
    }

    #[test]
    fn ui_07_dropdown_and_combo_with_search() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        for _ in 0..20 {
            if d.focused_label() == "Units" {
                break;
            }
            d.key(Key::Tab, Modifiers::NONE, &mut build);
        }
        // Closed: Down changes the choice directly.
        d.key(Key::Down, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().unit, 1);
        // Alt+Down opens with the choice highlighted; Down + Enter picks the next one.
        d.key(Key::Down, Modifiers { alt: true, ..Modifiers::NONE }, &mut build);
        d.frame(vec![Event::Modifiers(Modifiers::NONE)], &mut build);
        assert!(d.semantic("Inches").is_some());
        d.key(Key::Down, Modifiers::NONE, &mut build);
        d.key(Key::Enter, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().unit, 2);
        // The combo box: click, type to filter, Enter picks the first match.
        d.click(d.rect_of("Region").center(), &mut build);
        d.frame(vec![], &mut build);
        d.frame("he".chars().map(Event::Text).collect(), &mut build);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Cheek").is_some() && d.semantic("Forehead").is_some());
        assert!(d.semantic("Brow").is_none(), "filtered out");
        d.key(Key::Down, Modifiers::NONE, &mut build);
        d.key(Key::Enter, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().region, Some(4));
        // Nothing matches: the list says so and Enter picks nothing.
        d.click(d.rect_of("Region").center(), &mut build);
        d.frame(vec![], &mut build);
        d.frame("zz".chars().map(Event::Text).collect(), &mut build);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Cheek").is_none());
        d.key(Key::Enter, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().region, Some(4));
    }

    #[test]
    fn ui_09_modal_traps_the_focus_and_closes_on_escape() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        d.click(d.rect_of("Open dialog").center(), &mut build);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Export the character?").is_some());
        let inside = ["Close", "Cancel", "Export"];
        assert!(inside.contains(&d.focused_label().as_str()), "the focus moved into the dialog");
        for _ in 0..7 {
            d.key(Key::Tab, Modifiers::NONE, &mut build);
            let f = d.focused_label();
            assert!(inside.contains(&f.as_str()), "Tab left the dialog: {f}");
        }
        // A click on the panels behind the scrim only closes the dialog.
        d.click(d.rect_of("File").center(), &mut build);
        assert_eq!(model.borrow().closes, 1);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Quit").is_none(), "the menu behind did not open");
        d.click(d.rect_of("Open dialog").center(), &mut build);
        d.frame(vec![], &mut build);
        d.key(Key::Escape, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().closes, 2);
    }

    #[test]
    fn ui_07_toasts_stack_expire_and_dismiss() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        {
            let t = &mut model.borrow_mut().toasts;
            t.push(ToastKind::Success, "Saved", 1.0);
            t.push(ToastKind::Error, "Export failed", 0.0);
            t.push(ToastKind::Info, "Third", 0.0);
            t.push(ToastKind::Warning, "Fourth waits", 0.0);
        }
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Saved").is_some() && d.semantic("Third").is_some());
        assert!(d.semantic("Fourth waits").is_none(), "three at a time");
        assert_eq!(d.semantic("Export failed").map(|s| s.role), Some(Role::Custom("alert")));
        d.frame_dt(vec![], 1.1, &mut build);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Saved").is_none(), "expired");
        assert!(d.semantic("Fourth waits").is_some(), "the waiting one moves up");
        let first = d.semantic("Export failed").map_or(Rect::ZERO, |s| s.rect);
        let dismiss =
            d.ui.semantics()
                .iter()
                .filter(|s| d.ui.text(s.label) == "Dismiss")
                .map(|s| s.rect)
                .find(|r| first.contains(r.center()))
                .unwrap_or(Rect::ZERO);
        d.click(dismiss.center(), &mut build);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Export failed").is_none());
        assert_eq!(model.borrow().toasts.len(), 2);
        // The list is bounded.
        let mut many = Toasts::new();
        for _ in 0..100 {
            many.push(ToastKind::Info, "x", 0.0);
        }
        assert_eq!(many.len(), Toasts::MAX);
    }

    #[test]
    fn ac_wp12_2_popups_golden_at_100_150_200_percent() {
        // [UI-07] [AC-WP12-2] An open menu (icon, shortcut, separator, disabled and checked
        // items) with a keyboard highlight, a tooltip with rich text and toasts; then a modal.
        let rows = [("Region", "cheek.l"), ("Weight", "0.80")];
        let tip = Tip::new("Inflate")
            .text("Pushes the surface out along its normals.")
            .rows(&rows)
            .shortcut(Shortcut::new(Key::Letter('I'), Modifiers::NONE));
        for (scale, name) in [(1.0, "popups_100"), (1.5, "popups_150"), (2.0, "popups_200")] {
            let model = RefCell::new(Model::new());
            {
                let t = &mut model.borrow_mut().toasts;
                t.push(ToastKind::Success, "Character saved", 0.0);
                t.push(ToastKind::Error, "The texture could not be read: the file ends early.", 0.0);
            }
            let mut build = |ui: &mut Ui| {
                gallery(ui, &mut model.borrow_mut());
                show_tip(ui, Rect::new(250.0, 300.0, 40.0, 20.0), &tip);
            };
            let mut d = Driver::new(&Theme::dark().with_reduced_motion(true), 640.0, 420.0);
            d.scale = scale;
            d.frame(vec![], &mut build);
            d.frame(vec![], &mut build);
            d.click(d.rect_of("File").center(), &mut build);
            d.key(Key::Down, Modifiers::NONE, &mut build);
            d.frame(vec![], &mut build);
            d.golden(name);
        }
        let model = RefCell::new(Model::new());
        model.borrow_mut().dialog = true;
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = Driver::new(&Theme::light().with_reduced_motion(true), 480.0, 420.0);
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        d.golden("popups_modal_light");
    }
}
