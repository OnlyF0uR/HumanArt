//! Data views `[UI-07]`: the virtualised list, table and tree view, the image view with zoom and
//! pan, histogram and waveform plots, the timeline strip (the expression-sweep scrubber), the
//! property grid built from parameter descriptors, the console log panel and the keyboard-shortcut
//! overlay.
//!
//! Lists, tables, trees and the console share one virtualised row core: only the rows in view are
//! built (`[UI-04]` [`crate::layout::visible_rows`]), the list is one Tab stop, Up / Down / Page Up
//! / Page Down / Home / End move the selection and scroll it into view, Enter or a double click
//! activates the selected row, a secondary click or Shift+F10 asks for its context menu `[UI-09]`.
//! Cell text is written into stack buffers through [`std::fmt::Write`], so a steady frame does not
//! allocate however long the list is.

use std::collections::VecDeque;
use std::fmt::Write as _;

use crate::color::Color;
use crate::controls::{Collapsible, IconButton, checkbox_unlabelled, search_box};
use crate::editors::{DragValue, PickerResponse, color_edit};
use crate::geom::{Insets, Point, Size};
use crate::input::{Key, Keymap, Modifiers};
use crate::kit;
use crate::layout::{Align as BoxAlign, Dim, NodeId, Position, Style, visible_rows};
use crate::paint::{Corners, TextureId};
use crate::popup::{Modal, ModalOutcome, Tip, contains_ignore_case, tooltip};
use crate::text::Align;
use crate::ui::{Anchor, Area, CursorIcon, Id, Paint, Response, Role, SemanticValue, Sense, Ui};
use crate::widgets::{EditPhase, Slider, SliderResponse, TextField};

/// The longest cell text shown (bytes); longer text is cut.
pub const CELL_TEXT_MAX: usize = 192;

/// What a list, table, tree or console did this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RowsResponse {
    /// The list as a whole (`focused` while it has the keyboard focus).
    pub response: Response,
    /// The selection changed (click or keys).
    pub selection_changed: bool,
    /// A row was activated (Enter or a double click).
    pub activated: Option<usize>,
    /// A row asked for its context menu (secondary click, Shift+F10 on the selection).
    pub context: Option<usize>,
}

/// How a row is drawn this frame.
#[derive(Clone, Copy, Debug)]
struct RowState {
    selected: bool,
    hover: f32,
}

/// The rows core's memory.
#[derive(Clone, Copy, Debug, Default)]
struct RowsMemory {
    last_selected: Option<usize>,
}

/// The shared virtualised row list: a focusable box `height` tall holding `count` rows of `row_h`
/// pixels, of which only those in view are built; `draw` paints row `i` into its node.
#[allow(clippy::too_many_arguments)]
fn rows_core(
    ui: &mut Ui,
    id: Id,
    label: &str,
    count: usize,
    row_h: f32,
    height: f32,
    selected: &mut Option<usize>,
    draw: &mut dyn FnMut(&mut Ui, usize, NodeId, RowState),
) -> RowsResponse {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let row_h = if row_h.is_finite() { row_h.max(8.0) } else { m.row_height };
    if selected.is_some_and(|s| s >= count) {
        *selected = count.checked_sub(1);
    }
    let before = *selected;
    let mut out = RowsResponse::default();
    let outer = ui.begin(Style { height: Dim::Px(height), shrink: 0.0, ..Style::column() });
    let r = ui.interact(id, outer, Sense { click: false, drag: false, focus: true });
    let scroll_id = id.with("scroll");
    let (_, view) = ui.scroll_view(scroll_id);
    let view = if view > 0.0 { view } else { height };
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // A small positive count.
    let page = ((view / row_h).floor() as usize).max(1);
    if r.focused && count > 0 {
        let last = count - 1;
        let cur = *selected;
        let mut next = cur;
        if ui.consume_key(Key::Down, Modifiers::NONE) {
            next = Some(cur.map_or(0, |s| (s + 1).min(last)));
        }
        if ui.consume_key(Key::Up, Modifiers::NONE) {
            next = Some(cur.map_or(0, |s| s.saturating_sub(1)));
        }
        if ui.consume_key(Key::PageDown, Modifiers::NONE) {
            next = Some(cur.map_or(0, |s| (s + page).min(last)));
        }
        if ui.consume_key(Key::PageUp, Modifiers::NONE) {
            next = Some(cur.map_or(0, |s| s.saturating_sub(page)));
        }
        if ui.consume_key(Key::Home, Modifiers::NONE) {
            next = Some(0);
        }
        if ui.consume_key(Key::End, Modifiers::NONE) {
            next = Some(last);
        }
        *selected = next;
        if ui.consume_key(Key::Enter, Modifiers::NONE) {
            out.activated = *selected;
        }
    }
    if r.secondary_clicked {
        out.context = *selected;
    }
    kit::frame(ui, outer, Area::FULL, p.bg, m.radius_sm, if r.focus_visible { p.border_focus } else { p.border });
    let body = ui.begin_scroll(scroll_id, Style { grow: 1.0, padding: Insets::all(1.0), ..Style::column() });
    let _ = body;
    // The offset as this frame's scrolling left it (wheel, easing, `scroll_to`).
    let (offset, _) = ui.scroll_view(scroll_id);
    let rows = visible_rows(offset, view, row_h, count, 2);
    #[allow(clippy::cast_precision_loss)] // Row counts far below 2^24.
    let (above, below) = (rows.start as f32 * row_h, (count - rows.end) as f32 * row_h);
    if above > 0.0 {
        ui.node(Style { height: Dim::Px(above), shrink: 0.0, ..Style::default() });
    }
    for i in rows {
        let node = ui.begin(Style { height: Dim::Px(row_h), shrink: 0.0, ..Style::default() });
        let rr = ui.interact(id.with_index(i as u64), node, Sense { click: true, drag: false, focus: false });
        if rr.pressed || rr.secondary_clicked {
            *selected = Some(i);
            ui.request_focus(id);
        }
        if rr.double_clicked {
            out.activated = Some(i);
        }
        if rr.secondary_clicked {
            out.context = Some(i);
        }
        let sel = *selected == Some(i);
        let fill = if sel { if r.focused { p.selection } else { p.active } } else { p.hover.fade(rr.hover_t) };
        kit::fill(ui, node, Area::FULL, fill, m.radius_sm - 2.0);
        draw(ui, i, node, RowState { selected: sel, hover: rr.hover_t });
        ui.end();
    }
    if below > 0.0 {
        ui.node(Style { height: Dim::Px(below), shrink: 0.0, ..Style::default() });
    }
    ui.end_scroll(scroll_id);
    if count == 0 {
        let empty = ui.node(Style { height: Dim::Px(row_h), ..Style::default() });
        kit::text(ui, empty, Area::inset(m.space_4), "Empty", theme.text.small, p.text_faint, Align::Start);
    }
    ui.end();
    // Keep the selection in view when it moved.
    let mem = ui.memory::<RowsMemory>(id);
    let moved = mem.last_selected != *selected;
    mem.last_selected = *selected;
    if moved && let Some(s) = *selected {
        #[allow(clippy::cast_precision_loss)]
        let y = s as f32 * row_h;
        if y < offset {
            ui.scroll_to(scroll_id, y);
        } else if y + row_h > offset + view {
            ui.scroll_to(scroll_id, y + row_h - view + 2.0);
        }
    }
    out.selection_changed = *selected != before;
    #[allow(clippy::cast_precision_loss)]
    let value = SemanticValue::Number { value: selected.map_or(-1.0, |s| s as f64), min: 0.0, max: count as f64 };
    ui.semantic(id, Role::Custom("list"), label, value);
    out.response = r;
    out
}

/// A virtualised list of text rows `[UI-07]` (`item(i, out)` writes row `i`'s text). See the module
/// documentation for the keys.
pub fn list_view(
    ui: &mut Ui,
    id_label: &str,
    count: usize,
    height: f32,
    selected: &mut Option<usize>,
    item: &dyn Fn(usize, &mut dyn std::fmt::Write),
) -> RowsResponse {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let id = ui.id(id_label);
    rows_core(ui, id, id_label, count, m.row_height, height, selected, &mut |ui, i, node, st| {
        let mut text = kit::Buf::<CELL_TEXT_MAX>::new();
        item(i, &mut text);
        let ink = if st.selected { p.text } else { p.text.mix(p.text_muted, 0.2 * (1.0 - st.hover)) };
        kit::text(ui, node, kit::inset_x(m.space_4, m.space_4), text.as_str(), theme.text.small, ink, Align::Start);
        ui.semantic(id.with_index(i as u64), Role::Custom("row"), text.as_str(), SemanticValue::Toggle(st.selected));
    })
}

/// One column of a [`Table`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Column<'a> {
    /// The header text.
    pub title: &'a str,
    /// The starting width (the user can drag it); the last column takes the rest.
    pub width: f32,
    /// Text alignment in the cells.
    pub align: Align,
}

impl<'a> Column<'a> {
    /// A left-aligned column.
    pub const fn new(title: &'a str, width: f32) -> Self {
        Self { title, width, align: Align::Start }
    }

    /// A right-aligned column (numbers).
    #[must_use]
    pub const fn right(mut self) -> Self {
        self.align = Align::End;
        self
    }
}

/// The sort order a table's header asks for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sort {
    /// The column.
    pub column: usize,
    /// Largest first.
    pub descending: bool,
}

/// Column widths of a table, in [`Ui::memory`].
#[derive(Clone, Debug, Default)]
struct TableState {
    widths: Vec<f32>,
}

/// A virtualised table `[UI-07]`: a header (click a title to sort by it, again to reverse; drag
/// the divider after a title to resize the column) over rows of cells written by
/// `cell(row, column, out)`. The caller sorts its data by [`Sort`]. See the module documentation
/// for the keys.
#[derive(Debug)]
pub struct Table<'a> {
    id_label: &'a str,
    columns: &'a [Column<'a>],
    rows: usize,
    height: f32,
    sort: Option<&'a mut Option<Sort>>,
}

impl<'a> Table<'a> {
    /// A table identified by `id_label` with `columns` and `rows` rows, `height` pixels tall.
    pub fn new(id_label: &'a str, columns: &'a [Column<'a>], rows: usize, height: f32) -> Self {
        Self { id_label, columns, rows, height, sort: None }
    }

    /// Lets the header set the sort order.
    #[must_use]
    pub fn sort(mut self, sort: &'a mut Option<Sort>) -> Self {
        self.sort = Some(sort);
        self
    }

    /// Adds the table.
    #[allow(clippy::too_many_lines)]
    pub fn show(
        self,
        ui: &mut Ui,
        selected: &mut Option<usize>,
        cell: &dyn Fn(usize, usize, &mut dyn std::fmt::Write),
    ) -> RowsResponse {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let id = ui.id(self.id_label);
        let n = self.columns.len();
        let mut widths = std::mem::take(&mut ui.memory::<TableState>(id).widths);
        if widths.len() != n {
            widths.clear();
            widths.extend(self.columns.iter().map(|c| if c.width.is_finite() { c.width.max(32.0) } else { 80.0 }));
        }
        ui.begin(Style { gap: 0.0, ..Style::column() });
        // Header.
        let header = ui.begin(Style { height: Dim::Px(m.row_height), shrink: 0.0, ..Style::row() });
        kit::line(
            ui,
            header,
            (Anchor::start(0.0), Anchor::end(-0.5)),
            (Anchor::end(0.0), Anchor::end(-0.5)),
            m.border,
            p.border,
        );
        let sort_now = self.sort.as_deref().copied().flatten();
        let mut sort_to = None;
        for (c, col) in self.columns.iter().enumerate() {
            let last = c + 1 == n;
            let style = if last {
                Style { grow: 1.0, ..Style::default() }
            } else {
                Style { width: Dim::Px(widths[c]), shrink: 0.0, ..Style::default() }
            };
            let hnode = ui.node(style);
            let hid = id.with("header").with_index(c as u64);
            let clickable = self.sort.is_some();
            let hr = ui.interact(hid, hnode, Sense { click: clickable, drag: false, focus: false });
            if hr.clicked {
                sort_to = Some(match sort_now {
                    Some(s) if s.column == c => Sort { column: c, descending: !s.descending },
                    _ => Sort { column: c, descending: false },
                });
            }
            kit::fill(ui, hnode, Area::FULL, p.hover.fade(hr.hover_t * 0.6), 0.0);
            let arrow_w = if sort_now.is_some_and(|s| s.column == c) { 16.0 } else { 0.0 };
            ui.push_clip(hnode, Area::FULL);
            kit::text(
                ui,
                hnode,
                kit::inset_x(m.space_4, m.space_4 + arrow_w),
                col.title,
                theme.text.label,
                p.text_muted,
                col.align,
            );
            if let Some(s) = sort_now.filter(|s| s.column == c) {
                let glyph = if s.descending { crate::icons::CHEVRON_DOWN } else { crate::icons::CHEVRON_UP };
                kit::icon(
                    ui,
                    hnode,
                    Anchor::end(-m.space_4 - 6.0),
                    Anchor::at(0.5, 0.0),
                    glyph,
                    m.icon_sm,
                    p.text_muted,
                );
            }
            ui.pop_clip();
            ui.semantic(hid, Role::Custom("columnheader"), col.title, SemanticValue::None);
            if !last {
                let handle = ui.node(Style { width: Dim::Px(5.0), shrink: 0.0, ..Style::default() });
                let gr = ui.interact(
                    id.with("resize").with_index(c as u64),
                    handle,
                    Sense { click: false, drag: true, focus: false },
                );
                if gr.dragged {
                    widths[c] = (widths[c] + gr.drag_delta.x).clamp(32.0, 2000.0);
                }
                if gr.hovered || gr.dragged {
                    ui.set_cursor(CursorIcon::ResizeHorizontal);
                }
                let t = gr.hover_t.max(gr.press_t);
                kit::line(
                    ui,
                    handle,
                    (Anchor::at(0.5, 0.0), Anchor::start(6.0)),
                    (Anchor::at(0.5, 0.0), Anchor::end(-6.0)),
                    1.0 + t,
                    p.border.mix(p.focus_ring, t),
                );
            }
        }
        ui.end();
        if let (Some(s), Some(dst)) = (sort_to, self.sort) {
            *dst = Some(s);
        }
        let w = &widths;
        let body_h = (self.height - m.row_height).max(m.row_height);
        let out =
            rows_core(ui, id, self.id_label, self.rows, m.row_height, body_h, selected, &mut |ui, i, node, st| {
                let mut x = 0.0;
                let mut first = kit::Buf::<CELL_TEXT_MAX>::new();
                for (c, col) in self.columns.iter().enumerate() {
                    let last = c + 1 == n;
                    let cw = if last { f32::INFINITY } else { w[c] + 5.0 };
                    let mut text = kit::Buf::<CELL_TEXT_MAX>::new();
                    cell(i, c, &mut text);
                    if c == 0 {
                        first = text;
                    }
                    let area = if last {
                        Area {
                            x0: Anchor::start(x),
                            y0: Anchor::start(0.0),
                            x1: Anchor::end(0.0),
                            y1: Anchor::end(0.0),
                        }
                    } else {
                        Area {
                            x0: Anchor::start(x),
                            y0: Anchor::start(0.0),
                            x1: Anchor::start(x + cw),
                            y1: Anchor::end(0.0),
                        }
                    };
                    ui.push_clip(node, area);
                    let inner = Area {
                        x0: Anchor::start(x + m.space_4),
                        x1: if last { Anchor::end(-m.space_4) } else { Anchor::start(x + cw - m.space_4) },
                        ..area
                    };
                    let ink = if st.selected || c == 0 { p.text } else { p.text_muted };
                    kit::text(ui, node, inner, text.as_str(), theme.text.small, ink, col.align);
                    ui.pop_clip();
                    if !last {
                        x += cw;
                    }
                }
                ui.semantic(
                    id.with_index(i as u64),
                    Role::Custom("row"),
                    first.as_str(),
                    SemanticValue::Toggle(st.selected),
                );
            });
        ui.end();
        ui.memory::<TableState>(id).widths = widths;
        out
    }
}

/// One node of a [`TreeView`], in a flattened pre-order list: a node's children follow it with a
/// larger depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TreeItem<'a> {
    /// A stable key (selection and expansion survive reordering).
    pub key: u64,
    /// Its text.
    pub label: &'a str,
    /// Its depth (0 at the top).
    pub depth: u16,
    /// An icon before the text.
    pub icon: Option<char>,
}

/// What a tree view did this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TreeResponse {
    /// The tree as a whole.
    pub response: Response,
    /// The selection changed.
    pub selection_changed: bool,
    /// The key of a node activated (Enter or a double click).
    pub activated: Option<u64>,
    /// The key of a node whose context menu was asked for.
    pub context: Option<u64>,
    /// The key of a node expanded or collapsed.
    pub toggled: Option<u64>,
}

/// A tree view's memory: expanded keys (sorted) and the visible rows (indices into the items).
#[derive(Clone, Debug, Default)]
struct TreeState {
    known: bool,
    expanded: Vec<u64>,
    visible: Vec<u32>,
}

impl TreeState {
    fn is_expanded(&self, key: u64) -> bool {
        self.expanded.binary_search(&key).is_ok()
    }

    fn set_expanded(&mut self, key: u64, on: bool) {
        match (self.expanded.binary_search(&key), on) {
            (Err(at), true) => self.expanded.insert(at, key),
            (Ok(at), false) => {
                self.expanded.remove(at);
            }
            _ => {}
        }
    }
}

/// A tree view with virtualisation `[UI-07]` over a flattened pre-order list of [`TreeItem`]s
/// (an outliner, a region or bone hierarchy).
///
/// Keys (one Tab stop): Up / Down / Page Up / Page Down / Home / End move the selection over the
/// visible rows; Right expands a collapsed node or moves to its first child; Left collapses an
/// expanded node or moves to its parent; Enter activates; Space toggles expansion. Pointer: click
/// selects, a click on the chevron or a double click expands or collapses.
#[derive(Debug)]
pub struct TreeView<'a> {
    id_label: &'a str,
    items: &'a [TreeItem<'a>],
    height: f32,
    expand_depth: u16,
}

impl<'a> TreeView<'a> {
    /// A tree identified by `id_label`, `height` pixels tall.
    pub fn new(id_label: &'a str, items: &'a [TreeItem<'a>], height: f32) -> Self {
        Self { id_label, items, height, expand_depth: 0 }
    }

    /// Nodes shallower than `depth` start expanded (the first time the tree is shown).
    #[must_use]
    pub fn expand_depth(mut self, depth: u16) -> Self {
        self.expand_depth = depth;
        self
    }

    fn has_children(&self, i: usize) -> bool {
        match (self.items.get(i), self.items.get(i + 1)) {
            (Some(a), Some(b)) => b.depth > a.depth,
            _ => false,
        }
    }

    /// Adds the tree; `selected` is the key of the selected node.
    #[allow(clippy::too_many_lines)]
    pub fn show(self, ui: &mut Ui, selected: &mut Option<u64>) -> TreeResponse {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let id = ui.id(self.id_label);
        let items = self.items;
        let mut st = std::mem::take(ui.memory::<TreeState>(id));
        if !st.known {
            st.known = true;
            for (i, it) in items.iter().enumerate() {
                if it.depth < self.expand_depth && self.has_children(i) {
                    st.set_expanded(it.key, true);
                }
            }
        }
        let rebuild = |st: &mut TreeState| {
            st.visible.clear();
            let mut hide_below: Option<u16> = None;
            for (i, it) in items.iter().enumerate() {
                if let Some(d) = hide_below {
                    if it.depth > d {
                        continue;
                    }
                    hide_below = None;
                }
                st.visible.push(u32::try_from(i).unwrap_or(u32::MAX));
                if !st.is_expanded(it.key) {
                    hide_below = Some(it.depth);
                }
            }
        };
        rebuild(&mut st);
        let mut out = TreeResponse::default();
        let row_of = |st: &TreeState, key: Option<u64>| {
            key.and_then(|k| st.visible.iter().position(|&i| items.get(i as usize).is_some_and(|it| it.key == k)))
        };
        // Left / Right before the list takes the arrows.
        let focused = ui.focused() == Some(id);
        if focused && let Some(row) = row_of(&st, *selected) {
            let i = st.visible[row] as usize;
            let key = items[i].key;
            let open = st.is_expanded(key);
            if ui.consume_key(Key::Right, Modifiers::NONE) && self.has_children(i) {
                if open {
                    if let Some(&child) = st.visible.get(row + 1) {
                        *selected = items.get(child as usize).map(|c| c.key);
                    }
                } else {
                    st.set_expanded(key, true);
                    out.toggled = Some(key);
                }
            }
            if ui.consume_key(Key::Left, Modifiers::NONE) {
                if open && self.has_children(i) {
                    st.set_expanded(key, false);
                    out.toggled = Some(key);
                } else {
                    let depth = items[i].depth;
                    if let Some(parent) = items[..i].iter().rev().find(|it| it.depth < depth) {
                        *selected = Some(parent.key);
                    }
                }
            }
            if ui.consume_key(Key::Space, Modifiers::NONE) && self.has_children(i) {
                st.set_expanded(key, !open);
                out.toggled = Some(key);
            }
            if out.toggled.is_some() {
                rebuild(&mut st);
            }
        }
        let before = *selected;
        let mut sel_row = row_of(&st, *selected);
        let visible = std::mem::take(&mut st.visible);
        let mut toggle_key = None;
        let st_ref = &st;
        let rows = rows_core(
            ui,
            id,
            self.id_label,
            visible.len(),
            m.row_height,
            self.height,
            &mut sel_row,
            &mut |ui, row, node, rs| {
                let Some(it) = visible.get(row).and_then(|&i| items.get(i as usize)) else {
                    return;
                };
                let i = visible[row] as usize;
                let indent = f32::from(it.depth.min(64)) * 16.0 + m.space_3;
                if self.has_children(i) {
                    let chev = ui.node(Style {
                        position: Position::Absolute { left: Some(indent), top: Some(0.0), right: None, bottom: None },
                        width: Dim::Px(18.0),
                        height: Dim::Px(m.row_height),
                        ..Style::default()
                    });
                    let cr = ui.interact(
                        id.with("chevron").with_index(it.key),
                        chev,
                        Sense { click: true, drag: false, focus: false },
                    );
                    if cr.clicked {
                        toggle_key = Some(it.key);
                    }
                    let glyph = if st_ref.is_expanded(it.key) {
                        crate::icons::CHEVRON_DOWN
                    } else {
                        crate::icons::CHEVRON_RIGHT
                    };
                    kit::icon_centered(ui, chev, glyph, m.icon_sm, p.text_muted.mix(p.text, cr.hover_t));
                }
                let mut x = indent + 20.0;
                if let Some(g) = it.icon {
                    kit::icon(ui, node, Anchor::start(x + 7.0), Anchor::at(0.5, 0.0), g, m.icon_sm, p.text_muted);
                    x += 20.0;
                }
                let ink = if rs.selected { p.text } else { p.text.mix(p.text_muted, 0.2 * (1.0 - rs.hover)) };
                kit::text(ui, node, kit::inset_x(x, m.space_3), it.label, theme.text.small, ink, Align::Start);
                let value = if self.has_children(i) {
                    SemanticValue::Toggle(st_ref.is_expanded(it.key))
                } else {
                    SemanticValue::None
                };
                ui.semantic(id.with("row").with_index(it.key), Role::Custom("tree.row"), it.label, value);
                ui.semantic_node(node);
            },
        );
        st.visible = visible;
        if let Some(a) = rows.activated
            && let Some(&i) = st.visible.get(a)
            && let Some(it) = items.get(i as usize)
        {
            out.activated = Some(it.key);
            // A double click on a parent also opens or closes it (Enter only activates).
            if ui.pointer_pressed() && self.has_children(i as usize) && toggle_key.is_none() {
                toggle_key = Some(it.key);
            }
        }
        if let Some(k) = toggle_key {
            let open = st.is_expanded(k);
            st.set_expanded(k, !open);
            out.toggled = Some(k);
        }
        if rows.selection_changed || sel_row.is_some() {
            *selected = sel_row.and_then(|r| st.visible.get(r)).and_then(|&i| items.get(i as usize)).map(|it| it.key);
        }
        out.context =
            rows.context.and_then(|r| st.visible.get(r)).and_then(|&i| items.get(i as usize)).map(|it| it.key);
        out.selection_changed = *selected != before || rows.selection_changed;
        out.response = rows.response;
        *ui.memory::<TreeState>(id) = st;
        out
    }
}

/// The zoom and pan of an image view, in [`Ui::memory`].
#[derive(Clone, Copy, Debug, Default)]
struct ViewState {
    known: bool,
    zoom: f32,
    /// The image pixel at the centre of the view.
    center: Point,
}

/// Zoom limits of the image view.
pub const ZOOM_RANGE: (f32, f32) = (0.02, 64.0);

/// An image view with zoom and pan `[UI-07]` (texture previews, the eyedropper's source).
///
/// Pointer: the wheel zooms about the pointer, dragging pans, a double click fits. Keyboard
/// (focused): `+` / `-` zoom, `0` fits, `1` shows pixels 1:1, the arrows pan. The buttons in the
/// corner zoom and fit.
#[derive(Debug)]
pub struct ImageView<'a> {
    id_label: &'a str,
    texture: TextureId,
    image: Size,
    height: f32,
}

impl<'a> ImageView<'a> {
    /// A view of `texture` (an image of `image` pixels), `height` logical pixels tall.
    pub fn new(id_label: &'a str, texture: TextureId, image: Size, height: f32) -> Self {
        Self { id_label, texture, image, height }
    }

    /// Adds the view; returns its response and the current zoom (screen pixels per image pixel).
    #[allow(clippy::too_many_lines)]
    pub fn show(self, ui: &mut Ui) -> (Response, f32) {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let id = ui.id(self.id_label);
        let node = ui.begin(Style { height: Dim::Px(self.height), shrink: 0.0, ..Style::default() });
        let r = ui.interact(id, node, Sense::DRAG);
        let (iw, ih) = (self.image.w.max(1.0), self.image.h.max(1.0));
        let view = r.rect;
        let mut st = *ui.memory::<ViewState>(id);
        let fit = |st: &mut ViewState| {
            let z = if view.w > 0.0 && view.h > 0.0 { (view.w / iw).min(view.h / ih) * 0.95 } else { 1.0 };
            st.zoom = z.clamp(ZOOM_RANGE.0, ZOOM_RANGE.1);
            st.center = Point::new(iw * 0.5, ih * 0.5);
            st.known = view.w > 0.0;
        };
        if !st.known {
            fit(&mut st);
        }
        let zoom_about = |st: &mut ViewState, factor: f32, at: Point| {
            let old = st.zoom;
            let new = (old * factor).clamp(ZOOM_RANGE.0, ZOOM_RANGE.1);
            // Keep the image point under `at` where it is.
            let c = view.center();
            let img = Point::new(st.center.x + (at.x - c.x) / old, st.center.y + (at.y - c.y) / old);
            st.center = Point::new(img.x - (at.x - c.x) / new, img.y - (at.y - c.y) / new);
            st.zoom = new;
        };
        // Two fingers pan and pinch [UI-06]; otherwise the wheel zooms.
        let g = ui.gesture();
        let wheel = ui.wheel();
        if (g.zoom != 1.0 || g.pan != Point::ZERO) && view.contains(g.center) {
            st.center = Point::new(st.center.x - g.pan.x / st.zoom, st.center.y - g.pan.y / st.zoom);
            zoom_about(&mut st, g.zoom, g.center);
        } else if r.hovered
            && wheel.y != 0.0
            && let Some(pos) = ui.pointer()
        {
            zoom_about(&mut st, 1.0015_f32.powf(wheel.y), pos);
        }
        if r.dragged {
            st.center = Point::new(st.center.x - r.drag_delta.x / st.zoom, st.center.y - r.drag_delta.y / st.zoom);
            ui.set_cursor(CursorIcon::Grabbing);
        } else if r.hovered {
            ui.set_cursor(CursorIcon::Grab);
        }
        if r.double_clicked {
            fit(&mut st);
        }
        if r.focused {
            let typed = ui.typed_text().0.chars().next();
            match typed {
                Some('+' | '=') => zoom_about(&mut st, 1.25, view.center()),
                Some('-') => zoom_about(&mut st, 0.8, view.center()),
                Some('0') => fit(&mut st),
                Some('1') => st.zoom = 1.0,
                _ => {}
            }
            let step = 32.0 / st.zoom;
            for (key, d) in [
                (Key::Left, (-step, 0.0)),
                (Key::Right, (step, 0.0)),
                (Key::Up, (0.0, -step)),
                (Key::Down, (0.0, step)),
            ] {
                while ui.consume_key_any(key).is_some() {
                    st.center = Point::new(st.center.x + d.0, st.center.y + d.1);
                }
            }
        }
        st.center = Point::new(st.center.x.clamp(0.0, iw), st.center.y.clamp(0.0, ih));
        // Look: background, the image clipped to the view, the zoom readout.
        kit::frame(ui, node, Area::FULL, p.bg, m.radius_sm, p.border);
        let half = Point::new(view.w * 0.5, view.h * 0.5);
        let x0 = half.x - st.center.x * st.zoom;
        let y0 = half.y - st.center.y * st.zoom;
        ui.push_clip(node, Area::inset(1.0));
        ui.paint(
            node,
            Paint::Image {
                area: Area::offset(x0, y0, iw * st.zoom, ih * st.zoom),
                texture: self.texture,
                uv: [0.0, 0.0, 1.0, 1.0],
                tint: Color::WHITE,
                radius: 0.0,
            },
        );
        ui.pop_clip();
        let pct = kit::fmt64(format_args!("{:.0} %", st.zoom * 100.0));
        let tag =
            Area { x0: Anchor::start(6.0), y0: Anchor::end(-22.0), x1: Anchor::start(70.0), y1: Anchor::end(-6.0) };
        kit::fill(ui, node, tag, p.surface.fade(0.85), 4.0);
        kit::text(ui, node, tag, pct.as_str(), theme.text.tiny, p.text_muted, Align::Center);
        if r.focus_visible {
            kit::ring(ui, node, Area::FULL, m.radius_sm);
        }
        // The corner buttons.
        ui.begin(Style {
            position: Position::Absolute { left: None, top: Some(6.0), right: Some(6.0), bottom: None },
            gap: m.space_1,
            ..Style::row()
        });
        ui.push_id(id);
        let small = m.control_height_sm;
        if IconButton::new(crate::icons::ZOOM_OUT, "Zoom out").size(small).show(ui).clicked {
            zoom_about(&mut st, 0.8, view.center());
        }
        if IconButton::new(crate::icons::ZOOM_IN, "Zoom in").size(small).show(ui).clicked {
            zoom_about(&mut st, 1.25, view.center());
        }
        if IconButton::new(crate::icons::FIT, "Fit").size(small).show(ui).clicked {
            fit(&mut st);
        }
        ui.pop_id();
        ui.end();
        ui.end();
        *ui.memory::<ViewState>(id) = st;
        let value = ui.intern(pct.as_str());
        ui.semantic(id, Role::Image, self.id_label, SemanticValue::Text(value));
        (r, st.zoom)
    }
}

/// Most channels of a histogram.
pub const HISTOGRAM_CHANNELS: usize = 4;

/// A histogram plot `[UI-07]`: up to four channels of bin counts drawn as overlaid translucent
/// bars, scaled to the largest bin (or logarithmically). Bins are merged to at most one bar per
/// two pixels.
#[derive(Clone, Copy, Debug)]
pub struct Histogram<'a> {
    label: &'a str,
    channels: [Option<(&'a [u32], Color)>; HISTOGRAM_CHANNELS],
    height: f32,
    log: bool,
}

impl<'a> Histogram<'a> {
    /// An empty histogram labelled `label`.
    pub fn new(label: &'a str) -> Self {
        Self { label, channels: [None; HISTOGRAM_CHANNELS], height: 90.0, log: false }
    }

    /// Adds a channel (ignored beyond four).
    #[must_use]
    pub fn channel(mut self, bins: &'a [u32], color: Color) -> Self {
        if let Some(slot) = self.channels.iter_mut().find(|c| c.is_none()) {
            *slot = Some((bins, color));
        }
        self
    }

    /// Logarithmic heights.
    #[must_use]
    pub fn log(mut self, log: bool) -> Self {
        self.log = log;
        self
    }

    /// The plot height.
    #[must_use]
    pub fn height(mut self, height: f32) -> Self {
        if height.is_finite() {
            self.height = height.max(24.0);
        }
        self
    }

    /// Adds the plot.
    pub fn show(self, ui: &mut Ui) -> NodeId {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let node = ui.node(Style { height: Dim::Px(self.height), shrink: 0.0, ..Style::default() });
        let id = ui.id(self.label);
        let r = ui.interact(id, node, Sense::HOVER);
        kit::frame(ui, node, Area::FULL, p.bg, m.radius_sm, p.border);
        let width = (r.rect.w - 4.0).max(8.0);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // A few hundred bars.
        let max_bars = ((width / 2.0) as usize).max(1);
        let scale = |v: f32, max: f32| {
            if max <= 0.0 {
                0.0
            } else if self.log {
                (1.0 + v).ln() / (1.0 + max).ln()
            } else {
                v / max
            }
        };
        let mut total = 0_u64;
        for (bins, color) in self.channels.iter().flatten() {
            let n = bins.len();
            if n == 0 {
                continue;
            }
            let group = n.div_ceil(max_bars);
            let bars = n.div_ceil(group);
            #[allow(clippy::cast_precision_loss)]
            let max =
                bins.chunks(group).map(|g| g.iter().map(|&v| u64::from(v)).sum::<u64>()).max().unwrap_or(0) as f32;
            for (k, g) in bins.chunks(group).enumerate() {
                let sum: u64 = g.iter().map(|&v| u64::from(v)).sum();
                total += sum;
                #[allow(clippy::cast_precision_loss)]
                let h = scale(sum as f32, max);
                if h <= 0.0 {
                    continue;
                }
                #[allow(clippy::cast_precision_loss)]
                let (a, b) = (k as f32 / bars as f32, (k + 1) as f32 / bars as f32);
                let area = Area {
                    x0: Anchor { frac: a, px: 2.0 - 4.0 * a },
                    y0: Anchor { frac: 1.0 - h, px: -2.0 + 4.0 * h },
                    x1: Anchor { frac: b, px: 2.0 - 4.0 * b },
                    y1: Anchor::end(-2.0),
                };
                kit::fill(ui, node, area, color.fade(0.55), 0.0);
            }
        }
        let caption = kit::fmt64(format_args!("{} ({total} samples)", self.label));
        ui.semantic(id, Role::Custom("histogram"), caption.as_str(), SemanticValue::None);
        tooltip(ui, &r, &Tip::new(caption.as_str()));
        node
    }
}

/// A waveform plot `[UI-07]` of `values` (a signal over time, for example an expression channel
/// during a sweep) in `range`: one vertical min–max line per pixel column, so any number of
/// samples costs at most one primitive per column.
pub fn waveform(ui: &mut Ui, label: &str, values: &[f32], range: (f32, f32), color: Color, height: f32) -> NodeId {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let height = if height.is_finite() { height.max(24.0) } else { 60.0 };
    let node = ui.node(Style { height: Dim::Px(height), shrink: 0.0, ..Style::default() });
    let id = ui.id(label);
    let r = ui.interact(id, node, Sense::HOVER);
    kit::frame(ui, node, Area::FULL, p.bg, m.radius_sm, p.border);
    let (lo, hi) = if range.0 < range.1 { range } else { (range.1, range.0) };
    let span = (hi - lo).max(f32::EPSILON);
    if lo < 0.0 && hi > 0.0 {
        let z = 1.0 - (0.0 - lo) / span;
        kit::line(
            ui,
            node,
            (Anchor::start(2.0), Anchor::at(z, 0.0)),
            (Anchor::end(-2.0), Anchor::at(z, 0.0)),
            1.0,
            p.border,
        );
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let columns = ((r.rect.w - 4.0).max(8.0) as usize).max(1);
    let n = values.len();
    if n > 0 {
        let per = n.div_ceil(columns);
        let cols = n.div_ceil(per);
        for (k, chunk) in values.chunks(per).enumerate() {
            let (mut a, mut b) = (f32::INFINITY, f32::NEG_INFINITY);
            for v in chunk.iter().copied().filter(|v| v.is_finite()) {
                a = a.min(v);
                b = b.max(v);
            }
            if a > b {
                continue;
            }
            let fy = |v: f32| 1.0 - ((v - lo) / span).clamp(0.0, 1.0);
            #[allow(clippy::cast_precision_loss)]
            let fx = (k as f32 + 0.5) / cols as f32;
            let x = Anchor { frac: fx, px: 2.0 - 4.0 * fx };
            let (ya, yb) = (fy(b), fy(a));
            let y0 = Anchor { frac: ya, px: 2.0 - 4.0 * ya - 0.5 };
            let y1 = Anchor { frac: yb, px: 2.0 - 4.0 * yb + 0.5 };
            kit::line(ui, node, (x, y0), (x, y1), 1.5, color);
        }
    }
    ui.semantic(id, Role::Custom("waveform"), label, SemanticValue::None);
    node
}

/// A timeline strip `[UI-07]`: the expression-sweep scrubber. A ruler from 0 to `duration`
/// seconds, key markers, and the playhead at `time`.
///
/// Pointer: press or drag anywhere to scrub. Keyboard (focused): Left / Right step one frame
/// (Shift: ten), Page Up / Page Down jump to the previous / next key, Home / End to the ends, Space
/// toggles playback when the strip has a play state.
#[derive(Debug)]
pub struct Timeline<'a> {
    id_label: &'a str,
    time: &'a mut f32,
    duration: f32,
    fps: f32,
    keys: &'a [f32],
    playing: Option<&'a mut bool>,
}

impl<'a> Timeline<'a> {
    /// A timeline for `time` in `0..=duration` seconds at `fps` frames per second.
    pub fn new(id_label: &'a str, time: &'a mut f32, duration: f32, fps: f32) -> Self {
        let duration = if duration.is_finite() { duration.max(0.001) } else { 1.0 };
        let fps = if fps.is_finite() && fps > 0.0 { fps } else { 30.0 };
        Self { id_label, time, duration, fps, keys: &[], playing: None }
    }

    /// Key times to mark (seconds, sorted).
    #[must_use]
    pub fn keys(mut self, keys: &'a [f32]) -> Self {
        self.keys = keys;
        self
    }

    /// A play state Space toggles (shown as a play / pause button).
    #[must_use]
    pub fn playing(mut self, playing: &'a mut bool) -> Self {
        self.playing = Some(playing);
        self
    }

    /// Adds the strip.
    #[allow(clippy::too_many_lines)]
    pub fn show(self, ui: &mut Ui) -> SliderResponse {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let id = ui.id(self.id_label);
        let before = *self.time;
        let dur = self.duration;
        ui.begin(Style { gap: m.space_3, align_items: BoxAlign::Center, ..Style::row() });
        if let Some(playing) = self.playing {
            ui.push_id(id);
            let (glyph, tip) = if *playing { (crate::icons::PAUSE, "Pause") } else { (crate::icons::PLAY, "Play") };
            if IconButton::new(glyph, tip).show(ui).clicked {
                *playing = !*playing;
            }
            ui.pop_id();
            if ui.focused() == Some(id) && ui.consume_key(Key::Space, Modifiers::NONE) {
                *playing = !*playing;
            }
        }
        let node = ui.node(Style { height: Dim::Px(44.0), grow: 1.0, min_width: 120.0, ..Style::default() });
        let mut r = ui.interact(id, node, Sense { click: true, drag: true, focus: true });
        let usable = (r.rect.w - 2.0 * TIMELINE_PAD).max(1.0);
        let mut stepped = false;
        if (r.pressed || r.dragged)
            && let Some(pos) = ui.pointer()
        {
            *self.time = ((pos.x - r.rect.x - TIMELINE_PAD) / usable).clamp(0.0, 1.0) * dur;
        }
        if r.focused {
            let frame = 1.0 / self.fps;
            for (key, dir) in [(Key::Left, -1.0), (Key::Right, 1.0)] {
                while let Some(mods) = ui.consume_key_any(key) {
                    *self.time += dir * frame * if mods.shift { 10.0 } else { 1.0 };
                    stepped = true;
                }
            }
            if ui.consume_key(Key::Home, Modifiers::NONE) {
                *self.time = 0.0;
                stepped = true;
            }
            if ui.consume_key(Key::End, Modifiers::NONE) {
                *self.time = dur;
                stepped = true;
            }
            let eps = 0.5 / self.fps;
            if ui.consume_key(Key::PageDown, Modifiers::NONE)
                && let Some(k) = self.keys.iter().copied().find(|k| *k > *self.time + eps)
            {
                *self.time = k;
                stepped = true;
            }
            if ui.consume_key(Key::PageUp, Modifiers::NONE)
                && let Some(k) = self.keys.iter().rev().copied().find(|k| *k < *self.time - eps)
            {
                *self.time = k;
                stepped = true;
            }
        }
        *self.time = if self.time.is_finite() { self.time.clamp(0.0, dur) } else { before };
        r.changed = (*self.time - before).abs() > 0.0;
        let phase = match () {
            () if r.pressed => EditPhase::Begin,
            () if r.held && ui.pointer_released() => EditPhase::Commit,
            () if r.held && r.changed => EditPhase::Update,
            () if stepped && r.changed => EditPhase::Step,
            () => EditPhase::Idle,
        };

        // Look: track, ticks with labels, keys, playhead.
        kit::frame(ui, node, Area::FULL, p.bg, m.radius_sm, p.border);
        let x_of = |t: f32| TIMELINE_PAD + (t / dur).clamp(0.0, 1.0) * usable;
        // A tick spacing of at least 60 px.
        let step = nice_step(60.0 * dur / usable);
        let mut t = 0.0;
        let mut guard = 0;
        while t <= dur + 1e-4 && guard < 200 {
            let x = x_of(t);
            kit::line(
                ui,
                node,
                (Anchor::start(x), Anchor::start(4.0)),
                (Anchor::start(x), Anchor::start(12.0)),
                1.0,
                p.text_faint,
            );
            let label = kit::fmt64(format_args!("{t:.2}"));
            let area = Area {
                x0: Anchor::start(x + 3.0),
                y0: Anchor::start(1.0),
                x1: Anchor::start(x + 60.0),
                y1: Anchor::start(15.0),
            };
            kit::text(ui, node, area, label.as_str(), theme.text.tiny, p.text_faint, Align::Start);
            t += step;
            guard += 1;
        }
        kit::fill(
            ui,
            node,
            Area {
                x0: Anchor::start(TIMELINE_PAD),
                y0: Anchor::start(26.0),
                x1: Anchor::end(-TIMELINE_PAD),
                y1: Anchor::start(30.0),
            },
            p.active,
            2.0,
        );
        for &k in self.keys.iter().filter(|k| k.is_finite()).take(512) {
            let x = x_of(k);
            ui.paint(
                node,
                Paint::Rect {
                    area: Area {
                        x0: Anchor::start(x - 3.5),
                        y0: Anchor::start(24.5),
                        x1: Anchor::start(x + 3.5),
                        y1: Anchor::start(31.5),
                    },
                    fill: p.warning,
                    radius: Corners::all(1.5),
                    border: 0.0,
                    border_color: p.warning,
                },
            );
        }
        let x = x_of(*self.time);
        kit::line(
            ui,
            node,
            (Anchor::start(x), Anchor::start(16.0)),
            (Anchor::start(x), Anchor::end(-3.0)),
            2.0,
            p.accent,
        );
        ui.paint(
            node,
            Paint::Circle { cx: Anchor::start(x), cy: Anchor::start(18.0), radius: 4.0 + r.hover_t, color: p.accent },
        );
        if r.focus_visible {
            kit::ring(ui, node, Area::FULL, m.radius_sm);
        }
        let value_node =
            ui.node(Style { width: Dim::Px(84.0), shrink: 0.0, height: Dim::Px(44.0), ..Style::default() });
        #[allow(clippy::cast_possible_truncation)]
        let frame_no = (*self.time * self.fps).round() as i64;
        let text = kit::fmt64(format_args!("{:.2} s  f{frame_no}", *self.time));
        kit::text(ui, value_node, Area::FULL, text.as_str(), theme.text.mono, p.text, Align::End);
        ui.end();
        ui.semantic(
            id,
            Role::Slider,
            self.id_label,
            SemanticValue::Number { value: f64::from(*self.time), min: 0.0, max: f64::from(dur) },
        );
        SliderResponse { response: r, phase }
    }
}

/// Inset of a timeline's ruler inside its box.
const TIMELINE_PAD: f32 = 10.0;

/// The smallest of 1, 2 or 5 × 10^k that is at least `min_step` (tick spacing).
fn nice_step(min_step: f32) -> f32 {
    if !(min_step.is_finite() && min_step > 0.0) {
        return 1.0;
    }
    let mag = 10_f32.powf(min_step.log10().floor());
    [1.0, 2.0, 5.0, 10.0].into_iter().map(|k| k * mag).find(|s| *s >= min_step).unwrap_or(mag * 10.0)
}

/// The kind and limits of a property, from the parameter registry's typed API `[REG-04]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PropKind<'a> {
    /// A number with a hard range, an optional soft (typical) range and a step.
    Float {
        /// Smallest value.
        min: f32,
        /// Largest value.
        max: f32,
        /// The typical range drawn on the slider.
        soft: Option<(f32, f32)>,
        /// The value Delete restores.
        default: f32,
    },
    /// An integer in a range.
    Int {
        /// Smallest value.
        min: i64,
        /// Largest value.
        max: i64,
    },
    /// On or off.
    Bool,
    /// One of named options.
    Choice(&'a [&'a str]),
    /// A colour.
    Color,
    /// Three numbers.
    Vec3 {
        /// Change per dragged pixel.
        speed: f32,
    },
    /// Text.
    Text {
        /// Longest text.
        max_chars: usize,
    },
}

/// One property shown by [`property_grid`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Property<'a> {
    /// The stable id (also searched).
    pub id: &'a str,
    /// The display name.
    pub label: &'a str,
    /// The section it is listed under.
    pub group: &'a str,
    /// Kind and limits.
    pub kind: PropKind<'a>,
    /// Help text for the tooltip on the name.
    pub help: &'a str,
    /// Semantic tags (searched).
    pub tags: &'a [&'a str],
}

/// A property's current value.
#[derive(Clone, Debug, PartialEq)]
pub enum PropValue {
    /// For [`PropKind::Float`].
    Float(f32),
    /// For [`PropKind::Int`].
    Int(i64),
    /// For [`PropKind::Bool`].
    Bool(bool),
    /// For [`PropKind::Choice`]: the index.
    Choice(usize),
    /// For [`PropKind::Color`].
    Color(Color),
    /// For [`PropKind::Vec3`].
    Vec3([f32; 3]),
    /// For [`PropKind::Text`].
    Text(String),
}

/// An edit made in a property grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PropEdit {
    /// The index of the property (and of its value).
    pub index: usize,
    /// Undo integration.
    pub phase: EditPhase,
}

/// Whether property `p` matches the search text.
fn prop_matches(p: &Property<'_>, filter: &str) -> bool {
    let f = filter.trim();
    f.is_empty()
        || contains_ignore_case(p.label, f)
        || contains_ignore_case(p.id, f)
        || contains_ignore_case(p.group, f)
        || p.tags.iter().any(|t| contains_ignore_case(t, f))
}

/// A property grid `[UI-07]` generated from parameter descriptors (the registry's typed API,
/// `[REG-04]`): the properties matching `filter` (name, id, group or tag), grouped in collapsible
/// sections, each with its name (tooltip: the help text) and the editor its kind calls for. A value
/// whose variant does not match its kind is shown as unavailable rather than edited. Returns the
/// edit made this frame, if any.
pub fn property_grid(
    ui: &mut Ui,
    id_label: &str,
    props: &[Property<'_>],
    values: &mut [PropValue],
    filter: &str,
) -> Option<PropEdit> {
    let theme = *ui.theme();
    let m = theme.metrics;
    let id = ui.id(id_label);
    let mut edit = None;
    ui.begin(Style { gap: m.space_2, ..Style::column() });
    ui.push_id(id);
    let mut start = 0;
    while start < props.len() {
        let group = props[start].group;
        let end = props[start..].iter().position(|p| p.group != group).map_or(props.len(), |k| start + k);
        let any = props[start..end].iter().any(|p| prop_matches(p, filter));
        if any {
            let title = if group.is_empty() { "General" } else { group };
            Collapsible::new(title).default_open(true).show(ui, |ui| {
                for (i, p) in props.iter().enumerate().take(end).skip(start) {
                    if !prop_matches(p, filter) {
                        continue;
                    }
                    if let Some(v) = values.get_mut(i)
                        && let Some(phase) = property_row(ui, p, v)
                    {
                        edit = Some(PropEdit { index: i, phase });
                    }
                }
            });
        }
        start = end;
    }
    if edit.is_none() && !props.iter().any(|p| prop_matches(p, filter)) {
        let n = ui.node(Style { height: Dim::Px(m.row_height), ..Style::default() });
        let p = theme.palette;
        kit::text(ui, n, Area::inset(m.space_3), "No properties match", theme.text.small, p.text_faint, Align::Start);
    }
    ui.pop_id();
    ui.end();
    ui.semantic(id, Role::Group, id_label, SemanticValue::None);
    edit
}

/// One row of a property grid; returns the phase when the value changed (or a gesture ended).
fn property_row(ui: &mut Ui, prop: &Property<'_>, value: &mut PropValue) -> Option<EditPhase> {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    ui.begin(Style { gap: m.space_3, align_items: BoxAlign::Center, min_height: m.control_height, ..Style::row() });
    let name =
        ui.node(Style { width: Dim::Px(120.0), height: Dim::Px(m.control_height), shrink: 0.0, ..Style::default() });
    let nid = ui.id(prop.id).with("name");
    let nr = ui.interact(nid, name, Sense::HOVER);
    kit::text(ui, name, Area::FULL, prop.label, theme.text.small, p.text_muted, Align::Start);
    if !prop.help.is_empty() {
        tooltip(ui, &nr, &Tip::new(prop.label).text(prop.help));
    }
    ui.push_id(ui.id(prop.id));
    let mut phase = None;
    let active = |r: &SliderResponse| (r.response.changed || r.phase != EditPhase::Idle).then_some(r.phase);
    match (prop.kind, value) {
        (PropKind::Float { min, max, soft, default }, PropValue::Float(v)) => {
            let mut s = Slider::new(prop.label, v, min, max).label_width(0.0).default_value(default);
            if let Some((a, b)) = soft {
                s = s.soft_range(a, b);
            }
            phase = active(&s.show(ui));
        }
        (PropKind::Int { min, max }, PropValue::Int(v)) => {
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)] // UI integers are small.
            let mut f = *v as f32;
            #[allow(clippy::cast_precision_loss)]
            let r = DragValue::new(prop.label, &mut f).range(min as f32, max as f32).speed(0.1).decimals(0).show(ui);
            #[allow(clippy::cast_possible_truncation)]
            {
                *v = (f.round() as i64).clamp(min, max);
            }
            phase = active(&r);
        }
        (PropKind::Bool, PropValue::Bool(b)) => {
            if checkbox_unlabelled(ui, prop.label, b).changed {
                phase = Some(EditPhase::Step);
            }
        }
        (PropKind::Choice(options), PropValue::Choice(c)) => {
            let r = crate::popup::Dropdown::new(prop.label, c, options).show(ui);
            if r.changed {
                phase = Some(EditPhase::Step);
            }
        }
        (PropKind::Color, PropValue::Color(c)) => {
            let r: PickerResponse = color_edit(ui, prop.label, c, None, &[]);
            if r.changed || r.phase != EditPhase::Idle {
                phase = Some(r.phase);
            }
        }
        (PropKind::Vec3 { speed }, PropValue::Vec3(v)) => {
            ui.begin(Style { grow: 1.0, ..Style::row() });
            let r = vector_edit_cells(ui, prop.label, v, speed);
            ui.end();
            phase = active(&r);
        }
        (PropKind::Text { max_chars }, PropValue::Text(t)) => {
            let r = TextField::new(prop.label, t).max_chars(max_chars).show(ui);
            if r.response.changed {
                phase = Some(EditPhase::Step);
            }
        }
        _ => {
            let n = ui.node(Style { grow: 1.0, height: Dim::Px(m.control_height), ..Style::default() });
            kit::text(ui, n, Area::FULL, "Unavailable", theme.text.small, p.text_faint, Align::Start);
            ui.semantic(ui.id("mismatch"), Role::Label, "Unavailable", SemanticValue::None);
        }
    }
    ui.pop_id();
    ui.end();
    phase
}

/// The three cells of a vector without the vector editor's label column.
fn vector_edit_cells(ui: &mut Ui, label: &str, v: &mut [f32; 3], speed: f32) -> SliderResponse {
    let m = ui.theme().metrics;
    let mut out = SliderResponse::default();
    ui.begin(Style { grow: 1.0, gap: m.space_2, ..Style::row() });
    for (i, value) in v.iter_mut().enumerate() {
        let (tag, color) = match i {
            0 => ("X", ui.theme().palette.danger),
            1 => ("Y", ui.theme().palette.positive),
            _ => ("Z", ui.theme().palette.link),
        };
        let cell = kit::fmt64(format_args!("{label} {tag}"));
        let r = DragValue::new(cell.as_str(), value).speed(speed).decimals(3).prefix(tag, color).show(ui);
        out.response.changed |= r.response.changed;
        if out.phase == EditPhase::Idle {
            out.phase = r.phase;
        }
    }
    ui.end();
    out
}

/// The severity of a console line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    /// Diagnostics.
    Debug,
    /// Information.
    Info,
    /// Something to look at.
    Warn,
    /// Something failed.
    Error,
}

impl LogLevel {
    /// All levels, lowest first.
    pub const ALL: [Self; 4] = [Self::Debug, Self::Info, Self::Warn, Self::Error];

    /// Its short name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Debug => "Debug",
            Self::Info => "Info",
            Self::Warn => "Warning",
            Self::Error => "Error",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Debug => 0,
            Self::Info => 1,
            Self::Warn => 2,
            Self::Error => 3,
        }
    }
}

/// One console line.
#[derive(Clone, Debug, PartialEq)]
pub struct LogEntry {
    /// Severity.
    pub level: LogLevel,
    /// Seconds since the session started.
    pub time: f64,
    /// The text (one line; line breaks are shown as spaces).
    pub text: String,
}

/// The console log panel `[UI-07]`: a bounded log the application appends to, shown with level
/// filters, a search box, a clear button and a virtualised list that follows new lines while it is
/// scrolled to the bottom. Ctrl+C copies the selected line.
#[derive(Clone, Debug)]
pub struct ConsoleLog {
    entries: VecDeque<LogEntry>,
    shown: Vec<u32>,
    levels: [bool; 4],
    search: String,
    selected: Option<usize>,
    follow: bool,
    seen_len: usize,
    dropped: u64,
}

impl Default for ConsoleLog {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsoleLog {
    /// Most lines kept; the oldest go first.
    pub const MAX: usize = 5000;
    /// The longest line kept (bytes, cut at a character boundary).
    pub const MAX_LINE: usize = 2000;

    /// An empty log showing every level but Debug.
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            shown: Vec::new(),
            levels: [false, true, true, true],
            search: String::new(),
            selected: None,
            follow: true,
            seen_len: 0,
            dropped: 0,
        }
    }

    /// Appends a line.
    pub fn push(&mut self, level: LogLevel, time: f64, text: &str) {
        if self.entries.len() >= Self::MAX {
            self.entries.pop_front();
            self.dropped += 1;
            self.selected = None;
        }
        let mut cut = text.len().min(Self::MAX_LINE);
        while !text.is_char_boundary(cut) {
            cut -= 1;
        }
        let clean: String = text[..cut].chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
        self.entries.push_back(LogEntry { level, time, text: clean });
    }

    /// Removes every line.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.shown.clear();
        self.selected = None;
    }

    /// How many lines are kept.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether it is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Lines dropped because the log was full.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// The lines, oldest first.
    pub fn entries(&self) -> impl Iterator<Item = &LogEntry> {
        self.entries.iter()
    }

    /// Draws the panel, `height` pixels tall.
    #[allow(clippy::too_many_lines)]
    pub fn show(&mut self, ui: &mut Ui, id_label: &str, height: f32) {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let id = ui.id(id_label);
        ui.begin(Style { gap: m.space_3, ..Style::column() });
        ui.push_id(id);
        ui.begin(Style { gap: m.space_2, align_items: BoxAlign::Center, ..Style::row() });
        for level in LogLevel::ALL {
            let on = &mut self.levels[level.index()];
            let w = ui.measure_text(level.name(), theme.text.small).w + 2.0 * m.space_4;
            let node = ui.node(Style {
                width: Dim::Px(w),
                height: Dim::Px(m.control_height_sm),
                shrink: 0.0,
                ..Style::default()
            });
            let r = ui.interact(ui.id(level.name()), node, Sense::CLICK);
            if r.clicked {
                *on = !*on;
                self.selected = None;
            }
            let ink = level_color(ui, level);
            let fill = if *on { ink.fade(0.18) } else { p.hover.fade(r.hover_t) };
            kit::frame(
                ui,
                node,
                Area::FULL,
                fill,
                m.control_height_sm * 0.5,
                if *on { ink.fade(0.6) } else { p.border },
            );
            kit::text(
                ui,
                node,
                Area::FULL,
                level.name(),
                theme.text.small,
                if *on { p.text } else { p.text_muted },
                Align::Center,
            );
            if r.focus_visible {
                kit::ring(ui, node, Area::FULL, m.control_height_sm * 0.5);
            }
            ui.semantic(r.id, Role::Toggle, level.name(), SemanticValue::Toggle(*on));
        }
        ui.begin(Style { grow: 1.0, ..Style::row() });
        let search = search_box(ui, "Filter", &mut self.search, "Filter lines");
        ui.end();
        if search.response.changed {
            self.selected = None;
        }
        if IconButton::new(crate::icons::TRASH, "Clear the console").show(ui).clicked {
            self.clear();
        }
        ui.end();
        // The lines shown.
        self.shown.clear();
        for (i, e) in self.entries.iter().enumerate() {
            if self.levels[e.level.index()] && contains_ignore_case(&e.text, self.search.trim()) {
                self.shown.push(u32::try_from(i).unwrap_or(u32::MAX));
            }
        }
        let list_id = ui.id("lines");
        let scroll_id = list_id.with("scroll");
        let (offset, view) = ui.scroll_view(scroll_id);
        let row_h = 22.0;
        #[allow(clippy::cast_precision_loss)]
        let content = self.shown.len() as f32 * row_h;
        let at_bottom = view <= 0.0 || offset + view >= content - row_h - 1.0;
        let grew = self.entries.len() != self.seen_len;
        self.seen_len = self.entries.len();
        let shown = std::mem::take(&mut self.shown);
        let entries = &self.entries;
        let mut sel = self.selected;
        let rows = rows_core(ui, list_id, id_label, shown.len(), row_h, height, &mut sel, &mut |ui, row, node, rs| {
            let Some(e) = shown.get(row).and_then(|&i| entries.get(i as usize)) else {
                return;
            };
            let ink = level_color(ui, e.level);
            let stamp = kit::fmt64(format_args!("{:>8.2}", e.time));
            kit::text(
                ui,
                node,
                kit::at(m.space_3, 0.0, 64.0, row_h),
                stamp.as_str(),
                theme.text.mono,
                p.text_faint,
                Align::End,
            );
            kit::fill(ui, node, kit::at(76.0, 7.0, 3.0, row_h - 14.0), ink, 1.5);
            let text_ink = if e.level >= LogLevel::Warn || rs.selected { ink.mix(p.text, 0.35) } else { p.text };
            ui.push_clip(node, Area::FULL);
            kit::text(ui, node, kit::inset_x(86.0, m.space_3), &e.text, theme.text.mono, text_ink, Align::Start);
            ui.pop_clip();
            ui.semantic(list_id.with_index(row as u64), Role::Custom("logline"), &e.text, SemanticValue::None);
        });
        let last_row = shown.len().checked_sub(1);
        self.shown = shown;
        self.selected = sel;
        // Follow new lines until the user scrolls up or selects an earlier line; scrolling back to
        // the bottom (or End) follows again.
        let over = ui.pointer().is_some_and(|q| rows.response.rect.contains(q));
        let wheel = ui.wheel().y;
        if (over && wheel > 0.0) || (rows.selection_changed && self.selected != last_row) {
            self.follow = false;
        } else if (over && wheel < 0.0 && at_bottom) || (rows.selection_changed && self.selected == last_row) {
            self.follow = true;
        }
        if self.follow && (grew || !at_bottom) {
            ui.scroll_to(scroll_id, f32::MAX);
        }
        if rows.response.focused
            && ui.consume_key(Key::Letter('C'), Modifiers::CTRL)
            && let Some(e) = self.selected.and_then(|s| self.shown.get(s)).and_then(|&i| self.entries.get(i as usize))
        {
            let text = e.text.clone();
            ui.copy(&text);
        }
        ui.pop_id();
        ui.end();
    }
}

/// The colour of a log level.
fn level_color(ui: &Ui, level: LogLevel) -> Color {
    let p = ui.theme().palette;
    match level {
        LogLevel::Debug => p.text_faint,
        LogLevel::Info => p.link,
        LogLevel::Warn => p.warning,
        LogLevel::Error => p.danger,
    }
}

/// The keyboard-shortcut overlay `[UI-07]` `[UI-06]`: a modal listing every binding of `keymap`
/// grouped by the action's prefix (`edit.undo` under "edit"), filtered by `filter` (typed into its
/// search box). `describe(action)` gives the readable name of an action. Returns the dialog's
/// outcome (`close` on Escape, the close button or a click outside).
pub fn shortcut_overlay(
    ui: &mut Ui,
    keymap: &Keymap,
    filter: &mut String,
    describe: &dyn Fn(&str) -> &str,
) -> ModalOutcome {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let ((), out) = Modal::new("Keyboard shortcuts", 560.0, 520.0).show(ui, |ui| {
        search_box(ui, "Search shortcuts", filter, "Search actions");
        let list_id = ui.id("shortcut list");
        ui.begin_scroll(list_id, Style { grow: 1.0, gap: m.space_1, ..Style::column() });
        let bindings = keymap.bindings();
        let mut last_group = "";
        let mut any = false;
        for (k, (action, shortcut)) in bindings.iter().enumerate() {
            let name = describe(action);
            let f = filter.trim();
            if !(contains_ignore_case(name, f) || contains_ignore_case(action, f)) {
                continue;
            }
            any = true;
            let group = action.split_once('.').map_or("", |(g, _)| g);
            if group != last_group {
                last_group = group;
                let h = ui.node(Style { height: Dim::Px(m.row_height + m.space_3), shrink: 0.0, ..Style::default() });
                let title = if group.is_empty() { "General" } else { group };
                kit::text(
                    ui,
                    h,
                    Area { y0: Anchor::start(m.space_3), ..Area::FULL },
                    title,
                    theme.text.label,
                    p.text_muted,
                    Align::Start,
                );
                ui.semantic(list_id.with(group), Role::Label, title, SemanticValue::None);
            }
            let row = ui.node(Style { height: Dim::Px(m.row_height), shrink: 0.0, ..Style::default() });
            let mut keys = kit::Buf::<48>::new();
            let _ = write!(keys, "{shortcut}");
            let kw = ui.measure_text(keys.as_str(), theme.text.mono).w + 2.0 * m.space_3;
            kit::text(ui, row, kit::inset_x(m.space_4, kw + m.space_4), name, theme.text.small, p.text, Align::Start);
            let chip = Area {
                x0: Anchor::end(-kw),
                y0: Anchor::at(0.5, -10.0),
                x1: Anchor::end(0.0),
                y1: Anchor::at(0.5, 10.0),
            };
            kit::frame(ui, row, chip, p.surface, 4.0, p.border);
            kit::text(ui, row, chip, keys.as_str(), theme.text.mono, p.text_muted, Align::Center);
            let value = ui.intern(keys.as_str());
            ui.semantic(list_id.with_index(k as u64), Role::Label, name, SemanticValue::Text(value));
        }
        if !any {
            let n = ui.node(Style { height: Dim::Px(m.row_height), ..Style::default() });
            kit::text(
                ui,
                n,
                Area::inset(m.space_3),
                "No shortcuts match",
                theme.text.small,
                p.text_faint,
                Align::Start,
            );
        }
        ui.end_scroll(list_id);
    });
    out
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::input::{Event, PointerButton, Shortcut};
    use crate::tests::Driver;
    use crate::theme::Theme;

    const COLUMNS: [Column<'static>; 3] =
        [Column::new("Region", 120.0), Column::new("Weight", 80.0).right(), Column::new("Notes", 100.0)];

    fn tree_items() -> Vec<TreeItem<'static>> {
        let it = |key, label, depth| TreeItem { key, label, depth, icon: Some(crate::icons::BOX) };
        vec![
            it(1, "Head", 0),
            it(2, "Face", 1),
            it(3, "Cheek", 2),
            it(4, "Chin", 2),
            it(5, "Scalp", 1),
            it(6, "Torso", 0),
            it(7, "Chest", 1),
        ]
    }

    fn props() -> [Property<'static>; 8] {
        let p = |id, label, group, kind| Property { id, label, group, kind, help: "", tags: &[] };
        [
            Property {
                help: "Opens the jaw.",
                tags: &["mouth"],
                ..p(
                    "jaw_open",
                    "Jaw open",
                    "Face",
                    PropKind::Float { min: 0.0, max: 1.0, soft: Some((0.0, 0.8)), default: 0.0 },
                )
            },
            p("seed", "Seed", "Face", PropKind::Int { min: 0, max: 100 }),
            p("freckles", "Freckles", "Skin", PropKind::Bool),
            p("model", "Model", "Skin", PropKind::Choice(&["Default", "Subsurface", "Unlit"])),
            p("tint", "Tint", "Skin", PropKind::Color),
            p("offset", "Offset", "Skin", PropKind::Vec3 { speed: 0.01 }),
            p("note", "Note", "Other", PropKind::Text { max_chars: 40 }),
            p("broken", "Broken", "Other", PropKind::Bool),
        ]
    }

    fn values() -> Vec<PropValue> {
        vec![
            PropValue::Float(0.25),
            PropValue::Int(7),
            PropValue::Bool(false),
            PropValue::Choice(1),
            PropValue::Color(Color::hex(0xC68863)),
            PropValue::Vec3([0.0, 0.1, 0.0]),
            PropValue::Text("left side".into()),
            PropValue::Float(3.0),
        ]
    }

    /// The state the views gallery edits.
    struct Model {
        rows: usize,
        table_sel: Option<usize>,
        sort: Option<Sort>,
        activated: Option<usize>,
        list_sel: Option<usize>,
        tree: Vec<TreeItem<'static>>,
        tree_sel: Option<u64>,
        tree_out: TreeResponse,
        time: f32,
        playing: bool,
        bins: [Vec<u32>; 3],
        wave: Vec<f32>,
        values: Vec<PropValue>,
        filter: String,
        edits: Vec<PropEdit>,
        console: ConsoleLog,
        texture: TextureId,
        zoom: f32,
    }

    impl Model {
        fn new() -> Self {
            #[allow(clippy::cast_precision_loss)]
            let bins = [
                (0..256).map(|i| ((i as f32 / 40.0).sin().abs() * 900.0) as u32).collect(),
                (0..256)
                    .map(|i| (((i as f32 - 128.0) / 50.0).powi(2).mul_add(-300.0, 800.0)).max(0.0) as u32)
                    .collect(),
                (0..256).map(|i| (i * 3) as u32).collect(),
            ];
            #[allow(clippy::cast_precision_loss)]
            let wave = (0..20_000).map(|i| (i as f32 / 900.0).sin() * (i as f32 / 20_000.0)).collect();
            let mut console = ConsoleLog::new();
            console.push(LogLevel::Info, 0.12, "Loaded template 'adult_a'");
            console.push(LogLevel::Warn, 1.5, "Texture 'skin_albedo' is 8-bit; 16-bit is preferred");
            console.push(LogLevel::Error, 2.25, "Could not read 'hair.groom': the file ends early");
            console.push(LogLevel::Debug, 2.5, "Frame 120 took 6.1 ms");
            Self {
                rows: 2000,
                table_sel: None,
                sort: None,
                activated: None,
                list_sel: None,
                tree: tree_items(),
                tree_sel: None,
                tree_out: TreeResponse::default(),
                time: 0.5,
                playing: false,
                bins,
                wave,
                values: values(),
                filter: String::new(),
                edits: Vec::new(),
                console,
                texture: TextureId(1),
                zoom: 1.0,
            }
        }
    }

    const KEYS: [f32; 3] = [0.25, 1.0, 1.75];

    fn gallery(ui: &mut Ui, model: &mut Model) {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let root = ui.begin(Style {
            width: Dim::Percent(100.0),
            height: Dim::Percent(100.0),
            padding: Insets::all(m.space_6),
            gap: m.space_6,
            ..Style::row()
        });
        kit::fill(ui, root, Area::FULL, p.panel, 0.0);
        ui.begin(Style { width: Dim::Px(420.0), gap: m.space_4, shrink: 0.0, ..Style::column() });
        let rows = model.rows;
        let out = Table::new("Regions", &COLUMNS, rows, 190.0).sort(&mut model.sort).show(
            ui,
            &mut model.table_sel,
            &|r, c, out| {
                let _ = match c {
                    0 => write!(out, "Region {r}"),
                    1 => write!(out, "{:.2}", (r % 100) as f32 / 100.0),
                    _ => write!(out, "{}", if r % 3 == 0 { "mirrored" } else { "" }),
                };
            },
        );
        if out.activated.is_some() {
            model.activated = out.activated;
        }
        ui.begin(Style { gap: m.space_4, ..Style::row() });
        ui.begin(Style { width: Dim::Px(200.0), shrink: 0.0, ..Style::column() });
        model.tree_out = TreeView::new("Outliner", &model.tree, 170.0).expand_depth(1).show(ui, &mut model.tree_sel);
        ui.end();
        ui.begin(Style { grow: 1.0, ..Style::column() });
        list_view(ui, "Takes", 50, 170.0, &mut model.list_sel, &|i, out| {
            let _ = write!(out, "Take {:02}", i + 1);
        });
        ui.end();
        ui.end();
        Timeline::new("Sweep", &mut model.time, 2.0, 30.0).keys(&KEYS).playing(&mut model.playing).show(ui);
        ui.begin(Style { gap: m.space_4, ..Style::row() });
        ui.begin(Style { grow: 1.0, ..Style::column() });
        Histogram::new("Luminance")
            .channel(&model.bins[0], Color::hex(0xF06A6A))
            .channel(&model.bins[1], Color::hex(0x6ACF8F))
            .channel(&model.bins[2], Color::hex(0x6A9CF0))
            .height(80.0)
            .show(ui);
        ui.end();
        ui.begin(Style { grow: 1.0, ..Style::column() });
        waveform(ui, "Jaw channel", &model.wave, (-1.0, 1.0), p.accent, 80.0);
        ui.end();
        ui.end();
        ui.end();
        ui.begin(Style { grow: 1.0, gap: m.space_4, ..Style::column() });
        let (_, zoom) = ImageView::new("Albedo", model.texture, Size::new(64.0, 48.0), 150.0).show(ui);
        model.zoom = zoom;
        if let Some(e) = property_grid(ui, "Properties", &props(), &mut model.values, &model.filter) {
            model.edits.push(e);
        }
        model.console.show(ui, "Console", 110.0);
        ui.end();
        ui.end();
    }

    fn setup(theme: &Theme) -> Driver {
        Driver::new(&theme.with_reduced_motion(true), 960.0, 760.0)
    }

    fn focus(d: &mut Driver, build: &mut dyn FnMut(&mut Ui), label: &str) {
        for _ in 0..80 {
            if d.focused_label() == label {
                return;
            }
            d.key(Key::Tab, Modifiers::NONE, build);
        }
        panic!("{label} never got the focus");
    }

    fn drag(d: &mut Driver, build: &mut dyn FnMut(&mut Ui), from: Point, to: Point) {
        d.frame(vec![Event::PointerMoved(from)], build);
        d.frame(vec![Event::Button { button: PointerButton::Primary, pressed: true, pos: from }], build);
        d.frame(vec![Event::PointerMoved(Point::new(f32::midpoint(from.x, to.x), f32::midpoint(from.y, to.y)))], build);
        d.frame(vec![Event::PointerMoved(to)], build);
        d.frame(vec![Event::Button { button: PointerButton::Primary, pressed: false, pos: to }], build);
    }

    #[test]
    fn ui_07_table_and_list_select_scroll_sort_and_resize() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        // Only the rows in view exist.
        let rows = d.ui.semantics().iter().filter(|s| s.role == Role::Custom("row")).count();
        assert!(rows < 30, "{rows} rows built for 2000 + 50");
        d.click(d.rect_of("Region 3").center(), &mut build);
        assert_eq!(model.borrow().table_sel, Some(3));
        assert_eq!(d.focused_label(), "Regions");
        d.key(Key::Down, Modifiers::NONE, &mut build);
        d.key(Key::End, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().table_sel, Some(1999));
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Region 1999").is_some(), "the selection is scrolled into view");
        d.key(Key::PageUp, Modifiers::NONE, &mut build);
        let s = model.borrow().table_sel.unwrap_or(0);
        assert!((1985..1999).contains(&s), "{s}");
        d.key(Key::Enter, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().activated, Some(s));
        // The header sorts, the divider resizes.
        d.click(d.rect_of("Weight").center(), &mut build);
        assert_eq!(model.borrow().sort, Some(Sort { column: 1, descending: false }));
        d.click(d.rect_of("Weight").center(), &mut build);
        assert_eq!(model.borrow().sort, Some(Sort { column: 1, descending: true }));
        let w0 = d.rect_of("Weight").x;
        let region = d.rect_of("Region");
        let handle = Point::new(region.right() + 2.5, region.center().y);
        drag(&mut d, &mut build, handle, Point::new(handle.x + 40.0, handle.y));
        d.frame(vec![], &mut build);
        assert!((d.rect_of("Weight").x - w0 - 40.0).abs() < 1.5, "{} vs {w0}", d.rect_of("Weight").x);
        // The list: Home and Down.
        focus(&mut d, &mut build, "Takes");
        d.key(Key::Home, Modifiers::NONE, &mut build);
        d.key(Key::Down, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().list_sel, Some(1));
        assert!(d.semantic("Take 02").is_some_and(|s| s.value == SemanticValue::Toggle(true)));
    }

    /// Whether the tree shows a row labelled `label`.
    fn tree_row(d: &Driver, label: &str) -> bool {
        d.ui.semantics().iter().any(|s| s.role == Role::Custom("tree.row") && d.ui.text(s.label) == label)
    }

    #[test]
    fn ui_07_tree_view_expands_collapses_and_navigates() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        // Depth-0 nodes start expanded: Head, Face, Scalp, Torso, Chest; Face is closed.
        assert!(tree_row(&d, "Face") && tree_row(&d, "Chest"));
        assert!(!tree_row(&d, "Cheek"));
        focus(&mut d, &mut build, "Outliner");
        d.key(Key::Home, Modifiers::NONE, &mut build);
        d.key(Key::Down, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().tree_sel, Some(2));
        d.key(Key::Right, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().tree_out.toggled, Some(2));
        d.frame(vec![], &mut build);
        assert!(tree_row(&d, "Cheek"), "expanded");
        d.key(Key::Right, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().tree_sel, Some(3), "to the first child");
        d.key(Key::Left, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().tree_sel, Some(2), "to the parent");
        d.key(Key::Left, Modifiers::NONE, &mut build);
        d.frame(vec![], &mut build);
        assert!(!tree_row(&d, "Cheek"), "collapsed");
        d.key(Key::Enter, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().tree_out.activated, Some(2));
        // The chevron of Head collapses it; a double click on Torso does too.
        let head = d.rect_of("Head");
        d.click(Point::new(head.x + 1.0 * 16.0 - 5.0 + 9.0, head.center().y), &mut build);
        d.frame(vec![], &mut build);
        assert!(!tree_row(&d, "Face"), "Head collapsed");
        let torso = d.rect_of("Torso").center();
        d.click(torso, &mut build);
        d.frame(vec![Event::Button { button: PointerButton::Primary, pressed: true, pos: torso }], &mut build);
        d.frame(vec![Event::Button { button: PointerButton::Primary, pressed: false, pos: torso }], &mut build);
        d.frame(vec![], &mut build);
        assert!(!tree_row(&d, "Chest"), "Torso collapsed by a double click");
    }

    #[test]
    fn ui_07_image_view_zooms_about_the_pointer_pans_and_fits() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        let fit = model.borrow().zoom;
        let view = d.rect_of("Albedo");
        assert!(fit > 1.0 && fit * 48.0 <= view.h, "fits: {fit}");
        let at = Point::new(view.x + view.w * 0.3, view.y + view.h * 0.6);
        d.frame(vec![Event::PointerMoved(at)], &mut build);
        d.frame(vec![Event::Wheel { delta: Point::new(0.0, 200.0), precise: true }], &mut build);
        let z = model.borrow().zoom;
        assert!(z > fit * 1.2, "{z} vs {fit}");
        // Panning by a drag, then '0' fits again and '1' is 1:1.
        drag(&mut d, &mut build, at, Point::new(at.x + 30.0, at.y));
        focus(&mut d, &mut build, "Albedo");
        d.frame(vec![Event::Text('1')], &mut build);
        assert_eq!(model.borrow().zoom, 1.0);
        d.frame(vec![Event::Text('0')], &mut build);
        assert!((model.borrow().zoom - fit).abs() < 1e-4);
        d.click(d.rect_of("Zoom in").center(), &mut build);
        assert!((model.borrow().zoom - fit * 1.25).abs() < 1e-3);
    }

    #[test]
    fn ui_07_timeline_scrubs_steps_jumps_and_plays() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        let r = d.rect_of("Sweep");
        d.click(Point::new(r.x + TIMELINE_PAD + (r.w - 2.0 * TIMELINE_PAD) * 0.75, r.center().y), &mut build);
        assert!((model.borrow().time - 1.5).abs() < 0.01, "{}", model.borrow().time);
        focus(&mut d, &mut build, "Sweep");
        d.key(Key::Right, Modifiers::NONE, &mut build);
        assert!((model.borrow().time - (1.5 + 1.0 / 30.0)).abs() < 0.011);
        d.key(Key::PageDown, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().time, 1.75);
        d.key(Key::PageUp, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().time, 1.0);
        d.key(Key::Space, Modifiers::NONE, &mut build);
        assert!(model.borrow().playing);
        d.key(Key::End, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().time, 2.0);
        assert!((nice_step(0.07) - 0.1).abs() < 1e-6);
        assert!((nice_step(0.3) - 0.5).abs() < 1e-6);
        assert_eq!(nice_step(-1.0), 1.0);
    }

    #[test]
    fn ui_07_property_grid_edits_filters_and_survives_mismatches() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        for label in ["Jaw open", "Seed", "Freckles", "Model", "Tint", "Offset X", "Note"] {
            assert!(d.semantic(label).is_some(), "{label}");
        }
        assert!(d.semantic("Unavailable").is_some(), "a float value under a bool kind is not edited");
        d.click(d.rect_of("Freckles").center(), &mut build);
        assert_eq!(model.borrow().values[2], PropValue::Bool(true));
        assert_eq!(model.borrow().edits.last(), Some(&PropEdit { index: 2, phase: EditPhase::Step }));
        focus(&mut d, &mut build, "Jaw open");
        d.key(Key::End, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().values[0], PropValue::Float(1.0));
        assert_eq!(model.borrow().edits.last().map(|e| e.index), Some(0));
        // Filtering by a tag keeps only the match (and its section).
        model.borrow_mut().filter = "mouth".into();
        d.frame(vec![], &mut build);
        assert!(d.semantic("Jaw open").is_some() && d.semantic("Seed").is_none() && d.semantic("Skin").is_none());
        model.borrow_mut().filter = "zzz".into();
        d.frame(vec![], &mut build);
        assert!(d.semantic("No properties match").is_none(), "(painted text, not a widget)");
        assert!(d.semantic("Jaw open").is_none());
    }

    #[test]
    fn ui_07_console_filters_follows_copies_and_is_bounded() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Could not read 'hair.groom': the file ends early").is_some());
        assert!(d.semantic("Frame 120 took 6.1 ms").is_none(), "Debug is off by default");
        for i in 0..200 {
            model.borrow_mut().console.push(LogLevel::Info, f64::from(i), &format!("line {i}\nwith a break"));
        }
        // (The scroll range is known one frame after the content grows.)
        for _ in 0..3 {
            d.frame(vec![], &mut build);
        }
        assert!(d.semantic("line 199 with a break").is_some(), "follows new lines");
        // The level filter and the search.
        d.click(d.rect_of("Debug").center(), &mut build);
        model.borrow_mut().console.search.push_str("6.1 ms");
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Frame 120 took 6.1 ms").is_some());
        d.click(d.rect_of("Frame 120 took 6.1 ms").center(), &mut build);
        d.key(Key::Letter('C'), Modifiers::CTRL, &mut build);
        assert_eq!(d.output.copied.as_deref(), Some("Frame 120 took 6.1 ms"));
        let mut log = ConsoleLog::new();
        for i in 0..(ConsoleLog::MAX + 10) {
            log.push(LogLevel::Info, 0.0, if i % 2 == 0 { "even" } else { "odd" });
        }
        assert_eq!((log.len(), log.dropped()), (ConsoleLog::MAX, 10));
        let long = "\u{e9}".repeat(ConsoleLog::MAX_LINE);
        log.push(LogLevel::Warn, 0.0, &long);
        assert!(log.entries().last().is_some_and(|e| e.text.len() <= ConsoleLog::MAX_LINE));
    }

    /// Readable names of the test actions.
    fn describe(a: &str) -> &str {
        match a {
            "edit.undo" => "Undo",
            "edit.redo" => "Redo",
            "view.frame" => "Frame the selection",
            _ => "",
        }
    }

    #[test]
    fn ui_07_shortcut_overlay_lists_groups_and_filters() {
        let mut keymap = Keymap::new();
        keymap.bind("edit.undo", Shortcut::new(Key::Letter('Z'), Modifiers::CTRL));
        keymap.bind("edit.redo", Shortcut::new(Key::Letter('Y'), Modifiers::CTRL));
        keymap.bind("view.frame", Shortcut::new(Key::Letter('F'), Modifiers::NONE));
        let filter = RefCell::new(String::new());
        let closes = RefCell::new(0);
        let mut build = |ui: &mut Ui| {
            if shortcut_overlay(ui, &keymap, &mut filter.borrow_mut(), &describe).close {
                *closes.borrow_mut() += 1;
            }
        };
        let mut d = Driver::new(&Theme::dark(), 700.0, 600.0);
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        for label in ["edit", "view", "Undo", "Redo", "Frame the selection"] {
            assert!(d.semantic(label).is_some(), "{label}");
        }
        assert!(
            matches!(d.semantic("Undo").map(|s| s.value), Some(SemanticValue::Text(t)) if d.ui.text(t) == "Ctrl+Z")
        );
        filter.borrow_mut().push_str("frame");
        d.frame(vec![], &mut build);
        assert!(d.semantic("Undo").is_none() && d.semantic("Frame the selection").is_some());
        d.key(Key::Escape, Modifiers::NONE, &mut build);
        d.key(Key::Escape, Modifiers::NONE, &mut build);
        assert!(*closes.borrow() >= 1);
    }

    #[test]
    fn perf_r1_views_do_not_allocate_in_steady_frames() {
        // [PERF-R1] [UI-04] A 2000-row table, a list, a tree, plots of 20 000 samples, the
        // timeline, the property grid and the console: no allocation once warm.
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        for _ in 0..4 {
            d.frame(vec![], &mut build);
        }
        let stats = humanart_core::alloc::count_in(|| {
            for _ in 0..5 {
                d.frame(Vec::new(), &mut build);
            }
        });
        assert_eq!(stats.allocations + stats.reallocations, 0, "{stats:?}");
        let instances = d.ui.draw_list().instances().len();
        assert!(instances < 4000, "the plots are bounded by pixels, not samples: {instances}");
    }

    #[test]
    fn ac_wp12_2_views_golden_at_100_150_200_percent() {
        // [UI-07] [AC-WP12-2] Table (sorted, selected), tree, list, timeline, histogram,
        // waveform, image view, property grid and console.
        use crate::tests::{assert_golden, fonts, gpu_lock, render, renderer};
        let _lock = gpu_lock();
        let mut r = renderer(fonts().atlas());
        // A 64 × 48 test card: a checker with a gradient.
        let mut rgba = Vec::with_capacity(64 * 48 * 4);
        for y in 0..48_u32 {
            for x in 0..64_u32 {
                let check = ((x / 8) + (y / 8)) % 2 == 0;
                #[allow(clippy::cast_possible_truncation)]
                let g = (x * 4) as u8;
                rgba.extend_from_slice(&if check { [200, g, 90, 255] } else { [40, 60, g, 255] });
            }
        }
        let gpu = crate::tests::gpu();
        let texture = r.add_image(&gpu.device, &gpu.queue, 64, 48, rgba).expect("image");
        for (theme, scale, name) in [
            (Theme::dark(), 1.0, "views_100"),
            (Theme::dark(), 1.5, "views_150"),
            (Theme::dark(), 2.0, "views_200"),
            (Theme::light(), 1.0, "views_light"),
        ] {
            let model = RefCell::new(Model::new());
            model.borrow_mut().texture = texture;
            model.borrow_mut().table_sel = Some(2);
            model.borrow_mut().sort = Some(Sort { column: 1, descending: true });
            model.borrow_mut().tree_sel = Some(5);
            let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
            let mut d = setup(&theme);
            d.scale = scale;
            for _ in 0..3 {
                d.frame(vec![], &mut build);
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let (w, h) = ((960.0 * scale) as u32, (760.0 * scale) as u32);
            let px = render(&mut r, d.ui.draw_list(), w, h, d.ui.theme().palette.bg);
            assert!(r.stats().draw_calls <= 8, "[UI-02] {:?}", r.stats());
            assert_golden(name, w, h, &px, 2, (w * h / 1000) as usize);
        }
    }
}
