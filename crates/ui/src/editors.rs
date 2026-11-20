//! Value editors `[UI-07]`: the drag value and the vector editor, the range slider, the colour
//! editor and colour picker (HSV, linear RGB and Lab channels, hex entry, eyedropper request,
//! swatch history, reference swatches), the curve and gradient editors, the 2D pad and the angle
//! dial.
//!
//! Every editor is operable from the keyboard `[UI-09]` (the keys are in each doc comment),
//! describes itself semantically, and reports an [`EditPhase`] so the caller can make one undo step
//! per gesture, as the slider does.
//!
//! **Colour spaces.** Colours are linear Rec.709 ([`Color`]). The picker's HSV channels act on the
//! display-encoded (sRGB) values, as artists expect from HSV; its RGB channels are the linear
//! values; Lab is CIELAB under D65 from the linear values (`humanart_math::color`). The picker edits
//! colours inside the unit cube (alpha is kept). Its ramps are drawn as short linear-light gradient
//! segments, which follow the true ramps closely but not exactly.

use humanart_math::color::{hsv, lab, transfer};

use crate::color::Color;
use crate::controls::{IconButton, tabs};
use crate::geom::{Insets, Point, Rect};
use crate::input::{Key, Modifiers};
use crate::kit;
use crate::layout::{Align as BoxAlign, Dim, NodeId, Position, Style};
use crate::paint::{Axis, Corners, Layer};
use crate::popup::{Tip, tooltip};
use crate::text::Align;
use crate::ui::{Anchor, Area, CursorIcon, Id, Paint, Response, Role, SemanticValue, Sense, Ui};
use crate::widgets::{EditPhase, SliderResponse, TextField, format_value};

/// Rows of gradient cells drawing the saturation / value square.
const SV_ROWS: usize = 32;
/// Inset of a curve's plot inside its box (room for the end points' handles).
const CURVE_PAD: f32 = 8.0;
/// Line segments drawing a curve.
const CURVE_SEGMENTS: usize = 48;
/// Inset of a gradient's ramp inside its box.
const GRADIENT_PAD: f32 = 6.0;
/// Height of a gradient's ramp.
const RAMP_H: f32 = 22.0;
/// Height of a gradient's stop markers.
const MARK_H: f32 = 14.0;
/// Side of the angle dial.
const DIAL_SIZE: f32 = 44.0;

/// The undo phase of an interaction this frame: `Begin` on a press, `Update` while a held gesture
/// changes the value, `Commit` on its release, `Step` for a complete change in one frame.
fn phase_of(ui: &Ui, r: &Response, changed: bool, stepped: bool) -> EditPhase {
    let released = ui.pointer_released();
    if r.pressed && released {
        return if changed { EditPhase::Step } else { EditPhase::Idle };
    }
    if r.pressed {
        return EditPhase::Begin;
    }
    if r.held {
        if released {
            return EditPhase::Commit;
        }
        return if changed { EditPhase::Update } else { EditPhase::Idle };
    }
    if stepped && changed { EditPhase::Step } else { EditPhase::Idle }
}

/// The first phase that is not idle.
fn first_phase(phases: &[EditPhase]) -> EditPhase {
    phases.iter().copied().find(|p| *p != EditPhase::Idle).unwrap_or_default()
}

/// A finite number from user text.
fn parse_number(text: &str) -> Option<f32> {
    text.trim().parse::<f32>().ok().filter(|v| v.is_finite())
}

/// Arrow-key stepping for a focused editor: Right / Up add, Left / Down subtract, Page Up / Down
/// ten steps; Shift is a tenth of a step. Returns the total change in steps.
fn arrow_steps(ui: &mut Ui, horizontal: bool, vertical: bool) -> f32 {
    let mut total = 0.0;
    let mut keys: [(Key, f32); 6] = [
        (Key::Right, 1.0),
        (Key::Left, -1.0),
        (Key::Up, 1.0),
        (Key::Down, -1.0),
        (Key::PageUp, 10.0),
        (Key::PageDown, -10.0),
    ];
    if !horizontal {
        keys[0].1 = 0.0;
        keys[1].1 = 0.0;
    }
    if !vertical {
        keys[2].1 = 0.0;
        keys[3].1 = 0.0;
    }
    for (key, delta) in keys {
        if delta == 0.0 {
            continue;
        }
        while let Some(mods) = ui.consume_key_any(key) {
            total += delta * if mods.shift { 0.1 } else { 1.0 };
        }
    }
    total
}

/// The numeric-entry state of a drag value, in [`Ui::memory`].
#[derive(Clone, Debug, Default)]
struct NumberEntry {
    editing: bool,
    text: String,
    just_opened: bool,
}

/// A compact number box edited by dragging `[UI-07]` (the cells of vector editors).
///
/// Pointer: drag horizontally to change the value by `speed` per pixel (Shift: a tenth); a click
/// opens numeric entry. Keyboard (focused): Enter, Space or a click opens numeric entry (Enter
/// applies, Escape cancels), Right / Up and Left / Down step by ten `speed` (Shift: one), Page Up /
/// Down a hundred, Delete or Backspace restores the default.
#[derive(Debug)]
pub struct DragValue<'a> {
    label: &'a str,
    value: &'a mut f32,
    min: f32,
    max: f32,
    speed: f32,
    decimals: usize,
    default: Option<f32>,
    prefix: Option<(&'a str, Color)>,
    width: Option<f32>,
}

impl<'a> DragValue<'a> {
    /// A drag value for `value`, labelled (and identified) by `label`.
    pub fn new(label: &'a str, value: &'a mut f32) -> Self {
        Self {
            label,
            value,
            min: f32::MIN,
            max: f32::MAX,
            speed: 0.01,
            decimals: 3,
            default: None,
            prefix: None,
            width: None,
        }
    }

    /// The allowed range.
    #[must_use]
    pub fn range(mut self, min: f32, max: f32) -> Self {
        let (min, max) = if min <= max { (min, max) } else { (max, min) };
        if min.is_finite() && max.is_finite() {
            self.min = min;
            self.max = max;
        }
        self
    }

    /// Change per dragged pixel.
    #[must_use]
    pub fn speed(mut self, speed: f32) -> Self {
        if speed.is_finite() && speed > 0.0 {
            self.speed = speed;
        }
        self
    }

    /// Decimals shown.
    #[must_use]
    pub fn decimals(mut self, decimals: usize) -> Self {
        self.decimals = decimals.min(6);
        self
    }

    /// The value Delete restores.
    #[must_use]
    pub fn default_value(mut self, default: f32) -> Self {
        self.default = Some(default);
        self
    }

    /// A short coloured tag before the number (the axis letter of a vector editor).
    #[must_use]
    pub fn prefix(mut self, text: &'a str, color: Color) -> Self {
        self.prefix = Some((text, color));
        self
    }

    /// A fixed width (otherwise it grows).
    #[must_use]
    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width);
        self
    }

    /// Adds the box.
    pub fn show(self, ui: &mut Ui) -> SliderResponse {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let id = ui.id(self.label);
        let before = *self.value;
        let mut entry = std::mem::take(ui.memory::<NumberEntry>(id));
        let width = self.width.map_or(Dim::Auto, Dim::Px);
        let grow = if self.width.is_some() { 0.0 } else { kit::grow_x(ui) };
        let mut stepped = false;
        let mut r;
        if entry.editing {
            let field_id = id.with("entry");
            if entry.just_opened {
                ui.request_focus(field_id);
                entry.just_opened = false;
            }
            let cancel = ui.focused() == Some(field_id) && ui.consume_key(Key::Escape, Modifiers::NONE);
            ui.begin(Style { width, grow, min_width: 56.0, ..Style::row() });
            ui.push_id(id);
            let out = TextField::new("entry", &mut entry.text).mono(true).max_chars(32).show(ui);
            ui.pop_id();
            ui.end();
            r = Response { id, rect: out.response.rect, ..Response::default() };
            let lost = ui.focused() != Some(field_id) && !out.response.focused;
            if out.submitted || lost || cancel {
                if !cancel && let Some(v) = parse_number(&entry.text) {
                    *self.value = v;
                    stepped = true;
                }
                entry.editing = false;
                ui.request_focus(id);
            }
        } else {
            let node =
                ui.node(Style { width, grow, min_width: 56.0, height: Dim::Px(m.control_height), ..Style::default() });
            r = ui.interact(id, node, Sense::DRAG);
            if r.dragged {
                let fine = if ui.modifiers().shift { 0.1 } else { 1.0 };
                *self.value += r.drag_delta.x * self.speed * fine;
            }
            if r.focused {
                let steps = arrow_steps(ui, true, true);
                if steps != 0.0 {
                    *self.value += steps * self.speed * 10.0;
                    stepped = true;
                }
                if let Some(d) = self.default
                    && (ui.consume_key(Key::Delete, Modifiers::NONE) || ui.consume_key(Key::Backspace, Modifiers::NONE))
                {
                    *self.value = d;
                    stepped = true;
                }
            }
            if r.clicked {
                entry.editing = true;
                entry.just_opened = true;
                entry.text.clear();
                let mut buf = [0_u8; 32];
                entry.text.push_str(format_value(*self.value, self.decimals, &mut buf));
            }
            if r.hovered || r.dragged {
                ui.set_cursor(CursorIcon::ResizeHorizontal);
            }
            let hot = r.hover_t.max(r.press_t);
            kit::frame(
                ui,
                node,
                Area::FULL,
                p.bg.mix(p.hover, hot * 0.6),
                m.radius_sm,
                p.border_strong.mix(p.border_focus, hot),
            );
            let mut left = m.space_3;
            if let Some((tag, color)) = self.prefix {
                kit::text(
                    ui,
                    node,
                    kit::at(m.space_3, 0.0, 12.0, m.control_height),
                    tag,
                    theme.text.label,
                    color,
                    Align::Center,
                );
                left += 14.0;
            }
            let mut buf = [0_u8; 32];
            let shown = format_value(*self.value, self.decimals, &mut buf);
            kit::text(ui, node, kit::inset_x(left, m.space_3), shown, theme.text.mono, p.text, Align::End);
            if r.focus_visible {
                kit::ring(ui, node, Area::FULL, m.radius_sm);
            }
        }
        *self.value = if self.value.is_finite() { self.value.clamp(self.min, self.max) } else { before };
        r.changed = (*self.value - before).abs() > 0.0;
        let phase = phase_of(ui, &r, r.changed, stepped);
        *ui.memory::<NumberEntry>(id) = entry;
        ui.semantic(
            id,
            Role::Slider,
            self.label,
            SemanticValue::Number { value: f64::from(*self.value), min: f64::from(self.min), max: f64::from(self.max) },
        );
        SliderResponse { response: r, phase }
    }
}

/// Axis tags and their palette colours for vector editors.
fn axis_tag(ui: &Ui, i: usize) -> (&'static str, Color) {
    let p = ui.theme().palette;
    match i {
        0 => ("X", p.danger),
        1 => ("Y", p.positive),
        2 => ("Z", p.link),
        _ => ("W", p.text_muted),
    }
}

/// A vector editor `[UI-07]`: a label and one [`DragValue`] per component (tagged X, Y, Z, W),
/// each its own Tab stop. Returns a combined response: `changed` when any component changed, the
/// first non-idle phase.
pub fn vector_edit(ui: &mut Ui, label: &str, values: &mut [f32], speed: f32, decimals: usize) -> SliderResponse {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let id = ui.id(label);
    ui.begin(Style { gap: m.space_2, align_items: BoxAlign::Center, ..Style::row() });
    let name =
        ui.node(Style { width: Dim::Px(96.0), height: Dim::Px(m.control_height), shrink: 0.0, ..Style::default() });
    kit::text(ui, name, Area::FULL, label, theme.text.small, p.text_muted, Align::Start);
    ui.push_id(id);
    let mut out = SliderResponse { response: Response { id, ..Response::default() }, phase: EditPhase::Idle };
    for (i, v) in values.iter_mut().enumerate().take(4) {
        let (tag, color) = axis_tag(ui, i);
        let cell = kit::fmt64(format_args!("{label} {tag}"));
        let r = DragValue::new(cell.as_str(), v).speed(speed).decimals(decimals).prefix(tag, color).show(ui);
        out.response.changed |= r.response.changed;
        out.response.focused |= r.response.focused;
        if out.phase == EditPhase::Idle {
            out.phase = r.phase;
        }
    }
    ui.pop_id();
    ui.end();
    ui.semantic(id, Role::Group, label, SemanticValue::None);
    out
}

/// Which thumb of a range slider the track is moving, in [`Ui::memory`].
#[derive(Clone, Copy, Debug, Default)]
struct RangeGrab {
    hi: bool,
}

/// A slider with two thumbs choosing a sub-range `lo..=hi` of `min..=max` `[UI-07]`.
///
/// Each thumb is a Tab stop: arrows step it (Shift: a tenth), Page Up / Down ten steps, Home / End
/// the ends; the thumbs cannot cross. Pointer: drag a thumb, or press the track to move the nearer
/// thumb there and drag on.
#[derive(Debug)]
pub struct RangeSlider<'a> {
    label: &'a str,
    lo: &'a mut f32,
    hi: &'a mut f32,
    min: f32,
    max: f32,
    step: Option<f32>,
    decimals: usize,
}

impl<'a> RangeSlider<'a> {
    /// A range slider for `lo..=hi` inside `min..=max`, labelled (and identified) by `label`.
    pub fn new(label: &'a str, (lo, hi): (&'a mut f32, &'a mut f32), min: f32, max: f32) -> Self {
        let (min, max) = if min <= max { (min, max) } else { (max, min) };
        Self { label, lo, hi, min, max, step: None, decimals: 2 }
    }

    /// Snaps values to multiples of `step` from `min` (also the keyboard step).
    #[must_use]
    pub fn step(mut self, step: f32) -> Self {
        self.step = (step.is_finite() && step > 0.0).then_some(step);
        self
    }

    /// Decimals shown.
    #[must_use]
    pub fn decimals(mut self, decimals: usize) -> Self {
        self.decimals = decimals.min(6);
        self
    }

    fn quantize(&self, v: f32) -> f32 {
        let v = if v.is_finite() { v.clamp(self.min, self.max) } else { self.min };
        match self.step {
            Some(s) => (self.min + ((v - self.min) / s).round() * s).clamp(self.min, self.max),
            None => v,
        }
    }

    /// Adds the slider; the response is the track's (`changed` when either end moved).
    #[allow(clippy::too_many_lines)]
    pub fn show(self, ui: &mut Ui) -> SliderResponse {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let range = self.max - self.min;
        let before = (*self.lo, *self.hi);
        let id = ui.id(self.label);
        ui.begin(Style {
            height: Dim::Px(m.control_height),
            gap: m.space_4,
            align_items: BoxAlign::Stretch,
            grow: kit::grow_x(ui),
            ..Style::row()
        });
        let label_node = ui.node(Style { width: Dim::Px(96.0), shrink: 0.0, ..Style::default() });
        kit::text(ui, label_node, Area::FULL, self.label, theme.text.small, p.text_muted, Align::Start);
        let track = ui.begin(Style { grow: 1.0, min_width: 60.0, ..Style::default() });
        let mut r = ui.interact(id, track, Sense { click: true, drag: true, focus: false });
        let thumb = m.slider_thumb;
        let half = thumb * 0.5;
        let usable = (r.rect.w - thumb).max(1.0);
        let frac = |v: f32| if range > 0.0 { ((v - self.min) / range).clamp(0.0, 1.0) } else { 0.0 };
        let value_at = |x: f32| self.min + ((x - r.rect.x - half) / usable).clamp(0.0, 1.0) * range;
        // The thumbs: absolute children of the track, registered after it so they win the hover.
        let mut thumbs = [Response::default(); 2];
        let names = ["min", "max"];
        for (k, slot) in thumbs.iter_mut().enumerate() {
            let v = if k == 0 { *self.lo } else { *self.hi };
            let node = ui.node(Style {
                position: Position::Absolute {
                    left: Some(frac(v) * usable),
                    top: Some((m.control_height - thumb) * 0.5),
                    right: None,
                    bottom: None,
                },
                width: Dim::Px(thumb),
                height: Dim::Px(thumb),
                ..Style::default()
            });
            *slot = ui.interact(id.with(names[k]), node, Sense::DRAG);
        }
        ui.end();
        let mut stepped = false;
        // Track press: the nearer thumb jumps to the pointer and follows it.
        if r.pressed
            && let Some(pos) = ui.pointer()
        {
            let v = value_at(pos.x);
            let hi = (v - *self.hi).abs() < (v - *self.lo).abs() || (v > *self.hi);
            ui.memory::<RangeGrab>(id).hi = hi;
            ui.request_focus(id.with(names[usize::from(hi)]));
        }
        if (r.pressed || r.dragged)
            && let Some(pos) = ui.pointer()
        {
            let v = value_at(pos.x);
            if ui.memory::<RangeGrab>(id).hi {
                *self.hi = v.max(*self.lo);
            } else {
                *self.lo = v.min(*self.hi);
            }
        }
        let fine = if ui.modifiers().shift { 0.1 } else { 1.0 };
        let step = self.step.unwrap_or(range / 100.0);
        for (k, t) in thumbs.iter().enumerate() {
            let mut v = if k == 0 { *self.lo } else { *self.hi };
            if t.dragged {
                v += t.drag_delta.x / usable * range * fine;
            }
            if t.focused {
                let steps = arrow_steps(ui, true, true);
                if steps != 0.0 {
                    v += steps * step;
                    stepped = true;
                }
                if ui.consume_key(Key::Home, Modifiers::NONE) {
                    v = self.min;
                    stepped = true;
                }
                if ui.consume_key(Key::End, Modifiers::NONE) {
                    v = self.max;
                    stepped = true;
                }
            }
            if k == 0 {
                *self.lo = v.min(*self.hi);
            } else {
                *self.hi = v.max(*self.lo);
            }
        }
        let held = r.held || thumbs.iter().any(|t| t.held);
        let dragging = r.dragged || thumbs.iter().any(|t| t.dragged);
        if !dragging {
            *self.lo = self.quantize(*self.lo);
            *self.hi = self.quantize(*self.hi).max(*self.lo);
        }
        r.changed = (*self.lo, *self.hi) != before;
        r.held = held;
        r.pressed |= thumbs.iter().any(|t| t.pressed);
        r.focused = thumbs.iter().any(|t| t.focused);
        let phase = phase_of(ui, &r, r.changed, stepped);
        if held {
            ui.set_cursor(CursorIcon::Grabbing);
        }

        // Look.
        let th = m.slider_track;
        let band = |a: f32, b: f32| Area {
            x0: Anchor::start(half + a * usable),
            y0: Anchor::at(0.5, -th * 0.5),
            x1: Anchor::start(half + b * usable),
            y1: Anchor::at(0.5, th * 0.5),
        };
        kit::fill(ui, track, band(0.0, 1.0), p.active, th * 0.5);
        kit::fill(ui, track, band(frac(*self.lo), frac(*self.hi)), p.accent, th * 0.5);
        for (k, t) in thumbs.iter().enumerate() {
            let x = frac(if k == 0 { *self.lo } else { *self.hi }) * usable;
            let grow = 1.0 + 0.12 * t.hover_t.max(t.press_t);
            ui.paint(
                track,
                Paint::Circle {
                    cx: Anchor::start(x + half),
                    cy: Anchor::at(0.5, 0.0),
                    radius: half * grow,
                    color: p.text,
                },
            );
            if t.focus_visible {
                kit::ring(
                    ui,
                    track,
                    Area {
                        x0: Anchor::start(x),
                        y0: Anchor::at(0.5, -half),
                        x1: Anchor::start(x + thumb),
                        y1: Anchor::at(0.5, half),
                    },
                    half,
                );
            }
            let name = kit::fmt64(format_args!("{} {}", self.label, names[k]));
            let v = if k == 0 { *self.lo } else { *self.hi };
            ui.semantic(
                t.id,
                Role::Slider,
                name.as_str(),
                SemanticValue::Number { value: f64::from(v), min: f64::from(self.min), max: f64::from(self.max) },
            );
        }
        let value_node = ui.node(Style { width: Dim::Px(96.0), shrink: 0.0, ..Style::default() });
        let (mut a, mut b) = ([0_u8; 32], [0_u8; 32]);
        let text = kit::fmt64(format_args!(
            "{} \u{2013} {}",
            format_value(*self.lo, self.decimals, &mut a),
            format_value(*self.hi, self.decimals, &mut b)
        ));
        kit::text(ui, value_node, Area::FULL, text.as_str(), theme.text.mono, p.text, Align::End);
        ui.end();
        ui.semantic(id, Role::Group, self.label, SemanticValue::None);
        SliderResponse { response: r, phase }
    }
}

// ---------------------------------------------------------------------------------------------
// Colour.

/// The display-encoded (sRGB) components of a linear colour.
fn encode(c: Color) -> [f32; 3] {
    transfer::linear_to_srgb_rgb_f32([c.r.clamp(0.0, 1.0), c.g.clamp(0.0, 1.0), c.b.clamp(0.0, 1.0)])
}

/// A linear colour from display-encoded components and an alpha.
fn decode(e: [f32; 3], a: f32) -> Color {
    let l = transfer::srgb_to_linear_rgb_f32(e);
    Color::linear(l[0].clamp(0.0, 1.0), l[1].clamp(0.0, 1.0), l[2].clamp(0.0, 1.0), a)
}

/// HSV (hue in degrees, saturation and value 0..1) of the display-encoded colour.
pub fn color_to_hsv(c: Color) -> [f32; 3] {
    hsv::rgb_to_hsv_f32(encode(c))
}

/// The linear colour of HSV on display-encoded values, with alpha `a`.
pub fn hsv_to_color(h: [f32; 3], a: f32) -> Color {
    decode(hsv::hsv_to_rgb_f32([h[0], h[1].clamp(0.0, 1.0), h[2].clamp(0.0, 1.0)]), a)
}

/// CIELAB (D65) of a linear colour.
#[allow(clippy::cast_possible_truncation)] // Lab values are small.
pub fn color_to_lab(c: Color) -> [f32; 3] {
    let l = lab::rgb_to_lab([f64::from(c.r), f64::from(c.g), f64::from(c.b)]);
    [l[0] as f32, l[1] as f32, l[2] as f32]
}

/// The linear colour of CIELAB (D65), clamped into the unit cube, with alpha `a`.
#[allow(clippy::cast_possible_truncation)] // Clamped to 0..1.
pub fn lab_to_color(l: [f32; 3], a: f32) -> Color {
    let rgb = lab::lab_to_rgb([f64::from(l[0]), f64::from(l[1]), f64::from(l[2])]);
    let c = |v: f64| if v.is_finite() { v.clamp(0.0, 1.0) as f32 } else { 0.0 };
    Color::linear(c(rgb[0]), c(rgb[1]), c(rgb[2]), a)
}

/// Whether two colours are the same at 8-bit sRGB precision.
fn same_color(a: Color, b: Color) -> bool {
    a.to_srgb8() == b.to_srgb8()
}

/// A labelled colour (a reference swatch).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Swatch<'a> {
    /// Its name (tooltip and semantic label).
    pub label: &'a str,
    /// The colour.
    pub color: Color,
}

/// The colours picked recently, newest first `[UI-07]`. The application keeps one per document or
/// session and passes it to the pickers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SwatchHistory {
    colors: Vec<Color>,
}

impl SwatchHistory {
    /// Most colours kept.
    pub const MAX: usize = 12;

    /// An empty history.
    pub fn new() -> Self {
        Self::default()
    }

    /// Puts `color` first (moving it if it is already there) and drops the oldest beyond [`Self::MAX`].
    pub fn push(&mut self, color: Color) {
        self.colors.retain(|c| !same_color(*c, color));
        self.colors.insert(0, color);
        self.colors.truncate(Self::MAX);
    }

    /// The colours, newest first.
    pub fn colors(&self) -> &[Color] {
        &self.colors
    }
}

/// What a colour picker or editor did this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PickerResponse {
    /// The colour changed.
    pub changed: bool,
    /// Undo integration (one step per gesture, keyboard change, hex entry or swatch pick).
    pub phase: EditPhase,
    /// The eyedropper button was pressed: the application samples the viewport under the next
    /// click and writes the colour back.
    pub eyedropper: bool,
}

/// The picker's memory: the HSV kept across edits (so the hue survives grey) and the hex text.
#[derive(Clone, Debug, Default)]
struct PickerState {
    known: Option<[u8; 4]>,
    hsv: [f32; 3],
    mode: usize,
    hex: String,
}

/// Draws `segments` horizontal gradient quads over `area` of `node` following `color_at(t)`.
fn ramp(ui: &mut Ui, node: NodeId, segments: usize, radius: f32, color_at: &dyn Fn(f32) -> Color) {
    #[allow(clippy::cast_precision_loss)] // A handful of segments.
    let n = segments.max(1) as f32;
    for k in 0..segments.max(1) {
        #[allow(clippy::cast_precision_loss)]
        let (a, b) = (k as f32 / n, (k + 1) as f32 / n);
        let corners = Corners {
            tl: if k == 0 { radius } else { 0.0 },
            bl: if k == 0 { radius } else { 0.0 },
            tr: if k + 1 == segments { radius } else { 0.0 },
            br: if k + 1 == segments { radius } else { 0.0 },
        };
        let area =
            Area { x0: Anchor::at(a, 0.0), y0: Anchor::start(0.0), x1: Anchor::at(b, 0.0), y1: Anchor::end(0.0) };
        ui.paint(
            node,
            Paint::Gradient { area, from: color_at(a), to: color_at(b), radius: corners, axis: Axis::Horizontal },
        );
    }
}

/// A clickable colour square; `selected` draws it ringed.
fn swatch(ui: &mut Ui, id: Id, label: &str, color: Color, size: f32, selected: bool) -> Response {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let node = ui.node(Style { width: Dim::Px(size), height: Dim::Px(size), shrink: 0.0, ..Style::default() });
    let r = ui.interact(id, node, Sense::CLICK);
    let border = if selected { p.text } else { p.border_strong.mix(p.text_muted, r.hover_t) };
    kit::frame(ui, node, Area::FULL, color.with_alpha(1.0), 4.0, border);
    if r.focus_visible {
        kit::ring(ui, node, Area::FULL, 4.0);
    }
    let _ = m;
    let mut hex = kit::Buf::<16>::new();
    let [rr, gg, bb, _] = color.to_srgb8();
    let _ = std::fmt::write(&mut hex, format_args!("#{rr:02X}{gg:02X}{bb:02X}"));
    let value = ui.intern(hex.as_str());
    ui.semantic(id, Role::Button, label, SemanticValue::Text(value));
    let rows = [("sRGB", hex.as_str())];
    tooltip(ui, &r, &Tip::new(label).rows(&rows));
    r
}

/// One channel row of the picker: a name, a ramp bar edited by pointer and arrows, and a
/// [`DragValue`] for the exact number. Returns (changed, phase).
#[allow(clippy::too_many_arguments)]
fn channel_row(
    ui: &mut Ui,
    name: &str,
    value: &mut f32,
    (min, max): (f32, f32),
    decimals: usize,
    step: f32,
    color_at: &dyn Fn(f32) -> Color,
) -> (bool, EditPhase) {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let before = *value;
    let range = max - min;
    ui.begin(Style {
        gap: m.space_3,
        align_items: BoxAlign::Center,
        height: Dim::Px(m.control_height_sm + 4.0),
        ..Style::row()
    });
    let tag = ui.node(Style { width: Dim::Px(14.0), height: Dim::Percent(100.0), shrink: 0.0, ..Style::default() });
    kit::text(ui, tag, Area::FULL, name, theme.text.label, p.text_muted, Align::Center);
    let bar = ui.node(Style { grow: 1.0, height: Dim::Px(12.0), ..Style::default() });
    let id = ui.id(name).with("bar");
    let r = ui.interact(id, bar, Sense::DRAG);
    let mut stepped = false;
    if (r.pressed || r.dragged)
        && let Some(f) = ui.pointer_fraction(r.rect)
    {
        *value = min + f.x.clamp(0.0, 1.0) * range;
    }
    if r.focused {
        let steps = arrow_steps(ui, true, true);
        if steps != 0.0 {
            *value += steps * step;
            stepped = true;
        }
    }
    *value = value.clamp(min, max);
    ramp(ui, bar, 8, 3.0, &|t| color_at(min + t * range));
    let t = if range > 0.0 { (*value - min) / range } else { 0.0 };
    let marker =
        Area { x0: Anchor::at(t, -2.0), y0: Anchor::start(-2.0), x1: Anchor::at(t, 2.0), y1: Anchor::end(2.0) };
    ui.paint(
        bar,
        Paint::Rect {
            area: marker,
            fill: Color::WHITE,
            radius: Corners::all(1.5),
            border: 1.0,
            border_color: Color::BLACK.fade(0.6),
        },
    );
    if r.focus_visible {
        kit::ring(ui, bar, Area::FULL, 3.0);
    }
    ui.semantic(
        id,
        Role::Slider,
        name,
        SemanticValue::Number { value: f64::from(*value), min: f64::from(min), max: f64::from(max) },
    );
    let mut changed_bar = (*value - before).abs() > 0.0;
    let bar_phase = phase_of(ui, &r, changed_bar, stepped);
    let speed = range / 400.0;
    let dv = DragValue::new(name, value).range(min, max).speed(speed).decimals(decimals).width(64.0).show(ui);
    ui.end();
    changed_bar |= dv.response.changed;
    (changed_bar, first_phase(&[bar_phase, dv.phase]))
}

/// A colour picker `[UI-07]` laid out inline (the colour editor shows it in a popup): a
/// saturation / value square, a hue bar, HSV / RGB / Lab channel rows (tabs), a hex field, an
/// eyedropper button, the swatch history and optional reference swatches.
///
/// Keyboard: every part is a Tab stop. On the square Left / Right change the saturation and Up /
/// Down the value (Shift: a tenth of the step); on the hue bar and channel bars the arrows step;
/// channel numbers are [`DragValue`]s; the hex field applies `#RRGGBB` as it is typed.
#[derive(Debug)]
pub struct ColorPicker<'a> {
    id_label: &'a str,
    color: &'a mut Color,
    history: Option<&'a mut SwatchHistory>,
    reference: &'a [Swatch<'a>],
    width: f32,
}

impl<'a> ColorPicker<'a> {
    /// A picker for `color`, identified by `id_label`.
    pub fn new(id_label: &'a str, color: &'a mut Color) -> Self {
        Self { id_label, color, history: None, reference: &[], width: 260.0 }
    }

    /// The swatch history to show and to record committed colours in.
    #[must_use]
    pub fn history(mut self, history: &'a mut SwatchHistory) -> Self {
        self.history = Some(history);
        self
    }

    /// Reference swatches shown under the history (the application passes, for example, the
    /// Fitzpatrick reference swatches from its data table).
    #[must_use]
    pub fn reference(mut self, swatches: &'a [Swatch<'a>]) -> Self {
        self.reference = swatches;
        self
    }

    /// The width of the picker (default 260).
    #[must_use]
    pub fn width(mut self, width: f32) -> Self {
        if width.is_finite() {
            self.width = width.max(180.0);
        }
        self
    }

    /// Adds the picker.
    #[allow(clippy::too_many_lines)]
    pub fn show(self, ui: &mut Ui) -> PickerResponse {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let id = ui.id(self.id_label);
        let before = *self.color;
        let alpha = before.a;
        let mut st = std::mem::take(ui.memory::<PickerState>(id));
        // Re-derive the HSV when the colour changed from outside (keeping hue and saturation where
        // the colour does not define them).
        if st.known != Some(before.to_srgb8()) {
            let h = color_to_hsv(before);
            st.hsv = match (h[1] > 0.0, h[2] > 0.0) {
                (true, true) => h,
                (false, true) => [st.hsv[0], h[1], h[2]],
                _ => [st.hsv[0], st.hsv[1], h[2]],
            };
        }
        let mut out = PickerResponse::default();
        let mut phases = [EditPhase::Idle; 8];
        let mut hsv_edited = false;
        ui.begin(Style { width: Dim::Px(self.width), gap: m.space_4, ..Style::column() });
        ui.push_id(id);

        // Saturation / value square.
        let sv = ui.node(Style { height: Dim::Px(150.0), ..Style::default() });
        let sv_id = ui.id("Saturation and value");
        let r = ui.interact(sv_id, sv, Sense::DRAG);
        let mut stepped = false;
        if (r.pressed || r.dragged)
            && let Some(f) = ui.pointer_fraction(r.rect)
        {
            st.hsv[1] = f.x.clamp(0.0, 1.0);
            st.hsv[2] = 1.0 - f.y.clamp(0.0, 1.0);
            hsv_edited = true;
        }
        if r.focused {
            let ds = arrow_steps(ui, true, false);
            let dv = arrow_steps(ui, false, true);
            if ds != 0.0 || dv != 0.0 {
                st.hsv[1] = (st.hsv[1] + ds * 0.01).clamp(0.0, 1.0);
                st.hsv[2] = (st.hsv[2] + dv * 0.01).clamp(0.0, 1.0);
                hsv_edited = true;
                stepped = true;
            }
        }
        phases[0] = phase_of(ui, &r, hsv_edited, stepped);
        let hue = st.hsv[0];
        for row in 0..SV_ROWS {
            #[allow(clippy::cast_precision_loss)]
            let (y0, y1) = (row as f32 / SV_ROWS as f32, (row + 1) as f32 / SV_ROWS as f32);
            let v = 1.0 - f32::midpoint(y0, y1);
            for col in 0..4 {
                #[allow(clippy::cast_precision_loss)]
                let (x0, x1) = (col as f32 / 4.0, (col + 1) as f32 / 4.0);
                let area = Area {
                    x0: Anchor::at(x0, 0.0),
                    y0: Anchor::at(y0, 0.0),
                    x1: Anchor::at(x1, 0.0),
                    y1: Anchor::at(y1, 0.0),
                };
                let corners = Corners {
                    tl: if row == 0 && col == 0 { 4.0 } else { 0.0 },
                    tr: if row == 0 && col == 3 { 4.0 } else { 0.0 },
                    bl: if row == SV_ROWS - 1 && col == 0 { 4.0 } else { 0.0 },
                    br: if row == SV_ROWS - 1 && col == 3 { 4.0 } else { 0.0 },
                };
                ui.paint(
                    sv,
                    Paint::Gradient {
                        area,
                        from: hsv_to_color([hue, x0, v], 1.0),
                        to: hsv_to_color([hue, x1, v], 1.0),
                        radius: corners,
                        axis: Axis::Horizontal,
                    },
                );
            }
        }
        let (mx, my) = (st.hsv[1], 1.0 - st.hsv[2]);
        let knob = Area {
            x0: Anchor::at(mx, -7.0),
            y0: Anchor::at(my, -7.0),
            x1: Anchor::at(mx, 7.0),
            y1: Anchor::at(my, 7.0),
        };
        ui.paint(
            sv,
            Paint::Rect {
                area: knob,
                fill: Color::TRANSPARENT,
                radius: Corners::all(7.0),
                border: 2.0,
                border_color: Color::WHITE,
            },
        );
        let outer = Area {
            x0: Anchor::at(mx, -8.0),
            y0: Anchor::at(my, -8.0),
            x1: Anchor::at(mx, 8.0),
            y1: Anchor::at(my, 8.0),
        };
        ui.paint(
            sv,
            Paint::Rect {
                area: outer,
                fill: Color::TRANSPARENT,
                radius: Corners::all(8.0),
                border: 1.0,
                border_color: Color::BLACK.fade(0.5),
            },
        );
        if r.focus_visible {
            kit::ring(ui, sv, Area::FULL, 4.0);
        }
        ui.semantic(
            sv_id,
            Role::Custom("colorarea"),
            "Saturation and value",
            SemanticValue::Number { value: f64::from(st.hsv[2]), min: 0.0, max: 1.0 },
        );

        // Hue bar.
        let hue_bar = ui.node(Style { height: Dim::Px(12.0), ..Style::default() });
        let hue_id = ui.id("Hue");
        let hr = ui.interact(hue_id, hue_bar, Sense::DRAG);
        let mut hue_changed = false;
        let mut hue_stepped = false;
        if (hr.pressed || hr.dragged)
            && let Some(f) = ui.pointer_fraction(hr.rect)
        {
            st.hsv[0] = f.x.clamp(0.0, 1.0) * 360.0;
            hue_changed = true;
        }
        if hr.focused {
            let d = arrow_steps(ui, true, true);
            if d != 0.0 {
                st.hsv[0] = (st.hsv[0] + d).rem_euclid(360.0);
                hue_changed = true;
                hue_stepped = true;
            }
        }
        st.hsv[0] = st.hsv[0].clamp(0.0, 360.0);
        hsv_edited |= hue_changed;
        phases[1] = phase_of(ui, &hr, hue_changed, hue_stepped);
        ramp(ui, hue_bar, 12, 3.0, &|t| hsv_to_color([t * 360.0, 1.0, 1.0], 1.0));
        let ht = st.hsv[0] / 360.0;
        let marker =
            Area { x0: Anchor::at(ht, -2.0), y0: Anchor::start(-2.0), x1: Anchor::at(ht, 2.0), y1: Anchor::end(2.0) };
        ui.paint(
            hue_bar,
            Paint::Rect {
                area: marker,
                fill: Color::WHITE,
                radius: Corners::all(1.5),
                border: 1.0,
                border_color: Color::BLACK.fade(0.6),
            },
        );
        if hr.focus_visible {
            kit::ring(ui, hue_bar, Area::FULL, 3.0);
        }
        ui.semantic(
            hue_id,
            Role::Slider,
            "Hue",
            SemanticValue::Number { value: f64::from(st.hsv[0]), min: 0.0, max: 360.0 },
        );
        if hsv_edited {
            *self.color = hsv_to_color(st.hsv, alpha);
        }

        // Channels.
        let mut mode = st.mode.min(2);
        tabs(ui, "Colour model", &mut mode, &["HSV", "RGB", "Lab"]);
        st.mode = mode;
        match mode {
            0 => {
                let mut h = st.hsv;
                let (mut s100, mut v100) = (h[1] * 100.0, h[2] * 100.0);
                let hue_now = h[0];
                let (c0, p0) =
                    channel_row(ui, "H", &mut h[0], (0.0, 360.0), 0, 1.0, &|x| hsv_to_color([x, 1.0, 1.0], 1.0));
                let (c1, p1) = channel_row(ui, "S", &mut s100, (0.0, 100.0), 0, 1.0, &|x| {
                    hsv_to_color([hue_now, x / 100.0, h[2]], 1.0)
                });
                let (c2, p2) = channel_row(ui, "V", &mut v100, (0.0, 100.0), 0, 1.0, &|x| {
                    hsv_to_color([hue_now, h[1], x / 100.0], 1.0)
                });
                if c0 || c1 || c2 {
                    st.hsv = [h[0], s100 / 100.0, v100 / 100.0];
                    *self.color = hsv_to_color(st.hsv, alpha);
                    hsv_edited = true;
                }
                phases[2] = first_phase(&[p0, p1, p2]);
            }
            1 => {
                let mut c = [self.color.r, self.color.g, self.color.b];
                let base = c;
                let with = |i: usize, x: f32| {
                    let mut k = base;
                    k[i] = x;
                    Color::linear(k[0], k[1], k[2], 1.0)
                };
                let (c0, p0) = channel_row(ui, "R", &mut c[0], (0.0, 1.0), 3, 0.01, &|x| with(0, x));
                let (c1, p1) = channel_row(ui, "G", &mut c[1], (0.0, 1.0), 3, 0.01, &|x| with(1, x));
                let (c2, p2) = channel_row(ui, "B", &mut c[2], (0.0, 1.0), 3, 0.01, &|x| with(2, x));
                if c0 || c1 || c2 {
                    *self.color = Color::linear(c[0], c[1], c[2], alpha);
                }
                phases[2] = first_phase(&[p0, p1, p2]);
            }
            _ => {
                let mut l = color_to_lab(*self.color);
                let base = l;
                let with = |i: usize, x: f32| {
                    let mut k = base;
                    k[i] = x;
                    lab_to_color(k, 1.0)
                };
                let (c0, p0) = channel_row(ui, "L", &mut l[0], (0.0, 100.0), 1, 1.0, &|x| with(0, x));
                let (c1, p1) = channel_row(ui, "a", &mut l[1], (-128.0, 127.0), 1, 1.0, &|x| with(1, x));
                let (c2, p2) = channel_row(ui, "b", &mut l[2], (-128.0, 127.0), 1, 1.0, &|x| with(2, x));
                if c0 || c1 || c2 {
                    *self.color = lab_to_color(l, alpha);
                }
                phases[2] = first_phase(&[p0, p1, p2]);
            }
        }

        // Preview, hex and eyedropper.
        ui.begin(Style { gap: m.space_3, align_items: BoxAlign::Center, ..Style::row() });
        let preview = ui.node(Style {
            width: Dim::Px(m.control_height),
            height: Dim::Px(m.control_height),
            shrink: 0.0,
            ..Style::default()
        });
        kit::frame(ui, preview, Area::FULL, self.color.with_alpha(1.0), m.radius_sm, p.border_strong);
        let hex_id = ui.id("Hex");
        if ui.focused() != Some(hex_id) {
            st.hex.clear();
            self.color.with_alpha(1.0).write_hex(&mut st.hex);
        }
        let field = TextField::new("Hex", &mut st.hex).mono(true).max_chars(9).width(110.0).show(ui);
        if field.response.changed
            && st.hex.trim().trim_start_matches('#').len() == 6
            && let Some(c) = Color::parse_hex(st.hex.trim())
        {
            *self.color = c.with_alpha(alpha);
            phases[3] = EditPhase::Step;
        }
        ui.node(Style { grow: 1.0, ..Style::default() });
        if IconButton::new(crate::icons::PIPETTE, "Pick from the viewport").show(ui).clicked {
            out.eyedropper = true;
        }
        ui.end();

        // Swatches.
        if let Some(history) = self.history.as_deref()
            && !history.colors().is_empty()
        {
            ui.begin(Style { gap: m.space_2, wrap: true, ..Style::row() });
            for (i, c) in history.colors().iter().enumerate() {
                let label = kit::fmt64(format_args!("Recent colour {}", i + 1));
                let sid = ui.id("recent").with_index(i as u64);
                if swatch(ui, sid, label.as_str(), *c, 18.0, same_color(*c, *self.color)).clicked {
                    *self.color = c.with_alpha(alpha);
                    phases[4] = EditPhase::Step;
                }
            }
            ui.end();
        }
        if !self.reference.is_empty() {
            ui.begin(Style { gap: m.space_2, wrap: true, align_items: BoxAlign::Center, ..Style::row() });
            for (i, s) in self.reference.iter().enumerate() {
                let sid = ui.id("reference").with_index(i as u64);
                if swatch(ui, sid, s.label, s.color, 22.0, same_color(s.color, *self.color)).clicked {
                    *self.color = s.color.with_alpha(alpha);
                    phases[5] = EditPhase::Step;
                }
            }
            ui.end();
        }
        ui.pop_id();
        ui.end();

        out.changed = *self.color != before;
        if out.changed && !hsv_edited {
            // Set through RGB, Lab, hex or a swatch: re-derive the HSV, keeping the hue where the
            // colour does not define it.
            let h = color_to_hsv(*self.color);
            st.hsv = if h[1] > 0.0 && h[2] > 0.0 { h } else { [st.hsv[0], h[1], h[2]] };
        }
        st.known = Some(self.color.to_srgb8());
        out.phase = first_phase(&phases);
        if matches!(out.phase, EditPhase::Commit | EditPhase::Step)
            && let Some(history) = self.history
        {
            history.push(*self.color);
        }
        *ui.memory::<PickerState>(id) = st;
        ui.semantic(id, Role::Group, self.id_label, SemanticValue::None);
        out
    }
}

/// Whether a colour editor's popup is open, in [`Ui::memory`].
#[derive(Clone, Copy, Debug, Default)]
struct ColorPopup {
    open: bool,
    anchor: Rect,
}

/// A colour editor `[UI-07]`: a swatch with the colour's hex value that opens a [`ColorPicker`] in
/// a popup on click, Enter or Space. In the popup Tab stays inside the picker and Escape (or a
/// click outside) closes it.
pub fn color_edit(
    ui: &mut Ui,
    label: &str,
    color: &mut Color,
    history: Option<&mut SwatchHistory>,
    reference: &[Swatch<'_>],
) -> PickerResponse {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let id = ui.id(label);
    let node =
        ui.node(Style { width: Dim::Px(140.0), height: Dim::Px(m.control_height), shrink: 0.0, ..Style::default() });
    let r = ui.interact(id, node, Sense::CLICK);
    let mut state = *ui.memory::<ColorPopup>(id);
    if r.clicked && !state.open {
        state = ColorPopup { open: true, anchor: r.rect };
    }
    let hot = if state.open { 1.0 } else { r.hover_t };
    kit::frame(
        ui,
        node,
        Area::FULL,
        p.bg.mix(p.hover, hot * 0.6),
        m.radius_sm,
        p.border_strong.mix(p.border_focus, hot),
    );
    kit::fill(ui, node, kit::at(4.0, 4.0, m.control_height - 8.0, m.control_height - 8.0), color.with_alpha(1.0), 3.0);
    let mut hex = kit::Buf::<16>::new();
    let [rr, gg, bb, _] = color.to_srgb8();
    let _ = std::fmt::write(&mut hex, format_args!("#{rr:02X}{gg:02X}{bb:02X}"));
    kit::text(
        ui,
        node,
        kit::inset_x(m.control_height + 2.0, m.space_3),
        hex.as_str(),
        theme.text.mono,
        p.text,
        Align::Start,
    );
    if r.focus_visible {
        kit::ring(ui, node, Area::FULL, m.radius_sm);
    }
    let value = ui.intern(hex.as_str());
    ui.semantic(id, Role::Custom("colorwell"), label, SemanticValue::Text(value));
    let mut out = PickerResponse::default();
    if state.open {
        let vp = ui.viewport();
        let (w, h) = (284.0_f32, 470.0_f32);
        let below = state.anchor.bottom() + 6.0;
        let y = if below + h <= vp.bottom() - 8.0 { below } else { (state.anchor.y - 6.0 - h).max(vp.y + 8.0) };
        let x = state.anchor.x.min(vp.right() - w - 8.0).max(vp.x + 8.0);
        let catcher = ui.begin_overlay(vp, Layer::Popup, Style::default());
        let cr = ui.interact(id.with("outside"), catcher, Sense { click: true, drag: false, focus: false });
        let mut close = cr.pressed;
        ui.end_overlay();
        let card = ui.begin_overlay(
            Rect::new(x.round(), y.round(), w, h),
            Layer::Popup,
            Style { padding: Insets::all(12.0), ..Style::column() },
        );
        kit::card(ui, card, theme.shadows.menu, p.surface, m.radius, 1.0);
        ui.interact(id.with("card"), card, Sense { click: true, drag: false, focus: false });
        ui.begin_semantic_group(id);
        ui.begin_focus_trap();
        ui.push_id(id);
        let mut picker = ColorPicker::new("picker", color).width(w - 24.0).reference(reference);
        if let Some(h) = history {
            picker = picker.history(h);
        }
        out = picker.show(ui);
        ui.pop_id();
        ui.end_focus_trap();
        ui.end_semantic_group();
        ui.end_overlay();
        if ui.consume_key(Key::Escape, Modifiers::NONE) {
            close = true;
        }
        if close || out.eyedropper {
            state.open = false;
            ui.request_focus(id);
        }
    }
    *ui.memory::<ColorPopup>(id) = state;
    out
}

// ---------------------------------------------------------------------------------------------
// Curves and gradients.

/// Most control points of a curve.
pub const MAX_CURVE_POINTS: usize = 32;

/// The value of a curve through `points` (sorted by x, in any range) at `x`: linear between the
/// points, or with `smooth` a monotone cubic (Fritsch–Butland tangents: no overshoot, so a
/// monotone set of points gives a monotone curve). Outside the points the end values hold; an
/// empty curve is 0.
pub fn eval_curve(points: &[[f32; 2]], smooth: bool, x: f32) -> f32 {
    let n = points.len();
    let (Some(first), Some(last)) = (points.first(), points.last()) else {
        return 0.0;
    };
    if n == 1 || !x.is_finite() || x <= first[0] {
        return first[1];
    }
    if x >= last[0] {
        return last[1];
    }
    let k = points.windows(2).position(|w| x < w[1][0]).unwrap_or(n - 2);
    let ([x0, y0], [x1, y1]) = (points[k], points[k + 1]);
    let h = x1 - x0;
    if h <= 0.0 {
        return y1;
    }
    let t = (x - x0) / h;
    if !smooth {
        return y0 + (y1 - y0) * t;
    }
    let slope = |i: usize| {
        let ([ax, ay], [bx, by]) = (points[i], points[i + 1]);
        if bx > ax { (by - ay) / (bx - ax) } else { 0.0 }
    };
    let tangent = |i: usize| -> f32 {
        if i == 0 {
            return slope(0);
        }
        if i == n - 1 {
            return slope(n - 2);
        }
        let (d0, d1) = (slope(i - 1), slope(i));
        if d0 * d1 <= 0.0 {
            return 0.0;
        }
        let (h0, h1) = (points[i][0] - points[i - 1][0], points[i + 1][0] - points[i][0]);
        3.0 * (h0 + h1) / ((2.0 * h1 + h0) / d0 + (h1 + 2.0 * h0) / d1)
    };
    let (m0, m1) = (tangent(k), tangent(k + 1));
    let (t2, t3) = (t * t, t * t * t);
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
}

/// The selected and dragged points of a curve or gradient editor, in [`Ui::memory`].
#[derive(Clone, Copy, Debug, Default)]
struct PointEdit {
    selected: Option<usize>,
    dragging: Option<usize>,
}

/// A curve editor `[UI-07]` for falloff and rig-logic curves: control points in the unit square
/// (`x` 0..1 left to right, `y` 0..1 bottom to top), sorted by x, the first at x = 0 and the last
/// at x = 1, drawn linear or smooth ([`eval_curve`]).
///
/// Pointer: drag a point (it stays between its neighbours; the end points move only up and down),
/// double click on empty space to add one. Keyboard (one Tab stop): Page Up / Page Down select the
/// previous / next point, the arrows move it by 0.01 (Shift: 0.001), Insert adds a point after it,
/// Delete removes it (not the ends).
#[derive(Debug)]
pub struct CurveEditor<'a> {
    label: &'a str,
    points: &'a mut Vec<[f32; 2]>,
    smooth: bool,
    height: f32,
}

impl<'a> CurveEditor<'a> {
    /// An editor for `points`, labelled (and identified) by `label`.
    pub fn new(label: &'a str, points: &'a mut Vec<[f32; 2]>) -> Self {
        Self { label, points, smooth: true, height: 140.0 }
    }

    /// Draw (and evaluate) linear segments instead of the smooth curve.
    #[must_use]
    pub fn smooth(mut self, smooth: bool) -> Self {
        self.smooth = smooth;
        self
    }

    /// The height of the plot.
    #[must_use]
    pub fn height(mut self, height: f32) -> Self {
        if height.is_finite() {
            self.height = height.max(60.0);
        }
        self
    }

    /// Makes the points valid: within the unit square, sorted, ends at x = 0 and 1, at most
    /// [`MAX_CURVE_POINTS`], at least two.
    fn sanitise(points: &mut Vec<[f32; 2]>) {
        points.retain(|p| p[0].is_finite() && p[1].is_finite());
        for p in points.iter_mut() {
            p[0] = p[0].clamp(0.0, 1.0);
            p[1] = p[1].clamp(0.0, 1.0);
        }
        points.sort_by(|a, b| a[0].total_cmp(&b[0]));
        points.truncate(MAX_CURVE_POINTS);
        if points.len() < 2 {
            let y = points.first().map_or(0.0, |p| p[1]);
            points.clear();
            points.extend([[0.0, y], [1.0, y]]);
        }
        if let Some(f) = points.first_mut() {
            f[0] = 0.0;
        }
        if let Some(l) = points.last_mut() {
            l[0] = 1.0;
        }
    }

    /// Adds the editor.
    #[allow(clippy::too_many_lines)]
    pub fn show(self, ui: &mut Ui) -> SliderResponse {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let id = ui.id(self.label);
        let pts = self.points;
        Self::sanitise(pts);
        let before_len = pts.len();
        let before_hash = pts.iter().fold(0_u64, |h, q| {
            h.wrapping_mul(31).wrapping_add(u64::from(q[0].to_bits() ^ q[1].to_bits().rotate_left(7)))
        });
        let node = ui.node(Style { height: Dim::Px(self.height), shrink: 0.0, min_width: 120.0, ..Style::default() });
        let mut r = ui.interact(id, node, Sense::DRAG);
        let mut st = *ui.memory::<PointEdit>(id);
        let plot = Rect::new(
            r.rect.x + CURVE_PAD,
            r.rect.y + CURVE_PAD,
            (r.rect.w - 2.0 * CURVE_PAD).max(1.0),
            (r.rect.h - 2.0 * CURVE_PAD).max(1.0),
        );
        let to_screen = |q: [f32; 2]| Point::new(plot.x + q[0] * plot.w, plot.y + (1.0 - q[1]) * plot.h);
        let from_screen =
            |s: Point| [((s.x - plot.x) / plot.w).clamp(0.0, 1.0), (1.0 - (s.y - plot.y) / plot.h).clamp(0.0, 1.0)];
        let mut stepped = false;
        if r.pressed
            && let Some(pos) = ui.pointer()
        {
            let hit = pts.iter().position(|q| {
                let s = to_screen(*q);
                (s.x - pos.x).hypot(s.y - pos.y) <= 8.0
            });
            st.dragging = hit;
            st.selected = hit.or(st.selected);
            if hit.is_none() && r.double_clicked && pts.len() < MAX_CURVE_POINTS {
                let q = from_screen(pos);
                let at = pts.iter().position(|e| e[0] > q[0]).unwrap_or(pts.len() - 1).max(1);
                pts.insert(at, q);
                st.selected = Some(at);
                stepped = true;
            }
        }
        if r.dragged
            && let (Some(i), Some(pos)) = (st.dragging, ui.pointer())
            && i < pts.len()
        {
            let q = from_screen(pos);
            move_point(pts, i, q);
        }
        if !r.held {
            st.dragging = None;
        }
        if r.focused {
            let n = pts.len();
            if ui.consume_key(Key::PageDown, Modifiers::NONE) {
                st.selected = Some(st.selected.map_or(0, |s| (s + 1).min(n - 1)));
            }
            if ui.consume_key(Key::PageUp, Modifiers::NONE) {
                st.selected = Some(st.selected.map_or(0, |s| s.saturating_sub(1)));
            }
            if let Some(s) = st.selected.filter(|s| *s < n) {
                let mut moved = [0.0_f32; 2];
                for (key, axis, dir) in
                    [(Key::Left, 0, -1.0), (Key::Right, 0, 1.0), (Key::Down, 1, -1.0), (Key::Up, 1, 1.0)]
                {
                    while let Some(mods) = ui.consume_key_any(key) {
                        moved[axis] += dir * if mods.shift { 0.001 } else { 0.01 };
                    }
                }
                if moved != [0.0, 0.0] {
                    let q = pts[s];
                    move_point(pts, s, [q[0] + moved[0], q[1] + moved[1]]);
                    stepped = true;
                }
                if ui.consume_key(Key::Delete, Modifiers::NONE) && s > 0 && s + 1 < n {
                    pts.remove(s);
                    st.selected = Some(s - 1);
                    stepped = true;
                }
                if ui.consume_key(Key::Insert, Modifiers::NONE) && n < MAX_CURVE_POINTS {
                    let next = (s + 1).min(n - 1);
                    let (a, b) = (pts[s.min(next.saturating_sub(1))], pts[next]);
                    let x = f32::midpoint(a[0], b[0]);
                    let q = [x, eval_curve(pts, self.smooth, x)];
                    let at = next.max(1);
                    pts.insert(at, q);
                    st.selected = Some(at);
                    stepped = true;
                }
            }
        }
        if st.selected.is_some_and(|s| s >= pts.len()) {
            st.selected = None;
        }
        *ui.memory::<PointEdit>(id) = st;
        let after_hash = pts.iter().fold(0_u64, |h, q| {
            h.wrapping_mul(31).wrapping_add(u64::from(q[0].to_bits() ^ q[1].to_bits().rotate_left(7)))
        });
        r.changed = after_hash != before_hash || pts.len() != before_len;
        let phase = phase_of(ui, &r, r.changed, stepped);

        // Look: field, grid, curve, points.
        kit::frame(ui, node, Area::FULL, p.bg, m.radius_sm, p.border_strong.mix(p.border_focus, r.hover_t));
        let plot_area = |fx: f32, fy: f32| {
            (
                Anchor { frac: fx, px: CURVE_PAD * (1.0 - 2.0 * fx) },
                Anchor { frac: fy, px: CURVE_PAD * (1.0 - 2.0 * fy) },
            )
        };
        for k in 1..4 {
            #[allow(clippy::cast_precision_loss)]
            let f = k as f32 / 4.0;
            kit::line(ui, node, plot_area(f, 0.0), plot_area(f, 1.0), 1.0, p.border);
            kit::line(ui, node, plot_area(0.0, f), plot_area(1.0, f), 1.0, p.border);
        }
        let mut prev = plot_area(0.0, 1.0 - eval_curve(pts, self.smooth, 0.0));
        for k in 1..=CURVE_SEGMENTS {
            #[allow(clippy::cast_precision_loss)]
            let x = k as f32 / CURVE_SEGMENTS as f32;
            let next = plot_area(x, 1.0 - eval_curve(pts, self.smooth, x));
            kit::line(ui, node, prev, next, 2.0, p.accent);
            prev = next;
        }
        for (i, q) in pts.iter().enumerate() {
            let (cx, cy) = plot_area(q[0], 1.0 - q[1]);
            let sel = st.selected == Some(i);
            let rad = if sel { 5.5 } else { 4.0 };
            ui.paint(node, Paint::Circle { cx, cy, radius: rad + 1.5, color: p.bg });
            ui.paint(node, Paint::Circle { cx, cy, radius: rad, color: if sel { p.text } else { p.text_muted } });
            let label = kit::fmt64(format_args!("{} point {}", self.label, i + 1));
            ui.semantic(
                id.with_index(i as u64),
                Role::Custom("curve.point"),
                label.as_str(),
                SemanticValue::Number { value: f64::from(q[1]), min: 0.0, max: 1.0 },
            );
        }
        if r.focus_visible {
            kit::ring(ui, node, Area::FULL, m.radius_sm);
        }
        ui.semantic(id, Role::Custom("curve"), self.label, SemanticValue::None);
        SliderResponse { response: r, phase }
    }
}

/// Moves point `i` to `q`, keeping it between its neighbours; the end points keep their x.
fn move_point(points: &mut [[f32; 2]], i: usize, q: [f32; 2]) {
    let n = points.len();
    if i >= n {
        return;
    }
    let y = q[1].clamp(0.0, 1.0);
    let x = if i == 0 || i + 1 == n {
        points[i][0]
    } else {
        let gap = 0.001;
        q[0].clamp(points[i - 1][0] + gap, (points[i + 1][0] - gap).max(points[i - 1][0] + gap))
    };
    points[i] = [x, y];
}

/// One stop of a colour gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradientStop {
    /// Position, 0 to 1.
    pub pos: f32,
    /// Colour (linear).
    pub color: Color,
}

/// Most stops of a gradient.
pub const MAX_GRADIENT_STOPS: usize = 16;

/// The colour of a gradient (stops sorted by position) at `t`, interpolated in linear light; the
/// end colours hold outside the stops; an empty gradient is transparent.
pub fn eval_gradient(stops: &[GradientStop], t: f32) -> Color {
    let (Some(first), Some(last)) = (stops.first(), stops.last()) else {
        return Color::TRANSPARENT;
    };
    if !t.is_finite() || t <= first.pos {
        return first.color;
    }
    if t >= last.pos {
        return last.color;
    }
    for w in stops.windows(2) {
        if t < w[1].pos {
            let span = w[1].pos - w[0].pos;
            let f = if span > 0.0 { (t - w[0].pos) / span } else { 1.0 };
            return w[0].color.mix(w[1].color, f);
        }
    }
    last.color
}

/// A gradient editor `[UI-07]`: the ramp with a marker per stop, and a [`color_edit`] for the
/// selected stop.
///
/// Pointer: drag a marker to move its stop, click the ramp to add a stop there (with the colour
/// the gradient has there). Keyboard (one Tab stop): Page Up / Page Down select the previous /
/// next stop, Left / Right move it by 0.01 (Shift: 0.001), Delete removes it (two stops remain at
/// least).
#[derive(Debug)]
pub struct GradientEditor<'a> {
    label: &'a str,
    stops: &'a mut Vec<GradientStop>,
    history: Option<&'a mut SwatchHistory>,
}

impl<'a> GradientEditor<'a> {
    /// An editor for `stops`, labelled (and identified) by `label`.
    pub fn new(label: &'a str, stops: &'a mut Vec<GradientStop>) -> Self {
        Self { label, stops, history: None }
    }

    /// The swatch history for the stop colour editor.
    #[must_use]
    pub fn history(mut self, history: &'a mut SwatchHistory) -> Self {
        self.history = Some(history);
        self
    }

    /// Adds the editor.
    #[allow(clippy::too_many_lines)]
    pub fn show(self, ui: &mut Ui) -> SliderResponse {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let id = ui.id(self.label);
        let stops = self.stops;
        stops.retain(|s| s.pos.is_finite());
        for s in stops.iter_mut() {
            s.pos = s.pos.clamp(0.0, 1.0);
        }
        stops.sort_by(|a, b| a.pos.total_cmp(&b.pos));
        stops.truncate(MAX_GRADIENT_STOPS);
        while stops.len() < 2 {
            let c = stops.first().map_or(Color::BLACK, |s| s.color);
            stops.push(GradientStop { pos: if stops.is_empty() { 0.0 } else { 1.0 }, color: c });
        }
        let before: [(u32, [u8; 4]); MAX_GRADIENT_STOPS] = {
            let mut b = [(0, [0; 4]); MAX_GRADIENT_STOPS];
            for (slot, s) in b.iter_mut().zip(stops.iter()) {
                *slot = (s.pos.to_bits(), s.color.to_srgb8());
            }
            b
        };
        let before_len = stops.len();
        ui.begin(Style { gap: m.space_3, ..Style::column() });
        let node =
            ui.node(Style { height: Dim::Px(RAMP_H + MARK_H), shrink: 0.0, min_width: 120.0, ..Style::default() });
        let mut r = ui.interact(id, node, Sense::DRAG);
        let mut st = *ui.memory::<PointEdit>(id);
        let usable = (r.rect.w - 2.0 * GRADIENT_PAD).max(1.0);
        let pos_at = |x: f32| ((x - r.rect.x - GRADIENT_PAD) / usable).clamp(0.0, 1.0);
        let mut stepped = false;
        if r.pressed
            && let Some(pt) = ui.pointer()
        {
            let hit = stops
                .iter()
                .enumerate()
                .map(|(i, s)| (i, (r.rect.x + GRADIENT_PAD + s.pos * usable - pt.x).abs()))
                .filter(|(_, d)| *d <= 7.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i);
            if let Some(i) = hit {
                st.selected = Some(i);
                st.dragging = Some(i);
            } else if pt.y < r.rect.y + RAMP_H && stops.len() < MAX_GRADIENT_STOPS {
                let t = pos_at(pt.x);
                let color = eval_gradient(stops, t);
                let at = stops.iter().position(|s| s.pos > t).unwrap_or(stops.len());
                stops.insert(at, GradientStop { pos: t, color });
                st.selected = Some(at);
                st.dragging = None;
                stepped = true;
            }
        }
        if r.dragged
            && let (Some(i), Some(pt)) = (st.dragging, ui.pointer())
            && i < stops.len()
        {
            stops[i].pos = pos_at(pt.x);
            let moved = stops[i];
            stops.sort_by(|a, b| a.pos.total_cmp(&b.pos));
            let k = stops.iter().position(|s| *s == moved).unwrap_or(i);
            st.dragging = Some(k);
            st.selected = Some(k);
        }
        if !r.held {
            st.dragging = None;
        }
        if r.focused {
            let n = stops.len();
            if ui.consume_key(Key::PageDown, Modifiers::NONE) {
                st.selected = Some(st.selected.map_or(0, |s| (s + 1).min(n - 1)));
            }
            if ui.consume_key(Key::PageUp, Modifiers::NONE) {
                st.selected = Some(st.selected.map_or(0, |s| s.saturating_sub(1)));
            }
            if let Some(s) = st.selected.filter(|s| *s < n) {
                let d = arrow_steps(ui, true, false);
                if d != 0.0 {
                    stops[s].pos = (stops[s].pos + d * 0.01).clamp(0.0, 1.0);
                    let moved = stops[s];
                    stops.sort_by(|a, b| a.pos.total_cmp(&b.pos));
                    st.selected = stops.iter().position(|x| *x == moved);
                    stepped = true;
                }
                if ui.consume_key(Key::Delete, Modifiers::NONE) && n > 2 {
                    stops.remove(s);
                    st.selected = Some(s.saturating_sub(1));
                    stepped = true;
                }
            }
        }
        if st.selected.is_some_and(|s| s >= stops.len()) {
            st.selected = None;
        }
        *ui.memory::<PointEdit>(id) = st;

        // Look: the ramp (the end colours run out to the frame), its frame, the markers.
        let ramp_area = |a: f32, b: f32| Area {
            x0: Anchor::start(GRADIENT_PAD + a * usable),
            y0: Anchor::start(1.0),
            x1: Anchor::start(GRADIENT_PAD + b * usable),
            y1: Anchor::start(RAMP_H - 1.0),
        };
        let frame_area = kit::at(0.0, 0.0, 0.0, RAMP_H).with_full_width();
        if let (Some(f), Some(l)) = (stops.first(), stops.last()) {
            let head = Area {
                x0: Anchor::start(1.0),
                x1: Anchor::start(GRADIENT_PAD + f.pos * usable),
                ..ramp_area(0.0, 0.0)
            };
            let tail =
                Area { x0: Anchor::start(GRADIENT_PAD + l.pos * usable), x1: Anchor::end(-1.0), ..ramp_area(0.0, 0.0) };
            let r2 = m.radius_sm - 1.0;
            ui.paint(
                node,
                Paint::Rect {
                    area: head,
                    fill: f.color.with_alpha(1.0),
                    radius: Corners { tl: r2, bl: r2, tr: 0.0, br: 0.0 },
                    border: 0.0,
                    border_color: Color::TRANSPARENT,
                },
            );
            ui.paint(
                node,
                Paint::Rect {
                    area: tail,
                    fill: l.color.with_alpha(1.0),
                    radius: Corners { tl: 0.0, bl: 0.0, tr: r2, br: r2 },
                    border: 0.0,
                    border_color: Color::TRANSPARENT,
                },
            );
        }
        for w in stops.windows(2) {
            ui.paint(
                node,
                Paint::Gradient {
                    area: ramp_area(w[0].pos, w[1].pos),
                    from: w[0].color.with_alpha(1.0),
                    to: w[1].color.with_alpha(1.0),
                    radius: Corners::ZERO,
                    axis: Axis::Horizontal,
                },
            );
        }
        kit::frame(
            ui,
            node,
            frame_area,
            Color::TRANSPARENT,
            m.radius_sm,
            p.border_strong.mix(p.border_focus, r.hover_t),
        );
        for (i, s) in stops.iter().enumerate() {
            let x = GRADIENT_PAD + s.pos * usable;
            let sel = st.selected == Some(i);
            let area = Area {
                x0: Anchor::start(x - 5.0),
                y0: Anchor::start(RAMP_H + 2.0),
                x1: Anchor::start(x + 5.0),
                y1: Anchor::start(RAMP_H + MARK_H),
            };
            ui.paint(
                node,
                Paint::Rect {
                    area,
                    fill: s.color.with_alpha(1.0),
                    radius: Corners::all(2.0),
                    border: if sel { 2.0 } else { 1.0 },
                    border_color: if sel { p.text } else { p.border_strong },
                },
            );
            kit::line(
                ui,
                node,
                (Anchor::start(x), Anchor::start(RAMP_H - 4.0)),
                (Anchor::start(x), Anchor::start(RAMP_H + 2.0)),
                1.0,
                if sel { p.text } else { p.text_faint },
            );
            let label = kit::fmt64(format_args!("{} stop {}", self.label, i + 1));
            ui.semantic(
                id.with_index(i as u64),
                Role::Custom("gradient.stop"),
                label.as_str(),
                SemanticValue::Number { value: f64::from(s.pos), min: 0.0, max: 1.0 },
            );
        }
        if r.focus_visible {
            kit::ring(ui, node, kit::at(0.0, 0.0, 0.0, RAMP_H).with_full_width(), m.radius_sm);
        }
        ui.semantic(id, Role::Custom("gradient"), self.label, SemanticValue::None);
        let mut colour_phase = EditPhase::Idle;
        let mut colour_changed = false;
        if let Some(sel) = st.selected.filter(|s| *s < stops.len()) {
            ui.push_id(id);
            ui.begin(Style { gap: m.space_3, align_items: BoxAlign::Center, ..Style::row() });
            let mut pos_text = kit::Buf::<32>::new();
            let _ = std::fmt::write(&mut pos_text, format_args!("Stop {} at {:.2}", sel + 1, stops[sel].pos));
            let tnode = ui.node(Style { width: Dim::Px(110.0), height: Dim::Px(m.control_height), ..Style::default() });
            kit::text(ui, tnode, Area::FULL, pos_text.as_str(), theme.text.small, p.text_muted, Align::Start);
            let pr = color_edit(ui, "Stop colour", &mut stops[sel].color, self.history, &[]);
            colour_phase = pr.phase;
            colour_changed = pr.changed;
            ui.end();
            ui.pop_id();
        }
        ui.end();
        let mut changed = stops.len() != before_len;
        for (s, b) in stops.iter().zip(before.iter()) {
            changed |= (s.pos.to_bits(), s.color.to_srgb8()) != *b;
        }
        r.changed = changed || colour_changed;
        let phase = first_phase(&[phase_of(ui, &r, changed, stepped), colour_phase]);
        SliderResponse { response: r, phase }
    }
}

/// Area helpers.
trait AreaExt {
    fn with_full_width(self) -> Self;
}

impl AreaExt for Area {
    fn with_full_width(mut self) -> Self {
        self.x0 = Anchor::start(0.0);
        self.x1 = Anchor::end(0.0);
        self
    }
}

// ---------------------------------------------------------------------------------------------
// 2D pad and angle dial.

/// A 2D pad `[UI-07]`: a square that sets two values at once (x left to right, y bottom to top).
///
/// Pointer: press and drag to put the handle under the pointer; with Shift the drag moves it at a
/// tenth of the speed; a double click restores the default. Keyboard: Left / Right change x, Up /
/// Down change y by a hundredth of the range (Shift: a thousandth, Page Up / Down: a tenth of y),
/// Delete restores the default.
#[derive(Debug)]
pub struct Pad2D<'a> {
    label: &'a str,
    value: &'a mut [f32; 2],
    x: (f32, f32),
    y: (f32, f32),
    default: Option<[f32; 2]>,
    size: f32,
}

impl<'a> Pad2D<'a> {
    /// A pad for `value` with x in `x.0..=x.1` and y in `y.0..=y.1`.
    pub fn new(label: &'a str, value: &'a mut [f32; 2], x: (f32, f32), y: (f32, f32)) -> Self {
        let order = |r: (f32, f32)| if r.0 <= r.1 { r } else { (r.1, r.0) };
        Self { label, value, x: order(x), y: order(y), default: None, size: 120.0 }
    }

    /// The value a double click or Delete restores.
    #[must_use]
    pub fn default_value(mut self, default: [f32; 2]) -> Self {
        self.default = Some(default);
        self
    }

    /// The side of the square.
    #[must_use]
    pub fn size(mut self, size: f32) -> Self {
        if size.is_finite() {
            self.size = size.max(48.0);
        }
        self
    }

    /// Adds the pad.
    pub fn show(self, ui: &mut Ui) -> SliderResponse {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let id = ui.id(self.label);
        let before = *self.value;
        let (xr, yr) = (self.x.1 - self.x.0, self.y.1 - self.y.0);
        ui.begin(Style { gap: m.space_2, ..Style::column() });
        let node =
            ui.node(Style { width: Dim::Px(self.size), height: Dim::Px(self.size), shrink: 0.0, ..Style::default() });
        let mut r = ui.interact(id, node, Sense::DRAG);
        let mut stepped = false;
        let fine = ui.modifiers().shift;
        if r.pressed
            && !fine
            && let Some(f) = ui.pointer_fraction(r.rect)
        {
            *self.value = [self.x.0 + f.x.clamp(0.0, 1.0) * xr, self.y.0 + (1.0 - f.y.clamp(0.0, 1.0)) * yr];
        }
        if r.dragged {
            if fine {
                let k = 0.1 / r.rect.w.max(1.0);
                self.value[0] += r.drag_delta.x * k * xr;
                self.value[1] -= r.drag_delta.y * k * yr;
            } else if let Some(f) = ui.pointer_fraction(r.rect) {
                *self.value = [self.x.0 + f.x.clamp(0.0, 1.0) * xr, self.y.0 + (1.0 - f.y.clamp(0.0, 1.0)) * yr];
            }
        }
        if r.double_clicked
            && let Some(d) = self.default
        {
            *self.value = d;
            stepped = true;
        }
        if r.focused {
            let dx = arrow_steps(ui, true, false);
            let dy = arrow_steps(ui, false, true);
            if dx != 0.0 || dy != 0.0 {
                self.value[0] += dx * xr * 0.01;
                self.value[1] += dy * yr * 0.01;
                stepped = true;
            }
            if let Some(d) = self.default
                && (ui.consume_key(Key::Delete, Modifiers::NONE) || ui.consume_key(Key::Backspace, Modifiers::NONE))
            {
                *self.value = d;
                stepped = true;
            }
        }
        let clean = |v: f32, lo: f32, hi: f32, old: f32| if v.is_finite() { v.clamp(lo, hi) } else { old };
        *self.value =
            [clean(self.value[0], self.x.0, self.x.1, before[0]), clean(self.value[1], self.y.0, self.y.1, before[1])];
        r.changed = *self.value != before;
        let phase = phase_of(ui, &r, r.changed, stepped);
        if r.dragged {
            ui.set_cursor(CursorIcon::Grabbing);
        }
        // Look.
        kit::frame(
            ui,
            node,
            Area::FULL,
            p.bg,
            m.radius_sm,
            p.border_strong.mix(p.border_focus, r.hover_t.max(r.press_t)),
        );
        let mid = (Anchor::at(0.5, 0.0), Anchor::at(0.5, 0.0));
        kit::line(ui, node, (mid.0, Anchor::start(4.0)), (mid.0, Anchor::end(-4.0)), 1.0, p.border);
        kit::line(ui, node, (Anchor::start(4.0), mid.1), (Anchor::end(-4.0), mid.1), 1.0, p.border);
        let fx = if xr > 0.0 { (self.value[0] - self.x.0) / xr } else { 0.5 };
        let fy = if yr > 0.0 { 1.0 - (self.value[1] - self.y.0) / yr } else { 0.5 };
        let inset = |f: f32| Anchor { frac: f, px: 6.0 * (1.0 - 2.0 * f) };
        let (hx, hy) = (inset(fx), inset(fy));
        kit::line(ui, node, (hx, Anchor::start(1.0)), (hx, Anchor::end(-1.0)), 1.0, p.accent.fade(0.35));
        kit::line(ui, node, (Anchor::start(1.0), hy), (Anchor::end(-1.0), hy), 1.0, p.accent.fade(0.35));
        ui.paint(node, Paint::Circle { cx: hx, cy: hy, radius: 7.0, color: p.bg });
        ui.paint(node, Paint::Circle { cx: hx, cy: hy, radius: 5.5 + r.hover_t, color: p.text });
        if r.focus_visible {
            kit::ring(ui, node, Area::FULL, m.radius_sm);
        }
        let text = kit::fmt64(format_args!("{}  {:.2}, {:.2}", self.label, self.value[0], self.value[1]));
        let caption = ui.node(Style { height: Dim::Px(16.0), width: Dim::Px(self.size), ..Style::default() });
        kit::text(ui, caption, Area::FULL, text.as_str(), theme.text.tiny, p.text_muted, Align::Start);
        ui.end();
        let value = kit::fmt64(format_args!("{:.3}, {:.3}", self.value[0], self.value[1]));
        let value = ui.intern(value.as_str());
        ui.semantic(id, Role::Custom("pad"), self.label, SemanticValue::Text(value));
        SliderResponse { response: r, phase }
    }
}

/// An angle dial `[UI-07]` editing `degrees` in `0..360` (0 points right, angles grow
/// counter-clockwise).
///
/// Pointer: drag around the dial to point it at the pointer (Shift snaps to 15°). Keyboard: Right /
/// Up add a degree, Left / Down subtract one (Shift: a tenth), Page Up / Down 10°, Home sets 0.
pub fn angle_dial(ui: &mut Ui, label: &str, degrees: &mut f32) -> SliderResponse {
    let theme = *ui.theme();
    let (m, p) = (theme.metrics, theme.palette);
    let id = ui.id(label);
    let before = *degrees;
    ui.begin(Style { gap: m.space_4, align_items: BoxAlign::Center, ..Style::row() });
    let node =
        ui.node(Style { width: Dim::Px(DIAL_SIZE), height: Dim::Px(DIAL_SIZE), shrink: 0.0, ..Style::default() });
    let mut r = ui.interact(id, node, Sense::DRAG);
    let mut stepped = false;
    if (r.pressed || r.dragged)
        && let Some(pos) = ui.pointer()
    {
        let c = r.rect.center();
        let (dx, dy) = (pos.x - c.x, c.y - pos.y);
        if dx.hypot(dy) > 2.0 {
            let mut a = dy.atan2(dx).to_degrees().rem_euclid(360.0);
            if ui.modifiers().shift {
                a = ((a / 15.0).round() * 15.0).rem_euclid(360.0);
            }
            *degrees = a;
        }
    }
    if r.focused {
        let d = arrow_steps(ui, true, true);
        if d != 0.0 {
            *degrees += d;
            stepped = true;
        }
        if ui.consume_key(Key::Home, Modifiers::NONE) {
            *degrees = 0.0;
            stepped = true;
        }
    }
    *degrees = if degrees.is_finite() { degrees.rem_euclid(360.0) } else { before };
    // Keep one decimal of precision so float noise does not read as a change.
    *degrees = (*degrees * 10.0).round() / 10.0 % 360.0;
    r.changed = (*degrees - before).abs() > 0.0;
    let phase = phase_of(ui, &r, r.changed, stepped);
    if r.dragged {
        ui.set_cursor(CursorIcon::Grabbing);
    }
    let hot = r.hover_t.max(r.press_t);
    ui.paint(
        node,
        Paint::Rect {
            area: Area::FULL,
            fill: p.surface.mix(p.hover, hot),
            radius: Corners::all(DIAL_SIZE * 0.5),
            border: m.border,
            border_color: p.border_strong,
        },
    );
    let (s, c) = degrees.to_radians().sin_cos();
    let reach = DIAL_SIZE * 0.5 - 6.0;
    let centre = (Anchor::at(0.5, 0.0), Anchor::at(0.5, 0.0));
    let tip = (Anchor::at(0.5, c * reach), Anchor::at(0.5, -s * reach));
    kit::line(
        ui,
        node,
        (Anchor::end(-3.0), Anchor::at(0.5, 0.0)),
        (Anchor::end(-7.0), Anchor::at(0.5, 0.0)),
        1.0,
        p.text_faint,
    );
    kit::line(ui, node, centre, tip, 2.0, p.accent);
    ui.paint(node, Paint::Circle { cx: centre.0, cy: centre.1, radius: 3.0, color: p.accent });
    if r.focus_visible {
        kit::ring(ui, node, Area::FULL, DIAL_SIZE * 0.5);
    }
    let text = kit::fmt64(format_args!("{label}  {:.1}\u{b0}", *degrees));
    let caption = ui.node(Style { grow: 1.0, height: Dim::Px(DIAL_SIZE), ..Style::default() });
    kit::text(ui, caption, Area::FULL, text.as_str(), theme.text.small, p.text, Align::Start);
    ui.end();
    ui.semantic(id, Role::Slider, label, SemanticValue::Number { value: f64::from(*degrees), min: 0.0, max: 360.0 });
    SliderResponse { response: r, phase }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use humanart_core::rng::Rng;

    use super::*;
    use crate::input::{Event, PointerButton};
    use crate::tests::Driver;
    use crate::theme::Theme;
    use crate::widgets::{Slider, SliderPair};

    /// Neutral test swatches (not reference data: their values only exercise the widget).
    const REFERENCE: [Swatch<'static>; 3] = [
        Swatch { label: "Swatch one", color: Color::linear(0.6, 0.4, 0.3, 1.0) },
        Swatch { label: "Swatch two", color: Color::linear(0.3, 0.18, 0.12, 1.0) },
        Swatch { label: "Swatch three", color: Color::linear(0.08, 0.05, 0.03, 1.0) },
    ];

    /// The state the editors gallery edits.
    struct Model {
        scalar: f32,
        position: [f32; 3],
        lo: f32,
        hi: f32,
        weight: f32,
        locked: bool,
        brows: (f32, f32),
        linked: bool,
        curve: Vec<[f32; 2]>,
        stops: Vec<GradientStop>,
        gaze: [f32; 2],
        angle: f32,
        color: Color,
        history: SwatchHistory,
        last: PickerResponse,
        eyedropper: u32,
    }

    impl Model {
        fn new() -> Self {
            Self {
                scalar: 1.5,
                position: [0.0, 1.62, -0.05],
                lo: 0.2,
                hi: 0.7,
                weight: 0.8,
                locked: false,
                brows: (0.3, -0.3),
                linked: true,
                curve: vec![[0.0, 1.0], [0.5, 0.8], [1.0, 0.0]],
                stops: vec![
                    GradientStop { pos: 0.0, color: Color::hex(0x2B1B14) },
                    GradientStop { pos: 0.6, color: Color::hex(0xB07856) },
                    GradientStop { pos: 1.0, color: Color::hex(0xF1D3C0) },
                ],
                gaze: [0.25, -0.4],
                angle: 30.0,
                color: Color::hex(0xC68863),
                history: SwatchHistory::new(),
                last: PickerResponse::default(),
                eyedropper: 0,
            }
        }
    }

    /// Every editor of this module in a 700 × 720 window.
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
        ui.begin(Style { width: Dim::Px(360.0), gap: m.space_4, ..Style::column() });
        DragValue::new("Scale", &mut model.scalar).range(0.0, 10.0).speed(0.05).decimals(2).default_value(1.0).show(ui);
        vector_edit(ui, "Position", &mut model.position, 0.01, 3);
        RangeSlider::new("Falloff", (&mut model.lo, &mut model.hi), 0.0, 1.0).step(0.05).show(ui);
        Slider::new("Weight", &mut model.weight, 0.0, 1.5).soft_range(0.0, 1.0).lock(&mut model.locked).show(ui);
        SliderPair::new(
            "Brows",
            ["Left", "Right"],
            (&mut model.brows.0, &mut model.brows.1),
            -1.0,
            1.0,
            &mut model.linked,
        )
        .mirror(true)
        .show(ui);
        CurveEditor::new("Falloff curve", &mut model.curve).height(120.0).show(ui);
        GradientEditor::new("Ramp", &mut model.stops).history(&mut model.history).show(ui);
        ui.begin(Style { gap: m.space_6, align_items: BoxAlign::Start, ..Style::row() });
        Pad2D::new("Gaze", &mut model.gaze, (-1.0, 1.0), (-1.0, 1.0)).default_value([0.0, 0.0]).size(110.0).show(ui);
        angle_dial(ui, "Rotation", &mut model.angle);
        ui.end();
        ui.end();
        let out = ColorPicker::new("Skin tone", &mut model.color)
            .history(&mut model.history)
            .reference(&REFERENCE)
            .width(270.0)
            .show(ui);
        if out.eyedropper {
            model.eyedropper += 1;
        }
        if out.phase != EditPhase::Idle || out.changed {
            model.last = out;
        }
        ui.end();
    }

    fn setup(theme: &Theme) -> Driver {
        Driver::new(theme, 700.0, 720.0)
    }

    fn focus(d: &mut Driver, build: &mut dyn FnMut(&mut Ui), label: &str) {
        for _ in 0..60 {
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
        let mid = Point::new(f32::midpoint(from.x, to.x), f32::midpoint(from.y, to.y));
        d.frame(vec![Event::PointerMoved(mid)], build);
        d.frame(vec![Event::PointerMoved(to)], build);
        d.frame(vec![Event::Button { button: PointerButton::Primary, pressed: false, pos: to }], build);
    }

    fn number(d: &Driver, label: &str) -> f64 {
        match d.semantic(label).map(|s| s.value) {
            Some(SemanticValue::Number { value, .. }) => value,
            other => panic!("{label}: {other:?}"),
        }
    }

    #[test]
    fn colour_models_round_trip() {
        let mut rng = Rng::new(0x00C0_10A5);
        for _ in 0..2000 {
            let c = Color::linear(rng.f32(), rng.f32(), rng.f32(), 1.0);
            let back = hsv_to_color(color_to_hsv(c), 1.0);
            let lab = lab_to_color(color_to_lab(c), 1.0);
            for (a, b) in [(c.r, back.r), (c.g, back.g), (c.b, back.b), (c.r, lab.r), (c.g, lab.g), (c.b, lab.b)] {
                assert!((a - b).abs() < 1e-4, "{c:?} -> {back:?} / {lab:?}");
            }
        }
        let white = color_to_lab(Color::WHITE);
        assert!((white[0] - 100.0).abs() < 0.01 && white[1].abs() < 0.01 && white[2].abs() < 0.01);
        let red = color_to_hsv(Color::hex(0xFF0000));
        assert!(red[0].abs() < 1e-4 && (red[1] - 1.0).abs() < 1e-4 && (red[2] - 1.0).abs() < 1e-4, "{red:?}");
    }

    #[test]
    fn curves_interpolate_without_overshoot() {
        // Linear and smooth pass through the points; a monotone set gives a monotone curve within
        // its range (Fritsch-Butland tangents), for 500 random curves.
        let mut rng = Rng::new(0xC0_4E5);
        for _ in 0..500 {
            let n = 2 + rng.index(8);
            let mut pts: Vec<[f32; 2]> = (0..n).map(|_| [rng.f32(), rng.f32()]).collect();
            pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
            pts.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-3);
            let mut ys: Vec<f32> = pts.iter().map(|p| p[1]).collect();
            ys.sort_by(f32::total_cmp);
            for (p, y) in pts.iter_mut().zip(&ys) {
                p[1] = *y;
            }
            for p in &pts {
                assert!((eval_curve(&pts, true, p[0]) - p[1]).abs() < 1e-4);
                assert!((eval_curve(&pts, false, p[0]) - p[1]).abs() < 1e-4);
            }
            let (lo, hi) = (ys[0], ys[ys.len() - 1]);
            let mut prev = f32::MIN;
            for k in 0..=400 {
                let y = eval_curve(&pts, true, k as f32 / 400.0);
                assert!(y >= prev - 1e-5, "not monotone: {pts:?}");
                assert!(y >= lo - 1e-5 && y <= hi + 1e-5, "overshoot: {pts:?}");
                prev = y;
            }
        }
        assert_eq!(eval_curve(&[], true, 0.5), 0.0);
        assert_eq!(eval_curve(&[[0.3, 0.7]], true, 0.9), 0.7);
        let g = [GradientStop { pos: 0.2, color: Color::BLACK }, GradientStop { pos: 0.6, color: Color::WHITE }];
        assert_eq!(eval_gradient(&g, 0.0), Color::BLACK);
        assert_eq!(eval_gradient(&g, 0.9), Color::WHITE);
        assert!((eval_gradient(&g, 0.4).r - 0.5).abs() < 1e-6);
        let mut h = SwatchHistory::new();
        for i in 0..20_u32 {
            h.push(Color::hex(i * 0x0A0A0A));
        }
        h.push(Color::hex(0x0A0A0A * 15));
        assert_eq!(h.colors().len(), SwatchHistory::MAX);
        assert_eq!(h.colors()[0], Color::hex(0x0A0A0A * 15), "re-used colours move to the front");
        assert_eq!(h.colors()[1], Color::hex(0x0A0A0A * 19));
    }

    #[test]
    fn ui_07_drag_value_drags_steps_and_takes_numeric_entry() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        let c = d.rect_of("Scale").center();
        drag(&mut d, &mut build, c, Point::new(c.x + 20.0, c.y));
        let v = model.borrow().scalar;
        assert!((v - 2.5).abs() < 0.26, "20 px at 0.05 per px: {v}");
        // Keyboard: Delete restores the default, arrows step by ten times the speed.
        focus(&mut d, &mut build, "Scale");
        d.key(Key::Delete, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().scalar, 1.0);
        d.key(Key::Right, Modifiers::NONE, &mut build);
        assert!((model.borrow().scalar - 1.5).abs() < 1e-5);
        // Enter opens numeric entry; typing replaces the selected text; Enter applies.
        d.key(Key::Enter, Modifiers::NONE, &mut build);
        d.frame(vec![], &mut build);
        d.key(Key::Letter('A'), Modifiers::CTRL, &mut build);
        d.frame(vec![Event::Modifiers(Modifiers::NONE), Event::Text('4'), Event::Text('2')], &mut build);
        d.key(Key::Enter, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().scalar, 10.0, "clamped to the range");
        assert_eq!(d.focused_label(), "Scale", "the focus returns to the box");
        // Escape cancels an entry.
        d.key(Key::Enter, Modifiers::NONE, &mut build);
        d.frame(vec![], &mut build);
        d.frame(vec![Event::Text('7')], &mut build);
        d.key(Key::Escape, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().scalar, 10.0);
    }

    #[test]
    fn ui_07_vector_range_lock_and_pair() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        for label in ["Position X", "Position Y", "Position Z"] {
            assert!(d.semantic(label).is_some(), "{label}");
        }
        focus(&mut d, &mut build, "Position Y");
        d.key(Key::Up, Modifiers::SHIFT, &mut build);
        assert!((model.borrow().position[1] - 1.63).abs() < 1e-5, "{:?}", model.borrow().position);
        // Range slider: the thumbs are Tab stops and cannot cross.
        focus(&mut d, &mut build, "Falloff min");
        for _ in 0..3 {
            d.key(Key::PageUp, Modifiers::NONE, &mut build);
        }
        assert_eq!((model.borrow().lo, model.borrow().hi), (0.7, 0.7));
        d.key(Key::Tab, Modifiers::NONE, &mut build);
        assert_eq!(d.focused_label(), "Falloff max");
        d.key(Key::End, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().hi, 1.0);
        // A press on the track moves the nearer thumb there.
        d.frame(vec![Event::Modifiers(Modifiers::NONE)], &mut build);
        let lo = d.rect_of("Falloff min");
        let track_x = lo.x - (lo.w * 0.5) - 40.0;
        d.click(Point::new(track_x.max(lo.x - 200.0), lo.center().y), &mut build);
        assert!(model.borrow().lo < 0.7, "{}", model.borrow().lo);
        // The slider's lock freezes it; the pair mirrors.
        d.click(d.rect_of("Lock").center(), &mut build);
        assert!(model.borrow().locked);
        let w = d.rect_of("Weight");
        d.click(Point::new(w.x + 4.0, w.center().y), &mut build);
        assert_eq!(model.borrow().weight, 0.8, "a locked slider does not move");
        assert!(d.semantic("Weight").is_some_and(|s| s.disabled));
        focus(&mut d, &mut build, "Left");
        d.key(Key::End, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().brows, (1.0, -1.0));
    }

    #[test]
    fn ui_07_colour_picker_square_hex_models_swatches_and_history() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        let start = model.borrow().color;
        let hsv0 = color_to_hsv(start);
        focus(&mut d, &mut build, "Saturation and value");
        d.key(Key::Left, Modifiers::NONE, &mut build);
        let hsv1 = color_to_hsv(model.borrow().color);
        assert!((hsv1[1] - (hsv0[1] - 0.01)).abs() < 0.004, "{hsv0:?} -> {hsv1:?}");
        assert!((hsv1[0] - hsv0[0]).abs() < 0.5, "the hue holds");
        assert_eq!(model.borrow().last.phase, EditPhase::Step);
        assert_eq!(model.borrow().history.colors().len(), 1, "a step is recorded");
        // Hex entry applies as typed.
        d.click(d.rect_of("Hex").center(), &mut build);
        d.key(Key::Letter('A'), Modifiers::CTRL, &mut build);
        let typed: Vec<Event> =
            std::iter::once(Event::Modifiers(Modifiers::NONE)).chain("#3366CC".chars().map(Event::Text)).collect();
        d.frame(typed, &mut build);
        assert_eq!(model.borrow().color.to_srgb8(), [0x33, 0x66, 0xCC, 255]);
        // The Lab model shows CIELAB numbers.
        d.click(d.rect_of("Lab").center(), &mut build);
        d.frame(vec![], &mut build);
        let l = number(&d, "L");
        let expect = f64::from(color_to_lab(model.borrow().color)[0]);
        assert!((l - expect).abs() < 0.01, "{l} vs {expect}");
        // A drag on the hue bar is one gesture: Begin ... Commit, recorded once.
        let before = model.borrow().history.colors().len();
        let hue = d.rect_of("Hue");
        drag(
            &mut d,
            &mut build,
            Point::new(hue.x + 5.0, hue.center().y),
            Point::new(hue.x + hue.w * 0.33, hue.center().y),
        );
        assert_eq!(model.borrow().last.phase, EditPhase::Commit);
        assert_eq!(model.borrow().history.colors().len(), before + 1);
        let h = color_to_hsv(model.borrow().color)[0];
        assert!((h - 120.0).abs() < 8.0, "green-ish hue: {h}");
        // Reference swatches and the eyedropper.
        d.click(d.rect_of("Swatch two").center(), &mut build);
        assert_eq!(model.borrow().color.to_srgb8(), REFERENCE[1].color.to_srgb8());
        d.click(d.rect_of("Pick from the viewport").center(), &mut build);
        assert_eq!(model.borrow().eyedropper, 1);
        // A recent swatch brings an older colour back.
        let recent = model.borrow().history.colors()[1];
        d.click(d.rect_of("Recent colour 2").center(), &mut build);
        assert_eq!(model.borrow().color.to_srgb8(), recent.to_srgb8());
    }

    #[test]
    fn ui_07_curve_editor_adds_moves_and_deletes_points() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        let r = d.rect_of("Falloff curve");
        let plot = Rect::new(r.x + CURVE_PAD, r.y + CURVE_PAD, r.w - 2.0 * CURVE_PAD, r.h - 2.0 * CURVE_PAD);
        let at = |x: f32, y: f32| Point::new(plot.x + x * plot.w, plot.y + (1.0 - y) * plot.h);
        // Double click on empty space adds a point there.
        d.click(at(0.25, 0.3), &mut build);
        d.click(at(0.25, 0.3), &mut build);
        assert_eq!(model.borrow().curve.len(), 4);
        let added = model.borrow().curve[1];
        assert!((added[0] - 0.25).abs() < 0.01 && (added[1] - 0.3).abs() < 0.01, "{added:?}");
        // Dragging the middle point past its neighbour stops at the neighbour.
        drag(&mut d, &mut build, at(0.5, 0.8), at(0.95, 0.5));
        let c = model.borrow().curve.clone();
        assert!(c[2][0] < 1.0 && c[2][0] > 0.9 && (c[2][1] - 0.5).abs() < 0.01, "{c:?}");
        // An end point moves only up and down.
        drag(&mut d, &mut build, at(1.0, 0.0), at(0.7, 0.2));
        let c = model.borrow().curve.clone();
        assert_eq!(c[3][0], 1.0);
        assert!((c[3][1] - 0.2).abs() < 0.01);
        // Keyboard: select the second point, move it, delete it; the ends cannot be deleted.
        focus(&mut d, &mut build, "Falloff curve");
        d.key(Key::PageUp, Modifiers::NONE, &mut build);
        d.key(Key::PageUp, Modifiers::NONE, &mut build);
        d.key(Key::PageUp, Modifiers::NONE, &mut build);
        d.key(Key::PageDown, Modifiers::NONE, &mut build);
        d.key(Key::Up, Modifiers::NONE, &mut build);
        assert!((model.borrow().curve[1][1] - 0.31).abs() < 0.011);
        d.key(Key::Delete, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().curve.len(), 3);
        d.key(Key::PageUp, Modifiers::NONE, &mut build);
        d.key(Key::Delete, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().curve.len(), 3, "the first point stays");
        d.key(Key::Insert, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().curve.len(), 4);
    }

    #[test]
    fn ui_07_gradient_editor_adds_moves_and_removes_stops() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        let r = d.rect_of("Ramp");
        let x_at = |t: f32| r.x + GRADIENT_PAD + t * (r.w - 2.0 * GRADIENT_PAD);
        let expected = eval_gradient(&model.borrow().stops, 0.3);
        d.click(Point::new(x_at(0.3), r.y + 10.0), &mut build);
        assert_eq!(model.borrow().stops.len(), 4);
        let added = model.borrow().stops[1];
        assert!((added.pos - 0.3).abs() < 0.01);
        assert_eq!(added.color.to_srgb8(), expected.to_srgb8(), "the new stop keeps the gradient's colour");
        // Drag the marker of the stop at 0.6 to 0.9.
        drag(&mut d, &mut build, Point::new(x_at(0.6), r.y + 30.0), Point::new(x_at(0.9), r.y + 30.0));
        assert!((model.borrow().stops[2].pos - 0.9).abs() < 0.01, "{:?}", model.borrow().stops);
        assert!(d.semantic("Stop colour").is_some(), "the selected stop has a colour editor");
        focus(&mut d, &mut build, "Ramp");
        d.key(Key::Left, Modifiers::NONE, &mut build);
        assert!((model.borrow().stops[2].pos - 0.89).abs() < 0.002);
        for _ in 0..4 {
            d.key(Key::Delete, Modifiers::NONE, &mut build);
        }
        assert_eq!(model.borrow().stops.len(), 2, "two stops remain");
    }

    #[test]
    fn ui_07_pad_and_dial_follow_pointer_and_keys() {
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = setup(&Theme::dark());
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        let pad = d.rect_of("Gaze");
        drag(&mut d, &mut build, pad.center(), Point::new(pad.right() - 1.0, pad.y + 1.0));
        let g = model.borrow().gaze;
        assert!(g[0] > 0.95 && g[1] > 0.95, "{g:?}");
        focus(&mut d, &mut build, "Gaze");
        d.key(Key::Delete, Modifiers::NONE, &mut build);
        assert_eq!(model.borrow().gaze, [0.0, 0.0]);
        d.key(Key::Left, Modifiers::NONE, &mut build);
        d.key(Key::Up, Modifiers::NONE, &mut build);
        let g = model.borrow().gaze;
        assert!((g[0] + 0.02).abs() < 1e-5 && (g[1] - 0.02).abs() < 1e-5, "{g:?}");
        // The dial points at the pointer: straight up is 90 degrees.
        let dial = d.rect_of("Rotation");
        let c = dial.center();
        drag(&mut d, &mut build, Point::new(c.x + 15.0, c.y), Point::new(c.x, c.y - 15.0));
        assert!((model.borrow().angle - 90.0).abs() < 0.2, "{}", model.borrow().angle);
        focus(&mut d, &mut build, "Rotation");
        d.key(Key::Down, Modifiers::NONE, &mut build);
        assert!((model.borrow().angle - 89.0).abs() < 1e-3);
        d.key(Key::Home, Modifiers::NONE, &mut build);
        d.key(Key::Left, Modifiers::NONE, &mut build);
        assert!((model.borrow().angle - 359.0).abs() < 1e-3, "wraps");
    }

    #[test]
    fn ui_09_colour_editor_popup_traps_tab_and_closes_on_escape() {
        let color = RefCell::new(Color::hex(0x804020));
        let mut build = |ui: &mut Ui| {
            ui.begin(Style { padding: Insets::all(10.0), gap: 8.0, ..Style::column() });
            color_edit(ui, "Tint", &mut color.borrow_mut(), None, &[]);
            crate::widgets::button(ui, "After");
            ui.end();
        };
        let mut d = Driver::new(&Theme::dark(), 400.0, 600.0);
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        focus(&mut d, &mut build, "Tint");
        d.key(Key::Enter, Modifiers::NONE, &mut build);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Hue").is_some(), "the picker is open");
        for _ in 0..30 {
            d.key(Key::Tab, Modifiers::NONE, &mut build);
            assert_ne!(d.focused_label(), "After", "Tab left the popup");
            assert_ne!(d.focused_label(), "Tint");
        }
        d.key(Key::Escape, Modifiers::NONE, &mut build);
        d.frame(vec![], &mut build);
        assert!(d.semantic("Hue").is_none());
        assert_eq!(d.focused_label(), "Tint");
    }

    #[test]
    fn ac_wp12_2_editors_golden_at_100_150_200_percent() {
        // [UI-07] [AC-WP12-2] Drag values, vector, range slider, slider with soft range and lock,
        // linked pair, curve, gradient, pad, dial and the colour picker with history and swatches.
        for (scale, name) in [(1.0, "editors_100"), (1.5, "editors_150"), (2.0, "editors_200")] {
            let model = RefCell::new(Model::new());
            model.borrow_mut().history.push(Color::hex(0x8D5524));
            model.borrow_mut().history.push(Color::hex(0xE0AC69));
            let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
            let mut d = Driver::new(&Theme::dark().with_reduced_motion(true), 700.0, 720.0);
            d.scale = scale;
            d.frame(vec![], &mut build);
            d.frame(vec![], &mut build);
            focus(&mut d, &mut build, "Falloff min");
            d.frame(vec![], &mut build);
            d.golden(name);
        }
        let model = RefCell::new(Model::new());
        let mut build = |ui: &mut Ui| gallery(ui, &mut model.borrow_mut());
        let mut d = Driver::new(&Theme::light().with_reduced_motion(true), 700.0, 720.0);
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        d.golden("editors_light");
    }
}
