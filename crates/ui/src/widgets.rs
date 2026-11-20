//! The basic widgets `[UI-07]`: label, button, slider (numeric entry, fine adjust, reset, lock,
//! soft range, undo phases), linked / mirrored slider pair, single-line text field and multi-line
//! text area, with keyboard operation and focus rings `[UI-09]` and semantic descriptions. The
//! other widget groups are in `controls`, `popup`, `editors`, `views` and `dock`.
//!
//! # Writing a widget (the extension point for the rest of UI-07 and docking)
//!
//! A widget is a function (or a builder with a `show` method) over [`Ui`] that, every frame:
//!
//! 1. derives its [`crate::ui::Id`] from a label with [`Ui::id`] (inside the caller's id scope);
//! 2. adds its box (or a subtree, with [`Ui::begin`] / [`Ui::end`]) with a layout
//!    [`Style`] sized from the theme's metrics;
//! 3. calls [`Ui::interact`] with the [`Sense`] it needs and reacts to the [`Response`]; keyboard
//!    handling happens only while `response.focused`, through [`Ui::consume_key`];
//! 4. records its look as [`Paint`] operations relative to its box (colours, radii and text
//!    styles from [`Ui::theme`], never literals), including [`Paint::FocusRing`] when
//!    `response.focus_visible`;
//! 5. describes itself with [`Ui::semantic`] (role, label, value);
//! 6. keeps state that must outlive the frame in [`Ui::memory`] keyed by its id.
//!
//! Nothing here allocates per frame once the widgets' texts are in the layout cache.

use crate::color::Color;
use crate::geom::{Insets, Point, Rect};
use crate::input::{Key, Modifiers};
use crate::layout::{Align as BoxAlign, Dim, NodeId, Style};
use crate::paint::Corners;
use crate::text::Align;
use crate::theme::TextStyle;
use crate::ui::{Anchor, Area, CursorIcon, Paint, Response, Role, SemanticValue, Sense, Ui};

/// A line of text in the theme's body style and primary text colour.
pub fn label(ui: &mut Ui, text: &str) -> NodeId {
    let style = ui.theme().text.body;
    let color = ui.theme().palette.text;
    label_styled(ui, text, style, color)
}

/// A line of text in `style` and `color`.
pub fn label_styled(ui: &mut Ui, text: &str, style: TextStyle, color: Color) -> NodeId {
    let size = ui.measure_text(text, style);
    let node = ui.node(Style::leaf(size));
    let t = ui.intern(text);
    ui.paint(node, Paint::Text { area: Area::FULL, text: t, style, color, align: Align::Start, wrap: false });
    let id = ui.id(text);
    ui.semantic(id, Role::Label, text, SemanticValue::None);
    ui.semantic_node(node);
    node
}

/// The look of a button.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ButtonKind {
    /// The accent fill: the main action of a panel or dialog.
    Primary,
    /// A raised surface with a border.
    #[default]
    Secondary,
    /// No fill until hovered (toolbars).
    Ghost,
    /// A destructive action.
    Danger,
}

/// A push button.
#[derive(Clone, Copy, Debug)]
pub struct Button<'a> {
    label: &'a str,
    kind: ButtonKind,
    icon: Option<char>,
    width: Option<f32>,
    enabled: bool,
}

impl<'a> Button<'a> {
    /// A secondary button with a text label (also its id).
    pub fn new(label: &'a str) -> Self {
        Self { label, kind: ButtonKind::Secondary, icon: None, width: None, enabled: true }
    }

    /// The look.
    #[must_use]
    pub fn kind(mut self, kind: ButtonKind) -> Self {
        self.kind = kind;
        self
    }

    /// An icon (a code point of the icon face) before the label.
    #[must_use]
    pub fn icon(mut self, icon: char) -> Self {
        self.icon = Some(icon);
        self
    }

    /// A fixed width instead of the content's.
    #[must_use]
    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width);
        self
    }

    /// Whether it can be used.
    #[must_use]
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Adds the button; `clicked` on a click or Enter / Space while focused.
    pub fn show(self, ui: &mut Ui) -> Response {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let style = theme.text.label;
        let text = ui.measure_text(self.label, style);
        let icon_w =
            if self.icon.is_some() { m.icon + if self.label.is_empty() { 0.0 } else { m.space_3 } } else { 0.0 };
        let content_w = text.w + icon_w;
        let width =
            self.width.unwrap_or(if self.label.is_empty() { m.icon_button } else { content_w + 2.0 * m.control_pad_x });
        let node = ui.node(Style { width: Dim::Px(width), height: Dim::Px(m.control_height), ..Style::default() });
        let id = ui.id(self.label);
        let sense = if self.enabled { Sense::CLICK } else { Sense::HOVER };
        let mut r = ui.interact(id, node, sense);
        if !self.enabled {
            r.clicked = false;
        }
        let (h, pr) = (r.hover_t, r.press_t);
        let (fill, border, ink) = match self.kind {
            ButtonKind::Primary => (p.accent.fade(1.0 - 0.12 * h - 0.1 * pr), Color::TRANSPARENT, p.on_accent),
            ButtonKind::Secondary => (p.surface.mix(p.hover, h).mix(p.active, pr), p.border_strong, p.text),
            ButtonKind::Ghost => (p.hover.fade(h).mix(p.active, pr), Color::TRANSPARENT, p.text_muted.mix(p.text, h)),
            ButtonKind::Danger => (p.danger.fade(1.0 - 0.12 * h - 0.1 * pr), Color::TRANSPARENT, p.on_accent),
        };
        let alpha = if self.enabled { 1.0 } else { 0.45 };
        let radius = Corners::all(m.radius_sm);
        let border_w = if border.a > 0.0 { m.border } else { 0.0 };
        ui.paint(
            node,
            Paint::Rect {
                area: Area::FULL,
                fill: fill.fade(alpha),
                radius,
                border: border_w,
                border_color: border.fade(alpha),
            },
        );
        // Content centred: icon, gap, label.
        let start = (width - content_w) * 0.5;
        if let Some(icon) = self.icon {
            let mut buf = [0_u8; 4];
            let t = ui.intern(icon.encode_utf8(&mut buf));
            let area = Area {
                x0: Anchor::start(start),
                y0: Anchor::start(0.0),
                x1: Anchor::start(start + m.icon),
                y1: Anchor::end(0.0),
            };
            ui.paint(
                node,
                Paint::Text {
                    area,
                    text: t,
                    style: TextStyle::icon(m.icon),
                    color: ink.fade(alpha),
                    align: Align::Start,
                    wrap: false,
                },
            );
        }
        if !self.label.is_empty() {
            let t = ui.intern(self.label);
            let area = Area {
                x0: Anchor::start(start + icon_w),
                y0: Anchor::start(0.0),
                x1: Anchor::end(0.0),
                y1: Anchor::end(0.0),
            };
            ui.paint(
                node,
                Paint::Text { area, text: t, style, color: ink.fade(alpha), align: Align::Start, wrap: false },
            );
        }
        if r.focus_visible {
            ui.paint(node, Paint::FocusRing { area: Area::FULL, radius: m.radius_sm });
        }
        ui.semantic(id, Role::Button, self.label, SemanticValue::None);
        if !self.enabled {
            ui.semantic_disabled();
        }
        r
    }
}

/// A secondary button.
pub fn button(ui: &mut Ui, label: &str) -> Response {
    Button::new(label).show(ui)
}

/// Where a value edit stands, for undo integration `[UI-07]`: the application records one undo
/// step per [`EditPhase::Commit`] or [`EditPhase::Step`], with the value it saw at
/// [`EditPhase::Begin`] (or just before the step) as the old value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EditPhase {
    /// Nothing is being edited.
    #[default]
    Idle,
    /// A pointer gesture started (the value may already have changed).
    Begin,
    /// The value changed during a gesture.
    Update,
    /// The gesture ended: one undo step from the value at `Begin` to the current one.
    Commit,
    /// A complete change in one frame (keyboard, reset, numeric entry): one undo step.
    Step,
}

/// What a slider did this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SliderResponse {
    /// The track's interaction (`changed` when the value changed).
    pub response: Response,
    /// Undo integration.
    pub phase: EditPhase,
}

/// The numeric-entry state of a slider, in [`Ui::memory`].
#[derive(Clone, Debug, Default)]
struct SliderEntry {
    editing: bool,
    text: String,
    just_opened: bool,
}

/// A horizontal slider with a label and its value `[UI-07]`.
///
/// Pointer: a press jumps to the pointer, dragging moves relatively; Shift while dragging is the
/// fine-adjust modifier (a tenth of the speed); a double click on the track resets to the default;
/// a double click on the value opens numeric entry. Keyboard (when focused): Left / Down and Right
/// / Up step (Shift: a tenth of a step), Page Up / Down ten steps, Home / End the ends, Backspace /
/// Delete the default, Enter or F2 numeric entry (Enter applies, Escape cancels). A soft range is
/// drawn as a band on the track (values outside it show in the warning colour); a lock toggle
/// freezes the value. [`SliderResponse::phase`] integrates with undo.
#[derive(Debug)]
pub struct Slider<'a> {
    label: &'a str,
    value: &'a mut f32,
    min: f32,
    max: f32,
    step: Option<f32>,
    default: Option<f32>,
    decimals: usize,
    label_width: f32,
    soft: Option<(f32, f32)>,
    lock: Option<&'a mut bool>,
}

impl<'a> Slider<'a> {
    /// A slider for `value` in `min..=max`, labelled (and identified) by `label`.
    pub fn new(label: &'a str, value: &'a mut f32, min: f32, max: f32) -> Self {
        let (min, max) = if min <= max { (min, max) } else { (max, min) };
        Self {
            label,
            value,
            min,
            max,
            step: None,
            default: None,
            decimals: 2,
            label_width: 96.0,
            soft: None,
            lock: None,
        }
    }

    /// Snaps values to multiples of `step` from `min` (also the keyboard step).
    #[must_use]
    pub fn step(mut self, step: f32) -> Self {
        self.step = (step.is_finite() && step > 0.0).then_some(step);
        self
    }

    /// The value a double click or Delete restores.
    #[must_use]
    pub fn default_value(mut self, default: f32) -> Self {
        self.default = Some(default);
        self
    }

    /// Decimals shown.
    #[must_use]
    pub fn decimals(mut self, decimals: usize) -> Self {
        self.decimals = decimals.min(6);
        self
    }

    /// The width of the label column (0 for no label).
    #[must_use]
    pub fn label_width(mut self, width: f32) -> Self {
        self.label_width = width.max(0.0);
        self
    }

    /// The soft (typical) range inside the hard range, drawn on the track.
    #[must_use]
    pub fn soft_range(mut self, lo: f32, hi: f32) -> Self {
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        self.soft = (lo.is_finite() && hi.is_finite()).then_some((lo.max(self.min), hi.min(self.max)));
        self
    }

    /// A lock toggle after the value: while `locked` the value cannot change.
    #[must_use]
    pub fn lock(mut self, locked: &'a mut bool) -> Self {
        self.lock = Some(locked);
        self
    }

    fn quantize(&self, v: f32) -> f32 {
        let v = if v.is_finite() { v.clamp(self.min, self.max) } else { self.min };
        match self.step {
            Some(s) => (self.min + ((v - self.min) / s).round() * s).clamp(self.min, self.max),
            None => v,
        }
    }

    /// Adds the slider.
    #[allow(clippy::too_many_lines)]
    pub fn show(mut self, ui: &mut Ui) -> SliderResponse {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let range = self.max - self.min;
        let before = *self.value;
        let locked = self.lock.as_deref().copied().unwrap_or(false);
        ui.begin(Style {
            height: Dim::Px(m.control_height),
            gap: m.space_4,
            align_items: BoxAlign::Stretch,
            grow: crate::kit::grow_x(ui),
            ..Style::row()
        });
        if self.label_width > 0.0 {
            let label_node = ui.node(Style { width: Dim::Px(self.label_width), shrink: 0.0, ..Style::default() });
            crate::kit::text(ui, label_node, Area::FULL, self.label, theme.text.small, p.text_muted, Align::Start);
        }
        let track = ui.node(Style { grow: 1.0, min_width: 40.0, ..Style::default() });
        let id = ui.id(self.label);
        let mut r = ui.interact(id, track, if locked { Sense::HOVER } else { Sense::DRAG });
        let rect = r.rect;
        let half = m.slider_thumb * 0.5;
        let usable = (rect.w - m.slider_thumb).max(1.0);
        let fine = if ui.modifiers().shift { 0.1 } else { 1.0 };
        let mut v = *self.value;
        let mut phase = EditPhase::Idle;
        let mut stepped = false;
        if !locked {
            if r.pressed {
                phase = EditPhase::Begin;
                if !ui.modifiers().shift
                    && let Some(pos) = ui.pointer()
                {
                    v = self.min + ((pos.x - rect.x - half) / usable).clamp(0.0, 1.0) * range;
                }
            }
            if r.dragged {
                v += r.drag_delta.x / usable * range * fine;
            }
            if r.double_clicked
                && let Some(d) = self.default
            {
                v = d;
                stepped = true;
            }
            if r.focused {
                let step = self.step.unwrap_or(range / 100.0);
                for (key, delta) in [
                    (Key::Left, -1.0),
                    (Key::Down, -1.0),
                    (Key::Right, 1.0),
                    (Key::Up, 1.0),
                    (Key::PageDown, -10.0),
                    (Key::PageUp, 10.0),
                ] {
                    while let Some(mods) = ui.consume_key_any(key) {
                        let k = if mods.shift { 0.1 } else { 1.0 };
                        v += delta * step * k;
                        stepped = true;
                    }
                }
                if ui.consume_key(Key::Home, Modifiers::NONE) {
                    v = self.min;
                    stepped = true;
                }
                if ui.consume_key(Key::End, Modifiers::NONE) {
                    v = self.max;
                    stepped = true;
                }
                if let Some(d) = self.default
                    && (ui.consume_key(Key::Delete, Modifiers::NONE) || ui.consume_key(Key::Backspace, Modifiers::NONE))
                {
                    v = d;
                    stepped = true;
                }
            }
        }
        // While dragging finely the value keeps sub-step precision; it is snapped on display.
        *self.value = if r.dragged { v.clamp(self.min, self.max) } else { self.quantize(v) };
        if !r.dragged && r.held {
            *self.value = self.quantize(*self.value);
        }

        // Numeric entry: Enter / Space / F2 while focused, or a double click on the value.
        let value_node = ui.node(Style { width: Dim::Px(56.0), shrink: 0.0, ..Style::default() });
        let value_id = id.with("value");
        let vr = ui.interact(value_id, value_node, Sense { click: true, drag: false, focus: false });
        let mut entry = std::mem::take(ui.memory::<SliderEntry>(id));
        let keyboard_open = r.focused && ((r.clicked && !r.held) || ui.consume_key(Key::F(2), Modifiers::NONE));
        if !locked && !entry.editing && (keyboard_open || vr.double_clicked) {
            entry.editing = true;
            entry.just_opened = true;
            entry.text.clear();
            let mut buf = [0_u8; 32];
            entry.text.push_str(format_value(*self.value, self.decimals.max(1), &mut buf));
        }
        let field_id = id.with("entry");
        if entry.editing {
            if entry.just_opened {
                ui.request_focus(field_id);
                entry.just_opened = false;
            }
            let cancel = ui.focused() == Some(field_id) && ui.consume_key(Key::Escape, Modifiers::NONE);
            ui.push_id(id);
            // The field sits over the value column (absolute in the row's last 64 px).
            let fr = ui.begin(Style {
                position: crate::layout::Position::Absolute {
                    left: None,
                    top: Some(0.0),
                    right: Some(0.0),
                    bottom: Some(0.0),
                },
                width: Dim::Px(64.0),
                ..Style::row()
            });
            let _ = fr;
            let out = TextField::new("entry", &mut entry.text).mono(true).max_chars(32).show(ui);
            ui.end();
            ui.pop_id();
            let lost = ui.focused() != Some(field_id) && !out.response.focused;
            if out.submitted || lost || cancel {
                if !cancel && let Ok(parsed) = entry.text.trim().parse::<f32>() {
                    *self.value = self.quantize(parsed);
                    stepped = true;
                }
                entry.editing = false;
                ui.request_focus(id);
            }
        }
        let editing = entry.editing;
        *ui.memory::<SliderEntry>(id) = entry;

        // Lock toggle.
        if let Some(lock) = self.lock.as_deref_mut() {
            ui.push_id(id);
            let (glyph, tip) = if *lock { (crate::icons::LOCK, "Unlock") } else { (crate::icons::LOCK_OPEN, "Lock") };
            if crate::controls::IconButton::new(glyph, tip).selected(*lock).size(m.control_height_sm).show(ui).clicked {
                *lock = !*lock;
            }
            ui.pop_id();
        }
        ui.end();

        r.changed = (*self.value - before).abs() > 0.0;
        if phase == EditPhase::Idle && r.held {
            phase = if ui.pointer_released() {
                EditPhase::Commit
            } else if r.changed {
                EditPhase::Update
            } else {
                EditPhase::Idle
            };
        }
        if stepped && phase == EditPhase::Idle && r.changed {
            phase = EditPhase::Step;
        }
        if !locked && (r.dragged || r.held) {
            ui.set_cursor(CursorIcon::Grabbing);
        }

        // Look: track, soft range, fill, thumb, value.
        let alpha = if locked { 0.45 } else { 1.0 };
        let frac = |x: f32| if range > 0.0 { ((x - self.min) / range).clamp(0.0, 1.0) } else { 0.0 };
        let t = frac(*self.value);
        let cy = Anchor::at(0.5, 0.0);
        let th = m.slider_track;
        let band = |a: f32, b: f32, grow: f32| Area {
            x0: Anchor::start(half + a * usable),
            y0: Anchor::at(0.5, -th * 0.5 - grow),
            x1: Anchor::start(half + b * usable),
            y1: Anchor::at(0.5, th * 0.5 + grow),
        };
        ui.paint(
            track,
            Paint::Rect {
                area: band(0.0, 1.0, 0.0),
                fill: p.active.fade(alpha),
                radius: Corners::all(th * 0.5),
                border: 0.0,
                border_color: Color::TRANSPARENT,
            },
        );
        let outside_soft = self.soft.is_some_and(|(lo, hi)| *self.value < lo || *self.value > hi);
        if let Some((lo, hi)) = self.soft {
            let (a, b) = (frac(lo), frac(hi));
            ui.paint(
                track,
                Paint::Rect {
                    area: band(a, b, 2.0),
                    fill: p.border_strong.fade(0.6 * alpha),
                    radius: Corners::all(th * 0.5 + 2.0),
                    border: 0.0,
                    border_color: Color::TRANSPARENT,
                },
            );
            for x in [a, b] {
                let ax = Anchor::start(half + x * usable);
                ui.paint(
                    track,
                    Paint::Line {
                        ax,
                        ay: Anchor::at(0.5, -th * 0.5 - 5.0),
                        bx: ax,
                        by: Anchor::at(0.5, th * 0.5 + 5.0),
                        width: 1.0,
                        color: p.text_faint.fade(alpha),
                    },
                );
            }
        }
        let fill = if outside_soft { p.warning } else { p.accent };
        ui.paint(
            track,
            Paint::Rect {
                area: band(0.0, t, 0.0),
                fill: fill.fade(alpha),
                radius: Corners::all(th * 0.5),
                border: 0.0,
                border_color: Color::TRANSPARENT,
            },
        );
        let thumb_x = Anchor { frac: 0.0, px: half + t * usable };
        let s = theme.shadows.control;
        let thumb_area = Area {
            x0: Anchor { frac: 0.0, px: t * usable },
            y0: Anchor::at(0.5, -half + s.offset_y),
            x1: Anchor { frac: 0.0, px: t * usable + m.slider_thumb },
            y1: Anchor::at(0.5, half + s.offset_y),
        };
        ui.paint(
            track,
            Paint::Shadow { area: thumb_area, color: p.shadow.fade(s.strength * alpha), radius: half, blur: s.blur },
        );
        let grow = 1.0 + 0.12 * r.hover_t.max(r.press_t);
        ui.paint(track, Paint::Circle { cx: thumb_x, cy, radius: half * grow, color: p.text.fade(alpha) });
        if r.focus_visible && !editing {
            let ring = Area {
                x0: Anchor { frac: 0.0, px: t * usable },
                y0: Anchor::at(0.5, -half),
                x1: Anchor { frac: 0.0, px: t * usable + m.slider_thumb },
                y1: Anchor::at(0.5, half),
            };
            ui.paint(track, Paint::FocusRing { area: ring, radius: half });
        }
        if !editing {
            let mut text = [0_u8; 32];
            let shown = format_value(*self.value, self.decimals, &mut text);
            let ink = if outside_soft { p.warning } else { p.text.mix(p.text_muted, vr.hover_t * 0.4) };
            crate::kit::text(ui, value_node, Area::FULL, shown, theme.text.mono, ink.fade(alpha), Align::End);
        }
        ui.semantic(
            id,
            Role::Slider,
            self.label,
            SemanticValue::Number { value: f64::from(*self.value), min: f64::from(self.min), max: f64::from(self.max) },
        );
        if locked {
            ui.semantic_disabled();
        }
        SliderResponse { response: r, phase }
    }
}

/// Two sliders for a left / right (or paired) value with a link toggle `[UI-07]`: while linked,
/// moving one moves the other, mirrored (`b = -a`, for signed side-to-side values) or equal.
#[derive(Debug)]
pub struct SliderPair<'a> {
    label: &'a str,
    names: [&'a str; 2],
    a: &'a mut f32,
    b: &'a mut f32,
    min: f32,
    max: f32,
    linked: &'a mut bool,
    mirror: bool,
    default: Option<f32>,
}

impl<'a> SliderPair<'a> {
    /// A pair labelled `label`, its sliders named `names` (`["Left", "Right"]`).
    pub fn new(
        label: &'a str,
        names: [&'a str; 2],
        (a, b): (&'a mut f32, &'a mut f32),
        min: f32,
        max: f32,
        linked: &'a mut bool,
    ) -> Self {
        Self { label, names, a, b, min, max, linked, mirror: false, default: None }
    }

    /// Linked values mirror (`b = -a`) instead of being equal.
    #[must_use]
    pub fn mirror(mut self, mirror: bool) -> Self {
        self.mirror = mirror;
        self
    }

    /// The default of both sliders.
    #[must_use]
    pub fn default_value(mut self, default: f32) -> Self {
        self.default = Some(default);
        self
    }

    /// Adds the pair; returns the responses of the two sliders.
    pub fn show(self, ui: &mut Ui) -> [SliderResponse; 2] {
        let theme = *ui.theme();
        let m = theme.metrics;
        let id = ui.id(self.label);
        ui.begin(Style { gap: m.space_2, align_items: BoxAlign::Center, grow: crate::kit::grow_x(ui), ..Style::row() });
        ui.begin(Style { grow: 1.0, gap: m.space_1, ..Style::column() });
        ui.push_id(id);
        let partner = |v: f32, mirror: bool| if mirror { -v } else { v };
        let mut ra = Slider::new(self.names[0], self.a, self.min, self.max);
        let mut rb_default = None;
        if let Some(d) = self.default {
            ra = ra.default_value(d);
            rb_default = Some(d);
        }
        let ra = ra.show(ui);
        if ra.response.changed && *self.linked {
            *self.b = partner(*self.a, self.mirror).clamp(self.min, self.max);
        }
        let mut sb = Slider::new(self.names[1], self.b, self.min, self.max);
        if let Some(d) = rb_default {
            sb = sb.default_value(d);
        }
        let rb = sb.show(ui);
        if rb.response.changed && *self.linked {
            *self.a = partner(*self.b, self.mirror).clamp(self.min, self.max);
        }
        ui.pop_id();
        ui.end();
        ui.push_id(id);
        let (glyph, tip) = if *self.linked {
            (crate::icons::LINK, "Unlink the pair")
        } else {
            (crate::icons::UNLINK, "Link the pair")
        };
        if crate::controls::IconButton::new(glyph, tip).selected(*self.linked).show(ui).clicked {
            *self.linked = !*self.linked;
            if *self.linked {
                *self.b = partner(*self.a, self.mirror).clamp(self.min, self.max);
            }
        }
        ui.pop_id();
        ui.end();
        ui.semantic(id, Role::Group, self.label, SemanticValue::Toggle(*self.linked));
        [ra, rb]
    }
}

/// The most rows a [`TextArea`] shows.
pub const TEXT_AREA_MAX_ROWS: usize = 40;

/// A multi-line text field `[UI-07]` `[UI-06]` (notes, descriptions, agent prompts): text wraps at
/// the field's width and scrolls vertically.
///
/// Keys (focused): everything the single-line [`TextField`] takes, plus Enter (or Shift+Enter) for
/// a new line, Ctrl+Enter to submit ([`FieldResponse::submitted`]), Up / Down / Page Up / Page Down
/// by lines, Home / End to the line's ends (Ctrl: the text's), the wheel scrolls. Pasted line
/// breaks are kept.
#[derive(Debug)]
pub struct TextArea<'a> {
    id_label: &'a str,
    text: &'a mut String,
    placeholder: &'a str,
    rows: usize,
    mono: bool,
    max_chars: usize,
}

impl<'a> TextArea<'a> {
    /// A text area for `text`, identified by `id_label`, four rows tall.
    pub fn new(id_label: &'a str, text: &'a mut String) -> Self {
        Self { id_label, text, placeholder: "", rows: 4, mono: false, max_chars: 10_000 }
    }

    /// Text shown while empty.
    #[must_use]
    pub fn placeholder(mut self, placeholder: &'a str) -> Self {
        self.placeholder = placeholder;
        self
    }

    /// Visible rows (1 to [`TEXT_AREA_MAX_ROWS`]).
    #[must_use]
    pub fn rows(mut self, rows: usize) -> Self {
        self.rows = rows.clamp(1, TEXT_AREA_MAX_ROWS);
        self
    }

    /// The monospaced face.
    #[must_use]
    pub fn mono(mut self, mono: bool) -> Self {
        self.mono = mono;
        self
    }

    /// The most characters accepted.
    #[must_use]
    pub fn max_chars(mut self, max: usize) -> Self {
        self.max_chars = max;
        self
    }

    /// Adds the area.
    #[allow(clippy::too_many_lines)]
    pub fn show(self, ui: &mut Ui) -> FieldResponse {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let style = if self.mono { theme.text.mono } else { theme.text.body };
        let line_h = style.size * style.line_height;
        let (pad_x, pad_y) = (m.control_pad_x, m.space_3);
        #[allow(clippy::cast_precision_loss)] // At most 40 rows.
        let view_h = self.rows as f32 * line_h;
        let node = ui.node(Style {
            grow: crate::kit::grow_x(ui),
            min_width: 80.0,
            height: Dim::Px(view_h + 2.0 * pad_y),
            shrink: 0.0,
            ..Style::default()
        });
        let id = ui.id(self.id_label);
        let mut r = ui.interact(id, node, Sense { click: false, drag: true, focus: true });
        let text = self.text;
        let before_len = text.len();
        let mut edit = std::mem::take(ui.memory::<TextEdit>(id));
        edit.clamp(text);
        edit.caret_time += ui.dt();
        let inner_w = (r.rect.w - 2.0 * pad_x).max(1.0);
        let wrap = Some(inner_w);
        let mut submitted = false;
        let mut changed = false;
        let mut moved = false;

        // Pointer.
        if (r.pressed || r.dragged)
            && let Some(pos) = ui.pointer()
        {
            let local = Point::new(pos.x - r.rect.x - pad_x, pos.y - r.rect.y - pad_y + edit.scroll);
            let at = ui.text_layout(text, style, wrap).hit(local);
            edit.cursor = at;
            if r.pressed && !ui.modifiers().shift {
                edit.anchor = at;
            }
            if r.double_clicked {
                edit.anchor = prev_word(text, next_boundary(text, at));
                edit.cursor = next_word(text, edit.anchor);
            }
            moved = true;
        }
        if r.hovered {
            ui.set_cursor(CursorIcon::Text);
            let wheel = ui.wheel().y;
            if wheel != 0.0 {
                edit.scroll -= wheel;
            }
        }

        if r.focused {
            let cmd = |mods: Modifiers| mods.command() && !mods.alt;
            let (typed, committed) = ui.typed_text();
            let mut incoming = String::new();
            incoming.push_str(typed);
            incoming.push_str(committed);
            if let Some(paste) = ui.take_paste() {
                incoming.extend(paste.chars().filter(|c| *c != '\r'));
                edit.snapshot(text, false);
            } else if !incoming.is_empty() {
                edit.snapshot(text, true);
            }
            if !incoming.is_empty() {
                edit.replace(text, &incoming, self.max_chars);
                changed = true;
            }
            // Moves by lines need the wrapped layout of the current text.
            let line_move = |ui: &mut Ui, text: &str, from: usize, lines: i64| -> usize {
                let layout = ui.text_layout(text, style, wrap);
                let (x, line) = layout.caret(from);
                let target = i64::try_from(line).unwrap_or(0) + lines;
                if target < 0 {
                    return 0;
                }
                match usize::try_from(target).ok().and_then(|t| layout.lines.get(t)) {
                    Some(l) => layout.hit(Point::new(x, l.top + 1.0)),
                    None => text.len(),
                }
            };
            let mut keys = 0;
            while keys < 64 {
                keys += 1;
                if let Some(e) = edit_key(ui, text, &mut edit, self.max_chars) {
                    match e {
                        Edited::Changed => changed = true,
                        Edited::Moved => moved = true,
                        Edited::Nothing => {}
                    }
                    continue;
                }
                let vertical = [
                    (Key::Up, -1_i64),
                    (Key::Down, 1),
                    (Key::PageUp, -(self.rows as i64)),
                    (Key::PageDown, self.rows as i64),
                ];
                let mut handled = false;
                for (key, lines) in vertical {
                    if let Some(mods) = ui.consume_key_any(key) {
                        edit.cursor = line_move(ui, text, edit.cursor, lines);
                        if !mods.shift {
                            edit.anchor = edit.cursor;
                        }
                        moved = true;
                        handled = true;
                        break;
                    }
                }
                if handled {
                    continue;
                }
                if let Some(mods) = ui.consume_key_any(Key::Home) {
                    edit.cursor = if cmd(mods) {
                        0
                    } else {
                        let layout = ui.text_layout(text, style, wrap);
                        let (_, line) = layout.caret(edit.cursor);
                        layout.lines.get(line).map_or(0, |l| l.bytes.0 as usize)
                    };
                    if !mods.shift {
                        edit.anchor = edit.cursor;
                    }
                    moved = true;
                } else if let Some(mods) = ui.consume_key_any(Key::End) {
                    edit.cursor = if cmd(mods) {
                        text.len()
                    } else {
                        let layout = ui.text_layout(text, style, wrap);
                        let (_, line) = layout.caret(edit.cursor);
                        layout.lines.get(line).map_or(text.len(), |l| l.bytes.1 as usize)
                    };
                    if !mods.shift {
                        edit.anchor = edit.cursor;
                    }
                    moved = true;
                } else if let Some(mods) = ui.consume_key_any(Key::Enter) {
                    if cmd(mods) {
                        submitted = true;
                    } else {
                        edit.snapshot(text, false);
                        edit.replace(text, "\n", self.max_chars);
                        changed = true;
                    }
                } else if ui.consume_key(Key::Escape, Modifiers::NONE) {
                    ui.clear_focus();
                } else {
                    break;
                }
            }
            if moved {
                edit.typing = false;
            }
            ui.want_text_input(r.rect);
            if !theme.motion.reduced {
                ui.request_repaint();
            }
        }
        edit.clamp(text);
        if changed || moved {
            edit.caret_time = 0.0;
        }

        // Keep the caret in view; collect what the look needs from the layout.
        let sel = edit.selection();
        let layout = ui.text_layout(text, style, wrap);
        let (caret_x, caret_line) = layout.caret(edit.cursor);
        let caret_top = layout.lines.get(caret_line).map_or(0.0, |l| l.top);
        let content_h = layout.size.h.max(line_h);
        if changed || moved {
            if caret_top < edit.scroll {
                edit.scroll = caret_top;
            }
            if caret_top + line_h > edit.scroll + view_h {
                edit.scroll = caret_top + line_h - view_h;
            }
        }
        edit.scroll = edit.scroll.clamp(0.0, (content_h - view_h).max(0.0));
        // Selection rectangles of the visible lines (x0, x1, top).
        let mut spans = [(0.0_f32, 0.0_f32, 0.0_f32); TEXT_AREA_MAX_ROWS + 2];
        let mut n_spans = 0;
        if r.focused && !sel.is_empty() {
            let (_, first) = layout.caret(sel.start);
            let (_, last) = layout.caret(sel.end);
            for (i, line) in layout.lines.iter().enumerate().take(last + 1).skip(first) {
                if line.top + line_h < edit.scroll || line.top > edit.scroll + view_h || n_spans == spans.len() {
                    continue;
                }
                let x0 = if i == first { layout.caret(sel.start).0 } else { 0.0 };
                let x1 = if i == last { layout.caret(sel.end).0 } else { line.width + 4.0 };
                spans[n_spans] = (x0, x1, line.top);
                n_spans += 1;
            }
        }

        // Look.
        let border = if r.focused { p.border_focus } else { p.border_strong.mix(p.border_focus, r.hover_t * 0.5) };
        ui.paint(
            node,
            Paint::Rect {
                area: Area::FULL,
                fill: p.surface,
                radius: Corners::all(m.radius_sm),
                border: m.border,
                border_color: border,
            },
        );
        ui.push_clip(
            node,
            Area {
                x0: Anchor::start(m.border),
                y0: Anchor::start(pad_y * 0.5),
                x1: Anchor::end(-m.border),
                y1: Anchor::end(-pad_y * 0.5),
            },
        );
        let (x0, y0) = (pad_x, pad_y - edit.scroll);
        for &(a, b, top) in &spans[..n_spans] {
            let area = Area::offset(x0 + a, y0 + top, (b - a).max(2.0), line_h);
            ui.paint(
                node,
                Paint::Rect {
                    area,
                    fill: p.selection,
                    radius: Corners::all(2.0),
                    border: 0.0,
                    border_color: Color::TRANSPARENT,
                },
            );
        }
        let text_area = Area::offset(x0, y0, inner_w, content_h + line_h);
        let (shown, color) = if text.is_empty() { (self.placeholder, p.text_faint) } else { (text.as_str(), p.text) };
        let t = ui.intern(shown);
        ui.paint(node, Paint::Text { area: text_area, text: t, style, color, align: Align::Start, wrap: true });
        if r.focused {
            let period = theme.motion.caret_period.max(0.1);
            if theme.motion.reduced || edit.caret_time % period < theme.motion.caret_on {
                let area = Area::offset(x0 + caret_x, y0 + caret_top + 1.0, m.caret_width, line_h - 2.0);
                ui.paint(
                    node,
                    Paint::Rect {
                        area,
                        fill: p.text,
                        radius: Corners::ZERO,
                        border: 0.0,
                        border_color: Color::TRANSPARENT,
                    },
                );
            }
            if !ui.preedit().is_empty() {
                let pre = ui.intern(ui.preedit().to_owned().as_str());
                let area = Area::offset(x0 + caret_x, y0 + caret_top, inner_w, line_h);
                ui.paint(
                    node,
                    Paint::Text { area, text: pre, style, color: p.text_muted, align: Align::Start, wrap: false },
                );
            }
        }
        ui.pop_clip();
        if r.focus_visible {
            ui.paint(node, Paint::FocusRing { area: Area::FULL, radius: m.radius_sm });
        }
        let label = if self.placeholder.is_empty() { self.id_label } else { self.placeholder };
        let value = ui.intern(text);
        ui.semantic(id, Role::TextField, label, SemanticValue::Text(value));
        r.changed = changed || text.len() != before_len;
        *ui.memory::<TextEdit>(id) = edit;
        FieldResponse { response: r, submitted }
    }
}

/// What one editing key did to a text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Edited {
    /// A key was used but nothing changed (copy with nothing selected).
    Nothing,
    /// The caret or selection moved.
    Moved,
    /// The text changed.
    Changed,
}

/// Takes one of the editing keys every text widget shares (Left, Right, Backspace, Delete,
/// Ctrl+A / C / X / Z / Y) from this frame's keys and applies it; `None` when there is none left.
fn edit_key(ui: &mut Ui, text: &mut String, edit: &mut TextEdit, max_chars: usize) -> Option<Edited> {
    let cmd = |mods: Modifiers| mods.command() && !mods.alt;
    let mut out = Edited::Nothing;
    if let Some(mods) = ui.consume_key_any(Key::Left) {
        edit.cursor = if cmd(mods) {
            prev_word(text, edit.cursor)
        } else if !mods.shift && edit.cursor != edit.anchor {
            edit.selection().start
        } else {
            prev_boundary(text, edit.cursor)
        };
        if !mods.shift {
            edit.anchor = edit.cursor;
        }
        out = Edited::Moved;
    } else if let Some(mods) = ui.consume_key_any(Key::Right) {
        edit.cursor = if cmd(mods) {
            next_word(text, edit.cursor)
        } else if !mods.shift && edit.cursor != edit.anchor {
            edit.selection().end
        } else {
            next_boundary(text, edit.cursor)
        };
        if !mods.shift {
            edit.anchor = edit.cursor;
        }
        out = Edited::Moved;
    } else if let Some(mods) = ui.consume_key_any(Key::Backspace) {
        edit.snapshot(text, false);
        if edit.cursor == edit.anchor {
            edit.anchor = if cmd(mods) { prev_word(text, edit.cursor) } else { prev_boundary(text, edit.cursor) };
        }
        edit.replace(text, "", max_chars);
        out = Edited::Changed;
    } else if let Some(mods) = ui.consume_key_any(Key::Delete) {
        edit.snapshot(text, false);
        if edit.cursor == edit.anchor {
            edit.anchor = if cmd(mods) { next_word(text, edit.cursor) } else { next_boundary(text, edit.cursor) };
        }
        edit.replace(text, "", max_chars);
        out = Edited::Changed;
    } else if let Some(mods) = ui.consume_key_any(Key::Letter('A')).filter(|m| cmd(*m)) {
        let _ = mods;
        edit.anchor = 0;
        edit.cursor = text.len();
        out = Edited::Moved;
    } else if ui.consume_key_any(Key::Letter('C')).is_some_and(cmd) {
        let sel = edit.selection();
        if !sel.is_empty() {
            ui.copy(&text[sel]);
        }
    } else if ui.consume_key_any(Key::Letter('X')).is_some_and(cmd) {
        let sel = edit.selection();
        if !sel.is_empty() {
            ui.copy(&text[sel]);
            edit.snapshot(text, false);
            edit.replace(text, "", max_chars);
            out = Edited::Changed;
        }
    } else if let Some(mods) = ui.consume_key_any(Key::Letter('Z')).filter(|m| cmd(*m)) {
        let (from, to) = if mods.shift { (&mut edit.redo, &mut edit.undo) } else { (&mut edit.undo, &mut edit.redo) };
        if let Some((t, c, a)) = from.pop() {
            to.push((std::mem::replace(text, t), edit.cursor, edit.anchor));
            edit.cursor = c;
            edit.anchor = a;
            edit.typing = false;
            out = Edited::Changed;
        }
    } else if ui.consume_key_any(Key::Letter('Y')).is_some_and(cmd) {
        if let Some((t, c, a)) = edit.redo.pop() {
            edit.undo.push((std::mem::replace(text, t), edit.cursor, edit.anchor));
            edit.cursor = c;
            edit.anchor = a;
            edit.typing = false;
            out = Edited::Changed;
        }
    } else {
        return None;
    }
    Some(out)
}

/// Writes `v` with `decimals` decimals into `buf` without allocating; returns the text.
pub(crate) fn format_value(v: f32, decimals: usize, buf: &mut [u8; 32]) -> &str {
    use std::io::Write as _;
    let mut cursor = std::io::Cursor::new(&mut buf[..]);
    let _ = write!(cursor, "{v:.decimals$}");
    let len = usize::try_from(cursor.position()).unwrap_or(0);
    std::str::from_utf8(&buf[..len]).unwrap_or("")
}

/// The editing state of a text field, kept in [`Ui::memory`].
#[derive(Clone, Debug, Default)]
pub struct TextEdit {
    /// The caret, a byte offset on a character boundary.
    pub cursor: usize,
    /// The other end of the selection (equal to `cursor` when nothing is selected).
    pub anchor: usize,
    /// Horizontal scroll of the text, logical pixels.
    pub scroll: f32,
    /// Earlier states for undo: (text, cursor, anchor).
    undo: Vec<(String, usize, usize)>,
    /// Undone states for redo.
    redo: Vec<(String, usize, usize)>,
    /// Whether the last edit was typing (consecutive typing is one undo step).
    typing: bool,
    /// When the caret last moved (the blink restarts).
    caret_time: f32,
}

impl TextEdit {
    /// Most undo steps kept.
    pub const MAX_UNDO: usize = 100;

    /// The selected byte range.
    pub fn selection(&self) -> std::ops::Range<usize> {
        self.cursor.min(self.anchor)..self.cursor.max(self.anchor)
    }

    fn snapshot(&mut self, text: &str, typing: bool) {
        if typing && self.typing {
            return;
        }
        self.undo.push((text.to_owned(), self.cursor, self.anchor));
        if self.undo.len() > Self::MAX_UNDO {
            self.undo.remove(0);
        }
        self.redo.clear();
        self.typing = typing;
    }

    fn clamp(&mut self, text: &str) {
        self.cursor = floor_boundary(text, self.cursor);
        self.anchor = floor_boundary(text, self.anchor);
    }

    /// Replaces the selection with `insert`.
    fn replace(&mut self, text: &mut String, insert: &str, max_chars: usize) {
        let sel = self.selection();
        let room = max_chars.saturating_sub(text.chars().count() - text[sel.clone()].chars().count());
        let insert: &str = match insert.char_indices().nth(room) {
            Some((cut, _)) => &insert[..cut],
            None => insert,
        };
        text.replace_range(sel.clone(), insert);
        self.cursor = sel.start + insert.len();
        self.anchor = self.cursor;
    }
}

/// The largest character boundary at or before `i`.
fn floor_boundary(text: &str, i: usize) -> usize {
    let mut i = i.min(text.len());
    while !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// The previous character boundary before `i`.
fn prev_boundary(text: &str, i: usize) -> usize {
    text[..floor_boundary(text, i)].char_indices().next_back().map_or(0, |(j, _)| j)
}

/// The next character boundary after `i`.
fn next_boundary(text: &str, i: usize) -> usize {
    let i = floor_boundary(text, i);
    text[i..].chars().next().map_or(i, |c| i + c.len_utf8())
}

/// The start of the word before `i` (skipping spaces first).
fn prev_word(text: &str, i: usize) -> usize {
    let mut j = floor_boundary(text, i);
    while j > 0 && !is_word(text, prev_boundary(text, j)) {
        j = prev_boundary(text, j);
    }
    while j > 0 && is_word(text, prev_boundary(text, j)) {
        j = prev_boundary(text, j);
    }
    j
}

/// The end of the word after `i` (skipping spaces first).
fn next_word(text: &str, i: usize) -> usize {
    let mut j = floor_boundary(text, i);
    while j < text.len() && !is_word(text, j) {
        j = next_boundary(text, j);
    }
    while j < text.len() && is_word(text, j) {
        j = next_boundary(text, j);
    }
    j
}

fn is_word(text: &str, i: usize) -> bool {
    text[i..].chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_')
}

/// What a text field did this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FieldResponse {
    /// The field's interaction (`changed` when the text changed).
    pub response: Response,
    /// Enter was pressed.
    pub submitted: bool,
}

/// A single-line text field `[UI-06]`: caret, selection (Shift + arrows, drag, double click selects a
/// word, Ctrl+A), word movement (Ctrl + arrows), Backspace / Delete (Ctrl: a word), copy / cut /
/// paste through the shell's clipboard, undo / redo (Ctrl+Z, Ctrl+Y or Ctrl+Shift+Z; consecutive
/// typing is one step), IME composition shown at the caret, Enter submits, Escape leaves.
/// Line breaks in pasted text become spaces.
#[derive(Debug)]
pub struct TextField<'a> {
    id_label: &'a str,
    text: &'a mut String,
    placeholder: &'a str,
    width: Option<f32>,
    mono: bool,
    max_chars: usize,
    icon: Option<char>,
}

impl<'a> TextField<'a> {
    /// A field editing `text`, identified by `id_label`.
    pub fn new(id_label: &'a str, text: &'a mut String) -> Self {
        Self { id_label, text, placeholder: "", width: None, mono: false, max_chars: 10_000, icon: None }
    }

    /// An icon before the text (a search box's magnifier).
    #[must_use]
    pub fn icon(mut self, icon: char) -> Self {
        self.icon = Some(icon);
        self
    }

    /// Text shown while the field is empty.
    #[must_use]
    pub fn placeholder(mut self, placeholder: &'a str) -> Self {
        self.placeholder = placeholder;
        self
    }

    /// A fixed width (otherwise it grows to fill its row).
    #[must_use]
    pub fn width(mut self, width: f32) -> Self {
        self.width = Some(width);
        self
    }

    /// The monospaced face (numbers).
    #[must_use]
    pub fn mono(mut self, mono: bool) -> Self {
        self.mono = mono;
        self
    }

    /// The most characters accepted.
    #[must_use]
    pub fn max_chars(mut self, max: usize) -> Self {
        self.max_chars = max;
        self
    }

    /// Adds the field.
    #[allow(clippy::too_many_lines)]
    pub fn show(self, ui: &mut Ui) -> FieldResponse {
        let theme = *ui.theme();
        let (m, p) = (theme.metrics, theme.palette);
        let style = if self.mono { theme.text.mono } else { theme.text.body };
        let width = self.width.map_or(Dim::Auto, Dim::Px);
        let node = ui.node(Style {
            width,
            grow: if self.width.is_some() { 0.0 } else { crate::kit::grow_x(ui) },
            height: Dim::Px(m.control_height),
            padding: Insets::xy(m.control_pad_x, 0.0),
            ..Style::default()
        });
        let id = ui.id(self.id_label);
        // No click sense: Enter and Space belong to the text, not to an activation.
        let mut r = ui.interact(id, node, Sense { click: false, drag: true, focus: true });
        let text = self.text;
        let before_len = text.len();
        let mut edit = std::mem::take(ui.memory::<TextEdit>(id));
        edit.clamp(text);
        let dt = ui.dt();
        edit.caret_time += dt;
        let pad = m.control_pad_x;
        let pad_l = if self.icon.is_some() { pad + m.icon + m.space_3 } else { pad };
        let inner_w = (r.rect.w - pad_l - pad).max(0.0);
        let mut submitted = false;
        let mut changed = false;
        let mut moved = false;

        // Pointer: place the caret, extend with Shift or a drag, double click selects a word.
        if (r.pressed || r.dragged)
            && let Some(pos) = ui.pointer()
        {
            let x = pos.x - r.rect.x - pad_l + edit.scroll;
            let layout = ui.text_layout(text, style, None);
            let at = layout.hit(Point::new(x, 1.0));
            edit.cursor = at;
            if r.pressed && !ui.modifiers().shift {
                edit.anchor = at;
            }
            if r.double_clicked {
                edit.anchor = prev_word(text, next_boundary(text, at));
                edit.cursor = next_word(text, edit.anchor);
            }
            moved = true;
        }
        if r.hovered {
            ui.set_cursor(CursorIcon::Text);
        }

        if r.focused {
            // Typing and IME commits.
            let (typed, committed) = ui.typed_text();
            let mut incoming = String::new();
            incoming.push_str(typed);
            incoming.push_str(committed);
            if let Some(paste) = ui.take_paste() {
                incoming.extend(paste.chars().map(|c| if c == '\n' || c == '\r' || c == '\t' { ' ' } else { c }));
                edit.snapshot(text, false);
            } else if !incoming.is_empty() {
                edit.snapshot(text, true);
            }
            if !incoming.is_empty() {
                edit.replace(text, &incoming, self.max_chars);
                changed = true;
            }
            // Keys.
            let mut keys = 0;
            while keys < 64 {
                keys += 1;
                if let Some(e) = edit_key(ui, text, &mut edit, self.max_chars) {
                    match e {
                        Edited::Changed => changed = true,
                        Edited::Moved => moved = true,
                        Edited::Nothing => {}
                    }
                } else if let Some(mods) = ui.consume_key_any(Key::Home) {
                    edit.cursor = 0;
                    if !mods.shift {
                        edit.anchor = 0;
                    }
                    moved = true;
                } else if let Some(mods) = ui.consume_key_any(Key::End) {
                    edit.cursor = text.len();
                    if !mods.shift {
                        edit.anchor = edit.cursor;
                    }
                    moved = true;
                } else if ui.consume_key(Key::Enter, Modifiers::NONE) {
                    submitted = true;
                } else if ui.consume_key(Key::Escape, Modifiers::NONE) {
                    ui.clear_focus();
                } else {
                    break;
                }
            }
            if moved {
                edit.typing = false;
            }
            let area = Rect::new(r.rect.x, r.rect.y, r.rect.w, r.rect.h);
            ui.want_text_input(area);
            if !theme.motion.reduced {
                ui.request_repaint();
            }
        }
        edit.clamp(text);
        if changed || moved {
            edit.caret_time = 0.0;
        }

        // Keep the caret visible.
        let caret_x = ui.text_layout(text, style, None).caret(edit.cursor).0;
        if caret_x - edit.scroll > inner_w {
            edit.scroll = caret_x - inner_w;
        }
        if caret_x - edit.scroll < 0.0 {
            edit.scroll = caret_x;
        }
        let full_w = ui.text_layout(text, style, None).size.w;
        edit.scroll = edit.scroll.clamp(0.0, (full_w - inner_w + m.caret_width).max(0.0));

        // Look.
        let focused = r.focused;
        let border = if focused { p.border_focus } else { p.border_strong.mix(p.border_focus, r.hover_t * 0.5) };
        ui.paint(
            node,
            Paint::Rect {
                area: Area::FULL,
                fill: p.surface,
                radius: Corners::all(m.radius_sm),
                border: m.border,
                border_color: border,
            },
        );
        if let Some(icon) = self.icon {
            crate::kit::icon(
                ui,
                node,
                Anchor::start(pad + m.icon * 0.5 - 2.0),
                Anchor::at(0.5, 0.0),
                icon,
                m.icon_sm,
                p.text_faint,
            );
        }
        ui.push_clip(
            node,
            Area {
                x0: Anchor::start(pad_l - pad * 0.5),
                y0: Anchor::start(0.0),
                x1: Anchor::end(-pad * 0.5),
                y1: Anchor::end(0.0),
            },
        );
        let x0 = pad_l - edit.scroll;
        let line_h = style.size * style.line_height;
        let sel = edit.selection();
        if focused && !sel.is_empty() {
            let layout = ui.text_layout(text, style, None);
            let (a, _) = layout.caret(sel.start);
            let (b, _) = layout.caret(sel.end);
            let area = Area {
                x0: Anchor::start(x0 + a),
                y0: Anchor::at(0.5, -line_h * 0.5),
                x1: Anchor::start(x0 + b),
                y1: Anchor::at(0.5, line_h * 0.5),
            };
            ui.paint(
                node,
                Paint::Rect {
                    area,
                    fill: p.selection,
                    radius: Corners::all(2.0),
                    border: 0.0,
                    border_color: Color::TRANSPARENT,
                },
            );
        }
        let text_area =
            Area { x0: Anchor::start(x0), y0: Anchor::start(0.0), x1: Anchor::end(0.0), y1: Anchor::end(0.0) };
        if text.is_empty() {
            let t = ui.intern(self.placeholder);
            ui.paint(
                node,
                Paint::Text { area: text_area, text: t, style, color: p.text_faint, align: Align::Start, wrap: false },
            );
        } else {
            let t = ui.intern(text);
            ui.paint(
                node,
                Paint::Text { area: text_area, text: t, style, color: p.text, align: Align::Start, wrap: false },
            );
        }
        if focused {
            let preedit_w = if ui.preedit().is_empty() {
                0.0
            } else {
                let pre = ui.preedit().to_owned();
                let w = ui.measure_text(&pre, style).w;
                let t = ui.intern(&pre);
                let area = Area {
                    x0: Anchor::start(x0 + caret_x),
                    y0: Anchor::start(0.0),
                    x1: Anchor::start(x0 + caret_x + w + 1.0),
                    y1: Anchor::end(0.0),
                };
                ui.paint(
                    node,
                    Paint::Text { area, text: t, style, color: p.text_muted, align: Align::Start, wrap: false },
                );
                let base = Anchor::at(0.5, line_h * 0.5 - 1.0);
                ui.paint(
                    node,
                    Paint::Line {
                        ax: Anchor::start(x0 + caret_x),
                        ay: base,
                        bx: Anchor::start(x0 + caret_x + w),
                        by: base,
                        width: 1.0,
                        color: p.text_muted,
                    },
                );
                w
            };
            let period = theme.motion.caret_period.max(0.1);
            let visible = theme.motion.reduced || edit.caret_time % period < theme.motion.caret_on;
            if visible {
                let cx = x0 + caret_x + preedit_w;
                let area = Area {
                    x0: Anchor::start(cx),
                    y0: Anchor::at(0.5, -line_h * 0.5 + 1.0),
                    x1: Anchor::start(cx + m.caret_width),
                    y1: Anchor::at(0.5, line_h * 0.5 - 1.0),
                };
                ui.paint(
                    node,
                    Paint::Rect {
                        area,
                        fill: p.text,
                        radius: Corners::ZERO,
                        border: 0.0,
                        border_color: Color::TRANSPARENT,
                    },
                );
            }
        }
        ui.pop_clip();
        if r.focus_visible {
            ui.paint(node, Paint::FocusRing { area: Area::FULL, radius: m.radius_sm });
        }
        let label = if self.placeholder.is_empty() { self.id_label } else { self.placeholder };
        let value = ui.intern(text);
        ui.semantic(id, Role::TextField, label, SemanticValue::Text(value));
        r.changed = changed || text.len() != before_len;
        *ui.memory::<TextEdit>(id) = edit;
        FieldResponse { response: r, submitted }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Size;
    use crate::input::{Event, InputQueue, PointerButton};
    use crate::tests::{assert_golden, fonts, gpu_lock, render, renderer};
    use crate::theme::Theme;

    /// The state the gallery edits.
    struct Model {
        exposure: f32,
        roughness: f32,
        name: String,
        empty: String,
        saves: u32,
    }

    impl Model {
        fn new() -> Self {
            Self { exposure: 0.25, roughness: 0.5, name: "Skin \u{2013} cheek".into(), empty: String::new(), saves: 0 }
        }
    }

    /// A small panel with every widget of this module.
    fn gallery(ui: &mut Ui, model: &mut Model) {
        let theme = *ui.theme();
        let m = theme.metrics;
        let panel = ui.begin(Style {
            padding: Insets::all(m.space_6),
            gap: m.space_4,
            width: Dim::Px(320.0),
            ..Style::column()
        });
        ui.paint(
            panel,
            Paint::Rect {
                area: Area::FULL,
                fill: theme.palette.panel,
                radius: Corners::all(m.radius),
                border: m.border,
                border_color: theme.palette.border,
            },
        );
        label_styled(ui, "Material", theme.text.heading, theme.palette.text);
        label(ui, "Kerning: AV To Wa");
        Slider::new("Exposure", &mut model.exposure, -1.0, 1.0).step(0.05).default_value(0.0).show(ui);
        Slider::new("Roughness", &mut model.roughness, 0.0, 1.0).show(ui);
        TextField::new("name", &mut model.name).placeholder("Name").show(ui);
        TextField::new("search", &mut model.empty).placeholder("Search regions").show(ui);
        ui.begin(Style { gap: m.space_4, ..Style::row() });
        if Button::new("Save").kind(ButtonKind::Primary).show(ui).clicked {
            model.saves += 1;
        }
        button(ui, "Revert");
        Button::new("Delete").kind(ButtonKind::Danger).show(ui);
        Button::new("Locked").enabled(false).show(ui);
        ui.end();
        ui.end();
    }

    struct Harness {
        ui: Ui,
        model: Model,
        time: f64,
    }

    impl Harness {
        fn new(theme: &Theme) -> Self {
            Self { ui: Ui::new(fonts().clone(), *theme), model: Model::new(), time: 0.0 }
        }

        fn frame(&mut self, events: Vec<Event>, scale: f32) {
            self.time += 0.016;
            let mut q = InputQueue { events, time: self.time };
            self.ui.begin_frame(&mut q, Size::new(340.0, 300.0), scale);
            gallery(&mut self.ui, &mut self.model);
            self.ui.end_frame();
        }

        fn rect_of(&self, role: Role, label: &str) -> Rect {
            self.ui
                .semantics()
                .iter()
                .find(|s| s.role == role && self.ui.text(s.label) == label)
                .map_or(Rect::ZERO, |s| s.rect)
        }

        fn key(&mut self, key: Key, mods: Modifiers) {
            self.frame(vec![Event::Modifiers(mods), Event::Key { key, pressed: true, repeat: false }], 1.0);
        }

        fn click(&mut self, p: Point) {
            self.frame(
                vec![
                    Event::PointerMoved(p),
                    Event::Button { button: PointerButton::Primary, pressed: true, pos: p },
                    Event::Button { button: PointerButton::Primary, pressed: false, pos: p },
                ],
                1.0,
            );
        }
    }

    #[test]
    fn ui_09_keyboard_reaches_and_operates_every_widget() {
        let mut h = Harness::new(&Theme::dark());
        h.frame(vec![], 1.0);
        h.frame(vec![], 1.0);
        // Focus order: two sliders, two fields, three enabled buttons (the disabled one is skipped).
        let order: Vec<Role> = (0..8)
            .map(|_| {
                h.key(Key::Tab, Modifiers::NONE);
                let f = h.ui.focused();
                h.ui.semantics().iter().find(|s| Some(s.id) == f).map_or(Role::Window, |s| s.role)
            })
            .collect();
        assert_eq!(
            order,
            [
                Role::Slider,
                Role::Slider,
                Role::TextField,
                Role::TextField,
                Role::Button,
                Role::Button,
                Role::Button,
                Role::Slider
            ]
        );
        // Slider: arrows step, Home/End go to the ends, Delete restores the default.
        h.key(Key::Right, Modifiers::NONE);
        assert!((h.model.exposure - 0.30).abs() < 1e-5, "{}", h.model.exposure);
        h.key(Key::Left, Modifiers::SHIFT);
        assert!((h.model.exposure - 0.30).abs() < 1e-5, "a tenth of a step snaps back to the step grid");
        h.key(Key::End, Modifiers::NONE);
        assert_eq!(h.model.exposure, 1.0);
        h.key(Key::Delete, Modifiers::NONE);
        assert_eq!(h.model.exposure, 0.0);
        // The focus ring is drawn around the focused slider's thumb.
        let ring = h.ui.theme().palette.focus_ring.to_array();
        assert!(h.ui.draw_list().instances().iter().any(|i| i.color2 == ring));
        // Tab to Save and press Space.
        for _ in 0..4 {
            h.key(Key::Tab, Modifiers::NONE);
        }
        h.key(Key::Space, Modifiers::NONE);
        assert_eq!(h.model.saves, 1);
        let save = h.ui.semantics().iter().find(|s| h.ui.text(s.label) == "Save").copied();
        assert!(save.is_some_and(|s| s.focused));
        let locked = h.ui.semantics().iter().find(|s| h.ui.text(s.label) == "Locked").copied();
        assert!(locked.is_some_and(|s| s.disabled));
    }

    #[test]
    fn ui_06_slider_pointer_jump_drag_and_reset() {
        let mut h = Harness::new(&Theme::dark());
        h.frame(vec![], 1.0);
        h.frame(vec![], 1.0);
        let r = h.rect_of(Role::Slider, "Roughness");
        assert!(r.w > 100.0);
        let thumb = h.ui.theme().metrics.slider_thumb;
        // Press at 3/4 of the usable track: jumps there.
        let x = r.x + thumb * 0.5 + (r.w - thumb) * 0.75;
        let y = r.center().y;
        h.frame(
            vec![
                Event::PointerMoved(Point::new(x, y)),
                Event::Button { button: PointerButton::Primary, pressed: true, pos: Point::new(x, y) },
            ],
            1.0,
        );
        assert!((h.model.roughness - 0.75).abs() < 0.01, "{}", h.model.roughness);
        // Drag right by a tenth of the track.
        let dx = (r.w - thumb) * 0.1;
        h.frame(vec![Event::PointerMoved(Point::new(x + dx, y))], 1.0);
        assert!((h.model.roughness - 0.85).abs() < 0.01, "{}", h.model.roughness);
        // Shift while dragging: a tenth of the speed.
        h.frame(vec![Event::Modifiers(Modifiers::SHIFT), Event::PointerMoved(Point::new(x + 2.0 * dx, y))], 1.0);
        assert!((h.model.roughness - 0.86).abs() < 0.01, "{}", h.model.roughness);
        h.frame(
            vec![
                Event::Modifiers(Modifiers::NONE),
                Event::Button { button: PointerButton::Primary, pressed: false, pos: Point::new(x, y) },
            ],
            1.0,
        );
        // Double click resets the exposure slider to its default.
        let e = h.rect_of(Role::Slider, "Exposure");
        let p = Point::new(e.x + e.w * 0.9, e.center().y);
        h.click(p);
        h.click(p);
        assert_eq!(h.model.exposure, 0.0);
    }

    #[test]
    fn ui_06_text_field_editing_clipboard_and_undo() {
        let mut h = Harness::new(&Theme::dark());
        h.frame(vec![], 1.0);
        h.frame(vec![], 1.0);
        let field = h.rect_of(Role::TextField, "Search regions");
        h.click(field.center());
        assert!(h.ui.semantics().iter().any(|s| s.role == Role::TextField && s.focused));
        let typed: Vec<Event> = "cheek left".chars().map(Event::Text).collect();
        h.frame(typed, 1.0);
        assert_eq!(h.model.empty, "cheek left");
        // Ctrl+Backspace deletes a word; Ctrl+Z restores it.
        h.key(Key::Backspace, Modifiers::CTRL);
        assert_eq!(h.model.empty, "cheek ");
        h.key(Key::Letter('Z'), Modifiers::CTRL);
        assert_eq!(h.model.empty, "cheek left");
        // Shift+Home selects everything before the caret; typing replaces it.
        h.key(Key::Home, Modifiers::SHIFT);
        h.key(Key::Letter('C'), Modifiers::CTRL);
        h.frame(vec![Event::Text('x')], 1.0);
        assert_eq!(h.model.empty, "x");
        // Paste with a line break; undo the paste, redo it.
        h.frame(vec![Event::Paste("a\nb".into())], 1.0);
        assert_eq!(h.model.empty, "xa b");
        h.key(Key::Letter('Z'), Modifiers::CTRL);
        assert_eq!(h.model.empty, "x");
        h.key(Key::Letter('Y'), Modifiers::CTRL);
        assert_eq!(h.model.empty, "xa b");
        // Left by word, then delete forward; multi-byte characters stay intact.
        h.frame(vec![Event::Text('\u{e9}')], 1.0);
        h.key(Key::Left, Modifiers::NONE);
        h.key(Key::Delete, Modifiers::NONE);
        assert_eq!(h.model.empty, "xa b");
        h.key(Key::Letter('A'), Modifiers::CTRL);
        h.key(Key::Letter('X'), Modifiers::CTRL);
        assert_eq!(h.model.empty, "");
        // Escape leaves the field.
        h.key(Key::Escape, Modifiers::NONE);
        assert!(!h.ui.semantics().iter().any(|s| s.focused));
    }

    #[test]
    fn copy_reaches_the_frame_output() {
        let mut ui = Ui::new(fonts().clone(), Theme::dark());
        let mut text = String::from("copy me");
        let mut run = |ui: &mut Ui, events: Vec<Event>, t: f64| -> Option<String> {
            let mut q = InputQueue { events, time: t };
            ui.begin_frame(&mut q, Size::new(200.0, 50.0), 1.0);
            TextField::new("f", &mut text).show(ui);
            ui.end_frame().copied.clone()
        };
        run(&mut ui, vec![], 0.0);
        run(&mut ui, vec![], 0.1);
        let p = Point::new(50.0, 10.0);
        run(
            &mut ui,
            vec![
                Event::PointerMoved(p),
                Event::Button { button: PointerButton::Primary, pressed: true, pos: p },
                Event::Button { button: PointerButton::Primary, pressed: false, pos: p },
            ],
            0.2,
        );
        run(
            &mut ui,
            vec![Event::Modifiers(Modifiers::CTRL), Event::Key { key: Key::Letter('A'), pressed: true, repeat: false }],
            0.3,
        );
        let copied = run(&mut ui, vec![Event::Key { key: Key::Letter('C'), pressed: true, repeat: false }], 0.4);
        assert_eq!(copied.as_deref(), Some("copy me"));
    }

    #[test]
    fn ac_wp12_2_widget_golden_at_100_150_200_percent() {
        // [UI-07] [UI-09] The widget gallery with a focused slider (focus ring) and a focused text
        // field with a selection, at three UI scales.
        let _lock = gpu_lock();
        let f = fonts();
        let mut r = renderer(f.atlas());
        for (scale, name) in [(1.0, "widgets_100"), (1.5, "widgets_150"), (2.0, "widgets_200")] {
            let theme = Theme::dark().with_reduced_motion(true);
            let mut h = Harness::new(&theme);
            let frame = |h: &mut Harness, events: Vec<Event>| {
                h.time += 0.016;
                let mut q = InputQueue { events, time: h.time };
                h.ui.begin_frame(&mut q, Size::new(340.0, 300.0), scale);
                gallery(&mut h.ui, &mut h.model);
                h.ui.end_frame();
            };
            frame(&mut h, vec![]);
            frame(&mut h, vec![]);
            frame(&mut h, vec![Event::Key { key: Key::Tab, pressed: true, repeat: false }]);
            let list = h.ui.draw_list();
            let (w, hgt) = ((340.0 * scale) as u32, (300.0 * scale) as u32);
            let rgba = render(&mut r, list, w, hgt, theme.palette.bg);
            assert!(r.stats().draw_calls <= 2, "{:?}", r.stats());
            assert_golden(name, w, hgt, &rgba, 2, (w * hgt / 1000) as usize);
        }
        // The light theme with a text selection.
        let theme = Theme::light().with_reduced_motion(true);
        let mut h = Harness::new(&theme);
        h.frame(vec![], 1.0);
        h.frame(vec![], 1.0);
        let field = h.rect_of(Role::TextField, "Name");
        h.click(field.center());
        h.key(Key::Home, Modifiers::SHIFT);
        let rgba = render(&mut r, h.ui.draw_list(), 340, 300, theme.palette.bg);
        assert_golden("widgets_light_selection", 340, 300, &rgba, 2, 102);
    }
    #[test]
    fn ui_06_text_area_lines_wrap_and_submit() {
        let text = std::cell::RefCell::new(String::new());
        let submitted = std::cell::RefCell::new(0);
        let mut build = |ui: &mut Ui| {
            ui.begin(Style { padding: Insets::all(10.0), width: Dim::Px(260.0), ..Style::column() });
            if TextArea::new("notes", &mut text.borrow_mut()).rows(3).placeholder("Notes").show(ui).submitted {
                *submitted.borrow_mut() += 1;
            }
            ui.end();
        };
        let mut d = crate::tests::Driver::new(&Theme::dark().with_reduced_motion(true), 300.0, 200.0);
        d.frame(vec![], &mut build);
        d.frame(vec![], &mut build);
        d.click(d.rect_of("Notes").center(), &mut build);
        d.frame("first line".chars().map(Event::Text).collect(), &mut build);
        d.key(Key::Enter, Modifiers::NONE, &mut build);
        d.frame("second".chars().map(Event::Text).collect(), &mut build);
        assert_eq!(*text.borrow(), "first line\nsecond");
        // Up goes to the same column of the line above; Home and End stay on the line.
        d.key(Key::Up, Modifiers::NONE, &mut build);
        d.frame(vec![Event::Text('X')], &mut build);
        let t = text.borrow().clone();
        let (l1, l2) = t.split_once('\n').unwrap_or_default();
        assert!(l1.len() == 11 && l1.contains('X') && l2 == "second", "{t:?}");
        let first = l1.to_owned();
        d.key(Key::End, Modifiers::NONE, &mut build);
        d.frame(vec![Event::Text('!')], &mut build);
        d.key(Key::Down, Modifiers::NONE, &mut build);
        d.key(Key::Home, Modifiers::NONE, &mut build);
        d.frame(vec![Event::Text('>')], &mut build);
        assert_eq!(*text.borrow(), format!("{first}!\n>second"));
        // Paste keeps line breaks; undo takes it back; Ctrl+Enter submits.
        d.key(Key::End, Modifiers::CTRL, &mut build);
        d.frame(vec![Event::Modifiers(Modifiers::NONE), Event::Paste("\r\nthird\r\nfourth".into())], &mut build);
        assert_eq!(*text.borrow(), format!("{first}!\n>second\nthird\nfourth"));
        d.key(Key::Letter('Z'), Modifiers::CTRL, &mut build);
        assert_eq!(*text.borrow(), format!("{first}!\n>second"));
        d.key(Key::Enter, Modifiers::CTRL, &mut build);
        assert_eq!(*submitted.borrow(), 1);
        assert_eq!(*text.borrow(), format!("{first}!\n>second"), "Ctrl+Enter does not insert a line");
        // Long text wraps at the width and the view follows the caret down.
        let long = "word ".repeat(60);
        d.frame(vec![Event::Modifiers(Modifiers::NONE), Event::Paste(long)], &mut build);
        let field = d.semantic("Notes").map_or(Rect::ZERO, |s| s.rect);
        assert!(field.h < 120.0, "three rows: {field:?}");
        let layout_lines =
            d.ui.text_layout(
                &text.borrow(),
                d.ui.theme().text.body,
                Some(field.w - 2.0 * d.ui.theme().metrics.control_pad_x),
            )
            .lines
            .len();
        assert!(layout_lines > 6, "{layout_lines} lines");
        // [AC-WP12-2] The wrapped, scrolled area with its caret, looked at by eye.
        d.key(Key::Home, Modifiers::SHIFT, &mut build);
        d.golden("text_area");
    }
}
