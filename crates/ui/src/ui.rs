//! The immediate-mode core: ids, the state arena, input routing, focus, deferred painting and the
//! semantic description `[UI-01]` `[UI-06]` `[UI-09]`.
//!
//! # A frame
//!
//! ```text
//! ui.begin_frame(&mut queue, viewport, scale);   // input events -> pointer, keys, focus moves
//! ... application and widget code ...            // boxes, interactions, paint ops, semantics
//! let output = ui.end_frame();                   // layout, hit list, draw list, output
//! renderer.prepare(device, queue, ui.draw_list(), w, h); renderer.record(..);
//! ```
//!
//! Application code declares boxes ([`Ui::node`], [`Ui::begin`] / [`Ui::end`]) with layout
//! [`Style`]s, asks for interaction on them ([`Ui::interact`]) and records [`Paint`] operations whose
//! geometry is expressed relative to a box ([`Area`], [`Anchor`]). At [`Ui::end_frame`] the layout
//! is computed and the paint operations are replayed with the final rectangles, so what is drawn
//! is always this frame's layout. Interaction uses the rectangles of the previous frame, which is
//! what the user saw when they pointed or clicked.
//!
//! **Ids** ([`Id`]) are hashes of labels within the id stack ([`Ui::push_id`]); a widget's id must
//! be the same every frame. **State** that outlives a frame lives in the arena: [`WidgetState`]
//! (rectangle, hover / press / focus animation) for every interactive id, and typed per-widget
//! memory ([`Ui::memory`]). Entries not seen for [`Ui::FORGET_AFTER`] frames are dropped.
//!
//! **Focus** `[UI-09]`: focusable widgets register in order; Tab and Shift+Tab move the focus
//! through last frame's order, a click focuses what it hits, Escape clears the focus. The focus
//! ring is drawn only after keyboard navigation (`focus_visible`), like CSS `:focus-visible`.
//!
//! **Semantics** `[UI-09]`: widgets describe themselves ([`Ui::semantic`]: role, label, value);
//! after `end_frame`, [`Ui::semantics`] lists them with their rectangles and focus, in a stable
//! order, for a future screen-reader bridge and for tests and the agent.
//!
//! **No steady-state allocation** `[PERF-R1]`: every per-frame vector and the text arena keep their
//! memory; the state arena only grows when new widgets appear.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::color::Color;
use crate::geom::{Point, Rect, Size};
use crate::input::{Event, InputQueue, Key, Modifiers, PointerButton, TouchPhase};
use crate::layout::{LayoutTree, NodeId, Style};
use crate::paint::{Axis, Corners, DrawList, Layer, PaintStats, TextureId};
use crate::text::{Align, Fonts, TextCache, TextLayout, draw_layout};
use crate::theme::{TextStyle, Theme};

/// A stable widget identity: a hash of its label within the id stack.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Id(pub u64);

impl Id {
    /// The root of every id path.
    pub const ROOT: Self = Self(0xCBF2_9CE4_8422_2325);

    /// The id of `label` under `self`.
    #[must_use]
    pub fn with(self, label: &str) -> Self {
        let mut h = self.0;
        for &b in label.as_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3);
        }
        // Mark the end of the label so ("ab", "c") and ("a", "bc") differ.
        Self((h ^ 0xFF).wrapping_mul(0x0100_0000_01B3))
    }

    /// The id of item `index` under `self` (rows of a list).
    #[must_use]
    pub fn with_index(self, index: u64) -> Self {
        let mut h = self.0;
        for b in index.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3);
        }
        Self((h ^ 0xFE).wrapping_mul(0x0100_0000_01B3))
    }
}

/// A pass-through hasher for keys that are hashes already.
#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3);
        }
    }

    fn write_u64(&mut self, v: u64) {
        self.0 ^= v;
    }
}

type IdMap<V> = HashMap<Id, V, BuildHasherDefault<IdHasher>>;

/// Typed per-widget memory: (id, type) to (last frame used, value).
type Memory = HashMap<(Id, TypeId), (u64, Box<dyn Any>), BuildHasherDefault<IdHasher>>;

/// One coordinate relative to a box: `box.start + frac × box.size + px`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Anchor {
    /// The fraction of the box's size.
    pub frac: f32,
    /// Logical pixels added.
    pub px: f32,
}

impl Anchor {
    /// The box's start edge plus `px`.
    #[inline]
    pub const fn start(px: f32) -> Self {
        Self { frac: 0.0, px }
    }

    /// The box's end edge plus `px`.
    #[inline]
    pub const fn end(px: f32) -> Self {
        Self { frac: 1.0, px }
    }

    /// A fraction of the box plus `px`.
    #[inline]
    pub const fn at(frac: f32, px: f32) -> Self {
        Self { frac, px }
    }

    #[inline]
    fn resolve(self, start: f32, size: f32) -> f32 {
        start + self.frac * size + self.px
    }
}

/// A rectangle relative to a box (resolved when the layout is known).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Area {
    /// Left edge.
    pub x0: Anchor,
    /// Top edge.
    pub y0: Anchor,
    /// Right edge.
    pub x1: Anchor,
    /// Bottom edge.
    pub y1: Anchor,
}

impl Area {
    /// The whole box.
    pub const FULL: Self =
        Self { x0: Anchor::start(0.0), y0: Anchor::start(0.0), x1: Anchor::end(0.0), y1: Anchor::end(0.0) };

    /// The box shrunk by `d` on every side.
    pub const fn inset(d: f32) -> Self {
        Self { x0: Anchor::start(d), y0: Anchor::start(d), x1: Anchor::end(-d), y1: Anchor::end(-d) }
    }

    /// A rectangle at a pixel offset from the box's top-left corner.
    pub const fn offset(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x0: Anchor::start(x), y0: Anchor::start(y), x1: Anchor::start(x + w), y1: Anchor::start(y + h) }
    }

    /// The rectangle in `rect`'s coordinates.
    pub fn resolve(&self, rect: Rect) -> Rect {
        let x0 = self.x0.resolve(rect.x, rect.w);
        let y0 = self.y0.resolve(rect.y, rect.h);
        let x1 = self.x1.resolve(rect.x, rect.w);
        let y1 = self.y1.resolve(rect.y, rect.h);
        Rect::new(x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0))
    }
}

/// Text stored in the frame's text arena ([`Ui::intern`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextRef {
    start: u32,
    end: u32,
}

/// A deferred paint operation, resolved against its box after layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Paint {
    /// A rounded rectangle, optionally bordered (`border` logical pixels inside the edge).
    Rect {
        /// Where.
        area: Area,
        /// Fill.
        fill: Color,
        /// Corner radii.
        radius: Corners,
        /// Border width (0 for none).
        border: f32,
        /// Border colour.
        border_color: Color,
    },
    /// A soft shadow.
    Shadow {
        /// The casting shape.
        area: Area,
        /// Colour.
        color: Color,
        /// Corner radius.
        radius: f32,
        /// Blur.
        blur: f32,
    },
    /// A linear gradient.
    Gradient {
        /// Where.
        area: Area,
        /// Start colour.
        from: Color,
        /// End colour.
        to: Color,
        /// Corner radii.
        radius: Corners,
        /// Direction.
        axis: Axis,
    },
    /// A filled circle.
    Circle {
        /// Centre x.
        cx: Anchor,
        /// Centre y.
        cy: Anchor,
        /// Radius.
        radius: f32,
        /// Colour.
        color: Color,
    },
    /// A line with round caps.
    Line {
        /// Start x.
        ax: Anchor,
        /// Start y.
        ay: Anchor,
        /// End x.
        bx: Anchor,
        /// End y.
        by: Anchor,
        /// Width.
        width: f32,
        /// Colour.
        color: Color,
    },
    /// Text: one line centred vertically in `area`, or wrapped to its width from its top.
    Text {
        /// Where.
        area: Area,
        /// The text.
        text: TextRef,
        /// Face and size.
        style: TextStyle,
        /// Colour.
        color: Color,
        /// Horizontal alignment.
        align: Align,
        /// Wrap to the area's width.
        wrap: bool,
    },
    /// A registered image.
    Image {
        /// Where.
        area: Area,
        /// Which.
        texture: TextureId,
        /// Part of the texture, normalised.
        uv: [f32; 4],
        /// Multiplier.
        tint: Color,
        /// Corner radius.
        radius: f32,
    },
    /// The theme's focus ring around `area` `[UI-09]`.
    FocusRing {
        /// The focused control's shape.
        area: Area,
        /// Its corner radius.
        radius: f32,
    },
    /// Narrows the clip to `area` until the matching [`Paint::PopClip`].
    PushClip {
        /// The clip.
        area: Area,
    },
    /// Replaces the clip with `area` (not intersected with the enclosing clip) until the matching
    /// [`Paint::PopClip`]: the content of an overlay ([`Ui::begin_overlay`]).
    PushOverlayClip {
        /// The clip.
        area: Area,
    },
    /// Restores the previous clip.
    PopClip,
    /// Switches the drawing layer.
    Layer(Layer),
}

/// What an interaction responds to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sense {
    /// Clicks (and Enter / Space when focused).
    pub click: bool,
    /// Drags.
    pub drag: bool,
    /// Keyboard focus.
    pub focus: bool,
}

impl Sense {
    /// Hover only.
    pub const HOVER: Self = Self { click: false, drag: false, focus: false };
    /// Clicks, focusable (buttons).
    pub const CLICK: Self = Self { click: true, drag: false, focus: true };
    /// Drags and clicks, focusable (sliders).
    pub const DRAG: Self = Self { click: true, drag: true, focus: true };
}

/// The outcome of [`Ui::interact`] for one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Response {
    /// The widget.
    pub id: Id,
    /// Its rectangle as drawn last frame (clipped), logical pixels.
    pub rect: Rect,
    /// The pointer is over it and nothing else holds the pointer.
    pub hovered: bool,
    /// A button went down on it this frame.
    pub pressed: bool,
    /// It holds the pointer (pressed and not yet released).
    pub held: bool,
    /// Clicked (released over it without dragging, or Enter / Space while focused).
    pub clicked: bool,
    /// The second click of a double click.
    pub double_clicked: bool,
    /// The pointer moved past the drag threshold this frame.
    pub drag_started: bool,
    /// Being dragged.
    pub dragged: bool,
    /// Pointer movement this frame while dragged.
    pub drag_delta: Point,
    /// It has the keyboard focus.
    pub focused: bool,
    /// Its focus ring should show (keyboard navigation).
    pub focus_visible: bool,
    /// Eased 0..1 hover amount (for fills).
    pub hover_t: f32,
    /// Eased 0..1 press amount.
    pub press_t: f32,
    /// The widget changed its value this frame (set by the widget).
    pub changed: bool,
    /// The secondary button went down on it, or Shift+F10 / the context-menu key was pressed while
    /// it had the focus: open its context menu `[UI-07]`.
    pub secondary_clicked: bool,
}

/// State kept for an interactive id across frames.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WidgetState {
    /// Its rectangle last frame (clipped).
    pub rect: Rect,
    /// Its unclipped rectangle last frame.
    pub full_rect: Rect,
    /// Eased hover.
    pub hover_t: f32,
    /// Eased press.
    pub press_t: f32,
    /// Eased focus.
    pub focus_t: f32,
    /// The last frame it was used.
    pub seen: u64,
}

/// The role of a widget in the semantic description `[UI-09]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// A window or the root.
    Window,
    /// A panel.
    Panel,
    /// A group of related controls.
    Group,
    /// Static text.
    Label,
    /// A push button.
    Button,
    /// A slider.
    Slider,
    /// A text field.
    TextField,
    /// A checkbox or toggle.
    Toggle,
    /// A scroll area.
    ScrollArea,
    /// An image.
    Image,
    /// A role defined by a later widget (stable name, `"tree.row"`).
    Custom(&'static str),
}

/// The value of a widget in the semantic description.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SemanticValue {
    /// No value.
    #[default]
    None,
    /// A number in a range.
    Number {
        /// The value.
        value: f64,
        /// The smallest value.
        min: f64,
        /// The largest value.
        max: f64,
    },
    /// Text (in the frame's arena).
    Text(TextRef),
    /// On or off.
    Toggle(bool),
}

/// One entry of the semantic description.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Semantic {
    /// The widget.
    pub id: Id,
    /// What it is.
    pub role: Role,
    /// Its label ([`Ui::text`]).
    pub label: TextRef,
    /// Its value.
    pub value: SemanticValue,
    /// The enclosing semantic group, if any.
    pub parent: Option<Id>,
    /// Its rectangle this frame (after `end_frame`).
    pub rect: Rect,
    /// Whether it has the keyboard focus.
    pub focused: bool,
    /// Whether it is disabled.
    pub disabled: bool,
}

/// The pointer shape the shell should show.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CursorIcon {
    /// The arrow.
    #[default]
    Default,
    /// A hand over something clickable.
    Pointer,
    /// The I-beam over text.
    Text,
    /// Something can be grabbed.
    Grab,
    /// Something is being dragged.
    Grabbing,
    /// Horizontal resize.
    ResizeHorizontal,
    /// Vertical resize.
    ResizeVertical,
    /// Not allowed.
    NotAllowed,
}

/// What the shell does after a frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameOutput {
    /// The pointer shape.
    pub cursor: CursorIcon,
    /// Text to put on the clipboard.
    pub copied: Option<String>,
    /// A text field has the focus: enable the IME / on-screen keyboard.
    pub text_input: bool,
    /// Where the IME candidate window should go (logical pixels).
    pub ime_area: Option<Rect>,
    /// An animation is running: draw another frame even without input.
    pub animating: bool,
    /// The draw list's counters.
    pub paint: PaintStats,
    /// What changed on screen since the previous frame `[UI-01]`.
    pub dirty: Dirty,
}

/// The part of the target a frame changed `[UI-01]`, from comparing its draw list with the previous
/// frame's: the shell skips rendering a [`Dirty::Clean`] frame and redraws only the rectangle of a
/// [`Dirty::Region`] (scissored, the rest of the target kept:
/// [`crate::render::UiRenderer::record_region`]). Regions are reported only when the UI paints its
/// own opaque background ([`Ui::set_background`]); otherwise every change is [`Dirty::Full`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Dirty {
    /// Nothing changed: the previous image is still right.
    Clean,
    /// Only this rectangle (physical pixels, whole pixels) changed.
    Region(Rect),
    /// Everything must be redrawn (first frame, resize, scale change, no background).
    #[default]
    Full,
}

/// The pen's state as the platform reported it `[UI-06]`: pressure and tilt for widgets that want
/// them (brush previews, the 2D pad). Present from the first pen event until the pen lifts (the
/// primary button is released) or leaves the window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PenState {
    /// Pressure, 0 to 1.
    pub pressure: f32,
    /// Tilt from vertical, radians, along X and Y.
    pub tilt: [f32; 2],
}

/// The key that opens a context menu on Windows and Linux keyboards (the shell maps it to this name).
pub const CONTEXT_MENU_KEY: Key = Key::Named("ContextMenu");

/// The UI scales the toolkit supports `[UI-09]`: 75 % to 200 %. The shell multiplies the clamped
/// user scale with the display's scale factor and passes the product to [`Ui::begin_frame`].
pub const UI_SCALE_RANGE: (f32, f32) = (0.75, 2.0);

/// Clamps a user UI scale to [`UI_SCALE_RANGE`] (non-finite values give 1).
pub fn clamp_ui_scale(scale: f32) -> f32 {
    if scale.is_finite() { scale.clamp(UI_SCALE_RANGE.0, UI_SCALE_RANGE.1) } else { 1.0 }
}

/// Pointer state.
#[derive(Clone, Copy, Debug, Default)]
struct Pointer {
    pos: Option<Point>,
    delta: Point,
    down: [bool; 3],
    pressed_now: [bool; 3],
    released_now: [bool; 3],
    press_pos: Point,
    dragging: bool,
    drag_started_now: bool,
    click_count: u32,
    last_click_time: f64,
    last_click_pos: Point,
    wheel: Point,
}

/// A finger being tracked.
#[derive(Clone, Copy, Debug)]
struct Finger {
    id: u64,
    start: Point,
    last: Point,
    since: f64,
}

/// Touch state: the primary finger (the pointer) and a second one (pan and pinch).
#[derive(Clone, Copy, Debug, Default)]
struct Touch {
    primary: Option<Finger>,
    second: Option<Finger>,
    /// A long press fired or a second finger came: the primary finger no longer clicks.
    spent: bool,
}

/// A two-finger gesture of this frame `[UI-06]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gesture {
    /// Movement of the midpoint between the fingers, logical pixels.
    pub pan: Point,
    /// Change of the distance between the fingers as a factor (1: none, above 1: spreading).
    pub zoom: f32,
    /// The midpoint between the fingers.
    pub center: Point,
}

impl Default for Gesture {
    fn default() -> Self {
        Self { pan: Point::ZERO, zoom: 1.0, center: Point::ZERO }
    }
}

/// A widget registered this frame.
#[derive(Clone, Copy, Debug)]
struct FrameWidget {
    id: Id,
    node: NodeId,
    clip: u32,
    layer: Layer,
    sense: Sense,
    scroll: bool,
}

/// A clip region registered this frame: a box and the enclosing clip.
#[derive(Clone, Copy, Debug)]
struct Clip {
    node: NodeId,
    area: Area,
    parent: u32,
}

/// A hit-test entry from last frame.
#[derive(Clone, Copy, Debug)]
struct Hit {
    id: Id,
    rect: Rect,
    layer: Layer,
    sense: Sense,
    scroll: bool,
}

/// A deferred paint operation with its box.
#[derive(Clone, Copy, Debug)]
struct Op {
    node: NodeId,
    paint: Paint,
}

/// No clip / no node index.
const NONE: u32 = u32::MAX;

/// Scroll state of a scroll area.
#[derive(Clone, Copy, Debug, Default)]
struct ScrollState {
    offset: f32,
    target: f32,
    content: f32,
    viewport: f32,
    thumb_t: f32,
    dragging: bool,
    drag_from: f32,
}

/// The immediate-mode UI.
pub struct Ui {
    theme: Theme,
    fonts: Fonts,
    cache: TextCache,
    list: DrawList,
    tree: LayoutTree,
    scale: f32,
    viewport: Rect,
    time: f64,
    dt: f32,
    frame: u64,
    root: NodeId,
    id_stack: Vec<Id>,
    parents: Vec<NodeId>,
    ops: Vec<Op>,
    text_arena: String,
    widgets: Vec<FrameWidget>,
    clips: Vec<Clip>,
    clip_stack: Vec<u32>,
    layer: Layer,
    semantics: Vec<Semantic>,
    semantic_nodes: Vec<(usize, NodeId)>,
    semantic_stack: Vec<Id>,
    states: IdMap<WidgetState>,
    memory: Memory,
    scrolls: IdMap<ScrollState>,
    hits: Vec<Hit>,
    focus_order: Vec<Id>,
    prev_focus_order: Vec<Id>,
    focused: Option<Id>,
    focus_visible: bool,
    focus_seen: bool,
    pointer: Pointer,
    active: Option<Id>,
    hovered: Option<Id>,
    wheel_target: Option<Id>,
    modifiers: Modifiers,
    keys: Vec<(Key, Modifiers, bool)>,
    typed: String,
    paste: Option<String>,
    preedit: String,
    committed: String,
    output: FrameOutput,
    animating: bool,
    gc_scratch: Vec<Id>,
    pen: Option<PenState>,
    tip_rest: Option<(Id, f64)>,
    trap: Option<(usize, usize)>,
    trap_start: Option<usize>,
    prev_trap: Option<(usize, usize)>,
    background: Option<Color>,
    prev_instances: Vec<crate::paint::Instance>,
    prev_target: (u32, u32, u32),
    overlay_layers: Vec<Layer>,
    grab_depth: u32,
    grabbed: bool,
    prev_grabbed: bool,
    touch: Touch,
    gesture: Gesture,
}

impl std::fmt::Debug for Ui {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ui").field("frame", &self.frame).field("widgets", &self.widgets.len()).finish_non_exhaustive()
    }
}

/// Records a primary button press at `pos` (click counting for double clicks); returns true.
fn press_primary(p: &mut Pointer, pos: Point, time: f64, m: &crate::theme::Motion) -> bool {
    p.pos = Some(pos);
    p.down[0] = true;
    p.pressed_now[0] = true;
    p.press_pos = pos;
    p.dragging = false;
    let near = pos.distance(p.last_click_pos) <= m.double_click_distance;
    if near && time - p.last_click_time <= f64::from(m.double_click_time) {
        p.click_count += 1;
    } else {
        p.click_count = 1;
    }
    p.last_click_time = time;
    p.last_click_pos = pos;
    true
}

/// Moves `value` toward `target` by the standard exponential ease (reference section 16.5); snaps
/// when within 0.002 or when motion is reduced.
pub fn ease(value: f32, target: f32, dt: f32, rate: f32, reduced: bool) -> f32 {
    if reduced || (target - value).abs() < 0.002 {
        return target;
    }
    value + (target - value) * (1.0 - (-dt * rate).exp())
}

impl Ui {
    /// Frames an unused id keeps its state.
    pub const FORGET_AFTER: u64 = 300;

    /// A UI with these fonts and theme.
    pub fn new(fonts: Fonts, theme: Theme) -> Self {
        Self {
            theme,
            fonts,
            cache: TextCache::new(),
            list: DrawList::new(),
            tree: LayoutTree::new(),
            scale: 1.0,
            viewport: Rect::ZERO,
            time: 0.0,
            dt: 0.0,
            frame: 0,
            root: NodeId(0),
            id_stack: Vec::new(),
            parents: Vec::new(),
            ops: Vec::new(),
            text_arena: String::new(),
            widgets: Vec::new(),
            clips: Vec::new(),
            clip_stack: Vec::new(),
            layer: Layer::Base,
            semantics: Vec::new(),
            semantic_nodes: Vec::new(),
            semantic_stack: Vec::new(),
            states: IdMap::default(),
            memory: HashMap::default(),
            scrolls: IdMap::default(),
            hits: Vec::new(),
            focus_order: Vec::new(),
            prev_focus_order: Vec::new(),
            focused: None,
            focus_visible: false,
            focus_seen: false,
            pointer: Pointer::default(),
            active: None,
            hovered: None,
            wheel_target: None,
            modifiers: Modifiers::NONE,
            keys: Vec::new(),
            typed: String::new(),
            paste: None,
            preedit: String::new(),
            committed: String::new(),
            output: FrameOutput::default(),
            animating: false,
            gc_scratch: Vec::new(),
            pen: None,
            tip_rest: None,
            trap: None,
            trap_start: None,
            prev_trap: None,
            background: None,
            prev_instances: Vec::new(),
            prev_target: (0, 0, 0),
            overlay_layers: Vec::new(),
            grab_depth: 0,
            grabbed: false,
            prev_grabbed: false,
            touch: Touch::default(),
            gesture: Gesture::default(),
        }
    }

    /// The theme.
    #[inline]
    pub fn theme(&self) -> &Theme {
        &self.theme
    }

    /// Replaces the theme (applies from the next paint operation).
    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }

    /// The fonts.
    #[inline]
    pub fn fonts(&self) -> &Fonts {
        &self.fonts
    }

    /// Physical pixels per logical pixel this frame.
    #[inline]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Seconds since the previous frame (clamped by the theme's `dt_clamp`).
    #[inline]
    pub fn dt(&self) -> f32 {
        self.dt
    }

    /// The frame counter.
    #[inline]
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// The root box (the viewport).
    #[inline]
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// The draw list of the last finished frame.
    #[inline]
    pub fn draw_list(&self) -> &DrawList {
        &self.list
    }

    /// The layout of this frame (rectangles are final after `end_frame`).
    #[inline]
    pub fn layout(&self) -> &LayoutTree {
        &self.tree
    }

    /// The modifiers held.
    #[inline]
    pub fn modifiers(&self) -> Modifiers {
        self.modifiers
    }

    /// The pointer position, if inside the window.
    #[inline]
    pub fn pointer(&self) -> Option<Point> {
        self.pointer.pos
    }

    /// The viewport of this frame, logical pixels.
    #[inline]
    pub fn viewport(&self) -> Rect {
        self.viewport
    }

    /// The input clock of this frame, seconds.
    #[inline]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// The layer the following widgets and paint operations go to.
    #[inline]
    pub fn layer(&self) -> Layer {
        self.layer
    }

    /// The two-finger gesture of this frame `[UI-06]` (pan and pinch; `zoom` is 1 without one).
    pub fn gesture(&self) -> Gesture {
        self.gesture
    }

    /// The pen's pressure and tilt while a pen is down `[UI-06]` (`None` for a mouse or touch).
    #[inline]
    pub fn pen(&self) -> Option<PenState> {
        self.pen
    }

    /// Whether the primary button is down anywhere.
    #[inline]
    pub fn pointer_down(&self) -> bool {
        self.pointer.down[0]
    }

    /// Whether the primary button went down this frame (anywhere).
    #[inline]
    pub fn pointer_pressed(&self) -> bool {
        self.pointer.pressed_now[0]
    }

    /// Whether the primary button was released this frame (anywhere).
    #[inline]
    pub fn pointer_released(&self) -> bool {
        self.pointer.released_now[0]
    }

    /// The wheel movement of this frame in logical pixels (positive `y`: towards the top).
    #[inline]
    pub fn wheel(&self) -> Point {
        self.pointer.wheel
    }

    /// How far the pointer moved since the previous frame.
    #[inline]
    pub fn pointer_delta(&self) -> Point {
        self.pointer.delta
    }

    /// Takes the first key press of this frame that `keymap` binds to an action and returns the
    /// action `[UI-06]`. Call it after building the widgets, so a focused widget gets its keys
    /// first; while a text field has the focus, shortcuts without Ctrl, Alt or the logo key are
    /// left to the text.
    pub fn shortcut(&mut self, keymap: &crate::input::Keymap) -> Option<&'static str> {
        let typing = self.output.text_input;
        for entry in &mut self.keys {
            if entry.2 {
                continue;
            }
            let (key, mods) = (entry.0, entry.1);
            if typing && !(mods.ctrl || mods.alt || mods.meta) {
                continue;
            }
            if let Some(action) = keymap.action(crate::input::Shortcut::new(key, mods)) {
                entry.2 = true;
                return Some(action);
            }
        }
        None
    }

    /// The widget under the pointer (from last frame's rectangles), if any.
    #[inline]
    pub fn hovered(&self) -> Option<Id> {
        self.hovered
    }

    /// Paints an opaque background over the whole viewport at the start of every frame. With a
    /// background the frame output reports dirty regions ([`Dirty::Region`]) `[UI-01]`.
    pub fn set_background(&mut self, color: Option<Color>) {
        self.background = color.filter(|c| c.a >= 1.0);
    }

    /// Whether a tooltip for the widget of `r` should show this frame `[UI-09]`: after the pointer
    /// has rested on it for the theme's tooltip delay (no button held), or at once while it has the
    /// keyboard focus after keyboard navigation. Keeps frames coming while the delay runs.
    pub fn tip_ready(&mut self, r: &Response) -> bool {
        if r.focus_visible {
            return true;
        }
        let resting = r.hovered && !self.pointer.down[0] && !r.held;
        match self.tip_rest {
            Some((id, since)) if id == r.id => {
                if !resting {
                    self.tip_rest = None;
                    return false;
                }
                let due = self.time - since >= f64::from(self.theme.motion.tooltip_delay);
                if !due {
                    self.animating = true;
                }
                due
            }
            _ => {
                if resting {
                    self.tip_rest = Some((r.id, self.time));
                    self.animating = true;
                }
                false
            }
        }
    }

    /// Starts a keyboard grab: an open popup (menu, dropdown list) takes the keyboard from the
    /// widget that opened it `[UI-09]`. While a grab was held last frame, widgets registered outside
    /// [`Ui::begin_key_grab`] / [`Ui::end_key_grab`] get no keys (their [`Response::focused`] is
    /// false, Enter and Space do not click them) and Tab is left to the popup instead of moving the
    /// focus. Grabs nest.
    pub fn begin_key_grab(&mut self) {
        self.grab_depth += 1;
    }

    /// Ends the section started by [`Ui::begin_key_grab`]; `hold` keeps the grab for the next frame
    /// (false when the popup closed this frame).
    pub fn end_key_grab(&mut self, hold: bool) {
        self.grab_depth = self.grab_depth.saturating_sub(1);
        self.grabbed |= hold;
    }

    /// Starts a focus trap: Tab and Shift+Tab cycle only through the focusable widgets registered
    /// until [`Ui::end_focus_trap`] (an open modal dialog) `[UI-09]`. One trap per frame; the last
    /// one wins.
    pub fn begin_focus_trap(&mut self) {
        self.trap_start = Some(self.focus_order.len());
    }

    /// Ends the focus trap started by [`Ui::begin_focus_trap`]. When the focus is outside the trap
    /// it moves to the trap's first widget.
    pub fn end_focus_trap(&mut self) {
        if let Some(start) = self.trap_start.take() {
            let end = self.focus_order.len();
            if end > start {
                self.trap = Some((start, end));
                let inside = self.focused.is_some_and(|f| self.focus_order[start..end].contains(&f));
                if !inside {
                    self.focused = self.focus_order.get(start).copied();
                    self.focus_seen = true;
                }
            }
        }
    }

    /// Starts a frame: consumes `input`, sets the viewport (logical pixels) and the scale.
    #[allow(clippy::too_many_lines)]
    pub fn begin_frame(&mut self, input: &mut InputQueue, viewport: Size, scale: f32) {
        self.frame += 1;
        self.prev_grabbed = self.grabbed;
        self.grabbed = false;
        self.grab_depth = 0;
        let dt = if self.frame == 1 { 0.0 } else { (input.time - self.time) as f32 };
        self.dt = dt.clamp(0.0, self.theme.motion.dt_clamp);
        self.time = input.time;
        self.scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
        self.viewport = Rect::new(0.0, 0.0, viewport.w.max(0.0), viewport.h.max(0.0));
        self.cache.begin_frame();
        self.tree.clear();
        self.ops.clear();
        self.text_arena.clear();
        self.widgets.clear();
        self.clips.clear();
        self.clip_stack.clear();
        self.semantics.clear();
        self.semantic_nodes.clear();
        self.semantic_stack.clear();
        self.id_stack.clear();
        self.parents.clear();
        self.layer = Layer::Base;
        std::mem::swap(&mut self.focus_order, &mut self.prev_focus_order);
        self.focus_order.clear();
        self.focus_seen = false;
        self.keys.clear();
        self.typed.clear();
        self.paste = None;
        self.committed.clear();
        self.output.copied = None;
        self.output.cursor = CursorIcon::Default;
        self.output.text_input = false;
        self.output.ime_area = None;
        self.animating = false;
        self.overlay_layers.clear();
        self.prev_trap = self.trap.take();
        self.trap_start = None;

        // Pointer and keys.
        let p = &mut self.pointer;
        let before = p.pos;
        p.pressed_now = [false; 3];
        p.released_now = [false; 3];
        p.drag_started_now = false;
        p.wheel = Point::ZERO;
        self.gesture = Gesture::default();
        let mut press_now = false;
        for event in input.events.drain(..) {
            match event {
                Event::Touch { id, phase, pos } => {
                    if !(pos.x.is_finite() && pos.y.is_finite()) {
                        continue;
                    }
                    let t = &mut self.touch;
                    let finger = Finger { id, start: pos, last: pos, since: input.time };
                    match phase {
                        TouchPhase::Start if t.primary.is_none() => {
                            t.primary = Some(finger);
                            t.spent = false;
                            press_now |= press_primary(p, pos, input.time, &self.theme.motion);
                        }
                        TouchPhase::Start if t.second.is_none() => {
                            t.second = Some(finger);
                            // Two fingers: the primary press ends without a click.
                            t.spent = true;
                            p.down[0] = false;
                            p.dragging = false;
                            self.active = None;
                            if let Some(a) = t.primary {
                                p.pos = Some(a.last.lerp(pos, 0.5));
                            }
                        }
                        TouchPhase::Start => {}
                        TouchPhase::Move => match (t.primary, t.second) {
                            (Some(a), Some(b)) if id == a.id || id == b.id => {
                                let (na, nb) = if id == a.id { (pos, b.last) } else { (a.last, pos) };
                                let (old_c, new_c) = (a.last.lerp(b.last, 0.5), na.lerp(nb, 0.5));
                                let (old_d, new_d) = (a.last.distance(b.last), na.distance(nb));
                                let pan = new_c - old_c;
                                self.gesture.pan = self.gesture.pan + pan;
                                if old_d > 1.0 && new_d > 1.0 {
                                    self.gesture.zoom *= new_d / old_d;
                                }
                                self.gesture.center = new_c;
                                p.wheel = Point::new(p.wheel.x + pan.x, p.wheel.y + pan.y);
                                p.pos = Some(new_c);
                                if id == a.id {
                                    t.primary = Some(Finger { last: pos, ..a });
                                } else {
                                    t.second = Some(Finger { last: pos, ..b });
                                }
                            }
                            (Some(a), None) if id == a.id => {
                                t.primary = Some(Finger { last: pos, ..a });
                                p.pos = Some(pos);
                            }
                            _ => {}
                        },
                        TouchPhase::End | TouchPhase::Cancel => {
                            let is_primary = t.primary.is_some_and(|a| a.id == id);
                            let is_second = t.second.is_some_and(|b| b.id == id);
                            if is_primary && !t.spent && p.down[0] {
                                p.pos = Some(pos);
                                p.down[0] = false;
                                if phase == TouchPhase::End {
                                    p.released_now[0] = true;
                                } else {
                                    self.active = None;
                                }
                            }
                            if is_primary {
                                // The other finger, if any, only ends the gesture.
                                t.primary = t.second.take();
                                if t.primary.is_none() {
                                    t.spent = false;
                                }
                            } else if is_second {
                                t.second = None;
                            }
                        }
                    }
                }
                Event::PointerMoved(pos) => p.pos = Some(pos),
                Event::PointerLeft => {
                    p.pos = None;
                    self.pen = None;
                }
                Event::Button { button, pressed, pos } => {
                    let b = button_index(button);
                    p.pos = Some(pos);
                    if pressed {
                        if b == 0 {
                            press_now |= press_primary(p, pos, input.time, &self.theme.motion);
                        } else {
                            p.down[b] = true;
                            p.pressed_now[b] = true;
                        }
                    } else {
                        p.down[b] = false;
                        p.released_now[b] = true;
                        if b == 0 {
                            self.pen = None;
                        }
                    }
                }
                Event::Wheel { delta, precise } => {
                    let k = if precise { 1.0 } else { self.theme.motion.wheel_line };
                    p.wheel = Point::new(p.wheel.x + delta.x * k, p.wheel.y + delta.y * k);
                }
                Event::Key { key, pressed, .. } => {
                    if pressed {
                        self.keys.push((key, self.modifiers, false));
                    }
                }
                Event::Modifiers(m) => self.modifiers = m,
                Event::Text(c) => {
                    if !c.is_control() {
                        self.typed.push(c);
                    }
                }
                Event::ImePreedit(s) => {
                    self.preedit.clear();
                    self.preedit.push_str(&s);
                }
                Event::ImeCommit(s) => {
                    self.preedit.clear();
                    self.committed.push_str(&s);
                }
                Event::Paste(s) => self.paste = Some(s),
                Event::WindowFocus(false) => {
                    p.down = [false; 3];
                    self.active = None;
                }
                Event::Pen { pressure, tilt } => {
                    let clean = |v: f32, lo: f32, hi: f32| if v.is_finite() { v.clamp(lo, hi) } else { 0.0 };
                    let half_pi = std::f32::consts::FRAC_PI_2;
                    self.pen = Some(PenState {
                        pressure: clean(pressure, 0.0, 1.0),
                        tilt: [clean(tilt[0], -half_pi, half_pi), clean(tilt[1], -half_pi, half_pi)],
                    });
                }
                Event::WindowFocus(true) => {}
            }
        }
        // A finger held still long enough is a secondary click (and no longer a primary press).
        if let Some(a) = self.touch.primary
            && self.touch.second.is_none()
            && !self.touch.spent
        {
            let still = a.last.distance(a.start) <= self.theme.motion.drag_threshold;
            if still && input.time - a.since >= crate::input::LONG_PRESS_SECONDS {
                self.touch.spent = true;
                p.down[0] = false;
                p.dragging = false;
                self.active = None;
                p.pressed_now[1] = true;
                p.released_now[1] = true;
            } else if still {
                self.animating = true;
            }
        }
        p.delta = match (before, p.pos) {
            (Some(a), Some(b)) => b - a,
            _ => Point::ZERO,
        };
        if p.down[0]
            && !p.dragging
            && let Some(pos) = p.pos
            && pos.distance(p.press_pos) > self.theme.motion.drag_threshold
        {
            p.dragging = true;
            p.drag_started_now = true;
        }

        // Hover target from last frame's hit list (topmost layer, then last registered).
        self.hovered = None;
        self.wheel_target = None;
        if let Some(pos) = self.pointer.pos {
            let mut best: Option<(u8, usize)> = None;
            let mut wheel_best: Option<(u8, usize)> = None;
            for (i, h) in self.hits.iter().enumerate() {
                if !h.rect.contains(pos) {
                    continue;
                }
                let key = (h.layer as u8, i);
                if h.scroll {
                    if wheel_best.is_none_or(|b| key >= b) {
                        wheel_best = Some(key);
                        self.wheel_target = Some(h.id);
                    }
                } else if best.is_none_or(|b| key >= b) {
                    best = Some(key);
                    self.hovered = Some(h.id);
                }
            }
        }
        if press_now {
            // A press outside every focusable widget clears the focus.
            let target = self.hovered;
            let focusable = target.is_some_and(|t| self.hits.iter().any(|h| h.id == t && h.sense.focus));
            if !focusable {
                self.focused = None;
            }
            self.focus_visible = false;
        }

        // Tab navigation through last frame's focus order.
        let mut i = 0;
        while i < self.keys.len() {
            let (key, mods, _) = self.keys[i];
            if key == Key::Tab && !mods.ctrl && !mods.alt && !self.prev_grabbed {
                self.move_focus(if mods.shift { -1 } else { 1 });
                self.keys.remove(i);
            } else {
                i += 1;
            }
        }

        let root = self.tree.add(None, Style::column());
        self.root = root;
        self.parents.push(root);
        if let Some(bg) = self.background {
            self.ops.push(Op {
                node: root,
                paint: Paint::Rect { area: Area::FULL, fill: bg, radius: Corners::ZERO, border: 0.0, border_color: bg },
            });
        }
    }

    fn move_focus(&mut self, step: i32) {
        // A focus trap (an open modal dialog) keeps Tab inside its controls.
        let order = match self.prev_trap {
            Some((a, b)) => self.prev_focus_order.get(a..b).unwrap_or(&[]),
            None => &self.prev_focus_order[..],
        };
        if order.is_empty() {
            return;
        }
        let n = i64::try_from(order.len()).unwrap_or(i64::MAX);
        let current = self.focused.and_then(|f| order.iter().position(|&o| o == f));
        let next = match current {
            Some(i) => (i64::try_from(i).unwrap_or(0) + i64::from(step)).rem_euclid(n),
            None if step > 0 => 0,
            None => n - 1,
        };
        self.focused = usize::try_from(next).ok().and_then(|i| order.get(i)).copied();
        self.focus_visible = true;
    }

    /// The id of `label` under the current id scope.
    pub fn id(&self, label: &str) -> Id {
        self.id_stack.last().copied().unwrap_or(Id::ROOT).with(label)
    }

    /// Enters an id scope (a panel, a list row).
    pub fn push_id(&mut self, id: Id) {
        self.id_stack.push(id);
    }

    /// Leaves the current id scope.
    pub fn pop_id(&mut self) {
        self.id_stack.pop();
    }

    /// The current parent box.
    #[inline]
    pub fn parent(&self) -> NodeId {
        self.parents.last().copied().unwrap_or(self.root)
    }

    /// Adds a box under the current parent.
    pub fn node(&mut self, style: Style) -> NodeId {
        let parent = self.parent();
        self.tree.add(Some(parent), style)
    }

    /// Adds a box and makes it the parent of the following boxes until [`Ui::end`].
    pub fn begin(&mut self, style: Style) -> NodeId {
        let node = self.node(style);
        self.parents.push(node);
        node
    }

    /// Closes the box opened by the last [`Ui::begin`].
    pub fn end(&mut self) {
        if self.parents.len() > 1 {
            self.parents.pop();
        }
    }

    /// Selects the layer of the following paint operations and widgets (popups, tooltips).
    pub fn set_layer(&mut self, layer: Layer) {
        self.layer = layer;
        let root = self.root;
        self.ops.push(Op { node: root, paint: Paint::Layer(layer) });
    }

    /// Records a paint operation on `node`.
    #[inline]
    pub fn paint(&mut self, node: NodeId, paint: Paint) {
        self.ops.push(Op { node, paint });
    }

    /// Copies `text` into the frame's arena (for paint operations and semantics).
    pub fn intern(&mut self, text: &str) -> TextRef {
        let start = u32::try_from(self.text_arena.len()).unwrap_or(u32::MAX);
        self.text_arena.push_str(text);
        let end = u32::try_from(self.text_arena.len()).unwrap_or(u32::MAX);
        TextRef { start, end }
    }

    /// The text of a [`TextRef`] of this frame.
    pub fn text(&self, r: TextRef) -> &str {
        self.text_arena.get(r.start as usize..r.end as usize).unwrap_or("")
    }

    /// Narrows the clip of the following paint operations and the hit areas of the following
    /// widgets to `area` of `node`.
    pub fn push_clip(&mut self, node: NodeId, area: Area) {
        let parent = self.clip_stack.last().copied().unwrap_or(NONE);
        let index = u32::try_from(self.clips.len()).unwrap_or(NONE);
        self.clips.push(Clip { node, area, parent });
        self.clip_stack.push(index);
        self.ops.push(Op { node, paint: Paint::PushClip { area } });
    }

    /// Opens an overlay: a box under the root placed at `rect` (logical pixels, viewport
    /// coordinates) on `layer`, outside every enclosing clip and box (menus, dropdown lists,
    /// tooltips, dialogs, floating panels, toasts). Its content is clipped to the viewport only, so
    /// shadows may spill out of the box. Close it with [`Ui::end_overlay`]. Returns the box.
    pub fn begin_overlay(&mut self, rect: Rect, layer: Layer, style: Style) -> NodeId {
        self.overlay_layers.push(self.layer);
        if self.layer != layer {
            self.set_layer(layer);
        }
        let root = self.root;
        let style = Style {
            position: crate::layout::Position::Absolute {
                left: Some(rect.x),
                top: Some(rect.y),
                right: None,
                bottom: None,
            },
            width: crate::layout::Dim::Px(rect.w.max(0.0)),
            height: crate::layout::Dim::Px(rect.h.max(0.0)),
            ..style
        };
        let node = self.tree.add(Some(root), style);
        let index = u32::try_from(self.clips.len()).unwrap_or(NONE);
        self.clips.push(Clip { node: root, area: Area::FULL, parent: NONE });
        self.clip_stack.push(index);
        self.ops.push(Op { node: root, paint: Paint::PushOverlayClip { area: Area::FULL } });
        self.parents.push(node);
        node
    }

    /// Closes the overlay opened by the last [`Ui::begin_overlay`] and restores the layer.
    pub fn end_overlay(&mut self) {
        self.pop_clip();
        self.end();
        if let Some(layer) = self.overlay_layers.pop()
            && layer != self.layer
        {
            self.set_layer(layer);
        }
    }

    /// Restores the clip in force before the last [`Ui::push_clip`].
    pub fn pop_clip(&mut self) {
        if self.clip_stack.pop().is_some() {
            let root = self.root;
            self.ops.push(Op { node: root, paint: Paint::PopClip });
        }
    }

    /// The layout of `text` (cached across frames).
    pub fn text_layout(&mut self, text: &str, style: TextStyle, max_width: Option<f32>) -> &TextLayout {
        self.cache.layout(&self.fonts, text, style, max_width)
    }

    /// The size of `text` on one line.
    pub fn measure_text(&mut self, text: &str, style: TextStyle) -> Size {
        self.cache.layout(&self.fonts, text, style, None).size
    }

    /// The state of `id` (its rectangle and animations as of last frame), if known.
    pub fn state(&self, id: Id) -> Option<&WidgetState> {
        self.states.get(&id)
    }

    /// Typed per-widget memory that lives as long as the widget is used (text-field editing state,
    /// a tree's expanded rows). Created with `T::default()` on first use.
    ///
    /// # Panics
    ///
    /// Never: the entry is keyed by the type and always holds a `T`.
    pub fn memory<T: Any + Default>(&mut self, id: Id) -> &mut T {
        let frame = self.frame;
        let entry = self.memory.entry((id, TypeId::of::<T>())).or_insert_with(|| (frame, Box::new(T::default())));
        entry.0 = frame;
        if !entry.1.is::<T>() {
            entry.1 = Box::new(T::default());
        }
        // INVARIANT: the entry holds a `T`: its key carries `T`'s type id, and it was replaced by a
        // `T` just above otherwise.
        #[allow(clippy::expect_used)]
        entry.1.downcast_mut::<T>().expect("memory entry has the requested type")
    }

    /// The keyboard focus.
    #[inline]
    pub fn focused(&self) -> Option<Id> {
        self.focused
    }

    /// Gives `id` the keyboard focus.
    pub fn request_focus(&mut self, id: Id) {
        self.focused = Some(id);
        self.focus_seen = true;
    }

    /// Clears the keyboard focus.
    pub fn clear_focus(&mut self) {
        self.focused = None;
    }

    /// Takes a key press of this frame matching `key` and `mods` exactly; `true` if there was one.
    pub fn consume_key(&mut self, key: Key, mods: Modifiers) -> bool {
        if let Some(entry) = self.keys.iter_mut().find(|k| k.0 == key && k.1 == mods && !k.2) {
            entry.2 = true;
            true
        } else {
            false
        }
    }

    /// Takes a key press of this frame matching `key` with any modifiers; returns them.
    pub fn consume_key_any(&mut self, key: Key) -> Option<Modifiers> {
        let entry = self.keys.iter_mut().find(|k| k.0 == key && !k.2)?;
        entry.2 = true;
        Some(entry.1)
    }

    /// The characters typed this frame (plus committed IME text).
    pub fn typed_text(&self) -> (&str, &str) {
        (&self.typed, &self.committed)
    }

    /// The IME composition string in progress.
    pub fn preedit(&self) -> &str {
        &self.preedit
    }

    /// Takes the pasted text of this frame.
    pub fn take_paste(&mut self) -> Option<String> {
        self.paste.take()
    }

    /// Puts text on the clipboard after this frame.
    pub fn copy(&mut self, text: &str) {
        self.output.copied = Some(text.to_owned());
    }

    /// Sets the pointer shape for this frame.
    pub fn set_cursor(&mut self, cursor: CursorIcon) {
        self.output.cursor = cursor;
    }

    /// Asks the shell for text input (IME) with the candidate window near `area`.
    pub fn want_text_input(&mut self, area: Rect) {
        self.output.text_input = true;
        self.output.ime_area = Some(area);
    }

    /// Keeps frames coming while an animation runs.
    pub fn request_repaint(&mut self) {
        self.animating = true;
    }

    /// The pointer position relative to `rect` as fractions (clamped to 0..1).
    pub fn pointer_fraction(&self, rect: Rect) -> Option<Point> {
        let p = self.pointer.pos?;
        let fx = if rect.w > 0.0 { ((p.x - rect.x) / rect.w).clamp(0.0, 1.0) } else { 0.0 };
        let fy = if rect.h > 0.0 { ((p.y - rect.y) / rect.h).clamp(0.0, 1.0) } else { 0.0 };
        Some(Point::new(fx, fy))
    }

    /// Registers `node` as the interactive widget `id` and reports what the input did to it.
    pub fn interact(&mut self, id: Id, node: NodeId, sense: Sense) -> Response {
        self.register(id, node, sense, false);
        let frame = self.frame;
        let state = *self.states.entry(id).or_default();
        let p = self.pointer;
        let over = self.hovered == Some(id);
        let mut r = Response { id, rect: state.rect, ..Response::default() };
        r.hovered = over && self.active.is_none_or(|a| a == id);
        if sense.focus {
            self.focus_order.push(id);
        }
        if (sense.click || sense.drag) && p.pressed_now[0] && over && self.active.is_none() {
            self.active = Some(id);
            r.pressed = true;
            if sense.focus {
                self.focused = Some(id);
            }
            r.double_clicked = p.click_count >= 2;
        }
        if self.active == Some(id) {
            r.held = p.down[0] || p.released_now[0];
            if sense.drag && p.dragging {
                r.dragged = true;
                r.drag_started = p.drag_started_now;
                r.drag_delta = p.delta;
            }
            if p.released_now[0] {
                r.clicked = sense.click && over && !(sense.drag && p.dragging);
            }
        }
        // While a popup holds the keyboard, only the widgets inside it get keys.
        let keys = !self.prev_grabbed || self.grab_depth > 0;
        r.focused = self.focused == Some(id) && keys;
        r.focus_visible = r.focused && self.focus_visible;
        if self.focused == Some(id) {
            self.focus_seen = true;
        }
        if over && p.pressed_now[1] && self.active.is_none() {
            r.secondary_clicked = true;
            if sense.focus {
                self.focused = Some(id);
            }
        }
        if r.focused
            && (self.consume_key(Key::F(10), Modifiers::SHIFT) || self.consume_key(CONTEXT_MENU_KEY, Modifiers::NONE))
        {
            r.secondary_clicked = true;
        }
        if r.focused && sense.click {
            let activate =
                self.consume_key(Key::Enter, Modifiers::NONE) || self.consume_key(Key::Space, Modifiers::NONE);
            if activate {
                r.clicked = true;
            }
        }
        // Animations.
        let m = self.theme.motion;
        let s = self.states.entry(id).or_default();
        s.seen = frame;
        s.hover_t = ease(s.hover_t, if r.hovered || r.held { 1.0 } else { 0.0 }, self.dt, m.ease_rate, m.reduced);
        s.press_t = ease(s.press_t, if r.held { 1.0 } else { 0.0 }, self.dt, m.ease_rate, m.reduced);
        s.focus_t = ease(s.focus_t, if r.focus_visible { 1.0 } else { 0.0 }, self.dt, m.ease_rate, m.reduced);
        r.hover_t = s.hover_t;
        r.press_t = s.press_t;
        let settled = |v: f32| v == 0.0 || v == 1.0;
        if !(settled(s.hover_t) && settled(s.press_t) && settled(s.focus_t)) {
            self.animating = true;
        }
        if r.hovered && (sense.click || sense.drag) {
            self.output.cursor = CursorIcon::Pointer;
        }
        r
    }

    fn register(&mut self, id: Id, node: NodeId, sense: Sense, scroll: bool) {
        let clip = self.clip_stack.last().copied().unwrap_or(NONE);
        self.widgets.push(FrameWidget { id, node, clip, layer: self.layer, sense, scroll });
    }

    /// Describes `id` for the semantic tree `[UI-09]`.
    pub fn semantic(&mut self, id: Id, role: Role, label: &str, value: SemanticValue) {
        let label = self.intern(label);
        let parent = self.semantic_stack.last().copied();
        let focused = self.focused == Some(id);
        self.semantics.push(Semantic { id, role, label, value, parent, rect: Rect::ZERO, focused, disabled: false });
    }

    /// Marks the last semantic entry disabled.
    pub fn semantic_disabled(&mut self) {
        if let Some(s) = self.semantics.last_mut() {
            s.disabled = true;
        }
    }

    /// Gives the last semantic entry the rectangle of `node` (for entries whose id is not an
    /// interactive widget: labels, progress bars, rows described under their own ids).
    pub fn semantic_node(&mut self, node: NodeId) {
        if let Some(i) = self.semantics.len().checked_sub(1) {
            self.semantic_nodes.push((i, node));
        }
    }

    /// Opens a semantic group: the following entries name `id` as their parent.
    pub fn begin_semantic_group(&mut self, id: Id) {
        self.semantic_stack.push(id);
    }

    /// Closes the last semantic group.
    pub fn end_semantic_group(&mut self) {
        self.semantic_stack.pop();
    }

    /// The semantic description of the last finished frame.
    pub fn semantics(&self) -> &[Semantic] {
        &self.semantics
    }

    /// Opens a vertical scroll area `[UI-04]`: its children scroll with the wheel and the overlay
    /// scroll bar (reference section 16.6). Close it with [`Ui::end_scroll`]. Returns the box.
    pub fn begin_scroll(&mut self, id: Id, style: Style) -> NodeId {
        let node = self.begin(Style { scroll: true, ..style });
        self.register(id, node, Sense::HOVER, true);
        self.states.entry(id).or_default().seen = self.frame;
        let reduced = self.theme.motion.reduced;
        let rate = self.theme.motion.scroll_rate;
        let dt = self.dt;
        let wheel = if self.wheel_target == Some(id) { self.pointer.wheel.y } else { 0.0 };
        let st = self.scrolls.entry(id).or_default();
        let max = (st.content - st.viewport).max(0.0);
        st.target = (st.target - wheel).clamp(0.0, max);
        st.offset = if st.dragging { st.target } else { ease(st.offset, st.target, dt, rate, reduced) };
        if (st.offset - st.target).abs() < 0.5 {
            st.offset = st.target;
        }
        let offset = st.offset.round();
        if st.offset != st.target {
            self.animating = true;
        }
        self.tree.set_scroll(node, Point::new(0.0, offset));
        self.push_clip(node, Area::FULL);
        self.semantic(
            id,
            Role::ScrollArea,
            "",
            SemanticValue::Number { value: f64::from(offset), min: 0.0, max: f64::from(max) },
        );
        node
    }

    /// Closes the scroll area opened by [`Ui::begin_scroll`] and draws its scroll bar.
    pub fn end_scroll(&mut self, id: Id) {
        self.pop_clip();
        let node = self.parent();
        self.end();
        let rect = self.states.get(&id).map_or(Rect::ZERO, |s| s.full_rect);
        let m = self.theme.metrics;
        let pal = self.theme.palette;
        let pointer = self.pointer;
        let hovered_zone = pointer.pos.is_some_and(|p| rect.contains(p) && p.x >= rect.right() - m.scrollbar_zone);
        let reduced = self.theme.motion.reduced;
        let rate = self.theme.motion.ease_rate;
        let dt = self.dt;
        let Some(st) = self.scrolls.get_mut(&id) else { return };
        if st.content <= st.viewport + 0.5 || st.viewport <= 0.0 {
            return;
        }
        // Thumb geometry from last frame's sizes.
        let track = (st.viewport - 16.0).max(1.0);
        let thumb = (track * st.viewport / st.content).max(m.scrollbar_min_thumb).min(track);
        let max = st.content - st.viewport;
        let t = if max > 0.0 { st.offset / max } else { 0.0 };
        let thumb_y = 8.0 + t * (track - thumb);
        // Dragging the thumb (or pressing beside it to jump there).
        if pointer.pressed_now[0] && hovered_zone {
            st.dragging = true;
            let py = pointer.pos.map_or(0.0, |p| p.y - rect.y);
            if py < thumb_y || py > thumb_y + thumb {
                st.target = ((py - 8.0 - thumb * 0.5) / (track - thumb).max(1.0) * max).clamp(0.0, max);
            }
            st.drag_from = st.target;
            self.active = Some(id);
        }
        if st.dragging {
            if pointer.down[0] {
                let travel = (track - thumb).max(1.0);
                st.target = (st.target + pointer.delta.y / travel * max).clamp(0.0, max);
                st.offset = st.target;
            } else {
                st.dragging = false;
                if self.active == Some(id) {
                    self.active = None;
                }
            }
        }
        let goal = if hovered_zone || st.dragging { 1.0 } else { 0.0 };
        st.thumb_t = ease(st.thumb_t, goal, dt, rate, reduced);
        let thickness = m.scrollbar_thin + (m.scrollbar_thick - m.scrollbar_thin) * st.thumb_t;
        let color = pal.text_faint.fade(0.35 + 0.4 * st.thumb_t);
        let area = Area {
            x0: Anchor::end(-m.scrollbar_inset - thickness),
            y0: Anchor::start(thumb_y),
            x1: Anchor::end(-m.scrollbar_inset),
            y1: Anchor::start(thumb_y + thumb),
        };
        self.paint(
            node,
            Paint::Rect {
                area,
                fill: color,
                radius: Corners::all(thickness * 0.5),
                border: 0.0,
                border_color: Color::TRANSPARENT,
            },
        );
    }

    /// The scroll offset and viewport height of a scroll area as of last frame (for virtualised
    /// lists: [`crate::layout::visible_rows`]).
    pub fn scroll_view(&self, id: Id) -> (f32, f32) {
        self.scrolls.get(&id).map_or((0.0, 0.0), |s| (s.offset.round(), s.viewport))
    }

    /// Scrolls a scroll area so that `y` (content pixels) is visible.
    pub fn scroll_to(&mut self, id: Id, y: f32) {
        if let Some(s) = self.scrolls.get_mut(&id) {
            let max = (s.content - s.viewport).max(0.0);
            s.target = y.clamp(0.0, max);
        }
    }

    /// Finishes the frame: computes the layout, updates the arena, replays the paint operations
    /// into the draw list and returns what the shell must do.
    pub fn end_frame(&mut self) -> &FrameOutput {
        let scale = self.scale;
        self.tree.compute(self.viewport, scale);

        // Clip rectangles of this frame.
        let resolve_clip = |clips: &[Clip], tree: &LayoutTree, mut index: u32, viewport: Rect| -> Rect {
            let mut rect = viewport;
            let mut guard = 0;
            while index != NONE && guard < 4096 {
                let Some(c) = clips.get(index as usize) else { break };
                rect = rect.intersect(c.area.resolve(tree.rect(c.node)));
                index = c.parent;
                guard += 1;
            }
            rect
        };

        // Widget rectangles, the hit list for next frame, scroll sizes.
        self.hits.clear();
        for w in &self.widgets {
            let full = self.tree.rect(w.node);
            let clip = resolve_clip(&self.clips, &self.tree, w.clip, self.viewport);
            let rect = full.intersect(clip);
            if let Some(s) = self.states.get_mut(&w.id) {
                s.rect = rect;
                s.full_rect = full;
            }
            if w.scroll
                && let Some(st) = self.scrolls.get_mut(&w.id)
            {
                st.content = self.tree.content_size(w.node).h;
                st.viewport = full.h;
                let max = (st.content - st.viewport).max(0.0);
                st.target = st.target.min(max);
                st.offset = st.offset.min(max);
            }
            if w.sense != Sense::HOVER || w.scroll {
                self.hits.push(Hit { id: w.id, rect, layer: w.layer, sense: w.sense, scroll: w.scroll });
            }
        }
        for s in &mut self.semantics {
            if let Some(st) = self.states.get(&s.id) {
                s.rect = st.rect;
            }
            s.focused = self.focused == Some(s.id);
        }
        for &(i, node) in &self.semantic_nodes {
            if let Some(s) = self.semantics.get_mut(i)
                && s.rect == Rect::ZERO
            {
                s.rect = self.tree.rect(node);
            }
        }
        // A focused widget that was not built this frame loses the focus.
        if self.focused.is_some() && !self.focus_seen {
            self.focused = None;
        }
        // Unconsumed Escape clears the focus.
        if self.consume_key(Key::Escape, Modifiers::NONE) {
            self.focused = None;
        }
        if (self.pointer.released_now[0] || !self.pointer.down[0])
            && self.active.is_some_and(|a| !self.scrolls.get(&a).is_some_and(|s| s.dragging))
        {
            self.active = None;
        }

        // Paint.
        let (w, h) = (self.viewport.w * scale, self.viewport.h * scale);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        // Viewport sizes are small positive numbers.
        self.list.begin(w.round().max(0.0) as u32, h.round().max(0.0) as u32, scale);
        let theme = self.theme;
        for op in &self.ops {
            let rect = self.tree.rect(op.node);
            match op.paint {
                Paint::Rect { area, fill, radius, border, border_color } => {
                    self.list.bordered(area.resolve(rect), fill, radius, border, border_color);
                }
                Paint::Shadow { area, color, radius, blur } => {
                    self.list.shadow(area.resolve(rect), color, radius, blur);
                }
                Paint::Gradient { area, from, to, radius, axis } => {
                    self.list.gradient(area.resolve(rect), from, to, radius, axis);
                }
                Paint::Circle { cx, cy, radius, color } => {
                    let c = Point::new(cx.resolve(rect.x, rect.w), cy.resolve(rect.y, rect.h));
                    self.list.circle(c, radius, color);
                }
                Paint::Line { ax, ay, bx, by, width, color } => {
                    let a = Point::new(ax.resolve(rect.x, rect.w), ay.resolve(rect.y, rect.h));
                    let b = Point::new(bx.resolve(rect.x, rect.w), by.resolve(rect.y, rect.h));
                    self.list.line(a, b, width, color);
                }
                Paint::Text { area, text, style, color, align, wrap } => {
                    let r = area.resolve(rect);
                    let s = self.text_arena.get(text.start as usize..text.end as usize).unwrap_or("");
                    let layout = self.cache.layout(&self.fonts, s, style, wrap.then_some(r.w));
                    let y = if wrap { r.y } else { r.y + ((r.h - layout.size.h) * 0.5) };
                    draw_layout(&mut self.list, &self.fonts, layout, Point::new(r.x, y), align, r.w, color);
                }
                Paint::Image { area, texture, uv, tint, radius } => {
                    self.list.image(area.resolve(rect), texture, uv, tint, radius);
                }
                Paint::FocusRing { area, radius } => {
                    let m = theme.metrics;
                    self.list.focus_ring(
                        area.resolve(rect),
                        radius,
                        m.focus_ring_width,
                        m.focus_ring_offset,
                        theme.palette.focus_ring,
                    );
                }
                Paint::PushClip { area } => self.list.push_clip(area.resolve(rect)),
                Paint::PushOverlayClip { area } => self.list.push_clip_replace(area.resolve(rect)),
                Paint::PopClip => self.list.pop_clip(),
                Paint::Layer(layer) => self.list.set_layer(layer),
            }
        }
        self.output.paint = self.list.finish();
        self.output.animating = self.animating;
        self.output.dirty = self.dirty_region();

        // Forget ids not seen for a while.
        let frame = self.frame;
        if frame.is_multiple_of(64) {
            self.gc_scratch.clear();
            self.gc_scratch.extend(
                self.states
                    .iter()
                    .filter(|(_, s)| frame.saturating_sub(s.seen) > Self::FORGET_AFTER)
                    .map(|(id, _)| *id),
            );
            for id in &self.gc_scratch {
                self.states.remove(id);
                self.scrolls.remove(id);
            }
            self.memory.retain(|_, (seen, _)| frame.saturating_sub(*seen) <= Self::FORGET_AFTER);
        }
        &self.output
    }
}

impl Ui {
    /// Compares this frame's instances with the previous frame's and keeps a copy `[UI-01]`.
    fn dirty_region(&mut self) -> Dirty {
        let now = self.list.instances();
        let vp = self.list.viewport();
        let target = (vp.w.to_bits(), vp.h.to_bits(), self.scale.to_bits());
        let dirty = if target != self.prev_target {
            Dirty::Full
        } else if self.background.is_none() {
            if now == self.prev_instances.as_slice() { Dirty::Clean } else { Dirty::Full }
        } else {
            let mut region: Option<Rect> = None;
            let n = now.len().max(self.prev_instances.len());
            for i in 0..n {
                let (a, b) = (now.get(i), self.prev_instances.get(i));
                if a == b {
                    continue;
                }
                for inst in [a, b].into_iter().flatten() {
                    let r = instance_bounds(inst);
                    region = Some(region.map_or(r, |u| u.union(r)));
                }
            }
            match region {
                None => Dirty::Clean,
                Some(r) => {
                    let r = r.intersect(Rect::new(0.0, 0.0, vp.w * self.scale, vp.h * self.scale));
                    let (x0, y0) = (r.x.floor(), r.y.floor());
                    let (x1, y1) = (r.right().ceil(), r.bottom().ceil());
                    let r = Rect::new(x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0));
                    if r.w <= 0.0 || r.h <= 0.0 { Dirty::Clean } else { Dirty::Region(r) }
                }
            }
        };
        self.prev_target = target;
        self.prev_instances.clear();
        self.prev_instances.extend_from_slice(now);
        dirty
    }
}

/// The physical-pixel rectangle an instance can touch: its shape grown by its blur or half width
/// and one pixel of anti-aliasing, cut to its clip.
fn instance_bounds(i: &crate::paint::Instance) -> Rect {
    let [x0, y0, x1, y1] = i.rect;
    let kind = i.params[2];
    let spread = if kind == crate::paint::Kind::Shadow.code() || kind == crate::paint::Kind::Segment.code() {
        i.params[0].abs()
    } else {
        0.0
    };
    let grow = spread + 1.0;
    let (ax, bx) = (x0.min(x1) - grow, x0.max(x1) + grow);
    let (ay, by) = (y0.min(y1) - grow, y0.max(y1) + grow);
    let shape = Rect::new(ax, ay, bx - ax, by - ay);
    let [c0, c1, c2, c3] = i.clip;
    shape.intersect(Rect::new(c0, c1, (c2 - c0).max(0.0), (c3 - c1).max(0.0)))
}

/// The index of a pointer button.
fn button_index(b: PointerButton) -> usize {
    match b {
        PointerButton::Primary => 0,
        PointerButton::Secondary => 1,
        PointerButton::Middle => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::fonts;

    fn ui() -> Ui {
        Ui::new(fonts().clone(), Theme::dark())
    }

    /// Runs one frame with `events`, building `build`.
    fn frame(ui: &mut Ui, time: f64, events: Vec<Event>, mut build: impl FnMut(&mut Ui)) {
        let mut q = InputQueue { events, time };
        ui.begin_frame(&mut q, Size::new(400.0, 300.0), 1.0);
        build(ui);
        ui.end_frame();
    }

    fn click_at(p: Point) -> Vec<Event> {
        vec![
            Event::PointerMoved(p),
            Event::Button { button: PointerButton::Primary, pressed: true, pos: p },
            Event::Button { button: PointerButton::Primary, pressed: false, pos: p },
        ]
    }

    /// Two fixed-size boxes with click sense; returns their responses.
    fn two_buttons(ui: &mut Ui) -> [Response; 2] {
        let row = ui.begin(Style { gap: 10.0, ..Style::row() });
        let _ = row;
        let mut out = [Response::default(); 2];
        for (i, slot) in out.iter_mut().enumerate() {
            let id = ui.id(if i == 0 { "a" } else { "b" });
            let node = ui.node(Style::leaf(Size::new(50.0, 20.0)));
            *slot = ui.interact(id, node, Sense::CLICK);
            ui.semantic(id, Role::Button, if i == 0 { "A" } else { "B" }, SemanticValue::None);
        }
        ui.end();
        out
    }

    #[test]
    fn ids_are_stable_and_scoped() {
        let a = Id::ROOT.with("panel").with("ok");
        assert_eq!(a, Id::ROOT.with("panel").with("ok"));
        assert_ne!(Id::ROOT.with("ab").with("c"), Id::ROOT.with("a").with("bc"));
        assert_ne!(Id::ROOT.with_index(1), Id::ROOT.with_index(2));
        let mut ui = ui();
        frame(&mut ui, 0.0, vec![], |ui| {
            let outer = ui.id("x");
            ui.push_id(Id::ROOT.with("scope"));
            assert_ne!(ui.id("x"), outer);
            ui.pop_id();
            assert_eq!(ui.id("x"), outer);
        });
    }

    #[test]
    fn ui_06_click_uses_last_frames_rectangles() {
        let mut ui = ui();
        let mut r = [Response::default(); 2];
        frame(&mut ui, 0.0, vec![], |ui| r = two_buttons(ui));
        assert!(!r[0].clicked && r[0].rect == Rect::ZERO, "nothing is known on the first frame");
        frame(&mut ui, 0.1, vec![], |ui| r = two_buttons(ui));
        assert_eq!(r[1].rect, Rect::new(60.0, 0.0, 50.0, 20.0));
        frame(&mut ui, 0.2, click_at(Point::new(70.0, 10.0)), |ui| r = two_buttons(ui));
        assert!(r[1].clicked && r[1].pressed && r[1].focused && !r[1].focus_visible);
        assert!(!r[0].clicked && !r[0].hovered);
        assert_eq!(ui.focused(), Some(r[1].id));
        // Press inside, release outside: no click.
        let p = Point::new(10.0, 10.0);
        let down =
            vec![Event::PointerMoved(p), Event::Button { button: PointerButton::Primary, pressed: true, pos: p }];
        frame(&mut ui, 1.2, down, |ui| r = two_buttons(ui));
        assert!(r[0].pressed && r[0].held);
        let q = Point::new(300.0, 200.0);
        let up = vec![Event::PointerMoved(q), Event::Button { button: PointerButton::Primary, pressed: false, pos: q }];
        frame(&mut ui, 1.3, up, |ui| r = two_buttons(ui));
        assert!(!r[0].clicked);
    }

    #[test]
    fn ui_09_tab_navigation_shows_focus_and_enter_activates() {
        let mut ui = ui();
        let mut r = [Response::default(); 2];
        frame(&mut ui, 0.0, vec![], |ui| r = two_buttons(ui));
        let tab = |shift: bool| {
            vec![
                Event::Modifiers(Modifiers { shift, ..Modifiers::NONE }),
                Event::Key { key: Key::Tab, pressed: true, repeat: false },
            ]
        };
        frame(&mut ui, 0.1, tab(false), |ui| r = two_buttons(ui));
        assert!(r[0].focused && r[0].focus_visible);
        frame(&mut ui, 0.2, tab(false), |ui| r = two_buttons(ui));
        assert!(r[1].focused);
        frame(&mut ui, 0.3, tab(false), |ui| r = two_buttons(ui));
        assert!(r[0].focused, "wraps around");
        frame(&mut ui, 0.4, tab(true), |ui| r = two_buttons(ui));
        assert!(r[1].focused, "shift+tab goes back");
        let enter =
            vec![Event::Modifiers(Modifiers::NONE), Event::Key { key: Key::Enter, pressed: true, repeat: false }];
        frame(&mut ui, 0.5, enter, |ui| r = two_buttons(ui));
        assert!(r[1].clicked, "Enter activates the focused button");
        let esc = vec![Event::Key { key: Key::Escape, pressed: true, repeat: false }];
        frame(&mut ui, 0.6, esc, |ui| r = two_buttons(ui));
        assert_eq!(ui.focused(), None);
        // A click elsewhere hides the focus ring.
        frame(&mut ui, 0.7, tab(false), |ui| r = two_buttons(ui));
        frame(&mut ui, 0.8, click_at(Point::new(10.0, 10.0)), |ui| r = two_buttons(ui));
        assert!(r[0].focused && !r[0].focus_visible);
    }

    #[test]
    fn ui_09_semantics_list_roles_labels_and_rects() {
        let mut ui = ui();
        frame(&mut ui, 0.0, vec![], |ui| {
            two_buttons(ui);
        });
        frame(&mut ui, 0.1, vec![], |ui| {
            two_buttons(ui);
        });
        let s = ui.semantics();
        assert_eq!(s.len(), 2);
        assert_eq!((s[0].role, ui.text(s[0].label)), (Role::Button, "A"));
        assert_eq!(s[1].rect, Rect::new(60.0, 0.0, 50.0, 20.0));
    }

    #[test]
    fn topmost_layer_wins_the_hover() {
        let mut ui = ui();
        let mut base = Response::default();
        let mut popup = Response::default();
        let build = |ui: &mut Ui, base: &mut Response, popup: &mut Response| {
            let n = ui.node(Style::leaf(Size::new(100.0, 100.0)));
            let id = ui.id("base");
            *base = ui.interact(id, n, Sense::CLICK);
            ui.set_layer(Layer::Popup);
            let p = ui.node(Style {
                position: crate::layout::Position::Absolute {
                    left: Some(20.0),
                    top: Some(20.0),
                    right: None,
                    bottom: None,
                },
                width: crate::layout::Dim::Px(50.0),
                height: crate::layout::Dim::Px(50.0),
                ..Style::default()
            });
            let id = ui.id("popup");
            *popup = ui.interact(id, p, Sense::CLICK);
            ui.set_layer(Layer::Base);
        };
        frame(&mut ui, 0.0, vec![], |ui| build(ui, &mut base, &mut popup));
        frame(&mut ui, 0.1, vec![Event::PointerMoved(Point::new(30.0, 30.0))], |ui| build(ui, &mut base, &mut popup));
        assert!(popup.hovered && !base.hovered);
        frame(&mut ui, 0.2, vec![Event::PointerMoved(Point::new(90.0, 90.0))], |ui| build(ui, &mut base, &mut popup));
        assert!(base.hovered && !popup.hovered);
    }

    #[test]
    fn ui_04_scroll_area_wheel_clip_and_virtual_rows() {
        let mut ui = ui();
        let mut visible = 0..0;
        let build = |ui: &mut Ui, visible: &mut std::ops::Range<usize>| {
            let id = ui.id("list");
            let (offset, view) = ui.scroll_view(id);
            let node = ui.begin_scroll(id, Style { height: crate::layout::Dim::Px(100.0), ..Style::column() });
            let _ = node;
            // A virtualised list of 10 000 rows of 20 px: spacer, visible rows, spacer.
            let rows = crate::layout::visible_rows(offset, view, 20.0, 10_000, 1);
            #[allow(clippy::cast_precision_loss)]
            let (before, after) = (rows.start as f32 * 20.0, (10_000 - rows.end) as f32 * 20.0);
            ui.node(Style { height: crate::layout::Dim::Px(before), ..Style::default() });
            for i in rows.clone() {
                let n = ui.node(Style { height: crate::layout::Dim::Px(20.0), ..Style::default() });
                let row_id = id.with_index(i as u64);
                ui.interact(row_id, n, Sense::CLICK);
            }
            ui.node(Style { height: crate::layout::Dim::Px(after), ..Style::default() });
            ui.end_scroll(id);
            *visible = rows;
        };
        frame(&mut ui, 0.0, vec![], |ui| build(ui, &mut visible));
        frame(&mut ui, 0.1, vec![], |ui| build(ui, &mut visible));
        assert_eq!(visible, 0..6);
        let wheel = vec![
            Event::PointerMoved(Point::new(50.0, 50.0)),
            Event::Wheel { delta: Point::new(0.0, -10.0), precise: false },
        ];
        let mut t = 0.2;
        frame(&mut ui, t, wheel, |ui| build(ui, &mut visible));
        for _ in 0..60 {
            t += 0.016;
            frame(&mut ui, t, vec![], |ui| build(ui, &mut visible));
        }
        let (offset, view) = ui.scroll_view(ui.id("list"));
        assert_eq!(view, 100.0);
        assert!((offset - 480.0).abs() < 1.0, "{offset}");
        assert_eq!(visible, 23..30);
        // Rows outside the clip are not hit: the area's clip bounds every row's rectangle.
        let list = ui.id("list");
        let row = ui.state(list.with_index(23)).copied().unwrap_or_default();
        assert!(row.rect.y >= 0.0 && row.rect.bottom() <= 100.0, "{:?}", row.rect);
        assert!(ui.draw_list().stats().instances >= 1, "the scroll bar is drawn");
    }

    #[test]
    fn perf_r1_steady_frames_do_not_allocate() {
        // [UI-02] [PERF-R1] 600 widgets with labels, rebuilt every frame: no allocation once warm.
        let mut ui = ui();
        let build = |ui: &mut Ui| {
            let col = ui.begin(Style { wrap: true, gap: 2.0, ..Style::row() });
            let _ = col;
            for i in 0..600_u64 {
                let id = Id::ROOT.with_index(i);
                let n = ui.node(Style::leaf(Size::new(60.0, 18.0)));
                let r = ui.interact(id, n, Sense::CLICK);
                let fill = ui.theme().palette.surface.mix(ui.theme().palette.hover, r.hover_t);
                ui.paint(
                    n,
                    Paint::Rect {
                        area: Area::FULL,
                        fill,
                        radius: Corners::all(4.0),
                        border: 1.0,
                        border_color: Color::BLACK,
                    },
                );
                let text = ui.intern("Label");
                let style = ui.theme().text.small;
                ui.paint(
                    n,
                    Paint::Text {
                        area: Area::FULL,
                        text,
                        style,
                        color: Color::WHITE,
                        align: Align::Center,
                        wrap: false,
                    },
                );
                ui.semantic(id, Role::Button, "Label", SemanticValue::None);
            }
            ui.end();
        };
        let mut t = 0.0;
        for _ in 0..3 {
            t += 0.016;
            frame(&mut ui, t, vec![], build);
        }
        let stats = humanart_core::alloc::count_in(|| {
            t += 0.016;
            let mut q = InputQueue { events: Vec::new(), time: t };
            ui.begin_frame(&mut q, Size::new(400.0, 300.0), 1.0);
            build(&mut ui);
            ui.end_frame();
        });
        assert_eq!(stats.events(), 0);
        assert!(ui.draw_list().stats().instances > 600);
        assert_eq!(ui.draw_list().stats().batches, 1);
        // [UI-02] CPU cost of build + layout + record, median of 50 frames (reported, not asserted:
        // it depends on the machine and the build profile).
        let mut times: Vec<f64> = (0..50)
            .map(|_| {
                t += 0.016;
                let start = std::time::Instant::now();
                let mut q = InputQueue { events: Vec::new(), time: t };
                ui.begin_frame(&mut q, Size::new(400.0, 300.0), 1.0);
                build(&mut ui);
                ui.end_frame();
                start.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        times.sort_by(f64::total_cmp);
        eprintln!("UI build+layout+record, 600 widgets with text: median {:.3} ms", times[25]);
    }

    /// Two boxes whose fill follows their hover amount (so hovering changes pixels).
    fn two_swatches(ui: &mut Ui) {
        let p = ui.theme().palette;
        let row = ui.begin(Style { gap: 10.0, ..Style::row() });
        let _ = row;
        for (i, label) in ["a", "b"].into_iter().enumerate() {
            let id = ui.id(label);
            let node = ui.node(Style::leaf(Size::new(50.0, 20.0)));
            let r = ui.interact(id, node, Sense::CLICK);
            let fill = p.surface.mix(p.accent, r.hover_t);
            ui.paint(
                node,
                Paint::Rect { area: Area::FULL, fill, radius: Corners::all(4.0), border: 0.0, border_color: fill },
            );
            ui.semantic(id, Role::Button, if i == 0 { "A" } else { "B" }, SemanticValue::None);
        }
        ui.end();
    }

    #[test]
    fn ui_01_dirty_regions_cover_what_changed_and_nothing_else() {
        let mut ui = ui();
        ui.set_background(Some(ui.theme().palette.bg));
        let mut t = 0.0;
        let mut run = |ui: &mut Ui, events: Vec<Event>, size: Size| -> Dirty {
            t += 0.016;
            let mut q = InputQueue { events, time: t };
            ui.begin_frame(&mut q, size, 1.0);
            two_swatches(ui);
            ui.end_frame().dirty
        };
        let size = Size::new(400.0, 300.0);
        assert_eq!(run(&mut ui, vec![], size), Dirty::Full, "the first frame");
        assert_eq!(run(&mut ui, vec![], size), Dirty::Clean, "nothing changed");
        assert_eq!(run(&mut ui, vec![Event::PointerMoved(Point::new(300.0, 200.0))], size), Dirty::Clean);
        // Hovering B changes only B's pixels while its hover eases in.
        let dirty = run(&mut ui, vec![Event::PointerMoved(Point::new(70.0, 10.0))], size);
        let Dirty::Region(r) = dirty else { panic!("{dirty:?}") };
        assert!(r.x >= 55.0 && r.x <= 60.0 && r.right() >= 110.0 && r.right() <= 115.0, "{r:?}");
        assert!(r.y <= 0.0 && r.bottom() >= 20.0 && r.bottom() <= 25.0, "{r:?}");
        let mut settled = false;
        for _ in 0..200 {
            if run(&mut ui, vec![], size) == Dirty::Clean {
                settled = true;
                break;
            }
        }
        assert!(settled, "the hover settles and frames become clean");
        // A resize redraws everything.
        assert_eq!(run(&mut ui, vec![], Size::new(410.0, 300.0)), Dirty::Full);
        // Without an opaque background every change is a full redraw.
        let mut plain = Ui::new(fonts().clone(), Theme::dark());
        let size = Size::new(400.0, 300.0);
        run(&mut plain, vec![], size);
        run(&mut plain, vec![], size);
        assert_eq!(run(&mut plain, vec![Event::PointerMoved(Point::new(70.0, 10.0))], size), Dirty::Full);
    }

    #[test]
    fn ui_01_region_redraw_equals_a_full_redraw() {
        // [UI-01] Drawing only the dirty region over the previous image gives exactly the image a
        // full redraw gives (GPU, this PC's adapter).
        use crate::tests::{draw_and_read, gpu_lock, renderer, target};
        let _lock = gpu_lock();
        let mut ui = ui();
        ui.set_background(Some(ui.theme().palette.bg));
        let size = Size::new(200.0, 60.0);
        let mut t = 0.0;
        let mut run = |ui: &mut Ui, events: Vec<Event>| -> Dirty {
            t += 0.016;
            let mut q = InputQueue { events, time: t };
            ui.begin_frame(&mut q, size, 1.5);
            two_swatches(ui);
            ui.end_frame().dirty
        };
        run(&mut ui, vec![]);
        run(&mut ui, vec![]);
        let mut r = renderer(fonts().atlas());
        let incremental = target(300, 90);
        let full = target(300, 90);
        draw_and_read(&mut r, ui.draw_list(), &incremental, &mut |r, e, v| {
            r.record(e, v, Some(Color::hex(0xFF00FF)), ui.draw_list());
        });
        let dirty = run(&mut ui, vec![Event::PointerMoved(Point::new(70.0, 10.0))]);
        let Dirty::Region(region) = dirty else { panic!("{dirty:?}") };
        let mut calls = 0;
        let a = draw_and_read(&mut r, ui.draw_list(), &incremental, &mut |r, e, v| {
            calls = r.record_region(e, v, ui.draw_list(), region);
        });
        assert_eq!(calls, 1);
        let b = draw_and_read(&mut r, ui.draw_list(), &full, &mut |r, e, v| {
            r.record(e, v, Some(Color::hex(0xFF00FF)), ui.draw_list());
        });
        let differing = a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0).filter(|(x, y)| x != y).count();
        assert_eq!(differing, 0, "region {region:?}");
        assert!(region.w * region.h < 300.0 * 90.0 * 0.5, "the region is a small part: {region:?}");
    }

    #[test]
    fn ui_06_pen_pressure_and_tilt_reach_widgets_until_the_pen_lifts() {
        let mut ui = ui();
        let p = Point::new(10.0, 10.0);
        let pen = vec![
            Event::PointerMoved(p),
            Event::Pen { pressure: 1.7, tilt: [0.3, f32::NAN] },
            Event::Button { button: PointerButton::Primary, pressed: true, pos: p },
        ];
        frame(&mut ui, 0.0, pen, |_| {});
        assert_eq!(ui.pen(), Some(PenState { pressure: 1.0, tilt: [0.3, 0.0] }), "clamped and cleaned");
        frame(&mut ui, 0.1, vec![Event::Pen { pressure: 0.25, tilt: [-9.0, 0.1] }], |ui| {
            assert_eq!(ui.pen().map(|s| s.pressure), Some(0.25));
            assert_eq!(ui.pen().map(|s| s.tilt[0]), Some(-std::f32::consts::FRAC_PI_2));
        });
        frame(&mut ui, 0.2, vec![Event::Button { button: PointerButton::Primary, pressed: false, pos: p }], |_| {});
        assert_eq!(ui.pen(), None, "a mouse after the pen lifts has no pressure");
    }

    #[test]
    fn ui_06_keymap_shortcuts_dispatch_after_widgets() {
        use crate::input::{Keymap, Shortcut};
        let mut keymap = Keymap::new();
        keymap.bind("view.frame", Shortcut::new(Key::Letter('F'), Modifiers::NONE));
        keymap.bind("file.save", Shortcut::new(Key::Letter('S'), Modifiers::CTRL));
        let mut ui = ui();
        let mut text = String::new();
        let mut fired = Vec::new();
        let step = |ui: &mut Ui, t: f64, events: Vec<Event>, text: &mut String| {
            let mut q = InputQueue { events, time: t };
            ui.begin_frame(&mut q, Size::new(300.0, 100.0), 1.0);
            crate::widgets::TextField::new("name", text).show(ui);
            let action = ui.shortcut(&keymap);
            ui.end_frame();
            action
        };
        let key = |k: Key, m: Modifiers| vec![Event::Modifiers(m), Event::Key { key: k, pressed: true, repeat: false }];
        step(&mut ui, 0.0, vec![], &mut text);
        step(&mut ui, 0.1, vec![], &mut text);
        fired.push(step(&mut ui, 0.2, key(Key::Letter('F'), Modifiers::NONE), &mut text));
        // With the text field focused, a plain letter is text; Ctrl+S still reaches the keymap.
        step(&mut ui, 0.3, click_at(Point::new(50.0, 15.0)), &mut text);
        let mut typing = key(Key::Letter('F'), Modifiers::NONE);
        typing.push(Event::Text('f'));
        fired.push(step(&mut ui, 0.4, typing, &mut text));
        fired.push(step(&mut ui, 0.5, key(Key::Letter('S'), Modifiers::CTRL), &mut text));
        assert_eq!(fired, [Some("view.frame"), None, Some("file.save")]);
        assert_eq!(text, "f");
    }

    #[test]
    fn ui_09_open_popup_holds_the_keyboard() {
        let mut ui = ui();
        let mut log = Vec::new();
        let step = |ui: &mut Ui, t: f64, events: Vec<Event>, open: bool, log: &mut Vec<&'static str>| {
            let mut q = InputQueue { events, time: t };
            ui.begin_frame(&mut q, Size::new(300.0, 100.0), 1.0);
            let r = two_buttons(ui);
            if r[0].clicked {
                log.push("a clicked");
            }
            if open {
                ui.begin_key_grab();
                if ui.consume_key(Key::Enter, Modifiers::NONE) {
                    log.push("popup took Enter");
                }
                if ui.consume_key_any(Key::Tab).is_some() {
                    log.push("popup took Tab");
                }
                ui.end_key_grab(true);
            }
            ui.end_frame();
        };
        let key = |k: Key| vec![Event::Key { key: k, pressed: true, repeat: false }];
        step(&mut ui, 0.0, vec![], false, &mut log);
        step(&mut ui, 0.1, key(Key::Tab), false, &mut log);
        step(&mut ui, 0.2, key(Key::Enter), false, &mut log);
        step(&mut ui, 0.3, vec![], true, &mut log);
        step(&mut ui, 0.4, key(Key::Enter), true, &mut log);
        step(&mut ui, 0.5, key(Key::Tab), true, &mut log);
        assert_eq!(log, ["a clicked", "popup took Enter", "popup took Tab"]);
        assert_eq!(ui.focused(), Some(Id::ROOT.with("a")), "Tab did not move the focus under the popup");
    }

    #[test]
    fn ui_09_scale_is_clamped_to_75_to_200_percent() {
        assert_eq!(clamp_ui_scale(0.5), 0.75);
        assert_eq!(clamp_ui_scale(1.25), 1.25);
        assert_eq!(clamp_ui_scale(3.0), 2.0);
        assert_eq!(clamp_ui_scale(f32::NAN), 1.0);
    }

    #[test]
    fn ui_06_touch_taps_drags_long_presses_and_pinches() {
        use crate::input::TouchPhase;
        let touch = |id, phase, x: f32, y: f32| Event::Touch { id, phase, pos: Point::new(x, y) };
        let mut ui = ui();
        let mut r = [Response::default(); 2];
        frame(&mut ui, 0.0, vec![], |ui| r = two_buttons(ui));
        frame(&mut ui, 0.05, vec![], |ui| r = two_buttons(ui));
        // A tap is a click.
        frame(&mut ui, 0.1, vec![touch(1, TouchPhase::Start, 70.0, 10.0)], |ui| r = two_buttons(ui));
        assert!(r[1].pressed && !r[1].clicked);
        frame(&mut ui, 0.15, vec![touch(1, TouchPhase::End, 70.0, 10.0)], |ui| r = two_buttons(ui));
        assert!(r[1].clicked);
        // A cancelled touch is not.
        frame(&mut ui, 0.6, vec![touch(2, TouchPhase::Start, 10.0, 10.0)], |ui| r = two_buttons(ui));
        frame(&mut ui, 0.65, vec![touch(2, TouchPhase::Cancel, 10.0, 10.0)], |ui| r = two_buttons(ui));
        assert!(!r[0].clicked);
        // Holding still is a secondary click, and lifting then does not click.
        frame(&mut ui, 1.5, vec![touch(3, TouchPhase::Start, 10.0, 10.0)], |ui| r = two_buttons(ui));
        let mut q = InputQueue { events: Vec::new(), time: 1.8 };
        ui.begin_frame(&mut q, Size::new(400.0, 300.0), 1.0);
        r = two_buttons(&mut ui);
        let waiting = ui.end_frame().animating;
        assert!(!r[0].secondary_clicked && waiting, "frames keep coming while the long press is pending");
        frame(&mut ui, 2.05, vec![], |ui| r = two_buttons(ui));
        assert!(r[0].secondary_clicked, "long press");
        frame(&mut ui, 2.1, vec![touch(3, TouchPhase::End, 10.0, 10.0)], |ui| r = two_buttons(ui));
        assert!(!r[0].clicked);
        // Two fingers: no click, a pan and a pinch.
        frame(&mut ui, 3.0, vec![touch(4, TouchPhase::Start, 100.0, 100.0)], |ui| r = two_buttons(ui));
        frame(&mut ui, 3.05, vec![touch(5, TouchPhase::Start, 200.0, 100.0)], |ui| r = two_buttons(ui));
        let mut g = Gesture::default();
        frame(
            &mut ui,
            3.1,
            vec![touch(4, TouchPhase::Move, 80.0, 110.0), touch(5, TouchPhase::Move, 240.0, 110.0)],
            |ui| {
                r = two_buttons(ui);
                g = ui.gesture();
            },
        );
        assert!((g.zoom - 1.6).abs() < 1e-3, "{g:?}");
        assert!((g.pan.x - 10.0).abs() < 1e-3 && (g.pan.y - 10.0).abs() < 1e-3, "{g:?}");
        assert_eq!(ui.wheel(), g.pan, "two-finger pans scroll");
        frame(&mut ui, 3.15, vec![touch(4, TouchPhase::End, 80.0, 110.0)], |ui| r = two_buttons(ui));
        frame(&mut ui, 3.2, vec![touch(5, TouchPhase::End, 240.0, 110.0)], |ui| r = two_buttons(ui));
        assert!(!r[0].clicked && !r[1].clicked);
        assert_eq!(ui.gesture(), Gesture::default());
        // Not-finite positions are ignored.
        frame(&mut ui, 4.0, vec![touch(6, TouchPhase::Start, f32::NAN, 1.0)], |ui| r = two_buttons(ui));
        assert!(!ui.pointer_down());
    }
}
