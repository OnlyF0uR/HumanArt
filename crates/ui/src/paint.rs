//! The draw list: everything the UI draws in a frame, as instanced quads `[UI-02]`.
//!
//! Widgets call the primitives of [`DrawList`] in logical pixels: rounded rectangles with optional
//! borders ([`DrawList::rect`], [`DrawList::bordered`]), soft shadows ([`DrawList::shadow`]),
//! gradients, line segments, images and glyphs (the latter through [`crate::text`]). Each call
//! appends one [`Instance`], converted to physical pixels and pixel-snapped at record time. The GPU
//! side ([`crate::render`]) evaluates the shape as a signed distance field in `shaders/ui/quad.wgsl`.
//!
//! **Batching.** The clip rectangle is stored per instance and tested in the shader, so a clip
//! change never breaks a batch; only a change of image texture does. A frame with no images is one
//! draw call. Instances wholly outside the clip are dropped on the CPU.
//!
//! **Layers.** Content is recorded into [`Layer`]s (base, popups, tooltips) and merged in layer order
//! by [`DrawList::finish`], so a popup opened in the middle of a frame still draws above everything
//! recorded after it.
//!
//! **No steady-state allocation** `[PERF-R1]`: [`DrawList::begin`] clears the vectors but keeps
//! their capacity; once a frame of the same complexity has been recorded, recording allocates
//! nothing.

use humanart_gpu::pod_struct;

use crate::color::Color;
use crate::geom::{Point, Rect};

pod_struct! {
    /// One quad as the shader reads it (`struct Instance` in `shaders/ui/quad.wgsl`), physical pixels.
    #[derive(Debug, PartialEq)]
    pub struct Instance(size = 112, align = 4) {
        /// `x0, y0, x1, y1` of the shape; for a segment its two end points.
        pub rect: [f32; 4],
        /// Corner radii: top-left, top-right, bottom-right, bottom-left.
        pub radii: [f32; 4],
        /// Fill, glyph or tint colour, linear, straight alpha.
        pub color: [f32; 4],
        /// Border colour, or the end colour of a gradient.
        pub color2: [f32; 4],
        /// Texture coordinates `u0, v0, u1, v1` mapped onto `rect` (glyphs, images).
        pub uv: [f32; 4],
        /// Clip rectangle `x0, y0, x1, y1`.
        pub clip: [f32; 4],
        /// `[a, b, kind, layer]`: `a` is the border width (shapes), the blur (shadows), the screen
        /// distance range (glyphs), the half width (segments) or the axis (gradients); `layer` is the
        /// atlas layer of a glyph; `b` is the edge shift of a glyph (stem darkening).
        pub params: [f32; 4],
    }
}

/// What an [`Instance`] draws (`params[2]`; the same numbers as the `KIND_*` constants of the shader).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    /// A rounded rectangle, filled and optionally bordered.
    Shape = 0,
    /// A rounded rectangle blurred into a soft shadow or glow.
    Shadow = 1,
    /// A glyph from the multi-channel distance-field atlas.
    Glyph = 2,
    /// An image, clipped to a rounded rectangle and tinted.
    Image = 3,
    /// A line segment with round caps.
    Segment = 4,
    /// A rounded rectangle filled with a linear gradient.
    Gradient = 5,
}

impl Kind {
    /// The value written into `params[2]`.
    #[inline]
    pub const fn code(self) -> f32 {
        self as u8 as f32
    }
}

/// An image texture registered with the renderer; [`TextureId::NONE`] is the built-in white texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct TextureId(pub u32);

impl TextureId {
    /// No image: shapes, shadows and glyphs.
    pub const NONE: Self = Self(0);
}

/// A run of consecutive instances drawn with one texture: one draw call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Batch {
    /// The image texture bound for the run.
    pub texture: TextureId,
    /// The first instance.
    pub start: u32,
    /// One past the last instance.
    pub end: u32,
}

/// Corner radii in logical pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Corners {
    /// Top-left.
    pub tl: f32,
    /// Top-right.
    pub tr: f32,
    /// Bottom-right.
    pub br: f32,
    /// Bottom-left.
    pub bl: f32,
}

impl Corners {
    /// Square corners.
    pub const ZERO: Self = Self { tl: 0.0, tr: 0.0, br: 0.0, bl: 0.0 };

    /// The same radius at every corner.
    #[inline]
    pub const fn all(r: f32) -> Self {
        Self { tl: r, tr: r, br: r, bl: r }
    }

    /// Rounded top corners only (tabs).
    #[inline]
    pub const fn top(r: f32) -> Self {
        Self { tl: r, tr: r, br: 0.0, bl: 0.0 }
    }

    /// Rounded bottom corners only.
    #[inline]
    pub const fn bottom(r: f32) -> Self {
        Self { tl: 0.0, tr: 0.0, br: r, bl: r }
    }
}

impl From<f32> for Corners {
    fn from(r: f32) -> Self {
        Self::all(r)
    }
}

/// The direction of a gradient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Left to right.
    Horizontal,
    /// Top to bottom.
    Vertical,
}

/// Where in the stacking order content goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layer {
    /// Panels and their widgets.
    Base = 0,
    /// Menus, dropdowns, popovers, floating panels, modal dialogs.
    Popup = 1,
    /// Tooltips, drag previews, toasts: above everything.
    Tooltip = 2,
}

/// The number of layers.
const LAYERS: usize = 3;

/// A glyph quad in physical pixels, as [`crate::text`] produces it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlyphQuad {
    /// The quad, physical pixels.
    pub rect: [f32; 4],
    /// Its atlas coordinates, normalised.
    pub uv: [f32; 4],
    /// The atlas layer.
    pub layer: u32,
    /// The distance range of the atlas in screen pixels at this size.
    pub screen_range: f32,
    /// How far the edge moves outward, screen pixels (stem darkening of small text; 0 for none).
    pub darken: f32,
}

/// The instances and batches of one layer.
#[derive(Debug, Default)]
struct LayerList {
    instances: Vec<Instance>,
    batches: Vec<Batch>,
}

impl LayerList {
    fn clear(&mut self) {
        self.instances.clear();
        self.batches.clear();
    }

    fn push(&mut self, instance: Instance, texture: TextureId) {
        let index = u32::try_from(self.instances.len()).unwrap_or(u32::MAX);
        self.instances.push(instance);
        match self.batches.last_mut() {
            Some(batch) if batch.texture == texture && batch.end == index => batch.end = index + 1,
            _ => self.batches.push(Batch { texture, start: index, end: index + 1 }),
        }
    }
}

/// Counters of the last recorded frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PaintStats {
    /// Instances kept.
    pub instances: u32,
    /// Instances dropped because they were wholly outside the clip or invisible.
    pub culled: u32,
    /// Batches (draw calls) after merging the layers.
    pub batches: u32,
}

/// Everything drawn in one frame, recorded in logical pixels and stored in physical pixels.
#[derive(Debug)]
pub struct DrawList {
    /// Physical pixels per logical pixel (display scale × UI scale).
    scale: f32,
    /// The target size in physical pixels.
    target: [f32; 2],
    layers: [LayerList; LAYERS],
    layer: Layer,
    /// The clip stack in logical pixels; the top is the current clip.
    clips: Vec<Rect>,
    /// The current clip in physical pixels, snapped.
    clip_px: [f32; 4],
    merged: Vec<Instance>,
    merged_batches: Vec<Batch>,
    culled: u32,
}

impl Default for DrawList {
    fn default() -> Self {
        Self::new()
    }
}

impl DrawList {
    /// An empty list at scale 1.
    pub fn new() -> Self {
        Self {
            scale: 1.0,
            target: [0.0, 0.0],
            layers: Default::default(),
            layer: Layer::Base,
            clips: Vec::new(),
            clip_px: [0.0; 4],
            merged: Vec::new(),
            merged_batches: Vec::new(),
            culled: 0,
        }
    }

    /// Starts a frame: forgets the previous one (keeping the memory) and sets the target size in
    /// physical pixels and the scale (physical pixels per logical pixel; non-finite or non-positive
    /// values become 1). The clip is the whole target.
    pub fn begin(&mut self, target_width: u32, target_height: u32, scale: f32) {
        self.scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
        #[allow(clippy::cast_precision_loss)] // INVARIANT: target sizes are at most 16384.
        let target = [target_width as f32, target_height as f32];
        self.target = target;
        for layer in &mut self.layers {
            layer.clear();
        }
        self.layer = Layer::Base;
        self.clips.clear();
        self.clip_px = [0.0, 0.0, target[0], target[1]];
        self.merged.clear();
        self.merged_batches.clear();
        self.culled = 0;
    }

    /// Physical pixels per logical pixel.
    #[inline]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// The target size in logical pixels.
    #[inline]
    pub fn viewport(&self) -> Rect {
        Rect::new(0.0, 0.0, self.target[0] / self.scale, self.target[1] / self.scale)
    }

    /// Selects the layer the next primitives go to.
    #[inline]
    pub fn set_layer(&mut self, layer: Layer) {
        self.layer = layer;
    }

    /// The current layer.
    #[inline]
    pub fn layer(&self) -> Layer {
        self.layer
    }

    /// Narrows the clip to its intersection with `rect` (logical pixels) until the matching
    /// [`DrawList::pop_clip`].
    pub fn push_clip(&mut self, rect: Rect) {
        let clip = self.clip().intersect(rect);
        self.clips.push(clip);
        self.clip_px = self.physical_clip(clip);
    }

    /// Replaces the clip with `rect` cut to the target, ignoring the enclosing clip, until the
    /// matching [`DrawList::pop_clip`] (overlays: menus and dialogs opened from inside a clipped
    /// scroll area).
    pub fn push_clip_replace(&mut self, rect: Rect) {
        let clip = self.viewport().intersect(rect);
        self.clips.push(clip);
        self.clip_px = self.physical_clip(clip);
    }

    /// Restores the clip in force before the last [`DrawList::push_clip`]; does nothing on an empty
    /// stack.
    pub fn pop_clip(&mut self) {
        self.clips.pop();
        let clip = self.clip();
        self.clip_px = self.physical_clip(clip);
    }

    /// The current clip in logical pixels.
    pub fn clip(&self) -> Rect {
        self.clips.last().copied().unwrap_or_else(|| self.viewport())
    }

    /// The clip `rect` in snapped physical pixels, inside the target.
    fn physical_clip(&self, rect: Rect) -> [f32; 4] {
        let s = self.scale;
        let x0 = (rect.x * s).round().clamp(0.0, self.target[0]);
        let y0 = (rect.y * s).round().clamp(0.0, self.target[1]);
        let x1 = (rect.right() * s).round().clamp(x0, self.target[0]);
        let y1 = (rect.bottom() * s).round().clamp(y0, self.target[1]);
        [x0, y0, x1, y1]
    }

    /// Whether the physical bounds `[x0, y0, x1, y1]` touch the current clip.
    #[inline]
    fn visible(&self, b: [f32; 4]) -> bool {
        let c = self.clip_px;
        b[0] < c[2] && b[2] > c[0] && b[1] < c[3] && b[3] > c[1] && c[2] > c[0] && c[3] > c[1]
    }

    /// Appends an instance whose drawn area lies within `bounds` (physical pixels), unless it is
    /// invisible.
    fn push(&mut self, mut instance: Instance, bounds: [f32; 4], texture: TextureId) {
        if !self.visible(bounds) {
            self.culled += 1;
            return;
        }
        instance.clip = self.clip_px;
        self.layers[self.layer as usize].push(instance, texture);
    }

    /// `rect` in snapped physical pixels as `[x0, y0, x1, y1]`.
    #[inline]
    fn snapped(&self, rect: Rect) -> [f32; 4] {
        let s = self.scale;
        let r = rect.snap(s);
        [r.x * s, r.y * s, r.right() * s, r.bottom() * s]
    }

    /// Radii in physical pixels, each at most half the shorter side.
    #[inline]
    fn radii(&self, corners: Corners, px: [f32; 4]) -> [f32; 4] {
        let limit = ((px[2] - px[0]).min(px[3] - px[1]) * 0.5).max(0.0);
        let s = self.scale;
        [corners.tl, corners.tr, corners.br, corners.bl].map(|r| (r * s).clamp(0.0, limit))
    }

    /// A filled rounded rectangle.
    pub fn rect(&mut self, rect: Rect, color: Color, radius: impl Into<Corners>) {
        self.bordered(rect, color, radius, 0.0, Color::TRANSPARENT);
    }

    /// A rounded rectangle with a border of `width` logical pixels drawn inside its edge, over the
    /// fill (pass [`Color::TRANSPARENT`] as `fill` for an outline).
    pub fn bordered(&mut self, rect: Rect, fill: Color, radius: impl Into<Corners>, width: f32, border: Color) {
        let width = if width.is_finite() { width.max(0.0) } else { 0.0 };
        if rect.is_empty() || (fill.a <= 0.0 && (border.a <= 0.0 || width <= 0.0)) {
            self.culled += 1;
            return;
        }
        let px = self.snapped(rect);
        if px[2] <= px[0] || px[3] <= px[1] {
            self.culled += 1;
            return;
        }
        // Border widths round to whole physical pixels (at least one) so hairlines stay crisp.
        let border_px = if width > 0.0 { (width * self.scale).round().max(1.0) } else { 0.0 };
        let instance = Instance {
            rect: px,
            radii: self.radii(radius.into(), px),
            color: fill.to_array(),
            color2: border.to_array(),
            uv: [0.0; 4],
            clip: [0.0; 4],
            params: [border_px, 0.0, Kind::Shape.code(), 0.0],
        };
        self.push(instance, [px[0] - 1.0, px[1] - 1.0, px[2] + 1.0, px[3] + 1.0], TextureId::NONE);
    }

    /// A soft shadow or glow: `rect` as a rounded rectangle blurred by `blur` logical pixels (alpha
    /// `1 - smoothstep(-blur, blur, d)` over the signed distance `d`, as in the reference).
    pub fn shadow(&mut self, rect: Rect, color: Color, radius: f32, blur: f32) {
        if rect.is_empty() || color.a <= 0.0 {
            self.culled += 1;
            return;
        }
        let s = self.scale;
        let px = [rect.x * s, rect.y * s, rect.right() * s, rect.bottom() * s];
        let blur_px = if blur.is_finite() { (blur * s).max(0.5) } else { 0.5 };
        let instance = Instance {
            rect: px,
            radii: self.radii(Corners::all(radius), px),
            color: color.to_array(),
            color2: [0.0; 4],
            uv: [0.0; 4],
            clip: [0.0; 4],
            params: [blur_px, 0.0, Kind::Shadow.code(), 0.0],
        };
        let m = blur_px + 1.0;
        self.push(instance, [px[0] - m, px[1] - m, px[2] + m, px[3] + m], TextureId::NONE);
    }

    /// A rounded rectangle filled with a linear gradient from `from` to `to` along `axis`.
    pub fn gradient(&mut self, rect: Rect, from: Color, to: Color, radius: impl Into<Corners>, axis: Axis) {
        if rect.is_empty() || (from.a <= 0.0 && to.a <= 0.0) {
            self.culled += 1;
            return;
        }
        let px = self.snapped(rect);
        let instance = Instance {
            rect: px,
            radii: self.radii(radius.into(), px),
            color: from.to_array(),
            color2: to.to_array(),
            uv: [0.0; 4],
            clip: [0.0; 4],
            params: [if axis == Axis::Vertical { 1.0 } else { 0.0 }, 0.0, Kind::Gradient.code(), 0.0],
        };
        self.push(instance, [px[0] - 1.0, px[1] - 1.0, px[2] + 1.0, px[3] + 1.0], TextureId::NONE);
    }

    /// A line from `a` to `b`, `width` logical pixels wide, with round caps (not pixel-snapped).
    pub fn line(&mut self, a: Point, b: Point, width: f32, color: Color) {
        if color.a <= 0.0
            || !(width.is_finite() && width > 0.0)
            || !(a.x.is_finite() && a.y.is_finite() && b.x.is_finite() && b.y.is_finite())
        {
            self.culled += 1;
            return;
        }
        let s = self.scale;
        let half = width * s * 0.5;
        let (ax, ay, bx, by) = (a.x * s, a.y * s, b.x * s, b.y * s);
        let instance = Instance {
            rect: [ax, ay, bx, by],
            radii: [0.0; 4],
            color: color.to_array(),
            color2: [0.0; 4],
            uv: [0.0; 4],
            clip: [0.0; 4],
            params: [half, 0.0, Kind::Segment.code(), 0.0],
        };
        let m = half + 1.0;
        self.push(instance, [ax.min(bx) - m, ay.min(by) - m, ax.max(bx) + m, ay.max(by) + m], TextureId::NONE);
    }

    /// A filled circle (not pixel-snapped, so it can move smoothly).
    pub fn circle(&mut self, center: Point, radius: f32, color: Color) {
        if !(radius.is_finite() && radius > 0.0) || color.a <= 0.0 {
            self.culled += 1;
            return;
        }
        let s = self.scale;
        let r = radius * s;
        let px = [center.x * s - r, center.y * s - r, center.x * s + r, center.y * s + r];
        let instance = Instance {
            rect: px,
            radii: [r; 4],
            color: color.to_array(),
            color2: [0.0; 4],
            uv: [0.0; 4],
            clip: [0.0; 4],
            params: [0.0, 0.0, Kind::Shape.code(), 0.0],
        };
        self.push(instance, [px[0] - 1.0, px[1] - 1.0, px[2] + 1.0, px[3] + 1.0], TextureId::NONE);
    }

    /// An image (a texture registered with the renderer), the part `uv` (`u0, v0, u1, v1`, normalised,
    /// V down) stretched over `rect`, multiplied by `tint` and clipped to rounded corners.
    pub fn image(&mut self, rect: Rect, texture: TextureId, uv: [f32; 4], tint: Color, radius: impl Into<Corners>) {
        if rect.is_empty() || tint.a <= 0.0 {
            self.culled += 1;
            return;
        }
        let px = self.snapped(rect);
        let instance = Instance {
            rect: px,
            radii: self.radii(radius.into(), px),
            color: tint.to_array(),
            color2: [0.0; 4],
            uv,
            clip: [0.0; 4],
            params: [0.0, 0.0, Kind::Image.code(), 0.0],
        };
        self.push(instance, [px[0] - 1.0, px[1] - 1.0, px[2] + 1.0, px[3] + 1.0], texture);
    }

    /// A glyph quad (physical pixels, from [`crate::text`]) in `color`.
    pub fn glyph(&mut self, quad: &GlyphQuad, color: Color) {
        if color.a <= 0.0 {
            self.culled += 1;
            return;
        }
        #[allow(clippy::cast_precision_loss)] // INVARIANT: atlas layers are few (under 256).
        let layer = quad.layer as f32;
        let instance = Instance {
            rect: quad.rect,
            radii: [0.0; 4],
            color: color.to_array(),
            color2: [0.0; 4],
            uv: quad.uv,
            clip: [0.0; 4],
            params: [quad.screen_range, quad.darken, Kind::Glyph.code(), layer],
        };
        self.push(instance, quad.rect, TextureId::NONE);
    }

    /// The keyboard focus ring around `rect` `[UI-09]`: a `width`-wide outline `offset` outside the
    /// rectangle, its radius grown to follow the control's corners.
    pub fn focus_ring(&mut self, rect: Rect, radius: f32, width: f32, offset: f32, color: Color) {
        let ring = rect.expand(offset + width);
        self.bordered(ring, Color::TRANSPARENT, radius + offset + width, width, color);
    }

    /// Merges the layers in stacking order; call once after recording, before
    /// [`DrawList::instances`] and [`DrawList::batches`]. Returns the frame's counters.
    pub fn finish(&mut self) -> PaintStats {
        self.merged.clear();
        self.merged_batches.clear();
        for layer in &self.layers {
            let base = u32::try_from(self.merged.len()).unwrap_or(u32::MAX);
            self.merged.extend_from_slice(&layer.instances);
            for batch in &layer.batches {
                let (start, end) = (base + batch.start, base + batch.end);
                match self.merged_batches.last_mut() {
                    Some(last) if last.texture == batch.texture && last.end == start => last.end = end,
                    _ => self.merged_batches.push(Batch { texture: batch.texture, start, end }),
                }
            }
        }
        self.stats()
    }

    /// The merged instances (after [`DrawList::finish`]).
    #[inline]
    pub fn instances(&self) -> &[Instance] {
        &self.merged
    }

    /// The merged batches, one draw call each (after [`DrawList::finish`]).
    #[inline]
    pub fn batches(&self) -> &[Batch] {
        &self.merged_batches
    }

    /// The counters of the merged frame.
    pub fn stats(&self) -> PaintStats {
        PaintStats {
            instances: u32::try_from(self.merged.len()).unwrap_or(u32::MAX),
            culled: self.culled,
            batches: u32::try_from(self.merged_batches.len()).unwrap_or(u32::MAX),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_is_112_bytes_in_field_order() {
        let mut list = DrawList::new();
        list.begin(100, 100, 1.0);
        list.bordered(Rect::new(10.0, 10.0, 20.0, 10.0), Color::WHITE, 4.0, 1.0, Color::BLACK);
        list.finish();
        let bytes = humanart_gpu::pod::slice_as_bytes(list.instances());
        assert_eq!(bytes.len(), 112);
        assert_eq!(&bytes[0..4], &10.0_f32.to_le_bytes());
        assert_eq!(&bytes[104..108], &Kind::Shape.code().to_le_bytes());
    }

    #[test]
    fn ui_02_clip_changes_do_not_break_batches() {
        // [UI-02] Clipping is per instance; only image changes start a new batch.
        let mut list = DrawList::new();
        list.begin(400, 300, 1.5);
        for i in 0..50 {
            let f = i as f32;
            list.push_clip(Rect::new(f, f, 100.0, 100.0));
            list.rect(Rect::new(f, f, 30.0, 10.0), Color::WHITE, 2.0);
            list.pop_clip();
        }
        list.image(Rect::new(0.0, 0.0, 10.0, 10.0), TextureId(1), [0.0, 0.0, 1.0, 1.0], Color::WHITE, 0.0);
        list.rect(Rect::new(0.0, 0.0, 10.0, 10.0), Color::WHITE, 0.0);
        let stats = list.finish();
        assert_eq!(stats.instances, 52);
        assert_eq!(stats.batches, 3);
        assert_eq!(list.batches()[1], Batch { texture: TextureId(1), start: 50, end: 51 });
    }

    #[test]
    fn layers_draw_in_order_and_merge_batches() {
        let mut list = DrawList::new();
        list.begin(100, 100, 1.0);
        list.rect(Rect::new(0.0, 0.0, 10.0, 10.0), Color::WHITE, 0.0);
        list.set_layer(Layer::Tooltip);
        list.rect(Rect::new(1.0, 0.0, 10.0, 10.0), Color::BLACK, 0.0);
        list.set_layer(Layer::Base);
        list.rect(Rect::new(2.0, 0.0, 10.0, 10.0), Color::WHITE, 0.0);
        let stats = list.finish();
        assert_eq!(stats.batches, 1);
        let x: Vec<f32> = list.instances().iter().map(|i| i.rect[0]).collect();
        assert_eq!(x, [0.0, 2.0, 1.0]);
    }

    #[test]
    fn culls_outside_clip_and_invisible() {
        let mut list = DrawList::new();
        list.begin(100, 100, 2.0);
        list.push_clip(Rect::new(0.0, 0.0, 20.0, 20.0));
        list.rect(Rect::new(30.0, 30.0, 5.0, 5.0), Color::WHITE, 0.0);
        list.rect(Rect::new(1.0, 1.0, 5.0, 5.0), Color::TRANSPARENT, 0.0);
        list.rect(Rect::new(1.0, 1.0, -5.0, 5.0), Color::WHITE, 0.0);
        list.line(Point::new(0.0, 0.0), Point::new(f32::NAN, 1.0), 1.0, Color::WHITE);
        list.rect(Rect::new(1.0, 1.0, 5.0, 5.0), Color::WHITE, 0.0);
        let stats = list.finish();
        assert_eq!((stats.instances, stats.culled), (1, 4));
        assert_eq!(list.instances()[0].clip, [0.0, 0.0, 40.0, 40.0]);
        list.pop_clip();
        list.pop_clip(); // Extra pops are harmless.
        assert_eq!(list.clip(), Rect::new(0.0, 0.0, 50.0, 50.0));
    }

    #[test]
    fn ui_04_shapes_snap_to_physical_pixels() {
        let mut list = DrawList::new();
        list.begin(200, 200, 1.5);
        list.bordered(Rect::new(10.2, 3.3, 20.0, 7.0), Color::WHITE, 100.0, 1.0, Color::BLACK);
        list.finish();
        let i = list.instances()[0];
        for v in i.rect {
            assert_eq!(v, v.round(), "{:?}", i.rect);
        }
        // Border at least one whole physical pixel; radii at most half the short side.
        assert_eq!(i.params[0], 2.0);
        assert!(i.radii.iter().all(|&r| r <= (i.rect[3] - i.rect[1]) * 0.5));
    }

    #[test]
    fn perf_r1_recording_reuses_memory() {
        let mut list = DrawList::new();
        let frame = |list: &mut DrawList| {
            list.begin(800, 600, 1.0);
            for i in 0..200 {
                let f = i as f32;
                list.push_clip(Rect::new(0.0, f, 400.0, 30.0));
                list.bordered(Rect::new(4.0, f, 100.0, 20.0), Color::WHITE, 4.0, 1.0, Color::BLACK);
                list.set_layer(if i % 7 == 0 { Layer::Popup } else { Layer::Base });
                list.shadow(Rect::new(4.0, f, 100.0, 20.0), Color::BLACK, 4.0, 8.0);
                list.pop_clip();
            }
            list.finish();
        };
        frame(&mut list);
        // [PERF-R1] The second frame of the same complexity allocates nothing.
        let stats = humanart_core::alloc::count_in(|| frame(&mut list));
        assert!(humanart_core::alloc::is_installed());
        assert_eq!(stats.events(), 0);
    }
}
