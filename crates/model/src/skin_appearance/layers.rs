//! The skin layer stack `[SKN-02]` `[SKN-12]` `[DOC-04]`: fill layers (constant attribute values),
//! content layers (painted strokes, procedural features, imported decals: texels with coverage in
//! rectangles of the UV tiles), blend modes, opacity and zone masks, with the dirty rectangles
//! every change leaves behind.
//!
//! The stack is the CPU description the GPU compositor (`humanart-render::skin`) mirrors; the
//! per-texel reference of what it computes is [`composite_texel`]. Changing a fill layer dirties
//! every tile; changing content dirties only its rectangles; painting dirties only the painted
//! rectangle.

use super::channels::{ATTRIBUTE_COUNT, Attribute, AttributeSet, BaseValues};
use super::dirty::{DirtyRects, Rect};
use super::error::SkinError;
use super::zones::{ZONE_SLOTS, Zone, ZoneTable};

/// Most layers in one stack (the GPU layer table's size).
pub const MAX_LAYERS: usize = 32;
/// Attributes one content layer carries per texel (one RGBA texel).
pub const CONTENT_ATTRIBUTES: usize = 4;
/// Largest content rectangle side in texels.
pub const MAX_CONTENT_SIDE: u32 = 4096;

/// How a layer combines with what is below it. `w` is opacity × coverage × zone mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BlendMode {
    /// `v + (value - v) · w`.
    Mix = 0,
    /// `v + value · w` (offsets, e.g. "+0.1 melanin").
    Add = 1,
    /// `v · (1 + (value - 1) · w)`.
    Multiply = 2,
    /// `v + (max(v, value) - v) · w`.
    Max = 3,
    /// `v + (min(v, value) - v) · w`.
    Min = 4,
}

impl BlendMode {
    /// Every mode in GPU order.
    pub const ALL: [Self; 5] = [Self::Mix, Self::Add, Self::Multiply, Self::Max, Self::Min];

    /// The stable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mix => "mix",
            Self::Add => "add",
            Self::Multiply => "multiply",
            Self::Max => "max",
            Self::Min => "min",
        }
    }

    /// The mode with a stable name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.name() == name)
    }

    /// Blends `value` over `v` with weight `w`.
    pub fn apply(self, v: f32, value: f32, w: f32) -> f32 {
        match self {
            Self::Mix => v + (value - v) * w,
            Self::Add => v + value * w,
            Self::Multiply => v * (1.0 + (value - 1.0) * w),
            Self::Max => v + (v.max(value) - v) * w,
            Self::Min => v + (v.min(value) - v) * w,
        }
    }
}

/// A stable layer id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerId(pub u32);

/// Texels of a content layer in one rectangle of one tile.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerContent {
    /// The UV tile.
    pub tile: u32,
    /// Where the texels lie in the tile.
    pub rect: Rect,
    /// Per texel (row-major in `rect`), the values of the layer's content attributes.
    pub values: Vec<[f32; CONTENT_ATTRIBUTES]>,
    /// Per texel, how much the layer applies (0..1).
    pub coverage: Vec<f32>,
}

impl LayerContent {
    /// Empty content (zero values, zero coverage) over `rect` of `tile`.
    pub fn new(tile: u32, rect: Rect) -> Result<Self, SkinError> {
        if rect.is_empty() || rect.w > MAX_CONTENT_SIDE || rect.h > MAX_CONTENT_SIDE {
            return Err(SkinError::limit(
                "content rectangle side",
                u64::from(MAX_CONTENT_SIDE),
                u64::from(rect.w.max(rect.h)),
            ));
        }
        let n = rect.area() as usize;
        Ok(Self { tile, rect, values: vec![[0.0; CONTENT_ATTRIBUTES]; n], coverage: vec![0.0; n] })
    }

    /// The index of tile texel `(x, y)` in the content, if inside.
    pub fn index(&self, x: u32, y: u32) -> Option<usize> {
        if x < self.rect.x || y < self.rect.y || x >= self.rect.right() || y >= self.rect.bottom() {
            return None;
        }
        Some(((y - self.rect.y) * self.rect.w + (x - self.rect.x)) as usize)
    }

    /// Whether sizes agree with the rectangle.
    pub fn is_consistent(&self) -> bool {
        let n = self.rect.area() as usize;
        self.values.len() == n && self.coverage.len() == n
    }
}

/// What a layer holds.
#[derive(Clone, Debug, PartialEq)]
pub enum LayerKind {
    /// The same values everywhere for the attributes in the set.
    Fill {
        /// The attributes written.
        attributes: AttributeSet,
        /// Values, indexed by [`Attribute::index`].
        values: [f32; ATTRIBUTE_COUNT],
    },
    /// Texels with coverage in rectangles.
    Content {
        /// Which attribute each content component writes (`None`: unused).
        attributes: [Option<Attribute>; CONTENT_ATTRIBUTES],
        /// At most one rectangle per tile.
        tiles: Vec<LayerContent>,
    },
}

/// One layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    /// Stable id.
    pub id: LayerId,
    /// Display name.
    pub name: String,
    /// Blend mode.
    pub blend: BlendMode,
    /// Opacity 0..1.
    pub opacity: f32,
    /// Hidden layers are skipped.
    pub enabled: bool,
    /// Restricts the layer to a zone's mask.
    pub zone: Option<Zone>,
    /// The values.
    pub kind: LayerKind,
    /// Bumped on every change (the GPU side re-uploads content when it moves).
    pub version: u64,
}

impl Layer {
    /// A fill layer.
    pub fn fill(id: LayerId, name: &str, blend: BlendMode, values: &[(Attribute, f32)]) -> Self {
        let mut v = [0.0; ATTRIBUTE_COUNT];
        let mut attributes = AttributeSet::EMPTY;
        for &(a, x) in values {
            v[a.index()] = x;
            attributes = attributes.with(a);
        }
        Self {
            id,
            name: name.to_owned(),
            blend,
            opacity: 1.0,
            enabled: true,
            zone: None,
            kind: LayerKind::Fill { attributes, values: v },
            version: 0,
        }
    }

    /// A content layer without texels yet.
    pub fn content(
        id: LayerId,
        name: &str,
        blend: BlendMode,
        attributes: [Option<Attribute>; CONTENT_ATTRIBUTES],
    ) -> Self {
        Self {
            id,
            name: name.to_owned(),
            blend,
            opacity: 1.0,
            enabled: true,
            zone: None,
            kind: LayerKind::Content { attributes, tiles: Vec::new() },
            version: 0,
        }
    }

    /// The attributes the layer writes.
    pub fn attributes(&self) -> AttributeSet {
        match &self.kind {
            LayerKind::Fill { attributes, .. } => *attributes,
            LayerKind::Content { attributes, .. } => {
                attributes.iter().flatten().fold(AttributeSet::EMPTY, |s, a| s.with(*a))
            }
        }
    }

    /// The content of `tile`, if any.
    pub fn content_of(&self, tile: u32) -> Option<&LayerContent> {
        match &self.kind {
            LayerKind::Content { tiles, .. } => tiles.iter().find(|c| c.tile == tile),
            LayerKind::Fill { .. } => None,
        }
    }

    /// Marks what the layer covers in `dirty`.
    fn mark_footprint(&self, dirty: &mut DirtyRects) {
        match &self.kind {
            LayerKind::Fill { .. } => dirty.mark_all(),
            LayerKind::Content { tiles, .. } => {
                for c in tiles {
                    dirty.mark(c.tile, c.rect);
                }
            }
        }
    }
}

/// The ordered layers of one character's skin (bottom first) with their dirty rectangles.
#[derive(Clone, Debug)]
pub struct LayerStack {
    layers: Vec<Layer>,
    dirty: DirtyRects,
    next_id: u32,
}

impl LayerStack {
    /// An empty stack over `tiles` tiles of `tile_size²` texels; everything starts dirty.
    pub fn new(tiles: u32, tile_size: u32) -> Result<Self, SkinError> {
        let mut dirty = DirtyRects::new(tiles, tile_size)?;
        dirty.mark_all();
        Ok(Self { layers: Vec::with_capacity(MAX_LAYERS), dirty, next_id: 1 })
    }

    /// The layers, bottom first.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// The dirty rectangles.
    pub fn dirty(&self) -> &DirtyRects {
        &self.dirty
    }

    /// The dirty rectangles, to drain after a resolve.
    pub fn dirty_mut(&mut self) -> &mut DirtyRects {
        &mut self.dirty
    }

    /// The tile side in texels.
    pub fn tile_size(&self) -> u32 {
        self.dirty.tile_size()
    }

    /// A fresh layer id.
    pub fn allocate_id(&mut self) -> LayerId {
        let id = LayerId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1).max(1);
        id
    }

    /// The position of a layer.
    pub fn position(&self, id: LayerId) -> Option<usize> {
        self.layers.iter().position(|l| l.id == id)
    }

    /// A layer by id.
    pub fn get(&self, id: LayerId) -> Option<&Layer> {
        self.layers.iter().find(|l| l.id == id)
    }

    /// Checks a layer's content against the stack's tiles.
    fn check(&self, layer: &Layer) -> Result<(), SkinError> {
        if !(0.0..=1.0).contains(&layer.opacity) {
            return Err(SkinError::Invalid { what: "layer opacity outside 0..1" });
        }
        if let LayerKind::Content { tiles, .. } = &layer.kind {
            for (i, c) in tiles.iter().enumerate() {
                if c.tile >= self.dirty.tiles() {
                    return Err(SkinError::Invalid { what: "layer content on a tile the set does not have" });
                }
                if !c.is_consistent() || c.rect.clip(self.tile_size()) != c.rect || c.rect.is_empty() {
                    return Err(SkinError::Invalid { what: "layer content rectangle outside its tile or mis-sized" });
                }
                if tiles[..i].iter().any(|o| o.tile == c.tile) {
                    return Err(SkinError::Invalid { what: "two content rectangles on one tile" });
                }
            }
        }
        Ok(())
    }

    /// Inserts a layer at `index` (clamped to the top); its footprint becomes dirty.
    pub fn insert(&mut self, index: usize, layer: Layer) -> Result<(), SkinError> {
        if self.layers.len() >= MAX_LAYERS {
            return Err(SkinError::limit("layer count", MAX_LAYERS as u64, self.layers.len() as u64 + 1));
        }
        if self.get(layer.id).is_some() {
            return Err(SkinError::Invalid { what: "a layer with this id exists" });
        }
        self.check(&layer)?;
        layer.mark_footprint(&mut self.dirty);
        self.next_id = self.next_id.max(layer.id.0.wrapping_add(1)).max(1);
        let at = index.min(self.layers.len());
        self.layers.insert(at, layer);
        Ok(())
    }

    /// Adds a layer on top.
    pub fn push(&mut self, layer: Layer) -> Result<(), SkinError> {
        self.insert(usize::MAX, layer)
    }

    /// Removes a layer; its footprint becomes dirty.
    pub fn remove(&mut self, id: LayerId) -> Option<Layer> {
        let at = self.position(id)?;
        let layer = self.layers.remove(at);
        layer.mark_footprint(&mut self.dirty);
        Some(layer)
    }

    /// Replaces a layer (same id); the old and new footprints become dirty.
    pub fn replace(&mut self, mut layer: Layer) -> Result<(), SkinError> {
        let at = self.position(layer.id).ok_or(SkinError::Invalid { what: "no layer with this id" })?;
        self.check(&layer)?;
        self.layers[at].mark_footprint(&mut self.dirty);
        layer.mark_footprint(&mut self.dirty);
        layer.version = self.layers[at].version + 1;
        self.layers[at] = layer;
        Ok(())
    }

    /// Changes opacity / visibility / blend mode of a layer; its footprint becomes dirty.
    pub fn set_blend(&mut self, id: LayerId, opacity: f32, enabled: bool, blend: BlendMode) -> Result<(), SkinError> {
        if !(0.0..=1.0).contains(&opacity) {
            return Err(SkinError::Invalid { what: "layer opacity outside 0..1" });
        }
        let at = self.position(id).ok_or(SkinError::Invalid { what: "no layer with this id" })?;
        let layer = &mut self.layers[at];
        layer.opacity = opacity;
        layer.enabled = enabled;
        layer.blend = blend;
        layer.version += 1;
        let layer = &self.layers[at];
        layer.mark_footprint(&mut self.dirty);
        Ok(())
    }

    /// Paints into a content layer: `paint(tile_x, tile_y, values, coverage)` is called for every
    /// texel of `rect` that lies inside the layer's content on `tile`, and only that rectangle
    /// becomes dirty `[SKN-02]`. Returns the rectangle touched (empty when nothing was).
    pub fn paint(
        &mut self,
        id: LayerId,
        tile: u32,
        rect: Rect,
        mut paint: impl FnMut(u32, u32, &mut [f32; CONTENT_ATTRIBUTES], &mut f32),
    ) -> Result<Rect, SkinError> {
        let at = self.position(id).ok_or(SkinError::Invalid { what: "no layer with this id" })?;
        let layer = &mut self.layers[at];
        let LayerKind::Content { tiles, .. } = &mut layer.kind else {
            return Err(SkinError::Invalid { what: "painting needs a content layer" });
        };
        let Some(content) = tiles.iter_mut().find(|c| c.tile == tile) else { return Ok(Rect::default()) };
        let touched = rect.intersect(content.rect);
        for y in touched.y..touched.bottom() {
            for x in touched.x..touched.right() {
                let i = ((y - content.rect.y) * content.rect.w + (x - content.rect.x)) as usize;
                if let (Some(v), Some(c)) = (content.values.get_mut(i), content.coverage.get_mut(i)) {
                    paint(x, y, v, c);
                }
            }
        }
        if !touched.is_empty() {
            layer.version += 1;
            self.dirty.mark(tile, touched);
        }
        Ok(touched)
    }
}

/// The inputs of one texel the compositor reads besides the layers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TexelInputs {
    /// Zone mask weights.
    pub zones: [f32; ZONE_SLOTS],
    /// Baked thickness 0..1.
    pub thickness: f32,
    /// Baked cavity 0..1 (1 open).
    pub cavity: f32,
}

impl Default for TexelInputs {
    fn default() -> Self {
        Self { zones: [0.0; ZONE_SLOTS], thickness: 0.5, cavity: 1.0 }
    }
}

/// The composited attributes of one texel: base values, zone offsets, then every enabled layer
/// bottom to top, then each attribute clamped to its range. The CPU reference of the GPU
/// compositor's first pass.
pub fn composite_texel(
    stack: &LayerStack,
    base: &BaseValues,
    zones: &ZoneTable,
    inputs: &TexelInputs,
    tile: u32,
    x: u32,
    y: u32,
) -> [f32; ATTRIBUTE_COUNT] {
    let mut v = base.0;
    zones.apply(&inputs.zones, &mut v);
    for layer in stack.layers().iter().filter(|l| l.enabled && l.opacity > 0.0) {
        let zone_weight = layer.zone.map_or(1.0, |z| inputs.zones[z.slot()]);
        let w = layer.opacity * zone_weight;
        match &layer.kind {
            LayerKind::Fill { attributes, values } => {
                for a in attributes.iter() {
                    v[a.index()] = layer.blend.apply(v[a.index()], values[a.index()], w);
                }
            }
            LayerKind::Content { attributes, .. } => {
                let Some(content) = layer.content_of(tile) else { continue };
                let Some(i) = content.index(x, y) else { continue };
                let (Some(values), Some(&coverage)) = (content.values.get(i), content.coverage.get(i)) else {
                    continue;
                };
                let w = w * coverage.clamp(0.0, 1.0);
                for (a, &value) in attributes.iter().zip(values) {
                    if let Some(a) = a {
                        v[a.index()] = layer.blend.apply(v[a.index()], value, w);
                    }
                }
            }
        }
    }
    for (value, a) in v.iter_mut().zip(Attribute::ALL) {
        let (lo, hi) = a.range();
        *value = if value.is_finite() { value.clamp(lo, hi) } else { lo.max(0.0) };
    }
    v
}

/// Magic of encoded layer content.
pub const CONTENT_MAGIC: [u8; 4] = *b"SKLC";
/// Version of the encoding.
pub const CONTENT_VERSION: u16 = 1;
/// Bytes of the header.
const CONTENT_HEADER: usize = 4 + 2 + 2 + 4 * 5;

impl LayerContent {
    /// Encodes the content for the document (a layer's blob): magic `SKLC`, version u16, flags
    /// u16 (0), tile, x, y, w, h as u32, then `w·h` texels of 4 values and `w·h` coverages, all
    /// little-endian f32.
    pub fn encode(&self) -> Vec<u8> {
        let n = self.rect.area() as usize;
        let mut out = Vec::with_capacity(CONTENT_HEADER + n * 20);
        out.extend_from_slice(&CONTENT_MAGIC);
        out.extend_from_slice(&CONTENT_VERSION.to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes());
        for v in [self.tile, self.rect.x, self.rect.y, self.rect.w, self.rect.h] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for texel in self.values.iter().take(n) {
            for v in texel {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        for c in self.coverage.iter().take(n) {
            out.extend_from_slice(&c.to_le_bytes());
        }
        out
    }

    /// Decodes [`LayerContent::encode`]'s bytes `[ROB-01]`: the header is checked, the rectangle
    /// against [`MAX_CONTENT_SIDE`] and the texel count against the remaining bytes before
    /// anything is allocated; non-finite values are refused.
    pub fn decode(bytes: &[u8]) -> Result<Self, SkinError> {
        let format = |offset: usize, what: &'static str| SkinError::Format { offset: offset as u64, what };
        let header = bytes.get(..CONTENT_HEADER).ok_or(format(bytes.len(), "truncated header"))?;
        if header[..4] != CONTENT_MAGIC {
            return Err(format(0, "not skin layer content (bad magic)"));
        }
        if u16::from_le_bytes([header[4], header[5]]) != CONTENT_VERSION {
            return Err(format(4, "unsupported version"));
        }
        if u16::from_le_bytes([header[6], header[7]]) != 0 {
            return Err(format(6, "unknown flags"));
        }
        let word = |i: usize| {
            u32::from_le_bytes([header[8 + i * 4], header[9 + i * 4], header[10 + i * 4], header[11 + i * 4]])
        };
        let (tile, rect) = (word(0), Rect::new(word(1), word(2), word(3), word(4)));
        if rect.is_empty() || rect.w > MAX_CONTENT_SIDE || rect.h > MAX_CONTENT_SIDE {
            return Err(SkinError::limit(
                "content rectangle side",
                u64::from(MAX_CONTENT_SIDE),
                u64::from(rect.w.max(rect.h)),
            ));
        }
        if rect.x.checked_add(rect.w).is_none() || rect.y.checked_add(rect.h).is_none() {
            return Err(format(12, "rectangle overflows"));
        }
        let n = rect.area() as usize;
        let body = &bytes[CONTENT_HEADER..];
        if body.len() != n * 20 {
            return Err(format(CONTENT_HEADER, "texel data does not match the rectangle"));
        }
        let floats = body.as_chunks::<4>().0;
        if floats.iter().any(|b| !f32::from_le_bytes(*b).is_finite()) {
            return Err(format(CONTENT_HEADER, "non-finite value"));
        }
        let mut content = Self::new(tile, rect)?;
        for (texel, chunk) in content.values.iter_mut().zip(floats.as_chunks::<4>().0) {
            for (v, b) in texel.iter_mut().zip(chunk) {
                *v = f32::from_le_bytes(*b);
            }
        }
        for (c, b) in content.coverage.iter_mut().zip(&floats[n * 4..]) {
            *c = f32::from_le_bytes(*b);
        }
        Ok(content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack() -> LayerStack {
        LayerStack::new(2, 64).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn blend_modes() {
        assert!((BlendMode::Mix.apply(0.2, 0.6, 0.5) - 0.4).abs() < 1e-6);
        assert!((BlendMode::Add.apply(0.2, 0.6, 0.5) - 0.5).abs() < 1e-6);
        assert!((BlendMode::Multiply.apply(0.5, 0.0, 1.0)).abs() < 1e-7);
        assert_eq!(BlendMode::Multiply.apply(0.5, 0.0, 0.0), 0.5);
        assert_eq!(BlendMode::Max.apply(0.2, 0.6, 1.0), 0.6);
        assert_eq!(BlendMode::Max.apply(0.8, 0.6, 1.0), 0.8);
        assert!((BlendMode::Min.apply(0.8, 0.6, 0.5) - 0.7).abs() < 1e-6);
        for m in BlendMode::ALL {
            assert_eq!(BlendMode::from_name(m.name()), Some(m));
        }
    }

    #[test]
    fn skn02_layer_changes_dirty_only_their_footprint() {
        let mut s = stack();
        s.dirty_mut().clear();
        let id = s.allocate_id();
        let mut layer = Layer::content(id, "freckles", BlendMode::Add, [Some(Attribute::Melanin), None, None, None]);
        let content = LayerContent::new(1, Rect::new(10, 10, 8, 8)).unwrap_or_else(|e| panic!("{e}"));
        if let LayerKind::Content { tiles, .. } = &mut layer.kind {
            tiles.push(content);
        }
        s.push(layer).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(s.dirty().rects(1), &[Rect::new(10, 10, 8, 8)]);
        assert!(s.dirty().rects(0).is_empty());
        s.dirty_mut().clear();
        // Painting a stroke dirties only the painted part.
        let touched = s
            .paint(id, 1, Rect::new(12, 12, 2, 2), |_, _, v, c| {
                v[0] = 0.2;
                *c = 1.0;
            })
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(touched, Rect::new(12, 12, 2, 2));
        assert_eq!(s.dirty().rects(1), &[touched]);
        assert_eq!(s.get(id).map(|l| l.version), Some(1));
        // A fill layer dirties everything.
        s.dirty_mut().clear();
        let fill = Layer::fill(s.allocate_id(), "tan", BlendMode::Add, &[(Attribute::Melanin, 0.1)]);
        s.push(fill).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(s.dirty().dirty_area(), 2 * 64 * 64);
    }

    #[test]
    fn skn12_override_and_composite_order() {
        let mut s = stack();
        let base = BaseValues::default();
        let zones = ZoneTable::default();
        let inputs = TexelInputs::default();
        let v0 = composite_texel(&s, &base, &zones, &inputs, 0, 5, 5);
        assert_eq!(v0[Attribute::Melanin.index()], 0.3);
        s.push(Layer::fill(LayerId(1), "darker", BlendMode::Add, &[(Attribute::Melanin, 0.2)]))
            .unwrap_or_else(|e| panic!("{e}"));
        let mut paint = Layer::fill(
            LayerId(2),
            "paint",
            BlendMode::Mix,
            &[(Attribute::OverrideRed, 1.0), (Attribute::OverrideCoverage, 1.0)],
        );
        paint.opacity = 0.5;
        paint.zone = Some(Zone::Lips);
        s.push(paint).unwrap_or_else(|e| panic!("{e}"));
        let v = composite_texel(&s, &base, &zones, &inputs, 0, 5, 5);
        assert!((v[Attribute::Melanin.index()] - 0.5).abs() < 1e-6);
        assert_eq!(v[Attribute::OverrideCoverage.index()], 0.0, "outside the lips zone");
        let mut lips = inputs;
        lips.zones[Zone::Lips.slot()] = 1.0;
        let v = composite_texel(&s, &base, &zones, &lips, 0, 5, 5);
        assert_eq!(v[Attribute::OverrideCoverage.index()], 0.5);
        assert_eq!(v[Attribute::OverrideRed.index()], 0.5);
        // Disabled layers are skipped; ranges are clamped.
        s.set_blend(LayerId(1), 1.0, false, BlendMode::Add).unwrap_or_else(|e| panic!("{e}"));
        s.push(Layer::fill(LayerId(3), "huge", BlendMode::Add, &[(Attribute::Haemoglobin, 5.0)]))
            .unwrap_or_else(|e| panic!("{e}"));
        let v = composite_texel(&s, &base, &zones, &inputs, 0, 5, 5);
        assert_eq!(v[Attribute::Melanin.index()], 0.3);
        assert_eq!(v[Attribute::Haemoglobin.index()], 1.0);
    }

    #[test]
    fn stack_validation() {
        let mut s = stack();
        let mut bad = Layer::content(LayerId(9), "bad", BlendMode::Mix, [None; 4]);
        if let LayerKind::Content { tiles, .. } = &mut bad.kind {
            tiles.push(LayerContent::new(5, Rect::new(0, 0, 4, 4)).unwrap_or_else(|e| panic!("{e}")));
        }
        assert!(s.push(bad).is_err(), "unknown tile");
        let mut bad = Layer::content(LayerId(9), "bad", BlendMode::Mix, [None; 4]);
        if let LayerKind::Content { tiles, .. } = &mut bad.kind {
            tiles.push(LayerContent::new(0, Rect::new(60, 0, 8, 4)).unwrap_or_else(|e| panic!("{e}")));
        }
        assert!(s.push(bad).is_err(), "outside the tile");
        assert!(LayerContent::new(0, Rect::new(0, 0, 0, 4)).is_err());
        let mut l = Layer::fill(LayerId(1), "a", BlendMode::Mix, &[]);
        s.push(l.clone()).unwrap_or_else(|e| panic!("{e}"));
        assert!(s.push(l.clone()).is_err(), "duplicate id");
        l.opacity = 2.0;
        assert!(s.replace(l).is_err());
        for i in 0..MAX_LAYERS - 1 {
            s.push(Layer::fill(LayerId(100 + i as u32), "f", BlendMode::Mix, &[])).unwrap_or_else(|e| panic!("{e}"));
        }
        assert!(s.push(Layer::fill(LayerId(999), "f", BlendMode::Mix, &[])).is_err(), "full");
        assert!(s.remove(LayerId(1)).is_some());
        assert!(s.remove(LayerId(1)).is_none());
        assert!(s.allocate_id().0 > 999 - 1 || s.allocate_id().0 > 100);
    }

    #[test]
    fn content_round_trip_and_rob01_fuzz() {
        let mut c = LayerContent::new(1, Rect::new(3, 4, 5, 2)).unwrap_or_else(|e| panic!("{e}"));
        for (i, (v, cov)) in c.values.iter_mut().zip(c.coverage.iter_mut()).enumerate() {
            *v = [i as f32, -0.5, 0.25, 2.0];
            *cov = i as f32 / 10.0;
        }
        let bytes = c.encode();
        assert_eq!(LayerContent::decode(&bytes), Ok(c.clone()));
        assert!(LayerContent::decode(&bytes[..bytes.len() - 1]).is_err());
        let mut huge = bytes.clone();
        huge[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(LayerContent::decode(&huge).is_err(), "a huge rectangle is refused before allocating");
        // [ROB-01] Mutation fuzz: never panics, never loops, anything accepted re-encodes the same.
        humanart_core::fuzz::run(
            "skin layer content",
            &[bytes],
            20_000,
            0x5eed,
            std::time::Duration::from_secs(60),
            |data| {
                if let Ok(content) = LayerContent::decode(data) {
                    assert_eq!(content.encode(), data);
                }
            },
        );
    }
}
