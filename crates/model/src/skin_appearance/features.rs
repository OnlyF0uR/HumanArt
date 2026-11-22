//! The skin features library `[OTH-02]` `[TOOL-06]`: procedural generators for freckles, moles,
//! scars, acne, stretch marks, veins, tattoo decals, wrinkles, under-eye darkness, sun spots and
//! lip lines. Every generator writes a content layer ([`Layer`]) over one rectangle of one UV tile,
//! so its output is an ordinary skin layer: blendable, maskable, paintable, re-generated with a
//! new seed.
//!
//! Placement follows a **placement mask** (a weight per texel of the rectangle, from the region
//! masks; `None` = everywhere), sampled by Poisson-disk dart throwing (`[TOOL-06]` "mole placement
//! by Poisson sampling in regions"). Sizes are in millimetres and converted with the tile's texel
//! size. Every parameter is listed with a stable id, range and default in [`GeneratorKind::params`]
//! (registered under `skin.features.*` in `data/template/params/skin.json`, kept equal by a test);
//! defaults are artistic.
//!
//! Generation is editor-time work (a command), not a per-frame path.

use humanart_core::rng::Rng;
use humanart_io::{ColorSpace, Image, Samples};
use humanart_math::color::transfer::srgb8_to_linear;

use super::channels::Attribute;
use super::dirty::Rect;
use super::error::SkinError;
use super::layers::{BlendMode, CONTENT_ATTRIBUTES, Layer, LayerContent, LayerId, LayerKind};
use super::noise::fbm;

/// Most features one generator places.
pub const MAX_FEATURES: usize = 20_000;
/// Most parameters of one generator.
pub const MAX_PARAMS: usize = 8;
/// Deepest vein branching.
pub const MAX_VEIN_DEPTH: u32 = 6;

/// One generator parameter as the registry lists it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamSpec {
    /// Stable id (`skin.features.<generator>.<name>`).
    pub id: &'static str,
    /// Unit ("mm", "count", "fraction").
    pub unit: &'static str,
    /// Lowest value.
    pub min: f32,
    /// Highest value.
    pub max: f32,
    /// Default.
    pub default: f32,
}

const fn spec(id: &'static str, unit: &'static str, min: f32, max: f32, default: f32) -> ParamSpec {
    ParamSpec { id, unit, min, max, default }
}

/// The generators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GeneratorKind {
    /// Small pheomelanin-rich spots in clusters.
    Freckles,
    /// Larger dark naevi, few, placed by Poisson sampling.
    Moles,
    /// Raised or depressed streaks, paler or pinker than the skin.
    Scars,
    /// Clusters of red papules.
    Acne,
    /// Parallel depressed streaks.
    StretchMarks,
    /// Branching bluish vessels under thin skin.
    Veins,
    /// A user image placed as a colour override.
    Tattoo,
    /// Macro wrinkle lines (crow's feet fan, forehead rows, nasolabial fold).
    Wrinkles,
    /// Soft darkening under the eyes.
    UnderEye,
    /// Solar lentigines: larger irregular brown patches.
    SunSpots,
    /// Vertical fine lines across the lips.
    LipLines,
}

/// Which macro wrinkle pattern [`GeneratorKind::Wrinkles`] draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WrinklePattern {
    /// A fan of lines radiating from the rectangle's inner edge (crow's feet).
    Fan,
    /// Horizontal parallel rows (forehead).
    Rows,
    /// One long curved fold (nasolabial).
    Fold,
}

impl GeneratorKind {
    /// Every generator.
    pub const ALL: [Self; 11] = [
        Self::Freckles,
        Self::Moles,
        Self::Scars,
        Self::Acne,
        Self::StretchMarks,
        Self::Veins,
        Self::Tattoo,
        Self::Wrinkles,
        Self::UnderEye,
        Self::SunSpots,
        Self::LipLines,
    ];

    /// The stable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Freckles => "freckles",
            Self::Moles => "moles",
            Self::Scars => "scars",
            Self::Acne => "acne",
            Self::StretchMarks => "stretch-marks",
            Self::Veins => "veins",
            Self::Tattoo => "tattoo",
            Self::Wrinkles => "wrinkles",
            Self::UnderEye => "under-eye",
            Self::SunSpots => "sun-spots",
            Self::LipLines => "lip-lines",
        }
    }

    /// The generator with a stable name.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }

    /// The parameters, in the order of [`GeneratorParams::values`].
    pub const fn params(self) -> &'static [ParamSpec] {
        match self {
            Self::Freckles => {
                const {
                    &[
                        spec("skin.features.freckles.density", "per-cm2", 0.0, 200.0, 40.0),
                        spec("skin.features.freckles.size", "mm", 0.3, 4.0, 1.2),
                        spec("skin.features.freckles.amount", "fraction", 0.0, 1.0, 0.35),
                        spec("skin.features.freckles.clustering", "fraction", 0.0, 1.0, 0.6),
                    ]
                }
            }
            Self::Moles => {
                const {
                    &[
                        spec("skin.features.moles.count", "count", 0.0, 60.0, 4.0),
                        spec("skin.features.moles.size", "mm", 0.5, 8.0, 2.5),
                        spec("skin.features.moles.darkness", "fraction", 0.0, 1.0, 0.8),
                        spec("skin.features.moles.raised", "mm", 0.0, 1.0, 0.15),
                    ]
                }
            }
            Self::Scars => {
                const {
                    &[
                        spec("skin.features.scars.count", "count", 0.0, 10.0, 1.0),
                        spec("skin.features.scars.length", "mm", 2.0, 120.0, 25.0),
                        spec("skin.features.scars.width", "mm", 0.3, 8.0, 1.5),
                        spec("skin.features.scars.relief", "mm", -0.5, 0.5, 0.1),
                        spec("skin.features.scars.redness", "fraction", -1.0, 1.0, 0.3),
                    ]
                }
            }
            Self::Acne => {
                const {
                    &[
                        spec("skin.features.acne.count", "count", 0.0, 200.0, 20.0),
                        spec("skin.features.acne.size", "mm", 0.5, 5.0, 1.8),
                        spec("skin.features.acne.redness", "fraction", 0.0, 1.0, 0.5),
                        spec("skin.features.acne.clusters", "count", 1.0, 12.0, 3.0),
                    ]
                }
            }
            Self::StretchMarks => {
                const {
                    &[
                        spec("skin.features.stretch_marks.count", "count", 0.0, 40.0, 8.0),
                        spec("skin.features.stretch_marks.length", "mm", 5.0, 150.0, 40.0),
                        spec("skin.features.stretch_marks.width", "mm", 0.5, 10.0, 3.0),
                        spec("skin.features.stretch_marks.angle", "deg", -90.0, 90.0, 20.0),
                        spec("skin.features.stretch_marks.freshness", "fraction", 0.0, 1.0, 0.2),
                    ]
                }
            }
            Self::Veins => {
                const {
                    &[
                        spec("skin.features.veins.count", "count", 0.0, 20.0, 3.0),
                        spec("skin.features.veins.length", "mm", 5.0, 150.0, 40.0),
                        spec("skin.features.veins.width", "mm", 0.2, 4.0, 1.0),
                        spec("skin.features.veins.visibility", "fraction", 0.0, 1.0, 0.4),
                        spec("skin.features.veins.branching", "fraction", 0.0, 1.0, 0.5),
                    ]
                }
            }
            Self::Tattoo => {
                const {
                    &[
                        spec("skin.features.tattoo.center_u", "fraction", 0.0, 1.0, 0.5),
                        spec("skin.features.tattoo.center_v", "fraction", 0.0, 1.0, 0.5),
                        spec("skin.features.tattoo.size", "mm", 5.0, 400.0, 60.0),
                        spec("skin.features.tattoo.rotation", "deg", -180.0, 180.0, 0.0),
                        spec("skin.features.tattoo.opacity", "fraction", 0.0, 1.0, 0.9),
                    ]
                }
            }
            Self::Wrinkles => {
                const {
                    &[
                        spec("skin.features.wrinkles.depth", "mm", 0.0, 1.0, 0.2),
                        spec("skin.features.wrinkles.count", "count", 1.0, 12.0, 4.0),
                        spec("skin.features.wrinkles.width", "mm", 0.2, 4.0, 0.8),
                        spec("skin.features.wrinkles.age", "fraction", 0.0, 1.0, 0.5),
                    ]
                }
            }
            Self::UnderEye => {
                const {
                    &[
                        spec("skin.features.under_eye.darkness", "fraction", 0.0, 1.0, 0.4),
                        spec("skin.features.under_eye.blueness", "fraction", 0.0, 1.0, 0.4),
                        spec("skin.features.under_eye.softness", "fraction", 0.0, 1.0, 0.5),
                    ]
                }
            }
            Self::SunSpots => {
                const {
                    &[
                        spec("skin.features.sun_spots.count", "count", 0.0, 80.0, 10.0),
                        spec("skin.features.sun_spots.size", "mm", 1.0, 20.0, 5.0),
                        spec("skin.features.sun_spots.amount", "fraction", 0.0, 1.0, 0.3),
                    ]
                }
            }
            Self::LipLines => {
                const {
                    &[
                        spec("skin.features.lip_lines.density", "per-mm", 0.0, 2.0, 0.6),
                        spec("skin.features.lip_lines.depth", "mm", 0.0, 0.3, 0.05),
                        spec("skin.features.lip_lines.width", "mm", 0.05, 1.0, 0.2),
                    ]
                }
            }
        }
    }

    /// The layer's blend mode and content attributes.
    pub(super) const fn layout(self) -> (BlendMode, [Option<Attribute>; CONTENT_ATTRIBUTES]) {
        use Attribute as A;
        match self {
            Self::Freckles | Self::SunSpots => {
                (BlendMode::Add, [Some(A::Melanin), Some(A::EumelaninRatio), None, None])
            }
            Self::Moles => (BlendMode::Mix, [Some(A::Melanin), Some(A::EumelaninRatio), Some(A::Height), None]),
            Self::Scars => {
                (BlendMode::Add, [Some(A::Height), Some(A::Haemoglobin), Some(A::Melanin), Some(A::RoughnessPrimary)])
            }
            Self::Acne => (BlendMode::Add, [Some(A::Haemoglobin), Some(A::Height), Some(A::Oxygenation), None]),
            Self::StretchMarks => {
                (BlendMode::Add, [Some(A::Height), Some(A::Melanin), Some(A::Haemoglobin), Some(A::RoughnessPrimary)])
            }
            Self::Veins => {
                (BlendMode::Add, [Some(A::Oxygenation), Some(A::Haemoglobin), Some(A::EpidermalThickness), None])
            }
            Self::Tattoo => (
                BlendMode::Mix,
                [Some(A::OverrideRed), Some(A::OverrideGreen), Some(A::OverrideBlue), Some(A::OverrideCoverage)],
            ),
            Self::Wrinkles | Self::LipLines => (BlendMode::Add, [Some(A::Height), Some(A::Cavity), None, None]),
            Self::UnderEye => {
                (BlendMode::Add, [Some(A::Melanin), Some(A::Oxygenation), Some(A::EpidermalThickness), None])
            }
        }
    }
}

/// A generator's parameter values (in [`GeneratorKind::params`] order).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeneratorParams {
    /// The generator.
    pub kind: GeneratorKind,
    /// The values.
    pub values: [f32; MAX_PARAMS],
}

impl GeneratorParams {
    /// The defaults.
    pub fn new(kind: GeneratorKind) -> Self {
        let mut values = [0.0; MAX_PARAMS];
        for (v, s) in values.iter_mut().zip(kind.params()) {
            *v = s.default;
        }
        Self { kind, values }
    }

    /// Sets a parameter by stable id, clamped to its range; false for an unknown id. Non-finite
    /// values are ignored.
    pub fn set(&mut self, id: &str, value: f32) -> bool {
        let Some(i) = self.kind.params().iter().position(|s| s.id == id) else { return false };
        if value.is_finite() {
            let s = self.kind.params()[i];
            self.values[i] = value.clamp(s.min, s.max);
        }
        true
    }

    /// The value of parameter `i`.
    pub fn get(&self, i: usize) -> f32 {
        self.values.get(i).copied().unwrap_or(0.0)
    }
}

/// Where a generator writes.
#[derive(Clone, Copy, Debug)]
pub struct Target<'a> {
    /// The UV tile.
    pub tile: u32,
    /// The rectangle of the tile.
    pub rect: Rect,
    /// Texel size in mm on this tile.
    pub texel_mm: f32,
    /// Placement weight per texel of `rect` (row-major), 0..1; `None` = 1 everywhere.
    pub mask: Option<&'a [f32]>,
}

/// The working canvas of one generator over its rectangle.
struct Canvas {
    rect: Rect,
    values: Vec<[f32; CONTENT_ATTRIBUTES]>,
    coverage: Vec<f32>,
}

impl Canvas {
    fn new(rect: Rect, additive: bool) -> Result<Self, SkinError> {
        let content = LayerContent::new(0, rect)?;
        let n = content.values.len();
        Ok(Self { rect, values: content.values, coverage: vec![if additive { 1.0 } else { 0.0 }; n] })
    }

    /// Adds `values · profile(d)` around `(cx, cy)` (rect-local texels) with an elliptical
    /// footprint of radii `(rx, ry)` rotated by `angle`; `profile` maps the normalised distance
    /// (1 at the ellipse edge) to a weight. With `mix`, coverage takes the maximum weight and the
    /// values are written where that weight wins.
    fn splat(
        &mut self,
        (cx, cy): (f32, f32),
        (rx, ry): (f32, f32),
        angle: f32,
        values: [f32; CONTENT_ATTRIBUTES],
        mix: bool,
        profile: impl Fn(f32) -> f32,
    ) {
        let reach = rx.max(ry) * 1.5 + 1.0;
        let (w, h) = (self.rect.w as f32, self.rect.h as f32);
        let x0 = (cx - reach).floor().clamp(0.0, w) as u32;
        let x1 = (cx + reach).ceil().clamp(0.0, w) as u32;
        let y0 = (cy - reach).floor().clamp(0.0, h) as u32;
        let y1 = (cy + reach).ceil().clamp(0.0, h) as u32;
        let (s, c) = angle.sin_cos();
        let (rx, ry) = (rx.max(0.25), ry.max(0.25));
        for y in y0..y1 {
            for x in x0..x1 {
                let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                let (u, v) = ((dx * c + dy * s) / rx, (-dx * s + dy * c) / ry);
                let weight = profile((u * u + v * v).sqrt());
                if weight <= 0.0 {
                    continue;
                }
                let i = (y * self.rect.w + x) as usize;
                if mix {
                    if weight > self.coverage[i] {
                        self.coverage[i] = weight;
                        self.values[i] = values;
                    }
                } else {
                    for (dst, v) in self.values[i].iter_mut().zip(values) {
                        *dst += v * weight;
                    }
                }
            }
        }
    }

    /// A stroke along `points` (rect-local texels): capsules of `radius` between consecutive points.
    fn stroke(
        &mut self,
        points: &[(f32, f32)],
        radius: f32,
        values: [f32; CONTENT_ATTRIBUTES],
        profile: impl Fn(f32) -> f32,
    ) {
        // Splat discs at half-radius spacing with the profile; overlapping discs take the maximum
        // so the stroke does not darken where dabs overlap.
        let mut best = vec![0.0_f32; self.values.len()];
        let step = (radius * 0.5).max(0.5);
        for pair in points.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let len = (b.0 - a.0).hypot(b.1 - a.1);
            let n = (len / step).ceil().max(1.0) as u32;
            for k in 0..=n {
                let t = k as f32 / n as f32;
                let (cx, cy) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
                let reach = radius * 1.5 + 1.0;
                let (w, h) = (self.rect.w as f32, self.rect.h as f32);
                for y in (cy - reach).floor().clamp(0.0, h) as u32..(cy + reach).ceil().clamp(0.0, h) as u32 {
                    for x in (cx - reach).floor().clamp(0.0, w) as u32..(cx + reach).ceil().clamp(0.0, w) as u32 {
                        let d = (x as f32 + 0.5 - cx).hypot(y as f32 + 0.5 - cy) / radius.max(0.25);
                        let i = (y * self.rect.w + x) as usize;
                        best[i] = best[i].max(profile(d));
                    }
                }
            }
        }
        for (dst, &w) in self.values.iter_mut().zip(&best) {
            for (d, v) in dst.iter_mut().zip(values) {
                *d += v * w;
            }
        }
    }

    fn into_layer(self, id: LayerId, kind: GeneratorKind, tile: u32) -> Layer {
        let (blend, attributes) = kind.layout();
        let mut layer = Layer::content(id, kind.name(), blend, attributes);
        layer.kind = LayerKind::Content {
            attributes,
            tiles: vec![LayerContent { tile, rect: self.rect, values: self.values, coverage: self.coverage }],
        };
        layer
    }
}

/// A vein branch waiting to be drawn: start, heading, length, radius, depth.
type Branch = ((f32, f32), f32, f32, f32, u32);

/// Smooth bump: 1 at the centre, 0 at distance 1 and beyond.
fn bump(d: f32) -> f32 {
    if d >= 1.0 {
        0.0
    } else {
        let t = 1.0 - d * d;
        t * t
    }
}

/// Groove cross-section: Gaussian with σ = 1/2 of the radius, cut at 1.5.
pub(super) fn groove(d: f32) -> f32 {
    if d >= 1.5 { 0.0 } else { (-2.0 * d * d).exp() }
}

/// Poisson-dart placement of up to `count` points at least `min_dist` texels apart, accepted
/// with the probability of the mask (and `extra(x, y)`), deterministic from `rng`.
fn place(
    rng: &mut Rng,
    rect: Rect,
    mask: Option<&[f32]>,
    count: usize,
    min_dist: f32,
    extra: impl Fn(f32, f32) -> f32,
) -> Vec<(f32, f32)> {
    let mut points: Vec<(f32, f32)> = Vec::with_capacity(count.min(MAX_FEATURES));
    let attempts = count.min(MAX_FEATURES) * 30 + 30;
    for _ in 0..attempts {
        if points.len() >= count.min(MAX_FEATURES) {
            break;
        }
        let (x, y) = (rng.f32() * rect.w as f32, rng.f32() * rect.h as f32);
        let i = (y as u32).min(rect.h - 1) * rect.w + (x as u32).min(rect.w - 1);
        let weight = mask.and_then(|m| m.get(i as usize)).copied().unwrap_or(1.0).clamp(0.0, 1.0) * extra(x, y);
        if rng.f32() >= weight {
            continue;
        }
        if points.iter().any(|p| (p.0 - x).hypot(p.1 - y) < min_dist) {
            continue;
        }
        points.push((x, y));
    }
    points
}

/// A random walk of `segments` steps of `step` texels starting at `start` in direction `angle`,
/// turning by at most `wobble` radians per step.
fn walk(rng: &mut Rng, start: (f32, f32), angle: f32, segments: u32, step: f32, wobble: f32) -> Vec<(f32, f32)> {
    let mut points = Vec::with_capacity(segments as usize + 1);
    let mut p = start;
    let mut a = angle;
    points.push(p);
    for _ in 0..segments {
        a += rng.range_f32(-wobble, wobble);
        p = (p.0 + a.cos() * step, p.1 + a.sin() * step);
        points.push(p);
    }
    points
}

/// Runs a generator and returns its layer (id `id`, named after the generator).
pub fn generate(
    params: &GeneratorParams,
    target: &Target<'_>,
    id: LayerId,
    seed: u64,
    decal: Option<&Image>,
) -> Result<Layer, SkinError> {
    let rect = target.rect;
    if rect.is_empty() {
        return Err(SkinError::Invalid { what: "a generator needs a non-empty rectangle" });
    }
    if !target.texel_mm.is_finite() || target.texel_mm <= 0.0 {
        return Err(SkinError::Invalid { what: "texel size must be positive" });
    }
    if let Some(m) = target.mask
        && m.len() as u64 != rect.area()
    {
        return Err(SkinError::Invalid { what: "placement mask size differs from the rectangle" });
    }
    let kind = params.kind;
    let p = |i: usize| params.get(i);
    let mm = |v: f32| v / target.texel_mm;
    let mut rng = Rng::new(seed);
    let additive = kind.layout().0 != BlendMode::Mix;
    let mut canvas = Canvas::new(rect, additive)?;
    let area_cm2 = rect.area() as f32 * target.texel_mm * target.texel_mm / 100.0;
    let seed32 = seed as u32;
    match kind {
        GeneratorKind::Freckles => {
            let count = (p(0) * area_cm2) as usize;
            let radius = mm(p(1) * 0.5);
            let clustering = p(3);
            let scale = rect.w.max(rect.h) as f32;
            let cluster = |x: f32, y: f32| {
                let n = fbm(x / scale * 4.0, y / scale * 4.0, 64, 3, 0.5, seed32) + 0.5;
                (1.0 - clustering) + clustering * (n * 2.0 - 0.4).clamp(0.0, 1.0)
            };
            for (x, y) in place(&mut rng, rect, target.mask, count, radius * 1.2, cluster) {
                let r = radius * rng.range_f32(0.6, 1.4);
                let stretch = rng.range_f32(0.75, 1.0);
                canvas.splat(
                    (x, y),
                    (r, r * stretch),
                    rng.f32() * std::f32::consts::PI,
                    [p(2) * rng.range_f32(0.5, 1.0), -0.3, 0.0, 0.0],
                    false,
                    bump,
                );
            }
        }
        GeneratorKind::SunSpots => {
            let radius = mm(p(1) * 0.5);
            for (x, y) in place(&mut rng, rect, target.mask, p(0) as usize, radius * 1.5, |_, _| 1.0) {
                // Irregular outline: several overlapping lobes.
                for _ in 0..4 {
                    let (ox, oy) = (rng.range_f32(-0.4, 0.4) * radius, rng.range_f32(-0.4, 0.4) * radius);
                    let r = radius * rng.range_f32(0.5, 0.8);
                    canvas.splat(
                        (x + ox, y + oy),
                        (r, r * rng.range_f32(0.7, 1.0)),
                        rng.f32() * std::f32::consts::PI,
                        [p(2) * 0.4, 0.05, 0.0, 0.0],
                        false,
                        |d| bump(d).sqrt(),
                    );
                }
            }
        }
        GeneratorKind::Moles => {
            let radius = mm(p(1) * 0.5);
            for (x, y) in place(&mut rng, rect, target.mask, p(0) as usize, radius * 6.0, |_, _| 1.0) {
                let r = radius * rng.range_f32(0.6, 1.3);
                let melanin = 0.6 + 0.4 * p(2);
                let profile = |d: f32| if d >= 1.0 { 0.0 } else { (1.0 - d.powi(6)).clamp(0.0, 1.0) };
                canvas.splat(
                    (x, y),
                    (r, r * rng.range_f32(0.85, 1.0)),
                    rng.f32() * std::f32::consts::PI,
                    [melanin, 0.9, p(3), 0.0],
                    true,
                    profile,
                );
            }
        }
        GeneratorKind::Scars => {
            let (length, radius) = (mm(p(1)), mm(p(2) * 0.5));
            for (x, y) in place(&mut rng, rect, target.mask, p(0) as usize, length * 0.3, |_, _| 1.0) {
                let segments = 12;
                let heading = rng.f32() * std::f32::consts::TAU;
                let path = walk(&mut rng, (x, y), heading, segments, length / segments as f32, 0.15);
                let redness = p(4);
                canvas.stroke(&path, radius, [p(3), 0.25 * redness, -0.08, -0.15], bump);
            }
        }
        GeneratorKind::Acne => {
            let radius = mm(p(1) * 0.5);
            let clusters = place(&mut rng, rect, target.mask, p(3) as usize, radius * 10.0, |_, _| 1.0);
            let per = (p(0) as usize).div_ceil(clusters.len().max(1));
            for &(cx, cy) in &clusters {
                for _ in 0..per {
                    let spread = radius * 6.0;
                    let (x, y) = (cx + rng.normal_f32() * spread, cy + rng.normal_f32() * spread);
                    let r = radius * rng.range_f32(0.5, 1.2);
                    let red = p(2) * rng.range_f32(0.5, 1.0);
                    canvas.splat((x, y), (r * 1.8, r * 1.8), 0.0, [red * 0.5, 0.0, -0.1 * red, 0.0], false, bump);
                    canvas.splat((x, y), (r, r), 0.0, [0.0, 0.05 * r * target.texel_mm, 0.0, 0.0], false, bump);
                }
            }
        }
        GeneratorKind::StretchMarks => {
            let (length, width) = (mm(p(1)), mm(p(2)));
            let angle = p(3).to_radians();
            let fresh = p(4);
            for (x, y) in place(&mut rng, rect, target.mask, p(0) as usize, width * 2.0, |_, _| 1.0) {
                let l = length * rng.range_f32(0.6, 1.2);
                let a = angle + rng.range_f32(-0.12, 0.12);
                canvas.splat(
                    (x, y),
                    (l * 0.5, width * 0.5 * rng.range_f32(0.5, 1.0)),
                    a,
                    [-0.03, -0.1 * (1.0 - fresh), 0.3 * fresh, -0.1],
                    false,
                    |d| bump(d).sqrt(),
                );
            }
        }
        GeneratorKind::Veins => {
            let (length, radius) = (mm(p(1)), mm(p(2) * 0.5));
            let vis = p(3);
            let branching = p(4);
            let starts = place(&mut rng, rect, target.mask, p(0) as usize, length * 0.3, |_, _| 1.0);
            // Explicit stack instead of recursion; depth is limited.
            let mut stack: Vec<Branch> =
                starts.iter().map(|&s| (s, rng.f32() * std::f32::consts::TAU, length, radius, 0)).collect();
            let mut strokes = 0;
            while let Some((start, angle, len, r, depth)) = stack.pop() {
                strokes += 1;
                if strokes > MAX_FEATURES {
                    break;
                }
                let segments = 10;
                let path = walk(&mut rng, start, angle, segments, len / segments as f32, 0.25);
                canvas.stroke(&path, r, [-0.5 * vis, 0.15 * vis, -0.15 * vis, 0.0], groove);
                if depth + 1 < MAX_VEIN_DEPTH && r > 0.75 {
                    for &(px, py) in path.iter().skip(3).step_by(3) {
                        if rng.f32() < branching * 0.6 {
                            let side = if rng.chance(0.5) { 0.7 } else { -0.7 };
                            stack.push(((px, py), angle + side, len * 0.55, r * 0.65, depth + 1));
                        }
                    }
                }
            }
        }
        GeneratorKind::Tattoo => {
            let Some(image) = decal else {
                return Err(SkinError::Invalid { what: "the tattoo generator needs the user's image" });
            };
            stamp_decal(
                &mut canvas,
                image,
                (p(0) * rect.w as f32, p(1) * rect.h as f32),
                mm(p(2)),
                p(3).to_radians(),
                p(4),
            )?;
        }
        GeneratorKind::Wrinkles => {
            let age = p(3);
            let depth = -p(0) * (0.3 + 0.7 * age);
            let radius = mm(p(2) * 0.5) * (1.0 + 0.5 * age);
            let count = p(1) as u32;
            let (w, h) = (rect.w as f32, rect.h as f32);
            let pattern = match seed % 3 {
                0 => WrinklePattern::Rows,
                1 => WrinklePattern::Fan,
                _ => WrinklePattern::Fold,
            };
            draw_wrinkles(&mut canvas, &mut rng, pattern, count, (w, h), radius, depth);
        }
        GeneratorKind::UnderEye => {
            let (w, h) = (rect.w as f32, rect.h as f32);
            let soft = 0.3 + 0.7 * p(2);
            let profile = |d: f32| if d >= 1.0 { 0.0 } else { (1.0 - d).powf(soft) };
            canvas.splat(
                (w * 0.5, h * 0.4),
                (w * 0.48, h * 0.4),
                0.0,
                [0.12 * p(0), -0.4 * p(1), -0.2, 0.0],
                false,
                profile,
            );
        }
        GeneratorKind::LipLines => {
            let spacing = 1.0 / p(0).max(0.05);
            let radius = mm(p(2) * 0.5);
            let n = (rect.w as f32 / mm(spacing)).floor() as u32;
            for i in 0..n.min(MAX_FEATURES as u32) {
                let x = (i as f32 + 0.5 + rng.range_f32(-0.3, 0.3)) * mm(spacing);
                let top = rng.range_f32(0.0, 0.3) * rect.h as f32;
                let bottom = rng.range_f32(0.7, 1.0) * rect.h as f32;
                let path = walk(&mut rng, (x, top), std::f32::consts::FRAC_PI_2, 6, (bottom - top) / 6.0, 0.08);
                canvas.stroke(&path, radius, [-p(1) * rng.range_f32(0.5, 1.0), -0.3, 0.0, 0.0], groove);
            }
        }
    }
    // The mask also limits the result (features never spill outside their region).
    if let Some(mask) = target.mask {
        for ((v, c), &m) in canvas.values.iter_mut().zip(canvas.coverage.iter_mut()).zip(mask) {
            let m = m.clamp(0.0, 1.0);
            if additive {
                for x in v.iter_mut() {
                    *x *= m;
                }
            } else {
                *c *= m;
            }
        }
    }
    Ok(canvas.into_layer(id, kind, target.tile))
}

/// Draws a macro wrinkle pattern over a `w × h` canvas.
fn draw_wrinkles(
    canvas: &mut Canvas,
    rng: &mut Rng,
    pattern: WrinklePattern,
    count: u32,
    (w, h): (f32, f32),
    radius: f32,
    depth: f32,
) {
    let values = [depth, -0.4 * depth.abs().min(1.0) * 2.0, 0.0, 0.0];
    match pattern {
        WrinklePattern::Rows => {
            for i in 0..count {
                let y = h * (i as f32 + 1.0) / (count as f32 + 1.0);
                let path = walk(rng, (w * 0.05, y), 0.0, 16, w * 0.9 / 16.0, 0.05);
                canvas.stroke(&path, radius, values, groove);
            }
        }
        WrinklePattern::Fan => {
            for i in 0..count {
                let t = (i as f32 + 0.5) / count as f32;
                let angle = -0.6 + 1.2 * t;
                let path = walk(rng, (w * 0.02, h * 0.5), angle, 10, w * 0.6 / 10.0, 0.06);
                canvas.stroke(&path, radius * (1.0 - 0.3 * (t - 0.5).abs()), values, groove);
            }
        }
        WrinklePattern::Fold => {
            let path: Vec<(f32, f32)> = (0..=16)
                .map(|k| {
                    let t = k as f32 / 16.0;
                    (w * (0.3 + 0.4 * (t * std::f32::consts::FRAC_PI_2).sin()), h * (0.05 + 0.9 * t))
                })
                .collect();
            canvas.stroke(&path, radius * 2.0, values, groove);
        }
    }
}

/// Places a user image (`size` texels wide, rotated) with its alpha as coverage.
fn stamp_decal(
    canvas: &mut Canvas,
    image: &Image,
    centre: (f32, f32),
    size: f32,
    angle: f32,
    opacity: f32,
) -> Result<(), SkinError> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let ch = usize::from(image.channels);
    if iw == 0 || ih == 0 || !(1..=4).contains(&ch) || image.samples.len() != iw * ih * ch {
        return Err(SkinError::Invalid { what: "the decal image is malformed" });
    }
    if !size.is_finite() || size <= 0.0 {
        return Err(SkinError::Invalid { what: "the decal size must be positive" });
    }
    let srgb = image.color_space == ColorSpace::Srgb;
    let texel = |x: usize, y: usize| -> [f32; 4] {
        let base = (y * iw + x) * ch;
        let mut rgba = [0.0, 0.0, 0.0, 1.0];
        for k in 0..ch {
            let value = match &image.samples {
                Samples::U8(s) => {
                    let b = s.get(base + k).copied().unwrap_or(0);
                    if srgb && k < 3 && ch >= 3 { srgb8_to_linear(b) } else { f32::from(b) / 255.0 }
                }
                Samples::U16(s) => f32::from(s.get(base + k).copied().unwrap_or(0)) / 65535.0,
                Samples::F16(s) => humanart_math::f16_bits_to_f32(s.get(base + k).copied().unwrap_or(0)),
                Samples::F32(s) => s.get(base + k).copied().unwrap_or(0.0),
            };
            match (ch, k) {
                (1 | 2, 0) => rgba[..3].fill(value),
                (2, 1) | (4, 3) => rgba[3] = value,
                (_, k) if k < 3 => rgba[k] = value,
                _ => {}
            }
        }
        rgba
    };
    let scale = iw.max(ih) as f32 / size;
    let (s, c) = angle.sin_cos();
    for y in 0..canvas.rect.h {
        for x in 0..canvas.rect.w {
            let (dx, dy) = (x as f32 + 0.5 - centre.0, y as f32 + 0.5 - centre.1);
            let (u, v) = ((dx * c + dy * s) * scale + iw as f32 * 0.5, (-dx * s + dy * c) * scale + ih as f32 * 0.5);
            if u < 0.0 || v < 0.0 || u >= iw as f32 || v >= ih as f32 {
                continue;
            }
            let t = texel(u as usize, v as usize);
            let i = (y * canvas.rect.w + x) as usize;
            let cov = (t[3] * opacity).clamp(0.0, 1.0);
            if cov > canvas.coverage[i] {
                canvas.coverage[i] = cov;
                canvas.values[i] = [t[0].max(0.0), t[1].max(0.0), t[2].max(0.0), 1.0];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::channels::BaseValues;
    use super::super::layers::{LayerStack, TexelInputs, composite_texel};
    use super::super::zones::ZoneTable;
    use super::*;

    fn target(mask: Option<&[f32]>) -> Target<'_> {
        Target { tile: 0, rect: Rect::new(32, 32, 128, 128), texel_mm: 0.25, mask }
    }

    fn content(layer: &Layer) -> &LayerContent {
        match &layer.kind {
            LayerKind::Content { tiles, .. } => &tiles[0],
            LayerKind::Fill { .. } => panic!("generators make content layers"),
        }
    }

    #[test]
    fn oth02_every_generator_runs_and_is_deterministic() {
        let decal = Image::new(
            humanart_io::Format::Png,
            2,
            2,
            4,
            ColorSpace::Srgb,
            Samples::U8(vec![255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 0]),
        )
        .unwrap_or_else(|e| panic!("{e}"));
        for kind in GeneratorKind::ALL {
            assert_eq!(GeneratorKind::from_name(kind.name()), Some(kind));
            let params = GeneratorParams::new(kind);
            let a = generate(&params, &target(None), LayerId(1), 7, Some(&decal))
                .unwrap_or_else(|e| panic!("{kind:?}: {e}"));
            let b = generate(&params, &target(None), LayerId(1), 7, Some(&decal)).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(a, b, "{kind:?} is deterministic");
            let c = content(&a);
            assert!(c.is_consistent());
            assert!(c.values.iter().flatten().all(|v| v.is_finite()), "{kind:?}");
            let nonzero =
                c.values.iter().zip(&c.coverage).filter(|(v, cov)| **cov > 0.0 && v.iter().any(|x| *x != 0.0)).count();
            assert!(nonzero > 0, "{kind:?} drew nothing");
            // Every spec id is unique and well-formed.
            for s in kind.params() {
                assert!(s.id.starts_with("skin.features.") && s.min <= s.default && s.default <= s.max, "{s:?}");
            }
        }
        let mut p = GeneratorParams::new(GeneratorKind::Moles);
        assert!(p.set("skin.features.moles.count", 1000.0));
        assert_eq!(p.get(0), 60.0);
        assert!(!p.set("skin.features.freckles.size", 1.0));
        assert!(generate(&GeneratorParams::new(GeneratorKind::Tattoo), &target(None), LayerId(1), 1, None).is_err());
    }

    #[test]
    fn oth02_placement_follows_the_region_mask() {
        // Left half of the rectangle is the region: moles land only there.
        let mut mask = vec![0.0_f32; 128 * 128];
        for y in 0..128 {
            for x in 0..64 {
                mask[y * 128 + x] = 1.0;
            }
        }
        let mut params = GeneratorParams::new(GeneratorKind::Freckles);
        params.set("skin.features.freckles.density", 150.0);
        let layer = generate(&params, &target(Some(&mask)), LayerId(3), 11, None).unwrap_or_else(|e| panic!("{e}"));
        let c = content(&layer);
        let (mut left, mut right) = (0.0, 0.0);
        for y in 0..128 {
            for x in 0..128 {
                let v = c.values[y * 128 + x][0];
                if x < 64 { left += v } else { right += v }
            }
        }
        assert!(left > 1.0 && right == 0.0, "left {left} right {right}");
    }

    #[test]
    fn oth02_outputs_are_skin_layers() {
        // A generated layer goes on the stack like any other and changes the composite only where
        // it drew: a mole darkens its centre texel.
        let mut stack = LayerStack::new(1, 256).unwrap_or_else(|e| panic!("{e}"));
        let mut params = GeneratorParams::new(GeneratorKind::Moles);
        params.set("skin.features.moles.count", 1.0);
        params.set("skin.features.moles.size", 6.0);
        let layer = generate(&params, &target(None), stack.allocate_id(), 5, None).unwrap_or_else(|e| panic!("{e}"));
        let c = content(&layer).clone();
        stack.push(layer).unwrap_or_else(|e| panic!("{e}"));
        let (i, _) = c.coverage.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap_or((0, &0.0));
        let (x, y) = (c.rect.x + i as u32 % c.rect.w, c.rect.y + i as u32 / c.rect.w);
        let v =
            composite_texel(&stack, &BaseValues::default(), &ZoneTable::default(), &TexelInputs::default(), 0, x, y);
        assert!(v[Attribute::Melanin.index()] > 0.8, "{}", v[Attribute::Melanin.index()]);
        let far = composite_texel(
            &stack,
            &BaseValues::default(),
            &ZoneTable::default(),
            &TexelInputs::default(),
            0,
            250,
            250,
        );
        assert_eq!(far[Attribute::Melanin.index()], 0.3);
    }

    #[test]
    fn generator_input_checks() {
        let p = GeneratorParams::new(GeneratorKind::Acne);
        let bad = Target { tile: 0, rect: Rect::new(0, 0, 0, 5), texel_mm: 0.1, mask: None };
        assert!(generate(&p, &bad, LayerId(1), 1, None).is_err());
        let bad = Target { texel_mm: 0.0, ..target(None) };
        assert!(generate(&p, &bad, LayerId(1), 1, None).is_err());
        let short = [0.0; 3];
        assert!(generate(&p, &target(Some(&short)), LayerId(1), 1, None).is_err());
    }

    /// `[OTH-02]` "All generator parameters in the registry": every [`ParamSpec`] has an entry in
    /// `data/template/params/skin.json` with the same unit, hard range and default, and the registry
    /// has no `skin.features.*` entry the code does not read.
    #[test]
    fn oth02_every_generator_parameter_is_in_the_registry() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/template/params/skin.json");
        let text = std::fs::read(path).unwrap();
        let root = humanart_io::json::parse(&text).unwrap();
        let entries = root.get("parameters").and_then(humanart_io::json::Value::as_array).unwrap();
        let num = |e: &humanart_io::json::Value, key: &str| {
            e.get(key).and_then(humanart_io::json::Value::as_number).unwrap().as_f64() as f32
        };
        let mut registered = 0;
        for e in entries {
            let id = e.get("id").and_then(humanart_io::json::Value::as_str).unwrap();
            if !id.starts_with("skin.features.") {
                continue;
            }
            registered += 1;
            let spec = GeneratorKind::ALL
                .iter()
                .flat_map(|k| k.params())
                .find(|s| s.id == id)
                .unwrap_or_else(|| panic!("{id} is registered but no generator reads it"));
            assert_eq!(e.get("unit").and_then(humanart_io::json::Value::as_str), Some(spec.unit), "{id}");
            assert_eq!(num(e, "min"), spec.min, "{id}");
            assert_eq!(num(e, "max"), spec.max, "{id}");
            assert_eq!(num(e, "default"), spec.default, "{id}");
            let group = e.get("group").and_then(humanart_io::json::Value::as_str).unwrap();
            assert!(group.starts_with("skin/features/"), "{id}: {group}");
        }
        let total: usize = GeneratorKind::ALL.iter().map(|k| k.params().len()).sum();
        assert_eq!(registered, total);
    }
}
