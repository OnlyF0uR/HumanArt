//! Makeup (`todo.md` 1, the quality bar's item 2): eyeliner with a wing, a smudged lower lash line,
//! eyeshadow, lipstick with a lip liner and gloss, blush, contour and highlight, placed on the face
//! from the template's landmarks and drawn into the skin compositor as two content layers.
//!
//! **Placement.** [`Placement::new`] reads the landmarks once: per eye the upper lash line (the
//! `eye.upper` points between the canthi), the lid crease, the lower lash line; the outer lip
//! contour; per cheek the malar point, the zygion, the mouth corner and the jaw line; the nose
//! bridge and the cupid's bow. Each product is a coverage function of the rest position of a skin
//! point (metres), so it follows the face as the UV layout and the identity deform it:
//!
//! - **liner**: a band above the upper lash line, thin at the inner corner and widest at the
//!   outer one (0.8 to 3.0 mm with the width), with a wing that leaves the outer corner up and out
//!   (up to 12 mm) and tapers to a point;
//! - **lower liner**: a soft 1 mm band along the lower lash line, stronger toward the outer corner;
//! - **shadow**: the lid from the lash line, smoked out to about twice the crease's height and past
//!   the outer corner, deepest there;
//! - **lipstick**: the lips zone of the compositor (the vermilion), with a darker liner along the
//!   outer contour and a gloss that lowers the roughness and raises the specular;
//! - **blush**: an elliptical Gaussian on the cheek apple, swept up toward the temple;
//! - **contour**: a soft band under the cheekbone from the ear toward the mouth corner, along the
//!   jaw and down the sides of the nose;
//! - **highlight**: the top of the cheekbone, the nose bridge, the cupid's bow and the chin, a
//!   paler colour with a sheen.
//!
//! **Compositing.** The products are composited over each other per texel (premultiplied,
//! bottom to top: contour, blush, highlight, shadow, lipstick, lip liner, lower liner, liner) into
//! one colour, coverage, roughness and specular. Two layers carry that into the compositor
//! ([`MakeupCanvas::layers`]): the first mixes the colour override fully wherever any makeup is
//! (so the override colour is the makeup's own, not a mix with an unset override), the second
//! mixes the override coverage toward 1, the roughness and the specular by the makeup's coverage.
//! Under makeup a colour override below it (a tattoo) shows only through the makeup's coverage.
//!
//! The shapes, widths and the default colours are artistic (the references of the quality bar:
//! fit women of about 30 with defined liner, glossy lips, blush and contour); nothing is measured.

use humanart_math::Vec3;

use super::channels::Attribute;
use super::dirty::Rect;
use super::error::SkinError;
use super::feature_layers::{region_masks, surface_under_where};
use super::layers::{BlendMode, CONTENT_ATTRIBUTES, Layer, LayerContent, LayerId, LayerKind};
use crate::Template;

/// The UV tile makeup is drawn on.
pub const FACE_TILE: &str = "face";
/// Least makeup coverage a texel keeps (below it the texel is bare).
pub const MIN_COVERAGE: f32 = 1.0 / 512.0;

/// The scalar makeup inputs, in [`Makeup::get`] order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MakeupInput {
    /// Upper eyeliner opacity, 0..1.
    Liner = 0,
    /// Upper eyeliner width, 0..1 (0.8 to 3.0 mm at the outer corner).
    LinerWidth = 1,
    /// Wing length, 0..1 (0 to 12 mm).
    LinerWing = 2,
    /// Lower lash line opacity, 0..1.
    LowerLiner = 3,
    /// Eyeshadow opacity, 0..1.
    Shadow = 4,
    /// Lipstick opacity, 0..1.
    Lips = 5,
    /// Lipstick gloss, 0 (matte) .. 1 (gloss).
    LipsGloss = 6,
    /// Lip liner opacity, 0..1.
    LipLiner = 7,
    /// Blush opacity, 0..1.
    Blush = 8,
    /// Contour opacity, 0..1.
    Contour = 9,
    /// Highlight opacity, 0..1.
    Highlight = 10,
}

/// Scalar makeup inputs.
pub const MAKEUP_INPUTS: usize = 11;

impl MakeupInput {
    /// Every input, in order.
    pub const ALL: [Self; MAKEUP_INPUTS] = [
        Self::Liner,
        Self::LinerWidth,
        Self::LinerWing,
        Self::LowerLiner,
        Self::Shadow,
        Self::Lips,
        Self::LipsGloss,
        Self::LipLiner,
        Self::Blush,
        Self::Contour,
        Self::Highlight,
    ];

    /// The registry material target (`makeup.liner`, ...).
    pub const fn target(self) -> &'static str {
        match self {
            Self::Liner => "makeup.liner",
            Self::LinerWidth => "makeup.liner-width",
            Self::LinerWing => "makeup.liner-wing",
            Self::LowerLiner => "makeup.lower-liner",
            Self::Shadow => "makeup.shadow",
            Self::Lips => "makeup.lips",
            Self::LipsGloss => "makeup.lips-gloss",
            Self::LipLiner => "makeup.lip-liner",
            Self::Blush => "makeup.blush",
            Self::Contour => "makeup.contour",
            Self::Highlight => "makeup.highlight",
        }
    }

    /// The input with a registry material target.
    pub fn from_target(target: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|i| i.target() == target)
    }
}

/// The makeup colours, in [`Makeup::colors`] order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MakeupColor {
    /// Eyeliner and lower lash line.
    Liner = 0,
    /// Eyeshadow.
    Shadow = 1,
    /// Lipstick (the lip liner is a darker shade of it).
    Lips = 2,
    /// Blush.
    Blush = 3,
}

impl MakeupColor {
    /// Every colour, in order.
    pub const ALL: [Self; 4] = [Self::Liner, Self::Shadow, Self::Lips, Self::Blush];

    /// The registry material target (`makeup.liner-color`, ...).
    pub const fn target(self) -> &'static str {
        match self {
            Self::Liner => "makeup.liner-color",
            Self::Shadow => "makeup.shadow-color",
            Self::Lips => "makeup.lips-color",
            Self::Blush => "makeup.blush-color",
        }
    }

    /// The colour with a registry material target.
    pub fn from_target(target: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.target() == target)
    }
}

/// The default colours, linear Rec.709: a soft black liner, a warm brown shadow, a rose lipstick, a
/// peach-pink blush (the registry's defaults).
pub const DEFAULT_COLORS: [[f32; 3]; 4] =
    [[0.012, 0.01, 0.009], [0.16, 0.085, 0.06], [0.36, 0.08, 0.08], [0.6, 0.2, 0.17]];
/// The contour colour, linear: a cool brown about 40 % darker than a medium skin.
pub const CONTOUR_COLOR: [f32; 3] = [0.2, 0.12, 0.095];
/// The highlight colour, linear: a pale champagne.
pub const HIGHLIGHT_COLOR: [f32; 3] = [0.78, 0.63, 0.5];

/// The document's makeup: amounts and shapes ([`MakeupInput`]) and colours ([`MakeupColor`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Makeup {
    /// By [`MakeupInput`], 0..1.
    pub values: [f32; MAKEUP_INPUTS],
    /// By [`MakeupColor`], linear Rec.709, 0..1.
    pub colors: [[f32; 3]; 4],
}

impl Default for Makeup {
    /// Bare skin: every opacity 0, a medium liner width, a short wing, half gloss and
    /// [`DEFAULT_COLORS`].
    fn default() -> Self {
        let mut values = [0.0; MAKEUP_INPUTS];
        values[MakeupInput::LinerWidth as usize] = 0.4;
        values[MakeupInput::LinerWing as usize] = 0.4;
        values[MakeupInput::LipsGloss as usize] = 0.5;
        Self { values, colors: DEFAULT_COLORS }
    }
}

impl Makeup {
    /// One input.
    #[must_use]
    pub fn get(&self, input: MakeupInput) -> f32 {
        self.values[input as usize]
    }

    /// Sets one input (clamped to 0..1; a non-finite value is ignored).
    pub fn set(&mut self, input: MakeupInput, value: f32) {
        if value.is_finite() {
            self.values[input as usize] = value.clamp(0.0, 1.0);
        }
    }

    /// One colour.
    #[must_use]
    pub fn color(&self, color: MakeupColor) -> [f32; 3] {
        self.colors[color as usize]
    }

    /// Sets one colour (channels clamped to 0..1; a non-finite channel leaves it as it is).
    pub fn set_color(&mut self, color: MakeupColor, rgb: [f32; 3]) {
        if rgb.iter().all(|v| v.is_finite()) {
            self.colors[color as usize] = rgb.map(|v| v.clamp(0.0, 1.0));
        }
    }

    /// Whether no product shows (every opacity is 0).
    #[must_use]
    pub fn is_bare(&self) -> bool {
        [
            MakeupInput::Liner,
            MakeupInput::LowerLiner,
            MakeupInput::Shadow,
            MakeupInput::Lips,
            MakeupInput::LipLiner,
            MakeupInput::Blush,
            MakeupInput::Contour,
            MakeupInput::Highlight,
        ]
        .iter()
        .all(|&i| self.get(i) <= 0.0)
    }
}

/// `0` below `e0`, `1` above `e1`, the smooth cubic between.
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = if e1 > e0 { ((x - e0) / (e1 - e0)).clamp(0.0, 1.0) } else { f32::from(x >= e0) };
    t * t * (3.0 - 2.0 * t)
}

/// A polyline with its arc length.
#[derive(Clone, Debug, PartialEq)]
struct Polyline {
    points: Vec<Vec3>,
    /// Arc length at each point (the first 0).
    lengths: Vec<f32>,
}

/// The nearest point of a polyline.
#[derive(Clone, Copy, Debug)]
struct Nearest {
    /// Distance, metres.
    distance: f32,
    /// Arc length fraction, 0 at the first point, 1 at the last.
    s: f32,
    /// The point.
    point: Vec3,
}

impl Polyline {
    /// The polyline through `points`, `None` with fewer than two or a zero length.
    fn new(points: Vec<Vec3>) -> Option<Self> {
        if points.len() < 2 || points.iter().any(|p| !p.is_finite()) {
            return None;
        }
        let mut lengths = Vec::with_capacity(points.len());
        let mut sum = 0.0;
        lengths.push(0.0);
        for pair in points.windows(2) {
            sum += pair[0].distance(pair[1]);
            lengths.push(sum);
        }
        (sum > 1e-6).then_some(Self { points, lengths })
    }

    fn total(&self) -> f32 {
        self.lengths.last().copied().unwrap_or(0.0)
    }

    /// The nearest point to `p`.
    fn nearest(&self, p: Vec3) -> Nearest {
        let mut best = Nearest { distance: f32::INFINITY, s: 0.0, point: self.points[0] };
        let total = self.total();
        for (i, pair) in self.points.windows(2).enumerate() {
            let (a, b) = (pair[0], pair[1]);
            let ab = b - a;
            let len2 = ab.length_squared();
            let t = if len2 > 0.0 { ((p - a).dot(ab) / len2).clamp(0.0, 1.0) } else { 0.0 };
            let q = a + ab * t;
            let d = p.distance(q);
            if d < best.distance {
                let at = self.lengths[i] + t * len2.sqrt();
                best = Nearest { distance: d, s: at / total, point: q };
            }
        }
        best
    }

    /// The point at arc length fraction `s` (clamped).
    fn at(&self, s: f32) -> Vec3 {
        let target = s.clamp(0.0, 1.0) * self.total();
        for (i, pair) in self.points.windows(2).enumerate() {
            let (l0, l1) = (self.lengths[i], self.lengths[i + 1]);
            if target <= l1 || i + 2 == self.points.len() {
                let t = if l1 > l0 { ((target - l0) / (l1 - l0)).clamp(0.0, 1.0) } else { 0.0 };
                return pair[0].lerp(pair[1], t);
            }
        }
        self.points[self.points.len() - 1]
    }
}

/// One eye's lines.
#[derive(Clone, Debug, PartialEq)]
struct Eye {
    /// Upper lash line, medial canthus to lateral canthus.
    lash: Polyline,
    /// Lid crease, medial to lateral.
    crease: Polyline,
    /// Lower lash line, medial to lateral.
    lower: Polyline,
    /// The lateral canthus.
    lateral: Vec3,
    /// Unit direction from the medial to the lateral canthus.
    outward: Vec3,
}

/// One cheek's points.
#[derive(Clone, Debug, PartialEq)]
struct Cheek {
    malar: Vec3,
    zygion: Vec3,
    mouth_corner: Vec3,
    /// The jaw line, `None` without its landmarks.
    jaw: Option<Polyline>,
    /// The side of the nose: from beside the root to the alar crease.
    nose_side: Option<Polyline>,
}

/// Where the products go on a template (see the module documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    eyes: Vec<Eye>,
    cheeks: Vec<Cheek>,
    /// The outer lip contour, closed.
    lips: Option<Polyline>,
    bridge: Option<Polyline>,
    cupid: Option<Vec3>,
    chin: Option<Vec3>,
}

/// Millimetres in metres.
const MM: f32 = 1e-3;

impl Placement {
    /// Reads the landmarks of `template` (rest positions). A product whose landmarks are missing
    /// is left out.
    #[must_use]
    pub fn new(template: &Template) -> Self {
        let positions = template.mesh().positions();
        let find =
            |name: &str| template.landmarks().iter().find(|l| l.name == name).and_then(|l| l.position(positions));
        let run = |prefix: &str, count: usize| -> Option<Vec<Vec3>> {
            (1..=count).map(|i| find(&format!("{prefix}.{i:02}"))).collect()
        };
        // Orders `points` to start near `from`.
        let from_near = |mut points: Vec<Vec3>, from: Vec3| {
            if let (Some(first), Some(last)) = (points.first(), points.last())
                && first.distance(from) > last.distance(from)
            {
                points.reverse();
            }
            points
        };
        let mut eyes = Vec::new();
        let mut cheeks = Vec::new();
        for side in ["L", "R"] {
            let canthi = (find(&format!("{side}.eye.canthus.medial")), find(&format!("{side}.eye.canthus.lateral")));
            if let (Some(medial), Some(lateral)) = canthi {
                let line = |name: &str, ends: bool| {
                    let points = from_near(run(&format!("{side}.eye.{name}"), 7)?, medial);
                    let points = if ends { [vec![medial], points, vec![lateral]].concat() } else { points };
                    Polyline::new(points)
                };
                if let (Some(lash), Some(crease), Some(lower)) =
                    (line("upper", true), line("crease", false), line("lower", true))
                {
                    let outward = (lateral - medial).normalize_or(Vec3::X);
                    eyes.push(Eye { lash, crease, lower, lateral, outward });
                }
            }
            let points = (
                find(&format!("{side}.cheek.malar")),
                find(&format!("{side}.zygion")),
                find(&format!("{side}.lip.corner")),
            );
            if let (Some(malar), Some(zygion), Some(mouth_corner)) = points {
                let jaw = run(&format!("jaw.{side}"), 5).and_then(Polyline::new);
                let nose_side = match (
                    find("nose.root"),
                    find(&format!("{side}.eye.canthus.medial")),
                    find(&format!("{side}.nose.alar-crease")),
                ) {
                    (Some(root), Some(medial), Some(crease)) => {
                        Polyline::new(vec![root.lerp(medial, 0.45), crease.lerp(root.lerp(medial, 0.45), 0.25)])
                    }
                    _ => None,
                };
                cheeks.push(Cheek { malar, zygion, mouth_corner, jaw, nose_side });
            }
        }
        let lips = match (run("lip.upper.outer", 11), run("lip.lower.outer", 11)) {
            (Some(upper), Some(lower)) => {
                let end = upper.last().copied().unwrap_or(Vec3::ZERO);
                let mut lower = from_near(lower, end);
                lower.extend(upper.first().copied());
                Polyline::new([upper, lower].concat())
            }
            _ => None,
        };
        let bridge = match (find("nose.bridge.01"), find("nose.bridge.02"), find("nose.tip")) {
            (Some(a), Some(b), Some(tip)) => Polyline::new(vec![a, b, b.lerp(tip, 0.7)]),
            _ => None,
        };
        let cupid = find("lip.upper.outer.06").map(|p| p + Vec3::Y * (1.5 * MM));
        let chin = find("chin.mid");
        Self { eyes, cheeks, lips, bridge, cupid, chin }
    }

    /// Products that can be placed: (eyes, cheeks, lip contour).
    #[must_use]
    pub fn counts(&self) -> (usize, usize, bool) {
        (self.eyes.len(), self.cheeks.len(), self.lips.is_some())
    }
}

/// One texel's makeup.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MakeupTexel {
    /// Colour (straight, not premultiplied), linear.
    pub rgb: [f32; 3],
    /// Coverage, 0..1.
    pub alpha: f32,
    /// Primary roughness of the makeup.
    pub roughness: f32,
    /// Specular scale of the makeup (1: skin's).
    pub specular: f32,
}

/// The premultiplied running composite.
#[derive(Clone, Copy, Default)]
struct Over {
    rgb: [f32; 3],
    alpha: f32,
    roughness: f32,
    specular: f32,
}

impl Over {
    /// Lays `color` at coverage `a` with a surface of `roughness` and `specular` over what is there.
    fn lay(&mut self, color: [f32; 3], a: f32, roughness: f32, specular: f32) {
        let a = a.clamp(0.0, 1.0);
        if a <= 0.0 {
            return;
        }
        let keep = 1.0 - a;
        for (c, v) in self.rgb.iter_mut().zip(color) {
            *c = v * a + *c * keep;
        }
        self.roughness = roughness * a + self.roughness * keep;
        self.specular = specular * a + self.specular * keep;
        self.alpha = a + self.alpha * keep;
    }

    fn straight(self) -> MakeupTexel {
        if self.alpha <= 0.0 {
            return MakeupTexel::default();
        }
        let inv = 1.0 / self.alpha;
        MakeupTexel {
            rgb: self.rgb.map(|c| (c * inv).clamp(0.0, 1.0)),
            alpha: self.alpha.clamp(0.0, 1.0),
            roughness: (self.roughness * inv).clamp(0.0, 1.0),
            specular: (self.specular * inv).clamp(0.0, 4.0),
        }
    }
}

/// Roughness and specular of powder products (matte).
const POWDER: (f32, f32) = (0.55, 0.7);
/// Roughness and specular of the liner (satin).
const LINER: (f32, f32) = (0.35, 0.9);
/// Roughness and specular of the highlight (a sheen).
const SHEEN: (f32, f32) = (0.24, 1.8);
/// Roughness of matte and glossy lipstick, and their specular.
const LIPSTICK: [(f32, f32); 2] = [(0.5, 0.8), (0.1, 1.6)];
/// Widest the upper liner gets at the outer corner, at width 0 and 1.
const LINER_WIDTH_MM: (f32, f32) = (0.8, 3.0);
/// Longest wing, millimetres.
const WING_MM: f32 = 12.0;
/// Half the soft edge of a liner, metres.
const LINER_EDGE: f32 = 0.35 * MM;

/// A gaussian band of half-width `sigma` around a polyline, `taper` thinning it toward the end.
fn band(line: &Polyline, p: Vec3, sigma: f32, taper: f32) -> f32 {
    let n = line.nearest(p);
    let s = sigma * (1.0 - taper * n.s);
    if s <= 0.0 {
        return 0.0;
    }
    (-0.5 * (n.distance / s).powi(2)).exp()
}

/// The makeup at one rest position `p` with the lips zone weight `lips` (0..1) for `makeup` and
/// the wings `wings` (one per eye, from [`MakeupCanvas`]).
fn sample(placement: &Placement, wings: &[Option<Polyline>], makeup: &Makeup, p: Vec3, lips: f32) -> MakeupTexel {
    let mut over = Over::default();
    let v = |i: MakeupInput| makeup.get(i);
    let up = Vec3::Y;
    // Face products, bottom first.
    for cheek in &placement.cheeks {
        let contour = v(MakeupInput::Contour);
        if contour > 0.0 {
            let start = cheek.zygion - up * (12.0 * MM);
            let line = Polyline::new(vec![start, start.lerp(cheek.mouth_corner, 0.58)]);
            let mut w = line.map_or(0.0, |l| band(&l, p, 7.0 * MM, 0.5));
            if let Some(jaw) = &cheek.jaw {
                w = w.max(0.7 * band(jaw, p, 6.0 * MM, 0.3));
            }
            if let Some(side) = &cheek.nose_side {
                w = w.max(0.45 * band(side, p, 2.6 * MM, 0.0));
            }
            over.lay(CONTOUR_COLOR, 0.38 * contour * w, POWDER.0, POWDER.1);
        }
        let blush = v(MakeupInput::Blush);
        if blush > 0.0 {
            let span = cheek.zygion - cheek.malar;
            let len = span.length();
            if len > 1e-4 {
                let centre = cheek.malar + span * 0.3 + up * (2.0 * MM);
                let axis = (span + up * (0.35 * len)).normalize_or(Vec3::X);
                let d = p - centre;
                let along = d.dot(axis);
                let across = (d - axis * along).length();
                let (sa, sr) = (0.5 * len, 0.3 * len);
                let w = (-0.5 * ((along / sa).powi(2) + (across / sr).powi(2))).exp();
                over.lay(makeup.color(MakeupColor::Blush), 0.5 * blush * w, POWDER.0, POWDER.1);
            }
        }
        let highlight = v(MakeupInput::Highlight);
        if highlight > 0.0 {
            let a = cheek.malar + up * (7.0 * MM);
            let b = cheek.malar.lerp(cheek.zygion, 0.75) + up * (6.0 * MM);
            let w = Polyline::new(vec![a, b]).map_or(0.0, |l| band(&l, p, 4.0 * MM, 0.4));
            over.lay(HIGHLIGHT_COLOR, 0.3 * highlight * w, SHEEN.0, SHEEN.1);
        }
    }
    let highlight = v(MakeupInput::Highlight);
    if highlight > 0.0 {
        let mut w: f32 = 0.0;
        if let Some(bridge) = &placement.bridge {
            w = w.max(band(bridge, p, 1.8 * MM, 0.3));
        }
        if let Some(c) = placement.cupid {
            w = w.max((-0.5 * (p.distance(c) / (1.8 * MM)).powi(2)).exp());
        }
        if let Some(c) = placement.chin {
            w = w.max(0.7 * (-0.5 * (p.distance(c) / (4.5 * MM)).powi(2)).exp());
        }
        over.lay(HIGHLIGHT_COLOR, 0.3 * highlight * w, SHEEN.0, SHEEN.1);
    }
    // Eyes.
    for (eye, wing) in placement.eyes.iter().zip(wings) {
        let n = eye.lash.nearest(p);
        if n.distance > 25.0 * MM {
            continue;
        }
        let crease_at = eye.crease.at(n.s);
        let lid = crease_at - n.point;
        let lid_height = lid.length();
        let lid_up = lid.normalize_or(up);
        let height = (p - n.point).dot(lid_up);
        let shadow = v(MakeupInput::Shadow);
        if shadow > 0.0 && lid_height > 1e-4 {
            let u = height / lid_height;
            let off = (p - n.point - lid_up * height).length();
            // Smoked past the crease and out past the outer corner, deepest there.
            let w = smoothstep(-0.05, 0.05, u)
                * (1.0 - smoothstep(0.9, 2.0, u))
                * (1.0 - smoothstep(3.0 * MM, 11.0 * MM, off))
                * (0.5 + 0.5 * smoothstep(0.2, 0.95, n.s));
            over.lay(makeup.color(MakeupColor::Shadow), 0.9 * shadow * w, POWDER.0, POWDER.1);
        }
        let lower = v(MakeupInput::LowerLiner);
        if lower > 0.0 {
            let m = eye.lower.nearest(p);
            let width = (0.7 + 0.6 * m.s) * MM;
            let w = 1.0 - smoothstep(0.3 * width, width + 0.6 * MM, m.distance);
            over.lay(makeup.color(MakeupColor::Liner), 0.8 * lower * w * (0.5 + 0.5 * m.s), LINER.0, LINER.1);
        }
        let liner = v(MakeupInput::Liner);
        if liner > 0.0 {
            let k = v(MakeupInput::LinerWidth);
            let outer = (LINER_WIDTH_MM.0 + (LINER_WIDTH_MM.1 - LINER_WIDTH_MM.0) * k) * MM;
            let width = outer * (0.3 + 0.7 * smoothstep(0.1, 1.0, n.s));
            let mut w = if height > -0.4 * MM {
                1.0 - smoothstep(width - LINER_EDGE, width + LINER_EDGE, n.distance)
            } else {
                0.0
            };
            if let Some(wing) = wing {
                let m = wing.nearest(p);
                let wide = outer * (1.0 - m.s);
                w = w.max(1.0 - smoothstep(wide - LINER_EDGE, wide + LINER_EDGE, m.distance));
            }
            over.lay(makeup.color(MakeupColor::Liner), liner * w, LINER.0, LINER.1);
        }
    }
    // Lips last: lipstick and its liner.
    let lipstick = v(MakeupInput::Lips);
    if lipstick > 0.0 && lips > 0.0 {
        let g = v(MakeupInput::LipsGloss);
        let rough = LIPSTICK[0].0 + (LIPSTICK[1].0 - LIPSTICK[0].0) * g;
        let spec = LIPSTICK[0].1 + (LIPSTICK[1].1 - LIPSTICK[0].1) * g;
        over.lay(makeup.color(MakeupColor::Lips), 0.92 * lipstick * lips, rough, spec);
    }
    let lip_liner = v(MakeupInput::LipLiner);
    if lip_liner > 0.0
        && let Some(outline) = &placement.lips
    {
        let d = outline.nearest(p).distance;
        let w = 1.0 - smoothstep(0.3 * MM, 1.0 * MM, d);
        let shade = makeup.color(MakeupColor::Lips).map(|c| c * 0.6);
        over.lay(shade, 0.7 * lip_liner * w, LIPSTICK[0].0, LIPSTICK[0].1);
    }
    over.straight()
}

/// The rest surface of one tile under the makeup.
#[derive(Clone, Debug)]
struct TileCanvas {
    tile: u32,
    rect: Rect,
    /// Rest position per texel of `rect` (row-major), `None` where no skin is.
    positions: Vec<Option<Vec3>>,
    /// The lips zone weight per texel of `rect`.
    lips: Vec<f32>,
}

impl TileCanvas {
    fn index(&self, x: u32, y: u32) -> Option<usize> {
        let r = self.rect;
        if x < r.x || y < r.y || x >= r.right() || y >= r.bottom() {
            return None;
        }
        Some((y - r.y) as usize * r.w as usize + (x - r.x) as usize)
    }
}

/// The rest surface of the skin of the face and lids ([`MAKEUP_TILES`]) with the placement, made
/// once per template and reused for every makeup change ([`MakeupCanvas::layers`]).
#[derive(Clone, Debug)]
pub struct MakeupCanvas {
    placement: Placement,
    tiles: Vec<TileCanvas>,
}

/// The UV tiles makeup is drawn on: the face and neck, and the lids (the eye sockets lie on the
/// eyes tile).
pub const MAKEUP_TILES: [&str; 2] = [FACE_TILE, "eyes"];
/// The mesh part makeup is drawn on (not the eyeballs, tear lines or the mouth interior).
pub const SKIN_PART: &str = "body";

impl MakeupCanvas {
    /// Rasterises the rest positions of the skin (the [`SKIN_PART`]) on `template`'s
    /// [`MAKEUP_TILES`] at `tile_size`² texels; `lips` gives the lips zone weight of a texel
    /// (`tile`, `x`, `y`). `None` when the template has none of the tiles or they cover no skin.
    ///
    /// # Errors
    /// [`SkinError::Invalid`] when the template's tiles or region map are unusable.
    pub fn new(
        template: &Template,
        tile_size: u32,
        lips: impl Fn(u32, u32, u32) -> f32,
    ) -> Result<Option<Self>, SkinError> {
        let wanted: Vec<u32> = template
            .tiles()
            .iter()
            .enumerate()
            .filter(|(_, t)| MAKEUP_TILES.contains(&t.name.as_str()))
            .filter_map(|(i, _)| u32::try_from(i).ok())
            .collect();
        if wanted.is_empty() {
            return Ok(None);
        }
        let mesh = template.mesh();
        let skin = mesh.parts().iter().find(|p| p.name == SKIN_PART).map(|p| p.first_face..p.first_face + p.face_count);
        let triangles = mesh.triangulate();
        let mut tiles = Vec::new();
        for mask in region_masks(template, None, tile_size)? {
            if !wanted.contains(&mask.tile) {
                continue;
            }
            let keep = |face: u32| skin.as_ref().is_none_or(|r| r.contains(&face));
            let surface = surface_under_where(mesh, &triangles, &mask, tile_size, keep);
            if !surface.covered.iter().any(|&c| c) {
                continue;
            }
            let r = mask.rect;
            let positions = surface
                .texels
                .iter()
                .zip(&surface.covered)
                .map(|(t, &c)| c.then(|| Vec3::new(t[0], t[1], t[2])))
                .collect();
            let tile = mask.tile;
            let lips = (r.y..r.bottom())
                .flat_map(|y| (r.x..r.right()).map(move |x| (x, y)))
                .map(|(x, y)| lips(tile, x, y))
                .collect();
            tiles.push(TileCanvas { tile, rect: r, positions, lips });
        }
        if tiles.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self { placement: Placement::new(template), tiles }))
    }

    /// The placement.
    #[must_use]
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// The tiles the canvas covers.
    pub fn tiles(&self) -> impl Iterator<Item = u32> + '_ {
        self.tiles.iter().map(|t| t.tile)
    }

    /// The skin position nearest to `p` (the wing tip snapped onto the surface).
    fn snap(&self, p: Vec3) -> Vec3 {
        self.tiles
            .iter()
            .flat_map(|t| t.positions.iter().flatten().copied())
            .min_by(|a, b| a.distance_squared(p).total_cmp(&b.distance_squared(p)))
            .unwrap_or(p)
    }

    /// The wing of each eye for `makeup`, on the surface (`None` without a wing).
    fn wings(&self, makeup: &Makeup) -> Vec<Option<Polyline>> {
        let length = WING_MM * MM * makeup.get(MakeupInput::LinerWing);
        self.placement
            .eyes
            .iter()
            .map(|eye| {
                if length <= 0.5 * MM || makeup.get(MakeupInput::Liner) <= 0.0 {
                    return None;
                }
                let dir = (eye.outward * 0.85 + Vec3::Y * 0.55).normalize_or(eye.outward);
                let mid = self.snap(eye.lateral + dir * (0.5 * length));
                let tip = self.snap(eye.lateral + dir * length);
                Polyline::new(vec![eye.lateral, mid, tip])
            })
            .collect()
    }

    /// The makeup of the texel `(x, y)` of `tile` (bare outside the canvas).
    #[must_use]
    pub fn texel(&self, makeup: &Makeup, tile: u32, x: u32, y: u32) -> MakeupTexel {
        let wings = self.wings(makeup);
        self.tiles
            .iter()
            .find(|t| t.tile == tile)
            .map_or_else(MakeupTexel::default, |t| self.texel_with(t, &wings, makeup, x, y))
    }

    fn texel_with(&self, t: &TileCanvas, wings: &[Option<Polyline>], makeup: &Makeup, x: u32, y: u32) -> MakeupTexel {
        let Some(i) = t.index(x, y) else { return MakeupTexel::default() };
        match t.positions.get(i).copied().flatten() {
            Some(p) => sample(&self.placement, wings, makeup, p, t.lips.get(i).copied().unwrap_or(0.0)),
            None => MakeupTexel::default(),
        }
    }

    /// The two compositor layers of `makeup` (see the module documentation), with ids `ids`, one
    /// rectangle per tile the makeup touches; `None` when it covers no texel.
    ///
    /// # Errors
    /// [`SkinError::Limit`] when a covered rectangle is larger than a content layer holds.
    pub fn layers(&self, makeup: &Makeup, ids: [LayerId; 2]) -> Result<Option<[Layer; 2]>, SkinError> {
        if makeup.is_bare() {
            return Ok(None);
        }
        let wings = self.wings(makeup);
        let colour_attributes: [Option<Attribute>; CONTENT_ATTRIBUTES] =
            [Some(Attribute::OverrideRed), Some(Attribute::OverrideGreen), Some(Attribute::OverrideBlue), None];
        let surface_attributes: [Option<Attribute>; CONTENT_ATTRIBUTES] =
            [Some(Attribute::OverrideCoverage), Some(Attribute::RoughnessPrimary), Some(Attribute::Specular), None];
        let mut a = Layer::content(ids[0], "makeup colour", BlendMode::Mix, colour_attributes);
        let mut b = Layer::content(ids[1], "makeup", BlendMode::Mix, surface_attributes);
        let mut any = false;
        for tile in &self.tiles {
            let r = tile.rect;
            let mut texels = Vec::with_capacity(r.area() as usize);
            let (mut lo, mut hi) = ((u32::MAX, u32::MAX), (0, 0));
            for y in r.y..r.bottom() {
                for x in r.x..r.right() {
                    let t = self.texel_with(tile, &wings, makeup, x, y);
                    if t.alpha >= MIN_COVERAGE {
                        lo = (lo.0.min(x), lo.1.min(y));
                        hi = (hi.0.max(x), hi.1.max(y));
                    }
                    texels.push(t);
                }
            }
            if lo.0 == u32::MAX || hi.0 < lo.0 || hi.1 < lo.1 {
                continue;
            }
            let used = Rect::new(lo.0, lo.1, hi.0 - lo.0 + 1, hi.1 - lo.1 + 1);
            let mut color = LayerContent::new(tile.tile, used)?;
            let mut surface = LayerContent::new(tile.tile, used)?;
            for y in used.y..used.bottom() {
                for x in used.x..used.right() {
                    let (Some(from), Some(to)) = (tile.index(x, y), color.index(x, y)) else { continue };
                    let Some(t) = texels.get(from).filter(|t| t.alpha >= MIN_COVERAGE) else { continue };
                    if let (Some(v), Some(c)) = (color.values.get_mut(to), color.coverage.get_mut(to)) {
                        *v = [t.rgb[0], t.rgb[1], t.rgb[2], 0.0];
                        *c = 1.0;
                    }
                    if let (Some(v), Some(c)) = (surface.values.get_mut(to), surface.coverage.get_mut(to)) {
                        *v = [1.0, t.roughness, t.specular, 0.0];
                        *c = t.alpha;
                    }
                }
            }
            if let LayerKind::Content { tiles, .. } = &mut a.kind {
                tiles.push(color);
            }
            if let LayerKind::Content { tiles, .. } = &mut b.kind {
                tiles.push(surface);
            }
            any = true;
        }
        Ok(any.then_some([a, b]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(points: &[[f32; 3]]) -> Polyline {
        Polyline::new(points.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect()).unwrap()
    }

    #[test]
    fn polyline_nearest_and_at() {
        let l = line(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]);
        let n = l.nearest(Vec3::new(0.5, 0.2, 0.0));
        assert!((n.distance - 0.2).abs() < 1e-6 && (n.s - 0.25).abs() < 1e-6);
        let n = l.nearest(Vec3::new(2.0, 0.5, 0.0));
        assert!((n.s - 0.75).abs() < 1e-6 && (n.distance - 1.0).abs() < 1e-6);
        assert!(l.at(0.75).distance(Vec3::new(1.0, 0.5, 0.0)) < 1e-6);
        assert!(l.at(2.0).distance(Vec3::new(1.0, 1.0, 0.0)) < 1e-6);
        assert!(Polyline::new(vec![Vec3::ZERO]).is_none());
        assert!(Polyline::new(vec![Vec3::ZERO, Vec3::ZERO]).is_none());
    }

    #[test]
    fn over_composites_premultiplied() {
        let mut o = Over::default();
        o.lay([1.0, 0.0, 0.0], 0.5, 0.2, 1.0);
        o.lay([0.0, 0.0, 1.0], 0.5, 0.6, 2.0);
        let t = o.straight();
        assert!((t.alpha - 0.75).abs() < 1e-6);
        // Red 0.25 and blue 0.5 of the premultiplied sum, over 0.75.
        assert!((t.rgb[0] - 1.0 / 3.0).abs() < 1e-5 && (t.rgb[2] - 2.0 / 3.0).abs() < 1e-5);
        assert!((t.roughness - (0.1 * 0.5 + 0.3) / 0.75).abs() < 1e-5);
        assert_eq!(Over::default().straight(), MakeupTexel::default());
    }

    #[test]
    fn makeup_inputs_and_colors_round_trip_their_targets() {
        for i in MakeupInput::ALL {
            assert_eq!(MakeupInput::from_target(i.target()), Some(i));
        }
        for c in MakeupColor::ALL {
            assert_eq!(MakeupColor::from_target(c.target()), Some(c));
        }
        let mut m = Makeup::default();
        assert!(m.is_bare());
        m.set(MakeupInput::Liner, 7.0);
        assert_eq!(m.get(MakeupInput::Liner), 1.0);
        m.set(MakeupInput::Liner, f32::NAN);
        assert_eq!(m.get(MakeupInput::Liner), 1.0);
        assert!(!m.is_bare());
        m.set_color(MakeupColor::Lips, [2.0, 0.1, -1.0]);
        assert_eq!(m.color(MakeupColor::Lips), [1.0, 0.1, 0.0]);
        m.set_color(MakeupColor::Lips, [f32::NAN, 0.0, 0.0]);
        assert_eq!(m.color(MakeupColor::Lips), [1.0, 0.1, 0.0]);
    }

    /// A synthetic eye looking down +Z: the lash line along x at y 0, the crease 8 mm above.
    fn synthetic() -> Placement {
        let lash = line(&[[0.0, 0.0, 0.0], [0.015, 0.002, 0.0], [0.03, 0.0, 0.0]]);
        let crease = line(&[[0.0, 0.008, 0.0], [0.015, 0.01, 0.0], [0.03, 0.008, 0.0]]);
        let lower = line(&[[0.0, 0.0, 0.0], [0.015, -0.009, 0.0], [0.03, 0.0, 0.0]]);
        let eye = Eye { lash, crease, lower, lateral: Vec3::new(0.03, 0.0, 0.0), outward: Vec3::X };
        Placement { eyes: vec![eye], cheeks: Vec::new(), lips: None, bridge: None, cupid: None, chin: None }
    }

    #[test]
    fn todo_1_liner_is_widest_at_the_outer_corner_and_sits_above_the_lashes() {
        let p = synthetic();
        let mut m = Makeup::default();
        m.set(MakeupInput::Liner, 1.0);
        m.set(MakeupInput::LinerWidth, 0.5);
        let at = |x: f32, y: f32| sample(&p, &[None], &m, Vec3::new(x, y, 0.0), 0.0).alpha;
        // On the line: covered everywhere.
        assert!(at(0.0075, 0.001) > 0.99 && at(0.0225, 0.001) > 0.99);
        // 1.2 mm above: the inner corner's thin line has ended, the outer corner's band holds.
        assert!(at(0.003, 0.0012) < 0.05, "{}", at(0.003, 0.0012));
        assert!(at(0.028, 0.0012) > 0.95, "{}", at(0.028, 0.0012));
        // Well below the lash line nothing (the lid margin and the eye).
        assert!(at(0.015, -0.002) < 0.01);
        // The colour is the liner's.
        let t = sample(&p, &[None], &m, Vec3::new(0.02, 0.002, 0.0), 0.0);
        assert!(t.rgb.iter().zip(DEFAULT_COLORS[0]).all(|(a, b)| (a - b).abs() < 1e-5));
        // A wing adds coverage past the outer corner, tapering to its tip.
        let wing = line(&[[0.03, 0.0, 0.0], [0.035, 0.003, 0.0], [0.04, 0.006, 0.0]]);
        let with = |x: f32, y: f32| sample(&p, &[Some(wing.clone())], &m, Vec3::new(x, y, 0.0), 0.0).alpha;
        assert!(at(0.034, 0.0026) < 0.01 && with(0.034, 0.0026) > 0.99);
        assert!(with(0.0399, 0.0068) < 0.5, "the tip is a point");
    }

    #[test]
    fn todo_1_shadow_fills_the_lid_to_the_crease_and_fades_above() {
        let p = synthetic();
        let mut m = Makeup::default();
        m.set(MakeupInput::Shadow, 1.0);
        let at = |x: f32, y: f32| sample(&p, &[None], &m, Vec3::new(x, y, 0.0), 0.0).alpha;
        let (lid, above, below) = (at(0.025, 0.004), at(0.025, 0.016), at(0.025, -0.003));
        assert!(lid > 0.6, "{lid}");
        assert!(above < 0.05, "{above}");
        assert!(below < 0.01, "{below}");
        // Deeper at the outer corner than the inner.
        assert!(at(0.026, 0.004) > at(0.004, 0.004) + 0.1);
    }

    #[test]
    fn todo_1_lipstick_follows_the_lips_zone_with_gloss() {
        let p = synthetic();
        let mut m = Makeup::default();
        m.set(MakeupInput::Lips, 1.0);
        let far = Vec3::new(0.0, -0.06, 0.0);
        assert_eq!(sample(&p, &[None], &m, far, 0.0).alpha, 0.0);
        let matte = {
            m.set(MakeupInput::LipsGloss, 0.0);
            sample(&p, &[None], &m, far, 1.0)
        };
        m.set(MakeupInput::LipsGloss, 1.0);
        let gloss = sample(&p, &[None], &m, far, 1.0);
        assert!((matte.alpha - 0.92).abs() < 1e-6);
        assert!(matte.rgb.iter().zip(DEFAULT_COLORS[2]).all(|(a, b)| (a - b).abs() < 1e-5));
        assert!(gloss.roughness < matte.roughness - 0.3 && gloss.specular > matte.specular);
        // Half the zone, half the coverage.
        assert!((sample(&p, &[None], &m, far, 0.5).alpha - 0.46).abs() < 1e-6);
    }
}

#[cfg(test)]
mod template_tests {
    use super::*;
    use crate::generate::build::{Options, template};

    #[test]
    #[ignore = "diagnostic"]
    fn diag_liner_coverage_per_tile() {
        let t = template(&Options { lods: false, ..Options::default() }).unwrap().template;
        let canvas = MakeupCanvas::new(&t, 512, |_, _, _| 0.0).unwrap().unwrap();
        println!("placement {:?}", canvas.placement().counts());
        for eye in &canvas.placement().eyes {
            println!("lash {:?}", eye.lash.points);
            println!("crease {:?}", eye.crease.points);
        }
        let mut m = Makeup::default();
        m.set(MakeupInput::Liner, 1.0);
        let layers = canvas.layers(&m, [LayerId(1), LayerId(2)]).unwrap();
        let Some([_, b]) = layers else { panic!("no liner") };
        if let LayerKind::Content { tiles, .. } = &b.kind {
            for c in tiles {
                let covered = c.coverage.iter().filter(|&&a| a > 0.5).count();
                println!("tile {} ({}) rect {:?}: {covered} texels", c.tile, t.tiles()[c.tile as usize].name, c.rect);
            }
        }
        for tc in &canvas.tiles {
            let n = tc.positions.iter().flatten().count();
            let near = tc
                .positions
                .iter()
                .flatten()
                .filter(|p| canvas.placement().eyes.iter().any(|e| e.lash.nearest(**p).distance < 0.002))
                .count();
            println!("tile {}: {n} skin texels, {near} within 2 mm of a lash line", tc.tile);
        }
    }
}
