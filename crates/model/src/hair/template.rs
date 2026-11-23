//! Hair on the template `[HAIR-03]` `[HAIR-04]` `[HAIR-05]` (todo 1.7): the skin the hair grows
//! from, the scalp mask, the brow and lid-margin lines, the face anchors and the hair areas, all
//! read from a [`Template`]'s parts, regions and landmarks (read-only), and the character's groom
//! set built on them ([`character_grooms`]).
//!
//! **The surface.** [`TemplateHair`] keeps every vertex of the template in the template's own
//! order (so the GPU groom's attach step can read the deformation's output buffers directly) and
//! the triangles of the skin part ([`SKIN_PART`]) only, so closest-point queries never land on an
//! eyeball or a tooth. Positions and normals come from the caller: the evaluated character when a
//! viewer builds the hair, the rest shape in tests ([`TemplateHair::rest`]).
//!
//! **Areas.** A triangle belongs to a region list when at least two of its corners' vertex regions
//! lie in (the subtree of) one of the listed regions: [`SCALP_REGIONS`] for the scalp,
//! [`FUZZ_REGIONS`] for the face's vellus hair, [`BODY_HAIR_REGIONS`] for sparse body hair,
//! [`BEARD_REGIONS`] below [`BEARD_TOP`] above the mouth for facial hair. Names the template does
//! not have are skipped, so a template without, say, `malar-L` still grows hair.
//!
//! **Lines.** The brows follow the landmarks `L.brow.01..` / `R.brow.01..` (medial to lateral);
//! the lid margins run from `X.eye.canthus.medial` through `X.eye.upper.NN` (or `lower`) to
//! `X.eye.canthus.lateral`. The eye opening direction is from the upper margin's middle to the
//! lower margin's middle. A side whose landmarks are missing gets no brow or lashes.
//!
//! **The collision field** is the signed distance grid of the whole closed skin (a head-only grid
//! would be open at the neck and its inside could not be told from its outside), with cells sized
//! to fit [`MAX_GRID_AXIS`]. Only the scalp groom uses it.

use humanart_math::Vec3;

use super::collide::{MAX_GRID_AXIS, SdfGrid, TriangleGrid};
use super::error::{HairError, HairResult};
use super::facial::{BodyHairSpec, FaceAnchors, body_hair, body_region, facial_groom};
use super::features::{BROW_SHAPES, BrowSpec, Eyeball, FeatureGroom, LashSpec, brow_groom, lash_groom};
use super::params::{GroomMaps, GroomParams};
use super::placement::{GroomLayout, ScalpMask, place, place_parted};
use super::presets::preset_groom;
use super::strands::Guides;
use super::surface::{HairSurface, normalize_or};
use crate::Template;

/// The template part hair grows on: the skin.
pub const SKIN_PART: &str = "body";
/// The regions the scalp hair grows on.
pub const SCALP_REGIONS: &[&str] = &["scalp", "hairline"];
/// The regions with peach fuzz (vellus hair) on the face.
pub const FUZZ_REGIONS: &[&str] = &[
    "forehead", "temple-L", "temple-R", "cheek-L", "cheek-R", "malar-L", "malar-R", "jaw-L", "jaw-R", "chin",
    "philtrum",
];
/// The regions facial hair (stubble, beard, moustache) grows on, below [`BEARD_TOP`].
pub const BEARD_REGIONS: &[&str] = &["cheek-L", "cheek-R", "jaw-L", "jaw-R", "chin", "philtrum"];
/// The region of the upper lip (vermilion and the skin above it); facial hair grows on the part
/// above the vermilion border.
pub const UPPER_LIP_REGION: &str = "upper-lip";
/// How far above the mouth's centre (along the face's up direction) the beard area reaches,
/// metres: the lower cheeks, not the cheekbones (a styling choice).
pub const BEARD_TOP: f32 = 0.03;
/// The regions with sparse body hair: the arms and legs.
pub const BODY_HAIR_REGIONS: &[&str] =
    &["forearm-L", "forearm-R", "upper-arm-L", "upper-arm-R", "shin-L", "shin-R", "thigh-L", "thigh-R"];
/// The most landmark points read into one feature line.
pub const MAX_LINE_POINTS: usize = 32;
/// The finest collision grid cell, metres.
const SDF_CELL: f32 = 0.005;
/// The most strands of the scalp groom.
pub const MAX_SCALP_STRANDS: u32 = 400_000;

/// The brow and lid-margin lines of both eyes, world space; index 0 is the character's left
/// (`+X`), 1 the right. An empty line means the landmarks were missing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FeatureLines {
    /// The brow lines, medial to lateral.
    pub brows: [Vec<Vec3>; 2],
    /// The upper lid margins, inner to outer canthus.
    pub upper_lids: [Vec<Vec3>; 2],
    /// The lower lid margins, inner to outer canthus.
    pub lower_lids: [Vec<Vec3>; 2],
    /// The unit direction from the upper lid margin into the eye opening (down for an open eye).
    pub openings: [Vec3; 2],
}

/// The template's hair-growing skin, areas, lines and anchors (see the module documentation).
#[derive(Clone, Debug)]
pub struct TemplateHair {
    surface: HairSurface,
    grid: TriangleGrid,
    sdf: SdfGrid,
    scalp: ScalpMask,
    fuzz: Vec<u32>,
    beard: Vec<u32>,
    body: Vec<u32>,
    lines: FeatureLines,
    anchors: FaceAnchors,
    head: (Vec3, f32),
    eyeballs: [Option<Eyeball>; 2],
}

impl TemplateHair {
    /// The hair surface of `template` with vertex `positions` and unit `normals` (one per template
    /// vertex, in the template's order: the evaluated character, or the rest shape).
    ///
    /// # Errors
    /// [`HairError::Invalid`] when the arrays do not match the template, the template has no skin
    /// part or no scalp triangles; the surface, grid and mask errors otherwise.
    pub fn new(template: &Template, positions: &[Vec3], normals: &[Vec3]) -> HairResult<Self> {
        let mesh = template.mesh();
        let n = mesh.vertex_count();
        if positions.len() != n || normals.len() != n {
            return Err(HairError::invalid("template hair", 0, "positions or normals differ from the vertex count"));
        }
        let triangles = skin_triangles(template)?;
        let surface = HairSurface::new(positions.to_vec(), normals.to_vec(), mesh.uvs().to_vec(), triangles)?;
        let regions = template.regions();
        let area = |names: &[&str]| -> Vec<u32> {
            let ids: Vec<u16> = names.iter().filter_map(|name| regions.find(name)).collect();
            let inside = |v: u32| {
                regions.vertex_regions.get(v as usize).is_some_and(|&r| ids.iter().any(|&a| regions.is_within(r, a)))
            };
            // At least two corners inside: a triangle predicate (`triangles_where` takes a vertex one).
            (0..surface.triangles().len() as u32)
                .filter(|&t| surface.triangle(t).is_some_and(|c| c.iter().filter(|&&v| inside(v)).count() >= 2))
                .collect::<Vec<u32>>()
        };
        let scalp_triangles = area(SCALP_REGIONS);
        if scalp_triangles.is_empty() {
            return Err(HairError::invalid("template hair", 0, "the template has no scalp triangles"));
        }
        let head_triangles = area(&["head"]);
        let head = sphere_of(&surface, if head_triangles.is_empty() { &scalp_triangles } else { &head_triangles });
        let scalp = ScalpMask::new(&surface, scalp_triangles)?;
        let fuzz = area(FUZZ_REGIONS);
        let body = area(BODY_HAIR_REGIONS);
        let grid = TriangleGrid::new(surface.positions(), surface.triangles(), 0.01)?;
        let sdf = build_sdf(&grid)?;
        let landmark = |name: &str| -> Option<Vec3> {
            template.landmarks().iter().find(|l| l.name == name).and_then(|l| l.position(positions))
        };
        let lines = feature_lines(&landmark);
        let anchors = face_anchors(&landmark, head.0);
        // Height of a triangle's centroid above the mouth's centre.
        let height =
            |t: u32| surface.corners(t).map(|[a, b, c]| ((a + b + c) * (1.0 / 3.0) - anchors.mouth).dot(anchors.up));
        // The skin of the upper lip above the vermilion (its border's highest landmark + 1 mm)
        // takes a moustache too; the red lip does not.
        let vermilion = (1..=MAX_LINE_POINTS)
            .map_while(|k| landmark(&format!("lip.upper.outer.{k:02}")))
            .map(|p| (p - anchors.mouth).dot(anchors.up))
            .fold(f32::NAN, f32::max);
        let mut beard: Vec<u32> = area(BEARD_REGIONS);
        if vermilion.is_finite() {
            beard.extend(
                area(&[UPPER_LIP_REGION]).into_iter().filter(|&t| height(t).is_some_and(|h| h > vermilion + 0.001)),
            );
        }
        beard.retain(|&t| height(t).is_some_and(|h| h < BEARD_TOP));
        beard.sort_unstable();
        beard.dedup();
        let eyeballs = EYEBALL_PARTS.map(|name| eyeball(template, name, positions));
        Ok(Self { surface, grid, sdf, scalp, fuzz, beard, body, lines, anchors, head, eyeballs })
    }

    /// [`TemplateHair::new`] on the template's rest shape, with area-weighted vertex normals
    /// welded across UV seams ([`vertex_normals`]).
    ///
    /// # Errors
    /// As [`TemplateHair::new`].
    pub fn rest(template: &Template) -> HairResult<Self> {
        let mesh = template.mesh();
        let tri = mesh.triangulate();
        let normals = vertex_normals(mesh.positions(), &tri.indices, mesh.position_keys());
        Self::new(template, mesh.positions(), &normals)
    }

    /// The surface: every template vertex, the skin part's triangles.
    #[must_use]
    pub fn surface(&self) -> &HairSurface {
        &self.surface
    }

    /// The closest-point grid of the surface's triangles (same triangle indices).
    #[must_use]
    pub fn grid(&self) -> &TriangleGrid {
        &self.grid
    }

    /// The signed distance grid of the closed skin.
    #[must_use]
    pub fn sdf(&self) -> &SdfGrid {
        &self.sdf
    }

    /// The scalp mask.
    #[must_use]
    pub fn scalp(&self) -> &ScalpMask {
        &self.scalp
    }

    /// The face triangles with peach fuzz.
    #[must_use]
    pub fn fuzz_triangles(&self) -> &[u32] {
        &self.fuzz
    }

    /// The lower-face triangles facial hair grows on.
    #[must_use]
    pub fn beard_triangles(&self) -> &[u32] {
        &self.beard
    }

    /// The arm and leg triangles with body hair.
    #[must_use]
    pub fn body_triangles(&self) -> &[u32] {
        &self.body
    }

    /// The brow and lid lines.
    #[must_use]
    pub fn lines(&self) -> &FeatureLines {
        &self.lines
    }

    /// The mouth, chin and face-up anchors.
    #[must_use]
    pub fn anchors(&self) -> &FaceAnchors {
        &self.anchors
    }

    /// A sphere around the head's skin (centre, radius).
    #[must_use]
    pub fn head(&self) -> (Vec3, f32) {
        self.head
    }

    /// The eyeball of side `i` (0 the character's left, 1 the right), when the template has it.
    #[must_use]
    pub fn eyeball(&self, i: usize) -> Option<&Eyeball> {
        self.eyeballs.get(i).and_then(Option::as_ref)
    }
}

/// The eyeball parts the lashes must clear, the character's left then right.
pub const EYEBALL_PARTS: [&str; 2] = ["eyeball-L", "eyeball-R"];

/// The eyeball part `name` at vertex `positions`: its closed surface and centre, `None` when the
/// part is missing or degenerate.
fn eyeball(template: &Template, name: &str, positions: &[Vec3]) -> Option<Eyeball> {
    let triangles = part_triangles(template, name)?;
    let grid = TriangleGrid::new(positions, &triangles, 0.002).ok()?;
    let mut used: Vec<u32> = triangles.iter().flatten().copied().collect();
    used.sort_unstable();
    used.dedup();
    let sum = used.iter().filter_map(|&v| positions.get(v as usize)).fold(Vec3::ZERO, |a, &p| a + p);
    #[allow(clippy::cast_precision_loss)] // Vertex counts are far below 2^24.
    let centre = sum * (1.0 / used.len().max(1) as f32);
    Some(Eyeball { grid, centre })
}

/// The triangles of the template's skin part, in triangulation order.
fn skin_triangles(template: &Template) -> HairResult<Vec<[u32; 3]>> {
    let out = part_triangles(template, SKIN_PART).ok_or(HairError::invalid(
        "template hair",
        0,
        "the template has no skin part",
    ))?;
    if out.is_empty() {
        return Err(HairError::invalid("template hair", 0, "the skin part has no triangles"));
    }
    Ok(out)
}

/// The triangles of the template part `name` in triangulation order, `None` when there is no
/// such part.
fn part_triangles(template: &Template, name: &str) -> Option<Vec<[u32; 3]>> {
    let mesh = template.mesh();
    let faces = mesh.parts().iter().find(|p| p.name == name)?.faces();
    let tri = mesh.triangulate();
    Some(
        (0..tri.len())
            .filter(|&t| tri.triangle_faces.get(t).is_some_and(|&f| faces.contains(&(f as usize))))
            .filter_map(|t| tri.triangle(t))
            .collect(),
    )
}

/// Area-weighted unit vertex normals of the triangle list `indices` (three per triangle), summed
/// over every vertex with the same weld key (`keys[v]`, the lowest vertex at the same position) so
/// UV seams do not crease. Vertices on no triangle get `+Y`.
#[must_use]
pub fn vertex_normals(positions: &[Vec3], indices: &[u32], keys: &[u32]) -> Vec<Vec3> {
    let n = positions.len();
    let key = |v: u32| keys.get(v as usize).copied().filter(|&k| (k as usize) < n).unwrap_or(v) as usize;
    let mut sum = vec![Vec3::ZERO; n];
    for t in indices.as_chunks::<3>().0 {
        let (Some(&a), Some(&b), Some(&c)) =
            (positions.get(t[0] as usize), positions.get(t[1] as usize), positions.get(t[2] as usize))
        else {
            continue;
        };
        let area_normal = (b - a).cross(c - a);
        for &v in t {
            sum[key(v)] += area_normal;
        }
    }
    (0..n).map(|v| normalize_or(sum[key(v as u32)], Vec3::Y)).collect()
}

/// A bounding sphere (box centre, half diagonal) of the corners of `triangles`.
fn sphere_of(surface: &HairSurface, triangles: &[u32]) -> (Vec3, f32) {
    let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for &t in triangles {
        for p in surface.corners(t).into_iter().flatten() {
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    if lo.x > hi.x {
        return (Vec3::ZERO, 0.0);
    }
    ((lo + hi) * 0.5, 0.5 * (hi - lo).length())
}

/// The distance grid of the whole skin with the finest cell that fits the grid limits.
fn build_sdf(grid: &TriangleGrid) -> HairResult<SdfGrid> {
    let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for t in grid.triangles() {
        for &v in t {
            if let Some(&p) = grid.positions().get(v as usize) {
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
    }
    let extent = (hi - lo).max_element().max(1e-3);
    // Room for the margin of three cells either side.
    #[allow(clippy::cast_precision_loss)] // A small constant.
    let fit = extent / (MAX_GRID_AXIS as f32 - 8.0);
    let mut cell = SDF_CELL.max(fit);
    let mut last = None;
    for _ in 0..8 {
        match SdfGrid::build(grid, cell, 3.0 * cell) {
            Ok(sdf) => return Ok(sdf),
            Err(e @ HairError::Limit { .. }) => {
                last = Some(e);
                cell *= 1.25;
            }
            Err(e) => return Err(e),
        }
    }
    Err(last.unwrap_or(HairError::invalid("template hair", 0, "no distance grid fits")))
}

/// The brow and lid lines from the landmarks (see the module documentation).
fn feature_lines(landmark: &dyn Fn(&str) -> Option<Vec3>) -> FeatureLines {
    let numbered = |prefix: &str| -> Vec<Vec3> {
        (1..=MAX_LINE_POINTS).map_while(|k| landmark(&format!("{prefix}.{k:02}"))).collect()
    };
    let mut lines = FeatureLines { openings: [-Vec3::Y; 2], ..FeatureLines::default() };
    for (i, side) in ["L", "R"].into_iter().enumerate() {
        let brow = numbered(&format!("{side}.brow"));
        lines.brows[i] = if brow.len() >= 2 { brow } else { Vec::new() };
        let medial = landmark(&format!("{side}.eye.canthus.medial"));
        let lateral = landmark(&format!("{side}.eye.canthus.lateral"));
        let margin = |which: &str| -> Vec<Vec3> {
            let inner = numbered(&format!("{side}.eye.{which}"));
            if inner.is_empty() {
                return Vec::new();
            }
            medial.into_iter().chain(inner).chain(lateral).collect()
        };
        let (upper, lower) = (margin("upper"), margin("lower"));
        if let (Some(u), Some(l)) = (middle(&upper), middle(&lower)) {
            lines.openings[i] = normalize_or(l - u, -Vec3::Y);
        }
        lines.upper_lids[i] = if upper.len() >= 2 { upper } else { Vec::new() };
        lines.lower_lids[i] = if lower.len() >= 2 { lower } else { Vec::new() };
    }
    lines
}

/// The middle point of a polyline's vertices (by index).
fn middle(points: &[Vec3]) -> Option<Vec3> {
    if points.len() < 3 {
        return None;
    }
    points.get(points.len() / 2).copied()
}

/// The face anchors from the lip corners, the menton and the glabella; the head centre stands in
/// for missing ones.
fn face_anchors(landmark: &dyn Fn(&str) -> Option<Vec3>, centre: Vec3) -> FaceAnchors {
    let mouth = match (landmark("L.lip.commissure"), landmark("R.lip.commissure")) {
        (Some(a), Some(b)) => (a + b) * 0.5,
        _ => centre,
    };
    let chin = landmark("chin.menton").unwrap_or(centre);
    let up = landmark("forehead.glabella").map_or(Vec3::Y, |g| normalize_or(g - chin, Vec3::Y));
    FaceAnchors { mouth, chin, up }
}

/// The scalp hair style of a character's groom set; ids are stable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalpStyle {
    /// No scalp hair.
    Bald,
    /// Short hair, about 3 cm, combed back (preset `short_crop`).
    Short,
    /// Straight hair to the jaw, about 18 cm (preset `medium_straight`).
    MediumStraight,
    /// Wavy hair past the shoulders, about 40 cm (preset `long_wavy`).
    LongWavy,
    /// Curly hair, about 18 cm in 6 mm curls (preset `curly`).
    Curly,
    /// Tightly coiled hair, about 8 cm in 2 mm coils (preset `coily`).
    Coily,
    /// Coiled hair standing out in a rounded volume, about 12 cm (preset `afro`).
    Afro,
}

impl ScalpStyle {
    /// Every style, in menu order.
    pub const ALL: [Self; 7] =
        [Self::Bald, Self::Short, Self::MediumStraight, Self::LongWavy, Self::Curly, Self::Coily, Self::Afro];

    /// The stable id.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bald => "bald",
            Self::Short => "short",
            Self::MediumStraight => "medium-straight",
            Self::LongWavy => "long-wavy",
            Self::Curly => "curly",
            Self::Coily => "coily",
            Self::Afro => "afro",
        }
    }

    /// The style with id `name`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.name() == name)
    }

    /// The preset groom (`presets::PRESETS`) of the style; `None` for bald.
    #[must_use]
    pub const fn preset(self) -> Option<&'static str> {
        match self {
            Self::Bald => None,
            Self::Short => Some("short_crop"),
            Self::MediumStraight => Some("medium_straight"),
            Self::LongWavy => Some("long_wavy"),
            Self::Curly => Some("curly"),
            Self::Coily => Some("coily"),
            Self::Afro => Some("afro"),
        }
    }
}

/// What [`character_grooms`] grows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroomSetSpec {
    /// The scalp hair.
    pub scalp: ScalpStyle,
    /// Scalp strands (at most [`MAX_SCALP_STRANDS`]).
    pub scalp_strands: u32,
    /// Scalp guides.
    pub scalp_guides: usize,
    /// Eyebrows.
    pub brows: bool,
    /// Upper and lower lashes.
    pub lashes: bool,
    /// Vellus hair on the face.
    pub peach_fuzz: bool,
    /// Sparse hair on the arms and legs.
    pub body_hair: bool,
    /// Eumelanin of the scalp, brow, lash and body hair (pbrt scale: about 0.3 blonde, 1.3 brown,
    /// 8 black).
    pub eumelanin: f32,
    /// Pheomelanin of the same.
    pub pheomelanin: f32,
    /// The seed every random choice derives from.
    pub seed: u64,
    /// Brow hairs relative to the default brow (1 = 520 hairs per brow, 0 = none).
    pub brow_density: f32,
    /// Brow hair length relative to the default (1 = 8 mm).
    pub brow_length: f32,
    /// Lashes relative to the default lids (1 = 150 upper, 75 lower, 0 = none).
    pub lash_density: f32,
    /// Lash length relative to the default.
    pub lash_length: f32,
    /// Lash curl relative to the default.
    pub lash_curl: f32,
    /// Body hair density relative to the default (1 = a third of the arm preset's).
    pub body_density: f32,
    /// Share of unpigmented (grey) strands in the scalp, brow, facial and body hair, 0..1.
    pub grey_fraction: f32,
    /// Stubble over the beard area, 0 (none) to 1 (60 hairs per cm²).
    pub stubble: f32,
    /// Stubble length relative to the default 1 mm.
    pub stubble_length: f32,
    /// A trimmed beard (8 mm) over the beard area, 0 (none) to 1 (60 hairs per cm²).
    pub beard: f32,
    /// A moustache (12 mm) above the mouth, 0 (none) to 1 (70 hairs per cm²).
    pub moustache: f32,
}

impl Default for GroomSetSpec {
    fn default() -> Self {
        Self {
            scalp: ScalpStyle::Short,
            scalp_strands: 100_000,
            scalp_guides: 1500,
            brows: true,
            lashes: true,
            peach_fuzz: true,
            body_hair: false,
            eumelanin: 2.5,
            pheomelanin: 0.1,
            seed: 1,
            brow_density: 1.0,
            brow_length: 1.0,
            lash_density: 1.0,
            lash_length: 1.0,
            lash_curl: 1.0,
            body_density: 1.0,
            grey_fraction: 0.0,
            stubble: 0.0,
            stubble_length: 1.0,
            beard: 0.0,
            moustache: 0.0,
        }
    }
}

/// What a groom of the set is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroomKind {
    /// The scalp hair.
    Scalp,
    /// Both eyebrows.
    Brows,
    /// The upper and lower lashes of both eyes.
    Lashes,
    /// Vellus hair on the face.
    PeachFuzz,
    /// Hair on the arms and legs.
    Body,
    /// Stubble, beard or moustache.
    Facial,
}

impl GroomKind {
    /// A stable id for logs and file names.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalp => "scalp",
            Self::Brows => "brows",
            Self::Lashes => "lashes",
            Self::PeachFuzz => "peach-fuzz",
            Self::Body => "body",
            Self::Facial => "facial",
        }
    }
}

/// One groom of a character: guides, parameters and the strand layout, ready for the GPU groom
/// (`humanart-render::groom::GpuGroom`) or the CPU reference (`generate_groom`).
#[derive(Clone, Debug, PartialEq)]
pub struct CharacterGroom {
    /// What it is.
    pub kind: GroomKind,
    /// The guides.
    pub guides: Guides,
    /// The generation parameters (strand count, point count, colour).
    pub params: GroomParams,
    /// The strands' roots and guide weights.
    pub layout: GroomLayout,
    /// The mean guide length, metres.
    pub length: f32,
}

/// Guides of each facial hair groom.
const FACIAL_GUIDES: usize = 400;

/// Feature hairs: eyebrows and lashes, hair per guide.
const FEATURE_SALTS: [u64; 4] = [0xB10, 0xB11, 0x1A5, 0x1A6];

/// Grows the groom set `spec` on `hair` (see the module documentation): the scalp preset on the
/// scalp mask, both brows along the brow lines, upper and lower lashes along the lid margins (one
/// groom each kind, both sides), peach fuzz on the face, sparse body hair and the facial hair
/// (stubble, beard, moustache: one groom each over the beard area) the spec asks for; the brow,
/// lash and body settings scale the generators' defaults (todo 2.3). Deterministic for a seed; a
/// feature whose landmarks are missing is left out.
///
/// # Errors
/// [`HairError::Limit`] for a scalp strand count above [`MAX_SCALP_STRANDS`]; the generators'
/// errors otherwise.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)] // Hair counts are far below 2^24.
pub fn character_grooms(hair: &TemplateHair, spec: &GroomSetSpec) -> HairResult<Vec<CharacterGroom>> {
    if spec.scalp_strands > MAX_SCALP_STRANDS {
        return Err(HairError::limit("scalp strands", u64::from(MAX_SCALP_STRANDS), u64::from(spec.scalp_strands)));
    }
    let colour = |kind: GroomKind| pigments(kind, spec);
    let (eu, pheo, grey) = colour(GroomKind::Scalp);
    // A relative factor: finite, 0..4.
    let factor = |v: f32| if v.is_finite() { v.clamp(0.0, 4.0) } else { 1.0 };
    let surface = hair.surface();
    let maps = GroomMaps::default();
    let mut out = Vec::new();
    if let Some(preset) = spec.scalp.preset()
        && spec.scalp_strands > 0
    {
        let p = preset_groom(preset, surface, hair.scalp(), spec.scalp_guides.max(1), Some(hair.sdf()), spec.seed)?;
        let params = GroomParams {
            strand_count: spec.scalp_strands as f32,
            // A hairline that thins over 2 cm rather than ending at the region border (with
            // finer strands and baby hairs in the fade, `placement::HAIRLINE_WIDTH`; a wider fade
            // of long fine strands read as a veil at the temples in profile).
            hairline_softness: p.params.hairline_softness.max(0.02),
            eumelanin: eu,
            pheomelanin: pheo,
            grey_fraction: grey,
            ..p.params
        };
        let layout =
            place_parted(surface, hair.scalp(), &p.guides, &params, &maps, spec.seed ^ 0x5CA1, p.part.as_ref())?;
        out.push(groom(GroomKind::Scalp, p.guides, params, layout));
    }
    let lines = hair.lines();
    let feature = |kind: GroomKind, parts: Vec<FeatureGroom>| -> HairResult<Option<CharacterGroom>> {
        let Some(first) = parts.first() else { return Ok(None) };
        let mut guides = Guides::new(first.guides.points())?;
        for part in &parts {
            for i in 0..part.guides.len() {
                if let Some((t, bary)) = part.guides.root(i) {
                    guides.push(t, bary, part.guides.offsets(i))?;
                }
            }
        }
        let params = GroomParams {
            strand_count: guides.len() as f32,
            eumelanin: colour(kind).0,
            pheomelanin: colour(kind).1,
            color_variation: 0.05,
            grey_fraction: colour(kind).2,
            ..first.params
        };
        let layout = GroomLayout::one_per_guide(&guides, surface, &maps, spec.seed ^ kind as u64)?;
        Ok(Some(groom(kind, guides, params, layout)))
    };
    let brow = BrowSpec { shape: BROW_SHAPES[1], ..BrowSpec::default() };
    let brow = BrowSpec {
        count: (brow.count as f32 * factor(spec.brow_density)).round() as usize,
        length: brow.length * factor(spec.brow_length).max(0.1),
        ..brow
    };
    if spec.brows && brow.count > 0 {
        let mut parts = Vec::new();
        for (i, line) in lines.brows.iter().enumerate() {
            if line.len() < 2 {
                continue;
            }
            let up = hair.anchors().up;
            parts.push(brow_groom(surface, hair.grid(), line, up, &brow, spec.seed ^ FEATURE_SALTS[i])?);
        }
        // Brows and lashes are a little darker than the scalp hair.
        out.extend(feature(GroomKind::Brows, parts)?);
    }
    let lash = |l: LashSpec| LashSpec {
        count: (l.count as f32 * factor(spec.lash_density)).round() as usize,
        length: l.length * factor(spec.lash_length).max(0.1),
        curl: l.curl * factor(spec.lash_curl),
        ..l
    };
    let (upper, lower) = (lash(LashSpec::upper()), lash(LashSpec::lower()));
    if spec.lashes && upper.count + lower.count > 0 {
        let mut parts = Vec::new();
        for i in 0..2 {
            let opening = lines.openings[i];
            for (margin, lash, salt) in
                [(&lines.upper_lids[i], upper, FEATURE_SALTS[2]), (&lines.lower_lids[i], lower, FEATURE_SALTS[3])]
            {
                if margin.len() < 2 || lash.count == 0 {
                    continue;
                }
                let into = if lash.upper { opening } else { -opening };
                let seed = spec.seed ^ salt ^ ((i as u64) << 20);
                parts.push(lash_groom(surface, hair.grid(), margin, into, hair.eyeball(i), &lash, seed)?);
            }
        }
        // Lashes are darker than the scalp hair of nearly everyone (a styling choice, not a
        // measurement): dark brown hair (2.5) gets near-black lashes (7), blonde (0.3) brown ones.
        out.extend(feature(GroomKind::Lashes, parts)?);
    }
    if spec.peach_fuzz && !hair.fuzz_triangles().is_empty() {
        // Vellus hair: about 2.5 mm, 12 micrometres, lightly pigmented, lying down the face, dense
        // enough (150 per cm^2) to read as a fine down; the renderer never thins it (sparse
        // survivors would show as specks).
        let fuzz = BodyHairSpec {
            length: 0.002_5,
            density: 150.0,
            flow: -hair.anchors().up,
            lift: 0.12,
            curl_radius: 0.0,
            thickness: 0.000_012,
        };
        let mask = ScalpMask::new(surface, hair.fuzz_triangles().to_vec())?;
        let p = body_hair(surface, &mask, &fuzz, 600, spec.seed ^ 0xF422)?;
        let (eumelanin, pheomelanin, grey_fraction) = colour(GroomKind::PeachFuzz);
        let params = GroomParams { eumelanin, pheomelanin, grey_fraction, tip_ratio: 0.4, ..p.params };
        let layout = place(surface, &mask, &p.guides, &params, &maps, spec.seed ^ 0xF423)?;
        out.push(groom(GroomKind::PeachFuzz, p.guides, params, layout));
    }
    if spec.body_hair && factor(spec.body_density) > 0.0 && !hair.body_triangles().is_empty() {
        let base = body_region("arm").unwrap_or(BodyHairSpec {
            length: 0.01,
            density: 10.0,
            flow: -Vec3::Y,
            lift: 0.2,
            curl_radius: 0.0,
            thickness: 0.000_04,
        });
        // Sparse: a third of the arm preset's density.
        let sparse = BodyHairSpec { density: base.density / 3.0 * factor(spec.body_density), ..base };
        let mask = ScalpMask::new(surface, hair.body_triangles().to_vec())?;
        let p = body_hair(surface, &mask, &sparse, 1500, spec.seed ^ 0xB0D7)?;
        let (eumelanin, pheomelanin, grey_fraction) = colour(GroomKind::Body);
        let params = GroomParams { grey_fraction, eumelanin, pheomelanin, ..p.params };
        let layout = place(surface, &mask, &p.guides, &params, &maps, spec.seed ^ 0xB0D8)?;
        out.push(groom(GroomKind::Body, p.guides, params, layout));
    }
    let facial = [
        ("stubble", spec.stubble, factor(spec.stubble_length).max(0.1), 0xFAC1_u64),
        ("short_beard", spec.beard, 1.0, 0xFAC2),
        ("moustache", spec.moustache, 1.0, 0xFAC3),
    ];
    if !hair.beard_triangles().is_empty() {
        let area = ScalpMask::new(surface, hair.beard_triangles().to_vec())?;
        for (style, amount, length, salt) in facial {
            let amount = if amount.is_finite() { amount.clamp(0.0, 1.0) } else { 0.0 };
            if amount <= 0.0 {
                continue;
            }
            let p =
                facial_groom(style, surface, &area, hair.anchors(), FACIAL_GUIDES, Some(hair.sdf()), spec.seed ^ salt)?;
            let params = GroomParams {
                strand_count: (p.params.strand_count * amount).floor().max(1.0),
                length_scale: p.params.length_scale * length,
                eumelanin: colour(GroomKind::Facial).0,
                pheomelanin: colour(GroomKind::Facial).1,
                grey_fraction: colour(GroomKind::Facial).2,
                ..p.params
            };
            let mask = ScalpMask::new(surface, p.area)?;
            let layout = place(surface, &mask, &p.guides, &params, &maps, spec.seed ^ salt ^ 0x1)?;
            out.push(groom(GroomKind::Facial, p.guides, params, layout));
        }
    }
    Ok(out)
}

/// The eumelanin, pheomelanin (pbrt scale) and grey share [`character_grooms`] gives a groom of
/// `kind` for `spec`: the scalp and facial hair take the spec's colour; brows are a little darker
/// and lashes darker still (a styling choice, not a measurement: dark brown hair (2.5) gets
/// near-black lashes (7), blonde (0.3) brown ones), neither greys as fast (lashes not at all);
/// body hair is a little lighter; peach fuzz is lightly pigmented whatever the hair colour.
#[must_use]
pub fn pigments(kind: GroomKind, spec: &GroomSetSpec) -> (f32, f32, f32) {
    let finite = |v: f32, lo: f32, hi: f32, or: f32| if v.is_finite() { v.clamp(lo, hi) } else { or };
    let (eu, pheo) = (finite(spec.eumelanin, 0.0, 16.0, 2.5), finite(spec.pheomelanin, 0.0, 16.0, 0.1));
    let grey = finite(spec.grey_fraction, 0.0, 1.0, 0.0);
    match kind {
        GroomKind::Scalp | GroomKind::Facial => (eu, pheo, grey),
        GroomKind::Brows => ((eu * 1.3).clamp(0.5, 16.0), pheo, grey * 0.7),
        GroomKind::Lashes => ((eu * 2.0 + 2.0).clamp(4.0, 16.0), pheo, 0.0),
        GroomKind::PeachFuzz => (0.3, 0.12, 0.0),
        GroomKind::Body => ((eu * 0.8).clamp(0.0, 16.0), pheo, grey),
    }
}

fn groom(kind: GroomKind, guides: Guides, params: GroomParams, layout: GroomLayout) -> CharacterGroom {
    let n = guides.len().max(1);
    #[allow(clippy::cast_precision_loss)] // Guide counts are far below 2^24.
    let length = (0..guides.len()).map(|i| guides.length(i)).sum::<f32>() / n as f32 * params.length_scale;
    CharacterGroom { kind, guides, params, layout, length }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;
    use crate::generate::build::{Options, template};
    use crate::hair::collide::penetration;
    use crate::hair::generate_groom;

    fn fixture() -> &'static (Template, TemplateHair) {
        static F: OnceLock<(Template, TemplateHair)> = OnceLock::new();
        F.get_or_init(|| {
            let t = template(&Options { lods: false, ..Options::default() }).expect("the template generates").template;
            let hair = TemplateHair::rest(&t).expect("the template has hair areas");
            (t, hair)
        })
    }

    #[test]
    fn hair_04_template_areas_lines_and_anchors_come_from_regions_and_landmarks() {
        let (t, hair) = fixture();
        let s = hair.surface();
        assert_eq!(s.positions().len(), t.mesh().vertex_count(), "every template vertex, in order");
        // The scalp is on top of and behind the head, its area a plausible scalp (300..1200 cm^2).
        let area_cm2 = hair.scalp().area() * 1e4;
        let (centre, radius) = hair.head();
        println!(
            "scalp {} triangles, {area_cm2:.0} cm^2; fuzz {} triangles; body {} triangles; head r {radius:.3}",
            hair.scalp().triangles().len(),
            hair.fuzz_triangles().len(),
            hair.body_triangles().len()
        );
        let bbox = |tris: &[u32]| {
            let mut lo = Vec3::splat(f32::INFINITY);
            let mut hi = Vec3::splat(f32::NEG_INFINITY);
            for &tri in tris {
                for p in s.corners(tri).unwrap() {
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
            (lo, hi)
        };
        println!("scalp box {:?}, fuzz box {:?}", bbox(hair.scalp().triangles()), bbox(hair.fuzz_triangles()));
        assert!((300.0..1200.0).contains(&area_cm2), "{area_cm2}");
        assert!((0.08..0.25).contains(&radius), "{radius}");
        // Most of the scalp lies above the eyes (the rest is the nape and behind the ears), and
        // none of it is in front of the brows.
        let eyes = hair.lines().upper_lids.iter().flatten().fold(Vec3::ZERO, |a, &b| a + b)
            * (1.0 / hair.lines().upper_lids.iter().map(Vec::len).sum::<usize>().max(1) as f32);
        let brow_z = hair.lines().brows.iter().flatten().map(|p| p.z).fold(f32::INFINITY, f32::min);
        let (mut above, mut front) = (0.0, 0);
        for &tri in hair.scalp().triangles() {
            let c = s.corners(tri).unwrap();
            let m = (c[0] + c[1] + c[2]) * (1.0 / 3.0);
            if m.y > eyes.y {
                above += s.area(tri);
            }
            front += usize::from(m.z > brow_z && m.y < eyes.y + 0.03);
        }
        let above = above / hair.scalp().area();
        println!("scalp area above the eyes {above:.2}, triangles in front of the brows {front}; centre {centre:?}");
        assert!(above > 0.6, "the scalp lies on the upper head");
        assert_eq!(front, 0, "no scalp on the face");
        assert!(!hair.fuzz_triangles().is_empty() && !hair.body_triangles().is_empty());
        // Lines: both brows above both upper lids, the left at +X, the eye opening points down.
        let l = hair.lines();
        for i in 0..2 {
            assert!(l.brows[i].len() >= 5 && l.upper_lids[i].len() >= 4 && l.lower_lids[i].len() >= 4, "{i}");
            let mean = |p: &[Vec3]| p.iter().fold(Vec3::ZERO, |a, &b| a + b) * (1.0 / p.len() as f32);
            let (brow, upper, lower) = (mean(&l.brows[i]), mean(&l.upper_lids[i]), mean(&l.lower_lids[i]));
            assert!(brow.y > upper.y + 0.01 && upper.y > lower.y, "side {i}: {brow:?} {upper:?} {lower:?}");
            assert!((brow.x > 0.0) == (i == 0), "side {i}");
            assert!(l.openings[i].y < -0.5, "{:?}", l.openings[i]);
        }
        let a = hair.anchors();
        assert!(a.mouth.y > a.chin.y && a.up.y > 0.8, "{a:?}");
        // Normals point out of the head on the scalp.
        let outward = hair
            .scalp()
            .triangles()
            .iter()
            .filter(|&&tri| {
                let [v, ..] = s.triangle(tri).unwrap();
                s.normals()[v as usize].dot(s.positions()[v as usize] - centre) > 0.0
            })
            .count();
        assert!(outward * 100 > hair.scalp().triangles().len() * 97, "{outward}");
        // The distance grid knows the inside of the head.
        assert!(hair.sdf().sample(centre).0 < -0.02);
    }

    #[test]
    fn hair_03_04_the_groom_set_grows_from_the_template_and_is_deterministic() {
        let (_, hair) = fixture();
        let spec = GroomSetSpec { scalp_strands: 4000, scalp_guides: 200, body_hair: true, ..GroomSetSpec::default() };
        let set = character_grooms(hair, &spec).unwrap();
        let kinds: Vec<GroomKind> = set.iter().map(|g| g.kind).collect();
        assert_eq!(
            kinds,
            [GroomKind::Scalp, GroomKind::Brows, GroomKind::Lashes, GroomKind::PeachFuzz, GroomKind::Body]
        );
        let s = hair.surface();
        for g in &set {
            g.layout.check(s, g.guides.len()).unwrap();
            g.params.check().unwrap();
            println!(
                "{}: {} guides, {} strands, mean length {:.4} m",
                g.kind.name(),
                g.guides.len(),
                g.layout.len(),
                g.length
            );
            assert!(!g.layout.is_empty(), "{}", g.kind.name());
        }
        let by = |k: GroomKind| set.iter().find(|g| g.kind == k).unwrap();
        // Brows: hundreds of ~8 mm hairs on the brow ridge, above the eyes.
        let brows = by(GroomKind::Brows);
        assert!(brows.guides.len() > 600 && (0.004..0.012).contains(&brows.length));
        // Lashes: both lids of both eyes, roots near the lid margins.
        let lashes = by(GroomKind::Lashes);
        assert!(lashes.guides.len() > 300, "{}", lashes.guides.len());
        let margins: Vec<Vec3> =
            hair.lines().upper_lids.iter().chain(&hair.lines().lower_lids).flatten().copied().collect();
        let far = (0..lashes.guides.len())
            .filter(|&i| {
                let root = lashes.guides.frame(i, s).unwrap().origin;
                margins.iter().map(|m| (*m - root).length()).fold(f32::INFINITY, f32::min) > 0.004
            })
            .count();
        assert!(far * 20 < lashes.guides.len(), "{far} lash roots far from the lid margins");
        assert!(by(GroomKind::PeachFuzz).length < 0.005);
        // The scalp groom generates outside the head.
        let scalp = by(GroomKind::Scalp);
        let generated = generate_groom(s, &scalp.layout, &scalp.guides, &scalp.params, Some(hair.sdf())).unwrap();
        let st = &generated.strands;
        let points: Vec<Vec3> = (0..st.len().min(2000))
            .flat_map(|i| {
                let frame = s.frame(st.triangle[i], st.bary[i]).unwrap();
                (0..st.points as usize).map(move |k| frame.to_world(st.offset(i, k)))
            })
            .collect();
        let report = penetration(hair.grid(), Some(hair.sdf()), &points, 0.003);
        println!("scalp penetration: {report:?}");
        assert!(report.inside * 100 <= report.points, "{report:?}");
        // Contact sheet (front and the character's left): head skin, CPU-generated scalp hair,
        // brows and lashes.
        let (centre, _) = hair.head();
        let dots: Vec<Vec3> =
            s.positions().iter().copied().filter(|p| (*p - centre).length() < 0.16).step_by(3).collect();
        let strands = &points;
        let per = st.points as usize;
        let mut close = Vec::new();
        for g in set.iter().filter(|g| matches!(g.kind, GroomKind::Brows | GroomKind::Lashes)) {
            let n = g.guides.points() as usize;
            for i in 0..g.guides.len() {
                let f = g.guides.frame(i, s).unwrap();
                let w = g.guides.world_points(i, &f);
                let resampled: Vec<Vec3> = (0..4).map(|k| w[(k * (n - 1)) / 3]).collect();
                close.extend_from_slice(&resampled);
            }
        }
        let eye = hair.lines().upper_lids[0][2];
        let panels = [
            crate::hair::sheet::Panel { dots: &dots, strands, points: per, centre, scale: 1500.0, front: true },
            crate::hair::sheet::Panel { dots: &dots, strands, points: per, centre, scale: 1500.0, front: false },
            crate::hair::sheet::Panel {
                dots: &dots,
                strands: &close,
                points: 4,
                centre: eye,
                scale: 6000.0,
                front: true,
            },
            crate::hair::sheet::Panel {
                dots: &dots,
                strands: &close,
                points: 4,
                centre: eye,
                scale: 6000.0,
                front: false,
            },
        ];
        crate::hair::sheet::write("../hair-render/template_grooms.png", &panels, 2, 480, 480);
        // Deterministic per seed.
        let again = character_grooms(hair, &spec).unwrap();
        assert_eq!(set, again);
        let other = character_grooms(hair, &GroomSetSpec { seed: 2, ..spec }).unwrap();
        assert_ne!(set[0].guides, other[0].guides);
        // Every style resolves to a preset (or none) and back from its id.
        for style in ScalpStyle::ALL {
            assert_eq!(ScalpStyle::from_name(style.name()), Some(style));
            assert_eq!(style.preset().is_none(), style == ScalpStyle::Bald);
        }
        assert!(character_grooms(hair, &GroomSetSpec { scalp_strands: MAX_SCALP_STRANDS + 1, ..spec }).is_err());
    }

    /// `[HAIR-02]` Strands blend their guides in world directions: the template's scalp has
    /// frame flips (UV seams, where the root frame's tangent turns round), and a strand blending
    /// guides in its own root frame across one was bent away from the hair around it (strays
    /// standing out behind the occiput in the medium straight and long wavy stills). Measured: the
    /// strands whose first quarter leaves more than 60 degrees away from their main guide's, with
    /// the root-frame blend and with the world blend.
    #[test]
    fn hair_02_strands_blend_guides_in_world_directions() {
        use crate::hair::generate::{GenerateInput, generate, root_frames};
        let (_, hair) = fixture();
        let s = hair.surface();
        let spec = GroomSetSpec {
            scalp: ScalpStyle::MediumStraight,
            scalp_strands: 20_000,
            brows: false,
            lashes: false,
            peach_fuzz: false,
            ..GroomSetSpec::default()
        };
        let set = character_grooms(hair, &spec).unwrap();
        let scalp = set.iter().find(|g| g.kind == GroomKind::Scalp).unwrap();
        let roots: Vec<Vec3> = (0..scalp.guides.len()).map(|g| scalp.guides.frame(g, s).unwrap().origin).collect();
        let guide_t: Vec<Vec3> = (0..scalp.guides.len()).map(|g| scalp.guides.frame(g, s).unwrap().t).collect();
        let mut flipped = 0;
        for a in 0..roots.len() {
            for b in a + 1..roots.len() {
                if (roots[a] - roots[b]).length() < 0.02 && guide_t[a].dot(guide_t[b]) < 0.0 {
                    flipped += 1;
                }
            }
        }
        let n = scalp.params.point_count() as usize;
        let quarter = n / 4;
        // The main guide's direction over its first quarter, in the world.
        let guide_dir: Vec<Vec3> = (0..scalp.guides.len())
            .map(|g| {
                let w = scalp.guides.world_points(g, &scalp.guides.frame(g, s).unwrap());
                let q = (w.len() - 1) / 4;
                (w[q] - w[0]).normalize_or_zero()
            })
            .collect();
        let frames = root_frames(&scalp.layout, s).unwrap();
        let l = &scalp.layout;
        let strays = |world: bool| {
            let curves = if world {
                scalp.guides.resampled_world(n as u32, s).unwrap()
            } else {
                scalp.guides.resampled(n as u32)
            };
            let input = GenerateInput {
                layout: l,
                curves: &curves,
                params: &scalp.params,
                frames: &frames,
                sdf: Some(hair.sdf()),
                world_curves: world,
            };
            let g = generate(&input).unwrap();
            (0..l.len())
                .filter(|&i| {
                    let main = (0..4).max_by(|&a, &b| l.weights[i][a].total_cmp(&l.weights[i][b])).unwrap_or(0);
                    let dir = (g.shape[i * n + quarter] - g.shape[i * n]).normalize_or_zero();
                    dir.dot(guide_dir[l.guides[i][main] as usize]) < 0.5
                })
                .count()
        };
        let (local, world) = (strays(false), strays(true));
        println!(
            "[HAIR-02] {} guides: {flipped} guide pairs under 2 cm apart with opposite root frames; {} strands; \
             strays (first quarter over 60 degrees from the main guide's): root-frame blend {local}, world blend {world}",
            scalp.guides.len(),
            l.len()
        );
        assert!(world <= local, "{world} > {local}");
        assert!(world * 1000 < l.len(), "{world} strays");
    }

    /// `[HAIR-04]` Every lash point stays outside both eyeballs (the template's `eyeball-L` /
    /// `eyeball-R` parts, by the exact closest point of their closed surfaces) and outside the lid
    /// skin past its root: the lower lashes used to pierce the eyeball at the lateral canthus.
    #[test]
    fn hair_04_lashes_stay_outside_the_eyeballs_and_the_lids() {
        let (t, hair) = fixture();
        let spec = GroomSetSpec { scalp: ScalpStyle::Bald, brows: false, peach_fuzz: false, ..GroomSetSpec::default() };
        let set = character_grooms(hair, &spec).unwrap();
        let lashes = set.iter().find(|g| g.kind == GroomKind::Lashes).unwrap();
        let positions = t.mesh().positions();
        let eyes: Vec<TriangleGrid> = EYEBALL_PARTS
            .iter()
            .map(|name| TriangleGrid::new(positions, &part_triangles(t, name).unwrap(), 0.002).unwrap())
            .collect();
        assert!(hair.eyeball(0).is_some() && hair.eyeball(1).is_some());
        let s = hair.surface();
        let lateral: Vec<Vec3> = hair.lines().lower_lids.iter().filter_map(|l| l.last().copied()).collect();
        let (mut inside_eye, mut inside_skin, mut points) = (0, 0, 0);
        let mut nearest = f32::INFINITY;
        let mut worst: Option<(f32, Vec3, f32)> = None;
        // Crumpled lashes (the canthi used to show zigzags): the largest turn between two
        // segments of a lash.
        let (mut crumpled, mut sharpest) = (0, 0.0_f32);
        for i in 0..lashes.guides.len() {
            let w = lashes.guides.world_points(i, &lashes.guides.frame(i, s).unwrap());
            let turn = w
                .windows(3)
                .map(|q| {
                    (q[1] - q[0]).normalize_or_zero().dot((q[2] - q[1]).normalize_or_zero()).clamp(-1.0, 1.0).acos()
                })
                .fold(0.0_f32, f32::max);
            sharpest = sharpest.max(turn);
            crumpled += usize::from(turn > 0.5);
            for (k, &p) in w.iter().enumerate() {
                points += 1;
                let eye =
                    eyes.iter().filter_map(|g| g.closest(p)).map(|c| c.signed_distance).fold(f32::INFINITY, f32::min);
                nearest = nearest.min(eye);
                if eye < 0.000_2 {
                    inside_eye += 1;
                    let canthus = lateral.iter().map(|&c| (c - w[0]).length()).fold(f32::INFINITY, f32::min);
                    if worst.is_none_or(|(d, ..)| eye < d) {
                        worst = Some((eye, w[0], canthus));
                    }
                }
                if k > 0
                    && let Some(c) = hair.grid().closest(p)
                    && c.signed_distance < -0.000_2
                {
                    inside_skin += 1;
                    println!("lash {i} point {k} {:.2} mm inside the skin, root {:?}", c.signed_distance * 1e3, w[0]);
                }
            }
        }
        println!(
            "[HAIR-04] lashes: {} hairs, {points} points; within 0.2 mm of or inside an eyeball {inside_eye}, \
             inside the lid skin {inside_skin}; nearest eyeball distance {:.2} mm; worst (distance, root, \
             distance of the root to the lateral canthus) {worst:?}",
            lashes.guides.len(),
            nearest * 1e3
        );
        println!(
            "[HAIR-04] lashes with a turn above 0.5 rad between two segments: {crumpled}; sharpest {sharpest:.3} rad"
        );
        assert_eq!(crumpled, 0, "crumpled lashes");
        assert_eq!(inside_eye, 0, "lash points touch or pierce an eyeball");
        assert_eq!(inside_skin, 0, "lash points run inside the lid skin");
    }

    /// todo 2.3: the document's groom settings change the groom set: brow and lash densities and
    /// lengths scale the features, stubble, beard and moustache grow facial grooms on the beard
    /// area (below the cheekbones), grey reaches every greying groom.
    #[test]
    fn todo_2_3_groom_settings_scale_the_features_and_grow_facial_hair() {
        let (_, hair) = fixture();
        let base = GroomSetSpec { scalp: ScalpStyle::Bald, peach_fuzz: false, ..GroomSetSpec::default() };
        let strands = |set: &[CharacterGroom], kind: GroomKind| -> usize {
            set.iter().filter(|g| g.kind == kind).map(|g| g.layout.len()).sum()
        };
        let length = |set: &[CharacterGroom], kind: GroomKind| -> f32 {
            set.iter().filter(|g| g.kind == kind).map(|g| g.length).sum()
        };
        let plain = character_grooms(hair, &base).unwrap();
        assert_eq!(strands(&plain, GroomKind::Facial), 0, "no facial hair by default");
        let sparse = character_grooms(hair, &GroomSetSpec { brow_density: 0.5, lash_density: 0.5, ..base }).unwrap();
        let none = character_grooms(hair, &GroomSetSpec { brow_density: 0.0, lash_density: 0.0, ..base }).unwrap();
        let long = character_grooms(hair, &GroomSetSpec { brow_length: 1.5, lash_length: 1.5, ..base }).unwrap();
        for kind in [GroomKind::Brows, GroomKind::Lashes] {
            let (a, b) = (strands(&plain, kind), strands(&sparse, kind));
            println!(
                "[todo 2.3] {kind:?}: {a} hairs, at half density {b}, lengths {} -> {}",
                length(&plain, kind),
                length(&long, kind)
            );
            assert!(b * 10 < a * 6 && b * 10 > a * 4, "{kind:?} {a} -> {b}");
            assert_eq!(strands(&none, kind), 0);
            assert!(length(&long, kind) > 1.3 * length(&plain, kind));
        }
        assert!(!hair.beard_triangles().is_empty());
        let a = hair.anchors();
        for &t in hair.beard_triangles() {
            let [p, q, r] = hair.surface().corners(t).unwrap();
            assert!(((p + q + r) * (1.0 / 3.0) - a.mouth).dot(a.up) < BEARD_TOP);
        }
        let faced = GroomSetSpec { stubble: 1.0, beard: 0.5, moustache: 1.0, grey_fraction: 0.4, ..base };
        let set = character_grooms(hair, &faced).unwrap();
        let facial: Vec<&CharacterGroom> = set.iter().filter(|g| g.kind == GroomKind::Facial).collect();
        println!(
            "[todo 2.3] facial grooms: {:?}",
            facial.iter().map(|g| (g.layout.len(), g.length)).collect::<Vec<_>>()
        );
        assert_eq!(facial.len(), 3, "stubble, beard and moustache");
        assert!(facial.iter().all(|g| g.layout.len() > 100 && g.params.grey_fraction == 0.4));
        let half = character_grooms(hair, &GroomSetSpec { stubble: 0.5, ..base }).unwrap();
        let full = character_grooms(hair, &GroomSetSpec { stubble: 1.0, ..base }).unwrap();
        let (h, f) = (strands(&half, GroomKind::Facial), strands(&full, GroomKind::Facial));
        assert!(h * 10 < f * 6, "stubble density: {h} vs {f}");
        let longer = character_grooms(hair, &GroomSetSpec { stubble: 1.0, stubble_length: 1.8, ..base }).unwrap();
        assert!(length(&longer, GroomKind::Facial) > 1.5 * length(&full, GroomKind::Facial));
        let brows = set.iter().find(|g| g.kind == GroomKind::Brows).unwrap();
        assert_eq!(brows.params.grey_fraction, pigments(GroomKind::Brows, &faced).2);
        assert_eq!(set.iter().find(|g| g.kind == GroomKind::Lashes).unwrap().params.grey_fraction, 0.0);
    }
}
