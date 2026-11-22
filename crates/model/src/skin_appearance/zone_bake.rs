//! Zone masks baked from the template `[SKN-11]`: the soft masks the compositor reads (three RGBA8
//! layers per UV tile, zone slot `s` in layer `s / 4`, channel `s % 4`, see
//! [`super::layers::TexelInputs`]).
//!
//! Each [`Zone`] is found on the template **by name**, never by index, so a re-generated template
//! keeps working as long as its stable ids do ([`TEMPLATE_ZONES`]):
//!
//! - face zones are region subtrees (`cheek-L`, `malar-L`, `nose`, `lower-lid-L`, `upper-lip`, ...);
//! - knuckles, elbows and knees have no region of their own, so they are **caps** found from named
//!   joints of the rest skeleton (`L.index.01` .. `L.little.02`, `L.forearm`, `L.shin`): a ray from
//!   the joint towards the side that darkens ([`Facing`]: dorsum of the hand, back of the elbow, front
//!   of the knee) picks the skin vertex it leaves through, and the cap is every vertex of the limb's
//!   regions within a radius of that point.
//!
//! - the lips are the part of their regions **inside the vermilion border**, an outline through
//!   named landmarks ([`LIP_OUTLINE`]: `R.lip.corner`, `lip.upper.outer.NN`, `L.lip.corner`,
//!   `lip.lower.outer.NN`): the lip regions also hold the perioral skin up to the nose and the chin,
//!   which must not take the lip colour.
//!
//! A zone is first a hard vertex set on the welded surface (UV seams do not cut it). Its edge is then
//! feathered by an approximate geodesic distance to the set's boundary (multi-source Dijkstra over the
//! one-ring graph, seeded where the edges cross the boundary: at their midpoints, or for an outlined
//! zone where the outline crosses them): `s` is that signed distance in metres of surface (positive
//! inside) and the weight is `smoothstep(0.5 + s / feather)`, so the mask is 0.5 on the zone border
//! and reaches 0 and 1 half a feather width away. The rasteriser interpolates `s` (not the weight)
//! barycentrically to every texel centre of a UV tile and applies the smoothstep there, so a feather
//! narrower than the mesh edges stays that narrow on the texture; the tiles are then dilated a few
//! texels past the island borders so bilinear filtering at seams does not pull in black.
//!
//! The feather widths, cap radii, facings and the region choices are **artistic defaults** (a starting look
//! to tune against reference renders), not measured data. Baking is editor-time work.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use humanart_math::{Vec2, Vec3};

use super::error::SkinError;
use super::zones::{ZONE_SLOTS, Zone};
use crate::mesh::{Mesh, Triangulation};
use crate::region::RegionMap;
use crate::template::Template;
use crate::topology::Csr;

/// Largest mask tile edge in texels (three 4096² RGBA8 layers are 192 MiB).
pub const MAX_MASK_SIZE: u32 = 4096;
/// Most dilation passes past the island borders.
pub const MAX_DILATE: u32 = 64;
/// Mask layers per tile (four zone slots each).
pub const MASK_LAYERS: usize = ZONE_SLOTS / 4;
/// How far past half a feather width the signed distance is tracked, metres: more than the longest
/// template edge, so the distance interpolated across a triangle at the zone edge is not clamped.
pub const DISTANCE_MARGIN: f32 = 0.03;
/// Most landmarks in one zone outline.
pub const MAX_OUTLINE_POINTS: usize = 1024;

/// Where one zone comes from on a template.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZoneSource {
    /// The zone.
    pub zone: Zone,
    /// Regions whose whole subtree is in the zone.
    pub regions: &'static [&'static str],
    /// Joints whose rest position starts the ray that finds a cap.
    pub joints: &'static [&'static str],
    /// Which way the ray from each joint points.
    pub facing: Facing,
    /// Cap radius around the ray's exit point, metres.
    pub cap_radius: f32,
    /// Regions (subtrees) the caps are found in and confined to; empty = anywhere.
    pub within: &'static [&'static str],
    /// Feather width across the zone edge, metres of surface.
    pub feather: f32,
    /// A closed outline through named landmarks that clips the region part of the zone (empty: no
    /// clip). The region vertices whose projection along the outline's normal falls inside it are
    /// the zone, and the zone edge lies on the outline.
    pub outline: &'static [OutlineRun],
}

/// One run of a zone outline: a landmark, or a numbered series of landmarks walked in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutlineRun {
    /// A landmark name, or the stem of a numbered series: `lip.upper.outer` stands for
    /// `lip.upper.outer.01`, `lip.upper.outer.02`, ... in numeric order. An exact name wins.
    pub name: &'static str,
    /// Walk the numbered series from the highest number down.
    pub reverse: bool,
}

/// The vermilion border of the template, closed: right corner, upper border right to left, left
/// corner, lower border left to right (stable landmark names of `generate::landmarks`).
pub const LIP_OUTLINE: [OutlineRun; 4] = [
    OutlineRun { name: "R.lip.corner", reverse: false },
    OutlineRun { name: "lip.upper.outer", reverse: false },
    OutlineRun { name: "L.lip.corner", reverse: false },
    OutlineRun { name: "lip.lower.outer", reverse: true },
];

/// The direction of a cap ray in the rest pose (right-handed, Y up, the character faces +Z).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Facing {
    /// +Z (knee caps).
    Front,
    /// -Z (elbows: in the rest pose the palms face the body, so the olecranon points back).
    Back,
    /// Away from the midline along X (the back of the hand: the palms face the body).
    Lateral,
}

impl Facing {
    /// The ray direction for a joint at `at`.
    pub fn direction(self, at: Vec3) -> Vec3 {
        match self {
            Self::Front => Vec3::new(0.0, 0.0, 1.0),
            Self::Back => Vec3::new(0.0, 0.0, -1.0),
            Self::Lateral => Vec3::new(if at.x < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0),
        }
    }
}

const fn regions(zone: Zone, regions: &'static [&'static str], feather: f32) -> ZoneSource {
    ZoneSource {
        zone,
        regions,
        joints: &[],
        facing: Facing::Front,
        cap_radius: 0.0,
        within: &[],
        feather,
        outline: &[],
    }
}

const fn outlined(
    zone: Zone,
    regions: &'static [&'static str],
    outline: &'static [OutlineRun],
    feather: f32,
) -> ZoneSource {
    ZoneSource { zone, regions, joints: &[], facing: Facing::Front, cap_radius: 0.0, within: &[], feather, outline }
}

const fn caps(
    zone: Zone,
    joints: &'static [&'static str],
    facing: Facing,
    cap_radius: f32,
    within: &'static [&'static str],
    feather: f32,
) -> ZoneSource {
    ZoneSource { zone, regions: &[], joints, facing, cap_radius, within, feather, outline: &[] }
}

/// The zones of the generated template, by stable region and joint names (artistic defaults).
pub const TEMPLATE_ZONES: [ZoneSource; 12] = [
    regions(Zone::Cheeks, &["cheek-L", "cheek-R", "malar-L", "malar-R"], 0.025),
    regions(Zone::Nose, &["nose"], 0.010),
    regions(Zone::UnderEye, &["lower-lid-L", "lower-lid-R"], 0.012),
    // The philtrum region reaches down to the Cupid's bow, so it is a candidate too; the outline
    // keeps only the vermilion. 2 mm: the vermilion border is a sharp line (the white roll).
    outlined(Zone::Lips, &["upper-lip", "lower-lip", "lip-corner-L", "lip-corner-R", "philtrum"], &LIP_OUTLINE, 0.002),
    regions(Zone::Ears, &["ear-L", "ear-R"], 0.006),
    caps(
        Zone::Knuckles,
        &[
            "L.index.01",
            "L.middle.01",
            "L.ring.01",
            "L.little.01",
            "L.index.02",
            "L.middle.02",
            "L.ring.02",
            "L.little.02",
            "R.index.01",
            "R.middle.01",
            "R.ring.01",
            "R.little.01",
            "R.index.02",
            "R.middle.02",
            "R.ring.02",
            "R.little.02",
        ],
        Facing::Lateral,
        0.007,
        &["hand-L", "hand-R"],
        0.006,
    ),
    caps(
        Zone::Elbows,
        &["L.forearm", "R.forearm"],
        Facing::Back,
        0.025,
        &["upper-arm-L", "upper-arm-R", "forearm-L", "forearm-R"],
        0.020,
    ),
    caps(Zone::Knees, &["L.shin", "R.shin"], Facing::Front, 0.035, &["thigh-L", "thigh-R", "shin-L", "shin-R"], 0.025),
    regions(Zone::Temples, &["temple-L", "temple-R"], 0.020),
    regions(Zone::Forehead, &["forehead"], 0.020),
    regions(Zone::Chin, &["chin"], 0.012),
    regions(Zone::Eyelids, &["upper-lid-L", "upper-lid-R"], 0.006),
];

/// Names a template did not have (the zone parts they would have made are empty).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BakeReport {
    /// Region names not found.
    pub missing_regions: Vec<&'static str>,
    /// Joint names not found.
    pub missing_joints: Vec<&'static str>,
    /// Outline runs that matched no landmark (their zone is left empty: without its outline it
    /// would spread over the whole region).
    pub missing_landmarks: Vec<&'static str>,
}

impl BakeReport {
    /// Whether every name was found.
    pub fn complete(&self) -> bool {
        self.missing_regions.is_empty() && self.missing_joints.is_empty() && self.missing_landmarks.is_empty()
    }
}

/// Soft zone weights per render vertex, zone-major (structure of arrays), with the signed surface
/// distances they come from.
#[derive(Clone, Debug, PartialEq)]
pub struct ZoneWeights {
    vertex_count: usize,
    weights: Vec<f32>,
    distances: Vec<f32>,
    feathers: [f32; ZONE_SLOTS],
}

impl ZoneWeights {
    /// All-zero weights for `vertex_count` vertices.
    pub fn new(vertex_count: usize) -> Self {
        Self {
            vertex_count,
            weights: vec![0.0; vertex_count * ZONE_SLOTS],
            distances: vec![0.0; vertex_count * ZONE_SLOTS],
            feathers: [0.0; ZONE_SLOTS],
        }
    }

    /// The signed surface distance to the edge of zone slot `slot` per vertex, metres, positive
    /// inside and clamped to half the feather plus [`DISTANCE_MARGIN`] (empty for a slot out of
    /// range; all zero for a slot that was not baked from distances).
    pub fn distances(&self, slot: usize) -> &[f32] {
        self.distances.get(slot * self.vertex_count..(slot + 1) * self.vertex_count).unwrap_or(&[])
    }

    /// The feather width of zone slot `slot`, metres; 0 when the slot holds plain weights.
    pub fn feather(&self, slot: usize) -> f32 {
        self.feathers.get(slot).copied().unwrap_or(0.0)
    }

    /// Sets plain weights (no feather, no distances) for zone slot `slot`, one per vertex
    /// (clamped to 0..1; non-finite values become 0). False when the slot is out of range or
    /// `values` does not hold one weight per vertex.
    pub fn set_plain(&mut self, slot: usize, values: &[f32]) -> bool {
        if slot >= ZONE_SLOTS || values.len() != self.vertex_count {
            return false;
        }
        let range = slot * self.vertex_count..(slot + 1) * self.vertex_count;
        let (Some(w), Some(d)) = (self.weights.get_mut(range.clone()), self.distances.get_mut(range)) else {
            return false;
        };
        for (out, &v) in w.iter_mut().zip(values) {
            *out = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
        }
        d.fill(0.0);
        self.feathers[slot] = 0.0;
        true
    }

    /// How many vertices there are.
    pub const fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// The weights of one zone slot (empty for a slot out of range).
    pub fn slot(&self, slot: usize) -> &[f32] {
        self.weights.get(slot * self.vertex_count..(slot + 1) * self.vertex_count).unwrap_or(&[])
    }

    /// The weights of one zone.
    pub fn zone(&self, zone: Zone) -> &[f32] {
        self.slot(zone.slot())
    }
}

/// Computes the soft weight of every zone of `sources` at every vertex of the template's LOD0 mesh.
///
/// # Errors
/// [`SkinError::Invalid`] when the region map does not match the mesh, a feather or cap radius is
/// not usable, or an outline is degenerate; [`SkinError::Limit`] for an outline of more than
/// [`MAX_OUTLINE_POINTS`] landmarks.
pub fn vertex_weights(template: &Template, sources: &[ZoneSource]) -> Result<(ZoneWeights, BakeReport), SkinError> {
    let mesh = template.mesh();
    let regions = template.regions();
    let n = mesh.vertex_count();
    if regions.vertex_regions.len() != n {
        return Err(SkinError::Invalid { what: "the region map does not match the mesh" });
    }
    let topology = mesh.topology();
    let keys = mesh.position_keys();
    let positions = mesh.positions();
    let rest = template.skeleton().rest_world();
    let mut report = BakeReport::default();
    let mut weights = ZoneWeights::new(n);
    let mut inside = vec![false; n];
    let mut crossing = vec![f32::NAN; n];
    for source in sources {
        let feather_ok = source.feather.is_finite() && source.feather > 0.0;
        let cap_ok = source.cap_radius.is_finite() && source.cap_radius >= 0.0;
        if !(feather_ok && cap_ok) {
            return Err(SkinError::Invalid { what: "zone feather must be positive and the cap radius finite" });
        }
        inside.fill(false);
        crossing.fill(f32::NAN);
        let zone_regions = resolve_regions(regions, source.regions, &mut report.missing_regions);
        let within = resolve_regions(regions, source.within, &mut report.missing_regions);
        let in_within: Vec<bool> = regions
            .vertex_regions
            .iter()
            .map(|&region| within.is_empty() || within.iter().any(|&r| regions.is_within(region, r)))
            .collect();
        let mut centres: Vec<Vec3> = Vec::with_capacity(source.joints.len());
        for &name in source.joints {
            match template.skeleton().find(name).and_then(|j| rest.get(j as usize)) {
                Some(t) => {
                    let at = t.translation;
                    centres.extend(ray_exit(positions, &in_within, at, source.facing.direction(at)));
                }
                None => report.missing_joints.push(name),
            }
        }
        let outline = if source.outline.is_empty() {
            None
        } else {
            // A broken outline leaves the zone empty rather than spread over its whole regions.
            let Some(points) = outline_points(template, source.outline, &mut report.missing_landmarks)? else {
                continue;
            };
            Some(Outline::new(&points)?)
        };
        let r2 = source.cap_radius * source.cap_radius;
        for (v, flag) in inside.iter_mut().enumerate() {
            let in_region = zone_regions.iter().any(|&r| regions.is_within(regions.vertex_regions[v], r));
            let in_cap = in_within[v] && centres.iter().any(|c| positions[v].distance_squared(*c) <= r2);
            let in_outline = match &outline {
                Some(o) if in_region => {
                    let d = o.signed_distance(positions[v]);
                    crossing[v] = d;
                    d >= 0.0
                }
                _ => true,
            };
            *flag = (in_region && in_outline) || in_cap;
        }
        let hint = outline.as_ref().map(|_| crossing.as_slice());
        let reach = 0.5 * source.feather + DISTANCE_MARGIN;
        let distances = signed_distances(positions, keys, &topology.neighbors, &inside, hint, reach)?;
        let slot = source.zone.slot();
        weights.feathers[slot] = source.feather;
        let base = slot * n;
        for (i, &s) in distances.iter().enumerate() {
            weights.weights[base + i] = smoothstep(0.5 + s / source.feather);
            weights.distances[base + i] = s;
        }
    }
    report.missing_regions.sort_unstable();
    report.missing_regions.dedup();
    report.missing_joints.sort_unstable();
    report.missing_joints.dedup();
    report.missing_landmarks.sort_unstable();
    report.missing_landmarks.dedup();
    Ok((weights, report))
}

/// The positions of an outline's landmarks in walking order; `None` (with the run names added to
/// `missing`) when a run matches no landmark or fewer than three points result.
fn outline_points(
    template: &Template,
    runs: &[OutlineRun],
    missing: &mut Vec<&'static str>,
) -> Result<Option<Vec<Vec3>>, SkinError> {
    let landmarks = template.landmarks();
    let positions = template.mesh().positions();
    let mut points = Vec::new();
    let mut complete = true;
    for run in runs {
        let mut series: Vec<(u32, Vec3)> = Vec::new();
        if let Some(exact) = landmarks.iter().find(|l| l.name == run.name) {
            series.extend(exact.position(positions).map(|p| (0, p)));
        } else {
            for landmark in landmarks {
                let number = landmark
                    .name
                    .strip_prefix(run.name)
                    .and_then(|rest| rest.strip_prefix('.'))
                    .filter(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
                    .and_then(|digits| digits.parse::<u32>().ok());
                if let (Some(k), Some(p)) = (number, landmark.position(positions)) {
                    series.push((k, p));
                }
            }
            series.sort_by_key(|&(k, _)| k);
            if run.reverse {
                series.reverse();
            }
        }
        if series.is_empty() {
            missing.push(run.name);
            complete = false;
        }
        let total = points.len() + series.len();
        if total > MAX_OUTLINE_POINTS {
            return Err(SkinError::limit("zone outline points", MAX_OUTLINE_POINTS as u64, total as u64));
        }
        points.extend(series.into_iter().map(|(_, p)| p));
    }
    Ok((complete && points.len() >= 3).then_some(points))
}

/// A closed outline projected onto its own best-fit plane (Newell normal).
#[derive(Clone, Debug)]
struct Outline {
    origin: Vec3,
    u: Vec3,
    v: Vec3,
    points: Vec<Vec2>,
}

impl Outline {
    fn new(points: &[Vec3]) -> Result<Self, SkinError> {
        let count = points.len().max(1) as f32;
        let origin = points.iter().fold(Vec3::ZERO, |a, p| a + *p) / count;
        let mut normal = Vec3::ZERO;
        for (i, p) in points.iter().enumerate() {
            let q = points.get(i + 1).or(points.first()).copied().unwrap_or(*p);
            normal += (*p - origin).cross(q - origin);
        }
        let length = normal.length();
        if !(length.is_finite() && length > 1e-12) {
            return Err(SkinError::Invalid { what: "zone outline is degenerate" });
        }
        let normal = normal / length;
        let helper = if normal.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
        let u = helper.cross(normal).normalize();
        let v = normal.cross(u);
        let points = points.iter().map(|p| Vec2::new((*p - origin).dot(u), (*p - origin).dot(v))).collect();
        Ok(Self { origin, u, v, points })
    }

    /// The signed distance from the projection of `p` to the outline, positive inside.
    fn signed_distance(&self, p: Vec3) -> f32 {
        let q = Vec2::new((p - self.origin).dot(self.u), (p - self.origin).dot(self.v));
        let mut inside = false;
        let mut nearest = f32::INFINITY;
        for (i, &a) in self.points.iter().enumerate() {
            let b = self.points.get(i + 1).or(self.points.first()).copied().unwrap_or(a);
            if (a.y > q.y) != (b.y > q.y) {
                let x = a.x + (q.y - a.y) / (b.y - a.y) * (b.x - a.x);
                if q.x < x {
                    inside = !inside;
                }
            }
            let ab = b - a;
            let len2 = ab.dot(ab);
            let t = if len2 > 0.0 { ((q - a).dot(ab) / len2).clamp(0.0, 1.0) } else { 0.0 };
            nearest = nearest.min((a + ab * t - q).length());
        }
        if inside { nearest } else { -nearest }
    }
}

/// The vertex (among `allowed`) where a ray from `from` along `dir` leaves the skin: the one closest
/// to the ray on its forward side.
fn ray_exit(positions: &[Vec3], allowed: &[bool], from: Vec3, dir: Vec3) -> Option<Vec3> {
    let mut best: Option<(f32, Vec3)> = None;
    for (p, _) in positions.iter().zip(allowed).filter(|(_, a)| **a) {
        let d = *p - from;
        let t = d.dot(dir);
        if t <= 0.0 {
            continue;
        }
        let off = (d - dir * t).length_squared();
        if best.is_none_or(|(b, _)| off < b) {
            best = Some((off, *p));
        }
    }
    best.map(|(_, p)| p)
}

fn resolve_regions(map: &RegionMap, names: &[&'static str], missing: &mut Vec<&'static str>) -> Vec<u16> {
    let mut out = Vec::with_capacity(names.len());
    for &name in names {
        match map.find(name) {
            Some(r) => out.push(r),
            None => missing.push(name),
        }
    }
    out
}

/// A heap entry ordered by smallest distance first.
#[derive(Clone, Copy, PartialEq)]
struct Entry(f32, u32);

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0).then(other.1.cmp(&self.1))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Feathers a hard vertex set: per render vertex, `smoothstep(0.5 + s / feather)` with `s` the
/// signed distance to the set's boundary along the welded one-ring graph (positive inside).
/// `keys` maps every render vertex to its weld key; `inside` is read at weld keys.
///
/// # Errors
/// [`SkinError::Invalid`] when the arrays do not have one entry per vertex or a key is out of range.
pub fn feathered(
    positions: &[Vec3],
    keys: &[u32],
    neighbors: &Csr,
    inside: &[bool],
    feather: f32,
) -> Result<Vec<f32>, SkinError> {
    if !(feather.is_finite() && feather > 0.0) {
        return Err(SkinError::Invalid { what: "feather must be positive" });
    }
    let s = signed_distances(positions, keys, neighbors, inside, None, 0.5 * feather)?;
    Ok(s.into_iter().map(|s| smoothstep(0.5 + s / feather)).collect())
}

/// The signed distance (positive inside, clamped to `reach`) from every render vertex to the
/// boundary of a hard vertex set along the welded one-ring graph. An edge that crosses the boundary
/// is cut at its midpoint, or, where both ends carry a finite `crossing` value (a signed distance to
/// an outline, positive inside), where that value interpolates to zero.
fn signed_distances(
    positions: &[Vec3],
    keys: &[u32],
    neighbors: &Csr,
    inside: &[bool],
    crossing: Option<&[f32]>,
    reach: f32,
) -> Result<Vec<f32>, SkinError> {
    let n = positions.len();
    if keys.len() != n
        || inside.len() != n
        || neighbors.rows() != n
        || crossing.is_some_and(|c| c.len() != n)
        || keys.iter().any(|&k| k as usize >= n)
    {
        return Err(SkinError::Invalid { what: "feathering needs one position, key and flag per vertex" });
    }
    if !(reach.is_finite() && reach > 0.0) {
        return Err(SkinError::Invalid { what: "feather must be positive" });
    }
    let mut dist = vec![f32::INFINITY; n];
    let mut heap = BinaryHeap::new();
    // Seeds: both ends of every edge that crosses the boundary, at their distance to the cut.
    for a in 0..n {
        for &b in neighbors.row(a) {
            let b = b as usize;
            if b < n && inside[a] != inside[b] {
                let length = positions[a].distance(positions[b]);
                let t = crossing
                    .map(|c| (c[a], c[b]))
                    .filter(|(ca, cb)| ca.is_finite() && cb.is_finite() && ca != cb)
                    .map_or(0.5, |(ca, cb)| (ca / (ca - cb)).clamp(0.0, 1.0));
                let d = t * length;
                if d < dist[a] {
                    dist[a] = d;
                    heap.push(Entry(d, a as u32));
                }
            }
        }
    }
    while let Some(Entry(d, a)) = heap.pop() {
        let a = a as usize;
        if d > dist[a] || d > reach {
            continue;
        }
        for &b in neighbors.row(a) {
            let b = b as usize;
            // The distance is to the boundary of the set, so it only propagates on one side.
            if b >= n || inside[b] != inside[a] {
                continue;
            }
            let nd = d + positions[a].distance(positions[b]);
            if nd < dist[b] {
                dist[b] = nd;
                heap.push(Entry(nd, b as u32));
            }
        }
    }
    Ok(keys
        .iter()
        .map(|&k| {
            let k = k as usize;
            if inside[k] { dist[k].min(reach) } else { -dist[k].min(reach) }
        })
        .collect())
}

fn smoothstep(x: f32) -> f32 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The zone mask layers of one UV tile: [`MASK_LAYERS`] RGBA8 images of `size²` texels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZoneMasks {
    size: u32,
    layers: [Vec<u8>; MASK_LAYERS],
}

impl ZoneMasks {
    /// Edge length in texels.
    pub const fn size(&self) -> u32 {
        self.size
    }

    /// The layers, in the order the compositor's `upload_inputs` takes them.
    pub fn layers(&self) -> [&[u8]; MASK_LAYERS] {
        [&self.layers[0], &self.layers[1], &self.layers[2]]
    }

    /// The weight of zone slot `slot` at texel `(x, y)` (0 outside).
    pub fn weight(&self, slot: usize, x: u32, y: u32) -> f32 {
        if x >= self.size || y >= self.size {
            return 0.0;
        }
        let i = (y as usize * self.size as usize + x as usize) * 4 + slot % 4;
        self.layers.get(slot / 4).and_then(|l| l.get(i)).map_or(0.0, |&b| f32::from(b) / 255.0)
    }
}

/// Rasterises the vertex weights of the triangles on UV tile `tile` into `size²` mask texels
/// (texel `(x, y)` samples UV `((x + 0.5) / size, (y + 0.5) / size)`), then dilates `dilate` texels
/// past the islands.
///
/// # Errors
/// [`SkinError::Limit`] for a size above [`MAX_MASK_SIZE`] or a dilation above [`MAX_DILATE`];
/// [`SkinError::Invalid`] when the weights or triangles do not fit the mesh.
pub fn bake_tile(
    mesh: &Mesh,
    triangles: &Triangulation,
    weights: &ZoneWeights,
    tile: u16,
    size: u32,
    dilate: u32,
) -> Result<ZoneMasks, SkinError> {
    rasterize(mesh.uvs(), mesh.face_tiles(), triangles, weights, tile, size, dilate)
}

fn rasterize(
    uvs: &[Vec2],
    face_tiles: &[u16],
    triangles: &Triangulation,
    weights: &ZoneWeights,
    tile: u16,
    size: u32,
    dilate: u32,
) -> Result<ZoneMasks, SkinError> {
    if size == 0 || size > MAX_MASK_SIZE {
        return Err(SkinError::limit("zone mask size", u64::from(MAX_MASK_SIZE), u64::from(size)));
    }
    if dilate > MAX_DILATE {
        return Err(SkinError::limit("zone mask dilation", u64::from(MAX_DILATE), u64::from(dilate)));
    }
    if weights.vertex_count() != uvs.len() || triangles.indices.len() != 3 * triangles.triangle_faces.len() {
        return Err(SkinError::Invalid { what: "zone weights and triangles must match the mesh" });
    }
    let s = size as usize;
    let texels = s * s;
    let mut layers: [Vec<u8>; MASK_LAYERS] = std::array::from_fn(|_| vec![0; texels * 4]);
    let mut covered = vec![false; texels];
    let scale = size as f32;
    for t in 0..triangles.len() {
        let (Some(corners), Some(&face)) = (triangles.triangle(t), triangles.triangle_faces.get(t)) else { continue };
        if face_tiles.get(face as usize) != Some(&tile) {
            continue;
        }
        let Some(p) = corners.iter().map(|&v| uvs.get(v as usize).map(|uv| *uv * scale)).collect::<Option<Vec<Vec2>>>()
        else {
            return Err(SkinError::Invalid { what: "a triangle corner is not a vertex" });
        };
        let area = edge(p[0], p[1], p[2]);
        if !area.is_finite() || area.abs() < 1e-12 {
            continue;
        }
        let lo = |a: f32, b: f32, c: f32| (a.min(b).min(c) - 0.5).floor().clamp(0.0, scale - 1.0) as usize;
        let hi = |a: f32, b: f32, c: f32| (a.max(b).max(c) - 0.5).ceil().clamp(0.0, scale - 1.0) as usize;
        let (x0, x1) = (lo(p[0].x, p[1].x, p[2].x), hi(p[0].x, p[1].x, p[2].x));
        let (y0, y1) = (lo(p[0].y, p[1].y, p[2].y), hi(p[0].y, p[1].y, p[2].y));
        // Small tolerance so texel centres on a shared edge are covered by one of its triangles.
        let tol = -1e-4 * area.abs();
        for y in y0..=y1 {
            for x in x0..=x1 {
                let c = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let b0 = edge(p[1], p[2], c) * area.signum();
                let b1 = edge(p[2], p[0], c) * area.signum();
                let b2 = edge(p[0], p[1], c) * area.signum();
                if b0 < tol || b1 < tol || b2 < tol {
                    continue;
                }
                let inv = 1.0 / area.abs();
                let (l0, l1, l2) = (b0 * inv, b1 * inv, b2 * inv);
                let i = y * s + x;
                covered[i] = true;
                for slot in 0..ZONE_SLOTS {
                    // Distance-baked slots interpolate the signed distance and feather per texel.
                    let feather = weights.feather(slot);
                    let w = if feather > 0.0 { weights.distances(slot) } else { weights.slot(slot) };
                    let value = l0 * w[corners[0] as usize] + l1 * w[corners[1] as usize] + l2 * w[corners[2] as usize];
                    let value = if feather > 0.0 { smoothstep(0.5 + value / feather) } else { value };
                    layers[slot / 4][i * 4 + slot % 4] = quantize(value);
                }
            }
        }
    }
    dilate_layers(&mut layers, &mut covered, s, dilate);
    Ok(ZoneMasks { size, layers })
}

fn edge(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn quantize(w: f32) -> u8 {
    // The clamp keeps the value in 0..=255, so the cast cannot truncate.
    (w.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Grows the covered texels by one ring per pass; a new texel takes the mean of its covered
/// 4-neighbours.
fn dilate_layers(layers: &mut [Vec<u8>; MASK_LAYERS], covered: &mut [bool], s: usize, passes: u32) {
    let mut grown: Vec<usize> = Vec::new();
    for _ in 0..passes {
        grown.clear();
        for y in 0..s {
            for x in 0..s {
                let i = y * s + x;
                if covered[i] {
                    continue;
                }
                let mut sum = [0_u32; ZONE_SLOTS];
                let mut count = 0_u32;
                let neighbours = [
                    (x > 0).then(|| i - 1),
                    (x + 1 < s).then(|| i + 1),
                    (y > 0).then(|| i - s),
                    (y + 1 < s).then(|| i + s),
                ];
                for j in neighbours.into_iter().flatten() {
                    if covered[j] {
                        count += 1;
                        for (slot, acc) in sum.iter_mut().enumerate() {
                            *acc += u32::from(layers[slot / 4][j * 4 + slot % 4]);
                        }
                    }
                }
                if count > 0 {
                    for (slot, acc) in sum.iter().enumerate() {
                        // A rounded mean of u8 values fits in u8.
                        let mean = (acc + count / 2).checked_div(count).unwrap_or(0);
                        layers[slot / 4][i * 4 + slot % 4] = u8::try_from(mean).unwrap_or(u8::MAX);
                    }
                    grown.push(i);
                }
            }
        }
        if grown.is_empty() {
            break;
        }
        for &i in &grown {
            covered[i] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::build;

    /// A strip of `n` unit squares along X (two rows of vertices), 1 m apart, on one tile.
    fn strip(n: usize) -> (Vec<Vec3>, Vec<u32>, Csr) {
        let mut positions = Vec::new();
        for row in 0..2 {
            for i in 0..=n {
                positions.push(Vec3::new(i as f32, row as f32, 0.0));
            }
        }
        let m = n + 1;
        let mut pairs = Vec::new();
        for i in 0..m {
            if i + 1 < m {
                pairs.extend([(i, i + 1), (i + 1, i), (m + i, m + i + 1), (m + i + 1, m + i)]);
            }
            pairs.extend([(i, m + i), (m + i, i)]);
        }
        let pairs: Vec<(u32, u32)> = pairs.into_iter().map(|(a, b)| (a as u32, b as u32)).collect();
        let keys = (0..positions.len() as u32).collect();
        (positions.clone(), keys, Csr::from_pairs(positions.len(), &pairs))
    }

    #[test]
    fn skn11_feathering_is_half_on_the_border_and_saturates() {
        let (positions, keys, neighbors) = strip(20);
        // Columns 0..=9 inside, 10..=20 outside: the border lies at x = 9.5.
        let inside: Vec<bool> = positions.iter().map(|p| p.x < 9.75).collect();
        let w = feathered(&positions, &keys, &neighbors, &inside, 4.0).unwrap();
        let at = |x: usize| w[x];
        // s = +0.5 at x = 9 and -0.5 at x = 10: smoothstep(0.625) and smoothstep(0.375).
        assert!((at(9) - smoothstep(0.625)).abs() < 1e-6, "{}", at(9));
        assert!((at(10) - smoothstep(0.375)).abs() < 1e-6, "{}", at(10));
        assert!((at(9) + at(10) - 1.0).abs() < 1e-6);
        // Two metres (half the feather) from the border the mask is saturated.
        assert_eq!(at(7), 1.0);
        assert_eq!(at(12), 0.0);
        assert_eq!(at(0), 1.0);
        assert_eq!(at(20), 0.0);
        // Monotone across the border.
        assert!((0..20).all(|x| at(x) >= at(x + 1)));
    }

    #[test]
    fn feathering_follows_weld_keys_across_seams() {
        let (mut positions, mut keys, neighbors) = strip(4);
        // A seam copy of vertex 4 (outside) that has no edges of its own.
        positions.push(positions[4]);
        keys.push(4);
        let neighbors =
            Csr::from_rows((0..positions.len()).map(|r| if r < neighbors.rows() { neighbors.row(r) } else { &[][..] }));
        let mut inside: Vec<bool> = positions.iter().map(|p| p.x < 1.5).collect();
        inside[10] = false;
        let w = feathered(&positions, &keys, &neighbors, &inside, 2.0).unwrap();
        assert_eq!(w[10], w[4]);
        assert!(feathered(&positions, &keys, &neighbors, &inside[..3], 2.0).is_err());
        assert!(feathered(&positions, &keys, &neighbors, &inside, 0.0).is_err());
    }

    fn quad_triangles() -> Triangulation {
        Triangulation { indices: vec![0, 1, 2, 0, 2, 3], triangle_faces: vec![0, 0] }
    }

    #[test]
    fn skn11_raster_interpolates_and_dilates() {
        // A square island over the middle half of the tile, weight ramping with U in slot 5.
        let uvs = vec![Vec2::new(0.25, 0.25), Vec2::new(0.75, 0.25), Vec2::new(0.75, 0.75), Vec2::new(0.25, 0.75)];
        let mut weights = ZoneWeights::new(4);
        weights.weights[5 * 4..6 * 4].copy_from_slice(&[0.0, 1.0, 1.0, 0.0]);
        weights.weights[..4].copy_from_slice(&[1.0; 4]);
        let masks = rasterize(&uvs, &[0], &quad_triangles(), &weights, 0, 64, 4).unwrap();
        assert_eq!(masks.layers().map(<[u8]>::len), [64 * 64 * 4; 3]);
        // Texel 16 has its centre at u = 16.5/64, just inside the island; texel 47 at 47.5/64.
        let expect = |x: u32| ((x as f32 + 0.5) / 64.0 - 0.25) / 0.5;
        for x in [16, 24, 32, 40, 47] {
            assert!((masks.weight(5, x, 32) - expect(x)).abs() <= 0.5 / 255.0 + 1e-6, "x {x}");
            assert_eq!(masks.weight(0, x, 32), 1.0);
        }
        // Slot 5 is layer 1, channel 1.
        assert_eq!(masks.layers()[1][(32 * 64 + 47) * 4 + 1], quantize(expect(47)));
        // Dilated 4 texels out, nothing beyond.
        assert_eq!(masks.weight(0, 12, 32), 1.0);
        assert_eq!(masks.weight(0, 11, 32), 0.0);
        assert_eq!(masks.weight(0, 51, 32), 1.0);
        assert_eq!(masks.weight(0, 52, 32), 0.0);
        // Other tiles see nothing; limits are enforced.
        let other = rasterize(&uvs, &[0], &quad_triangles(), &weights, 1, 64, 0).unwrap();
        assert!(other.layers().iter().all(|l| l.iter().all(|&b| b == 0)));
        assert!(rasterize(&uvs, &[0], &quad_triangles(), &weights, 0, MAX_MASK_SIZE + 1, 0).is_err());
        assert!(rasterize(&uvs, &[0], &quad_triangles(), &weights, 0, 64, MAX_DILATE + 1).is_err());
        assert!(rasterize(&uvs[..3], &[0], &quad_triangles(), &weights, 0, 64, 0).is_err());
    }

    /// `[SKN-11]` on the generated template: every zone name resolves, every zone is non-empty and
    /// sits where it should (cheeks on both sides of the face, knees low, ears off the midline),
    /// and the baked tile masks carry the weights.
    #[test]
    fn skn11_zone_masks_bake_from_the_generated_template() {
        let generated = build::template(&build::Options { lods: false, ..build::Options::default() }).unwrap();
        let template = &generated.template;
        let (weights, report) = vertex_weights(template, &TEMPLATE_ZONES).unwrap();
        assert!(report.complete(), "{report:?}");
        let positions = template.mesh().positions();
        let centroid = |zone: Zone| {
            let w = weights.zone(zone);
            let total: f32 = w.iter().sum();
            let mut c = Vec3::ZERO;
            for (p, &wi) in positions.iter().zip(w) {
                c += *p * wi;
            }
            (c / total.max(1e-6), total, w.iter().filter(|&&x| x > 0.99).count())
        };
        let (head, _, _) = centroid(Zone::Forehead);
        for zone in Zone::ALL {
            let (c, total, full) = centroid(zone);
            assert!(total > 1.0 && full > 0, "{zone:?} is empty");
            match zone {
                Zone::Knees => assert!(c.y < 0.7, "knees at {c:?}"),
                Zone::Elbows => assert!(c.y > 0.8 && c.y < head.y, "elbows at {c:?}"),
                Zone::Knuckles | Zone::Ears | Zone::Temples | Zone::Cheeks | Zone::Chin | Zone::Eyelids => {
                    assert!(c.x.abs() < 0.02, "{zone:?} symmetric");
                }
                _ => {}
            }
        }
        // Knee caps are in front of the knee joints, elbow caps behind the elbows, knuckles on the
        // lateral (dorsal) side of the hands.
        let rest = template.skeleton().rest_world();
        let joint = |name: &str| rest[template.skeleton().find(name).unwrap() as usize].translation;
        let side_centroid = |zone: Zone, left: bool| {
            let w = weights.zone(zone);
            let (mut c, mut t) = (Vec3::ZERO, 0.0);
            for (p, &wi) in positions.iter().zip(w) {
                if (p.x > 0.0) == left {
                    c += *p * wi;
                    t += wi;
                }
            }
            c / t
        };
        for (left, l) in [(true, "L"), (false, "R")] {
            assert!(side_centroid(Zone::Knees, left).z > joint(&format!("{l}.shin")).z + 0.02);
            assert!(side_centroid(Zone::Elbows, left).z < joint(&format!("{l}.forearm")).z - 0.01);
            let k = side_centroid(Zone::Knuckles, left);
            assert!(k.x.abs() > joint(&format!("{l}.middle.01")).x.abs(), "{k:?}");
        }
        // Lips are under the nose, the nose under the forehead, the face zones face forward (+Z).
        let (lips, _, _) = centroid(Zone::Lips);
        let (nose, _, _) = centroid(Zone::Nose);
        assert!(lips.y < nose.y && nose.y < head.y);
        assert!(nose.z > head.z);
        // Each side carries cheek weight; the midline of the nose tip carries none.
        let w = weights.zone(Zone::Cheeks);
        let left: f32 = positions.iter().zip(w).filter(|(p, _)| p.x > 0.0).map(|(_, &x)| x).sum();
        let right: f32 = positions.iter().zip(w).filter(|(p, _)| p.x < 0.0).map(|(_, &x)| x).sum();
        assert!((left - right).abs() < 0.1 * (left + right), "{left} vs {right}");
        // Feathering: weights between 0 and 1 exist (soft edges), and lips (3 mm feather) are
        // sharper than cheeks (25 mm).
        let soft = |zone: Zone| weights.zone(zone).iter().filter(|&&x| x > 0.05 && x < 0.95).count();
        let hard = |zone: Zone| weights.zone(zone).iter().filter(|&&x| x >= 0.95).count();
        assert!(soft(Zone::Cheeks) > 0 && soft(Zone::Lips) > 0);
        assert!(
            (soft(Zone::Lips) as f32 / hard(Zone::Lips) as f32)
                < (soft(Zone::Cheeks) as f32 / hard(Zone::Cheeks) as f32)
        );
        // Bake every tile at 256²: each zone shows up on some tile.
        let triangles = template.mesh().triangulate();
        let mut seen = [0_usize; ZONE_SLOTS];
        for tile in 0..template.tiles().len() {
            let masks = bake_tile(template.mesh(), &triangles, &weights, tile as u16, 256, 4).unwrap();
            for zone in Zone::ALL {
                for y in 0..256 {
                    for x in 0..256 {
                        if masks.weight(zone.slot(), x, y) > 0.5 {
                            seen[zone.slot()] += 1;
                        }
                    }
                }
            }
        }
        for zone in Zone::ALL {
            assert!(seen[zone.slot()] > 0, "{zone:?} has no texels");
        }
        // The chin is under the lips, the upper lids above the under-eye zone.
        let (chin, _, _) = centroid(Zone::Chin);
        let (lids, _, _) = centroid(Zone::Eyelids);
        let (under, _, _) = centroid(Zone::UnderEye);
        assert!(chin.y < lips.y && lids.y > under.y);
    }

    #[test]
    fn missing_names_are_reported() {
        let generated = build::template(&build::Options { lods: false, ..build::Options::default() }).unwrap();
        let sources = [
            ZoneSource { regions: &["no-such-region"], ..TEMPLATE_ZONES[0] },
            ZoneSource { joints: &["no.such.joint"], ..TEMPLATE_ZONES[6] },
        ];
        let (weights, report) = vertex_weights(&generated.template, &sources).unwrap();
        assert_eq!(report.missing_regions, ["no-such-region"]);
        assert_eq!(report.missing_joints, ["no.such.joint"]);
        assert!(weights.zone(Zone::Cheeks).iter().all(|&x| x == 0.0));
        let bad = [ZoneSource { feather: 0.0, ..TEMPLATE_ZONES[0] }];
        assert!(vertex_weights(&generated.template, &bad).is_err());
    }

    const BROKEN: [OutlineRun; 1] = [OutlineRun { name: "lip.no-such-line", reverse: false }];

    /// Lip-mask coverage of one baked tile, areas in mm² of surface weighted by the mask.
    #[derive(Debug, Default)]
    struct LipSpill {
        /// All lip mask.
        total: f32,
        /// Mask more than 2 mm (projected) outside the vermilion outline.
        beyond: f32,
        /// Largest mask value more than 2 mm outside the outline.
        beyond_max: f32,
        /// Mask on texels outside the lip regions (philtrum, chin, cheeks, nose, ...).
        off_regions: f32,
        /// Mean mask on the vermilion more than 2 mm inside the outline.
        core_mean: f32,
    }

    fn lip_spill(template: &Template, sources: &[ZoneSource]) -> LipSpill {
        let (weights, report) = vertex_weights(template, sources).unwrap();
        assert!(report.complete(), "{report:?}");
        let mesh = template.mesh();
        let map = template.regions();
        let lip_regions: Vec<u16> =
            ["upper-lip", "lower-lip", "lip-corner-L", "lip-corner-R"].iter().map(|r| map.find(r).unwrap()).collect();
        let in_lips = |v: u32| lip_regions.iter().any(|&r| map.is_within(map.vertex_regions[v as usize], r));
        let mut missing = Vec::new();
        let outline = Outline::new(&outline_points(template, &LIP_OUTLINE, &mut missing).unwrap().unwrap()).unwrap();
        let triangles = mesh.triangulate();
        let tile = (0..triangles.len())
            .find_map(|t| {
                let c = triangles.triangle(t).unwrap();
                in_lips(c[0]).then(|| mesh.face_tiles()[triangles.triangle_faces[t] as usize])
            })
            .unwrap();
        let size = 2048_u32;
        let masks = bake_tile(mesh, &triangles, &weights, tile, size, 0).unwrap();
        let (uvs, positions) = (mesh.uvs(), mesh.positions());
        let scale = size as f32;
        let mut spill = LipSpill::default();
        let (mut core_sum, mut core_n) = (0.0_f32, 0_u32);
        for t in 0..triangles.len() {
            let c = triangles.triangle(t).unwrap();
            if mesh.face_tiles()[triangles.triangle_faces[t] as usize] != tile {
                continue;
            }
            let p = c.map(|v| uvs[v as usize] * scale);
            let q = c.map(|v| positions[v as usize]);
            let area = edge(p[0], p[1], p[2]);
            if area.abs() < 1e-9 {
                continue;
            }
            let texel_mm2 = 0.5 * (q[1] - q[0]).cross(q[2] - q[0]).length() * 1e6 / (0.5 * area.abs());
            let lo = |i: usize| (p.iter().map(|v| v[i]).fold(f32::INFINITY, f32::min) - 0.5).floor().max(0.0) as u32;
            let hi = |i: usize| ((p.iter().map(|v| v[i]).fold(0.0, f32::max) - 0.5).ceil() as u32).min(size - 1);
            for y in lo(1)..=hi(1) {
                for x in lo(0)..=hi(0) {
                    let at = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                    let b = [edge(p[1], p[2], at), edge(p[2], p[0], at), edge(p[0], p[1], at)].map(|e| e / area);
                    if b.iter().any(|&l| l < 0.0) {
                        continue;
                    }
                    let point = q[0] * b[0] + q[1] * b[1] + q[2] * b[2];
                    let dominant = (0..3).max_by(|&i, &j| b[i].total_cmp(&b[j])).unwrap();
                    let w = masks.weight(Zone::Lips.slot(), x, y);
                    let sd = outline.signed_distance(point);
                    spill.total += w * texel_mm2;
                    if sd < -0.002 {
                        spill.beyond += w * texel_mm2;
                        spill.beyond_max = spill.beyond_max.max(w);
                    }
                    if !in_lips(c[dominant]) {
                        spill.off_regions += w * texel_mm2;
                    }
                    if sd > 0.002 && in_lips(c[dominant]) {
                        core_sum += w;
                        core_n += 1;
                    }
                }
            }
        }
        spill.core_mean = core_sum / core_n.max(1) as f32;
        spill
    }

    /// `[SKN-11]` The lip mask stays on the vermilion: on the generated template, nothing of it lies
    /// more than 2 mm outside the vermilion border (the landmark outline), almost nothing on the
    /// philtrum, chin or cheeks, and the vermilion itself is fully covered. The old region-only lip
    /// zone (the whole lip regions, perioral skin included) is measured for comparison.
    #[test]
    fn skn11_lip_mask_stays_on_the_vermilion() {
        let generated = build::template(&build::Options { lods: false, ..build::Options::default() }).unwrap();
        let template = &generated.template;
        let lips = TEMPLATE_ZONES.iter().find(|s| s.zone == Zone::Lips).copied().unwrap();
        let old = ZoneSource {
            regions: &["upper-lip", "lower-lip", "lip-corner-L", "lip-corner-R"],
            outline: &[],
            feather: 0.003,
            ..lips
        };
        let before = lip_spill(template, &[old]);
        let after = lip_spill(template, &[lips]);
        println!("lip spill before (regions only): {before:?}");
        println!("lip spill after (vermilion outline): {after:?}");
        // The vermilion of a 50 mm wide mouth with 9 + 10 mm high lips: a few hundred mm².
        assert!(after.total > 300.0 && after.total < 1500.0, "{after:?}");
        assert!(after.core_mean > 0.99, "{after:?}");
        assert!(after.beyond_max < 1.0 / 255.0, "{after:?}");
        assert!(after.off_regions < 0.05 * after.total, "{after:?}");
        // The old zone spread over more than the vermilion.
        assert!(before.beyond > after.total, "{before:?}");
        // Missing landmarks leave the zone empty and are reported.
        let (weights, report) = vertex_weights(template, &[ZoneSource { outline: &BROKEN, ..lips }]).unwrap();
        assert_eq!(report.missing_landmarks, ["lip.no-such-line"]);
        assert!(weights.zone(Zone::Lips).iter().all(|&x| x == 0.0));
    }

    #[test]
    fn outline_signed_distance_is_positive_inside() {
        let square =
            [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)];
        let o = Outline::new(&square).unwrap();
        assert!((o.signed_distance(Vec3::new(0.5, 0.5, 3.0)) - 0.5).abs() < 1e-6);
        assert!((o.signed_distance(Vec3::new(0.5, 1.25, -1.0)) + 0.25).abs() < 1e-6);
        assert!((o.signed_distance(Vec3::new(2.0, 0.5, 0.0)) + 1.0).abs() < 1e-6);
        assert!(Outline::new(&[Vec3::ZERO; 3]).is_err());
    }

    /// Writes front and back point-splat previews of the baked zones of the generated template to
    /// `target/agents/skin-finish/previews/` (red: cheeks, lips, ears; green: nose, forehead,
    /// temples; blue: under-eye, knuckles, elbows, knees).
    #[test]
    fn zone_mask_preview() {
        let generated = build::template(&build::Options { lods: false, ..build::Options::default() }).unwrap();
        let template = &generated.template;
        let (weights, _) = vertex_weights(template, &TEMPLATE_ZONES).unwrap();
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/skin-finish/previews");
        std::fs::create_dir_all(&dir).unwrap();
        let groups: [&[Zone]; 3] = [
            &[Zone::Cheeks, Zone::Lips, Zone::Ears],
            &[Zone::Nose, Zone::Forehead, Zone::Temples],
            &[Zone::UnderEye, Zone::Knuckles, Zone::Elbows, Zone::Knees],
        ];
        let positions = template.mesh().positions();
        let views = [
            ("zones_front", 1.0_f32, 0.0_f32, 1.0_f32),
            ("zones_back", -1.0, 0.0, 1.0),
            ("zones_face", 1.0, 1.62, 0.16),
        ];
        for (name, facing, centre_y, half) in views {
            let (w, h) = (600_usize, 900_usize);
            let (lo_y, hi_y) = if half < 1.0 { (centre_y - half, centre_y + half) } else { (-0.05, 1.85) };
            let scale = h as f32 / (hi_y - lo_y);
            let mut depth = vec![f32::NEG_INFINITY; w * h];
            let mut rgb = vec![40_u8; w * h * 3];
            for (v, p) in positions.iter().enumerate() {
                let x = (p.x * facing * scale + w as f32 / 2.0) as i64;
                let y = ((hi_y - p.y) * scale) as i64;
                let z = p.z * facing;
                let c: Vec<f32> =
                    groups.iter().map(|g| g.iter().map(|z| weights.zone(*z)[v]).sum::<f32>().min(1.0)).collect();
                let colour = [
                    0.55 + 0.45 * c[0] - 0.3 * (c[1] + c[2]),
                    0.55 + 0.45 * c[1] - 0.3 * (c[0] + c[2]),
                    0.55 + 0.45 * c[2] - 0.3 * (c[0] + c[1]),
                ];
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (px, py) = (x + dx, y + dy);
                        if px < 0 || py < 0 || px >= w as i64 || py >= h as i64 {
                            continue;
                        }
                        let i = py as usize * w + px as usize;
                        if z > depth[i] {
                            depth[i] = z;
                            for k in 0..3 {
                                rgb[i * 3 + k] = (colour[k].clamp(0.0, 1.0) * 255.0) as u8;
                            }
                        }
                    }
                }
            }
            let image = humanart_io::Image::new(
                humanart_io::Format::Png,
                w as u32,
                h as u32,
                3,
                humanart_io::ColorSpace::Srgb,
                humanart_io::Samples::U8(rgb),
            )
            .unwrap();
            std::fs::write(dir.join(format!("{name}.png")), humanart_io::png::encode(&image).unwrap()).unwrap();
        }
    }
}
