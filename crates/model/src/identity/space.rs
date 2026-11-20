//! The surface context the identity recipes are written against (WP-06, `[ID-01]` (c), `[ID-07]`).
//!
//! A [`Space`] wraps the shared [`FaceField`] (welded nodes, normals, regions, geodesic distances,
//! names of landmarks, joints and regions) and adds what whole-body identity deltas need:
//!
//! - **part kinds** ([`PartKind`]): the body surface, *helper* parts that sit on it and follow it
//!   (tear lines, eye occlusion, the lip-line helper: each helper node copies the displacement of the
//!   nearest body node, so they never detach), and *rigid* parts (eyeballs, teeth, gums, tongue)
//!   that only a recipe meant for them moves;
//! - the **mirror map** of nodes across the sagittal plane `x = 0`, so symmetric components are
//!   made exactly mirror-symmetric by copying the left half (+X is the character's left) to the right;
//! - **region confinement**: a component names the regions it may move; nodes outside get zero and
//!   nodes inside ramp up smoothly (C1) over a geodesic margin from the border, so a component never
//!   moves another region and never leaves a crease at a region border `[ID-02]`;
//! - **smoothing** of per-node amounts (Jacobi diffusion over the node graph).
//!
//! [`Space::finish`] turns per-node displacements into an [`IdentityComponent`]: helper copy,
//! confinement, mirror, vertex expansion (every vertex of a welded node gets the node's
//! displacement, so UV seams never split) and the bounding region.

use std::collections::HashMap;

use humanart_math::Vec3;

use super::error::{IdResult, IdentityError};
use crate::delta::DeltaKind;
use crate::expression::field::{FaceField, falloff};
use crate::template::Template;

/// Displacements shorter than this (metres) are not stored.
pub const EPSILON: f32 = 1.0e-6;
/// Two nodes are mirror images when they lie within this distance (metres) of each other's mirror.
pub const MIRROR_TOLERANCE: f32 = 5.0e-4;
/// The looser mirror tolerance of separate parts (metres), see [`Space::mirror_map`].
pub const PART_MIRROR_TOLERANCE: f32 = 0.003;
/// The farthest a helper node may be from the body node it follows (metres).
pub const ATTACH_RADIUS: f32 = 0.08;

/// How a mesh part takes part in identity deltas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PartKind {
    /// The skin surface (`body`).
    Body,
    /// A part that sits on the skin and follows it (`tear-line-*`, `eye-occlusion-*`,
    /// `mouth-helper`, and any part attached to the skin that is not listed as rigid, such as the
    /// attached anatomical parts of ADR-039).
    Helper,
    /// A separate rigid part moved only by recipes meant for it (eyeballs, teeth, gums, tongue).
    Rigid,
}

impl PartKind {
    /// The kind of the mesh part called `name`.
    #[must_use]
    pub fn of(name: &str) -> Self {
        if name == "body" {
            Self::Body
        } else if ["eyeball", "teeth", "gums", "tongue"].iter().any(|r| name.starts_with(r)) {
            Self::Rigid
        } else {
            Self::Helper
        }
    }
}

/// Whether a component is mirror-symmetric.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Symmetry {
    /// The right half is the mirror image of the left half; joints likewise.
    Mirror,
    /// No symmetry is imposed (asymmetry components `[ID-05]`).
    Free,
}

/// One authored identity component: its delta, its joint-regressor row and what it may move.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentityComponent {
    /// Stable component name, the registry's `component` target (`global.height`).
    pub name: String,
    /// Its delta kind (always [`DeltaKind::Semantic`] for authored components, see the module
    /// documentation of [`super`]).
    pub kind: DeltaKind,
    /// The bounding region: the lowest common ancestor of the regions of the moved vertices.
    pub region: u16,
    /// `(vertex, displacement)` per unit weight, metres, vertices strictly increasing.
    pub entries: Vec<(u32, Vec3)>,
    /// `(joint, offset)` per unit weight, metres, joints strictly increasing: the regressor row.
    pub joints: Vec<(u16, Vec3)>,
    /// Whether it is mirror-symmetric.
    pub symmetry: Symmetry,
    /// The regions it may move (every region index of the allowed subtrees).
    pub allowed: Vec<u16>,
}

/// What a recipe may move: region subtrees, minus some subtrees, with a confinement margin.
#[derive(Clone, Copy, Debug)]
pub struct Reach<'a> {
    /// Region names whose subtrees may move.
    pub regions: &'a [&'a str],
    /// Region names whose subtrees may not move even inside `regions`.
    pub except: &'a [&'a str],
    /// The geodesic ramp from the border of the allowed area, metres (0: no ramp).
    pub margin: f32,
}

/// Limb joints (without the `L.` / `R.` prefix) and the skin landmarks around them whose mean
/// displacement the joint follows in every component ([`Space::component`], `[MOD-07]`).
pub const SKIN_RINGS: &[(&str, &[&str])] = &[
    ("shin", &["knee.center", "knee.lateral", "knee.medial", "knee.back"]),
    ("foot", &["ankle.lateral", "ankle.medial"]),
    ("forearm", &["elbow.lateral", "elbow.medial"]),
    ("hand", &["wrist.radial", "wrist.ulnar"]),
];

/// Small detail regions that every component except their own carries rigidly
/// ([`Space::carry_details`]).
pub const DETAILS: &[&str] = &["nipple-L", "nipple-R", "navel"];

/// The whole template may move (skeletal components such as height).
pub const EVERYWHERE: Reach<'static> = Reach { regions: &["body"], except: &[], margin: 0.0 };

/// One node as a recipe sees it.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    /// The node.
    pub node: usize,
    /// Its rest position.
    pub p: Vec3,
    /// Its unit rest normal (zero for degenerate nodes).
    pub n: Vec3,
    /// Its mesh part.
    pub part: u16,
    /// How its part takes part.
    pub kind: PartKind,
    /// Its region.
    pub region: u16,
}

impl Point {
    /// A free point (a joint position): no node, no normal.
    #[must_use]
    pub const fn at(p: Vec3) -> Self {
        Self { node: usize::MAX, p, n: Vec3::ZERO, part: u16::MAX, kind: PartKind::Body, region: 0 }
    }
}

/// The identity surface context of one template (see the module documentation).
#[derive(Clone, Debug)]
pub struct Space<'t> {
    field: FaceField<'t>,
    kinds: Vec<PartKind>,
    mirror: Vec<u32>,
    attach: Vec<u32>,
}

/// A uniform grid of node indices for nearest-node queries.
struct Grid {
    cell: f32,
    cells: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl Grid {
    fn new(cell: f32, nodes: impl Iterator<Item = (u32, Vec3)>) -> Self {
        let mut cells: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (n, p) in nodes {
            cells.entry(Self::key(cell, p)).or_default().push(n);
        }
        Self { cell, cells }
    }

    fn key(cell: f32, p: Vec3) -> (i32, i32, i32) {
        ((p.x / cell).floor() as i32, (p.y / cell).floor() as i32, (p.z / cell).floor() as i32)
    }

    /// The node nearest to `p` within `rings` cells around it.
    fn nearest(&self, p: Vec3, positions: &[Vec3], rings: i32) -> Option<(u32, f32)> {
        let (cx, cy, cz) = Self::key(self.cell, p);
        let mut best: Option<(u32, f32)> = None;
        for dx in -rings..=rings {
            for dy in -rings..=rings {
                for dz in -rings..=rings {
                    let Some(list) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) else { continue };
                    for &n in list {
                        let Some(q) = positions.get(n as usize) else { continue };
                        let d = q.distance(p);
                        if best.is_none_or(|(_, bd)| d < bd) {
                            best = Some((n, d));
                        }
                    }
                }
            }
        }
        best
    }
}

/// The mirror image of a vector or point across `x = 0`.
#[must_use]
pub fn mirror(v: Vec3) -> Vec3 {
    Vec3::new(-v.x, v.y, v.z)
}

/// A C1 ramp from 0 at `e0` to 1 at `e1` (either order; constant beyond).
#[must_use]
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    if e0 == e1 {
        return if x < e0 { 0.0 } else { 1.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A compact C1 bump: `(1 - (d / r)^2)^2` for `d < r`, 0 beyond; 1 at `d = 0`.
#[must_use]
pub fn bump(d: f32, r: f32) -> f32 {
    if r <= 0.0 || d.is_nan() || d >= r {
        return 0.0;
    }
    let t = 1.0 - (d / r) * (d / r);
    t * t
}

/// An ellipsoidal compact bump around `c`: radius `along` in direction `axis` (unit) and `across`
/// perpendicular to it.
#[must_use]
pub fn bump_along(p: Vec3, c: Vec3, axis: Vec3, along: f32, across: f32) -> f32 {
    let d = p - c;
    let a = d.dot(axis);
    let r = d - axis * a;
    if along <= 0.0 || across <= 0.0 {
        return 0.0;
    }
    let s = (a / along) * (a / along) + r.length_squared() / (across * across);
    if s >= 1.0 {
        return 0.0;
    }
    (1.0 - s) * (1.0 - s)
}

/// Solves the 3 x 3 system `m x = rhs` (vector right-hand sides) by Cramer's rule; `None` when
/// `m` is singular (relative determinant below 1e-9).
#[must_use]
pub fn solve3(m: [[f32; 3]; 3], rhs: [Vec3; 3]) -> Option<[Vec3; 3]> {
    let det3 = |a: [[f32; 3]; 3]| {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let det = det3(m);
    let scale = m.iter().flatten().fold(0.0_f32, |a, &v| a.max(v.abs()));
    if !det.is_finite() || det.abs() <= 1.0e-9 * scale * scale * scale {
        return None;
    }
    let mut out = [Vec3::ZERO; 3];
    for (col, slot) in out.iter_mut().enumerate() {
        let mut comp = [0.0_f32; 3];
        for (axis, value) in comp.iter_mut().enumerate() {
            let mut a = m;
            for (row, line) in a.iter_mut().enumerate() {
                let r = rhs[row];
                line[col] = [r.x, r.y, r.z][axis];
            }
            *value = det3(a) / det;
        }
        *slot = Vec3::new(comp[0], comp[1], comp[2]);
    }
    Some(out)
}

/// The distance from point `p` to the segment `a`-`b`.
#[must_use]
pub fn segment_distance(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    let t = if len2 > 0.0 { ((p - a).dot(ab) / len2).clamp(0.0, 1.0) } else { 0.0 };
    p.distance(a + ab * t)
}

impl<'t> Space<'t> {
    /// Builds the context: the field, part kinds, the mirror map and the helper attachments.
    pub fn new(template: &'t Template) -> IdResult<Self> {
        let field = FaceField::new(template)?;
        let kinds: Vec<PartKind> = template.mesh().parts().iter().map(|p| PartKind::of(&p.name)).collect();
        let body = template
            .mesh()
            .parts()
            .iter()
            .position(|p| p.name == "body")
            .ok_or_else(|| IdentityError::missing("mesh part", "body"))? as u16;
        let pos = field.positions();
        let count = pos.len();
        // Mirror map: per part, the node within the tolerance of the mirrored position.
        let grid = Grid::new(0.002, (0..count as u32).filter_map(|n| pos.get(n as usize).map(|&p| (n, p))));
        // The mirror part of each part: the part whose name swaps a `-L` / `-R` suffix, else itself.
        let parts = template.mesh().parts();
        let mirror_part: Vec<u16> = parts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let swapped = if let Some(b) = p.name.strip_suffix("-L") {
                    format!("{b}-R")
                } else if let Some(b) = p.name.strip_suffix("-R") {
                    format!("{b}-L")
                } else {
                    p.name.clone()
                };
                parts.iter().position(|q| q.name == swapped).unwrap_or(i) as u16
            })
            .collect();
        let mut mirror_map = vec![u32::MAX; count];
        for (n, slot) in mirror_map.iter_mut().enumerate() {
            let Some(&p) = pos.get(n) else { continue };
            let part = mirror_part.get(field.part_of(n) as usize).copied().unwrap_or(u16::MAX);
            let target = mirror(p);
            // Search the candidates around the target, same part only. Separate parts may be
            // tessellated with a phase shift between the sides (eyeballs, tear lines): they take
            // the nearest node within the looser tolerance.
            let body_part = kinds.get(part as usize) == Some(&PartKind::Body);
            let (tolerance, rings) = if body_part { (MIRROR_TOLERANCE, 1) } else { (PART_MIRROR_TOLERANCE, 2) };
            let (cx, cy, cz) = Grid::key(grid.cell, target);
            let mut best: Option<(u32, f32)> = None;
            for dx in -rings..=rings {
                for dy in -rings..=rings {
                    for dz in -rings..=rings {
                        let Some(list) = grid.cells.get(&(cx + dx, cy + dy, cz + dz)) else { continue };
                        for &m in list {
                            if field.part_of(m as usize) != part {
                                continue;
                            }
                            let Some(q) = pos.get(m as usize) else { continue };
                            let d = q.distance(target);
                            if d <= tolerance && best.is_none_or(|(_, bd)| d < bd) {
                                best = Some((m, d));
                            }
                        }
                    }
                }
            }
            if let Some((m, _)) = best {
                *slot = m;
            }
        }
        // Helper attachments: the nearest body node.
        let body_grid = Grid::new(
            0.005,
            (0..count as u32)
                .filter(|&n| field.part_of(n as usize) == body)
                .filter_map(|n| pos.get(n as usize).map(|&p| (n, p))),
        );
        let max_rings = (ATTACH_RADIUS / 0.005).ceil() as i32;
        let mut attach = vec![u32::MAX; count];
        for (n, slot) in attach.iter_mut().enumerate() {
            let kind = kinds.get(field.part_of(n) as usize).copied().unwrap_or(PartKind::Rigid);
            if kind != PartKind::Helper {
                continue;
            }
            let Some(&p) = pos.get(n) else { continue };
            // Grow the search until a body node is found (one ring beyond the first hit is enough
            // for the nearest within the grid metric).
            let mut rings = 1;
            while rings <= max_rings {
                if let Some(hit) = body_grid.nearest(p, pos, rings) {
                    let better = body_grid.nearest(p, pos, rings + 1).filter(|h| h.1 < hit.1);
                    *slot = better.unwrap_or(hit).0;
                    break;
                }
                rings += 1;
            }
        }
        Ok(Self { field, kinds, mirror: mirror_map, attach })
    }

    /// The shared surface context.
    #[must_use]
    pub const fn field(&self) -> &FaceField<'t> {
        &self.field
    }

    /// The template.
    #[must_use]
    pub const fn template(&self) -> &'t Template {
        self.field.template()
    }

    /// How many welded nodes there are.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.field.node_count()
    }

    /// The mirror node of each node (`u32::MAX` where there is none): within [`MIRROR_TOLERANCE`] on
    /// the body, the nearest within [`PART_MIRROR_TOLERANCE`] on separate parts.
    #[must_use]
    pub fn mirror_map(&self) -> &[u32] {
        &self.mirror
    }

    /// The part kind of node `n`.
    #[must_use]
    pub fn kind_of(&self, n: usize) -> PartKind {
        self.kinds.get(self.field.part_of(n) as usize).copied().unwrap_or(PartKind::Rigid)
    }

    /// Node `n` as a recipe sees it (`None` when out of range).
    #[must_use]
    pub fn point(&self, n: usize) -> Option<Point> {
        let p = *self.field.positions().get(n)?;
        let normal = self.field.normals().get(n).copied().unwrap_or(Vec3::ZERO);
        Some(Point {
            node: n,
            p,
            n: normal,
            part: self.field.part_of(n),
            kind: self.kind_of(n),
            region: self.field.region_of(n),
        })
    }

    /// Evaluates `f` at every node.
    pub fn eval(&self, f: impl Fn(&Point) -> Vec3) -> Vec<Vec3> {
        (0..self.node_count()).map(|n| self.point(n).map_or(Vec3::ZERO, |pt| f(&pt))).collect()
    }

    /// `f` at `pt` plus the mirror image of `f` at the mirrored point: a field written for the left
    /// side (+X) made symmetric, so it stays continuous across the midline.
    pub fn sym(&self, pt: &Point, f: &impl Fn(&Point) -> Vec3) -> Vec3 {
        let q = self
            .mirror
            .get(pt.node)
            .copied()
            .filter(|&m| m != u32::MAX)
            .and_then(|m| self.point(m as usize))
            .unwrap_or(Point { p: mirror(pt.p), n: mirror(pt.n), ..*pt });
        f(pt) + mirror(f(&q))
    }

    /// The rest position of the landmark called `name`.
    pub fn landmark(&self, name: &str) -> IdResult<Vec3> {
        Ok(self.field.landmark(name)?)
    }

    /// The nodes carrying the landmarks called `names`.
    pub fn landmark_nodes(&self, names: &[&str]) -> IdResult<Vec<u32>> {
        names.iter().map(|n| Ok(self.field.landmark_node(n)?)).collect()
    }

    /// The rest positions of the landmarks `prefix.01`, `prefix.02`, ... (at least one must exist).
    pub fn landmark_series(&self, prefix: &str) -> IdResult<Vec<Vec3>> {
        let s = self.field.landmark_series(prefix);
        if s.is_empty() {
            return Err(IdentityError::missing("landmark series", prefix));
        }
        Ok(s)
    }

    /// The nodes of the landmarks `prefix.01`, `prefix.02`, ... (at least one must exist).
    pub fn series_nodes(&self, prefix: &str) -> IdResult<Vec<u32>> {
        let mut out = Vec::new();
        for k in 1..100 {
            match self.field.landmark_node(&format!("{prefix}.{k:02}")) {
                Ok(n) => out.push(n),
                Err(_) => break,
            }
        }
        if out.is_empty() {
            return Err(IdentityError::missing("landmark series", prefix));
        }
        Ok(out)
    }

    /// The index of the joint called `name`.
    pub fn joint_index(&self, name: &str) -> IdResult<u16> {
        Ok(self.field.joint_index(name)?)
    }

    /// The rest world position of the joint called `name`.
    pub fn joint(&self, name: &str) -> IdResult<Vec3> {
        Ok(self.field.joint(name)?)
    }

    /// The index of the region called `name`.
    pub fn region(&self, name: &str) -> IdResult<u16> {
        Ok(self.field.region(name)?)
    }

    /// The index of the mesh part called `name`.
    pub fn part(&self, name: &str) -> IdResult<u16> {
        Ok(self.field.part(name)?)
    }

    /// The nodes of the mesh part called `name`.
    pub fn part_nodes(&self, name: &str) -> IdResult<Vec<u32>> {
        let part = self.part(name)?;
        Ok((0..self.node_count() as u32).filter(|&n| self.field.part_of(n as usize) == part).collect())
    }

    /// The region indices of `reach` (allowed subtrees minus excepted subtrees), as a flag per region.
    pub fn reach_flags(&self, reach: &Reach<'_>) -> IdResult<Vec<bool>> {
        let regions = self.template().regions();
        let count = regions.regions.len();
        let mut flags = vec![false; count];
        let mut roots = Vec::with_capacity(reach.regions.len());
        for name in reach.regions {
            roots.push(self.region(name)?);
        }
        let mut excepted = Vec::with_capacity(reach.except.len());
        for name in reach.except {
            excepted.push(self.region(name)?);
        }
        for (r, flag) in flags.iter_mut().enumerate() {
            let r = r as u16;
            *flag =
                roots.iter().any(|&a| regions.is_within(r, a)) && !excepted.iter().any(|&a| regions.is_within(r, a));
        }
        Ok(flags)
    }

    /// Per node, a weight for each node from a table of `(region name, amount)`: a node takes the
    /// amount of its own region or of its nearest listed ancestor, 0 when none is listed.
    pub fn region_amounts(&self, table: &[(&str, f32)]) -> IdResult<Vec<f32>> {
        let regions = self.template().regions();
        let mut per_region = vec![None; regions.regions.len()];
        for &(name, amount) in table {
            let r = self.region(name)?;
            if let Some(slot) = per_region.get_mut(r as usize) {
                *slot = Some(amount);
            }
        }
        // Resolve every region to its nearest listed ancestor (depth-limited walk).
        let mut resolved = vec![0.0_f32; regions.regions.len()];
        for (r, out) in resolved.iter_mut().enumerate() {
            let mut cur = Some(r as u16);
            let mut steps = 0;
            while let Some(c) = cur {
                if let Some(Some(a)) = per_region.get(c as usize) {
                    *out = *a;
                    break;
                }
                steps += 1;
                if steps > regions.regions.len() {
                    break;
                }
                cur = regions.regions.get(c as usize).and_then(|x| x.parent).filter(|&p| p != c);
            }
        }
        Ok((0..self.node_count())
            .map(|n| resolved.get(self.field.region_of(n) as usize).copied().unwrap_or(0.0))
            .collect())
    }

    /// Smooths per-node values in place by `iterations` Jacobi steps (each node moves halfway to its
    /// neighbours' mean). Only nodes of part kind `Body` change.
    pub fn smooth(&self, values: &mut [f32], iterations: usize) {
        let mut next = values.to_vec();
        for _ in 0..iterations {
            for (n, out) in next.iter_mut().enumerate() {
                if self.kind_of(n) != PartKind::Body {
                    continue;
                }
                let nb = self.field.neighbours(n);
                if nb.is_empty() {
                    continue;
                }
                let sum: f32 = nb.iter().filter_map(|&m| values.get(m as usize)).sum();
                let own = values.get(n).copied().unwrap_or(0.0);
                *out = 0.5 * own + 0.5 * sum / nb.len() as f32;
            }
            values.copy_from_slice(&next);
        }
    }

    /// Smooths per-node displacements in place by `iterations` Jacobi steps (each node moves
    /// halfway to its neighbours' inverse-distance weighted mean), changing only skin nodes of the regions of `reach` (its
    /// margin is ignored): spreads a shape's steepest transition over a few rings of the actual
    /// mesh, whatever its resolution, without blurring fine detail elsewhere (the face).
    pub fn smooth_field(&self, disp: &mut [Vec3], iterations: usize, reach: &Reach<'_>) -> IdResult<()> {
        let flags = self.reach_flags(reach)?;
        let active: Vec<usize> = (0..disp.len().min(self.node_count()))
            .filter(|&n| {
                self.kind_of(n) == PartKind::Body
                    && flags.get(self.field.region_of(n) as usize).copied().unwrap_or(false)
                    && !self.field.neighbours(n).is_empty()
            })
            .collect();
        let pos = self.field.positions();
        let mut next = disp.to_vec();
        for _ in 0..iterations {
            for &n in &active {
                // Inverse-distance weights: where the mesh is refined (around the nipples and the
                // navel) the near neighbours count more, so the result does not depend on the
                // local spacing.
                let p = pos.get(n).copied().unwrap_or(Vec3::ZERO);
                let (mut sum, mut weight) = (Vec3::ZERO, 0.0_f32);
                for &m in self.field.neighbours(n) {
                    let (Some(&q), Some(&d)) = (pos.get(m as usize), disp.get(m as usize)) else { continue };
                    let w = 1.0 / q.distance(p).max(1.0e-4);
                    sum += d * w;
                    weight += w;
                }
                let own = disp.get(n).copied().unwrap_or(Vec3::ZERO);
                if let Some(out) = next.get_mut(n)
                    && weight > 0.0
                {
                    *out = own * 0.5 + sum * (0.5 / weight);
                }
            }
            disp.copy_from_slice(&next);
        }
        Ok(())
    }

    /// Carries each small detail of [`DETAILS`] (the nipples, the navel) with the skin around it:
    /// the displacement of the ring of nodes just outside the region is fitted (least squares) by
    /// an affine map over the ring's tangent plane, `d0 + gu u + gv v`, and every node of the region
    /// takes that map. A field along the normal then neither inflates a nipple nor fills the navel,
    /// and a stretch of the skin around it (waist width, areola size) stretches it alike. Regions
    /// the template lacks, or with fewer than 4 ring nodes, are skipped.
    pub fn carry_details(&self, disp: &mut [Vec3]) {
        let pos = self.field.positions();
        let normals = self.field.normals();
        for name in DETAILS {
            let Ok(r) = self.region(name) else { continue };
            let inside = |n: usize| self.kind_of(n) == PartKind::Body && self.field.in_region(n, r);
            let nodes: Vec<usize> = (0..disp.len().min(self.node_count())).filter(|&n| inside(n)).collect();
            let mut ring: Vec<usize> = nodes
                .iter()
                .flat_map(|&n| self.field.neighbours(n).iter().map(|&m| m as usize))
                .filter(|&m| !inside(m) && m < disp.len())
                .collect();
            ring.sort_unstable();
            ring.dedup();
            if ring.len() < 4 {
                continue;
            }
            let count = ring.len() as f32;
            let c = ring.iter().filter_map(|&m| pos.get(m)).fold(Vec3::ZERO, |a, &b| a + b) / count;
            let axis = ring.iter().filter_map(|&m| normals.get(m)).fold(Vec3::ZERO, |a, &b| a + b).normalize_or_zero();
            let t1 = axis.any_orthonormal_vector();
            let t2 = axis.cross(t1);
            let uv = |p: Vec3| [1.0, (p - c).dot(t1), (p - c).dot(t2)];
            // Normal equations of the fit: m (3 x 3, symmetric) times the coefficients = rhs.
            let mut m = [[0.0_f32; 3]; 3];
            let mut rhs = [Vec3::ZERO; 3];
            for &k in &ring {
                let (Some(&p), Some(&d)) = (pos.get(k), disp.get(k)) else { continue };
                let b = uv(p);
                for (row, &bi) in m.iter_mut().zip(&b) {
                    for (cell, &bj) in row.iter_mut().zip(&b) {
                        *cell += bi * bj;
                    }
                }
                for (slot, &bi) in rhs.iter_mut().zip(&b) {
                    *slot += d * bi;
                }
            }
            let Some(coef) = solve3(m, rhs) else { continue };
            for n in nodes {
                let Some(&p) = pos.get(n) else { continue };
                let b = uv(p);
                if let Some(d) = disp.get_mut(n) {
                    *d = coef[0] * b[0] + coef[1] * b[1] + coef[2] * b[2];
                }
            }
        }
    }

    /// Geodesic distances from `seeds` (metres), infinite beyond `limit`.
    #[must_use]
    pub fn geodesic(&self, seeds: &[u32], limit: f32) -> Vec<f32> {
        self.field.distances(seeds, limit)
    }

    /// A geodesic falloff from the landmarks `names`: 1 up to `inner`, 0 from `outer` on.
    pub fn around(&self, names: &[&str], inner: f32, outer: f32) -> IdResult<Vec<f32>> {
        let seeds = self.landmark_nodes(names)?;
        Ok(self.geodesic(&seeds, outer).iter().map(|&d| falloff(d, inner, outer)).collect())
    }

    /// Like [`around`](Self::around), but inside the region subtree `solid` the weight is the larger
    /// of the geodesic and a Euclidean falloff, so folded surfaces (the inside of the nostrils) move
    /// with the outside they are close to.
    pub fn around_solid(&self, names: &[&str], inner: f32, outer: f32, solid: &str) -> IdResult<Vec<f32>> {
        let mut w = self.around(names, inner, outer)?;
        let root = self.region(solid)?;
        let mut points = Vec::with_capacity(names.len());
        for n in names {
            points.push(self.landmark(n)?);
        }
        let regions = self.template().regions();
        for (n, slot) in w.iter_mut().enumerate() {
            if !regions.is_within(self.field.region_of(n), root) {
                continue;
            }
            let Some(&p) = self.field.positions().get(n) else { continue };
            let d = points.iter().map(|q| q.distance(p)).fold(f32::INFINITY, f32::min);
            *slot = slot.max(falloff(d, inner, outer));
        }
        Ok(w)
    }

    /// Unit normals smoothed by `iterations` Jacobi steps over the body nodes: offsets along them
    /// stay nearly parallel across folds (armpits, finger webs) instead of colliding.
    #[must_use]
    pub fn smooth_normals(&self, iterations: usize) -> Vec<Vec3> {
        let mut cur = self.field.normals().to_vec();
        let mut next = cur.clone();
        for _ in 0..iterations {
            for (n, out) in next.iter_mut().enumerate() {
                if self.kind_of(n) != PartKind::Body {
                    continue;
                }
                let sum = self
                    .field
                    .neighbours(n)
                    .iter()
                    .filter_map(|&m| cur.get(m as usize))
                    .fold(Vec3::ZERO, |a, &b| a + b);
                let own = cur.get(n).copied().unwrap_or(Vec3::ZERO);
                *out = (own + sum.normalize_or_zero()).normalize_or_zero();
            }
            cur.copy_from_slice(&next);
        }
        cur
    }

    /// Per node, the confinement weight of `reach`: 0 outside, ramping to 1 over `margin` metres of
    /// geodesic distance inside; non-body nodes get 1 inside and 0 outside.
    pub fn confinement(&self, reach: &Reach<'_>) -> IdResult<Vec<f32>> {
        let flags = self.reach_flags(reach)?;
        let inside = |n: usize| flags.get(self.field.region_of(n) as usize).copied().unwrap_or(false);
        let count = self.node_count();
        if reach.margin <= 0.0 {
            return Ok((0..count).map(|n| if inside(n) { 1.0 } else { 0.0 }).collect());
        }
        let seeds: Vec<u32> =
            (0..count).filter(|&n| !inside(n) && self.kind_of(n) == PartKind::Body).map(|n| n as u32).collect();
        let dist = self.geodesic(&seeds, reach.margin);
        Ok((0..count)
            .map(|n| {
                if !inside(n) {
                    0.0
                } else if self.kind_of(n) == PartKind::Body {
                    1.0 - falloff(dist.get(n).copied().unwrap_or(f32::INFINITY), 0.0, reach.margin)
                } else {
                    1.0
                }
            })
            .collect())
    }

    /// Confines per-node displacements to `reach` in place: body and rigid nodes are multiplied by
    /// the confinement weight (for mirror components the smaller of a node's and its mirror's), helper
    /// nodes copy their body node and are multiplied likewise.
    pub fn confine(&self, reach: &Reach<'_>, symmetry: Symmetry, disp: &mut [Vec3]) -> IdResult<()> {
        let count = self.node_count();
        if disp.len() != count {
            return Err(IdentityError::invalid(
                "identity component",
                0,
                "displacement count differs from the node count",
            ));
        }
        let mut weight = self.confinement(reach)?;
        if symmetry == Symmetry::Mirror {
            let w0 = weight.clone();
            for (n, w) in weight.iter_mut().enumerate() {
                if let Some(&m) = self.mirror.get(n)
                    && let Some(&wm) = w0.get(m as usize)
                {
                    *w = w.min(wm);
                }
            }
        }
        for (n, d) in disp.iter_mut().enumerate() {
            if self.kind_of(n) != PartKind::Helper {
                *d *= weight.get(n).copied().unwrap_or(0.0);
            }
        }
        for n in 0..count {
            if self.kind_of(n) != PartKind::Helper {
                continue;
            }
            let follow = self.attach.get(n).and_then(|&m| disp.get(m as usize)).copied().unwrap_or(Vec3::ZERO);
            if let Some(d) = disp.get_mut(n) {
                *d = follow * weight.get(n).copied().unwrap_or(0.0);
            }
        }
        Ok(())
    }

    /// [`confine`](Self::confine) to `reach`, then [`component`](Self::component).
    pub fn finish(
        &self,
        name: &str,
        symmetry: Symmetry,
        reach: &Reach<'_>,
        mut disp: Vec<Vec3>,
        joints: Vec<(u16, Vec3)>,
    ) -> IdResult<IdentityComponent> {
        self.confine(reach, symmetry, &mut disp)?;
        self.component(name, symmetry, &[*reach], disp, joints)
    }

    /// Turns confined per-node displacements and joint offsets into a component: symmetry (the
    /// right half copies the mirrored left half), vertex expansion, regressor row and bounding
    /// region; `reaches` are recorded as what the component may move.
    pub fn component(
        &self,
        name: &str,
        symmetry: Symmetry,
        reaches: &[Reach<'_>],
        mut disp: Vec<Vec3>,
        joints: Vec<(u16, Vec3)>,
    ) -> IdResult<IdentityComponent> {
        let count = self.node_count();
        if disp.len() != count {
            return Err(IdentityError::invalid(
                "identity component",
                0,
                "displacement count differs from the node count",
            ));
        }
        if let Some(i) = disp.iter().position(|d| !d.is_finite()) {
            return Err(IdentityError::invalid("identity displacement", i, "not finite"));
        }
        if symmetry == Symmetry::Mirror {
            let positions = self.field.positions();
            for n in 0..count {
                let x = positions.get(n).map_or(0.0, |p| p.x);
                let m = self.mirror.get(n).copied().unwrap_or(u32::MAX);
                if x < 0.0 && m != u32::MAX && m as usize != n {
                    let src = disp.get(m as usize).copied().unwrap_or(Vec3::ZERO);
                    if let Some(d) = disp.get_mut(n) {
                        *d = mirror(src);
                    }
                } else if (x == 0.0 || m as usize == n)
                    && let Some(d) = disp.get_mut(n)
                {
                    d.x = 0.0;
                }
            }
        }
        let entries = self.field.vertex_entries(&disp, EPSILON);
        let joints = self.follow_skin(&disp, joints);
        let joints = self.joint_row(symmetry, joints)?;
        let region = self.bounding_region(entries.iter().map(|e| e.0));
        let mut flags = vec![false; self.template().regions().regions.len()];
        for reach in reaches {
            for (f, g) in flags.iter_mut().zip(self.reach_flags(reach)?) {
                *f |= g;
            }
        }
        let allowed = flags.iter().enumerate().filter(|(_, f)| **f).map(|(r, _)| r as u16).collect();
        Ok(IdentityComponent {
            name: name.to_owned(),
            kind: DeltaKind::Semantic,
            region,
            entries,
            joints,
            symmetry,
            allowed,
        })
    }

    /// `joints` with the offsets of the limb joints of [`SKIN_RINGS`] replaced by the mean
    /// displacement of the skin landmarks around them (`[MOD-07]`): the knee, ankle, elbow and
    /// wrist joints stay inside their skin whatever the recipe and whatever the template's current
    /// geometry. Rings whose joint or landmarks the template lacks are left alone.
    fn follow_skin(&self, disp: &[Vec3], mut joints: Vec<(u16, Vec3)>) -> Vec<(u16, Vec3)> {
        let landmarks = self.template().landmarks();
        let node_disp = |v: u32| disp.get(self.field.node_of(v as usize) as usize).copied().unwrap_or(Vec3::ZERO);
        for side in ["L.", "R."] {
            for &(joint, marks) in SKIN_RINGS {
                let Ok(j) = self.joint_index(&format!("{side}{joint}")) else { continue };
                let mut sum = Vec3::ZERO;
                let mut found = 0_u32;
                for mark in marks {
                    let name = format!("{side}{mark}");
                    let Some(l) = landmarks.iter().find(|l| l.name == name) else { continue };
                    sum += l.vertices.iter().zip(l.weights).fold(Vec3::ZERO, |a, (&v, w)| a + node_disp(v) * w);
                    found += 1;
                }
                if found as usize != marks.len() {
                    continue;
                }
                joints.retain(|e| e.0 != j);
                joints.push((j, sum / found as f32));
            }
        }
        joints
    }

    /// Sorts, merges and (for mirror components) symmetrises a regressor row: the offset of an
    /// `R.` joint is the mirror image of its `L.` partner's, centre joints lose their x offset.
    fn joint_row(&self, symmetry: Symmetry, joints: Vec<(u16, Vec3)>) -> IdResult<Vec<(u16, Vec3)>> {
        let skeleton = &self.template().skeleton().joints;
        let mut merged: Vec<(u16, Vec3)> = Vec::with_capacity(joints.len() * 2);
        for (j, d) in joints {
            if j as usize >= skeleton.len() {
                return Err(IdentityError::invalid("regressor joint", j as usize, "joint index out of range"));
            }
            if !d.is_finite() {
                return Err(IdentityError::invalid("regressor joint", j as usize, "offset is not finite"));
            }
            match merged.iter_mut().find(|e| e.0 == j) {
                Some(e) => e.1 += d,
                None => merged.push((j, d)),
            }
        }
        if symmetry == Symmetry::Mirror {
            let mut out: Vec<(u16, Vec3)> = Vec::with_capacity(merged.len() * 2);
            for &(j, d) in &merged {
                let name = skeleton.get(j as usize).map_or("", |x| x.name.as_str());
                if let Some(rest) = name.strip_prefix("L.") {
                    out.push((j, d));
                    if let Some(r) = self.template().skeleton().find(&format!("R.{rest}")) {
                        out.push((r, mirror(d)));
                    }
                } else if !name.starts_with("R.") {
                    out.push((j, Vec3::new(0.0, d.y, d.z)));
                }
            }
            merged = out;
        }
        merged.retain(|e| e.1.length() > EPSILON);
        merged.sort_by_key(|e| e.0);
        merged.dedup_by_key(|e| e.0);
        Ok(merged)
    }

    /// The lowest common ancestor of the regions of `vertices` (the root for none).
    pub fn bounding_region(&self, vertices: impl Iterator<Item = u32>) -> u16 {
        let regions = self.template().regions();
        let chain = |r: u16| {
            let mut out = vec![r];
            let mut cur = r;
            while let Some(p) = regions.regions.get(cur as usize).and_then(|x| x.parent) {
                if out.len() > regions.regions.len() || p == cur {
                    break;
                }
                out.push(p);
                cur = p;
            }
            out
        };
        let mut lca: Option<Vec<u16>> = None;
        for v in vertices {
            let r = regions.vertex_regions.get(v as usize).copied().unwrap_or(0);
            lca = Some(match lca {
                None => chain(r),
                Some(cur) => {
                    if cur.first().is_some_and(|&c| regions.is_within(r, c)) {
                        cur
                    } else {
                        let mine = chain(r);
                        cur.into_iter().skip_while(|a| !mine.contains(a)).collect()
                    }
                }
            });
        }
        lca.and_then(|c| c.first().copied()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve3_solves_and_refuses_singular() {
        let m = [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]];
        let x = [Vec3::new(1.0, 0.0, -1.0), Vec3::new(0.5, 2.0, 0.0), Vec3::new(-1.0, 1.0, 3.0)];
        let rhs: Vec<Vec3> = m.iter().map(|row| x[0] * row[0] + x[1] * row[1] + x[2] * row[2]).collect();
        let got = solve3(m, [rhs[0], rhs[1], rhs[2]]).unwrap();
        for (g, w) in got.iter().zip(&x) {
            assert!(g.distance(*w) < 1.0e-5, "{g:?} {w:?}");
        }
        assert!(solve3([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0]], [Vec3::ONE; 3]).is_none());
    }

    #[test]
    fn ramps_and_bumps() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-6);
        assert!((smoothstep(1.0, 0.0, 0.25) - smoothstep(0.0, 1.0, 0.75)).abs() < 1e-6);
        assert_eq!(smoothstep(1.0, 1.0, 0.5), 0.0);
        assert_eq!(bump(0.0, 1.0), 1.0);
        assert_eq!(bump(1.0, 1.0), 0.0);
        assert_eq!(bump(f32::NAN, 1.0), 0.0);
        assert_eq!(bump(0.0, 0.0), 0.0);
        let c = Vec3::ZERO;
        assert_eq!(bump_along(c, c, Vec3::Y, 2.0, 1.0), 1.0);
        assert!(bump_along(Vec3::Y * 1.5, c, Vec3::Y, 2.0, 1.0) > 0.0);
        assert_eq!(bump_along(Vec3::X * 1.5, c, Vec3::Y, 2.0, 1.0), 0.0);
        assert!((segment_distance(Vec3::new(0.5, 1.0, 0.0), Vec3::ZERO, Vec3::X) - 1.0).abs() < 1e-6);
        assert!((segment_distance(Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO, Vec3::X) - 1.0).abs() < 1e-6);
        assert_eq!(mirror(Vec3::new(1.0, 2.0, 3.0)), Vec3::new(-1.0, 2.0, 3.0));
        assert_eq!(PartKind::of("body"), PartKind::Body);
        assert_eq!(PartKind::of("tear-line-L"), PartKind::Helper);
        assert_eq!(PartKind::of("mouth-helper"), PartKind::Helper);
        assert_eq!(PartKind::of("teeth-upper"), PartKind::Rigid);
    }
}
