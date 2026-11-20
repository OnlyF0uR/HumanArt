//! Root placement, guide interpolation weights and clump assignment `[HAIR-02]`.
//!
//! [`place`] scatters the groom's strands over a [`ScalpMask`] (a triangle set of the surface):
//! triangles are chosen by area, points uniformly inside them, and each candidate is kept with the
//! probability `density map x hairline factor`, where the hairline factor fades the density over
//! `hairline.softness` metres from the mask boundary, the distance perturbed by value noise
//! (`hairline.irregularity`). Every strand then gets the nearest `strands.guides` guides with
//! inverse-square-distance weights, and the nearest of `clump.count` clump centres (themselves
//! strands). The result, [`GroomLayout`], is everything per strand the GPU modifiers need besides
//! the guides; it changes only when the scalp, the density, the count or the guide roots change.
//! Placement runs on the CPU and is deterministic given the seed.

use std::collections::HashMap;

use humanart_core::rng::Rng;
use humanart_math::{Vec2, Vec3};

use super::error::{HairError, HairResult};
use super::params::{GroomMaps, GroomParams};
use super::strands::{Guides, MAX_STRANDS};
use super::surface::HairSurface;
use super::{hash_u32, strand_key};

/// The triangles a groom grows on, with the area table and hairline distances placement uses.
#[derive(Clone, Debug)]
pub struct ScalpMask {
    triangles: Vec<u32>,
    cumulative: Vec<f64>,
    hairline: Vec<f32>,
}

impl ScalpMask {
    /// The mask of `triangles` on `surface` (indices out of range are an error).
    pub fn new(surface: &HairSurface, triangles: Vec<u32>) -> HairResult<Self> {
        if let Some(i) = triangles.iter().position(|&t| surface.triangle(t).is_none()) {
            return Err(HairError::invalid("mask triangle", i, "out of range"));
        }
        let mut cumulative = Vec::with_capacity(triangles.len());
        let mut sum = 0.0_f64;
        for &t in &triangles {
            sum += f64::from(surface.area(t));
            cumulative.push(sum);
        }
        let hairline = surface.distance_to_mask_boundary(&triangles);
        Ok(Self { triangles, cumulative, hairline })
    }

    /// The triangles.
    pub fn triangles(&self) -> &[u32] {
        &self.triangles
    }

    /// The total area in square metres.
    pub fn area(&self) -> f32 {
        self.cumulative.last().copied().unwrap_or(0.0) as f32
    }

    /// The distance along the surface from the mask boundary at `bary` of `triangle`.
    pub fn hairline_distance(&self, surface: &HairSurface, triangle: u32, bary: [f32; 2]) -> f32 {
        let Some([a, b, c]) = surface.triangle(triangle) else { return 0.0 };
        let d = |v: u32| self.hairline.get(v as usize).copied().filter(|x| x.is_finite()).unwrap_or(0.0);
        d(a) * (1.0 - bary[0] - bary[1]) + d(b) * bary[0] + d(c) * bary[1]
    }

    /// A root uniformly distributed over the mask's area (`None` for an empty mask).
    pub fn sample_root(&self, rng: &mut Rng) -> Option<(u32, [f32; 2])> {
        let t = self.pick(rng.f64())?;
        let (r1, r2) = (rng.f32().sqrt(), rng.f32());
        Some((t, [r1 * (1.0 - r2), r1 * r2]))
    }

    /// A triangle chosen with probability proportional to its area for `r` in `[0, 1)`.
    pub fn pick(&self, r: f64) -> Option<u32> {
        let total = *self.cumulative.last()?;
        let x = r * total;
        let i = self.cumulative.partition_point(|&c| c <= x).min(self.triangles.len() - 1);
        self.triangles.get(i).copied()
    }
}

/// Per-strand placement and interpolation data (structure of arrays): the input of the
/// generation modifiers next to the guides.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GroomLayout {
    /// Root triangle.
    pub triangle: Vec<u32>,
    /// Root barycentric `(u, v)`.
    pub bary: Vec<[f32; 2]>,
    /// The guides blended into the strand (unused slots repeat the first with weight 0).
    pub guides: Vec<[u32; 4]>,
    /// Their weights, summing to 1.
    pub weights: Vec<[f32; 4]>,
    /// The strand at the centre of this strand's clump (itself for a centre).
    pub clump: Vec<u32>,
    /// The strand's random key ([`strand_key`]).
    pub key: Vec<u32>,
    /// Map factors: length, clump, curl, frizz multipliers.
    pub factors: Vec<[f32; 4]>,
    /// Direction-map flow in the root frame's `(t, b)` plane.
    pub direction: Vec<[f32; 2]>,
    /// Colour-map offset.
    pub color: Vec<f32>,
}

impl GroomLayout {
    /// The number of strands.
    pub fn len(&self) -> usize {
        self.triangle.len()
    }

    /// No strands.
    pub fn is_empty(&self) -> bool {
        self.triangle.is_empty()
    }

    /// One strand per guide, rooted where the guide is, following only it, no clumping: the
    /// layout of grooms whose every hair is authored (brows, lashes).
    pub fn one_per_guide(guides: &Guides, surface: &HairSurface, maps: &GroomMaps, seed: u64) -> HairResult<Self> {
        guides.check(surface)?;
        let n = guides.len();
        let seed32 = (seed ^ (seed >> 32)) as u32;
        let mut layout = Self {
            triangle: guides.triangles().to_vec(),
            bary: guides.barys().to_vec(),
            guides: (0..n as u32).map(|i| [i; 4]).collect(),
            weights: vec![[1.0, 0.0, 0.0, 0.0]; n],
            clump: (0..n as u32).collect(),
            key: (0..n as u32).map(|i| strand_key(seed32, i)).collect(),
            ..Self::default()
        };
        layout.sample_maps(surface, maps);
        Ok(layout)
    }

    /// Re-samples the per-strand map factors (after a map is painted).
    pub fn sample_maps(&mut self, surface: &HairSurface, maps: &GroomMaps) {
        let n = self.len();
        self.factors.resize(n, [1.0; 4]);
        self.direction.resize(n, [0.0; 2]);
        self.color.resize(n, 0.0);
        for i in 0..n {
            let uv = surface.uv_at(self.triangle[i], self.bary[i]).unwrap_or(Vec2::ZERO);
            let (f, d, c) = maps.factors_at(uv);
            self.factors[i] = f;
            self.direction[i] = d;
            self.color[i] = c;
        }
    }

    /// Checks array lengths and indices against `surface` and `guide_count`.
    pub fn check(&self, surface: &HairSurface, guide_count: usize) -> HairResult<()> {
        let n = self.len();
        if n > MAX_STRANDS {
            return Err(HairError::limit("strand count", MAX_STRANDS as u64, n as u64));
        }
        let lens = [
            self.bary.len(),
            self.guides.len(),
            self.weights.len(),
            self.clump.len(),
            self.key.len(),
            self.factors.len(),
            self.direction.len(),
            self.color.len(),
        ];
        if lens.iter().any(|&l| l != n) {
            return Err(HairError::invalid("layout", 0, "arrays differ in length"));
        }
        for i in 0..n {
            if surface.triangle(self.triangle[i]).is_none() {
                return Err(HairError::invalid("strand", i, "root triangle out of range"));
            }
            if self.guides[i].iter().any(|&g| g as usize >= guide_count) {
                return Err(HairError::invalid("strand", i, "guide index out of range"));
            }
            if self.clump[i] as usize >= n {
                return Err(HairError::invalid("strand", i, "clump centre out of range"));
            }
        }
        Ok(())
    }
}

/// Most nearest guides a strand's weights are chosen from.
const GUIDE_CANDIDATES: usize = 8;
/// Least cosine between a guide's root tangent and the strand's for the guide to be blended.
pub const GUIDE_FRAME_AGREEMENT: f32 = 0.5;

/// A hash grid of points for nearest-neighbour queries.
#[derive(Clone, Debug)]
pub struct PointGrid {
    cell: f32,
    cells: HashMap<[i32; 3], Vec<u32>>,
    points: Vec<Vec3>,
}

impl PointGrid {
    /// Buckets `points` in cells of `cell` metres.
    pub fn new(points: Vec<Vec3>, cell: f32) -> Self {
        let cell = if cell.is_finite() && cell > 1e-6 { cell } else { 0.01 };
        let mut cells: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
        for (i, &p) in points.iter().enumerate() {
            cells.entry(Self::key(p, cell)).or_default().push(i as u32);
        }
        Self { cell, cells, points }
    }

    fn key(p: Vec3, cell: f32) -> [i32; 3] {
        let c = |v: f32| (v / cell).floor().clamp(-1e9, 1e9) as i32;
        [c(p.x), c(p.y), c(p.z)]
    }

    /// The points.
    pub fn points(&self) -> &[Vec3] {
        &self.points
    }

    /// Writes the up to `out.len()` nearest points to `p` as (index, distance), nearest first;
    /// returns how many were found. Ties go to the lower index.
    pub fn nearest(&self, p: Vec3, out: &mut [(u32, f32)]) -> usize {
        let k = out.len();
        if k == 0 || self.points.is_empty() {
            return 0;
        }
        let mut found = 0;
        let centre = Self::key(p, self.cell);
        let max_ring: i32 = 64;
        for ring in 0..=max_ring {
            for dz in -ring..=ring {
                for dy in -ring..=ring {
                    for dx in -ring..=ring {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != ring {
                            continue;
                        }
                        let key = [centre[0] + dx, centre[1] + dy, centre[2] + dz];
                        let Some(list) = self.cells.get(&key) else { continue };
                        for &i in list {
                            let d = self.points[i as usize].distance(p);
                            insert_sorted(out, &mut found, (i, d));
                        }
                    }
                }
            }
            if found == k && out[k - 1].1 <= ring as f32 * self.cell {
                return found;
            }
        }
        // Far from everything: finish by brute force.
        found = 0;
        for (i, q) in self.points.iter().enumerate() {
            insert_sorted(out, &mut found, (i as u32, q.distance(p)));
        }
        found
    }
}

fn insert_sorted(out: &mut [(u32, f32)], found: &mut usize, item: (u32, f32)) {
    let k = out.len();
    if out[..*found].iter().any(|e| e.0 == item.0) {
        return;
    }
    let better = |a: (u32, f32), b: (u32, f32)| a.1 < b.1 || (a.1 == b.1 && a.0 < b.0);
    if *found == k && !better(item, out[k - 1]) {
        return;
    }
    let mut at = (*found).min(k - 1);
    if *found < k {
        *found += 1;
    }
    out[at] = item;
    while at > 0 && better(out[at], out[at - 1]) {
        out.swap(at, at - 1);
        at -= 1;
    }
}

/// Smooth 3D value noise in `[0, 1)` on a lattice of unit spacing, from [`hash_u32`].
pub fn value_noise(p: Vec3, seed: u32) -> f32 {
    let f = p.floor();
    let t = p - f;
    let s = t * t * (Vec3::splat(3.0) - t * 2.0);
    let (x, y, z) = (f.x as i32, f.y as i32, f.z as i32);
    let h = |a: i32, b: i32, c: i32| {
        let k = hash_u32(seed ^ hash_u32((a as u32) ^ hash_u32((b as u32) ^ hash_u32(c as u32))));
        (k >> 8) as f32 * (1.0 / 16_777_216.0)
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(h(x, y, z), h(x + 1, y, z), s.x);
    let x10 = lerp(h(x, y + 1, z), h(x + 1, y + 1, z), s.x);
    let x01 = lerp(h(x, y, z + 1), h(x + 1, y, z + 1), s.x);
    let x11 = lerp(h(x, y + 1, z + 1), h(x + 1, y + 1, z + 1), s.x);
    lerp(lerp(x00, x10, s.y), lerp(x01, x11, s.y), s.z)
}

/// The scale of the hairline noise: one lattice cell per centimetre.
const HAIRLINE_NOISE_SCALE: f32 = 100.0;

/// Places `params.strand_count()` strands on `mask` (fewer if the density rejects too many
/// candidates: at most 32 candidates per strand are drawn) and binds them to `guides`.
pub fn place(
    surface: &HairSurface,
    mask: &ScalpMask,
    guides: &Guides,
    params: &GroomParams,
    maps: &GroomMaps,
    seed: u64,
) -> HairResult<GroomLayout> {
    params.check()?;
    guides.check(surface)?;
    if guides.is_empty() {
        return Err(HairError::invalid("groom", 0, "no guides to interpolate"));
    }
    if mask.triangles().is_empty() || mask.area() <= 0.0 {
        return Err(HairError::invalid("groom", 0, "the scalp mask has no area"));
    }
    let target = params.strand_count() as usize;
    if target > MAX_STRANDS {
        return Err(HairError::limit("strand count", MAX_STRANDS as u64, target as u64));
    }
    let seed32 = (seed ^ (seed >> 32)) as u32;
    let mut rng = Rng::new(seed);
    let mut layout = GroomLayout::default();
    layout.triangle.reserve(target);
    layout.bary.reserve(target);
    let mut roots = Vec::with_capacity(target);
    let softness = params.hairline_softness;
    let irregularity = params.hairline_irregularity;
    let attempts = target.saturating_mul(32);
    for _ in 0..attempts {
        if layout.triangle.len() == target {
            break;
        }
        let Some(t) = mask.pick(rng.f64()) else { break };
        let (r1, r2) = (rng.f32().sqrt(), rng.f32());
        let bary = [r1 * (1.0 - r2), r1 * r2];
        let accept = rng.f32();
        let uv = surface.uv_at(t, bary).unwrap_or(Vec2::ZERO);
        let p = surface.point_at(t, bary).unwrap_or(Vec3::ZERO);
        let mut density = maps.density_at(uv);
        if softness > 0.0 {
            let noise = value_noise(p * HAIRLINE_NOISE_SCALE, seed32) * 2.0 - 1.0;
            let d = mask.hairline_distance(surface, t, bary) + softness * irregularity * noise;
            let x = (d / softness).clamp(0.0, 1.0);
            density *= x * x * (3.0 - 2.0 * x);
        }
        if accept < density {
            layout.triangle.push(t);
            layout.bary.push(bary);
            roots.push(p);
        }
    }
    let n = layout.triangle.len();

    // Guide weights: inverse square distance to the nearest guides whose root frame turns the
    // same way as the strand's. Guide shapes are blended in the root frames, so a guide across a
    // UV seam (where the frame's tangent flips) would bend the strand away from the hair around
    // it; such guides are skipped, from twice as many candidates.
    let guide_roots: Vec<Vec3> = (0..guides.len())
        .map(|g| {
            let (t, b) = guides.root(g).unwrap_or((0, [0.0, 0.0]));
            surface.point_at(t, b).unwrap_or(Vec3::ZERO)
        })
        .collect();
    let guide_tangents: Vec<Vec3> = (0..guides.len())
        .map(|g| {
            let (t, b) = guides.root(g).unwrap_or((0, [0.0, 0.0]));
            surface.frame(t, b).map_or(Vec3::ZERO, |f| f.t)
        })
        .collect();
    let spacing = (mask.area() / guides.len() as f32).sqrt();
    let grid = PointGrid::new(guide_roots, spacing);
    let k = params.guides_per_strand().min(guides.len());
    let eps2 = (0.25 * spacing) * (0.25 * spacing);
    layout.guides.reserve(n);
    layout.weights.reserve(n);
    let candidates = (2 * k).min(GUIDE_CANDIDATES).min(guides.len());
    let mut near = [(0_u32, 0.0_f32); GUIDE_CANDIDATES];
    for (i, &p) in roots.iter().enumerate() {
        let found = grid.nearest(p, &mut near[..candidates]);
        let own = surface.frame(layout.triangle[i], layout.bary[i]).map_or(Vec3::ZERO, |f| f.t);
        let agrees = |id: u32| {
            own == Vec3::ZERO || guide_tangents.get(id as usize).is_none_or(|g| g.dot(own) > GUIDE_FRAME_AGREEMENT)
        };
        let mut ids = [near[0].0; 4];
        let mut w = [0.0_f32; 4];
        let mut used = 0;
        for &(id, d) in near[..found].iter().filter(|(id, _)| agrees(*id)).take(k) {
            ids[used] = id;
            w[used] = 1.0 / (d * d + eps2);
            used += 1;
        }
        if used == 0 && found > 0 {
            // No guide around turns the same way: the nearest alone.
            ids[0] = near[0].0;
            w[0] = 1.0;
        }
        let sum: f32 = w.iter().sum();
        for x in &mut w {
            *x /= sum.max(1e-30);
        }
        layout.guides.push(ids);
        layout.weights.push(w);
    }

    // Clumps: the first `clump.count` strands (a random subset, placement is random) are centres.
    let centres = (params.clump_centres() as usize).min(n);
    layout.clump = (0..n as u32).collect();
    if centres > 0 {
        let cell = (mask.area() / centres as f32).sqrt();
        let grid = PointGrid::new(roots[..centres].to_vec(), cell);
        let mut one = [(0_u32, 0.0_f32)];
        for (i, &p) in roots.iter().enumerate().skip(centres) {
            if grid.nearest(p, &mut one) == 1 {
                layout.clump[i] = one[0].0;
            }
        }
    }
    layout.key = (0..n as u32).map(|i| strand_key(seed32, i)).collect();
    layout.sample_maps(surface, maps);
    Ok(layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::params::GroomMap;
    use crate::hair::surface::ProxyHead;

    fn setup(guides_n: usize) -> (ProxyHead, ScalpMask, Guides) {
        let head = ProxyHead::new(48).unwrap();
        let mask = ScalpMask::new(&head.surface, head.scalp.clone()).unwrap();
        let mut guides = Guides::new(8).unwrap();
        let mut rng = Rng::new(3);
        for _ in 0..guides_n {
            let t = mask.pick(rng.f64()).unwrap();
            guides.push(t, [0.3, 0.3], &[Vec3::ZERO, Vec3::new(0.0, 0.02, 0.05)]).unwrap();
        }
        (head, mask, guides)
    }

    #[test]
    fn nearest_matches_brute_force() {
        let mut rng = Rng::new(1);
        let pts: Vec<Vec3> = (0..500).map(|_| Vec3::new(rng.f32(), rng.f32(), rng.f32())).collect();
        let grid = PointGrid::new(pts.clone(), 0.07);
        for _ in 0..100 {
            let p = Vec3::new(rng.range_f32(-0.5, 1.5), rng.f32(), rng.f32());
            let mut out = [(0, 0.0); 3];
            assert_eq!(grid.nearest(p, &mut out), 3);
            let mut brute: Vec<(u32, f32)> = pts.iter().enumerate().map(|(i, q)| (i as u32, q.distance(p))).collect();
            brute.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            assert_eq!(&out[..], &brute[..3]);
        }
    }

    #[test]
    fn ac_wp17_1_placement_count_roots_and_determinism() {
        let (head, mask, guides) = setup(200);
        let mut params = GroomParams { strand_count: 20_000.0, ..GroomParams::default() };
        let maps = GroomMaps::default();
        let a = place(&head.surface, &mask, &guides, &params, &maps, 42).unwrap();
        let b = place(&head.surface, &mask, &guides, &params, &maps, 42).unwrap();
        assert_eq!(a, b, "deterministic given the seed");
        let c = place(&head.surface, &mask, &guides, &params, &maps, 43).unwrap();
        assert_ne!(a.triangle, c.triangle);
        assert_eq!(a.len(), 20_000);
        a.check(&head.surface, guides.len()).unwrap();
        // Roots are on the scalp mask, weights sum to one.
        let scalp: std::collections::HashSet<u32> = head.scalp.iter().copied().collect();
        assert!(a.triangle.iter().all(|t| scalp.contains(t)));
        assert!(a.weights.iter().all(|w| (w.iter().sum::<f32>() - 1.0).abs() < 1e-5));
        // 400 clump centres are their own centre.
        assert_eq!(a.clump.iter().enumerate().filter(|&(i, &c)| i as u32 == c).count(), 400);
        // The hairline fades the density: few roots within 2 mm of the boundary.
        let near =
            (0..a.len()).filter(|&i| mask.hairline_distance(&head.surface, a.triangle[i], a.bary[i]) < 0.002).count();
        let area_share = 0.002 * 2.0 / 0.5;
        assert!((near as f32) < 20_000.0 * area_share, "{near}");
        // A zero density map places nothing.
        let maps0 = GroomMaps { density: Some(GroomMap::new(4, 4, 1, &[0.0]).unwrap()), ..GroomMaps::default() };
        params.strand_count = 100.0;
        assert_eq!(place(&head.surface, &mask, &guides, &params, &maps0, 1).unwrap().len(), 0);
        assert!(place(&head.surface, &mask, &Guides::new(4).unwrap(), &params, &maps, 1).is_err());
    }

    #[test]
    fn one_per_guide_binds_each_strand_to_its_guide() {
        let (head, _, guides) = setup(10);
        let l = GroomLayout::one_per_guide(&guides, &head.surface, &GroomMaps::default(), 5).unwrap();
        assert_eq!(l.len(), 10);
        assert_eq!(l.guides[3], [3; 4]);
        assert_eq!(l.clump[7], 7);
        l.check(&head.surface, 10).unwrap();
        assert!(l.check(&head.surface, 3).is_err());
    }

    #[test]
    fn value_noise_is_smooth_and_bounded() {
        let mut last = value_noise(Vec3::new(0.0, 0.3, 0.7), 1);
        for i in 1..1000 {
            let v = value_noise(Vec3::new(i as f32 * 0.01, 0.3, 0.7), 1);
            assert!((0.0..1.0).contains(&v));
            assert!((v - last).abs() < 0.05);
            last = v;
        }
    }
}
