//! The hair-card map baker `[BAKE-06]` for the hair-card LOD generator `[HAIR-10]`.
//!
//! A card is a rectangle in space onto which a bundle of strands of the groom is projected
//! orthographically along its normal. The baker draws every strand segment as a capsule whose
//! width runs linearly from the root width to the tip width, with analytic anti-aliasing, and
//! writes the maps a card shader needs.
//!
//! **Card frame** ([`Card`]): `origin` is the card corner at texel `(0, 0)` (top-left, the root
//! edge), `across` spans the card's width (texture +x, U), `along` its length from the root edge
//! to the tip edge (texture +y, V down), the normal is `normalize(along × across)` (the side from
//! which the texture reads unmirrored) and `depth` the half-thickness of the slab the strands are
//! expected in.
//!
//! **Coverage:** a segment at distance `d` (metres, in the card plane) from a texel centre, with
//! half-width `r` at that point and texel footprint `p` (geometric mean of the texel's width and
//! height), covers `(r / r') · clamp((r' − d) / p + ½, 0, 1)` of the texel with `r' = max(r, p/2)`:
//! the usual edge ramp for strands wider than a texel, and a one-texel line of reduced intensity
//! for thinner ones, so the coverage summed across a row stays the strand's width in texels. Within
//! one strand the coverage of a texel is the maximum over its segments (joints do not double);
//! strands combine as `alpha = 1 − Π(1 − cᵢ)`.
//!
//! **Maps** ([`HairCardMaps`]), all `width × height`, empty texels [`super::Coverage::Empty`]:
//!
//! - `alpha`: the combined coverage, `[0, 1]`;
//! - `depth`: `0.5 + 0.5 · (position · normal) / depth`, clamped to `[0, 1]` (0.5 = the card plane,
//!   1 = `depth` in front), coverage-weighted over the strands in the texel;
//! - `root`: the position along the strand, 0 at the root to 1 at the tip (arc length),
//!   coverage-weighted;
//! - `id`: a per-strand random value in `[0, 1)` (the hash of the strand id) of the strand with
//!   the largest coverage in the texel, for colour variation;
//! - `flow`: the unit root-to-tip direction in texture space (x right, y down), the
//!   coverage-weighted mean; `(0, 1)` where empty.

use humanart_math::{Vec2, Vec3};

use super::sampling::pcg_hash;
use super::{BakeError, BakeMap, BakeTarget, Coverage, Result};

/// The most strands one card bake takes.
pub const MAX_STRANDS: usize = 1 << 20;
/// The most points over all strands.
pub const MAX_POINTS: usize = 1 << 24;

/// Strands as structure-of-arrays `[PERF-R2]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Strands {
    /// All points, strand after strand, root first.
    pub points: Vec<Vec3>,
    /// `strand_count + 1` offsets into `points`, starting at 0; at least two points per strand.
    pub offsets: Vec<u32>,
    /// The root width of every strand, metres (≥ 0).
    pub root_width: Vec<f32>,
    /// The tip width of every strand, metres (≥ 0).
    pub tip_width: Vec<f32>,
    /// The id of every strand (stable across bakes: the groom's strand index).
    pub ids: Vec<u32>,
}

impl Strands {
    /// The number of strands.
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Whether there are no strands.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Checks the arrays.
    pub fn check(&self) -> Result<()> {
        let n = self.len();
        if n > MAX_STRANDS || self.points.len() > MAX_POINTS {
            return Err(BakeError::invalid("hair card: too many strands or points"));
        }
        if (self.offsets.is_empty() && !self.points.is_empty())
            || self.offsets.first().is_some_and(|&o| o != 0)
            || self.offsets.last().is_some_and(|&o| o as usize != self.points.len())
            || self.offsets.windows(2).any(|w| w[1] < w[0].saturating_add(2))
        {
            return Err(BakeError::invalid(
                "hair card: strand offsets must rise by at least 2 and end at the point count",
            ));
        }
        if self.root_width.len() != n || self.tip_width.len() != n || self.ids.len() != n {
            return Err(BakeError::invalid("hair card: per-strand arrays differ in length"));
        }
        let ok = |w: &f32| *w >= 0.0 && w.is_finite();
        if self.points.iter().any(|p| !p.is_finite())
            || !self.root_width.iter().all(ok)
            || !self.tip_width.iter().all(ok)
        {
            return Err(BakeError::invalid("hair card: a point or width is not finite and non-negative"));
        }
        Ok(())
    }
}

/// The card rectangle (see the module documentation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Card {
    /// The corner at texel `(0, 0)`, on the root edge.
    pub origin: Vec3,
    /// The width vector (texture +x).
    pub across: Vec3,
    /// The length vector, root edge to tip edge (texture +y).
    pub along: Vec3,
    /// Half the thickness of the strand slab, metres (> 0).
    pub depth: f32,
}

/// The maps of [`bake_hair_card`].
#[derive(Clone, Debug, PartialEq)]
pub struct HairCardMaps {
    /// Coverage.
    pub alpha: BakeMap,
    /// Normalised depth.
    pub depth: BakeMap,
    /// Root-to-tip parameter.
    pub root: BakeMap,
    /// Per-strand random id.
    pub id: BakeMap,
    /// Root-to-tip direction (2 channels).
    pub flow: BakeMap,
}

/// The `[0, 1)` random value of strand id `id`.
pub fn strand_value(id: u32) -> f32 {
    (pcg_hash(id) >> 8) as f32 / 16_777_216.0
}

/// One strand's contribution to a texel.
#[derive(Clone, Copy, Default)]
struct Fragment {
    coverage: f32,
    depth: f32,
    root: f32,
    flow: Vec2,
}

/// Bakes the card maps of `strands` on `card` at `width × height`.
pub fn bake_hair_card(strands: &Strands, card: &Card, width: u32, height: u32) -> Result<HairCardMaps> {
    strands.check()?;
    let target = BakeTarget::new(0, width, height);
    target.check()?;
    let (la, ll) = (card.across.length(), card.along.length());
    let normal = card.along.cross(card.across);
    let Some(normal) = normal.try_normalize() else {
        return Err(BakeError::invalid("hair card: the card axes are zero or parallel"));
    };
    if !(la.is_finite() && ll.is_finite() && card.origin.is_finite() && card.depth > 0.0 && card.depth.is_finite()) {
        return Err(BakeError::invalid("hair card: the card is not finite or has no depth"));
    }
    // In-plane axes in metres (the along axis made orthogonal to across).
    let ex = card.across / la;
    let ey = (card.along - ex * card.along.dot(ex)).normalize_or(normal.cross(ex));
    let (sx, sy) = (la / width as f32, card.along.dot(ey) / height as f32);
    if sy.is_nan() || sy <= 0.0 {
        return Err(BakeError::invalid("hair card: the length axis points backwards"));
    }
    let footprint = (sx * sy).sqrt();
    let n = target.texel_count();
    let mut transmit = vec![1.0_f32; n];
    let mut sums = vec![[0.0_f32; 5]; n]; // weight, depth, root, flow x, flow y
    let mut best = vec![(0.0_f32, 0.0_f32); n]; // (coverage, id value)
    let mut strand_frag: Vec<Fragment> = vec![Fragment::default(); n];
    let mut touched: Vec<usize> = Vec::new();
    for s in 0..strands.len() {
        let (a, b) = (strands.offsets[s] as usize, strands.offsets[s + 1] as usize);
        let pts = &strands.points[a..b];
        let total: f32 = pts.windows(2).map(|w| w[0].distance(w[1])).sum();
        let mut walked = 0.0;
        for seg in pts.windows(2) {
            let len = seg[0].distance(seg[1]);
            let (t0, t1) = if total > 0.0 { (walked / total, (walked + len) / total) } else { (0.0, 1.0) };
            walked += len;
            let local = |p: Vec3| {
                let q = p - card.origin;
                (Vec2::new(q.dot(ex), q.dot(ey)), q.dot(normal))
            };
            let ((p0, d0), (p1, d1)) = (local(seg[0]), local(seg[1]));
            let width_at = |t: f32| strands.root_width[s] + (strands.tip_width[s] - strands.root_width[s]) * t;
            let r_max = 0.5 * width_at(t0).max(width_at(t1)).max(footprint);
            let dir = (p1 - p0).normalize_or(Vec2::new(0.0, 1.0));
            let flow = Vec2::new(dir.x, dir.y);
            let reach = r_max + footprint;
            let x0 = ((p0.x.min(p1.x) - reach) / sx).floor().max(0.0);
            let x1 = ((p0.x.max(p1.x) + reach) / sx).ceil().min(width as f32 - 1.0);
            let y0 = ((p0.y.min(p1.y) - reach) / sy).floor().max(0.0);
            let y1 = ((p0.y.max(p1.y) + reach) / sy).ceil().min(height as f32 - 1.0);
            if !(x0 <= x1 && y0 <= y1) {
                continue;
            }
            let seg_vec = p1 - p0;
            let seg_len2 = seg_vec.length_squared();
            for y in y0 as u32..=y1 as u32 {
                for x in x0 as u32..=x1 as u32 {
                    let c = Vec2::new((x as f32 + 0.5) * sx, (y as f32 + 0.5) * sy);
                    let h = if seg_len2 > 0.0 { ((c - p0).dot(seg_vec) / seg_len2).clamp(0.0, 1.0) } else { 0.0 };
                    let dist = c.distance(p0 + seg_vec * h);
                    let r = 0.5 * width_at(t0 + (t1 - t0) * h);
                    let re = r.max(0.5 * footprint);
                    let cov = (r / re) * ((re - dist) / footprint + 0.5).clamp(0.0, 1.0);
                    if cov <= 0.0 {
                        continue;
                    }
                    let i = (y * width + x) as usize;
                    let f = &mut strand_frag[i];
                    if f.coverage == 0.0 {
                        touched.push(i);
                    }
                    if cov > f.coverage {
                        *f = Fragment { coverage: cov, depth: d0 + (d1 - d0) * h, root: t0 + (t1 - t0) * h, flow };
                    }
                }
            }
        }
        let value = strand_value(strands.ids[s]);
        for &i in &touched {
            let f = strand_frag[i];
            transmit[i] *= 1.0 - f.coverage;
            let w = f.coverage;
            let sum = &mut sums[i];
            sum[0] += w;
            sum[1] += w * f.depth;
            sum[2] += w * f.root;
            sum[3] += w * f.flow.x;
            sum[4] += w * f.flow.y;
            if f.coverage > best[i].0 {
                best[i] = (f.coverage, value);
            }
            strand_frag[i] = Fragment::default();
        }
        touched.clear();
    }
    let mut maps = HairCardMaps {
        alpha: BakeMap::new(&target, 1, 0.0)?,
        depth: BakeMap::new(&target, 1, 0.5)?,
        root: BakeMap::new(&target, 1, 0.0)?,
        id: BakeMap::new(&target, 1, 0.0)?,
        flow: BakeMap::new(&target, 2, 0.0)?,
    };
    for i in 0..n {
        let sum = sums[i];
        if sum[0] <= 0.0 {
            maps.flow.data[2 * i + 1] = 1.0;
            continue;
        }
        let w = sum[0];
        let depth = (0.5 + 0.5 * (sum[1] / w) / card.depth).clamp(0.0, 1.0);
        let flow = Vec2::new(sum[3], sum[4]).normalize_or(Vec2::new(0.0, 1.0));
        maps.alpha.set(i, &[1.0 - transmit[i]], Coverage::Surface);
        maps.depth.set(i, &[depth], Coverage::Surface);
        maps.root.set(i, &[sum[2] / w], Coverage::Surface);
        maps.id.set(i, &[best[i].1], Coverage::Surface);
        maps.flow.set(i, &[flow.x, flow.y], Coverage::Surface);
    }
    Ok(maps)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2 x 4 cm card facing +Z, root edge at y = 0.04 (top), tips down.
    fn card() -> Card {
        Card {
            origin: Vec3::new(-0.01, 0.04, 0.0),
            across: Vec3::new(0.02, 0.0, 0.0),
            along: Vec3::new(0.0, -0.04, 0.0),
            depth: 0.002,
        }
    }

    fn strand(x: f32, z: f32, w: f32, id: u32) -> Strands {
        let points: Vec<Vec3> = (0..=8).map(|k| Vec3::new(x, 0.038 - 0.0045 * k as f32, z)).collect();
        Strands {
            offsets: vec![0, points.len() as u32],
            points,
            root_width: vec![w],
            tip_width: vec![w],
            ids: vec![id],
        }
    }

    fn join(a: Strands, b: Strands) -> Strands {
        let n = a.points.len() as u32;
        let mut out = a;
        out.points.extend(b.points);
        out.offsets.push(n + b.offsets[1]);
        out.root_width.extend(b.root_width);
        out.tip_width.extend(b.tip_width);
        out.ids.extend(b.ids);
        out
    }

    #[test]
    fn bake_06_coverage_conserves_strand_width() {
        // 64 x 128 texels: 0.3125 mm per texel. Strands of 0.1 mm (sub-texel) and 1 mm.
        for w in [0.0001_f32, 0.001] {
            let maps = bake_hair_card(&strand(0.0013, 0.0, w, 1), &card(), 64, 128).unwrap();
            let row = 64; // a row in the middle of the strand
            let sum: f32 = (0..64).map(|x| maps.alpha.data[row * 64 + x]).sum();
            let want = w / 0.0003125;
            assert!((sum - want).abs() < 0.03 * want.max(1.0), "w {w}: {sum} vs {want}");
        }
    }

    #[test]
    fn bake_06_root_flow_depth_and_id_maps() {
        let maps = bake_hair_card(
            &join(strand(-0.004_843_75, 0.001, 0.0005, 7), strand(0.005_156_25, 0.0, 0.0005, 8)),
            &card(),
            64,
            128,
        )
        .unwrap();
        // The two strands run through the centres of texel columns 16 and 48.
        let at = |m: &BakeMap, x: usize, y: usize| m.data[(y * 64 + x) * m.channels as usize];
        assert!(at(&maps.alpha, 16, 40) > 0.9 && at(&maps.alpha, 32, 40) == 0.0);
        // Root runs from 0 at the top to 1 at the tip, monotonically along the strand.
        let r: Vec<f32> = [10, 40, 70, 100, 118].iter().map(|&y| at(&maps.root, 16, y)).collect();
        assert!(r.windows(2).all(|w| w[1] > w[0]) && r[0] < 0.2 && r[4] > 0.8, "{r:?}");
        // Flow points down the texture.
        assert!((maps.flow.data[(40 * 64 + 16) * 2 + 1] - 1.0).abs() < 1e-5);
        // Depth: the left strand is 1 mm in front of the card (half of `depth`): 0.75.
        assert!((at(&maps.depth, 16, 40) - 0.75).abs() < 1e-5);
        assert!((at(&maps.depth, 48, 40) - 0.5).abs() < 1e-5);
        assert_eq!(at(&maps.id, 16, 40), strand_value(7));
        assert_eq!(at(&maps.id, 48, 40), strand_value(8));
        assert_eq!(maps.alpha.coverage[32], Coverage::Empty);
    }

    #[test]
    fn bad_strands_and_cards_are_errors() {
        let mut s = strand(0.0, 0.0, 0.001, 1);
        s.offsets = vec![0, 1];
        assert!(bake_hair_card(&s, &card(), 8, 8).is_err());
        let s = strand(0.0, 0.0, f32::NAN, 1);
        assert!(bake_hair_card(&s, &card(), 8, 8).is_err());
        let flat = Card { along: Vec3::new(0.02, 0.0, 0.0), ..card() };
        assert!(bake_hair_card(&strand(0.0, 0.0, 0.001, 1), &flat, 8, 8).is_err());
        assert!(bake_hair_card(&strand(0.0, 0.0, 0.001, 1), &card(), 0, 8).is_err());
        assert!(bake_hair_card(&Strands::default(), &card(), 8, 8).is_ok());
    }
}
