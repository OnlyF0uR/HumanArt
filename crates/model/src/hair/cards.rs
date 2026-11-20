//! Hair-card geometry for game LODs `[HAIR-10]` `[HR-04]`: strands grouped into cards, one ribbon
//! per card following the group's mean centreline, and the atlas layout the card maps are baked
//! into (the baking is `humanart-tools::groom::cards`, with the `[BAKE-06]` card baker).
//!
//! **Grouping.** Strands are clustered by k-means on a feature of three points per strand (root
//! weighted twice, middle, tip), so a card holds strands that start together and go the same way.
//! The centres are fitted on a deterministic sample of at most [`KMEANS_SAMPLE`] strands
//! (seeded [`Rng`]), then every strand is assigned to its nearest centre. Clusters that end up
//! empty are dropped, so a set may have fewer cards than asked for.
//!
//! **Card frame.** The centreline is the mean of the members' points at `segments + 1` equally
//! spaced point parameters. At each centreline point the card's `across` axis is
//! `normalize(n × t)` (`n` the members' mean root normal, `t` the centreline tangent): the card
//! lies flat over the head and faces out, as hand-made cards do. Its `normal` is `t × across`
//! re-orthogonalised. The width is four standard deviations of the members' offsets along
//! `across` (the widest over the centreline, at least [`MIN_CARD_WIDTH`]); the depth three along
//! `normal` (at least [`MIN_CARD_DEPTH`]).
//!
//! **Straightened space** ([`HairCard::straighten`]): x across the card from its left edge
//! (`0..width`), y the arc length along the centreline from the root edge, z the offset along the
//! normal. The baker draws the members in this space on a flat card, so a curved ribbon gets an
//! undistorted texture.
//!
//! **Atlas.** Card `i` owns cell `(i mod cols, i div cols)` of a `cols × rows` grid; its UVs run
//! `u` across the cell, `v` from the root edge (top of the cell, V down) to the tip.

use humanart_core::rng::Rng;
use humanart_math::{Vec2, Vec3};

use super::error::{HairError, HairResult};
use super::strands::{MAX_STRAND_POINTS, MAX_STRANDS};
use super::surface::normalize_or;

/// Strands the k-means centres are fitted on, at most.
pub const KMEANS_SAMPLE: usize = 16_384;
/// k-means iterations on the sample.
pub const KMEANS_ITERATIONS: u32 = 10;
/// The most cards in one set.
pub const MAX_CARDS: u32 = 4096;
/// The narrowest card, metres.
pub const MIN_CARD_WIDTH: f32 = 0.002;
/// The thinnest card slab (half-thickness), metres.
pub const MIN_CARD_DEPTH: f32 = 0.000_5;

/// One card density.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CardSpec {
    /// Cards to fit (fewer when clusters come out empty).
    pub cards: u32,
    /// Ribbon segments along a card (1..=16).
    pub segments: u32,
}

impl CardSpec {
    /// The default LOD chain: 256, 96 and 32 cards (styling choice, not a standard).
    pub const LODS: [Self; 3] =
        [Self { cards: 256, segments: 8 }, Self { cards: 96, segments: 6 }, Self { cards: 32, segments: 4 }];
}

/// One card.
#[derive(Clone, Debug, PartialEq)]
pub struct HairCard {
    /// The strands drawn on the card (indices into the groom).
    pub members: Vec<u32>,
    /// The centreline, `segments + 1` points from the root edge to the tip edge.
    pub centre: Vec<Vec3>,
    /// The unit `across` axis at each centreline point.
    pub across: Vec<Vec3>,
    /// The unit card normal at each centreline point.
    pub normal: Vec<Vec3>,
    /// The arc length from the root edge at each centreline point (first 0).
    pub arc: Vec<f32>,
    /// The card width, metres.
    pub width: f32,
    /// Half the thickness of the strand slab, metres.
    pub depth: f32,
}

impl HairCard {
    /// The centreline length.
    #[must_use]
    pub fn length(&self) -> f32 {
        self.arc.last().copied().unwrap_or(0.0)
    }

    /// `p` in the card's straightened space (see the module documentation): the closest point of
    /// the centreline gives y, the offsets along that point's `across` and `normal` give x and z.
    #[must_use]
    pub fn straighten(&self, p: Vec3) -> Vec3 {
        let mut best = (f32::INFINITY, Vec3::ZERO);
        for k in 1..self.centre.len() {
            let (a, b) = (self.centre[k - 1], self.centre[k]);
            let d = b - a;
            let len2 = d.length_squared();
            let t = if len2 > 0.0 { ((p - a).dot(d) / len2).clamp(0.0, 1.0) } else { 0.0 };
            let q = a + d * t;
            let dist = (p - q).length_squared();
            if dist < best.0 {
                let across = normalize_or(self.across[k - 1].lerp(self.across[k], t), self.across[k]);
                let normal = normalize_or(self.normal[k - 1].lerp(self.normal[k], t), self.normal[k]);
                let y = self.arc[k - 1] + (self.arc[k] - self.arc[k - 1]) * t;
                let off = p - q;
                best = (dist, Vec3::new(off.dot(across) + 0.5 * self.width, y, off.dot(normal)));
            }
        }
        best.1
    }
}

/// A card mesh: two triangles per ribbon segment, double-sided by material.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CardMesh {
    /// Vertex positions.
    pub positions: Vec<Vec3>,
    /// Vertex normals.
    pub normals: Vec<Vec3>,
    /// Atlas UVs (origin top-left, V down).
    pub uvs: Vec<Vec2>,
    /// Triangle list.
    pub indices: Vec<u32>,
}

/// A set of cards at one density.
#[derive(Clone, Debug, PartialEq)]
pub struct CardSet {
    /// The cards; card `i` owns atlas cell `i`.
    pub cards: Vec<HairCard>,
    /// The ribbon mesh of every card.
    pub mesh: CardMesh,
    /// Atlas grid columns.
    pub cols: u32,
    /// Atlas grid rows.
    pub rows: u32,
}

/// The k-means feature of strand `i`: root (twice), middle, tip.
fn feature(points: &[Vec3], per: usize, i: usize) -> [Vec3; 3] {
    let s = &points[i * per..(i + 1) * per];
    [s[0] * 2.0, s[per / 2], s[per - 1]]
}

fn distance2(a: &[Vec3; 3], b: &[Vec3; 3]) -> f32 {
    (a[0] - b[0]).length_squared() + (a[1] - b[1]).length_squared() + (a[2] - b[2]).length_squared()
}

fn nearest(f: &[Vec3; 3], centres: &[[Vec3; 3]]) -> usize {
    let mut best = (f32::INFINITY, 0);
    for (c, centre) in centres.iter().enumerate() {
        let d = distance2(f, centre);
        if d < best.0 {
            best = (d, c);
        }
    }
    best.1
}

/// Fits cards to the strands in `points` (`per` world points each, root first) with per-strand
/// `root_normals`.
pub fn build_cards(
    points: &[Vec3],
    per: u32,
    root_normals: &[Vec3],
    spec: &CardSpec,
    seed: u64,
) -> HairResult<CardSet> {
    if !(2..=MAX_STRAND_POINTS).contains(&per) {
        return Err(HairError::limit("strand points", u64::from(MAX_STRAND_POINTS), u64::from(per)));
    }
    let per = per as usize;
    let n = root_normals.len();
    if n > MAX_STRANDS || points.len() != n * per {
        return Err(HairError::invalid("card strands", 0, "point count is not strands x points per strand"));
    }
    if n == 0 {
        return Err(HairError::invalid("card strands", 0, "no strands"));
    }
    if spec.cards == 0 || spec.cards > MAX_CARDS {
        return Err(HairError::limit("card count", u64::from(MAX_CARDS), u64::from(spec.cards)));
    }
    if !(1..=16).contains(&spec.segments) {
        return Err(HairError::limit("card segments", 16, u64::from(spec.segments)));
    }
    if let Some(i) = points.iter().position(|p| !p.is_finite()) {
        return Err(HairError::invalid("strand", i / per, "point not finite"));
    }
    let mut rng = Rng::new(seed);
    // The sample: a shuffled prefix of the strand indices.
    let mut order: Vec<u32> = (0..n as u32).collect();
    rng.shuffle(&mut order);
    let sample: Vec<[Vec3; 3]> = order.iter().take(KMEANS_SAMPLE).map(|&i| feature(points, per, i as usize)).collect();
    let k = (spec.cards as usize).min(sample.len());
    let mut centres: Vec<[Vec3; 3]> = sample[..k].to_vec();
    let mut assign = vec![0_usize; sample.len()];
    for _ in 0..KMEANS_ITERATIONS {
        for (a, f) in assign.iter_mut().zip(&sample) {
            *a = nearest(f, &centres);
        }
        let mut sum = vec![[Vec3::ZERO; 3]; k];
        let mut count = vec![0_u32; k];
        for (&a, f) in assign.iter().zip(&sample) {
            for j in 0..3 {
                sum[a][j] += f[j];
            }
            count[a] += 1;
        }
        for c in 0..k {
            if count[c] > 0 {
                let inv = 1.0 / count[c] as f32;
                centres[c] = [sum[c][0] * inv, sum[c][1] * inv, sum[c][2] * inv];
            }
        }
    }
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); k];
    for i in 0..n {
        members[nearest(&feature(points, per, i), &centres)].push(i as u32);
    }
    members.retain(|m| !m.is_empty());
    let cols = (members.len() as f32).sqrt().ceil().max(1.0) as u32;
    let rows = (members.len() as u32).div_ceil(cols);
    let mut set = CardSet { cards: Vec::with_capacity(members.len()), mesh: CardMesh::default(), cols, rows };
    for (index, m) in members.into_iter().enumerate() {
        let card = fit_card(points, per, root_normals, m, spec.segments as usize);
        append_ribbon(&mut set.mesh, &card, index as u32, cols, rows);
        set.cards.push(card);
    }
    Ok(set)
}

/// The point at parameter `u` (0 root .. 1 tip) of strand `i`.
fn strand_at(points: &[Vec3], per: usize, i: usize, u: f32) -> Vec3 {
    let x = u.clamp(0.0, 1.0) * (per - 1) as f32;
    let k = (x.floor() as usize).min(per - 2);
    let s = &points[i * per..(i + 1) * per];
    s[k].lerp(s[k + 1], x - k as f32)
}

fn fit_card(points: &[Vec3], per: usize, root_normals: &[Vec3], members: Vec<u32>, segments: usize) -> HairCard {
    let inv = 1.0 / members.len() as f32;
    let mut centre = Vec::with_capacity(segments + 1);
    for s in 0..=segments {
        let u = s as f32 / segments as f32;
        let sum = members.iter().fold(Vec3::ZERO, |acc, &i| acc + strand_at(points, per, i as usize, u));
        centre.push(sum * inv);
    }
    let n_ref = normalize_or(members.iter().fold(Vec3::ZERO, |acc, &i| acc + root_normals[i as usize]), Vec3::Z);
    let mut across = Vec::with_capacity(segments + 1);
    let mut normal = Vec::with_capacity(segments + 1);
    let mut arc = Vec::with_capacity(segments + 1);
    let mut prev_across = Vec3::ZERO;
    let mut length = 0.0;
    for s in 0..=segments {
        let t = if s == 0 {
            centre[1] - centre[0]
        } else if s == segments {
            centre[s] - centre[s - 1]
        } else {
            centre[s + 1] - centre[s - 1]
        };
        if s > 0 {
            length += (centre[s] - centre[s - 1]).length();
        }
        arc.push(length);
        let t = normalize_or(t, -Vec3::Y);
        let fallback = if prev_across == Vec3::ZERO { t.any_orthonormal_vector() } else { prev_across };
        let mut a = normalize_or(n_ref.cross(t), fallback);
        if prev_across != Vec3::ZERO && a.dot(prev_across) < 0.0 {
            a = -a;
        }
        prev_across = a;
        across.push(a);
        normal.push(normalize_or(t.cross(a), n_ref));
    }
    let mut card = HairCard { members, centre, across, normal, arc, width: MIN_CARD_WIDTH, depth: MIN_CARD_DEPTH };
    // Spread of the members around the centreline, per centreline point.
    let (mut width, mut depth) = (0.0_f32, 0.0_f32);
    for s in 0..=segments {
        let u = s as f32 / segments as f32;
        let (mut sx, mut sz) = (0.0_f32, 0.0_f32);
        for &i in &card.members {
            let off = strand_at(points, per, i as usize, u) - card.centre[s];
            sx += off.dot(card.across[s]).powi(2);
            sz += off.dot(card.normal[s]).powi(2);
        }
        width = width.max(4.0 * (sx * inv).sqrt());
        depth = depth.max(3.0 * (sz * inv).sqrt());
    }
    card.width = width.max(MIN_CARD_WIDTH);
    card.depth = depth.max(MIN_CARD_DEPTH);
    card
}

fn append_ribbon(mesh: &mut CardMesh, card: &HairCard, index: u32, cols: u32, rows: u32) {
    let base = mesh.positions.len() as u32;
    let (cx, cy) = ((index % cols) as f32, (index / cols) as f32);
    let length = card.length().max(1e-6);
    for s in 0..card.centre.len() {
        let v = (cy + card.arc[s] / length) / rows as f32;
        for side in [0.0_f32, 1.0] {
            mesh.positions.push(card.centre[s] + card.across[s] * ((side - 0.5) * card.width));
            mesh.normals.push(card.normal[s]);
            mesh.uvs.push(Vec2::new((cx + side) / cols as f32, v));
        }
    }
    for s in 0..card.centre.len().saturating_sub(1) as u32 {
        let (a, b, c, d) = (base + 2 * s, base + 2 * s + 1, base + 2 * s + 2, base + 2 * s + 3);
        mesh.indices.extend_from_slice(&[a, c, b, b, c, d]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::placement::place;
    use crate::hair::presets::preset_groom;
    use crate::hair::surface::ProxyHead;
    use crate::hair::{GroomMaps, GroomParams, ScalpMask, generate_groom};

    struct Groom {
        world: Vec<Vec3>,
        normals: Vec<Vec3>,
        per: u32,
    }

    fn groom(count: f32) -> Groom {
        let head = ProxyHead::new(32).unwrap();
        let mask = ScalpMask::new(&head.surface, head.scalp.clone()).unwrap();
        let p = preset_groom("long_straight", &head.surface, &mask, 200, None, 3).unwrap();
        let params = GroomParams { strand_count: count, ..p.params };
        let layout = place(&head.surface, &mask, &p.guides, &params, &GroomMaps::default(), 5).unwrap();
        let g = generate_groom(&head.surface, &layout, &p.guides, &params, None).unwrap();
        let normals =
            (0..layout.len()).map(|i| head.surface.frame(layout.triangle[i], layout.bary[i]).unwrap().n).collect();
        Groom { world: g.world, normals, per: g.strands.points }
    }

    #[test]
    fn hair_10_cards_cover_their_strands() {
        let g = groom(6000.0);
        for spec in CardSpec::LODS {
            let set = build_cards(&g.world, g.per, &g.normals, &spec, 1).unwrap();
            assert!(set.cards.len() as u32 <= spec.cards && set.cards.len() as u32 > spec.cards / 2);
            assert_eq!(set.cards.iter().map(|c| c.members.len()).sum::<usize>(), 6000);
            assert!(set.cols * set.rows >= set.cards.len() as u32);
            let segs = spec.segments as usize;
            assert_eq!(set.mesh.positions.len(), set.cards.len() * (segs + 1) * 2);
            assert_eq!(set.mesh.indices.len(), set.cards.len() * segs * 6);
            assert!(set.mesh.uvs.iter().all(|uv| (0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y)));
            // Most member points fall inside their card's straightened rectangle and slab.
            let (mut inside, mut total) = (0, 0);
            for card in &set.cards {
                assert!(card.across.iter().zip(&card.normal).all(|(a, n)| a.dot(*n).abs() < 1e-3));
                for &i in &card.members {
                    for k in 0..g.per as usize {
                        let q = card.straighten(g.world[i as usize * g.per as usize + k]);
                        total += 1;
                        let ok = (0.0..=card.width).contains(&q.x)
                            && (-0.01..=card.length() + 0.01).contains(&q.y)
                            && q.z.abs() <= card.depth;
                        inside += usize::from(ok);
                    }
                }
            }
            assert!(inside * 100 > total * 90, "{spec:?}: {inside} of {total}");
        }
        // Deterministic.
        let a = build_cards(&g.world, g.per, &g.normals, &CardSpec::LODS[2], 4).unwrap();
        assert_eq!(a, build_cards(&g.world, g.per, &g.normals, &CardSpec::LODS[2], 4).unwrap());
    }

    #[test]
    fn hair_10_bad_input_is_refused() {
        let g = groom(100.0);
        let spec = CardSpec::LODS[2];
        assert!(build_cards(&g.world[1..], g.per, &g.normals, &spec, 1).is_err());
        assert!(build_cards(&g.world, 1, &g.normals, &spec, 1).is_err());
        assert!(build_cards(&g.world, g.per, &g.normals, &CardSpec { cards: 0, segments: 4 }, 1).is_err());
        assert!(build_cards(&g.world, g.per, &g.normals, &CardSpec { cards: 4, segments: 99 }, 1).is_err());
        let mut nan = g.world.clone();
        nan[3] = Vec3::splat(f32::NAN);
        assert!(build_cards(&nan, g.per, &g.normals, &spec, 1).is_err());
        assert!(build_cards(&[], g.per, &[], &spec, 1).is_err());
        // More cards than strands: one card per strand at most.
        let few = build_cards(&g.world[..g.per as usize * 3], g.per, &g.normals[..3], &spec, 1).unwrap();
        assert!(few.cards.len() <= 3);
    }
}
