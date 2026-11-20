//! Displacement primitives and their composition (WP-19, `[EXP-01]`).
//!
//! An expression target is authored as a sequence of [`Action`]s, each weighted per node by a
//! sparse [`Mask`]. A [`Shape`] accumulates the displacement of every welded node: each action sees
//! the result of the previous ones (a rotation after a translation rotates the moved point), so a
//! recipe reads like a sculpting session. Masks are built from geodesic distances to landmark seeds
//! ([`Mask::geodesic`]) and combined with per-node factors ([`Mask::scale`]).
//!
//! Two clean-up operations keep targets well formed: [`Shape::relax`] (Laplacian smoothing of the
//! displacement field, which removes kinks that would fold triangles) and
//! [`Shape::keep_outside`] (pushes nodes out of a sphere: eyelids over the eyeball, `[EXP-05]`).

use humanart_math::{F16, Quat, Vec3};

use super::field::{FaceField, falloff};

/// The area-ratio bounds of [`Shape::regularize`].
#[derive(Clone, Debug, PartialEq)]
pub struct AreaBounds {
    /// The smallest deformed / rest area ratio allowed.
    pub min_ratio: f32,
    /// The largest deformed / rest area ratio allowed.
    pub max_ratio: f32,
    /// Triangles with a smaller rest area (m²) are only checked for folds.
    pub min_area: f32,
    /// Parts whose triangles are only checked for folds.
    pub exempt_parts: Vec<u16>,
}

/// One displacement primitive; `w` below is the node's mask weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    /// Moves a node by `offset · w`.
    Translate {
        /// The full displacement, metres.
        offset: Vec3,
    },
    /// Rotates a node about the axis through `pivot` by `angle · w`.
    Rotate {
        /// A point on the axis.
        pivot: Vec3,
        /// The axis direction (normalised when applied).
        axis: Vec3,
        /// The full angle, radians (right-handed about `axis`).
        angle: f32,
    },
    /// Moves a node along its rest normal by `amount · w` (negative: inward).
    Normal {
        /// The full distance, metres.
        amount: f32,
    },
    /// Moves a node toward `target` by `amount · w`, never past it.
    Pull {
        /// The point pulled toward.
        target: Vec3,
        /// The full distance, metres.
        amount: f32,
    },
    /// Scales a node's offset from `centre` per axis by `1 + (factor - 1) · w`.
    Scale {
        /// The fixed point.
        centre: Vec3,
        /// The full per-axis factor.
        factor: Vec3,
    },
}

impl Action {
    /// Where the action takes point `p` with normal `normal` at weight `w`.
    #[must_use]
    pub fn apply(&self, p: Vec3, normal: Vec3, w: f32) -> Vec3 {
        match *self {
            Self::Translate { offset } => p + offset * w,
            Self::Rotate { pivot, axis, angle } => {
                let axis = axis.normalize_or_zero();
                if axis == Vec3::ZERO {
                    return p;
                }
                pivot + Quat::from_axis_angle(axis, angle * w) * (p - pivot)
            }
            Self::Normal { amount } => p + normal * (amount * w),
            Self::Pull { target, amount } => {
                let d = target - p;
                let len = d.length();
                if len <= f32::EPSILON {
                    return p;
                }
                p + d * ((amount * w).min(len) / len)
            }
            Self::Scale { centre, factor } => {
                let f = Vec3::ONE + (factor - Vec3::ONE) * w;
                centre + (p - centre) * f
            }
        }
    }
}

/// Sparse per-node weights: `(node, weight)` pairs with weights in `(0, 1]`, nodes ascending and
/// unique.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mask {
    entries: Vec<(u32, f32)>,
}

impl Mask {
    /// A mask from pairs; non-finite or non-positive weights are dropped, weights above 1 clamped,
    /// a node given twice keeps its largest weight.
    #[must_use]
    pub fn from_pairs(mut pairs: Vec<(u32, f32)>) -> Self {
        pairs.retain(|&(_, w)| w.is_finite() && w > 0.0);
        pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
        pairs.dedup_by_key(|e| e.0);
        for e in &mut pairs {
            e.1 = e.1.min(1.0);
        }
        Self { entries: pairs }
    }

    /// Weight `falloff(d, inner, outer)` of the geodesic distance `d` from `seeds`.
    #[must_use]
    pub fn geodesic(field: &FaceField<'_>, seeds: &[u32], inner: f32, outer: f32) -> Self {
        let d = field.distances(seeds, outer);
        Self::from_pairs(
            d.iter()
                .enumerate()
                .filter(|(_, x)| x.is_finite())
                .map(|(n, &x)| (n as u32, falloff(x, inner, outer)))
                .collect(),
        )
    }

    /// Every node for which `f` is positive, weighted by `f` (clamped to 1).
    #[must_use]
    pub fn from_fn(field: &FaceField<'_>, f: impl Fn(usize) -> f32) -> Self {
        Self::from_pairs((0..field.node_count()).map(|n| (n as u32, f(n))).collect())
    }

    /// The weights multiplied by `f(node)` (results outside `(0, 1]` handled as in
    /// [`Mask::from_pairs`]).
    #[must_use]
    pub fn scale(&self, f: impl Fn(usize) -> f32) -> Self {
        Self::from_pairs(self.entries.iter().map(|&(n, w)| (n, w * f(n as usize))).collect())
    }

    /// The larger weight of `self` and `other` per node.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let mut all = self.entries.clone();
        all.extend_from_slice(&other.entries);
        Self::from_pairs(all)
    }

    /// The `(node, weight)` pairs.
    #[must_use]
    pub fn entries(&self) -> &[(u32, f32)] {
        &self.entries
    }

    /// The weight of `node` (0 when absent).
    #[must_use]
    pub fn weight(&self, node: u32) -> f32 {
        self.entries.binary_search_by_key(&node, |e| e.0).map_or(0.0, |i| self.entries[i].1)
    }

    /// Whether no node has a weight.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The displacement of every welded node of one target under construction.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    disp: Vec<Vec3>,
}

impl Shape {
    /// A shape with no displacement over `node_count` nodes.
    #[must_use]
    pub fn new(node_count: usize) -> Self {
        Self { disp: vec![Vec3::ZERO; node_count] }
    }

    /// Applies `action` to the masked nodes (each node from its current displaced position).
    pub fn apply(&mut self, field: &FaceField<'_>, mask: &Mask, action: Action) {
        let (pos, nrm) = (field.positions(), field.normals());
        for &(n, w) in mask.entries() {
            let i = n as usize;
            if let (Some(d), Some(&p), Some(&nm)) = (self.disp.get_mut(i), pos.get(i), nrm.get(i)) {
                *d = action.apply(p + *d, nm, w) - p;
            }
        }
    }

    /// Adds `other · weight` to the displacement.
    pub fn add(&mut self, other: &Self, weight: f32) {
        for (d, o) in self.disp.iter_mut().zip(&other.disp) {
            *d += *o * weight;
        }
    }

    /// Multiplies every node's displacement by `f(node)`.
    pub fn scale(&mut self, f: impl Fn(usize) -> f32) {
        for (n, d) in self.disp.iter_mut().enumerate() {
            *d *= f(n);
        }
    }

    /// `iterations` Jacobi steps of Laplacian smoothing of the displacement over the nodes where
    /// `movable` is true (`lambda` in `(0, 1]` is the step); other nodes keep theirs.
    pub fn relax(&mut self, field: &FaceField<'_>, iterations: usize, lambda: f32, movable: impl Fn(usize) -> bool) {
        let active: Vec<usize> = (0..self.disp.len())
            .filter(|&n| {
                movable(n)
                    && (self.disp[n] != Vec3::ZERO
                        || field.neighbours(n).iter().any(|&m| self.disp[m as usize] != Vec3::ZERO))
            })
            .collect();
        let mut next: Vec<Vec3> = vec![Vec3::ZERO; active.len()];
        for _ in 0..iterations {
            for (k, &n) in active.iter().enumerate() {
                let nb = field.neighbours(n);
                if nb.is_empty() {
                    next[k] = self.disp[n];
                    continue;
                }
                let mut avg = Vec3::ZERO;
                for &m in nb {
                    avg += self.disp.get(m as usize).copied().unwrap_or(Vec3::ZERO);
                }
                avg /= nb.len() as f32;
                next[k] = self.disp[n] + (avg - self.disp[n]) * lambda;
            }
            for (k, &n) in active.iter().enumerate() {
                self.disp[n] = next[k];
            }
        }
    }

    /// Pushes the masked nodes radially out of the sphere around `centre` whose radius in the
    /// direction of a node is `radius(direction)`, so their displaced position keeps at least that
    /// distance (eyelid over eyeball contact, `[EXP-05]`).
    pub fn keep_outside(&mut self, field: &FaceField<'_>, mask: &Mask, centre: Vec3, radius: impl Fn(Vec3) -> f32) {
        let pos = field.positions();
        for &(n, _) in mask.entries() {
            let i = n as usize;
            let (Some(d), Some(&p)) = (self.disp.get_mut(i), pos.get(i)) else { continue };
            let q = p + *d - centre;
            let len = q.length();
            if len <= f32::EPSILON {
                continue;
            }
            let dir = q / len;
            let r = radius(dir);
            if len < r {
                *d = centre + dir * r - p;
            }
        }
    }

    /// The triangles of `field` that the displacement, rounded to f16 as templates store it,
    /// turns over (normal more than 90 degrees from its rest direction).
    #[must_use]
    pub fn inverted(&self, field: &FaceField<'_>) -> Vec<usize> {
        let pos = field.positions();
        let at = |n: u32| pos.get(n as usize).copied().unwrap_or(Vec3::ZERO);
        // Templates store deltas as f16: test the displacement as it will be stored.
        let q = |x: f32| F16::from_f32(x).to_f32();
        let moved = |n: u32| {
            let d = self.disp.get(n as usize).copied().unwrap_or(Vec3::ZERO);
            at(n) + Vec3::new(q(d.x), q(d.y), q(d.z))
        };
        field
            .triangles()
            .iter()
            .enumerate()
            .filter(|(_, t)| t.iter().any(|&n| self.disp.get(n as usize).is_some_and(|d| *d != Vec3::ZERO)))
            .filter(|(_, t)| {
                let n0 = (at(t[1]) - at(t[0])).cross(at(t[2]) - at(t[0]));
                let n1 = (moved(t[1]) - moved(t[0])).cross(moved(t[2]) - moved(t[0]));
                n0.dot(n1) < 0.0
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Local fold repair: while some triangle is turned over (at most `rounds` times), relaxes the
    /// displacement on the nodes of the turned triangles and their neighbours; then gives the nodes
    /// of any triangle still turned over (slivers thinner than the f16 rounding) their mean
    /// displacement. Returns how many triangles are still turned over.
    pub fn repair_folds(&mut self, field: &FaceField<'_>, rounds: usize) -> usize {
        let mut bad = self.inverted(field);
        let mut zone: Vec<bool> = vec![false; self.disp.len()];
        for _ in 0..rounds {
            if bad.is_empty() {
                break;
            }
            zone.fill(false);
            for &t in &bad {
                for &n in field.triangles().get(t).into_iter().flatten() {
                    if let Some(z) = zone.get_mut(n as usize) {
                        *z = true;
                    }
                    for &m in field.neighbours(n as usize) {
                        if let Some(z) = zone.get_mut(m as usize) {
                            *z = true;
                        }
                    }
                }
            }
            self.relax(field, 3, 0.5, |n| zone.get(n).copied().unwrap_or(false));
            bad = self.inverted(field);
        }
        // Slivers thinner than the f16 rounding of their displacement: move them as one piece.
        // Jacobi style (every mean from the same state, then applied), so the result does not
        // depend on the triangle order and mirrored targets get mirrored repairs.
        let mut sum: Vec<Vec3> = vec![Vec3::ZERO; self.disp.len()];
        let mut count: Vec<u32> = vec![0; self.disp.len()];
        for _ in 0..rounds {
            if bad.is_empty() {
                break;
            }
            for &t in &bad {
                let Some(tri) = field.triangles().get(t) else { continue };
                let mean = tri
                    .iter()
                    .map(|&n| self.disp.get(n as usize).copied().unwrap_or(Vec3::ZERO))
                    .fold(Vec3::ZERO, |a, d| a + d)
                    / 3.0;
                for &n in tri {
                    if let (Some(s), Some(c)) = (sum.get_mut(n as usize), count.get_mut(n as usize)) {
                        *s += mean;
                        *c += 1;
                    }
                }
            }
            for ((d, s), c) in self.disp.iter_mut().zip(&mut sum).zip(&mut count) {
                if *c > 0 {
                    *d = *s / *c as f32;
                    *s = Vec3::ZERO;
                    *c = 0;
                }
            }
            bad = self.inverted(field);
        }
        bad.len()
    }

    /// The triangles of `field` that the displacement (rounded to f16) turns over, or whose area
    /// ratio (deformed / rest) leaves `bounds` while their rest area is at least
    /// [`AreaBounds::min_area`] and their first node's part is not exempt.
    #[must_use]
    pub fn bad_triangles(&self, field: &FaceField<'_>, bounds: &AreaBounds) -> Vec<usize> {
        let pos = field.positions();
        let at = |n: u32| pos.get(n as usize).copied().unwrap_or(Vec3::ZERO);
        let q = |x: f32| F16::from_f32(x).to_f32();
        let moved = |n: u32| {
            let d = self.disp.get(n as usize).copied().unwrap_or(Vec3::ZERO);
            at(n) + Vec3::new(q(d.x), q(d.y), q(d.z))
        };
        field
            .triangles()
            .iter()
            .enumerate()
            .filter(|(_, t)| t.iter().any(|&n| self.disp.get(n as usize).is_some_and(|d| *d != Vec3::ZERO)))
            .filter(|(_, t)| {
                let n0 = (at(t[1]) - at(t[0])).cross(at(t[2]) - at(t[0]));
                let n1 = (moved(t[1]) - moved(t[0])).cross(moved(t[2]) - moved(t[0]));
                if n0.dot(n1) < 0.0 {
                    return true;
                }
                let a0 = 0.5 * n0.length();
                if a0 < bounds.min_area || bounds.exempt_parts.contains(&field.part_of(t[0] as usize)) {
                    return false;
                }
                let r = 0.5 * n1.length() / a0;
                r < bounds.min_ratio || r > bounds.max_ratio
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Regularisation against folds and over-stretch: while some triangle is bad by
    /// [`Shape::bad_triangles`] (at most `rounds` times), relaxes the displacement over the nodes
    /// of the bad triangles and `2 + round / 8` rings of neighbours where `movable` is true (the
    /// zone widens while bad triangles persist); then repairs any
    /// remaining fold ([`Shape::repair_folds`]). Returns how many triangles are still bad.
    pub fn regularize(
        &mut self,
        field: &FaceField<'_>,
        rounds: usize,
        bounds: &AreaBounds,
        movable: impl Fn(usize) -> bool,
    ) -> usize {
        let mut bad = self.bad_triangles(field, bounds);
        let mut zone: Vec<bool> = vec![false; self.disp.len()];
        let mut ring: Vec<u32> = Vec::new();
        for round in 0..rounds {
            if bad.is_empty() {
                break;
            }
            zone.fill(false);
            ring.clear();
            // Breadth-first: every node enters `ring` once (marked in `zone` when it does). Without
            // the mark, each ring re-adds every neighbour of every listed node and the list grows
            // about sixfold per ring (8 GiB on 2026-11-20 when bad triangles persisted).
            for &t in &bad {
                for &n in field.triangles().get(t).into_iter().flatten() {
                    if let Some(z) = zone.get_mut(n as usize)
                        && !*z
                    {
                        *z = true;
                        ring.push(n);
                    }
                }
            }
            let mut first = 0;
            // The zone widens while the bad triangles persist: a compression that local
            // smoothing cannot spread needs a larger area to spread over.
            for _ in 0..2 + round / 8 {
                let last = ring.len();
                for k in first..last {
                    let n = ring[k] as usize;
                    for &m in field.neighbours(n) {
                        if let Some(z) = zone.get_mut(m as usize)
                            && !*z
                        {
                            *z = true;
                            ring.push(m);
                        }
                    }
                }
                first = last;
            }
            self.relax(field, 3, 0.5, |n| zone.get(n).copied().unwrap_or(false) && movable(n));
            bad = self.bad_triangles(field, bounds);
        }
        self.repair_folds(field, rounds);
        self.bad_triangles(field, bounds).len()
    }

    /// The displacement per node.
    #[must_use]
    pub fn displacements(&self) -> &[Vec3] {
        &self.disp
    }

    /// The displacement per node, mutable.
    pub fn displacements_mut(&mut self) -> &mut [Vec3] {
        &mut self.disp
    }

    /// The largest node displacement, metres.
    #[must_use]
    pub fn max_displacement(&self) -> f32 {
        self.disp.iter().map(|d| d.length()).fold(0.0, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::{Template, fixtures};

    #[test]
    fn primitives() {
        let p = Vec3::new(1.0, 0.0, 0.0);
        assert_eq!(Action::Translate { offset: Vec3::Y }.apply(p, Vec3::Z, 0.5), Vec3::new(1.0, 0.5, 0.0));
        let r = Action::Rotate { pivot: Vec3::ZERO, axis: Vec3::Z, angle: std::f32::consts::FRAC_PI_2 }.apply(
            p,
            Vec3::Z,
            1.0,
        );
        assert!((r - Vec3::Y).length() < 1e-6);
        assert_eq!(Action::Rotate { pivot: Vec3::ZERO, axis: Vec3::ZERO, angle: 1.0 }.apply(p, Vec3::Z, 1.0), p);
        assert_eq!(Action::Normal { amount: -0.2 }.apply(p, Vec3::Z, 0.5), Vec3::new(1.0, 0.0, -0.1));
        // A pull never overshoots its target.
        assert_eq!(Action::Pull { target: Vec3::ZERO, amount: 5.0 }.apply(p, Vec3::Z, 1.0), Vec3::ZERO);
        assert_eq!(Action::Pull { target: Vec3::ZERO, amount: 0.25 }.apply(p, Vec3::Z, 1.0), Vec3::new(0.75, 0.0, 0.0));
        let s = Action::Scale { centre: Vec3::ZERO, factor: Vec3::new(3.0, 1.0, 1.0) }.apply(p, Vec3::Z, 0.5);
        assert_eq!(s, Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn masks_and_shapes() {
        let m = Mask::from_pairs(vec![(3, 0.5), (1, 2.0), (3, 0.7), (2, -1.0), (4, f32::NAN)]);
        assert_eq!(m.entries(), &[(1, 1.0), (3, 0.7)]);
        assert_eq!(m.weight(3), 0.7);
        assert_eq!(m.weight(2), 0.0);
        let u = m.union(&Mask::from_pairs(vec![(2, 0.1), (3, 0.9)]));
        assert_eq!(u.entries(), &[(1, 1.0), (2, 0.1), (3, 0.9)]);
        assert_eq!(m.scale(|n| if n == 1 { 0.5 } else { 0.0 }).entries(), &[(1, 0.5)]);

        let t = Template::new(fixtures::cube_template()).unwrap();
        let f = FaceField::new(&t).unwrap();
        let all = Mask::from_fn(&f, |_| 1.0);
        assert_eq!(all.entries().len(), f.node_count());
        let g = Mask::geodesic(&f, &[0], 0.0, 0.5);
        assert_eq!(g.weight(0), 1.0);
        assert!(g.entries().len() < f.node_count());
        let mut s = Shape::new(f.node_count());
        s.apply(&f, &all, Action::Translate { offset: Vec3::X });
        s.apply(&f, &Mask::from_pairs(vec![(0, 1.0)]), Action::Translate { offset: Vec3::X });
        assert!((s.max_displacement() - 2.0).abs() < 1e-6);
        // Relaxing spreads the spike toward its neighbours' displacement.
        s.relax(&f, 3, 0.5, |_| true);
        assert!(s.max_displacement() < 2.0 && s.max_displacement() > 1.0);
        let mut c = Shape::new(f.node_count());
        let centre = f.positions().iter().fold(Vec3::ZERO, |a, &p| a + p) / f.node_count() as f32;
        c.keep_outside(&f, &all, centre, |_| 10.0);
        for (n, d) in c.displacements().iter().enumerate() {
            assert!(((f.positions()[n] + *d - centre).length() - 10.0).abs() < 1e-4);
        }
        // A displacement that turns triangles over is repaired locally.
        let mut fold = Shape::new(f.node_count());
        let far = f.positions()[0] - centre;
        fold.displacements_mut()[0] = -far * 3.0;
        assert!(!fold.inverted(&f).is_empty());
        assert_eq!(fold.repair_folds(&f, 50), 0);
        let mut z = c.clone();
        z.scale(|_| 0.0);
        assert_eq!(z.max_displacement(), 0.0);
        z.add(&c, 1.0);
        assert_eq!(z, c);
    }
}
