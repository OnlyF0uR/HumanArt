//! The groom data model `[HAIR-01]`: guide strands (what artists and agents edit) and generated
//! strands (what the generator produces and the renderer and exporters consume), both
//! structure-of-arrays.
//!
//! A strand is a polyline of a fixed number of control points per set. Point 0 is the root; every
//! point is an offset in the root's [`RootFrame`](super::surface::RootFrame), attached to a surface
//! triangle through barycentric coordinates, so strands ride the deformed skin. Guides keep `f32`
//! offsets (they are edited repeatedly); generated strands store them as `f16` as `[HAIR-01]`
//! requires, with the per-strand attributes length, root and tip thickness, colour variation id,
//! curl phase, clump id, stiffness hint and grey (unpigmented) level.

use humanart_math::{F16, Vec3};

use super::error::{HairError, HairResult};
use super::surface::{HairSurface, RootFrame};

/// The most control points of one guide.
pub const MAX_GUIDE_POINTS: u32 = 64;
/// The most control points of one generated strand (the GPU modifiers keep a strand in registers).
pub const MAX_STRAND_POINTS: u32 = 32;
/// The most guides of one groom.
pub const MAX_GUIDES: usize = 1 << 18;
/// The most strands of one groom.
pub const MAX_STRANDS: usize = 1 << 22;

/// Checks a root: barycentric coordinates in the triangle and finite.
fn check_root(item: &'static str, i: usize, triangle: u32, bary: [f32; 2], surface: &HairSurface) -> HairResult<()> {
    if surface.triangle(triangle).is_none() {
        return Err(HairError::invalid(item, i, "root triangle out of range"));
    }
    let [u, v] = bary;
    let ok = u.is_finite() && v.is_finite() && u >= -1e-4 && v >= -1e-4 && u + v <= 1.0 + 1e-4;
    if ok { Ok(()) } else { Err(HairError::invalid(item, i, "root barycentric coordinates outside the triangle")) }
}

/// The length of a polyline.
pub fn polyline_length(points: &[Vec3]) -> f32 {
    points.windows(2).map(|w| (w[1] - w[0]).length()).sum()
}

/// Resamples the polyline `src` (at least one point) to `out.len()` points equally spaced along
/// its length; the first and last points are kept.
pub fn resample_polyline(src: &[Vec3], out: &mut [Vec3]) {
    let Some(&first) = src.first() else { return };
    let n = out.len();
    if n == 0 {
        return;
    }
    let total = polyline_length(src);
    if src.len() < 2 || total <= 0.0 || n == 1 {
        out.fill(first);
        return;
    }
    let (mut seg, mut acc) = (0_usize, 0.0_f32);
    for (k, slot) in out.iter_mut().enumerate() {
        let s = total * (k as f32 / (n - 1) as f32);
        while seg + 2 < src.len() && acc + (src[seg + 1] - src[seg]).length() < s {
            acc += (src[seg + 1] - src[seg]).length();
            seg += 1;
        }
        let d = src[seg + 1] - src[seg];
        let len = d.length();
        let t = if len > 1e-12 { ((s - acc) / len).clamp(0.0, 1.0) } else { 0.0 };
        *slot = src[seg] + d * t;
    }
}

/// Guide strands: `points` control points each, `f32` offsets in the root frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Guides {
    points: u32,
    triangle: Vec<u32>,
    bary: Vec<[f32; 2]>,
    offsets: Vec<Vec3>,
}

impl Guides {
    /// An empty set with `points` control points per guide (2..=[`MAX_GUIDE_POINTS`]).
    pub fn new(points: u32) -> HairResult<Self> {
        if !(2..=MAX_GUIDE_POINTS).contains(&points) {
            return Err(HairError::limit("guide points", u64::from(MAX_GUIDE_POINTS), u64::from(points)));
        }
        Ok(Self { points, triangle: Vec::new(), bary: Vec::new(), offsets: Vec::new() })
    }

    /// A set from its arrays as stored (no resampling): `offsets` holds `points` entries per guide.
    /// Lengths, the point count, finiteness and the limit are checked; roots are checked against a
    /// surface by [`Guides::check`].
    pub fn from_raw(points: u32, triangle: Vec<u32>, bary: Vec<[f32; 2]>, offsets: Vec<Vec3>) -> HairResult<Self> {
        let mut g = Self::new(points)?;
        if triangle.len() > MAX_GUIDES {
            return Err(HairError::limit("guide count", MAX_GUIDES as u64, triangle.len() as u64));
        }
        if bary.len() != triangle.len() || offsets.len() != triangle.len() * points as usize {
            return Err(HairError::invalid("guides", 0, "arrays differ in length"));
        }
        if let Some(i) = bary.iter().position(|b| !(b[0].is_finite() && b[1].is_finite())) {
            return Err(HairError::invalid("guide", i, "barycentric root not finite"));
        }
        if let Some(i) = offsets.iter().position(|p| !p.is_finite()) {
            return Err(HairError::invalid("guide", i / points as usize, "offset not finite"));
        }
        g.triangle = triangle;
        g.bary = bary;
        g.offsets = offsets;
        Ok(g)
    }

    /// Control points per guide.
    pub fn points(&self) -> u32 {
        self.points
    }

    /// The number of guides.
    pub fn len(&self) -> usize {
        self.triangle.len()
    }

    /// No guides.
    pub fn is_empty(&self) -> bool {
        self.triangle.is_empty()
    }

    /// Adds a guide rooted at `bary` of `triangle` with local `offsets` (resampled to the set's
    /// point count; the first is moved to the root). Returns its index.
    pub fn push(&mut self, triangle: u32, bary: [f32; 2], offsets: &[Vec3]) -> HairResult<usize> {
        if self.len() >= MAX_GUIDES {
            return Err(HairError::limit("guide count", MAX_GUIDES as u64, self.len() as u64 + 1));
        }
        if offsets.is_empty() || offsets.iter().any(|p| !p.is_finite()) {
            return Err(HairError::invalid("guide", self.len(), "no points or a point that is not finite"));
        }
        let start = self.offsets.len();
        self.offsets.resize(start + self.points as usize, Vec3::ZERO);
        let first = offsets[0];
        let shifted: Vec<Vec3> = offsets.iter().map(|&p| p - first).collect();
        resample_polyline(&shifted, &mut self.offsets[start..]);
        self.triangle.push(triangle);
        self.bary.push(bary);
        Ok(self.len() - 1)
    }

    /// Removes guide `i` (later guides move down by one). `false` if out of range.
    pub fn remove(&mut self, i: usize) -> bool {
        if i >= self.len() {
            return false;
        }
        let p = self.points as usize;
        self.triangle.remove(i);
        self.bary.remove(i);
        self.offsets.drain(i * p..(i + 1) * p);
        true
    }

    /// Keeps the guides for which `keep(index)` is true.
    pub fn retain(&mut self, mut keep: impl FnMut(usize) -> bool) {
        let p = self.points as usize;
        let mut out = 0;
        for i in 0..self.len() {
            if keep(i) {
                self.triangle[out] = self.triangle[i];
                self.bary[out] = self.bary[i];
                self.offsets.copy_within(i * p..(i + 1) * p, out * p);
                out += 1;
            }
        }
        self.triangle.truncate(out);
        self.bary.truncate(out);
        self.offsets.truncate(out * p);
    }

    /// The root of guide `i`: triangle and barycentric coordinates.
    pub fn root(&self, i: usize) -> Option<(u32, [f32; 2])> {
        Some((*self.triangle.get(i)?, *self.bary.get(i)?))
    }

    /// Moves the root of guide `i`.
    pub fn set_root(&mut self, i: usize, triangle: u32, bary: [f32; 2]) -> bool {
        match (self.triangle.get_mut(i), self.bary.get_mut(i)) {
            (Some(t), Some(b)) => {
                *t = triangle;
                *b = bary;
                true
            }
            _ => false,
        }
    }

    /// The root triangles of all guides.
    pub fn triangles(&self) -> &[u32] {
        &self.triangle
    }

    /// The root barycentric coordinates of all guides.
    pub fn barys(&self) -> &[[f32; 2]] {
        &self.bary
    }

    /// The local offsets of guide `i`.
    pub fn offsets(&self, i: usize) -> &[Vec3] {
        let p = self.points as usize;
        self.offsets.get(i * p..(i + 1) * p).unwrap_or(&[])
    }

    /// The local offsets of guide `i`, writable.
    pub fn offsets_mut(&mut self, i: usize) -> &mut [Vec3] {
        let p = self.points as usize;
        self.offsets.get_mut(i * p..(i + 1) * p).unwrap_or(&mut [])
    }

    /// Every offset, guide after guide.
    pub fn all_offsets(&self) -> &[Vec3] {
        &self.offsets
    }

    /// The length of guide `i`.
    pub fn length(&self, i: usize) -> f32 {
        polyline_length(self.offsets(i))
    }

    /// The root frame of guide `i` on `surface`.
    pub fn frame(&self, i: usize, surface: &HairSurface) -> Option<RootFrame> {
        let (t, b) = self.root(i)?;
        surface.frame(t, b)
    }

    /// The world points of guide `i` with root frame `frame`.
    pub fn world_points(&self, i: usize, frame: &RootFrame) -> Vec<Vec3> {
        self.offsets(i).iter().map(|&p| frame.to_world(p)).collect()
    }

    /// Replaces guide `i`'s shape by the world points `world` (resampled; the first point is
    /// replaced by the root) seen from `frame`.
    pub fn set_world_points(&mut self, i: usize, frame: &RootFrame, world: &[Vec3]) -> HairResult<()> {
        if i >= self.len() || world.is_empty() || world.iter().any(|p| !p.is_finite()) {
            return Err(HairError::invalid("guide", i, "out of range or points not finite"));
        }
        let local: Vec<Vec3> = world.iter().map(|&p| frame.to_local(p)).collect();
        let first = local[0];
        let shifted: Vec<Vec3> = local.iter().map(|&p| p - first).collect();
        resample_polyline(&shifted, self.offsets_mut(i));
        Ok(())
    }

    /// Every guide's offsets resampled to `points` points (equal spacing along the curve), guide
    /// after guide: the input of the generation modifiers.
    pub fn resampled(&self, points: u32) -> Vec<Vec3> {
        let p = points as usize;
        let mut out = vec![Vec3::ZERO; self.len() * p];
        for i in 0..self.len() {
            resample_polyline(self.offsets(i), &mut out[i * p..(i + 1) * p]);
        }
        out
    }

    /// Checks every root against `surface` and every offset for finiteness.
    pub fn check(&self, surface: &HairSurface) -> HairResult<()> {
        for i in 0..self.len() {
            check_root("guide", i, self.triangle[i], self.bary[i], surface)?;
        }
        if let Some(i) = self.offsets.iter().position(|p| !p.is_finite()) {
            return Err(HairError::invalid("guide", i / self.points as usize, "offset not finite"));
        }
        Ok(())
    }
}

/// Generated strands `[HAIR-01]`: `points` control points each, structure of arrays.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Strands {
    /// Control points per strand (2..=[`MAX_STRAND_POINTS`]).
    pub points: u32,
    /// Root triangle per strand.
    pub triangle: Vec<u32>,
    /// Root barycentric `(u, v)` per strand.
    pub bary: Vec<[f32; 2]>,
    /// Control-point offsets in the root frame, `f16`, three per point, `points` per strand.
    pub offsets: Vec<F16>,
    /// Polyline length in metres.
    pub length: Vec<f32>,
    /// Fibre diameter at the root, metres.
    pub width_root: Vec<f32>,
    /// Fibre diameter at the tip, metres.
    pub width_tip: Vec<f32>,
    /// Colour variation id (the shader maps it to a small melanin offset).
    pub color_id: Vec<u16>,
    /// Curl phase, fraction of a turn in `[0, 1)`.
    pub curl_phase: Vec<f32>,
    /// The index of the strand at the centre of this strand's clump.
    pub clump: Vec<u32>,
    /// Stiffness hint for game-side dynamics, 0 (limp) to 1 (stiff).
    pub stiffness: Vec<f32>,
    /// Unpigmented level: 0 pigmented, 255 white `[HAIR-02]` grey fraction.
    pub grey: Vec<u8>,
}

impl Strands {
    /// An empty set with `points` control points per strand.
    pub fn new(points: u32) -> HairResult<Self> {
        if !(2..=MAX_STRAND_POINTS).contains(&points) {
            return Err(HairError::limit("strand points", u64::from(MAX_STRAND_POINTS), u64::from(points)));
        }
        Ok(Self { points, ..Self::default() })
    }

    /// The number of strands.
    pub fn len(&self) -> usize {
        self.triangle.len()
    }

    /// No strands.
    pub fn is_empty(&self) -> bool {
        self.triangle.is_empty()
    }

    /// The decoded offset of point `k` of strand `i`.
    pub fn offset(&self, i: usize, k: usize) -> Vec3 {
        let at = (i * self.points as usize + k) * 3;
        match self.offsets.get(at..at + 3) {
            Some(h) => Vec3::new(h[0].to_f32(), h[1].to_f32(), h[2].to_f32()),
            None => Vec3::ZERO,
        }
    }

    /// Checks array lengths, roots against `surface` (when given), clump ids, finiteness.
    pub fn check(&self, surface: Option<&HairSurface>) -> HairResult<()> {
        let n = self.len();
        if n > MAX_STRANDS {
            return Err(HairError::limit("strand count", MAX_STRANDS as u64, n as u64));
        }
        if !(2..=MAX_STRAND_POINTS).contains(&self.points) {
            return Err(HairError::limit("strand points", u64::from(MAX_STRAND_POINTS), u64::from(self.points)));
        }
        let lens = [
            self.bary.len(),
            self.length.len(),
            self.width_root.len(),
            self.width_tip.len(),
            self.color_id.len(),
            self.curl_phase.len(),
            self.clump.len(),
            self.stiffness.len(),
            self.grey.len(),
        ];
        if lens.iter().any(|&l| l != n) || self.offsets.len() != n * self.points as usize * 3 {
            return Err(HairError::invalid("strands", 0, "attribute arrays differ in length"));
        }
        if let Some(i) = self.clump.iter().position(|&c| c as usize >= n) {
            return Err(HairError::invalid("strand", i, "clump id out of range"));
        }
        if let Some(s) = surface {
            for i in 0..n {
                check_root("strand", i, self.triangle[i], self.bary[i], s)?;
            }
        }
        let finite = |v: &[f32]| v.iter().position(|x| !x.is_finite());
        for v in [&self.length, &self.width_root, &self.width_tip, &self.curl_phase, &self.stiffness] {
            if let Some(i) = finite(v) {
                return Err(HairError::invalid("strand", i, "attribute not finite"));
            }
        }
        if let Some(i) = self.offsets.iter().position(|h| !h.to_f32().is_finite()) {
            return Err(HairError::invalid("strand", i / (self.points as usize * 3), "offset not finite"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hair::surface::ProxyHead;

    #[test]
    fn resampling_keeps_ends_and_spaces_evenly() {
        let src = [Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 3.0, 1.0)];
        let mut out = [Vec3::ZERO; 5];
        resample_polyline(&src, &mut out);
        assert_eq!(out[0], Vec3::ZERO);
        assert!((out[4] - src[2]).length() < 1e-6);
        for w in out.windows(2) {
            assert!(((w[1] - w[0]).length() - 1.0).abs() < 1e-5);
        }
        let mut one = [Vec3::X; 3];
        resample_polyline(&[Vec3::Y], &mut one);
        assert_eq!(one, [Vec3::Y; 3]);
    }

    #[test]
    fn guides_push_remove_retain_and_check() {
        let head = ProxyHead::new(16).unwrap();
        let mut g = Guides::new(8).unwrap();
        assert!(Guides::new(1).is_err() && Guides::new(65).is_err());
        let shape = [Vec3::new(0.0, 0.0, 0.01), Vec3::new(0.0, 0.0, 0.05), Vec3::new(0.0, 0.05, 0.05)];
        for t in 0..5 {
            g.push(head.scalp[t], [0.2, 0.3], &shape).unwrap();
        }
        assert_eq!(g.len(), 5);
        assert_eq!(g.offsets(0)[0], Vec3::ZERO);
        assert!((g.length(0) - 0.09).abs() < 5e-3, "the corner is cut by resampling");
        assert!(g.check(&head.surface).is_ok());
        assert!(g.remove(1) && !g.remove(9));
        g.retain(|i| i != 0);
        assert_eq!(g.len(), 3);
        assert_eq!(g.root(0).unwrap().0, head.scalp[2]);
        assert!(g.push(0, [0.0, 0.0], &[Vec3::splat(f32::NAN)]).is_err());
        g.set_root(0, u32::MAX, [0.0, 0.0]);
        assert!(g.check(&head.surface).is_err());
        let frame = g.frame(1, &head.surface).unwrap();
        let world = g.world_points(1, &frame);
        g.set_world_points(1, &frame, &world).unwrap();
        assert!((g.length(1) - 0.09).abs() < 5e-3);
        assert_eq!(g.resampled(4).len(), 12);
    }

    #[test]
    fn strands_check_their_arrays() {
        let mut s = Strands::new(2).unwrap();
        assert!(s.check(None).is_ok());
        s.triangle.push(0);
        assert!(s.check(None).is_err());
        s.bary.push([0.0, 0.0]);
        s.offsets.extend([F16::from_f32(0.0); 6]);
        s.length.push(0.1);
        s.width_root.push(1e-4);
        s.width_tip.push(1e-5);
        s.color_id.push(3);
        s.curl_phase.push(0.5);
        s.clump.push(0);
        s.stiffness.push(0.5);
        s.grey.push(0);
        assert!(s.check(None).is_ok());
        s.clump[0] = 1;
        assert!(s.check(None).is_err());
    }
}
