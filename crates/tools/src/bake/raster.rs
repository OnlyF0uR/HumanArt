//! UV-space rasterisation: which triangle covers each texel centre of a tile, and where.
//!
//! Texel `(x, y)` of a `w × h` map has its centre at `uv = ((x + ½) / w, (y + ½) / h)` (origin
//! top-left, V down). A texel belongs to a triangle when its centre lies inside the triangle's UV
//! image. Centres exactly on an edge follow the **top-left rule**, so the triangles of a mesh
//! partition the texels of an island: none is covered twice, none between two triangles is
//! dropped. The edge function of an edge is evaluated in `f64` with its end points in a canonical
//! order, so both triangles sharing an edge see the same value with opposite signs and exactly one
//! of them takes a texel lying on it. The GPU rasteriser (`shaders/bake/gbuffer.wgsl`) applies the
//! same rule with its fixed-point snapping; the two agree except on texels within a snapping step
//! of an edge.
//!
//! Where UV islands overlap (they should not in a template) the later triangle wins, as the
//! hardware does.

use super::{BakeMap, BakeMesh, BakeTarget, Coverage, Result, SurfacePoint};

/// The triangle id of a texel no triangle covers.
pub const NO_TRIANGLE: u32 = u32::MAX;

/// The rasterised surface of one tile: per texel the covering triangle and the barycentric
/// coordinates of the texel centre in it.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceMap {
    /// The tile and size.
    pub target: BakeTarget,
    /// Per texel (row-major): the covering triangle or [`NO_TRIANGLE`].
    pub triangles: Vec<u32>,
    /// Per texel: the barycentric weights `(b1, b2)` of the triangle's second and third corner
    /// (`b0 = 1 − b1 − b2`).
    pub barycentrics: Vec<[f32; 2]>,
}

impl SurfaceMap {
    /// The covered texels: `(texel index, triangle, b1, b2)`.
    pub fn covered(&self) -> impl Iterator<Item = (usize, usize, f32, f32)> + '_ {
        self.triangles
            .iter()
            .zip(&self.barycentrics)
            .enumerate()
            .filter(|(_, (t, _))| **t != NO_TRIANGLE)
            .map(|(i, (&t, b))| (i, t as usize, b[0], b[1]))
    }

    /// The number of covered texels.
    pub fn covered_count(&self) -> usize {
        self.triangles.iter().filter(|&&t| t != NO_TRIANGLE).count()
    }

    /// A map of `channels` values per texel: `texel(x, y, point, out)` fills the values of every
    /// covered texel from its surface point; the other texels keep `fill` and stay
    /// [`Coverage::Empty`].
    pub fn bake(
        &self,
        mesh: &BakeMesh,
        channels: u8,
        fill: f32,
        mut texel: impl FnMut(u32, u32, &SurfacePoint, &mut [f32]),
    ) -> Result<BakeMap> {
        let mut map = BakeMap::new(&self.target, channels, fill)?;
        let width = self.target.width as usize;
        for (i, t, b1, b2) in self.covered() {
            let Some(point) = mesh.point(t, b1, b2) else { continue };
            let mut values = [fill; 4];
            texel((i % width) as u32, (i / width) as u32, &point, &mut values[..channels as usize]);
            map.set(i, &values, Coverage::Surface);
        }
        Ok(map)
    }
}

/// Rasterises the triangles of `target.tile` into a [`SurfaceMap`].
pub fn rasterize(mesh: &BakeMesh, target: &BakeTarget) -> Result<SurfaceMap> {
    target.check()?;
    let n = target.texel_count();
    let mut map = SurfaceMap { target: *target, triangles: vec![NO_TRIANGLE; n], barycentrics: vec![[0.0; 2]; n] };
    let width = target.width as usize;
    for (t, &tile) in mesh.triangle_tiles().iter().enumerate() {
        if tile != target.tile {
            continue;
        }
        let Some(corners) = mesh.triangle(t) else { continue };
        let uv = corners.map(|v| mesh.uvs()[v]);
        triangle_texels(uv.map(|p| [f64::from(p.x), f64::from(p.y)]), target.width, target.height, |x, y, b| {
            let i = y as usize * width + x as usize;
            map.triangles[i] = t as u32;
            map.barycentrics[i] = [b[1] as f32, b[2] as f32];
        });
    }
    Ok(map)
}

/// The edge function of the directed edge `p0 → p1` at `p` (twice the signed area of
/// `p0, p1, p`), computed with the end points in lexicographic order so that the reversed edge
/// gives exactly the negated value.
fn edge(p0: [f64; 2], p1: [f64; 2], p: [f64; 2]) -> f64 {
    let raw = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    if (p0[0], p0[1]) > (p1[0], p1[1]) { -raw(p1, p0) } else { raw(p0, p1) }
}

/// Whether a texel centre exactly on the directed edge `p0 → p1` of a positively oriented
/// triangle belongs to it: the edge's inward normal `(−dy, dx)` points to +x, or along +y when
/// vertical-free (the top-left rule in a y-down raster).
fn takes_edge(p0: [f64; 2], p1: [f64; 2]) -> bool {
    let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
    -dy > 0.0 || (dy == 0.0 && dx > 0.0)
}

/// Calls `visit(x, y, [b0, b1, b2])` for every texel of a `width × height` map whose centre the
/// triangle with UV corners `uv` covers (top-left rule), with the barycentric weights of the
/// corners in their given order.
pub(crate) fn triangle_texels(uv: [[f64; 2]; 3], width: u32, height: u32, mut visit: impl FnMut(u32, u32, [f64; 3])) {
    let (w, h) = (f64::from(width), f64::from(height));
    let mut p = uv.map(|c| [c[0] * w, c[1] * h]);
    if p.iter().flatten().any(|c| !c.is_finite()) {
        return;
    }
    let mut area = edge(p[0], p[1], p[2]);
    if area == 0.0 {
        return;
    }
    // Orient positively; remember the swap to report the weights in the given corner order.
    let swapped = area < 0.0;
    if swapped {
        p.swap(1, 2);
        area = -area;
    }
    let lo = |k: usize| p[0][k].min(p[1][k]).min(p[2][k]);
    let hi = |k: usize| p[0][k].max(p[1][k]).max(p[2][k]);
    let x0 = (lo(0) - 0.5).ceil().max(0.0);
    let x1 = (hi(0) - 0.5).floor().min(w - 1.0);
    let y0 = (lo(1) - 0.5).ceil().max(0.0);
    let y1 = (hi(1) - 0.5).floor().min(h - 1.0);
    if x0 > x1 || y0 > y1 {
        return;
    }
    let take = [takes_edge(p[1], p[2]), takes_edge(p[2], p[0]), takes_edge(p[0], p[1])];
    let inside = |e: f64, k: usize| e > 0.0 || (e == 0.0 && take[k]);
    for y in y0 as u32..=y1 as u32 {
        for x in x0 as u32..=x1 as u32 {
            let c = [f64::from(x) + 0.5, f64::from(y) + 0.5];
            let e0 = edge(p[1], p[2], c);
            let e1 = edge(p[2], p[0], c);
            let e2 = edge(p[0], p[1], c);
            if inside(e0, 0) && inside(e1, 1) && inside(e2, 2) {
                let b = [e0 / area, e1 / area, e2 / area];
                visit(x, y, if swapped { [b[0], b[2], b[1]] } else { b });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use humanart_core::rng::Rng;

    use super::*;

    /// A jittered `n × n` grid of quads over `[lo, hi]²`, each split into two triangles, with
    /// both diagonals: the texel counts of all triangles add up to every texel exactly once.
    #[test]
    fn bake_08_top_left_rule_partitions_texels() {
        let mut rng = Rng::new(7);
        let (size, n) = (64_u32, 9_usize);
        // Grid points, jittered inside; the border stays on texel boundaries (whole map covered).
        let mut pts = vec![[0.0_f64; 2]; (n + 1) * (n + 1)];
        for j in 0..=n {
            for i in 0..=n {
                let mut x = i as f64 / n as f64;
                let mut y = j as f64 / n as f64;
                if i > 0 && i < n {
                    x += f64::from(rng.range_f32(-0.02, 0.02));
                }
                if j > 0 && j < n {
                    y += f64::from(rng.range_f32(-0.02, 0.02));
                }
                // Snap some interior points exactly onto texel centres to exercise ties.
                if (i + j) % 3 == 0 && i > 0 && i < n && j > 0 && j < n {
                    x = ((x * 64.0).floor() + 0.5) / 64.0;
                    y = ((y * 64.0).floor() + 0.5) / 64.0;
                }
                pts[j * (n + 1) + i] = [x, y];
            }
        }
        let mut count = vec![0_u32; (size * size) as usize];
        for j in 0..n {
            for i in 0..n {
                let a = pts[j * (n + 1) + i];
                let b = pts[j * (n + 1) + i + 1];
                let c = pts[(j + 1) * (n + 1) + i + 1];
                let d = pts[(j + 1) * (n + 1) + i];
                let tris = if (i + j) % 2 == 0 { [[a, b, c], [a, c, d]] } else { [[a, b, d], [b, c, d]] };
                for (k, tri) in tris.into_iter().enumerate() {
                    // Mixed winding: the rasteriser must not care.
                    let tri = if k == 1 { [tri[0], tri[2], tri[1]] } else { tri };
                    triangle_texels(tri, size, size, |x, y, b| {
                        assert!((b.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                        assert!(b.iter().all(|&w| w >= -1e-12));
                        count[(y * size + x) as usize] += 1;
                    });
                }
            }
        }
        assert!(
            count.iter().all(|&c| c == 1),
            "every texel exactly once: {:?}",
            count.iter().filter(|&&c| c != 1).count()
        );
    }

    #[test]
    fn bake_08_barycentrics_reproduce_the_texel_centre() {
        let tri = [[0.1, 0.2], [0.9, 0.3], [0.4, 0.95]];
        let mut n = 0;
        triangle_texels(tri, 32, 32, |x, y, b| {
            let u = b[0] * tri[0][0] + b[1] * tri[1][0] + b[2] * tri[2][0];
            let v = b[0] * tri[0][1] + b[1] * tri[1][1] + b[2] * tri[2][1];
            assert!((u - (f64::from(x) + 0.5) / 32.0).abs() < 1e-12);
            assert!((v - (f64::from(y) + 0.5) / 32.0).abs() < 1e-12);
            n += 1;
        });
        // Area 0.2775 of the unit square: about 284 of 1024 texels.
        assert!((270..300).contains(&n), "{n}");
    }

    #[test]
    fn degenerate_and_outside_triangles_cover_nothing() {
        let mut n = 0;
        triangle_texels([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], 16, 16, |_, _, _| n += 1);
        triangle_texels([[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]], 16, 16, |_, _, _| n += 1);
        triangle_texels([[f64::NAN, 0.0], [1.0, 0.0], [0.0, 1.0]], 16, 16, |_, _, _| n += 1);
        assert_eq!(n, 0);
    }
}
