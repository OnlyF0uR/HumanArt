//! The UV overlap guard of the LOD generator `[MOD-05]`: live triangles by UV cell per tile, and
//! the overlap test the validator applies (`crate::validate`, UV overlap check), slightly stricter.
//!
//! A collapse that slides a UV-island boundary vertex replaces the boundary path `v - u - x` by the
//! chord `v - x`; where the island is concave at `u` the chord sweeps over UV space outside the
//! island, which can hold another island or the far side of a slit (the mouth). The local flip
//! checks cannot see that, so every triangle a collapse changes is tested against the live
//! triangles of its tile found through this grid.
//!
//! The grid is insert-only: a triangle is entered in every cell its UV bounds cover whenever it is
//! created or changed, and queries test the triangles' current corners, so stale entries only cost
//! a test.

/// The largest grid side per tile.
const MAX_SIDE: usize = 256;

/// Two interiors overlap when their projections on every edge normal overlap by more than this
/// (UV units); the validator uses `UV_EPSILON` (1e-6) in `f32`, so this is stricter.
const OVERLAP_MARGIN: f64 = 1e-7;

/// Triangles by UV cell, per tile.
#[derive(Clone, Debug, Default)]
pub(crate) struct UvGrid {
    side: usize,
    tiles: usize,
    cells: Vec<Vec<u32>>,
}

impl UvGrid {
    /// An empty grid for `tiles` tiles sized for `triangles` triangles.
    pub(crate) fn new(tiles: usize, triangles: usize) -> Self {
        let tiles = tiles.max(1);
        let side = ((triangles / tiles) as f64).sqrt().ceil().clamp(1.0, MAX_SIDE as f64) as usize;
        Self { side, tiles, cells: vec![Vec::new(); tiles * side * side] }
    }

    /// The cell range `(x0, x1, y0, y1)` covering UV bounds `lo..hi` (clamped to the tile).
    fn range(&self, lo: [f64; 2], hi: [f64; 2]) -> (usize, usize, usize, usize) {
        let s = self.side as f64;
        let cell = |x: f64| ((x * s).floor().max(0.0) as usize).min(self.side - 1);
        (cell(lo[0]), cell(hi[0]), cell(lo[1]), cell(hi[1]))
    }

    /// Enters triangle `t` of `tile` with UV corners `c`.
    pub(crate) fn insert(&mut self, tile: u16, t: u32, c: [[f64; 2]; 3]) {
        let tile = usize::from(tile).min(self.tiles - 1);
        let (lo, hi) = bounds(c);
        let (x0, x1, y0, y1) = self.range(lo, hi);
        let base = tile * self.side * self.side;
        for y in y0..=y1 {
            for x in x0..=x1 {
                self.cells[base + y * self.side + x].push(t);
            }
        }
    }

    /// Appends to `out` the triangles entered in the cells that UV corners `c` of `tile` cover
    /// (with repeats; ascending after the caller sorts).
    pub(crate) fn query(&self, tile: u16, c: [[f64; 2]; 3], out: &mut Vec<u32>) {
        let tile = usize::from(tile).min(self.tiles - 1);
        let (lo, hi) = bounds(c);
        let (x0, x1, y0, y1) = self.range(lo, hi);
        let base = tile * self.side * self.side;
        for y in y0..=y1 {
            for x in x0..=x1 {
                out.extend_from_slice(&self.cells[base + y * self.side + x]);
            }
        }
    }
}

/// The UV bounds of three corners.
fn bounds(c: [[f64; 2]; 3]) -> ([f64; 2], [f64; 2]) {
    let lo = [c[0][0].min(c[1][0]).min(c[2][0]), c[0][1].min(c[1][1]).min(c[2][1])];
    let hi = [c[0][0].max(c[1][0]).max(c[2][0]), c[0][1].max(c[1][1]).max(c[2][1])];
    (lo, hi)
}

/// Whether the interiors of two UV triangles overlap by more than [`OVERLAP_MARGIN`] (separating
/// axes along the edge normals, as the validator tests).
pub(crate) fn overlap(a: [[f64; 2]; 3], b: [[f64; 2]; 3]) -> bool {
    // A collapsed triangle has no interior (the validator skips them too).
    let area = |t: [[f64; 2]; 3]| {
        ((t[1][0] - t[0][0]) * (t[2][1] - t[0][1]) - (t[1][1] - t[0][1]) * (t[2][0] - t[0][0])).abs()
    };
    if !(area(a) > 0.0 && area(b) > 0.0) {
        return false;
    }
    for tri in [a, b] {
        for i in 0..3 {
            let (p, q) = (tri[i], tri[(i + 1) % 3]);
            let n = [p[1] - q[1], q[0] - p[0]];
            let len = (n[0] * n[0] + n[1] * n[1]).sqrt();
            if !(len.is_finite() && len > 0.0) {
                continue;
            }
            let n = [n[0] / len, n[1] / len];
            let project = |t: [[f64; 2]; 3]| {
                let d = t.map(|c| n[0] * c[0] + n[1] * c[1]);
                (d[0].min(d[1]).min(d[2]), d[0].max(d[1]).max(d[2]))
            };
            let ((a0, a1), (b0, b1)) = (project(a), project(b));
            if a1 <= b0 + OVERLAP_MARGIN || b1 <= a0 + OVERLAP_MARGIN {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mod_05_uv_overlap_matches_shared_edges_and_crossings() {
        let a = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]];
        // Sharing an edge or a corner is not an overlap; crossing or containment is.
        assert!(!overlap(a, [[0.1, 0.0], [0.1, 0.1], [0.0, 0.1]]));
        assert!(!overlap(a, [[0.1, 0.0], [0.2, 0.0], [0.2, 0.1]]));
        assert!(overlap(a, [[0.05, 0.01], [0.2, 0.01], [0.05, 0.2]]));
        assert!(overlap(a, [[0.01, 0.01], [0.02, 0.01], [0.01, 0.02]]));
        // A degenerate triangle has no interior.
        assert!(!overlap(a, [[0.01, 0.01], [0.01, 0.01], [0.01, 0.01]]));
    }

    #[test]
    fn mod_05_uv_grid_finds_triangles_by_bounds() {
        let mut g = UvGrid::new(2, 400);
        g.insert(0, 7, [[0.1, 0.1], [0.2, 0.1], [0.1, 0.2]]);
        g.insert(1, 8, [[0.1, 0.1], [0.2, 0.1], [0.1, 0.2]]);
        // Out-of-range UVs and tiles clamp instead of indexing out of bounds.
        g.insert(9, 9, [[-1.0, 2.0], [3.0, 0.5], [0.5, f64::NAN]]);
        let mut out = Vec::new();
        g.query(0, [[0.15, 0.15], [0.16, 0.15], [0.15, 0.16]], &mut out);
        assert_eq!(out, vec![7]);
        out.clear();
        g.query(0, [[0.8, 0.8], [0.9, 0.8], [0.8, 0.9]], &mut out);
        assert!(out.is_empty());
        out.clear();
        g.query(1, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], &mut out);
        out.sort_unstable();
        out.dedup();
        assert_eq!(out, vec![8, 9]);
    }
}
