//! The in-house attribute-preserving LOD generator `[MOD-05]` `[AC-WP28-1]`: LOD1 to LOD4 from
//! the hero mesh (LOD0) by constrained quadric edge collapse, with the per-level transfer of
//! regions, skin weights, deltas and landmarks, an error report per level, and the switching
//! metric.
//!
//! # Method
//!
//! The mesh is triangulated ([`Mesh::triangulate`]) and simplified on its welded topology by
//! **half-edge collapses** `u -> v` ranked by **Garland–Heckbert plane quadrics** (area-weighted
//! planes of the triangles merged into a vertex, plus planes through every feature edge,
//! perpendicular to its triangles, weighted by `feature_weight * length²`), **Hoppe's attribute
//! quadric** for the UV coordinates (scaled to metres by `sqrt(surface area / UV area)` and
//! `uv_weight`), and two attribute terms, the estimated **delta** and **skin-weight**
//! interpolation errors (below; skin scaled to metres by `skin_length`), each squared and
//! multiplied by the merged area. A collapse keeps `v` where it is, so every LOD vertex is an LOD0
//! vertex: its barycentric source is itself (`[v, v, v]`, weights `[1, 0, 0]`), positions and
//! every per-vertex attribute transfer exactly, and [`LodLevel::transfer_skin`] (renormalised)
//! and [`LodLevel::transfer_deltas`] (sparse, same components) give valid data. This is the
//! vertex placement of meshoptimizer's simplifier; optimal placement would lower the geometric
//! error a little but would need projected sources and resampled attributes.
//!
//! # Constraints
//!
//! An edge is a *feature edge* when it is a mesh border, a UV seam or hard edge (its two triangles
//! use different wedges at one end), a part boundary, a region front (both ends in one region, a
//! triangle on it touches another region) or an edge of an active protected loop. A vertex with no
//! feature edge may collapse into any neighbour; a vertex with exactly two feature edges of the
//! same kind (equal signature) may only slide along them; any other vertex (junctions, ends,
//! corners where kinds change) is locked. Further:
//!
//! - both ends lie in the same region (so region labels transfer exactly and fronts stay put);
//! - the link condition holds (manifoldness), no triangle is created twice, no triangle flips or
//!   tilts beyond `min_normal_dot`, no triangle degenerates, no UV triangle flips or drops below a
//!   minimum area, and no changed triangle overlaps another of its tile in UV (`uvgrid`); the
//!   corners of a LOD0 polygon that holds a UV-sliver triangle stay, and a LOD0 polygon whose
//!   triangles all survive unchanged is output whole, so every level is a valid UV layout;
//! - wedges map consistently across the edge (seams stay seams, tiles stay tiles, no UV crack);
//! - a part keeps at least 4 (closed) or 2 (open) triangles, a loop at least 3 vertices;
//! - **protected loops** (eyelids, lips) keep every vertex up to level `lock_through` and stay
//!   feature lines up to `keep_through`.
//!
//! # Errors: estimated, measured, stated
//!
//! Each level has three budgets: geometric (metres), delta (metres, deltas at unit weight) and
//! skin (L1 weight distance). While collapsing, every live vertex carries *estimates* of the
//! errors of the vertices merged into it: a collapse `u -> v` adds to `u`'s estimates the errors
//! at `u` itself against the new triangles (distance to them, and to the new segment for a
//! feature-line vertex; the largest per-component difference between `u`'s delta and the delta
//! interpolated there; the L1 difference of the skin weights), the result is the maximum of that
//! and `v`'s estimates, and collapses whose estimates exceed a budget are refused.
//!
//! The estimates miss what a collapse does to the vertices merged into `u`'s *neighbours*, so they
//! do not bound the *measured* errors ([`LodErrorReport`]). Those project every welded LOD0 vertex
//! onto the level (see `report`; candidates are the triangles around the vertex it was merged into
//! and its neighbours that touch its own region): the distance, the largest over components of the
//! difference between its delta and the interpolated delta, the L1 difference of its skin weights,
//! and for feature-line vertices the distance to the level's feature lines. Before the run applies
//! a collapse it makes it on the topology and measures, the same way, every LOD0 vertex whose
//! candidates or feature lines the collapse changes (those merged into `v`, its neighbours and the
//! neighbours of `u`'s former neighbours); if one exceeds a budget the collapse is undone and
//! refused for the rest of the level (`refused_collapses`). As a safety net (vertices no triangle
//! used are measured against every triangle, which a local check cannot see) the measured maxima
//! are checked after the run; when one is exceeded, the longest prefix of the level's collapses
//! that measures within all budgets is found by bisection (to 1/64 of the collapse count) and
//! replayed (`undone_collapses`). So the budgets are the *stated* bounds and every measured
//! maximum is within them (at the LOD0 vertices; between vertices the surfaces are linear
//! interpolations of them). The geometric maximum is the level's [`LodLevel::declared_error`],
//! and it bounds what [`crate::validate`] measures.
//!
//! # Robustness
//!
//! Vertices whose neighbourhood is not one consistently wound disc or half-disc (non-manifold
//! edges, bow-ties, inconsistent winding) and vertices whose duplicates lie in different regions
//! are locked (they neither move nor receive collapses); zero-area triangles contribute no quadric
//! and are never created; duplicate positions that are not welded behave as borders. Every pop of
//! the collapse heap either removes a vertex or retires an entry (see `Engine::run`), so the loop
//! ends; the bisection halves its range each step. Iteration orders and tie-breaks are by index,
//! so the output is deterministic.

mod collapse;
mod quadric;
mod report;
mod switch;
mod uvgrid;

use crate::delta::DeltaBases;
use crate::error::{ModelError, Result};
use crate::landmark::Landmark;
use crate::lod::{LodLevel, MAX_LOD_LEVELS};
use crate::mesh::Mesh;
use crate::region::RegionMap;
use crate::skin::SkinWeights;
use crate::template::Template;

pub use report::LodErrorReport;
pub use switch::{choose_lod, projected_size};

use collapse::{DeltaRows, Engine, Rules};

/// The most protected loops.
pub const MAX_LOOPS: usize = 1024;

/// A closed vertex loop to preserve (eyelid margins, lip lines).
#[derive(Clone, Debug, PartialEq)]
pub struct ProtectedLoop {
    /// What the loop is (for reports).
    pub name: String,
    /// The loop's LOD0 vertices in order; consecutive ones (and last to first) share an edge.
    pub vertices: Vec<u32>,
    /// Levels up to this one (1-based) keep every loop vertex.
    pub lock_through: u8,
    /// Levels up to this one keep the loop as a feature line (simplified only along itself).
    pub keep_through: u8,
}

/// The settings of one generated level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelSettings {
    /// Target triangle count as a fraction of LOD0's, in `(0, 1]`.
    pub triangle_ratio: f32,
    /// Geometric budget, metres: the largest distance of a LOD0 vertex from the level's surface
    /// (and of a feature-line vertex from the level's feature lines).
    pub max_geometric_error: f32,
    /// Delta budget, metres at unit weight: the largest difference between a LOD0 vertex's delta
    /// and the level's interpolated delta, over all components.
    pub max_delta_error: f32,
    /// Skin budget: the largest L1 distance between a LOD0 vertex's skin weights and the level's
    /// interpolated weights, in `[0, 2]`.
    pub max_skin_error: f32,
    /// Screen-height fraction below which the level is used ([`LodLevel::switch_size`]).
    pub switch_size: f32,
    /// Relative dead band around `switch_size` ([`LodLevel::hysteresis`]).
    pub hysteresis: f32,
}

/// The generator settings.
#[derive(Clone, Debug, PartialEq)]
pub struct LodSettings {
    /// LOD1 onwards, at most [`MAX_LOD_LEVELS`]; triangle ratios non-increasing, budgets
    /// non-decreasing, switch sizes decreasing.
    pub levels: Vec<LevelSettings>,
    /// Multiplier on the UV deviation (UVs are first scaled to metres).
    pub uv_weight: f32,
    /// Weight of the feature-edge planes relative to the surface planes.
    pub feature_weight: f32,
    /// Weight of the delta term of the cost.
    pub delta_weight: f32,
    /// Metres per unit of L1 skin-weight difference in the cost.
    pub skin_length: f32,
    /// Smallest cosine between a triangle's normal before and after a collapse, in `[-1, 1]`.
    pub min_normal_dot: f32,
}

impl Default for LodSettings {
    /// Halving triangle counts from LOD1 (50 %) to LOD4 (5 %), with budgets for a 1.7 m human.
    fn default() -> Self {
        let level = |triangle_ratio, geo, delta, skin, switch_size| LevelSettings {
            triangle_ratio,
            max_geometric_error: geo,
            max_delta_error: delta,
            max_skin_error: skin,
            switch_size,
            hysteresis: 0.1,
        };
        Self {
            levels: vec![
                level(0.5, 0.0005, 0.0005, 0.1, 0.5),
                level(0.25, 0.0015, 0.0015, 0.2, 0.25),
                level(0.12, 0.004, 0.004, 0.35, 0.12),
                level(0.05, 0.01, 0.01, 0.5, 0.06),
            ],
            uv_weight: 1.0,
            feature_weight: 4.0,
            delta_weight: 1.0,
            skin_length: 0.01,
            min_normal_dot: 0.2,
        }
    }
}

impl LodSettings {
    /// Checks the settings.
    pub fn check(&self) -> Result<()> {
        if self.levels.is_empty() || self.levels.len() > MAX_LOD_LEVELS {
            return Err(ModelError::limit("LOD settings level count", MAX_LOD_LEVELS as u64, self.levels.len() as u64));
        }
        let finite_non_negative = |x: f32| x.is_finite() && x >= 0.0;
        if !(finite_non_negative(self.uv_weight)
            && finite_non_negative(self.feature_weight)
            && finite_non_negative(self.delta_weight)
            && finite_non_negative(self.skin_length)
            && (-1.0..=1.0).contains(&self.min_normal_dot))
        {
            return Err(ModelError::invalid("LOD settings", 0, "a weight is negative or not finite"));
        }
        let mut prev: Option<&LevelSettings> = None;
        for (i, l) in self.levels.iter().enumerate() {
            let bad_budget = |x: f32| x.is_nan() || x < 0.0;
            if l.triangle_ratio.is_nan()
                || l.triangle_ratio <= 0.0
                || l.triangle_ratio > 1.0
                || bad_budget(l.max_geometric_error)
                || bad_budget(l.max_delta_error)
                || bad_budget(l.max_skin_error)
            {
                return Err(ModelError::invalid(
                    "LOD settings level",
                    i + 1,
                    "ratio outside (0, 1] or a negative budget",
                ));
            }
            if l.switch_size.is_nan()
                || l.switch_size <= 0.0
                || l.switch_size > 1.0
                || !(0.0..=0.5).contains(&l.hysteresis)
            {
                return Err(ModelError::invalid(
                    "LOD settings level",
                    i + 1,
                    "switch size outside (0, 1] or hysteresis outside [0, 0.5]",
                ));
            }
            if let Some(p) = prev
                && (l.triangle_ratio > p.triangle_ratio
                    || l.max_geometric_error < p.max_geometric_error
                    || l.max_delta_error < p.max_delta_error
                    || l.max_skin_error < p.max_skin_error
                    || l.switch_size >= p.switch_size)
            {
                return Err(ModelError::invalid(
                    "LOD settings level",
                    i + 1,
                    "ratio grows, a budget shrinks or the switch size does not decrease",
                ));
            }
            prev = Some(l);
        }
        Ok(())
    }
}

/// What the generator reads: LOD0 and its attributes. Indices refer to LOD0 vertices.
#[derive(Clone, Copy, Debug)]
pub struct LodInput<'a> {
    /// LOD0.
    pub mesh: &'a Mesh,
    /// Per-vertex regions (fronts are preserved, labels never mix).
    pub regions: Option<&'a RegionMap>,
    /// Skin weights (the skin term and bound; the measured skin error).
    pub skin: Option<&'a SkinWeights>,
    /// Delta bases (the delta term and bound; the measured delta error).
    pub deltas: Option<&'a DeltaBases>,
    /// Landmarks to re-express on each level.
    pub landmarks: &'a [Landmark],
    /// Loops to preserve.
    pub loops: &'a [ProtectedLoop],
    /// Vertices that never move.
    pub locked: &'a [u32],
}

impl<'a> LodInput<'a> {
    /// The input of a template's LOD0 with its regions, skin, deltas and landmarks.
    pub fn from_template(t: &'a Template, loops: &'a [ProtectedLoop]) -> Self {
        Self {
            mesh: t.mesh(),
            regions: Some(t.regions()),
            skin: Some(t.skin()),
            deltas: Some(t.deltas()),
            landmarks: t.landmarks(),
            loops,
            locked: &[],
        }
    }
}

/// One generated level.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedLod {
    /// The level, ready for [`crate::template::TemplateParts::lods`].
    pub level: LodLevel,
    /// The input landmarks re-expressed as barycentrics on a triangle of this level (vertex indices
    /// of this level's mesh), in input order.
    pub landmarks: Vec<Landmark>,
    /// The measured errors.
    pub report: LodErrorReport,
}

/// Generates the levels of `settings` from `input` (each level continues from the previous one).
pub fn generate_lods(input: &LodInput<'_>, settings: &LodSettings) -> Result<Vec<GeneratedLod>> {
    settings.check()?;
    let rows = DeltaRows::new(input.deltas, input.mesh.position_keys())?;
    let mut engine = Engine::new(input, settings, &rows)?;
    let base = report::Baseline::new(&engine, input)?;
    let base_triangles = engine.tris.len();
    let positions = input.mesh.positions();
    let mut out = Vec::with_capacity(settings.levels.len());
    for (i, l) in settings.levels.iter().enumerate() {
        let rules =
            Rules { geometric: f64::from(l.max_geometric_error), delta: l.max_delta_error, skin: l.max_skin_error };
        engine.begin_level(i as u8 + 1, rules);
        let target = (f64::from(l.triangle_ratio) * base_triangles as f64).ceil() as usize;
        let start = engine.clone();
        engine.run(target);
        let refused = engine.refused();
        // The run verified every collapse against the measurement around it, so the level is
        // within its budgets unless a LOD0 vertex no triangle used (measured against every
        // triangle) moved out of them. As a safety net the measured errors decide: the level
        // starts within its budgets (the previous level met tighter ones), so the longest
        // measured-valid prefix of the level's collapses is searched for (to 1/64 of their count).
        let mut undone = 0;
        let within = |e: &Engine<'_>| report::measure(e, &base, positions, &report::live_triangles(e)).within(l);
        if !within(&engine) {
            let log = std::mem::take(&mut engine.log);
            let (mut lo, mut hi) = (0, log.len());
            let step = (log.len() / 64).max(1);
            while hi - lo > step {
                let mid = lo + (hi - lo) / 2;
                let mut probe = start.clone();
                probe.replay(&log[..mid]);
                if within(&probe) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            engine = start;
            engine.replay(&log[..lo]);
            undone = log.len() - lo;
        }
        let mut generated = report::output(&engine, &base, input, i + 1, l)?;
        generated.report.undone_collapses = undone;
        generated.report.refused_collapses = refused;
        out.push(generated);
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
