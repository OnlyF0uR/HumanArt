//! Level output and measurement `[MOD-05]` `[AC-WP28-1]`: the reduced mesh with its barycentric
//! sources, the landmarks re-expressed on it, and the error report.
//!
//! Every welded LOD0 vertex is projected onto the level's surface. A vertex merged during the
//! collapses is searched for among the triangles around the vertex it was merged into (its
//! representative) and that vertex's neighbours (the two-ring); a vertex no triangle used is
//! measured against every triangle. The nearest point found this way is never nearer than the true
//! nearest point, so the reported maximum distance bounds the one-sided distance from LOD0 to the
//! level that [`crate::validate`] measures, and it is the level's declared error. The local search
//! also keeps the projection on the right sheet where surfaces nearly touch (lips, eyelids), which
//! is what the attribute errors need; for the same reason only triangles with a corner in the
//! vertex's own region are candidates (a finger's vertex is not measured against the next finger).
//!
//! The same per-vertex measurement ([`vertex_error`]) is what the engine checks before it applies
//! a collapse, for every LOD0 vertex whose candidates the collapse changes, so a level's measured
//! maxima stay within its budgets by construction.

use std::fmt;

use humanart_math::Vec3;
use humanart_math::geom::closest_point_on_triangle;

use super::collapse::{Engine, LoopMode, NO_REGION, NONE, Rules};
use super::{GeneratedLod, LevelSettings, LodInput};
use crate::error::{ModelError, Result};
use crate::landmark::Landmark;
use crate::lod::LodLevel;
use crate::mesh::{Mesh, MeshData, MeshPart};

/// The measured errors of one generated level.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LodErrorReport {
    /// 1 for LOD1, ...
    pub level: usize,
    /// Vertices of the level (wedges).
    pub vertices: usize,
    /// Triangles of the level.
    pub triangles: usize,
    /// Triangles relative to LOD0's triangulation.
    pub triangle_ratio: f32,
    /// The requested ratio (the level stops above it when no valid collapse within the budgets is
    /// left).
    pub target_ratio: f32,
    /// Largest distance from a welded LOD0 vertex to the level's surface, metres; the level's
    /// declared error.
    pub geometric_error: f32,
    /// RMS of the same distances, metres.
    pub geometric_rms: f32,
    /// Welded LOD0 vertices measured.
    pub samples: usize,
    /// The generator's accumulated estimate of the geometric error, metres.
    pub geometric_estimate: f32,
    /// The level's geometric budget, metres: the stated bound, which `geometric_error` never
    /// exceeds.
    pub geometric_budget: f32,
    /// Measured delta error: the largest difference, over LOD0 vertices and components, between a
    /// vertex's delta and the delta interpolated at its projection, metres at unit weight.
    pub delta_error: f32,
    /// RMS of the same differences over the (vertex, component) pairs where either is non-zero.
    pub delta_rms: f32,
    /// The component with the largest measured delta error.
    pub delta_worst_component: Option<u32>,
    /// The generator's accumulated estimate of the delta error (largest over the level's vertices).
    pub delta_estimate: f32,
    /// The level's delta budget, metres: the stated bound, which `delta_error` never exceeds.
    pub delta_budget: f32,
    /// Largest L1 distance between a LOD0 vertex's skin weights and the weights interpolated at its
    /// projection.
    pub skin_error: f32,
    /// Mean of the same distances.
    pub skin_mean: f32,
    /// The generator's accumulated estimate of the skin error.
    pub skin_estimate: f32,
    /// The level's skin budget: the stated bound, which `skin_error` never exceeds.
    pub skin_budget: f32,
    /// Largest distance from a LOD0 vertex on a seam, border, part boundary or region front to the
    /// level's feature edges near it, metres (within the geometric budget).
    pub feature_error: f32,
    /// LOD0 vertices on feature lines.
    pub feature_vertices: usize,
    /// Of those, the ones the level keeps.
    pub kept_feature_vertices: usize,
    /// Largest displacement of a re-expressed landmark, metres.
    pub landmark_error: f32,
    /// Collapses refused at this level because the measurement of a LOD0 vertex near them would
    /// have exceeded a budget (the estimates had let them through).
    pub refused_collapses: usize,
    /// Collapses of this level undone because the measured errors exceeded a budget after the run
    /// (a safety net: refusals keep the measured vertices within the budgets).
    pub undone_collapses: usize,
}

impl fmt::Display for LodErrorReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LOD{}: {} vertices, {} triangles ({:.3} of LOD0, target {:.3}); geometric max {:.3e} m rms {:.3e} m over {} vertices (estimate {:.3e}, bound {:.1e}); delta max {:.3e} m rms {:.3e} m (worst component {}; estimate {:.3e}, bound {:.1e}); skin L1 max {:.3} mean {:.4} (estimate {:.3}, bound {:.2}); features {} of {} kept, max deviation {:.3e} m; landmarks max {:.3e} m; {} collapses refused, {} undone",
            self.level,
            self.vertices,
            self.triangles,
            self.triangle_ratio,
            self.target_ratio,
            self.geometric_error,
            self.geometric_rms,
            self.samples,
            self.geometric_estimate,
            self.geometric_budget,
            self.delta_error,
            self.delta_rms,
            self.delta_worst_component.map_or_else(|| "-".to_owned(), |c| c.to_string()),
            self.delta_estimate,
            self.delta_budget,
            self.skin_error,
            self.skin_mean,
            self.skin_estimate,
            self.skin_budget,
            self.kept_feature_vertices,
            self.feature_vertices,
            self.feature_error,
            self.landmark_error,
            self.refused_collapses,
            self.undone_collapses,
        )
    }
}

/// What is measured on every level, fixed at LOD0.
pub(crate) struct Baseline {
    /// Welded LOD0 vertices (their own weld key).
    measured: Vec<u32>,
    /// Welded vertices on permanent feature lines.
    feature: Vec<u32>,
}

impl Baseline {
    pub(crate) fn new(engine: &Engine<'_>, input: &LodInput<'_>) -> Result<Self> {
        let n = engine.weld.len();
        for (i, l) in input.landmarks.iter().enumerate() {
            if l.vertices.iter().any(|&v| v as usize >= n) || !l.weights.iter().all(|w| w.is_finite()) {
                return Err(ModelError::invalid("landmark", i, "vertex out of range or weight not finite"));
            }
        }
        let measured: Vec<u32> = (0..n as u32).filter(|&v| engine.weld[v as usize] == v).collect();
        let feature = measured.iter().copied().filter(|&v| engine.feature[v as usize]).collect();
        Ok(Self { measured, feature })
    }
}

/// Per welded vertex of the engine's LOD0 state: whether it lies on a permanent feature line.
pub(crate) fn feature_flags(e: &Engine<'_>) -> Vec<bool> {
    let (mut pairs, mut nbrs) = (Vec::new(), Vec::new());
    (0..e.weld.len() as u32)
        .map(|v| {
            e.weld[v as usize] == v && e.used[v as usize] && {
                feature_neighbours(e, v, &mut pairs, &mut nbrs);
                !nbrs.is_empty()
            }
        })
        .collect()
}

/// The neighbours `x` of live vertex `v` with a permanent feature edge `(v, x)`.
fn feature_neighbours(e: &Engine<'_>, v: u32, pairs: &mut Vec<(u32, u32)>, out: &mut Vec<u32>) {
    pairs.clear();
    for &t in &e.fans[v as usize] {
        if e.tri_alive[t as usize] {
            for &w in &e.tris[t as usize] {
                let x = e.weld[w as usize];
                if x != v {
                    pairs.push((x, t));
                }
            }
        }
    }
    pairs.sort_unstable();
    out.clear();
    for g in pairs.chunk_by(|a, b| a.0 == b.0) {
        if g.len() <= 2 && e.edge_sig(v, g[0].0, g, LoopMode::Ignore).is_feature() {
            out.push(g[0].0);
        }
    }
}

/// The live triangles around live vertex `r` and its neighbours, ascending (appended to `out`).
pub(crate) fn ring(e: &Engine<'_>, r: u32, out: &mut Vec<u32>) {
    // Each neighbour's fan once (up to the first 32 neighbours; beyond, repeats are deduplicated).
    let mut seen = [NONE; 32];
    let mut n = 0;
    let live = |t: &&u32| e.tri_alive[**t as usize];
    out.extend(e.fans[r as usize].iter().filter(live));
    for &t in &e.fans[r as usize] {
        if !e.tri_alive[t as usize] {
            continue;
        }
        for &w in &e.tris[t as usize] {
            let x = e.weld[w as usize];
            if x == r || seen[..n].contains(&x) {
                continue;
            }
            if n < seen.len() {
                seen[n] = x;
                n += 1;
            }
            out.extend(e.fans[x as usize].iter().filter(live));
        }
    }
    out.sort_unstable();
    out.dedup();
}

/// The nearest point to `p` on the candidate triangles with a corner in `region` (any triangle
/// when it is [`NO_REGION`]): (distance, triangle, barycentrics).
fn nearest(
    e: &Engine<'_>,
    positions: &[Vec3],
    p: Vec3,
    candidates: &[u32],
    region: u16,
) -> Option<(f32, u32, [f32; 3])> {
    let mut best: Option<(f32, u32, [f32; 3])> = None;
    for &t in candidates {
        let corners = e.tris[t as usize];
        if region != NO_REGION && corners.iter().all(|&w| e.region[e.weld[w as usize] as usize] != region) {
            continue;
        }
        let c = corners.map(|w| positions[w as usize]);
        let (q, b) = closest_point_on_triangle(p, c[0], c[1], c[2]);
        let d = p.distance(q);
        if best.is_none_or(|(bd, _, _)| d < bd) {
            best = Some((d, t, [b.x, b.y, b.z]));
        }
    }
    best
}

/// Builds the level's mesh: live triangles in order (so parts stay contiguous), wedges renumbered
/// in ascending LOD0 order; a LOD0 face whose triangles all survive unchanged is kept as the
/// original polygon (its triangles alone may have slivers of UV area that read as flipped).
/// Returns the mesh and the LOD0 wedge of each new vertex.
fn snapshot(e: &Engine<'_>, mesh: &Mesh, new_of: &mut [u32]) -> Result<(Mesh, Vec<u32>)> {
    let faces: Vec<usize> = (0..e.tris.len()).filter(|&t| e.tri_alive[t]).collect();
    let mut wedges: Vec<u32> = faces.iter().flat_map(|&t| e.tris[t]).collect();
    wedges.sort_unstable();
    wedges.dedup();
    new_of.fill(NONE);
    for (i, &w) in wedges.iter().enumerate() {
        new_of[w as usize] = i as u32;
    }
    let n = e.weld.len();
    let (mut first_weld, mut first_normal) = (vec![NONE; n], vec![NONE; n]);
    let mut data = MeshData { face_offsets: Vec::with_capacity(faces.len() + 1), ..MeshData::default() };
    for (i, &w) in wedges.iter().enumerate() {
        let wi = w as usize;
        data.positions.push(mesh.positions()[wi]);
        data.uvs.push(mesh.uvs()[wi]);
        let k = e.weld[wi] as usize;
        if first_weld[k] == NONE {
            first_weld[k] = i as u32;
        }
        data.position_keys.push(first_weld[k]);
        let nk = mesh.normal_keys()[wi] as usize;
        if first_normal[nk] == NONE {
            first_normal[nk] = i as u32;
        }
        data.normal_keys.push(first_normal[nk]);
    }
    // LOD0 faces kept whole: every triangle of the face alive and unchanged.
    let mut intact = vec![true; mesh.face_count()];
    for (t, &f) in e.tri_face.iter().enumerate() {
        if !e.tri_alive[t] || e.tri_changed[t] {
            intact[f as usize] = false;
        }
    }
    let mut part_faces = vec![0_u32; mesh.parts().len()];
    data.face_offsets.push(0);
    for (i, &t) in faces.iter().enumerate() {
        let f = e.tri_face[t] as usize;
        if intact[f] {
            // Once per face, at its first triangle.
            if i > 0 && e.tri_face[faces[i - 1]] as usize == f {
                continue;
            }
            data.face_vertices.extend(mesh.face(f).iter().map(|&w| new_of[w as usize]));
        } else {
            data.face_vertices.extend(e.tris[t].map(|w| new_of[w as usize]));
        }
        data.face_offsets.push(data.face_vertices.len() as u32);
        data.face_tiles.push(e.tri_tile[t]);
        part_faces[e.tri_part[t] as usize] += 1;
    }
    let mut first_face = 0_u32;
    for (p, part) in mesh.parts().iter().enumerate() {
        let face_count = part_faces[p];
        data.parts.push(MeshPart { name: part.name.clone(), first_face, face_count, closed: part.closed });
        first_face += face_count;
    }
    Ok((Mesh::new(data)?, wedges))
}

/// Accumulators of the attribute errors.
#[derive(Clone, Debug, Default)]
pub(crate) struct Attr {
    acc: Vec<[f32; 3]>,
    touched: Vec<u32>,
    comp_max: Vec<f32>,
    pub(crate) delta_max: f32,
    delta_sq: f64,
    delta_n: u64,
    skin: Vec<(u16, f32)>,
    pub(crate) skin_max: f32,
    skin_sum: f64,
    skin_n: u64,
}

impl Attr {
    /// Adds the delta and skin differences of LOD0 vertex `a` against the level triangle `t` at
    /// barycentrics `b`; returns `a`'s largest delta difference and its skin L1 difference.
    fn add(&mut self, e: &Engine<'_>, a: u32, t: u32, b: [f32; 3]) -> (f32, f32) {
        let corners = e.tris[t as usize].map(|w| e.weld[w as usize]);
        let add_row = |v: u32, s: f32, this: &mut Self| {
            let (comps, values) = e.deltas.row(v);
            for (&c, d) in comps.iter().zip(values) {
                let slot = &mut this.acc[c as usize];
                if *slot == [0.0; 3] {
                    this.touched.push(c);
                }
                for k in 0..3 {
                    slot[k] += s * d[k];
                }
            }
        };
        add_row(a, 1.0, self);
        for k in 0..3 {
            add_row(corners[k], -b[k], self);
        }
        self.touched.sort_unstable();
        self.touched.dedup();
        let mut delta = 0.0_f32;
        for i in 0..self.touched.len() {
            let c = self.touched[i] as usize;
            let d = self.acc[c];
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            delta = delta.max(len);
            self.comp_max[c] = self.comp_max[c].max(len);
            self.delta_max = self.delta_max.max(len);
            self.delta_sq += f64::from(len) * f64::from(len);
            self.delta_n += 1;
            self.acc[c] = [0.0; 3];
        }
        self.touched.clear();

        let Some(skin) = e.skin else { return (delta, 0.0) };
        self.skin.clear();
        let (j, w) = skin.vertex(a as usize);
        self.skin.extend(j.iter().zip(w).map(|(&j, &w)| (j, w)));
        for k in 0..3 {
            let (j, w) = skin.vertex(corners[k] as usize);
            self.skin.extend(j.iter().zip(w).map(|(&j, &w)| (j, -w * b[k])));
        }
        self.skin.sort_unstable_by_key(|x| x.0);
        let mut l1 = 0.0_f32;
        for g in self.skin.chunk_by(|x, y| x.0 == y.0) {
            l1 += g.iter().map(|x| x.1).sum::<f32>().abs();
        }
        self.skin_max = self.skin_max.max(l1);
        self.skin_sum += f64::from(l1);
        self.skin_n += 1;
        (delta, l1)
    }

    /// Empty accumulators for `components` delta components.
    pub(crate) fn new(components: usize) -> Self {
        Self { acc: vec![[0.0; 3]; components], comp_max: vec![0.0; components], ..Self::default() }
    }
}

/// Reusable buffers of the per-vertex measurement, with the feature segments of the last
/// representative asked for (valid until [`Probe::reset`]).
#[derive(Clone, Debug)]
pub(crate) struct Probe {
    near: Vec<u32>,
    far: Vec<u32>,
    pairs: Vec<(u32, u32)>,
    segs: Vec<(u32, u32)>,
    segs_of: u32,
}

impl Default for Probe {
    fn default() -> Self {
        Self { near: Vec::new(), far: Vec::new(), pairs: Vec::new(), segs: Vec::new(), segs_of: NONE }
    }
}

impl Probe {
    /// Forgets the cached feature segments (the topology changed).
    pub(crate) fn reset(&mut self) {
        self.segs_of = NONE;
    }
}

/// The measured errors of one welded LOD0 vertex.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VertexError {
    /// Distance to the level's surface, metres.
    pub geo: f32,
    /// Largest delta difference, metres.
    pub delta: f32,
    /// Skin-weight L1 difference.
    pub skin: f32,
    /// Distance to the level's feature lines (0 off them), metres.
    pub feature: f32,
}

impl VertexError {
    /// Whether every error is within the rules' limits.
    pub(crate) fn within(&self, r: &Rules) -> bool {
        f64::from(self.geo) <= r.geometric
            && f64::from(self.feature) <= r.geometric
            && self.delta <= r.delta
            && self.skin <= r.skin
    }
}

/// The errors of used welded LOD0 vertex `a` whose representative is `r`, measured on `cands` (the
/// [`ring`] of `r`); attribute differences are also added to `attr`. `None` when no candidate
/// touches `a`'s region.
pub(crate) fn vertex_error(
    e: &Engine<'_>,
    positions: &[Vec3],
    a: u32,
    r: u32,
    cands: &[u32],
    probe: &mut Probe,
    attr: &mut Attr,
) -> Option<(VertexError, u32, [f32; 3])> {
    let p = positions[a as usize];
    let (d, t, b) = nearest(e, positions, p, cands, e.region[a as usize])?;
    let (delta, skin) = attr.add(e, a, t, b);
    let feature = if e.feature[a as usize] { feature_distance(e, positions, p, r, probe) } else { 0.0 };
    Some((VertexError { geo: d, delta, skin, feature }, t, b))
}

/// The distance from feature-line point `p` (merged into `rep`) to the feature edges within two
/// steps of `rep` along the lines.
pub(crate) fn feature_distance(e: &Engine<'_>, positions: &[Vec3], p: Vec3, rep: u32, probe: &mut Probe) -> f32 {
    if probe.segs_of != rep {
        probe.segs.clear();
        feature_neighbours(e, rep, &mut probe.pairs, &mut probe.near);
        if probe.near.is_empty() {
            // No feature edge left: the distance to the representative itself.
            probe.segs.push((rep, rep));
        }
        for i in 0..probe.near.len() {
            let x = probe.near[i];
            probe.segs.push((rep, x));
            feature_neighbours(e, x, &mut probe.pairs, &mut probe.far);
            probe.segs.extend(probe.far.iter().map(|&y| (x, y)));
        }
        probe.segs_of = rep;
    }
    probe.segs.iter().fold(f32::INFINITY, |best, &(x, y)| {
        best.min(segment_distance(p, positions[x as usize], positions[y as usize]))
    })
}

/// Distance from `p` to segment `ab`.
fn segment_distance(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    let t = if len2 > 0.0 { ((p - a).dot(ab) / len2).clamp(0.0, 1.0) } else { 0.0 };
    p.distance(a + ab * t)
}

/// The measured errors of the engine's current state.
pub(crate) struct Measured {
    pub(crate) geo_max: f32,
    geo_sq: f64,
    samples: usize,
    pub(crate) feature_max: f32,
    pub(crate) attr: Attr,
}

impl Measured {
    /// Whether the geometric, delta and skin maxima are within the level's budgets.
    pub(crate) fn within(&self, l: &LevelSettings) -> bool {
        self.geo_max <= l.max_geometric_error
            && self.feature_max <= l.max_geometric_error
            && self.attr.delta_max <= l.max_delta_error
            && self.attr.skin_max <= l.max_skin_error
    }
}

/// Projects every welded LOD0 vertex onto the current surface and measures the errors there.
/// `live` lists the live triangles (the fallback candidates of vertices no triangle used).
pub(crate) fn measure(e: &Engine<'_>, base: &Baseline, positions: &[Vec3], live: &[u32]) -> Measured {
    let mut m =
        Measured { geo_max: 0.0, geo_sq: 0.0, samples: 0, feature_max: 0.0, attr: Attr::new(e.deltas.components) };
    let mut cands: Vec<u32> = Vec::new();
    let mut probe = Probe::default();
    probe.reset();
    for &a in &base.measured {
        let p = positions[a as usize];
        let d = if e.used[a as usize] {
            let r = e.rep(a);
            cands.clear();
            ring(e, r, &mut cands);
            let Some((err, _, _)) = vertex_error(e, positions, a, r, &cands, &mut probe, &mut m.attr) else {
                continue;
            };
            m.feature_max = m.feature_max.max(err.feature);
            err.geo
        } else {
            let Some((d, _, _)) = nearest(e, positions, p, live, NO_REGION) else { continue };
            d
        };
        m.geo_max = m.geo_max.max(d);
        m.geo_sq += f64::from(d) * f64::from(d);
        m.samples += 1;
    }
    m
}

/// The live triangles of the engine.
pub(crate) fn live_triangles(e: &Engine<'_>) -> Vec<u32> {
    (0..e.tris.len() as u32).filter(|&t| e.tri_alive[t as usize]).collect()
}

/// Snapshots the engine as level `level` and measures it.
pub(crate) fn output(
    e: &Engine<'_>,
    base: &Baseline,
    input: &LodInput<'_>,
    level: usize,
    settings: &LevelSettings,
) -> Result<GeneratedLod> {
    let mesh = input.mesh;
    let positions = mesh.positions();
    let mut new_of = vec![NONE; e.weld.len()];
    let (lod_mesh, wedges) = snapshot(e, mesh, &mut new_of)?;
    let live = live_triangles(e);

    let mut r = LodErrorReport {
        level,
        vertices: lod_mesh.vertex_count(),
        triangles: live.len(),
        triangle_ratio: live.len() as f32 / e.tris.len() as f32,
        target_ratio: settings.triangle_ratio,
        geometric_budget: settings.max_geometric_error,
        delta_budget: settings.max_delta_error,
        skin_budget: settings.max_skin_error,
        feature_vertices: base.feature.len(),
        ..LodErrorReport::default()
    };
    let m = measure(e, base, positions, &live);
    let attr = &m.attr;
    r.geometric_error = m.geo_max;
    r.samples = m.samples;
    r.geometric_rms = (m.geo_sq / r.samples.max(1) as f64).sqrt() as f32;
    let mut cands: Vec<u32> = Vec::new();
    r.delta_error = attr.delta_max;
    r.delta_rms = (attr.delta_sq / attr.delta_n.max(1) as f64).sqrt() as f32;
    r.delta_worst_component =
        attr.comp_max.iter().enumerate().filter(|x| *x.1 > 0.0).max_by(|x, y| x.1.total_cmp(y.1)).map(|x| x.0 as u32);
    r.skin_error = attr.skin_max;
    r.skin_mean = (attr.skin_sum / attr.skin_n.max(1) as f64) as f32;
    for v in 0..e.weld.len() {
        if e.alive[v] {
            r.geometric_estimate = r.geometric_estimate.max(e.geo_err[v]);
            r.delta_estimate = r.delta_estimate.max(e.delta_err[v]);
            r.skin_estimate = r.skin_estimate.max(e.skin_err[v]);
        }
    }

    r.feature_error = m.feature_max;
    r.kept_feature_vertices = base.feature.iter().filter(|&&a| e.alive[a as usize]).count();

    // Landmarks: the nearest point on the triangles around their corners' representatives.
    let mut landmarks = Vec::with_capacity(input.landmarks.len());
    for (i, l) in input.landmarks.iter().enumerate() {
        let p = l.position(positions).ok_or(ModelError::invalid("landmark", i, "vertex out of range"))?;
        cands.clear();
        for &v in &l.vertices {
            let k = e.weld[v as usize];
            if e.used[k as usize] {
                ring(e, e.rep(k), &mut cands);
            }
        }
        let found = nearest(e, positions, p, if cands.is_empty() { &live } else { &cands }, NO_REGION);
        let Some((d, t, b)) = found else {
            return Err(ModelError::invalid("landmark", i, "no triangle to re-express it on"));
        };
        r.landmark_error = r.landmark_error.max(d);
        let b = b.map(|x| x.clamp(0.0, 1.0));
        let sum = b[0] + b[1] + b[2];
        let weights = if sum > 0.0 { b.map(|x| x / sum) } else { [1.0, 0.0, 0.0] };
        landmarks.push(Landmark {
            name: l.name.clone(),
            group: l.group,
            vertices: e.tris[t as usize].map(|w| new_of[w as usize]),
            weights,
        });
    }

    let level = LodLevel {
        sources: wedges.iter().map(|&w| [w, w, w]).collect(),
        source_weights: vec![[1.0, 0.0, 0.0]; wedges.len()],
        mesh: lod_mesh,
        declared_error: r.geometric_error,
        switch_size: settings.switch_size,
        hysteresis: settings.hysteresis,
    };
    Ok(GeneratedLod { level, landmarks, report: r })
}

/// Diagnostics for tests: the worst measured vertices.
#[cfg(test)]
pub(crate) mod diag {
    use super::{Attr, Baseline, Engine, Probe, ring, vertex_error};
    use humanart_math::Vec3;

    /// A measured vertex: `(a, distance, skin L1, nearest triangle)`.
    pub(crate) type Worst = (u32, f32, f32, u32);

    /// Per welded LOD0 vertex `(a, distance, skin L1, nearest triangle)`, the `k` largest by skin
    /// and the `k` largest by distance.
    pub(crate) fn worst(e: &Engine<'_>, base: &Baseline, positions: &[Vec3], k: usize) -> (Vec<Worst>, Vec<Worst>) {
        let mut all = Vec::new();
        let mut cands = Vec::new();
        for &a in &base.measured {
            if !e.used[a as usize] {
                continue;
            }
            cands.clear();
            ring(e, e.rep(a), &mut cands);
            let mut attr = Attr::new(e.deltas.components);
            let Some((err, t, _)) = vertex_error(e, positions, a, e.rep(a), &cands, &mut Probe::default(), &mut attr)
            else {
                continue;
            };
            all.push((a, err.geo.max(err.feature), err.skin, t));
        }
        let mut by_skin = all.clone();
        by_skin.sort_by(|x, y| y.2.total_cmp(&x.2));
        by_skin.truncate(k);
        all.sort_by(|x, y| y.1.total_cmp(&x.1));
        all.truncate(k);
        (by_skin, all)
    }
}
