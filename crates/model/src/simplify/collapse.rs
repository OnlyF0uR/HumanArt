//! The constrained half-edge-collapse engine of the LOD generator `[MOD-05]`.
//!
//! The engine works on the triangulated LOD0 mesh with welded topology: a *vertex* is a weld key
//! (`Mesh::position_keys`), a *wedge* is an original mesh vertex (one of the duplicates a UV seam or
//! hard edge splits a position into), and every triangle corner stores its wedge. A collapse
//! `u -> v` removes the welded vertex `u` and moves each of its triangle corners to the wedge of
//! `v` on the same side of every seam, so seams, tiles and parts survive without cracks.
//!
//! Every surviving vertex is an original LOD0 vertex at its original position (half-edge collapse,
//! the vertex placement of meshoptimizer's simplifier); the Garland–Heckbert quadrics rank the
//! collapses. See the module docs of [`crate::simplify`] for the constraints and the cost.
//!
//! Before a collapse is applied in a run, it is made on the topology only and every LOD0 vertex
//! whose measurement it can change (the vertices merged into `v`'s two-ring) is measured exactly
//! as the level report measures it (`report::vertex_error`); a collapse that takes one of them over
//! a budget is undone and vetoed for the rest of the level.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};

use humanart_math::Vec3;
use humanart_math::geom::closest_point_on_triangle;

use super::quadric::{AttrQuadric, D3, Quadric, cross, dot, length, scale, sub};
use super::report::{self, Attr, Probe};
use super::uvgrid::{self, UvGrid};
use super::{LodInput, LodSettings, MAX_LOOPS};
use crate::delta::DeltaBases;
use crate::error::{ModelError, Result};
use crate::skin::SkinWeights;

/// "No vertex" / "no parent".
pub(crate) const NONE: u32 = u32::MAX;
/// "No region" (no region map, or no foreign region next to an edge).
pub(crate) const NO_REGION: u16 = u16::MAX;
/// "No loop".
pub(crate) const NO_LOOP: u16 = u16::MAX;

/// Smallest live triangle count a closed part keeps (a tetrahedron).
const MIN_CLOSED_PART_TRIANGLES: u32 = 4;
/// Smallest live triangle count an open part keeps.
const MIN_OPEN_PART_TRIANGLES: u32 = 2;
/// Smallest twice-area (UV units) of a UV triangle a collapse changes (and of a triangle of a split
/// LOD0 polygon whose corners may move): well above the rounding of
/// the validator's `f32` shoelace sum over absolute UVs (about 1e-7 for UVs near 1), so a changed
/// triangle never reads as flipped or collapsed there (`crate::validate::UV_EPSILON`).
const MIN_UV_DOUBLE_AREA: f64 = 4e-7;

/// The feature signature of an edge: a vertex lies on one feature line when its two feature edges
/// have equal signatures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Sig {
    border: bool,
    seam: bool,
    parts: (u16, u16),
    other_region: u16,
    loop_id: u16,
}

impl Sig {
    pub(crate) fn is_feature(&self) -> bool {
        self.border
            || self.seam
            || self.parts.0 != self.parts.1
            || self.other_region != NO_REGION
            || self.loop_id != NO_LOOP
    }
}

/// The per-level limits of a collapse.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Rules {
    /// RMS plane-distance estimate limit, metres.
    pub geometric: f64,
    /// Tracked delta bound limit, metres.
    pub delta: f32,
    /// Tracked skin-weight bound limit (L1).
    pub skin: f32,
}

/// Which protected loops count as feature lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoopMode {
    /// None (the permanent features only).
    Ignore,
    /// The loops active at the current level.
    Active,
    /// Every loop.
    All,
}

/// The protected loops: per welded vertex its loop and neighbours along it.
#[derive(Clone, Debug, Default)]
pub(crate) struct Loops {
    pub of: Vec<u16>,
    pub next: Vec<u32>,
    pub prev: Vec<u32>,
    pub len: Vec<u32>,
    pub lock_through: Vec<u8>,
    pub keep_through: Vec<u8>,
    pub locked: Vec<bool>,
    pub active: Vec<bool>,
}

impl Loops {
    /// The loop whose edge `(u, x)` is, if `mode` counts that loop.
    fn edge(&self, u: u32, x: u32, mode: LoopMode) -> u16 {
        let l = self.of[u as usize];
        if l == NO_LOOP {
            return NO_LOOP;
        }
        let counted = match mode {
            LoopMode::Ignore => false,
            LoopMode::Active => self.active[l as usize],
            LoopMode::All => true,
        };
        if self.of[x as usize] != l || !counted {
            return NO_LOOP;
        }
        if self.next[u as usize] == x || self.prev[u as usize] == x { l } else { NO_LOOP }
    }
}

/// Vertex-major sparse deltas: per welded vertex its `(component, delta)` entries by component.
#[derive(Clone, Debug, Default)]
pub(crate) struct DeltaRows {
    pub offsets: Vec<u32>,
    pub comps: Vec<u32>,
    pub values: Vec<[f32; 3]>,
    pub components: usize,
}

impl DeltaRows {
    pub(crate) fn new(bases: Option<&DeltaBases>, weld: &[u32]) -> Result<Self> {
        let n = weld.len();
        let Some(bases) = bases else {
            return Ok(Self { offsets: vec![0; n + 1], ..Self::default() });
        };
        let mut counts = vec![0_u32; n + 1];
        for c in 0..bases.len() {
            for &i in bases.indices(c) {
                let Some(&k) = weld.get(i as usize) else {
                    return Err(ModelError::invalid("delta entry", i as usize, "vertex out of range"));
                };
                // Duplicates carry the same delta as their key (a validation rule); the key's is used.
                if k == i {
                    counts[i as usize + 1] += 1;
                }
            }
        }
        for i in 0..n {
            counts[i + 1] += counts[i];
        }
        let total = counts[n] as usize;
        let mut fill = counts.clone();
        let (mut comps, mut values) = (vec![0_u32; total], vec![[0.0_f32; 3]; total]);
        for c in 0..bases.len() {
            for (&i, &d) in bases.indices(c).iter().zip(bases.deltas(c)) {
                if weld[i as usize] == i {
                    let slot = fill[i as usize] as usize;
                    fill[i as usize] += 1;
                    comps[slot] = c as u32;
                    let d = DeltaBases::delta_f32(d);
                    values[slot] = [d.x, d.y, d.z];
                }
            }
        }
        Ok(Self { offsets: counts, comps, values, components: bases.len() })
    }

    /// The entries of welded vertex `v`.
    pub(crate) fn row(&self, v: u32) -> (&[u32], &[[f32; 3]]) {
        let (a, b) = (self.offsets[v as usize] as usize, self.offsets[v as usize + 1] as usize);
        (&self.comps[a..b], &self.values[a..b])
    }
}

/// Reusable buffers of the evaluation.
#[derive(Clone, Debug, Default)]
pub(crate) struct Scratch {
    fan: Vec<u32>,
    pairs: Vec<(u32, u32)>,
    nbrs: Vec<u32>,
    targets: Vec<u32>,
    nv: Vec<u32>,
    edge: Vec<u32>,
    map: Vec<(u32, u32)>,
    requeue: Vec<u32>,
    retarget: Vec<[u32; 3]>,
    acc: Vec<[f32; 3]>,
    touched: Vec<u32>,
    weights: Vec<(u16, f32)>,
    /// Whether `targets` are the two neighbours along a feature line.
    on_line: bool,
    /// The representatives a collapse's verification measures.
    reps: Vec<u32>,
    /// The candidate triangles of one representative.
    cands: Vec<u32>,
    /// Triangle corners a tentative collapse changed: (triangle, corner, previous wedge).
    undo_corners: Vec<(u32, u8, u32)>,
    /// The fan of `v` before a tentative collapse.
    undo_fan: Vec<u32>,
    /// `v` and its neighbours during a verification.
    near1: Vec<u32>,
    /// The triangles a tentative collapse changed.
    changed: Vec<u32>,
    /// The nearest triangles and distances a verification measured: (LOD0 vertex, triangle, distance).
    updates: Vec<(u32, u32, f32)>,
}

/// Why a vertex cannot collapse (or why one of its collapses is invalid), in the order the checks
/// run: the census of a stalled level counts these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Block {
    /// The vertex is dead, complex (non-manifold, bow-tie, mixed regions) or locked by the caller.
    Locked,
    /// The vertex is on an active protected loop locked at this level.
    LoopLocked,
    /// An edge of the vertex has more than two triangles.
    NonManifold,
    /// The vertex has one feature edge, more than two, or two of different signatures.
    Junction,
    /// The target is dead, complex or in another region.
    Target,
    /// The vertices share no edge or the edge has more than two triangles.
    Edge,
    /// The link condition fails (the collapse would make the surface non-manifold).
    Link,
    /// A part or a protected loop would drop below its minimum.
    Minimum,
    /// The wedges do not map consistently across the edge.
    WedgeMap,
    /// A triangle would degenerate, flip or tilt beyond the normal limit.
    Normal,
    /// A UV triangle would flip or collapse.
    UvFlip,
    /// A triangle would exist twice.
    Duplicate,
    /// The geometric estimate exceeds the level's limit.
    Geometric,
    /// The delta estimate exceeds the level's limit.
    Delta,
    /// The skin estimate exceeds the level's limit.
    Skin,
    /// The cost is not finite.
    Cost,
}

/// The outcome of a valid collapse check.
#[derive(Clone, Copy, Debug)]
struct Checked {
    cost: f64,
    geo: f32,
    delta: f32,
    skin: f32,
}

/// The working state.
#[derive(Clone)]
pub(crate) struct Engine<'a> {
    pub weld: &'a [u32],
    /// LOD0 positions per wedge (what the measurement uses).
    positions: &'a [Vec3],
    /// Per welded vertex: whether it lies on a permanent feature line at LOD0.
    pub feature: Vec<bool>,
    /// Per live vertex: the welded LOD0 vertices merged into it (itself included).
    members: Vec<Vec<u32>>,
    /// Per welded LOD0 vertex: its nearest live triangle and distance when last measured by a
    /// verification (`NONE`: unknown, measure again).
    near: Vec<(u32, f32)>,
    /// Collapses refused by verification at this level.
    vetoes: BTreeSet<(u32, u32)>,
    probe: Probe,
    attr: Attr,
    pub pos: Vec<D3>,
    pub uv: Vec<[f64; 2]>,
    /// The LOD0 UVs per wedge, unscaled (the flip and overlap checks).
    uv_raw: Vec<[f64; 2]>,
    /// Live triangles by UV cell (the overlap check).
    uv_grid: UvGrid,
    pub region: Vec<u16>,
    pub tris: Vec<[u32; 3]>,
    pub tri_part: Vec<u16>,
    pub tri_tile: Vec<u16>,
    /// The LOD0 face of each triangle.
    pub tri_face: Vec<u32>,
    /// Whether a collapse moved a corner of the triangle.
    pub tri_changed: Vec<bool>,
    pub tri_alive: Vec<bool>,
    pub part_alive: Vec<u32>,
    part_min: Vec<u32>,
    pub fans: Vec<Vec<u32>>,
    pub alive: Vec<bool>,
    /// Vertices that had triangles in LOD0.
    pub used: Vec<bool>,
    pub parent: Vec<u32>,
    complex: Vec<bool>,
    user_lock: Vec<bool>,
    gq: Vec<Quadric>,
    aq: Vec<AttrQuadric>,
    area: Vec<f64>,
    /// Per live vertex: the accumulated estimate of the distance of the vertices merged into it.
    pub geo_err: Vec<f32>,
    /// The same for the delta interpolation error.
    pub delta_err: Vec<f32>,
    /// The same for the skin-weight interpolation error (L1).
    pub skin_err: Vec<f32>,
    pub deltas: &'a DeltaRows,
    /// The collapses of the current level, in order.
    pub log: Vec<(u32, u32)>,
    pub skin: Option<&'a SkinWeights>,
    pub loops: Loops,
    version: Vec<u32>,
    heap: BinaryHeap<Reverse<(u64, u32, u32, u32)>>,
    pub alive_tris: usize,
    min_n2: f64,
    min_normal_dot: f64,
    delta_weight: f64,
    skin_weight: f64,
    rules: Rules,
    s: Scratch,
}

/// The corner of triangle `t` whose wedge welds to `u` (the caller knows there is one).
#[inline]
fn corner(tri: [u32; 3], weld: &[u32], u: u32) -> usize {
    if weld[tri[0] as usize] == u {
        0
    } else if weld[tri[1] as usize] == u {
        1
    } else {
        2
    }
}

/// The distance from `p` to the segment `ab`.
fn segment_distance(p: D3, a: D3, b: D3) -> f64 {
    let ab = sub(b, a);
    let len2 = dot(ab, ab);
    let t = if len2 > 0.0 { (dot(sub(p, a), ab) / len2).clamp(0.0, 1.0) } else { 0.0 };
    length(sub(p, [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t]))
}

/// The signed UV double area of three UV points.
#[inline]
fn uv_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

impl<'a> Engine<'a> {
    /// Builds the working state from the input (see [`LodInput`] for what is checked).
    pub(crate) fn new(input: &LodInput<'a>, settings: &LodSettings, deltas: &'a DeltaRows) -> Result<Self> {
        let mesh = input.mesh;
        let n = mesh.vertex_count();
        let weld = mesh.position_keys();
        let tri = mesh.triangulate();
        let tris: Vec<[u32; 3]> = tri.indices.as_chunks::<3>().0.to_vec();
        if tris.is_empty() {
            return Err(ModelError::invalid("LOD input", 0, "the mesh has no faces"));
        }
        let mut face_part = vec![0_u16; mesh.face_count()];
        for (i, part) in mesh.parts().iter().enumerate() {
            for f in part.faces() {
                if let Some(slot) = face_part.get_mut(f) {
                    *slot = i as u16;
                }
            }
        }
        let tri_part: Vec<u16> = tri.triangle_faces.iter().map(|&f| face_part[f as usize]).collect();
        let tri_tile: Vec<u16> = tri.triangle_faces.iter().map(|&f| mesh.face_tiles()[f as usize]).collect();
        let mut part_alive = vec![0_u32; mesh.parts().len()];
        for &p in &tri_part {
            part_alive[p as usize] += 1;
        }
        let part_min = mesh
            .parts()
            .iter()
            .map(|p| if p.closed { MIN_CLOSED_PART_TRIANGLES } else { MIN_OPEN_PART_TRIANGLES })
            .collect();

        // Positions relative to the centroid, in f64.
        let mut centre = [0.0_f64; 3];
        for p in mesh.positions() {
            centre = [centre[0] + f64::from(p.x), centre[1] + f64::from(p.y), centre[2] + f64::from(p.z)];
        }
        let centre = scale(centre, 1.0 / n.max(1) as f64);
        let pos: Vec<D3> =
            mesh.positions().iter().map(|p| sub([f64::from(p.x), f64::from(p.y), f64::from(p.z)], centre)).collect();
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for p in &pos {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let diag = length(sub(hi, lo)).max(1e-9);
        let eps = diag * 1e-6;
        let min_n2 = eps * eps * eps * eps;

        // Regions per weld key; wedges of one position in different regions lock it.
        let mut region = vec![NO_REGION; n];
        let mut complex = vec![false; n];
        if let Some(map) = input.regions {
            if map.vertex_regions.len() != n {
                return Err(ModelError::invalid("LOD input", 0, "region map does not match the vertex count"));
            }
            for (w, &r) in map.vertex_regions.iter().enumerate() {
                let k = weld[w] as usize;
                if region[k] == NO_REGION {
                    region[k] = r;
                } else if region[k] != r {
                    complex[k] = true;
                }
            }
        }
        let mut user_lock = vec![false; n];
        for &v in input.locked {
            let k = *weld.get(v as usize).ok_or(ModelError::invalid("locked vertex", v as usize, "out of range"))?;
            user_lock[k as usize] = true;
        }
        if let Some(skin) = input.skin
            && skin.influences > 0
            && skin.vertex_count() != n
        {
            return Err(ModelError::invalid("LOD input", 0, "skin weights do not match the vertex count"));
        }

        let mut fans: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (t, c) in tris.iter().enumerate() {
            for &w in c {
                fans[weld[w as usize] as usize].push(t as u32);
            }
        }
        let used: Vec<bool> = fans.iter().map(|f| !f.is_empty()).collect();
        let members = (0..n as u32)
            .map(|v| if weld[v as usize] == v && used[v as usize] { vec![v] } else { Vec::new() })
            .collect();
        let mut engine = Self {
            weld,
            positions: mesh.positions(),
            feature: Vec::new(),
            members,
            near: vec![(NONE, 0.0); n],
            vetoes: BTreeSet::new(),
            probe: Probe::default(),
            attr: Attr::new(deltas.components),
            uv: Vec::new(),
            uv_raw: Vec::new(),
            uv_grid: UvGrid::default(),
            region,
            tri_part,
            tri_tile,
            tri_alive: vec![true; tris.len()],
            tri_changed: vec![false; tris.len()],
            tri_face: tri.triangle_faces.clone(),
            part_alive,
            part_min,
            alive: used.clone(),
            used,
            parent: vec![NONE; n],
            complex,
            user_lock,
            gq: vec![Quadric::default(); n],
            aq: vec![AttrQuadric::default(); n],
            area: vec![0.0; n],
            geo_err: vec![0.0; n],
            delta_err: vec![0.0; n],
            skin_err: vec![0.0; n],
            deltas,
            log: Vec::new(),
            skin: input.skin.filter(|s| s.influences > 0),
            loops: Loops::default(),
            version: vec![0; n],
            heap: BinaryHeap::new(),
            alive_tris: tris.len(),
            min_n2,
            min_normal_dot: f64::from(settings.min_normal_dot),
            delta_weight: f64::from(settings.delta_weight),
            skin_weight: f64::from(settings.skin_length) * f64::from(settings.skin_length),
            rules: Rules { geometric: f64::INFINITY, delta: f32::INFINITY, skin: f32::INFINITY },
            s: Scratch { acc: vec![[0.0; 3]; deltas.components], ..Scratch::default() },
            tris,
            fans,
            pos,
        };
        engine.loops = engine.build_loops(input)?;
        engine.mark_complex();
        engine.build_quadrics(mesh, settings);
        engine.feature = report::feature_flags(&engine);
        Ok(engine)
    }

    fn build_loops(&self, input: &LodInput<'_>) -> Result<Loops> {
        let n = self.weld.len();
        if input.loops.len() > MAX_LOOPS {
            return Err(ModelError::limit("protected loop count", MAX_LOOPS as u64, input.loops.len() as u64));
        }
        let mut l = Loops {
            of: vec![NO_LOOP; n],
            next: vec![NONE; n],
            prev: vec![NONE; n],
            len: Vec::new(),
            lock_through: Vec::new(),
            keep_through: Vec::new(),
            locked: vec![false; input.loops.len()],
            active: vec![false; input.loops.len()],
        };
        let mut keys = Vec::new();
        for (i, lp) in input.loops.iter().enumerate() {
            keys.clear();
            for &v in &lp.vertices {
                let k = *self.weld.get(v as usize).ok_or(ModelError::invalid(
                    "protected loop",
                    i,
                    "vertex out of range",
                ))?;
                keys.push(k);
            }
            if keys.len() < 3 {
                return Err(ModelError::invalid("protected loop", i, "loop has fewer than 3 vertices"));
            }
            for (j, &k) in keys.iter().enumerate() {
                let next = keys[(j + 1) % keys.len()];
                if l.of[k as usize] != NO_LOOP {
                    return Err(ModelError::invalid("protected loop", i, "vertex used twice or by two loops"));
                }
                if !self.fans[k as usize]
                    .iter()
                    .any(|&t| self.tris[t as usize].iter().any(|&w| self.weld[w as usize] == next))
                {
                    return Err(ModelError::invalid("protected loop", i, "consecutive loop vertices share no edge"));
                }
                l.of[k as usize] = i as u16;
                l.next[k as usize] = next;
                l.prev[next as usize] = k;
            }
            l.len.push(keys.len() as u32);
            l.lock_through.push(lp.lock_through);
            l.keep_through.push(lp.keep_through);
        }
        Ok(l)
    }

    /// Locks vertices whose neighbourhood is not a single consistently wound disc or half-disc:
    /// non-manifold edges, bow-ties, inconsistent winding.
    fn mark_complex(&mut self) {
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        let mut comp: Vec<u32> = Vec::new();
        for u in 0..self.weld.len() as u32 {
            if !self.used[u as usize] || self.complex[u as usize] {
                continue;
            }
            pairs.clear();
            let fan = &self.fans[u as usize];
            for (slot, &t) in fan.iter().enumerate() {
                for &w in &self.tris[t as usize] {
                    let x = self.weld[w as usize];
                    if x != u {
                        pairs.push((x, slot as u32));
                    }
                }
            }
            pairs.sort_unstable();
            // Union-find over the fan's triangles (by slot).
            comp.clear();
            comp.extend(0..fan.len() as u32);
            let find = |comp: &mut Vec<u32>, mut a: u32| {
                while comp[a as usize] != a {
                    let up = comp[comp[a as usize] as usize];
                    comp[a as usize] = up;
                    a = up;
                }
                a
            };
            let (mut bad, mut borders) = (false, 0);
            for group in pairs.chunk_by(|a, b| a.0 == b.0) {
                match group {
                    [_] => borders += 1,
                    [(x, s0), (_, s1)] => {
                        let (t0, t1) = (self.tris[fan[*s0 as usize] as usize], self.tris[fan[*s1 as usize] as usize]);
                        // Consistent winding: one triangle runs u -> x, the other x -> u.
                        let runs = |t: [u32; 3]| {
                            let k = corner(t, self.weld, u);
                            self.weld[t[(k + 1) % 3] as usize] == *x
                        };
                        if runs(t0) == runs(t1) {
                            bad = true;
                        }
                        let (a, b) = (find(&mut comp, *s0), find(&mut comp, *s1));
                        comp[a as usize] = b;
                    }
                    _ => bad = true,
                }
            }
            let roots = (0..fan.len() as u32).filter(|&s| find(&mut comp, s) == s).count();
            if bad || roots != 1 || !(borders == 0 || borders == 2) {
                self.complex[u as usize] = true;
            }
        }
    }

    fn build_quadrics(&mut self, mesh: &crate::mesh::Mesh, settings: &LodSettings) {
        // UV attributes in metres: scaled by sqrt(surface area / UV area) and the UV weight.
        let (mut area3, mut area_uv) = (0.0_f64, 0.0_f64);
        let uv_raw: Vec<[f64; 2]> = mesh.uvs().iter().map(|t| [f64::from(t.x), f64::from(t.y)]).collect();
        for c in &self.tris {
            let p = c.map(|w| self.pos[self.weld[w as usize] as usize]);
            area3 += 0.5 * length(cross(sub(p[1], p[0]), sub(p[2], p[0])));
            area_uv += 0.5 * uv_cross(uv_raw[c[0] as usize], uv_raw[c[1] as usize], uv_raw[c[2] as usize]).abs();
        }
        let uv_scale = if area_uv > 0.0 { (area3 / area_uv).sqrt() * f64::from(settings.uv_weight) } else { 0.0 };
        self.uv = uv_raw.iter().map(|t| [t[0] * uv_scale, t[1] * uv_scale]).collect();
        // A LOD0 polygon split into triangles of which one is a UV sliver is only valid whole: its
        // corners stay, so its triangles stay unchanged and the level keeps the polygon.
        for (t, c) in self.tris.iter().enumerate() {
            let f = self.tri_face[t];
            let split = (t > 0 && self.tri_face[t - 1] == f) || self.tri_face.get(t + 1) == Some(&f);
            if split
                && uv_cross(uv_raw[c[0] as usize], uv_raw[c[1] as usize], uv_raw[c[2] as usize]).abs()
                    < MIN_UV_DOUBLE_AREA
            {
                for &w in mesh.face(f as usize) {
                    self.user_lock[self.weld[w as usize] as usize] = true;
                }
            }
        }
        self.uv_raw = uv_raw;
        let tiles = self.tri_tile.iter().map(|&t| usize::from(t) + 1).max().unwrap_or(1);
        self.uv_grid = UvGrid::new(tiles, self.tris.len());
        for (t, c) in self.tris.iter().enumerate() {
            self.uv_grid.insert(self.tri_tile[t], t as u32, c.map(|w| self.uv_raw[w as usize]));
        }

        for c in &self.tris {
            let k = c.map(|w| self.weld[w as usize] as usize);
            let p = k.map(|i| self.pos[i]);
            let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
            let n2 = dot(n, n);
            if n2.is_nan() || n2 <= self.min_n2 {
                continue;
            }
            let len = n2.sqrt();
            let area = 0.5 * len;
            let q = Quadric::plane(p[0], scale(n, 1.0 / len), area);
            let a = AttrQuadric::triangle(p, c.map(|w| self.uv[w as usize]), area, self.min_n2);
            for (j, &i) in k.iter().enumerate() {
                self.gq[i].add(&q);
                self.area[i] += area;
                self.aq[c[j] as usize].add(&a);
            }
        }
        // Feature edges: planes through the edge, perpendicular to each adjacent triangle.
        let fw = f64::from(settings.feature_weight);
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for u in 0..self.weld.len() as u32 {
            if !self.used[u as usize] {
                continue;
            }
            pairs.clear();
            for &t in &self.fans[u as usize] {
                for &w in &self.tris[t as usize] {
                    let x = self.weld[w as usize];
                    if x > u {
                        pairs.push((x, t));
                    }
                }
            }
            pairs.sort_unstable();
            for group in pairs.chunk_by(|a, b| a.0 == b.0) {
                let x = group[0].0;
                if group.len() > 2 || !self.edge_sig(u, x, group, LoopMode::All).is_feature() {
                    continue;
                }
                let e = sub(self.pos[x as usize], self.pos[u as usize]);
                let e2 = dot(e, e);
                for &(_, t) in group {
                    let p = self.tris[t as usize].map(|w| self.pos[self.weld[w as usize] as usize]);
                    let nt = cross(sub(p[1], p[0]), sub(p[2], p[0]));
                    let m = cross(e, nt);
                    let ml = length(m);
                    if !ml.is_finite() || ml <= 0.0 {
                        continue;
                    }
                    let q = Quadric::plane(self.pos[u as usize], scale(m, 1.0 / ml), fw * e2);
                    self.gq[u as usize].add(&q);
                    self.gq[x as usize].add(&q);
                }
            }
        }
    }

    /// The signature of edge `(u, x)` given its triangles (`group` holds `(x, t)` pairs).
    pub(crate) fn edge_sig(&self, u: u32, x: u32, group: &[(u32, u32)], loops: LoopMode) -> Sig {
        let t0 = self.tris[group[0].1 as usize];
        let p0 = self.tri_part[group[0].1 as usize];
        let mut sig =
            Sig { border: group.len() == 1, seam: false, parts: (p0, p0), other_region: NO_REGION, loop_id: NO_LOOP };
        if let [_, (_, t1)] = *group {
            let p1 = self.tri_part[t1 as usize];
            let t1 = self.tris[t1 as usize];
            sig.parts = (p0.min(p1), p0.max(p1));
            let (cu0, cu1) = (corner(t0, self.weld, u), corner(t1, self.weld, u));
            let (cx0, cx1) = (corner(t0, self.weld, x), corner(t1, self.weld, x));
            sig.seam = t0[cu0] != t1[cu1] || t0[cx0] != t1[cx1];
        }
        let ru = self.region[u as usize];
        if self.region[x as usize] == ru {
            for &(_, t) in group {
                for &w in &self.tris[t as usize] {
                    let r = self.region[self.weld[w as usize] as usize];
                    if r != ru {
                        sig.other_region = sig.other_region.min(r);
                    }
                }
            }
        }
        sig.loop_id = self.loops.edge(u, x, loops);
        sig
    }

    /// Sets the rules and loop states of level `level` (1-based) and rebuilds the queue.
    pub(crate) fn begin_level(&mut self, level: u8, rules: Rules) {
        self.rules = rules;
        for i in 0..self.loops.len.len() {
            self.loops.locked[i] = level <= self.loops.lock_through[i];
            self.loops.active[i] = level <= self.loops.keep_through[i] || self.loops.locked[i];
        }
        self.heap.clear();
        self.log.clear();
        self.vetoes.clear();
        for u in 0..self.weld.len() as u32 {
            self.version[u as usize] = self.version[u as usize].wrapping_add(1);
            self.push(u);
        }
    }

    fn push(&mut self, u: u32) {
        if let Some((cost, v)) = self.evaluate(u) {
            self.heap.push(Reverse((cost.to_bits(), u, v, self.version[u as usize])));
        }
    }

    /// Collapses until at most `target` triangles are left or no valid collapse remains.
    ///
    /// Progress: every pop either collapses (removing a vertex), drops a stale or invalid entry, or
    /// re-queues a vertex whose cost rose since it was queued; a re-queued entry pops with an
    /// unchanged cost unless a collapse happened in between, so between two collapses each vertex is
    /// popped at most twice. The pop cap is a second guard.
    pub(crate) fn run(&mut self, target: usize) {
        let cap = 64 * (self.weld.len() + self.tris.len()) + 1024;
        let mut pops = 0_usize;
        while self.alive_tris > target && pops < cap {
            pops += 1;
            let Some(Reverse((cost, u, _, ver))) = self.heap.pop() else { break };
            if ver != self.version[u as usize] || !self.alive[u as usize] {
                continue;
            }
            self.version[u as usize] = self.version[u as usize].wrapping_add(1);
            match self.evaluate(u) {
                None => {}
                Some((c, v)) if c.to_bits() <= cost => {
                    if !self.collapse(u, v, true) {
                        // Refused by verification: try `u`'s other targets.
                        self.vetoes.insert((u, v));
                        self.push(u);
                    }
                }
                Some((c, v)) => self.heap.push(Reverse((c.to_bits(), u, v, self.version[u as usize]))),
            }
        }
    }

    fn gather_fan(&self, u: u32, fan: &mut Vec<u32>) {
        fan.clear();
        fan.extend(self.fans[u as usize].iter().copied().filter(|&t| self.tri_alive[t as usize]));
    }

    /// The cheapest valid collapse of `u` and its target.
    fn evaluate(&mut self, u: u32) -> Option<(f64, u32)> {
        let ui = u as usize;
        if self.movable(ui).is_err() {
            return None;
        }
        let mut s = std::mem::take(&mut self.s);
        let mut best: Option<(f64, u32)> = None;
        self.gather_fan(u, &mut s.fan);
        if self.targets(u, &mut s).is_ok() {
            for i in 0..s.targets.len() {
                let v = s.targets[i];
                if !self.vetoes.is_empty() && self.vetoes.contains(&(u, v)) {
                    continue;
                }
                if let Ok(c) = self.check(u, v, &mut s)
                    && best.is_none_or(|(b, _)| c.cost < b)
                {
                    best = Some((c.cost, v));
                }
            }
        }
        self.s = s;
        best
    }

    /// Whether welded vertex `ui` may move at all at this level.
    fn movable(&self, ui: usize) -> std::result::Result<(), Block> {
        if !self.alive[ui] || self.complex[ui] || self.user_lock[ui] {
            return Err(Block::Locked);
        }
        let l = self.loops.of[ui];
        if l != NO_LOOP && self.loops.locked[l as usize] {
            return Err(Block::LoopLocked);
        }
        Ok(())
    }

    /// Fills `s.nbrs` and `s.targets` (both ascending) from `s.fan`; an error when `u` cannot move.
    fn targets(&self, u: u32, s: &mut Scratch) -> std::result::Result<(), Block> {
        s.pairs.clear();
        for &t in &s.fan {
            for &w in &self.tris[t as usize] {
                let x = self.weld[w as usize];
                if x != u {
                    s.pairs.push((x, t));
                }
            }
        }
        s.pairs.sort_unstable();
        s.nbrs.clear();
        s.targets.clear();
        let mut feats: [(u32, Option<Sig>); 2] = [(NONE, None); 2];
        let mut k = 0;
        for group in s.pairs.chunk_by(|a, b| a.0 == b.0) {
            let x = group[0].0;
            s.nbrs.push(x);
            if group.len() > 2 {
                return Err(Block::NonManifold);
            }
            let sig = self.edge_sig(u, x, group, LoopMode::Active);
            if sig.is_feature() {
                if k < 2 {
                    feats[k] = (x, Some(sig));
                }
                k += 1;
            }
        }
        s.on_line = k == 2;
        match k {
            0 => s.targets.extend_from_slice(&s.nbrs),
            2 if feats[0].1 == feats[1].1 => s.targets.extend([feats[0].0, feats[1].0]),
            _ => return Err(Block::Junction),
        }
        Ok(())
    }

    /// Checks the collapse `u -> v` (with `s.fan` and `s.nbrs` of `u`) and returns its cost; fills
    /// `s.edge` and `s.map`.
    fn check(&self, u: u32, v: u32, s: &mut Scratch) -> std::result::Result<Checked, Block> {
        let (ui, vi) = (u as usize, v as usize);
        if !self.alive[vi] || self.complex[vi] || self.region[ui] != self.region[vi] {
            return Err(Block::Target);
        }
        s.edge.clear();
        s.edge.extend(
            s.fan.iter().copied().filter(|&t| self.tris[t as usize].iter().any(|&w| self.weld[w as usize] == v)),
        );
        if s.edge.is_empty() || s.edge.len() > 2 {
            return Err(Block::Edge);
        }
        // Link condition: the common neighbours are exactly the apexes of the edge's triangles.
        s.nv.clear();
        for &t in &self.fans[vi] {
            if self.tri_alive[t as usize] {
                s.nv.extend(self.tris[t as usize].iter().map(|&w| self.weld[w as usize]).filter(|&x| x != v));
            }
        }
        s.nv.sort_unstable();
        s.nv.dedup();
        let (mut i, mut j, mut common) = (0, 0, 0);
        while i < s.nbrs.len() && j < s.nv.len() {
            match s.nbrs[i].cmp(&s.nv[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    common += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        if common != s.edge.len() {
            return Err(Block::Link);
        }
        // Parts keep a minimum of triangles.
        for (k, &t) in s.edge.iter().enumerate() {
            let p = self.tri_part[t as usize] as usize;
            let removed =
                1 + s.edge[k + 1..].iter().filter(|&&o| self.tri_part[o as usize] as usize == p).count() as u32;
            if s.edge[..k].iter().all(|&o| self.tri_part[o as usize] as usize != p)
                && self.part_alive[p] < self.part_min[p] + removed
            {
                return Err(Block::Minimum);
            }
        }
        // Loops keep at least three vertices.
        let l = self.loops.edge(u, v, LoopMode::Active);
        if l != NO_LOOP && self.loops.len[l as usize] <= 3 {
            return Err(Block::Minimum);
        }
        // Wedge map from the triangles on the edge.
        s.map.clear();
        for &t in &s.edge {
            let c = self.tris[t as usize];
            let (wu, wv) = (c[corner(c, self.weld, u)], c[corner(c, self.weld, v)]);
            match s.map.iter().find(|m| m.0 == wu) {
                Some(m) if m.1 != wv => return Err(Block::WedgeMap),
                Some(_) => {}
                None => s.map.push((wu, wv)),
            }
        }
        let (pu, pv) = (self.pos[ui], self.pos[vi]);
        s.retarget.clear();
        for &t in &s.fan {
            if s.edge.contains(&t) {
                continue;
            }
            let c = self.tris[t as usize];
            let k = corner(c, self.weld, u);
            let wu = c[k];
            let wv = s.map.iter().find(|m| m.0 == wu).ok_or(Block::WedgeMap)?.1;
            let (a, b) = (c[(k + 1) % 3], c[(k + 2) % 3]);
            let (ka, kb) = (self.weld[a as usize], self.weld[b as usize]);
            let (pa, pb) = (self.pos[ka as usize], self.pos[kb as usize]);
            let n_old = cross(sub(pa, pu), sub(pb, pu));
            let n_new = cross(sub(pa, pv), sub(pb, pv));
            let (o2, n2) = (dot(n_old, n_old), dot(n_new, n_new));
            if n2.is_nan() || n2 <= self.min_n2 {
                return Err(Block::Normal);
            }
            if o2 > self.min_n2 && dot(n_old, n_new) < self.min_normal_dot * (o2 * n2).sqrt() {
                return Err(Block::Normal);
            }
            // The UV triangle keeps its orientation and keeps at least the smallest area.
            let (ua, ub) = (self.uv_raw[a as usize], self.uv_raw[b as usize]);
            let (uo, un) = (uv_cross(self.uv_raw[wu as usize], ua, ub), uv_cross(self.uv_raw[wv as usize], ua, ub));
            if uo != 0.0 && (un.is_nan() || un * uo.signum() < MIN_UV_DOUBLE_AREA) {
                return Err(Block::UvFlip);
            }
            // No duplicate triangle (the tetrahedron case).
            let dup = self.fans[vi].iter().any(|&o| {
                self.tri_alive[o as usize] && {
                    let oc = self.tris[o as usize];
                    oc.iter().any(|&w| self.weld[w as usize] == ka) && oc.iter().any(|&w| self.weld[w as usize] == kb)
                }
            });
            if dup {
                return Err(Block::Duplicate);
            }
            s.retarget.push([v, ka, kb]);
        }
        // The errors at u against the new triangles, added to what u already carries.
        let (mut dist, corners, bary) = self.nearest_new(u, v, &s.retarget);
        if s.on_line
            && let Some(&x) = s.targets.iter().find(|&&x| x != v)
        {
            // A feature-line vertex also leaves the line: its distance to the new segment v-x.
            dist = dist.max(segment_distance(self.pos[ui], pv, self.pos[x as usize]) as f32);
        }
        let geo_e = self.geo_err[vi].max(self.geo_err[ui] + dist);
        if f64::from(geo_e) > self.rules.geometric {
            return Err(Block::Geometric);
        }
        let (local_delta, local_skin) = self.interpolation_error(u, corners, bary, s);
        let delta = self.delta_err[vi].max(self.delta_err[ui] + local_delta);
        if delta > self.rules.delta {
            return Err(Block::Delta);
        }
        let skin = self.skin_err[vi].max(self.skin_err[ui] + local_skin);
        if skin > self.rules.skin {
            return Err(Block::Skin);
        }
        let mut q = self.gq[ui];
        q.add(&self.gq[vi]);
        let geo = q.eval(pv);
        let area = self.area[ui] + self.area[vi];
        let mut attr = 0.0;
        for (k, &(wu, wv)) in s.map.iter().enumerate() {
            attr += self.aq[wu as usize].eval(pv, self.uv[wv as usize]);
            if s.map[..k].iter().all(|m| m.1 != wv) {
                attr += self.aq[wv as usize].eval(pv, self.uv[wv as usize]);
            }
        }
        let d = f64::from(delta);
        let w = f64::from(skin);
        let cost = geo + attr + area * (self.delta_weight * d * d + self.skin_weight * w * w);
        if cost.is_finite() && cost >= 0.0 { Ok(Checked { cost, geo: geo_e, delta, skin }) } else { Err(Block::Cost) }
    }

    /// The nearest point to `u` on the triangles `retarget` (welded corners after the collapse
    /// `u -> v`): distance, corners, barycentrics; `v` itself when there are none.
    fn nearest_new(&self, u: u32, v: u32, retarget: &[[u32; 3]]) -> (f32, [u32; 3], [f32; 3]) {
        let f = |x: u32| {
            let p = self.pos[x as usize];
            Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)
        };
        let pu = f(u);
        let mut best = (pu.distance(f(v)), [v, v, v], [1.0, 0.0, 0.0]);
        let mut first = true;
        for &c in retarget {
            let (q, b) = closest_point_on_triangle(pu, f(c[0]), f(c[1]), f(c[2]));
            let d = pu.distance(q);
            if first || d < best.0 {
                best = (d, c, [b.x, b.y, b.z]);
                first = false;
            }
        }
        best
    }

    /// The largest per-component delta difference (metres) and the L1 skin-weight difference
    /// between vertex `a` and the values interpolated at barycentrics `b` of welded `corners`.
    pub(crate) fn interpolation_error(&self, a: u32, corners: [u32; 3], b: [f32; 3], s: &mut Scratch) -> (f32, f32) {
        let mut add = |v: u32, w: f32| {
            let (comps, values) = self.deltas.row(v);
            for (&c, d) in comps.iter().zip(values) {
                let slot = &mut s.acc[c as usize];
                if *slot == [0.0; 3] {
                    s.touched.push(c);
                }
                *slot = [slot[0] + w * d[0], slot[1] + w * d[1], slot[2] + w * d[2]];
            }
        };
        add(a, 1.0);
        for (&c, &w) in corners.iter().zip(&b) {
            add(c, -w);
        }
        let mut delta = 0.0_f32;
        for &c in &s.touched {
            let d = s.acc[c as usize];
            delta = delta.max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
            s.acc[c as usize] = [0.0; 3];
        }
        s.touched.clear();
        let Some(skin) = self.skin else { return (delta, 0.0) };
        s.weights.clear();
        let (j, w) = skin.vertex(a as usize);
        s.weights.extend(j.iter().zip(w).map(|(&j, &w)| (j, w)));
        for (&c, &bw) in corners.iter().zip(&b) {
            let (j, w) = skin.vertex(c as usize);
            s.weights.extend(j.iter().zip(w).map(|(&j, &w)| (j, -w * bw)));
        }
        s.weights.sort_unstable_by_key(|x| x.0);
        let l1 = s.weights.chunk_by(|x, y| x.0 == y.0).map(|g| g.iter().map(|x| x.1).sum::<f32>().abs()).sum();
        (delta, l1)
    }

    /// Applies the collapse `u -> v` if it is valid. With `verified` (a run) it is first made on
    /// the topology and kept only if every LOD0 vertex whose measurement it changes stays within
    /// the level's limits, and the neighbourhood is re-queued; without (a replay of a logged
    /// collapse) neither. Returns whether the collapse was applied.
    fn collapse(&mut self, u: u32, v: u32, verified: bool) -> bool {
        let mut s = std::mem::take(&mut self.s);
        self.gather_fan(u, &mut s.fan);
        let checked = self.targets(u, &mut s).and_then(|()| self.check(u, v, &mut s));
        let Ok(checked) = checked else {
            self.s = s;
            return false;
        };
        let (ui, vi) = (u as usize, v as usize);
        s.undo_corners.clear();
        s.undo_fan.clear();
        s.undo_fan.extend_from_slice(&self.fans[vi]);
        for &t in &s.edge {
            self.tri_alive[t as usize] = false;
            self.alive_tris -= 1;
            self.part_alive[self.tri_part[t as usize] as usize] -= 1;
        }
        for &t in &s.fan {
            if !self.tri_alive[t as usize] {
                continue;
            }
            let c = &mut self.tris[t as usize];
            let k = corner(*c, self.weld, u);
            if let Some(m) = s.map.iter().find(|m| m.0 == c[k]) {
                s.undo_corners.push((t, k as u8, c[k]));
                c[k] = m.1;
                // Entered at its new UV bounds (an undone entry is only stale).
                let uv = c.map(|w| self.uv_raw[w as usize]);
                self.uv_grid.insert(self.tri_tile[t as usize], t, uv);
            }
            self.fans[vi].push(t);
        }
        self.alive[ui] = false;
        self.parent[ui] = v;
        let fan_u = std::mem::take(&mut self.fans[ui]);
        let alive = &self.tri_alive;
        self.fans[vi].retain(|&t| alive[t as usize]);
        if verified && !self.verify(u, v, &mut s) {
            for &(t, k, w) in s.undo_corners.iter().rev() {
                self.tris[t as usize][k as usize] = w;
            }
            for &t in &s.edge {
                self.tri_alive[t as usize] = true;
                self.alive_tris += 1;
                self.part_alive[self.tri_part[t as usize] as usize] += 1;
            }
            self.fans[vi].clear();
            self.fans[vi].extend_from_slice(&s.undo_fan);
            self.fans[ui] = fan_u;
            self.alive[ui] = true;
            self.parent[ui] = NONE;
            self.s = s;
            return false;
        }
        for &(t, _, _) in &s.undo_corners {
            self.tri_changed[t as usize] = true;
        }
        if verified {
            for &(a, t, d) in &s.updates {
                self.near[a as usize] = (t, d);
            }
        }
        let moved = std::mem::take(&mut self.members[ui]);
        self.members[vi].extend_from_slice(&moved);
        let gu = self.gq[ui];
        self.gq[vi].add(&gu);
        for &(wu, wv) in &s.map {
            let a = self.aq[wu as usize];
            self.aq[wv as usize].add(&a);
        }
        self.area[vi] += self.area[ui];
        self.geo_err[vi] = checked.geo;
        self.delta_err[vi] = checked.delta;
        self.log.push((u, v));
        self.skin_err[vi] = checked.skin;
        // Splice the loop when the collapse ran along it.
        let l = self.loops.edge(u, v, LoopMode::Active);
        if l != NO_LOOP {
            let (prev, next) = (self.loops.prev[ui], self.loops.next[ui]);
            if next == v {
                self.loops.next[prev as usize] = v;
                self.loops.prev[vi] = prev;
            } else {
                self.loops.prev[next as usize] = v;
                self.loops.next[vi] = next;
            }
            self.loops.len[l as usize] -= 1;
            self.loops.of[ui] = NO_LOOP;
        }
        if !verified {
            self.s = s;
            return true;
        }
        // Re-queue v and its neighbours.
        s.requeue.clear();
        s.requeue.push(v);
        for &t in &self.fans[vi] {
            s.requeue.extend(self.tris[t as usize].iter().map(|&w| self.weld[w as usize]));
        }
        s.requeue.sort_unstable();
        s.requeue.dedup();
        let requeue = std::mem::take(&mut s.requeue);
        self.s = s;
        for &x in &requeue {
            self.version[x as usize] = self.version[x as usize].wrapping_add(1);
            self.push(x);
        }
        self.s.requeue = requeue;
        true
    }

    /// Whether no triangle the tentative collapse `u -> v` changed overlaps another in UV, and every
    /// welded LOD0 vertex whose measurement it can change measures within the level's limits.
    ///
    /// The changed triangles and feature edges all have `v` and a former neighbour of `u` (in
    /// `s.nbrs`) as corners, so the representatives whose candidates (`report::ring`) or feature
    /// lines changed are `v`, its neighbours and the neighbours of `u`'s former neighbours; `u`'s
    /// merged vertices now belong to `v`.
    fn verify(&mut self, u: u32, v: u32, s: &mut Scratch) -> bool {
        // No changed triangle overlaps another live triangle of its tile in UV.
        for i in 0..s.undo_corners.len() {
            let t = s.undo_corners[i].0;
            let tile = self.tri_tile[t as usize];
            let uv = self.tris[t as usize].map(|w| self.uv_raw[w as usize]);
            s.cands.clear();
            self.uv_grid.query(tile, uv, &mut s.cands);
            s.cands.sort_unstable();
            s.cands.dedup();
            let hit = s.cands.iter().any(|&o| {
                o != t
                    && self.tri_alive[o as usize]
                    && self.tri_tile[o as usize] == tile
                    && uvgrid::overlap(uv, self.tris[o as usize].map(|w| self.uv_raw[w as usize]))
            });
            if hit {
                return false;
            }
        }
        // A LOD0 vertex keeps its errors when its cached nearest triangle is untouched and every
        // triangle that is new or changed among its candidates is farther: for those merged into
        // `v`'s neighbours that is `v`'s fan (they may have gained it), for those merged into the
        // neighbours of `u`'s former neighbours the changed triangles (their rings only lost the
        // edge's triangles). Everything merged into `v` (its ring gained several fans) is measured.
        s.changed.clear();
        s.changed.extend(s.undo_corners.iter().map(|c| c.0));
        s.near1.clear();
        s.near1.push(v);
        for &t in &self.fans[v as usize] {
            s.near1.extend(self.tris[t as usize].iter().map(|&w| self.weld[w as usize]));
        }
        s.near1.sort_unstable();
        s.near1.dedup();
        s.reps.clear();
        s.reps.extend_from_slice(&s.near1);
        for &x in &s.nbrs {
            for &t in &self.fans[x as usize] {
                if self.tri_alive[t as usize] {
                    s.reps.extend(self.tris[t as usize].iter().map(|&w| self.weld[w as usize]));
                }
            }
        }
        s.reps.sort_unstable();
        s.reps.dedup();
        let (mut probe, mut attr) = (std::mem::take(&mut self.probe), std::mem::take(&mut self.attr));
        probe.reset();
        s.updates.clear();
        let mut ok = true;
        'reps: for &r in &s.reps {
            let whole = r == v;
            let fresh: &[u32] = if s.near1.binary_search(&r).is_ok() { &self.fans[v as usize] } else { &s.changed };
            let mut ring_built = false;
            let extra: &[u32] = if r == v { &self.members[u as usize] } else { &[] };
            for &a in self.members[r as usize].iter().chain(extra) {
                let (ta, da) = self.near[a as usize];
                let full = whole
                    || ta == NONE
                    || s.edge.contains(&ta)
                    || s.changed.contains(&ta)
                    || self.within_distance(a, da, fresh);
                let within = if full {
                    if !ring_built {
                        s.cands.clear();
                        report::ring(self, r, &mut s.cands);
                        ring_built = true;
                    }
                    match report::vertex_error(self, self.positions, a, r, &s.cands, &mut probe, &mut attr) {
                        Some((err, t, _)) => {
                            s.updates.push((a, t, err.geo));
                            err.within(&self.rules)
                        }
                        None => false,
                    }
                } else {
                    // Only the feature lines near `r` can have changed.
                    !self.feature[a as usize] || {
                        let p = self.positions[a as usize];
                        f64::from(report::feature_distance(self, self.positions, p, r, &mut probe))
                            <= self.rules.geometric
                    }
                };
                if !within {
                    ok = false;
                    break 'reps;
                }
            }
        }
        self.probe = probe;
        self.attr = attr;
        ok
    }

    /// Whether one of triangles `tris` with a corner in `a`'s region is no farther from LOD0
    /// vertex `a` than `d`.
    fn within_distance(&self, a: u32, d: f32, tris: &[u32]) -> bool {
        let p = self.positions[a as usize];
        let region = self.region[a as usize];
        tris.iter().any(|&t| {
            let c = self.tris[t as usize];
            (region == NO_REGION || c.iter().any(|&w| self.region[self.weld[w as usize] as usize] == region)) && {
                let q = c.map(|w| self.positions[w as usize]);
                p.distance(closest_point_on_triangle(p, q[0], q[1], q[2]).0) <= d
            }
        })
    }

    /// The collapses verification refused at this level.
    pub(crate) fn refused(&self) -> usize {
        self.vetoes.len()
    }

    /// Re-applies logged collapses (from the state the log started in) without queueing.
    pub(crate) fn replay(&mut self, log: &[(u32, u32)]) {
        for &(u, v) in log {
            self.collapse(u, v, false);
        }
        // Replayed collapses are not measured: the cached nearest triangles are stale.
        self.near.fill((NONE, 0.0));
    }

    /// The live vertex `a` was merged into (itself when alive).
    pub(crate) fn rep(&self, a: u32) -> u32 {
        let mut x = a;
        for _ in 0..self.parent.len() {
            let p = self.parent[x as usize];
            if p == NONE {
                break;
            }
            x = p;
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_rows_and_differences() {
        use crate::delta::{DeltaBasesBuilder, DeltaKind};
        use humanart_math::Vec3;
        let mut b = DeltaBasesBuilder::new();
        b.add("a", DeltaKind::Semantic, 0, &[(0, Vec3::X), (2, Vec3::Y)]).unwrap();
        b.add("b", DeltaKind::Semantic, 0, &[(2, Vec3::Z * 2.0), (3, Vec3::X)]).unwrap();
        let d = b.finish().unwrap();
        // Vertex 3 is a duplicate of 2: its entries are ignored in favour of its key's.
        let rows = DeltaRows::new(Some(&d), &[0, 1, 2, 2]).unwrap();
        assert_eq!(rows.row(2).0, &[0, 1]);
        assert!(rows.row(3).0.is_empty());
        assert_eq!(rows.row(0).1, &[[1.0, 0.0, 0.0]]);
        assert!(DeltaRows::new(Some(&d), &[0, 1]).is_err());
    }
}

/// The census of a stalled level (diagnostics for tests): why every live vertex cannot collapse.
#[cfg(test)]
pub(crate) mod census {
    use std::fmt::Write;

    use super::{Block, Engine, LoopMode, NO_LOOP, NO_REGION};

    /// Counts, per live vertex, the furthest check any of its collapses reached, and for vertices
    /// stopped at their feature edges, the kinds of those edges.
    pub(crate) fn census(e: &mut Engine<'_>) -> String {
        let mut reasons: Vec<(Block, usize)> = Vec::new();
        let mut kinds: Vec<(String, usize)> = Vec::new();
        let mut free = 0;
        let mut s = std::mem::take(&mut e.s);
        let bump = |list: &mut Vec<(Block, usize)>, b: Block| match list.iter_mut().find(|x| x.0 == b) {
            Some(x) => x.1 += 1,
            None => list.push((b, 1)),
        };
        for u in 0..e.weld.len() as u32 {
            if !e.alive[u as usize] || !e.used[u as usize] {
                continue;
            }
            if let Err(b) = e.movable(u as usize) {
                bump(&mut reasons, b);
                continue;
            }
            e.gather_fan(u, &mut s.fan);
            if let Err(b) = e.targets(u, &mut s) {
                bump(&mut reasons, b);
                if b == Block::Junction {
                    let key = junction_kind(e, u, &s.fan);
                    match kinds.iter_mut().find(|x| x.0 == key) {
                        Some(x) => x.1 += 1,
                        None => kinds.push((key, 1)),
                    }
                }
                continue;
            }
            let mut best: Option<Block> = None;
            let mut ok = false;
            for i in 0..s.targets.len() {
                let v = s.targets[i];
                match e.check(u, v, &mut s) {
                    Ok(_) => ok = true,
                    Err(b) => best = Some(best.map_or(b, |x| x.max(b))),
                }
            }
            if ok {
                free += 1;
            } else if let Some(b) = best {
                bump(&mut reasons, b);
            }
        }
        e.s = s;
        reasons.sort_by_key(|x| x.0);
        kinds.sort_by_key(|x| std::cmp::Reverse(x.1));
        let mut out = format!("collapsible {free}; blocked:");
        for (b, n) in &reasons {
            write!(out, " {b:?} {n},").unwrap();
        }
        out += "\n  junction kinds (count, B border S seam P part R region L loop):";
        for (k, n) in kinds.iter().take(16) {
            write!(out, " [{k}] {n},").unwrap();
        }
        out
    }

    /// The kinds of the feature edges of `u`: their count and per edge its kind letters.
    fn junction_kind(e: &Engine<'_>, u: u32, fan: &[u32]) -> String {
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for &t in fan {
            for &w in &e.tris[t as usize] {
                let x = e.weld[w as usize];
                if x != u {
                    pairs.push((x, t));
                }
            }
        }
        pairs.sort_unstable();
        let mut edges: Vec<String> = Vec::new();
        for g in pairs.chunk_by(|a, b| a.0 == b.0) {
            let sig = e.edge_sig(u, g[0].0, g, LoopMode::Active);
            if !sig.is_feature() {
                continue;
            }
            let mut k = String::new();
            if sig.border {
                k.push('B');
            }
            if sig.seam {
                k.push('S');
            }
            if sig.parts.0 != sig.parts.1 {
                k.push('P');
            }
            if sig.other_region != NO_REGION {
                k.push('R');
            }
            if sig.loop_id != NO_LOOP {
                k.push('L');
            }
            edges.push(k);
        }
        edges.sort();
        format!("{} {}", edges.len(), edges.join("/"))
    }
}
