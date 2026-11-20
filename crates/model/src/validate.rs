//! Template validation `[AC-WP06-1]`: watertight and manifold meshes, region coverage, UV overlap,
//! texel density, landmark validity, skin weight normalisation and the LOD error report.
//!
//! [`validate`] never fails and never panics on a [`Template`]: it measures and returns a
//! [`ValidationReport`] with one sub-report per check and a list of [`Issue`]s. A template *passes*
//! when no issue is an [`Severity::Error`]; warnings (for example fewer landmarks than `[MOD-06]`
//! asks for, or faces below the target texel density) are for the generator's authors to read. The
//! report prints as plain text (`Display`).
//!
//! Structural consistency (indices in range, sorted arrays, finite values) is not re-checked here:
//! [`Template::new`] guarantees it.

use std::fmt;

use humanart_math::geom::closest_point_on_triangle;
use humanart_math::{F16, Vec2, Vec3};

use crate::landmark::LandmarkGroup;
use crate::lod::LodLevel;
use crate::mesh::Mesh;
use crate::template::Template;
use crate::topology::{Csr, Topology};

/// At most this many issues are listed per check; the sub-reports carry the full counts.
pub const MAX_ISSUES_PER_CHECK: usize = 64;
/// Tolerance on weight sums (skin weights, soft region weights, barycentric coordinates).
pub const WEIGHT_TOLERANCE: f32 = 1e-4;
/// `[MOD-06]`: the least number of face landmarks.
pub const MIN_FACE_LANDMARKS: usize = 98;
/// `[MOD-06]`: the least number of body landmarks.
pub const MIN_BODY_LANDMARKS: usize = 40;
/// UV distances below this (in tile units) count as touching, not overlapping.
pub const UV_EPSILON: f32 = 1e-6;
/// At most this many LOD0 vertices are measured against each LOD surface (evenly strided).
pub const MAX_LOD_SAMPLES: usize = 50_000;

/// How serious an issue is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Worth a look; does not fail validation.
    Warning,
    /// Fails validation.
    Error,
}

/// Which check found an issue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Check {
    /// A part meant to be closed has boundary edges.
    Watertight,
    /// Non-manifold edges or vertices, or inconsistent face orientation.
    Manifold,
    /// Faces with (almost) zero area.
    DegenerateFace,
    /// A leaf region without vertices, or duplicates of one vertex in different regions.
    RegionCoverage,
    /// Soft region weights that do not sum to 1.
    SoftRegionWeights,
    /// UVs outside the tile, or one vertex used by faces of different tiles.
    UvRange,
    /// Faces whose UVs are mirrored or collapsed.
    UvFlipped,
    /// Faces whose UVs overlap other faces of the same tile.
    UvOverlap,
    /// Faces below the tile's target texel density.
    TexelDensity,
    /// A landmark that is not a valid barycentric point on one face, or too few landmarks.
    Landmark,
    /// Skin weights that do not sum to 1, are not sorted, or differ across a seam.
    SkinWeights,
    /// Delta entries outside their bounding region, or deltas that differ across a seam.
    Deltas,
    /// A LOD level whose measured error exceeds its declared error.
    LodError,
}

impl Check {
    /// A short name for reports.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Watertight => "watertight",
            Self::Manifold => "manifold",
            Self::DegenerateFace => "degenerate face",
            Self::RegionCoverage => "region coverage",
            Self::SoftRegionWeights => "soft region weights",
            Self::UvRange => "UV range",
            Self::UvFlipped => "UV flipped",
            Self::UvOverlap => "UV overlap",
            Self::TexelDensity => "texel density",
            Self::Landmark => "landmark",
            Self::SkinWeights => "skin weights",
            Self::Deltas => "deltas",
            Self::LodError => "LOD error",
        }
    }
}

/// One finding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Issue {
    /// How serious it is.
    pub severity: Severity,
    /// Which check found it.
    pub check: Check,
    /// The item it is about (a face, vertex, part, landmark, tile, component or LOD level, as the
    /// message says), if any.
    pub index: Option<u64>,
    /// What is wrong.
    pub what: &'static str,
}

/// Topology of one mesh part.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PartReport {
    /// The part's name.
    pub name: String,
    /// Whether the part is meant to be closed.
    pub closed: bool,
    /// Faces.
    pub faces: usize,
    /// Welded vertices (render duplicates counted once).
    pub vertices: usize,
    /// Edges.
    pub edges: usize,
    /// Edges with one face.
    pub boundary_edges: usize,
    /// Edges with more than two faces.
    pub non_manifold_edges: usize,
    /// Two-face edges traversed in the same direction by both faces (flipped orientation).
    pub inconsistent_edges: usize,
    /// Vertices whose faces form more than one fan (bow ties).
    pub non_manifold_vertices: usize,
    /// Faces with an area below 1e-12 m².
    pub degenerate_faces: usize,
    /// Connected components.
    pub components: usize,
    /// Euler characteristic V − E + F (2 per closed genus-0 component).
    pub euler: i64,
}

impl PartReport {
    /// No boundary and no non-manifold edge.
    #[must_use]
    pub const fn watertight(&self) -> bool {
        self.boundary_edges == 0 && self.non_manifold_edges == 0
    }

    /// No non-manifold edge or vertex and a consistent orientation.
    #[must_use]
    pub const fn manifold(&self) -> bool {
        self.non_manifold_edges == 0 && self.non_manifold_vertices == 0 && self.inconsistent_edges == 0
    }
}

/// Region coverage.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RegionReport {
    /// Vertices per region (directly, not counting the subtree).
    pub vertex_counts: Vec<u32>,
    /// Vertices with a region: always every vertex (`[MOD-03]`, guaranteed by the data model).
    pub covered_vertices: usize,
    /// Leaf regions without vertices.
    pub empty_leaves: usize,
    /// Duplicate groups whose vertices are in different regions.
    pub split_duplicates: usize,
    /// Largest deviation of a vertex's soft weights from summing to 1 (0 without soft weights).
    pub soft_max_deviation: f32,
}

/// UV layout and texel density of one tile.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TileReport {
    /// The tile's name.
    pub name: String,
    /// Faces in the tile.
    pub faces: usize,
    /// Share of the tile covered by faces (sum of UV areas; above 1 means overlap).
    pub coverage: f32,
    /// Faces with a UV outside `[0, 1]`.
    pub out_of_range_faces: usize,
    /// Faces with mirrored or collapsed UVs.
    pub flipped_faces: usize,
    /// Pairs of triangles whose UVs overlap.
    pub overlapping_pairs: usize,
    /// The tile's target, texels per millimetre.
    pub target_texels_per_mm: f32,
    /// Smallest face texel density, texels per millimetre.
    pub density_min: f32,
    /// Median face texel density.
    pub density_median: f32,
    /// Largest face texel density.
    pub density_max: f32,
    /// Surface-area-weighted mean density.
    pub density_mean: f32,
    /// Share of the tile's surface area below the target density.
    pub area_below_target: f32,
}

/// Landmark validity.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LandmarkReport {
    /// Face landmarks.
    pub face: usize,
    /// Body landmarks.
    pub body: usize,
    /// Landmarks that are not a valid barycentric point on one face.
    pub invalid: usize,
}

impl LandmarkReport {
    /// Whether the counts meet `[MOD-06]` (≥ 98 face, ≥ 40 body).
    #[must_use]
    pub const fn meets_mod06(&self) -> bool {
        self.face >= MIN_FACE_LANDMARKS && self.body >= MIN_BODY_LANDMARKS
    }
}

/// Skin weight normalisation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WeightReport {
    /// Influences per vertex.
    pub influences: usize,
    /// Largest deviation of a vertex's weights from summing to 1.
    pub max_sum_deviation: f32,
    /// Vertices whose weights deviate by more than [`WEIGHT_TOLERANCE`].
    pub unnormalised: usize,
    /// Vertices whose weights are not in decreasing order.
    pub unsorted: usize,
    /// Duplicate groups whose vertices have different weights.
    pub seam_mismatches: usize,
}

/// Delta bases.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeltaReport {
    /// Components.
    pub components: usize,
    /// Entries over all components.
    pub entries: usize,
    /// Components from scan statistics (0 in no-data operation `[ID-07]`).
    pub scan_derived: usize,
    /// Entries whose vertex lies outside the component's bounding region.
    pub outside_region: usize,
    /// Entries on a duplicated vertex whose duplicates have a different displacement.
    pub seam_mismatches: usize,
    /// Largest displacement length, metres per unit weight.
    pub max_displacement: f32,
}

/// The error of one LOD level against LOD0.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LodReport {
    /// 1 for LOD1, ...
    pub level: usize,
    /// Vertices of the level.
    pub vertices: usize,
    /// Triangles of the level.
    pub triangles: usize,
    /// Triangles relative to LOD0.
    pub triangle_ratio: f32,
    /// The error its generator declared, metres.
    pub declared_error: f32,
    /// Largest distance between a LOD vertex and its interpolated source position, metres.
    pub source_error: f32,
    /// Largest distance from a (sampled) LOD0 vertex to the level's surface, metres.
    pub surface_error: f32,
    /// LOD0 vertices measured.
    pub samples: usize,
}

/// The full validation report.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ValidationReport {
    /// One report per mesh part of LOD0.
    pub parts: Vec<PartReport>,
    /// Region coverage.
    pub regions: RegionReport,
    /// One report per UV tile.
    pub tiles: Vec<TileReport>,
    /// Landmarks.
    pub landmarks: LandmarkReport,
    /// Skin weights.
    pub weights: WeightReport,
    /// Delta bases.
    pub deltas: DeltaReport,
    /// One report per LOD level.
    pub lods: Vec<LodReport>,
    /// Findings, at most [`MAX_ISSUES_PER_CHECK`] per check.
    pub issues: Vec<Issue>,
}

impl ValidationReport {
    /// Whether no issue is an error.
    #[must_use]
    pub fn passed(&self) -> bool {
        !self.issues.iter().any(|i| i.severity == Severity::Error)
    }

    /// How many issues of a check were listed.
    #[must_use]
    pub fn count(&self, check: Check) -> usize {
        self.issues.iter().filter(|i| i.check == check).count()
    }

    fn push(&mut self, severity: Severity, check: Check, index: Option<usize>, what: &'static str) {
        if self.count(check) < MAX_ISSUES_PER_CHECK {
            self.issues.push(Issue { severity, check, index: index.map(|i| i as u64), what });
        }
    }
}

/// Validates `template` and returns the report.
#[must_use]
pub fn validate(template: &Template) -> ValidationReport {
    let mesh = template.mesh();
    let triangles = mesh.triangulate();
    let topo = Topology::new(mesh, &triangles);
    let mut report = ValidationReport::default();
    check_parts(mesh, &topo, &mut report);
    check_regions(template, &topo, &mut report);
    check_uvs(template, &triangles, &mut report);
    check_landmarks(template, &topo, &mut report);
    check_weights(template, &topo, &mut report);
    check_deltas(template, &topo, &mut report);
    check_lods(template, triangles.len(), &mut report);
    report
}

/// A union-find over `n` items.
struct Dsu(Vec<u32>);

impl Dsu {
    fn new(n: usize) -> Self {
        Self((0..n as u32).collect())
    }

    fn find(&mut self, mut a: u32) -> u32 {
        while self.0[a as usize] != a {
            let up = self.0[self.0[a as usize] as usize];
            self.0[a as usize] = up;
            a = up;
        }
        a
    }

    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.0[ra.max(rb) as usize] = ra.min(rb);
        }
    }
}

fn check_parts(mesh: &Mesh, topo: &Topology, report: &mut ValidationReport) {
    let keys = mesh.position_keys();
    let offsets = mesh.face_offsets();
    let corners = mesh.face_vertices();
    // Traversal directions per edge: [forward (low key to high), backward].
    let mut dirs = vec![[0_u32; 2]; topo.edges.len()];
    for w in offsets.windows(2) {
        let (a, b) = (w[0] as usize, w[1] as usize);
        for c in a..b {
            let next = if c + 1 == b { a } else { c + 1 };
            let forward = keys[corners[c] as usize] < keys[corners[next] as usize];
            dirs[topo.corner_edges[c] as usize][usize::from(!forward)] += 1;
        }
    }
    // Faces around each weld key.
    let mut pairs = Vec::with_capacity(corners.len());
    for (f, w) in offsets.windows(2).enumerate() {
        for &c in &corners[w[0] as usize..w[1] as usize] {
            pairs.push((keys[c as usize], f as u32));
        }
    }
    let key_faces = Csr::from_pairs(mesh.vertex_count(), &pairs);

    let mut edge_stamp = vec![u32::MAX; topo.edges.len()];
    let mut key_stamp = vec![u32::MAX; mesh.vertex_count()];
    for (p, part) in mesh.parts().iter().enumerate() {
        let stamp = p as u32;
        let faces = part.faces();
        let mut r =
            PartReport { name: part.name.clone(), closed: part.closed, faces: faces.len(), ..PartReport::default() };
        let mut dsu = Dsu::new(faces.len());
        let mut part_keys: Vec<u32> = Vec::new();
        for f in faces.clone() {
            if mesh.face_vector_area(f).length() < 1e-12 {
                r.degenerate_faces += 1;
                report.push(Severity::Warning, Check::DegenerateFace, Some(f), "face has (almost) zero area");
            }
            for c in offsets[f] as usize..offsets[f + 1] as usize {
                let e = topo.corner_edges[c] as usize;
                if edge_stamp[e] != stamp {
                    edge_stamp[e] = stamp;
                    r.edges += 1;
                    let around = topo.edge_faces.row(e);
                    match around.len() {
                        1 => r.boundary_edges += 1,
                        2 if dirs[e] != [1, 1] => r.inconsistent_edges += 1,
                        2 => {}
                        _ => r.non_manifold_edges += 1,
                    }
                    let local: Vec<u32> = around
                        .iter()
                        .filter(|&&g| faces.contains(&(g as usize)))
                        .map(|&g| g - part.first_face)
                        .collect();
                    for pair in local.windows(2) {
                        dsu.union(pair[0], pair[1]);
                    }
                }
                let k = keys[corners[c] as usize];
                if key_stamp[k as usize] != stamp {
                    key_stamp[k as usize] = stamp;
                    part_keys.push(k);
                }
            }
        }
        r.vertices = part_keys.len();
        r.components = (0..faces.len() as u32).filter(|&i| dsu.find(i) == i).count();
        r.euler = r.vertices as i64 - r.edges as i64 + r.faces as i64;
        for &k in &part_keys {
            if fans_around(k, key_faces.row(k as usize), mesh, topo) > 1 {
                r.non_manifold_vertices += 1;
                report.push(Severity::Error, Check::Manifold, Some(k as usize), "vertex joins separate fans of faces");
            }
        }
        if r.closed && !r.watertight() {
            report.push(Severity::Error, Check::Watertight, Some(p), "closed part has boundary or non-manifold edges");
        }
        if r.non_manifold_edges > 0 {
            report.push(Severity::Error, Check::Manifold, Some(p), "part has edges with more than two faces");
        }
        if r.inconsistent_edges > 0 {
            report.push(Severity::Error, Check::Manifold, Some(p), "part has faces with flipped orientation");
        }
        report.parts.push(r);
    }
}

/// How many fans the faces around weld key `k` form (faces linked through edges incident to `k`).
fn fans_around(k: u32, faces: &[u32], mesh: &Mesh, topo: &Topology) -> usize {
    let keys = mesh.position_keys();
    let offsets = mesh.face_offsets();
    let mut dsu = Dsu::new(faces.len());
    let mut seen: Vec<(u32, u32)> = Vec::with_capacity(2 * faces.len());
    for (i, &f) in faces.iter().enumerate() {
        let (a, b) = (offsets[f as usize] as usize, offsets[f as usize + 1] as usize);
        let Some(c) = (a..b).find(|&c| keys[mesh.face_vertices()[c] as usize] == k) else { continue };
        let prev = if c == a { b - 1 } else { c - 1 };
        for e in [topo.corner_edges[c], topo.corner_edges[prev]] {
            match seen.iter().find(|s| s.0 == e) {
                Some(&(_, j)) => dsu.union(i as u32, j),
                None => seen.push((e, i as u32)),
            }
        }
    }
    (0..faces.len() as u32).filter(|&i| dsu.find(i) == i).count()
}

fn check_regions(t: &Template, topo: &Topology, report: &mut ValidationReport) {
    let map = t.regions();
    let mut r = RegionReport {
        vertex_counts: map.vertex_counts(),
        covered_vertices: map.vertex_regions.len(),
        ..RegionReport::default()
    };
    for (i, &n) in r.vertex_counts.iter().enumerate() {
        if n == 0 && !map.has_children(i as u16) {
            r.empty_leaves += 1;
            report.push(Severity::Warning, Check::RegionCoverage, Some(i), "leaf region has no vertices");
        }
    }
    for group in topo.duplicate_groups.iter() {
        let first = map.vertex_regions[group[0] as usize];
        if group.iter().any(|&v| map.vertex_regions[v as usize] != first) {
            r.split_duplicates += 1;
            report.push(
                Severity::Warning,
                Check::RegionCoverage,
                Some(group[0] as usize),
                "duplicates of one vertex lie in different regions",
            );
        }
    }
    if let Some(soft) = &map.soft {
        for v in 0..map.vertex_regions.len() {
            let (_, weights) = soft.vertex(v);
            if weights.is_empty() {
                continue;
            }
            let dev = (weights.iter().sum::<f32>() - 1.0).abs();
            r.soft_max_deviation = r.soft_max_deviation.max(dev);
            if dev > WEIGHT_TOLERANCE {
                report.push(Severity::Error, Check::SoftRegionWeights, Some(v), "soft region weights do not sum to 1");
            }
        }
    }
    report.regions = r;
}

fn check_uvs(t: &Template, triangles: &crate::mesh::Triangulation, report: &mut ValidationReport) {
    // Tile indices are below MAX_TILES, so the two top values are free to mean "unused" and
    // "already reported".
    const UNUSED: u16 = u16::MAX;
    const REPORTED: u16 = u16::MAX - 1;
    let mesh = t.mesh();
    let uvs = mesh.uvs();
    let tiles_of_faces = mesh.face_tiles();
    // A vertex must stay in one tile (UVs are tile-local).
    let mut vertex_tile = vec![UNUSED; mesh.vertex_count()];
    for (f, face) in mesh.faces().enumerate() {
        let tile = tiles_of_faces[f];
        for &v in face {
            let slot = &mut vertex_tile[v as usize];
            if *slot == UNUSED {
                *slot = tile;
            } else if *slot != tile && *slot != REPORTED {
                *slot = REPORTED;
                report.push(
                    Severity::Error,
                    Check::UvRange,
                    Some(v as usize),
                    "vertex is used by faces of different UV tiles",
                );
            }
        }
    }
    let tri_pairs: Vec<(u32, u32)> = triangles
        .triangle_faces
        .iter()
        .enumerate()
        .map(|(i, &f)| (u32::from(tiles_of_faces[f as usize]), i as u32))
        .collect();
    let tile_tris = Csr::from_pairs(t.tiles().len(), &tri_pairs);
    for (ti, tile) in t.tiles().iter().enumerate() {
        let mut r = TileReport {
            name: tile.name.clone(),
            target_texels_per_mm: tile.target_texels_per_mm,
            ..TileReport::default()
        };
        let res2 = f64::from(tile.resolution) * f64::from(tile.resolution);
        let mut densities: Vec<(f32, f32)> = Vec::new();
        let mut coverage = 0.0_f64;
        for f in (0..mesh.face_count()).filter(|&f| tiles_of_faces[f] as usize == ti) {
            r.faces += 1;
            let face = mesh.face(f);
            if face.iter().any(|&v| {
                let uv = uvs[v as usize];
                !(-UV_EPSILON..=1.0 + UV_EPSILON).contains(&uv.x) || !(-UV_EPSILON..=1.0 + UV_EPSILON).contains(&uv.y)
            }) {
                r.out_of_range_faces += 1;
                report.push(Severity::Error, Check::UvRange, Some(f), "face has a UV outside its tile");
            }
            let uv_area = mesh.face_uv_area(f);
            coverage += f64::from(uv_area.abs());
            if uv_area <= UV_EPSILON * UV_EPSILON {
                r.flipped_faces += 1;
                report.push(Severity::Error, Check::UvFlipped, Some(f), "face UVs are mirrored or collapsed");
            }
            let area_mm2 = f64::from(mesh.face_vector_area(f).length()) * 1e6;
            if area_mm2 > 1e-8 {
                let density = (f64::from(uv_area.abs()) * res2 / area_mm2).sqrt() as f32;
                densities.push((density, area_mm2 as f32));
            }
        }
        r.coverage = coverage as f32;
        r.overlapping_pairs = count_uv_overlaps(mesh, triangles, tile_tris.row(ti), report);
        density_stats(&mut r, &mut densities);
        if r.area_below_target > 0.0 {
            report.push(
                Severity::Warning,
                Check::TexelDensity,
                Some(ti),
                "part of the tile is below its target texel density",
            );
        }
        report.tiles.push(r);
    }
}

fn density_stats(r: &mut TileReport, densities: &mut [(f32, f32)]) {
    if densities.is_empty() {
        return;
    }
    densities.sort_by(|a, b| a.0.total_cmp(&b.0));
    r.density_min = densities[0].0;
    r.density_max = densities[densities.len() - 1].0;
    r.density_median = densities[densities.len() / 2].0;
    let total: f64 = densities.iter().map(|d| f64::from(d.1)).sum();
    if total > 0.0 {
        let weighted: f64 = densities.iter().map(|d| f64::from(d.0) * f64::from(d.1)).sum();
        r.density_mean = (weighted / total) as f32;
        // A fold from +0.0: an empty float `sum()` is -0.0, which would print as "-0.0%".
        let below = densities.iter().filter(|d| d.0 < r.target_texels_per_mm).fold(0.0_f64, |s, d| s + f64::from(d.1));
        r.area_below_target = (below / total) as f32;
    }
}

/// Counts overlapping pairs among the triangles `tris` (all in one tile) with a uniform grid.
fn count_uv_overlaps(
    mesh: &Mesh,
    triangles: &crate::mesh::Triangulation,
    tris: &[u32],
    report: &mut ValidationReport,
) -> usize {
    let uvs = mesh.uvs();
    let corner_uvs = |t: u32| -> [Vec2; 3] {
        let i = 3 * t as usize;
        [0, 1, 2].map(|k| uvs[triangles.indices[i + k] as usize])
    };
    let mut lo = Vec2::splat(f32::INFINITY);
    let mut hi = Vec2::splat(f32::NEG_INFINITY);
    let mut items: Vec<(u32, [Vec2; 3])> = Vec::with_capacity(tris.len());
    for &t in tris {
        let c = corner_uvs(t);
        // Collapsed triangles cannot overlap anything; they are reported as flipped faces.
        if (c[1] - c[0]).perp_dot(c[2] - c[0]).abs() <= UV_EPSILON * UV_EPSILON {
            continue;
        }
        for p in c {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        items.push((t, c));
    }
    if items.len() < 2 {
        return 0;
    }
    let extent = (hi - lo).max(Vec2::splat(1e-6));
    let side = ((items.len() as f32).sqrt().ceil() as usize).clamp(1, 1024);
    let cell = Vec2::new(extent.x / side as f32, extent.y / side as f32);
    let cell_of = |p: Vec2| -> (usize, usize) {
        let q = (p - lo) / cell;
        ((q.x.max(0.0) as usize).min(side - 1), (q.y.max(0.0) as usize).min(side - 1))
    };
    let bounds = |c: &[Vec2; 3]| -> ((usize, usize), (usize, usize)) {
        let (mn, mx) = (c[0].min(c[1]).min(c[2]), c[0].max(c[1]).max(c[2]));
        (cell_of(mn), cell_of(mx))
    };
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for (i, (_, c)) in items.iter().enumerate() {
        let ((x0, y0), (x1, y1)) = bounds(c);
        for y in y0..=y1 {
            for x in x0..=x1 {
                pairs.push(((y * side + x) as u32, i as u32));
            }
        }
    }
    let grid = Csr::from_pairs(side * side, &pairs);
    let mut overlaps = 0;
    for (cell_index, members) in grid.iter().enumerate() {
        for (a, &i) in members.iter().enumerate() {
            for &j in &members[a + 1..] {
                let (ti, ci) = &items[i as usize];
                let (tj, cj) = &items[j as usize];
                // Test each pair once: in the first cell both cover.
                let ((ax0, ay0), _) = bounds(ci);
                let ((bx0, by0), _) = bounds(cj);
                if ax0.max(bx0) + ay0.max(by0) * side != cell_index {
                    continue;
                }
                if triangles.triangle_faces[*ti as usize] == triangles.triangle_faces[*tj as usize] {
                    continue;
                }
                if triangles_overlap(ci, cj) {
                    overlaps += 1;
                    report.push(
                        Severity::Error,
                        Check::UvOverlap,
                        Some(triangles.triangle_faces[*ti as usize] as usize),
                        "face UVs overlap another face",
                    );
                }
            }
        }
    }
    overlaps
}

/// Whether two 2D triangles' interiors overlap by more than [`UV_EPSILON`] (separating axes).
fn triangles_overlap(a: &[Vec2; 3], b: &[Vec2; 3]) -> bool {
    for tri in [a, b] {
        for i in 0..3 {
            let e = tri[(i + 1) % 3] - tri[i];
            let Some(n) = Vec2::new(-e.y, e.x).try_normalize() else { continue };
            let project = |t: &[Vec2; 3]| {
                let d = [n.dot(t[0]), n.dot(t[1]), n.dot(t[2])];
                (d[0].min(d[1]).min(d[2]), d[0].max(d[1]).max(d[2]))
            };
            let ((a0, a1), (b0, b1)) = (project(a), project(b));
            if a1 <= b0 + UV_EPSILON || b1 <= a0 + UV_EPSILON {
                return false;
            }
        }
    }
    true
}

fn check_landmarks(t: &Template, topo: &Topology, report: &mut ValidationReport) {
    let mesh = t.mesh();
    let keys = mesh.position_keys();
    let positions = mesh.positions();
    let mut r = LandmarkReport::default();
    for (i, l) in t.landmarks().iter().enumerate() {
        match l.group {
            LandmarkGroup::Face => r.face += 1,
            LandmarkGroup::Body => r.body += 1,
        }
        let lk = l.vertices.map(|v| keys[v as usize]);
        let distinct = lk[0] != lk[1] && lk[1] != lk[2] && lk[0] != lk[2];
        let on_face = distinct
            && topo.vertex_faces.row(l.vertices[0] as usize).iter().any(|&f| {
                let face = mesh.face(f as usize);
                lk.iter().all(|k| face.iter().any(|&c| keys[c as usize] == *k))
            });
        let weights_ok = l.weights.iter().all(|w| (-WEIGHT_TOLERANCE..=1.0 + WEIGHT_TOLERANCE).contains(w))
            && (l.weights.iter().sum::<f32>() - 1.0).abs() <= WEIGHT_TOLERANCE;
        let [a, b, c] = l.vertices.map(|v| positions[v as usize]);
        let area_ok = (b - a).cross(c - a).length() > 1e-12;
        if !(on_face && weights_ok && area_ok) {
            r.invalid += 1;
            let what = if !on_face {
                "landmark corners are not on one face"
            } else if !weights_ok {
                "landmark weights are not a barycentric triple"
            } else {
                "landmark triangle is degenerate"
            };
            report.push(Severity::Error, Check::Landmark, Some(i), what);
        }
    }
    if !r.meets_mod06() {
        report.push(
            Severity::Warning,
            Check::Landmark,
            None,
            "fewer landmarks than MOD-06 asks for (98 face, 40 body)",
        );
    }
    report.landmarks = r;
}

fn check_weights(t: &Template, topo: &Topology, report: &mut ValidationReport) {
    let skin = t.skin();
    let mut r = WeightReport { influences: skin.influences as usize, ..WeightReport::default() };
    if skin.influences == 0 {
        report.weights = r;
        return;
    }
    for v in 0..t.mesh().vertex_count() {
        let (_, w) = skin.vertex(v);
        let dev = (w.iter().sum::<f32>() - 1.0).abs();
        r.max_sum_deviation = r.max_sum_deviation.max(dev);
        if dev > WEIGHT_TOLERANCE {
            r.unnormalised += 1;
            report.push(Severity::Error, Check::SkinWeights, Some(v), "skin weights do not sum to 1");
        }
        if w.windows(2).any(|p| p[0] < p[1]) {
            r.unsorted += 1;
            report.push(Severity::Error, Check::SkinWeights, Some(v), "skin weights are not in decreasing order");
        }
    }
    for group in topo.duplicate_groups.iter() {
        let first = skin.vertex(group[0] as usize);
        if group.iter().any(|&v| skin.vertex(v as usize) != first) {
            r.seam_mismatches += 1;
            report.push(
                Severity::Error,
                Check::SkinWeights,
                Some(group[0] as usize),
                "duplicates of one vertex have different skin weights",
            );
        }
    }
    report.weights = r;
}

fn check_deltas(t: &Template, topo: &Topology, report: &mut ValidationReport) {
    let d = t.deltas();
    let regions = t.regions();
    let mut group_of = vec![u32::MAX; t.mesh().vertex_count()];
    for (g, members) in topo.duplicate_groups.iter().enumerate() {
        for &v in members {
            group_of[v as usize] = g as u32;
        }
    }
    let mut r = DeltaReport {
        components: d.len(),
        entries: d.all_indices().len(),
        scan_derived: t.scan_component_count(),
        ..DeltaReport::default()
    };
    for (c, comp) in d.components().iter().enumerate() {
        let (indices, deltas) = (d.indices(c), d.deltas(c));
        let lookup = |v: u32| -> Option<[F16; 3]> { indices.binary_search(&v).ok().map(|i| deltas[i]) };
        let mut outside = false;
        let mut mismatch = false;
        for (&v, &x) in indices.iter().zip(deltas) {
            r.max_displacement = r.max_displacement.max(crate::delta::DeltaBases::delta_f32(x).length());
            if !regions.is_within(regions.vertex_regions[v as usize], comp.region) {
                r.outside_region += 1;
                outside = true;
            }
            let g = group_of[v as usize];
            if g != u32::MAX {
                let same = |y: Option<[F16; 3]>| y.is_some_and(|y| y.map(F16::to_bits) == x.map(F16::to_bits));
                if !topo.duplicate_groups.row(g as usize).iter().all(|&u| same(lookup(u))) {
                    r.seam_mismatches += 1;
                    mismatch = true;
                }
            }
        }
        if outside {
            report.push(
                Severity::Error,
                Check::Deltas,
                Some(c),
                "component moves vertices outside its bounding region",
            );
        }
        if mismatch {
            report.push(
                Severity::Error,
                Check::Deltas,
                Some(c),
                "component moves duplicates of one vertex differently",
            );
        }
    }
    report.deltas = r;
}

fn check_lods(t: &Template, base_triangles: usize, report: &mut ValidationReport) {
    let base = t.mesh();
    // The welded LOD0 vertices, evenly strided down to the sample budget.
    let welded: Vec<Vec3> = (0..base.vertex_count())
        .filter(|&v| base.position_keys()[v] as usize == v)
        .map(|v| base.positions()[v])
        .collect();
    let stride = welded.len().div_ceil(MAX_LOD_SAMPLES).max(1);
    for (i, level) in t.lods().iter().enumerate() {
        let tris = level.mesh.triangulate();
        let source_error = level.transfer_positions(base.positions()).map_or(f32::INFINITY, |p| {
            p.iter().zip(level.mesh.positions()).map(|(a, b)| a.distance(*b)).fold(0.0, f32::max)
        });
        let grid = TriangleGrid::new(level, &tris.indices);
        let mut surface_error = 0.0_f32;
        let mut samples = 0;
        for &p in welded.iter().step_by(stride) {
            surface_error = surface_error.max(grid.distance(p));
            samples += 1;
        }
        let r = LodReport {
            level: i + 1,
            vertices: level.mesh.vertex_count(),
            triangles: tris.len(),
            triangle_ratio: if base_triangles == 0 { 0.0 } else { tris.len() as f32 / base_triangles as f32 },
            declared_error: level.declared_error,
            source_error,
            surface_error,
            samples,
        };
        let allowed = level.declared_error * (1.0 + 1e-3) + 1e-6;
        if r.surface_error.is_nan() || r.surface_error > allowed {
            report.push(
                Severity::Error,
                Check::LodError,
                Some(i + 1),
                "LOD surface is farther from LOD0 than its declared error",
            );
        }
        if r.source_error.is_nan() || r.source_error > allowed {
            report.push(
                Severity::Error,
                Check::LodError,
                Some(i + 1),
                "LOD vertices are farther from their sources than the declared error",
            );
        }
        report.lods.push(r);
    }
}

/// Triangles of a LOD level in a uniform grid, for nearest-surface distance queries.
struct TriangleGrid {
    tris: Vec<[Vec3; 3]>,
    origin: Vec3,
    cell: f32,
    dims: [usize; 3],
    cells: Csr,
}

impl TriangleGrid {
    fn new(level: &LodLevel, indices: &[u32]) -> Self {
        let p = level.mesh.positions();
        let tris: Vec<[Vec3; 3]> = indices.as_chunks::<3>().0.iter().map(|t| t.map(|v| p[v as usize])).collect();
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for &v in p {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if tris.is_empty() {
            return Self { tris, origin: Vec3::ZERO, cell: 1.0, dims: [1, 1, 1], cells: Csr::default() };
        }
        let extent = (hi - lo).max(Vec3::splat(1e-6));
        let cell =
            (extent.x * extent.y * extent.z / tris.len() as f32).cbrt().max(extent.max_element() / 128.0).max(1e-6);
        let dims = [extent.x, extent.y, extent.z].map(|e| ((e / cell).ceil() as usize).clamp(1, 128));
        let mut grid = Self { tris, origin: lo, cell, dims, cells: Csr::default() };
        let mut pairs = Vec::new();
        for (i, t) in grid.tris.iter().enumerate() {
            let a = grid.cell_of(t[0].min(t[1]).min(t[2]));
            let b = grid.cell_of(t[0].max(t[1]).max(t[2]));
            for z in a[2]..=b[2] {
                for y in a[1]..=b[1] {
                    for x in a[0]..=b[0] {
                        pairs.push((grid.index([x, y, z]) as u32, i as u32));
                    }
                }
            }
        }
        grid.cells = Csr::from_pairs(dims[0] * dims[1] * dims[2], &pairs);
        grid
    }

    fn cell_of(&self, p: Vec3) -> [usize; 3] {
        let q = (p - self.origin) / self.cell;
        let q = [q.x, q.y, q.z];
        [0, 1, 2].map(|i| {
            let c = if q[i].is_nan() { 0.0 } else { q[i].max(0.0) };
            (c as usize).min(self.dims[i] - 1)
        })
    }

    fn index(&self, c: [usize; 3]) -> usize {
        (c[2] * self.dims[1] + c[1]) * self.dims[0] + c[0]
    }

    /// Distance from `p` to the nearest triangle (infinite when there are none).
    fn distance(&self, p: Vec3) -> f32 {
        if self.tris.is_empty() {
            return f32::INFINITY;
        }
        let c = self.cell_of(p);
        let mut best = f32::INFINITY;
        let max_ring = self.dims.iter().copied().max().unwrap_or(1);
        for ring in 0..=max_ring {
            let lo = c.map(|x| x.saturating_sub(ring));
            for z in lo[2]..=(c[2] + ring).min(self.dims[2] - 1) {
                for y in lo[1]..=(c[1] + ring).min(self.dims[1] - 1) {
                    for x in lo[0]..=(c[0] + ring).min(self.dims[0] - 1) {
                        let chebyshev = x.abs_diff(c[0]).max(y.abs_diff(c[1])).max(z.abs_diff(c[2]));
                        if chebyshev != ring {
                            continue;
                        }
                        for &t in self.cells.row(self.index([x, y, z])) {
                            let [a, b, d] = self.tris[t as usize];
                            best = best.min(p.distance(closest_point_on_triangle(p, a, b, d).0));
                        }
                    }
                }
            }
            // Cells beyond this ring are at least `ring` cells away from p's (clamped) cell.
            if best <= ring as f32 * self.cell {
                break;
            }
        }
        best
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "template validation: {}", if self.passed() { "passed" } else { "FAILED" })?;
        for p in &self.parts {
            writeln!(
                f,
                "  part {}: {} faces, {} vertices, {} edges, {} boundary, {} non-manifold edges, {} non-manifold vertices, {} flipped, {} components, euler {}{}",
                p.name,
                p.faces,
                p.vertices,
                p.edges,
                p.boundary_edges,
                p.non_manifold_edges,
                p.non_manifold_vertices,
                p.inconsistent_edges,
                p.components,
                p.euler,
                if p.closed { if p.watertight() { ", watertight" } else { ", NOT watertight" } } else { ", open" },
            )?;
        }
        let r = &self.regions;
        writeln!(
            f,
            "  regions: {} regions, {} vertices covered, {} empty leaves, {} split duplicates, soft deviation {:e}",
            r.vertex_counts.len(),
            r.covered_vertices,
            r.empty_leaves,
            r.split_duplicates,
            r.soft_max_deviation
        )?;
        for t in &self.tiles {
            writeln!(
                f,
                "  tile {}: {} faces, coverage {:.3}, {} out of range, {} flipped, {} overlapping pairs, density min {:.3} median {:.3} mean {:.3} max {:.3} texels/mm (target {}), {:.1}% of area below target",
                t.name,
                t.faces,
                t.coverage,
                t.out_of_range_faces,
                t.flipped_faces,
                t.overlapping_pairs,
                t.density_min,
                t.density_median,
                t.density_mean,
                t.density_max,
                t.target_texels_per_mm,
                100.0 * t.area_below_target
            )?;
        }
        let l = &self.landmarks;
        writeln!(f, "  landmarks: {} face, {} body, {} invalid", l.face, l.body, l.invalid)?;
        let w = &self.weights;
        writeln!(
            f,
            "  skin weights: {} influences, max sum deviation {:e}, {} unnormalised, {} unsorted, {} seam mismatches",
            w.influences, w.max_sum_deviation, w.unnormalised, w.unsorted, w.seam_mismatches
        )?;
        let d = &self.deltas;
        writeln!(
            f,
            "  deltas: {} components ({} from scans), {} entries, {} outside region, {} seam mismatches, max {:.4} m",
            d.components, d.scan_derived, d.entries, d.outside_region, d.seam_mismatches, d.max_displacement
        )?;
        for l in &self.lods {
            writeln!(
                f,
                "  LOD{}: {} vertices, {} triangles ({:.3} of LOD0), declared error {:e} m, surface error {:e} m over {} samples, source error {:e} m",
                l.level,
                l.vertices,
                l.triangles,
                l.triangle_ratio,
                l.declared_error,
                l.surface_error,
                l.samples,
                l.source_error
            )?;
        }
        for i in &self.issues {
            let sev = if i.severity == Severity::Error { "error" } else { "warning" };
            match i.index {
                Some(n) => writeln!(f, "  {sev}: {} [{n}]: {}", i.check.name(), i.what)?,
                None => writeln!(f, "  {sev}: {}: {}", i.check.name(), i.what)?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta::{DeltaBasesBuilder, DeltaKind};
    use crate::mesh::fixtures::{CUBE_FACES, grid};
    use crate::mesh::{MeshBuilder, MeshData};
    use crate::package::{PackageLimits, read_package, write_package};
    use crate::region::Region;
    use crate::skeleton::JointRegressor;
    use crate::template::fixtures::{bare_grid, cube_template};
    use crate::template::{TemplateParts, UvTile};

    fn report(p: TemplateParts) -> ValidationReport {
        validate(&Template::new(p).unwrap())
    }

    fn errors(r: &ValidationReport, check: Check) -> usize {
        r.issues.iter().filter(|i| i.check == check && i.severity == Severity::Error).count()
    }

    /// A bare template (one region, no skeleton, no deltas) around `mesh`.
    fn bare(mesh: Mesh) -> TemplateParts {
        let mut p = bare_grid(1);
        p.regions.vertex_regions = vec![0; mesh.vertex_count()];
        p.mesh = mesh;
        p
    }

    /// A mesh of one part from positions (UV = (x, 1 - y)) and faces.
    fn mesh_of(positions: &[Vec3], faces: &[&[u32]], closed: bool) -> Mesh {
        let mut b = MeshBuilder::new();
        b.begin_part("part", closed).unwrap();
        for &p in positions {
            b.vertex(p, Vec2::new(p.x, 1.0 - p.y));
        }
        for f in faces {
            b.face(f, 0).unwrap();
        }
        b.finish().unwrap()
    }

    fn corner(i: usize) -> Vec3 {
        Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32)
    }

    /// The shared-vertex cube's faces `range`, the first reversed when `flip_first`, as a closed part.
    fn cube_report(range: std::ops::Range<usize>, flip_first: bool) -> ValidationReport {
        let positions: Vec<Vec3> = (0..8).map(corner).collect();
        let faces: Vec<Vec<u32>> = CUBE_FACES[range]
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let mut f: Vec<u32> = f.iter().map(|&c| c as u32).collect();
                if flip_first && i == 0 {
                    f.reverse();
                }
                f
            })
            .collect();
        let faces: Vec<&[u32]> = faces.iter().map(Vec::as_slice).collect();
        report(bare(mesh_of(&positions, &faces, true)))
    }

    #[test]
    fn ac_wp06_1_template_package_passes_full_validation() {
        let t = Template::new(cube_template()).unwrap();
        let back = read_package(&write_package(&t).unwrap(), &PackageLimits::DEFAULT).unwrap();
        let r = validate(&back);
        assert!(r.passed(), "{r}");
        assert_eq!(r, validate(&t));
        // Watertight and manifold: one closed genus-0 component.
        assert_eq!(r.parts.len(), 1);
        let p = &r.parts[0];
        assert_eq!((p.faces, p.vertices, p.edges, p.components, p.euler), (6, 8, 12, 1, 2));
        assert!(p.closed && p.watertight() && p.manifold());
        assert_eq!(p.degenerate_faces, 0);
        // Region coverage.
        assert_eq!(r.regions.covered_vertices, 24);
        assert_eq!(r.regions.vertex_counts, [0, 12, 12]);
        assert_eq!((r.regions.empty_leaves, r.regions.split_duplicates), (0, 0));
        assert_eq!(r.regions.soft_max_deviation, 0.0);
        // UV overlap and texel density: each face is a 0.29 x 0.46 cell on 1 m² at 1024 texels.
        let tile = &r.tiles[0];
        assert_eq!(tile.faces, 6);
        assert_eq!((tile.out_of_range_faces, tile.flipped_faces, tile.overlapping_pairs), (0, 0, 0));
        assert!((tile.coverage - 6.0 * 0.29 * 0.46).abs() < 1e-4, "{}", tile.coverage);
        let expect = (0.29_f64 * 0.46 * 1024.0 * 1024.0 / 1e6).sqrt() as f32;
        for d in [tile.density_min, tile.density_median, tile.density_mean, tile.density_max] {
            assert!((d - expect).abs() < 1e-4, "{d} vs {expect}");
        }
        assert_eq!(tile.area_below_target, 1.0, "target 0.5 texels/mm is above {expect}");
        // Landmarks: valid, but fewer than MOD-06 asks for.
        assert_eq!(r.landmarks, LandmarkReport { face: 1, body: 1, invalid: 0 });
        assert!(!r.landmarks.meets_mod06());
        // Weight normalisation.
        assert_eq!(r.weights.influences, 4);
        assert!(r.weights.max_sum_deviation <= WEIGHT_TOLERANCE);
        assert_eq!((r.weights.unnormalised, r.weights.unsorted, r.weights.seam_mismatches), (0, 0, 0));
        // Deltas: authored only, inside their regions, seam-consistent.
        let d = &r.deltas;
        assert_eq!((d.components, d.scan_derived, d.outside_region, d.seam_mismatches), (6, 0, 0, 0));
        assert_eq!(d.max_displacement, 0.25);
        // LOD error report: LOD1 is the same cube with shared vertices.
        assert_eq!(r.lods.len(), 1);
        let l = &r.lods[0];
        assert_eq!((l.level, l.vertices, l.triangles, l.samples), (1, 8, 12, 8));
        assert_eq!(l.triangle_ratio, 1.0);
        assert!(l.surface_error < 1e-6 && l.source_error < 1e-6, "{l:?}");
        // Only warnings: the landmark count and the texel density.
        assert!(r.issues.iter().all(|i| i.severity == Severity::Warning), "{r}");
        assert_eq!((r.count(Check::Landmark), r.count(Check::TexelDensity)), (1, 1));
        let text = r.to_string();
        assert!(text.starts_with("template validation: passed"), "{text}");
        assert!(text.contains("part cube: 6 faces, 8 vertices, 12 edges") && text.contains("watertight"), "{text}");
        assert!(text.contains("LOD1: 8 vertices, 12 triangles"), "{text}");
    }

    #[test]
    fn open_grid_passes_with_boundary() {
        let r = report(bare_grid(3));
        assert!(r.passed(), "{r}");
        let p = &r.parts[0];
        assert!(!p.closed && !p.watertight() && p.manifold());
        assert_eq!((p.faces, p.vertices, p.edges, p.boundary_edges, p.euler), (9, 16, 24, 12, 1));
        assert_eq!(r.tiles[0].overlapping_pairs, 0);
        assert!((r.tiles[0].coverage - 1.0).abs() < 1e-5);
        assert_eq!(r.weights.influences, 0);
        assert!(r.lods.is_empty());
    }

    #[test]
    fn closed_part_with_a_hole_is_not_watertight() {
        let r = cube_report(0..5, false);
        assert!(!r.passed());
        assert_eq!(r.parts[0].boundary_edges, 4);
        assert_eq!(r.parts[0].euler, 1);
        assert_eq!(errors(&r, Check::Watertight), 1);
        assert_eq!(errors(&r, Check::Manifold), 0);
        assert!(r.to_string().contains("NOT watertight"));
    }

    #[test]
    fn flipped_face_breaks_orientation() {
        let r = cube_report(0..6, true);
        assert_eq!(r.parts[0].inconsistent_edges, 4);
        assert!(r.parts[0].watertight() && !r.parts[0].manifold());
        assert_eq!(errors(&r, Check::Manifold), 1);
        assert!(r.to_string().starts_with("template validation: FAILED"));
    }

    #[test]
    fn non_manifold_edge_and_vertex_are_errors() {
        // Three quads around the edge 0-1.
        let p = [
            Vec3::ZERO,
            Vec3::X,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 1.0),
        ];
        let r = report(bare(mesh_of(&p, &[&[0, 1, 3, 2], &[1, 0, 4, 5], &[0, 1, 7, 6]], false)));
        assert_eq!(r.parts[0].non_manifold_edges, 1);
        assert!(errors(&r, Check::Manifold) >= 1);
        // Two triangles touching at one vertex: a bow tie.
        let p = [Vec3::ZERO, Vec3::X, Vec3::Y, -Vec3::X, -Vec3::Y];
        let r = report(bare(mesh_of(&p, &[&[0, 1, 2], &[0, 3, 4]], false)));
        assert_eq!(r.parts[0].non_manifold_vertices, 1);
        assert_eq!(r.parts[0].non_manifold_edges, 0);
        assert_eq!(r.parts[0].components, 2);
        assert_eq!(errors(&r, Check::Manifold), 1);
        assert_eq!(r.issues.iter().find(|i| i.check == Check::Manifold).unwrap().index, Some(0));
    }

    #[test]
    fn degenerate_face_is_a_warning() {
        let p = [Vec3::ZERO, Vec3::X, Vec3::X * 2.0];
        let r = report(bare(mesh_of(&p, &[&[0, 1, 2]], false)));
        assert_eq!(r.parts[0].degenerate_faces, 1);
        assert_eq!(r.count(Check::DegenerateFace), 1);
        assert_eq!(errors(&r, Check::DegenerateFace), 0);
    }

    /// Two unit quads side by side whose UVs are given per quad (corners counter-clockwise).
    fn two_quads(uv_a: [Vec2; 4], uv_b: [Vec2; 4]) -> Mesh {
        let mut b = MeshBuilder::new();
        for (x0, uvs) in [(0.0, uv_a), (2.0, uv_b)] {
            let q = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
            let ids: Vec<u32> =
                q.iter().zip(uvs).map(|(&(x, y), uv)| b.vertex(Vec3::new(x0 + x, y, 0.0), uv)).collect();
            b.face(&ids, 0).unwrap();
        }
        b.finish().unwrap()
    }

    /// UVs of a quad spanning `x0..x1` in U and the whole tile in V (mirrored when `x1 < x0`).
    fn uv_square(x0: f32, x1: f32) -> [Vec2; 4] {
        [Vec2::new(x0, 1.0), Vec2::new(x1, 1.0), Vec2::new(x1, 0.0), Vec2::new(x0, 0.0)]
    }

    #[test]
    fn uv_overlap_range_and_flips_are_errors() {
        // Side by side in UV: no overlap, half the tile each.
        let r = report(bare(two_quads(uv_square(0.0, 0.5), uv_square(0.5, 1.0))));
        assert!(r.passed(), "{r}");
        assert_eq!(r.tiles[0].overlapping_pairs, 0);
        // Both on the whole tile: each triangle of one overlaps the identical triangle of the other.
        let r = report(bare(two_quads(uv_square(0.0, 1.0), uv_square(0.0, 1.0))));
        assert_eq!(r.tiles[0].overlapping_pairs, 2, "{r}");
        assert!((r.tiles[0].coverage - 2.0).abs() < 1e-5);
        assert_eq!(errors(&r, Check::UvOverlap), 2);
        // Partly overlapping.
        let r = report(bare(two_quads(uv_square(0.0, 0.6), uv_square(0.4, 1.0))));
        assert!(r.tiles[0].overlapping_pairs > 0);
        // Outside the tile.
        let r = report(bare(two_quads(uv_square(0.0, 0.5), uv_square(1.0, 1.5))));
        assert_eq!(r.tiles[0].out_of_range_faces, 1);
        assert_eq!(errors(&r, Check::UvRange), 1);
        // Mirrored.
        let r = report(bare(two_quads(uv_square(0.0, 0.5), uv_square(1.0, 0.5))));
        assert_eq!(r.tiles[0].flipped_faces, 1);
        assert_eq!(errors(&r, Check::UvFlipped), 1);
        assert_eq!(r.tiles[0].overlapping_pairs, 0);
    }

    #[test]
    fn vertex_shared_by_two_tiles_is_an_error() {
        let mut data: MeshData = grid(2).into_data();
        data.face_tiles[0] = 1;
        let mut p = bare(Mesh::new(data).unwrap());
        p.tiles.push(UvTile { name: "second".into(), resolution: 256, target_texels_per_mm: 0.0 });
        let r = report(p);
        // Face 0's corners 1, 3 and 4 are shared with tile-0 faces.
        assert_eq!(errors(&r, Check::UvRange), 3, "{r}");
        assert_eq!(r.tiles.len(), 2);
        assert_eq!((r.tiles[0].faces, r.tiles[1].faces), (3, 1));
        assert!(!r.to_string().contains("-0.0%"), "{r}");
    }

    #[test]
    fn texel_density_is_measured_per_face() {
        // grid(2): four faces of 0.25 m² each covering a quarter of a 1000-texel tile.
        let mut p = bare_grid(2);
        p.tiles[0] = UvTile { name: "all".into(), resolution: 1000, target_texels_per_mm: 0.5 };
        let r = report(p);
        let t = &r.tiles[0];
        // sqrt(0.25 * 1000² / 250000 mm²) = 1 texel/mm.
        assert!((t.density_min - 1.0).abs() < 1e-5 && (t.density_max - 1.0).abs() < 1e-5, "{t:?}");
        assert_eq!(t.area_below_target, 0.0);
        assert_eq!(r.count(Check::TexelDensity), 0);
    }

    #[test]
    fn invalid_landmarks_are_errors() {
        let mut p = cube_template();
        p.landmarks[0].weights = [0.5, 0.5, 0.5];
        // Corners (0,0,0), (0,1,0) and (1,1,1) share no face.
        p.landmarks[1].vertices = [0, 1, 6];
        let r = report(p);
        assert_eq!(r.landmarks.invalid, 2);
        assert_eq!(errors(&r, Check::Landmark), 2);
        let whats: Vec<&str> = r.issues.iter().filter(|i| i.check == Check::Landmark).map(|i| i.what).collect();
        assert!(whats.contains(&"landmark weights are not a barycentric triple"), "{whats:?}");
        assert!(whats.contains(&"landmark corners are not on one face"), "{whats:?}");
    }

    #[test]
    fn unnormalised_unsorted_and_split_skin_weights_are_errors() {
        let mut p = cube_template();
        let k = p.skin.influences as usize;
        p.skin.weights[0] = 0.5;
        // Vertex 1 at (0, 1, 0): weight 1 on joint 1; store it unsorted behind a zero.
        p.skin.weights[k..2 * k].copy_from_slice(&[0.0, 1.0, 0.0, 0.0]);
        p.skin.joints[k..2 * k].copy_from_slice(&[0, 1, 0, 0]);
        let r = report(p);
        assert_eq!((r.weights.unnormalised, r.weights.unsorted), (1, 1), "{r}");
        assert_eq!(r.weights.seam_mismatches, 2);
        assert_eq!(errors(&r, Check::SkinWeights), 4);
        assert!((r.weights.max_sum_deviation - 0.5).abs() < 1e-6);
    }

    #[test]
    fn deltas_outside_region_or_split_at_seams_are_errors() {
        let base = cube_template();
        let top = base.mesh.positions().iter().position(|p| p.y > 0.5).unwrap() as u32;
        let mut p = bare_grid(1);
        p.mesh = base.mesh;
        p.regions = base.regions;
        let mut b = DeltaBasesBuilder::new();
        // Region 1 is the lower half; the vertex is in the upper half.
        b.add("lower.wrong", DeltaKind::Semantic, 1, &[(top, Vec3::X)]).unwrap();
        // Vertex 0 has duplicates (corner 0 is on three faces) that do not move.
        b.add("seam.split", DeltaKind::Semantic, 0, &[(0, Vec3::X)]).unwrap();
        p.deltas = b.finish().unwrap();
        p.skeleton.regressor = JointRegressor::empty(2);
        let r = report(p);
        assert_eq!((r.deltas.outside_region, r.deltas.seam_mismatches), (1, 2), "{r}");
        assert_eq!(errors(&r, Check::Deltas), 3);
        assert_eq!(r.deltas.max_displacement, 1.0);
    }

    #[test]
    fn lod_error_report_compares_with_declared_error() {
        let mut p = cube_template();
        let mut data = p.lods[0].mesh.clone().into_data();
        for q in &mut data.positions {
            q.x += 0.1;
        }
        p.lods[0].mesh = Mesh::new(data).unwrap();
        let r = report(p.clone());
        let l = &r.lods[0];
        assert!((l.surface_error - 0.1).abs() < 1e-5 && (l.source_error - 0.1).abs() < 1e-5, "{l:?}");
        assert_eq!(errors(&r, Check::LodError), 2);
        p.lods[0].declared_error = 0.1;
        let r = report(p);
        assert_eq!(errors(&r, Check::LodError), 0, "{r}");
        assert!(r.passed(), "{r}");
    }

    #[test]
    fn region_coverage_and_soft_weights() {
        let mut p = cube_template();
        p.regions.regions.push(Region { name: "empty".into(), parent: Some(0), side: crate::names::Side::Left });
        p.regions.vertex_regions[0] = 2;
        if let Some(soft) = &mut p.regions.soft {
            soft.weights[1] = 0.5;
        }
        let r = report(p);
        assert_eq!(r.regions.empty_leaves, 1);
        assert_eq!(r.regions.split_duplicates, 1);
        assert!((r.regions.soft_max_deviation - 0.5).abs() < 1e-6);
        assert_eq!(r.count(Check::RegionCoverage), 2);
        assert_eq!(errors(&r, Check::RegionCoverage), 0);
        assert_eq!(errors(&r, Check::SoftRegionWeights), 1);
    }

    #[test]
    fn issues_are_capped_per_check() {
        let mut data = grid(12).into_data();
        for uv in &mut data.uvs {
            uv.x += 2.0;
        }
        let r = report(bare(Mesh::new(data).unwrap()));
        assert_eq!(r.tiles[0].out_of_range_faces, 144);
        assert_eq!(errors(&r, Check::UvRange), MAX_ISSUES_PER_CHECK);
    }
}
