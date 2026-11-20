//! Template mesh topology `[MOD-01]`: positions and UVs as separate arrays, quad-dominant polygon
//! faces, the duplicate-vertex tables for UV seams and hard edges, and mesh parts.
//!
//! # Layout
//!
//! - Vertices are **render vertices**: each has one position ([`Mesh::positions`]) and one UV
//!   ([`Mesh::uvs`]); the attributes are separate arrays (structure of arrays `[PERF-R2]`), so the
//!   evaluator and the GPU upload read only what they need.
//! - Faces are polygons of 3 to [`MAX_FACE_SIDES`] corners in compressed form: face `f` is
//!   `face_vertices[face_offsets[f]..face_offsets[f + 1]]`, counter-clockwise seen from outside.
//!   Each face carries the index of its UV tile ([`Mesh::face_tiles`], see [`crate::template::UvTile`]);
//!   UVs are tile-local in `[0, 1]`, origin top-left, V down.
//! - A vertex is **duplicated** where the surface has a UV seam or a hard edge (eyelid edge, lip
//!   vermilion edge, ear rim `[EVAL-04]`). Duplicates have bit-identical positions. Two per-vertex
//!   tables record them: [`Mesh::position_keys`] maps each vertex to the lowest-indexed vertex with
//!   the same position (its *weld key*; topology checks and edges use these keys), and
//!   [`Mesh::normal_keys`] maps it to the lowest-indexed vertex it shares its normal with. A UV-seam
//!   duplicate shares its normal (smooth across the seam); a hard-edge duplicate has its own.
//! - Faces are grouped into contiguous **parts** ([`MeshPart`]): the body surface and the separate
//!   meshes of `[MOD-01]` (eyeballs, tear lines, teeth, gums, tongue, helper meshes). A part that is
//!   meant to be watertight says so, and validation checks it.
//!
//! A [`Mesh`] is immutable and always satisfies the rules checked by [`Mesh::new`], so code that
//! holds one can index its arrays with the indices it stores. Build one with [`MeshBuilder`] or from
//! raw arrays with [`MeshData`].

use humanart_math::{Vec2, Vec3};

use crate::error::{ModelError, Result};
use crate::names::check_names;
use crate::topology::Topology;

/// The most vertices one mesh may have.
pub const MAX_VERTICES: usize = 1 << 24;
/// The most faces one mesh may have.
pub const MAX_FACES: usize = 1 << 24;
/// The most corners one face may have.
pub const MAX_FACE_SIDES: usize = 64;
/// The most parts one mesh may have.
pub const MAX_PARTS: usize = 256;

/// A named, contiguous range of faces: one separately meaningful piece of geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshPart {
    /// Stable name ("body", "eyeball-L", "teeth-upper").
    pub name: String,
    /// The first face of the part.
    pub first_face: u32,
    /// How many faces it has (at least one).
    pub face_count: u32,
    /// Whether the part must be a closed (watertight) surface.
    pub closed: bool,
}

impl MeshPart {
    /// The face indices of the part.
    #[must_use]
    pub fn faces(&self) -> std::ops::Range<usize> {
        let start = self.first_face as usize;
        start..start + self.face_count as usize
    }
}

/// The raw arrays of a mesh, as a generator or a package reader produces them. [`Mesh::new`] checks
/// them; the field docs state the rules.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshData {
    /// One position per vertex, metres; all finite.
    pub positions: Vec<Vec3>,
    /// One UV per vertex, tile-local; all finite.
    pub uvs: Vec<Vec2>,
    /// `face_count + 1` offsets into `face_vertices`, starting at 0, each face 3 to
    /// [`MAX_FACE_SIDES`] corners, the last equal to `face_vertices.len()`.
    pub face_offsets: Vec<u32>,
    /// The corners of all faces; no two corners of a face share a weld key.
    pub face_vertices: Vec<u32>,
    /// One UV tile index per face.
    pub face_tiles: Vec<u16>,
    /// Per vertex: the lowest-indexed vertex with the same position (itself if not duplicated).
    pub position_keys: Vec<u32>,
    /// Per vertex: the lowest-indexed vertex sharing its normal; it has the same weld key.
    pub normal_keys: Vec<u32>,
    /// Contiguous face ranges covering every face in order (empty only when there are no faces).
    pub parts: Vec<MeshPart>,
}

/// A checked template mesh. See the module documentation for the layout.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    data: MeshData,
}

impl Mesh {
    /// Checks `data` against the rules of [`MeshData`] and wraps it.
    pub fn new(data: MeshData) -> Result<Self> {
        check(&data)?;
        Ok(Self { data })
    }

    /// The raw arrays back.
    #[must_use]
    pub fn into_data(self) -> MeshData {
        self.data
    }

    /// The raw arrays.
    #[must_use]
    pub const fn data(&self) -> &MeshData {
        &self.data
    }

    /// How many vertices the mesh has.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.data.positions.len()
    }

    /// How many faces the mesh has.
    #[must_use]
    pub fn face_count(&self) -> usize {
        self.data.face_offsets.len().saturating_sub(1)
    }

    /// The vertex positions.
    #[must_use]
    pub fn positions(&self) -> &[Vec3] {
        &self.data.positions
    }

    /// The vertex UVs (tile-local).
    #[must_use]
    pub fn uvs(&self) -> &[Vec2] {
        &self.data.uvs
    }

    /// The face offsets (`face_count + 1` entries).
    #[must_use]
    pub fn face_offsets(&self) -> &[u32] {
        &self.data.face_offsets
    }

    /// All face corners.
    #[must_use]
    pub fn face_vertices(&self) -> &[u32] {
        &self.data.face_vertices
    }

    /// The UV tile index of each face.
    #[must_use]
    pub fn face_tiles(&self) -> &[u16] {
        &self.data.face_tiles
    }

    /// The weld key of each vertex.
    #[must_use]
    pub fn position_keys(&self) -> &[u32] {
        &self.data.position_keys
    }

    /// The normal key of each vertex.
    #[must_use]
    pub fn normal_keys(&self) -> &[u32] {
        &self.data.normal_keys
    }

    /// The parts.
    #[must_use]
    pub fn parts(&self) -> &[MeshPart] {
        &self.data.parts
    }

    /// The corners of face `f`; empty when `f` is out of range.
    #[must_use]
    pub fn face(&self, f: usize) -> &[u32] {
        match (self.data.face_offsets.get(f), self.data.face_offsets.get(f + 1)) {
            (Some(&a), Some(&b)) => &self.data.face_vertices[a as usize..b as usize],
            _ => &[],
        }
    }

    /// Iterates over the faces' corner lists.
    pub fn faces(&self) -> impl Iterator<Item = &[u32]> + '_ {
        self.data.face_offsets.windows(2).map(|w| &self.data.face_vertices[w[0] as usize..w[1] as usize])
    }

    /// The part a face belongs to, if the face exists.
    #[must_use]
    pub fn part_of_face(&self, f: usize) -> Option<usize> {
        self.data.parts.iter().position(|p| p.faces().contains(&f))
    }

    /// The vector area of face `f` (Newell's method): its direction is the face normal and its
    /// length the area for a planar polygon. Zero for an out-of-range face.
    #[must_use]
    pub fn face_vector_area(&self, f: usize) -> Vec3 {
        polygon_vector_area(self.face(f), &self.data.positions)
    }

    /// The signed UV area of face `f` (positive when the UVs wind the same way as a counter-clockwise
    /// polygon in a V-up frame, i.e. clockwise on the V-down texture).
    #[must_use]
    pub fn face_uv_area(&self, f: usize) -> f32 {
        let corners = self.face(f);
        let mut twice = 0.0_f32;
        for (i, &a) in corners.iter().enumerate() {
            let b = corners[(i + 1) % corners.len()];
            let (pa, pb) = (self.data.uvs[a as usize], self.data.uvs[b as usize]);
            twice += pa.x * pb.y - pb.x * pa.y;
        }
        // V points down, so a counter-clockwise outward face maps to a negative shoelace sum.
        -0.5 * twice
    }

    /// Splits every face into triangles (see [`Triangulation`]).
    #[must_use]
    pub fn triangulate(&self) -> Triangulation {
        triangulate(self)
    }

    /// Computes the edge and adjacency tables of the mesh (see [`Topology`]).
    #[must_use]
    pub fn topology(&self) -> Topology {
        Topology::new(self, &self.triangulate())
    }
}

/// Newell's vector area of a polygon given by vertex indices into `positions`. The caller
/// guarantees the indices are in range.
pub(crate) fn polygon_vector_area(corners: &[u32], positions: &[Vec3]) -> Vec3 {
    let mut n = Vec3::ZERO;
    for (i, &a) in corners.iter().enumerate() {
        let b = corners[(i + 1) % corners.len()];
        let (p, q) = (positions[a as usize], positions[b as usize]);
        n.x += (p.y - q.y) * (p.z + q.z);
        n.y += (p.z - q.z) * (p.x + q.x);
        n.z += (p.x - q.x) * (p.y + q.y);
    }
    n * 0.5
}

/// The triangles of a mesh: three vertex indices per triangle, counter-clockwise, and the face each
/// came from.
///
/// A quad is split along its shorter diagonal in the rest positions (ties: the 0-2 diagonal); a
/// polygon with more corners is split as a fan from its first corner (the generator only makes
/// convex ones). The split depends on the rest positions only, so it is fixed for a template.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Triangulation {
    /// `3 * triangle_count` vertex indices.
    pub indices: Vec<u32>,
    /// The face of each triangle.
    pub triangle_faces: Vec<u32>,
}

impl Triangulation {
    /// How many triangles there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.triangle_faces.len()
    }

    /// Whether there are no triangles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triangle_faces.is_empty()
    }

    /// The corners of triangle `t`, if it exists.
    #[must_use]
    pub fn triangle(&self, t: usize) -> Option<[u32; 3]> {
        let s = self.indices.get(3 * t..3 * t + 3)?;
        Some([s[0], s[1], s[2]])
    }
}

fn triangulate(mesh: &Mesh) -> Triangulation {
    let corners = mesh.data.face_vertices.len();
    let faces = mesh.face_count();
    let mut out = Triangulation {
        indices: Vec::with_capacity(3 * corners.saturating_sub(2 * faces)),
        triangle_faces: Vec::with_capacity(corners.saturating_sub(2 * faces)),
    };
    let p = &mesh.data.positions;
    for (f, face) in mesh.faces().enumerate() {
        let f = f as u32;
        if let [a, b, c, d] = *face {
            let d02 = p[a as usize].distance_squared(p[c as usize]);
            let d13 = p[b as usize].distance_squared(p[d as usize]);
            if d13 < d02 {
                out.indices.extend_from_slice(&[a, b, d, b, c, d]);
            } else {
                out.indices.extend_from_slice(&[a, b, c, a, c, d]);
            }
            out.triangle_faces.extend_from_slice(&[f, f]);
        } else {
            for i in 1..face.len() - 1 {
                out.indices.extend_from_slice(&[face[0], face[i], face[i + 1]]);
                out.triangle_faces.push(f);
            }
        }
    }
    out
}

fn check(d: &MeshData) -> Result<()> {
    let v = d.positions.len();
    if v > MAX_VERTICES {
        return Err(ModelError::limit("vertex count", MAX_VERTICES as u64, v as u64));
    }
    if d.uvs.len() != v || d.position_keys.len() != v || d.normal_keys.len() != v {
        return Err(ModelError::invalid("mesh", 0, "per-vertex arrays differ in length"));
    }
    if let Some(i) = d.positions.iter().position(|p| !p.is_finite()) {
        return Err(ModelError::invalid("vertex", i, "position is not finite"));
    }
    if let Some(i) = d.uvs.iter().position(|p| !p.is_finite()) {
        return Err(ModelError::invalid("vertex", i, "UV is not finite"));
    }
    // Weld and normal keys are canonical: they point at a lower-or-equal vertex that is its own key.
    for (i, (&pk, &nk)) in d.position_keys.iter().zip(&d.normal_keys).enumerate() {
        let (pk, nk) = (pk as usize, nk as usize);
        if pk > i || d.position_keys[pk] as usize != pk {
            return Err(ModelError::invalid("vertex", i, "weld key is not canonical"));
        }
        if d.positions[pk] != d.positions[i] {
            return Err(ModelError::invalid("vertex", i, "duplicate does not have its key's position"));
        }
        if nk > i || d.normal_keys[nk] as usize != nk || d.position_keys[nk] as usize != pk {
            return Err(ModelError::invalid("vertex", i, "normal key is not canonical or has another position"));
        }
    }
    check_faces(d)?;
    check_parts(d)
}

fn check_faces(d: &MeshData) -> Result<()> {
    let v = d.positions.len();
    let Some((&first, rest)) = d.face_offsets.split_first() else {
        return Err(ModelError::invalid("mesh", 0, "face offsets are empty (need face_count + 1 entries)"));
    };
    let faces = rest.len();
    if faces > MAX_FACES {
        return Err(ModelError::limit("face count", MAX_FACES as u64, faces as u64));
    }
    if first != 0 {
        return Err(ModelError::invalid("face", 0, "face offsets do not start at 0"));
    }
    if d.face_tiles.len() != faces {
        return Err(ModelError::invalid("mesh", 0, "face tile count differs from the face count"));
    }
    let mut start = 0_usize;
    let mut keys: Vec<u32> = Vec::with_capacity(MAX_FACE_SIDES);
    for (f, &end) in rest.iter().enumerate() {
        let end = end as usize;
        let sides = end.checked_sub(start).ok_or(ModelError::invalid("face", f, "face offsets decrease"))?;
        if !(3..=MAX_FACE_SIDES).contains(&sides) {
            return Err(ModelError::invalid("face", f, "face has fewer than 3 or more than 64 corners"));
        }
        let corners =
            d.face_vertices.get(start..end).ok_or(ModelError::invalid("face", f, "face offset past the corners"))?;
        keys.clear();
        for &c in corners {
            let key =
                *d.position_keys.get(c as usize).ok_or(ModelError::invalid("face", f, "vertex index out of range"))?;
            keys.push(key);
        }
        keys.sort_unstable();
        if keys.windows(2).any(|w| w[0] == w[1]) {
            return Err(ModelError::invalid("face", f, "face uses one position twice"));
        }
        start = end;
    }
    if start != d.face_vertices.len() {
        return Err(ModelError::invalid("mesh", 0, "last face offset differs from the corner count"));
    }
    debug_assert!(v <= MAX_VERTICES);
    Ok(())
}

fn check_parts(d: &MeshData) -> Result<()> {
    let faces = d.face_offsets.len() - 1;
    if d.parts.len() > MAX_PARTS {
        return Err(ModelError::limit("mesh part count", MAX_PARTS as u64, d.parts.len() as u64));
    }
    check_names("mesh part", d.parts.iter().map(|p| p.name.as_str()))?;
    let mut next = 0_usize;
    for (i, part) in d.parts.iter().enumerate() {
        if part.first_face as usize != next {
            return Err(ModelError::invalid("mesh part", i, "parts are not contiguous in face order"));
        }
        if part.face_count == 0 {
            return Err(ModelError::invalid("mesh part", i, "part has no faces"));
        }
        next += part.face_count as usize;
    }
    if next != faces {
        return Err(ModelError::invalid("mesh", 0, "parts do not cover every face"));
    }
    Ok(())
}

/// Builds a [`Mesh`] vertex by vertex and face by face, for the generator and for tests.
///
/// ```
/// use humanart_math::{Vec2, Vec3};
/// use humanart_model::mesh::MeshBuilder;
/// let mut b = MeshBuilder::new();
/// b.begin_part("quad", false)?;
/// let v: Vec<u32> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
///     .iter()
///     .map(|&(x, y)| b.vertex(Vec3::new(x, y, 0.0), Vec2::new(x, 1.0 - y)))
///     .collect();
/// b.face(&v, 0)?;
/// let mesh = b.finish()?;
/// assert_eq!((mesh.vertex_count(), mesh.face_count()), (4, 1));
/// # Ok::<(), humanart_model::ModelError>(())
/// ```
#[derive(Clone, Debug)]
pub struct MeshBuilder {
    data: MeshData,
}

impl Default for MeshBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MeshBuilder {
    /// An empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self { data: MeshData { face_offsets: vec![0], ..MeshData::default() } }
    }

    /// Adds a vertex and returns its index.
    pub fn vertex(&mut self, position: Vec3, uv: Vec2) -> u32 {
        let i = self.data.positions.len() as u32;
        self.data.positions.push(position);
        self.data.uvs.push(uv);
        self.data.position_keys.push(i);
        self.data.normal_keys.push(i);
        i
    }

    /// Adds a duplicate of vertex `of` with its own UV: across a UV seam when `smooth` (the
    /// duplicate shares `of`'s normal), across a hard edge when not.
    pub fn duplicate(&mut self, of: u32, uv: Vec2, smooth: bool) -> Result<u32> {
        let o = of as usize;
        let (Some(&p), Some(&pk), Some(&nk)) =
            (self.data.positions.get(o), self.data.position_keys.get(o), self.data.normal_keys.get(o))
        else {
            return Err(ModelError::invalid("vertex", o, "duplicate of a vertex that does not exist"));
        };
        let i = self.data.positions.len() as u32;
        self.data.positions.push(p);
        self.data.uvs.push(uv);
        self.data.position_keys.push(pk);
        self.data.normal_keys.push(if smooth { nk } else { i });
        Ok(i)
    }

    /// Starts a new part; the faces added from now on belong to it.
    pub fn begin_part(&mut self, name: &str, closed: bool) -> Result<()> {
        crate::names::check_name("mesh part", name)?;
        let first_face = (self.data.face_offsets.len() - 1) as u32;
        self.data.parts.push(MeshPart { name: name.to_owned(), first_face, face_count: 0, closed });
        Ok(())
    }

    /// Adds a face with the given corners (counter-clockwise from outside) in UV tile `tile`, and
    /// returns its index. Faces added before any [`begin_part`](Self::begin_part) go to an open
    /// part named `mesh`.
    pub fn face(&mut self, corners: &[u32], tile: u16) -> Result<u32> {
        let f = self.data.face_offsets.len() - 1;
        if !(3..=MAX_FACE_SIDES).contains(&corners.len()) {
            return Err(ModelError::invalid("face", f, "face has fewer than 3 or more than 64 corners"));
        }
        if corners.iter().any(|&c| c as usize >= self.data.positions.len()) {
            return Err(ModelError::invalid("face", f, "vertex index out of range"));
        }
        if self.data.parts.is_empty() {
            self.data.parts.push(MeshPart { name: "mesh".to_owned(), first_face: 0, face_count: 0, closed: false });
        }
        self.data.face_vertices.extend_from_slice(corners);
        self.data.face_offsets.push(self.data.face_vertices.len() as u32);
        self.data.face_tiles.push(tile);
        if let Some(part) = self.data.parts.last_mut() {
            part.face_count += 1;
        }
        Ok(f as u32)
    }

    /// How many vertices have been added.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.data.positions.len()
    }

    /// The position of a vertex added earlier.
    #[must_use]
    pub fn position(&self, v: u32) -> Option<Vec3> {
        self.data.positions.get(v as usize).copied()
    }

    /// Checks the mesh and returns it.
    pub fn finish(self) -> Result<Mesh> {
        Mesh::new(self.data)
    }
}

/// Test fixtures shared by the crate's tests: small synthetic meshes.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    /// A closed unit cube of six quads, one UV tile, every vertex shared (UVs are a cross layout
    /// squashed, not seam-correct; topology tests do not look at them).
    pub fn cube_shared() -> Mesh {
        let mut b = MeshBuilder::new();
        b.begin_part("cube", true).unwrap();
        let mut v = Vec::new();
        for i in 0..8_u32 {
            let p = Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32);
            v.push(b.vertex(p, Vec2::new(p.x * 0.5 + p.z * 0.25, p.y * 0.5)));
        }
        for face in CUBE_FACES {
            b.face(&face.map(|i| v[i]), 0).unwrap();
        }
        b.finish().unwrap()
    }

    /// The cube's faces as corner lists into the 8 corners `(x, y, z)` = bits of the index,
    /// counter-clockwise from outside.
    pub const CUBE_FACES: [[usize; 4]; 6] = [
        [0, 2, 3, 1], // z = 0, normal -z
        [4, 5, 7, 6], // z = 1, normal +z
        [0, 1, 5, 4], // y = 0, normal -y
        [2, 6, 7, 3], // y = 1, normal +y
        [0, 4, 6, 2], // x = 0, normal -x
        [1, 3, 7, 5], // x = 1, normal +x
    ];

    /// A closed cube whose six faces each have their own four vertices (24 vertices), each face
    /// laid out in its own cell of a 3x2 grid in tile 0: UV seams everywhere, no UV overlap.
    /// `hard` makes the duplicates hard edges (flat shading) instead of smooth seams.
    pub fn cube_split(hard: bool) -> Mesh {
        let mut b = MeshBuilder::new();
        b.begin_part("cube", true).unwrap();
        let corner = |i: usize| Vec3::new((i & 1) as f32, ((i >> 1) & 1) as f32, ((i >> 2) & 1) as f32);
        let mut first: [Option<u32>; 8] = [None; 8];
        for (f, face) in CUBE_FACES.iter().enumerate() {
            let (cx, cy) = ((f % 3) as f32 / 3.0, (f / 3) as f32 / 2.0);
            let cell = [(0.02, 0.48), (0.31, 0.48), (0.31, 0.02), (0.02, 0.02)];
            let mut ids = [0_u32; 4];
            for (k, &c) in face.iter().enumerate() {
                let uv = Vec2::new(cx + cell[k].0, cy + cell[k].1);
                ids[k] = match first[c] {
                    None => {
                        let id = b.vertex(corner(c), uv);
                        first[c] = Some(id);
                        id
                    }
                    Some(of) => b.duplicate(of, uv, !hard).unwrap(),
                };
            }
            b.face(&ids, 0).unwrap();
        }
        b.finish().unwrap()
    }

    /// An open `n` x `n` grid of quads on the unit square in the XY plane (normal +Z), UVs covering
    /// the tile.
    pub fn grid(n: usize) -> Mesh {
        let mut b = MeshBuilder::new();
        b.begin_part("grid", false).unwrap();
        for j in 0..=n {
            for i in 0..=n {
                let (x, y) = (i as f32 / n as f32, j as f32 / n as f32);
                b.vertex(Vec3::new(x, y, 0.0), Vec2::new(x, 1.0 - y));
            }
        }
        let id = |i: usize, j: usize| (j * (n + 1) + i) as u32;
        for j in 0..n {
            for i in 0..n {
                b.face(&[id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)], 0).unwrap();
            }
        }
        b.finish().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    #[test]
    fn builder_makes_a_checked_cube() {
        let m = cube_shared();
        assert_eq!((m.vertex_count(), m.face_count()), (8, 6));
        assert_eq!(m.parts().len(), 1);
        assert!(m.parts()[0].closed);
        assert_eq!(m.face(1), &[4, 5, 7, 6]);
        assert!(m.face(6).is_empty());
        assert_eq!(m.faces().count(), 6);
        // Outward normals: face 1 (z = 1) points +z with area 1.
        let n = m.face_vector_area(1);
        assert!((n - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-6, "{n:?}");
        for f in 0..6 {
            assert!((m.face_vector_area(f).length() - 1.0).abs() < 1e-6);
        }
        assert_eq!(m.part_of_face(5), Some(0));
        assert_eq!(m.part_of_face(6), None);
    }

    #[test]
    fn duplicates_record_weld_and_normal_keys() {
        let smooth = cube_split(false);
        assert_eq!(smooth.vertex_count(), 24);
        let hard = cube_split(true);
        // Every vertex welds to one of the 8 corners.
        let mut weld: Vec<u32> = smooth.position_keys().to_vec();
        weld.sort_unstable();
        weld.dedup();
        assert_eq!(weld.len(), 8);
        // Smooth seams share the normal key; hard edges give each duplicate its own.
        assert!(smooth.normal_keys().iter().zip(smooth.position_keys()).all(|(n, p)| n == p));
        assert!(hard.normal_keys().iter().enumerate().all(|(i, &n)| n as usize == i));
        // UV areas are positive (counter-clockwise faces on a V-down texture) and nonzero.
        for f in 0..6 {
            assert!(smooth.face_uv_area(f) > 0.0, "face {f}: {}", smooth.face_uv_area(f));
        }
    }

    #[test]
    fn grid_uv_area_matches_the_layout() {
        let g = grid(4);
        let total: f32 = (0..g.face_count()).map(|f| g.face_uv_area(f)).sum();
        assert!((total - 1.0).abs() < 1e-5, "{total}");
    }

    #[test]
    fn quads_split_along_the_shorter_diagonal() {
        // A rhombus long along 0-2: the 1-3 diagonal is shorter.
        let mut b = MeshBuilder::new();
        let w: Vec<u32> = [(-2.0, 0.0), (0.0, -1.0), (2.0, 0.0), (0.0, 1.0)]
            .iter()
            .map(|&(x, y)| b.vertex(Vec3::new(x, y, 0.0), Vec2::ZERO))
            .collect();
        b.face(&w, 0).unwrap();
        let fan: Vec<u32> = (0..5)
            .map(|k| {
                let a = k as f32 * std::f32::consts::TAU / 5.0;
                b.vertex(Vec3::new(a.cos() + 5.0, a.sin(), 0.0), Vec2::ZERO)
            })
            .collect();
        b.face(&fan, 0).unwrap();
        let m = b.finish().unwrap();
        let t = m.triangulate();
        assert_eq!(t.len(), 2 + 3);
        assert_eq!(t.triangle(0), Some([0, 1, 3]));
        assert_eq!(t.triangle(1), Some([1, 2, 3]));
        assert_eq!(t.triangle(2), Some([4, 5, 6]));
        assert_eq!(t.triangle(4), Some([4, 7, 8]));
        assert_eq!(t.triangle_faces, vec![0, 0, 1, 1, 1]);
        assert_eq!(t.triangle(5), None);
    }

    #[test]
    fn checks_refuse_broken_arrays() {
        type Change = Box<dyn Fn(&mut MeshData)>;
        let good = cube_shared().into_data();
        let cases: Vec<(&str, Change)> = vec![
            ("uv length", Box::new(|d| d.uvs.truncate(7))),
            ("nan", Box::new(|d| d.positions[3].y = f32::NAN)),
            ("uv inf", Box::new(|d| d.uvs[0].x = f32::INFINITY)),
            ("index", Box::new(|d| d.face_vertices[2] = 99)),
            ("repeat", Box::new(|d| d.face_vertices[2] = d.face_vertices[0])),
            ("offsets start", Box::new(|d| d.face_offsets[0] = 1)),
            ("offsets decrease", Box::new(|d| d.face_offsets.swap(1, 2))),
            ("too few sides", Box::new(|d| d.face_offsets[1] = 2)),
            ("offsets end", Box::new(|d| *d.face_offsets.last_mut().unwrap() = 23)),
            ("empty offsets", Box::new(|d| d.face_offsets.clear())),
            ("tiles", Box::new(|d| d.face_tiles.truncate(5))),
            ("weld key forward", Box::new(|d| d.position_keys[0] = 1)),
            ("weld key position", Box::new(|d| d.position_keys[1] = 0)),
            ("normal key other position", Box::new(|d| d.normal_keys[1] = 0)),
            ("part gap", Box::new(|d| d.parts[0].first_face = 1)),
            ("part short", Box::new(|d| d.parts[0].face_count = 5)),
            (
                "part empty",
                Box::new(|d| d.parts.push(MeshPart { name: "x".into(), first_face: 6, face_count: 0, closed: false })),
            ),
            ("part name", Box::new(|d| d.parts[0].name = " bad".into())),
            ("no parts", Box::new(|d| d.parts.clear())),
        ];
        for (what, change) in cases {
            let mut d = good.clone();
            change(&mut d);
            assert!(Mesh::new(d).is_err(), "{what} was accepted");
        }
        assert!(Mesh::new(good).is_ok());
        // An empty mesh is valid.
        assert!(MeshBuilder::new().finish().is_ok());
    }

    #[test]
    fn builder_refuses_bad_input() {
        let mut b = MeshBuilder::new();
        let a = b.vertex(Vec3::ZERO, Vec2::ZERO);
        assert!(b.face(&[a, a], 0).is_err());
        assert!(b.face(&[a, 1, 2], 0).is_err());
        assert!(b.duplicate(7, Vec2::ZERO, true).is_err());
        assert!(b.begin_part("", false).is_err());
        let d = b.duplicate(a, Vec2::ONE, true).unwrap();
        assert_eq!(b.position(d), Some(Vec3::ZERO));
        assert_eq!(b.vertex_count(), 2);
        // A face over a vertex and its duplicate uses one position twice.
        let c = b.vertex(Vec3::X, Vec2::ZERO);
        b.face(&[a, d, c], 0).unwrap();
        assert!(b.finish().is_err());
    }
}
