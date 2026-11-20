//! A captured scan as a triangle mesh in the template's conventions `[WP-25]`: metres, Y up, the
//! subject facing +Z with its left on +X, standing on `y = 0` and centred on the vertical axis.
//!
//! [`ScanMesh::parse_obj`] reads a Wavefront OBJ (photogrammetry exports such as RealityCapture's)
//! through `humanart_io::obj`, welds it into one vertex per (position, texture coordinate) pair
//! ([`humanart_io::obj::Obj::to_indexed`]) and computes smooth area-weighted normals that ignore UV
//! seams (vertices split at a seam share their source position, [`ScanMesh::weld`]).
//! [`ScanMesh::normalise`] then applies the unit scale and the axis change the caller states
//! ([`Units`], [`Orientation`]) or guesses them ([`Units::guess`], [`Orientation::guess_forward`]).
//!
//! Limits `[ROB-01]`: the OBJ reader enforces its own [`humanart_io::Limits`]; on top of that a scan
//! has at most [`MAX_SCAN_VERTICES`] vertices and [`MAX_SCAN_TRIANGLES`] triangles, and every value
//! must be finite.

use humanart_io::Limits;
use humanart_io::obj::{IndexedMesh, Obj};
use humanart_math::{Vec2, Vec3};

use super::{CaptureError, Result};

/// Most vertices of one scan (after welding).
pub const MAX_SCAN_VERTICES: usize = 1 << 24;
/// Most triangles of one scan.
pub const MAX_SCAN_TRIANGLES: usize = 1 << 25;

/// The length unit of a scan file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Units {
    /// Metres (scale 1).
    Metres,
    /// Centimetres (scale 0.01).
    Centimetres,
    /// Millimetres (scale 0.001).
    Millimetres,
}

impl Units {
    /// Every unit, in the order of [`Units::NAMES`].
    pub const ALL: [Self; 3] = [Self::Metres, Self::Centimetres, Self::Millimetres];
    /// The command-line names of [`Units::ALL`].
    pub const NAMES: [&'static str; 3] = ["m", "cm", "mm"];

    /// Metres per file unit.
    pub const fn scale(self) -> f32 {
        match self {
            Self::Metres => 1.0,
            Self::Centimetres => 0.01,
            Self::Millimetres => 0.001,
        }
    }

    /// The unit with a command-line name (`m`, `cm`, `mm`).
    pub fn from_name(name: &str) -> Option<Self> {
        Self::NAMES.iter().position(|n| *n == name).and_then(|i| Self::ALL.get(i).copied())
    }

    /// The unit in which `extent` (the largest side of the scan's bounding box, file units) is the
    /// size of a human body part between 5 cm and 3 m: a whole body, a head or a hand. `None` when
    /// no unit fits.
    pub fn guess(extent: f32) -> Option<Self> {
        Self::ALL.into_iter().find(|u| (0.05..=3.0).contains(&(extent * u.scale())))
    }
}

/// A signed coordinate axis of a scan file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// +X.
    PosX,
    /// −X.
    NegX,
    /// +Y.
    PosY,
    /// −Y.
    NegY,
    /// +Z.
    PosZ,
    /// −Z.
    NegZ,
}

impl Axis {
    /// Every axis, in the order of [`Axis::NAMES`].
    pub const ALL: [Self; 6] = [Self::PosX, Self::NegX, Self::PosY, Self::NegY, Self::PosZ, Self::NegZ];
    /// The command-line names of [`Axis::ALL`].
    pub const NAMES: [&'static str; 6] = ["+x", "-x", "+y", "-y", "+z", "-z"];

    /// The unit vector.
    pub const fn vector(self) -> Vec3 {
        match self {
            Self::PosX => Vec3::new(1.0, 0.0, 0.0),
            Self::NegX => Vec3::new(-1.0, 0.0, 0.0),
            Self::PosY => Vec3::new(0.0, 1.0, 0.0),
            Self::NegY => Vec3::new(0.0, -1.0, 0.0),
            Self::PosZ => Vec3::new(0.0, 0.0, 1.0),
            Self::NegZ => Vec3::new(0.0, 0.0, -1.0),
        }
    }

    /// The axis with a command-line name (`+x`, `-z`, ...).
    pub fn from_name(name: &str) -> Option<Self> {
        Self::NAMES.iter().position(|n| *n == name).and_then(|i| Self::ALL.get(i).copied())
    }

    /// The coordinate index (0 for X, 1 for Y, 2 for Z).
    pub const fn index(self) -> usize {
        match self {
            Self::PosX | Self::NegX => 0,
            Self::PosY | Self::NegY => 1,
            Self::PosZ | Self::NegZ => 2,
        }
    }
}

/// Where up and forward (the direction the subject faces) point in a scan file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Orientation {
    /// The file's up axis.
    pub up: Axis,
    /// The file's forward axis.
    pub forward: Axis,
}

impl Orientation {
    /// Our own convention: Y up, facing +Z.
    pub const OURS: Self = Self { up: Axis::PosY, forward: Axis::PosZ };

    /// The rotation into our convention as three rows (`p' = (r0·p, r1·p, r2·p)`): the subject's
    /// left `up × forward` becomes +X, up +Y, forward +Z. A proper rotation (no mirroring).
    ///
    /// # Errors
    /// [`CaptureError::Invalid`] when up and forward lie on the same axis.
    pub fn rotation(self) -> Result<[Vec3; 3]> {
        if self.up.index() == self.forward.index() {
            return Err(CaptureError::invalid("up and forward must be different axes"));
        }
        let (up, forward) = (self.up.vector(), self.forward.vector());
        Ok([up.cross(forward), up, forward])
    }

    /// Guesses the forward axis of a standing full-body scan with the given up axis: the toes
    /// reach further forward from the shins than the heels reach back, while the feet spread
    /// about equally to both sides. Measures, along each signed horizontal axis, how far the lowest
    /// 4 % of the height reaches past the centre of the shins (15 to 25 % of the height) and picks
    /// the axis whose reach exceeds its opposite's by the most. `None` when the scan has no such
    /// bands (a head or a hand scan).
    pub fn guess_forward(positions: &[Vec3], up: Axis) -> Option<Axis> {
        let u = up.vector();
        let height = |p: Vec3| p.dot(u);
        let (lo, hi) =
            positions.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &p| (lo.min(height(p)), hi.max(height(p))));
        let h = hi - lo;
        if !(h.is_finite() && h > 0.0) {
            return None;
        }
        let (mut shin_sum, mut shin_n) = (Vec3::ZERO, 0_usize);
        for &p in positions {
            let t = (height(p) - lo) / h;
            if (0.15..0.25).contains(&t) {
                shin_sum += p;
                shin_n += 1;
            }
        }
        if shin_n == 0 {
            return None;
        }
        let shin = shin_sum / shin_n as f32;
        // Largest reach along each of the six signed axes (those along `up` stay unused).
        let mut reach = [f32::MIN; 6];
        let mut feet = 0_usize;
        for &p in positions {
            if (height(p) - lo) / h < 0.04 {
                feet += 1;
                for (r, a) in reach.iter_mut().zip(Axis::ALL) {
                    *r = r.max((p - shin).dot(a.vector()));
                }
            }
        }
        if feet == 0 {
            return None;
        }
        // Axis::ALL pairs each axis with its opposite: (+x, -x), (+y, -y), (+z, -z).
        let mut best: Option<(f32, Axis)> = None;
        for pair in 0..3 {
            let (pos, neg) = (Axis::ALL[pair * 2], Axis::ALL[pair * 2 + 1]);
            if pos.index() == up.index() {
                continue;
            }
            let lead = reach[pair * 2] - reach[pair * 2 + 1];
            let (excess, axis) = if lead >= 0.0 { (lead, pos) } else { (-lead, neg) };
            if best.is_none_or(|(e, _)| excess > e) {
                best = Some((excess, axis));
            }
        }
        best.map(|(_, a)| a)
    }
}

/// What [`ScanMesh::normalise`] applied: `p' = rotation · (scale · p) + translation`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Normalisation {
    /// Metres per file unit.
    pub scale: f32,
    /// The rotation rows ([`Orientation::rotation`]).
    pub rotation: [Vec3; 3],
    /// The translation (metres) that puts the lowest point on `y = 0` and the bounding box's
    /// centre on the vertical axis.
    pub translation: Vec3,
}

impl Normalisation {
    /// Applies the normalisation to a point in file units.
    pub fn apply(&self, p: Vec3) -> Vec3 {
        let s = p * self.scale;
        Vec3::new(self.rotation[0].dot(s), self.rotation[1].dot(s), self.rotation[2].dot(s)) + self.translation
    }

    /// The point in file units that [`Normalisation::apply`] takes to `p` (the rotation rows are
    /// orthonormal, so its inverse is its transpose).
    pub fn invert(&self, p: Vec3) -> Vec3 {
        let q = p - self.translation;
        let [r0, r1, r2] = self.rotation;
        (r0 * q.x + r1 * q.y + r2 * q.z) / self.scale
    }
}

/// A scan as an indexed triangle mesh (structure of arrays). See the module documentation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScanMesh {
    /// Vertex positions.
    pub positions: Vec<Vec3>,
    /// Unit vertex normals, smooth across UV seams.
    pub normals: Vec<Vec3>,
    /// Texture coordinates (origin top-left, V down); empty when the scan has none.
    pub uvs: Vec<Vec2>,
    /// Per vertex, the index of its source position: vertices split at a UV seam share it.
    pub weld: Vec<u32>,
    /// Three vertex indices per triangle, counter-clockwise seen from outside.
    pub indices: Vec<u32>,
}

impl ScanMesh {
    /// Parses an OBJ file held in memory (see the module documentation).
    ///
    /// # Errors
    /// [`CaptureError::Io`] for a malformed OBJ or one beyond `limits`; the checks of
    /// [`ScanMesh::from_indexed`].
    pub fn parse_obj(bytes: &[u8], limits: &Limits) -> Result<Self> {
        let obj = Obj::parse(bytes, limits)?;
        if obj.face_count() == 0 {
            return Err(CaptureError::invalid("the OBJ file has no faces"));
        }
        let mesh = obj.to_indexed()?;
        Self::from_indexed(mesh)
    }

    /// The scan of a welded OBJ mesh: checks the limits and finite values, drops the OBJ's own
    /// normals and computes smooth ones ([`ScanMesh::recompute_normals`]).
    ///
    /// # Errors
    /// [`CaptureError::Limit`] beyond [`MAX_SCAN_VERTICES`] or [`MAX_SCAN_TRIANGLES`];
    /// [`CaptureError::Invalid`] for non-finite values or indices out of range.
    pub fn from_indexed(mesh: IndexedMesh) -> Result<Self> {
        let n = mesh.positions.len();
        if n > MAX_SCAN_VERTICES {
            return Err(CaptureError::limit("scan vertex count", MAX_SCAN_VERTICES, n));
        }
        let triangles = mesh.indices.len() / 3;
        if triangles > MAX_SCAN_TRIANGLES {
            return Err(CaptureError::limit("scan triangle count", MAX_SCAN_TRIANGLES, triangles));
        }
        let positions: Vec<Vec3> = mesh.positions.iter().map(|&p| Vec3::from_array(p)).collect();
        let uvs: Vec<Vec2> = mesh.tex_coords.iter().map(|&t| Vec2::from_array(t)).collect();
        let mut scan = Self { positions, normals: Vec::new(), uvs, weld: mesh.source_positions, indices: mesh.indices };
        scan.validate()?;
        scan.recompute_normals();
        Ok(scan)
    }

    /// Checks the streams: lengths, whole triangles, indices in range, finite values, the limits.
    ///
    /// # Errors
    /// [`CaptureError::Invalid`] or [`CaptureError::Limit`] naming the first problem.
    pub fn validate(&self) -> Result<()> {
        let n = self.positions.len();
        if n > MAX_SCAN_VERTICES {
            return Err(CaptureError::limit("scan vertex count", MAX_SCAN_VERTICES, n));
        }
        if self.indices.len() / 3 > MAX_SCAN_TRIANGLES {
            return Err(CaptureError::limit("scan triangle count", MAX_SCAN_TRIANGLES, self.indices.len() / 3));
        }
        if !self.indices.len().is_multiple_of(3) {
            return Err(CaptureError::invalid("the index count is not a multiple of 3"));
        }
        if self.weld.len() != n || (!self.uvs.is_empty() && self.uvs.len() != n) {
            return Err(CaptureError::invalid("the vertex streams differ in length"));
        }
        if !self.normals.is_empty() && self.normals.len() != n {
            return Err(CaptureError::invalid("the vertex streams differ in length"));
        }
        if self.indices.iter().any(|&i| i as usize >= n) {
            return Err(CaptureError::invalid("a triangle index is past the last vertex"));
        }
        if !self.positions.iter().all(|p| p.is_finite()) || !self.uvs.iter().all(|t| t.is_finite()) {
            return Err(CaptureError::invalid("a vertex attribute is not finite"));
        }
        Ok(())
    }

    /// Appends `other` (the parts of one atlas, which share a coordinate frame): its vertices,
    /// triangles and source positions follow this mesh's. Texture coordinates are kept only when
    /// both meshes have them.
    ///
    /// # Errors
    /// [`CaptureError::Limit`] when the result exceeds [`MAX_SCAN_VERTICES`] or
    /// [`MAX_SCAN_TRIANGLES`].
    pub fn append(&mut self, other: &Self) -> Result<()> {
        let n = self.positions.len() + other.positions.len();
        if n > MAX_SCAN_VERTICES {
            return Err(CaptureError::limit("scan vertex count", MAX_SCAN_VERTICES, n));
        }
        let t = self.triangle_count() + other.triangle_count();
        if t > MAX_SCAN_TRIANGLES {
            return Err(CaptureError::limit("scan triangle count", MAX_SCAN_TRIANGLES, t));
        }
        let first = self.positions.len() as u32;
        let sources = self.weld.iter().map(|&w| w + 1).max().unwrap_or(0);
        let keep_uvs = (self.positions.is_empty() || !self.uvs.is_empty()) && !other.uvs.is_empty();
        if keep_uvs {
            self.uvs.extend_from_slice(&other.uvs);
        } else {
            self.uvs.clear();
        }
        self.positions.extend_from_slice(&other.positions);
        self.normals.extend_from_slice(&other.normals);
        self.weld.extend(other.weld.iter().map(|&w| w + sources));
        self.indices.extend(other.indices.iter().map(|&i| i + first));
        Ok(())
    }

    /// Per triangle, whether it lies on the rim of a hole or the open border of the scan: one of
    /// its edges (between source positions, so UV seams do not count) belongs to no other
    /// triangle.
    pub fn boundary_triangles(&self) -> Vec<bool> {
        let key = |v: u32| self.weld.get(v as usize).copied().unwrap_or(v);
        let mut edges: Vec<(u32, u32, u32)> = Vec::with_capacity(self.indices.len());
        for (t, tri) in self.indices.as_chunks::<3>().0.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (key(tri[k]), key(tri[(k + 1) % 3]));
                edges.push((a.min(b), a.max(b), t as u32));
            }
        }
        edges.sort_unstable();
        let mut rim = vec![false; self.triangle_count()];
        let mut i = 0;
        while i < edges.len() {
            let mut j = i + 1;
            while j < edges.len() && edges[j].0 == edges[i].0 && edges[j].1 == edges[i].1 {
                j += 1;
            }
            if j - i == 1
                && let Some(r) = rim.get_mut(edges[i].2 as usize)
            {
                *r = true;
            }
            i = j;
        }
        rim
    }

    /// The number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// The corners of triangle `t`, if it exists.
    pub fn triangle(&self, t: usize) -> Option<[u32; 3]> {
        let s = self.indices.get(t * 3..t * 3 + 3)?;
        Some([s[0], s[1], s[2]])
    }

    /// Recomputes smooth, area-weighted unit normals, summed per source position so UV seams do
    /// not show. Vertices of degenerate neighbourhoods get +Y.
    pub fn recompute_normals(&mut self) {
        let sources = self.weld.iter().map(|&w| w as usize + 1).max().unwrap_or(0);
        let mut sum = vec![Vec3::ZERO; sources];
        for t in self.indices.as_chunks::<3>().0 {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let (Some(&pa), Some(&pb), Some(&pc)) =
                (self.positions.get(a), self.positions.get(b), self.positions.get(c))
            else {
                continue;
            };
            let face = (pb - pa).cross(pc - pa);
            for v in [a, b, c] {
                if let Some(s) = self.weld.get(v).and_then(|&w| sum.get_mut(w as usize)) {
                    *s += face;
                }
            }
        }
        self.normals = self
            .weld
            .iter()
            .map(|&w| sum.get(w as usize).copied().unwrap_or(Vec3::ZERO).normalize_or(Vec3::Y))
            .collect();
    }

    /// The lowest and highest corner of the bounding box (`(+MAX, −MAX)` when empty).
    pub fn extent(&self) -> (Vec3, Vec3) {
        self.positions
            .iter()
            .fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), &p| (lo.min(p), hi.max(p)))
    }

    /// Scales and rotates the scan into our convention (`units` and `orientation` of the file),
    /// then puts its lowest point on `y = 0` and its bounding box's centre on the vertical axis.
    /// Normals are rotated with the positions.
    ///
    /// # Errors
    /// [`CaptureError::Invalid`] for an orientation with up and forward on one axis, or an empty
    /// scan.
    pub fn normalise(&mut self, units: Units, orientation: Orientation) -> Result<Normalisation> {
        if self.positions.is_empty() {
            return Err(CaptureError::invalid("the scan has no vertices"));
        }
        let rotation = orientation.rotation()?;
        let mut n = Normalisation { scale: units.scale(), rotation, translation: Vec3::ZERO };
        for p in &mut self.positions {
            *p = n.apply(*p);
        }
        for v in &mut self.normals {
            *v = Vec3::new(rotation[0].dot(*v), rotation[1].dot(*v), rotation[2].dot(*v));
        }
        let (lo, hi) = self.extent();
        let centre = (lo + hi) * 0.5;
        let translation = Vec3::new(-centre.x, -lo.y, -centre.z);
        for p in &mut self.positions {
            *p += translation;
        }
        n.translation = translation;
        Ok(n)
    }

    /// Millimetres of surface per UV unit: `sqrt(surface area / UV area)` over the triangles with
    /// texture coordinates, in millimetres (positions in metres). `None` without UVs or UV area.
    pub fn mm_per_uv(&self) -> Option<f32> {
        if self.uvs.is_empty() {
            return None;
        }
        let (mut area, mut uv_area) = (0.0_f64, 0.0_f64);
        for t in self.indices.as_chunks::<3>().0 {
            let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let (Some(&pa), Some(&pb), Some(&pc)) =
                (self.positions.get(a), self.positions.get(b), self.positions.get(c))
            else {
                continue;
            };
            let (Some(&ua), Some(&ub), Some(&uc)) = (self.uvs.get(a), self.uvs.get(b), self.uvs.get(c)) else {
                continue;
            };
            area += f64::from((pb - pa).cross(pc - pa).length()) * 0.5;
            let (e1, e2) = (ub - ua, uc - ua);
            uv_area += f64::from((e1.x * e2.y - e1.y * e2.x).abs()) * 0.5;
        }
        (uv_area > 0.0).then(|| ((area / uv_area).sqrt() * 1000.0) as f32)
    }

    /// The triangles whose three corners pass `keep`, with their vertices compacted (source
    /// positions renumbered in order of first use).
    #[must_use]
    pub fn crop(&self, keep: impl Fn(Vec3) -> bool) -> Self {
        let mut map = vec![u32::MAX; self.positions.len()];
        let mut weld_map = vec![u32::MAX; self.weld.iter().map(|&w| w as usize + 1).max().unwrap_or(0)];
        let mut out = Self::default();
        let has_uvs = self.uvs.len() == self.positions.len();
        let mut sources = 0_u32;
        for t in self.indices.as_chunks::<3>().0 {
            if !t.iter().all(|&v| self.positions.get(v as usize).is_some_and(|&p| keep(p))) {
                continue;
            }
            for &v in t {
                let i = v as usize;
                let Some(slot) = map.get_mut(i) else { continue };
                if *slot == u32::MAX {
                    *slot = out.positions.len() as u32;
                    out.positions.push(self.positions.get(i).copied().unwrap_or(Vec3::ZERO));
                    out.normals.push(self.normals.get(i).copied().unwrap_or(Vec3::Y));
                    if has_uvs {
                        out.uvs.push(self.uvs.get(i).copied().unwrap_or(Vec2::ZERO));
                    }
                    let source = self.weld.get(i).copied().unwrap_or(v);
                    let w = match weld_map.get_mut(source as usize) {
                        Some(w) if *w != u32::MAX => *w,
                        Some(w) => {
                            *w = sources;
                            sources += 1;
                            *w
                        }
                        None => {
                            sources += 1;
                            sources - 1
                        }
                    };
                    out.weld.push(w);
                }
                out.indices.push(*slot);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TETRA: &str = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nvt 0 0\nvt 1 0\nvt 0 1\nvt 1 1\n\
                         f 1/1 3/3 2/2\nf 1/1 2/2 4/4\nf 1/1 4/4 3/3\nf 2/2 3/3 4/1\n";

    #[test]
    fn wp25_obj_scan_welds_and_keeps_seams_smooth() {
        let scan = ScanMesh::parse_obj(TETRA.as_bytes(), &Limits::DEFAULT).unwrap();
        assert_eq!(scan.triangle_count(), 4);
        // Vertex 4 (index 3) appears with two texture coordinates, vertex 1 with one, ...
        assert_eq!(scan.positions.len(), 5);
        assert_eq!(scan.weld.len(), 5);
        // Both copies of the seam vertex share one normal.
        let copies: Vec<usize> = (0..5).filter(|&v| scan.weld[v] == 3).collect();
        assert_eq!(copies.len(), 2);
        assert_eq!(scan.normals[copies[0]], scan.normals[copies[1]]);
        // The normal of the corner at the origin points out of the tetrahedron (all negative).
        let origin = (0..5).find(|&v| scan.weld[v] == 0).unwrap();
        let n = scan.normals[origin];
        assert!(n.x < 0.0 && n.y < 0.0 && n.z < 0.0, "{n:?}");
        assert!((n.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn wp25_units_and_axes_parse_and_guess() {
        assert_eq!(Units::from_name("mm"), Some(Units::Millimetres));
        assert_eq!(Units::from_name("inch"), None);
        assert_eq!(Units::guess(1.8), Some(Units::Metres));
        assert_eq!(Units::guess(172.0), Some(Units::Centimetres));
        assert_eq!(Units::guess(1720.0), Some(Units::Millimetres));
        assert_eq!(Units::guess(1.0e7), None);
        assert_eq!(Axis::from_name("-z"), Some(Axis::NegZ));
        assert_eq!(Axis::from_name("z"), None);
        assert!(Orientation { up: Axis::PosZ, forward: Axis::NegZ }.rotation().is_err());
        // Ours is the identity.
        assert_eq!(Orientation::OURS.rotation().unwrap(), [Vec3::X, Vec3::Y, Vec3::Z]);
    }

    #[test]
    fn wp25_normalise_maps_z_up_millimetres_to_ours() {
        // A Z-up, -Y-forward scan in millimetres (BodyParts3D's frame): a point at the top front
        // right must land at +Y, +Z, -X.
        let mut scan = ScanMesh {
            positions: vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(-100.0, -50.0, 1700.0), Vec3::new(100.0, 50.0, 10.0)],
            normals: vec![Vec3::new(0.0, -1.0, 0.0); 3],
            uvs: Vec::new(),
            weld: vec![0, 1, 2],
            indices: vec![0, 1, 2],
        };
        let n = scan.normalise(Units::Millimetres, Orientation { up: Axis::PosZ, forward: Axis::NegY }).unwrap();
        let (lo, hi) = scan.extent();
        assert!((lo.y).abs() < 1e-6 && (hi.y - 1.7).abs() < 1e-5, "{lo:?} {hi:?}");
        assert!((lo.x + hi.x).abs() < 1e-6 && (lo.z + hi.z).abs() < 1e-6);
        // Forward (-Y in the file) is +Z now; the subject's left is up x forward = +X in both frames.
        assert!(scan.normals[0].abs_diff_eq(Vec3::Z, 1e-6), "{:?}", scan.normals[0]);
        let top = scan.positions[1];
        assert!(top.y > 1.6 && top.z > 0.0 && top.x < 0.0, "{top:?}");
        assert!(n.apply(Vec3::new(-100.0, -50.0, 1700.0)).abs_diff_eq(top, 1e-6));
        assert!(n.invert(top).abs_diff_eq(Vec3::new(-100.0, -50.0, 1700.0), 1e-3));
    }

    #[test]
    fn wp25_crop_keeps_whole_triangles_and_compacts() {
        let scan = ScanMesh::parse_obj(TETRA.as_bytes(), &Limits::DEFAULT).unwrap();
        // Dropping the apex (0, 0, 1) keeps only the base triangle.
        let base = scan.crop(|p| p.z < 0.5);
        assert_eq!(base.triangle_count(), 1);
        assert_eq!(base.positions.len(), 3);
        assert_eq!(base.uvs.len(), 3);
        assert!(base.validate().is_ok());
        assert!(base.positions.iter().all(|p| p.z == 0.0));
        let all = scan.crop(|_| true);
        assert_eq!(all.triangle_count(), scan.triangle_count());
        assert!(all.validate().is_ok());
        assert_eq!(scan.crop(|_| false).triangle_count(), 0);
    }

    #[test]
    fn wp25_guess_forward_finds_the_toes() {
        // Two "legs" (vertical lines of points) and feet reaching forward along -Y in a Z-up frame.
        let mut pts = Vec::new();
        for i in 0..100 {
            let z = i as f32 * 0.01;
            pts.push(Vec3::new(-0.1, 0.0, z));
            pts.push(Vec3::new(0.1, 0.0, z));
        }
        pts.push(Vec3::new(0.1, -0.2, 0.0));
        pts.push(Vec3::new(-0.1, 0.05, 0.0));
        // A wide stance: the feet spread further sideways than the toes reach, but symmetrically.
        pts.push(Vec3::new(0.35, 0.0, 0.0));
        pts.push(Vec3::new(-0.33, 0.0, 0.0));
        assert_eq!(Orientation::guess_forward(&pts, Axis::PosZ), Some(Axis::NegY));
        assert_eq!(Orientation::guess_forward(&[], Axis::PosZ), None);
    }

    #[test]
    fn wp25_append_offsets_indices_and_sources() {
        let mut a = ScanMesh::parse_obj(TETRA.as_bytes(), &Limits::DEFAULT).unwrap();
        let b = a.clone();
        a.append(&b).unwrap();
        a.validate().unwrap();
        assert_eq!((a.positions.len(), a.triangle_count()), (10, 8));
        assert_eq!(a.weld[5], b.weld[0] + 4);
        assert_eq!(a.indices[12], b.indices[0] + 5);
        assert_eq!(a.uvs.len(), 10);
        let mut empty = ScanMesh::default();
        empty.append(&b).unwrap();
        assert_eq!(empty, b);
    }

    #[test]
    fn wp25_boundary_triangles_ignore_uv_seams() {
        // The closed tetrahedron (with a UV seam) has no rim; without one face, the other three do.
        let mut scan = ScanMesh::parse_obj(TETRA.as_bytes(), &Limits::DEFAULT).unwrap();
        assert_eq!(scan.boundary_triangles(), vec![false; 4]);
        scan.indices.truncate(9);
        assert_eq!(scan.boundary_triangles(), vec![true; 3]);
    }

    #[test]
    fn wp25_mm_per_uv_of_a_unit_square() {
        let scan = ScanMesh {
            positions: vec![Vec3::ZERO, Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.0, 0.1, 0.0)],
            normals: Vec::new(),
            uvs: vec![Vec2::ZERO, Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)],
            weld: vec![0, 1, 2],
            indices: vec![0, 1, 2],
        };
        assert!((scan.mm_per_uv().unwrap() - 100.0).abs() < 1e-3);
    }

    #[test]
    fn wp25_hostile_obj_scans_fail_cleanly() {
        // [ROB-01] Mutations of valid OBJ scans never panic or hang: they parse into a valid scan or
        // return an error.
        let corpus = vec![TETRA.as_bytes().to_vec(), b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n".to_vec()];
        humanart_core::fuzz::run("scan-obj", &corpus, 3000, 0x5ca7_da7a, std::time::Duration::from_secs(60), |b| {
            if let Ok(scan) = ScanMesh::parse_obj(b, &Limits::FUZZ) {
                scan.validate().unwrap();
            }
        });
        assert!(ScanMesh::parse_obj(b"v 0 0 0\n", &Limits::DEFAULT).is_err());
    }
}
