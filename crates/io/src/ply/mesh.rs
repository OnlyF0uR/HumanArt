//! The usual mesh inside a PLY file, by the conventional property names.

use super::{Column, Element, Encoding, FORMAT, Ply, Property, PropertyData, ScalarType};
use crate::error::{Error, Result};

/// A polygon mesh or point cloud read from (or to be written to) PLY.
///
/// Positions are required; the other vertex arrays are either empty or one entry per position.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlyMesh {
    /// Vertex positions (`x`, `y`, `z`).
    pub positions: Vec<[f32; 3]>,
    /// Vertex normals (`nx`, `ny`, `nz`).
    pub normals: Vec<[f32; 3]>,
    /// Vertex colours, RGBA in 0..1 (`red`, `green`, `blue`, optionally `alpha`, or the `diffuse_`
    /// forms): integer values are divided by their type's maximum, float values kept; a missing alpha
    /// is 1.
    pub colors: Vec<[f32; 4]>,
    /// Texture coordinates (`u`/`v`, `s`/`t`, `texture_u`/`texture_v` or `texture_s`/`texture_t`),
    /// converted to the project convention (origin top left, V down: `1 − v`).
    pub tex_coords: Vec<[f32; 2]>,
    /// For each face, the index one past its last corner in [`PlyMesh::face_vertices`].
    pub face_ends: Vec<u32>,
    /// The vertex indices of every face, face after face.
    pub face_vertices: Vec<u32>,
}

/// An [`Error::Content`] about the data as a whole, for what only shows once it is read.
const fn bad_mesh(what: &'static str) -> Error {
    Error::Content { format: FORMAT, item: None, what }
}

/// An [`Error::Content`] about face `index`.
const fn bad_face(index: usize, what: &'static str) -> Error {
    Error::Content { format: FORMAT, item: Some(("face", index)), what }
}

/// The scalar column of a property, if the element has it.
fn scalar<'a>(element: &'a Element, name: &str) -> Result<Option<&'a Column>> {
    match element.property(name) {
        None => Ok(None),
        Some(Property { data: PropertyData::Scalar(column), .. }) => Ok(Some(column)),
        Some(_) => Err(bad_mesh("a vertex attribute is a list")),
    }
}

/// The first of several spellings of a group of scalar properties that the element has completely.
fn columns<'a, const N: usize>(element: &'a Element, spellings: &[[&str; N]]) -> Result<Option<[&'a Column; N]>> {
    for names in spellings {
        let mut found = Vec::with_capacity(N);
        for name in names {
            match scalar(element, name)? {
                Some(column) => found.push(column),
                None => break,
            }
        }
        if let Ok(all) = <[&Column; N]>::try_from(found) {
            return Ok(Some(all));
        }
    }
    Ok(None)
}

/// A colour value in 0..1: integers divided by their type's maximum.
fn unit(column: &Column, i: usize) -> f32 {
    let scale = match column.scalar_type() {
        ScalarType::I8 => f64::from(i8::MAX),
        ScalarType::U8 => f64::from(u8::MAX),
        ScalarType::I16 => f64::from(i16::MAX),
        ScalarType::U16 => f64::from(u16::MAX),
        ScalarType::I32 => f64::from(i32::MAX),
        ScalarType::U32 => f64::from(u32::MAX),
        ScalarType::F32 | ScalarType::F64 => 1.0,
    };
    (column.get(i).unwrap_or(0.0) / scale) as f32
}

/// Value `i` of a column as `f32` (validated lengths make the fallback unreachable).
fn value(column: &Column, i: usize) -> f32 {
    column.get(i).unwrap_or(0.0) as f32
}

impl Ply {
    /// Extracts the mesh: the element `vertex` (positions `x`, `y`, `z` required, normals, colours
    /// and texture coordinates when all their components are present; see [`PlyMesh`]) and the
    /// element `face`, if any, with its list `vertex_indices` (or `vertex_index`). Without faces the
    /// result is a point cloud. Values are converted as they are, NaN included (organised scans use
    /// it for missing points).
    ///
    /// Errors ([`Error::Content`]; the ones about one face name its index): no `vertex` element or
    /// no positions, a vertex attribute or the face indices stored the wrong way (a list, a float
    /// list, a negative or out-of-range index), a face with fewer than 3 vertices, a mesh over
    /// 2^32 − 1 face corners.
    pub fn to_mesh(&self) -> Result<PlyMesh> {
        let vertex = self.element("vertex").ok_or(bad_mesh("no vertex element"))?;
        let n = vertex.count;
        let check = |columns: &[&Column]| columns.iter().all(|c| c.len() == n);
        let [x, y, z] = columns(vertex, &[["x", "y", "z"]])?.ok_or(bad_mesh("no vertex positions"))?;
        let mut mesh = PlyMesh::default();
        if !check(&[x, y, z]) {
            return Err(bad_mesh("a vertex column has the wrong length"));
        }
        mesh.positions = (0..n).map(|i| [value(x, i), value(y, i), value(z, i)]).collect();
        if let Some([nx, ny, nz]) = columns(vertex, &[["nx", "ny", "nz"]])? {
            if !check(&[nx, ny, nz]) {
                return Err(bad_mesh("a vertex column has the wrong length"));
            }
            mesh.normals = (0..n).map(|i| [value(nx, i), value(ny, i), value(nz, i)]).collect();
        }
        if let Some([r, g, b]) =
            columns(vertex, &[["red", "green", "blue"], ["diffuse_red", "diffuse_green", "diffuse_blue"]])?
        {
            let alpha = columns(vertex, &[["alpha"], ["diffuse_alpha"]])?.map(|[a]| a);
            if !check(&[r, g, b]) || alpha.is_some_and(|a| a.len() != n) {
                return Err(bad_mesh("a vertex column has the wrong length"));
            }
            mesh.colors =
                (0..n).map(|i| [unit(r, i), unit(g, i), unit(b, i), alpha.map_or(1.0, |a| unit(a, i))]).collect();
        }
        let tex = [["u", "v"], ["s", "t"], ["texture_u", "texture_v"], ["texture_s", "texture_t"]];
        if let Some([u, v]) = columns(vertex, &tex)? {
            if !check(&[u, v]) {
                return Err(bad_mesh("a vertex column has the wrong length"));
            }
            mesh.tex_coords = (0..n).map(|i| [value(u, i), (1.0 - v.get(i).unwrap_or(0.0)) as f32]).collect();
        }

        let Some(face) = self.element("face") else { return Ok(mesh) };
        let Some(indices) = face.property("vertex_indices").or_else(|| face.property("vertex_index")) else {
            return Err(bad_mesh("face element without vertex indices"));
        };
        let PropertyData::List { ends, values, .. } = &indices.data else {
            return Err(bad_mesh("face vertex indices are not a list"));
        };
        if !values.scalar_type().is_integer() {
            return Err(bad_mesh("face vertex indices are not integers"));
        }
        if values.len() > u32::MAX as usize {
            return Err(bad_mesh("too many face corners"));
        }
        let mut previous = 0_u32;
        for (face_index, &end) in ends.iter().enumerate() {
            if end < previous.saturating_add(3) || end as usize > values.len() {
                return Err(bad_face(face_index, "a face has fewer than 3 vertices"));
            }
            previous = end;
        }
        mesh.face_vertices.reserve(values.len());
        for i in 0..values.len() {
            let index = values.get(i).unwrap_or(-1.0);
            if index < 0.0 || index >= n as f64 {
                // The face whose corner this is: the first one ending past corner `i`.
                let face_index = ends.partition_point(|&end| end as usize <= i);
                return Err(bad_face(face_index, "face vertex index out of range"));
            }
            mesh.face_vertices.push(index as u32);
        }
        if previous as usize != values.len() {
            return Err(bad_mesh("face lists are inconsistent"));
        }
        mesh.face_ends.clone_from(ends);
        Ok(mesh)
    }
}

impl PlyMesh {
    /// The number of faces.
    #[must_use]
    pub fn face_count(&self) -> usize {
        self.face_ends.len()
    }

    /// The vertex indices of face `index`, or `None` past the last face.
    #[must_use]
    pub fn face(&self, index: usize) -> Option<&[u32]> {
        let end = *self.face_ends.get(index)? as usize;
        let start = if index == 0 { 0 } else { *self.face_ends.get(index - 1)? as usize };
        self.face_vertices.get(start..end)
    }

    /// Builds a PLY file of the mesh: an element `vertex` with `float x y z`, then (when set) `float
    /// nx ny nz`, `uchar red green blue` (and `alpha` when any colour is not opaque; values clamped to
    /// 0..1 and rounded to 1/255) and `float u v` (V flipped back), and an element `face` with `list
    /// uchar uint vertex_indices` (`list uint uint` when a face has more than 255 vertices).
    ///
    /// Errors: vertex arrays whose lengths differ from the position count, face ends that do not
    /// describe faces of at least 3 vertices, a face index past the last vertex.
    pub fn to_ply(&self, encoding: Encoding) -> Result<Ply> {
        let refuse = |what| Err(Error::Encode { format: FORMAT, what });
        let n = self.positions.len();
        if [self.normals.len(), self.colors.len(), self.tex_coords.len()].iter().any(|&len| len != 0 && len != n) {
            return refuse("a vertex array's length differs from the position count");
        }
        if self.face_vertices.iter().any(|&v| v as usize >= n) {
            return refuse("face vertex index out of range");
        }
        let mut previous = 0_u32;
        let mut longest = 0;
        for &end in &self.face_ends {
            if end < previous.saturating_add(3) {
                return refuse("a face has fewer than 3 vertices");
            }
            longest = longest.max(end - previous);
            previous = end;
        }
        if previous as usize != self.face_vertices.len() {
            return refuse("face ends are inconsistent");
        }

        let scalar = |name: &str, column: Column| Property { name: name.into(), data: PropertyData::Scalar(column) };
        let floats = |f: &dyn Fn(usize) -> f32| Column::F32((0..n).map(f).collect());
        let mut properties = vec![
            scalar("x", floats(&|i| self.positions[i][0])),
            scalar("y", floats(&|i| self.positions[i][1])),
            scalar("z", floats(&|i| self.positions[i][2])),
        ];
        if !self.normals.is_empty() {
            for (k, name) in ["nx", "ny", "nz"].into_iter().enumerate() {
                properties.push(scalar(name, floats(&|i| self.normals[i][k])));
            }
        }
        if !self.colors.is_empty() {
            let channels = if self.colors.iter().any(|c| c[3] != 1.0) { 4 } else { 3 };
            for (k, name) in ["red", "green", "blue", "alpha"].into_iter().enumerate().take(channels) {
                let bytes = self.colors.iter().map(|c| (c[k].clamp(0.0, 1.0) * 255.0).round() as u8).collect();
                properties.push(scalar(name, Column::U8(bytes)));
            }
        }
        if !self.tex_coords.is_empty() {
            properties.push(scalar("u", floats(&|i| self.tex_coords[i][0])));
            properties.push(scalar("v", floats(&|i| (1.0 - f64::from(self.tex_coords[i][1])) as f32)));
        }
        let mut elements = vec![Element { name: "vertex".into(), count: n, properties }];
        if !self.face_ends.is_empty() {
            let count_type = if longest > u32::from(u8::MAX) { ScalarType::U32 } else { ScalarType::U8 };
            let data = PropertyData::List {
                count_type,
                ends: self.face_ends.clone(),
                values: Column::U32(self.face_vertices.clone()),
            };
            elements.push(Element {
                name: "face".into(),
                count: self.face_ends.len(),
                properties: vec![Property { name: "vertex_indices".into(), data }],
            });
        }
        Ok(Ply { encoding, comments: Vec::new(), obj_info: Vec::new(), elements })
    }
}
