//! Wavefront OBJ meshes and MTL material libraries, read and written `[IO-11]` `[DATA-11]`.
//!
//! [`Obj`] keeps a file's data as the format has it: separate position, texture-coordinate and
//! normal arrays, polygon faces whose corners index each array independently, polylines, and the
//! object / group / material / smoothing-group state as runs of faces ([`FaceGroup`]). Arrays are
//! structure-of-arrays `[PERF-R2]`. [`Obj::to_indexed`] welds the corners into the single-index
//! triangle list a renderer or the model layer wants.
//!
//! # Reading
//!
//! [`ObjParser`] reads a file in chunks of any size (a scan of several gigabytes never has to be in
//! memory at once): every statement is parsed in place, without per-line allocation, as soon as its
//! line is complete. [`Obj::parse`] is the one-chunk shorthand. Read are:
//!
//! - `v x y z [w]` (the weight is dropped) and the common colour extension `v x y z r g b`;
//! - `vt u [v [w]]` (`w` dropped) and `vn x y z`;
//! - `f` / `fo` faces of 3 or more corners in the forms `v`, `v/vt`, `v//vn`, `v/vt/vn`, with
//!   absolute (1-based) and relative (negative) indices, which must refer to elements defined
//!   earlier in the file;
//! - `l` polylines (positions only; texture indices are accepted and dropped);
//! - `o`, `g`, `usemtl`, `mtllib`, `s`;
//! - `#` comments (a whole line, or the end of a numeric statement), `\` line continuations, CRLF
//!   line ends, a UTF-8 byte-order mark.
//!
//! Points (`p`), free-form attributes and display / render statements (`vp`, `deg`, `bmat`, `step`,
//! `parm`, `trim`, `hole`, `scrv`, `sp`, `end`, `con`, `mg`, `bevel`, `c_interp`, `d_interp`, `lod`,
//! `usemap`, `maplib`, `shadow_obj`, `trace_obj`, `ctech`, `stech`) and any other statement whose
//! keyword is a plain word are skipped. Free-form geometry (`cstype`, `curv`, `curv2`, `surf`) is
//! [`Error::Unsupported`]: dropping it would lose geometry silently.
//!
//! # Conventions
//!
//! Positions and normals are stored as written (OBJ has no unit; a scan importer normalises units
//! and orientation, `[DATA-11]`). Texture coordinates are converted to the project convention, origin
//! at the top left with V down: `v' = 1 − v` on reading, and back on writing. Vertex colours are
//! whatever the file holds (usually sRGB-encoded values in 0..1); [`Obj::colors`] is either empty or
//! one colour per position, uncoloured vertices of a partly coloured file reading as white.
//!
//! # Limits `[ROB-02]`
//!
//! Every array (positions, texture coordinates, normals, colours, corners, faces, polyline vertices)
//! holds at most [`Limits::max_elements`] entries, and together they hold at most
//! [`Limits::max_bytes`]; a logical line (continuations joined) is at most [`MAX_LINE_LEN`] bytes and a
//! face or polyline has at most [`MAX_FACE_CORNERS`] corners.

use std::collections::HashMap;
use std::ops::Range;

use crate::error::{Error, Format, Result};
use crate::image::Limits;

mod mtl;
mod write;

pub use mtl::{Material, TextureMap, parse_mtl, write_mtl};
pub use write::write;

/// The format every error of this module names.
const FORMAT: Format = Format::Obj;

/// The longest logical line, continuation lines joined, in bytes `[ROB-02]`.
pub const MAX_LINE_LEN: usize = 1 << 20;

/// The most corners of one face, and the most vertices of one polyline `[ROB-02]`.
pub const MAX_FACE_CORNERS: usize = 4096;

/// The UTF-8 byte-order mark some editors put at the start of a text file.
const BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

/// An [`Error::Invalid`] of this module.
const fn invalid(offset: usize, what: &'static str) -> Error {
    Error::Invalid { format: FORMAT, offset, what }
}

/// One corner of a face: indices (0-based) into [`Obj::positions`], [`Obj::tex_coords`] and
/// [`Obj::normals`]. A corner without a texture coordinate or normal holds [`Corner::NONE`] there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Corner {
    /// The position index.
    pub position: u32,
    /// The texture-coordinate index, or [`Corner::NONE`].
    pub tex_coord: u32,
    /// The normal index, or [`Corner::NONE`].
    pub normal: u32,
}

impl Corner {
    /// The index that stands for "this corner has no such attribute".
    pub const NONE: u32 = u32::MAX;

    /// A corner from its position index and optional texture-coordinate and normal indices.
    #[must_use]
    pub const fn new(position: u32, tex_coord: Option<u32>, normal: Option<u32>) -> Self {
        let tex_coord = match tex_coord {
            Some(i) => i,
            None => Self::NONE,
        };
        let normal = match normal {
            Some(i) => i,
            None => Self::NONE,
        };
        Self { position, tex_coord, normal }
    }

    /// The texture-coordinate index, if the corner has one.
    #[must_use]
    pub const fn tex_coord(self) -> Option<u32> {
        if self.tex_coord == Self::NONE { None } else { Some(self.tex_coord) }
    }

    /// The normal index, if the corner has one.
    #[must_use]
    pub const fn normal(self) -> Option<u32> {
        if self.normal == Self::NONE { None } else { Some(self.normal) }
    }
}

/// A run of consecutive faces that share an object name, group name, material and smoothing group.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FaceGroup {
    /// The object name (`o`); empty before the first `o` statement.
    pub object: String,
    /// The group name(s) (`g`, the rest of the line as written); empty when none is set. An `o`
    /// statement clears it.
    pub group: String,
    /// The material (`usemtl`), if one is set.
    pub material: Option<String>,
    /// The smoothing group (`s`); 0 means off.
    pub smoothing: u32,
    /// The faces of the run, as indices into the face list.
    pub faces: Range<u32>,
}

impl FaceGroup {
    /// Whether two runs have the same object, group, material and smoothing group.
    fn same_state(&self, other: &Self) -> bool {
        self.object == other.object
            && self.group == other.group
            && self.material == other.material
            && self.smoothing == other.smoothing
    }
}

/// The contents of an OBJ file. See the module documentation for what is read and how.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Obj {
    /// Vertex positions (`v`).
    pub positions: Vec<[f32; 3]>,
    /// Vertex colours (`v x y z r g b`): empty, or one per position.
    pub colors: Vec<[f32; 3]>,
    /// Texture coordinates (`vt`), origin top left, V down.
    pub tex_coords: Vec<[f32; 2]>,
    /// Normals (`vn`), as written (not renormalised).
    pub normals: Vec<[f32; 3]>,
    /// The corners of every face, face after face.
    pub corners: Vec<Corner>,
    /// For each face, the index one past its last corner in [`Obj::corners`].
    pub face_ends: Vec<u32>,
    /// Runs of faces with their object / group / material / smoothing state, in file order. A parsed
    /// file has runs that cover every face exactly once.
    pub groups: Vec<FaceGroup>,
    /// The position indices of every polyline (`l`), polyline after polyline.
    pub line_vertices: Vec<u32>,
    /// For each polyline, the index one past its last vertex in [`Obj::line_vertices`].
    pub line_ends: Vec<u32>,
    /// The material libraries named by `mtllib`, in order (file names relative to the OBJ file).
    pub material_libraries: Vec<String>,
}

/// A welded triangle mesh: one index per vertex for every attribute, made by [`Obj::to_indexed`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IndexedMesh {
    /// Vertex positions.
    pub positions: Vec<[f32; 3]>,
    /// Vertex normals: empty when no corner has one, else one per vertex (zero where a corner had
    /// none).
    pub normals: Vec<[f32; 3]>,
    /// Texture coordinates (origin top left, V down): empty when no corner has one, else one per
    /// vertex (zero where a corner had none).
    pub tex_coords: Vec<[f32; 2]>,
    /// Vertex colours: empty when the OBJ has none, else one per vertex.
    pub colors: Vec<[f32; 3]>,
    /// For each vertex, the index of the OBJ position it came from (a position used with several
    /// texture coordinates or normals becomes several vertices).
    pub source_positions: Vec<u32>,
    /// Three vertex indices per triangle.
    pub indices: Vec<u32>,
    /// For each run of [`Obj::groups`], its triangles as a range of triangle indices.
    pub group_triangles: Vec<Range<u32>>,
}

impl Obj {
    /// Parses a whole OBJ file held in memory; see [`ObjParser`] to read one in pieces.
    pub fn parse(data: &[u8], limits: &Limits) -> Result<Self> {
        let mut parser = ObjParser::new(*limits);
        parser.feed(data)?;
        parser.finish()
    }

    /// The number of faces.
    #[must_use]
    pub fn face_count(&self) -> usize {
        self.face_ends.len()
    }

    /// The corners of face `index`, or `None` past the last face (or for inconsistent face ends).
    #[must_use]
    pub fn face(&self, index: usize) -> Option<&[Corner]> {
        let end = *self.face_ends.get(index)? as usize;
        let start = if index == 0 { 0 } else { *self.face_ends.get(index - 1)? as usize };
        self.corners.get(start..end)
    }

    /// The faces in order. Stops at the first face whose range is inconsistent.
    pub fn faces(&self) -> impl Iterator<Item = &[Corner]> {
        (0..self.face_count()).map_while(|i| self.face(i))
    }

    /// The number of polylines.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.line_ends.len()
    }

    /// The position indices of polyline `index`, or `None` past the last one.
    #[must_use]
    pub fn line(&self, index: usize) -> Option<&[u32]> {
        let end = *self.line_ends.get(index)? as usize;
        let start = if index == 0 { 0 } else { *self.line_ends.get(index - 1)? as usize };
        self.line_vertices.get(start..end)
    }

    /// The number of triangles the faces split into: `corners − 2` per face.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.faces().map(|f| f.len().saturating_sub(2)).sum()
    }

    /// Calls `visit(face, triangle)` for every triangle of a fan split of every face: corners
    /// (0, i, i + 1). This is exact for the convex polygons exporters write; a concave polygon is
    /// split wrongly, as by most OBJ readers.
    pub fn for_each_triangle(&self, mut visit: impl FnMut(usize, [Corner; 3])) {
        for (index, face) in self.faces().enumerate() {
            if let Some((&first, rest)) = face.split_first() {
                for pair in rest.windows(2) {
                    visit(index, [first, pair[0], pair[1]]);
                }
            }
        }
    }

    /// Checks that the arrays are consistent: every index in range, faces of at least 3 and at most
    /// [`MAX_FACE_CORNERS`] corners, polylines of at least 2 vertices, runs in order inside the face
    /// list, colours empty or one per position, every value finite. A parsed file always passes;
    /// [`write`] and [`Obj::to_indexed`] call this on what they are given.
    pub fn validate(&self) -> Result<()> {
        let bad = |what: &'static str| Err(Error::Encode { format: FORMAT, what });
        if !self.colors.is_empty() && self.colors.len() != self.positions.len() {
            return bad("colour count differs from position count");
        }
        let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
        if !self.positions.iter().all(|p| finite(p))
            || !self.colors.iter().all(|c| finite(c))
            || !self.normals.iter().all(|n| finite(n))
            || !self.tex_coords.iter().all(|t| finite(t))
        {
            return bad("a value is not finite");
        }
        let (positions, tex_coords, normals) = (self.positions.len(), self.tex_coords.len(), self.normals.len());
        for c in &self.corners {
            if c.position as usize >= positions
                || c.tex_coord().is_some_and(|t| t as usize >= tex_coords)
                || c.normal().is_some_and(|n| n as usize >= normals)
            {
                return bad("corner index out of range");
            }
        }
        check_ends(&self.face_ends, self.corners.len(), 3, "face ends are inconsistent")?;
        check_ends(&self.line_ends, self.line_vertices.len(), 2, "polyline ends are inconsistent")?;
        if self.line_vertices.iter().any(|&v| v as usize >= positions) {
            return bad("polyline index out of range");
        }
        let mut next = 0;
        for group in &self.groups {
            if group.faces.start < next
                || group.faces.end < group.faces.start
                || group.faces.end as usize > self.face_count()
            {
                return bad("face groups are out of order");
            }
            next = group.faces.end;
        }
        Ok(())
    }

    /// Welds the faces into an indexed triangle mesh: corners with the same position, texture
    /// coordinate and normal indices become one vertex, and faces are split as by
    /// [`Obj::for_each_triangle`]. Fails only when [`Obj::validate`] does.
    pub fn to_indexed(&self) -> Result<IndexedMesh> {
        self.validate()?;
        let has_normals = self.corners.iter().any(|c| c.normal().is_some());
        let has_tex_coords = self.corners.iter().any(|c| c.tex_coord().is_some());
        let has_colors = !self.colors.is_empty();
        let mut mesh = IndexedMesh::default();
        let mut welded: HashMap<Corner, u32> = HashMap::new();
        // The first triangle of every face, plus the total at the end.
        let mut first_triangle = Vec::with_capacity(self.face_count() + 1);
        let mut triangles = 0_u32;
        for face in self.faces() {
            first_triangle.push(triangles);
            let mut vertex = |corner: Corner| -> u32 {
                *welded.entry(corner).or_insert_with(|| {
                    let at = mesh.positions.len() as u32;
                    let p = corner.position as usize;
                    // Indices were checked by validate().
                    mesh.positions.push(self.positions.get(p).copied().unwrap_or_default());
                    mesh.source_positions.push(corner.position);
                    if has_colors {
                        mesh.colors.push(self.colors.get(p).copied().unwrap_or_default());
                    }
                    if has_normals {
                        let n = corner.normal().and_then(|n| self.normals.get(n as usize));
                        mesh.normals.push(n.copied().unwrap_or_default());
                    }
                    if has_tex_coords {
                        let t = corner.tex_coord().and_then(|t| self.tex_coords.get(t as usize));
                        mesh.tex_coords.push(t.copied().unwrap_or_default());
                    }
                    at
                })
            };
            if let Some((&first, rest)) = face.split_first() {
                let first = vertex(first);
                let mut previous = None;
                for &corner in rest {
                    let here = vertex(corner);
                    if let Some(before) = previous {
                        mesh.indices.extend([first, before, here]);
                        triangles += 1;
                    }
                    previous = Some(here);
                }
            }
        }
        first_triangle.push(triangles);
        mesh.group_triangles = self
            .groups
            .iter()
            .map(|g| {
                let at = |face: u32| first_triangle.get(face as usize).copied().unwrap_or(triangles);
                at(g.faces.start)..at(g.faces.end)
            })
            .collect();
        Ok(mesh)
    }
}

/// Checks a list of end indices: non-decreasing steps of at least `min` and at most
/// [`MAX_FACE_CORNERS`], the last one equal to `total`.
fn check_ends(ends: &[u32], total: usize, min: u32, what: &'static str) -> Result<()> {
    let mut previous = 0_u32;
    for &end in ends {
        let size = end.saturating_sub(previous);
        if size < min || size as usize > MAX_FACE_CORNERS {
            return Err(Error::Encode { format: FORMAT, what });
        }
        previous = end;
    }
    if previous as usize != total {
        return Err(Error::Encode { format: FORMAT, what });
    }
    Ok(())
}

/// The bytes of `s` without leading and trailing ASCII whitespace.
fn trim(s: &[u8]) -> &[u8] {
    let start = s.iter().position(|b| !b.is_ascii_whitespace()).unwrap_or(s.len());
    let end = s.iter().rposition(|b| !b.is_ascii_whitespace()).map_or(start, |i| i + 1);
    s.get(start..end).unwrap_or(&[])
}

/// The whitespace-separated tokens of a numeric statement, up to a `#` comment.
fn number_tokens(rest: &[u8]) -> impl Iterator<Item = &[u8]> {
    rest.split(u8::is_ascii_whitespace).filter(|t| !t.is_empty()).take_while(|t| t.first() != Some(&b'#'))
}

/// Parses a decimal number token (any form Rust's float parser takes) and requires it to be finite.
fn parse_f32(token: &[u8], at: usize) -> Result<f32> {
    std::str::from_utf8(token)
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .ok_or_else(|| invalid(at, "bad number"))
}

/// Parses a signed decimal integer token of at most 18 digits (the range of every index and count).
fn parse_int(token: &[u8], at: usize) -> Result<i64> {
    let (negative, digits) = match token.split_first() {
        Some((b'-', rest)) => (true, rest),
        Some((b'+', rest)) => (false, rest),
        _ => (false, token),
    };
    if digits.is_empty() || digits.len() > 18 || !digits.iter().all(u8::is_ascii_digit) {
        return Err(invalid(at, "bad integer"));
    }
    let value = digits.iter().fold(0_i64, |v, &d| v * 10 + i64::from(d - b'0'));
    Ok(if negative { -value } else { value })
}

/// Resolves an OBJ index (1-based, or negative counting back from the newest element) against the
/// number of elements defined so far.
fn resolve(index: i64, count: usize, at: usize) -> Result<u32> {
    let count = count as i64;
    let zero_based = if index > 0 { index - 1 } else { count + index };
    if index == 0 || zero_based < 0 || zero_based >= count {
        return Err(invalid(at, "index refers to an element not defined before it"));
    }
    Ok(zero_based as u32)
}

/// A UTF-8 name from the rest of a statement.
fn name(rest: &[u8], at: usize) -> Result<String> {
    std::str::from_utf8(trim(rest)).map(str::to_owned).map_err(|_| invalid(at, "name is not UTF-8"))
}

/// Reads an OBJ file piece by piece: [`ObjParser::feed`] the bytes in chunks of any size, then
/// [`ObjParser::finish`]. Statements are parsed as soon as their line is complete; only an
/// unfinished line is buffered (at most [`MAX_LINE_LEN`] bytes).
#[derive(Debug)]
pub struct ObjParser {
    limits: Limits,
    obj: Obj,
    /// An unfinished line, or the joined part of a line continued with `\`.
    pending: Vec<u8>,
    /// The file offset where `pending` starts.
    pending_at: usize,
    /// How many bytes have been fed.
    consumed: usize,
    /// The bytes the arrays hold so far, for [`Limits::max_bytes`].
    bytes: u64,
    /// The object / group / material / smoothing state that the next face gets.
    state: FaceGroup,
}

impl ObjParser {
    /// A parser that enforces `limits`.
    #[must_use]
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
            obj: Obj::default(),
            pending: Vec::new(),
            pending_at: 0,
            consumed: 0,
            bytes: 0,
            state: FaceGroup::default(),
        }
    }

    /// Parses the next piece of the file. After an error the parser must not be used further.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<()> {
        let mut at = self.consumed;
        self.consumed += chunk.len();
        let mut rest = chunk;
        while let Some(newline) = rest.iter().position(|&b| b == b'\n') {
            let (line, after) = rest.split_at(newline);
            self.end_of_line(line, at)?;
            at += newline + 1;
            rest = after.get(1..).unwrap_or(&[]);
        }
        if !rest.is_empty() {
            if self.pending.is_empty() {
                self.pending_at = at;
            }
            self.push_pending(rest)?;
        }
        Ok(())
    }

    /// Parses what is left (a last line without a line end) and returns the file's contents.
    pub fn finish(mut self) -> Result<Obj> {
        if !self.pending.is_empty() {
            let mut line = std::mem::take(&mut self.pending);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            // A continuation at the very end of the file continues into nothing.
            if line.last() == Some(&b'\\') {
                line.pop();
            }
            self.statement(&line, self.pending_at)?;
        }
        Ok(self.obj)
    }

    /// Appends to the pending line, within [`MAX_LINE_LEN`].
    fn push_pending(&mut self, bytes: &[u8]) -> Result<()> {
        let len = self.pending.len() + bytes.len();
        if len > MAX_LINE_LEN {
            return Err(Error::Limit {
                format: FORMAT,
                what: "line length",
                limit: MAX_LINE_LEN as u64,
                got: len as u64,
            });
        }
        self.pending.extend_from_slice(bytes);
        Ok(())
    }

    /// Handles a line that ended with `\n` (the `\n` not included) and started at file offset `at`.
    fn end_of_line(&mut self, piece: &[u8], at: usize) -> Result<()> {
        if self.pending.is_empty() {
            if piece.len() > MAX_LINE_LEN {
                return Err(Error::Limit {
                    format: FORMAT,
                    what: "line length",
                    limit: MAX_LINE_LEN as u64,
                    got: piece.len() as u64,
                });
            }
            let line = piece.strip_suffix(b"\r").unwrap_or(piece);
            if let Some(continued) = line.strip_suffix(b"\\") {
                self.pending_at = at;
                self.push_pending(continued)?;
                return self.push_pending(b" ");
            }
            return self.statement(line, at);
        }
        self.push_pending(piece)?;
        let mut line = std::mem::take(&mut self.pending);
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        if line.last() == Some(&b'\\') {
            line.pop();
            line.push(b' ');
            self.pending = line;
            return Ok(());
        }
        let result = self.statement(&line, self.pending_at);
        // Keep the allocation for the next unfinished line.
        line.clear();
        self.pending = line;
        result
    }

    /// Accounts for `add` more entries of `item_bytes` bytes in an array that holds `len`.
    fn grow(&mut self, len: usize, add: usize, item_bytes: u64, what: &'static str) -> Result<()> {
        // Indices are 32-bit, with u32::MAX reserved for "none".
        let limit = self.limits.max_elements.min(u64::from(u32::MAX - 1));
        let count = (len + add) as u64;
        if count > limit {
            return Err(Error::Limit { format: FORMAT, what, limit, got: count });
        }
        let bytes = self.bytes + add as u64 * item_bytes;
        if bytes > self.limits.max_bytes {
            return Err(Error::Limit {
                format: FORMAT,
                what: "decoded mesh size",
                limit: self.limits.max_bytes,
                got: bytes,
            });
        }
        self.bytes = bytes;
        Ok(())
    }

    /// Parses one logical line that starts at file offset `at`.
    fn statement(&mut self, line: &[u8], at: usize) -> Result<()> {
        let line = if at == 0 { line.strip_prefix(BOM).unwrap_or(line) } else { line };
        let line = trim(line);
        let keyword_len = line.iter().position(u8::is_ascii_whitespace).unwrap_or(line.len());
        let (keyword, rest) = line.split_at(keyword_len);
        match keyword {
            b"" => Ok(()),
            _ if keyword.first() == Some(&b'#') => Ok(()),
            b"v" => self.vertex(rest, at),
            b"vt" => self.tex_coord(rest, at),
            b"vn" => self.normal(rest, at),
            b"f" | b"fo" => self.face(rest, at),
            b"l" => self.polyline(rest, at),
            b"o" => {
                self.state.object = name(rest, at)?;
                self.state.group.clear();
                Ok(())
            }
            b"g" => {
                self.state.group = name(rest, at)?;
                Ok(())
            }
            b"usemtl" => {
                let material = name(rest, at)?;
                self.state.material = (!material.is_empty()).then_some(material);
                Ok(())
            }
            b"mtllib" => {
                for file in rest.split(u8::is_ascii_whitespace).filter(|t| !t.is_empty()) {
                    let file = name(file, at)?;
                    self.grow(self.obj.material_libraries.len(), 1, file.len() as u64, "material library count")?;
                    self.obj.material_libraries.push(file);
                }
                Ok(())
            }
            b"s" => {
                let token = number_tokens(rest).next().ok_or_else(|| invalid(at, "smoothing group is missing"))?;
                self.state.smoothing = if token == b"off" {
                    0
                } else {
                    u32::try_from(parse_int(token, at)?).map_err(|_| invalid(at, "bad smoothing group"))?
                };
                Ok(())
            }
            b"cstype" | b"curv" | b"curv2" | b"surf" => {
                Err(Error::Unsupported { format: FORMAT, what: "free-form geometry" })
            }
            _ if keyword.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'_') => Ok(()),
            _ => Err(invalid(at, "not an OBJ statement")),
        }
    }

    /// `v x y z [w]` or `v x y z r g b`.
    fn vertex(&mut self, rest: &[u8], at: usize) -> Result<()> {
        let mut values = [0.0_f32; 6];
        let mut count = 0;
        for token in number_tokens(rest) {
            let slot = values.get_mut(count).ok_or_else(|| invalid(at, "too many vertex values"))?;
            *slot = parse_f32(token, at)?;
            count += 1;
        }
        let colored = match count {
            3 | 4 => false,
            6 => true,
            _ => return Err(invalid(at, "a vertex has 3 coordinates, a weight or 3 colour values")),
        };
        let len = self.obj.positions.len();
        self.grow(len, 1, 12, "vertex count")?;
        self.obj.positions.push([values[0], values[1], values[2]]);
        if colored || !self.obj.colors.is_empty() {
            let missing = len + 1 - self.obj.colors.len();
            self.grow(self.obj.colors.len(), missing, 12, "vertex colour count")?;
            self.obj.colors.resize(len, [1.0; 3]);
            self.obj.colors.push(if colored { [values[3], values[4], values[5]] } else { [1.0; 3] });
        }
        Ok(())
    }

    /// `vt u [v [w]]`, stored with V flipped.
    fn tex_coord(&mut self, rest: &[u8], at: usize) -> Result<()> {
        let mut values = [0.0_f32; 3];
        let mut count = 0;
        for token in number_tokens(rest) {
            let slot = values.get_mut(count).ok_or_else(|| invalid(at, "too many texture coordinate values"))?;
            *slot = parse_f32(token, at)?;
            count += 1;
        }
        if count == 0 {
            return Err(invalid(at, "texture coordinate without values"));
        }
        self.grow(self.obj.tex_coords.len(), 1, 8, "texture coordinate count")?;
        self.obj.tex_coords.push([values[0], flip_v(values[1])]);
        Ok(())
    }

    /// `vn x y z`.
    fn normal(&mut self, rest: &[u8], at: usize) -> Result<()> {
        let mut values = [0.0_f32; 3];
        let mut count = 0;
        for token in number_tokens(rest) {
            let slot = values.get_mut(count).ok_or_else(|| invalid(at, "a normal has 3 values"))?;
            *slot = parse_f32(token, at)?;
            count += 1;
        }
        if count != 3 {
            return Err(invalid(at, "a normal has 3 values"));
        }
        self.grow(self.obj.normals.len(), 1, 12, "normal count")?;
        self.obj.normals.push(values);
        Ok(())
    }

    /// One `v`, `v/vt`, `v//vn` or `v/vt/vn` corner.
    fn corner(&self, token: &[u8], at: usize) -> Result<Corner> {
        let mut parts = token.split(|&b| b == b'/');
        let position = parts.next().ok_or_else(|| invalid(at, "empty corner"))?;
        let position = resolve(parse_int(position, at)?, self.obj.positions.len(), at)?;
        let tex_coord = match parts.next() {
            None | Some(b"") => None,
            Some(t) => Some(resolve(parse_int(t, at)?, self.obj.tex_coords.len(), at)?),
        };
        let normal = match parts.next() {
            None => None,
            Some(n) => Some(resolve(parse_int(n, at)?, self.obj.normals.len(), at)?),
        };
        if parts.next().is_some() {
            return Err(invalid(at, "a corner has at most 3 indices"));
        }
        Ok(Corner::new(position, tex_coord, normal))
    }

    /// `f` with 3 or more corners.
    fn face(&mut self, rest: &[u8], at: usize) -> Result<()> {
        let start = self.obj.corners.len();
        for token in number_tokens(rest) {
            if self.obj.corners.len() - start >= MAX_FACE_CORNERS {
                self.obj.corners.truncate(start);
                return Err(Error::Limit {
                    format: FORMAT,
                    what: "corners of one face",
                    limit: MAX_FACE_CORNERS as u64,
                    got: MAX_FACE_CORNERS as u64 + 1,
                });
            }
            let corner = self.corner(token, at)?;
            self.grow(self.obj.corners.len(), 1, 12, "corner count")?;
            self.obj.corners.push(corner);
        }
        let end = self.obj.corners.len();
        if end - start < 3 {
            self.obj.corners.truncate(start);
            return Err(invalid(at, "a face has at least 3 corners"));
        }
        let face = self.obj.face_ends.len() as u32;
        self.grow(self.obj.face_ends.len(), 1, 4, "face count")?;
        self.obj.face_ends.push(end as u32);
        match self.obj.groups.last_mut() {
            Some(group) if group.same_state(&self.state) => group.faces.end = face + 1,
            _ => {
                let mut group = self.state.clone();
                group.faces = face..face + 1;
                self.grow(self.obj.groups.len(), 1, 64, "face group count")?;
                self.obj.groups.push(group);
            }
        }
        Ok(())
    }

    /// `l` with 2 or more vertices.
    fn polyline(&mut self, rest: &[u8], at: usize) -> Result<()> {
        let start = self.obj.line_vertices.len();
        for token in number_tokens(rest) {
            if self.obj.line_vertices.len() - start >= MAX_FACE_CORNERS {
                return Err(Error::Limit {
                    format: FORMAT,
                    what: "vertices of one polyline",
                    limit: MAX_FACE_CORNERS as u64,
                    got: MAX_FACE_CORNERS as u64 + 1,
                });
            }
            let corner = self.corner(token, at)?;
            if corner.normal().is_some() {
                return Err(invalid(at, "a polyline vertex has no normal"));
            }
            self.grow(self.obj.line_vertices.len(), 1, 4, "polyline vertex count")?;
            self.obj.line_vertices.push(corner.position);
        }
        let end = self.obj.line_vertices.len();
        if end - start < 2 {
            return Err(invalid(at, "a polyline has at least 2 vertices"));
        }
        self.grow(self.obj.line_ends.len(), 1, 4, "polyline count")?;
        self.obj.line_ends.push(end as u32);
        Ok(())
    }
}

/// OBJ's V (up from the bottom) as the project's V (down from the top), and back: `1 − v`, computed
/// in double precision so that the only rounding is the final one.
fn flip_v(v: f32) -> f32 {
    (1.0 - f64::from(v)) as f32
}

#[cfg(test)]
mod tests;
