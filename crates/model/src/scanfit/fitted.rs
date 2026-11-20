//! The result of a template-to-scan fit `[WP-25]` as a file: the fitted surface in the template's
//! frame, which the template generator consumes as a displacement field.
//!
//! **Why a field, not vertex positions.** The generated template's topology changes whenever the
//! generator does, so vertex indices are not a stable key. The file holds the fitted nodes with
//! their rest positions (the template the fit started from) and their fitted positions, both in the
//! template's frame (the landmark similarity is undone), plus the triangles between them.
//! [`DisplacementField::at`] gives the displacement of any point near the rest surface (closest
//! point, barycentric interpolation), so a later template can take the scan's shape on whatever
//! vertices it has. The template vertex of each node is kept for the same template version.
//!
//! **File format** `humanart-scan-fit` version 1 (`.hafit`, little-endian, written field by field):
//!
//! | bytes | content |
//! |---|---|
//! | 8 | magic `HAFIT\0\0\0` |
//! | 4 | version (1) |
//! | 3 × (4 + n) | source scan id, attribution, template name: `u32` byte length + UTF-8 (each ≤ [`MAX_TEXT`]) |
//! | 8 | template version, template vertex count (`u32` each) |
//! | 52 | the template-to-scan similarity: rotation rows (9 `f32`), scale, translation (3 `f32`) |
//! | 24 | report: RMS error, max error, matched fraction, landmark RMS, folded fraction (`f32`, metres or fractions), flags (`u32`) |
//! | 8 | node count `n` (≤ [`MAX_NODES`]), triangle count `m` (≤ [`MAX_TRIANGLES`]) |
//! | 37 n | per node: template vertex `u32`, rest `3 f32`, fitted `3 f32`, fit error `f32` (metres, negative: no scan point found), data weight `f32`, matched `u8` |
//! | 12 m | per triangle: three node indices `u32` |
//!
//! Nothing may follow. Every count is checked against the remaining input before allocating.

use humanart_io::bytes::{Reader, Writer};
use humanart_io::error::Format;
use humanart_math::Vec3;

use super::nearest::TriangleTree;
use super::{Result, ScanFitError, Similarity};

/// The magic bytes.
pub const MAGIC: [u8; 8] = *b"HAFIT\0\0\0";
/// The format version written.
pub const VERSION: u32 = 1;
/// Longest text field, bytes.
pub const MAX_TEXT: usize = 4096;
/// Most nodes.
pub const MAX_NODES: usize = 1 << 21;
/// Most triangles.
pub const MAX_TRIANGLES: usize = 4 * MAX_NODES;
/// Bytes per node record.
const NODE_BYTES: usize = 37;

/// A fitted node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FittedNode {
    /// A template vertex at the node (valid for the template name and version of the file).
    pub template_vertex: u32,
    /// The rest position, template frame, metres.
    pub rest: Vec3,
    /// The fitted position, template frame, metres.
    pub fitted: Vec3,
    /// Distance to the scan after the fit, metres (negative: no scan point within reach).
    pub error: f32,
    /// The data weight the fit used (0: the node only followed its neighbours).
    pub weight: f32,
    /// Whether the node had a scan correspondence in the last iteration.
    pub matched: bool,
}

/// The summary numbers of the fit.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FittedSummary {
    /// RMS fit error of the matched weighted nodes, metres (scan frame).
    pub rms_error: f32,
    /// Largest fit error of a matched weighted node, metres.
    pub max_error: f32,
    /// Fraction of weighted nodes with a correspondence.
    pub matched_fraction: f32,
    /// RMS landmark residual, metres.
    pub landmark_rms: f32,
    /// Fraction of folded triangles.
    pub folded_fraction: f32,
    /// The fit's failure flags (`humanart_tools::capture::register::FitFlags` bits).
    pub flags: u32,
}

/// A fitted surface in the template's frame (module documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct FittedSurface {
    /// The scan it was fitted to (an id such as `HSRD-100/HSR0042-Body-045`).
    pub source: String,
    /// The attribution the source's licence asks for.
    pub attribution: String,
    /// The template's name.
    pub template_name: String,
    /// The template's version.
    pub template_version: u32,
    /// The template's LOD0 vertex count.
    pub template_vertices: u32,
    /// The similarity that took the template into the scan's frame.
    pub alignment: Similarity,
    /// The fit's numbers.
    pub summary: FittedSummary,
    /// The nodes.
    pub nodes: Vec<FittedNode>,
    /// Three node indices per triangle.
    pub triangles: Vec<u32>,
}

fn bad(what: &'static str) -> ScanFitError {
    ScanFitError::invalid(what)
}

fn read_vec3(r: &mut Reader<'_>) -> Result<Vec3> {
    Ok(Vec3::new(r.f32_le()?, r.f32_le()?, r.f32_le()?))
}

fn write_vec3(w: &mut Writer, v: Vec3) {
    w.f32_le(v.x);
    w.f32_le(v.y);
    w.f32_le(v.z);
}

fn read_text(r: &mut Reader<'_>) -> Result<String> {
    let n = r.len_u32_le()?;
    if n > MAX_TEXT {
        return Err(ScanFitError::limit("fit file text length", MAX_TEXT, n));
    }
    let bytes = r.bytes(n)?;
    String::from_utf8(bytes.to_vec()).map_err(|_| bad("a fit file text is not UTF-8"))
}

fn write_text(w: &mut Writer, text: &str) {
    let bytes = text.as_bytes();
    let n = bytes.len().min(MAX_TEXT);
    // Truncate on a character boundary.
    let n = (0..=n).rev().find(|&k| text.is_char_boundary(k)).unwrap_or(0);
    w.u32_le(n as u32);
    w.bytes(&bytes[..n]);
}

impl FittedSurface {
    /// Checks lengths, index ranges and finite values.
    ///
    /// # Errors
    /// [`ScanFitError::Invalid`] or [`ScanFitError::Limit`] naming the problem.
    pub fn validate(&self) -> Result<()> {
        if self.nodes.len() > MAX_NODES {
            return Err(ScanFitError::limit("fit file node count", MAX_NODES, self.nodes.len()));
        }
        if self.triangles.len() / 3 > MAX_TRIANGLES {
            return Err(ScanFitError::limit("fit file triangle count", MAX_TRIANGLES, self.triangles.len() / 3));
        }
        let n = self.nodes.len();
        if !self.triangles.len().is_multiple_of(3) || self.triangles.iter().any(|&i| i as usize >= n) {
            return Err(bad("fit file triangles name missing nodes"));
        }
        if !self.nodes.iter().all(|v| {
            v.rest.is_finite() && v.fitted.is_finite() && v.error.is_finite() && (0.0..=1.0).contains(&v.weight)
        }) {
            return Err(bad("a fit file node is not finite or its weight is outside 0..=1"));
        }
        let a = &self.alignment;
        if !(a.rotation.iter().all(|r| r.is_finite())
            && a.translation.is_finite()
            && a.scale.is_finite()
            && a.scale > 0.0)
        {
            return Err(bad("the fit file alignment is not a finite similarity"));
        }
        Ok(())
    }

    /// The file bytes.
    ///
    /// # Errors
    /// As [`FittedSurface::validate`].
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut w = Writer::with_capacity(128 + self.nodes.len() * NODE_BYTES + self.triangles.len() * 4);
        w.bytes(&MAGIC);
        w.u32_le(VERSION);
        write_text(&mut w, &self.source);
        write_text(&mut w, &self.attribution);
        write_text(&mut w, &self.template_name);
        w.u32_le(self.template_version);
        w.u32_le(self.template_vertices);
        for r in self.alignment.rotation {
            write_vec3(&mut w, r);
        }
        w.f32_le(self.alignment.scale);
        write_vec3(&mut w, self.alignment.translation);
        let s = &self.summary;
        for v in [s.rms_error, s.max_error, s.matched_fraction, s.landmark_rms, s.folded_fraction] {
            w.f32_le(v);
        }
        w.u32_le(s.flags);
        w.u32_le(self.nodes.len() as u32);
        w.u32_le((self.triangles.len() / 3) as u32);
        for v in &self.nodes {
            w.u32_le(v.template_vertex);
            write_vec3(&mut w, v.rest);
            write_vec3(&mut w, v.fitted);
            w.f32_le(v.error);
            w.f32_le(v.weight);
            w.u8(u8::from(v.matched));
        }
        for &i in &self.triangles {
            w.u32_le(i);
        }
        Ok(w.into_vec())
    }

    /// Reads a file.
    ///
    /// # Errors
    /// [`ScanFitError::Io`] for truncated input, [`ScanFitError::Invalid`] /
    /// [`ScanFitError::Limit`] for anything malformed.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes, Format::Chunk);
        if r.array::<8>()? != MAGIC {
            return Err(bad("not a fit file (magic)"));
        }
        if r.u32_le()? != VERSION {
            return Err(bad("unsupported fit file version"));
        }
        let source = read_text(&mut r)?;
        let attribution = read_text(&mut r)?;
        let template_name = read_text(&mut r)?;
        let template_version = r.u32_le()?;
        let template_vertices = r.u32_le()?;
        let rotation = [read_vec3(&mut r)?, read_vec3(&mut r)?, read_vec3(&mut r)?];
        let scale = r.f32_le()?;
        let translation = read_vec3(&mut r)?;
        let summary = FittedSummary {
            rms_error: r.f32_le()?,
            max_error: r.f32_le()?,
            matched_fraction: r.f32_le()?,
            landmark_rms: r.f32_le()?,
            folded_fraction: r.f32_le()?,
            flags: r.u32_le()?,
        };
        let n = r.len_u32_le()?;
        let m = r.len_u32_le()?;
        if n > MAX_NODES {
            return Err(ScanFitError::limit("fit file node count", MAX_NODES, n));
        }
        if m > MAX_TRIANGLES {
            return Err(ScanFitError::limit("fit file triangle count", MAX_TRIANGLES, m));
        }
        let node_bytes = r.check_count(n, NODE_BYTES)?;
        let tri_bytes = m.checked_mul(12).ok_or_else(|| bad("fit file triangle count overflows"))?;
        if node_bytes + tri_bytes != r.remaining() {
            return Err(bad("the fit file length does not match its counts"));
        }
        let mut nodes = Vec::with_capacity(n);
        for _ in 0..n {
            nodes.push(FittedNode {
                template_vertex: r.u32_le()?,
                rest: read_vec3(&mut r)?,
                fitted: read_vec3(&mut r)?,
                error: r.f32_le()?,
                weight: r.f32_le()?,
                matched: match r.u8()? {
                    0 => false,
                    1 => true,
                    _ => return Err(bad("a fit file matched flag is not 0 or 1")),
                },
            });
        }
        let mut triangles = Vec::with_capacity(3 * m);
        for _ in 0..3 * m {
            triangles.push(r.u32_le()?);
        }
        let out = Self {
            source,
            attribution,
            template_name,
            template_version,
            template_vertices,
            alignment: Similarity { rotation, scale, translation },
            summary,
            nodes,
            triangles,
        };
        out.validate()?;
        Ok(out)
    }

    /// The displacement field of this fit (builds a closest-point tree over the rest surface).
    ///
    /// # Errors
    /// [`ScanFitError::Invalid`] for an empty or malformed surface.
    pub fn field(&self) -> Result<DisplacementField<'_>> {
        self.validate()?;
        let rest: Vec<Vec3> = self.nodes.iter().map(|v| v.rest).collect();
        let tree = TriangleTree::build(&rest, &self.triangles)?;
        Ok(DisplacementField { surface: self, tree })
    }
}

/// The fitted displacement as a function of position near the rest surface.
#[derive(Debug)]
pub struct DisplacementField<'a> {
    surface: &'a FittedSurface,
    tree: TriangleTree,
}

/// One displacement sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Displacement {
    /// Fitted minus rest position, template frame, metres.
    pub offset: Vec3,
    /// The interpolated data weight of the fit there (how much the scan, not the smoothness,
    /// decided the shape).
    pub weight: f32,
    /// Distance of the query point from the rest surface, metres.
    pub distance: f32,
}

impl DisplacementField<'_> {
    /// The displacement at `p` (template frame): the closest point of the rest surface within
    /// `max_distance`, its triangle's node offsets interpolated barycentrically; `None` when no
    /// fitted triangle is that close (the point is outside the fitted area).
    #[must_use]
    pub fn at(&self, p: Vec3, max_distance: f32) -> Option<Displacement> {
        let hit = self.tree.nearest(p, max_distance, |_| true)?;
        let t = hit.triangle as usize;
        let tri = self.surface.triangles.get(3 * t..3 * t + 3)?;
        let (mut offset, mut weight) = (Vec3::ZERO, 0.0);
        for (&i, &b) in tri.iter().zip(&hit.bary) {
            let v = self.surface.nodes.get(i as usize)?;
            offset += (v.fitted - v.rest) * b;
            weight += v.weight * b;
        }
        Some(Displacement { offset, weight, distance: hit.distance })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> FittedSurface {
        let node = |x: f32, y: f32, dz: f32| FittedNode {
            template_vertex: (x * 10.0 + y) as u32,
            rest: Vec3::new(x, y, 0.0),
            fitted: Vec3::new(x, y, dz),
            error: 0.001,
            weight: 1.0,
            matched: true,
        };
        FittedSurface {
            source: "HSRD-100/test".into(),
            attribution: "test (CC BY 4.0)".into(),
            template_name: "generated".into(),
            template_version: 3,
            template_vertices: 100,
            alignment: Similarity { scale: 1.1, ..Similarity::IDENTITY },
            summary: FittedSummary { rms_error: 0.001, flags: 2, ..FittedSummary::default() },
            nodes: vec![node(0.0, 0.0, 0.0), node(1.0, 0.0, 0.01), node(0.0, 1.0, 0.02), node(1.0, 1.0, 0.03)],
            triangles: vec![0, 1, 2, 1, 3, 2],
        }
    }

    #[test]
    fn wp25_fit_file_round_trips_and_interpolates() {
        let s = sample();
        let bytes = s.to_bytes().unwrap();
        assert_eq!(&bytes[..8], b"HAFIT\0\0\0");
        let back = FittedSurface::parse(&bytes).unwrap();
        assert_eq!(back, s);
        let field = back.field().unwrap();
        // Inside the first triangle: offsets 0, 0.01, 0.02 at the corners.
        let d = field.at(Vec3::new(0.25, 0.25, 0.005), 0.1).unwrap();
        assert!((d.offset.z - (0.25 * 0.01 + 0.25 * 0.02)).abs() < 1e-6, "{d:?}");
        assert!((d.distance - 0.005).abs() < 1e-6 && d.weight == 1.0);
        assert!(field.at(Vec3::new(5.0, 5.0, 0.0), 0.1).is_none());
    }

    #[test]
    fn wp25_bad_fit_files_fail_cleanly() {
        let bytes = sample().to_bytes().unwrap();
        for cut in [0, 7, 12, 40, bytes.len() - 1] {
            assert!(FittedSurface::parse(&bytes[..cut]).is_err(), "cut at {cut}");
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(FittedSurface::parse(&longer).is_err());
        let mut bad_tri = sample();
        bad_tri.triangles[0] = 9;
        assert!(bad_tri.to_bytes().is_err());
        let mut nan = sample();
        nan.nodes[1].fitted.x = f32::NAN;
        assert!(nan.to_bytes().is_err());
        humanart_core::fuzz::run("scan-fit", &[bytes], 4000, 0x5eed, std::time::Duration::from_secs(60), |b| {
            if let Ok(s) = FittedSurface::parse(b) {
                assert!(s.validate().is_ok());
            }
        });
    }
}
