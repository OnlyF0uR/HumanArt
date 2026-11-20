//! The face shape model `[WP-25]` `[WP-27]` (todo 1.2): the mean of several template-to-scan fits
//! as an offset field on the template's head, plus its principal shape components.
//!
//! **What it is.** Every subject's fit ([`super::fitted::FittedSurface`]) moved the same template
//! nodes; their fitted positions, brought into the template's frame and scaled to the template's
//! head size, are averaged. The model stores, per node of the template's head and neck, the rest
//! position (the generated template the fits started from), the mean offset `fitted − rest`, and
//! per shape component the offset of `+1` standard deviation. Offsets are already multiplied by
//! the blend mask the builder chose (1 in the face, fading over the lower neck, low under the
//! scalp), so a consumer adds them as they are: `p + mean(p) + Σₖ cₖ componentₖ(p)` with `cₖ` in
//! standard deviations. Like [`super::fitted`] the model is keyed by position (closest point on the
//! rest surface, barycentric interpolation), not by vertex index.
//!
//! **File format** `humanart-face-model` version 1 (`.hafm`, little-endian, written field by field):
//!
//! | bytes | content |
//! |---|---|
//! | 8 | magic `HAFACE\0\0` |
//! | 4 | version (1) |
//! | 2 × (4 + n) | description (sources, attribution), template name: `u32` byte length + UTF-8 (each ≤ [`MAX_TEXT`]) |
//! | 4 | template version (`u32`) |
//! | 16 | node count `n` (≤ [`MAX_NODES`]), triangle count `m` (≤ [`MAX_TRIANGLES`]), component count `k` (≤ [`MAX_COMPONENTS`]), subject count (≤ [`MAX_SUBJECTS`]) (`u32` each) |
//! | 4 | offset unit `q`, metres (`f32`, in `(0, 1e-3]`) |
//! | 16 n | per node: template vertex `u32`, rest position `3 f32` (metres) |
//! | 6 n | mean offset per node: `3 i16`, times `q` |
//! | k (4 + 6 n) | per component: its share of the subjects' shape variance (`f32`, `0..=1`), then the `+1 SD` offset per node (`3 i16`, times `q`) |
//! | 12 m | per triangle: three node indices `u32` |
//!
//! Nothing may follow. Every count is checked against its limit and the remaining input before
//! anything is allocated.

use humanart_io::bytes::{Reader, Writer};
use humanart_io::error::Format;
use humanart_math::Vec3;

use super::nearest::TriangleTree;
use super::{Result, ScanFitError};

/// The magic bytes.
pub const MAGIC: [u8; 8] = *b"HAFACE\0\0";
/// The format version written.
pub const VERSION: u32 = 1;
/// Longest text field, bytes.
pub const MAX_TEXT: usize = 4096;
/// Most nodes.
pub const MAX_NODES: usize = 1 << 20;
/// Most triangles.
pub const MAX_TRIANGLES: usize = 4 * MAX_NODES;
/// Most shape components.
pub const MAX_COMPONENTS: usize = 64;
/// Most subjects a model may claim.
pub const MAX_SUBJECTS: usize = 1 << 16;
/// The offset unit written by [`FaceModel::to_bytes`]: 0.01 mm (offsets up to ±0.327 m).
pub const OFFSET_UNIT: f32 = 1.0e-5;

/// One shape component.
#[derive(Clone, Debug, PartialEq)]
pub struct Component {
    /// Its share of the subjects' total shape variance, `0..=1`.
    pub variance: f32,
    /// Per node: the offset of `+1` standard deviation, metres.
    pub offsets: Vec<Vec3>,
}

/// The face shape model (module documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct FaceModel {
    /// Where the data comes from: sources and the attribution their licences ask for.
    pub description: String,
    /// The template's name.
    pub template_name: String,
    /// The template's version.
    pub template_version: u32,
    /// How many subjects were averaged.
    pub subjects: u32,
    /// Per node: a template vertex at it (valid for the template name and version).
    pub template_vertices: Vec<u32>,
    /// Per node: the rest position, template frame, metres.
    pub rest: Vec<Vec3>,
    /// Per node: the mean offset (blend mask applied), metres.
    pub mean: Vec<Vec3>,
    /// The shape components, largest variance first.
    pub components: Vec<Component>,
    /// Three node indices per triangle.
    pub triangles: Vec<u32>,
}

fn bad(what: &'static str) -> ScanFitError {
    ScanFitError::invalid(what)
}

fn read_text(r: &mut Reader<'_>) -> Result<String> {
    let n = r.len_u32_le()?;
    if n > MAX_TEXT {
        return Err(ScanFitError::limit("face model text length", MAX_TEXT, n));
    }
    let bytes = r.bytes(n)?;
    String::from_utf8(bytes.to_vec()).map_err(|_| bad("a face model text is not UTF-8"))
}

fn write_text(w: &mut Writer, text: &str) {
    let n = text.len().min(MAX_TEXT);
    // Truncate on a character boundary.
    let n = (0..=n).rev().find(|&k| text.is_char_boundary(k)).unwrap_or(0);
    w.u32_le(n as u32);
    w.bytes(text.as_bytes().get(..n).unwrap_or(&[]));
}

fn read_offsets(r: &mut Reader<'_>, n: usize, unit: f32) -> Result<Vec<Vec3>> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let (x, y, z) = (r.i16_le()?, r.i16_le()?, r.i16_le()?);
        out.push(Vec3::new(f32::from(x), f32::from(y), f32::from(z)) * unit);
    }
    Ok(out)
}

/// The offset in units of [`OFFSET_UNIT`], `None` beyond the `i16` range.
fn quantise(v: Vec3) -> Option<[i16; 3]> {
    let q = |c: f32| {
        let u = (c / OFFSET_UNIT).round();
        (u.abs() <= f32::from(i16::MAX)).then_some(u as i16)
    };
    Some([q(v.x)?, q(v.y)?, q(v.z)?])
}

fn write_offsets(w: &mut Writer, offsets: &[Vec3]) -> Result<()> {
    for &v in offsets {
        let q = quantise(v).ok_or(bad("a face model offset is beyond the file's range (±0.327 m)"))?;
        for c in q {
            w.i16_le(c);
        }
    }
    Ok(())
}

impl FaceModel {
    /// Checks lengths, limits, index ranges and finite values.
    ///
    /// # Errors
    /// [`ScanFitError::Invalid`] or [`ScanFitError::Limit`] naming the problem.
    pub fn validate(&self) -> Result<()> {
        let n = self.rest.len();
        if n > MAX_NODES {
            return Err(ScanFitError::limit("face model node count", MAX_NODES, n));
        }
        if self.triangles.len() / 3 > MAX_TRIANGLES {
            return Err(ScanFitError::limit("face model triangle count", MAX_TRIANGLES, self.triangles.len() / 3));
        }
        if self.components.len() > MAX_COMPONENTS {
            return Err(ScanFitError::limit("face model component count", MAX_COMPONENTS, self.components.len()));
        }
        if self.subjects as usize > MAX_SUBJECTS {
            return Err(ScanFitError::limit("face model subject count", MAX_SUBJECTS, self.subjects as usize));
        }
        if self.template_vertices.len() != n
            || self.mean.len() != n
            || self.components.iter().any(|c| c.offsets.len() != n)
        {
            return Err(bad("the face model's per-node tables differ in length"));
        }
        if !self.triangles.len().is_multiple_of(3) || self.triangles.iter().any(|&i| i as usize >= n) {
            return Err(bad("face model triangles name missing nodes"));
        }
        let finite = |v: &[Vec3]| v.iter().all(|p| p.is_finite());
        if !finite(&self.rest) || !finite(&self.mean) || !self.components.iter().all(|c| finite(&c.offsets)) {
            return Err(bad("a face model position or offset is not finite"));
        }
        if !self.components.iter().all(|c| (0.0..=1.0).contains(&c.variance)) {
            return Err(bad("a face model component's variance share is outside 0..=1"));
        }
        Ok(())
    }

    /// The file bytes (offsets quantised to [`OFFSET_UNIT`]).
    ///
    /// # Errors
    /// As [`FaceModel::validate`]; [`ScanFitError::Invalid`] for an offset beyond ±0.327 m.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let n = self.rest.len();
        let mut w = Writer::with_capacity(64 + n * (22 + 6 * self.components.len()) + self.triangles.len() * 4);
        w.bytes(&MAGIC);
        w.u32_le(VERSION);
        write_text(&mut w, &self.description);
        write_text(&mut w, &self.template_name);
        w.u32_le(self.template_version);
        w.u32_le(n as u32);
        w.u32_le((self.triangles.len() / 3) as u32);
        w.u32_le(self.components.len() as u32);
        w.u32_le(self.subjects);
        w.f32_le(OFFSET_UNIT);
        for (&v, p) in self.template_vertices.iter().zip(&self.rest) {
            w.u32_le(v);
            w.f32_le(p.x);
            w.f32_le(p.y);
            w.f32_le(p.z);
        }
        write_offsets(&mut w, &self.mean)?;
        for c in &self.components {
            w.f32_le(c.variance);
            write_offsets(&mut w, &c.offsets)?;
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
            return Err(bad("not a face model file (magic)"));
        }
        if r.u32_le()? != VERSION {
            return Err(bad("unsupported face model version"));
        }
        let description = read_text(&mut r)?;
        let template_name = read_text(&mut r)?;
        let template_version = r.u32_le()?;
        let n = r.len_u32_le()?;
        let m = r.len_u32_le()?;
        let k = r.len_u32_le()?;
        let subjects = r.u32_le()?;
        if n > MAX_NODES {
            return Err(ScanFitError::limit("face model node count", MAX_NODES, n));
        }
        if m > MAX_TRIANGLES {
            return Err(ScanFitError::limit("face model triangle count", MAX_TRIANGLES, m));
        }
        if k > MAX_COMPONENTS {
            return Err(ScanFitError::limit("face model component count", MAX_COMPONENTS, k));
        }
        if subjects as usize > MAX_SUBJECTS {
            return Err(ScanFitError::limit("face model subject count", MAX_SUBJECTS, subjects as usize));
        }
        let unit = r.f32_le()?;
        if !(unit > 0.0 && unit <= 1.0e-3) {
            return Err(bad("the face model offset unit is outside (0, 1 mm]"));
        }
        // Every table's size, checked against the input before anything is allocated.
        let node_bytes = r.check_count(n, 22)?;
        let component_bytes = n.checked_mul(6).and_then(|b| b.checked_add(4)).and_then(|b| b.checked_mul(k));
        let tri_bytes = m.checked_mul(12);
        let (Some(component_bytes), Some(tri_bytes)) = (component_bytes, tri_bytes) else {
            return Err(bad("face model counts overflow"));
        };
        if node_bytes.checked_add(component_bytes).and_then(|b| b.checked_add(tri_bytes)) != Some(r.remaining()) {
            return Err(bad("the face model length does not match its counts"));
        }
        let mut template_vertices = Vec::with_capacity(n);
        let mut rest = Vec::with_capacity(n);
        for _ in 0..n {
            template_vertices.push(r.u32_le()?);
            rest.push(Vec3::new(r.f32_le()?, r.f32_le()?, r.f32_le()?));
        }
        let mean = read_offsets(&mut r, n, unit)?;
        let mut components = Vec::with_capacity(k);
        for _ in 0..k {
            let variance = r.f32_le()?;
            components.push(Component { variance, offsets: read_offsets(&mut r, n, unit)? });
        }
        let mut triangles = Vec::with_capacity(3 * m);
        for _ in 0..3 * m {
            triangles.push(r.u32_le()?);
        }
        let out = Self {
            description,
            template_name,
            template_version,
            subjects,
            template_vertices,
            rest,
            mean,
            components,
            triangles,
        };
        out.validate()?;
        Ok(out)
    }

    /// The closest-point field over the rest surface.
    ///
    /// # Errors
    /// As [`FaceModel::validate`] and [`TriangleTree::build`].
    pub fn field(&self) -> Result<FaceField<'_>> {
        self.validate()?;
        let tree = TriangleTree::build(&self.rest, &self.triangles)?;
        Ok(FaceField { model: self, tree })
    }
}

/// Where a point lies on the model's rest surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceSample {
    /// The three nodes of the closest triangle.
    pub nodes: [u32; 3],
    /// Their barycentric weights at the closest point.
    pub bary: [f32; 3],
    /// Distance of the query point from the rest surface, metres.
    pub distance: f32,
}

/// [`FaceModel`] with a closest-point tree over its rest surface.
#[derive(Debug)]
pub struct FaceField<'a> {
    model: &'a FaceModel,
    tree: TriangleTree,
}

impl FaceField<'_> {
    /// The closest point of the rest surface to `p` within `max_distance` (`None`: outside the
    /// model's area).
    #[must_use]
    pub fn sample(&self, p: Vec3, max_distance: f32) -> Option<FaceSample> {
        let hit = self.tree.nearest(p, max_distance, |_| true)?;
        let t = hit.triangle as usize;
        let tri = self.model.triangles.get(3 * t..3 * t + 3)?;
        Some(FaceSample { nodes: [tri[0], tri[1], tri[2]], bary: hit.bary, distance: hit.distance })
    }

    fn interpolate(table: &[Vec3], s: &FaceSample) -> Vec3 {
        s.nodes
            .iter()
            .zip(s.bary)
            .map(|(&i, b)| table.get(i as usize).copied().unwrap_or(Vec3::ZERO) * b)
            .fold(Vec3::ZERO, |a, b| a + b)
    }

    /// The mean offset at a sample.
    #[must_use]
    pub fn mean(&self, s: &FaceSample) -> Vec3 {
        Self::interpolate(&self.model.mean, s)
    }

    /// The `+1 SD` offset of component `k` at a sample (zero for a missing component).
    #[must_use]
    pub fn component(&self, s: &FaceSample, k: usize) -> Vec3 {
        self.model.components.get(k).map_or(Vec3::ZERO, |c| Self::interpolate(&c.offsets, s))
    }

    /// The model.
    #[must_use]
    pub const fn model(&self) -> &FaceModel {
        self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> FaceModel {
        let rest = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), Vec3::ONE];
        FaceModel {
            description: "HSRD-100 (CC BY 4.0), test".into(),
            template_name: "humanart-neutral".into(),
            template_version: 1,
            subjects: 8,
            template_vertices: vec![3, 4, 5, 6],
            rest,
            mean: vec![Vec3::ZERO, Vec3::new(0.0, 0.0, 0.001), Vec3::new(0.0, 0.0, 0.002), Vec3::new(0.0, 0.0, 0.003)],
            components: vec![Component {
                variance: 0.4,
                offsets: vec![Vec3::new(0.0005, 0.0, 0.0), Vec3::ZERO, Vec3::new(0.0, -0.0012, 0.0), Vec3::ZERO],
            }],
            triangles: vec![0, 1, 2, 1, 3, 2],
        }
    }

    fn close(a: &FaceModel, b: &FaceModel) -> bool {
        let near = |x: &[Vec3], y: &[Vec3]| x.iter().zip(y).all(|(p, q)| p.abs_diff_eq(*q, OFFSET_UNIT * 0.51));
        a.rest == b.rest
            && a.triangles == b.triangles
            && a.template_vertices == b.template_vertices
            && near(&a.mean, &b.mean)
            && a.components.len() == b.components.len()
            && a.components
                .iter()
                .zip(&b.components)
                .all(|(c, d)| c.variance == d.variance && near(&c.offsets, &d.offsets))
    }

    #[test]
    fn wp25_face_model_round_trips_and_interpolates() {
        let m = sample();
        let bytes = m.to_bytes().unwrap();
        assert_eq!(&bytes[..8], b"HAFACE\0\0");
        let back = FaceModel::parse(&bytes).unwrap();
        assert!(close(&m, &back), "{back:?}");
        assert_eq!((back.subjects, back.description.as_str()), (8, "HSRD-100 (CC BY 4.0), test"));
        let field = back.field().unwrap();
        let s = field.sample(Vec3::new(0.25, 0.25, 0.004), 0.1).unwrap();
        assert!((s.distance - 0.004).abs() < 1e-6);
        // Mean offsets 0, 1, 2 mm at the first triangle's corners: 0.25 + 0.5 mm.
        assert!((field.mean(&s).z - 0.00075).abs() < 2e-6, "{:?}", field.mean(&s));
        assert!((field.component(&s, 0).x - 0.5 * 0.0005).abs() < 2e-6);
        assert_eq!(field.component(&s, 3), Vec3::ZERO);
        assert!(field.sample(Vec3::new(5.0, 5.0, 0.0), 0.1).is_none());
    }

    #[test]
    fn wp25_bad_face_models_fail_cleanly() {
        let bytes = sample().to_bytes().unwrap();
        for cut in [0, 7, 12, 30, bytes.len() / 2, bytes.len() - 1] {
            assert!(FaceModel::parse(&bytes[..cut]).is_err(), "cut at {cut}");
        }
        let mut longer = bytes.clone();
        longer.push(0);
        assert!(FaceModel::parse(&longer).is_err());
        let mut far = sample();
        far.mean[1].x = 0.5;
        assert!(far.to_bytes().is_err(), "beyond the i16 range");
        let mut tri = sample();
        tri.triangles[0] = 9;
        assert!(tri.to_bytes().is_err());
        let mut nan = sample();
        nan.rest[2].y = f32::NAN;
        assert!(nan.to_bytes().is_err());
        let mut short = sample();
        short.components[0].offsets.pop();
        assert!(short.to_bytes().is_err());
        humanart_core::fuzz::run("face-model", &[bytes], 4000, 0xface, std::time::Duration::from_secs(60), |b| {
            if let Ok(m) = FaceModel::parse(b) {
                assert!(m.validate().is_ok());
            }
        });
    }
}
