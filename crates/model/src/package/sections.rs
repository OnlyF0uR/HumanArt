//! Encoding and decoding of the section payloads of a `.antmpl` package (see
//! `docs/format-antmpl.md` for the byte layout of each).
//!
//! Every count read from a payload is checked against its documented limit, and every array
//! against the bytes left in the payload, before anything is allocated `[ROB-01]`. A payload must
//! be consumed exactly; trailing bytes are an error.

use humanart_io::bytes::{Reader, Writer};
use humanart_io::json::{self, JsonLimits, Object, Value};
use humanart_io::{Error as IoError, Format};
use humanart_math::{F16, Quat, Transform, Vec2, Vec3};

use super::PackageLimits;
use crate::delta::{DeltaBases, DeltaComponent, DeltaKind, MAX_COMPONENTS};
use crate::error::{ModelError, Result};
use crate::expression::{
    CombinationCorrective, CombineOp, Expression, ExpressionKind, ExpressionSet, MAX_COMBINATION_INPUTS,
    MAX_DEFINITIONS, PoseCorrective,
};
use crate::landmark::{Landmark, LandmarkGroup, MAX_LANDMARKS};
use crate::lod::{LodLevel, MAX_LOD_LEVELS};
use crate::mesh::{MAX_FACES, MAX_PARTS, MAX_VERTICES, Mesh, MeshData, MeshPart};
use crate::names::{MAX_NAME_LEN, Side};
use crate::region::{MAX_REGIONS, Region, RegionMap, SoftRegionWeights};
use crate::skeleton::{Joint, JointKind, JointLimits, JointRegressor, MAX_JOINTS, Skeleton};
use crate::skin::{MAX_INFLUENCES, SkinWeights};
use crate::template::{MAX_PROVENANCE, MAX_TEXT_LEN, MAX_TILES, ProvenanceEntry, TemplateInfo, UvTile};

/// The `u16` that stands for "no parent".
const NO_PARENT: u16 = u16::MAX;
/// The value of the `format` key of the metadata section.
pub(crate) const META_FORMAT: &str = "humanart-template";

// ---------------------------------------------------------------------------------------------
// Writing

fn put_str(w: &mut Writer, s: &str) {
    // Names are at most 64 bytes (checked by `Template::new`).
    w.u16_le(s.len() as u16);
    w.bytes(s.as_bytes());
}

fn put_vec3(w: &mut Writer, v: Vec3) {
    w.f32_le(v.x);
    w.f32_le(v.y);
    w.f32_le(v.z);
}

fn put_parent(w: &mut Writer, p: Option<u16>) {
    w.u16_le(p.unwrap_or(NO_PARENT));
}

pub(crate) fn write_meta(info: &TemplateInfo) -> Result<Vec<u8>> {
    let mut root = Object::default();
    root.insert("format", META_FORMAT);
    root.insert("name", info.name.as_str());
    root.insert("version", info.version);
    root.insert("generator", info.generator.as_str());
    root.insert("licence", info.licence.as_str());
    let provenance: Vec<Value> = info
        .provenance
        .iter()
        .map(|p| {
            let mut o = Object::default();
            o.insert("item", p.item.as_str());
            o.insert("source", p.source.as_str());
            o.insert("licence", p.licence.as_str());
            Value::Object(o)
        })
        .collect();
    root.insert("provenance", provenance);
    root.insert("extra", info.extra.clone());
    Ok(json::to_vec(&Value::Object(root))?)
}

pub(crate) fn write_mesh(w: &mut Writer, m: &Mesh) {
    let d = m.data();
    w.u32_le(d.positions.len() as u32);
    w.u32_le(m.face_count() as u32);
    w.u32_le(d.face_vertices.len() as u32);
    w.u32_le(d.parts.len() as u32);
    for &p in &d.positions {
        put_vec3(w, p);
    }
    for uv in &d.uvs {
        w.f32_le(uv.x);
        w.f32_le(uv.y);
    }
    for &k in d.position_keys.iter().chain(&d.normal_keys).chain(&d.face_offsets).chain(&d.face_vertices) {
        w.u32_le(k);
    }
    for &t in &d.face_tiles {
        w.u16_le(t);
    }
    for p in &d.parts {
        put_str(w, &p.name);
        w.u32_le(p.first_face);
        w.u32_le(p.face_count);
        w.u8(u8::from(p.closed));
    }
}

pub(crate) fn write_tiles(w: &mut Writer, tiles: &[UvTile]) {
    w.u32_le(tiles.len() as u32);
    for t in tiles {
        put_str(w, &t.name);
        w.u32_le(t.resolution);
        w.f32_le(t.target_texels_per_mm);
    }
}

pub(crate) fn write_regions(w: &mut Writer, r: &RegionMap) {
    w.u32_le(r.regions.len() as u32);
    for x in &r.regions {
        put_str(w, &x.name);
        put_parent(w, x.parent);
        w.u8(x.side.to_u8());
    }
    w.u32_le(r.vertex_regions.len() as u32);
    for &v in &r.vertex_regions {
        w.u16_le(v);
    }
    match &r.soft {
        None => w.u8(0),
        Some(s) => {
            w.u8(1);
            w.u32_le(s.regions.len() as u32);
            for &o in &s.offsets {
                w.u32_le(o);
            }
            for &x in &s.regions {
                w.u16_le(x);
            }
            for &x in &s.weights {
                w.f32_le(x);
            }
        }
    }
}

pub(crate) fn write_landmarks(w: &mut Writer, landmarks: &[Landmark]) {
    w.u32_le(landmarks.len() as u32);
    for l in landmarks {
        put_str(w, &l.name);
        w.u8(l.group.to_u8());
        for &v in &l.vertices {
            w.u32_le(v);
        }
        for &x in &l.weights {
            w.f32_le(x);
        }
    }
}

pub(crate) fn write_skeleton(w: &mut Writer, s: &Skeleton) {
    w.u32_le(s.joints.len() as u32);
    for j in &s.joints {
        put_str(w, &j.name);
        put_parent(w, j.parent);
        w.u8(j.kind.to_u8());
        put_vec3(w, j.rest.translation);
        let q = j.rest.rotation;
        for c in [q.x, q.y, q.z, q.w] {
            w.f32_le(c);
        }
        put_vec3(w, j.rest.scale);
        put_vec3(w, j.limits.min);
        put_vec3(w, j.limits.max);
    }
    let r = &s.regressor;
    w.u32_le(r.offsets.len().saturating_sub(1) as u32);
    w.u32_le(r.joints.len() as u32);
    for &o in &r.offsets {
        w.u32_le(o);
    }
    for &j in &r.joints {
        w.u16_le(j);
    }
    for &d in &r.deltas {
        put_vec3(w, d);
    }
}

pub(crate) fn write_skin(w: &mut Writer, s: &SkinWeights) {
    w.u8(s.influences);
    w.u32_le(s.vertex_count() as u32);
    for &j in &s.joints {
        w.u16_le(j);
    }
    for &x in &s.weights {
        w.f32_le(x);
    }
    w.u32_le(s.dual_quaternion_regions.len() as u32);
    for &r in &s.dual_quaternion_regions {
        w.u16_le(r);
    }
}

pub(crate) fn write_deltas(w: &mut Writer, d: &DeltaBases) {
    w.u32_le(d.len() as u32);
    w.u32_le(d.all_indices().len() as u32);
    for c in d.components() {
        put_str(w, &c.name);
        w.u8(c.kind.to_u8());
        w.u16_le(c.region);
    }
    for &o in d.offsets() {
        w.u32_le(o);
    }
    for &i in d.all_indices() {
        w.u32_le(i);
    }
    for xyz in d.all_deltas() {
        for h in xyz {
            w.u16_le(h.to_bits());
        }
    }
}

pub(crate) fn write_expressions(w: &mut Writer, e: &ExpressionSet) {
    w.u32_le(e.expressions.len() as u32);
    for x in &e.expressions {
        put_str(w, &x.name);
        w.u8(x.kind.to_u8());
        w.u8(x.side.to_u8());
        w.u16_le(x.facs);
        w.u32_le(x.component);
    }
    w.u32_le(e.combinations.len() as u32);
    for c in &e.combinations {
        put_str(w, &c.name);
        w.u8(c.op.to_u8());
        w.u8(c.inputs.len() as u8);
        for &i in &c.inputs {
            w.u32_le(i);
        }
        w.u32_le(c.component);
    }
    w.u32_le(e.pose_correctives.len() as u32);
    for p in &e.pose_correctives {
        put_str(w, &p.name);
        w.u16_le(p.joint);
        for c in [p.target.x, p.target.y, p.target.z, p.target.w] {
            w.f32_le(c);
        }
        w.f32_le(p.radius);
        w.u32_le(p.component);
    }
}

pub(crate) fn write_lods(w: &mut Writer, lods: &[LodLevel]) {
    w.u32_le(lods.len() as u32);
    for l in lods {
        w.f32_le(l.declared_error);
        w.f32_le(l.switch_size);
        w.f32_le(l.hysteresis);
        write_mesh(w, &l.mesh);
        for s in &l.sources {
            for &v in s {
                w.u32_le(v);
            }
        }
        for s in &l.source_weights {
            for &x in s {
                w.f32_le(x);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Reading

/// A checked reader over one section payload.
pub(crate) struct Dec<'a> {
    r: Reader<'a>,
    limits: &'a PackageLimits,
}

impl<'a> Dec<'a> {
    pub(crate) const fn new(data: &'a [u8], limits: &'a PackageLimits) -> Self {
        Self { r: Reader::new(data, Format::Chunk), limits }
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.r.u8()?)
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(self.r.u16_le()?)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(self.r.u32_le()?)
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(self.r.f32_le()?)
    }

    fn vec3(&mut self) -> Result<Vec3> {
        Ok(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }

    fn quat(&mut self) -> Result<Quat> {
        Ok(Quat::from_xyzw(self.f32()?, self.f32()?, self.f32()?, self.f32()?))
    }

    /// Reads a `u32` count and checks it against `limit` (and the package's element limit).
    fn count(&mut self, what: &'static str, limit: usize) -> Result<usize> {
        let n = self.u32()? as usize;
        let limit = limit.min(self.limits.max_elements);
        if n > limit {
            return Err(ModelError::limit(what, limit as u64, n as u64));
        }
        Ok(n)
    }

    /// Checks that `n` items of at least `item` bytes each fit in the rest of the payload.
    fn fits(&self, n: usize, item: usize) -> Result<()> {
        self.r.check_count(n, item)?;
        Ok(())
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u16()? as usize;
        if n > MAX_NAME_LEN.max(MAX_TEXT_LEN) {
            return Err(ModelError::limit("string length", MAX_TEXT_LEN as u64, n as u64));
        }
        let bytes = self.r.bytes(n)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| {
            IoError::Invalid { format: Format::Chunk, offset: self.r.pos(), what: "string is not UTF-8" }.into()
        })
    }

    fn parent(&mut self) -> Result<Option<u16>> {
        Ok(match self.u16()? {
            NO_PARENT => None,
            p => Some(p),
        })
    }

    fn array<T>(&mut self, n: usize, item: usize, mut read: impl FnMut(&mut Self) -> Result<T>) -> Result<Vec<T>> {
        self.fits(n, item)?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(read(self)?);
        }
        Ok(out)
    }

    fn u32s(&mut self, n: usize) -> Result<Vec<u32>> {
        self.array(n, 4, Self::u32)
    }

    fn u16s(&mut self, n: usize) -> Result<Vec<u16>> {
        self.array(n, 2, Self::u16)
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        self.array(n, 4, Self::f32)
    }

    fn enum_byte<T>(&mut self, what: &'static str, parse: impl Fn(u8) -> Option<T>) -> Result<T> {
        let at = self.r.pos();
        let b = self.u8()?;
        parse(b).ok_or_else(|| IoError::Invalid { format: Format::Chunk, offset: at, what }.into())
    }

    /// Fails unless the whole payload has been read.
    pub(crate) fn finish(&self) -> Result<()> {
        if self.r.is_empty() {
            Ok(())
        } else {
            Err(IoError::Invalid { format: Format::Chunk, offset: self.r.pos(), what: "trailing bytes in section" }
                .into())
        }
    }
}

pub(crate) fn read_meta(bytes: &[u8]) -> Result<TemplateInfo> {
    let value = json::parse_with(bytes, &JsonLimits::DEFAULT).map_err(|e| ModelError::Codec(e.into()))?;
    let root = value.node();
    let format: &str = root.get("format")?;
    if format != META_FORMAT {
        return Err(ModelError::Metadata(root.field("format")?.invalid("not a HumanArt template")));
    }
    let mut provenance = Vec::new();
    let list = root.field("provenance")?;
    if list.array_len()? > MAX_PROVENANCE {
        return Err(ModelError::limit("provenance entry count", MAX_PROVENANCE as u64, list.array_len()? as u64));
    }
    for p in list.elements()? {
        provenance.push(ProvenanceEntry { item: p.get("item")?, source: p.get("source")?, licence: p.get("licence")? });
    }
    let extra_node = root.field("extra")?;
    let extra = match extra_node.read::<&Value>()? {
        Value::Object(o) => o.clone(),
        _ => return Err(ModelError::Metadata(extra_node.invalid("expected an object"))),
    };
    Ok(TemplateInfo {
        name: root.get("name")?,
        version: root.get("version")?,
        generator: root.get("generator")?,
        licence: root.get("licence")?,
        provenance,
        extra,
    })
}

pub(crate) fn read_mesh(d: &mut Dec<'_>) -> Result<Mesh> {
    let v = d.count("vertex count", MAX_VERTICES.min(d.limits.max_vertices))?;
    let f = d.count("face count", MAX_FACES)?;
    let k = d.count("corner count", MAX_FACES.saturating_mul(4))?;
    let p = d.count("mesh part count", MAX_PARTS)?;
    let positions = d.array(v, 12, Dec::vec3)?;
    let uvs = d.array(v, 8, |d| Ok(Vec2::new(d.f32()?, d.f32()?)))?;
    let position_keys = d.u32s(v)?;
    let normal_keys = d.u32s(v)?;
    let face_offsets = d.u32s(f + 1)?;
    let face_vertices = d.u32s(k)?;
    let face_tiles = d.u16s(f)?;
    let parts = d.array(p, 11, |d| {
        Ok(MeshPart { name: d.string()?, first_face: d.u32()?, face_count: d.u32()?, closed: d.u8()? != 0 })
    })?;
    Mesh::new(MeshData { positions, uvs, face_offsets, face_vertices, face_tiles, position_keys, normal_keys, parts })
}

pub(crate) fn read_tiles(d: &mut Dec<'_>) -> Result<Vec<UvTile>> {
    let n = d.count("UV tile count", MAX_TILES)?;
    d.array(n, 10, |d| Ok(UvTile { name: d.string()?, resolution: d.u32()?, target_texels_per_mm: d.f32()? }))
}

pub(crate) fn read_regions(d: &mut Dec<'_>) -> Result<RegionMap> {
    let n = d.count("region count", MAX_REGIONS)?;
    let regions = d.array(n, 5, |d| {
        Ok(Region { name: d.string()?, parent: d.parent()?, side: d.enum_byte("bad side", Side::from_u8)? })
    })?;
    let v = d.count("vertex count", MAX_VERTICES.min(d.limits.max_vertices))?;
    let vertex_regions = d.u16s(v)?;
    let soft = match d.u8()? {
        0 => None,
        1 => {
            let e = d.count("soft region entry count", MAX_VERTICES.saturating_mul(16))?;
            let offsets = d.u32s(v + 1)?;
            let regions = d.u16s(e)?;
            let weights = d.f32s(e)?;
            Some(SoftRegionWeights { offsets, regions, weights })
        }
        _ => return Err(IoError::Invalid { format: Format::Chunk, offset: 0, what: "bad soft-weights flag" }.into()),
    };
    Ok(RegionMap { regions, vertex_regions, soft })
}

pub(crate) fn read_landmarks(d: &mut Dec<'_>) -> Result<Vec<Landmark>> {
    let n = d.count("landmark count", MAX_LANDMARKS)?;
    d.array(n, 27, |d| {
        Ok(Landmark {
            name: d.string()?,
            group: d.enum_byte("bad landmark group", LandmarkGroup::from_u8)?,
            vertices: [d.u32()?, d.u32()?, d.u32()?],
            weights: [d.f32()?, d.f32()?, d.f32()?],
        })
    })
}

pub(crate) fn read_skeleton(d: &mut Dec<'_>) -> Result<Skeleton> {
    let n = d.count("joint count", MAX_JOINTS)?;
    let joints = d.array(n, 69, |d| {
        Ok(Joint {
            name: d.string()?,
            parent: d.parent()?,
            kind: d.enum_byte("bad joint kind", JointKind::from_u8)?,
            rest: Transform::new(d.vec3()?, d.quat()?, d.vec3()?),
            limits: JointLimits { min: d.vec3()?, max: d.vec3()? },
        })
    })?;
    let c = d.count("regressor component count", MAX_COMPONENTS)?;
    let e = d.count("regressor entry count", MAX_COMPONENTS.saturating_mul(MAX_JOINTS))?;
    let offsets = d.u32s(c + 1)?;
    let joints_of = d.u16s(e)?;
    let deltas = d.array(e, 12, Dec::vec3)?;
    Ok(Skeleton { joints, regressor: JointRegressor { offsets, joints: joints_of, deltas } })
}

pub(crate) fn read_skin(d: &mut Dec<'_>) -> Result<SkinWeights> {
    let k = d.u8()? as usize;
    if k > MAX_INFLUENCES {
        return Err(ModelError::limit("skin influences", MAX_INFLUENCES as u64, k as u64));
    }
    let v = d.count("vertex count", MAX_VERTICES.min(d.limits.max_vertices))?;
    let joints = d.u16s(v * k)?;
    let weights = d.f32s(v * k)?;
    let r = d.count("dual-quaternion region count", MAX_REGIONS)?;
    let dual_quaternion_regions = d.u16s(r)?;
    Ok(SkinWeights { influences: k as u8, joints, weights, dual_quaternion_regions })
}

pub(crate) fn read_deltas(d: &mut Dec<'_>) -> Result<DeltaBases> {
    let c = d.count("delta component count", MAX_COMPONENTS)?;
    let e = d.count("delta entry count", crate::delta::MAX_DELTA_ENTRIES)?;
    let components = d.array(c, 5, |d| {
        Ok(DeltaComponent {
            name: d.string()?,
            kind: d.enum_byte("bad delta kind", DeltaKind::from_u8)?,
            region: d.u16()?,
        })
    })?;
    let offsets = d.u32s(c + 1)?;
    let indices = d.u32s(e)?;
    let deltas =
        d.array(e, 6, |d| Ok([F16::from_bits(d.u16()?), F16::from_bits(d.u16()?), F16::from_bits(d.u16()?)]))?;
    DeltaBases::from_parts(components, offsets, indices, deltas)
}

pub(crate) fn read_expressions(d: &mut Dec<'_>) -> Result<ExpressionSet> {
    let n = d.count("expression count", MAX_DEFINITIONS)?;
    let expressions = d.array(n, 10, |d| {
        Ok(Expression {
            name: d.string()?,
            kind: d.enum_byte("bad expression kind", ExpressionKind::from_u8)?,
            side: d.enum_byte("bad side", Side::from_u8)?,
            facs: d.u16()?,
            component: d.u32()?,
        })
    })?;
    let n = d.count("combination corrective count", MAX_DEFINITIONS)?;
    let combinations = d.array(n, 8, |d| {
        let name = d.string()?;
        let op = d.enum_byte("bad combination op", CombineOp::from_u8)?;
        let inputs = d.u8()? as usize;
        if inputs > MAX_COMBINATION_INPUTS {
            return Err(ModelError::limit("combination input count", MAX_COMBINATION_INPUTS as u64, inputs as u64));
        }
        Ok(CombinationCorrective { name, op, inputs: d.u32s(inputs)?, component: d.u32()? })
    })?;
    let n = d.count("pose corrective count", MAX_DEFINITIONS)?;
    let pose_correctives = d.array(n, 28, |d| {
        Ok(PoseCorrective {
            name: d.string()?,
            joint: d.u16()?,
            target: d.quat()?,
            radius: d.f32()?,
            component: d.u32()?,
        })
    })?;
    Ok(ExpressionSet { expressions, combinations, pose_correctives })
}

pub(crate) fn read_lods(d: &mut Dec<'_>) -> Result<Vec<LodLevel>> {
    let n = d.count("LOD level count", MAX_LOD_LEVELS)?;
    let mut lods = Vec::with_capacity(n);
    for _ in 0..n {
        let (declared_error, switch_size, hysteresis) = (d.f32()?, d.f32()?, d.f32()?);
        let mesh = read_mesh(d)?;
        let v = mesh.vertex_count();
        let sources = d.array(v, 12, |d| Ok([d.u32()?, d.u32()?, d.u32()?]))?;
        let source_weights = d.array(v, 12, |d| Ok([d.f32()?, d.f32()?, d.f32()?]))?;
        lods.push(LodLevel { mesh, sources, source_weights, declared_error, switch_size, hysteresis });
    }
    Ok(lods)
}
