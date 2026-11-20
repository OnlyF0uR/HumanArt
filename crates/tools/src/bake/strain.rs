//! Strain-reference baking `[BAKE-07]`: rest edge lengths and region-wise strain normalisation
//! tables.
//!
//! The runtime measures edge strain `s = l / l₀ − 1` against the rest lengths and reduces it per
//! region into a mean stretch and a mean compression (`humanart_model::eval::strain`,
//! `[EVAL-06]`). How much strain a region ever reaches differs a lot (the forehead compresses by a
//! few per cent, the corners of the mouth by tens), so the wrinkle and tension drivers need a
//! per-region scale. [`bake_strain_reference`] measures it: it evaluates the same reduction
//! (the model's own [`StrainField`], so the numbers are bit-identical to the runtime's) on a set of
//! reference poses (typically every expression target at full weight and the range-of-motion
//! extremes) and keeps, per region, the largest mean stretch and the largest mean compression seen.
//! [`StrainReference::normalise`] divides a runtime measurement by them, clamped to `[0, 1]`.
//!
//! **Binary form** ([`StrainReference::to_bytes`], [`StrainReference::from_bytes`]), little-endian,
//! stored as one chunk of a template or character package:
//!
//! ```text
//! magic  "HASTRAIN"                          8 bytes
//! u32    version (1)
//! u32    vertex count V                      ≤ 2^24
//! u32    edge count E                        ≤ 3 · 2^24
//! u32    region count R                      ≤ 4096
//! E ×    u32 a, u32 b                        weld-key end points, a < b < V
//! E ×    f32 rest length                     finite, ≥ 0
//! R ×    f32 max stretch, f32 max compression  finite, ≥ 0
//! ```
//!
//! The reader checks every count against its limit and the remaining input before allocating
//! `[ROB-01]`, and rejects trailing bytes.

use humanart_io::Format;
use humanart_io::bytes::Reader;
use humanart_math::Vec3;
use humanart_model::eval::strain::StrainField;
use humanart_model::region::{MAX_REGIONS, RegionMap};
use humanart_model::{Mesh, mesh::MAX_VERTICES};

use super::{BakeError, Result};

/// The file magic.
pub const MAGIC: [u8; 8] = *b"HASTRAIN";
/// The format version.
pub const VERSION: u32 = 1;
/// The most edges a table may have.
pub const MAX_EDGES: usize = 3 * MAX_VERTICES;
/// Region maxima below this are treated as "never strained": the region normalises to 0.
pub const MIN_REFERENCE: f32 = 1e-6;

/// The per-region reference strain.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RegionReference {
    /// The largest mean stretch of the region over the reference poses (≥ 0).
    pub stretch: f32,
    /// The largest mean compression of the region over the reference poses (≥ 0).
    pub compression: f32,
}

/// The baked strain reference of a template.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StrainReference {
    /// The number of vertices of the mesh it was baked for.
    pub vertex_count: u32,
    /// The welded edges (weld-key pairs, `a < b`).
    pub edges: Vec<[u32; 2]>,
    /// The rest length of every edge, metres.
    pub rest_lengths: Vec<f32>,
    /// One entry per region of the region map.
    pub regions: Vec<RegionReference>,
}

/// Measures the strain reference of `mesh` with its region map: rest lengths from `rest`, region
/// maxima over `poses` (each one position per vertex).
pub fn bake_strain_reference(
    mesh: &Mesh,
    regions: &RegionMap,
    rest: &[Vec3],
    poses: &[&[Vec3]],
) -> Result<StrainReference> {
    let n = mesh.vertex_count();
    regions.check(n)?;
    if rest.len() != n || poses.iter().any(|p| p.len() != n) {
        return Err(BakeError::invalid("strain: a pose does not have one position per vertex"));
    }
    if rest.iter().chain(poses.iter().flat_map(|p| p.iter())).any(|p| !p.is_finite()) {
        return Err(BakeError::invalid("strain: a position is not finite"));
    }
    let topology = mesh.topology();
    let mut field = StrainField::new(mesh, &topology, regions);
    field.set_rest(rest);
    let mut table = vec![RegionReference::default(); regions.regions.len()];
    for pose in poses {
        field.compute(pose);
        for (r, entry) in table.iter_mut().enumerate() {
            let s = field.region_stretch().get(r).copied().unwrap_or(0.0);
            let c = field.region_compression().get(r).copied().unwrap_or(0.0);
            entry.stretch = entry.stretch.max(s);
            entry.compression = entry.compression.max(c);
        }
    }
    let edges: Vec<[u32; 2]> = field.edges().iter().map(|&[a, b]| [a.min(b), a.max(b)]).collect();
    Ok(StrainReference { vertex_count: n as u32, edges, rest_lengths: field.rest_lengths().to_vec(), regions: table })
}

impl StrainReference {
    /// A runtime region measurement `(stretch, compression)` of region `r` divided by the
    /// region's reference, each clamped to `[0, 1]`; 0 for a region the poses never strained or
    /// that does not exist.
    pub fn normalise(&self, r: usize, stretch: f32, compression: f32) -> (f32, f32) {
        let Some(e) = self.regions.get(r) else { return (0.0, 0.0) };
        let f = |v: f32, max: f32| if max >= MIN_REFERENCE && v.is_finite() { (v / max).clamp(0.0, 1.0) } else { 0.0 };
        (f(stretch, e.stretch), f(compression, e.compression))
    }

    /// The binary form (see the module documentation).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(24 + self.edges.len() * 12 + self.regions.len() * 8);
        out.extend_from_slice(&MAGIC);
        for v in [VERSION, self.vertex_count, self.edges.len() as u32, self.regions.len() as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &[a, b] in &self.edges {
            out.extend_from_slice(&a.to_le_bytes());
            out.extend_from_slice(&b.to_le_bytes());
        }
        for l in &self.rest_lengths {
            out.extend_from_slice(&l.to_le_bytes());
        }
        for r in &self.regions {
            out.extend_from_slice(&r.stretch.to_le_bytes());
            out.extend_from_slice(&r.compression.to_le_bytes());
        }
        out
    }

    /// Reads the binary form, checking every count and value (see the module documentation).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes, Format::Chunk);
        if r.array::<8>()? != MAGIC {
            return Err(BakeError::invalid("strain table: bad magic"));
        }
        if r.u32_le()? != VERSION {
            return Err(BakeError::invalid("strain table: unsupported version"));
        }
        let vertex_count = r.u32_le()?;
        let edge_count = r.len_u32_le()?;
        let region_count = r.len_u32_le()?;
        if vertex_count as usize > MAX_VERTICES || edge_count > MAX_EDGES || region_count > MAX_REGIONS {
            return Err(BakeError::invalid("strain table: a count exceeds its limit"));
        }
        // Everything that follows has a fixed size: check it all before allocating.
        let need = edge_count.checked_mul(12).and_then(|e| e.checked_add(region_count * 8));
        if need != Some(r.remaining()) {
            return Err(BakeError::invalid("strain table: the size does not match the counts"));
        }
        let mut edges = Vec::with_capacity(edge_count);
        for _ in 0..edge_count {
            let (a, b) = (r.u32_le()?, r.u32_le()?);
            if a >= b || b >= vertex_count {
                return Err(BakeError::invalid("strain table: an edge is out of range or not ordered"));
            }
            edges.push([a, b]);
        }
        let ok = |v: f32| v.is_finite() && v >= 0.0;
        let mut rest_lengths = Vec::with_capacity(edge_count);
        for _ in 0..edge_count {
            let l = r.f32_le()?;
            if !ok(l) {
                return Err(BakeError::invalid("strain table: a rest length is negative or not finite"));
            }
            rest_lengths.push(l);
        }
        let mut regions = Vec::with_capacity(region_count);
        for _ in 0..region_count {
            let (stretch, compression) = (r.f32_le()?, r.f32_le()?);
            if !ok(stretch) || !ok(compression) {
                return Err(BakeError::invalid("strain table: a region maximum is negative or not finite"));
            }
            regions.push(RegionReference { stretch, compression });
        }
        Ok(Self { vertex_count, edges, rest_lengths, regions })
    }
}

#[cfg(test)]
mod tests {
    use humanart_model::region::Region;
    use humanart_model::{MeshBuilder, Side};

    use super::*;

    /// A strip of 4 quads; region 0 = left half, region 1 = right half, region 2 their parent.
    fn strip() -> (Mesh, RegionMap) {
        let mut b = MeshBuilder::new();
        b.begin_part("strip", false).unwrap();
        let mut top = Vec::new();
        let mut bottom = Vec::new();
        for i in 0..5 {
            let x = i as f32 * 0.01;
            top.push(b.vertex(Vec3::new(x, 0.01, 0.0), humanart_math::Vec2::new(x * 10.0, 0.0)));
            bottom.push(b.vertex(Vec3::new(x, 0.0, 0.0), humanart_math::Vec2::new(x * 10.0, 0.1)));
        }
        for i in 0..4 {
            b.face(&[top[i], bottom[i], bottom[i + 1], top[i + 1]], 0).unwrap();
        }
        let mesh = b.finish().unwrap();
        let region = |name: &str, parent| Region { name: name.into(), parent, side: Side::Center };
        let regions = RegionMap {
            regions: vec![region("cheek", None), region("cheek.left", Some(0)), region("cheek.right", Some(0))],
            vertex_regions: mesh.positions().iter().map(|p| if p.x < 0.02 { 1 } else { 2 }).collect(),
            soft: None,
        };
        (mesh, regions)
    }

    #[test]
    fn bake_07_table_holds_the_largest_region_strain() {
        let (mesh, regions) = strip();
        let rest = mesh.positions().to_vec();
        // Pose A stretches everything along x by 10 %; pose B compresses the right half by 20 %.
        let a: Vec<Vec3> = rest.iter().map(|p| Vec3::new(p.x * 1.1, p.y, p.z)).collect();
        let b: Vec<Vec3> =
            rest.iter().map(|p| if p.x > 0.02 { Vec3::new(0.02 + (p.x - 0.02) * 0.8, p.y, p.z) } else { *p }).collect();
        let table = bake_strain_reference(&mesh, &regions, &rest, &[&a, &b]).unwrap();
        assert_eq!(table.edges.len(), 13);
        assert!(table.rest_lengths.iter().all(|&l| (l - 0.01).abs() < 1e-6));
        // Same reduction as the runtime: compute pose B through the model and compare.
        let mut field = StrainField::new(&mesh, &mesh.topology(), &regions);
        field.compute(&b);
        assert_eq!(table.regions[2].compression, field.region_compression()[2]);
        assert!(table.regions[2].compression > 0.05);
        assert_eq!(table.regions[1].compression, field.region_compression()[1]);
        assert!(table.regions[1].stretch > 0.03 && table.regions[1].stretch < 0.1);
        // Normalised: the reference pose maps to 1, half of it to 0.5, a missing region to 0.
        let r2 = table.regions[2];
        assert_eq!(table.normalise(2, r2.stretch, r2.compression), (1.0, 1.0));
        let (s, _) = table.normalise(2, 0.5 * r2.stretch, 0.0);
        assert!((s - 0.5).abs() < 1e-6);
        assert_eq!(table.normalise(99, 1.0, 1.0), (0.0, 0.0));
        assert!(bake_strain_reference(&mesh, &regions, &rest[1..], &[]).is_err());
    }

    #[test]
    fn bake_07_binary_round_trip_and_rejects_bad_input() {
        let (mesh, regions) = strip();
        let rest = mesh.positions().to_vec();
        let a: Vec<Vec3> = rest.iter().map(|p| *p * 1.05).collect();
        let table = bake_strain_reference(&mesh, &regions, &rest, &[&a]).unwrap();
        let bytes = table.to_bytes();
        assert_eq!(StrainReference::from_bytes(&bytes).unwrap(), table);
        assert!(StrainReference::from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(StrainReference::from_bytes(&extra).is_err());
        let mut huge = bytes.clone();
        huge[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(StrainReference::from_bytes(&huge).is_err());
        let mut nan = bytes;
        let at = nan.len() - 4;
        nan[at..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(StrainReference::from_bytes(&nan).is_err());
    }

    #[test]
    fn rob_01_strain_table_fuzz() {
        let (mesh, regions) = strip();
        let rest = mesh.positions().to_vec();
        let a: Vec<Vec3> = rest.iter().map(|p| *p * 0.97).collect();
        let corpus = vec![bake_strain_reference(&mesh, &regions, &rest, &[&a]).unwrap().to_bytes()];
        humanart_core::fuzz::run("strain-table", &corpus, 20_000, 0x57A1, std::time::Duration::from_secs(60), |b| {
            if let Ok(t) = StrainReference::from_bytes(b) {
                assert_eq!(t.edges.len(), t.rest_lengths.len());
                assert_eq!(StrainReference::from_bytes(&t.to_bytes()).unwrap(), t);
            }
        });
    }
}
