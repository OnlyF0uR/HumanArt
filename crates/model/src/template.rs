//! The template `[MOD-01]`..`[MOD-07]` `[IO-21]`: everything the rest of the system knows about the
//! base character, in one checked value.
//!
//! [`TemplateParts`] holds the pieces with public fields, as a generator, the Template Adapter or
//! the package reader produce them; [`Template::new`] checks every cross-reference (vertex, region,
//! joint, tile and component indices, array lengths, names) and returns a [`Template`] that is
//! always consistent. The quality checks of `[AC-WP06-1]` (watertightness, UV overlap, ...) are in
//! [`crate::validate`].

use humanart_io::json::Object;
use humanart_math::Vec3;

use crate::delta::DeltaBases;
use crate::error::{ModelError, Result};
use crate::expression::ExpressionSet;
use crate::landmark::{Landmark, check_landmarks};
use crate::lod::{LodLevel, check_lods};
use crate::mesh::Mesh;
use crate::names::{check_name, check_names};
use crate::region::RegionMap;
use crate::skeleton::Skeleton;
use crate::skin::SkinWeights;

/// The most UV tiles a template may have.
pub const MAX_TILES: usize = 256;
/// The longest free-text field of the template info, in bytes.
pub const MAX_TEXT_LEN: usize = 4096;
/// The most provenance entries.
pub const MAX_PROVENANCE: usize = 1024;

/// One UV tile `[MOD-04]`: a texture page that faces map into, tile-local UVs in `[0, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct UvTile {
    /// Stable name (`face`, `torso`, `arms`, `legs`, `hands-feet`, `eyes`, `mouth`, `scalp`).
    pub name: String,
    /// Texels per side of the tile's hero texture, 1 to 65536.
    pub resolution: u32,
    /// The texel density the layout aims for, texels per millimetre (10 for face and neck).
    pub target_texels_per_mm: f32,
}

/// One line of the licence and provenance manifest: what contributed to the template.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProvenanceEntry {
    /// What the entry covers ("topology", "expression deltas").
    pub item: String,
    /// Where it came from ("procedural: humanart-model::generate").
    pub source: String,
    /// Its licence.
    pub licence: String,
}

/// Descriptive data of a template; stored as the JSON metadata section of the package.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TemplateInfo {
    /// Stable name of the template.
    pub name: String,
    /// Content version, raised whenever the template's data changes.
    pub version: u32,
    /// What produced it (generator and version, or the adapter and its source).
    pub generator: String,
    /// The licence of the template as a whole.
    pub licence: String,
    /// The provenance manifest.
    pub provenance: Vec<ProvenanceEntry>,
    /// Further JSON data carried with the template (parameter registry data, rig logic), by key.
    pub extra: Object,
}

impl TemplateInfo {
    /// Checks the name and text lengths.
    pub fn check(&self) -> Result<()> {
        check_name("template", &self.name)?;
        if self.provenance.len() > MAX_PROVENANCE {
            return Err(ModelError::limit(
                "provenance entry count",
                MAX_PROVENANCE as u64,
                self.provenance.len() as u64,
            ));
        }
        let texts = [&self.generator, &self.licence]
            .into_iter()
            .chain(self.provenance.iter().flat_map(|p| [&p.item, &p.source, &p.licence]));
        for t in texts {
            if t.len() > MAX_TEXT_LEN {
                return Err(ModelError::limit("template info text length", MAX_TEXT_LEN as u64, t.len() as u64));
            }
        }
        Ok(())
    }
}

/// The pieces of a template, unchecked.
#[derive(Clone, Debug, PartialEq)]
pub struct TemplateParts {
    /// Name, version, licence and provenance.
    pub info: TemplateInfo,
    /// LOD0, the hero mesh.
    pub mesh: Mesh,
    /// The UV tile table.
    pub tiles: Vec<UvTile>,
    /// Regions and the region of every vertex.
    pub regions: RegionMap,
    /// Landmarks.
    pub landmarks: Vec<Landmark>,
    /// The skeleton and joint regressor.
    pub skeleton: Skeleton,
    /// Skin weights.
    pub skin: SkinWeights,
    /// All delta components.
    pub deltas: DeltaBases,
    /// Expression, combination and pose-corrective definitions.
    pub expressions: ExpressionSet,
    /// LOD1 and further.
    pub lods: Vec<LodLevel>,
}

/// A checked template.
#[derive(Clone, Debug, PartialEq)]
pub struct Template {
    parts: TemplateParts,
}

impl Template {
    /// Checks every cross-reference of `parts` and wraps them.
    pub fn new(parts: TemplateParts) -> Result<Self> {
        let p = &parts;
        p.info.check()?;
        let v = p.mesh.vertex_count();
        check_tiles(&p.tiles)?;
        if let Some(f) = p.mesh.face_tiles().iter().position(|&t| t as usize >= p.tiles.len()) {
            return Err(ModelError::invalid("face", f, "UV tile index out of range"));
        }
        p.regions.check(v)?;
        check_landmarks(&p.landmarks, v)?;
        p.deltas.check_vertices(v, p.regions.regions.len())?;
        p.skeleton.check(p.deltas.len())?;
        p.skin.check(v, p.skeleton.joints.len(), p.regions.regions.len())?;
        p.expressions.check(&p.deltas, p.skeleton.joints.len())?;
        check_lods(&p.lods, v, p.tiles.len())?;
        Ok(Self { parts })
    }

    /// The pieces back.
    #[must_use]
    pub fn into_parts(self) -> TemplateParts {
        self.parts
    }

    /// All pieces.
    #[must_use]
    pub const fn parts(&self) -> &TemplateParts {
        &self.parts
    }

    /// Name, version, licence and provenance.
    #[must_use]
    pub const fn info(&self) -> &TemplateInfo {
        &self.parts.info
    }

    /// LOD0.
    #[must_use]
    pub const fn mesh(&self) -> &Mesh {
        &self.parts.mesh
    }

    /// The UV tiles.
    #[must_use]
    pub fn tiles(&self) -> &[UvTile] {
        &self.parts.tiles
    }

    /// The region map.
    #[must_use]
    pub const fn regions(&self) -> &RegionMap {
        &self.parts.regions
    }

    /// The landmarks.
    #[must_use]
    pub fn landmarks(&self) -> &[Landmark] {
        &self.parts.landmarks
    }

    /// The skeleton.
    #[must_use]
    pub const fn skeleton(&self) -> &Skeleton {
        &self.parts.skeleton
    }

    /// The skin weights.
    #[must_use]
    pub const fn skin(&self) -> &SkinWeights {
        &self.parts.skin
    }

    /// The delta bases.
    #[must_use]
    pub const fn deltas(&self) -> &DeltaBases {
        &self.parts.deltas
    }

    /// The expression definitions.
    #[must_use]
    pub const fn expressions(&self) -> &ExpressionSet {
        &self.parts.expressions
    }

    /// LOD1 and further.
    #[must_use]
    pub fn lods(&self) -> &[LodLevel] {
        &self.parts.lods
    }

    /// How many delta components come from scan statistics (0 in no-data operation `[ID-07]`).
    #[must_use]
    pub fn scan_component_count(&self) -> usize {
        self.parts.deltas.components().iter().filter(|c| c.kind.is_scan_derived()).count()
    }

    /// The LOD0 positions with weighted delta components added (`weights[c]` for component `c`,
    /// missing weights 0): the CPU form of the rest-shape sum of section 6.2.
    #[must_use]
    pub fn deformed_positions(&self, weights: &[f32]) -> Vec<Vec3> {
        let mut p = self.parts.mesh.positions().to_vec();
        for (c, &w) in weights.iter().enumerate().take(self.parts.deltas.len()) {
            if w != 0.0 {
                // The bases were checked against this mesh in `new`, so this cannot fail; an error
                // would leave `p` unchanged for that component.
                let _ = self.parts.deltas.accumulate(c, w, &mut p);
            }
        }
        p
    }
}

fn check_tiles(tiles: &[UvTile]) -> Result<()> {
    if tiles.len() > MAX_TILES {
        return Err(ModelError::limit("UV tile count", MAX_TILES as u64, tiles.len() as u64));
    }
    check_names("UV tile", tiles.iter().map(|t| t.name.as_str()))?;
    for (i, t) in tiles.iter().enumerate() {
        if !(1..=65_536).contains(&t.resolution) {
            return Err(ModelError::invalid("UV tile", i, "resolution outside 1..=65536"));
        }
        if !(t.target_texels_per_mm.is_finite() && t.target_texels_per_mm >= 0.0) {
            return Err(ModelError::invalid("UV tile", i, "target texel density is not finite and non-negative"));
        }
    }
    Ok(())
}

/// Small synthetic templates for the crate's tests.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;
    use crate::delta::{DeltaBasesBuilder, DeltaKind};
    use crate::expression::{CombinationCorrective, CombineOp, Expression, ExpressionKind, PoseCorrective};
    use crate::landmark::LandmarkGroup;
    use crate::mesh::fixtures::{cube_split, grid};
    use crate::names::Side;
    use crate::region::{Region, SoftRegionWeights};
    use crate::skeleton::{JointRegressor, simple_joint};
    use humanart_math::Quat;

    /// A complete template on the seam-split unit cube: two regions, landmarks, a two-joint chain,
    /// skin weights, semantic, expression and corrective deltas, one LOD level.
    pub fn cube_template() -> TemplateParts {
        let mesh = cube_split(false);
        let v = mesh.vertex_count();
        let regions = RegionMap {
            regions: vec![
                Region { name: "body".into(), parent: None, side: Side::Center },
                Region { name: "lower".into(), parent: Some(0), side: Side::Center },
                Region { name: "upper".into(), parent: Some(0), side: Side::Center },
            ],
            vertex_regions: mesh.positions().iter().map(|p| if p.y < 0.5 { 1 } else { 2 }).collect(),
            soft: Some(SoftRegionWeights {
                offsets: (0..=v as u32).collect(),
                regions: mesh.positions().iter().map(|p| if p.y < 0.5 { 1 } else { 2 }).collect(),
                weights: vec![1.0; v],
            }),
        };
        // Face 1 (z = 1) is vertices 4..8 of the split cube: corners (0,0,1), (1,0,1), (1,1,1), (0,1,1).
        let landmarks = vec![
            Landmark {
                name: "front.center".into(),
                group: LandmarkGroup::Face,
                vertices: [4, 5, 6],
                weights: [0.5, 0.0, 0.5],
            },
            Landmark {
                name: "C.base".into(),
                group: LandmarkGroup::Body,
                vertices: [0, 1, 2],
                weights: [0.2, 0.3, 0.5],
            },
        ];
        let mut b = DeltaBasesBuilder::new();
        let top: Vec<(u32, Vec3)> =
            (0..v as u32).filter(|&i| mesh.positions()[i as usize].y > 0.5).map(|i| (i, Vec3::Y * 0.25)).collect();
        let wide: Vec<(u32, Vec3)> =
            (0..v as u32).map(|i| (i, Vec3::new(mesh.positions()[i as usize].x - 0.5, 0.0, 0.0) * 0.1)).collect();
        b.add("height", DeltaKind::Semantic, 2, &top).unwrap();
        b.add("width", DeltaKind::Semantic, 0, &wide).unwrap();
        // Selections by position, so every duplicate of a corner moves alike (seam-consistent).
        let upper_where = |keep: &dyn Fn(Vec3) -> bool| -> Vec<(u32, Vec3)> {
            top.iter().copied().filter(|&(i, _)| keep(mesh.positions()[i as usize])).collect()
        };
        b.add("smile", DeltaKind::Expression, 2, &upper_where(&|p| p.x < 0.5)).unwrap();
        b.add("blink", DeltaKind::Expression, 2, &upper_where(&|p| p.x > 0.5)).unwrap();
        b.add("smile.blink", DeltaKind::CombinationCorrective, 2, &upper_where(&|p| p.x < 0.5 && p.z < 0.5)).unwrap();
        b.add("tip.bend", DeltaKind::PoseCorrective, 2, &upper_where(&|p| p.z > 0.5)).unwrap();
        let deltas = b.finish().unwrap();
        let find = |n: &str| deltas.find(n).unwrap();
        let mut regressor = JointRegressor::empty(deltas.len());
        let h = find("height") as usize;
        regressor.joints = vec![1];
        regressor.deltas = vec![Vec3::Y * 0.25];
        for o in &mut regressor.offsets[h + 1..] {
            *o = 1;
        }
        let skeleton = Skeleton {
            joints: vec![simple_joint("root", None, Vec3::new(0.5, 0.0, 0.5)), simple_joint("tip", Some(0), Vec3::Y)],
            regressor,
        };
        let lists: Vec<Vec<(u16, f32)>> = mesh.positions().iter().map(|p| vec![(0, 1.0 - p.y), (1, p.y)]).collect();
        let mut skin = SkinWeights::from_lists(4, lists.iter().map(Vec::as_slice), 0).unwrap();
        skin.dual_quaternion_regions = vec![2];
        let expressions = ExpressionSet {
            expressions: vec![
                Expression {
                    name: "smile".into(),
                    kind: ExpressionKind::ActionUnit,
                    side: Side::Center,
                    facs: 12,
                    component: find("smile"),
                },
                Expression {
                    name: "blink".into(),
                    kind: ExpressionKind::ActionUnit,
                    side: Side::Center,
                    facs: 45,
                    component: find("blink"),
                },
            ],
            combinations: vec![CombinationCorrective {
                name: "smile.blink".into(),
                op: CombineOp::Min,
                inputs: vec![0, 1],
                component: find("smile.blink"),
            }],
            pose_correctives: vec![PoseCorrective {
                name: "tip.bend".into(),
                joint: 1,
                target: Quat::from_axis_angle(Vec3::X, 1.0),
                radius: 1.0,
                component: find("tip.bend"),
            }],
        };
        // LOD1: the shared-vertex cube, each corner sourced from its first split vertex.
        let lod_mesh = crate::mesh::fixtures::cube_shared();
        let sources: Vec<[u32; 3]> = lod_mesh
            .positions()
            .iter()
            .map(|p| {
                let s = mesh.positions().iter().position(|q| q == p).unwrap() as u32;
                [s, s, s]
            })
            .collect();
        let lods = vec![LodLevel {
            source_weights: vec![[1.0, 0.0, 0.0]; sources.len()],
            sources,
            mesh: lod_mesh,
            declared_error: 0.0,
            switch_size: 0.3,
            hysteresis: 0.1,
        }];
        let mut extra = Object::default();
        extra.insert("rig_logic", Object::default());
        TemplateParts {
            info: TemplateInfo {
                name: "test-cube".into(),
                version: 3,
                generator: "humanart-model tests".into(),
                licence: "in-house".into(),
                provenance: vec![ProvenanceEntry {
                    item: "everything".into(),
                    source: "procedural: test fixture".into(),
                    licence: "in-house".into(),
                }],
                extra,
            },
            mesh,
            tiles: vec![UvTile { name: "body".into(), resolution: 1024, target_texels_per_mm: 0.5 }],
            regions,
            landmarks,
            skeleton,
            skin,
            deltas,
            expressions,
            lods,
        }
    }

    /// The smallest template: an open grid, one region, no skeleton, no deltas.
    pub fn bare_grid(n: usize) -> TemplateParts {
        let mesh = grid(n);
        let v = mesh.vertex_count();
        TemplateParts {
            info: TemplateInfo { name: "grid".into(), ..TemplateInfo::default() },
            tiles: vec![UvTile { name: "all".into(), resolution: 256, target_texels_per_mm: 0.0 }],
            regions: RegionMap {
                regions: vec![Region { name: "all".into(), parent: None, side: Side::Center }],
                vertex_regions: vec![0; v],
                soft: None,
            },
            landmarks: Vec::new(),
            skeleton: Skeleton { joints: Vec::new(), regressor: JointRegressor::empty(0) },
            skin: SkinWeights::default(),
            deltas: DeltaBases::empty(),
            expressions: ExpressionSet::default(),
            lods: Vec::new(),
            mesh,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    #[test]
    fn synthetic_templates_check() {
        let t = Template::new(cube_template()).unwrap();
        assert_eq!(t.mesh().vertex_count(), 24);
        assert_eq!(t.deltas().len(), 6);
        assert_eq!(t.scan_component_count(), 0);
        assert_eq!(t.skeleton().joints.len(), 2);
        assert_eq!(t.lods().len(), 1);
        let g = Template::new(bare_grid(2)).unwrap();
        assert_eq!(g.mesh().face_count(), 4);
        assert_eq!(g.into_parts().mesh.vertex_count(), 9);
    }

    #[test]
    fn deformed_positions_add_weighted_deltas() {
        let t = Template::new(cube_template()).unwrap();
        let h = t.deltas().find("height").unwrap() as usize;
        let mut w = vec![0.0; t.deltas().len()];
        w[h] = 2.0;
        let p = t.deformed_positions(&w);
        for (a, b) in t.mesh().positions().iter().zip(&p) {
            let expect = if a.y > 0.5 { a.y + 0.5 } else { a.y };
            assert_eq!(b.y, expect);
        }
        assert_eq!(t.deformed_positions(&[]), t.mesh().positions());
    }

    #[test]
    fn cross_references_are_checked() {
        type Change = fn(&mut TemplateParts);
        let cases: [(&str, Change); 10] = [
            ("tile index", |p| p.tiles.clear()),
            ("tile resolution", |p| p.tiles[0].resolution = 0),
            ("tile density", |p| p.tiles[0].target_texels_per_mm = f32::NAN),
            ("info name", |p| p.info.name = String::new()),
            ("info text", |p| p.info.licence = "x".repeat(MAX_TEXT_LEN + 1)),
            ("regions", |p| p.regions.vertex_regions.truncate(3)),
            ("landmarks", |p| p.landmarks[0].vertices[0] = 99),
            ("skin", |p| p.skin.joints[0] = 5),
            ("skeleton", |p| p.skeleton.regressor.offsets.truncate(3)),
            ("lods", |p| p.lods[0].sources[0][0] = 99),
        ];
        for (what, change) in cases {
            let mut p = cube_template();
            change(&mut p);
            assert!(Template::new(p).is_err(), "{what} was accepted");
        }
        let mut p = cube_template();
        p.expressions.expressions[0].component = 0;
        assert!(Template::new(p).is_err());
        let mut p = bare_grid(1);
        p.deltas = cube_template().deltas;
        assert!(Template::new(p).is_err());
    }
}
