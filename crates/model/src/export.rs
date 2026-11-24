//! The character as glTF 2.0 (WP-29, `[EXP-P1]`): its rest shape with every identity and sculpt
//! setting applied, the skeleton as a node hierarchy, the skin weights with inverse bind
//! matrices, one primitive per mesh part and UV tile, and the expressions as morph targets.
//!
//! - **Geometry.** The rest shape comes from an [`Evaluator`] evaluated with the document's shape
//!   and layer weights, no expression and the rest pose ([`RestShape::of`]): positions, normals
//!   and tangents (`w` the handedness). Faces are triangulated as the renderer does. A render
//!   vertex lies on exactly one UV tile (UV seams duplicate vertices), so each `(part, tile)` group
//!   becomes one primitive whose `TEXCOORD_0` is the tile's own UV (top-left origin, V down, as
//!   glTF has it).
//! - **Skeleton.** One node per joint, parents first, with the rest rotation and the bind position
//!   the joint regressor gives for the shape (`[MOD-07]`), relative to the parent; the skin's
//!   inverse bind matrices invert the joints' world rest transforms. Rotating a node about its
//!   local axes poses the joint exactly as a pose rotation does here (Y along the bone).
//! - **Weights.** `JOINTS_0` / `WEIGHTS_0`, the template's four influences per vertex normalised
//!   to sum 1.
//! - **Morph targets.** Every expression target as a sparse `POSITION` target, named in
//!   `extras.targetNames`, default weight 0. Combination and pose correctives depend on other
//!   weights and on the pose, which glTF cannot express; they are left out and named in the mesh's
//!   extras, so a runtime that knows them can add them.
//! - **Materials.** One per part: skin (base colour from the skin values through the pigment
//!   model, the primary roughness, the subsurface radii in extras), eyes, teeth, gums, the mouth's
//!   inside and the tongue with authored colours. No textures: the skin maps are baked on the GPU
//!   (`humanart-render`), which this crate does not use.
//!
//! Coordinates are the project's (right-handed, Y up, metres, facing +Z), which are glTF's.
//! The asset's `copyright` carries the template's licence; the face model's attribution
//! (`DATA-ATTRIBUTION.md`) goes in the asset's extras.

use std::collections::BTreeMap;

use humanart_io::gltf::{
    Builder, ElementType, Gltf, Material, Mesh as GltfMesh, Node, Pbr, Primitive, Scene, Skin, Target, Transform,
};
use humanart_io::json::{Object, Value};
use humanart_math::{Mat4, Quat, Vec3, Vec4};

use crate::error::{ModelError, Result};
use crate::eval::{EvalInput, Evaluator};
use crate::skin_appearance::channels::Attribute;
use crate::skin_appearance::pigment::{PigmentModel, PigmentParams};
use crate::skin_appearance::values::SkinValues;

/// The character's rest shape as the exporter needs it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RestShape {
    /// Positions per render vertex.
    pub positions: Vec<Vec3>,
    /// Unit normals per render vertex.
    pub normals: Vec<Vec3>,
    /// Tangents per render vertex (`w` = handedness).
    pub tangents: Vec<Vec4>,
    /// The joints' bind positions for the shape.
    pub bind_positions: Vec<Vec3>,
}

impl RestShape {
    /// The rest shape of `eval`'s template for the shape and layer weights of `input` (its
    /// expressions and pose are ignored). `eval` is not changed.
    pub fn of(eval: &Evaluator, input: &EvalInput<'_>) -> Result<Self> {
        let mut e = eval.clone();
        let rest = EvalInput { shape: input.shape, layers: input.layers, ..EvalInput::default() };
        e.evaluate(&rest).map_err(|_| ModelError::invalid("export", 0, "the rest shape could not be evaluated"))?;
        Ok(Self {
            positions: e.positions().to_vec(),
            normals: e.normals().to_vec(),
            tangents: e.tangents().to_vec(),
            bind_positions: e.joints().bind_positions.clone(),
        })
    }
}

/// What to export.
#[derive(Clone, Debug, PartialEq)]
pub struct ExportOptions {
    /// The scene's and the mesh's name.
    pub name: String,
    /// Whether the expressions become morph targets.
    pub expressions: bool,
    /// The skin values the skin material's colour is made from.
    pub skin: SkinValues,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self { name: "character".into(), expressions: true, skin: SkinValues::default() }
    }
}

/// What an export wrote.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExportReport {
    /// Primitives (one per part and UV tile).
    pub primitives: usize,
    /// Vertices over all primitives.
    pub vertices: usize,
    /// Triangles over all primitives.
    pub triangles: usize,
    /// Joints in the skin.
    pub joints: usize,
    /// Morph targets.
    pub targets: usize,
}

/// The part's material colour (linear Rec.709) and roughness for the parts that are not skin.
fn part_look(part: &str) -> Option<([f32; 3], f32)> {
    let p = part.to_ascii_lowercase();
    Some(if p.starts_with("eyeball") || p.starts_with("eye-") {
        ([0.80, 0.78, 0.76], 0.15)
    } else if p.starts_with("teeth") {
        ([0.78, 0.74, 0.62], 0.3)
    } else if p.starts_with("gums") {
        ([0.55, 0.20, 0.20], 0.35)
    } else if p.starts_with("tongue") {
        ([0.55, 0.22, 0.22], 0.4)
    } else if p.starts_with("mouth") {
        ([0.35, 0.10, 0.10], 0.4)
    } else {
        return None;
    })
}

/// The skin material for `skin`: the pigment model's diffuse albedo, the primary roughness and
/// the subsurface radii (millimetres, red, green, blue) in extras.
fn skin_material(skin: &SkinValues) -> Material {
    let v = |a: Attribute| f64::from(skin.base.0.get(a.index()).copied().unwrap_or(0.0));
    let params = PigmentParams {
        melanin: v(Attribute::Melanin),
        eumelanin_ratio: v(Attribute::EumelaninRatio),
        haemoglobin: v(Attribute::Haemoglobin),
        oxygenation: v(Attribute::Oxygenation),
        carotene: v(Attribute::Carotene),
        epidermal_thickness: v(Attribute::EpidermalThickness),
    };
    let response = PigmentModel::default().evaluate(&params);
    let [r, g, b] = response.albedo.map(|c| c as f32);
    let mut extras = Object::new();
    extras.insert("subsurfaceRadiusMm", Value::Array(response.sss_radius_mm.iter().map(|&x| Value::from(x)).collect()));
    let mut m = Material {
        name: Some("skin".into()),
        pbr: Pbr {
            base_color_factor: [r, g, b, 1.0],
            metallic_factor: 0.0,
            roughness_factor: v(Attribute::RoughnessPrimary).clamp(0.0, 1.0) as f32,
            ..Pbr::default()
        },
        ..Material::default()
    };
    m.extra.extras = Some(Value::Object(extras));
    m
}

/// The world rest transform of joint `j` for the shape: rotation `rest`, position `bind`.
fn world(rest: Quat, bind: Vec3) -> Mat4 {
    Mat4::from_rotation_translation(rest, bind)
}

/// Builds the glTF of `template` (the template `rest` was evaluated on).
pub fn character_gltf(
    template: &crate::Template,
    rest: &RestShape,
    options: &ExportOptions,
) -> Result<(Gltf, ExportReport)> {
    let mesh = template.mesh();
    let n = mesh.vertex_count();
    if rest.positions.len() != n || rest.normals.len() != n || rest.tangents.len() != n {
        return Err(ModelError::invalid("export", 0, "the rest shape does not match the template"));
    }
    let skeleton = template.skeleton();
    let joints = skeleton.joints.len();
    if rest.bind_positions.len() != joints {
        return Err(ModelError::invalid("export", 0, "the bind positions do not match the skeleton"));
    }
    let gltf_err = |_| ModelError::invalid("export", 0, "the glTF builder refused the data");
    let mut b = Builder::new("HumanArt (humanart-model export)");
    let info = template.info();
    b.document.asset.copyright = Some(format!("{} ({})", info.name, info.licence));
    let mut asset_extras = Object::new();
    asset_extras.insert(
        "provenance",
        Value::Array(
            info.provenance
                .iter()
                .map(|p| {
                    let mut o = Object::new();
                    o.insert("item", p.item.as_str());
                    o.insert("source", p.source.as_str());
                    o.insert("licence", p.licence.as_str());
                    Value::Object(o)
                })
                .collect(),
        ),
    );
    b.document.asset.extra.extras = Some(Value::Object(asset_extras));

    // Skeleton nodes: parents first, so node i is joint i.
    let rest_world = skeleton.rest_world();
    let mut roots = Vec::new();
    let mut inverse_bind: Vec<f32> = Vec::with_capacity(joints * 16);
    for (j, joint) in skeleton.joints.iter().enumerate() {
        let rot = rest_world.get(j).map_or(Quat::IDENTITY, |t| t.rotation).normalize();
        let pos = rest.bind_positions[j];
        let (local_rot, local_pos) = if let Some(p) = joint.parent {
            let p = p as usize;
            let prot = rest_world.get(p).map_or(Quat::IDENTITY, |t| t.rotation).normalize();
            let ppos = rest.bind_positions.get(p).copied().unwrap_or(Vec3::ZERO);
            let inv = prot.conjugate();
            ((inv * rot).normalize(), inv.mul_vec3(pos - ppos))
        } else {
            roots.push(j);
            (rot, pos)
        };
        b.document.nodes.push(Node {
            name: Some(joint.name.clone()),
            transform: Transform::Trs {
                translation: local_pos.to_array(),
                rotation: local_rot.to_array(),
                scale: [1.0; 3],
            },
            ..Node::default()
        });
        let inv = world(rot, pos).inverse_affine().ok_or(ModelError::invalid(
            "export",
            j,
            "a joint's rest transform cannot be inverted",
        ))?;
        inverse_bind.extend_from_slice(&inv.to_cols_array());
    }
    for (j, joint) in skeleton.joints.iter().enumerate() {
        if let Some(p) = joint.parent
            && let Some(parent) = b.document.nodes.get_mut(p as usize)
        {
            parent.children.push(j);
        }
    }
    let ibm = b.accessor_f32(&inverse_bind, ElementType::Mat4, None).map_err(gltf_err)?;
    b.document.skins.push(Skin {
        name: Some(format!("{}-skin", options.name)),
        inverse_bind_matrices: Some(ibm),
        skeleton: roots.first().copied(),
        joints: (0..joints).collect(),
        ..Skin::default()
    });

    // Materials: skin for the body, one per other part kind.
    let mut materials: BTreeMap<String, usize> = BTreeMap::new();
    let mut material_of = |b: &mut Builder, part: &str| -> usize {
        let (key, make): (String, Box<dyn Fn() -> Material>) = match part_look(part) {
            Some((rgb, rough)) => {
                let key = part.trim_end_matches(['L', 'R', '-']).to_owned();
                let name = key.clone();
                (
                    key,
                    Box::new(move || Material {
                        name: Some(name.clone()),
                        pbr: Pbr {
                            base_color_factor: [rgb[0], rgb[1], rgb[2], 1.0],
                            metallic_factor: 0.0,
                            roughness_factor: rough,
                            ..Pbr::default()
                        },
                        ..Material::default()
                    }),
                )
            }
            None => ("skin".to_owned(), Box::new(|| skin_material(&options.skin))),
        };
        *materials.entry(key).or_insert_with(|| {
            b.document.materials.push(make());
            b.document.materials.len() - 1
        })
    };

    // Primitives by (part, tile).
    let tri = mesh.triangulate();
    let tiles = mesh.face_tiles();
    let mut groups: BTreeMap<(usize, u16), Vec<[u32; 3]>> = BTreeMap::new();
    for (corners, &face) in tri.indices.as_chunks::<3>().0.iter().zip(&tri.triangle_faces) {
        let f = face as usize;
        let Some(part) = mesh.part_of_face(f) else { continue };
        let tile = tiles.get(f).copied().unwrap_or(0);
        groups.entry((part, tile)).or_default().push(*corners);
    }
    let expressions: Vec<(String, usize)> = if options.expressions {
        template.expressions().expressions.iter().map(|e| (e.name.clone(), e.component as usize)).collect()
    } else {
        Vec::new()
    };
    // The dense delta of every expression, per render vertex (sparse in the template).
    let deltas = template.deltas();
    let skin = template.skin();
    let uvs = mesh.uvs();
    let mut report = ExportReport { joints, targets: expressions.len(), ..ExportReport::default() };
    let mut primitives = Vec::with_capacity(groups.len());
    let mut local = vec![u32::MAX; n];
    for ((part, _tile), triangles) in &groups {
        // Local vertex numbering in first-use order.
        let mut used: Vec<u32> = Vec::new();
        for t in triangles {
            for &v in t {
                if let Some(slot) = local.get_mut(v as usize)
                    && *slot == u32::MAX
                {
                    *slot = used.len() as u32;
                    used.push(v);
                }
            }
        }
        let indices: Vec<u32> = triangles.iter().flatten().map(|&v| local[v as usize]).collect();
        let pick3 = |data: &[Vec3]| -> Vec<f32> { used.iter().flat_map(|&v| data[v as usize].to_array()).collect() };
        let positions = pick3(&rest.positions);
        let normals: Vec<f32> =
            used.iter().flat_map(|&v| rest.normals[v as usize].normalize_or(Vec3::Y).to_array()).collect();
        let tangents: Vec<f32> = used
            .iter()
            .flat_map(|&v| {
                let t = rest.tangents[v as usize];
                let xyz = Vec3::new(t.x, t.y, t.z).normalize_or(Vec3::X);
                [xyz.x, xyz.y, xyz.z, if t.w < 0.0 { -1.0 } else { 1.0 }]
            })
            .collect();
        let texcoords: Vec<f32> =
            used.iter().flat_map(|&v| uvs.get(v as usize).map_or([0.0, 0.0], |uv| [uv.x, uv.y])).collect();
        let mut joints4: Vec<[u16; 4]> = Vec::with_capacity(used.len());
        let mut weights4: Vec<f32> = Vec::with_capacity(used.len() * 4);
        for &v in &used {
            let (js, ws) = skin.vertex(v as usize);
            let mut j = [0_u16; 4];
            let mut w = [0.0_f32; 4];
            for (k, (&jj, &ww)) in js.iter().zip(ws).take(4).enumerate() {
                j[k] = jj;
                w[k] = ww.max(0.0);
            }
            let sum: f32 = w.iter().sum();
            if sum > 0.0 {
                for x in &mut w {
                    *x /= sum;
                }
            } else {
                // A vertex without influences follows the root.
                j = [roots.first().copied().unwrap_or(0) as u16, 0, 0, 0];
                w = [1.0, 0.0, 0.0, 0.0];
            }
            joints4.push(j);
            weights4.extend_from_slice(&w);
        }
        let attributes = vec![
            (
                "POSITION".to_owned(),
                b.accessor_f32(&positions, ElementType::Vec3, Some(Target::ArrayBuffer)).map_err(gltf_err)?,
            ),
            (
                "NORMAL".to_owned(),
                b.accessor_f32(&normals, ElementType::Vec3, Some(Target::ArrayBuffer)).map_err(gltf_err)?,
            ),
            (
                "TANGENT".to_owned(),
                b.accessor_f32(&tangents, ElementType::Vec4, Some(Target::ArrayBuffer)).map_err(gltf_err)?,
            ),
            (
                "TEXCOORD_0".to_owned(),
                b.accessor_f32(&texcoords, ElementType::Vec2, Some(Target::ArrayBuffer)).map_err(gltf_err)?,
            ),
            ("JOINTS_0".to_owned(), b.accessor_joints(&joints4).map_err(gltf_err)?),
            (
                "WEIGHTS_0".to_owned(),
                b.accessor_f32(&weights4, ElementType::Vec4, Some(Target::ArrayBuffer)).map_err(gltf_err)?,
            ),
        ];
        let mut targets = Vec::with_capacity(expressions.len());
        let mut dense = vec![0.0_f32; used.len() * 3];
        for (_, c) in &expressions {
            dense.fill(0.0);
            for (&v, &d) in deltas.indices(*c).iter().zip(deltas.deltas(*c)) {
                let l = local.get(v as usize).copied().unwrap_or(u32::MAX);
                if l != u32::MAX {
                    let d = crate::delta::DeltaBases::delta_f32(d);
                    let at = l as usize * 3;
                    dense[at..at + 3].copy_from_slice(&d.to_array());
                }
            }
            targets.push(vec![("POSITION".to_owned(), b.accessor_morph(&dense, ElementType::Vec3).map_err(gltf_err)?)]);
        }
        let part_name = mesh.parts().get(*part).map_or("body", |p| p.name.as_str());
        let material = material_of(&mut b, part_name);
        primitives.push(Primitive {
            attributes,
            indices: Some(b.accessor_indices(&indices).map_err(gltf_err)?),
            material: Some(material),
            targets,
            ..Primitive::default()
        });
        report.primitives += 1;
        report.vertices += used.len();
        report.triangles += triangles.len();
        for &v in &used {
            local[v as usize] = u32::MAX;
        }
    }
    let mut gmesh = GltfMesh {
        name: Some(options.name.clone()),
        primitives,
        weights: vec![0.0; expressions.len()],
        ..GltfMesh::default()
    };
    let names: Vec<&str> = expressions.iter().map(|(n, _)| n.as_str()).collect();
    if !names.is_empty() {
        gmesh.set_target_names(&names);
    }
    if gmesh.extra.extras.is_none() {
        gmesh.extra.extras = Some(Value::Object(Object::new()));
    }
    if let Some(Value::Object(o)) = &mut gmesh.extra.extras {
        let set = template.expressions();
        o.insert(
            "notExported",
            Value::Array(
                set.combinations
                    .iter()
                    .map(|c| c.name.as_str())
                    .chain(set.pose_correctives.iter().map(|p| p.name.as_str()))
                    .map(Value::from)
                    .collect(),
            ),
        );
    }
    b.document.meshes.push(gmesh);
    b.document.nodes.push(Node { name: Some(options.name.clone()), mesh: Some(0), skin: Some(0), ..Node::default() });
    let mesh_node = b.document.nodes.len() - 1;
    let mut scene_nodes = roots;
    scene_nodes.push(mesh_node);
    b.document.scenes.push(Scene { name: Some(options.name.clone()), nodes: scene_nodes, ..Scene::default() });
    b.document.scene = Some(0);
    let gltf = b.finish().map_err(gltf_err)?;
    Ok((gltf, report))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use humanart_io::Limits;
    use humanart_io::gltf::read_self_contained;

    use super::*;
    use crate::generate::build::{Options, template};

    #[test]
    fn exp_p1_the_character_exports_as_a_valid_skinned_glb_with_morph_targets() {
        let t = Arc::new(template(&Options { lods: false, ..Options::default() }).unwrap().template);
        let eval = Evaluator::new(t.clone());
        // A shape weight on: the export carries the evaluated shape, not the template's.
        let mut shape = vec![0.0; t.deltas().len()];
        let sex = t.deltas().find(crate::identity::sex::SWITCH).unwrap() as usize;
        shape[sex] = -1.0;
        let rest = RestShape::of(&eval, &EvalInput { shape: &shape, ..EvalInput::default() }).unwrap();
        let (gltf, report) = character_gltf(&t, &rest, &ExportOptions::default()).unwrap();
        println!("[EXP-P1] {report:?}");
        assert_eq!(report.joints, t.skeleton().joints.len());
        assert_eq!(report.targets, t.expressions().expressions.len());
        assert_eq!(report.triangles, t.mesh().triangulate().len());
        let glb = gltf.to_glb().unwrap();
        println!("[EXP-P1] {} bytes", glb.len());
        let back = read_self_contained(&glb, &Limits::DEFAULT).unwrap();
        assert_eq!(back, gltf);
        let doc = &back.document;
        assert_eq!(doc.meshes[0].target_names().len(), report.targets);
        // Positions round-trip bit for bit, and they are the evaluated shape's.
        let mut seen = 0;
        let mut weight_error = 0.0_f32;
        for p in &doc.meshes[0].primitives {
            assert_eq!(p.targets.len(), report.targets, "every primitive carries every target");
            let pos = back.read_f32(p.attribute("POSITION").unwrap()).unwrap();
            let w = back.read_f32(p.attribute("WEIGHTS_0").unwrap()).unwrap();
            for q in w.chunks(4) {
                weight_error = weight_error.max((q.iter().sum::<f32>() - 1.0).abs());
            }
            seen += pos.len() / 3;
        }
        assert_eq!(seen, report.vertices);
        assert!(weight_error < 1e-5, "weights sum to 1 within {weight_error}");
        let all: Vec<f32> = doc.meshes[0]
            .primitives
            .iter()
            .flat_map(|p| back.read_f32(p.attribute("POSITION").unwrap()).unwrap())
            .collect();
        let has = |q: Vec3| all.chunks(3).any(|c| c[0] == q.x && c[1] == q.y && c[2] == q.z);
        assert!(has(rest.positions[0]) && has(rest.positions[rest.positions.len() / 2]));
        assert_ne!(rest.positions, t.mesh().positions(), "the shape weight moved the mesh");
        // The inverse bind matrices invert the joints' world rest transforms built from the nodes.
        let ibm = back.read_f32(doc.skins[0].inverse_bind_matrices.unwrap()).unwrap();
        let mut worlds: Vec<Mat4> = Vec::with_capacity(doc.skins[0].joints.len());
        for (j, joint) in t.skeleton().joints.iter().enumerate() {
            let local = Mat4::from_cols_array(doc.nodes[j].transform.matrix());
            let w = match joint.parent {
                Some(p) => worlds[p as usize] * local,
                None => local,
            };
            let m = Mat4::from_cols_array(ibm[j * 16..j * 16 + 16].try_into().unwrap()) * w;
            let err = (m - Mat4::IDENTITY).to_cols_array().iter().fold(0.0_f32, |a, x| a.max(x.abs()));
            assert!(err < 1e-4, "joint {} inverse bind error {err}", joint.name);
            worlds.push(w);
        }
        // Without expressions: no targets.
        let (plain, r) =
            character_gltf(&t, &rest, &ExportOptions { expressions: false, ..ExportOptions::default() }).unwrap();
        assert_eq!(r.targets, 0);
        assert!(plain.document.meshes[0].primitives.iter().all(|p| p.targets.is_empty()));
        // A shape that does not match is refused.
        let mut short = rest.clone();
        short.positions.pop();
        assert!(character_gltf(&t, &short, &ExportOptions::default()).is_err());
    }
}
