//! Tests of the glTF reader, writer and export builder: the fixtures written independently by
//! `tests/fixtures/gltf/make_fixtures.py` (decoded values against its `.expect.json` files), writer
//! round trips, the export round trip of `[EXP-P2]`, rejection of broken documents and mutation fuzz
//! `[ROB-01]`.

use super::*;
use crate::json::{self, Value};
use humanart_core::fuzz;
use humanart_core::rng::Rng;
use std::time::Duration;

const TRIANGLE: &[u8] = include_bytes!("../../tests/fixtures/gltf/triangle.gltf");
const TRIANGLE_EXPECT: &str = include_str!("../../tests/fixtures/gltf/triangle.expect.json");
const SPARSE: &[u8] = include_bytes!("../../tests/fixtures/gltf/sparse_morph.gltf");
const SPARSE_BIN: &[u8] = include_bytes!("../../tests/fixtures/gltf/sparse morph.bin");
const SPARSE_EXPECT: &str = include_str!("../../tests/fixtures/gltf/sparse_morph.expect.json");
const SKINNED: &[u8] = include_bytes!("../../tests/fixtures/gltf/skinned.glb");
const SKINNED_EXPECT: &str = include_str!("../../tests/fixtures/gltf/skinned.expect.json");
const QUANTIZED: &[u8] = include_bytes!("../../tests/fixtures/gltf/quantized.gltf");
const QUANTIZED_EXPECT: &str = include_str!("../../tests/fixtures/gltf/quantized.expect.json");
const PNG_SMALL: &[u8] = include_bytes!("../../tests/fixtures/png/rgba8_i_1x1.png");
const PNG_RGB: &[u8] = include_bytes!("../../tests/fixtures/png/c2_d8.png");

/// The resolver of the fixtures: only `sparse morph.bin` exists.
fn resolve(path: &str) -> Option<Vec<u8>> {
    (path == "sparse morph.bin").then(|| SPARSE_BIN.to_vec())
}

fn load(data: &[u8]) -> Gltf {
    read(data, &Limits::FUZZ, resolve).unwrap()
}

/// Checks every accessor named in an `.expect.json` against the values the fixture script computed.
fn check_expected(gltf: &Gltf, expect: &str) {
    let expect = json::parse_str(expect).unwrap();
    let members = expect.as_object().unwrap();
    assert!(!members.is_empty());
    for (key, values) in members {
        let index: usize = key.parse().unwrap();
        let got = gltf.read_f32(index).unwrap();
        let want: Vec<f64> = values.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(got.len(), want.len(), "accessor {index}");
        for (g, w) in got.iter().zip(&want) {
            assert!((f64::from(*g) - w).abs() <= 1e-6, "accessor {index}: {g} != {w}");
        }
    }
}

#[test]
fn fixtures_decode_to_the_values_of_the_independent_script_io_10() {
    check_expected(&load(TRIANGLE), TRIANGLE_EXPECT);
    check_expected(&load(SPARSE), SPARSE_EXPECT);
    check_expected(&load(SKINNED), SKINNED_EXPECT);
    check_expected(&load(QUANTIZED), QUANTIZED_EXPECT);
}

#[test]
fn triangle_materials_textures_and_extras_are_typed() {
    let g = load(TRIANGLE);
    let d = &g.document;
    assert_eq!(d.asset.generator.as_deref(), Some("make_fixtures.py"));
    assert_eq!(d.scene, Some(0));
    assert_eq!(g.read_indices(&d.meshes[0].primitives[0]).unwrap(), [0, 1, 2]);
    let m = &d.materials[0];
    assert_eq!(m.pbr.base_color_factor, [0.8, 0.6, 0.5, 1.0]);
    assert_eq!((m.pbr.metallic_factor, m.pbr.roughness_factor), (0.0, 0.5));
    assert!(m.double_sided);
    assert_eq!(m.emissive_strength, Some(2.0));
    assert_eq!(m.ior, Some(1.4));
    assert_eq!(m.specular.as_ref().map(|s| s.factor), Some(0.5));
    // Typed extensions are taken out of the kept ones; the extras stay.
    assert!(m.extra.extensions.is_empty());
    assert!(m.extra.extras.as_ref().unwrap().get("humanart").is_some());
    let base = m.pbr.base_color_texture.as_ref().unwrap();
    assert_eq!(base.transform.unwrap().offset, [0.5, 0.0]);
    assert_eq!(base.transform.unwrap().scale, [2.0, 2.0]);
    assert_eq!(d.samplers[0].min_filter, Some(9987));
    assert_eq!((d.samplers[0].wrap_s, d.samplers[0].wrap_t), (33071, 33648));
    assert_eq!(g.image_source(0, &Limits::FUZZ).unwrap(), ImageSource::Data(PNG_SMALL.to_vec()));
    assert_eq!(d.extra.extras.as_ref().unwrap()["humanart"]["regions"][1].as_u64(), Some(2));
    match d.nodes[0].transform {
        Transform::Trs { translation, rotation, scale } => {
            assert_eq!((translation, rotation, scale), ([0.0, 1.5, 0.0], [0.0, 0.0, 0.6, 0.8], [1.0; 3]));
        }
        Transform::Matrix(_) => panic!("TRS expected"),
    }
}

#[test]
fn sparse_accessors_morph_targets_and_external_buffers() {
    let g = load(SPARSE);
    let mesh = &g.document.meshes[0];
    assert_eq!(mesh.target_names(), ["up", "corner"]);
    assert_eq!(mesh.weights, [0.25, 0.75]);
    assert_eq!(mesh.primitives[0].targets.len(), 2);
    assert!(g.document.accessors[3].buffer_view.is_none());
    assert_eq!(g.read_vec::<3>(1).unwrap()[3], [0.0, 2.0, 0.0]);
    assert!(g.read_vec::<2>(1).is_err(), "component count mismatch");
    assert_eq!(g.read_u32(0).unwrap(), [0, 1, 2, 0, 2, 3]);
    assert!(g.read_u32(1).is_err(), "float accessor read as integers");
    // The external buffer is required; without it the file does not load.
    assert!(matches!(read_self_contained(SPARSE, &Limits::FUZZ), Err(Error::Invalid { .. })));
}

#[test]
fn glb_skin_hierarchy_camera_light_and_unknown_extensions() {
    let g = load(SKINNED);
    let d = &g.document;
    assert_eq!(d.asset.copyright.as_deref(), Some("HumanArt test fixture"));
    assert_eq!(d.skins[0].joints, [1, 2]);
    let ibm = g.inverse_bind_matrices(0).unwrap();
    assert_eq!(ibm[1][13], -1.0);
    assert_eq!(g.read_u32(3).unwrap()[..4], [0, 1, 0, 0]);
    assert!(matches!(d.nodes[2].transform, Transform::Matrix(m) if m[13] == 0.5));
    let rig = &d.nodes[4];
    assert_eq!((rig.camera, rig.light), (Some(0), Some(0)));
    assert_eq!(rig.extra.extensions.keys().collect::<Vec<_>>(), ["EXT_humanart_test"]);
    assert!(rig.extra.extras.as_ref().unwrap()["landmarks"]["nose_tip"].as_array().is_some());
    assert_eq!(d.lights[0].kind, LightKind::Spot { inner_cone_angle: 0.2, outer_cone_angle: 0.6 });
    assert_eq!(d.lights[0].range, Some(10.0));
    assert!(matches!(d.cameras[0].projection, Projection::Perspective { zfar: None, aspect_ratio: Some(1.5), .. }));
    assert_eq!(d.animations[0].samplers[1].interpolation, Interpolation::CubicSpline);
    assert_eq!(d.animations[0].channels[1].path, Path::Rotation);
    assert_eq!(g.image_source(0, &Limits::FUZZ).unwrap(), ImageSource::View(PNG_RGB));
    // The extension list written back keeps the unknown extension.
    let again = read_self_contained(&g.to_glb().unwrap(), &Limits::FUZZ).unwrap();
    assert!(again.document.extensions_used.iter().any(|e| e == "EXT_humanart_test"));
}

#[test]
fn fixtures_survive_glb_and_gltf_round_trips() {
    for data in [TRIANGLE, SPARSE, SKINNED, QUANTIZED] {
        let g = load(data);
        let glb = g.to_glb().unwrap();
        assert_eq!(g.to_glb().unwrap(), glb, "deterministic output");
        let from_glb = read(&glb, &Limits::FUZZ, resolve).unwrap();
        // `extensionsUsed` is rewritten sorted and complete, and a first buffer given as a data URI
        // comes back from the BIN chunk (no uri); everything else is equal.
        let mut sorted = g.clone();
        sorted.document.extensions_used.sort();
        if let Some(first) = sorted.document.buffers.first_mut()
            && first.uri.as_deref().is_some_and(|u| u.starts_with("data:"))
        {
            first.uri = None;
        }
        assert_eq!(from_glb, sorted);
        let mut embedded = g.clone();
        embedded.embed_buffers();
        for style in [json::Style::COMPACT, json::Style::PRETTY] {
            let text = embedded.to_gltf(style).unwrap();
            let back = read(&text, &Limits::FUZZ, resolve).unwrap();
            assert_eq!(back.buffers, g.buffers);
            assert_eq!(back.document.accessors, g.document.accessors);
            assert_eq!(back.document.materials, g.document.materials);
            assert_eq!(back.document.nodes, g.document.nodes);
        }
    }
    // A buffer that must go to a file cannot be written into a .gltf without a URI.
    let built = Builder::new("t").finish().unwrap();
    assert!(built.to_gltf(json::Style::COMPACT).is_ok(), "no buffers at all");
    let mut b = Builder::new("t");
    b.view(&[1, 2, 3, 4], None, None).unwrap();
    assert!(matches!(b.finish().unwrap().to_gltf(json::Style::COMPACT), Err(Error::Encode { .. })));
}

/// A mesh of `n` vertices for the export round trip.
struct TestMesh {
    positions: Vec<f32>,
    normals: Vec<f32>,
    uvs: Vec<f32>,
    joints: Vec<[u16; 4]>,
    weights: Vec<f32>,
    indices: Vec<u32>,
    targets: [Vec<f32>; 3],
}

fn test_mesh(n: usize, rng: &mut Rng) -> TestMesh {
    let mut v = |k: usize, lo: f32, hi: f32| (0..k).map(|_| rng.range_f32(lo, hi)).collect::<Vec<f32>>();
    let positions = v(n * 3, -1.0, 1.0);
    let normals = v(n * 3, -1.0, 1.0);
    let uvs = v(n * 2, 0.0, 1.0);
    let mut sparse = vec![0.0; n * 3];
    for i in (0..n).step_by(17) {
        sparse[i * 3 + 1] = 0.01 * i as f32;
    }
    let dense = v(n * 3, -0.1, 0.1);
    let joints = (0..n).map(|i| [(i % 300) as u16, 1, 2, 3]).collect();
    let weights = (0..n).flat_map(|_| [0.5, 0.25, 0.125, 0.125]).collect();
    let indices = (0..n as u32 - 2).flat_map(|i| [i, i + 1, i + 2]).collect();
    TestMesh { positions, normals, uvs, joints, weights, indices, targets: [sparse, dense, vec![0.0; n * 3]] }
}

/// Builds a skinned, morphing character with every typed material extension.
fn build(mesh: &TestMesh) -> Gltf {
    let mut b = Builder::new("HumanArt test");
    let pos = b.accessor_f32(&mesh.positions, ElementType::Vec3, Some(Target::ArrayBuffer)).unwrap();
    let nrm = b.accessor_f32(&mesh.normals, ElementType::Vec3, Some(Target::ArrayBuffer)).unwrap();
    let uv = b.accessor_f32(&mesh.uvs, ElementType::Vec2, Some(Target::ArrayBuffer)).unwrap();
    let jnt = b.accessor_joints(&mesh.joints).unwrap();
    let wgt = b.accessor_f32(&mesh.weights, ElementType::Vec4, Some(Target::ArrayBuffer)).unwrap();
    let idx = b.accessor_indices(&mesh.indices).unwrap();
    let targets: Vec<_> = mesh
        .targets
        .iter()
        .map(|t| vec![("POSITION".to_owned(), b.accessor_morph(t, ElementType::Vec3).unwrap())])
        .collect();
    let ibm: Vec<f32> = (0..300)
        .flat_map(|j| {
            Transform::Trs { translation: [0.0, -(j as f32), 0.0], rotation: [0.0, 0.0, 0.0, 1.0], scale: [1.0; 3] }
                .matrix()
        })
        .collect();
    let ibm = b.accessor_f32(&ibm, ElementType::Mat4, None).unwrap();
    let png = b.image(PNG_RGB, "image/png", Some("albedo")).unwrap();
    let ktx = b.image(&[0xAB, b'K', b'T', b'X'], "image/ktx2", None).unwrap();
    let d = &mut b.document;
    d.textures.push(Texture { source: Some(png), basisu_source: Some(ktx), ..Texture::default() });
    let info = TextureInfo {
        transform: Some(TextureTransform { rotation: 0.5, ..TextureTransform::default() }),
        ..TextureInfo::new(0)
    };
    d.materials.push(Material {
        name: Some("skin".into()),
        pbr: Pbr { base_color_texture: Some(info.clone()), metallic_factor: 0.0, ..Pbr::default() },
        normal_texture: Some(TextureInfo { scale: 0.7, ..TextureInfo::new(0) }),
        occlusion_texture: Some(TextureInfo { scale: 0.3, tex_coord: 1, ..TextureInfo::new(0) }),
        alpha_mode: AlphaMode::Mask,
        alpha_cutoff: 0.25,
        emissive_strength: Some(3.0),
        ior: Some(1.4),
        specular: Some(Specular { factor: 0.4, color_texture: Some(info), ..Specular::default() }),
        clearcoat: Some(Clearcoat { factor: 1.0, roughness_factor: 0.05, ..Clearcoat::default() }),
        sheen: Some(Sheen { color_factor: [0.1; 3], roughness_factor: 0.6, ..Sheen::default() }),
        transmission: Some(Transmission { factor: 0.1, texture: None }),
        volume: Some(Volume { thickness_factor: 0.01, attenuation_distance: Some(0.003), ..Volume::default() }),
        diffuse_transmission: Some(DiffuseTransmission {
            factor: 0.3,
            color_factor: [1.0, 0.3, 0.2],
            ..DiffuseTransmission::default()
        }),
        ..Material::default()
    });
    d.materials.push(Material { unlit: true, ..Material::default() });
    let mut m = Mesh {
        name: Some("body".into()),
        primitives: vec![Primitive {
            attributes: vec![
                ("POSITION".into(), pos),
                ("NORMAL".into(), nrm),
                ("TEXCOORD_0".into(), uv),
                ("JOINTS_0".into(), jnt),
                ("WEIGHTS_0".into(), wgt),
            ],
            indices: Some(idx),
            material: Some(0),
            targets,
            ..Primitive::default()
        }],
        weights: vec![0.0, 0.5, 1.0],
        extra: Extra::default(),
    };
    m.set_target_names(&["brow_raise", "jaw_open", "neutral"]);
    d.meshes.push(m);
    d.nodes.push(Node { name: Some("root".into()), children: (1..=300).collect(), ..Node::default() });
    d.nodes.extend((0..300).map(|j| Node { name: Some(format!("joint{j}")), ..Node::default() }));
    d.nodes.push(Node { name: Some("body".into()), mesh: Some(0), skin: Some(0), ..Node::default() });
    d.nodes[0].children.push(301);
    d.skins.push(Skin {
        inverse_bind_matrices: Some(ibm),
        skeleton: Some(0),
        joints: (1..=300).collect(),
        ..Skin::default()
    });
    d.scenes.push(Scene { nodes: vec![0], ..Scene::default() });
    d.scene = Some(0);
    d.lights.push(Light {
        name: None,
        color: [1.0; 3],
        intensity: 5.0,
        kind: LightKind::Directional,
        range: None,
        extra: Extra::default(),
    });
    d.nodes[0].light = Some(0);
    b.finish().unwrap()
}

#[test]
fn export_round_trip_reproduces_positions_and_morph_targets_exp_p2() {
    let mut rng = Rng::new(0x676C_5446);
    let mesh = test_mesh(1000, &mut rng);
    let gltf = build(&mesh);
    let glb = gltf.to_glb().unwrap();
    let back = read_self_contained(&glb, &Limits::DEFAULT).unwrap();
    assert_eq!(back, gltf);
    let p = &back.document.meshes[0].primitives[0];
    // [EXP-P2]: rest positions within 1e-5 (they are bit-identical) and identical morph targets.
    let positions = back.read_f32(p.attribute("POSITION").unwrap()).unwrap();
    assert_eq!(
        positions.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        mesh.positions.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    for (target, want) in p.targets.iter().zip(&mesh.targets) {
        assert_eq!(&back.read_f32(find(target, "POSITION").unwrap()).unwrap(), want);
    }
    assert_eq!(back.read_indices(p).unwrap(), mesh.indices);
    let joints = back.read_u32(p.attribute("JOINTS_0").unwrap()).unwrap();
    assert_eq!(
        joints.chunks(4).map(|j| j[0] as u16).collect::<Vec<_>>(),
        mesh.joints.iter().map(|j| j[0]).collect::<Vec<_>>()
    );
    assert_eq!(back.document.meshes[0].target_names(), ["brow_raise", "jaw_open", "neutral"]);
    // Storage: the sparse target is sparse, the dense one dense, the zero one has no data at all.
    let acc = |t: usize| &back.document.accessors[find(&p.targets[t], "POSITION").unwrap()];
    assert!(acc(0).buffer_view.is_none() && acc(0).sparse.as_ref().is_some_and(|s| s.count == 58));
    assert!(acc(1).buffer_view.is_some() && acc(1).sparse.is_none());
    assert!(acc(2).buffer_view.is_none() && acc(2).sparse.is_none());
    assert_eq!(acc(2).min, [0.0; 3]);
    // min / max are exact.
    let pos = &back.document.accessors[p.attribute("POSITION").unwrap()];
    for c in 0..3 {
        let column = mesh.positions.iter().skip(c).step_by(3).copied();
        assert_eq!(pos.min[c], f64::from(column.clone().fold(f32::INFINITY, f32::min)));
        assert_eq!(pos.max[c], f64::from(column.fold(f32::NEG_INFINITY, f32::max)));
    }
    // 16-bit indices (below 65535), u16 joints (some are 256 or more).
    assert_eq!(back.document.accessors[p.indices.unwrap()].component_type, ComponentType::U16);
    assert_eq!(back.document.accessors[p.attribute("JOINTS_0").unwrap()].component_type, ComponentType::U16);
    let used = &back.document.extensions_used;
    for name in [
        "KHR_lights_punctual",
        "KHR_materials_unlit",
        "KHR_materials_volume",
        "KHR_texture_basisu",
        "KHR_texture_transform",
    ] {
        assert!(used.iter().any(|u| u == name), "{name}");
    }
    // The embedded .gltf form reads back to the same data.
    let mut embedded = gltf.clone();
    embedded.embed_buffers();
    let text = embedded.to_gltf(json::Style::PRETTY).unwrap();
    assert_eq!(read_self_contained(&text, &Limits::DEFAULT).unwrap().buffers, gltf.buffers);
}

#[test]
fn builder_refuses_bad_data() {
    let mut b = Builder::new("t");
    assert!(b.accessor_f32(&[], ElementType::Vec3, None).is_err());
    assert!(b.accessor_f32(&[1.0, 2.0], ElementType::Vec3, None).is_err());
    assert!(b.accessor_f32(&[f32::NAN, 0.0, 0.0], ElementType::Vec3, None).is_err());
    assert!(b.accessor_f32(&[0.0; 4], ElementType::Mat2, None).is_err());
    assert!(b.accessor_morph(&[0.0; 3], ElementType::Scalar).is_err());
    assert!(b.accessor_indices(&[]).is_err());
    assert!(b.accessor_joints(&[]).is_err());
    let big = b.accessor_indices(&[0, 70_000]).unwrap();
    assert_eq!(b.document.accessors[big].component_type, ComponentType::U32);
    // A node that is its own child fails validation in finish.
    b.document.nodes.push(Node { children: vec![0], ..Node::default() });
    assert!(b.finish().is_err());
}

/// Applies `edit` to the JSON of the triangle fixture and reads the result.
fn read_edited(edit: impl FnOnce(&mut Value)) -> Result<Gltf> {
    let mut v = json::parse(TRIANGLE).unwrap();
    edit(&mut v);
    read(&json::to_vec(&v).unwrap(), &Limits::FUZZ, resolve)
}

fn set(v: &mut Value, path: &[&str], value: Value) {
    let mut at = v;
    for (i, key) in path.iter().enumerate() {
        let next = match key.parse::<usize>() {
            Ok(n) => at.at_mut(n),
            Err(_) if i + 1 == path.len() => {
                at.as_object_mut().unwrap().insert(*key, value);
                return;
            }
            Err(_) => at.get_mut(key),
        };
        at = next.unwrap();
    }
    *at = value;
}

/// One edit of a test document.
type Edit = Box<dyn FnOnce(&mut Value)>;

/// Sets the member at `path` to the JSON `text`.
fn set_json(v: &mut Value, path: &[&str], text: &str) {
    set(v, path, json::parse_str(text).unwrap());
}

#[test]
fn broken_documents_are_refused_rob_01() {
    let n = |x: i64| Value::from(json::Number::Int(x));
    let f = |x: f64| Value::from(json::Number::Float(x));
    let unsupported: [(&str, Edit); 3] = [
        ("major version", Box::new(|v| set(v, &["asset", "version"], "3.0".into()))),
        ("minVersion", Box::new(|v| set(v, &["asset", "minVersion"], "2.1".into()))),
        (
            "required extension",
            Box::new(|v| {
                set(v, &["extensionsUsed"], Value::from(vec!["KHR_draco_mesh_compression"]));
                set(v, &["extensionsRequired"], Value::from(vec!["KHR_draco_mesh_compression"]));
            }),
        ),
    ];
    for (what, edit) in unsupported {
        assert!(matches!(read_edited(edit), Err(Error::Unsupported { .. })), "{what}");
    }
    let cases: Vec<(&str, Edit)> = vec![
        ("no asset", Box::new(|v| drop(v.as_object_mut().unwrap().remove("asset")))),
        ("required not used", Box::new(|v| set(v, &["extensionsRequired"], Value::from(vec!["KHR_materials_sheen"])))),
        ("accessor view index", Box::new(move |v| set(v, &["accessors", "0", "bufferView"], n(99)))),
        ("accessor count", Box::new(move |v| set(v, &["accessors", "1", "count"], n(4)))),
        ("accessor count zero", Box::new(move |v| set(v, &["accessors", "1", "count"], n(0)))),
        ("misaligned", Box::new(move |v| set(v, &["accessors", "1", "byteOffset"], n(2)))),
        ("negative index", Box::new(move |v| set(v, &["meshes", "0", "primitives", "0", "indices"], n(-1)))),
        ("fractional index", Box::new(move |v| set(v, &["meshes", "0", "primitives", "0", "indices"], f(0.5)))),
        ("view past buffer", Box::new(move |v| set(v, &["bufferViews", "1", "byteLength"], n(4000)))),
        ("stride", Box::new(move |v| set(v, &["bufferViews", "1", "byteStride"], n(3)))),
        ("indices are floats", Box::new(move |v| set(v, &["meshes", "0", "primitives", "0", "indices"], n(1)))),
        (
            "position type",
            Box::new(move |v| set(v, &["meshes", "0", "primitives", "0", "attributes", "POSITION"], n(2))),
        ),
        ("no primitives", Box::new(|v| set(v, &["meshes", "0", "primitives"], Value::Array(Vec::new())))),
        ("mode", Box::new(move |v| set(v, &["meshes", "0", "primitives", "0", "mode"], n(7)))),
        ("matrix and trs", Box::new(|v| set(v, &["nodes", "0", "matrix"], Value::from([1.0_f64; 16].to_vec())))),
        ("rotation", Box::new(|v| set(v, &["nodes", "0", "rotation"], Value::from(vec![0.0, 0.0, 0.0, 2.0])))),
        ("self child", Box::new(|v| set(v, &["nodes", "0", "children"], Value::from(vec![0_u64])))),
        ("scene root", Box::new(move |v| set(v, &["scenes", "0", "nodes", "0"], n(5)))),
        ("default scene", Box::new(move |v| set(v, &["scene"], n(1)))),
        ("material", Box::new(move |v| set(v, &["meshes", "0", "primitives", "0", "material"], n(3)))),
        ("texture", Box::new(move |v| set(v, &["textures", "0", "source"], n(3)))),
        ("filter", Box::new(move |v| set(v, &["samplers", "0", "magFilter"], n(9987)))),
        ("alpha mode", Box::new(|v| set(v, &["materials", "0", "alphaMode"], "GLASS".into()))),
        ("factor", Box::new(|v| set(v, &["materials", "0", "emissiveFactor"], Value::from(vec![1.0, 1.0])))),
        ("huge float", Box::new(move |v| set(v, &["materials", "0", "alphaCutoff"], f(1e300)))),
        ("buffer short", Box::new(move |v| set(v, &["buffers", "0", "byteLength"], n(4000)))),
        ("buffer zero", Box::new(move |v| set(v, &["buffers", "0", "byteLength"], n(0)))),
        ("data uri", Box::new(|v| set(v, &["buffers", "0", "uri"], "data:application/octet-stream,AAAA".into()))),
        ("bad base64", Box::new(|v| set(v, &["buffers", "0", "uri"], "data:;base64,A*AA".into()))),
        ("missing file", Box::new(|v| set(v, &["buffers", "0", "uri"], "missing.bin".into()))),
        (
            "no uri outside GLB",
            Box::new(|v| drop(v.get_mut("buffers").unwrap().at_mut(0).unwrap().as_object_mut().unwrap().remove("uri"))),
        ),
        ("bounds length", Box::new(|v| set(v, &["accessors", "1", "min"], Value::from(vec![0.0])))),
        ("image both", Box::new(move |v| set(v, &["images", "0", "bufferView"], n(0)))),
        (
            "camera",
            Box::new(|v| set_json(v, &["cameras"], r#"[{"type":"perspective","perspective":{"yfov":1,"znear":0}}]"#)),
        ),
        (
            "light",
            Box::new(|v| {
                set_json(
                    v,
                    &["extensions"],
                    r#"{"KHR_lights_punctual":{"lights":[{"type":"spot","spot":{"innerConeAngle":1,"outerConeAngle":0.5}}]}}"#,
                );
            }),
        ),
        (
            "node light",
            Box::new(|v| set_json(v, &["nodes", "0", "extensions"], r#"{"KHR_lights_punctual":{"light":0}}"#)),
        ),
        ("root kind", Box::new(|v| *v = Value::Array(Vec::new()))),
    ];
    for (what, edit) in cases {
        assert!(matches!(read_edited(edit), Err(Error::Invalid { .. })), "{what}");
    }
    assert!(read_edited(|_| {}).is_ok());
}

#[test]
fn hierarchy_cycles_shared_children_and_depth_are_refused() {
    let doc = |nodes: Vec<Node>| Document { nodes, ..Document::default() };
    let child = |c: Vec<usize>| Node { children: c, ..Node::default() };
    assert!(validate(&doc(vec![child(vec![1]), child(vec![0])]), &[], &Limits::FUZZ).is_err(), "cycle");
    assert!(validate(&doc(vec![child(vec![1]), child(vec![2]), child(vec![1])]), &[], &Limits::FUZZ).is_err(), "cycle");
    assert!(
        validate(&doc(vec![child(vec![2]), child(vec![2]), child(vec![])]), &[], &Limits::FUZZ).is_err(),
        "two parents"
    );
    let chain = |n: usize| doc((0..n).map(|i| child(if i + 1 < n { vec![i + 1] } else { vec![] })).collect());
    let limit = Limits::FUZZ.max_depth as usize;
    assert!(validate(&chain(limit), &[], &Limits::FUZZ).is_ok());
    assert!(matches!(validate(&chain(limit + 1), &[], &Limits::FUZZ), Err(Error::Limit { .. })));
    // A long chain listed leaf first is checked in linear time (each node's depth is found once).
    let bottom_up = doc((0..200_000).map(|i| child(if i > 0 { vec![i - 1] } else { vec![] })).collect());
    let unlimited = Limits { max_depth: u32::MAX, ..Limits::DEFAULT };
    assert!(validate(&bottom_up, &[], &unlimited).is_ok());
    assert!(matches!(validate(&bottom_up, &[], &Limits::DEFAULT), Err(Error::Limit { .. })));
}

#[test]
fn sparse_and_accessor_ranges_are_checked() {
    let g = load(SPARSE);
    let with = |edit: &dyn Fn(&mut Gltf)| {
        let mut g = g.clone();
        edit(&mut g);
        validate(&g.document, &g.buffers, &Limits::FUZZ)
    };
    assert!(with(&|_| {}).is_ok());
    // Indices 1, 3 -> 3, 3: not strictly increasing.
    let view = g.document.buffer_views[g.document.accessors[1].sparse.as_ref().unwrap().indices_view].byte_offset;
    assert!(with(&|g| g.buffers[0][view..view + 2].copy_from_slice(&3_u16.to_le_bytes())).is_err());
    // Index 4 of 4 elements.
    assert!(with(&|g| g.buffers[0][view + 2..view + 4].copy_from_slice(&4_u16.to_le_bytes())).is_err());
    assert!(with(&|g| g.document.accessors[1].sparse.as_mut().unwrap().count = 5).is_err());
    assert!(with(&|g| g.document.accessors[1].sparse.as_mut().unwrap().indices_type = ComponentType::I16).is_err());
    assert!(with(&|g| g.document.accessors[1].sparse.as_mut().unwrap().values_offset = 4).is_err());
    assert!(with(&|g| g.document.accessors[3].byte_offset = 4).is_err(), "offset without a view");
    assert!(with(&|g| g.document.accessors[0].normalized = true).is_err(), "normalized indices");
    assert!(with(&|g| g.document.accessors[4].normalized = true).is_err(), "normalized float");
    assert!(with(&|g| g.document.meshes[0].weights.push(1.0)).is_err(), "weights length");
    assert!(with(&|g| g.document.animations[0].samplers[0].output = 4).is_err(), "output count");
    assert!(with(&|g| g.buffers[0].truncate(8)).is_err(), "short buffer data");
    // Reading an unvalidated document with broken ranges fails instead of panicking.
    let mut broken = g.clone();
    broken.document.accessors[1].count = usize::MAX / 2;
    broken.document.accessors[2].byte_offset = usize::MAX - 3;
    broken.document.buffer_views[0].byte_length = usize::MAX;
    for i in 0..=broken.document.accessors.len() {
        let _ = broken.read_f32(i);
        let _ = broken.read_u32(i);
    }
    assert!(broken.read_f32(2).is_err());
    assert!(broken.read_f32(0).is_err());
}

#[test]
fn limits_are_checked_before_decoding() {
    let tight = Limits { max_bytes: 100, ..Limits::FUZZ };
    assert!(matches!(read(TRIANGLE, &tight, resolve), Err(Error::Limit { .. })));
    let few = Limits { max_elements: 2, ..Limits::FUZZ };
    assert!(matches!(read(TRIANGLE, &few, resolve), Err(Error::Limit { .. })));
    let shallow = Limits { max_depth: 1, ..Limits::FUZZ };
    assert!(matches!(read(SKINNED, &shallow, resolve), Err(Error::Limit { .. })));
    // A huge count on an accessor without data is a limit, not an allocation.
    let r = read_edited(|v| {
        let accessors = v.get_mut("accessors").unwrap().as_array_mut().unwrap();
        accessors.push(json::parse_str(r#"{"componentType":5126,"count":4294967296,"type":"MAT4"}"#).unwrap());
    });
    assert!(matches!(r, Err(Error::Limit { .. })));
}

/// Replaces about one number in `rate` of `v` with a value that sits on an edge validation must
/// handle (zero, negative, fractional, huge, just past a count).
fn mutate_numbers(v: &mut Value, rng: &mut Rng, rate: usize) {
    match v {
        Value::Number(_) if rng.index(rate) == 0 => {
            let edges: [f64; 13] =
                [0.0, 1.0, -1.0, 0.5, 2.0, 3.0, 4.0, 255.0, 65535.0, 4_294_967_296.0, 1e300, 5126.0, 5121.0];
            let pick = edges[rng.index(edges.len())];
            *v = if pick.fract() == 0.0 && pick.abs() < 1e18 {
                Value::from(json::Number::Int(pick as i64))
            } else {
                Value::from(json::Number::Float(pick))
            };
        }
        Value::Array(items) => items.iter_mut().for_each(|i| mutate_numbers(i, rng, rate)),
        Value::Object(o) => o.iter_mut().for_each(|(_, i)| mutate_numbers(i, rng, rate)),
        _ => {}
    }
}

/// What the fuzz closures do with an input: read it and, when it is accepted, decode every accessor
/// and image and write it again.
fn exercise(bytes: &[u8]) {
    if let Ok(g) = read(bytes, &Limits::FUZZ, resolve) {
        for i in 0..g.document.accessors.len() {
            let _ = g.read_f32(i);
            let _ = g.read_u32(i);
        }
        for p in g.document.meshes.iter().flat_map(|m| &m.primitives) {
            let _ = g.read_indices(p);
        }
        for i in 0..g.document.skins.len() {
            let _ = g.inverse_bind_matrices(i);
        }
        for i in 0..g.document.images.len() {
            let _ = g.image_source(i, &Limits::FUZZ);
        }
        // Whatever is accepted is valid enough to be written and read again.
        let again = g.to_glb().unwrap();
        assert!(read(&again, &Limits::FUZZ, resolve).is_ok());
    }
}

fn corpus() -> Vec<Vec<u8>> {
    let mut rng = Rng::new(7);
    let built = build(&test_mesh(40, &mut rng)).to_glb().unwrap();
    vec![TRIANGLE.to_vec(), SPARSE.to_vec(), SKINNED.to_vec(), QUANTIZED.to_vec(), built]
}

#[test]
fn mutation_fuzz_bytes_rob_01() {
    fuzz::run("gltf", &corpus(), 400, 0x676C_5446, Duration::from_secs(120), exercise);
    fuzz::run("gltf-base64", &[b"Zm9vYmFy".to_vec(), b"AAEC/w==".to_vec()], 400, 64, Duration::from_secs(30), |b| {
        let _ = base64::decode(b, 1 << 10);
    });
}

#[test]
fn mutation_fuzz_json_numbers_rob_01() {
    // Byte mutations mostly break the JSON syntax; these keep the syntax and change the numbers, so
    // validation itself is what gets exercised.
    let mut rng = Rng::new(0x4A53_4F4E);
    let mut cases = Vec::new();
    for source in corpus() {
        let (text, bin) = if glb::is_glb(&source) {
            let parts = glb::split(&source).unwrap();
            (parts.json.to_vec(), parts.bin.map(<[u8]>::to_vec))
        } else {
            (source.clone(), None)
        };
        let value = json::parse(&text).unwrap();
        for _ in 0..150 {
            let mut v = value.clone();
            mutate_numbers(&mut v, &mut rng, 12);
            let json_text = json::to_vec(&v).unwrap();
            cases.push(match &bin {
                Some(bin) => glb::join(&json_text, bin).unwrap(),
                None => json_text,
            });
        }
    }
    let accepted = cases.iter().filter(|c| read(c, &Limits::FUZZ, resolve).is_ok()).count();
    // Most mutants break a rule; some (a changed name length, an equal value) stay valid.
    assert!(accepted < cases.len() / 2, "{accepted} of {} accepted", cases.len());
    fuzz::run("gltf-json-numbers", &cases, 2, 0x4E55_4D53, Duration::from_secs(120), exercise);
}

#[test]
fn to_glb_moves_a_data_uri_first_buffer_into_the_bin_chunk() {
    // glTF 2.0 GLB: an embedded first buffer belongs in the BIN chunk, not in base64 in the JSON.
    let mut rng = Rng::new(0x0617_B1B0);
    let gltf = build(&test_mesh(30, &mut rng));
    let reference = gltf.to_glb().unwrap();
    let mut embedded = gltf.clone();
    embedded.embed_buffers();
    assert!(embedded.document.buffers[0].uri.as_deref().is_some_and(|u| u.starts_with("data:")));
    let glb = embedded.to_glb().unwrap();
    assert_eq!(glb, reference, "the same file as from the buffer without a uri");
    let parts = glb::split(&glb).unwrap();
    assert!(!String::from_utf8_lossy(parts.json).contains("data:"), "no data URI left in the JSON");
    let bin = parts.bin.unwrap();
    assert_eq!(&bin[..gltf.buffers[0].len()], &gltf.buffers[0][..]);
    let again = read_self_contained(&glb, &Limits::DEFAULT).unwrap();
    assert!(again.document.buffers[0].uri.is_none());
    assert_eq!(again.buffers, gltf.buffers);
    // The caller's document is left as it was.
    assert!(embedded.document.buffers[0].uri.is_some());

    // A second buffer with a data URI stays a data URI; an external first buffer stays external.
    let mut two = gltf.clone();
    two.document.buffers.push(Buffer { uri: None, byte_length: 4, ..two.document.buffers[0].clone() });
    two.buffers.push(vec![1, 2, 3, 4]);
    assert!(two.to_glb().is_err(), "a second buffer without a uri cannot be written");
    two.embed_buffers();
    let glb = two.to_glb().unwrap();
    let again = read_self_contained(&glb, &Limits::DEFAULT).unwrap();
    assert!(again.document.buffers[0].uri.is_none());
    assert!(again.document.buffers[1].uri.as_deref().is_some_and(|u| u.starts_with("data:")));
    assert_eq!(again.buffers, two.buffers);
    let mut external = gltf.clone();
    external.document.buffers[0].uri = Some("mesh.bin".to_owned());
    let glb = external.to_glb().unwrap();
    assert!(glb::split(&glb).unwrap().bin.is_none_or(<[u8]>::is_empty));
    let again = read(&glb, &Limits::DEFAULT, |path| (path == "mesh.bin").then(|| gltf.buffers[0].clone())).unwrap();
    assert_eq!(again.document.buffers[0].uri.as_deref(), Some("mesh.bin"));
}
