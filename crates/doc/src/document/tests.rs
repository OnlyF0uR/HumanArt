use super::*;
use crate::registry::tests::real_registry;
use crate::text::{TextLimits, invert};
use humanart_core::hash::hash128;
use humanart_core::rng::Rng;

/// The version 1 fixture: the canonical text of [`sample`].
const FIXTURE_V1: &str = include_str!("../../tests/fixtures/document-v1.anthropos");

fn human() -> Author {
    Author { id: "human.artist".into(), kind: AuthorKind::Human, name: "Artist".into() }
}

fn agent() -> Author {
    Author { id: "agent.assistant".into(), kind: AuthorKind::Agent, name: "Assistant".into() }
}

fn oid(n: u128) -> Id128 {
    Id128::from_u128(0x0123_4567_89ab_cdef_0000_0000_0000_0000 + n)
}

/// A document that uses every section and every kind of value.
pub(crate) fn sample(registry: &Registry) -> Document {
    let mut doc = Document::new(oid(1), "Ada \"test\" Lovelace", "humanart-template", 3, 1_763_500_000_000, human());
    doc.meta.modified_ms = 1_763_500_123_456;
    doc.add_author(agent());
    let h = "human.artist";
    let a = "agent.assistant";
    doc.set_param(registry, "face.nose.bridge_height", ParamValue::Scalar(0.35), h).unwrap();
    doc.set_param(registry, "expression.au12_lip_corner_puller.l", ParamValue::Scalar(0.6), a).unwrap();
    doc.set_param(registry, "eyes.iris.color.l", ParamValue::Color([0.2, 0.1, 0.05]), h).unwrap();
    doc.set_param(registry, "eyes.iris.heterochromia", ParamValue::Enum("central".into()), a).unwrap();
    doc.set_param(registry, "skin.pores.density_map", ParamValue::AssetRef(hash128(b"pore map")), h).unwrap();
    doc.pose.insert("neck-01".into(), [0.0, 0.0, 0.0, 1.0]);
    doc.sculpt_layers.push(SculptLayer {
        id: oid(10),
        name: "Nose tweak".into(),
        region: Some("nasal-tip".into()),
        opacity: 0.75,
        payload: hash128(b"sculpt deltas"),
        locked: false,
        visible: true,
        author: h.into(),
        created_by_command: Some(oid(99)),
    });
    doc.sculpt_layers.push(SculptLayer {
        id: oid(11),
        name: "Whole body".into(),
        region: None,
        opacity: 1.0,
        payload: Hash128::ZERO,
        locked: true,
        visible: false,
        author: a.into(),
        created_by_command: None,
    });
    doc.skin_layers.push(SkinLayer {
        id: oid(20),
        name: "Freckles".into(),
        channels: vec![SkinChannel::AlbedoMultiply, SkinChannel::FreckleMask],
        blend: SkinBlend::Multiply,
        opacity: 0.5,
        region: Some("cheek-L".into()),
        source: SkinSource::Procedural {
            generator: "freckles".into(),
            params: BTreeMap::from([("density".into(), 0.4), ("size".into(), 0.002)]),
            seed: 42,
        },
        locked: false,
        visible: true,
        author: a.into(),
    });
    doc.skin_layers.push(SkinLayer {
        id: oid(21),
        name: "Scar".into(),
        channels: vec![SkinChannel::NormalDetail],
        blend: SkinBlend::Mix,
        opacity: 1.0,
        region: None,
        source: SkinSource::Painted { texture: hash128(b"scar texture") },
        locked: true,
        visible: true,
        author: h.into(),
    });
    doc.skin_layers.push(SkinLayer {
        id: oid(22),
        name: "Captured".into(),
        channels: vec![SkinChannel::AlbedoTint, SkinChannel::Roughness],
        blend: SkinBlend::Overlay,
        opacity: 0.25,
        region: None,
        source: SkinSource::Captured { set: hash128(b"capture set") },
        locked: false,
        visible: false,
        author: h.into(),
    });
    doc.grooms.push(Groom {
        id: oid(30),
        name: "Short crop".into(),
        kind: GroomKind::Scalp,
        guides: hash128(b"guides"),
        params: BTreeMap::from([("curl_radius".into(), 0.01)]),
        maps: BTreeMap::from([("density".into(), hash128(b"density map"))]),
        seed: 7,
        locked: false,
        visible: true,
        author: h.into(),
    });
    doc.expression_presets.push(ExpressionPreset {
        id: oid(40),
        name: "Smile".into(),
        weights: BTreeMap::from([
            ("expression.au06_cheek_raiser.l".into(), 0.5),
            ("expression.au12_lip_corner_puller.l".into(), 1.0),
        ]),
    });
    doc.pose_presets.push(PosePreset {
        id: oid(41),
        name: "Head tilt".into(),
        joints: BTreeMap::from([("neck-01".into(), [0.0, 0.0, 0.258_819_045_102_520_74, 0.965_925_826_289_068_3])]),
    });
    doc.locks.push(Lock { id: oid(50), target: LockTarget::Region("nasal-tip".into()), owner: h.into() });
    doc.locks.push(Lock { id: oid(51), target: LockTarget::Parameter("identity.age".into()), owner: a.into() });
    doc.pins.push(Pin { id: oid(52), landmark: "nose.tip".into(), position: [0.0, 1.62, 0.11], owner: h.into() });
    doc.snapshots.push(Snapshot {
        id: oid(60),
        name: "Before nose".into(),
        state: hash128(b"snapshot state"),
        created_ms: 1_763_500_100_000,
        author: h.into(),
    });
    doc.attachments.push(Attachment {
        id: oid(70),
        name: "Glasses".into(),
        mesh: hash128(b"glasses obj"),
        format: MeshFormat::Obj,
        materials: hash128(b"glasses mtl"),
        joint: "head".into(),
        binding: Binding::Rigid,
        translation: [0.0, 0.05, 0.09],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
        visible: true,
        author: a.into(),
    });
    doc.references.push(Reference {
        id: oid(80),
        name: "Front photo".into(),
        kind: ReferenceKind::Photo,
        path: "photos/front.png".into(),
        hash: hash128(b"photo bytes"),
    });
    doc.check(registry).unwrap();
    doc
}

#[test]
fn ac_wp07_1_document_round_trips_byte_identical() {
    let registry = real_registry();
    let doc = sample(&registry);
    let text = doc.write().unwrap();
    let back = Document::parse(text.as_bytes(), &registry, &TextLimits::DEFAULT).unwrap();
    assert_eq!(back, doc);
    assert_eq!(back.write().unwrap(), text);
}

#[test]
fn ac_wp07_1_v1_fixture_loads_and_rewrites_byte_identical() {
    // Every version 1..=FORMAT_VERSION has a fixture; today that is version 1 only.
    assert_eq!(FORMAT_VERSION, 1, "a new format version needs a fixture and a migration (DOC-23)");
    let registry = real_registry();
    let doc = Document::parse(FIXTURE_V1.as_bytes(), &registry, &TextLimits::DEFAULT).unwrap();
    assert_eq!(doc, sample(&registry));
    assert_eq!(doc.write().unwrap(), FIXTURE_V1);
}

#[test]
fn doc_02_parameters_are_sparse_and_checked() {
    let registry = real_registry();
    let mut doc = sample(&registry);
    let id = "face.nose.bridge_height";
    assert_eq!(doc.param(&registry, id).unwrap(), &ParamValue::Scalar(0.35));
    assert_eq!(doc.param(&registry, "identity.age").unwrap(), &ParamValue::Scalar(0.0));
    assert_eq!(doc.param_authors[id], "human.artist");
    // Back to the default: the entry and its author disappear.
    assert_eq!(
        doc.set_param(&registry, id, ParamValue::Scalar(0.0), "agent.assistant").unwrap(),
        Some(ParamValue::Scalar(0.35))
    );
    assert!(!doc.params.contains_key(id) && !doc.param_authors.contains_key(id));
    assert!(doc.set_param(&registry, id, ParamValue::Scalar(9.0), "human.artist").is_err());
    assert!(doc.set_param(&registry, id, ParamValue::Bool(true), "human.artist").is_err());
    assert!(doc.set_param(&registry, id, ParamValue::Scalar(0.1), "nobody").is_err());
    assert!(doc.set_param(&registry, "measure.face_width", ParamValue::Scalar(0.1), "human.artist").is_err());
    assert!(matches!(
        doc.set_param(&registry, "no.such", ParamValue::Scalar(0.1), "human.artist"),
        Err(DocError::UnknownParameter(_))
    ));
    // A default value in a hand-written file is accepted and dropped.
    let text = doc.write().unwrap().replace("[params]\n", "[params]\nidentity.age = 0.0\n");
    let back = Document::parse(text.as_bytes(), &registry, &TextLimits::DEFAULT).unwrap();
    assert!(!back.params.contains_key("identity.age"));
}

#[test]
fn doc_21_invalid_documents_are_refused() {
    let registry = real_registry();
    let text = sample(&registry).write().unwrap();
    let cases: [(&str, &str); 17] = [
        ("[meta]\n", "[meta]\nextra = 1\n"),
        ("humanart-character 1", "humanart-character 2"),
        ("name = \"Ada", "nome = \"Ada"),
        ("[snapshot ", "[unknown "),
        ("[sculpt_layer 0123456789abcdef000000000000000a]", "[sculpt_layer nose]"),
        ("blend = \"additive\"", "blend = \"subtract\""),
        ("opacity = 0.75", "opacity = 1.5"),
        ("author = \"human.artist\"\nblend = \"additive\"", "author = \"nobody\"\nblend = \"additive\""),
        ("face.nose.bridge_height = 0.35", "face.nose.bridge_height = 7.0"),
        ("face.nose.bridge_height = 0.35", "no.such.param = 0.35"),
        ("[params]\n", "[params]\nmeasure.face_width = 0.1\n"),
        ("neck-01 = [0.0,0.0,0.0,1.0]", "neck-01 = [0.0,0.0,0.0,2.0]"),
        ("path = \"photos/front.png\"", "path = \"../secret.png\""),
        ("\"kind\":\"painted\"", "\"kind\":\"pianted\""),
        ("\"density\":\"", "\"bogus\":\""),
        ("[settings]", "[settings x]"),
        ("kind = \"human\"", "kind = \"robot\""),
    ];
    for (from, to) in cases {
        if from == to {
            continue;
        }
        assert!(text.contains(from), "{from}");
        let bad = text.replacen(from, to, 1);
        assert!(Document::parse(bad.as_bytes(), &registry, &TextLimits::DEFAULT).is_err(), "{to}");
    }
    assert!(Document::parse(b"humanart-character 1\n", &registry, &TextLimits::DEFAULT).is_err(), "no meta");
}

#[test]
fn doc_11_and_22_snapshots_and_blob_references() {
    let registry = real_registry();
    let doc = sample(&registry);
    let refs = doc.blob_refs();
    for b in [
        &b"pore map"[..],
        b"sculpt deltas",
        b"scar texture",
        b"capture set",
        b"guides",
        b"density map",
        b"snapshot state",
        b"glasses obj",
        b"glasses mtl",
    ] {
        assert!(refs.contains(&hash128(b)), "{}", String::from_utf8_lossy(b));
    }
    assert_eq!(refs.len(), 9);
    assert!(!refs.contains(&hash128(b"photo bytes")), "references are not blobs");
    let snap = doc.snapshot_text().unwrap();
    assert!(!snap.contains("[snapshot"));
    let restored = Document::parse(snap.as_bytes(), &registry, &TextLimits::DEFAULT).unwrap();
    assert!(restored.snapshots.is_empty());
    assert_eq!(restored.sculpt_layers, doc.sculpt_layers);
}

/// One random change of the kind commands make.
fn mutate(doc: &mut Document, registry: &Registry, rng: &mut Rng) {
    let author = if rng.chance(0.5) { "human.artist" } else { "agent.assistant" };
    match rng.index(7) {
        0 | 1 => {
            let params: Vec<&crate::registry::Param> = registry
                .params()
                .iter()
                .filter(|p| !p.read_only && p.kind == crate::registry::ParamKind::Scalar)
                .collect();
            let p = params[rng.index(params.len())];
            let r = p.range.unwrap();
            let v = if rng.chance(0.2) { p.default.clone() } else { ParamValue::Scalar(rng.range_f64(r.min, r.max)) };
            doc.set_param(registry, &p.id, v, author).unwrap();
        }
        2 => {
            let n = doc.sculpt_layers.len();
            doc.sculpt_layers.insert(
                rng.index(n + 1),
                SculptLayer {
                    id: Id128::from_u128(u128::from(rng.next_u64()) | 1),
                    name: format!("Layer {}", rng.index(100)),
                    region: None,
                    opacity: rng.f64(),
                    payload: Hash128::from_parts(rng.next_u64(), 1),
                    locked: rng.chance(0.5),
                    visible: true,
                    author: author.into(),
                    created_by_command: None,
                },
            );
        }
        3 if !doc.sculpt_layers.is_empty() => {
            let i = rng.index(doc.sculpt_layers.len());
            doc.sculpt_layers.remove(i);
        }
        4 if doc.skin_layers.len() > 1 => {
            let i = rng.index(doc.skin_layers.len());
            let layer = doc.skin_layers.remove(i);
            let j = rng.index(doc.skin_layers.len() + 1);
            doc.skin_layers.insert(j, layer);
        }
        5 if !doc.skin_layers.is_empty() => {
            let i = rng.index(doc.skin_layers.len());
            doc.skin_layers[i].opacity = rng.f64();
            doc.skin_layers[i].locked = !doc.skin_layers[i].locked;
        }
        _ => {
            doc.meta.modified_ms += 1 + rng.index(1000) as u64;
            doc.settings.protect_human_edits = rng.chance(0.5);
        }
    }
}

#[test]
fn ac_wp07_1_every_change_is_an_exact_value_diff() {
    let registry = real_registry();
    let base = sample(&registry);
    let mut rng = Rng::new(0xd1ff_0007);
    for case in 0..200 {
        let mut a = base.clone();
        for _ in 0..rng.index(4) {
            mutate(&mut a, &registry, &mut rng);
        }
        let mut b = a.clone();
        for _ in 0..=rng.index(6) {
            mutate(&mut b, &registry, &mut rng);
        }
        let edits = a.diff(&b).unwrap();
        let mut x = a.clone();
        x.apply_edits(&edits, &registry).unwrap_or_else(|e| panic!("case {case}: {e}"));
        assert_eq!(x, b, "case {case}");
        x.apply_edits(&invert(&edits), &registry).unwrap();
        assert_eq!(x, a, "case {case}");
    }
}

#[test]
fn apply_edits_refuses_conflicts_and_non_canonical_results() {
    let registry = real_registry();
    let a = sample(&registry);
    let mut b = a.clone();
    b.set_param(&registry, "identity.age", ParamValue::Scalar(1.0), "human.artist").unwrap();
    let edits = a.diff(&b).unwrap();
    // Someone else set the same value meanwhile: the edit's `before` (absent) no longer matches.
    let mut c = a.clone();
    c.set_param(&registry, "identity.age", ParamValue::Scalar(-1.0), "agent.assistant").unwrap();
    let before = c.clone();
    assert!(matches!(c.apply_edits(&edits, &registry), Err(DocError::Conflict { .. })));
    assert_eq!(c, before);
    // Writing a default value explicitly would not survive a round trip: refused.
    let to_default = vec![Edit::Set {
        section: crate::text::SectionRef::new("params", None),
        key: "identity.mass".into(),
        before: None,
        after: Some(humanart_io::json::Value::from(humanart_io::json::Number::Float(0.0))),
    }];
    let mut d = a.clone();
    assert!(d.apply_edits(&to_default, &registry).is_err());
    assert_eq!(d, a);
}

#[test]
fn rob_01_document_parser_survives_mutation_fuzz() {
    let registry = real_registry();
    let corpus = vec![sample(&registry).write().unwrap().into_bytes()];
    humanart_core::fuzz::run("document", &corpus, 3000, 0xd0c, std::time::Duration::from_secs(120), move |bytes| {
        if let Ok(doc) = Document::parse(bytes, &registry, &TextLimits::FUZZ) {
            let text = doc.write().unwrap();
            let again = Document::parse(text.as_bytes(), &registry, &TextLimits::FUZZ).unwrap();
            assert_eq!(again, doc);
        }
    });
}
