use humanart_core::hash::hash128;
use humanart_core::platform::MemoryFs;
use humanart_core::rng::Rng;
use humanart_io::json::{self, JsonLimits, Number, Object, Value};

use super::codec::{CommandLimits, command_from_json, command_to_json, edits_from_json, edits_to_json};
use super::*;
use crate::blob::BlobStore;
use crate::document::tests::sample;
use crate::document::{AuthorKind, Document};
use crate::registry::Registry;
use crate::registry::tests::real_registry;
use crate::text::{SectionRef, invert};

fn oid(n: u128) -> Id128 {
    Id128::from_u128(0x0123_4567_89ab_cdef_0000_0000_0000_0000 + n)
}

fn float(v: f64) -> Value {
    Value::from(Number::Float(v))
}

fn scalar(id: &str, v: f64) -> Action {
    Action::SetParams { values: vec![(id.into(), ParamValue::Scalar(v))], relative: false, policy: RangePolicy::Reject }
}

/// One action of every `[CMD-02]` kind, valid on [`sample`] with the blobs in `store`.
fn every_kind(store: &BlobStore<'_>) -> Vec<Action> {
    let blob = store.put(b"stroke delta", crate::blob::Compression::None).unwrap();
    let mut layer = Object::new();
    layer.insert("name", "Cheek fill");
    layer.insert("region", "cheek-L");
    let mut skin = Object::new();
    skin.insert("name", "Paint");
    skin.insert("channels", Value::Array(vec![Value::from("albedo_tint")]));
    let mut source = Object::new();
    source.insert("kind", "painted");
    source.insert("texture", Value::Null);
    skin.insert("source", Value::Object(source));
    let mut groom = Object::new();
    groom.insert("name", "Brows");
    groom.insert("kind", "eyebrows");
    let mut attach = Object::new();
    attach.insert("name", "Earring");
    attach.insert("mesh", blob.to_hex().as_str());
    attach.insert("format", "glb");
    attach.insert("joint", "head");
    let mut reference = Object::new();
    reference.insert("name", "Side photo");
    reference.insert("kind", "photo");
    reference.insert("path", "photos/side.png");
    let mut opacity = Object::new();
    opacity.insert("opacity", float(0.4));
    let mut procedural = Object::new();
    let mut src = Object::new();
    src.insert("generator", "freckles");
    src.insert("kind", "procedural");
    let mut params = Object::new();
    params.insert("density", float(0.7));
    src.insert("params", Value::Object(params));
    src.insert("seed", Value::from(Number::from(43_u64)));
    procedural.insert("source", Value::Object(src));
    let mut replace = Object::new();
    replace.insert("path", "photos/front-v2.png");
    vec![
        scalar("face.nose.bridge_height", 0.4),
        Action::SetParams {
            values: vec![
                ("eyes.iris.color.l".into(), ParamValue::Color([0.3, 0.2, 0.1])),
                ("expression.au12_lip_corner_puller.l".into(), ParamValue::Scalar(0.1)),
            ],
            relative: false,
            policy: RangePolicy::Reject,
        },
        Action::ResetParams { ids: vec!["eyes.iris.heterochromia".into()] },
        Action::SetPose { joints: vec![("neck-01".into(), None), ("head".into(), Some([0.0, 0.0, 0.6, 0.8]))] },
        Action::ApplyExpressionPreset { preset: oid(40) },
        Action::ApplyPosePreset { preset: oid(41) },
        Action::AddObject { kind: ObjectKind::SculptLayer, id: Id128::NIL, index: Some(0), fields: layer },
        Action::AddObject { kind: ObjectKind::SkinLayer, id: oid(23), index: None, fields: skin },
        Action::AddObject { kind: ObjectKind::Groom, id: Id128::NIL, index: None, fields: groom },
        Action::AddObject { kind: ObjectKind::Attachment, id: Id128::NIL, index: None, fields: attach },
        Action::AddObject { kind: ObjectKind::Reference, id: Id128::NIL, index: None, fields: reference },
        Action::ModifyObject { kind: ObjectKind::SculptLayer, id: oid(10), fields: opacity },
        Action::ModifyObject { kind: ObjectKind::SkinLayer, id: oid(20), fields: procedural },
        Action::ModifyObject { kind: ObjectKind::Reference, id: oid(80), fields: replace },
        Action::DeleteObject { kind: ObjectKind::SkinLayer, id: oid(22) },
        Action::DeleteObject { kind: ObjectKind::Attachment, id: oid(70) },
        Action::ReorderObjects { kind: ObjectKind::SkinLayer, order: vec![oid(23), oid(21), oid(20)] },
        Action::Stroke {
            stroke: StrokeKind::Sculpt,
            target: oid(10),
            blob,
            regions: vec!["nose".into()],
            elements: 120,
        },
        Action::Stroke { stroke: StrokeKind::Paint, target: oid(21), blob, regions: vec![], elements: 4096 },
        Action::Stroke { stroke: StrokeKind::Guides, target: oid(30), blob, regions: vec![], elements: 32 },
        Action::SetLock { target: LockTarget::Parameter("face.nose.bridge_height".into()), locked: true },
        Action::SetLock { target: LockTarget::Region("nasal-tip".into()), locked: false },
        Action::SetObjectLock { kind: ObjectKind::Groom, id: oid(30), locked: true },
        Action::PinLandmark { landmark: "nose.tip".into(), position: [0.0, 1.6, 0.12] },
        Action::PinLandmark { landmark: "L.iris.center".into(), position: [0.03, 1.65, 0.1] },
        Action::UnpinLandmark { landmark: "nose.tip".into() },
        Action::CreateSnapshot { id: Id128::NIL, name: "Now".into(), state: Hash128::ZERO },
        Action::AddAuthor {
            author: Author { id: "agent.second".into(), kind: AuthorKind::Agent, name: "Second".into() },
        },
        Action::SetSettings { protect_human_edits: false },
        Action::Batch { actions: vec![scalar("face.nose.bridge_height", 0.2), scalar("face.nose.bridge_height", 0.3)] },
        Action::Edits {
            edits: vec![Edit::Set {
                section: SectionRef::new("meta", None),
                key: "name".into(),
                before: Some(Value::from("Ada \"test\" Lovelace")),
                after: Some(Value::from("Ada")),
            }],
        },
    ]
}

fn command(action: Action, n: u128) -> Command {
    Command {
        id: Id128::from_u128(1000 + n),
        author: "human.artist".into(),
        time_ms: 1_763_600_000_000 + n as u64,
        label: format!("step {n}"),
        gesture: false,
        action,
    }
}

#[test]
fn cmd02_every_kind_resolves_applies_inverts_and_round_trips_as_json() {
    let reg = real_registry();
    let fs = MemoryFs::new();
    let store = BlobStore::new(&fs, "blobs");
    let ctx = Context { registry: &reg, blobs: Some(&store) };
    let mut doc = sample(&reg);
    let mut kinds = Vec::new();
    for (n, action) in every_kind(&store).into_iter().enumerate() {
        let cmd = command(action, n as u128);
        // [CMD-01] commands are data: the JSON form reads back to the same command.
        let json = command_to_json(&cmd);
        let back = command_from_json(&json, &reg, &CommandLimits::DEFAULT).unwrap();
        assert_eq!(back, cmd, "json round trip of {}", cmd.action.kind_name());
        let resolved = resolve(&doc, &cmd, &ctx).unwrap_or_else(|e| panic!("{}: {e}", cmd.action.kind_name()));
        assert!(!resolved.edits.is_empty(), "{} changed nothing", cmd.action.kind_name());
        let edits_json = edits_to_json(&resolved.edits);
        assert_eq!(edits_from_json(&edits_json, &CommandLimits::DEFAULT).unwrap(), resolved.edits);
        let before = doc.clone();
        apply_edits(&mut doc, &reg, &resolved.edits).unwrap_or_else(|e| panic!("{}: {e}", cmd.action.kind_name()));
        assert_ne!(doc, before);
        // [CMD-06] the outcome lists something for every command.
        assert!(!describe(&resolved.edits, &reg).is_empty(), "{}", cmd.action.kind_name());
        // [CMD-01] the inverse restores the state exactly.
        let mut undone = doc.clone();
        apply_edits(&mut undone, &reg, &invert(&resolved.edits)).unwrap();
        assert_eq!(undone, before, "inverse of {}", cmd.action.kind_name());
        kinds.push(cmd.action.kind_name());
    }
    for name in [
        "set_params",
        "add_sculpt_layer",
        "modify_skin_layer",
        "delete_skin_layer",
        "reorder_skin_layers",
        "sculpt_stroke",
        "paint_stroke",
        "groom_guide_edit",
        "add_groom",
        "set_lock",
        "pin_landmark",
        "create_snapshot",
        "attach",
        "detach",
        "import_reference",
        "replace_reference",
        "batch",
        "apply_expression_preset",
        "apply_pose_preset",
        "set_pose",
    ] {
        assert!(kinds.contains(&name), "{name} not covered");
    }
    // The stroke reported the elements it touched; the new objects carry the command's author.
    assert!(doc.sculpt_layers.iter().any(|l| l.name == "Cheek fill" && l.author == "human.artist"));
    assert_eq!(doc.sculpt_layers[0].name, "Cheek fill");
}

#[test]
fn cmd02_restore_snapshot_brings_back_the_character_state() {
    let reg = real_registry();
    let fs = MemoryFs::new();
    let store = BlobStore::new(&fs, "blobs");
    let ctx = Context { registry: &reg, blobs: Some(&store) };
    let mut doc = sample(&reg);
    let snap = command(Action::CreateSnapshot { id: oid(61), name: "Base".into(), state: Hash128::ZERO }, 1);
    let e = resolve(&doc, &snap, &ctx).unwrap().edits;
    apply_edits(&mut doc, &reg, &e).unwrap();
    let at_snapshot = doc.clone();
    for (n, action) in
        [scalar("face.nose.bridge_height", -0.2), Action::DeleteObject { kind: ObjectKind::Groom, id: oid(30) }]
            .into_iter()
            .enumerate()
    {
        let cmd = command(action, 10 + n as u128);
        let e = resolve(&doc, &cmd, &ctx).unwrap().edits;
        apply_edits(&mut doc, &reg, &e).unwrap();
    }
    let restore = command(Action::RestoreSnapshot { id: oid(61) }, 20);
    let edits = resolve(&doc, &restore, &ctx).unwrap().edits;
    apply_edits(&mut doc, &reg, &edits).unwrap();
    assert_eq!(doc, at_snapshot);
}

#[test]
fn cmd05_range_policy_rejects_or_clamps_and_says_so() {
    let reg = real_registry();
    let doc = sample(&reg);
    let ctx = Context { registry: &reg, blobs: None };
    let id = "face.nose.bridge_height";
    let range = reg.get(id).unwrap().range.unwrap();
    let mut cmd = command(scalar(id, range.max + 5.0), 1);
    let err = resolve(&doc, &cmd, &ctx).unwrap_err();
    assert_eq!(err.code(), "out_of_range", "{err}");
    cmd.action = Action::SetParams {
        values: vec![(id.into(), ParamValue::Scalar(range.max + 5.0))],
        relative: false,
        policy: RangePolicy::Clamp,
    };
    let r = resolve(&doc, &cmd, &ctx).unwrap();
    assert!(r.warnings.iter().any(|w| w.code == WarningCode::Clamped && w.target == id));
    match &describe(&r.edits, &reg)[0] {
        Change::Param { new, .. } => assert_eq!(*new, ParamValue::Scalar(range.max)),
        other => panic!("{other:?}"),
    }
    // Outside the plausible span but inside the range: applied as asked, with a warning.
    if range.soft_max < range.max {
        cmd.action = scalar(id, f64::midpoint(range.soft_max, range.max));
        let r = resolve(&doc, &cmd, &ctx).unwrap();
        assert!(r.warnings.iter().any(|w| w.code == WarningCode::Implausible));
    }
    // Relative mode adds; a parameter named twice is refused; unknown parameters are invalid.
    cmd.action = Action::SetParams {
        values: vec![(id.into(), ParamValue::Scalar(0.05))],
        relative: true,
        policy: RangePolicy::Reject,
    };
    match &describe(&resolve(&doc, &cmd, &ctx).unwrap().edits, &reg)[0] {
        Change::Param { old, new, .. } => {
            assert_eq!((old.clone(), new.clone()), (ParamValue::Scalar(0.35), ParamValue::Scalar(0.35 + 0.05)));
        }
        other => panic!("{other:?}"),
    }
    cmd.action = Action::SetParams {
        values: vec![(id.into(), ParamValue::Scalar(0.1)), (id.into(), ParamValue::Scalar(0.2))],
        relative: false,
        policy: RangePolicy::Reject,
    };
    assert_eq!(resolve(&doc, &cmd, &ctx).unwrap_err().code(), "invalid");
    cmd.action = scalar("no.such.param", 0.1);
    assert_eq!(resolve(&doc, &cmd, &ctx).unwrap_err().code(), "invalid");
}

/// Random edits of the fast-path sections, valid and invalid.
fn random_fast_edit(doc: &Document, reg: &Registry, rng: &mut Rng) -> Edit {
    let scalars: Vec<&crate::registry::Param> =
        reg.params().iter().filter(|p| p.kind == crate::registry::ParamKind::Scalar && p.range.is_some()).collect();
    let p = scalars[rng.index(scalars.len())];
    let r = p.range.unwrap_or(crate::registry::Range { min: 0.0, max: 1.0, soft_min: 0.0, soft_max: 1.0, step: 0.1 });
    let current = doc.params.get(&p.id).map(crate::registry::value_to_json);
    let pick = |rng: &mut Rng| -> Option<Value> {
        match rng.below(6) {
            0 => None,
            1 => Some(float(r.min + (r.max - r.min) * rng.f64())),
            2 => Some(crate::registry::value_to_json(&p.default)),
            3 => Some(float(r.max + 1.0)),
            4 => Some(Value::from(Number::from(1_u64))),
            _ => Some(float(((r.min + (r.max - r.min) * rng.f64()) * 4.0).round() / 4.0)),
        }
    };
    match rng.below(5) {
        0 | 1 => {
            let before = if rng.chance(0.85) { current } else { pick(rng) };
            Edit::Set { section: SectionRef::new("params", None), key: p.id.clone(), before, after: pick(rng) }
        }
        2 => {
            let before = doc.param_authors.get(&p.id).map(|a| Value::from(a.as_str()));
            let after = match rng.below(4) {
                0 => None,
                1 => Some(Value::from("nobody")),
                2 => Some(Value::from("agent.assistant")),
                _ => Some(Value::from("human.artist")),
            };
            Edit::Set { section: SectionRef::new("param_authors", None), key: p.id.clone(), before, after }
        }
        _ => {
            let joint = ["neck-01", "head", "jaw", "bad joint"][rng.index(4)];
            let before = doc.pose.get(joint).map(|q| Value::Array(q.iter().map(|v| float(*v)).collect()));
            let after = match rng.below(4) {
                0 => None,
                1 => Some(Value::Array(vec![float(0.0), float(0.0), float(0.6), float(0.8)])),
                2 => Some(Value::Array(vec![float(0.0), float(0.0), float(0.0), float(2.0)])),
                _ => Some(Value::Array(vec![float(0.0), float(0.0), float(0.0), Value::from(Number::from(1_u64))])),
            };
            Edit::Set { section: SectionRef::new("pose", None), key: joint.into(), before, after }
        }
    }
}

#[test]
fn fast_path_accepts_and_refuses_exactly_what_document_apply_edits_does() {
    let reg = real_registry();
    let mut rng = Rng::new(0x005e_ed08);
    let mut fast = sample(&reg);
    let mut accepted = 0;
    for case in 0..3000 {
        let n = 1 + rng.index(3);
        let edits: Vec<Edit> = (0..n).map(|_| random_fast_edit(&fast, &reg, &mut rng)).collect();
        let mut slow = fast.clone();
        let slow_result = slow.apply_edits(&edits, &reg);
        let before = fast.clone();
        let fast_result = apply_edits(&mut fast, &reg, &edits);
        assert_eq!(fast_result.is_ok(), slow_result.is_ok(), "case {case}: {edits:?} {fast_result:?} {slow_result:?}");
        if fast_result.is_ok() {
            accepted += 1;
        } else {
            assert_eq!(fast, before, "case {case}: a refused edit list changed the document");
        }
        assert_eq!(fast, slow, "case {case}");
    }
    assert!(accepted > 300, "only {accepted} edit lists were valid");
}

#[test]
fn rob01_command_decoding_survives_mutated_input() {
    let reg = real_registry();
    let fs = MemoryFs::new();
    let store = BlobStore::new(&fs, "blobs");
    let corpus: Vec<Vec<u8>> = every_kind(&store)
        .into_iter()
        .enumerate()
        .map(|(n, a)| json::to_string(&command_to_json(&command(a, n as u128))).unwrap().into_bytes())
        .collect();
    let limits = JsonLimits { max_depth: 32, ..JsonLimits::default() };
    humanart_core::fuzz::run(
        "command-json",
        &corpus,
        60,
        0x0c0d_e808,
        std::time::Duration::from_secs(120),
        move |bytes| {
            if let Ok(value) = json::parse_with(bytes, &limits)
                && let Ok(cmd) = command_from_json(&value, &reg, &CommandLimits::DEFAULT)
            {
                let doc = sample(&reg);
                let ctx = Context { registry: &reg, blobs: None };
                if let Ok(r) = resolve(&doc, &cmd, &ctx) {
                    let mut d = doc.clone();
                    if apply_edits(&mut d, &reg, &r.edits).is_ok() {
                        apply_edits(&mut d, &reg, &invert(&r.edits)).unwrap();
                        assert_eq!(d, doc);
                    }
                }
            }
        },
    );
}

#[test]
fn codec_limits_refuse_oversized_and_deep_input() {
    let reg = real_registry();
    let small = CommandLimits { max_edits: 2, max_actions: 3, max_values: 2, max_index: 10 };
    let mut nested = json::parse_str(r#"{"type":"set_settings","protect_human_edits":true}"#).unwrap();
    for _ in 0..6 {
        let mut o = Object::new();
        o.insert("type", "batch");
        o.insert("actions", Value::Array(vec![nested]));
        nested = Value::Object(o);
    }
    assert!(super::codec::action_from_json(&nested, &reg, &CommandLimits::DEFAULT).is_err());
    let many = json::parse_str(r#"{"type":"reset_params","ids":["a","b","c"]}"#).unwrap();
    assert!(super::codec::action_from_json(&many, &reg, &small).is_err());
    let far =
        json::parse_str(r#"[{"op":"insert","index":11,"section":"pin","section_key":"x","entries":{}}]"#).unwrap();
    assert!(edits_from_json(&far, &small).is_err());
    let unknown = json::parse_str(r#"{"type":"unpin_landmark","landmark":"a","extra":1}"#).unwrap();
    assert!(super::codec::action_from_json(&unknown, &reg, &small).is_err());
    // Absent and null differ for before/after.
    let e = json::parse_str(r#"[{"op":"set","section":"params","key":"a","after":null}]"#).unwrap();
    match &edits_from_json(&e, &small).unwrap()[0] {
        Edit::Set { before, after, .. } => assert_eq!((before.clone(), after.clone()), (None, Some(Value::Null))),
        other => panic!("{other:?}"),
    }
    assert!(!hash128(b"x").is_zero());
}

#[test]
fn coalescing_targets_and_kind_names_are_stable() {
    let a = scalar("face.b", 0.1);
    let b = Action::SetParams {
        values: vec![("face.b".into(), ParamValue::Scalar(0.2))],
        relative: true,
        policy: RangePolicy::Clamp,
    };
    assert_eq!(a.coalesce_target(), b.coalesce_target());
    assert_eq!(a.kind_name(), "set_params");
    assert_eq!(Action::DeleteObject { kind: ObjectKind::Attachment, id: oid(1) }.kind_name(), "detach");
    assert!(Action::SetSettings { protect_human_edits: true }.coalesce_target().is_none());
    assert_ne!(derive_id(oid(1), 1), derive_id(oid(1), 2));
    assert_ne!(derive_id(oid(1), 1), derive_id(oid(2), 1));
}
