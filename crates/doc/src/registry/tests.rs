use super::*;
use humanart_core::platform::MemoryFs;
use humanart_core::rng::Rng;
use humanart_io::json::Value;

const SMALL: &str = r#"{
  "format": "humanart-params",
  "version": 1,
  "description": "test file",
  "defaults": {"group": "face/nose", "kind": "scalar", "unit": "coefficient", "min": -3.0, "max": 3.0,
               "soft_min": -2.0, "soft_max": 2.0, "step": 0.01, "default": 0.0, "affects": ["shape"],
               "plausibility_weight": 1.0},
  "parameters": [
    {"id": "face.nose.width", "display_name": "Nose width", "region": "nasal-tip", "tags": ["nose", "width"],
     "description": "Widens the nose at the alar base; negative values narrow it.",
     "evaluation": {"kind": "component", "target": "semantic.nose_width"}},
    {"id": "expression.au12_lip_corner_puller", "sides": true, "display_name": "Lip corner puller",
     "group": "expression/mouth", "region": "lip-corner-{side}", "unit": "weight", "min": 0.0, "max": 1.0,
     "soft_min": 0.0, "soft_max": 1.0, "affects": ["expression"], "plausibility_weight": 0.0,
     "description": "FACS AU12: pulls the lip corner up and back, as in a smile.",
     "evaluation": {"kind": "expression", "target": "AU12-{side}"}},
    {"id": "eyes.iris.heterochromia", "display_name": "Heterochromia", "group": "eyes/iris", "kind": "enum",
     "choices": ["none", "central"], "default": "none", "unit": "", "min": null, "max": null, "soft_min": null,
     "soft_max": null, "step": null, "affects": ["eyes"],
     "description": "Iris colour pattern: none, or a central ring of a second colour around the pupil.",
     "evaluation": {"kind": "material", "target": "eye.heterochromia"}},
    {"id": "eyes.iris.color", "display_name": "Iris colour", "group": "eyes/iris", "kind": "color",
     "unit": "linear_rgb", "min": 0.0, "max": 1.0, "soft_min": 0.0, "soft_max": 1.0,
     "default": [0.1, 0.05, 0.02], "affects": ["eyes"], "provisional": "no reference data yet",
     "description": "Base colour of the iris stroma, linear Rec.709.",
     "evaluation": {"kind": "material", "target": "eye.iris_color"}},
    {"id": "skin.pores.map", "display_name": "Pore map", "group": "skin/pores", "kind": "asset_ref",
     "default": null, "unit": "", "min": null, "max": null, "soft_min": null, "soft_max": null, "step": null,
     "affects": ["skin"], "description": "Painted per-region modulation map of the pore density.",
     "evaluation": {"kind": "material", "target": "skin.pore_map"}},
    {"id": "measure.face_width", "display_name": "Face width", "group": "identity/measurements", "unit": "m",
     "min": 0.0, "max": 1.0, "soft_min": 0.1, "soft_max": 0.2, "default": 0.0, "read_only": true,
     "affects": ["shape"], "description": "Distance between the zygion landmarks, measured on the rest shape.",
     "evaluation": {"kind": "measurement", "landmarks": ["L.zygion", "R.zygion"], "axis": "x"}},
    {"id": "lighting_camera.exposure", "display_name": "Exposure", "group": "lighting_camera", "unit": "ev",
     "min": -10.0, "max": 10.0, "soft_min": -4.0, "soft_max": 4.0, "affects": ["view"],
     "description": "Viewport exposure compensation in stops; does not change the character.",
     "evaluation": {"kind": "editor", "target": "camera.exposure"}},
    {"id": "skin.curve", "display_name": "Curve", "group": "skin", "kind": "curve", "unit": "fraction",
     "min": 0.0, "max": 1.0, "soft_min": 0.0, "soft_max": 1.0, "default": [[0.0, 0.0], [1.0, 1.0]],
     "affects": ["skin"], "description": "A test curve mapping age to wrinkle depth.",
     "evaluation": {"kind": "material", "target": "skin.curve"}},
    {"id": "body.flag", "display_name": "Flag", "group": "body", "kind": "bool", "default": false, "unit": "",
     "min": null, "max": null, "soft_min": null, "soft_max": null, "step": null, "affects": ["shape"],
     "description": "A test switch that turns something on.", "evaluation": {"kind": "rig", "target": "root"}},
    {"id": "body.offset", "display_name": "Offset", "group": "body", "kind": "vector", "dimension": 3,
     "unit": "m", "min": -1.0, "max": 1.0, "soft_min": -0.5, "soft_max": 0.5, "default": [0.0, 0.0, 0.0],
     "affects": ["pose"], "description": "A test offset vector in metres.",
     "evaluation": {"kind": "rig", "target": "root"}}
  ]
}"#;

fn small() -> Registry {
    Registry::from_files(&[("small.json", SMALL.as_bytes())], &RegistryLimits::DEFAULT).unwrap()
}

#[test]
fn reg_01_parses_every_kind_with_defaults_and_sides() {
    let r = small();
    assert_eq!(r.len(), 11);
    let ids: Vec<&str> = r.params().iter().map(|p| p.id.as_str()).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(ids, sorted);
    let left = r.get("expression.au12_lip_corner_puller.l").unwrap();
    assert_eq!(left.display_name, "Lip corner puller (left)");
    assert_eq!(left.region.as_deref(), Some("lip-corner-L"));
    assert_eq!(left.evaluation, Evaluation::Expression("AU12-L".into()));
    assert_eq!(
        left.symmetry,
        Symmetry::Mirror { partner: "expression.au12_lip_corner_puller.r".into(), side: Side::Left }
    );
    let nose = r.get("face.nose.width").unwrap();
    assert_eq!(nose.range.unwrap().soft_min, -2.0);
    assert_eq!(nose.tags, ["nose", "width"]);
    assert_eq!(nose.plausibility_weight, 1.0);
    assert!(r.get("measure.face_width").unwrap().read_only);
    assert!(r.get("lighting_camera.exposure").unwrap().editor_only());
    assert!(!nose.editor_only());
    assert_eq!(r.get("skin.pores.map").unwrap().default, ParamValue::AssetRef(Hash128::ZERO));
    assert_eq!(r.in_group("eyes").count(), 2);
    assert_eq!(r.in_group("eye").count(), 0);
    assert_eq!(r.in_region("nasal-tip").count(), 1);
    assert_eq!(r.with_tag("nose").count(), 1);
    assert!(matches!(r.require("nope"), Err(DocError::UnknownParameter(_))));
}

#[test]
fn reg_01_values_are_checked_and_clamped() {
    let r = small();
    let nose = r.get("face.nose.width").unwrap();
    assert!(nose.check_value(&ParamValue::Scalar(2.5)).is_ok());
    assert!(nose.check_value(&ParamValue::Scalar(3.5)).is_err());
    assert!(nose.check_value(&ParamValue::Scalar(f64::NAN)).is_err());
    assert!(nose.check_value(&ParamValue::Bool(true)).is_err());
    assert_eq!(nose.clamp_value(&ParamValue::Scalar(9.0)), ParamValue::Scalar(3.0));
    assert_eq!(nose.clamp_value(&ParamValue::Bool(true)), nose.default);
    let het = r.get("eyes.iris.heterochromia").unwrap();
    assert!(het.check_value(&ParamValue::Enum("central".into())).is_ok());
    assert!(het.check_value(&ParamValue::Enum("sectoral".into())).is_err());
    let curve = r.get("skin.curve").unwrap();
    assert!(curve.check_value(&ParamValue::Curve(vec![[0.0, 0.5], [0.5, 0.2], [1.0, 1.0]])).is_ok());
    assert!(curve.check_value(&ParamValue::Curve(vec![[0.5, 0.5], [0.5, 0.2]])).is_err());
    assert!(curve.check_value(&ParamValue::Curve(vec![[0.0, 0.5]])).is_err());
    let offset = r.get("body.offset").unwrap();
    assert!(offset.check_value(&ParamValue::Vector(vec![0.0, 0.1])).is_err());
    assert_eq!(offset.clamp_value(&ParamValue::Vector(vec![2.0])), ParamValue::Vector(vec![1.0, 0.0, 0.0]));
    let json = value_to_json(&ParamValue::Color([0.25, 0.5, 1.0]));
    let color = r.get("eyes.iris.color").unwrap();
    assert_eq!(value_from_json(color, &json).unwrap(), ParamValue::Color([0.25, 0.5, 1.0]));
    assert!(value_from_json(color, &Value::from("red")).is_err());
}

fn with_entry(entry: &str) -> String {
    format!(
        r#"{{"format": "humanart-params", "version": 1, "parameters": [{{"id": "a.b", "display_name": "A",
        "description": "A parameter used to test the checks.", "group": "body", "kind": "scalar", "unit": "m",
        "min": 0.0, "max": 1.0, "default": 0.5, "affects": ["shape"],
        "evaluation": {{"kind": "component", "target": "a"}}{entry}}}]}}"#
    )
}

#[test]
fn reg_01_invalid_entries_are_refused() {
    let limits = RegistryLimits::DEFAULT;
    assert!(parse_file("ok", with_entry("").as_bytes(), &limits).is_ok());
    let bad = [
        r#", "id": "A.b""#,
        r#", "group": "nowhere""#,
        r#", "min": 2.0"#,
        r#", "soft_min": -1.0"#,
        r#", "default": 2.0"#,
        r#", "step": 0.0"#,
        r#", "kind": "vector""#,
        r#", "dimension": 2"#,
        r#", "choices": ["x"]"#,
        r#", "region": "bad region""#,
        r#", "affects": []"#,
        r#", "affects": ["smell"]"#,
        r#", "read_only": true"#,
        r#", "plausibility_weight": -1.0"#,
        r#", "since_version": 0"#,
        r#", "tags": ["Bad Tag"]"#,
        r#", "provisional": " ""#,
        r#", "unknown": 1"#,
        r#", "symmetry": {"kind": "mirror", "partner": "a.b", "side": "left"}"#,
        r#", "symmetry": {"kind": "none", "side": "left"}"#,
        r#", "evaluation": {"kind": "teleport", "target": "a"}"#,
        r#", "evaluation": {"kind": "measurement", "landmarks": ["a", "a"]}"#,
        r#", "evaluation": {"kind": "component", "target": "a", "axis": "x"}"#,
        r#", "editor_only": true"#,
        r#", "description": "short""#,
        r#", "sides": true, "symmetry": {"kind": "sided", "side": "left"}"#,
    ];
    for fragment in bad {
        assert!(parse_file("bad", with_entry(fragment).as_bytes(), &limits).is_err(), "{fragment}");
    }
    let wrong_format = SMALL.replace("humanart-params", "other");
    assert!(parse_file("f", wrong_format.as_bytes(), &limits).is_err());
    let wrong_version = SMALL.replace("\"version\": 1", "\"version\": 2");
    assert!(matches!(parse_file("f", wrong_version.as_bytes(), &limits), Err(DocError::Version { .. })));
    let tiny = RegistryLimits { max_params: 3, ..RegistryLimits::DEFAULT };
    assert!(matches!(parse_file("f", SMALL.as_bytes(), &tiny), Err(DocError::Limit { .. })));
    let twice = [("a.json", SMALL.as_bytes()), ("b.json", SMALL.as_bytes())];
    assert!(Registry::from_files(&twice, &limits).is_err(), "duplicate ids across files");
}

#[test]
fn reg_01_mirror_pairs_must_be_reciprocal() {
    let mut params = small().params().to_vec();
    let i = params.iter().position(|p| p.id == "expression.au12_lip_corner_puller.r").unwrap();
    params[i].symmetry = Symmetry::Sided { side: Side::Right };
    assert!(Registry::new(params.clone()).is_err());
    params.remove(i);
    assert!(Registry::new(params).is_err(), "partner missing");
}

#[test]
fn reg_04_search_ranks_names_before_descriptions() {
    let r = small();
    assert_eq!(r.search("face.nose.width")[0].id, "face.nose.width");
    let hits: Vec<&str> = r.search("Nose").iter().map(|p| p.id.as_str()).collect();
    assert_eq!(hits, ["face.nose.width"]);
    let smile: Vec<&str> = r.search("smile").iter().map(|p| p.id.as_str()).collect();
    assert_eq!(smile, ["expression.au12_lip_corner_puller.l", "expression.au12_lip_corner_puller.r"]);
    assert_eq!(r.search("lip corner left").len(), 1);
    assert!(r.search("").is_empty());
    assert!(r.search("zzz").is_empty());
}

/// The members every described entry has `[AC-WP07-2]`.
const DESCRIBED: [&str; 25] = [
    "id",
    "display_name",
    "description",
    "group",
    "region",
    "kind",
    "dimension",
    "choices",
    "unit",
    "min",
    "max",
    "soft_min",
    "soft_max",
    "step",
    "default",
    "symmetry",
    "affects",
    "evaluation",
    "read_only",
    "locked_by_default",
    "plausibility_weight",
    "since_version",
    "tags",
    "provisional",
    "editor_only",
];

/// Checks that the description of `r` is complete: one object per entry, every member present,
/// and each object reads back to the entry it describes.
fn assert_description_complete(r: &Registry) {
    let text = r.describe_parameters().unwrap();
    let value = humanart_io::json::parse(text.as_bytes()).unwrap();
    assert_eq!(value.get("format").and_then(Value::as_str), Some(DESCRIBE_FORMAT_NAME));
    assert_eq!(value.get("count").and_then(Value::as_u64), Some(r.len() as u64));
    let list = value.get("parameters").and_then(Value::as_array).unwrap();
    assert_eq!(list.len(), r.len());
    for (described, param) in list.iter().zip(r.params()) {
        let object = described.as_object().unwrap();
        let keys: Vec<&str> = object.keys().collect();
        assert_eq!(keys, DESCRIBED, "{}", param.id);
        assert_eq!(&param_from_description(described).unwrap(), param, "{}", param.id);
    }
}

#[test]
fn ac_wp07_2_description_of_the_test_registry_is_complete() {
    assert_description_complete(&small());
}

/// A random valid parameter (and its mirror partner, sometimes).
fn random_params(rng: &mut Rng, n: usize) -> Vec<Param> {
    let mut out = Vec::new();
    for i in 0..n {
        let kind = ParamKind::ALL[rng.index(ParamKind::ALL.len())];
        let min = rng.range_f64(-10.0, 0.0);
        let max = min + rng.range_f64(0.5, 10.0);
        let soft_min = min + (max - min) * rng.range_f64(0.0, 0.4);
        let soft_max = max - (max - min) * rng.range_f64(0.0, 0.4);
        let range = kind.is_numeric().then_some(Range { min, max, soft_min, soft_max, step: (max - min) / 50.0 });
        let mid = rng.range_f64(min, max);
        let dimension = match kind {
            ParamKind::Vector => 2 + rng.index(5) as u8,
            ParamKind::Color => 3,
            _ => 1,
        };
        let choices: Vec<String> =
            if kind == ParamKind::Enum { (0..=rng.index(4)).map(|c| format!("c{c}")).collect() } else { Vec::new() };
        let default = match kind {
            ParamKind::Scalar => ParamValue::Scalar(mid),
            ParamKind::Vector => ParamValue::Vector(vec![mid; usize::from(dimension)]),
            ParamKind::Color => ParamValue::Color([mid; 3]),
            ParamKind::Enum => ParamValue::Enum(choices[rng.index(choices.len())].clone()),
            ParamKind::Bool => ParamValue::Bool(rng.chance(0.5)),
            ParamKind::Curve => ParamValue::Curve(vec![[0.0, mid], [rng.range_f64(0.1, 0.9), min], [1.0, max]]),
            ParamKind::AssetRef => ParamValue::AssetRef(Hash128::from_parts(rng.next_u64(), rng.next_u64())),
        };
        let measurement = rng.chance(0.1);
        let base = Param {
            id: format!("g{}.p{i}", rng.index(3)),
            display_name: format!("Param {i}"),
            description: format!("Random parameter number {i} for the property tests."),
            group: format!("{}/sub{}", GROUPS[rng.index(GROUPS.len())], rng.index(3)),
            region: rng.chance(0.5).then(|| format!("region-{}", rng.index(5))),
            kind,
            dimension,
            choices,
            unit: ["", "m", "weight", "coefficient"][rng.index(4)].to_owned(),
            range,
            default,
            symmetry: Symmetry::None,
            affects: {
                let mut a: Vec<Channel> = (0..=rng.index(3)).map(|_| Channel::ALL[rng.index(8)]).collect();
                a.sort_unstable();
                a.dedup();
                a
            },
            evaluation: if measurement {
                Evaluation::Measurement {
                    landmarks: [format!("L.p{i}"), format!("R.p{i}")],
                    axis: [None, Some(Axis::X), Some(Axis::Y), Some(Axis::Z)][rng.index(4)],
                }
            } else {
                Evaluation::Material(format!("slot.p{i}"))
            },
            read_only: measurement,
            locked_by_default: rng.chance(0.2),
            plausibility_weight: rng.range_f64(0.0, 2.0),
            since_version: 1 + rng.index(3) as u32,
            tags: {
                let mut t: Vec<String> = (0..rng.index(3)).map(|k| format!("tag{k}")).collect();
                t.sort();
                t
            },
            provisional: rng.chance(0.3).then(|| "test reason".to_owned()),
        };
        if rng.chance(0.3) {
            let mut left = base.clone();
            let mut right = base;
            left.id.push_str(".l");
            right.id.push_str(".r");
            left.symmetry = Symmetry::Mirror { partner: right.id.clone(), side: Side::Left };
            right.symmetry = Symmetry::Mirror { partner: left.id.clone(), side: Side::Right };
            out.push(left);
            out.push(right);
        } else {
            if rng.chance(0.1) {
                let mut sided = base;
                sided.symmetry = Symmetry::Sided { side: Side::Right };
                out.push(sided);
                continue;
            }
            out.push(base);
        }
    }
    out
}

fn random_value(rng: &mut Rng, p: &Param) -> ParamValue {
    let wide = |rng: &mut Rng| rng.range_f64(-30.0, 30.0);
    match p.kind {
        ParamKind::Scalar => ParamValue::Scalar(wide(rng)),
        ParamKind::Vector => ParamValue::Vector((0..rng.index(6)).map(|_| wide(rng)).collect()),
        ParamKind::Color => ParamValue::Color([wide(rng), wide(rng), wide(rng)]),
        ParamKind::Enum => ParamValue::Enum(format!("c{}", rng.index(6))),
        ParamKind::Bool => ParamValue::Bool(rng.chance(0.5)),
        ParamKind::Curve => ParamValue::Curve((0..rng.index(5)).map(|k| [k as f64 / 4.0, wide(rng)]).collect()),
        ParamKind::AssetRef => ParamValue::AssetRef(Hash128::from_parts(rng.next_u64(), 0)),
    }
}

#[test]
fn ac_wp07_1_registry_invariants_hold_for_random_registries() {
    let mut rng = Rng::new(0x5eed_0701);
    for case in 0..300 {
        let n = 1 + rng.index(30);
        let params = random_params(&mut rng, n);
        let r = Registry::new(params).unwrap_or_else(|e| panic!("case {case}: {e}"));
        // Sorted, unique, every id found where it is.
        assert!(r.params().windows(2).all(|w| w[0].id < w[1].id));
        for (i, p) in r.params().iter().enumerate() {
            assert_eq!(r.index_of(&p.id), Some(i));
            assert_eq!(r.search(&p.id)[0].id, p.id, "case {case}: an exact id ranks first");
            assert!(p.check_value(&p.default).is_ok());
            if let Some(range) = p.range {
                assert!(range.min <= range.soft_min && range.soft_min < range.soft_max && range.soft_max <= range.max);
            }
            if let Symmetry::Mirror { partner, side } = &p.symmetry {
                let other = r.get(partner).unwrap();
                assert_eq!(other.side(), Some(side.mirror()));
            }
            for _ in 0..8 {
                let v = random_value(&mut rng, p);
                let clamped = p.clamp_value(&v);
                assert!(p.check_value(&clamped).is_ok(), "case {case}: {} {v:?} -> {clamped:?}", p.id);
                assert_eq!(p.clamp_value(&clamped), clamped, "clamping is idempotent");
                if p.check_value(&v).is_ok() {
                    assert_eq!(clamped, v, "valid values are not changed");
                }
                let json = value_to_json(&clamped);
                assert_eq!(value_from_json(p, &json).unwrap(), clamped, "values round-trip through JSON");
            }
        }
        assert_description_complete(&r);
    }
}

/// Reads the real registry under `data/template/params/` into a memory file system.
pub(crate) fn real_registry() -> Registry {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/template/params");
    let fs = MemoryFs::new();
    fs.create_dir_all("params").unwrap();
    let mut files = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().into_string().unwrap();
        fs.write(&format!("params/{name}"), &std::fs::read(entry.path()).unwrap()).unwrap();
        files += 1;
    }
    assert!(files > 0);
    Registry::load_dir(&fs, "params", &RegistryLimits::DEFAULT).unwrap()
}

#[test]
fn ac_wp07_2_describe_parameters_is_complete_for_the_real_registry() {
    let r = real_registry();
    assert!(r.len() > 100, "{} parameters", r.len());
    assert_description_complete(&r);
    // Every REG-03 group except the editor-only one has parameters.
    for group in GROUPS.iter().filter(|g| **g != EDITOR_GROUP) {
        assert!(r.in_group(group).next().is_some(), "group {group} is empty");
    }
}

#[test]
fn reg_03_real_registry_covers_appendix_b() {
    let r = real_registry();
    let aus = [
        1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 38, 39, 41, 42, 43, 44, 45, 46,
    ];
    for au in aus {
        let prefix = format!("expression.au{au:02}_");
        assert!(r.params().iter().any(|p| p.id.starts_with(&prefix)), "AU{au} missing");
    }
    for p in r.in_group("expression") {
        let range = p.range.unwrap();
        assert_eq!((range.min, range.max, &p.default), (0.0, 1.0, &ParamValue::Scalar(0.0)), "{}", p.id);
        assert!(matches!(p.evaluation, Evaluation::Expression(_)), "{}", p.id);
    }
    // Physical constants from the specification keep their values `[EYE-02]`.
    assert_eq!(r.get("eyes.cornea.ior").unwrap().default, ParamValue::Scalar(1.376));
    assert_eq!(r.get("eyes.aqueous.ior").unwrap().default, ParamValue::Scalar(1.336));
    // Measurements are read-only `[REG-05]`.
    assert!(r.params().iter().filter(|p| p.read_only).count() >= 3);
}

#[test]
fn reg_02_template_coverage_reports_both_directions() {
    use humanart_math::{Vec2, Vec3};
    use humanart_model::delta::{DeltaBasesBuilder, DeltaKind};
    use humanart_model::expression::{Expression, ExpressionKind, ExpressionSet};
    use humanart_model::landmark::{Landmark, LandmarkGroup};
    use humanart_model::region::{Region, RegionMap};
    use humanart_model::skeleton::{JointRegressor, Skeleton, simple_joint};
    use humanart_model::skin::SkinWeights;
    use humanart_model::template::ProvenanceEntry;
    use humanart_model::{MeshBuilder, Template, TemplateInfo, TemplateParts, UvTile};

    let mut mb = MeshBuilder::new();
    let a = mb.vertex(Vec3::new(0.0, 0.0, 0.0), Vec2::new(0.0, 1.0));
    let b = mb.vertex(Vec3::new(1.0, 0.0, 0.0), Vec2::new(1.0, 1.0));
    let c = mb.vertex(Vec3::new(1.0, 1.0, 0.0), Vec2::new(1.0, 0.0));
    let d = mb.vertex(Vec3::new(0.0, 1.0, 0.0), Vec2::new(0.0, 0.0));
    mb.begin_part("skin", false).unwrap();
    mb.face(&[a, b, c, d], 0).unwrap();
    let mesh = mb.finish().unwrap();
    let mut db = DeltaBasesBuilder::new();
    db.add("semantic.nose_width", DeltaKind::Semantic, 0, &[(0, Vec3::X * 0.01)]).unwrap();
    db.add("semantic.orphan", DeltaKind::Semantic, 0, &[(1, Vec3::X * 0.01)]).unwrap();
    db.add("AU12-L", DeltaKind::Expression, 0, &[(2, Vec3::Y * 0.01)]).unwrap();
    let deltas = db.finish().unwrap();
    let expression = Expression {
        name: "AU12-L".into(),
        kind: ExpressionKind::ActionUnit,
        side: humanart_model::Side::Left,
        facs: 12,
        component: deltas.find("AU12-L").unwrap(),
    };
    let landmark = |name: &str| Landmark {
        name: name.into(),
        group: LandmarkGroup::Face,
        vertices: [0, 1, 2],
        weights: [0.2, 0.3, 0.5],
    };
    let parts = TemplateParts {
        info: TemplateInfo {
            name: "coverage-test".into(),
            version: 1,
            generator: "humanart-doc tests".into(),
            licence: "in-house".into(),
            provenance: vec![ProvenanceEntry {
                item: "everything".into(),
                source: "test".into(),
                licence: "in-house".into(),
            }],
            extra: humanart_io::json::Object::new(),
        },
        tiles: vec![UvTile { name: "face".into(), resolution: 1024, target_texels_per_mm: 1.0 }],
        regions: RegionMap {
            regions: vec![Region { name: "nasal-tip".into(), parent: None, side: humanart_model::Side::Center }],
            vertex_regions: vec![0; 4],
            soft: None,
        },
        landmarks: vec![landmark("L.zygion"), landmark("R.zygion")],
        skeleton: Skeleton {
            joints: vec![simple_joint("root", None, Vec3::ZERO)],
            regressor: JointRegressor::empty(deltas.len()),
        },
        skin: SkinWeights::from_lists(1, (0..4).map(|_| &[(0_u16, 1.0_f32)][..]), 0).unwrap(),
        expressions: ExpressionSet {
            expressions: vec![expression],
            combinations: Vec::new(),
            pose_correctives: Vec::new(),
        },
        lods: Vec::new(),
        mesh,
        deltas,
    };
    let template = Template::new(parts).unwrap();
    let coverage = small().check_template(&template);
    assert!(!coverage.passed());
    assert_eq!(coverage.unregistered_components, ["semantic.orphan"]);
    assert!(coverage.unregistered_expressions.is_empty());
    assert_eq!(coverage.unknown_targets, [("expression.au12_lip_corner_puller.r".to_owned(), "AU12-R".to_owned())]);
    assert_eq!(coverage.unknown_regions.len(), 2, "{coverage:?}"); // lip-corner-L and lip-corner-R
    assert!(coverage.unknown_landmarks.is_empty());
    assert_eq!(coverage.unchecked, 5);
}

#[test]
fn rob_01_registry_parser_survives_mutation_fuzz() {
    let corpus = vec![SMALL.as_bytes().to_vec(), with_entry("").into_bytes()];
    humanart_core::fuzz::run("registry", &corpus, 1500, 0x4e9, std::time::Duration::from_secs(120), |bytes| {
        if let Ok(params) = parse_file("fuzz", bytes, &RegistryLimits::DEFAULT) {
            for p in &params {
                assert!(p.check_value(&p.default).is_ok());
                assert_eq!(&param_from_description(&describe_param(p)).unwrap(), p);
            }
        }
    });
}
