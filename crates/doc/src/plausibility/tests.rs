//! Tests of the registry plausibility prior and its warnings in command outcomes (`todo.md` 2.3).

use humanart_core::id::Id128;
use humanart_core::platform::{ManualClock, MemoryFs};

use super::*;
use crate::command::{Action, Command, Outcome, RangePolicy, WarningCode};
use crate::document::{Author, AuthorKind};
use crate::registry::tests::real_registry;
use crate::session::{Session, SessionConfig};

const HUMAN: &str = "human.artist";
const AGENT: &str = "agent.assistant";

fn fresh() -> Document {
    let human = Author { id: HUMAN.into(), kind: AuthorKind::Human, name: "Artist".into() };
    let mut doc = Document::new(Id128::from_u128(0x91a), "Prior", "humanart-template", 1, 0, human);
    doc.add_author(Author { id: AGENT.into(), kind: AuthorKind::Agent, name: "Assistant".into() });
    doc
}

fn set(author: &str, values: &[(&str, f64)]) -> Command {
    Command::new(
        author,
        Action::SetParams {
            values: values.iter().map(|(id, v)| ((*id).to_owned(), ParamValue::Scalar(*v))).collect(),
            relative: false,
            policy: RangePolicy::Reject,
        },
    )
}

fn implausible<'o>(o: &'o Outcome, target: &str) -> Option<&'o str> {
    o.warnings.iter().find(|w| w.code == WarningCode::Implausible && w.target == target).map(|w| w.message.as_str())
}

/// The weighted face shape parameters (an extreme set of many of them is an implausible face).
fn face_shape(reg: &Registry) -> Vec<&str> {
    reg.params()
        .iter()
        .filter(|p| p.id.starts_with("face.") && weighted(p) && p.range.is_some_and(|r| r.max >= 3.0))
        .map(|p| p.id.as_str())
        .collect()
}

#[test]
fn id_04_defaults_score_one_and_only_weighted_parameters_count() {
    let reg = real_registry();
    let doc = fresh();
    let (d2, k) = distance(&doc, &reg);
    assert!(d2.abs() < 1e-12, "the defaults lie at the centre: d2 {d2}");
    assert!((score(&doc, &reg) - 1.0).abs() < 1e-12);
    let expected: usize = reg
        .params()
        .iter()
        .filter(|p| weighted(p))
        .map(|p| match &p.default {
            ParamValue::Vector(v) => v.len(),
            ParamValue::Color(_) => 3,
            _ => 1,
        })
        .sum();
    assert_eq!(k, expected);
    assert!(k >= 60, "{k} weighted components");
    // Expressions, materials and measurements carry no weight.
    for id in ["expression.au12_lip_corner_puller.l", "skin.pigment.melanin", "measure.stature"] {
        assert!(!weighted(reg.require(id).unwrap()), "{id}");
    }
}

#[test]
fn id_04_the_ends_of_the_plausible_span_are_two_standard_deviations() {
    let reg = real_registry();
    let jaw = reg.require("face.jaw.width").unwrap();
    let r = jaw.range.unwrap();
    assert!((deviation(jaw, 0, r.soft_max).unwrap() - SPAN_SIGMAS).abs() < 1e-12);
    assert!((deviation(jaw, 0, r.soft_min).unwrap() + SPAN_SIGMAS).abs() < 1e-12);
    assert_eq!(deviation(jaw, 0, 0.0), Some(0.0));
    assert_eq!(deviation(jaw, 1, 0.0), None, "a scalar has one component");
    assert_eq!(deviation(jaw, 0, f64::NAN), None);
    // A default on the end of its span: the other side's half measures the outside.
    let mut edge = jaw.clone();
    edge.default = ParamValue::Scalar(r.soft_max);
    let half = (r.soft_max - r.soft_min) / 2.0;
    assert!((deviation(&edge, 0, r.soft_max + half).unwrap() - SPAN_SIGMAS).abs() < 1e-12);
    assert!((deviation(&edge, 0, r.soft_min).unwrap() + SPAN_SIGMAS).abs() < 1e-12);
    // One value at its hard maximum barely moves a prior of many dimensions.
    let mut doc = fresh();
    doc.set_param(&reg, "face.jaw.width", ParamValue::Scalar(r.max), HUMAN).unwrap();
    assert!(score(&doc, &reg) > 0.9, "{}", score(&doc, &reg));
    assert_eq!(largest_deviations(&doc, &reg, 5), vec![("face.jaw.width", 3.0)]);
}

#[test]
fn cmd_06_every_session_reports_plausibility_and_single_implausible_values() {
    let reg = real_registry();
    let (fs, clock) = (MemoryFs::new(), ManualClock::new(1_763_800_000_000));
    let mut s = Session::new(fresh(), &reg, &fs, &clock, SessionConfig::DEFAULT, Id128::from_u128(5));
    // Inside the plausible span: scored, no warning.
    let o = s.execute(set(HUMAN, &[("face.jaw.width", 1.0)])).unwrap();
    let [before, after] = o.plausibility.expect("the default scorer is installed");
    // One value one standard deviation out in a prior of many dimensions: still fully plausible.
    assert!((before - 1.0).abs() < 1e-12 && after <= before && after > 0.9, "{before} {after}");
    assert!(o.warnings.is_empty(), "{:?}", o.warnings);
    // Outside the span (inside the range): the value is named, the document is still plausible.
    let o = s.execute(set(HUMAN, &[("face.jaw.width", 2.5)])).unwrap();
    assert!(implausible(&o, "face.jaw.width").is_some_and(|m| m.contains("plausible span")), "{:?}", o.warnings);
    assert!(implausible(&o, "document").is_none());
    let json = humanart_io::json::to_string(&o.to_json()).unwrap();
    assert!(json.contains("\"plausibility\":[") && json.contains("\"implausible\""), "{json}");
}

#[test]
fn id_04_an_implausible_combination_is_reported_with_the_values_furthest_out() {
    let reg = real_registry();
    let ids = face_shape(&reg);
    assert!(ids.len() >= 30, "{} weighted face parameters", ids.len());
    let (fs, clock) = (MemoryFs::new(), ManualClock::new(1_763_800_000_000));
    let mut s = Session::new(fresh(), &reg, &fs, &clock, SessionConfig::DEFAULT, Id128::from_u128(6));
    // Every face value at the end of its plausible span at once: each one is plausible alone.
    let ends: Vec<(&str, f64)> = ids.iter().map(|id| (*id, reg.require(id).unwrap().range.unwrap().soft_max)).collect();
    let o = s.execute(set(HUMAN, &ends)).unwrap();
    let [_, after] = o.plausibility.unwrap();
    assert!(after < WARNING_THRESHOLD, "score {after}");
    let message = implausible(&o, "document").expect("the combination is reported");
    assert!(message.contains("implausible combination") && message.contains("furthest out:"), "{message}");
    assert!(o.warnings.iter().all(|w| w.target == "document"), "no single value is outside: {:?}", o.warnings);
    // Bringing one value back raises the score: no new combination warning.
    let o = s.execute(set(HUMAN, &[(ids[0], 0.0)])).unwrap();
    let [b, a] = o.plausibility.unwrap();
    assert!(a > b, "{b} -> {a}");
    assert!(implausible(&o, "document").is_none());
    // The same in a proposal of the agent.
    let mut s = Session::new(fresh(), &reg, &fs, &clock, SessionConfig::DEFAULT, Id128::from_u128(7));
    let p = s.open_proposal(AGENT, "wild face").unwrap();
    let o = s.execute_in(p, set(AGENT, &ends)).unwrap();
    assert!(o.plausibility.is_some_and(|[_, a]| a < WARNING_THRESHOLD));
    assert!(implausible(&o, "document").is_some(), "{:?}", o.warnings);
}

#[test]
fn id_04_scoring_does_not_depend_on_unweighted_values() {
    let reg = real_registry();
    let mut doc = fresh();
    doc.set_param(&reg, "expression.au26_jaw_drop", ParamValue::Scalar(1.0), HUMAN).unwrap();
    doc.set_param(&reg, "skin.pigment.melanin", ParamValue::Scalar(1.0), HUMAN).unwrap();
    assert!((score(&doc, &reg) - 1.0).abs() < 1e-12);
    assert!(largest_deviations(&doc, &reg, 3).is_empty());
    assert!((RegistryPrior.score(&doc, &reg) - 1.0).abs() < 1e-12);
}
