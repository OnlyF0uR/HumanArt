//! Tests of the document → expression rig glue on the generated template and the real registry.

use std::sync::{Arc, OnceLock};

use humanart_core::id::Id128;
use humanart_math::{Quat, Vec3};
use humanart_model::generate::build::{Options, template};

use super::*;
use crate::document::{Author, AuthorKind};
use crate::registry::tests::real_registry;
use crate::text::TextLimits;

fn generated() -> Arc<Template> {
    static T: OnceLock<Arc<Template>> = OnceLock::new();
    T.get_or_init(|| Arc::new(template(&Options { lods: false, ..Options::default() }).unwrap().template)).clone()
}

fn doc() -> Document {
    let author = Author { id: "human.test".into(), kind: AuthorKind::Human, name: "Test".into() };
    Document::new(Id128::from_u128(7), "drive", "humanart-template", 1, 0, author)
}

fn expression(t: &Template, name: &str) -> usize {
    t.expressions().expressions.iter().position(|e| e.name == name).unwrap()
}

fn joint(t: &Template, name: &str) -> usize {
    t.skeleton().joints.iter().position(|j| j.name == name).unwrap()
}

#[test]
fn exp_03_every_expression_parameter_is_a_rig_control() {
    let (t, reg) = (generated(), real_registry());
    let drive = ExpressionDrive::new(&t, &reg).unwrap();
    let expression_params: Vec<usize> =
        (0..reg.len()).filter(|&i| reg.params()[i].id.starts_with("expression.")).collect();
    assert!(expression_params.len() >= 80, "{} expression parameters", expression_params.len());
    for &i in &expression_params {
        assert!(drive.drives_param(i), "{} is not a rig control", reg.params()[i].id);
    }
    assert_eq!(drive.bound_params(), (0..reg.len()).filter(|&i| drive.drives_param(i)).count());
    assert!(!drive.drives_param(usize::MAX));
    let joints = drive.driven_joints();
    assert!(joints.windows(2).all(|w| w[0] < w[1]), "sorted and distinct: {joints:?}");
    assert!(joints.contains(&u16::try_from(joint(&t, "jaw")).unwrap()));
}

#[test]
fn exp_03_document_values_reach_weights_and_the_jaw() {
    let (t, reg) = (generated(), real_registry());
    let mut drive = ExpressionDrive::new(&t, &reg).unwrap();
    let mut d = doc();
    drive.evaluate(&d, &reg).unwrap();
    assert!(drive.expressions().iter().all(|&w| w == 0.0), "the neutral document drives nothing");
    assert!(drive.pose().iter().all(|q| *q == Quat::IDENTITY));

    d.set_param(&reg, "expression.au12_lip_corner_puller.l", ParamValue::Scalar(1.0), "human.test").unwrap();
    drive.evaluate(&d, &reg).unwrap();
    let au12l = expression(&t, "AU12-L");
    assert!(drive.expressions()[au12l] > 0.99, "AU12-L {}", drive.expressions()[au12l]);
    assert!(drive.expressions()[expression(&t, "AU12-R")].abs() < 1e-6);

    // The jaw drop is skeleton-driven: the jaw joint rotates (opening: the chin goes down) and
    // the support shape is weighted.
    d.set_param(&reg, "expression.au12_lip_corner_puller.l", ParamValue::Scalar(0.0), "human.test").unwrap();
    d.set_param(&reg, "expression.au26_jaw_drop", ParamValue::Scalar(1.0), "human.test").unwrap();
    drive.evaluate(&d, &reg).unwrap();
    let jaw = joint(&t, "jaw");
    let q = drive.pose()[jaw];
    assert!(q != Quat::IDENTITY, "the jaw rotates");
    assert!(drive.expressions()[expression(&t, "AU26")] > 0.99);
    assert!(drive.expressions()[au12l].abs() < 1e-6, "a value set back to 0 is gone");
    // Rest-relative rotation of the jaw about world X: a point in front of the joint goes down.
    let rest = t.skeleton().rest_world();
    let r = rest[jaw].rotation;
    let world = r * q * r.conjugate();
    let moved = world * Vec3::new(0.0, 0.0, 0.1);
    assert!(moved.y < -0.01, "{moved:?}");

    // `write` adds the weights and composes the rotation after the given pose.
    let mut w = vec![0.25; t.expressions().expressions.len()];
    let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
    drive.write(&mut w, &mut pose);
    assert!((w[expression(&t, "AU26")] - 1.25).abs() < 1e-5);
    assert!((w[au12l] - 0.25).abs() < 1e-6);
    assert!((pose[jaw].dot(q)).abs() > 0.9999);
    drive.write(&mut [], &mut []);
}

#[test]
fn exp_03_controls_without_parameters_and_errors() {
    let (t, reg) = (generated(), real_registry());
    let mut drive = ExpressionDrive::new(&t, &reg).unwrap();
    assert!(drive.set_control("viseme.jaw-open-wide", 1.0));
    assert!(!drive.set_control("viseme.no-such-control", 1.0));
    drive.run().unwrap();
    assert!(drive.pose()[joint(&t, "jaw")] != Quat::IDENTITY, "a viseme control opens the jaw");
    let c = drive.rig().controls().iter().position(|c| c == "viseme.jaw-open-wide").unwrap();
    assert!(drive.set_control("viseme.jaw-open-wide", f32::NAN));
    assert!(drive.controls()[c] == 0.0, "non-finite values count as 0");
    drive.run().unwrap();
    assert!(drive.pose().iter().all(|q| *q == Quat::IDENTITY));

    // A registry of another size is refused.
    let small = Registry::new(reg.params()[..reg.len() - 1].to_vec()).unwrap();
    let e = drive.evaluate(&doc(), &small).unwrap_err();
    assert!(matches!(e, DriveError::RegistryChanged { .. }), "{e}");
    assert!(std::error::Error::source(&e).is_none());
    let rig = DriveError::from(ExprError::missing("landmark", "x"));
    assert!(rig.to_string().contains("expression rig"));
    assert!(std::error::Error::source(&rig).is_some());
}

#[test]
fn exp_03_a_saved_and_loaded_document_drives_the_same_face() {
    let (t, reg) = (generated(), real_registry());
    let mut d = doc();
    for (id, v) in [
        ("expression.au06_cheek_raiser.l", 0.5),
        ("expression.au06_cheek_raiser.r", 0.5),
        ("expression.au12_lip_corner_puller.l", 0.8),
        ("expression.au12_lip_corner_puller.r", 0.8),
        ("expression.au25_lips_part", 0.6),
        ("expression.au26_jaw_drop", 0.3),
    ] {
        d.set_param(&reg, id, ParamValue::Scalar(v), "human.test").unwrap();
    }
    let mut before = ExpressionDrive::new(&t, &reg).unwrap();
    before.evaluate(&d, &reg).unwrap();
    let text = d.write().unwrap();
    let back = Document::parse(text.as_bytes(), &reg, &TextLimits::DEFAULT).unwrap();
    let mut after = ExpressionDrive::new(&t, &reg).unwrap();
    after.evaluate(&back, &reg).unwrap();
    assert_eq!(before.controls(), after.controls());
    assert_eq!(before.expressions(), after.expressions());
    assert_eq!(before.pose(), after.pose());
    assert!(after.expressions()[expression(&t, "AU12-L")] > 0.79);
    assert!(after.pose()[joint(&t, "jaw")] != Quat::IDENTITY);
}

#[test]
fn rob_01_bad_document_values_do_not_reach_the_face() {
    let (t, reg) = (generated(), real_registry());
    let mut drive = ExpressionDrive::new(&t, &reg).unwrap();
    let au12l = expression(&t, "AU12-L");
    let mut d = doc();
    // Values the command layer would refuse, written straight into the map: a non-finite number,
    // a value far out of range and a value of the wrong kind.
    d.params.insert("expression.au12_lip_corner_puller.l".into(), ParamValue::Scalar(f64::NAN));
    d.params.insert("expression.au26_jaw_drop".into(), ParamValue::Scalar(f64::INFINITY));
    d.params.insert("expression.au25_lips_part".into(), ParamValue::Enum("wide".into()));
    drive.evaluate(&d, &reg).unwrap();
    assert!(drive.expressions().iter().all(|w| w.is_finite()));
    assert!(drive.expressions()[au12l] == 0.0, "NaN counts as 0");
    assert!(drive.expressions()[expression(&t, "AU25")] == 0.0, "a wrong kind counts as 0");
    assert!(drive.pose().iter().all(|q| q.x.is_finite() && q.y.is_finite() && q.z.is_finite() && q.w.is_finite()));
    d.params.insert("expression.au12_lip_corner_puller.l".into(), ParamValue::Scalar(1.0e30));
    drive.evaluate(&d, &reg).unwrap();
    let w = drive.expressions()[au12l];
    assert!(w.is_finite() && w <= 1.0 + 1e-6, "out-of-range values are clamped by the rig: {w}");
    // The command layer refuses them before they reach the document.
    let mut ok = doc();
    assert!(
        ok.set_param(&reg, "expression.au12_lip_corner_puller.l", ParamValue::Scalar(f64::NAN), "human.test").is_err()
    );
    assert!(ok.set_param(&reg, "expression.no_such_unit", ParamValue::Scalar(0.5), "human.test").is_err());
}

#[test]
fn perf_r1_evaluate_does_not_allocate() {
    let (t, reg) = (generated(), real_registry());
    let mut drive = ExpressionDrive::new(&t, &reg).unwrap();
    let mut d = doc();
    d.set_param(&reg, "expression.au26_jaw_drop", ParamValue::Scalar(0.5), "human.test").unwrap();
    drive.evaluate(&d, &reg).unwrap();
    let mut w = vec![0.0; t.expressions().expressions.len()];
    let mut pose = vec![Quat::IDENTITY; t.skeleton().joints.len()];
    let counts = humanart_core::alloc::count_in(|| {
        let _ = drive.evaluate(&d, &reg);
        drive.write(&mut w, &mut pose);
    });
    assert_eq!(counts.allocations, 0, "[PERF-R1] the expression drive does not allocate");
}
