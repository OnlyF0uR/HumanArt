//! Tests of the document → expression rig glue on the generated template and the real registry.

use std::sync::{Arc, OnceLock};

use humanart_core::id::Id128;
use humanart_math::{Quat, Vec3};
use humanart_model::generate::build::{Options, template};

use super::*;
use crate::document::{Author, AuthorKind};
use crate::registry::tests::real_registry;

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
