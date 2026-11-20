//! The registry-against-template check of `[REG-02]`: every parameter reaches something the
//! template has, and everything the template exposes as an input has a parameter.
//!
//! Targets in the template (delta components, expressions, joints, regions, landmarks) are checked
//! by name. Material, groom and editor targets live in shaders and generators that the template
//! does not describe; they are counted in [`Coverage::unchecked`] and checked by the crates that
//! consume them.

use std::collections::BTreeSet;

use humanart_model::Template;
use humanart_model::delta::DeltaKind;

use super::{Evaluation, Registry};

/// The result of [`Registry::check_template`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Coverage {
    /// `(parameter id, target)` for component, expression and rig targets the template lacks.
    pub unknown_targets: Vec<(String, String)>,
    /// `(parameter id, region)` for regions the template lacks.
    pub unknown_regions: Vec<(String, String)>,
    /// `(parameter id, landmark)` for measurement landmarks the template lacks.
    pub unknown_landmarks: Vec<(String, String)>,
    /// Identity, semantic and asymmetry delta components of the template with no parameter.
    pub unregistered_components: Vec<String>,
    /// Expression targets of the template with no parameter.
    pub unregistered_expressions: Vec<String>,
    /// Parameters whose targets the template cannot confirm (material, groom, editor).
    pub unchecked: usize,
}

impl Coverage {
    /// `true` when every list is empty.
    pub fn passed(&self) -> bool {
        self.unknown_targets.is_empty()
            && self.unknown_regions.is_empty()
            && self.unknown_landmarks.is_empty()
            && self.unregistered_components.is_empty()
            && self.unregistered_expressions.is_empty()
    }
}

pub(super) fn check(registry: &Registry, template: &Template) -> Coverage {
    let components: BTreeSet<&str> = template.deltas().components().iter().map(|c| c.name.as_str()).collect();
    let expressions: BTreeSet<&str> = template.expressions().expressions.iter().map(|e| e.name.as_str()).collect();
    let joints: BTreeSet<&str> = template.skeleton().joints.iter().map(|j| j.name.as_str()).collect();
    let regions: BTreeSet<&str> = template.regions().regions.iter().map(|r| r.name.as_str()).collect();
    let landmarks: BTreeSet<&str> = template.landmarks().iter().map(|l| l.name.as_str()).collect();
    let mut report = Coverage::default();
    let mut used_components = BTreeSet::new();
    let mut used_expressions = BTreeSet::new();
    for p in registry.params() {
        if let Some(region) = &p.region
            && !regions.contains(region.as_str())
        {
            report.unknown_regions.push((p.id.clone(), region.clone()));
        }
        let (found, target) = match &p.evaluation {
            Evaluation::Component(t) => {
                used_components.insert(t.as_str());
                (components.contains(t.as_str()), t)
            }
            Evaluation::Expression(t) => {
                used_expressions.insert(t.as_str());
                (expressions.contains(t.as_str()), t)
            }
            Evaluation::Rig(t) => (joints.contains(t.as_str()), t),
            Evaluation::Measurement { landmarks: pair, .. } => {
                for l in pair {
                    if !landmarks.contains(l.as_str()) {
                        report.unknown_landmarks.push((p.id.clone(), l.clone()));
                    }
                }
                continue;
            }
            Evaluation::Material(_) | Evaluation::Groom(_) | Evaluation::Editor(_) => {
                report.unchecked += 1;
                continue;
            }
        };
        if !found {
            report.unknown_targets.push((p.id.clone(), target.clone()));
        }
    }
    for c in template.deltas().components() {
        let input = matches!(
            c.kind,
            DeltaKind::IdentityGlobal | DeltaKind::IdentityRegional | DeltaKind::Semantic | DeltaKind::Asymmetry
        );
        if input && !used_components.contains(c.name.as_str()) {
            report.unregistered_components.push(c.name.clone());
        }
    }
    for e in &template.expressions().expressions {
        if !used_expressions.contains(e.name.as_str()) {
            report.unregistered_expressions.push(e.name.clone());
        }
    }
    report
}
