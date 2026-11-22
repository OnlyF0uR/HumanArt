//! The registry's plausibility prior `[ID-04]` (`todo.md` 2.3): how plausible a document's values
//! are, from the plausible spans and `plausibility_weight`s the registry carries.
//!
//! **Model.** Every numeric parameter with a positive weight is treated as an independent Gaussian
//! around its default whose plausible ("soft") span ends at two standard deviations on each side,
//! the convention the registry files state for standardised coefficients (usual span ±2 of ±3).
//! A value's deviation in standard deviations is [`deviation`]: `z = 2 (x − d) / (soft_max − d)`
//! above the default `d`, `z = 2 (d − x) / (d − soft_min)` below it; when the default sits on an end
//! of the span, the other side's half is used. The document's squared distance is
//! `d² = Σ w·z²` over every component of every weighted parameter, and the score is the chi-square
//! survival `P(χ²_k > d²)` with `k` the number of those components
//! ([`humanart_model::eval::prior::chi_square_survival`], the identity prior's score): 1 at the
//! defaults, the fraction of documents drawn from the prior that lie further out, towards 0 when
//! many values sit far outside their spans together. A score below [`WARNING_THRESHOLD`] is an
//! implausible combination; single values outside their span get their own warning from the
//! command resolver. Nothing is ever blocked.
//!
//! Parameters with weight 0 (expressions, materials, hair, editor values) and read-only
//! measurements do not take part. The registry does not describe correlations between parameters
//! (no "implausible combination" rules): the prior is diagonal until a fitted identity prior
//! (`humanart_model::eval::IdentityPrior`) replaces it for the shape components.
//!
//! [`RegistryPrior`] is the [`crate::session::Plausibility`] scorer every session uses unless
//! another one is installed, so command outcomes carry the score before and after `[CMD-06]`.

use humanart_model::eval::prior::chi_square_survival;

use crate::document::Document;
use crate::registry::{Param, ParamKind, ParamValue, Registry};
use crate::session::Plausibility;

/// A document scoring below this is an implausible combination (section 17: "overall score < 0.2
/// warns; never blocks"; the identity prior's threshold).
pub const WARNING_THRESHOLD: f64 = 0.2;

/// The deviation, in standard deviations, at the ends of the plausible span.
pub const SPAN_SIGMAS: f64 = 2.0;

/// The deviation of `x` from the parameter's default in standard deviations (see the module
/// documentation), signed: positive above the default. `None` when the parameter has no numeric
/// range or no span to measure against.
pub fn deviation(param: &Param, component: usize, x: f64) -> Option<f64> {
    let r = param.range?;
    let d = default_component(&param.default, component)?;
    let above = r.soft_max - d;
    let below = d - r.soft_min;
    let fallback = (r.soft_max - r.soft_min) / 2.0;
    let side = if x >= d { above } else { below };
    let span = if side > 0.0 { side } else { fallback };
    if !(span > 0.0 && x.is_finite()) {
        return None;
    }
    Some(SPAN_SIGMAS * (x - d) / span)
}

fn default_component(default: &ParamValue, component: usize) -> Option<f64> {
    match default {
        ParamValue::Scalar(v) => (component == 0).then_some(*v),
        ParamValue::Vector(vs) => vs.get(component).copied(),
        ParamValue::Color(c) => c.get(component).copied(),
        _ => None,
    }
}

/// `true` for the parameters that take part in the prior.
pub fn weighted(param: &Param) -> bool {
    param.plausibility_weight > 0.0
        && !param.read_only
        && param.range.is_some()
        && matches!(param.kind, ParamKind::Scalar | ParamKind::Vector | ParamKind::Color)
}

/// Calls `f(component, value)` for every numeric component of `value`.
fn components(value: &ParamValue, mut f: impl FnMut(usize, f64)) {
    match value {
        ParamValue::Scalar(v) => f(0, *v),
        ParamValue::Vector(vs) => vs.iter().enumerate().for_each(|(i, v)| f(i, *v)),
        ParamValue::Color(c) => c.iter().enumerate().for_each(|(i, v)| f(i, *v)),
        _ => {}
    }
}

/// The document's squared distance `d² = Σ w·z²` and the number of components `k`. Allocation-free.
pub fn distance(doc: &Document, registry: &Registry) -> (f64, usize) {
    let mut d2 = 0.0;
    let mut k = 0;
    for param in registry.params().iter().filter(|p| weighted(p)) {
        let value = doc.params.get(&param.id).unwrap_or(&param.default);
        components(&param.default, |_, _| k += 1);
        components(value, |i, x| {
            if let Some(z) = deviation(param, i, x) {
                d2 += param.plausibility_weight * z * z;
            }
        });
    }
    (d2, k)
}

/// The plausibility score of `doc`, 0..1, higher is more plausible (see the module documentation).
pub fn score(doc: &Document, registry: &Registry) -> f64 {
    let (d2, k) = distance(doc, registry);
    chi_square_survival(d2, k)
}

/// The parameters that contribute most to the distance, largest first: id and the weighted
/// deviation `√(w·Σz²)` in standard deviations. At most `n` entries; parameters at their defaults
/// are left out.
pub fn largest_deviations<'r>(doc: &Document, registry: &'r Registry, n: usize) -> Vec<(&'r str, f64)> {
    let mut out: Vec<(&str, f64)> = Vec::new();
    for param in registry.params().iter().filter(|p| weighted(p)) {
        let Some(value) = doc.params.get(&param.id) else { continue };
        let mut sum = 0.0;
        components(value, |i, x| {
            if let Some(z) = deviation(param, i, x) {
                sum += z * z;
            }
        });
        let z = (param.plausibility_weight * sum).sqrt();
        if z > 0.0 {
            out.push((param.id.as_str(), z));
        }
    }
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    out.truncate(n);
    out
}

/// The registry's diagonal prior as a session scorer (see the module documentation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegistryPrior;

impl Plausibility for RegistryPrior {
    fn score(&self, doc: &Document, registry: &Registry) -> f64 {
        score(doc, registry)
    }
}

#[cfg(test)]
mod tests;
