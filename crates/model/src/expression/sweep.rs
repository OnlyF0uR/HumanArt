//! Range-of-motion sweep metrics (WP-19, `[EXP-07]`, `[AC-WP19-1]`, `[AC-WP19-2]`).
//!
//! [`sweep`] drives every control of a [`ControlRig`] alone from 0 to 1 in `steps` steps through
//! the CPU reference evaluator and measures, against the neutral evaluation, per control:
//!
//! - **inverted triangles**: triangles whose normal turned more than 90° from its neutral
//!   direction (a fold), counted at the worst step;
//! - **displacement**: the largest vertex displacement;
//! - **area change**: the smallest and largest triangle area ratio (deformed / neutral) among the
//!   triangles that moved and have at least [`MIN_RATIO_AREA`] of neutral area (smaller ones are
//!   pinched triangles in the template's creases, whose ratio says nothing about the surface;
//!   they still count for inversions) and do not belong to an eye overlay part
//!   ([`AREA_EXEMPT_PREFIXES`]: tear lines and eye-occlusion lenses are rendering helpers that
//!   compress and stretch under closing lids by design; their inversions still count);
//! - **strain evidence**: the region with the largest mean edge compression and its value (the
//!   input of the wrinkle-map blending `[EXP-04]`, see [`crate::eval::strain`]).
//!
//! [`SweepLimits`] are the pass/fail bounds; their defaults are authored quality bounds (model
//! choices, not measurements of faces). The sweep evaluates both directions' samples once: the
//! controls are pure functions of their value, so 1 → 0 retraces 0 → 1 exactly.

use std::sync::Arc;

use humanart_math::{Quat, Vec3};

use super::controls::ControlRig;
use super::error::ExprResult;
use crate::eval::{EvalInput, Evaluator};
use crate::template::Template;

/// The smallest neutral triangle area (m², 0.1 mm²) whose area ratio the sweep reports.
pub const MIN_RATIO_AREA: f32 = 1.0e-7;

/// Mesh parts (by name prefix) left out of the area-ratio measure.
pub const AREA_EXEMPT_PREFIXES: [&str; 2] = ["tear-line", "eye-occlusion"];

/// Pass/fail bounds of the sweep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweepLimits {
    /// The most inverted triangles allowed.
    pub max_inverted: usize,
    /// The largest vertex displacement allowed, metres.
    pub max_displacement: f32,
    /// The smallest triangle area ratio allowed.
    pub min_area_ratio: f32,
    /// The largest triangle area ratio allowed.
    pub max_area_ratio: f32,
}

impl Default for SweepLimits {
    /// Authored bounds: no fold, at most 40 mm of motion (the open jaw's chin), triangles neither
    /// shrinking below a fifth nor growing beyond five times their area.
    fn default() -> Self {
        Self { max_inverted: 0, max_displacement: 0.04, min_area_ratio: 0.2, max_area_ratio: 5.0 }
    }
}

/// The metrics of one control.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlMetrics {
    /// The control name.
    pub control: String,
    /// The most inverted triangles at any step.
    pub inverted: usize,
    /// The largest vertex displacement, metres.
    pub max_displacement: f32,
    /// The smallest area ratio among moved triangles (1 when none moved).
    pub min_area_ratio: f32,
    /// The largest area ratio among moved triangles (1 when none moved).
    pub max_area_ratio: f32,
    /// The region with the largest mean compression at full weight, and that compression.
    pub most_compressed: (String, f32),
}

impl ControlMetrics {
    /// Whether the metrics are within `limits`.
    #[must_use]
    pub fn passes(&self, limits: &SweepLimits) -> bool {
        self.inverted <= limits.max_inverted
            && self.max_displacement <= limits.max_displacement
            && self.min_area_ratio >= limits.min_area_ratio
            && self.max_area_ratio <= limits.max_area_ratio
    }
}

fn normal_and_area(p: &[Vec3], t: &[u32]) -> (Vec3, f32) {
    let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
    let n = (b - a).cross(c - a);
    let len = n.length();
    (if len > 0.0 { n / len } else { Vec3::ZERO }, 0.5 * len)
}

/// Sweeps every control of `rig` alone over `steps` values in `(0, 1]` and returns its metrics.
pub fn sweep(template: &Arc<Template>, rig: &ControlRig, steps: usize) -> ExprResult<Vec<ControlMetrics>> {
    let mut ev = Evaluator::new(template.clone());
    let joints = template.skeleton().joints.len();
    let n_expr = template.expressions().expressions.len();
    let mut regs = vec![0.0; rig.register_count()];
    let mut weights = vec![0.0; n_expr];
    let mut pose = vec![Quat::IDENTITY; joints];
    let mut controls = vec![0.0; rig.controls().len()];
    ev.evaluate(&EvalInput::default())?;
    let neutral = ev.positions().to_vec();
    let tris = ev.triangles().to_vec();
    let rest: Vec<(Vec3, f32)> = tris.chunks(3).map(|t| normal_and_area(&neutral, t)).collect();
    let mesh = template.mesh();
    let exempt: Vec<bool> = mesh
        .triangulate()
        .triangle_faces
        .iter()
        .map(|&f| {
            mesh.part_of_face(f as usize)
                .and_then(|p| mesh.parts().get(p))
                .is_some_and(|p| AREA_EXEMPT_PREFIXES.iter().any(|x| p.name.starts_with(x)))
        })
        .collect();
    let region_names: Vec<String> = template.regions().regions.iter().map(|r| r.name.clone()).collect();
    let mut out = Vec::with_capacity(controls.len());
    for c in 0..controls.len() {
        let mut m = ControlMetrics {
            control: rig.controls().get(c).cloned().unwrap_or_default(),
            inverted: 0,
            max_displacement: 0.0,
            min_area_ratio: 1.0,
            max_area_ratio: 1.0,
            most_compressed: (String::new(), 0.0),
        };
        for k in 1..=steps.max(1) {
            controls.fill(0.0);
            controls[c] = k as f32 / steps.max(1) as f32;
            weights.fill(0.0);
            pose.fill(Quat::IDENTITY);
            rig.evaluate(&controls, &mut regs, &mut weights, &mut pose)?;
            ev.evaluate(&EvalInput { expressions: &weights, pose: &pose, ..EvalInput::default() })?;
            let pos = ev.positions();
            let moved: Vec<bool> = pos.iter().zip(&neutral).map(|(a, b)| (*a - *b).length() > 1e-6).collect();
            for (a, b) in pos.iter().zip(&neutral) {
                m.max_displacement = m.max_displacement.max((*a - *b).length());
            }
            let mut inverted = 0;
            for (i, (t, &(n0, a0))) in tris.chunks(3).zip(&rest).enumerate() {
                if !t.iter().any(|&v| moved[v as usize]) || a0 <= 1e-12 {
                    continue;
                }
                let (n1, a1) = normal_and_area(pos, t);
                if n1.dot(n0) < 0.0 {
                    inverted += 1;
                }
                if a0 >= MIN_RATIO_AREA && !exempt.get(i).copied().unwrap_or(false) {
                    let r = a1 / a0;
                    m.min_area_ratio = m.min_area_ratio.min(r);
                    m.max_area_ratio = m.max_area_ratio.max(r);
                }
            }
            m.inverted = m.inverted.max(inverted);
            if k == steps.max(1) {
                let comp = ev.strain().region_compression();
                // Leaf regions only: parents repeat their subtree's value.
                let best = comp
                    .iter()
                    .enumerate()
                    .filter(|(r, _)| !template.regions().has_children(*r as u16))
                    .max_by(|a, b| a.1.total_cmp(b.1));
                if let Some((r, &v)) = best {
                    m.most_compressed = (region_names.get(r).cloned().unwrap_or_default(), v);
                }
            }
        }
        out.push(m);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_decide() {
        let m = ControlMetrics {
            control: "x".into(),
            inverted: 0,
            max_displacement: 0.01,
            min_area_ratio: 0.5,
            max_area_ratio: 2.0,
            most_compressed: (String::new(), 0.0),
        };
        let l = SweepLimits::default();
        assert!(m.passes(&l));
        assert!(!ControlMetrics { inverted: 1, ..m.clone() }.passes(&l));
        assert!(!ControlMetrics { max_displacement: 0.05, ..m.clone() }.passes(&l));
        assert!(!ControlMetrics { min_area_ratio: 0.1, ..m.clone() }.passes(&l));
        assert!(!ControlMetrics { max_area_ratio: 6.0, ..m }.passes(&l));
        let p = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let (n, a) = normal_and_area(&p, &[0, 1, 2]);
        assert_eq!((n, a), (Vec3::Z, 0.5));
        assert_eq!(normal_and_area(&p, &[0, 0, 2]), (Vec3::ZERO, 0.0));
    }
}
