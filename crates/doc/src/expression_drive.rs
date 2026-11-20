//! The document's expression parameters driven through the template's expression control rig
//! (`todo.md` 1.8, `[EXP-01]` `[EXP-03]`).
//!
//! The registry's expression parameters (`expression.au12_lip_corner_puller.l`, ...) are the
//! controls of [`humanart_model::expression::controls::ControlRig`]. Binding a parameter straight to
//! its expression target would skip the rig: the jaw and tongue would not rotate (their targets are
//! only soft-tissue support shapes for a skeleton-driven motion), and the rig's combination logic
//! would not run. [`ExpressionDrive`] reads the document's values into the rig's controls, evaluates
//! the rig, and hands out the expression weights and the driven joints' rotations for the
//! evaluator's input (the engine's `Model::sync`, the CLI and the editor all go through it).
//!
//! Controls that have no registry parameter (the `viseme.*` controls) keep the value last given
//! with [`ExpressionDrive::set_control`], so presets can drive them.
//!
//! [`ExpressionDrive::evaluate`] does no heap allocation `[PERF-R1]`.

use std::fmt;

use humanart_math::Quat;
use humanart_model::Template;
use humanart_model::expression::controls::ControlRig;
use humanart_model::expression::error::ExprError;

use crate::document::Document;
use crate::registry::{ParamValue, Registry};

/// What can go wrong driving the expression rig.
#[derive(Debug)]
pub enum DriveError {
    /// The template's rig could not be built or evaluated.
    Rig(ExprError),
    /// The registry is not the one the drive was bound to.
    RegistryChanged {
        /// Parameters when bound.
        bound: usize,
        /// Parameters now.
        now: usize,
    },
}

impl fmt::Display for DriveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rig(e) => write!(f, "expression rig: {e}"),
            Self::RegistryChanged { bound, now } => {
                write!(f, "the registry has {now} parameters, the expression drive was bound to {bound}")
            }
        }
    }
}

impl std::error::Error for DriveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Rig(e) => Some(e),
            Self::RegistryChanged { .. } => None,
        }
    }
}

impl From<ExprError> for DriveError {
    fn from(e: ExprError) -> Self {
        Self::Rig(e)
    }
}

/// The document → control rig → evaluator glue (see the module documentation).
#[derive(Clone, Debug)]
pub struct ExpressionDrive {
    rig: ControlRig,
    /// `(registry index, control index)` of every parameter that is a rig control.
    bound: Vec<(u32, u32)>,
    /// Per registry parameter: whether the rig drives it.
    drives: Vec<bool>,
    controls: Vec<f32>,
    registers: Vec<f32>,
    expressions: Vec<f32>,
    pose: Vec<Quat>,
    /// The joints any driver rotates (sorted, distinct).
    joints: Vec<u16>,
}

fn scalar(value: &ParamValue) -> f32 {
    match value {
        #[allow(clippy::cast_possible_truncation)] // Expression weights are small coefficients.
        ParamValue::Scalar(v) if v.is_finite() => *v as f32,
        ParamValue::Bool(true) => 1.0,
        _ => 0.0,
    }
}

impl ExpressionDrive {
    /// Binds the registry's parameters to the controls of `template`'s own rig
    /// ([`ControlRig::builtin`]).
    ///
    /// # Errors
    /// [`DriveError::Rig`] when the rig data does not resolve against the template.
    pub fn new(template: &Template, registry: &Registry) -> Result<Self, DriveError> {
        Self::with_rig(template, registry, ControlRig::builtin(template)?)
    }

    /// Binds the registry's parameters to the controls of `rig` (built for `template`).
    ///
    /// # Errors
    /// [`DriveError::Rig`] when a count does not fit the rig's index types.
    pub fn with_rig(template: &Template, registry: &Registry, rig: ControlRig) -> Result<Self, DriveError> {
        let mut bound = Vec::new();
        let mut drives = vec![false; registry.len()];
        for (i, p) in registry.params().iter().enumerate() {
            if let Some(c) = rig.controls().iter().position(|c| *c == p.id)
                && let (Ok(i32_), Ok(c32)) = (u32::try_from(i), u32::try_from(c))
            {
                bound.push((i32_, c32));
                if let Some(d) = drives.get_mut(i) {
                    *d = true;
                }
            }
        }
        let mut joints: Vec<u16> = rig.drivers().iter().map(|d| d.joint).collect();
        joints.sort_unstable();
        joints.dedup();
        Ok(Self {
            controls: vec![0.0; rig.controls().len()],
            registers: vec![0.0; rig.register_count()],
            expressions: vec![0.0; template.expressions().expressions.len()],
            pose: vec![Quat::IDENTITY; template.skeleton().joints.len()],
            bound,
            drives,
            joints,
            rig,
        })
    }

    /// The rig.
    #[must_use]
    pub const fn rig(&self) -> &ControlRig {
        &self.rig
    }

    /// Whether registry parameter `index` is a rig control (its value reaches the evaluator only
    /// through the rig, never as a direct expression weight).
    #[must_use]
    pub fn drives_param(&self, index: usize) -> bool {
        self.drives.get(index).copied().unwrap_or(false)
    }

    /// How many registry parameters are rig controls.
    #[must_use]
    pub fn bound_params(&self) -> usize {
        self.bound.len()
    }

    /// Sets a control by name (for controls without a registry parameter, such as `viseme.*`);
    /// returns whether the rig has it. Non-finite values count as 0.
    pub fn set_control(&mut self, name: &str, value: f32) -> bool {
        let Some(c) = self.rig.controls().iter().position(|c| c == name) else { return false };
        if let Some(slot) = self.controls.get_mut(c) {
            *slot = if value.is_finite() { value } else { 0.0 };
        }
        true
    }

    /// The current control values, indexed like [`ControlRig::controls`].
    #[must_use]
    pub fn controls(&self) -> &[f32] {
        &self.controls
    }

    /// Reads the document's values of the bound parameters (registry defaults where the document
    /// has none) into the controls and evaluates the rig. Allocation-free.
    ///
    /// # Errors
    /// [`DriveError::RegistryChanged`] when `registry` is not the bound one; [`DriveError::Rig`]
    /// when the rig fails (its arrays are sized here, so only on an internal mismatch).
    pub fn evaluate(&mut self, doc: &Document, registry: &Registry) -> Result<(), DriveError> {
        if registry.len() != self.drives.len() {
            return Err(DriveError::RegistryChanged { bound: self.drives.len(), now: registry.len() });
        }
        let params = registry.params();
        for &(i, c) in &self.bound {
            let Some(p) = params.get(i as usize) else { continue };
            let v = scalar(doc.params.get(&p.id).unwrap_or(&p.default));
            if let Some(slot) = self.controls.get_mut(c as usize) {
                *slot = v;
            }
        }
        self.run()
    }

    /// Evaluates the rig from the current controls. Allocation-free.
    ///
    /// # Errors
    /// [`DriveError::Rig`] on an internal size mismatch.
    pub fn run(&mut self) -> Result<(), DriveError> {
        self.expressions.fill(0.0);
        for &j in &self.joints {
            if let Some(q) = self.pose.get_mut(j as usize) {
                *q = Quat::IDENTITY;
            }
        }
        self.rig.evaluate(&self.controls, &mut self.registers, &mut self.expressions, &mut self.pose)?;
        Ok(())
    }

    /// The expression weights of the last evaluation (one per template expression; expressions
    /// the rig does not output are 0).
    #[must_use]
    pub fn expressions(&self) -> &[f32] {
        &self.expressions
    }

    /// The joint rotations of the last evaluation (rest-relative, one per joint; identity for the
    /// joints no driver rotates).
    #[must_use]
    pub fn pose(&self) -> &[Quat] {
        &self.pose
    }

    /// The joints the rig rotates.
    #[must_use]
    pub fn driven_joints(&self) -> &[u16] {
        &self.joints
    }

    /// Adds the rig's expression weights to `expressions` and applies the driven joints'
    /// rotations after the ones in `pose` (`pose[j] = pose[j] · rig[j]`). Arrays of other lengths
    /// are written as far as they reach. Allocation-free.
    pub fn write(&self, expressions: &mut [f32], pose: &mut [Quat]) {
        for (out, w) in expressions.iter_mut().zip(&self.expressions) {
            *out += *w;
        }
        for &j in &self.joints {
            if let (Some(out), Some(q)) = (pose.get_mut(j as usize), self.pose.get(j as usize)) {
                *out = (*out * *q).normalize();
            }
        }
    }
}

#[cfg(test)]
mod tests;
