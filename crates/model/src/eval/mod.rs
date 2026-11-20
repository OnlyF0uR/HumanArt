//! The CPU reference evaluation (WP-09, `[EVAL-01]`): the deformation pipeline of section 6.2, the
//! rig logic, wrinkle drivers, edge strain, the landmark solver and the identity prior.
//!
//! It is used by the tests, the agent's headless metrics, offline tools and as the runtime fallback;
//! the GPU evaluation (WP-10) is compared against it.
//!
//! - [`error`]: [`EvalError`], the error type of the evaluation.
//! - [`layer`]: [`SculptLayer`], sparse f32 sculpt deltas (the `ΔLAYER` terms).
//! - [`skinning`]: joint posing from regressed bind positions, LBS and DQS `[SKIN-01]`.
//! - [`normals`]: the triangle and gather passes for normals and tangents `[EVAL-04]`.
//! - [`strain`]: edge strain and its per-region reduction `[EVAL-06]`.
//! - [`rig`]: [`RigLogic`], the data-driven control-to-expression graph `[EXP-03]`.
//! - [`wrinkle`]: [`WrinkleDrivers`], per-region fold and stretch activations `[EXP-04]`.
//! - [`landmarks`]: [`LandmarkSolver`], landmark drags with pins and locks `[LM-01]`..`[LM-03]`.
//! - [`prior`]: [`IdentityPrior`], the plausibility score `[ID-04]` and sampling `[ID-06]`.
//! - [`evaluator`]: [`Evaluator`], steps 1–5 with incremental rest-shape accumulation `[EVAL-03]`.
//!
//! Everything here is plain Rust on `f32` SoA buffers (solvers in `f64`), wasm-safe, and
//! allocation-free once set up.

pub mod error;
pub mod evaluator;
pub mod landmarks;
pub mod layer;
pub mod normals;
pub mod penetration;
pub mod prior;
pub mod rig;
pub mod skinning;
pub mod strain;
pub mod wrinkle;

pub use error::{EvalError, EvalResult};
pub use evaluator::{EvalInput, Evaluator};
pub use landmarks::{LandmarkGoal, LandmarkSolver, SolveReport, SolveRequest, SolveSettings};
pub use layer::SculptLayer;
pub use prior::{Covariance, GroupScore, GroupSpec, IdentityPrior, Plausibility, SampleRequest};
pub use rig::{RigLogic, RigOp};
pub use skinning::{JointFrames, SkinningMode};
pub use strain::StrainField;
pub use wrinkle::{WrinkleDriver, WrinkleDrivers};

#[cfg(test)]
mod tests;
