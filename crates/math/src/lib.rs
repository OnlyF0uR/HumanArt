//! Vectors, matrices, quaternions, f16, colour science, dense and sparse solvers, geometry and curves.
#![forbid(unsafe_code)]

#[cfg(feature = "color")]
pub mod color;
#[cfg(feature = "linalg")]
mod linalg;
#[cfg(feature = "solve")]
pub mod solve;

#[cfg(feature = "linalg")]
pub use linalg::*;
