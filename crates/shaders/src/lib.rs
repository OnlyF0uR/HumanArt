//! Every WGSL shader, the shader manifest and the include preprocessor, with naga validation tests.
//!
//! - [`manifest`]: every file under `shaders/` embedded with `include_str!`, the shared libraries
//!   and the shaders with their entry points, stages, override constants and flag sets.
//! - [`preprocess`]: the in-house `#include` / `#ifdef` composition with a line map back to the
//!   source files.
//! - `validate` (native targets): naga validation of every shader for every flag set and override
//!   combination, and pipeline creation on a device `[AC-WP05-1]`.
//!
//! A renderer gets a module's text with [`preprocess::compose`] (once, at pipeline creation), hands it
//! to `humanart_gpu::cache::PipelineCache::shader`, and builds pipelines with the override values of
//! [`manifest::ShaderDesc::combination`] (`[PERF-R5]`, `[PERF-R4]`).
#![forbid(unsafe_code)]

pub mod error;
pub mod manifest;
pub mod preprocess;
#[cfg(not(target_arch = "wasm32"))]
pub mod validate;

pub use error::{Location, ShaderError};
pub use manifest::{ShaderDesc, Stage};
pub use preprocess::{Composed, compose};
