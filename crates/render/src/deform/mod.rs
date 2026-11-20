//! GPU evaluation and normals (WP-10): the deformation pipeline of spec section 6.2 on the GPU,
//! matching the CPU reference `humanart_model::eval` `[EVAL-01]`.
//!
//! - [`tables`]: the load-time tables of a template (sparse f16 delta entries vertex-major and
//!   component-major `[EVAL-02]`, skin, normal gathers, strain topology) as flat SoA arrays.
//! - [`deformer`]: [`GpuDeformer`], the persistent buffers, cached pipelines and bind groups of one
//!   character, the per-frame diff of the evaluation input with incremental rest-shape updates
//!   `[EVAL-03]`, and the compute passes: rest accumulation, expressions + correctives + LBS/DQS
//!   skinning, normals and tangents `[EVAL-04]`, edge strain with its per-region reduction
//!   `[EVAL-06]`, previous positions for motion vectors, reduced levels of detail.
//! - [`error`] holds [`DeformError`], the error type.
//!
//! Shaders: `shaders/deform/*.wgsl` (manifest group `deform` in `humanart-shaders`).
//!
//! **Hooking it into the renderer.** Outputs are storage buffers in the scene's stream layout
//! (packed f32 xyz positions and normals, `vec4<f32>` tangents), so a renderer either binds
//! [`DeformOutputs::positions`], [`DeformOutputs::prev_positions`] and [`DeformOutputs::normals`]
//! in place of the scene streams of the character's mesh (vertex pulling, no copy `[EVAL-05]`) or
//! copies them into its streams at the mesh's base vertex. Per frame: `update` with the snapshot,
//! `encode` into the frame's encoder before the passes that read the positions, submit.

pub mod deformer;
pub mod error;
pub mod tables;

pub use deformer::{
    DeformOutputs, DeformSettings, GpuDeformer, JOB_SLOTS, LodOutputs, MAX_GPU_LAYERS, Pass, Stage, UpdatePlan,
};
pub use error::DeformError;
pub use tables::{DeformTables, LodTables};

#[cfg(test)]
pub(crate) mod tests;
