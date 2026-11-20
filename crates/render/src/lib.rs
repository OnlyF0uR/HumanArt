//! The frame graph and the render passes: skin, eyes, hair, teeth, lighting, post, debug views and stills.
#![forbid(unsafe_code)]

pub mod camera;
#[cfg(not(target_arch = "wasm32"))]
pub mod capture;
pub mod compress;
pub mod deform;
pub mod error;
pub mod eye;
pub mod graph;
pub mod groom;
pub mod hair;
pub mod ibl;
pub mod light;
pub mod mouth;
pub mod renderer;
pub mod scene;
pub mod skin;
pub mod sss;
pub mod still;
pub mod surface;
pub mod uniforms;

pub use error::{GraphError, RenderError};

#[cfg(test)]
mod tests;
