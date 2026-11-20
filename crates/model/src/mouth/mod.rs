//! Mouth interior (WP-16): teeth, gums and tongue parameters `[MTH-01]`..`[MTH-03]` and the
//! dynamic cavity occlusion `[MTH-04]`.
//!
//! - [`params`]: [`MouthParams`] — tooth proportions, enamel colour gradient, translucency, wear,
//!   staining and wetness; gum colour, scattering, wetness and gum line; tongue shape, coating,
//!   papillae and wetness.
//! - [`cavity`]: [`MouthOpening`] — the lip opening from the deformed landmarks and the form
//!   factor that darkens the interior.
//!
//! The shading lives in `shaders/mouth/mouth.wgsl` (render crate, `humanart_render::feature`).

pub mod cavity;
pub mod params;

pub use cavity::{LipLandmarks, MouthOpening};
pub use params::{GumParams, MouthParams, TeethParams, TongueParams};
