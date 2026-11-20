//! Hair rendering `[HR-01]`..`[HR-05]`: the fibre scattering model and its tables ([`bsdf`],
//! [`luts`]), the constant blocks and per-view choices of the passes ([`view`]: deep opacity
//! frames, level of detail) and the passes themselves ([`renderer`]: [`HairRenderer`]).

pub mod bsdf;
pub mod error;
pub mod luts;
pub mod renderer;
pub mod view;

pub use error::HairError;
pub use renderer::{HairFrame, HairGroom, HairPath, HairRenderer, HairSettings, HairTargets};
