//! The HumanArt editor (WP-21 part 1, the application shell): the window content, its panels and
//! modes, preferences, provenance and agent presence, built on the engine of `humanart-agent` and the
//! UI toolkit of `humanart-ui`. The binary (`main.rs`) adds the window; everything here also runs
//! offscreen, which is how the tests and the shots drive it.
//!
//! - [`app`]: [`app::App`], the editor's state and frame, [`app::Intent`].
//! - [`shell`]: the window content `[APP-01]`: toolbar, dock layout, panels, overlays.
//! - [`studio`]: the GPU side: viewers per viewport and frame composition.
//! - [`modes`]: the editor modes `[APP-02]`.
//! - [`provenance`]: author badges, change flashes, "changed since I last looked" `[APP-03]`.
//! - [`agent_link`]: agent status, pause and revoke `[APP-04]`.
//! - [`prefs`]: preferences `[APP-05]`.
//! - [`window`]: winit events to toolkit input, the window surface's configuration.
//! - [`error`] holds [`error::AppError`].
#![forbid(unsafe_code)]

pub mod agent_link;
pub mod app;
pub mod error;
pub mod modes;
pub mod prefs;
pub mod provenance;
pub mod shell;
pub mod studio;
#[cfg(not(target_arch = "wasm32"))]
pub mod window;

#[cfg(test)]
mod tests;
