//! Eyes (WP-16): parameters, optics, gaze and lid aperture of the eye model `[EYE-01]`..`[EYE-07]`.
//!
//! The eye is analytic in the shader: one eyeball mesh per eye (the template parts `eyeball-L`,
//! `eyeball-R`: a scleral sphere with the corneal cap) whose fragments are classified and shaded
//! from the eye's centre, gaze axis and the parameters here; the iris is the plane behind the
//! cornea, seen through the exact two-surface refraction (`shaders/eye/eye.wgsl`, mirrored by
//! [`optics::EyeGeometry`]).
//!
//! - [`params`]: [`EyeParams`] — eyeball variation `[EYE-07]`, iris layers `[EYE-02]`, sclera
//!   `[EYE-03]`, with sourced physical constants.
//! - [`optics`]: [`EyeGeometry`] — the analytic eyeball, the refraction to the iris plane and
//!   the paraxial entrance pupil (`[AC-WP16-2]`).
//! - [`gaze`]: [`Gaze`] — gaze angles, look-at targets, the eye joint rotation `[EYE-06]`.
//! - [`aperture`]: [`LidAperture`] — the lid opening for occlusion and lid shadow `[EYE-05]`.
//! - [`frame`]: [`EyeFrame`] — centre, radius and axis fitted to the deformed eyeball.
//!
//! Everything is plain Rust, wasm-safe; the per-frame paths (frame fit, aperture) allocate
//! nothing.

pub mod aperture;
pub mod frame;
pub mod gaze;
pub mod optics;
pub mod params;

pub use aperture::{APERTURE_SAMPLES, LidAperture, LidLandmarks};
pub use frame::{EyeFrame, EyeVertices, fit_sphere};
pub use gaze::Gaze;
pub use optics::{EyeGeometry, EyeHit};
pub use params::{EyeParams, IrisParams, ScleraParams};
