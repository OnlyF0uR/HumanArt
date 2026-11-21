//! The skin appearance system, CPU side (WP-13): the resolved channel layout `[SKN-01]`, the layer
//! stack and dirty-rectangle tracking the GPU compositor consumes `[SKN-02]`, the physically-inspired
//! colour model `[SKN-10]` with region modulation, flush and wetness `[SKN-11]` `[SKN-31]`
//! `[SKN-32]`, the micro-detail generator's CPU reference `[SKN-20]` `[SKN-21]` `[SKN-22]` with
//! histogram-preserving anti-tiling and the statistics that test it `[AC-WP13-2]`, and the
//! procedural skin features `[OTH-02]`.
//!
//! The module is named `skin_appearance` because [`crate::skin`] already holds the skinning
//! weights. Nothing here touches the GPU; `humanart-render::skin` uploads what is built here and
//! runs the compositor, the micro-detail generator and the streaming integration.
//!
//! Module map:
//!
//! - [`pigment`]: the spectral pigment model and its GPU band table `[SKN-10]`.
//! - [`channels`]: attributes, planes and the resolved texture set with colour spaces `[SKN-01]`.
//! - [`dirty`]: dirty rectangles per UV tile `[SKN-02]`.
//! - [`layers`]: the layer stack, blend modes and the per-texel compositing reference `[SKN-02]`
//!   `[SKN-12]`.
//! - [`zones`]: region modulation, flush and wetness `[SKN-11]` `[SKN-31]` `[SKN-32]`.
//! - [`resolve`]: attributes to resolved channels, the reference of the GPU resolve.
//! - [`microdetail`]: the detail exemplar (pores, fine lines, relief, age), horizon map,
//!   micro-shadowing and vellus `[SKN-20]` `[SKN-21]` `[SKN-22]`.
//! - [`antitile`]: histogram-preserving tiling and blending (Heitz & Neyret 2018).
//! - [`spectrum`]: FFT, power spectra, autocorrelation and the tiling detector `[AC-WP13-2]`.
//! - [`features`]: the procedural skin features library `[OTH-02]`.
//! - [`noise`]: the PCG hash and periodic noise shared with the WGSL.
//! - [`regional`]: measured regional specular and sebum per face zone (todo 1.4).
//! - [`values`]: the document's global skin values and what they set (todo 1.4).
//! - [`wrinkle_author`]: the template's procedural expression wrinkle maps (todo 1.4).
//! - [`variation`]: low-frequency skin tone variation over the rest surface (todo 1.4).
//! - [`baked`]: baked thickness and occlusion to the compositor's input bytes (todo 1.4).
//! - [`error`]: the error enum, [`SkinError`].

pub mod antitile;
pub mod baked;
pub mod channels;
pub mod dirty;
pub mod error;
pub mod features;
pub mod layers;
pub mod microdetail;
pub mod noise;
pub mod pigment;
pub mod regional;
pub mod resolve;
mod spectra_table;
pub mod spectrum;
pub mod values;
pub mod variation;
pub mod wrinkle;
pub mod wrinkle_author;
pub mod zone_bake;
pub mod zones;

pub use error::SkinError;
pub use pigment::{Mapping, PigmentModel, PigmentParams, PigmentResponse};
