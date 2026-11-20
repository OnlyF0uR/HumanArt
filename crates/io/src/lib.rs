//! In-house codecs and containers: JSON, PNG, JPEG, Radiance HDR, EXR, KTX2, glTF/GLB, OBJ, PLY,
//! TrueType, compression.
//!
//! Everything here works on byte slices and returns values or [`Error`]: no files, no threads, no
//! clocks, so the crate compiles for wasm `[PLAT-03]`. Every decoder takes [`Limits`] and checks each
//! size it reads before allocating, and each has a mutation-fuzz test `[ROB-01]`. Each codec family is
//! a cargo feature, all enabled by default (ADR-007).
#![forbid(unsafe_code)]

pub mod bytes;
pub mod error;
pub mod image;

#[cfg(feature = "bc")]
pub mod bc;
#[cfg(feature = "deflate")]
pub mod deflate;
#[cfg(feature = "exr")]
pub mod exr;
#[cfg(feature = "gltf")]
pub mod gltf;
#[cfg(feature = "hdr")]
pub mod hdr;
#[cfg(feature = "jpeg")]
pub mod jpeg;
#[cfg(feature = "json")]
pub mod json;
#[cfg(feature = "ktx2")]
pub mod ktx2;
#[cfg(feature = "lz")]
pub mod lz;
#[cfg(feature = "mip")]
pub mod mip;
#[cfg(feature = "obj")]
pub mod obj;
#[cfg(feature = "ply")]
pub mod ply;
#[cfg(feature = "png")]
pub mod png;
#[cfg(feature = "ttf")]
pub mod ttf;

pub use error::{Error, Format, Result};
pub use image::{ColorSpace, Image, Limits, Samples};
