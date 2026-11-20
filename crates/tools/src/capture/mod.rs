//! Scan import and registration (WP-25, `todo.md` section 3): bringing a captured scan into the
//! template's conventions and fitting the template surface onto it.
//!
//! | module | contents |
//! |---|---|
//! | [`scan`] | [`scan::ScanMesh`]: an OBJ scan welded into a triangle mesh, normalised to metres, Y up, facing +Z |
//! | [`albedo`] | [`albedo::Albedo`]: a scan's colour texture in linear Rec.709, box-filtered to a working size |
//! | [`landmarks`] | [`landmarks::LandmarkSet`]: named scan points with the template's landmark ids, JSON annotation files |
//! | [`nearest`] | [`nearest::TriangleTree`]: closest points on a mesh, the correspondence search |
//! | [`align`] | [`align::fit_similarity`]: rigid + scale alignment from landmark pairs |
//! | [`register`] | [`register::register`]: landmark-guided non-rigid ICP, coarse to fine, masked |
//! | [`fitted`] | [`fitted::FittedSurface`]: a fit in the template's frame, the `.hafit` file, the displacement field |
//! | [`transfer`] | [`transfer::transfer_and_clean`]: scan albedo into the template's UV tiles, occlusion and specks removed |
//! | [`template_fit`] | [`template_fit::TemplateFit`]: the template's selected faces welded into a fit surface, node weights, landmarks |
//!
//! Everything here is offline tooling: it reads bytes the caller loaded (no files, no clocks) and
//! returns [`CaptureError`] for anything malformed. Scans are untrusted input: every size is checked
//! against the documented limits before anything is allocated.

pub mod albedo;
pub mod align;
pub mod face_model;
pub mod fitted;
pub mod landmarks;
pub mod nearest;
pub mod register;
pub mod scan;
pub mod template_fit;
pub mod transfer;

use std::fmt;

use humanart_math::solve::SolveError;
use humanart_model::ModelError;
use humanart_model::scanfit::ScanFitError;

/// Why a capture operation failed.
#[derive(Debug)]
pub enum CaptureError {
    /// The input or the request is malformed.
    Invalid {
        /// What is wrong.
        what: &'static str,
    },
    /// A size is beyond its documented limit.
    Limit {
        /// What was counted.
        what: &'static str,
        /// The limit.
        limit: u64,
        /// The value found.
        got: u64,
    },
    /// A landmark annotation file is malformed (the message names the place).
    Annotation(String),
    /// A file could not be decoded.
    Io(humanart_io::Error),
    /// The template refused the data.
    Model(ModelError),
    /// A linear solve failed.
    Solve(SolveError),
    /// A ray hierarchy or bake step failed.
    Bake(crate::bake::BakeError),
    /// Fit data (closest-point tree, fit file, face model) is malformed.
    ScanFit(ScanFitError),
}

impl CaptureError {
    /// A [`CaptureError::Invalid`] error.
    pub const fn invalid(what: &'static str) -> Self {
        Self::Invalid { what }
    }

    /// A [`CaptureError::Limit`] error.
    pub fn limit(what: &'static str, limit: usize, got: usize) -> Self {
        Self::Limit { what, limit: limit as u64, got: got as u64 }
    }
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { what } => write!(f, "capture: {what}"),
            Self::Limit { what, limit, got } => write!(f, "capture: {what} is {got}, the limit is {limit}"),
            Self::Annotation(what) => write!(f, "capture: landmark annotation: {what}"),
            Self::Io(e) => write!(f, "capture: {e}"),
            Self::Model(e) => write!(f, "capture: {e}"),
            Self::Solve(e) => write!(f, "capture: {e}"),
            Self::Bake(e) => write!(f, "capture: {e}"),
            Self::ScanFit(e) => write!(f, "capture: {e}"),
        }
    }
}

impl std::error::Error for CaptureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Invalid { .. } | Self::Limit { .. } | Self::Annotation(_) => None,
            Self::Io(e) => Some(e),
            Self::Model(e) => Some(e),
            Self::Solve(e) => Some(e),
            Self::Bake(e) => Some(e),
            Self::ScanFit(e) => Some(e),
        }
    }
}

impl From<humanart_io::Error> for CaptureError {
    fn from(e: humanart_io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<ModelError> for CaptureError {
    fn from(e: ModelError) -> Self {
        Self::Model(e)
    }
}

impl From<ScanFitError> for CaptureError {
    fn from(e: ScanFitError) -> Self {
        Self::ScanFit(e)
    }
}

impl From<SolveError> for CaptureError {
    fn from(e: SolveError) -> Self {
        Self::Solve(e)
    }
}

/// The result type of this module.
pub type Result<T> = std::result::Result<T, CaptureError>;
