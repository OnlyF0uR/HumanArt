//! Scan-fit data the template generator consumes `[WP-25]` (todo 1.2): the result of fitting the
//! template onto a scan, and the face shape model averaged from several such fits.
//!
//! Pure CPU and wasm-safe: the fitting itself (scan import, registration, albedo transfer) lives in
//! `humanart_tools::capture`, which re-exports these types.
//!
//! | module | what |
//! |---|---|
//! | [`similarity`] | [`Similarity`]: rotation, uniform scale, translation |
//! | [`nearest`] | [`nearest::TriangleTree`]: closest points on a triangle mesh |
//! | [`fitted`] | [`fitted::FittedSurface`]: one fit in the template's frame, the `.hafit` file |
//! | [`face_model`] | [`face_model::FaceModel`]: mean face offset + shape components, the `.hafm` file |

use std::fmt;

pub mod face_model;
pub mod fitted;
pub mod nearest;
pub mod similarity;

pub use similarity::Similarity;

/// Why reading or using scan-fit data failed.
#[derive(Debug)]
pub enum ScanFitError {
    /// The input is malformed.
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
    /// The bytes could not be read (truncated input).
    Io(humanart_io::Error),
}

impl ScanFitError {
    /// A [`ScanFitError::Invalid`] error.
    #[must_use]
    pub const fn invalid(what: &'static str) -> Self {
        Self::Invalid { what }
    }

    /// A [`ScanFitError::Limit`] error.
    #[must_use]
    pub fn limit(what: &'static str, limit: usize, got: usize) -> Self {
        Self::Limit { what, limit: limit as u64, got: got as u64 }
    }
}

impl fmt::Display for ScanFitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { what } => write!(f, "scan fit: {what}"),
            Self::Limit { what, limit, got } => write!(f, "scan fit: {what} is {got}, the limit is {limit}"),
            Self::Io(e) => write!(f, "scan fit: {e}"),
        }
    }
}

impl std::error::Error for ScanFitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Invalid { .. } | Self::Limit { .. } => None,
            Self::Io(e) => Some(e),
        }
    }
}

impl From<humanart_io::Error> for ScanFitError {
    fn from(e: humanart_io::Error) -> Self {
        Self::Io(e)
    }
}

/// The result type of this module.
pub type Result<T> = std::result::Result<T, ScanFitError>;
