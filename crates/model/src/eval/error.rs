//! The error type of the CPU reference evaluation `[ERR-01]`.

use std::fmt;

use humanart_io::json::AccessError;
use humanart_math::solve::SolveError;

use crate::error::ModelError;

/// Why an evaluation, a solve or the set-up of an evaluator was refused.
///
/// Every input the evaluation reads from a caller (weights, rotations, targets, layers, data files)
/// is checked; nothing in [`crate::eval`] panics on bad input.
#[derive(Clone, Debug, PartialEq)]
pub enum EvalError {
    /// An input array is longer than the table it indexes ("expression weights: 12 > 10").
    Length {
        /// Which input.
        what: &'static str,
        /// The most entries it may have.
        limit: usize,
        /// The entries it has.
        got: usize,
    },
    /// An input value is NaN or infinite, or a rotation is not a usable quaternion.
    NonFinite {
        /// Which input.
        what: &'static str,
        /// The index of the offending entry.
        index: usize,
    },
    /// An input value is outside the range it must be in.
    Range {
        /// Which input.
        what: &'static str,
        /// The index of the offending entry.
        index: usize,
        /// The rule it breaks, as a short phrase.
        rule: &'static str,
    },
    /// A reference to something that does not exist (a landmark, a region, a name).
    Missing {
        /// What kind of thing was referenced.
        what: &'static str,
        /// The index of the reference in its input.
        index: usize,
    },
    /// A size or count exceeds a documented limit `[ROB-02]`.
    Limit {
        /// What was measured.
        what: &'static str,
        /// The limit in force.
        limit: u64,
        /// The value found.
        got: u64,
    },
    /// A data-model error from the template or a structure built for it.
    Model(ModelError),
    /// A JSON data file (rig logic, wrinkle drivers) has the wrong shape.
    Json(AccessError),
    /// A linear solve failed (a singular system, a covariance that is not positive definite).
    Solve(SolveError),
}

impl EvalError {
    /// Shorthand for [`EvalError::Limit`].
    #[must_use]
    pub const fn limit(what: &'static str, limit: usize, got: usize) -> Self {
        Self::Limit { what, limit: limit as u64, got: got as u64 }
    }
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length { what, limit, got } => write!(f, "{what}: {got} entries, at most {limit} allowed"),
            Self::NonFinite { what, index } => write!(f, "{what} {index}: not a finite value"),
            Self::Range { what, index, rule } => write!(f, "{what} {index}: {rule}"),
            Self::Missing { what, index } => write!(f, "reference {index}: no such {what}"),
            Self::Limit { what, limit, got } => write!(f, "{what} is {got}, over the limit of {limit}"),
            Self::Model(e) => write!(f, "model: {e}"),
            Self::Json(e) => write!(f, "data file: {e}"),
            Self::Solve(e) => write!(f, "solver: {e}"),
        }
    }
}

impl std::error::Error for EvalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::Solve(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ModelError> for EvalError {
    fn from(e: ModelError) -> Self {
        Self::Model(e)
    }
}

impl From<AccessError> for EvalError {
    fn from(e: AccessError) -> Self {
        Self::Json(e)
    }
}

impl From<SolveError> for EvalError {
    fn from(e: SolveError) -> Self {
        Self::Solve(e)
    }
}

/// The result type of [`crate::eval`].
pub type EvalResult<T> = std::result::Result<T, EvalError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn messages_and_sources() {
        let e = EvalError::Length { what: "expression weights", limit: 2, got: 3 };
        assert_eq!(e.to_string(), "expression weights: 3 entries, at most 2 allowed");
        assert!(e.source().is_none());
        let m: EvalError = ModelError::invalid("joint", 1, "bad").into();
        assert!(m.source().is_some());
        assert_eq!(m.to_string(), "model: joint 1: bad");
        assert_eq!(EvalError::limit("rows", 4, 5).to_string(), "rows is 5, over the limit of 4");
        assert_eq!(EvalError::NonFinite { what: "pose", index: 2 }.to_string(), "pose 2: not a finite value");
    }
}
