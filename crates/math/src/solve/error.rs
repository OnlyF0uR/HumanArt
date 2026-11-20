//! The error enum of the solvers `[ERR-01]`.

use std::fmt;

/// Why a solver refused its input or failed to produce a result.
///
/// No routine of [`crate::solve`] panics on bad dimensions, non-finite input or a matrix that does not
/// have the property the method needs: it returns one of these.
#[derive(Clone, Debug, PartialEq)]
pub enum SolveError {
    /// A Cholesky-type factorisation met a pivot that is not strictly positive: the matrix is not
    /// (numerically) symmetric positive definite. `pivot` is the zero-based row of the failure.
    NotPositiveDefinite {
        /// The row at which the factorisation broke down.
        pivot: usize,
    },
    /// The matrix is singular (or rank deficient) to working precision and the operation needs it not
    /// to be.
    Singular,
    /// Two sizes that have to agree do not.
    DimensionMismatch {
        /// Which quantity was checked (a short static description, for example `"rhs length"`).
        what: &'static str,
        /// The size the routine needed.
        expected: usize,
        /// The size it was given.
        found: usize,
    },
    /// An iterative method reached its iteration cap before meeting its tolerance.
    NotConverged {
        /// Iterations (or sweeps) performed.
        iterations: usize,
        /// The method's residual measure at the point it gave up.
        residual: f64,
    },
    /// The input, or an intermediate result, contains a NaN or an infinity.
    NonFinite,
    /// An iterative method met a quantity that must be positive and is not (for conjugate gradients:
    /// the curvature `pᵀAp` or the preconditioned residual product), which means the operator or the
    /// preconditioner is not symmetric positive definite.
    Breakdown {
        /// The iteration at which the breakdown was detected.
        iteration: usize,
    },
    /// A scalar parameter is outside its documented range (a negative regularisation weight, a zero
    /// component count, a non-positive kernel width).
    InvalidParameter {
        /// Which parameter (a short static description).
        what: &'static str,
    },
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPositiveDefinite { pivot } => write!(f, "matrix is not positive definite (pivot {pivot})"),
            Self::Singular => f.write_str("matrix is singular to working precision"),
            Self::DimensionMismatch { what, expected, found } => {
                write!(f, "dimension mismatch: {what} should be {expected}, found {found}")
            }
            Self::NotConverged { iterations, residual } => {
                write!(f, "no convergence after {iterations} iterations (residual {residual:e})")
            }
            Self::NonFinite => f.write_str("non-finite value (NaN or infinity)"),
            Self::Breakdown { iteration } => {
                write!(f, "iteration {iteration} broke down: operator or preconditioner is not positive definite")
            }
            Self::InvalidParameter { what } => write!(f, "invalid parameter: {what}"),
        }
    }
}

impl std::error::Error for SolveError {}

/// `Ok` when `found == expected`, the matching [`SolveError::DimensionMismatch`] otherwise.
#[inline]
pub(crate) fn check_dim(what: &'static str, expected: usize, found: usize) -> Result<(), SolveError> {
    if expected == found { Ok(()) } else { Err(SolveError::DimensionMismatch { what, expected, found }) }
}

/// `Ok` when every element of `x` is finite, [`SolveError::NonFinite`] otherwise.
#[inline]
pub(crate) fn check_finite(x: &[f64]) -> Result<(), SolveError> {
    if x.iter().all(|v| v.is_finite()) { Ok(()) } else { Err(SolveError::NonFinite) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_names_the_problem() {
        let cases = [
            (SolveError::NotPositiveDefinite { pivot: 3 }, "pivot 3"),
            (SolveError::Singular, "singular"),
            (
                SolveError::DimensionMismatch { what: "rhs length", expected: 4, found: 5 },
                "rhs length should be 4, found 5",
            ),
            (SolveError::NotConverged { iterations: 7, residual: 0.5 }, "after 7 iterations"),
            (SolveError::NonFinite, "non-finite"),
            (SolveError::Breakdown { iteration: 2 }, "iteration 2"),
            (SolveError::InvalidParameter { what: "lambda" }, "lambda"),
        ];
        for (error, needle) in cases {
            let text = error.to_string();
            assert!(text.contains(needle), "{text:?} should contain {needle:?}");
            let as_dyn: &dyn std::error::Error = &error;
            assert!(as_dyn.source().is_none());
        }
    }

    #[test]
    fn checks_report_sizes_and_non_finite_values() {
        assert_eq!(check_dim("n", 2, 2), Ok(()));
        assert_eq!(check_dim("n", 2, 3), Err(SolveError::DimensionMismatch { what: "n", expected: 2, found: 3 }));
        assert_eq!(check_finite(&[0.0, -1.5]), Ok(()));
        assert_eq!(check_finite(&[0.0, f64::NAN]), Err(SolveError::NonFinite));
        assert_eq!(check_finite(&[f64::INFINITY]), Err(SolveError::NonFinite));
    }
}
