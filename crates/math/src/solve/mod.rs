//! Dense and sparse solvers on `f64` `[AC-WP02-1]`.
//!
//! Everything here works on plain `f64` slices, on the module's own row-major [`DenseMatrix`] and on
//! small arrays (`[f64; 3]`, `[[f64; 3]; 3]`, row-major: `m[row][column]`), so that it depends on no
//! other part of this crate. No routine panics on bad dimensions, non-finite input, or a matrix
//! lacking the property its method needs: all of them return [`SolveError`].
//!
//! | module | contents |
//! |---|---|
//! | [`error`] | [`SolveError`], the one error enum of the solvers |
//! | [`vector`] | BLAS-1 kernels on slices: `dot`, `axpy`, norms |
//! | [`dense`] | [`DenseMatrix`] (owned) and [`MatRef`] (borrowed), products, Gram matrices |
//! | [`lu`] | LU with partial pivoting: general square solves, determinant, inverse |
//! | [`cholesky`] | Cholesky `L·Lᵀ` for symmetric positive definite systems, pivot-free `L·D·Lᵀ` |
//! | [`qr`] | column-pivoted Householder QR: least squares, numerical rank |
//! | [`eigen`] | symmetric eigen-decomposition: cyclic Jacobi, and a heap-free 3×3 version |
//! | [`lsq`] | Tikhonov (ridge) least squares, non-negative least squares, Levenberg-Marquardt |
//! | [`sparse`] | [`CsrMatrix`] built from triplets, mat-vec, transpose; the [`LinearOperator`] trait |
//! | [`pca`] | principal component analysis (Gram / covariance), explained variance, leave-one-out selection |
//! | [`pcg`] | preconditioned conjugate gradients with Jacobi and IC(0) preconditioners |
//! | [`svd`] | one-sided Jacobi SVD: pseudo-inverse, rank, condition number; 3×3 SVD, polar decomposition, Kabsch and Procrustes point fits |
//!
//! Conventions: matrices are row-major; factorisation objects (`Lu`, `Cholesky`, `Qr`, ...) are made
//! once and solve many right-hand sides; methods ending in `_into` or `_in_place` write into storage
//! the caller owns and do not allocate once that storage is warm `[PERF-R2]`.

pub mod cholesky;
pub mod dense;
pub mod eigen;
pub mod error;
pub mod lsq;
pub mod lu;
pub mod pca;
pub mod pcg;
pub mod qr;
pub mod sparse;
pub mod svd;
pub mod vector;

#[cfg(test)]
mod testutil;

pub use cholesky::{Cholesky, Ldlt};
pub use dense::{DenseMatrix, MatRef};
pub use eigen::{Eigen3, SymmetricEigen, symmetric_eigen3};
pub use error::SolveError;
pub use lsq::{
    LeastSquaresProblem, LmReport, LmSettings, LmStop, LmWorkspace, NnlsReport, NnlsWorkspace, TikhonovWorkspace,
    levenberg_marquardt, nnls, nnls_into, tikhonov, tikhonov_into,
};
pub use lu::Lu;
pub use pca::{LeaveOneOut, Pca, PcaMethod, RandomizedPca, leave_one_out};
pub use pcg::{
    IdentityPreconditioner, IncompleteCholesky, JacobiPreconditioner, PcgReport, PcgSettings, PcgWorkspace,
    Preconditioner, pcg, pcg_solve,
};
pub use qr::Qr;
pub use sparse::{CsrMatrix, LinearOperator};
pub use svd::{Polar3, RigidFit, Svd, Svd3, kabsch, nearest_rotation3, polar3, procrustes, svd3};
