//! Helpers shared by the unit tests of the solver modules (compiled for tests only).

use humanart_core::rng::Rng;

use super::dense::DenseMatrix;
use super::vector::{dot, norm};

/// A `rows × cols` matrix of uniform values in `[-1, 1)`.
pub fn random_matrix(rng: &mut Rng, rows: usize, cols: usize) -> DenseMatrix {
    DenseMatrix::from_fn(rows, cols, |_, _| rng.range_f64(-1.0, 1.0))
}

/// A vector of uniform values in `[-1, 1)`.
pub fn random_vec(rng: &mut Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.range_f64(-1.0, 1.0)).collect()
}

/// A random symmetric positive definite matrix `BᵀB + (n/2)·I`: well conditioned (order 10).
pub fn random_spd(rng: &mut Rng, n: usize) -> DenseMatrix {
    let mut a = random_matrix(rng, n, n).gram();
    a.add_to_diagonal(n as f64 * 0.5);
    a
}

/// A random symmetric matrix with elements in `[-1, 1)`.
pub fn random_symmetric(rng: &mut Rng, n: usize) -> DenseMatrix {
    let b = random_matrix(rng, n, n);
    DenseMatrix::from_fn(n, n, |i, j| f64::midpoint(b[(i, j)], b[(j, i)]))
}

/// `‖A·x − b‖₂ / ‖b‖₂`.
pub fn relative_residual(a: &DenseMatrix, x: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.cols(), x.len());
    assert_eq!(a.rows(), b.len());
    let r: Vec<f64> = (0..a.rows()).map(|i| dot(a.row(i), x) - b[i]).collect();
    norm(&r) / norm(b)
}

/// The largest absolute element-wise difference of two equally shaped matrices.
pub fn max_abs_diff(a: &DenseMatrix, b: &DenseMatrix) -> f64 {
    assert_eq!((a.rows(), a.cols()), (b.rows(), b.cols()));
    a.as_slice().iter().zip(b.as_slice()).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

/// `max |VᵀV − I|`: how far the columns of `v` are from orthonormal.
pub fn orthonormality_error(v: &DenseMatrix) -> f64 {
    max_abs_diff(&v.gram(), &DenseMatrix::identity(v.cols()))
}

/// The naive product `a · b` (the reference the optimised products are compared with).
pub fn naive_mul(a: &DenseMatrix, b: &DenseMatrix) -> DenseMatrix {
    assert_eq!(a.cols(), b.rows());
    DenseMatrix::from_fn(a.rows(), b.cols(), |r, c| (0..a.cols()).map(|k| a[(r, k)] * b[(k, c)]).sum())
}
