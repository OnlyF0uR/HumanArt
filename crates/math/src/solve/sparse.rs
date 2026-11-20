//! Sparse matrices in compressed sparse row (CSR) form: the operators of the large linear systems
//! (Laplacian smoothing, the deformation-transfer and registration normal equations) that
//! [`super::pcg`] solves.
//!
//! A [`CsrMatrix`] stores, for each row, the column indices of its stored entries in strictly
//! increasing order and their values, in three flat arrays (`row_ptr`, `col_idx`, `values`): the
//! structure-of-arrays layout of `[PERF-R1]`, whose mat-vec walks memory once front to back. It is built
//! from coordinate triplets ([`CsrMatrix::from_triplets`], duplicates summed), from validated raw
//! arrays ([`CsrMatrix::from_parts`]) or from a dense matrix. Every constructor checks indices against
//! the dimensions and values for finiteness, so a matrix that exists is well formed and no method
//! indexes out of bounds.
//!
//! Allocation: [`CsrMatrix::mul_vec_into`] and [`CsrMatrix::tr_mul_vec_into`] write into caller storage
//! and allocate nothing `[PERF-R2]`; changing values without changing the pattern
//! ([`CsrMatrix::values_mut`]) keeps the matrix and everything built on its pattern reusable.

use super::dense::DenseMatrix;
use super::error::{SolveError, check_dim, check_finite};

/// A square or rectangular sparse matrix in compressed sparse row form.
#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix {
    rows: usize,
    cols: usize,
    /// `rows + 1` offsets: the entries of row `i` are `row_ptr[i]..row_ptr[i + 1]`.
    row_ptr: Vec<usize>,
    /// The column of each stored entry, strictly increasing within a row.
    col_idx: Vec<usize>,
    /// The value of each stored entry.
    values: Vec<f64>,
}

/// A linear operator `y = A·x`: what the iterative solvers need of a matrix. Implemented by
/// [`CsrMatrix`] and [`DenseMatrix`]; matrix-free operators can implement it too. The solvers are
/// generic over it, so the call is statically dispatched `[PERF-R4]`.
pub trait LinearOperator {
    /// The number of rows: the length of `y`.
    fn rows(&self) -> usize;
    /// The number of columns: the length of `x`.
    fn cols(&self) -> usize;
    /// Writes `A·x` into `y`. The caller guarantees `x.len() == cols()` and `y.len() == rows()`; the
    /// implementation must not allocate.
    fn apply(&self, x: &[f64], y: &mut [f64]);
}

impl CsrMatrix {
    /// The `rows × cols` matrix without stored entries.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, row_ptr: vec![0; rows + 1], col_idx: Vec::new(), values: Vec::new() }
    }

    /// The `n × n` identity.
    pub fn identity(n: usize) -> Self {
        Self { rows: n, cols: n, row_ptr: (0..=n).collect(), col_idx: (0..n).collect(), values: vec![1.0; n] }
    }

    /// Builds a matrix from `(row, column, value)` triplets in any order. Entries with the same
    /// position are summed; explicit zeros are kept as stored entries (they reserve the position in
    /// the pattern).
    ///
    /// Errors: [`SolveError::InvalidParameter`] for an index outside the dimensions,
    /// [`SolveError::NonFinite`] for a NaN or infinite value.
    pub fn from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize, f64)]) -> Result<Self, SolveError> {
        let mut counts = vec![0_usize; rows + 1];
        for &(r, c, v) in triplets {
            if r >= rows || c >= cols {
                return Err(SolveError::InvalidParameter { what: "sparse: triplet index outside the matrix" });
            }
            if !v.is_finite() {
                return Err(SolveError::NonFinite);
            }
            counts[r + 1] += 1;
        }
        for i in 0..rows {
            counts[i + 1] += counts[i];
        }
        // Bucket the triplets by row (a counting sort), then sort each row by column.
        let mut next = counts.clone();
        let mut entries = vec![(0_usize, 0.0_f64); triplets.len()];
        for &(r, c, v) in triplets {
            entries[next[r]] = (c, v);
            next[r] += 1;
        }
        let mut row_ptr = Vec::with_capacity(rows + 1);
        let mut col_idx = Vec::with_capacity(triplets.len());
        let mut values = Vec::with_capacity(triplets.len());
        row_ptr.push(0);
        for i in 0..rows {
            let row = &mut entries[counts[i]..counts[i + 1]];
            row.sort_unstable_by_key(|&(c, _)| c);
            let start = col_idx.len();
            for &(c, v) in row.iter() {
                if col_idx.len() > start && col_idx.last() == Some(&c) {
                    if let Some(last) = values.last_mut() {
                        *last += v;
                    }
                } else {
                    col_idx.push(c);
                    values.push(v);
                }
            }
            row_ptr.push(col_idx.len());
        }
        check_finite(&values)?;
        Ok(Self { rows, cols, row_ptr, col_idx, values })
    }

    /// Takes raw CSR arrays after checking them: `row_ptr` has `rows + 1` non-decreasing entries from 0
    /// to `col_idx.len()`, `col_idx` and `values` have the same length, the columns of each row are
    /// strictly increasing and below `cols`, every value is finite.
    pub fn from_parts(
        rows: usize,
        cols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, SolveError> {
        check_dim("sparse: row pointer count", rows + 1, row_ptr.len())?;
        check_dim("sparse: value count", col_idx.len(), values.len())?;
        if row_ptr.first() != Some(&0) || row_ptr.last() != Some(&col_idx.len()) {
            return Err(SolveError::InvalidParameter {
                what: "sparse: row pointers must run from 0 to the entry count",
            });
        }
        for window in row_ptr.windows(2) {
            if window[0] > window[1] {
                return Err(SolveError::InvalidParameter { what: "sparse: row pointers decrease" });
            }
            let row = &col_idx[window[0]..window[1]];
            if row.windows(2).any(|pair| pair[0] >= pair[1]) || row.last().is_some_and(|&c| c >= cols) {
                return Err(SolveError::InvalidParameter {
                    what: "sparse: columns of a row must be increasing and inside the matrix",
                });
            }
        }
        check_finite(&values)?;
        Ok(Self { rows, cols, row_ptr, col_idx, values })
    }

    /// The entries of `a` whose magnitude exceeds `drop_tolerance` (use 0 to keep every non-zero).
    pub fn from_dense(a: &DenseMatrix, drop_tolerance: f64) -> Result<Self, SolveError> {
        check_finite(a.as_slice())?;
        let mut row_ptr = Vec::with_capacity(a.rows() + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for i in 0..a.rows() {
            for (j, &v) in a.row(i).iter().enumerate() {
                if v.abs() > drop_tolerance {
                    col_idx.push(j);
                    values.push(v);
                }
            }
            row_ptr.push(col_idx.len());
        }
        Ok(Self { rows: a.rows(), cols: a.cols(), row_ptr, col_idx, values })
    }

    /// The dense copy of this matrix.
    pub fn to_dense(&self) -> DenseMatrix {
        let mut out = DenseMatrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            let (cols, values) = self.row(i);
            let row = out.row_mut(i);
            for (&c, &v) in cols.iter().zip(values) {
                row[c] = v;
            }
        }
        out
    }

    /// The number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The number of stored entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// The `rows + 1` row offsets.
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// The column index of each stored entry.
    pub fn col_indices(&self) -> &[usize] {
        &self.col_idx
    }

    /// The value of each stored entry.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The values, writable: the pattern stays fixed. Writing a non-finite value makes later solves
    /// fail with [`SolveError::NonFinite`] or [`SolveError::Breakdown`] instead of returning garbage.
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// The range of stored entries of row `i` (empty for `i >= rows`).
    fn range(&self, i: usize) -> std::ops::Range<usize> {
        match (self.row_ptr.get(i), self.row_ptr.get(i + 1)) {
            (Some(&start), Some(&end)) => start..end,
            _ => 0..0,
        }
    }

    /// The column indices and values of row `i` (both empty for `i >= rows`).
    pub fn row(&self, i: usize) -> (&[usize], &[f64]) {
        let range = self.range(i);
        (&self.col_idx[range.clone()], &self.values[range])
    }

    /// The element at `(i, j)`: the stored value, or 0 when the position is not stored or outside the
    /// matrix. Binary search within the row.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let (cols, values) = self.row(i);
        cols.binary_search(&j).map_or(0.0, |k| values[k])
    }

    /// Writes the main diagonal (`min(rows, cols)` values, 0 where not stored) into `out`.
    pub fn diagonal_into(&self, out: &mut [f64]) -> Result<(), SolveError> {
        check_dim("sparse diagonal: output length", self.rows.min(self.cols), out.len())?;
        for (i, d) in out.iter_mut().enumerate() {
            *d = self.get(i, i);
        }
        Ok(())
    }

    /// `y = A·x` into `y`. No allocation.
    pub fn mul_vec_into(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        check_dim("sparse mul_vec: x length", self.cols, x.len())?;
        check_dim("sparse mul_vec: y length", self.rows, y.len())?;
        self.mul_vec_unchecked(x, y);
        Ok(())
    }

    /// `A·x`.
    pub fn mul_vec(&self, x: &[f64]) -> Result<Vec<f64>, SolveError> {
        let mut y = vec![0.0; self.rows];
        self.mul_vec_into(x, &mut y)?;
        Ok(y)
    }

    /// The mat-vec kernel; lengths already checked. Column indices are below `cols` by construction.
    fn mul_vec_unchecked(&self, x: &[f64], y: &mut [f64]) {
        for (yi, window) in y.iter_mut().zip(self.row_ptr.windows(2)) {
            let range = window[0]..window[1];
            let mut sum = 0.0;
            for (&c, &v) in self.col_idx[range.clone()].iter().zip(&self.values[range]) {
                sum += v * x.get(c).copied().unwrap_or(0.0);
            }
            *yi = sum;
        }
    }

    /// `y = Aᵀ·x` into `y` without forming the transpose. No allocation.
    pub fn tr_mul_vec_into(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        check_dim("sparse tr_mul_vec: x length", self.rows, x.len())?;
        check_dim("sparse tr_mul_vec: y length", self.cols, y.len())?;
        y.fill(0.0);
        for (&xi, window) in x.iter().zip(self.row_ptr.windows(2)) {
            if xi == 0.0 {
                continue;
            }
            let range = window[0]..window[1];
            for (&c, &v) in self.col_idx[range.clone()].iter().zip(&self.values[range]) {
                if let Some(target) = y.get_mut(c) {
                    *target += v * xi;
                }
            }
        }
        Ok(())
    }

    /// `Aᵀ·x`.
    pub fn tr_mul_vec(&self, x: &[f64]) -> Result<Vec<f64>, SolveError> {
        let mut y = vec![0.0; self.cols];
        self.tr_mul_vec_into(x, &mut y)?;
        Ok(y)
    }

    /// The transpose, in CSR form (columns sorted). A counting sort: `O(nnz + rows + cols)`.
    pub fn transpose(&self) -> Self {
        let mut row_ptr = vec![0_usize; self.cols + 1];
        for &c in &self.col_idx {
            row_ptr[c + 1] += 1;
        }
        for j in 0..self.cols {
            row_ptr[j + 1] += row_ptr[j];
        }
        let mut next = row_ptr.clone();
        let mut col_idx = vec![0_usize; self.nnz()];
        let mut values = vec![0.0; self.nnz()];
        for i in 0..self.rows {
            let range = self.range(i);
            for (&c, &v) in self.col_idx[range.clone()].iter().zip(&self.values[range]) {
                let slot = next[c];
                col_idx[slot] = i;
                values[slot] = v;
                next[c] += 1;
            }
        }
        Self { rows: self.cols, cols: self.rows, row_ptr, col_idx, values }
    }

    /// Whether the matrix is square and `|aᵢⱼ − aⱼᵢ| ≤ tolerance` for every stored entry (a position
    /// stored on one side only compares with 0).
    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        self.rows == self.cols
            && (0..self.rows).all(|i| {
                let (cols, values) = self.row(i);
                cols.iter().zip(values).all(|(&j, &v)| (v - self.get(j, i)).abs() <= tolerance)
            })
    }
}

impl LinearOperator for CsrMatrix {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.mul_vec_unchecked(x, y);
    }
}

impl LinearOperator for DenseMatrix {
    fn rows(&self) -> usize {
        DenseMatrix::rows(self)
    }

    fn cols(&self) -> usize {
        DenseMatrix::cols(self)
    }

    fn apply(&self, x: &[f64], y: &mut [f64]) {
        for (i, yi) in y.iter_mut().enumerate() {
            *yi = super::vector::dot(self.row(i), x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solve::testutil::{max_abs_diff, random_matrix, random_vec};
    use humanart_core::rng::Rng;

    /// A random sparse matrix: each element non-zero with probability `density`.
    fn random_sparse_dense(rng: &mut Rng, rows: usize, cols: usize, density: f64) -> DenseMatrix {
        DenseMatrix::from_fn(
            rows,
            cols,
            |_, _| if rng.range_f64(0.0, 1.0) < density { rng.range_f64(-1.0, 1.0) } else { 0.0 },
        )
    }

    #[test]
    fn ac_wp02_1_triplets_are_sorted_and_duplicates_summed() {
        let a = CsrMatrix::from_triplets(
            3,
            4,
            &[(2, 3, 1.0), (0, 1, 2.0), (2, 0, -1.0), (0, 1, 0.5), (1, 2, 0.0), (0, 0, 4.0), (2, 3, 2.0)],
        )
        .unwrap();
        assert_eq!(a.row_ptr(), &[0, 2, 3, 5]);
        assert_eq!(a.col_indices(), &[0, 1, 2, 0, 3]);
        assert_eq!(a.values(), &[4.0, 2.5, 0.0, -1.0, 3.0]);
        assert_eq!(a.nnz(), 5);
        assert_eq!((a.get(0, 1), a.get(2, 3), a.get(1, 1), a.get(7, 7)), (2.5, 3.0, 0.0, 0.0));
        let dense = a.to_dense();
        assert_eq!(dense.row(2), &[-1.0, 0.0, 0.0, 3.0]);
        assert_eq!(CsrMatrix::from_parts(3, 4, vec![0, 2, 3, 5], vec![0, 1, 2, 0, 3], a.values().to_vec()).unwrap(), a);
        let mut diagonal = [0.0; 3];
        a.diagonal_into(&mut diagonal).unwrap();
        assert_eq!(diagonal, [4.0, 0.0, 0.0]);
        assert_eq!(CsrMatrix::identity(3).to_dense(), DenseMatrix::identity(3));
        assert_eq!(CsrMatrix::zeros(2, 5).to_dense(), DenseMatrix::zeros(2, 5));
    }

    #[test]
    fn ac_wp02_1_products_match_the_dense_reference() {
        let mut rng = Rng::new(0x70);
        for (m, n, density) in [(1, 1, 1.0), (12, 7, 0.3), (7, 12, 0.3), (50, 50, 0.05), (30, 30, 0.0)] {
            let dense = random_sparse_dense(&mut rng, m, n, density);
            // Build from shuffled triplets, each value split into two duplicates.
            let mut triplets = Vec::new();
            for i in 0..m {
                for j in 0..n {
                    let v = dense[(i, j)];
                    if v != 0.0 {
                        triplets.push((i, j, 0.25 * v));
                        triplets.push((i, j, 0.75 * v));
                    }
                }
            }
            for k in (1..triplets.len()).rev() {
                triplets.swap(k, rng.index(k + 1));
            }
            let a = CsrMatrix::from_triplets(m, n, &triplets).unwrap();
            assert!(max_abs_diff(&a.to_dense(), &dense) < 1e-15);
            assert_eq!(CsrMatrix::from_dense(&dense, 0.0).unwrap().nnz(), a.nnz());
            let x = random_vec(&mut rng, n);
            let y = a.mul_vec(&x).unwrap();
            for (p, q) in y.iter().zip(dense.mul_vec(&x).unwrap()) {
                assert!((p - q).abs() < 1e-14);
            }
            let x = random_vec(&mut rng, m);
            let y = a.tr_mul_vec(&x).unwrap();
            let t = a.transpose();
            assert_eq!((t.rows(), t.cols()), (n, m));
            assert!(max_abs_diff(&t.to_dense(), &dense.transpose()) < 1e-15);
            assert_eq!(t.transpose(), a);
            for ((p, q), r) in y.iter().zip(dense.tr_mul_vec(&x).unwrap()).zip(t.mul_vec(&x).unwrap()) {
                assert!((p - q).abs() < 1e-14 && (p - r).abs() < 1e-14);
            }
            // The generic operator gives the same product for both storage forms.
            let x = random_vec(&mut rng, n);
            let (mut y_sparse, mut y_dense) = (vec![0.0; m], vec![0.0; m]);
            LinearOperator::apply(&a, &x, &mut y_sparse);
            LinearOperator::apply(&dense, &x, &mut y_dense);
            for (p, q) in y_sparse.iter().zip(&y_dense) {
                assert!((p - q).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn symmetry_and_dropping() {
        let mut rng = Rng::new(0x71);
        let b = random_matrix(&mut rng, 6, 6);
        let a = CsrMatrix::from_dense(&b.gram(), 0.0).unwrap();
        assert!(a.is_symmetric(1e-15));
        assert!(!CsrMatrix::from_dense(&b, 0.0).unwrap().is_symmetric(1e-3));
        assert!(!CsrMatrix::zeros(2, 3).is_symmetric(1.0));
        let dropped = CsrMatrix::from_dense(&b, 0.5).unwrap();
        assert!(dropped.values().iter().all(|v| v.abs() > 0.5));
        let mut a = a;
        a.values_mut()[0] = 9.0;
        assert_eq!(a.get(0, 0), 9.0);
    }

    #[test]
    fn malformed_input_is_rejected() {
        assert!(matches!(CsrMatrix::from_triplets(2, 2, &[(2, 0, 1.0)]), Err(SolveError::InvalidParameter { .. })));
        assert!(matches!(CsrMatrix::from_triplets(2, 2, &[(0, 2, 1.0)]), Err(SolveError::InvalidParameter { .. })));
        assert_eq!(CsrMatrix::from_triplets(2, 2, &[(0, 0, f64::NAN)]), Err(SolveError::NonFinite));
        // Two finite duplicates that overflow when summed.
        assert_eq!(CsrMatrix::from_triplets(1, 1, &[(0, 0, f64::MAX), (0, 0, f64::MAX)]), Err(SolveError::NonFinite));
        let parts = |row_ptr: Vec<usize>, col_idx: Vec<usize>| {
            let values = vec![1.0; col_idx.len()];
            CsrMatrix::from_parts(2, 3, row_ptr, col_idx, values)
        };
        assert!(parts(vec![0, 1, 2], vec![0, 2]).is_ok());
        assert!(matches!(parts(vec![0, 2], vec![0, 1]), Err(SolveError::DimensionMismatch { .. })));
        assert!(parts(vec![1, 1, 2], vec![0, 2]).is_err());
        assert!(parts(vec![0, 1, 3], vec![0, 2]).is_err());
        assert!(parts(vec![0, 2, 1], vec![0, 2]).is_err());
        assert!(parts(vec![0, 2, 2], vec![2, 1]).is_err());
        assert!(parts(vec![0, 2, 2], vec![1, 1]).is_err());
        assert!(parts(vec![0, 1, 2], vec![0, 3]).is_err());
        assert!(CsrMatrix::from_parts(1, 1, vec![0, 1], vec![0], vec![]).is_err());
        assert!(CsrMatrix::from_parts(1, 1, vec![0, 1], vec![0], vec![f64::INFINITY]).is_err());
        let a = CsrMatrix::identity(3);
        assert!(matches!(a.mul_vec(&[1.0; 2]), Err(SolveError::DimensionMismatch { .. })));
        assert!(a.mul_vec_into(&[1.0; 3], &mut [0.0; 4]).is_err());
        assert!(a.tr_mul_vec(&[1.0; 4]).is_err());
        assert!(a.diagonal_into(&mut [0.0; 2]).is_err());
        assert_eq!(a.row(5), (&[][..], &[][..]));
        assert!(CsrMatrix::from_dense(&DenseMatrix::from_diagonal(&[f64::NAN]), 0.0).is_err());
    }
}
