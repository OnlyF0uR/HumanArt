//! The dense matrix of the solvers: [`DenseMatrix`] (owned) and [`MatRef`] (borrowed view).
//!
//! Storage is one contiguous row-major `f64` buffer `[PERF-R2]`: element `(r, c)` lives at
//! `r * cols + c`. Every product is written so that its inner loop walks contiguous memory (row dot
//! products and row `axpy`s), and every `*_into` method writes into caller-owned storage without
//! allocating once that storage has grown to size (`DenseMatrix::resize` keeps its buffer).
//!
//! Dimension errors are reported as [`SolveError::DimensionMismatch`]. Only the element accessors that
//! mirror slice indexing (`m[(r, c)]`, [`MatRef::row`], [`MatRef::at`]) panic on an out-of-range index,
//! exactly like a slice; [`MatRef::get`] is the checked form.

use std::ops::{Index, IndexMut};

use super::error::{SolveError, check_dim};
use super::vector::{axpy, dot};

/// A borrowed, read-only dense matrix: `rows × cols` values in row-major order.
///
/// A view costs nothing to make, so hot paths (the landmark solver, the per-iteration Jacobians) can
/// hand any `&[f64]` they already own to the solvers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatRef<'a> {
    rows: usize,
    cols: usize,
    data: &'a [f64],
}

impl<'a> MatRef<'a> {
    /// A view of `data` as a `rows × cols` row-major matrix.
    ///
    /// Fails with [`SolveError::DimensionMismatch`] when `data.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, data: &'a [f64]) -> Result<Self, SolveError> {
        let expected = rows.checked_mul(cols).ok_or(SolveError::DimensionMismatch {
            what: "rows * cols",
            expected: usize::MAX,
            found: data.len(),
        })?;
        check_dim("matrix data length", expected, data.len())?;
        Ok(Self { rows, cols, data })
    }

    /// A view whose length the caller has already established (`data.len() == rows * cols`). A wrong
    /// length cannot break memory safety; it would surface as a slice-index panic on first use.
    #[inline]
    pub(crate) fn from_parts(rows: usize, cols: usize, data: &'a [f64]) -> Self {
        debug_assert_eq!(rows * cols, data.len());
        Self { rows, cols, data }
    }

    /// The number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `true` when the matrix has as many rows as columns.
    #[inline]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// The row-major element buffer.
    #[inline]
    pub fn as_slice(&self) -> &'a [f64] {
        self.data
    }

    /// Row `r` as a slice. Panics when `r >= rows`, like slice indexing.
    #[inline]
    pub fn row(&self, r: usize) -> &'a [f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Element `(r, c)`. Panics when the index is out of range, like slice indexing.
    #[inline]
    pub fn at(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Element `(r, c)`, or `None` when the index is out of range.
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> Option<f64> {
        if r < self.rows && c < self.cols { self.data.get(r * self.cols + c).copied() } else { None }
    }

    /// `true` when no element is a NaN or an infinity.
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }

    /// The Frobenius norm `sqrt(Σ aᵢⱼ²)`.
    pub fn frobenius_norm(&self) -> f64 {
        dot(self.data, self.data).sqrt()
    }

    /// The largest absolute element (0 for an empty matrix).
    pub fn max_abs(&self) -> f64 {
        self.data.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
    }

    /// `true` when the matrix is square and `|aᵢⱼ − aⱼᵢ| <= tolerance` for every pair.
    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        if !self.is_square() {
            return false;
        }
        let n = self.rows;
        (0..n).all(|i| (0..i).all(|j| (self.data[i * n + j] - self.data[j * n + i]).abs() <= tolerance))
    }

    /// `y ← A·x`. No allocation.
    pub fn mul_vec_into(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        check_dim("mul_vec: x length", self.cols, x.len())?;
        check_dim("mul_vec: y length", self.rows, y.len())?;
        for (r, yr) in y.iter_mut().enumerate() {
            *yr = dot(self.row(r), x);
        }
        Ok(())
    }

    /// `y ← Aᵀ·x`. No allocation.
    pub fn tr_mul_vec_into(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        check_dim("tr_mul_vec: x length", self.rows, x.len())?;
        check_dim("tr_mul_vec: y length", self.cols, y.len())?;
        y.fill(0.0);
        for (r, &xr) in x.iter().enumerate() {
            if xr != 0.0 {
                axpy(xr, self.row(r), y);
            }
        }
        Ok(())
    }

    /// `out ← A·B`. `out` is resized (its buffer is reused, so a warm `out` does not allocate).
    pub fn mul_into(&self, rhs: MatRef<'_>, out: &mut DenseMatrix) -> Result<(), SolveError> {
        check_dim("mul: rhs rows", self.cols, rhs.rows)?;
        out.resize(self.rows, rhs.cols);
        if rhs.cols == 0 {
            return Ok(());
        }
        for (r, out_row) in out.data.chunks_exact_mut(rhs.cols).enumerate() {
            for (k, &a) in self.row(r).iter().enumerate() {
                if a != 0.0 {
                    axpy(a, rhs.row(k), out_row);
                }
            }
        }
        Ok(())
    }

    /// `out ← Aᵀ·A`, the `cols × cols` Gram matrix of the columns. `out` is resized, not reallocated
    /// when warm. Exactly symmetric.
    pub fn gram_into(&self, out: &mut DenseMatrix) {
        let n = self.cols;
        out.resize(n, n);
        for r in 0..self.rows {
            let row = self.row(r);
            for (i, &a) in row.iter().enumerate() {
                if a != 0.0 {
                    axpy(a, &row[i..], &mut out.data[i * n + i..(i + 1) * n]);
                }
            }
        }
        out.mirror_upper_to_lower();
    }

    /// `out ← A·Aᵀ`, the `rows × rows` Gram matrix of the rows. `out` is resized, not reallocated when
    /// warm. Exactly symmetric.
    pub fn outer_gram_into(&self, out: &mut DenseMatrix) {
        let m = self.rows;
        out.resize(m, m);
        for i in 0..m {
            let row_i = self.row(i);
            for j in 0..=i {
                let v = dot(row_i, self.row(j));
                out.data[i * m + j] = v;
                out.data[j * m + i] = v;
            }
        }
    }

    /// `out ← Aᵀ`. `out` is resized, not reallocated when warm.
    pub fn transpose_into(&self, out: &mut DenseMatrix) {
        out.resize(self.cols, self.rows);
        for r in 0..self.rows {
            for (c, &v) in self.row(r).iter().enumerate() {
                out.data[c * self.rows + r] = v;
            }
        }
    }

    /// An owned copy.
    pub fn to_owned(&self) -> DenseMatrix {
        DenseMatrix { rows: self.rows, cols: self.cols, data: self.data.to_vec() }
    }
}

/// An owned dense matrix of `f64`, row-major.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DenseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl DenseMatrix {
    /// A `rows × cols` matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows.saturating_mul(cols)] }
    }

    /// The `n × n` identity.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// A square matrix with `diagonal` on its diagonal and zeros elsewhere.
    pub fn from_diagonal(diagonal: &[f64]) -> Self {
        let n = diagonal.len();
        let mut m = Self::zeros(n, n);
        for (i, &d) in diagonal.iter().enumerate() {
            m.data[i * n + i] = d;
        }
        m
    }

    /// Takes ownership of a row-major buffer.
    ///
    /// Fails with [`SolveError::DimensionMismatch`] when `data.len() != rows * cols`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, SolveError> {
        MatRef::new(rows, cols, &data)?;
        Ok(Self { rows, cols, data })
    }

    /// Builds a matrix from equally long rows.
    ///
    /// Fails with [`SolveError::DimensionMismatch`] when the rows differ in length.
    pub fn from_rows<R: AsRef<[f64]>>(rows: &[R]) -> Result<Self, SolveError> {
        let cols = rows.first().map_or(0, |r| r.as_ref().len());
        let mut data = Vec::with_capacity(rows.len().saturating_mul(cols));
        for row in rows {
            check_dim("row length", cols, row.as_ref().len())?;
            data.extend_from_slice(row.as_ref());
        }
        Ok(Self { rows: rows.len(), cols, data })
    }

    /// Builds a matrix whose element `(r, c)` is `f(r, c)`.
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> f64) -> Self {
        let mut data = Vec::with_capacity(rows.saturating_mul(cols));
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Self { rows, cols, data }
    }

    /// The number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `true` when the matrix has as many rows as columns.
    #[inline]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// A borrowed view of the whole matrix.
    #[inline]
    pub fn view(&self) -> MatRef<'_> {
        MatRef { rows: self.rows, cols: self.cols, data: &self.data }
    }

    /// The row-major element buffer.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// The row-major element buffer, mutable.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Gives up the row-major buffer.
    pub fn into_vec(self) -> Vec<f64> {
        self.data
    }

    /// Row `r` as a slice. Panics when `r >= rows`, like slice indexing.
    #[inline]
    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Row `r` as a mutable slice. Panics when `r >= rows`, like slice indexing.
    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f64] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Element `(r, c)`, or `None` when the index is out of range.
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> Option<f64> {
        self.view().get(r, c)
    }

    /// Copies column `c` into `out`.
    pub fn col_into(&self, c: usize, out: &mut [f64]) -> Result<(), SolveError> {
        if c >= self.cols {
            return Err(SolveError::DimensionMismatch { what: "column index", expected: self.cols, found: c });
        }
        check_dim("column length", self.rows, out.len())?;
        for (r, o) in out.iter_mut().enumerate() {
            *o = self.data[r * self.cols + c];
        }
        Ok(())
    }

    /// Column `c` as a new vector.
    pub fn col(&self, c: usize) -> Result<Vec<f64>, SolveError> {
        let mut out = vec![0.0; self.rows];
        self.col_into(c, &mut out)?;
        Ok(out)
    }

    /// Changes the shape to `rows × cols` and zeroes every element. The buffer is kept, so shrinking or
    /// returning to an earlier size does not allocate.
    pub fn resize(&mut self, rows: usize, cols: usize) {
        self.rows = rows;
        self.cols = cols;
        self.data.clear();
        self.data.resize(rows.saturating_mul(cols), 0.0);
    }

    /// Keeps the first `rows` rows (no effect when the matrix has no more). No allocation.
    pub fn truncate_rows(&mut self, rows: usize) {
        if rows < self.rows {
            self.rows = rows;
            self.data.truncate(rows * self.cols);
        }
    }

    /// Sets every element to `value`.
    pub fn fill(&mut self, value: f64) {
        self.data.fill(value);
    }

    /// Makes `self` a copy of `source`, reusing the buffer.
    pub fn copy_from(&mut self, source: MatRef<'_>) {
        self.rows = source.rows;
        self.cols = source.cols;
        self.data.clear();
        self.data.extend_from_slice(source.data);
    }

    /// `true` when no element is a NaN or an infinity.
    pub fn is_finite(&self) -> bool {
        self.view().is_finite()
    }

    /// The Frobenius norm `sqrt(Σ aᵢⱼ²)`.
    pub fn frobenius_norm(&self) -> f64 {
        self.view().frobenius_norm()
    }

    /// The largest absolute element (0 for an empty matrix).
    pub fn max_abs(&self) -> f64 {
        self.view().max_abs()
    }

    /// `true` when the matrix is square and `|aᵢⱼ − aⱼᵢ| <= tolerance` for every pair.
    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        self.view().is_symmetric(tolerance)
    }

    /// The transpose, as a new matrix.
    pub fn transpose(&self) -> Self {
        let mut out = Self::default();
        self.view().transpose_into(&mut out);
        out
    }

    /// The product `self · rhs`, as a new matrix.
    pub fn mul(&self, rhs: &Self) -> Result<Self, SolveError> {
        let mut out = Self::default();
        self.view().mul_into(rhs.view(), &mut out)?;
        Ok(out)
    }

    /// The product `self · x`, as a new vector.
    pub fn mul_vec(&self, x: &[f64]) -> Result<Vec<f64>, SolveError> {
        let mut y = vec![0.0; self.rows];
        self.view().mul_vec_into(x, &mut y)?;
        Ok(y)
    }

    /// `y ← self · x`. No allocation.
    pub fn mul_vec_into(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.view().mul_vec_into(x, y)
    }

    /// The product `selfᵀ · x`, as a new vector.
    pub fn tr_mul_vec(&self, x: &[f64]) -> Result<Vec<f64>, SolveError> {
        let mut y = vec![0.0; self.cols];
        self.view().tr_mul_vec_into(x, &mut y)?;
        Ok(y)
    }

    /// `y ← selfᵀ · x`. No allocation.
    pub fn tr_mul_vec_into(&self, x: &[f64], y: &mut [f64]) -> Result<(), SolveError> {
        self.view().tr_mul_vec_into(x, y)
    }

    /// The Gram matrix of the columns, `AᵀA` (`cols × cols`).
    pub fn gram(&self) -> Self {
        let mut out = Self::default();
        self.view().gram_into(&mut out);
        out
    }

    /// The Gram matrix of the rows, `AAᵀ` (`rows × rows`).
    pub fn outer_gram(&self) -> Self {
        let mut out = Self::default();
        self.view().outer_gram_into(&mut out);
        out
    }

    /// `self ← self + alpha · other`.
    pub fn add_scaled(&mut self, alpha: f64, other: &Self) -> Result<(), SolveError> {
        check_dim("add_scaled: rows", self.rows, other.rows)?;
        check_dim("add_scaled: cols", self.cols, other.cols)?;
        axpy(alpha, &other.data, &mut self.data);
        Ok(())
    }

    /// `self ← alpha · self`.
    pub fn scale(&mut self, alpha: f64) {
        super::vector::scale(alpha, &mut self.data);
    }

    /// Adds `shift` to every diagonal element (of the leading square part).
    pub fn add_to_diagonal(&mut self, shift: f64) {
        for i in 0..self.rows.min(self.cols) {
            self.data[i * self.cols + i] += shift;
        }
    }

    /// Swaps rows `a` and `b`.
    pub fn swap_rows(&mut self, a: usize, b: usize) -> Result<(), SolveError> {
        if a >= self.rows || b >= self.rows {
            return Err(SolveError::DimensionMismatch { what: "row index", expected: self.rows, found: a.max(b) });
        }
        swap_rows_in(&mut self.data, self.cols, a, b);
        Ok(())
    }

    /// Copies the upper triangle onto the lower one (square matrices; a no-op otherwise).
    pub(crate) fn mirror_upper_to_lower(&mut self) {
        if self.rows != self.cols {
            return;
        }
        let n = self.rows;
        for i in 0..n {
            for j in 0..i {
                self.data[i * n + j] = self.data[j * n + i];
            }
        }
    }
}

/// Swaps rows `a` and `b` of a row-major buffer with `cols` columns. Indices must be in range.
pub(crate) fn swap_rows_in(data: &mut [f64], cols: usize, a: usize, b: usize) {
    if a == b || cols == 0 {
        return;
    }
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    let (head, tail) = data.split_at_mut(hi * cols);
    head[lo * cols..(lo + 1) * cols].swap_with_slice(&mut tail[..cols]);
}

impl Index<(usize, usize)> for DenseMatrix {
    type Output = f64;

    /// Element `(row, column)`. Panics when the index is out of range, like slice indexing.
    #[inline]
    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        &self.row(r)[c]
    }
}

impl IndexMut<(usize, usize)> for DenseMatrix {
    /// Element `(row, column)`. Panics when the index is out of range, like slice indexing.
    #[inline]
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f64 {
        &mut self.row_mut(r)[c]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    fn random(rng: &mut Rng, rows: usize, cols: usize) -> DenseMatrix {
        DenseMatrix::from_fn(rows, cols, |_, _| rng.range_f64(-1.0, 1.0))
    }

    fn naive_mul(a: &DenseMatrix, b: &DenseMatrix) -> DenseMatrix {
        DenseMatrix::from_fn(a.rows(), b.cols(), |r, c| (0..a.cols()).map(|k| a[(r, k)] * b[(k, c)]).sum())
    }

    fn max_diff(a: &DenseMatrix, b: &DenseMatrix) -> f64 {
        assert_eq!((a.rows(), a.cols()), (b.rows(), b.cols()));
        a.as_slice().iter().zip(b.as_slice()).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
    }

    #[test]
    fn construction_and_indexing() {
        let m = DenseMatrix::from_rows(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unwrap();
        assert_eq!((m.rows(), m.cols()), (2, 3));
        assert_eq!(m[(1, 2)], 6.0);
        assert_eq!(m.get(1, 2), Some(6.0));
        assert_eq!(m.get(2, 0), None);
        assert_eq!(m.get(0, 3), None);
        assert_eq!(m.row(1), &[4.0, 5.0, 6.0]);
        assert_eq!(m.col(1).unwrap(), vec![2.0, 5.0]);
        assert!(m.col(3).is_err());
        assert_eq!(m.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut m = m;
        m[(0, 0)] = 9.0;
        assert_eq!(m.view().at(0, 0), 9.0);
        assert_eq!(DenseMatrix::identity(2).as_slice(), &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(DenseMatrix::from_diagonal(&[2.0, 3.0]).as_slice(), &[2.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn bad_shapes_are_errors_not_panics() {
        assert!(matches!(DenseMatrix::from_vec(2, 2, vec![0.0; 3]), Err(SolveError::DimensionMismatch { .. })));
        assert!(MatRef::new(usize::MAX, 2, &[]).is_err());
        assert!(DenseMatrix::from_rows(&[vec![1.0, 2.0], vec![1.0]]).is_err());
        let a = DenseMatrix::zeros(2, 3);
        let b = DenseMatrix::zeros(2, 3);
        assert!(a.mul(&b).is_err());
        assert!(a.mul_vec(&[0.0; 2]).is_err());
        assert!(a.tr_mul_vec(&[0.0; 3]).is_err());
        let mut y = [0.0; 3];
        assert!(a.mul_vec_into(&[0.0; 3], &mut y).is_err());
        let mut c = DenseMatrix::zeros(3, 2);
        assert!(c.add_scaled(1.0, &a).is_err());
        assert!(c.swap_rows(0, 3).is_err());
    }

    #[test]
    fn products_match_the_naive_triple_loop() {
        let mut rng = Rng::new(2);
        let a = random(&mut rng, 5, 7);
        let b = random(&mut rng, 7, 4);
        assert!(max_diff(&a.mul(&b).unwrap(), &naive_mul(&a, &b)) < 1e-14);
        let at = a.transpose();
        assert_eq!((at.rows(), at.cols()), (7, 5));
        assert_eq!(at[(6, 4)], a[(4, 6)]);
        assert!(max_diff(&a.gram(), &naive_mul(&at, &a)) < 1e-14);
        assert!(max_diff(&a.outer_gram(), &naive_mul(&a, &at)) < 1e-14);
        assert!(a.gram().is_symmetric(0.0));
        assert!(a.outer_gram().is_symmetric(0.0));

        let x: Vec<f64> = (0..7).map(|i| f64::from(i) - 3.0).collect();
        let y = a.mul_vec(&x).unwrap();
        for r in 0..5 {
            let expect: f64 = (0..7).map(|c| a[(r, c)] * x[c]).sum();
            assert!((y[r] - expect).abs() < 1e-14);
        }
        let z = a.tr_mul_vec(&y).unwrap();
        let expect = at.mul_vec(&y).unwrap();
        for c in 0..7 {
            assert!((z[c] - expect[c]).abs() < 1e-13);
        }
    }

    #[test]
    fn empty_matrices_are_handled() {
        let a = DenseMatrix::zeros(0, 3);
        let b = DenseMatrix::zeros(3, 0);
        let ab = a.mul(&b).unwrap();
        assert_eq!((ab.rows(), ab.cols()), (0, 0));
        let ba = b.mul(&a).unwrap();
        assert_eq!((ba.rows(), ba.cols()), (3, 3));
        assert_eq!(ba.max_abs(), 0.0);
        assert_eq!(a.gram().rows(), 3);
        assert_eq!(a.outer_gram().rows(), 0);
        assert_eq!(a.transpose().rows(), 3);
        assert_eq!(a.frobenius_norm(), 0.0);
    }

    #[test]
    fn into_variants_reuse_the_buffer() {
        // [PERF-R2] the `*_into` methods write into caller storage: after the first call the
        // destination's buffer pointer does not change, i.e. nothing was reallocated.
        let mut rng = Rng::new(3);
        let a = random(&mut rng, 6, 4);
        let mut out = DenseMatrix::zeros(6, 6);
        let before = out.as_slice().as_ptr();
        a.view().gram_into(&mut out);
        a.view().outer_gram_into(&mut out);
        a.view().transpose_into(&mut out);
        a.view().mul_into(a.transpose().view(), &mut out).unwrap();
        assert_eq!(out.as_slice().as_ptr(), before);
        assert_eq!((out.rows(), out.cols()), (6, 6));
    }

    #[test]
    fn row_operations_and_norms() {
        let mut m = DenseMatrix::from_rows(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).unwrap();
        m.swap_rows(0, 2).unwrap();
        assert_eq!(m.as_slice(), &[5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
        m.swap_rows(1, 1).unwrap();
        m.scale(2.0);
        assert_eq!(m.max_abs(), 12.0);
        m.add_to_diagonal(1.0);
        assert_eq!(m[(0, 0)], 11.0);
        assert_eq!(m[(1, 1)], 9.0);
        let other = m.clone();
        m.add_scaled(-1.0, &other).unwrap();
        assert_eq!(m.frobenius_norm(), 0.0);
        assert!(m.is_finite());
        m[(0, 0)] = f64::NAN;
        assert!(!m.is_finite());
        let mut c = DenseMatrix::default();
        c.copy_from(other.view());
        assert_eq!(c, other);
        c.fill(1.0);
        assert_eq!(c.frobenius_norm(), 6.0_f64.sqrt());
        assert!(!c.is_symmetric(0.0));
        assert_eq!(c.view().to_owned().into_vec(), vec![1.0; 6]);
    }
}
