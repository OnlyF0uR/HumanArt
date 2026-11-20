//! Vector kernels on plain `f64` slices: the inner loops of every solver in this module `[PERF-R2]`.
//!
//! The functions take slices and never allocate. Where two slices are combined, the operation runs over
//! their common prefix (the shorter length); the solvers check dimensions before they call in here, so
//! that is never observable through the public solver APIs. Non-finite values propagate by the usual
//! IEEE rules.

/// The dot product `Σ xᵢ·yᵢ` over the common prefix of `x` and `y`.
///
/// Summed in four independent accumulators, which lets the compiler vectorise the loop and roughly
/// halves the rounding error of a plain running sum.
#[inline]
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    let (x, y) = (&x[..n], &y[..n]);
    let mut acc = [0.0_f64; 4];
    let (x_blocks, x_rest) = x.as_chunks::<4>();
    let (y_blocks, y_rest) = y.as_chunks::<4>();
    for (a, b) in x_blocks.iter().zip(y_blocks) {
        acc[0] += a[0] * b[0];
        acc[1] += a[1] * b[1];
        acc[2] += a[2] * b[2];
        acc[3] += a[3] * b[3];
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (a, b) in x_rest.iter().zip(y_rest) {
        sum += a * b;
    }
    sum
}

/// `y ← y + a·x` over the common prefix.
#[inline]
pub fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

/// `x ← a·x`.
#[inline]
pub fn scale(a: f64, x: &mut [f64]) {
    for xi in x {
        *xi *= a;
    }
}

/// The squared Euclidean norm `Σ xᵢ²`.
#[inline]
pub fn norm_sq(x: &[f64]) -> f64 {
    dot(x, x)
}

/// The Euclidean norm `‖x‖₂`.
///
/// Computed as `sqrt(Σ xᵢ²)` without rescaling: it overflows for elements beyond about `1e154` and
/// loses elements below about `1e-154`, far outside the ranges this crate works in (metres, weights).
#[inline]
pub fn norm(x: &[f64]) -> f64 {
    norm_sq(x).sqrt()
}

/// The maximum norm `max |xᵢ|` (0 for an empty slice). A NaN element is ignored by the maximum; use
/// [`all_finite`] to detect it.
#[inline]
pub fn norm_inf(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// The Euclidean distance `‖x − y‖₂` over the common prefix.
#[inline]
pub fn distance(x: &[f64], y: &[f64]) -> f64 {
    distance_sq(x, y).sqrt()
}

/// The squared Euclidean distance `‖x − y‖₂²` over the common prefix.
#[inline]
pub fn distance_sq(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum()
}

/// `out ← x − y` over the common prefix of the three slices.
#[inline]
pub fn sub_into(x: &[f64], y: &[f64], out: &mut [f64]) {
    for ((o, a), b) in out.iter_mut().zip(x).zip(y) {
        *o = a - b;
    }
}

/// `out ← x + y` over the common prefix of the three slices.
#[inline]
pub fn add_into(x: &[f64], y: &[f64], out: &mut [f64]) {
    for ((o, a), b) in out.iter_mut().zip(x).zip(y) {
        *o = a + b;
    }
}

/// Scales `x` to unit Euclidean length and returns the length it had.
///
/// A zero or non-finite vector is left unchanged (the returned norm tells the caller).
#[inline]
pub fn normalize(x: &mut [f64]) -> f64 {
    let length = norm(x);
    if length > 0.0 && length.is_finite() {
        scale(1.0 / length, x);
    }
    length
}

/// Applies the plane rotation `(x, y) ← (c·x − s·y, s·x + c·y)` element-wise over the common prefix:
/// the kernel of the Jacobi eigenvalue and singular value iterations.
#[inline]
pub fn rotate(x: &mut [f64], y: &mut [f64], c: f64, s: f64) {
    for (xi, yi) in x.iter_mut().zip(y) {
        let (a, b) = (*xi, *yi);
        *xi = c * a - s * b;
        *yi = s * a + c * b;
    }
}

/// `true` when no element is a NaN or an infinity.
#[inline]
pub fn all_finite(x: &[f64]) -> bool {
    x.iter().all(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    #[test]
    fn dot_matches_a_plain_sum_for_every_remainder_length() {
        let mut rng = Rng::new(11);
        for n in 0..23 {
            let x: Vec<f64> = (0..n).map(|_| rng.range_f64(-2.0, 2.0)).collect();
            let y: Vec<f64> = (0..n).map(|_| rng.range_f64(-2.0, 2.0)).collect();
            let plain: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
            assert!((dot(&x, &y) - plain).abs() <= 1e-13 * (1.0 + plain.abs()), "n = {n}");
        }
    }

    #[test]
    fn dot_uses_the_common_prefix() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0]), 14.0);
        assert_eq!(dot(&[], &[1.0]), 0.0);
    }

    #[test]
    fn axpy_scale_add_sub() {
        let mut y = [1.0, 2.0, 3.0];
        axpy(2.0, &[1.0, 1.0, 1.0], &mut y);
        assert_eq!(y, [3.0, 4.0, 5.0]);
        scale(-1.0, &mut y);
        assert_eq!(y, [-3.0, -4.0, -5.0]);
        let mut out = [0.0; 3];
        sub_into(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0], &mut out);
        assert_eq!(out, [-2.0, 0.0, 2.0]);
        add_into(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0], &mut out);
        assert_eq!(out, [4.0, 4.0, 4.0]);
    }

    #[test]
    fn norms_and_distances() {
        let x = [3.0, -4.0];
        assert_eq!(norm(&x), 5.0);
        assert_eq!(norm_sq(&x), 25.0);
        assert_eq!(norm_inf(&x), 4.0);
        assert_eq!(norm_inf(&[]), 0.0);
        assert_eq!(distance(&x, &[0.0, 0.0]), 5.0);
        assert_eq!(distance_sq(&x, &[3.0, 0.0]), 16.0);
    }

    #[test]
    fn normalize_leaves_degenerate_vectors_alone() {
        let mut x = [0.0, 3.0, 4.0];
        assert_eq!(normalize(&mut x), 5.0);
        assert!((norm(&x) - 1.0).abs() < 1e-15);
        let mut zero = [0.0, 0.0];
        assert_eq!(normalize(&mut zero), 0.0);
        assert_eq!(zero, [0.0, 0.0]);
        let mut bad = [f64::INFINITY, 1.0];
        assert!(normalize(&mut bad).is_infinite());
        assert_eq!(bad[1], 1.0);
    }

    #[test]
    fn rotate_is_a_plane_rotation() {
        let (c, s) = (0.6, 0.8);
        let mut x = [1.0, 0.0, 2.0];
        let mut y = [0.0, 1.0, -1.0];
        let before = norm_sq(&x) + norm_sq(&y);
        rotate(&mut x, &mut y, c, s);
        assert!((x[0] - 0.6).abs() < 1e-15 && (y[0] - 0.8).abs() < 1e-15);
        assert!((x[1] + 0.8).abs() < 1e-15 && (y[1] - 0.6).abs() < 1e-15);
        assert!((norm_sq(&x) + norm_sq(&y) - before).abs() < 1e-14);
    }

    #[test]
    fn all_finite_detects_nan_and_infinity() {
        assert!(all_finite(&[0.0, 1.0e300]));
        assert!(!all_finite(&[0.0, f64::NAN]));
        assert!(!all_finite(&[f64::NEG_INFINITY]));
    }
}
