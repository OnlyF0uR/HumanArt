//! Scalar helpers: interpolation, remapping, angle wrapping and the approximate comparisons the tests
//! of every numeric crate use.

use core::f32::consts::{PI, TAU};

/// Linear interpolation: `a` at `t = 0`, `b` at `t = 1`; `t` outside `[0, 1]` extrapolates.
///
/// Written as `a + (b - a) * t`, which is exact at `t = 0` and monotonic in `t`.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// [`lerp`] in double precision.
#[inline]
pub fn lerp_f64(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// The `t` for which `lerp(a, b, t) == v`. Returns `0` when `a == b` (the range is empty).
#[inline]
pub fn inverse_lerp(a: f32, b: f32, v: f32) -> f32 {
    let span = b - a;
    if span == 0.0 { 0.0 } else { (v - a) / span }
}

/// Maps `v` from the range `[in_lo, in_hi]` to `[out_lo, out_hi]` without clamping.
///
/// An empty input range maps everything to `out_lo`.
#[inline]
pub fn remap(v: f32, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> f32 {
    lerp(out_lo, out_hi, inverse_lerp(in_lo, in_hi, v))
}

/// [`remap`] with the result clamped to the output range.
#[inline]
pub fn remap_clamped(v: f32, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> f32 {
    lerp(out_lo, out_hi, saturate(inverse_lerp(in_lo, in_hi, v)))
}

/// Clamps to `[0, 1]`. NaN becomes `0`.
#[inline]
#[allow(clippy::manual_clamp)]
pub fn saturate(v: f32) -> f32 {
    // `max` returns the other operand for NaN, so NaN goes to 0 where `clamp` would propagate it.
    v.max(0.0).min(1.0)
}

/// Wraps an angle in radians to `(-π, π]`. Non-finite input returns `0`.
#[inline]
pub fn wrap_angle(angle: f32) -> f32 {
    if !angle.is_finite() {
        return 0.0;
    }
    let wrapped = angle - TAU * ((angle + PI) / TAU).floor();
    // Rounding can land a hair outside the interval, and the lower end is exclusive.
    if wrapped > PI {
        wrapped - TAU
    } else if wrapped <= -PI {
        wrapped + TAU
    } else {
        wrapped
    }
}

/// Wraps an angle in radians to `[0, 2π)`. Non-finite input returns `0`.
#[inline]
pub fn wrap_angle_positive(angle: f32) -> f32 {
    if !angle.is_finite() {
        return 0.0;
    }
    let wrapped = angle.rem_euclid(TAU);
    if wrapped >= TAU { 0.0 } else { wrapped }
}

/// The signed shortest rotation from angle `from` to angle `to`, in `(-π, π]`.
#[inline]
pub fn angle_delta(from: f32, to: f32) -> f32 {
    wrap_angle(to - from)
}

/// Interpolates between two angles along the shortest arc; the result is wrapped to `(-π, π]`.
#[inline]
pub fn lerp_angle(from: f32, to: f32, t: f32) -> f32 {
    wrap_angle(from + angle_delta(from, to) * t)
}

/// Whether `a` and `b` differ by at most `epsilon` (absolute). False when either is NaN.
#[inline]
pub fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
    (a - b).abs() <= epsilon
}

/// [`approx_eq`] in double precision.
#[inline]
pub fn approx_eq_f64(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() <= epsilon
}

/// Whether `a` and `b` agree to a relative tolerance: `|a - b| <= max(abs, rel * max(|a|, |b|))`.
///
/// The absolute floor `abs` keeps the comparison meaningful near zero. False when either is NaN.
#[inline]
pub fn approx_eq_rel(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    (a - b).abs() <= abs.max(rel * a.abs().max(b.abs()))
}

/// [`approx_eq_rel`] in double precision.
#[inline]
pub fn approx_eq_rel_f64(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    (a - b).abs() <= abs.max(rel * a.abs().max(b.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    #[test]
    fn lerp_hits_both_ends_and_inverts() {
        assert_eq!(lerp(2.0, 6.0, 0.0), 2.0);
        assert_eq!(lerp(2.0, 6.0, 1.0), 6.0);
        assert_eq!(lerp(2.0, 6.0, 0.25), 3.0);
        assert_eq!(lerp_f64(-1.0, 1.0, 0.5), 0.0);
        assert_eq!(inverse_lerp(2.0, 6.0, 3.0), 0.25);
        assert_eq!(inverse_lerp(5.0, 5.0, 9.0), 0.0);
        let mut rng = Rng::new(11);
        for _ in 0..1000 {
            let (a, b, t) = (rng.range_f32(-50.0, 50.0), rng.range_f32(-50.0, 50.0), rng.f32());
            if (a - b).abs() > 1e-3 {
                assert!(approx_eq(inverse_lerp(a, b, lerp(a, b, t)), t, 1e-3));
            }
        }
    }

    #[test]
    fn remap_and_saturate() {
        assert_eq!(remap(5.0, 0.0, 10.0, 100.0, 200.0), 150.0);
        assert_eq!(remap(15.0, 0.0, 10.0, 100.0, 200.0), 250.0);
        assert_eq!(remap_clamped(15.0, 0.0, 10.0, 100.0, 200.0), 200.0);
        assert_eq!(remap_clamped(-3.0, 0.0, 10.0, 100.0, 200.0), 100.0);
        assert_eq!(remap(1.0, 4.0, 4.0, 7.0, 9.0), 7.0);
        assert_eq!(saturate(-0.5), 0.0);
        assert_eq!(saturate(0.25), 0.25);
        assert_eq!(saturate(3.0), 1.0);
        assert_eq!(saturate(f32::NAN), 0.0);
    }

    #[test]
    fn angles_wrap_to_their_intervals() {
        assert!(approx_eq(wrap_angle(3.0 * PI), PI, 1e-5));
        assert!(approx_eq(wrap_angle(-PI), PI, 1e-6));
        assert!(approx_eq(wrap_angle(0.5), 0.5, 0.0));
        assert_eq!(wrap_angle(f32::INFINITY), 0.0);
        assert_eq!(wrap_angle_positive(f32::NAN), 0.0);
        assert!(approx_eq(wrap_angle_positive(-0.25), TAU - 0.25, 1e-6));
        let mut rng = Rng::new(5);
        for _ in 0..5000 {
            let a = rng.range_f32(-100.0, 100.0);
            let w = wrap_angle(a);
            assert!(w > -PI && w <= PI, "{a} -> {w}");
            assert!(approx_eq((a - w).sin(), 0.0, 1e-4) && approx_eq((a - w).cos(), 1.0, 1e-4));
            let p = wrap_angle_positive(a);
            assert!((0.0..TAU).contains(&p), "{a} -> {p}");
        }
        assert!(approx_eq(angle_delta(3.0, -3.0), TAU - 6.0, 1e-5));
        assert!(approx_eq(lerp_angle(3.0, -3.0, 0.5), wrap_angle(3.0 + (TAU - 6.0) * 0.5), 1e-5));
    }

    #[test]
    fn approximate_comparisons() {
        assert!(approx_eq(1.0, 1.0 + 1e-7, 1e-6));
        assert!(!approx_eq(1.0, 1.1, 1e-6));
        assert!(!approx_eq(f32::NAN, f32::NAN, 1.0));
        assert!(approx_eq_f64(1.0, 1.0 + 1e-13, 1e-12));
        assert!(approx_eq_rel(1.0e6, 1.0e6 + 1.0, 1e-5, 0.0));
        assert!(!approx_eq_rel(1.0e-6, 2.0e-6, 1e-5, 0.0));
        assert!(approx_eq_rel(1.0e-9, 2.0e-9, 1e-5, 1e-6));
        assert!(approx_eq_rel_f64(1.0e12, 1.0e12 + 1.0, 1e-9, 0.0));
    }
}
