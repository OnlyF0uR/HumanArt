//! Spline and curve utilities: cubic Bézier, Hermite and Catmull-Rom segments (uniform,
//! centripetal, chordal), arc-length tables and resampling, polyline resampling (brush stamps,
//! strand control points `[HAIR-01]`), rotation-minimising frames along strands, a monotone 1D
//! interpolant for response curves and the smoothstep family.
//!
//! The segment functions are generic over [`CurvePoint`] (`f32`, `Vec2`, `Vec3`, `Vec4`), resolved
//! at compile time: no trait objects. Nothing here allocates; functions that fill an output slice
//! fill as much of it as their input allows and return the count.

use core::ops::{Add, Mul, Sub};

use super::vec::{Vec2, Vec3, Vec4};

/// A value curves can be built from: an affine space with a distance.
pub trait CurvePoint: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<f32, Output = Self> {
    /// The Euclidean distance to `other`.
    fn distance_to(self, other: Self) -> f32;
}

impl CurvePoint for f32 {
    #[inline]
    fn distance_to(self, other: Self) -> f32 {
        (self - other).abs()
    }
}

impl CurvePoint for Vec2 {
    #[inline]
    fn distance_to(self, other: Self) -> f32 {
        self.distance(other)
    }
}

impl CurvePoint for Vec3 {
    #[inline]
    fn distance_to(self, other: Self) -> f32 {
        self.distance(other)
    }
}

impl CurvePoint for Vec4 {
    #[inline]
    fn distance_to(self, other: Self) -> f32 {
        self.distance(other)
    }
}

/// `a + (b - a) * t`.
#[inline]
fn mix<P: CurvePoint>(a: P, b: P, t: f32) -> P {
    a + (b - a) * t
}

/// The cubic Bézier curve with control points `p` at `t` in `[0, 1]` (de Casteljau, stable).
#[inline]
pub fn cubic_bezier<P: CurvePoint>(p: [P; 4], t: f32) -> P {
    let (a, b, c) = (mix(p[0], p[1], t), mix(p[1], p[2], t), mix(p[2], p[3], t));
    mix(mix(a, b, t), mix(b, c, t), t)
}

/// The derivative of [`cubic_bezier`] with respect to `t`.
#[inline]
pub fn cubic_bezier_derivative<P: CurvePoint>(p: [P; 4], t: f32) -> P {
    let (a, b, c) = (p[1] - p[0], p[2] - p[1], p[3] - p[2]);
    mix(mix(a, b, t), mix(b, c, t), t) * 3.0
}

/// Splits a cubic Bézier curve at `t` into two whose union is the original.
#[inline]
pub fn cubic_bezier_split<P: CurvePoint>(p: [P; 4], t: f32) -> ([P; 4], [P; 4]) {
    let (a, b, c) = (mix(p[0], p[1], t), mix(p[1], p[2], t), mix(p[2], p[3], t));
    let (d, e) = (mix(a, b, t), mix(b, c, t));
    let m = mix(d, e, t);
    ([p[0], a, d, m], [m, e, c, p[3]])
}

/// The cubic Hermite curve from `p0` (tangent `m0`) to `p1` (tangent `m1`) at `t` in `[0, 1]`.
#[inline]
pub fn hermite<P: CurvePoint>(p0: P, m0: P, p1: P, m1: P, t: f32) -> P {
    let (t2, t3) = (t * t, t * t * t);
    p0 * (2.0 * t3 - 3.0 * t2 + 1.0) + m0 * (t3 - 2.0 * t2 + t) + p1 * (3.0 * t2 - 2.0 * t3) + m1 * (t3 - t2)
}

/// The derivative of [`hermite`] with respect to `t`.
#[inline]
pub fn hermite_derivative<P: CurvePoint>(p0: P, m0: P, p1: P, m1: P, t: f32) -> P {
    let t2 = t * t;
    p0 * (6.0 * t2 - 6.0 * t) + m0 * (3.0 * t2 - 4.0 * t + 1.0) + p1 * (6.0 * t - 6.0 * t2) + m1 * (3.0 * t2 - 2.0 * t)
}

/// The Catmull-Rom parameterisation exponent for uniform knots: fast, may cusp and self-intersect.
pub const CATMULL_ROM_UNIFORM: f32 = 0.0;
/// The exponent for centripetal knots: no cusps or self-intersections within a segment (Yuksel et
/// al. 2011). The default for strands and strokes.
pub const CATMULL_ROM_CENTRIPETAL: f32 = 0.5;
/// The exponent for chordal knots.
pub const CATMULL_ROM_CHORDAL: f32 = 1.0;

/// The Hermite tangents (scaled to the segment's parameter range) of the Catmull-Rom segment from
/// `p[1]` to `p[2]` with knot exponent `alpha`. Coincident points get the neighbouring interval,
/// so the result is always finite for finite input.
#[inline]
fn catmull_rom_tangents<P: CurvePoint>(p: [P; 4], alpha: f32) -> (P, P) {
    let knot = |a: P, b: P| a.distance_to(b).powf(alpha);
    let usable = |dt: f32| dt > 1e-12 && dt.is_finite();
    let dt1 = Some(knot(p[1], p[2])).filter(|&d| usable(d)).unwrap_or(1.0);
    let dt0 = Some(knot(p[0], p[1])).filter(|&d| usable(d)).unwrap_or(dt1);
    let dt2 = Some(knot(p[2], p[3])).filter(|&d| usable(d)).unwrap_or(dt1);
    let m1 = (p[1] - p[0]) * (1.0 / dt0) - (p[2] - p[0]) * (1.0 / (dt0 + dt1)) + (p[2] - p[1]) * (1.0 / dt1);
    let m2 = (p[2] - p[1]) * (1.0 / dt1) - (p[3] - p[1]) * (1.0 / (dt1 + dt2)) + (p[3] - p[2]) * (1.0 / dt2);
    (m1 * dt1, m2 * dt1)
}

/// The Catmull-Rom segment between `p[1]` (at `t = 0`) and `p[2]` (at `t = 1`), shaped by the
/// neighbours `p[0]` and `p[3]`, with knot exponent `alpha` ([`CATMULL_ROM_UNIFORM`],
/// [`CATMULL_ROM_CENTRIPETAL`], [`CATMULL_ROM_CHORDAL`] or anything between).
#[inline]
pub fn catmull_rom<P: CurvePoint>(p: [P; 4], alpha: f32, t: f32) -> P {
    let (m1, m2) = catmull_rom_tangents(p, alpha);
    hermite(p[1], m1, p[2], m2, t)
}

/// The derivative of [`catmull_rom`] with respect to `t`.
#[inline]
pub fn catmull_rom_derivative<P: CurvePoint>(p: [P; 4], alpha: f32, t: f32) -> P {
    let (m1, m2) = catmull_rom_tangents(p, alpha);
    hermite_derivative(p[1], m1, p[2], m2, t)
}

/// The Bézier control points of the Catmull-Rom segment `p[1]`–`p[2]`, for renderers and
/// subdivision that work on Bézier form.
#[inline]
pub fn catmull_rom_to_bezier<P: CurvePoint>(p: [P; 4], alpha: f32) -> [P; 4] {
    let (m1, m2) = catmull_rom_tangents(p, alpha);
    [p[1], p[1] + m1 * (1.0 / 3.0), p[2] - m2 * (1.0 / 3.0), p[2]]
}

/// The four control points of segment `segment` of the Catmull-Rom spline through `points`, with
/// the missing outer neighbours reflected (`2 p₀ - p₁`). `None` when there is no such segment.
#[inline]
pub fn catmull_rom_segment<P: CurvePoint>(points: &[P], segment: usize) -> Option<[P; 4]> {
    let p1 = *points.get(segment)?;
    let p2 = *points.get(segment + 1)?;
    let p0 = match segment.checked_sub(1).and_then(|i| points.get(i)) {
        Some(&p) => p,
        None => p1 + (p1 - p2),
    };
    let p3 = points.get(segment + 2).copied().unwrap_or(p2 + (p2 - p1));
    Some([p0, p1, p2, p3])
}

/// The Catmull-Rom spline through every point of `points` at the knot-index parameter `u` in
/// `[0, points.len() - 1]` (point `i` at `u = i`; clamped). One point gives that point; no points
/// give `None`.
pub fn catmull_rom_spline<P: CurvePoint>(points: &[P], alpha: f32, u: f32) -> Option<P> {
    let last = points.len().checked_sub(1)?;
    if last == 0 {
        return points.first().copied();
    }
    let u = if u.is_nan() { 0.0 } else { u.clamp(0.0, last as f32) };
    let segment = (u.floor() as usize).min(last - 1);
    let p = catmull_rom_segment(points, segment)?;
    Some(catmull_rom(p, alpha, u - segment as f32))
}

/// The length of the polyline through `points`.
pub fn polyline_length<P: CurvePoint>(points: &[P]) -> f32 {
    points.windows(2).map(|w| w[0].distance_to(w[1])).sum()
}

/// The point at arc length `s` along the polyline, walking forward from segment `*segment` whose
/// start lies at arc length `*segment_start` (both advanced). Clamps to the last point.
fn walk_polyline<P: CurvePoint>(points: &[P], s: f32, segment: &mut usize, segment_start: &mut f32) -> Option<P> {
    loop {
        let a = *points.get(*segment)?;
        let Some(&b) = points.get(*segment + 1) else {
            return Some(a);
        };
        let len = a.distance_to(b);
        if s <= *segment_start + len || *segment + 2 >= points.len() {
            let t = if len > 0.0 { ((s - *segment_start) / len).clamp(0.0, 1.0) } else { 1.0 };
            return Some(mix(a, b, t));
        }
        *segment_start += len;
        *segment += 1;
    }
}

/// Fills `out` with points evenly spaced by arc length along the polyline through `points`, the
/// first at its start and the last at its end (strand control-point resampling). Returns
/// `out.len()`, or 0 (leaving `out` untouched) when there are no points; a single point fills
/// every slot with it. One pass over both slices.
pub fn resample_polyline<P: CurvePoint>(points: &[P], out: &mut [P]) -> usize {
    let Some(&first) = points.first() else {
        return 0;
    };
    let total = polyline_length(points);
    let steps = out.len().saturating_sub(1).max(1) as f32;
    let (mut segment, mut start) = (0, 0.0);
    for (i, o) in out.iter_mut().enumerate() {
        let s = total * (i as f32 / steps);
        *o = walk_polyline(points, s, &mut segment, &mut start).unwrap_or(first);
    }
    out.len()
}

/// Writes the points at arc lengths `offset`, `offset + spacing`, `offset + 2 spacing`, ... up to
/// the end of the polyline into `out` (brush stamps along a stroke). Returns the count written,
/// limited by `out.len()`. A non-positive or non-finite `spacing` or `offset`, or no points,
/// write nothing. The next stamp of a continuing stroke is at
/// `offset + count * spacing - polyline_length(points)` along the next piece.
pub fn resample_polyline_spacing<P: CurvePoint>(points: &[P], spacing: f32, offset: f32, out: &mut [P]) -> usize {
    if !(spacing > 0.0 && spacing.is_finite() && offset >= 0.0 && offset.is_finite()) || points.is_empty() {
        return 0;
    }
    let total = polyline_length(points);
    let (mut segment, mut start) = (0, 0.0);
    let mut count = 0;
    for o in out.iter_mut() {
        let s = offset + count as f32 * spacing;
        if s > total {
            break;
        }
        match walk_polyline(points, s, &mut segment, &mut start) {
            Some(p) => *o = p,
            None => break,
        }
        count += 1;
    }
    count
}

/// Fills `table` with the cumulative arc length of the curve `eval` (over `t` in `[0, 1]`) at the
/// parameters `i / (table.len() - 1)`, measured along the chords between those samples, and
/// returns the total. Tables shorter than two entries are zero-filled and give 0. More entries
/// give a closer approximation (the chord sum converges from below).
pub fn arc_length_table<P: CurvePoint>(eval: impl Fn(f32) -> P, table: &mut [f32]) -> f32 {
    if table.len() < 2 {
        table.fill(0.0);
        return 0.0;
    }
    let steps = (table.len() - 1) as f32;
    let mut previous = eval(0.0);
    let mut total = 0.0;
    for (i, entry) in table.iter_mut().enumerate() {
        let p = eval(i as f32 / steps);
        total += p.distance_to(previous);
        previous = p;
        *entry = total;
    }
    total
}

/// The curve parameter in `[0, 1]` at arc length `s` according to a table from
/// [`arc_length_table`] (binary search, then linear between samples). Clamps outside the curve;
/// an empty table gives 0.
pub fn param_at_length(table: &[f32], s: f32) -> f32 {
    let Some(&total) = table.last() else {
        return 0.0;
    };
    if table.len() < 2 || s.is_nan() || s <= 0.0 {
        return 0.0;
    }
    if s >= total {
        return 1.0;
    }
    let steps = (table.len() - 1) as f32;
    // The first sample at or past `s`; at least 1 because table[0] = 0 < s.
    let i = table.partition_point(|&l| l < s).clamp(1, table.len() - 1);
    let (l0, l1) = (table[i - 1], table[i]);
    let frac = if l1 > l0 { (s - l0) / (l1 - l0) } else { 0.0 };
    ((i - 1) as f32 + frac) / steps
}

/// Fills `out` with points of the curve `eval` evenly spaced by arc length (first at `t = 0`, last
/// at `t = 1`) using a table from [`arc_length_table`]. Returns `out.len()`.
pub fn resample_curve<P: CurvePoint>(eval: impl Fn(f32) -> P, table: &[f32], out: &mut [P]) -> usize {
    let total = table.last().copied().unwrap_or(0.0);
    let steps = out.len().saturating_sub(1).max(1) as f32;
    for (i, o) in out.iter_mut().enumerate() {
        *o = eval(param_at_length(table, total * (i as f32 / steps)));
    }
    out.len()
}

/// Rotation-minimising frames along a polyline (double reflection, Wang et al. 2008): the frame
/// that turns as little as possible about the tangent, which keeps strand curl and ribbons from
/// twisting. Writes unit tangents (central differences) and unit normals perpendicular to them,
/// starting from `first_normal` made perpendicular to the first tangent (an arbitrary normal when
/// that is impossible). The binormal is `tangent × normal`. Returns the count written, the length
/// of the shortest slice. Repeated points reuse the previous tangent; a polyline of one point or
/// of coincident points gets the +Z tangent.
pub fn rotation_minimizing_frames(
    points: &[Vec3],
    first_normal: Vec3,
    tangents: &mut [Vec3],
    normals: &mut [Vec3],
) -> usize {
    let n = points.len().min(tangents.len()).min(normals.len());
    if n == 0 {
        return 0;
    }
    // Tangents by central differences (one-sided at the ends). Indices stay below n <= points.len().
    let (mut previous, mut first_real) = (None, None);
    for i in 0..n {
        let prev = points[i.saturating_sub(1)];
        let next = points[(i + 1).min(points.len() - 1)];
        tangents[i] = match (next - prev).try_normalize() {
            Some(t) => {
                first_real.get_or_insert(i);
                previous = Some(t);
                t
            }
            None => previous.unwrap_or(Vec3::Z),
        };
    }
    // Repeated start points inherit the first real direction.
    if let Some(f) = first_real {
        let t = tangents[f];
        tangents[..f].fill(t);
    }
    let t0 = tangents[0];
    let mut r = first_normal.reject_from(t0).try_normalize().unwrap_or_else(|| t0.any_orthonormal_vector());
    normals[0] = r;
    for i in 0..n - 1 {
        let (ti, tn) = (tangents[i], tangents[i + 1]);
        let v1 = points[i + 1] - points[i];
        let c1 = v1.dot(v1);
        let (r_l, t_l) =
            if c1 > 0.0 { (r - v1 * (2.0 / c1 * v1.dot(r)), ti - v1 * (2.0 / c1 * v1.dot(ti))) } else { (r, ti) };
        let v2 = tn - t_l;
        let c2 = v2.dot(v2);
        let next = if c2 > 0.0 { r_l - v2 * (2.0 / c2 * v2.dot(r_l)) } else { r_l };
        // Re-orthonormalise against rounding drift over long strands.
        r = next.reject_from(tn).try_normalize().unwrap_or_else(|| tn.any_orthonormal_vector());
        normals[i + 1] = r;
    }
    n
}

/// Monotone piecewise-cubic interpolation through the knots `(xs[i], ys[i])` (Steffen 1990): no
/// overshoot, monotone wherever the data is, continuous first derivative, and local (no solve,
/// no allocation). For response curves and falloff profiles.
///
/// Clamps `x` to the knot range. `None` when the slices are empty, differ in length, or the
/// knots around `x` are not strictly increasing in `x`.
pub fn monotone_cubic(xs: &[f32], ys: &[f32], x: f32) -> Option<f32> {
    if xs.len() != ys.len() || xs.is_empty() || x.is_nan() {
        return None;
    }
    let last = xs.len() - 1;
    if last == 0 {
        return ys.first().copied();
    }
    let (&x_first, &x_last) = (xs.first()?, xs.last()?);
    if !(x_first.is_finite() && x_last.is_finite() && x_first < x_last) {
        return None;
    }
    let x = x.clamp(x_first, x_last);
    // The segment [i, i + 1] holding x.
    let i = xs.partition_point(|&k| k <= x).saturating_sub(1).min(last - 1);
    let secant = |j: usize| -> Option<f32> {
        let h = xs.get(j + 1)? - xs.get(j)?;
        (h > 0.0).then(|| (ys[j + 1] - ys[j]) / h)
    };
    let h = xs[i + 1] - xs[i];
    secant(i)?;
    // Steffen's slope at knot k from the secants on either side.
    let slope = |k: usize| -> Option<f32> {
        let left = if k > 0 { Some(secant(k - 1)?) } else { None };
        let right = if k < last { Some(secant(k)?) } else { None };
        Some(match (left, right) {
            (Some(sl), Some(sr)) => {
                let (hl, hr) = (xs[k] - xs[k - 1], xs[k + 1] - xs[k]);
                let p = (sl * hr + sr * hl) / (hl + hr);
                if sl * sr <= 0.0 { 0.0 } else { sl.signum() * (2.0 * sl.abs().min(sr.abs())).min(p.abs()) }
            }
            (Some(one), None) | (None, Some(one)) => one,
            (None, None) => 0.0,
        })
    };
    let (m0, m1) = (slope(i)?, slope(i + 1)?);
    let t = (x - xs[i]) / h;
    Some(hermite(ys[i], m0 * h, ys[i + 1], m1 * h, t))
}

/// `0` below `edge0`, `1` above `edge1`, the cubic `3t² - 2t³` between (C¹). Equal edges give a
/// step at the edge.
#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = smooth_t(edge0, edge1, x);
    t * t * (3.0 - 2.0 * t)
}

/// `0` below `edge0`, `1` above `edge1`, the quintic `6t⁵ - 15t⁴ + 10t³` between (C²). Equal edges
/// give a step at the edge.
#[inline]
pub fn smootherstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = smooth_t(edge0, edge1, x);
    t * t * t * (t * (6.0 * t - 15.0) + 10.0)
}

/// `(x - edge0) / (edge1 - edge0)` clamped to `[0, 1]`, a step for equal edges, 0 for NaN.
#[inline]
fn smooth_t(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = (x - edge0) / (edge1 - edge0);
    if t.is_nan() { if x >= edge1 { 1.0 } else { 0.0 } } else { t.clamp(0.0, 1.0) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    fn rand_vec3(rng: &mut Rng, extent: f32) -> Vec3 {
        Vec3::new(rng.range_f32(-extent, extent), rng.range_f32(-extent, extent), rng.range_f32(-extent, extent))
    }

    #[test]
    fn bezier_and_hermite_segments() {
        let mut rng = Rng::new(0x80);
        for _ in 0..1000 {
            let p = [
                rand_vec3(&mut rng, 2.0),
                rand_vec3(&mut rng, 2.0),
                rand_vec3(&mut rng, 2.0),
                rand_vec3(&mut rng, 2.0),
            ];
            assert_eq!(cubic_bezier(p, 0.0), p[0]);
            assert!(cubic_bezier(p, 1.0).abs_diff_eq(p[3], 1e-6));
            let t = rng.f32();
            // The Bernstein form agrees.
            let u = 1.0 - t;
            let bern = p[0] * (u * u * u) + p[1] * (3.0 * u * u * t) + p[2] * (3.0 * u * t * t) + p[3] * (t * t * t);
            assert!(cubic_bezier(p, t).abs_diff_eq(bern, 1e-5));
            // The derivative matches central differences.
            let h = 1e-2;
            let fd = (cubic_bezier(p, t + h) - cubic_bezier(p, t - h)) / (2.0 * h);
            assert!(cubic_bezier_derivative(p, t).abs_diff_eq(fd, 2e-3));
            assert!(cubic_bezier_derivative(p, 0.0).abs_diff_eq((p[1] - p[0]) * 3.0, 1e-5));
            // Splitting reproduces both halves.
            let (left, right) = cubic_bezier_split(p, t);
            let s = rng.f32();
            assert!(cubic_bezier(left, s).abs_diff_eq(cubic_bezier(p, s * t), 1e-5));
            assert!(cubic_bezier(right, s).abs_diff_eq(cubic_bezier(p, t + s * (1.0 - t)), 1e-5));
            // Hermite: ends and end tangents.
            let (m0, m1) = (rand_vec3(&mut rng, 2.0), rand_vec3(&mut rng, 2.0));
            assert_eq!(hermite(p[0], m0, p[1], m1, 0.0), p[0]);
            assert!(hermite(p[0], m0, p[1], m1, 1.0).abs_diff_eq(p[1], 1e-6));
            assert!(hermite_derivative(p[0], m0, p[1], m1, 0.0).abs_diff_eq(m0, 1e-6));
            assert!(hermite_derivative(p[0], m0, p[1], m1, 1.0).abs_diff_eq(m1, 1e-5));
            let fd = (hermite(p[0], m0, p[1], m1, t + h) - hermite(p[0], m0, p[1], m1, t - h)) / (2.0 * h);
            assert!(hermite_derivative(p[0], m0, p[1], m1, t).abs_diff_eq(fd, 2e-3));
        }
        // Scalars work too.
        assert_eq!(cubic_bezier([0.0_f32, 1.0, 2.0, 3.0], 0.5), 1.5);
        assert_eq!(hermite(0.0_f32, 0.0, 1.0, 0.0, 0.5), 0.5);
    }

    #[test]
    fn catmull_rom_interpolates_with_any_knot_exponent() {
        let mut rng = Rng::new(0x81);
        for _ in 0..1000 {
            let p = [
                rand_vec3(&mut rng, 2.0),
                rand_vec3(&mut rng, 2.0),
                rand_vec3(&mut rng, 2.0),
                rand_vec3(&mut rng, 2.0),
            ];
            for alpha in [CATMULL_ROM_UNIFORM, CATMULL_ROM_CENTRIPETAL, CATMULL_ROM_CHORDAL] {
                assert!(catmull_rom(p, alpha, 0.0).abs_diff_eq(p[1], 1e-6));
                assert!(catmull_rom(p, alpha, 1.0).abs_diff_eq(p[2], 1e-5));
                let t = rng.f32();
                assert!(cubic_bezier(catmull_rom_to_bezier(p, alpha), t).abs_diff_eq(catmull_rom(p, alpha, t), 1e-4));
                let h = 1e-2;
                let fd = (catmull_rom(p, alpha, t + h) - catmull_rom(p, alpha, t - h)) / (2.0 * h);
                assert!(catmull_rom_derivative(p, alpha, t).abs_diff_eq(fd, 5e-3));
            }
            // Uniform: the classic tangent (p2 - p0) / 2.
            assert!(catmull_rom_derivative(p, 0.0, 0.0).abs_diff_eq((p[2] - p[0]) * 0.5, 1e-5));
            let t = rng.f32();
            let (t2, t3) = (t * t, t * t * t);
            let classic = (p[1] * 2.0
                + (p[2] - p[0]) * t
                + (p[0] * 2.0 - p[1] * 5.0 + p[2] * 4.0 - p[3]) * t2
                + (p[1] * 3.0 - p[0] - p[2] * 3.0 + p[3]) * t3)
                * 0.5;
            assert!(catmull_rom(p, 0.0, t).abs_diff_eq(classic, 1e-4));
        }
        // Coincident points stay finite.
        let q = [Vec3::ZERO, Vec3::ZERO, Vec3::X, Vec3::X];
        for alpha in [0.0, 0.5, 1.0] {
            for i in 0..=10 {
                assert!(catmull_rom(q, alpha, i as f32 / 10.0).is_finite());
            }
            assert!(catmull_rom([Vec3::ONE; 4], alpha, 0.5).abs_diff_eq(Vec3::ONE, 1e-6));
        }
        // Centripetal overshoots less than uniform on a short segment between long ones (where
        // uniform forms a loop).
        let hook =
            [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0), Vec3::new(1.1, 1.0, 0.0), Vec3::new(0.0, 0.1, 0.0)];
        let max_x =
            (0..=100).map(|i| catmull_rom(hook, CATMULL_ROM_CENTRIPETAL, i as f32 / 100.0).x).fold(0.0, f32::max);
        let max_x_uniform =
            (0..=100).map(|i| catmull_rom(hook, CATMULL_ROM_UNIFORM, i as f32 / 100.0).x).fold(0.0, f32::max);
        assert!(max_x < max_x_uniform, "{max_x} vs {max_x_uniform}");
    }

    #[test]
    fn hair_01_splines_pass_through_every_point() {
        let points = [
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.5, 0.2),
            Vec3::new(1.5, 1.0, 0.8),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        for alpha in [0.0, 0.5, 1.0] {
            for (i, &p) in points.iter().enumerate() {
                let got = catmull_rom_spline(&points, alpha, i as f32).expect("non-empty");
                assert!(got.abs_diff_eq(p, 1e-5), "{alpha} {i}");
            }
            // Continuous across segment joins.
            for i in 1..points.len() - 1 {
                let (a, b) = (
                    catmull_rom_spline(&points, alpha, i as f32 - 1e-4).expect("valid"),
                    catmull_rom_spline(&points, alpha, i as f32 + 1e-4).expect("valid"),
                );
                assert!(a.distance(b) < 1e-2);
            }
        }
        assert_eq!(catmull_rom_spline::<Vec3>(&[], 0.5, 0.3), None);
        assert_eq!(catmull_rom_spline(&[Vec3::X], 0.5, 0.3), Some(Vec3::X));
        assert_eq!(catmull_rom_spline(&points, 0.5, -3.0), Some(points[0]));
        assert!(catmull_rom_spline(&points, 0.5, 99.0).expect("valid").abs_diff_eq(points[4], 1e-5));
        assert_eq!(catmull_rom_spline(&points, 0.5, f32::NAN), Some(points[0]));
        assert_eq!(catmull_rom_segment(&points, 4), None);
        let seg = catmull_rom_segment(&points, 0).expect("valid");
        assert_eq!(seg[0], points[0] * 2.0 - points[1], "reflected start");
        let seg = catmull_rom_segment(&points, 3).expect("valid");
        assert_eq!(seg[3], points[4] * 2.0 - points[3], "reflected end");
        // Two points: a straight line.
        let two = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        assert!(catmull_rom_spline(&two, 0.5, 0.25).expect("valid").abs_diff_eq(Vec3::new(0.5, 0.0, 0.0), 1e-5));
    }

    #[test]
    fn polylines_resample_evenly() {
        let line = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(4.0, 2.0, 0.0),
        ];
        assert_eq!(polyline_length(&line), 6.0);
        let mut out = [Vec3::NAN; 7];
        assert_eq!(resample_polyline(&line, &mut out), 7);
        let want = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (3.0, 2.0), (4.0, 2.0)];
        for (o, (x, y)) in out.iter().zip(want) {
            assert!(o.abs_diff_eq(Vec3::new(x, y, 0.0), 1e-6), "{o:?}");
        }
        let mut one = [Vec3::NAN; 1];
        assert_eq!(resample_polyline(&line, &mut one), 1);
        assert_eq!(one[0], Vec3::ZERO);
        let mut same = [Vec3::NAN; 3];
        assert_eq!(resample_polyline(&[Vec3::X], &mut same), 3);
        assert_eq!(same, [Vec3::X; 3]);
        assert_eq!(resample_polyline::<Vec3>(&[], &mut same), 0);

        let mut stamps = [Vec3::NAN; 10];
        assert_eq!(resample_polyline_spacing(&line, 1.5, 0.5, &mut stamps), 4, "at 0.5, 2, 3.5, 5");
        assert!(stamps[1].abs_diff_eq(Vec3::new(1.0, 1.0, 0.0), 1e-6));
        assert!(stamps[3].abs_diff_eq(Vec3::new(3.0, 2.0, 0.0), 1e-6));
        assert!(stamps[4].is_nan());
        assert_eq!(resample_polyline_spacing(&line, 0.001, 0.0, &mut stamps), 10, "bounded by out");
        assert_eq!(resample_polyline_spacing(&line, 0.0, 0.0, &mut stamps), 0);
        assert_eq!(resample_polyline_spacing(&line, f32::NAN, 0.0, &mut stamps), 0);
        assert_eq!(resample_polyline_spacing(&line, 1.0, -1.0, &mut stamps), 0);
        assert_eq!(resample_polyline_spacing(&line, 1.0, 7.0, &mut stamps), 0);
        assert_eq!(resample_polyline_spacing(&[Vec2::ONE], 1.0, 0.0, &mut [Vec2::ZERO; 4]), 1);
        // Random polylines: consecutive samples are equally far apart along the line.
        let mut rng = Rng::new(0x82);
        for _ in 0..200 {
            let pts: Vec<Vec2> = (0..2 + rng.index(10)).map(|_| Vec2::new(rng.f32(), rng.f32())).collect();
            let total = polyline_length(&pts);
            let mut samples = [Vec2::ZERO; 33];
            resample_polyline(&pts, &mut samples);
            assert!(samples[0] == pts[0] && samples[32].abs_diff_eq(pts[pts.len() - 1], 1e-5));
            // Re-measuring the resampled polyline never exceeds the original length.
            assert!(polyline_length(&samples) <= total + 1e-4);
        }
    }

    #[test]
    fn arc_length_reparameterisation_is_uniform() {
        // A quarter circle as a Bézier with a very uneven parameterisation.
        let p = [Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.05), Vec2::new(0.3, 1.0), Vec2::new(0.0, 1.0)];
        let eval = |t: f32| cubic_bezier(p, t);
        let mut table = [0.0_f32; 257];
        let total = arc_length_table(eval, &mut table);
        assert!(table[0] == 0.0 && table.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(table[256], total);
        let mut out = [Vec2::ZERO; 11];
        assert_eq!(resample_curve(eval, &table, &mut out), 11);
        assert_eq!(out[0], p[0]);
        assert!(out[10].abs_diff_eq(p[3], 1e-6));
        // Each of the ten pieces between consecutive samples has a tenth of the length, measured
        // independently along a fine polyline.
        let piece = |u0: f32, u1: f32| {
            (0..200)
                .map(|k| {
                    eval(u0 + (u1 - u0) * k as f32 / 200.0).distance(eval(u0 + (u1 - u0) * (k + 1) as f32 / 200.0))
                })
                .sum::<f32>()
        };
        let params: Vec<f32> = (0..=10).map(|i| param_at_length(&table, total * i as f32 / 10.0)).collect();
        for (i, w) in params.windows(2).enumerate() {
            assert!(
                (piece(w[0], w[1]) - total / 10.0).abs() < 1e-3 * total,
                "{i}: {} vs {}",
                piece(w[0], w[1]),
                total / 10.0
            );
            assert!(out[i].abs_diff_eq(eval(w[0]), 1e-6));
        }
        // param_at_length inverts the table at its samples and clamps outside.
        for (i, &l) in table.iter().enumerate().skip(1) {
            assert!((param_at_length(&table, l) - i as f32 / 256.0).abs() < 1e-5);
        }
        assert_eq!((param_at_length(&table, -1.0), param_at_length(&table, total * 2.0)), (0.0, 1.0));
        assert_eq!(param_at_length(&table, f32::NAN), 0.0);
        assert_eq!(param_at_length(&[], 1.0), 0.0);
        let mut tiny = [5.0_f32; 1];
        assert_eq!(arc_length_table(eval, &mut tiny), 0.0);
        assert_eq!(tiny, [0.0]);
        // A curve standing still (zero length) resamples to its point.
        let still = |_t: f32| Vec3::ONE;
        let mut table = [0.0_f32; 9];
        assert_eq!(arc_length_table(still, &mut table), 0.0);
        let mut out = [Vec3::ZERO; 3];
        resample_curve(still, &table, &mut out);
        assert_eq!(out, [Vec3::ONE; 3]);
    }

    #[test]
    fn hair_01_rotation_minimizing_frames_do_not_twist() {
        // A planar curve: the frame normal stays the plane normal all along.
        let pts: Vec<Vec3> = (0..40)
            .map(|i| {
                let a = i as f32 * 0.15;
                Vec3::new(a.cos() * (1.0 + 0.1 * a), a.sin() * (1.0 + 0.1 * a), 0.0)
            })
            .collect();
        let (mut t, mut n) = (vec![Vec3::ZERO; 40], vec![Vec3::ZERO; 40]);
        // The first normal is made perpendicular to the first tangent.
        let first = Vec3::Z + (pts[1] - pts[0]).normalize() * 0.5;
        assert_eq!(rotation_minimizing_frames(&pts, first, &mut t, &mut n), 40);
        for (ti, ni) in t.iter().zip(&n) {
            assert!(ti.is_normalized() && ni.is_normalized() && ti.dot(*ni).abs() < 1e-5);
            assert!(ni.abs_diff_eq(Vec3::Z, 1e-4), "{ni:?}");
        }
        // A helix: frames stay orthonormal, and the RMF twist rate matches the analytic value
        // (the Frenet frame rotates relative to the RMF by the torsion).
        let mut rng = Rng::new(0x83);
        let pts: Vec<Vec3> = (0..200)
            .map(|i| {
                let a = i as f32 * 0.05;
                Vec3::new(a.cos(), a * 0.3, a.sin())
            })
            .collect();
        let (mut t, mut n) = (vec![Vec3::ZERO; 200], vec![Vec3::ZERO; 200]);
        rotation_minimizing_frames(&pts, rand_vec3(&mut rng, 1.0), &mut t, &mut n);
        for (ti, ni) in t.iter().zip(&n) {
            assert!(ti.is_normalized() && ni.is_normalized() && ti.dot(*ni).abs() < 1e-5);
        }
        // Adjacent normals differ only by the tangent turning: no rotation about the tangent.
        for i in 0..199 {
            let twist = n[i + 1].dot(t[i].cross(n[i]));
            // A Frenet frame would turn by the torsion, about 0.015 per step here.
            assert!(twist.abs() < 0.004, "{i}: {twist}");
        }
        // Degenerate input.
        let (mut t, mut n) = ([Vec3::NAN; 3], [Vec3::NAN; 3]);
        assert_eq!(rotation_minimizing_frames(&[Vec3::ONE], Vec3::Y, &mut t, &mut n), 1);
        assert_eq!((t[0], n[0]), (Vec3::Z, Vec3::Y));
        assert_eq!(
            rotation_minimizing_frames(&[Vec3::ONE, Vec3::ONE, Vec3::new(1.0, 1.0, 3.0)], Vec3::Z, &mut t, &mut n),
            3
        );
        assert!(t.iter().all(|t| t.abs_diff_eq(Vec3::Z, 1e-6)), "{t:?}");
        assert!(n.iter().all(|n| n.is_normalized() && n.z.abs() < 1e-6));
        assert_eq!(rotation_minimizing_frames(&[], Vec3::Y, &mut t, &mut n), 0);
        assert_eq!(rotation_minimizing_frames(&pts, Vec3::Y, &mut t[..2], &mut n), 2);
    }

    #[test]
    fn monotone_cubic_never_overshoots() {
        let xs = [0.0, 1.0, 2.0, 3.0, 5.0, 6.0];
        let ys = [0.0, 0.1, 0.1, 2.0, 2.1, 5.0];
        for (&x, &y) in xs.iter().zip(&ys) {
            assert!((monotone_cubic(&xs, &ys, x).expect("valid") - y).abs() < 1e-6);
        }
        let mut previous = f32::NEG_INFINITY;
        for i in 0..=600 {
            let x = i as f32 / 100.0;
            let y = monotone_cubic(&xs, &ys, x).expect("valid");
            assert!(y >= previous - 1e-6, "monotone at {x}");
            // Between two equal knots the curve is flat: no overshoot.
            if (1.0..=2.0).contains(&x) {
                assert!((y - 0.1).abs() < 1e-6);
            }
            previous = y;
        }
        // Continuous slope at an interior knot.
        let h = 1e-3;
        let left =
            (monotone_cubic(&xs, &ys, 3.0).expect("valid") - monotone_cubic(&xs, &ys, 3.0 - h).expect("valid")) / h;
        let right =
            (monotone_cubic(&xs, &ys, 3.0 + h).expect("valid") - monotone_cubic(&xs, &ys, 3.0).expect("valid")) / h;
        assert!((left - right).abs() < 0.05, "{left} {right}");
        // A straight line is reproduced exactly.
        assert!((monotone_cubic(&[0.0, 1.0, 3.0], &[1.0, 3.0, 7.0], 2.0).expect("valid") - 5.0).abs() < 1e-6);
        // Clamping and bad input.
        assert_eq!(monotone_cubic(&xs, &ys, -5.0), Some(0.0));
        assert_eq!(monotone_cubic(&xs, &ys, 50.0), Some(5.0));
        assert_eq!(monotone_cubic(&[2.0], &[7.0], 0.0), Some(7.0));
        assert_eq!(monotone_cubic(&[], &[], 0.0), None);
        assert_eq!(monotone_cubic(&xs, &ys[..3], 0.0), None);
        assert_eq!(monotone_cubic(&xs, &ys, f32::NAN), None);
        assert_eq!(monotone_cubic(&[0.0, 0.0], &[1.0, 2.0], 0.0), None, "repeated knot");
        assert_eq!(monotone_cubic(&[0.0, 1.0, 0.5], &[0.0, 1.0, 2.0], 0.9), None, "not increasing");
    }

    #[test]
    fn smoothsteps() {
        assert_eq!((smoothstep(0.0, 1.0, -1.0), smoothstep(0.0, 1.0, 0.5), smoothstep(0.0, 1.0, 2.0)), (0.0, 0.5, 1.0));
        assert_eq!(
            (smootherstep(0.0, 1.0, -1.0), smootherstep(0.0, 1.0, 0.5), smootherstep(0.0, 1.0, 2.0)),
            (0.0, 0.5, 1.0)
        );
        assert_eq!(smoothstep(2.0, 4.0, 3.0), 0.5);
        assert_eq!((smoothstep(1.0, 1.0, 0.5), smoothstep(1.0, 1.0, 1.0)), (0.0, 1.0), "equal edges: a step");
        assert_eq!(smoothstep(0.0, 1.0, f32::NAN), 0.0);
        assert!((smoothstep(0.0, 1.0, 0.25) - 0.156_25).abs() < 1e-7);
        assert!((smootherstep(0.0, 1.0, 0.25) - 0.103_515_63).abs() < 1e-7);
    }
}
