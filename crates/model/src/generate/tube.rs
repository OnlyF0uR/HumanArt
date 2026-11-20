//! Frames along limb axes and the slot-angle bookkeeping of ring tubes.
//!
//! A ring's vertices are its *slots*. A tube keeps the slot order of the ring it starts from, and
//! each later ring places slot `j` at an angle around the tube axis, measured in a frame that is
//! carried along the axis without twist. Sides are handled by mirrored frames: a right-side frame is
//! the mirror image of the left one (so it is left-handed), angles mean the same anatomical
//! direction on both sides, and slot order comes from the start ring, which is counter-clockwise on
//! both sides by construction.

use std::f32::consts::{PI, TAU};

use humanart_math::Vec3;

use super::cage::Cage;

/// A frame across a tube: `w` along the axis, `u` and `v` across it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    /// Origin on the axis.
    pub o: Vec3,
    /// First cross axis (angle 0).
    pub u: Vec3,
    /// Second cross axis (angle 90 degrees).
    pub v: Vec3,
    /// The axis direction.
    pub w: Vec3,
}

impl Frame {
    /// A frame at `o` with axis `w` and first cross axis as close to `u_hint` as possible;
    /// `side < 0` mirrors the handedness (`v = u x w` instead of `w x u`).
    pub fn new(o: Vec3, w: Vec3, u_hint: Vec3, side: f32) -> Self {
        let w = w.normalize();
        let u = (u_hint - w * u_hint.dot(w)).normalize();
        let v = if side >= 0.0 { w.cross(u) } else { u.cross(w) };
        Self { o, u, v, w }
    }

    /// The point at angle `theta` with radii `a` (along `u`) and `b` (along `v`).
    pub fn at(&self, theta: f32, a: f32, b: f32) -> Vec3 {
        self.o + self.u * (a * theta.cos()) + self.v * (b * theta.sin())
    }

    /// The point with local coordinates `(x, y, z)` (along `u`, `v`, `w`).
    pub fn local(&self, x: f32, y: f32, z: f32) -> Vec3 {
        self.o + self.u * x + self.v * y + self.w * z
    }

    /// The angle of `p` around the axis.
    pub fn angle(&self, p: Vec3) -> f32 {
        let d = p - self.o;
        d.dot(self.v).atan2(d.dot(self.u))
    }

    /// This frame carried to `o` with axis `w` by the smallest rotation (no twist), keeping its
    /// handedness.
    pub fn carry(&self, o: Vec3, w: Vec3) -> Self {
        let w = w.normalize();
        let u = (self.u - w * self.u.dot(w)).normalize();
        let right_handed = self.u.cross(self.v).dot(self.w) > 0.0;
        let v = if right_handed { w.cross(u) } else { u.cross(w) };
        Self { o, u, v, w }
    }
}

/// Makes a sequence of angles continuous (steps between `-PI` and `PI`).
pub fn unwrap(angles: &mut [f32]) {
    for i in 1..angles.len() {
        while angles[i] - angles[i - 1] > PI {
            angles[i] -= TAU;
        }
        while angles[i] - angles[i - 1] < -PI {
            angles[i] += TAU;
        }
    }
}

/// The unwrapped angles of the ring's vertices in `frame`.
pub fn ring_angles(cage: &Cage, ring: &[u32], frame: &Frame) -> Vec<f32> {
    let mut a: Vec<f32> = ring.iter().map(|&v| frame.angle(cage.pos[v as usize])).collect();
    unwrap(&mut a);
    a
}

/// Assigns pattern angles (one full turn, increasing) to the slots of a ring whose unwrapped angles
/// are `start` (increasing or decreasing): returns, per slot, the pattern index it gets and its
/// angle, unwrapped near the slot's start angle. The assignment keeps the cyclic order and picks
/// the rotation with the least total angle change.
pub fn match_pattern(start: &[f32], pattern: &[f32]) -> Vec<(usize, f32)> {
    let n = start.len();
    let increasing = n < 2 || start[n - 1] >= start[0];
    let mut best: Option<(f32, Vec<(usize, f32)>)> = None;
    for shift in 0..pattern.len() {
        let mut out = Vec::with_capacity(n);
        let mut cost = 0.0;
        for (j, &s) in start.iter().enumerate() {
            let k =
                if increasing { (shift + j) % pattern.len() } else { (shift + pattern.len() * n - j) % pattern.len() };
            let mut a = pattern[k];
            while a - s > PI {
                a -= TAU;
            }
            while a - s < -PI {
                a += TAU;
            }
            cost += (a - s).abs();
            out.push((k, a));
        }
        if best.as_ref().is_none_or(|b| cost < b.0) {
            best = Some((cost, out));
        }
    }
    best.map(|b| b.1).unwrap_or_default()
}

/// Smoothstep on `[0, 1]`.
pub fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation.
pub fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Piecewise-linear lookup in a table of `(x, y)` sorted by `x`, clamped at the ends.
pub fn lookup(table: &[(f32, f32)], x: f32) -> f32 {
    let Some(&(x0, y0)) = table.first() else { return 0.0 };
    if x <= x0 {
        return y0;
    }
    for w in table.windows(2) {
        let ((a, ya), (b, yb)) = (w[0], w[1]);
        if x <= b {
            let t = if b > a { (x - a) / (b - a) } else { 1.0 };
            return mix(ya, yb, t);
        }
    }
    table.last().map_or(0.0, |l| l.1)
}

/// Smooth (Catmull-Rom) lookup in a table of `(x, y)` sorted by `x`, clamped at the ends.
pub fn spline(table: &[(f32, f32)], x: f32) -> f32 {
    let n = table.len();
    if n < 3 {
        return lookup(table, x);
    }
    if x <= table[0].0 {
        return table[0].1;
    }
    if x >= table[n - 1].0 {
        return table[n - 1].1;
    }
    let i = table.windows(2).position(|w| x <= w[1].0).unwrap_or(n - 2);
    let (x1, y1) = table[i];
    let (x2, y2) = table[i + 1];
    let y0 = if i > 0 { table[i - 1].1 } else { y1 - (y2 - y1) };
    let y3 = if i + 2 < n { table[i + 2].1 } else { y2 + (y2 - y1) };
    let t = if x2 > x1 { (x - x1) / (x2 - x1) } else { 0.0 };
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * (2.0 * y1
        + (-y0 + y2) * t
        + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t2
        + (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t3)
}

/// The signed turning of a ring around `axis` (positive when counter-clockwise).
pub fn winding(cage: &Cage, ring: &[u32], axis: Vec3) -> f32 {
    let c = ring.iter().fold(Vec3::ZERO, |s, &v| s + cage.pos[v as usize]) / ring.len().max(1) as f32;
    let mut n = Vec3::ZERO;
    for i in 0..ring.len() {
        let (a, b) = (cage.pos[ring[i] as usize] - c, cage.pos[ring[(i + 1) % ring.len()] as usize] - c);
        n += a.cross(b);
    }
    n.dot(axis)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_mirror_and_carry() {
        let f = Frame::new(Vec3::ZERO, Vec3::X, Vec3::Y, 1.0);
        assert!((f.u.cross(f.v) - f.w).length() < 1e-6);
        let m = Frame::new(Vec3::ZERO, -Vec3::X, Vec3::Y, -1.0);
        // The mirrored frame's point at the same angle is the mirror image.
        let (p, q) = (f.at(0.7, 1.0, 2.0), m.at(0.7, 1.0, 2.0));
        assert!((p.x + q.x).abs() < 1e-6 && (p.y - q.y).abs() < 1e-6 && (p.z - q.z).abs() < 1e-6);
        let c = m.carry(Vec3::ONE, Vec3::new(-1.0, -1.0, 0.0));
        assert!(c.u.cross(c.v).dot(c.w) < 0.0);
    }

    #[test]
    fn pattern_matching_follows_direction() {
        let pattern: Vec<f32> = (0..4).map(|k| k as f32 * TAU / 4.0).collect();
        let up: Vec<f32> = (0..4).map(|k| 0.1 + k as f32 * TAU / 4.0).collect();
        let m = match_pattern(&up, &pattern);
        assert_eq!(m.iter().map(|x| x.0).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        let down: Vec<f32> = (0..4).map(|k| 0.1 + PI / 2.0 - k as f32 * TAU / 4.0).collect();
        let m = match_pattern(&down, &pattern);
        assert_eq!(m.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 0, 3, 2]);
    }

    #[test]
    fn spline_interpolates_knots() {
        let t = [(0.0, 1.0), (1.0, 3.0), (2.0, 2.0), (4.0, 0.0)];
        for &(x, y) in &t {
            assert!((spline(&t, x) - y).abs() < 1e-5);
        }
        assert!((lookup(&t, 3.0) - 1.0).abs() < 1e-6);
    }
}
