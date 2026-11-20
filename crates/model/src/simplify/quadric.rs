//! Error quadrics in `f64` for the LOD generator `[MOD-05]`.
//!
//! [`Quadric`] is the Garland–Heckbert plane quadric: the area-weighted sum of squared distances to
//! a set of planes, `Q(p) = pᵀAp + 2bᵀp + c`. [`AttrQuadric`] is Hoppe's attribute quadric
//! ("New quadric metric for simplifying meshes with appearance attributes", 1999) for two scalar
//! attributes (the UV coordinates, scaled to metres): per triangle, each attribute is a linear
//! function `g·p + d` over the triangle's plane, and the quadric sums
//! `area * (g·p + d - s)²` over attributes, so it measures how far the attribute value `s` kept at a
//! point strays from the attribute fields of the triangles merged into it.
//!
//! Positions are relative to the mesh centroid so the `f64` sums stay well conditioned.

/// A position or direction in `f64`.
pub(crate) type D3 = [f64; 3];

/// `a - b`.
#[inline]
pub(crate) fn sub(a: D3, b: D3) -> D3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `a · b`.
#[inline]
pub(crate) fn dot(a: D3, b: D3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `a × b`.
#[inline]
pub(crate) fn cross(a: D3, b: D3) -> D3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// `a * s`.
#[inline]
pub(crate) fn scale(a: D3, s: f64) -> D3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// `|a|`.
#[inline]
pub(crate) fn length(a: D3) -> f64 {
    dot(a, a).sqrt()
}

/// A symmetric 3x3 matrix as `[xx, xy, xz, yy, yz, zz]`.
type Sym3 = [f64; 6];

#[inline]
fn sym_outer(a: D3, w: f64) -> Sym3 {
    [w * a[0] * a[0], w * a[0] * a[1], w * a[0] * a[2], w * a[1] * a[1], w * a[1] * a[2], w * a[2] * a[2]]
}

#[inline]
fn sym_eval(m: &Sym3, p: D3) -> f64 {
    m[0] * p[0] * p[0]
        + m[3] * p[1] * p[1]
        + m[5] * p[2] * p[2]
        + 2.0 * (m[1] * p[0] * p[1] + m[2] * p[0] * p[2] + m[4] * p[1] * p[2])
}

#[inline]
fn add6(a: &mut Sym3, b: &Sym3) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// The Garland–Heckbert plane quadric `pᵀAp + 2bᵀp + c`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Quadric {
    a: Sym3,
    b: D3,
    c: f64,
}

impl Quadric {
    /// The quadric of the plane through `point` with unit normal `normal`, weighted by `weight`.
    pub(crate) fn plane(point: D3, normal: D3, weight: f64) -> Self {
        let d = -dot(normal, point);
        Self { a: sym_outer(normal, weight), b: scale(normal, weight * d), c: weight * d * d }
    }

    /// Adds `other` to this quadric.
    pub(crate) fn add(&mut self, other: &Self) {
        add6(&mut self.a, &other.a);
        for (a, b) in self.b.iter_mut().zip(&other.b) {
            *a += b;
        }
        self.c += other.c;
    }

    /// The weighted squared distance sum at `p` (clamped at 0 against rounding).
    pub(crate) fn eval(&self, p: D3) -> f64 {
        (sym_eval(&self.a, p) + 2.0 * dot(self.b, p) + self.c).max(0.0)
    }
}

/// Hoppe's attribute quadric for two scalar attributes (see the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct AttrQuadric {
    /// `Σ w g gᵀ` over attributes and triangles.
    a: Sym3,
    /// Per attribute `Σ w g`.
    g: [D3; 2],
    /// `Σ w` (the coefficient of `s²`).
    w: f64,
    /// `Σ w d g` over attributes.
    bp: D3,
    /// Per attribute `Σ w d`.
    bs: [f64; 2],
    /// `Σ w d²` over attributes.
    c: f64,
}

impl AttrQuadric {
    /// The quadric of one triangle with corner positions `p` and attribute pairs `s`, weighted by
    /// `weight` (its area). A degenerate triangle (`|n|² <= min_n2`) contributes nothing.
    pub(crate) fn triangle(p: [D3; 3], s: [[f64; 2]; 3], weight: f64, min_n2: f64) -> Self {
        let (e1, e2) = (sub(p[1], p[0]), sub(p[2], p[0]));
        let n = cross(e1, e2);
        let n2 = dot(n, n);
        let mut q = Self::default();
        if n2.is_nan() || n2 <= min_n2 || weight.is_nan() || weight <= 0.0 {
            return q;
        }
        let (u, v) = (cross(e2, n), cross(n, e1));
        for (j, (qg, qbs)) in q.g.iter_mut().zip(&mut q.bs).enumerate() {
            let (ds1, ds2) = (s[1][j] - s[0][j], s[2][j] - s[0][j]);
            let g = [(ds1 * u[0] + ds2 * v[0]) / n2, (ds1 * u[1] + ds2 * v[1]) / n2, (ds1 * u[2] + ds2 * v[2]) / n2];
            let d = s[0][j] - dot(g, p[0]);
            add6(&mut q.a, &sym_outer(g, weight));
            *qg = scale(g, weight);
            for (b, gk) in q.bp.iter_mut().zip(g) {
                *b += weight * d * gk;
            }
            *qbs = weight * d;
            q.c += weight * d * d;
        }
        // The s² coefficient is the same for both attributes.
        q.w = weight;
        q
    }

    /// Adds `other` to this quadric.
    pub(crate) fn add(&mut self, other: &Self) {
        add6(&mut self.a, &other.a);
        for (g, o) in self.g.iter_mut().zip(&other.g) {
            for (a, b) in g.iter_mut().zip(o) {
                *a += b;
            }
        }
        for (a, b) in self.bs.iter_mut().zip(&other.bs) {
            *a += b;
        }
        for (a, b) in self.bp.iter_mut().zip(&other.bp) {
            *a += b;
        }
        self.w += other.w;
        self.c += other.c;
    }

    /// The weighted squared attribute deviation at position `p` with attribute values `s`.
    pub(crate) fn eval(&self, p: D3, s: [f64; 2]) -> f64 {
        let mut e = sym_eval(&self.a, p) + 2.0 * dot(self.bp, p) + self.c;
        for ((&sj, &g), &bs) in s.iter().zip(&self.g).zip(&self.bs) {
            e += -2.0 * sj * dot(g, p) + self.w * sj * sj - 2.0 * bs * sj;
        }
        e.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_quadric_measures_squared_distance() {
        let mut q = Quadric::plane([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 2.0);
        assert!((q.eval([5.0, -3.0, 1.0])).abs() < 1e-12);
        assert!((q.eval([0.0, 0.0, 4.0]) - 18.0).abs() < 1e-12);
        q.add(&Quadric::plane([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0));
        assert!((q.eval([3.0, 0.0, 1.0]) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn attribute_quadric_is_zero_on_the_linear_field() {
        // s0 = x, s1 = 2y + 1 over a triangle in the z = 0 plane.
        let p = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let f = |q: D3| [q[0], 2.0 * q[1] + 1.0];
        let q = AttrQuadric::triangle(p, p.map(f), 0.5, 1e-20);
        for probe in [[0.3, 0.3, 0.0], [2.0, -1.0, 0.0], [0.1, 0.2, 0.7]] {
            // Off-plane displacement does not change the in-plane gradient fields.
            assert!(q.eval(probe, f(probe)) < 1e-12, "{probe:?}");
        }
        let e = q.eval([0.3, 0.3, 0.0], [0.3 + 0.1, 1.6]);
        assert!((e - 0.5 * 0.01).abs() < 1e-12, "{e}");
        let mut sum = q;
        sum.add(&q);
        assert!((sum.eval([0.3, 0.3, 0.0], [0.4, 1.6]) - 0.01).abs() < 1e-12);
        // Degenerate triangles contribute nothing.
        let flat = AttrQuadric::triangle([p[0], p[1], p[1]], [[0.0; 2]; 3], 1.0, 1e-20);
        assert_eq!(flat, AttrQuadric::default());
    }
}
