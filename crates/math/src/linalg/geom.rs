//! Geometric primitives and queries: rays, planes, axis-aligned boxes, spheres, the view frustum
//! (wgpu clip space, reversed Z), triangle helpers (barycentrics `[MOD-06]`, closest points) and a
//! watertight ray-triangle test with a CPU mesh ray cast for brush hit tests and picking
//! fallbacks (`[VP-01]`).
//!
//! Triangles are counter-clockwise when seen from the front: the face normal is
//! `(b - a) × (c - a)` (right-hand rule). Barycentric coordinates are `(wa, wb, wc)`, the weights of
//! `a`, `b`, `c`, summing to one. Nothing here panics on degenerate input: queries that have no
//! answer return `None`, and indices read from a mesh are checked.

use super::mat::Mat4;
use super::vec::{Vec2, Vec3, Vec4};

/// A half-line `origin + t * direction`, `t >= 0`. The constructors make `direction` unit length,
/// so `t` is a distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    /// The start point.
    pub origin: Vec3,
    /// The direction, unit length when built by a constructor.
    pub direction: Vec3,
}

impl Ray {
    /// The ray from `origin` towards `direction` (normalised), or `None` when the direction is zero
    /// or anything is not finite.
    #[inline]
    pub fn new(origin: Vec3, direction: Vec3) -> Option<Self> {
        let direction = direction.try_normalize()?;
        origin.is_finite().then_some(Self { origin, direction })
    }

    /// The ray from `from` through `to`, or `None` when the points coincide or are not finite.
    #[inline]
    pub fn between(from: Vec3, to: Vec3) -> Option<Self> {
        Self::new(from, to - from)
    }

    /// The camera ray through a point of normalised device coordinates (`x`, `y` in `[-1, 1]`, `y`
    /// up), given the INVERSE of the view-projection matrix. Works for the reversed-Z perspective
    /// projections (finite and infinite far plane) and orthographic ones: the ray starts on the
    /// near plane (`z = 1`) and passes through the point at `z = 0.5`. `None` when the matrix maps
    /// those points to infinity or onto each other.
    pub fn from_ndc(inverse_view_proj: Mat4, ndc: Vec2) -> Option<Self> {
        let near = inverse_view_proj.project_point3(Vec3::new(ndc.x, ndc.y, 1.0));
        let mid = inverse_view_proj.project_point3(Vec3::new(ndc.x, ndc.y, 0.5));
        Self::between(near, mid)
    }

    /// The point at parameter `t`.
    #[inline]
    pub fn at(self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// The parameter of the point of the ray closest to `p` (`0` when `p` is behind the origin).
    #[inline]
    pub fn closest_t(self, p: Vec3) -> f32 {
        ((p - self.origin).dot(self.direction) / self.direction.length_squared()).max(0.0)
    }

    /// The squared distance from `p` to the ray.
    #[inline]
    pub fn distance_squared_to_point(self, p: Vec3) -> f32 {
        self.at(self.closest_t(p)).distance_squared(p)
    }

    /// The watertight triangle tester for this ray (precomputes the per-ray part; reuse it for
    /// every triangle of a mesh).
    #[inline]
    pub fn triangle_tester(self) -> RayTriangleTester {
        RayTriangleTester::new(self)
    }

    /// The nearest intersection with the triangle `a`, `b`, `c` from either side; see
    /// [`RayTriangleTester::intersect`].
    #[inline]
    pub fn intersect_triangle(self, a: Vec3, b: Vec3, c: Vec3) -> Option<TriangleHit> {
        self.triangle_tester().intersect(a, b, c, f32::INFINITY)
    }
}

/// The plane of points `p` with `normal · p + d = 0`. The constructors make `normal` unit length,
/// so [`Plane::signed_distance`] is a distance (positive on the side the normal points to).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    /// The normal, unit length when built by a constructor.
    pub normal: Vec3,
    /// The offset: minus the distance of the plane from the origin along `normal`.
    pub d: f32,
}

impl Plane {
    /// The plane through `point` with the (normalised) `normal`, or `None` when the normal is zero
    /// or anything is not finite.
    #[inline]
    pub fn from_point_normal(point: Vec3, normal: Vec3) -> Option<Self> {
        let normal = normal.try_normalize()?;
        let d = -normal.dot(point);
        d.is_finite().then_some(Self { normal, d })
    }

    /// The plane through three points, its normal by the right-hand rule (counter-clockwise points
    /// face the normal). `None` when the points are collinear or not finite.
    #[inline]
    pub fn from_points(a: Vec3, b: Vec3, c: Vec3) -> Option<Self> {
        Self::from_point_normal(a, (b - a).cross(c - a))
    }

    /// The plane `v.xyz · p + v.w = 0` scaled so that its normal has unit length, or `None` when
    /// the normal is zero or not finite.
    #[inline]
    pub fn from_vec4(v: Vec4) -> Option<Self> {
        let len = v.xyz().length();
        (len > 0.0 && len.is_finite() && v.w.is_finite()).then(|| Self { normal: v.xyz() / len, d: v.w / len })
    }

    /// The signed distance of `p` (positive in front, on the normal's side).
    #[inline]
    pub fn signed_distance(self, p: Vec3) -> f32 {
        self.normal.dot(p) + self.d
    }

    /// The point of the plane closest to `p`.
    #[inline]
    pub fn project_point(self, p: Vec3) -> Vec3 {
        p - self.normal * self.signed_distance(p)
    }

    /// The ray parameter where the ray meets the plane, from either side, or `None` when the ray
    /// is parallel to the plane or the plane is behind it.
    #[inline]
    pub fn intersect_ray(self, ray: Ray) -> Option<f32> {
        let denom = self.normal.dot(ray.direction);
        let t = -self.signed_distance(ray.origin) / denom;
        (t >= 0.0 && t.is_finite()).then_some(t)
    }
}

/// An axis-aligned bounding box `[min, max]`. [`Aabb::EMPTY`] (min `+∞`, max `-∞`) is the identity
/// of [`Aabb::union`] and contains nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// The minimum corner.
    pub min: Vec3,
    /// The maximum corner.
    pub max: Vec3,
}

impl Aabb {
    /// The empty box.
    pub const EMPTY: Self = Self { min: Vec3::INFINITY, max: Vec3::NEG_INFINITY };

    /// The box with these corners (taken component-wise, so the order does not matter).
    #[inline]
    pub fn new(a: Vec3, b: Vec3) -> Self {
        Self { min: a.min(b), max: a.max(b) }
    }

    /// The box around a center with the given half extents (taken as absolute values).
    #[inline]
    pub fn from_center_half_extents(center: Vec3, half_extents: Vec3) -> Self {
        let h = half_extents.abs();
        Self { min: center - h, max: center + h }
    }

    /// The smallest box holding every point; [`Self::EMPTY`] for no points. NaN components are
    /// ignored.
    pub fn from_points(points: &[Vec3]) -> Self {
        points.iter().fold(Self::EMPTY, |b, &p| b.extend(p))
    }

    /// The smallest box holding every point of a structure-of-arrays position channel (as many
    /// points as the shortest slice holds); [`Self::EMPTY`] for no points.
    pub fn from_points_soa(x: &[f32], y: &[f32], z: &[f32]) -> Self {
        x.iter().zip(y).zip(z).fold(Self::EMPTY, |b, ((&x, &y), &z)| b.extend(Vec3::new(x, y, z)))
    }

    /// Whether the box holds no point.
    #[inline]
    pub fn is_empty(self) -> bool {
        !(self.min.x <= self.max.x && self.min.y <= self.max.y && self.min.z <= self.max.z)
    }

    /// The center (meaningless for the empty box).
    #[inline]
    pub fn center(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half the size along each axis.
    #[inline]
    pub fn half_extents(self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// The size along each axis; zero for the empty box.
    #[inline]
    pub fn size(self) -> Vec3 {
        if self.is_empty() { Vec3::ZERO } else { self.max - self.min }
    }

    /// The surface area (the cost metric of bounding volume hierarchies); zero for the empty box.
    #[inline]
    pub fn surface_area(self) -> f32 {
        let s = self.size();
        2.0 * (s.x * s.y + s.y * s.z + s.z * s.x)
    }

    /// The box grown to hold `p` (NaN components leave it unchanged).
    #[inline]
    pub fn extend(self, p: Vec3) -> Self {
        Self { min: self.min.min(p), max: self.max.max(p) }
    }

    /// The smallest box holding both.
    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self { min: self.min.min(other.min), max: self.max.max(other.max) }
    }

    /// The overlap of both (empty when they do not overlap).
    #[inline]
    pub fn intersection(self, other: Self) -> Self {
        let out = Self { min: self.min.max(other.min), max: self.max.min(other.max) };
        if out.is_empty() { Self::EMPTY } else { out }
    }

    /// The box grown by `margin` on every side.
    #[inline]
    pub fn expand(self, margin: f32) -> Self {
        if self.is_empty() {
            self
        } else {
            Self { min: self.min - Vec3::splat(margin), max: self.max + Vec3::splat(margin) }
        }
    }

    /// Whether `p` is inside or on the boundary.
    #[inline]
    pub fn contains_point(self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// Whether the boxes overlap or touch.
    #[inline]
    pub fn intersects(self, other: Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// The point of the box closest to `p` (`p` itself when inside). Meaningless for the empty box.
    #[inline]
    pub fn closest_point(self, p: Vec3) -> Vec3 {
        p.max(self.min).min(self.max)
    }

    /// The squared distance from `p` to the box (zero inside); infinite for the empty box.
    #[inline]
    pub fn distance_squared_to_point(self, p: Vec3) -> f32 {
        if self.is_empty() { f32::INFINITY } else { self.closest_point(p).distance_squared(p) }
    }

    /// The bounding box of this box transformed by the affine matrix `m` (Arvo's method: exact for
    /// the transformed corners). The empty box stays empty.
    pub fn transform(self, m: Mat4) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        let center = m.transform_point3(self.center());
        let h = self.half_extents();
        let [c0, c1, c2] = [m.cols[0].xyz(), m.cols[1].xyz(), m.cols[2].xyz()];
        let half = c0.abs() * h.x + c1.abs() * h.y + c2.abs() * h.z;
        Self { min: center - half, max: center + half }
    }

    /// The parameters `(entry, exit)` of the part of the ray inside the box (slab test), with
    /// `entry` clamped to zero when the origin is inside, or `None` when the ray misses.
    pub fn intersect_ray(self, ray: Ray) -> Option<(f32, f32)> {
        if self.is_empty() {
            return None;
        }
        let (mut entry, mut exit) = (0.0_f32, f32::INFINITY);
        let axes = ray
            .origin
            .to_array()
            .into_iter()
            .zip(ray.direction.to_array())
            .zip(self.min.to_array())
            .zip(self.max.to_array());
        for (((o, d), lo), hi) in axes {
            if d == 0.0 {
                // Parallel to this slab: all of the ray or none of it is between its planes.
                if o < lo || o > hi {
                    return None;
                }
                continue;
            }
            let (t1, t2) = ((lo - o) / d, (hi - o) / d);
            entry = entry.max(t1.min(t2));
            exit = exit.min(t1.max(t2));
        }
        (entry <= exit).then_some((entry, exit))
    }
}

/// A sphere (also used as a bounding sphere).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    /// The center.
    pub center: Vec3,
    /// The radius.
    pub radius: f32,
}

impl Sphere {
    /// A sphere from its center and radius.
    #[inline]
    pub const fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }

    /// A sphere holding every point: centered on their bounding box, with the largest distance as
    /// radius (conservative, not minimal). Zero radius at the origin for no points.
    pub fn from_points(points: &[Vec3]) -> Self {
        let b = Aabb::from_points(points);
        if b.is_empty() {
            return Self::new(Vec3::ZERO, 0.0);
        }
        let center = b.center();
        let r2 = points.iter().filter(|p| p.is_finite()).fold(0.0_f32, |r, &p| r.max(p.distance_squared(center)));
        Self::new(center, r2.sqrt())
    }

    /// The sphere through the corners of a box.
    #[inline]
    pub fn from_aabb(b: Aabb) -> Self {
        Self::new(b.center(), b.half_extents().length())
    }

    /// Whether `p` is inside or on the surface.
    #[inline]
    pub fn contains_point(self, p: Vec3) -> bool {
        p.distance_squared(self.center) <= self.radius * self.radius
    }

    /// Whether the spheres overlap or touch.
    #[inline]
    pub fn intersects(self, other: Self) -> bool {
        let r = self.radius + other.radius;
        self.center.distance_squared(other.center) <= r * r
    }

    /// Whether the sphere and the box overlap or touch.
    #[inline]
    pub fn intersects_aabb(self, b: Aabb) -> bool {
        b.distance_squared_to_point(self.center) <= self.radius * self.radius
    }

    /// The bounding sphere of this sphere under the affine matrix `m` (the radius grows by the
    /// largest axis scale).
    #[inline]
    pub fn transform(self, m: Mat4) -> Self {
        let scale = m.cols[0]
            .xyz()
            .length_squared()
            .max(m.cols[1].xyz().length_squared())
            .max(m.cols[2].xyz().length_squared());
        Self::new(m.transform_point3(self.center), self.radius * scale.sqrt())
    }

    /// The first ray parameter `t >= 0` on the surface (the exit point when the origin is inside),
    /// or `None` when the ray misses.
    pub fn intersect_ray(self, ray: Ray) -> Option<f32> {
        // Solve |o + t d - c|² = r² with the numerically stable form of the quadratic.
        let oc = ray.origin - self.center;
        let a = ray.direction.length_squared();
        let half_b = oc.dot(ray.direction);
        // The discriminant from the distance to the closest point (Hearn & Baker / Ray Tracing
        // Gems ch. 7): no cancellation for distant spheres.
        let perp = oc - ray.direction * (half_b / a);
        let disc = a * (self.radius * self.radius - perp.length_squared());
        if disc.is_nan() || disc < 0.0 || a == 0.0 {
            return None;
        }
        let q = -(half_b + half_b.signum() * disc.sqrt());
        let (t0, t1) = (q / a, (oc.length_squared() - self.radius * self.radius) / q);
        let (near, far) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
        if near >= 0.0 && near.is_finite() {
            Some(near)
        } else if far >= 0.0 && far.is_finite() {
            Some(far)
        } else if q == 0.0 && oc.length_squared() <= self.radius * self.radius {
            // The origin is the center: the exit is one radius along the ray.
            Some(self.radius / a.sqrt())
        } else {
            None
        }
    }
}

/// The view frustum as six inward-facing planes, extracted from a view-projection matrix for
/// wgpu clip space (`-w <= x, y <= w`, `0 <= z <= w`) with reversed Z (Gribb & Hartmann).
///
/// With an infinite far plane the far plane is degenerate; it is stored as an always-inside plane
/// (zero normal, `d = +∞`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    /// The planes in the order [`Frustum::LEFT`], `RIGHT`, `BOTTOM`, `TOP`, `NEAR`, `FAR`, normals
    /// pointing inside.
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Index of the left plane (`x = -w`).
    pub const LEFT: usize = 0;
    /// Index of the right plane (`x = w`).
    pub const RIGHT: usize = 1;
    /// Index of the bottom plane (`y = -w`).
    pub const BOTTOM: usize = 2;
    /// Index of the top plane (`y = w`).
    pub const TOP: usize = 3;
    /// Index of the near plane (`z = w` under reversed Z).
    pub const NEAR: usize = 4;
    /// Index of the far plane (`z = 0` under reversed Z).
    pub const FAR: usize = 5;

    /// The frustum of the view-projection matrix `view_proj` (world to clip space).
    pub fn from_view_proj(view_proj: Mat4) -> Self {
        let [r0, r1, r2, r3] = [view_proj.row(0), view_proj.row(1), view_proj.row(2), view_proj.row(3)];
        let plane = |v: Vec4| {
            Plane::from_vec4(v)
                .unwrap_or(Plane { normal: Vec3::ZERO, d: if v.w >= 0.0 { f32::INFINITY } else { f32::NEG_INFINITY } })
        };
        Self { planes: [plane(r3 + r0), plane(r3 - r0), plane(r3 + r1), plane(r3 - r1), plane(r3 - r2), plane(r2)] }
    }

    /// Whether `p` is inside or on the boundary.
    #[inline]
    pub fn contains_point(&self, p: Vec3) -> bool {
        self.planes.iter().all(|pl| pl.signed_distance(p) >= 0.0)
    }

    /// Whether the sphere may be visible: `false` only when it is entirely outside one plane
    /// (conservative near the frustum's edges, like every plane test).
    #[inline]
    pub fn intersects_sphere(&self, s: Sphere) -> bool {
        self.planes.iter().all(|pl| pl.signed_distance(s.center) >= -s.radius)
    }

    /// Whether the box may be visible: `false` only when it is entirely outside one plane
    /// (conservative near the frustum's edges). The empty box is never visible.
    #[inline]
    pub fn intersects_aabb(&self, b: Aabb) -> bool {
        if b.is_empty() {
            return false;
        }
        self.planes.iter().all(|pl| {
            // The corner farthest along the normal.
            let n = pl.normal;
            let p = Vec3::new(
                if n.x >= 0.0 { b.max.x } else { b.min.x },
                if n.y >= 0.0 { b.max.y } else { b.min.y },
                if n.z >= 0.0 { b.max.z } else { b.min.z },
            );
            pl.signed_distance(p) >= 0.0
        })
    }

    /// Culls bounding spheres given as structure-of-arrays channels: `visible[i]` is set to
    /// [`Self::intersects_sphere`]. Processes as many as the shortest slice holds and returns that
    /// count; the rest of `visible` is untouched.
    pub fn cull_spheres(&self, centers: &[Vec3], radii: &[f32], visible: &mut [bool]) -> usize {
        let n = centers.len().min(radii.len()).min(visible.len());
        for ((v, &c), &r) in visible.iter_mut().zip(centers).zip(radii) {
            *v = self.intersects_sphere(Sphere::new(c, r));
        }
        n
    }
}

/// The (unnormalised) face normal `(b - a) × (c - a)`: its length is twice the area, its direction
/// faces a viewer who sees `a`, `b`, `c` counter-clockwise.
#[inline]
pub fn triangle_normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    (b - a).cross(c - a)
}

/// The area of the triangle.
#[inline]
pub fn triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    0.5 * triangle_normal(a, b, c).length()
}

/// The barycentric coordinates `(wa, wb, wc)` of `p` projected onto the triangle's plane, or
/// `None` for a degenerate (zero-area) triangle. The coordinates are outside `[0, 1]` when the
/// projection is outside the triangle.
pub fn barycentric(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<Vec3> {
    // Ericson, Real-Time Collision Detection 3.4 (Cramer's rule on the normal equations).
    let (v0, v1, v2) = (b - a, c - a, p - a);
    let (d00, d01, d11) = (v0.dot(v0), v0.dot(v1), v1.dot(v1));
    let (d20, d21) = (v2.dot(v0), v2.dot(v1));
    let denom = d00 * d11 - d01 * d01;
    // Relative test: the Gram determinant is |v0 × v1|², which rounding can leave slightly
    // non-zero for collinear points.
    if denom.is_nan() || denom <= f32::EPSILON * d00 * d11 || !denom.is_finite() {
        return None;
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let out = Vec3::new(1.0 - v - w, v, w);
    out.is_finite().then_some(out)
}

/// The barycentric coordinates `(wa, wb, wc)` of `p` in a 2D triangle (texture space), or `None`
/// for a degenerate one.
pub fn barycentric_2d(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> Option<Vec3> {
    let (v0, v1, v2) = (b - a, c - a, p - a);
    let denom = v0.perp_dot(v1);
    if denom == 0.0 || !denom.is_finite() {
        return None;
    }
    let v = v2.perp_dot(v1) / denom;
    let w = v0.perp_dot(v2) / denom;
    let out = Vec3::new(1.0 - v - w, v, w);
    out.is_finite().then_some(out)
}

/// The point with barycentric coordinates `bary` in the triangle.
#[inline]
pub fn from_barycentric(a: Vec3, b: Vec3, c: Vec3, bary: Vec3) -> Vec3 {
    a * bary.x + b * bary.y + c * bary.z
}

/// The attribute with barycentric coordinates `bary` between three 2D attributes (UVs).
#[inline]
pub fn from_barycentric_2d(a: Vec2, b: Vec2, c: Vec2, bary: Vec3) -> Vec2 {
    a * bary.x + b * bary.y + c * bary.z
}

/// The point of the segment `a`–`b` closest to `p` and its parameter in `[0, 1]` (`0` for a
/// zero-length segment).
#[inline]
pub fn closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> (Vec3, f32) {
    let ab = b - a;
    let len2 = ab.length_squared();
    let t = if len2 > 0.0 { ((p - a).dot(ab) / len2).clamp(0.0, 1.0) } else { 0.0 };
    (a + ab * t, if t.is_nan() { 0.0 } else { t })
}

/// The point of the (solid) triangle closest to `p` and its barycentric coordinates. Degenerate
/// triangles are handled as their longest edge (or their single point).
pub fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, Vec3) {
    // Ericson, Real-Time Collision Detection 5.1.5: find the Voronoi region of p.
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, Vec3::X);
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return (b, Vec3::Y);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 && d1 - d3 > 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, Vec3::new(1.0 - v, v, 0.0));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return (c, Vec3::Z);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 && d2 - d6 > 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, Vec3::new(1.0 - w, 0.0, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 && (d4 - d3) + (d5 - d6) > 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, Vec3::new(0.0, 1.0 - w, w));
    }
    let sum = va + vb + vc;
    if sum > 0.0 && sum.is_finite() {
        let (v, w) = (vb / sum, vc / sum);
        return (a + ab * v + ac * w, Vec3::new(1.0 - v - w, v, w));
    }
    // Degenerate (or non-finite) triangle: the closest of the three edges.
    let edges = [(a, b, 0, 1), (b, c, 1, 2), (c, a, 2, 0)];
    let mut best = (a, Vec3::X);
    let mut best_d = f32::INFINITY;
    for (s, e, i, j) in edges {
        let (q, t) = closest_point_on_segment(p, s, e);
        let d = q.distance_squared(p);
        if d < best_d {
            best_d = d;
            let mut bary = Vec3::ZERO;
            bary[i] = 1.0 - t;
            bary[j] = t;
            best = (q, bary);
        }
    }
    best
}

/// Where a ray meets a triangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleHit {
    /// The ray parameter of the hit.
    pub t: f32,
    /// The barycentric coordinates `(wa, wb, wc)` of the hit point.
    pub bary: Vec3,
    /// Whether the ray hit the front face (the counter-clockwise side).
    pub front_face: bool,
}

/// The watertight ray-triangle test of Woop, Benthin and Wald ("Watertight Ray/Triangle
/// Intersection", JCGT 2013): a ray that hits the shared edge or vertex of two triangles hits at
/// least one of them, so brush strokes and picking never fall through the seams of a mesh.
///
/// Built once per ray by [`Ray::triangle_tester`], then tested against any number of triangles.
#[derive(Clone, Copy, Debug)]
pub struct RayTriangleTester {
    origin: Vec3,
    direction: Vec3,
    /// The permutation that makes the dominant direction axis `z`, as component indices.
    kx: usize,
    ky: usize,
    kz: usize,
    /// The shear that maps the ray onto the +z axis.
    shear: Vec3,
}

impl RayTriangleTester {
    /// The tester for `ray`.
    pub fn new(ray: Ray) -> Self {
        let d = ray.direction.abs();
        let kz = if d.x >= d.y && d.x >= d.z {
            0
        } else if d.y >= d.z {
            1
        } else {
            2
        };
        let (mut kx, mut ky) = ((kz + 1) % 3, (kz + 2) % 3);
        let dir = ray.direction.to_array();
        if dir[kz] < 0.0 {
            core::mem::swap(&mut kx, &mut ky);
        }
        let dz = dir[kz];
        Self {
            origin: ray.origin,
            direction: ray.direction,
            kx,
            ky,
            kz,
            shear: Vec3::new(dir[kx] / dz, dir[ky] / dz, 1.0 / dz),
        }
    }

    /// The intersection with the triangle `a`, `b`, `c` from either side with `0 <= t <= t_max`,
    /// or `None`. Degenerate triangles and rays (a zero direction) never hit.
    pub fn intersect(&self, a: Vec3, b: Vec3, c: Vec3, t_max: f32) -> Option<TriangleHit> {
        let (kx, ky, kz, s) = (self.kx, self.ky, self.kz, self.shear);
        let (a, b, c) = ((a - self.origin).to_array(), (b - self.origin).to_array(), (c - self.origin).to_array());
        let (ax, ay) = (a[kx] - s.x * a[kz], a[ky] - s.y * a[kz]);
        let (bx, by) = (b[kx] - s.x * b[kz], b[ky] - s.y * b[kz]);
        let (cx, cy) = (c[kx] - s.x * c[kz], c[ky] - s.y * c[kz]);
        let (mut u, mut v, mut w) = (cx * by - cy * bx, ax * cy - ay * cx, bx * ay - by * ax);
        if u == 0.0 || v == 0.0 || w == 0.0 {
            // On an edge in single precision: decide in double so neighbours agree.
            let f = f64::from;
            u = (f(cx) * f(by) - f(cy) * f(bx)) as f32;
            v = (f(ax) * f(cy) - f(ay) * f(cx)) as f32;
            w = (f(bx) * f(ay) - f(by) * f(ax)) as f32;
        }
        if (u < 0.0 || v < 0.0 || w < 0.0) && (u > 0.0 || v > 0.0 || w > 0.0) {
            return None;
        }
        let det = u + v + w;
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let t_scaled = u * (s.z * a[kz]) + v * (s.z * b[kz]) + w * (s.z * c[kz]);
        let t = t_scaled / det;
        if !(t >= 0.0 && t <= t_max) {
            return None;
        }
        let bary = Vec3::new(u, v, w) / det;
        // Front face: the triangle's normal opposes the ray.
        let n = (Vec3::from_array(b) - Vec3::from_array(a)).cross(Vec3::from_array(c) - Vec3::from_array(a));
        Some(TriangleHit { t, bary, front_face: n.dot(self.direction) < 0.0 })
    }
}

/// A ray's nearest hit on an indexed triangle mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshHit {
    /// The index of the triangle (its indices are `3 * triangle .. 3 * triangle + 3`).
    pub triangle: u32,
    /// Where the ray met it.
    pub hit: TriangleHit,
}

/// The nearest hit of `ray` on the indexed triangle mesh (`positions`, three `indices` per
/// triangle) within `t_max`, from either side, or `None`. Triangles with an index outside
/// `positions` are skipped, as is a trailing partial triangle. The brute-force CPU fallback for
/// picking and brush hit tests (`[VP-01]`); it allocates nothing.
pub fn ray_cast_mesh(ray: Ray, positions: &[Vec3], indices: &[u32], t_max: f32) -> Option<MeshHit> {
    let tester = ray.triangle_tester();
    let mut best: Option<MeshHit> = None;
    let mut limit = t_max;
    for (triangle, tri) in indices.as_chunks::<3>().0.iter().enumerate() {
        let corner = |i: u32| positions.get(i as usize).copied();
        let (Some(a), Some(b), Some(c)) = (corner(tri[0]), corner(tri[1]), corner(tri[2])) else {
            continue;
        };
        if let Some(hit) = tester.intersect(a, b, c, limit) {
            limit = hit.t;
            best = Some(MeshHit { triangle: triangle as u32, hit });
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    fn rand_vec3(rng: &mut Rng, extent: f32) -> Vec3 {
        Vec3::new(rng.range_f32(-extent, extent), rng.range_f32(-extent, extent), rng.range_f32(-extent, extent))
    }

    fn rand_unit(rng: &mut Rng) -> Vec3 {
        loop {
            if let Some(u) = Vec3::new(rng.normal_f32(), rng.normal_f32(), rng.normal_f32()).try_normalize() {
                return u;
            }
        }
    }

    #[test]
    fn rays_and_planes() {
        let r = Ray::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, -2.0)).expect("valid");
        assert_eq!(r.direction, -Vec3::Z);
        assert_eq!(r.at(2.0), Vec3::new(1.0, 2.0, 1.0));
        assert_eq!(r.closest_t(Vec3::new(5.0, 2.0, 0.0)), 3.0);
        assert_eq!(r.closest_t(Vec3::new(5.0, 2.0, 9.0)), 0.0, "behind the origin");
        assert_eq!(r.distance_squared_to_point(Vec3::new(4.0, 2.0, 0.0)), 9.0);
        assert!(Ray::new(Vec3::ZERO, Vec3::ZERO).is_none());
        assert!(Ray::new(Vec3::NAN, Vec3::X).is_none());
        assert!(Ray::between(Vec3::ONE, Vec3::ONE).is_none());

        let p = Plane::from_points(Vec3::ZERO, Vec3::X, Vec3::Y).expect("valid");
        assert_eq!(p.normal, Vec3::Z, "counter-clockwise faces +Z");
        assert_eq!(p.signed_distance(Vec3::new(3.0, 4.0, -2.0)), -2.0);
        assert_eq!(p.project_point(Vec3::new(3.0, 4.0, -2.0)), Vec3::new(3.0, 4.0, 0.0));
        assert_eq!(p.intersect_ray(r), Some(3.0));
        assert_eq!(p.intersect_ray(Ray::new(Vec3::Z, Vec3::Z).expect("valid")), None, "behind");
        assert_eq!(p.intersect_ray(Ray::new(Vec3::Z, Vec3::X).expect("valid")), None, "parallel");
        assert!(Plane::from_points(Vec3::ZERO, Vec3::X, Vec3::X * 2.0).is_none());
        let q = Plane::from_vec4(Vec4::new(0.0, 2.0, 0.0, -4.0)).expect("valid");
        assert_eq!((q.normal, q.d), (Vec3::Y, -2.0));
        assert!(Plane::from_vec4(Vec4::new(0.0, 0.0, 0.0, 1.0)).is_none());
        let s = Plane::from_point_normal(Vec3::new(0.0, 5.0, 0.0), Vec3::Y * 3.0).expect("valid");
        assert_eq!(s.signed_distance(Vec3::new(9.0, 7.0, 1.0)), 2.0);
    }

    #[test]
    fn camera_rays_come_out_of_the_reversed_z_projections() {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 1.5, 3.0), Vec3::new(0.0, 1.5, 0.0), Vec3::Y);
        let projections = [
            Mat4::perspective_infinite_reversed_z(1.0, 1.5, 0.1),
            Mat4::perspective_reversed_z(1.0, 1.5, 0.1, 100.0),
            Mat4::orthographic_reversed_z(-2.0, 2.0, -1.0, 1.0, 0.1, 50.0),
        ];
        for proj in projections {
            let view_proj = proj * view;
            let inv = view_proj.inverse().expect("invertible");
            let mut rng = Rng::new(0x6e);
            for _ in 0..500 {
                let ndc = Vec2::new(rng.range_f32(-1.0, 1.0), rng.range_f32(-1.0, 1.0));
                let ray = Ray::from_ndc(inv, ndc).expect("valid ray");
                // Every point of the ray projects back to the same pixel, in front of the camera.
                for t in [0.0, 0.5, 3.0, 20.0] {
                    let clip = view_proj * ray.at(t).extend(1.0);
                    let back = clip.xyz() / clip.w;
                    assert!(back.xy().abs_diff_eq(ndc, 1e-3), "{t}: {back:?} vs {ndc:?}");
                    assert!(back.z <= 1.0 + 1e-4 && clip.w > 0.0);
                }
                assert!(ray.direction.z < 0.0, "the camera looks down -Z");
            }
        }
        assert!(Ray::from_ndc(Mat4::ZERO, Vec2::ZERO).is_none());
    }

    #[test]
    fn boxes() {
        let b = Aabb::new(Vec3::new(1.0, -1.0, 0.0), Vec3::new(-1.0, 1.0, 2.0));
        assert_eq!((b.min, b.max), (Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 2.0)));
        assert_eq!((b.center(), b.half_extents(), b.size()), (Vec3::new(0.0, 0.0, 1.0), Vec3::ONE, Vec3::splat(2.0)));
        assert_eq!(b.surface_area(), 24.0);
        assert!(b.contains_point(Vec3::new(1.0, 0.0, 0.0)) && !b.contains_point(Vec3::new(1.1, 0.0, 0.0)));
        assert_eq!(Aabb::from_center_half_extents(Vec3::new(0.0, 0.0, 1.0), -Vec3::ONE), b);
        assert!(Aabb::EMPTY.is_empty() && !b.is_empty());
        assert_eq!(Aabb::EMPTY.size(), Vec3::ZERO);
        assert_eq!(Aabb::EMPTY.union(b), b);
        assert_eq!(Aabb::EMPTY.expand(1.0), Aabb::EMPTY);
        assert!(!Aabb::EMPTY.contains_point(Vec3::ZERO) && !Aabb::EMPTY.intersects(b));
        assert_eq!(Aabb::EMPTY.distance_squared_to_point(Vec3::ZERO), f32::INFINITY);
        assert_eq!(Aabb::from_points(&[]), Aabb::EMPTY);
        let pts = [Vec3::new(1.0, 5.0, -2.0), Vec3::new(-3.0, 0.0, 4.0), Vec3::NAN, Vec3::new(0.0, 1.0, 0.0)];
        let fb = Aabb::from_points(&pts);
        assert_eq!((fb.min, fb.max), (Vec3::new(-3.0, 0.0, -2.0), Vec3::new(1.0, 5.0, 4.0)), "NaN ignored");
        let (x, y, z) = (pts.map(|p| p.x), pts.map(|p| p.y), pts.map(|p| p.z));
        assert_eq!(Aabb::from_points_soa(&x, &y, &z), fb);
        assert_eq!(Aabb::from_points_soa(&x[..2], &y, &z), Aabb::from_points(&pts[..2]));
        let other = Aabb::new(Vec3::new(0.5, 0.5, 1.5), Vec3::splat(3.0));
        assert!(b.intersects(other));
        assert_eq!(b.intersection(other), Aabb::new(Vec3::new(0.5, 0.5, 1.5), Vec3::new(1.0, 1.0, 2.0)));
        assert_eq!(b.intersection(Aabb::new(Vec3::splat(5.0), Vec3::splat(6.0))), Aabb::EMPTY);
        assert_eq!(b.closest_point(Vec3::new(3.0, 0.5, -1.0)), Vec3::new(1.0, 0.5, 0.0));
        assert_eq!(b.distance_squared_to_point(Vec3::new(3.0, 0.5, -1.0)), 5.0);
        assert_eq!(b.distance_squared_to_point(Vec3::new(0.2, 0.5, 1.0)), 0.0);
        assert_eq!(b.expand(0.5).size(), Vec3::splat(3.0));

        // The transformed box holds every transformed corner and touches each face.
        let mut rng = Rng::new(0x6a);
        for _ in 0..500 {
            let m = Mat4::from_scale_rotation_translation(
                Vec3::new(rng.range_f32(0.2, 2.0), rng.range_f32(-2.0, -0.2), rng.range_f32(0.2, 2.0)),
                super::super::quat::Quat::from_axis_angle(rand_unit(&mut rng), rng.range_f32(-3.0, 3.0)),
                rand_vec3(&mut rng, 5.0),
            );
            let tb = b.transform(m);
            let corners: [Vec3; 8] = core::array::from_fn(|i| {
                Vec3::new(
                    if i & 1 == 0 { b.min.x } else { b.max.x },
                    if i & 2 == 0 { b.min.y } else { b.max.y },
                    if i & 4 == 0 { b.min.z } else { b.max.z },
                )
            });
            let exact = Aabb::from_points(&corners.map(|c| m.transform_point3(c)));
            assert!(tb.min.abs_diff_eq(exact.min, 1e-4) && tb.max.abs_diff_eq(exact.max, 1e-4));
        }
        assert_eq!(Aabb::EMPTY.transform(Mat4::IDENTITY), Aabb::EMPTY);
    }

    #[test]
    fn ray_box_slabs() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::ONE);
        let ray = |o: Vec3, d: Vec3| Ray::new(o, d).expect("valid");
        assert_eq!(b.intersect_ray(ray(Vec3::new(-5.0, 0.0, 0.0), Vec3::X)), Some((4.0, 6.0)));
        assert_eq!(b.intersect_ray(ray(Vec3::ZERO, Vec3::X)), Some((0.0, 1.0)), "origin inside");
        assert_eq!(b.intersect_ray(ray(Vec3::new(5.0, 0.0, 0.0), Vec3::X)), None, "behind");
        assert_eq!(b.intersect_ray(ray(Vec3::new(-5.0, 2.0, 0.0), Vec3::X)), None, "parallel, outside");
        // Lying in a face plane: 0 · ∞ must not lose the hit.
        assert_eq!(b.intersect_ray(ray(Vec3::new(-5.0, 1.0, 0.0), Vec3::X)), Some((4.0, 6.0)));
        assert_eq!(Aabb::EMPTY.intersect_ray(ray(Vec3::ZERO, Vec3::X)), None);
        // Random rays agree with sampling along the ray.
        let mut rng = Rng::new(0x6b);
        for _ in 0..2000 {
            let r = ray(rand_vec3(&mut rng, 4.0), rand_unit(&mut rng));
            match b.intersect_ray(r) {
                Some((t0, t1)) => {
                    assert!(t0 <= t1);
                    assert!(b.expand(1e-4).contains_point(r.at(t0)) && b.expand(1e-4).contains_point(r.at(t1)));
                    assert!(b.contains_point(r.at(f32::midpoint(t0, t1))));
                }
                None => {
                    for i in 0..200 {
                        assert!(!b.expand(-1e-3).contains_point(r.at(i as f32 * 0.05)));
                    }
                }
            }
        }
    }

    #[test]
    fn spheres() {
        let s = Sphere::new(Vec3::new(0.0, 0.0, -5.0), 1.0);
        let ray = |o: Vec3, d: Vec3| Ray::new(o, d).expect("valid");
        assert_eq!(s.intersect_ray(ray(Vec3::ZERO, -Vec3::Z)), Some(4.0));
        assert_eq!(s.intersect_ray(ray(Vec3::new(0.0, 0.0, -5.0), Vec3::X)), Some(1.0), "from the center");
        assert_eq!(s.intersect_ray(ray(Vec3::new(0.0, 0.0, -5.5), -Vec3::Z)), Some(0.5), "from inside");
        assert_eq!(s.intersect_ray(ray(Vec3::ZERO, Vec3::Z)), None, "behind");
        assert_eq!(s.intersect_ray(ray(Vec3::new(0.0, 1.5, 0.0), -Vec3::Z)), None, "miss");
        // A tiny sphere far away: the stable form still finds it.
        let far = Sphere::new(Vec3::new(0.0, 0.0, -1.0e4), 1.0e-2);
        let t = far.intersect_ray(ray(Vec3::ZERO, -Vec3::Z)).expect("hit");
        assert!((t - (1.0e4 - 1.0e-2)).abs() < 1e-3);
        assert!(s.contains_point(Vec3::new(0.0, 1.0, -5.0)) && !s.contains_point(Vec3::ZERO));
        assert!(s.intersects(Sphere::new(Vec3::new(0.0, 2.0, -5.0), 1.0)));
        assert!(!s.intersects(Sphere::new(Vec3::new(0.0, 2.1, -5.0), 1.0)));
        assert!(s.intersects_aabb(Aabb::new(Vec3::new(0.5, 0.5, -5.0), Vec3::splat(3.0))));
        assert!(!s.intersects_aabb(Aabb::new(Vec3::new(0.8, 0.8, -5.0), Vec3::splat(3.0))));
        let pts = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 0.5, 0.0)];
        let bs = Sphere::from_points(&pts);
        assert!(pts.iter().all(|&p| bs.contains_point(p)));
        assert_eq!(Sphere::from_points(&[]), Sphere::new(Vec3::ZERO, 0.0));
        assert_eq!(Sphere::from_aabb(Aabb::new(Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0))), Sphere::new(Vec3::X, 1.0));
        let m = Mat4::from_translation(Vec3::X) * Mat4::from_scale(Vec3::new(1.0, 3.0, 2.0));
        assert_eq!(s.transform(m), Sphere::new(Vec3::new(1.0, 0.0, -10.0), 3.0));
        // Random rays: hits lie on the surface; misses pass farther than the radius.
        let mut rng = Rng::new(0x6c);
        for _ in 0..2000 {
            let r = ray(rand_vec3(&mut rng, 6.0), rand_unit(&mut rng));
            match s.intersect_ray(r) {
                Some(t) => assert!((r.at(t).distance(s.center) - 1.0).abs() < 1e-4 && t >= 0.0),
                None => assert!(r.distance_squared_to_point(s.center) > 1.0 - 1e-4),
            }
        }
    }

    #[test]
    fn frustum_planes_match_the_clip_volume() {
        let view = Mat4::look_at_rh(Vec3::new(1.0, 2.0, 4.0), Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
        let mut rng = Rng::new(0x6d);
        for (proj, has_far) in [
            (Mat4::perspective_reversed_z(1.2, 1.3, 0.5, 20.0), true),
            (Mat4::perspective_infinite_reversed_z(1.2, 1.3, 0.5), false),
            (Mat4::orthographic_reversed_z(-3.0, 3.0, -2.0, 2.0, 0.5, 20.0), true),
        ] {
            let view_proj = proj * view;
            let f = Frustum::from_view_proj(view_proj);
            if !has_far {
                assert_eq!(f.planes[Frustum::FAR].normal, Vec3::ZERO);
            }
            for _ in 0..5000 {
                let p = rand_vec3(&mut rng, 30.0);
                let c = view_proj * p.extend(1.0);
                let inside = c.w > 0.0 && c.x.abs() <= c.w && c.y.abs() <= c.w && c.z >= 0.0 && c.z <= c.w;
                // Points within rounding of a plane may go either way.
                let margin = f.planes.iter().map(|pl| pl.signed_distance(p).abs()).fold(f32::INFINITY, f32::min);
                if margin > 1e-3 {
                    assert_eq!(f.contains_point(p), inside, "{p:?}");
                    assert_eq!(f.intersects_sphere(Sphere::new(p, 0.0)), inside);
                    assert_eq!(f.intersects_aabb(Aabb::new(p, p)), inside);
                }
                // A sphere is culled only when it is entirely outside.
                let s = Sphere::new(p, rng.range_f32(0.0, 3.0));
                if !f.intersects_sphere(s) {
                    for _ in 0..20 {
                        let q = s.center + rand_unit(&mut rng) * s.radius * rng.f32();
                        assert!(!f.contains_point(q));
                    }
                }
                let b = Aabb::from_center_half_extents(p, Vec3::splat(rng.range_f32(0.0, 2.0)));
                if !f.intersects_aabb(b) {
                    for _ in 0..20 {
                        let q = b.min + (b.max - b.min) * Vec3::new(rng.f32(), rng.f32(), rng.f32());
                        assert!(!f.contains_point(q));
                    }
                }
            }
        }
        let f = Frustum::from_view_proj(Mat4::perspective_infinite_reversed_z(1.0, 1.0, 0.1));
        assert!(f.contains_point(Vec3::new(0.0, 0.0, -1.0e6)), "no far plane");
        assert!(!f.contains_point(Vec3::new(0.0, 0.0, -0.05)), "before the near plane");
        assert!(!f.intersects_aabb(Aabb::EMPTY));
        let mut visible = [true; 3];
        let n = f.cull_spheres(&[Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 5.0)], &[1.0, 1.0, 1.0], &mut visible);
        assert_eq!((n, visible), (2, [true, false, true]));
    }

    #[test]
    fn mod_06_barycentrics_round_trip() {
        let mut rng = Rng::new(0x6f);
        for _ in 0..5000 {
            let (a, b, c) = (rand_vec3(&mut rng, 3.0), rand_vec3(&mut rng, 3.0), rand_vec3(&mut rng, 3.0));
            if triangle_area(a, b, c) < 0.5 {
                continue;
            }
            let w = Vec3::new(rng.range_f32(-0.5, 1.5), rng.range_f32(-0.5, 1.5), 0.0);
            let w = Vec3::new(w.x, w.y, 1.0 - w.x - w.y);
            let p = from_barycentric(a, b, c, w);
            let got = barycentric(p, a, b, c).expect("non-degenerate");
            assert!(got.abs_diff_eq(w, 2e-3), "{w:?} vs {got:?}");
            // Off the plane: the coordinates of the projection.
            let n = triangle_normal(a, b, c).normalize();
            assert!(barycentric(p + n * 2.0, a, b, c).expect("valid").abs_diff_eq(w, 2e-3));
            let (a2, b2, c2) = (a.xy(), b.xy(), c.xy());
            if (b2 - a2).perp_dot(c2 - a2).abs() > 1.0 {
                let uv = from_barycentric_2d(a2, b2, c2, w);
                assert!(barycentric_2d(uv, a2, b2, c2).expect("valid").abs_diff_eq(w, 2e-3));
            }
        }
        assert_eq!(triangle_normal(Vec3::ZERO, Vec3::X, Vec3::Y), Vec3::Z);
        assert_eq!(triangle_area(Vec3::ZERO, Vec3::X * 2.0, Vec3::Y * 3.0), 3.0);
        assert_eq!(barycentric(Vec3::ONE, Vec3::ZERO, Vec3::X, Vec3::X * 2.0), None, "collinear");
        assert_eq!(barycentric(Vec3::ONE, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO), None);
        assert_eq!(barycentric(Vec3::NAN, Vec3::ZERO, Vec3::X, Vec3::Y), None);
        assert_eq!(barycentric_2d(Vec2::ONE, Vec2::ZERO, Vec2::X, Vec2::X * 3.0), None);
        let w = barycentric_2d(Vec2::new(0.25, 0.5), Vec2::ZERO, Vec2::X, Vec2::Y).expect("valid");
        assert_eq!(w, Vec3::new(0.25, 0.25, 0.5));
    }

    #[test]
    fn closest_points_on_segments_and_triangles() {
        assert_eq!(
            closest_point_on_segment(Vec3::new(0.5, 3.0, 0.0), Vec3::ZERO, Vec3::X),
            (Vec3::new(0.5, 0.0, 0.0), 0.5)
        );
        assert_eq!(closest_point_on_segment(Vec3::new(-2.0, 1.0, 0.0), Vec3::ZERO, Vec3::X), (Vec3::ZERO, 0.0));
        assert_eq!(closest_point_on_segment(Vec3::ONE, Vec3::Y, Vec3::Y), (Vec3::Y, 0.0));
        let mut rng = Rng::new(0x70);
        for _ in 0..3000 {
            let (a, b, c) = (rand_vec3(&mut rng, 2.0), rand_vec3(&mut rng, 2.0), rand_vec3(&mut rng, 2.0));
            let p = rand_vec3(&mut rng, 4.0);
            let (q, bary) = closest_point_on_triangle(p, a, b, c);
            // The coordinates describe the point and are a convex combination.
            assert!(from_barycentric(a, b, c, bary).abs_diff_eq(q, 1e-4));
            assert!(bary.min_element() >= -1e-4 && (bary.element_sum() - 1.0).abs() < 1e-5);
            // No sampled point of the triangle is closer.
            let d = q.distance_squared(p);
            for _ in 0..50 {
                let (u, v) = (rng.f32(), rng.f32());
                let (u, v) = if u + v > 1.0 { (1.0 - u, 1.0 - v) } else { (u, v) };
                let s = from_barycentric(a, b, c, Vec3::new(1.0 - u - v, u, v));
                assert!(s.distance_squared(p) >= d - 1e-4, "{p:?}");
            }
        }
        // Degenerate triangles: collinear and a single point.
        let (q, bary) = closest_point_on_triangle(Vec3::new(1.5, 1.0, 0.0), Vec3::ZERO, Vec3::X, Vec3::X * 2.0);
        assert!(q.abs_diff_eq(Vec3::new(1.5, 0.0, 0.0), 1e-6) && (bary.element_sum() - 1.0).abs() < 1e-6);
        assert_eq!(closest_point_on_triangle(Vec3::ONE, Vec3::Y, Vec3::Y, Vec3::Y).0, Vec3::Y);
        // NaN input must not panic; the value is meaningless.
        let _ = closest_point_on_triangle(Vec3::NAN, Vec3::ZERO, Vec3::X, Vec3::Y);
    }

    #[test]
    fn vp_01_ray_triangle_is_watertight() {
        let ray = |o: Vec3, d: Vec3| Ray::new(o, d).expect("valid");
        let (a, b, c) = (Vec3::ZERO, Vec3::X, Vec3::Y);
        let hit = ray(Vec3::new(0.25, 0.25, 2.0), -Vec3::Z).intersect_triangle(a, b, c).expect("hit");
        assert_eq!(hit.t, 2.0);
        assert!(hit.bary.abs_diff_eq(Vec3::new(0.5, 0.25, 0.25), 1e-6) && hit.front_face);
        let back = ray(Vec3::new(0.25, 0.25, -2.0), Vec3::Z).intersect_triangle(a, b, c).expect("two-sided");
        assert!(!back.front_face && back.t == 2.0);
        assert!(ray(Vec3::new(0.75, 0.75, 2.0), -Vec3::Z).intersect_triangle(a, b, c).is_none());
        assert!(ray(Vec3::new(0.25, 0.25, 2.0), Vec3::Z).intersect_triangle(a, b, c).is_none(), "behind");
        assert!(ray(Vec3::new(0.25, 0.25, 2.0), Vec3::X).intersect_triangle(a, b, c).is_none(), "parallel");
        assert!(ray(Vec3::new(0.25, 0.0, 2.0), -Vec3::Z).intersect_triangle(a, b, Vec3::X * 2.0).is_none());
        let tester = ray(Vec3::new(0.25, 0.25, 2.0), -Vec3::Z).triangle_tester();
        assert!(tester.intersect(a, b, c, 1.5).is_none(), "beyond t_max");

        let mut rng = Rng::new(0x71);
        // Random hits agree with the ray and the barycentric point.
        for _ in 0..3000 {
            let (a, b, c) = (rand_vec3(&mut rng, 2.0), rand_vec3(&mut rng, 2.0), rand_vec3(&mut rng, 2.0));
            let o = rand_vec3(&mut rng, 5.0);
            let w = Vec3::new(rng.f32(), rng.f32(), rng.f32());
            if w.min_element() < 1e-2 * w.element_sum() {
                continue;
            }
            let target = from_barycentric(a, b, c, w / w.element_sum());
            let Some(r) = Ray::between(o, target) else { continue };
            if triangle_area(a, b, c) < 0.05 || triangle_normal(a, b, c).normalize().dot(r.direction).abs() < 0.05 {
                continue;
            }
            let hit = r.intersect_triangle(a, b, c).expect("aimed inside");
            assert!(r.at(hit.t).abs_diff_eq(from_barycentric(a, b, c, hit.bary), 1e-3));
            assert!(hit.bary.abs_diff_eq(w / w.element_sum(), 1e-3));
            assert_eq!(hit.front_face, triangle_normal(a, b, c).dot(r.direction) < 0.0);
        }
        // Watertightness: rays through a fan of triangles sharing a center vertex, aimed exactly
        // at shared edges and at the shared vertex, always hit at least one triangle.
        let center = Vec3::new(0.3, -0.2, 0.0);
        let rim: [Vec3; 7] = core::array::from_fn(|i| {
            let ang = i as f32 * core::f32::consts::TAU / 7.0 + 0.1;
            center + Vec3::new(ang.cos(), ang.sin(), 0.02 * (i as f32))
        });
        let mut positions = vec![center];
        positions.extend_from_slice(&rim);
        let mut indices = Vec::new();
        for i in 0..7_u32 {
            indices.extend_from_slice(&[0, 1 + i, 1 + (i + 1) % 7]);
        }
        for _ in 0..20_000 {
            let i = rng.index(7);
            let edge_point = center.lerp(rim[i], rng.f32());
            let target = if rng.chance(0.1) { center } else { edge_point };
            let origin =
                target + Vec3::new(rng.range_f32(-0.1, 0.1), rng.range_f32(-0.1, 0.1), 1.0) * rng.range_f32(0.5, 4.0);
            let Some(r) = Ray::between(origin, target) else { continue };
            let hit = ray_cast_mesh(r, &positions, &indices, f32::INFINITY);
            assert!(hit.is_some(), "fell through at {target:?} from {origin:?}");
        }
    }

    #[test]
    fn mesh_ray_cast_finds_the_nearest_triangle_and_survives_bad_indices() {
        // Two parallel quads at z = 0 and z = -1, plus junk triangles.
        let positions = [
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
        ];
        let indices = [4, 5, 6, 4, 6, 7, 0, 1, 99, 0, 1, 2, 0, 2, 3, 7, 7];
        let r = Ray::new(Vec3::new(0.2, 0.5, 3.0), -Vec3::Z).expect("valid");
        let hit = ray_cast_mesh(r, &positions, &indices, f32::INFINITY).expect("hit");
        assert_eq!(hit.triangle, 4, "the nearer quad, the triangle after the junk");
        assert!((hit.hit.t - 3.0).abs() < 1e-5 && hit.hit.front_face);
        let hit = ray_cast_mesh(r, &positions, &indices, 3.5);
        assert_eq!(hit.map(|h| h.triangle), Some(4));
        assert!(ray_cast_mesh(r, &positions, &indices, 2.0).is_none());
        assert!(ray_cast_mesh(r, &positions, &[], f32::INFINITY).is_none());
        assert!(ray_cast_mesh(r, &[], &indices, f32::INFINITY).is_none());
        let from_below = Ray::new(Vec3::new(0.2, 0.5, -3.0), Vec3::Z).expect("valid");
        let hit = ray_cast_mesh(from_below, &positions, &indices, f32::INFINITY).expect("hit");
        assert_eq!((hit.triangle, hit.hit.front_face), (1, false));
        assert!((hit.hit.t - 2.0).abs() < 1e-5);
        // Hostile input: random indices and positions never panic.
        let mut rng = Rng::new(0x72);
        for _ in 0..200 {
            let pos: Vec<Vec3> = (0..rng.index(10)).map(|_| rand_vec3(&mut rng, 2.0)).collect();
            let idx: Vec<u32> = (0..rng.index(40)).map(|_| rng.next_u32() % 12).collect();
            let r = Ray::new(rand_vec3(&mut rng, 3.0), rand_unit(&mut rng)).expect("valid");
            if let Some(h) = ray_cast_mesh(r, &pos, &idx, f32::INFINITY) {
                assert!((h.triangle as usize) * 3 + 2 < idx.len() && h.hit.t >= 0.0);
            }
        }
    }
}
