//! Eye optics `[EYE-01]` `[EYE-02]`: the analytic eyeball (scleral sphere with a corneal cap), the
//! exact two-surface refraction through the cornea onto the iris plane, and the paraxial
//! entrance pupil.
//!
//! Frame: the eyeball centre is the origin, +Z the optical axis (gaze), millimetres. The anterior
//! corneal sphere meets the scleral sphere at the limbus circle (radius `limbus`, height
//! `limbus_z = sqrt(r² − limbus²)`); its centre lies on the axis so that it passes through that
//! circle. The posterior corneal sphere has its vertex `thickness` behind the anterior vertex
//! (apex). The iris is the plane `z = iris_z`, `iris_depth` behind the apex, a disc of the limbus
//! radius with the pupil hole in its centre.
//!
//! [`EyeGeometry::trace`] follows a camera ray: it hits the sclera or the cornea; through the
//! cornea it refracts at the anterior surface (air → cornea), at the posterior surface (cornea →
//! aqueous; rays that reach the iris plane before the posterior surface, near the limbus, stay in
//! the corneal medium) and lands on the iris plane. This is the reference of the eye shader
//! (`shaders/eye/eye.wgsl` does the same intersections in f32) and of the parallax tests
//! `[AC-WP16-2]`. [`EyeGeometry::paraxial_entrance_pupil`] gives the first-order image of the iris
//! plane through the cornea (ray-transfer matrices), which the exact trace reproduces for paraxial
//! rays; with the default eye it is 3.05 mm behind the apex at 1.13× magnification, the
//! entrance pupil of the schematic eyes (Atchison & Smith 2000, table 5.1).

use humanart_math::DVec3;

use super::params::{AQUEOUS_IOR, CORNEA_IOR, EyeParams};

/// The analytic eye of one set of parameters, in millimetres (f64: this is the reference).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeGeometry {
    /// Scleral sphere radius.
    pub radius: f64,
    /// Limbus radius.
    pub limbus: f64,
    /// Height of the limbus circle above the centre.
    pub limbus_z: f64,
    /// Anterior corneal radius.
    pub cornea_radius: f64,
    /// Height of the anterior corneal sphere's centre.
    pub cornea_centre_z: f64,
    /// Height of the corneal vertex (apex).
    pub apex_z: f64,
    /// Posterior corneal radius.
    pub back_radius: f64,
    /// Height of the posterior corneal sphere's centre.
    pub back_centre_z: f64,
    /// Height of the iris plane.
    pub iris_z: f64,
    /// Pupil radius.
    pub pupil: f64,
    /// Refractive index of the cornea.
    pub cornea_ior: f64,
    /// Refractive index of the aqueous humour.
    pub aqueous_ior: f64,
}

/// What a camera ray sees on the eye.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EyeHit {
    /// The ray misses the eyeball.
    Miss,
    /// The ray hits the sclera at `point` (unit outward `normal`).
    Sclera {
        /// The hit point.
        point: DVec3,
        /// The outward unit normal.
        normal: DVec3,
    },
    /// The ray enters the cornea at `point` (unit outward `normal`) and lands on the iris plane at
    /// `iris`, `radius` from the axis: the pupil when `radius < pupil`, the iris up to the limbus
    /// radius, beyond it the iris root hidden in the angle.
    Cornea {
        /// The point on the anterior cornea.
        point: DVec3,
        /// The anterior surface's outward unit normal there.
        normal: DVec3,
        /// Where the refracted ray meets the iris plane.
        iris: DVec3,
        /// Distance of `iris` from the axis.
        radius: f64,
    },
}

/// Refraction of unit direction `i` at a surface with unit normal `n` facing against `i`, from
/// index `n1` into `n2` (GLSL `refract` with `eta = n1 / n2`); `None` on total internal reflection.
#[must_use]
pub fn refract(i: DVec3, n: DVec3, n1: f64, n2: f64) -> Option<DVec3> {
    let eta = n1 / n2;
    let cos_i = -i.dot(n);
    let k = 1.0 - eta * eta * (1.0 - cos_i * cos_i);
    if k < 0.0 {
        return None;
    }
    Some(i * eta + n * (eta * cos_i - k.sqrt()))
}

/// The ray parameters where `origin + t dir` (unit `dir`) meets the sphere `(centre, radius)`,
/// nearer first.
fn sphere_hits(origin: DVec3, dir: DVec3, centre: DVec3, radius: f64) -> Option<(f64, f64)> {
    let oc = origin - centre;
    let b = oc.dot(dir);
    let c = oc.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    Some((-b - s, -b + s))
}

impl EyeGeometry {
    /// The geometry of `params` (call [`EyeParams::validate`] first for meaningful values).
    #[must_use]
    pub fn new(params: &EyeParams) -> Self {
        let radius = f64::from(params.eyeball_radius_mm);
        let limbus = f64::from(params.limbus_radius_mm);
        let cornea_radius = f64::from(params.cornea_radius_mm);
        let limbus_z = (radius * radius - limbus * limbus).max(0.0).sqrt();
        let cornea_centre_z = limbus_z - (cornea_radius * cornea_radius - limbus * limbus).max(0.0).sqrt();
        let apex_z = cornea_centre_z + cornea_radius;
        let back_radius = f64::from(params.cornea_back_radius_mm);
        let back_centre_z = apex_z - f64::from(params.cornea_thickness_mm) - back_radius;
        Self {
            radius,
            limbus,
            limbus_z,
            cornea_radius,
            cornea_centre_z,
            apex_z,
            back_radius,
            back_centre_z,
            iris_z: apex_z - f64::from(params.iris_depth_mm),
            pupil: f64::from(params.pupil_radius_mm()),
            cornea_ior: f64::from(CORNEA_IOR),
            aqueous_ior: f64::from(AQUEOUS_IOR),
        }
    }

    /// Distance from the centre to the outer surface (sclera or cornea) along unit direction `d`.
    #[must_use]
    pub fn surface_distance(&self, d: DVec3) -> f64 {
        if d.z * self.radius <= self.limbus_z {
            return self.radius;
        }
        let c = self.cornea_centre_z;
        let b = d.z * c;
        let disc = b * b - (c * c - self.cornea_radius * self.cornea_radius);
        if disc < 0.0 {
            return self.radius;
        }
        (b + disc.sqrt()).max(self.radius)
    }

    /// Follows the camera ray `origin + t dir` (`dir` need not be unit) into the eye.
    #[must_use]
    pub fn trace(&self, origin: DVec3, dir: DVec3) -> EyeHit {
        let dir = dir.normalize_or_zero();
        if dir == DVec3::ZERO {
            return EyeHit::Miss;
        }
        let ca = DVec3::new(0.0, 0.0, self.cornea_centre_z);
        // The nearest visible surface: the corneal cap above the limbus or the sclera below it.
        let mut best: Option<(f64, bool)> = None;
        if let Some((t0, t1)) = sphere_hits(origin, dir, ca, self.cornea_radius) {
            for t in [t0, t1] {
                if t > 0.0 && (origin + dir * t).z >= self.limbus_z {
                    best = Some((t, true));
                    break;
                }
            }
        }
        if let Some((t0, t1)) = sphere_hits(origin, dir, DVec3::ZERO, self.radius) {
            for t in [t0, t1] {
                if t > 0.0 && (origin + dir * t).z <= self.limbus_z && best.is_none_or(|b| t < b.0) {
                    best = Some((t, false));
                    break;
                }
            }
        }
        let Some((t, cornea)) = best else { return EyeHit::Miss };
        let point = origin + dir * t;
        if !cornea {
            return EyeHit::Sclera { point, normal: point.normalize_or_zero() };
        }
        let normal = (point - ca).normalize_or_zero();
        match self.through_cornea(point, normal, dir) {
            Some((iris, radius)) => EyeHit::Cornea { point, normal, iris, radius },
            None => EyeHit::Miss,
        }
    }

    /// From a point on the anterior cornea (outward unit `normal`) and the incoming unit
    /// direction: the point on the iris plane and its distance from the axis.
    #[must_use]
    pub fn through_cornea(&self, point: DVec3, normal: DVec3, dir: DVec3) -> Option<(DVec3, f64)> {
        let inside = refract(dir, normal, 1.0, self.cornea_ior)?;
        let plane_t = |o: DVec3, d: DVec3| if d.z < -1e-12 { Some((self.iris_z - o.z) / d.z) } else { None };
        let cp = DVec3::new(0.0, 0.0, self.back_centre_z);
        let to_plane = plane_t(point, inside);
        let back = sphere_hits(point, inside, cp, self.back_radius).map(|h| h.0).filter(|&t| t > 0.0);
        let (o, d) = match (back, to_plane) {
            (Some(tb), Some(tp)) if tb < tp => {
                let q = point + inside * tb;
                let n = (q - cp).normalize_or_zero();
                (q, refract(inside, n, self.cornea_ior, self.aqueous_ior)?)
            }
            _ => (point, inside),
        };
        let t = plane_t(o, d)?;
        let iris = o + d * t;
        Some((iris, (iris.x * iris.x + iris.y * iris.y).sqrt()))
    }

    /// The paraxial image of the iris plane through the cornea, seen from outside: (distance of
    /// the image behind the apex, lateral magnification). Ray-transfer matrices on (height,
    /// reduced angle) from the iris plane through the posterior surface, the corneal thickness and
    /// the anterior surface.
    #[must_use]
    pub fn paraxial_entrance_pupil(&self) -> (f64, f64) {
        type M = [[f64; 2]; 2];
        let mul = |a: M, b: M| -> M {
            [
                [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
                [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
            ]
        };
        let translate = |d: f64, n: f64| -> M { [[1.0, d / n], [0.0, 1.0]] };
        // Travelling +Z the centres of both surfaces lie behind them: radii negative.
        let refract_m = |n1: f64, n2: f64, r: f64| -> M { [[1.0, 0.0], [-(n2 - n1) / r, 1.0]] };
        let back_vertex = self.back_centre_z + self.back_radius;
        let m = mul(
            refract_m(self.cornea_ior, 1.0, -self.cornea_radius),
            mul(
                translate(self.apex_z - back_vertex, self.cornea_ior),
                mul(
                    refract_m(self.aqueous_ior, self.cornea_ior, -self.back_radius),
                    translate(back_vertex - self.iris_z, self.aqueous_ior),
                ),
            ),
        );
        // Image where y no longer depends on the angle: B + s D = 0; magnification A + s C.
        let s = -m[0][1] / m[1][1];
        (-s, m[0][0] + s * m[1][0])
    }

    /// Traces a ray from a point on the iris plane outward (unit `dir`, +Z-going) through the
    /// cornea into the air; returns the emerging ray (origin on the anterior cornea, direction).
    #[must_use]
    pub fn trace_out(&self, from: DVec3, dir: DVec3) -> Option<(DVec3, DVec3)> {
        let cp = DVec3::new(0.0, 0.0, self.back_centre_z);
        let ca = DVec3::new(0.0, 0.0, self.cornea_centre_z);
        let (_, tb) = sphere_hits(from, dir, cp, self.back_radius)?;
        let q = from + dir * tb;
        // The normal facing against the outgoing ray points inward.
        let inside = refract(dir, -(q - cp).normalize_or_zero(), self.aqueous_ior, self.cornea_ior)?;
        let (_, ta) = sphere_hits(q, inside, ca, self.cornea_radius)?;
        let p = q + inside * ta;
        let out = refract(inside, -(p - ca).normalize_or_zero(), self.cornea_ior, 1.0)?;
        Some((p, out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eye() -> EyeGeometry {
        EyeGeometry::new(&EyeParams::default())
    }

    /// Where two rays in the XZ plane (origins, directions) meet, extended both ways.
    fn meet(o1: DVec3, d1: DVec3, o2: DVec3, d2: DVec3) -> DVec3 {
        // o1 + a d1 = o2 + b d2 in x and z.
        let det = d1.x * (-d2.z) - d1.z * (-d2.x);
        let rx = o2.x - o1.x;
        let rz = o2.z - o1.z;
        let a = (rx * (-d2.z) - rz * (-d2.x)) / det;
        o1 + d1 * a
    }

    #[test]
    fn geometry_is_closed_at_the_limbus() {
        let g = eye();
        // The corneal sphere passes through the limbus circle.
        let p = DVec3::new(g.limbus, 0.0, g.limbus_z);
        assert!((p.distance(DVec3::new(0.0, 0.0, g.cornea_centre_z)) - g.cornea_radius).abs() < 1e-9);
        assert!((p.length() - g.radius).abs() < 1e-9);
        // Sagitta 2.64 mm, apex 13.12 mm from the centre, iris plane behind the limbus plane.
        assert!((g.apex_z - g.limbus_z - 2.641).abs() < 1e-3, "{}", g.apex_z - g.limbus_z);
        assert!(g.iris_z < g.limbus_z && g.iris_z > 0.0);
        assert!((g.surface_distance(DVec3::Z) - g.apex_z).abs() < 1e-9);
        assert!((g.surface_distance(DVec3::X) - g.radius).abs() < 1e-12);
    }

    /// `[AC-WP16-2]` The exact trace reproduces the paraxial image of the iris plane, and that
    /// image is the schematic eye's entrance pupil: about 3.05 mm behind the corneal vertex,
    /// magnified about 1.13× (Atchison & Smith 2000; the single-surface value with these
    /// indices is 3.048 mm and 1.131×).
    #[test]
    fn ac_wp16_2_paraxial_entrance_pupil_matches_the_exact_trace() {
        let g = eye();
        let (depth, mag) = g.paraxial_entrance_pupil();
        assert!((depth - 3.05).abs() < 0.04, "entrance pupil {depth} mm behind the apex");
        assert!((mag - 1.13).abs() < 0.015, "magnification {mag}");
        // Two paraxial rays from an iris point 0.02 mm off axis.
        let q = DVec3::new(0.02, 0.0, g.iris_z);
        let a = 0.002_f64;
        let (p1, d1) = g.trace_out(q, DVec3::new(a.sin(), 0.0, a.cos())).unwrap();
        let (p2, d2) = g.trace_out(q, DVec3::new(-a.sin(), 0.0, a.cos())).unwrap();
        let image = meet(p1, d1, p2, d2);
        let traced_depth = g.apex_z - image.z;
        let traced_mag = image.x / q.x;
        assert!((traced_depth - depth).abs() < 1e-4, "{traced_depth} vs {depth}");
        assert!((traced_mag - mag).abs() < 1e-4, "{traced_mag} vs {mag}");
    }

    /// `[AC-WP16-2]` Parallax: seen from a camera that orbits the eye, the iris point under the
    /// corneal apex moves on the iris plane as a point at the refracted (apparent) depth does,
    /// not as one at the physical depth; deeper irises move more; with the refraction switched off
    /// (cornea index 1) the shift is the plain geometric one.
    #[test]
    fn ac_wp16_2_iris_parallax_follows_the_refracted_depth() {
        let shift_at = |params: &EyeParams, angle: f64, ior: Option<(f64, f64)>| {
            let mut g = EyeGeometry::new(params);
            if let Some((c, a)) = ior {
                g.cornea_ior = c;
                g.aqueous_ior = a;
            }
            // Camera 300 mm away, aimed at the apex, orbiting about the eyeball centre in XZ.
            let apex = DVec3::new(0.0, 0.0, g.apex_z);
            let cam = DVec3::new(angle.sin(), 0.0, angle.cos()) * 300.0;
            match g.trace(cam, apex - cam) {
                EyeHit::Cornea { iris, .. } => iris.x,
                other => panic!("{other:?}"),
            }
        };
        let p = EyeParams::default();
        let angle = 0.25_f64;
        let refracted = shift_at(&p, angle, None);
        let plain = shift_at(&p, angle, Some((1.0, 1.0)));
        // Without refraction the ray goes straight: the apex-to-plane drop times the tangent of
        // the view angle (from the apex, the camera direction is close to the orbit angle).
        let g = EyeGeometry::new(&p);
        let cam = DVec3::new(angle.sin(), 0.0, angle.cos()) * 300.0;
        let dir = (DVec3::new(0.0, 0.0, g.apex_z) - cam).normalize();
        let expected_plain = dir.x / dir.z * (g.iris_z - g.apex_z);
        assert!((plain - expected_plain).abs() < 1e-9, "{plain} vs {expected_plain}");
        // Refraction bends the ray toward the normal: a smaller shift, between the paraxial
        // apparent-depth prediction and the plain one.
        assert!(refracted.abs() < plain.abs() * 0.85, "{refracted} vs {plain}");
        assert_eq!(refracted.signum(), plain.signum());
        // Deeper iris, more parallax; monotonic.
        let mut deeper = p;
        deeper.iris_depth_mm = 4.2;
        let mut shallower = p;
        shallower.iris_depth_mm = 2.8;
        let (s1, s2) = (shift_at(&shallower, angle, None).abs(), shift_at(&deeper, angle, None).abs());
        assert!(s1 < refracted.abs() && refracted.abs() < s2, "{s1} {refracted} {s2}");
        // Small angles: the shift per radian of view angle matches what the paraxial image
        // predicts for a point at the apparent depth, scaled back by the magnification.
        let small = 0.01_f64;
        let rate = shift_at(&p, small, None) / small;
        let (depth, mag) = g.paraxial_entrance_pupil();
        // An image point at `depth` behind the apex moves by -depth * angle on the image; on the
        // iris it is that over the magnification. The camera direction from the apex differs
        // from the orbit angle by the apex offset (apex_z / 300).
        let view = small * (1.0 + g.apex_z / (300.0 - g.apex_z));
        let predicted = -depth * view / mag / small;
        assert!((rate - predicted).abs() < 0.02 * predicted.abs(), "{rate} vs {predicted}");
    }

    #[test]
    fn trace_classifies_sclera_pupil_and_iris() {
        let g = eye();
        let cam = DVec3::new(0.0, 0.0, 300.0);
        // Straight at the apex: the pupil.
        match g.trace(cam, -DVec3::Z) {
            EyeHit::Cornea { radius, .. } => assert!(radius < 1e-9),
            other => panic!("{other:?}"),
        }
        // Aimed at a point 4 mm off axis on the cornea: the iris, refracted inward (smaller).
        let target = DVec3::new(4.0, 0.0, g.cornea_centre_z + (g.cornea_radius.powi(2) - 16.0).sqrt());
        match g.trace(cam, target - cam) {
            EyeHit::Cornea { radius, .. } => assert!(radius > g.pupil && radius < 4.0, "{radius}"),
            other => panic!("{other:?}"),
        }
        // Off to the side: the sclera.
        match g.trace(DVec3::new(9.0, 0.0, 300.0), -DVec3::Z) {
            EyeHit::Sclera { point, normal } => {
                assert!((point.length() - g.radius).abs() < 1e-9);
                assert!((normal.length() - 1.0).abs() < 1e-12);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(g.trace(DVec3::new(30.0, 0.0, 300.0), -DVec3::Z), EyeHit::Miss);
        assert_eq!(g.trace(cam, DVec3::ZERO), EyeHit::Miss);
    }

    #[test]
    fn refraction_obeys_snell() {
        let n = DVec3::Z;
        let i = DVec3::new(0.6, 0.0, -0.8);
        let t = refract(i, n, 1.0, 1.336).unwrap();
        assert!((t.length() - 1.0).abs() < 1e-12);
        let (s1, s2) = (0.6_f64, t.x);
        assert!((s1 - 1.336 * s2).abs() < 1e-12);
        // Total internal reflection.
        assert!(refract(DVec3::new(0.9, 0.0, -0.436), n, 1.5, 1.0).is_none());
    }
}
