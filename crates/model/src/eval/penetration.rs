//! Collision and penetration guards `[EVAL-08]`: eyelid–eyeball and lip–teeth.
//!
//! Expression shapes and lid helper joints can push lid skin into the eyeball (a closed lid over
//! an eye looking up, a squint over the corneal bulge) and lips into the teeth. The guard runs
//! after the evaluation (skinning) on the deformed positions and moves every guarded vertex that
//! lies inside a collider back onto its surface plus a clearance — the solved corrective of
//! `[EVAL-08]`, computed per evaluation rather than baked (it depends on the gaze, which the
//! shapes do not know). [`PenetrationGuard::measure`] is the automated check of the quality
//! metrics (section 10.8): the deepest penetration and the number of penetrating vertices.
//!
//! Colliders:
//! - **Eyeball**: the analytic eye of [`crate::eye::EyeGeometry`] (scleral sphere with the
//!   corneal cap), placed by the sphere fitted to the deformed eyeball part
//!   ([`crate::eye::EyeVertices::frame`]), so it follows the gaze. Guarded: the `body` vertices
//!   of the eye region (`eye-L` / `eye-R` subtree: lids, canthi, socket) within 10 mm of the
//!   eyeball's surface at rest.
//! - **Teeth**: one ellipsoid per tooth (each connected piece of `teeth-upper` / `teeth-lower`),
//!   its axes the principal axes of the tooth's rest vertices and its semi-axes their extents,
//!   moved rigidly with the tooth (a frame from the tooth's centroid and two of its vertices).
//!   Guarded: the `body` vertices of the lip regions (`upper-lip`, `lower-lip`, the lip corners).
//!
//! A vertex is moved radially (eyeball) or along the scaled radius (ellipsoid) to the surface;
//! vertices outside are untouched, so the correction is continuous and zero when nothing
//! penetrates. The caller recomputes normals afterwards (`eval::normals`). [`PenetrationGuard::solve`]
//! allocates nothing.

use humanart_math::{DVec3, Vec3};

use crate::error::{ModelError, Result};
use crate::eye::{EyeGeometry, EyeParams, EyeVertices};
use crate::template::Template;

/// Default clearance kept between lid skin and the eyeball, metres: the tear film is a few
/// micrometres thick, but the eyeball mesh (a subdivision fit of the analytic surface) deviates
/// from it by up to a few hundredths of a millimetre near the limbus, so 0.1 mm keeps the lids
/// clear of the mesh too.
pub const EYE_CLEARANCE: f32 = 1.0e-4;

/// Default clearance kept between lip skin and the teeth, metres.
pub const TEETH_CLEARANCE: f32 = 1.0e-4;

/// Penetration statistics of one collider kind.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Penetration {
    /// Deepest penetration, metres (0 when none).
    pub max_depth: f32,
    /// Vertices inside (closer than the surface; the clearance is not counted).
    pub count: usize,
}

impl Penetration {
    fn add(&mut self, depth: f32) {
        if depth > 0.0 {
            self.max_depth = self.max_depth.max(depth);
            self.count += 1;
        }
    }
}

/// The result of a check or a solve (for a solve: before the correction).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PenetrationReport {
    /// Eyelid–eyeball.
    pub eyelid: Penetration,
    /// Lip–teeth.
    pub lip: Penetration,
}

/// One eye's collider and guarded vertices.
#[derive(Clone, Debug)]
struct EyeGuard {
    vertices: EyeVertices,
    guarded: Vec<u32>,
}

/// One tooth's ellipsoid in its rest pose and the vertices that carry it.
#[derive(Clone, Debug)]
struct Tooth {
    vertices: Vec<u32>,
    /// Two reference vertices for the rigid frame.
    refs: [u32; 2],

    /// Ellipsoid axes in the rest frame's coordinates (rows: unit axes).
    axes: [Vec3; 3],
    /// Semi-axes along `axes`, metres.
    semi: [f32; 3],
    /// Ellipsoid centre relative to the rest centroid, in rest frame coordinates.
    offset: Vec3,
}

/// The guard of one template. See the module documentation.
#[derive(Clone, Debug)]
pub struct PenetrationGuard {
    eye: EyeParams,
    eyes: Vec<EyeGuard>,
    teeth: Vec<Tooth>,
    lips: Vec<u32>,
    /// Clearance to the eyeball, metres.
    pub eye_clearance: f32,
    /// Clearance to the teeth, metres.
    pub teeth_clearance: f32,
}

/// The vertices used by faces of part `name` (sorted, unique); empty when there is no such part.
#[must_use]
pub fn part_vertices(template: &Template, name: &str) -> Vec<u32> {
    let mesh = template.mesh();
    let mut used = vec![false; mesh.vertex_count()];
    for p in mesh.parts().iter().filter(|p| p.name == name) {
        for f in p.faces() {
            for &v in mesh.face(f) {
                if let Some(u) = used.get_mut(v as usize) {
                    *u = true;
                }
            }
        }
    }
    (0..used.len() as u32).filter(|&v| used[v as usize]).collect()
}

/// An orthonormal frame from a centre and two points.
fn frame_of(c: Vec3, a: Vec3, b: Vec3) -> Option<[Vec3; 3]> {
    let e1 = (a - c).try_normalize()?;
    let e2 = (b - c).reject_from(e1).try_normalize()?;
    Some([e1, e2, e1.cross(e2)])
}

/// Eigenvectors of a symmetric 3×3 matrix (Jacobi rotations), as rows.
fn principal_axes(mut m: [[f64; 3]; 3]) -> [Vec3; 3] {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0_f64]];
    for _ in 0..32 {
        let (mut p, mut q, mut big) = (0, 1, 0.0_f64);
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            if m[i][j].abs() > big {
                (p, q, big) = (i, j, m[i][j].abs());
            }
        }
        if big < 1e-18 {
            break;
        }
        let theta = 0.5 * (2.0 * m[p][q]).atan2(m[q][q] - m[p][p]);
        let (s, c) = theta.sin_cos();
        for row in &mut m {
            let (mkp, mkq) = (row[p], row[q]);
            row[p] = c * mkp - s * mkq;
            row[q] = s * mkp + c * mkq;
        }
        let (rp, rq) = (m[p], m[q]);
        for k in 0..3 {
            m[p][k] = c * rp[k] - s * rq[k];
            m[q][k] = s * rp[k] + c * rq[k];
        }
        for row in &mut v {
            let (a, b) = (row[p], row[q]);
            row[p] = c * a - s * b;
            row[q] = s * a + c * b;
        }
    }
    // Columns of v are the eigenvectors.
    [0, 1, 2].map(|k| Vec3::new(v[0][k] as f32, v[1][k] as f32, v[2][k] as f32).normalize_or(Vec3::X))
}

/// The union-find root of `v` (with path halving); `parent` holds valid indices only.
fn root(parent: &mut [u32], mut v: u32) -> u32 {
    while parent[v as usize] != v {
        let up = parent[parent[v as usize] as usize];
        parent[v as usize] = up;
        v = up;
    }
    v
}

/// Splits the vertices of `part` into connected pieces (faces sharing a position key).
fn pieces(template: &Template, part: &str) -> Vec<Vec<u32>> {
    let mesh = template.mesh();
    if mesh.vertex_count() == 0 {
        return Vec::new();
    }
    let keys = mesh.position_keys();
    let n = mesh.vertex_count();
    let key = |v: u32| keys.get(v as usize).copied().unwrap_or(v).min(n.saturating_sub(1) as u32);
    let mut parent: Vec<u32> = (0..n as u32).collect();
    let verts = part_vertices(template, part);
    for p in mesh.parts().iter().filter(|p| p.name == part) {
        for f in p.faces() {
            let face = mesh.face(f);
            let Some(&first) = face.first() else { continue };
            let a = root(&mut parent, key(first));
            for &v in &face[1..] {
                let b = root(&mut parent, key(v));
                parent[b as usize] = a;
            }
        }
    }
    let mut groups: std::collections::BTreeMap<u32, Vec<u32>> = std::collections::BTreeMap::new();
    for v in verts {
        let r = root(&mut parent, key(v));
        groups.entry(r).or_default().push(v);
    }
    groups.into_values().collect()
}

fn tooth(positions: &[Vec3], vertices: Vec<u32>) -> Option<Tooth> {
    let n = vertices.len() as f32;
    if vertices.len() < 4 {
        return None;
    }
    let c = vertices.iter().map(|&v| positions[v as usize]).fold(Vec3::ZERO, |s, p| s + p) / n;
    let far = |from: Vec3, score: &dyn Fn(Vec3) -> f32| {
        vertices
            .iter()
            .copied()
            .max_by(|&a, &b| score(positions[a as usize] - from).total_cmp(&score(positions[b as usize] - from)))
    };
    let r0 = far(c, &|d| d.length_squared())?;
    let d0 = positions[r0 as usize] - c;
    let r1 = far(c, &|d| d.cross(d0).length_squared())?;
    let rest_frame = frame_of(c, positions[r0 as usize], positions[r1 as usize])?;
    let local = |p: Vec3| {
        let d = p - c;
        Vec3::new(d.dot(rest_frame[0]), d.dot(rest_frame[1]), d.dot(rest_frame[2]))
    };
    let mut cov = [[0.0_f64; 3]; 3];
    for &v in &vertices {
        let d = local(positions[v as usize]).as_dvec3();
        let a = [d.x, d.y, d.z];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += a[i] * a[j];
            }
        }
    }
    let axes = principal_axes(cov);
    // Extents along each axis (box), centred.
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for &v in &vertices {
        let d = local(positions[v as usize]);
        for k in 0..3 {
            let x = d.dot(axes[k]);
            lo[k] = lo[k].min(x);
            hi[k] = hi[k].max(x);
        }
    }
    let semi = [0, 1, 2].map(|k| ((hi[k] - lo[k]) * 0.5).max(1e-5));
    let offset = axes[0] * f32::midpoint(hi[0], lo[0])
        + axes[1] * f32::midpoint(hi[1], lo[1])
        + axes[2] * f32::midpoint(hi[2], lo[2]);
    Some(Tooth { vertices, refs: [r0, r1], axes, semi, offset })
}

impl Tooth {
    /// The ellipsoid on deformed positions: centre and world axes.
    fn pose(&self, positions: &[Vec3]) -> Option<(Vec3, [Vec3; 3])> {
        let mut c = Vec3::ZERO;
        for &v in &self.vertices {
            c += *positions.get(v as usize)?;
        }
        c /= self.vertices.len() as f32;
        let now = frame_of(c, *positions.get(self.refs[0] as usize)?, *positions.get(self.refs[1] as usize)?)?;
        let to_world = |l: Vec3| now[0] * l.x + now[1] * l.y + now[2] * l.z;
        Some((c + to_world(self.offset), self.axes.map(to_world)))
    }
}

/// The body vertices whose region lies in the subtree of `region`.
fn region_vertices(template: &Template, body: &[u32], region: &str) -> Result<Vec<u32>> {
    let regions = template.regions();
    let r = regions.find(region).ok_or(ModelError::invalid("penetration guard", 0, "region missing"))?;
    Ok(body
        .iter()
        .copied()
        .filter(|&v| regions.vertex_regions.get(v as usize).is_some_and(|&x| regions.is_within(x, r)))
        .collect())
}

impl PenetrationGuard {
    /// Builds the guard for a template with the eyes of `eye` (its corneal shape; the eyeball
    /// radius is fitted). Needs the parts `body`, `eyeball-L`, `eyeball-R`; the teeth guard is
    /// built when `teeth-upper` / `teeth-lower` exist.
    ///
    /// # Errors
    /// [`ModelError::Invalid`] when a part or region is missing or the eyeballs are not spheres
    /// with a corneal bulge.
    pub fn new(template: &Template, eye: &EyeParams) -> Result<Self> {
        let positions = template.mesh().positions();
        let body = part_vertices(template, "body");
        if body.is_empty() {
            return Err(ModelError::invalid("penetration guard", 1, "the template has no body part"));
        }
        let mut eyes = Vec::new();
        for (part, region) in [("eyeball-L", "eye-L"), ("eyeball-R", "eye-R")] {
            let vertices = EyeVertices::find(template, part)?;
            let frame = vertices.frame(positions).ok_or(ModelError::invalid(
                "penetration guard",
                2,
                "the eyeball does not fit a sphere",
            ))?;
            let guarded = region_vertices(template, &body, region)?
                .into_iter()
                .filter(|&v| positions[v as usize].distance(frame.centre) < frame.radius + 0.01)
                .collect();
            eyes.push(EyeGuard { vertices, guarded });
        }
        let mut teeth = Vec::new();
        for part in ["teeth-upper", "teeth-lower"] {
            for piece in pieces(template, part) {
                if let Some(t) = tooth(positions, piece) {
                    teeth.push(t);
                }
            }
        }
        let mut lips = Vec::new();
        if !teeth.is_empty() {
            for region in ["upper-lip", "lower-lip", "lip-corner-L", "lip-corner-R"] {
                lips.extend(region_vertices(template, &body, region)?);
            }
        }
        Ok(Self { eye: *eye, eyes, teeth, lips, eye_clearance: EYE_CLEARANCE, teeth_clearance: TEETH_CLEARANCE })
    }

    /// Every guarded vertex: the eyelid vertices of both eyes, then the lip vertices (a vertex
    /// guarded by both comes twice). A caller mirroring the correction into another copy of the
    /// surface (the GPU deformation's outputs) copies these.
    pub fn guarded(&self) -> impl Iterator<Item = u32> + '_ {
        self.eyes.iter().flat_map(|e| e.guarded.iter().copied()).chain(self.lips.iter().copied())
    }

    /// The number of guarded vertices (eyes, lips) and of tooth colliders.
    pub fn counts(&self) -> (usize, usize, usize) {
        (self.eyes.iter().map(|e| e.guarded.len()).sum(), self.lips.len(), self.teeth.len())
    }

    /// One eye collider on `positions`: centre, analytic geometry (millimetres) and the local
    /// basis (side, third, axis).
    fn eye_pose(&self, eye: &EyeGuard, positions: &[Vec3]) -> Option<EyePose> {
        let frame = eye.vertices.frame(positions)?;
        let mut p = self.eye;
        let s = frame.radius * 1000.0 / p.eyeball_radius_mm.max(1e-3);
        p.eyeball_radius_mm *= s;
        p.limbus_radius_mm *= s;
        p.cornea_radius_mm *= s;
        p.cornea_back_radius_mm *= s;
        let axis = frame.axis.as_dvec3();
        let side = axis.any_orthonormal_vector();
        Some(EyePose { centre: frame.centre, geometry: EyeGeometry::new(&p), basis: [side, axis.cross(side), axis] })
    }

    /// The penetration depth of `pos` into the eye (metres; negative outside) and where it must
    /// go when it is closer than the clearance.
    fn eye_fix(&self, pose: &EyePose, pos: Vec3) -> Option<(f32, Vec3)> {
        let q = (pos - pose.centre).as_dvec3() * 1000.0;
        let dist = q.length();
        if dist < 1e-9 {
            return None;
        }
        let d = q / dist;
        let [side, third, axis] = pose.basis;
        let surface = pose.geometry.surface_distance(DVec3::new(d.dot(side), d.dot(third), d.dot(axis)));
        let wanted = surface + f64::from(self.eye_clearance) * 1000.0;
        // A vertex already put at the clearance (to within f32 rounding, 0.1 µm) stays.
        if dist >= wanted - 1e-4 {
            return None;
        }
        let out = d * (wanted * 1e-3);
        Some((((surface - dist) * 1e-3) as f32, pose.centre + Vec3::new(out.x as f32, out.y as f32, out.z as f32)))
    }

    /// The penetration depth of `pos` into a tooth ellipsoid (centre, world axes) and where it
    /// must go.
    fn tooth_fix(&self, tooth: &Tooth, c: Vec3, axes: &[Vec3; 3], pos: Vec3) -> Option<(f32, Vec3)> {
        let d = pos - c;
        let reach = tooth.semi.iter().copied().fold(0.0, f32::max) + self.teeth_clearance;
        if d.length_squared() > reach * reach {
            return None;
        }
        let u =
            Vec3::new(d.dot(axes[0]) / tooth.semi[0], d.dot(axes[1]) / tooth.semi[1], d.dot(axes[2]) / tooth.semi[2]);
        let e = u.length();
        if !(1e-6..1.0).contains(&e) {
            return None;
        }
        // Depth along the scaled radius; the surface point pushed out by the clearance.
        let len = d.length();
        let surface = d / e;
        Some((len / e - len, c + surface + surface.normalize_or_zero() * self.teeth_clearance))
    }

    /// The automated penetration check of the quality metrics (section 10.8).
    pub fn measure(&self, positions: &[Vec3]) -> PenetrationReport {
        let mut report = PenetrationReport::default();
        for eye in &self.eyes {
            let Some(pose) = self.eye_pose(eye, positions) else { continue };
            for &v in &eye.guarded {
                if let Some((depth, _)) = positions.get(v as usize).and_then(|&p| self.eye_fix(&pose, p)) {
                    report.eyelid.add(depth);
                }
            }
        }
        for tooth in &self.teeth {
            let Some((c, axes)) = tooth.pose(positions) else { continue };
            for &v in &self.lips {
                if let Some((depth, _)) = positions.get(v as usize).and_then(|&p| self.tooth_fix(tooth, c, &axes, p)) {
                    report.lip.add(depth);
                }
            }
        }
        report
    }

    /// Corrects `positions` in place; returns the penetration found before the correction. The
    /// colliders are posed from their own vertices, which no guard moves. No allocation.
    pub fn solve(&self, positions: &mut [Vec3]) -> PenetrationReport {
        let mut report = PenetrationReport::default();
        for eye in &self.eyes {
            let Some(pose) = self.eye_pose(eye, positions) else { continue };
            for &v in &eye.guarded {
                let Some(p) = positions.get_mut(v as usize) else { continue };
                if let Some((depth, to)) = self.eye_fix(&pose, *p) {
                    report.eyelid.add(depth);
                    *p = to;
                }
            }
        }
        for tooth in &self.teeth {
            let Some((c, axes)) = tooth.pose(positions) else { continue };
            for &v in &self.lips {
                let Some(p) = positions.get_mut(v as usize) else { continue };
                if let Some((depth, to)) = self.tooth_fix(tooth, c, &axes, *p) {
                    report.lip.add(depth);
                    *p = to;
                }
            }
        }
        report
    }
}

/// One eye collider posed on a deformed mesh.
struct EyePose {
    centre: Vec3,
    geometry: EyeGeometry,
    basis: [DVec3; 3],
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, OnceLock};

    use humanart_math::Quat;

    use super::*;
    use crate::eval::{EvalInput, Evaluator};
    use crate::expression::controls::ControlRig;
    use crate::eye::Gaze;
    use crate::generate::build::{Options, template};

    fn generated() -> Arc<Template> {
        static T: OnceLock<Arc<Template>> = OnceLock::new();
        T.get_or_init(|| Arc::new(template(&Options { lods: false, ..Options::default() }).unwrap().template)).clone()
    }

    /// Closest point on triangle (a, b, c) to p (Ericson, Real-Time Collision Detection 5.1.5).
    fn closest(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
        let (ab, ac, ap) = (b - a, c - a, p - a);
        let (d1, d2) = (ab.dot(ap), ac.dot(ap));
        if d1 <= 0.0 && d2 <= 0.0 {
            return a;
        }
        let bp = p - b;
        let (d3, d4) = (ab.dot(bp), ac.dot(bp));
        if d3 >= 0.0 && d4 <= d3 {
            return b;
        }
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            return a + ab * (d1 / (d1 - d3));
        }
        let cp = p - c;
        let (d5, d6) = (ab.dot(cp), ac.dot(cp));
        if d6 >= 0.0 && d5 <= d6 {
            return c;
        }
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            return a + ac * (d2 / (d2 - d6));
        }
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
        }
        let denom = 1.0 / (va + vb + vc);
        a + ab * (vb * denom) + ac * (vc * denom)
    }

    /// Independent check against the deformed eyeball MESH (not the analytic collider): the
    /// deepest guarded vertex below the nearest eyeball triangle (signed by the triangle's
    /// outward normal), metres.
    fn mesh_depth(guard: &PenetrationGuard, eyeball_tris: &[[u32; 3]], pos: &[Vec3]) -> f32 {
        let mut worst = 0.0_f32;
        for eye in &guard.eyes {
            let Some(frame) = eye.vertices.frame(pos) else { continue };
            let Some(pose) = guard.eye_pose(eye, pos) else { continue };
            for &v in &eye.guarded {
                let p = pos[v as usize];
                // Only vertices within 0.5 mm of the collider can be under the mesh.
                let q = (p - pose.centre).as_dvec3() * 1000.0;
                let d = q.normalize_or_zero();
                let [a, b, c] = pose.basis;
                let surface = pose.geometry.surface_distance(DVec3::new(d.dot(a), d.dot(b), d.dot(c)));
                if q.length() - surface > 0.5 {
                    continue;
                }
                let mut best = (f32::MAX, 0.0_f32);
                for t in eyeball_tris {
                    let [a, b, c] = t.map(|i| pos[i as usize]);
                    let q = closest(p, a, b, c);
                    let d = p.distance(q);
                    if d < best.0 {
                        let mut n = (b - a).cross(c - a).normalize_or_zero();
                        if n.dot(a - frame.centre) < 0.0 {
                            n = -n;
                        }
                        best = (d, (p - q).dot(n));
                    }
                }
                worst = worst.max(-best.1);
            }
        }
        worst
    }

    #[test]
    fn eval_08_guard_builds_and_cleans_the_rest_shape() {
        let t = generated();
        let guard = PenetrationGuard::new(&t, &EyeParams::default()).unwrap();
        let (eyes, lips, teeth) = guard.counts();
        println!("guarded eye vertices {eyes}, lip vertices {lips}, tooth colliders {teeth}");
        assert!(eyes > 100 && lips > 100 && teeth == 28, "{eyes} {lips} {teeth}");
        // The rest shape is reported (the generated lids touch the corneal bulge); after the
        // solve nothing is inside.
        let mut pos = t.mesh().positions().to_vec();
        let report = guard.solve(&mut pos);
        println!("rest before the guard: {report:?}");
        let after = guard.measure(&pos);
        assert_eq!(after.eyelid.count + after.lip.count, 0, "{after:?}");
    }

    #[test]
    fn eval_08_solve_moves_penetrating_vertices_out_and_nothing_else() {
        let t = generated();
        let guard = PenetrationGuard::new(&t, &EyeParams::default()).unwrap();
        let mut pos = t.mesh().positions().to_vec();
        guard.solve(&mut pos);
        // Push some lid vertices 1 mm into the left eyeball.
        let eye = &guard.eyes[0];
        let frame = eye.vertices.frame(&pos).unwrap();
        let pushed: Vec<u32> = eye.guarded.iter().copied().take(20).collect();
        for &v in &pushed {
            let p = pos[v as usize];
            pos[v as usize] = frame.centre + (p - frame.centre).normalize() * (frame.radius - 0.001);
        }
        let untouched = pos.clone();
        let before = guard.solve(&mut pos);
        assert_eq!(before.eyelid.count, pushed.len());
        assert!(before.eyelid.max_depth > 0.0009, "{before:?}");
        let after = guard.measure(&pos);
        assert_eq!(after.eyelid.count, 0);
        let moved = pos.iter().zip(&untouched).filter(|(a, b)| a != b).count();
        assert_eq!(moved, pushed.len());
        // Solving again changes nothing.
        let again = pos.clone();
        guard.solve(&mut pos);
        assert_eq!(pos, again);
    }

    #[test]
    fn eval_08_the_evaluator_runs_the_guard_before_the_normals() {
        let t = generated();
        let guard = PenetrationGuard::new(&t, &EyeParams::default()).unwrap();
        assert!(guard.guarded().count() > 200);
        let mut ev = Evaluator::new(t.clone());
        assert!(guard.measure(ev.positions()).eyelid.count > 0, "the rest lids touch the corneal bulge");
        let unguarded_normals = ev.normals().to_vec();
        ev.set_guard(Some(guard.clone()));
        assert_eq!(guard.measure(ev.positions()).eyelid.count, 0, "set_guard corrects the current output");
        assert_ne!(ev.normals(), &unguarded_normals[..], "and recomputes the normals");
        // A lid-closing expression on a gaze: corrected inside evaluate, normals unit length.
        let closing = t.expressions().expressions.iter().position(|e| e.name.contains("AU43")).unwrap();
        let mut weights = vec![0.0; t.expressions().expressions.len()];
        weights[closing] = 0.75;
        ev.evaluate(&EvalInput { expressions: &weights, ..EvalInput::default() }).unwrap();
        assert!(ev.guard_report().eyelid.count > 0, "{:?}", ev.guard_report());
        assert_eq!(guard.measure(ev.positions()).eyelid.count, 0);
        assert!(ev.normals().iter().all(|n| (n.length() - 1.0).abs() < 1e-3 || n.length() == 0.0));
        ev.set_guard(None);
        ev.evaluate(&EvalInput { expressions: &weights, ..EvalInput::default() }).unwrap();
        assert!(guard.measure(ev.positions()).eyelid.count > 0, "off again: the raw evaluation");
    }

    /// `[AC-WP16-1]` (geometry) `[EVAL-08]`: every control of the expression rig alone at 0.25,
    /// 0.5, 0.75 and 1, and the lid controls combined with gazes at the limits: after the guard
    /// no lid vertex is inside the eyeball, checked against the analytic collider and
    /// independently against the deformed eyeball mesh.
    #[test]
    fn ac_wp16_1_no_eyelid_penetration_across_the_expression_sweep() {
        let t = generated();
        let rig = ControlRig::builtin(&t).unwrap();
        let guard = PenetrationGuard::new(&t, &EyeParams::default()).unwrap();
        let mesh = t.mesh();
        let tris = mesh.triangulate();
        let eyeball_tris: Vec<[u32; 3]> = (0..tris.len())
            .filter(|&i| {
                let f = tris.triangle_faces[i] as usize;
                mesh.part_of_face(f).and_then(|p| mesh.parts().get(p)).is_some_and(|p| p.name.starts_with("eyeball"))
            })
            .filter_map(|i| tris.triangle(i))
            .collect();
        let mut ev = Evaluator::new(t.clone());
        let joints = t.skeleton().joints.len();
        let mut regs = vec![0.0; rig.register_count()];
        let mut weights = vec![0.0; t.expressions().expressions.len()];
        let mut pose = vec![Quat::IDENTITY; joints];
        let mut controls = vec![0.0; rig.controls().len()];
        let eye_joints: Vec<usize> =
            ["L.eye", "R.eye"].iter().map(|n| t.skeleton().find(n).unwrap() as usize).collect();
        let rest_world = t.skeleton().rest_world();
        let lid_controls: Vec<usize> = rig
            .controls()
            .iter()
            .enumerate()
            .filter(|(_, n)| ["au43", "au07", "au44", "au42", "au45", "au05", "au06"].iter().any(|k| n.contains(k)))
            .map(|(i, _)| i)
            .collect();
        let mut cases: Vec<(Vec<(usize, f32)>, Gaze)> = Vec::new();
        for c in 0..controls.len() {
            for k in 1..=4 {
                cases.push((vec![(c, k as f32 / 4.0)], Gaze::default()));
            }
        }
        let p = EyeParams::default();
        let gazes = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.7, 0.7), (-0.7, -0.7)]
            .map(|(y, x)| Gaze { yaw: y * p.gaze_limit_yaw, pitch: x * p.gaze_limit_pitch });
        for g in gazes {
            cases.push((Vec::new(), g));
            for &c in &lid_controls {
                for v in [0.5, 1.0] {
                    cases.push((vec![(c, v)], g));
                }
            }
        }
        let (mut worst_before, mut worst_after, mut worst_mesh, mut cases_with) = (0.0_f32, 0.0_f32, 0.0_f32, 0);
        let mut worst_case = String::new();
        for (set, gaze) in &cases {
            controls.fill(0.0);
            for &(c, v) in set {
                controls[c] = v;
            }
            weights.fill(0.0);
            pose.fill(Quat::IDENTITY);
            rig.evaluate(&controls, &mut regs, &mut weights, &mut pose).unwrap();
            for &j in &eye_joints {
                pose[j] = gaze.joint_rotation(rest_world[j].rotation) * pose[j];
            }
            ev.evaluate(&EvalInput { expressions: &weights, pose: &pose, ..EvalInput::default() }).unwrap();
            let mut pos = ev.positions().to_vec();
            let before = guard.solve(&mut pos);
            let after = guard.measure(&pos);
            if before.eyelid.count > 0 {
                cases_with += 1;
            }
            if before.eyelid.max_depth > worst_before {
                worst_before = before.eyelid.max_depth;
                worst_case = format!(
                    "{:?} {gaze:?}",
                    set.iter().map(|&(c, v)| (rig.controls()[c].clone(), v)).collect::<Vec<_>>()
                );
            }
            worst_after = worst_after.max(after.eyelid.max_depth);
            assert_eq!(after.eyelid.count, 0, "{set:?} {gaze:?}");
            worst_mesh = worst_mesh.max(mesh_depth(&guard, &eyeball_tris, &pos));
        }
        println!(
            "{} cases; before the guard {cases_with} had lid vertices inside the eyeball, deepest {:.3} mm ({worst_case}); after: analytic {:.4} mm, mesh {:.4} mm",
            cases.len(),
            worst_before * 1e3,
            worst_after * 1e3,
            worst_mesh * 1e3
        );
        assert_eq!(worst_after, 0.0);
        // The mesh is a subdivided fit of the analytic surface: its faces lie within a few
        // micrometres of it.
        assert!(worst_mesh < 2e-5, "{worst_mesh}");
    }
}
