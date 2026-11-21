//! What part of the shadow casters a camera sees, so a shadow map covers that part instead of the
//! whole character (todo 1.9: shadow quality on the face).
//!
//! A 2048 map over the bounding sphere of a standing body has texels of about 1.2 mm: an eyelid
//! fold or a nostril rim is one or two texels. [`CasterSlabs`] keeps sixteen boxes of the casters,
//! one per band of height (a head is narrow, the arms are wide, so one box of the whole body would
//! give nothing in a profile view); [`CasterSlabs::visible`] clips them against the camera's
//! frustum and returns the sphere around what is left. `ShadowFrustum::fit_focused` in
//! [`crate::light`] fits the map to that sphere and keeps the depth range of all casters, so a
//! caster outside the view still shadows what is inside it.
//!
//! Everything here works on the stack: it runs every frame `[PERF-R1]`.

use humanart_math::{Aabb, Sphere, Vec3};

use crate::camera::Camera;

/// Bands of height the casters are binned into.
pub const CASTER_SLABS: usize = 16;
/// How much wider than the camera's field of view the clipping frustum is: the TAA jitter, the
/// shadow filter and the subsurface blur read a little past the image edge.
const FRUSTUM_MARGIN: f32 = 1.04;
/// Most corners of a box face clipped by the five planes of the frustum (4 + 5), with headroom.
const MAX_CORNERS: usize = 12;

/// The shadow casters as boxes per band of height, in world space at upload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CasterSlabs {
    boxes: [Aabb; CASTER_SLABS],
    low: f32,
    high: f32,
}

impl CasterSlabs {
    /// No casters.
    pub const EMPTY: Self = Self { boxes: [Aabb::EMPTY; CASTER_SLABS], low: 0.0, high: 0.0 };

    /// Empty slabs for casters whose heights lie in `[low, high]` (points outside go to the first
    /// or the last band).
    pub fn new(low: f32, high: f32) -> Self {
        let ok = low.is_finite() && high.is_finite() && high > low;
        Self { boxes: [Aabb::EMPTY; CASTER_SLABS], low: if ok { low } else { 0.0 }, high: if ok { high } else { 0.0 } }
    }

    /// Adds a caster point.
    pub fn add(&mut self, p: Vec3) {
        if !p.is_finite() {
            return;
        }
        let span = self.high - self.low;
        let t = if span > 0.0 { ((p.y - self.low) / span).clamp(0.0, 1.0) } else { 0.0 };
        let band = ((t * CASTER_SLABS as f32) as usize).min(CASTER_SLABS - 1);
        if let Some(b) = self.boxes.get_mut(band) {
            *b = b.extend(p);
        }
    }

    /// The boxes (empty where a band has no caster).
    pub const fn boxes(&self) -> &[Aabb; CASTER_SLABS] {
        &self.boxes
    }

    /// The sphere around the part of the boxes inside the frustum of `camera` at `aspect` (width
    /// over height of the whole image, not of a still's tile); `None` when the camera sees none
    /// of them.
    pub fn visible(&self, camera: &Camera, aspect: f32) -> Option<Sphere> {
        let forward = camera.forward();
        let right = forward.cross(camera.up).try_normalize().unwrap_or_else(|| forward.any_orthonormal_vector());
        let up = right.cross(forward);
        let eye = camera.position;
        let tan_y = (0.5 * camera.fov_y()).tan() * FRUSTUM_MARGIN;
        let tan_x = tan_y * aspect.max(1e-3);
        let near = camera.near.max(1e-4);
        let to_view = |p: Vec3| {
            let d = p - eye;
            Vec3::new(d.dot(right), d.dot(up), d.dot(forward))
        };
        // Inside is where every one of these is >= 0 (view space: x right, y up, z ahead).
        let planes: [Plane; 5] = [
            (Vec3::Z, -near),
            (Vec3::new(-1.0, 0.0, tan_x), 0.0),
            (Vec3::new(1.0, 0.0, tan_x), 0.0),
            (Vec3::new(0.0, -1.0, tan_y), 0.0),
            (Vec3::new(0.0, 1.0, tan_y), 0.0),
        ];
        let mut seen = Aabb::EMPTY;
        for b in self.boxes.iter().filter(|b| !b.is_empty()) {
            let corner = |i: usize| {
                to_view(Vec3::new(
                    if i & 1 == 0 { b.min.x } else { b.max.x },
                    if i & 2 == 0 { b.min.y } else { b.max.y },
                    if i & 4 == 0 { b.min.z } else { b.max.z },
                ))
            };
            for face in BOX_FACES {
                let mut poly = [Vec3::ZERO; MAX_CORNERS];
                let mut count = 4;
                for (slot, &i) in poly.iter_mut().zip(&face) {
                    *slot = corner(i);
                }
                for plane in planes {
                    (poly, count) = clip(&poly, count, plane);
                }
                for p in poly.iter().take(count) {
                    seen = seen.extend(*p);
                }
            }
            // A camera inside the box: the near rectangle is part of what it sees.
            for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let v = Vec3::new(sx * tan_x * near, sy * tan_y * near, near);
                let w = eye + right * v.x + up * v.y + forward * v.z;
                let inside = w.x >= b.min.x
                    && w.x <= b.max.x
                    && w.y >= b.min.y
                    && w.y <= b.max.y
                    && w.z >= b.min.z
                    && w.z <= b.max.z;
                if inside {
                    seen = seen.extend(v);
                }
            }
        }
        if seen.is_empty() {
            return None;
        }
        let c = seen.center();
        let radius = seen.half_extents().length().max(1e-3);
        Some(Sphere { center: eye + right * c.x + up * c.y + forward * c.z, radius })
    }
}

/// The corner indices (bit 0 x, bit 1 y, bit 2 z) of the six faces of a box, each in order around
/// the face.
const BOX_FACES: [[usize; 4]; 6] = [[0, 2, 6, 4], [1, 3, 7, 5], [0, 1, 5, 4], [2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6]];

/// A half space in view coordinates: inside is where `normal . p + offset >= 0`.
type Plane = (Vec3, f32);

/// Sutherland-Hodgman: the part of the convex polygon `poly[..count]` inside `plane`.
fn clip(poly: &[Vec3; MAX_CORNERS], count: usize, plane: Plane) -> ([Vec3; MAX_CORNERS], usize) {
    let mut out = [Vec3::ZERO; MAX_CORNERS];
    let mut n = 0;
    let mut push = |p: Vec3| {
        if let Some(slot) = out.get_mut(n) {
            *slot = p;
            n += 1;
        }
    };
    for i in 0..count.min(MAX_CORNERS) {
        let a = poly[i];
        let b = poly[(i + 1) % count];
        let (da, db) = (plane.0.dot(a) + plane.1, plane.0.dot(b) + plane.1);
        if da >= 0.0 {
            push(a);
        }
        if (da >= 0.0) != (db >= 0.0) {
            let t = da / (da - db);
            push(a + (b - a) * t);
        }
    }
    (out, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A standing figure: a narrow head on top of a wide band of arms.
    fn figure() -> CasterSlabs {
        let mut s = CasterSlabs::new(0.0, 1.8);
        for i in 0..200 {
            let t = i as f32 / 199.0;
            // Legs and trunk: 0.2 m wide; the arms at 1.4 m reach 0.8 m to each side.
            s.add(Vec3::new(0.2 * (t - 0.5), 1.5 * t, 0.1));
            s.add(Vec3::new(0.2 * (t - 0.5), 1.5 * t, -0.1));
            s.add(Vec3::new(1.6 * (t - 0.5), 1.4, 0.0));
            // The head: a 0.2 m ball at 1.65 m.
            let a = t * std::f32::consts::TAU;
            s.add(Vec3::new(0.1 * a.cos(), 1.65 + 0.1 * a.sin(), 0.1));
            s.add(Vec3::new(0.1 * a.cos(), 1.65 + 0.1 * a.sin(), -0.1));
        }
        s
    }

    fn portrait(azimuth: f32) -> Camera {
        let center = Vec3::new(0.0, 1.65, 0.0);
        Camera {
            position: center + Vec3::new(azimuth.sin(), 0.0, azimuth.cos()) * 1.0,
            target: center,
            near: 0.05,
            ..Camera::default()
        }
    }

    #[test]
    fn a_portrait_sees_only_the_head_bands() {
        let slabs = figure();
        let whole = 0.5 * Vec3::new(1.6, 1.8, 0.2).length();
        // In profile the view runs along the arms, whose band the widening frustum touches.
        for (azimuth, part) in
            [(0.0, 0.35), (0.8, 0.35), (std::f32::consts::FRAC_PI_2, 0.45), (std::f32::consts::PI, 0.35)]
        {
            let camera = portrait(azimuth);
            let s = slabs.visible(&camera, 0.8).unwrap();
            assert!(s.radius < part * whole, "azimuth {azimuth}: {} of {whole}", s.radius);
            // Near the line of sight through the head (in profile the arm behind the head moves
            // the centre along that line, not off it).
            let off = s.center - camera.target;
            let across = off - camera.forward() * off.dot(camera.forward());
            assert!((s.center.y - 1.65).abs() < 0.15 && across.length() < 0.2, "{s:?}");
        }
    }

    #[test]
    fn everything_the_camera_sees_is_inside_the_sphere() {
        let slabs = figure();
        let mut rng = humanart_core::rng::Rng::new(7);
        for _ in 0..40 {
            let azimuth = rng.f32() * std::f32::consts::TAU;
            let mut camera = portrait(azimuth);
            camera.target.y = 0.4 + 1.4 * rng.f32();
            camera.position.y = camera.target.y + 0.4 * (rng.f32() - 0.5);
            camera.focal_mm = 35.0 + 100.0 * rng.f32();
            let aspect = 0.6 + rng.f32();
            let Some(s) = slabs.visible(&camera, aspect) else { continue };
            let m = camera.projection(aspect) * camera.view();
            // Points of every box that project into the image are inside the sphere.
            for b in slabs.boxes().iter().filter(|b| !b.is_empty()) {
                for i in 0..512 {
                    let f = Vec3::new((i % 8) as f32 / 7.0, ((i / 8) % 8) as f32 / 7.0, (i / 64) as f32 / 7.0);
                    let p = b.min + (b.max - b.min) * f;
                    if (p - camera.position).dot(camera.forward()) < camera.near {
                        continue;
                    }
                    let ndc = m.project_point3(p);
                    if ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0 {
                        assert!(p.distance(s.center) <= s.radius * 1.001, "{p:?} outside {s:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_camera_that_looks_away_sees_nothing_and_bad_points_are_ignored() {
        let slabs = figure();
        let mut away = portrait(0.0);
        away.target = away.position + Vec3::Z;
        assert_eq!(slabs.visible(&away, 1.0), None);
        let mut s = CasterSlabs::new(1.0, 1.0);
        s.add(Vec3::new(f32::NAN, 0.0, 0.0));
        assert!(s.boxes().iter().all(|b| b.is_empty()));
        s.add(Vec3::new(0.0, 5.0, 0.0));
        assert_eq!(s.boxes().iter().filter(|b| !b.is_empty()).count(), 1);
        assert_eq!(CasterSlabs::EMPTY.visible(&portrait(0.0), 1.0), None);
    }
}
