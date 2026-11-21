//! The physical camera `[CAM-01]` `[LIT-03]`: focal length, sensor, orbit controls, the standard
//! review cameras, exposure, TAA jitter and the sub-frustum of a still tile `[QUAL-04]`.
//!
//! Conventions (CLAUDE.md): right-handed, Y up, metres, the character faces +Z; the projection is
//! the infinite reversed-Z perspective of `humanart_math` (near plane at depth 1, infinity at 0).

use humanart_math::color::tonemap::{EV100_RANGE, PhysicalCamera, exposure_from_ev100};
use humanart_math::{Mat4, Vec2, Vec3, pixel_offset_to_ndc};

use crate::error::RenderError;

/// Shortest focal length `[CAM-01]`, millimetres.
pub const MIN_FOCAL_MM: f32 = 14.0;
/// Longest focal length `[CAM-01]`, millimetres.
pub const MAX_FOCAL_MM: f32 = 300.0;
/// Height of a full-frame sensor, millimetres (the default).
pub const FULL_FRAME_SENSOR_HEIGHT_MM: f32 = 24.0;
/// Closest near plane, metres.
pub const MIN_NEAR: f32 = 1e-4;

/// How the image is exposed `[LIT-03]`. Auto-exposure does not exist in review: the value is locked
/// and persists with the document, so renders stay comparable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exposure {
    /// Exposure value at ISO 100.
    pub ev100: f32,
    /// Stops added on top of `ev100` (positive brightens).
    pub compensation: f32,
}

impl Exposure {
    /// An exposure at `ev100` without compensation.
    pub const fn ev100(ev100: f32) -> Self {
        Self { ev100, compensation: 0.0 }
    }

    /// The exposure of a physical camera (aperture, shutter, ISO).
    pub fn from_physical(camera: &PhysicalCamera) -> Self {
        Self::ev100(camera.ev100() as f32)
    }

    /// The factor applied to scene radiance before tone mapping: `1 / (1.2 · 2^(EV100 - c))`, with
    /// `EV100` clamped to the documented range.
    pub fn scale(&self) -> f32 {
        let ev = f64::from(self.ev100 - self.compensation).clamp(EV100_RANGE[0], EV100_RANGE[1]);
        exposure_from_ev100(ev) as f32
    }
}

impl Default for Exposure {
    /// EV100 9: a bright studio, the exposure the built-in rigs are balanced for.
    fn default() -> Self {
        Self::ev100(9.0)
    }
}

/// The standard review cameras `[CAM-01]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReviewCamera {
    /// Straight at the face.
    Front,
    /// 45 degrees to the character's left.
    ThreeQuarterLeft,
    /// 45 degrees to the character's right.
    ThreeQuarterRight,
    /// From the character's left side.
    ProfileLeft,
    /// From the character's right side.
    ProfileRight,
    /// From behind (hair).
    HairBack,
    /// A close-up at the focus point (eye, cheek, lips, ear: the caller passes the landmark).
    Macro,
    /// The whole body, front.
    FullBody,
}

impl ReviewCamera {
    /// Every review camera.
    pub const ALL: [Self; 8] = [
        Self::Front,
        Self::ThreeQuarterLeft,
        Self::ThreeQuarterRight,
        Self::ProfileLeft,
        Self::ProfileRight,
        Self::HairBack,
        Self::Macro,
        Self::FullBody,
    ];

    /// The stable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Front => "front",
            Self::ThreeQuarterLeft => "three_quarter_left",
            Self::ThreeQuarterRight => "three_quarter_right",
            Self::ProfileLeft => "profile_left",
            Self::ProfileRight => "profile_right",
            Self::HairBack => "hair_back",
            Self::Macro => "macro",
            Self::FullBody => "full_body",
        }
    }

    /// Azimuth around +Y from the +Z axis (the character's facing), radians; the character's left
    /// is +X.
    const fn azimuth(self) -> f32 {
        match self {
            Self::Front | Self::Macro | Self::FullBody => 0.0,
            Self::ThreeQuarterLeft => std::f32::consts::FRAC_PI_4,
            Self::ThreeQuarterRight => -std::f32::consts::FRAC_PI_4,
            Self::ProfileLeft => std::f32::consts::FRAC_PI_2,
            Self::ProfileRight => -std::f32::consts::FRAC_PI_2,
            Self::HairBack => std::f32::consts::PI,
        }
    }

    /// The focal length the camera uses, millimetres.
    const fn focal_mm(self) -> f32 {
        match self {
            Self::Macro => 100.0,
            Self::FullBody => 50.0,
            _ => 85.0,
        }
    }
}

/// The aperture and focus of a still's lens (todo 1.9): what the depth-of-field pass
/// (`shaders/render/dof.wgsl`) blurs by. The focal length and the sensor are the camera's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lens {
    /// The f-number (focal length over the aperture's diameter), [`Lens::F_NUMBER_RANGE`].
    pub f_number: f32,
    /// The distance of the plane in focus along the view axis, metres.
    pub focus_distance: f32,
    /// The largest circle of confusion drawn, as its radius over the image height: what lies
    /// farther out of focus is blurred no more than this (the renderer also stops at
    /// `SCREEN_SPACE_REACH` pixels).
    pub max_blur: f32,
}

impl Lens {
    /// The f-numbers accepted.
    pub const F_NUMBER_RANGE: [f32; 2] = [0.7, 64.0];
    /// The default cap of the blur radius: 3 % of the image height.
    pub const DEFAULT_MAX_BLUR: f32 = 0.03;

    /// A lens at `f_number` focused at `focus_distance` metres.
    pub const fn new(f_number: f32, focus_distance: f32) -> Self {
        Self { f_number, focus_distance, max_blur: Self::DEFAULT_MAX_BLUR }
    }

    /// Checks the parameters.
    pub fn validate(&self) -> Result<(), RenderError> {
        let ok = (Self::F_NUMBER_RANGE[0]..=Self::F_NUMBER_RANGE[1]).contains(&self.f_number)
            && self.focus_distance.is_finite()
            && self.focus_distance >= 0.01
            && (0.0..=0.25).contains(&self.max_blur);
        if ok {
            Ok(())
        } else {
            Err(RenderError::InvalidParameter {
                what: "lens: f-number 0.7-64, focus distance from 1 cm, blur cap up to a quarter of the image",
            })
        }
    }

    /// The radius in pixels of the circle of confusion of a point infinitely far away, in an
    /// image `image_height` pixels high: half of `f² / (N (s - f))` on the sensor (thin lens, `f`
    /// the focal length, `N` the f-number, `s` the focus distance), scaled to pixels. A focus
    /// distance at or inside the focal length counts as just outside it.
    pub fn confusion_at_infinity(&self, camera: &Camera, image_height: f32) -> f32 {
        let f = camera.focal_mm;
        let s = (self.focus_distance * 1000.0).max(f * 1.01);
        let on_sensor = f * f / (self.f_number.max(1e-3) * (s - f));
        0.5 * on_sensor / camera.sensor_height_mm.max(1e-3) * image_height
    }

    /// The radius in pixels of the circle of confusion of a point `distance` metres along the
    /// view axis, before the cap: the radius at infinity times `|1 - s / z|`.
    pub fn confusion(&self, camera: &Camera, image_height: f32, distance: f32) -> f32 {
        self.confusion_at_infinity(camera, image_height) * (1.0 - self.focus_distance / distance.max(1e-6)).abs()
    }
}

/// The cameras a portrait photographer sets up (todo 1.9): a focal length, a framing (which
/// gives the distance) and a camera height.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortraitPreset {
    /// Head and shoulders at 85 mm, from the front at eye height.
    HeadAndShoulders,
    /// The face alone at 105 mm.
    TightFace,
    /// The whole figure at 50 mm, the camera at waist height.
    FullBody,
    /// Head and shoulders at 85 mm from 40 degrees to the character's left, slightly above.
    ThreeQuarter,
    /// Head and shoulders at 85 mm from the character's left side.
    Profile,
}

/// What a portrait camera frames: the head and the standing figure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PortraitSubject {
    /// The centre of the head.
    pub head_center: Vec3,
    /// The head's radius (half its height), metres.
    pub head_radius: f32,
    /// The lowest point of the figure (the floor under it).
    pub feet: f32,
    /// The highest point of the figure.
    pub top: f32,
}

impl PortraitPreset {
    /// Every preset.
    pub const ALL: [Self; 5] =
        [Self::HeadAndShoulders, Self::TightFace, Self::FullBody, Self::ThreeQuarter, Self::Profile];

    /// The stable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::HeadAndShoulders => "head_and_shoulders",
            Self::TightFace => "tight_face",
            Self::FullBody => "full_body",
            Self::ThreeQuarter => "three_quarter",
            Self::Profile => "profile",
        }
    }

    /// The focal length, millimetres.
    pub const fn focal_mm(self) -> f32 {
        match self {
            Self::TightFace => 105.0,
            Self::FullBody => 50.0,
            Self::HeadAndShoulders | Self::ThreeQuarter | Self::Profile => 85.0,
        }
    }

    /// Azimuth around +Y from the character's facing (+Z) towards its left (+X), radians.
    const fn azimuth(self) -> f32 {
        match self {
            Self::HeadAndShoulders | Self::TightFace | Self::FullBody => 0.0,
            Self::ThreeQuarter => 0.698,
            Self::Profile => std::f32::consts::FRAC_PI_2,
        }
    }
}

/// A perspective camera with a physical lens `[CAM-01]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// Eye position, metres.
    pub position: Vec3,
    /// The orbit centre the camera looks at.
    pub target: Vec3,
    /// Roughly up.
    pub up: Vec3,
    /// Focal length, millimetres, `[MIN_FOCAL_MM, MAX_FOCAL_MM]`.
    pub focal_mm: f32,
    /// Sensor height, millimetres.
    pub sensor_height_mm: f32,
    /// Near plane, metres (the far plane is at infinity).
    pub near: f32,
    /// Exposure.
    pub exposure: Exposure,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 1.6, 1.2),
            target: Vec3::new(0.0, 1.6, 0.0),
            up: Vec3::Y,
            focal_mm: 85.0,
            sensor_height_mm: FULL_FRAME_SENSOR_HEIGHT_MM,
            near: 0.01,
            exposure: Exposure::default(),
        }
    }
}

impl Camera {
    /// Checks the parameters: finite, focal length in range, a positive sensor and near plane, the
    /// eye away from the target.
    pub fn validate(&self) -> Result<(), RenderError> {
        let finite = self.position.is_finite() && self.target.is_finite() && self.up.is_finite();
        if !finite || self.position.distance(self.target) <= 0.0 {
            return Err(RenderError::InvalidParameter {
                what: "camera position and target must be finite and distinct",
            });
        }
        if !(MIN_FOCAL_MM..=MAX_FOCAL_MM).contains(&self.focal_mm) {
            return Err(RenderError::InvalidParameter { what: "focal length outside 14-300 mm" });
        }
        if !(self.sensor_height_mm > 0.0 && self.sensor_height_mm.is_finite()) {
            return Err(RenderError::InvalidParameter { what: "sensor height must be positive" });
        }
        if !(self.near >= MIN_NEAR && self.near.is_finite()) {
            return Err(RenderError::InvalidParameter { what: "near plane below 0.1 mm" });
        }
        Ok(())
    }

    /// The vertical field of view, radians: `2 atan(sensor / 2f)`.
    pub fn fov_y(&self) -> f32 {
        2.0 * (self.sensor_height_mm / (2.0 * self.focal_mm)).atan()
    }

    /// World to view.
    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    /// View to clip, infinite reversed-Z `[RND-02]`.
    pub fn projection(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_infinite_reversed_z(self.fov_y(), aspect, self.near)
    }

    /// The unit view direction.
    pub fn forward(&self) -> Vec3 {
        (self.target - self.position).normalize_or(-Vec3::Z)
    }

    /// Orbits around the target by `yaw` (around world Y) and `pitch` (around the camera's right
    /// axis), radians; the pitch stops 1 degree short of the poles.
    pub fn orbit(&mut self, yaw: f32, pitch: f32) {
        let offset = self.position - self.target;
        let radius = offset.length();
        if radius <= 0.0 {
            return;
        }
        let current_pitch = (offset.y / radius).clamp(-1.0, 1.0).asin();
        let limit = std::f32::consts::FRAC_PI_2 - 1f32.to_radians();
        let new_pitch = (current_pitch + pitch).clamp(-limit, limit);
        let current_yaw = offset.x.atan2(offset.z);
        let new_yaw = current_yaw + yaw;
        let horizontal = radius * new_pitch.cos();
        self.position =
            self.target + Vec3::new(horizontal * new_yaw.sin(), radius * new_pitch.sin(), horizontal * new_yaw.cos());
    }

    /// Moves eye and target together in the view plane by `right` and `up` metres.
    pub fn pan(&mut self, right: f32, up: f32) {
        let f = self.forward();
        let r = f.cross(self.up).normalize_or(Vec3::X);
        let u = r.cross(f);
        let shift = r * right + u * up;
        self.position += shift;
        self.target += shift;
    }

    /// Scales the eye-target distance by `factor` (`< 1` moves closer), keeping it above the near plane.
    pub fn dolly(&mut self, factor: f32) {
        if !(factor > 0.0 && factor.is_finite()) {
            return;
        }
        let offset = self.position - self.target;
        let distance = (offset.length() * factor).max(self.near * 2.0);
        self.position = self.target + offset.normalize_or(Vec3::Z) * distance;
    }

    /// Turns the camera around the target by `angle` radians about world Y (a turntable step).
    pub fn turntable(&mut self, angle: f32) {
        self.orbit(angle, 0.0);
    }

    /// The camera of a portrait preset on `subject` at `aspect` (width over height). The framing
    /// sets the distance: the height the image shows at the subject is 4.6 head radii for head and
    /// shoulders (the eyes a third from the top), 2.5 for the face, the figure plus 12 % for the
    /// full body; a narrow image moves the camera back until the width shows 0.8 of that (head)
    /// or the figure's arms-down width (body).
    pub fn portrait(preset: PortraitPreset, subject: &PortraitSubject, aspect: f32) -> Self {
        let r = subject.head_radius.max(1e-3);
        let figure = (subject.top - subject.feet).max(4.0 * r);
        // What the image shows at the subject (height, least width), the point looked at, and how
        // far the camera stands above that point as a fraction of the distance.
        let (height, width, target, raise) = match preset {
            PortraitPreset::HeadAndShoulders | PortraitPreset::Profile => {
                (4.6 * r, 3.6 * r, subject.head_center - Vec3::Y * (0.75 * r), 0.0)
            }
            PortraitPreset::ThreeQuarter => (4.6 * r, 3.6 * r, subject.head_center - Vec3::Y * (0.75 * r), 0.07),
            PortraitPreset::TightFace => (2.5 * r, 2.0 * r, subject.head_center - Vec3::Y * (0.08 * r), 0.0),
            PortraitPreset::FullBody => {
                let middle =
                    Vec3::new(subject.head_center.x, f32::midpoint(subject.top, subject.feet), subject.head_center.z);
                (1.12 * figure, 0.4 * figure, middle, 0.0)
            }
        };
        let mut camera = Self { focal_mm: preset.focal_mm(), target, ..Self::default() };
        let half = (0.5 * camera.fov_y()).tan();
        let distance = (0.5 * height / half).max(0.5 * width / (half * aspect.max(1e-3)));
        let a = preset.azimuth();
        camera.position = target + Vec3::new(a.sin(), raise, a.cos()) * distance;
        camera.near = (distance - 6.0 * r).max(distance * 0.05).max(MIN_NEAR);
        camera
    }

    /// A standard review camera framing a subject: `center` and `radius` bound what must fit
    /// vertically (the head for most, the body for [`ReviewCamera::FullBody`], the feature for
    /// [`ReviewCamera::Macro`]); the distance follows from the focal length.
    pub fn review(kind: ReviewCamera, center: Vec3, radius: f32, aspect: f32) -> Self {
        let mut camera = Self { focal_mm: kind.focal_mm(), target: center, ..Self::default() };
        let half_fov = 0.5 * camera.fov_y();
        // Fit the bounding sphere in the narrower of the two fields of view, with 10 % margin.
        let half_fov_x = (half_fov.tan() * aspect.max(1e-3)).atan();
        let fit = half_fov.min(half_fov_x);
        let distance = 1.1 * radius.max(1e-3) / fit.sin();
        let a = kind.azimuth();
        camera.position = center + Vec3::new(a.sin(), 0.0, a.cos()) * distance;
        camera.near = (distance - 2.0 * radius).max(distance * 0.05).max(MIN_NEAR);
        camera
    }
}

/// The matrices of one view for one frame: everything the passes need, computed once on the CPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewMatrices {
    /// World to view.
    pub view: Mat4,
    /// View to clip, without jitter.
    pub projection: Mat4,
    /// World to clip, without jitter (motion vectors, reprojection).
    pub view_projection: Mat4,
    /// World to clip with the TAA jitter (rasterisation).
    pub jittered: Mat4,
    /// Clip to world of the unjittered projection.
    pub inverse_view_projection: Mat4,
    /// The jitter in pixels.
    pub jitter_pixels: Vec2,
    /// Eye position.
    pub eye: Vec3,
}

/// The pixel rectangle of a tile inside a larger image: origin top-left, Y down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileRect {
    /// Left pixel.
    pub x: i32,
    /// Top pixel.
    pub y: i32,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

/// The projection that renders only `tile` of a `full_width` x `full_height` image made with
/// `projection` `[QUAL-04]`: clip-space x and y are scaled and shifted so the tile's pixels map to
/// the whole viewport. Rasterising the tile with it gives the same pixel centres as the full image.
pub fn tile_projection(projection: Mat4, full_width: u32, full_height: u32, tile: TileRect) -> Mat4 {
    let (fw, fh) = (full_width.max(1) as f32, full_height.max(1) as f32);
    let (tw, th) = (tile.width.max(1) as f32, tile.height.max(1) as f32);
    let sx = fw / tw;
    let sy = fh / th;
    // The tile centre in full-image NDC (Y up).
    let cx = (tile.x as f32 + 0.5 * tw) / fw * 2.0 - 1.0;
    let cy = 1.0 - (tile.y as f32 + 0.5 * th) / fh * 2.0;
    let crop =
        Mat4::from_cols_array([sx, 0.0, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -sx * cx, -sy * cy, 0.0, 1.0]);
    crop * projection
}

impl ViewMatrices {
    /// The matrices of `camera` for a `width` x `height` target, shifted by `jitter_pixels`, and
    /// cropped to `tile` of a larger image when one is given.
    pub fn new(
        camera: &Camera,
        width: u32,
        height: u32,
        jitter_pixels: Vec2,
        tile: Option<(u32, u32, TileRect)>,
    ) -> Self {
        let view = camera.view();
        let projection = match tile {
            Some((fw, fh, rect)) => tile_projection(camera.projection(fw as f32 / fh.max(1) as f32), fw, fh, rect),
            None => camera.projection(width as f32 / height.max(1) as f32),
        };
        let viewport = Vec2::new(width as f32, height as f32);
        let jittered_projection = projection.with_jitter(pixel_offset_to_ndc(jitter_pixels, viewport));
        let view_projection = projection * view;
        Self {
            view,
            projection,
            view_projection,
            jittered: jittered_projection * view,
            inverse_view_projection: view_projection.inverse().unwrap_or(Mat4::IDENTITY),
            jitter_pixels,
            eye: camera.position,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cam_01_field_of_view_follows_the_lens() {
        let mut c = Camera { focal_mm: 50.0, ..Camera::default() };
        // A 50 mm lens on a full-frame sensor: 2 atan(12 / 50) = 26.99 degrees.
        assert!((c.fov_y().to_degrees() - 26.99).abs() < 0.01);
        c.focal_mm = 10.0;
        assert!(c.validate().is_err());
        c.focal_mm = 85.0;
        c.near = 0.0;
        assert!(c.validate().is_err());
        assert!(Camera::default().validate().is_ok());
    }

    #[test]
    fn rnd_02_projection_is_reversed_z() {
        let c = Camera::default();
        let m = c.projection(1.0) * c.view();
        let near_point = c.position + c.forward() * c.near;
        let far_point = c.position + c.forward() * 1e6;
        assert!((m.project_point3(near_point).z - 1.0).abs() < 1e-4);
        assert!(m.project_point3(far_point).z < 1e-7);
        let mid = m.project_point3(c.target).z;
        assert!(mid > 0.0 && mid < 1.0);
    }

    #[test]
    fn cam_01_orbit_pan_dolly_keep_their_invariants() {
        let mut c = Camera::default();
        let r = c.position.distance(c.target);
        c.orbit(0.7, 0.3);
        assert!((c.position.distance(c.target) - r).abs() < 1e-5);
        c.orbit(0.0, 10.0);
        assert!(c.position.y - c.target.y < r, "the pitch stops before the pole");
        let before = c.position - c.target;
        c.pan(0.1, -0.2);
        assert!((c.position - c.target - before).length() < 1e-5);
        c.dolly(0.5);
        assert!((c.position.distance(c.target) - 0.5 * r).abs() < 1e-4);
        c.dolly(-1.0);
        assert!(c.position.is_finite());
    }

    #[test]
    fn cam_01_review_cameras_frame_the_subject() {
        for kind in ReviewCamera::ALL {
            let c = Camera::review(kind, Vec3::new(0.0, 1.6, 0.0), 0.15, 1.5);
            c.validate().unwrap();
            let m = c.projection(1.5) * c.view();
            for d in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
                let p = m.project_point3(Vec3::new(0.0, 1.6, 0.0) + d * 0.15);
                assert!(p.x.abs() <= 1.0 && p.y.abs() <= 1.0, "{} cuts off the subject", kind.name());
            }
        }
        let front = Camera::review(ReviewCamera::Front, Vec3::ZERO, 0.1, 1.0);
        assert!(front.position.z > 0.0, "the character faces +Z");
        let left = Camera::review(ReviewCamera::ProfileLeft, Vec3::ZERO, 0.1, 1.0);
        assert!(left.position.x > 0.0, "the character's left is +X");
    }

    /// todo 1.9: each portrait preset shows what its name says on a 1.71 m figure (head radius
    /// 11 cm), in a 4:5 image: the crown inside the top of the frame, the chin above the bottom,
    /// the whole figure for the full body, the camera on the named side.
    #[test]
    fn todo_1_9_portrait_presets_frame_what_they_name() {
        let subject =
            PortraitSubject { head_center: Vec3::new(0.0, 1.6, 0.0), head_radius: 0.11, feet: 0.0, top: 1.71 };
        let aspect = 0.8;
        // Image height fraction from the top (0) to the bottom (1) of a point.
        let row = |c: &Camera, p: Vec3| 0.5 - 0.5 * (c.projection(aspect) * c.view()).project_point3(p).y;
        let crown = Vec3::new(0.0, subject.top, 0.0);
        let chin = Vec3::new(0.0, 1.6 - 0.11, 0.05);
        for preset in PortraitPreset::ALL {
            let c = Camera::portrait(preset, &subject, aspect);
            c.validate().unwrap();
            assert!((c.focal_mm - preset.focal_mm()).abs() < 1e-6);
            let (top, bottom) = (row(&c, crown), row(&c, chin));
            println!("{}: crown at {top:.3}, chin at {bottom:.3} of the height", preset.name());
            assert!(top > 0.0 && bottom < 1.0, "{} cuts off the head", preset.name());
            match preset {
                PortraitPreset::HeadAndShoulders | PortraitPreset::ThreeQuarter | PortraitPreset::Profile => {
                    assert!(top > 0.03 && top < 0.2 && bottom > 0.45 && bottom < 0.65, "{}", preset.name());
                }
                PortraitPreset::TightFace => assert!(top < 0.12 && bottom > 0.8, "the face fills the frame"),
                PortraitPreset::FullBody => {
                    let feet = row(&c, Vec3::ZERO);
                    assert!(top > 0.01 && top < 0.1 && feet > 0.9 && feet < 0.99, "the whole figure: {top} {feet}");
                }
            }
        }
        let side = |p| Camera::portrait(p, &subject, aspect).position;
        assert!(
            side(PortraitPreset::HeadAndShoulders).z > 1.0 && side(PortraitPreset::HeadAndShoulders).x.abs() < 1e-4
        );
        let tq = side(PortraitPreset::ThreeQuarter);
        assert!(tq.x > 0.0 && tq.z > 0.0 && tq.y > 1.5, "three-quarter from the left, a little above");
        assert!(side(PortraitPreset::Profile).x > 1.0, "profile from the left");
        // 85 mm head and shoulders stands about 1.8 m away: the distance of a portrait lens.
        let d = side(PortraitPreset::HeadAndShoulders).distance(subject.head_center);
        assert!(d > 1.4 && d < 2.4, "{d}");
    }

    /// todo 1.9: the thin-lens circle of confusion: 85 mm at f/2 focused at 1.5 m blurs infinity
    /// to f^2 / (N (s - f)) = 2.55 mm on the sensor; the plane in focus is sharp; the f-number
    /// and focus are checked.
    #[test]
    fn todo_1_9_lens_confusion_is_the_thin_lens_one() {
        let camera = Camera { focal_mm: 85.0, ..Camera::default() };
        let lens = Lens::new(2.0, 1.5);
        lens.validate().unwrap();
        let expected = 0.5 * 85.0 * 85.0 / (2.0 * (1500.0 - 85.0)) / 24.0 * 1000.0;
        assert!((lens.confusion_at_infinity(&camera, 1000.0) - expected).abs() < 1e-3);
        assert!(lens.confusion(&camera, 1000.0, 1.5) < 1e-4);
        let behind = lens.confusion(&camera, 1000.0, 1.6);
        let front = lens.confusion(&camera, 1000.0, 1.4);
        assert!(behind > 0.0 && front > behind, "nearer is blurred more for the same 10 cm");
        assert!(Lens::new(4.0, 1.5).confusion(&camera, 1000.0, 1.6) < 0.51 * behind, "stopping down halves it");
        for bad in [Lens::new(0.5, 1.0), Lens::new(2.0, 0.001), Lens::new(f32::NAN, 1.0), Lens::new(2.0, f32::INFINITY)]
        {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn lit_03_exposure_matches_the_physical_camera() {
        let e = Exposure::from_physical(&PhysicalCamera::SUNNY_16);
        assert!((e.ev100 - 14.966).abs() < 0.01);
        let s = Exposure::ev100(0.0).scale();
        assert!((s - 1.0 / 1.2).abs() < 1e-6);
        let brighter = Exposure { ev100: 0.0, compensation: 1.0 }.scale();
        assert!((brighter / s - 2.0).abs() < 1e-5);
        assert!(Exposure::ev100(1e9).scale() > 0.0, "clamped to the documented range");
    }

    /// `[QUAL-04]` a tile's projection maps the tile's pixels onto the whole viewport, so each
    /// world point lands on the same full-image pixel.
    #[test]
    fn qual_04_tile_projection_matches_the_full_image() {
        let c = Camera::default();
        let (fw, fh) = (1000_u32, 600_u32);
        let full = c.projection(fw as f32 / fh as f32) * c.view();
        let tile = TileRect { x: 300, y: 100, width: 250, height: 200 };
        let tiled = tile_projection(c.projection(fw as f32 / fh as f32), fw, fh, tile) * c.view();
        for p in [c.target, c.target + Vec3::new(0.02, 0.05, 0.0), c.target + Vec3::new(-0.1, 0.03, 0.05)] {
            let a = full.project_point3(p);
            let b = tiled.project_point3(p);
            let full_px = humanart_math::ndc_to_screen(a.xy(), Vec2::new(fw as f32, fh as f32));
            let tile_px = humanart_math::ndc_to_screen(b.xy(), Vec2::new(tile.width as f32, tile.height as f32));
            let shifted = tile_px + Vec2::new(tile.x as f32, tile.y as f32);
            assert!((full_px - shifted).length() < 1e-3, "{full_px:?} vs {shifted:?}");
            assert!((a.z - b.z).abs() < 1e-7, "depth is unchanged");
        }
    }

    #[test]
    fn taa_jitter_shifts_by_the_requested_pixels() {
        let c = Camera::default();
        let a = ViewMatrices::new(&c, 200, 100, Vec2::ZERO, None);
        let b = ViewMatrices::new(&c, 200, 100, Vec2::new(0.5, -0.25), None);
        let pa = a.jittered.project_point3(c.target);
        let pb = b.jittered.project_point3(c.target);
        let dx = (pb.x - pa.x) * 100.0;
        let dy = (pb.y - pa.y) * 50.0;
        assert!((dx - 0.5).abs() < 1e-4 && (dy - 0.25).abs() < 1e-4, "{dx} {dy}");
        assert_eq!(a.view_projection, b.view_projection, "the unjittered matrices stay");
    }
}
