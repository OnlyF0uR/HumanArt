//! Lights and lighting rigs `[LIT-01]` `[LIT-02]` `[LIT-04]`.
//!
//! Units are photometric-like: a directional light gives an illuminance in lux on a surface facing
//! it, a spot light has a luminous intensity in candela, a rect-area light a luminance in nits; the
//! environment's relative radiance is scaled by `environment_nits`. The colour is a colour
//! temperature (a black body through `humanart_math::color::temperature`) or a linear Rec.709 tint,
//! normalised to a luminance of 1 so it never changes the intensity.
//!
//! Rect-area shading: the diffuse term is the exact irradiance of a polygonal Lambertian emitter
//! (Lambert's edge-integral formula, clamped at the horizon); the specular term uses the
//! representative-point approximation (Karis 2013, "Real Shading in Unreal Engine 4"). The
//! LTC-based rect shading `[LIT-01]` names is not implemented yet: it needs a fitted LTC table.

use humanart_math::color::temperature::blackbody_rgb;
use humanart_math::{Mat4, Sphere, Vec3};

use crate::error::RenderError;

/// Most lights in one rig (the size of the GPU light array).
pub const MAX_LIGHTS: usize = 8;
/// Most lights with a shadow map (the layers of the shadow atlas).
pub const MAX_SHADOWED: usize = 4;

/// The kind and shape of a light.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightKind {
    /// Parallel light; `intensity` is an illuminance in lux.
    Directional {
        /// Angular diameter of the source, radians (the sun is 0.0093): the penumbra of soft shadows.
        angular_diameter: f32,
    },
    /// A cone from a point; `intensity` is a luminous intensity in candela.
    Spot {
        /// Half-angle where the falloff starts, radians.
        inner: f32,
        /// Half-angle where the light ends, radians (< 90 degrees).
        outer: f32,
        /// Distance where the light fades to zero, metres.
        range: f32,
        /// Radius of the emitter, metres: the penumbra of soft shadows.
        radius: f32,
    },
    /// A one-sided rectangle; `intensity` is a luminance in nits.
    Rect {
        /// Width along the light's right axis, metres.
        width: f32,
        /// Height along its up axis, metres.
        height: f32,
    },
}

impl LightKind {
    /// The kind code the shaders use.
    pub const fn code(&self) -> u32 {
        match self {
            Self::Directional { .. } => 0,
            Self::Spot { .. } => 1,
            Self::Rect { .. } => 2,
        }
    }
}

/// The colour of a light.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightColor {
    /// A black body at this temperature, kelvin.
    Temperature(f32),
    /// A linear Rec.709 tint.
    Linear([f32; 3]),
}

impl LightColor {
    /// The linear Rec.709 colour at a luminance of 1.
    pub fn rgb(&self) -> [f32; 3] {
        match *self {
            Self::Temperature(k) => blackbody_rgb(f64::from(k)).map(|c| c as f32),
            Self::Linear(c) => {
                let y = 0.212_639 * c[0] + 0.715_169 * c[1] + 0.072_192 * c[2];
                if y > 0.0 && y.is_finite() { c.map(|v| v.max(0.0) / y) } else { [1.0; 3] }
            }
        }
    }
}

/// Shadow settings of a light.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowSettings {
    /// Whether the light casts shadows (at most [`MAX_SHADOWED`] per rig get a map).
    pub enabled: bool,
    /// Soft shadows (PCSS-style penumbra from the source size) instead of a fixed small filter.
    pub soft: bool,
    /// Normal-offset bias in shadow-map texels.
    pub normal_bias: f32,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self { enabled: true, soft: true, normal_bias: 1.5 }
    }
}

/// One light.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    /// Kind and shape.
    pub kind: LightKind,
    /// Position (spot, rect), metres; ignored for directional lights.
    pub position: Vec3,
    /// The direction the light travels (out of the emitter).
    pub direction: Vec3,
    /// Up axis of a rect light (orthogonalised against `direction`).
    pub up: Vec3,
    /// Colour.
    pub color: LightColor,
    /// Lux, candela or nits, see [`LightKind`].
    pub intensity: f32,
    /// Shadows.
    pub shadow: ShadowSettings,
}

impl Light {
    /// A directional light travelling along `direction` with `lux`.
    pub fn directional(direction: Vec3, lux: f32, kelvin: f32) -> Self {
        Self {
            kind: LightKind::Directional { angular_diameter: 0.02 },
            position: Vec3::ZERO,
            direction: direction.normalize_or(-Vec3::Y),
            up: Vec3::Y,
            color: LightColor::Temperature(kelvin),
            intensity: lux,
            shadow: ShadowSettings::default(),
        }
    }

    /// A rect softbox at `position` facing `target`, `width` x `height` metres, `nits` luminance.
    pub fn softbox(position: Vec3, target: Vec3, width: f32, height: f32, nits: f32, kelvin: f32) -> Self {
        Self {
            kind: LightKind::Rect { width, height },
            position,
            direction: (target - position).normalize_or(-Vec3::Z),
            up: Vec3::Y,
            color: LightColor::Temperature(kelvin),
            intensity: nits,
            shadow: ShadowSettings::default(),
        }
    }

    /// A spot at `position` aimed at `target`.
    pub fn spot(position: Vec3, target: Vec3, candela: f32, outer: f32, kelvin: f32) -> Self {
        Self {
            kind: LightKind::Spot { inner: outer * 0.8, outer, range: 20.0, radius: 0.05 },
            position,
            direction: (target - position).normalize_or(-Vec3::Y),
            up: Vec3::Y,
            color: LightColor::Temperature(kelvin),
            intensity: candela,
            shadow: ShadowSettings::default(),
        }
    }

    /// Checks the parameters.
    pub fn validate(&self) -> Result<(), RenderError> {
        let ok = self.position.is_finite()
            && self.direction.is_finite()
            && self.direction.length() > 0.0
            && self.intensity.is_finite()
            && self.intensity >= 0.0;
        if !ok {
            return Err(RenderError::InvalidParameter {
                what: "light position, direction and intensity must be finite",
            });
        }
        match self.kind {
            LightKind::Directional { angular_diameter } if !(0.0..1.0).contains(&angular_diameter) => {
                Err(RenderError::InvalidParameter { what: "directional angular diameter outside [0, 1) rad" })
            }
            LightKind::Spot { inner, outer, range, radius }
                if !(inner >= 0.0 && inner <= outer && outer < 1.55 && range > 0.0 && radius >= 0.0) =>
            {
                Err(RenderError::InvalidParameter {
                    what: "spot needs 0 <= inner <= outer < 89 degrees, a positive range",
                })
            }
            LightKind::Rect { width, height }
                if !(width > 0.0 && height > 0.0 && width.is_finite() && height.is_finite()) =>
            {
                Err(RenderError::InvalidParameter { what: "rect light needs a positive size" })
            }
            _ => Ok(()),
        }
    }

    /// The light's right and up axes (rect lights).
    pub fn axes(&self) -> (Vec3, Vec3) {
        let d = self.direction.normalize_or(-Vec3::Z);
        let right = d.cross(self.up).try_normalize().unwrap_or_else(|| d.any_orthonormal_vector());
        (right, right.cross(d))
    }

    /// The illuminance the light gives a surface at `point` facing it, lux, ignoring shadows: the
    /// CPU reference the shading tests compare against (rect lights: the centre-facing
    /// approximation `L · A · cos / d²`).
    pub fn illuminance_at(&self, point: Vec3) -> f32 {
        match self.kind {
            LightKind::Directional { .. } => self.intensity,
            LightKind::Spot { inner, outer, range, .. } => {
                let to_point = point - self.position;
                let d2 = to_point.length_squared().max(1e-8);
                let cos = to_point.normalize_or(Vec3::Y).dot(self.direction.normalize_or(-Vec3::Y));
                let (ci, co) = (inner.cos(), outer.cos());
                let cone = ((cos - co) / (ci - co).max(1e-4)).clamp(0.0, 1.0);
                let window = (1.0 - (d2 / (range * range)).powi(2)).clamp(0.0, 1.0).powi(2);
                self.intensity * cone * cone * window / d2
            }
            LightKind::Rect { width, height } => {
                let to_point = point - self.position;
                let d2 = to_point.length_squared().max(1e-8);
                let cos = to_point.normalize_or(Vec3::Y).dot(self.direction.normalize_or(-Vec3::Z)).max(0.0);
                self.intensity * width * height * cos / d2
            }
        }
    }
}

humanart_gpu::pod_struct! {
    /// One light as the shaders read it (`struct Light` in `shaders/render/common.wgsl`).
    #[derive(Debug, PartialEq)]
    pub struct GpuLight(size = 96, align = 4) {
        /// Linear colour times intensity; w = kind code.
        pub color_kind: [f32; 4],
        /// Position; w = range (spot) or 0.
        pub position_range: [f32; 4],
        /// Unit direction the light travels; w = cos of the outer angle (spot).
        pub direction_cos_outer: [f32; 4],
        /// Right axis times half width (rect); w = cos of the inner angle (spot).
        pub right_cos_inner: [f32; 4],
        /// Up axis times half height (rect); w = source size (angular diameter or radius).
        pub up_size: [f32; 4],
        /// x = shadow layer or -1, y = soft (1/0), z = normal bias in texels, w = unused.
        pub shadow: [f32; 4],
    }
}

impl GpuLight {
    /// An unused slot.
    pub const NONE: Self = Self {
        color_kind: [0.0; 4],
        position_range: [0.0; 4],
        direction_cos_outer: [0.0; 4],
        right_cos_inner: [0.0; 4],
        up_size: [0.0; 4],
        shadow: [-1.0, 0.0, 0.0, 0.0],
    };

    /// Packs `light`, with `shadow_layer` its layer in the shadow atlas.
    pub fn new(light: &Light, shadow_layer: Option<usize>) -> Self {
        let c = light.color.rgb();
        let d = light.direction.normalize_or(-Vec3::Y);
        let (right, up) = light.axes();
        let (range, cos_outer, cos_inner, half_w, half_h, size) = match light.kind {
            LightKind::Directional { angular_diameter } => (0.0, 0.0, 0.0, 0.0, 0.0, angular_diameter),
            LightKind::Spot { inner, outer, range, radius } => (range, outer.cos(), inner.cos(), 0.0, 0.0, radius),
            LightKind::Rect { width, height } => (0.0, 0.0, 0.0, 0.5 * width, 0.5 * height, 0.0),
        };
        let i = light.intensity;
        Self {
            color_kind: [c[0] * i, c[1] * i, c[2] * i, light.kind.code() as f32],
            position_range: [light.position.x, light.position.y, light.position.z, range],
            direction_cos_outer: [d.x, d.y, d.z, cos_outer],
            right_cos_inner: [right.x * half_w, right.y * half_w, right.z * half_w, cos_inner],
            up_size: [up.x * half_h, up.y * half_h, up.z * half_h, size],
            shadow: [
                shadow_layer.map_or(-1.0, |l| l as f32),
                if light.shadow.soft { 1.0 } else { 0.0 },
                light.shadow.normal_bias,
                0.0,
            ],
        }
    }
}

/// The procedural environments generated in code (there are no downloaded HDRIs in the project;
/// none of these is a measurement of a real place).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProceduralEnvironment {
    /// A dark studio: grey gradient walls, a floor, and three soft rectangular softbox panels.
    Studio,
    /// A uniform overcast sky over a grey ground.
    Overcast,
    /// A clear sky gradient with a sun disk in `sun_direction` (pointing to the sun).
    ClearSky {
        /// Unit vector towards the sun.
        sun_direction: Vec3,
    },
    /// The same radiance (1) in every direction: the white furnace of the IBL tests.
    Uniform,
    /// A sun close to the horizon: an orange band along the horizon that is brightest around the
    /// sun, a sky that turns blue towards the zenith, a dark ground.
    Sunset {
        /// Unit vector towards the sun.
        sun_direction: Vec3,
    },
    /// The dark studio of [`ProceduralEnvironment::Studio`] without its panels: the rigs whose
    /// softboxes are rect lights use it, so a catchlight in the eye is the rig's own source and
    /// not a panel painted somewhere else.
    StudioRoom,
}

impl ProceduralEnvironment {
    /// The kind code of `shaders/render/ibl.wgsl`.
    pub const fn code(&self) -> u32 {
        match self {
            Self::Studio => 0,
            Self::Overcast => 1,
            Self::ClearSky { .. } => 2,
            Self::Uniform => 3,
            Self::Sunset { .. } => 4,
            Self::StudioRoom => 5,
        }
    }
}

/// What is behind the character `[LIT-04]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Background {
    /// A flat grey, given as the display-linear value after exposure (0.18 = middle grey), so it
    /// looks the same at every exposure. The review default.
    Gray(f32),
    /// The environment itself.
    Environment,
    /// Seamless studio paper behind and under the subject, lit by the rig (todo 1.9).
    Backdrop(Backdrop),
}

/// Which backdrop a viewer shows behind the character.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BackdropChoice {
    /// What the rig says (a flat grey for the studio rigs: the tests' backdrop).
    #[default]
    Rig,
    /// Seamless studio paper where the rig has a flat grey ([`LightRig::with_studio_backdrop`]):
    /// the review default of the command-line tool.
    Studio,
    /// A flat middle grey, whatever the rig says.
    Flat,
    /// The rig's environment.
    Environment,
}

/// A seamless paper backdrop: a wall behind the subject (as the camera sees it) that curves into
/// the floor under the subject's feet. It is shaded as a matte surface by the rig's lights and
/// environment, receives the subject's shadow, and darkens softly away from the spot behind the
/// subject.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Backdrop {
    /// The paper's reflectance, linear Rec.709.
    pub color: [f32; 3],
    /// How far behind the subject's centre the wall stands, metres.
    pub distance: f32,
    /// How much darker the paper is far from the spot behind the subject, `[0, 1]` (0: even).
    pub falloff: f32,
    /// The distance from that spot at which the falloff is complete, metres.
    pub falloff_radius: f32,
}

impl Backdrop {
    /// Mid-grey paper 2.5 m behind the subject with a gentle falloff: the review default.
    pub const GRAY: Self = Self { color: [0.45, 0.45, 0.46], distance: 2.5, falloff: 0.5, falloff_radius: 2.4 };
    /// Near-black paper for rim and low-key rigs.
    pub const BLACK: Self = Self { color: [0.02, 0.02, 0.022], distance: 3.0, falloff: 0.5, falloff_radius: 2.0 };

    /// Checks the parameters.
    pub fn validate(&self) -> Result<(), RenderError> {
        let ok = self.color.iter().all(|c| (0.0..=1.0).contains(c))
            && (0.2..=50.0).contains(&self.distance)
            && (0.0..=1.0).contains(&self.falloff)
            && (0.1..=100.0).contains(&self.falloff_radius);
        if ok {
            Ok(())
        } else {
            Err(RenderError::InvalidParameter {
                what: "backdrop: colour and falloff in [0, 1], distance 0.2-50 m, falloff radius 0.1-100 m",
            })
        }
    }
}

/// A lighting rig `[LIT-02]`: lights, the environment light and the backdrop.
#[derive(Debug, Clone, PartialEq)]
pub struct LightRig {
    /// Stable name.
    pub name: &'static str,
    /// The lights, at most [`MAX_LIGHTS`].
    pub lights: Vec<Light>,
    /// The environment the IBL is made from.
    pub environment: ProceduralEnvironment,
    /// Scale of the environment's relative radiance, nits.
    pub environment_nits: f32,
    /// Rotation of the environment about +Y, radians.
    pub environment_rotation: f32,
    /// The backdrop.
    pub background: Background,
    /// The exposure the rig is balanced for.
    pub ev100: f32,
}

/// The built-in rigs `[LIT-02]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BuiltinRig {
    /// Key, fill and rim (3-point) in a studio.
    NeutralStudio,
    /// Clamshell beauty light: soft key above the camera, fill from below.
    FlatBeauty,
    /// A hard light from the side.
    HarshSidelight,
    /// A light from behind (ear translucency test).
    Backlight,
    /// Overcast sky only.
    OvercastOutdoor,
    /// Low warm sun with sky.
    GoldenHour,
    /// The starting point of a manually matched rig: one neutral key, to be edited.
    ReferenceMatched,
    /// Three-point portrait light from soft sources with a visible size: a softbox key 40 degrees
    /// to the side, a larger and weaker fill by the camera, a strip light behind as the rim.
    ThreePoint,
    /// Rembrandt: a small softbox high and 50 degrees to the character's right, almost no fill: the
    /// left cheek keeps only a triangle of light. The three-quarter camera (from the left) looks
    /// into that shadow side (short light).
    Rembrandt,
    /// Loop: the key 30 degrees to the side and a little above, a 4:1 fill, a hair light.
    Loop,
    /// Clamshell: a wide softbox above the lens and a reflector below it, two catchlights.
    Clamshell,
    /// Two strip lights behind the subject and nothing from the front: the outline only.
    RimOnly,
    /// A low orange sun in a sunset sky.
    Sunset,
}

impl BuiltinRig {
    /// Every built-in rig.
    pub const ALL: [Self; 13] = [
        Self::NeutralStudio,
        Self::FlatBeauty,
        Self::HarshSidelight,
        Self::Backlight,
        Self::OvercastOutdoor,
        Self::GoldenHour,
        Self::ReferenceMatched,
        Self::ThreePoint,
        Self::Rembrandt,
        Self::Loop,
        Self::Clamshell,
        Self::RimOnly,
        Self::Sunset,
    ];

    /// The stable name (the `name` of the rig [`LightRig::builtin`] makes).
    pub const fn name(self) -> &'static str {
        match self {
            Self::NeutralStudio => "neutral_studio",
            Self::FlatBeauty => "flat_beauty",
            Self::HarshSidelight => "harsh_sidelight",
            Self::Backlight => "backlight",
            Self::OvercastOutdoor => "overcast_outdoor",
            Self::GoldenHour => "golden_hour",
            Self::ReferenceMatched => "reference_matched",
            Self::ThreePoint => "three_point",
            Self::Rembrandt => "rembrandt",
            Self::Loop => "loop",
            Self::Clamshell => "clamshell",
            Self::RimOnly => "rim_only",
            Self::Sunset => "sunset",
        }
    }
}

impl LightRig {
    /// The built-in rig `kind` placed around a subject at `center` (the head) with radius `radius`;
    /// the character faces +Z, its left is +X.
    pub fn builtin(kind: BuiltinRig, center: Vec3, radius: f32) -> Self {
        let at = |x: f32, y: f32, z: f32| center + Vec3::new(x, y, z) * (radius * 8.0);
        let size = radius * 6.0;
        // Softbox luminance giving `lux` at the subject: E = L A / d^2 with d = 8 r * |dir|.
        let nits_for = |lux: f32, p: Vec3, w: f32, h: f32| lux * p.distance_squared(center) / (w * h);
        let base = |name, lights, environment, environment_nits, background, ev100| Self {
            name,
            lights,
            environment,
            environment_nits,
            environment_rotation: 0.0,
            background,
            ev100,
        };
        let gray = Background::Gray(0.18);
        match kind {
            BuiltinRig::NeutralStudio => {
                let key = at(0.7, 0.5, 0.8);
                let fill = at(-0.9, 0.0, 0.6);
                let mut rim = Light::directional(Vec3::new(0.3, -0.4, 1.0), 900.0, 6500.0);
                rim.shadow.enabled = false;
                let mut fill_light =
                    Light::softbox(fill, center, size, size, nits_for(450.0, fill, size, size), 5600.0);
                fill_light.shadow.enabled = false;
                base(
                    "neutral_studio",
                    vec![
                        Light::softbox(key, center, size, size, nits_for(1800.0, key, size, size), 5600.0),
                        fill_light,
                        rim,
                    ],
                    ProceduralEnvironment::Studio,
                    60.0,
                    gray,
                    9.0,
                )
            }
            BuiltinRig::FlatBeauty => {
                let key = at(0.0, 0.6, 1.0);
                let fill = at(0.0, -0.6, 1.0);
                let mut fill_light = Light::softbox(
                    fill,
                    center,
                    size * 1.5,
                    size * 0.5,
                    nits_for(700.0, fill, size * 1.5, size * 0.5),
                    5600.0,
                );
                fill_light.shadow.enabled = false;
                base(
                    "flat_beauty",
                    vec![
                        Light::softbox(key, center, size * 1.5, size, nits_for(1600.0, key, size * 1.5, size), 5600.0),
                        fill_light,
                    ],
                    ProceduralEnvironment::Studio,
                    80.0,
                    gray,
                    9.0,
                )
            }
            BuiltinRig::HarshSidelight => {
                let mut side = Light::directional(Vec3::new(-1.0, -0.1, -0.05), 2500.0, 5000.0);
                side.kind = LightKind::Directional { angular_diameter: 0.005 };
                side.shadow.soft = false;
                base("harsh_sidelight", vec![side], ProceduralEnvironment::Studio, 15.0, gray, 9.0)
            }
            BuiltinRig::Backlight => {
                let back = Light::directional(Vec3::new(0.0, -0.25, 1.0), 3000.0, 5600.0);
                let front = at(0.0, 0.2, 1.0);
                let mut fill = Light::softbox(front, center, size, size, nits_for(250.0, front, size, size), 5600.0);
                fill.shadow.enabled = false;
                base("backlight", vec![back, fill], ProceduralEnvironment::Studio, 20.0, gray, 9.0)
            }
            BuiltinRig::OvercastOutdoor => base(
                "overcast_outdoor",
                Vec::new(),
                ProceduralEnvironment::Overcast,
                2500.0,
                Background::Environment,
                12.0,
            ),
            BuiltinRig::GoldenHour => {
                let sun_direction = Vec3::new(0.8, 0.18, 0.55).normalize_or(Vec3::Y);
                let mut sun = Light::directional(-sun_direction, 12_000.0, 3200.0);
                sun.kind = LightKind::Directional { angular_diameter: 0.0093 };
                base(
                    "golden_hour",
                    vec![sun],
                    ProceduralEnvironment::ClearSky { sun_direction },
                    1500.0,
                    Background::Environment,
                    12.0,
                )
            }
            BuiltinRig::ReferenceMatched => {
                let key = at(0.5, 0.4, 0.9);
                base(
                    "reference_matched",
                    vec![Light::softbox(key, center, size, size, nits_for(1800.0, key, size, size), 5600.0)],
                    ProceduralEnvironment::Studio,
                    40.0,
                    gray,
                    9.0,
                )
            }
            BuiltinRig::ThreePoint => {
                let key = at(0.62, 0.42, 0.74);
                let fill = at(-0.55, 0.05, 0.95);
                let rim = at(-0.6, 0.75, -0.75);
                let (fw, rw, rh) = (size * 1.6, size * 0.4, size * 1.6);
                let mut fill_light = Light::softbox(fill, center, fw, fw, nits_for(500.0, fill, fw, fw), 5600.0);
                fill_light.shadow.enabled = false;
                base(
                    "three_point",
                    vec![
                        Light::softbox(key, center, size, size * 1.3, nits_for(1900.0, key, size, size * 1.3), 5600.0),
                        fill_light,
                        Light::softbox(rim, center, rw, rh, nits_for(1500.0, rim, rw, rh), 6000.0),
                    ],
                    ProceduralEnvironment::StudioRoom,
                    45.0,
                    gray,
                    9.0,
                )
            }
            BuiltinRig::Rembrandt => {
                let key = at(-0.74, 0.62, 0.52);
                let fill = at(0.8, 0.0, 0.7);
                let small = size * 0.6;
                let (fw, fh) = (size * 1.5, size * 2.0);
                let mut fill_light = Light::softbox(fill, center, fw, fh, nits_for(160.0, fill, fw, fh), 5600.0);
                fill_light.shadow.enabled = false;
                base(
                    "rembrandt",
                    vec![
                        Light::softbox(key, center, small, small, nits_for(2400.0, key, small, small), 5200.0),
                        fill_light,
                    ],
                    ProceduralEnvironment::StudioRoom,
                    18.0,
                    gray,
                    9.0,
                )
            }
            BuiltinRig::Loop => {
                let key = at(0.48, 0.42, 0.86);
                let fill = at(-0.7, 0.1, 0.8);
                let hair = at(-0.35, 0.95, -0.55);
                let (fw, hw) = (size * 1.4, size * 0.5);
                let mut fill_light = Light::softbox(fill, center, fw, fw, nits_for(480.0, fill, fw, fw), 5600.0);
                fill_light.shadow.enabled = false;
                base(
                    "loop",
                    vec![
                        Light::softbox(key, center, size, size, nits_for(1900.0, key, size, size), 5600.0),
                        fill_light,
                        Light::softbox(hair, center, hw, hw, nits_for(900.0, hair, hw, hw), 6000.0),
                    ],
                    ProceduralEnvironment::StudioRoom,
                    40.0,
                    gray,
                    9.0,
                )
            }
            BuiltinRig::Clamshell => {
                let key = at(0.0, 0.55, 0.95);
                let fill = at(0.0, -0.6, 0.85);
                let (kw, kh) = (size * 1.6, size);
                let (fw, fh) = (size * 1.5, size * 0.6);
                let mut fill_light = Light::softbox(fill, center, fw, fh, nits_for(850.0, fill, fw, fh), 5600.0);
                fill_light.shadow.enabled = false;
                base(
                    "clamshell",
                    vec![Light::softbox(key, center, kw, kh, nits_for(1700.0, key, kw, kh), 5600.0), fill_light],
                    ProceduralEnvironment::StudioRoom,
                    45.0,
                    gray,
                    9.0,
                )
            }
            BuiltinRig::RimOnly => {
                let left = at(0.85, 0.25, -0.7);
                let right = at(-0.85, 0.25, -0.7);
                let (w, h) = (size * 0.4, size * 1.8);
                base(
                    "rim_only",
                    vec![
                        Light::softbox(left, center, w, h, nits_for(2600.0, left, w, h), 5600.0),
                        Light::softbox(right, center, w, h, nits_for(2200.0, right, w, h), 5600.0),
                    ],
                    ProceduralEnvironment::StudioRoom,
                    6.0,
                    Background::Gray(0.01),
                    9.0,
                )
            }
            BuiltinRig::Sunset => {
                let sun_direction = Vec3::new(0.72, 0.1, 0.68).normalize_or(Vec3::Y);
                let mut sun = Light::directional(-sun_direction, 5500.0, 2600.0);
                sun.kind = LightKind::Directional { angular_diameter: 0.0093 };
                base(
                    "sunset",
                    vec![sun],
                    ProceduralEnvironment::Sunset { sun_direction },
                    900.0,
                    Background::Environment,
                    11.0,
                )
            }
        }
    }

    /// The rig with the backdrop a viewer chose.
    #[must_use]
    pub fn with_backdrop(mut self, choice: BackdropChoice) -> Self {
        match choice {
            BackdropChoice::Rig => self,
            BackdropChoice::Studio => self.with_studio_backdrop(),
            BackdropChoice::Flat => {
                self.background = Background::Gray(0.18);
                self
            }
            BackdropChoice::Environment => {
                self.background = Background::Environment;
                self
            }
        }
    }

    /// The rig with seamless paper in place of a flat backdrop (todo 1.9: review renders): a
    /// [`Background::Gray`] becomes [`Backdrop::GRAY`], or [`Backdrop::BLACK`] when the grey was
    /// darker than 0.05; an environment backdrop and a paper one stay.
    #[must_use]
    pub fn with_studio_backdrop(mut self) -> Self {
        if let Background::Gray(g) = self.background {
            self.background = Background::Backdrop(if g < 0.05 { Backdrop::BLACK } else { Backdrop::GRAY });
        }
        self
    }

    /// Checks every light and the counts.
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.lights.len() > MAX_LIGHTS {
            return Err(RenderError::TooLarge {
                what: "lights in a rig",
                value: self.lights.len() as u64,
                limit: MAX_LIGHTS as u64,
            });
        }
        if !(self.environment_nits.is_finite() && self.environment_nits >= 0.0 && self.ev100.is_finite()) {
            return Err(RenderError::InvalidParameter { what: "environment intensity and EV100 must be finite" });
        }
        if let Background::Backdrop(b) = &self.background {
            b.validate()?;
        }
        self.lights.iter().try_for_each(Light::validate)
    }

    /// The indices of the lights that get a shadow map, at most [`MAX_SHADOWED`], in rig order.
    pub fn shadowed(&self) -> impl Iterator<Item = usize> + '_ {
        self.lights.iter().enumerate().filter(|(_, l)| l.shadow.enabled).map(|(i, _)| i).take(MAX_SHADOWED)
    }
}

/// A light's shadow map projection, fitted tightly to the bounding sphere (`center`, `radius`) of
/// the shadow casters `[RND-03]` item 4. Reversed-Z, like the camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowFrustum {
    /// World to shadow clip.
    pub view_projection: Mat4,
    /// Perspective (spot, rect) rather than orthographic (directional).
    pub perspective: bool,
    /// Near distance, metres.
    pub near: f32,
    /// Far distance, metres.
    pub far: f32,
    /// Orthographic: half the side of the square, metres; perspective: tan of the half angle.
    pub half_extent: f32,
    /// Orthographic: penumbra width per metre between blocker and receiver (2 tan of half the
    /// angular diameter); perspective: the emitter's size, metres.
    pub penumbra: f32,
}

impl ShadowFrustum {
    /// The frustum of `light` around the sphere (`center`, `radius`). Rect lights project from
    /// the panel's centre.
    pub fn fit(light: &Light, center: Vec3, radius: f32) -> Self {
        let radius = radius.max(1e-3);
        Self::fit_focused(light, Sphere { center, radius }, Sphere { center, radius })
    }

    /// The frustum of `light` whose map covers only the sphere `focus` (what the camera sees,
    /// [`crate::focus`]) while its depth range still holds every caster of the sphere `casters`:
    /// a caster outside the view shadows what is inside it. With `focus` equal to `casters` this
    /// is [`ShadowFrustum::fit`].
    pub fn fit_focused(light: &Light, casters: Sphere, focus: Sphere) -> Self {
        let caster_radius = casters.radius.max(1e-3);
        let radius = focus.radius.max(1e-3);
        let center = focus.center;
        let d = light.direction.normalize_or(-Vec3::Y);
        let up_for = |dir: Vec3| if dir.normalize_or(Vec3::Y).y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
        match light.kind {
            LightKind::Directional { angular_diameter } => {
                // Along the light, the casters span `along - R ..= along + R` from the focus centre.
                let along = (casters.center - center).dot(d);
                let eye = center + d * (along - caster_radius * 2.0);
                let view = Mat4::look_to_rh(eye, d, up_for(d));
                // The far plane reaches well past the casters so receivers behind them (the floor) keep
                // a real depth for the penumbra estimate.
                let (near, far) = (caster_radius * 0.5, caster_radius * 12.0);
                Self {
                    view_projection: Mat4::orthographic_reversed_z(-radius, radius, -radius, radius, near, far) * view,
                    perspective: false,
                    near,
                    far,
                    half_extent: radius,
                    penumbra: 2.0 * (0.5 * angular_diameter).tan(),
                }
            }
            LightKind::Spot { .. } | LightKind::Rect { .. } => {
                let eye = light.position;
                let to_center = center - eye;
                let distance = to_center.length().max(radius * 1.01);
                let view = Mat4::look_to_rh(eye, to_center, up_for(to_center));
                // The cone that just contains the sphere, with 5 % margin.
                let half = (radius / distance).clamp(1e-3, 0.999).asin() * 1.05;
                let far = distance + radius;
                // The nearest a caster can be to the light, never behind the focus sphere's front.
                let near = (casters.center.distance(eye) - caster_radius).min(distance - radius).max(far * 0.02);
                let penumbra = match light.kind {
                    LightKind::Spot { radius, .. } => 2.0 * radius,
                    LightKind::Rect { width, height } => width.max(height),
                    LightKind::Directional { .. } => 0.0,
                };
                Self {
                    view_projection: Mat4::perspective_reversed_z(2.0 * half, 1.0, near, far) * view,
                    perspective: true,
                    near,
                    far,
                    half_extent: half.tan(),
                    penumbra,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lit_01_colour_temperature_is_normalised() {
        for k in [2700.0, 5600.0, 6500.0, 10_000.0] {
            let c = LightColor::Temperature(k).rgb();
            let y = 0.212_639 * c[0] + 0.715_169 * c[1] + 0.072_192 * c[2];
            assert!((y - 1.0).abs() < 1e-3, "{k} K has luminance {y}");
        }
        let warm = LightColor::Temperature(3000.0).rgb();
        assert!(warm[0] > warm[2], "3000 K is warm");
        assert_eq!(LightColor::Linear([0.0; 3]).rgb(), [1.0; 3]);
    }

    #[test]
    fn lit_02_builtin_rigs_are_valid_and_balanced() {
        let center = Vec3::new(0.0, 1.6, 0.0);
        for kind in BuiltinRig::ALL {
            let rig = LightRig::builtin(kind, center, 0.12);
            rig.validate().unwrap();
            assert!(rig.shadowed().count() <= MAX_SHADOWED);
            let lux: f32 = rig.lights.iter().map(|l| l.illuminance_at(center)).sum();
            assert!(lux.is_finite());
        }
        let studio = LightRig::builtin(BuiltinRig::NeutralStudio, center, 0.12);
        let key = studio.lights[0].illuminance_at(center);
        assert!((key - 1800.0).abs() / 1800.0 < 0.01, "the key gives its design illuminance: {key}");
        assert_eq!(studio.background, Background::Gray(0.18), "[LIT-04] grey backdrop by default");
    }

    /// todo 1.9: the portrait rigs put their key where their name says (the character faces +Z,
    /// its left is +X), with soft sources of a visible size; the viewer's backdrop choice swaps
    /// only a flat grey for paper and keeps the rest of the rig.
    #[test]
    fn todo_1_9_portrait_rigs_and_backdrop_choices() {
        let center = Vec3::new(0.0, 1.6, 0.0);
        let rig = |kind| LightRig::builtin(kind, center, 0.12);
        let key = |kind| {
            let r = rig(kind);
            let l = r.lights[0];
            assert!(matches!(l.kind, LightKind::Rect { .. }), "{}: a soft key", kind.name());
            (l.position - center).normalize_or(Vec3::Z)
        };
        let rembrandt = key(BuiltinRig::Rembrandt);
        assert!(rembrandt.x < -0.5 && rembrandt.y > 0.4, "Rembrandt: high, to the character's right");
        let clamshell = key(BuiltinRig::Clamshell);
        assert!(clamshell.x.abs() < 1e-4 && clamshell.y > 0.4 && clamshell.z > 0.6, "clamshell: above the lens");
        let fill = rig(BuiltinRig::Clamshell).lights[1].position - center;
        assert!(fill.y < 0.0 && fill.z > 0.0, "clamshell: a reflector below");
        let loop_key = key(BuiltinRig::Loop);
        assert!(loop_key.x > 0.3 && loop_key.z > 0.6, "loop: 30 degrees to the side");
        for l in &rig(BuiltinRig::RimOnly).lights {
            assert!(l.position.z < center.z, "rim only: every light behind the subject");
        }
        // The key's design illuminance, and every new rig's lights are soft with a visible size.
        let three = rig(BuiltinRig::ThreePoint);
        assert!((three.lights[0].illuminance_at(center) - 1900.0).abs() < 19.0);
        let names: Vec<&str> = BuiltinRig::ALL.iter().map(|k| k.name()).collect();
        for (i, n) in names.iter().enumerate() {
            assert!(!names[..i].contains(n), "{n} is unique");
            assert_eq!(rig(BuiltinRig::ALL[i]).name, *n);
        }

        let studio = rig(BuiltinRig::NeutralStudio);
        assert_eq!(studio.clone().with_backdrop(BackdropChoice::Rig), studio);
        let paper = studio.clone().with_backdrop(BackdropChoice::Studio);
        assert_eq!(paper.background, Background::Backdrop(Backdrop::GRAY));
        assert_eq!(paper.lights, studio.lights);
        assert_eq!(
            rig(BuiltinRig::RimOnly).with_backdrop(BackdropChoice::Studio).background,
            Background::Backdrop(Backdrop::BLACK),
            "a dark grey becomes black paper"
        );
        let outdoor = rig(BuiltinRig::GoldenHour);
        assert_eq!(outdoor.clone().with_backdrop(BackdropChoice::Studio).background, Background::Environment);
        assert_eq!(outdoor.with_backdrop(BackdropChoice::Flat).background, Background::Gray(0.18));
        assert_eq!(studio.with_backdrop(BackdropChoice::Environment).background, Background::Environment);
        Backdrop::GRAY.validate().unwrap();
        Backdrop::BLACK.validate().unwrap();
        assert!(Backdrop { distance: 0.0, ..Backdrop::GRAY }.validate().is_err());
    }

    #[test]
    fn lights_reject_bad_parameters() {
        let mut l = Light::spot(Vec3::Y, Vec3::ZERO, 100.0, 0.5, 5600.0);
        l.validate().unwrap();
        l.kind = LightKind::Spot { inner: 0.6, outer: 0.5, range: 1.0, radius: 0.0 };
        assert!(l.validate().is_err());
        l = Light::directional(Vec3::new(f32::NAN, 0.0, 0.0), 1.0, 5000.0);
        assert!(l.validate().is_ok(), "a NaN direction falls back to straight down");
        l.intensity = -1.0;
        assert!(l.validate().is_err());
        let mut rig = LightRig::builtin(BuiltinRig::NeutralStudio, Vec3::ZERO, 0.1);
        rig.lights = vec![Light::directional(-Vec3::Y, 1.0, 5000.0); MAX_LIGHTS + 1];
        assert!(rig.validate().is_err());
    }

    #[test]
    fn spot_falls_off_with_distance_and_angle() {
        let l = Light::spot(Vec3::new(0.0, 2.0, 0.0), Vec3::ZERO, 1000.0, 0.4, 5600.0);
        let on_axis = l.illuminance_at(Vec3::ZERO);
        assert!((on_axis - 250.0).abs() < 5.0, "I / d^2 = 250 lux: {on_axis}");
        assert_eq!(l.illuminance_at(Vec3::new(5.0, 0.0, 0.0)), 0.0, "outside the cone");
    }

    #[test]
    fn rnd_03_shadow_frusta_contain_the_bounds() {
        let center = Vec3::new(0.0, 1.0, 0.0);
        let r = 0.5;
        let lights = [
            Light::directional(Vec3::new(0.3, -1.0, 0.2), 1000.0, 5600.0),
            Light::directional(-Vec3::Y, 1000.0, 5600.0),
            Light::spot(Vec3::new(1.0, 3.0, 2.0), center, 100.0, 0.6, 5600.0),
            Light::softbox(Vec3::new(-2.0, 2.0, 2.0), center, 1.0, 1.0, 100.0, 5600.0),
        ];
        for light in &lights {
            let m = ShadowFrustum::fit(light, center, r).view_projection;
            assert!(m.is_finite());
            for d in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
                let p = m.project_point3(center + d * (r * 0.99));
                assert!(p.x.abs() <= 1.0 && p.y.abs() <= 1.0 && (0.0..=1.0).contains(&p.z), "{light:?}: {p:?}");
            }
        }
    }

    #[test]
    fn gpu_light_packs_fields() {
        let l = Light::softbox(Vec3::new(0.0, 0.0, 2.0), Vec3::ZERO, 1.0, 0.5, 10.0, 6500.0);
        let g = GpuLight::new(&l, Some(2));
        assert_eq!(g.color_kind[3], 2.0);
        assert_eq!(g.shadow[0], 2.0);
        let right = Vec3::new(g.right_cos_inner[0], g.right_cos_inner[1], g.right_cos_inner[2]);
        let up = Vec3::new(g.up_size[0], g.up_size[1], g.up_size[2]);
        assert!((right.length() - 0.5).abs() < 1e-6 && (up.length() - 0.25).abs() < 1e-6);
        assert!(right.dot(up).abs() < 1e-6);
        assert_eq!(GpuLight::NONE.shadow[0], -1.0);
    }
}
