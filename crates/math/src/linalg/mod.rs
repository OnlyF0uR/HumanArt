//! Linear algebra, half floats, vertex packing, geometry and curves.
//!
//! Conventions (CLAUDE.md): right-handed, Y up, metres, the character faces +Z. Matrices are
//! column-major (`Mat4::cols`), a point transforms as `M * p`. Quaternions are `(x, y, z, w)`. Depth
//! is reversed-Z in wgpu clip space (`z` in `[0, 1]`, 1 at the near plane). Angles are radians.
//!
//! Every `#[repr(C)]` type here has a fixed, padding-free layout that `humanart-gpu` relies on for its `Pod`
//! implementations; the layouts are asserted at compile time next to each type.

pub mod curve;
pub mod geom;
mod half;
mod mat;
pub mod pack;
mod quat;
pub mod scalar;
mod vec;

pub use geom::{Aabb, Frustum, MeshHit, Plane, Ray, RayTriangleTester, Sphere, TriangleHit};
pub use half::{
    F16, f16_bits_to_f32, f16_le_bytes_to_f32, f16_to_f32_slice, f32_to_f16_bits, f32_to_f16_le_bytes, f32_to_f16_slice,
};
pub use mat::{
    DMat3, DMat4, Mat2, Mat3, Mat4, halton, ndc_to_screen, pixel_offset_to_ndc, screen_to_ndc, taa_jitter_pixels,
};
pub use quat::{DQuat, DualQuat, Quat, Transform};
pub use scalar::{
    angle_delta, approx_eq, approx_eq_f64, approx_eq_rel, approx_eq_rel_f64, inverse_lerp, lerp, lerp_angle, lerp_f64,
    remap, remap_clamped, saturate, wrap_angle, wrap_angle_positive,
};
pub use vec::{DVec2, DVec3, DVec4, IVec2, IVec3, UVec2, UVec3, UVec4, Vec2, Vec3, Vec4};
