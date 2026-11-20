//! Rotations and rigid transforms: `Quat` / `DQuat`, `DualQuat` for dual-quaternion skinning and the
//! translation-rotation-scale `Transform`.
//!
//! A quaternion is stored `(x, y, z, w)` with `w` the scalar part. The product is Hamilton's:
//! `(a * b) * v` rotates `v` by `b` first, then by `a`, the same order as matrices. Rotations follow
//! the right-hand rule. `q` and `-q` are the same rotation.
//!
//! Nothing here panics on bad numbers: normalising a zero or non-finite quaternion gives the identity.

use core::ops::{Add, Mul, MulAssign, Neg, Sub};

use super::mat::{DMat3, DMat4, Mat3, Mat4};
use super::vec::{DVec3, DVec4, Vec3, Vec4};

/// A quaternion type and the matrix constructors that need it.
macro_rules! quat_type {
    ($(#[$meta:meta])* $Q:ident, $V3:ident, $V4:ident, $M3:ident, $M4:ident, $T:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[repr(C)]
        pub struct $Q {
            /// The `x` component of the vector part.
            pub x: $T,
            /// The `y` component of the vector part.
            pub y: $T,
            /// The `z` component of the vector part.
            pub z: $T,
            /// The scalar part.
            pub w: $T,
        }

        // The layout is a contract with `humanart-gpu` (`Pod`): four scalars, no padding.
        const _: () = assert!(size_of::<$Q>() == 4 * size_of::<$T>() && align_of::<$Q>() == align_of::<$T>());

        impl $Q {
            /// No rotation.
            pub const IDENTITY: Self = Self::from_xyzw(0.0, 0.0, 0.0, 1.0);

            /// A quaternion from its components, not normalised.
            #[inline]
            pub const fn from_xyzw(x: $T, y: $T, z: $T, w: $T) -> Self {
                Self { x, y, z, w }
            }

            /// A quaternion from `[x, y, z, w]`, not normalised.
            #[inline]
            pub const fn from_array(a: [$T; 4]) -> Self {
                let [x, y, z, w] = a;
                Self { x, y, z, w }
            }

            /// The components as `[x, y, z, w]`.
            #[inline]
            pub const fn to_array(self) -> [$T; 4] {
                [self.x, self.y, self.z, self.w]
            }

            /// A quaternion from a vector `(x, y, z, w)`, not normalised.
            #[inline]
            pub const fn from_vec4(v: $V4) -> Self {
                Self { x: v.x, y: v.y, z: v.z, w: v.w }
            }

            /// The components as a vector `(x, y, z, w)`.
            #[inline]
            pub const fn to_vec4(self) -> $V4 {
                $V4::new(self.x, self.y, self.z, self.w)
            }

            /// The vector part `(x, y, z)`.
            #[inline]
            pub const fn xyz(self) -> $V3 {
                $V3::new(self.x, self.y, self.z)
            }

            /// The rotation by `angle` radians about the unit vector `axis` (right-hand rule).
            #[inline]
            pub fn from_axis_angle(axis: $V3, angle: $T) -> Self {
                let (s, c) = (0.5 * angle).sin_cos();
                Self::from_xyzw(axis.x * s, axis.y * s, axis.z * s, c)
            }

            /// The unit axis and the angle in `[0, π]` of this unit quaternion's rotation (the
            /// shorter way round). The identity returns `(+X, 0)`.
            #[inline]
            pub fn to_axis_angle(self) -> ($V3, $T) {
                let q = if self.w < 0.0 { -self } else { self };
                let v = q.xyz();
                let s = v.length();
                if s > 0.0 && s.is_finite() { (v / s, 2.0 * s.atan2(q.w)) } else { ($V3::X, 0.0) }
            }

            /// The rotation by `angle` radians about +X.
            #[inline]
            pub fn from_rotation_x(angle: $T) -> Self {
                let (s, c) = (0.5 * angle).sin_cos();
                Self::from_xyzw(s, 0.0, 0.0, c)
            }

            /// The rotation by `angle` radians about +Y.
            #[inline]
            pub fn from_rotation_y(angle: $T) -> Self {
                let (s, c) = (0.5 * angle).sin_cos();
                Self::from_xyzw(0.0, s, 0.0, c)
            }

            /// The rotation by `angle` radians about +Z.
            #[inline]
            pub fn from_rotation_z(angle: $T) -> Self {
                let (s, c) = (0.5 * angle).sin_cos();
                Self::from_xyzw(0.0, 0.0, s, c)
            }

            /// The rotation with Euler angles in the order YXZ: `yaw` about +Y, then `pitch` about
            /// the rotated X axis, then `roll` about the twice-rotated Z axis (intrinsic), which is
            /// the product `Ry(yaw) * Rx(pitch) * Rz(roll)`. With Y up and the character facing +Z,
            /// yaw turns the head left, positive pitch nods down, roll tilts it.
            pub fn from_euler_yxz(yaw: $T, pitch: $T, roll: $T) -> Self {
                Self::from_rotation_y(yaw) * Self::from_rotation_x(pitch) * Self::from_rotation_z(roll)
            }

            /// The Euler angles `(yaw, pitch, roll)` of this unit quaternion in the order of
            /// [`Self::from_euler_yxz`]: yaw and roll in `(-π, π]`, pitch in `[-π/2, π/2]`.
            ///
            /// At gimbal lock (pitch at ±π/2, where yaw and roll turn about the same axis) roll is
            /// reported as zero and yaw carries the whole turn; the angles still rebuild the same
            /// rotation.
            pub fn to_euler_yxz(self) -> ($T, $T, $T) {
                let $Q { x, y, z, w } = self;
                // Elements m[row][col] of the rotation matrix Ry * Rx * Rz.
                let m12 = 2.0 * (y * z - w * x);
                let pitch = (-m12).clamp(-1.0, 1.0).asin();
                if m12.abs() < 1.0 - 16.0 * $T::EPSILON {
                    let (m02, m22) = (2.0 * (x * z + w * y), 1.0 - 2.0 * (x * x + y * y));
                    let (m10, m11) = (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z));
                    (m02.atan2(m22), pitch, m10.atan2(m11))
                } else {
                    let (m20, m00) = (2.0 * (x * z - w * y), 1.0 - 2.0 * (y * y + z * z));
                    ((-m20).atan2(m00), pitch, 0.0)
                }
            }

            /// The rotation of the rotation matrix `m`, normalised. `m` must be orthonormal with
            /// determinant +1 (see `orthonormalize` on the matrix); anything else gives a
            /// meaningless but finite unit quaternion.
            pub fn from_mat3(m: $M3) -> Self {
                // Shepperd's method: pick the largest of w, x, y, z to divide by.
                let [[m00, m10, m20], [m01, m11, m21], [m02, m12, m22]] = m.to_cols_array_2d();
                let trace = m00 + m11 + m22;
                let q = if trace > 0.0 {
                    let s = (trace + 1.0).sqrt() * 2.0;
                    Self::from_xyzw((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
                } else if m00 > m11 && m00 > m22 {
                    let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
                    Self::from_xyzw(0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
                } else if m11 > m22 {
                    let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
                    Self::from_xyzw((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
                } else {
                    let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
                    Self::from_xyzw((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
                };
                q.normalize()
            }

            /// The rotation matrix of this unit quaternion.
            #[inline]
            pub fn to_mat3(self) -> $M3 {
                let $Q { x, y, z, w } = self;
                let (x2, y2, z2) = (x + x, y + y, z + z);
                let (xx, yy, zz) = (x * x2, y * y2, z * z2);
                let (xy, xz, yz) = (x * y2, x * z2, y * z2);
                let (wx, wy, wz) = (w * x2, w * y2, w * z2);
                $M3::from_cols(
                    $V3::new(1.0 - (yy + zz), xy + wz, xz - wy),
                    $V3::new(xy - wz, 1.0 - (xx + zz), yz + wx),
                    $V3::new(xz + wy, yz - wx, 1.0 - (xx + yy)),
                )
            }

            /// The shortest rotation that takes the unit vector `from` to the unit vector `to`.
            /// For opposite vectors, where every perpendicular axis is equally short, an arbitrary
            /// one is used.
            pub fn from_rotation_arc(from: $V3, to: $V3) -> Self {
                // The unnormalised result is (from × to, 1 + from · to). Written through the sum
                // `h = from + to` it has no cancellation when the vectors are nearly opposite:
                // from × to = from × h and 1 + from · to = |h|² / 2.
                let h = from + to;
                let c = from.cross(h);
                match Self::from_xyzw(c.x, c.y, c.z, 0.5 * h.length_squared()).try_normalize() {
                    Some(q) => q,
                    None => {
                        let axis = from.any_orthonormal_vector();
                        Self::from_xyzw(axis.x, axis.y, axis.z, 0.0).normalize()
                    }
                }
            }

            /// The four-dimensional dot product: the cosine of half the angle between two unit
            /// quaternions' rotations (negative when they are on opposite hemispheres).
            #[inline]
            pub fn dot(self, rhs: Self) -> $T {
                self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
            }

            /// The squared norm.
            #[inline]
            pub fn length_squared(self) -> $T {
                self.dot(self)
            }

            /// The norm; `1` for a rotation.
            #[inline]
            pub fn length(self) -> $T {
                self.length_squared().sqrt()
            }

            /// The unit quaternion in this direction, or `None` when the norm is zero or not finite.
            #[inline]
            pub fn try_normalize(self) -> Option<Self> {
                let len = self.length();
                if len > 0.0 && len.is_finite() { Some(self * len.recip()) } else { None }
            }

            /// The unit quaternion in this direction. A zero or non-finite quaternion gives
            /// [`Self::IDENTITY`].
            #[inline]
            pub fn normalize(self) -> Self {
                self.try_normalize().unwrap_or(Self::IDENTITY)
            }

            /// Whether the squared norm is within `1e-4` of one.
            #[inline]
            pub fn is_normalized(self) -> bool {
                (self.length_squared() - 1.0).abs() <= 1.0e-4
            }

            /// Whether every component is finite.
            #[inline]
            pub fn is_finite(self) -> bool {
                self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
            }

            /// Whether any component is NaN.
            #[inline]
            pub fn is_nan(self) -> bool {
                self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
            }

            /// Whether every component is within `epsilon` of the matching component of `rhs`.
            /// `q` and `-q` compare different here; see [`Self::angle_between`].
            #[inline]
            pub fn abs_diff_eq(self, rhs: Self, epsilon: $T) -> bool {
                self.to_vec4().abs_diff_eq(rhs.to_vec4(), epsilon)
            }

            /// The angle in `[0, π]` of the rotation that takes one unit quaternion's orientation to
            /// the other's; `0` for `q` and `-q`.
            #[inline]
            pub fn angle_between(self, rhs: Self) -> $T {
                // The chord between unit quaternions is 2 sin(half their 4D angle), and the 4D angle
                // is half the rotation angle. Unlike acos of the dot product this is accurate near
                // zero. The shorter chord of the two hemispheres is at most √2.
                let chord = (self - rhs).length().min((self + rhs).length());
                (4.0 * (0.5 * chord).min(1.0).asin()).min(core::$T::consts::PI)
            }

            /// The conjugate `(-x, -y, -z, w)`: the inverse rotation of a unit quaternion.
            #[inline]
            pub fn conjugate(self) -> Self {
                Self::from_xyzw(-self.x, -self.y, -self.z, self.w)
            }

            /// The multiplicative inverse, for quaternions of any norm. A zero or non-finite
            /// quaternion gives [`Self::IDENTITY`]. For a unit quaternion [`Self::conjugate`] is
            /// the same and cheaper.
            #[inline]
            pub fn inverse(self) -> Self {
                let len_sq = self.length_squared();
                if len_sq > 0.0 && len_sq.is_finite() { self.conjugate() * len_sq.recip() } else { Self::IDENTITY }
            }

            /// Rotates `v` by this unit quaternion.
            #[inline]
            pub fn mul_vec3(self, v: $V3) -> $V3 {
                let u = self.xyz();
                let t = u.cross(v) * 2.0;
                v + t * self.w + u.cross(t)
            }

            /// Normalised linear interpolation along the shorter arc: cheap, commutative, not
            /// constant in angular speed. `self` at `t = 0`, `end` (or `-end`) at `t = 1`.
            #[inline]
            pub fn nlerp(self, end: Self, t: $T) -> Self {
                let end = if self.dot(end) < 0.0 { -end } else { end };
                (self + (end - self) * t).normalize()
            }

            /// Spherical linear interpolation along the shorter arc at constant angular speed.
            ///
            /// Stable everywhere: the angle comes from the chord between the two (accurate where
            /// `acos` of the dot product is not), and inputs closer than rounding can resolve (and
            /// nearly opposite ones, which are nearly the same rotation) fall back to
            /// [`Self::nlerp`], which is then identical to rounding.
            pub fn slerp(self, end: Self, t: $T) -> Self {
                let end = if self.dot(end) < 0.0 { -end } else { end };
                // The 4D angle between the two, in [0, π/2] after the hemisphere choice.
                let theta = 2.0 * (0.5 * (end - self).length()).min(1.0).asin();
                let s = theta.sin();
                if s > $T::EPSILON.sqrt() {
                    (self * (((1.0 - t) * theta).sin() / s) + end * ((t * theta).sin() / s)).normalize()
                } else {
                    // The difference between the chord and the arc is of order theta³: below rounding.
                    (self + (end - self) * t).normalize()
                }
            }

            /// Splits this unit quaternion into a twist about the unit vector `axis` and a swing
            /// about an axis perpendicular to it: `self == swing * twist`, returned as
            /// `(swing, twist)`.
            ///
            /// When the rotation is a half turn about an axis perpendicular to `axis` the twist is
            /// undefined and reported as the identity.
            pub fn swing_twist(self, axis: $V3) -> (Self, Self) {
                let p = axis * self.xyz().dot(axis);
                let twist = Self::from_xyzw(p.x, p.y, p.z, self.w);
                let twist = if twist.length_squared() > $T::EPSILON { twist.normalize() } else { Self::IDENTITY };
                (self * twist.conjugate(), twist)
            }

            /// The natural logarithm of a unit quaternion as a vector: the rotation axis scaled by
            /// HALF the rotation angle, so that [`Self::exp`] undoes it. `ln` of `-1` (a full turn,
            /// whose axis is arbitrary) is `(π, 0, 0)`.
            pub fn ln(self) -> $V3 {
                let v = self.xyz();
                let s = v.length();
                if s > 0.0 {
                    v * (s.atan2(self.w) / s)
                } else if self.w >= 0.0 {
                    // The vector part underflowed: atan2(s, w) / s tends to 1 / w = 1.
                    v
                } else {
                    $V3::X * core::$T::consts::PI
                }
            }

            /// The exponential of the pure quaternion `(v, 0)`: the unit quaternion that rotates
            /// about `v` by TWICE its length. Accurate for arbitrarily small `v`.
            pub fn exp(v: $V3) -> Self {
                let theta = v.length();
                // sin(theta) / theta, by its series where the division would lose precision.
                let t2 = theta * theta;
                let sinc = if theta < 1.0e-3 { 1.0 - t2 / 6.0 * (1.0 - t2 / 20.0) } else { theta.sin() / theta };
                let v = v * sinc;
                Self::from_xyzw(v.x, v.y, v.z, theta.cos())
            }

            /// The rotation vector of this unit quaternion: the axis scaled by the angle in
            /// `[0, π]` (the shorter way round). Angular velocities and pose deltas use this form.
            #[inline]
            pub fn to_scaled_axis(self) -> $V3 {
                let q = if self.w < 0.0 { -self } else { self };
                q.ln() * 2.0
            }

            /// The rotation about `v` by its length in radians: the inverse of
            /// [`Self::to_scaled_axis`].
            #[inline]
            pub fn from_scaled_axis(v: $V3) -> Self {
                Self::exp(v * 0.5)
            }
        }

        /// The identity rotation.
        impl Default for $Q {
            #[inline]
            fn default() -> Self {
                Self::IDENTITY
            }
        }

        /// The Hamilton product: `(a * b) * v` rotates by `b` first.
        impl Mul for $Q {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                Self::from_xyzw(
                    self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
                    self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
                    self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
                    self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
                )
            }
        }

        impl MulAssign for $Q {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        /// Rotates the vector by this unit quaternion.
        impl Mul<$V3> for $Q {
            type Output = $V3;
            #[inline]
            fn mul(self, rhs: $V3) -> $V3 {
                self.mul_vec3(rhs)
            }
        }

        /// Scales every component (for blending; the result is not a rotation until normalised).
        impl Mul<$T> for $Q {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: $T) -> Self {
                Self::from_xyzw(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
            }
        }

        /// Component-wise sum (for blending; the result is not a rotation until normalised).
        impl Add for $Q {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                Self::from_xyzw(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z, self.w + rhs.w)
            }
        }

        /// Component-wise difference.
        impl Sub for $Q {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self::from_xyzw(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z, self.w - rhs.w)
            }
        }

        /// The same rotation on the opposite hemisphere.
        impl Neg for $Q {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self::from_xyzw(-self.x, -self.y, -self.z, -self.w)
            }
        }

        impl $M3 {
            /// The rotation matrix of the unit quaternion `q`.
            #[inline]
            pub fn from_quat(q: $Q) -> Self {
                q.to_mat3()
            }
        }

        impl $M4 {
            /// The rotation matrix of the unit quaternion `q`.
            #[inline]
            pub fn from_quat(q: $Q) -> Self {
                Self::from_mat3(q.to_mat3())
            }

            /// The rigid transform `p -> rotation * p + translation`.
            #[inline]
            pub fn from_rotation_translation(rotation: $Q, translation: $V3) -> Self {
                Self::from_mat3_translation(rotation.to_mat3(), translation)
            }

            /// The transform `p -> rotation * (scale * p) + translation`: the matrix
            /// `T * R * S`.
            #[inline]
            pub fn from_scale_rotation_translation(scale: $V3, rotation: $Q, translation: $V3) -> Self {
                let r = rotation.to_mat3();
                Self::from_mat3_translation(
                    $M3::from_cols(r.cols[0] * scale.x, r.cols[1] * scale.y, r.cols[2] * scale.z),
                    translation,
                )
            }

            /// Decomposes an affine transform into `(scale, rotation, translation)` such that
            /// [`Self::from_scale_rotation_translation`] rebuilds it.
            ///
            /// The rebuild is exact (to rounding) for matrices of the form `T * R * S`. A mirrored
            /// matrix (negative determinant) reports a negative `scale.x`. Shear cannot be
            /// represented and is dropped; a zero-scale axis leaves the rotation about it
            /// arbitrary. The rotation returned is always a finite unit quaternion.
            pub fn to_scale_rotation_translation(self) -> ($V3, $Q, $V3) {
                let m = self.to_mat3();
                let mut scale = $V3::new(m.cols[0].length(), m.cols[1].length(), m.cols[2].length());
                let mut basis = m;
                if m.determinant() < 0.0 {
                    scale.x = -scale.x;
                    basis.cols[0] = -basis.cols[0];
                }
                (scale, $Q::from_mat3(basis.orthonormalize()), self.translation())
            }
        }
    };
}

quat_type!(
    /// A rotation as a unit `f32` quaternion `(x, y, z, w)`. 16 bytes.
    Quat, Vec3, Vec4, Mat3, Mat4, f32
);
quat_type!(
    /// A rotation as a unit `f64` quaternion `(x, y, z, w)`, for numerics that need double precision.
    DQuat, DVec3, DVec4, DMat3, DMat4, f64
);

impl Quat {
    /// The same quaternion in double precision (exact).
    #[inline]
    pub const fn as_dquat(self) -> DQuat {
        DQuat::from_xyzw(self.x as f64, self.y as f64, self.z as f64, self.w as f64)
    }
}

impl DQuat {
    /// The same quaternion rounded to single precision (renormalise if it must stay a rotation).
    #[inline]
    pub const fn as_quat(self) -> Quat {
        Quat::from_xyzw(self.x as f32, self.y as f32, self.z as f32, self.w as f32)
    }
}

impl From<Quat> for DQuat {
    #[inline]
    fn from(q: Quat) -> Self {
        q.as_dquat()
    }
}

/// A rigid transform (rotation + translation) as a unit dual quaternion, for dual-quaternion
/// skinning: blending these and renormalising keeps volume where blending matrices collapses it.
///
/// `real` is the rotation; `dual` is `0.5 * (translation, 0) * real`. 32 bytes, two `vec4`s.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct DualQuat {
    /// The real part: the rotation.
    pub real: Quat,
    /// The dual part: half the translation times the rotation.
    pub dual: Quat,
}

// The layout is a contract with `humanart-gpu` (`Pod`): eight `f32`, no padding.
const _: () = assert!(size_of::<DualQuat>() == 32 && align_of::<DualQuat>() == 4);

impl DualQuat {
    /// The identity transform.
    pub const IDENTITY: Self = Self { real: Quat::IDENTITY, dual: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0) };
    /// All zeros: the start value for accumulating a weighted blend. Not a transform.
    pub const ZERO: Self =
        Self { real: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0), dual: Quat::from_xyzw(0.0, 0.0, 0.0, 0.0) };

    /// The transform `p -> rotation * p + translation`; `rotation` must be a unit quaternion.
    #[inline]
    pub fn from_rotation_translation(rotation: Quat, translation: Vec3) -> Self {
        let t = Quat::from_xyzw(translation.x, translation.y, translation.z, 0.0);
        Self { real: rotation, dual: (t * rotation) * 0.5 }
    }

    /// The rotation of this unit dual quaternion.
    #[inline]
    pub fn rotation(self) -> Quat {
        self.real
    }

    /// The translation of this unit dual quaternion.
    #[inline]
    pub fn translation(self) -> Vec3 {
        (self.dual * self.real.conjugate()).xyz() * 2.0
    }

    /// The unit dual quaternion with the same direction: the real part gets norm one and the dual
    /// part is made orthogonal to it. A zero or non-finite real part gives [`Self::IDENTITY`].
    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.real.length();
        if len > 0.0 && len.is_finite() {
            let inv = len.recip();
            let (real, dual) = (self.real * inv, self.dual * inv);
            let out = Self { real, dual: dual - real * real.dot(dual) };
            if out.dual.is_finite() { out } else { Self::IDENTITY }
        } else {
            Self::IDENTITY
        }
    }

    /// The inverse of a unit dual quaternion.
    #[inline]
    pub fn inverse(self) -> Self {
        Self { real: self.real.conjugate(), dual: self.dual.conjugate() }
    }

    /// Transforms a point by this unit dual quaternion.
    #[inline]
    pub fn transform_point3(self, p: Vec3) -> Vec3 {
        self.real * p + self.translation()
    }

    /// Rotates a direction by this unit dual quaternion (the translation does not apply).
    #[inline]
    pub fn transform_vector3(self, v: Vec3) -> Vec3 {
        self.real * v
    }

    /// The matrix of this unit dual quaternion.
    #[inline]
    pub fn to_mat4(self) -> Mat4 {
        Mat4::from_rotation_translation(self.real, self.translation())
    }

    /// `self + other * weight`, with `other` first flipped onto the hemisphere of `pivot` so that
    /// the two representations of one rotation (`q` and `-q`) do not cancel. This is one step of
    /// dual-quaternion linear blending: start from [`Self::ZERO`], accumulate every influence with
    /// the first influence as `pivot`, then [`Self::normalize`].
    #[inline]
    pub fn accumulate(self, pivot: Self, other: Self, weight: f32) -> Self {
        let weight = if pivot.real.dot(other.real) < 0.0 { -weight } else { weight };
        self + other * weight
    }

    /// The normalised weighted blend of rigid transforms (dual-quaternion linear blending).
    /// Pairs are taken up to the shorter slice; no influences, or weights that cancel, give
    /// [`Self::IDENTITY`].
    pub fn blend(transforms: &[Self], weights: &[f32]) -> Self {
        let Some(&pivot) = transforms.first() else {
            return Self::IDENTITY;
        };
        let mut sum = Self::ZERO;
        for (&dq, &weight) in transforms.iter().zip(weights) {
            sum = sum.accumulate(pivot, dq, weight);
        }
        sum.normalize()
    }
}

/// The identity transform.
impl Default for DualQuat {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Composition: `(a * b)` applies `b` first.
impl Mul for DualQuat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self { real: self.real * rhs.real, dual: self.real * rhs.dual + self.dual * rhs.real }
    }
}

/// Scales all eight components (for blending).
impl Mul<f32> for DualQuat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self { real: self.real * rhs, dual: self.dual * rhs }
    }
}

/// Component-wise sum (for blending).
impl Add for DualQuat {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { real: self.real + rhs.real, dual: self.dual + rhs.dual }
    }
}

/// A translation, rotation and scale: `p -> rotation * (scale * p) + translation`, the matrix
/// `T * R * S`. The form joints, cameras and scene objects are stored and animated in. 40 bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Transform {
    /// The translation, applied last.
    pub translation: Vec3,
    /// The rotation (a unit quaternion), applied after the scale.
    pub rotation: Quat,
    /// The per-axis scale, applied first.
    pub scale: Vec3,
}

// Ten `f32`, no padding.
const _: () = assert!(size_of::<Transform>() == 40 && align_of::<Transform>() == 4);

impl Transform {
    /// The identity transform.
    pub const IDENTITY: Self = Self { translation: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE };

    /// A transform from its parts.
    #[inline]
    pub const fn new(translation: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self { translation, rotation, scale }
    }

    /// A pure translation.
    #[inline]
    pub const fn from_translation(translation: Vec3) -> Self {
        Self { translation, ..Self::IDENTITY }
    }

    /// A pure rotation.
    #[inline]
    pub const fn from_rotation(rotation: Quat) -> Self {
        Self { rotation, ..Self::IDENTITY }
    }

    /// A pure scale.
    #[inline]
    pub const fn from_scale(scale: Vec3) -> Self {
        Self { scale, ..Self::IDENTITY }
    }

    /// A rigid transform: rotation, then translation.
    #[inline]
    pub const fn from_rotation_translation(rotation: Quat, translation: Vec3) -> Self {
        Self { translation, rotation, scale: Vec3::ONE }
    }

    /// The matrix `T * R * S`.
    #[inline]
    pub fn to_mat4(self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// The transform of an affine matrix (see [`Mat4::to_scale_rotation_translation`] for what is
    /// lost when the matrix has shear).
    #[inline]
    pub fn from_mat4(m: Mat4) -> Self {
        let (scale, rotation, translation) = m.to_scale_rotation_translation();
        Self { translation, rotation, scale }
    }

    /// Transforms a point: scale, rotate, translate.
    #[inline]
    pub fn transform_point3(self, p: Vec3) -> Vec3 {
        self.rotation * (self.scale * p) + self.translation
    }

    /// Transforms a direction: scale and rotate.
    #[inline]
    pub fn transform_vector3(self, v: Vec3) -> Vec3 {
        self.rotation * (self.scale * v)
    }

    /// Maps a point back through this transform: the exact inverse of [`Self::transform_point3`]
    /// for any non-zero scale. A zero scale component gives non-finite components.
    #[inline]
    pub fn inverse_transform_point3(self, p: Vec3) -> Vec3 {
        (self.rotation.conjugate() * (p - self.translation)) / self.scale
    }

    /// Maps a direction back through this transform (see [`Self::inverse_transform_point3`]).
    #[inline]
    pub fn inverse_transform_vector3(self, v: Vec3) -> Vec3 {
        (self.rotation.conjugate() * v) / self.scale
    }

    /// The composition `self ∘ child`: the transform of a child joint in the space `self` lives
    /// in (`child` is applied first).
    ///
    /// Exact when `self.scale` is uniform. A non-uniform parent scale under a rotated child would
    /// need shear, which this form cannot hold: the result then keeps the scales multiplied per
    /// axis, as animation systems conventionally do. Use the matrices where shear matters.
    #[inline]
    pub fn mul_transform(self, child: Self) -> Self {
        Self {
            translation: self.transform_point3(child.translation),
            rotation: (self.rotation * child.rotation).normalize(),
            scale: self.scale * child.scale,
        }
    }

    /// The inverse transform, or `None` when a scale component is zero or anything is not finite.
    ///
    /// Exact when the scale is uniform. With a non-uniform scale the true inverse is `S⁻¹ R⁻¹ T⁻¹`,
    /// which has no translation-rotation-scale form; the result is then only an approximation
    /// and [`Self::inverse_transform_point3`] or the inverse matrix must be used instead.
    #[inline]
    pub fn inverse(self) -> Option<Self> {
        let scale = self.scale.recip();
        let rotation = self.rotation.conjugate();
        let out = Self { translation: -(rotation * self.translation) * scale, rotation, scale };
        (out.translation.is_finite() && out.rotation.is_finite() && out.scale.is_finite()).then_some(out)
    }

    /// Interpolates between two transforms: translation and scale linearly, rotation by slerp
    /// along the shorter arc.
    #[inline]
    pub fn interpolate(self, end: Self, t: f32) -> Self {
        Self {
            translation: self.translation.lerp(end.translation, t),
            rotation: self.rotation.slerp(end.rotation, t),
            scale: self.scale.lerp(end.scale, t),
        }
    }

    /// The rigid part as a dual quaternion (the scale is dropped), for dual-quaternion skinning.
    #[inline]
    pub fn to_dual_quat(self) -> DualQuat {
        DualQuat::from_rotation_translation(self.rotation, self.translation)
    }
}

/// The identity transform.
impl Default for Transform {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Composition, see [`Transform::mul_transform`]: `(a * b)` applies `b` first.
impl Mul for Transform {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_transform(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    use humanart_core::rng::Rng;

    fn rand_vec3(rng: &mut Rng, extent: f32) -> Vec3 {
        Vec3::new(rng.range_f32(-extent, extent), rng.range_f32(-extent, extent), rng.range_f32(-extent, extent))
    }

    fn rand_unit(rng: &mut Rng) -> Vec3 {
        loop {
            let v = rand_vec3(rng, 1.0);
            if v.length_squared() > 0.01 && v.length_squared() <= 1.0 {
                return v.normalize();
            }
        }
    }

    /// A uniformly distributed random rotation (normalised 4D Gaussian).
    fn rand_quat(rng: &mut Rng) -> Quat {
        rand_dquat(rng).as_quat().normalize()
    }

    fn rand_dquat(rng: &mut Rng) -> DQuat {
        DQuat::from_xyzw(rng.normal(), rng.normal(), rng.normal(), rng.normal()).normalize()
    }

    /// Whether two unit quaternions are the same rotation (`q` and `-q` are).
    fn same_rotation(a: Quat, b: Quat, epsilon: f32) -> bool {
        a.abs_diff_eq(b, epsilon) || a.abs_diff_eq(-b, epsilon)
    }

    #[test]
    fn layouts_are_the_gpu_contract() {
        assert_eq!((size_of::<Quat>(), align_of::<Quat>()), (16, 4));
        assert_eq!((size_of::<DQuat>(), align_of::<DQuat>()), (32, align_of::<f64>()));
        assert_eq!((size_of::<DualQuat>(), align_of::<DualQuat>()), (32, 4));
        assert_eq!((size_of::<Transform>(), align_of::<Transform>()), (40, 4));
        assert_eq!(core::mem::offset_of!(Quat, w), 12);
        assert_eq!(core::mem::offset_of!(DualQuat, dual), 16);
        assert_eq!(core::mem::offset_of!(Transform, rotation), 12);
        assert_eq!(core::mem::offset_of!(Transform, scale), 28);
        let q = Quat::from_array([1.0, 2.0, 3.0, 4.0]);
        assert_eq!((q.x, q.y, q.z, q.w), (1.0, 2.0, 3.0, 4.0));
        assert_eq!(q.to_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(Quat::from_vec4(q.to_vec4()), q);
        assert_eq!(q.xyz(), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(Quat::default(), Quat::IDENTITY);
        assert_eq!(DQuat::from(q).as_quat(), q);
    }

    #[test]
    fn axis_angle_rotations_are_right_handed() {
        assert!((Quat::from_rotation_z(FRAC_PI_2) * Vec3::X).abs_diff_eq(Vec3::Y, 1e-6));
        assert!((Quat::from_rotation_x(FRAC_PI_2) * Vec3::Y).abs_diff_eq(Vec3::Z, 1e-6));
        assert!((Quat::from_rotation_y(FRAC_PI_2) * Vec3::Z).abs_diff_eq(Vec3::X, 1e-6));
        assert_eq!(Quat::IDENTITY * Vec3::new(1.0, 2.0, 3.0), Vec3::new(1.0, 2.0, 3.0));
        let mut rng = Rng::new(0x51);
        for _ in 0..2000 {
            let (axis, angle) = (rand_unit(&mut rng), rng.range_f32(0.01, PI - 0.01));
            let q = Quat::from_axis_angle(axis, angle);
            assert!(q.is_normalized());
            let (axis2, angle2) = q.to_axis_angle();
            assert!(axis2.abs_diff_eq(axis, 2e-4) && (angle2 - angle).abs() < 1e-4, "{axis:?} {angle}");
            // The long way round comes back as the short way about the opposite axis.
            let (axis3, angle3) = Quat::from_axis_angle(axis, angle - core::f32::consts::TAU).to_axis_angle();
            assert!(axis3.abs_diff_eq(axis, 2e-4) && (angle3 - angle).abs() < 1e-4);
            let v = rand_vec3(&mut rng, 3.0);
            assert!((q * v).abs_diff_eq(Mat3::from_axis_angle(axis, angle) * v, 1e-4));
            assert!(((q * v).length() - v.length()).abs() < 1e-4);
        }
        assert_eq!(Quat::IDENTITY.to_axis_angle(), (Vec3::X, 0.0));
        assert_eq!(Quat::from_rotation_x(0.3), Quat::from_axis_angle(Vec3::X, 0.3));
    }

    #[test]
    fn products_compose_like_matrices() {
        let mut rng = Rng::new(0x52);
        for _ in 0..2000 {
            let (a, b, c) = (rand_quat(&mut rng), rand_quat(&mut rng), rand_quat(&mut rng));
            let v = rand_vec3(&mut rng, 2.0);
            assert!(((a * b) * v).abs_diff_eq(a * (b * v), 1e-4));
            assert!(((a * b) * c).abs_diff_eq(a * (b * c), 1e-5));
            assert!((a * b).to_mat3().abs_diff_eq(a.to_mat3() * b.to_mat3(), 1e-5));
            assert!((a * a.conjugate()).abs_diff_eq(Quat::IDENTITY, 1e-5));
            assert!((a.inverse() * (a * v)).abs_diff_eq(v, 1e-4));
            assert!(((a * 3.0).inverse() * (a * 3.0)).abs_diff_eq(Quat::IDENTITY, 1e-5));
            assert!(((a * b).length() - 1.0).abs() < 1e-5);
            assert!((-a * v).abs_diff_eq(a * v, 1e-5), "q and -q are one rotation");
            let mut d = a;
            d *= b;
            assert_eq!(d, a * b);
            assert_eq!(a.mul_vec3(v), a * v);
            assert_eq!(Mat4::from_quat(a).to_mat3(), Mat3::from_quat(a));
        }
        // i * j = k: Hamilton's convention.
        let (i, j, k) = (
            Quat::from_xyzw(1.0, 0.0, 0.0, 0.0),
            Quat::from_xyzw(0.0, 1.0, 0.0, 0.0),
            Quat::from_xyzw(0.0, 0.0, 1.0, 0.0),
        );
        assert_eq!(i * j, k);
        assert_eq!(j * k, i);
        assert_eq!(k * i, j);
        assert_eq!(i * i, -Quat::IDENTITY);
    }

    #[test]
    fn matrix_round_trip_in_every_shepperd_branch() {
        let mut rng = Rng::new(0x53);
        for _ in 0..5000 {
            let q = rand_quat(&mut rng);
            let m = q.to_mat3();
            assert!((m * m.transpose()).abs_diff_eq(Mat3::IDENTITY, 1e-5));
            assert!((m.determinant() - 1.0).abs() < 1e-5);
            assert!(same_rotation(Quat::from_mat3(m), q, 1e-5), "{q:?}");
            let d = rand_dquat(&mut rng);
            let back = DQuat::from_mat3(d.to_mat3());
            assert!(back.abs_diff_eq(d, 1e-14) || back.abs_diff_eq(-d, 1e-14));
        }
        // The half turns exercise the three trace <= 0 branches exactly.
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let q = Quat::from_axis_angle(axis, PI);
            assert!(same_rotation(Quat::from_mat3(q.to_mat3()), q, 1e-6));
        }
        assert_eq!(Quat::from_mat3(Mat3::IDENTITY), Quat::IDENTITY);
        // Garbage in: still a finite unit quaternion.
        for m in [Mat3::ZERO, Mat3::IDENTITY * f32::NAN, Mat3::IDENTITY * -5.0, Mat3::IDENTITY * f32::INFINITY] {
            let q = Quat::from_mat3(m);
            assert!(q.is_finite() && q.is_normalized(), "{m:?} -> {q:?}");
        }
    }

    #[test]
    fn euler_yxz_order_is_yaw_pitch_roll() {
        let mut rng = Rng::new(0x54);
        for _ in 0..3000 {
            let yaw = rng.range_f32(-PI, PI);
            let pitch = rng.range_f32(-FRAC_PI_2 + 0.02, FRAC_PI_2 - 0.02);
            let roll = rng.range_f32(-PI, PI);
            let q = Quat::from_euler_yxz(yaw, pitch, roll);
            let m = Mat3::from_rotation_y(yaw) * Mat3::from_rotation_x(pitch) * Mat3::from_rotation_z(roll);
            assert!(q.to_mat3().abs_diff_eq(m, 1e-5));
            let (y2, p2, r2) = q.to_euler_yxz();
            assert!(same_rotation(Quat::from_euler_yxz(y2, p2, r2), q, 1e-4));
            let wrap = |a: f32| (a + PI).rem_euclid(core::f32::consts::TAU) - PI;
            assert!(wrap(y2 - yaw).abs() < 2e-3 && (p2 - pitch).abs() < 2e-3 && wrap(r2 - roll).abs() < 2e-3);
        }
        // Facing +Z: yaw turns towards +X, positive pitch nods down (towards -Y).
        assert!((Quat::from_euler_yxz(FRAC_PI_2, 0.0, 0.0) * Vec3::Z).abs_diff_eq(Vec3::X, 1e-6));
        assert!((Quat::from_euler_yxz(0.0, FRAC_PI_2, 0.0) * Vec3::Z).abs_diff_eq(-Vec3::Y, 1e-6));
        // Gimbal lock: the angles differ but the rotation is rebuilt.
        for pitch in [core::f64::consts::FRAC_PI_2, -core::f64::consts::FRAC_PI_2] {
            let q = DQuat::from_euler_yxz(0.7, pitch, -0.4);
            let (y, p, r) = q.to_euler_yxz();
            assert_eq!(r, 0.0);
            let back = DQuat::from_euler_yxz(y, p, r);
            assert!(back.abs_diff_eq(q, 1e-6) || back.abs_diff_eq(-q, 1e-6), "{q:?} vs {back:?}");
        }
        assert_eq!(Quat::IDENTITY.to_euler_yxz(), (0.0, 0.0, 0.0));
    }

    #[test]
    fn rotation_arc_takes_from_to_to() {
        let mut rng = Rng::new(0x55);
        for _ in 0..3000 {
            let (a, b) = (rand_unit(&mut rng), rand_unit(&mut rng));
            let q = Quat::from_rotation_arc(a, b);
            assert!(q.is_normalized());
            assert!((q * a).abs_diff_eq(b, 1e-4), "{a:?} -> {b:?}");
            // Shortest: the rotation angle is the angle between the vectors.
            assert!((q.to_axis_angle().1 - a.angle_between(b)).abs() < 2e-3);
            assert!((Quat::from_rotation_arc(a, -a) * a).abs_diff_eq(-a, 1e-5), "half turn");
            assert!(Quat::from_rotation_arc(a, a).abs_diff_eq(Quat::IDENTITY, 1e-6));
        }
        let q = DQuat::from_rotation_arc(DVec3::X, DVec3::new(-1.0, 1e-9, 0.0).normalize());
        assert!((q * DVec3::X).abs_diff_eq(DVec3::new(-1.0, 1e-9, 0.0), 1e-8));
    }

    #[test]
    fn normalisation_never_fails() {
        assert_eq!(Quat::from_xyzw(0.0, 0.0, 0.0, 0.0).normalize(), Quat::IDENTITY);
        assert_eq!(Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0).normalize(), Quat::IDENTITY);
        assert_eq!(Quat::from_xyzw(f32::INFINITY, 0.0, 0.0, 1.0).normalize(), Quat::IDENTITY);
        assert_eq!(Quat::from_xyzw(0.0, 0.0, 0.0, 0.0).try_normalize(), None);
        assert_eq!(Quat::from_xyzw(0.0, 0.0, 0.0, 0.0).inverse(), Quat::IDENTITY);
        assert_eq!(Quat::from_xyzw(0.0, 3.0, 0.0, 4.0).normalize(), Quat::from_xyzw(0.0, 0.6, 0.0, 0.8));
        assert_eq!(Quat::from_xyzw(0.0, 3.0, 0.0, 4.0).length(), 5.0);
        assert!(Quat::from_xyzw(f32::NAN, 0.0, 0.0, 1.0).is_nan() && !Quat::IDENTITY.is_nan());
        assert!(!Quat::from_xyzw(f32::INFINITY, 0.0, 0.0, 1.0).is_finite());
    }

    #[test]
    fn slerp_has_constant_angular_speed_on_the_short_arc() {
        let mut rng = Rng::new(0x56);
        for _ in 0..2000 {
            let (a, b) = (rand_dquat(&mut rng), rand_dquat(&mut rng));
            let total = a.angle_between(b);
            assert!(a.slerp(b, 0.0).abs_diff_eq(a, 1e-12));
            assert!(a.slerp(b, 1.0).angle_between(b) < 1e-7);
            let t = rng.f64();
            let q = a.slerp(b, t);
            assert!((q.length() - 1.0).abs() < 1e-12);
            // Unless the two are nearly a half turn apart (where the short arc is ambiguous), the
            // angle travelled is t times the whole angle, and the rest is (1 - t) times it.
            if total < core::f64::consts::PI - 1e-3 {
                assert!((a.angle_between(q) - t * total).abs() < 1e-6, "{t} {total}");
                assert!((q.angle_between(b) - (1.0 - t) * total).abs() < 1e-6);
            }
            // The antipodal representation gives the same path.
            assert!(a.slerp(-b, t).angle_between(q) < 1e-7);
            // nlerp follows the same arc with the same ends, only at a different speed.
            let n = a.nlerp(b, t);
            assert!((n.length() - 1.0).abs() < 1e-12);
            assert!((a.angle_between(n) + n.angle_between(b) - total).abs() < 1e-6 || total > 3.0);
        }
        // Analytic: halfway to a turn about an axis is half the turn.
        let axis = Vec3::new(0.0, 0.6, 0.8);
        for (angle, t) in [(2.0_f32, 0.25_f32), (3.0, 0.5), (0.5, 0.9), (1e-3, 0.3), (3.1, 0.7)] {
            let q = Quat::IDENTITY.slerp(Quat::from_axis_angle(axis, angle), t);
            assert!(q.abs_diff_eq(Quat::from_axis_angle(axis, angle * t), 1e-6), "{angle} {t}");
        }
    }

    #[test]
    fn slerp_is_stable_near_zero_and_near_pi() {
        let axis = DVec3::new(0.6, 0.0, 0.8);
        let a = DQuat::from_axis_angle(DVec3::Y, 0.3);
        // Nearly equal inputs, across the nlerp threshold, down to identical.
        for delta in [0.0, 1e-300, 1e-12, 1e-6, 1e-3, 0.06, 0.07, 0.1] {
            let b = a * DQuat::from_axis_angle(axis, delta);
            for i in 0..=10 {
                let t = f64::from(i) / 10.0;
                let q = a.slerp(b, t);
                let want = a * DQuat::from_axis_angle(axis, delta * t);
                assert!(q.is_finite() && q.abs_diff_eq(want, 1e-9), "{delta} {t}: {q:?} vs {want:?}");
            }
        }
        // Nearly opposite quaternions are nearly the same rotation: the result stays next to `a`.
        for delta in [0.0, 1e-12, 1e-6, 1e-3] {
            let b = -(a * DQuat::from_axis_angle(axis, delta));
            for i in 0..=10 {
                let q = a.slerp(b, f64::from(i) / 10.0);
                assert!(q.is_finite() && q.angle_between(a) <= delta + 1e-7, "{delta}");
            }
        }
        // A rotation difference of almost exactly π (dot ~ 0): finite, unit, and on the arc.
        for delta in [-1e-9, 0.0, 1e-9] {
            let b = a * DQuat::from_axis_angle(axis, core::f64::consts::PI + delta);
            let q = a.slerp(b, 0.5);
            assert!(q.is_finite() && (q.length() - 1.0).abs() < 1e-12);
            assert!((a.angle_between(q) - core::f64::consts::FRAC_PI_2).abs() < 1e-6);
        }
        let q = Quat::IDENTITY.slerp(Quat::IDENTITY, 0.5);
        assert_eq!(q, Quat::IDENTITY);
        assert_eq!(Quat::IDENTITY.angle_between(-Quat::IDENTITY), 0.0);
        assert!((Quat::IDENTITY.angle_between(Quat::from_rotation_x(PI)) - PI).abs() < 1e-6);
        assert!((DQuat::IDENTITY.angle_between(DQuat::from_rotation_x(1e-9)) - 1e-9).abs() < 1e-15);
    }

    #[test]
    fn swing_twist_recomposes_and_separates_the_axes() {
        let mut rng = Rng::new(0x57);
        for _ in 0..3000 {
            let (q, axis) = (rand_quat(&mut rng), rand_unit(&mut rng));
            let (swing, twist) = q.swing_twist(axis);
            assert!(same_rotation(swing * twist, q, 1e-5));
            assert!(swing.is_normalized() && twist.is_normalized());
            assert!(twist.xyz().cross(axis).length() < 1e-5, "twist turns about the axis");
            assert!(swing.xyz().dot(axis).abs() < 1e-4, "swing turns about a perpendicular axis");
            // A pure twist has no swing and the other way round.
            let angle = rng.range_f32(-3.0, 3.0);
            let (s, t) = Quat::from_axis_angle(axis, angle).swing_twist(axis);
            assert!(same_rotation(s, Quat::IDENTITY, 1e-5));
            assert!(same_rotation(t, Quat::from_axis_angle(axis, angle), 1e-5));
            let perp = axis.any_orthonormal_vector();
            let (s, t) = Quat::from_axis_angle(perp, angle).swing_twist(axis);
            assert!(same_rotation(t, Quat::IDENTITY, 1e-5));
            assert!(same_rotation(s, Quat::from_axis_angle(perp, angle), 1e-5));
        }
        // The singular case: a half turn about a perpendicular axis.
        let (s, t) = Quat::from_xyzw(1.0, 0.0, 0.0, 0.0).swing_twist(Vec3::Y);
        assert_eq!((s, t), (Quat::from_xyzw(1.0, 0.0, 0.0, 0.0), Quat::IDENTITY));
    }

    #[test]
    fn log_and_exp_are_inverses() {
        let mut rng = Rng::new(0x58);
        for _ in 0..3000 {
            let q = rand_dquat(&mut rng);
            assert!(DQuat::exp(q.ln()).abs_diff_eq(q, 1e-12), "{q:?}");
            let (axis, angle) = (rand_unit(&mut rng).as_dvec3().normalize(), rng.range_f64(0.0, 3.1));
            let q = DQuat::from_axis_angle(axis, angle);
            assert!(q.ln().abs_diff_eq(axis * (angle * 0.5), 1e-12));
            assert!(q.to_scaled_axis().abs_diff_eq(axis * angle, 1e-12));
            assert!(DQuat::from_scaled_axis(axis * angle).abs_diff_eq(q, 1e-12));
            // The rotation vector of the antipode is the same (shorter way round).
            assert!((-q).to_scaled_axis().abs_diff_eq(axis * angle, 1e-9));
            let f = rand_quat(&mut rng);
            assert!(Quat::exp(f.ln()).abs_diff_eq(f, 1e-5));
        }
        // Tiny rotations keep full relative precision.
        for len in [0.0, 1e-200, 1e-12, 1e-6, 0.999e-3, 1.001e-3] {
            let v = DVec3::new(0.6, 0.0, -0.8) * len;
            let q = DQuat::exp(v);
            assert!((q.length() - 1.0).abs() < 1e-14);
            assert!(q.xyz().abs_diff_eq(v * (len.sin() / len.max(1e-300)).min(1.0), len * 1e-14 + 1e-300), "{len}");
            assert!(q.ln().abs_diff_eq(v, len * 1e-12 + 1e-300), "{len}");
        }
        assert_eq!(Quat::IDENTITY.ln(), Vec3::ZERO);
        assert_eq!(Quat::exp(Vec3::ZERO), Quat::IDENTITY);
        assert_eq!((-Quat::IDENTITY).ln(), Vec3::X * PI);
        assert_eq!((-Quat::IDENTITY).to_scaled_axis(), Vec3::ZERO);
    }

    #[test]
    fn trs_compose_and_decompose() {
        let mut rng = Rng::new(0x59);
        for _ in 0..3000 {
            let (t, r) = (rand_vec3(&mut rng, 5.0), rand_quat(&mut rng));
            let mut s = Vec3::new(rng.range_f32(0.2, 3.0), rng.range_f32(0.2, 3.0), rng.range_f32(0.2, 3.0));
            if rng.chance(0.3) {
                s.x = -s.x;
            }
            let m = Mat4::from_scale_rotation_translation(s, r, t);
            let p = rand_vec3(&mut rng, 2.0);
            assert!(m.transform_point3(p).abs_diff_eq(r * (s * p) + t, 1e-4));
            let expect = Mat4::from_translation(t) * Mat4::from_quat(r) * Mat4::from_scale(s);
            assert!(m.abs_diff_eq(expect, 1e-5));
            let (s2, r2, t2) = m.to_scale_rotation_translation();
            assert!(s2.abs_diff_eq(s, 1e-4), "{s:?} vs {s2:?}");
            assert!(same_rotation(r2, r, 1e-4));
            assert_eq!(t2, t);
            assert!(Mat4::from_scale_rotation_translation(s2, r2, t2).abs_diff_eq(m, 1e-4));
            assert_eq!(Mat4::from_rotation_translation(r, t), Mat4::from_scale_rotation_translation(Vec3::ONE, r, t));
        }
        // Degenerate matrices decompose to something finite that never panics.
        for m in [Mat4::ZERO, Mat4::from_scale(Vec3::new(1.0, 0.0, 2.0)), Mat4::from_scale(Vec3::ZERO)] {
            let (s, r, t) = m.to_scale_rotation_translation();
            assert!(s.is_finite() && r.is_normalized() && t.is_finite(), "{m:?}");
        }
        let (s, r, _) = Mat4::from_scale(Vec3::new(1.0, 0.0, 2.0)).to_scale_rotation_translation();
        assert!(s.abs_diff_eq(Vec3::new(1.0, 0.0, 2.0), 1e-6));
        let rebuilt = Mat4::from_scale_rotation_translation(s, r, Vec3::ZERO);
        assert!(rebuilt.abs_diff_eq(Mat4::from_scale(Vec3::new(1.0, 0.0, 2.0)), 1e-6), "{rebuilt:?}");
        let d = DMat4::from_scale_rotation_translation(
            DVec3::new(1.0, 2.0, 3.0),
            DQuat::from_rotation_y(0.5),
            DVec3::new(4.0, 5.0, 6.0),
        );
        let (s, r, t) = d.to_scale_rotation_translation();
        assert!(s.abs_diff_eq(DVec3::new(1.0, 2.0, 3.0), 1e-14) && t == DVec3::new(4.0, 5.0, 6.0));
        assert!(r.abs_diff_eq(DQuat::from_rotation_y(0.5), 1e-14));
    }

    #[test]
    fn dual_quaternions_are_rigid_transforms() {
        let mut rng = Rng::new(0x5A);
        for _ in 0..3000 {
            let (r1, t1) = (rand_quat(&mut rng), rand_vec3(&mut rng, 3.0));
            let (r2, t2) = (rand_quat(&mut rng), rand_vec3(&mut rng, 3.0));
            let (a, b) = (DualQuat::from_rotation_translation(r1, t1), DualQuat::from_rotation_translation(r2, t2));
            let p = rand_vec3(&mut rng, 2.0);
            assert!(a.transform_point3(p).abs_diff_eq(r1 * p + t1, 1e-4));
            assert_eq!(a.transform_vector3(p), r1 * p);
            assert!(a.translation().abs_diff_eq(t1, 1e-5));
            assert_eq!(a.rotation(), r1);
            assert!((a * b).transform_point3(p).abs_diff_eq(a.transform_point3(b.transform_point3(p)), 2e-4));
            assert!(a.inverse().transform_point3(a.transform_point3(p)).abs_diff_eq(p, 2e-4));
            assert!(a.to_mat4().abs_diff_eq(Mat4::from_rotation_translation(r1, t1), 1e-5));
            // Unit dual quaternions satisfy |real| = 1 and real · dual = 0, and products keep it.
            let ab = (a * b).normalize();
            assert!((ab.real.length() - 1.0).abs() < 1e-5 && ab.real.dot(ab.dual).abs() < 1e-4);
            assert!(ab.transform_point3(p).abs_diff_eq((a * b).transform_point3(p), 2e-4));
        }
        assert_eq!(DualQuat::default(), DualQuat::IDENTITY);
        assert_eq!(DualQuat::IDENTITY.transform_point3(Vec3::new(1.0, 2.0, 3.0)), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(DualQuat::ZERO.normalize(), DualQuat::IDENTITY);
        assert_eq!((DualQuat::IDENTITY * f32::NAN).normalize(), DualQuat::IDENTITY);
    }

    #[test]
    fn dual_quaternion_blend_is_normalised_and_hemisphere_safe() {
        let mut rng = Rng::new(0x5B);
        for _ in 0..2000 {
            let a = DualQuat::from_rotation_translation(rand_quat(&mut rng), rand_vec3(&mut rng, 2.0));
            let p = rand_vec3(&mut rng, 1.0);
            // Blending a transform with itself, in either representation, gives it back.
            let antipode = DualQuat { real: -a.real, dual: -a.dual };
            let w = rng.range_f32(0.05, 0.95);
            let same = DualQuat::blend(&[a, antipode, a], &[w, 1.0 - w, 0.5]);
            assert!(same.transform_point3(p).abs_diff_eq(a.transform_point3(p), 2e-4));
            // A blend is always a rigid transform: distances are preserved.
            let b = DualQuat::from_rotation_translation(rand_quat(&mut rng), rand_vec3(&mut rng, 2.0));
            let blended = DualQuat::blend(&[a, b], &[w, 1.0 - w]);
            let q = rand_vec3(&mut rng, 1.0);
            let (d0, d1) = (p.distance(q), blended.transform_point3(p).distance(blended.transform_point3(q)));
            assert!((d0 - d1).abs() < 1e-4);
            assert!((blended.real.length() - 1.0).abs() < 1e-5 && blended.real.dot(blended.dual).abs() < 1e-4);
            // Weights 1 and 0 select an input.
            let first = DualQuat::blend(&[a, b], &[1.0, 0.0]);
            assert!(first.transform_point3(p).abs_diff_eq(a.transform_point3(p), 2e-4));
        }
        // Two pure translations blend to the weighted mean translation.
        let a = DualQuat::from_rotation_translation(Quat::IDENTITY, Vec3::new(2.0, 0.0, 0.0));
        let b = DualQuat::from_rotation_translation(Quat::IDENTITY, Vec3::new(0.0, 4.0, 0.0));
        assert!(DualQuat::blend(&[a, b], &[0.25, 0.75]).translation().abs_diff_eq(Vec3::new(0.5, 3.0, 0.0), 1e-6));
        // Two rotations about one axis blend to the rotation halfway.
        let a = DualQuat::from_rotation_translation(Quat::from_rotation_z(0.2), Vec3::ZERO);
        let b = DualQuat::from_rotation_translation(Quat::from_rotation_z(1.0), Vec3::ZERO);
        assert!(DualQuat::blend(&[a, b], &[0.5, 0.5]).real.abs_diff_eq(Quat::from_rotation_z(0.6), 1e-6));
        assert_eq!(DualQuat::blend(&[], &[]), DualQuat::IDENTITY);
        assert_eq!(DualQuat::blend(&[a, b], &[]), DualQuat::IDENTITY);
        assert_eq!(DualQuat::blend(&[a, a], &[1.0, -1.0]), DualQuat::IDENTITY, "weights cancel");
        assert_eq!(DualQuat::blend(&[a], &[1.0, 5.0]), a.normalize());
    }

    #[test]
    fn transform_matches_its_matrix() {
        let mut rng = Rng::new(0x5C);
        for _ in 0..3000 {
            let s = Vec3::new(rng.range_f32(0.3, 2.0), rng.range_f32(0.3, 2.0), rng.range_f32(0.3, 2.0));
            let a = Transform::new(rand_vec3(&mut rng, 4.0), rand_quat(&mut rng), s);
            let (p, v) = (rand_vec3(&mut rng, 2.0), rand_vec3(&mut rng, 2.0));
            let m = a.to_mat4();
            assert!(a.transform_point3(p).abs_diff_eq(m.transform_point3(p), 1e-4));
            assert!(a.transform_vector3(v).abs_diff_eq(m.transform_vector3(v), 1e-4));
            // The point-wise inverse is exact for any scale.
            assert!(a.inverse_transform_point3(a.transform_point3(p)).abs_diff_eq(p, 1e-4));
            assert!(a.inverse_transform_vector3(a.transform_vector3(v)).abs_diff_eq(v, 1e-4));
            let back = Transform::from_mat4(m);
            assert!(back.translation == a.translation && back.scale.abs_diff_eq(a.scale, 1e-4));
            assert!(same_rotation(back.rotation, a.rotation, 1e-4));
            // With a uniform parent scale, composition and inversion agree with the matrices.
            let parent = Transform::new(rand_vec3(&mut rng, 4.0), rand_quat(&mut rng), Vec3::splat(s.x));
            assert!((parent * a).to_mat4().abs_diff_eq(parent.to_mat4() * m, 2e-4));
            assert_eq!(parent * a, parent.mul_transform(a));
            let inv = parent.inverse().expect("non-zero scale");
            assert!(inv.to_mat4().abs_diff_eq(parent.to_mat4().inverse().expect("invertible"), 2e-4));
            assert!((inv * parent).to_mat4().abs_diff_eq(Mat4::IDENTITY, 2e-4));
            // Interpolation hits both ends and stays a valid transform between them.
            assert!(a.interpolate(parent, 0.0).to_mat4().abs_diff_eq(m, 1e-5));
            assert!(a.interpolate(parent, 1.0).to_mat4().abs_diff_eq(parent.to_mat4(), 1e-4));
            let mid = a.interpolate(parent, 0.5);
            assert!(mid.rotation.is_normalized() && mid.translation == a.translation.lerp(parent.translation, 0.5));
            let dq = a.to_dual_quat();
            assert!(dq.transform_point3(p).abs_diff_eq(a.rotation * p + a.translation, 1e-4));
        }
        assert_eq!(Transform::default(), Transform::IDENTITY);
        assert_eq!(Transform::IDENTITY.to_mat4(), Mat4::IDENTITY);
        assert_eq!(Transform::from_translation(Vec3::X).transform_point3(Vec3::Y), Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(Transform::from_scale(Vec3::splat(2.0)).transform_point3(Vec3::Y), Vec3::Y * 2.0);
        assert!(Transform::from_rotation(Quat::from_rotation_z(FRAC_PI_4)).transform_point3(Vec3::X).y > 0.7);
        assert_eq!(Transform::from_rotation_translation(Quat::IDENTITY, Vec3::Z).translation, Vec3::Z);
        assert_eq!(Transform::from_scale(Vec3::new(1.0, 0.0, 1.0)).inverse(), None);
        assert_eq!(Transform::from_translation(Vec3::NAN).inverse(), None);
    }
}
