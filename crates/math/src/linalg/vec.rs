//! Vectors: `f32` (`Vec2`, `Vec3`, `Vec4`), `f64` (`DVec2`, `DVec3`, `DVec4`) and integer
//! (`IVec2`, `IVec3`, `UVec2`, `UVec3`, `UVec4`).
//!
//! All are `Copy`, `#[repr(C)]` and padding-free: the field order is the memory order, which is what
//! `humanart-gpu` uploads. The `f32` and `f64` families are generated from the same macros and have the same
//! API. Operators are component-wise; `*` and `/` also take a scalar on the right, and `scalar * v`
//! works too.

use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

/// `a.m(b).m(c)...` over a list of expressions.
macro_rules! chain_method {
    ($m:ident; $a:expr) => { $a };
    ($m:ident; $a:expr, $b:expr $(, $rest:expr)*) => { chain_method!($m; $a.$m($b) $(, $rest)*) };
}

/// The struct, constructors, component-wise operators and min/max family shared by every vector type.
macro_rules! vector_base {
    (
        $(#[$meta:meta])*
        $V:ident, $T:ident, $n:literal, zero = $zero:literal, one = $one:literal,
        [$f0:ident = $i0:tt $(, $f:ident = $i:tt)*]
    ) => {
        $(#[$meta])*
        #[repr(C)]
        pub struct $V {
            #[doc = concat!("The `", stringify!($f0), "` component.")]
            pub $f0: $T,
            $(
                #[doc = concat!("The `", stringify!($f), "` component.")]
                pub $f: $T,
            )*
        }

        // The layout is a contract with `humanart-gpu` (`Pod`): no padding, the scalar's alignment.
        const _: () = assert!(size_of::<$V>() == $n * size_of::<$T>() && align_of::<$V>() == align_of::<$T>());

        impl $V {
            /// Every component zero.
            pub const ZERO: Self = Self::splat($zero);
            /// Every component one.
            pub const ONE: Self = Self::splat($one);

            /// A vector from its components.
            #[inline]
            pub const fn new($f0: $T $(, $f: $T)*) -> Self {
                Self { $f0 $(, $f)* }
            }

            /// A vector with every component equal to `v`.
            #[inline]
            pub const fn splat(v: $T) -> Self {
                Self { $f0: v $(, $f: v)* }
            }

            /// A vector from an array in component order.
            #[inline]
            pub const fn from_array(a: [$T; $n]) -> Self {
                let [$f0 $(, $f)*] = a;
                Self { $f0 $(, $f)* }
            }

            /// The components as an array.
            #[inline]
            pub const fn to_array(self) -> [$T; $n] {
                [self.$f0 $(, self.$f)*]
            }

            /// A vector from the first components of a slice, or `None` when the slice is too short.
            #[inline]
            pub fn from_slice(s: &[$T]) -> Option<Self> {
                match s {
                    [$f0 $(, $f)*, ..] => Some(Self { $f0: *$f0 $(, $f: *$f)* }),
                    _ => None,
                }
            }

            /// Component-wise minimum.
            #[inline]
            pub fn min(self, rhs: Self) -> Self {
                Self { $f0: self.$f0.min(rhs.$f0) $(, $f: self.$f.min(rhs.$f))* }
            }

            /// Component-wise maximum.
            #[inline]
            pub fn max(self, rhs: Self) -> Self {
                Self { $f0: self.$f0.max(rhs.$f0) $(, $f: self.$f.max(rhs.$f))* }
            }

            /// Component-wise clamp to `[lo, hi]`. Never panics: where `lo > hi` the result is `hi`.
            #[inline]
            pub fn clamp(self, lo: Self, hi: Self) -> Self {
                self.max(lo).min(hi)
            }

            /// The smallest component.
            #[inline]
            pub fn min_element(self) -> $T {
                chain_method!(min; self.$f0 $(, self.$f)*)
            }

            /// The largest component.
            #[inline]
            pub fn max_element(self) -> $T {
                chain_method!(max; self.$f0 $(, self.$f)*)
            }

            /// The sum of the components.
            #[inline]
            pub fn element_sum(self) -> $T {
                self.$f0 $(+ self.$f)*
            }

            /// The product of the components.
            #[inline]
            pub fn element_product(self) -> $T {
                self.$f0 $(* self.$f)*
            }
        }

        impl Add for $V {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                Self { $f0: self.$f0 + rhs.$f0 $(, $f: self.$f + rhs.$f)* }
            }
        }

        impl Sub for $V {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self { $f0: self.$f0 - rhs.$f0 $(, $f: self.$f - rhs.$f)* }
            }
        }

        impl Mul for $V {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                Self { $f0: self.$f0 * rhs.$f0 $(, $f: self.$f * rhs.$f)* }
            }
        }

        impl Div for $V {
            type Output = Self;
            #[inline]
            fn div(self, rhs: Self) -> Self {
                Self { $f0: self.$f0 / rhs.$f0 $(, $f: self.$f / rhs.$f)* }
            }
        }

        impl Mul<$T> for $V {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: $T) -> Self {
                Self { $f0: self.$f0 * rhs $(, $f: self.$f * rhs)* }
            }
        }

        impl Mul<$V> for $T {
            type Output = $V;
            #[inline]
            fn mul(self, rhs: $V) -> $V {
                $V { $f0: self * rhs.$f0 $(, $f: self * rhs.$f)* }
            }
        }

        impl Div<$T> for $V {
            type Output = Self;
            #[inline]
            fn div(self, rhs: $T) -> Self {
                Self { $f0: self.$f0 / rhs $(, $f: self.$f / rhs)* }
            }
        }

        impl AddAssign for $V {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl SubAssign for $V {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl MulAssign for $V {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl DivAssign for $V {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }

        impl MulAssign<$T> for $V {
            #[inline]
            fn mul_assign(&mut self, rhs: $T) {
                *self = *self * rhs;
            }
        }

        impl DivAssign<$T> for $V {
            #[inline]
            fn div_assign(&mut self, rhs: $T) {
                *self = *self / rhs;
            }
        }

        /// Component access by index, like an array: an index past the last component panics.
        impl Index<usize> for $V {
            type Output = $T;
            #[inline]
            fn index(&self, index: usize) -> &$T {
                match index {
                    $i0 => &self.$f0,
                    $($i => &self.$f,)*
                    _ => panic!("vector component index {index} out of range"),
                }
            }
        }

        /// Mutable component access by index, like an array: an index past the last component panics.
        impl IndexMut<usize> for $V {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut $T {
                match index {
                    $i0 => &mut self.$f0,
                    $($i => &mut self.$f,)*
                    _ => panic!("vector component index {index} out of range"),
                }
            }
        }

        impl From<[$T; $n]> for $V {
            #[inline]
            fn from(a: [$T; $n]) -> Self {
                Self::from_array(a)
            }
        }

        impl From<$V> for [$T; $n] {
            #[inline]
            fn from(v: $V) -> Self {
                v.to_array()
            }
        }

        impl Sum for $V {
            #[inline]
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::ZERO, |a, b| a + b)
            }
        }
    };
}

/// The methods every floating-point vector has, whatever its dimension.
macro_rules! float_vector {
    ($V:ident, $T:ident, [$f0:ident $(, $f:ident)*]) => {
        impl $V {
            /// Every component NaN.
            pub const NAN: Self = Self::splat($T::NAN);
            /// Every component positive infinity.
            pub const INFINITY: Self = Self::splat($T::INFINITY);
            /// Every component negative infinity.
            pub const NEG_INFINITY: Self = Self::splat($T::NEG_INFINITY);

            /// The dot product.
            #[inline]
            pub fn dot(self, rhs: Self) -> $T {
                self.$f0 * rhs.$f0 $(+ self.$f * rhs.$f)*
            }

            /// The squared Euclidean length.
            #[inline]
            pub fn length_squared(self) -> $T {
                self.dot(self)
            }

            /// The Euclidean length.
            #[inline]
            pub fn length(self) -> $T {
                self.length_squared().sqrt()
            }

            /// The distance to `rhs`.
            #[inline]
            pub fn distance(self, rhs: Self) -> $T {
                (self - rhs).length()
            }

            /// The squared distance to `rhs`.
            #[inline]
            pub fn distance_squared(self, rhs: Self) -> $T {
                (self - rhs).length_squared()
            }

            /// The vector divided by its length, unchecked: a zero vector gives NaN components and a
            /// vector whose squared length overflows gives zeros. Use [`Self::try_normalize`] or
            /// [`Self::normalize_or`] when the input is not known to be sound.
            #[inline]
            pub fn normalize(self) -> Self {
                self / self.length()
            }

            /// The unit vector in this direction, or `None` when the length is zero, not finite, or so
            /// small that its square underflows to zero.
            #[inline]
            pub fn try_normalize(self) -> Option<Self> {
                let len = self.length();
                if len > 0.0 && len.is_finite() { Some(self / len) } else { None }
            }

            /// The unit vector in this direction, or `fallback` where [`Self::try_normalize`] fails.
            #[inline]
            pub fn normalize_or(self, fallback: Self) -> Self {
                self.try_normalize().unwrap_or(fallback)
            }

            /// The unit vector in this direction, or zero where [`Self::try_normalize`] fails.
            #[inline]
            pub fn normalize_or_zero(self) -> Self {
                self.normalize_or(Self::ZERO)
            }

            /// Whether the squared length is within `1e-4` of one.
            #[inline]
            pub fn is_normalized(self) -> bool {
                (self.length_squared() - 1.0).abs() <= 1.0e-4
            }

            /// Linear interpolation: `self` at `t = 0`, `rhs` at `t = 1`; extrapolates outside.
            #[inline]
            pub fn lerp(self, rhs: Self, t: $T) -> Self {
                self + (rhs - self) * t
            }

            /// The point halfway to `rhs`.
            #[inline]
            pub fn midpoint(self, rhs: Self) -> Self {
                (self + rhs) * 0.5
            }

            /// Component-wise absolute value.
            #[inline]
            pub fn abs(self) -> Self {
                self.map($T::abs)
            }

            /// Component-wise floor.
            #[inline]
            pub fn floor(self) -> Self {
                self.map($T::floor)
            }

            /// Component-wise ceiling.
            #[inline]
            pub fn ceil(self) -> Self {
                self.map($T::ceil)
            }

            /// Component-wise rounding, halves away from zero.
            #[inline]
            pub fn round(self) -> Self {
                self.map($T::round)
            }

            /// Component-wise `x - floor(x)`, in `[0, 1)` for finite input.
            #[inline]
            pub fn fract(self) -> Self {
                self - self.floor()
            }

            /// Component-wise sign: `1` or `-1` by the sign bit, NaN for NaN.
            #[inline]
            pub fn signum(self) -> Self {
                self.map($T::signum)
            }

            /// Component-wise reciprocal.
            #[inline]
            pub fn recip(self) -> Self {
                self.map($T::recip)
            }

            /// Component-wise square root.
            #[inline]
            pub fn sqrt(self) -> Self {
                self.map($T::sqrt)
            }

            /// Component-wise power.
            #[inline]
            pub fn powf(self, n: $T) -> Self {
                self.map(|v| v.powf(n))
            }

            /// Component-wise `e^x`.
            #[inline]
            pub fn exp(self) -> Self {
                self.map($T::exp)
            }

            /// Component-wise clamp to `[0, 1]`; NaN components become `0`.
            #[inline]
            pub fn saturate(self) -> Self {
                self.max(Self::ZERO).min(Self::ONE)
            }

            /// Applies `f` to every component.
            #[inline]
            pub fn map(self, f: impl Fn($T) -> $T) -> Self {
                Self { $f0: f(self.$f0) $(, $f: f(self.$f))* }
            }

            /// Whether every component is finite.
            #[inline]
            pub fn is_finite(self) -> bool {
                self.$f0.is_finite() $(&& self.$f.is_finite())*
            }

            /// Whether any component is NaN.
            #[inline]
            pub fn is_nan(self) -> bool {
                self.$f0.is_nan() $(|| self.$f.is_nan())*
            }

            /// Whether every component is within `epsilon` of the matching component of `rhs`.
            #[inline]
            pub fn abs_diff_eq(self, rhs: Self, epsilon: $T) -> bool {
                (self.$f0 - rhs.$f0).abs() <= epsilon $(&& (self.$f - rhs.$f).abs() <= epsilon)*
            }

            /// The vector shortened to length `max` when it is longer; unchanged otherwise.
            #[inline]
            pub fn clamp_length_max(self, max: $T) -> Self {
                let len_sq = self.length_squared();
                if len_sq > max * max { self * (max / len_sq.sqrt()) } else { self }
            }

            /// The projection of `self` onto the direction of `onto` (which need not be unit length).
            /// A zero `onto` gives zero.
            #[inline]
            pub fn project_onto(self, onto: Self) -> Self {
                let d = onto.length_squared();
                if d > 0.0 { onto * (self.dot(onto) / d) } else { Self::ZERO }
            }

            /// The part of `self` perpendicular to `from`: `self - self.project_onto(from)`.
            #[inline]
            pub fn reject_from(self, from: Self) -> Self {
                self - self.project_onto(from)
            }

            /// Mirrors `self` about the plane with unit normal `normal`.
            #[inline]
            pub fn reflect(self, normal: Self) -> Self {
                self - normal * (2.0 * self.dot(normal))
            }

            /// The angle between the two directions in `[0, π]`; `0` when either vector is zero.
            #[inline]
            pub fn angle_between(self, rhs: Self) -> $T {
                let d = (self.length_squared() * rhs.length_squared()).sqrt();
                if d > 0.0 { (self.dot(rhs) / d).clamp(-1.0, 1.0).acos() } else { 0.0 }
            }
        }

        impl Neg for $V {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self { $f0: -self.$f0 $(, $f: -self.$f)* }
            }
        }
    };
}

/// A two-component floating-point vector type.
macro_rules! float_vec2 {
    ($(#[$meta:meta])* $V2:ident, $V3:ident, $T:ident) => {
        vector_base!(
            $(#[$meta])*
            #[derive(Clone, Copy, Debug, Default, PartialEq)]
            $V2, $T, 2, zero = 0.0, one = 1.0, [x = 0, y = 1]
        );
        float_vector!($V2, $T, [x, y]);

        impl $V2 {
            /// The unit vector along +X.
            pub const X: Self = Self::new(1.0, 0.0);
            /// The unit vector along +Y.
            pub const Y: Self = Self::new(0.0, 1.0);

            /// The vector rotated a quarter turn counter-clockwise: `(-y, x)`.
            #[inline]
            pub fn perp(self) -> Self {
                Self::new(-self.y, self.x)
            }

            /// The 2D cross product `self.x * rhs.y - self.y * rhs.x`: twice the signed area of the
            /// triangle `(0, self, rhs)`, positive when `rhs` is counter-clockwise from `self`.
            #[inline]
            pub fn perp_dot(self, rhs: Self) -> $T {
                self.x * rhs.y - self.y * rhs.x
            }

            /// The unit vector at `angle` radians counter-clockwise from +X.
            #[inline]
            pub fn from_angle(angle: $T) -> Self {
                let (s, c) = angle.sin_cos();
                Self::new(c, s)
            }

            /// The angle of this vector counter-clockwise from +X, in `(-π, π]`.
            #[inline]
            pub fn angle(self) -> $T {
                self.y.atan2(self.x)
            }

            /// The vector rotated counter-clockwise by `angle` radians.
            #[inline]
            pub fn rotate(self, angle: $T) -> Self {
                let (s, c) = angle.sin_cos();
                Self::new(self.x * c - self.y * s, self.x * s + self.y * c)
            }

            /// `(x, y, z)` with the given `z`.
            #[inline]
            pub const fn extend(self, z: $T) -> $V3 {
                $V3::new(self.x, self.y, z)
            }

            /// The swizzle `(y, x)`.
            #[inline]
            pub const fn yx(self) -> Self {
                Self::new(self.y, self.x)
            }
        }
    };
}

/// A three-component floating-point vector type.
macro_rules! float_vec3 {
    ($(#[$meta:meta])* $V3:ident, $V2:ident, $V4:ident, $T:ident) => {
        vector_base!(
            $(#[$meta])*
            #[derive(Clone, Copy, Debug, Default, PartialEq)]
            $V3, $T, 3, zero = 0.0, one = 1.0, [x = 0, y = 1, z = 2]
        );
        float_vector!($V3, $T, [x, y, z]);

        impl $V3 {
            /// The unit vector along +X.
            pub const X: Self = Self::new(1.0, 0.0, 0.0);
            /// The unit vector along +Y (up).
            pub const Y: Self = Self::new(0.0, 1.0, 0.0);
            /// The unit vector along +Z (the way the character faces).
            pub const Z: Self = Self::new(0.0, 0.0, 1.0);

            /// The right-handed cross product.
            #[inline]
            pub fn cross(self, rhs: Self) -> Self {
                Self::new(
                    self.y * rhs.z - self.z * rhs.y,
                    self.z * rhs.x - self.x * rhs.z,
                    self.x * rhs.y - self.y * rhs.x,
                )
            }

            /// The scalar triple product `self · (b × c)`: the signed volume of the parallelepiped.
            #[inline]
            pub fn triple(self, b: Self, c: Self) -> $T {
                self.dot(b.cross(c))
            }

            /// `(x, y, z, w)` with the given `w`.
            #[inline]
            pub const fn extend(self, w: $T) -> $V4 {
                $V4::new(self.x, self.y, self.z, w)
            }

            /// `(x, y)`, dropping `z`.
            #[inline]
            pub const fn truncate(self) -> $V2 {
                $V2::new(self.x, self.y)
            }

            /// The swizzle `(x, y)`.
            #[inline]
            pub const fn xy(self) -> $V2 {
                $V2::new(self.x, self.y)
            }

            /// The swizzle `(x, z)`.
            #[inline]
            pub const fn xz(self) -> $V2 {
                $V2::new(self.x, self.z)
            }

            /// The swizzle `(y, z)`.
            #[inline]
            pub const fn yz(self) -> $V2 {
                $V2::new(self.y, self.z)
            }

            /// The index (0, 1 or 2) of the largest component; the lowest index wins ties.
            #[inline]
            pub fn max_axis(self) -> usize {
                if self.x >= self.y && self.x >= self.z {
                    0
                } else if self.y >= self.z {
                    1
                } else {
                    2
                }
            }

            /// The index (0, 1 or 2) of the smallest component; the lowest index wins ties.
            #[inline]
            pub fn min_axis(self) -> usize {
                if self.x <= self.y && self.x <= self.z {
                    0
                } else if self.y <= self.z {
                    1
                } else {
                    2
                }
            }

            /// Two unit vectors `(t, b)` that complete the unit vector `self` to a right-handed
            /// orthonormal frame: `t × b == self`. `self` must be normalised.
            ///
            /// The branch-free construction of Duff et al., "Building an Orthonormal Basis,
            /// Revisited" (JCGT 2017): continuous everywhere except at `-Z`.
            #[inline]
            pub fn any_orthonormal_pair(self) -> (Self, Self) {
                let sign = (1.0 as $T).copysign(self.z);
                let a = -1.0 / (sign + self.z);
                let b = self.x * self.y * a;
                (
                    Self::new(1.0 + sign * self.x * self.x * a, sign * b, -sign * self.x),
                    Self::new(b, sign + self.y * self.y * a, -self.y),
                )
            }

            /// A unit vector perpendicular to the unit vector `self`.
            #[inline]
            pub fn any_orthonormal_vector(self) -> Self {
                self.any_orthonormal_pair().0
            }
        }
    };
}

/// A four-component floating-point vector type.
macro_rules! float_vec4 {
    ($(#[$meta:meta])* $V4:ident, $V3:ident, $V2:ident, $T:ident) => {
        vector_base!(
            $(#[$meta])*
            #[derive(Clone, Copy, Debug, Default, PartialEq)]
            $V4, $T, 4, zero = 0.0, one = 1.0, [x = 0, y = 1, z = 2, w = 3]
        );
        float_vector!($V4, $T, [x, y, z, w]);

        impl $V4 {
            /// The unit vector along +X.
            pub const X: Self = Self::new(1.0, 0.0, 0.0, 0.0);
            /// The unit vector along +Y.
            pub const Y: Self = Self::new(0.0, 1.0, 0.0, 0.0);
            /// The unit vector along +Z.
            pub const Z: Self = Self::new(0.0, 0.0, 1.0, 0.0);
            /// The unit vector along +W.
            pub const W: Self = Self::new(0.0, 0.0, 0.0, 1.0);

            /// `(x, y, z)`, dropping `w`.
            #[inline]
            pub const fn truncate(self) -> $V3 {
                $V3::new(self.x, self.y, self.z)
            }

            /// The swizzle `(x, y, z)`.
            #[inline]
            pub const fn xyz(self) -> $V3 {
                $V3::new(self.x, self.y, self.z)
            }

            /// The swizzle `(x, y)`.
            #[inline]
            pub const fn xy(self) -> $V2 {
                $V2::new(self.x, self.y)
            }

            /// The swizzle `(z, w)`.
            #[inline]
            pub const fn zw(self) -> $V2 {
                $V2::new(self.z, self.w)
            }

            /// `(x, y, z) / w`: the Cartesian point of a homogeneous one. Unchecked: `w == 0` gives
            /// non-finite components.
            #[inline]
            pub fn project(self) -> $V3 {
                self.truncate() / self.w
            }
        }
    };
}

float_vec2!(
    /// A 2D `f32` vector: UVs, screen positions, 2D offsets. 8 bytes.
    Vec2, Vec3, f32
);
float_vec3!(
    /// A 3D `f32` vector: positions, directions, linear colours. 12 bytes, no padding.
    Vec3, Vec2, Vec4, f32
);
float_vec4!(
    /// A 4D `f32` vector: homogeneous points, colours with alpha, matrix columns. 16 bytes.
    Vec4, Vec3, Vec2, f32
);
float_vec2!(
    /// A 2D `f64` vector, for numerics that need double precision.
    DVec2, DVec3, f64
);
float_vec3!(
    /// A 3D `f64` vector, for numerics that need double precision (fitting, registration, solvers).
    DVec3, DVec2, DVec4, f64
);
float_vec4!(
    /// A 4D `f64` vector; the column type of [`DMat4`](super::DMat4).
    DVec4, DVec3, DVec2, f64
);

/// Lossless widening and explicit narrowing between the `f32` and `f64` vector of one dimension.
macro_rules! float_convert {
    ($V:ident, $D:ident, $to_d:ident, $to_v:ident, [$($f:ident),+]) => {
        impl $V {
            /// The same vector in double precision (exact).
            #[inline]
            pub const fn $to_d(self) -> $D {
                $D { $($f: self.$f as f64),+ }
            }
        }

        impl $D {
            /// The same vector rounded to single precision.
            #[inline]
            pub const fn $to_v(self) -> $V {
                $V { $($f: self.$f as f32),+ }
            }
        }

        impl From<$V> for $D {
            #[inline]
            fn from(v: $V) -> Self {
                v.$to_d()
            }
        }
    };
}

float_convert!(Vec2, DVec2, as_dvec2, as_vec2, [x, y]);
float_convert!(Vec3, DVec3, as_dvec3, as_vec3, [x, y, z]);
float_convert!(Vec4, DVec4, as_dvec4, as_vec4, [x, y, z, w]);

vector_base!(
    /// A 2D `i32` vector: signed pixel and cell coordinates.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    IVec2, i32, 2, zero = 0, one = 1, [x = 0, y = 1]
);
vector_base!(
    /// A 3D `i32` vector: grid cell coordinates.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    IVec3, i32, 3, zero = 0, one = 1, [x = 0, y = 1, z = 2]
);
vector_base!(
    /// A 2D `u32` vector: texture and viewport sizes.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    UVec2, u32, 2, zero = 0, one = 1, [x = 0, y = 1]
);
vector_base!(
    /// A 3D `u32` vector: triangle indices, volume sizes.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    UVec3, u32, 3, zero = 0, one = 1, [x = 0, y = 1, z = 2]
);
vector_base!(
    /// A 4D `u32` vector: joint indices.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    UVec4, u32, 4, zero = 0, one = 1, [x = 0, y = 1, z = 2, w = 3]
);

/// Numeric casts between an integer vector and the `f32` vector of the same dimension.
macro_rules! int_convert {
    ($I:ident, $IT:ident, $V:ident, $to_v:ident, $to_i:ident, [$($f:ident),+]) => {
        impl $I {
            /// The components converted to `f32` (rounded to nearest above 2^24).
            #[inline]
            pub const fn $to_v(self) -> $V {
                $V { $($f: self.$f as f32),+ }
            }
        }

        impl $V {
            /// The components cast to integers: truncated toward zero, saturated at the integer
            /// range, NaN becomes `0`.
            #[inline]
            pub const fn $to_i(self) -> $I {
                $I { $($f: self.$f as $IT),+ }
            }
        }
    };
}

int_convert!(IVec2, i32, Vec2, as_vec2, as_ivec2, [x, y]);
int_convert!(IVec3, i32, Vec3, as_vec3, as_ivec3, [x, y, z]);
int_convert!(UVec2, u32, Vec2, as_vec2, as_uvec2, [x, y]);
int_convert!(UVec3, u32, Vec3, as_vec3, as_uvec3, [x, y, z]);
int_convert!(UVec4, u32, Vec4, as_vec4, as_uvec4, [x, y, z, w]);

impl Neg for IVec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

impl Neg for IVec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    fn rand3(rng: &mut Rng) -> Vec3 {
        Vec3::new(rng.range_f32(-10.0, 10.0), rng.range_f32(-10.0, 10.0), rng.range_f32(-10.0, 10.0))
    }

    #[test]
    fn layouts_are_the_gpu_contract() {
        assert_eq!((size_of::<Vec2>(), align_of::<Vec2>()), (8, 4));
        assert_eq!((size_of::<Vec3>(), align_of::<Vec3>()), (12, 4));
        assert_eq!((size_of::<Vec4>(), align_of::<Vec4>()), (16, 4));
        assert_eq!((size_of::<DVec2>(), align_of::<DVec2>()), (16, align_of::<f64>()));
        assert_eq!((size_of::<DVec3>(), align_of::<DVec3>()), (24, align_of::<f64>()));
        assert_eq!((size_of::<DVec4>(), align_of::<DVec4>()), (32, align_of::<f64>()));
        assert_eq!((size_of::<IVec2>(), align_of::<IVec2>()), (8, 4));
        assert_eq!((size_of::<IVec3>(), align_of::<IVec3>()), (12, 4));
        assert_eq!((size_of::<UVec2>(), align_of::<UVec2>()), (8, 4));
        assert_eq!((size_of::<UVec3>(), align_of::<UVec3>()), (12, 4));
        assert_eq!((size_of::<UVec4>(), align_of::<UVec4>()), (16, 4));
        // Field order is memory order.
        assert_eq!(core::mem::offset_of!(Vec3, x), 0);
        assert_eq!(core::mem::offset_of!(Vec3, y), 4);
        assert_eq!(core::mem::offset_of!(Vec3, z), 8);
        assert_eq!(core::mem::offset_of!(Vec4, w), 12);
        assert_eq!(core::mem::offset_of!(Vec2, y), 4);
        assert_eq!(core::mem::offset_of!(UVec4, w), 12);
    }

    #[test]
    fn operators_are_component_wise() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, -5.0, 0.5);
        assert_eq!(a + b, Vec3::new(5.0, -3.0, 3.5));
        assert_eq!(a - b, Vec3::new(-3.0, 7.0, 2.5));
        assert_eq!(a * b, Vec3::new(4.0, -10.0, 1.5));
        assert_eq!(a / b, Vec3::new(0.25, -0.4, 6.0));
        assert_eq!(a * 2.0, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(2.0 * a, a * 2.0);
        assert_eq!(a / 2.0, Vec3::new(0.5, 1.0, 1.5));
        assert_eq!(-a, Vec3::new(-1.0, -2.0, -3.0));
        let mut c = a;
        c += b;
        c -= a;
        assert_eq!(c, b);
        c *= 2.0;
        c /= 2.0;
        c *= a;
        c /= a;
        assert!(c.abs_diff_eq(b, 1e-6));
        assert_eq!([a, b, a].into_iter().sum::<Vec3>(), a * 2.0 + b);
        assert_eq!(Vec4::new(1.0, 2.0, 3.0, 4.0) * Vec4::splat(2.0), Vec4::new(2.0, 4.0, 6.0, 8.0));
        assert_eq!(DVec2::new(1.0, 2.0) + DVec2::ONE, DVec2::new(2.0, 3.0));
    }

    #[test]
    fn constants_arrays_and_indexing() {
        const C: Vec3 = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(C.to_array(), [1.0, 2.0, 3.0]);
        assert_eq!(Vec3::ZERO.to_array(), [0.0; 3]);
        assert_eq!(Vec3::ONE, Vec3::splat(1.0));
        assert_eq!(Vec3::X + Vec3::Y + Vec3::Z, Vec3::ONE);
        assert_eq!(Vec4::X + Vec4::Y + Vec4::Z + Vec4::W, Vec4::ONE);
        assert_eq!(Vec2::X + Vec2::Y, Vec2::ONE);
        assert!(Vec3::NAN.is_nan() && !Vec3::NAN.is_finite());
        assert!(!Vec3::INFINITY.is_finite() && !Vec3::NEG_INFINITY.is_finite());
        let v = Vec4::from_array([1.0, 2.0, 3.0, 4.0]);
        assert_eq!((v[0], v[1], v[2], v[3]), (1.0, 2.0, 3.0, 4.0));
        let mut m = v;
        m[2] = 9.0;
        assert_eq!(m.z, 9.0);
        assert_eq!(<[f32; 4]>::from(v), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(Vec4::from([1.0, 2.0, 3.0, 4.0]), v);
        assert_eq!(Vec3::from_slice(&[1.0, 2.0, 3.0, 4.0]), Some(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(Vec3::from_slice(&[1.0, 2.0]), None);
        assert_eq!(Vec2::from_slice(&[]), None);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn indexing_past_the_end_panics_like_an_array() {
        let v = Vec3::ONE;
        let _ = v[3];
    }

    #[test]
    fn dot_and_cross_identities() {
        assert_eq!(Vec3::X.cross(Vec3::Y), Vec3::Z);
        assert_eq!(Vec3::Y.cross(Vec3::Z), Vec3::X);
        assert_eq!(Vec3::Z.cross(Vec3::X), Vec3::Y);
        let mut rng = Rng::new(0xA11CE);
        for _ in 0..2000 {
            let (a, b, c) = (rand3(&mut rng), rand3(&mut rng), rand3(&mut rng));
            let n = a.cross(b);
            let scale = a.length() * b.length() + 1.0;
            assert!(n.dot(a).abs() <= 2e-4 * scale * a.length());
            assert!(n.dot(b).abs() <= 2e-4 * scale * b.length());
            // Lagrange: |a × b|² = |a|²|b|² − (a·b)².
            let lagrange = a.length_squared() * b.length_squared() - a.dot(b) * a.dot(b);
            assert!((n.length_squared() - lagrange).abs() <= 1e-3 * (1.0 + lagrange.abs()));
            assert!((a.cross(b) + b.cross(a)).length() <= 1e-4);
            assert!((a.triple(b, c) - b.triple(c, a)).abs() <= 2e-3 * (1.0 + a.triple(b, c).abs()));
            assert_eq!(a.dot(b), b.dot(a));
            assert_eq!(a.distance(b), (a - b).length());
            assert!((a.distance_squared(b) - a.distance(b).powi(2)).abs() <= 1e-3);
        }
        let (a, b) = (DVec3::new(1.0, 2.0, 3.0), DVec3::new(-4.0, 0.5, 2.0));
        assert!(a.cross(b).dot(a).abs() < 1e-12);
        assert_eq!(Vec2::new(1.0, 0.0).perp_dot(Vec2::new(0.0, 1.0)), 1.0);
        assert_eq!(Vec4::new(1.0, 2.0, 3.0, 4.0).dot(Vec4::new(4.0, 3.0, 2.0, 1.0)), 20.0);
    }

    #[test]
    fn normalisation_variants() {
        let v = Vec3::new(3.0, 0.0, 4.0);
        assert_eq!(v.length(), 5.0);
        assert!(v.normalize().abs_diff_eq(Vec3::new(0.6, 0.0, 0.8), 1e-6));
        assert!(v.normalize().is_normalized());
        assert!(!v.is_normalized());
        assert_eq!(Vec3::ZERO.try_normalize(), None);
        assert_eq!(Vec3::NAN.try_normalize(), None);
        assert_eq!(Vec3::splat(f32::INFINITY).try_normalize(), None);
        assert_eq!(Vec3::splat(1e30).try_normalize(), None, "squared length overflows");
        assert_eq!(Vec3::splat(1e-30).try_normalize(), None, "squared length underflows");
        assert!(Vec3::ZERO.normalize().is_nan());
        assert_eq!(Vec3::ZERO.normalize_or(Vec3::Y), Vec3::Y);
        assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
        assert_eq!(Vec2::new(0.0, -2.0).normalize_or_zero(), Vec2::new(0.0, -1.0));
        let mut rng = Rng::new(7);
        for _ in 0..1000 {
            let a = rand3(&mut rng);
            if let Some(n) = a.try_normalize() {
                assert!((n.length() - 1.0).abs() < 1e-5);
                assert!(n.cross(a).length() <= 1e-4 * a.length());
            }
        }
        assert!((DVec3::new(1.0, 1.0, 1.0).normalize().length() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn min_max_clamp_abs_and_rounding() {
        let a = Vec3::new(1.0, -2.0, 3.5);
        let b = Vec3::new(0.0, 4.0, 3.0);
        assert_eq!(a.min(b), Vec3::new(0.0, -2.0, 3.0));
        assert_eq!(a.max(b), Vec3::new(1.0, 4.0, 3.5));
        assert_eq!(a.clamp(Vec3::ZERO, Vec3::ONE), Vec3::new(1.0, 0.0, 1.0));
        assert_eq!(a.clamp(Vec3::ONE, Vec3::ZERO), Vec3::ZERO, "inverted bounds do not panic");
        assert_eq!(a.abs(), Vec3::new(1.0, 2.0, 3.5));
        assert_eq!(a.floor(), Vec3::new(1.0, -2.0, 3.0));
        assert_eq!(a.ceil(), Vec3::new(1.0, -2.0, 4.0));
        assert_eq!(Vec2::new(0.5, -1.5).round(), Vec2::new(1.0, -2.0));
        assert_eq!(Vec2::new(1.25, -0.25).fract(), Vec2::new(0.25, 0.75));
        assert_eq!(a.signum(), Vec3::new(1.0, -1.0, 1.0));
        assert_eq!(Vec2::new(2.0, 4.0).recip(), Vec2::new(0.5, 0.25));
        assert_eq!(Vec2::new(4.0, 9.0).sqrt(), Vec2::new(2.0, 3.0));
        assert_eq!(Vec2::new(2.0, 3.0).powf(2.0), Vec2::new(4.0, 9.0));
        assert!(Vec2::new(0.0, 1.0).exp().abs_diff_eq(Vec2::new(1.0, core::f32::consts::E), 1e-6));
        assert_eq!(Vec3::new(-1.0, 0.5, f32::NAN).saturate(), Vec3::new(0.0, 0.5, 0.0));
        assert_eq!((a.min_element(), a.max_element()), (-2.0, 3.5));
        assert_eq!((a.element_sum(), a.element_product()), (2.5, -7.0));
        assert_eq!((a.max_axis(), a.min_axis()), (2, 1));
        assert_eq!((Vec3::ONE.max_axis(), Vec3::ONE.min_axis()), (0, 0));
        assert_eq!(Vec3::new(0.0, 2.0, 1.0).max_axis(), 1);
        assert_eq!(a.map(|v| v * 2.0), a * 2.0);
    }

    #[test]
    fn interpolation_projection_and_reflection() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(3.0, 6.0, -1.0);
        assert_eq!(a.lerp(b, 0.0), a);
        assert_eq!(a.lerp(b, 1.0), b);
        assert_eq!(a.lerp(b, 0.5), a.midpoint(b));
        assert_eq!(a.midpoint(b), Vec3::new(2.0, 4.0, 1.0));
        assert_eq!(Vec3::new(2.0, 3.0, 4.0).project_onto(Vec3::X * 5.0), Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(Vec3::new(2.0, 3.0, 4.0).reject_from(Vec3::X * 5.0), Vec3::new(0.0, 3.0, 4.0));
        assert_eq!(a.project_onto(Vec3::ZERO), Vec3::ZERO);
        assert_eq!(Vec3::new(1.0, -1.0, 0.0).reflect(Vec3::Y), Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(Vec3::new(0.0, 10.0, 0.0).clamp_length_max(2.0), Vec3::new(0.0, 2.0, 0.0));
        assert_eq!(Vec3::new(0.0, 1.0, 0.0).clamp_length_max(2.0), Vec3::new(0.0, 1.0, 0.0));
        assert!((Vec3::X.angle_between(Vec3::Y) - core::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!((Vec3::X.angle_between(-Vec3::X * 3.0) - core::f32::consts::PI).abs() < 1e-6);
        assert_eq!(Vec3::X.angle_between(Vec3::ZERO), 0.0);
    }

    #[test]
    fn swizzles_extend_and_truncate() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.xyz(), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(v.truncate(), v.xyz());
        assert_eq!(v.xy(), Vec2::new(1.0, 2.0));
        assert_eq!(v.zw(), Vec2::new(3.0, 4.0));
        assert_eq!(v.xyz().extend(4.0), v);
        assert_eq!(v.xyz().truncate(), Vec2::new(1.0, 2.0));
        assert_eq!(v.xyz().xy(), Vec2::new(1.0, 2.0));
        assert_eq!(v.xyz().xz(), Vec2::new(1.0, 3.0));
        assert_eq!(v.xyz().yz(), Vec2::new(2.0, 3.0));
        assert_eq!(v.xy().yx(), Vec2::new(2.0, 1.0));
        assert_eq!(v.xy().extend(3.0), v.xyz());
        assert_eq!(Vec4::new(2.0, 4.0, 6.0, 2.0).project(), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn vec2_rotation_helpers() {
        use core::f32::consts::FRAC_PI_2;
        assert!(Vec2::X.rotate(FRAC_PI_2).abs_diff_eq(Vec2::Y, 1e-6));
        assert_eq!(Vec2::X.perp(), Vec2::Y);
        assert!(Vec2::from_angle(FRAC_PI_2).abs_diff_eq(Vec2::Y, 1e-6));
        assert!((Vec2::new(0.0, 3.0).angle() - FRAC_PI_2).abs() < 1e-6);
        let mut rng = Rng::new(3);
        for _ in 0..500 {
            let a = rng.range_f32(-3.0, 3.0);
            let v = Vec2::from_angle(a);
            assert!((v.angle() - a).abs() < 1e-5);
            assert!((v.rotate(-a) - Vec2::X).length() < 1e-5);
        }
    }

    #[test]
    fn orthonormal_pair_is_a_right_handed_frame() {
        let mut rng = Rng::new(99);
        let check = |n: Vec3| {
            let (t, b) = n.any_orthonormal_pair();
            assert!((t.length() - 1.0).abs() < 1e-5 && (b.length() - 1.0).abs() < 1e-5, "{n:?}");
            assert!(t.dot(b).abs() < 1e-5 && t.dot(n).abs() < 1e-5 && b.dot(n).abs() < 1e-5, "{n:?}");
            assert!(t.cross(b).abs_diff_eq(n, 1e-5), "{n:?}");
            assert_eq!(n.any_orthonormal_vector(), t);
        };
        for axis in [Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, -Vec3::Y, -Vec3::Z] {
            check(axis);
        }
        for _ in 0..2000 {
            if let Some(n) = rand3(&mut rng).try_normalize() {
                check(n);
            }
        }
        let (t, b) = DVec3::new(0.0, 0.6, 0.8).any_orthonormal_pair();
        assert!(t.cross(b).abs_diff_eq(DVec3::new(0.0, 0.6, 0.8), 1e-14));
    }

    #[test]
    fn precision_conversions() {
        let v = Vec3::new(0.1, -2.5, 1.0e10);
        assert_eq!(v.as_dvec3().as_vec3(), v);
        assert_eq!(DVec3::from(v), v.as_dvec3());
        assert_eq!(v.as_dvec3().x, f64::from(0.1_f32));
        assert_eq!(Vec2::new(1.5, 2.5).as_dvec2().as_vec2(), Vec2::new(1.5, 2.5));
        assert_eq!(Vec4::splat(0.25).as_dvec4().as_vec4(), Vec4::splat(0.25));
        assert_eq!(DVec4::from(Vec4::W), DVec4::W);
    }

    #[test]
    fn integer_vectors() {
        let a = IVec2::new(3, -4);
        assert_eq!(a + IVec2::ONE, IVec2::new(4, -3));
        assert_eq!(a * 2, IVec2::new(6, -8));
        assert_eq!(-a, IVec2::new(-3, 4));
        assert_eq!(-IVec3::new(1, -2, 3), IVec3::new(-1, 2, -3));
        assert_eq!(a.min(IVec2::ZERO), IVec2::new(0, -4));
        assert_eq!(a.max(IVec2::ZERO), IVec2::new(3, 0));
        assert_eq!(a.clamp(IVec2::splat(-1), IVec2::splat(1)), IVec2::new(1, -1));
        assert_eq!(a.as_vec2(), Vec2::new(3.0, -4.0));
        assert_eq!(Vec2::new(3.9, -4.9).as_ivec2(), a);
        assert_eq!(Vec2::new(-1.0, 7.5).as_uvec2(), UVec2::new(0, 7));
        assert_eq!(Vec2::new(f32::NAN, 1e20).as_ivec2(), IVec2::new(0, i32::MAX));
        assert_eq!(Vec3::new(1.5, -1.5, 0.0).floor().as_ivec3(), IVec3::new(1, -2, 0));
        let size = UVec2::new(1920, 1080);
        assert_eq!(size / 2, UVec2::new(960, 540));
        assert_eq!(size.as_vec2(), Vec2::new(1920.0, 1080.0));
        assert_eq!((size.min_element(), size.max_element(), size.element_product()), (1080, 1920, 2_073_600));
        let tri = UVec3::new(7, 8, 9);
        assert_eq!((tri[0], tri[1], tri[2]), (7, 8, 9));
        assert_eq!(tri.to_array(), [7, 8, 9]);
        assert_eq!(tri.as_vec3().as_uvec3(), tri);
        assert_eq!(IVec3::new(1, 2, 3).as_vec3(), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(UVec4::from_array([1, 2, 3, 4]).element_sum(), 10);
        assert_eq!(UVec4::new(1, 2, 3, 4).as_vec4().as_uvec4(), UVec4::new(1, 2, 3, 4));
        assert_eq!([UVec2::ONE, UVec2::ONE].into_iter().sum::<UVec2>(), UVec2::splat(2));
        let mut set = std::collections::HashSet::new();
        set.insert(IVec3::new(1, 2, 3));
        assert!(set.contains(&IVec3::new(1, 2, 3)));
    }
}
