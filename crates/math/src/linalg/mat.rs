//! Matrices: `Mat2`, `Mat3`, `Mat4` (`f32`) and `DMat3`, `DMat4` (`f64`), with the view and projection
//! constructors the renderer uses.
//!
//! Storage is column-major: `cols[c]` is column `c`, and a point transforms as `M * p`, so
//! `(A * B) * p` applies `B` first. All types are `#[repr(C)]` and padding-free; the layout is asserted
//! at compile time because `humanart-gpu` uploads them as they are.
//!
//! The world is right-handed with Y up. A view matrix looks down its own `-Z`. Projections target wgpu
//! clip space with REVERSED depth: `x`, `y` in `[-1, 1]` (Y up), `z` in `[0, 1]` with `1` at the near
//! plane and `0` at the far plane (or at infinity).
//!
//! Nothing here panics on bad numbers: a singular matrix has no inverse (`None`), and constructors
//! given degenerate parameters (zero-size frustum, zero aspect) return a matrix with non-finite
//! elements that [`Mat4::is_finite`] detects.

use core::ops::{Add, Mul, MulAssign, Neg, Sub};

use super::vec::{DVec2, DVec3, DVec4, Vec2, Vec3, Vec4};

/// The layout assertion, constants, constructors and operators every square matrix type shares.
macro_rules! matrix_common {
    ($M:ident, $V:ident, $T:ident, $n:literal, $mul_vec:ident, [$f0:ident = $i0:tt $(, $f:ident = $i:tt)*]) => {
        // The layout is a contract with `humanart-gpu` (`Pod`): n*n scalars, no padding.
        const _: () = assert!(size_of::<$M>() == $n * $n * size_of::<$T>() && align_of::<$M>() == align_of::<$T>());

        impl $M {
            /// Every element zero.
            pub const ZERO: Self = Self { cols: [$V::ZERO; $n] };

            /// A matrix from its columns.
            #[inline]
            pub const fn from_cols($f0: $V $(, $f: $V)*) -> Self {
                Self { cols: [$f0 $(, $f)*] }
            }

            /// A matrix from an array of columns (`a[column][row]`).
            #[inline]
            pub const fn from_cols_array_2d(a: [[$T; $n]; $n]) -> Self {
                Self { cols: [$V::from_array(a[$i0]) $(, $V::from_array(a[$i]))*] }
            }

            /// The matrix as an array of columns (`a[column][row]`).
            #[inline]
            pub const fn to_cols_array_2d(self) -> [[$T; $n]; $n] {
                [self.cols[$i0].to_array() $(, self.cols[$i].to_array())*]
            }

            /// Column `index`. Like an array, an index past the last column panics.
            #[inline]
            pub fn col(self, index: usize) -> $V {
                self.cols[index]
            }

            /// Row `index`. Like an array, an index past the last row panics.
            #[inline]
            pub fn row(self, index: usize) -> $V {
                $V::new(self.cols[$i0][index] $(, self.cols[$i][index])*)
            }

            /// The transpose: rows become columns.
            #[inline]
            pub fn transpose(self) -> Self {
                Self::from_cols(self.row($i0) $(, self.row($i))*)
            }

            /// The product `self * v` with `v` as a column vector.
            #[inline]
            pub fn $mul_vec(self, v: $V) -> $V {
                self.cols[$i0] * v.$f0 $(+ self.cols[$i] * v.$f)*
            }

            /// Whether every element is finite.
            #[inline]
            pub fn is_finite(self) -> bool {
                self.cols[$i0].is_finite() $(&& self.cols[$i].is_finite())*
            }

            /// Whether any element is NaN.
            #[inline]
            pub fn is_nan(self) -> bool {
                self.cols[$i0].is_nan() $(|| self.cols[$i].is_nan())*
            }

            /// Whether every element is within `epsilon` of the matching element of `rhs`.
            #[inline]
            pub fn abs_diff_eq(self, rhs: Self, epsilon: $T) -> bool {
                self.cols[$i0].abs_diff_eq(rhs.cols[$i0], epsilon)
                    $(&& self.cols[$i].abs_diff_eq(rhs.cols[$i], epsilon))*
            }
        }

        /// The identity matrix.
        impl Default for $M {
            #[inline]
            fn default() -> Self {
                Self::IDENTITY
            }
        }

        impl Mul<$V> for $M {
            type Output = $V;
            #[inline]
            fn mul(self, rhs: $V) -> $V {
                self.$mul_vec(rhs)
            }
        }

        /// The matrix product: `(a * b) * v == a * (b * v)`.
        impl Mul for $M {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Self) -> Self {
                Self { cols: [self.$mul_vec(rhs.cols[$i0]) $(, self.$mul_vec(rhs.cols[$i]))*] }
            }
        }

        impl MulAssign for $M {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl Mul<$T> for $M {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: $T) -> Self {
                Self { cols: [self.cols[$i0] * rhs $(, self.cols[$i] * rhs)*] }
            }
        }

        impl Mul<$M> for $T {
            type Output = $M;
            #[inline]
            fn mul(self, rhs: $M) -> $M {
                rhs * self
            }
        }

        impl Add for $M {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                Self { cols: [self.cols[$i0] + rhs.cols[$i0] $(, self.cols[$i] + rhs.cols[$i])*] }
            }
        }

        impl Sub for $M {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self { cols: [self.cols[$i0] - rhs.cols[$i0] $(, self.cols[$i] - rhs.cols[$i])*] }
            }
        }

        impl Neg for $M {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self { cols: [-self.cols[$i0] $(, -self.cols[$i])*] }
            }
        }
    };
}

/// A 2×2 `f32` matrix, column-major: 2D rotations and scales, UV transforms. 16 bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Mat2 {
    /// The two columns.
    pub cols: [Vec2; 2],
}

matrix_common!(Mat2, Vec2, f32, 2, mul_vec2, [x = 0, y = 1]);

impl Mat2 {
    /// The identity.
    pub const IDENTITY: Self = Self::from_cols(Vec2::X, Vec2::Y);

    /// A matrix from four elements in column order: `[m00, m10, m01, m11]`.
    #[inline]
    pub const fn from_cols_array(a: [f32; 4]) -> Self {
        let [a0, a1, a2, a3] = a;
        Self::from_cols(Vec2::new(a0, a1), Vec2::new(a2, a3))
    }

    /// The four elements in column order.
    #[inline]
    pub const fn to_cols_array(self) -> [f32; 4] {
        let [x, y] = self.cols;
        [x.x, x.y, y.x, y.y]
    }

    /// The counter-clockwise rotation by `angle` radians.
    #[inline]
    pub fn from_angle(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::from_cols(Vec2::new(c, s), Vec2::new(-s, c))
    }

    /// The scale by `scale.x` along X and `scale.y` along Y.
    #[inline]
    pub const fn from_diagonal(scale: Vec2) -> Self {
        Self::from_cols(Vec2::new(scale.x, 0.0), Vec2::new(0.0, scale.y))
    }

    /// The determinant.
    #[inline]
    pub fn determinant(self) -> f32 {
        self.cols[0].perp_dot(self.cols[1])
    }

    /// The inverse, or `None` when the determinant is zero or the result is not finite.
    #[inline]
    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let [a, b] = self.cols;
        let inv = Self::from_cols(Vec2::new(b.y, -a.y), Vec2::new(-b.x, a.x)) * det.recip();
        inv.is_finite().then_some(inv)
    }
}

/// A 3×3 floating-point matrix type.
macro_rules! mat3_type {
    ($(#[$meta:meta])* $M3:ident, $V3:ident, $V2:ident, $T:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[repr(C)]
        pub struct $M3 {
            /// The three columns.
            pub cols: [$V3; 3],
        }

        matrix_common!($M3, $V3, $T, 3, mul_vec3, [x = 0, y = 1, z = 2]);

        impl $M3 {
            /// The identity.
            pub const IDENTITY: Self = Self::from_cols($V3::X, $V3::Y, $V3::Z);

            /// A matrix from nine elements in column order: the first three are column 0.
            #[inline]
            pub const fn from_cols_array(a: [$T; 9]) -> Self {
                let [a0, a1, a2, a3, a4, a5, a6, a7, a8] = a;
                Self::from_cols($V3::new(a0, a1, a2), $V3::new(a3, a4, a5), $V3::new(a6, a7, a8))
            }

            /// The nine elements in column order.
            #[inline]
            pub const fn to_cols_array(self) -> [$T; 9] {
                let [x, y, z] = self.cols;
                [x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z]
            }

            /// The scale by `scale.x`, `scale.y`, `scale.z` along the axes.
            #[inline]
            pub const fn from_diagonal(scale: $V3) -> Self {
                Self::from_cols(
                    $V3::new(scale.x, 0.0, 0.0),
                    $V3::new(0.0, scale.y, 0.0),
                    $V3::new(0.0, 0.0, scale.z),
                )
            }

            /// The rotation by `angle` radians about the unit vector `axis`, counter-clockwise when
            /// the axis points at the viewer (right-hand rule). `axis` must be normalised.
            #[inline]
            pub fn from_axis_angle(axis: $V3, angle: $T) -> Self {
                let (s, c) = angle.sin_cos();
                let t = 1.0 - c;
                let $V3 { x, y, z } = axis;
                Self::from_cols(
                    $V3::new(t * x * x + c, t * x * y + s * z, t * x * z - s * y),
                    $V3::new(t * x * y - s * z, t * y * y + c, t * y * z + s * x),
                    $V3::new(t * x * z + s * y, t * y * z - s * x, t * z * z + c),
                )
            }

            /// The rotation by `angle` radians about +X (takes +Y towards +Z).
            #[inline]
            pub fn from_rotation_x(angle: $T) -> Self {
                let (s, c) = angle.sin_cos();
                Self::from_cols($V3::X, $V3::new(0.0, c, s), $V3::new(0.0, -s, c))
            }

            /// The rotation by `angle` radians about +Y (takes +Z towards +X).
            #[inline]
            pub fn from_rotation_y(angle: $T) -> Self {
                let (s, c) = angle.sin_cos();
                Self::from_cols($V3::new(c, 0.0, -s), $V3::Y, $V3::new(s, 0.0, c))
            }

            /// The rotation by `angle` radians about +Z (takes +X towards +Y).
            #[inline]
            pub fn from_rotation_z(angle: $T) -> Self {
                let (s, c) = angle.sin_cos();
                Self::from_cols($V3::new(c, s, 0.0), $V3::new(-s, c, 0.0), $V3::Z)
            }

            /// The 2D affine transform (homogeneous 3×3) that translates by `translation`.
            #[inline]
            pub const fn from_translation_2d(translation: $V2) -> Self {
                Self::from_cols($V3::X, $V3::Y, $V3::new(translation.x, translation.y, 1.0))
            }

            /// The 2D affine transform that scales by `scale` about the origin.
            #[inline]
            pub const fn from_scale_2d(scale: $V2) -> Self {
                Self::from_diagonal($V3::new(scale.x, scale.y, 1.0))
            }

            /// The 2D affine transform that rotates counter-clockwise by `angle` radians about the
            /// origin (the same matrix as [`Self::from_rotation_z`]).
            #[inline]
            pub fn from_rotation_2d(angle: $T) -> Self {
                Self::from_rotation_z(angle)
            }

            /// Transforms a 2D point by this 2D affine transform (`w = 1`, no perspective divide).
            #[inline]
            pub fn transform_point2(self, p: $V2) -> $V2 {
                (self.cols[0] * p.x + self.cols[1] * p.y + self.cols[2]).truncate()
            }

            /// Transforms a 2D direction by this 2D affine transform (`w = 0`: no translation).
            #[inline]
            pub fn transform_vector2(self, v: $V2) -> $V2 {
                (self.cols[0] * v.x + self.cols[1] * v.y).truncate()
            }

            /// The determinant: the signed volume scale of the transform.
            #[inline]
            pub fn determinant(self) -> $T {
                self.cols[0].triple(self.cols[1], self.cols[2])
            }

            /// The cofactor matrix: `determinant * inverse-transpose`, defined for singular matrices
            /// too. Its columns are the cross products of the other two columns.
            #[inline]
            pub fn cofactor(self) -> Self {
                let [a, b, c] = self.cols;
                Self::from_cols(b.cross(c), c.cross(a), a.cross(b))
            }

            /// The inverse, or `None` when the determinant is zero or not finite, or the result is
            /// not finite.
            #[inline]
            pub fn inverse(self) -> Option<Self> {
                let det = self.determinant();
                if det == 0.0 || !det.is_finite() {
                    return None;
                }
                let inv = self.cofactor().transpose() * det.recip();
                inv.is_finite().then_some(inv)
            }

            /// The matrix that transforms normals when `self` transforms positions: the
            /// inverse-transpose, which keeps normals perpendicular under non-uniform scale and
            /// flips them correctly under mirroring. The result does not preserve length.
            ///
            /// Where `self` has no inverse the cofactor matrix is returned instead: it has the same
            /// direction wherever a normal is still defined.
            #[inline]
            pub fn normal_matrix(self) -> Self {
                let det = self.determinant();
                let cofactor = self.cofactor();
                if det == 0.0 || !det.is_finite() {
                    return cofactor;
                }
                let m = cofactor * det.recip();
                if m.is_finite() { m } else { cofactor }
            }

            /// A right-handed rotation from this matrix by Gram-Schmidt: column 0 keeps its
            /// direction, column 1 is made perpendicular to it, column 2 is their cross product.
            ///
            /// Never fails. A zero column is rebuilt from the other two where they allow it (so a
            /// rotation scaled to zero along one axis comes back as that rotation); what is still
            /// undetermined becomes +X or an arbitrary perpendicular.
            pub fn orthonormalize(self) -> Self {
                let [c0, c1, c2] = self.cols;
                let x = c0.try_normalize().unwrap_or_else(|| c1.cross(c2).normalize_or($V3::X));
                let y = match c1.reject_from(x).try_normalize() {
                    Some(y) => y,
                    None => match c2.reject_from(x).try_normalize() {
                        Some(z) => z.cross(x),
                        None => x.any_orthonormal_vector(),
                    },
                };
                Self::from_cols(x, y, x.cross(y))
            }
        }
    };
}

mat3_type!(
    /// A 3×3 `f32` matrix, column-major: rotations, normal matrices, 2D affine transforms. 36 bytes.
    Mat3, Vec3, Vec2, f32
);
mat3_type!(
    /// A 3×3 `f64` matrix, column-major, for numerics that need double precision.
    DMat3, DVec3, DVec2, f64
);

/// A 4×4 floating-point matrix type.
macro_rules! mat4_type {
    ($(#[$meta:meta])* $M4:ident, $M3:ident, $V4:ident, $V3:ident, $V2:ident, $T:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[repr(C)]
        pub struct $M4 {
            /// The four columns; `cols[3]` holds the translation of an affine transform.
            pub cols: [$V4; 4],
        }

        matrix_common!($M4, $V4, $T, 4, mul_vec4, [x = 0, y = 1, z = 2, w = 3]);

        impl $M4 {
            /// The identity.
            pub const IDENTITY: Self = Self::from_cols($V4::X, $V4::Y, $V4::Z, $V4::W);

            /// A matrix from sixteen elements in column order: the first four are column 0.
            #[inline]
            pub const fn from_cols_array(a: [$T; 16]) -> Self {
                let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15] = a;
                Self::from_cols(
                    $V4::new(a0, a1, a2, a3),
                    $V4::new(a4, a5, a6, a7),
                    $V4::new(a8, a9, a10, a11),
                    $V4::new(a12, a13, a14, a15),
                )
            }

            /// The sixteen elements in column order.
            #[inline]
            pub const fn to_cols_array(self) -> [$T; 16] {
                let [x, y, z, w] = self.cols;
                [x.x, x.y, x.z, x.w, y.x, y.y, y.z, y.w, z.x, z.y, z.z, z.w, w.x, w.y, w.z, w.w]
            }

            /// The affine transform with linear part `m` and no translation.
            #[inline]
            pub const fn from_mat3(m: $M3) -> Self {
                Self::from_mat3_translation(m, $V3::ZERO)
            }

            /// The affine transform `p -> m * p + translation`.
            #[inline]
            pub const fn from_mat3_translation(m: $M3, translation: $V3) -> Self {
                let [x, y, z] = m.cols;
                Self::from_cols(x.extend(0.0), y.extend(0.0), z.extend(0.0), translation.extend(1.0))
            }

            /// The translation by `translation`.
            #[inline]
            pub const fn from_translation(translation: $V3) -> Self {
                Self::from_mat3_translation($M3::IDENTITY, translation)
            }

            /// The scale by `scale` about the origin.
            #[inline]
            pub const fn from_scale(scale: $V3) -> Self {
                Self::from_mat3($M3::from_diagonal(scale))
            }

            /// The rotation by `angle` radians about the unit vector `axis` (right-hand rule).
            #[inline]
            pub fn from_axis_angle(axis: $V3, angle: $T) -> Self {
                Self::from_mat3($M3::from_axis_angle(axis, angle))
            }

            /// The rotation by `angle` radians about +X.
            #[inline]
            pub fn from_rotation_x(angle: $T) -> Self {
                Self::from_mat3($M3::from_rotation_x(angle))
            }

            /// The rotation by `angle` radians about +Y.
            #[inline]
            pub fn from_rotation_y(angle: $T) -> Self {
                Self::from_mat3($M3::from_rotation_y(angle))
            }

            /// The rotation by `angle` radians about +Z.
            #[inline]
            pub fn from_rotation_z(angle: $T) -> Self {
                Self::from_mat3($M3::from_rotation_z(angle))
            }

            /// The upper-left 3×3 block: the linear part of an affine transform.
            #[inline]
            pub const fn to_mat3(self) -> $M3 {
                let [x, y, z, _] = self.cols;
                $M3::from_cols(x.truncate(), y.truncate(), z.truncate())
            }

            /// The translation of an affine transform: the first three elements of column 3.
            #[inline]
            pub const fn translation(self) -> $V3 {
                self.cols[3].truncate()
            }

            /// Transforms a point by an affine transform (`w = 1`, no perspective divide). For a
            /// projection use [`Self::project_point3`].
            #[inline]
            pub fn transform_point3(self, p: $V3) -> $V3 {
                (self.cols[0] * p.x + self.cols[1] * p.y + self.cols[2] * p.z + self.cols[3]).truncate()
            }

            /// Transforms a direction (`w = 0`: the translation does not apply).
            #[inline]
            pub fn transform_vector3(self, v: $V3) -> $V3 {
                (self.cols[0] * v.x + self.cols[1] * v.y + self.cols[2] * v.z).truncate()
            }

            /// Transforms a point and divides by the resulting `w`: world or view space to normalised
            /// device coordinates through a projection, or back through its inverse. Unchecked: a
            /// point on the plane `w = 0` gives non-finite components.
            #[inline]
            pub fn project_point3(self, p: $V3) -> $V3 {
                (self.cols[0] * p.x + self.cols[1] * p.y + self.cols[2] * p.z + self.cols[3]).project()
            }

            /// Maps a point in normalised device coordinates (`z` is reversed depth: 1 near, 0 far)
            /// back through this view-projection matrix. `None` when the matrix is singular or the
            /// point maps to infinity (depth 0 under an infinite projection).
            ///
            /// Inverts the matrix on every call; to unproject many points, invert once and use
            /// [`Self::project_point3`] on the inverse.
            #[inline]
            pub fn unproject_point3(self, ndc: $V3) -> Option<$V3> {
                let p = self.inverse()?.project_point3(ndc);
                p.is_finite().then_some(p)
            }

            /// The matrix that transforms normals when `self` transforms positions: the
            /// inverse-transpose of the linear part (see the 3×3 `normal_matrix`).
            #[inline]
            pub fn normal_matrix(self) -> $M3 {
                self.to_mat3().normal_matrix()
            }

            /// The determinant.
            #[inline]
            pub fn determinant(self) -> $T {
                let [[m00, m01, m02, m03], [m10, m11, m12, m13], [m20, m21, m22, m23], [m30, m31, m32, m33]] =
                    self.to_cols_array_2d();
                let (s0, s1, s2) = (m00 * m11 - m10 * m01, m00 * m12 - m10 * m02, m00 * m13 - m10 * m03);
                let (s3, s4, s5) = (m01 * m12 - m11 * m02, m01 * m13 - m11 * m03, m02 * m13 - m12 * m03);
                let (c5, c4, c3) = (m22 * m33 - m32 * m23, m21 * m33 - m31 * m23, m21 * m32 - m31 * m22);
                let (c2, c1, c0) = (m20 * m33 - m30 * m23, m20 * m32 - m30 * m22, m20 * m31 - m30 * m21);
                s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0
            }

            /// The inverse of any invertible matrix (projections included), or `None` when the
            /// determinant is zero or not finite, or the result is not finite.
            pub fn inverse(self) -> Option<Self> {
                // Laplace expansion by 2×2 minors. The formula is its own transpose, so reading the
                // column arrays as rows and writing the result the same way is consistent.
                let [[m00, m01, m02, m03], [m10, m11, m12, m13], [m20, m21, m22, m23], [m30, m31, m32, m33]] =
                    self.to_cols_array_2d();
                let (s0, s1, s2) = (m00 * m11 - m10 * m01, m00 * m12 - m10 * m02, m00 * m13 - m10 * m03);
                let (s3, s4, s5) = (m01 * m12 - m11 * m02, m01 * m13 - m11 * m03, m02 * m13 - m12 * m03);
                let (c5, c4, c3) = (m22 * m33 - m32 * m23, m21 * m33 - m31 * m23, m21 * m32 - m31 * m22);
                let (c2, c1, c0) = (m20 * m33 - m30 * m23, m20 * m32 - m30 * m22, m20 * m31 - m30 * m21);
                let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
                if det == 0.0 || !det.is_finite() {
                    return None;
                }
                let d = det.recip();
                let inv = Self::from_cols_array_2d([
                    [
                        (m11 * c5 - m12 * c4 + m13 * c3) * d,
                        (-m01 * c5 + m02 * c4 - m03 * c3) * d,
                        (m31 * s5 - m32 * s4 + m33 * s3) * d,
                        (-m21 * s5 + m22 * s4 - m23 * s3) * d,
                    ],
                    [
                        (-m10 * c5 + m12 * c2 - m13 * c1) * d,
                        (m00 * c5 - m02 * c2 + m03 * c1) * d,
                        (-m30 * s5 + m32 * s2 - m33 * s1) * d,
                        (m20 * s5 - m22 * s2 + m23 * s1) * d,
                    ],
                    [
                        (m10 * c4 - m11 * c2 + m13 * c0) * d,
                        (-m00 * c4 + m01 * c2 - m03 * c0) * d,
                        (m30 * s4 - m31 * s2 + m33 * s0) * d,
                        (-m20 * s4 + m21 * s2 - m23 * s0) * d,
                    ],
                    [
                        (-m10 * c3 + m11 * c1 - m12 * c0) * d,
                        (m00 * c3 - m01 * c1 + m02 * c0) * d,
                        (-m30 * s3 + m31 * s1 - m32 * s0) * d,
                        (m20 * s3 - m21 * s1 + m22 * s0) * d,
                    ],
                ]);
                inv.is_finite().then_some(inv)
            }

            /// The inverse of an affine transform (bottom row `0, 0, 0, 1`, which is assumed, not
            /// checked): cheaper and more accurate than [`Self::inverse`]. `None` when the linear
            /// part is singular.
            #[inline]
            pub fn inverse_affine(self) -> Option<Self> {
                let inv = self.to_mat3().inverse()?;
                let m = Self::from_mat3_translation(inv, -(inv * self.translation()));
                m.is_finite().then_some(m)
            }

            /// The right-handed view matrix of a camera at `eye` looking along `dir` with `up`
            /// roughly upwards: world space to view space, where the camera looks down `-Z`, `+X` is
            /// right and `+Y` is up.
            ///
            /// `dir` and `up` need not be normalised. Never fails: a zero `dir` looks down world
            /// `-Z`, and an `up` parallel to `dir` is replaced by an arbitrary perpendicular.
            pub fn look_to_rh(eye: $V3, dir: $V3, up: $V3) -> Self {
                let f = dir.normalize_or(-$V3::Z);
                let s = f.cross(up).try_normalize().unwrap_or_else(|| f.any_orthonormal_vector());
                let u = s.cross(f);
                Self::from_cols(
                    $V4::new(s.x, u.x, -f.x, 0.0),
                    $V4::new(s.y, u.y, -f.y, 0.0),
                    $V4::new(s.z, u.z, -f.z, 0.0),
                    $V4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
                )
            }

            /// The right-handed view matrix of a camera at `eye` looking at `target`
            /// (see [`Self::look_to_rh`]).
            #[inline]
            pub fn look_at_rh(eye: $V3, target: $V3, up: $V3) -> Self {
                Self::look_to_rh(eye, target - eye, up)
            }

            /// A right-handed perspective projection to wgpu clip space with reversed depth: view
            /// depth `near` maps to `z = 1` and `far` to `z = 0`.
            ///
            /// `fov_y` is the full vertical field of view in radians, `aspect` is width / height,
            /// `near` and `far` are positive distances. Degenerate parameters (`aspect == 0`,
            /// `near == far`, `fov_y == 0`) give non-finite elements.
            #[inline]
            pub fn perspective_reversed_z(fov_y: $T, aspect: $T, near: $T, far: $T) -> Self {
                let f = (0.5 * fov_y).tan().recip();
                let a = near / (far - near);
                Self::from_cols(
                    $V4::new(f / aspect, 0.0, 0.0, 0.0),
                    $V4::new(0.0, f, 0.0, 0.0),
                    $V4::new(0.0, 0.0, a, -1.0),
                    $V4::new(0.0, 0.0, a * far, 0.0),
                )
            }

            /// [`Self::perspective_reversed_z`] with the far plane at infinity: view depth `near`
            /// maps to `z = 1` and depth tends to `0` with distance (`z = near / distance`). This is
            /// the projection the renderer uses: with a 32-bit float depth buffer it keeps precision
            /// nearly uniform over the whole range.
            #[inline]
            pub fn perspective_infinite_reversed_z(fov_y: $T, aspect: $T, near: $T) -> Self {
                let f = (0.5 * fov_y).tan().recip();
                Self::from_cols(
                    $V4::new(f / aspect, 0.0, 0.0, 0.0),
                    $V4::new(0.0, f, 0.0, 0.0),
                    $V4::new(0.0, 0.0, 0.0, -1.0),
                    $V4::new(0.0, 0.0, near, 0.0),
                )
            }

            /// A right-handed orthographic projection to wgpu clip space with reversed depth: the
            /// view-space box `[left, right] × [bottom, top]` maps to `[-1, 1]²`, view depth `near`
            /// to `z = 1` and `far` to `z = 0`. An empty box gives non-finite elements.
            #[inline]
            pub fn orthographic_reversed_z(left: $T, right: $T, bottom: $T, top: $T, near: $T, far: $T) -> Self {
                let (rw, rh, rd) = ((right - left).recip(), (top - bottom).recip(), (far - near).recip());
                Self::from_cols(
                    $V4::new(2.0 * rw, 0.0, 0.0, 0.0),
                    $V4::new(0.0, 2.0 * rh, 0.0, 0.0),
                    $V4::new(0.0, 0.0, rd, 0.0),
                    $V4::new(-(right + left) * rw, -(top + bottom) * rh, far * rd, 1.0),
                )
            }

            /// This projection with every projected point shifted by `jitter_ndc` in normalised
            /// device coordinates: the sub-pixel camera jitter of temporal anti-aliasing. Works for
            /// perspective and orthographic projections alike. Convert a pixel offset with
            /// [`pixel_offset_to_ndc`].
            #[inline]
            pub fn with_jitter(self, jitter_ndc: $V2) -> Self {
                // x_clip += jitter.x * w_clip, y_clip += jitter.y * w_clip: add the jitter-scaled
                // bottom row to the first two rows.
                let mut m = self;
                for c in &mut m.cols {
                    c.x += jitter_ndc.x * c.w;
                    c.y += jitter_ndc.y * c.w;
                }
                m
            }
        }
    };
}

mat4_type!(
    /// A 4×4 `f32` matrix, column-major: affine transforms and projections. 64 bytes.
    Mat4, Mat3, Vec4, Vec3, Vec2, f32
);
mat4_type!(
    /// A 4×4 `f64` matrix, column-major, for numerics that need double precision.
    DMat4, DMat3, DVec4, DVec3, DVec2, f64
);

/// Lossless widening and explicit narrowing between the `f32` and `f64` matrix of one dimension.
macro_rules! matrix_convert {
    ($M:ident, $D:ident, $to_d:ident, $to_m:ident, $vec_to_d:ident, $vec_to_m:ident, [$($i:tt),+]) => {
        impl $M {
            /// The same matrix in double precision (exact).
            #[inline]
            pub const fn $to_d(self) -> $D {
                $D { cols: [$(self.cols[$i].$vec_to_d()),+] }
            }
        }

        impl $D {
            /// The same matrix rounded to single precision.
            #[inline]
            pub const fn $to_m(self) -> $M {
                $M { cols: [$(self.cols[$i].$vec_to_m()),+] }
            }
        }

        impl From<$M> for $D {
            #[inline]
            fn from(m: $M) -> Self {
                m.$to_d()
            }
        }
    };
}

matrix_convert!(Mat3, DMat3, as_dmat3, as_mat3, as_dvec3, as_vec3, [0, 1, 2]);
matrix_convert!(Mat4, DMat4, as_dmat4, as_mat4, as_dvec4, as_vec4, [0, 1, 2, 3]);

/// Element `index` (from 1) of the Halton low-discrepancy sequence in base `base`, in `[0, 1)`.
///
/// Returns `0` for `index == 0` or a base below 2.
#[inline]
pub fn halton(index: u32, base: u32) -> f32 {
    if base < 2 {
        return 0.0;
    }
    let inv_base = 1.0 / f64::from(base);
    let (mut i, mut fraction, mut result) = (index, 1.0_f64, 0.0_f64);
    // `i` strictly decreases because `base >= 2`.
    while i > 0 {
        fraction *= inv_base;
        result += fraction * f64::from(i % base);
        i /= base;
    }
    // The f64 sum is below 1; rounding to f32 could reach it, so stay inside the interval.
    (result as f32).min(1.0 - f32::EPSILON / 2.0)
}

/// The sub-pixel jitter of temporal anti-aliasing for frame `frame`, in pixels in `[-0.5, 0.5)²`:
/// the Halton (2, 3) sequence, repeating every `period` frames (a `period` of 0 counts as 1).
#[inline]
pub fn taa_jitter_pixels(frame: u32, period: u32) -> Vec2 {
    let index = frame % period.max(1) + 1;
    Vec2::new(halton(index, 2) - 0.5, halton(index, 3) - 0.5)
}

/// Converts an offset in pixels (Y down) to an offset in normalised device coordinates (Y up) for a
/// viewport of `viewport` pixels: the argument of [`Mat4::with_jitter`]. A zero-size viewport gives
/// a zero offset.
#[inline]
pub fn pixel_offset_to_ndc(offset: Vec2, viewport: Vec2) -> Vec2 {
    if viewport.x > 0.0 && viewport.y > 0.0 {
        Vec2::new(2.0 * offset.x / viewport.x, -2.0 * offset.y / viewport.y)
    } else {
        Vec2::ZERO
    }
}

/// Converts a position in pixels (origin top-left, Y down) to normalised device coordinates
/// (`[-1, 1]²`, Y up). A zero-size viewport gives the centre, `(0, 0)`.
#[inline]
pub fn screen_to_ndc(pixel: Vec2, viewport: Vec2) -> Vec2 {
    if viewport.x > 0.0 && viewport.y > 0.0 {
        Vec2::new(2.0 * pixel.x / viewport.x - 1.0, 1.0 - 2.0 * pixel.y / viewport.y)
    } else {
        Vec2::ZERO
    }
}

/// Converts normalised device coordinates (`[-1, 1]²`, Y up) to a position in pixels (origin
/// top-left, Y down).
#[inline]
pub fn ndc_to_screen(ndc: Vec2, viewport: Vec2) -> Vec2 {
    Vec2::new((ndc.x * 0.5 + 0.5) * viewport.x, (0.5 - ndc.y * 0.5) * viewport.y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::{FRAC_PI_2, FRAC_PI_3, PI};
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

    /// A random well-conditioned affine transform: rotation, scale in [0.5, 2], translation.
    fn rand_affine(rng: &mut Rng) -> Mat4 {
        let r = Mat3::from_axis_angle(rand_unit(rng), rng.range_f32(-PI, PI));
        let s = Vec3::new(rng.range_f32(0.5, 2.0), rng.range_f32(0.5, 2.0), rng.range_f32(0.5, 2.0));
        Mat4::from_mat3_translation(r * Mat3::from_diagonal(s), rand_vec3(rng, 5.0))
    }

    fn rand_dmat4(rng: &mut Rng) -> DMat4 {
        let mut a = [0.0_f64; 16];
        for v in &mut a {
            *v = rng.range_f64(-1.0, 1.0);
        }
        DMat4::from_cols_array(a)
    }

    #[test]
    fn layouts_are_the_gpu_contract() {
        assert_eq!((size_of::<Mat2>(), align_of::<Mat2>()), (16, 4));
        assert_eq!((size_of::<Mat3>(), align_of::<Mat3>()), (36, 4));
        assert_eq!((size_of::<Mat4>(), align_of::<Mat4>()), (64, 4));
        assert_eq!((size_of::<DMat3>(), align_of::<DMat3>()), (72, align_of::<f64>()));
        assert_eq!((size_of::<DMat4>(), align_of::<DMat4>()), (128, align_of::<f64>()));
        assert_eq!(core::mem::offset_of!(Mat4, cols), 0);
    }

    #[test]
    fn arrays_are_column_major() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let m = Mat4::from_cols_array(a);
        assert_eq!(m.to_cols_array(), a);
        assert_eq!(m.cols[1], Vec4::new(5.0, 6.0, 7.0, 8.0));
        assert_eq!(m.col(3), Vec4::new(13.0, 14.0, 15.0, 16.0));
        assert_eq!(m.row(0), Vec4::new(1.0, 5.0, 9.0, 13.0));
        assert_eq!(m.to_cols_array_2d()[2], [9.0, 10.0, 11.0, 12.0]);
        assert_eq!(Mat4::from_cols_array_2d(m.to_cols_array_2d()), m);
        assert_eq!(m.transpose().row(1), m.col(1));
        assert_eq!(m.transpose().transpose(), m);
        assert_eq!(m.translation(), Vec3::new(13.0, 14.0, 15.0));
        let m3 = Mat3::from_cols_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(m3.to_cols_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(m3.row(2), Vec3::new(3.0, 6.0, 9.0));
        assert_eq!(m.to_mat3(), Mat3::from_cols_array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.0]));
        let m2 = Mat2::from_cols_array([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m2.to_cols_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m2.row(1), Vec2::new(2.0, 4.0));
        assert_eq!(Mat4::default(), Mat4::IDENTITY);
        assert_eq!(Mat4::ZERO.to_cols_array(), [0.0; 16]);
        assert_eq!(m3.as_dmat3().as_mat3(), m3);
        assert_eq!(DMat4::from(m).as_mat4(), m);
        assert!(Mat4::IDENTITY.is_finite() && !Mat4::IDENTITY.is_nan());
        assert!((Mat4::IDENTITY * f32::NAN).is_nan());
    }

    #[test]
    fn products_follow_m_times_p() {
        // (A * B) * p applies B first.
        let t = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let s = Mat4::from_scale(Vec3::new(2.0, 2.0, 2.0));
        let p = Vec3::new(1.0, 1.0, 1.0);
        assert_eq!((t * s).transform_point3(p), Vec3::new(3.0, 4.0, 5.0));
        assert_eq!((s * t).transform_point3(p), Vec3::new(4.0, 6.0, 8.0));
        assert_eq!((t * s).transform_vector3(p), Vec3::new(2.0, 2.0, 2.0));
        assert_eq!(t * Vec4::new(0.0, 0.0, 0.0, 1.0), Vec4::new(1.0, 2.0, 3.0, 1.0));
        assert_eq!(Mat2::from_diagonal(Vec2::new(2.0, 3.0)) * Vec2::ONE, Vec2::new(2.0, 3.0));
        let mut rng = Rng::new(0x3A7);
        for _ in 0..500 {
            let (a, b, c) = (rand_affine(&mut rng), rand_affine(&mut rng), rand_affine(&mut rng));
            let p = rand_vec3(&mut rng, 3.0);
            assert!(((a * b) * c).abs_diff_eq(a * (b * c), 2e-3));
            assert!((a * b).transform_point3(p).abs_diff_eq(a.transform_point3(b.transform_point3(p)), 2e-3));
            assert_eq!(a * Mat4::IDENTITY, a);
            assert_eq!(Mat4::IDENTITY * a, a);
            assert!((a + a).abs_diff_eq(a * 2.0, 0.0) && (a - a).abs_diff_eq(Mat4::ZERO, 0.0));
            assert_eq!(-a, a * -1.0);
            assert_eq!(2.0 * a, a * 2.0);
            let mut d = a;
            d *= b;
            assert_eq!(d, a * b);
            assert!((a * b).transpose().abs_diff_eq(b.transpose() * a.transpose(), 1e-4));
        }
    }

    #[test]
    fn rotations_are_right_handed() {
        assert!((Mat3::from_rotation_z(FRAC_PI_2) * Vec3::X).abs_diff_eq(Vec3::Y, 1e-6));
        assert!((Mat3::from_rotation_x(FRAC_PI_2) * Vec3::Y).abs_diff_eq(Vec3::Z, 1e-6));
        assert!((Mat3::from_rotation_y(FRAC_PI_2) * Vec3::Z).abs_diff_eq(Vec3::X, 1e-6));
        assert!((Mat2::from_angle(FRAC_PI_2) * Vec2::X).abs_diff_eq(Vec2::Y, 1e-6));
        let mut rng = Rng::new(21);
        for _ in 0..500 {
            let angle = rng.range_f32(-PI, PI);
            assert!(Mat3::from_axis_angle(Vec3::X, angle).abs_diff_eq(Mat3::from_rotation_x(angle), 1e-6));
            assert!(Mat3::from_axis_angle(Vec3::Y, angle).abs_diff_eq(Mat3::from_rotation_y(angle), 1e-6));
            assert!(Mat3::from_axis_angle(Vec3::Z, angle).abs_diff_eq(Mat3::from_rotation_z(angle), 1e-6));
            let axis = rand_unit(&mut rng);
            let r = Mat3::from_axis_angle(axis, angle);
            // Orthonormal, determinant +1, the axis is fixed, the inverse is the transpose.
            assert!((r * r.transpose()).abs_diff_eq(Mat3::IDENTITY, 1e-5));
            assert!((r.determinant() - 1.0).abs() < 1e-5);
            assert!((r * axis).abs_diff_eq(axis, 1e-5));
            assert!(r.inverse().is_some_and(|inv| inv.abs_diff_eq(r.transpose(), 1e-5)));
            // The rotation angle about the axis is `angle`, signed by the right-hand rule.
            let v = axis.any_orthonormal_vector();
            let w = r * v;
            assert!((v.dot(w) - angle.cos()).abs() < 1e-5);
            assert!((v.cross(w).dot(axis) - angle.sin()).abs() < 1e-5);
            assert_eq!(Mat4::from_axis_angle(axis, angle).to_mat3(), r);
        }
        assert_eq!(Mat4::from_rotation_x(0.3).to_mat3(), Mat3::from_rotation_x(0.3));
        assert_eq!(Mat4::from_rotation_y(0.3).to_mat3(), Mat3::from_rotation_y(0.3));
        assert_eq!(Mat4::from_rotation_z(0.3).to_mat3(), Mat3::from_rotation_z(0.3));
        let d = DMat3::from_axis_angle(DVec3::new(0.0, 0.6, 0.8), 1.1);
        assert!((d * d.transpose()).abs_diff_eq(DMat3::IDENTITY, 1e-15));
    }

    #[test]
    fn determinants_match_analytic_cases() {
        assert_eq!(Mat2::from_cols_array([1.0, 2.0, 3.0, 4.0]).determinant(), -2.0);
        assert_eq!(Mat3::from_diagonal(Vec3::new(2.0, 3.0, 4.0)).determinant(), 24.0);
        assert_eq!(Mat3::from_cols_array([2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 1.0, 1.0, 1.0]).determinant(), 0.0);
        assert_eq!(Mat3::from_cols_array([1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]).determinant(), 1.0);
        assert_eq!(Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0)).determinant(), 24.0);
        assert_eq!(Mat4::IDENTITY.determinant(), 1.0);
        let m =
            DMat4::from_cols_array([1.0, 3.0, 5.0, 9.0, 1.0, 3.0, 1.0, 7.0, 4.0, 3.0, 9.0, 7.0, 5.0, 2.0, 0.0, 9.0]);
        assert_eq!(m.determinant(), -376.0);
        assert_eq!(m.transpose().determinant(), -376.0);
        let mut rng = Rng::new(77);
        for _ in 0..500 {
            let (a, b) = (rand_dmat4(&mut rng), rand_dmat4(&mut rng));
            let (da, db) = (a.determinant(), b.determinant());
            assert!(((a * b).determinant() - da * db).abs() < 1e-12);
            let (a3, b3) = (a.to_mat3(), b.to_mat3());
            assert!(((a3 * b3).determinant() - a3.determinant() * b3.determinant()).abs() < 1e-13);
        }
    }

    #[test]
    fn inverses_invert_and_singular_matrices_have_none() {
        let mut rng = Rng::new(0xD00D);
        let mut tested = 0;
        for _ in 0..2000 {
            let m = rand_dmat4(&mut rng);
            if m.determinant().abs() < 1e-2 {
                assert!(m.inverse().is_none_or(DMat4::is_finite));
                continue;
            }
            tested += 1;
            let inv = m.inverse().expect("well-conditioned");
            assert!((m * inv).abs_diff_eq(DMat4::IDENTITY, 1e-10), "{m:?}");
            assert!((inv * m).abs_diff_eq(DMat4::IDENTITY, 1e-10), "{m:?}");
            assert!((inv.determinant() * m.determinant() - 1.0).abs() < 1e-9);
            let m3 = m.to_mat3();
            if m3.determinant().abs() > 1e-2 {
                let inv3 = m3.inverse().expect("well-conditioned");
                assert!((m3 * inv3).abs_diff_eq(DMat3::IDENTITY, 1e-11));
            }
        }
        assert!(tested > 1500);
        for _ in 0..1000 {
            let m = rand_affine(&mut rng);
            let inv = m.inverse().expect("affine");
            let fast = m.inverse_affine().expect("affine");
            assert!((m * inv).abs_diff_eq(Mat4::IDENTITY, 2e-4));
            assert!((m * fast).abs_diff_eq(Mat4::IDENTITY, 2e-4));
            assert!(inv.abs_diff_eq(fast, 2e-4));
            assert_eq!(fast.row(3), Vec4::W);
            let p = rand_vec3(&mut rng, 4.0);
            assert!(fast.transform_point3(m.transform_point3(p)).abs_diff_eq(p, 1e-3));
        }
        let m2 = Mat2::from_cols_array([4.0, 2.0, 7.0, 6.0]);
        let inv2 = m2.inverse().expect("det 10");
        assert!((m2 * inv2).abs_diff_eq(Mat2::IDENTITY, 1e-6));
        assert!(inv2.abs_diff_eq(Mat2::from_cols_array([0.6, -0.2, -0.7, 0.4]), 1e-6));
        // Singular, NaN and overflowing inputs have no inverse, and nothing panics.
        assert_eq!(Mat2::from_cols_array([1.0, 2.0, 2.0, 4.0]).inverse(), None);
        assert_eq!(Mat3::from_diagonal(Vec3::new(1.0, 0.0, 1.0)).inverse(), None);
        assert_eq!(Mat4::from_scale(Vec3::new(1.0, 1.0, 0.0)).inverse(), None);
        assert_eq!(Mat4::from_scale(Vec3::new(1.0, 1.0, 0.0)).inverse_affine(), None);
        assert_eq!(Mat4::ZERO.inverse(), None);
        assert_eq!((Mat4::IDENTITY * f32::NAN).inverse(), None);
        assert_eq!((Mat3::IDENTITY * f32::INFINITY).inverse(), None);
        assert_eq!((Mat3::IDENTITY * 1e-20).inverse(), None, "determinant underflows");
        assert_eq!((Mat4::IDENTITY * 1e30).inverse(), None, "determinant overflows");
        assert!((DMat3::IDENTITY * 1e-20).inverse().is_some());
    }

    #[test]
    fn normal_matrix_keeps_normals_perpendicular() {
        let mut rng = Rng::new(1234);
        for _ in 0..1000 {
            let m = rand_affine(&mut rng);
            let nm = m.normal_matrix();
            let n = rand_unit(&mut rng);
            let (t, b) = n.any_orthonormal_pair();
            let n2 = nm * n;
            assert!(n2.dot(m.transform_vector3(t)).abs() < 1e-4);
            assert!(n2.dot(m.transform_vector3(b)).abs() < 1e-4);
            // The normal stays on the side the surface faces (positive determinant here).
            assert!(n2.dot(m.transform_vector3(t).cross(m.transform_vector3(b))) > 0.0);
            assert!(nm.abs_diff_eq(m.to_mat3().inverse().expect("affine").transpose(), 1e-5));
        }
        // A mirror flips the normal with the surface: +X facing becomes -X facing.
        let mirror = Mat3::from_diagonal(Vec3::new(-1.0, 1.0, 1.0));
        assert_eq!(mirror.normal_matrix() * Vec3::X, -Vec3::X);
        // Singular: flattening along Y leaves the Y normal direction defined.
        let flat = Mat3::from_diagonal(Vec3::new(2.0, 0.0, 3.0));
        assert_eq!((flat.normal_matrix() * Vec3::Y).normalize(), Vec3::Y);
        assert_eq!(Mat3::IDENTITY.cofactor(), Mat3::IDENTITY);
    }

    #[test]
    fn orthonormalize_returns_a_rotation() {
        let mut rng = Rng::new(8);
        for _ in 0..500 {
            let m = rand_affine(&mut rng).to_mat3();
            let r = m.orthonormalize();
            assert!((r * r.transpose()).abs_diff_eq(Mat3::IDENTITY, 1e-5));
            assert!((r.determinant() - 1.0).abs() < 1e-5);
            assert!(r.cols[0].abs_diff_eq(m.cols[0].normalize(), 1e-6));
        }
        let r = Mat3::ZERO.orthonormalize();
        assert!((r * r.transpose()).abs_diff_eq(Mat3::IDENTITY, 1e-6));
        let r = Mat3::from_cols(Vec3::Y, Vec3::Y * 2.0, Vec3::ZERO).orthonormalize();
        assert!((r * r.transpose()).abs_diff_eq(Mat3::IDENTITY, 1e-6) && (r.determinant() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn affine_2d_helpers() {
        let m = Mat3::from_translation_2d(Vec2::new(5.0, 1.0))
            * Mat3::from_rotation_2d(FRAC_PI_2)
            * Mat3::from_scale_2d(Vec2::new(2.0, 3.0));
        assert!(m.transform_point2(Vec2::new(1.0, 1.0)).abs_diff_eq(Vec2::new(2.0, 3.0), 1e-5));
        assert!(m.transform_vector2(Vec2::new(1.0, 1.0)).abs_diff_eq(Vec2::new(-3.0, 2.0), 1e-5));
        let inv = m.inverse().expect("invertible");
        assert!(inv.transform_point2(m.transform_point2(Vec2::new(0.3, -0.8))).abs_diff_eq(Vec2::new(0.3, -0.8), 1e-5));
    }

    #[test]
    fn look_at_builds_a_right_handed_view() {
        let eye = Vec3::new(1.0, 2.0, 5.0);
        let target = Vec3::new(1.0, 2.0, -3.0);
        let view = Mat4::look_at_rh(eye, target, Vec3::Y);
        assert!(view.transform_point3(eye).abs_diff_eq(Vec3::ZERO, 1e-6));
        assert!(view.transform_point3(target).abs_diff_eq(Vec3::new(0.0, 0.0, -8.0), 1e-6));
        assert!(view.transform_vector3(Vec3::X).abs_diff_eq(Vec3::X, 1e-6), "world +X is to the right");
        assert!(view.transform_vector3(Vec3::Y).abs_diff_eq(Vec3::Y, 1e-6));
        let mut rng = Rng::new(404);
        for _ in 0..1000 {
            let (eye, target) = (rand_vec3(&mut rng, 10.0), rand_vec3(&mut rng, 10.0));
            let dist = eye.distance(target);
            if dist < 0.1 || (target - eye).normalize().dot(Vec3::Y).abs() > 0.99 {
                continue;
            }
            let view = Mat4::look_at_rh(eye, target, Vec3::Y);
            let r = view.to_mat3();
            assert!((r * r.transpose()).abs_diff_eq(Mat3::IDENTITY, 1e-5));
            assert!((r.determinant() - 1.0).abs() < 1e-5);
            assert!(view.transform_point3(eye).length() < 1e-4);
            assert!(view.transform_point3(target).abs_diff_eq(Vec3::new(0.0, 0.0, -dist), 1e-4));
            assert!(view.transform_vector3(Vec3::Y).y > 0.0, "up stays up");
            assert!(view.transform_vector3(Vec3::Y).x.abs() < 1e-5, "no roll");
        }
        // Degenerate inputs still give a rigid transform.
        for (dir, up) in [(Vec3::ZERO, Vec3::Y), (Vec3::Y, Vec3::Y), (-Vec3::Y * 3.0, Vec3::Y), (Vec3::X, Vec3::ZERO)] {
            let r = Mat4::look_to_rh(Vec3::ONE, dir, up).to_mat3();
            assert!((r * r.transpose()).abs_diff_eq(Mat3::IDENTITY, 1e-5), "{dir:?} {up:?}");
            assert!((r.determinant() - 1.0).abs() < 1e-5);
        }
        assert_eq!(Mat4::look_to_rh(Vec3::ZERO, Vec3::ZERO, Vec3::Y), Mat4::IDENTITY);
    }

    #[test]
    fn perspective_is_reversed_z_in_zero_to_one() {
        let (fov, aspect, near, far) = (FRAC_PI_3, 16.0 / 9.0, 0.1_f32, 100.0_f32);
        let proj = Mat4::perspective_reversed_z(fov, aspect, near, far);
        let inf = Mat4::perspective_infinite_reversed_z(fov, aspect, near);
        assert!((proj.project_point3(Vec3::new(0.0, 0.0, -near)).z - 1.0).abs() < 1e-6);
        assert!(proj.project_point3(Vec3::new(0.0, 0.0, -far)).z.abs() < 1e-6);
        assert!((inf.project_point3(Vec3::new(0.0, 0.0, -near)).z - 1.0).abs() < 1e-6);
        assert!((inf.project_point3(Vec3::new(0.0, 0.0, -25.0)).z - near / 25.0).abs() < 1e-7);
        assert!(inf.project_point3(Vec3::new(0.0, 0.0, -1.0e9)).z < 1e-9);
        // The frustum edges land on the clip-space edges, with +Y up and +X right.
        let t = (0.5 * fov).tan();
        for m in [proj, inf] {
            let corner = m.project_point3(Vec3::new(aspect * t * 7.0, t * 7.0, -7.0));
            assert!(corner.truncate().abs_diff_eq(Vec2::ONE, 1e-5));
            let corner = m.project_point3(Vec3::new(-aspect * t * 7.0, -t * 7.0, -7.0));
            assert!(corner.truncate().abs_diff_eq(-Vec2::ONE, 1e-5));
            assert_eq!(m.mul_vec4(Vec4::new(0.0, 0.0, -3.0, 1.0)).w, 3.0, "w is the view distance");
        }
        // Depth decreases monotonically with distance, and stays in [0, 1] inside the frustum.
        let mut last = 2.0;
        for i in 0..=1000 {
            let d = near + (far - near) * (i as f32 / 1000.0);
            let z = proj.project_point3(Vec3::new(0.0, 0.0, -d)).z;
            assert!((-1e-6..=1.0 + 1e-6).contains(&z) && z < last, "{d} -> {z}");
            last = z;
        }
        let d = DMat4::perspective_reversed_z(1.0, 1.5, 0.25, 50.0);
        assert!((d.project_point3(DVec3::new(0.0, 0.0, -0.25)).z - 1.0).abs() < 1e-15);
        assert!(d.project_point3(DVec3::new(0.0, 0.0, -50.0)).z.abs() < 1e-15);
        // The finite projection tends to the infinite one as the far plane recedes.
        assert!(Mat4::perspective_reversed_z(fov, aspect, near, 1e12).abs_diff_eq(inf, 1e-6));
        assert!(!Mat4::perspective_reversed_z(fov, 0.0, near, far).is_finite());
        assert!(!Mat4::perspective_reversed_z(fov, aspect, near, near).is_finite());
    }

    #[test]
    fn orthographic_is_reversed_z_in_zero_to_one() {
        let proj = Mat4::orthographic_reversed_z(-2.0, 4.0, -1.0, 3.0, 0.5, 10.0);
        assert!(proj.project_point3(Vec3::new(-2.0, -1.0, -0.5)).abs_diff_eq(Vec3::new(-1.0, -1.0, 1.0), 1e-6));
        assert!(proj.project_point3(Vec3::new(4.0, 3.0, -10.0)).abs_diff_eq(Vec3::new(1.0, 1.0, 0.0), 1e-6));
        assert!(proj.project_point3(Vec3::new(1.0, 1.0, -5.25)).abs_diff_eq(Vec3::new(0.0, 0.0, 0.5), 1e-6));
        assert_eq!(proj.row(3), Vec4::W);
        assert!(!Mat4::orthographic_reversed_z(1.0, 1.0, 0.0, 1.0, 0.0, 1.0).is_finite());
    }

    #[test]
    fn unprojection_round_trips() {
        let mut rng = Rng::new(0xCAFE);
        let viewport = Vec2::new(1920.0, 1080.0);
        for _ in 0..1000 {
            let eye = rand_vec3(&mut rng, 5.0);
            let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y).as_dmat4();
            let projs = [
                DMat4::perspective_reversed_z(1.0, 1.6, 0.1, 100.0),
                DMat4::perspective_infinite_reversed_z(1.0, 1.6, 0.1),
                DMat4::orthographic_reversed_z(-8.0, 8.0, -5.0, 5.0, 0.1, 100.0),
            ];
            for proj in projs {
                let view_proj = proj * view;
                let p = rand_vec3(&mut rng, 5.0).as_dvec3();
                let ndc = view_proj.project_point3(p);
                if !ndc.is_finite() || ndc.z <= 1e-3 {
                    continue;
                }
                let back = view_proj.unproject_point3(ndc).expect("invertible");
                assert!(back.abs_diff_eq(p, 1e-7), "{p:?} -> {ndc:?} -> {back:?}");
            }
            let px = Vec2::new(rng.range_f32(0.0, 1920.0), rng.range_f32(0.0, 1080.0));
            assert!(ndc_to_screen(screen_to_ndc(px, viewport), viewport).abs_diff_eq(px, 1e-3));
        }
        assert_eq!(screen_to_ndc(Vec2::ZERO, viewport), Vec2::new(-1.0, 1.0), "top-left");
        assert_eq!(screen_to_ndc(viewport, viewport), Vec2::new(1.0, -1.0), "bottom-right");
        assert_eq!(screen_to_ndc(Vec2::ONE, Vec2::ZERO), Vec2::ZERO);
        assert_eq!(Mat4::ZERO.unproject_point3(Vec3::ZERO), None);
        // Depth 0 under an infinite projection is a point at infinity.
        let inf = Mat4::perspective_infinite_reversed_z(1.0, 1.0, 0.1);
        assert_eq!(inf.unproject_point3(Vec3::new(0.2, 0.2, 0.0)), None);
        assert!(inf.unproject_point3(Vec3::new(0.0, 0.0, 1.0)).is_some_and(|p| p.abs_diff_eq(-Vec3::Z * 0.1, 1e-6)));
    }

    #[test]
    fn jitter_shifts_every_point_by_the_same_ndc_offset() {
        let mut rng = Rng::new(60);
        let viewport = Vec2::new(1280.0, 720.0);
        let projs = [
            Mat4::perspective_infinite_reversed_z(1.0, 16.0 / 9.0, 0.1),
            Mat4::perspective_reversed_z(1.0, 16.0 / 9.0, 0.1, 50.0),
            Mat4::orthographic_reversed_z(-4.0, 4.0, -2.0, 2.0, 0.1, 50.0),
        ];
        for frame in 0..64 {
            let px = taa_jitter_pixels(frame, 16);
            assert!((-0.5..0.5).contains(&px.x) && (-0.5..0.5).contains(&px.y));
            assert_eq!(px, taa_jitter_pixels(frame + 16, 16));
            let jitter = pixel_offset_to_ndc(px, viewport);
            for proj in projs {
                let p = Vec3::new(rng.range_f32(-1.0, 1.0), rng.range_f32(-1.0, 1.0), rng.range_f32(-20.0, -1.0));
                let (a, b) = (proj.project_point3(p), proj.with_jitter(jitter).project_point3(p));
                assert!((b - a).abs_diff_eq(jitter.extend(0.0), 1e-5));
                // In pixels the shift is the requested offset.
                let shift = ndc_to_screen(b.truncate(), viewport) - ndc_to_screen(a.truncate(), viewport);
                assert!(shift.abs_diff_eq(px, 2e-3), "{shift:?} vs {px:?}");
            }
        }
        assert_eq!(pixel_offset_to_ndc(Vec2::ONE, Vec2::ZERO), Vec2::ZERO);
        assert_eq!(taa_jitter_pixels(5, 0), taa_jitter_pixels(0, 1));
        let d = DMat4::perspective_infinite_reversed_z(1.0, 1.0, 0.1).with_jitter(DVec2::new(0.01, -0.02));
        let p = d.project_point3(DVec3::new(0.0, 0.0, -3.0));
        assert!(p.truncate().abs_diff_eq(DVec2::new(0.01, -0.02), 1e-15));
    }

    #[test]
    fn halton_matches_the_sequence() {
        let base2: Vec<f32> = (1..=7).map(|i| halton(i, 2)).collect();
        assert_eq!(base2, [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875]);
        assert!((halton(1, 3) - 1.0 / 3.0).abs() < 1e-7 && (halton(2, 3) - 2.0 / 3.0).abs() < 1e-7);
        assert!((halton(3, 3) - 1.0 / 9.0).abs() < 1e-7 && (halton(4, 3) - 4.0 / 9.0).abs() < 1e-7);
        assert_eq!((halton(0, 2), halton(5, 0), halton(5, 1)), (0.0, 0.0, 0.0));
        for i in [u32::MAX, u32::MAX - 1, 1 << 31] {
            assert!((0.0..1.0).contains(&halton(i, 2)) && (0.0..1.0).contains(&halton(i, 3)));
        }
    }
}
