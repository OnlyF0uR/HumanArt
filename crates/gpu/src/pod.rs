//! Plain-old-data casting for GPU buffers `[UNS-03]`.
//!
//! wgpu takes bytes; shaders read structs. This module is the one place in the crate where a typed
//! value is reinterpreted as bytes and back, and therefore the one place that contains `unsafe`
//! `[UNS-01]`. Everything it exports is a safe function; the only `unsafe` obligation that reaches a
//! user is implementing [`Pod`], which [`pod_struct!`](crate::pod_struct) does with static checks.
//!
//! Bytes are in the machine's byte order. Every supported target is little-endian (asserted below),
//! which is also what WGSL and every file format of the project use.
//!
//! The reference application uses `bytemuck` for this job; the project's dependency rule replaces
//! it with this module (ADR-001).

use core::fmt;
use core::mem::{align_of, size_of, size_of_val};

// GPU buffers and the readers in this module assume little-endian layout.
const _: () = assert!(cfg!(target_endian = "little"), "humanart-gpu supports little-endian targets only");

/// A type whose values can be viewed as bytes and created from any bytes.
///
/// # Safety
///
/// An implementor promises all of the following:
///
/// - the type is a primitive number, an array of `Pod`, or a `#[repr(C)]` (or `#[repr(transparent)]`)
///   struct whose fields are all `Pod`;
/// - it has **no padding bytes**, neither between fields nor at the end (the sum of the field sizes
///   equals `size_of::<Self>()`);
/// - **every bit pattern** is a valid value (so no `bool`, `char`, enums, references or pointers);
/// - it has no interior mutability and no `Drop`.
///
/// Implement it with [`pod_struct!`](crate::pod_struct), which checks size, alignment, padding and
/// the fields at compile time, rather than by hand.
pub unsafe trait Pod: Copy + 'static {}

/// Implements [`Pod`] for primitive number types.
macro_rules! impl_pod_primitive {
    ($($ty:ty),* $(,)?) => {
        $(
            // SAFETY: a primitive integer or float has no padding and every bit pattern is a value.
            unsafe impl Pod for $ty {}
        )*
    };
}

impl_pod_primitive!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

// SAFETY: an array is `N` elements laid out back to back with no padding between or after them,
// and each element is `Pod`.
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

/// Why bytes could not be turned into a typed value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PodError {
    /// The byte count is not what the target type needs.
    SizeMismatch {
        /// Bytes the target needs (for a slice: the element size the length must be a multiple of).
        expected: usize,
        /// Bytes that were given.
        actual: usize,
    },
    /// The bytes do not start at an address aligned for the target type.
    Misaligned {
        /// Alignment the target type requires.
        required: usize,
    },
    /// The target type has size zero, so a byte count says nothing about an element count.
    ZeroSized,
}

impl fmt::Display for PodError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeMismatch { expected, actual } => {
                write!(f, "{actual} bytes do not fit a type of {expected} bytes")
            }
            Self::Misaligned { required } => write!(f, "bytes are not aligned to {required}"),
            Self::ZeroSized => f.write_str("a zero-sized type cannot be cast from bytes"),
        }
    }
}

impl std::error::Error for PodError {}

/// The bytes of one value.
#[inline]
pub fn bytes_of<T: Pod>(value: &T) -> &[u8] {
    // SAFETY: `value` is a valid reference, so its `size_of::<T>()` bytes are readable for the
    // returned lifetime; `T: Pod` has no padding, so every one of them is initialised; `u8` has
    // alignment 1.
    unsafe { core::slice::from_raw_parts(core::ptr::from_ref(value).cast::<u8>(), size_of::<T>()) }
}

/// The bytes of one value, writable. Any bytes written are a valid `T` because `T` is [`Pod`].
#[inline]
pub fn bytes_of_mut<T: Pod>(value: &mut T) -> &mut [u8] {
    // SAFETY: as in `bytes_of`; the exclusive borrow of `value` is carried by the returned slice,
    // and `T: Pod` accepts every bit pattern, so no write can make `value` invalid.
    unsafe { core::slice::from_raw_parts_mut(core::ptr::from_mut(value).cast::<u8>(), size_of::<T>()) }
}

/// The bytes of a slice of values, in order.
#[inline]
pub fn slice_as_bytes<T: Pod>(slice: &[T]) -> &[u8] {
    // SAFETY: the slice's `size_of_val` bytes are one readable allocation; `T: Pod` has no padding,
    // so all of them are initialised; `u8` has alignment 1. A slice never exceeds `isize::MAX` bytes.
    unsafe { core::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), size_of_val(slice)) }
}

/// The bytes of a slice of values, writable.
#[inline]
pub fn slice_as_bytes_mut<T: Pod>(slice: &mut [T]) -> &mut [u8] {
    // SAFETY: as in `slice_as_bytes`; the exclusive borrow is carried over, and every bit pattern is
    // a valid `T`.
    unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<u8>(), size_of_val(slice)) }
}

/// A value whose bytes are all zero.
#[inline]
pub fn zeroed<T: Pod>() -> T {
    // SAFETY: `T: Pod` accepts every bit pattern, including all zeroes.
    unsafe { core::mem::zeroed() }
}

/// Reads one value from exactly `size_of::<T>()` bytes. The bytes need not be aligned.
pub fn from_bytes<T: Pod>(bytes: &[u8]) -> Result<T, PodError> {
    if bytes.len() != size_of::<T>() {
        return Err(PodError::SizeMismatch { expected: size_of::<T>(), actual: bytes.len() });
    }
    // SAFETY: `bytes` holds exactly `size_of::<T>()` readable bytes (checked above);
    // `read_unaligned` has no alignment requirement; every bit pattern is a valid `T`.
    Ok(unsafe { core::ptr::read_unaligned(bytes.as_ptr().cast::<T>()) })
}

/// Views bytes as a slice of values without copying.
///
/// Fails when `T` is zero-sized, when the length is not a multiple of the element size, or when the
/// bytes are not aligned for `T` (bytes from a mapped GPU buffer are aligned to
/// [`wgpu::MAP_ALIGNMENT`]; bytes from an arbitrary `Vec<u8>` may not be: use [`copy_from_bytes`]).
pub fn cast_slice<T: Pod>(bytes: &[u8]) -> Result<&[T], PodError> {
    if size_of::<T>() == 0 {
        return Err(PodError::ZeroSized);
    }
    if !bytes.len().is_multiple_of(size_of::<T>()) {
        return Err(PodError::SizeMismatch { expected: size_of::<T>(), actual: bytes.len() });
    }
    if bytes.is_empty() {
        // An empty byte slice may point anywhere (a literal `&[]` is aligned to 1 only).
        return Ok(&[]);
    }
    if !(bytes.as_ptr() as usize).is_multiple_of(align_of::<T>()) {
        return Err(PodError::Misaligned { required: align_of::<T>() });
    }
    // SAFETY: the pointer is aligned for `T` and the byte length is a whole number of elements
    // (both checked above); the memory is readable for the borrow of `bytes`; every bit pattern is a
    // valid `T`.
    Ok(unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), bytes.len() / size_of::<T>()) })
}

/// Copies bytes into typed values. The bytes need not be aligned; the lengths must match exactly.
pub fn copy_from_bytes<T: Pod>(bytes: &[u8], out: &mut [T]) -> Result<(), PodError> {
    let target = slice_as_bytes_mut(out);
    if target.len() != bytes.len() {
        return Err(PodError::SizeMismatch { expected: target.len(), actual: bytes.len() });
    }
    target.copy_from_slice(bytes);
    Ok(())
}

/// Implements [`Pod`] for `humanart-math` types made only of one scalar type, with a compile-time
/// check that each is exactly as large and as aligned as the array of scalars it is declared as.
macro_rules! impl_pod_math {
    ($($ty:ty => [$scalar:ty; $count:expr]),* $(,)?) => {
        $(
            const _: () = assert!(
                size_of::<$ty>() == size_of::<[$scalar; $count]>() && align_of::<$ty>() == align_of::<$scalar>(),
                "a humanart-math type is not the plain array of scalars it is declared as"
            );
            // SAFETY: `humanart-math` declares the type `#[repr(C)]` (or `#[repr(transparent)]`)
            // with fields that are all `$scalar` or arrays and structs of it, and asserts its layout;
            // the check above proves it holds exactly `$count` scalars with no padding, so every bit
            // pattern is a valid value. The type is `Copy`, has no `Drop` and no interior mutability.
            unsafe impl Pod for $ty {}
        )*
    };
}

// [UNS-03] The math types upload as they are: vectors, matrices (column-major), quaternions,
// dual quaternions, transforms and half floats. The math crate forbids unsafe code, so the impls
// live here.
impl_pod_math!(
    humanart_math::Vec2 => [f32; 2],
    humanart_math::Vec3 => [f32; 3],
    humanart_math::Vec4 => [f32; 4],
    humanart_math::DVec2 => [f64; 2],
    humanart_math::DVec3 => [f64; 3],
    humanart_math::DVec4 => [f64; 4],
    humanart_math::IVec2 => [i32; 2],
    humanart_math::IVec3 => [i32; 3],
    humanart_math::UVec2 => [u32; 2],
    humanart_math::UVec3 => [u32; 3],
    humanart_math::UVec4 => [u32; 4],
    humanart_math::Mat2 => [f32; 4],
    humanart_math::Mat3 => [f32; 9],
    humanart_math::Mat4 => [f32; 16],
    humanart_math::DMat3 => [f64; 9],
    humanart_math::DMat4 => [f64; 16],
    humanart_math::Quat => [f32; 4],
    humanart_math::DQuat => [f64; 4],
    humanart_math::DualQuat => [f32; 8],
    humanart_math::Transform => [f32; 10],
    humanart_math::F16 => [u16; 1],
);

/// Compiles only when `T` is [`Pod`]; [`pod_struct!`](crate::pod_struct) calls it for every field.
#[doc(hidden)]
pub const fn assert_pod<T: Pod>() {}

/// Declares a `#[repr(C)]` struct and implements [`Pod`](crate::pod::Pod) for it, with the size and
/// the alignment written out and checked at compile time `[UNS-03]`.
///
/// ```
/// humanart_gpu::pod_struct! {
///     /// Mirrors `struct Globals` in the shader.
///     #[derive(Debug, PartialEq)]
///     pub struct Globals(size = 32, align = 4) {
///         /// Physical pixels.
///         pub viewport: [f32; 2],
///         /// Physical pixels per logical pixel.
///         pub scale: f32,
///         /// Seconds.
///         pub time: f32,
///         /// Explicit tail padding, as WGSL would add.
///         pub pad: [f32; 4],
///     }
/// }
/// let g = Globals { viewport: [1.0, 2.0], scale: 1.0, time: 0.0, pad: [0.0; 4] };
/// assert_eq!(humanart_gpu::pod::bytes_of(&g).len(), 32);
/// ```
///
/// Compilation fails when: the declared size or alignment is not the real one; the fields do not
/// add up to the size (the struct would have padding: add an explicit `pad` field); a field type is
/// not `Pod`. The struct is always `Clone + Copy`; further derives go in the attributes.
///
/// The expansion contains one `unsafe impl`. It comes from this crate's macro, so a crate that
/// forbids unsafe code can still use it; the checks above are what make that impl sound.
#[macro_export]
macro_rules! pod_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident (size = $size:expr, align = $align:expr) {
            $(
                $(#[$field_meta:meta])*
                $field_vis:vis $field:ident : $field_ty:ty
            ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[repr(C)]
        #[derive(Clone, Copy)]
        $vis struct $name {
            $(
                $(#[$field_meta])*
                $field_vis $field: $field_ty,
            )*
        }

        const _: () = {
            assert!(::core::mem::size_of::<$name>() == $size, "pod_struct: the declared size is wrong");
            assert!(::core::mem::align_of::<$name>() == $align, "pod_struct: the declared alignment is wrong");
            assert!(
                ::core::mem::size_of::<$name>() == 0 $(+ ::core::mem::size_of::<$field_ty>())*,
                "pod_struct: the struct has padding; add an explicit pad field"
            );
            $($crate::pod::assert_pod::<$field_ty>();)*
        };

        // SAFETY: the struct is `#[repr(C)]` and `Copy`; the assertions above prove that the field
        // sizes add up to the struct's size (no padding anywhere) and that every field is `Pod`
        // (so every bit pattern is valid and there is no interior mutability or `Drop`).
        unsafe impl $crate::pod::Pod for $name {}
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uns_03_math_types_upload_in_field_order() {
        use humanart_math::{DualQuat, F16, Mat4, Quat, Transform, Vec3, Vec4};
        let floats = |bytes: &[u8]| -> Vec<f32> {
            let mut out = vec![0.0_f32; bytes.len() / 4];
            copy_from_bytes(bytes, &mut out).unwrap();
            out
        };
        let v = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        assert_eq!(floats(bytes_of(&v)), [1.0, 2.0, 3.0]);
        // Column-major: the first column's four values come first.
        let column = |base: f32| Vec4 { x: base, y: base + 1.0, z: base + 2.0, w: base + 3.0 };
        let m = Mat4 { cols: [column(0.0), column(4.0), column(8.0), column(12.0)] };
        assert_eq!(floats(bytes_of(&m)), (0..16).map(|i| i as f32).collect::<Vec<_>>());
        let q = Quat { x: 0.5, y: 0.25, z: 0.125, w: 1.0 };
        assert_eq!(floats(bytes_of(&q)), [0.5, 0.25, 0.125, 1.0], "(x, y, z, w)");
        let dq = DualQuat { real: q, dual: Quat { x: 9.0, y: 8.0, z: 7.0, w: 6.0 } };
        assert_eq!(floats(bytes_of(&dq))[4..], [9.0, 8.0, 7.0, 6.0]);
        let t = Transform { translation: v, rotation: q, scale: Vec3 { x: 4.0, y: 5.0, z: 6.0 } };
        assert_eq!(floats(bytes_of(&t)), [1.0, 2.0, 3.0, 0.5, 0.25, 0.125, 1.0, 4.0, 5.0, 6.0]);
        let halves = [F16::from_f32(1.0), F16::from_f32(-2.0)];
        assert_eq!(slice_as_bytes(&halves), [0x00, 0x3c, 0x00, 0xc0]);
        let back: Vec3 = from_bytes(bytes_of(&v)).unwrap();
        assert_eq!(back, v);
        assert_eq!(cast_slice::<Vec4>(slice_as_bytes(&m.cols)).unwrap(), &m.cols);
    }

    pod_struct! {
        /// A mixed-field struct without padding.
        #[derive(Debug, PartialEq)]
        struct Mixed(size = 16, align = 4) {
            a: u32,
            b: [u16; 2],
            c: f32,
            d: [u8; 4],
        }
    }

    pod_struct! {
        /// Alignment 8 with an explicit pad that fills what would be padding.
        #[derive(Debug, PartialEq)]
        struct Wide(size = 16, align = 8) {
            a: u64,
            b: u32,
            pad: u32,
        }
    }

    #[test]
    fn uns_03_bytes_of_covers_the_whole_value_in_little_endian_order() {
        let v = Mixed { a: 0x0403_0201, b: [0x0605, 0x0807], c: 1.0, d: [13, 14, 15, 16] };
        let bytes = bytes_of(&v);
        assert_eq!(bytes.len(), 16);
        assert_eq!(&bytes[..8], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(&bytes[8..12], &1.0f32.to_le_bytes());
        assert_eq!(&bytes[12..], &[13, 14, 15, 16]);
    }

    #[test]
    fn uns_03_slice_as_bytes_is_the_elements_back_to_back() {
        let values = [1u32, 2, 0xffff_ffff];
        let bytes = slice_as_bytes(&values);
        assert_eq!(bytes, &[1, 0, 0, 0, 2, 0, 0, 0, 255, 255, 255, 255]);
        let empty: [u32; 0] = [];
        assert!(slice_as_bytes(&empty).is_empty(), "an empty slice gives no bytes");
        let nested = [[1.0f32, 2.0], [3.0, 4.0]];
        assert_eq!(slice_as_bytes(&nested).len(), 16);
    }

    #[test]
    fn uns_03_from_bytes_round_trips_and_rejects_a_wrong_length() {
        let v = Wide { a: u64::MAX - 5, b: 77, pad: 0 };
        let bytes = bytes_of(&v).to_vec();
        assert_eq!(from_bytes::<Wide>(&bytes), Ok(v));
        assert_eq!(from_bytes::<Wide>(&bytes[..15]), Err(PodError::SizeMismatch { expected: 16, actual: 15 }));
        assert_eq!(from_bytes::<u32>(&[]), Err(PodError::SizeMismatch { expected: 4, actual: 0 }));
    }

    #[test]
    fn uns_03_from_bytes_reads_from_an_unaligned_address() {
        // One byte in front of the value guarantees that at least one of the two reads below starts
        // at an odd address.
        let mut storage = [0u8; 24];
        for offset in [0usize, 1] {
            storage[offset..offset + 8].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
            assert_eq!(from_bytes::<u64>(&storage[offset..offset + 8]), Ok(0x1122_3344_5566_7788));
        }
    }

    #[test]
    fn uns_03_cast_slice_checks_alignment_length_and_zero_size() {
        let values = [1u32, 2, 3, 4];
        let bytes = slice_as_bytes(&values);
        assert_eq!(cast_slice::<u32>(bytes), Ok(&values[..]));
        assert_eq!(cast_slice::<u32>(&bytes[..6]), Err(PodError::SizeMismatch { expected: 4, actual: 6 }));
        // `bytes` starts aligned to 4, so one byte further cannot be.
        assert_eq!(cast_slice::<u32>(&bytes[1..5]), Err(PodError::Misaligned { required: 4 }));
        assert_eq!(cast_slice::<[u8; 0]>(bytes), Err(PodError::ZeroSized));
        assert_eq!(cast_slice::<u8>(&bytes[1..4]).map(<[u8]>::len), Ok(3), "bytes are always aligned");
        assert_eq!(cast_slice::<u32>(&[]).map(<[u32]>::len), Ok(0));
    }

    #[test]
    fn uns_03_copy_from_bytes_accepts_unaligned_input_and_rejects_a_wrong_length() {
        let source = [9u8, 1, 0, 0, 0, 2, 0, 0, 0];
        let mut out = [0u32; 2];
        assert_eq!(copy_from_bytes(&source[1..], &mut out), Ok(()));
        assert_eq!(out, [1, 2]);
        assert_eq!(copy_from_bytes(&source, &mut out), Err(PodError::SizeMismatch { expected: 8, actual: 9 }));
    }

    #[test]
    fn uns_03_mutable_views_write_through() {
        let mut v = Mixed { a: 0, b: [0, 0], c: 0.0, d: [0; 4] };
        bytes_of_mut(&mut v)[..4].copy_from_slice(&7u32.to_le_bytes());
        assert_eq!(v.a, 7);
        let mut values = [0u16; 3];
        slice_as_bytes_mut(&mut values).fill(0xff);
        assert_eq!(values, [0xffff; 3]);
    }

    #[test]
    fn uns_03_zeroed_is_all_zero_bytes() {
        let v: Wide = zeroed();
        assert_eq!(v, Wide { a: 0, b: 0, pad: 0 });
        assert!(bytes_of(&zeroed::<[f32; 7]>()).iter().all(|&b| b == 0));
    }

    #[test]
    fn pod_errors_read_as_sentences() {
        assert_eq!(
            PodError::SizeMismatch { expected: 16, actual: 15 }.to_string(),
            "15 bytes do not fit a type of 16 bytes"
        );
        assert_eq!(PodError::Misaligned { required: 4 }.to_string(), "bytes are not aligned to 4");
        assert_eq!(PodError::ZeroSized.to_string(), "a zero-sized type cannot be cast from bytes");
    }
}
