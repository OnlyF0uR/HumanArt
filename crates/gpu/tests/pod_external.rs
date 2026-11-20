//! `pod_struct!` used from another crate that forbids unsafe code `[UNS-01, UNS-03]`.
//!
//! Every crate above `humanart-gpu` carries `#![forbid(unsafe_code)]` and still has to declare the structs
//! it uploads. This test is such a crate: it compiles only if the macro's `unsafe impl` (which comes
//! from `humanart-gpu` and is guarded by the macro's static checks) is accepted there.
#![forbid(unsafe_code)]

use humanart_gpu::pod::{bytes_of, cast_slice, from_bytes, slice_as_bytes};

humanart_gpu::pod_struct! {
    /// A frame-constants block as a renderer would declare it.
    #[derive(Debug, PartialEq)]
    pub struct FrameConstants(size = 80, align = 4) {
        /// Column-major view-projection matrix.
        pub view_proj: [[f32; 4]; 4],
        /// Camera position, metres; `w` unused.
        pub eye: [f32; 4],
    }
}

humanart_gpu::pod_struct! {
    /// A struct nested in another one.
    #[derive(Debug, PartialEq)]
    struct Pair(size = 24, align = 8) {
        id: u64,
        frame: [u32; 2],
        weights: [f32; 2],
    }
}

#[test]
fn uns_03_pod_struct_works_in_a_crate_that_forbids_unsafe_code() {
    let constants = FrameConstants { view_proj: [[1.0, 0.0, 0.0, 0.0]; 4], eye: [0.0, 1.6, 2.0, 1.0] };
    let bytes = bytes_of(&constants);
    assert_eq!(bytes.len(), 80);
    assert_eq!(from_bytes::<FrameConstants>(bytes), Ok(constants));
}

#[test]
fn uns_03_slices_of_declared_structs_cast_both_ways() {
    let pairs =
        [Pair { id: 1, frame: [2, 3], weights: [0.5, 0.25] }, Pair { id: 9, frame: [8, 7], weights: [1.0, 0.0] }];
    let bytes = slice_as_bytes(&pairs);
    assert_eq!(bytes.len(), 48);
    assert_eq!(cast_slice::<Pair>(bytes), Ok(&pairs[..]));
}
