//! The constant blocks the render shaders read, laid out exactly like their WGSL structs
//! (`Frame` in `shaders/render/frame.wgsl`, `ShadowView` in `shadow.wgsl`, `TonemapParams` in
//! `post/tonemap.wgsl`, `IblParams` in `ibl.wgsl`). Little-endian `#[repr(C)]` fields of `f32`,
//! `u32` and arrays of them, checked by `pod_struct!` for size and padding.

use humanart_gpu::pod_struct;

use crate::light::{GpuLight, MAX_LIGHTS, MAX_SHADOWED};

/// Bytes the shaders see at each dynamic offset of the uniform ring: the largest block
/// ([`FrameUniforms`]) rounded up to 256.
pub const RING_BINDING_SIZE: u32 = 1792;

pod_struct! {
    /// The per-frame constants (`struct Frame`).
    #[derive(Debug, PartialEq)]
    pub struct FrameUniforms(size = 1632, align = 4) {
        /// World to clip with the TAA jitter.
        pub view_proj: [f32; 16],
        /// World to clip without jitter.
        pub unjittered_view_proj: [f32; 16],
        /// The previous frame's unjittered world to clip.
        pub prev_view_proj: [f32; 16],
        /// Clip to world of the unjittered matrix.
        pub inv_view_proj: [f32; 16],
        /// World to shadow clip per atlas layer.
        pub shadow_view_proj: [[f32; 16]; MAX_SHADOWED],
        /// Per layer: near, far, half extent, penumbra.
        pub shadow_params: [[f32; 4]; MAX_SHADOWED],
        /// Per layer: 0 orthographic, 1 perspective.
        pub shadow_kind: [f32; 4],
        /// Eye position, exposure scale.
        pub eye_exposure: [f32; 4],
        /// Width, height, 1 / width, 1 / height.
        pub viewport: [f32; 4],
        /// Tile origin x, y in the full image; full width, full height.
        pub tile: [f32; 4],
        /// Current jitter (pixels), previous jitter.
        pub jitter: [f32; 4],
        /// Environment scale (nits), rotation, specular mip count, light count.
        pub environment: [f32; 4],
        /// Backdrop: the flat radiance (w = 0), nothing (w = 1: the environment) or the paper's
        /// reflectance (w = 2: the studio backdrop).
        pub background: [f32; 4],
        /// Studio backdrop: the wall's unit normal (towards the camera) and its plane offset
        /// (`normal . p = w` on the wall).
        pub backdrop_wall: [f32; 4],
        /// Studio backdrop: the floor's height, the radius of the cove between wall and floor,
        /// the falloff strength and the falloff radius.
        pub backdrop_floor: [f32; 4],
        /// Studio backdrop: the spot on the wall behind the subject; w unused.
        pub backdrop_spot: [f32; 4],
        /// Depth of field: the focus distance (metres along the view axis), the radius of the
        /// circle of confusion at infinity (pixels), the largest radius drawn (pixels), on (1/0).
        pub lens: [f32; 4],
        /// Screen-space occlusion: the ambient radius (metres), the contact-shadow length
        /// (metres), the tangent of half the vertical field of view, 0 off or the resolution divisor
        /// of the occlusion targets (1 full, 2 half).
        pub occlusion: [f32; 4],
        /// TAA minimum weight, mode (0 TAA, 1 accumulate), frames accumulated, reset.
        pub taa: [f32; 4],
        /// Frame index, debug view, shadow map size, flags.
        pub counters: [u32; 4],
        /// The lights.
        pub lights: [GpuLight; MAX_LIGHTS],
        /// World to clip of the whole image with the TAA jitter: `view_proj` before a still's
        /// tile is cropped from it, `view_proj` itself otherwise. The geometry pass snaps its
        /// vertices to this image's sub-pixel grid, so a tile rasterises like the untiled image
        /// `[QUAL-04]`.
        pub image_view_proj: [f32; 16],
    }
}

impl FrameUniforms {
    /// All zero, no lights.
    pub const ZERO: Self = Self {
        view_proj: [0.0; 16],
        unjittered_view_proj: [0.0; 16],
        prev_view_proj: [0.0; 16],
        inv_view_proj: [0.0; 16],
        shadow_view_proj: [[0.0; 16]; MAX_SHADOWED],
        shadow_params: [[0.0; 4]; MAX_SHADOWED],
        shadow_kind: [0.0; 4],
        eye_exposure: [0.0; 4],
        viewport: [0.0; 4],
        tile: [0.0; 4],
        jitter: [0.0; 4],
        environment: [0.0; 4],
        background: [0.0; 4],
        backdrop_wall: [0.0; 4],
        backdrop_floor: [0.0; 4],
        backdrop_spot: [0.0; 4],
        lens: [0.0; 4],
        occlusion: [0.0; 4],
        taa: [0.0; 4],
        counters: [0; 4],
        lights: [GpuLight::NONE; MAX_LIGHTS],
        image_view_proj: [0.0; 16],
    };
}

pod_struct! {
    /// The constants of one shadow-map draw (`struct ShadowView`).
    #[derive(Debug, PartialEq)]
    pub struct ShadowViewUniforms(size = 64, align = 4) {
        /// World to shadow clip.
        pub view_proj: [f32; 16],
    }
}

pod_struct! {
    /// The constants of the display transform (`struct TonemapParams` of `post/tonemap.wgsl`).
    #[derive(Debug, PartialEq)]
    pub struct TonemapUniforms(size = 16, align = 4) {
        /// The exposure scale.
        pub exposure: f32,
        /// The dither added before the 8-bit rounding, in steps of 1/255 (0: none).
        pub dither: f32,
        /// The origin of a still's tile in the full image, pixels (the dither's seed).
        pub tile: [f32; 2],
    }
}

pod_struct! {
    /// The constants of one IBL dispatch (`struct IblParams`).
    #[derive(Debug, PartialEq)]
    pub struct IblUniforms(size = 48, align = 4) {
        /// Face size written, mip, samples, procedural kind.
        pub ints: [u32; 4],
        /// Roughness, source face size, source mip count, unused.
        pub floats: [f32; 4],
        /// Towards the sun.
        pub sun: [f32; 4],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sizes the WGSL structs have (uniform layout: 16-byte aligned members).
    #[test]
    fn blocks_match_their_wgsl_sizes() {
        assert_eq!(std::mem::size_of::<FrameUniforms>(), 4 * 64 + 4 * 64 + 4 * 16 + 14 * 16 + 8 * 96 + 64);
        assert!(std::mem::size_of::<FrameUniforms>() as u32 <= RING_BINDING_SIZE);
        assert_eq!(RING_BINDING_SIZE % 256, 0);
        assert_eq!(std::mem::size_of::<IblUniforms>(), 48);
    }

    /// The WGSL source declares the members in the same order as the Rust struct.
    #[test]
    fn frame_struct_members_match_the_shader() {
        let text = humanart_shaders::manifest::source("render/frame.wgsl").unwrap();
        let start = text.find("struct Frame {").unwrap();
        let body = &text[start..text[start..].find('}').unwrap() + start];
        let members: Vec<&str> = body
            .lines()
            .map(str::trim)
            .filter(|l| !l.starts_with("//") && l.contains(':'))
            .map(|l| l.split(':').next().unwrap().trim())
            .collect();
        assert_eq!(
            members,
            [
                "view_proj",
                "unjittered_view_proj",
                "prev_view_proj",
                "inv_view_proj",
                "shadow_view_proj",
                "shadow_params",
                "shadow_kind",
                "eye_exposure",
                "viewport",
                "tile",
                "jitter",
                "environment",
                "background",
                "backdrop_wall",
                "backdrop_floor",
                "backdrop_spot",
                "lens",
                "occlusion",
                "taa",
                "counters",
                "lights",
                "image_view_proj",
            ]
        );
    }
}
