//! The bake group of the manifest: the WGSL under `shaders/bake/`, owned by the bakers
//! (`humanart-tools::bake::gpu`, WP-15, `[BAKE-01]`..`[BAKE-03]`).
//!
//! `bake/common.wgsl` is the shared library (ray directions, mesh and surface reconstruction, BVH
//! traversal); the G-buffer is a render pass into an `r32uint` + `rg32float` pair; the other four
//! are compute shaders. `humanart-tools::bake::gpu::GpuBaker` builds every pipeline once
//! (`[PERF-R4]`); the dilation mode is an override constant (`[PERF-R5]`).

use super::{
    EntryPoint, Group, Library, NO_FLAGS, Override, OverrideType, Purpose, ShaderDesc, SourceFile, Stage, Targets,
    compute,
};

/// A `@vertex` entry point.
const fn vertex(name: &'static str) -> EntryPoint {
    EntryPoint { name, stage: Stage::Vertex }
}

/// A `@fragment` entry point.
const fn fragment(name: &'static str) -> EntryPoint {
    EntryPoint { name, stage: Stage::Fragment }
}

/// The bake group, listed in [`super::GROUPS`].
pub static GROUP: Group = Group { name: "bake", sources: SOURCES, libraries: LIBRARIES, shaders: SHADERS };

/// Every file under `shaders/bake/`.
static SOURCES: &[SourceFile] = &[
    source!("bake/common.wgsl"),
    source!("bake/gbuffer.wgsl"),
    source!("bake/thickness.wgsl"),
    source!("bake/cavity_ao.wgsl"),
    source!("bake/dilate.wgsl"),
    source!("bake/curvature.wgsl"),
];

/// The include-only file of the group.
static LIBRARIES: &[Library] = &[Library {
    path: "bake/common.wgsl",
    summary: "bake ray directions (PCG-rotated Hammersley, cosine and cone warps), mesh bindings, surface point, BVH traversal",
    functions: &[
        "bake_pcg",
        "bake_texel_hash",
        "bake_rotation",
        "bake_point",
        "bake_cosine_hemisphere",
        "bake_uniform_cone",
        "bake_frame_t",
        "bake_to_world",
        "bake_keep_side",
        "bake_normalize_or",
        "bake_surface",
        "bake_to_tangent",
        "bake_enter",
        "bake_intersect",
    ],
    purpose: Purpose::Runtime,
}];

/// The bake shaders.
static SHADERS: &[ShaderDesc] = &[
    ShaderDesc {
        path: "bake/gbuffer.wgsl",
        summary: "UV-space G-buffer: triangle id + barycentrics per texel, top-left rule [BAKE-01..03]",
        purpose: Purpose::Runtime,
        entry_points: &[vertex("vs_gbuffer"), fragment("fs_gbuffer")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[wgpu::TextureFormat::R32Uint, wgpu::TextureFormat::Rg32Float], depth: None },
    },
    ShaderDesc {
        path: "bake/thickness.wgsl",
        summary: "thickness and translucency by ray casting into a cone, per-region scale [BAKE-01]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("thickness")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "bake/cavity_ao.wgsl",
        summary: "ambient occlusion, cavity and tangent-space bent normals [BAKE-02]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("cavity_ao")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "bake/curvature.wgsl",
        summary: "mean-curvature map: per-vertex H interpolated into the G-buffer texels [BAKE-03]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("curvature")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "bake/dilate.wgsl",
        summary: "one seam-padding dilation pass: mean (MODE 0) or first filled neighbour (MODE 1) [BAKE-03]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("dilate")],
        overrides: &[Override { name: "MODE", ty: OverrideType::U32, values: &[0.0, 1.0] }],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
];
