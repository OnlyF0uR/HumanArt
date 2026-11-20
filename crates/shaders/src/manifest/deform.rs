//! The deformation group of the manifest: the WGSL under `shaders/deform/`, owned by the GPU
//! evaluation (`humanart-render::deform`, WP-10, `[EVAL-02]`..`[EVAL-06]`).
//!
//! All files are compute shaders without `#ifdef`s or override constants; `humanart-render::deform`
//! builds one pipeline per entry point once, when a character is loaded (`[PERF-R4]`).

use super::{Group, Library, NO_FLAGS, Purpose, ShaderDesc, SourceFile, Targets, compute};

/// The deformation group, listed in [`super::GROUPS`].
pub static GROUP: Group = Group { name: "deform", sources: SOURCES, libraries: LIBRARIES, shaders: SHADERS };

/// Every file under `shaders/deform/`.
static SOURCES: &[SourceFile] = &[
    source!("deform/common.wgsl"),
    source!("deform/accumulate.wgsl"),
    source!("deform/skin.wgsl"),
    source!("deform/normals.wgsl"),
    source!("deform/strain.wgsl"),
    source!("deform/lod.wgsl"),
];

/// The include-only file of the group.
static LIBRARIES: &[Library] = &[Library {
    path: "deform/common.wgsl",
    summary: "the 2D dispatch index of 64-wide workgroups, exact f16 delta decoding without subnormals",
    functions: &["thread_index", "decode_f16", "decode_delta"],
    purpose: Purpose::Runtime,
}];

/// The deformation shaders.
static SHADERS: &[ShaderDesc] = &[
    ShaderDesc {
        path: "deform/accumulate.wgsl",
        summary: "rest shape: full gather of the vertex-major shape deltas, per-component and sculpt-layer scatter [EVAL-02] [EVAL-03]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("gather_rest"), compute("scatter_component"), compute("scatter_layer")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "deform/skin.wgsl",
        summary: "expressions, combination and pose correctives, LBS / DQS skinning per vertex [SKIN-01] [SKIN-02]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("skin")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "deform/normals.wgsl",
        summary: "triangle frames, then angle-weighted vertex normals and tangents over CSR gathers [EVAL-04]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("triangle_frames"), compute("vertex_frames")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "deform/strain.wgsl",
        summary: "edge strain, per-region workgroup reduction, subtree normalisation, per-vertex strain [EVAL-06]",
        purpose: Purpose::Runtime,
        entry_points: &[
            compute("edge_strain"),
            compute("region_sums"),
            compute("region_strain"),
            compute("vertex_strain"),
        ],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "deform/lod.wgsl",
        summary: "positions of a reduced level of detail from the deformed LOD0 by barycentric sources [MOD-05]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("transfer")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
];
