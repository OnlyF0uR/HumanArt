//! The skin group of the manifest: the WGSL under `shaders/skin/`, owned by the skin appearance
//! work (`humanart-render::skin`, WP-13: `[SKN-01]` `[SKN-02]` `[SKN-10]` `[SKN-20]` `[SKN-21]`
//! `[SKN-03]` `[SKN-30]`).
//!
//! Every shader is compute-only without `#ifdef`s or override constants; `humanart-render::skin`
//! creates every pipeline once at start-up (`[PERF-R4]`).

use super::{Group, Library, NO_FLAGS, Purpose, ShaderDesc, SourceFile, Targets, compute};

/// The skin group, listed in [`super::GROUPS`].
pub static GROUP: Group = Group { name: "skin", sources: SOURCES, libraries: LIBRARIES, shaders: SHADERS };

/// Every file under `shaders/skin/`.
static SOURCES: &[SourceFile] = &[
    source!("skin/noise.wgsl"),
    source!("skin/antitile.wgsl"),
    source!("skin/surface.wgsl"),
    source!("skin/common.wgsl"),
    source!("skin/composite.wgsl"),
    source!("skin/resolve.wgsl"),
    source!("skin/microdetail.wgsl"),
    source!("skin/wrinkle.wgsl"),
    source!("skin/wrinkle_drivers.wgsl"),
    source!("skin/page_copy.wgsl"),
];

/// The include-only files.
static LIBRARIES: &[Library] = &[
    Library {
        path: "skin/noise.wgsl",
        summary: "PCG hash (Jarzynski & Olano 2020) and periodic value / fractal noise, bit-exact twins of the CPU reference",
        functions: &["pcg", "hash3", "hash_unit", "fade", "value_noise", "fbm"],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "skin/antitile.wgsl",
        summary: "histogram-preserving tiling and blending (Heitz & Neyret 2018) [SKN-20]",
        functions: &["antitile_triangle", "antitile_offset", "antitile_fetch", "antitile_sample"],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "skin/surface.wgsl",
        summary: "skin shading helpers: micro-shadowing, horizon visibility, flushed albedo [SKN-21] [SKN-31]",
        functions: &["skin_micro_shadow", "skin_horizon_visibility", "skin_flushed_albedo"],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "skin/common.wgsl",
        summary: "the skin compositor's buffer layout: globals, zone table, dispatch parameters, attribute ranges [SKN-01]",
        functions: &["attribute_range", "base_value"],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "skin/wrinkle.wgsl",
        summary: "wrinkle and tension map fetch and blending by driver activations [SKN-30] [EXP-04]",
        functions: &["wrinkle_fetch", "wrinkle_driver", "wrinkle_blend"],
        purpose: Purpose::Runtime,
    },
];

/// The skin shaders.
static SHADERS: &[ShaderDesc] = &[
    ShaderDesc {
        path: "skin/composite.wgsl",
        summary: "skin compositor pass 1: base + zone offsets + layers over one dirty rectangle into the parameter planes [SKN-02]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("composite")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "skin/resolve.wgsl",
        summary: "skin compositor pass 2: spectral pigment model, overrides, flush and wetness into the resolved set [SKN-10]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("resolve")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "skin/microdetail.wgsl",
        summary: "micro-detail exemplar (pores, fine lines, relief), its normals/cavity, and anti-tiled synthesis [SKN-20]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("generate"), compute("derive"), compute("synthesize")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "skin/wrinkle_drivers.wgsl",
        summary: "wrinkle driver activations from expression terms and GPU region strain, and wrinkle blending over a rectangle [SKN-30] [EVAL-06]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("activate"), compute("blend")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "skin/page_copy.wgsl",
        summary: "one streaming page from the resolved skin set, box-averaged to its mip, for the GPU-to-pool copy [SKN-03]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("copy_f16"), compute("copy_unorm8")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
];
