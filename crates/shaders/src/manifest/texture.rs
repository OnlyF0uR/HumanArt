//! The texture group of the manifest: the WGSL under `shaders/texture/`, owned by the texture
//! compression and mip generation work (`humanart-render::compress`, `[SKN-04]` `[IO-05]`).
//!
//! All four files are compute shaders without `#ifdef`s; their variants are override constants
//! (`[PERF-R5]`), and `humanart-render::compress` builds every listed combination once at start-up
//! (`[PERF-R4]`).

use super::{Group, Library, NO_FLAGS, Override, OverrideType, Purpose, ShaderDesc, SourceFile, Targets, compute};

/// The texture group, listed in [`super::GROUPS`].
pub static GROUP: Group = Group { name: "texture", sources: SOURCES, libraries: LIBRARIES, shaders: SHADERS };

/// Every file under `shaders/texture/`.
static SOURCES: &[SourceFile] = &[
    source!("texture/mip.wgsl"),
    source!("texture/bc7_encode.wgsl"),
    source!("texture/bc6h_encode.wgsl"),
    source!("texture/bc_fetch.wgsl"),
];

/// The group has no include-only files.
static LIBRARIES: &[Library] = &[];

/// The texture shaders.
static SHADERS: &[ShaderDesc] = &[
    ShaderDesc {
        path: "texture/mip.wgsl",
        summary: "one mip level by the exact area filter: sRGB-correct, premultiplied alpha, data, normal + Toksvig/vMF roughness [IO-05]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("reduce_unorm"), compute("reduce_float")],
        overrides: &[
            // 0 linear colour, 1 sRGB colour, 2 data, 3 normal + roughness.
            Override { name: "COLOR_SPACE", ty: OverrideType::U32, values: &[1.0, 0.0, 2.0, 3.0] },
            Override { name: "FROM_SOURCE", ty: OverrideType::Bool, values: &[1.0, 0.0] },
        ],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "texture/bc7_encode.wgsl",
        summary: "BC7 encoder, one invocation per block: modes 0-7 by quality level [SKN-04]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("encode")],
        overrides: &[Override { name: "QUALITY", ty: OverrideType::U32, values: &[1.0, 0.0, 2.0] }],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "texture/bc6h_encode.wgsl",
        summary: "BC6H encoder (unsigned or signed), one invocation per block: all 14 modes [SKN-04]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("encode")],
        overrides: &[
            Override { name: "SIGNED", ty: OverrideType::Bool, values: &[0.0, 1.0] },
            Override { name: "QUALITY", ty: OverrideType::U32, values: &[1.0, 0.0, 2.0] },
        ],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "texture/bc_fetch.wgsl",
        summary: "test kernel: textureLoad of every texel, for hardware BC decode parity [SKN-04]",
        purpose: Purpose::Test,
        entry_points: &[compute("fetch")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
];
