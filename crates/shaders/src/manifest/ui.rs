//! The UI group of the manifest: the WGSL under `shaders/ui/`, owned by the UI work (`humanart-ui`).
//!
//! To register a file: put it under `shaders/ui/`, add `source!("ui/<name>.wgsl")` to `SOURCES`,
//! and describe it in `LIBRARIES` (include-only files) or `SHADERS` (modules: entry points with
//! stages, override constants with their values, flag sets, attachment formats). The manifest
//! tests and the validation in `crate::validate` pick it up through [`GROUP`]; nothing else in
//! this crate changes. Files may `#include` the core libraries (`"lib/color.wgsl"`, ...).

use super::{
    EntryPoint, Group, Library, NO_FLAGS, Override, OverrideType, Purpose, ShaderDesc, SourceFile, Stage, Targets,
};

/// The UI group, listed in [`super::GROUPS`].
pub static GROUP: Group = Group { name: "ui", sources: SOURCES, libraries: LIBRARIES, shaders: SHADERS };

/// Every file under `shaders/ui/`.
static SOURCES: &[SourceFile] = &[source!("ui/quad.wgsl")];

/// The UI's include-only files.
static LIBRARIES: &[Library] = &[];

/// The UI's shaders.
static SHADERS: &[ShaderDesc] = &[ShaderDesc {
    path: "ui/quad.wgsl",
    summary: "The UI pass: instanced SDF quads (rounded rectangles, borders, shadows, MSDF glyphs, images,               segments, gradients) read from a storage buffer, clipped per instance [UI-02]",
    purpose: Purpose::Runtime,
    entry_points: &[
        EntryPoint { name: "vs_main", stage: Stage::Vertex },
        EntryPoint { name: "fs_main", stage: Stage::Fragment },
    ],
    // true for a UNORM target (the shader encodes sRGB), false for an *Srgb target.
    overrides: &[Override { name: "ENCODE_SRGB", ty: OverrideType::Bool, values: &[1.0, 0.0] }],
    flag_sets: NO_FLAGS,
    targets: Targets { color: &[wgpu::TextureFormat::Rgba8Unorm], depth: None },
}];
