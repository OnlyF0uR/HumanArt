//! The groom group of the manifest: the WGSL under `shaders/groom/`, owned by the hair system
//! (`humanart-render::groom`, WP-17, `[HAIR-02]`).
//!
//! Compute shaders without `#ifdef`s or override constants; `humanart-render::groom` builds one
//! pipeline per entry point once, when a groom is loaded (`[PERF-R4]`).

use super::{Group, Library, NO_FLAGS, Purpose, ShaderDesc, SourceFile, Targets, compute};

/// The groom group, listed in [`super::GROUPS`].
pub static GROUP: Group = Group { name: "groom", sources: SOURCES, libraries: LIBRARIES, shaders: SHADERS };

/// Every file under `shaders/groom/`.
static SOURCES: &[SourceFile] = &[source!("groom/common.wgsl"), source!("groom/generate.wgsl")];

/// The include-only file of the group.
static LIBRARIES: &[Library] = &[Library {
    path: "groom/common.wgsl",
    summary: "PCG hash and per-strand random values, root frames on the skin, angle wrap, 2D dispatch index",
    functions: &[
        "groom_hash",
        "groom_rand",
        "groom_normalize_or",
        "groom_fallback_tangent",
        "groom_root_frame",
        "groom_to_world",
        "groom_to_local",
        "groom_wrap_angle",
        "groom_thread",
    ],
    purpose: Purpose::Runtime,
}];

/// The groom shaders.
static SHADERS: &[ShaderDesc] = &[ShaderDesc {
    path: "groom/generate.wgsl",
    summary: "hair generation modifiers: guide interpolation, length, direction, parting, cowlicks, curl, wave, frizz, flyaways, clumping, scalp collision, f16 storage, attach to the deformed skin [HAIR-02]",
    purpose: Purpose::Runtime,
    entry_points: &[compute("shape"), compute("finish"), compute("attach")],
    overrides: &[],
    flag_sets: NO_FLAGS,
    targets: Targets::NONE,
}];
