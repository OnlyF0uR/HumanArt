//! The hair group of the manifest: the WGSL under `shaders/hair/`, owned by the hair renderer
//! (`humanart-render::hair`, WP-18, `[HR-01]`..`[HR-05]`).
//!
//! No `#ifdef`s or override constants: `humanart-render::hair` builds every pipeline once, when
//! the hair renderer is created (`[PERF-R4]`).

use super::{EntryPoint, Group, Library, NO_FLAGS, Purpose, ShaderDesc, SourceFile, Stage, Targets, compute};

/// A vertex entry point.
const fn vertex(name: &'static str) -> EntryPoint {
    EntryPoint { name, stage: Stage::Vertex }
}

/// A fragment entry point.
const fn fragment(name: &'static str) -> EntryPoint {
    EntryPoint { name, stage: Stage::Fragment }
}

/// The hair group, listed in [`super::GROUPS`].
pub static GROUP: Group = Group { name: "hair", sources: SOURCES, libraries: LIBRARIES, shaders: SHADERS };

/// Every file under `shaders/hair/`.
static SOURCES: &[SourceFile] = &[
    source!("hair/bsdf.wgsl"),
    source!("hair/common.wgsl"),
    source!("hair/raster.wgsl"),
    source!("hair/dom.wgsl"),
    source!("hair/resolve.wgsl"),
    source!("hair/motion.wgsl"),
    source!("hair/ribbon.wgsl"),
    source!("hair/probe.wgsl"),
];

/// The HDR target.
const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Reversed-Z depth.
const DEPTH: Option<wgpu::TextureFormat> = Some(wgpu::TextureFormat::Depth32Float);

/// The include-only files of the group.
static LIBRARIES: &[Library] = &[
    Library {
        path: "hair/common.wgsl",
        summary: "hair pass constants (view, draw, groom colours), strand absorption from melanin, the visibility record, deep-opacity-map frames and layers [HR-01] [HR-03]",
        functions: &[
            "hair_hash",
            "hair_kept",
            "hair_sigma_a",
            "hair_pack_record",
            "hair_record_with_coverage",
            "hair_record_coverage",
            "hair_alpha",
            "hair_record_groom",
            "hair_record_grey",
            "hair_record_color",
            "hair_dom_local",
            "hair_dom_tau",
            "hair_thread",
        ],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "hair/bsdf.wgsl",
        summary: "far-field hair fibre scattering: R/TT/TRT lobes with tilted longitudinal lobes, tabulated azimuthal terms, melanin absorption, dual scattering [HR-02]",
        functions: &[
            "hair_i0",
            "hair_log_i0",
            "hair_longitudinal",
            "hair_tilt",
            "hair_sigma_coord",
            "hair_azimuthal_terms",
            "hair_lobes",
            "hair_single",
            "hair_dual_cell",
            "hair_back",
            "hair_dual_scattering",
            "hair_albedo",
        ],
        purpose: Purpose::Runtime,
    },
];

/// The hair shaders.
static SHADERS: &[ShaderDesc] = &[
    ShaderDesc {
        path: "hair/raster.wgsl",
        summary: "compute software rasteriser of sub-pixel strands: coverage sum and nearest depth, visibility records with motion, fibre density grid, scene depth tiles of the hidden-segment test [HR-01] [HR-04] [HR-05]",
        purpose: Purpose::Runtime,
        entry_points: &[
            compute("raster_depth"),
            compute("raster_records"),
            compute("voxelize"),
            compute("depth_tiles"),
        ],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "hair/dom.wgsl",
        summary: "deep opacity maps: first fibre depth (lines, depth only) and four additive opacity layers per light [HR-03]",
        purpose: Purpose::Runtime,
        entry_points: &[vertex("vs_dom"), fragment("fs_dom_opacity")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[HDR], depth: None },
    },
    ShaderDesc {
        path: "hair/resolve.wgsl",
        summary: "deferred hair shading (dual scattering, deep opacity, shadow map, environment with density occlusion) and composite over the scene, hair shadows on the skin [HR-02] [HR-03]",
        purpose: Purpose::Runtime,
        entry_points: &[vertex("vs_resolve"), fragment("fs_resolve")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[HDR], depth: None },
    },
    ShaderDesc {
        path: "hair/motion.wgsl",
        summary: "hair motion and depth into the TAA inputs where hair covers half a pixel [HR-05]",
        purpose: Purpose::Runtime,
        entry_points: &[vertex("vs_motion"), fragment("fs_motion")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[wgpu::TextureFormat::Rgba32Uint], depth: DEPTH },
    },
    ShaderDesc {
        path: "hair/ribbon.wgsl",
        summary: "hardware ribbon rasteriser fallback feeding the same visibility buffer [HR-01]",
        purpose: Purpose::Runtime,
        entry_points: &[vertex("vs_ribbon"), fragment("fs_ribbon_depth"), fragment("fs_ribbon_records")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[wgpu::TextureFormat::R8Unorm], depth: DEPTH },
    },
    ShaderDesc {
        path: "hair/probe.wgsl",
        summary: "GPU test kernel for hair/bsdf.wgsl (dual-scattered fibre response per case) [HR-02]",
        purpose: Purpose::Test,
        entry_points: &[compute("probe")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
];
