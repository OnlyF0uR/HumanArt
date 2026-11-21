//! The render group of the manifest: the WGSL under `shaders/render/`, owned by the renderer
//! (`humanart-render`, WP-11).
//!
//! Bind groups shared by these files: group 0 the per-frame constants from the uniform ring
//! (`render/frame.wgsl`), group 1 the scene streams (`render/scene.wgsl`) or a post pass's inputs,
//! group 2 the light resources (`render/lighting.wgsl`) or the overlay primitives, group 3 of the
//! geometry pass the skin surface (`render/skin_surface.wgsl`).

use super::{
    EntryPoint, Group, Library, NO_FLAGS, Override, OverrideType, Purpose, ShaderDesc, SourceFile, Stage, Targets,
    compute,
};

/// The render group, listed in [`super::GROUPS`].
pub static GROUP: Group = Group { name: "render", sources: SOURCES, libraries: LIBRARIES, shaders: SHADERS };

/// Every file under `shaders/render/`.
static SOURCES: &[SourceFile] = &[
    source!("render/frame.wgsl"),
    source!("render/scene.wgsl"),
    source!("render/lighting.wgsl"),
    source!("render/subsurface.wgsl"),
    source!("render/skin_surface.wgsl"),
    source!("render/detail_mips.wgsl"),
    source!("render/sss.wgsl"),
    source!("render/geometry.wgsl"),
    source!("render/shadow.wgsl"),
    source!("render/composite.wgsl"),
    source!("render/taa.wgsl"),
    source!("render/debug.wgsl"),
    source!("render/overlay.wgsl"),
    source!("render/pick.wgsl"),
    source!("render/ibl.wgsl"),
    source!("render/occlusion.wgsl"),
    source!("render/dof.wgsl"),
];

/// The renderer's include-only files.
static LIBRARIES: &[Library] = &[
    Library {
        path: "render/frame.wgsl",
        summary: "the per-frame constants (cameras, shadows, exposure, viewport, tile, environment, lights) at group 0, the full-screen triangle",
        functions: &["global_pixel", "fullscreen_position", "pixel_to_ndc", "pixel_direction", "rotate_y"],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "render/scene.wgsl",
        summary: "the scene streams at group 1 (positions, previous positions, normals, uvs, objects, indices) and vertex pulling",
        functions: &[
            "fetch_position",
            "fetch_prev_position",
            "fetch_normal",
            "fetch_uv",
            "world_position",
            "prev_world_position",
            "world_normal",
        ],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "render/lighting.wgsl",
        summary: "light resources at group 2: SH irradiance, prefiltered specular, DFG, shadow atlas with PCSS; directional, spot and rect lights",
        functions: &[
            "sh_irradiance",
            "env_prefiltered",
            "dfg_lookup",
            "shadow_linear_depth",
            "golden_turn",
            "shadow_coords",
            "shadow_search_taps",
            "shadow_filter_taps",
            "shadow_sample",
            "shadow_visibility",
            "polygon_form_factor",
            "incident_light",
            "shade_light",
        ],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "render/subsurface.wgsl",
        summary: "skin in the geometry pass: profile transmittance, shadow-map thickness, pre-integrated falloff [SHD-03] [SHD-04]",
        functions: &["profile_plane", "skin_transmittance", "shadow_slab", "shadow_thickness", "preintegrated_falloff"],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "render/skin_surface.wgsl",
        summary: "surfaced skin at group 3: resolved set and zone masks, Langer-oriented anti-tiled micro-detail, wrinkles, coat, sheen, meso mottling [SKN-20] [SKN-30] [SHD-06..09]",
        functions: &[
            "surface_frame",
            "langer_direction",
            "antitile_vertex_position",
            "mip_level",
            "micro_detail",
            "wrinkle_activation",
            "wrinkle_at",
            "plain_surface",
            "skin_surface",
            "coat_strength",
            "coat_roughness",
            "d_charlie",
            "v_neubelt",
            "coat_lobe",
        ],
        purpose: Purpose::Runtime,
    },
];

const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH: Option<wgpu::TextureFormat> = Some(wgpu::TextureFormat::Depth32Float);
const DISPLAY: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A vertex entry point.
const fn vertex(name: &'static str) -> EntryPoint {
    EntryPoint { name, stage: Stage::Vertex }
}

/// A fragment entry point.
const fn fragment(name: &'static str) -> EntryPoint {
    EntryPoint { name, stage: Stage::Fragment }
}

/// The renderer's shaders.
static SHADERS: &[ShaderDesc] = &[
    ShaderDesc {
        path: "render/geometry.wgsl",
        summary: "opaque geometry pass: lights, PCSS shadows, IBL, dual-lobe skin with translucency; diffuse and specular MRT, G-buffer-lite with motion [RND-03] 5 [SHD-01] [SHD-03]",
        purpose: Purpose::Runtime,
        entry_points: &[vertex("vs_geometry"), fragment("fs_geometry")],
        // PCF taps per shadowed light: Review, Interactive, Still [QUAL-01]; skin diffusion:
        // separable pass (0) or pre-integrated (1) [SHD-04].
        overrides: &[
            Override { name: "SHADOW_TAPS", ty: OverrideType::U32, values: &[16.0, 8.0, 32.0] },
            Override { name: "SSS_MODE", ty: OverrideType::U32, values: &[0.0, 1.0] },
        ],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[HDR, HDR, wgpu::TextureFormat::Rgba32Uint], depth: DEPTH },
    },
    ShaderDesc {
        path: "render/shadow.wgsl",
        summary: "shadow-map pass: depth only, reversed-Z, one atlas layer per light [RND-03] 4",
        purpose: Purpose::Runtime,
        entry_points: &[vertex("vs_shadow")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[], depth: DEPTH },
    },
    ShaderDesc {
        path: "render/sss.wgsl",
        summary: "separable screen-space subsurface scattering of skin, horizontal and vertical [SHD-02] [SHD-05]",
        purpose: Purpose::Runtime,
        entry_points: &[vertex("vs_sss"), fragment("fs_sss_h"), fragment("fs_sss_v")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[HDR], depth: None },
    },
    ShaderDesc {
        path: "render/composite.wgsl",
        summary: "lighting composite: diffuse x albedo + specular, backdrop [RND-03] 9 [LIT-04]",
        purpose: Purpose::Runtime,
        entry_points: &[vertex("vs_composite"), fragment("fs_composite")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[HDR], depth: None },
    },
    ShaderDesc {
        path: "render/occlusion.wgsl",
        summary: "screen-space ambient occlusion (horizon-based) and per-light contact shadows from the prepass depth (todo 1.9)",
        purpose: Purpose::Runtime,
        entry_points: &[vertex("vs_occlusion"), fragment("fs_occlusion")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[wgpu::TextureFormat::R8Unorm, DISPLAY], depth: None },
    },
    ShaderDesc {
        path: "render/dof.wgsl",
        summary: "thin-lens depth of field for stills: one scatter-as-gather pass before the temporal accumulation (todo 1.9)",
        purpose: Purpose::Runtime,
        entry_points: &[vertex("vs_dof"), fragment("fs_dof")],
        // Taps of the spiral: Review and Still, Interactive.
        overrides: &[Override { name: "DOF_TAPS", ty: OverrideType::U32, values: &[48.0, 24.0] }],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[HDR], depth: None },
    },
    ShaderDesc {
        path: "render/taa.wgsl",
        summary: "temporal anti-aliasing and still accumulation with motion-vector reprojection [RND-03] 10 [QUAL-02]",
        purpose: Purpose::Runtime,
        entry_points: &[vertex("vs_taa"), fragment("fs_taa")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[HDR], depth: None },
    },
    ShaderDesc {
        path: "render/debug.wgsl",
        summary: "debug and analysis views (section 8.5) to the display target",
        purpose: Purpose::Runtime,
        entry_points: &[vertex("vs_debug"), fragment("fs_debug")],
        overrides: &[Override { name: "ENCODE_SRGB", ty: OverrideType::Bool, values: &[1.0, 0.0] }],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[DISPLAY], depth: None },
    },
    ShaderDesc {
        path: "render/overlay.wgsl",
        summary: "overlays: screen-space lines and points, wireframe, selection rim [RND-03] 11",
        purpose: Purpose::Runtime,
        entry_points: &[vertex("vs_lines"), vertex("vs_wire"), vertex("vs_selection"), fragment("fs_overlay")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[DISPLAY], depth: DEPTH },
    },
    ShaderDesc {
        path: "render/pick.wgsl",
        summary: "picking buffer: object pick id, depth, triangle and barycentrics under the cursor [VP-01]",
        purpose: Purpose::Runtime,
        entry_points: &[vertex("vs_pick"), fragment("fs_pick")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[wgpu::TextureFormat::Rgba32Uint], depth: DEPTH },
    },
    ShaderDesc {
        path: "render/detail_mips.wgsl",
        summary: "skin micro-detail exemplar to mip-mapped slopes and slope moments for the geometry pass [SKN-20] [SKN-21]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("convert"), compute("downsample")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "render/ibl.wgsl",
        summary: "IBL preprocessing: procedural or imported environment cube, mips, GGX prefilter, SH irradiance, DFG table [LIT-01]",
        purpose: Purpose::Runtime,
        entry_points: &[
            compute("generate"),
            compute("equirect_to_cube"),
            compute("downsample"),
            compute("prefilter"),
            compute("irradiance_sh"),
            compute("dfg"),
        ],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
];
