//! The eyes-and-mouth group of the manifest: the WGSL under `shaders/eye/` and `shaders/mouth/`,
//! owned by the eye and mouth work (`humanart-render::eye`, `humanart-render::mouth`, WP-16:
//! `[EYE-01]`..`[EYE-07]`, `[MTH-01]`..`[MTH-04]`).
//!
//! The shading files are include-only libraries of the geometry pass (`render/geometry.wgsl`): no
//! pipeline of their own, no `#ifdef`s or override constants (`[PERF-R4]`). The guard kernel
//! mirrors the CPU penetration guard into the GPU deformation's outputs (`[EVAL-08]`). The probe
//! is a test kernel that runs the analytic eye against its f64 reference (`[AC-WP16-2]`).

use super::{Group, Library, NO_FLAGS, Purpose, ShaderDesc, SourceFile, Targets, compute};

/// The eyes-and-mouth group, listed in [`super::GROUPS`].
pub static GROUP: Group = Group { name: "eyes-mouth", sources: SOURCES, libraries: LIBRARIES, shaders: SHADERS };

/// Every file under `shaders/eye/` and `shaders/mouth/`.
static SOURCES: &[SourceFile] = &[
    source!("eye/features.wgsl"),
    source!("eye/optics.wgsl"),
    source!("eye/eye.wgsl"),
    source!("eye/probe.wgsl"),
    source!("eye/guard.wgsl"),
    source!("mouth/mouth.wgsl"),
];

/// The include-only files.
static LIBRARIES: &[Library] = &[
    Library {
        path: "eye/features.wgsl",
        summary: "the eye and mouth table of the geometry pass (object slots, 16-vec4 entries), value noise, and the lighting of their wet dielectric surfaces [EYE-01] [MTH-01]",
        functions: &[
            "feature_slot",
            "feature_vec",
            "feature_float",
            "feature_kind",
            "feature_hash",
            "feature_noise",
            "feature_fbm",
            "feature_lit",
        ],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "eye/optics.wgsl",
        summary: "the analytic eyeball and the two-surface refraction through the cornea onto the iris plane, f32 twin of humanart_model::eye::EyeGeometry [EYE-01] [EYE-02]",
        functions: &["eye_shape", "eye_sphere_hits", "eye_height", "eye_through_cornea", "eye_trace"],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "eye/eye.wgsl",
        summary: "eye shading: traced sclera / cornea / refracted iris, procedural iris and sclera layers, caruncle, wet tear film, limbus shadow on the iris, catchlights, lid and lash shadow, occlusion film and tear meniscus composited over the traced eyeball [EYE-02]..[EYE-05]",
        functions: &[
            "eye_pick",
            "eye_shape_of",
            "eye_head_local",
            "eye_lid_openness",
            "eye_lid_factors",
            "eye_sclera_albedo",
            "eye_iris_albedo",
            "eye_iris_light",
            "eye_mirror_light",
            "eye_film_normal",
            "eye_film",
            "eye_ball",
            "eye_shade",
        ],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "mouth/mouth.wgsl",
        summary: "teeth, gums, tongue and mouth-bag shading with the lip-opening form-factor occlusion of the cavity [MTH-01]..[MTH-04]",
        functions: &["mouth_occlusion", "mouth_teeth", "mouth_gums", "mouth_tongue", "mouth_bag", "mouth_shade"],
        purpose: Purpose::Runtime,
    },
];

/// The guard kernel and the test kernel.
static SHADERS: &[ShaderDesc] = &[
    ShaderDesc {
        path: "eye/guard.wgsl",
        summary: "writes the CPU penetration guard's corrected positions and normals over the GPU deformation's outputs [EVAL-08]",
        purpose: Purpose::Runtime,
        entry_points: &[compute("apply_fixes")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "eye/probe.wgsl",
        summary: "GPU test kernel for eye/optics.wgsl: one traced ray per case, compared with the f64 reference [AC-WP16-2]",
        purpose: Purpose::Test,
        entry_points: &[compute("probe")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
];
