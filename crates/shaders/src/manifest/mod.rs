//! The shader manifest: every WGSL file of the project, embedded, with what it contains.
//!
//! The manifest is split into [`Group`]s, one per owner, all listed in [`GROUPS`]: the core group
//! in this file (`lib/`, `post/`, `test/`) and the UI group in `ui.rs` (`shaders/ui/`, owned by the
//! UI work). A group's `sources` embed its files (`include_str!`, no build script); its `libraries`
//! list the include-only files and the functions they provide; its `shaders` list the files that
//! become modules, with their entry points and stages, the override constants and the values each
//! is specialised with (`[PERF-R5]`), and the capability-flag sets they are composed with.
//! [`sources`], [`libraries`] and [`shaders`] iterate over every group. The validation
//! (`crate::validate`, native) and the tests compose and compile every shader of every group for
//! every flag set and every combination of override values `[AC-WP05-1]`.
//!
//! Adding a shader: put the file under `shaders/`, add `source!("dir/file.wgsl")` to its group's
//! `sources` and a [`Library`] or [`ShaderDesc`] entry to the same group; the manifest tests fail
//! until both are done (and `tools/check-tree.sh` checks the file is named under
//! `crates/shaders/src`). A new group is a new file here plus one line in [`GROUPS`].

/// One embedded WGSL file.
#[derive(Debug, Clone, Copy)]
pub struct SourceFile {
    /// The path under `shaders/`, `/`-separated.
    pub path: &'static str,
    /// The file's text.
    pub text: &'static str,
}

/// Embeds `shaders/<path>` under `path`.
macro_rules! source {
    ($path:literal) => {
        $crate::manifest::SourceFile {
            path: $path,
            text: include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../shaders/", $path)),
        }
    };
}

mod bake;
mod deform;
mod eye;
mod groom;
mod hair;
mod render;
mod skin;
mod texture;
mod ui;

/// The shaders of one owner.
#[derive(Debug, Clone, Copy)]
pub struct Group {
    /// The owner, for messages (`"core"`, `"ui"`).
    pub name: &'static str,
    /// Every file of the group.
    pub sources: &'static [SourceFile],
    /// Its include-only files.
    pub libraries: &'static [Library],
    /// Its shaders.
    pub shaders: &'static [ShaderDesc],
}

/// Every group of the manifest.
pub static GROUPS: &[&Group] = &[
    &CORE,
    &ui::GROUP,
    &render::GROUP,
    &texture::GROUP,
    &deform::GROUP,
    &skin::GROUP,
    &bake::GROUP,
    &groom::GROUP,
    &hair::GROUP,
    &eye::GROUP,
];

/// Every embedded WGSL file, group by group.
pub fn sources() -> impl Iterator<Item = &'static SourceFile> {
    GROUPS.iter().flat_map(|group| group.sources)
}

/// Every shared library, group by group.
pub fn libraries() -> impl Iterator<Item = &'static Library> {
    GROUPS.iter().flat_map(|group| group.libraries)
}

/// Every shader that becomes a module, group by group.
pub fn shaders() -> impl Iterator<Item = &'static ShaderDesc> {
    GROUPS.iter().flat_map(|group| group.shaders)
}

/// The core group: the shared libraries, the post passes and the test kernels.
pub static CORE: Group =
    Group { name: "core", sources: CORE_SOURCES, libraries: CORE_LIBRARIES, shaders: CORE_SHADERS };

/// The files of the core group.
static CORE_SOURCES: &[SourceFile] = &[
    source!("lib/math.wgsl"),
    source!("lib/color.wgsl"),
    source!("lib/brdf.wgsl"),
    source!("lib/sampling.wgsl"),
    source!("lib/packing.wgsl"),
    source!("post/tonemap.wgsl"),
    source!("test/harness.wgsl"),
    source!("test/math.wgsl"),
    source!("test/color.wgsl"),
    source!("test/brdf.wgsl"),
    source!("test/sampling.wgsl"),
    source!("test/packing.wgsl"),
];

/// The text of the embedded file `path` (under `shaders/`).
pub fn source(path: &str) -> Option<&'static str> {
    sources().find(|file| file.path == path).map(|file| file.text)
}

/// Who uses a file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// Shipped and used by the renderer, UI or runtime.
    Runtime,
    /// Only the GPU test harness of this crate uses it.
    Test,
}

/// An include-only file of shared functions.
#[derive(Debug, Clone, Copy)]
pub struct Library {
    /// The path under `shaders/`.
    pub path: &'static str,
    /// What it is for.
    pub summary: &'static str,
    /// The functions it defines (checked against the file in the tests).
    pub functions: &'static [&'static str],
    /// Who uses it.
    pub purpose: Purpose,
}

/// The shared WGSL libraries of the core group.
static CORE_LIBRARIES: &[Library] = &[
    Library {
        path: "lib/math.wgsl",
        summary: "constants (PI, TAU, HALF_PI, INV_PI, INV_TAU, F32_MAX), small helpers, orthonormal bases, quaternions",
        functions: &[
            "sq",
            "max3",
            "min3",
            "remap01",
            "safe_normalize",
            "orthonormal_basis",
            "to_local",
            "from_local",
            "quat_mul",
            "quat_rotate",
        ],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "lib/color.wgsl",
        summary: "sRGB transfer, Rec.709 luminance, EV100 exposure, the PBR Neutral / ACES fitted / clamp tone mappers",
        functions: &[
            "luminance",
            "srgb_to_linear",
            "linear_to_srgb",
            "srgb_to_linear_rgb",
            "linear_to_srgb_rgb",
            "exposure_from_ev100",
            "tone_map_clamp",
            "pbr_neutral",
            "aces_rrt_odt_fit",
            "aces_fitted",
            "tone_map",
        ],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "lib/brdf.wgsl",
        summary: "GGX, height-correlated Smith, Schlick and exact Fresnel, dual lobe, DFG integral, energy compensation",
        functions: &[
            "alpha_from_roughness",
            "d_ggx",
            "g1_smith_ggx",
            "v_smith_ggx_correlated",
            "schlick_weight",
            "f_schlick",
            "f_schlick_scalar",
            "f0_from_ior",
            "f_dielectric",
            "specular_ggx",
            "dual_lobe_dv",
            "diffuse_lambert",
            "pdf_ggx_vndf",
            "integrate_dfg",
            "specular_energy_compensation",
        ],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "lib/sampling.wgsl",
        summary: "Hammersley, R2, Cranley-Patterson, interleaved gradient noise, blue-noise hook, hemisphere and GGX warps",
        functions: &[
            "fixed_to_unit",
            "radical_inverse_vdc",
            "hammersley",
            "r2_sequence",
            "cranley_patterson",
            "interleaved_gradient_noise",
            "blue_noise_sample",
            "sample_disk_concentric",
            "sample_uniform_sphere",
            "sample_uniform_hemisphere",
            "sample_cosine_hemisphere",
            "pdf_cosine_hemisphere",
            "sample_ggx_ndf",
            "sample_ggx_vndf",
        ],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "lib/packing.wgsl",
        summary: "normalised integers, 10-10-10-2, half pairs, octahedral normals and tangents (bit-exact with humanart_math::pack)",
        functions: &[
            "is_nan_f32",
            "quantize_snorm",
            "quantize_unorm",
            "dequantize_snorm",
            "pack_unorm_10_10_10_2",
            "unpack_unorm_10_10_10_2",
            "pack_snorm_10_10_10_2",
            "unpack_snorm_10_10_10_2",
            "pack4x16float",
            "unpack4x16float",
            "sign_not_zero",
            "oct_encode",
            "oct_decode",
            "oct_quantize",
            "pack_oct16",
            "unpack_oct16",
            "pack_oct8",
            "unpack_oct8",
            "pack_tangent",
            "unpack_tangent",
        ],
        purpose: Purpose::Runtime,
    },
    Library {
        path: "test/harness.wgsl",
        summary: "the bindings of the GPU test kernels (uniform params, input and output word arrays)",
        functions: &["input_f32", "store_f32"],
        purpose: Purpose::Test,
    },
];

/// The pipeline stage of an entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// `@vertex`.
    Vertex,
    /// `@fragment`.
    Fragment,
    /// `@compute`.
    Compute,
}

impl Stage {
    /// `"vertex"`, `"fragment"` or `"compute"`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
            Self::Compute => "compute",
        }
    }
}

/// An entry point of a shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntryPoint {
    /// The function name.
    pub name: &'static str,
    /// Its stage.
    pub stage: Stage,
}

/// The WGSL type of an override constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverrideType {
    /// `bool` (values 0 and 1).
    Bool,
    /// `u32`.
    U32,
    /// `i32`.
    I32,
    /// `f32`.
    F32,
}

impl OverrideType {
    /// The WGSL spelling.
    pub const fn wgsl(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
        }
    }

    /// Whether `value` is a value of this type (as wgpu passes it, an `f64`).
    pub fn accepts(self, value: f64) -> bool {
        match self {
            Self::Bool => value == 0.0 || value == 1.0,
            Self::U32 => value.fract() == 0.0 && (0.0..=f64::from(u32::MAX)).contains(&value),
            Self::I32 => value.fract() == 0.0 && (f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&value),
            Self::F32 => value.is_finite() && value.abs() <= f64::from(f32::MAX),
        }
    }
}

/// An override constant of a shader and the values it is specialised with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Override {
    /// The WGSL name.
    pub name: &'static str,
    /// The declared type.
    pub ty: OverrideType,
    /// The values pipelines are built with; the first is the default variant.
    pub values: &'static [f64],
}

/// The attachments render pipelines of a shader are validated with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Targets {
    /// One format per `@location` output of the fragment entry points.
    pub color: &'static [wgpu::TextureFormat],
    /// The depth attachment, if the pass has one (reversed-Z, `GreaterEqual`).
    pub depth: Option<wgpu::TextureFormat>,
}

impl Targets {
    /// No attachments: compute-only shaders.
    pub const NONE: Self = Self { color: &[], depth: None };
}

/// The most override constants one shader may declare.
pub const MAX_OVERRIDES: usize = 8;

/// One override value per constant of a variant, as wgpu takes them.
pub type Constants<'a> = &'a [(&'static str, f64)];

/// A shader that becomes a module.
#[derive(Debug, Clone, Copy)]
pub struct ShaderDesc {
    /// The path under `shaders/`.
    pub path: &'static str,
    /// What it does.
    pub summary: &'static str,
    /// Who uses it.
    pub purpose: Purpose,
    /// Its entry points.
    pub entry_points: &'static [EntryPoint],
    /// Its override constants, at most [`MAX_OVERRIDES`].
    pub overrides: &'static [Override],
    /// The capability-flag sets it is composed with; `&[&[]]` for none.
    pub flag_sets: &'static [&'static [&'static str]],
    /// The attachments of its render pipelines.
    pub targets: Targets,
}

impl ShaderDesc {
    /// The number of override combinations: the product of the value counts (1 without overrides).
    pub fn combination_count(&self) -> usize {
        self.overrides.iter().fold(1_usize, |n, o| n.saturating_mul(o.values.len()))
    }

    /// Writes combination `index` (`0..combination_count()`, the first override varying fastest)
    /// into `out` and returns the filled part; `None` for an index out of range or more than
    /// [`MAX_OVERRIDES`] overrides. Combination 0 is every override's first value. No allocation.
    pub fn combination<'b>(
        &self,
        index: usize,
        out: &'b mut [(&'static str, f64); MAX_OVERRIDES],
    ) -> Option<&'b [(&'static str, f64)]> {
        if index >= self.combination_count() || self.overrides.len() > MAX_OVERRIDES {
            return None;
        }
        let mut rest = index;
        for (slot, o) in out.iter_mut().zip(self.overrides) {
            let count = o.values.len();
            *slot = (o.name, *o.values.get(rest % count)?);
            rest /= count;
        }
        out.get(..self.overrides.len())
    }

    /// The entry point called `name`.
    pub fn entry_point(&self, name: &str) -> Option<&EntryPoint> {
        self.entry_points.iter().find(|e| e.name == name)
    }

    /// The embedded text of the root file.
    pub fn source(&self) -> Option<&'static str> {
        source(self.path)
    }
}

/// Every flag set of a shader without `#ifdef`s.
pub const NO_FLAGS: &[&[&str]] = &[&[]];

/// A compute entry point.
pub const fn compute(name: &'static str) -> EntryPoint {
    EntryPoint { name, stage: Stage::Compute }
}

/// The shaders of the core group.
static CORE_SHADERS: &[ShaderDesc] = &[
    ShaderDesc {
        path: "post/tonemap.wgsl",
        summary: "display transform: exposure, tone mapper, sRGB encoding, full-screen triangle",
        purpose: Purpose::Runtime,
        entry_points: &[
            EntryPoint { name: "vs_fullscreen", stage: Stage::Vertex },
            EntryPoint { name: "fs_tonemap", stage: Stage::Fragment },
        ],
        overrides: &[
            Override { name: "TONE_MAPPER", ty: OverrideType::U32, values: &[0.0, 1.0, 2.0] },
            Override { name: "ENCODE_SRGB", ty: OverrideType::Bool, values: &[1.0, 0.0] },
        ],
        flag_sets: NO_FLAGS,
        targets: Targets { color: &[wgpu::TextureFormat::Rgba8Unorm], depth: None },
    },
    ShaderDesc {
        path: "test/math.wgsl",
        summary: "GPU test kernels for lib/math.wgsl",
        purpose: Purpose::Test,
        entry_points: &[compute("quaternions"), compute("basis")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "test/color.wgsl",
        summary: "GPU test kernels for lib/color.wgsl",
        purpose: Purpose::Test,
        entry_points: &[compute("transfer"), compute("tone_mappers")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "test/brdf.wgsl",
        summary: "GPU test kernels for lib/brdf.wgsl: white furnace, DFG, Fresnel",
        purpose: Purpose::Test,
        entry_points: &[compute("furnace_rows"), compute("dfg"), compute("fresnel")],
        overrides: &[
            Override { name: "WORKGROUP_SIZE", ty: OverrideType::U32, values: &[64.0, 128.0, 256.0] },
            Override { name: "SAMPLE_COUNT", ty: OverrideType::U32, values: &[1024.0, 64.0, 4096.0] },
        ],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "test/sampling.wgsl",
        summary: "GPU test kernels for lib/sampling.wgsl",
        purpose: Purpose::Test,
        entry_points: &[compute("sequences"), compute("warps")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
    ShaderDesc {
        path: "test/packing.wgsl",
        summary: "GPU test kernels for lib/packing.wgsl",
        purpose: Purpose::Test,
        entry_points: &[compute("encode"), compute("decode")],
        overrides: &[],
        flag_sets: NO_FLAGS,
        targets: Targets::NONE,
    },
];

/// The manifest entry of the shader `path`.
pub fn shader(path: &str) -> Option<&'static ShaderDesc> {
    shaders().find(|s| s.path == path)
}

/// The library `path`.
pub fn library(path: &str) -> Option<&'static Library> {
    libraries().find(|l| l.path == path)
}

/// The preprocessor flag a capability sets, for the capabilities shaders branch on with `#ifdef`;
/// `None` for capabilities that only change host code.
pub const fn capability_flag(capability: humanart_gpu::Capability) -> Option<&'static str> {
    match capability {
        humanart_gpu::Capability::ShaderF16 => Some("SHADER_F16"),
        humanart_gpu::Capability::Subgroups => Some("SUBGROUPS"),
        humanart_gpu::Capability::Float32Filterable => Some("FLOAT32_FILTERABLE"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess;

    /// Every `.wgsl` file on disk under `shaders/`, as paths relative to it.
    fn files_on_disk() -> Vec<String> {
        fn walk(dir: &std::path::Path, root: &std::path::Path, out: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(&path, root, out);
                } else if path.extension().is_some_and(|e| e == "wgsl") {
                    let rel = path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
                    out.push(rel);
                }
            }
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders");
        let mut out = Vec::new();
        walk(&root, &root, &mut out);
        out.sort();
        out
    }

    #[test]
    fn plat_04_every_file_on_disk_is_embedded_and_described_once() {
        let mut embedded: Vec<String> = sources().map(|f| f.path.to_owned()).collect();
        embedded.sort();
        assert_eq!(files_on_disk(), embedded, "the groups must list exactly the files under shaders/");
        for group in GROUPS {
            for file in group.sources {
                let described = group.libraries.iter().any(|l| l.path == file.path)
                    || group.shaders.iter().any(|s| s.path == file.path);
                assert!(described, "{} is described outside its group {}", file.path, group.name);
            }
        }
        embedded.dedup();
        assert_eq!(embedded.len(), sources().count(), "a file is embedded twice");
        for file in sources() {
            assert!(preprocess::is_shader_path(file.path), "{}", file.path);
            let libs = libraries().filter(|l| l.path == file.path).count();
            let shaders = shaders().filter(|s| s.path == file.path).count();
            assert_eq!(libs + shaders, 1, "{} must be exactly one library or one shader", file.path);
            assert!(!file.text.contains('\r'), "{} must use LF line endings", file.path);
        }
    }

    /// The word `name` followed by `(` (a definition `fn name(`).
    fn defines(text: &str, name: &str) -> bool {
        text.contains(&format!("fn {name}("))
    }

    #[test]
    fn libraries_define_what_they_list() {
        for lib in libraries() {
            let text = source(lib.path).unwrap();
            for function in lib.functions {
                assert!(defines(text, function), "{} does not define {function}", lib.path);
            }
            // Every function the file defines is listed.
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("fn ") {
                    let name = rest.split('(').next().unwrap();
                    assert!(lib.functions.contains(&name), "{} defines unlisted {name}", lib.path);
                }
            }
            if lib.purpose == Purpose::Runtime {
                let attributes = text.lines().any(|l| l.trim_start().starts_with('@'));
                assert!(!attributes, "{} has entry points or bindings", lib.path);
            }
        }
    }

    #[test]
    fn shaders_declare_what_they_list() {
        for shader in shaders() {
            let text = shader.source().unwrap();
            assert!(!shader.entry_points.is_empty(), "{}", shader.path);
            for entry in shader.entry_points {
                assert!(defines(text, entry.name), "{} has no entry point {}", shader.path, entry.name);
            }
            assert!(shader.overrides.len() <= MAX_OVERRIDES);
            for o in shader.overrides {
                assert!(text.contains(&format!("override {}: {}", o.name, o.ty.wgsl())), "{} {}", shader.path, o.name);
                assert!(
                    !o.values.is_empty() && o.values.iter().all(|&v| o.ty.accepts(v)),
                    "{} {}",
                    shader.path,
                    o.name
                );
            }
            let renders = shader.entry_points.iter().any(|e| e.stage == Stage::Fragment);
            assert_eq!(
                renders,
                !shader.targets.color.is_empty(),
                "{} needs targets exactly when it renders",
                shader.path
            );
            assert!(!shader.flag_sets.is_empty());
            for flags in shader.flag_sets {
                assert!(flags.iter().all(|f| preprocess::is_identifier(f)));
            }
            assert!(!shader.summary.is_empty());
        }
    }

    #[test]
    fn perf_r5_combinations_cover_the_product_of_the_values() {
        let tonemap = shader("post/tonemap.wgsl").unwrap();
        assert_eq!(tonemap.combination_count(), 6);
        let mut out = [("", 0.0); MAX_OVERRIDES];
        let mut seen = Vec::new();
        for index in 0..tonemap.combination_count() {
            let c = tonemap.combination(index, &mut out).unwrap().to_vec();
            assert_eq!(c.len(), 2);
            assert_eq!((c[0].0, c[1].0), ("TONE_MAPPER", "ENCODE_SRGB"));
            seen.push((c[0].1, c[1].1));
        }
        assert_eq!(seen[0], (0.0, 1.0), "combination 0 is the defaults");
        seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
        seen.dedup();
        assert_eq!(seen.len(), 6);
        assert!(tonemap.combination(6, &mut out).is_none());
        let plain = shader("test/math.wgsl").unwrap();
        assert_eq!(plain.combination_count(), 1);
        assert_eq!(plain.combination(0, &mut out).unwrap(), &[] as &[(&str, f64)]);
    }

    #[test]
    fn every_file_composes() {
        for file in sources() {
            let composed = preprocess::compose(file.path, &[]).unwrap();
            assert!(composed.line_count() > 0);
            assert!(!composed.source().lines().any(|l| l.trim_start().starts_with('#')), "{}", file.path);
        }
    }

    #[test]
    fn override_types_check_values() {
        assert!(OverrideType::Bool.accepts(1.0) && !OverrideType::Bool.accepts(2.0));
        assert!(
            OverrideType::U32.accepts(4096.0) && !OverrideType::U32.accepts(-1.0) && !OverrideType::U32.accepts(0.5)
        );
        assert!(OverrideType::I32.accepts(-5.0) && !OverrideType::I32.accepts(3e9));
        assert!(
            OverrideType::F32.accepts(0.25)
                && !OverrideType::F32.accepts(f64::NAN)
                && !OverrideType::F32.accepts(1e300)
        );
    }

    #[test]
    fn capability_flags_are_identifiers() {
        for capability in humanart_gpu::Capability::ALL {
            if let Some(flag) = capability_flag(capability) {
                assert!(preprocess::is_identifier(flag));
            }
        }
    }
}
