//! Shader validation `[AC-WP05-1]` `[PLAT-04]`: every shader of the manifest is composed for each of
//! its flag sets and checked
//!
//! 1. without a GPU, by naga (reached through wgpu's re-export): the generic module, the agreement
//!    of its entry points and override declarations with the manifest, and one specialised module
//!    per override combination (each `override` declaration rewritten to a `const` with the
//!    combination's value, so naga's constant evaluation sees the concrete values);
//! 2. on a device, by creating the shader module and one pipeline per entry point (compute) or per
//!    vertex/fragment pair (render) and per override combination inside error scopes, which runs
//!    wgpu's own specialisation and the backend compiler.
//!
//! Native only: naga is not part of the browser build. Not for per-frame use; it allocates freely.

use std::fmt::Write as _;

use humanart_gpu::{Capability, Gpu};
use wgpu::naga;

use crate::error::ShaderError;
use crate::manifest::{self, Constants, MAX_OVERRIDES, OverrideType, ShaderDesc, Stage};
use crate::preprocess::{self, Composed};

/// What a validation run covered.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Report {
    /// Shaders validated.
    pub shaders: usize,
    /// Composed modules (shader x flag set) validated.
    pub modules: usize,
    /// Specialised variants (module x override combination) validated.
    pub variants: usize,
    /// Pipelines created on a device.
    pub pipelines: usize,
    /// Flag sets not tried on a device because it lacks the capability they need.
    pub skipped_flag_sets: usize,
}

/// The naga capabilities a module composed with `flags` may use.
fn naga_capabilities(flags: &[&str]) -> naga::valid::Capabilities {
    // naga's defaults plus what every WebGPU-compliant device has (wgpu grants it through the
    // downlevel flag SHADER_F16_IN_F32): `pack2x16float` / `unpack2x16float`.
    let mut capabilities = naga::valid::Capabilities::default() | naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32;
    if flags.contains(&"SHADER_F16") {
        capabilities |= naga::valid::Capabilities::SHADER_FLOAT16;
    }
    if flags.contains(&"SUBGROUPS") {
        capabilities |= naga::valid::Capabilities::SUBGROUP | naga::valid::Capabilities::SUBGROUP_BARRIER;
    }
    capabilities
}

/// A compile error at naga's `location` in the composed text, mapped back to the source file.
fn compile_error(
    shader: &str,
    composed: &Composed,
    location: Option<naga::SourceLocation>,
    message: String,
) -> ShaderError {
    let at = location.and_then(|l| composed.locate(l.line_number));
    let column = location.map_or(0, |l| l.line_position);
    ShaderError::Compile { shader: shader.to_owned(), at, column, message }
}

/// `error` and its causes, joined by `": "`.
fn chain(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut cause = error.source();
    while let Some(inner) = cause {
        message.push_str(": ");
        message.push_str(&inner.to_string());
        cause = inner.source();
    }
    message
}

/// Parses and validates `text` (the composed source of `shader`, possibly specialised; its lines
/// are those of `composed`).
fn check_text(
    shader: &str,
    composed: &Composed,
    text: &str,
    flags: &[&str],
) -> Result<(naga::Module, naga::valid::ModuleInfo), ShaderError> {
    let module = naga::front::wgsl::parse_str(text)
        .map_err(|e| compile_error(shader, composed, e.location(text), e.message().to_owned()))?;
    let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga_capabilities(flags));
    let info = validator
        .validate(&module)
        .map_err(|e| compile_error(shader, composed, e.location(text), chain(e.as_inner())))?;
    Ok((module, info))
}

/// The manifest stage of a naga stage; `None` for stages the manifest does not use.
fn stage_of(stage: naga::ShaderStage) -> Option<Stage> {
    match stage {
        naga::ShaderStage::Vertex => Some(Stage::Vertex),
        naga::ShaderStage::Fragment => Some(Stage::Fragment),
        naga::ShaderStage::Compute => Some(Stage::Compute),
        _ => None,
    }
}

/// The manifest type of a naga scalar type.
fn override_type(module: &naga::Module, ty: naga::Handle<naga::Type>) -> Option<OverrideType> {
    match module.types.get_handle(ty).ok()?.inner {
        naga::TypeInner::Scalar(s) if s == naga::Scalar::BOOL => Some(OverrideType::Bool),
        naga::TypeInner::Scalar(s) if s == naga::Scalar::U32 => Some(OverrideType::U32),
        naga::TypeInner::Scalar(s) if s == naga::Scalar::I32 => Some(OverrideType::I32),
        naga::TypeInner::Scalar(s) if s == naga::Scalar::F32 => Some(OverrideType::F32),
        _ => None,
    }
}

/// Checks that `module` has exactly the manifest's entry points (name and stage) and override
/// constants (name and type).
pub fn check_interface(desc: &ShaderDesc, module: &naga::Module) -> Result<(), ShaderError> {
    let mismatch = |message: String| ShaderError::Manifest { shader: desc.path.to_owned(), message };
    for entry in &module.entry_points {
        let stage = stage_of(entry.stage);
        match desc.entry_point(&entry.name) {
            Some(listed) if Some(listed.stage) == stage => {}
            Some(listed) => {
                return Err(mismatch(format!(
                    "entry point {} is {:?}, the manifest says {}",
                    entry.name,
                    entry.stage,
                    listed.stage.name()
                )));
            }
            None => return Err(mismatch(format!("entry point {} is not in the manifest", entry.name))),
        }
    }
    for listed in desc.entry_points {
        if !module.entry_points.iter().any(|e| e.name == listed.name) {
            return Err(mismatch(format!("the manifest lists entry point {}, the module has none", listed.name)));
        }
    }
    for (_, o) in module.overrides.iter() {
        let name = o.name.as_deref().unwrap_or("");
        let Some(listed) = desc.overrides.iter().find(|l| l.name == name) else {
            return Err(mismatch(format!("override {name} is not in the manifest")));
        };
        if override_type(module, o.ty) != Some(listed.ty) {
            return Err(mismatch(format!("override {name} is not a {}", listed.ty.wgsl())));
        }
    }
    for listed in desc.overrides {
        if !module.overrides.iter().any(|(_, o)| o.name.as_deref() == Some(listed.name)) {
            return Err(mismatch(format!("the manifest lists override {}, the module has none", listed.name)));
        }
    }
    Ok(())
}

/// The WGSL literal of `value` as a `ty`.
fn literal(ty: OverrideType, value: f64) -> String {
    match ty {
        OverrideType::Bool => (if value == 0.0 { "false" } else { "true" }).to_owned(),
        OverrideType::U32 => format!("{}u", value as u32),
        OverrideType::I32 => format!("{}i", value as i32),
        OverrideType::F32 => format!("{:?}f", value as f32),
    }
}

/// The composed `source` of `desc` with every override declaration replaced by a `const` holding
/// its value in `constants`. Declarations must sit on one line (`override NAME: TYPE...;`, optionally
/// after `@id(n)`); the line count is unchanged, so the line map of the composition still applies.
pub fn specialise(desc: &ShaderDesc, source: &str, constants: Constants<'_>) -> Result<String, ShaderError> {
    let mismatch = |message: String| ShaderError::Manifest { shader: desc.path.to_owned(), message };
    let mut out = String::with_capacity(source.len() + 64);
    let mut replaced = 0_usize;
    for line in source.lines() {
        let trimmed = line.trim_start();
        let declaration = match trimmed.strip_prefix("@id(") {
            Some(rest) => rest.split_once(')').map(|(_, after)| after.trim_start()),
            None => Some(trimmed),
        };
        if let Some(rest) = declaration.and_then(|d| d.strip_prefix("override ")) {
            let name = rest.split(|c: char| c == ':' || c == '=' || c == ';' || c.is_whitespace()).next().unwrap_or("");
            if !trimmed.trim_end().ends_with(';') {
                return Err(mismatch(format!("override {name} must be declared on one line")));
            }
            let listed = desc.overrides.iter().find(|o| o.name == name);
            let value = constants.iter().find(|(n, _)| *n == name);
            let (Some(listed), Some((_, value))) = (listed, value) else {
                return Err(mismatch(format!("override {name} has no value in this combination")));
            };
            let _ = write!(out, "const {name}: {} = {};", listed.ty.wgsl(), literal(listed.ty, *value));
            replaced += 1;
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    if replaced != desc.overrides.len() {
        return Err(mismatch(format!("{replaced} override declarations found, {} listed", desc.overrides.len())));
    }
    Ok(out)
}

/// Validates `desc` with naga for every flag set and override combination, without a GPU.
pub fn validate_naga(desc: &ShaderDesc, report: &mut Report) -> Result<(), ShaderError> {
    let mut values = [("", 0.0); MAX_OVERRIDES];
    for flags in desc.flag_sets {
        let composed = preprocess::compose(desc.path, flags)?;
        let (module, _) = check_text(desc.path, &composed, composed.source(), flags)?;
        check_interface(desc, &module)?;
        report.modules += 1;
        for index in 0..desc.combination_count() {
            let constants = desc.combination(index, &mut values).ok_or_else(|| ShaderError::Manifest {
                shader: desc.path.to_owned(),
                message: format!("more than {MAX_OVERRIDES} overrides"),
            })?;
            let text = specialise(desc, composed.source(), constants)?;
            check_text(desc.path, &composed, &text, flags)?;
            report.variants += 1;
        }
    }
    report.shaders += 1;
    Ok(())
}

/// [`validate_naga`] for every shader of the manifest.
pub fn validate_all_naga() -> Result<Report, ShaderError> {
    let mut report = Report::default();
    for desc in manifest::shaders() {
        validate_naga(desc, &mut report)?;
    }
    Ok(report)
}

/// The capability a preprocessor flag stands for.
fn capability_of(flag: &str) -> Option<Capability> {
    Capability::ALL.into_iter().find(|c| manifest::capability_flag(*c) == Some(flag))
}

/// `NAME=value` pairs for messages.
fn describe(constants: Constants<'_>) -> String {
    constants.iter().map(|(name, value)| format!("{name}={value}")).collect::<Vec<_>>().join(", ")
}

/// Creates the module and every pipeline of every override combination of `desc` on `gpu`, for
/// every flag set whose capabilities the device has. Every pipeline is dropped again.
pub fn validate_on_device(gpu: &Gpu, desc: &ShaderDesc, report: &mut Report) -> Result<(), ShaderError> {
    let device = &gpu.device;
    let failed = |entry_point: String, constants: String, source: humanart_gpu::GpuError| ShaderError::Device {
        shader: desc.path.to_owned(),
        entry_point,
        constants,
        source: Box::new(source),
    };
    let color: Vec<Option<wgpu::ColorTargetState>> = desc
        .targets
        .color
        .iter()
        .map(|&format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL }))
        .collect();
    let depth_stencil = desc.targets.depth.map(|format| wgpu::DepthStencilState {
        format,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    });
    let mut values = [("", 0.0); MAX_OVERRIDES];
    for flags in desc.flag_sets {
        if flags.iter().any(|flag| capability_of(flag).is_some_and(|c| !gpu.has(c))) {
            report.skipped_flag_sets += 1;
            continue;
        }
        let composed = preprocess::compose(desc.path, flags)?;
        let module = humanart_gpu::error::capture(device, "shader module", || {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(desc.path),
                source: wgpu::ShaderSource::Wgsl(composed.source().into()),
            })
        })
        .map_err(|e| failed(String::new(), String::new(), e))?;
        for index in 0..desc.combination_count() {
            let Some(constants) = desc.combination(index, &mut values) else { break };
            let options = || wgpu::PipelineCompilationOptions { constants, zero_initialize_workgroup_memory: true };
            for entry in desc.entry_points.iter().filter(|e| e.stage == Stage::Compute) {
                humanart_gpu::error::capture(device, "compute pipeline", || {
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(entry.name),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry.name),
                        compilation_options: options(),
                        cache: None,
                    })
                })
                .map_err(|e| failed(entry.name.to_owned(), describe(constants), e))?;
                report.pipelines += 1;
            }
            let fragments: Vec<Option<&str>> = {
                let named: Vec<Option<&str>> =
                    desc.entry_points.iter().filter(|e| e.stage == Stage::Fragment).map(|e| Some(e.name)).collect();
                if named.is_empty() { vec![None] } else { named }
            };
            for vertex in desc.entry_points.iter().filter(|e| e.stage == Stage::Vertex) {
                for fragment in &fragments {
                    let label = format!("{} + {}", vertex.name, fragment.unwrap_or("no fragment"));
                    humanart_gpu::error::capture(device, "render pipeline", || {
                        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                            label: Some(&label),
                            layout: None,
                            vertex: wgpu::VertexState {
                                module: &module,
                                entry_point: Some(vertex.name),
                                compilation_options: options(),
                                buffers: &[],
                            },
                            primitive: wgpu::PrimitiveState::default(),
                            depth_stencil: depth_stencil.clone(),
                            multisample: wgpu::MultisampleState::default(),
                            fragment: fragment.map(|name| wgpu::FragmentState {
                                module: &module,
                                entry_point: Some(name),
                                compilation_options: options(),
                                targets: &color,
                            }),
                            multiview_mask: None,
                            cache: None,
                        })
                    })
                    .map_err(|e| failed(label.clone(), describe(constants), e))?;
                    report.pipelines += 1;
                }
            }
            report.variants += 1;
        }
        report.modules += 1;
    }
    report.shaders += 1;
    Ok(())
}

/// [`validate_on_device`] for every shader of the manifest.
pub fn validate_all_on_device(gpu: &Gpu) -> Result<Report, ShaderError> {
    let mut report = Report::default();
    for desc in manifest::shaders() {
        validate_on_device(gpu, desc, &mut report)?;
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{EntryPoint, Override, Purpose, Targets};

    const DESC: ShaderDesc = ShaderDesc {
        path: "inline.wgsl",
        summary: "test",
        purpose: Purpose::Test,
        entry_points: &[EntryPoint { name: "main", stage: Stage::Compute }],
        overrides: &[
            Override { name: "SIZE", ty: OverrideType::U32, values: &[64.0, 0.0] },
            Override { name: "ON", ty: OverrideType::Bool, values: &[1.0] },
            Override { name: "SCALE", ty: OverrideType::F32, values: &[0.5] },
        ],
        flag_sets: &[&[]],
        targets: Targets::NONE,
    };

    const SOURCE: &str = "override SIZE: u32 = 64u;\n@id(3) override ON: bool;\noverride SCALE: f32 = 1.0;\n\
        @compute @workgroup_size(SIZE)\nfn main() { if (ON) { _ = SCALE; } }\n";

    fn composed(text: &'static str) -> Composed {
        preprocess::compose_with("inline.wgsl", &[], move |_| Some(text)).unwrap()
    }

    #[test]
    fn specialise_rewrites_every_override_and_keeps_lines() {
        let constants = [("SIZE", 64.0), ("ON", 1.0), ("SCALE", 0.5)];
        let text = specialise(&DESC, SOURCE, &constants).unwrap();
        assert!(text.starts_with("const SIZE: u32 = 64u;\nconst ON: bool = true;\nconst SCALE: f32 = 0.5f;\n"));
        assert_eq!(text.lines().count(), SOURCE.lines().count());
        let c = composed(SOURCE);
        let (module, _) = check_text("inline.wgsl", &c, &text, &[]).unwrap();
        assert!(module.overrides.is_empty());
        let missing = specialise(&DESC, SOURCE, &constants[..2]).unwrap_err();
        assert!(matches!(missing, ShaderError::Manifest { .. }), "{missing}");
    }

    #[test]
    fn ac_wp05_1_a_bad_combination_is_caught_without_a_gpu() {
        // SIZE = 0 is a valid override value for the generic module but an invalid workgroup size.
        let c = composed(SOURCE);
        let (module, _) = check_text("inline.wgsl", &c, c.source(), &[]).unwrap();
        check_interface(&DESC, &module).unwrap();
        let text = specialise(&DESC, c.source(), &[("SIZE", 0.0), ("ON", 1.0), ("SCALE", 0.5)]).unwrap();
        let error = check_text("inline.wgsl", &c, &text, &[]).unwrap_err();
        assert!(matches!(error, ShaderError::Compile { .. }), "{error}");
    }

    #[test]
    fn compile_errors_point_at_the_source_file_and_line() {
        let table: &[(&str, &str)] = &[
            ("main.wgsl", "#include \"lib/a.wgsl\"\nfn main_fn() -> f32 { return a(); }\n"),
            ("lib/a.wgsl", "// line 1\n// line 2\nfn a() -> f32 { return undefined_name; }\n"),
        ];
        let c = preprocess::compose_with("main.wgsl", &[], |p| table.iter().find(|(n, _)| *n == p).map(|(_, t)| *t))
            .unwrap();
        let error = check_text("main.wgsl", &c, c.source(), &[]).unwrap_err();
        let ShaderError::Compile { at: Some(at), column, .. } = &error else { panic!("{error}") };
        assert_eq!((at.file.as_str(), at.line), ("lib/a.wgsl", 3));
        assert!(*column > 1);
        // Validation (not parse) errors map too: a runtime-sized array in the private space, line 2.
        let table2: &[(&str, &str)] = &[("m.wgsl", "// x\nvar<private> x: array<f32>;\n")];
        let c2 =
            preprocess::compose_with("m.wgsl", &[], |p| table2.iter().find(|(n, _)| *n == p).map(|(_, t)| *t)).unwrap();
        let error = check_text("m.wgsl", &c2, c2.source(), &[]).unwrap_err();
        assert!(matches!(&error, ShaderError::Compile { at: Some(at), .. } if at.line == 2), "{error}");
    }

    #[test]
    fn interface_mismatches_are_reported() {
        let c = composed(SOURCE);
        let (module, _) = check_text("inline.wgsl", &c, c.source(), &[]).unwrap();
        let wrong_stage = ShaderDesc { entry_points: &[EntryPoint { name: "main", stage: Stage::Vertex }], ..DESC };
        assert!(matches!(check_interface(&wrong_stage, &module), Err(ShaderError::Manifest { .. })));
        let extra = ShaderDesc {
            entry_points: &[
                EntryPoint { name: "main", stage: Stage::Compute },
                EntryPoint { name: "other", stage: Stage::Compute },
            ],
            ..DESC
        };
        assert!(check_interface(&extra, &module).is_err());
        let wrong_type = ShaderDesc {
            overrides: &[
                Override { name: "SIZE", ty: OverrideType::I32, values: &[1.0] },
                Override { name: "ON", ty: OverrideType::Bool, values: &[1.0] },
                Override { name: "SCALE", ty: OverrideType::F32, values: &[0.5] },
            ],
            ..DESC
        };
        assert!(check_interface(&wrong_type, &module).is_err());
        let missing = ShaderDesc { overrides: &DESC.overrides[..2], ..DESC };
        assert!(check_interface(&missing, &module).is_err());
    }

    #[test]
    fn literals_are_wgsl() {
        assert_eq!(literal(OverrideType::Bool, 0.0), "false");
        assert_eq!(literal(OverrideType::U32, 4096.0), "4096u");
        assert_eq!(literal(OverrideType::I32, -3.0), "-3i");
        assert_eq!(literal(OverrideType::F32, 1e-5), "1e-5f");
        assert_eq!(literal(OverrideType::F32, 2.0), "2.0f");
    }
}
