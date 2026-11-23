//! [`GpuDeformer`]: the evaluation of one character on the GPU (spec 6.2 steps 1 to 6, WP-10).
//!
//! **Set-up** ([`GpuDeformer::new`]): the template's tables ([`DeformTables`]) are uploaded once
//! into persistent buffers tagged [`Category::Deformation`] `[GPU-01]` `[GPU-05]`; every pipeline
//! and bind group is created once `[PERF-R4]`; output buffers (positions, previous positions,
//! normals, tangents, strain) are allocated at their final size.
//!
//! **Per frame**:
//!
//! 1. [`GpuDeformer::update`] takes the evaluation input (the snapshot's weights and pose) and
//!    diffs it against what the GPU already holds, on the CPU, without allocating `[PERF-R1]`:
//!    - shape and sculpt-layer weights that changed become *scatter jobs* that add only the weight
//!      difference of those components to the rest shape `[EVAL-03]`; more changes than job slots,
//!      a replaced layer or [`DeformSettings::refresh_interval`] incremental updates since the last
//!      full pass make it a *full re-accumulation* instead, which bounds the drift;
//!    - expression weights, combination activations `[EXP-02]` and pose-corrective activations
//!      `[SKIN-02]` become one weight per dynamic component;
//!    - joint bind positions (joint regressor `[MOD-07]`) and posed frames are computed with
//!      [`JointFrames`], the CPU reference's own code, and uploaded as five `vec4` per joint;
//!    - only what changed is written to the GPU (a few hundred bytes for one slider).
//! 2. [`GpuDeformer::encode`] records the work into the caller's encoder: the previous positions
//!    are kept for motion vectors, then one compute pass runs the accumulation (full gather or the
//!    scatter jobs), skinning, the triangle and vertex passes of the normals, the strain passes, and
//!    the reduced levels of detail. Nothing is recorded when nothing changed.
//!
//! Outputs stay on the GPU ([`DeformOutputs`]); the renderer pulls them as storage buffers
//! `[EVAL-05]`. Reading them back goes through the readback ring `[PERF-R3]`.
//!
//! **Not on the GPU:** the rig logic (`[EXP-03]`) and the joint hierarchy are evaluated on the CPU:
//! both are short sequential chains (each node and joint depends on the previous ones) that cost a
//! few microseconds, and their results are tiny uploads.

use std::sync::Arc;

use humanart_gpu::Gpu;
use humanart_gpu::cache::PipelineCache;
use humanart_gpu::pod::slice_as_bytes;
use humanart_gpu::profiler::GpuProfiler;
use humanart_gpu::resources::{BufferDesc, BufferId, Category, Resources, Tag};
use humanart_math::{Quat, Vec3};
use humanart_model::Template;
use humanart_model::eval::TwistRig;
use humanart_model::eval::evaluator::{FULL_REFRESH_INTERVAL, MAX_WEIGHT};
use humanart_model::eval::{EvalError, EvalInput, JointFrames, SculptLayer, SkinningMode};
use humanart_model::expression::pose_space::PoseSpace;

use super::error::DeformError;
use super::tables::{DeformTables, LodTables};

/// Bytes between two uniform parameter slots (the largest uniform offset alignment WebGPU allows).
const SLOT: u64 = 256;
/// Scatter jobs one frame can hold; more changed components make a full re-accumulation.
pub const JOB_SLOTS: usize = 64;
/// The most sculpt layers one deformer holds on the GPU.
pub const MAX_GPU_LAYERS: usize = 32;
/// The most reduced levels of detail.
const MAX_LODS: usize = humanart_model::lod::MAX_LOD_LEVELS;
/// Parameter slots: gather, skin, normals, strain, two per level of detail, then the jobs.
const SLOT_GATHER: u32 = 0;
const SLOT_SKIN: u32 = 1;
const SLOT_NORMALS: u32 = 2;
const SLOT_STRAIN: u32 = 3;
const SLOT_LOD: u32 = 4;
const SLOT_JOBS: u32 = SLOT_LOD + 2 * MAX_LODS as u32;
const SLOT_COUNT: usize = SLOT_JOBS as usize + JOB_SLOTS;
/// Invocations per workgroup of the 1D passes (`WORKGROUP` in `deform/common.wgsl`).
const WORKGROUP: u32 = 64;
/// The largest workgroup count of one dimension under the default limits.
const MAX_GROUPS: u32 = 65_535;
/// Words per joint: three matrix columns with the translation in `w`, the dual quaternion.
const JOINT_VEC4S: usize = 5;
/// The smallest buffer made: one `vec4` element, so an empty table still binds.
const MIN_BUFFER: u64 = 16;

/// Tuning of a [`GpuDeformer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeformSettings {
    /// Incremental rest-shape updates after which the rest shape is re-accumulated from scratch
    /// `[EVAL-03]` (the CPU reference uses [`FULL_REFRESH_INTERVAL`]). At least 1.
    pub refresh_interval: u32,
    /// Changed components per frame up to which the update is incremental, at most
    /// [`JOB_SLOTS`]; above it a full re-accumulation is cheaper.
    pub max_jobs: usize,
    /// Whether to deform the template's reduced levels of detail too.
    pub lods: bool,
}

impl Default for DeformSettings {
    fn default() -> Self {
        Self { refresh_interval: FULL_REFRESH_INTERVAL, max_jobs: JOB_SLOTS, lods: true }
    }
}

/// What [`GpuDeformer::update`] scheduled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UpdatePlan {
    /// The rest shape is re-accumulated from scratch.
    pub full: bool,
    /// Scatter jobs (changed shape components and layers) of an incremental update.
    pub jobs: usize,
    /// Skinning, normals and strain run (something changed).
    pub deform: bool,
}

/// The compute passes of the deformation, in recording order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pass {
    /// Full rest-shape gather.
    Gather,
    /// Incremental scatter of one delta component.
    ScatterComponent,
    /// Scatter of one sculpt layer.
    ScatterLayer,
    /// Expressions, correctives and skinning.
    Skin,
    /// Triangle normals, tangents and corner angles.
    Triangles,
    /// Vertex normals and tangents.
    Vertices,
    /// Edge strain.
    EdgeStrain,
    /// Per-region own sums.
    RegionSums,
    /// Per-region subtree strain.
    RegionStrain,
    /// Per-vertex strain.
    VertexStrain,
    /// Level-of-detail positions.
    LodTransfer,
}

impl Pass {
    const ALL: [Self; 11] = [
        Self::Gather,
        Self::ScatterComponent,
        Self::ScatterLayer,
        Self::Skin,
        Self::Triangles,
        Self::Vertices,
        Self::EdgeStrain,
        Self::RegionSums,
        Self::RegionStrain,
        Self::VertexStrain,
        Self::LodTransfer,
    ];

    /// `(shader, entry point, bindings)`; a binding is `(index, kind)`.
    const fn layout(self) -> (Shader, &'static str, &'static [(u32, Binding)]) {
        use Binding::{Read, Write};
        match self {
            Self::Gather => {
                (Shader::Accumulate, "gather_rest", &[(1, Write), (2, Read), (3, Read), (4, Read), (5, Read)])
            }
            Self::ScatterComponent => (Shader::Accumulate, "scatter_component", &[(1, Write), (6, Read), (7, Read)]),
            Self::ScatterLayer => (Shader::Accumulate, "scatter_layer", &[(1, Write), (8, Read), (9, Read)]),
            Self::Skin => {
                (Shader::Skin, "skin", &[(1, Read), (2, Read), (3, Read), (4, Read), (5, Read), (6, Read), (7, Write)])
            }
            Self::Triangles => (Shader::Normals, "triangle_frames", &[(1, Read), (2, Read), (3, Read), (4, Write)]),
            Self::Vertices => {
                (Shader::Normals, "vertex_frames", &[(5, Read), (6, Read), (7, Read), (8, Write), (9, Write)])
            }
            Self::EdgeStrain => (Shader::Strain, "edge_strain", &[(1, Read), (2, Read), (3, Read), (4, Write)]),
            Self::RegionSums => (Shader::Strain, "region_sums", &[(5, Read), (6, Read), (7, Read), (8, Write)]),
            Self::RegionStrain => {
                (Shader::Strain, "region_strain", &[(9, Read), (10, Read), (11, Read), (17, Read), (12, Write)])
            }
            Self::VertexStrain => {
                (Shader::Strain, "vertex_strain", &[(5, Read), (13, Read), (14, Read), (15, Read), (16, Write)])
            }
            Self::LodTransfer => (Shader::Lod, "transfer", &[(1, Read), (2, Read), (3, Read), (4, Write)]),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Binding {
    Read,
    Write,
}

/// The deformation shaders, with their manifest paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Shader {
    Accumulate,
    Skin,
    Normals,
    Strain,
    Lod,
}

impl Shader {
    const ALL: [Self; 5] = [Self::Accumulate, Self::Skin, Self::Normals, Self::Strain, Self::Lod];

    const fn path(self) -> &'static str {
        match self {
            Self::Accumulate => "deform/accumulate.wgsl",
            Self::Skin => "deform/skin.wgsl",
            Self::Normals => "deform/normals.wgsl",
            Self::Strain => "deform/strain.wgsl",
            Self::Lod => "deform/lod.wgsl",
        }
    }
}

/// Keys of the pipeline cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Key {
    Shader(Shader),
    Layout(Pass),
    PipelineLayout(Pass),
    Pipeline(Pass),
}

/// The GPU buffers a renderer or a readback reads. Positions and normals are packed f32 triples
/// per vertex (the layout of the scene's position and normal streams), tangents `vec4<f32>` with
/// the handedness in `w`.
#[derive(Clone, Copy, Debug)]
pub struct DeformOutputs<'a> {
    /// Deformed positions (steps 1 to 4).
    pub positions: &'a wgpu::Buffer,
    /// The positions before the last change, for motion vectors `[RND-04]`; equal to
    /// `positions` once a frame passed without change.
    pub prev_positions: &'a wgpu::Buffer,
    /// Unit vertex normals.
    pub normals: &'a wgpu::Buffer,
    /// Unit tangents with handedness.
    pub tangents: &'a wgpu::Buffer,
    /// The rest shape (step 1).
    pub rest: &'a wgpu::Buffer,
    /// Per edge: (strain, rest length).
    pub edge_strain: &'a wgpu::Buffer,
    /// Per region: (stretch, compression) `[EVAL-06]`.
    pub region_strain: &'a wgpu::Buffer,
    /// Per vertex: mean signed strain.
    pub vertex_strain: &'a wgpu::Buffer,
}

/// The GPU buffers of one reduced level of detail.
#[derive(Clone, Copy, Debug)]
pub struct LodOutputs<'a> {
    /// Vertices of the level.
    pub vertex_count: u32,
    /// Deformed positions, packed f32 triples.
    pub positions: &'a wgpu::Buffer,
    /// Unit vertex normals, packed f32 triples.
    pub normals: &'a wgpu::Buffer,
    /// Unit tangents with handedness.
    pub tangents: &'a wgpu::Buffer,
}

/// A stage of the deformation for timestamp queries (see [`GpuDeformer::encode_timed`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Rest-shape accumulation (full or incremental).
    Accumulate,
    /// Expressions, correctives, skinning.
    Skin,
    /// Normals and tangents.
    Normals,
    /// Strain.
    Strain,
    /// Levels of detail.
    Lod,
}

impl Stage {
    /// Every stage, in recording order; a timed encode writes two timestamps per stage at
    /// `2 · index` and `2 · index + 1`.
    pub const ALL: [Self; 5] = [Self::Accumulate, Self::Skin, Self::Normals, Self::Strain, Self::Lod];
}

/// What [`GpuDeformer::prepare`] recorded and what remains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Prepared {
    Nothing,
    CopyOnly,
    Run,
}

/// How an encode is timed.
#[derive(Clone, Copy)]
enum Timing<'q> {
    None,
    Stages(&'q wgpu::QuerySet, u32),
    Total(&'q wgpu::QuerySet, u32),
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Job {
    /// A sculpt layer (else a delta component).
    layer: bool,
    /// Component or layer index.
    index: u32,
    /// First entry (components).
    first: u32,
    /// Entries.
    count: u32,
    /// Weight (difference, or the full weight of a layer in a full pass).
    weight: f32,
}

/// One sculpt layer on the GPU.
#[derive(Debug)]
struct GpuLayer {
    indices: BufferId,
    deltas: BufferId,
    count: u32,
    bind_group: wgpu::BindGroup,
}

/// One reduced level of detail on the GPU.
#[derive(Debug)]
struct GpuLod {
    vertex_count: u32,
    triangle_count: u32,
    positions: wgpu::Buffer,
    normals: wgpu::Buffer,
    tangents: wgpu::Buffer,
    transfer: wgpu::BindGroup,
    triangles: wgpu::BindGroup,
    vertices: wgpu::BindGroup,
}

/// Raw handles of the persistent buffers, for bind groups and copies.
#[derive(Debug)]
struct Raw {
    params: wgpu::Buffer,
    rest: wgpu::Buffer,
    positions: wgpu::Buffer,
    prev_positions: wgpu::Buffer,
    normals: wgpu::Buffer,
    tangents: wgpu::Buffer,
    edge_strain: wgpu::Buffer,
    region_strain: wgpu::Buffer,
    vertex_strain: wgpu::Buffer,
}

/// Bind groups of the fixed passes.
#[derive(Debug)]
struct Groups {
    gather: wgpu::BindGroup,
    scatter: wgpu::BindGroup,
    skin: wgpu::BindGroup,
    triangles: wgpu::BindGroup,
    vertices: wgpu::BindGroup,
    edge_strain: wgpu::BindGroup,
    region_sums: wgpu::BindGroup,
    region_strain: wgpu::BindGroup,
    vertex_strain: wgpu::BindGroup,
}

/// Buffer ids of the buffers written per update, and of everything the deformer made (for
/// [`GpuDeformer::destroy`]).
#[derive(Debug)]
struct Owned {
    params: BufferId,
    shape_weights: BufferId,
    dynamic_weights: BufferId,
    joints: BufferId,
    all: Vec<BufferId>,
}

/// The GPU evaluation of one template. See the module documentation.
#[derive(Debug)]
pub struct GpuDeformer {
    template: Arc<Template>,
    settings: DeformSettings,
    cache: PipelineCache<Key>,
    owned: Owned,
    raw: Raw,
    groups: Groups,
    layers: Vec<GpuLayer>,
    lods: Vec<GpuLod>,
    // Counts.
    vertex_count: u32,
    triangle_count: u32,
    edge_count: u32,
    region_count: u32,
    block_count: u32,
    influences: u32,
    // CPU state of what the GPU holds.
    is_shape: Vec<bool>,
    dynamic_components: Vec<u32>,
    scratch_dynamic: Vec<f32>,
    pose_space: PoseSpace,
    pose_activations: Vec<f32>,
    twist: TwistRig,
    driven_pose: Vec<Quat>,
    component_offsets: Vec<u32>,
    applied_shape: Vec<f32>,
    applied_layers: Vec<f32>,
    dynamic_weights: Vec<f32>,
    joint_words: Vec<[f32; 4]>,
    parents: Vec<Option<u16>>,
    bind_rotations: Vec<Quat>,
    rest_world_positions: Vec<Vec3>,
    pose: Vec<Quat>,
    joints: JointFrames,
    mode: SkinningMode,
    incremental_updates: u32,
    // Work scheduled for the next encode.
    jobs: Vec<Job>,
    full: bool,
    dirty: bool,
    moved: bool,
    params: Vec<u8>,
    params_dirty: bool,
}

/// A uniform layout entry with a dynamic offset.
const fn uniform_entry() -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(16),
        },
        count: None,
    }
}

const fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// The workgroup grid for `items` 1D items (`x` at most [`MAX_GROUPS`]).
fn grid(items: u32) -> (u32, u32) {
    let groups = items.div_ceil(WORKGROUP);
    if groups == 0 {
        return (0, 0);
    }
    let x = groups.min(MAX_GROUPS);
    (x, groups.div_ceil(x))
}

fn dispatch(pass: &mut wgpu::ComputePass<'_>, items: u32) {
    let (x, y) = grid(items);
    if x > 0 {
        pass.dispatch_workgroups(x, y, 1);
    }
}

/// Writes four words into parameter slot `slot` of `mirror`.
fn put_slot(mirror: &mut [u8], slot: u32, words: [u32; 4]) {
    let start = slot as usize * SLOT as usize;
    if let Some(dst) = mirror.get_mut(start..start + 16) {
        for (chunk, w) in dst.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&w.to_le_bytes());
        }
    }
}

const fn mode_word(mode: SkinningMode) -> u32 {
    match mode {
        SkinningMode::ByRegion => 0,
        SkinningMode::Linear => 1,
        SkinningMode::DualQuaternion => 2,
    }
}

/// Checks evaluation weights as the CPU reference does (`Evaluator::check_input`).
fn check_weights(what: &'static str, weights: &[f32], limit: usize) -> Result<(), EvalError> {
    if weights.len() > limit {
        return Err(EvalError::Length { what, limit, got: weights.len() });
    }
    for (i, w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(EvalError::NonFinite { what, index: i });
        }
        if w.abs() > MAX_WEIGHT {
            return Err(EvalError::Range { what, index: i, rule: "magnitude above MAX_WEIGHT" });
        }
    }
    Ok(())
}

/// Buffer creation with accounting and the device's binding limit.
struct Uploader<'r> {
    resources: &'r mut Resources,
    made: Vec<BufferId>,
    max_binding: u64,
}

impl Uploader<'_> {
    fn make(
        &mut self,
        size: u64,
        usage: wgpu::BufferUsages,
        purpose: &'static str,
    ) -> Result<(BufferId, wgpu::Buffer), DeformError> {
        let size = size.max(MIN_BUFFER);
        if usage.contains(wgpu::BufferUsages::STORAGE) && size > self.max_binding {
            return Err(DeformError::TooLarge { what: purpose, value: size, limit: self.max_binding });
        }
        let (id, _) = self.resources.create_buffer(&BufferDesc {
            size,
            usage,
            tag: Tag::shared(Category::Deformation, purpose),
        })?;
        self.made.push(id);
        let raw = self
            .resources
            .buffer(id)
            .ok_or(humanart_gpu::GpuError::Stale { what: "deformation buffer" })?
            .raw()
            .clone();
        Ok((id, raw))
    }

    fn upload(
        &mut self,
        bytes: &[u8],
        usage: wgpu::BufferUsages,
        purpose: &'static str,
    ) -> Result<(BufferId, wgpu::Buffer), DeformError> {
        let (id, raw) = self.make(bytes.len() as u64, usage | wgpu::BufferUsages::COPY_DST, purpose)?;
        if !bytes.is_empty() {
            self.resources.write_buffer(id, 0, bytes)?;
        }
        Ok((id, raw))
    }

    fn table<T: humanart_gpu::Pod>(&mut self, items: &[T], purpose: &'static str) -> Result<wgpu::Buffer, DeformError> {
        Ok(self.upload(slice_as_bytes(items), wgpu::BufferUsages::STORAGE, purpose)?.1)
    }

    fn output(&mut self, size: u64, purpose: &'static str) -> Result<(BufferId, wgpu::Buffer), DeformError> {
        self.make(
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            purpose,
        )
    }
}

fn bind_group(
    device: &wgpu::Device,
    cache: &PipelineCache<Key>,
    pass: Pass,
    params: &wgpu::Buffer,
    buffers: &[&wgpu::Buffer],
) -> Result<wgpu::BindGroup, DeformError> {
    let (_, _, bindings) = pass.layout();
    if bindings.len() != buffers.len() {
        return Err(DeformError::Invalid { what: "a deformation bind group with the wrong buffer count" });
    }
    let layout = cache
        .get_bind_group_layout(&Key::Layout(pass))
        .ok_or(humanart_gpu::GpuError::Stale { what: "deformation layout" })?;
    let mut entries: [Option<wgpu::BindGroupEntry<'_>>; 8] = Default::default();
    entries[0] = Some(wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: params,
            offset: 0,
            size: wgpu::BufferSize::new(16),
        }),
    });
    for (slot, (&(binding, _), buffer)) in entries.iter_mut().skip(1).zip(bindings.iter().zip(buffers)) {
        *slot = Some(wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() });
    }
    let list: Vec<wgpu::BindGroupEntry<'_>> = entries.into_iter().flatten().collect();
    // Layouts and buffers are made by this module and checked above; a validation error would be
    // a bug and reaches the device's uncaptured-error queue `[ERR-05]`.
    Ok(device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("deformation"), layout, entries: &list }))
}

impl GpuDeformer {
    /// Uploads `template` and builds every pipeline and bind group. The neutral state (all weights
    /// 0, rest pose, by-region skinning) is scheduled, so the first [`encode`](Self::encode) fills
    /// every output.
    pub fn new(
        gpu: &Gpu,
        resources: &mut Resources,
        template: Arc<Template>,
        settings: DeformSettings,
    ) -> Result<Self, DeformError> {
        if settings.refresh_interval == 0 || settings.max_jobs > JOB_SLOTS {
            return Err(DeformError::Invalid { what: "deformation settings out of range" });
        }
        let t = DeformTables::new(&template)?;
        if t.region_count > MAX_GROUPS {
            return Err(DeformError::TooLarge {
                what: "regions",
                value: u64::from(t.region_count),
                limit: u64::from(MAX_GROUPS),
            });
        }
        let lod_tables: Vec<LodTables> = if settings.lods {
            template
                .lods()
                .iter()
                .take(MAX_LODS)
                .map(|level| LodTables::new(level, t.vertex_count as usize))
                .collect::<Result<_, _>>()?
        } else {
            Vec::new()
        };

        let mut cache = PipelineCache::new(gpu);
        for shader in Shader::ALL {
            let source = humanart_shaders::compose(shader.path(), &[])?.into_source();
            cache.shader(Key::Shader(shader), shader.path(), &source)?;
        }
        for pass in Pass::ALL {
            let (shader, entry_point, bindings) = pass.layout();
            let mut entries = [uniform_entry(); 8];
            for (slot, &(binding, kind)) in entries.iter_mut().skip(1).zip(bindings) {
                *slot = storage_entry(binding, kind == Binding::Read);
            }
            cache.bind_group_layout(Key::Layout(pass), "deformation", &entries[..=bindings.len()])?;
            cache.pipeline_layout(Key::PipelineLayout(pass), "deformation", &[Some(&Key::Layout(pass))])?;
            cache.compute_pipeline(Key::Pipeline(pass), |parts| {
                Ok(parts.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: Some(parts.pipeline_layout(&Key::PipelineLayout(pass))?),
                    module: parts.shader(&Key::Shader(shader))?,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                }))
            })?;
        }
        cache.mark_warm();

        let v = u64::from(t.vertex_count);
        let max_binding = gpu.limits().max_storage_buffer_binding_size;
        let mut up = Uploader { resources, made: Vec::new(), max_binding };
        let storage = wgpu::BufferUsages::STORAGE;
        let built = (|| -> Result<_, DeformError> {
            let (params_id, params) = up.make(
                SLOT * SLOT_COUNT as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                "deformation parameters",
            )?;
            let template_positions = up.table(&t.template_positions, "template positions")?;
            let uvs = up.table(&t.uvs, "deformation uvs")?;
            let shape_offsets = up.table(&t.shape.slice_offsets, "shape delta slices")?;
            let shape_entries = up.table(&t.shape.entries, "shape delta entries")?;
            let shape_weights_bytes = vec![0_u8; t.component_count as usize * 4];
            let (shape_weights_id, shape_weights) = up.upload(&shape_weights_bytes, storage, "shape weights")?;
            let (dynamic_weights_id, dynamic_weights) = up.upload(&shape_weights_bytes, storage, "dynamic weights")?;
            let component_indices = up.table(&t.component_indices, "delta indices")?;
            let component_deltas = up.table(&t.component_deltas, "delta values")?;
            let dynamic_offsets = up.table(&t.dynamic.slice_offsets, "dynamic delta slices")?;
            let dynamic_entries = up.table(&t.dynamic.entries, "dynamic delta entries")?;
            let joint_bytes = vec![0_u8; t.joint_count as usize * JOINT_VEC4S * 16];
            let (joints_id, joints) = up.upload(&joint_bytes, storage, "joint transforms")?;
            let influences = up.table(&t.skin, "skin influences")?;
            let triangles = up.table(&t.triangles, "deformation triangles")?;
            let gather_offsets = up.table(&t.gather.offsets, "normal gather offsets")?;
            let gather_corners = up.table(&t.gather.items, "normal gather corners")?;
            let edges = up.table(&t.edges, "strain edges")?;
            let region_blocks = up.table(&t.region_blocks, "strain blocks")?;
            let region_block_offsets = up.table(&t.region_block_offsets, "strain region blocks")?;
            let region_edges = up.table(&t.region_edges.items, "strain region edges")?;
            let subtree_offsets = up.table(&t.subtrees.offsets, "strain subtree offsets")?;
            let subtree_regions = up.table(&t.subtrees.items, "strain subtree regions")?;
            let position_keys = up.table(&t.position_keys, "strain position keys")?;
            let key_offsets = up.table(&t.key_edges.offsets, "strain key offsets")?;
            let key_edges = up.table(&t.key_edges.items, "strain key edges")?;
            let (_, rest) = up.output(v * 12, "rest positions")?;
            let (_, positions) = up.output(v * 12, "deformed positions")?;
            let (_, prev_positions) = up.output(v * 12, "previous deformed positions")?;
            let (_, normals) = up.output(v * 12, "deformed normals")?;
            let (_, tangents) = up.output(v * 16, "deformed tangents")?;
            let (_, frames) = up.output(u64::from(t.triangle_count()) * 48, "triangle frames")?;
            let (_, edge_strain) = up.output(t.edges.len() as u64 * 8, "edge strain")?;
            let (_, block_sums) = up.output(t.region_blocks.len() as u64 * 16, "strain block sums")?;
            let (_, region_strain) = up.output(u64::from(t.region_count) * 8, "region strain")?;
            let (_, vertex_strain) = up.output(v * 4, "vertex strain")?;

            let device = &gpu.device;
            let bg = |pass: Pass, buffers: &[&wgpu::Buffer]| bind_group(device, &cache, pass, &params, buffers);
            let groups = Groups {
                gather: bg(
                    Pass::Gather,
                    &[&rest, &template_positions, &shape_offsets, &shape_entries, &shape_weights],
                )?,
                scatter: bg(Pass::ScatterComponent, &[&rest, &component_indices, &component_deltas])?,
                skin: bg(
                    Pass::Skin,
                    &[&rest, &dynamic_offsets, &dynamic_entries, &dynamic_weights, &joints, &influences, &positions],
                )?,
                triangles: bg(Pass::Triangles, &[&positions, &uvs, &triangles, &frames])?,
                vertices: bg(Pass::Vertices, &[&frames, &gather_offsets, &gather_corners, &normals, &tangents])?,
                edge_strain: bg(Pass::EdgeStrain, &[&rest, &positions, &edges, &edge_strain])?,
                region_sums: bg(Pass::RegionSums, &[&edge_strain, &region_blocks, &region_edges, &block_sums])?,
                region_strain: bg(
                    Pass::RegionStrain,
                    &[&block_sums, &subtree_offsets, &subtree_regions, &region_block_offsets, &region_strain],
                )?,
                vertex_strain: bg(
                    Pass::VertexStrain,
                    &[&edge_strain, &position_keys, &key_offsets, &key_edges, &vertex_strain],
                )?,
            };
            let mut lods = Vec::with_capacity(lod_tables.len());
            for lod in &lod_tables {
                let n = u64::from(lod.vertex_count);
                let sources = up.table(&lod.sources, "level sources")?;
                let weights = up.table(&lod.weights, "level source weights")?;
                let lod_uvs = up.table(&lod.uvs, "level uvs")?;
                let lod_triangles = up.table(&lod.triangles, "level triangles")?;
                let lod_offsets = up.table(&lod.gather.offsets, "level gather offsets")?;
                let lod_corners = up.table(&lod.gather.items, "level gather corners")?;
                let (_, lod_positions) = up.output(n * 12, "level positions")?;
                let (_, lod_normals) = up.output(n * 12, "level normals")?;
                let (_, lod_tangents) = up.output(n * 16, "level tangents")?;
                let lod_frames_bytes = (lod.triangles.len() / 3) as u64 * 48;
                let (_, lod_frames) = up.output(lod_frames_bytes, "level triangle frames")?;
                lods.push(GpuLod {
                    vertex_count: lod.vertex_count,
                    triangle_count: (lod.triangles.len() / 3) as u32,
                    transfer: bg(Pass::LodTransfer, &[&positions, &sources, &weights, &lod_positions])?,
                    triangles: bg(Pass::Triangles, &[&lod_positions, &lod_uvs, &lod_triangles, &lod_frames])?,
                    vertices: bg(
                        Pass::Vertices,
                        &[&lod_frames, &lod_offsets, &lod_corners, &lod_normals, &lod_tangents],
                    )?,
                    positions: lod_positions,
                    normals: lod_normals,
                    tangents: lod_tangents,
                });
            }
            let raw = Raw {
                params,
                rest,
                positions,
                prev_positions,
                normals,
                tangents,
                edge_strain,
                region_strain,
                vertex_strain,
            };
            Ok((params_id, shape_weights_id, dynamic_weights_id, joints_id, raw, groups, lods))
        })();
        let (params_id, shape_weights_id, dynamic_weights_id, joints_id, raw, groups, lods) = match built {
            Ok(b) => b,
            Err(e) => {
                for id in up.made {
                    let _ = up.resources.destroy_buffer(id);
                }
                return Err(e);
            }
        };

        let skeleton = template.skeleton();
        let rest_world = skeleton.rest_world();
        let joint_count = t.joint_count as usize;
        let mut params = vec![0_u8; SLOT_COUNT * SLOT as usize];
        put_slot(&mut params, SLOT_GATHER, [t.vertex_count, 0, 0, 0]);
        put_slot(
            &mut params,
            SLOT_SKIN,
            [t.vertex_count, t.influences, t.joint_count, mode_word(SkinningMode::ByRegion)],
        );
        put_slot(&mut params, SLOT_NORMALS, [t.vertex_count, t.triangle_count(), 0, 0]);
        let blocks = t.region_blocks.len() as u32;
        put_slot(&mut params, SLOT_STRAIN, [t.edges.len() as u32, t.region_count, t.vertex_count, blocks]);
        for (k, lod) in lods.iter().enumerate() {
            let slot = SLOT_LOD + 2 * k as u32;
            put_slot(&mut params, slot, [lod.vertex_count, t.vertex_count, 0, 0]);
            put_slot(&mut params, slot + 1, [lod.vertex_count, lod.triangle_count, 0, 0]);
        }
        let mut deformer = Self {
            settings,
            cache,
            owned: Owned {
                params: params_id,
                shape_weights: shape_weights_id,
                dynamic_weights: dynamic_weights_id,
                joints: joints_id,
                all: up.made,
            },
            raw,
            groups,
            layers: Vec::new(),
            lods,
            vertex_count: t.vertex_count,
            triangle_count: t.triangle_count(),
            edge_count: t.edges.len() as u32,
            region_count: t.region_count,
            influences: t.influences,
            block_count: t.region_blocks.len() as u32,
            dynamic_components: (0..t.component_count).filter(|&c| t.is_dynamic[c as usize]).collect(),
            scratch_dynamic: vec![0.0; t.component_count as usize],
            // The template's check solved every driver already; own kernels are the fallback.
            pose_space: PoseSpace::new(template.expressions()).unwrap_or_default(),
            pose_activations: vec![0.0; template.expressions().pose_correctives.len()],
            twist: TwistRig::new(template.skeleton()),
            driven_pose: vec![Quat::IDENTITY; template.skeleton().joints.len()],
            applied_shape: vec![0.0; t.component_count as usize],
            applied_layers: Vec::new(),
            dynamic_weights: vec![0.0; t.component_count as usize],
            joint_words: vec![[0.0; 4]; joint_count * JOINT_VEC4S],
            parents: skeleton.joints.iter().map(|j| j.parent).collect(),
            bind_rotations: rest_world.iter().map(|w| w.rotation).collect(),
            rest_world_positions: rest_world.iter().map(|w| w.translation).collect(),
            pose: vec![Quat::IDENTITY; joint_count],
            joints: JointFrames::new(joint_count),
            mode: SkinningMode::ByRegion,
            incremental_updates: 0,
            jobs: Vec::with_capacity(JOB_SLOTS),
            full: true,
            dirty: true,
            moved: false,
            params,
            params_dirty: true,
            is_shape: t.is_shape,
            component_offsets: t.component_offsets,
            template,
        };
        // The neutral state: pose correctives may be active at the rest pose.
        deformer.resolve_dynamic(&EvalInput::default());
        deformer.pose_joints();
        deformer.upload(resources, true, true, true)?;
        Ok(deformer)
    }

    /// The template.
    #[must_use]
    pub fn template(&self) -> &Arc<Template> {
        &self.template
    }

    /// The settings.
    #[must_use]
    pub fn settings(&self) -> DeformSettings {
        self.settings
    }

    /// Vertices of the deformed mesh.
    #[must_use]
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Triangles the normals are computed on.
    #[must_use]
    pub fn triangle_count(&self) -> u32 {
        self.triangle_count
    }

    /// Welded edges of the strain.
    #[must_use]
    pub fn edge_count(&self) -> u32 {
        self.edge_count
    }

    /// Regions of the region strain.
    #[must_use]
    pub fn region_count(&self) -> u32 {
        self.region_count
    }

    /// The output buffers.
    #[must_use]
    pub fn outputs(&self) -> DeformOutputs<'_> {
        DeformOutputs {
            positions: &self.raw.positions,
            prev_positions: &self.raw.prev_positions,
            normals: &self.raw.normals,
            tangents: &self.raw.tangents,
            rest: &self.raw.rest,
            edge_strain: &self.raw.edge_strain,
            region_strain: &self.raw.region_strain,
            vertex_strain: &self.raw.vertex_strain,
        }
    }

    /// The output buffers of reduced level `level` (1-based: level 1 is the template's first
    /// reduced level), when the deformer has it.
    #[must_use]
    pub fn lod_outputs(&self, level: usize) -> Option<LodOutputs<'_>> {
        let lod = self.lods.get(level.checked_sub(1)?)?;
        Some(LodOutputs {
            vertex_count: lod.vertex_count,
            positions: &lod.positions,
            normals: &lod.normals,
            tangents: &lod.tangents,
        })
    }

    /// Reduced levels on the GPU.
    #[must_use]
    pub fn lod_count(&self) -> usize {
        self.lods.len()
    }

    /// The posed joint frames of the last update (bind positions from the regressor, skinning
    /// transforms).
    #[must_use]
    pub fn joints(&self) -> &JointFrames {
        &self.joints
    }

    /// Pipelines created after set-up (always 0: everything is built in [`new`](Self::new)).
    #[must_use]
    pub fn late_pipelines(&self) -> u32 {
        self.cache.stats().late
    }

    /// Adds a sculpt layer (weight 0 until an input gives it one) and returns its index. Uploads
    /// the layer once; allocation happens here, never per frame.
    pub fn add_layer(
        &mut self,
        gpu: &Gpu,
        resources: &mut Resources,
        layer: &SculptLayer,
    ) -> Result<usize, DeformError> {
        if self.layers.len() >= MAX_GPU_LAYERS {
            return Err(DeformError::TooLarge {
                what: "sculpt layers on the GPU",
                value: self.layers.len() as u64 + 1,
                limit: MAX_GPU_LAYERS as u64,
            });
        }
        let made = self.upload_layer(gpu, resources, layer)?;
        self.layers.push(made);
        self.applied_layers.push(0.0);
        Ok(self.layers.len() - 1)
    }

    /// Replaces layer `index` (its weight carries over). The next update re-accumulates the rest
    /// shape in full.
    pub fn replace_layer(
        &mut self,
        gpu: &Gpu,
        resources: &mut Resources,
        index: usize,
        layer: &SculptLayer,
    ) -> Result<(), DeformError> {
        if index >= self.layers.len() {
            return Err(EvalError::Missing { what: "sculpt layer", index }.into());
        }
        let made = self.upload_layer(gpu, resources, layer)?;
        let old = std::mem::replace(&mut self.layers[index], made);
        let _ = resources.destroy_buffer(old.indices);
        let _ = resources.destroy_buffer(old.deltas);
        if self.applied_layers.get(index).is_some_and(|&w| w != 0.0) {
            self.full = true;
            self.dirty = true;
        }
        Ok(())
    }

    fn upload_layer(&self, gpu: &Gpu, resources: &mut Resources, layer: &SculptLayer) -> Result<GpuLayer, DeformError> {
        if layer.indices().last().is_some_and(|&v| v >= self.vertex_count) {
            return Err(EvalError::Missing {
                what: "vertex",
                index: layer.indices().last().copied().unwrap_or(0) as usize,
            }
            .into());
        }
        let count = u32::try_from(layer.len()).map_err(|_| DeformError::TooLarge {
            what: "layer entries",
            value: layer.len() as u64,
            limit: u64::from(u32::MAX),
        })?;
        let deltas: Vec<f32> = layer.deltas().iter().flat_map(|d| d.to_array()).collect();
        let max_binding = gpu.limits().max_storage_buffer_binding_size;
        let mut up = Uploader { resources, made: Vec::new(), max_binding };
        let made = (|| -> Result<_, DeformError> {
            let (indices_id, indices) =
                up.upload(slice_as_bytes(layer.indices()), wgpu::BufferUsages::STORAGE, "sculpt layer indices")?;
            let (deltas_id, deltas_raw) =
                up.upload(slice_as_bytes(&deltas), wgpu::BufferUsages::STORAGE, "sculpt layer deltas")?;
            let bind = bind_group(
                &gpu.device,
                &self.cache,
                Pass::ScatterLayer,
                &self.raw.params,
                &[&self.raw.rest, &indices, &deltas_raw],
            )?;
            Ok(GpuLayer { indices: indices_id, deltas: deltas_id, count, bind_group: bind })
        })();
        if made.is_err() {
            for id in up.made {
                let _ = up.resources.destroy_buffer(id);
            }
        }
        made
    }

    /// Checks `input` as the CPU reference does, without changing anything.
    pub fn check_input(&self, input: &EvalInput<'_>) -> Result<(), DeformError> {
        let expressions = self.template.expressions().expressions.len();
        check_weights("shape weights", input.shape, self.applied_shape.len())?;
        check_weights("expression weights", input.expressions, expressions)?;
        check_weights("layer weights", input.layers, self.layers.len())?;
        if input.pose.len() > self.pose.len() {
            return Err(
                EvalError::Length { what: "joint rotations", limit: self.pose.len(), got: input.pose.len() }.into()
            );
        }
        for (i, q) in input.pose.iter().enumerate() {
            if !q.is_finite() || q.length() < 1e-6 {
                return Err(EvalError::NonFinite { what: "joint rotation", index: i }.into());
            }
        }
        Ok(())
    }

    /// Takes the evaluation input of this frame and schedules the GPU work for it (see the module
    /// documentation). On error nothing changes. Allocation-free.
    pub fn update(&mut self, resources: &mut Resources, input: &EvalInput<'_>) -> Result<UpdatePlan, DeformError> {
        let (dynamic_changed, joints_changed) = self.plan(input)?;
        self.upload(resources, self.full, dynamic_changed, joints_changed)?;
        Ok(UpdatePlan { full: self.full, jobs: if self.full { 0 } else { self.jobs.len() }, deform: self.dirty })
    }

    /// The CPU half of [`update`](Self::update): checks, diffs and schedules, returning whether the
    /// dynamic weights and the joint words changed. Nothing is written to the GPU.
    pub(crate) fn plan(&mut self, input: &EvalInput<'_>) -> Result<(bool, bool), DeformError> {
        self.check_input(input)?;
        let rest_changed = self.diff_rest(input.shape, input.layers);
        let dynamic_changed = self.resolve_dynamic(input);
        let mut pose_changed = rest_changed;
        // The twist joints are driven from the limbs' roll, as in the CPU evaluator.
        for (slot, q) in self.driven_pose.iter_mut().enumerate() {
            *q = input.pose.get(slot).map_or(Quat::IDENTITY, |q| q.normalize());
        }
        self.twist.apply(&mut self.driven_pose);
        for (q, &next) in self.pose.iter_mut().zip(&self.driven_pose) {
            if *q != next {
                *q = next;
                pose_changed = true;
            }
        }
        if input.skinning != self.mode {
            self.mode = input.skinning;
            let words = [self.vertex_count, self.influences, self.pose.len() as u32, mode_word(self.mode)];
            put_slot(&mut self.params, SLOT_SKIN, words);
            self.params_dirty = true;
            self.dirty = true;
        }
        let joints_changed = pose_changed && self.pose_joints();
        self.dirty |= rest_changed || dynamic_changed || joints_changed;
        Ok((dynamic_changed, joints_changed))
    }

    /// Step 1 bookkeeping: schedules scatter jobs for changed weights or a full pass.
    fn diff_rest(&mut self, shape: &[f32], layers: &[f32]) -> bool {
        let mut changed = false;
        for c in 0..self.applied_shape.len() {
            let w = if self.is_shape[c] { shape.get(c).copied().unwrap_or(0.0) } else { 0.0 };
            let old = self.applied_shape[c];
            if w == old {
                continue;
            }
            self.applied_shape[c] = w;
            self.incremental_updates += 1;
            changed = true;
            let (Some(&first), Some(&end)) = (self.component_offsets.get(c), self.component_offsets.get(c + 1)) else {
                continue;
            };
            let count = end.saturating_sub(first);
            if count > 0 && !self.full {
                self.push_job(Job { layer: false, index: c as u32, first, count, weight: w - old });
            }
        }
        for l in 0..self.applied_layers.len() {
            let w = layers.get(l).copied().unwrap_or(0.0);
            let old = self.applied_layers[l];
            if w == old {
                continue;
            }
            self.applied_layers[l] = w;
            self.incremental_updates += 1;
            changed = true;
            let count = self.layers.get(l).map_or(0, |layer| layer.count);
            if count > 0 && !self.full {
                self.push_job(Job { layer: true, index: l as u32, first: 0, count, weight: w - old });
            }
        }
        if self.incremental_updates >= self.settings.refresh_interval {
            self.full = true;
        }
        if self.full {
            self.incremental_updates = 0;
            self.jobs.clear();
            for (l, (&w, layer)) in self.applied_layers.iter().zip(&self.layers).enumerate() {
                if w != 0.0 && layer.count > 0 {
                    self.jobs.push(Job { layer: true, index: l as u32, first: 0, count: layer.count, weight: w });
                }
            }
        }
        changed || self.full
    }

    fn push_job(&mut self, job: Job) {
        if self.jobs.len() < self.settings.max_jobs {
            self.jobs.push(job);
        } else {
            self.full = true;
        }
    }

    /// Steps 2 and 3 weights: one weight per dynamic component (expression weights, combination
    /// activations, pose-corrective activations, summed when a component is named twice). Returns
    /// whether any changed.
    fn resolve_dynamic(&mut self, input: &EvalInput<'_>) -> bool {
        let set = self.template.expressions();
        let next = &mut self.scratch_dynamic;
        for &c in &self.dynamic_components {
            next[c as usize] = 0.0;
        }
        let mut add = |c: u32, w: f32| {
            if let Some(slot) = next.get_mut(c as usize) {
                *slot += w;
            }
        };
        for (e, &w) in set.expressions.iter().zip(input.expressions) {
            add(e.component, w);
        }
        for c in &set.combinations {
            add(c.component, c.activation(input.expressions));
        }
        self.pose_space.activations(&set.pose_correctives, input.pose, &mut self.pose_activations);
        for (pc, &a) in set.pose_correctives.iter().zip(&self.pose_activations) {
            add(pc.component, a);
        }
        let mut changed = false;
        for &c in &self.dynamic_components {
            let (Some(&n), Some(slot)) =
                (self.scratch_dynamic.get(c as usize), self.dynamic_weights.get_mut(c as usize))
            else {
                continue;
            };
            if *slot != n {
                *slot = n;
                changed = true;
            }
        }
        changed
    }

    /// Step 4 joints: bind positions from the regressor for the applied shape weights, posed
    /// frames, packed. Returns whether the packed words changed.
    fn pose_joints(&mut self) -> bool {
        if self.joints.is_empty() {
            return false;
        }
        let regressor = &self.template.skeleton().regressor;
        self.joints.bind_positions.copy_from_slice(&self.rest_world_positions);
        for (c, &w) in self.applied_shape.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            let (joints, offsets) = regressor.component(c);
            for (&j, &d) in joints.iter().zip(offsets) {
                if let Some(p) = self.joints.bind_positions.get_mut(j as usize) {
                    *p += d * w;
                }
            }
        }
        self.joints.pose(&self.parents, &self.bind_rotations, &self.pose);
        let mut changed = false;
        for (j, words) in self.joint_words.as_chunks_mut::<JOINT_VEC4S>().0.iter_mut().enumerate() {
            let (Some(m), Some(t), Some(dq)) = (
                self.joints.skin_matrices.get(j),
                self.joints.skin_translations.get(j),
                self.joints.skin_dual_quats.get(j),
            ) else {
                continue;
            };
            let next = [
                [m.cols[0].x, m.cols[0].y, m.cols[0].z, t.x],
                [m.cols[1].x, m.cols[1].y, m.cols[1].z, t.y],
                [m.cols[2].x, m.cols[2].y, m.cols[2].z, t.z],
                [dq.real.x, dq.real.y, dq.real.z, dq.real.w],
                [dq.dual.x, dq.dual.y, dq.dual.z, dq.dual.w],
            ];
            if *words != next {
                *words = next;
                changed = true;
            }
        }
        changed
    }

    /// Writes what changed: shape weights (full pass), dynamic weights, joints, parameters.
    fn upload(
        &mut self,
        resources: &mut Resources,
        full: bool,
        dynamic: bool,
        joints: bool,
    ) -> Result<(), DeformError> {
        if full {
            resources.write_buffer(self.owned.shape_weights, 0, slice_as_bytes(&self.applied_shape))?;
        }
        if dynamic {
            resources.write_buffer(self.owned.dynamic_weights, 0, slice_as_bytes(&self.dynamic_weights))?;
        }
        if joints && !self.joint_words.is_empty() {
            resources.write_buffer(self.owned.joints, 0, slice_as_bytes(&self.joint_words))?;
        }
        for (k, job) in self.jobs.iter().enumerate() {
            put_slot(&mut self.params, SLOT_JOBS + k as u32, [job.count, job.first, job.weight.to_bits(), 0]);
            self.params_dirty = true;
        }
        if self.params_dirty {
            let used = (SLOT_JOBS as usize + self.jobs.len()) * SLOT as usize;
            if let Some(bytes) = self.params.get(..used) {
                resources.write_buffer(self.owned.params, 0, bytes)?;
            }
            self.params_dirty = false;
        }
        Ok(())
    }

    /// Records the deformation of the last [`update`](Self::update) into `encoder`: keeps the
    /// previous positions, then one compute pass. Returns whether anything was recorded.
    /// Allocation-free.
    pub fn encode(&mut self, encoder: &mut wgpu::CommandEncoder) -> bool {
        self.encode_inner(encoder, Timing::None)
    }

    /// [`encode`](Self::encode) with one compute pass per [`Stage`], each bracketed by timestamp
    /// writes into `queries` at `first + 2 · stage` and `first + 2 · stage + 1` (the device needs
    /// timestamp queries; `queries` needs `first + 10` entries). For benchmarks `[PERF-R6]`; the
    /// pass boundaries add a little GPU time of their own.
    pub fn encode_timed(&mut self, encoder: &mut wgpu::CommandEncoder, queries: &wgpu::QuerySet, first: u32) -> bool {
        self.encode_inner(encoder, Timing::Stages(queries, first))
    }

    /// [`encode`](Self::encode) as its one compute pass, bracketed by timestamp writes into
    /// `queries` at `first` and `first + 1`.
    pub fn encode_timed_total(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queries: &wgpu::QuerySet,
        first: u32,
    ) -> bool {
        self.encode_inner(encoder, Timing::Total(queries, first))
    }

    /// [`encode`](Self::encode) with the pass timed by the frame profiler under the scope name
    /// `"deformation"` `[PERF-R6]`.
    pub fn encode_profiled(&mut self, encoder: &mut wgpu::CommandEncoder, profiler: &mut GpuProfiler) -> bool {
        match self.prepare(encoder) {
            Prepared::Nothing => return false,
            Prepared::CopyOnly => return true,
            Prepared::Run => {}
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("deformation"),
            timestamp_writes: profiler.compute_pass_writes("deformation"),
        });
        for stage in Stage::ALL {
            self.record(&mut pass, stage);
        }
        drop(pass);
        self.finish();
        true
    }

    /// Records the copy into the previous positions when the positions changed this frame or the
    /// last one (so motion becomes zero one frame after the last change), and says what else runs.
    fn prepare(&mut self, encoder: &mut wgpu::CommandEncoder) -> Prepared {
        if !self.dirty && !self.moved {
            return Prepared::Nothing;
        }
        let bytes = u64::from(self.vertex_count) * 12;
        if bytes > 0 {
            encoder.copy_buffer_to_buffer(&self.raw.positions, 0, &self.raw.prev_positions, 0, bytes);
        }
        self.moved = self.dirty;
        if self.dirty { Prepared::Run } else { Prepared::CopyOnly }
    }

    fn finish(&mut self) {
        self.jobs.clear();
        self.full = false;
        self.dirty = false;
    }

    fn encode_inner(&mut self, encoder: &mut wgpu::CommandEncoder, timing: Timing<'_>) -> bool {
        match self.prepare(encoder) {
            Prepared::Nothing => return false,
            Prepared::CopyOnly => return true,
            Prepared::Run => {}
        }
        let stamps = |queries, at| {
            Some(wgpu::ComputePassTimestampWrites {
                query_set: queries,
                beginning_of_pass_write_index: Some(at),
                end_of_pass_write_index: Some(at + 1),
            })
        };
        match timing {
            Timing::None | Timing::Total(..) => {
                let timestamp_writes = match timing {
                    Timing::Total(queries, first) => stamps(queries, first),
                    _ => None,
                };
                let mut pass = encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("deformation"), timestamp_writes });
                for stage in Stage::ALL {
                    self.record(&mut pass, stage);
                }
            }
            Timing::Stages(queries, first) => {
                for (k, stage) in Stage::ALL.into_iter().enumerate() {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("deformation stage"),
                        timestamp_writes: stamps(queries, first + 2 * k as u32),
                    });
                    self.record(&mut pass, stage);
                }
            }
        }
        self.finish();
        true
    }

    fn pipeline(&self, pass: Pass) -> Option<&wgpu::ComputePipeline> {
        self.cache.get_compute_pipeline(&Key::Pipeline(pass))
    }

    fn run(&self, pass: &mut wgpu::ComputePass<'_>, which: Pass, group: &wgpu::BindGroup, slot: u32, items: u32) {
        if let Some(pipeline) = self.pipeline(which) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[slot * SLOT as u32]);
            dispatch(pass, items);
        }
    }

    fn record(&self, pass: &mut wgpu::ComputePass<'_>, stage: Stage) {
        let g = &self.groups;
        match stage {
            Stage::Accumulate => {
                if self.full {
                    self.run(pass, Pass::Gather, &g.gather, SLOT_GATHER, self.vertex_count);
                }
                for (k, job) in self.jobs.iter().enumerate() {
                    let slot = SLOT_JOBS + k as u32;
                    if job.layer {
                        if let Some(layer) = self.layers.get(job.index as usize) {
                            self.run(pass, Pass::ScatterLayer, &layer.bind_group, slot, job.count);
                        }
                    } else {
                        self.run(pass, Pass::ScatterComponent, &g.scatter, slot, job.count);
                    }
                }
            }
            Stage::Skin => self.run(pass, Pass::Skin, &g.skin, SLOT_SKIN, self.vertex_count),
            Stage::Normals => {
                self.run(pass, Pass::Triangles, &g.triangles, SLOT_NORMALS, self.triangle_count);
                self.run(pass, Pass::Vertices, &g.vertices, SLOT_NORMALS, self.vertex_count);
            }
            Stage::Strain => {
                self.run(pass, Pass::EdgeStrain, &g.edge_strain, SLOT_STRAIN, self.edge_count);
                if let Some(pipeline) = self.pipeline(Pass::RegionSums)
                    && self.block_count > 0
                {
                    // One workgroup per block.
                    let x = self.block_count.min(MAX_GROUPS);
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &g.region_sums, &[SLOT_STRAIN * SLOT as u32]);
                    pass.dispatch_workgroups(x, self.block_count.div_ceil(x), 1);
                }
                self.run(pass, Pass::RegionStrain, &g.region_strain, SLOT_STRAIN, self.region_count);
                self.run(pass, Pass::VertexStrain, &g.vertex_strain, SLOT_STRAIN, self.vertex_count);
            }
            Stage::Lod => {
                for (k, lod) in self.lods.iter().enumerate() {
                    let slot = SLOT_LOD + 2 * k as u32;
                    self.run(pass, Pass::LodTransfer, &lod.transfer, slot, lod.vertex_count);
                    self.run(pass, Pass::Triangles, &lod.triangles, slot + 1, lod.triangle_count);
                    self.run(pass, Pass::Vertices, &lod.vertices, slot + 1, lod.vertex_count);
                }
            }
        }
    }

    /// Frees every buffer of the deformer.
    pub fn destroy(self, resources: &mut Resources) {
        for id in self.owned.all {
            let _ = resources.destroy_buffer(id);
        }
        for layer in self.layers {
            let _ = resources.destroy_buffer(layer.indices);
            let _ = resources.destroy_buffer(layer.deltas);
        }
    }
}
