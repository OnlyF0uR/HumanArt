//! Hair generation on the GPU `[HAIR-02]`: the generation modifiers of
//! `humanart_model::hair::generate` as compute passes (`shaders/groom/generate.wgsl`), run when a
//! groom is loaded or edited, plus the per-frame attach of the generated strands to the deformed
//! skin.
//!
//! [`GpuGroom`] owns every buffer of one groom, sized once for a [`GroomCapacity`], the three
//! pipelines (built once, `[PERF-R4]`) and their bind groups. The CPU side is
//! `humanart_model::hair`: [`GpuGroom::set_layout`] uploads the root placement and interpolation
//! data (`GroomLayout`, after a placement change), [`GpuGroom::set_guides`] the resampled guides
//! (after a guide edit), [`GpuGroom::set_params`] the modifiers. [`GpuGroom::encode_generate`]
//! records the `shape` and `finish` stages; [`GpuGroom::encode_attach`] the per-frame `attach`
//! stage against the skin bound with [`GpuGroom::bind_skin`] (the rest surface, or the
//! deformation's output buffers, `humanart-render::deform`). Uploads reuse one staging vector;
//! `encode_attach` allocates nothing (`[PERF-R1]`).
//!
//! Outputs ([`GpuGroom::outputs`]): `shape` (stage-1 world points, vec4 per point), `stored`
//! (offsets in the rest root frame as two `pack2x16float` words per point, `[HAIR-01]`), `attrs`
//! (two vec4<u32> per strand: length, root and tip diameter, curl phase as f32 bits; colour id |
//! grey << 16, stiffness bits, clump centre), `world` (position and diameter per point, for the
//! hair renderer, WP-18).

use humanart_gpu::cache::PipelineCache;
use humanart_gpu::resources::{BufferDesc, BufferId, Category, Resources, Tag};
use humanart_gpu::{Gpu, GpuError};
use humanart_math::Vec3;
use humanart_model::hair::{GroomLayout, GroomParams, HairSurface, SdfGrid};
use humanart_shaders::ShaderError;

pub mod error;

pub use error::GroomGpuError;

/// The shader file of the groom passes.
const SHADER: &str = "groom/generate.wgsl";
/// Invocations per workgroup.
const WORKGROUP: u32 = 64;
/// The most workgroups along one dispatch axis.
const MAX_GROUPS: u32 = 65_535;
/// Bytes of one strand record.
const STRAND_RECORD: u64 = 96;
/// Bytes of the parameter block.
const PARAMS_BYTES: usize = 20 * 16;
/// Most control points per strand in the shader.
const MAX_POINTS: u32 = 32;

/// The sizes a [`GpuGroom`] is built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroomCapacity {
    /// Most strands.
    pub strands: u32,
    /// Most guides.
    pub guides: u32,
    /// Control points per strand and per resampled guide (2..=32).
    pub points: u32,
}

/// The output buffers of a groom.
#[derive(Clone, Copy, Debug)]
pub struct GroomOutputs<'a> {
    /// Stage-1 world points (vec4 per point).
    pub shape: &'a wgpu::Buffer,
    /// Stored offsets, `f16` (vec2<u32> per point).
    pub stored: &'a wgpu::Buffer,
    /// Per-strand attributes (two vec4<u32> per strand).
    pub attrs: &'a wgpu::Buffer,
    /// Attached world points and diameters (vec4 per point).
    pub world: &'a wgpu::Buffer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Key {
    Shader,
    GenerateLayout,
    AttachLayout,
    GeneratePipelineLayout,
    AttachPipelineLayout,
    Shape,
    Finish,
    Attach,
}

/// The bindings of the generate layout (`shape`, `finish`): (binding, read-only).
const GENERATE_BINDINGS: [(u32, bool); 8] =
    [(1, true), (2, true), (3, true), (4, true), (5, false), (6, true), (7, false), (8, false)];
/// The bindings of the attach layout.
const ATTACH_BINDINGS: [(u32, bool); 6] = [(1, true), (7, false), (8, false), (9, true), (10, true), (11, false)];

const fn entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn layout_entries<const N: usize>(bindings: [(u32, bool); N]) -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut entries = vec![entry(0, wgpu::BufferBindingType::Uniform)];
    entries.extend(bindings.iter().map(|&(b, read_only)| entry(b, wgpu::BufferBindingType::Storage { read_only })));
    entries
}

/// The workgroup grid for `items` invocations: (x, y) with x at most [`MAX_GROUPS`].
fn grid(items: u32) -> (u32, u32) {
    let groups = items.div_ceil(WORKGROUP);
    if groups == 0 {
        return (0, 0);
    }
    let x = groups.min(MAX_GROUPS);
    (x, groups.div_ceil(x))
}

/// One owned buffer.
#[derive(Debug)]
struct Owned {
    id: BufferId,
    raw: wgpu::Buffer,
}

/// The GPU side of one groom. See the module documentation.
#[derive(Debug)]
pub struct GpuGroom {
    cache: PipelineCache<Key>,
    capacity: GroomCapacity,
    vertex_count: u32,
    params: Owned,
    strands: Owned,
    curves: Owned,
    rest_positions: Owned,
    rest_normals: Owned,
    shape: Owned,
    sdf: Owned,
    stored: Owned,
    attrs: Owned,
    world: Owned,
    generate_group: wgpu::BindGroup,
    attach_group: wgpu::BindGroup,
    strand_count: u32,
    guide_count: u32,
    has_sdf: bool,
    sdf_header: ([f32; 4], [u32; 4]),
    groom_params: GroomParams,
    staging: Vec<u8>,
}

fn f32_bytes(out: &mut Vec<u8>, values: impl IntoIterator<Item = f32>) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn u32_bytes(out: &mut Vec<u8>, values: impl IntoIterator<Item = u32>) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

impl GpuGroom {
    /// Builds the pipelines and buffers for `capacity` on the rest skin `surface`, with the head's
    /// distance grid `sdf` (no scalp collision without).
    pub fn new(
        gpu: &Gpu,
        resources: &mut Resources,
        surface: &HairSurface,
        sdf: Option<&SdfGrid>,
        capacity: GroomCapacity,
    ) -> Result<Self, GroomGpuError> {
        if !(2..=MAX_POINTS).contains(&capacity.points) || capacity.strands == 0 || capacity.guides == 0 {
            return Err(GroomGpuError::Invalid {
                what: "groom capacity: 2..=32 points, at least one strand and guide",
            });
        }
        let source = humanart_shaders::compose(SHADER, &[]).map_err(GroomGpuError::Shader)?.into_source();
        let mut cache = PipelineCache::new(gpu);
        cache.shader(Key::Shader, SHADER, &source)?;
        cache.bind_group_layout(Key::GenerateLayout, "groom generate", &layout_entries(GENERATE_BINDINGS))?;
        cache.bind_group_layout(Key::AttachLayout, "groom attach", &layout_entries(ATTACH_BINDINGS))?;
        cache.pipeline_layout(Key::GeneratePipelineLayout, "groom generate", &[Some(&Key::GenerateLayout)])?;
        cache.pipeline_layout(Key::AttachPipelineLayout, "groom attach", &[Some(&Key::AttachLayout)])?;
        for (key, layout, entry_point) in [
            (Key::Shape, Key::GeneratePipelineLayout, "shape"),
            (Key::Finish, Key::GeneratePipelineLayout, "finish"),
            (Key::Attach, Key::AttachPipelineLayout, "attach"),
        ] {
            cache.compute_pipeline(key, |parts| {
                Ok(parts.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: Some(parts.pipeline_layout(&layout)?),
                    module: parts.shader(&Key::Shader)?,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                }))
            })?;
        }
        cache.mark_warm();

        let max_binding = gpu.limits().max_storage_buffer_binding_size;
        let points = u64::from(capacity.strands) * u64::from(capacity.points);
        let vertex_count = surface.positions().len() as u32;
        let mut made: Vec<BufferId> = Vec::new();
        let mut make = |resources: &mut Resources, size: u64, usage: wgpu::BufferUsages, purpose: &'static str| {
            let size = size.max(16).next_multiple_of(4);
            if usage.contains(wgpu::BufferUsages::STORAGE) && size > max_binding {
                return Err(GroomGpuError::TooLarge { what: purpose, value: size, limit: max_binding });
            }
            let (id, _) = resources.create_buffer(&BufferDesc {
                size,
                usage: usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                tag: Tag::shared(Category::Hair, purpose),
            })?;
            made.push(id);
            let raw = resources.buffer(id).ok_or(GpuError::Stale { what: "groom buffer" })?.raw().clone();
            Ok(Owned { id, raw })
        };
        let storage = wgpu::BufferUsages::STORAGE;
        let built = (|| -> Result<_, GroomGpuError> {
            let params = make(resources, PARAMS_BYTES as u64, wgpu::BufferUsages::UNIFORM, "groom parameters")?;
            let strands = make(resources, u64::from(capacity.strands) * STRAND_RECORD, storage, "groom strands")?;
            let curves = make(
                resources,
                u64::from(capacity.guides) * u64::from(capacity.points) * 16,
                storage,
                "groom guide curves",
            )?;
            let rest_positions = make(resources, u64::from(vertex_count) * 12, storage, "groom rest positions")?;
            let rest_normals = make(resources, u64::from(vertex_count) * 12, storage, "groom rest normals")?;
            let shape = make(resources, points * 16, storage, "groom shape")?;
            let sdf_len = sdf.map_or(8, |g| g.values.len() as u64);
            let sdf_buffer = make(resources, sdf_len * 4, storage, "groom distance grid")?;
            let stored = make(resources, points * 8, storage, "groom stored offsets")?;
            let attrs = make(resources, u64::from(capacity.strands) * 32, storage, "groom attributes")?;
            let world = make(resources, points * 16, storage, "groom world points")?;
            Ok((params, strands, curves, rest_positions, rest_normals, shape, sdf_buffer, stored, attrs, world))
        })();
        let (params, strands, curves, rest_positions, rest_normals, shape, sdf_buffer, stored, attrs, world) =
            match built {
                Ok(b) => b,
                Err(e) => {
                    for id in made {
                        let _ = resources.destroy_buffer(id);
                    }
                    return Err(e);
                }
            };
        let device = &gpu.device;
        let generate_layout = cache
            .get_bind_group_layout(&Key::GenerateLayout)
            .ok_or(GpuError::Stale { what: "groom generate layout" })?;
        let generate_group = bind_group(
            device,
            generate_layout,
            &params.raw,
            &[
                (1, &strands.raw),
                (2, &curves.raw),
                (3, &rest_positions.raw),
                (4, &rest_normals.raw),
                (5, &shape.raw),
                (6, &sdf_buffer.raw),
                (7, &stored.raw),
                (8, &attrs.raw),
            ],
        );
        let attach_layout =
            cache.get_bind_group_layout(&Key::AttachLayout).ok_or(GpuError::Stale { what: "groom attach layout" })?;
        let attach_group = bind_group(
            device,
            attach_layout,
            &params.raw,
            &[
                (1, &strands.raw),
                (7, &stored.raw),
                (8, &attrs.raw),
                (9, &rest_positions.raw),
                (10, &rest_normals.raw),
                (11, &world.raw),
            ],
        );
        let mut groom = Self {
            cache,
            capacity,
            vertex_count,
            params,
            strands,
            curves,
            rest_positions,
            rest_normals,
            shape,
            sdf: sdf_buffer,
            stored,
            attrs,
            world,
            generate_group,
            attach_group,
            strand_count: 0,
            guide_count: 0,
            has_sdf: false,
            sdf_header: ([0.0; 4], [2, 2, 2, 0]),
            groom_params: GroomParams::default(),
            staging: Vec::new(),
        };
        groom.set_rest_surface(resources, surface.positions(), surface.normals())?;
        if let Some(grid) = sdf {
            groom.staging.clear();
            f32_bytes(&mut groom.staging, grid.values.iter().copied());
            resources.write_buffer(groom.sdf.id, 0, &groom.staging)?;
            groom.has_sdf = true;
            groom.sdf_header = (
                [grid.origin.x, grid.origin.y, grid.origin.z, grid.cell],
                [grid.dims[0], grid.dims[1], grid.dims[2], 0],
            );
        }
        groom.write_params(resources)?;
        Ok(groom)
    }

    /// The capacity the groom was built for.
    pub fn capacity(&self) -> GroomCapacity {
        self.capacity
    }

    /// Strands currently laid out.
    pub fn strand_count(&self) -> u32 {
        self.strand_count
    }

    /// Pipelines created after start-up (0 in a healthy run, `[PERF-R4]`).
    pub fn late_pipelines(&self) -> u32 {
        self.cache.stats().late
    }

    /// Replaces the rest skin positions and normals (same vertex count): a re-evaluated identity.
    pub fn set_rest_surface(
        &mut self,
        resources: &mut Resources,
        positions: &[Vec3],
        normals: &[Vec3],
    ) -> Result<(), GroomGpuError> {
        if positions.len() != self.vertex_count as usize || normals.len() != positions.len() {
            return Err(GroomGpuError::Invalid { what: "rest surface vertex count changed" });
        }
        self.staging.clear();
        f32_bytes(&mut self.staging, positions.iter().flat_map(|p| [p.x, p.y, p.z]));
        resources.write_buffer(self.rest_positions.id, 0, &self.staging)?;
        self.staging.clear();
        f32_bytes(&mut self.staging, normals.iter().flat_map(|p| [p.x, p.y, p.z]));
        resources.write_buffer(self.rest_normals.id, 0, &self.staging)?;
        Ok(())
    }

    /// Uploads the strand layout (`layout` checked against `surface` and the current guides).
    pub fn set_layout(
        &mut self,
        resources: &mut Resources,
        surface: &HairSurface,
        layout: &GroomLayout,
    ) -> Result<(), GroomGpuError> {
        let n = layout.len();
        if n > self.capacity.strands as usize {
            return Err(GroomGpuError::TooLarge {
                what: "strand count",
                value: n as u64,
                limit: u64::from(self.capacity.strands),
            });
        }
        if surface.positions().len() != self.vertex_count as usize {
            return Err(GroomGpuError::Invalid { what: "layout surface differs from the groom's skin" });
        }
        layout.check(surface, self.capacity.guides as usize).map_err(GroomGpuError::Hair)?;
        self.staging.clear();
        for i in 0..n {
            let t = layout.triangle[i];
            let [v0, v1, v2] = surface.triangle(t).unwrap_or([0; 3]);
            let coef = surface.tangent_coefficients(t).unwrap_or([1.0, 0.0]);
            u32_bytes(&mut self.staging, [v0, v1, v2, layout.clump[i]]);
            f32_bytes(&mut self.staging, [layout.bary[i][0], layout.bary[i][1], coef[0], coef[1]]);
            u32_bytes(&mut self.staging, layout.guides[i]);
            f32_bytes(&mut self.staging, layout.weights[i]);
            f32_bytes(&mut self.staging, layout.factors[i]);
            f32_bytes(&mut self.staging, [layout.direction[i][0], layout.direction[i][1], layout.color[i]]);
            u32_bytes(&mut self.staging, [layout.key[i]]);
        }
        resources.write_buffer(self.strands.id, 0, &self.staging)?;
        self.strand_count = n as u32;
        self.write_params(resources)
    }

    /// Uploads the guides resampled to the groom's point count (`Guides::resampled`), guide
    /// after guide. The guide count may change; strands must not refer past it.
    pub fn set_guides(&mut self, resources: &mut Resources, curves: &[Vec3]) -> Result<(), GroomGpuError> {
        let p = self.capacity.points as usize;
        let count = curves.len() / p;
        if count * p != curves.len() || count > self.capacity.guides as usize {
            return Err(GroomGpuError::Invalid { what: "guide curves: not a whole number of guides, or too many" });
        }
        self.staging.clear();
        f32_bytes(&mut self.staging, curves.iter().flat_map(|c| [c.x, c.y, c.z, 0.0]));
        resources.write_buffer(self.curves.id, 0, &self.staging)?;
        self.guide_count = count as u32;
        self.write_params(resources)
    }

    /// Sets the modifiers. `params.points` must equal the capacity's point count.
    pub fn set_params(&mut self, resources: &mut Resources, params: &GroomParams) -> Result<(), GroomGpuError> {
        params.check().map_err(GroomGpuError::Hair)?;
        if params.point_count() != self.capacity.points {
            return Err(GroomGpuError::Invalid { what: "point count differs from the groom's capacity" });
        }
        self.groom_params = *params;
        self.write_params(resources)
    }

    fn write_params(&mut self, resources: &mut Resources) -> Result<(), GroomGpuError> {
        let p = &self.groom_params;
        let n = self.capacity.points;
        let mut bytes = [0_u8; PARAMS_BYTES];
        let mut words = [0_u32; PARAMS_BYTES / 4];
        let (strand_x, _) = grid(self.strand_count);
        let (point_x, _) = grid(self.strand_count.saturating_mul(n));
        let part = p.parting();
        let mut f = |slot: usize, v: [f32; 4]| {
            for (k, x) in v.into_iter().enumerate() {
                words[slot * 4 + k] = x.to_bits();
            }
        };
        f(2, [p.length_scale, p.length_variation, p.direction_strength, p.noise]);
        f(3, [p.curl_radius, p.curl_frequency, p.curl_variation, p.wave_amplitude]);
        f(4, [p.wave_frequency, p.frizz, p.flyaway_fraction, p.flyaway_strength]);
        f(5, [p.clump_strength, p.clump_spread, p.clump_tip_spread, p.collision_offset]);
        f(6, [p.thickness, p.tip_ratio, p.thickness_variation, p.grey_fraction]);
        f(7, [p.stiffness, 0.0, 0.0, 0.0]);
        f(8, [part.point.x, part.point.y, part.point.z, part.width]);
        f(9, [part.normal.x, part.normal.y, part.normal.z, part.strength]);
        for c in 0..4 {
            if let Some(cw) = p.cowlick(c) {
                f(10 + 2 * c, [cw.centre.x, cw.centre.y, cw.centre.z, cw.radius]);
                f(11 + 2 * c, [cw.strength, 0.0, 0.0, 0.0]);
            }
        }
        f(18, self.sdf_header.0);
        let u = [
            [self.strand_count, n, self.guide_count, strand_x.max(1)],
            [point_x.max(1), u32::from(self.has_sdf), 0, 0],
        ];
        for (slot, v) in u.iter().enumerate() {
            words[slot * 4..slot * 4 + 4].copy_from_slice(v);
        }
        words[76..80].copy_from_slice(&self.sdf_header.1);
        for (chunk, w) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            *chunk = w.to_le_bytes();
        }
        resources.write_buffer(self.params.id, 0, &bytes)?;
        Ok(())
    }

    /// Binds the skin the strands attach to: packed f32 xyz positions and normals with the rest
    /// surface's vertex count (the deformation's outputs). Creates one bind group: call when the
    /// buffers change, not per frame.
    pub fn bind_skin(
        &mut self,
        device: &wgpu::Device,
        positions: &wgpu::Buffer,
        normals: &wgpu::Buffer,
    ) -> Result<(), GroomGpuError> {
        let need = u64::from(self.vertex_count) * 12;
        if positions.size() < need || normals.size() < need {
            return Err(GroomGpuError::Invalid { what: "skin buffers smaller than the groom's vertex count" });
        }
        let layout = self
            .cache
            .get_bind_group_layout(&Key::AttachLayout)
            .ok_or(GpuError::Stale { what: "groom attach layout" })?;
        self.attach_group = bind_group(
            device,
            layout,
            &self.params.raw,
            &[
                (1, &self.strands.raw),
                (7, &self.stored.raw),
                (8, &self.attrs.raw),
                (9, positions),
                (10, normals),
                (11, &self.world.raw),
            ],
        );
        Ok(())
    }

    /// Binds the rest surface as the skin (no deformation).
    pub fn bind_rest_skin(&mut self, device: &wgpu::Device) -> Result<(), GroomGpuError> {
        let (p, n) = (self.rest_positions.raw.clone(), self.rest_normals.raw.clone());
        self.bind_skin(device, &p, &n)
    }

    /// Records the `shape` and `finish` stages (regenerates the groom). `false` with no strands.
    pub fn encode_generate(&self, encoder: &mut wgpu::CommandEncoder) -> bool {
        self.encode_generate_timed(encoder, None)
    }

    /// [`encode_generate`](Self::encode_generate) bracketed by timestamp writes at `first` and
    /// `first + 1` of `queries` (benchmarks, `[PERF-R6]`).
    pub fn encode_generate_timed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timing: Option<(&wgpu::QuerySet, u32)>,
    ) -> bool {
        let (Some(shape), Some(finish)) =
            (self.cache.get_compute_pipeline(&Key::Shape), self.cache.get_compute_pipeline(&Key::Finish))
        else {
            return false;
        };
        if self.strand_count == 0 || self.guide_count == 0 {
            return false;
        }
        let writes = timing.map(|(q, first)| wgpu::ComputePassTimestampWrites {
            query_set: q,
            beginning_of_pass_write_index: Some(first),
            end_of_pass_write_index: Some(first + 1),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("groom generate"),
            timestamp_writes: writes,
        });
        let (x, y) = grid(self.strand_count);
        pass.set_bind_group(0, &self.generate_group, &[]);
        pass.set_pipeline(shape);
        pass.dispatch_workgroups(x, y, 1);
        pass.set_pipeline(finish);
        pass.dispatch_workgroups(x, y, 1);
        true
    }

    /// Records the per-frame `attach` stage. Allocation-free. `false` with no strands.
    pub fn encode_attach(&self, encoder: &mut wgpu::CommandEncoder) -> bool {
        self.encode_attach_timed(encoder, None)
    }

    /// [`encode_attach`](Self::encode_attach) with optional timestamp writes.
    pub fn encode_attach_timed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timing: Option<(&wgpu::QuerySet, u32)>,
    ) -> bool {
        let Some(attach) = self.cache.get_compute_pipeline(&Key::Attach) else { return false };
        if self.strand_count == 0 {
            return false;
        }
        let writes = timing.map(|(q, first)| wgpu::ComputePassTimestampWrites {
            query_set: q,
            beginning_of_pass_write_index: Some(first),
            end_of_pass_write_index: Some(first + 1),
        });
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("groom attach"), timestamp_writes: writes });
        let (x, y) = grid(self.strand_count * self.capacity.points);
        pass.set_bind_group(0, &self.attach_group, &[]);
        pass.set_pipeline(attach);
        pass.dispatch_workgroups(x, y, 1);
        true
    }

    /// The output buffers.
    pub fn outputs(&self) -> GroomOutputs<'_> {
        GroomOutputs {
            shape: &self.shape.raw,
            stored: &self.stored.raw,
            attrs: &self.attrs.raw,
            world: &self.world.raw,
        }
    }

    /// Releases every buffer back to `resources`.
    pub fn destroy(self, resources: &mut Resources) {
        for b in [
            self.params,
            self.strands,
            self.curves,
            self.rest_positions,
            self.rest_normals,
            self.shape,
            self.sdf,
            self.stored,
            self.attrs,
            self.world,
        ] {
            let _ = resources.destroy_buffer(b.id);
        }
    }
}

fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    buffers: &[(u32, &wgpu::Buffer)],
) -> wgpu::BindGroup {
    let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() }];
    entries
        .extend(buffers.iter().map(|&(binding, b)| wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }));
    // Layouts and buffers are made by this module; a validation error would be a bug and reaches
    // the device's uncaptured-error queue `[ERR-05]`.
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("groom"), layout, entries: &entries })
}

impl From<GpuError> for GroomGpuError {
    fn from(e: GpuError) -> Self {
        Self::Gpu(e)
    }
}

impl From<ShaderError> for GroomGpuError {
    fn from(e: ShaderError) -> Self {
        Self::Shader(e)
    }
}

#[cfg(test)]
mod tests;
