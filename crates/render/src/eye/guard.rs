//! [`GuardMirror`]: the CPU penetration guard's correction `[EVAL-08]` carried into the GPU
//! deformation's outputs.
//!
//! The CPU reference evaluator runs `humanart_model::eval::penetration::PenetrationGuard` after
//! skinning and recomputes its normals (`Evaluator::set_guard`); the GPU deformation
//! ([`GpuDeformer`]) does not. A viewer that draws from the deformer but also evaluates on the CPU
//! (the engine's viewer does both, for picking and the eye and mouth inputs) keeps the two in step
//! with this mirror: on every change of the CPU result, [`GuardMirror::update`] writes the
//! corrected position and normal of every guarded vertex into one buffer (allocated at creation,
//! written in place `[PERF-R1]`), and after each deformation pass [`GuardMirror::encode`] records one
//! small compute dispatch (`shaders/eye/guard.wgsl`) that writes them over the deformer's outputs.
//! The pipeline and bind group are created once `[PERF-R4]`.
//!
//! The deformer's own normals of the guarded vertices' neighbours outside the guarded set stay as
//! the GPU computed them (they move by far less than the guarded vertices); everything the guard
//! moves is inside the set.

use humanart_gpu::resources::{BufferDesc, BufferId, Category, Resources, Tag};
use humanart_math::Vec3;

use crate::deform::GpuDeformer;
use crate::error::RenderError;

/// Bytes per guarded vertex in the fix buffer: (position, index bits), (normal, 0).
const FIX_BYTES: usize = 32;
/// Threads per workgroup of `apply_fixes`.
const WORKGROUP: u32 = 64;

/// See the module documentation.
pub struct GuardMirror {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    params: BufferId,
    fixes: BufferId,
    vertices: Vec<u32>,
    bytes: Vec<u8>,
    ready: bool,
}

impl std::fmt::Debug for GuardMirror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuardMirror")
            .field("vertices", &self.vertices.len())
            .field("ready", &self.ready)
            .finish_non_exhaustive()
    }
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

impl GuardMirror {
    /// A mirror of the guarded vertices `guarded` (duplicates allowed) onto `deformer`'s outputs.
    ///
    /// # Errors
    /// [`RenderError::InvalidParameter`] when there is no guarded vertex or one is beyond the
    /// deformer's vertices; GPU errors of the shader, pipeline and buffers.
    pub fn new(
        resources: &mut Resources,
        deformer: &GpuDeformer,
        guarded: impl IntoIterator<Item = u32>,
    ) -> Result<Self, RenderError> {
        let mut vertices: Vec<u32> = guarded.into_iter().collect();
        vertices.sort_unstable();
        vertices.dedup();
        if vertices.is_empty() {
            return Err(RenderError::InvalidParameter { what: "a guard mirror without guarded vertices" });
        }
        if vertices.last().is_some_and(|&v| v >= deformer.vertex_count()) {
            return Err(RenderError::InvalidParameter { what: "a guarded vertex beyond the deformer's vertices" });
        }
        let device = resources.device().clone();
        let composed = humanart_shaders::compose("eye/guard.wgsl", &[])?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("eye/guard.wgsl"),
            source: wgpu::ShaderSource::Wgsl(composed.source().into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("guard mirror"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, false),
                storage(3, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("guard mirror"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("apply_fixes"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("apply_fixes"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let tag = Tag::shared(Category::Deformation, "penetration guard mirror");
        let (params, _) = resources.create_buffer(&BufferDesc {
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            tag,
        })?;
        let size = (vertices.len() * FIX_BYTES) as u64;
        let fixes = match resources.create_buffer(&BufferDesc {
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            tag,
        }) {
            Ok((id, _)) => id,
            Err(e) => {
                let _ = resources.destroy_buffer(params);
                return Err(e.into());
            }
        };
        let count = u32::try_from(vertices.len()).unwrap_or(u32::MAX);
        let mut header = [0_u8; 16];
        header[..4].copy_from_slice(&count.to_le_bytes());
        let stale = || RenderError::Gpu(humanart_gpu::GpuError::Stale { what: "guard mirror buffer" });
        let made = resources.write_buffer(params, 0, &header).map_err(RenderError::from).and_then(|()| {
            let outputs = deformer.outputs();
            let p = resources.buffer(params).ok_or_else(stale)?.raw();
            let f = resources.buffer(fixes).ok_or_else(stale)?.raw();
            Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("guard mirror"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: p.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: f.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: outputs.positions.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: outputs.normals.as_entire_binding() },
                ],
            }))
        });
        let bind = match made {
            Ok(b) => b,
            Err(e) => {
                let _ = resources.destroy_buffer(params);
                let _ = resources.destroy_buffer(fixes);
                return Err(e);
            }
        };
        let bytes = vec![0; vertices.len() * FIX_BYTES];
        Ok(Self { pipeline, bind, params, fixes, vertices, bytes, ready: false })
    }

    /// The guarded vertices (sorted, unique).
    #[must_use]
    pub fn vertices(&self) -> &[u32] {
        &self.vertices
    }

    /// Takes the guarded vertices' corrected positions and normals from the CPU result (one per
    /// template vertex). No allocation.
    ///
    /// # Errors
    /// [`RenderError::InvalidParameter`] when a slice is shorter than the guarded vertices need
    /// or a value is not finite (nothing is written then); GPU errors of the write.
    pub fn update(
        &mut self,
        resources: &mut Resources,
        positions: &[Vec3],
        normals: &[Vec3],
    ) -> Result<(), RenderError> {
        for (chunk, &v) in self.bytes.as_chunks_mut::<FIX_BYTES>().0.iter_mut().zip(&self.vertices) {
            let (Some(p), Some(n)) = (positions.get(v as usize), normals.get(v as usize)) else {
                return Err(RenderError::InvalidParameter { what: "guard mirror input shorter than the template" });
            };
            if !(p.is_finite() && n.is_finite()) {
                return Err(RenderError::InvalidParameter { what: "guard mirror input not finite" });
            }
            let words =
                [p.x.to_bits(), p.y.to_bits(), p.z.to_bits(), v, n.x.to_bits(), n.y.to_bits(), n.z.to_bits(), 0];
            for (out, w) in chunk.as_chunks_mut::<4>().0.iter_mut().zip(words) {
                out.copy_from_slice(&w.to_le_bytes());
            }
        }
        resources.write_buffer(self.fixes, 0, &self.bytes)?;
        self.ready = true;
        Ok(())
    }

    /// Records the dispatch that writes the fixes over the deformer's outputs. Call it after
    /// every recorded deformation pass (`GpuDeformer::encode` returned `true`); nothing is
    /// recorded before the first [`update`](Self::update). Allocation-free.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.ready {
            return;
        }
        let count = u32::try_from(self.vertices.len()).unwrap_or(u32::MAX);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("penetration guard"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.dispatch_workgroups(count.div_ceil(WORKGROUP), 1, 1);
    }

    /// Frees the buffers.
    pub fn destroy(self, resources: &mut Resources) {
        let _ = resources.destroy_buffer(self.params);
        let _ = resources.destroy_buffer(self.fixes);
    }
}
