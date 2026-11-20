//! The per-frame uniform ring `[GPU-03]`.
//!
//! One uniform buffer holds every frame and view constant block of a frame. Each block is pushed into
//! a CPU staging copy at the next offset aligned to the device's `min_uniform_buffer_offset_alignment`;
//! the caller binds the ring's single bind group with that offset as the dynamic offset. At the end of
//! the frame the used part goes to the GPU in one queue write. There are no per-draw uniform buffers
//! and no allocation per frame `[PERF-R1]`: the staging copy is sized once.
//!
//! The ring starts over at offset 0 every frame. Reusing the bytes is safe because queue writes are
//! ordered with submissions: the previous frame's commands read the old contents before the next
//! write lands.
//!
//! The buffer is made through [`Resources`], so it is tagged and counted `[GPU-05]`; after a device
//! loss [`UniformRing::rebuild`] makes the bind group again for the rebuilt buffer.

use crate::error::{GpuError, guarded};
use crate::pod::{Pod, bytes_of};
use crate::resources::{BufferDesc, BufferId, Category, Resources, Tag};

/// How a [`UniformRing`] is made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UniformRingDesc {
    /// Bytes of the ring: the most constants one frame may push, alignment included.
    pub capacity: u32,
    /// Bytes the shader sees at each dynamic offset: the size of the largest block, a multiple of
    /// 16 (WGSL uniform layout). Every pushed block must fit in it.
    pub binding_size: u32,
    /// The stages that read the constants.
    pub visibility: wgpu::ShaderStages,
    /// The binding index in the ring's bind group.
    pub binding: u32,
    /// Accounting label and debug name.
    pub purpose: &'static str,
}

impl UniformRingDesc {
    /// A ring of `capacity` bytes with blocks of up to `binding_size` bytes at binding 0, visible to
    /// every stage.
    pub const fn new(capacity: u32, binding_size: u32, purpose: &'static str) -> Self {
        Self {
            capacity,
            binding_size,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT.union(wgpu::ShaderStages::COMPUTE),
            binding: 0,
            purpose,
        }
    }
}

/// What one frame of the ring used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RingStats {
    /// Frames finished.
    pub frames: u64,
    /// Blocks pushed in the current frame.
    pub blocks: u32,
    /// Bytes used in the current frame, alignment included.
    pub used: u32,
    /// The most bytes any frame used.
    pub high_water: u32,
    /// Pushes refused because the ring was full.
    pub overflows: u32,
}

/// The ring. See the module documentation.
#[derive(Debug)]
pub struct UniformRing {
    /// The uniform buffer in [`Resources`].
    buffer: BufferId,
    /// The request it was made from.
    desc: UniformRingDesc,
    /// The device's offset alignment.
    alignment: u32,
    /// The CPU copy of this frame's blocks, `capacity` bytes, allocated once.
    staging: Vec<u8>,
    /// Next free offset.
    cursor: u32,
    /// Bytes to upload this frame: the end of the last block's bound range.
    end: u32,
    /// The layout of the ring's bind group: one uniform buffer with a dynamic offset.
    layout: wgpu::BindGroupLayout,
    /// The ring's bind group.
    bind_group: wgpu::BindGroup,
    /// The counters.
    stats: RingStats,
}

impl UniformRing {
    /// Makes the ring's buffer in `resources` and its bind group. The capacity is rounded up to
    /// whole aligned blocks; a binding size of zero, above the device's uniform binding limit, or
    /// larger than the capacity is [`GpuError::Invalid`].
    pub fn new(resources: &mut Resources, desc: UniformRingDesc) -> Result<Self, GpuError> {
        let limits = resources.limits();
        let alignment = limits.min_uniform_buffer_offset_alignment.max(16);
        if desc.binding_size == 0 || !desc.binding_size.is_multiple_of(16) {
            return Err(GpuError::Invalid { what: "a uniform binding size that is zero or not a multiple of 16" });
        }
        if u64::from(desc.binding_size) > limits.max_uniform_buffer_binding_size {
            return Err(GpuError::Invalid { what: "a uniform binding size above the device's limit" });
        }
        let capacity = desc
            .capacity
            .checked_next_multiple_of(alignment)
            .ok_or(GpuError::Invalid { what: "a uniform ring capacity that overflows" })?;
        if capacity < desc.binding_size {
            return Err(GpuError::Invalid { what: "a uniform ring smaller than one block" });
        }
        let desc = UniformRingDesc { capacity, ..desc };
        let (buffer, _) = resources.create_buffer(&BufferDesc {
            size: u64::from(capacity),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            tag: Tag::shared(Category::Uniforms, desc.purpose),
        })?;
        let (layout, bind_group) = match make_bind_group(resources, buffer, &desc) {
            Ok(made) => made,
            Err(error) => {
                // Do not leak the buffer's budget on failure.
                let _ = resources.destroy_buffer(buffer);
                return Err(error);
            }
        };
        Ok(Self {
            buffer,
            desc,
            alignment,
            staging: vec![0; capacity as usize],
            cursor: 0,
            end: 0,
            layout,
            bind_group,
            stats: RingStats::default(),
        })
    }

    /// The layout to put in pipeline layouts where the constants are read.
    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// The bind group to set with the offsets [`UniformRing::push`] returns.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// The buffer in [`Resources`].
    pub fn buffer(&self) -> BufferId {
        self.buffer
    }

    /// The request, with the capacity as allocated.
    pub fn desc(&self) -> &UniformRingDesc {
        &self.desc
    }

    /// The device's offset alignment the blocks are placed at.
    pub fn alignment(&self) -> u32 {
        self.alignment
    }

    /// The counters.
    pub fn stats(&self) -> &RingStats {
        &self.stats
    }

    /// Starts a frame: the ring is empty again.
    pub fn begin_frame(&mut self) {
        self.cursor = 0;
        self.end = 0;
        self.stats.blocks = 0;
        self.stats.used = 0;
    }

    /// Pushes one constant block and returns its dynamic offset. Never allocates.
    pub fn push<T: Pod>(&mut self, value: &T) -> Result<u32, GpuError> {
        self.push_bytes(bytes_of(value))
    }

    /// Pushes raw bytes as one block and returns its dynamic offset. A block larger than the
    /// binding size is [`GpuError::Invalid`]; a full ring is [`GpuError::Full`].
    pub fn push_bytes(&mut self, bytes: &[u8]) -> Result<u32, GpuError> {
        let Ok(len) = u32::try_from(bytes.len()) else {
            return Err(GpuError::Invalid { what: "a uniform block larger than the binding size" });
        };
        if len > self.desc.binding_size {
            return Err(GpuError::Invalid { what: "a uniform block larger than the binding size" });
        }
        let offset = self.cursor;
        // The shader reads `binding_size` bytes from the offset: all of them must be in the ring.
        let Some(bound_end) = offset.checked_add(self.desc.binding_size).filter(|&end| end <= self.desc.capacity)
        else {
            self.stats.overflows += 1;
            return Err(GpuError::Full { what: "the uniform ring" });
        };
        let start = offset as usize;
        let Some(target) = self.staging.get_mut(start..start + bytes.len()) else {
            self.stats.overflows += 1;
            return Err(GpuError::Full { what: "the uniform ring" });
        };
        target.copy_from_slice(bytes);
        self.end = bound_end;
        // The next block starts at the next aligned offset after this one's bytes (a smaller block
        // leaves the rest of its bound range to the next). Near the end the cursor may pass the
        // capacity; the next push then reports the ring full.
        self.cursor = offset.saturating_add(len.max(1)).checked_next_multiple_of(self.alignment).unwrap_or(u32::MAX);
        self.stats.blocks += 1;
        self.stats.used = self.cursor.min(self.desc.capacity);
        Ok(offset)
    }

    /// Uploads this frame's blocks in one queue write and returns the bytes written.
    pub fn finish_frame(&mut self, resources: &mut Resources) -> Result<u64, GpuError> {
        let end = self.end as usize;
        let Some(bytes) = self.staging.get(..end) else {
            return Err(GpuError::OutOfRange {
                what: "uniform ring upload",
                offset: 0,
                len: u64::from(self.end),
                capacity: u64::from(self.desc.capacity),
            });
        };
        resources.write_buffer(self.buffer, 0, bytes)?;
        self.stats.frames += 1;
        self.stats.high_water = self.stats.high_water.max(self.stats.used);
        Ok(u64::from(self.end))
    }

    /// Makes the bind group again after [`Resources::rebuild`] replaced the buffer `[ERR-05]`. The
    /// frame in progress is dropped.
    pub fn rebuild(&mut self, resources: &Resources) -> Result<(), GpuError> {
        (self.layout, self.bind_group) = make_bind_group(resources, self.buffer, &self.desc)?;
        self.begin_frame();
        Ok(())
    }
}

/// The ring's layout and bind group: one uniform buffer with a dynamic offset and a fixed size.
fn make_bind_group(
    resources: &Resources,
    buffer: BufferId,
    desc: &UniformRingDesc,
) -> Result<(wgpu::BindGroupLayout, wgpu::BindGroup), GpuError> {
    let raw = resources.buffer(buffer).ok_or(GpuError::Stale { what: "uniform ring buffer id" })?.raw();
    let device = resources.device();
    guarded(device, "uniform ring bind group", || {
        let size = wgpu::BufferSize::new(u64::from(desc.binding_size));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(desc.purpose),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: desc.binding,
                visibility: desc.visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: size,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(desc.purpose),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: desc.binding,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: raw, offset: 0, size }),
            }],
        });
        (layout, bind_group)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::Budget;
    use crate::testing;

    /// Test-only WGSL: copies `block.x` to `out[block.y]` for the block at the dynamic offset.
    const STORE: &str = "struct Block { value: vec4<u32> }
@group(0) @binding(0) var<uniform> block: Block;
@group(1) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() { out[block.value.y] = block.value.x; }";

    #[test]
    fn gpu_03_blocks_land_at_aligned_dynamic_offsets_and_shaders_read_them() {
        let gpu = testing::gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        let mut ring = UniformRing::new(&mut resources, UniformRingDesc::new(4096, 16, "frame constants")).unwrap();
        assert_eq!(resources.stats().of(Category::Uniforms), 4096, "the ring is accounted [GPU-05]");
        let alignment = ring.alignment();
        assert!(alignment.is_power_of_two() && alignment >= 16);

        let device = &gpu.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("store"),
            source: wgpu::ShaderSource::Wgsl(STORE.into()),
        });
        let out_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("out"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("store"),
            bind_group_layouts: &[Some(ring.layout()), Some(&out_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("store"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("out"),
            layout: &out_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: out.as_entire_binding() }],
        });

        // Two frames; the second overwrites the first's blocks at the same offsets.
        for frame in 0..2_u32 {
            ring.begin_frame();
            let offsets: Vec<u32> =
                (0..4_u32).map(|slot| ring.push(&[100 * frame + slot + 1, slot, 0, 0]).unwrap()).collect();
            assert_eq!(offsets, [0, alignment, 2 * alignment, 3 * alignment]);
            assert_eq!(ring.finish_frame(&mut resources).unwrap(), u64::from(3 * alignment + 16));
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(1, &out_group, &[]);
                for &offset in &offsets {
                    pass.set_bind_group(0, ring.bind_group(), &[offset]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            gpu.queue.submit([encoder.finish()]);
            let mut values = [0_u32; 4];
            crate::pod::copy_from_bytes(&testing::read_buffer(gpu, &out, 0, 16), &mut values).unwrap();
            let base = 100 * frame;
            assert_eq!(values, [base + 1, base + 2, base + 3, base + 4], "frame {frame}");
        }
        assert_eq!(ring.stats().frames, 2);
        assert_eq!(ring.stats().blocks, 4);
        assert_eq!(gpu.events().pending(), 0);
    }

    #[test]
    fn gpu_03_overflow_and_bad_blocks_are_errors() {
        let gpu = testing::gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        assert!(matches!(
            UniformRing::new(&mut resources, UniformRingDesc::new(4096, 0, "zero")),
            Err(GpuError::Invalid { .. })
        ));
        assert!(matches!(
            UniformRing::new(&mut resources, UniformRingDesc::new(4096, 20, "unaligned")),
            Err(GpuError::Invalid { .. })
        ));
        assert!(matches!(
            UniformRing::new(&mut resources, UniformRingDesc::new(4096, 1 << 20, "too big")),
            Err(GpuError::Invalid { .. })
        ));
        assert_eq!(resources.live(), (0, 0), "failed rings leave nothing behind");

        // A ring of exactly two aligned slots of 32-byte blocks.
        let alignment = gpu.limits().min_uniform_buffer_offset_alignment.max(16);
        let mut ring = UniformRing::new(&mut resources, UniformRingDesc::new(2 * alignment - 1, 32, "small")).unwrap();
        assert_eq!(ring.desc().capacity, 2 * alignment, "rounded up to whole slots");
        ring.begin_frame();
        assert_eq!(ring.push(&[1.0_f32; 8]).unwrap(), 0);
        assert!(matches!(ring.push(&[0_u32; 9]), Err(GpuError::Invalid { .. })), "larger than the binding");
        assert_eq!(ring.push(&7_u32).unwrap(), alignment, "small blocks take a whole slot");
        assert!(matches!(ring.push(&7_u32), Err(GpuError::Full { .. })));
        assert_eq!(ring.stats().overflows, 1);
        ring.finish_frame(&mut resources).unwrap();
        let bytes = testing::read_buffer(gpu, resources.buffer(ring.buffer()).unwrap().raw(), 0, 36);
        assert_eq!(&bytes[..4], &1.0_f32.to_le_bytes());

        // After a frame boundary the ring has room again, and a rebuild keeps it usable.
        ring.begin_frame();
        assert_eq!(ring.push(&7_u32).unwrap(), 0);
        ring.rebuild(&resources).unwrap();
        assert_eq!(ring.stats().blocks, 0, "a rebuild drops the frame in progress");
        assert_eq!(ring.stats().high_water, 2 * alignment);
    }
}
