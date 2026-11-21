//! Device-loss recovery `[ERR-05]` `[AC-WP04-2]`.
//!
//! A lost device (driver reset, GPU removed, [`simulate_loss`]) never panics: [`crate::DeviceEvents`]
//! records it and [`Gpu::is_lost`] turns true. Once per frame the application calls
//! [`Recovery::check`]; when the device is lost it:
//!
//! 1. creates a new device with the same request on the same instance ([`Gpu::recreate`]);
//! 2. re-creates every buffer, texture and sampler of [`Resources`] under its old id, empty;
//! 3. calls [`Rebuild::rebuild_after_loss`] on every other GPU owner (rings, profiler, streaming pool,
//!    pipeline cache) so they point at the new objects;
//! 4. replaces the caller's [`Gpu`].
//!
//! The contents are gone: the owners refill them from their CPU state (the document, the mirrors of
//! the dirty-range uploads) and re-run their pipeline pre-warm, which is the same code as at startup.
//! A device that cannot be created at once is retried on the next frames, up to
//! [`MAX_ATTEMPTS`]; then the error is returned.

use crate::cache::PipelineCache;
use crate::device::Gpu;
use crate::error::GpuError;
use crate::profiler::GpuProfiler;
use crate::readback::ReadbackRing;
use crate::resources::Resources;
use crate::streaming::StreamingPool;
use crate::uniform_ring::UniformRing;

/// Frames in a row a new device may fail to come up before recovery gives up.
pub const MAX_ATTEMPTS: u32 = 8;

/// A GPU owner that must follow the device to a new one after a loss.
pub trait Rebuild {
    /// Re-creates or re-points this object's GPU state on `gpu`, after `resources` was rebuilt there.
    fn rebuild_after_loss(&mut self, gpu: &Gpu, resources: &mut Resources) -> Result<(), GpuError>;
}

impl Rebuild for UniformRing {
    fn rebuild_after_loss(&mut self, _gpu: &Gpu, resources: &mut Resources) -> Result<(), GpuError> {
        self.rebuild(resources)
    }
}

impl Rebuild for ReadbackRing {
    fn rebuild_after_loss(&mut self, _gpu: &Gpu, resources: &mut Resources) -> Result<(), GpuError> {
        self.rebuild(resources)
    }
}

impl Rebuild for GpuProfiler {
    fn rebuild_after_loss(&mut self, gpu: &Gpu, resources: &mut Resources) -> Result<(), GpuError> {
        self.rebuild(gpu, resources)
    }
}

impl Rebuild for StreamingPool {
    fn rebuild_after_loss(&mut self, _gpu: &Gpu, _resources: &mut Resources) -> Result<(), GpuError> {
        self.rebuild();
        Ok(())
    }
}

impl<K: std::hash::Hash + Eq> Rebuild for PipelineCache<K> {
    fn rebuild_after_loss(&mut self, gpu: &Gpu, _resources: &mut Resources) -> Result<(), GpuError> {
        self.reset(gpu);
        Ok(())
    }
}

/// What [`Recovery::check`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryOutcome {
    /// The device is fine.
    Healthy,
    /// The device was lost and everything was rebuilt on a new one: refill contents and pre-warm.
    Recovered {
        /// Buffers, textures and samplers re-created in [`Resources`].
        resources: usize,
    },
    /// The device is lost and a new one could not be made yet; skip the frame and check again.
    Retrying {
        /// Failed attempts so far.
        attempts: u32,
    },
}

/// The per-frame loss check and its retry state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Recovery {
    /// Failed attempts since the last loss.
    attempts: u32,
    /// Successful recoveries so far.
    recoveries: u32,
}

impl Recovery {
    /// A fresh state.
    pub const fn new() -> Self {
        Self { attempts: 0, recoveries: 0 }
    }

    /// Successful recoveries so far.
    pub fn recoveries(&self) -> u32 {
        self.recoveries
    }

    /// Checks the device (one atomic load when it is fine) and recovers it when lost, blocking until
    /// the new device exists. Native only; in a browser use [`Recovery::check_async`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn check(
        &mut self,
        gpu: &mut Gpu,
        surface: Option<&wgpu::Surface<'_>>,
        resources: &mut Resources,
        owners: &mut [&mut dyn Rebuild],
    ) -> Result<RecoveryOutcome, GpuError> {
        if !gpu.is_lost() {
            return Ok(RecoveryOutcome::Healthy);
        }
        crate::device::block_on(self.recover(gpu, surface, resources, owners))
    }

    /// [`Recovery::check`] for async callers.
    pub async fn check_async(
        &mut self,
        gpu: &mut Gpu,
        surface: Option<&wgpu::Surface<'_>>,
        resources: &mut Resources,
        owners: &mut [&mut dyn Rebuild],
    ) -> Result<RecoveryOutcome, GpuError> {
        if !gpu.is_lost() {
            return Ok(RecoveryOutcome::Healthy);
        }
        self.recover(gpu, surface, resources, owners).await
    }

    /// Makes the new device and rebuilds everything on it.
    async fn recover(
        &mut self,
        gpu: &mut Gpu,
        surface: Option<&wgpu::Surface<'_>>,
        resources: &mut Resources,
        owners: &mut [&mut dyn Rebuild],
    ) -> Result<RecoveryOutcome, GpuError> {
        let fresh = match gpu.recreate(surface).await {
            Ok(fresh) => fresh,
            Err(error) => {
                self.attempts += 1;
                if self.attempts >= MAX_ATTEMPTS {
                    return Err(error);
                }
                return Ok(RecoveryOutcome::Retrying { attempts: self.attempts });
            }
        };
        let count = resources.rebuild(&fresh)?;
        for owner in owners.iter_mut() {
            owner.rebuild_after_loss(&fresh, resources)?;
        }
        *gpu = fresh;
        self.attempts = 0;
        self.recoveries += 1;
        Ok(RecoveryOutcome::Recovered { resources: count })
    }
}

/// Loses the device on purpose, the way a driver reset would: for tests and the debug menu. The
/// loss is reported through the device-lost callback, so [`Gpu::is_lost`] turns true.
pub fn simulate_loss(gpu: &Gpu) {
    gpu.device.destroy();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceRequest;
    use crate::error::DeviceEvent;
    use crate::resources::{Budget, BufferDesc, Category, DirtyRanges, Tag, TextureDesc};
    use crate::streaming::{StreamingDesc, TileRequest};
    use crate::uniform_ring::UniformRingDesc;

    /// Test-only WGSL: doubles every element of a storage buffer.
    const DOUBLE: &str = "@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&data) { data[id.x] = data[id.x] * 2u; }
}";

    /// The owner's startup code: the pipeline and its bind group, made through the cache. Run again
    /// after a recovery, unchanged.
    fn warm_up(cache: &mut PipelineCache<u8>, resources: &Resources, data: crate::resources::BufferId) {
        let storage = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        cache.shader(0, "double", DOUBLE).unwrap();
        cache.bind_group_layout(0, "double", &[storage]).unwrap();
        cache.pipeline_layout(0, "double", &[Some(&0)]).unwrap();
        cache
            .compute_pipeline(0, |parts| {
                Ok(parts.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("double"),
                    layout: Some(parts.pipeline_layout(&0)?),
                    module: parts.shader(&0)?,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                }))
            })
            .unwrap();
        let resource = resources.buffer(data).unwrap().raw().as_entire_binding();
        cache.bind_group(0, "double", &0, &[wgpu::BindGroupEntry { binding: 0, resource }]).unwrap();
        cache.mark_warm();
    }
    /// Doubles `data` on the GPU and reads it back through the readback ring.
    fn double_and_read(
        gpu: &Gpu,
        cache: &PipelineCache<u8>,
        readback: &mut ReadbackRing,
        resources: &Resources,
        data: crate::resources::BufferId,
    ) -> Vec<u32> {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(cache.get_compute_pipeline(&0).unwrap());
            pass.set_bind_group(0, cache.get_bind_group(&0).unwrap(), &[]);
            pass.dispatch_workgroups(4, 1, 1);
        }
        let buffer = resources.buffer(data).unwrap().raw();
        let ticket = readback.copy_buffer(&mut encoder, buffer, 0, 1024, 0).unwrap();
        gpu.queue.submit([encoder.finish()]);
        readback.after_submit();
        let mut bytes = vec![0_u8; 1024];
        readback.read_blocking(&gpu.device, ticket, &mut bytes).unwrap();
        let mut words = vec![0_u32; 256];
        crate::pod::copy_from_bytes(&bytes, &mut words).unwrap();
        words
    }

    #[test]
    fn ac_wp04_2_a_lost_device_is_rebuilt_and_refilled_from_cpu_state() {
        let mut gpu = Gpu::headless_blocking(&DeviceRequest::default()).unwrap();
        let mut resources = Resources::new(&gpu, Budget::UNLIMITED);

        // CPU state: a buffer mirror and a texture's texels.
        let values: Vec<u32> = (0..256).map(|i| i * 3).collect();
        let mirror = crate::pod::slice_as_bytes(&values).to_vec();
        let texels: Vec<u8> = (0..64).collect();
        let (data, _) = resources
            .create_buffer(&BufferDesc {
                size: 1024,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                tag: Tag::new(Category::Deformation, 1, "positions"),
            })
            .unwrap();
        let usage =
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC;
        let (texture, _) = resources
            .create_texture(&TextureDesc::d2(
                4,
                4,
                wgpu::TextureFormat::Rgba8Unorm,
                usage,
                Tag::new(Category::Textures, 1, "albedo"),
            ))
            .unwrap();
        resources.sampler(&crate::resources::SamplerKey::LINEAR_CLAMP).unwrap();
        let upload = |resources: &mut Resources| {
            let mut dirty = DirtyRanges::new();
            dirty.mark_all(1024);
            resources.flush_dirty(data, &mirror, &mut dirty).unwrap();
            resources.write_texture_level(texture, 0, 0, &texels).unwrap();
        };
        upload(&mut resources);

        let mut cache = PipelineCache::new(&gpu);
        warm_up(&mut cache, &resources, data);
        let mut ring = UniformRing::new(&mut resources, UniformRingDesc::new(1024, 16, "frame")).unwrap();
        let mut readback = ReadbackRing::new(&mut resources, 2, 1024, "readback").unwrap();
        let mut profiler = GpuProfiler::new(&gpu, &mut resources, 2, 4).unwrap();
        let mut streaming = StreamingPool::new(
            &mut resources,
            StreamingDesc {
                tile_size: 4,
                format: wgpu::TextureFormat::Rgba8Unorm,
                tiles_per_side: 2,
                layers: 1,
                max_pages: 64,
                queue_capacity: 8,
                purpose: "streaming",
            },
        )
        .unwrap();
        let face = streaming.add_texture(16, 16).unwrap();
        streaming
            .update(&mut resources, 4, |_, bytes| {
                bytes.fill(1);
                Ok(())
            })
            .unwrap();
        let before = double_and_read(&gpu, &cache, &mut readback, &resources, data);
        assert_eq!(before, values.iter().map(|v| v * 2).collect::<Vec<_>>());

        let mut recovery = Recovery::new();
        let mut owners: [&mut dyn Rebuild; 5] = [&mut ring, &mut readback, &mut profiler, &mut streaming, &mut cache];
        assert_eq!(recovery.check(&mut gpu, None, &mut resources, &mut owners).unwrap(), RecoveryOutcome::Healthy);

        // The loss.
        simulate_loss(&gpu);
        let _ = crate::device::wait_for(&gpu.device, None, crate::device::GPU_WAIT_LIMIT, "the queued work");
        assert!(gpu.is_lost());
        let lost = std::iter::from_fn(|| gpu.events().poll()).find(|event| matches!(event, DeviceEvent::Lost { .. }));
        assert!(lost.is_some(), "the loss is reported as an event");
        let old_device = gpu.device.clone();

        // The recovery, then the owner's ordinary startup code refills everything from CPU state.
        let outcome = recovery.check(&mut gpu, None, &mut resources, &mut owners).unwrap();
        let live = resources.live();
        assert_eq!(outcome, RecoveryOutcome::Recovered { resources: live.0 + live.1 + 1 });
        assert!(!gpu.is_lost());
        assert!(gpu.device != old_device, "a new device");
        assert_eq!(recovery.recoveries(), 1);
        assert!(cache.is_empty(), "the cache was emptied for the new device");
        upload(&mut resources);
        warm_up(&mut cache, &resources, data);
        let after = double_and_read(&gpu, &cache, &mut readback, &resources, data);
        assert_eq!(after, before, "the same result on the new device, from CPU state alone");

        // The texture came back from its texels, and the other owners work on the new device.
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let raw = resources.texture(texture).unwrap().raw().clone();
        let ticket = readback.copy_texture(&mut encoder, &raw, 0, [0, 0, 0], [4, 4, 1], 0).unwrap();
        profiler.begin_frame();
        profiler.end_frame(&mut encoder);
        gpu.queue.submit([encoder.finish()]);
        readback.after_submit();
        profiler.after_submit();
        let mut read = [0_u8; 64];
        readback.read_blocking(&gpu.device, ticket, &mut read).unwrap();
        assert_eq!(&read[..], &texels[..]);
        ring.begin_frame();
        ring.push(&[1_u32; 4]).unwrap();
        ring.finish_frame(&mut resources).unwrap();
        assert_eq!(streaming.queued(), 1, "the pinned top waits to be streamed in again");
        assert_eq!(
            streaming
                .update(&mut resources, 4, |_, bytes| {
                    bytes.fill(1);
                    Ok(())
                })
                .unwrap()
                .loaded,
            1
        );
        assert!(streaming.lookup(TileRequest { texture: face, mip: 0, x: 0, y: 0 }).is_some());
        crate::device::wait_for(&gpu.device, None, crate::device::GPU_WAIT_LIMIT, "the queued work").unwrap();
        assert_eq!(gpu.events().pending(), 0, "no error on the new device");
        let mut owners: [&mut dyn Rebuild; 1] = [&mut ring];
        assert_eq!(recovery.check(&mut gpu, None, &mut resources, &mut owners).unwrap(), RecoveryOutcome::Healthy);
    }
}
