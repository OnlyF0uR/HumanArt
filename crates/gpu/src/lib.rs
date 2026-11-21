//! wgpu device management, buffers, the pipeline cache, the timestamp profiler, the readback ring and texture streaming.
//!
//! `humanart-gpu` is the layer between wgpu and the renderer, the UI and the runtime (WP-04). It knows
//! nothing about characters. It compiles for `wasm32-unknown-unknown` `[PLAT-03]`: no clocks, threads
//! or files; the few blocking helpers are compiled only on native targets.
//!
//! Module map:
//!
//! - [`device`]: instance, adapter and device creation the reference's way, headless creation, the
//!   limits request `[GPU-02]`, the capability model with its debug mask and tier `[PLAT-05]`, and
//!   surface configuration.
//! - [`error`]: [`GpuError`], wgpu error scopes as `Result`s, and the queue of uncaptured errors and
//!   device loss `[ERR-05]`.
//! - [`pod`]: the `unsafe trait Pod`, safe byte-casting helpers and the [`pod_struct!`] macro
//!   `[UNS-03]`. The only module with `unsafe` code.
//! - [`resources`]: the buffer, texture, texture-array and sampler manager with tagged memory
//!   accounting and budgets `[GPU-01]` `[GPU-04]` `[GPU-05]`.
//! - [`cache`]: shader modules, layouts, pipelines and bind groups created once under a key, with
//!   pre-warm accounting `[PERF-R4]`.
//! - [`uniform_ring`]: the per-frame uniform ring with dynamic offsets `[GPU-03]`.
//! - [`readback`]: the asynchronous readback ring of staging buffers `[PERF-R3]`.
//! - [`profiler`]: timestamp queries around passes, CPU frame time and pass counters as the
//!   fallback, overlay data and a JSON dump `[PERF-R6]`.
//! - [`streaming`]: virtual textures paged into a fixed tile pool with a page table and feedback
//!   `[SKN-03]`.
//! - [`lifecycle`]: the process-wide lock around device creation and drop, and the shared headless
//!   device of a process (a driver deadlock under parallel creation).
//! - [`recovery`]: device-loss detection and the rebuild on a new device `[ERR-05]`.
// [UNS-01] Unsafe code is allowed in this crate for POD casts only, and only in `pod`.
#![deny(unsafe_code)]

pub mod cache;
pub mod device;
pub mod error;
pub mod lifecycle;
#[allow(unsafe_code)]
pub mod pod;
pub mod profiler;
pub mod readback;
pub mod recovery;
pub mod resources;
pub mod streaming;
pub mod uniform_ring;

pub use device::{Capabilities, Capability, CapabilityOverrides, DeviceRequest, Gpu, Tier};
pub use error::{DeviceEvent, DeviceEvents, ErrorScope, GpuError};
pub use lifecycle::LifecycleGuard;
#[cfg(not(target_arch = "wasm32"))]
pub use lifecycle::shared_headless;
pub use pod::Pod;

/// Runs a call into wgpu.
///
/// The identity in normal builds. In this crate's unit tests it books the heap activity of the call
/// separately, which is how the steady-state test `[AC-WP04-3]` tells wgpu's own allocations (not
/// ours to remove) from this crate's (which must be zero). Every wgpu call on a per-frame path of
/// this crate goes through it.
#[inline(always)]
pub(crate) fn into_wgpu<R>(call: impl FnOnce() -> R) -> R {
    #[cfg(test)]
    {
        testing::outside_our_code(call)
    }
    #[cfg(not(test))]
    {
        call()
    }
}

// [AC-WP04-3] The unit-test binary counts allocations per thread; see `tests::steady_state`.
#[cfg(test)]
#[global_allocator]
static ALLOCATOR: humanart_core::alloc::CountingAllocator = humanart_core::alloc::CountingAllocator;

/// Shared helpers of the unit tests: one headless device per test binary (device creation is slow)
/// and blocking reads.
#[cfg(test)]
pub(crate) mod testing {
    use crate::device::Gpu;
    use std::cell::Cell;

    thread_local! {
        /// Allocator calls made inside wgpu on this thread, through `into_wgpu`.
        static INSIDE_WGPU: Cell<u64> = const { Cell::new(0) };
    }

    /// Runs `call` and books its allocator calls as wgpu's.
    pub(crate) fn outside_our_code<R>(call: impl FnOnce() -> R) -> R {
        let (value, stats) = humanart_core::alloc::measure(call);
        INSIDE_WGPU.with(|cell| cell.set(cell.get() + stats.events()));
        value
    }

    /// Allocator calls this thread made inside wgpu so far.
    pub(crate) fn inside_wgpu() -> u64 {
        INSIDE_WGPU.with(Cell::get)
    }

    /// The device every test shares (the process's shared device). Tests that lose a device make
    /// their own.
    pub(crate) fn gpu() -> &'static Gpu {
        crate::lifecycle::shared_headless().expect("this machine has a GPU that meets the baseline")
    }

    /// Waits for the GPU and reads a whole `MAP_READ` buffer.
    pub(crate) fn read_mappable(gpu: &Gpu, buffer: &wgpu::Buffer) -> Vec<u8> {
        buffer.slice(..).map_async(wgpu::MapMode::Read, |result| result.expect("the buffer maps"));
        crate::device::wait_for(&gpu.device, None, crate::device::GPU_WAIT_LIMIT, "the queued work")
            .expect("the device answers");
        let bytes = buffer.slice(..).get_mapped_range().expect("the buffer is mapped").to_vec();
        buffer.unmap();
        bytes
    }

    /// Copies `size` bytes of any `COPY_SRC` buffer to a staging buffer and reads them.
    pub(crate) fn read_buffer(gpu: &Gpu, buffer: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u8> {
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test") });
        encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
        gpu.queue.submit([encoder.finish()]);
        read_mappable(gpu, &staging)
    }
}

#[cfg(test)]
mod tests {
    use crate::cache::PipelineCache;
    use crate::profiler::GpuProfiler;
    use crate::readback::ReadbackRing;
    use crate::resources::{Budget, BufferDesc, Category, DirtyRanges, Resources, Tag};
    use crate::streaming::{StreamingDesc, StreamingPool, TileRequest};
    use crate::uniform_ring::{UniformRing, UniformRingDesc};
    use crate::{into_wgpu, testing};

    /// Test-only WGSL: adds the frame's constant to every element.
    const ADD: &str = "struct Frame { add: vec4<u32> }
@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&data) { data[id.x] = data[id.x] + frame.add.x; }
}";

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    enum Key {
        Add,
        Data,
    }

    /// `[AC-WP04-3]` `[PERF-R1]`: 600 frames through every per-frame path of the crate (uniform ring,
    /// dirty-range upload, cache lookups, profiler scopes, readback ring, texture streaming feedback and tile loads, device-loss check) make no
    /// heap allocation in this crate. wgpu's own allocations, booked by `into_wgpu`, are reported.
    #[test]
    fn ac_wp04_3_steady_state_frames_allocate_nothing_in_this_crate() {
        assert!(humanart_core::alloc::is_installed());
        let gpu = testing::gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        let mut ring = UniformRing::new(&mut resources, UniformRingDesc::new(4096, 16, "frame")).unwrap();
        let mut readback = ReadbackRing::new(&mut resources, 3, 256, "picking").unwrap();
        let mut profiler = GpuProfiler::new(gpu, &mut resources, 3, 4).unwrap();
        let (data, _) = resources
            .create_buffer(&BufferDesc {
                size: 1024,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                tag: Tag::new(Category::Deformation, 1, "data"),
            })
            .unwrap();
        let mut mirror = vec![0_u8; 1024];
        let mut dirty = DirtyRanges::new();
        dirty.mark_all(1024);

        // Startup: everything the frame needs is created here, once.
        let mut cache = PipelineCache::new(gpu);
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
        cache.shader(Key::Add, "add", ADD).unwrap();
        cache.bind_group_layout(Key::Data, "data", &[storage]).unwrap();
        let layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("add"),
            bind_group_layouts: &[Some(ring.layout()), cache.get_bind_group_layout(&Key::Data)],
            immediate_size: 0,
        });
        cache
            .compute_pipeline(Key::Add, |parts| {
                Ok(parts.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("add"),
                    layout: Some(&layout),
                    module: parts.shader(&Key::Add)?,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                }))
            })
            .unwrap();
        let binding = resources.buffer(data).unwrap().raw().as_entire_binding();
        cache
            .bind_group(Key::Data, "data", &Key::Data, &[wgpu::BindGroupEntry { binding: 0, resource: binding }])
            .unwrap();
        cache.mark_warm();
        let mut streaming = StreamingPool::new(
            &mut resources,
            StreamingDesc {
                tile_size: 8,
                format: wgpu::TextureFormat::Rgba8Unorm,
                tiles_per_side: 4,
                layers: 1,
                max_pages: 128,
                queue_capacity: 16,
                purpose: "streaming",
            },
        )
        .unwrap();
        let face = streaming.add_texture(64, 64).unwrap();

        let mut delivered = 0_u64;
        let mut run_frame = |frame: u32,
                             resources: &mut Resources,
                             ring: &mut UniformRing,
                             readback: &mut ReadbackRing,
                             profiler: &mut GpuProfiler,
                             mirror: &mut [u8],
                             dirty: &mut DirtyRanges| {
            assert!(!gpu.is_lost());
            profiler.begin_frame();
            ring.begin_frame();
            let offset = ring.push(&[1_u32, frame, 0, 0]).unwrap();
            ring.finish_frame(resources).unwrap();
            // A brush-like edit: a few bytes of the CPU mirror change and only they are uploaded.
            let at = (frame as usize * 12) % 1000;
            mirror[at..at + 4].copy_from_slice(&frame.to_le_bytes());
            dirty.mark(at as u64, 4);
            resources.flush_dirty(data, mirror, dirty).unwrap();

            let pipeline = cache.get_compute_pipeline(&Key::Add).unwrap();
            let group = cache.get_bind_group(&Key::Data).unwrap();
            let mut encoder =
                into_wgpu(|| gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default()));
            let timestamp_writes = profiler.compute_pass_writes("add");
            into_wgpu(|| {
                let mut pass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("add"), timestamp_writes });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, ring.bind_group(), &[offset]);
                pass.set_bind_group(1, group, &[]);
                pass.dispatch_workgroups(4, 1, 1);
            });
            if frame.is_multiple_of(4) {
                let buffer = resources.buffer(data).unwrap().raw();
                // A full ring only means this pick waits for a later frame.
                let _ = readback.copy_buffer(&mut encoder, buffer, 0, 16, u64::from(frame));
            }
            profiler.end_frame(&mut encoder);
            let commands = into_wgpu(|| encoder.finish());
            into_wgpu(|| gpu.queue.submit([commands]));
            readback.after_submit();
            profiler.after_submit();
            readback.poll(&gpu.device).unwrap();
            delivered += readback.drain_ready(|bytes| assert_eq!(bytes.packed_len(), 16)) as u64;
            profiler.collect();
            // Streaming: this frame's feedback (a wandering view over the face), then the loads.
            streaming.begin_frame();
            let feedback = [
                TileRequest { texture: face, mip: 0, x: (frame % 8) as u16, y: ((frame / 8) % 8) as u16 }.pack(),
                TileRequest { texture: face, mip: 1, x: ((frame / 3) % 4) as u16, y: 1 }.pack(),
                u32::MAX,
            ];
            streaming.note_feedback(&feedback);
            streaming
                .update(resources, 2, |tile, bytes| {
                    bytes.fill(tile.mip);
                    Ok(())
                })
                .unwrap();
        };

        for frame in 0..20 {
            run_frame(frame, &mut resources, &mut ring, &mut readback, &mut profiler, &mut mirror, &mut dirty);
        }
        let wgpu_before = testing::inside_wgpu();
        let ((), stats) = humanart_core::alloc::measure(|| {
            for frame in 20..620 {
                run_frame(frame, &mut resources, &mut ring, &mut readback, &mut profiler, &mut mirror, &mut dirty);
            }
        });
        let inside_wgpu = testing::inside_wgpu() - wgpu_before;
        let ours = stats.events() - inside_wgpu;
        println!(
            "steady state, 600 frames: {} allocator events in total, {inside_wgpu} inside wgpu, {ours} in humanart-gpu; \
             {delivered} readbacks delivered, {} profiler reports",
            stats.events(),
            profiler.stats().reports
        );
        assert_eq!(ours, 0, "humanart-gpu allocated in a steady-state frame");
        assert_eq!(cache.stats().late, 0);
        assert!(delivered > 0);
        println!("streaming during the loop: {:?}", streaming.stats());
        assert!(streaming.stats().evictions > 0, "the streaming path really ran");
        assert_eq!(gpu.events().pending(), 0, "no uncaptured GPU error");
    }
}
