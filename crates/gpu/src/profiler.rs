//! The GPU pass profiler: timestamp queries around passes, with CPU frame timing and pass counters
//! as the fallback `[PERF-R6]` (spec 14.5).
//!
//! Per frame:
//!
//! 1. [`GpuProfiler::begin_frame`].
//! 2. For every pass, [`GpuProfiler::compute_pass_writes`] or [`GpuProfiler::render_pass_writes`]
//!    with the pass's name; the result goes into the pass descriptor's `timestamp_writes`. It is
//!    `None` (the pass is only counted) when the device has no timestamp queries, the profiler is
//!    disabled, or the frame has used every scope.
//! 3. [`GpuProfiler::end_frame`] resolves the queries into the frame's staging buffer, in the same
//!    encoder, before it is submitted; [`GpuProfiler::after_submit`] after the submission.
//! 4. Every frame, after the device was polled, [`GpuProfiler::collect`] turns a finished frame into
//!    the [`FrameReport`] the overlay shows ([`GpuProfiler::latest`]) and
//!    [`GpuProfiler::write_dump`] writes as JSON.
//!
//! Results arrive a few frames late and nothing waits for them: each frame in flight has its own
//! query set and staging buffer, and a frame whose slot is still busy is counted and not timed. All
//! storage is made at creation; steady-state frames allocate nothing in this crate `[PERF-R1]`.
//! Disabled, every call is one branch.
//!
//! CPU frame time comes from `humanart_core::time` (the clock the application installs); it is 0
//! when no clock is installed.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::device::{Capability, Gpu};
use crate::error::GpuError;
use crate::into_wgpu;
use crate::resources::{BufferDesc, BufferId, Category, Resources, Tag};

/// Most timed passes per frame.
pub const MAX_SCOPES: usize = 64;
/// Most frames in flight the profiler keeps slots for.
pub const MAX_FRAMES_IN_FLIGHT: usize = 4;

/// Slot states, shared with the map callback.
mod state {
    /// Free for a new frame.
    pub const FREE: u8 = 0;
    /// Queries were resolved into the staging buffer; waiting for the submission.
    pub const RECORDED: u8 = 1;
    /// Mapping requested.
    pub const MAPPING: u8 = 2;
    /// Mapped: timestamps can be read.
    pub const READY: u8 = 3;
    /// Mapping failed (device lost).
    pub const FAILED: u8 = 4;
}

/// One timed pass of a report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeTiming {
    /// The pass's name.
    pub name: &'static str,
    /// GPU time between the start and the end of the pass, in nanoseconds; `None` without
    /// timestamp queries.
    pub gpu_nanos: Option<u64>,
}

/// What one frame cost: the overlay's data.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FrameReport {
    /// The frame's number (counted by [`GpuProfiler::begin_frame`]).
    pub frame: u64,
    /// CPU time from `begin_frame` to `end_frame`, in nanoseconds.
    pub cpu_nanos: u64,
    /// Passes recorded, timed or not (the fallback's pass counter).
    pub passes: u32,
    /// The timed passes in recording order (or, without timestamps, the counted ones).
    pub scopes: Vec<ScopeTiming>,
    /// True when the GPU times come from timestamp queries.
    pub gpu_timed: bool,
}

impl FrameReport {
    /// Sum of the passes' GPU times; `None` without timestamp queries.
    pub fn gpu_total_nanos(&self) -> Option<u64> {
        if !self.gpu_timed {
            return None;
        }
        Some(self.scopes.iter().filter_map(|scope| scope.gpu_nanos).sum())
    }
}

/// Counters across frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProfilerStats {
    /// Frames begun while enabled.
    pub frames: u64,
    /// Frames whose passes were recorded for GPU timing.
    pub recorded: u64,
    /// Reports produced (one per [`GpuProfiler::collect`] that found finished frames; older
    /// finished frames are dropped in favour of the newest).
    pub reports: u64,
    /// Frames not timed on the GPU because their slot was still waiting for an older frame.
    pub skipped: u64,
    /// Passes not timed because the frame had used every scope.
    pub overflowed: u64,
    /// Staging buffers that could not be mapped.
    pub failed: u64,
}

/// The per-frame-in-flight GPU objects.
#[derive(Debug)]
struct Slot {
    /// `2 * scopes` timestamp queries.
    query_set: wgpu::QuerySet,
    /// Where the queries are resolved (`QUERY_RESOLVE | COPY_SRC`).
    resolve: wgpu::Buffer,
    /// The mappable copy.
    staging: wgpu::Buffer,
    /// The two buffers in [`Resources`].
    ids: [BufferId; 2],
    /// See [`state`].
    state: Arc<AtomicU8>,
    /// The frame recorded into the slot.
    frame: u64,
    /// The frame's CPU time.
    cpu_nanos: u64,
    /// The frame's pass count.
    passes: u32,
    /// Names of the timed passes; capacity [`MAX_SCOPES`], allocated once.
    names: Vec<&'static str>,
}

/// The profiler. See the module documentation.
#[derive(Debug)]
pub struct GpuProfiler {
    /// One slot per frame in flight; empty without timestamp queries.
    slots: Vec<Slot>,
    /// Frames in flight asked for.
    frames_in_flight: usize,
    /// Timed passes per frame.
    scopes: u32,
    /// Nanoseconds per timestamp tick.
    period: f64,
    /// Whether anything is recorded.
    enabled: bool,
    /// The slot of the frame being recorded.
    current: usize,
    /// The frame being recorded may use its slot.
    recording: bool,
    /// Frame counter.
    frame: u64,
    /// `humanart_core::time::now_nanos` at `begin_frame`.
    frame_start: u64,
    /// Passes of the frame being recorded.
    passes: u32,
    /// Untimed pass names of the frame being recorded (fallback reports); capacity [`MAX_SCOPES`].
    counted: Vec<&'static str>,
    /// The newest report.
    latest: FrameReport,
    /// The counters.
    stats: ProfilerStats,
}

impl GpuProfiler {
    /// Makes the profiler for `frames_in_flight` frames (1 to 4) of up to `scopes` timed passes (1
    /// to 64). Without [`Capability::TimestampQueries`] no GPU object is made and the profiler only
    /// counts passes and times the CPU side.
    pub fn new(gpu: &Gpu, resources: &mut Resources, frames_in_flight: usize, scopes: usize) -> Result<Self, GpuError> {
        if !(1..=MAX_FRAMES_IN_FLIGHT).contains(&frames_in_flight) || !(1..=MAX_SCOPES).contains(&scopes) {
            return Err(GpuError::Invalid { what: "a profiler needs 1 to 4 frames in flight and 1 to 64 scopes" });
        }
        let mut profiler = Self {
            slots: Vec::with_capacity(frames_in_flight),
            frames_in_flight,
            scopes: scopes as u32,
            period: f64::from(gpu.queue.get_timestamp_period()),
            enabled: true,
            current: 0,
            recording: false,
            frame: 0,
            frame_start: 0,
            passes: 0,
            counted: Vec::with_capacity(MAX_SCOPES),
            latest: FrameReport { scopes: Vec::with_capacity(MAX_SCOPES), ..FrameReport::default() },
            stats: ProfilerStats::default(),
        };
        if gpu.has(Capability::TimestampQueries) {
            for _ in 0..frames_in_flight {
                match make_slot(gpu, resources, profiler.scopes) {
                    Ok(slot) => profiler.slots.push(slot),
                    Err(error) => {
                        profiler.destroy(resources);
                        return Err(error);
                    }
                }
            }
        }
        Ok(profiler)
    }

    /// Releases the staging buffers back to `resources`; the profiler only counts afterwards.
    pub fn destroy(&mut self, resources: &mut Resources) {
        for slot in self.slots.drain(..) {
            for id in slot.ids {
                // An id already gone has nothing left to release.
                let _ = resources.destroy_buffer(id);
            }
        }
    }

    /// True when passes are timed on the GPU.
    pub fn has_timestamps(&self) -> bool {
        !self.slots.is_empty()
    }

    /// Switches recording on or off. Off, every call is one branch and no report is made.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.recording = false;
    }

    /// True while recording.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// The newest finished frame, for the overlay.
    pub fn latest(&self) -> &FrameReport {
        &self.latest
    }

    /// The counters.
    pub fn stats(&self) -> &ProfilerStats {
        &self.stats
    }

    /// Starts a frame.
    pub fn begin_frame(&mut self) {
        if !self.enabled {
            return;
        }
        self.frame += 1;
        self.stats.frames += 1;
        self.frame_start = humanart_core::time::now_nanos();
        self.passes = 0;
        self.counted.clear();
        self.recording = false;
        if self.slots.is_empty() {
            return;
        }
        self.current = (self.current + 1) % self.slots.len();
        let slot = &mut self.slots[self.current];
        if slot.state.load(Ordering::Acquire) == state::FREE {
            slot.names.clear();
            slot.frame = self.frame;
            self.recording = true;
        } else {
            self.stats.skipped += 1;
        }
    }

    /// Counts a pass and, when it can be timed, returns its query indices.
    fn scope(&mut self, name: &'static str) -> Option<(&wgpu::QuerySet, u32)> {
        if !self.enabled {
            return None;
        }
        self.passes += 1;
        if !self.recording {
            if self.slots.is_empty() && self.counted.len() < MAX_SCOPES {
                self.counted.push(name);
            }
            return None;
        }
        let slot = self.slots.get_mut(self.current)?;
        if slot.names.len() >= self.scopes as usize {
            self.stats.overflowed += 1;
            return None;
        }
        let index = slot.names.len() as u32;
        slot.names.push(name);
        Some((&slot.query_set, 2 * index))
    }

    /// The timestamp writes for a compute pass named `name`, or `None` when it is only counted.
    pub fn compute_pass_writes(&mut self, name: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.scope(name).map(|(query_set, first)| wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(first),
            end_of_pass_write_index: Some(first + 1),
        })
    }

    /// The timestamp writes for a render pass named `name`, or `None` when it is only counted.
    pub fn render_pass_writes(&mut self, name: &'static str) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.scope(name).map(|(query_set, first)| wgpu::RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(first),
            end_of_pass_write_index: Some(first + 1),
        })
    }

    /// Ends the frame: records the query resolve into `encoder` (submit it with the frame's passes).
    /// Without timestamps the frame's report is made at once.
    pub fn end_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.enabled {
            return;
        }
        let cpu_nanos = humanart_core::time::now_nanos().saturating_sub(self.frame_start);
        if self.slots.is_empty() {
            self.latest.frame = self.frame;
            self.latest.cpu_nanos = cpu_nanos;
            self.latest.passes = self.passes;
            self.latest.gpu_timed = false;
            self.latest.scopes.clear();
            self.latest.scopes.extend(self.counted.iter().map(|&name| ScopeTiming { name, gpu_nanos: None }));
            self.stats.reports += 1;
            return;
        }
        if !std::mem::take(&mut self.recording) {
            return;
        }
        let passes = self.passes;
        let Some(slot) = self.slots.get_mut(self.current) else {
            return;
        };
        slot.cpu_nanos = cpu_nanos;
        slot.passes = passes;
        let queries = 2 * slot.names.len() as u32;
        if queries > 0 {
            let bytes = u64::from(queries) * u64::from(wgpu::QUERY_SIZE);
            into_wgpu(|| {
                encoder.resolve_query_set(&slot.query_set, 0..queries, &slot.resolve, 0);
                encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.staging, 0, bytes);
            });
        }
        slot.state.store(state::RECORDED, Ordering::Release);
        self.stats.recorded += 1;
    }

    /// Asks wgpu to map the frame's timestamps. Call once after the submission holding
    /// [`GpuProfiler::end_frame`]'s commands.
    pub fn after_submit(&mut self) {
        for slot in &self.slots {
            if slot.state.load(Ordering::Acquire) != state::RECORDED {
                continue;
            }
            if slot.names.is_empty() {
                // No timed pass: nothing to read, the report needs no GPU data.
                slot.state.store(state::READY, Ordering::Release);
                continue;
            }
            slot.state.store(state::MAPPING, Ordering::Release);
            let flag = Arc::clone(&slot.state);
            let bytes = 2 * slot.names.len() as u64 * u64::from(wgpu::QUERY_SIZE);
            into_wgpu(|| {
                slot.staging.slice(..bytes).map_async(wgpu::MapMode::Read, move |result| {
                    flag.store(if result.is_ok() { state::READY } else { state::FAILED }, Ordering::Release);
                });
            });
        }
    }

    /// Turns the newest finished frame into [`GpuProfiler::latest`] and frees every finished slot.
    /// Returns true when a new report was made. Poll the device before (once per frame).
    pub fn collect(&mut self) -> bool {
        let mut newest: Option<usize> = None;
        for index in 0..self.slots.len() {
            match self.slots[index].state.load(Ordering::Acquire) {
                state::READY => {
                    if newest.is_none_or(|best| self.slots[best].frame < self.slots[index].frame) {
                        newest = Some(index);
                    }
                }
                state::FAILED => {
                    self.stats.failed += 1;
                    self.slots[index].state.store(state::FREE, Ordering::Release);
                }
                _ => {}
            }
        }
        let Some(newest) = newest else {
            return false;
        };
        for index in 0..self.slots.len() {
            if self.slots[index].state.load(Ordering::Acquire) != state::READY {
                continue;
            }
            if index == newest {
                self.report(index);
            }
            let slot = &self.slots[index];
            if !slot.names.is_empty() {
                into_wgpu(|| slot.staging.unmap());
            }
            slot.state.store(state::FREE, Ordering::Release);
        }
        true
    }

    /// Reads a mapped slot into [`GpuProfiler::latest`].
    fn report(&mut self, index: usize) {
        let Some(slot) = self.slots.get(index) else {
            return;
        };
        let latest = &mut self.latest;
        latest.frame = slot.frame;
        latest.cpu_nanos = slot.cpu_nanos;
        latest.passes = slot.passes;
        latest.gpu_timed = true;
        latest.scopes.clear();
        if slot.names.is_empty() {
            self.stats.reports += 1;
            return;
        }
        let bytes = 2 * slot.names.len() as u64 * u64::from(wgpu::QUERY_SIZE);
        let Ok(view) = into_wgpu(|| slot.staging.slice(..bytes).get_mapped_range()) else {
            self.stats.failed += 1;
            return;
        };
        for (pair, &name) in view.as_chunks::<16>().0.iter().zip(&slot.names) {
            let (start, end) = pair.split_at(8);
            let tick = |bytes: &[u8]| bytes.try_into().map_or(0, u64::from_le_bytes);
            let ticks = tick(end).saturating_sub(tick(start));
            let gpu_nanos = (ticks as f64 * self.period).round() as u64;
            latest.scopes.push(ScopeTiming { name, gpu_nanos: Some(gpu_nanos) });
        }
        into_wgpu(|| drop(view));
        self.stats.reports += 1;
    }

    /// Writes [`GpuProfiler::latest`] as one line of JSON: the machine-readable dump `[PERF-R6]`.
    pub fn write_dump(&self, out: &mut impl fmt::Write) -> fmt::Result {
        let report = &self.latest;
        write!(
            out,
            "{{\"frame\":{},\"cpu_ns\":{},\"passes\":{},\"gpu_timed\":{},",
            report.frame, report.cpu_nanos, report.passes, report.gpu_timed
        )?;
        match report.gpu_total_nanos() {
            Some(total) => write!(out, "\"gpu_ns\":{total},")?,
            None => out.write_str("\"gpu_ns\":null,")?,
        }
        out.write_str("\"scopes\":[")?;
        for (index, scope) in report.scopes.iter().enumerate() {
            if index > 0 {
                out.write_char(',')?;
            }
            out.write_str("{\"name\":\"")?;
            for c in scope.name.chars() {
                match c {
                    '"' => out.write_str("\\\"")?,
                    '\\' => out.write_str("\\\\")?,
                    c if c.is_control() => write!(out, "\\u{:04x}", u32::from(c))?,
                    c => out.write_char(c)?,
                }
            }
            match scope.gpu_nanos {
                Some(nanos) => write!(out, "\",\"gpu_ns\":{nanos}}}")?,
                None => out.write_str("\",\"gpu_ns\":null}")?,
            }
        }
        out.write_str("]}")
    }

    /// Makes the GPU objects again on `gpu`'s device after a loss `[ERR-05]`. The old staging
    /// buffers must already be gone from `resources` or be released here; frames in flight are
    /// dropped.
    pub fn rebuild(&mut self, gpu: &Gpu, resources: &mut Resources) -> Result<(), GpuError> {
        self.destroy(resources);
        self.period = f64::from(gpu.queue.get_timestamp_period());
        self.recording = false;
        if gpu.has(Capability::TimestampQueries) {
            for _ in 0..self.frames_in_flight {
                self.slots.push(make_slot(gpu, resources, self.scopes)?);
            }
        }
        Ok(())
    }
}

/// Makes one frame slot: the query set and its two buffers.
fn make_slot(gpu: &Gpu, resources: &mut Resources, scopes: u32) -> Result<Slot, GpuError> {
    let queries = 2 * scopes;
    let bytes = u64::from(queries) * u64::from(wgpu::QUERY_SIZE);
    let query_set = crate::error::guarded(&gpu.device, "timestamp query set", || {
        gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("profiler"),
            ty: wgpu::QueryType::Timestamp,
            count: queries,
        })
    })?;
    let (resolve, _) = resources.create_buffer(&BufferDesc {
        size: bytes,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        tag: Tag::shared(Category::Staging, "profiler resolve"),
    })?;
    let staging = match resources.create_buffer(&BufferDesc {
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        tag: Tag::shared(Category::Staging, "profiler staging"),
    }) {
        Ok((id, _)) => id,
        Err(error) => {
            let _ = resources.destroy_buffer(resolve);
            return Err(error);
        }
    };
    let raw = |id: BufferId| resources.buffer(id).map(|entry| entry.raw().clone());
    let (Some(resolve_buffer), Some(staging_buffer)) = (raw(resolve), raw(staging)) else {
        return Err(GpuError::Stale { what: "profiler buffer" });
    };
    Ok(Slot {
        query_set,
        resolve: resolve_buffer,
        staging: staging_buffer,
        ids: [resolve, staging],
        state: Arc::new(AtomicU8::new(state::FREE)),
        frame: 0,
        cpu_nanos: 0,
        passes: 0,
        names: Vec::with_capacity(scopes as usize),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{CapabilityOverrides, DeviceRequest};
    use crate::resources::Budget;
    use crate::testing;

    /// Test-only WGSL: a loop long enough to take measurable GPU time.
    const BUSY: &str = "@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var x = id.x;
    for (var i = 0u; i < 20000u; i++) { x = x * 1664525u + 1013904223u; }
    data[id.x] = x;
}";

    /// A compute pipeline and bind group running [`BUSY`] over 4096 invocations.
    fn busy_work(gpu: &Gpu) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("busy"),
            source: wgpu::ShaderSource::Wgsl(BUSY.into()),
        });
        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("busy"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("busy"),
            size: 4096 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("busy"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        (pipeline, group)
    }

    /// Records one frame: a busy pass and an empty pass, both through the profiler.
    fn frame(gpu: &Gpu, profiler: &mut GpuProfiler, work: &(wgpu::ComputePipeline, wgpu::BindGroup)) {
        profiler.begin_frame();
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let timestamp_writes = profiler.compute_pass_writes("busy");
            let mut pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("busy"), timestamp_writes });
            pass.set_pipeline(&work.0);
            pass.set_bind_group(0, &work.1, &[]);
            pass.dispatch_workgroups(64, 1, 1);
        }
        {
            let timestamp_writes = profiler.compute_pass_writes("empty \"quoted\"");
            let _pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("empty"), timestamp_writes });
        }
        profiler.end_frame(&mut encoder);
        gpu.queue.submit([encoder.finish()]);
        profiler.after_submit();
    }

    #[test]
    fn perf_r6_passes_are_timed_with_timestamp_queries() {
        let gpu = testing::gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        let mut profiler = GpuProfiler::new(gpu, &mut resources, 3, 8).unwrap();
        assert_eq!(profiler.has_timestamps(), gpu.has(Capability::TimestampQueries));
        assert!(profiler.has_timestamps(), "this machine's adapter offers timestamp queries");
        let work = busy_work(gpu);
        let mut reports = 0;
        for _ in 0..6 {
            frame(gpu, &mut profiler, &work);
            gpu.device.poll(wgpu::PollType::Poll).unwrap();
            reports += usize::from(profiler.collect());
        }
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        reports += usize::from(profiler.collect());
        assert!(reports >= 1);
        let report = profiler.latest().clone();
        assert!(report.gpu_timed);
        assert_eq!(report.passes, 2);
        assert_eq!(report.scopes.len(), 2);
        assert_eq!(report.scopes[0].name, "busy");
        let busy = report.scopes[0].gpu_nanos.unwrap();
        let empty = report.scopes[1].gpu_nanos.unwrap();
        assert!(busy > 0 && busy > empty, "busy {busy} ns, empty {empty} ns");
        assert_eq!(report.gpu_total_nanos(), Some(busy + empty));
        let stats = *profiler.stats();
        assert_eq!(stats.frames, 6);
        assert_eq!(stats.recorded + stats.skipped, 6, "{stats:?}");
        let mut dump = String::new();
        profiler.write_dump(&mut dump).unwrap();
        assert!(dump.starts_with(&format!("{{\"frame\":{},", report.frame)), "{dump}");
        assert!(dump.contains(&format!("{{\"name\":\"busy\",\"gpu_ns\":{busy}}}")), "{dump}");
        assert!(dump.contains("\"name\":\"empty \\\"quoted\\\"\""), "{dump}");
        assert!(dump.ends_with("]}"));
        println!("profiler: {dump} ({} of 6 frames timed, {} skipped)", stats.recorded, stats.skipped);

        // Disabled: nothing is recorded or counted.
        profiler.set_enabled(false);
        frame(gpu, &mut profiler, &work);
        assert_eq!(profiler.stats().frames, 6);
        assert!(profiler.compute_pass_writes("off").is_none());
        profiler.set_enabled(true);
        assert!(profiler.is_enabled());

        // A frame with more passes than scopes times the first ones and counts the rest.
        let mut small = GpuProfiler::new(gpu, &mut resources, 1, 1).unwrap();
        frame(gpu, &mut small, &work);
        assert_eq!(small.stats().overflowed, 1);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert!(small.collect());
        assert_eq!((small.latest().passes, small.latest().scopes.len()), (2, 1));
        small.destroy(&mut resources);
        profiler.destroy(&mut resources);
        assert_eq!(resources.live(), (0, 0));
        assert!(GpuProfiler::new(gpu, &mut resources, 5, 8).is_err());
        assert!(GpuProfiler::new(gpu, &mut resources, 1, 0).is_err());
        assert_eq!(gpu.events().pending(), 0);
    }

    #[test]
    fn perf_r6_without_timestamps_the_fallback_counts_passes_and_times_the_cpu() {
        let clock = humanart_core::platform::ManualClock::new(0);
        clock.set_auto_advance(250_000);
        let installed = humanart_core::time::install_clock(Arc::new(clock));
        let request =
            DeviceRequest::default().with_overrides(CapabilityOverrides::NONE.disable(Capability::TimestampQueries));
        let gpu = Gpu::headless_blocking(&request).unwrap();
        let mut resources = Resources::new(&gpu, Budget::UNLIMITED);
        let mut profiler = GpuProfiler::new(&gpu, &mut resources, 2, 8).unwrap();
        assert!(!profiler.has_timestamps());
        assert_eq!(resources.live(), (0, 0), "no GPU objects without timestamps");
        let work = busy_work(&gpu);
        frame(&gpu, &mut profiler, &work);
        let report = profiler.latest();
        assert!(!report.gpu_timed);
        assert_eq!(report.passes, 2);
        assert_eq!(report.scopes.iter().map(|scope| scope.name).collect::<Vec<_>>(), ["busy", "empty \"quoted\""]);
        assert!(report.scopes.iter().all(|scope| scope.gpu_nanos.is_none()));
        assert_eq!(report.gpu_total_nanos(), None);
        if installed {
            // At least the one step between begin_frame and end_frame; tests running in parallel
            // read the same process clock and may add steps.
            assert!(report.cpu_nanos >= 250_000 && report.cpu_nanos.is_multiple_of(250_000), "{}", report.cpu_nanos);
        }
        let mut dump = String::new();
        profiler.write_dump(&mut dump).unwrap();
        assert!(dump.contains("\"gpu_ns\":null,\"scopes\":[{\"name\":\"busy\",\"gpu_ns\":null}"), "{dump}");
        assert!(!profiler.collect(), "nothing to collect: the report was made at end_frame");
    }
}
