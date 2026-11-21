//! The asynchronous readback ring `[PERF-R3]`.
//!
//! Picking, metrics, screenshots and agent renders read GPU data back without stalling the frame:
//!
//! 1. While recording, [`ReadbackRing::copy_buffer`] or [`ReadbackRing::copy_texture`] puts a copy into
//!    a free staging slot and returns a [`ReadbackTicket`].
//! 2. After the queue submission that holds the copies, [`ReadbackRing::after_submit`] asks wgpu to
//!    map the slots. Nothing waits.
//! 3. One or more frames later, after [`ReadbackRing::poll`] (native) or on its own (browser), a slot
//!    is ready; [`ReadbackRing::take`] or [`ReadbackRing::drain_ready`] hands its bytes to the caller
//!    and frees the slot.
//!
//! The slots are a fixed set of staging buffers made once through [`Resources`] (tagged
//! [`Category::Staging`], `[GPU-05]`); when all are busy a request is [`GpuError::Full`], never a new
//! allocation. Steps 1 to 3 allocate nothing in this crate `[PERF-R1]`.
//!
//! [`ReadbackRing::read_blocking`] is the one synchronous path, for the command-line tool and tests on
//! native targets.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::error::GpuError;
use crate::into_wgpu;
use crate::resources::{BufferDesc, BufferId, Category, Resources, Tag, block_bytes};

/// Most slots a ring may have.
pub const MAX_SLOTS: usize = 64;

/// Slot states, stored in an atomic shared with wgpu's map callback.
mod state {
    /// Free for a new request.
    pub const FREE: u8 = 0;
    /// A copy was recorded; waiting for [`super::ReadbackRing::after_submit`].
    pub const RECORDED: u8 = 1;
    /// Mapping was requested; waiting for the GPU.
    pub const MAPPING: u8 = 2;
    /// Mapped: the bytes can be read.
    pub const READY: u8 = 3;
    /// Mapping failed (device lost).
    pub const FAILED: u8 = 4;
}

/// Names one pending readback. Stays unique: a slot reused later gets a new generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadbackTicket {
    /// The slot.
    slot: u32,
    /// The slot's generation when the request was made.
    generation: u32,
}

/// How the bytes of one slot are laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Layout {
    /// Bytes mapped: whole rows, padded.
    len: u64,
    /// Meaningful bytes per row.
    row_bytes: u32,
    /// Bytes from one row to the next in the slot (256-aligned for textures).
    stride: u32,
    /// Rows (of texel blocks), over all layers.
    rows: u32,
}

/// One staging buffer of the ring.
#[derive(Debug)]
struct Slot {
    /// The buffer in [`Resources`].
    id: BufferId,
    /// A handle to the same wgpu buffer.
    buffer: wgpu::Buffer,
    /// See [`state`]; shared with the map callback.
    state: Arc<AtomicU8>,
    /// Bumped whenever the slot is freed.
    generation: u32,
    /// The current request's layout.
    layout: Layout,
    /// The caller's value for the current request.
    user: u64,
}

/// The bytes of a finished readback, handed to the closures of [`ReadbackRing::take`] and
/// [`ReadbackRing::drain_ready`].
#[derive(Debug)]
pub struct Readback<'a> {
    /// The mapped bytes, rows padded to `stride`.
    bytes: &'a [u8],
    /// The layout.
    layout: Layout,
    /// The ticket.
    ticket: ReadbackTicket,
    /// The caller's value given with the request.
    user: u64,
}

impl Readback<'_> {
    /// The ticket of the request.
    pub fn ticket(&self) -> ReadbackTicket {
        self.ticket
    }

    /// The value the caller gave with the request (a pick id, a frame number).
    pub fn user(&self) -> u64 {
        self.user
    }

    /// Number of rows: 1 for a buffer, the rows of texel blocks for a texture (all layers).
    pub fn rows(&self) -> u32 {
        self.layout.rows
    }

    /// Meaningful bytes per row.
    pub fn row_bytes(&self) -> u32 {
        self.layout.row_bytes
    }

    /// Tightly packed size: `rows * row_bytes`.
    pub fn packed_len(&self) -> usize {
        self.layout.rows as usize * self.layout.row_bytes as usize
    }

    /// One row without its padding, or `None` past the last row.
    pub fn row(&self, index: u32) -> Option<&[u8]> {
        if index >= self.layout.rows {
            return None;
        }
        let start = index as usize * self.layout.stride as usize;
        self.bytes.get(start..start + self.layout.row_bytes as usize)
    }

    /// Copies the rows, tightly packed, to the front of `out` and returns the bytes written.
    /// `out` shorter than [`Readback::packed_len`] is [`GpuError::OutOfRange`].
    pub fn copy_to(&self, out: &mut [u8]) -> Result<usize, GpuError> {
        let len = self.packed_len();
        if out.len() < len {
            return Err(GpuError::OutOfRange {
                what: "readback destination",
                offset: 0,
                len: len as u64,
                capacity: out.len() as u64,
            });
        }
        let row_bytes = self.layout.row_bytes as usize;
        for (index, target) in out[..len].chunks_exact_mut(row_bytes.max(1)).enumerate() {
            let row = self.row(index as u32).ok_or(GpuError::Stale { what: "readback row" })?;
            target.copy_from_slice(row);
        }
        Ok(len)
    }
}

/// What the ring has done.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReadbackStats {
    /// Requests accepted.
    pub requested: u64,
    /// Readbacks handed to the caller.
    pub delivered: u64,
    /// Requests refused because every slot was busy.
    pub full: u64,
    /// Mappings that failed.
    pub failed: u64,
}

/// The ring. See the module documentation.
#[derive(Debug)]
pub struct ReadbackRing {
    /// The staging slots, fixed at creation.
    slots: Vec<Slot>,
    /// Bytes per slot.
    slot_size: u64,
    /// The counters.
    stats: ReadbackStats,
}

impl ReadbackRing {
    /// Makes `slots` staging buffers of `slot_size` bytes each (rounded up to 256) in `resources`.
    pub fn new(
        resources: &mut Resources,
        slots: usize,
        slot_size: u64,
        purpose: &'static str,
    ) -> Result<Self, GpuError> {
        if slots == 0 || slots > MAX_SLOTS {
            return Err(GpuError::Invalid { what: "a readback ring needs between 1 and 64 slots" });
        }
        let slot_size = slot_size
            .checked_next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT.into())
            .filter(|&size| size > 0)
            .ok_or(GpuError::Invalid { what: "a readback slot size of zero or one that overflows" })?;
        let mut ring = Self { slots: Vec::with_capacity(slots), slot_size, stats: ReadbackStats::default() };
        for _ in 0..slots {
            let made = resources.create_buffer(&BufferDesc {
                size: slot_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                tag: Tag::shared(Category::Staging, purpose),
            });
            let id = match made {
                Ok((id, _)) => id,
                Err(error) => {
                    ring.destroy(resources);
                    return Err(error);
                }
            };
            let buffer = resources.buffer(id).ok_or(GpuError::Stale { what: "readback slot buffer" })?.raw().clone();
            ring.slots.push(Slot {
                id,
                buffer,
                state: Arc::new(AtomicU8::new(state::FREE)),
                generation: 0,
                layout: Layout::default(),
                user: 0,
            });
        }
        Ok(ring)
    }

    /// Releases every slot's buffer back to `resources`. The ring is empty afterwards.
    pub fn destroy(&mut self, resources: &mut Resources) {
        for slot in self.slots.drain(..) {
            // A buffer already gone (a stale id) has nothing left to release.
            let _ = resources.destroy_buffer(slot.id);
        }
    }

    /// Bytes per slot.
    pub fn slot_size(&self) -> u64 {
        self.slot_size
    }

    /// Number of slots.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Slots free for a new request.
    pub fn free_slots(&self) -> usize {
        self.slots.iter().filter(|slot| slot.state.load(Ordering::Acquire) == state::FREE).count()
    }

    /// The counters.
    pub fn stats(&self) -> &ReadbackStats {
        &self.stats
    }

    /// A free slot for a request of `len` bytes.
    fn claim(&mut self, len: u64) -> Result<usize, GpuError> {
        if len == 0 || len > self.slot_size {
            return Err(GpuError::OutOfRange {
                what: "readback size against the slot size",
                offset: 0,
                len,
                capacity: self.slot_size,
            });
        }
        let free = self.slots.iter().position(|slot| slot.state.load(Ordering::Acquire) == state::FREE);
        if free.is_none() {
            self.stats.full += 1;
        }
        free.ok_or(GpuError::Full { what: "the readback ring" })
    }

    /// Marks a claimed slot as recorded and makes its ticket.
    fn record(&mut self, index: usize, layout: Layout, user: u64) -> Result<ReadbackTicket, GpuError> {
        let slot = self.slots.get_mut(index).ok_or(GpuError::Stale { what: "readback slot" })?;
        slot.layout = layout;
        slot.user = user;
        slot.state.store(state::RECORDED, Ordering::Release);
        self.stats.requested += 1;
        Ok(ReadbackTicket { slot: index as u32, generation: slot.generation })
    }

    /// Records a copy of `size` bytes of `source` from `offset` into a free slot. `source` needs
    /// `COPY_SRC`; offset and size must be multiples of 4. `user` comes back with the bytes.
    pub fn copy_buffer(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        offset: u64,
        size: u64,
        user: u64,
    ) -> Result<ReadbackTicket, GpuError> {
        if !offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) || !size.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
            return Err(GpuError::Invalid { what: "a buffer readback that is not aligned to 4 bytes" });
        }
        if offset.checked_add(size).is_none_or(|end| end > source.size()) {
            return Err(GpuError::OutOfRange { what: "buffer readback", offset, len: size, capacity: source.size() });
        }
        let index = self.claim(size)?;
        let slot = &self.slots[index];
        into_wgpu(|| encoder.copy_buffer_to_buffer(source, offset, &slot.buffer, 0, size));
        let row_bytes = u32::try_from(size).map_err(|_| GpuError::Invalid { what: "a readback above 4 GiB" })?;
        self.record(index, Layout { len: size, row_bytes, stride: row_bytes, rows: 1 }, user)
    }

    /// Records a copy of a `width` x `height` rectangle of layers `origin[2]..origin[2] + layers` of
    /// one mip level into a free slot. The texture needs `COPY_SRC`; the format must be copyable as
    /// a whole (not a combined depth-stencil format).
    pub fn copy_texture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        mip: u32,
        origin: [u32; 3],
        size: [u32; 3],
        user: u64,
    ) -> Result<ReadbackTicket, GpuError> {
        let format = texture.format();
        if format.block_copy_size(None).is_none() {
            return Err(GpuError::Invalid { what: "a texture readback of a format with several aspects" });
        }
        let [width, height, layers] = size;
        if width == 0 || height == 0 || layers == 0 || mip >= texture.mip_level_count() {
            return Err(GpuError::Invalid { what: "an empty texture readback or one past the mip levels" });
        }
        let level = texture.size().mip_level_size(mip, texture.dimension());
        let inside = |start: u32, len: u32, max: u32| start.checked_add(len).is_some_and(|end| end <= max);
        if !inside(origin[0], width, level.width)
            || !inside(origin[1], height, level.height)
            || !inside(origin[2], layers, level.depth_or_array_layers)
        {
            return Err(GpuError::OutOfRange {
                what: "texture readback rectangle",
                offset: u64::from(origin[0].max(origin[1])),
                len: u64::from(width.max(height)),
                capacity: u64::from(level.width.min(level.height)),
            });
        }
        let (block_width, block_height) = format.block_dimensions();
        let row_bytes = width.div_ceil(block_width).saturating_mul(block_bytes(format));
        let stride = row_bytes
            .checked_next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            .ok_or(GpuError::Invalid { what: "a texture readback row above 4 GiB" })?;
        let rows_per_layer = height.div_ceil(block_height);
        let rows = rows_per_layer.saturating_mul(layers);
        let len = u64::from(stride) * u64::from(rows);
        let index = self.claim(len)?;
        let slot = &self.slots[index];
        into_wgpu(|| {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d { x: origin[0], y: origin[1], z: origin[2] },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &slot.buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(stride),
                        rows_per_image: Some(rows_per_layer),
                    },
                },
                wgpu::Extent3d { width, height, depth_or_array_layers: layers },
            );
        });
        self.record(index, Layout { len, row_bytes, stride, rows }, user)
    }

    /// Asks wgpu to map every slot whose copy was recorded. Call it once after the queue
    /// submission that holds the copies (mapping earlier would make that submission invalid).
    pub fn after_submit(&mut self) {
        for slot in &self.slots {
            if slot.state.load(Ordering::Acquire) != state::RECORDED {
                continue;
            }
            slot.state.store(state::MAPPING, Ordering::Release);
            let flag = Arc::clone(&slot.state);
            into_wgpu(|| {
                slot.buffer.slice(..slot.layout.len).map_async(wgpu::MapMode::Read, move |result| {
                    flag.store(if result.is_ok() { state::READY } else { state::FAILED }, Ordering::Release);
                });
            });
        }
    }

    /// Lets wgpu run finished map callbacks without waiting (native). In a browser the callbacks
    /// run on their own and this does nothing.
    pub fn poll(&self, device: &wgpu::Device) -> Result<(), GpuError> {
        into_wgpu(|| device.poll(wgpu::PollType::Poll))?;
        Ok(())
    }

    /// True when the ticket's bytes can be taken.
    pub fn is_ready(&self, ticket: ReadbackTicket) -> bool {
        self.slots.get(ticket.slot as usize).is_some_and(|slot| {
            slot.generation == ticket.generation && slot.state.load(Ordering::Acquire) == state::READY
        })
    }

    /// Hands the ticket's bytes to `read` and frees the slot. `Ok(None)` while the GPU is not done;
    /// [`GpuError::Stale`] for a ticket already taken; [`GpuError::Map`] when mapping failed (the
    /// slot is freed too).
    pub fn take<R>(
        &mut self,
        ticket: ReadbackTicket,
        read: impl FnOnce(&Readback<'_>) -> R,
    ) -> Result<Option<R>, GpuError> {
        let slot = self
            .slots
            .get(ticket.slot as usize)
            .filter(|slot| slot.generation == ticket.generation)
            .ok_or(GpuError::Stale { what: "readback ticket" })?;
        match slot.state.load(Ordering::Acquire) {
            state::READY => self.deliver(ticket.slot as usize, read).map(Some),
            state::FAILED => {
                self.stats.failed += 1;
                self.free(ticket.slot as usize);
                Err(GpuError::Map { message: String::from("the readback buffer could not be mapped") })
            }
            state::FREE => Err(GpuError::Stale { what: "readback ticket" }),
            _ => Ok(None),
        }
    }

    /// Hands every ready readback to `read`, oldest slot first, and frees the slots. Failed slots
    /// are freed and counted in [`ReadbackStats::failed`]. Returns how many were delivered.
    pub fn drain_ready(&mut self, mut read: impl FnMut(&Readback<'_>)) -> usize {
        let mut delivered = 0;
        for index in 0..self.slots.len() {
            match self.slots[index].state.load(Ordering::Acquire) {
                state::READY => {
                    if self.deliver(index, &mut read).is_ok() {
                        delivered += 1;
                    }
                }
                state::FAILED => {
                    self.stats.failed += 1;
                    self.free(index);
                }
                _ => {}
            }
        }
        delivered
    }

    /// Reads a mapped slot, unmaps it and frees it.
    fn deliver<R>(&mut self, index: usize, read: impl FnOnce(&Readback<'_>) -> R) -> Result<R, GpuError> {
        let slot = self.slots.get(index).ok_or(GpuError::Stale { what: "readback slot" })?;
        let ticket = ReadbackTicket { slot: index as u32, generation: slot.generation };
        let result = {
            let mapped = into_wgpu(|| slot.buffer.slice(..slot.layout.len).get_mapped_range());
            match mapped {
                Ok(view) => {
                    let value = read(&Readback { bytes: &view, layout: slot.layout, ticket, user: slot.user });
                    into_wgpu(|| drop(view));
                    Ok(value)
                }
                Err(error) => Err(GpuError::Map { message: error.to_string() }),
            }
        };
        into_wgpu(|| slot.buffer.unmap());
        if result.is_ok() {
            self.stats.delivered += 1;
        } else {
            self.stats.failed += 1;
        }
        self.free(index);
        result
    }

    /// Frees a slot and retires its ticket.
    fn free(&mut self, index: usize) {
        if let Some(slot) = self.slots.get_mut(index) {
            slot.generation = slot.generation.wrapping_add(1);
            slot.state.store(state::FREE, Ordering::Release);
        }
    }

    /// Waits for one readback and copies its rows, tightly packed, into `out`. Native only: for the
    /// command-line tool and tests, never for frame code.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_blocking(
        &mut self,
        device: &wgpu::Device,
        ticket: ReadbackTicket,
        out: &mut [u8],
    ) -> Result<usize, GpuError> {
        let recorded = self.slots.get(ticket.slot as usize).is_some_and(|slot| {
            slot.generation == ticket.generation && slot.state.load(Ordering::Acquire) == state::RECORDED
        });
        if recorded {
            return Err(GpuError::Invalid { what: "a readback waited for before after_submit" });
        }
        // One wait finishes every submission and runs the map callbacks; the second round only
        // covers a callback that lands between the two checks.
        for _ in 0..2 {
            if let Some(result) = self.take(ticket, |readback| readback.copy_to(out))? {
                return result;
            }
            crate::device::wait_for(device, None, crate::device::GPU_WAIT_LIMIT, "a readback")?;
        }
        match self.take(ticket, |readback| readback.copy_to(out))? {
            Some(result) => result,
            None => {
                Err(GpuError::Internal { scope: "readback", message: String::from("the GPU never finished the copy") })
            }
        }
    }

    /// Points the slots at the buffers [`Resources::rebuild`] made after a device loss `[ERR-05]`.
    /// Pending requests are dropped and their tickets become stale.
    pub fn rebuild(&mut self, resources: &Resources) -> Result<(), GpuError> {
        for index in 0..self.slots.len() {
            let id = self.slots[index].id;
            let buffer = resources.buffer(id).ok_or(GpuError::Stale { what: "readback slot buffer" })?.raw().clone();
            let slot = &mut self.slots[index];
            slot.buffer = buffer;
            // The old map callback may still fire for the lost buffer: give the slot a new flag.
            slot.state = Arc::new(AtomicU8::new(state::FREE));
            slot.generation = slot.generation.wrapping_add(1);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{Budget, TextureDesc};
    use crate::testing;

    fn source(gpu: &crate::Gpu, words: &[u32]) -> wgpu::Buffer {
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("source"),
            size: (words.len() * 4) as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&buffer, 0, crate::pod::slice_as_bytes(words));
        buffer
    }

    fn new_encoder(gpu: &crate::Gpu) -> wgpu::CommandEncoder {
        gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback test") })
    }

    #[test]
    fn perf_r3_frames_never_wait_and_every_readback_arrives_later() {
        let gpu = testing::gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        let mut ring = ReadbackRing::new(&mut resources, 4, 64, "picking").unwrap();
        assert_eq!(resources.stats().of(Category::Staging), 4 * 256, "slots are accounted [GPU-05]");
        let words: Vec<u32> = (0..16).map(|i| i * 7).collect();
        let buffer = source(gpu, &words);

        // Twelve frames of about 4 ms, one pick request each, only non-blocking polls: requests are
        // answered in later frames, each with its own value, and nothing is lost. On a GPU shared
        // with other processes the first answers can take longer than twelve frames, so frames go
        // on (up to 2 s) until one has arrived; no frame ever waits.
        let mut received = Vec::new();
        let mut pending = 0;
        let mut frames = 0_u64;
        for frame in 0..500_u64 {
            if frame >= 12 && !received.is_empty() {
                break;
            }
            frames += 1;
            std::thread::sleep(std::time::Duration::from_millis(4));
            let mut encoder = new_encoder(gpu);
            let offset = (frame % 16) * 4;
            match ring.copy_buffer(&mut encoder, &buffer, offset, 4, frame) {
                Ok(_) => pending += 1,
                Err(GpuError::Full { .. }) => {}
                Err(other) => panic!("{other}"),
            }
            gpu.queue.submit([encoder.finish()]);
            ring.after_submit();
            ring.poll(&gpu.device).unwrap();
            pending -= ring.drain_ready(|readback| {
                let mut word = [0_u8; 4];
                readback.copy_to(&mut word).unwrap();
                received.push((readback.user(), u32::from_le_bytes(word)));
            });
        }
        let during_frames = received.len();
        assert!(during_frames > 0, "readbacks arrive in later frames without any wait");
        while pending > 0 {
            crate::device::wait_for(&gpu.device, None, crate::device::GPU_WAIT_LIMIT, "the queued work").unwrap();
            pending -= ring.drain_ready(|readback| {
                received.push((readback.user(), u32::from_le_bytes(readback.row(0).unwrap().try_into().unwrap())));
            });
        }
        assert_eq!(received.len() as u64 + ring.stats().full, frames);
        for &(frame, value) in &received {
            assert_eq!(value, words[(frame % 16) as usize], "frame {frame}");
        }
        assert_eq!(ring.free_slots(), 4);
        assert_eq!(ring.stats().delivered, received.len() as u64);
        println!(
            "readback: {} delivered ({during_frames} during the frames), {} refused while all slots were busy",
            received.len(),
            ring.stats().full
        );
    }

    #[test]
    fn perf_r3_tickets_are_taken_once_and_a_full_ring_refuses() {
        let gpu = testing::gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        let mut ring = ReadbackRing::new(&mut resources, 2, 16, "small").unwrap();
        let buffer = source(gpu, &[1, 2, 3, 4]);
        let mut encoder = new_encoder(gpu);
        let first = ring.copy_buffer(&mut encoder, &buffer, 0, 16, 1).unwrap();
        let second = ring.copy_buffer(&mut encoder, &buffer, 8, 8, 2).unwrap();
        assert!(matches!(ring.copy_buffer(&mut encoder, &buffer, 0, 4, 3), Err(GpuError::Full { .. })));
        let mut out = [0_u8; 16];
        assert!(matches!(ring.read_blocking(&gpu.device, first, &mut out), Err(GpuError::Invalid { .. })));
        gpu.queue.submit([encoder.finish()]);
        ring.after_submit();
        assert_eq!(ring.read_blocking(&gpu.device, first, &mut out).unwrap(), 16);
        assert_eq!(out, *crate::pod::bytes_of(&[1_u32, 2, 3, 4]));
        assert!(matches!(ring.take(first, |_| ()), Err(GpuError::Stale { .. })), "taken once");
        // The device is shared by the tests running in parallel: another test's poll may be running
        // the second map callback on its own thread right now, so wait for it instead of assuming.
        for _ in 0..1000 {
            if ring.is_ready(second) {
                break;
            }
            crate::device::wait_for(&gpu.device, None, crate::device::GPU_WAIT_LIMIT, "the queued work").unwrap();
            std::thread::yield_now();
        }
        let value = ring.take(second, |readback| (readback.user(), readback.packed_len())).unwrap();
        assert_eq!(value, Some((2, 8)));
        assert_eq!(ring.free_slots(), 2);

        // Bad requests.
        let mut encoder = new_encoder(gpu);
        assert!(matches!(ring.copy_buffer(&mut encoder, &buffer, 2, 4, 0), Err(GpuError::Invalid { .. })));
        assert!(matches!(ring.copy_buffer(&mut encoder, &buffer, 8, 12, 0), Err(GpuError::OutOfRange { .. })));
        let big = source(gpu, &[0; 128]);
        assert!(matches!(ring.copy_buffer(&mut encoder, &big, 0, 512, 0), Err(GpuError::OutOfRange { .. })));
        assert!(matches!(ring.copy_buffer(&mut encoder, &big, 0, 0, 0), Err(GpuError::OutOfRange { .. })));
        assert!(matches!(ReadbackRing::new(&mut resources, 0, 16, "none"), Err(GpuError::Invalid { .. })));
        assert!(matches!(ReadbackRing::new(&mut resources, 65, 16, "many"), Err(GpuError::Invalid { .. })));
        let mut small = [0_u8; 3];
        let ticket = ring.copy_buffer(&mut encoder, &buffer, 0, 4, 0).unwrap();
        gpu.queue.submit([encoder.finish()]);
        ring.after_submit();
        assert!(matches!(ring.read_blocking(&gpu.device, ticket, &mut small), Err(GpuError::OutOfRange { .. })));
        assert_eq!(ring.free_slots(), 2, "a failed copy-out still frees the slot");
        ring.destroy(&mut resources);
        assert_eq!(resources.live(), (0, 0));
        assert_eq!(gpu.events().pending(), 0);
    }

    #[test]
    fn perf_r3_texture_rows_come_back_without_padding() {
        let gpu = testing::gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        let usage = wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
        let desc =
            TextureDesc::array(5, 3, 2, wgpu::TextureFormat::Rgba8Unorm, usage, Tag::shared(Category::Other, "t"));
        let (id, _) = resources.create_texture(&desc).unwrap();
        let level =
            |layer: u8| -> Vec<u8> { (0..60).map(|i: u8| i.wrapping_mul(5).wrapping_add(layer * 100)).collect() };
        for layer in 0..2 {
            resources.write_texture_level(id, 0, u32::from(layer), &level(layer)).unwrap();
        }
        let mut ring = ReadbackRing::new(&mut resources, 1, 4096, "screenshot").unwrap();
        let texture = resources.texture(id).unwrap().raw().clone();
        let mut encoder = new_encoder(gpu);
        // Columns 1..4 of rows 1..3 of both layers.
        let ticket = ring.copy_texture(&mut encoder, &texture, 0, [1, 1, 0], [3, 2, 2], 9).unwrap();
        gpu.queue.submit([encoder.finish()]);
        ring.after_submit();
        let mut out = [0_u8; 3 * 4 * 2 * 2];
        assert_eq!(ring.read_blocking(&gpu.device, ticket, &mut out).unwrap(), 48);
        let mut expected = Vec::new();
        for layer in 0..2 {
            let texels = level(layer);
            for row in 1..3 {
                expected.extend_from_slice(&texels[row * 20 + 4..row * 20 + 16]);
            }
        }
        assert_eq!(&out[..], &expected[..]);

        let mut encoder = new_encoder(gpu);
        assert!(matches!(
            ring.copy_texture(&mut encoder, &texture, 0, [3, 0, 0], [3, 1, 1], 0),
            Err(GpuError::OutOfRange { .. })
        ));
        assert!(matches!(
            ring.copy_texture(&mut encoder, &texture, 1, [0, 0, 0], [1, 1, 1], 0),
            Err(GpuError::Invalid { .. })
        ));
        assert!(matches!(
            ring.copy_texture(&mut encoder, &texture, 0, [0, 0, 1], [1, 1, 2], 0),
            Err(GpuError::OutOfRange { .. })
        ));
        assert_eq!(gpu.events().pending(), 0);
    }
}
