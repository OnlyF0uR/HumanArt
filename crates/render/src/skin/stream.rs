//! Streaming integration of the skin texture set `[SKN-03]` `[AC-WP13-3]`.
//!
//! Each UV tile of a resolved channel is one virtual texture of `humanart_gpu::streaming`'s pool:
//! a mip pyramid of square pages, the coarsest pinned. Every frame the renderer tells
//! [`SkinStreaming`] what it sees of each UV tile (the visible texel rectangle and how many screen
//! pixels the whole UV tile would span); the wanted mip is the level whose texel density matches
//! the screen (`floor(log2(texels / pixels))`), so a face close-up asks for mip 0 of the face
//! tile's visible part and a distant body for its coarse levels. The GPU feedback pass's words go
//! through [`SkinStreaming::note_feedback`] as well. [`SkinStreaming::update`] loads queued pages
//! from a [`TileSource`] (the package's stored tiles, or a readback of the editor's resolved set),
//! evicting the least recently used, or, in the editor, straight from the compositor's resolved
//! textures on the GPU ([`SkinStreaming::update_from_gpu`], see [`super::page_copy`]). Requests and
//! updates allocate nothing `[PERF-R1]`.

use humanart_gpu::resources::Resources;
use humanart_gpu::streaming::{MAX_TILES_PER_AXIS, StreamingDesc, StreamingPool, TileRequest, UpdateReport};
use humanart_gpu::{Gpu, GpuError};
use humanart_model::skin_appearance::dirty::Rect;

use super::SkinGpuError;
use super::page_copy::{MAX_COPIES_PER_UPDATE, ResolvedPages};

/// What the renderer sees of one UV tile this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileView {
    /// The UV tile.
    pub uv_tile: u32,
    /// The visible part, in level-0 texels of the UV tile.
    pub visible: Rect,
    /// Screen pixels the whole UV tile would span along one side at the current distance.
    pub pixels_across: f32,
}

/// Where page texels come from.
pub trait TileSource {
    /// Writes the page `(x, y)` of level `mip` of UV tile `uv_tile`: `page_size²` texels of the
    /// pool's format, tightly packed rows.
    fn fill(&self, uv_tile: u32, mip: u8, x: u16, y: u16, out: &mut [u8]) -> Result<(), GpuError>;
}

/// The streaming state of one skin channel.
pub struct SkinStreaming {
    pool: StreamingPool,
    /// Virtual texture id per UV tile.
    ids: Vec<u8>,
    uv_tile_size: u32,
    page_size: u32,
    mips: u8,
}

impl SkinStreaming {
    /// A pool of `slots_per_side² × layers` pages of `page_size` texels in `format`, with one
    /// virtual texture per UV tile of `uv_tile_size²` texels.
    pub fn new(
        resources: &mut Resources,
        uv_tiles: u32,
        uv_tile_size: u32,
        page_size: u32,
        slots_per_side: u32,
        layers: u32,
        format: wgpu::TextureFormat,
    ) -> Result<Self, SkinGpuError> {
        if uv_tiles == 0 || uv_tiles > 255 || page_size == 0 || !uv_tile_size.is_multiple_of(page_size) {
            return Err(SkinGpuError::Invalid { what: "UV tiles must be 1..255 and a whole number of pages" });
        }
        if uv_tile_size / page_size > MAX_TILES_PER_AXIS {
            return Err(SkinGpuError::Invalid { what: "too many pages per UV tile" });
        }
        let per_tile: u32 = {
            let mut n = 0;
            let mut side = uv_tile_size / page_size;
            loop {
                n += side * side;
                if side <= 1 {
                    break n;
                }
                side = side.div_ceil(2);
            }
        };
        let desc = StreamingDesc {
            tile_size: page_size,
            format,
            tiles_per_side: slots_per_side,
            layers,
            max_pages: per_tile * uv_tiles,
            queue_capacity: 1024,
            purpose: "skin streaming pool",
        };
        let mut pool = StreamingPool::new(resources, desc)?;
        let mut ids = Vec::with_capacity(uv_tiles as usize);
        for _ in 0..uv_tiles {
            ids.push(pool.add_texture(uv_tile_size, uv_tile_size)?);
        }
        let mips = ((uv_tile_size / page_size).max(1).ilog2() + 1) as u8;
        Ok(Self { pool, ids, uv_tile_size, page_size, mips })
    }

    /// The pool (page table and texture for the shaders).
    pub const fn pool(&self) -> &StreamingPool {
        &self.pool
    }

    /// The virtual texture id of a UV tile.
    pub fn texture_of(&self, uv_tile: u32) -> Option<u8> {
        self.ids.get(uv_tile as usize).copied()
    }

    /// Mip levels per UV tile.
    pub const fn mips(&self) -> u8 {
        self.mips
    }

    /// The wanted mip for a view: texel density matched to the screen.
    pub fn wanted_mip(&self, view: &TileView) -> u8 {
        if !(view.pixels_across.is_finite() && view.pixels_across > 0.0) {
            return self.mips - 1;
        }
        let ratio = self.uv_tile_size as f32 / view.pixels_across;
        if ratio <= 1.0 {
            return 0;
        }
        (ratio.log2().floor() as u8).min(self.mips - 1)
    }

    /// The page rectangle (inclusive-exclusive page coordinates) of a view at `mip`.
    fn pages(&self, view: &TileView, mip: u8) -> (u32, u32, u32, u32) {
        let scale = self.page_size << mip;
        let visible = view.visible.clip(self.uv_tile_size);
        let x0 = visible.x / scale;
        let y0 = visible.y / scale;
        let x1 = visible.right().div_ceil(scale);
        let y1 = visible.bottom().div_ceil(scale);
        (x0, y0, x1, y1)
    }

    /// Starts a frame.
    pub fn begin_frame(&mut self) {
        self.pool.begin_frame();
    }

    /// Requests every page of a view at its wanted mip; returns the number of pages requested.
    pub fn request_view(&mut self, view: &TileView) -> Result<u32, SkinGpuError> {
        let Some(texture) = self.texture_of(view.uv_tile) else {
            return Err(SkinGpuError::Invalid { what: "no such UV tile" });
        };
        let mip = self.wanted_mip(view);
        let (x0, y0, x1, y1) = self.pages(view, mip);
        let mut count = 0;
        for y in y0..y1 {
            for x in x0..x1 {
                self.pool.request(TileRequest { texture, mip, x: x as u16, y: y as u16 })?;
                count += 1;
            }
        }
        Ok(count)
    }

    /// Whether every page of a view is resident at its wanted mip.
    pub fn is_view_resident(&self, view: &TileView) -> bool {
        let Some(texture) = self.texture_of(view.uv_tile) else { return false };
        let mip = self.wanted_mip(view);
        let (x0, y0, x1, y1) = self.pages(view, mip);
        (y0..y1)
            .all(|y| (x0..x1).all(|x| self.pool.is_resident(TileRequest { texture, mip, x: x as u16, y: y as u16 })))
    }

    /// Requests the pages named by GPU feedback words; returns the valid ones.
    pub fn note_feedback(&mut self, words: &[u32]) -> usize {
        self.pool.note_feedback(words)
    }

    /// Loads up to `max_loads` queued pages from `source`.
    pub fn update(
        &mut self,
        resources: &mut Resources,
        max_loads: u32,
        source: &impl TileSource,
    ) -> Result<UpdateReport, SkinGpuError> {
        let ids = &self.ids;
        let report = self.pool.update(resources, max_loads, |request, out| {
            let uv_tile = ids
                .iter()
                .position(|&id| id == request.texture)
                .ok_or(GpuError::Invalid { what: "page of an unknown UV tile" })?;
            source.fill(uv_tile as u32, request.mip, request.x, request.y, out)
        })?;
        Ok(report)
    }

    /// Loads up to `max_loads` (at most [`MAX_COPIES_PER_UPDATE`]) queued pages from the
    /// compositor's textures on the GPU: the pool assigns the slots and updates the page table
    /// ([`StreamingPool::update_external`]: no texels are uploaded from the CPU, only the changed
    /// page-table rows), and the page contents are recorded into `encoder` (box-averaged to the
    /// page's mip, then copied into the slot). Submit `encoder` in this frame, before the pool is
    /// sampled: until then a new slot shows what it held before. The pool's format must be that of
    /// `pages` and its UV tiles the size of the source's.
    pub fn update_from_gpu(
        &mut self,
        gpu: &Gpu,
        resources: &mut Resources,
        encoder: &mut wgpu::CommandEncoder,
        max_loads: u32,
        pages: &mut ResolvedPages,
    ) -> Result<UpdateReport, SkinGpuError> {
        if pages.format() != self.pool.desc().format
            || pages.source_size() != self.uv_tile_size
            || pages.page_size() != self.page_size
        {
            return Err(SkinGpuError::Invalid { what: "the resolved pages do not match the streaming pool" });
        }
        let max_loads = max_loads.min(MAX_COPIES_PER_UPDATE as u32);
        pages.pending.clear();
        let pending = &mut pages.pending;
        // The pool hands out the slots and uploads nothing for them: the copies recorded below are
        // the only write of their texels. A page that cannot be recorded is refused, so it stays
        // missing instead of resident with another page's texels.
        let updated = self.pool.update_external(resources, max_loads, |request, _| {
            if pending.len() >= MAX_COPIES_PER_UPDATE {
                return Err(GpuError::Invalid { what: "more pages than one update can copy" });
            }
            pending.push(request);
            Ok(())
        });
        // Pages that got a slot before a failure are resident: their copies are recorded too.
        let pool_texture = resources
            .texture(self.pool.pool_texture())
            .ok_or(GpuError::Invalid { what: "the streaming pool texture is gone" })?
            .raw();
        let (ids, pool) = (&self.ids, &self.pool);
        let targets = pages.pending.iter().filter_map(|&request| {
            let uv_tile = ids.iter().position(|&id| id == request.texture)?;
            let entry = pool.lookup(request).filter(|e| e.mip == request.mip)?;
            let at = pool.slot_location(entry.slot);
            Some((request, uv_tile as u32, [at.x, at.y, at.layer]))
        });
        let mut targets_buffer =
            [(TileRequest { texture: 0, mip: 0, x: 0, y: 0 }, 0_u32, [0_u32; 3]); MAX_COPIES_PER_UPDATE];
        let mut n = 0;
        for t in targets {
            targets_buffer[n] = t;
            n += 1;
        }
        pages.record(gpu, encoder, pool_texture, targets_buffer[..n].iter().copied())?;
        Ok(updated?)
    }
}
