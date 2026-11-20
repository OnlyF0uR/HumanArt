//! The texture streaming manager: virtual textures paged into a fixed tile pool `[SKN-03]` `[RT-05]`.
//!
//! WebGPU has no sparse textures, so residency is done the virtual-texturing way:
//!
//! - Each registered **virtual texture** is a mip pyramid of square tiles of `tile_size` texels.
//! - The **pool** is one 2D texture array made once through [`Resources`]: `tiles_per_side` x
//!   `tiles_per_side` slots per layer, `layers` layers. Its size is the streaming memory budget: the
//!   number of slots is fixed and a full pool evicts the least recently used tile.
//! - The **page table** is a storage buffer with one `u32` per tile of every virtual texture
//!   ([`PageEntry`] encoding). Every entry names the finest *resident* tile that covers it (the tile
//!   itself or an ancestor), so a shader does one lookup and scales its UV by the mip difference.
//! - The coarsest level of every virtual texture is **pinned**: queued at registration, never evicted,
//!   so every lookup has a fallback once it is in.
//! - A **feedback pass** (in the renderer) writes the tiles it wanted as packed `u32` words
//!   ([`TileRequest::pack`]); the words come back through the readback ring and go to
//!   [`StreamingPool::note_feedback`]. Requesting a tile also requests its missing ancestors, coarse
//!   first.
//! - [`StreamingPool::update`] loads up to a given number of queued tiles per frame through the
//!   caller's fill function, evicts what it must, and uploads the changed page-table rows.
//!
//! All tables are structure-of-arrays sized at creation; requests, updates and feedback allocate
//! nothing `[PERF-R1]`. Tile borders for filtering are the fill function's business: the manager
//! stores whole tiles.

use crate::error::GpuError;
use crate::resources::{
    BufferDesc, BufferId, Category, DirtyRanges, Resources, Tag, TextureDesc, TextureId, level_bytes,
};

/// Most mip levels of a virtual texture.
pub const MAX_MIPS: usize = 16;
/// Most virtual textures in one pool (the 8-bit texture field of a [`TileRequest`]).
pub const MAX_TEXTURES: usize = 256;
/// Most tiles per axis of a level (the 10-bit coordinate fields of a [`TileRequest`]).
pub const MAX_TILES_PER_AXIS: u32 = 1024;
/// Most slots in a pool (the 16-bit slot field of a [`PageEntry`]).
pub const MAX_SLOTS: u32 = 1 << 16;

/// "No slot" / "no page".
const NONE: u32 = u32::MAX;

/// One tile of one virtual texture: what the feedback pass asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileRequest {
    /// The virtual texture, as returned by [`StreamingPool::add_texture`].
    pub texture: u8,
    /// The mip level, 0 the finest.
    pub mip: u8,
    /// Tile column.
    pub x: u16,
    /// Tile row.
    pub y: u16,
}

impl TileRequest {
    /// The request as the feedback pass writes it: bits 0..10 `x`, 10..20 `y`, 20..24 `mip`, 24..32
    /// `texture`. Fields too wide for their bits are masked.
    pub const fn pack(self) -> u32 {
        (self.x as u32 & 0x3ff)
            | ((self.y as u32 & 0x3ff) << 10)
            | ((self.mip as u32 & 0xf) << 20)
            | ((self.texture as u32) << 24)
    }

    /// The inverse of [`TileRequest::pack`].
    pub const fn unpack(word: u32) -> Self {
        Self {
            texture: (word >> 24) as u8,
            mip: ((word >> 20) & 0xf) as u8,
            x: (word & 0x3ff) as u16,
            y: ((word >> 10) & 0x3ff) as u16,
        }
    }
}

/// A decoded page-table entry. Encoding in the buffer: bit 31 valid, bits 0..16 slot, bits 16..21
/// the mip level of the resident tile the entry points to; 0 means nothing resident yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageEntry {
    /// The pool slot holding the tile.
    pub slot: u32,
    /// The mip level of that tile: the requested level or a coarser one.
    pub mip: u8,
}

impl PageEntry {
    /// The bit that marks an entry as valid.
    pub const VALID: u32 = 1 << 31;

    /// The entry as stored in the page table.
    pub const fn encode(self) -> u32 {
        Self::VALID | (self.slot & 0xffff) | ((self.mip as u32 & 0x1f) << 16)
    }

    /// The inverse of [`PageEntry::encode`]; `None` for an invalid entry.
    pub const fn decode(word: u32) -> Option<Self> {
        if word & Self::VALID == 0 {
            None
        } else {
            Some(Self { slot: word & 0xffff, mip: ((word >> 16) & 0x1f) as u8 })
        }
    }
}

/// How a pool is made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingDesc {
    /// Texels per tile side.
    pub tile_size: u32,
    /// The pool's texel format.
    pub format: wgpu::TextureFormat,
    /// Slots per pool row and column.
    pub tiles_per_side: u32,
    /// Pool layers.
    pub layers: u32,
    /// Page-table entries for all virtual textures together.
    pub max_pages: u32,
    /// Tiles waiting to load at most; further requests are dropped until the next feedback.
    pub queue_capacity: u32,
    /// Accounting label.
    pub purpose: &'static str,
}

/// Where a slot is in the pool texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotLocation {
    /// Texel x of the tile's corner.
    pub x: u32,
    /// Texel y of the tile's corner.
    pub y: u32,
    /// The array layer.
    pub layer: u32,
}

/// One level of a virtual texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Level {
    /// Tiles per row.
    tiles_x: u32,
    /// Tiles per column.
    tiles_y: u32,
    /// First page of the level.
    offset: u32,
}

/// A registered virtual texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VirtualTexture {
    /// The levels, finest first.
    levels: [Level; MAX_MIPS],
    /// Levels in use.
    mips: u8,
}

/// What the pool did so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StreamingStats {
    /// Tiles loaded.
    pub loads: u64,
    /// Tiles evicted to make room.
    pub evictions: u64,
    /// Requests for tiles that were not resident.
    pub misses: u64,
    /// Requests dropped because the queue was full (they come back with the next feedback).
    pub dropped: u64,
    /// Feedback words naming no tile of a registered texture.
    pub invalid_feedback: u64,
    /// Loads that waited because every slot was in use this frame or pinned.
    pub starved: u64,
}

/// What one [`StreamingPool::update`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UpdateReport {
    /// Tiles loaded.
    pub loaded: u32,
    /// Tiles evicted.
    pub evicted: u32,
    /// Page-table bytes uploaded.
    pub table_bytes: u64,
    /// Tiles still queued.
    pub queued: u32,
}

/// The streaming manager. See the module documentation.
#[derive(Debug)]
pub struct StreamingPool {
    /// The request it was made from.
    desc: StreamingDesc,
    /// The pool texture array in [`Resources`].
    pool: TextureId,
    /// The page table buffer in [`Resources`].
    table_buffer: BufferId,
    /// Registered virtual textures; capacity [`MAX_TEXTURES`].
    textures: Vec<VirtualTexture>,
    /// Pages handed out so far.
    pages_used: u32,
    /// Per page: the packed address ([`TileRequest::pack`]).
    page_address: Vec<u32>,
    /// Per page: the slot holding it, or `NONE`.
    page_slot: Vec<u32>,
    /// Per page: the frame it was last requested in.
    page_last_used: Vec<u64>,
    /// Per page: waiting in the queue.
    page_queued: Vec<bool>,
    /// Per page: never evicted.
    page_pinned: Vec<bool>,
    /// The page table as uploaded: [`PageEntry`] words.
    table: Vec<u32>,
    /// The bytes of `table`, kept in step, for the dirty upload.
    table_bytes: Vec<u8>,
    /// Changed parts of the table.
    dirty: DirtyRanges,
    /// Per slot: the page it holds, or `NONE`.
    slot_page: Vec<u32>,
    /// Free slots (a stack).
    free_slots: Vec<u32>,
    /// The load queue: a ring of pages.
    queue: Vec<u32>,
    /// Index of the oldest queued page.
    queue_head: usize,
    /// Pages queued.
    queue_len: usize,
    /// One tile's bytes, filled by the caller and uploaded; allocated once.
    tile_bytes: Vec<u8>,
    /// The frame counter.
    frame: u64,
    /// The counters.
    stats: StreamingStats,
}

impl StreamingPool {
    /// Makes the pool texture and the page table in `resources`. Limits: tile size 1 to 4096,
    /// `tiles_per_side * tile_size` within the device's 2D size, layers within its array limit, at
    /// most [`MAX_SLOTS`] slots, a format the device can upload without depth or planes.
    pub fn new(resources: &mut Resources, desc: StreamingDesc) -> Result<Self, GpuError> {
        let limits = resources.limits();
        if desc.tile_size == 0 || desc.tile_size > 4096 || desc.tiles_per_side == 0 || desc.layers == 0 {
            return Err(GpuError::Invalid { what: "a streaming pool with a zero or oversized tile or pool" });
        }
        let side = desc
            .tile_size
            .checked_mul(desc.tiles_per_side)
            .filter(|&side| side <= limits.max_texture_dimension_2d)
            .ok_or(GpuError::Invalid { what: "a streaming pool wider than the device's textures" })?;
        if desc.layers > limits.max_texture_array_layers {
            return Err(GpuError::Invalid { what: "a streaming pool with more layers than the device allows" });
        }
        let slots = u64::from(desc.tiles_per_side) * u64::from(desc.tiles_per_side) * u64::from(desc.layers);
        if slots > u64::from(MAX_SLOTS) {
            return Err(GpuError::Invalid { what: "a streaming pool with more than 65536 slots" });
        }
        if desc.max_pages == 0 || desc.queue_capacity == 0 {
            return Err(GpuError::Invalid { what: "a streaming pool without page-table entries or queue" });
        }
        let (block_width, block_height) = desc.format.block_dimensions();
        if desc.format.block_copy_size(None).is_none() || !desc.tile_size.is_multiple_of(block_width.max(block_height))
        {
            return Err(GpuError::Invalid { what: "a streaming format that is not copyable in whole tiles" });
        }
        let tile_len = usize::try_from(level_bytes(desc.format, desc.tile_size, desc.tile_size))
            .map_err(|_| GpuError::Invalid { what: "a tile too large for memory" })?;
        let pool_desc = TextureDesc::array(
            side,
            side,
            desc.layers,
            desc.format,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            Tag::shared(Category::Textures, desc.purpose),
        );
        let (pool, _) = resources.create_texture(&pool_desc)?;
        let table_buffer = match resources.create_buffer(&BufferDesc {
            size: u64::from(desc.max_pages) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            tag: Tag::shared(Category::Textures, "streaming page table"),
        }) {
            Ok((id, _)) => id,
            Err(error) => {
                let _ = resources.destroy_texture(pool);
                return Err(error);
            }
        };
        let pages = desc.max_pages as usize;
        let slots = slots as u32;
        Ok(Self {
            desc,
            pool,
            table_buffer,
            textures: Vec::with_capacity(MAX_TEXTURES),
            pages_used: 0,
            page_address: vec![0; pages],
            page_slot: vec![NONE; pages],
            page_last_used: vec![0; pages],
            page_queued: vec![false; pages],
            page_pinned: vec![false; pages],
            table: vec![0; pages],
            table_bytes: vec![0; pages * 4],
            dirty: DirtyRanges::new(),
            slot_page: vec![NONE; slots as usize],
            free_slots: (0..slots).rev().collect(),
            queue: vec![NONE; desc.queue_capacity as usize],
            queue_head: 0,
            queue_len: 0,
            tile_bytes: vec![0; tile_len],
            frame: 0,
            stats: StreamingStats::default(),
        })
    }

    /// The request, as made.
    pub fn desc(&self) -> &StreamingDesc {
        &self.desc
    }

    /// The pool texture array, to bind as `texture_2d_array`.
    pub fn pool_texture(&self) -> TextureId {
        self.pool
    }

    /// The page table, to bind as `array<u32>`.
    pub fn page_table(&self) -> BufferId {
        self.table_buffer
    }

    /// The counters.
    pub fn stats(&self) -> &StreamingStats {
        &self.stats
    }

    /// Number of pool slots: the budget in tiles.
    pub fn slot_count(&self) -> u32 {
        self.slot_page.len() as u32
    }

    /// Tiles resident now.
    pub fn resident(&self) -> u32 {
        self.slot_count() - self.free_slots.len() as u32
    }

    /// Tiles waiting to load.
    pub fn queued(&self) -> u32 {
        self.queue_len as u32
    }

    /// Where a slot is in the pool texture (the shader does the same arithmetic).
    pub fn slot_location(&self, slot: u32) -> SlotLocation {
        let per_layer = self.desc.tiles_per_side * self.desc.tiles_per_side;
        let in_layer = slot % per_layer;
        SlotLocation {
            x: (in_layer % self.desc.tiles_per_side) * self.desc.tile_size,
            y: (in_layer / self.desc.tiles_per_side) * self.desc.tile_size,
            layer: slot / per_layer,
        }
    }

    /// Registers a virtual texture of `width` x `height` texels and returns its id. The levels go
    /// down to the one that fits in a single tile, which is pinned and queued for loading. Its
    /// first page-table entry is [`StreamingPool::first_page`].
    pub fn add_texture(&mut self, width: u32, height: u32) -> Result<u8, GpuError> {
        if self.textures.len() >= MAX_TEXTURES {
            return Err(GpuError::Full { what: "the streaming texture table" });
        }
        if width == 0 || height == 0 {
            return Err(GpuError::Invalid { what: "a virtual texture with a zero dimension" });
        }
        let tile = self.desc.tile_size;
        let mut texture = VirtualTexture { levels: [Level::default(); MAX_MIPS], mips: 0 };
        let mut offset = u64::from(self.pages_used);
        for mip in 0..MAX_MIPS as u32 {
            let tiles_x = crate::resources::mip_dimension(width, mip).div_ceil(tile);
            let tiles_y = crate::resources::mip_dimension(height, mip).div_ceil(tile);
            if tiles_x > MAX_TILES_PER_AXIS || tiles_y > MAX_TILES_PER_AXIS {
                return Err(GpuError::Invalid { what: "a virtual texture with more than 1024 tiles a side" });
            }
            texture.levels[mip as usize] = Level { tiles_x, tiles_y, offset: offset as u32 };
            offset += u64::from(tiles_x) * u64::from(tiles_y);
            texture.mips += 1;
            if tiles_x == 1 && tiles_y == 1 {
                break;
            }
        }
        let last = texture.levels[usize::from(texture.mips) - 1];
        if last.tiles_x != 1 || last.tiles_y != 1 {
            return Err(GpuError::Invalid { what: "a virtual texture with more than 16 mip levels of tiles" });
        }
        if offset > u64::from(self.desc.max_pages) {
            return Err(GpuError::Full { what: "the streaming page table" });
        }
        if self.queue_len >= self.queue.len() {
            // The pinned top tile must be queued; refuse before anything changes.
            return Err(GpuError::Full { what: "the streaming load queue" });
        }
        let id = self.textures.len() as u8;
        for (mip, level) in texture.levels[..usize::from(texture.mips)].iter().enumerate() {
            for y in 0..level.tiles_y {
                for x in 0..level.tiles_x {
                    let page = (level.offset + y * level.tiles_x + x) as usize;
                    let request = TileRequest { texture: id, mip: mip as u8, x: x as u16, y: y as u16 };
                    self.page_address[page] = request.pack();
                    self.page_slot[page] = NONE;
                    self.page_queued[page] = false;
                    self.page_pinned[page] = false;
                    self.page_last_used[page] = 0;
                    self.set_entry(page, 0);
                }
            }
        }
        self.pages_used = offset as u32;
        self.textures.push(texture);
        let top = last.offset as usize;
        self.page_pinned[top] = true;
        // Room was checked above.
        let _ = self.enqueue(top as u32);
        Ok(id)
    }

    /// The page-table index of a virtual texture's first entry (its level 0, row-major, then the
    /// coarser levels): what the shader adds to its tile index.
    pub fn first_page(&self, texture: u8) -> Option<u32> {
        self.textures.get(usize::from(texture)).map(|t| t.levels[0].offset)
    }

    /// The page of a request, or `None` when it names no tile of a registered texture.
    fn page_of(&self, request: TileRequest) -> Option<u32> {
        let texture = self.textures.get(usize::from(request.texture))?;
        if request.mip >= texture.mips {
            return None;
        }
        let level = texture.levels[usize::from(request.mip)];
        let (x, y) = (u32::from(request.x), u32::from(request.y));
        (x < level.tiles_x && y < level.tiles_y).then_some(level.offset + y * level.tiles_x + x)
    }

    /// The parent of a tile (clamped at the level's edge, as sampling is), or `None` at the top.
    fn parent(&self, request: TileRequest) -> Option<TileRequest> {
        let texture = self.textures.get(usize::from(request.texture))?;
        let mip = usize::from(request.mip) + 1;
        if mip >= usize::from(texture.mips) {
            return None;
        }
        let level = texture.levels[mip];
        Some(TileRequest {
            texture: request.texture,
            mip: mip as u8,
            x: (u32::from(request.x) / 2).min(level.tiles_x - 1) as u16,
            y: (u32::from(request.y) / 2).min(level.tiles_y - 1) as u16,
        })
    }

    /// What the page table says for a tile now.
    pub fn lookup(&self, request: TileRequest) -> Option<PageEntry> {
        let page = self.page_of(request)?;
        PageEntry::decode(self.table[page as usize])
    }

    /// True when the tile itself is resident.
    pub fn is_resident(&self, request: TileRequest) -> bool {
        self.page_of(request).is_some_and(|page| self.page_slot[page as usize] != NONE)
    }

    /// Starts a frame: requests from now on count as this frame's use.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// Asks for a tile: marks it and its ancestors used this frame and queues the missing ones,
    /// coarsest first. A request naming no tile is [`GpuError::OutOfRange`].
    pub fn request(&mut self, request: TileRequest) -> Result<(), GpuError> {
        if self.page_of(request).is_none() {
            return Err(GpuError::OutOfRange {
                what: "streaming tile request",
                offset: u64::from(request.mip),
                len: u64::from(request.x.max(request.y)),
                capacity: self.textures.len() as u64,
            });
        }
        // Walk to the top, remembering the chain, then queue from the top down.
        let mut chain = [NONE; MAX_MIPS];
        let mut depth = 0;
        let mut current = Some(request);
        while let Some(tile) = current {
            let Some(page) = self.page_of(tile) else { break };
            if depth < MAX_MIPS {
                chain[depth] = page;
                depth += 1;
            }
            current = self.parent(tile);
        }
        for &page in chain[..depth].iter().rev() {
            let index = page as usize;
            self.page_last_used[index] = self.frame;
            if self.page_slot[index] == NONE && !self.page_queued[index] {
                if page == chain[0] {
                    self.stats.misses += 1;
                }
                if !self.enqueue(page) {
                    self.stats.dropped += 1;
                    // Finer tiles would only wait behind this one.
                    break;
                }
            }
        }
        Ok(())
    }

    /// Requests every tile named by feedback words (the feedback pass's output). Words naming no
    /// registered tile are counted in [`StreamingStats::invalid_feedback`] and skipped: the words
    /// come from the GPU and are not trusted. Returns the valid requests.
    pub fn note_feedback(&mut self, words: &[u32]) -> usize {
        let mut valid = 0;
        for &word in words {
            if self.request(TileRequest::unpack(word)).is_ok() {
                valid += 1;
            } else {
                self.stats.invalid_feedback += 1;
            }
        }
        valid
    }

    /// Adds a page to the load queue; false when the queue is full.
    fn enqueue(&mut self, page: u32) -> bool {
        if self.queue_len >= self.queue.len() {
            return false;
        }
        let tail = (self.queue_head + self.queue_len) % self.queue.len();
        self.queue[tail] = page;
        self.queue_len += 1;
        self.page_queued[page as usize] = true;
        true
    }

    /// Takes the oldest queued page.
    fn dequeue(&mut self) -> Option<u32> {
        if self.queue_len == 0 {
            return None;
        }
        let page = self.queue[self.queue_head];
        self.queue_head = (self.queue_head + 1) % self.queue.len();
        self.queue_len -= 1;
        if let Some(queued) = self.page_queued.get_mut(page as usize) {
            *queued = false;
        }
        Some(page)
    }

    /// A slot for a new tile: a free one, else the least recently used tile not used this frame and
    /// not pinned is evicted. `None` when there is none.
    fn take_slot(&mut self, report: &mut UpdateReport) -> Option<u32> {
        if let Some(slot) = self.free_slots.pop() {
            return Some(slot);
        }
        let mut best: Option<(u32, u64)> = None;
        for (slot, &page) in self.slot_page.iter().enumerate() {
            let Some(&last_used) = self.page_last_used.get(page as usize) else { continue };
            if self.page_pinned[page as usize] || last_used >= self.frame {
                continue;
            }
            if best.is_none_or(|(_, oldest)| last_used < oldest) {
                best = Some((slot as u32, last_used));
            }
        }
        let (slot, _) = best?;
        let page = self.slot_page[slot as usize];
        self.page_slot[page as usize] = NONE;
        self.slot_page[slot as usize] = NONE;
        self.refresh_subtree(page);
        self.stats.evictions += 1;
        report.evicted += 1;
        Some(slot)
    }

    /// Loads up to `max_loads` queued tiles. `fill` writes one tile's texels (tightly packed rows of
    /// `tile_size` texels) into the buffer it is given; an error from it stops the update and the
    /// tile stays missing. Then the changed page-table rows are uploaded.
    pub fn update(
        &mut self,
        resources: &mut Resources,
        max_loads: u32,
        mut fill: impl FnMut(TileRequest, &mut [u8]) -> Result<(), GpuError>,
    ) -> Result<UpdateReport, GpuError> {
        let mut report = UpdateReport::default();
        while report.loaded < max_loads {
            let Some(page) = self.dequeue() else { break };
            let index = page as usize;
            if self.page_slot[index] != NONE {
                continue;
            }
            let Some(slot) = self.take_slot(&mut report) else {
                // Every slot is in use this frame: try again next frame.
                self.stats.starved += 1;
                let _ = self.enqueue(page);
                break;
            };
            let request = TileRequest::unpack(self.page_address[index]);
            if let Err(error) = fill(request, &mut self.tile_bytes) {
                self.free_slots.push(slot);
                return Err(error);
            }
            let location = self.slot_location(slot);
            let tile = self.desc.tile_size;
            if let Err(error) = resources.write_texture_region(
                self.pool,
                0,
                [location.x, location.y, location.layer],
                tile,
                tile,
                &self.tile_bytes,
            ) {
                self.free_slots.push(slot);
                return Err(error);
            }
            self.page_slot[index] = slot;
            self.slot_page[slot as usize] = page;
            self.refresh_subtree(page);
            self.stats.loads += 1;
            report.loaded += 1;
        }
        report.table_bytes = resources.flush_dirty(self.table_buffer, &self.table_bytes, &mut self.dirty)?;
        report.queued = self.queue_len as u32;
        Ok(report)
    }

    /// Writes one page-table entry and marks it dirty.
    fn set_entry(&mut self, page: usize, word: u32) {
        if self.table[page] != word {
            self.table[page] = word;
            self.table_bytes[page * 4..page * 4 + 4].copy_from_slice(&word.to_le_bytes());
            self.dirty.mark(page as u64 * 4, 4);
        }
    }

    /// Recomputes the entries of a page and every finer tile under it, top-down: each points to
    /// itself when resident, else to what its parent points to.
    fn refresh_subtree(&mut self, page: u32) {
        let request = TileRequest::unpack(self.page_address[page as usize]);
        let Some(&texture) = self.textures.get(usize::from(request.texture)) else { return };
        let (mut x_lo, mut x_hi) = (u32::from(request.x), u32::from(request.x));
        let (mut y_lo, mut y_hi) = (u32::from(request.y), u32::from(request.y));
        for mip in (0..=usize::from(request.mip)).rev() {
            let level = texture.levels[mip];
            if mip < usize::from(request.mip) {
                // The children of [lo, hi] one level up; the last tile also adopts the clamped ones.
                let above = texture.levels[mip + 1];
                let widen = |lo: u32, hi: u32, last_above: u32, count: u32| {
                    let new_hi = if hi + 1 >= last_above { count - 1 } else { (2 * hi + 1).min(count - 1) };
                    ((2 * lo).min(count - 1), new_hi)
                };
                (x_lo, x_hi) = widen(x_lo, x_hi, above.tiles_x, level.tiles_x);
                (y_lo, y_hi) = widen(y_lo, y_hi, above.tiles_y, level.tiles_y);
            }
            for y in y_lo..=y_hi {
                for x in x_lo..=x_hi {
                    let index = (level.offset + y * level.tiles_x + x) as usize;
                    let word = if self.page_slot[index] == NONE {
                        let tile = TileRequest { texture: request.texture, mip: mip as u8, x: x as u16, y: y as u16 };
                        self.parent(tile).and_then(|parent| self.page_of(parent)).map_or(0, |p| self.table[p as usize])
                    } else {
                        PageEntry { slot: self.page_slot[index], mip: mip as u8 }.encode()
                    };
                    self.set_entry(index, word);
                }
            }
        }
    }

    /// Forgets every virtual texture and every resident tile (unloading a character's set). The
    /// pool and the page table stay allocated.
    pub fn clear(&mut self) {
        self.textures.clear();
        for page in 0..self.pages_used as usize {
            self.set_entry(page, 0);
        }
        self.pages_used = 0;
        self.page_slot.fill(NONE);
        self.page_queued.fill(false);
        self.page_pinned.fill(false);
        self.slot_page.fill(NONE);
        self.free_slots.clear();
        self.free_slots.extend((0..self.slot_count()).rev());
        self.queue_head = 0;
        self.queue_len = 0;
    }

    /// After a device loss [`Resources::rebuild`] made the pool and table again, empty: every tile
    /// becomes missing, the pinned ones are queued again, and the whole table is re-uploaded at the
    /// next update `[ERR-05]`.
    pub fn rebuild(&mut self) {
        self.page_slot.fill(NONE);
        self.page_queued.fill(false);
        self.slot_page.fill(NONE);
        self.free_slots.clear();
        self.free_slots.extend((0..self.slot_count()).rev());
        self.queue_head = 0;
        self.queue_len = 0;
        for page in 0..self.pages_used as usize {
            self.set_entry(page, 0);
            if self.page_pinned[page] {
                let _ = self.enqueue(page as u32);
            }
        }
        self.dirty.mark_all(u64::from(self.pages_used) * 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readback::ReadbackRing;
    use crate::resources::Budget;
    use crate::testing;

    fn desc(tiles_per_side: u32, queue_capacity: u32) -> StreamingDesc {
        StreamingDesc {
            tile_size: 4,
            format: wgpu::TextureFormat::Rgba8Unorm,
            tiles_per_side,
            layers: 1,
            max_pages: 256,
            queue_capacity,
            purpose: "test pool",
        }
    }

    /// Every texel of a tile is (texture, mip, x, y). Returns a `Result` because `update` wants one.
    #[allow(clippy::unnecessary_wraps)]
    fn fill(request: TileRequest, bytes: &mut [u8]) -> Result<(), GpuError> {
        for texel in bytes.as_chunks_mut::<4>().0 {
            texel.copy_from_slice(&[request.texture, request.mip, request.x as u8, request.y as u8]);
        }
        Ok(())
    }

    /// The table, the slot maps and the free list agree, and every entry names the finest resident
    /// tile covering it.
    fn check_invariants(pool: &StreamingPool) {
        for page in 0..pool.pages_used as usize {
            let mut current = Some(TileRequest::unpack(pool.page_address[page]));
            let mut expected = None;
            while let Some(tile) = current {
                let slot = pool.page_slot[pool.page_of(tile).unwrap() as usize];
                if slot != NONE {
                    expected = Some(PageEntry { slot, mip: tile.mip });
                    break;
                }
                current = pool.parent(tile);
            }
            assert_eq!(PageEntry::decode(pool.table[page]), expected, "page {page}");
            assert_eq!(&pool.table_bytes[page * 4..page * 4 + 4], &pool.table[page].to_le_bytes());
        }
        let mut resident = 0;
        for (slot, &page) in pool.slot_page.iter().enumerate() {
            if page != NONE {
                assert_eq!(pool.page_slot[page as usize], slot as u32);
                resident += 1;
            }
        }
        assert_eq!(resident + pool.free_slots.len(), pool.slot_page.len());
    }

    #[test]
    fn skn_03_tiles_load_coarse_first_and_the_page_table_points_to_the_finest_resident() {
        let gpu = testing::gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        let mut pool = StreamingPool::new(&mut resources, desc(4, 32)).unwrap();
        assert_eq!(pool.slot_count(), 16);
        assert_eq!(resources.stats().of(Category::Textures), 16 * 16 * 4 + 256 * 4, "pool and table are accounted");
        let square = pool.add_texture(32, 32).unwrap();
        let odd = pool.add_texture(20, 12).unwrap();
        assert_eq!(pool.pages_used, 85 + 24);
        assert_eq!((pool.first_page(square), pool.first_page(odd), pool.first_page(2)), (Some(0), Some(85), None));
        assert_eq!(pool.queued(), 2, "both pinned tops are queued");

        pool.begin_frame();
        let report = pool.update(&mut resources, 8, fill).unwrap();
        assert_eq!((report.loaded, report.evicted, report.queued), (2, 0, 0));
        let top = pool.lookup(TileRequest { texture: square, mip: 3, x: 0, y: 0 }).unwrap();
        let fine = TileRequest { texture: square, mip: 0, x: 5, y: 5 };
        assert_eq!(pool.lookup(fine), Some(PageEntry { slot: top.slot, mip: 3 }), "falls back to the top");
        check_invariants(&pool);

        pool.begin_frame();
        pool.request(fine).unwrap();
        assert_eq!(pool.queued(), 3, "the tile and its two missing ancestors");
        assert_eq!(pool.stats().misses, 1);
        let report = pool.update(&mut resources, 8, fill).unwrap();
        assert_eq!(report.loaded, 3);
        assert!(report.table_bytes > 0);
        assert_eq!(pool.lookup(fine).unwrap().mip, 0);
        let sibling = pool.lookup(TileRequest { texture: square, mip: 0, x: 4, y: 4 }).unwrap();
        assert_eq!(sibling.mip, 1, "a sibling uses the shared parent");
        assert!(pool.is_resident(TileRequest { texture: square, mip: 1, x: 2, y: 2 }));
        // The odd-sized texture's clamped edge tiles resolve too.
        pool.request(TileRequest { texture: odd, mip: 0, x: 4, y: 2 }).unwrap();
        pool.update(&mut resources, 8, fill).unwrap();
        assert_eq!(pool.lookup(TileRequest { texture: odd, mip: 0, x: 4, y: 2 }).unwrap().mip, 0);
        check_invariants(&pool);

        // The GPU has the same table and the tile's texels in its slot.
        let table = resources.buffer(pool.page_table()).unwrap().raw();
        let bytes = testing::read_buffer(gpu, table, 0, u64::from(pool.pages_used) * 4);
        assert_eq!(bytes, pool.table_bytes[..bytes.len()]);
        let slot = pool.lookup(fine).unwrap().slot;
        let location = pool.slot_location(slot);
        let mut readback = ReadbackRing::new(&mut resources, 1, 4096, "check").unwrap();
        let texture = resources.texture(pool.pool_texture()).unwrap().raw().clone();
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let ticket = readback
            .copy_texture(&mut encoder, &texture, 0, [location.x, location.y, location.layer], [4, 4, 1], 0)
            .unwrap();
        gpu.queue.submit([encoder.finish()]);
        readback.after_submit();
        let mut texels = [0_u8; 64];
        readback.read_blocking(&gpu.device, ticket, &mut texels).unwrap();
        assert!(texels.as_chunks::<4>().0.iter().all(|texel| *texel == [square, 0, 5, 5]), "{texels:?}");
        assert_eq!(gpu.events().pending(), 0);
    }

    #[test]
    fn skn_03_random_requests_keep_the_table_consistent_and_evict_least_recently_used() {
        let gpu = testing::gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        let mut pool = StreamingPool::new(&mut resources, desc(3, 16)).unwrap();
        let textures = [pool.add_texture(32, 32).unwrap(), pool.add_texture(20, 12).unwrap()];
        let mut rng = humanart_core::rng::Rng::new(0x5354_5245);
        let mut tops_loaded = false;
        for _ in 0..300 {
            pool.begin_frame();
            let mut used_resident = [NONE; 8];
            for wanted in &mut used_resident {
                let texture = textures[rng.index(2)];
                let levels = pool.textures[usize::from(texture)];
                let mip = rng.index(usize::from(levels.mips)) as u8;
                let level = levels.levels[usize::from(mip)];
                let request = TileRequest {
                    texture,
                    mip,
                    x: rng.index(level.tiles_x as usize) as u16,
                    y: rng.index(level.tiles_y as usize) as u16,
                };
                pool.request(request).unwrap();
                if pool.is_resident(request) {
                    *wanted = pool.page_of(request).unwrap();
                }
            }
            pool.update(&mut resources, rng.index(4) as u32 + 1, fill).unwrap();
            check_invariants(&pool);
            for page in used_resident.into_iter().filter(|&page| page != NONE) {
                assert_ne!(pool.page_slot[page as usize], NONE, "a tile used this frame was evicted");
            }
            let tops = textures
                .map(|t| pool.textures[usize::from(t)].levels[usize::from(pool.textures[usize::from(t)].mips) - 1]);
            if tops.iter().all(|top| pool.page_slot[top.offset as usize] != NONE) {
                tops_loaded = true;
            } else {
                assert!(!tops_loaded, "a pinned tile was evicted");
            }
        }
        assert!(tops_loaded);
        let stats = *pool.stats();
        assert!(stats.evictions > 0 && stats.loads > stats.evictions, "{stats:?}");
        println!("streaming: 300 frames, 9 slots: {stats:?}");
        // The GPU table matches the CPU one after all those partial uploads.
        let table = resources.buffer(pool.page_table()).unwrap().raw();
        let bytes = testing::read_buffer(gpu, table, 0, u64::from(pool.pages_used) * 4);
        assert_eq!(bytes, pool.table_bytes[..bytes.len()]);
    }

    #[test]
    fn rob_01_feedback_and_requests_are_checked_and_limits_hold() {
        let request = TileRequest { texture: 3, mip: 9, x: 1023, y: 517 };
        assert_eq!(TileRequest::unpack(request.pack()), request);
        let entry = PageEntry { slot: 65535, mip: 15 };
        assert_eq!(PageEntry::decode(entry.encode()), Some(entry));
        assert_eq!(PageEntry::decode(0), None);

        let gpu = testing::gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        // One slot, a queue of three.
        let mut pool = StreamingPool::new(&mut resources, desc(1, 3)).unwrap();
        let texture = pool.add_texture(32, 32).unwrap();
        let valid = TileRequest { texture, mip: 0, x: 7, y: 7 }.pack();
        let words = [
            valid,
            TileRequest { texture: 9, mip: 0, x: 0, y: 0 }.pack(),
            TileRequest { texture, mip: 7, x: 0, y: 0 }.pack(),
            TileRequest { texture, mip: 0, x: 8, y: 0 }.pack(),
            u32::MAX,
        ];
        assert_eq!(pool.note_feedback(&words), 1);
        assert_eq!(pool.stats().invalid_feedback, 4);
        // The queue holds the pinned top and two ancestors; the requested tile itself is dropped
        // (the next feedback asks again).
        assert_eq!(pool.queued(), 3);
        assert_eq!(pool.stats().dropped, 1);
        // Random words never panic.
        let mut rng = humanart_core::rng::Rng::new(7);
        let noise: Vec<u32> = (0..2000).map(|_| rng.next_u32()).collect();
        pool.note_feedback(&noise);
        check_invariants(&pool);

        // One slot: the pinned top takes it and the rest waits, counted as starved.
        pool.begin_frame();
        pool.request(TileRequest::unpack(valid)).unwrap();
        let report = pool.update(&mut resources, 8, fill).unwrap();
        assert_eq!((report.loaded, report.queued), (1, 2));
        assert_eq!(pool.stats().starved, 1);
        check_invariants(&pool);

        // A failing fill is an error and leaves the tables consistent.
        let mut roomy = StreamingPool::new(&mut resources, desc(4, 32)).unwrap();
        roomy.add_texture(32, 32).unwrap();
        let error = roomy.update(&mut resources, 8, |_, _| Err(GpuError::Invalid { what: "no data" }));
        assert!(matches!(error, Err(GpuError::Invalid { what: "no data" })));
        assert_eq!(roomy.resident(), 0);
        check_invariants(&roomy);

        // Bad pools and textures.
        let bad = |change: fn(&mut StreamingDesc)| {
            let mut d = desc(2, 4);
            change(&mut d);
            let mut resources = Resources::new(gpu, Budget::UNLIMITED);
            StreamingPool::new(&mut resources, d).is_err()
        };
        assert!(bad(|d| d.tile_size = 0));
        assert!(bad(|d| d.layers = 100_000));
        assert!(bad(|d| d.tiles_per_side = 100_000));
        assert!(bad(|d| d.max_pages = 0));
        assert!(bad(|d| {
            d.format = wgpu::TextureFormat::Depth24PlusStencil8;
        }));
        assert!(matches!(pool.add_texture(0, 4), Err(GpuError::Invalid { .. })));
        assert!(matches!(pool.add_texture(8192, 8192), Err(GpuError::Invalid { .. })), "more than 1024 tiles a side");
        assert!(matches!(pool.add_texture(1024, 1024), Err(GpuError::Full { .. })), "page table full");
        assert!(pool.request(TileRequest { texture, mip: 0, x: 8, y: 8 }).is_err());
    }

    #[test]
    fn err_05_rebuild_reloads_pinned_tiles_and_clear_forgets_everything() {
        let gpu = testing::gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        let mut pool = StreamingPool::new(&mut resources, desc(4, 32)).unwrap();
        let texture = pool.add_texture(16, 16).unwrap();
        pool.begin_frame();
        pool.request(TileRequest { texture, mip: 0, x: 3, y: 3 }).unwrap();
        pool.update(&mut resources, 8, fill).unwrap();
        assert_eq!(pool.resident(), 3);

        resources.rebuild(gpu).unwrap();
        pool.rebuild();
        assert_eq!((pool.resident(), pool.queued()), (0, 1), "only the pinned top comes back by itself");
        assert_eq!(pool.lookup(TileRequest { texture, mip: 0, x: 3, y: 3 }), None);
        let report = pool.update(&mut resources, 8, fill).unwrap();
        assert_eq!(report.loaded, 1);
        assert_eq!(report.table_bytes, u64::from(pool.pages_used) * 4, "the whole table goes up again");
        check_invariants(&pool);

        pool.clear();
        assert_eq!((pool.resident(), pool.queued(), pool.pages_used), (0, 0, 0));
        assert!(pool.request(TileRequest { texture, mip: 0, x: 0, y: 0 }).is_err());
        let again = pool.add_texture(16, 16).unwrap();
        assert_eq!(again, 0);
        check_invariants(&pool);
    }
}
