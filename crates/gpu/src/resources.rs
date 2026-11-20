//! Buffers, textures, texture arrays and samplers behind one manager with memory accounting.
//!
//! - `[GPU-01]` Resources are long-lived and addressed by generational ids. Buffers are updated by
//!   queue writes of dirty ranges only ([`DirtyRanges`], [`Resources::flush_dirty`]).
//! - `[GPU-04]` Texture sets are texture arrays ([`TextureDesc::array`]): one binding per material
//!   class, no bindless feature needed.
//! - `[GPU-05]` Every allocation carries a [`Tag`] (category, owner, purpose). The manager keeps
//!   per-category totals against a configurable [`Budget`] and answers each request with a
//!   [`BudgetDecision`]: allow, allow but degrade quality, or refuse. A refusal is an error value
//!   ([`GpuError::OverBudget`]), never a crash.
//! - `[PERF-R4]` Samplers are created once per distinct [`SamplerKey`] and looked up after that.
//! - `[ERR-05]` After a device loss [`Resources::rebuild`] re-creates every live resource on the new
//!   device under its old id; the owner then refills them from CPU state.
//!
//! Sizes are the theoretical footprint (texels times bytes per texel, all mips and layers); the
//! driver's padding is not visible through WebGPU.

use humanart_core::alloc::{Handle, Slab};
use humanart_core::hash::FastMap;

use crate::device::Gpu;
use crate::error::{GpuError, guarded};
use crate::into_wgpu;

/// What a GPU allocation is for: the rows of the memory overlay `[GPU-05]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    /// Vertex, index and topology buffers.
    Geometry,
    /// Deformation inputs and outputs (blend shapes, skinning, simulation state).
    Deformation,
    /// Material textures and texture arrays.
    Textures,
    /// Colour, depth and intermediate render targets.
    RenderTargets,
    /// Uniform and parameter buffers.
    Uniforms,
    /// Staging buffers for readbacks and uploads.
    Staging,
    /// Hair strands and their acceleration data.
    Hair,
    /// The user interface (glyph and image atlases, instance buffers).
    Ui,
    /// Anything else.
    Other,
}

impl Category {
    /// Number of categories.
    pub const COUNT: usize = 9;
    /// Every category, in overlay order.
    pub const ALL: [Self; Self::COUNT] = [
        Self::Geometry,
        Self::Deformation,
        Self::Textures,
        Self::RenderTargets,
        Self::Uniforms,
        Self::Staging,
        Self::Hair,
        Self::Ui,
        Self::Other,
    ];

    /// The category's name in the overlay and in errors.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Geometry => "geometry",
            Self::Deformation => "deformation",
            Self::Textures => "textures",
            Self::RenderTargets => "render targets",
            Self::Uniforms => "uniforms",
            Self::Staging => "staging",
            Self::Hair => "hair",
            Self::Ui => "ui",
            Self::Other => "other",
        }
    }
}

/// The owner id of allocations that belong to no character.
pub const SHARED_OWNER: u32 = 0;

/// The label every allocation carries `[GPU-05]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tag {
    /// The overlay row.
    pub category: Category,
    /// The character (or other owner) the allocation belongs to; [`SHARED_OWNER`] for none.
    pub owner: u32,
    /// What it holds; also the wgpu debug label.
    pub purpose: &'static str,
}

impl Tag {
    /// A tag.
    pub const fn new(category: Category, owner: u32, purpose: &'static str) -> Self {
        Self { category, owner, purpose }
    }

    /// A tag for an allocation shared by all owners.
    pub const fn shared(category: Category, purpose: &'static str) -> Self {
        Self { category, owner: SHARED_OWNER, purpose }
    }
}

/// The memory budget `[GPU-05]`. `u64::MAX` means unlimited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Total bytes above which allocations still succeed but the caller is told to degrade quality.
    pub soft: u64,
    /// Total bytes that are never exceeded: an allocation that would is refused.
    pub hard: u64,
    /// Bytes each category may hold at most, indexed by `Category as usize`.
    pub category_hard: [u64; Category::COUNT],
}

impl Budget {
    /// No limit at all.
    pub const UNLIMITED: Self = Self { soft: u64::MAX, hard: u64::MAX, category_hard: [u64::MAX; Category::COUNT] };

    /// A total budget with a degrade threshold (`soft`) and a refusal threshold (`hard`).
    pub const fn new(soft: u64, hard: u64) -> Self {
        Self { soft, hard, category_hard: [u64::MAX; Category::COUNT] }
    }

    /// The budget with one category capped.
    #[must_use]
    pub const fn with_category(mut self, category: Category, hard: u64) -> Self {
        self.category_hard[category as usize] = hard;
        self
    }
}

/// The manager's answer to "may I allocate this?" `[GPU-05]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetDecision {
    /// Within budget.
    Allow,
    /// The allocation is made, but the total is over the soft budget: lower quality (drop mips,
    /// shrink targets, evict) until [`MemoryStats::over_soft`] is zero again.
    Degrade {
        /// Bytes over the soft budget once this allocation exists.
        over_soft: u64,
    },
    /// The allocation would exceed a hard budget and is not made.
    Refuse {
        /// `"total"` or the name of the capped category.
        scope: &'static str,
        /// Bytes in use in that scope.
        used: u64,
        /// The scope's hard budget.
        budget: u64,
    },
}

impl BudgetDecision {
    /// True unless the allocation is refused.
    pub const fn is_allowed(self) -> bool {
        !matches!(self, Self::Refuse { .. })
    }
}

/// Memory in use, for the profiler overlay `[GPU-05]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStats {
    /// Bytes per category, indexed by `Category as usize`.
    pub bytes: [u64; Category::COUNT],
    /// Live allocations per category.
    pub count: [u32; Category::COUNT],
    /// Bytes in all categories.
    pub total: u64,
    /// The largest total seen.
    pub peak: u64,
    /// The budget in force.
    pub budget: Budget,
    /// Allocations refused so far.
    pub refused: u64,
    /// Bytes written to buffers and textures through the manager so far.
    pub bytes_uploaded: u64,
}

impl MemoryStats {
    /// Bytes over the soft budget (0 when within it).
    pub const fn over_soft(&self) -> u64 {
        self.total.saturating_sub(self.budget.soft)
    }

    /// Bytes of one category.
    pub const fn of(&self, category: Category) -> u64 {
        self.bytes[category as usize]
    }
}

/// A buffer request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDesc {
    /// Size in bytes; rounded up to a multiple of 4 so every queue write is legal.
    pub size: u64,
    /// How the buffer is used.
    pub usage: wgpu::BufferUsages,
    /// The accounting label.
    pub tag: Tag,
}

/// A texture request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureDesc {
    /// Width of mip 0 in texels.
    pub width: u32,
    /// Height of mip 0 in texels.
    pub height: u32,
    /// Array layers (2D) or depth (3D).
    pub depth_or_layers: u32,
    /// Mip levels.
    pub mip_levels: u32,
    /// Samples per texel.
    pub sample_count: u32,
    /// 2D or 3D.
    pub dimension: wgpu::TextureDimension,
    /// The texel format.
    pub format: wgpu::TextureFormat,
    /// How the texture is used.
    pub usage: wgpu::TextureUsages,
    /// View the texture as a 2D array even with one layer `[GPU-04]`.
    pub array: bool,
    /// The accounting label.
    pub tag: Tag,
}

impl TextureDesc {
    /// A single 2D texture with one mip level.
    pub const fn d2(
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        tag: Tag,
    ) -> Self {
        Self {
            width,
            height,
            depth_or_layers: 1,
            mip_levels: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            array: false,
            tag,
        }
    }

    /// A 2D texture array with one mip level `[GPU-04]`.
    pub const fn array(
        width: u32,
        height: u32,
        layers: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        tag: Tag,
    ) -> Self {
        let mut desc = Self::d2(width, height, format, usage, tag);
        desc.depth_or_layers = layers;
        desc.array = true;
        desc
    }

    /// The request with a mip count.
    #[must_use]
    pub const fn mips(mut self, mip_levels: u32) -> Self {
        self.mip_levels = mip_levels;
        self
    }

    /// The request with the full mip chain down to one texel.
    #[must_use]
    pub const fn full_mip_chain(mut self) -> Self {
        self.mip_levels = full_mip_count(self.width, self.height);
        self
    }

    /// The request with a sample count.
    #[must_use]
    pub const fn samples(mut self, sample_count: u32) -> Self {
        self.sample_count = sample_count;
        self
    }

    /// Size of a mip level in texels (width, height), never below 1.
    pub const fn mip_size(&self, mip: u32) -> (u32, u32) {
        (mip_dimension(self.width, mip), mip_dimension(self.height, mip))
    }

    /// Theoretical bytes of one layer of one mip level.
    pub fn mip_bytes(&self, mip: u32) -> u64 {
        let (width, height) = self.mip_size(mip);
        level_bytes(self.format, width, height)
    }

    /// Theoretical bytes of the whole texture: all mips, layers and samples.
    pub fn bytes(&self) -> u64 {
        let mut total = 0_u64;
        for mip in 0..self.mip_levels {
            let layers = match self.dimension {
                wgpu::TextureDimension::D3 => mip_dimension(self.depth_or_layers, mip),
                _ => self.depth_or_layers,
            };
            total = total.saturating_add(self.mip_bytes(mip).saturating_mul(u64::from(layers)));
        }
        total.saturating_mul(u64::from(self.sample_count.max(1)))
    }
}

/// A dimension at a mip level: halved per level, at least 1.
pub const fn mip_dimension(size: u32, mip: u32) -> u32 {
    let shifted = if mip >= 32 { 0 } else { size >> mip };
    if shifted == 0 { 1 } else { shifted }
}

/// Mip levels of a full chain for a 2D size.
pub const fn full_mip_count(width: u32, height: u32) -> u32 {
    let largest = if width > height { width } else { height };
    if largest == 0 { 1 } else { 32 - largest.leading_zeros() }
}

/// Bytes of one texel block of a format, as far as WebGPU tells; depth formats are estimated.
pub fn block_bytes(format: wgpu::TextureFormat) -> u32 {
    use wgpu::TextureFormat as F;
    match format.block_copy_size(None) {
        Some(bytes) => bytes,
        None => match format {
            F::Stencil8 => 1,
            F::Depth16Unorm => 2,
            F::NV12 => 3,
            F::P010 => 6,
            F::Depth32FloatStencil8 => 8,
            _ => 4,
        },
    }
}

/// Theoretical bytes of a `width` x `height` level of a format, whole blocks.
pub fn level_bytes(format: wgpu::TextureFormat, width: u32, height: u32) -> u64 {
    let (block_width, block_height) = format.block_dimensions();
    u64::from(width.div_ceil(block_width)) * u64::from(height.div_ceil(block_height)) * u64::from(block_bytes(format))
}

/// A live buffer in the manager.
#[derive(Debug)]
pub struct BufferEntry {
    /// The wgpu buffer.
    buffer: wgpu::Buffer,
    /// The request it was made from, with the size rounded up.
    desc: BufferDesc,
}

impl BufferEntry {
    /// The wgpu buffer.
    pub fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// The request it was made from; `size` is the allocated size.
    pub fn desc(&self) -> &BufferDesc {
        &self.desc
    }
}

/// A live texture in the manager.
#[derive(Debug)]
pub struct TextureEntry {
    /// The wgpu texture.
    texture: wgpu::Texture,
    /// The view of everything: 2D, 2D array or 3D.
    view: wgpu::TextureView,
    /// The request it was made from.
    desc: TextureDesc,
    /// Accounted bytes.
    bytes: u64,
}

impl TextureEntry {
    /// The wgpu texture.
    pub fn raw(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// The view of the whole texture (a 2D-array view for texture arrays).
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The request it was made from.
    pub fn desc(&self) -> &TextureDesc {
        &self.desc
    }

    /// Accounted bytes.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// Names a buffer in [`Resources`]. Stays valid across [`Resources::rebuild`].
pub type BufferId = Handle<BufferEntry>;
/// Names a texture in [`Resources`]. Stays valid across [`Resources::rebuild`].
pub type TextureId = Handle<TextureEntry>;

/// What makes two samplers the same `[PERF-R4]`.
#[derive(Debug, Clone, Copy)]
pub struct SamplerKey {
    /// Addressing along u, v and w.
    pub address_mode: [wgpu::AddressMode; 3],
    /// Magnification filter.
    pub mag_filter: wgpu::FilterMode,
    /// Minification filter.
    pub min_filter: wgpu::FilterMode,
    /// Filter between mip levels.
    pub mipmap_filter: wgpu::MipmapFilterMode,
    /// Smallest level of detail.
    pub lod_min_clamp: f32,
    /// Largest level of detail.
    pub lod_max_clamp: f32,
    /// Comparison function, for shadow samplers.
    pub compare: Option<wgpu::CompareFunction>,
    /// Anisotropy (1 = off). Needs all three filters linear.
    pub anisotropy_clamp: u16,
}

impl SamplerKey {
    /// Linear filtering, clamped to the edge, level 0 only: the reference application's sampler.
    pub const LINEAR_CLAMP: Self = Self {
        address_mode: [wgpu::AddressMode::ClampToEdge; 3],
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        lod_min_clamp: 0.0,
        lod_max_clamp: 32.0,
        compare: None,
        anisotropy_clamp: 1,
    };

    /// Nearest filtering, clamped to the edge.
    pub const NEAREST_CLAMP: Self =
        Self { mag_filter: wgpu::FilterMode::Nearest, min_filter: wgpu::FilterMode::Nearest, ..Self::LINEAR_CLAMP };

    /// Trilinear filtering with repeat addressing and the given anisotropy: material textures.
    pub const fn trilinear_repeat(anisotropy: u16) -> Self {
        Self {
            address_mode: [wgpu::AddressMode::Repeat; 3],
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: if anisotropy == 0 { 1 } else { anisotropy },
            ..Self::LINEAR_CLAMP
        }
    }

    /// The key as plain bits, so that it can be hashed and compared exactly.
    fn bits(&self) -> SamplerBits {
        (
            (self.address_mode, self.mag_filter, self.min_filter),
            self.mipmap_filter,
            self.lod_min_clamp.to_bits(),
            self.lod_max_clamp.to_bits(),
            self.compare,
            self.anisotropy_clamp,
        )
    }
}

/// A [`SamplerKey`] with its floats as bit patterns.
type SamplerBits = (
    ([wgpu::AddressMode; 3], wgpu::FilterMode, wgpu::FilterMode),
    wgpu::MipmapFilterMode,
    u32,
    u32,
    Option<wgpu::CompareFunction>,
    u16,
);

impl PartialEq for SamplerKey {
    fn eq(&self, other: &Self) -> bool {
        self.bits() == other.bits()
    }
}

impl Eq for SamplerKey {}

impl std::hash::Hash for SamplerKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bits().hash(state);
    }
}

/// Most dirty ranges tracked per buffer before neighbours are merged.
pub const MAX_DIRTY_RANGES: usize = 8;

/// The byte ranges of a CPU-side buffer mirror that changed since the last upload `[GPU-01]`.
///
/// Fixed capacity, no allocation: when more than [`MAX_DIRTY_RANGES`] separate ranges are marked,
/// the two closest are merged (uploading the gap between them too). Ranges are widened to multiples
/// of 4 bytes, the granularity of queue writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DirtyRanges {
    /// Sorted, separate `(start, end)` pairs.
    ranges: [(u64, u64); MAX_DIRTY_RANGES],
    /// Pairs in use.
    len: usize,
}

impl DirtyRanges {
    /// Nothing dirty.
    pub const fn new() -> Self {
        Self { ranges: [(0, 0); MAX_DIRTY_RANGES], len: 0 }
    }

    /// Marks `len` bytes from `offset` as changed.
    pub fn mark(&mut self, offset: u64, len: u64) {
        if len == 0 {
            return;
        }
        let mut start = offset & !3;
        let mut end = offset.saturating_add(len).saturating_add(3) & !3;
        let mut merged = [(0_u64, 0_u64); MAX_DIRTY_RANGES + 1];
        let mut count = 0;
        let mut placed = false;
        for &(range_start, range_end) in &self.ranges[..self.len] {
            if range_end < start {
                merged[count] = (range_start, range_end);
                count += 1;
            } else if range_start > end {
                if !placed {
                    merged[count] = (start, end);
                    count += 1;
                    placed = true;
                }
                merged[count] = (range_start, range_end);
                count += 1;
            } else {
                start = start.min(range_start);
                end = end.max(range_end);
            }
        }
        if !placed {
            merged[count] = (start, end);
            count += 1;
        }
        if count > MAX_DIRTY_RANGES {
            // Close the smallest gap. `count` is at least 2 here.
            let mut best = 0;
            let mut best_gap = u64::MAX;
            for i in 0..count - 1 {
                let gap = merged[i + 1].0 - merged[i].1;
                if gap < best_gap {
                    best_gap = gap;
                    best = i;
                }
            }
            merged[best].1 = merged[best + 1].1;
            merged.copy_within(best + 2..count, best + 1);
            count -= 1;
        }
        self.ranges.copy_from_slice(&merged[..MAX_DIRTY_RANGES]);
        self.len = count;
    }

    /// Marks everything up to `len` bytes as changed.
    pub fn mark_all(&mut self, len: u64) {
        self.len = 0;
        self.mark(0, len);
    }

    /// True when nothing changed.
    pub const fn is_clean(&self) -> bool {
        self.len == 0
    }

    /// The dirty `(start, end)` ranges, sorted and separate.
    pub fn ranges(&self) -> &[(u64, u64)] {
        &self.ranges[..self.len]
    }

    /// Bytes covered by the dirty ranges.
    pub fn total_bytes(&self) -> u64 {
        self.ranges().iter().map(|&(start, end)| end - start).sum()
    }

    /// Forgets every range.
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

/// The resource manager: every buffer, texture and sampler of the application `[GPU-05]`.
#[derive(Debug)]
pub struct Resources {
    /// The device resources are made on.
    device: wgpu::Device,
    /// The queue writes go through.
    queue: wgpu::Queue,
    /// The device's features, to refuse formats it cannot hold.
    features: wgpu::Features,
    /// The device's limits, to refuse sizes it cannot hold.
    limits: wgpu::Limits,
    /// Live buffers.
    buffers: Slab<BufferEntry>,
    /// Live textures.
    textures: Slab<TextureEntry>,
    /// One sampler per distinct key.
    samplers: FastMap<SamplerKey, wgpu::Sampler>,
    /// Samplers created so far (not lookups).
    samplers_created: u32,
    /// The accounting.
    stats: MemoryStats,
}

impl Resources {
    /// An empty manager on `gpu`'s device with a budget.
    pub fn new(gpu: &Gpu, budget: Budget) -> Self {
        Self {
            device: gpu.device.clone(),
            queue: gpu.queue.clone(),
            features: gpu.features(),
            limits: gpu.limits().clone(),
            buffers: Slab::new(),
            textures: Slab::new(),
            samplers: FastMap::default(),
            samplers_created: 0,
            stats: MemoryStats {
                bytes: [0; Category::COUNT],
                count: [0; Category::COUNT],
                total: 0,
                peak: 0,
                budget,
                refused: 0,
                bytes_uploaded: 0,
            },
        }
    }

    /// The device the resources live on (the new one after [`Resources::rebuild`]).
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue writes go through.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// The device's limits.
    pub fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }

    /// The accounting, for the overlay.
    pub fn stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Changes the budget. Existing allocations stay; [`MemoryStats::over_soft`] tells how much to
    /// give back.
    pub fn set_budget(&mut self, budget: Budget) {
        self.stats.budget = budget;
    }

    /// Bytes held by one owner (a character) across all categories.
    pub fn bytes_of_owner(&self, owner: u32) -> u64 {
        let buffers: u64 =
            self.buffers.iter().filter(|(_, e)| e.desc.tag.owner == owner).map(|(_, e)| e.desc.size).sum();
        let textures: u64 = self.textures.iter().filter(|(_, e)| e.desc.tag.owner == owner).map(|(_, e)| e.bytes).sum();
        buffers + textures
    }

    /// Calls `visit` with the tag and bytes of every live allocation, for the overlay's detail view.
    pub fn for_each_allocation(&self, mut visit: impl FnMut(Tag, u64)) {
        for (_, entry) in self.buffers.iter() {
            visit(entry.desc.tag, entry.desc.size);
        }
        for (_, entry) in self.textures.iter() {
            visit(entry.desc.tag, entry.bytes);
        }
    }

    /// What the budget says about allocating `bytes` in `category`, without allocating.
    pub fn decide(&self, category: Category, bytes: u64) -> BudgetDecision {
        let index = category as usize;
        let budget = &self.stats.budget;
        if self.stats.bytes[index].saturating_add(bytes) > budget.category_hard[index] {
            return BudgetDecision::Refuse {
                scope: category.name(),
                used: self.stats.bytes[index],
                budget: budget.category_hard[index],
            };
        }
        let total = self.stats.total.saturating_add(bytes);
        if total > budget.hard {
            BudgetDecision::Refuse { scope: "total", used: self.stats.total, budget: budget.hard }
        } else if total > budget.soft {
            BudgetDecision::Degrade { over_soft: total - budget.soft }
        } else {
            BudgetDecision::Allow
        }
    }

    /// [`Resources::decide`] with the refusal turned into the error creation returns.
    fn admit(&mut self, category: Category, bytes: u64) -> Result<BudgetDecision, GpuError> {
        match self.decide(category, bytes) {
            BudgetDecision::Refuse { scope, used, budget } => {
                self.stats.refused += 1;
                Err(GpuError::OverBudget { category: scope, requested: bytes, used, budget })
            }
            decision => Ok(decision),
        }
    }

    /// Books an allocation that was made.
    fn book(&mut self, category: Category, bytes: u64) {
        let index = category as usize;
        self.stats.bytes[index] = self.stats.bytes[index].saturating_add(bytes);
        self.stats.count[index] = self.stats.count[index].saturating_add(1);
        self.stats.total = self.stats.total.saturating_add(bytes);
        self.stats.peak = self.stats.peak.max(self.stats.total);
    }

    /// Books an allocation that was released.
    fn unbook(&mut self, category: Category, bytes: u64) {
        let index = category as usize;
        self.stats.bytes[index] = self.stats.bytes[index].saturating_sub(bytes);
        self.stats.count[index] = self.stats.count[index].saturating_sub(1);
        self.stats.total = self.stats.total.saturating_sub(bytes);
    }

    /// Creates a buffer. The decision is `Allow` or `Degrade`; a refusal is
    /// [`GpuError::OverBudget`] and nothing is allocated.
    pub fn create_buffer(&mut self, desc: &BufferDesc) -> Result<(BufferId, BudgetDecision), GpuError> {
        if desc.size == 0 {
            return Err(GpuError::Invalid { what: "a buffer of zero bytes" });
        }
        let size = desc.size.checked_next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT).unwrap_or(u64::MAX);
        if size > self.limits.max_buffer_size {
            return Err(GpuError::OutOfRange {
                what: "buffer size against the device's max_buffer_size",
                offset: 0,
                len: desc.size,
                capacity: self.limits.max_buffer_size,
            });
        }
        let decision = self.admit(desc.tag.category, size)?;
        let desc = BufferDesc { size, ..*desc };
        let buffer = make_buffer(&self.device, &desc)?;
        let id = self
            .buffers
            .try_insert(BufferEntry { buffer, desc })
            .map_err(|_| GpuError::Full { what: "the buffer table" })?;
        self.book(desc.tag.category, size);
        Ok((id, decision))
    }

    /// Creates a texture or texture array `[GPU-04]`. The decision is `Allow` or `Degrade`; a
    /// refusal is [`GpuError::OverBudget`] and nothing is allocated.
    pub fn create_texture(&mut self, desc: &TextureDesc) -> Result<(TextureId, BudgetDecision), GpuError> {
        self.validate_texture(desc)?;
        let bytes = desc.bytes();
        let decision = self.admit(desc.tag.category, bytes)?;
        let (texture, view) = make_texture(&self.device, desc)?;
        let id = self
            .textures
            .try_insert(TextureEntry { texture, view, desc: *desc, bytes })
            .map_err(|_| GpuError::Full { what: "the texture table" })?;
        self.book(desc.tag.category, bytes);
        Ok((id, decision))
    }

    /// Checks a texture request against the device before anything is allocated.
    fn validate_texture(&self, desc: &TextureDesc) -> Result<(), GpuError> {
        if desc.width == 0 || desc.height == 0 || desc.depth_or_layers == 0 || desc.mip_levels == 0 {
            return Err(GpuError::Invalid { what: "a texture with a zero dimension or no mip level" });
        }
        let (max_side, max_depth, what) = match desc.dimension {
            wgpu::TextureDimension::D3 => {
                (self.limits.max_texture_dimension_3d, self.limits.max_texture_dimension_3d, "3D texture size")
            }
            _ => (self.limits.max_texture_dimension_2d, self.limits.max_texture_array_layers, "2D texture size"),
        };
        for (value, max) in [(desc.width, max_side), (desc.height, max_side), (desc.depth_or_layers, max_depth)] {
            if value > max {
                return Err(GpuError::OutOfRange { what, offset: 0, len: u64::from(value), capacity: u64::from(max) });
            }
        }
        if desc.mip_levels > full_mip_count(desc.width, desc.height) {
            return Err(GpuError::Invalid { what: "more mip levels than the texture size has" });
        }
        if !self.features.contains(desc.format.required_features()) {
            return Err(GpuError::Invalid { what: "a texture format the device has no feature for" });
        }
        Ok(())
    }

    /// The buffer behind an id.
    pub fn buffer(&self, id: BufferId) -> Option<&BufferEntry> {
        self.buffers.get(id)
    }

    /// The texture behind an id.
    pub fn texture(&self, id: TextureId) -> Option<&TextureEntry> {
        self.textures.get(id)
    }

    /// Destroys a buffer and returns its bytes to the budget.
    pub fn destroy_buffer(&mut self, id: BufferId) -> Result<(), GpuError> {
        let entry = self.buffers.remove(id).ok_or(GpuError::Stale { what: "buffer id" })?;
        entry.buffer.destroy();
        self.unbook(entry.desc.tag.category, entry.desc.size);
        Ok(())
    }

    /// Destroys a texture and returns its bytes to the budget.
    pub fn destroy_texture(&mut self, id: TextureId) -> Result<(), GpuError> {
        let entry = self.textures.remove(id).ok_or(GpuError::Stale { what: "texture id" })?;
        entry.texture.destroy();
        self.unbook(entry.desc.tag.category, entry.bytes);
        Ok(())
    }

    /// Writes `bytes` into a buffer at `offset`. Offset and length must be multiples of 4 and
    /// inside the buffer. Steady-state safe: no allocation in this crate.
    pub fn write_buffer(&mut self, id: BufferId, offset: u64, bytes: &[u8]) -> Result<(), GpuError> {
        let entry = self.buffers.get(id).ok_or(GpuError::Stale { what: "buffer id" })?;
        let len = bytes.len() as u64;
        if !offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) || !len.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
            return Err(GpuError::Invalid { what: "a buffer write that is not aligned to 4 bytes" });
        }
        if offset.checked_add(len).is_none_or(|end| end > entry.desc.size) {
            return Err(GpuError::OutOfRange { what: "buffer write", offset, len, capacity: entry.desc.size });
        }
        if len > 0 {
            into_wgpu(|| self.queue.write_buffer(&entry.buffer, offset, bytes));
            self.stats.bytes_uploaded += len;
        }
        Ok(())
    }

    /// Uploads the dirty ranges of `mirror`, the CPU-side copy of the buffer's start, and clears
    /// them `[GPU-01]`. Returns the bytes written. `mirror.len()` must be a multiple of 4.
    pub fn flush_dirty(&mut self, id: BufferId, mirror: &[u8], dirty: &mut DirtyRanges) -> Result<u64, GpuError> {
        let mut written = 0;
        for &(start, end) in dirty.ranges() {
            let range = usize::try_from(start).ok().zip(usize::try_from(end).ok());
            let Some(bytes) = range.and_then(|(start, end)| mirror.get(start..end)) else {
                return Err(GpuError::OutOfRange {
                    what: "dirty range against the CPU mirror",
                    offset: start,
                    len: end - start,
                    capacity: mirror.len() as u64,
                });
            };
            self.write_buffer(id, start, bytes)?;
            written += end - start;
        }
        dirty.clear();
        Ok(written)
    }

    /// Writes one whole mip level of one layer. `bytes` is tightly packed rows of blocks.
    pub fn write_texture_level(&mut self, id: TextureId, mip: u32, layer: u32, bytes: &[u8]) -> Result<(), GpuError> {
        let entry = self.textures.get(id).ok_or(GpuError::Stale { what: "texture id" })?;
        if mip >= entry.desc.mip_levels {
            return Err(GpuError::OutOfRange {
                what: "texture mip level",
                offset: u64::from(mip),
                len: 1,
                capacity: u64::from(entry.desc.mip_levels),
            });
        }
        let (width, height) = entry.desc.mip_size(mip);
        self.write_texture_region(id, mip, [0, 0, layer], width, height, bytes)
    }

    /// Writes a rectangle of one mip level; `origin` is (x, y, layer). The rectangle must start on
    /// a block boundary and either end on one or reach the level's edge. `bytes` is tightly packed.
    pub fn write_texture_region(
        &mut self,
        id: TextureId,
        mip: u32,
        origin: [u32; 3],
        width: u32,
        height: u32,
        bytes: &[u8],
    ) -> Result<(), GpuError> {
        let entry = self.textures.get(id).ok_or(GpuError::Stale { what: "texture id" })?;
        let desc = &entry.desc;
        let [x, y, layer] = origin;
        // A 3D texture's depth halves with each mip like its width and height; array layers do not.
        let layers = match desc.dimension {
            wgpu::TextureDimension::D3 => mip_dimension(desc.depth_or_layers, mip),
            _ => desc.depth_or_layers,
        };
        if mip >= desc.mip_levels || layer >= layers || width == 0 || height == 0 {
            return Err(GpuError::Invalid { what: "a texture write outside the mip levels or layers, or empty" });
        }
        let (mip_width, mip_height) = desc.mip_size(mip);
        let (block_width, block_height) = desc.format.block_dimensions();
        let fits = |start: u32, len: u32, size: u32, block: u32| {
            start.is_multiple_of(block)
                && start.checked_add(len).is_some_and(|end| end <= size && (end.is_multiple_of(block) || end == size))
        };
        if !fits(x, width, mip_width, block_width) || !fits(y, height, mip_height, block_height) {
            return Err(GpuError::OutOfRange {
                what: "texture write rectangle",
                offset: u64::from(x.max(y)),
                len: u64::from(width.max(height)),
                capacity: u64::from(mip_width.min(mip_height)),
            });
        }
        let expected = level_bytes(desc.format, width, height);
        if bytes.len() as u64 != expected {
            return Err(GpuError::OutOfRange {
                what: "texture write data against the rectangle's size",
                offset: 0,
                len: bytes.len() as u64,
                capacity: expected,
            });
        }
        let bytes_per_row = width.div_ceil(block_width).saturating_mul(block_bytes(desc.format));
        let rows = height.div_ceil(block_height);
        into_wgpu(|| {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &entry.texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d { x, y, z: layer },
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(rows),
                },
                wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            );
        });
        self.stats.bytes_uploaded += expected;
        Ok(())
    }

    /// The sampler for a key: created at first use, looked up without allocation after `[PERF-R4]`.
    pub fn sampler(&mut self, key: &SamplerKey) -> Result<&wgpu::Sampler, GpuError> {
        if !self.samplers.contains_key(key) {
            let sampler = make_sampler(&self.device, key)?;
            self.samplers.insert(*key, sampler);
            self.samplers_created += 1;
        }
        self.samplers.get(key).ok_or(GpuError::Stale { what: "sampler key" })
    }

    /// Samplers created so far. Does not grow in steady state.
    pub fn samplers_created(&self) -> u32 {
        self.samplers_created
    }

    /// Number of live buffers and textures.
    pub fn live(&self) -> (usize, usize) {
        (self.buffers.len(), self.textures.len())
    }

    /// Re-creates every live buffer, texture and sampler on `gpu`'s device after a device loss
    /// `[ERR-05]`. Ids, descriptors and accounting stay as they are; the contents are gone and the
    /// owner writes them again from CPU state. Returns how many resources were re-created.
    pub fn rebuild(&mut self, gpu: &Gpu) -> Result<usize, GpuError> {
        self.device = gpu.device.clone();
        self.queue = gpu.queue.clone();
        self.features = gpu.features();
        self.limits = gpu.limits().clone();
        let mut count = 0;
        for (_, entry) in self.buffers.iter_mut() {
            entry.buffer = make_buffer(&self.device, &entry.desc)?;
            count += 1;
        }
        for (_, entry) in self.textures.iter_mut() {
            (entry.texture, entry.view) = make_texture(&self.device, &entry.desc)?;
            count += 1;
        }
        for (key, sampler) in &mut self.samplers {
            *sampler = make_sampler(&self.device, key)?;
            count += 1;
        }
        Ok(count)
    }
}

/// Creates the wgpu buffer of a request.
fn make_buffer(device: &wgpu::Device, desc: &BufferDesc) -> Result<wgpu::Buffer, GpuError> {
    guarded(device, "buffer creation", || {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(desc.tag.purpose),
            size: desc.size,
            usage: desc.usage,
            mapped_at_creation: false,
        })
    })
}

/// Creates the wgpu texture of a request and the view of all of it.
fn make_texture(device: &wgpu::Device, desc: &TextureDesc) -> Result<(wgpu::Texture, wgpu::TextureView), GpuError> {
    guarded(device, "texture creation", || {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(desc.tag.purpose),
            size: wgpu::Extent3d {
                width: desc.width,
                height: desc.height,
                depth_or_array_layers: desc.depth_or_layers,
            },
            mip_level_count: desc.mip_levels,
            sample_count: desc.sample_count,
            dimension: desc.dimension,
            format: desc.format,
            usage: desc.usage,
            view_formats: &[],
        });
        let dimension = match desc.dimension {
            wgpu::TextureDimension::D2 if desc.array => Some(wgpu::TextureViewDimension::D2Array),
            _ => None,
        };
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(desc.tag.purpose),
            dimension,
            ..Default::default()
        });
        (texture, view)
    })
}

/// Creates the wgpu sampler of a key.
fn make_sampler(device: &wgpu::Device, key: &SamplerKey) -> Result<wgpu::Sampler, GpuError> {
    guarded(device, "sampler creation", || {
        device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cached sampler"),
            address_mode_u: key.address_mode[0],
            address_mode_v: key.address_mode[1],
            address_mode_w: key.address_mode[2],
            mag_filter: key.mag_filter,
            min_filter: key.min_filter,
            mipmap_filter: key.mipmap_filter,
            lod_min_clamp: key.lod_min_clamp,
            lod_max_clamp: key.lod_max_clamp,
            compare: key.compare,
            anisotropy_clamp: key.anisotropy_clamp,
            border_color: None,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing;

    const STORAGE: wgpu::BufferUsages =
        wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_DST).union(wgpu::BufferUsages::COPY_SRC);
    const SAMPLED: wgpu::TextureUsages =
        wgpu::TextureUsages::TEXTURE_BINDING.union(wgpu::TextureUsages::COPY_DST).union(wgpu::TextureUsages::COPY_SRC);

    fn buffer(size: u64, category: Category, owner: u32) -> BufferDesc {
        BufferDesc { size, usage: STORAGE, tag: Tag::new(category, owner, "test buffer") }
    }

    #[test]
    fn gpu_05_allocations_are_tagged_and_totalled_per_category() {
        let mut resources = Resources::new(testing::gpu(), Budget::UNLIMITED);
        let (a, decision) = resources.create_buffer(&buffer(1000, Category::Geometry, 7)).unwrap();
        assert_eq!(decision, BudgetDecision::Allow);
        let (b, _) = resources.create_buffer(&buffer(4096, Category::Deformation, 7)).unwrap();
        let texture = TextureDesc::array(
            64,
            32,
            3,
            wgpu::TextureFormat::Rgba8Unorm,
            SAMPLED,
            Tag::new(Category::Textures, 9, "albedo"),
        )
        .full_mip_chain();
        assert_eq!(texture.mip_levels, 7);
        // 64x32 + 32x16 + 16x8 + 8x4 + 4x2 + 2x1 + 1x1 texels, 4 bytes, 3 layers.
        let texels = 2048 + 512 + 128 + 32 + 8 + 2 + 1;
        assert_eq!(texture.bytes(), texels * 4 * 3);
        let (t, _) = resources.create_texture(&texture).unwrap();

        let stats = *resources.stats();
        assert_eq!(stats.of(Category::Geometry), 1000);
        assert_eq!(stats.of(Category::Deformation), 4096);
        assert_eq!(stats.of(Category::Textures), texture.bytes());
        assert_eq!(stats.total, 1000 + 4096 + texture.bytes());
        assert_eq!(stats.count[Category::Textures as usize], 1);
        assert_eq!(resources.bytes_of_owner(7), 5096);
        assert_eq!(resources.bytes_of_owner(9), texture.bytes());
        let mut seen = Vec::new();
        resources.for_each_allocation(|tag, bytes| seen.push((tag.category, tag.owner, tag.purpose, bytes)));
        assert!(seen.contains(&(Category::Textures, 9, "albedo", texture.bytes())));
        assert_eq!(seen.len(), 3);
        assert_eq!(resources.buffer(a).unwrap().raw().size(), 1000);
        assert_eq!(resources.texture(t).unwrap().raw().depth_or_array_layers(), 3);

        resources.destroy_buffer(b).unwrap();
        resources.destroy_texture(t).unwrap();
        assert_eq!(resources.stats().total, 1000);
        assert_eq!(resources.stats().peak, stats.total);
        assert!(matches!(resources.destroy_buffer(b), Err(GpuError::Stale { .. })), "a stale id is an error");
        assert!(resources.texture(t).is_none());
        assert_eq!(resources.live(), (1, 0));
    }

    #[test]
    fn gpu_05_budget_allows_degrades_and_refuses_without_crashing() {
        let budget = Budget::new(10_000, 20_000).with_category(Category::Hair, 4_000);
        let mut resources = Resources::new(testing::gpu(), budget);
        assert_eq!(resources.decide(Category::Geometry, 10_000), BudgetDecision::Allow);
        let (_, first) = resources.create_buffer(&buffer(8_000, Category::Geometry, 1)).unwrap();
        assert_eq!(first, BudgetDecision::Allow);

        // Over the soft budget: made, with the order to degrade.
        let (second, decision) = resources.create_buffer(&buffer(6_000, Category::Geometry, 1)).unwrap();
        assert_eq!(decision, BudgetDecision::Degrade { over_soft: 4_000 });
        assert_eq!(resources.stats().over_soft(), 4_000);

        // Over the hard budget: refused as an error value, nothing allocated or booked.
        let error = resources.create_buffer(&buffer(8_000, Category::Geometry, 1)).unwrap_err();
        assert!(error.is_out_of_memory());
        assert!(
            matches!(error, GpuError::OverBudget { category: "total", requested: 8_000, used: 14_000, budget: 20_000 }),
            "{error}"
        );
        assert_eq!(resources.stats().total, 14_000);
        assert_eq!(resources.stats().refused, 1);
        assert_eq!(resources.live().0, 2);

        // A capped category refuses although the total has room.
        let error = resources.create_buffer(&buffer(4_004, Category::Hair, 1)).unwrap_err();
        assert!(matches!(error, GpuError::OverBudget { category: "hair", .. }), "{error}");
        let texture =
            TextureDesc::d2(64, 64, wgpu::TextureFormat::Rgba8Unorm, SAMPLED, Tag::shared(Category::Hair, "t"));
        assert!(!resources.decide(Category::Hair, texture.bytes()).is_allowed());
        assert!(resources.create_texture(&texture).is_err());

        // Degrading (here: freeing) brings the total back under the soft budget.
        resources.destroy_buffer(second).unwrap();
        assert_eq!(resources.stats().over_soft(), 0);
        resources.set_budget(Budget::UNLIMITED);
        assert_eq!(resources.decide(Category::Hair, 1 << 40), BudgetDecision::Allow);
    }

    #[test]
    fn err_02_bad_requests_are_errors_not_panics() {
        let gpu = testing::gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        assert!(matches!(resources.create_buffer(&buffer(0, Category::Other, 0)), Err(GpuError::Invalid { .. })));
        let too_big = gpu.limits().max_buffer_size + 4;
        assert!(matches!(
            resources.create_buffer(&buffer(too_big, Category::Other, 0)),
            Err(GpuError::OutOfRange { .. })
        ));
        let tag = Tag::shared(Category::Textures, "bad");
        let side = gpu.limits().max_texture_dimension_2d + 1;
        let huge = TextureDesc::d2(side, 4, wgpu::TextureFormat::R8Unorm, SAMPLED, tag);
        assert!(matches!(resources.create_texture(&huge), Err(GpuError::OutOfRange { .. })));
        let mips = TextureDesc::d2(4, 4, wgpu::TextureFormat::R8Unorm, SAMPLED, tag).mips(9);
        assert!(matches!(resources.create_texture(&mips), Err(GpuError::Invalid { .. })));
        let zero = TextureDesc::d2(0, 4, wgpu::TextureFormat::R8Unorm, SAMPLED, tag);
        assert!(matches!(resources.create_texture(&zero), Err(GpuError::Invalid { .. })));
        // A request wgpu itself rejects (a buffer mapped for reading and writing) comes back as a
        // validation error through the error scope.
        let both = BufferDesc {
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE,
            tag: Tag::shared(Category::Other, "invalid usage"),
        };
        assert!(matches!(resources.create_buffer(&both), Err(GpuError::Validation { .. })));
        assert_eq!(resources.stats().total, 0, "nothing was booked for failed requests");
        assert_eq!(resources.live(), (0, 0));
    }

    #[test]
    fn dirty_ranges_merge_sort_and_stay_within_capacity() {
        let mut dirty = DirtyRanges::new();
        assert!(dirty.is_clean());
        dirty.mark(100, 8);
        dirty.mark(10, 1);
        dirty.mark(0, 0);
        assert_eq!(dirty.ranges(), &[(8, 12), (100, 108)], "widened to 4 bytes and sorted");
        dirty.mark(12, 4);
        assert_eq!(dirty.ranges(), &[(8, 16), (100, 108)], "touching ranges merge");
        dirty.mark(14, 90);
        assert_eq!(dirty.ranges(), &[(8, 108)], "a bridge merges both sides");
        assert_eq!(dirty.total_bytes(), 100);

        // Nine separate ranges: the two closest merge, so at most eight remain and all marked
        // bytes stay covered.
        let mut dirty = DirtyRanges::new();
        let starts = [0_u64, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 7016];
        for start in starts {
            dirty.mark(start, 8);
        }
        assert_eq!(dirty.ranges().len(), MAX_DIRTY_RANGES);
        assert_eq!(dirty.ranges()[7], (7000, 7024));
        for start in starts {
            assert!(dirty.ranges().iter().any(|&(s, e)| s <= start && start + 8 <= e));
        }
        dirty.mark_all(64);
        assert_eq!(dirty.ranges(), &[(0, 64)]);
        dirty.clear();
        assert!(dirty.is_clean());
    }

    #[test]
    fn dirty_ranges_cover_every_marked_byte_under_random_marks() {
        let mut rng = humanart_core::rng::Rng::new(0x6770_7501);
        for _ in 0..200 {
            let mut dirty = DirtyRanges::new();
            let mut marked = [false; 512];
            for _ in 0..=rng.index(23) {
                let offset = rng.index(500);
                let len = rng.index(12);
                dirty.mark(offset as u64, len as u64);
                marked[offset..offset + len].fill(true);
            }
            let ranges = dirty.ranges();
            assert!(ranges.len() <= MAX_DIRTY_RANGES);
            assert!(ranges.windows(2).all(|pair| pair[0].1 < pair[1].0), "sorted and separate: {ranges:?}");
            assert!(ranges.iter().all(|&(s, e)| s % 4 == 0 && e % 4 == 0 && s < e));
            for (byte, _) in marked.iter().enumerate().filter(|(_, m)| **m) {
                assert!(ranges.iter().any(|&(s, e)| s <= byte as u64 && (byte as u64) < e), "byte {byte} covered");
            }
        }
    }

    #[test]
    fn gpu_01_only_dirty_ranges_are_uploaded_and_the_buffer_matches_the_mirror() {
        let gpu = testing::gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        let (id, _) = resources.create_buffer(&buffer(256, Category::Deformation, 1)).unwrap();
        let mut mirror: Vec<u8> = (0..=255).collect();
        let mut dirty = DirtyRanges::new();
        dirty.mark_all(256);
        assert_eq!(resources.flush_dirty(id, &mirror, &mut dirty).unwrap(), 256);
        assert!(dirty.is_clean());

        mirror[17] = 0xAA;
        dirty.mark(17, 1);
        mirror[200..210].fill(0xBB);
        dirty.mark(200, 10);
        let before = resources.stats().bytes_uploaded;
        assert_eq!(resources.flush_dirty(id, &mirror, &mut dirty).unwrap(), 4 + 12);
        assert_eq!(resources.stats().bytes_uploaded - before, 16, "16 bytes went to the queue, not 256");
        assert_eq!(resources.flush_dirty(id, &mirror, &mut dirty).unwrap(), 0, "clean: nothing to write");
        gpu.queue.submit([]);
        assert_eq!(testing::read_buffer(gpu, resources.buffer(id).unwrap().raw(), 0, 256), mirror);

        // Writes outside the buffer or the mirror, and unaligned ones, are errors.
        assert!(matches!(resources.write_buffer(id, 252, &[0; 8]), Err(GpuError::OutOfRange { .. })));
        assert!(matches!(resources.write_buffer(id, 2, &[0; 4]), Err(GpuError::Invalid { .. })));
        assert!(matches!(resources.write_buffer(id, u64::MAX - 3, &[0; 4]), Err(GpuError::OutOfRange { .. })));
        dirty.mark(300, 4);
        assert!(matches!(resources.flush_dirty(id, &mirror, &mut dirty), Err(GpuError::OutOfRange { .. })));
    }

    #[test]
    fn gpu_04_texture_array_layers_and_mips_are_written_and_read_back() {
        let gpu = testing::gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        let desc = TextureDesc::array(
            8,
            4,
            3,
            wgpu::TextureFormat::Rgba8Unorm,
            SAMPLED,
            Tag::new(Category::Textures, 1, "skin set"),
        )
        .mips(2);
        let (id, _) = resources.create_texture(&desc).unwrap();
        let level = |layer: u8, mip: u32| -> Vec<u8> {
            (0..desc.mip_bytes(mip)).map(|i| (i as u8).wrapping_mul(3).wrapping_add(layer * 50 + mip as u8)).collect()
        };
        for layer in 0..3 {
            for mip in 0..2 {
                resources.write_texture_level(id, mip, u32::from(layer), &level(layer, mip)).unwrap();
            }
        }
        // Read layer 2, mip 1 (4x2 texels) back through a buffer.
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("texture readback"),
            size: 256 * 2,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: resources.texture(id).unwrap().raw(),
                mip_level: 1,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 2 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: None },
            },
            wgpu::Extent3d { width: 4, height: 2, depth_or_array_layers: 1 },
        );
        gpu.queue.submit([encoder.finish()]);
        let padded = testing::read_mappable(gpu, &staging);
        let rows: Vec<u8> = padded.chunks(256).flat_map(|row| &row[..16]).copied().collect();
        assert_eq!(rows, level(2, 1));

        // Wrong sizes, levels and layers are errors.
        assert!(matches!(resources.write_texture_level(id, 0, 0, &[0; 5]), Err(GpuError::OutOfRange { .. })));
        assert!(matches!(resources.write_texture_level(id, 2, 0, &[0; 4]), Err(GpuError::OutOfRange { .. })));
        assert!(matches!(resources.write_texture_level(id, 0, 3, &level(0, 0)), Err(GpuError::Invalid { .. })));
        assert!(matches!(
            resources.write_texture_region(id, 0, [6, 0, 0], 4, 1, &[0; 16]),
            Err(GpuError::OutOfRange { .. })
        ));
        resources.write_texture_region(id, 0, [6, 3, 1], 2, 1, &[9; 8]).unwrap();
        assert_eq!(gpu.events().pending(), 0, "no uncaptured error");
    }

    #[test]
    fn plat_05_block_compressed_sizes_and_missing_format_features() {
        let tag = Tag::shared(Category::Textures, "bc");
        let bc = TextureDesc::d2(10, 6, wgpu::TextureFormat::Bc7RgbaUnorm, SAMPLED, tag);
        assert_eq!(bc.mip_bytes(0), 3 * 2 * 16, "whole 4x4 blocks of 16 bytes");
        assert_eq!(level_bytes(wgpu::TextureFormat::Bc1RgbaUnorm, 4, 4), 8);
        assert_eq!(block_bytes(wgpu::TextureFormat::Depth32Float), 4);
        assert_eq!(block_bytes(wgpu::TextureFormat::Depth24PlusStencil8), 4);
        assert_eq!((mip_dimension(5, 1), mip_dimension(5, 3), mip_dimension(5, 40)), (2, 1, 1));
        assert_eq!((full_mip_count(1, 1), full_mip_count(8192, 16), full_mip_count(5, 3)), (1, 14, 3));

        // On a device without BC (forced off), a BC texture is refused before wgpu is asked.
        let gpu = testing::gpu();
        let mut resources = Resources::new(gpu, Budget::UNLIMITED);
        resources.features.remove(wgpu::Features::TEXTURE_COMPRESSION_BC);
        assert!(matches!(resources.create_texture(&bc), Err(GpuError::Invalid { .. })));
    }

    #[test]
    fn perf_r4_samplers_are_created_once_per_key() {
        let mut resources = Resources::new(testing::gpu(), Budget::UNLIMITED);
        resources.sampler(&SamplerKey::LINEAR_CLAMP).unwrap();
        resources.sampler(&SamplerKey::trilinear_repeat(8)).unwrap();
        assert_eq!(resources.samplers_created(), 2);
        for _ in 0..100 {
            resources.sampler(&SamplerKey::LINEAR_CLAMP).unwrap();
            resources.sampler(&SamplerKey::trilinear_repeat(8)).unwrap();
        }
        assert_eq!(resources.samplers_created(), 2, "lookups create nothing");
        resources.sampler(&SamplerKey { lod_max_clamp: 4.0, ..SamplerKey::LINEAR_CLAMP }).unwrap();
        resources.sampler(&SamplerKey::NEAREST_CLAMP).unwrap();
        assert_eq!(resources.samplers_created(), 4);
        // Anisotropy with a nearest filter is invalid in WebGPU: an error value, and not cached.
        let invalid = SamplerKey { anisotropy_clamp: 4, ..SamplerKey::NEAREST_CLAMP };
        assert!(matches!(resources.sampler(&invalid), Err(GpuError::Validation { .. })));
        assert_eq!(resources.samplers_created(), 4);
    }
}
