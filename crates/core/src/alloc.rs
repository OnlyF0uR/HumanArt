//! Allocation tools for `[PERF-R1]`: an allocation counter, the per-frame arena and the slab.
//!
//! - [`CountingAllocator`] wraps the system allocator and counts, per thread, every allocation,
//!   reallocation and free. A test binary installs it with `#[global_allocator]`; [`count_in`] then
//!   reports what a closure did on the calling thread, which is how the zero-allocation tests of the
//!   render, UI, deformation and brush paths are written. Counting per thread keeps tests that run in
//!   parallel from disturbing each other.
//! - [`Arena`] is a bump allocator that is reset once per frame: after the first frames have sized
//!   it, a frame's temporary data costs no heap allocation at all `[AC-WP02-2]`.
//! - [`Slab`] stores values in reusable slots addressed by generational [`Handle`]s, so a stale handle
//!   is detected instead of reaching the slot's next occupant.
//!
//! `[UNS-01]` This is the only module of `humanart-core` that may contain `unsafe` (ADR-005): the
//! `GlobalAlloc` implementation and the arena need it. The slab is safe code.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::{Cell, RefCell};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::ArenaError;

// ---------------------------------------------------------------------------------------------------
// Counting allocator
// ---------------------------------------------------------------------------------------------------

/// What the heap did on one thread: totals since the thread started, or the difference between two
/// such totals (see [`count_in`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AllocStats {
    /// Calls to `alloc` and `alloc_zeroed`.
    pub allocations: u64,
    /// Calls to `realloc`.
    pub reallocations: u64,
    /// Calls to `dealloc`.
    pub frees: u64,
    /// Bytes requested by allocations, plus the new size of every reallocation.
    pub bytes_allocated: u64,
    /// Bytes returned by frees, plus the old size of every reallocation.
    pub bytes_freed: u64,
}

impl AllocStats {
    /// No heap activity.
    pub const ZERO: Self = Self { allocations: 0, reallocations: 0, frees: 0, bytes_allocated: 0, bytes_freed: 0 };

    /// The number of calls into the allocator of any kind. A steady-state path asserts this is 0.
    #[inline]
    pub const fn events(&self) -> u64 {
        self.allocations + self.reallocations + self.frees
    }

    /// `true` when the allocator was not called at all.
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.events() == 0
    }

    /// The activity between an `earlier` snapshot of the same thread and this one.
    #[must_use]
    pub const fn since(&self, earlier: &Self) -> Self {
        Self {
            allocations: self.allocations.wrapping_sub(earlier.allocations),
            reallocations: self.reallocations.wrapping_sub(earlier.reallocations),
            frees: self.frees.wrapping_sub(earlier.frees),
            bytes_allocated: self.bytes_allocated.wrapping_sub(earlier.bytes_allocated),
            bytes_freed: self.bytes_freed.wrapping_sub(earlier.bytes_freed),
        }
    }
}

thread_local! {
    // A const initialiser and no destructor: touching this from inside the allocator neither
    // allocates nor registers anything on targets with native thread-local storage (all the
    // platforms of `[PLAT-01]`; single-threaded wasm uses a plain static).
    static THREAD_STATS: Cell<AllocStats> = const { Cell::new(AllocStats::ZERO) };
}

/// Set by the first allocation that goes through [`CountingAllocator`].
static INSTALLED: AtomicBool = AtomicBool::new(false);

#[inline]
fn record(update: impl FnOnce(&mut AllocStats)) {
    // `try_with` because the allocator is still called while a thread's storage is torn down.
    let _ = THREAD_STATS.try_with(|cell| {
        let mut stats = cell.get();
        update(&mut stats);
        cell.set(stats);
    });
}

/// A global allocator that forwards to [`System`] and counts what each thread does.
///
/// Install it in a test binary (never in the shipped application, which keeps the plain system
/// allocator):
///
/// ```
/// use humanart_core::alloc::{CountingAllocator, count_in, is_installed};
///
/// #[global_allocator]
/// static ALLOCATOR: CountingAllocator = CountingAllocator;
///
/// let mut buffer = Vec::with_capacity(16);
/// let stats = count_in(|| buffer.push(1_u32));
/// assert!(is_installed());
/// assert_eq!(stats.events(), 0);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct CountingAllocator;

// SAFETY: every method forwards its arguments unchanged to `System`, so `System`'s guarantees
// (valid, aligned, non-overlapping blocks; no unwinding) are this allocator's. The bookkeeping in
// `record` only copies a `Cell` in thread-local storage and uses wrapping arithmetic: it cannot
// allocate, unwind or re-enter the allocator.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if !INSTALLED.load(Ordering::Relaxed) {
            INSTALLED.store(true, Ordering::Relaxed);
        }
        record(|s| {
            s.allocations = s.allocations.wrapping_add(1);
            s.bytes_allocated = s.bytes_allocated.wrapping_add(layout.size() as u64);
        });
        // SAFETY: the caller upholds the contract of `GlobalAlloc::alloc`, which is `System`'s.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if !INSTALLED.load(Ordering::Relaxed) {
            INSTALLED.store(true, Ordering::Relaxed);
        }
        record(|s| {
            s.allocations = s.allocations.wrapping_add(1);
            s.bytes_allocated = s.bytes_allocated.wrapping_add(layout.size() as u64);
        });
        // SAFETY: the caller upholds the contract of `GlobalAlloc::alloc_zeroed`, which is `System`'s.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        record(|s| {
            s.frees = s.frees.wrapping_add(1);
            s.bytes_freed = s.bytes_freed.wrapping_add(layout.size() as u64);
        });
        // SAFETY: `ptr` came from this allocator, that is from `System`, with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(|s| {
            s.reallocations = s.reallocations.wrapping_add(1);
            s.bytes_allocated = s.bytes_allocated.wrapping_add(new_size as u64);
            s.bytes_freed = s.bytes_freed.wrapping_add(layout.size() as u64);
        });
        // SAFETY: `ptr` came from `System` with this `layout`, and the caller guarantees `new_size`
        // is valid for the layout's alignment.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// `true` once any allocation went through [`CountingAllocator`], that is when the running binary
/// installed it. A zero-allocation test asserts this, so it cannot pass by not counting.
pub fn is_installed() -> bool {
    INSTALLED.load(Ordering::Relaxed)
}

/// The calling thread's totals since it started (all zero when the allocator is not installed).
pub fn thread_stats() -> AllocStats {
    THREAD_STATS.try_with(Cell::get).unwrap_or(AllocStats::ZERO)
}

/// Runs `f` and returns its value with the heap activity of the calling thread during the call.
///
/// Work `f` hands to other threads is not counted. Calls nest.
pub fn measure<R>(f: impl FnOnce() -> R) -> (R, AllocStats) {
    let before = thread_stats();
    let value = f();
    let after = thread_stats();
    (value, after.since(&before))
}

/// Runs `f` and returns the heap activity of the calling thread during the call `[PERF-R1]`.
///
/// The counts are all zero when [`CountingAllocator`] is not the global allocator of the running
/// binary: assert [`is_installed`] next to the count.
pub fn count_in(f: impl FnOnce()) -> AllocStats {
    measure(f).1
}

// ---------------------------------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------------------------------

/// The smallest chunk an arena asks the system for.
const MIN_CHUNK: usize = 4096;
/// The alignment of the chunks themselves; values with a larger alignment are placed by padding.
const CHUNK_ALIGN: usize = 16;

/// One block of memory owned by an [`Arena`].
struct Chunk {
    ptr: NonNull<u8>,
    size: usize,
}

impl Chunk {
    /// Allocates `size` bytes (`size` must not be 0).
    fn new(size: usize) -> Result<Self, ArenaError> {
        if size == 0 {
            return Err(ArenaError::TooLarge);
        }
        let layout = Layout::from_size_align(size, CHUNK_ALIGN).map_err(|_| ArenaError::TooLarge)?;
        // SAFETY: `layout` has a non-zero size, checked above.
        let ptr = unsafe { std::alloc::alloc(layout) };
        NonNull::new(ptr).map(|ptr| Self { ptr, size }).ok_or(ArenaError::OutOfMemory)
    }
}

impl Drop for Chunk {
    fn drop(&mut self) {
        // `new` validated exactly this size and alignment, so the layout is rebuilt without a check
        // failing; if it somehow did, leaking the block is the safe answer.
        if let Ok(layout) = Layout::from_size_align(self.size, CHUNK_ALIGN) {
            // SAFETY: `ptr` was returned by `std::alloc::alloc` with this layout in `Chunk::new` and
            // is freed only here, once.
            unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

/// A bump allocator for data that lives for one frame `[PERF-R1]`.
///
/// Values are placed one after another in a chunk of memory; [`Arena::reset`] forgets them all at
/// once. When a frame needs more than the current chunk holds, a larger chunk is added, and the next
/// `reset` replaces all chunks by one that is big enough for everything: after this warm-up a frame
/// allocates nothing from the heap `[AC-WP02-2]`.
///
/// Destructors never run for values in an arena, so only types without drop glue are accepted (a
/// compile-time check). References handed out borrow the arena, which is why `reset` takes
/// `&mut self`: the borrow checker proves nothing from the last frame is still in use.
///
/// An arena belongs to one thread at a time (it is `Send`, not `Sync`).
///
/// ```
/// use humanart_core::alloc::Arena;
///
/// let mut arena = Arena::with_capacity(1024)?;
/// let squares = arena.alloc_slice_with(4, |i| (i * i) as u32)?;
/// let label = arena.alloc_str("frame 1")?;
/// assert_eq!((&*squares, &*label), (&[0, 1, 4, 9][..], "frame 1"));
/// arena.reset();
/// assert_eq!(arena.used(), 0);
/// # Ok::<(), humanart_core::error::ArenaError>(())
/// ```
pub struct Arena {
    /// Start of the current chunk (dangling while there is none).
    base: Cell<NonNull<u8>>,
    /// Size of the current chunk in bytes (0 while there is none).
    cap: Cell<usize>,
    /// Bytes of the current chunk already handed out.
    used: Cell<usize>,
    /// Bytes handed out from the chunks before the current one.
    used_before: Cell<usize>,
    /// The most bytes any frame used, padding included.
    high_water: Cell<usize>,
    /// Every chunk; the last one is the current one.
    chunks: RefCell<Vec<Chunk>>,
}

// SAFETY: an arena owns its chunks outright. Moving it to another thread requires that no reference
// into it is alive (they all borrow the arena), and the values left inside are never read or dropped
// again, so no `!Send` value is ever touched from the new thread.
unsafe impl Send for Arena {}

impl Arena {
    /// An arena that owns no memory yet; the first allocation creates a chunk.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base: Cell::new(NonNull::dangling()),
            cap: Cell::new(0),
            used: Cell::new(0),
            used_before: Cell::new(0),
            high_water: Cell::new(0),
            chunks: RefCell::new(Vec::new()),
        }
    }

    /// An arena with one chunk of `bytes` bytes, so frames that fit never touch the heap.
    pub fn with_capacity(bytes: usize) -> Result<Self, ArenaError> {
        let arena = Self::new();
        if bytes > 0 {
            arena.add_chunk(bytes)?;
        }
        Ok(arena)
    }

    /// Bytes handed out since the last reset, alignment padding included.
    pub fn used(&self) -> usize {
        self.used_before.get() + self.used.get()
    }

    /// Bytes of memory the arena owns.
    pub fn capacity(&self) -> usize {
        self.chunks.borrow().iter().map(|chunk| chunk.size).sum()
    }

    /// The number of chunks the arena owns; 1 (or 0 for an unused arena) in steady state.
    pub fn chunk_count(&self) -> usize {
        self.chunks.borrow().len()
    }

    /// The largest [`Arena::used`] any frame reached, for sizing and for the profiler overlay.
    pub fn high_water(&self) -> usize {
        self.high_water.get().max(self.used())
    }

    /// Forgets every value and makes the whole capacity available again.
    ///
    /// If the frame overflowed into extra chunks they are replaced by a single chunk large enough
    /// for all of them, which is the only time `reset` calls the system allocator.
    pub fn reset(&mut self) {
        self.high_water.set(self.high_water());
        let chunks = self.chunks.get_mut();
        if chunks.len() > 1 {
            let total = chunks.iter().fold(0_usize, |sum, chunk| sum.saturating_add(chunk.size));
            let wanted = total.checked_next_power_of_two().unwrap_or(total);
            // Free the old chunks first so the peak is one frame's worth of memory, not two.
            let last = chunks.pop();
            chunks.clear();
            match Chunk::new(wanted) {
                Ok(chunk) => {
                    drop(last);
                    chunks.push(chunk);
                }
                // No memory for the merged chunk: keep the newest (largest) one and grow again
                // next frame if needed.
                Err(_) => chunks.extend(last),
            }
            let (base, cap) = chunks.last().map_or((NonNull::dangling(), 0), |chunk| (chunk.ptr, chunk.size));
            self.base.set(base);
            self.cap.set(cap);
        }
        self.used.set(0);
        self.used_before.set(0);
    }

    /// Moves `value` into the arena.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc<T>(&self, value: T) -> Result<&mut T, ArenaError> {
        const { assert!(!std::mem::needs_drop::<T>(), "arena values are never dropped") };
        let ptr = self.reserve::<T>(1)?;
        // SAFETY: `reserve` returned memory that is valid for writes of one `T`, aligned for `T`,
        // and handed out to nobody else (the bump offset is already past it). The chunk it lies in
        // is freed only by `reset` or drop, which need `&mut self` and so end the returned borrow.
        unsafe {
            ptr.as_ptr().write(value);
            Ok(&mut *ptr.as_ptr())
        }
    }

    /// A slice of `len` elements, element `i` being `fill(i)`.
    ///
    /// `fill` may itself allocate from the arena.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_slice_with<T>(&self, len: usize, mut fill: impl FnMut(usize) -> T) -> Result<&mut [T], ArenaError> {
        const { assert!(!std::mem::needs_drop::<T>(), "arena values are never dropped") };
        let ptr = self.reserve::<T>(len)?;
        for i in 0..len {
            let value = fill(i);
            // SAFETY: `i < len`, so the element lies inside the block `reserve` returned for `len`
            // values of `T`; the block was reserved before `fill` ran, so nested allocations made by
            // `fill` cannot overlap it.
            unsafe { ptr.as_ptr().add(i).write(value) };
        }
        // SAFETY: all `len` elements were initialised by the loop above; the block is exclusively
        // ours and lives until `reset` or drop, as in `alloc`. If `fill` panicked we never get here
        // and the partially written block is simply abandoned (no drop glue, checked above).
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), len) })
    }

    /// A slice of `len` copies of `value`.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_slice_fill<T: Copy>(&self, len: usize, value: T) -> Result<&mut [T], ArenaError> {
        self.alloc_slice_with(len, |_| value)
    }

    /// A copy of `source`.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_slice_copy<T: Copy>(&self, source: &[T]) -> Result<&mut [T], ArenaError> {
        let ptr = self.reserve::<T>(source.len())?;
        // SAFETY: the destination is a fresh block of `source.len()` values of `T`, valid for writes
        // and aligned. It cannot overlap `source`: that is a live shared borrow, so it lies either
        // outside the arena or in a block handed out earlier. Afterwards every element is
        // initialised (`T: Copy`), and the block is exclusively ours until `reset` or drop.
        unsafe {
            std::ptr::copy_nonoverlapping(source.as_ptr(), ptr.as_ptr(), source.len());
            Ok(std::slice::from_raw_parts_mut(ptr.as_ptr(), source.len()))
        }
    }

    /// A copy of `text`.
    #[allow(clippy::mut_from_ref)]
    pub fn alloc_str(&self, text: &str) -> Result<&mut str, ArenaError> {
        let bytes = self.alloc_slice_copy(text.as_bytes())?;
        // SAFETY: `bytes` is a byte-for-byte copy of a `str`, hence valid UTF-8.
        Ok(unsafe { std::str::from_utf8_unchecked_mut(bytes) })
    }

    /// Reserves room for `len` values of `T` and returns the (uninitialised) start of the block.
    ///
    /// For a zero-sized request this is a dangling, aligned pointer, which is all such a value needs.
    fn reserve<T>(&self, len: usize) -> Result<NonNull<T>, ArenaError> {
        let layout = Layout::array::<T>(len).map_err(|_| ArenaError::TooLarge)?;
        if layout.size() == 0 {
            return Ok(NonNull::dangling());
        }
        match self.bump(layout) {
            Some(ptr) => Ok(ptr.cast()),
            None => self.grow(layout).map(NonNull::cast),
        }
    }

    /// The fast path: carves `layout` out of the current chunk, or returns `None` if it does not fit.
    #[inline]
    fn bump(&self, layout: Layout) -> Option<NonNull<u8>> {
        let base = self.base.get();
        let base_addr = base.as_ptr().addr();
        let aligned = base_addr.checked_add(self.used.get())?.checked_next_multiple_of(layout.align())?;
        let offset = aligned - base_addr;
        let end = offset.checked_add(layout.size())?;
        if end > self.cap.get() {
            return None;
        }
        self.used.set(end);
        // SAFETY: `offset <= end <= cap`, so the result is inside the chunk allocation `base` points
        // to (with `cap > 0` here, since `layout.size() > 0` is guaranteed by `reserve`). The chunk
        // was allocated with `cap <= isize::MAX` bytes, so the offset cannot overflow.
        Some(unsafe { base.add(offset) })
    }

    /// The slow path: adds a chunk that certainly fits `layout`, then carves from it.
    #[cold]
    fn grow(&self, layout: Layout) -> Result<NonNull<u8>, ArenaError> {
        // Worst-case padding is `align - 1` bytes, whatever address the chunk gets.
        let needed = layout.size().checked_add(layout.align()).ok_or(ArenaError::TooLarge)?;
        let size = needed.max(self.cap.get().saturating_mul(2)).max(MIN_CHUNK);
        // Doubling may ask for more than the address space allows while `needed` alone would fit.
        if self.add_chunk(size).is_err() {
            self.add_chunk(needed)?;
        }
        self.bump(layout).ok_or(ArenaError::TooLarge)
    }

    /// Makes a new chunk of `size` bytes the current one; earlier chunks stay alive until `reset`.
    fn add_chunk(&self, size: usize) -> Result<(), ArenaError> {
        let chunk = Chunk::new(size)?;
        self.base.set(chunk.ptr);
        self.cap.set(chunk.size);
        self.used_before.set(self.used_before.get() + self.used.get());
        self.used.set(0);
        // No user code runs while the list is borrowed, so the borrow cannot be contended.
        self.chunks.borrow_mut().push(chunk);
        Ok(())
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Arena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arena")
            .field("used", &self.used())
            .field("capacity", &self.capacity())
            .field("chunks", &self.chunk_count())
            .field("high_water", &self.high_water())
            .finish()
    }
}

// ---------------------------------------------------------------------------------------------------
// Slab
// ---------------------------------------------------------------------------------------------------

/// A reference to a value in a [`Slab<T>`]: a slot index and the generation the slot had when the
/// value was inserted.
///
/// A handle is 8 bytes, `Copy`, and does not keep the value alive: once the value is removed, the
/// slab answers `None` for the handle, also after the slot has been reused.
pub struct Handle<T> {
    index: u32,
    generation: u32,
    marker: PhantomData<fn() -> T>,
}

impl<T> Handle<T> {
    /// A handle that never refers to a value (its bits are all zero).
    pub const NULL: Self = Self { index: 0, generation: 0, marker: PhantomData };

    /// The slot index; dense and small, usable as a key into side tables.
    #[inline]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// The generation of the slot this handle belongs to (0 only for [`Handle::NULL`]).
    #[inline]
    pub const fn generation(self) -> u32 {
        self.generation
    }

    /// `true` for [`Handle::NULL`].
    #[inline]
    pub const fn is_null(self) -> bool {
        self.generation == 0
    }

    /// The handle as one integer (generation in the high half), for storage in plain-data records.
    #[inline]
    pub const fn to_bits(self) -> u64 {
        ((self.generation as u64) << 32) | self.index as u64
    }

    /// The handle [`Handle::to_bits`] encoded. Any value is accepted: a made-up handle is simply
    /// not found by the slab.
    #[inline]
    pub const fn from_bits(bits: u64) -> Self {
        Self { index: bits as u32, generation: (bits >> 32) as u32, marker: PhantomData }
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}

impl<T> Eq for Handle<T> {}

impl<T> Hash for Handle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.to_bits());
    }
}

impl<T> PartialOrd for Handle<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Handle<T> {
    /// Orders by slot index, then generation.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.index, self.generation).cmp(&(other.index, other.generation))
    }
}

impl<T> Default for Handle<T> {
    fn default() -> Self {
        Self::NULL
    }
}

impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Handle({}v{})", self.index, self.generation)
    }
}

/// Marks the end of the free list.
const NO_SLOT: u32 = u32::MAX;

enum Entry<T> {
    Occupied(T),
    /// A free slot and the index of the next free slot ([`NO_SLOT`] at the end of the list).
    Free(u32),
}

struct Slot<T> {
    /// The generation of the current (or, for a free slot, the next) occupant. 0 marks a slot whose
    /// generations are used up: it is never handed out again.
    generation: u32,
    entry: Entry<T>,
}

/// A pool of values in reusable slots, addressed by generational [`Handle`]s.
///
/// Insertion, lookup and removal are O(1). Removing a value frees its slot for the next insertion
/// and bumps the slot's generation, so handles to the removed value stop resolving. Memory grows
/// only when every slot is occupied: a slab that has reached its working size does not allocate
/// `[PERF-R1]`.
pub struct Slab<T> {
    slots: Vec<Slot<T>>,
    free_head: u32,
    len: u32,
}

impl<T> Slab<T> {
    /// The most slots a slab can have.
    pub const MAX_SLOTS: usize = (u32::MAX - 1) as usize;

    /// An empty slab that owns no memory.
    #[must_use]
    pub const fn new() -> Self {
        Self { slots: Vec::new(), free_head: NO_SLOT, len: 0 }
    }

    /// An empty slab with room for `capacity` values before it allocates again.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self { slots: Vec::with_capacity(capacity.min(Self::MAX_SLOTS)), free_head: NO_SLOT, len: 0 }
    }

    /// The number of values stored.
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// `true` when no value is stored.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of values the slab can hold without allocating (slots of exhausted generations,
    /// which only arise after 4 billion reuses of one slot, are not subtracted).
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// Makes room for `additional` more values.
    pub fn reserve(&mut self, additional: usize) {
        let free = self.slots.len() - self.len();
        self.slots.reserve(additional.saturating_sub(free));
    }

    /// Stores `value` and returns its handle, or gives `value` back when the slab already has
    /// [`Slab::MAX_SLOTS`] slots and none is free.
    pub fn try_insert(&mut self, value: T) -> Result<Handle<T>, T> {
        if let Some(slot) = self.slots.get_mut(self.free_head as usize)
            && let Entry::Free(next) = slot.entry
        {
            let handle = Handle { index: self.free_head, generation: slot.generation, marker: PhantomData };
            slot.entry = Entry::Occupied(value);
            self.free_head = next;
            self.len += 1;
            return Ok(handle);
        }
        if self.slots.len() >= Self::MAX_SLOTS {
            return Err(value);
        }
        let index = self.slots.len() as u32;
        self.slots.push(Slot { generation: 1, entry: Entry::Occupied(value) });
        self.len += 1;
        Ok(Handle { index, generation: 1, marker: PhantomData })
    }

    /// Stores `value` and returns its handle.
    ///
    /// # Panics
    ///
    /// Panics on capacity overflow, exactly as `Vec::push` does: when the slab already has
    /// [`Slab::MAX_SLOTS`] (4 294 967 294) slots and none is free. Use [`Slab::try_insert`] where
    /// that count can be reached by outside input.
    pub fn insert(&mut self, value: T) -> Handle<T> {
        let Ok(handle) = self.try_insert(value) else { panic!("slab capacity overflow") };
        handle
    }

    /// `true` when `handle` refers to a value that is still stored.
    pub fn contains(&self, handle: Handle<T>) -> bool {
        self.get(handle).is_some()
    }

    /// The value `handle` refers to, or `None` when it was removed (or the handle is not from this
    /// slab's history at all).
    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        match self.slots.get(handle.index as usize) {
            Some(Slot { generation, entry: Entry::Occupied(value) }) if *generation == handle.generation => Some(value),
            _ => None,
        }
    }

    /// Mutable access to the value `handle` refers to.
    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        match self.slots.get_mut(handle.index as usize) {
            Some(Slot { generation, entry: Entry::Occupied(value) }) if *generation == handle.generation => Some(value),
            _ => None,
        }
    }

    /// Removes and returns the value `handle` refers to; `None` when it is already gone.
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        if slot.generation != handle.generation || !matches!(slot.entry, Entry::Occupied(_)) {
            return None;
        }
        let entry = std::mem::replace(&mut slot.entry, Entry::Free(NO_SLOT));
        slot.generation = slot.generation.wrapping_add(1);
        if slot.generation != 0 {
            // A slot whose generation wrapped to 0 stays out of the free list for good, so an old
            // handle can never match a new occupant.
            slot.entry = Entry::Free(self.free_head);
            self.free_head = handle.index;
        }
        self.len -= 1;
        match entry {
            Entry::Occupied(value) => Some(value),
            Entry::Free(_) => None,
        }
    }

    /// Removes every value. All handles stop resolving; the memory is kept.
    pub fn clear(&mut self) {
        self.free_head = NO_SLOT;
        for (index, slot) in self.slots.iter_mut().enumerate().rev() {
            if matches!(slot.entry, Entry::Occupied(_)) {
                slot.generation = slot.generation.wrapping_add(1);
            }
            if slot.generation == 0 {
                slot.entry = Entry::Free(NO_SLOT);
            } else {
                slot.entry = Entry::Free(self.free_head);
                self.free_head = index as u32;
            }
        }
        self.len = 0;
    }

    /// The stored values with their handles, in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| match &slot.entry {
            Entry::Occupied(value) => {
                Some((Handle { index: index as u32, generation: slot.generation, marker: PhantomData }, value))
            }
            Entry::Free(_) => None,
        })
    }

    /// The stored values, mutably, with their handles, in slot order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Handle<T>, &mut T)> {
        self.slots.iter_mut().enumerate().filter_map(|(index, slot)| match &mut slot.entry {
            Entry::Occupied(value) => {
                Some((Handle { index: index as u32, generation: slot.generation, marker: PhantomData }, value))
            }
            Entry::Free(_) => None,
        })
    }
}

impl<T> Default for Slab<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug> fmt::Debug for Slab<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use std::collections::HashMap;

    // [PERF-R1] The counter sees exactly what a thread does.
    #[test]
    fn counting_allocator_counts_allocations_reallocations_and_frees() {
        let (vec, stats) = measure(|| {
            let mut v: Vec<u64> = Vec::with_capacity(4);
            v.extend([1, 2, 3, 4]);
            v.reserve_exact(60);
            v
        });
        assert!(is_installed());
        assert_eq!((stats.allocations, stats.reallocations, stats.frees), (1, 1, 0), "{stats:?}");
        assert_eq!(stats.bytes_allocated, 32 + 64 * 8);
        assert_eq!(stats.bytes_freed, 32);
        assert_eq!(stats.events(), 2);
        assert!(!stats.is_zero());
        let freed = count_in(|| drop(vec));
        assert_eq!((freed.allocations, freed.frees, freed.bytes_freed), (0, 1, 64 * 8));

        let zeroed = count_in(|| drop(vec![0_u8; 100]));
        assert_eq!((zeroed.allocations, zeroed.frees, zeroed.bytes_allocated), (1, 1, 100));

        let nothing = count_in(|| {
            let mut sum = 0_u64;
            for i in 0..100 {
                sum += std::hint::black_box(i);
            }
            std::hint::black_box(sum);
        });
        assert!(nothing.is_zero() && nothing == AllocStats::ZERO && nothing == AllocStats::default());
    }

    // [PERF-R1] Parallel tests do not disturb each other: another thread's allocations are not ours.
    #[test]
    fn counting_is_per_thread_and_nests() {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let (done_tx, done_rx) = std::sync::mpsc::channel::<AllocStats>();
        let noisy = std::thread::spawn(move || {
            let stats = count_in(|| {
                for i in 0..2000_usize {
                    std::hint::black_box(vec![0_u8; 1 + i % 64]);
                }
            });
            done_tx.send(stats).expect("send");
            rx.recv().ok();
        });
        let outer = count_in(|| {
            let other = done_rx.recv().expect("stats of the other thread");
            assert_eq!((other.allocations, other.frees), (2000, 2000));
            let inner = count_in(|| drop(std::hint::black_box(Box::new(7_u32))));
            assert_eq!((inner.allocations, inner.frees, inner.bytes_allocated), (1, 1, 4));
        });
        // The outer count includes the inner one, and nothing of the 2000 on the other thread. (The
        // channel receive itself may allocate a little on this thread; it stays far below 2000.)
        assert!(outer.allocations >= 1 && outer.allocations < 100, "{outer:?}");
        tx.send(()).expect("release the other thread");
        noisy.join().expect("join");
        let before = thread_stats();
        let later = thread_stats();
        assert_eq!(later.since(&before), AllocStats::ZERO);
    }

    // [AC-WP02-2] [PERF-R1] The arena allocates while it warms up and never afterwards.
    #[test]
    fn arena_steady_state_does_not_touch_the_heap() {
        let mut arena = Arena::new();
        assert_eq!((arena.capacity(), arena.chunk_count(), arena.used()), (0, 0, 0));
        let frame = |arena: &Arena, frame: usize| -> u64 {
            let mut sum = 0_u64;
            for i in 0..200_usize {
                let xs = arena.alloc_slice_fill(64 + (i + frame) % 32, i as u32).expect("slice");
                let p = arena.alloc([i as f32; 4]).expect("value");
                let name = arena.alloc_str("bone.spine.03").expect("str");
                let ys = arena.alloc_slice_copy(&xs[..8]).expect("copy");
                sum += u64::from(xs[0]) + p[3] as u64 + name.len() as u64 + u64::from(ys[7]);
            }
            sum
        };
        // Warm-up: the first frame grows through several chunks, the reset merges them.
        let warm = count_in(|| {
            std::hint::black_box(frame(&arena, 0));
        });
        assert!(warm.allocations > 1, "{warm:?}");
        assert!(arena.chunk_count() > 1);
        arena.reset();
        assert_eq!((arena.chunk_count(), arena.used()), (1, 0));
        let capacity = arena.capacity();
        let high = arena.high_water();
        assert!(high > 200 * (64 * 4 + 16 + 13 + 32) && capacity >= high);

        let steady = count_in(|| {
            for f in 0..600 {
                std::hint::black_box(frame(&arena, f));
                arena.reset();
            }
        });
        assert!(is_installed());
        assert_eq!(steady.events(), 0, "{steady:?}");
        assert_eq!((arena.capacity(), arena.chunk_count()), (capacity, 1));

        // A pre-sized arena needs no warm-up at all.
        let (mut sized, creation) = measure(|| Arena::with_capacity(1 << 20).expect("arena"));
        assert_eq!(creation.allocations, 2, "one chunk and the chunk list: {creation:?}");
        let first = count_in(|| {
            std::hint::black_box(frame(&sized, 3));
            sized.reset();
        });
        assert_eq!(first.events(), 0, "{first:?}");
    }

    // [UNS-02] Boundary: alignment, including alignments above the chunk's own.
    #[test]
    fn arena_respects_alignment_and_never_overlaps() {
        #[derive(Clone, Copy)]
        #[repr(align(64))]
        struct Line([u8; 64]);
        #[derive(Clone, Copy)]
        #[repr(align(4096))]
        struct Page(u8);

        let mut arena = Arena::with_capacity(256).expect("arena");
        let mut rng = Rng::new(11);
        for round in 0..20 {
            let mut blocks: Vec<(usize, usize)> = Vec::new();
            let mut note = |addr: usize, size: usize, align: usize| {
                assert_eq!(addr % align, 0, "misaligned for {align}");
                blocks.push((addr, addr + size));
            };
            for _ in 0..200 {
                match rng.index(6) {
                    0 => {
                        let v = arena.alloc(rng.next_u32() as u8).expect("u8");
                        note(std::ptr::from_mut(v).addr(), 1, 1);
                    }
                    1 => {
                        let v = arena.alloc(rng.next_u64()).expect("u64");
                        note(std::ptr::from_mut(v).addr(), 8, 8);
                    }
                    2 => {
                        let v = arena.alloc(u128::from(rng.next_u64())).expect("u128");
                        note(std::ptr::from_mut(v).addr(), 16, align_of::<u128>());
                    }
                    3 => {
                        let v = arena.alloc(Line([round as u8; 64])).expect("line");
                        assert_eq!(v.0[63], round as u8);
                        note(std::ptr::from_mut(v).addr(), 64, 64);
                    }
                    4 => {
                        let v = arena.alloc(Page(3)).expect("page");
                        assert_eq!(v.0, 3);
                        note(std::ptr::from_mut(v).addr(), 4096, 4096);
                    }
                    _ => {
                        let len = rng.index(40);
                        let v = arena.alloc_slice_fill(len, 0xABCD_u16).expect("u16s");
                        assert!(v.iter().all(|&x| x == 0xABCD));
                        note(v.as_ptr().addr(), len * 2, 2);
                    }
                }
            }
            blocks.sort_unstable();
            for pair in blocks.windows(2) {
                assert!(pair[0].1 <= pair[1].0, "blocks overlap: {pair:?}");
            }
            let used: usize = blocks.iter().map(|b| b.1 - b.0).sum();
            assert!(arena.used() >= used);
            arena.reset();
        }
        assert_eq!(arena.chunk_count(), 1);
    }

    // [UNS-02] Boundary: values written through one reference are not disturbed by later ones.
    #[test]
    fn arena_values_keep_their_contents_across_growth() {
        let arena = Arena::new();
        let mut kept: Vec<(&mut [u32], u32)> = Vec::new();
        for i in 0..500_u32 {
            let len = 1 + (i as usize * 7) % 300;
            let slice = arena.alloc_slice_with(len, |j| i.wrapping_mul(31).wrapping_add(j as u32)).expect("slice");
            kept.push((slice, i));
        }
        assert!(arena.chunk_count() > 2, "the test must cross chunk boundaries");
        for (slice, i) in &mut kept {
            for (j, value) in slice.iter_mut().enumerate() {
                assert_eq!(*value, i.wrapping_mul(31).wrapping_add(j as u32));
                *value = !*value;
            }
        }
        for (slice, i) in &kept {
            assert!(slice.iter().enumerate().all(|(j, &v)| v == !(i.wrapping_mul(31).wrapping_add(j as u32))));
        }
        assert_eq!(arena.capacity(), arena.chunks.borrow().iter().map(|c| c.size).sum::<usize>());
        assert!(arena.used() <= arena.capacity());
    }

    // [UNS-02] Boundary: zero-sized values, empty slices, exact fits and nested allocation.
    #[test]
    fn arena_handles_zero_sizes_exact_fits_and_reentrant_fill() {
        let mut arena = Arena::with_capacity(MIN_CHUNK).expect("arena");
        let unit = arena.alloc(()).expect("zst");
        assert_eq!(*unit, ());
        let none: &mut [u64] = arena.alloc_slice_fill(0, 0).expect("empty");
        assert!(none.is_empty());
        let zsts = arena.alloc_slice_with(1000, |_| ()).expect("zst slice");
        assert_eq!(zsts.len(), 1000);
        assert_eq!(arena.alloc_str("").expect("empty str"), "");
        assert_eq!(arena.used(), 0, "zero-sized requests take no memory");

        // Exactly the whole chunk, then one more byte.
        let all = arena.alloc_slice_fill(MIN_CHUNK, 0x5A_u8).expect("whole chunk");
        assert_eq!((all.len(), arena.used(), arena.chunk_count()), (MIN_CHUNK, MIN_CHUNK, 1));
        let one = arena.alloc(1_u8).expect("one more");
        assert_eq!((*one, arena.chunk_count()), (1, 2));
        assert!(all.iter().all(|&b| b == 0x5A));
        arena.reset();
        assert_eq!((arena.chunk_count(), arena.used()), (1, 0));
        assert!(arena.capacity() >= MIN_CHUNK * 3);

        // `fill` allocates from the same arena while its own block is being written.
        let outer = arena
            .alloc_slice_with(50, |i| {
                let inner = arena.alloc_slice_fill(i + 1, i as u64).expect("inner");
                inner.iter().sum::<u64>()
            })
            .expect("outer");
        for (i, value) in outer.iter().enumerate() {
            assert_eq!(*value, (i as u64) * (i as u64 + 1));
        }
        let text = arena.alloc_str("grüße 𝄞").expect("str");
        text.make_ascii_uppercase();
        assert_eq!(text, "GRüßE 𝄞");
        assert!(format!("{arena:?}").starts_with("Arena { used: "));
    }

    // [ERR-02] Absurd sizes are an error, not a panic or an abort.
    #[test]
    fn arena_rejects_sizes_beyond_the_address_space() {
        let arena = Arena::default();
        assert_eq!(arena.alloc_slice_fill(usize::MAX, 0_u8).err(), Some(ArenaError::TooLarge));
        assert_eq!(arena.alloc_slice_fill(usize::MAX / 2, 0_u32).err(), Some(ArenaError::TooLarge));
        assert_eq!(arena.alloc_slice_fill(isize::MAX as usize, 0_u8).err(), Some(ArenaError::TooLarge));
        assert_eq!(Arena::with_capacity(usize::MAX).err(), Some(ArenaError::TooLarge));
        assert_eq!(Chunk::new(0).err(), Some(ArenaError::TooLarge));
        // The arena is still usable afterwards.
        assert_eq!(*arena.alloc(5_i32).expect("value"), 5);
        assert_eq!(arena.chunk_count(), 1);
    }

    // [UNS-02] Boundary of `unsafe impl Send`: an arena moves to another thread and back.
    #[test]
    fn arena_moves_between_threads() {
        let mut arena = Arena::with_capacity(64).expect("arena");
        assert_eq!(arena.alloc_slice_fill(10, 1_u8).expect("bytes").len(), 10);
        arena.reset();
        let mut arena = std::thread::spawn(move || {
            let sum: u32 = arena.alloc_slice_with(100, |i| i as u32).expect("slice").iter().sum();
            assert_eq!(sum, 4950);
            arena
        })
        .join()
        .expect("join");
        assert_eq!(arena.used(), 400);
        arena.reset();
        assert_eq!(arena.high_water(), 400);
    }

    #[test]
    fn slab_insert_get_remove_and_stale_handles() {
        let mut slab = Slab::<String>::new();
        assert!(slab.is_empty() && slab.capacity() == 0);
        let a = slab.insert("a".to_owned());
        let b = slab.insert("b".to_owned());
        assert_eq!((a.index(), a.generation(), b.index(), slab.len()), (0, 1, 1, 2));
        assert_eq!(slab.get(a).map(String::as_str), Some("a"));
        slab.get_mut(b).expect("b").push('!');
        assert_eq!(slab.remove(a).as_deref(), Some("a"));
        assert_eq!(slab.remove(a), None);
        assert!(!slab.contains(a) && slab.contains(b));
        assert_eq!(slab.get(a), None);
        assert_eq!(slab.get_mut(a), None);

        // The slot is reused under a new generation: the old handle stays dead.
        let c = slab.insert("c".to_owned());
        assert_eq!((c.index(), c.generation()), (0, 2));
        assert_ne!(a, c);
        assert_eq!(slab.get(a), None);
        assert_eq!(slab.get(c).map(String::as_str), Some("c"));
        assert_eq!(slab.iter().map(|(h, v)| (h, v.as_str())).collect::<Vec<_>>(), [(c, "c"), (b, "b!")]);
        for (_, value) in slab.iter_mut() {
            value.push('?');
        }
        assert_eq!(format!("{slab:?}"), r#"{Handle(0v2): "c?", Handle(1v1): "b!?"}"#);

        slab.clear();
        assert!(slab.is_empty() && !slab.contains(b) && !slab.contains(c));
        let d = slab.insert("d".to_owned());
        assert_eq!((d.index(), d.generation()), (0, 3));
        assert_eq!(slab.try_insert("e".to_owned()).map(Handle::index), Ok(1));
    }

    #[test]
    fn handle_bits_null_and_ordering() {
        let h = Handle::<u8> { index: 7, generation: 3, marker: PhantomData };
        assert_eq!(h.to_bits(), (3 << 32) | 7);
        assert_eq!(Handle::<u8>::from_bits(h.to_bits()), h);
        assert!(Handle::<u8>::NULL.is_null() && !h.is_null());
        assert_eq!(Handle::<u8>::NULL.to_bits(), 0);
        assert_eq!(Handle::<u8>::default(), Handle::NULL);
        let mut slab = Slab::<u8>::with_capacity(4);
        let real = slab.insert(1);
        assert!(!slab.contains(Handle::NULL));
        assert!(!slab.contains(Handle::from_bits(u64::MAX)));
        assert!(!slab.contains(Handle::from_bits(real.to_bits() + 1)));
        assert_eq!(slab.remove(Handle::NULL), None);
        let other = Handle::<u8>::from_bits(8);
        assert!(h < other && h.cmp(&h) == std::cmp::Ordering::Equal);
        assert_eq!(h.partial_cmp(&other), Some(std::cmp::Ordering::Less));
        let set: std::collections::HashSet<Handle<u8>> = [h, h, other].into();
        assert_eq!(set.len(), 2);
        // A handle is plain data whatever `T` is.
        let copy = Handle::<String>::NULL;
        assert_eq!((copy, copy), (Handle::NULL, Handle::NULL));
        assert_eq!(size_of::<Handle<String>>(), 8);
    }

    #[test]
    fn slab_matches_a_model_under_random_operations() {
        let mut rng = Rng::new(2024);
        let mut slab = Slab::<u64>::new();
        let mut model: HashMap<Handle<u64>, u64> = HashMap::new();
        let mut dead: Vec<Handle<u64>> = Vec::new();
        let mut live: Vec<Handle<u64>> = Vec::new();
        for step in 0..20_000_u64 {
            match rng.index(10) {
                0..=4 => {
                    let h = slab.insert(step);
                    assert!(model.insert(h, step).is_none(), "handle {h:?} handed out twice");
                    live.push(h);
                }
                5..=7 if !live.is_empty() => {
                    let h = live.swap_remove(rng.index(live.len()));
                    assert_eq!(slab.remove(h), model.remove(&h));
                    dead.push(h);
                }
                8 if !live.is_empty() => {
                    let h = live[rng.index(live.len())];
                    *slab.get_mut(h).expect("live") += 1;
                    *model.get_mut(&h).expect("live") += 1;
                }
                _ if step % 4001 == 0 => {
                    slab.clear();
                    dead.append(&mut live);
                    model.clear();
                }
                _ => {}
            }
            assert_eq!(slab.len(), model.len());
            if let Some(&h) = dead.get(rng.index(dead.len().max(1))) {
                assert_eq!(slab.get(h), None, "stale handle {h:?} resolved");
            }
        }
        assert_eq!(slab.iter().count(), model.len());
        for (h, value) in slab.iter() {
            assert_eq!(model.get(&h), Some(value));
        }
        for h in &dead {
            assert!(!slab.contains(*h));
        }
        // Slots are reused: far fewer slots than insertions.
        assert!(slab.slots.len() < 5000, "{}", slab.slots.len());
    }

    // [PERF-R1] A slab at its working size does not allocate.
    #[test]
    fn slab_steady_state_does_not_allocate() {
        let mut slab = Slab::<[u32; 4]>::with_capacity(64);
        let mut handles = [Handle::NULL; 64];
        let stats = count_in(|| {
            for round in 0..600_u32 {
                for (i, slot) in handles.iter_mut().enumerate() {
                    *slot = slab.insert([round, i as u32, 0, 0]);
                }
                for (i, handle) in handles.iter().enumerate() {
                    assert_eq!(slab.remove(*handle), Some([round, i as u32, 0, 0]));
                }
            }
        });
        assert!(is_installed());
        assert_eq!(stats.events(), 0, "{stats:?}");
        assert_eq!(slab.capacity(), 64);
        slab.reserve(64);
        assert_eq!(slab.capacity(), 64, "free slots count as room");
        for _ in 0..64 {
            slab.insert([0; 4]);
        }
        slab.reserve(10);
        assert!(slab.capacity() >= 74);
    }

    // A slot whose generation counter is used up is retired instead of wrapping onto old handles.
    #[test]
    fn slab_retires_a_slot_when_its_generations_run_out() {
        let mut slab = Slab::<u8>::new();
        let first = slab.insert(1);
        slab.slots[0].generation = u32::MAX;
        let last = Handle::<u8> { index: 0, generation: u32::MAX, marker: PhantomData };
        assert!(!slab.contains(first));
        assert_eq!(slab.remove(last), Some(1));
        assert_eq!(slab.len(), 0);
        let next = slab.insert(2);
        assert_eq!((next.index(), next.generation()), (1, 1), "slot 0 is never handed out again");
        assert!(!slab.contains(Handle::from_bits(0)) && !slab.contains(last));

        slab.slots[1].generation = u32::MAX;
        slab.clear();
        let after = slab.insert(3);
        assert_eq!((after.index(), after.generation(), slab.len()), (2, 1, 1));
        slab.clear();
        assert_eq!(slab.insert(4).index(), 2);
    }
}
