//! Platform traits, error base, logging, arenas and slabs, the job system, time, PRNG and hashing.
//!
//! `humanart-core` is the bottom of the dependency order: every other crate uses it and it uses nothing but
//! `std`. It compiles for `wasm32-unknown-unknown` `[PLAT-03]`: files, clocks, threads and randomness
//! reach the library crates only through the traits in [`platform`].
//!
//! Module map:
//!
//! - [`alloc`]: `CountingAllocator` + `count_in` (zero-allocation tests `[PERF-R1]`), the per-frame
//!   bump `Arena`, and `Slab<T>` with generational `Handle<T>`.
//! - [`collections`]: fixed-capacity `ArrayVec`, `ArrayString`, `RingBuffer`, and the growable `BitSet`.
//! - [`error`]: the crate's error enums (`PlatformError`, `JobError`, `LogError`, `HexError`, `CoreError`).
//! - [`executor`]: `block_on` and the per-frame, never-blocking `LocalExecutor` for futures.
//! - [`fuzz`]: the deterministic mutation-fuzz harness every parser test uses `[ROB-01]`.
//! - [`hash`]: `Hash128` (MurmurHash3 x64_128, content addressing), `Hash64` + `BuildFast` (hash maps),
//!   `crc32`, `adler32`.
//! - [`hex`]: strict lower-case hexadecimal encoding and decoding without allocation.
//! - [`id`]: `Id128` random document and object ids, and the `Sequence` counter.
//! - [`jobs`]: the job system: `Jobs`, `JobHandle`, `parallel_for`; inline when there are no threads.
//! - [`log`]: the logger: levels, targets, ring buffer for the console panel, sinks, `log_info!` etc.
//! - [`platform`]: the `FileSystem`, `Clock`, `Spawner` and `Entropy` traits, with `MemoryFs`,
//!   `ManualClock`, `InlineSpawner` and `SeededEntropy` for tests and wasm.
//! - [`profile`]: the always-compiled CPU scope profiler `[PERF-R6]`.
//! - [`rng`]: the deterministic PRNG (xoshiro256++).
//! - [`time`]: the process-wide installable monotonic clock and `Stopwatch`.
// [UNS-01] Unsafe code is allowed only in `alloc` (see that module).
#![deny(unsafe_code)]

#[allow(unsafe_code)]
pub mod alloc;
pub mod collections;
pub mod error;
pub mod executor;
pub mod fuzz;
pub mod hash;
pub mod hex;
pub mod id;
pub mod jobs;
pub mod log;
pub mod platform;
pub mod profile;
pub mod rng;
pub mod time;

pub use error::CoreError;

/// The unit tests count allocations `[PERF-R1]`, so the test binary runs on the counting allocator.
#[cfg(test)]
#[global_allocator]
static TEST_ALLOCATOR: alloc::CountingAllocator = alloc::CountingAllocator;
