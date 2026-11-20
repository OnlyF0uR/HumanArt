//! Native [`Clock`], [`Spawner`] and [`Entropy`] `[PLAT-03]`: [`SystemClock`], [`ThreadSpawner`] and
//! [`OsEntropy`].
//!
//! The library crates never name `Instant`, `SystemTime` or `thread::spawn`; the binaries build these
//! and hand them over as trait objects.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use humanart_core::error::PlatformError;
use humanart_core::platform::{Clock, Entropy, Spawner, ThreadWork};

/// The operating system's clocks: a monotonic [`Instant`] for intervals and [`SystemTime`] for
/// timestamps.
#[derive(Clone, Copy, Debug)]
pub struct SystemClock {
    origin: Instant,
}

impl SystemClock {
    /// A clock whose monotonic time starts at 0 now.
    #[must_use]
    pub fn new() -> Self {
        Self { origin: Instant::now() }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn now_nanos(&self) -> u64 {
        // 2^64 ns is 584 years of uptime: saturating instead of wrapping is only a formality.
        u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    fn unix_millis(&self) -> u64 {
        // A system clock set before 1970 reads as 1970 rather than failing.
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| u64::try_from(since.as_millis()).unwrap_or(u64::MAX))
    }
}

/// Starts named, detached `std` threads.
#[derive(Clone, Copy, Debug, Default)]
pub struct ThreadSpawner {
    stack_bytes: Option<usize>,
}

impl ThreadSpawner {
    /// Threads with the platform's default stack size.
    #[must_use]
    pub const fn new() -> Self {
        Self { stack_bytes: None }
    }

    /// Threads with `bytes` of stack (the platform may round it up).
    #[must_use]
    pub const fn with_stack_size(bytes: usize) -> Self {
        Self { stack_bytes: Some(bytes) }
    }
}

impl Spawner for ThreadSpawner {
    fn available_threads(&self) -> usize {
        // A native target always has at least the thread that is asking.
        std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
    }

    fn spawn(&self, name: &str, work: ThreadWork) -> Result<(), PlatformError> {
        if name.contains('\0') {
            // `Builder::name` panics on an interior NUL.
            return Err(PlatformError::Other("thread name contains a NUL character".to_owned()));
        }
        let mut builder = std::thread::Builder::new().name(name.to_owned());
        if let Some(bytes) = self.stack_bytes {
            builder = builder.stack_size(bytes);
        }
        builder
            .spawn(work)
            .map(drop)
            .map_err(|error| PlatformError::Other(format!("cannot start thread {name}: {error}")))
    }
}

/// The operating system's cryptographic random source: `BCryptGenRandom` with the system-preferred
/// generator on Windows, `getentropy` on macOS, `/dev/urandom` on Linux and the other Unix systems.
#[derive(Clone, Copy, Debug, Default)]
pub struct OsEntropy;

impl Entropy for OsEntropy {
    fn fill(&self, out: &mut [u8]) -> Result<(), PlatformError> {
        os_random::fill(out)
    }
}

#[cfg(windows)]
mod os_random {
    use humanart_core::error::PlatformError;

    /// `BCRYPT_USE_SYSTEM_PREFERRED_RNG`: no algorithm handle needed.
    const SYSTEM_PREFERRED_RNG: u32 = 0x0000_0002;

    #[link(name = "bcrypt")]
    unsafe extern "system" {
        fn BCryptGenRandom(algorithm: *mut core::ffi::c_void, buffer: *mut u8, len: u32, flags: u32) -> i32;
    }

    pub(super) fn fill(out: &mut [u8]) -> Result<(), PlatformError> {
        for chunk in out.chunks_mut(u32::MAX as usize) {
            // SAFETY: `chunk` is a live, writable buffer of exactly `chunk.len()` bytes (at most
            // u32::MAX, so the length cast is exact); a null algorithm handle is what
            // BCRYPT_USE_SYSTEM_PREFERRED_RNG requires. The function writes only inside the buffer.
            let status = unsafe {
                BCryptGenRandom(core::ptr::null_mut(), chunk.as_mut_ptr(), chunk.len() as u32, SYSTEM_PREFERRED_RNG)
            };
            if status < 0 {
                return Err(PlatformError::Other(format!("BCryptGenRandom failed with NTSTATUS {status:#x}")));
            }
        }
        Ok(())
    }
}

/// UNVERIFIED (blocker B-003): type-checked for aarch64-apple-darwin, never linked or run.
#[cfg(target_os = "macos")]
mod os_random {
    use humanart_core::error::PlatformError;

    // libSystem, which every macOS process links.
    unsafe extern "C" {
        fn getentropy(buffer: *mut core::ffi::c_void, len: usize) -> i32;
    }

    pub(super) fn fill(out: &mut [u8]) -> Result<(), PlatformError> {
        // getentropy refuses requests above 256 bytes.
        for chunk in out.chunks_mut(256) {
            // SAFETY: `chunk` is a live, writable buffer of `chunk.len()` <= 256 bytes.
            let status = unsafe { getentropy(chunk.as_mut_ptr().cast(), chunk.len()) };
            if status != 0 {
                return Err(PlatformError::Other(format!("getentropy failed: {}", std::io::Error::last_os_error())));
            }
        }
        Ok(())
    }
}

/// UNVERIFIED (blocker B-003): type-checked for x86_64-unknown-linux-gnu, never run. No FFI: the kernel's
/// `/dev/urandom` never blocks once the pool is initialised and is always present.
#[cfg(all(unix, not(target_os = "macos")))]
mod os_random {
    use std::io::Read;

    use humanart_core::error::PlatformError;

    pub(super) fn fill(out: &mut [u8]) -> Result<(), PlatformError> {
        let mut device = std::fs::File::open("/dev/urandom")?;
        device.read_exact(out)?;
        Ok(())
    }
}

#[cfg(not(any(windows, unix)))]
mod os_random {
    use humanart_core::error::PlatformError;

    pub(super) fn fill(_out: &mut [u8]) -> Result<(), PlatformError> {
        Err(PlatformError::Unsupported)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

    #[test]
    fn system_clock_is_monotonic_and_reports_wall_time() {
        let clock = SystemClock::new();
        let mut last = clock.now_nanos();
        for _ in 0..1000 {
            let now = clock.now_nanos();
            assert!(now >= last);
            last = now;
        }
        std::thread::sleep(Duration::from_millis(5));
        assert!(clock.now_nanos() >= 5_000_000);
        // 2020-01-01 and 2200-01-01: anything outside is a broken conversion, not a wrong clock.
        let wall = clock.unix_millis();
        assert!((1_577_836_800_000..7_258_118_400_000).contains(&wall), "{wall}");
    }

    #[test]
    fn thread_spawner_names_threads_and_reports_parallelism() {
        let spawner = ThreadSpawner::new();
        assert!(spawner.available_threads() >= 1);
        let (send, receive) = mpsc::channel();
        spawner
            .spawn(
                "humanart-test-worker",
                Box::new(move || {
                    let name = std::thread::current().name().map(str::to_owned);
                    let _ = send.send(name);
                }),
            )
            .unwrap();
        let name = receive.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(name.as_deref(), Some("humanart-test-worker"));
        assert!(spawner.spawn("bad\0name", Box::new(|| {})).is_err());
    }

    #[test]
    fn thread_spawner_with_stack_size_runs_work() {
        let spawner = ThreadSpawner::with_stack_size(256 * 1024);
        let (send, receive) = mpsc::channel();
        spawner
            .spawn(
                "small-stack",
                Box::new(move || {
                    let _ = send.send(7);
                }),
            )
            .unwrap();
        assert_eq!(receive.recv_timeout(Duration::from_secs(10)).unwrap(), 7);
    }

    #[test]
    fn jobs_run_on_the_thread_spawner() {
        let jobs = humanart_core::jobs::Jobs::new(&ThreadSpawner::new(), 4).unwrap();
        let sums = jobs.parallel_map(1000, 10, Iterator::sum::<usize>).unwrap();
        assert_eq!(sums.iter().sum::<usize>(), 999 * 1000 / 2);
    }

    #[test]
    fn os_entropy_fills_every_byte_and_differs_between_calls() {
        let entropy: Arc<dyn Entropy> = Arc::new(OsEntropy);
        // Odd sizes, an empty buffer, and one large enough that a stuck byte would show.
        let mut empty = [0u8; 0];
        entropy.fill(&mut empty).unwrap();
        let mut a = vec![0u8; 4097];
        let mut b = vec![0u8; 4097];
        entropy.fill(&mut a).unwrap();
        entropy.fill(&mut b).unwrap();
        assert_ne!(a, b);
        // Every byte value is equally likely: 4097 bytes contain far more than 200 distinct values
        // and no value more than 60 times unless the source is broken.
        let mut counts = [0u32; 256];
        for &byte in &a {
            counts[usize::from(byte)] += 1;
        }
        assert!(counts.iter().filter(|&&c| c > 0).count() > 200);
        assert!(counts.iter().all(|&c| c < 60), "{counts:?}");
        let id = humanart_core::id::Id128::random(entropy.as_ref()).unwrap();
        assert_ne!(id, humanart_core::id::Id128::NIL);
    }
}
