//! One process-wide lock around the creation and destruction of wgpu instances, adapters and
//! devices, and the shared headless device of a process.
//!
//! Why: on Windows with the NVIDIA driver, creating and dropping several devices (Vulkan and GL) on
//! parallel threads can deadlock inside the driver. A thread inside `vkCreateDevice` holds a driver
//! lock and waits for the Windows loader lock (the driver loads libraries while it creates the
//! device), while another thread that exits, or creates a GL context, holds the loader lock and
//! spins on the driver lock. Test binaries, which run many tests on parallel threads that each made
//! their own device, hung about once in 23 runs under GPU contention (found by the skin-shading
//! work; stacks in its stress notes). No poll timeout ends it: it is not a wait on the GPU.
//!
//! Two measures, both here:
//!
//! - [`LifecycleGuard`]: every instance, adapter and device that [`crate::Gpu`] creates or drops is
//!   created or dropped while this guard is held, so at most one thread of the process is inside the
//!   driver's device creation or destruction at a time. Code outside [`crate::Gpu`] that creates or
//!   drops wgpu instances or devices itself should hold it too.
//! - [`shared_headless`]: one headless device per process, created once under the lock and never
//!   dropped. Test binaries and tools that only need "a GPU" share it, so no device is created or
//!   destroyed while test threads run and exit.
//!
//! The lock cannot stop an unrelated thread from exiting while a device is being created, so it
//! narrows the window rather than proving the deadlock impossible; the shared device closes it for
//! everything that uses it after the first call.
//!
//! On `wasm32` the browser owns the driver and there are no threads: the guard is a no-op there and
//! never blocks (a `std` condition variable cannot wait on that target).

#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Condvar, Mutex, PoisonError};

#[cfg(not(target_arch = "wasm32"))]
use crate::device::{DeviceRequest, Gpu};
#[cfg(not(target_arch = "wasm32"))]
use crate::error::GpuError;

/// True while some thread holds the lifecycle guard.
#[cfg(not(target_arch = "wasm32"))]
static HELD: Mutex<bool> = Mutex::new(false);
/// Signalled when the guard is released.
#[cfg(not(target_arch = "wasm32"))]
static RELEASED: Condvar = Condvar::new();

/// Proof that this thread may create or drop wgpu instances, adapters and devices: the process-wide
/// lifecycle lock is held until the guard drops.
///
/// Unlike a `MutexGuard` the guard is `Send` and `Sync`, so it can be stored in a [`crate::Gpu`]
/// while its fields are dropped and can be held across the (on native, immediately ready) awaits
/// of device creation. The lock is not re-entrant: a thread that holds a guard must not acquire a
/// second one.
#[derive(Debug)]
pub struct LifecycleGuard {
    /// Keeps the guard from being built outside [`LifecycleGuard::acquire`].
    _private: (),
}

impl LifecycleGuard {
    /// Blocks until no other thread holds the lifecycle lock, then takes it.
    #[must_use = "the lock is released as soon as the guard drops"]
    pub fn acquire() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // The mutex only guards a flag and nothing can panic while it is held, so a poisoned
            // mutex still holds a correct flag.
            let mut held = HELD.lock().unwrap_or_else(PoisonError::into_inner);
            while *held {
                held = RELEASED.wait(held).unwrap_or_else(PoisonError::into_inner);
            }
            *held = true;
        }
        Self { _private: () }
    }

    /// Takes the lifecycle lock only if no other thread holds it.
    pub fn try_acquire() -> Option<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut held = HELD.lock().unwrap_or_else(PoisonError::into_inner);
            if *held {
                return None;
            }
            *held = true;
        }
        Some(Self { _private: () })
    }
}

impl Drop for LifecycleGuard {
    fn drop(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut held = HELD.lock().unwrap_or_else(PoisonError::into_inner);
            *held = false;
            drop(held);
            RELEASED.notify_one();
        }
    }
}

/// The process's shared headless device, made from [`DeviceRequest::default`] on the first call
/// and never dropped.
///
/// For test binaries and tools that need a GPU but not a device of their own (tests that lose a
/// device on purpose, or need other features or limits, make their own with
/// [`Gpu::headless_blocking`]). A failure is remembered: every later call returns the same error
/// without trying again. Native only: a browser cannot block.
///
/// # Errors
///
/// The error of the one creation attempt, when the machine has no adapter that meets the baseline.
#[cfg(not(target_arch = "wasm32"))]
pub fn shared_headless() -> Result<&'static Gpu, &'static GpuError> {
    static SHARED: std::sync::OnceLock<Result<Gpu, GpuError>> = std::sync::OnceLock::new();
    SHARED.get_or_init(|| Gpu::headless_blocking(&DeviceRequest::default())).as_ref()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    use super::*;

    /// One test at a time may assert on the lock's state: the other tests of the binary create
    /// devices, which take it too.
    static SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn the_guard_admits_one_thread_at_a_time() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let inside = Arc::new(AtomicU32::new(0));
        let most = Arc::new(AtomicU32::new(0));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let inside = Arc::clone(&inside);
                let most = Arc::clone(&most);
                std::thread::spawn(move || {
                    for _ in 0..20 {
                        let _guard = LifecycleGuard::acquire();
                        let now = inside.fetch_add(1, Ordering::SeqCst) + 1;
                        most.fetch_max(now, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_micros(200));
                        inside.fetch_sub(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(most.load(Ordering::SeqCst), 1, "two threads held the lifecycle guard at once");
    }

    /// True if another thread can take the lock within `limit` (a broken release would hang it).
    fn acquirable_within(limit: Duration) -> bool {
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _guard = LifecycleGuard::acquire();
            let _ = send.send(());
        });
        receive.recv_timeout(limit).is_ok()
    }

    #[test]
    fn try_acquire_fails_while_the_guard_is_held_and_the_lock_is_free_after() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let guard = LifecycleGuard::acquire();
        let other = std::thread::spawn(|| LifecycleGuard::try_acquire().is_some()).join().unwrap();
        assert!(!other, "a second thread took the held lock");
        drop(guard);
        assert!(acquirable_within(Duration::from_secs(60)), "the released lock could not be taken");
    }

    #[test]
    fn a_gpu_drops_under_the_lock_and_releases_it_after() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        let gpu = Gpu::headless_blocking(&DeviceRequest::default()).unwrap();
        assert!(acquirable_within(Duration::from_secs(60)), "creation kept the lock");
        // While this thread holds the lock, dropping the device on another thread must wait.
        let guard = LifecycleGuard::acquire();
        let (send, receive) = std::sync::mpsc::channel();
        let dropper = std::thread::spawn(move || {
            drop(gpu);
            let _ = send.send(());
        });
        assert!(receive.recv_timeout(Duration::from_millis(200)).is_err(), "the device dropped without the lock");
        drop(guard);
        assert!(receive.recv_timeout(Duration::from_secs(60)).is_ok(), "the drop never finished");
        dropper.join().unwrap();
        assert!(acquirable_within(Duration::from_secs(60)), "the drop kept the lock");
    }

    #[test]
    fn devices_created_and_dropped_on_parallel_threads_all_work() {
        let threads: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    let gpu = Gpu::headless_blocking(&DeviceRequest::default()).unwrap();
                    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("lifecycle test"),
                        size: 256,
                        usage: wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    gpu.queue.write_buffer(&buffer, 0, &[1; 256]);
                    gpu.queue.submit([]);
                    !gpu.is_lost()
                })
            })
            .collect();
        for thread in threads {
            assert!(thread.join().unwrap(), "a device was lost");
        }
    }

    #[test]
    fn the_shared_device_is_made_once() {
        let first = shared_headless().unwrap();
        let second = std::thread::spawn(|| std::ptr::from_ref(shared_headless().unwrap()) as usize).join().unwrap();
        assert_eq!(std::ptr::from_ref(first) as usize, second);
        assert!(!first.is_lost());
    }
}
