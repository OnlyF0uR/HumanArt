//! The crate's error type, wgpu error scopes as `Result`s, and the device event queue `[ERR-05]`.
//!
//! wgpu reports problems in three ways, and all three end in [`GpuError`]:
//!
//! - **Error scopes** catch validation, out-of-memory and internal errors of the calls made while
//!   the scope is open. [`ErrorScope`] and [`capture`] turn them into `Err`.
//! - **Uncaptured errors** (anything outside a scope) would panic by default. [`DeviceEvents`]
//!   installs a handler that queues them instead; the application polls the queue once per frame.
//! - **Device loss** arrives through a callback, also queued by [`DeviceEvents`], and sets a flag that
//!   costs one atomic load to read. Recovery is in [`crate::recovery`].
//!
//! The reference application logs the first eight uncaptured errors and keeps a lost flag; it has no
//! error scopes. The queue here is the same idea with the errors kept for the application's own
//! error type instead of being written to a log this crate does not own.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::pod::PodError;

/// Everything that can go wrong in `humanart-gpu`.
#[derive(Debug)]
pub enum GpuError {
    /// No graphics adapter satisfies the request.
    NoAdapter(wgpu::RequestAdapterError),
    /// The adapter refused to create a device.
    Device(wgpu::RequestDeviceError),
    /// A surface could not be created for the window.
    Surface(wgpu::CreateSurfaceError),
    /// The surface offers no texture format at all.
    NoSurfaceFormat,
    /// The adapter is below the required baseline `[PLAT-05]`.
    BaselineNotMet {
        /// Name of the missing capability or limit.
        what: &'static str,
        /// Value the baseline needs (1 for a capability).
        required: u64,
        /// Value the adapter offers (0 for a missing capability).
        available: u64,
    },
    /// wgpu rejected a call: a bug in the caller's code or data.
    Validation {
        /// What was being done.
        scope: &'static str,
        /// wgpu's description.
        message: String,
    },
    /// The GPU ran out of memory.
    OutOfMemory {
        /// What was being done.
        scope: &'static str,
    },
    /// The driver or wgpu failed in a way WebGPU does not name.
    Internal {
        /// What was being done.
        scope: &'static str,
        /// wgpu's description.
        message: String,
    },
    /// The device is gone; every GPU object must be rebuilt (see [`crate::recovery`]).
    DeviceLost {
        /// Why, as far as the driver says.
        reason: wgpu::DeviceLostReason,
        /// The driver's message.
        message: String,
    },
    /// An allocation would exceed its memory budget and the degrade policy did not make room
    /// `[GPU-05]`.
    OverBudget {
        /// Name of the memory category.
        category: &'static str,
        /// Bytes asked for.
        requested: u64,
        /// Bytes already in use in the category.
        used: u64,
        /// The category's budget in bytes.
        budget: u64,
    },
    /// A write, copy or read would leave the resource.
    OutOfRange {
        /// What was being accessed.
        what: &'static str,
        /// First byte (or element) of the access.
        offset: u64,
        /// Length of the access.
        len: u64,
        /// Size of the resource.
        capacity: u64,
    },
    /// An argument is not usable (zero size, wrong alignment, wrong format, ...).
    Invalid {
        /// Which rule was broken.
        what: &'static str,
    },
    /// A fixed-capacity structure (readback ring, uniform ring, registry) has no free slot.
    Full {
        /// Which structure.
        what: &'static str,
    },
    /// A ticket or handle does not name a live entry (already taken, or never issued).
    Stale {
        /// Which kind of handle.
        what: &'static str,
    },
    /// Mapping a buffer for reading failed.
    Map {
        /// wgpu's description.
        message: String,
    },
    /// Waiting for the GPU failed or timed out.
    Poll(wgpu::PollError),
    /// Bytes could not be cast to the requested type.
    Pod(PodError),
}

impl GpuError {
    /// True when the device is lost and the GPU resources must be rebuilt.
    pub fn is_device_lost(&self) -> bool {
        matches!(self, Self::DeviceLost { .. })
    }

    /// True for the errors that ask for lower quality rather than a bug fix: out of GPU memory, or
    /// over a memory budget.
    pub fn is_out_of_memory(&self) -> bool {
        matches!(self, Self::OutOfMemory { .. } | Self::OverBudget { .. })
    }
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter(e) => write!(f, "no usable graphics adapter: {e}"),
            Self::Device(e) => write!(f, "the graphics device could not be created: {e}"),
            Self::Surface(e) => write!(f, "the window surface could not be created: {e}"),
            Self::NoSurfaceFormat => f.write_str("the window surface offers no texture format"),
            Self::BaselineNotMet { what, required, available } => {
                write!(f, "the graphics adapter is below the baseline: {what} is {available}, needed {required}")
            }
            Self::Validation { scope, message } => write!(f, "GPU validation error in {scope}: {message}"),
            Self::OutOfMemory { scope } => write!(f, "out of GPU memory in {scope}"),
            Self::Internal { scope, message } => write!(f, "internal GPU error in {scope}: {message}"),
            Self::DeviceLost { reason, message } => write!(f, "the graphics device was lost ({reason:?}): {message}"),
            Self::OverBudget { category, requested, used, budget } => write!(
                f,
                "GPU memory budget for {category} exceeded: {requested} bytes requested, {used} of {budget} in use"
            ),
            Self::OutOfRange { what, offset, len, capacity } => {
                write!(f, "{what}: {len} at offset {offset} does not fit in {capacity}")
            }
            Self::Invalid { what } => write!(f, "invalid GPU request: {what}"),
            Self::Full { what } => write!(f, "{what} is full"),
            Self::Stale { what } => write!(f, "{what} is no longer valid"),
            Self::Map { message } => write!(f, "a GPU buffer could not be mapped: {message}"),
            Self::Poll(e) => write!(f, "waiting for the GPU failed: {e}"),
            Self::Pod(e) => write!(f, "GPU data has the wrong shape: {e}"),
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoAdapter(e) => Some(e),
            Self::Device(e) => Some(e),
            Self::Surface(e) => Some(e),
            Self::Poll(e) => Some(e),
            Self::Pod(e) => Some(e),
            _ => None,
        }
    }
}

impl From<PodError> for GpuError {
    fn from(e: PodError) -> Self {
        Self::Pod(e)
    }
}

impl From<wgpu::PollError> for GpuError {
    fn from(e: wgpu::PollError) -> Self {
        Self::Poll(e)
    }
}

/// Turns a wgpu error into the crate's, tagged with what was being done.
fn from_wgpu(scope: &'static str, error: wgpu::Error) -> GpuError {
    match error {
        wgpu::Error::OutOfMemory { .. } => GpuError::OutOfMemory { scope },
        wgpu::Error::Validation { description, .. } => GpuError::Validation { scope, message: description },
        wgpu::Error::Internal { description, .. } => GpuError::Internal { scope, message: description },
    }
}

/// One open wgpu error scope. Errors of the given kind raised by calls on the device between
/// [`ErrorScope::push`] and the pop are caught instead of reaching the uncaptured-error handler.
///
/// Scopes are a per-thread stack: pop them in reverse order of pushing, on the thread that pushed.
/// Dropping a scope without popping discards what it caught.
#[must_use = "an error scope must be popped to get the errors it caught"]
pub struct ErrorScope {
    /// wgpu's guard.
    guard: wgpu::ErrorScopeGuard,
    /// What is being done, for the error message.
    what: &'static str,
}

impl ErrorScope {
    /// Opens a scope catching one kind of error. `what` names the operation in error messages.
    pub fn push(device: &wgpu::Device, filter: wgpu::ErrorFilter, what: &'static str) -> Self {
        Self { guard: device.push_error_scope(filter), what }
    }

    /// Opens a scope catching validation errors.
    pub fn validation(device: &wgpu::Device, what: &'static str) -> Self {
        Self::push(device, wgpu::ErrorFilter::Validation, what)
    }

    /// Opens a scope catching out-of-memory errors.
    pub fn out_of_memory(device: &wgpu::Device, what: &'static str) -> Self {
        Self::push(device, wgpu::ErrorFilter::OutOfMemory, what)
    }

    /// Closes the scope and reports the first error it caught. The scope is closed as soon as this
    /// is called; only the answer is awaited (in a browser it comes from the GPU process).
    pub async fn pop(self) -> Result<(), GpuError> {
        let what = self.what;
        match self.guard.pop().await {
            None => Ok(()),
            Some(error) => Err(from_wgpu(what, error)),
        }
    }

    /// Closes the scope and reports the first error it caught, without waiting.
    ///
    /// On native targets wgpu answers a pop at once, so this is the form used outside async code.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn pop_now(self) -> Result<(), GpuError> {
        let what = self.what;
        let mut future = std::pin::pin!(self.guard.pop());
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(None) => Ok(()),
            std::task::Poll::Ready(Some(error)) => Err(from_wgpu(what, error)),
            // Native wgpu resolves the pop synchronously; if a backend ever does not, the caller
            // gets an error rather than a wrong "ok".
            std::task::Poll::Pending => {
                Err(GpuError::Internal { scope: what, message: "the error scope did not resolve at once".to_owned() })
            }
        }
    }
}

/// Runs `work` with validation, out-of-memory and internal error scopes open and returns its value,
/// or the first error wgpu raised during it (out of memory first, then validation, then internal).
///
/// This is how fallible GPU setup (creating buffers, textures, pipelines) returns `Result` instead
/// of panicking `[ERR-02]`. Not for per-frame use: a scope costs a few allocations inside wgpu.
#[cfg(not(target_arch = "wasm32"))]
pub fn capture<T>(device: &wgpu::Device, what: &'static str, work: impl FnOnce() -> T) -> Result<T, GpuError> {
    let internal = ErrorScope::push(device, wgpu::ErrorFilter::Internal, what);
    let memory = ErrorScope::out_of_memory(device, what);
    let validation = ErrorScope::validation(device, what);
    let value = work();
    let validation = validation.pop_now();
    let memory = memory.pop_now();
    let internal = internal.pop_now();
    memory.and(validation).and(internal).map(|()| value)
}

/// [`capture`] for async callers; the only form available in a browser.
pub async fn capture_async<T>(
    device: &wgpu::Device,
    what: &'static str,
    work: impl FnOnce() -> T,
) -> Result<T, GpuError> {
    let internal = ErrorScope::push(device, wgpu::ErrorFilter::Internal, what);
    let memory = ErrorScope::out_of_memory(device, what);
    let validation = ErrorScope::validation(device, what);
    let value = work();
    // Each pop closes its scope immediately; only the results are awaited afterwards, so the
    // scopes are closed in reverse order as wgpu requires.
    let validation = validation.guard.pop();
    let memory = memory.guard.pop();
    let internal = internal.guard.pop();
    let first = match memory.await {
        Some(error) => Some(error),
        None => match validation.await {
            Some(error) => Some(error),
            None => internal.await,
        },
    };
    match first {
        None => Ok(value),
        Some(error) => Err(from_wgpu(what, error)),
    }
}

/// Runs fallible GPU setup with error scopes around it where their answer can be read at once.
///
/// On native targets this is [`capture`]. In a browser the answer of a scope only comes back
/// asynchronously, so the work runs unguarded and its errors arrive in [`DeviceEvents`] instead.
pub(crate) fn guarded<T>(device: &wgpu::Device, what: &'static str, work: impl FnOnce() -> T) -> Result<T, GpuError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        capture(device, what, work)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (device, what);
        Ok(work())
    }
}

/// Events kept when nobody polls; older ones are dropped and counted.
pub const MAX_PENDING_EVENTS: usize = 64;

/// Something the device reported outside an error scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceEvent {
    /// A validation error nobody captured.
    Validation {
        /// wgpu's description.
        message: String,
    },
    /// An out-of-memory error nobody captured.
    OutOfMemory,
    /// An internal error nobody captured.
    Internal {
        /// wgpu's description.
        message: String,
    },
    /// The device was lost.
    Lost {
        /// Why, as far as the driver says.
        reason: wgpu::DeviceLostReason,
        /// The driver's message.
        message: String,
    },
}

impl DeviceEvent {
    /// The event as an error for the application's error type.
    pub fn into_error(self) -> GpuError {
        const SCOPE: &str = "an uncaptured GPU call";
        match self {
            Self::Validation { message } => GpuError::Validation { scope: SCOPE, message },
            Self::OutOfMemory => GpuError::OutOfMemory { scope: SCOPE },
            Self::Internal { message } => GpuError::Internal { scope: SCOPE, message },
            Self::Lost { reason, message } => GpuError::DeviceLost { reason, message },
        }
    }
}

/// State shared between the application and wgpu's callbacks.
struct EventsInner {
    /// Set by the device-lost callback; never cleared (a new device gets a new queue).
    lost: AtomicBool,
    /// Events discarded because the queue was full.
    dropped: AtomicU32,
    /// Pending events, oldest first; capacity [`MAX_PENDING_EVENTS`], allocated once.
    queue: Mutex<VecDeque<DeviceEvent>>,
}

impl EventsInner {
    /// The queue, also after a panic elsewhere poisoned the lock (the data is a plain list and
    /// stays consistent).
    fn queue(&self) -> MutexGuard<'_, VecDeque<DeviceEvent>> {
        self.queue.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Appends an event, discarding the oldest when full.
    fn push(&self, event: DeviceEvent) {
        let mut queue = self.queue();
        if queue.len() >= MAX_PENDING_EVENTS {
            queue.pop_front();
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        queue.push_back(event);
    }
}

/// The queue of uncaptured errors and device loss of one device `[ERR-05]`.
///
/// Cloning gives another handle to the same queue. The application calls [`DeviceEvents::is_lost`]
/// at the start of a frame (one atomic load) and drains [`DeviceEvents::poll`] into its own error
/// handling.
#[derive(Clone)]
pub struct DeviceEvents {
    /// The shared state.
    inner: Arc<EventsInner>,
}

impl Default for DeviceEvents {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for DeviceEvents {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceEvents")
            .field("lost", &self.is_lost())
            .field("pending", &self.pending())
            .field("dropped", &self.dropped())
            .finish()
    }
}

impl DeviceEvents {
    /// An empty queue, not yet attached to a device.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(EventsInner {
                lost: AtomicBool::new(false),
                dropped: AtomicU32::new(0),
                queue: Mutex::new(VecDeque::with_capacity(MAX_PENDING_EVENTS)),
            }),
        }
    }

    /// Routes the device's uncaptured errors and its loss into this queue. After this, an
    /// uncaptured wgpu error no longer panics.
    pub fn attach(&self, device: &wgpu::Device) {
        let errors = Arc::clone(&self.inner);
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            errors.push(match error {
                wgpu::Error::OutOfMemory { .. } => DeviceEvent::OutOfMemory,
                wgpu::Error::Validation { description, .. } => DeviceEvent::Validation { message: description },
                wgpu::Error::Internal { description, .. } => DeviceEvent::Internal { message: description },
            });
        }));
        let lost = Arc::clone(&self.inner);
        device.set_device_lost_callback(move |reason, message| {
            lost.lost.store(true, Ordering::Release);
            lost.push(DeviceEvent::Lost { reason, message });
        });
    }

    /// True once the device has been lost. One atomic load: check it every frame.
    #[inline]
    pub fn is_lost(&self) -> bool {
        self.inner.lost.load(Ordering::Acquire)
    }

    /// Takes the oldest pending event.
    pub fn poll(&self) -> Option<DeviceEvent> {
        self.inner.queue().pop_front()
    }

    /// Number of events waiting to be polled.
    pub fn pending(&self) -> usize {
        self.inner.queue().len()
    }

    /// Number of events discarded because more than [`MAX_PENDING_EVENTS`] were pending.
    pub fn dropped(&self) -> u32 {
        self.inner.dropped.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_queue_keeps_the_newest_and_counts_what_it_dropped() {
        let events = DeviceEvents::new();
        assert_eq!(events.poll(), None);
        for i in 0..MAX_PENDING_EVENTS + 3 {
            events.inner.push(DeviceEvent::Validation { message: i.to_string() });
        }
        assert_eq!(events.pending(), MAX_PENDING_EVENTS);
        assert_eq!(events.dropped(), 3);
        assert_eq!(events.poll(), Some(DeviceEvent::Validation { message: "3".to_owned() }), "the oldest three went");
        assert!(!events.is_lost());
    }

    #[test]
    fn clones_share_one_queue() {
        let events = DeviceEvents::new();
        let other = events.clone();
        events.inner.push(DeviceEvent::OutOfMemory);
        assert_eq!(other.poll(), Some(DeviceEvent::OutOfMemory));
        assert_eq!(events.pending(), 0);
    }

    #[test]
    fn events_convert_to_errors_with_predicates() {
        let lost = DeviceEvent::Lost { reason: wgpu::DeviceLostReason::Destroyed, message: "gone".to_owned() };
        let error = lost.into_error();
        assert!(error.is_device_lost());
        assert_eq!(error.to_string(), "the graphics device was lost (Destroyed): gone");
        assert!(DeviceEvent::OutOfMemory.into_error().is_out_of_memory());
        let over = GpuError::OverBudget { category: "textures", requested: 10, used: 95, budget: 100 };
        assert!(over.is_out_of_memory());
        assert_eq!(over.to_string(), "GPU memory budget for textures exceeded: 10 bytes requested, 95 of 100 in use");
    }

    #[test]
    fn errors_chain_their_source() {
        use std::error::Error as _;
        let error = GpuError::from(PodError::ZeroSized);
        assert!(error.source().is_some());
        assert_eq!(error.to_string(), "GPU data has the wrong shape: a zero-sized type cannot be cast from bytes");
        assert!(GpuError::NoSurfaceFormat.source().is_none());
        let range = GpuError::OutOfRange { what: "uniform buffer write", offset: 8, len: 16, capacity: 16 };
        assert_eq!(range.to_string(), "uniform buffer write: 16 at offset 8 does not fit in 16");
    }
}
