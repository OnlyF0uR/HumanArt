//! Native shims: the native file system, clocks, threads and entropy behind the `humanart_core::platform`
//! traits, plus clipboard, file dialogs and pen input `[PLAT-03]` `[APP-07]` (WP-31). Native targets only.
//!
//! One of the three crates allowed `unsafe` (OS calls through hand-declared `extern` blocks, no
//! binding crates); every block carries a `// SAFETY:` comment and a test at its boundary.
//!
//! Module map:
//!
//! - [`fs`]: [`StdFileSystem`], rooted (cannot leave its directory) or host-wide.
//! - [`system`]: [`SystemClock`], [`ThreadSpawner`], [`OsEntropy`].
//! - [`clipboard`]: [`Clipboard`], system text with an in-process fallback.
//! - [`dialog`]: [`FileDialogs`], open / save / folder dialogs that report `Unavailable` when absent.
//! - [`pen`]: [`PenInput`], pen pressure and tilt next to winit's touch events; mouse semantics when
//!   disabled.
//! - [`error`]: [`ShimError`], why a shim fell back.
//!
//! [`Shims`] bundles the three user-facing shims so that one preference (or a command-line switch)
//! turns them all off `[AC-WP31-1]`.
//!
//! **DPI** needs no shim: winit makes the process per-monitor DPI aware (v2 on Windows) and reports
//! the scale through `Window::scale_factor()` and `WindowEvent::ScaleFactorChanged` on every
//! platform; the application reads it there.
//!
//! Windows (MSVC) is built and tested. Linux and macOS code type-checks (`cargo clippy --target
//! x86_64-unknown-linux-gnu` / `aarch64-apple-darwin`) but has never been linked or run (blocker
//! B-003); it is marked UNVERIFIED where it appears.

pub mod clipboard;
pub mod dialog;
pub mod error;
pub mod fs;
pub mod pen;
pub mod system;
#[cfg(windows)]
mod win;

pub use clipboard::{Clipboard, ClipboardTarget};
pub use dialog::{DialogKind, DialogOutcome, DialogRequest, FileDialogs, FileFilter, PendingDialog, RawWindow};
pub use error::ShimError;
pub use fs::StdFileSystem;
pub use pen::{PenInput, PenSample, PointerKind};
pub use system::{OsEntropy, SystemClock, ThreadSpawner};

/// The user-facing shims, all native or all replaced by their in-app fallbacks.
#[derive(Debug)]
pub struct Shims {
    /// Clipboard text.
    pub clipboard: Clipboard,
    /// Open, save and folder dialogs.
    pub dialogs: FileDialogs,
    /// Pen pressure and tilt.
    pub pen: PenInput,
}

impl Shims {
    /// The native shims where the platform has them, fallbacks where it does not.
    #[must_use]
    pub fn native() -> Self {
        Self { clipboard: Clipboard::native(), dialogs: FileDialogs::native(), pen: PenInput::native() }
    }

    /// Only the fallbacks: in-process clipboard, `Disabled` dialogs (the application's own path
    /// picker), mouse-like pen.
    #[must_use]
    pub fn disabled() -> Self {
        Self { clipboard: Clipboard::in_process(), dialogs: FileDialogs::disabled(), pen: PenInput::disabled() }
    }

    /// [`Shims::native`] when `enabled`, [`Shims::disabled`] otherwise.
    #[must_use]
    pub fn new(enabled: bool) -> Self {
        if enabled { Self::native() } else { Self::disabled() }
    }
}

// [PERF-R1] The test binary counts allocations so the file-system tests can measure them.
#[cfg(test)]
#[global_allocator]
static ALLOCATOR: humanart_core::alloc::CountingAllocator = humanart_core::alloc::CountingAllocator;

#[cfg(test)]
mod tests {
    use super::*;

    /// `[AC-WP31-1]`: with every shim disabled the application still copies and pastes, learns that
    /// it must show its own path picker, and paints with full pressure.
    #[test]
    fn every_fallback_functions_when_the_shims_are_disabled_ac_wp31_1() {
        let shims = Shims::new(false);
        assert!(!shims.clipboard.is_native());
        assert_eq!(shims.clipboard.set_text("copied"), ClipboardTarget::InProcess);
        assert_eq!(shims.clipboard.get_text().as_deref(), Some("copied"));
        assert!(!shims.dialogs.is_available());
        let request = DialogRequest::new(DialogKind::Save, "Save character").with_filter("Character", &["anth"]);
        assert_eq!(shims.dialogs.show(&request), Err(ShimError::Disabled));
        assert!(!shims.pen.is_available());
        assert_eq!(shims.pen.touch_sample(3, Some(0.1)).pressure, 1.0);
        assert_eq!(shims.pen.mouse_sample().kind, PointerKind::Mouse);
        let native = Shims::new(true);
        assert_eq!(native.dialogs.is_available(), cfg!(windows));
        assert_eq!(native.clipboard.is_native(), cfg!(windows));
    }
}
