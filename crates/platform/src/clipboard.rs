//! [`Clipboard`]: system clipboard text with an in-process fallback `[APP-07]` `[AC-WP31-1]`.
//!
//! On Windows the text goes through `OpenClipboard` / `CF_UNICODETEXT` (UTF-16, CRLF line ends on
//! the clipboard, LF inside the application). Everywhere else, and whenever the shim is disabled or
//! the system clipboard fails, copy and paste still work inside the application through a buffer
//! held here: the editor never loses a copy, it only cannot share it with other programs.
//!
//! Linux and macOS have no native implementation yet (X11 and Wayland selections need a window
//! connection; macOS needs the Objective-C runtime): they always use the in-process buffer.

use std::sync::{Mutex, PoisonError};

use crate::error::ShimError;

/// The largest text the clipboard shim reads or writes: 64 MiB of UTF-16. Anything larger is
/// [`ShimError::TooLarge`] rather than an allocation of whatever size another program announced.
pub const MAX_CLIPBOARD_BYTES: usize = 64 << 20;

/// Where [`Clipboard::set_text`] put the text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipboardTarget {
    /// The system clipboard (and the in-process buffer): other programs can paste it.
    System,
    /// Only the in-process buffer: the shim is disabled, unavailable or failed (see
    /// [`Clipboard::take_last_error`]).
    InProcess,
}

/// Clipboard text for the editor. Cheap to share behind an `Arc`; every method takes `&self`.
#[derive(Debug)]
pub struct Clipboard {
    native: bool,
    buffer: Mutex<String>,
    last_error: Mutex<Option<ShimError>>,
}

impl Clipboard {
    /// The system clipboard where this platform has a shim (Windows), the in-process buffer
    /// otherwise.
    #[must_use]
    pub fn native() -> Self {
        Self { native: cfg!(windows), buffer: Mutex::new(String::new()), last_error: Mutex::new(None) }
    }

    /// The in-process buffer only: the fallback, and what tests use so that they never touch the
    /// owner's clipboard.
    #[must_use]
    pub fn in_process() -> Self {
        Self { native: false, buffer: Mutex::new(String::new()), last_error: Mutex::new(None) }
    }

    /// `true` when the system clipboard is used.
    pub fn is_native(&self) -> bool {
        self.native
    }

    /// Copies `text`. It always lands in the in-process buffer; the result says whether the system
    /// clipboard took it too.
    pub fn set_text(&self, text: &str) -> ClipboardTarget {
        text.clone_into(&mut self.buffer.lock().unwrap_or_else(PoisonError::into_inner));
        if !self.native {
            return ClipboardTarget::InProcess;
        }
        match system::set_text(text) {
            Ok(()) => ClipboardTarget::System,
            Err(error) => {
                self.record(error);
                ClipboardTarget::InProcess
            }
        }
    }

    /// The text to paste: the system clipboard's when it can be read (`None` when it holds no
    /// text, for example a picture copied elsewhere), otherwise the in-process buffer's (`None`
    /// when nothing was copied).
    pub fn get_text(&self) -> Option<String> {
        if self.native {
            match system::get_text() {
                Ok(text) => return text,
                Err(error) => self.record(error),
            }
        }
        let buffer = self.buffer.lock().unwrap_or_else(PoisonError::into_inner);
        (!buffer.is_empty()).then(|| buffer.clone())
    }

    /// Why the system clipboard was last bypassed, once.
    pub fn take_last_error(&self) -> Option<ShimError> {
        self.last_error.lock().unwrap_or_else(PoisonError::into_inner).take()
    }

    fn record(&self, error: ShimError) {
        *self.last_error.lock().unwrap_or_else(PoisonError::into_inner) = Some(error);
    }
}

impl Default for Clipboard {
    fn default() -> Self {
        Self::native()
    }
}

/// Text as the Windows clipboard stores it: CRLF line ends, UTF-16, NUL-terminated. A lone `\n`
/// becomes `\r\n`; an existing `\r\n` stays as it is.
pub fn encode_windows_text(text: &str) -> Vec<u16> {
    let mut out = Vec::with_capacity(text.len() + text.len() / 16 + 1);
    let mut previous = 0u16;
    for unit in text.encode_utf16() {
        if unit == u16::from(b'\n') && previous != u16::from(b'\r') {
            out.push(u16::from(b'\r'));
        }
        out.push(unit);
        previous = unit;
    }
    out.push(0);
    out
}

/// The application's text from what the Windows clipboard holds: cut at the first NUL (the block
/// may be larger than the text), CRLF turned into LF, unpaired surrogates replaced by U+FFFD.
pub fn decode_windows_text(units: &[u16]) -> String {
    let end = units.iter().position(|&u| u == 0).unwrap_or(units.len());
    String::from_utf16_lossy(&units[..end]).replace("\r\n", "\n")
}

#[cfg(windows)]
mod system {
    use core::ffi::c_void;
    use std::time::Duration;

    use super::{MAX_CLIPBOARD_BYTES, decode_windows_text, encode_windows_text};
    use crate::error::ShimError;
    use crate::win::last_error;

    /// UTF-16 text, NUL-terminated.
    const CF_UNICODETEXT: u32 = 13;
    /// `GlobalAlloc` flag: a movable block, which the clipboard requires.
    const GMEM_MOVEABLE: u32 = 0x0002;

    #[link(name = "user32")]
    unsafe extern "system" {
        fn OpenClipboard(owner: *mut c_void) -> i32;
        safe fn CloseClipboard() -> i32;
        safe fn EmptyClipboard() -> i32;
        safe fn IsClipboardFormatAvailable(format: u32) -> i32;
        safe fn GetClipboardData(format: u32) -> *mut c_void;
        fn SetClipboardData(format: u32, memory: *mut c_void) -> *mut c_void;
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        safe fn GlobalAlloc(flags: u32, bytes: usize) -> *mut c_void;
        fn GlobalLock(memory: *mut c_void) -> *mut c_void;
        fn GlobalUnlock(memory: *mut c_void) -> i32;
        fn GlobalSize(memory: *mut c_void) -> usize;
        fn GlobalFree(memory: *mut c_void) -> *mut c_void;
    }

    /// The clipboard, open on this thread until dropped.
    struct Open;

    impl Open {
        /// Opens the clipboard, retrying for a moment if another program has it open.
        fn new() -> Result<Self, ShimError> {
            for attempt in 0..5 {
                if attempt > 0 {
                    std::thread::sleep(Duration::from_millis(10));
                }
                // SAFETY: a null owner is allowed; it ties the open clipboard to this thread,
                // which `Drop` closes again on every path out.
                if unsafe { OpenClipboard(core::ptr::null_mut()) } != 0 {
                    return Ok(Self);
                }
            }
            Err(ShimError::Busy)
        }
    }

    impl Drop for Open {
        fn drop(&mut self) {
            CloseClipboard();
        }
    }

    /// A movable global memory block that is freed on drop unless handed to the system.
    pub(super) struct Global(pub(super) *mut c_void);

    impl Global {
        /// A block holding exactly `units`.
        pub(super) fn from_units(units: &[u16]) -> Result<Self, ShimError> {
            let bytes = std::mem::size_of_val(units);
            if bytes > MAX_CLIPBOARD_BYTES {
                return Err(ShimError::TooLarge { size: bytes, limit: MAX_CLIPBOARD_BYTES });
            }
            let memory = GlobalAlloc(GMEM_MOVEABLE, bytes.max(2));
            if memory.is_null() {
                return Err(ShimError::Os { call: "GlobalAlloc", code: last_error() });
            }
            let block = Self(memory);
            // SAFETY: `memory` is a live movable block of at least `bytes` bytes that this
            // function owns. While locked, `target` points to it (suitably aligned for u16: global
            // blocks are 8-byte aligned); exactly `units.len()` units are written, then unlocked.
            unsafe {
                let target = GlobalLock(memory).cast::<u16>();
                if target.is_null() {
                    return Err(ShimError::Os { call: "GlobalLock", code: last_error() });
                }
                core::ptr::copy_nonoverlapping(units.as_ptr(), target, units.len());
                GlobalUnlock(memory);
            }
            Ok(block)
        }

        /// The ownership of the block passes to the caller (the clipboard).
        fn into_raw(self) -> *mut c_void {
            let raw = self.0;
            std::mem::forget(self);
            raw
        }
    }

    impl Drop for Global {
        fn drop(&mut self) {
            // SAFETY: the block came from GlobalAlloc, is owned by this value and is unlocked.
            unsafe { GlobalFree(self.0) };
        }
    }

    /// A copy of the UTF-16 units in global block `memory`, at most [`MAX_CLIPBOARD_BYTES`].
    ///
    /// # Safety
    /// `memory` must be a valid global memory handle for the duration of the call (one owned by the
    /// caller, or the clipboard's while the clipboard is open on this thread).
    pub(super) unsafe fn units_of(memory: *mut c_void) -> Result<Vec<u16>, ShimError> {
        // SAFETY: `memory` is a valid global handle (the caller's promise).
        let bytes = unsafe { GlobalSize(memory) };
        if bytes > MAX_CLIPBOARD_BYTES {
            return Err(ShimError::TooLarge { size: bytes, limit: MAX_CLIPBOARD_BYTES });
        }
        // SAFETY: as above; while locked, `source` points to `bytes` readable bytes, aligned for
        // u16, of which the whole units are copied before the unlock.
        unsafe {
            let source = GlobalLock(memory).cast::<u16>();
            if source.is_null() {
                return Err(ShimError::Os { call: "GlobalLock", code: last_error() });
            }
            let copy = std::slice::from_raw_parts(source, bytes / 2).to_vec();
            GlobalUnlock(memory);
            Ok(copy)
        }
    }

    pub(super) fn get_text() -> Result<Option<String>, ShimError> {
        let _open = Open::new()?;
        if IsClipboardFormatAvailable(CF_UNICODETEXT) == 0 {
            return Ok(None);
        }
        let memory = GetClipboardData(CF_UNICODETEXT);
        if memory.is_null() {
            return Err(ShimError::Os { call: "GetClipboardData", code: last_error() });
        }
        // SAFETY: the clipboard is open on this thread (`_open`), so the handle stays valid until
        // the end of this function, and CF_UNICODETEXT data is a global memory block.
        let units = unsafe { units_of(memory)? };
        Ok(Some(decode_windows_text(&units)))
    }

    pub(super) fn set_text(text: &str) -> Result<(), ShimError> {
        let block = Global::from_units(&encode_windows_text(text))?;
        let _open = Open::new()?;
        if EmptyClipboard() == 0 {
            return Err(ShimError::Os { call: "EmptyClipboard", code: last_error() });
        }
        let raw = block.into_raw();
        // SAFETY: the clipboard is open and emptied on this thread; `raw` is an unlocked movable
        // global block holding NUL-terminated UTF-16, as CF_UNICODETEXT requires. On success the
        // system owns it; on failure it is still ours and is freed here.
        unsafe {
            if SetClipboardData(CF_UNICODETEXT, raw).is_null() {
                let code = last_error();
                GlobalFree(raw);
                return Err(ShimError::Os { call: "SetClipboardData", code });
            }
        }
        Ok(())
    }
}

#[cfg(not(windows))]
mod system {
    use crate::error::ShimError;

    pub(super) fn get_text() -> Result<Option<String>, ShimError> {
        Err(ShimError::Unavailable)
    }

    pub(super) fn set_text(_text: &str) -> Result<(), ShimError> {
        Err(ShimError::Unavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_process_clipboard_copies_and_pastes_without_the_system_ac_wp31_1() {
        let clipboard = Clipboard::in_process();
        assert!(!clipboard.is_native());
        assert_eq!(clipboard.get_text(), None);
        assert_eq!(clipboard.set_text("smile \u{1f600}\nline 2"), ClipboardTarget::InProcess);
        assert_eq!(clipboard.get_text().as_deref(), Some("smile \u{1f600}\nline 2"));
        clipboard.set_text("");
        assert_eq!(clipboard.get_text(), None);
        assert_eq!(clipboard.take_last_error(), None);
    }

    #[test]
    fn windows_text_uses_crlf_on_the_clipboard_and_lf_inside() {
        let units = encode_windows_text("a\nb\r\nc\u{e9}\u{1f600}");
        assert_eq!(String::from_utf16(&units).unwrap(), "a\r\nb\r\nc\u{e9}\u{1f600}\0");
        assert_eq!(decode_windows_text(&units), "a\nb\nc\u{e9}\u{1f600}");
        assert_eq!(encode_windows_text(""), [0]);
        // Bytes after the terminator (blocks are rounded up) and lone surrogates.
        assert_eq!(decode_windows_text(&[0x68, 0x69, 0, 0x41, 0x42]), "hi");
        assert_eq!(decode_windows_text(&[0xd800, 0x41]), "\u{fffd}A");
        assert_eq!(decode_windows_text(&[0x41, 0x42]), "AB");
        assert_eq!(decode_windows_text(&[]), "");
    }

    /// The unsafe global-memory code at its boundary, without touching the clipboard.
    #[cfg(windows)]
    #[test]
    fn global_blocks_round_trip_text_and_refuse_oversized_input() {
        for text in ["", "x", "two\nlines \u{1f600}", &"long ".repeat(10_000)] {
            let units = encode_windows_text(text);
            let block = system::Global::from_units(&units).unwrap();
            // SAFETY: `block` owns a valid global block for the whole call.
            let back = unsafe { system::units_of(block.0) }.unwrap();
            assert_eq!(&back[..units.len()], &units[..]);
            assert_eq!(decode_windows_text(&back), text);
        }
        let huge = vec![0x41u16; MAX_CLIPBOARD_BYTES / 2 + 1];
        assert!(matches!(system::Global::from_units(&huge), Err(ShimError::TooLarge { .. })));
    }

    /// Reads the owner's real clipboard (it does not change it). Run by hand:
    /// `cargo test -p humanart-platform -- --ignored system_clipboard_can_be_read`.
    #[test]
    #[ignore = "opens the user's real clipboard"]
    fn system_clipboard_can_be_read() {
        let clipboard = Clipboard::native();
        let text = clipboard.get_text();
        // The content is private: only its size is reported.
        eprintln!(
            "clipboard text: {:?} chars, error: {:?}",
            text.map(|t| t.chars().count()),
            clipboard.take_last_error()
        );
    }

    /// Overwrites the owner's real clipboard. Run by hand only, on a test machine:
    /// `cargo test -p humanart-platform -- --ignored system_clipboard_round_trips_text`.
    #[test]
    #[ignore = "overwrites the user's real clipboard"]
    fn system_clipboard_round_trips_text() {
        let clipboard = Clipboard::native();
        let text = "HumanArt clipboard test \u{e9}\u{1f600}\nsecond line";
        assert_eq!(clipboard.set_text(text), ClipboardTarget::System, "{:?}", clipboard.take_last_error());
        // A fresh instance has an empty buffer, so this can only come from the system clipboard.
        assert_eq!(Clipboard::native().get_text().as_deref(), Some(text));
    }
}
