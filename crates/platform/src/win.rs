//! Small helpers the Windows shims share: UTF-16 strings and the thread's last error code.

#[link(name = "kernel32")]
unsafe extern "system" {
    safe fn GetLastError() -> u32;
}

/// `text` as NUL-terminated UTF-16, or `None` when it contains a NUL of its own (the system would
/// silently cut it there).
pub(crate) fn wide(text: &str) -> Option<Vec<u16>> {
    if text.contains('\0') {
        return None;
    }
    Some(text.encode_utf16().chain(std::iter::once(0)).collect())
}

/// The calling thread's last Win32 error code (`GetLastError`).
pub(crate) fn last_error() -> i64 {
    i64::from(GetLastError())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_strings_are_terminated_and_refuse_inner_nuls() {
        assert_eq!(wide("A\u{e9}").unwrap(), [0x41, 0xe9, 0]);
        assert_eq!(wide("\u{1f600}").unwrap(), [0xd83d, 0xde00, 0]);
        assert!(wide("a\0b").is_none());
        // Reading the last error is always defined.
        let _ = last_error();
    }
}
