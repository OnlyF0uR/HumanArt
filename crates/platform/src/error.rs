//! [`ShimError`]: why a native shim (clipboard, file dialog, pen) could not do what was asked.
//!
//! The file system, clock, spawner and entropy implementations report
//! [`humanart_core::error::PlatformError`], because the traits they implement say so.

use std::fmt;

/// Why a shim failed. Every shim has an in-app fallback `[AC-WP31-1]`; this error says why the
/// fallback was taken, for the log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShimError {
    /// The shim was switched off (preferences, a test, `--no-native-shims`).
    Disabled,
    /// There is no native implementation on this operating system, or the operating system lacks
    /// the component (no clipboard service, no dialog library).
    Unavailable,
    /// Another program holds the resource (the clipboard is open elsewhere) and kept it past the
    /// retries.
    Busy,
    /// The data is larger than the documented limit.
    TooLarge {
        /// The size found, in bytes.
        size: usize,
        /// The limit, in bytes.
        limit: usize,
    },
    /// A path the operating system returned cannot be represented as UTF-8.
    NonUtf8Path,
    /// The request itself is malformed (an empty filter, a NUL character in a title).
    InvalidRequest(&'static str),
    /// An operating-system call failed.
    Os {
        /// The function that failed.
        call: &'static str,
        /// Its error code (`GetLastError`, an `HRESULT`, or `errno`).
        code: i64,
    },
    /// The helper thread a blocking shim runs on could not be started or panicked.
    Thread(String),
}

impl fmt::Display for ShimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => f.write_str("the native shim is disabled"),
            Self::Unavailable => f.write_str("no native implementation is available on this system"),
            Self::Busy => f.write_str("the resource is held by another program"),
            Self::TooLarge { size, limit } => write!(f, "{size} bytes exceed the limit of {limit} bytes"),
            Self::NonUtf8Path => f.write_str("the system returned a path that is not valid UTF-8"),
            Self::InvalidRequest(why) => write!(f, "invalid request: {why}"),
            Self::Os { call, code } => write!(f, "{call} failed with code {code:#x}"),
            Self::Thread(why) => write!(f, "helper thread failed: {why}"),
        }
    }
}

impl std::error::Error for ShimError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_has_a_message() {
        let all = [
            ShimError::Disabled,
            ShimError::Unavailable,
            ShimError::Busy,
            ShimError::TooLarge { size: 10, limit: 4 },
            ShimError::NonUtf8Path,
            ShimError::InvalidRequest("empty filter"),
            ShimError::Os { call: "OpenClipboard", code: 5 },
            ShimError::Thread("spawn".to_owned()),
        ];
        for error in all {
            assert!(!error.to_string().is_empty());
        }
        assert_eq!(ShimError::Os { call: "X", code: 0x80 }.to_string(), "X failed with code 0x80");
    }
}
