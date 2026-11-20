//! The error enums of `humanart-core` `[ERR-01]`.
//!
//! Every fallible function of this crate returns one of the small enums below; [`CoreError`] is the
//! umbrella the other crates wrap as the `source()` of their own errors. None of them carries a path
//! or a job name: the caller, which knows what it was doing, adds that context in its own error type.

use std::fmt;

/// A failure of a platform service (file system, threads, entropy), see [`crate::platform`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlatformError {
    /// The file or directory does not exist.
    NotFound,
    /// The operating system (or a read-only file system) refused the operation.
    PermissionDenied,
    /// The target exists and the operation does not replace it.
    AlreadyExists,
    /// The path is empty, malformed, or leaves the root of the file system.
    InvalidPath,
    /// A directory was required and something else was found.
    NotADirectory,
    /// A file was required and a directory was found.
    IsADirectory,
    /// The directory cannot be removed because it still has entries.
    NotEmpty,
    /// The platform cannot do this at all (threads on single-threaded wasm, for example).
    Unsupported,
    /// Anything else, with the platform's description.
    Other(String),
}

impl fmt::Display for PlatformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => f.write_str("not found"),
            Self::PermissionDenied => f.write_str("permission denied"),
            Self::AlreadyExists => f.write_str("already exists"),
            Self::InvalidPath => f.write_str("invalid path"),
            Self::NotADirectory => f.write_str("not a directory"),
            Self::IsADirectory => f.write_str("is a directory"),
            Self::NotEmpty => f.write_str("directory not empty"),
            Self::Unsupported => f.write_str("not supported on this platform"),
            Self::Other(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for PlatformError {}

/// The mapping every native implementation of the platform traits uses, so that the same failure has
/// the same kind on every operating system.
impl From<std::io::Error> for PlatformError {
    fn from(error: std::io::Error) -> Self {
        use std::io::ErrorKind;
        match error.kind() {
            ErrorKind::NotFound => Self::NotFound,
            ErrorKind::PermissionDenied | ErrorKind::ReadOnlyFilesystem => Self::PermissionDenied,
            ErrorKind::AlreadyExists => Self::AlreadyExists,
            ErrorKind::InvalidInput | ErrorKind::InvalidFilename => Self::InvalidPath,
            ErrorKind::NotADirectory => Self::NotADirectory,
            ErrorKind::IsADirectory => Self::IsADirectory,
            ErrorKind::DirectoryNotEmpty => Self::NotEmpty,
            ErrorKind::Unsupported => Self::Unsupported,
            _ => Self::Other(error.to_string()),
        }
    }
}

/// Why a job produced no value, see [`crate::jobs`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobError {
    /// The job panicked; the worker survived. Carries the panic message when it was a string.
    Panicked(String),
    /// The job was cancelled through its handle, or its job system shut down, before it finished.
    Cancelled,
    /// The result was already taken from the handle.
    AlreadyTaken,
    /// The work cannot finish without blocking, and this platform has no threads to block on
    /// (single-threaded wasm); see [`crate::executor::block_on`].
    WouldBlock,
}

impl fmt::Display for JobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Panicked(message) => write!(f, "job panicked: {message}"),
            Self::Cancelled => f.write_str("job cancelled"),
            Self::AlreadyTaken => f.write_str("job result already taken"),
            Self::WouldBlock => f.write_str("waiting would block a platform without threads"),
        }
    }
}

impl std::error::Error for JobError {}

/// A failure of a log sink, see [`crate::log`].
#[derive(Debug)]
pub enum LogError {
    /// Writing to or flushing the underlying writer failed.
    Io(std::io::Error),
    /// The file system behind a file sink failed.
    Platform(PlatformError),
    /// A sink failed for a reason of its own.
    Sink(String),
}

impl fmt::Display for LogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(_) => f.write_str("log sink write failed"),
            Self::Platform(_) => f.write_str("log file write failed"),
            Self::Sink(message) => write!(f, "log sink failed: {message}"),
        }
    }
}

impl std::error::Error for LogError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Platform(error) => Some(error),
            Self::Sink(_) => None,
        }
    }
}

impl From<std::io::Error> for LogError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<PlatformError> for LogError {
    fn from(error: PlatformError) -> Self {
        Self::Platform(error)
    }
}

/// Why a hexadecimal string was rejected, see [`crate::hex`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HexError {
    /// The text does not have exactly the number of digits the value needs.
    Length {
        /// Digits required.
        expected: usize,
        /// Bytes of text given.
        found: usize,
    },
    /// A byte that is not one of `0-9a-f` (upper case is rejected: every value has one spelling).
    Digit {
        /// Byte offset of the offending character.
        index: usize,
    },
}

impl fmt::Display for HexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length { expected, found } => write!(f, "expected {expected} hex digits, found {found} bytes"),
            Self::Digit { index } => write!(f, "invalid hex digit at offset {index} (only 0-9 and a-f are accepted)"),
        }
    }
}

impl std::error::Error for HexError {}

/// Why an [`Arena`](crate::alloc::Arena) could not hand out memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaError {
    /// The requested size (element size times count, plus alignment) does not fit the address space.
    TooLarge,
    /// The system allocator had no memory for a new chunk.
    OutOfMemory,
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge => f.write_str("arena request too large"),
            Self::OutOfMemory => f.write_str("out of memory growing an arena"),
        }
    }
}

impl std::error::Error for ArenaError {}

/// Any error of this crate, for callers that handle them together.
#[derive(Debug)]
pub enum CoreError {
    /// A frame arena could not hand out memory.
    Arena(ArenaError),
    /// A platform service failed.
    Platform(PlatformError),
    /// A job produced no value.
    Job(JobError),
    /// A log sink failed.
    Log(LogError),
    /// A hexadecimal id or hash did not parse.
    Hex(HexError),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Arena(_) => f.write_str("allocation error"),
            Self::Platform(_) => f.write_str("platform error"),
            Self::Job(_) => f.write_str("job error"),
            Self::Log(_) => f.write_str("logging error"),
            Self::Hex(_) => f.write_str("malformed hexadecimal value"),
        }
    }
}

impl std::error::Error for CoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Arena(error) => Some(error),
            Self::Platform(error) => Some(error),
            Self::Job(error) => Some(error),
            Self::Log(error) => Some(error),
            Self::Hex(error) => Some(error),
        }
    }
}

impl From<ArenaError> for CoreError {
    fn from(error: ArenaError) -> Self {
        Self::Arena(error)
    }
}

impl From<PlatformError> for CoreError {
    fn from(error: PlatformError) -> Self {
        Self::Platform(error)
    }
}

impl From<JobError> for CoreError {
    fn from(error: JobError) -> Self {
        Self::Job(error)
    }
}

impl From<LogError> for CoreError {
    fn from(error: LogError) -> Self {
        Self::Log(error)
    }
}

impl From<HexError> for CoreError {
    fn from(error: HexError) -> Self {
        Self::Hex(error)
    }
}

/// Writes `error` and its whole `source()` chain as `outer: inner: innermost`.
///
/// This is the form log lines and user-facing messages use; the chain is cut after 16 links so a
/// cyclic `source()` implementation cannot loop forever.
pub fn write_chain(out: &mut dyn fmt::Write, error: &dyn std::error::Error) -> fmt::Result {
    write!(out, "{error}")?;
    let mut next = error.source();
    for _ in 0..16 {
        let Some(inner) = next else { break };
        write!(out, ": {inner}")?;
        next = inner.source();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    // [ERR-01]
    #[test]
    fn display_is_specific_for_every_variant() {
        let all = [
            PlatformError::NotFound,
            PlatformError::PermissionDenied,
            PlatformError::AlreadyExists,
            PlatformError::InvalidPath,
            PlatformError::NotADirectory,
            PlatformError::IsADirectory,
            PlatformError::NotEmpty,
            PlatformError::Unsupported,
            PlatformError::Other("disk on fire".to_owned()),
        ];
        let mut seen = std::collections::HashSet::new();
        for e in &all {
            let text = e.to_string();
            assert!(!text.is_empty());
            assert!(seen.insert(text), "duplicate message for {e:?}");
        }
        assert_eq!(all[8].to_string(), "disk on fire");
        assert_eq!(JobError::Panicked("boom".to_owned()).to_string(), "job panicked: boom");
        assert_eq!(JobError::Cancelled.to_string(), "job cancelled");
        assert_eq!(JobError::AlreadyTaken.to_string(), "job result already taken");
        assert!(JobError::WouldBlock.to_string().contains("without threads"));
        assert_eq!(HexError::Length { expected: 32, found: 3 }.to_string(), "expected 32 hex digits, found 3 bytes");
        assert!(HexError::Digit { index: 5 }.to_string().contains("offset 5"));
    }

    // [ERR-01]
    #[test]
    fn io_errors_map_to_the_same_kinds_everywhere() {
        use std::io::{Error, ErrorKind};
        let map = |kind| PlatformError::from(Error::new(kind, "x"));
        assert_eq!(map(ErrorKind::NotFound), PlatformError::NotFound);
        assert_eq!(map(ErrorKind::PermissionDenied), PlatformError::PermissionDenied);
        assert_eq!(map(ErrorKind::ReadOnlyFilesystem), PlatformError::PermissionDenied);
        assert_eq!(map(ErrorKind::AlreadyExists), PlatformError::AlreadyExists);
        assert_eq!(map(ErrorKind::InvalidInput), PlatformError::InvalidPath);
        assert_eq!(map(ErrorKind::NotADirectory), PlatformError::NotADirectory);
        assert_eq!(map(ErrorKind::IsADirectory), PlatformError::IsADirectory);
        assert_eq!(map(ErrorKind::DirectoryNotEmpty), PlatformError::NotEmpty);
        assert_eq!(map(ErrorKind::Unsupported), PlatformError::Unsupported);
        assert_eq!(map(ErrorKind::TimedOut), PlatformError::Other("x".to_owned()));
    }

    // [ERR-01]
    #[test]
    fn sources_chain_through_the_umbrella() {
        let io = std::io::Error::other("pipe closed");
        let core = CoreError::from(LogError::from(io));
        let log = core.source().expect("log error");
        assert_eq!(log.to_string(), "log sink write failed");
        assert_eq!(log.source().expect("io error").to_string(), "pipe closed");
        let mut text = String::new();
        write_chain(&mut text, &core).expect("write to a String");
        assert_eq!(text, "logging error: log sink write failed: pipe closed");

        assert!(LogError::Sink("full".to_owned()).source().is_none());
        let platform = LogError::from(PlatformError::NotFound);
        assert_eq!(platform.to_string(), "log file write failed");
        assert_eq!(platform.source().expect("platform error").to_string(), "not found");
        assert!(PlatformError::NotFound.source().is_none());
        for (error, inner) in [
            (CoreError::from(PlatformError::NotFound), "not found"),
            (CoreError::from(JobError::Cancelled), "job cancelled"),
            (CoreError::from(ArenaError::TooLarge), "arena request too large"),
            (CoreError::from(ArenaError::OutOfMemory), "out of memory growing an arena"),
            (
                CoreError::from(HexError::Digit { index: 0 }),
                "invalid hex digit at offset 0 (only 0-9 and a-f are accepted)",
            ),
        ] {
            assert_eq!(error.source().expect("source").to_string(), inner);
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn write_chain_stops_on_a_cyclic_source() {
        #[derive(Debug)]
        struct Cyclic;
        impl fmt::Display for Cyclic {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("again")
            }
        }
        impl Error for Cyclic {
            fn source(&self) -> Option<&(dyn Error + 'static)> {
                Some(&Cyclic)
            }
        }
        let mut text = String::new();
        write_chain(&mut text, &Cyclic).expect("write to a String");
        assert_eq!(text.matches("again").count(), 17);
    }
}
