//! The error enum of `humanart-doc` `[ERR-01]`.
//!
//! Every fallible function of this crate returns [`DocError`]. Errors from parsing carry the line or
//! the path of the offending item; errors from the platform and the codecs keep the original error
//! as their [`source`](std::error::Error::source).

use std::fmt;

use humanart_core::error::PlatformError;
use humanart_core::hash::Hash128;
use humanart_io::json::{AccessError, ParseError};

/// Everything that can go wrong in the registry, the document, the blob store, the journal and the
/// project directory.
#[derive(Debug)]
pub enum DocError {
    /// The line-oriented text breaks the grammar of `docs/format-anthropos.md`.
    Syntax {
        /// The line, counting from 1.
        line: usize,
        /// What is wrong, as a short phrase.
        what: &'static str,
    },
    /// The JSON value on a line of the text does not parse.
    Value {
        /// The line, counting from 1.
        line: usize,
        /// The JSON parser's error (its offset is relative to the start of the value).
        source: ParseError,
    },
    /// A JSON file (a registry file) does not parse.
    Json {
        /// The file name or another label of the input.
        file: String,
        /// The JSON parser's error.
        source: ParseError,
    },
    /// A size or count exceeds a documented limit `[ROB-02]`.
    Limit {
        /// What was measured ("sections in a document").
        what: &'static str,
        /// The limit in force.
        limit: u64,
    },
    /// The input has a format version this build cannot read, or a migration chain has a gap.
    Version {
        /// What was being read ("document", "journal", "blob").
        format: &'static str,
        /// The version found.
        found: u64,
        /// The newest version this build reads.
        supported: u64,
    },
    /// A value of a JSON file (a registry file) has the wrong shape.
    Access {
        /// The file name or another label of the input.
        file: String,
        /// The typed-access error, with the path inside the file.
        source: AccessError,
    },
    /// A registry entry is inconsistent (range order, symmetry partner, duplicate id ...).
    Registry {
        /// The parameter id, or the file name when the problem is not one entry's.
        id: String,
        /// What is wrong.
        what: String,
    },
    /// A document section or entry has a wrong or missing value.
    Field {
        /// The section, as `kind` or `kind key`.
        section: String,
        /// The entry key ("" when the section as a whole is wrong).
        key: String,
        /// What is wrong.
        what: String,
    },
    /// A parameter id is not in the registry.
    UnknownParameter(String),
    /// A parameter value does not fit its registry entry (kind, range, choices).
    ParamValue {
        /// The parameter id.
        id: String,
        /// What is wrong.
        what: String,
    },
    /// An edit does not match the text it is applied to: a value or section is not what the edit
    /// expects (the target changed since the edit was made).
    Conflict {
        /// The section, as `kind` or `kind key`.
        section: String,
        /// The entry key ("" for a whole-section edit).
        key: String,
        /// What did not match.
        what: &'static str,
    },
    /// A blob file is damaged, has the wrong hash, or a referenced blob is missing.
    Blob {
        /// The blob's content hash.
        hash: Hash128,
        /// What is wrong.
        what: &'static str,
    },
    /// The command journal is damaged in a place that is not its torn tail.
    Journal {
        /// The line, counting from 1.
        line: usize,
        /// What is wrong.
        what: &'static str,
    },
    /// A migration from one format version to the next failed.
    Migration {
        /// The version the migration starts from.
        from: u32,
        /// What is wrong.
        what: String,
    },
    /// A codec of `humanart-io` (JSON writer, LZ) failed.
    Codec(humanart_io::Error),
    /// A file-system operation failed.
    Platform {
        /// The path of the file or directory.
        path: String,
        /// The platform's error.
        source: PlatformError,
    },
}

impl DocError {
    /// A [`DocError::Platform`] for `path`.
    pub fn platform(path: &str, source: PlatformError) -> Self {
        Self::Platform { path: path.to_owned(), source }
    }

    /// A [`DocError::Field`].
    pub fn field(section: impl Into<String>, key: impl Into<String>, what: impl Into<String>) -> Self {
        Self::Field { section: section.into(), key: key.into(), what: what.into() }
    }

    /// A [`DocError::Registry`].
    pub fn registry(id: impl Into<String>, what: impl Into<String>) -> Self {
        Self::Registry { id: id.into(), what: what.into() }
    }

    /// A [`DocError::ParamValue`].
    pub fn param_value(id: impl Into<String>, what: impl Into<String>) -> Self {
        Self::ParamValue { id: id.into(), what: what.into() }
    }
}

impl fmt::Display for DocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax { line, what } => write!(f, "line {line}: {what}"),
            Self::Value { line, source } => write!(f, "line {line}: bad value: {source}"),
            Self::Json { file, source } => write!(f, "{file}: {source}"),
            Self::Limit { what, limit } => write!(f, "too many {what} (limit {limit})"),
            Self::Version { format, found, supported } => {
                write!(f, "{format} version {found} is not supported (this build reads up to {supported})")
            }
            Self::Access { file, source } => write!(f, "{file}: {source}"),
            Self::Registry { id, what } => write!(f, "parameter registry: {id}: {what}"),
            Self::Field { section, key, what } => {
                if key.is_empty() {
                    write!(f, "[{section}]: {what}")
                } else {
                    write!(f, "[{section}] {key}: {what}")
                }
            }
            Self::UnknownParameter(id) => write!(f, "unknown parameter {id}"),
            Self::ParamValue { id, what } => write!(f, "parameter {id}: {what}"),
            Self::Conflict { section, key, what } => {
                if key.is_empty() {
                    write!(f, "edit conflict at [{section}]: {what}")
                } else {
                    write!(f, "edit conflict at [{section}] {key}: {what}")
                }
            }
            Self::Blob { hash, what } => write!(f, "blob {hash}: {what}"),
            Self::Journal { line, what } => write!(f, "journal line {line}: {what}"),
            Self::Migration { from, what } => write!(f, "migration from version {from}: {what}"),
            Self::Codec(error) => write!(f, "{error}"),
            Self::Platform { path, source } => write!(f, "{path}: {source}"),
        }
    }
}

impl std::error::Error for DocError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Value { source, .. } | Self::Json { source, .. } => Some(source),
            Self::Access { source, .. } => Some(source),
            Self::Codec(error) => Some(error),
            Self::Platform { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<humanart_io::Error> for DocError {
    fn from(error: humanart_io::Error) -> Self {
        Self::Codec(error)
    }
}

/// The result type of this crate.
pub type Result<T, E = DocError> = std::result::Result<T, E>;
