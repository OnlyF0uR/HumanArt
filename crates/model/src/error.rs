//! The error type of the model crate `[ERR-01]`.
//!
//! Building a template from parts, reading a `.antmpl` package and checking names all return
//! [`ModelError`]. The quality checks of [`crate::validate`] never fail: they return a report.

use std::fmt;

use humanart_io::json::AccessError;

/// Why a template structure or package was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelError {
    /// A structure breaks a rule of the data model ("face 12: vertex index out of range").
    Invalid {
        /// The kind of item that is wrong ("face", "landmark", "delta component").
        item: &'static str,
        /// Its index in its table.
        index: u64,
        /// What is wrong, as a short phrase.
        what: &'static str,
    },
    /// A size or count exceeds a documented limit `[ROB-02]`.
    Limit {
        /// What was measured ("vertex count").
        what: &'static str,
        /// The limit in force.
        limit: u64,
        /// The value found.
        got: u64,
    },
    /// A stable name breaks the naming rules or is used twice in one table.
    Name {
        /// The table the name belongs to ("region", "joint").
        table: &'static str,
        /// The offending name (cut to 64 bytes).
        name: String,
        /// What is wrong with it.
        what: &'static str,
    },
    /// The bytes of a package are malformed (truncated, bad checksum, bad compressed stream).
    Codec(humanart_io::Error),
    /// The JSON metadata section has the wrong shape.
    Metadata(AccessError),
    /// An error inside one package section, with the section's tag for context.
    Section {
        /// The four-byte tag of the section ("MESH").
        tag: [u8; 4],
        /// The error found while decoding it.
        source: Box<ModelError>,
    },
}

impl ModelError {
    /// Shorthand for [`ModelError::Invalid`].
    #[must_use]
    pub const fn invalid(item: &'static str, index: usize, what: &'static str) -> Self {
        Self::Invalid { item, index: index as u64, what }
    }

    /// Shorthand for [`ModelError::Limit`].
    #[must_use]
    pub const fn limit(what: &'static str, limit: u64, got: u64) -> Self {
        Self::Limit { what, limit, got }
    }

    /// Wraps `self` with the section tag it was found in.
    #[must_use]
    pub fn in_section(self, tag: [u8; 4]) -> Self {
        Self::Section { tag, source: Box::new(self) }
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { item, index, what } => write!(f, "{item} {index}: {what}"),
            Self::Limit { what, limit, got } => write!(f, "{what} is {got}, over the limit of {limit}"),
            Self::Name { table, name, what } => write!(f, "{table} name {name:?}: {what}"),
            Self::Codec(e) => write!(f, "template package: {e}"),
            Self::Metadata(e) => write!(f, "template metadata: {e}"),
            Self::Section { tag, source } => {
                write!(f, "section {}: {source}", String::from_utf8_lossy(tag))
            }
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Codec(e) => Some(e),
            Self::Metadata(e) => Some(e),
            Self::Section { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

impl From<humanart_io::Error> for ModelError {
    fn from(e: humanart_io::Error) -> Self {
        Self::Codec(e)
    }
}

impl From<AccessError> for ModelError {
    fn from(e: AccessError) -> Self {
        Self::Metadata(e)
    }
}

/// The result of a model operation.
pub type Result<T> = std::result::Result<T, ModelError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn display_and_source() {
        let e = ModelError::invalid("face", 12, "vertex index out of range");
        assert_eq!(e.to_string(), "face 12: vertex index out of range");
        assert!(e.source().is_none());
        let wrapped = e.in_section(*b"MESH");
        assert_eq!(wrapped.to_string(), "section MESH: face 12: vertex index out of range");
        assert!(wrapped.source().is_some());
        let codec = ModelError::from(humanart_io::Error::Truncated { format: humanart_io::Format::Chunk, offset: 4 });
        assert!(codec.to_string().contains("byte 4"));
        assert!(codec.source().is_some());
        assert_eq!(ModelError::limit("vertex count", 10, 11).to_string(), "vertex count is 11, over the limit of 10");
    }
}
