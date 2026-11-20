//! The error type of the hair system `[ERR-01]`.

use std::fmt;

/// Why a groom, guide set, map or `.anthair` file was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum HairError {
    /// A structure breaks a rule of the groom data model ("guide 3: root triangle out of range").
    Invalid {
        /// The kind of item that is wrong ("guide", "strand", "map").
        item: &'static str,
        /// Its index in its table.
        index: u64,
        /// What is wrong, as a short phrase.
        what: &'static str,
    },
    /// A size or count exceeds a documented limit `[ROB-02]`.
    Limit {
        /// What was measured ("strand count").
        what: &'static str,
        /// The limit in force.
        limit: u64,
        /// The value found.
        got: u64,
    },
    /// A generation parameter is unknown or outside its range.
    Param {
        /// The parameter key ("curl.radius").
        name: String,
        /// What is wrong.
        what: &'static str,
    },
    /// The bytes of an `.anthair` file are malformed.
    Format {
        /// The byte offset of the problem.
        offset: u64,
        /// What is wrong, as a short phrase.
        what: &'static str,
    },
    /// A compressed chunk of an `.anthair` file could not be expanded.
    Codec(humanart_io::Error),
}

impl HairError {
    /// Shorthand for [`HairError::Invalid`].
    #[must_use]
    pub const fn invalid(item: &'static str, index: usize, what: &'static str) -> Self {
        Self::Invalid { item, index: index as u64, what }
    }

    /// Shorthand for [`HairError::Limit`].
    #[must_use]
    pub const fn limit(what: &'static str, limit: u64, got: u64) -> Self {
        Self::Limit { what, limit, got }
    }

    /// Shorthand for [`HairError::Format`].
    #[must_use]
    pub const fn format(offset: usize, what: &'static str) -> Self {
        Self::Format { offset: offset as u64, what }
    }
}

impl fmt::Display for HairError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { item, index, what } => write!(f, "{item} {index}: {what}"),
            Self::Limit { what, limit, got } => write!(f, "{what} is {got}, above the limit of {limit}"),
            Self::Param { name, what } => write!(f, "groom parameter {name:?}: {what}"),
            Self::Format { offset, what } => write!(f, "hair file at byte {offset}: {what}"),
            Self::Codec(e) => write!(f, "hair file chunk: {e}"),
        }
    }
}

impl std::error::Error for HairError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Codec(e) => Some(e),
            _ => None,
        }
    }
}

impl From<humanart_io::Error> for HairError {
    fn from(e: humanart_io::Error) -> Self {
        Self::Codec(e)
    }
}

/// The result type of the hair system.
pub type HairResult<T> = Result<T, HairError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_display_and_chain() {
        let e = HairError::invalid("guide", 3, "root triangle out of range");
        assert_eq!(e.to_string(), "guide 3: root triangle out of range");
        assert!(std::error::Error::source(&e).is_none());
        let e = HairError::limit("strand count", 10, 11);
        assert!(e.to_string().contains("above the limit of 10"));
        let e = HairError::Param { name: "curl.radius".into(), what: "out of range" };
        assert!(e.to_string().contains("curl.radius"));
        assert!(HairError::format(4, "bad magic").to_string().contains("byte 4"));
    }
}
