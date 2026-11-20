//! The error type of the skin appearance system `[ERR-01]`.

use std::fmt;

/// Why a skin appearance operation was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkinError {
    /// A value or structure breaks a rule ("a rectangle outside its tile").
    Invalid {
        /// What is wrong, as a short phrase.
        what: &'static str,
    },
    /// A size or count exceeds a documented limit `[ROB-02]`.
    Limit {
        /// What was measured.
        what: &'static str,
        /// The limit in force.
        limit: u64,
        /// The value found.
        got: u64,
    },
    /// Encoded layer content is malformed.
    Format {
        /// Byte offset where the problem was found.
        offset: u64,
        /// What is wrong.
        what: &'static str,
    },
}

impl SkinError {
    /// Shorthand for [`SkinError::Limit`].
    pub const fn limit(what: &'static str, limit: u64, got: u64) -> Self {
        Self::Limit { what, limit, got }
    }
}

impl fmt::Display for SkinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { what } => write!(f, "skin appearance: {what}"),
            Self::Limit { what, limit, got } => write!(f, "skin appearance: {what} is {got}, the limit is {limit}"),
            Self::Format { offset, what } => write!(f, "skin layer content at byte {offset}: {what}"),
        }
    }
}

impl std::error::Error for SkinError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        assert_eq!(
            SkinError::limit("layer count", 64, 65).to_string(),
            "skin appearance: layer count is 65, the limit is 64"
        );
        assert_eq!(
            SkinError::Format { offset: 3, what: "bad tag" }.to_string(),
            "skin layer content at byte 3: bad tag"
        );
    }
}
