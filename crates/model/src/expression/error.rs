//! The error type of the expression model build (WP-19) `[ERR-01]`.

use std::fmt;

use humanart_io::json::{AccessError, ParseError};

use crate::error::ModelError;
use crate::eval::error::EvalError;

/// Why an expression build, a data file or a control rig was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprError {
    /// The template lacks something a recipe or data file names (a landmark, joint, region, part,
    /// expression or control).
    Missing {
        /// What kind of thing was looked up ("landmark", "joint").
        what: &'static str,
        /// The name that was not found (cut to 64 bytes).
        name: String,
    },
    /// A value or structure breaks a rule ("corrective 3: needs 2 to 4 inputs").
    Invalid {
        /// The kind of item that is wrong.
        item: &'static str,
        /// Its index in its table.
        index: usize,
        /// What is wrong, as a short phrase.
        what: &'static str,
    },
    /// A data-model error from the template or the structures built for it.
    Model(ModelError),
    /// An evaluation or rig-logic error.
    Eval(EvalError),
    /// A JSON data file has the wrong shape.
    Json(AccessError),
    /// A JSON data file is not valid JSON.
    Parse(ParseError),
}

impl ExprError {
    /// Shorthand for [`ExprError::Missing`]; long names are cut to 64 bytes.
    #[must_use]
    pub fn missing(what: &'static str, name: &str) -> Self {
        let mut cut = name.len().min(64);
        while !name.is_char_boundary(cut) {
            cut -= 1;
        }
        Self::Missing { what, name: name[..cut].to_owned() }
    }

    /// Shorthand for [`ExprError::Invalid`].
    #[must_use]
    pub const fn invalid(item: &'static str, index: usize, what: &'static str) -> Self {
        Self::Invalid { item, index, what }
    }
}

impl fmt::Display for ExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { what, name } => write!(f, "no {what} called '{name}'"),
            Self::Invalid { item, index, what } => write!(f, "{item} {index}: {what}"),
            Self::Model(e) => write!(f, "model: {e}"),
            Self::Eval(e) => write!(f, "evaluation: {e}"),
            Self::Json(e) => write!(f, "data file: {e}"),
            Self::Parse(e) => write!(f, "data file: {e}"),
        }
    }
}

impl std::error::Error for ExprError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(e) => Some(e),
            Self::Eval(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::Parse(e) => Some(e),
            Self::Missing { .. } | Self::Invalid { .. } => None,
        }
    }
}

impl From<ModelError> for ExprError {
    fn from(e: ModelError) -> Self {
        Self::Model(e)
    }
}

impl From<EvalError> for ExprError {
    fn from(e: EvalError) -> Self {
        Self::Eval(e)
    }
}

impl From<AccessError> for ExprError {
    fn from(e: AccessError) -> Self {
        Self::Json(e)
    }
}

impl From<ParseError> for ExprError {
    fn from(e: ParseError) -> Self {
        Self::Parse(e)
    }
}

/// The result type of the expression build.
pub type ExprResult<T> = std::result::Result<T, ExprError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn messages_and_sources() {
        let e = ExprError::missing("landmark", "L.brow.09");
        assert_eq!(e.to_string(), "no landmark called 'L.brow.09'");
        assert!(e.source().is_none());
        let long = "x".repeat(100);
        assert!(matches!(ExprError::missing("joint", &long), ExprError::Missing { name, .. } if name.len() == 64));
        let m: ExprError = ModelError::invalid("face", 2, "bad").into();
        assert!(m.source().is_some());
        assert_eq!(
            ExprError::invalid("corrective", 3, "needs 2 to 4 inputs").to_string(),
            "corrective 3: needs 2 to 4 inputs"
        );
    }
}
