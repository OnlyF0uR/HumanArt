//! The error type of the authored identity build (WP-06, `[ID-07]`) `[ERR-01]`.

use std::fmt;

use crate::error::ModelError;
use crate::expression::error::ExprError;

/// Why the identity deltas could not be built for a template.
#[derive(Clone, Debug, PartialEq)]
pub enum IdentityError {
    /// The template lacks something a recipe names (a landmark, joint, region or mesh part).
    Missing {
        /// What kind of thing was looked up ("landmark", "joint").
        what: &'static str,
        /// The name that was not found (cut to 64 bytes).
        name: String,
    },
    /// A structure breaks a rule ("component 3: displacement is not finite").
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
    /// An error of the shared surface context (`expression::field`).
    Field(ExprError),
}

/// Shorthand for results of this module.
pub type IdResult<T> = std::result::Result<T, IdentityError>;

impl IdentityError {
    /// Shorthand for [`IdentityError::Missing`]; long names are cut to 64 bytes.
    #[must_use]
    pub fn missing(what: &'static str, name: &str) -> Self {
        let mut cut = name.len().min(64);
        while !name.is_char_boundary(cut) {
            cut -= 1;
        }
        Self::Missing { what, name: name[..cut].to_owned() }
    }

    /// Shorthand for [`IdentityError::Invalid`].
    #[must_use]
    pub const fn invalid(item: &'static str, index: usize, what: &'static str) -> Self {
        Self::Invalid { item, index, what }
    }
}

impl fmt::Display for IdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { what, name } => write!(f, "no {what} called '{name}'"),
            Self::Invalid { item, index, what } => write!(f, "{item} {index}: {what}"),
            Self::Model(e) => write!(f, "model: {e}"),
            Self::Field(e) => write!(f, "surface context: {e}"),
        }
    }
}

impl std::error::Error for IdentityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(e) => Some(e),
            Self::Field(e) => Some(e),
            Self::Missing { .. } | Self::Invalid { .. } => None,
        }
    }
}

impl From<ModelError> for IdentityError {
    fn from(e: ModelError) -> Self {
        Self::Model(e)
    }
}

/// For callers that report [`ModelError`] (the template generator): model errors pass through,
/// the others become an `Invalid` "identity recipe" error (the detail is lost; call the identity
/// functions directly to see it).
impl From<IdentityError> for ModelError {
    fn from(e: IdentityError) -> Self {
        match e {
            IdentityError::Model(m) => m,
            IdentityError::Missing { .. } => Self::invalid(
                "identity recipe",
                0,
                "a recipe names a landmark, joint, region or part the template lacks",
            ),
            IdentityError::Invalid { index, .. } => {
                Self::invalid("identity recipe", index, "an identity component is invalid")
            }
            IdentityError::Field(_) => Self::invalid("identity recipe", 0, "the surface context could not be built"),
        }
    }
}

impl From<ExprError> for IdentityError {
    fn from(e: ExprError) -> Self {
        Self::Field(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_source_and_cut() {
        let long = "é".repeat(40);
        let IdentityError::Missing { name, .. } = IdentityError::missing("landmark", &long) else {
            panic!("not a missing error");
        };
        assert!(name.len() <= 64 && long.starts_with(&name));
        let e = IdentityError::invalid("component", 3, "bad");
        assert_eq!(e.to_string(), "component 3: bad");
        assert!(std::error::Error::source(&e).is_none());
        let m = IdentityError::from(ModelError::invalid("face", 1, "x"));
        assert!(std::error::Error::source(&m).is_some());
        assert!(m.to_string().starts_with("model: "));
        assert_eq!(ModelError::from(m), ModelError::invalid("face", 1, "x"));
        assert!(matches!(ModelError::from(IdentityError::missing("joint", "x")), ModelError::Invalid { .. }));
        assert!(matches!(ModelError::from(e), ModelError::Invalid { index: 3, .. }));
    }
}
