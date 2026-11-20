//! Groom tools `[TOOL-05]` `[HAIR-03]` and hair-card LODs `[HAIR-10]`.
//!
//! - [`ops`]: the guide-editing operations behind the editor's groom tools: guide placement,
//!   combing, cutting, clump and density painting, hairline editing, parting and cowlick
//!   placement, mirror. They change [`humanart_model::hair::Guides`], groom maps, masks or
//!   parameters in place and report what they touched; the editor wraps the result in a document
//!   command (`groom_guide_edit` with the guide blob of `humanart_model::hair::format`,
//!   `modify_groom` for parameters and maps) `[CMD-02]`.
//! - [`cards`]: baking the card maps of a card set into an atlas and writing card LODs as glTF.
//!
//! [`GroomToolError`] is the error of both.

pub mod cards;
pub mod ops;

use std::fmt;

use humanart_model::hair::HairError;

use crate::bake::BakeError;

/// Why a groom tool failed.
#[derive(Debug)]
pub enum GroomToolError {
    /// The request or its input is malformed.
    Invalid(&'static str),
    /// The hair data model refused it.
    Hair(HairError),
    /// The card baker failed.
    Bake(BakeError),
    /// Encoding an image or the glTF file failed.
    Io(humanart_io::Error),
}

impl fmt::Display for GroomToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(what) => write!(f, "groom tool: {what}"),
            Self::Hair(e) => write!(f, "groom tool: {e}"),
            Self::Bake(e) => write!(f, "groom tool: {e}"),
            Self::Io(e) => write!(f, "groom tool: {e}"),
        }
    }
}

impl std::error::Error for GroomToolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Invalid(_) => None,
            Self::Hair(e) => Some(e),
            Self::Bake(e) => Some(e),
            Self::Io(e) => Some(e),
        }
    }
}

impl From<HairError> for GroomToolError {
    fn from(e: HairError) -> Self {
        Self::Hair(e)
    }
}

impl From<BakeError> for GroomToolError {
    fn from(e: BakeError) -> Self {
        Self::Bake(e)
    }
}

impl From<humanart_io::Error> for GroomToolError {
    fn from(e: humanart_io::Error) -> Self {
        Self::Io(e)
    }
}

/// The result type of the groom tools.
pub type Result<T> = std::result::Result<T, GroomToolError>;

#[cfg(test)]
mod tests;
