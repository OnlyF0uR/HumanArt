//! [`HairError`], the error type of the hair renderer.

use std::fmt;

use humanart_gpu::GpuError;
use humanart_shaders::ShaderError;

use super::bsdf::BadFibre;
use crate::groom::GroomGpuError;

/// Why the hair renderer could not be built, configured or run.
#[derive(Debug)]
pub enum HairError {
    /// The device refused a resource.
    Gpu(GpuError),
    /// A hair shader did not compose.
    Shader(ShaderError),
    /// A fibre setting is out of range.
    Fibre(BadFibre),
    /// The groom's GPU data is unusable.
    Groom(GroomGpuError),
    /// A size exceeds a limit.
    TooLarge {
        /// What was measured.
        what: &'static str,
        /// Its value.
        value: u64,
        /// The limit.
        limit: u64,
    },
    /// An argument breaks a rule of the API.
    Invalid {
        /// What is wrong.
        what: &'static str,
    },
}

impl fmt::Display for HairError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu(e) => write!(f, "hair: {e}"),
            Self::Shader(e) => write!(f, "hair: shader: {e}"),
            Self::Fibre(e) => write!(f, "hair: {e}"),
            Self::Groom(e) => write!(f, "hair: {e}"),
            Self::TooLarge { what, value, limit } => write!(f, "hair: {what} is {value}, above the limit {limit}"),
            Self::Invalid { what } => write!(f, "hair: {what}"),
        }
    }
}

impl std::error::Error for HairError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gpu(e) => Some(e),
            Self::Shader(e) => Some(e),
            Self::Fibre(e) => Some(e),
            Self::Groom(e) => Some(e),
            Self::TooLarge { .. } | Self::Invalid { .. } => None,
        }
    }
}

impl From<GpuError> for HairError {
    fn from(e: GpuError) -> Self {
        Self::Gpu(e)
    }
}

impl From<ShaderError> for HairError {
    fn from(e: ShaderError) -> Self {
        Self::Shader(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_display_and_chain() {
        let e = HairError::TooLarge { what: "grooms", value: 17, limit: 16 };
        assert!(e.to_string().contains("above the limit 16"));
        assert!(std::error::Error::source(&e).is_none());
        let e = HairError::Fibre(BadFibre { what: "x" });
        assert!(std::error::Error::source(&e).is_some());
    }
}
