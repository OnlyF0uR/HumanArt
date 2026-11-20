//! [`GroomGpuError`], the error type of the GPU hair generation.

use std::fmt;

use humanart_gpu::GpuError;
use humanart_model::hair::HairError;
use humanart_shaders::ShaderError;

/// Why a GPU groom could not be built or updated.
#[derive(Debug)]
pub enum GroomGpuError {
    /// The device refused a resource.
    Gpu(GpuError),
    /// The groom shader did not compose.
    Shader(ShaderError),
    /// The groom data is invalid.
    Hair(HairError),
    /// A size exceeds the capacity or a device limit.
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

impl fmt::Display for GroomGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu(e) => write!(f, "groom: {e}"),
            Self::Shader(e) => write!(f, "groom: shader: {e}"),
            Self::Hair(e) => write!(f, "groom: {e}"),
            Self::TooLarge { what, value, limit } => write!(f, "groom: {what} is {value}, above the limit {limit}"),
            Self::Invalid { what } => write!(f, "groom: {what}"),
        }
    }
}

impl std::error::Error for GroomGpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gpu(e) => Some(e),
            Self::Shader(e) => Some(e),
            Self::Hair(e) => Some(e),
            Self::TooLarge { .. } | Self::Invalid { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_display() {
        let e = GroomGpuError::TooLarge { what: "strand count", value: 3, limit: 2 };
        assert!(e.to_string().contains("above the limit 2"));
        assert!(std::error::Error::source(&e).is_none());
        let e = GroomGpuError::Hair(HairError::invalid("strand", 1, "bad"));
        assert!(std::error::Error::source(&e).is_some());
    }
}
