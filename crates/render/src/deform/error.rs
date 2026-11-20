//! [`DeformError`], the error type of the GPU deformation.

use std::fmt;

use humanart_gpu::GpuError;
use humanart_model::eval::EvalError;
use humanart_shaders::ShaderError;

/// Why a GPU deformation could not be set up or updated.
#[derive(Debug)]
pub enum DeformError {
    /// The device, a buffer or a pipeline failed.
    Gpu(GpuError),
    /// A deformation shader did not compose.
    Shader(ShaderError),
    /// An input was refused by the evaluation's rules (lengths, finiteness, weight range).
    Eval(EvalError),
    /// A table of the template is larger than the format or the device allows.
    TooLarge {
        /// What.
        what: &'static str,
        /// Its size (count or bytes).
        value: u64,
        /// The limit.
        limit: u64,
    },
    /// The template's data is inconsistent with what the GPU tables need.
    Invalid {
        /// What is wrong.
        what: &'static str,
    },
}

impl fmt::Display for DeformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu(e) => write!(f, "deformation: {e}"),
            Self::Shader(e) => write!(f, "deformation: shader: {e}"),
            Self::Eval(e) => write!(f, "deformation: {e}"),
            Self::TooLarge { what, value, limit } => {
                write!(f, "deformation: {what} is {value}, above the limit {limit}")
            }
            Self::Invalid { what } => write!(f, "deformation: invalid template data: {what}"),
        }
    }
}

impl std::error::Error for DeformError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gpu(e) => Some(e),
            Self::Shader(e) => Some(e),
            Self::Eval(e) => Some(e),
            Self::TooLarge { .. } | Self::Invalid { .. } => None,
        }
    }
}

impl From<GpuError> for DeformError {
    fn from(e: GpuError) -> Self {
        Self::Gpu(e)
    }
}

impl From<ShaderError> for DeformError {
    fn from(e: ShaderError) -> Self {
        Self::Shader(e)
    }
}

impl From<EvalError> for DeformError {
    fn from(e: EvalError) -> Self {
        Self::Eval(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_display_and_chain() {
        let e = DeformError::TooLarge { what: "delta entries", value: 10, limit: 5 };
        assert_eq!(e.to_string(), "deformation: delta entries is 10, above the limit 5");
        assert!(std::error::Error::source(&e).is_none());
        let e = DeformError::from(GpuError::Invalid { what: "x" });
        assert!(std::error::Error::source(&e).is_some());
        assert!(DeformError::Invalid { what: "y" }.to_string().contains("invalid template data: y"));
    }
}
