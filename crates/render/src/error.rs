//! The errors of the renderer.

use std::fmt;

use humanart_gpu::GpuError;
use humanart_shaders::ShaderError;

/// Everything that can go wrong building the frame graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// More passes or resources than the documented limits ([`crate::graph::MAX_PASSES`],
    /// [`crate::graph::MAX_RESOURCES`], [`crate::graph::MAX_ACCESSES`]).
    TooMany {
        /// What overflowed.
        what: &'static str,
        /// The limit.
        limit: usize,
    },
    /// A resource id that the builder did not hand out.
    UnknownResource {
        /// The pass that named it.
        pass: &'static str,
    },
    /// A pass reads a transient resource that no earlier pass wrote.
    ReadBeforeWrite {
        /// The reading pass.
        pass: &'static str,
        /// The resource.
        resource: &'static str,
    },
    /// A resource with a zero dimension.
    EmptyResource {
        /// The resource.
        resource: &'static str,
    },
    /// An output that no pass writes.
    OutputNeverWritten {
        /// The resource.
        resource: &'static str,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooMany { what, limit } => write!(f, "frame graph: more than {limit} {what}"),
            Self::UnknownResource { pass } => write!(f, "frame graph: pass '{pass}' names an unknown resource"),
            Self::ReadBeforeWrite { pass, resource } => {
                write!(f, "frame graph: pass '{pass}' reads '{resource}' before any pass writes it")
            }
            Self::EmptyResource { resource } => write!(f, "frame graph: resource '{resource}' has a zero dimension"),
            Self::OutputNeverWritten { resource } => write!(f, "frame graph: output '{resource}' is never written"),
        }
    }
}

impl std::error::Error for GraphError {}

/// Everything that can go wrong rendering.
#[derive(Debug)]
pub enum RenderError {
    /// The device, a resource or a pipeline failed.
    Gpu(GpuError),
    /// A shader did not compose.
    Shader(ShaderError),
    /// The frame graph is inconsistent.
    Graph(GraphError),
    /// Reading or writing an image failed.
    Io(humanart_io::Error),
    /// The skin appearance system on the GPU refused an operation (`crate::skin`).
    Skin(crate::skin::SkinGpuError),
    /// The hair passes refused an operation (`crate::hair`).
    Hair(crate::hair::HairError),
    /// A parameter is out of its documented range.
    InvalidParameter {
        /// Which parameter and why.
        what: &'static str,
    },
    /// A size or count above a documented limit.
    TooLarge {
        /// What.
        what: &'static str,
        /// The value asked for.
        value: u64,
        /// The limit.
        limit: u64,
    },
    /// The scene's mesh data is inconsistent (index out of range, mismatched array lengths).
    InvalidMesh {
        /// What is wrong.
        what: &'static str,
    },
    /// The GPU did not finish a blocking wait within its limit (a slow, shared or hung GPU).
    GpuTimeout {
        /// What was waited for.
        what: &'static str,
        /// The limit in milliseconds.
        limit_ms: u64,
    },
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu(e) => write!(f, "render: {e}"),
            Self::Shader(e) => write!(f, "render: shader: {e}"),
            Self::Graph(e) => write!(f, "render: {e}"),
            Self::Io(e) => write!(f, "render: image: {e}"),
            Self::Skin(e) => write!(f, "render: {e}"),
            Self::Hair(e) => write!(f, "render: {e}"),
            Self::InvalidParameter { what } => write!(f, "render: invalid parameter: {what}"),
            Self::TooLarge { what, value, limit } => write!(f, "render: {what} is {value}, above the limit {limit}"),
            Self::InvalidMesh { what } => write!(f, "render: invalid mesh: {what}"),
            Self::GpuTimeout { what, limit_ms } => {
                write!(f, "render: the GPU did not finish the {what} within {limit_ms} ms")
            }
        }
    }
}

impl std::error::Error for RenderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Gpu(e) => Some(e),
            Self::Shader(e) => Some(e),
            Self::Graph(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::Skin(e) => Some(e),
            Self::Hair(e) => Some(e),
            Self::InvalidParameter { .. }
            | Self::TooLarge { .. }
            | Self::InvalidMesh { .. }
            | Self::GpuTimeout { .. } => None,
        }
    }
}

impl From<GpuError> for RenderError {
    fn from(e: GpuError) -> Self {
        Self::Gpu(e)
    }
}

impl From<ShaderError> for RenderError {
    fn from(e: ShaderError) -> Self {
        Self::Shader(e)
    }
}

impl From<GraphError> for RenderError {
    fn from(e: GraphError) -> Self {
        Self::Graph(e)
    }
}

impl From<crate::skin::SkinGpuError> for RenderError {
    fn from(e: crate::skin::SkinGpuError) -> Self {
        Self::Skin(e)
    }
}

impl From<crate::hair::HairError> for RenderError {
    fn from(e: crate::hair::HairError) -> Self {
        Self::Hair(e)
    }
}

impl From<humanart_io::Error> for RenderError {
    fn from(e: humanart_io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_display_and_chain() {
        let e = RenderError::from(GraphError::ReadBeforeWrite { pass: "taa", resource: "history" });
        assert!(e.to_string().contains("'taa' reads 'history'"));
        assert!(std::error::Error::source(&e).is_some());
        let e = RenderError::TooLarge { what: "tile", value: 9000, limit: 8192 };
        assert!(e.to_string().contains("9000"));
        assert!(std::error::Error::source(&e).is_none());
        let e = RenderError::GpuTimeout { what: "texture readback", limit_ms: 60_000 };
        assert!(e.to_string().contains("texture readback within 60000 ms"));
    }
}
