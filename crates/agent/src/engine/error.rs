//! The engine's error type.

use std::fmt;

use humanart_doc::command::CommandError;
use humanart_gpu::GpuError;
use humanart_model::ModelError;
use humanart_model::eval::EvalError;
use humanart_model::skin_appearance::error::SkinError;
use humanart_render::deform::DeformError;
use humanart_render::error::RenderError;
use humanart_tools::bake::BakeError;

/// What can go wrong in the engine.
#[derive(Debug)]
pub enum EngineError {
    /// The template could not be read or built.
    Model(ModelError),
    /// The evaluation refused its input.
    Eval(EvalError),
    /// The renderer failed.
    Render(RenderError),
    /// The GPU deformation failed.
    Deform(DeformError),
    /// The GPU failed.
    Gpu(GpuError),
    /// A command failed.
    Command(CommandError),
    /// The skin data (zone masks, micro-detail plan) could not be built.
    Skin(SkinError),
    /// The skin's thickness or occlusion could not be baked.
    Bake(BakeError),
    /// A request did not fit the engine's state.
    Invalid {
        /// What was wrong.
        what: &'static str,
    },
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(e) => write!(f, "template: {e}"),
            Self::Eval(e) => write!(f, "evaluation: {e}"),
            Self::Render(e) => write!(f, "rendering: {e}"),
            Self::Deform(e) => write!(f, "deformation: {e}"),
            Self::Gpu(e) => write!(f, "GPU: {e}"),
            Self::Command(e) => write!(f, "command: {e}"),
            Self::Skin(e) => write!(f, "skin: {e}"),
            Self::Bake(e) => write!(f, "skin bake: {e}"),
            Self::Invalid { what } => write!(f, "invalid request: {what}"),
        }
    }
}

impl std::error::Error for EngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(e) => Some(e),
            Self::Eval(e) => Some(e),
            Self::Render(e) => Some(e),
            Self::Deform(e) => Some(e),
            Self::Gpu(e) => Some(e),
            Self::Command(e) => Some(e),
            Self::Skin(e) => Some(e),
            Self::Bake(e) => Some(e),
            Self::Invalid { .. } => None,
        }
    }
}

impl From<ModelError> for EngineError {
    fn from(e: ModelError) -> Self {
        Self::Model(e)
    }
}

impl From<EvalError> for EngineError {
    fn from(e: EvalError) -> Self {
        Self::Eval(e)
    }
}

impl From<RenderError> for EngineError {
    fn from(e: RenderError) -> Self {
        Self::Render(e)
    }
}

impl From<DeformError> for EngineError {
    fn from(e: DeformError) -> Self {
        Self::Deform(e)
    }
}

impl From<GpuError> for EngineError {
    fn from(e: GpuError) -> Self {
        Self::Gpu(e)
    }
}

impl From<SkinError> for EngineError {
    fn from(e: SkinError) -> Self {
        Self::Skin(e)
    }
}

impl From<BakeError> for EngineError {
    fn from(e: BakeError) -> Self {
        Self::Bake(e)
    }
}

impl From<CommandError> for EngineError {
    fn from(e: CommandError) -> Self {
        Self::Command(e)
    }
}
