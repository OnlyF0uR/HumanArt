//! The application's error type.

use std::fmt;

use humanart_agent::engine::EngineError;
use humanart_core::error::PlatformError;
use humanart_doc::DocError;
use humanart_doc::command::CommandError;
use humanart_gpu::GpuError;
use humanart_render::error::RenderError;
use humanart_ui::UiError;

use crate::prefs::PrefsError;

/// What can go wrong in the editor.
#[derive(Debug)]
pub enum AppError {
    /// The engine (template, evaluation, viewers).
    Engine(EngineError),
    /// The UI toolkit.
    Ui(UiError),
    /// The GPU.
    Gpu(GpuError),
    /// The renderer (captures).
    Render(RenderError),
    /// A command or the session.
    Command(CommandError),
    /// The document or registry.
    Doc(DocError),
    /// A file.
    Platform {
        /// The path.
        path: String,
        /// The cause.
        source: PlatformError,
    },
    /// The preferences.
    Prefs(PrefsError),
    /// The agent server could not be started (sandbox, entropy, socket).
    AgentServer(String),
    /// A request that does not fit the state.
    Invalid {
        /// What was wrong.
        what: &'static str,
    },
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Engine(e) => write!(f, "{e}"),
            Self::Ui(e) => write!(f, "UI: {e}"),
            Self::Gpu(e) => write!(f, "GPU: {e}"),
            Self::Render(e) => write!(f, "rendering: {e}"),
            Self::Command(e) => write!(f, "{e}"),
            Self::Doc(e) => write!(f, "{e}"),
            Self::Platform { path, source } => write!(f, "{path}: {source}"),
            Self::Prefs(e) => write!(f, "{e}"),
            Self::AgentServer(what) => write!(f, "agent server: {what}"),
            Self::Invalid { what } => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for AppError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Engine(e) => Some(e),
            Self::Ui(e) => Some(e),
            Self::Gpu(e) => Some(e),
            Self::Render(e) => Some(e),
            Self::Command(e) => Some(e),
            Self::Doc(e) => Some(e),
            Self::Platform { source, .. } => Some(source),
            Self::Prefs(e) => Some(e),
            Self::AgentServer(_) | Self::Invalid { .. } => None,
        }
    }
}

impl From<EngineError> for AppError {
    fn from(e: EngineError) -> Self {
        Self::Engine(e)
    }
}

impl From<UiError> for AppError {
    fn from(e: UiError) -> Self {
        Self::Ui(e)
    }
}

impl From<GpuError> for AppError {
    fn from(e: GpuError) -> Self {
        Self::Gpu(e)
    }
}

impl From<RenderError> for AppError {
    fn from(e: RenderError) -> Self {
        Self::Render(e)
    }
}

impl From<CommandError> for AppError {
    fn from(e: CommandError) -> Self {
        Self::Command(e)
    }
}

impl From<DocError> for AppError {
    fn from(e: DocError) -> Self {
        Self::Doc(e)
    }
}

impl From<PrefsError> for AppError {
    fn from(e: PrefsError) -> Self {
        Self::Prefs(e)
    }
}
