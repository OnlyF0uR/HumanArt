//! The error type of the UI toolkit `[ERR-01]`.
//!
//! Every fallible operation of the crate returns [`UiError`]: a bad theme file, a damaged baked font
//! (`.anfont`), a font the baker cannot read, a shader that does not compose and a GPU failure. The
//! variants keep the underlying error as their [`source`](std::error::Error::source) where there
//! is one.

use std::fmt;

/// Why a UI operation failed.
#[derive(Debug)]
pub enum UiError {
    /// A theme file is not valid JSON or names an unknown token or a value out of range `[UI-08]`.
    ThemeFile(String),
    /// A baked font file (`.anfont`) is damaged or exceeds a documented limit `[ROB-01]`.
    FontFile {
        /// The byte offset where the problem was found.
        offset: usize,
        /// What is wrong, as a short phrase.
        what: &'static str,
    },
    /// The baker was asked for something it cannot produce (an empty character set, an atlas over
    /// the size limit, a glyph size out of range).
    Bake(&'static str),
    /// A codec of `humanart-io` (TrueType, PNG) refused its input.
    Io(humanart_io::Error),
    /// A UI shader did not compose.
    Shader(humanart_shaders::ShaderError),
    /// The GPU refused an operation or was lost.
    Gpu(humanart_gpu::GpuError),
    /// A saved docking layout is not valid JSON, names an unknown field or panel, or exceeds a
    /// documented limit `[UI-05]` `[ROB-01]`.
    DockLayout(String),
    /// A keymap file is not valid JSON or names an unknown key, modifier or action `[UI-06]`.
    Keymap(String),
    /// A size handed to the painter is out of range (a zero-sized target, a texture over the limit).
    Size {
        /// What was measured.
        what: &'static str,
        /// The value found.
        got: u64,
        /// The limit in force.
        limit: u64,
    },
}

impl fmt::Display for UiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThemeFile(message) => write!(f, "theme file: {message}"),
            Self::FontFile { offset, what } => write!(f, "baked font: {what} at byte {offset}"),
            Self::Bake(what) => write!(f, "font bake: {what}"),
            Self::Io(e) => write!(f, "{e}"),
            Self::Shader(e) => write!(f, "UI shader: {e}"),
            Self::Gpu(e) => write!(f, "UI GPU: {e}"),
            Self::DockLayout(message) => write!(f, "docking layout: {message}"),
            Self::Keymap(message) => write!(f, "keymap: {message}"),
            Self::Size { what, got, limit } => write!(f, "{what} is {got}, outside the limit of {limit}"),
        }
    }
}

impl std::error::Error for UiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Shader(e) => Some(e),
            Self::Gpu(e) => Some(e),
            Self::ThemeFile(_)
            | Self::DockLayout(_)
            | Self::Keymap(_)
            | Self::FontFile { .. }
            | Self::Bake(_)
            | Self::Size { .. } => None,
        }
    }
}

impl From<humanart_io::Error> for UiError {
    fn from(e: humanart_io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<humanart_shaders::ShaderError> for UiError {
    fn from(e: humanart_shaders::ShaderError) -> Self {
        Self::Shader(e)
    }
}

impl From<humanart_gpu::GpuError> for UiError {
    fn from(e: humanart_gpu::GpuError) -> Self {
        Self::Gpu(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn messages_and_sources() {
        let e = UiError::FontFile { offset: 12, what: "glyph count over the limit" };
        assert_eq!(e.to_string(), "baked font: glyph count over the limit at byte 12");
        assert!(e.source().is_none());
        let io = humanart_io::Error::Truncated { format: humanart_io::Format::Ttf, offset: 4 };
        let e = UiError::from(io);
        assert!(e.source().is_some());
        assert!(e.to_string().contains("byte 4"));
    }
}
