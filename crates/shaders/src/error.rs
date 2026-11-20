//! The errors of shader composition and validation.

use std::fmt;

/// A place in a shader source file: the path under `shaders/` and the 1-based line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    /// The file, relative to `shaders/` (`"lib/math.wgsl"`).
    pub file: String,
    /// The 1-based line in that file.
    pub line: u32,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.line)
    }
}

/// Everything that can go wrong composing or validating a shader.
#[derive(Debug)]
pub enum ShaderError {
    /// No embedded file has this path (an `#include` target or a requested shader).
    UnknownFile {
        /// The path asked for.
        path: String,
        /// The `#include` line that asked for it, `None` for the root file.
        included_from: Option<Location>,
    },
    /// The manifest has no shader with this path.
    UnknownShader {
        /// The path asked for.
        path: String,
    },
    /// A malformed directive: a bad `#include` path, a missing flag name, trailing text.
    BadDirective {
        /// Where.
        at: Location,
        /// What is wrong.
        message: &'static str,
    },
    /// A line starting with `#` that is not one of the known directives.
    UnknownDirective {
        /// Where.
        at: Location,
        /// The directive word.
        directive: String,
    },
    /// A file includes itself, directly or through others.
    IncludeCycle {
        /// The chain of files, the repeated one last.
        chain: Vec<String>,
    },
    /// Includes nested deeper than [`crate::preprocess::MAX_INCLUDE_DEPTH`].
    IncludeTooDeep {
        /// The `#include` line that went too deep.
        at: Location,
    },
    /// `#ifdef` nested deeper than [`crate::preprocess::MAX_CONDITIONAL_DEPTH`].
    ConditionalTooDeep {
        /// The `#ifdef`/`#ifndef` line that went too deep.
        at: Location,
    },
    /// `#else`/`#endif` without an open `#ifdef`, a second `#else`, or a file ending inside one.
    Unbalanced {
        /// Where.
        at: Location,
        /// What is wrong.
        message: &'static str,
    },
    /// The composed source grew beyond [`crate::preprocess::MAX_OUTPUT_BYTES`].
    OutputTooLarge {
        /// The root file being composed.
        root: String,
    },
    /// A capability flag passed to the preprocessor is not an identifier.
    BadFlag {
        /// The flag.
        flag: String,
    },
    /// naga rejected the composed source (parse or validation).
    Compile {
        /// The manifest path of the shader.
        shader: String,
        /// The source line the error points at, when naga gave a span.
        at: Option<Location>,
        /// The 1-based column on that line, 0 when unknown.
        column: u32,
        /// naga's message with its causes.
        message: String,
    },
    /// The manifest and the module disagree (entry points, stages, override declarations).
    Manifest {
        /// The manifest path of the shader.
        shader: String,
        /// What disagrees.
        message: String,
    },
    /// The device rejected a module or pipeline.
    Device {
        /// The manifest path of the shader.
        shader: String,
        /// The entry point(s) of the pipeline, empty for the module itself.
        entry_point: String,
        /// The override constants of the variant, as `NAME=value` pairs.
        constants: String,
        /// The wgpu error (boxed: it is large and rare).
        source: Box<humanart_gpu::GpuError>,
    },
}

impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownFile { path, included_from: Some(at) } => {
                write!(f, "{at}: no shader file \"{path}\" is embedded")
            }
            Self::UnknownFile { path, included_from: None } => write!(f, "no shader file \"{path}\" is embedded"),
            Self::UnknownShader { path } => write!(f, "the shader manifest has no shader \"{path}\""),
            Self::BadDirective { at, message } | Self::Unbalanced { at, message } => write!(f, "{at}: {message}"),
            Self::UnknownDirective { at, directive } => write!(f, "{at}: unknown directive #{directive}"),
            Self::IncludeCycle { chain } => write!(f, "include cycle: {}", chain.join(" -> ")),
            Self::IncludeTooDeep { at } => {
                write!(f, "{at}: includes nested deeper than {}", crate::preprocess::MAX_INCLUDE_DEPTH)
            }
            Self::ConditionalTooDeep { at } => {
                write!(f, "{at}: #ifdef nested deeper than {}", crate::preprocess::MAX_CONDITIONAL_DEPTH)
            }
            Self::OutputTooLarge { root } => {
                write!(f, "{root}: the composed source exceeds {} bytes", crate::preprocess::MAX_OUTPUT_BYTES)
            }
            Self::BadFlag { flag } => write!(f, "capability flag \"{flag}\" is not an identifier"),
            Self::Compile { shader, at: Some(at), column, message } => {
                write!(f, "{shader}: {at}:{column}: {message}")
            }
            Self::Compile { shader, at: None, message, .. } => write!(f, "{shader}: {message}"),
            Self::Manifest { shader, message } => write!(f, "{shader}: manifest mismatch: {message}"),
            Self::Device { shader, entry_point, constants, source } => {
                write!(f, "{shader} [{entry_point}] {{{constants}}}: the device rejected it: {source}")
            }
        }
    }
}

impl std::error::Error for ShaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Device { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_the_file_and_line() {
        let at = Location { file: "lib/math.wgsl".to_owned(), line: 12 };
        let error = ShaderError::UnknownDirective { at, directive: "define".to_owned() };
        assert_eq!(error.to_string(), "lib/math.wgsl:12: unknown directive #define");
        let cycle = ShaderError::IncludeCycle { chain: vec!["a.wgsl".into(), "b.wgsl".into(), "a.wgsl".into()] };
        assert_eq!(cycle.to_string(), "include cycle: a.wgsl -> b.wgsl -> a.wgsl");
        assert!(std::error::Error::source(&cycle).is_none());
    }
}
