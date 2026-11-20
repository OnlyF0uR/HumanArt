//! The command line's errors and their exit codes `[CLI-03]`.

use std::fmt;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use humanart_agent::engine::EngineError;

use crate::args::UsageError;

/// What can go wrong in a command, one exit code per class.
#[derive(Debug)]
pub enum CliError {
    /// The command line does not fit the command (exit code 2).
    Usage(UsageError),
    /// The registry, the template or the skin data could not be read or built (exit code 3).
    Data(String),
    /// No GPU adapter or device (exit code 4).
    Gpu(String),
    /// The engine or renderer failed (exit code 5).
    Render(EngineError),
    /// A file could not be read or written (exit code 6).
    Io {
        /// The file or folder.
        path: PathBuf,
        /// The cause.
        source: std::io::Error,
    },
}

impl CliError {
    /// An [`CliError::Io`] for `path`.
    pub fn io(path: &Path, source: std::io::Error) -> Self {
        Self::Io { path: path.to_path_buf(), source }
    }

    /// The process exit code of this error class.
    pub const fn exit_code(&self) -> u8 {
        match self {
            Self::Usage(_) => 2,
            Self::Data(_) => 3,
            Self::Gpu(_) => 4,
            Self::Render(_) => 5,
            Self::Io { .. } => 6,
        }
    }

    /// The error class name used in the JSON form.
    pub const fn class(&self) -> &'static str {
        match self {
            Self::Usage(_) => "usage",
            Self::Data(_) => "data",
            Self::Gpu(_) => "gpu",
            Self::Render(_) => "render",
            Self::Io { .. } => "io",
        }
    }

    /// The error as one line of JSON: `{"error":{"class":..,"code":..,"message":..}}`.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"error\":{{\"class\":\"{}\",\"code\":{},\"message\":\"{}\"}}}}",
            self.class(),
            self.exit_code(),
            json_escape(&self.to_string())
        )
    }
}

/// `text` as the inside of a JSON string.
fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(e) => write!(f, "usage: {e}"),
            Self::Data(what) => write!(f, "data: {what}"),
            Self::Gpu(what) => write!(f, "GPU: {what}"),
            Self::Render(e) => write!(f, "render: {e}"),
            Self::Io { path, source } => write!(f, "{}: {source}", path.display()),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Usage(e) => Some(e),
            Self::Render(e) => Some(e),
            Self::Io { source, .. } => Some(source),
            Self::Data(_) | Self::Gpu(_) => None,
        }
    }
}

impl From<UsageError> for CliError {
    fn from(e: UsageError) -> Self {
        Self::Usage(e)
    }
}

impl From<EngineError> for CliError {
    fn from(e: EngineError) -> Self {
        Self::Render(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_03_codes_are_distinct_and_json_is_escaped() {
        let errors = [
            CliError::Usage(UsageError("bad".into())),
            CliError::Data("d".into()),
            CliError::Gpu("g".into()),
            CliError::Render(EngineError::Invalid { what: "r" }),
            CliError::io(Path::new("x.png"), std::io::Error::other("denied")),
        ];
        let mut codes: Vec<u8> = errors.iter().map(CliError::exit_code).collect();
        codes.dedup();
        assert_eq!(codes, [2, 3, 4, 5, 6]);
        let json = CliError::Data("a \"quoted\"\npath\\x".into()).to_json();
        assert_eq!(json, r#"{"error":{"class":"data","code":3,"message":"data: a \"quoted\"\npath\\x"}}"#);
    }
}
