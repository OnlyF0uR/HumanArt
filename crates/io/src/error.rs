//! The one error type of every codec and container in this crate `[ERR-01]`.

use std::fmt;

/// Which format a codec error came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Format {
    /// JSON text.
    Json,
    /// A raw DEFLATE or zlib stream.
    Deflate,
    /// The in-house LZ block codec.
    Lz,
    /// PNG.
    Png,
    /// JPEG.
    Jpeg,
    /// Radiance HDR (RGBE).
    Hdr,
    /// OpenEXR.
    Exr,
    /// KTX2.
    Ktx2,
    /// A BC6H or BC7 block stream.
    Bc,
    /// glTF 2.0 JSON or a GLB container.
    Gltf,
    /// Wavefront OBJ or MTL.
    Obj,
    /// Stanford PLY.
    Ply,
    /// A TrueType / OpenType font.
    Ttf,
    /// An in-house chunked container (`.antmpl`, `.anchar`, `.anthair`).
    Chunk,
}

impl Format {
    /// The name used in messages.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Json => "JSON",
            Self::Deflate => "DEFLATE",
            Self::Lz => "LZ",
            Self::Png => "PNG",
            Self::Jpeg => "JPEG",
            Self::Hdr => "Radiance HDR",
            Self::Exr => "OpenEXR",
            Self::Ktx2 => "KTX2",
            Self::Bc => "BC texture block",
            Self::Gltf => "glTF",
            Self::Obj => "OBJ",
            Self::Ply => "PLY",
            Self::Ttf => "TrueType",
            Self::Chunk => "chunked container",
        }
    }
}

/// Why a codec refused its input or could not produce its output.
///
/// Every variant names the format; the ones about input carry the byte offset where the problem was
/// found, so a message can point into the file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The input ends before the data it promises.
    Truncated {
        /// The format being read.
        format: Format,
        /// The byte offset at which more input was needed.
        offset: usize,
    },
    /// The input breaks the format's rules.
    Invalid {
        /// The format being read.
        format: Format,
        /// The byte offset of the offending data.
        offset: usize,
        /// What is wrong, as a short phrase ("bad chunk CRC").
        what: &'static str,
    },
    /// The input is valid but uses a feature this codec does not implement (documented per codec).
    Unsupported {
        /// The format being read or written.
        format: Format,
        /// The feature ("arithmetic coding").
        what: &'static str,
    },
    /// A size, count or nesting depth exceeds the configured limit `[ROB-02]`.
    Limit {
        /// The format being read or written.
        format: Format,
        /// What was measured ("image width").
        what: &'static str,
        /// The limit in force.
        limit: u64,
        /// The value found.
        got: u64,
    },
    /// The input parsed, but its content is not what was asked of it (a PLY face index past the last
    /// vertex). There is no byte offset: the problem shows only once the parsed data is read, so the
    /// error names the item instead, when there is one.
    Content {
        /// The format being read.
        format: Format,
        /// The offending item: the element or record kind and its zero-based index (`("face", 12)`),
        /// or `None` when the problem concerns the data as a whole ("no vertex element").
        item: Option<(&'static str, usize)>,
        /// What is wrong, as a short phrase.
        what: &'static str,
    },
    /// The data handed to an encoder cannot be written in this format.
    Encode {
        /// The format being written.
        format: Format,
        /// What is wrong ("sample count does not match the dimensions").
        what: &'static str,
    },
}

impl Error {
    /// The format the error belongs to.
    #[must_use]
    pub const fn format(&self) -> Format {
        match self {
            Self::Truncated { format, .. }
            | Self::Invalid { format, .. }
            | Self::Unsupported { format, .. }
            | Self::Limit { format, .. }
            | Self::Content { format, .. }
            | Self::Encode { format, .. } => *format,
        }
    }

    /// The byte offset in the input, for the variants that have one.
    #[must_use]
    pub const fn offset(&self) -> Option<usize> {
        match self {
            Self::Truncated { offset, .. } | Self::Invalid { offset, .. } => Some(*offset),
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { format, offset } => write!(f, "{}: input ends early at byte {offset}", format.name()),
            Self::Invalid { format, offset, what } => write!(f, "{}: {what} at byte {offset}", format.name()),
            Self::Unsupported { format, what } => write!(f, "{}: {what} is not supported", format.name()),
            Self::Limit { format, what, limit, got } => {
                write!(f, "{}: {what} is {got}, over the limit of {limit}", format.name())
            }
            Self::Content { format, item: Some((kind, index)), what } => {
                write!(f, "{}: {kind} {index}: {what}", format.name())
            }
            Self::Content { format, item: None, what } => write!(f, "{}: {what}", format.name()),
            Self::Encode { format, what } => write!(f, "{}: cannot encode: {what}", format.name()),
        }
    }
}

impl std::error::Error for Error {}

/// The result of a codec operation.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_the_format_and_place() {
        let e = Error::Invalid { format: Format::Png, offset: 33, what: "bad chunk CRC" };
        assert_eq!(e.to_string(), "PNG: bad chunk CRC at byte 33");
        assert_eq!(e.offset(), Some(33));
        assert_eq!(e.format(), Format::Png);
        let e = Error::Limit { format: Format::Jpeg, what: "image width", limit: 10, got: 11 };
        assert_eq!(e.to_string(), "JPEG: image width is 11, over the limit of 10");
        assert_eq!(e.offset(), None);
        assert!(Error::Truncated { format: Format::Exr, offset: 4 }.to_string().contains("ends early"));
        assert!(
            Error::Unsupported { format: Format::Jpeg, what: "arithmetic coding" }
                .to_string()
                .contains("not supported")
        );
        assert!(Error::Encode { format: Format::Hdr, what: "x" }.to_string().contains("cannot encode"));
    }
}
