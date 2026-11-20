//! PNG reading and writing, in-house on [`crate::deflate`] `[IO-01]`.
//!
//! **Decoding** ([`decode`]) takes every colour type and bit depth of the specification (grey 1, 2,
//! 4, 8, 16; RGB 8, 16; indexed 1, 2, 4, 8; grey + alpha and RGBA 8, 16), Adam7 interlacing, and
//! tRNS. The [`Image`](crate::Image) it returns holds:
//!
//! - 16-bit images as [`Samples::U16`](crate::Samples::U16), everything else as [`Samples::U8`](crate::Samples::U8); grey of 1, 2 or 4 bits is
//!   scaled to the full 8-bit range (`v * 255 / (2^d - 1)`);
//! - grey as 1 channel, grey + alpha as 2, RGB and indexed as 3, RGBA as 4; a tRNS chunk adds an
//!   alpha channel to grey, RGB and indexed images (0 on the key colour, the maximum elsewhere).
//!
//! The colour space `[COL-01]` comes from the chunks before the image data, in the precedence the
//! PNG specification gives them: sRGB or iCCP mean [`ColorSpace::Srgb`] (an ICC profile is not
//! interpreted; the studio has no colour management beyond sRGB and linear, and [`Info`] records
//! that a profile was present); otherwise a gAMA of 1.0 (within 5%) means [`ColorSpace::Linear`];
//! otherwise, including no colour chunk at all, the data is sRGB. A `tEXt` chunk with the keyword
//! [`COLOR_SPACE_KEYWORD`] (written by [`encode`] for normal maps and data) overrides all of these.
//! cHRM is reported in [`Info`] but does not change the colour space.
//!
//! Hostile files `[ROB-01]`: every chunk length is checked against the input and the limit of the
//! specification (2^31 - 1), every CRC of a critical chunk is verified (an ancillary chunk with a
//! bad CRC is skipped, as libpng does), the image's dimensions and its decoded size are checked
//! against [`Limits`](crate::Limits) before anything is allocated, the decompressed stream must have exactly the
//! size the header implies, and at most [`MAX_CHUNKS`] chunks are read. Unknown ancillary chunks
//! (APNG frames among them) are skipped; an unknown critical chunk is [`Error::Unsupported`](crate::Error::Unsupported).
//!
//! **Encoding** ([`encode`], [`encode_with`]) writes 8- or 16-bit grey, grey + alpha, RGB and RGBA
//! (1 to 4 channels of [`Samples::U8`](crate::Samples::U8) or [`Samples::U16`](crate::Samples::U16)), never interlaced, with a filter chosen
//! per row by the minimum-sum-of-absolute-differences heuristic, and the colour space as chunks
//! [`decode`] maps back: sRGB (+ gAMA 1/2.2) for [`ColorSpace::Srgb`], gAMA 1.0 for the others,
//! plus the keyword text for [`ColorSpace::Normal`] and [`ColorSpace::Data`].

mod decode;
mod encode;
mod filter;

pub use decode::{decode, decode_with_info, read_info};
pub use encode::{encode, encode_with};

use crate::deflate::Level;
use crate::error::Format;
use crate::image::ColorSpace;

const FORMAT: Format = Format::Png;

/// The eight bytes every PNG file starts with.
pub const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];

/// The `tEXt` keyword whose value (`"srgb"`, `"linear"`, `"normal"` or `"data"`) states an image's
/// colour space explicitly.
pub const COLOR_SPACE_KEYWORD: &str = "HumanArt colour space";

/// The most chunks a decoder reads from one file (each costs at least 12 bytes of input, so this
/// bounds the work on a file made of empty ancillary chunks).
pub const MAX_CHUNKS: usize = 1 << 20;

/// The largest chunk length the PNG specification allows.
pub const MAX_CHUNK_LEN: usize = (1 << 31) - 1;

/// How a PNG stores its pixels (the IHDR colour type).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorType {
    /// Colour type 0: one grey sample.
    Grey,
    /// Colour type 2: red, green, blue.
    Rgb,
    /// Colour type 3: an index into the PLTE palette.
    Indexed,
    /// Colour type 4: grey and alpha.
    GreyAlpha,
    /// Colour type 6: red, green, blue, alpha.
    Rgba,
}

impl ColorType {
    /// The colour type's number in IHDR.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::Grey => 0,
            Self::Rgb => 2,
            Self::Indexed => 3,
            Self::GreyAlpha => 4,
            Self::Rgba => 6,
        }
    }

    /// The colour type with IHDR number `code`.
    #[must_use]
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Grey),
            2 => Some(Self::Rgb),
            3 => Some(Self::Indexed),
            4 => Some(Self::GreyAlpha),
            6 => Some(Self::Rgba),
            _ => None,
        }
    }

    /// Samples per pixel as stored in the file.
    #[must_use]
    pub const fn stored_channels(self) -> u8 {
        match self {
            Self::Grey | Self::Indexed => 1,
            Self::GreyAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    /// Whether `depth` is a legal bit depth for this colour type.
    #[must_use]
    pub const fn allows_depth(self, depth: u8) -> bool {
        match self {
            Self::Grey => matches!(depth, 1 | 2 | 4 | 8 | 16),
            Self::Indexed => matches!(depth, 1 | 2 | 4 | 8),
            Self::Rgb | Self::GreyAlpha | Self::Rgba => matches!(depth, 8 | 16),
        }
    }
}

/// What a PNG's header and colour chunks say, as [`read_info`] reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Info {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Bits per stored sample (1, 2, 4, 8 or 16).
    pub bit_depth: u8,
    /// How pixels are stored.
    pub color_type: ColorType,
    /// Whether the image is Adam7-interlaced.
    pub interlaced: bool,
    /// Whether a tRNS chunk adds transparency.
    pub has_transparency: bool,
    /// Channels of the decoded [`Image`](crate::Image) (1 to 4).
    pub channels: u8,
    /// The colour space the decoded image gets.
    pub color_space: ColorSpace,
    /// The sRGB chunk's rendering intent, if there is one.
    pub srgb_intent: Option<u8>,
    /// The gAMA chunk's value (gamma × 100 000), if there is one.
    pub gamma: Option<u32>,
    /// The cHRM chunk (white point, red, green, blue as x, y pairs × 100 000), if there is one.
    pub chromaticities: Option<[u32; 8]>,
    /// Whether an iCCP chunk (an embedded ICC profile) is present.
    pub has_icc_profile: bool,
}

/// What the encoder does beyond the image itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeOptions {
    /// The DEFLATE effort.
    pub level: Level,
    /// How rows are filtered.
    pub filter: FilterChoice,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self { level: Level::Default, filter: FilterChoice::Adaptive }
    }
}

/// The scanline filter the encoder uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FilterChoice {
    /// Per row, the filter whose output has the smallest sum of absolute values.
    #[default]
    Adaptive,
    /// Filter type 0 on every row.
    None,
    /// Filter type 1 on every row.
    Sub,
    /// Filter type 2 on every row.
    Up,
    /// Filter type 3 on every row.
    Average,
    /// Filter type 4 on every row.
    Paeth,
}

/// The name of a colour space in the [`COLOR_SPACE_KEYWORD`] text.
const fn color_space_name(space: ColorSpace) -> &'static str {
    match space {
        ColorSpace::Srgb => "srgb",
        ColorSpace::Linear => "linear",
        ColorSpace::Normal => "normal",
        ColorSpace::Data => "data",
    }
}

#[cfg(test)]
mod tests;
