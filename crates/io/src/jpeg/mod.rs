//! JPEG reading and writing, in-house `[IO-02]`.
//!
//! **Decoding** ([`decode`], [`decode_with`]) takes baseline and extended sequential (SOF0, SOF1)
//! and progressive (SOF2) Huffman-coded JPEG with 8-bit samples: grey, YCbCr, RGB, and CMYK / YCCK
//! (Adobe APP14), any sampling factors 1 to 4 (4:2:0, 4:2:2, 4:4:0, 4:1:1 and the unusual ones,
//! fractional ratios included), restart intervals, and the EXIF Orientation and ColorSpace tags.
//! The [`Image`](crate::Image) is 8-bit sRGB: grey as 1 channel, every other model as 3 (RGB).
//! Chroma is upsampled with a triangle filter and converted with libjpeg's fixed-point equations
//! (see `output.rs`); the IDCT is the AAN float transform, within 1 of the exact IDCT
//! (see `idct.rs`). By default the EXIF orientation is applied, so the image comes out upright;
//! [`DecodeOptions::apply_orientation`] turns that off and [`Info::exif`] reports the tag. An
//! embedded ICC profile is not interpreted (the studio manages only sRGB and linear); [`Info`]
//! records that one was present, and the EXIF ColorSpace value.
//!
//! Not supported, each refused with [`Error::Unsupported`](crate::Error::Unsupported): arithmetic
//! coding, 12-bit samples, lossless and hierarchical JPEG, 2-component frames, and a height left to
//! a DNL marker. The spec `[IO-02]` asks for none of them.
//!
//! Hostile files `[ROB-01]`: every segment length is checked against the input and the segment's
//! own structure, the frame's dimensions and the memory it needs against [`Limits`](crate::Limits)
//! before anything is allocated, Huffman tables are validated when defined, every coefficient index
//! and run against the block or spectral band, restart markers must come in order, and a scan whose
//! entropy-coded data ends early is [`Error::Truncated`](crate::Error::Truncated) (no partial
//! images). At most [`MAX_SCANS`] scans are decoded.
//!
//! **Encoding** ([`encode`], [`encode_with`]) writes baseline JFIF with the Annex K tables scaled to
//! an IJG quality, 4:4:4 or 4:2:0, optionally with restart intervals.

mod decoder;
mod encoder;
mod exif;
mod fdct;
mod huffman;
mod idct;
mod output;
mod tables;

pub use encoder::{encode, encode_with};
pub use exif::{Exif, MAX_IFD_ENTRIES};

use crate::error::{Format, Result};
use crate::image::{Image, Limits};

const FORMAT: Format = Format::Jpeg;

/// The most scans one file may have (progressive files use about 10).
pub const MAX_SCANS: usize = 256;

/// How the components of a JPEG encode colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorModel {
    /// One component: luminance.
    Grey,
    /// Three components, JFIF YCbCr.
    YCbCr,
    /// Three components stored as RGB (Adobe transform 0, or component ids `R`, `G`, `B`).
    Rgb,
    /// Four components, Adobe (inverted) CMYK.
    Cmyk,
    /// Four components, YCbCr + K (Adobe transform 2).
    Ycck,
}

/// What a JPEG's headers say, as [`read_info`] reports it (as of the first scan).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Info {
    /// Width in pixels as stored (before the orientation is applied).
    pub width: u32,
    /// Height in pixels as stored.
    pub height: u32,
    /// Number of components, 1, 3 or 4.
    pub components: u8,
    /// Whether the frame is progressive (SOF2).
    pub progressive: bool,
    /// How the components encode colour.
    pub color_model: ColorModel,
    /// Horizontal and vertical sampling factors of each component (unused entries are zero).
    pub sampling: [[u8; 2]; 4],
    /// MCUs per restart interval (0 = none) in force at the first scan.
    pub restart_interval: u16,
    /// Whether a JFIF APP0 segment is present.
    pub jfif: bool,
    /// The Adobe APP14 colour transform (0 = none, 1 = YCbCr, 2 = YCCK), if the segment is present.
    pub adobe_transform: Option<u8>,
    /// The EXIF tags the decoder reads (all `None` without an EXIF block).
    pub exif: Exif,
    /// Whether an ICC profile (APP2) is embedded.
    pub has_icc_profile: bool,
}

impl Info {
    /// The EXIF orientation, 1 (upright) when absent.
    pub fn orientation(&self) -> u8 {
        self.exif.orientation.unwrap_or(1)
    }

    /// Width and height of the decoded image once the orientation is applied.
    pub fn oriented_size(&self) -> (u32, u32) {
        if self.orientation() >= 5 { (self.height, self.width) } else { (self.width, self.height) }
    }
}

/// How [`decode_with`] decodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Size limits.
    pub limits: Limits,
    /// Whether to rotate / mirror the image upright as its EXIF orientation says.
    pub apply_orientation: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self { limits: Limits::DEFAULT, apply_orientation: true }
    }
}

/// Reads the headers up to the first scan, without decoding the image.
pub fn read_info(bytes: &[u8], limits: &Limits) -> Result<Info> {
    decoder::decode(bytes, limits, true).map(|(info, _)| info)
}

/// Decodes a JPEG into an upright 8-bit sRGB image (1 channel for grey, 3 otherwise).
pub fn decode(bytes: &[u8], limits: &Limits) -> Result<Image> {
    decode_with(bytes, &DecodeOptions { limits: *limits, apply_orientation: true }).map(|(_, image)| image)
}

/// Decodes a JPEG and reports its headers.
pub fn decode_with(bytes: &[u8], options: &DecodeOptions) -> Result<(Info, Image)> {
    let (info, frame) = decoder::decode(bytes, &options.limits, false)?;
    let Some(frame) = frame else {
        return Err(crate::error::Error::Invalid { format: FORMAT, offset: 0, what: "no frame" });
    };
    let image = output::to_image(&frame, info.color_model, &options.limits)?;
    drop(frame);
    let image = if options.apply_orientation { output::orient(image, info.orientation())? } else { image };
    Ok((info, image))
}

/// Chroma subsampling of the encoder.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Subsampling {
    /// Full-resolution chroma.
    S444,
    /// Chroma at half resolution in both directions (the usual photo setting).
    #[default]
    S420,
}

/// How [`encode_with`] encodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeOptions {
    /// IJG quality 1 to 100 (clamped): 50 uses the Annex K tables as printed, 100 all ones.
    pub quality: u8,
    /// Chroma subsampling (ignored for grey).
    pub subsampling: Subsampling,
    /// MCUs per restart interval, 0 for none.
    pub restart_interval: u16,
}

impl Default for EncodeOptions {
    /// Quality 90, 4:2:0, no restart intervals.
    fn default() -> Self {
        Self { quality: 90, subsampling: Subsampling::S420, restart_interval: 0 }
    }
}

#[cfg(test)]
mod tests;
