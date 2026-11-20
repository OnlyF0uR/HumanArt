//! Hexadecimal colour notation: `#RRGGBB`, `#RRGGBBAA`, `#RGB` and `#RGBA` `[COL-03]`.
//!
//! The text form of a display colour in the picker, in documents and in the agent protocol. The
//! digits are 8-bit sRGB-encoded channels (alpha is a plain 0–255 coverage); the `*_linear` helpers
//! convert to and from the linear working space through [`super::transfer`].
//!
//! Parsing accepts an optional leading `#`, either letter case, the three- and four-digit short forms
//! of CSS (each digit doubled: `#f80` is `#ff8800`) and ASCII white space around the text. A form
//! without alpha is opaque. Nothing else is accepted, and the input is checked against
//! [`MAX_TEXT_LEN`] before it is looked at, so hostile text costs nothing `[ROB-01]`. Formatting
//! always writes the long lower-case form with the `#`, without allocating.

use super::transfer;
use std::fmt;

/// The longest text [`parse_hex`] looks at, in bytes: generous for `#RRGGBBAA` (9 bytes) with
/// surrounding white space. Longer input is rejected unread.
pub const MAX_TEXT_LEN: usize = 64;

/// Why a text is not a hexadecimal colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HexColorError {
    /// The text is longer than [`MAX_TEXT_LEN`] bytes.
    TooLong {
        /// The length of the text in bytes.
        len: usize,
    },
    /// There are no digits (the text is empty, white space, or a lone `#`).
    Empty,
    /// The number of digits is not 3, 4, 6 or 8.
    DigitCount {
        /// The number of bytes after the optional `#`.
        count: usize,
    },
    /// A character is not a hexadecimal digit.
    Digit {
        /// The zero-based position of the offending byte among the digits.
        index: usize,
        /// The offending byte.
        byte: u8,
    },
}

impl fmt::Display for HexColorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::TooLong { len } => {
                write!(f, "a hex colour is at most {MAX_TEXT_LEN} bytes of text, got {len}")
            }
            Self::Empty => f.write_str("a hex colour needs digits, for example #ff8800"),
            Self::DigitCount { count } => {
                write!(f, "a hex colour has 3, 4, 6 or 8 digits, got {count}")
            }
            Self::Digit { index, byte } => {
                if byte.is_ascii_graphic() {
                    write!(f, "'{}' at digit {} is not a hexadecimal digit", char::from(byte), index + 1)
                } else {
                    write!(f, "byte 0x{byte:02x} at digit {} is not a hexadecimal digit", index + 1)
                }
            }
        }
    }
}

impl std::error::Error for HexColorError {}

/// The value of one hexadecimal digit.
#[inline]
const fn digit_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Parses a hexadecimal colour to 8-bit `[r, g, b, a]` (sRGB-encoded channels, alpha 255 when the
/// text has none). See the module documentation for the accepted forms.
pub fn parse_hex(text: &str) -> Result<[u8; 4], HexColorError> {
    parse_hex_bytes(text.as_bytes())
}

/// [`parse_hex`] for text that has not been checked to be UTF-8 (every accepted form is ASCII).
pub fn parse_hex_bytes(text: &[u8]) -> Result<[u8; 4], HexColorError> {
    if text.len() > MAX_TEXT_LEN {
        return Err(HexColorError::TooLong { len: text.len() });
    }
    let trimmed = text.trim_ascii();
    let digits = trimmed.strip_prefix(b"#").unwrap_or(trimmed);
    if digits.is_empty() {
        return Err(HexColorError::Empty);
    }
    let count = digits.len();
    if !matches!(count, 3 | 4 | 6 | 8) {
        return Err(HexColorError::DigitCount { count });
    }
    // At most eight nibbles, as checked above.
    let mut nibbles = [0_u8; 8];
    for (index, (slot, &byte)) in nibbles.iter_mut().zip(digits).enumerate() {
        *slot = digit_value(byte).ok_or(HexColorError::Digit { index, byte })?;
    }
    let mut out = [0, 0, 0, 255];
    if count <= 4 {
        for (channel, &nibble) in out.iter_mut().zip(&nibbles[..count]) {
            *channel = nibble * 17;
        }
    } else {
        for (channel, pair) in out.iter_mut().zip(nibbles[..count].as_chunks::<2>().0) {
            *channel = pair[0] * 16 + pair[1];
        }
    }
    Ok(out)
}

/// Parses a hexadecimal colour to linear Rec.709 `[r, g, b, a]`: the colour channels go through the
/// sRGB decode, the alpha is `a / 255`.
pub fn parse_hex_linear(text: &str) -> Result<[f32; 4], HexColorError> {
    let [r, g, b, a] = parse_hex(text)?;
    let [r, g, b] = transfer::srgb8_to_linear_rgb([r, g, b]);
    Ok([r, g, b, transfer::unorm8_to_f32(a)])
}

/// The 8-bit sRGB `[r, g, b, a]` nearest to a linear Rec.709 colour with alpha, each channel clamped
/// to its range: what [`HexRgba`] prints for a linear colour.
pub const fn linear_to_srgb8_rgba(rgba: [f32; 4]) -> [u8; 4] {
    let [r, g, b] = transfer::linear_to_srgb8_rgb([rgba[0], rgba[1], rgba[2]]);
    [r, g, b, transfer::f32_to_unorm8(rgba[3])]
}

const DIGITS: &[u8; 16] = b"0123456789abcdef";

/// The ASCII text `#rrggbb` of an 8-bit colour.
pub const fn hex_rgb_bytes(rgb: [u8; 3]) -> [u8; 7] {
    let mut out = [b'#'; 7];
    let mut i = 0;
    while i < 3 {
        out[1 + 2 * i] = DIGITS[(rgb[i] >> 4) as usize];
        out[2 + 2 * i] = DIGITS[(rgb[i] & 15) as usize];
        i += 1;
    }
    out
}

/// The ASCII text `#rrggbbaa` of an 8-bit colour with alpha.
pub const fn hex_rgba_bytes(rgba: [u8; 4]) -> [u8; 9] {
    let mut out = [b'#'; 9];
    let mut i = 0;
    while i < 4 {
        out[1 + 2 * i] = DIGITS[(rgba[i] >> 4) as usize];
        out[2 + 2 * i] = DIGITS[(rgba[i] & 15) as usize];
        i += 1;
    }
    out
}

/// Writes ASCII bytes produced by this module to a formatter.
fn write_ascii(f: &mut fmt::Formatter<'_>, bytes: &[u8]) -> fmt::Result {
    for &byte in bytes {
        fmt::Write::write_char(f, char::from(byte))?;
    }
    Ok(())
}

/// An 8-bit sRGB colour that displays as `#rrggbb`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HexRgb(pub [u8; 3]);

impl HexRgb {
    /// The display colour nearest to a linear Rec.709 colour (clamped to the gamut).
    pub const fn from_linear(rgb: [f32; 3]) -> Self {
        Self(transfer::linear_to_srgb8_rgb(rgb))
    }
}

impl fmt::Display for HexRgb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_ascii(f, &hex_rgb_bytes(self.0))
    }
}

/// An 8-bit sRGB colour with alpha that displays as `#rrggbbaa`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HexRgba(pub [u8; 4]);

impl HexRgba {
    /// The display colour nearest to a linear Rec.709 colour with alpha (clamped to the gamut).
    pub const fn from_linear(rgba: [f32; 4]) -> Self {
        Self(linear_to_srgb8_rgba(rgba))
    }
}

impl fmt::Display for HexRgba {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_ascii(f, &hex_rgba_bytes(self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::fuzz;
    use humanart_core::rng::Rng;
    use std::time::Duration;

    #[test]
    fn parses_every_accepted_form() {
        assert_eq!(parse_hex("#ff8800"), Ok([255, 136, 0, 255]));
        assert_eq!(parse_hex("FF8800"), Ok([255, 136, 0, 255]));
        assert_eq!(parse_hex("#Ff880080"), Ok([255, 136, 0, 128]));
        assert_eq!(parse_hex("#f80"), Ok([255, 136, 0, 255]));
        assert_eq!(parse_hex("f808"), Ok([255, 136, 0, 136]));
        assert_eq!(parse_hex("  #000000\t\n"), Ok([0, 0, 0, 255]));
        assert_eq!(parse_hex("#abcdef12"), Ok([0xab, 0xcd, 0xef, 0x12]));
        assert_eq!(parse_hex("#ABCDEF12"), Ok([0xab, 0xcd, 0xef, 0x12]));
        assert_eq!(parse_hex_bytes(b"#123"), Ok([0x11, 0x22, 0x33, 255]));
    }

    #[test]
    fn rejects_everything_else_with_a_reason() {
        assert_eq!(parse_hex(""), Err(HexColorError::Empty));
        assert_eq!(parse_hex("   "), Err(HexColorError::Empty));
        assert_eq!(parse_hex("#"), Err(HexColorError::Empty));
        for (text, count) in [("#1", 1), ("12", 2), ("#12345", 5), ("1234567", 7), ("#123456789", 9), ("##123", 4)] {
            if text == "##123" {
                // Four bytes after the first '#', the second '#' is then a bad digit.
                assert_eq!(parse_hex(text), Err(HexColorError::Digit { index: 0, byte: b'#' }));
            } else {
                assert_eq!(parse_hex(text), Err(HexColorError::DigitCount { count }));
            }
        }
        assert_eq!(parse_hex("#ff88zz"), Err(HexColorError::Digit { index: 4, byte: b'z' }));
        assert_eq!(parse_hex("#ff 800"), Err(HexColorError::Digit { index: 2, byte: b' ' }));
        assert_eq!(parse_hex("#ffé8"), Err(HexColorError::DigitCount { count: 5 }));
        assert_eq!(parse_hex("-ff"), Err(HexColorError::Digit { index: 0, byte: b'-' }));
        let long = "f".repeat(MAX_TEXT_LEN + 1);
        assert_eq!(parse_hex(&long), Err(HexColorError::TooLong { len: MAX_TEXT_LEN + 1 }));
        // Every error explains itself.
        for error in [
            HexColorError::TooLong { len: 99 },
            HexColorError::Empty,
            HexColorError::DigitCount { count: 5 },
            HexColorError::Digit { index: 4, byte: b'z' },
            HexColorError::Digit { index: 0, byte: 0x07 },
        ] {
            let message = error.to_string();
            assert!(!message.is_empty() && message.is_ascii(), "{message}");
        }
        assert!(HexColorError::Digit { index: 4, byte: b'z' }.to_string().contains("'z' at digit 5"));
        assert!(HexColorError::Digit { index: 0, byte: 0x07 }.to_string().contains("0x07"));
    }

    #[test]
    fn formatting_round_trips_through_parsing() {
        assert_eq!(HexRgb([255, 136, 0]).to_string(), "#ff8800");
        assert_eq!(HexRgba([255, 136, 0, 128]).to_string(), "#ff880080");
        assert_eq!(&hex_rgb_bytes([0, 1, 254]), b"#0001fe");
        assert_eq!(&hex_rgba_bytes([0xab, 0xcd, 0xef, 0x09]), b"#abcdef09");
        let mut rng = Rng::new(0x4E8);
        for _ in 0..5000 {
            let c = [rng.below(256) as u8, rng.below(256) as u8, rng.below(256) as u8, rng.below(256) as u8];
            assert_eq!(parse_hex(&HexRgba(c).to_string()), Ok(c));
            assert_eq!(parse_hex(&HexRgb([c[0], c[1], c[2]]).to_string()), Ok([c[0], c[1], c[2], 255]));
            assert_eq!(parse_hex(&HexRgba(c).to_string().to_uppercase()), Ok(c));
        }
    }

    #[test]
    fn linear_conversions_use_the_srgb_curve() {
        let linear = parse_hex_linear("#ffffff80").unwrap();
        assert_eq!(linear, [1.0, 1.0, 1.0, 128.0 / 255.0]);
        let grey = parse_hex_linear("#808080").unwrap();
        assert!((f64::from(grey[0]) - 0.215_860_5).abs() < 1e-6);
        assert_eq!(grey[3], 1.0);
        assert_eq!(HexRgba::from_linear(grey).to_string(), "#808080ff");
        assert_eq!(HexRgb::from_linear([grey[0], grey[1], grey[2]]).to_string(), "#808080");
        // Out-of-gamut and non-finite values clamp.
        assert_eq!(linear_to_srgb8_rgba([2.0, -1.0, f32::NAN, 7.0]), [255, 0, 0, 255]);
        assert!(parse_hex_linear("#xyz").is_err());
        let mut rng = Rng::new(0x11E);
        for _ in 0..2000 {
            let c = [rng.below(256) as u8, rng.below(256) as u8, rng.below(256) as u8, rng.below(256) as u8];
            let linear = parse_hex_linear(&HexRgba(c).to_string()).unwrap();
            assert_eq!(linear_to_srgb8_rgba(linear), c);
        }
    }

    #[test]
    fn fuzz_rob_01_hex_parser_survives_mutated_input() {
        let corpus: Vec<Vec<u8>> =
            ["#ff8800", "#FF880080", "#f80", "f808", "  #000000  ", "", "#", "#12345", "\u{e9}#ff"]
                .iter()
                .map(|text| text.as_bytes().to_vec())
                .collect();
        fuzz::run("hex-colour", &corpus, 20_000, 0xC01_0125, Duration::from_secs(60), |bytes| {
            let parsed = parse_hex_bytes(bytes);
            if let Ok(text) = std::str::from_utf8(bytes) {
                assert_eq!(parse_hex(text), parsed);
            }
            if let Ok(rgba) = parsed {
                // What parses prints back to something that parses to the same colour.
                assert_eq!(parse_hex_bytes(&hex_rgba_bytes(rgba)), Ok(rgba));
            }
        });
    }
}
