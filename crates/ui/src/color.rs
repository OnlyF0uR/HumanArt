//! The UI colour: linear Rec.709 RGB with straight (not premultiplied) alpha.
//!
//! The project convention is that colours are linear unless a name says `srgb`. Design tokens are
//! written as sRGB hex codes (that is how designers and the reference application state them), so
//! [`Color::hex`] decodes sRGB to linear at compile time through the exact 8-bit table of
//! `humanart_math::color::transfer`. The UI shader encodes back to sRGB when the render target is
//! not an `*Srgb` format, so blending happens in display space as in the reference application.

use humanart_math::color::transfer::{linear_to_srgb8, srgb8_to_linear};

/// A colour in linear Rec.709 with straight alpha.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Color {
    /// Red, linear.
    pub r: f32,
    /// Green, linear.
    pub g: f32,
    /// Blue, linear.
    pub b: f32,
    /// Opacity, 0 to 1.
    pub a: f32,
}

impl Color {
    /// Fully transparent black.
    pub const TRANSPARENT: Self = Self::linear(0.0, 0.0, 0.0, 0.0);
    /// Opaque black.
    pub const BLACK: Self = Self::linear(0.0, 0.0, 0.0, 1.0);
    /// Opaque white.
    pub const WHITE: Self = Self::linear(1.0, 1.0, 1.0, 1.0);
    /// The 18 % linear mid grey of a neutral viewport surround (sRGB `#767676`).
    pub const MID_GREY: Self = Self::linear(0.18, 0.18, 0.18, 1.0);

    /// A colour from linear components.
    #[inline]
    pub const fn linear(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// An opaque colour from an sRGB hex code `0xRRGGBB`.
    #[inline]
    pub const fn hex(rgb: u32) -> Self {
        Self::hexa(rgb, 1.0)
    }

    /// A colour from an sRGB hex code `0xRRGGBB` and a straight alpha.
    #[inline]
    pub const fn hexa(rgb: u32, alpha: f32) -> Self {
        Self {
            r: srgb8_to_linear(((rgb >> 16) & 0xff) as u8),
            g: srgb8_to_linear(((rgb >> 8) & 0xff) as u8),
            b: srgb8_to_linear((rgb & 0xff) as u8),
            a: alpha,
        }
    }

    /// A colour from 8-bit sRGB components and an 8-bit alpha.
    #[inline]
    pub const fn srgb8(rgba: [u8; 4]) -> Self {
        Self {
            r: srgb8_to_linear(rgba[0]),
            g: srgb8_to_linear(rgba[1]),
            b: srgb8_to_linear(rgba[2]),
            a: rgba[3] as f32 / 255.0,
        }
    }

    /// The nearest 8-bit sRGB components and 8-bit alpha.
    #[inline]
    pub fn to_srgb8(self) -> [u8; 4] {
        [
            linear_to_srgb8(self.r),
            linear_to_srgb8(self.g),
            linear_to_srgb8(self.b),
            (self.a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        ]
    }

    /// The colour with its alpha replaced.
    #[inline]
    #[must_use]
    pub const fn with_alpha(self, a: f32) -> Self {
        Self { a, ..self }
    }

    /// The colour with its alpha multiplied by `factor`.
    #[inline]
    #[must_use]
    pub fn fade(self, factor: f32) -> Self {
        Self { a: self.a * factor, ..self }
    }

    /// Linear interpolation of every component: exactly `self` at `t = 0` and exactly `other` at
    /// `t = 1`.
    #[inline]
    #[must_use]
    pub fn mix(self, other: Self, t: f32) -> Self {
        let s = 1.0 - t;
        Self {
            r: self.r * s + other.r * t,
            g: self.g * s + other.g * t,
            b: self.b * s + other.b * t,
            a: self.a * s + other.a * t,
        }
    }

    /// Relative luminance Y (Rec.709 weights on linear components).
    #[inline]
    pub fn luminance(self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    /// The grey of the same luminance and alpha: no hue, no cast (the colour-neutral mode `[UI-08]`).
    #[inline]
    #[must_use]
    pub fn neutral(self) -> Self {
        if self.r == self.g && self.g == self.b {
            // Already a grey: keep it bit for bit, so the operation is idempotent.
            return self;
        }
        let y = self.luminance();
        Self::linear(y, y, y, self.a)
    }

    /// The WCAG contrast ratio between two opaque colours (1 to 21).
    #[inline]
    pub fn contrast(self, other: Self) -> f32 {
        let (a, b) = (self.luminance(), other.luminance());
        let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
        (hi + 0.05) / (lo + 0.05)
    }

    /// The components as an array `[r, g, b, a]`, linear.
    #[inline]
    pub const fn to_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Writes `#RRGGBB` (opaque) or `#RRGGBBAA` into `out` (sRGB).
    pub fn write_hex(self, out: &mut String) {
        const DIGITS: &[u8; 16] = b"0123456789ABCDEF";
        let c = self.to_srgb8();
        out.push('#');
        let n = if c[3] == 255 { 3 } else { 4 };
        for &v in &c[..n] {
            out.push(char::from(DIGITS[usize::from(v >> 4)]));
            out.push(char::from(DIGITS[usize::from(v & 15)]));
        }
    }

    /// Parses `#RGB`, `#RRGGBB` or `#RRGGBBAA` (sRGB, the leading `#` optional).
    pub fn parse_hex(text: &str) -> Option<Self> {
        humanart_math::color::hex::parse_hex(text).ok().map(Self::srgb8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips_through_srgb8() {
        for code in [0x121212_u32, 0xEDEDED, 0x7DB3FF, 0x000000, 0xFFFFFF, 0x2563EB] {
            let c = Color::hex(code);
            let back = c.to_srgb8();
            let expect = [(code >> 16) as u8, (code >> 8) as u8, code as u8, 255];
            assert_eq!(back, expect, "{code:06X}");
        }
        assert!((Color::hex(0x767676).r - 0.18).abs() < 0.003);
    }

    #[test]
    fn hex_text_round_trip() {
        let mut s = String::new();
        Color::hexa(0x2563EB, 0.16).write_hex(&mut s);
        assert_eq!(s, "#2563EB29");
        let back = Color::parse_hex(&s).map(Color::to_srgb8);
        assert_eq!(back, Some([0x25, 0x63, 0xEB, 0x29]));
        assert!(Color::parse_hex("#zz").is_none());
    }

    #[test]
    fn neutral_keeps_luminance_and_drops_hue() {
        let c = Color::hex(0xF26D6D).neutral();
        assert_eq!(c.r, c.g);
        assert_eq!(c.g, c.b);
        assert!((c.luminance() - Color::hex(0xF26D6D).luminance()).abs() < 1e-6);
    }

    #[test]
    fn contrast_of_black_and_white_is_21() {
        assert!((Color::BLACK.contrast(Color::WHITE) - 21.0).abs() < 1e-4);
        assert!((Color::WHITE.contrast(Color::WHITE) - 1.0).abs() < 1e-6);
    }
}
