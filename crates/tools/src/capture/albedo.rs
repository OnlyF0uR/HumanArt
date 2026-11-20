//! A scan's colour texture as linear Rec.709 albedo at a working size `[WP-25]`.
//!
//! Photogrammetry textures are large (HSRD-100: 16384² sRGB PNG). [`Albedo::from_image`] converts a
//! decoded image (`humanart_io::Image`, 8- or 16-bit, grey to RGBA) to linear values through a
//! lookup table and box-filters it by a whole factor in linear space, so the result is at most
//! `max_side` texels a side. Alpha is ignored: scan textures are opaque. [`Albedo::sample`]
//! reads it bilinearly at a UV (origin top-left, V down).
//!
//! The values are what the scanner baked into the texture: photogrammetry albedo still holds
//! whatever lighting the capture rig left in it (see the data check in the agent's STATUS).

use humanart_io::{ColorSpace, Image, Samples};
use humanart_math::Vec2;
use humanart_math::color::transfer::srgb_to_linear_f32;

use super::{CaptureError, Result};

/// The largest working texture side.
pub const MAX_ALBEDO_SIDE: u32 = 16384;

/// Linear RGB texels, rows top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub struct Albedo {
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// `width × height` linear Rec.709 colours.
    pub texels: Vec<[f32; 3]>,
}

impl Albedo {
    /// The linear albedo of `image`, box-filtered by the smallest whole factor that brings both
    /// sides to at most `max_side` (at least 1, at most [`MAX_ALBEDO_SIDE`]).
    ///
    /// # Errors
    /// [`CaptureError::Invalid`] for a float image, a normal or data image, an image whose sample
    /// count does not match its size, or a `max_side` out of range.
    pub fn from_image(image: &Image, max_side: u32) -> Result<Self> {
        if max_side == 0 || max_side > MAX_ALBEDO_SIDE {
            return Err(CaptureError::invalid("the albedo working size is out of range"));
        }
        let channels = usize::from(image.channels);
        if !(1..=4).contains(&channels) || image.width == 0 || image.height == 0 {
            return Err(CaptureError::invalid("the albedo image has no pixels or a bad channel count"));
        }
        if image.samples.len() != image.pixel_count() * channels {
            return Err(CaptureError::invalid("the albedo image's sample count does not match its size"));
        }
        let srgb = match image.color_space {
            ColorSpace::Srgb => true,
            ColorSpace::Linear => false,
            ColorSpace::Normal | ColorSpace::Data => {
                return Err(CaptureError::invalid("the albedo image is a normal or data map"));
            }
        };
        let decode = |v: f32| if srgb { srgb_to_linear_f32(v) } else { v };
        let lut8: Vec<f32> = (0..=255_u16).map(|v| decode(f32::from(v) / 255.0)).collect();
        let factor = image.width.div_ceil(max_side).max(image.height.div_ceil(max_side)).max(1);
        let (w, h) = ((image.width / factor).max(1), (image.height / factor).max(1));
        let mut sums = vec![[0.0_f32; 3]; w as usize * h as usize];
        let filter = BoxFilter {
            src_w: image.width as usize,
            src_h: image.height as usize,
            w: w as usize,
            h: h as usize,
            factor: factor as usize,
            channels,
        };
        match &image.samples {
            Samples::U8(v) => filter.accumulate(&mut sums, |i| v.get(i).map_or(0.0, |&b| lut8[usize::from(b)])),
            Samples::U16(v) => {
                filter.accumulate(&mut sums, |i| v.get(i).map_or(0.0, |&b| decode(f32::from(b) / 65535.0)));
            }
            Samples::F16(_) | Samples::F32(_) => {
                return Err(CaptureError::invalid("the albedo image holds floating-point samples"));
            }
        }
        let inv = 1.0 / (factor * factor) as f32;
        let texels = sums.into_iter().map(|s| [s[0] * inv, s[1] * inv, s[2] * inv]).collect();
        Ok(Self { width: w, height: h, texels })
    }

    /// The bilinear sample at `uv` (origin top-left, V down; texel centres at half-integers),
    /// clamped to the edges. Black for an empty albedo.
    pub fn sample(&self, uv: Vec2) -> [f32; 3] {
        if self.texels.is_empty() || !uv.is_finite() {
            return [0.0; 3];
        }
        let x = (uv.x * self.width as f32 - 0.5).clamp(0.0, (self.width - 1) as f32);
        let y = (uv.y * self.height as f32 - 0.5).clamp(0.0, (self.height - 1) as f32);
        let (x0, y0) = (x.floor() as u32, y.floor() as u32);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let at = |x: u32, y: u32| self.texels.get((y * self.width + x) as usize).copied().unwrap_or([0.0; 3]);
        let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
        let mut out = [0.0; 3];
        for k in 0..3 {
            let top = a[k] + (b[k] - a[k]) * fx;
            let bottom = c[k] + (d[k] - c[k]) * fx;
            out[k] = top + (bottom - top) * fy;
        }
        out
    }
}

/// The geometry of a whole-factor box filter.
struct BoxFilter {
    src_w: usize,
    src_h: usize,
    w: usize,
    h: usize,
    factor: usize,
    channels: usize,
}

impl BoxFilter {
    /// Adds every source pixel's linear RGB (`read` maps a sample index to its linear value) to
    /// its destination texel's sum.
    fn accumulate(&self, sums: &mut [[f32; 3]], read: impl Fn(usize) -> f32) {
        for y in 0..(self.h * self.factor).min(self.src_h) {
            let row = (y / self.factor) * self.w;
            for x in 0..(self.w * self.factor).min(self.src_w) {
                let at = (y * self.src_w + x) * self.channels;
                let c = if self.channels < 3 {
                    let g = read(at);
                    [g, g, g]
                } else {
                    [read(at), read(at + 1), read(at + 2)]
                };
                if let Some(s) = sums.get_mut(row + x / self.factor) {
                    s[0] += c[0];
                    s[1] += c[1];
                    s[2] += c[2];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_io::Format;

    fn rgb8(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Image {
        let mut v = Vec::new();
        for y in 0..h {
            for x in 0..w {
                v.extend_from_slice(&f(x, y));
            }
        }
        Image::new(Format::Png, w, h, 3, ColorSpace::Srgb, Samples::U8(v)).unwrap()
    }

    #[test]
    fn wp25_albedo_is_linear_and_box_filtered_in_linear_space() {
        // A checker of black and white texels averages to 0.5 linear (not to sRGB 0.5 = 0.214).
        let image = rgb8(8, 4, |x, y| if (x + y) % 2 == 0 { [255; 3] } else { [0; 3] });
        let a = Albedo::from_image(&image, 2).unwrap();
        assert_eq!((a.width, a.height), (2, 1));
        for t in &a.texels {
            assert!((t[0] - 0.5).abs() < 1e-6, "{t:?}");
        }
        // Unfiltered, sRGB 128 is 0.2158 linear.
        let grey = rgb8(2, 2, |_, _| [128; 3]);
        let g = Albedo::from_image(&grey, 16).unwrap();
        assert_eq!((g.width, g.height), (2, 2));
        assert!((g.texels[0][1] - 0.2158605).abs() < 1e-4);
    }

    #[test]
    fn wp25_albedo_sampling_is_bilinear_with_v_down() {
        // Top row red, bottom row blue: v = 0 is the top.
        let image = rgb8(2, 2, |_, y| if y == 0 { [255, 0, 0] } else { [0, 0, 255] });
        let a = Albedo::from_image(&image, 2).unwrap();
        let top = a.sample(Vec2::new(0.5, 0.0));
        let mid = a.sample(Vec2::new(0.5, 0.5));
        assert!(top[0] > 0.99 && top[2] < 0.01, "{top:?}");
        assert!((mid[0] - 0.5).abs() < 1e-5 && (mid[2] - 0.5).abs() < 1e-5, "{mid:?}");
        assert_eq!(a.sample(Vec2::new(f32::NAN, 0.0)), [0.0; 3]);
    }

    #[test]
    fn wp25_albedo_rejects_data_maps_and_bad_sizes() {
        let mut image = rgb8(2, 2, |_, _| [1; 3]);
        assert!(Albedo::from_image(&image, 0).is_err());
        assert!(Albedo::from_image(&image, MAX_ALBEDO_SIDE + 1).is_err());
        image.color_space = ColorSpace::Normal;
        assert!(Albedo::from_image(&image, 4).is_err());
        image.color_space = ColorSpace::Srgb;
        image.samples = Samples::U8(vec![0; 5]);
        assert!(Albedo::from_image(&image, 4).is_err());
    }
}
