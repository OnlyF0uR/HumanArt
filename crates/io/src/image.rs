//! The decoded image every image codec produces and consumes, and the size limits they enforce.

use crate::error::{Error, Format, Result};

/// How the values of a texture or image are to be interpreted `[COL-01]`. Mandatory on every image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    /// Colour encoded with the sRGB transfer function (Rec.709 primaries).
    Srgb,
    /// Colour in linear Rec.709.
    Linear,
    /// A tangent-space or object-space normal map: vectors, never colour-converted.
    Normal,
    /// Any other data (masks, roughness, height, ids): never colour-converted.
    Data,
}

/// The sample storage of an [`Image`]: interleaved channels, rows top to bottom.
#[derive(Clone, Debug, PartialEq)]
pub enum Samples {
    /// 8-bit unsigned normalised.
    U8(Vec<u8>),
    /// 16-bit unsigned normalised.
    U16(Vec<u16>),
    /// 16-bit IEEE half floats, as their bit patterns.
    F16(Vec<u16>),
    /// 32-bit floats.
    F32(Vec<f32>),
}

impl Samples {
    /// The number of samples (pixels × channels).
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::U16(v) | Self::F16(v) => v.len(),
            Self::F32(v) => v.len(),
        }
    }

    /// Whether there are no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes per sample.
    #[must_use]
    pub const fn sample_size(&self) -> usize {
        match self {
            Self::U8(_) => 1,
            Self::U16(_) | Self::F16(_) => 2,
            Self::F32(_) => 4,
        }
    }
}

/// A decoded raster image: `width × height` pixels of `channels` interleaved samples, first row at
/// the top.
///
/// Channel meaning by count: 1 = grey (or a single data channel), 2 = grey + alpha, 3 = RGB,
/// 4 = RGBA. Alpha is straight (not premultiplied).
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    /// Width in pixels (at least 1).
    pub width: u32,
    /// Height in pixels (at least 1).
    pub height: u32,
    /// Samples per pixel, 1 to 4.
    pub channels: u8,
    /// How the values are to be interpreted.
    pub color_space: ColorSpace,
    /// The samples.
    pub samples: Samples,
}

impl Image {
    /// Builds an image, checking that the sample count matches the dimensions.
    ///
    /// `format` labels the error (the codec that is building the image).
    pub fn new(
        format: Format,
        width: u32,
        height: u32,
        channels: u8,
        color_space: ColorSpace,
        samples: Samples,
    ) -> Result<Self> {
        if width == 0 || height == 0 || !(1..=4).contains(&channels) {
            return Err(Error::Encode { format, what: "image dimensions or channel count are out of range" });
        }
        let expected = (width as usize).checked_mul(height as usize).and_then(|p| p.checked_mul(channels as usize));
        if expected != Some(samples.len()) {
            return Err(Error::Encode { format, what: "sample count does not match the dimensions" });
        }
        Ok(Self { width, height, channels, color_space, samples })
    }

    /// The number of pixels.
    #[must_use]
    pub const fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Samples per row.
    #[must_use]
    pub const fn row_len(&self) -> usize {
        self.width as usize * self.channels as usize
    }

    /// The 8-bit samples, if that is how the image is stored.
    #[must_use]
    pub fn as_u8(&self) -> Option<&[u8]> {
        if let Samples::U8(v) = &self.samples { Some(v) } else { None }
    }

    /// The 16-bit unsigned samples, if that is how the image is stored.
    #[must_use]
    pub fn as_u16(&self) -> Option<&[u16]> {
        if let Samples::U16(v) = &self.samples { Some(v) } else { None }
    }

    /// The 32-bit float samples, if that is how the image is stored.
    #[must_use]
    pub fn as_f32(&self) -> Option<&[f32]> {
        if let Samples::F32(v) = &self.samples { Some(v) } else { None }
    }
}

/// Upper bounds a decoder enforces before it allocates `[ROB-02]`.
///
/// The defaults admit every asset the studio handles (16K textures, large scans) and nothing absurd:
/// a decompression bomb is refused from its header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Largest image width or height, in pixels.
    pub max_dimension: u32,
    /// Largest pixel count of one image.
    pub max_pixels: u64,
    /// Largest size in bytes of anything a decoder allocates for one input (decoded samples,
    /// decompressed streams, vertex arrays).
    pub max_bytes: u64,
    /// Largest element count of one array in a mesh, scene or document (vertices, faces, strands,
    /// JSON members).
    pub max_elements: u64,
    /// Deepest nesting of a recursive structure (JSON values, scene nodes).
    pub max_depth: u32,
}

impl Limits {
    /// The limits used when a caller does not choose: 32768 pixels a side, 2^29 pixels, 4 GiB of
    /// decoded data, 2^28 elements, 128 levels of nesting.
    pub const DEFAULT: Self =
        Self { max_dimension: 32_768, max_pixels: 1 << 29, max_bytes: 4 << 30, max_elements: 1 << 28, max_depth: 128 };

    /// Small limits for fuzz tests, so a mutated header cannot make a test allocate gigabytes.
    pub const FUZZ: Self =
        Self { max_dimension: 4096, max_pixels: 1 << 22, max_bytes: 64 << 20, max_elements: 1 << 20, max_depth: 64 };

    /// Checks image dimensions against the limits and returns the pixel count.
    pub fn check_image(&self, format: Format, width: u64, height: u64) -> Result<usize> {
        if width == 0 || height == 0 {
            return Err(Error::Invalid { format, offset: 0, what: "image has a zero dimension" });
        }
        let side = width.max(height);
        if side > u64::from(self.max_dimension) {
            return Err(Error::Limit {
                format,
                what: "image dimension",
                limit: u64::from(self.max_dimension),
                got: side,
            });
        }
        let pixels = width * height;
        if pixels > self.max_pixels {
            return Err(Error::Limit { format, what: "pixel count", limit: self.max_pixels, got: pixels });
        }
        Ok(pixels as usize)
    }

    /// Checks a byte size a decoder is about to allocate.
    pub fn check_bytes(&self, format: Format, what: &'static str, bytes: u64) -> Result<usize> {
        if bytes > self.max_bytes {
            return Err(Error::Limit { format, what, limit: self.max_bytes, got: bytes });
        }
        usize::try_from(bytes).map_err(|_| Error::Limit { format, what, limit: usize::MAX as u64, got: bytes })
    }

    /// Checks an element count a decoder is about to allocate for.
    pub fn check_elements(&self, format: Format, what: &'static str, count: u64) -> Result<usize> {
        if count > self.max_elements {
            return Err(Error::Limit { format, what, limit: self.max_elements, got: count });
        }
        Ok(count as usize)
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_checks_its_sample_count() {
        let ok = Image::new(Format::Png, 2, 3, 4, ColorSpace::Srgb, Samples::U8(vec![0; 24])).unwrap();
        assert_eq!(ok.pixel_count(), 6);
        assert_eq!(ok.row_len(), 8);
        assert!(ok.as_u8().is_some() && ok.as_u16().is_none() && ok.as_f32().is_none());
        assert!(Image::new(Format::Png, 2, 3, 4, ColorSpace::Srgb, Samples::U8(vec![0; 23])).is_err());
        assert!(Image::new(Format::Png, 0, 3, 4, ColorSpace::Srgb, Samples::U8(vec![])).is_err());
        assert!(Image::new(Format::Png, 1, 1, 5, ColorSpace::Data, Samples::F32(vec![0.0; 5])).is_err());
        assert_eq!(Samples::F16(vec![0; 3]).sample_size(), 2);
        assert!(Samples::F32(vec![]).is_empty());
    }

    #[test]
    fn limits_refuse_bombs_from_the_header() {
        let l = Limits::FUZZ;
        assert_eq!(l.check_image(Format::Png, 16, 16).unwrap(), 256);
        assert!(matches!(l.check_image(Format::Png, 0, 16), Err(Error::Invalid { .. })));
        assert!(matches!(l.check_image(Format::Png, 5000, 1), Err(Error::Limit { what: "image dimension", .. })));
        assert!(matches!(l.check_image(Format::Png, 4096, 4096), Err(Error::Limit { what: "pixel count", .. })));
        assert!(matches!(
            Limits::DEFAULT.check_image(Format::Png, u64::from(u32::MAX), u64::from(u32::MAX)),
            Err(Error::Limit { .. })
        ));
        assert!(l.check_bytes(Format::Exr, "decoded size", 1 << 40).is_err());
        assert_eq!(l.check_bytes(Format::Exr, "decoded size", 10).unwrap(), 10);
        assert!(l.check_elements(Format::Obj, "vertex count", 1 << 21).is_err());
        assert_eq!(Limits::default(), Limits::DEFAULT);
    }
}
