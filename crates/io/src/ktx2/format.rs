//! The Vulkan formats a KTX2 file can carry here: block geometry, type size and the samples of the
//! Khronos Data Format Descriptor (DFD) that describes each one `[IO-04]`.
//!
//! Block-compressed payloads (BC, ETC2/EAC, ASTC) are carried as opaque blocks: this module knows
//! how large a block is and how many a level holds, never what is inside one (the compressors are
//! WP-28's). Packed formats with shared exponents and Basis Universal (`VK_FORMAT_UNDEFINED`) are not
//! in the table and are refused as unsupported.

/// The colour models of the Khronos Data Format specification used by the formats in the table.
pub mod model {
    /// Uncompressed RGB, signed/unsigned, depth, alpha (`KHR_DF_MODEL_RGBSDA`).
    pub const RGBSDA: u8 = 1;
    /// BC1 with or without alpha.
    pub const BC1A: u8 = 128;
    /// BC2.
    pub const BC2: u8 = 129;
    /// BC3.
    pub const BC3: u8 = 130;
    /// BC4.
    pub const BC4: u8 = 131;
    /// BC5.
    pub const BC5: u8 = 132;
    /// BC6H.
    pub const BC6H: u8 = 133;
    /// BC7.
    pub const BC7: u8 = 134;
    /// ETC2 and EAC.
    pub const ETC2: u8 = 161;
    /// ASTC.
    pub const ASTC: u8 = 162;
}

/// The transfer functions of the Khronos Data Format specification used here.
pub mod transfer {
    /// Linear values.
    pub const LINEAR: u8 = 1;
    /// The sRGB transfer function.
    pub const SRGB: u8 = 2;
}

/// The family of a format: how its texels are grouped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    /// One texel per block, channels in memory order.
    Uncompressed,
    /// BC1 to BC7 (S3TC, RGTC, BPTC), 4×4 blocks.
    Bc,
    /// ETC2 and EAC, 4×4 blocks.
    Etc2,
    /// ASTC LDR, 4×4 to 12×12 blocks.
    Astc,
}

/// How the bits of a channel are interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Numeric {
    /// Unsigned normalised.
    Unorm,
    /// Signed normalised.
    Snorm,
    /// Unsigned normalised colour with the sRGB transfer function (alpha stays linear).
    Srgb,
    /// Signed floating point.
    Sfloat,
    /// Unsigned floating point.
    Ufloat,
}

/// One sample of a format's DFD: which channel, at which bits of the block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SampleLayout {
    /// The channel id of the colour model (low four bits of `channelType`), e.g. 0 = red,
    /// 15 = alpha.
    pub channel: u8,
    /// The first bit of the sample in the block.
    pub bit_offset: u16,
    /// The number of bits (the DFD stores this minus one).
    pub bit_length: u8,
}

/// Everything this crate knows about one Vulkan format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FormatInfo {
    /// The `VkFormat` value.
    pub vk_format: u32,
    /// The Vulkan name without the `VK_FORMAT_` prefix.
    pub name: &'static str,
    /// How texels are grouped.
    pub family: Family,
    /// How channel bits are interpreted.
    pub numeric: Numeric,
    /// The DFD colour model.
    pub color_model: u8,
    /// Block width in texels (1 for uncompressed formats).
    pub block_width: u8,
    /// Block height in texels (1 for uncompressed formats).
    pub block_height: u8,
    /// Bytes per block (per texel for uncompressed formats).
    pub block_bytes: u8,
    /// The KTX2 `typeSize`: the size of the data type for endianness conversion (1 for
    /// block-compressed and 8-bit formats).
    pub type_size: u8,
    /// The DFD samples in order.
    pub samples: &'static [SampleLayout],
}

impl FormatInfo {
    /// Looks a format up by its `VkFormat` value.
    #[must_use]
    pub fn from_vk(vk_format: u32) -> Option<&'static Self> {
        FORMATS.iter().find(|f| f.vk_format == vk_format)
    }

    /// Whether colour channels use the sRGB transfer function.
    #[must_use]
    pub const fn is_srgb(&self) -> bool {
        matches!(self.numeric, Numeric::Srgb)
    }

    /// Whether the format is block-compressed.
    #[must_use]
    pub const fn is_compressed(&self) -> bool {
        !matches!(self.family, Family::Uncompressed)
    }

    /// The DFD transfer function this format implies.
    #[must_use]
    pub const fn transfer(&self) -> u8 {
        if self.is_srgb() { transfer::SRGB } else { transfer::LINEAR }
    }

    /// The number of blocks along one level dimension of `texels` texels (rounding up).
    #[must_use]
    pub const fn blocks(texels: u32, block: u8) -> u64 {
        (texels as u64).div_ceil(block as u64)
    }

    /// The byte size of one image (one layer, one face) of `width × height × depth` texels, or
    /// `None` on overflow.
    #[must_use]
    pub fn image_size(&self, width: u32, height: u32, depth: u32) -> Option<u64> {
        Self::blocks(width, self.block_width)
            .checked_mul(Self::blocks(height, self.block_height))?
            .checked_mul(u64::from(depth))?
            .checked_mul(u64::from(self.block_bytes))
    }

    /// The `channelType` byte (channel id plus qualifier bits) of `sample` in this format.
    #[must_use]
    pub const fn channel_type(&self, sample: SampleLayout) -> u8 {
        let qualifiers = match self.numeric {
            Numeric::Unorm => 0,
            Numeric::Snorm => SIGNED,
            // Alpha stays linear in an sRGB format.
            Numeric::Srgb => {
                if sample.channel == ALPHA {
                    LINEAR
                } else {
                    0
                }
            }
            Numeric::Sfloat => FLOAT | SIGNED,
            Numeric::Ufloat => FLOAT,
        };
        sample.channel | qualifiers
    }

    /// The `sampleLower` and `sampleUpper` words of `sample`, as KTX-Software writes them: the
    /// integer range of the bits for normalised uncompressed formats, the full 32-bit range for
    /// normalised block formats, and -1.0 (0.0 when unsigned) to 1.0 as `f32` bit patterns for
    /// float formats.
    #[must_use]
    pub const fn sample_bounds(&self, sample: SampleLayout) -> (u32, u32) {
        let compressed = self.is_compressed();
        match self.numeric {
            Numeric::Unorm | Numeric::Srgb => {
                if compressed || sample.bit_length >= 32 {
                    (0, u32::MAX)
                } else {
                    (0, (1_u32 << sample.bit_length) - 1)
                }
            }
            Numeric::Snorm => {
                if compressed || sample.bit_length >= 32 {
                    (0x8000_0000, 0x7FFF_FFFF)
                } else {
                    let max = (1_u32 << (sample.bit_length - 1)) - 1;
                    ((max as i32).wrapping_neg() as u32, max)
                }
            }
            Numeric::Sfloat => (0xBF80_0000, 0x3F80_0000),
            Numeric::Ufloat => (0, 0x3F80_0000),
        }
    }
}

/// The alpha channel id of every colour model used here.
pub const ALPHA: u8 = 15;
/// `channelType` qualifier: the sample is linear even though the transfer function is not.
pub const LINEAR: u8 = 0x10;
/// `channelType` qualifier: the sample is an exponent (unused by the formats here).
pub const EXPONENT: u8 = 0x20;
/// `channelType` qualifier: the sample is signed.
pub const SIGNED: u8 = 0x40;
/// `channelType` qualifier: the sample is floating point.
pub const FLOAT: u8 = 0x80;

const fn s(channel: u8, bit_offset: u16, bit_length: u8) -> SampleLayout {
    SampleLayout { channel, bit_offset, bit_length }
}

const R8: &[SampleLayout] = &[s(0, 0, 8)];
const RG8: &[SampleLayout] = &[s(0, 0, 8), s(1, 8, 8)];
const RGBA8: &[SampleLayout] = &[s(0, 0, 8), s(1, 8, 8), s(2, 16, 8), s(15, 24, 8)];
const BGRA8: &[SampleLayout] = &[s(2, 0, 8), s(1, 8, 8), s(0, 16, 8), s(15, 24, 8)];
const R16: &[SampleLayout] = &[s(0, 0, 16)];
const RG16: &[SampleLayout] = &[s(0, 0, 16), s(1, 16, 16)];
const RGBA16: &[SampleLayout] = &[s(0, 0, 16), s(1, 16, 16), s(2, 32, 16), s(15, 48, 16)];
const R32: &[SampleLayout] = &[s(0, 0, 32)];
const RG32: &[SampleLayout] = &[s(0, 0, 32), s(1, 32, 32)];
const RGBA32: &[SampleLayout] = &[s(0, 0, 32), s(1, 32, 32), s(2, 64, 32), s(15, 96, 32)];
/// One 64-bit sample (BC1 RGB, BC4, ETC2 RGB, EAC R11).
const ONE64: &[SampleLayout] = &[s(0, 0, 64)];
/// One 128-bit sample (BC6H, BC7, ASTC).
const ONE128: &[SampleLayout] = &[s(0, 0, 128)];
const BC1_RGBA: &[SampleLayout] = &[s(0, 0, 64), s(15, 0, 64)];
const ALPHA_COLOR: &[SampleLayout] = &[s(15, 0, 64), s(0, 64, 64)];
const TWO_CHANNEL: &[SampleLayout] = &[s(0, 0, 64), s(1, 64, 64)];
const ETC2_RGB: &[SampleLayout] = &[s(2, 0, 64)];
const ETC2_RGBA1: &[SampleLayout] = &[s(2, 0, 64), s(15, 0, 64)];
const ETC2_RGBA8: &[SampleLayout] = &[s(15, 0, 64), s(2, 64, 64)];

const fn plain(
    vk_format: u32,
    name: &'static str,
    numeric: Numeric,
    bytes: u8,
    type_size: u8,
    samples: &'static [SampleLayout],
) -> FormatInfo {
    FormatInfo {
        vk_format,
        name,
        family: Family::Uncompressed,
        numeric,
        color_model: model::RGBSDA,
        block_width: 1,
        block_height: 1,
        block_bytes: bytes,
        type_size,
        samples,
    }
}

const fn block(
    vk_format: u32,
    name: &'static str,
    family: Family,
    numeric: Numeric,
    color_model: u8,
    dims: (u8, u8),
    bytes: u8,
    samples: &'static [SampleLayout],
) -> FormatInfo {
    FormatInfo {
        vk_format,
        name,
        family,
        numeric,
        color_model,
        block_width: dims.0,
        block_height: dims.1,
        block_bytes: bytes,
        type_size: 1,
        samples,
    }
}

const fn astc(vk_format: u32, name: &'static str, numeric: Numeric, w: u8, h: u8) -> FormatInfo {
    block(vk_format, name, Family::Astc, numeric, model::ASTC, (w, h), 16, ONE128)
}

use Family::{Bc, Etc2};
use Numeric::{Sfloat, Snorm, Srgb, Ufloat, Unorm};

/// Every format this crate reads and writes, by `VkFormat` value.
pub const FORMATS: &[FormatInfo] = &[
    plain(9, "R8_UNORM", Unorm, 1, 1, R8),
    plain(15, "R8_SRGB", Srgb, 1, 1, R8),
    plain(16, "R8G8_UNORM", Unorm, 2, 1, RG8),
    plain(22, "R8G8_SRGB", Srgb, 2, 1, RG8),
    plain(37, "R8G8B8A8_UNORM", Unorm, 4, 1, RGBA8),
    plain(43, "R8G8B8A8_SRGB", Srgb, 4, 1, RGBA8),
    plain(44, "B8G8R8A8_UNORM", Unorm, 4, 1, BGRA8),
    plain(50, "B8G8R8A8_SRGB", Srgb, 4, 1, BGRA8),
    plain(70, "R16_UNORM", Unorm, 2, 2, R16),
    plain(76, "R16_SFLOAT", Sfloat, 2, 2, R16),
    plain(77, "R16G16_UNORM", Unorm, 4, 2, RG16),
    plain(83, "R16G16_SFLOAT", Sfloat, 4, 2, RG16),
    plain(91, "R16G16B16A16_UNORM", Unorm, 8, 2, RGBA16),
    plain(97, "R16G16B16A16_SFLOAT", Sfloat, 8, 2, RGBA16),
    plain(100, "R32_SFLOAT", Sfloat, 4, 4, R32),
    plain(103, "R32G32_SFLOAT", Sfloat, 8, 4, RG32),
    plain(109, "R32G32B32A32_SFLOAT", Sfloat, 16, 4, RGBA32),
    block(131, "BC1_RGB_UNORM_BLOCK", Bc, Unorm, model::BC1A, (4, 4), 8, ONE64),
    block(132, "BC1_RGB_SRGB_BLOCK", Bc, Srgb, model::BC1A, (4, 4), 8, ONE64),
    block(133, "BC1_RGBA_UNORM_BLOCK", Bc, Unorm, model::BC1A, (4, 4), 8, BC1_RGBA),
    block(134, "BC1_RGBA_SRGB_BLOCK", Bc, Srgb, model::BC1A, (4, 4), 8, BC1_RGBA),
    block(135, "BC2_UNORM_BLOCK", Bc, Unorm, model::BC2, (4, 4), 16, ALPHA_COLOR),
    block(136, "BC2_SRGB_BLOCK", Bc, Srgb, model::BC2, (4, 4), 16, ALPHA_COLOR),
    block(137, "BC3_UNORM_BLOCK", Bc, Unorm, model::BC3, (4, 4), 16, ALPHA_COLOR),
    block(138, "BC3_SRGB_BLOCK", Bc, Srgb, model::BC3, (4, 4), 16, ALPHA_COLOR),
    block(139, "BC4_UNORM_BLOCK", Bc, Unorm, model::BC4, (4, 4), 8, ONE64),
    block(140, "BC4_SNORM_BLOCK", Bc, Snorm, model::BC4, (4, 4), 8, ONE64),
    block(141, "BC5_UNORM_BLOCK", Bc, Unorm, model::BC5, (4, 4), 16, TWO_CHANNEL),
    block(142, "BC5_SNORM_BLOCK", Bc, Snorm, model::BC5, (4, 4), 16, TWO_CHANNEL),
    block(143, "BC6H_UFLOAT_BLOCK", Bc, Ufloat, model::BC6H, (4, 4), 16, ONE128),
    block(144, "BC6H_SFLOAT_BLOCK", Bc, Sfloat, model::BC6H, (4, 4), 16, ONE128),
    block(145, "BC7_UNORM_BLOCK", Bc, Unorm, model::BC7, (4, 4), 16, ONE128),
    block(146, "BC7_SRGB_BLOCK", Bc, Srgb, model::BC7, (4, 4), 16, ONE128),
    block(147, "ETC2_R8G8B8_UNORM_BLOCK", Etc2, Unorm, model::ETC2, (4, 4), 8, ETC2_RGB),
    block(148, "ETC2_R8G8B8_SRGB_BLOCK", Etc2, Srgb, model::ETC2, (4, 4), 8, ETC2_RGB),
    block(149, "ETC2_R8G8B8A1_UNORM_BLOCK", Etc2, Unorm, model::ETC2, (4, 4), 8, ETC2_RGBA1),
    block(150, "ETC2_R8G8B8A1_SRGB_BLOCK", Etc2, Srgb, model::ETC2, (4, 4), 8, ETC2_RGBA1),
    block(151, "ETC2_R8G8B8A8_UNORM_BLOCK", Etc2, Unorm, model::ETC2, (4, 4), 16, ETC2_RGBA8),
    block(152, "ETC2_R8G8B8A8_SRGB_BLOCK", Etc2, Srgb, model::ETC2, (4, 4), 16, ETC2_RGBA8),
    block(153, "EAC_R11_UNORM_BLOCK", Etc2, Unorm, model::ETC2, (4, 4), 8, ONE64),
    block(154, "EAC_R11_SNORM_BLOCK", Etc2, Snorm, model::ETC2, (4, 4), 8, ONE64),
    block(155, "EAC_R11G11_UNORM_BLOCK", Etc2, Unorm, model::ETC2, (4, 4), 16, TWO_CHANNEL),
    block(156, "EAC_R11G11_SNORM_BLOCK", Etc2, Snorm, model::ETC2, (4, 4), 16, TWO_CHANNEL),
    astc(157, "ASTC_4x4_UNORM_BLOCK", Unorm, 4, 4),
    astc(158, "ASTC_4x4_SRGB_BLOCK", Srgb, 4, 4),
    astc(159, "ASTC_5x4_UNORM_BLOCK", Unorm, 5, 4),
    astc(160, "ASTC_5x4_SRGB_BLOCK", Srgb, 5, 4),
    astc(161, "ASTC_5x5_UNORM_BLOCK", Unorm, 5, 5),
    astc(162, "ASTC_5x5_SRGB_BLOCK", Srgb, 5, 5),
    astc(163, "ASTC_6x5_UNORM_BLOCK", Unorm, 6, 5),
    astc(164, "ASTC_6x5_SRGB_BLOCK", Srgb, 6, 5),
    astc(165, "ASTC_6x6_UNORM_BLOCK", Unorm, 6, 6),
    astc(166, "ASTC_6x6_SRGB_BLOCK", Srgb, 6, 6),
    astc(167, "ASTC_8x5_UNORM_BLOCK", Unorm, 8, 5),
    astc(168, "ASTC_8x5_SRGB_BLOCK", Srgb, 8, 5),
    astc(169, "ASTC_8x6_UNORM_BLOCK", Unorm, 8, 6),
    astc(170, "ASTC_8x6_SRGB_BLOCK", Srgb, 8, 6),
    astc(171, "ASTC_8x8_UNORM_BLOCK", Unorm, 8, 8),
    astc(172, "ASTC_8x8_SRGB_BLOCK", Srgb, 8, 8),
    astc(173, "ASTC_10x5_UNORM_BLOCK", Unorm, 10, 5),
    astc(174, "ASTC_10x5_SRGB_BLOCK", Srgb, 10, 5),
    astc(175, "ASTC_10x6_UNORM_BLOCK", Unorm, 10, 6),
    astc(176, "ASTC_10x6_SRGB_BLOCK", Srgb, 10, 6),
    astc(177, "ASTC_10x8_UNORM_BLOCK", Unorm, 10, 8),
    astc(178, "ASTC_10x8_SRGB_BLOCK", Srgb, 10, 8),
    astc(179, "ASTC_10x10_UNORM_BLOCK", Unorm, 10, 10),
    astc(180, "ASTC_10x10_SRGB_BLOCK", Srgb, 10, 10),
    astc(181, "ASTC_12x10_UNORM_BLOCK", Unorm, 12, 10),
    astc(182, "ASTC_12x10_SRGB_BLOCK", Srgb, 12, 10),
    astc(183, "ASTC_12x12_UNORM_BLOCK", Unorm, 12, 12),
    astc(184, "ASTC_12x12_SRGB_BLOCK", Srgb, 12, 12),
];

/// `VkFormat` values of the formats the rest of the studio names directly.
pub mod vk {
    /// `VK_FORMAT_R8G8B8A8_UNORM`.
    pub const R8G8B8A8_UNORM: u32 = 37;
    /// `VK_FORMAT_R8G8B8A8_SRGB`.
    pub const R8G8B8A8_SRGB: u32 = 43;
    /// `VK_FORMAT_R16G16B16A16_SFLOAT`.
    pub const R16G16B16A16_SFLOAT: u32 = 97;
    /// `VK_FORMAT_R32G32B32A32_SFLOAT`.
    pub const R32G32B32A32_SFLOAT: u32 = 109;
    /// `VK_FORMAT_BC6H_UFLOAT_BLOCK`.
    pub const BC6H_UFLOAT: u32 = 143;
    /// `VK_FORMAT_BC6H_SFLOAT_BLOCK`.
    pub const BC6H_SFLOAT: u32 = 144;
    /// `VK_FORMAT_BC7_UNORM_BLOCK`.
    pub const BC7_UNORM: u32 = 145;
    /// `VK_FORMAT_BC7_SRGB_BLOCK`.
    pub const BC7_SRGB: u32 = 146;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_consistent() {
        for (i, f) in FORMATS.iter().enumerate() {
            // Unique values, found by lookup.
            assert_eq!(FormatInfo::from_vk(f.vk_format), Some(f), "{}", f.name);
            assert!(FORMATS[i + 1..].iter().all(|g| g.vk_format != f.vk_format));
            // Samples lie inside the block and plain formats use all of it.
            let bits = u32::from(f.block_bytes) * 8;
            for sample in f.samples {
                assert!(u32::from(sample.bit_offset) + u32::from(sample.bit_length) <= bits, "{}", f.name);
            }
            if f.is_compressed() {
                assert!(f.block_width >= 4 && f.block_height >= 4, "{}", f.name);
            } else {
                let total: u32 = f.samples.iter().map(|s| u32::from(s.bit_length)).sum();
                assert_eq!(total, bits, "{}", f.name);
                assert_eq!(u32::from(f.type_size) * 8, u32::from(f.samples[0].bit_length), "{}", f.name);
            }
            assert_eq!(f.name.contains("SRGB"), f.is_srgb(), "{}", f.name);
        }
        assert_eq!(FormatInfo::from_vk(0), None);
        assert_eq!(FormatInfo::from_vk(vk::BC7_SRGB).map(|f| f.name), Some("BC7_SRGB_BLOCK"));
    }

    #[test]
    fn block_counts_round_up() {
        let bc7 = FormatInfo::from_vk(vk::BC7_UNORM).unwrap();
        assert_eq!(bc7.image_size(1, 1, 1), Some(16));
        assert_eq!(bc7.image_size(5, 4, 1), Some(32));
        assert_eq!(bc7.image_size(8, 9, 1), Some(2 * 3 * 16));
        let astc = FormatInfo::from_vk(183).unwrap();
        assert_eq!(astc.image_size(13, 12, 1), Some(2 * 16));
        let rgba = FormatInfo::from_vk(vk::R8G8B8A8_UNORM).unwrap();
        assert_eq!(rgba.image_size(3, 2, 2), Some(48));
        assert_eq!(rgba.image_size(u32::MAX, u32::MAX, u32::MAX), None);
    }

    #[test]
    fn channel_types_and_bounds() {
        let srgb = FormatInfo::from_vk(vk::R8G8B8A8_SRGB).unwrap();
        assert_eq!(srgb.channel_type(srgb.samples[0]), 0);
        assert_eq!(srgb.channel_type(srgb.samples[3]), ALPHA | LINEAR);
        assert_eq!(srgb.sample_bounds(srgb.samples[0]), (0, 255));
        let half = FormatInfo::from_vk(vk::R16G16B16A16_SFLOAT).unwrap();
        assert_eq!(half.channel_type(half.samples[1]), 1 | FLOAT | SIGNED);
        assert_eq!(half.sample_bounds(half.samples[1]), (0xBF80_0000, 0x3F80_0000));
        let bc4s = FormatInfo::from_vk(140).unwrap();
        assert_eq!(bc4s.sample_bounds(bc4s.samples[0]), (0x8000_0000, 0x7FFF_FFFF));
        let r16 = FormatInfo::from_vk(70).unwrap();
        assert_eq!(r16.sample_bounds(r16.samples[0]), (0, 65535));
    }
}
