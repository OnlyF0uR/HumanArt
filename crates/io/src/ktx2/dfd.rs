//! The Khronos Data Format Descriptor of a KTX2 file: the basic descriptor block, read, checked
//! against the file's `vkFormat`, and written `[IO-04]`.

use super::format::{FLOAT, FormatInfo, SIGNED};
use crate::bytes::{Reader, Writer};
use crate::error::{Error, Format, Result};

/// `KHR_DF_PRIMARIES_BT709`: Rec.709 / sRGB primaries.
pub const PRIMARIES_BT709: u8 = 1;
/// `KHR_DF_FLAG_ALPHA_PREMULTIPLIED`.
pub const FLAG_PREMULTIPLIED: u8 = 1;
/// `KHR_DF_VERSIONNUMBER_1_3`, the version of the basic block this crate reads and writes.
pub const VERSION: u16 = 2;
/// The largest sample count accepted in a basic block (a texel block of the formats here has at
/// most four; 16 leaves room for multi-plane descriptors without admitting absurd sizes).
pub const MAX_SAMPLES: usize = 16;
/// The size of the basic block without its samples.
const BASIC_HEADER: usize = 24;
/// The size of one sample.
const SAMPLE_LEN: usize = 16;

/// One sample of a basic descriptor block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sample {
    /// The first bit of the sample in the texel block.
    pub bit_offset: u16,
    /// The number of bits (stored minus one in the file).
    pub bit_length: u16,
    /// The channel id (low four bits) and qualifier bits (linear, exponent, signed, float).
    pub channel_type: u8,
    /// The sample position within the texel block.
    pub position: [u8; 4],
    /// `sampleLower`.
    pub lower: u32,
    /// `sampleUpper`.
    pub upper: u32,
}

/// The basic descriptor block of a DFD.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Dfd {
    /// `colorModel` (see [`super::format::model`]).
    pub color_model: u8,
    /// `colorPrimaries` (1 = BT.709).
    pub color_primaries: u8,
    /// `transferFunction` (see [`super::format::transfer`]).
    pub transfer: u8,
    /// `flags` (bit 0: alpha premultiplied).
    pub flags: u8,
    /// `texelBlockDimension0..3`: block size minus one per axis.
    pub texel_block: [u8; 4],
    /// `bytesPlane0..7`.
    pub bytes_plane: [u8; 8],
    /// The samples.
    pub samples: Vec<Sample>,
}

impl Dfd {
    /// The descriptor of `format` as this crate writes it. `supercompressed` sets `bytesPlane0` to 0
    /// (the data is unsized), as the KTX2 specification asks for supercompressed files.
    #[must_use]
    pub fn for_format(format: &FormatInfo, primaries: u8, premultiplied: bool, supercompressed: bool) -> Self {
        let samples = format
            .samples
            .iter()
            .map(|&layout| {
                let (lower, upper) = format.sample_bounds(layout);
                Sample {
                    bit_offset: layout.bit_offset,
                    bit_length: u16::from(layout.bit_length),
                    channel_type: format.channel_type(layout),
                    position: [0; 4],
                    lower,
                    upper,
                }
            })
            .collect();
        let mut bytes_plane = [0_u8; 8];
        if !supercompressed {
            bytes_plane[0] = format.block_bytes;
        }
        Self {
            color_model: format.color_model,
            color_primaries: primaries,
            transfer: format.transfer(),
            flags: if premultiplied { FLAG_PREMULTIPLIED } else { 0 },
            texel_block: [format.block_width - 1, format.block_height - 1, 0, 0],
            bytes_plane,
            samples,
        }
    }

    /// The byte length of the whole DFD as written: the total-size word and one basic block.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        4 + BASIC_HEADER + SAMPLE_LEN * self.samples.len()
    }

    /// Appends the DFD (total size, then the basic block).
    pub fn write(&self, w: &mut Writer) {
        let block_size = BASIC_HEADER + SAMPLE_LEN * self.samples.len();
        w.u32_le(self.byte_len() as u32);
        // vendorId 0 (Khronos), descriptorType 0 (basic).
        w.u32_le(0);
        w.u32_le(u32::from(VERSION) | ((block_size as u32) << 16));
        w.bytes(&[self.color_model, self.color_primaries, self.transfer, self.flags]);
        w.bytes(&self.texel_block);
        w.bytes(&self.bytes_plane);
        for sample in &self.samples {
            w.u16_le(sample.bit_offset);
            w.u8(sample.bit_length.saturating_sub(1) as u8);
            w.u8(sample.channel_type);
            w.bytes(&sample.position);
            w.u32_le(sample.lower);
            w.u32_le(sample.upper);
        }
    }

    /// Reads the DFD in `data` (exactly `dfdByteLength` bytes; `base` is its offset in the file, for
    /// error messages). Descriptor blocks other than the Khronos basic block are skipped.
    pub fn parse(data: &[u8], base: usize) -> Result<Self> {
        let invalid = |at: usize, what| Error::Invalid { format: Format::Ktx2, offset: base + at, what };
        let mut r = Reader::new(data, Format::Ktx2);
        let total = r.len_u32_le()?;
        if total != data.len() {
            return Err(invalid(0, "DFD total size does not match dfdByteLength"));
        }
        let mut basic = None;
        while !r.is_empty() {
            let start = r.pos();
            let kind = r.u32_le().map_err(|_| invalid(start, "truncated DFD block header"))?;
            let version_size = r.u32_le().map_err(|_| invalid(start, "truncated DFD block header"))?;
            let size = (version_size >> 16) as usize;
            // Every block is at least its 8-byte header, so the loop always advances.
            if size < 8 || size > data.len() - start {
                return Err(invalid(start, "DFD block size is out of range"));
            }
            let body = &data[start + 8..start + size];
            r.seek(start + size)?;
            if kind == 0 && basic.is_none() {
                if version_size & 0xFFFF != u32::from(VERSION) {
                    return Err(Error::Unsupported { format: Format::Ktx2, what: "DFD basic block version" });
                }
                basic = Some(Self::parse_basic(body, base + start + 8)?);
            }
        }
        basic.ok_or_else(|| invalid(0, "DFD has no basic descriptor block"))
    }

    fn parse_basic(body: &[u8], base: usize) -> Result<Self> {
        let invalid = |what| Error::Invalid { format: Format::Ktx2, offset: base, what };
        if body.len() < BASIC_HEADER - 8 || !(body.len() + 8 - BASIC_HEADER).is_multiple_of(SAMPLE_LEN) {
            return Err(invalid("DFD basic block size is not 24 + 16 × samples"));
        }
        let count = (body.len() + 8 - BASIC_HEADER) / SAMPLE_LEN;
        if count == 0 || count > MAX_SAMPLES {
            return Err(invalid("DFD sample count is out of range"));
        }
        let mut r = Reader::new(body, Format::Ktx2);
        let [color_model, color_primaries, transfer, flags] = r.array::<4>()?;
        let texel_block = r.array::<4>()?;
        let bytes_plane = r.array::<8>()?;
        let mut samples = Vec::with_capacity(count);
        for _ in 0..count {
            let bit_offset = r.u16_le()?;
            let bit_length = u16::from(r.u8()?) + 1;
            let channel_type = r.u8()?;
            let position = r.array::<4>()?;
            let lower = r.u32_le()?;
            let upper = r.u32_le()?;
            samples.push(Sample { bit_offset, bit_length, channel_type, position, lower, upper });
        }
        Ok(Self { color_model, color_primaries, transfer, flags, texel_block, bytes_plane, samples })
    }

    /// Checks that this descriptor describes `format`: colour model, texel block size, bytes per
    /// block (0 is also accepted for supercompressed data), transfer function, and every sample's
    /// channel, bits and float/signed qualifiers. `offset` locates errors.
    pub fn check(&self, format: &FormatInfo, supercompressed: bool, offset: usize) -> Result<()> {
        let invalid = |what| Err(Error::Invalid { format: Format::Ktx2, offset, what });
        if self.color_model != format.color_model {
            return invalid("DFD colour model does not match vkFormat");
        }
        if self.texel_block != [format.block_width - 1, format.block_height - 1, 0, 0] {
            return invalid("DFD texel block size does not match vkFormat");
        }
        let plane_ok = self.bytes_plane[0] == format.block_bytes || (supercompressed && self.bytes_plane[0] == 0);
        if !plane_ok || self.bytes_plane[1..].iter().any(|&b| b != 0) {
            return invalid("DFD bytes per plane do not match vkFormat");
        }
        if self.transfer != format.transfer() {
            return invalid("DFD transfer function does not match vkFormat");
        }
        if self.samples.len() != format.samples.len() {
            return invalid("DFD sample count does not match vkFormat");
        }
        for (sample, layout) in self.samples.iter().zip(format.samples) {
            let expected = format.channel_type(*layout);
            let qualifiers = FLOAT | SIGNED;
            if sample.channel_type & 0x0F != layout.channel
                || sample.channel_type & qualifiers != expected & qualifiers
                || sample.bit_offset != layout.bit_offset
                || sample.bit_length != u16::from(layout.bit_length)
            {
                return invalid("DFD sample does not match vkFormat");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::format::{FORMATS, vk};
    use super::*;

    #[test]
    fn every_format_round_trips_and_checks() {
        for format in FORMATS {
            for supercompressed in [false, true] {
                let dfd = Dfd::for_format(format, PRIMARIES_BT709, false, supercompressed);
                let mut w = Writer::new();
                dfd.write(&mut w);
                assert_eq!(w.len(), dfd.byte_len());
                let back = Dfd::parse(w.as_slice(), 0).unwrap();
                assert_eq!(back, dfd, "{}", format.name);
                back.check(format, supercompressed, 0).unwrap();
            }
        }
    }

    #[test]
    fn mismatches_are_refused() {
        let bc7 = FormatInfo::from_vk(vk::BC7_SRGB).unwrap();
        let rgba = FormatInfo::from_vk(vk::R8G8B8A8_SRGB).unwrap();
        let dfd = Dfd::for_format(bc7, PRIMARIES_BT709, false, false);
        assert!(dfd.check(rgba, false, 0).is_err());
        let linear = FormatInfo::from_vk(vk::BC7_UNORM).unwrap();
        assert!(dfd.check(linear, false, 0).is_err(), "transfer function differs");
        let mut wrong_bits = Dfd::for_format(rgba, PRIMARIES_BT709, false, false);
        wrong_bits.samples[2].bit_offset = 8;
        assert!(wrong_bits.check(rgba, false, 0).is_err());
        let mut zero_plane = Dfd::for_format(rgba, PRIMARIES_BT709, false, false);
        zero_plane.bytes_plane[0] = 0;
        assert!(zero_plane.check(rgba, false, 0).is_err());
        assert!(zero_plane.check(rgba, true, 0).is_ok());
    }

    #[test]
    fn hostile_layouts_are_refused() {
        let dfd = Dfd::for_format(FormatInfo::from_vk(vk::R8G8B8A8_UNORM).unwrap(), 1, false, false);
        let mut w = Writer::new();
        dfd.write(&mut w);
        let good = w.into_vec();
        // Total size disagreeing with the slice.
        assert!(Dfd::parse(&good[..good.len() - 4], 0).is_err());
        // Block size smaller than its header, and larger than the data.
        for size in [0_u16, 4, 0xFFFF] {
            let mut bad = good.clone();
            bad[10..12].copy_from_slice(&size.to_le_bytes());
            assert!(Dfd::parse(&bad, 0).is_err(), "{size}");
        }
        // A vendor block alone is not enough.
        let mut vendor = good.clone();
        vendor[4] = 1;
        assert!(Dfd::parse(&vendor, 0).is_err());
        // Every truncation fails cleanly.
        for len in 0..good.len() {
            let mut cut = good[..len].to_vec();
            if cut.len() >= 4 {
                cut[..4].copy_from_slice(&(len as u32).to_le_bytes());
            }
            assert!(Dfd::parse(&cut, 0).is_err(), "{len}");
        }
    }
}
