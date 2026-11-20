//! KTX2 texture containers: header, level index, data format descriptor, key/value data and mip
//! levels, read and written `[IO-04]`.
//!
//! Level payloads are kept as bytes. Uncompressed formats convert to and from [`Image`]; block
//! formats (BC, ETC2/EAC, ASTC) are opaque blocks whose count and size are checked against the
//! format's block dimensions at every level, never decoded here (the BC7/BC6H compressors are WP-28's).
//!
//! **Supercompression:** none; the in-house LZ codec ([`crate::lz`], one LZ frame per level,
//! scheme id [`SCHEME_LZ`], a private value outside the registered range); and, when the `deflate`
//! feature is enabled, the registered ZLIB scheme (3). BasisLZ and Zstandard are refused as
//! unsupported, as is `VK_FORMAT_UNDEFINED` (Basis Universal payloads).
//!
//! **Limits `[ROB-01]`:** the header, level index, DFD and key/value data are checked against the
//! input length before they are read; the total size of all levels is checked against
//! [`Limits::max_bytes`], the dimensions against [`Limits::max_dimension`] and the layer × face count
//! against [`Limits::max_elements`] before anything is allocated. At most [`MAX_LEVELS`] levels and
//! [`MAX_KEY_VALUES`] key/value entries.

pub mod dfd;
pub mod format;

pub use dfd::{Dfd, Sample};
pub use format::{FORMATS, Family, FormatInfo, Numeric, vk};

use crate::bytes::{Reader, Writer};
use crate::error::{Error, Format, Result};
use crate::image::{ColorSpace, Image, Limits, Samples};

/// The 12-byte file identifier `«KTX 20»\r\n\x1A\n`.
pub const IDENTIFIER: [u8; 12] = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];
/// The length of the fixed header (identifier, nine fields and the index).
pub const HEADER_LEN: usize = 80;
/// The length of one level index entry.
pub const LEVEL_ENTRY_LEN: usize = 24;
/// The most mip levels a file may have (a 2^31-texel side has 32).
pub const MAX_LEVELS: u32 = 32;
/// The most key/value entries a file may have.
pub const MAX_KEY_VALUES: usize = 1024;
/// The largest key/value data a writer produces (offsets in the index are 32-bit).
pub const MAX_KVD_BYTES: usize = 1 << 30;
/// Supercompression scheme id: none.
pub const SCHEME_NONE: u32 = 0;
/// Supercompression scheme id: BasisLZ (refused).
pub const SCHEME_BASIS_LZ: u32 = 1;
/// Supercompression scheme id: Zstandard (refused).
pub const SCHEME_ZSTD: u32 = 2;
/// Supercompression scheme id: ZLIB (RFC 1950), one stream per level.
pub const SCHEME_ZLIB: u32 = 3;
/// Supercompression scheme id of the in-house LZ codec: the bytes `HALZ` read as a little-endian
/// `u32`, far outside the registered range, so other tools refuse such a file instead of misreading
/// it. Each level is one complete LZ frame (`docs/format-lz.md`).
pub const SCHEME_LZ: u32 = u32::from_le_bytes(*b"HALZ");
/// The `KTXwriter` value written when the caller supplies none.
pub const WRITER_NAME: &str = concat!("HumanArt humanart-io ", env!("CARGO_PKG_VERSION"));

/// How level data is supercompressed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Supercompression {
    /// Stored as it is.
    #[default]
    None,
    /// The in-house LZ codec ([`SCHEME_LZ`]).
    Lz,
    /// ZLIB ([`SCHEME_ZLIB`]); needs the `deflate` feature.
    Zlib,
}

impl Supercompression {
    /// The `supercompressionScheme` value.
    #[must_use]
    pub const fn id(self) -> u32 {
        match self {
            Self::None => SCHEME_NONE,
            Self::Lz => SCHEME_LZ,
            Self::Zlib => SCHEME_ZLIB,
        }
    }

    /// The scheme of a `supercompressionScheme` value, or the error a reader reports for it.
    pub const fn from_id(id: u32) -> Result<Self> {
        match id {
            SCHEME_NONE => Ok(Self::None),
            SCHEME_LZ => Ok(Self::Lz),
            SCHEME_ZLIB => Ok(Self::Zlib),
            SCHEME_BASIS_LZ => Err(Error::Unsupported { format: Format::Ktx2, what: "BasisLZ supercompression" }),
            SCHEME_ZSTD => Err(Error::Unsupported { format: Format::Ktx2, what: "Zstandard supercompression" }),
            _ => Err(Error::Unsupported { format: Format::Ktx2, what: "unknown supercompression scheme" }),
        }
    }
}

/// One entry of the level index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LevelEntry {
    /// Offset of the level's data in the file.
    pub offset: u64,
    /// Length of the level's data in the file.
    pub length: u64,
    /// Length of the level's data once decompressed.
    pub uncompressed_length: u64,
}

/// The fixed header, index and level index of a KTX2 file, as stored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    /// `vkFormat`.
    pub vk_format: u32,
    /// `typeSize`.
    pub type_size: u32,
    /// `pixelWidth`.
    pub width: u32,
    /// `pixelHeight` (0 for a 1D texture).
    pub height: u32,
    /// `pixelDepth` (0 unless 3D).
    pub depth: u32,
    /// `layerCount` (0 unless an array).
    pub layers: u32,
    /// `faceCount` (1, or 6 for a cube map).
    pub faces: u32,
    /// `levelCount` (0 asks the loader to generate mipmaps from the one stored level).
    pub level_count: u32,
    /// `supercompressionScheme`.
    pub scheme: u32,
    /// `dfdByteOffset`, `dfdByteLength`.
    pub dfd: (u32, u32),
    /// `kvdByteOffset`, `kvdByteLength`.
    pub kvd: (u32, u32),
    /// `sgdByteOffset`, `sgdByteLength`.
    pub sgd: (u64, u64),
    /// The level index, level 0 (largest) first; one entry when `level_count` is 0.
    pub levels: Vec<LevelEntry>,
}

/// A KTX2 texture: its shape, its format and every mip level's data, decompressed.
///
/// Each level holds, in this order, every layer, every face of the layer, every depth slice of the
/// face and every row of blocks of the slice, as the KTX2 specification lays them out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Texture {
    /// The texel format.
    pub format: &'static FormatInfo,
    /// Width in texels (at least 1).
    pub width: u32,
    /// Height in texels; 0 for a 1D texture.
    pub height: u32,
    /// Depth in texels; 0 unless the texture is 3D.
    pub depth: u32,
    /// Array layer count; 0 unless the texture is an array.
    pub layers: u32,
    /// Face count: 1, or 6 for a cube map.
    pub faces: u32,
    /// The levels, level 0 (largest) first.
    pub levels: Vec<Vec<u8>>,
    /// Whether the file asks the loader to generate mipmaps (`levelCount` 0, one level stored).
    pub generate_mipmaps: bool,
    /// DFD `colorPrimaries` (1 = BT.709).
    pub color_primaries: u8,
    /// Whether colour is premultiplied by alpha.
    pub premultiplied_alpha: bool,
    /// Key/value entries, sorted by key when read. Text values carry their terminating NUL, as
    /// the KTX2 specification asks for the `KTX`-prefixed keys.
    pub key_values: Vec<(String, Vec<u8>)>,
    /// How [`write`] compresses the levels; what the file used when read.
    pub supercompression: Supercompression,
}

const fn invalid(offset: usize, what: &'static str) -> Error {
    Error::Invalid { format: Format::Ktx2, offset, what }
}

const fn encode(what: &'static str) -> Error {
    Error::Encode { format: Format::Ktx2, what }
}

impl Texture {
    /// A 2D texture of `format` with the given levels (level 0 first), checked by [`validate`](Self::validate).
    pub fn new_2d(format: &'static FormatInfo, width: u32, height: u32, levels: Vec<Vec<u8>>) -> Result<Self> {
        let texture = Self {
            format,
            width,
            height,
            depth: 0,
            layers: 0,
            faces: 1,
            levels,
            generate_mipmaps: false,
            color_primaries: dfd::PRIMARIES_BT709,
            premultiplied_alpha: false,
            key_values: Vec::new(),
            supercompression: Supercompression::None,
        };
        texture.validate()?;
        Ok(texture)
    }

    /// The size in texels of `level` (each axis halved per level, at least 1).
    #[must_use]
    pub fn level_extent(&self, level: u32) -> (u32, u32, u32) {
        let shrink = |v: u32| v.checked_shr(level).unwrap_or(0).max(1);
        (shrink(self.width), shrink(self.height), shrink(self.depth))
    }

    /// The most levels a texture of this size can have: `floor(log2(largest side)) + 1`.
    #[must_use]
    pub const fn max_levels(&self) -> u32 {
        let side = max3(self.width, self.height, self.depth);
        if side == 0 { 1 } else { 32 - side.leading_zeros() }
    }

    /// The byte size of one image (one layer, one face) of `level`, or `None` on overflow.
    #[must_use]
    pub fn image_size(&self, level: u32) -> Option<u64> {
        let (w, h, d) = self.level_extent(level);
        self.format.image_size(w, h, d)
    }

    /// The byte size of all of `level` (every layer and face), or `None` on overflow.
    #[must_use]
    pub fn level_size(&self, level: u32) -> Option<u64> {
        self.image_size(level)?.checked_mul(u64::from(self.layers.max(1)))?.checked_mul(u64::from(self.faces))
    }

    /// The data of one image (one layer, one face, all depth slices) of `level`.
    #[must_use]
    pub fn image(&self, level: u32, layer: u32, face: u32) -> Option<&[u8]> {
        if layer >= self.layers.max(1) || face >= self.faces {
            return None;
        }
        let size = usize::try_from(self.image_size(level)?).ok()?;
        let index = (layer as usize).checked_mul(self.faces as usize)?.checked_add(face as usize)?;
        let start = index.checked_mul(size)?;
        self.levels.get(level as usize)?.get(start..start.checked_add(size)?)
    }

    /// Checks the shape, the level count and every level's size against the format's block
    /// dimensions, and the key/value entries; [`write`] runs this first.
    pub fn validate(&self) -> Result<()> {
        self.check_shape().map_err(encode)?;
        if self.levels.is_empty() || self.levels.len() > self.max_levels() as usize {
            return Err(encode("level count is outside 1 ..= log2(largest side) + 1"));
        }
        if self.generate_mipmaps && self.levels.len() != 1 {
            return Err(encode("a texture asking for generated mipmaps stores exactly one level"));
        }
        for (i, level) in self.levels.iter().enumerate() {
            if self.level_size(i as u32) != Some(level.len() as u64) {
                return Err(encode("a level's size does not match the format's blocks"));
            }
        }
        for (i, (key, _)) in self.key_values.iter().enumerate() {
            if key.is_empty() || key.contains('\0') {
                return Err(encode("a key is empty or contains NUL"));
            }
            if self.key_values[..i].iter().any(|(k, _)| k == key) {
                return Err(encode("a key appears twice"));
            }
        }
        let added = usize::from(!self.key_values.iter().any(|(k, _)| k == "KTXwriter"));
        if self.key_values.len() + added > MAX_KEY_VALUES {
            return Err(encode("too many key/value entries"));
        }
        let kvd_bytes: usize = self.key_values.iter().map(|(k, v)| k.len() + v.len() + 8).sum();
        if kvd_bytes > MAX_KVD_BYTES {
            return Err(encode("key/value data is too large"));
        }
        Ok(())
    }

    /// The shape rules of the KTX2 specification, as a reason when broken.
    fn check_shape(&self) -> std::result::Result<(), &'static str> {
        if self.width == 0 {
            return Err("width is 0");
        }
        if self.depth > 0 && self.height == 0 {
            return Err("a 3D texture has height 0");
        }
        if self.format.is_compressed() && self.height == 0 {
            return Err("a block-compressed texture cannot be 1D");
        }
        match self.faces {
            1 => Ok(()),
            6 if self.width == self.height && self.depth == 0 => Ok(()),
            6 => Err("a cube map must be square and 2D"),
            _ => Err("face count is not 1 or 6"),
        }
    }

    /// A single-level 2D texture holding `image` in an uncompressed format: 8-bit images become
    /// `R8`/`R8G8`/`R8G8B8A8` (sRGB variants for [`ColorSpace::Srgb`]), 16-bit unsigned images
    /// `R16`/`R16G16`/`R16G16B16A16_UNORM`, half and float images the matching `SFLOAT` formats.
    /// Three-channel images gain an opaque alpha channel (there is no widely supported RGB format).
    pub fn from_image(image: &Image) -> Result<Self> {
        let (format, data) = image_to_level(image)?;
        Self::new_2d(format, image.width, image.height, vec![data])
    }

    /// A 2D texture holding a mip chain, largest first; every image must have the format, colour
    /// space and channel count of the first and half its predecessor's size (rounded down, at
    /// least 1).
    pub fn from_mip_chain(images: &[Image]) -> Result<Self> {
        let first = images.first().ok_or(encode("the mip chain is empty"))?;
        let (format, data) = image_to_level(first)?;
        let mut levels = vec![data];
        for (level, image) in images.iter().enumerate().skip(1) {
            let expect = |side: u32| (side >> level.min(31)).max(1);
            if image.width != expect(first.width)
                || image.height != expect(first.height)
                || image.channels != first.channels
                || image.color_space != first.color_space
            {
                return Err(encode("a mip level does not halve the previous one or changes format"));
            }
            let (f, data) = image_to_level(image)?;
            if f != format {
                return Err(encode("a mip level changes the sample type"));
            }
            levels.push(data);
        }
        Self::new_2d(format, first.width, first.height, levels)
    }

    /// One image of an uncompressed texture as an [`Image`] (RGBA for four channels; BGRA formats are
    /// swizzled). The colour space is [`ColorSpace::Srgb`] for sRGB formats and
    /// [`ColorSpace::Linear`] otherwise; set it to `Data` or `Normal` where that is what the texture
    /// holds. 3D textures and block formats are refused as unsupported.
    pub fn to_image(&self, level: u32, layer: u32, face: u32) -> Result<Image> {
        if self.format.is_compressed() {
            return Err(Error::Unsupported { format: Format::Ktx2, what: "decoding block-compressed texels" });
        }
        let (w, h, d) = self.level_extent(level);
        if d > 1 {
            return Err(Error::Unsupported { format: Format::Ktx2, what: "3D textures as images" });
        }
        let data = self.image(level, layer, face).ok_or(encode("level, layer or face is out of range"))?;
        let channels = self.format.samples.len() as u8;
        let color_space = if self.format.is_srgb() { ColorSpace::Srgb } else { ColorSpace::Linear };
        let samples = match (self.format.type_size, self.format.numeric) {
            (1, _) => {
                let mut v = data.to_vec();
                if self.format.samples[0].channel == 2 {
                    for px in v.as_chunks_mut::<4>().0 {
                        px.swap(0, 2);
                    }
                }
                Samples::U8(v)
            }
            (2, Numeric::Unorm) => {
                Samples::U16(data.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect())
            }
            (2, _) => Samples::F16(data.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect()),
            _ => Samples::F32(data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()),
        };
        Image::new(Format::Ktx2, w, h, channels, color_space, samples)
    }
}

const fn max3(a: u32, b: u32, c: u32) -> u32 {
    let ab = if a > b { a } else { b };
    if ab > c { ab } else { c }
}

/// The uncompressed format and little-endian level bytes for `image`.
fn image_to_level(image: &Image) -> Result<(&'static FormatInfo, Vec<u8>)> {
    let channels = image.channels as usize;
    let out_channels = if channels == 3 { 4 } else { channels };
    let srgb = image.color_space == ColorSpace::Srgb;
    let vk_format = match (&image.samples, out_channels, srgb) {
        (Samples::U8(_), 1, false) => 9,
        (Samples::U8(_), 1, true) => 15,
        (Samples::U8(_), 2, false) => 16,
        (Samples::U8(_), 2, true) => 22,
        (Samples::U8(_), _, false) => vk::R8G8B8A8_UNORM,
        (Samples::U8(_), _, true) => vk::R8G8B8A8_SRGB,
        (_, _, true) => return Err(encode("only 8-bit images can be stored with the sRGB transfer function")),
        (Samples::U16(_), 1, _) => 70,
        (Samples::U16(_), 2, _) => 77,
        (Samples::U16(_), _, _) => 91,
        (Samples::F16(_), 1, _) => 76,
        (Samples::F16(_), 2, _) => 83,
        (Samples::F16(_), _, _) => vk::R16G16B16A16_SFLOAT,
        (Samples::F32(_), 1, _) => 100,
        (Samples::F32(_), 2, _) => 103,
        (Samples::F32(_), _, _) => vk::R32G32B32A32_SFLOAT,
    };
    let format = FormatInfo::from_vk(vk_format).ok_or(encode("no format for this image"))?;
    let mut w = Writer::with_capacity(image.pixel_count() * out_channels * image.samples.sample_size());
    let pixels = image.pixel_count();
    for px in 0..pixels {
        for c in 0..out_channels {
            let i = px * channels + c;
            let alpha = c >= channels;
            match &image.samples {
                Samples::U8(s) => w.u8(if alpha { u8::MAX } else { s.get(i).copied().unwrap_or(0) }),
                Samples::U16(s) => w.u16_le(if alpha { u16::MAX } else { s.get(i).copied().unwrap_or(0) }),
                // 0x3C00 is 1.0 as a half float.
                Samples::F16(s) => w.u16_le(if alpha { 0x3C00 } else { s.get(i).copied().unwrap_or(0) }),
                Samples::F32(s) => w.f32_le(if alpha { 1.0 } else { s.get(i).copied().unwrap_or(0.0) }),
            }
        }
    }
    Ok((format, w.into_vec()))
}

/// Reads the fixed header and the level index, checking only that they are present.
pub fn read_header(data: &[u8]) -> Result<Header> {
    let mut r = Reader::new(data, Format::Ktx2);
    if r.array::<12>()? != IDENTIFIER {
        return Err(invalid(0, "not a KTX2 file"));
    }
    let mut field = [0_u32; 9];
    for f in &mut field {
        *f = r.u32_le()?;
    }
    let [vk_format, type_size, width, height, depth, layers, faces, level_count, scheme] = field;
    let dfd = (r.u32_le()?, r.u32_le()?);
    let kvd = (r.u32_le()?, r.u32_le()?);
    let sgd = (r.u64_le()?, r.u64_le()?);
    if level_count > MAX_LEVELS {
        return Err(Error::Limit {
            format: Format::Ktx2,
            what: "level count",
            limit: u64::from(MAX_LEVELS),
            got: u64::from(level_count),
        });
    }
    let n = level_count.max(1) as usize;
    r.check_count(n, LEVEL_ENTRY_LEN)?;
    let mut levels = Vec::with_capacity(n);
    for _ in 0..n {
        levels.push(LevelEntry { offset: r.u64_le()?, length: r.u64_le()?, uncompressed_length: r.u64_le()? });
    }
    Ok(Header { vk_format, type_size, width, height, depth, layers, faces, level_count, scheme, dfd, kvd, sgd, levels })
}

/// The byte range `offset .. offset + length` of the file, if it lies inside `len`.
fn range(offset: u64, length: u64, len: usize) -> Option<std::ops::Range<usize>> {
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(usize::try_from(length).ok()?)?;
    (end <= len).then_some(start..end)
}

/// Reads a KTX2 file: every structure is checked, every level decompressed and its size checked.
pub fn read(data: &[u8], limits: &Limits) -> Result<Texture> {
    let header = read_header(data)?;
    if header.vk_format == 0 {
        return Err(Error::Unsupported { format: Format::Ktx2, what: "VK_FORMAT_UNDEFINED (Basis Universal)" });
    }
    let format = FormatInfo::from_vk(header.vk_format)
        .ok_or(Error::Unsupported { format: Format::Ktx2, what: "this vkFormat" })?;
    if header.type_size != u32::from(format.type_size) {
        return Err(invalid(16, "typeSize does not match vkFormat"));
    }
    let supercompression = Supercompression::from_id(header.scheme)?;
    let mut texture = Texture {
        format,
        width: header.width,
        height: header.height,
        depth: header.depth,
        layers: header.layers,
        faces: header.faces,
        levels: Vec::new(),
        generate_mipmaps: header.level_count == 0,
        color_primaries: dfd::PRIMARIES_BT709,
        premultiplied_alpha: false,
        key_values: Vec::new(),
        supercompression,
    };
    texture.check_shape().map_err(|what| invalid(20, what))?;
    limits.check_image(Format::Ktx2, u64::from(header.width), u64::from(header.height.max(1)))?;
    if header.depth > limits.max_dimension {
        return Err(Error::Limit {
            format: Format::Ktx2,
            what: "texture depth",
            limit: u64::from(limits.max_dimension),
            got: u64::from(header.depth),
        });
    }
    limits.check_elements(Format::Ktx2, "layers × faces", u64::from(header.layers.max(1)) * u64::from(header.faces))?;
    let level_count = header.levels.len() as u32;
    if level_count > texture.max_levels() {
        return Err(invalid(40, "more levels than log2(largest side) + 1"));
    }

    // The DFD follows the level index directly; the key/value data follows the DFD.
    let index_end = HEADER_LEN + LEVEL_ENTRY_LEN * header.levels.len();
    let (dfd_offset, dfd_len) = (header.dfd.0 as usize, header.dfd.1 as usize);
    if dfd_offset != index_end || dfd_len == 0 {
        return Err(invalid(48, "the DFD does not follow the level index"));
    }
    let dfd_range = range(dfd_offset as u64, dfd_len as u64, data.len())
        .ok_or(Error::Truncated { format: Format::Ktx2, offset: dfd_offset })?;
    let descriptor = Dfd::parse(&data[dfd_range.clone()], dfd_offset)?;
    descriptor.check(format, supercompression != Supercompression::None, dfd_offset)?;
    texture.color_primaries = descriptor.color_primaries;
    texture.premultiplied_alpha = descriptor.flags & dfd::FLAG_PREMULTIPLIED != 0;
    let mut meta_end = dfd_range.end;
    if header.kvd.1 > 0 {
        let kvd_offset = header.kvd.0 as usize;
        if kvd_offset != dfd_range.end {
            return Err(invalid(56, "the key/value data does not follow the DFD"));
        }
        let kvd_range = range(kvd_offset as u64, u64::from(header.kvd.1), data.len())
            .ok_or(Error::Truncated { format: Format::Ktx2, offset: kvd_offset })?;
        texture.key_values = parse_kvd(&data[kvd_range.clone()], kvd_offset)?;
        meta_end = kvd_range.end;
    } else if header.kvd.0 != 0 {
        return Err(invalid(56, "empty key/value data has a non-zero offset"));
    }
    if header.sgd != (0, 0) {
        return Err(invalid(64, "supercompression global data is present for a scheme without it"));
    }

    // Level sizes, ranges, order and alignment, then the total against the limit.
    let align = if supercompression == Supercompression::None { lcm4(format.block_bytes) } else { 1 };
    let mut total = 0_u64;
    let mut ranges = Vec::with_capacity(header.levels.len());
    for (i, entry) in header.levels.iter().enumerate() {
        let at = HEADER_LEN + LEVEL_ENTRY_LEN * i;
        let expected = texture.level_size(i as u32).ok_or(invalid(at, "level size overflows"))?;
        if entry.uncompressed_length != expected {
            return Err(invalid(at, "uncompressedByteLength does not match the format's blocks"));
        }
        if supercompression == Supercompression::None && entry.length != expected {
            return Err(invalid(at, "byteLength of an uncompressed level does not match its size"));
        }
        if entry.length == 0 {
            return Err(invalid(at, "a level has no data"));
        }
        let r = range(entry.offset, entry.length, data.len())
            .ok_or(Error::Truncated { format: Format::Ktx2, offset: at })?;
        if r.start < meta_end || r.start % align != 0 {
            return Err(invalid(at, "level data overlaps the metadata or is misaligned"));
        }
        if let Some(larger) = ranges.last().map(|r: &std::ops::Range<usize>| r.start)
            && r.end > larger
        {
            return Err(invalid(at, "levels are not stored smallest first without overlap"));
        }
        total = total.saturating_add(expected);
        ranges.push(r);
    }
    limits.check_bytes(Format::Ktx2, "texture data", total)?;

    for (i, r) in ranges.into_iter().enumerate() {
        let expected =
            usize::try_from(header.levels[i].uncompressed_length).map_err(|_| invalid(0, "level too large"))?;
        let payload = &data[r.clone()];
        let level = match supercompression {
            Supercompression::None => payload.to_vec(),
            Supercompression::Lz => {
                let level_limits = Limits { max_bytes: expected as u64, ..*limits };
                crate::lz::decompress(payload, &level_limits)?
            }
            Supercompression::Zlib => inflate_level(payload, expected)?,
        };
        if level.len() != expected {
            return Err(invalid(r.start, "a supercompressed level decompresses to the wrong size"));
        }
        texture.levels.push(level);
    }
    Ok(texture)
}

#[cfg(feature = "deflate")]
fn inflate_level(payload: &[u8], expected: usize) -> Result<Vec<u8>> {
    crate::deflate::zlib_decompress(payload, expected)
}

#[cfg(not(feature = "deflate"))]
fn inflate_level(_payload: &[u8], _expected: usize) -> Result<Vec<u8>> {
    Err(Error::Unsupported { format: Format::Ktx2, what: "ZLIB supercompression without the deflate feature" })
}

// Same signature as the variant without the feature, which can fail.
#[cfg(feature = "deflate")]
#[allow(clippy::unnecessary_wraps)]
fn deflate_level(level: &[u8]) -> Result<Vec<u8>> {
    Ok(crate::deflate::zlib_compress(level, crate::deflate::Level::Default))
}

#[cfg(not(feature = "deflate"))]
fn deflate_level(_level: &[u8]) -> Result<Vec<u8>> {
    Err(Error::Unsupported { format: Format::Ktx2, what: "ZLIB supercompression without the deflate feature" })
}

/// The alignment of uncompressed level data: the least common multiple of the block size and 4.
const fn lcm4(block_bytes: u8) -> usize {
    match block_bytes {
        8 => 8,
        16 => 16,
        _ => 4,
    }
}

/// Reads the key/value entries: each a length, a NUL-terminated UTF-8 key and a value, padded to
/// four bytes; keys strictly ascending by their bytes.
fn parse_kvd(data: &[u8], base: usize) -> Result<Vec<(String, Vec<u8>)>> {
    let mut r = Reader::new(data, Format::Ktx2);
    let mut out: Vec<(String, Vec<u8>)> = Vec::new();
    while !r.is_empty() {
        let at = base + r.pos();
        if out.len() == MAX_KEY_VALUES {
            return Err(Error::Limit {
                format: Format::Ktx2,
                what: "key/value entries",
                limit: MAX_KEY_VALUES as u64,
                got: MAX_KEY_VALUES as u64 + 1,
            });
        }
        let len = r.len_u32_le().map_err(|_| invalid(at, "truncated key/value entry"))?;
        let entry = r.bytes(len).map_err(|_| invalid(at, "key/value entry runs past the key/value data"))?;
        let nul = entry.iter().position(|&b| b == 0).ok_or(invalid(at, "key is not NUL-terminated"))?;
        if nul == 0 {
            return Err(invalid(at, "empty key"));
        }
        let key = std::str::from_utf8(&entry[..nul]).map_err(|_| invalid(at, "key is not UTF-8"))?;
        if out.last().is_some_and(|(prev, _)| prev.as_bytes() >= key.as_bytes()) {
            return Err(invalid(at, "keys are not sorted or repeat"));
        }
        out.push((key.to_owned(), entry[nul + 1..].to_vec()));
        // Padding to four bytes; the last entry's may be absent.
        let pad = (4 - (base + r.pos()) % 4) % 4;
        let pad = pad.min(r.remaining());
        if r.bytes(pad)?.iter().any(|&b| b != 0) {
            return Err(invalid(at, "key/value padding is not zero"));
        }
    }
    Ok(out)
}

/// Writes `texture` as a KTX2 file: deterministic, levels stored smallest first, uncompressed levels
/// aligned to the least common multiple of the block size and 4, key/value entries sorted, and a
/// `KTXwriter` entry added when there is none.
pub fn write(texture: &Texture) -> Result<Vec<u8>> {
    texture.validate()?;
    let supercompression = texture.supercompression;
    let payloads: Vec<Vec<u8>> = match supercompression {
        Supercompression::None => Vec::new(),
        Supercompression::Lz => texture.levels.iter().map(|l| crate::lz::compress(l)).collect(),
        Supercompression::Zlib => texture.levels.iter().map(|l| deflate_level(l)).collect::<Result<_>>()?,
    };
    let level_bytes = |i: usize| -> &[u8] { payloads.get(i).map_or(texture.levels[i].as_slice(), Vec::as_slice) };

    let mut entries: Vec<(&str, &[u8])> = texture.key_values.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
    let writer_value = [WRITER_NAME.as_bytes(), b"\0"].concat();
    if !entries.iter().any(|(k, _)| *k == "KTXwriter") {
        entries.push(("KTXwriter", &writer_value));
    }
    entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

    let n = texture.levels.len();
    let descriptor = Dfd::for_format(
        texture.format,
        texture.color_primaries,
        texture.premultiplied_alpha,
        supercompression != Supercompression::None,
    );
    let mut w = Writer::new();
    w.bytes(&IDENTIFIER);
    for v in [
        texture.format.vk_format,
        u32::from(texture.format.type_size),
        texture.width,
        texture.height,
        texture.depth,
        texture.layers,
        texture.faces,
        if texture.generate_mipmaps { 0 } else { n as u32 },
        supercompression.id(),
    ] {
        w.u32_le(v);
    }
    let dfd_offset = HEADER_LEN + LEVEL_ENTRY_LEN * n;
    w.u32_le(dfd_offset as u32);
    w.u32_le(descriptor.byte_len() as u32);
    let kvd_at = w.len();
    w.u32_le(0);
    w.u32_le(0);
    w.u64_le(0);
    w.u64_le(0);
    let index_at = w.len();
    w.bytes(&vec![0; LEVEL_ENTRY_LEN * n]);
    descriptor.write(&mut w);
    let kvd_offset = w.len();
    for (key, value) in &entries {
        w.u32_le((key.len() + 1 + value.len()) as u32);
        w.bytes(key.as_bytes());
        w.u8(0);
        w.bytes(value);
        w.pad_to(4);
    }
    let kvd_len = w.len() - kvd_offset;
    if kvd_len > 0 {
        w.patch_u32_le(kvd_at, kvd_offset as u32);
        w.patch_u32_le(kvd_at + 4, kvd_len as u32);
    }
    let align = if supercompression == Supercompression::None { lcm4(texture.format.block_bytes) } else { 1 };
    for i in (0..n).rev() {
        w.pad_to(align);
        let offset = w.len();
        let bytes = level_bytes(i);
        w.bytes(bytes);
        let at = index_at + LEVEL_ENTRY_LEN * i;
        w.patch_u64_le(at, offset as u64);
        w.patch_u64_le(at + 8, bytes.len() as u64);
        w.patch_u64_le(at + 16, texture.levels[i].len() as u64);
    }
    Ok(w.into_vec())
}

#[cfg(test)]
mod tests;
