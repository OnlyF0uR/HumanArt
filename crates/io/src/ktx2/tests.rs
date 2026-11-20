//! KTX2 tests: a file built byte by byte from the KTX2 specification, round trips of every format and
//! shape through every supercompression scheme, image conversion, rejection of broken files and the
//! mutation fuzz `[IO-04]` `[AC-WP03-1]` `[AC-WP03-2]` `[ROB-01]`.

use super::*;
use humanart_core::fuzz;
use humanart_core::rng::Rng;
use std::time::Duration;

fn u32s(out: &mut Vec<u8>, values: &[u32]) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// A 2×2 `R8G8B8A8_UNORM` file with one level and a `KTXwriter` entry, laid out by hand from the
/// KTX2 specification (section 3) and the Khronos Data Format specification (section 5).
fn hand_built() -> Vec<u8> {
    let mut f = IDENTIFIER.to_vec();
    // vkFormat, typeSize, width, height, depth, layers, faces, levels, supercompression.
    u32s(&mut f, &[37, 1, 2, 2, 0, 0, 1, 1, 0]);
    // DFD at 80 + 24 = 104, 92 bytes; KVD at 196, 20 bytes; no SGD.
    u32s(&mut f, &[104, 92, 196, 20]);
    f.extend_from_slice(&0_u64.to_le_bytes());
    f.extend_from_slice(&0_u64.to_le_bytes());
    // Level 0 at 216, 16 bytes, 16 bytes uncompressed.
    for v in [216_u64, 16, 16] {
        f.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(f.len(), 104);
    // dfdTotalSize; vendor 0 / type 0; version 2 | blockSize 88 << 16.
    u32s(&mut f, &[92, 0, 2 | (88 << 16)]);
    // RGBSDA, BT.709, linear, straight alpha; 1×1×1×1 texel block; 4 bytes in plane 0.
    f.extend_from_slice(&[1, 1, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0]);
    for (channel, offset) in [(0_u32, 0_u32), (1, 8), (2, 16), (15, 24)] {
        u32s(&mut f, &[offset | (7 << 16) | (channel << 24), 0, 0, 255]);
    }
    assert_eq!(f.len(), 196);
    u32s(&mut f, &[15]);
    f.extend_from_slice(b"KTXwriter\0test\0\0");
    assert_eq!(f.len(), 216);
    f.extend_from_slice(&[255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 9, 9, 9, 0]);
    f
}

#[test]
fn hand_built_file_reads_and_rewrites_identically() {
    let file = hand_built();
    let header = read_header(&file).unwrap();
    assert_eq!(header.levels, [LevelEntry { offset: 216, length: 16, uncompressed_length: 16 }]);
    let t = read(&file, &Limits::DEFAULT).unwrap();
    assert_eq!(t.format.name, "R8G8B8A8_UNORM");
    assert_eq!((t.width, t.height, t.depth, t.layers, t.faces), (2, 2, 0, 0, 1));
    assert_eq!(t.key_values, [("KTXwriter".to_owned(), b"test\0".to_vec())]);
    let image = t.to_image(0, 0, 0).unwrap();
    assert_eq!(image.as_u8().unwrap()[12..], [9, 9, 9, 0]);
    // The writer lays the same file out byte for byte.
    assert_eq!(write(&t).unwrap(), file);
}

fn random_texture(
    rng: &mut Rng,
    format: &'static FormatInfo,
    shape: (u32, u32, u32, u32, u32),
    all_levels: bool,
) -> Texture {
    let (width, height, depth, layers, faces) = shape;
    let mut t = Texture {
        format,
        width,
        height,
        depth,
        layers,
        faces,
        levels: Vec::new(),
        generate_mipmaps: false,
        color_primaries: 1,
        premultiplied_alpha: false,
        key_values: vec![("KTXorientation".into(), b"rd\0".to_vec()), ("AAA custom".into(), vec![1, 2, 3])],
        supercompression: Supercompression::None,
    };
    let count = if all_levels { t.max_levels() } else { 1 };
    for level in 0..count {
        let mut data = vec![0_u8; t.level_size(level).unwrap() as usize];
        // Half random, half zeros, so LZ has something to find.
        let half = data.len() / 2;
        rng.fill(&mut data[..half]);
        t.levels.push(data);
    }
    t
}

fn schemes() -> Vec<Supercompression> {
    let mut s = vec![Supercompression::None, Supercompression::Lz];
    if cfg!(feature = "deflate") {
        s.push(Supercompression::Zlib);
    }
    s
}

#[test]
fn every_format_round_trips_with_mip_chains_ac_wp03_1() {
    let mut rng = Rng::new(0x6B7478);
    for format in FORMATS {
        for scheme in schemes() {
            let mut t = random_texture(&mut rng, format, (13, 7, 0, 0, 1), true);
            t.supercompression = scheme;
            let file = write(&t).unwrap();
            assert_eq!(write(&t).unwrap(), file, "deterministic");
            let mut back = read(&file, &Limits::DEFAULT).unwrap();
            assert_eq!(
                back.key_values.iter().find(|(k, _)| k == "KTXwriter").map(|(_, v)| v.clone()),
                Some([WRITER_NAME.as_bytes(), b"\0"].concat())
            );
            back.key_values.retain(|(k, _)| k != "KTXwriter");
            back.key_values.sort();
            t.key_values.sort();
            assert_eq!(back, t, "{} {scheme:?}", format.name);
        }
    }
}

#[test]
fn arrays_cubes_volumes_and_1d_round_trip() {
    let mut rng = Rng::new(5);
    let rgba = FormatInfo::from_vk(vk::R8G8B8A8_SRGB).unwrap();
    let bc7 = FormatInfo::from_vk(vk::BC7_UNORM).unwrap();
    let half = FormatInfo::from_vk(vk::R16G16B16A16_SFLOAT).unwrap();
    for (format, shape) in [
        (bc7, (16, 16, 0, 3, 6)),
        (bc7, (20, 12, 0, 4, 1)),
        (rgba, (9, 0, 0, 0, 1)),
        (half, (5, 6, 7, 0, 1)),
        (rgba, (8, 8, 0, 2, 6)),
    ] {
        for scheme in schemes() {
            let mut t = random_texture(&mut rng, format, shape, true);
            t.supercompression = scheme;
            t.premultiplied_alpha = true;
            let file = write(&t).unwrap();
            let mut back = read(&file, &Limits::DEFAULT).unwrap();
            back.key_values.retain(|(k, _)| k != "KTXwriter");
            t.key_values.sort();
            assert_eq!(back, t, "{shape:?} {scheme:?}");
            // Every image of every level is addressable and has the block-derived size.
            for level in 0..t.levels.len() as u32 {
                for layer in 0..shape.3.max(1) {
                    for face in 0..shape.4 {
                        assert_eq!(
                            back.image(level, layer, face).unwrap().len() as u64,
                            back.image_size(level).unwrap()
                        );
                    }
                }
                assert!(back.image(level, shape.3.max(1), 0).is_none());
            }
        }
    }
}

#[test]
fn mip_sizes_follow_block_dimensions() {
    // A 10×6 ASTC 6×6 texture: levels 10×6, 5×3, 2×1, 1×1 each need 2, 1, 1, 1 blocks of 16 bytes.
    let astc = FormatInfo::from_vk(165).unwrap();
    let t = Texture::new_2d(astc, 10, 6, vec![vec![0; 32], vec![0; 16], vec![0; 16], vec![0; 16]]).unwrap();
    assert_eq!(t.max_levels(), 4);
    assert_eq!(t.level_extent(2), (2, 1, 1));
    // One level too many, or a level of the wrong size, is refused.
    assert!(
        Texture::new_2d(astc, 10, 6, vec![vec![0; 32], vec![0; 16], vec![0; 16], vec![0; 16], vec![0; 16]]).is_err()
    );
    assert!(Texture::new_2d(astc, 10, 6, vec![vec![0; 16]]).is_err());
    // A BC1 1×1 level is still one 8-byte block.
    let bc1 = FormatInfo::from_vk(131).unwrap();
    assert!(Texture::new_2d(bc1, 1, 1, vec![vec![0; 8]]).is_ok());
}

#[test]
fn images_convert_both_ways() {
    let rgba = Image::new(Format::Png, 3, 2, 4, ColorSpace::Srgb, Samples::U8((0..24).collect())).unwrap();
    let t = Texture::from_image(&rgba).unwrap();
    assert_eq!(t.format.vk_format, vk::R8G8B8A8_SRGB);
    let back = read(&write(&t).unwrap(), &Limits::DEFAULT).unwrap().to_image(0, 0, 0).unwrap();
    assert_eq!(back, rgba);

    let rgb = Image::new(Format::Png, 1, 1, 3, ColorSpace::Linear, Samples::U8(vec![1, 2, 3])).unwrap();
    let t = Texture::from_image(&rgb).unwrap();
    assert_eq!(t.levels[0], [1, 2, 3, 255]);

    let half =
        Image::new(Format::Exr, 2, 1, 4, ColorSpace::Linear, Samples::F16(vec![1, 2, 3, 4, 5, 6, 7, 8])).unwrap();
    let t = Texture::from_image(&half).unwrap();
    assert_eq!(t.format.vk_format, vk::R16G16B16A16_SFLOAT);
    assert_eq!(t.to_image(0, 0, 0).unwrap(), half);

    let float = Image::new(Format::Exr, 1, 2, 2, ColorSpace::Data, Samples::F32(vec![0.5, -1.0, 2.0, 3.0])).unwrap();
    let back = Texture::from_image(&float).unwrap().to_image(0, 0, 0).unwrap();
    assert_eq!(back.samples, float.samples);

    let deep = Image::new(Format::Png, 1, 1, 1, ColorSpace::Data, Samples::U16(vec![60000])).unwrap();
    assert_eq!(Texture::from_image(&deep).unwrap().to_image(0, 0, 0).unwrap().samples, deep.samples);
    let srgb16 = Image::new(Format::Png, 1, 1, 1, ColorSpace::Srgb, Samples::U16(vec![1])).unwrap();
    assert!(Texture::from_image(&srgb16).is_err());

    // A mip chain of 4×2, 2×1, 1×1.
    let level = |w: u32, h: u32| {
        Image::new(Format::Png, w, h, 4, ColorSpace::Srgb, Samples::U8(vec![7; (w * h * 4) as usize])).unwrap()
    };
    let chain = Texture::from_mip_chain(&[level(4, 2), level(2, 1), level(1, 1)]).unwrap();
    assert_eq!(chain.levels.len(), 3);
    assert!(Texture::from_mip_chain(&[level(4, 2), level(1, 1)]).is_err());
    assert!(Texture::from_mip_chain(&[]).is_err());

    // BGRA is swizzled to RGBA; block formats are not decoded.
    let bgra = Texture::new_2d(FormatInfo::from_vk(44).unwrap(), 1, 1, vec![vec![1, 2, 3, 4]]).unwrap();
    assert_eq!(bgra.to_image(0, 0, 0).unwrap().as_u8(), Some(&[3, 2, 1, 4][..]));
    let bc7 = Texture::new_2d(FormatInfo::from_vk(vk::BC7_UNORM).unwrap(), 4, 4, vec![vec![0; 16]]).unwrap();
    assert!(matches!(bc7.to_image(0, 0, 0), Err(Error::Unsupported { .. })));
}

fn patch_u32(file: &mut [u8], at: usize, v: u32) {
    file[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn patch_u64(file: &mut [u8], at: usize, v: u64) {
    file[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

#[test]
fn broken_files_are_refused() {
    let good = hand_built();
    let refuse = |f: &[u8]| read(f, &Limits::DEFAULT).unwrap_err();
    let mut f = good.clone();
    f[0] = 0;
    assert!(matches!(refuse(&f), Error::Invalid { offset: 0, .. }));
    // typeSize, vkFormat, schemes.
    let mut f = good.clone();
    patch_u32(&mut f, 16, 2);
    assert!(matches!(refuse(&f), Error::Invalid { .. }));
    for (vk_format, scheme) in [(0, 0), (1000, 0), (37, SCHEME_BASIS_LZ), (37, SCHEME_ZSTD), (37, 77)] {
        let mut f = good.clone();
        patch_u32(&mut f, 12, vk_format);
        patch_u32(&mut f, 44, scheme);
        assert!(matches!(refuse(&f), Error::Unsupported { .. }), "{vk_format} {scheme}");
    }
    // A cube map that is not square; face count 2; too many levels for a 2×2 texture.
    let mut f = good.clone();
    patch_u32(&mut f, 24, 6);
    patch_u32(&mut f, 36, 6);
    assert!(refuse(&f).to_string().contains("square"));
    let mut f = good.clone();
    patch_u32(&mut f, 36, 2);
    assert!(refuse(&f).to_string().contains("face count"));
    let mut f = good.clone();
    patch_u32(&mut f, 40, MAX_LEVELS + 1);
    assert!(matches!(refuse(&f), Error::Limit { .. }));
    // Level size, range and alignment.
    let mut f = good.clone();
    patch_u64(&mut f, 96, 15);
    assert!(refuse(&f).to_string().contains("uncompressedByteLength"));
    let mut f = good.clone();
    patch_u64(&mut f, 80, 218);
    assert!(matches!(refuse(&f), Error::Truncated { .. }));
    let mut f = good.clone();
    f.extend_from_slice(&[0; 4]);
    patch_u64(&mut f, 80, 218);
    assert!(refuse(&f).to_string().contains("misaligned"));
    let mut f = good.clone();
    patch_u64(&mut f, 80, 200);
    assert!(refuse(&f).to_string().contains("metadata"));
    // DFD and KVD placement, KVD content.
    let mut f = good.clone();
    patch_u32(&mut f, 48, 108);
    assert!(refuse(&f).to_string().contains("DFD"));
    let mut f = good.clone();
    f[104 + 14] = 2;
    assert!(refuse(&f).to_string().contains("transfer"));
    let mut f = good.clone();
    patch_u32(&mut f, 196, 100);
    assert!(refuse(&f).to_string().contains("key/value"));
    let mut f = good.clone();
    f[200 + 9] = b'x';
    f[200 + 14] = b'x';
    assert!(refuse(&f).to_string().contains("NUL"));
    let mut f = good.clone();
    patch_u64(&mut f, 64, 8);
    assert!(refuse(&f).to_string().contains("global data"));
    // Every truncation fails without panicking.
    for len in 0..good.len() {
        assert!(read(&good[..len], &Limits::DEFAULT).is_err(), "{len}");
    }
}

#[test]
fn level_order_is_enforced() {
    let mut rng = Rng::new(9);
    let t = random_texture(&mut rng, FormatInfo::from_vk(vk::R8G8B8A8_UNORM).unwrap(), (4, 4, 0, 0, 1), true);
    let mut file = write(&t).unwrap();
    let header = read_header(&file).unwrap();
    // Swap the index entries of levels 1 and 2 so the larger level is placed first.
    let (a, b) = (header.levels[1], header.levels[2]);
    assert!(b.offset < a.offset, "the writer stores smaller levels first");
    patch_u64(&mut file, HEADER_LEN + LEVEL_ENTRY_LEN, b.offset);
    patch_u64(&mut file, HEADER_LEN + 2 * LEVEL_ENTRY_LEN, a.offset);
    assert!(read(&file, &Limits::DEFAULT).is_err());
}

#[test]
fn unsorted_keys_and_bombs_are_refused() {
    let mut t = Texture::new_2d(FormatInfo::from_vk(9).unwrap(), 1, 1, vec![vec![5]]).unwrap();
    t.key_values = vec![("b".into(), vec![]), ("a".into(), vec![1])];
    let mut file = write(&t).unwrap();
    // The writer sorted them; reverse the first byte of each key to break the order.
    let a = file.windows(2).position(|w| w == b"a\0").unwrap();
    let b = file.windows(2).position(|w| w == b"b\0").unwrap();
    file[a] = b'b';
    file[b] = b'a';
    assert!(read(&file, &Limits::DEFAULT).unwrap_err().to_string().contains("sorted"));
    t.key_values.push(("a".into(), vec![]));
    assert!(write(&t).is_err(), "duplicate key");

    // A header promising 4096² RGBA32F is refused by the FUZZ limits before any allocation.
    let mut f = hand_built();
    patch_u32(&mut f, 12, vk::R32G32B32A32_SFLOAT);
    patch_u32(&mut f, 16, 4);
    patch_u32(&mut f, 20, 4096);
    patch_u32(&mut f, 24, 4096);
    assert!(matches!(read(&f, &Limits::FUZZ), Err(Error::Limit { .. })));
    // An LZ level whose frame promises more than the level holds.
    let mut lz = Texture::new_2d(FormatInfo::from_vk(9).unwrap(), 64, 64, vec![vec![3; 4096]]).unwrap();
    lz.supercompression = Supercompression::Lz;
    let mut file = write(&lz).unwrap();
    let start = HEADER_LEN + file[HEADER_LEN..].windows(4).position(|w| w == b"HALZ").unwrap();
    patch_u64(&mut file, start + 8, 1 << 40);
    assert!(read(&file, &Limits::DEFAULT).is_err());
}

#[test]
fn mutation_fuzz_rob_01() {
    let mut rng = Rng::new(0xF22);
    let mut corpus = vec![hand_built()];
    for (vk_format, shape) in [
        (vk::BC7_SRGB, (8, 8, 0, 0, 1)),
        (165, (12, 6, 0, 2, 1)),
        (vk::R16G16B16A16_SFLOAT, (4, 4, 0, 0, 6)),
        (9, (5, 3, 2, 0, 1)),
    ] {
        for scheme in schemes() {
            let mut t = random_texture(&mut rng, FormatInfo::from_vk(vk_format).unwrap(), shape, true);
            t.supercompression = scheme;
            corpus.push(write(&t).unwrap());
        }
    }
    fuzz::run("ktx2", &corpus, 400, 0x4B54_5832, Duration::from_secs(120), |bytes| {
        if let Ok(t) = read(bytes, &Limits::FUZZ) {
            // Whatever is accepted is consistent enough to be written again.
            let again = write(&t).expect("accepted texture rewrites");
            assert!(read(&again, &Limits::FUZZ).is_ok());
        }
        let _ = read_header(bytes);
    });
}
