//! Tests of the PNG codec: fixtures from an independent Python writer and from ffmpeg, round trips
//! through the encoder, hand-made broken files, and the mutation fuzz `[AC-WP03-1]` `[AC-WP03-2]`.

use super::*;
use crate::deflate::{Level, zlib_compress};
use crate::error::Error;
use crate::image::{Image, Limits, Samples};
use humanart_core::hash::{crc32, crc32_update};
use humanart_core::rng::Rng;
use std::time::{Duration, Instant};

macro_rules! fixtures {
    ($($name:literal),* $(,)?) => {
        [$(($name,
            include_bytes!(concat!("../../tests/fixtures/png/", $name, ".png")).as_slice(),
            include_bytes!(concat!("../../tests/fixtures/png/", $name, ".bin")).as_slice())),*]
    };
}

/// `(name, png, expected)`: see `tests/fixtures/png/make_fixtures.py` for the expected layout.
const FIXTURES: [(&str, &[u8], &[u8]); 54] = fixtures![
    "c0_d1",
    "c0_d1_i",
    "c0_d2",
    "c0_d2_i",
    "c0_d4",
    "c0_d4_i",
    "c0_d8",
    "c0_d8_i",
    "c0_d16",
    "c0_d16_i",
    "c2_d8",
    "c2_d8_i",
    "c2_d16",
    "c2_d16_i",
    "c3_d1",
    "c3_d1_i",
    "c3_d2",
    "c3_d2_i",
    "c3_d4",
    "c3_d4_i",
    "c3_d8",
    "c3_d8_i",
    "c4_d8",
    "c4_d8_i",
    "c4_d16",
    "c4_d16_i",
    "c6_d8",
    "c6_d8_i",
    "c6_d16",
    "c6_d16_i",
    "rgba8_i_1x1",
    "rgba8_i_2x3",
    "rgba8_i_5x1",
    "rgba8_i_1x9",
    "rgba8_i_9x2",
    "grey1_i_3x5",
    "c0_d2_trns",
    "c0_d8_trns_i",
    "c0_d16_trns",
    "c2_d8_trns",
    "c2_d16_trns_i",
    "c3_d4_trns",
    "c3_d8_short",
    "srgb",
    "gama_linear",
    "gama_22",
    "iccp_over_gama",
    "split_idat",
    "ff_rgb24",
    "ff_rgba64",
    "ff_gray16",
    "ff_pal8",
    "ff_monob",
    "ff_ya8",
];

fn fixture(name: &str) -> &'static [u8] {
    FIXTURES.iter().find(|f| f.0 == name).map(|f| f.1).unwrap()
}

/// The image a fixture's `.bin` describes.
fn expected(bin: &[u8]) -> Image {
    let w = u32::from_le_bytes(bin[0..4].try_into().unwrap());
    let h = u32::from_le_bytes(bin[4..8].try_into().unwrap());
    let (ch, bits) = (bin[8], bin[9]);
    let space = [ColorSpace::Srgb, ColorSpace::Linear, ColorSpace::Normal, ColorSpace::Data][usize::from(bin[10])];
    let body = &bin[11..];
    let samples = if bits == 16 {
        Samples::U16(body.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect())
    } else {
        Samples::U8(body.to_vec())
    };
    Image::new(Format::Png, w, h, ch, space, samples).unwrap()
}

/// A PNG made of the given chunks, with correct CRCs.
fn png(chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let mut out = SIGNATURE.to_vec();
    for (kind, data) in chunks {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(*kind);
        out.extend_from_slice(data);
        out.extend_from_slice(&crc32_update(crc32(*kind), data).to_be_bytes());
    }
    out
}

fn ihdr(w: u32, h: u32, depth: u8, color_type: u8, interlace: u8) -> Vec<u8> {
    let mut d = Vec::new();
    d.extend_from_slice(&w.to_be_bytes());
    d.extend_from_slice(&h.to_be_bytes());
    d.extend_from_slice(&[depth, color_type, 0, 0, interlace]);
    d
}

/// Recomputes every chunk CRC, so a mutated file reaches the code behind the CRC check.
fn fix_crcs(bytes: &[u8]) -> Vec<u8> {
    let mut out = bytes.to_vec();
    let mut pos = 8;
    while pos + 12 <= out.len() {
        let len = u32::from_be_bytes(out[pos..pos + 4].try_into().unwrap()) as usize;
        let Some(end) = (pos + 8).checked_add(len).filter(|&e| e + 4 <= out.len()) else { break };
        let crc = crc32_update(crc32(&out[pos + 4..pos + 8]), &out[pos + 8..end]);
        out[end..end + 4].copy_from_slice(&crc.to_be_bytes());
        pos = end + 4;
    }
    out
}

/// A test image: smooth gradients with a little noise, like a texture.
fn texture(width: u32, height: u32, channels: u8, wide: bool, seed: u64) -> Image {
    let mut rng = Rng::new(seed);
    let n = width as usize * height as usize * usize::from(channels);
    let mut values = Vec::with_capacity(n);
    for y in 0..height {
        for x in 0..width {
            for c in 0..u32::from(channels) {
                let smooth = (x * (3 + c) + y * (5 - c.min(4)) + (x * y) / 7) as f64 / 3.0;
                values.push(smooth + rng.range_f64(0.0, 6.0));
            }
        }
    }
    let samples = if wide {
        Samples::U16(values.iter().map(|v| ((v * 257.0) as u64 % 65_536) as u16).collect())
    } else {
        Samples::U8(values.iter().map(|&v| (v as u64 % 256) as u8).collect())
    };
    Image::new(Format::Png, width, height, channels, ColorSpace::Srgb, samples).unwrap()
}

#[test]
fn fixtures_decode_to_the_expected_samples() {
    // [AC-WP03-1] every colour type and bit depth, Adam7, tRNS, palettes, 16-bit, colour chunks; the
    // Python fixtures are cross-checked against ffmpeg (see STATUS), the ff_* ones written by it.
    for (name, file, bin) in FIXTURES {
        let want = expected(bin);
        let (info, got) = decode_with_info(file, &Limits::DEFAULT).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!((got.width, got.height, got.channels), (want.width, want.height, want.channels), "{name}");
        assert_eq!(got.color_space, want.color_space, "{name}");
        assert!(got.samples == want.samples, "{name}: samples differ");
        assert_eq!(info.channels, got.channels, "{name}");
        assert_eq!(read_info(file, &Limits::DEFAULT).unwrap(), info, "{name}: read_info agrees");
    }
}

#[test]
fn colour_chunks_set_the_colour_space_in_the_specified_precedence() {
    // [COL-01] sRGB or iCCP over gAMA; gAMA 1.0 is linear; no chunk is sRGB.
    let info = read_info(fixture("srgb"), &Limits::DEFAULT).unwrap();
    assert_eq!((info.srgb_intent, info.color_space), (Some(0), ColorSpace::Srgb));
    let info = read_info(fixture("gama_linear"), &Limits::DEFAULT).unwrap();
    assert_eq!((info.gamma, info.color_space, info.bit_depth), (Some(100_000), ColorSpace::Linear, 16));
    let info = read_info(fixture("gama_22"), &Limits::DEFAULT).unwrap();
    assert_eq!((info.gamma, info.color_space), (Some(45_455), ColorSpace::Srgb));
    let info = read_info(fixture("iccp_over_gama"), &Limits::DEFAULT).unwrap();
    assert!(info.has_icc_profile);
    assert_eq!(info.color_space, ColorSpace::Srgb, "iCCP takes precedence over a linear gAMA");
    assert_eq!(info.chromaticities, Some([31_270, 32_900, 64_000, 33_000, 30_000, 60_000, 15_000, 6_000]));
    let info = read_info(fixture("c3_d2_i"), &Limits::DEFAULT).unwrap();
    assert_eq!((info.color_type, info.bit_depth, info.interlaced), (ColorType::Indexed, 2, true));
    assert_eq!((info.has_transparency, info.channels, info.color_space), (false, 3, ColorSpace::Srgb));
    let info = read_info(fixture("c3_d4_trns"), &Limits::DEFAULT).unwrap();
    assert_eq!((info.has_transparency, info.channels), (true, 4));
}

#[test]
fn round_trips_every_layout_filter_and_colour_space() {
    // [IO-01] 8/16-bit grey, grey + alpha, RGB, RGBA through every filter choice.
    let filters = [
        FilterChoice::Adaptive,
        FilterChoice::None,
        FilterChoice::Sub,
        FilterChoice::Up,
        FilterChoice::Average,
        FilterChoice::Paeth,
    ];
    let spaces = [ColorSpace::Srgb, ColorSpace::Linear, ColorSpace::Normal, ColorSpace::Data];
    let mut case = 0_u64;
    for channels in 1..=4_u8 {
        for wide in [false, true] {
            for (w, h) in [(1, 1), (1, 7), (7, 1), (33, 17)] {
                for filter in filters {
                    case += 1;
                    let mut image = texture(w, h, channels, wide, case);
                    image.color_space = spaces[case as usize % 4];
                    let level = [Level::Store, Level::Fast, Level::Default, Level::Best][case as usize % 4];
                    let bytes = encode_with(&image, &EncodeOptions { level, filter }).unwrap();
                    let back = decode(&bytes, &Limits::DEFAULT).unwrap();
                    assert_eq!(back, image, "{channels} channels, 16-bit {wide}, {w}x{h}, {filter:?}");
                }
            }
        }
    }
}

#[test]
fn decoded_fixtures_re_encode_losslessly() {
    for (name, file, _) in FIXTURES {
        let image = decode(file, &Limits::DEFAULT).unwrap();
        let back = decode(&encode(&image).unwrap(), &Limits::DEFAULT).unwrap();
        assert_eq!(back, image, "{name}");
    }
}

#[test]
fn adaptive_filtering_beats_no_filtering_on_smooth_images() {
    let image = texture(128, 128, 3, false, 1);
    let none = encode_with(&image, &EncodeOptions { level: Level::Default, filter: FilterChoice::None }).unwrap();
    let adaptive = encode(&image).unwrap();
    assert!(adaptive.len() < none.len(), "adaptive {} vs none {}", adaptive.len(), none.len());
}

#[test]
fn broken_files_are_refused_with_the_right_error() {
    fn what(r: crate::error::Result<Image>) -> &'static str {
        match r {
            Err(Error::Invalid { what, .. } | Error::Unsupported { what, .. } | Error::Limit { what, .. }) => what,
            Err(Error::Truncated { .. }) => "truncated",
            other => panic!("expected an error, got {other:?}"),
        }
    }
    let l = Limits::DEFAULT;
    let grey = ihdr(2, 2, 8, 0, 0);
    let rows = zlib_compress(&[0, 1, 2, 0, 3, 4], Level::Default);
    let good = png(&[(b"IHDR", &grey), (b"IDAT", &rows), (b"IEND", &[])]);
    assert_eq!(decode(&good, &l).unwrap().as_u8(), Some(&[1_u8, 2, 3, 4][..]));

    assert_eq!(what(decode(b"GIF89a, not a PNG", &l)), "not a PNG signature");
    assert_eq!(what(decode(&SIGNATURE[..5], &l)), "truncated");
    assert_eq!(what(decode(&png(&[(b"IDAT", &rows)]), &l)), "the first chunk is not IHDR");
    let mut bad_crc = good.clone();
    bad_crc[8 + 8 + 13] ^= 1;
    assert_eq!(what(decode(&bad_crc, &l)), "bad chunk CRC");
    // An ancillary chunk with a bad CRC is skipped.
    let mut text = png(&[(b"IHDR", &grey), (b"tEXt", b"k\0v"), (b"IDAT", &rows), (b"IEND", &[])]);
    text[8 + 25 + 8 + 3] ^= 1;
    assert!(decode(&text, &l).is_ok());
    assert_eq!(what(decode(&png(&[(b"IHDR", &grey[..12])]), &l)), "IHDR is not 13 bytes long");
    for (depth, ct) in [(3, 0), (4, 2), (16, 3), (1, 6), (8, 1), (8, 7)] {
        let h = ihdr(2, 2, depth, ct, 0);
        assert_eq!(what(decode(&png(&[(b"IHDR", &h)]), &l)), "illegal colour type and bit depth combination");
    }
    assert_eq!(
        what(decode(&png(&[(b"IHDR", &ihdr(0, 2, 8, 0, 0))]), &l)),
        "image width or height is 0 or over 2^31 - 1"
    );
    assert_eq!(what(decode(&png(&[(b"IHDR", &ihdr(2, 2, 8, 0, 2))]), &l)), "unknown interlace method");
    let huge = png(&[(b"IHDR", &ihdr(5000, 1, 8, 0, 0)), (b"IDAT", &rows), (b"IEND", &[])]);
    assert_eq!(what(decode(&huge, &Limits::FUZZ)), "image dimension");
    assert!(matches!(read_info(&huge, &Limits::FUZZ), Err(Error::Limit { .. })));
    let bomb = png(&[(b"IHDR", &ihdr(30_000, 30_000, 16, 6, 0)), (b"IDAT", &rows), (b"IEND", &[])]);
    assert!(matches!(decode(&bomb, &Limits::DEFAULT), Err(Error::Limit { .. })), "refused before allocating");
    assert_eq!(
        what(decode(&png(&[(b"IHDR", &grey), (b"ABCD", &[]), (b"IDAT", &rows), (b"IEND", &[])]), &l)),
        "unknown critical chunk"
    );
    assert!(decode(&png(&[(b"IHDR", &grey), (b"abCD", &[1]), (b"IDAT", &rows), (b"IEND", &[])]), &l).is_ok());
    assert_eq!(
        what(decode(&png(&[(b"IHDR", &grey), (b"ID1T", &[]), (b"IEND", &[])]), &l)),
        "chunk type is not four letters"
    );
    let mut long = png(&[(b"IHDR", &grey), (b"IDAT", &rows)]);
    long[33..37].copy_from_slice(&0x8000_0000_u32.to_be_bytes());
    assert_eq!(what(decode(&long, &l)), "chunk length over 2^31 - 1");
    assert_eq!(
        what(decode(&png(&[(b"IHDR", &ihdr(2, 2, 8, 3, 0)), (b"IDAT", &rows), (b"IEND", &[])]), &l)),
        "indexed image without a PLTE chunk"
    );
    let indexed = ihdr(2, 2, 8, 3, 0);
    let two_colours = [0_u8, 0, 0, 255, 255, 255];
    let ok = png(&[
        (b"IHDR", &indexed),
        (b"PLTE", &two_colours),
        (b"IDAT", &zlib_compress(&[0, 0, 1, 0, 1, 0], Level::Fast)),
        (b"IEND", &[]),
    ]);
    assert_eq!(decode(&ok, &l).unwrap().as_u8().unwrap().len(), 12);
    let bad_index = png(&[(b"IHDR", &indexed), (b"PLTE", &two_colours), (b"IDAT", &rows), (b"IEND", &[])]);
    assert_eq!(what(decode(&bad_index, &l)), "palette index out of range");
    assert_eq!(
        what(decode(&png(&[(b"IHDR", &indexed), (b"tRNS", &[0]), (b"PLTE", &two_colours)]), &l)),
        "tRNS chunk before PLTE"
    );
    assert_eq!(what(decode(&png(&[(b"IHDR", &grey), (b"PLTE", &two_colours)]), &l)), "PLTE chunk in a grey image");
    assert_eq!(
        what(decode(&png(&[(b"IHDR", &ihdr(2, 2, 8, 2, 0)), (b"PLTE", &two_colours[..4])]), &l)),
        "PLTE length is not 3 to 768 bytes in whole entries"
    );
    let bad_filter = zlib_compress(&[5, 1, 2, 0, 3, 4], Level::Default);
    assert_eq!(
        what(decode(&png(&[(b"IHDR", &grey), (b"IDAT", &bad_filter), (b"IEND", &[])]), &l)),
        "unknown filter type"
    );
    let short = zlib_compress(&[0, 1, 2], Level::Default);
    assert_eq!(
        what(decode(&png(&[(b"IHDR", &grey), (b"IDAT", &short), (b"IEND", &[])]), &l)),
        "image data is shorter than the image"
    );
    let long_data = zlib_compress(&[0, 1, 2, 0, 3, 4, 0], Level::Default);
    assert_eq!(
        what(decode(&png(&[(b"IHDR", &grey), (b"IDAT", &long_data), (b"IEND", &[])]), &l)),
        "image data is longer than the image"
    );
    let (first, second) = rows.split_at(4);
    assert_eq!(
        what(decode(
            &png(&[(b"IHDR", &grey), (b"IDAT", first), (b"tEXt", b"a\0b"), (b"IDAT", second), (b"IEND", &[])]),
            &l
        )),
        "IDAT chunks are not consecutive"
    );
    assert!(decode(&png(&[(b"IHDR", &grey), (b"IDAT", first), (b"IDAT", second), (b"IEND", &[])]), &l).is_ok());
    assert_eq!(what(decode(&png(&[(b"IHDR", &grey), (b"IEND", &[])]), &l)), "no IDAT chunk");
    assert_eq!(what(decode(&png(&[(b"IHDR", &grey), (b"IDAT", &rows)]), &l)), "truncated");
    // Every strict prefix of a valid file is an error.
    let small = fixture("c3_d4_trns");
    for cut in 0..small.len() {
        assert!(decode(&small[..cut], &l).is_err(), "prefix of {cut} bytes");
    }
}

#[test]
fn the_encoder_refuses_what_png_cannot_hold() {
    let float = Image::new(Format::Png, 1, 1, 1, ColorSpace::Linear, Samples::F32(vec![0.5])).unwrap();
    assert!(matches!(encode(&float), Err(Error::Encode { .. })));
    let mut bad = Image::new(Format::Png, 2, 1, 1, ColorSpace::Srgb, Samples::U8(vec![1, 2])).unwrap();
    bad.samples = Samples::U8(vec![1]);
    assert!(matches!(encode(&bad), Err(Error::Encode { what: "sample count does not match the dimensions", .. })));
    bad.channels = 5;
    assert!(matches!(encode(&bad), Err(Error::Encode { what: "PNG holds 1 to 4 channels", .. })));
}

/// Small fixtures of every structure, for the fuzz corpus.
fn fuzz_corpus() -> Vec<Vec<u8>> {
    ["c0_d1_i", "c3_d4_trns", "c2_d16_trns_i", "c6_d8_i", "split_idat", "iccp_over_gama", "c0_d2_trns", "rgba8_i_2x3"]
        .iter()
        .map(|name| fixture(name).to_vec())
        .collect()
}

#[test]
fn mutation_fuzz_never_panics() {
    // [ROB-01] [AC-WP03-2] mutated files as they come (most fail a CRC) ...
    humanart_core::fuzz::run("png", &fuzz_corpus(), 800, 0x0000_9A9E, Duration::from_secs(120), |bytes| {
        let _ = read_info(bytes, &Limits::FUZZ);
        let _ = decode(bytes, &Limits::FUZZ);
    });
}

#[test]
fn mutation_fuzz_behind_the_crc_check_never_panics_and_round_trips() {
    // ... and with their CRCs repaired, so the mutations reach IHDR, PLTE, tRNS, the zlib stream and
    // the filters. Whatever decodes must survive an encode/decode round trip unchanged.
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let decoded = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&decoded);
    humanart_core::fuzz::run("png-crc", &fuzz_corpus(), 800, 0x00C3_C0DE, Duration::from_secs(120), move |bytes| {
        let fixed = fix_crcs(bytes);
        let info = read_info(&fixed, &Limits::FUZZ);
        if let Ok((full, image)) = decode_with_info(&fixed, &Limits::FUZZ) {
            counter.fetch_add(1, Ordering::Relaxed);
            assert_eq!(info.as_ref().ok(), Some(&full), "read_info and decode agree");
            let again = decode(&encode(&image).unwrap(), &Limits::FUZZ).unwrap();
            assert!(again == image, "round trip of a fuzzed image");
        }
    });
    // The repaired mutations do reach the end of the decoder, not just its first checks.
    let decoded = decoded.load(Ordering::Relaxed);
    println!("png-crc fuzz: {decoded} of {} cases decoded", 8 * 801);
    assert!(decoded > 200, "only {decoded} fuzz cases decoded");
}

/// Measures throughput: `cargo test --release -p humanart-io --features png -- --ignored --nocapture
/// png_throughput`.
#[test]
#[ignore = "timing benchmark; run on demand in release mode"]
fn png_throughput() {
    for (channels, wide) in [(4_u8, false), (3, false), (4, true)] {
        let image = texture(4096, 2048, channels, wide, 3);
        let mb = (image.samples.len() * image.samples.sample_size()) as f64 / 1e6;
        for level in [Level::Fast, Level::Default] {
            let t = Instant::now();
            let bytes = encode_with(&image, &EncodeOptions { level, filter: FilterChoice::Adaptive }).unwrap();
            let enc = t.elapsed().as_secs_f64();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = Instant::now();
                let back = decode(&bytes, &Limits::DEFAULT).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
                assert_eq!(back.samples.len(), image.samples.len());
            }
            println!(
                "4096x2048 {channels} ch {}-bit, {level:?}: {:.1}% of raw, encode {:.0} MB/s, decode {:.0} MB/s output",
                if wide { 16 } else { 8 },
                100.0 * bytes.len() as f64 / (mb * 1e6),
                mb / enc,
                mb / best
            );
        }
    }
}

/// Writes our encoding of every fixture's image to `target/agents/io-png/xcheck/png`, with the
/// expected samples, so `make_fixtures.py --check <dir>` can decode them with ffmpeg.
#[test]
#[ignore = "writes files for the dev-time cross-check with ffmpeg"]
fn write_cross_check_files() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/io-png/xcheck/png");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, _, bin) in FIXTURES {
        let image = expected(bin);
        std::fs::write(dir.join(format!("{name}.png")), encode(&image).unwrap()).unwrap();
        std::fs::write(dir.join(format!("{name}.bin")), bin).unwrap();
    }
}
