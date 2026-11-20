//! Tests of the DEFLATE and zlib codecs: fixtures made by Python's zlib, round trips through the
//! encoder at every level, hand-made broken streams, and the mutation fuzz `[AC-WP03-1]`
//! `[AC-WP03-2]`.

use super::*;
use humanart_core::rng::Rng;
use std::time::{Duration, Instant};

const INPUT: &[u8] = include_bytes!("../../tests/fixtures/deflate/input.bin");
const ZLIB_L0: &[u8] = include_bytes!("../../tests/fixtures/deflate/zlib_l0.zz");
const ZLIB_L1: &[u8] = include_bytes!("../../tests/fixtures/deflate/zlib_l1.zz");
const ZLIB_L6: &[u8] = include_bytes!("../../tests/fixtures/deflate/zlib_l6.zz");
const ZLIB_L9: &[u8] = include_bytes!("../../tests/fixtures/deflate/zlib_l9.zz");
const ZLIB_W9: &[u8] = include_bytes!("../../tests/fixtures/deflate/zlib_w9.zz");
const ZLIB_FIXED: &[u8] = include_bytes!("../../tests/fixtures/deflate/zlib_fixed.zz");
const ZLIB_HUFFMAN: &[u8] = include_bytes!("../../tests/fixtures/deflate/zlib_huffman.zz");
const ZLIB_RLE: &[u8] = include_bytes!("../../tests/fixtures/deflate/zlib_rle.zz");
const RAW_L6: &[u8] = include_bytes!("../../tests/fixtures/deflate/raw_l6.deflate");
const ZEROS_4M: &[u8] = include_bytes!("../../tests/fixtures/deflate/zeros_4m.zz");
const ZLIB_DICT: &[u8] = include_bytes!("../../tests/fixtures/deflate/zlib_dict.zz");

const LEVELS: [Level; 4] = [Level::Store, Level::Fast, Level::Default, Level::Best];

/// Inputs of every character: empty, tiny, runs, periodic, text-like, noise, and mixtures long
/// enough to span several blocks and the whole window.
fn sample_inputs() -> Vec<Vec<u8>> {
    let mut rng = Rng::new(0xDEF1_A7E5);
    let mut noise = vec![0_u8; 70_000];
    rng.fill(&mut noise);
    let text: Vec<u8> = INPUT[..30_000].to_vec();
    let mut mixed = Vec::new();
    for i in 0..40 {
        let take = 1000 + rng.index(9000);
        let from = rng.index(INPUT.len() - take);
        mixed.extend_from_slice(&INPUT[from..from + take]);
        if i % 3 == 0 {
            mixed.extend(std::iter::repeat_n((i * 7) as u8, rng.index(600)));
        }
    }
    let periodic: Vec<u8> = (0..50_000_usize).map(|i| [1_u8, 2, 3, 5, 8, 13, 21][i % 7]).collect();
    vec![
        Vec::new(),
        vec![42],
        vec![7, 7],
        b"abcabcabc".to_vec(),
        vec![0; 100_000],
        vec![0xFF; 258 * 3 + 2],
        periodic,
        text,
        noise,
        mixed,
        INPUT.to_vec(),
    ]
}

#[test]
fn python_zlib_fixtures_decode_exactly() {
    // [AC-WP03-1] every block type and both window sizes, as zlib 1.3.1 (zlib-ng) writes them.
    for (name, stream) in [
        ("l0", ZLIB_L0),
        ("l1", ZLIB_L1),
        ("l6", ZLIB_L6),
        ("l9", ZLIB_L9),
        ("w9", ZLIB_W9),
        ("fixed", ZLIB_FIXED),
        ("huffman", ZLIB_HUFFMAN),
        ("rle", ZLIB_RLE),
    ] {
        let out = zlib_decompress(stream, 1 << 20).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(out == INPUT, "{name}: output differs");
        let mut exact = vec![0_u8; INPUT.len()];
        let d = zlib_decompress_into(stream, &mut exact).unwrap_or_else(|e| panic!("{name} into: {e}"));
        assert_eq!(d, Decoded { written: INPUT.len(), consumed: stream.len() }, "{name}");
        assert!(exact == INPUT, "{name}: slice output differs");
    }
    assert_eq!(inflate(RAW_L6, 1 << 20).unwrap(), INPUT);
    assert_eq!(adler32(INPUT), 0x3c27_4040, "the script's recorded checksum");
}

#[test]
fn decompression_bombs_stop_at_the_bound() {
    // [ROB-02] 4 MiB of zeros in 4 KiB: refused under a 1 MiB bound, before it is written.
    match zlib_decompress(ZEROS_4M, 1 << 20) {
        Err(Error::Limit { what: "decompressed size", limit, .. }) => assert_eq!(limit, 1 << 20),
        other => panic!("expected a limit error, got {other:?}"),
    }
    let mut small = vec![0_u8; 4 << 10];
    assert!(matches!(zlib_decompress_into(ZEROS_4M, &mut small), Err(Error::Limit { .. })));
    let out = zlib_decompress(ZEROS_4M, 4 << 20).unwrap();
    assert_eq!(out.len(), 4 << 20);
    assert!(out.iter().all(|&b| b == 0));
    // The exact bound passes, one byte less fails.
    assert!(zlib_decompress(ZLIB_L6, INPUT.len()).is_ok());
    assert!(matches!(zlib_decompress(ZLIB_L6, INPUT.len() - 1), Err(Error::Limit { .. })));
    let mut short = vec![0_u8; INPUT.len() - 1];
    assert!(matches!(zlib_decompress_into(ZLIB_L6, &mut short), Err(Error::Limit { .. })));
}

#[test]
fn round_trips_at_every_level() {
    // [IO-01] the encoder's output decodes to its input, at every level and through both sinks.
    for (i, input) in sample_inputs().iter().enumerate() {
        for level in LEVELS {
            let raw = deflate(input, level);
            let back = inflate(&raw, input.len()).unwrap_or_else(|e| panic!("input {i}, {level:?}: {e}"));
            assert!(&back == input, "input {i}, {level:?}: raw round trip differs");
            let z = zlib_compress(input, level);
            let back = zlib_decompress(&z, input.len()).unwrap_or_else(|e| panic!("input {i}, {level:?}: {e}"));
            assert!(&back == input, "input {i}, {level:?}: zlib round trip differs");
            let mut exact = vec![0_u8; input.len()];
            let d = zlib_decompress_into(&z, &mut exact).unwrap();
            assert_eq!(d, Decoded { written: input.len(), consumed: z.len() });
            assert_eq!(&exact, input);
        }
    }
}

#[test]
fn levels_trade_speed_for_size() {
    let sizes: Vec<usize> = LEVELS.iter().map(|&level| deflate(INPUT, level).len()).collect();
    assert!(sizes[0] > sizes[1] && sizes[1] >= sizes[2] && sizes[2] >= sizes[3], "sizes {sizes:?}");
    // Within a few percent of zlib at the matching levels (zlib-ng 1.3.1: l1 32607, l6 28207, l9
    // 28156 bytes for this input).
    let zlib6 = zlib_compress(INPUT, Level::Default).len();
    let zlib9 = zlib_compress(INPUT, Level::Best).len();
    assert!(zlib6 * 100 <= ZLIB_L6.len() * 104, "Default: {zlib6} vs zlib -6 {}", ZLIB_L6.len());
    assert!(zlib9 * 100 <= ZLIB_L9.len() * 104, "Best: {zlib9} vs zlib -9 {}", ZLIB_L9.len());
    // Noise is stored, not expanded by more than the block headers (5 bytes per block of 16384
    // symbols).
    let mut noise = vec![0_u8; 100_000];
    Rng::new(5).fill(&mut noise);
    assert!(deflate(&noise, Level::Best).len() <= noise.len() + 5 * 7 + 8);
}

#[test]
fn hand_made_streams_decode() {
    // A stored block, then the same five bytes as a fixed block, as zlib writes them.
    assert_eq!(inflate(&[0x01, 0x05, 0x00, 0xFA, 0xFF, b'h', b'e', b'l', b'l', b'o'], 10).unwrap(), b"hello");
    assert_eq!(inflate(&[0xCB, 0x48, 0xCD, 0xC9, 0xC9, 0x07, 0x00], 10).unwrap(), b"hello");
    // An empty fixed block; an empty stored block; an empty zlib stream.
    assert!(inflate(&[0x03, 0x00], 0).unwrap().is_empty());
    assert!(inflate(&[0x01, 0x00, 0x00, 0xFF, 0xFF], 0).unwrap().is_empty());
    assert!(zlib_decompress(&[0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01], 0).unwrap().is_empty());
    // A non-final stored block followed by a final fixed one; trailing bytes are not consumed.
    let stream = [0x00, 0x01, 0x00, 0xFE, 0xFF, b'x', 0x03, 0x00, 0xAA, 0xBB];
    let mut out = [0_u8; 4];
    assert_eq!(inflate_into(&stream, &mut out).unwrap(), Decoded { written: 1, consumed: 8 });
    assert_eq!(out[0], b'x');
    let mut v = b"prefix".to_vec();
    assert_eq!(inflate_append(&stream, &mut v, 1).unwrap(), 8);
    assert_eq!(v, b"prefixx");
}

#[test]
fn broken_streams_are_refused_with_the_right_error() {
    let invalid = |r: Result<Vec<u8>>, what: &str| match r {
        Err(Error::Invalid { what: w, .. }) => assert_eq!(w, what),
        other => panic!("expected `{what}`, got {other:?}"),
    };
    invalid(inflate(&[0x07], 10), "reserved block type");
    invalid(inflate(&[0x01, 0x05, 0x00, 0xFA, 0xFE], 10), "stored block length does not match its complement");
    // A fixed block whose first symbol is a match (length 3, distance 1) with no output before it.
    invalid(inflate(&[0x03, 0x02, 0x00, 0x00], 10), "match distance reaches before the start of the output");
    // A dynamic block declaring 257 + 30 literal/length codes (HLIT = 30; 286 is the most).
    invalid(inflate(&[0b1111_0101, 0x00, 0x00, 0x00], 10), "too many length or distance symbols");
    // HLIT = 29 is legal; an all-zero code-length code is not.
    invalid(inflate(&[0b1110_1101, 0x00, 0x00, 0x00], 10), "incomplete code length set");
    invalid(zlib_decompress(&[0x79, 0x9C, 0x03, 0x00], 10), "zlib compression method is not DEFLATE");
    invalid(zlib_decompress(&[0x88, 0x98, 0x03, 0x00], 10), "zlib window size over 32 KiB");
    invalid(zlib_decompress(&[0x78, 0x9D, 0x03, 0x00], 10), "zlib header checksum");
    invalid(zlib_decompress(&[0x78, 0x9C, 0x03, 0x00, 0x00, 0x00, 0x00, 0x02], 10), "zlib Adler-32 checksum mismatch");
    assert!(matches!(zlib_decompress(ZLIB_DICT, 100), Err(Error::Unsupported { .. })));
    // Every strict prefix of a valid stream is an error, never a panic or a short success.
    let z = zlib_compress(&INPUT[..3000], Level::Default);
    for cut in 0..z.len() {
        assert!(zlib_decompress(&z[..cut], 1 << 16).is_err(), "prefix of {cut} bytes");
        assert!(zlib_decompress_into(&z[..cut], &mut [0_u8; 3000]).is_err(), "prefix of {cut} bytes");
    }
    assert!(matches!(inflate(&[], 10), Err(Error::Truncated { .. })));
    // A wrong Adler offset points at the trailer.
    let mut bad = z.clone();
    let at = bad.len() - 1;
    bad[at] ^= 1;
    assert_eq!(zlib_decompress(&bad, 1 << 16).unwrap_err().offset(), Some(z.len() - 4));
}

#[test]
fn fast_and_careful_decoders_agree() {
    // The fast loop runs only with margins of input and output room; decoding into an exact-size
    // slice, a slice with room to spare and a growing vector exercises both paths on the same data.
    for input in sample_inputs() {
        for level in [Level::Fast, Level::Best] {
            let raw = deflate(&input, level);
            for spare in [0, 1, 300] {
                let mut out = vec![0_u8; input.len() + spare];
                let d = inflate_into(&raw, &mut out).unwrap();
                assert_eq!(d.written, input.len());
                assert_eq!(out[..d.written], input[..]);
            }
        }
    }
}

/// A corpus of small valid streams of every block type.
fn fuzz_corpus() -> Vec<Vec<u8>> {
    let text = &INPUT[..2000];
    let mut corpus = vec![
        zlib_compress(text, Level::Store),
        zlib_compress(text, Level::Fast),
        zlib_compress(text, Level::Best),
        zlib_compress(&[0; 3000], Level::Default),
        ZLIB_DICT.to_vec(),
        ZEROS_4M.to_vec(),
    ];
    // A fixed-code block: zlib uses it for short inputs.
    corpus.push(zlib_compress(b"abcabcabcabd", Level::Default));
    corpus
}

#[test]
fn mutation_fuzz_never_panics_or_overruns() {
    // [ROB-01] [AC-WP03-2] mutated streams through every entry point; whatever decodes must decode
    // the same through the vector and the slice paths.
    humanart_core::fuzz::run("deflate", &fuzz_corpus(), 1500, 0x00DE_F1A7, Duration::from_secs(120), |bytes| {
        let vec_result = zlib_decompress(bytes, 1 << 18);
        let mut buf = vec![0_u8; 1 << 18];
        let slice_result = zlib_decompress_into(bytes, &mut buf);
        match (&vec_result, &slice_result) {
            (Ok(v), Ok(d)) => assert!(d.written == v.len() && buf[..d.written] == v[..], "sinks disagree"),
            (Err(Error::Limit { .. }), _) | (_, Err(Error::Limit { .. })) | (Err(_), Err(_)) => {}
            _ => panic!("one sink failed, the other did not: {vec_result:?} vs {slice_result:?}"),
        }
        let raw = bytes.get(2..).unwrap_or_default();
        if let Ok(v) = inflate(raw, 1 << 18) {
            let mut exact = vec![0_u8; v.len()];
            let d = inflate_into(raw, &mut exact);
            assert!(matches!(d, Ok(d) if d.written == v.len()) && exact == v, "exact-size decode disagrees");
        }
    });
}

#[test]
fn random_bit_streams_never_panic() {
    // Pure noise reaches the dynamic-header paths more often than mutations of valid streams do.
    let mut rng = Rng::new(0xB175);
    let mut buf = vec![0_u8; 1 << 16];
    for _ in 0..20_000 {
        let mut noise = vec![0_u8; 1 + rng.index(300)];
        rng.fill(&mut noise);
        // Half the time a dynamic-block header (BTYPE = 2) with a final bit.
        if rng.chance(0.5) {
            noise[0] = (noise[0] & !0b110) | 0b101;
        }
        let _ = inflate(&noise, 1 << 16);
        let _ = inflate_into(&noise, &mut buf);
    }
}

/// Measures throughput: `cargo test --release -p humanart-io --features deflate -- --ignored --nocapture
/// deflate_throughput`.
#[test]
#[ignore = "timing benchmark; run on demand in release mode"]
fn deflate_throughput() {
    let mut data = Vec::new();
    while data.len() < 32 << 20 {
        data.extend_from_slice(INPUT);
    }
    // Mutate each copy a little so the window does not see a perfect repeat.
    let mut rng = Rng::new(9);
    for _ in 0..data.len() / 64 {
        let i = rng.index(data.len());
        data[i] = rng.next_u32() as u8;
    }
    let mb = data.len() as f64 / 1e6;
    for level in LEVELS {
        let t = Instant::now();
        let z = zlib_compress(&data, level);
        let enc = t.elapsed().as_secs_f64();
        let mut out = vec![0_u8; data.len()];
        let t = Instant::now();
        let d = zlib_decompress_into(&z, &mut out).unwrap();
        let dec = t.elapsed().as_secs_f64();
        assert_eq!(d.written, data.len());
        println!(
            "{level:?}: {:.1}% of input, encode {:.0} MB/s, decode {:.0} MB/s output",
            100.0 * z.len() as f64 / data.len() as f64,
            mb / enc,
            mb / dec
        );
    }
}

/// Writes the encoder's output for every level to `target/agents/io-png/xcheck` so Python's zlib
/// can decode it (dev-time cross-check `[DEP-04]`).
#[test]
#[ignore = "writes files for the dev-time cross-check with Python's zlib"]
fn write_cross_check_streams() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/io-png/xcheck");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("input.bin"), INPUT).unwrap();
    for level in LEVELS {
        std::fs::write(dir.join(format!("ours_{level:?}.zz")), zlib_compress(INPUT, level)).unwrap();
    }
}
