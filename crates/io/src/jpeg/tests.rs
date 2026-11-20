//! Tests of the JPEG codec: fixtures from ffmpeg and from an independent Python writer with ffmpeg's
//! decoding as the reference, progressive files against their sequential twins, hand-made broken
//! files, and the mutation fuzz `[AC-WP03-1]` `[AC-WP03-2]`.

use super::decoder::{self, Frame};
use super::*;
use crate::error::Error;
use crate::image::{ColorSpace, Image, Limits, Samples};
use std::time::Duration;

macro_rules! fixtures {
    ($($name:literal),* $(,)?) => {
        [$(($name,
            include_bytes!(concat!("../../tests/fixtures/jpeg/", $name, ".jpg")).as_slice(),
            include_bytes!(concat!("../../tests/fixtures/jpeg/", $name, ".bin")).as_slice())),*]
    };
}

/// `(name, jpeg, reference planes)`: see `tests/fixtures/jpeg/make_fixtures.py`.
const FIXTURES: [(&str, &[u8], &[u8]); 22] = fixtures![
    "py_grey",
    "py_444",
    "py_420",
    "py_420_restart",
    "py_422_separate",
    "py_440",
    "py_411",
    "py_odd",
    "py_chroma_full",
    "py_prog_420",
    "py_prog_420_restart",
    "py_prog_grey",
    "py_prog_444",
    "py_rgb",
    "py_cmyk",
    "py_ycck",
    "py_exif6",
    "py_exif8",
    "py_big",
    "ff_420",
    "ff_422",
    "ff_444",
];

/// `(name, ffmpeg's RGB24 decoding)` of the 4:4:4 fixtures.
const RGB: [(&str, &[u8]); 2] = [
    ("ff_444", include_bytes!("../../tests/fixtures/jpeg/ff_444_rgb.bin").as_slice()),
    ("py_444", include_bytes!("../../tests/fixtures/jpeg/py_444_rgb.bin").as_slice()),
];

fn fixture(name: &str) -> &'static [u8] {
    FIXTURES.iter().find(|f| f.0 == name).map(|f| f.1).unwrap()
}

fn frame(bytes: &[u8]) -> (Info, Frame) {
    let (info, frame) = decoder::decode(bytes, &Limits::DEFAULT, false).unwrap();
    (info, frame.unwrap())
}

/// The component planes of a frame, cropped to their real sizes and concatenated (the layout of the
/// reference files).
fn planes(frame: &Frame) -> Vec<u8> {
    let mut out = Vec::new();
    for c in &frame.components {
        for row in c.plane.chunks(c.stride()).take(c.height) {
            out.extend_from_slice(&row[..c.width]);
        }
    }
    out
}

#[test]
fn fixtures_decode_to_ffmpegs_planes_within_one() {
    // [AC-WP03-1] [IO-02] Every layout, sequential and progressive, against ffmpeg's decoding. The
    // IDCTs differ (ours is AAN float, ffmpeg's an integer one), so a sample may differ by 1.
    let (mut total, mut differing) = (0, 0);
    for (name, jpeg, reference) in FIXTURES {
        let (_, f) = frame(jpeg);
        let ours = planes(&f);
        assert_eq!(ours.len(), reference.len(), "{name}: plane sizes");
        let mut worst = 0;
        for (a, b) in ours.iter().zip(reference) {
            worst = worst.max(a.abs_diff(*b));
            differing += usize::from(a != b);
        }
        total += ours.len();
        assert!(worst <= 1, "{name}: a sample differs by {worst}");
    }
    println!("jpeg fixtures: {differing} of {total} samples differ by 1 from ffmpeg");
    assert!(differing * 20 < total, "{differing} of {total} samples differ");
}

#[test]
fn headers_are_reported() {
    type Case = (&'static str, ColorModel, bool, [[u8; 2]; 4], u16);
    let cases: [Case; 9] = [
        ("py_grey", ColorModel::Grey, false, [[1, 1], [0, 0], [0, 0], [0, 0]], 0),
        ("py_420_restart", ColorModel::YCbCr, false, [[2, 2], [1, 1], [1, 1], [0, 0]], 3),
        ("py_prog_420_restart", ColorModel::YCbCr, true, [[2, 2], [1, 1], [1, 1], [0, 0]], 2),
        ("py_411", ColorModel::YCbCr, false, [[4, 1], [1, 1], [1, 1], [0, 0]], 0),
        ("py_rgb", ColorModel::Rgb, false, [[1, 1], [1, 1], [1, 1], [0, 0]], 0),
        ("py_cmyk", ColorModel::Cmyk, false, [[1, 1], [1, 1], [1, 1], [1, 1]], 0),
        ("py_ycck", ColorModel::Ycck, false, [[2, 2], [1, 1], [1, 1], [2, 2]], 0),
        // ffmpeg writes 4:2:2 as Y 2×2, chroma 1×2.
        ("ff_422", ColorModel::YCbCr, false, [[2, 2], [1, 2], [1, 2], [0, 0]], 0),
        ("py_exif6", ColorModel::YCbCr, false, [[2, 2], [1, 1], [1, 1], [0, 0]], 0),
    ];
    for (name, model, progressive, sampling, restart) in cases {
        let info = read_info(fixture(name), &Limits::DEFAULT).unwrap();
        assert_eq!(info.color_model, model, "{name}");
        assert_eq!(info.progressive, progressive, "{name}");
        assert_eq!(info.sampling, sampling, "{name}");
        assert_eq!(info.restart_interval, restart, "{name}");
        // read_info and a full decode agree.
        let (full, _) = decode_with(fixture(name), &DecodeOptions::default()).unwrap();
        assert_eq!(full, info, "{name}");
    }
    let info = read_info(fixture("py_ycck"), &Limits::DEFAULT).unwrap();
    assert_eq!((info.adobe_transform, info.jfif, info.components), (Some(2), false, 4));
    let info = read_info(fixture("py_exif6"), &Limits::DEFAULT).unwrap();
    assert_eq!((info.width, info.height, info.orientation(), info.oriented_size()), (20, 12, 6, (12, 20)));
    assert_eq!(info.exif, Exif { orientation: Some(6), color_space: None });
    assert!(info.jfif && !info.has_icc_profile);
}

#[test]
fn progressive_and_restart_files_equal_their_sequential_twins() {
    // [AC-WP03-1] The Python writer gives these files identical coefficients, so the planes must be
    // identical bit for bit, not just close.
    for (a, b) in [
        ("py_prog_420", "py_420"),
        ("py_prog_420_restart", "py_420"),
        ("py_420_restart", "py_420"),
        ("py_prog_grey", "py_grey"),
        ("py_prog_444", "py_444"),
    ] {
        assert!(planes(&frame(fixture(a)).1) == planes(&frame(fixture(b)).1), "{a} vs {b}");
        assert_eq!(decode(fixture(a), &Limits::DEFAULT).unwrap(), decode(fixture(b), &Limits::DEFAULT).unwrap());
    }
}

#[test]
fn rgb_output_matches_ffmpegs_conversion() {
    // [IO-02] Colour conversion against ffmpeg's (swscale, accurate rounding), on 4:4:4 files.
    for (name, rgb) in RGB {
        let image = decode(fixture(name), &Limits::DEFAULT).unwrap();
        let ours = image.as_u8().unwrap();
        assert_eq!(ours.len(), rgb.len(), "{name}");
        let diffs: Vec<u8> = ours.iter().zip(rgb).map(|(a, b)| a.abs_diff(*b)).collect();
        let mean = diffs.iter().map(|&d| f64::from(d)).sum::<f64>() / diffs.len() as f64;
        let worst = diffs.iter().copied().max().unwrap_or(0);
        println!("{name}: mean |diff| {mean:.3}, worst {worst}");
        assert!(mean < 0.2 && worst <= 3, "{name}: mean {mean}, worst {worst}");
    }
}

#[test]
fn upsampling_matches_a_float_model_of_centred_interpolation() {
    // [IO-02] Every subsampled layout: our RGB output against an independent double-precision model
    // built from ffmpeg's planes: bilinear interpolation at the JFIF-centred chroma positions
    // (x_c = (x + 0.5) * h / hmax - 0.5, clamped), then the JFIF equations. Planes differ by up to 1
    // and the fixed-point path rounds twice, hence the tolerance of 3.
    for name in
        ["py_420", "py_422_separate", "py_440", "py_411", "py_odd", "py_chroma_full", "py_big", "ff_420", "ff_422"]
    {
        let (_, jpeg, reference) = FIXTURES.iter().find(|f| f.0 == name).copied().unwrap();
        let (_, f) = frame(jpeg);
        let image = decode(jpeg, &Limits::DEFAULT).unwrap();
        let mut offsets = Vec::new();
        let mut at = 0;
        for c in &f.components {
            offsets.push(at);
            at += c.width * c.height;
        }
        let sample = |ci: usize, x: usize, y: usize| -> f64 {
            let c = &f.components[ci];
            let pos = |i: usize, factor: usize, max: usize, len: usize| {
                let p = ((i as f64 + 0.5) * factor as f64 / max as f64 - 0.5).clamp(0.0, (len - 1) as f64);
                let i0 = p.floor() as usize;
                (i0, (i0 + 1).min(len - 1), p - i0 as f64)
            };
            let (x0, x1, fx) = pos(x, c.h, f.hmax, c.width);
            let (y0, y1, fy) = pos(y, c.v, f.vmax, c.height);
            let s = |xx: usize, yy: usize| f64::from(reference[offsets[ci] + yy * c.width + xx]);
            (s(x0, y0) * (1.0 - fx) + s(x1, y0) * fx) * (1.0 - fy) + (s(x0, y1) * (1.0 - fx) + s(x1, y1) * fx) * fy
        };
        let mut worst = 0.0_f64;
        for (i, px) in image.as_u8().unwrap().chunks(3).enumerate() {
            let (x, y) = (i % f.width, i / f.width);
            let (yy, cb, cr) = (sample(0, x, y), sample(1, x, y) - 128.0, sample(2, x, y) - 128.0);
            let expected = [yy + 1.402 * cr, yy - 0.344_136 * cb - 0.714_136 * cr, yy + 1.772 * cb];
            for (&got, e) in px.iter().zip(expected) {
                worst = worst.max((f64::from(got) - e.clamp(0.0, 255.0)).abs());
            }
        }
        assert!(worst <= 3.0, "{name}: worst error {worst}");
    }
}

#[test]
fn adobe_models_convert_as_adobe_writes_them() {
    // Full-resolution RGB and CMYK files: the decoded pixels follow from the reference planes by
    // the documented equations (R = C * K / 255 for inverted CMYK).
    for name in ["py_rgb", "py_cmyk"] {
        let (_, jpeg, reference) = FIXTURES.iter().find(|f| f.0 == name).copied().unwrap();
        let image = decode(jpeg, &Limits::DEFAULT).unwrap();
        let n = image.pixel_count();
        let plane = |c: usize, i: usize| f64::from(reference[c * n + i]);
        for (i, px) in image.as_u8().unwrap().chunks(3).enumerate() {
            for (c, &v) in px.iter().enumerate() {
                let expected = if name == "py_cmyk" { plane(c, i) * plane(3, i) / 255.0 } else { plane(c, i) };
                assert!((f64::from(v) - expected).abs() <= 2.0, "{name} pixel {i}: {v} vs {expected}");
            }
        }
    }
    // YCCK: K multiplies the YCbCr result; at a pixel where every plane is known exactly (Y and K
    // at full resolution, chroma flat by the edge clamp at the corner) the formula holds.
    let (_, f) = frame(fixture("py_ycck"));
    let image = decode(fixture("py_ycck"), &Limits::DEFAULT).unwrap();
    let p = |c: usize| f64::from(f.components[c].plane[0]);
    let (y, cb, cr, k) = (p(0), p(1) - 128.0, p(2) - 128.0, p(3));
    let expected =
        [y + 1.402 * cr, y - 0.344_136 * cb - 0.714_136 * cr, y + 1.772 * cb].map(|v| v.clamp(0.0, 255.0) * k / 255.0);
    for (got, e) in image.as_u8().unwrap()[..3].iter().zip(expected) {
        assert!((f64::from(*got) - e).abs() <= 1.0, "{got} vs {e}");
    }
}

#[test]
fn exif_orientation_is_applied_by_default() {
    for (name, orientation) in [("py_exif6", 6_u8), ("py_exif8", 8)] {
        let raw =
            decode_with(fixture(name), &DecodeOptions { limits: Limits::DEFAULT, apply_orientation: false }).unwrap().1;
        let upright = decode(fixture(name), &Limits::DEFAULT).unwrap();
        assert_eq!((raw.width, raw.height, upright.width, upright.height), (20, 12, 12, 20));
        let (w, h) = (raw.width as usize, raw.height as usize);
        let (r, u) = (raw.as_u8().unwrap(), upright.as_u8().unwrap());
        for oy in 0..w {
            for ox in 0..h {
                let (sx, sy) = if orientation == 6 { (oy, h - 1 - ox) } else { (w - 1 - oy, ox) };
                assert_eq!(u[(oy * h + ox) * 3..][..3], r[(sy * w + sx) * 3..][..3], "{name} ({ox}, {oy})");
            }
        }
    }
}

#[test]
fn images_are_srgb_bytes_with_one_or_three_channels() {
    let grey = decode(fixture("py_grey"), &Limits::DEFAULT).unwrap();
    assert_eq!((grey.width, grey.height, grey.channels, grey.color_space), (37, 29, 1, ColorSpace::Srgb));
    let colour = decode(fixture("py_cmyk"), &Limits::DEFAULT).unwrap();
    assert_eq!((colour.channels, colour.samples.sample_size()), (3, 1));
}

// --- Broken files ---------------------------------------------------------------------------------

/// The offset of the first marker `code` in a file.
fn find_marker(bytes: &[u8], code: u8) -> usize {
    bytes.windows(2).position(|w| w == [0xFF, code]).unwrap()
}

fn patched(name: &str, at: usize, values: &[u8]) -> Vec<u8> {
    let mut v = fixture(name).to_vec();
    v[at..at + values.len()].copy_from_slice(values);
    v
}

#[test]
fn broken_files_are_refused_with_the_right_error() {
    let limits = Limits::DEFAULT;
    let good = fixture("py_420");
    let sof = find_marker(good, 0xC0);
    // Not a JPEG.
    assert!(matches!(decode(b"\x89PNG", &limits), Err(Error::Invalid { offset: 0, .. })));
    assert!(matches!(decode(&[], &limits), Err(Error::Invalid { .. })));
    // Truncated anywhere: an error, never a panic or a partial image.
    for n in 0..good.len() - 2 {
        assert!(decode(&good[..n], &limits).is_err(), "prefix of {n} bytes decoded");
    }
    assert!(matches!(decode(&good[..good.len() - 2], &limits), Err(Error::Truncated { .. })));
    // The entropy-coded data cut short, then EOI: the decoder notices it ran out of bits.
    let sos = find_marker(good, 0xDA);
    let mut cut = good[..sos + 40].to_vec();
    cut.extend_from_slice(&[0xFF, 0xD9]);
    assert!(matches!(decode(&cut, &limits), Err(Error::Truncated { .. })));
    // 12-bit, arithmetic coding, lossless, two components, DNL height.
    assert!(matches!(
        decode(&patched("py_420", sof + 4, &[12]), &limits),
        Err(Error::Unsupported { what: "12-bit samples", .. })
    ));
    assert!(matches!(
        decode(&patched("py_420", sof + 1, &[0xC9]), &limits),
        Err(Error::Unsupported { what: "arithmetic coding", .. })
    ));
    assert!(matches!(decode(&patched("py_420", sof + 1, &[0xC3]), &limits), Err(Error::Unsupported { .. })));
    assert!(matches!(decode(&patched("py_420", sof + 5, &[0, 0]), &limits), Err(Error::Unsupported { .. })));
    assert!(matches!(decode(&patched("py_420", sof + 4, &[7]), &limits), Err(Error::Invalid { .. })));
    // Sampling factor 0 and 5, quantisation selector 4.
    assert!(matches!(decode(&patched("py_420", sof + 11, &[0x02]), &limits), Err(Error::Invalid { .. })));
    assert!(matches!(decode(&patched("py_420", sof + 11, &[0x51]), &limits), Err(Error::Invalid { .. })));
    assert!(matches!(decode(&patched("py_420", sof + 12, &[4]), &limits), Err(Error::Invalid { .. })));
    // Dimensions over the limits are refused from the header.
    let small = Limits { max_dimension: 40, ..Limits::DEFAULT };
    assert!(matches!(decode(good, &small), Err(Error::Limit { what: "image dimension", .. })));
    assert!(matches!(read_info(good, &small), Err(Error::Limit { .. })));
    let tight = Limits { max_bytes: 1000, ..Limits::DEFAULT };
    assert!(matches!(decode(good, &tight), Err(Error::Limit { .. })));
    // A zero in a quantisation table.
    let dqt = find_marker(good, 0xDB);
    assert!(matches!(decode(&patched("py_420", dqt + 5, &[0]), &limits), Err(Error::Invalid { .. })));
    // A Huffman table with an impossible code count.
    let dht = find_marker(good, 0xC4);
    assert!(matches!(decode(&patched("py_420", dht + 5, &[3]), &limits), Err(Error::Invalid { .. })));
    // A scan naming a component that does not exist, and a scan before the frame.
    assert!(matches!(decode(&patched("py_420", sos + 5, &[9]), &limits), Err(Error::Invalid { .. })));
    let mut early = good[..2].to_vec();
    early.extend_from_slice(&good[sos..]);
    assert!(matches!(decode(&early, &limits), Err(Error::Invalid { .. })));
    // A segment length running past the end, and one under 2.
    assert!(matches!(decode(&patched("py_420", dqt + 2, &[0xFF, 0xFF]), &limits), Err(Error::Truncated { .. })));
    assert!(matches!(decode(&patched("py_420", dqt + 2, &[0, 1]), &limits), Err(Error::Invalid { .. })));
    // Garbage where a marker belongs; an unknown marker; a second SOI.
    assert!(matches!(decode(&patched("py_420", dqt, &[0x12]), &limits), Err(Error::Invalid { .. })));
    assert!(matches!(decode(&patched("py_420", dqt + 1, &[0x02]), &limits), Err(Error::Invalid { .. })));
    assert!(matches!(decode(&patched("py_420", dqt + 1, &[0xD8]), &limits), Err(Error::Invalid { .. })));
    // No scan at all.
    let mut empty = good[..sos].to_vec();
    empty.extend_from_slice(&[0xFF, 0xD9]);
    assert!(matches!(decode(&empty, &limits), Err(Error::Invalid { .. })));
}

#[test]
fn restart_markers_must_come_in_order() {
    let good = fixture("py_420_restart");
    let sos = find_marker(good, 0xDA);
    let rst = sos + find_marker(&good[sos..], 0xD0);
    assert!(decode(good, &Limits::DEFAULT).is_ok());
    let wrong = patched("py_420_restart", rst + 1, &[0xD3]);
    assert!(matches!(
        decode(&wrong, &Limits::DEFAULT),
        Err(Error::Invalid { what: "missing or out-of-order restart marker", .. })
    ));
}

#[test]
fn a_progressive_file_needs_every_component_coded() {
    // Keeping only the first and the last scan (the interleaved DC scan, the luma AC refinement)
    // still codes every component, so it decodes; keeping only a luma AC scan leaves Cb and Cr
    // uncoded, which is refused.
    let good = fixture("py_prog_420");
    let sos: Vec<usize> = good.windows(2).enumerate().filter(|(_, w)| *w == [0xFF, 0xDA]).map(|(i, _)| i).collect();
    assert_eq!(sos.len(), 10);
    // Each scan's DHT segment comes right before its SOS (the DC refinement scan has none).
    let dht_before = |scan: usize| good[..sos[scan]].windows(2).rposition(|w| w == [0xFF, 0xC4]).unwrap();
    let mut fewer = good[..dht_before(1)].to_vec();
    fewer.extend_from_slice(&good[dht_before(9)..]);
    let image = decode(&fewer, &Limits::DEFAULT).unwrap();
    assert_eq!((image.width, image.height), (45, 31));
    let mut luma_only = good[..dht_before(0)].to_vec();
    luma_only.extend_from_slice(&good[dht_before(1)..dht_before(2)]);
    luma_only.extend_from_slice(&[0xFF, 0xD9]);
    assert!(matches!(
        decode(&luma_only, &Limits::DEFAULT),
        Err(Error::Invalid { what: "a component is never coded in any scan", .. })
    ));
}

// --- Fuzz -----------------------------------------------------------------------------------------

fn fuzz_corpus() -> Vec<Vec<u8>> {
    ["py_grey", "py_420_restart", "py_prog_420_restart", "py_prog_grey", "py_ycck", "py_exif6", "py_odd", "ff_422"]
        .iter()
        .map(|name| fixture(name).to_vec())
        .collect()
}

#[test]
fn mutation_fuzz_never_panics() {
    // [ROB-01] [AC-WP03-2]
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let decoded = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&decoded);
    humanart_core::fuzz::run("jpeg", &fuzz_corpus(), 1000, 0x0000_1F1F, Duration::from_secs(120), move |bytes| {
        let info = read_info(bytes, &Limits::FUZZ);
        if let Ok((full, image)) = decode_with(bytes, &DecodeOptions { limits: Limits::FUZZ, apply_orientation: true })
        {
            counter.fetch_add(1, Ordering::Relaxed);
            assert_eq!(info.as_ref().ok(), Some(&full), "read_info and decode agree");
            assert_eq!(image.samples.len(), image.pixel_count() * usize::from(image.channels));
        }
    });
    // The mutations do reach the end of the decoder, not just its first checks.
    let decoded = decoded.load(Ordering::Relaxed);
    println!("jpeg fuzz: {decoded} of {} cases decoded", 8 * 1001);
    assert!(decoded > 300, "only {decoded} fuzz cases decoded");
}

/// An image for the encoder and benchmark tests: smooth shading, edges and texture.
pub(super) fn texture(width: u32, height: u32, channels: u8, seed: u64) -> Image {
    let mut rng = humanart_core::rng::Rng::new(seed);
    let mut samples = Vec::with_capacity((width * height) as usize * usize::from(channels));
    for y in 0..height {
        for x in 0..width {
            let (fx, fy) = (f64::from(x) / f64::from(width), f64::from(y) / f64::from(height));
            let disc = (fx - 0.6).powi(2) + (fy - 0.4).powi(2) < 0.04;
            for c in 0..channels {
                let base = 128.0 + 90.0 * ((fx * 9.0 + f64::from(c)).sin() * (fy * 7.0 - f64::from(c)).cos());
                let v = if disc { 220.0 - 50.0 * f64::from(c) } else { base } + rng.index(9) as f64 - 4.0;
                samples.push(v.clamp(0.0, 255.0) as u8);
            }
        }
    }
    Image::new(crate::error::Format::Jpeg, width, height, channels, ColorSpace::Srgb, Samples::U8(samples)).unwrap()
}

// --- Encoder --------------------------------------------------------------------------------------

/// Peak signal-to-noise ratio of `b` against `a`, in dB.
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse = a.iter().zip(b).map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (255.0 * 255.0 / mse.max(1e-12)).log10()
}

fn round_trip(image: &Image, options: EncodeOptions) -> (usize, f64) {
    let bytes = encode_with(image, &options).unwrap();
    let back = decode(&bytes, &Limits::DEFAULT).unwrap();
    assert_eq!((back.width, back.height), (image.width, image.height));
    let original: Vec<u8> = if image.channels == 4 || image.channels == 2 {
        image.as_u8().unwrap().chunks(usize::from(image.channels)).flat_map(|p| p[..p.len() - 1].to_vec()).collect()
    } else {
        image.as_u8().unwrap().to_vec()
    };
    (bytes.len(), psnr(&original, back.as_u8().unwrap()))
}

#[test]
fn encoder_round_trips_with_the_expected_quality() {
    // [IO-02] [AC-WP03-1] PSNR rises with the quality and stays above a floor at each setting.
    let colour = texture(256, 192, 3, 5);
    let grey = texture(256, 192, 1, 6);
    let floors: [(u8, f64, f64, f64); 4] =
        [(50, 34.0, 33.0, 35.0), (75, 36.0, 34.0, 37.0), (90, 38.0, 36.0, 40.0), (100, 48.0, 37.0, 55.0)];
    let mut previous = (0.0, 0.0, 0.0);
    for (quality, floor_444, floor_420, floor_grey) in floors {
        let o444 = EncodeOptions { quality, subsampling: Subsampling::S444, restart_interval: 0 };
        let o420 = EncodeOptions { quality, subsampling: Subsampling::S420, restart_interval: 0 };
        let (size_444, p444) = round_trip(&colour, o444);
        let (size_420, p420) = round_trip(&colour, o420);
        let (size_grey, pgrey) = round_trip(&grey, o444);
        println!(
            "q{quality}: 4:4:4 {p444:.2} dB {size_444} B, 4:2:0 {p420:.2} dB {size_420} B, grey {pgrey:.2} dB {size_grey} B"
        );
        assert!(p444 > floor_444 && p420 > floor_420 && pgrey > floor_grey, "q{quality}");
        assert!(p444 > previous.0 && p420 > previous.1 && pgrey > previous.2, "q{quality} not better");
        assert!(size_420 < size_444);
        previous = (p444, p420, pgrey);
    }
}

#[test]
fn encoder_handles_odd_sizes_alpha_and_restarts() {
    for (w, h) in [(1, 1), (17, 9), (8, 16), (33, 2)] {
        for channels in [1, 2, 3, 4] {
            // At these sizes the pattern's colour changes within a few pixels, which 4:2:0 cannot
            // follow; 4:4:4 shows that the edge padding and the block layout are right.
            let image = texture(w, h, channels, u64::from(w * h));
            let o444 = EncodeOptions { subsampling: Subsampling::S444, ..EncodeOptions::default() };
            let (_, p444) = round_trip(&image, o444);
            let (_, p420) = round_trip(&image, EncodeOptions::default());
            assert!(p444 > 30.0 && p420 > 15.0, "{w}x{h}x{channels}: {p444} / {p420} dB");
        }
    }
    // Restart intervals change the stream, not the pixels.
    let image = texture(100, 60, 3, 9);
    let plain = encode_with(&image, &EncodeOptions::default()).unwrap();
    let restarts = encode_with(&image, &EncodeOptions { restart_interval: 5, ..EncodeOptions::default() }).unwrap();
    assert_ne!(plain, restarts);
    assert_eq!(read_info(&restarts, &Limits::DEFAULT).unwrap().restart_interval, 5);
    assert_eq!(decode(&plain, &Limits::DEFAULT).unwrap(), decode(&restarts, &Limits::DEFAULT).unwrap());
    // The header says what was asked for.
    let info = read_info(&encode(&image, 80).unwrap(), &Limits::DEFAULT).unwrap();
    assert_eq!(
        (info.color_model, info.sampling[0], info.jfif, info.progressive),
        (ColorModel::YCbCr, [2, 2], true, false)
    );
}

#[test]
fn the_encoder_refuses_what_it_cannot_write() {
    let mut image = texture(4, 4, 3, 1);
    image.color_space = ColorSpace::Linear;
    assert!(matches!(encode(&image, 90), Err(Error::Encode { what: "JPEG holds sRGB colour only", .. })));
    let floats = Image::new(crate::error::Format::Jpeg, 2, 2, 1, ColorSpace::Srgb, Samples::F32(vec![0.0; 4])).unwrap();
    assert!(matches!(encode(&floats, 90), Err(Error::Encode { .. })));
    let wide = Image {
        width: 70_000,
        height: 1,
        channels: 1,
        color_space: ColorSpace::Srgb,
        samples: Samples::U8(vec![0; 70_000]),
    };
    assert!(matches!(encode(&wide, 90), Err(Error::Encode { .. })));
    let mut bad = texture(4, 4, 3, 1);
    bad.channels = 1;
    assert!(matches!(encode(&bad, 90), Err(Error::Encode { .. })));
}

/// Measures decode speed: `cargo test --release -p humanart-io --features jpeg -- --ignored --nocapture
/// jpeg_throughput`.
#[test]
#[ignore = "timing benchmark; run on demand in release mode"]
fn jpeg_throughput() {
    use std::time::Instant;
    let image = texture(4096, 2048, 3, 3);
    let mp = 4096.0 * 2048.0 / 1e6;
    for (label, subsampling) in [("4:2:0", Subsampling::S420), ("4:4:4", Subsampling::S444)] {
        let t = Instant::now();
        let bytes = encode_with(&image, &EncodeOptions { quality: 90, subsampling, restart_interval: 0 }).unwrap();
        let enc = t.elapsed().as_secs_f64();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            let back = decode(&bytes, &Limits::DEFAULT).unwrap();
            best = best.min(t.elapsed().as_secs_f64());
            assert_eq!(back.width, 4096);
        }
        let mut planes_only = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            decoder::decode(&bytes, &Limits::DEFAULT, false).unwrap();
            planes_only = planes_only.min(t.elapsed().as_secs_f64());
        }
        let (info, frame) = decoder::decode(&bytes, &Limits::DEFAULT, false).unwrap();
        let frame = frame.unwrap();
        let mut colour = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            super::output::to_image(&frame, info.color_model, &Limits::DEFAULT).unwrap();
            colour = colour.min(t.elapsed().as_secs_f64());
        }
        println!(
            "4096x2048 {label} q90: {} KB, encode {:.0} MP/s, decode {:.0} MP/s ({:.0} MP/s to planes; upsampling and colour {:.1} ms)",
            bytes.len() / 1024,
            mp / enc,
            mp / best,
            mp / planes_only,
            colour * 1e3
        );
    }
    // Heavier texture: noise of ±24 makes about 3 bits per pixel, like a detailed photo.
    let mut noisy = image.clone();
    let mut rng = humanart_core::rng::Rng::new(77);
    if let Samples::U8(v) = &mut noisy.samples {
        for s in v.iter_mut() {
            *s = (i32::from(*s) + rng.index(49) as i32 - 24).clamp(0, 255) as u8;
        }
    }
    let bytes = encode_with(&noisy, &EncodeOptions { quality: 90, ..EncodeOptions::default() }).unwrap();
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        decode(&bytes, &Limits::DEFAULT).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    let mut planes_only = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        decoder::decode(&bytes, &Limits::DEFAULT, false).unwrap();
        planes_only = planes_only.min(t.elapsed().as_secs_f64());
    }
    let (info, frame) = decoder::decode(&bytes, &Limits::DEFAULT, false).unwrap();
    let frame = frame.unwrap();
    let mut colour = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        super::output::to_image(&frame, info.color_model, &Limits::DEFAULT).unwrap();
        colour = colour.min(t.elapsed().as_secs_f64());
    }
    println!(
        "4096x2048 4:2:0 q90 noisy: {:.2} bits/pixel, decode {:.0} MP/s ({:.0} MP/s to planes, before upsampling and colour; upsampling and colour alone {:.1} ms)",
        bytes.len() as f64 * 8.0 / (mp * 1e6),
        mp / best,
        mp / planes_only,
        colour * 1e3
    );
    // Progressive against sequential, same coefficients: files from `make_fixtures.py --bench DIR`.
    if let Ok(dir) = std::env::var("HUMANART_JPEG_BENCH") {
        for name in ["bench_seq", "bench_prog"] {
            let bytes = std::fs::read(std::path::Path::new(&dir).join(format!("{name}.jpg"))).unwrap();
            let mut best = f64::MAX;
            for _ in 0..10 {
                let t = Instant::now();
                decode(&bytes, &Limits::DEFAULT).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!("{name} 1024x768 4:2:0: decode {:.0} MP/s", 1024.0 * 768.0 / 1e6 / best);
        }
    }
}

/// Writes encoder outputs to `target/agents/io-jpeg/xcheck` with our decoding (`.rgb` raw RGB24,
/// `.planes` the Y, Cb, Cr planes), for the dev-time check that ffmpeg decodes them to the same
/// samples.
#[test]
#[ignore = "writes files for the dev-time cross-check with ffmpeg"]
fn write_cross_check_files() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/io-jpeg/xcheck");
    std::fs::create_dir_all(&dir).unwrap();
    let image = texture(203, 117, 3, 21);
    for (name, subsampling, restart_interval) in
        [("enc_444", Subsampling::S444, 0), ("enc_420", Subsampling::S420, 0), ("enc_420_rst", Subsampling::S420, 7)]
    {
        let bytes = encode_with(&image, &EncodeOptions { quality: 85, subsampling, restart_interval }).unwrap();
        std::fs::write(dir.join(format!("{name}.jpg")), &bytes).unwrap();
        std::fs::write(dir.join(format!("{name}.planes")), planes(&frame(&bytes).1)).unwrap();
        std::fs::write(dir.join(format!("{name}.rgb")), decode(&bytes, &Limits::DEFAULT).unwrap().as_u8().unwrap())
            .unwrap();
    }
}
