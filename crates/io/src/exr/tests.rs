use super::*;
use humanart_core::rng::Rng;
use std::time::Duration;

/// A fixture written by ffmpeg's `exr` encoder and its decode by ffmpeg in the decoder's native
/// format (planar G, B, R [, A] or Y; f16 LE for half files, f32 LE for float ones); provenance in
/// the agent STATUS file.
struct Fixture {
    exr: &'static [u8],
    decoded: &'static [u8],
    compression: Compression,
    pixel_type: PixelType,
    names: &'static [&'static str],
}

const FIXTURES: [Fixture; 6] = [
    Fixture {
        exr: include_bytes!("../../tests/fixtures/exr/ffmpeg_rgb_half_zip.exr"),
        decoded: include_bytes!("../../tests/fixtures/exr/ffmpeg_rgb_half_zip.gbrpf16le.bin"),
        compression: Compression::Zip,
        pixel_type: PixelType::Half,
        names: &["B", "G", "R"],
    },
    Fixture {
        exr: include_bytes!("../../tests/fixtures/exr/ffmpeg_rgb_half_zips.exr"),
        decoded: include_bytes!("../../tests/fixtures/exr/ffmpeg_rgb_half_zips.gbrpf16le.bin"),
        compression: Compression::Zips,
        pixel_type: PixelType::Half,
        names: &["B", "G", "R"],
    },
    Fixture {
        exr: include_bytes!("../../tests/fixtures/exr/ffmpeg_rgba_float_zips.exr"),
        decoded: include_bytes!("../../tests/fixtures/exr/ffmpeg_rgba_float_zips.gbrapf32le.bin"),
        compression: Compression::Zips,
        pixel_type: PixelType::Float,
        names: &["A", "B", "G", "R"],
    },
    Fixture {
        exr: include_bytes!("../../tests/fixtures/exr/ffmpeg_rgba_float_rle.exr"),
        decoded: include_bytes!("../../tests/fixtures/exr/ffmpeg_rgba_float_rle.gbrapf32le.bin"),
        compression: Compression::Rle,
        pixel_type: PixelType::Float,
        names: &["A", "B", "G", "R"],
    },
    Fixture {
        exr: include_bytes!("../../tests/fixtures/exr/ffmpeg_rgb_float_none.exr"),
        decoded: include_bytes!("../../tests/fixtures/exr/ffmpeg_rgb_float_none.gbrpf32le.bin"),
        compression: Compression::None,
        pixel_type: PixelType::Float,
        names: &["B", "G", "R"],
    },
    Fixture {
        exr: include_bytes!("../../tests/fixtures/exr/ffmpeg_y_half_rle.exr"),
        decoded: include_bytes!("../../tests/fixtures/exr/ffmpeg_y_half_rle.grayf16le.bin"),
        compression: Compression::Rle,
        pixel_type: PixelType::Half,
        names: &["Y"],
    },
];

const ALL: [Compression; 5] =
    [Compression::None, Compression::Rle, Compression::Zips, Compression::Zip, Compression::Piz];

fn options(compression: Compression) -> EncodeOptions {
    EncodeOptions { compression, ..EncodeOptions::default() }
}

/// A channel image with every sample type: smooth areas, noise, and special values.
fn mixed(width: u32, height: u32, seed: u64) -> ChannelImage {
    let mut rng = Rng::new(seed);
    let n = width as usize * height as usize;
    let specials = [0x7C00_u16, 0xFC00, 0x7E00, 0x0001, 0x8000, 0x3C00, 0x7BFF];
    let half: Vec<u16> = (0..n)
        .map(|i| match i % 7 {
            0 => specials[rng.index(specials.len())],
            1 | 2 => rng.next_u32() as u16,
            _ => f32_to_f16_bits((i % 40) as f32 * 0.125),
        })
        .collect();
    let float: Vec<f32> = (0..n)
        .map(|i| if i % 5 == 0 { f32::from_bits(rng.next_u32()) } else { (i as f32 * 0.01).sin() * 1000.0 })
        .collect();
    let ids: Vec<u32> = (0..n).map(|i| if i % 3 == 0 { rng.next_u32() } else { (i / 50) as u32 }).collect();
    let small: Vec<u16> = (0..n).map(|i| f32_to_f16_bits(((i % 9) as f32) / 8.0)).collect();
    ChannelImage {
        width,
        height,
        channels: vec![
            Channel { name: "diffuse.R".into(), linear: false, data: ChannelData::Half(half) },
            Channel { name: "Z".into(), linear: true, data: ChannelData::Float(float) },
            Channel { name: "objectId".into(), linear: false, data: ChannelData::Uint(ids) },
            Channel { name: "A".into(), linear: false, data: ChannelData::Half(small) },
        ],
    }
}

/// Bitwise equality of two channel images, channel order ignored (NaN payloads included).
fn same_channels(a: &ChannelImage, b: &ChannelImage) -> bool {
    let bits = |d: &ChannelData| -> Vec<u32> {
        match d {
            ChannelData::Uint(v) => v.clone(),
            ChannelData::Half(v) => v.iter().map(|&h| u32::from(h)).collect(),
            ChannelData::Float(v) => v.iter().map(|f| f.to_bits()).collect(),
        }
    };
    a.width == b.width
        && a.height == b.height
        && a.channels.len() == b.channels.len()
        && a.channels.iter().all(|c| {
            b.channel(&c.name).is_some_and(|d| {
                d.linear == c.linear && d.data.pixel_type() == c.data.pixel_type() && bits(&d.data) == bits(&c.data)
            })
        })
}

#[test]
fn ac_wp03_1_decodes_the_ffmpeg_fixtures_like_ffmpeg() {
    for f in &FIXTURES {
        let (info, channels) = decode_channels(f.exr, &Limits::DEFAULT).unwrap();
        assert_eq!((info.width(), info.height()), (37, 21));
        assert_eq!(info.compression, f.compression);
        assert_eq!(info.line_order, LineOrder::IncreasingY);
        let names: Vec<&str> = info.channels.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, f.names);
        assert!(info.channels.iter().all(|c| c.pixel_type == f.pixel_type));
        // ffmpeg's planes: G, B, R, A or Y.
        let planes: &[&str] = if f.names.len() == 1 {
            &["Y"]
        } else if f.names.len() == 3 {
            &["G", "B", "R"]
        } else {
            &["G", "B", "R", "A"]
        };
        let plane = 37 * 21;
        let size = f.pixel_type.size();
        assert_eq!(f.decoded.len(), plane * planes.len() * size);
        for (k, name) in planes.iter().enumerate() {
            let data = &channels.channel(name).unwrap().data;
            for p in 0..plane {
                let at = (k * plane + p) * size;
                let (ours, theirs) = match data {
                    ChannelData::Half(v) => {
                        (u32::from(v[p]), u32::from(u16::from_le_bytes([f.decoded[at], f.decoded[at + 1]])))
                    }
                    ChannelData::Float(v) => (
                        v[p].to_bits(),
                        u32::from_le_bytes([f.decoded[at], f.decoded[at + 1], f.decoded[at + 2], f.decoded[at + 3]]),
                    ),
                    ChannelData::Uint(_) => panic!("ffmpeg writes no uint channels"),
                };
                assert_eq!(ours, theirs, "{:?} {name} pixel {p}", f.compression);
            }
        }
        // The image view: RGB(A) or Y, F16 for half files, linear.
        let image = decode(f.exr, &Limits::DEFAULT).unwrap();
        assert_eq!(image.channels as usize, f.names.len());
        assert_eq!(image.color_space, ColorSpace::Linear);
        assert_eq!(matches!(image.samples, Samples::F16(_)), f.pixel_type == PixelType::Half);
        // ffmpeg writes no pixelAspectRatio: the default applies; its extra attributes are kept.
        assert_eq!(info.pixel_aspect_ratio, 1.0);
        assert!(info.attributes.iter().any(|a| a.name == "writer" && a.type_name == "string" && a.value == b"lavc"));
        assert_eq!(read_info(f.exr, &Limits::DEFAULT).unwrap(), info);
    }
}

#[test]
fn ac_wp03_1_fixtures_re_encode_losslessly_in_every_compression() {
    for f in &FIXTURES {
        let (_, original) = decode_channels(f.exr, &Limits::DEFAULT).unwrap();
        for compression in ALL {
            let bytes = encode_channels(&original, &options(compression)).unwrap();
            let (info, back) = decode_channels(&bytes, &Limits::DEFAULT).unwrap();
            assert_eq!(info.compression, compression);
            assert!(same_channels(&original, &back), "{:?} -> {compression:?}", f.compression);
        }
    }
}

#[test]
fn ac_wp03_1_mixed_channels_round_trip_in_every_compression() {
    for (width, height, seed) in [(1, 1, 1), (2, 3, 2), (37, 33, 3), (5, 70, 4), (64, 17, 5), (300, 2, 6)] {
        let image = mixed(width, height, seed);
        for compression in ALL {
            let bytes = encode_channels(&image, &options(compression)).unwrap();
            assert_eq!(bytes, encode_channels(&image, &options(compression)).unwrap(), "deterministic");
            let (info, back) = decode_channels(&bytes, &Limits::DEFAULT).unwrap();
            let names: Vec<&str> = info.channels.iter().map(|c| c.name.as_str()).collect();
            assert_eq!(names, ["A", "Z", "diffuse.R", "objectId"], "sorted by name");
            assert!(same_channels(&image, &back), "{width}x{height} {compression:?}");
        }
    }
}

#[test]
fn compression_pays_on_smooth_data_and_piz_takes_both_wavelet_paths() {
    let n = 128 * 64;
    let smooth: Vec<u16> = (0..n).map(|i| f32_to_f16_bits(((i % 128) as f32 / 128.0).powi(2))).collect();
    let image = ChannelImage {
        width: 128,
        height: 64,
        channels: vec![Channel { name: "Y".into(), linear: false, data: ChannelData::Half(smooth) }],
    };
    let raw = encode_channels(&image, &options(Compression::None)).unwrap().len();
    for compression in [Compression::Rle, Compression::Zips, Compression::Zip, Compression::Piz] {
        let size = encode_channels(&image, &options(compression)).unwrap().len();
        // RLE only catches runs of equal predicted bytes and ZIPS pays zlib's overhead on every
        // line; ZIP and PIZ at least halve the file.
        let factor = if matches!(compression, Compression::Rle | Compression::Zips) { 1 } else { 2 };
        assert!(size * factor < raw, "{compression:?}: {size} vs {raw}");
    }
    // Few distinct words: the 14-bit wavelet. Every word of a range of 2^15: the 16-bit one.
    let mut rng = Rng::new(8);
    for spread in [16_u32, 1 << 16] {
        let data: Vec<u16> = (0..n).map(|_| (rng.next_u32() % spread) as u16).collect();
        let image = ChannelImage {
            width: 128,
            height: 64,
            channels: vec![Channel { name: "Y".into(), linear: false, data: ChannelData::Half(data) }],
        };
        let bytes = encode_channels(&image, &options(Compression::Piz)).unwrap();
        assert!(same_channels(&image, &decode_channels(&bytes, &Limits::DEFAULT).unwrap().1));
    }
}

#[test]
fn ac_wp03_1_images_round_trip_with_type_conversion() {
    let mut rng = Rng::new(12);
    for channels in 1..=4_u8 {
        let n = 19 * 7 * channels as usize;
        let floats: Vec<f32> = (0..n).map(|_| rng.range_f32(-4.0, 60_000.0)).collect();
        let image = Image::new(FORMAT, 19, 7, channels, ColorSpace::Linear, Samples::F32(floats.clone())).unwrap();
        for compression in ALL {
            let opts = options(compression);
            let back = decode(&encode_with(&image, &opts).unwrap(), &Limits::DEFAULT).unwrap();
            assert_eq!(back, image, "{channels} {compression:?}");
            // As halves: every sample rounded to nearest even, and F16 samples come back.
            let half = EncodeOptions { pixel_type: Some(PixelType::Half), ..opts };
            let back = decode(&encode_with(&image, &half).unwrap(), &Limits::DEFAULT).unwrap();
            let expected: Vec<u16> = floats.iter().map(|&f| f32_to_f16_bits(f)).collect();
            assert_eq!(back.samples, Samples::F16(expected.clone()));
            // And widened back to float, exactly.
            let widen = EncodeOptions { pixel_type: Some(PixelType::Float), ..opts };
            let back = decode(&encode_with(&back, &widen).unwrap(), &Limits::DEFAULT).unwrap();
            let wide: Vec<f32> = expected.iter().map(|&h| f16_bits_to_f32(h)).collect();
            assert_eq!(back.samples, Samples::F32(wide));
        }
    }
    let one = Image::new(FORMAT, 1, 1, 1, ColorSpace::Linear, Samples::F32(vec![2.0])).unwrap();
    assert_eq!(encode(&one).unwrap(), encode_with(&one, &options(Compression::Zip)).unwrap());
    let srgb = Image::new(FORMAT, 1, 1, 3, ColorSpace::Srgb, Samples::F32(vec![0.5; 3])).unwrap();
    assert!(matches!(encode(&srgb), Err(Error::Encode { .. })));
    let bytes = Image::new(FORMAT, 1, 1, 3, ColorSpace::Linear, Samples::U8(vec![0; 3])).unwrap();
    assert!(matches!(encode(&bytes), Err(Error::Encode { .. })));
    let uint = EncodeOptions { pixel_type: Some(PixelType::Uint), ..EncodeOptions::default() };
    assert!(matches!(encode_with(&one, &uint), Err(Error::Encode { .. })));
}

#[test]
fn channel_layouts_map_to_images() {
    let channel = |name: &str, data: ChannelData| Channel { name: name.into(), linear: false, data };
    let image = |channels| ChannelImage { width: 2, height: 1, channels };
    // Y + A: grey and alpha, F16 when both are half.
    let ya = image(vec![channel("A", ChannelData::Half(vec![1, 2])), channel("Y", ChannelData::Half(vec![3, 4]))]);
    let back = ya.to_image().unwrap();
    assert_eq!((back.channels, back.samples), (2, Samples::F16(vec![3, 1, 4, 2])));
    // Mixed types: F32. A lone channel of another name: data.
    let mixed = image(vec![
        channel("Y", ChannelData::Half(vec![0x3C00, 0])),
        channel("A", ChannelData::Uint(vec![7, 1 << 30])),
    ]);
    assert_eq!(mixed.to_image().unwrap().samples, Samples::F32(vec![1.0, 7.0, 0.0, (1_u32 << 30) as f32]));
    let depth = image(vec![channel("Z", ChannelData::Float(vec![1.5, 2.5]))]);
    assert_eq!(depth.to_image().unwrap().color_space, ColorSpace::Data);
    let aov =
        image(vec![channel("Z", ChannelData::Float(vec![0.0; 2])), channel("N.x", ChannelData::Float(vec![0.0; 2]))]);
    assert!(matches!(aov.to_image(), Err(Error::Unsupported { .. })));
    // Encoder refusals.
    let refused =
        |image: ChannelImage| matches!(encode_channels(&image, &EncodeOptions::default()), Err(Error::Encode { .. }));
    assert!(refused(image(vec![])));
    assert!(refused(image(vec![channel("", ChannelData::Half(vec![0; 2]))])));
    assert!(refused(image(vec![channel("a\0b", ChannelData::Half(vec![0; 2]))])));
    assert!(refused(image(vec![channel(&"x".repeat(256), ChannelData::Half(vec![0; 2]))])));
    assert!(refused(image(vec![
        channel("Y", ChannelData::Half(vec![0; 2])),
        channel("Y", ChannelData::Half(vec![0; 2]))
    ])));
    assert!(refused(image(vec![channel("Y", ChannelData::Half(vec![0; 3]))])));
    assert!(refused(ChannelImage { width: 0, height: 1, channels: vec![] }));
    // A name over 31 bytes sets the long-names flag and reads back.
    let long = "layer.with.a.rather.long.name.for.its.channel.R";
    let bytes =
        encode_channels(&image(vec![channel(long, ChannelData::Half(vec![1, 2]))]), &EncodeOptions::default()).unwrap();
    assert_eq!(bytes[5] & 0x04, 0x04);
    assert_eq!(decode_channels(&bytes, &Limits::DEFAULT).unwrap().1.channels[0].name, long);
}

/// Builds an uncompressed file by hand: any data window and line order, chunks in file order as
/// given by `order` (indices of lines), each line's samples one `half` channel `Y`.
fn handmade(window: Window, line_order: u8, order: &[usize], extra: &[(&str, &str, &[u8])]) -> Vec<u8> {
    let width = window.width().unwrap() as usize;
    let mut w = Writer::new();
    w.u32_le(header::MAGIC);
    w.u32_le(2);
    let attr = |w: &mut Writer, name: &str, ty: &str, value: &[u8]| {
        w.bytes(name.as_bytes());
        w.u8(0);
        w.bytes(ty.as_bytes());
        w.u8(0);
        w.i32_le(value.len() as i32);
        w.bytes(value);
    };
    attr(&mut w, "channels", "chlist", b"Y\0\x01\0\0\0\0\0\0\0\x01\0\0\0\x01\0\0\0\0");
    attr(&mut w, "compression", "compression", &[0]);
    let mut b = Writer::new();
    for v in [window.x_min, window.y_min, window.x_max, window.y_max] {
        b.i32_le(v);
    }
    attr(&mut w, "dataWindow", "box2i", b.as_slice());
    attr(&mut w, "lineOrder", "lineOrder", &[line_order]);
    for (name, ty, value) in extra {
        attr(&mut w, name, ty, value);
    }
    w.u8(0);
    let table_at = w.len();
    for _ in order {
        w.u64_le(0);
    }
    for &line in order {
        w.patch_u64_le(table_at + line * 8, w.len() as u64);
        w.i32_le(window.y_min + line as i32);
        w.i32_le(width as i32 * 2);
        for x in 0..width {
            w.u16_le((line * 100 + x) as u16);
        }
    }
    w.into_vec()
}

#[test]
fn data_windows_line_orders_and_unknown_attributes() {
    let window = Window { x_min: 10, y_min: -3, x_max: 12, y_max: 0 };
    let expected: Vec<u16> = (0..4).flat_map(|line| (0..3).map(move |x| (line * 100 + x) as u16)).collect();
    for (order, chunks) in [(0, vec![0, 1, 2, 3]), (1, vec![3, 2, 1, 0]), (2, vec![2, 0, 3, 1])] {
        let bytes = handmade(window, order, &chunks, &[("owner", "string", b"test")]);
        let (info, image) = decode_with_info(&bytes, &Limits::DEFAULT).unwrap();
        assert_eq!(info.data_window, window);
        assert_eq!(info.display_window, window, "defaults to the data window");
        assert_eq!(
            info.line_order,
            [LineOrder::IncreasingY, LineOrder::DecreasingY, LineOrder::RandomY][order as usize]
        );
        assert_eq!(
            info.attributes,
            [Attribute { name: "owner".into(), type_name: "string".into(), value: b"test".to_vec() }]
        );
        assert_eq!((image.width, image.height), (3, 4));
        assert_eq!(image.samples, Samples::F16(expected.clone()));
    }
}

#[test]
fn bad_headers_are_refused_with_the_reason() {
    let good = encode_channels(&mixed(4, 3, 1), &options(Compression::Zip)).unwrap();
    assert!(decode_channels(&good, &Limits::DEFAULT).is_ok());
    let patched = |at: usize, bytes: &[u8]| {
        let mut b = good.clone();
        b[at..at + bytes.len()].copy_from_slice(bytes);
        decode_channels(&b, &Limits::DEFAULT)
    };
    assert!(matches!(
        patched(0, b"\x76\x2f\x31\x02"),
        Err(Error::Invalid { what: "not an OpenEXR file (bad magic number)", .. })
    ));
    assert!(matches!(patched(4, &[1]), Err(Error::Unsupported { .. })));
    for (flag, what) in
        [(0x02_u8, "tiled images"), (0x08, "deep data"), (0x10, "multi-part files"), (0x80, "unknown version flags")]
    {
        assert_eq!(patched(5, &[flag]), Err(Error::Unsupported { format: FORMAT, what }));
    }
    // The compression byte follows "compression\0compression\0" and its size.
    let at = good.windows(12).position(|w| w == b"compression\0").unwrap() + 24 + 4;
    for (code, what) in
        [(5, "PXR24 compression"), (6, "B44 compression"), (8, "DWAA compression"), (9, "DWAB compression")]
    {
        assert_eq!(patched(at, &[code]), Err(Error::Unsupported { format: FORMAT, what }));
    }
    assert!(matches!(patched(at, &[10]), Err(Error::Invalid { what: "unknown compression", .. })));

    let window = Window { x_min: 0, y_min: 0, x_max: 1, y_max: 1 };
    let reason = |extra: &[(&str, &str, &[u8])]| match decode(&handmade(window, 0, &[0, 1], extra), &Limits::DEFAULT) {
        Err(Error::Invalid { what, .. } | Error::Unsupported { what, .. }) => what,
        other => panic!("{other:?}"),
    };
    assert_eq!(reason(&[("lineOrder", "int", &[0; 4])]), "attribute has the wrong type or size");
    assert_eq!(reason(&[("pixelAspectRatio", "float", &[0; 3])]), "attribute has the wrong type or size");
    assert_eq!(reason(&[("channels", "chlist", b"\0")]), "the channel list is empty");
    assert_eq!(
        reason(&[("channels", "chlist", b"Y\0\x03\0\0\0\0\0\0\0\x01\0\0\0\x01\0\0\0\0")]),
        "unknown channel pixel type"
    );
    assert_eq!(
        reason(&[("channels", "chlist", b"Y\0\x01\0\0\0\0\0\0\0\x02\0\0\0\x01\0\0\0\0")]),
        "subsampled channels"
    );
    assert_eq!(
        reason(&[("channels", "chlist", b"Y\0\x01\0\0\0\0\0\0\0\0\0\0\0\x01\0\0\0\0")]),
        "channel sampling is not positive"
    );
    let twice = b"Y\0\x01\0\0\0\0\0\0\0\x01\0\0\0\x01\0\0\0Y\0\x01\0\0\0\0\0\0\0\x01\0\0\0\x01\0\0\0\0";
    assert_eq!(reason(&[("channels", "chlist", twice)]), "duplicate channel name");
    assert_eq!(reason(&[("lineOrder", "lineOrder", &[3])]), "unknown line order");
    assert_eq!(reason(&[("type", "string", b"tiledimage")]), "parts other than scan-line images");
    let empty = Window { x_min: 0, y_min: 2, x_max: 1, y_max: 1 };
    let bytes = handmade(
        window,
        0,
        &[0, 1],
        &[("dataWindow", "box2i", &{
            let mut b = Writer::new();
            for v in [empty.x_min, empty.y_min, empty.x_max, empty.y_max] {
                b.i32_le(v);
            }
            b.into_vec()
        })],
    );
    assert!(matches!(decode(&bytes, &Limits::DEFAULT), Err(Error::Invalid { what: "the data window is empty", .. })));

    // A name without its NUL within 256 bytes.
    let mut long = good[..8].to_vec();
    long.extend(std::iter::repeat_n(b'x', 300));
    assert!(matches!(
        decode(&long, &Limits::DEFAULT),
        Err(Error::Invalid { what: "name is longer than 255 bytes", .. })
    ));
    // Required attributes.
    let mut bare = good[..8].to_vec();
    bare.push(0);
    assert!(matches!(
        decode(&bare, &Limits::DEFAULT),
        Err(Error::Invalid { what: "the channels attribute is missing", .. })
    ));
}

#[test]
fn offset_tables_and_chunk_headers_are_checked() {
    let window = Window { x_min: 0, y_min: 5, x_max: 2, y_max: 6 };
    let good = handmade(window, 0, &[0, 1], &[]);
    let table_at = good.len() - 2 * (8 + 6) - 16;
    assert!(decode(&good, &Limits::DEFAULT).is_ok());
    let what = |at: usize, bytes: &[u8]| {
        let mut b = good.clone();
        b[at..at + bytes.len()].copy_from_slice(bytes);
        match decode(&b, &Limits::DEFAULT) {
            Err(Error::Invalid { what, .. }) => what,
            Err(Error::Truncated { .. }) => "truncated",
            other => panic!("{other:?}"),
        }
    };
    assert_eq!(what(table_at, &u64::MAX.to_le_bytes()), "chunk offset is past the end of the file");
    assert_eq!(what(table_at, &(good.len() as u64 - 2).to_le_bytes()), "truncated");
    let first_chunk = table_at + 16;
    assert_eq!(what(first_chunk, &6_i32.to_le_bytes()), "chunk has the wrong line number");
    assert_eq!(what(first_chunk + 4, &(-1_i32).to_le_bytes()), "negative chunk size");
    assert_eq!(what(first_chunk + 4, &5_i32.to_le_bytes()), "uncompressed chunk has the wrong size");
    assert_eq!(what(first_chunk + 4, &7_i32.to_le_bytes()), "chunk is larger than its uncompressed size");
    assert_eq!(what(first_chunk + 4, &1000_i32.to_le_bytes()), "truncated");
}

#[test]
fn rob_02_sizes_are_limited_before_allocation() {
    let huge = Window { x_min: 0, y_min: 0, x_max: 99_999, y_max: 0 };
    let bytes = handmade(
        Window { x_min: 0, y_min: 0, x_max: 0, y_max: 0 },
        0,
        &[0],
        &[("dataWindow", "box2i", &{
            let mut b = Writer::new();
            for v in [huge.x_min, huge.y_min, huge.x_max, huge.y_max] {
                b.i32_le(v);
            }
            b.into_vec()
        })],
    );
    assert!(matches!(decode(&bytes, &Limits::DEFAULT), Err(Error::Limit { what: "image dimension", .. })));
    // Plausible header, nothing behind it: refused before the samples are allocated.
    let tall = Window { x_min: 0, y_min: 0, x_max: 16_000, y_max: 16_000 };
    let bytes = handmade(
        Window { x_min: 0, y_min: 0, x_max: 0, y_max: 0 },
        0,
        &[],
        &[("dataWindow", "box2i", &{
            let mut b = Writer::new();
            for v in [tall.x_min, tall.y_min, tall.x_max, tall.y_max] {
                b.i32_le(v);
            }
            b.into_vec()
        })],
    );
    assert!(matches!(decode(&bytes, &Limits::DEFAULT), Err(Error::Truncated { .. })));
    let good = encode_channels(&mixed(30, 30, 2), &options(Compression::Piz)).unwrap();
    let small = Limits { max_bytes: 5000, ..Limits::DEFAULT };
    assert!(matches!(decode_channels(&good, &small), Err(Error::Limit { what: "decoded size", .. })));
    assert!(matches!(read_info(&good, &Limits { max_dimension: 20, ..Limits::DEFAULT }), Err(Error::Limit { .. })));
    assert!(matches!(read_info(&good, &Limits { max_elements: 0, ..Limits::DEFAULT }), Err(Error::Limit { .. })));
}

#[test]
fn ac_wp03_2_every_truncation_is_an_error() {
    for compression in ALL {
        let bytes = encode_channels(&mixed(9, 4, 3), &options(compression)).unwrap();
        for len in 0..bytes.len() {
            assert!(decode_channels(&bytes[..len], &Limits::DEFAULT).is_err(), "{compression:?} {len}");
        }
    }
}

#[test]
fn ac_wp03_2_rob_01_mutation_fuzz() {
    let mut corpus: Vec<Vec<u8>> = FIXTURES.iter().map(|f| f.exr.to_vec()).collect();
    for compression in ALL {
        corpus.push(encode_channels(&mixed(13, 35, 9), &options(compression)).unwrap());
    }
    humanart_core::fuzz::run("exr", &corpus, 6000, 0x4558_5221, Duration::from_secs(180), |bytes| {
        if let Ok((info, image)) = decode_channels(bytes, &Limits::FUZZ) {
            let pixels = info.width() as usize * info.height() as usize;
            assert!(image.channels.iter().all(|c| c.data.len() == pixels));
            let _ = image.to_image();
        }
        let _ = read_info(bytes, &Limits::FUZZ);
    });
}

/// Dev-time cross-check `[DEP-04]`: writes our encodes with the samples they hold, for ffmpeg to
/// decode (`gen/xcheck.py`). Run with `--ignored`.
#[test]
#[ignore = "writes files for a dev-time ffmpeg cross-check"]
fn write_cross_check_files() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/agents/io-exr/xcheck");
    std::fs::create_dir_all(&dir).unwrap();
    let mut rng = Rng::new(77);
    for (channels, w, h) in [(1_u8, 37, 70), (3, 41, 33), (4, 64, 40)] {
        let n = w * h * channels as usize;
        let floats: Vec<f32> = (0..n)
            .map(|i| match i % 4 {
                0 => rng.range_f32(0.0, 1.0),
                1 => ((i / 7) as f32 * 0.37).sin() * 3.0,
                2 => 2.0_f32.powi(rng.range_i64(-20, 15) as i32),
                _ => (i % 50) as f32,
            })
            .collect();
        let image = Image::new(FORMAT, w as u32, h as u32, channels, ColorSpace::Linear, Samples::F32(floats)).unwrap();
        for compression in ALL {
            for pixel_type in [PixelType::Half, PixelType::Float] {
                let opts = EncodeOptions { compression, pixel_type: Some(pixel_type), ..EncodeOptions::default() };
                let bytes = encode_with(&image, &opts).unwrap();
                let back = decode(&bytes, &Limits::DEFAULT).unwrap();
                let name = format!("c{channels}_{compression:?}_{pixel_type:?}_{w}x{h}").to_lowercase();
                std::fs::write(dir.join(format!("{name}.exr")), &bytes).unwrap();
                // Our decode, interleaved f32 LE, for the comparison.
                let values: Vec<u8> = match &back.samples {
                    Samples::F16(v) => v.iter().flat_map(|&h| f16_bits_to_f32(h).to_le_bytes()).collect(),
                    Samples::F32(v) => v.iter().flat_map(|f| f.to_le_bytes()).collect(),
                    other => panic!("EXR decodes to floats, not {other:?}"),
                };
                std::fs::write(dir.join(format!("{name}.ours.f32")), values).unwrap();
            }
        }
    }
}
