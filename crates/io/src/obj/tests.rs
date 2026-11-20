//! Tests of the OBJ reader and writer: the hand-written fixtures, every statement form, streaming in
//! pieces of every size, limits, round trips and mutation `[ROB-01]`.
#![allow(clippy::format_push_string)] // Test text is assembled for readability, not speed.

use super::*;
use std::time::Duration;

const CUBE_OBJ: &[u8] = include_bytes!("../../tests/fixtures/obj/cube.obj");
const CUBE_MTL: &[u8] = include_bytes!("../../tests/fixtures/obj/cube.mtl");

fn parse(text: &str) -> Result<Obj> {
    Obj::parse(text.as_bytes(), &Limits::DEFAULT)
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[test]
fn reads_the_cube_fixture() {
    let obj = Obj::parse(CUBE_OBJ, &Limits::DEFAULT).unwrap();
    assert_eq!((obj.positions.len(), obj.tex_coords.len(), obj.normals.len()), (8, 4, 6));
    assert_eq!((obj.face_count(), obj.corners.len(), obj.triangle_count()), (6, 24, 12));
    assert!(obj.colors.is_empty() && obj.line_count() == 0);
    assert_eq!(obj.material_libraries, vec!["cube.mtl".to_string()]);
    assert_eq!(obj.positions[6], [1.0, 1.0, 1.0]);
    // V is flipped to the project convention: OBJ (0, 0) is the bottom left, here (0, 1).
    assert_eq!(obj.tex_coords, vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
    assert_eq!(obj.face(1).unwrap()[2], Corner::new(6, Some(2), Some(1)));
    assert_eq!(obj.face(6), None);
    let red = FaceGroup {
        object: "Cube".into(),
        group: String::new(),
        material: Some("red".into()),
        smoothing: 0,
        faces: 0..3,
    };
    let blue = FaceGroup { material: Some("blue".into()), smoothing: 1, faces: 3..6, ..red.clone() };
    assert_eq!(obj.groups, vec![red, blue]);
    // Every face winds counter-clockwise around its normal, seen from outside.
    obj.for_each_triangle(|_, [a, b, c]| {
        let p = |c: Corner| obj.positions[c.position as usize];
        let n = obj.normals[a.normal as usize];
        assert!(dot(cross(sub(p(b), p(a)), sub(p(c), p(a))), n) > 0.0);
    });

    let mesh = obj.to_indexed().unwrap();
    assert_eq!((mesh.positions.len(), mesh.normals.len(), mesh.tex_coords.len(), mesh.indices.len()), (24, 24, 24, 36));
    assert!(mesh.colors.is_empty());
    assert_eq!(mesh.group_triangles, vec![0..6, 6..12]);
    assert_eq!(mesh.source_positions[..4], [0, 3, 2, 1]);
    assert_eq!(mesh.indices[..6], [0, 1, 2, 0, 2, 3]);

    let materials = parse_mtl(CUBE_MTL, &Limits::DEFAULT).unwrap();
    assert_eq!(materials.len(), 2);
    assert_eq!(materials[0].name, "red");
    assert_eq!(materials[0].diffuse_map.as_ref().map(|t| t.file.as_str()), Some("red.png"));
    assert_eq!(materials[1].dissolve, Some(0.5));
    assert_eq!(materials[1].bump_map.as_ref().and_then(|t| t.bump_multiplier), Some(0.25));
    for group in &obj.groups {
        assert!(materials.iter().any(|m| Some(&m.name) == group.material.as_ref()));
    }
}

#[test]
fn every_statement_form() {
    let text = "\u{FEFF}# header\r\n\
        v 0 0 0\r\n\
        v 1 0 0 1.0\r\n\
        v 1 1 0 0.5 0.25 1\n\
        v 0 1 0 # trailing comment\n\
        vt 0.25\n\
        vt 0.5 0.75 0.0\n\
        vn 0 0 1\n\
        g left side\n\
        f 1 2 3\n\
        f 1/1 2/2 3/1\n\
        fo -4//-1 -3//1 -1//1\n\
        f 1/1/1 \\\n  3/2/1 4/1/1\n\
        l 1 2/1 3\n\
        p 1 2\n\
        vp 0.5\n\
        usemap none\n\
        o second\n\
        usemtl\n\
        s 4\n\
        f 4 3 2 1\n\
        s 0\n\
        f 1 2 3";
    let obj = parse(text).unwrap();
    assert_eq!(obj.positions.len(), 4);
    // The first vertex with a colour makes every vertex coloured; the others are white.
    assert_eq!(obj.colors, vec![[1.0; 3], [1.0; 3], [0.5, 0.25, 1.0], [1.0; 3]]);
    assert_eq!(obj.tex_coords, vec![[0.25, 1.0], [0.5, 0.25]]);
    assert_eq!(obj.face_count(), 6);
    assert_eq!(
        obj.face(1).unwrap(),
        &[Corner::new(0, Some(0), None), Corner::new(1, Some(1), None), Corner::new(2, Some(0), None)]
    );
    assert_eq!(
        obj.face(2).unwrap(),
        &[Corner::new(0, None, Some(0)), Corner::new(1, None, Some(0)), Corner::new(3, None, Some(0))]
    );
    assert_eq!(obj.face(3).unwrap().len(), 3, "a continued line is one statement");
    assert_eq!(obj.face(4).unwrap().len(), 4);
    assert_eq!((obj.line_count(), obj.line(0)), (1, Some(&[0, 1, 2][..])));
    let groups: Vec<(&str, &str, u32, Range<u32>)> =
        obj.groups.iter().map(|g| (g.object.as_str(), g.group.as_str(), g.smoothing, g.faces.clone())).collect();
    assert_eq!(groups, vec![("", "left side", 0, 0..4), ("second", "", 4, 4..5), ("second", "", 0, 5..6)]);
    assert!(obj.groups.iter().all(|g| g.material.is_none()));
    assert_eq!(obj.triangle_count(), 7);
    let mesh = obj.to_indexed().unwrap();
    assert_eq!(mesh.indices.len(), 21);
    assert_eq!(mesh.colors.len(), mesh.positions.len());
    assert_eq!(mesh.group_triangles, vec![0..4, 4..6, 6..7]);
    // Missing attributes of a welded vertex are zero.
    assert_eq!((mesh.normals[0], mesh.tex_coords[0]), ([0.0; 3], [0.0; 2]));
}

#[test]
fn errors_point_at_their_line() {
    let at = |text: &str| match parse(text) {
        Err(Error::Invalid { offset, .. }) => offset,
        other => panic!("{text:?}: {other:?}"),
    };
    assert_eq!(at("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 0\n"), 24, "index 0");
    assert_eq!(at("v 0 0 0\nf 1 1 2\n"), 8, "a position defined later");
    assert_eq!(at("v 0 0 0\nf 1 1 -2\n"), 8, "relative index before the first");
    assert_eq!(at("v 0 0 0\nf 1/1 1/1 1/1\n"), 8, "no texture coordinates");
    assert_eq!(at("v 0 0 0\nf 1//1 1//1 1//1\n"), 8, "no normals");
    assert_eq!(at("v 0 0 0\nf 1 1\n"), 8, "two corners");
    assert_eq!(at("v 0 0 0\nf 1/1/1/1 1 1\n"), 8);
    assert_eq!(at("v 0 0 0\nf 1 1 x\n"), 8);
    assert_eq!(at("v 0 0 0\nf 1 1 99999999999999999999\n"), 8);
    assert_eq!(at("v 0 0\n"), 0);
    assert_eq!(at("v 0 0 0 0 0\n"), 0);
    assert_eq!(at("v 0 0 0 1 1 1 1\n"), 0);
    assert_eq!(at("v 0 0 nan\n"), 0);
    assert_eq!(at("v 0 0 1e39\n"), 0, "overflows to infinity");
    assert_eq!(at("vn 0 1\n"), 0);
    assert_eq!(at("vt\n"), 0);
    assert_eq!(at("vt 1 2 3 4\n"), 0);
    assert_eq!(at("v 0 0 0\nl 1\n"), 8);
    assert_eq!(at("v 0 0 0\nvn 0 0 1\nl 1//1 1//1\n"), 17);
    assert_eq!(at("s\n"), 0);
    assert_eq!(at("s -1\n"), 0);
    assert_eq!(at("\n\n\u{1}garbage\n"), 2);
    assert!(matches!(Obj::parse(b"v 0 0 0\ng \xFF\n", &Limits::DEFAULT), Err(Error::Invalid { offset: 8, .. })));
    assert!(matches!(parse("cstype bspline\n"), Err(Error::Unsupported { .. })));
    assert!(matches!(parse("surf 0 1 0 1 1 2 3 4\n"), Err(Error::Unsupported { .. })));
    // An unknown statement made of a plain word is skipped.
    assert_eq!(parse("bevel on\nshadow_obj x.obj\n").unwrap(), Obj::default());
}

#[test]
fn limits_are_enforced() {
    let limits = Limits { max_elements: 3, ..Limits::DEFAULT };
    let four = b"v 0 0 0\nv 0 0 0\nv 0 0 0\nv 0 0 0\n";
    assert!(matches!(Obj::parse(four, &limits), Err(Error::Limit { what: "vertex count", got: 4, .. })));
    assert!(matches!(Obj::parse(b"v 0 0 0\nf 1 1 1 1\n", &limits), Err(Error::Limit { what: "corner count", .. })));
    let limits = Limits { max_bytes: 30, ..Limits::DEFAULT };
    assert!(matches!(Obj::parse(four, &limits), Err(Error::Limit { what: "decoded mesh size", .. })));
    // One face of too many corners.
    let mut face = String::from("v 0 0 0\nf");
    for _ in 0..=MAX_FACE_CORNERS {
        face.push_str(" 1");
    }
    assert!(matches!(parse(&face), Err(Error::Limit { what: "corners of one face", .. })));
    let polyline = face.replace('f', "l");
    assert!(matches!(parse(&polyline), Err(Error::Limit { what: "vertices of one polyline", .. })));
    // A line longer than the limit, whole or in pieces, or continued.
    let long = vec![b'#'; MAX_LINE_LEN + 1];
    assert!(matches!(Obj::parse(&long, &Limits::DEFAULT), Err(Error::Limit { what: "line length", .. })));
    let mut with_end = long.clone();
    with_end.push(b'\n');
    assert!(matches!(Obj::parse(&with_end, &Limits::DEFAULT), Err(Error::Limit { what: "line length", .. })));
    let mut parser = ObjParser::new(Limits::DEFAULT);
    let mut result = Ok(());
    for _ in 0..=MAX_LINE_LEN / 4096 {
        result = result.and_then(|()| parser.feed(&[b'x'; 4096]));
    }
    assert!(matches!(result, Err(Error::Limit { what: "line length", .. })));
    let continued = "g a \\\n".repeat(MAX_LINE_LEN / 5 + 1);
    assert!(matches!(parse(&continued), Err(Error::Limit { what: "line length", .. })));
}

/// A mesh of every kind of content, for the streaming and round-trip tests.
fn sample_text() -> String {
    let mut text = String::from("mtllib a.mtl b.mtl\n# a comment\r\n");
    for i in 0..40 {
        let f = i as f32;
        text.push_str(&format!(
            "v {} {} {}{}\n",
            f * 0.125,
            -f,
            f * f / 3.0,
            if i % 7 == 3 { " 0.5 0.25 1" } else { "" }
        ));
        text.push_str(&format!("vt {} {}\nvn {} 0 1\n", f / 64.0, 1.0 - f / 32.0, f / 40.0));
    }
    for i in 0..30 {
        if i % 9 == 0 {
            text.push_str(&format!("o part{i}\ng g{i} extra\nusemtl m{}\ns {}\n", i % 2, i % 3));
        }
        text.push_str(&match i % 4 {
            0 => format!("f {} {} {}\n", i + 1, i + 2, i + 3),
            1 => format!("f {}/{} {}/{} {}/{} {}/{}\n", i + 1, i + 1, i + 2, i + 2, i + 3, i + 3, i + 4, i + 4),
            2 => format!("f {}//{} {}//{} \\\n {}//{}\n", i + 1, i + 1, i + 2, i + 2, i + 3, i + 3),
            _ => "f -1/-1/-1 -2/-2/-2 -3/-3/-3\n".to_string(),
        });
    }
    text.push_str("l 1 2 3 4\nl 5 6");
    text
}

#[test]
fn streaming_in_pieces_of_every_size_gives_the_same_result() {
    let text = sample_text();
    let whole = parse(&text).unwrap();
    assert_eq!(whole.face_count(), 30);
    for size in (1..64).chain([97, 1000, 4096]) {
        let mut parser = ObjParser::new(Limits::DEFAULT);
        for chunk in text.as_bytes().chunks(size) {
            parser.feed(chunk).unwrap();
        }
        assert_eq!(parser.finish().unwrap(), whole, "chunks of {size}");
    }
    // An error is reported at the same file offset whatever the pieces.
    let broken = format!("{text}\nf 1 2 x\n");
    let expected = parse(&broken).unwrap_err();
    assert_eq!(expected.offset(), Some(text.len() + 1));
    for size in [1, 2, 3, 5, 64] {
        let mut parser = ObjParser::new(Limits::DEFAULT);
        let mut result = Ok(());
        for chunk in broken.as_bytes().chunks(size) {
            result = result.and_then(|()| parser.feed(chunk));
        }
        let error = result.and_then(|()| parser.finish().map(|_| ())).unwrap_err();
        assert_eq!(error, expected, "chunks of {size}");
    }
}

#[test]
fn write_then_read_gives_the_same_mesh() {
    for source in [sample_text().into_bytes(), CUBE_OBJ.to_vec()] {
        let obj = Obj::parse(&source, &Limits::DEFAULT).unwrap();
        let written = write(&obj).unwrap();
        assert_eq!(Obj::parse(&written, &Limits::DEFAULT).unwrap(), obj);
    }
    // Random geometry: positions and normals come back bit for bit.
    let mut rng = humanart_core::rng::Rng::new(0x0B1);
    let mut obj = Obj::default();
    for _ in 0..500 {
        let mut v = || rng.range_f32(-1000.0, 1000.0) * 10_f32.powi(rng.range_i64(-30, 5) as i32);
        obj.positions.push([v(), v(), v()]);
        obj.normals.push([v(), v(), v()]);
        obj.tex_coords.push([v(), 0.5 + rng.f32() * 0.5]);
    }
    for i in 0..498_u32 {
        obj.corners.extend([
            Corner::new(i, Some(i), Some(i)),
            Corner::new(i + 1, None, Some(i)),
            Corner::new(i + 2, Some(i), None),
        ]);
        obj.face_ends.push(obj.corners.len() as u32);
    }
    obj.groups.push(FaceGroup {
        object: "o".into(),
        group: "g h".into(),
        material: Some("m".into()),
        smoothing: 2,
        faces: 0..498,
    });
    obj.line_vertices = vec![0, 1, 2];
    obj.line_ends = vec![3];
    let back = Obj::parse(&write(&obj).unwrap(), &Limits::DEFAULT).unwrap();
    assert_eq!(back, obj);
}

#[test]
fn invalid_meshes_are_not_written() {
    let base = Obj::parse(CUBE_OBJ, &Limits::DEFAULT).unwrap();
    let refused = |change: &dyn Fn(&mut Obj)| {
        let mut obj = base.clone();
        change(&mut obj);
        assert!(matches!(write(&obj), Err(Error::Encode { .. })));
        assert!(matches!(obj.to_indexed(), Err(Error::Encode { .. })));
    };
    refused(&|o| o.colors = vec![[1.0; 3]]);
    refused(&|o| o.positions[0][1] = f32::NAN);
    refused(&|o| o.normals[0][1] = f32::INFINITY);
    refused(&|o| o.tex_coords[0][1] = f32::NAN);
    refused(&|o| o.corners[0].position = 8);
    refused(&|o| o.corners[0].tex_coord = 4);
    refused(&|o| o.corners[0].normal = 6);
    refused(&|o| o.face_ends[0] = 2);
    refused(&|o| o.face_ends[5] = 23);
    refused(&|o| o.face_ends.swap(1, 2));
    refused(&|o| o.groups[1].faces = 2..6);
    refused(&|o| o.groups[1].faces = 3..7);
    refused(&|o| o.line_ends = vec![1]);
    refused(&|o| {
        o.line_vertices = vec![0, 9];
        o.line_ends = vec![2];
    });
    let mut named = base.clone();
    named.groups[0].object = "two\nlines".into();
    assert!(matches!(write(&named), Err(Error::Encode { .. })));
    assert!(named.to_indexed().is_ok(), "names do not matter to the mesh");
}

#[test]
fn faces_outside_every_run_are_written_with_the_state_before_them() {
    let mut obj = parse("v 0 0 0\nf 1 1 1\nf 1 1 1\nf 1 1 1\n").unwrap();
    obj.groups = vec![FaceGroup { material: Some("m".into()), faces: 1..2, ..FaceGroup::default() }];
    let back = Obj::parse(&write(&obj).unwrap(), &Limits::DEFAULT).unwrap();
    assert_eq!(back.groups.len(), 2);
    assert_eq!((back.groups[0].faces.clone(), back.groups[1].faces.clone()), (0..1, 1..3));
}

/// Parses, welds, and checks that writing and reading back gives the same arrays.
fn exercise(bytes: &[u8]) {
    let Ok(obj) = Obj::parse(bytes, &Limits::FUZZ) else { return };
    obj.validate().expect("a parsed file is consistent");
    let mesh = obj.to_indexed().expect("a parsed file welds");
    assert_eq!(mesh.indices.len(), 3 * obj.triangle_count());
    assert!(mesh.indices.iter().all(|&i| (i as usize) < mesh.positions.len()));
    let covered: u32 = obj.groups.iter().map(|g| g.faces.len() as u32).sum();
    assert_eq!(covered as usize, obj.face_count(), "runs cover every face");
    if let Ok(text) = write(&obj) {
        let back = Obj::parse(&text, &Limits::FUZZ).expect("written text reads back");
        let close = |a: f32, b: f32| a == b || (a - b).abs() <= 4.0 * f32::EPSILON * a.abs().max(1.0);
        assert!(back.tex_coords.iter().zip(&obj.tex_coords).all(|(a, b)| close(a[0], b[0]) && close(a[1], b[1])));
        assert_eq!(Obj { tex_coords: Vec::new(), ..back }, Obj { tex_coords: Vec::new(), ..obj });
    }
}

/// `[ROB-01]` Mutated OBJ files: no panic, no hang, and whatever parses writes and reads back.
#[test]
fn fuzz_obj() {
    let corpus = vec![
        CUBE_OBJ.to_vec(),
        sample_text().into_bytes(),
        b"v 1 2 3 0 0 1\nv 1 1 1\nvt 0.5 0.5\nf 1/1 2/1 -1/1\n".to_vec(),
    ];
    humanart_core::fuzz::run("obj", &corpus, 6000, 0x0B1_0001, Duration::from_secs(120), exercise);
}

/// `[IO-11]` Throughput of the streaming reader on a generated scan-like mesh (printed; the number
/// that is recorded comes from a release build, `--release -- --nocapture`).
#[test]
fn large_file_throughput() {
    let mut text = String::new();
    let side = 200_u32;
    for y in 0..side {
        for x in 0..side {
            let (fx, fy) = (x as f32 * 0.001, y as f32 * 0.001);
            text.push_str(&format!("v {fx:.6} {fy:.6} {:.6} 0.8 0.6 0.5\nvt {fx:.6} {fy:.6}\n", (fx * fy).sin()));
        }
    }
    for y in 0..side - 1 {
        for x in 0..side - 1 {
            let i = y * side + x + 1;
            text.push_str(&format!("f {i}/{i} {}/{} {}/{}\n", i + 1, i + 1, i + side, i + side));
        }
    }
    let started = std::time::Instant::now();
    let mut parser = ObjParser::new(Limits::DEFAULT);
    for chunk in text.as_bytes().chunks(1 << 16) {
        parser.feed(chunk).unwrap();
    }
    let obj = parser.finish().unwrap();
    let seconds = started.elapsed().as_secs_f64();
    assert_eq!((obj.positions.len(), obj.face_count()), (40_000, 39_601));
    println!(
        "OBJ read: {:.1} MB in {:.1} ms, {:.0} MB/s",
        text.len() as f64 / 1e6,
        seconds * 1e3,
        text.len() as f64 / 1e6 / seconds
    );
}
