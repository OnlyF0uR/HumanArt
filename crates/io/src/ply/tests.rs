//! Tests of the PLY reader and writer: the fixtures written independently by
//! `tests/fixtures/ply/make_fixtures.py` in all three encodings, header and data errors, limits,
//! round trips and mutation `[ROB-01]`.

use super::*;
use std::time::Duration;

const ASCII: &[u8] = include_bytes!("../../tests/fixtures/ply/tetra_ascii.ply");
const LE: &[u8] = include_bytes!("../../tests/fixtures/ply/tetra_le.ply");
const BE: &[u8] = include_bytes!("../../tests/fixtures/ply/tetra_be.ply");

fn parse(data: &[u8]) -> Result<Ply> {
    Ply::parse(data, &Limits::DEFAULT)
}

#[test]
fn the_three_encodings_of_the_fixture_read_the_same() {
    let ascii = parse(ASCII).unwrap();
    assert_eq!(ascii.encoding, Encoding::Ascii);
    assert_eq!(ascii.comments, vec!["made by make_fixtures.py for the HumanArt PLY tests".to_string()]);
    assert_eq!(ascii.obj_info, vec!["tetrahedron with every scalar type".to_string()]);
    let names: Vec<(&str, usize, usize)> =
        ascii.elements.iter().map(|e| (e.name.as_str(), e.count, e.properties.len())).collect();
    assert_eq!(names, vec![("vertex", 4, 13), ("face", 4, 2), ("edge", 3, 4)]);
    let vertex = ascii.element("vertex").unwrap();
    assert_eq!(vertex.property("quality").unwrap().scalar(), Some(&Column::F64(vec![0.1, 0.2, 12345.678, -3.5])));
    assert_eq!(vertex.property("alpha").unwrap().scalar(), Some(&Column::U8(vec![255, 255, 128, 255])));
    let face = ascii.element("face").unwrap();
    assert_eq!(face.property("flags").unwrap().scalar(), Some(&Column::I16(vec![0, -1, 32767, -32768])));
    let (items, range) = face.property("vertex_indices").unwrap().list(3).unwrap();
    assert_eq!((items.scalar_type(), range.clone()), (ScalarType::I32, 9..12));
    assert_eq!(range.map(|i| items.get(i).unwrap()).collect::<Vec<_>>(), [1.0, 2.0, 3.0]);
    assert!(face.property("flags").unwrap().list(0).is_none());
    let edge = ascii.element("edge").unwrap();
    assert_eq!(edge.property("vertex2").unwrap().scalar(), Some(&Column::U32(vec![1, 2, 3])));
    assert_eq!(edge.property("sign").unwrap().scalar(), Some(&Column::I8(vec![1, -1, -128])));
    assert_eq!(edge.property("weight").unwrap().scalar(), Some(&Column::U16(vec![0, 65535, 7])));
    assert!(ascii.element("material").is_none() && edge.property("nothing").is_none());

    for (binary, encoding) in [(LE, Encoding::BinaryLittleEndian), (BE, Encoding::BinaryBigEndian)] {
        let ply = parse(binary).unwrap();
        assert_eq!(ply.encoding, encoding);
        assert_eq!(Ply { encoding: Encoding::Ascii, ..ply }, ascii);
    }
}

#[test]
fn the_writer_reproduces_the_fixtures_byte_for_byte() {
    let ascii = parse(ASCII).unwrap();
    for (bytes, encoding) in
        [(ASCII, Encoding::Ascii), (LE, Encoding::BinaryLittleEndian), (BE, Encoding::BinaryBigEndian)]
    {
        let ply = Ply { encoding, ..ascii.clone() };
        assert_eq!(write(&ply).unwrap(), bytes, "{encoding:?}");
    }
}

#[test]
fn the_fixture_as_a_mesh() {
    let mesh = parse(LE).unwrap().to_mesh().unwrap();
    assert_eq!(mesh.positions, vec![[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]);
    assert_eq!(mesh.normals[1], [-0.5, -0.5, 0.5]);
    assert_eq!(mesh.colors[2], [0.0, 0.0, 1.0, 128.0 / 255.0]);
    // V flipped: the file's (0.5, 0.25) is (0.5, 0.75) with the origin at the top.
    assert_eq!(mesh.tex_coords, vec![[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.5, 0.75]]);
    assert_eq!((mesh.face_count(), mesh.face(0), mesh.face(4)), (4, Some(&[0, 2, 1][..]), None));
    // Every face winds counter-clockwise seen from outside (the centroid is the origin).
    for i in 0..mesh.face_count() {
        let f = mesh.face(i).unwrap();
        let p = |k: usize| mesh.positions[f[k] as usize];
        let (a, b, c) = (p(0), p(1), p(2));
        let (u, v) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [c[0] - a[0], c[1] - a[1], c[2] - a[2]]);
        let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        assert!(n[0] * (a[0] + b[0] + c[0]) + n[1] * (a[1] + b[1] + c[1]) + n[2] * (a[2] + b[2] + c[2]) > 0.0);
    }

    // Back to PLY and again to a mesh: the same, as the colours are multiples of 1/255.
    for encoding in [Encoding::Ascii, Encoding::BinaryLittleEndian, Encoding::BinaryBigEndian] {
        let ply = mesh.to_ply(encoding).unwrap();
        let back = parse(&write(&ply).unwrap()).unwrap();
        assert_eq!(back.to_mesh().unwrap(), mesh, "{encoding:?}");
        let vertex = back.element("vertex").unwrap();
        assert!(vertex.property("alpha").is_some(), "one colour is translucent");
        assert_eq!(
            back.element("face").unwrap().properties[0].data,
            PropertyData::List {
                count_type: ScalarType::U8,
                ends: vec![3, 6, 9, 12],
                values: Column::U32(mesh.face_vertices.clone()),
            }
        );
    }
}

#[test]
fn point_clouds_and_other_spellings() {
    let text = b"ply\r\nformat ascii 1.0\r\nelement vertex 2\r\nproperty double x\r\nproperty int y\r\nproperty uchar z\r\n\
        property ushort diffuse_red\r\nproperty ushort diffuse_green\r\nproperty ushort diffuse_blue\r\nproperty float s\r\n\
        property float t\r\nend_header\r\n0.5 -2 7 65535 0 32768 0.25 0.5\r\nnan 3 0 0 0 0 1 1\r\n";
    let mesh = parse(text).unwrap().to_mesh().unwrap();
    assert_eq!(mesh.positions[0], [0.5, -2.0, 7.0]);
    assert!(mesh.positions[1][0].is_nan(), "missing points of organised scans stay NaN");
    assert_eq!(mesh.colors[0], [1.0, 0.0, (32768.0 / 65535.0) as f32, 1.0]);
    assert_eq!(mesh.tex_coords[0], [0.25, 0.5]);
    assert!(mesh.normals.is_empty() && mesh.face_count() == 0);
    let alone = mesh.to_ply(Encoding::Ascii).unwrap();
    assert!(alone.element("face").is_none() && alone.element("vertex").unwrap().property("alpha").is_none());

    let with_index = b"ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n\
        element face 1\nproperty list ushort uchar vertex_index\nend_header\n";
    let mut data = with_index.to_vec();
    for v in [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
        data.extend(v.to_le_bytes());
    }
    data.extend([3, 0, 0, 1, 2]);
    data.extend(b"trailing bytes are ignored");
    let mesh = parse(&data).unwrap().to_mesh().unwrap();
    assert_eq!((mesh.face_ends.clone(), mesh.face_vertices.clone()), (vec![3], vec![0, 1, 2]));
    // Faces longer than 255 vertices get a 32-bit count.
    let big = PlyMesh {
        positions: vec![[0.0; 3]; 300],
        face_ends: vec![300],
        face_vertices: (0..300).collect(),
        ..PlyMesh::default()
    };
    let ply = big.to_ply(Encoding::BinaryBigEndian).unwrap();
    assert!(matches!(ply.elements[1].properties[0].data, PropertyData::List { count_type: ScalarType::U32, .. }));
    assert_eq!(parse(&write(&ply).unwrap()).unwrap().to_mesh().unwrap(), big);
}

#[test]
fn header_errors() {
    let header = |text: &str| parse(text.as_bytes());
    let invalid_at = |text: &str| match header(text) {
        Err(Error::Invalid { offset, .. }) => offset,
        other => panic!("{text:?}: {other:?}"),
    };
    assert_eq!(invalid_at("plx\nformat ascii 1.0\nend_header\n"), 0);
    assert_eq!(invalid_at("ply\nend_header\n"), 0, "no format line");
    assert_eq!(invalid_at("ply\nformat ascii 1.0\nformat ascii 1.0\nend_header\n"), 21);
    assert_eq!(invalid_at("ply\nformat text 1.0\nend_header\n"), 4);
    assert_eq!(invalid_at("ply\nformat ascii\nend_header\n"), 4);
    assert_eq!(invalid_at("ply\nformat ascii 1.0 x\nend_header\n"), 4);
    assert_eq!(invalid_at("ply\nformat ascii 1.0\nproperty float x\nend_header\n"), 21);
    assert_eq!(invalid_at("ply\nformat ascii 1.0\nelement v -1\nend_header\n"), 21);
    assert_eq!(invalid_at("ply\nformat ascii 1.0\nelement v x\nend_header\n"), 21);
    assert_eq!(invalid_at("ply\nformat ascii 1.0\nelement v 1\nproperty half x\nend_header\n"), 33);
    assert_eq!(invalid_at("ply\nformat ascii 1.0\nelement v 1\nproperty list float int x\nend_header\n"), 33);
    assert_eq!(invalid_at("ply\nformat ascii 1.0\nelement v 1\nproperty list uchar\nend_header\n"), 33);
    assert_eq!(invalid_at("ply\nformat ascii 1.0\nmaterial x\nend_header\n"), 21);
    assert!(matches!(header("ply\nformat ascii 2.0\nend_header\n"), Err(Error::Unsupported { .. })));
    assert!(matches!(header("ply\nformat ascii 1.0\n"), Err(Error::Truncated { .. })));
    assert!(matches!(header("ply"), Err(Error::Truncated { .. })));
    assert!(matches!(header(""), Err(Error::Truncated { .. })));
    let long = format!(
        "ply\nformat ascii 1.0\n{}end_header\n",
        "comment xxxxxxxxxxxxxxxxxxxxxxx\n".repeat(MAX_HEADER_LEN / 32)
    );
    assert!(matches!(header(&long), Err(Error::Limit { what: "header length", .. })));
    let mut bad_utf8 = b"ply\nformat ascii 1.0\ncomment ".to_vec();
    bad_utf8.extend([0xFF, b'\n']);
    bad_utf8.extend(b"end_header\n");
    assert!(matches!(parse(&bad_utf8), Err(Error::Invalid { offset: 21, .. })));
    // An empty file is fine.
    let empty = header("ply\nformat binary_big_endian 1.0\nend_header\n").unwrap();
    assert!(empty.elements.is_empty());
    assert!(matches!(empty.to_mesh(), Err(Error::Invalid { .. })));
}

#[test]
fn data_errors_and_limits() {
    const HEAD: &str = "ply\nformat ascii 1.0\nelement vertex 2\nproperty uchar x\nproperty float y\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n";
    let at = HEAD.len();
    let data = |rows: &str| parse(format!("{HEAD}{rows}").as_bytes());
    assert!(data("1 2\n3 4\n2 0 1\n").is_ok());
    assert!(matches!(data("256 2\n3 4\n2 0 1\n"), Err(Error::Invalid { offset, .. }) if offset == at), "uchar range");
    assert!(matches!(data("1.5 2\n3 4\n2 0 1\n"), Err(Error::Invalid { offset, .. }) if offset == at), "integer type");
    assert!(matches!(data("1 x\n3 4\n2 0 1\n"), Err(Error::Invalid { offset, .. }) if offset == at + 2));
    assert!(matches!(data("1 2\n3 4\n-1 0 1\n"), Err(Error::Invalid { .. })), "negative list length");
    assert!(matches!(data("1 2\n3 4\n2 0\n"), Err(Error::Truncated { .. })));
    assert!(matches!(data("1 2\n3 4\n9 0 1\n"), Err(Error::Truncated { .. })));
    assert!(matches!(data("1 2\n3 4\n2 0 1\n5\n"), Err(Error::Invalid { what: "data after the last element", .. })));
    assert!(matches!(data("1 2\n3"), Err(Error::Truncated { .. })));
    // Mesh-level errors.
    let mesh_text = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n\
        element face 1\nproperty list uchar uint vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n";
    let mesh = |face: &str, header_change: (&str, &str)| {
        parse(format!("{mesh_text}{face}\n").replace(header_change.0, header_change.1).as_bytes()).unwrap().to_mesh()
    };
    assert!(mesh("3 0 1 2", ("", "")).is_ok());
    assert!(matches!(
        data("1 2\n3 4\n2 0 1\n").unwrap().to_mesh(),
        Err(Error::Invalid { what: "no vertex positions", .. })
    ));
    assert!(matches!(mesh("2 0 1", ("", "")), Err(Error::Invalid { what: "a face has fewer than 3 vertices", .. })));
    assert!(matches!(mesh("3 0 1 3", ("", "")), Err(Error::Invalid { what: "face vertex index out of range", .. })));
    assert!(matches!(
        mesh("3 0 1 2", ("uchar uint", "uchar float")),
        Err(Error::Invalid { what: "face vertex indices are not integers", .. })
    ));
    assert!(matches!(mesh("1", ("list uchar uint vertex_indices", "uint vertex_indices")), Err(Error::Invalid { .. })));
    assert!(matches!(mesh("3 0 1 2", ("vertex_indices", "corners")), Err(Error::Invalid { .. })));
    assert!(matches!(mesh("3 0 1 2", ("float z", "list uchar float z")), Err(Error::Invalid { .. })));

    // Counts that the input cannot hold are refused before allocation.
    let huge = "ply\nformat binary_little_endian 1.0\nelement vertex 4000000000\nproperty float x\nend_header\n";
    assert!(matches!(parse(huge.as_bytes()), Err(Error::Truncated { .. })));
    let huge_ascii = "ply\nformat ascii 1.0\nelement vertex 99999999999\nproperty float x\nend_header\n1\n";
    assert!(matches!(parse(huge_ascii.as_bytes()), Err(Error::Truncated { .. })));
    let mut list =
        b"ply\nformat binary_big_endian 1.0\nelement face 1\nproperty list uint double v\nend_header\n".to_vec();
    list.extend(0xFFFF_FFF0_u32.to_be_bytes());
    list.extend([0; 64]);
    assert!(matches!(parse(&list), Err(Error::Truncated { .. })));
    // And the configured limits hold.
    let three_rows = "ply\nformat ascii 1.0\nelement vertex 3\nproperty double x\nend_header\n1 2 3\n";
    let few = Limits { max_elements: 2, ..Limits::DEFAULT };
    assert!(matches!(Ply::parse(three_rows.as_bytes(), &few), Err(Error::Limit { .. })));
    let small = Limits { max_bytes: 16, ..Limits::DEFAULT };
    assert!(matches!(Ply::parse(three_rows.as_bytes(), &small), Err(Error::Limit { what: "decoded size", .. })));
    let lists = "ply\nformat ascii 1.0\nelement face 2\nproperty list uchar int v\nend_header\n2 1 2 2 3 4\n";
    assert!(matches!(
        Ply::parse(lists.as_bytes(), &Limits { max_elements: 3, ..Limits::DEFAULT }),
        Err(Error::Limit { .. })
    ));
    // An element of no properties has rows but no data.
    let bare = parse(b"ply\nformat ascii 1.0\nelement nothing 1000000\nend_header\n").unwrap();
    assert_eq!(bare.elements[0].count, 1_000_000);
    assert_eq!(write(&bare).unwrap(), b"ply\nformat ascii 1.0\nelement nothing 1000000\nend_header\n");
}

#[test]
fn every_value_survives_a_write_and_a_read() {
    let mut rng = humanart_core::rng::Rng::new(0x9_1A7);
    let n = 300;
    let mut column = |ty: ScalarType| {
        let mut c = Column::with_capacity(ty, n);
        for _ in 0..n {
            let bits = rng.next_u64();
            match &mut c {
                Column::I8(v) => v.push(bits as i8),
                Column::U8(v) => v.push(bits as u8),
                Column::I16(v) => v.push(bits as i16),
                Column::U16(v) => v.push(bits as u16),
                Column::I32(v) => v.push(bits as i32),
                Column::U32(v) => v.push(bits as u32),
                // Any finite bit pattern: subnormals, huge and tiny values.
                Column::F32(v) => v.push(Some(f32::from_bits(bits as u32)).filter(|x| x.is_finite()).unwrap_or(1.5)),
                Column::F64(v) => v.push(Some(f64::from_bits(bits)).filter(|x| x.is_finite()).unwrap_or(-2.5)),
            }
        }
        c
    };
    let types = [
        ScalarType::I8,
        ScalarType::U8,
        ScalarType::I16,
        ScalarType::U16,
        ScalarType::I32,
        ScalarType::U32,
        ScalarType::F32,
        ScalarType::F64,
    ];
    let mut properties: Vec<Property> = types
        .iter()
        .map(|&ty| Property { name: format!("p_{}", ty.name()), data: PropertyData::Scalar(column(ty)) })
        .collect();
    let ends: Vec<u32> = (1..=n as u32).map(|i| i * 2 - i % 3).collect();
    let items = *ends.last().unwrap() as usize;
    let values = Column::F64((0..items).map(|i| i as f64 / 7.0).collect());
    properties
        .push(Property { name: "list".into(), data: PropertyData::List { count_type: ScalarType::U16, ends, values } });
    let mut ply = Ply {
        encoding: Encoding::Ascii,
        comments: vec![String::new(), "  spaced  ".into()],
        obj_info: vec!["info".into()],
        elements: vec![Element { name: "thing".into(), count: n, properties }],
    };
    for encoding in [Encoding::Ascii, Encoding::BinaryLittleEndian, Encoding::BinaryBigEndian] {
        ply.encoding = encoding;
        let bytes = write(&ply).unwrap();
        assert_eq!(parse(&bytes).unwrap(), ply, "{encoding:?}");
    }
    // NaN and infinities survive in binary bit for bit, in ASCII as NaN and infinities.
    let odd = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(0x7FC0_1234), -0.0];
    let column = PropertyData::Scalar(Column::F32(odd.to_vec()));
    for encoding in [Encoding::Ascii, Encoding::BinaryLittleEndian, Encoding::BinaryBigEndian] {
        let ply = Ply {
            encoding,
            comments: vec![],
            obj_info: vec![],
            elements: vec![Element {
                name: "v".into(),
                count: 5,
                properties: vec![Property { name: "x".into(), data: column.clone() }],
            }],
        };
        let back = parse(&write(&ply).unwrap()).unwrap();
        let Some(Column::F32(values)) = back.elements[0].properties[0].scalar() else { panic!("{back:?}") };
        assert!(
            values[0].is_nan() && values[3].is_nan() && values[1] == f32::INFINITY && values[2] == f32::NEG_INFINITY
        );
        assert!(values[4] == 0.0 && values[4].is_sign_negative());
        if encoding != Encoding::Ascii {
            assert_eq!(values[3].to_bits(), 0x7FC0_1234);
        }
    }
}

/// Parses, extracts the mesh, and checks that a write of what was read reads back to the same bytes.
fn exercise(bytes: &[u8]) {
    let Ok(ply) = Ply::parse(bytes, &Limits::FUZZ) else { return };
    let _ = ply.to_mesh().map(|mesh| mesh.to_ply(Encoding::BinaryLittleEndian));
    if let Ok(written) = write(&ply) {
        let again = Ply::parse(&written, &Limits::FUZZ).expect("written data reads back");
        assert_eq!(write(&again).ok().as_deref(), Some(&written[..]));
    }
}

/// `[ROB-01]` Mutated PLY files in every encoding: no panic, no hang, stable round trips.
#[test]
fn fuzz_ply() {
    let corpus = vec![ASCII.to_vec(), LE.to_vec(), BE.to_vec()];
    humanart_core::fuzz::run("ply", &corpus, 8000, 0x504C_5901, Duration::from_secs(120), exercise);
}
