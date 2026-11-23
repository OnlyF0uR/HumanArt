//! Tests of the `.antmpl` container and section codecs: round trips, corruption, limits, fuzz.

use std::time::Duration;

use super::*;
use crate::template::fixtures::{bare_grid, cube_template};

fn cube() -> Template {
    Template::new(cube_template()).unwrap()
}

fn grid(n: usize) -> Template {
    Template::new(bare_grid(n)).unwrap()
}

#[test]
fn io_21_round_trip_is_byte_identical() {
    for t in [cube(), grid(1), grid(12)] {
        let bytes = write_package(&t).unwrap();
        let back = read_package(&bytes, &PackageLimits::DEFAULT).unwrap();
        assert_eq!(back, t);
        assert_eq!(write_package(&back).unwrap(), bytes, "second write differs");
        // Deterministic: writing the same template twice gives the same bytes.
        assert_eq!(write_package(&t).unwrap(), bytes);
    }
}

#[test]
fn directory_lists_every_section_in_order() {
    let bytes = write_package(&grid(24)).unwrap();
    assert_eq!(&bytes[..4], b"HATP");
    let dir = read_directory(&bytes, &PackageLimits::DEFAULT).unwrap();
    let tags: Vec<[u8; 4]> = dir.iter().map(|e| e.tag).collect();
    assert_eq!(tags, tag::ALL);
    assert!(dir.iter().all(|e| e.required() && e.version == SECTION_VERSION));
    assert!(dir.iter().all(|e| e.offset % SECTION_ALIGN as u64 == 0));
    // The regular grid arrays compress.
    let mesh = dir.iter().find(|e| e.tag == tag::MESH).unwrap();
    assert!(mesh.compressed() && mesh.stored_len < mesh.raw_len, "{mesh:?}");
    let raw = section_payload(&bytes, mesh).unwrap();
    assert_eq!(raw.len() as u64, mesh.raw_len);
    // The metadata section is JSON.
    let meta = section_payload(&bytes, &dir[0]).unwrap();
    assert!(meta.starts_with(br#"{"format":"humanart-template","name":"grid""#), "{}", String::from_utf8_lossy(&meta));
}

#[test]
fn rob_01_every_truncation_is_refused() {
    let bytes = write_package(&cube()).unwrap();
    for len in 0..bytes.len() {
        assert!(read_package(&bytes[..len], &PackageLimits::DEFAULT).is_err(), "prefix of {len} bytes accepted");
    }
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(read_package(&longer, &PackageLimits::DEFAULT).is_err());
}

#[test]
fn rob_01_every_flipped_byte_is_refused() {
    let bytes = write_package(&cube()).unwrap();
    for i in 0..bytes.len() {
        let mut bad = bytes.clone();
        bad[i] ^= 0x10;
        assert!(read_package(&bad, &PackageLimits::DEFAULT).is_err(), "flip at {i} accepted");
    }
}

/// Rebuilds a package from its payloads with one change, so checksums are valid again.
fn reassemble(bytes: &[u8], change: impl Fn(&mut Vec<([u8; 4], u16, Vec<u8>)>)) -> Vec<u8> {
    let dir = read_directory(bytes, &PackageLimits::DEFAULT).unwrap();
    let mut sections: Vec<([u8; 4], u16, Vec<u8>)> =
        dir.iter().map(|e| (e.tag, e.flags & FLAG_REQUIRED, section_payload(bytes, e).unwrap())).collect();
    change(&mut sections);
    let views: Vec<([u8; 4], u16, &[u8])> = sections.iter().map(|(t, f, p)| (*t, *f, p.as_slice())).collect();
    assemble(&views)
}

#[test]
fn io_24_unknown_sections_are_skipped_or_refused() {
    let bytes = write_package(&cube()).unwrap();
    let optional = reassemble(&bytes, |s| s.insert(3, (*b"XTRA", 0, vec![1, 2, 3])));
    assert_eq!(read_package(&optional, &PackageLimits::DEFAULT).unwrap(), cube());
    let required = reassemble(&bytes, |s| s.push((*b"XTRA", FLAG_REQUIRED, vec![1, 2, 3])));
    assert!(read_package(&required, &PackageLimits::DEFAULT).is_err());
    let missing = reassemble(&bytes, |s| drop(s.remove(4)));
    let err = read_package(&missing, &PackageLimits::DEFAULT).unwrap_err();
    assert!(err.to_string().contains("LMRK"), "{err}");
    let twice = reassemble(&bytes, |s| {
        let again = s[1].clone();
        s.push(again);
    });
    assert!(read_package(&twice, &PackageLimits::DEFAULT).is_err());
    // Sections may come in any order.
    let reordered = reassemble(&bytes, |s| s.reverse());
    assert_eq!(read_package(&reordered, &PackageLimits::DEFAULT).unwrap(), cube());
    // Trailing bytes inside a section payload are refused.
    let trailing = reassemble(&bytes, |s| s[2].2.push(0));
    assert!(read_package(&trailing, &PackageLimits::DEFAULT).is_err());
}

#[test]
fn header_errors() {
    let bytes = write_package(&grid(2)).unwrap();
    let patch = |at: usize, v: &[u8]| {
        let mut b = bytes.clone();
        b[at..at + v.len()].copy_from_slice(v);
        // Re-seal the header checksum so only the patched field is wrong.
        let n = u32::from_le_bytes(b[8..12].try_into().unwrap()).min(64) as usize;
        let end = (HEADER_LEN + n * ENTRY_LEN).min(b.len());
        let sum = table_checksum(&b[..24], &b[HEADER_LEN..end]);
        b[24..32].copy_from_slice(&sum.to_le_bytes());
        read_package(&b, &PackageLimits::DEFAULT)
    };
    assert!(patch(0, b"HATQ").is_err());
    assert!(matches!(patch(4, &2_u16.to_le_bytes()), Err(ModelError::Codec(IoError::Unsupported { .. }))));
    assert!(patch(6, &7_u16.to_le_bytes()).is_ok(), "a newer minor version must be read");
    assert!(matches!(patch(8, &65_u32.to_le_bytes()), Err(ModelError::Limit { .. })));
    assert!(patch(12, &1_u32.to_le_bytes()).is_err());
    assert!(patch(16, &1_u64.to_le_bytes()).is_err());
    assert!(patch(HEADER_LEN + 6, &4_u16.to_le_bytes()).is_err()); // unknown section flag
    assert!(patch(HEADER_LEN + 4, &2_u16.to_le_bytes()).is_err()); // section version
    assert!(patch(HEADER_LEN + 8, &48_u64.to_le_bytes()).is_err()); // misplaced section
}

#[test]
fn limits_are_enforced() {
    let bytes = write_package(&grid(8)).unwrap();
    let tight = PackageLimits { max_vertices: 80, ..PackageLimits::FUZZ };
    assert!(matches!(read_package(&bytes, &tight), Err(ModelError::Section { .. })));
    assert!(read_package(&bytes, &PackageLimits { max_vertices: 81, ..PackageLimits::FUZZ }).is_ok());
    let small_file = PackageLimits { max_file_bytes: bytes.len() as u64 - 1, ..PackageLimits::DEFAULT };
    assert!(matches!(read_package(&bytes, &small_file), Err(ModelError::Limit { .. })));
    let small_section = PackageLimits { max_section_bytes: 64, ..PackageLimits::DEFAULT };
    assert!(matches!(read_package(&bytes, &small_section), Err(ModelError::Limit { .. })));
    // A count far beyond the payload is refused before allocating.
    let huge = reassemble(&bytes, |s| s[1].2[..4].copy_from_slice(&0x00FF_FFFF_u32.to_le_bytes()));
    assert!(read_package(&huge, &PackageLimits::DEFAULT).is_err());
}

#[test]
fn rob_01_fuzz_package_bytes() {
    let corpus = vec![write_package(&cube()).unwrap(), write_package(&grid(2)).unwrap()];
    humanart_core::fuzz::run("antmpl", &corpus, 1500, 0x41_4E54_0001, Duration::from_secs(120), |bytes| {
        let _ = read_package(bytes, &PackageLimits::FUZZ);
    });
}

#[test]
fn rob_01_fuzz_section_payloads() {
    // Checksums stop almost every mutation of a whole file at the table, so the section decoders are
    // fuzzed directly: one payload at a time is mutated, the others stay valid.
    let payloads = encode_sections(cube().parts()).unwrap();
    for (s, &tag) in tag::ALL.iter().enumerate() {
        let others = payloads.clone();
        let corpus = vec![payloads[s].clone(), encode_sections(grid(2).parts()).unwrap()[s].clone()];
        let name = format!("antmpl-{}", String::from_utf8_lossy(&tag));
        humanart_core::fuzz::run(
            &name,
            &corpus,
            2000,
            0x41_4E54_0100 + s as u64,
            Duration::from_secs(120),
            move |bytes| {
                let mut views: Vec<&[u8]> = others.iter().map(Vec::as_slice).collect();
                views[s] = bytes;
                if let Ok(t) = decode_template(&views, Vec::new(), &PackageLimits::FUZZ) {
                    // Whatever is accepted is a consistent template that writes and reads back.
                    let again = write_package(&t).unwrap();
                    assert_eq!(read_package(&again, &PackageLimits::FUZZ).unwrap(), t);
                }
            },
        );
    }
}

/// `[ID-07]` `[AC-WP06-3]`: a template with no scan data at all (only authored semantic axes,
/// expression and corrective deltas) loads from its package through the normal path and functions
/// fully: it validates, its semantic axes deform the mesh and move the regressed joints, its
/// expressions, combination and pose correctives evaluate, and its LODs follow the deformation.
#[test]
fn id_07_no_data_template_loads_and_functions() {
    use crate::delta::DeltaKind;
    use humanart_math::{Quat, Vec3};

    let t = read_package(&write_package(&cube()).unwrap(), &PackageLimits::DEFAULT).unwrap();
    // No scan data of any kind.
    assert_eq!(t.scan_component_count(), 0);
    for kind in [DeltaKind::IdentityGlobal, DeltaKind::IdentityRegional, DeltaKind::Asymmetry] {
        assert_eq!(t.deltas().count_kind(kind), 0);
    }
    assert!(t.deltas().components().iter().all(|c| !c.kind.is_scan_derived()));
    assert_eq!(t.deltas().count_kind(DeltaKind::Semantic), 2);
    // It passes validation.
    let report = crate::validate::validate(&t);
    assert!(report.passed(), "{report}");
    assert_eq!(report.deltas.scan_derived, 0);

    // An authored semantic axis deforms the mesh and moves the regressed joints.
    let d = t.deltas();
    let c = |name: &str| d.find(name).unwrap() as usize;
    let mut w = vec![0.0; d.len()];
    w[c("height")] = 1.0;
    let p = t.deformed_positions(&w);
    for (a, b) in t.mesh().positions().iter().zip(&p) {
        let lift = if a.y > 0.5 { 0.25 } else { 0.0 };
        assert!((*b - *a - Vec3::Y * lift).length() < 1e-6, "{a:?} -> {b:?}");
    }
    let rest = t.skeleton().regressed_positions(&[]);
    let moved = t.skeleton().regressed_positions(&w);
    assert!((moved[1] - rest[1] - Vec3::Y * 0.25).length() < 1e-6);
    assert_eq!(moved[0], rest[0]);

    // Expressions drive their components; the combination and pose correctives activate.
    let e = t.expressions();
    let expression_weights = [0.8_f32, 0.5];
    for (x, &ew) in e.expressions.iter().zip(&expression_weights) {
        w[x.component as usize] = ew;
    }
    let combo = &e.combinations[0];
    assert_eq!(combo.activation(&expression_weights), 0.5);
    w[combo.component as usize] = combo.activation(&expression_weights);
    let pose = &e.pose_correctives[0];
    assert!((pose.activation(pose.target) - 1.0).abs() < 1e-5);
    let half = pose.activation(Quat::from_axis_angle(Vec3::X, 0.5));
    assert!((half - 0.5625).abs() < 1e-4, "{half}");
    assert_eq!(pose.activation(Quat::IDENTITY), 0.0);
    w[pose.component as usize] = half;
    let posed = t.deformed_positions(&w);
    let mut expect = t.mesh().positions().to_vec();
    for (i, &wi) in w.iter().enumerate() {
        d.accumulate(i, wi, &mut expect).unwrap();
    }
    assert_eq!(posed, expect);
    assert_ne!(posed, p);

    // The LOD follows the deformed LOD0 through its barycentric sources.
    let lod = t.lods()[0].transfer_positions(&p).unwrap();
    for q in &lod {
        assert_eq!(q.y, if q.y > 0.5 { 1.25 } else { 0.0 });
    }
}
