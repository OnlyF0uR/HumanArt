//! The OBJ writer.

use std::fmt::Write as _;

use super::{FORMAT, FaceGroup, Obj, flip_v};
use crate::error::{Error, Result};

/// Checks that a name survives a write and a read unchanged: no line breaks, no NUL, no surrounding
/// whitespace and no trailing `\` (which would continue the line).
fn check_name(name: &str) -> Result<()> {
    let bad = name.contains(['\n', '\r', '\0'])
        || name.ends_with('\\')
        || name.trim_matches(|c: char| c.is_ascii_whitespace()).len() != name.len();
    if bad { Err(Error::Encode { format: FORMAT, what: "a name cannot be written as OBJ" }) } else { Ok(()) }
}

/// The value to write for a stored (V down) texture coordinate `v`: the `f32` whose flip reads back
/// as exactly `v`, when one exists, else the nearest.
fn written_v(v: f32) -> f32 {
    let w = flip_v(v);
    [w, w.next_up(), w.next_down()].into_iter().find(|&c| flip_v(c) == v).unwrap_or(w)
}

/// Writes `obj` as OBJ text: `mtllib`, positions (with colours when [`Obj::colors`] is set),
/// texture coordinates (V flipped back to OBJ's convention), normals, the faces with the `o` / `g` /
/// `usemtl` / `s` statements their runs need, then the polylines. Numbers are written in the
/// shortest form that reads back as the same `f32`, so a write followed by [`Obj::parse`] gives the
/// same arrays. Texture coordinates come back exactly for V in 0.5..=1 and for any V in 0..=1 that
/// is a multiple of 2^-24; elsewhere V can have finer steps than `1 − V` and comes back within
/// 4 · `f32::EPSILON`.
///
/// Faces that no run covers are written with the state of the run before them. Errors: the checks
/// of [`Obj::validate`], a name that cannot be written (line breaks, NUL, surrounding whitespace, a
/// trailing `\`), `Some("")` as a material and a material library name that is empty or contains
/// whitespace.
pub fn write(obj: &Obj) -> Result<Vec<u8>> {
    obj.validate()?;
    for group in &obj.groups {
        check_name(&group.object)?;
        check_name(&group.group)?;
        if let Some(material) = &group.material {
            check_name(material)?;
            if material.is_empty() {
                return Err(Error::Encode { format: FORMAT, what: "a material name is empty" });
            }
        }
    }
    if obj
        .material_libraries
        .iter()
        .any(|lib| lib.is_empty() || lib.contains(char::is_whitespace) || check_name(lib).is_err())
    {
        return Err(Error::Encode { format: FORMAT, what: "a material library name is empty or contains whitespace" });
    }

    let mut out = String::with_capacity(32 * (obj.positions.len() + obj.corners.len()) + 64);
    out.push_str("# HumanArt OBJ\n");
    if !obj.material_libraries.is_empty() {
        out.push_str("mtllib");
        for lib in &obj.material_libraries {
            out.push(' ');
            out.push_str(lib);
        }
        out.push('\n');
    }
    // Writing to a String cannot fail; the results of write! are discarded for that reason.
    for (i, p) in obj.positions.iter().enumerate() {
        let _ = write!(out, "v {} {} {}", p[0], p[1], p[2]);
        if let Some(&[r, g, b]) = obj.colors.get(i) {
            let _ = write!(out, " {r} {g} {b}");
        }
        out.push('\n');
    }
    for t in &obj.tex_coords {
        let _ = writeln!(out, "vt {} {}", t[0], written_v(t[1]));
    }
    for n in &obj.normals {
        let _ = writeln!(out, "vn {} {} {}", n[0], n[1], n[2]);
    }

    // The state a reader has after the statements written so far.
    let mut state = FaceGroup::default();
    let mut groups = obj.groups.iter().peekable();
    for (index, face) in obj.faces().enumerate() {
        while let Some(group) = groups.next_if(|g| g.faces.start as usize <= index) {
            if group.faces.is_empty() {
                continue;
            }
            if group.object != state.object {
                let _ = writeln!(out, "o {}", group.object);
                state.object.clone_from(&group.object);
                state.group.clear();
            }
            if group.group != state.group {
                let _ = writeln!(out, "g {}", group.group);
                state.group.clone_from(&group.group);
            }
            if group.material != state.material {
                let _ = writeln!(out, "usemtl {}", group.material.as_deref().unwrap_or(""));
                state.material.clone_from(&group.material);
            }
            if group.smoothing != state.smoothing {
                if group.smoothing == 0 {
                    out.push_str("s off\n");
                } else {
                    let _ = writeln!(out, "s {}", group.smoothing);
                }
                state.smoothing = group.smoothing;
            }
        }
        out.push('f');
        for c in face {
            let _ = write!(out, " {}", c.position + 1);
            match (c.tex_coord(), c.normal()) {
                (None, None) => {}
                (Some(t), None) => {
                    let _ = write!(out, "/{}", t + 1);
                }
                (None, Some(n)) => {
                    let _ = write!(out, "//{}", n + 1);
                }
                (Some(t), Some(n)) => {
                    let _ = write!(out, "/{}/{}", t + 1, n + 1);
                }
            }
        }
        out.push('\n');
    }
    for index in 0..obj.line_count() {
        out.push('l');
        for v in obj.line(index).unwrap_or(&[]) {
            let _ = write!(out, " {}", v + 1);
        }
        out.push('\n');
    }
    Ok(out.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Limits;
    use crate::obj::Corner;

    #[test]
    fn names_that_would_not_read_back_are_refused() {
        for bad in ["a\nb", "a\r", "x\\", " lead", "trail ", "nul\0"] {
            assert!(check_name(bad).is_err(), "{bad:?}");
        }
        assert!(check_name("Body Mesh.001").is_ok());
        assert!(check_name("").is_ok());
        let mut obj = Obj { positions: vec![[0.0; 3]; 3], ..Obj::default() };
        obj.corners = (0..3).map(|i| Corner::new(i, None, None)).collect();
        obj.face_ends = vec![3];
        obj.groups = vec![FaceGroup { material: Some(String::new()), faces: 0..1, ..FaceGroup::default() }];
        assert!(matches!(write(&obj), Err(Error::Encode { .. })));
        obj.groups[0].material = Some("skin".into());
        obj.material_libraries = vec!["my lib.mtl".into()];
        assert!(matches!(write(&obj), Err(Error::Encode { .. })));
        obj.material_libraries = vec!["lib.mtl".into()];
        let text = String::from_utf8(write(&obj).unwrap()).unwrap();
        assert_eq!(text, "# HumanArt OBJ\nmtllib lib.mtl\nv 0 0 0\nv 0 0 0\nv 0 0 0\nusemtl skin\nf 1 2 3\n");
        assert_eq!(Obj::parse(text.as_bytes(), &Limits::DEFAULT).unwrap(), obj);
    }

    #[test]
    fn texture_v_round_trips_through_the_flip() {
        let mut rng = humanart_core::rng::Rng::new(7);
        for _ in 0..20_000 {
            let v = rng.range_f32(-2.0, 3.0);
            let w = written_v(v);
            let back: f32 = w.to_string().parse().unwrap();
            assert_eq!(back, w, "the shortest form reads back exactly");
            let read = flip_v(back);
            // From 0.5 to 1 every V has an exact partner; elsewhere V can have finer steps than 1 − V.
            if (0.5..=1.0).contains(&v) {
                assert_eq!(read, v);
            } else {
                assert!((read - v).abs() <= 4.0 * f32::EPSILON, "{v} -> {w} -> {read}");
            }
        }
        for v in [0.0_f32, 1.0, 0.5, 0.25, 0.125, 2.0, -1.0] {
            assert_eq!(flip_v(written_v(v)), v);
        }
    }
}
