//! MTL material libraries: the basics that OBJ exporters and scanning tools write `[IO-11]`.
//!
//! Read: `newmtl`, the colours `Ka` `Kd` `Ks` `Ke` `Tf` (three values, or one for grey), `Ns`, `Ni`,
//! `d`, `Tr` (as `d = 1 − Tr`), `illum`, the PBR extension `Pr` / `Pm`, and the texture maps
//! `map_Ka` `map_Kd` `map_Ks` `map_Ns` `map_d` `map_Ke` `map_Pr` `map_Pm`, `map_Bump` / `map_bump` /
//! `bump`, `norm` and `disp`. Of the texture options only `-bm` (bump multiplier) is kept; the
//! others (`-blendu` `-blendv` `-boost` `-cc` `-clamp` `-imfchan` `-mm` `-o` `-s` `-t` `-texres`
//! `-type`) are parsed and dropped. Other statements made of a plain word are skipped; spectral
//! (`spectral`) and CIE XYZ (`xyz`) colours are [`Error::Unsupported`]. Line continuations are not
//! part of MTL and are not read.

use std::fmt::Write as _;

use super::{FORMAT, invalid, name, number_tokens, parse_f32, parse_int, trim};
use crate::error::{Error, Result};
use crate::image::Limits;

/// A texture map of a material.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextureMap {
    /// The image file, relative to the MTL file, as written (it may contain spaces).
    pub file: String,
    /// The bump multiplier (`-bm`), for bump and normal maps.
    pub bump_multiplier: Option<f32>,
}

/// One material of an MTL library. Every property is optional: `None` when the library does not set
/// it. Colours are as written (usually sRGB-encoded values in 0..1).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Material {
    /// The name `usemtl` refers to.
    pub name: String,
    /// The ambient colour (`Ka`).
    pub ambient: Option<[f32; 3]>,
    /// The diffuse colour (`Kd`).
    pub diffuse: Option<[f32; 3]>,
    /// The specular colour (`Ks`).
    pub specular: Option<[f32; 3]>,
    /// The emissive colour (`Ke`).
    pub emissive: Option<[f32; 3]>,
    /// The transmission filter (`Tf`).
    pub transmission_filter: Option<[f32; 3]>,
    /// The specular exponent (`Ns`).
    pub shininess: Option<f32>,
    /// The index of refraction (`Ni`).
    pub ior: Option<f32>,
    /// The opacity (`d`, or `1 − Tr`), 1 opaque.
    pub dissolve: Option<f32>,
    /// The illumination model (`illum`).
    pub illumination: Option<u32>,
    /// The PBR roughness (`Pr`).
    pub roughness: Option<f32>,
    /// The PBR metalness (`Pm`).
    pub metallic: Option<f32>,
    /// `map_Ka`.
    pub ambient_map: Option<TextureMap>,
    /// `map_Kd`.
    pub diffuse_map: Option<TextureMap>,
    /// `map_Ks`.
    pub specular_map: Option<TextureMap>,
    /// `map_Ns`.
    pub shininess_map: Option<TextureMap>,
    /// `map_d`.
    pub dissolve_map: Option<TextureMap>,
    /// `map_Ke`.
    pub emissive_map: Option<TextureMap>,
    /// `map_Pr`.
    pub roughness_map: Option<TextureMap>,
    /// `map_Pm`.
    pub metallic_map: Option<TextureMap>,
    /// `map_Bump`, `map_bump` or `bump`.
    pub bump_map: Option<TextureMap>,
    /// `norm`.
    pub normal_map: Option<TextureMap>,
    /// `disp`.
    pub displacement_map: Option<TextureMap>,
}

/// The first whitespace-separated token of `s` and the trimmed rest.
fn split_token(s: &[u8]) -> (&[u8], &[u8]) {
    let s = trim(s);
    let end = s.iter().position(u8::is_ascii_whitespace).unwrap_or(s.len());
    let (token, rest) = s.split_at(end);
    (token, trim(rest))
}

/// A colour statement: three values, or one for grey.
fn color(rest: &[u8], at: usize) -> Result<[f32; 3]> {
    if let Some(first) = number_tokens(rest).next()
        && (first == b"spectral" || first == b"xyz")
    {
        return Err(Error::Unsupported { format: FORMAT, what: "spectral and CIE XYZ material colours" });
    }
    let mut values = [0.0_f32; 3];
    let mut count = 0;
    for token in number_tokens(rest) {
        let slot = values.get_mut(count).ok_or_else(|| invalid(at, "a colour has 1 or 3 values"))?;
        *slot = parse_f32(token, at)?;
        count += 1;
    }
    match count {
        1 => Ok([values[0]; 3]),
        3 => Ok(values),
        _ => Err(invalid(at, "a colour has 1 or 3 values")),
    }
}

/// A statement of exactly one number.
fn scalar(rest: &[u8], at: usize) -> Result<f32> {
    let mut tokens = number_tokens(rest);
    let value = parse_f32(tokens.next().ok_or_else(|| invalid(at, "a value is missing"))?, at)?;
    if tokens.next().is_some() {
        return Err(invalid(at, "one value expected"));
    }
    Ok(value)
}

/// A texture map statement: options, then the file name (the rest of the line).
fn texture(rest: &[u8], at: usize) -> Result<TextureMap> {
    let mut map = TextureMap::default();
    let mut rest = trim(rest);
    while rest.first() == Some(&b'-') {
        let (option, after) = split_token(rest);
        rest = after;
        // How many arguments the option takes: a fixed count, or (for -o -s -t) one to three numbers.
        let (min, max) = match option {
            b"-blendu" | b"-blendv" | b"-cc" | b"-clamp" | b"-boost" | b"-texres" | b"-imfchan" | b"-type" | b"-bm" => {
                (1, 1)
            }
            b"-mm" => (2, 2),
            b"-o" | b"-s" | b"-t" => (1, 3),
            _ => return Err(invalid(at, "unknown texture option")),
        };
        for i in 0..max {
            let (argument, after) = split_token(rest);
            if argument.is_empty() {
                return Err(invalid(at, "texture option without its value"));
            }
            if i >= min && parse_f32(argument, at).is_err() {
                break;
            }
            if option == b"-bm" {
                map.bump_multiplier = Some(parse_f32(argument, at)?);
            }
            rest = after;
        }
    }
    map.file = name(rest, at)?;
    if map.file.is_empty() {
        return Err(invalid(at, "texture map without a file"));
    }
    Ok(map)
}

/// Parses an MTL library. At most [`Limits::max_elements`] materials are read.
pub fn parse_mtl(data: &[u8], limits: &Limits) -> Result<Vec<Material>> {
    let mut materials: Vec<Material> = Vec::new();
    let mut at = 0;
    for raw in data.split(|&b| b == b'\n') {
        let line_at = at;
        at += raw.len() + 1;
        let line = if line_at == 0 { raw.strip_prefix(super::BOM).unwrap_or(raw) } else { raw };
        let (keyword, rest) = split_token(line);
        if keyword.is_empty() || keyword.first() == Some(&b'#') {
            continue;
        }
        if keyword == b"newmtl" {
            if materials.len() as u64 >= limits.max_elements {
                return Err(Error::Limit {
                    format: FORMAT,
                    what: "material count",
                    limit: limits.max_elements,
                    got: materials.len() as u64 + 1,
                });
            }
            let name = name(rest, line_at)?;
            if name.is_empty() {
                return Err(invalid(line_at, "material without a name"));
            }
            materials.push(Material { name, ..Material::default() });
            continue;
        }
        let known = matches!(
            keyword,
            b"Ka"
                | b"Kd"
                | b"Ks"
                | b"Ke"
                | b"Tf"
                | b"Ns"
                | b"Ni"
                | b"d"
                | b"Tr"
                | b"illum"
                | b"Pr"
                | b"Pm"
                | b"map_Ka"
                | b"map_Kd"
                | b"map_Ks"
                | b"map_Ns"
                | b"map_d"
                | b"map_Ke"
                | b"map_Pr"
                | b"map_Pm"
                | b"map_Bump"
                | b"map_bump"
                | b"bump"
                | b"norm"
                | b"disp"
        );
        if !known {
            if keyword.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'_') {
                continue;
            }
            return Err(invalid(line_at, "not an MTL statement"));
        }
        let Some(m) = materials.last_mut() else {
            return Err(invalid(line_at, "material property before newmtl"));
        };
        match keyword {
            b"Ka" => m.ambient = Some(color(rest, line_at)?),
            b"Kd" => m.diffuse = Some(color(rest, line_at)?),
            b"Ks" => m.specular = Some(color(rest, line_at)?),
            b"Ke" => m.emissive = Some(color(rest, line_at)?),
            b"Tf" => m.transmission_filter = Some(color(rest, line_at)?),
            b"Ns" => m.shininess = Some(scalar(rest, line_at)?),
            b"Ni" => m.ior = Some(scalar(rest, line_at)?),
            b"d" => {
                // `d -halo 0.5` is a dissolve that depends on the view: its factor is what is kept.
                let rest = rest.strip_prefix(b"-halo").unwrap_or(rest);
                m.dissolve = Some(scalar(rest, line_at)?);
            }
            b"Tr" => m.dissolve = Some(1.0 - scalar(rest, line_at)?),
            b"illum" => {
                let token = number_tokens(rest).next().ok_or_else(|| invalid(line_at, "a value is missing"))?;
                m.illumination = Some(
                    u32::try_from(parse_int(token, line_at)?)
                        .map_err(|_| invalid(line_at, "bad illumination model"))?,
                );
            }
            b"Pr" => m.roughness = Some(scalar(rest, line_at)?),
            b"Pm" => m.metallic = Some(scalar(rest, line_at)?),
            b"map_Ka" => m.ambient_map = Some(texture(rest, line_at)?),
            b"map_Kd" => m.diffuse_map = Some(texture(rest, line_at)?),
            b"map_Ks" => m.specular_map = Some(texture(rest, line_at)?),
            b"map_Ns" => m.shininess_map = Some(texture(rest, line_at)?),
            b"map_d" => m.dissolve_map = Some(texture(rest, line_at)?),
            b"map_Ke" => m.emissive_map = Some(texture(rest, line_at)?),
            b"map_Pr" => m.roughness_map = Some(texture(rest, line_at)?),
            b"map_Pm" => m.metallic_map = Some(texture(rest, line_at)?),
            b"norm" => m.normal_map = Some(texture(rest, line_at)?),
            b"disp" => m.displacement_map = Some(texture(rest, line_at)?),
            // map_Bump, map_bump, bump: the only keywords left.
            _ => m.bump_map = Some(texture(rest, line_at)?),
        }
    }
    Ok(materials)
}

/// Checks that a name or file name survives a write and a read unchanged.
fn check_text(text: &str, what: &'static str) -> Result<()> {
    let bad = text.is_empty()
        || text.contains(['\n', '\r', '\0'])
        || text.trim_matches(|c: char| c.is_ascii_whitespace()).len() != text.len();
    if bad { Err(Error::Encode { format: FORMAT, what }) } else { Ok(()) }
}

/// Writes materials as an MTL library, every property that is set, in the order of [`Material`]'s
/// fields. A write followed by [`parse_mtl`] gives the same materials. Errors: an empty name or file
/// name, one with line breaks, NUL or surrounding whitespace, a file name starting with `-` (it
/// would read as an option), a value that is not finite.
pub fn write_mtl(materials: &[Material]) -> Result<Vec<u8>> {
    let mut out = String::from("# HumanArt MTL\n");
    for m in materials {
        check_text(&m.name, "a material name cannot be written as MTL")?;
        let _ = writeln!(out, "newmtl {}", m.name);
        let colors = [
            ("Ka", m.ambient),
            ("Kd", m.diffuse),
            ("Ks", m.specular),
            ("Ke", m.emissive),
            ("Tf", m.transmission_filter),
        ];
        // Writing to a String cannot fail; the results of write! are discarded for that reason.
        for (keyword, value) in colors {
            if let Some(c) = value {
                if !c.iter().all(|v| v.is_finite()) {
                    return Err(Error::Encode { format: FORMAT, what: "a material value is not finite" });
                }
                let _ = writeln!(out, "{keyword} {} {} {}", c[0], c[1], c[2]);
            }
        }
        let scalars = [("Ns", m.shininess), ("Ni", m.ior), ("d", m.dissolve), ("Pr", m.roughness), ("Pm", m.metallic)];
        for (keyword, value) in scalars {
            if let Some(v) = value {
                if !v.is_finite() {
                    return Err(Error::Encode { format: FORMAT, what: "a material value is not finite" });
                }
                let _ = writeln!(out, "{keyword} {v}");
            }
        }
        if let Some(illum) = m.illumination {
            let _ = writeln!(out, "illum {illum}");
        }
        let maps = [
            ("map_Ka", &m.ambient_map),
            ("map_Kd", &m.diffuse_map),
            ("map_Ks", &m.specular_map),
            ("map_Ns", &m.shininess_map),
            ("map_d", &m.dissolve_map),
            ("map_Ke", &m.emissive_map),
            ("map_Pr", &m.roughness_map),
            ("map_Pm", &m.metallic_map),
            ("map_Bump", &m.bump_map),
            ("norm", &m.normal_map),
            ("disp", &m.displacement_map),
        ];
        for (keyword, map) in maps {
            let Some(map) = map else { continue };
            check_text(&map.file, "a texture file name cannot be written as MTL")?;
            if map.file.starts_with('-') {
                return Err(Error::Encode { format: FORMAT, what: "a texture file name starts with '-'" });
            }
            out.push_str(keyword);
            if let Some(bm) = map.bump_multiplier {
                if !bm.is_finite() {
                    return Err(Error::Encode { format: FORMAT, what: "a material value is not finite" });
                }
                let _ = write!(out, " -bm {bm}");
            }
            let _ = writeln!(out, " {}", map.file);
        }
    }
    Ok(out.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const LIBRARY: &str = "# exported\r\n\
        newmtl skin\r\n\
        Ka 0.1\r\n\
        Kd 0.8 0.6 0.5\r\n\
        Ks 0.04 0.04 0.04\r\n\
        Ns 250\r\n\
        Ni 1.4\r\n\
        Tr 0.25\r\n\
        illum 2\r\n\
        Pr 0.45\r\n\
        map_Kd -blendu on -o 0.5 0.5 -s 2 skin albedo.png\r\n\
        map_Bump -bm 0.8 -imfchan l skin_bump.png\r\n\
        norm skin_normal.png\r\n\
        sharpness 60\r\n\
        \r\n\
        newmtl eye\n\
        Kd 1 1 1\n\
        d -halo 0.9\n\
        Pm 0\n\
        map_Pr -mm 0 1 rough.png\n";

    #[test]
    fn reads_the_basics_with_texture_options() {
        let m = parse_mtl(LIBRARY.as_bytes(), &Limits::DEFAULT).unwrap();
        assert_eq!(m.len(), 2);
        let skin = &m[0];
        assert_eq!(skin.name, "skin");
        assert_eq!(skin.ambient, Some([0.1; 3]));
        assert_eq!(skin.diffuse, Some([0.8, 0.6, 0.5]));
        assert_eq!(skin.specular, Some([0.04; 3]));
        assert_eq!((skin.shininess, skin.ior, skin.dissolve), (Some(250.0), Some(1.4), Some(0.75)));
        assert_eq!((skin.illumination, skin.roughness, skin.metallic), (Some(2), Some(0.45), None));
        assert_eq!(skin.diffuse_map, Some(TextureMap { file: "skin albedo.png".into(), bump_multiplier: None }));
        assert_eq!(skin.bump_map, Some(TextureMap { file: "skin_bump.png".into(), bump_multiplier: Some(0.8) }));
        assert_eq!(skin.normal_map.as_ref().map(|t| t.file.as_str()), Some("skin_normal.png"));
        let eye = &m[1];
        assert_eq!((eye.dissolve, eye.metallic), (Some(0.9), Some(0.0)));
        assert_eq!(eye.roughness_map.as_ref().map(|t| t.file.as_str()), Some("rough.png"));
        assert_eq!(eye.diffuse_map, None);
    }

    #[test]
    fn write_then_read_gives_the_same_materials() {
        let m = parse_mtl(LIBRARY.as_bytes(), &Limits::DEFAULT).unwrap();
        let text = write_mtl(&m).unwrap();
        assert_eq!(parse_mtl(&text, &Limits::DEFAULT).unwrap(), m);
        let every = Material {
            name: "all of it".into(),
            ambient: Some([0.0, 0.5, 1.0]),
            diffuse: Some([0.1, 0.2, 0.3]),
            specular: Some([1.0e-7, 2.0, 3.5]),
            emissive: Some([4.0, 5.0, 6.0]),
            transmission_filter: Some([0.25; 3]),
            shininess: Some(96.078_43),
            ior: Some(1.45),
            dissolve: Some(0.5),
            illumination: Some(7),
            roughness: Some(0.3),
            metallic: Some(1.0),
            ambient_map: Some(TextureMap { file: "a.png".into(), bump_multiplier: None }),
            diffuse_map: Some(TextureMap { file: "dir/b c.png".into(), bump_multiplier: None }),
            specular_map: Some(TextureMap { file: "c.png".into(), bump_multiplier: None }),
            shininess_map: Some(TextureMap { file: "d.png".into(), bump_multiplier: None }),
            dissolve_map: Some(TextureMap { file: "e.png".into(), bump_multiplier: None }),
            emissive_map: Some(TextureMap { file: "f.png".into(), bump_multiplier: None }),
            roughness_map: Some(TextureMap { file: "g.png".into(), bump_multiplier: None }),
            metallic_map: Some(TextureMap { file: "h.png".into(), bump_multiplier: None }),
            bump_map: Some(TextureMap { file: "i.png".into(), bump_multiplier: Some(-0.5) }),
            normal_map: Some(TextureMap { file: "j.png".into(), bump_multiplier: Some(2.0) }),
            displacement_map: Some(TextureMap { file: "k.png".into(), bump_multiplier: None }),
        };
        let list = vec![every, Material { name: "bare".into(), ..Material::default() }];
        assert_eq!(parse_mtl(&write_mtl(&list).unwrap(), &Limits::DEFAULT).unwrap(), list);
    }

    #[test]
    fn errors_and_what_cannot_be_written() {
        let parse = |text: &str| parse_mtl(text.as_bytes(), &Limits::DEFAULT);
        assert!(matches!(parse("Kd 1 1 1\n"), Err(Error::Invalid { offset: 0, .. })));
        assert!(matches!(parse("newmtl a\nKd 1 1\n"), Err(Error::Invalid { offset: 9, .. })));
        assert!(matches!(parse("newmtl a\nKd spectral x.rfl\n"), Err(Error::Unsupported { .. })));
        assert!(matches!(parse("newmtl a\nNs inf\n"), Err(Error::Invalid { .. })));
        assert!(matches!(parse("newmtl a\nmap_Kd -zz 1 x.png\n"), Err(Error::Invalid { .. })));
        assert!(matches!(parse("newmtl a\nmap_Kd -bm\n"), Err(Error::Invalid { .. })));
        assert!(matches!(parse("newmtl a\nmap_Kd -o 1 2 3\n"), Err(Error::Invalid { .. })), "no file left");
        assert!(matches!(parse("newmtl\n"), Err(Error::Invalid { .. })));
        assert!(matches!(parse("newmtl a\n\u{1}x 1\n"), Err(Error::Invalid { .. })));
        assert!(matches!(parse("newmtl a\nillum -1\n"), Err(Error::Invalid { .. })));
        let limits = Limits { max_elements: 2, ..Limits::DEFAULT };
        assert!(matches!(parse_mtl(b"newmtl a\nnewmtl b\nnewmtl c\n", &limits), Err(Error::Limit { .. })));
        assert_eq!(parse("\u{FEFF}newmtl a\n").unwrap().len(), 1);

        let bad = |m: Material| assert!(matches!(write_mtl(&[m]), Err(Error::Encode { .. })));
        bad(Material::default());
        bad(Material { name: " x".into(), ..Material::default() });
        bad(Material { name: "x".into(), shininess: Some(f32::NAN), ..Material::default() });
        bad(Material { name: "x".into(), diffuse: Some([0.0, f32::INFINITY, 0.0]), ..Material::default() });
        let map = |file: &str, bm| Some(TextureMap { file: file.into(), bump_multiplier: bm });
        bad(Material { name: "x".into(), diffuse_map: map("-x.png", None), ..Material::default() });
        bad(Material { name: "x".into(), diffuse_map: map("a\nb", None), ..Material::default() });
        bad(Material { name: "x".into(), bump_map: map("a.png", Some(f32::NAN)), ..Material::default() });
    }

    /// `[ROB-01]` Mutated libraries: no panic, no hang.
    #[test]
    fn fuzz_mtl() {
        let corpus = vec![LIBRARY.as_bytes().to_vec(), b"newmtl a\nKd 1 0 0\nmap_Kd -s 1 1 1 -bm 2 t.png\n".to_vec()];
        humanart_core::fuzz::run("mtl", &corpus, 20_000, 0x4D54_4C01, Duration::from_secs(120), |bytes| {
            if let Ok(materials) = parse_mtl(bytes, &Limits::FUZZ)
                && let Ok(text) = write_mtl(&materials)
            {
                assert_eq!(parse_mtl(&text, &Limits::FUZZ).ok().as_ref(), Some(&materials));
            }
        });
    }
}
