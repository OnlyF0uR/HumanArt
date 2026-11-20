//! The PLY writer.

use std::fmt::Write as _;

use super::{Column, Encoding, FORMAT, Ply, PropertyData, ScalarType};
use crate::error::{Error, Result};

/// An [`Error::Encode`] of this module.
const fn refuse(what: &'static str) -> Error {
    Error::Encode { format: FORMAT, what }
}

/// Checks that an element or property name is one header word.
fn check_name(name: &str) -> Result<()> {
    if name.is_empty() || name.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err(refuse("a name is empty or contains whitespace"));
    }
    Ok(())
}

/// The largest list length a count type can hold.
const fn max_count(ty: ScalarType) -> u64 {
    match ty {
        ScalarType::I8 => i8::MAX as u64,
        ScalarType::U8 => u8::MAX as u64,
        ScalarType::I16 => i16::MAX as u64,
        ScalarType::U16 => u16::MAX as u64,
        ScalarType::I32 => i32::MAX as u64,
        ScalarType::U32 => u32::MAX as u64,
        ScalarType::F32 | ScalarType::F64 => 0,
    }
}

/// Checks that the data matches the declared shape: every column `count` long, list ends that never
/// decrease and end at the item count, list lengths that fit their count type.
fn validate(ply: &Ply) -> Result<()> {
    for text in ply.comments.iter().chain(&ply.obj_info) {
        if text.contains(['\n', '\r']) {
            return Err(refuse("a header comment contains a line break"));
        }
    }
    for element in &ply.elements {
        check_name(&element.name)?;
        for property in &element.properties {
            check_name(&property.name)?;
            match &property.data {
                PropertyData::Scalar(column) => {
                    if column.len() != element.count {
                        return Err(refuse("a column's length differs from its element's row count"));
                    }
                }
                PropertyData::List { count_type, ends, values } => {
                    if !count_type.is_integer() {
                        return Err(refuse("a list count type is not an integer type"));
                    }
                    if ends.len() != element.count {
                        return Err(refuse("a list's row count differs from its element's"));
                    }
                    let mut previous = 0_u32;
                    for &end in ends {
                        if end < previous || u64::from(end - previous) > max_count(*count_type) {
                            return Err(refuse("a list's ends decrease or a length does not fit its count type"));
                        }
                        previous = end;
                    }
                    if previous as usize != values.len() {
                        return Err(refuse("a list's last end differs from its item count"));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Appends one value: decimal text and a separator, or its bytes in the encoding's order.
macro_rules! emit {
    ($out:expr, $x:expr, $encoding:expr) => {{
        let x = $x;
        match $encoding {
            Encoding::Ascii => {
                // Writing to a Vec cannot fail.
                let _ = std::io::Write::write_fmt($out, format_args!("{x} "));
            }
            Encoding::BinaryLittleEndian => $out.extend_from_slice(&x.to_le_bytes()),
            Encoding::BinaryBigEndian => $out.extend_from_slice(&x.to_be_bytes()),
        }
    }};
}

/// Appends value `i` of a column (which [`validate`] has checked to be long enough).
fn put(out: &mut Vec<u8>, column: &Column, i: usize, encoding: Encoding) {
    match column {
        Column::I8(v) => emit!(out, v.get(i).copied().unwrap_or_default(), encoding),
        Column::U8(v) => emit!(out, v.get(i).copied().unwrap_or_default(), encoding),
        Column::I16(v) => emit!(out, v.get(i).copied().unwrap_or_default(), encoding),
        Column::U16(v) => emit!(out, v.get(i).copied().unwrap_or_default(), encoding),
        Column::I32(v) => emit!(out, v.get(i).copied().unwrap_or_default(), encoding),
        Column::U32(v) => emit!(out, v.get(i).copied().unwrap_or_default(), encoding),
        Column::F32(v) => emit!(out, v.get(i).copied().unwrap_or_default(), encoding),
        Column::F64(v) => emit!(out, v.get(i).copied().unwrap_or_default(), encoding),
    }
}

/// Appends a list length in its count type (which [`validate`] has checked it fits).
fn put_count(out: &mut Vec<u8>, ty: ScalarType, len: u32, encoding: Encoding) {
    match ty {
        ScalarType::I8 => emit!(out, len as i8, encoding),
        ScalarType::U8 => emit!(out, len as u8, encoding),
        ScalarType::I16 => emit!(out, len as i16, encoding),
        ScalarType::U16 => emit!(out, len as u16, encoding),
        ScalarType::I32 => emit!(out, len as i32, encoding),
        ScalarType::U32 | ScalarType::F32 | ScalarType::F64 => emit!(out, len, encoding),
    }
}

/// Writes a PLY file in its [`Ply::encoding`]. Every value is written in its declared type; ASCII
/// numbers in the shortest form that reads back as the same value, one row per line. A write
/// followed by [`Ply::parse`] gives the same file contents (bit for bit in binary; in ASCII every
/// value but NaN compares equal, and NaN stays NaN).
///
/// Errors: names that are empty or contain whitespace, header text with line breaks, and data whose
/// shape differs from the declaration (column lengths, list ends, list lengths over their count
/// type, a float count type).
pub fn write(ply: &Ply) -> Result<Vec<u8>> {
    validate(ply)?;
    let mut header = String::from("ply\n");
    let _ = writeln!(header, "format {} 1.0", ply.encoding.name());
    for comment in &ply.comments {
        let _ = writeln!(header, "comment {comment}");
    }
    for info in &ply.obj_info {
        let _ = writeln!(header, "obj_info {info}");
    }
    for element in &ply.elements {
        let _ = writeln!(header, "element {} {}", element.name, element.count);
        for property in &element.properties {
            match &property.data {
                PropertyData::Scalar(column) => {
                    let _ = writeln!(header, "property {} {}", column.scalar_type().name(), property.name);
                }
                PropertyData::List { count_type, values, .. } => {
                    let item = values.scalar_type().name();
                    let _ = writeln!(header, "property list {} {item} {}", count_type.name(), property.name);
                }
            }
        }
    }
    header.push_str("end_header\n");

    let mut out = header.into_bytes();
    let encoding = ply.encoding;
    for element in &ply.elements {
        if element.properties.is_empty() {
            continue;
        }
        for row in 0..element.count {
            for property in &element.properties {
                match &property.data {
                    PropertyData::Scalar(column) => put(&mut out, column, row, encoding),
                    PropertyData::List { count_type, ends, values } => {
                        let end = ends.get(row).copied().unwrap_or(0);
                        let start = if row == 0 { 0 } else { ends.get(row - 1).copied().unwrap_or(0) };
                        put_count(&mut out, *count_type, end - start, encoding);
                        for i in start as usize..end as usize {
                            put(&mut out, values, i, encoding);
                        }
                    }
                }
            }
            if encoding == Encoding::Ascii {
                // Replace the separator after the last value with the line end.
                if out.last() == Some(&b' ') {
                    out.pop();
                }
                out.push(b'\n');
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Limits;
    use crate::ply::{Element, Property};

    fn one_list(count_type: ScalarType, ends: Vec<u32>, values: Column) -> Ply {
        Ply {
            encoding: Encoding::Ascii,
            comments: vec![],
            obj_info: vec![],
            elements: vec![Element {
                name: "face".into(),
                count: ends.len(),
                properties: vec![Property {
                    name: "vertex_indices".into(),
                    data: PropertyData::List { count_type, ends, values },
                }],
            }],
        }
    }

    #[test]
    fn data_that_does_not_match_its_declaration_is_refused() {
        let ok = one_list(ScalarType::U8, vec![3, 3, 7], Column::U32(vec![0, 1, 2, 3, 4, 5, 6]));
        assert_eq!(
            String::from_utf8(write(&ok).unwrap()).unwrap().lines().skip(4).collect::<Vec<_>>(),
            ["end_header", "3 0 1 2", "0", "4 3 4 5 6"]
        );
        assert_eq!(Ply::parse(&write(&ok).unwrap(), &Limits::DEFAULT).unwrap(), ok);
        let refused = |ply: Ply| assert!(matches!(write(&ply), Err(Error::Encode { .. })));
        refused(one_list(ScalarType::U8, vec![3, 2], Column::U32(vec![0, 1, 2])));
        refused(one_list(ScalarType::U8, vec![3], Column::U32(vec![0, 1, 2, 3])));
        refused(one_list(ScalarType::F32, vec![3], Column::U32(vec![0, 1, 2])));
        refused(one_list(ScalarType::I8, vec![128], Column::U8(vec![0; 128])));
        let mut short = ok.clone();
        short.elements[0].count = 4;
        refused(short);
        let mut named = ok.clone();
        named.elements[0].properties[0].name = "two words".into();
        refused(named);
        let mut commented = ok.clone();
        commented.comments.push("a\nb".into());
        refused(commented);
        let mut scalar = ok;
        scalar.elements[0]
            .properties
            .push(Property { name: "flags".into(), data: PropertyData::Scalar(Column::U8(vec![1])) });
        refused(scalar);
    }
}
