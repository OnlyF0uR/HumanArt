//! Stanford PLY, read and written in all three encodings: ASCII, binary little-endian and binary
//! big-endian `[IO-11]` `[DATA-11]`.
//!
//! [`Ply`] is the file as the format defines it: a header of elements, each with a row count and
//! typed properties (scalars, or lists with their own count type), and the data, stored column by
//! column in its declared type `[PERF-R2]`, so nothing is lost or converted and a binary file writes
//! back bit for bit. [`Ply::to_mesh`] extracts the usual mesh (positions, normals, colours, texture
//! coordinates, polygon faces) from the conventional names, and [`PlyMesh::to_ply`] builds a file
//! from one.
//!
//! # Reading
//!
//! The header must start with `ply`, declare one `format` (`ascii`, `binary_little_endian` or
//! `binary_big_endian`, version `1.0`) and end with `end_header`; `comment` and `obj_info` lines are
//! kept. Types are `char`/`int8`, `uchar`/`uint8`, `short`/`int16`, `ushort`/`uint16`, `int`/`int32`,
//! `uint`/`uint32`, `float`/`float32`, `double`/`float64`; a list count has an integer type and is
//! never negative. ASCII data is read as a stream of whitespace-separated values (line breaks carry no
//! meaning, as most readers treat them); integers must be written as integers within their type's
//! range, floats in any form Rust's float parser takes (`nan` and `inf` included). After the last
//! element, ASCII data may hold only whitespace; trailing bytes after binary data are ignored.
//!
//! # Limits `[ROB-02]`
//!
//! The header is at most [`MAX_HEADER_LEN`] bytes. A row count is checked against the remaining input
//! (every value takes at least one byte), against [`Limits::max_elements`] and, in memory, against
//! [`Limits::max_bytes`] before anything is allocated; list values are counted against the same
//! limits as they are read.

use crate::bytes::Reader;
use crate::error::{Error, Format, Result};
use crate::image::Limits;

mod mesh;
mod write;

pub use mesh::PlyMesh;
pub use write::write;

/// The format every error of this module names.
const FORMAT: Format = Format::Ply;

/// The longest header, `end_header` line included `[ROB-02]`.
pub const MAX_HEADER_LEN: usize = 1 << 16;

/// An [`Error::Invalid`] of this module.
const fn invalid(offset: usize, what: &'static str) -> Error {
    Error::Invalid { format: FORMAT, offset, what }
}

/// How the data after the header is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// Whitespace-separated decimal text.
    Ascii,
    /// Binary, least significant byte first.
    BinaryLittleEndian,
    /// Binary, most significant byte first.
    BinaryBigEndian,
}

impl Encoding {
    /// The name in the `format` line.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ascii => "ascii",
            Self::BinaryLittleEndian => "binary_little_endian",
            Self::BinaryBigEndian => "binary_big_endian",
        }
    }
}

/// The type of a scalar value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarType {
    /// `char` / `int8`.
    I8,
    /// `uchar` / `uint8`.
    U8,
    /// `short` / `int16`.
    I16,
    /// `ushort` / `uint16`.
    U16,
    /// `int` / `int32`.
    I32,
    /// `uint` / `uint32`.
    U32,
    /// `float` / `float32`.
    F32,
    /// `double` / `float64`.
    F64,
}

impl ScalarType {
    /// The type of a header type name, in either spelling.
    #[must_use]
    pub fn from_name(name: &[u8]) -> Option<Self> {
        Some(match name {
            b"char" | b"int8" => Self::I8,
            b"uchar" | b"uint8" => Self::U8,
            b"short" | b"int16" => Self::I16,
            b"ushort" | b"uint16" => Self::U16,
            b"int" | b"int32" => Self::I32,
            b"uint" | b"uint32" => Self::U32,
            b"float" | b"float32" => Self::F32,
            b"double" | b"float64" => Self::F64,
            _ => return None,
        })
    }

    /// The name the writer uses (the original PLY spelling).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::I8 => "char",
            Self::U8 => "uchar",
            Self::I16 => "short",
            Self::U16 => "ushort",
            Self::I32 => "int",
            Self::U32 => "uint",
            Self::F32 => "float",
            Self::F64 => "double",
        }
    }

    /// The size of one value in bytes, in binary data and in memory.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Whether the type holds integers.
    #[must_use]
    pub const fn is_integer(self) -> bool {
        !matches!(self, Self::F32 | Self::F64)
    }
}

/// The values of one scalar property (or the items of one list property), in their declared type.
#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    /// `char` values.
    I8(Vec<i8>),
    /// `uchar` values.
    U8(Vec<u8>),
    /// `short` values.
    I16(Vec<i16>),
    /// `ushort` values.
    U16(Vec<u16>),
    /// `int` values.
    I32(Vec<i32>),
    /// `uint` values.
    U32(Vec<u32>),
    /// `float` values.
    F32(Vec<f32>),
    /// `double` values.
    F64(Vec<f64>),
}

impl Column {
    /// An empty column of the type, with room for `capacity` values.
    #[must_use]
    pub fn with_capacity(ty: ScalarType, capacity: usize) -> Self {
        match ty {
            ScalarType::I8 => Self::I8(Vec::with_capacity(capacity)),
            ScalarType::U8 => Self::U8(Vec::with_capacity(capacity)),
            ScalarType::I16 => Self::I16(Vec::with_capacity(capacity)),
            ScalarType::U16 => Self::U16(Vec::with_capacity(capacity)),
            ScalarType::I32 => Self::I32(Vec::with_capacity(capacity)),
            ScalarType::U32 => Self::U32(Vec::with_capacity(capacity)),
            ScalarType::F32 => Self::F32(Vec::with_capacity(capacity)),
            ScalarType::F64 => Self::F64(Vec::with_capacity(capacity)),
        }
    }

    /// The type of the values.
    #[must_use]
    pub const fn scalar_type(&self) -> ScalarType {
        match self {
            Self::I8(_) => ScalarType::I8,
            Self::U8(_) => ScalarType::U8,
            Self::I16(_) => ScalarType::I16,
            Self::U16(_) => ScalarType::U16,
            Self::I32(_) => ScalarType::I32,
            Self::U32(_) => ScalarType::U32,
            Self::F32(_) => ScalarType::F32,
            Self::F64(_) => ScalarType::F64,
        }
    }

    /// The number of values.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::I8(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    /// Whether there are no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Value `i` as an `f64` (exact for every type), or `None` past the end.
    #[must_use]
    pub fn get(&self, i: usize) -> Option<f64> {
        match self {
            Self::I8(v) => v.get(i).map(|&x| f64::from(x)),
            Self::U8(v) => v.get(i).map(|&x| f64::from(x)),
            Self::I16(v) => v.get(i).map(|&x| f64::from(x)),
            Self::U16(v) => v.get(i).map(|&x| f64::from(x)),
            Self::I32(v) => v.get(i).map(|&x| f64::from(x)),
            Self::U32(v) => v.get(i).map(|&x| f64::from(x)),
            Self::F32(v) => v.get(i).map(|&x| f64::from(x)),
            Self::F64(v) => v.get(i).copied(),
        }
    }

    /// Reads one binary value and appends it.
    fn read_binary(&mut self, r: &mut Reader<'_>, big: bool) -> Result<()> {
        match self {
            Self::I8(v) => v.push(r.i8()?),
            Self::U8(v) => v.push(r.u8()?),
            Self::I16(v) => v.push(if big { r.i16_be()? } else { r.i16_le()? }),
            Self::U16(v) => v.push(if big { r.u16_be()? } else { r.u16_le()? }),
            Self::I32(v) => v.push(if big { r.i32_be()? } else { r.i32_le()? }),
            Self::U32(v) => v.push(if big { r.u32_be()? } else { r.u32_le()? }),
            Self::F32(v) => v.push(if big { r.f32_be()? } else { r.f32_le()? }),
            Self::F64(v) => v.push(if big { r.f64_be()? } else { r.f64_le()? }),
        }
        Ok(())
    }

    /// Parses one ASCII value and appends it.
    fn read_ascii(&mut self, token: &[u8], at: usize) -> Result<()> {
        let bad = || invalid(at, "bad value");
        match self {
            Self::I8(v) => v.push(i8::try_from(parse_int(token, at)?).map_err(|_| bad())?),
            Self::U8(v) => v.push(u8::try_from(parse_int(token, at)?).map_err(|_| bad())?),
            Self::I16(v) => v.push(i16::try_from(parse_int(token, at)?).map_err(|_| bad())?),
            Self::U16(v) => v.push(u16::try_from(parse_int(token, at)?).map_err(|_| bad())?),
            Self::I32(v) => v.push(i32::try_from(parse_int(token, at)?).map_err(|_| bad())?),
            Self::U32(v) => v.push(u32::try_from(parse_int(token, at)?).map_err(|_| bad())?),
            Self::F32(v) => v.push(text(token).and_then(|s| s.parse().ok()).ok_or_else(bad)?),
            Self::F64(v) => v.push(text(token).and_then(|s| s.parse().ok()).ok_or_else(bad)?),
        }
        Ok(())
    }
}

/// What a property holds.
#[derive(Clone, Debug, PartialEq)]
pub enum PropertyData {
    /// One value per row.
    Scalar(Column),
    /// A list per row: the items of every row one after the other in `values`, and for each row the
    /// index one past its last item in `ends`.
    List {
        /// The integer type the item count of every row is stored in.
        count_type: ScalarType,
        /// For each row, the index one past its last item.
        ends: Vec<u32>,
        /// The items.
        values: Column,
    },
}

/// A property of an element.
#[derive(Clone, Debug, PartialEq)]
pub struct Property {
    /// The name (`x`, `red`, `vertex_indices`).
    pub name: String,
    /// The values.
    pub data: PropertyData,
}

impl Property {
    /// The values of a scalar property; `None` for a list.
    #[must_use]
    pub const fn scalar(&self) -> Option<&Column> {
        match &self.data {
            PropertyData::Scalar(column) => Some(column),
            PropertyData::List { .. } => None,
        }
    }

    /// The items of row `row` of a list property; `None` for a scalar or past the last row.
    #[must_use]
    pub fn list(&self, row: usize) -> Option<(&Column, std::ops::Range<usize>)> {
        let PropertyData::List { ends, values, .. } = &self.data else { return None };
        let end = *ends.get(row)? as usize;
        let start = if row == 0 { 0 } else { *ends.get(row - 1)? as usize };
        (start <= end && end <= values.len()).then_some((values, start..end))
    }
}

/// An element: a named table of `count` rows.
#[derive(Clone, Debug, PartialEq)]
pub struct Element {
    /// The name (`vertex`, `face`).
    pub name: String,
    /// The number of rows.
    pub count: usize,
    /// The columns, in header order.
    pub properties: Vec<Property>,
}

impl Element {
    /// The property with the name, if there is one.
    #[must_use]
    pub fn property(&self, name: &str) -> Option<&Property> {
        self.properties.iter().find(|p| p.name == name)
    }
}

/// A PLY file. See the module documentation.
#[derive(Clone, Debug, PartialEq)]
pub struct Ply {
    /// How the data is (or is to be) stored.
    pub encoding: Encoding,
    /// The `comment` lines of the header, without the keyword.
    pub comments: Vec<String>,
    /// The `obj_info` lines of the header, without the keyword.
    pub obj_info: Vec<String>,
    /// The elements, in file order.
    pub elements: Vec<Element>,
}

/// A token as text, if it is UTF-8.
fn text(token: &[u8]) -> Option<&str> {
    std::str::from_utf8(token).ok()
}

/// Parses a signed decimal integer token of at most 18 digits.
fn parse_int(token: &[u8], at: usize) -> Result<i64> {
    let (negative, digits) = match token.split_first() {
        Some((b'-', rest)) => (true, rest),
        Some((b'+', rest)) => (false, rest),
        _ => (false, token),
    };
    if digits.is_empty() || digits.len() > 18 || !digits.iter().all(u8::is_ascii_digit) {
        return Err(invalid(at, "bad integer"));
    }
    let value = digits.iter().fold(0_i64, |v, &d| v * 10 + i64::from(d - b'0'));
    Ok(if negative { -value } else { value })
}

/// An element as the header declares it: name, row count and properties.
type DeclaredElement = (String, usize, Vec<(String, Declared)>);

/// The header's description of a property, before its data is read.
enum Declared {
    Scalar(ScalarType),
    List(ScalarType, ScalarType),
}

/// The bytes of `s` from `at` to the next line end, the line end excluded, and the offset after it.
fn header_line(s: &[u8], at: usize) -> Option<(&[u8], usize)> {
    let rest = s.get(at..)?;
    let newline = rest.iter().position(|&b| b == b'\n')?;
    let line = &rest[..newline];
    Some((line.strip_suffix(b"\r").unwrap_or(line), at + newline + 1))
}

/// A whitespace tokenizer over ASCII data that knows its file offset.
struct Tokens<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Tokens<'a> {
    /// The next token and its offset, or `None` at the end of the data.
    fn next(&mut self) -> Option<(&'a [u8], usize)> {
        let rest = self.data.get(self.pos..)?;
        let start = self.pos + rest.iter().position(|b| !b.is_ascii_whitespace())?;
        let tail = self.data.get(start..)?;
        let len = tail.iter().position(u8::is_ascii_whitespace).unwrap_or(tail.len());
        self.pos = start + len;
        Some((&tail[..len], start))
    }

    /// The next token, or an error at the end of the data.
    fn need(&mut self) -> Result<(&'a [u8], usize)> {
        self.next().ok_or(Error::Truncated { format: FORMAT, offset: self.data.len() })
    }
}

/// Splits a header line into its keyword and the rest.
fn split_keyword(line: &[u8]) -> (&[u8], &[u8]) {
    let end = line.iter().position(u8::is_ascii_whitespace).unwrap_or(line.len());
    let (keyword, rest) = line.split_at(end);
    (keyword, rest)
}

/// The text of a `comment` / `obj_info` line: everything after the one separator that follows the
/// keyword.
fn header_text(rest: &[u8], at: usize) -> Result<String> {
    let rest = rest.get(1..).unwrap_or(&[]);
    text(rest).map(str::to_owned).ok_or_else(|| invalid(at, "header text is not UTF-8"))
}

impl Ply {
    /// Parses a PLY file held in memory.
    pub fn parse(data: &[u8], limits: &Limits) -> Result<Self> {
        let header_end = data.len().min(MAX_HEADER_LEN);
        let header = &data[..header_end];
        let (first, mut at) = header_line(header, 0).ok_or_else(|| Self::header_cut(data))?;
        if first != b"ply" {
            return Err(invalid(0, "not a PLY file"));
        }
        let mut encoding = None;
        let mut comments = Vec::new();
        let mut obj_info = Vec::new();
        let mut declared: Vec<DeclaredElement> = Vec::new();
        loop {
            let line_at = at;
            let (line, next) = header_line(header, at).ok_or_else(|| Self::header_cut(data))?;
            at = next;
            let (keyword, rest) = split_keyword(line);
            let mut words = rest.split(u8::is_ascii_whitespace).filter(|w| !w.is_empty());
            let mut word = || words.next().ok_or_else(|| invalid(line_at, "header line is incomplete"));
            match keyword {
                b"end_header" => break,
                b"comment" => comments.push(header_text(rest, line_at)?),
                b"obj_info" => obj_info.push(header_text(rest, line_at)?),
                b"format" => {
                    if encoding.is_some() {
                        return Err(invalid(line_at, "second format line"));
                    }
                    encoding = Some(match word()? {
                        b"ascii" => Encoding::Ascii,
                        b"binary_little_endian" => Encoding::BinaryLittleEndian,
                        b"binary_big_endian" => Encoding::BinaryBigEndian,
                        _ => return Err(invalid(line_at, "unknown PLY format")),
                    });
                    if word()? != b"1.0" {
                        return Err(Error::Unsupported { format: FORMAT, what: "PLY versions other than 1.0" });
                    }
                }
                b"element" => {
                    let name = text(word()?).ok_or_else(|| invalid(line_at, "name is not UTF-8"))?.to_owned();
                    let count = parse_int(word()?, line_at)?;
                    let count = usize::try_from(count).map_err(|_| invalid(line_at, "negative element count"))?;
                    declared.push((name, count, Vec::new()));
                }
                b"property" => {
                    let Some((_, _, properties)) = declared.last_mut() else {
                        return Err(invalid(line_at, "property before any element"));
                    };
                    let ty =
                        |w: &[u8]| ScalarType::from_name(w).ok_or_else(|| invalid(line_at, "unknown property type"));
                    let first = word()?;
                    let kind = if first == b"list" {
                        let count = ty(word()?)?;
                        if !count.is_integer() {
                            return Err(invalid(line_at, "list count type is not an integer type"));
                        }
                        Declared::List(count, ty(word()?)?)
                    } else {
                        Declared::Scalar(ty(first)?)
                    };
                    let name = text(word()?).ok_or_else(|| invalid(line_at, "name is not UTF-8"))?.to_owned();
                    properties.push((name, kind));
                }
                _ => return Err(invalid(line_at, "unknown header line")),
            }
            if words.next().is_some() && !matches!(keyword, b"comment" | b"obj_info") {
                return Err(invalid(line_at, "header line has extra words"));
            }
        }
        let encoding = encoding.ok_or_else(|| invalid(0, "header without a format line"))?;

        let mut reader = Reader::new(data, FORMAT);
        reader.seek(at)?;
        let mut tokens = Tokens { data, pos: at };
        let mut elements = Vec::with_capacity(declared.len());
        let mut budget = Budget { limits, bytes: 0 };
        for (name, count, properties) in declared {
            let element = match encoding {
                Encoding::Ascii => read_ascii_element(&mut tokens, name, count, properties, &mut budget)?,
                Encoding::BinaryLittleEndian | Encoding::BinaryBigEndian => {
                    let big = encoding == Encoding::BinaryBigEndian;
                    read_binary_element(&mut reader, big, name, count, properties, &mut budget)?
                }
            };
            elements.push(element);
        }
        if encoding == Encoding::Ascii
            && let Some((_, at)) = tokens.next()
        {
            return Err(invalid(at, "data after the last element"));
        }
        Ok(Self { encoding, comments, obj_info, elements })
    }

    /// The error for a header that does not end within the input and [`MAX_HEADER_LEN`].
    fn header_cut(data: &[u8]) -> Error {
        if data.len() >= MAX_HEADER_LEN {
            Error::Limit { format: FORMAT, what: "header length", limit: MAX_HEADER_LEN as u64, got: data.len() as u64 }
        } else {
            Error::Truncated { format: FORMAT, offset: data.len() }
        }
    }

    /// The element with the name, if there is one.
    #[must_use]
    pub fn element(&self, name: &str) -> Option<&Element> {
        self.elements.iter().find(|e| e.name == name)
    }
}

/// What the data read so far holds, against the limits.
struct Budget<'l> {
    limits: &'l Limits,
    bytes: u64,
}

impl Budget<'_> {
    /// Accounts for `count` values of `size` bytes in one column.
    fn take(&mut self, count: usize, size: usize, what: &'static str) -> Result<()> {
        // List ends are 32-bit.
        let limit = self.limits.max_elements.min(u64::from(u32::MAX));
        if count as u64 > limit {
            return Err(Error::Limit { format: FORMAT, what, limit, got: count as u64 });
        }
        let bytes = self.bytes.saturating_add((count as u64).saturating_mul(size as u64));
        if bytes > self.limits.max_bytes {
            return Err(Error::Limit {
                format: FORMAT,
                what: "decoded size",
                limit: self.limits.max_bytes,
                got: bytes,
            });
        }
        self.bytes = bytes;
        Ok(())
    }
}

/// Allocates the columns of an element after checking `count` against the input left and the
/// limits. `min_row` is the fewest input bytes one row can take.
fn start_element(
    count: usize,
    min_row: usize,
    remaining: usize,
    at: usize,
    properties: &[(String, Declared)],
    budget: &mut Budget<'_>,
) -> Result<Vec<PropertyData>> {
    if count.checked_mul(min_row).is_none_or(|total| total > remaining) {
        return Err(Error::Truncated { format: FORMAT, offset: at.saturating_add(remaining) });
    }
    let mut out = Vec::with_capacity(properties.len());
    for (_, declared) in properties {
        out.push(match *declared {
            Declared::Scalar(ty) => {
                budget.take(count, ty.size(), "element row count")?;
                PropertyData::Scalar(Column::with_capacity(ty, count))
            }
            Declared::List(count_type, ty) => {
                budget.take(count, 4, "element row count")?;
                PropertyData::List { count_type, ends: Vec::with_capacity(count), values: Column::with_capacity(ty, 0) }
            }
        });
    }
    Ok(out)
}

/// Checks a list length read at `at` against the input left (`per_item` bytes at least per item)
/// and accounts for its items.
fn list_length(
    len: i64,
    at: usize,
    remaining: usize,
    per_item: usize,
    total: usize,
    ty: ScalarType,
    budget: &mut Budget<'_>,
) -> Result<usize> {
    let len = usize::try_from(len).map_err(|_| invalid(at, "negative list length"))?;
    if len.saturating_mul(per_item) > remaining {
        return Err(Error::Truncated { format: FORMAT, offset: at });
    }
    budget.take(total + len, 0, "list item count")?;
    budget.take(len, ty.size(), "list item count")?;
    Ok(len)
}

/// Assembles an element from its columns.
fn finish_element(name: String, count: usize, properties: Vec<(String, Declared)>, data: Vec<PropertyData>) -> Element {
    let properties = properties.into_iter().zip(data).map(|((name, _), data)| Property { name, data }).collect();
    Element { name, count, properties }
}

/// Reads the rows of an element from ASCII data.
fn read_ascii_element(
    tokens: &mut Tokens<'_>,
    name: String,
    count: usize,
    properties: Vec<(String, Declared)>,
    budget: &mut Budget<'_>,
) -> Result<Element> {
    let remaining = tokens.data.len().saturating_sub(tokens.pos);
    // Every value is at least one character.
    let mut data = start_element(count, properties.len(), remaining, tokens.pos, &properties, budget)?;
    if properties.is_empty() {
        return Ok(finish_element(name, count, properties, data));
    }
    for _ in 0..count {
        for column in &mut data {
            match column {
                PropertyData::Scalar(values) => {
                    let (token, at) = tokens.need()?;
                    values.read_ascii(token, at)?;
                }
                PropertyData::List { ends, values, .. } => {
                    let (token, at) = tokens.need()?;
                    let remaining = tokens.data.len().saturating_sub(tokens.pos);
                    let len = list_length(
                        parse_int(token, at)?,
                        at,
                        remaining,
                        1,
                        values.len(),
                        values.scalar_type(),
                        budget,
                    )?;
                    for _ in 0..len {
                        let (token, at) = tokens.need()?;
                        values.read_ascii(token, at)?;
                    }
                    ends.push(values.len() as u32);
                }
            }
        }
    }
    Ok(finish_element(name, count, properties, data))
}

/// Reads a binary list count as an integer.
fn read_count(r: &mut Reader<'_>, ty: ScalarType, big: bool) -> Result<i64> {
    Ok(match ty {
        ScalarType::I8 => i64::from(r.i8()?),
        ScalarType::U8 => i64::from(r.u8()?),
        ScalarType::I16 => i64::from(if big { r.i16_be()? } else { r.i16_le()? }),
        ScalarType::U16 => i64::from(if big { r.u16_be()? } else { r.u16_le()? }),
        ScalarType::I32 => i64::from(if big { r.i32_be()? } else { r.i32_le()? }),
        ScalarType::U32 => i64::from(if big { r.u32_be()? } else { r.u32_le()? }),
        // The header refuses float count types.
        ScalarType::F32 | ScalarType::F64 => return Err(r.invalid("list count type is not an integer type")),
    })
}

/// Reads the rows of an element from binary data.
fn read_binary_element(
    r: &mut Reader<'_>,
    big: bool,
    name: String,
    count: usize,
    properties: Vec<(String, Declared)>,
    budget: &mut Budget<'_>,
) -> Result<Element> {
    let min_row = properties
        .iter()
        .map(|(_, d)| match d {
            Declared::Scalar(ty) | Declared::List(ty, _) => ty.size(),
        })
        .sum();
    let mut data = start_element(count, min_row, r.remaining(), r.pos(), &properties, budget)?;
    if properties.is_empty() {
        return Ok(finish_element(name, count, properties, data));
    }
    for _ in 0..count {
        for column in &mut data {
            match column {
                PropertyData::Scalar(values) => values.read_binary(r, big)?,
                PropertyData::List { count_type, ends, values } => {
                    let at = r.pos();
                    let len = read_count(r, *count_type, big)?;
                    let ty = values.scalar_type();
                    let len = list_length(len, at, r.remaining(), ty.size(), values.len(), ty, budget)?;
                    for _ in 0..len {
                        values.read_binary(r, big)?;
                    }
                    ends.push(values.len() as u32);
                }
            }
        }
    }
    Ok(finish_element(name, count, properties, data))
}

#[cfg(test)]
mod tests;
