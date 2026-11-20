//! Deterministic JSON output: compact and pretty, from a [`Value`] or streamed `[DEP-05]`.
//!
//! The same input always gives the same bytes `[AC-WP07-1]`:
//!
//! - object members are written in their stored order;
//! - numbers are written by [`number`](super::number): integers in full, floats with the shortest
//!   digits that read back as the same value, always with a `.` or an `e`;
//! - strings are written as UTF-8 with the fewest escapes: `\"`, `\\`, the short forms `\b` `\f` `\n`
//!   `\r` `\t`, and `\u00XX` (lower-case hexadecimal) for the other control characters below U+0020.
//!   Nothing else is escaped, not `/`, not U+007F, not U+2028;
//! - no byte order mark and no trailing newline are written.
//!
//! NaN and the infinities have no JSON form: writing one is an error, never a silent `null`.
//!
//! Output goes to any [`Sink`], which `String` and `Vec<u8>` are. Nothing is allocated per value: the
//! writers push text straight into the sink; [`JsonWriter`] keeps one small stack of open containers,
//! and writing a [`Value`] keeps one stack of iterators for the containers it is inside.

use super::number::{self, Number};
use super::value::Value;
use crate::error::{Error, Format, Result};
use std::fmt;

/// Where JSON text is written: a growable buffer that accepts UTF-8.
pub trait Sink {
    /// Appends `text`.
    fn push_str(&mut self, text: &str);
}

impl Sink for String {
    #[inline]
    fn push_str(&mut self, text: &str) {
        String::push_str(self, text);
    }
}

impl Sink for Vec<u8> {
    #[inline]
    fn push_str(&mut self, text: &str) {
        self.extend_from_slice(text.as_bytes());
    }
}

/// Lets the number formatters, which write through `fmt::Write`, write into a [`Sink`].
struct Formatted<'a, S: Sink>(&'a mut S);

impl<S: Sink> fmt::Write for Formatted<'_, S> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.0.push_str(text);
        Ok(())
    }
}

/// How nested containers are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Indent {
    /// Everything on one line with no white space at all.
    Compact,
    /// One member per line, nested by this many spaces per level.
    Spaces(u8),
    /// One member per line, nested by one tab per level.
    Tab,
}

/// The layout of written JSON.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Style {
    /// The indentation of nested containers.
    pub indent: Indent,
    /// With an indenting style: write an array that holds no array or object on one line, as
    /// `[1.0, 2.0, 3.0]`. Vectors, matrices and id lists then take one line instead of one line per
    /// number. Applies to arrays written from a [`Value`]; a streaming caller chooses per array with
    /// [`JsonWriter::begin_array_inline`].
    pub inline_scalar_arrays: bool,
}

impl Style {
    /// No white space: `{"a":[1,2]}`.
    pub const COMPACT: Self = Self { indent: Indent::Compact, inline_scalar_arrays: false };
    /// Two spaces per level, one member or element per line.
    pub const PRETTY: Self = Self { indent: Indent::Spaces(2), inline_scalar_arrays: false };
    /// Two spaces per level, one member per line, arrays of scalars on one line. The layout meant
    /// for documents that people read and diff.
    pub const PRETTY_INLINE: Self = Self { indent: Indent::Spaces(2), inline_scalar_arrays: true };

    const fn is_compact(self) -> bool {
        matches!(self.indent, Indent::Compact)
    }
}

const fn encode(what: &'static str) -> Error {
    Error::Encode { format: Format::Json, what }
}

/// The escapes of the control characters U+0000 to U+001F.
const CONTROL_ESCAPES: [&str; 32] = [
    "\\u0000", "\\u0001", "\\u0002", "\\u0003", "\\u0004", "\\u0005", "\\u0006", "\\u0007", "\\b", "\\t", "\\n",
    "\\u000b", "\\f", "\\r", "\\u000e", "\\u000f", "\\u0010", "\\u0011", "\\u0012", "\\u0013", "\\u0014", "\\u0015",
    "\\u0016", "\\u0017", "\\u0018", "\\u0019", "\\u001a", "\\u001b", "\\u001c", "\\u001d", "\\u001e", "\\u001f",
];

/// Writes `text` as a JSON string literal, quotes included, with the minimal escapes described in
/// the module comment.
pub fn write_string<S: Sink>(out: &mut S, text: &str) {
    out.push_str("\"");
    let mut written = 0;
    for (at, &byte) in text.as_bytes().iter().enumerate() {
        let escape = match byte {
            b'"' => "\\\"",
            b'\\' => "\\\\",
            0..=0x1F => CONTROL_ESCAPES[byte as usize],
            _ => continue,
        };
        // `at` is the position of an ASCII byte, so both cuts fall on character boundaries.
        out.push_str(text.get(written..at).unwrap_or_default());
        out.push_str(escape);
        written = at + 1;
    }
    out.push_str(text.get(written..).unwrap_or_default());
    out.push_str("\"");
}

/// Writes a number. Fails for NaN and the infinities.
pub fn write_number<S: Sink>(out: &mut S, value: Number) -> Result<()> {
    use fmt::Write as _;
    if !value.is_finite() {
        return Err(encode("NaN and infinity have no JSON form"));
    }
    write!(Formatted(out), "{value}").map_err(|_| encode("a number could not be formatted"))
}

/// One open container of a [`JsonWriter`].
#[derive(Clone, Copy, Debug)]
struct Level {
    object: bool,
    /// Nothing has been written inside yet.
    empty: bool,
    /// The container and everything inside it stay on one line.
    inline: bool,
}

/// A streaming JSON writer that checks its own nesting.
///
/// Calls mirror the text: `begin_object`, then `key` and a value for each member, then `end_object`;
/// `begin_array`, values, `end_array`; scalars through [`null`](Self::null), [`bool`](Self::bool),
/// [`i64`](Self::i64), [`u64`](Self::u64), [`f64`](Self::f64), [`f32`](Self::f32),
/// [`string`](Self::string); whole subtrees through [`value`](Self::value). Every call returns the
/// writer, so calls chain with `?`. A call that would produce invalid JSON (a value without its key,
/// a key in an array, closing the wrong container, a second top-level value, a non-finite number)
/// returns [`Error::Encode`] and writes nothing; [`finish`](Self::finish) checks that exactly one
/// complete value was written. The writer does not detect duplicate keys: the caller owns the key
/// set.
///
/// ```
/// use humanart_io::json::JsonWriter;
/// let mut text = String::new();
/// let mut w = JsonWriter::compact(&mut text);
/// w.begin_object()?.key("name")?.string("skin")?.key("size")?.begin_array()?.u64(4)?.u64(8)?.end_array()?.end_object()?;
/// w.finish()?;
/// assert_eq!(text, r#"{"name":"skin","size":[4,8]}"#);
/// # Ok::<(), humanart_io::Error>(())
/// ```
#[derive(Debug)]
pub struct JsonWriter<'a, S: Sink> {
    out: &'a mut S,
    style: Style,
    stack: Vec<Level>,
    /// The top-level value has been started.
    started: bool,
    /// Inside an object: a key has been written and its value is due.
    key_written: bool,
}

const SPACES: &str = "                                                                ";
const TABS: &str = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t";

impl<'a, S: Sink> JsonWriter<'a, S> {
    /// A writer in the given style that appends to `out`.
    pub const fn new(out: &'a mut S, style: Style) -> Self {
        Self { out, style, stack: Vec::new(), started: false, key_written: false }
    }

    /// A writer of [`Style::COMPACT`] text.
    pub const fn compact(out: &'a mut S) -> Self {
        Self::new(out, Style::COMPACT)
    }

    /// A writer of [`Style::PRETTY`] text.
    pub const fn pretty(out: &'a mut S) -> Self {
        Self::new(out, Style::PRETTY)
    }

    /// How many containers are open.
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Starts a new line indented for `depth` open containers.
    fn line(&mut self, depth: usize) {
        let (unit, per_level) = match self.style.indent {
            Indent::Compact => return,
            Indent::Spaces(n) => (SPACES, usize::from(n)),
            Indent::Tab => (TABS, 1),
        };
        self.out.push_str("\n");
        let mut left = depth * per_level;
        while left > 0 {
            let n = left.min(unit.len());
            self.out.push_str(unit.get(..n).unwrap_or_default());
            left -= n;
        }
    }

    /// Writes what separates an element or member from the one before it and starts its line.
    fn separate(&mut self, level: Level) {
        if !level.empty {
            self.out.push_str(",");
        }
        if !level.inline {
            self.line(self.stack.len());
        } else if !level.empty && !self.style.is_compact() {
            self.out.push_str(" ");
        }
    }

    /// Checks that a value may be written here and writes what precedes it.
    fn before_value(&mut self) -> Result<()> {
        match self.stack.last().copied() {
            None => {
                if self.started {
                    return Err(encode("a JSON text holds one top-level value"));
                }
                self.started = true;
            }
            Some(level) if level.object => {
                if !self.key_written {
                    return Err(encode("an object member needs a key before its value"));
                }
                self.key_written = false;
            }
            Some(level) => {
                self.separate(level);
                if let Some(top) = self.stack.last_mut() {
                    top.empty = false;
                }
            }
        }
        Ok(())
    }

    fn begin(&mut self, object: bool, inline: bool) -> Result<&mut Self> {
        let inline = inline || self.style.is_compact() || self.stack.last().is_some_and(|level| level.inline);
        self.before_value()?;
        self.stack.push(Level { object, empty: true, inline });
        self.out.push_str(if object { "{" } else { "[" });
        Ok(self)
    }

    fn end(&mut self, object: bool) -> Result<&mut Self> {
        let Some(level) = self.stack.last().copied() else {
            return Err(encode("no container is open"));
        };
        if level.object != object {
            return Err(encode(if object { "end_object inside an array" } else { "end_array inside an object" }));
        }
        if self.key_written {
            return Err(encode("an object key has no value"));
        }
        self.stack.pop();
        if !level.empty && !level.inline {
            self.line(self.stack.len());
        }
        self.out.push_str(if object { "}" } else { "]" });
        Ok(self)
    }

    /// Opens an object.
    pub fn begin_object(&mut self) -> Result<&mut Self> {
        self.begin(true, false)
    }

    /// Closes the object opened last.
    pub fn end_object(&mut self) -> Result<&mut Self> {
        self.end(true)
    }

    /// Opens an array.
    pub fn begin_array(&mut self) -> Result<&mut Self> {
        self.begin(false, false)
    }

    /// Opens an array that is written on one line even in an indenting style, with everything
    /// inside it: `[1.0, 2.0, 3.0]`.
    pub fn begin_array_inline(&mut self) -> Result<&mut Self> {
        self.begin(false, true)
    }

    /// Closes the array opened last.
    pub fn end_array(&mut self) -> Result<&mut Self> {
        self.end(false)
    }

    /// Writes the key of the next member of the open object.
    pub fn key(&mut self, key: &str) -> Result<&mut Self> {
        let level = match self.stack.last().copied() {
            Some(level) if level.object => level,
            _ => return Err(encode("a key is only allowed inside an object")),
        };
        if self.key_written {
            return Err(encode("an object key has no value"));
        }
        self.separate(level);
        if let Some(top) = self.stack.last_mut() {
            top.empty = false;
        }
        write_string(self.out, key);
        self.out.push_str(if self.style.is_compact() { ":" } else { ": " });
        self.key_written = true;
        Ok(self)
    }

    /// Writes `null`.
    pub fn null(&mut self) -> Result<&mut Self> {
        self.before_value()?;
        self.out.push_str("null");
        Ok(self)
    }

    /// Writes `true` or `false`.
    pub fn bool(&mut self, value: bool) -> Result<&mut Self> {
        self.before_value()?;
        self.out.push_str(if value { "true" } else { "false" });
        Ok(self)
    }

    /// Writes a number. Fails for NaN and the infinities.
    pub fn number(&mut self, value: Number) -> Result<&mut Self> {
        if !value.is_finite() {
            return Err(encode("NaN and infinity have no JSON form"));
        }
        self.before_value()?;
        write_number(self.out, value)?;
        Ok(self)
    }

    /// Writes a signed integer.
    pub fn i64(&mut self, value: i64) -> Result<&mut Self> {
        self.number(Number::Int(value))
    }

    /// Writes an unsigned integer.
    pub fn u64(&mut self, value: u64) -> Result<&mut Self> {
        self.number(Number::UInt(value))
    }

    /// Writes a float with the shortest digits that read back as the same `f64`, always as a float
    /// (`1.0`, not `1`). Fails for NaN and the infinities.
    pub fn f64(&mut self, value: f64) -> Result<&mut Self> {
        self.number(Number::Float(value))
    }

    /// Writes an `f32` with the shortest digits that read back (as `f64`, then narrowed) as the same
    /// `f32`: `0.1`, where the `f64` route would write `0.10000000149011612`. Fails for NaN and the
    /// infinities.
    pub fn f32(&mut self, value: f32) -> Result<&mut Self> {
        if !value.is_finite() {
            return Err(encode("NaN and infinity have no JSON form"));
        }
        self.before_value()?;
        number::write_f32(&mut Formatted(self.out), value).map_err(|_| encode("a number could not be formatted"))?;
        Ok(self)
    }

    /// Writes a string.
    pub fn string(&mut self, value: &str) -> Result<&mut Self> {
        self.before_value()?;
        write_string(self.out, value);
        Ok(self)
    }

    /// Writes a whole value with everything inside it. A non-finite number inside it is reported
    /// when it is reached; what was written before it stays in the output.
    pub fn value(&mut self, value: &Value) -> Result<&mut Self> {
        enum Open<'v> {
            Array(std::slice::Iter<'v, Value>),
            Object(std::slice::Iter<'v, (String, Value)>),
        }
        let mut open: Vec<Open<'_>> = Vec::new();
        let mut due = Some(value);
        loop {
            if let Some(value) = due.take() {
                match value {
                    Value::Null => {
                        self.null()?;
                    }
                    Value::Bool(b) => {
                        self.bool(*b)?;
                    }
                    Value::Number(n) => {
                        self.number(*n)?;
                    }
                    Value::String(s) => {
                        self.string(s)?;
                    }
                    Value::Array(items) => {
                        let inline = self.style.inline_scalar_arrays
                            && !items.iter().any(|item| matches!(item, Value::Array(_) | Value::Object(_)));
                        self.begin(false, inline)?;
                        open.push(Open::Array(items.iter()));
                    }
                    Value::Object(object) => {
                        self.begin(true, false)?;
                        open.push(Open::Object(object.entries().iter()));
                    }
                }
            }
            match open.last_mut() {
                None => return Ok(self),
                Some(Open::Array(items)) => {
                    if let Some(item) = items.next() {
                        due = Some(item);
                    } else {
                        self.end(false)?;
                        open.pop();
                    }
                }
                Some(Open::Object(members)) => {
                    if let Some((key, member)) = members.next() {
                        self.key(key)?;
                        due = Some(member);
                    } else {
                        self.end(true)?;
                        open.pop();
                    }
                }
            }
        }
    }

    /// Writes any [`ToJson`] value: an array element, or a member value after [`key`](Self::key).
    pub fn put<T: ToJson + ?Sized>(&mut self, value: &T) -> Result<&mut Self> {
        value.to_json(self)?;
        Ok(self)
    }

    /// Writes the member `key` with the value `value`: `key` then [`put`](Self::put).
    pub fn field<T: ToJson + ?Sized>(&mut self, key: &str, value: &T) -> Result<&mut Self> {
        self.key(key)?;
        self.put(value)
    }

    /// Writes the member `key` when `value` is `Some`, and nothing when it is `None`; the reading
    /// side ([`Node::get_opt`](super::Node::get_opt)) treats the two alike.
    pub fn opt_field<T: ToJson>(&mut self, key: &str, value: Option<&T>) -> Result<&mut Self> {
        match value {
            Some(value) => self.field(key, value),
            None => Ok(self),
        }
    }

    /// Ends the text: fails unless exactly one complete value has been written.
    pub fn finish(self) -> Result<()> {
        if !self.stack.is_empty() {
            return Err(encode("a container is still open"));
        }
        if !self.started {
            return Err(encode("no value was written"));
        }
        Ok(())
    }
}

/// A type that writes itself as one JSON value through a [`JsonWriter`]: the writing half of
/// [`FromJson`](super::FromJson). Implement it for every document and protocol type.
///
/// ```
/// use humanart_io::Result;
/// use humanart_io::json::{JsonWriter, Sink, Style, ToJson, stringify};
///
/// struct Lod { name: String, scale: [f32; 3], parent: Option<u32> }
///
/// impl ToJson for Lod {
///     fn to_json<S: Sink>(&self, w: &mut JsonWriter<'_, S>) -> Result<()> {
///         w.begin_object()?.field("name", &self.name)?.field("scale", &self.scale)?;
///         w.opt_field("parent", self.parent.as_ref())?.end_object()?;
///         Ok(())
///     }
/// }
///
/// let lod = Lod { name: "lod0".into(), scale: [1.0, 0.5, 0.1], parent: None };
/// assert_eq!(stringify(&lod, Style::COMPACT)?, r#"{"name":"lod0","scale":[1.0,0.5,0.1]}"#);
/// # Ok::<(), humanart_io::Error>(())
/// ```
pub trait ToJson {
    /// Whether the value is written as a scalar. Arrays of scalars go on one line in a style with
    /// [`Style::inline_scalar_arrays`].
    const SCALAR: bool = false;

    /// Writes the value as one JSON value.
    fn to_json<S: Sink>(&self, w: &mut JsonWriter<'_, S>) -> Result<()>;
}

macro_rules! to_json_scalar {
    ($($ty:ty => |$v:ident, $w:ident| $body:expr;)*) => {$(
        impl ToJson for $ty {
            const SCALAR: bool = true;

            fn to_json<S: Sink>(&self, $w: &mut JsonWriter<'_, S>) -> Result<()> {
                let $v = self;
                $body.map(|_| ())
            }
        }
    )*};
}

to_json_scalar! {
    bool => |v, w| w.bool(*v);
    i8 => |v, w| w.i64(i64::from(*v));
    i16 => |v, w| w.i64(i64::from(*v));
    i32 => |v, w| w.i64(i64::from(*v));
    i64 => |v, w| w.i64(*v);
    isize => |v, w| w.number(Number::from(*v));
    u8 => |v, w| w.i64(i64::from(*v));
    u16 => |v, w| w.i64(i64::from(*v));
    u32 => |v, w| w.i64(i64::from(*v));
    u64 => |v, w| w.number(Number::from(*v));
    usize => |v, w| w.number(Number::from(*v));
    f32 => |v, w| w.f32(*v);
    f64 => |v, w| w.f64(*v);
    Number => |v, w| w.number(*v);
    str => |v, w| w.string(v);
    String => |v, w| w.string(v);
}

impl ToJson for Value {
    fn to_json<S: Sink>(&self, w: &mut JsonWriter<'_, S>) -> Result<()> {
        w.value(self).map(|_| ())
    }
}

impl<T: ToJson + ?Sized> ToJson for &T {
    const SCALAR: bool = T::SCALAR;

    fn to_json<S: Sink>(&self, w: &mut JsonWriter<'_, S>) -> Result<()> {
        (**self).to_json(w)
    }
}

/// `None` is written as `null`; [`JsonWriter::opt_field`] leaves the member out instead.
impl<T: ToJson> ToJson for Option<T> {
    const SCALAR: bool = T::SCALAR;

    fn to_json<S: Sink>(&self, w: &mut JsonWriter<'_, S>) -> Result<()> {
        match self {
            Some(value) => value.to_json(w),
            None => w.null().map(|_| ()),
        }
    }
}

impl<T: ToJson> ToJson for [T] {
    fn to_json<S: Sink>(&self, w: &mut JsonWriter<'_, S>) -> Result<()> {
        let inline = T::SCALAR && w.style.inline_scalar_arrays;
        w.begin(false, inline)?;
        for item in self {
            item.to_json(w)?;
        }
        w.end_array().map(|_| ())
    }
}

impl<T: ToJson, const N: usize> ToJson for [T; N] {
    fn to_json<S: Sink>(&self, w: &mut JsonWriter<'_, S>) -> Result<()> {
        self.as_slice().to_json(w)
    }
}

impl<T: ToJson> ToJson for Vec<T> {
    fn to_json<S: Sink>(&self, w: &mut JsonWriter<'_, S>) -> Result<()> {
        self.as_slice().to_json(w)
    }
}

/// The text of any [`ToJson`] value in the given style.
pub fn stringify<T: ToJson + ?Sized>(value: &T, style: Style) -> Result<String> {
    let mut out = String::new();
    let mut writer = JsonWriter::new(&mut out, style);
    value.to_json(&mut writer)?;
    writer.finish()?;
    Ok(out)
}

/// Appends `value` to `out` in the given style. On an error (a non-finite number) `out` holds the
/// incomplete text written up to it.
pub fn write<S: Sink>(value: &Value, style: Style, out: &mut S) -> Result<()> {
    let mut writer = JsonWriter::new(out, style);
    writer.value(value)?;
    writer.finish()
}

/// The compact text of `value`.
pub fn to_string(value: &Value) -> Result<String> {
    let mut out = String::new();
    write(value, Style::COMPACT, &mut out)?;
    Ok(out)
}

/// The text of `value` in [`Style::PRETTY`].
pub fn to_string_pretty(value: &Value) -> Result<String> {
    let mut out = String::new();
    write(value, Style::PRETTY, &mut out)?;
    Ok(out)
}

/// The compact text of `value` as bytes.
pub fn to_vec(value: &Value) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    write(value, Style::COMPACT, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::{Object, parse_str};

    fn sample() -> Value {
        parse_str(r#"{"name":"a\"b","list":[1,2.5,"x",null,true],"empty":[],"none":{},"nested":{"m":[[1,2],[3,4]],"o":[{"k":1}]}}"#)
            .unwrap()
    }

    #[test]
    fn compact_output_has_no_white_space() {
        let text = to_string(&sample()).unwrap();
        assert_eq!(
            text,
            r#"{"name":"a\"b","list":[1,2.5,"x",null,true],"empty":[],"none":{},"nested":{"m":[[1,2],[3,4]],"o":[{"k":1}]}}"#
        );
        assert_eq!(to_vec(&sample()).unwrap(), text.as_bytes());
        for (value, expected) in [
            (Value::Null, "null"),
            (Value::from(false), "false"),
            (Value::from(-3), "-3"),
            (Value::from(u64::MAX), "18446744073709551615"),
            (Value::from(1.0), "1.0"),
            (Value::from(-0.0), "-0.0"),
            (Value::from(1e300), "1e300"),
            (Value::from(""), "\"\""),
            (Value::Array(Vec::new()), "[]"),
            (Value::Object(Object::new()), "{}"),
        ] {
            assert_eq!(to_string(&value).unwrap(), expected);
            assert_eq!(to_string_pretty(&value).unwrap(), expected);
        }
    }

    #[test]
    fn pretty_output_is_indented() {
        let expected = r#"{
  "name": "a\"b",
  "list": [
    1,
    2.5,
    "x",
    null,
    true
  ],
  "empty": [],
  "none": {},
  "nested": {
    "m": [
      [
        1,
        2
      ],
      [
        3,
        4
      ]
    ],
    "o": [
      {
        "k": 1
      }
    ]
  }
}"#;
        assert_eq!(to_string_pretty(&sample()).unwrap(), expected);
        let inline = r#"{
  "name": "a\"b",
  "list": [1, 2.5, "x", null, true],
  "empty": [],
  "none": {},
  "nested": {
    "m": [
      [1, 2],
      [3, 4]
    ],
    "o": [
      {
        "k": 1
      }
    ]
  }
}"#;
        let mut out = String::new();
        write(&sample(), Style::PRETTY_INLINE, &mut out).unwrap();
        assert_eq!(out, inline);
        // Tabs and other widths.
        let small = parse_str(r#"{"a":[1,{"b":null}]}"#).unwrap();
        let mut out = String::new();
        write(&small, Style { indent: Indent::Tab, inline_scalar_arrays: false }, &mut out).unwrap();
        assert_eq!(out, "{\n\t\"a\": [\n\t\t1,\n\t\t{\n\t\t\t\"b\": null\n\t\t}\n\t]\n}");
        let mut out = Vec::new();
        write(&small, Style { indent: Indent::Spaces(1), inline_scalar_arrays: false }, &mut out).unwrap();
        assert_eq!(out, b"{\n \"a\": [\n  1,\n  {\n   \"b\": null\n  }\n ]\n}");
        // Zero spaces still breaks lines.
        let mut out = String::new();
        write(&small, Style { indent: Indent::Spaces(0), inline_scalar_arrays: true }, &mut out).unwrap();
        assert_eq!(out, "{\n\"a\": [\n1,\n{\n\"b\": null\n}\n]\n}");
        // Every layout reads back as the same value.
        for style in [Style::COMPACT, Style::PRETTY, Style::PRETTY_INLINE] {
            let mut out = String::new();
            write(&sample(), style, &mut out).unwrap();
            assert_eq!(parse_str(&out).unwrap(), sample());
        }
    }

    #[test]
    fn deep_indentation_is_exact() {
        // 100 levels of 3 spaces: past the 64-space chunk.
        let depth = 100;
        let text = format!("{}1{}", "[".repeat(depth), "]".repeat(depth));
        let value = parse_str(&text).unwrap();
        let mut out = String::new();
        write(&value, Style { indent: Indent::Spaces(3), inline_scalar_arrays: false }, &mut out).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2 * depth + 1);
        for (i, line) in lines.iter().enumerate() {
            let level = if i <= depth { i } else { 2 * depth - i };
            let content = match i.cmp(&depth) {
                std::cmp::Ordering::Less => "[",
                std::cmp::Ordering::Equal => "1",
                std::cmp::Ordering::Greater => "]",
            };
            assert_eq!(*line, format!("{}{content}", " ".repeat(3 * level)), "line {i}");
        }
        assert_eq!(parse_str(&out).unwrap(), value);
    }

    #[test]
    fn strings_get_minimal_escapes() {
        let mut out = String::new();
        write_string(&mut out, "plain é€😀 / \u{7F} \u{2028}");
        assert_eq!(out, "\"plain é€😀 / \u{7F} \u{2028}\"");
        out.clear();
        write_string(&mut out, "\"\\\u{8}\u{C}\n\r\t");
        assert_eq!(out, r#""\"\\\b\f\n\r\t""#);
        out.clear();
        write_string(&mut out, "a\0b\u{1}\u{B}\u{1F}c\u{20}");
        assert_eq!(out, r#""a\u0000b\u0001\u000b\u001fc ""#);
        // Every control character round-trips, and the table is in order.
        for code in 0..0x20_u8 {
            let text = char::from(code).to_string();
            let mut out = String::new();
            write_string(&mut out, &text);
            assert!(out.len() == 4 || out == format!("\"\\u{code:04x}\""), "{out}");
            assert_eq!(parse_str(&out).unwrap().as_str(), Some(text.as_str()));
        }
        // Every scalar value round-trips through a string.
        let mut all = String::new();
        for code in (0..0x11_0000_u32).step_by(7) {
            if let Some(c) = char::from_u32(code) {
                all.push(c);
            }
        }
        let mut out = String::new();
        write_string(&mut out, &all);
        assert_eq!(parse_str(&out).unwrap().as_str(), Some(all.as_str()));
    }

    #[test]
    fn non_finite_numbers_are_refused() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let e = to_string(&Value::from(bad)).unwrap_err();
            assert_eq!(e, Error::Encode { format: Format::Json, what: "NaN and infinity have no JSON form" });
            assert!(to_string(&Value::from(vec![Value::from(1), Value::from(bad)])).is_err());
            let mut out = String::new();
            let mut w = JsonWriter::compact(&mut out);
            assert!(w.f64(bad).is_err());
            assert!(w.f32(bad as f32).is_err());
            // A refused value leaves the writer usable.
            w.f64(1.5).unwrap();
            w.finish().unwrap();
            assert_eq!(out, "1.5");
            let mut out = String::new();
            assert!(write_number(&mut out, Number::Float(bad)).is_err());
            assert!(out.is_empty());
        }
    }

    #[test]
    fn the_streaming_writer_builds_documents() {
        let mut out = String::new();
        let mut w = JsonWriter::pretty(&mut out);
        w.begin_object().unwrap();
        w.key("asset").unwrap().begin_object().unwrap();
        w.key("version").unwrap().string("2.0").unwrap();
        w.end_object().unwrap();
        w.key("translation").unwrap().begin_array_inline().unwrap();
        w.f32(0.1).unwrap().f32(-2.5).unwrap().f64(0.1).unwrap();
        w.end_array().unwrap();
        w.key("matrix").unwrap().begin_array_inline().unwrap();
        w.begin_array().unwrap().i64(1).unwrap().i64(0).unwrap().end_array().unwrap();
        w.begin_object().unwrap().key("k").unwrap().null().unwrap().end_object().unwrap();
        w.end_array().unwrap();
        w.key("count").unwrap().u64(u64::MAX).unwrap();
        w.key("flags").unwrap().begin_array().unwrap().bool(true).unwrap().null().unwrap().end_array().unwrap();
        w.key("sub").unwrap().value(&parse_str(r#"{"a":[1,2],"b":{}}"#).unwrap()).unwrap();
        assert_eq!(w.depth(), 1);
        w.end_object().unwrap();
        w.finish().unwrap();
        let expected = r#"{
  "asset": {
    "version": "2.0"
  },
  "translation": [0.1, -2.5, 0.1],
  "matrix": [[1, 0], {"k": null}],
  "count": 18446744073709551615,
  "flags": [
    true,
    null
  ],
  "sub": {
    "a": [
      1,
      2
    ],
    "b": {}
  }
}"#;
        assert_eq!(out, expected);
        assert!(parse_str(&out).is_ok());

        // The same calls in compact style, into bytes.
        let mut bytes = Vec::new();
        let mut w = JsonWriter::compact(&mut bytes);
        w.begin_array().unwrap().begin_array_inline().unwrap().i64(-1).unwrap().f32(1.0).unwrap().end_array().unwrap();
        w.begin_object().unwrap().key("a\n").unwrap().string("b\t").unwrap().end_object().unwrap().end_array().unwrap();
        w.finish().unwrap();
        assert_eq!(bytes, br#"[[-1,1.0],{"a\n":"b\t"}]"#);
    }

    #[test]
    fn the_streaming_writer_refuses_invalid_nesting() {
        fn check(build: impl Fn(&mut JsonWriter<'_, String>) -> Result<()>, expected: &str) {
            let mut out = String::new();
            let mut w = JsonWriter::compact(&mut out);
            match build(&mut w) {
                Err(Error::Encode { format: Format::Json, what }) => assert_eq!(what, expected),
                other => panic!("expected {expected:?}, got {other:?}"),
            }
        }
        check(|w| w.null()?.null().map(|_| ()), "a JSON text holds one top-level value");
        check(|w| w.begin_array()?.end_array()?.begin_array().map(|_| ()), "a JSON text holds one top-level value");
        check(|w| w.begin_object()?.null().map(|_| ()), "an object member needs a key before its value");
        check(|w| w.begin_object()?.begin_array().map(|_| ()), "an object member needs a key before its value");
        check(|w| w.begin_object()?.key("a")?.key("b").map(|_| ()), "an object key has no value");
        check(|w| w.begin_object()?.key("a")?.end_object().map(|_| ()), "an object key has no value");
        check(|w| w.begin_array()?.key("a").map(|_| ()), "a key is only allowed inside an object");
        check(|w| w.key("a").map(|_| ()), "a key is only allowed inside an object");
        check(|w| w.begin_array()?.end_object().map(|_| ()), "end_object inside an array");
        check(|w| w.begin_object()?.end_array().map(|_| ()), "end_array inside an object");
        check(|w| w.end_array().map(|_| ()), "no container is open");
        check(|w| w.null()?.end_object().map(|_| ()), "no container is open");
        let mut out = String::new();
        let mut w = JsonWriter::compact(&mut out);
        w.begin_array().unwrap();
        assert_eq!(w.finish(), Err(encode("a container is still open")));
        let mut out = String::new();
        assert_eq!(JsonWriter::compact(&mut out).finish(), Err(encode("no value was written")));
        // A refused call writes nothing.
        let mut out = String::new();
        let mut w = JsonWriter::pretty(&mut out);
        w.begin_object().unwrap();
        assert!(w.string("x").is_err());
        assert!(w.end_array().is_err());
        w.end_object().unwrap();
        w.finish().unwrap();
        assert_eq!(out, "{}");
    }
}
