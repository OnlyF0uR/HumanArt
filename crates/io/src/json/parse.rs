//! The strict RFC 8259 parser `[DEP-05]` `[ROB-01]` `[ROB-02]`.
//!
//! The input is a byte slice holding one JSON text. What is accepted is exactly the grammar of
//! RFC 8259, with these decisions on the points the RFC leaves open:
//!
//! - the text must be valid UTF-8; a UTF-8 byte order mark at the very start is skipped (the RFC lets
//!   a parser ignore one; editors on Windows still write them);
//! - any value may stand at the top level, not just an array or an object;
//! - **an object with two members of the same key is an error**, because every way of resolving it
//!   silently loses data;
//! - a `\u` escape must give a Unicode scalar value: a surrogate has to come as a high-low pair, a
//!   lone one is an error (it has no UTF-8 form);
//! - a number too large for an `f64` is an error (see [`number`](super::number)); integer literals
//!   that fit `i64` or `u64` stay integers;
//! - no comments, no trailing commas, no single quotes, no `NaN` or `Infinity`, no control characters
//!   inside strings, nothing but white space after the value.
//!
//! The parser is iterative: nesting lives on an explicit stack whose height is limited by
//! [`JsonLimits::max_depth`], so hostile nesting cannot exhaust the call stack. Every size is checked
//! against [`JsonLimits`] before anything is allocated for it, and nothing is ever allocated from a
//! count the input merely claims: memory grows only with bytes actually consumed. Errors carry the
//! byte offset and the line and column of the place the input went wrong.

use super::number::{self, ConvertError, ScanError};
use super::value::{Object, Value};
use crate::error::{Error, Format};
use crate::image::Limits;
use std::fmt;

/// The deepest nesting any limit can allow. Dropping, cloning and comparing a [`Value`] recurse once
/// per level, so the depth of what the parser builds has to stay well inside a small thread stack.
pub const MAX_DEPTH_CAP: u32 = 1024;

/// The size limits of one parse `[ROB-02]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JsonLimits {
    /// Deepest nesting of arrays and objects (`[[1]]` has depth 2). Values above
    /// [`MAX_DEPTH_CAP`] count as that.
    pub max_depth: u32,
    /// Largest input, in bytes.
    pub max_input_bytes: u64,
    /// Longest string or key, measured as the bytes between its quotes in the input (which is never
    /// less than its decoded length).
    pub max_string_bytes: u64,
    /// Most elements of one array or members of one object.
    pub max_members: u64,
    /// Most values in the whole text, counting every scalar, array and object. This bounds the
    /// memory of the tree: one value costs at most a small constant (under 200 bytes with its share
    /// of key index and array slack) plus its text.
    pub max_values: u64,
}

impl JsonLimits {
    /// The limits used by [`parse`]: 128 levels, 1 GiB of input, 256 MiB per string, 2^24 members per
    /// container, 2^26 values.
    pub const DEFAULT: Self = Self {
        max_depth: 128,
        max_input_bytes: 1 << 30,
        max_string_bytes: 1 << 28,
        max_members: 1 << 24,
        max_values: 1 << 26,
    };

    /// Small limits for fuzz tests and for untrusted peers: 64 levels, 16 MiB of input, 1 MiB per
    /// string, 2^16 members per container, 2^20 values.
    pub const FUZZ: Self = Self {
        max_depth: 64,
        max_input_bytes: 16 << 20,
        max_string_bytes: 1 << 20,
        max_members: 1 << 16,
        max_values: 1 << 20,
    };

    /// The JSON limits that correspond to the crate-wide [`Limits`]: its depth, its byte budget for
    /// the input and for one string, and its element count for one container and for the whole text.
    #[must_use]
    pub const fn from_limits(limits: &Limits) -> Self {
        Self {
            max_depth: limits.max_depth,
            max_input_bytes: limits.max_bytes,
            max_string_bytes: limits.max_bytes,
            max_members: limits.max_elements,
            max_values: limits.max_elements,
        }
    }
}

impl Default for JsonLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Why a JSON text was refused, and where.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    /// What is wrong: [`Error::Truncated`], [`Error::Invalid`] or [`Error::Limit`], with
    /// [`Format::Json`].
    pub error: Error,
    /// The byte offset in the input.
    pub offset: usize,
    /// The line of that offset, counting from 1; lines end at `\n`.
    pub line: usize,
    /// The column of that offset, counting characters from 1 (a multi-byte character is one column).
    pub column: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (line {}, column {})", self.error, self.line, self.column)
    }
}

impl std::error::Error for ParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Keeps the codec error and drops the line and column (the offset stays inside the codec error for
/// `Truncated` and `Invalid`).
impl From<ParseError> for Error {
    fn from(e: ParseError) -> Self {
        e.error
    }
}

/// The line and column (both from 1) of a byte offset in `input`, as [`ParseError`] reports them.
/// An offset past the end counts as the end.
#[must_use]
pub fn line_column(input: &[u8], offset: usize) -> (usize, usize) {
    let before = input.get(..offset).unwrap_or(input);
    let mut line = 1;
    let mut line_start = 0;
    for (i, &b) in before.iter().enumerate() {
        if b == b'\n' {
            line += 1;
            line_start = i + 1;
        }
    }
    let on_line = before.get(line_start..).unwrap_or_default();
    // UTF-8 continuation bytes do not start a character.
    let column = 1 + on_line.iter().filter(|&&b| b & 0xC0 != 0x80).count();
    (line, column)
}

/// Parses one JSON text with [`JsonLimits::DEFAULT`].
pub fn parse(input: &[u8]) -> Result<Value, ParseError> {
    parse_with(input, &JsonLimits::DEFAULT)
}

/// Parses one JSON text held in a string, with [`JsonLimits::DEFAULT`].
pub fn parse_str(input: &str) -> Result<Value, ParseError> {
    parse(input.as_bytes())
}

/// Parses one JSON text within the given limits.
pub fn parse_with(input: &[u8], limits: &JsonLimits) -> Result<Value, ParseError> {
    let mut parser = Parser { input, pos: 0, limits: *limits, values: 0 };
    parser.document().map_err(|stop| {
        let (offset, error) = match stop {
            Stop::Truncated(offset) => (offset, Error::Truncated { format: Format::Json, offset }),
            Stop::Invalid(offset, what) => (offset, Error::Invalid { format: Format::Json, offset, what }),
            Stop::Limit(offset, what, limit, got) => (offset, Error::Limit { format: Format::Json, what, limit, got }),
        };
        let offset = offset.min(input.len());
        let (line, column) = line_column(input, offset);
        ParseError { error, offset, line, column }
    })
}

impl std::str::FromStr for Value {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, ParseError> {
        parse_str(s)
    }
}

/// Why the parser stopped; turned into a [`ParseError`] once, at the top.
#[derive(Clone, Copy, Debug)]
enum Stop {
    /// The input ends at this offset, inside the text.
    Truncated(usize),
    /// The byte at this offset breaks the grammar.
    Invalid(usize, &'static str),
    /// A limit was exceeded at this offset: what, the limit, the value found.
    Limit(usize, &'static str, u64, u64),
}

/// A container that is still being filled.
enum Frame {
    Array(Vec<Value>),
    /// The object so far and the key whose value is being read.
    Object(Object, String),
}

/// What closing or continuing a container asks of the main loop.
enum Next {
    /// Another value follows.
    Value,
    /// The container on top of the stack is complete.
    Close,
}

struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
    limits: JsonLimits,
    /// Values begun so far.
    values: u64,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn skip_space(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.pos += 1;
        }
    }

    fn document(&mut self) -> Result<Value, Stop> {
        let length = self.input.len() as u64;
        if length > self.limits.max_input_bytes {
            return Err(Stop::Limit(0, "input size", self.limits.max_input_bytes, length));
        }
        if self.input.starts_with(&[0xEF, 0xBB, 0xBF]) {
            self.pos = 3;
        }
        let max_depth = self.limits.max_depth.min(MAX_DEPTH_CAP) as usize;
        let mut stack: Vec<Frame> = Vec::new();
        loop {
            // A value is due here.
            self.skip_space();
            let start = self.pos;
            self.values += 1;
            if self.values > self.limits.max_values {
                return Err(Stop::Limit(start, "value count", self.limits.max_values, self.values));
            }
            let mut value = match self.peek() {
                None => return Err(Stop::Truncated(start)),
                Some(open @ (b'[' | b'{')) => {
                    if stack.len() >= max_depth {
                        return Err(Stop::Limit(start, "nesting depth", max_depth as u64, stack.len() as u64 + 1));
                    }
                    self.pos += 1;
                    self.skip_space();
                    let close = if open == b'[' { b']' } else { b'}' };
                    if self.peek() == Some(close) {
                        self.pos += 1;
                        if open == b'[' { Value::Array(Vec::new()) } else { Value::Object(Object::new()) }
                    } else {
                        if open == b'[' {
                            stack.push(Frame::Array(Vec::new()));
                        } else {
                            let key = self.key(&Object::new())?;
                            stack.push(Frame::Object(Object::new(), key));
                        }
                        continue;
                    }
                }
                Some(b'"') => Value::String(self.string()?),
                Some(b'-' | b'0'..=b'9') => Value::Number(self.number()?),
                Some(b't') => self.literal(b"true", Value::Bool(true))?,
                Some(b'f') => self.literal(b"false", Value::Bool(false))?,
                Some(b'n') => self.literal(b"null", Value::Null)?,
                Some(b']') if matches!(stack.last(), Some(Frame::Array(items)) if !items.is_empty()) => {
                    return Err(Stop::Invalid(start, "trailing comma in an array"));
                }
                Some(_) => return Err(Stop::Invalid(start, "expected a value")),
            };
            // A value is complete: hand it to the container it belongs to, closing every container
            // that ends here.
            loop {
                let next = match stack.last_mut() {
                    None => {
                        self.skip_space();
                        if self.pos < self.input.len() {
                            return Err(Stop::Invalid(self.pos, "unexpected data after the JSON value"));
                        }
                        return Ok(value);
                    }
                    Some(Frame::Array(items)) => {
                        items.push(value);
                        self.skip_space();
                        match self.peek() {
                            Some(b',') => {
                                self.check_members(items.len())?;
                                self.pos += 1;
                                Next::Value
                            }
                            Some(b']') => {
                                self.pos += 1;
                                Next::Close
                            }
                            None => return Err(Stop::Truncated(self.pos)),
                            Some(_) => return Err(Stop::Invalid(self.pos, "expected ',' or ']' in an array")),
                        }
                    }
                    Some(Frame::Object(object, key)) => {
                        object.push_new(std::mem::take(key), value);
                        self.skip_space();
                        match self.peek() {
                            Some(b',') => {
                                self.check_members(object.len())?;
                                self.pos += 1;
                                self.skip_space();
                                *key = self.key(object)?;
                                Next::Value
                            }
                            Some(b'}') => {
                                self.pos += 1;
                                Next::Close
                            }
                            None => return Err(Stop::Truncated(self.pos)),
                            Some(_) => return Err(Stop::Invalid(self.pos, "expected ',' or '}' in an object")),
                        }
                    }
                };
                match next {
                    Next::Value => break,
                    Next::Close => {
                        value = match stack.pop() {
                            Some(Frame::Array(items)) => Value::Array(items),
                            Some(Frame::Object(object, _)) => Value::Object(object),
                            // The stack was not empty a moment ago; an empty document cannot close.
                            None => return Err(Stop::Invalid(self.pos, "unbalanced container")),
                        };
                    }
                }
            }
        }
    }

    /// Refuses a further member when a container already holds `len`.
    fn check_members(&self, len: usize) -> Result<(), Stop> {
        if len as u64 >= self.limits.max_members {
            return Err(Stop::Limit(self.pos, "member count", self.limits.max_members, len as u64 + 1));
        }
        Ok(())
    }

    /// Reads `"key" :` for the next member of `object` and leaves the position at its value.
    fn key(&mut self, object: &Object) -> Result<String, Stop> {
        let start = self.pos;
        let key = match self.peek() {
            Some(b'"') => self.string()?,
            None => return Err(Stop::Truncated(start)),
            Some(b'}') if !object.is_empty() => return Err(Stop::Invalid(start, "trailing comma in an object")),
            Some(_) => return Err(Stop::Invalid(start, "expected a string as object key")),
        };
        if object.contains_key(&key) {
            return Err(Stop::Invalid(start, "duplicate object key"));
        }
        self.skip_space();
        match self.peek() {
            Some(b':') => {
                self.pos += 1;
                Ok(key)
            }
            None => Err(Stop::Truncated(self.pos)),
            Some(_) => Err(Stop::Invalid(self.pos, "expected ':' after an object key")),
        }
    }

    fn literal(&mut self, word: &'static [u8], value: Value) -> Result<Value, Stop> {
        let rest = self.input.get(self.pos..).unwrap_or_default();
        if rest.starts_with(word) {
            self.pos += word.len();
            Ok(value)
        } else if word.starts_with(rest) {
            Err(Stop::Truncated(self.input.len()))
        } else {
            Err(Stop::Invalid(self.pos, "expected a value"))
        }
    }

    fn number(&mut self) -> Result<number::Number, Stop> {
        let start = self.pos;
        let (literal, end) = number::scan(self.input, start).map_err(|e| match e {
            ScanError::Truncated(at) => Stop::Truncated(at),
            ScanError::Invalid(at, what) => Stop::Invalid(at, what),
        })?;
        self.pos = end;
        number::convert(&literal).map_err(|e| match e {
            ConvertError::Overflow => Stop::Invalid(start, "number is too large for a 64-bit float"),
            ConvertError::TooLong => Stop::Invalid(start, "number is too long to convert"),
        })
    }

    /// Reads the string whose opening quote is at the position.
    fn string(&mut self) -> Result<String, Stop> {
        let start = self.pos + 1;
        let mut at = start;
        let mut escaped = false;
        loop {
            match self.input.get(at) {
                None => return Err(Stop::Truncated(self.input.len())),
                Some(b'"') => break,
                Some(b'\\') => {
                    // Whatever follows the backslash cannot end the string; the decoder checks it.
                    escaped = true;
                    at += 2;
                }
                Some(0..=0x1F) => return Err(Stop::Invalid(at, "control character in a string")),
                Some(_) => at += 1,
            }
        }
        let raw = self.input.get(start..at).unwrap_or_default();
        if raw.len() as u64 > self.limits.max_string_bytes {
            return Err(Stop::Limit(start, "string size", self.limits.max_string_bytes, raw.len() as u64));
        }
        self.pos = at + 1;
        if escaped { unescape(raw, start) } else { utf8(raw, start).map(str::to_owned) }
    }
}

fn utf8(raw: &[u8], base: usize) -> Result<&str, Stop> {
    std::str::from_utf8(raw).map_err(|e| Stop::Invalid(base + e.valid_up_to(), "invalid UTF-8 in a string"))
}

/// Four hexadecimal digits at `raw[at..]`; `base` is the input offset of `raw`.
fn hex4(raw: &[u8], at: usize, base: usize) -> Result<u32, Stop> {
    let mut code = 0_u32;
    for i in at..at + 4 {
        let digit = match raw.get(i) {
            Some(b @ b'0'..=b'9') => b - b'0',
            Some(b @ b'a'..=b'f') => b - b'a' + 10,
            Some(b @ b'A'..=b'F') => b - b'A' + 10,
            _ => return Err(Stop::Invalid(base + i, "expected four hexadecimal digits after \\u")),
        };
        code = code << 4 | u32::from(digit);
    }
    Ok(code)
}

/// Decodes the inside of a string that contains escapes. `base` is the input offset of `raw`.
fn unescape(raw: &[u8], base: usize) -> Result<String, Stop> {
    // Every escape is at least as long as the UTF-8 it stands for.
    let mut out = String::with_capacity(raw.len());
    let mut at = 0;
    while at < raw.len() {
        let run = raw.get(at..).unwrap_or_default();
        let plain = run.iter().position(|&b| b == b'\\').unwrap_or(run.len());
        if plain > 0 {
            out.push_str(utf8(run.get(..plain).unwrap_or_default(), base + at)?);
            at += plain;
            continue;
        }
        let decoded = match raw.get(at + 1) {
            Some(b'"') => '"',
            Some(b'\\') => '\\',
            Some(b'/') => '/',
            Some(b'b') => '\u{8}',
            Some(b'f') => '\u{C}',
            Some(b'n') => '\n',
            Some(b'r') => '\r',
            Some(b't') => '\t',
            Some(b'u') => {
                let high = hex4(raw, at + 2, base)?;
                let code = match high {
                    0xD800..=0xDBFF => {
                        if raw.get(at + 6) != Some(&b'\\') || raw.get(at + 7) != Some(&b'u') {
                            return Err(Stop::Invalid(base + at, "unpaired surrogate in a \\u escape"));
                        }
                        let low = hex4(raw, at + 8, base)?;
                        if !(0xDC00..=0xDFFF).contains(&low) {
                            return Err(Stop::Invalid(base + at, "unpaired surrogate in a \\u escape"));
                        }
                        at += 6;
                        0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
                    }
                    0xDC00..=0xDFFF => return Err(Stop::Invalid(base + at, "unpaired surrogate in a \\u escape")),
                    code => code,
                };
                at += 4;
                char::from_u32(code).ok_or(Stop::Invalid(base + at, "invalid \\u escape"))?
            }
            _ => return Err(Stop::Invalid(base + at + 1, "invalid escape in a string")),
        };
        out.push(decoded);
        at += 2;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::Number;

    fn fails(text: &[u8]) -> ParseError {
        match parse(text) {
            Err(e) => e,
            Ok(v) => panic!("{:?} was accepted as {v:?}", String::from_utf8_lossy(text)),
        }
    }

    fn what(text: &[u8]) -> &'static str {
        match fails(text).error {
            Error::Invalid { what, .. } | Error::Limit { what, .. } => what,
            Error::Truncated { .. } => "truncated",
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn scalars_and_containers_parse() {
        assert_eq!(parse(b"null").unwrap(), Value::Null);
        assert_eq!(parse(b" true ").unwrap(), Value::Bool(true));
        assert_eq!(parse(b"\tfalse\r\n").unwrap(), Value::Bool(false));
        assert_eq!(parse(b"42").unwrap(), Value::from(42));
        assert_eq!(parse(b"-0.5e1").unwrap(), Value::from(-5.0));
        assert_eq!(parse(b"\"hi\"").unwrap(), Value::from("hi"));
        assert_eq!(parse(b"[]").unwrap(), Value::Array(Vec::new()));
        assert_eq!(parse(b"{}").unwrap(), Value::Object(Object::new()));
        assert_eq!(parse(b"[ ]").unwrap(), Value::Array(Vec::new()));
        assert_eq!(parse(b"{ \n}").unwrap(), Value::Object(Object::new()));
        let doc = parse(br#" { "b" : [1, 2.0, "x", null, true, {"k": []}], "a": {"z": -1, "y": {}} } "#).unwrap();
        assert_eq!(doc.as_object().unwrap().keys().collect::<Vec<_>>(), ["b", "a"]);
        assert_eq!(doc["b"].as_array().unwrap().len(), 6);
        assert_eq!(doc["b"][0], Value::Number(Number::Int(1)));
        assert_eq!(doc["b"][1], Value::Number(Number::Float(2.0)));
        assert_eq!(doc["b"][5]["k"], Value::Array(Vec::new()));
        assert_eq!(doc["a"]["z"].as_i64(), Some(-1));
        assert_eq!(doc["a"].as_object().unwrap().keys().collect::<Vec<_>>(), ["z", "y"]);
        assert_eq!("[1]".parse::<Value>().unwrap(), Value::from(vec![1]));
        assert_eq!(parse_str("18446744073709551615").unwrap().as_u64(), Some(u64::MAX));
    }

    #[test]
    fn strings_decode_every_escape() {
        let v = parse(br#""\" \\ \/ \b \f \n \r \t \u0041 \u00e9 \u20AC \ud83d\ude00 \u0000""#).unwrap();
        assert_eq!(v.as_str().unwrap(), "\" \\ / \u{8} \u{C} \n \r \t A \u{E9} \u{20AC} \u{1F600} \0");
        assert_eq!(parse("\"é€😀\u{7F}\"".as_bytes()).unwrap().as_str().unwrap(), "é€😀\u{7F}");
        assert_eq!(parse(br#""\uDBFF\uDFFF""#).unwrap().as_str().unwrap(), "\u{10FFFF}");
        assert_eq!(parse(br#""\uFFFF""#).unwrap().as_str().unwrap(), "\u{FFFF}");
        assert_eq!(parse(br#""a\\""#).unwrap().as_str().unwrap(), "a\\");
        assert_eq!(parse(br#""\\\"""#).unwrap().as_str().unwrap(), "\\\"");
        assert_eq!(parse(br#"{"\u006b":1}"#).unwrap()["k"], Value::from(1));

        assert_eq!(what(br#""\x41""#), "invalid escape in a string");
        assert_eq!(what(br#""\u12G4""#), "expected four hexadecimal digits after \\u");
        assert_eq!(what(br#""\u12""#), "expected four hexadecimal digits after \\u");
        assert_eq!(what(br#""\uD800""#), "unpaired surrogate in a \\u escape");
        assert_eq!(what(br#""\uD800\n""#), "unpaired surrogate in a \\u escape");
        assert_eq!(what(br#""\uD800\u0041""#), "unpaired surrogate in a \\u escape");
        assert_eq!(what(br#""\uDC00\uD800""#), "unpaired surrogate in a \\u escape");
        assert_eq!(what(b"\"a\nb\""), "control character in a string");
        assert_eq!(what(b"\"a\0b\""), "control character in a string");
        assert_eq!(what(b"\"\\\n\""), "invalid escape in a string");
        assert_eq!(what(b"\"\xFF\""), "invalid UTF-8 in a string");
        assert_eq!(what(b"\"\xC0\x80\""), "invalid UTF-8 in a string");
        assert_eq!(what(b"\"\xED\xA0\x80\""), "invalid UTF-8 in a string");
        assert_eq!(what(b"\"ok\\n\xE2\x82\""), "invalid UTF-8 in a string");
        assert_eq!(what(b"\"abc"), "truncated");
        assert_eq!(what(b"\"abc\\"), "truncated");
        assert_eq!(what(b"\"abc\\\""), "truncated");
        // The offset points at the offending byte.
        assert_eq!(fails(b"[\"ab\xFFcd\"]").offset, 4);
        assert_eq!(fails(b"[\"a\\n\xFF\"]").offset, 5);
        assert_eq!(fails(br#"["ab\q"]"#).offset, 5);
        assert_eq!(fails(br#"["\u00zz"]"#).offset, 6);
    }

    #[test]
    fn the_grammar_is_strict() {
        for (text, expected) in [
            (&b""[..], "truncated"),
            (b"   ", "truncated"),
            (b"[", "truncated"),
            (b"[1", "truncated"),
            (b"[1,", "truncated"),
            (b"{", "truncated"),
            (b"{\"a\"", "truncated"),
            (b"{\"a\":", "truncated"),
            (b"{\"a\":1", "truncated"),
            (b"{\"a\":1,", "truncated"),
            (b"tru", "truncated"),
            (b"n", "truncated"),
            (b"-", "truncated"),
            (b"1e", "truncated"),
            (b"[1,]", "trailing comma in an array"),
            (b"{\"a\":1,}", "trailing comma in an object"),
            (b"[,1]", "expected a value"),
            (b"[1,,2]", "expected a value"),
            (b"]", "expected a value"),
            (b"}", "expected a value"),
            (b"[}", "expected a value"),
            (b"{]", "expected a string as object key"),
            (b"{,}", "expected a string as object key"),
            (b"{1:2}", "expected a string as object key"),
            (b"{'a':1}", "expected a string as object key"),
            (b"{a:1}", "expected a string as object key"),
            (b"{\"a\" 1}", "expected ':' after an object key"),
            (b"{\"a\",1}", "expected ':' after an object key"),
            (b"{\"a\":1 \"b\":2}", "expected ',' or '}' in an object"),
            (b"{\"a\":1]", "expected ',' or '}' in an object"),
            (b"[1 2]", "expected ',' or ']' in an array"),
            (b"[1}", "expected ',' or ']' in an array"),
            (b"[1:2]", "expected ',' or ']' in an array"),
            (b"1 2", "unexpected data after the JSON value"),
            (b"[] []", "unexpected data after the JSON value"),
            (b"{}x", "unexpected data after the JSON value"),
            (b"nullx", "unexpected data after the JSON value"),
            (b"1.5.2", "unexpected data after the JSON value"),
            (b"// c\n1", "expected a value"),
            (b"[1 /* c */]", "expected ',' or ']' in an array"),
            (b"'a'", "expected a value"),
            (b"NaN", "expected a value"),
            (b"Infinity", "expected a value"),
            (b"-Infinity", "expected a digit"),
            (b"nul!", "expected a value"),
            (b"True", "expected a value"),
            (b"+1", "expected a value"),
            (b".5", "expected a value"),
            (b"01", "a number has a leading zero"),
            (b"1.e2", "expected a digit after the decimal point"),
            (b"1e+x", "expected a digit in the exponent"),
            (b"1e400", "number is too large for a 64-bit float"),
            (b"[-1e999]", "number is too large for a 64-bit float"),
            (b"{\"a\":1,\"a\":2}", "duplicate object key"),
            (b"{\"a\":1,\"b\":2,\"\\u0061\":3}", "duplicate object key"),
            (b"\x0C1", "expected a value"),
            (b"\xC2\xA01", "expected a value"),
            (b"\xEF\xBB\xBF", "truncated"),
            (b"1\xEF\xBB\xBF", "unexpected data after the JSON value"),
            (b"\xEF\xBB\xBF\xEF\xBB\xBF1", "expected a value"),
        ] {
            assert_eq!(what(text), expected, "{:?}", String::from_utf8_lossy(text));
        }
        // A byte order mark at the start, and only there, is skipped.
        assert_eq!(parse(b"\xEF\xBB\xBF{\"a\":1}").unwrap()["a"], Value::from(1));
        // Duplicates are found in large objects too, through the index.
        let mut text = String::from("{");
        for i in 0..200 {
            text.push_str("\"k");
            text.push_str(&i.to_string());
            text.push_str("\":0,");
        }
        text.push_str("\"k57\":0}");
        let e = fails(text.as_bytes());
        assert_eq!(
            e.error,
            Error::Invalid { format: Format::Json, offset: text.len() - 8, what: "duplicate object key" }
        );
    }

    #[test]
    fn errors_say_where() {
        let e = fails(b"{\n  \"a\": [1, 2,\n  \"b\": x\n}");
        assert_eq!((e.offset, e.line, e.column), (21, 3, 6));
        assert_eq!(e.to_string(), "JSON: expected ',' or ']' in an array at byte 21 (line 3, column 6)");
        assert_eq!(Error::from(e.clone()), e.error);
        assert!(std::error::Error::source(&e).is_some());
        // Columns count characters, not bytes; \r\n is one line break.
        let e = fails("[\"é€😀\", x]".as_bytes());
        assert_eq!((e.offset, e.line, e.column), (14, 1, 9));
        let e = fails(b"[1,\r\n2,\r\n?]");
        assert_eq!((e.offset, e.line, e.column), (9, 3, 1));
        let e = fails(b"[1, 2");
        assert_eq!((e.offset, e.line, e.column), (5, 1, 6));
        assert_eq!(e.to_string(), "JSON: input ends early at byte 5 (line 1, column 6)");
        assert_eq!(line_column(b"ab\ncd", 99), (2, 3));
        assert_eq!(line_column(b"", 0), (1, 1));
    }

    #[test]
    fn rob_02_limits_are_enforced() {
        let limits =
            JsonLimits { max_depth: 3, max_input_bytes: 64, max_string_bytes: 5, max_members: 3, max_values: 9 };
        let limit = |text: &[u8]| match parse_with(text, &limits) {
            Err(ParseError { error: Error::Limit { what, limit, got, format: Format::Json }, .. }) => {
                (what, limit, got)
            }
            other => panic!("{other:?}"),
        };
        assert!(parse_with(b"[[[1]]]", &limits).is_ok());
        assert!(parse_with(b"[[[]],{\"a\":{}}]", &limits).is_ok());
        assert_eq!(limit(b"[[[[1]]]]"), ("nesting depth", 3, 4));
        assert_eq!(limit(b"[{\"a\":[{\"b\":1}]}]"), ("nesting depth", 3, 4));
        assert!(parse_with(b"\"12345\"", &limits).is_ok());
        assert_eq!(limit(b"\"123456\""), ("string size", 5, 6));
        assert_eq!(limit(b"{\"123456\":1}"), ("string size", 5, 6));
        assert!(parse_with(b"[1,2,3]", &limits).is_ok());
        assert_eq!(limit(b"[1,2,3,4]"), ("member count", 3, 4));
        assert!(parse_with(b"{\"a\":1,\"b\":2,\"c\":3}", &limits).is_ok());
        assert_eq!(limit(b"{\"a\":1,\"b\":2,\"c\":3,\"d\":4}"), ("member count", 3, 4));
        assert!(parse_with(b"[[1,2,3],[4,5,6]]", &limits).is_ok());
        assert_eq!(limit(b"[[1,2,3],[4,5,6],7]"), ("value count", 9, 10));
        assert_eq!(limit(&[b' '; 65]), ("input size", 64, 65));
        // Limit errors still say where.
        let e = parse_with(b"[1,\n [[[2]]]]", &limits).unwrap_err();
        assert_eq!((e.offset, e.line, e.column), (7, 2, 4));
        assert_eq!(e.to_string(), "JSON: nesting depth is 4, over the limit of 3 (line 2, column 4)");

        assert_eq!(JsonLimits::default(), JsonLimits::DEFAULT);
        let from = JsonLimits::from_limits(&Limits::FUZZ);
        assert_eq!((from.max_depth, from.max_values, from.max_input_bytes), (64, 1 << 20, 64 << 20));
    }

    #[test]
    fn deep_nesting_is_refused_without_recursion() {
        // [ROB-01] A megabyte of brackets comes back as an error at once.
        for open in [&b"["[..], b"{\"a\":", b"[{\"k\":["] {
            let text = open.repeat(1_000_000 / open.len());
            let e = parse(&text).unwrap_err();
            assert!(matches!(e.error, Error::Limit { what: "nesting depth", limit: 128, got: 129, .. }), "{e}");
        }
        // The default depth is fine, one more is not.
        let ok = [b"[".repeat(128), b"]".repeat(128)].concat();
        assert!(parse(&ok).is_ok());
        let over = [b"[".repeat(129), b"]".repeat(129)].concat();
        assert!(parse(&over).is_err());
        // No limit goes beyond the cap, and a value that deep can be cloned, compared and dropped.
        let huge = JsonLimits { max_depth: u32::MAX, ..JsonLimits::DEFAULT };
        let cap = MAX_DEPTH_CAP as usize;
        let at_cap = [b"[".repeat(cap), b"]".repeat(cap)].concat();
        let value = parse_with(&at_cap, &huge).unwrap();
        assert_eq!(value.clone(), value);
        let over_cap = [b"[".repeat(cap + 1), b"]".repeat(cap + 1)].concat();
        assert!(matches!(parse_with(&over_cap, &huge).unwrap_err().error, Error::Limit { what: "nesting depth", .. }));
    }

    #[test]
    fn huge_counts_fail_fast_and_small() {
        // [ROB-01] Five million elements against a limit of a thousand: the parser stops at the
        // limit, having built a thousand values, not five million.
        let limits = JsonLimits { max_members: 1000, ..JsonLimits::FUZZ };
        let mut text = Vec::with_capacity(10_000_002);
        text.push(b'[');
        for _ in 0..5_000_000 {
            text.extend_from_slice(b"0,");
        }
        text.extend_from_slice(b"0]");
        let e = parse_with(&text, &limits).unwrap_err();
        assert_eq!(e.error, Error::Limit { format: Format::Json, what: "member count", limit: 1000, got: 1001 });
        assert_eq!(e.offset, 2000);
        let limits = JsonLimits { max_values: 5000, ..JsonLimits::FUZZ };
        let e = parse_with(&text, &limits).unwrap_err();
        assert_eq!(e.error, Error::Limit { format: Format::Json, what: "value count", limit: 5000, got: 5001 });
        assert!(e.offset < 10_002);
        // A string longer than the limit is refused before it is copied.
        let mut text = vec![b'"'; 1];
        text.resize(3_000_000, b'a');
        text.push(b'"');
        let e = parse_with(&text, &JsonLimits::FUZZ).unwrap_err();
        assert_eq!(e.error, Error::Limit { format: Format::Json, what: "string size", limit: 1 << 20, got: 2_999_999 });
    }
}
