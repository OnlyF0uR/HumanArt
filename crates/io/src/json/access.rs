//! Typed reading of a parsed document, with errors that say where `[DEP-05]`.
//!
//! There is no serde: every document and protocol type of the project is read by hand through this
//! module. A [`Node`] is a value together with the path that led to it; asking it for a member, an
//! element or a typed scalar either succeeds or returns an [`AccessError`] that names the path
//! (`$.lods[2].vertex_count`) and what was wrong. Nothing is allocated while access succeeds: the
//! path is a chain of borrowed steps on the stack and is only turned into text for an error.
//!
//! ```
//! use humanart_io::json::{self, AccessError, FromJson, Node};
//!
//! struct Lod {
//!     name: String,
//!     vertex_count: u32,
//!     scale: [f32; 3],
//!     weight: f32,
//! }
//!
//! impl FromJson<'_> for Lod {
//!     fn from_json(node: &Node<'_, '_>) -> Result<Self, AccessError> {
//!         node.deny_unknown(&["name", "vertex_count", "scale", "weight"])?;
//!         Ok(Self {
//!             name: node.get("name")?,
//!             vertex_count: node.get("vertex_count")?,
//!             scale: node.get_or("scale", [1.0; 3])?,
//!             weight: node.get_or("weight", 1.0)?,
//!         })
//!     }
//! }
//!
//! let doc = json::parse_str(r#"{"lods": [{"name": "lod0", "vertex_count": 18000}, {"name": "lod1"}]}"#)?;
//! let root = doc.node();
//! let first: Lod = root.field("lods")?.at(0)?.read()?;
//! assert_eq!((first.name.as_str(), first.vertex_count, first.scale), ("lod0", 18000, [1.0; 3]));
//! let error = root.get::<Vec<Lod>>("lods").err().map(|e| e.to_string());
//! assert_eq!(error.as_deref(), Some("$.lods[1].vertex_count: required member is missing"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Conventions: [`Node::get`] requires the member, [`Node::get_opt`] and [`Node::get_or`] treat a
//! missing member and a `null` one alike (producers differ in which they write). Integers are
//! range-checked into the asked type (`300` is not a `u8`); a float that is a whole number is accepted
//! as an integer (`3.0` reads as `3`), since other tools write integers that way. `f32` accepts any
//! number whose rounding to `f32` is finite.

use super::number::Number;
use super::value::{Kind, Value};
use super::write::write_string;
use std::fmt;

/// One step on the path from the root to a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step<'v> {
    /// The root itself.
    Root,
    /// The member with this key.
    Key(&'v str),
    /// The element with this index.
    Index(usize),
}

/// A value of a document and the path that led to it, for typed reading with located errors.
///
/// `'v` is the lifetime of the document, `'p` the lifetime of the parent node the path continues
/// from. Children borrow their parent, so a node is used where it was made, typically in one
/// `from_json` function; the values it hands out ([`value`](Self::value), [`str`](Self::str)) live
/// as long as the document.
#[derive(Clone, Copy)]
pub struct Node<'v, 'p> {
    value: &'v Value,
    step: Step<'v>,
    parent: Option<&'p Node<'v, 'p>>,
}

impl Value {
    /// This value as the root of typed access: errors below it start their path at `$`.
    #[must_use]
    pub const fn node(&self) -> Node<'_, '_> {
        Node::root(self)
    }
}

impl<'v> Node<'v, '_> {
    /// `value` as the root of a path (`$`).
    #[must_use]
    pub const fn root(value: &'v Value) -> Node<'v, 'v> {
        Node { value, step: Step::Root, parent: None }
    }

    /// The value itself.
    #[must_use]
    pub const fn value(&self) -> &'v Value {
        self.value
    }

    /// The kind of the value.
    #[must_use]
    pub const fn kind(&self) -> Kind {
        self.value.kind()
    }

    /// Whether the value is `null`.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        self.value.is_null()
    }

    /// The path of this node as text: `$`, `$.name`, `$.lods[2]`, `$["a key"]`. Allocates; meant for
    /// messages.
    #[must_use]
    pub fn path(&self) -> String {
        let mut steps = Vec::new();
        let mut cursor = Some(self);
        while let Some(node) = cursor {
            steps.push(node.step);
            cursor = node.parent;
        }
        let mut text = String::new();
        for step in steps.iter().rev() {
            push_step(&mut text, *step);
        }
        text
    }

    /// An error at this node. Use it for checks the types cannot express:
    /// `return Err(node.error(What::Invalid("must be positive")))`.
    #[must_use]
    pub fn error(&self, what: What) -> AccessError {
        AccessError { path: self.path(), what }
    }

    /// An [`What::Invalid`] error at this node, with the given reason.
    #[must_use]
    pub fn invalid(&self, why: &'static str) -> AccessError {
        self.error(What::Invalid(why))
    }

    fn kind_error(&self, expected: &'static str) -> AccessError {
        self.error(What::Kind { expected, found: self.kind() })
    }

    fn child<'a>(&'a self, value: &'v Value, step: Step<'v>) -> Node<'v, 'a> {
        Node { value, step, parent: Some(self) }
    }

    // --- Navigation ---------------------------------------------------------------------------------

    /// The member `key` of this object. Errors when this is not an object or has no such member.
    pub fn field<'a>(&'a self, key: &str) -> Result<Node<'v, 'a>, AccessError> {
        let object = self.value.as_object().ok_or_else(|| self.kind_error("an object"))?;
        if let Some((stored, value)) = object.position(key).and_then(|at| object.get_index(at)) {
            return Ok(self.child(value, Step::Key(stored)));
        }
        let mut path = self.path();
        push_step(&mut path, Step::Key(key));
        Err(AccessError { path, what: What::Missing })
    }

    /// The member `key` of this object, or `None` when it is missing or `null`. Errors when this is
    /// not an object.
    pub fn opt_field<'a>(&'a self, key: &str) -> Result<Option<Node<'v, 'a>>, AccessError> {
        let object = self.value.as_object().ok_or_else(|| self.kind_error("an object"))?;
        Ok(object
            .position(key)
            .and_then(|at| object.get_index(at))
            .filter(|(_, value)| !value.is_null())
            .map(|(stored, value)| self.child(value, Step::Key(stored))))
    }

    /// Whether this is an object with a member `key` (of any value, `null` included).
    #[must_use]
    pub fn has(&self, key: &str) -> bool {
        self.value.as_object().is_some_and(|object| object.contains_key(key))
    }

    /// The element `index` of this array. Errors when this is not an array or is too short.
    pub fn at<'a>(&'a self, index: usize) -> Result<Node<'v, 'a>, AccessError> {
        let items = self.value.as_array().ok_or_else(|| self.kind_error("an array"))?;
        match items.get(index) {
            Some(value) => Ok(self.child(value, Step::Index(index))),
            None => Err(self.error(What::Index { index, len: items.len() })),
        }
    }

    /// The number of elements of this array.
    pub fn array_len(&self) -> Result<usize, AccessError> {
        self.value.as_array().map(<[Value]>::len).ok_or_else(|| self.kind_error("an array"))
    }

    /// The elements of this array, in order, each with its path.
    pub fn elements<'a>(&'a self) -> Result<Elements<'v, 'a>, AccessError> {
        let items = self.value.as_array().ok_or_else(|| self.kind_error("an array"))?;
        Ok(Elements { parent: self, items: items.iter().enumerate() })
    }

    /// The members of this object, in order, each value with its path.
    pub fn members<'a>(&'a self) -> Result<Members<'v, 'a>, AccessError> {
        let object = self.value.as_object().ok_or_else(|| self.kind_error("an object"))?;
        Ok(Members { parent: self, entries: object.entries().iter() })
    }

    /// Errors at the first member of this object whose key is not in `known`; for formats that
    /// refuse what they do not understand. Errors when this is not an object.
    pub fn deny_unknown(&self, known: &[&str]) -> Result<(), AccessError> {
        let object = self.value.as_object().ok_or_else(|| self.kind_error("an object"))?;
        match object.keys().find(|key| !known.contains(key)) {
            Some(key) => {
                let mut path = self.path();
                push_step(&mut path, Step::Key(key));
                Err(AccessError { path, what: What::UnknownMember })
            }
            None => Ok(()),
        }
    }

    // --- Scalars ------------------------------------------------------------------------------------

    /// The boolean.
    pub fn bool(&self) -> Result<bool, AccessError> {
        self.value.as_bool().ok_or_else(|| self.kind_error("a boolean"))
    }

    /// The string, borrowed from the document.
    pub fn str(&self) -> Result<&'v str, AccessError> {
        self.value.as_str().ok_or_else(|| self.kind_error("a string"))
    }

    /// The number as stored.
    pub fn number(&self) -> Result<Number, AccessError> {
        self.value.as_number().ok_or_else(|| self.kind_error("a number"))
    }

    /// The number as `f64` (integers beyond 2^53 round).
    pub fn f64(&self) -> Result<f64, AccessError> {
        self.number().map(Number::as_f64)
    }

    /// The number rounded to `f32`. Errors when it is beyond the `f32` range.
    #[allow(clippy::cast_possible_truncation)]
    pub fn f32(&self) -> Result<f32, AccessError> {
        // Rounding to the nearest f32 is the point; out-of-range values turn infinite and are refused.
        let value = self.f64()? as f32;
        if value.is_finite() { Ok(value) } else { Err(self.error(What::Range { expected: "a finite f32" })) }
    }

    /// The number as `i64`: an integer, or a float that is a whole number, in range.
    pub fn i64(&self) -> Result<i64, AccessError> {
        self.number()?.as_i64().ok_or_else(|| self.error(What::Range { expected: "an i64 integer" }))
    }

    /// The number as `u64`: a non-negative integer, or a float that is a whole number, in range.
    pub fn u64(&self) -> Result<u64, AccessError> {
        self.number()?.as_u64().ok_or_else(|| self.error(What::Range { expected: "a u64 integer" }))
    }

    /// The string, which must be one of the names in `choices`; returns the value paired with it.
    /// For enumerations stored as names: `node.choice(&[("linear", Filter::Linear), ("nearest",
    /// Filter::Nearest)])`.
    pub fn choice<T: Copy>(&self, choices: &[(&str, T)]) -> Result<T, AccessError> {
        let text = self.str()?;
        if let Some((_, value)) = choices.iter().find(|(name, _)| *name == text) {
            return Ok(*value);
        }
        let mut expected = String::new();
        for (i, (name, _)) in choices.iter().enumerate() {
            if i > 0 {
                expected.push_str(", ");
            }
            write_string(&mut expected, name);
        }
        Err(self.error(What::Choice { found: text.to_owned(), expected }))
    }

    // --- Typed --------------------------------------------------------------------------------------

    /// This value as `T`.
    pub fn read<T: FromJson<'v>>(&self) -> Result<T, AccessError> {
        T::from_json(self)
    }

    /// The required member `key` as `T`.
    pub fn get<T: FromJson<'v>>(&self, key: &str) -> Result<T, AccessError> {
        T::from_json(&self.field(key)?)
    }

    /// The member `key` as `T`, or `None` when it is missing or `null`.
    pub fn get_opt<T: FromJson<'v>>(&self, key: &str) -> Result<Option<T>, AccessError> {
        match self.opt_field(key)? {
            Some(node) => T::from_json(&node).map(Some),
            None => Ok(None),
        }
    }

    /// The member `key` as `T`, or `default` when it is missing or `null`.
    pub fn get_or<T: FromJson<'v>>(&self, key: &str, default: T) -> Result<T, AccessError> {
        Ok(self.get_opt(key)?.unwrap_or(default))
    }

    /// Reads this array into `out`, which must have exactly as many slots as the array has
    /// elements. Fills existing storage (a vertex buffer, a matrix) without allocating.
    pub fn read_into<T: FromJson<'v>>(&self, out: &mut [T]) -> Result<(), AccessError> {
        let len = self.array_len()?;
        if len != out.len() {
            return Err(self.error(What::Length { expected: out.len(), found: len }));
        }
        for (slot, node) in out.iter_mut().zip(self.elements()?) {
            *slot = T::from_json(&node)?;
        }
        Ok(())
    }

    /// Appends the elements of this array to `out`, each read as `T`. Reserves once.
    pub fn extend_into<T: FromJson<'v>>(&self, out: &mut Vec<T>) -> Result<(), AccessError> {
        let elements = self.elements()?;
        out.reserve(elements.len());
        for node in elements {
            out.push(T::from_json(&node)?);
        }
        Ok(())
    }
}

impl fmt::Debug for Node<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node").field("path", &self.path()).field("value", self.value).finish()
    }
}

/// Appends one path step: `$`, `.key` for identifier-like keys, `["key"]` for the rest, `[3]`.
fn push_step(text: &mut String, step: Step<'_>) {
    match step {
        Step::Root => text.push('$'),
        Step::Key(key) => {
            let mut chars = key.chars();
            let plain = chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
            if plain {
                text.push('.');
                text.push_str(key);
            } else {
                text.push('[');
                write_string(text, key);
                text.push(']');
            }
        }
        Step::Index(index) => {
            text.push('[');
            text.push_str(&index.to_string());
            text.push(']');
        }
    }
}

/// The iterator of [`Node::elements`].
#[derive(Clone)]
pub struct Elements<'v, 'a> {
    parent: &'a Node<'v, 'a>,
    items: std::iter::Enumerate<std::slice::Iter<'v, Value>>,
}

impl<'v, 'a> Iterator for Elements<'v, 'a> {
    type Item = Node<'v, 'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.items.next().map(|(index, value)| self.parent.child(value, Step::Index(index)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.items.size_hint()
    }
}

impl ExactSizeIterator for Elements<'_, '_> {}

/// The iterator of [`Node::members`].
#[derive(Clone)]
pub struct Members<'v, 'a> {
    parent: &'a Node<'v, 'a>,
    entries: std::slice::Iter<'v, (String, Value)>,
}

impl<'v, 'a> Iterator for Members<'v, 'a> {
    type Item = (&'v str, Node<'v, 'a>);

    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next().map(|(key, value)| (key.as_str(), self.parent.child(value, Step::Key(key))))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.entries.size_hint()
    }
}

impl ExactSizeIterator for Members<'_, '_> {}

// --- Errors ----------------------------------------------------------------------------------------

/// What was wrong at the place an [`AccessError`] names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum What {
    /// A required member is missing (the path names the member).
    Missing,
    /// The value has the wrong kind.
    Kind {
        /// What was asked for ("a string").
        expected: &'static str,
        /// What is there.
        found: Kind,
    },
    /// The number does not fit the asked type.
    Range {
        /// The asked type ("a u8 integer").
        expected: &'static str,
    },
    /// The array is shorter than the index asked for.
    Index {
        /// The index.
        index: usize,
        /// The length of the array.
        len: usize,
    },
    /// The array has the wrong number of elements.
    Length {
        /// The number required.
        expected: usize,
        /// The number present.
        found: usize,
    },
    /// The object has a member the format does not define (the path names the member).
    UnknownMember,
    /// The string is not one of the allowed names.
    Choice {
        /// The string found.
        found: String,
        /// The allowed names, quoted and comma-separated.
        expected: String,
    },
    /// The value breaks a rule of the format (the reason is given by the reader).
    Invalid(&'static str),
}

impl fmt::Display for What {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing => f.write_str("required member is missing"),
            Self::Kind { expected, found } => write!(f, "expected {expected}, found {found}"),
            Self::Range { expected } => write!(f, "number does not fit {expected}"),
            Self::Index { index, len } => write!(f, "index {index} is past the end of an array of {len}"),
            Self::Length { expected, found } => write!(f, "expected {expected} elements, found {found}"),
            Self::UnknownMember => f.write_str("unknown member"),
            Self::Choice { found, expected } => {
                f.write_str("unknown name ")?;
                let mut quoted = String::new();
                write_string(&mut quoted, found);
                write!(f, "{quoted}, expected one of {expected}")
            }
            Self::Invalid(why) => f.write_str(why),
        }
    }
}

/// A value of a document is not what its reader asks for: where (a path such as `$.lods[2].name`)
/// and what.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AccessError {
    /// The path of the offending value, as [`Node::path`] writes it.
    pub path: String,
    /// What is wrong there.
    pub what: What,
}

impl fmt::Display for AccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.what)
    }
}

impl std::error::Error for AccessError {}

// --- FromJson --------------------------------------------------------------------------------------

/// A type that can be read from a JSON value. Implement it for every document and protocol type;
/// [`Node::read`], [`Node::get`] and the container impls (`Vec<T>`, `[T; N]`, `Option<T>`) then
/// work with it.
pub trait FromJson<'v>: Sized {
    /// Reads `node` as `Self`, reporting problems at their path.
    fn from_json(node: &Node<'v, '_>) -> Result<Self, AccessError>;
}

impl<'v> FromJson<'v> for bool {
    fn from_json(node: &Node<'v, '_>) -> Result<Self, AccessError> {
        node.bool()
    }
}

impl<'v> FromJson<'v> for f64 {
    fn from_json(node: &Node<'v, '_>) -> Result<Self, AccessError> {
        node.f64()
    }
}

impl<'v> FromJson<'v> for f32 {
    fn from_json(node: &Node<'v, '_>) -> Result<Self, AccessError> {
        node.f32()
    }
}

impl<'v> FromJson<'v> for i64 {
    fn from_json(node: &Node<'v, '_>) -> Result<Self, AccessError> {
        node.i64()
    }
}

impl<'v> FromJson<'v> for u64 {
    fn from_json(node: &Node<'v, '_>) -> Result<Self, AccessError> {
        node.u64()
    }
}

macro_rules! from_json_int {
    ($($ty:ty => $via:ident, $expected:literal;)*) => {$(
        impl<'v> FromJson<'v> for $ty {
            fn from_json(node: &Node<'v, '_>) -> Result<Self, AccessError> {
                let range = || node.error(What::Range { expected: $expected });
                let wide = node.number()?.$via().ok_or_else(range)?;
                <$ty>::try_from(wide).map_err(|_| range())
            }
        }
    )*};
}

from_json_int! {
    i8 => as_i64, "an i8 integer";
    i16 => as_i64, "an i16 integer";
    i32 => as_i64, "an i32 integer";
    isize => as_i64, "an isize integer";
    u8 => as_u64, "a u8 integer";
    u16 => as_u64, "a u16 integer";
    u32 => as_u64, "a u32 integer";
    usize => as_u64, "a usize integer";
}

impl<'v> FromJson<'v> for Number {
    fn from_json(node: &Node<'v, '_>) -> Result<Self, AccessError> {
        node.number()
    }
}

impl<'v> FromJson<'v> for &'v str {
    fn from_json(node: &Node<'v, '_>) -> Result<Self, AccessError> {
        node.str()
    }
}

impl<'v> FromJson<'v> for String {
    fn from_json(node: &Node<'v, '_>) -> Result<Self, AccessError> {
        node.str().map(str::to_owned)
    }
}

/// Any value, borrowed.
impl<'v> FromJson<'v> for &'v Value {
    fn from_json(node: &Node<'v, '_>) -> Result<Self, AccessError> {
        Ok(node.value())
    }
}

/// Any value, copied.
impl<'v> FromJson<'v> for Value {
    fn from_json(node: &Node<'v, '_>) -> Result<Self, AccessError> {
        Ok(node.value().clone())
    }
}

/// `null` is `None`; anything else must read as `T`.
impl<'v, T: FromJson<'v>> FromJson<'v> for Option<T> {
    fn from_json(node: &Node<'v, '_>) -> Result<Self, AccessError> {
        if node.is_null() { Ok(None) } else { T::from_json(node).map(Some) }
    }
}

/// An array of any length.
impl<'v, T: FromJson<'v>> FromJson<'v> for Vec<T> {
    fn from_json(node: &Node<'v, '_>) -> Result<Self, AccessError> {
        let mut out = Vec::new();
        node.extend_into(&mut out)?;
        Ok(out)
    }
}

/// An array of exactly `N` elements: vectors (`[f32; 3]`), quaternions, matrices (`[f32; 16]`).
impl<'v, T: FromJson<'v>, const N: usize> FromJson<'v> for [T; N] {
    fn from_json(node: &Node<'v, '_>) -> Result<Self, AccessError> {
        let len = node.array_len()?;
        if len != N {
            return Err(node.error(What::Length { expected: N, found: len }));
        }
        let mut slots: [Option<T>; N] = std::array::from_fn(|_| None);
        for (slot, element) in slots.iter_mut().zip(node.elements()?) {
            *slot = Some(T::from_json(&element)?);
        }
        #[allow(clippy::expect_used)]
        let array = slots.map(|slot| {
            // INVARIANT: the array has exactly N elements (checked above) and each was read into its
            // slot, or the loop returned the error.
            slot.expect("every slot of a checked-length array is filled")
        });
        Ok(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse_str;

    fn doc(text: &str) -> Value {
        parse_str(text).unwrap()
    }

    #[test]
    fn scalars_read_with_range_checks() {
        let v = doc(r#"{"b": true, "s": "hi", "i": -5, "u": 300, "f": 2.5, "w": 3.0, "big": 1e300, "n": null}"#);
        let n = v.node();
        assert!(n.get::<bool>("b").unwrap());
        assert_eq!(n.get::<&str>("s").unwrap(), "hi");
        assert_eq!(n.get::<String>("s").unwrap(), "hi");
        assert_eq!(n.get::<i8>("i").unwrap(), -5);
        assert_eq!(n.get::<i64>("i").unwrap(), -5);
        assert_eq!(n.get::<u16>("u").unwrap(), 300);
        assert_eq!(n.get::<f32>("f").unwrap(), 2.5);
        assert_eq!(n.get::<f64>("u").unwrap(), 300.0);
        // A whole float reads as an integer; a fractional one does not.
        assert_eq!(n.get::<u8>("w").unwrap(), 3);
        assert_eq!(n.get::<u8>("f").unwrap_err().to_string(), "$.f: number does not fit a u8 integer");
        assert_eq!(n.get::<u8>("u").unwrap_err().to_string(), "$.u: number does not fit a u8 integer");
        assert_eq!(n.get::<u32>("i").unwrap_err().what, What::Range { expected: "a u32 integer" });
        assert_eq!(n.get::<f32>("big").unwrap_err().what, What::Range { expected: "a finite f32" });
        assert_eq!(n.get::<f64>("big").unwrap(), 1e300);
        assert_eq!(n.get::<Number>("i").unwrap(), Number::Int(-5));
        assert_eq!(n.get::<bool>("s").unwrap_err().to_string(), "$.s: expected a boolean, found string");
        assert_eq!(n.get::<Option<u8>>("n").unwrap(), None);
        assert_eq!(n.get::<Option<u8>>("w").unwrap(), Some(3));
        assert_eq!(n.get::<&Value>("n").unwrap(), &Value::Null);
        assert_eq!(n.get::<Value>("b").unwrap(), Value::Bool(true));
    }

    #[test]
    fn optional_members_and_defaults() {
        let v = doc(r#"{"a": 1, "n": null}"#);
        let n = v.node();
        assert_eq!(n.get_opt::<u32>("a").unwrap(), Some(1));
        assert_eq!(n.get_opt::<u32>("n").unwrap(), None);
        assert_eq!(n.get_opt::<u32>("missing").unwrap(), None);
        assert_eq!(n.get_or("missing", 7_u32).unwrap(), 7);
        assert_eq!(n.get_or("n", 7_u32).unwrap(), 7);
        assert_eq!(n.get_or("a", 7_u32).unwrap(), 1);
        // A present member of the wrong type is still an error.
        assert!(n.get_or("a", String::new()).is_err());
        assert!(n.has("n") && !n.has("missing"));
        assert!(n.opt_field("n").unwrap().is_none());
        let e = n.get::<u32>("missing").unwrap_err();
        assert_eq!((e.path.as_str(), &e.what), ("$.missing", &What::Missing));
        // Asking a non-object for a member names the non-object.
        let e = n.field("a").unwrap().get::<u32>("x").unwrap_err();
        assert_eq!(e.to_string(), "$.a: expected an object, found number");
    }

    #[test]
    fn paths_name_keys_and_indices() {
        let v = doc(r#"{"lods": [{"name": "a"}, {"name": 5}], "odd key": {"é": [[0, "x"]]}}"#);
        let root = v.node();
        let lods = root.field("lods").unwrap();
        let second = lods.at(1).unwrap();
        assert_eq!(second.path(), "$.lods[1]");
        assert_eq!(
            second.get::<String>("name").unwrap_err().to_string(),
            "$.lods[1].name: expected a string, found number"
        );
        let e = root.field("odd key").unwrap().field("é").unwrap().at(0).unwrap().at(1).unwrap().i64().unwrap_err();
        assert_eq!(e.path, r#"$["odd key"]["é"][0][1]"#);
        assert_eq!(lods.at(5).unwrap_err().to_string(), "$.lods: index 5 is past the end of an array of 2");
        assert_eq!(root.at(0).unwrap_err().to_string(), "$: expected an array, found object");
        let quoted = Value::from([("a\"b", 1)].into_iter().collect::<crate::json::Object>());
        assert_eq!(quoted.node().field("a\"b").unwrap().path(), r#"$["a\"b"]"#);
        assert_eq!(root.path(), "$");
        assert!(format!("{root:?}").contains("path"));
    }

    #[test]
    fn containers_read_in_place_and_by_length() {
        let v = doc(r#"{"v": [1, 2.5, -3], "m": [[1, 0], [0, 1]], "bad": [1, "x"]}"#);
        let n = v.node();
        assert_eq!(n.get::<[f32; 3]>("v").unwrap(), [1.0, 2.5, -3.0]);
        assert_eq!(n.get::<Vec<f64>>("v").unwrap(), [1.0, 2.5, -3.0]);
        assert_eq!(n.get::<[[u8; 2]; 2]>("m").unwrap(), [[1, 0], [0, 1]]);
        assert_eq!(n.get::<Vec<Vec<u8>>>("m").unwrap(), [vec![1, 0], vec![0, 1]]);
        assert_eq!(n.get::<[f32; 4]>("v").unwrap_err().what, What::Length { expected: 4, found: 3 });
        assert_eq!(n.get::<[u8; 2]>("bad").unwrap_err().to_string(), "$.bad[1]: expected a number, found string");
        assert_eq!(n.get::<Vec<u8>>("bad").unwrap_err().path, "$.bad[1]");
        let mut buffer = [0.0_f32; 3];
        n.field("v").unwrap().read_into(&mut buffer).unwrap();
        assert_eq!(buffer, [1.0, 2.5, -3.0]);
        let mut short = [0.0_f32; 2];
        assert!(n.field("v").unwrap().read_into(&mut short).is_err());
        let mut list = vec![9.0_f32];
        n.field("v").unwrap().extend_into(&mut list).unwrap();
        assert_eq!(list, [9.0, 1.0, 2.5, -3.0]);
        assert_eq!(n.field("v").unwrap().array_len().unwrap(), 3);
        let elements = n.field("m").unwrap();
        let paths: Vec<String> = elements.elements().unwrap().map(|e| e.path()).collect();
        assert_eq!(paths, ["$.m[0]", "$.m[1]"]);
        assert_eq!(elements.elements().unwrap().len(), 2);
        let keys: Vec<(&str, String)> = n.members().unwrap().map(|(k, node)| (k, node.path())).collect();
        assert_eq!(keys, [("v", "$.v".to_owned()), ("m", "$.m".to_owned()), ("bad", "$.bad".to_owned())]);
        assert!(n.field("v").unwrap().members().is_err());
    }

    #[test]
    fn choices_and_unknown_members() {
        #[derive(Clone, Copy, Debug, PartialEq)]
        enum Filter {
            Linear,
            Nearest,
        }
        let v = doc(r#"{"filter": "nearest", "other": "cubic", "extra": 1}"#);
        let n = v.node();
        let choices = [("linear", Filter::Linear), ("nearest", Filter::Nearest)];
        assert_eq!(n.field("filter").unwrap().choice(&choices).unwrap(), Filter::Nearest);
        assert_eq!(
            n.field("other").unwrap().choice(&choices).unwrap_err().to_string(),
            r#"$.other: unknown name "cubic", expected one of "linear", "nearest""#
        );
        assert!(n.deny_unknown(&["filter", "other", "extra"]).is_ok());
        let e = n.deny_unknown(&["filter", "other"]).unwrap_err();
        assert_eq!(e.to_string(), "$.extra: unknown member");
        assert_eq!(n.field("filter").unwrap().invalid("must be linear").to_string(), "$.filter: must be linear");
        assert!(n.field("extra").unwrap().deny_unknown(&[]).is_err());
    }

    /// A document type read and written by hand, as the project's formats are.
    #[derive(Debug, PartialEq)]
    struct Material {
        name: String,
        roughness: f32,
        tint: [f32; 3],
        layers: Vec<u16>,
        note: Option<String>,
    }

    impl FromJson<'_> for Material {
        fn from_json(node: &Node<'_, '_>) -> Result<Self, AccessError> {
            node.deny_unknown(&["name", "roughness", "tint", "layers", "note"])?;
            let roughness = node.get("roughness")?;
            if !(0.0..=1.0).contains(&roughness) {
                return Err(node.field("roughness")?.invalid("roughness must be in 0..=1"));
            }
            Ok(Self {
                name: node.get("name")?,
                roughness,
                tint: node.get_or("tint", [1.0; 3])?,
                layers: node.get_or("layers", Vec::new())?,
                note: node.get_opt("note")?,
            })
        }
    }

    #[test]
    fn a_hand_written_reader_reports_where() {
        let v = doc(r#"[{"name": "skin", "roughness": 0.4, "layers": [1, 2]}, {"name": "eye", "roughness": 1.5}]"#);
        let first: Material = v.node().at(0).unwrap().read().unwrap();
        assert_eq!(
            first,
            Material { name: "skin".into(), roughness: 0.4, tint: [1.0; 3], layers: vec![1, 2], note: None }
        );
        let all = v.node().read::<Vec<Material>>().unwrap_err();
        assert_eq!(all.to_string(), "$[1].roughness: roughness must be in 0..=1");
        let source: &dyn std::error::Error = &all;
        assert!(source.source().is_none());
    }
}
