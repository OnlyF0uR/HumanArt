//! The JSON value tree: [`Value`] and the insertion-ordered [`Object`].
//!
//! An object keeps its members in the order they were inserted (or read), so writing a parsed
//! document gives its members back in the same order `[AC-WP07-1]`. Lookup by key is a linear scan
//! while the object is small and goes through a hash index once it has more than
//! [`Object::INDEX_THRESHOLD`] members; the index uses the standard library's keyed hasher, because
//! keys come straight from files and the network, where a fixed hash function could be flooded with
//! colliding keys. The index never influences the order of anything.
//!
//! `Value` is a plain tree: dropping, cloning and comparing one recurse once per nesting level. The
//! parser bounds the depth of what it builds ([`MAX_DEPTH_CAP`](super::parse::MAX_DEPTH_CAP)); code
//! that builds values by hand is expected to stay as shallow.

use super::number::Number;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::BuildHasher;

/// The six kinds of JSON value, for messages and dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool,
    /// A number.
    Number,
    /// A string.
    String,
    /// An array.
    Array,
    /// An object.
    Object,
}

impl Kind {
    /// The lower-case name used in messages ("number", "array").
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool => "boolean",
            Self::Number => "number",
            Self::String => "string",
            Self::Array => "array",
            Self::Object => "object",
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A JSON value.
///
/// Equality is structural and exact: numbers compare as [`Number`] does (an integer never equals a
/// float, floats compare by bit pattern) and objects compare with their member order, so two values
/// are equal exactly when they serialise to the same bytes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Value {
    /// `null`.
    #[default]
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number, with its integer-ness kept.
    Number(Number),
    /// A string.
    String(String),
    /// An array.
    Array(Vec<Value>),
    /// An object, in insertion order.
    Object(Object),
}

/// What indexing a value with a missing key or index refers to.
static NULL: Value = Value::Null;

impl Value {
    /// Which kind of value this is.
    #[must_use]
    pub const fn kind(&self) -> Kind {
        match self {
            Self::Null => Kind::Null,
            Self::Bool(_) => Kind::Bool,
            Self::Number(_) => Kind::Number,
            Self::String(_) => Kind::String,
            Self::Array(_) => Kind::Array,
            Self::Object(_) => Kind::Object,
        }
    }

    /// Whether this is `null`.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// The boolean, if this is one.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The number, if this is one.
    #[must_use]
    pub const fn as_number(&self) -> Option<Number> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// The number as `f64` (integers are converted), if this is a number.
    #[must_use]
    pub const fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(n.as_f64()),
            _ => None,
        }
    }

    /// The number as `i64`, if this is a number that is a whole value in range.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        self.as_number().and_then(Number::as_i64)
    }

    /// The number as `u64`, if this is a number that is a whole, non-negative value in range.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        self.as_number().and_then(Number::as_u64)
    }

    /// The text, if this is a string.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// The elements, if this is an array.
    #[must_use]
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    /// The elements for editing, if this is an array.
    #[must_use]
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Value>> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    /// The members, if this is an object.
    #[must_use]
    pub const fn as_object(&self) -> Option<&Object> {
        match self {
            Self::Object(object) => Some(object),
            _ => None,
        }
    }

    /// The members for editing, if this is an object.
    #[must_use]
    pub fn as_object_mut(&mut self) -> Option<&mut Object> {
        match self {
            Self::Object(object) => Some(object),
            _ => None,
        }
    }

    /// The member `key`, if this is an object that has it.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_object().and_then(|object| object.get(key))
    }

    /// The member `key` for editing, if this is an object that has it.
    #[must_use]
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.as_object_mut().and_then(|object| object.get_mut(key))
    }

    /// The element `index`, if this is an array that long.
    #[must_use]
    pub fn at(&self, index: usize) -> Option<&Value> {
        self.as_array().and_then(|items| items.get(index))
    }

    /// The element `index` for editing, if this is an array that long.
    #[must_use]
    pub fn at_mut(&mut self, index: usize) -> Option<&mut Value> {
        self.as_array_mut().and_then(|items| items.get_mut(index))
    }

    /// Moves the value out, leaving `null`.
    #[must_use]
    pub fn take(&mut self) -> Value {
        std::mem::take(self)
    }

    /// Sorts the members of every object in the tree by key (see [`Object::sort_keys`]), giving the
    /// canonical form used for content hashes and order-independent comparison. Iterative, so any
    /// depth is safe.
    pub fn sort_keys_deep(&mut self) {
        let mut pending: Vec<&mut Value> = vec![self];
        while let Some(value) = pending.pop() {
            match value {
                Self::Array(items) => pending.extend(items.iter_mut()),
                Self::Object(object) => {
                    object.sort_keys();
                    pending.extend(object.iter_mut().map(|(_, v)| v));
                }
                _ => {}
            }
        }
    }
}

/// `value["key"]`: the member, or `null` when this is not an object or has no such member. Never
/// panics, so chains like `value["a"]["b"]` are safe on any input.
impl std::ops::Index<&str> for Value {
    type Output = Value;

    fn index(&self, key: &str) -> &Value {
        self.get(key).unwrap_or(&NULL)
    }
}

/// `value[3]`: the element, or `null` when this is not an array or is too short. Never panics.
impl std::ops::Index<usize> for Value {
    type Output = Value;

    fn index(&self, index: usize) -> &Value {
        self.at(index).unwrap_or(&NULL)
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<Number> for Value {
    fn from(v: Number) -> Self {
        Self::Number(v)
    }
}

macro_rules! value_from_number {
    ($($ty:ty),*) => {$(
        impl From<$ty> for Value {
            fn from(v: $ty) -> Self {
                Self::Number(Number::from(v))
            }
        }
    )*};
}

value_from_number!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Self::String(v.to_owned())
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&String> for Value {
    fn from(v: &String) -> Self {
        Self::String(v.clone())
    }
}

impl From<Object> for Value {
    fn from(v: Object) -> Self {
        Self::Object(v)
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Self::Array(v.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<Value> + Clone> From<&[T]> for Value {
    fn from(v: &[T]) -> Self {
        Self::Array(v.iter().cloned().map(Into::into).collect())
    }
}

impl<T: Into<Value>, const N: usize> From<[T; N]> for Value {
    fn from(v: [T; N]) -> Self {
        Self::Array(v.into_iter().map(Into::into).collect())
    }
}

/// `None` becomes `null`.
impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(v: Option<T>) -> Self {
        v.map_or(Self::Null, Into::into)
    }
}

impl<T: Into<Value>> FromIterator<T> for Value {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::Array(iter.into_iter().map(Into::into).collect())
    }
}

/// The marker of a free slot in the hash index.
const EMPTY: u32 = u32::MAX;

/// The hash index of a large object: open addressing over entry numbers.
#[derive(Clone)]
struct Index {
    hasher: RandomState,
    /// Entry numbers or [`EMPTY`]; the length is a power of two at least twice the entry count, so
    /// a probe always meets a free slot.
    slots: Vec<u32>,
}

impl Index {
    /// Builds the index of `entries`, or `None` when there are too many to number with `u32`.
    fn build(entries: &[(String, Value)]) -> Option<Box<Self>> {
        if entries.len() >= (EMPTY / 2) as usize {
            return None;
        }
        let capacity = (entries.len() * 2).next_power_of_two().max(64);
        let mut index = Box::new(Self { hasher: RandomState::new(), slots: vec![EMPTY; capacity] });
        for (number, (key, _)) in entries.iter().enumerate() {
            index.place(key, number as u32);
        }
        Some(index)
    }

    fn start(&self, key: &str) -> usize {
        self.hasher.hash_one(key) as usize & (self.slots.len() - 1)
    }

    /// Records that entry `number` has `key`. The caller keeps the table at most half full.
    fn place(&mut self, key: &str, number: u32) {
        let mask = self.slots.len() - 1;
        let mut at = self.start(key);
        // At most half the slots are taken, so a free one is reached.
        for _ in 0..self.slots.len() {
            if self.slots[at] == EMPTY {
                self.slots[at] = number;
                return;
            }
            at = (at + 1) & mask;
        }
    }

    fn find(&self, entries: &[(String, Value)], key: &str) -> Option<usize> {
        let mask = self.slots.len() - 1;
        let mut at = self.start(key);
        for _ in 0..self.slots.len() {
            let number = self.slots[at];
            if number == EMPTY {
                return None;
            }
            if entries.get(number as usize).is_some_and(|(k, _)| k == key) {
                return Some(number as usize);
            }
            at = (at + 1) & mask;
        }
        None
    }
}

/// A JSON object: string keys to values, in insertion order, with unique keys.
///
/// Two objects are equal when they hold the same members in the same order.
#[derive(Clone, Default)]
pub struct Object {
    entries: Vec<(String, Value)>,
    /// Present once the object is larger than [`Object::INDEX_THRESHOLD`].
    index: Option<Box<Index>>,
}

impl Object {
    /// Objects with more members than this are looked up through a hash index; smaller ones are
    /// scanned, which is faster and needs no extra memory.
    pub const INDEX_THRESHOLD: usize = 16;

    /// An empty object.
    #[must_use]
    pub const fn new() -> Self {
        Self { entries: Vec::new(), index: None }
    }

    /// An empty object with room for `capacity` members.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self { entries: Vec::with_capacity(capacity), index: None }
    }

    /// The number of members.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the object has no members.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The members in order, as a slice of `(key, value)` pairs.
    #[must_use]
    pub fn entries(&self) -> &[(String, Value)] {
        &self.entries
    }

    fn find(&self, key: &str) -> Option<usize> {
        match &self.index {
            Some(index) => index.find(&self.entries, key),
            None => self.entries.iter().position(|(k, _)| k == key),
        }
    }

    /// The position of the member `key` in the insertion order.
    #[must_use]
    pub fn position(&self, key: &str) -> Option<usize> {
        self.find(key)
    }

    /// Whether the object has a member `key`.
    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        self.find(key).is_some()
    }

    /// The value of the member `key`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.find(key).and_then(|i| self.entries.get(i)).map(|(_, v)| v)
    }

    /// The value of the member `key`, for editing.
    #[must_use]
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.find(key).and_then(|i| self.entries.get_mut(i)).map(|(_, v)| v)
    }

    /// The member at `position` in the insertion order.
    #[must_use]
    pub fn get_index(&self, position: usize) -> Option<(&str, &Value)> {
        self.entries.get(position).map(|(k, v)| (k.as_str(), v))
    }

    /// Appends a member whose key the caller knows to be absent.
    pub(crate) fn push_new(&mut self, key: String, value: Value) {
        let number = self.entries.len();
        match &mut self.index {
            Some(index) if (number + 1) * 2 <= index.slots.len() => index.place(&key, number as u32),
            Some(_) => {
                self.entries.push((key, value));
                self.index = Index::build(&self.entries);
                return;
            }
            None => {}
        }
        self.entries.push((key, value));
        if self.index.is_none() && self.entries.len() == Self::INDEX_THRESHOLD + 1 {
            self.index = Index::build(&self.entries);
        }
    }

    /// Sets the member `key`. A new key is appended; an existing key keeps its place in the order
    /// and its previous value is returned.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Value>) -> Option<Value> {
        let key = key.into();
        let value = value.into();
        if let Some(slot) = self.find(&key).and_then(|i| self.entries.get_mut(i)) {
            return Some(std::mem::replace(&mut slot.1, value));
        }
        self.push_new(key, value);
        None
    }

    /// Removes the member `key` and returns its value. The other members keep their order.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        let position = self.find(key)?;
        let (_, value) = self.entries.remove(position);
        self.reindex();
        Some(value)
    }

    /// Removes every member.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.index = None;
    }

    /// Sorts the members by key (byte order of the UTF-8 text), for canonical output.
    pub fn sort_keys(&mut self) {
        self.entries.sort_by(|a, b| a.0.cmp(&b.0));
        self.reindex();
    }

    fn reindex(&mut self) {
        self.index = if self.entries.len() > Self::INDEX_THRESHOLD { Index::build(&self.entries) } else { None };
    }

    /// The members in insertion order.
    pub fn iter(&self) -> Iter<'_> {
        Iter { inner: self.entries.iter() }
    }

    /// The members in insertion order, with their values editable.
    pub fn iter_mut(&mut self) -> IterMut<'_> {
        IterMut { inner: self.entries.iter_mut() }
    }

    /// The keys in insertion order.
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &str> + DoubleEndedIterator + '_ {
        self.entries.iter().map(|(k, _)| k.as_str())
    }

    /// The values in insertion order.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &Value> + DoubleEndedIterator + '_ {
        self.entries.iter().map(|(_, v)| v)
    }
}

impl PartialEq for Object {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries
    }
}

impl Eq for Object {}

impl fmt::Debug for Object {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.entries.iter().map(|(k, v)| (k, v))).finish()
    }
}

/// The iterator of [`Object::iter`].
#[derive(Clone, Debug)]
pub struct Iter<'a> {
    inner: std::slice::Iter<'a, (String, Value)>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = (&'a str, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (k.as_str(), v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl DoubleEndedIterator for Iter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|(k, v)| (k.as_str(), v))
    }
}

impl ExactSizeIterator for Iter<'_> {}

/// The iterator of [`Object::iter_mut`].
#[derive(Debug)]
pub struct IterMut<'a> {
    inner: std::slice::IterMut<'a, (String, Value)>,
}

impl<'a> Iterator for IterMut<'a> {
    type Item = (&'a str, &'a mut Value);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, v)| (k.as_str(), v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for IterMut<'_> {}

impl<'a> IntoIterator for &'a Object {
    type Item = (&'a str, &'a Value);
    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut Object {
    type Item = (&'a str, &'a mut Value);
    type IntoIter = IterMut<'a>;

    fn into_iter(self) -> IterMut<'a> {
        self.iter_mut()
    }
}

impl IntoIterator for Object {
    type Item = (String, Value);
    type IntoIter = std::vec::IntoIter<(String, Value)>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

/// Later members replace earlier ones with the same key, as [`Object::insert`] does.
impl<K: Into<String>, V: Into<Value>> Extend<(K, V)> for Object {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<K: Into<String>, V: Into<Value>> FromIterator<(K, V)> for Object {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut object = Self::new();
        object.extend(iter);
        object
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;

    #[test]
    fn objects_keep_insertion_order_and_unique_keys() {
        let mut o = Object::new();
        assert!(o.is_empty());
        assert_eq!(o.insert("zeta", 1), None);
        assert_eq!(o.insert("alpha", "x"), None);
        assert_eq!(o.insert("mid", true), None);
        assert_eq!(o.keys().collect::<Vec<_>>(), ["zeta", "alpha", "mid"]);
        // Replacing keeps the place.
        assert_eq!(o.insert("alpha", 2.5), Some(Value::from("x")));
        assert_eq!(o.keys().collect::<Vec<_>>(), ["zeta", "alpha", "mid"]);
        assert_eq!(o.get("alpha"), Some(&Value::from(2.5)));
        assert_eq!(o.position("mid"), Some(2));
        assert_eq!(o.get_index(0), Some(("zeta", &Value::from(1))));
        assert_eq!(o.get_index(3), None);
        assert!(o.contains_key("zeta") && !o.contains_key("Zeta"));
        *o.get_mut("zeta").unwrap() = Value::Null;
        assert_eq!(o.remove("zeta"), Some(Value::Null));
        assert_eq!(o.remove("zeta"), None);
        assert_eq!(o.iter().map(|(k, _)| k).collect::<Vec<_>>(), ["alpha", "mid"]);
        assert_eq!(o.len(), 2);
        for (_, v) in &mut o {
            *v = Value::from(0);
        }
        assert!(o.values().all(|v| *v == Value::from(0)));
        o.clear();
        assert!(o.is_empty() && o.get("alpha").is_none());
    }

    #[test]
    fn large_objects_behave_like_small_ones() {
        // The same random operations on an Object and on a plain list of pairs.
        let mut rng = Rng::new(0x0B1EC7);
        for round in 0..20 {
            let mut object = Object::new();
            let mut model: Vec<(String, Value)> = Vec::new();
            let key_space = 8 + round * 40;
            for step in 0..2500 {
                let key = format!("k{}", rng.index(key_space));
                match rng.index(10) {
                    0..=5 => {
                        let value = Value::from(step);
                        let old = object.insert(key.clone(), value.clone());
                        if let Some(slot) = model.iter_mut().find(|(k, _)| *k == key) {
                            assert_eq!(old, Some(std::mem::replace(&mut slot.1, value)));
                        } else {
                            assert_eq!(old, None);
                            model.push((key, value));
                        }
                    }
                    6 | 7 => {
                        let expected = model.iter().position(|(k, _)| *k == key).map(|i| model.remove(i).1);
                        assert_eq!(object.remove(&key), expected);
                    }
                    _ => {
                        assert_eq!(object.get(&key), model.iter().find(|(k, _)| *k == key).map(|(_, v)| v));
                        assert_eq!(object.position(&key), model.iter().position(|(k, _)| *k == key));
                    }
                }
                assert_eq!(object.len(), model.len());
            }
            assert_eq!(object.entries(), model.as_slice());
            assert_eq!(object.index.is_some(), model.len() > Object::INDEX_THRESHOLD);
            let copy = object.clone();
            assert_eq!(copy, object);
            for (key, value) in &model {
                assert_eq!(copy.get(key), Some(value));
            }
            object.sort_keys();
            model.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(object.entries(), model.as_slice());
            for (key, value) in &model {
                assert_eq!(object.get(key), Some(value));
            }
        }
    }

    #[test]
    fn a_hundred_thousand_members_are_found_quickly() {
        let mut object = Object::new();
        for i in 0..100_000_u32 {
            object.push_new(format!("member-{i}"), Value::from(i));
        }
        for i in (0..100_000_u32).step_by(7) {
            assert_eq!(object.get(&format!("member-{i}")), Some(&Value::from(i)));
        }
        assert_eq!(object.get("member-100000"), None);
        assert_eq!(object.insert("member-5", "new"), Some(Value::from(5)));
        assert_eq!(object.len(), 100_000);
    }

    #[test]
    fn equality_is_exact_and_ordered() {
        let a: Object = [("x", 1), ("y", 2)].into_iter().collect();
        let b: Object = [("y", 2), ("x", 1)].into_iter().collect();
        assert_ne!(a, b);
        assert_eq!(a, a.clone());
        assert_ne!(Value::from(1), Value::from(1.0));
        assert_eq!(Value::from(1_u8), Value::from(1_i64));
        assert_eq!(format!("{a:?}"), r#"{"x": Number(Int(1)), "y": Number(Int(2))}"#);
        // A later duplicate replaces the earlier value.
        let c: Object = [("x", 1), ("x", 3)].into_iter().collect();
        assert_eq!(c.entries(), [("x".to_owned(), Value::from(3))]);
        let owned: Vec<(String, Value)> = a.into_iter().collect();
        assert_eq!(owned.len(), 2);
    }

    #[test]
    fn values_convert_and_read_back() {
        assert_eq!(Value::default(), Value::Null);
        assert!(Value::Null.is_null() && !Value::from(false).is_null());
        assert_eq!(Value::from(true).as_bool(), Some(true));
        assert_eq!(Value::from(7_u64).as_u64(), Some(7));
        assert_eq!(Value::from(-7).as_i64(), Some(-7));
        assert_eq!(Value::from(-7).as_u64(), None);
        assert_eq!(Value::from(2.5_f32).as_f64(), Some(2.5));
        assert_eq!(Value::from(3_usize).as_number(), Some(Number::Int(3)));
        assert_eq!(Value::from("text").as_str(), Some("text"));
        assert_eq!(Value::from(String::from("s")).as_str(), Some("s"));
        assert_eq!(Value::from(&String::from("s")).as_str(), Some("s"));
        assert_eq!(Value::from(None::<i32>), Value::Null);
        assert_eq!(Value::from(Some(1)), Value::from(1));
        assert_eq!(Value::from(Number::Float(0.5)).as_f64(), Some(0.5));
        let array = Value::from(vec![1, 2, 3]);
        assert_eq!(array.as_array().map(<[Value]>::len), Some(3));
        assert_eq!(Value::from([1.0, 2.0]), Value::from(&[1.0, 2.0][..]));
        assert_eq!((1..=3).collect::<Value>(), array);
        assert_eq!(array.at(1), Some(&Value::from(2)));
        assert_eq!(array.at(3), None);
        for (value, kind) in [
            (Value::Null, "null"),
            (Value::from(true), "boolean"),
            (Value::from(1), "number"),
            (Value::from(""), "string"),
            (Value::Array(Vec::new()), "array"),
            (Value::Object(Object::new()), "object"),
        ] {
            assert_eq!(value.kind().to_string(), kind);
            // Each accessor answers only for its own kind.
            let hits = [
                value.as_bool().is_some(),
                value.as_number().is_some(),
                value.as_str().is_some(),
                value.as_array().is_some(),
                value.as_object().is_some(),
            ];
            assert_eq!(hits.iter().filter(|&&h| h).count(), usize::from(!value.is_null()));
        }
    }

    #[test]
    fn indexing_never_panics() {
        let mut object = Object::new();
        object.insert("list", vec![10, 20]);
        let mut doc = Value::from(object);
        assert_eq!(doc["list"][1], Value::from(20));
        assert_eq!(doc["list"][2], Value::Null);
        assert_eq!(doc["missing"]["deeper"][4], Value::Null);
        assert_eq!(doc[0], Value::Null);
        assert_eq!(doc.get("list").and_then(|l| l.at(0)), Some(&Value::from(10)));
        *doc.get_mut("list").unwrap().at_mut(0).unwrap() = Value::from("ten");
        doc.as_object_mut().unwrap().insert("more", Value::Null);
        doc.get_mut("list").unwrap().as_array_mut().unwrap().push(Value::from(30));
        assert_eq!(doc["list"], Value::from(vec![Value::from("ten"), Value::from(20), Value::from(30)]));
        let taken = doc.get_mut("list").unwrap().take();
        assert_eq!(taken.as_array().map(<[Value]>::len), Some(3));
        assert_eq!(doc["list"], Value::Null);
        assert!(doc.get_mut("nope").is_none() && doc.at_mut(0).is_none());
    }

    #[test]
    fn deep_key_sorting_reaches_every_object() {
        let inner: Object = [("b", 1), ("a", 2)].into_iter().collect();
        let mut outer = Object::new();
        outer.insert("z", Value::from(vec![Value::from(inner.clone())]));
        outer.insert("y", inner);
        let mut doc = Value::from(outer);
        // A deep chain does not overflow the stack.
        for _ in 0..100_000 {
            doc = Value::from(vec![doc]);
        }
        doc.sort_keys_deep();
        let mut cursor = &doc;
        while let Some(first) = cursor.at(0) {
            cursor = first;
        }
        assert_eq!(cursor.as_object().unwrap().keys().collect::<Vec<_>>(), ["y", "z"]);
        assert_eq!(cursor["y"].as_object().unwrap().keys().collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(cursor["z"][0].as_object().unwrap().keys().collect::<Vec<_>>(), ["a", "b"]);
        // Dropping a 100 000-deep tree recurses; take it apart by hand instead.
        while let Value::Array(mut items) = doc {
            doc = items.pop().unwrap_or_default();
        }
    }
}
