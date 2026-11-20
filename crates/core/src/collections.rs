//! Fixed-capacity containers for per-frame code `[PERF-R1]`, and a growable bit set.
//!
//! [`ArrayVec`], [`ArrayString`] and [`RingBuffer`] store their elements inline and never touch the
//! heap, so they can live on the stack of a frame function or inside a pre-sized structure. They are
//! written in safe Rust, which is why the element type must be `Copy + Default` (the unused slots hold
//! default values). [`BitSet`] owns a `Vec` of words and allocates only when it has to grow.

use std::fmt;
use std::ops::{Deref, DerefMut, Range};

/// A vector of at most `N` elements stored inline.
#[derive(Clone, Copy)]
pub struct ArrayVec<T, const N: usize> {
    items: [T; N],
    len: usize,
}

impl<T: Copy + Default, const N: usize> ArrayVec<T, N> {
    /// An empty vector.
    pub fn new() -> Self {
        Self { items: [T::default(); N], len: 0 }
    }

    /// The number of elements stored.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// `true` when nothing is stored.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `true` when another `push` would fail.
    pub const fn is_full(&self) -> bool {
        self.len == N
    }

    /// The fixed capacity `N`.
    pub const fn capacity(&self) -> usize {
        N
    }

    /// How many more elements fit.
    pub const fn remaining(&self) -> usize {
        N - self.len
    }

    /// Appends `value`, or hands it back when the vector is full.
    pub fn push(&mut self, value: T) -> Result<(), T> {
        match self.items.get_mut(self.len) {
            Some(slot) => {
                *slot = value;
                self.len += 1;
                Ok(())
            }
            None => Err(value),
        }
    }

    /// Removes and returns the last element.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(self.items[self.len])
    }

    /// Removes every element.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Keeps the first `len` elements; does nothing when the vector is already that short.
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// The stored elements.
    pub fn as_slice(&self) -> &[T] {
        &self.items[..self.len]
    }

    /// The stored elements, mutably.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.items[..self.len]
    }

    /// Inserts `value` before position `index`, shifting the tail. Hands the value back when the
    /// vector is full or `index > len`.
    pub fn insert(&mut self, index: usize, value: T) -> Result<(), T> {
        if self.len == N || index > self.len {
            return Err(value);
        }
        self.items.copy_within(index..self.len, index + 1);
        self.items[index] = value;
        self.len += 1;
        Ok(())
    }

    /// Removes and returns the element at `index`, shifting the tail down (order is kept).
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        let value = self.items[index];
        self.items.copy_within(index + 1..self.len, index);
        self.len -= 1;
        Some(value)
    }

    /// Removes and returns the element at `index`, moving the last element into its place (O(1),
    /// order is not kept).
    pub fn swap_remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }
        let value = self.items[index];
        self.len -= 1;
        self.items[index] = self.items[self.len];
        Some(value)
    }

    /// Appends as many elements of `values` as fit and returns how many were copied.
    pub fn extend_from_slice(&mut self, values: &[T]) -> usize {
        let count = values.len().min(N - self.len);
        self.items[self.len..self.len + count].copy_from_slice(&values[..count]);
        self.len += count;
        count
    }

    /// Keeps the elements for which `keep` returns `true`, in order.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let mut kept = 0;
        for i in 0..self.len {
            let value = self.items[i];
            if keep(&value) {
                self.items[kept] = value;
                kept += 1;
            }
        }
        self.len = kept;
    }
}

impl<T: Copy + Default, const N: usize> Default for ArrayVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default, const N: usize> Deref for ArrayVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Copy + Default, const N: usize> DerefMut for ArrayVec<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Copy + Default + fmt::Debug, const N: usize> fmt::Debug for ArrayVec<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T: Copy + Default + PartialEq, const N: usize, const M: usize> PartialEq<ArrayVec<T, M>> for ArrayVec<T, N> {
    fn eq(&self, other: &ArrayVec<T, M>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Copy + Default + Eq, const N: usize> Eq for ArrayVec<T, N> {}

impl<'a, T: Copy + Default, const N: usize> IntoIterator for &'a ArrayVec<T, N> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a, T: Copy + Default, const N: usize> IntoIterator for &'a mut ArrayVec<T, N> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

/// A UTF-8 string of at most `N` bytes stored inline.
///
/// Appending more than fits keeps the longest prefix that ends on a character boundary and sets the
/// [`is_truncated`](Self::is_truncated) flag; once truncated, later appends are ignored (so the text
/// never has a hole in the middle) until [`clear`](Self::clear). It implements [`fmt::Write`], which
/// is how the logger and the UI format text without a `String`: `write!` into it never fails.
#[derive(Clone, Copy)]
pub struct ArrayString<const N: usize> {
    bytes: [u8; N],
    len: usize,
    truncated: bool,
}

impl<const N: usize> ArrayString<N> {
    /// An empty string.
    pub const fn new() -> Self {
        Self { bytes: [0; N], len: 0, truncated: false }
    }

    /// A string holding `text`, or the longest prefix of it that fits.
    pub fn from_str_truncated(text: &str) -> Self {
        let mut out = Self::new();
        out.push_str(text);
        out
    }

    /// The text.
    pub fn as_str(&self) -> &str {
        // The bytes up to `len` are always whole characters copied from a `str`, so this cannot fail;
        // the empty fallback only keeps the function total.
        std::str::from_utf8(&self.bytes[..self.len]).unwrap_or("")
    }

    /// The text as bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    /// The length in bytes.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// `true` when the string holds no text.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The fixed capacity `N` in bytes.
    pub const fn capacity(&self) -> usize {
        N
    }

    /// `true` when an append did not fit and text was dropped.
    pub const fn is_truncated(&self) -> bool {
        self.truncated
    }

    /// Appends `text`. Returns `false` when it did not fit completely (see the type's description).
    pub fn push_str(&mut self, text: &str) -> bool {
        if self.truncated {
            return false;
        }
        let space = N - self.len;
        let fits = text.len() <= space;
        let mut take = text.len().min(space);
        while !text.is_char_boundary(take) {
            take -= 1;
        }
        self.bytes[self.len..self.len + take].copy_from_slice(&text.as_bytes()[..take]);
        self.len += take;
        self.truncated = !fits;
        fits
    }

    /// Appends one character. Returns `false` when it did not fit.
    pub fn push(&mut self, c: char) -> bool {
        let mut utf8 = [0_u8; 4];
        self.push_str(c.encode_utf8(&mut utf8))
    }

    /// Removes and returns the last character.
    pub fn pop(&mut self) -> Option<char> {
        let c = self.as_str().chars().next_back()?;
        self.len -= c.len_utf8();
        Some(c)
    }

    /// Shortens the string to at most `len` bytes, backing up to a character boundary.
    pub fn truncate(&mut self, len: usize) {
        if len >= self.len {
            return;
        }
        let mut cut = len;
        while !self.as_str().is_char_boundary(cut) {
            cut -= 1;
        }
        self.len = cut;
    }

    /// Empties the string and clears the truncation flag.
    pub fn clear(&mut self) {
        self.len = 0;
        self.truncated = false;
    }
}

impl<const N: usize> Default for ArrayString<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> fmt::Write for ArrayString<N> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.push_str(text);
        Ok(())
    }
}

impl<const N: usize> Deref for ArrayString<N> {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl<const N: usize> AsRef<str> for ArrayString<N> {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<const N: usize> fmt::Display for ArrayString<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl<const N: usize> fmt::Debug for ArrayString<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl<const N: usize, const M: usize> PartialEq<ArrayString<M>> for ArrayString<N> {
    fn eq(&self, other: &ArrayString<M>) -> bool {
        self.as_str() == other.as_str()
    }
}

impl<const N: usize> Eq for ArrayString<N> {}

impl<const N: usize> PartialEq<str> for ArrayString<N> {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl<const N: usize> PartialEq<&str> for ArrayString<N> {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl<const N: usize> std::hash::Hash for ArrayString<N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

/// A set of non-negative integers stored one bit each.
///
/// It grows when a bit beyond its current size is inserted and allocates only then: every other
/// operation, including the set algebra, works in place. Indices that come from a file or the network
/// must be range-checked by the caller, because inserting a huge index allocates memory for all the
/// bits below it.
#[derive(Clone, Default)]
pub struct BitSet {
    words: Vec<u64>,
}

const WORD_BITS: usize = 64;

impl BitSet {
    /// An empty set that owns no memory.
    pub const fn new() -> Self {
        Self { words: Vec::new() }
    }

    /// An empty set that can hold the indices `0..bits` without allocating again.
    pub fn with_capacity(bits: usize) -> Self {
        let mut set = Self::new();
        set.reserve(bits);
        set
    }

    /// Makes room for the indices `0..bits`.
    pub fn reserve(&mut self, bits: usize) {
        let words = bits.div_ceil(WORD_BITS);
        if words > self.words.len() {
            self.words.resize(words, 0);
        }
    }

    /// The number of indices that can be stored without growing.
    pub fn capacity(&self) -> usize {
        self.words.len().saturating_mul(WORD_BITS)
    }

    /// Adds `index`. Returns `true` when it was not in the set before.
    pub fn insert(&mut self, index: usize) -> bool {
        let word = index / WORD_BITS;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        let mask = 1_u64 << (index % WORD_BITS);
        let slot = &mut self.words[word];
        let added = *slot & mask == 0;
        *slot |= mask;
        added
    }

    /// Adds every index in `range`.
    pub fn insert_range(&mut self, range: Range<usize>) {
        if range.start >= range.end {
            return;
        }
        self.reserve(range.end);
        let (first, last) = (range.start / WORD_BITS, (range.end - 1) / WORD_BITS);
        let low = u64::MAX << (range.start % WORD_BITS);
        let high = u64::MAX >> (WORD_BITS - 1 - (range.end - 1) % WORD_BITS);
        if first == last {
            self.words[first] |= low & high;
            return;
        }
        self.words[first] |= low;
        for word in &mut self.words[first + 1..last] {
            *word = u64::MAX;
        }
        self.words[last] |= high;
    }

    /// Removes `index`. Returns `true` when it was in the set.
    pub fn remove(&mut self, index: usize) -> bool {
        let mask = 1_u64 << (index % WORD_BITS);
        match self.words.get_mut(index / WORD_BITS) {
            Some(slot) => {
                let present = *slot & mask != 0;
                *slot &= !mask;
                present
            }
            None => false,
        }
    }

    /// Adds or removes `index` according to `present`.
    pub fn set(&mut self, index: usize, present: bool) {
        if present {
            self.insert(index);
        } else {
            self.remove(index);
        }
    }

    /// Flips the membership of `index` and returns the new state.
    pub fn toggle(&mut self, index: usize) -> bool {
        if self.remove(index) {
            false
        } else {
            self.insert(index);
            true
        }
    }

    /// `true` when `index` is in the set.
    pub fn contains(&self, index: usize) -> bool {
        self.words.get(index / WORD_BITS).is_some_and(|word| word & (1 << (index % WORD_BITS)) != 0)
    }

    /// Removes every index and keeps the memory.
    pub fn clear(&mut self) {
        self.words.fill(0);
    }

    /// The number of indices in the set.
    pub fn count(&self) -> usize {
        self.words.iter().map(|word| word.count_ones() as usize).sum()
    }

    /// `true` when the set has no index.
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&word| word == 0)
    }

    /// The smallest index in the set.
    pub fn first(&self) -> Option<usize> {
        self.iter().next()
    }

    /// The indices in the set, ascending.
    pub fn iter(&self) -> BitIter<'_> {
        BitIter { words: &self.words, next_word: 0, current: 0 }
    }

    /// Adds every index of `other` (set union).
    pub fn union_with(&mut self, other: &Self) {
        if other.words.len() > self.words.len() {
            // Grow only as far as the other set really has members.
            let used = other.words.iter().rposition(|&word| word != 0).map_or(0, |last| last + 1);
            if used > self.words.len() {
                self.words.resize(used, 0);
            }
        }
        for (mine, theirs) in self.words.iter_mut().zip(&other.words) {
            *mine |= theirs;
        }
    }

    /// Keeps only the indices that are also in `other` (set intersection).
    pub fn intersect_with(&mut self, other: &Self) {
        let shared = self.words.len().min(other.words.len());
        for (mine, theirs) in self.words.iter_mut().zip(&other.words) {
            *mine &= theirs;
        }
        self.words[shared..].fill(0);
    }

    /// Removes every index that is in `other` (set difference).
    pub fn difference_with(&mut self, other: &Self) {
        for (mine, theirs) in self.words.iter_mut().zip(&other.words) {
            *mine &= !theirs;
        }
    }

    /// `true` when every index of this set is also in `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        self.words.iter().enumerate().all(|(i, &word)| word & !other.words.get(i).copied().unwrap_or(0) == 0)
    }

    /// `true` when the two sets share at least one index.
    pub fn intersects(&self, other: &Self) -> bool {
        self.words.iter().zip(&other.words).any(|(a, b)| a & b != 0)
    }
}

impl PartialEq for BitSet {
    /// Two sets are equal when they hold the same indices, whatever their capacities.
    fn eq(&self, other: &Self) -> bool {
        let shared = self.words.len().min(other.words.len());
        self.words[..shared] == other.words[..shared]
            && self.words[shared..].iter().all(|&word| word == 0)
            && other.words[shared..].iter().all(|&word| word == 0)
    }
}

impl Eq for BitSet {}

impl fmt::Debug for BitSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl Extend<usize> for BitSet {
    fn extend<I: IntoIterator<Item = usize>>(&mut self, indices: I) {
        for index in indices {
            self.insert(index);
        }
    }
}

impl FromIterator<usize> for BitSet {
    fn from_iter<I: IntoIterator<Item = usize>>(indices: I) -> Self {
        let mut set = Self::new();
        set.extend(indices);
        set
    }
}

impl<'a> IntoIterator for &'a BitSet {
    type Item = usize;
    type IntoIter = BitIter<'a>;

    fn into_iter(self) -> BitIter<'a> {
        self.iter()
    }
}

/// The ascending iterator over the members of a [`BitSet`].
#[derive(Clone, Debug)]
pub struct BitIter<'a> {
    words: &'a [u64],
    /// Index of the word that will be loaded into `current` next.
    next_word: usize,
    /// The not yet reported bits of word `next_word - 1`.
    current: u64,
}

impl Iterator for BitIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        while self.current == 0 {
            self.current = *self.words.get(self.next_word)?;
            self.next_word += 1;
        }
        let bit = self.current.trailing_zeros() as usize;
        self.current &= self.current - 1;
        Some((self.next_word - 1) * WORD_BITS + bit)
    }
}

/// A first-in first-out queue of at most `N` elements stored inline.
///
/// [`push`](Self::push) overwrites the oldest element when the buffer is full, which is what history
/// buffers (frame times, recent events) want; [`try_push`](Self::try_push) refuses instead, which is
/// what bounded queues want.
#[derive(Clone, Copy)]
pub struct RingBuffer<T, const N: usize> {
    items: [T; N],
    /// Position of the oldest element.
    head: usize,
    len: usize,
}

impl<T: Copy + Default, const N: usize> RingBuffer<T, N> {
    /// An empty buffer.
    pub fn new() -> Self {
        Self { items: [T::default(); N], head: 0, len: 0 }
    }

    /// The number of elements stored.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// `true` when nothing is stored.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `true` when the next `push` will overwrite the oldest element.
    pub const fn is_full(&self) -> bool {
        self.len == N
    }

    /// The fixed capacity `N`.
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Removes every element.
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// The storage position of the element `offset` places after the oldest one.
    fn slot(&self, offset: usize) -> usize {
        // `head < N` and `offset <= N`, so the sum cannot overflow for any array that fits in memory.
        let position = self.head + offset;
        if position >= N { position - N } else { position }
    }

    /// Appends `value` as the newest element. When the buffer is full the oldest element is
    /// overwritten and returned (with `N == 0` the value itself comes straight back).
    pub fn push(&mut self, value: T) -> Option<T> {
        if N == 0 {
            return Some(value);
        }
        if self.len < N {
            let at = self.slot(self.len);
            self.items[at] = value;
            self.len += 1;
            return None;
        }
        let evicted = std::mem::replace(&mut self.items[self.head], value);
        self.head = self.slot(1);
        Some(evicted)
    }

    /// Appends `value` as the newest element, or hands it back when the buffer is full.
    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        if self.len == N {
            return Err(value);
        }
        let at = self.slot(self.len);
        self.items[at] = value;
        self.len += 1;
        Ok(())
    }

    /// Removes and returns the oldest element.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let value = self.items[self.head];
        self.head = self.slot(1);
        self.len -= 1;
        Some(value)
    }

    /// The element `index` places after the oldest one (`get(0)` is the oldest).
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        self.items.get(self.slot(index))
    }

    /// The oldest element.
    pub fn front(&self) -> Option<&T> {
        self.get(0)
    }

    /// The newest element.
    pub fn back(&self) -> Option<&T> {
        self.get(self.len.checked_sub(1)?)
    }

    /// The elements as two slices, oldest first: the second continues where the first ends.
    pub fn as_slices(&self) -> (&[T], &[T]) {
        let first = self.len.min(N - self.head);
        (&self.items[self.head..self.head + first], &self.items[..self.len - first])
    }

    /// The elements from oldest to newest.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &T> + ExactSizeIterator + Clone {
        let (older, newer) = self.as_slices();
        RingIter { older: older.iter(), newer: newer.iter() }
    }
}

impl<T: Copy + Default, const N: usize> Default for RingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default + fmt::Debug, const N: usize> fmt::Debug for RingBuffer<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// The two halves of a ring buffer chained, with an exact length.
#[derive(Clone)]
struct RingIter<'a, T> {
    older: std::slice::Iter<'a, T>,
    newer: std::slice::Iter<'a, T>,
}

impl<'a, T> Iterator for RingIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.older.next().or_else(|| self.newer.next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.older.len() + self.newer.len();
        (len, Some(len))
    }
}

impl<T> DoubleEndedIterator for RingIter<'_, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.newer.next_back().or_else(|| self.older.next_back())
    }
}

impl<T> ExactSizeIterator for RingIter<'_, T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::count_in;
    use crate::rng::Rng;
    use std::collections::{BTreeSet, VecDeque};
    use std::fmt::Write as _;

    #[test]
    fn array_vec_push_pop_and_capacity() {
        let mut v = ArrayVec::<u32, 4>::new();
        assert!(v.is_empty() && !v.is_full());
        assert_eq!((v.capacity(), v.remaining()), (4, 4));
        for i in 0..4 {
            assert_eq!(v.push(i), Ok(()));
        }
        assert!(v.is_full());
        assert_eq!(v.push(99), Err(99));
        assert_eq!(v.as_slice(), &[0, 1, 2, 3]);
        assert_eq!(v.pop(), Some(3));
        assert_eq!(v.len(), 3);
        v.truncate(10);
        assert_eq!(v.len(), 3);
        v.truncate(1);
        assert_eq!(&*v, &[0]);
        v.clear();
        assert_eq!(v.pop(), None);
        // A zero-capacity vector is legal and always full.
        let mut empty = ArrayVec::<u8, 0>::default();
        assert_eq!(empty.push(1), Err(1));
        assert_eq!(empty.extend_from_slice(&[1, 2]), 0);
    }

    #[test]
    fn array_vec_insert_remove_retain() {
        let mut v = ArrayVec::<u32, 6>::new();
        assert_eq!(v.extend_from_slice(&[10, 20, 30, 40]), 4);
        assert_eq!(v.insert(1, 15), Ok(()));
        assert_eq!(v.insert(5, 50), Ok(()));
        assert_eq!(v.as_slice(), &[10, 15, 20, 30, 40, 50]);
        assert_eq!(v.insert(0, 1), Err(1));
        assert_eq!(v.remove(0), Some(10));
        assert_eq!(v.insert(9, 1), Err(1));
        assert_eq!(v.remove(5), None);
        assert_eq!(v.swap_remove(0), Some(15));
        assert_eq!(v.as_slice(), &[50, 20, 30, 40]);
        assert_eq!(v.swap_remove(3), Some(40));
        assert_eq!(v.swap_remove(3), None);
        v.retain(|&x| x != 20);
        assert_eq!(v.as_slice(), &[50, 30]);
        assert_eq!(v.extend_from_slice(&[1, 2, 3, 4, 5, 6]), 4);
        assert_eq!(v.len(), 6);
        v.as_mut_slice()[0] = 7;
        v[1] = 8;
        assert_eq!(format!("{v:?}"), "[7, 8, 1, 2, 3, 4]");
        let mut sum = 0;
        for x in &v {
            sum += x;
        }
        for x in &mut v {
            *x += 1;
        }
        assert_eq!((sum, v.iter().sum::<u32>()), (25, 31));
        let mut other = ArrayVec::<u32, 9>::new();
        other.extend_from_slice(&v);
        assert_eq!(v, other);
    }

    #[test]
    fn array_vec_matches_a_vec_under_random_operations() {
        let mut rng = Rng::new(21);
        let mut ours = ArrayVec::<u16, 16>::new();
        let mut model: Vec<u16> = Vec::new();
        for _ in 0..20_000 {
            let value = rng.next_u32() as u16;
            let index = rng.index(18);
            match rng.index(6) {
                0 | 1 => {
                    let ok = ours.push(value).is_ok();
                    assert_eq!(ok, model.len() < 16);
                    if ok {
                        model.push(value);
                    }
                }
                2 => assert_eq!(ours.pop(), model.pop()),
                3 => {
                    let ok = ours.insert(index, value).is_ok();
                    assert_eq!(ok, model.len() < 16 && index <= model.len());
                    if ok {
                        model.insert(index, value);
                    }
                }
                4 => assert_eq!(ours.remove(index), (index < model.len()).then(|| model.remove(index))),
                _ => assert_eq!(ours.swap_remove(index), (index < model.len()).then(|| model.swap_remove(index))),
            }
            assert_eq!(ours.as_slice(), model.as_slice());
        }
    }

    #[test]
    fn array_string_appends_and_reports_truncation() {
        let mut s = ArrayString::<8>::new();
        assert!(s.is_empty() && !s.is_truncated());
        assert!(s.push_str("abc"));
        assert!(s.push('d'));
        assert_eq!(s, "abcd");
        assert_eq!((s.len(), s.capacity()), (4, 8));
        assert!(!s.push_str("efghij"));
        assert_eq!(s.as_str(), "abcdefgh");
        assert!(s.is_truncated());
        // Once truncated nothing more is appended, even if it would fit after a pop.
        assert_eq!(s.pop(), Some('h'));
        assert!(!s.push('z'));
        assert_eq!(s.as_str(), "abcdefg");
        s.clear();
        assert!(!s.is_truncated() && s.is_empty());
        assert!(s.push_str("12345678"));
        assert!(!s.is_truncated());
        assert_eq!(s.as_bytes(), b"12345678");
    }

    #[test]
    fn array_string_truncates_on_character_boundaries() {
        // "é" is 2 bytes, "€" is 3, "𝄞" is 4.
        for (capacity_text, expect) in [("aé", "a"), ("a€", "a"), ("𝄞", "")] {
            let mut s = ArrayString::<2>::new();
            assert!(!s.push_str(capacity_text) || capacity_text.len() <= 2);
            assert_eq!(s.as_str(), expect);
            assert!(s.is_truncated());
        }
        let mut s = ArrayString::<7>::from_str_truncated("a€𝄞€");
        assert_eq!(s.as_str(), "a€");
        assert!(s.is_truncated());
        s.clear();
        s.push_str("é€x");
        s.truncate(4);
        assert_eq!(s.as_str(), "é");
        s.truncate(100);
        assert_eq!(s.as_str(), "é");
        s.truncate(1);
        assert_eq!(s.as_str(), "");
        assert_eq!(s.pop(), None);
        assert!(!s.is_truncated());
        assert!(!s.push('𝄞') || s.capacity() >= 4);
    }

    #[test]
    fn array_string_formats_without_allocating() {
        let mut s = ArrayString::<32>::new();
        let stats = count_in(|| {
            let _ = write!(s, "{}-{:>4}-{:.2}", 7, "ab", 1.5_f32);
        });
        assert!(crate::alloc::is_installed());
        assert_eq!(stats.events(), 0, "{stats:?}");
        assert_eq!(s, "7-  ab-1.50");
        let mut short = ArrayString::<4>::new();
        assert!(write!(short, "{}", 123_456).is_ok());
        assert_eq!(short, "1234");
        assert!(short.is_truncated());
        assert_eq!(format!("{short} {short:?}"), "1234 \"1234\"");
        assert_eq!(short, ArrayString::<9>::from_str_truncated("1234"));
        assert_eq!(short.as_ref() as &str, &*short);
        let set: std::collections::HashSet<ArrayString<4>> = [short, ArrayString::from_str_truncated("1234")].into();
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn array_string_random_appends_stay_valid_utf8() {
        let mut rng = Rng::new(4);
        let pieces = ["a", "é", "€", "𝄞", "xyz", ""];
        for _ in 0..2000 {
            let mut s = ArrayString::<11>::new();
            let mut model = String::new();
            let mut cut = false;
            for _ in 0..rng.index(8) {
                let piece = pieces[rng.index(pieces.len())];
                let fitted = s.push_str(piece);
                if !cut {
                    if model.len() + piece.len() <= 11 {
                        model.push_str(piece);
                        assert!(fitted);
                    } else {
                        let mut take = 11 - model.len();
                        while !piece.is_char_boundary(take) {
                            take -= 1;
                        }
                        model.push_str(&piece[..take]);
                        cut = true;
                        assert!(!fitted);
                    }
                }
                assert_eq!(s.as_str(), model);
                assert_eq!(s.is_truncated(), cut);
            }
        }
    }

    #[test]
    fn bit_set_basic_membership() {
        let mut set = BitSet::new();
        assert!(set.is_empty());
        assert_eq!((set.capacity(), set.count(), set.first()), (0, 0, None));
        assert!(set.insert(3));
        assert!(!set.insert(3));
        assert!(set.insert(64));
        assert!(set.insert(1000));
        assert!(set.contains(3) && set.contains(64) && set.contains(1000));
        assert!(!set.contains(4) && !set.contains(1_000_000));
        assert_eq!(set.count(), 3);
        assert_eq!(set.iter().collect::<Vec<_>>(), [3, 64, 1000]);
        assert_eq!(set.first(), Some(3));
        assert!(set.remove(64));
        assert!(!set.remove(64));
        assert!(!set.remove(1_000_000));
        assert!(set.toggle(5));
        assert!(!set.toggle(5));
        set.set(7, true);
        set.set(3, false);
        assert_eq!(format!("{set:?}"), "{7, 1000}");
        set.clear();
        assert!(set.is_empty());
        assert!(set.capacity() >= 1001);
    }

    #[test]
    fn bit_set_insert_range_matches_single_inserts() {
        for (start, end) in
            [(0, 0), (5, 5), (9, 3), (0, 1), (0, 64), (1, 63), (63, 65), (64, 128), (3, 200), (130, 131)]
        {
            let mut ranged = BitSet::new();
            ranged.insert(500);
            ranged.insert_range(start..end);
            let mut single = BitSet::new();
            single.insert(500);
            single.extend(start..end);
            assert_eq!(ranged, single, "{start}..{end}");
            assert_eq!(ranged.count(), 1 + end.saturating_sub(start));
        }
    }

    #[test]
    fn bit_set_algebra_matches_a_tree_set() {
        let mut rng = Rng::new(8);
        for round in 0..200 {
            let limit_a = 1 + rng.index(400);
            let limit_b = 1 + rng.index(400);
            let a_model: BTreeSet<usize> = (0..rng.index(120)).map(|_| rng.index(limit_a)).collect();
            let b_model: BTreeSet<usize> = (0..rng.index(120)).map(|_| rng.index(limit_b)).collect();
            let a: BitSet = a_model.iter().copied().collect();
            let b: BitSet = b_model.iter().copied().collect();
            assert_eq!(a.count(), a_model.len());
            assert!(a.iter().eq(a_model.iter().copied()), "round {round}");

            let mut union = a.clone();
            union.union_with(&b);
            assert!(union.iter().eq(a_model.union(&b_model).copied()));
            let mut both = a.clone();
            both.intersect_with(&b);
            assert!(both.iter().eq(a_model.intersection(&b_model).copied()));
            let mut only_a = a.clone();
            only_a.difference_with(&b);
            assert!(only_a.iter().eq(a_model.difference(&b_model).copied()));

            assert_eq!(a.is_subset(&b), a_model.is_subset(&b_model));
            assert!(both.is_subset(&a) && both.is_subset(&b) && a.is_subset(&union));
            assert_eq!(a.intersects(&b), !a_model.is_disjoint(&b_model));
            assert_eq!(a == b, a_model == b_model);
        }
    }

    #[test]
    fn bit_set_equality_ignores_capacity_and_steady_state_does_not_allocate() {
        let mut small = BitSet::new();
        small.insert(1);
        let mut large = BitSet::with_capacity(10_000);
        large.insert(1);
        assert_eq!(small, large);
        assert_eq!(large, small);
        large.insert(9_999);
        assert_ne!(small, large);
        assert!(small.is_subset(&large) && !large.is_subset(&small));
        // The union with a set that has spare capacity but no high members does not grow.
        let mut roomy = BitSet::with_capacity(100_000);
        roomy.insert(2);
        small.union_with(&roomy);
        assert_eq!(small.capacity(), 64);

        let other: BitSet = (0..10_000).step_by(7).collect();
        let stats = count_in(|| {
            large.clear();
            large.insert_range(100..9_000);
            large.union_with(&other);
            large.intersect_with(&other);
            large.difference_with(&small);
            assert_eq!(large.count(), large.iter().count());
        });
        assert_eq!(stats.events(), 0, "{stats:?}");
    }

    #[test]
    fn ring_buffer_overwrites_the_oldest() {
        let mut ring = RingBuffer::<u32, 3>::new();
        assert!(ring.is_empty());
        assert_eq!((ring.front(), ring.back()), (None, None));
        assert_eq!(ring.pop(), None);
        assert_eq!(ring.push(1), None);
        assert_eq!(ring.push(2), None);
        assert_eq!(ring.push(3), None);
        assert!(ring.is_full());
        assert_eq!(ring.try_push(4), Err(4));
        assert_eq!(ring.push(4), Some(1));
        assert_eq!(ring.push(5), Some(2));
        assert_eq!(ring.iter().copied().collect::<Vec<_>>(), [3, 4, 5]);
        assert_eq!(ring.iter().rev().copied().collect::<Vec<_>>(), [5, 4, 3]);
        assert_eq!(ring.iter().len(), 3);
        assert_eq!((ring.front(), ring.back()), (Some(&3), Some(&5)));
        assert_eq!((ring.get(1), ring.get(3)), (Some(&4), None));
        assert_eq!(ring.pop(), Some(3));
        assert_eq!(ring.try_push(6), Ok(()));
        assert_eq!(format!("{ring:?}"), "[4, 5, 6]");
        let (older, newer) = ring.as_slices();
        assert_eq!([older, newer].concat(), [4, 5, 6]);
        ring.clear();
        assert_eq!((ring.len(), ring.capacity()), (0, 3));
        // Zero capacity: nothing is ever stored.
        let mut none = RingBuffer::<u8, 0>::default();
        assert_eq!(none.push(9), Some(9));
        assert_eq!(none.try_push(9), Err(9));
        assert_eq!((none.pop(), none.iter().count()), (None, 0));
    }

    #[test]
    fn ring_buffer_matches_a_deque_under_random_operations() {
        let mut rng = Rng::new(77);
        let mut ours = RingBuffer::<u32, 7>::new();
        let mut model: VecDeque<u32> = VecDeque::new();
        for step in 0..30_000 {
            let value = rng.next_u32();
            match rng.index(5) {
                0 | 1 => {
                    let evicted = ours.push(value);
                    assert_eq!(evicted, if model.len() == 7 { model.pop_front() } else { None });
                    model.push_back(value);
                }
                2 => {
                    let ok = ours.try_push(value).is_ok();
                    assert_eq!(ok, model.len() < 7);
                    if ok {
                        model.push_back(value);
                    }
                }
                3 => assert_eq!(ours.pop(), model.pop_front()),
                _ => {
                    let index = rng.index(8);
                    assert_eq!(ours.get(index), model.get(index));
                }
            }
            assert_eq!(ours.len(), model.len(), "step {step}");
            assert!(ours.iter().eq(model.iter()));
            assert_eq!((ours.front(), ours.back()), (model.front(), model.back()));
        }
    }
}
