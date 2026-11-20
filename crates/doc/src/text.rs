//! The line-oriented text syntax of character documents, and edits on it `[DOC-21]`.
//!
//! A file is a header line, then sections; a section is a `[kind]` or `[kind key]` line followed by
//! `key = value` lines, where the value is one compact JSON value (`humanart_io::json`, strict per
//! ADR-010). The grammar is in `docs/format-anthropos.md`. This module knows nothing about what the
//! sections mean; [`crate::document`] maps them to the typed document.
//!
//! **Canonical form.** The writer emits the header, then for each section one empty line, the
//! section line and the entries in ascending byte order of their keys, each as `key = value` with
//! the value in compact JSON. For any file the writer produced, parse then write gives the same
//! bytes `[AC-WP07-1]`. The parser also accepts hand-edited files: blank lines and `#` comment lines
//! anywhere after the header, CR LF line ends, spaces around `=`, and entries in any order (they are
//! sorted on reading); such files are rewritten in canonical form.
//!
//! **Edits.** Every change of a document can be written as a list of [`Edit`]s: set or remove one
//! entry, insert or remove one whole section. [`diff`] computes the edits between two files,
//! [`apply`] applies them atomically and checks that each one finds what it expects (so an edit made
//! against an older state is reported as a [`DocError::Conflict`] instead of overwriting newer
//! values), and [`invert`] gives the edits that undo them. The command system (WP-08) stores these as
//! the forward and inverse payloads of commands.
//!
//! **Hostile input `[ROB-01]`.** Lines, sections, entries and value nesting are counted against
//! [`TextLimits`] before anything is stored; the JSON parser bounds each value by the length of its
//! line. Parsing is one forward pass over the lines without recursion of its own.

use std::cmp::Ordering;
use std::fmt;

use humanart_io::json::{self, JsonLimits, Style, Value};

use crate::error::{DocError, Result};

/// The first word of the header line.
pub const HEADER_WORD: &str = "humanart-character";

/// The longest section kind, in bytes.
pub const MAX_KIND_LEN: usize = 64;

/// The longest entry or section key, in bytes.
pub const MAX_KEY_LEN: usize = 128;

/// The size limits of one parse `[ROB-02]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextLimits {
    /// Largest input, in bytes.
    pub max_bytes: u64,
    /// Longest line, in bytes without its line end.
    pub max_line_bytes: u64,
    /// Most sections in one file.
    pub max_sections: u64,
    /// Most entries in one section.
    pub max_entries: u64,
    /// Deepest nesting of arrays and objects inside one value.
    pub max_value_depth: u32,
}

impl TextLimits {
    /// The limits used when the caller does not choose: 256 MiB of text, 16 MiB per line, 2^20
    /// sections, 2^20 entries per section, 16 levels of nesting in a value.
    pub const DEFAULT: Self = Self {
        max_bytes: 256 << 20,
        max_line_bytes: 16 << 20,
        max_sections: 1 << 20,
        max_entries: 1 << 20,
        max_value_depth: 16,
    };

    /// Small limits for fuzz tests.
    pub const FUZZ: Self = Self {
        max_bytes: 1 << 20,
        max_line_bytes: 64 << 10,
        max_sections: 4096,
        max_entries: 4096,
        max_value_depth: 8,
    };
}

impl Default for TextLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// `true` when `kind` is a valid section kind: `[a-z][a-z0-9_]*`, at most [`MAX_KIND_LEN`] bytes.
pub fn is_kind(kind: &str) -> bool {
    let bytes = kind.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= MAX_KIND_LEN
        && bytes[0].is_ascii_lowercase()
        && bytes.iter().all(|&b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// `true` when `key` is a valid entry or section key: `[A-Za-z0-9][A-Za-z0-9._-]*`, at most
/// [`MAX_KEY_LEN`] bytes. Parameter ids, region names, hexadecimal ids and hashes all fit.
pub fn is_key(key: &str) -> bool {
    let bytes = key.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= MAX_KEY_LEN
        && bytes[0].is_ascii_alphanumeric()
        && bytes.iter().all(|&b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// Which section an edit or lookup means: its kind and its key. Sections of one file never share
/// both.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SectionRef {
    /// The section kind.
    pub kind: String,
    /// The section key, if the kind has one.
    pub key: Option<String>,
}

impl SectionRef {
    /// A reference to the section `[kind key]` or `[kind]`.
    pub fn new(kind: &str, key: Option<&str>) -> Self {
        Self { kind: kind.to_owned(), key: key.map(str::to_owned) }
    }
}

impl fmt::Display for SectionRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.key {
            Some(key) => write!(f, "{} {key}", self.kind),
            None => f.write_str(&self.kind),
        }
    }
}

/// One section: a kind, an optional key and entries kept in ascending byte order of their keys.
#[derive(Clone, Debug, PartialEq)]
pub struct Section {
    kind: String,
    key: Option<String>,
    entries: Vec<(String, Value)>,
}

impl Section {
    /// An empty section. Fails when `kind` or `key` breaks the grammar ([`is_kind`], [`is_key`]).
    pub fn new(kind: &str, key: Option<&str>) -> Result<Self> {
        if !is_kind(kind) {
            return Err(DocError::field(kind, "", "invalid section kind"));
        }
        if let Some(key) = key
            && !is_key(key)
        {
            return Err(DocError::field(kind, key, "invalid section key"));
        }
        Ok(Self { kind: kind.to_owned(), key: key.map(str::to_owned), entries: Vec::new() })
    }

    /// The section kind.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// The section key.
    pub fn key(&self) -> Option<&str> {
        self.key.as_deref()
    }

    /// The reference that finds this section.
    pub fn reference(&self) -> SectionRef {
        SectionRef { kind: self.kind.clone(), key: self.key.clone() }
    }

    /// The entries, in ascending byte order of their keys.
    pub fn entries(&self) -> &[(String, Value)] {
        &self.entries
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when the section has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn position(&self, key: &str) -> std::result::Result<usize, usize> {
        self.entries.binary_search_by(|(k, _)| k.as_str().cmp(key))
    }

    /// The value of an entry.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.position(key).ok().map(|i| &self.entries[i].1)
    }

    /// Sets an entry, keeping the order, and returns the value it replaces. Fails when `key` breaks
    /// the grammar.
    pub fn set(&mut self, key: &str, value: Value) -> Result<Option<Value>> {
        if !is_key(key) {
            return Err(DocError::field(self.reference().to_string(), key, "invalid entry key"));
        }
        Ok(match self.position(key) {
            Ok(i) => Some(std::mem::replace(&mut self.entries[i].1, value)),
            Err(i) => {
                self.entries.insert(i, (key.to_owned(), value));
                None
            }
        })
    }

    /// Removes an entry and returns its value.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.position(key).ok().map(|i| self.entries.remove(i).1)
    }

    fn label(&self) -> String {
        self.reference().to_string()
    }
}

/// A whole file: the format version from the header and the sections in file order.
#[derive(Clone, Debug, PartialEq)]
pub struct TextFile {
    /// The format version of the header line.
    pub version: u32,
    sections: Vec<Section>,
}

impl TextFile {
    /// An empty file of the given version.
    pub fn new(version: u32) -> Self {
        Self { version, sections: Vec::new() }
    }

    /// The sections, in file order.
    pub fn sections(&self) -> &[Section] {
        &self.sections
    }

    /// The index of the section `[kind key]`.
    pub fn find(&self, kind: &str, key: Option<&str>) -> Option<usize> {
        self.sections.iter().position(|s| s.kind == kind && s.key.as_deref() == key)
    }

    /// The section `[kind key]`.
    pub fn section(&self, kind: &str, key: Option<&str>) -> Option<&Section> {
        self.find(kind, key).map(|i| &self.sections[i])
    }

    /// The section `[kind key]`, mutable.
    pub fn section_mut(&mut self, kind: &str, key: Option<&str>) -> Option<&mut Section> {
        self.find(kind, key).map(|i| &mut self.sections[i])
    }

    /// The sections of one kind, in file order.
    pub fn sections_of<'a>(&'a self, kind: &'a str) -> impl Iterator<Item = &'a Section> + 'a {
        self.sections.iter().filter(move |s| s.kind == kind)
    }

    /// Appends a section. Fails when a section with the same kind and key exists.
    pub fn push(&mut self, section: Section) -> Result<()> {
        self.insert(self.sections.len(), section)
    }

    /// Inserts a section at `index` (clamped to the length). Fails when a section with the same kind
    /// and key exists.
    pub fn insert(&mut self, index: usize, section: Section) -> Result<()> {
        if self.find(&section.kind, section.key.as_deref()).is_some() {
            return Err(DocError::field(section.label(), "", "duplicate section"));
        }
        let index = index.min(self.sections.len());
        self.sections.insert(index, section);
        Ok(())
    }

    /// Removes and returns the section at `index`.
    pub fn remove(&mut self, index: usize) -> Option<Section> {
        (index < self.sections.len()).then(|| self.sections.remove(index))
    }

    /// Removes every section of one kind.
    pub fn remove_kind(&mut self, kind: &str) {
        self.sections.retain(|s| s.kind != kind);
    }
}

// --- Parsing ---------------------------------------------------------------------------------------

fn syntax(line: usize, what: &'static str) -> DocError {
    DocError::Syntax { line, what }
}

/// Parses a decimal `u32` with no sign and no superfluous leading zero.
fn parse_decimal(text: &str) -> Option<u32> {
    let bytes = text.as_bytes();
    if bytes.is_empty()
        || bytes.len() > 10
        || !bytes.iter().all(u8::is_ascii_digit)
        || (bytes.len() > 1 && bytes[0] == b'0')
    {
        return None;
    }
    text.parse().ok()
}

/// Reads the header line `humanart-character <version>` and returns the version.
pub fn parse_header(line: &str) -> Option<u32> {
    let rest = line.strip_prefix(HEADER_WORD)?.strip_prefix(' ')?;
    parse_decimal(rest)
}

/// Parses a file `[DOC-21]`. See the module documentation for what is accepted.
pub fn parse(bytes: &[u8], limits: &TextLimits) -> Result<TextFile> {
    if bytes.len() as u64 > limits.max_bytes {
        return Err(DocError::Limit { what: "bytes of document text", limit: limits.max_bytes });
    }
    let text = std::str::from_utf8(bytes).map_err(|_| syntax(0, "the text is not UTF-8"))?;
    let mut lines = text.split('\n').enumerate();
    let header = lines.next().map_or("", |(_, l)| l);
    let header = header.strip_suffix('\r').unwrap_or(header);
    let version =
        parse_header(header).ok_or_else(|| syntax(1, "the first line is not `humanart-character <version>`"))?;
    let mut file = TextFile::new(version);
    let value_limits = |len: usize| JsonLimits {
        max_depth: limits.max_value_depth,
        max_input_bytes: len as u64,
        max_string_bytes: len as u64,
        max_members: len as u64,
        max_values: len as u64,
    };
    let mut current: Option<Section> = None;
    for (index, raw) in lines {
        let number = index + 1;
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        if line.len() as u64 > limits.max_line_bytes {
            return Err(DocError::Limit { what: "bytes in one line", limit: limits.max_line_bytes });
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(inner) = line.strip_prefix('[') {
            let inner = inner.strip_suffix(']').ok_or_else(|| syntax(number, "a section line must end with `]`"))?;
            let (kind, key) = match inner.split_once(' ') {
                Some((kind, key)) => (kind, Some(key)),
                None => (inner, None),
            };
            if !is_kind(kind) {
                return Err(syntax(number, "invalid section kind"));
            }
            if key.is_some_and(|k| !is_key(k)) {
                return Err(syntax(number, "invalid section key"));
            }
            if let Some(done) = current.take() {
                finish_section(&mut file, done, limits)?;
            }
            if file.find(kind, key).is_some() {
                return Err(syntax(number, "duplicate section"));
            }
            current = Some(Section { kind: kind.to_owned(), key: key.map(str::to_owned), entries: Vec::new() });
            continue;
        }
        let Some(section) = current.as_mut() else {
            return Err(syntax(number, "an entry before the first section"));
        };
        let end = line.find([' ', '=']).ok_or_else(|| syntax(number, "an entry needs `key = value`"))?;
        let key = &line[..end];
        if !is_key(key) {
            return Err(syntax(number, "invalid entry key"));
        }
        let rest = line[end..].trim_start_matches(' ');
        let value_text = rest.strip_prefix('=').ok_or_else(|| syntax(number, "an entry needs `key = value`"))?;
        let value_text = value_text.trim_matches(' ');
        if value_text.is_empty() {
            return Err(syntax(number, "an entry without a value"));
        }
        if section.entries.len() as u64 >= limits.max_entries {
            return Err(DocError::Limit { what: "entries in one section", limit: limits.max_entries });
        }
        let value = json::parse_with(value_text.as_bytes(), &value_limits(value_text.len()))
            .map_err(|source| DocError::Value { line: number, source })?;
        section.entries.push((key.to_owned(), value));
    }
    if let Some(done) = current.take() {
        finish_section(&mut file, done, limits)?;
    }
    Ok(file)
}

/// Sorts a parsed section's entries, refuses duplicate keys and appends it.
fn finish_section(file: &mut TextFile, mut section: Section, limits: &TextLimits) -> Result<()> {
    if file.sections.len() as u64 >= limits.max_sections {
        return Err(DocError::Limit { what: "sections in one document", limit: limits.max_sections });
    }
    section.entries.sort_by(|a, b| a.0.cmp(&b.0));
    if section.entries.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err(DocError::field(section.label(), "", "duplicate entry key"));
    }
    file.sections.push(section);
    Ok(())
}

// --- Writing ---------------------------------------------------------------------------------------

/// Writes a file in canonical form (see the module documentation). Fails only when a value holds a
/// NaN or an infinity, which JSON cannot write.
pub fn write(file: &TextFile) -> Result<String> {
    let mut out = String::new();
    out.push_str(HEADER_WORD);
    out.push(' ');
    out.push_str(&file.version.to_string());
    out.push('\n');
    for section in &file.sections {
        out.push_str("\n[");
        out.push_str(&section.kind);
        if let Some(key) = &section.key {
            out.push(' ');
            out.push_str(key);
        }
        out.push_str("]\n");
        for (key, value) in &section.entries {
            out.push_str(key);
            out.push_str(" = ");
            json::write(value, Style::COMPACT, &mut out)?;
            out.push('\n');
        }
    }
    Ok(out)
}

// --- Edits -----------------------------------------------------------------------------------------

/// One change of a [`TextFile`].
#[derive(Clone, Debug, PartialEq)]
pub enum Edit {
    /// Sets, adds or removes one entry of an existing section. `before` is what the entry holds
    /// before (`None`: absent), `after` what it holds after (`None`: removed).
    Set {
        /// The section.
        section: SectionRef,
        /// The entry key.
        key: String,
        /// The value before the edit.
        before: Option<Value>,
        /// The value after the edit.
        after: Option<Value>,
    },
    /// Inserts a whole section at a position.
    Insert {
        /// The index the section has after the insertion.
        index: usize,
        /// The section.
        section: Section,
    },
    /// Removes a whole section from a position.
    Remove {
        /// The index the section has before the removal.
        index: usize,
        /// The section as it is before the removal (checked when applying).
        section: Section,
    },
}

impl Edit {
    /// The edit that undoes this one.
    #[must_use]
    pub fn inverse(&self) -> Self {
        match self {
            Self::Set { section, key, before, after } => {
                Self::Set { section: section.clone(), key: key.clone(), before: after.clone(), after: before.clone() }
            }
            Self::Insert { index, section } => Self::Remove { index: *index, section: section.clone() },
            Self::Remove { index, section } => Self::Insert { index: *index, section: section.clone() },
        }
    }

    /// The section the edit touches.
    pub fn target(&self) -> SectionRef {
        match self {
            Self::Set { section, .. } => section.clone(),
            Self::Insert { section, .. } | Self::Remove { section, .. } => section.reference(),
        }
    }
}

/// The edits that undo `edits`: each one inverted, in reverse order.
pub fn invert(edits: &[Edit]) -> Vec<Edit> {
    edits.iter().rev().map(Edit::inverse).collect()
}

fn apply_one(file: &mut TextFile, edit: &Edit) -> Result<()> {
    match edit {
        Edit::Set { section, key, before, after } => {
            let conflict = |what| DocError::Conflict { section: section.to_string(), key: key.clone(), what };
            let index = file.find(&section.kind, section.key.as_deref()).ok_or_else(|| conflict("no such section"))?;
            let target = &mut file.sections[index];
            if target.get(key) != before.as_ref() {
                return Err(conflict("the entry does not hold the expected value"));
            }
            if let Some(value) = after {
                target.set(key, value.clone())?;
            } else {
                if before.is_none() {
                    return Err(conflict("an edit that neither sets nor removes"));
                }
                target.remove(key);
            }
        }
        Edit::Insert { index, section } => {
            if *index > file.sections.len() {
                return Err(DocError::Conflict {
                    section: section.label(),
                    key: String::new(),
                    what: "index out of range",
                });
            }
            if file.find(&section.kind, section.key.as_deref()).is_some() {
                return Err(DocError::Conflict {
                    section: section.label(),
                    key: String::new(),
                    what: "section exists",
                });
            }
            file.sections.insert(*index, section.clone());
        }
        Edit::Remove { index, section } => {
            if file.sections.get(*index) != Some(section) {
                return Err(DocError::Conflict {
                    section: section.label(),
                    key: String::new(),
                    what: "the section at the index is not the expected one",
                });
            }
            file.sections.remove(*index);
        }
    }
    Ok(())
}

/// Applies `edits` in order, all or nothing: when one fails, the ones applied before it are undone
/// and the file is left as it was.
pub fn apply(file: &mut TextFile, edits: &[Edit]) -> Result<()> {
    for (done, edit) in edits.iter().enumerate() {
        if let Err(error) = apply_one(file, edit) {
            for applied in edits[..done].iter().rev() {
                // Undoing an edit that has just been applied restores exactly what it found, so the
                // inverse always matches; an error here would mean `apply_one` is not reversible.
                if apply_one(file, &applied.inverse()).is_err() {
                    break;
                }
            }
            return Err(error);
        }
    }
    Ok(())
}

/// The edits that turn `a` into `b` (the version is not compared). Applying the result to `a`
/// gives a file equal to `b`.
///
/// Sections are matched by kind and key. Sections only in `a` are removed, sections only in `b`
/// inserted, and sections in both are kept in place when their relative order allows (the longest
/// run in the same order stays, the others are removed and inserted at their new place); entries
/// of kept sections differ by [`Edit::Set`]s.
pub fn diff(a: &TextFile, b: &TextFile) -> Vec<Edit> {
    let mut edits = Vec::new();
    let in_b = |s: &Section| b.find(&s.kind, s.key.as_deref());
    // 1. Sections not in `b`, from the back so the indices stay valid.
    let mut current: Vec<(usize, &Section)> = Vec::new();
    for section in &a.sections {
        if let Some(j) = in_b(section) {
            current.push((j, section));
        }
    }
    for (i, section) in a.sections.iter().enumerate().rev() {
        if in_b(section).is_none() {
            edits.push(Edit::Remove { index: i, section: section.clone() });
        }
    }
    // 2. Common sections outside the longest increasing run of their positions in `b` move.
    let positions: Vec<usize> = current.iter().map(|(j, _)| *j).collect();
    let keep = longest_increasing(&positions);
    for i in (0..current.len()).rev() {
        if !keep[i] {
            edits.push(Edit::Remove { index: i, section: current[i].1.clone() });
        }
    }
    let kept: Vec<&Section> = current.iter().zip(&keep).filter(|(_, k)| **k).map(|((_, s), _)| *s).collect();
    // 3. Walk `b`: kept sections line up in order; everything else is inserted where `b` has it.
    let mut next = 0;
    for (j, target) in b.sections.iter().enumerate() {
        if let Some(old) = kept.get(next)
            && old.kind == target.kind
            && old.key == target.key
        {
            next += 1;
            diff_entries(old, target, &mut edits);
        } else {
            edits.push(Edit::Insert { index: j, section: target.clone() });
        }
    }
    edits
}

fn diff_entries(a: &Section, b: &Section, edits: &mut Vec<Edit>) {
    if a == b {
        return;
    }
    let reference = a.reference();
    let mut set = |key: &str, before: Option<&Value>, after: Option<&Value>| {
        edits.push(Edit::Set {
            section: reference.clone(),
            key: key.to_owned(),
            before: before.cloned(),
            after: after.cloned(),
        });
    };
    let (mut i, mut j) = (0, 0);
    while i < a.entries.len() || j < b.entries.len() {
        let order = match (a.entries.get(i), b.entries.get(j)) {
            (Some((ka, _)), Some((kb, _))) => ka.cmp(kb),
            (Some(_), None) => Ordering::Less,
            _ => Ordering::Greater,
        };
        match order {
            Ordering::Less => {
                set(&a.entries[i].0, Some(&a.entries[i].1), None);
                i += 1;
            }
            Ordering::Greater => {
                set(&b.entries[j].0, None, Some(&b.entries[j].1));
                j += 1;
            }
            Ordering::Equal => {
                if a.entries[i].1 != b.entries[j].1 {
                    set(&a.entries[i].0, Some(&a.entries[i].1), Some(&b.entries[j].1));
                }
                i += 1;
                j += 1;
            }
        }
    }
}

/// Marks the elements of one longest strictly increasing subsequence of `values` (patience sorting,
/// O(n log n)).
fn longest_increasing(values: &[usize]) -> Vec<bool> {
    let mut tails: Vec<usize> = Vec::new(); // index into `values` of the smallest tail per length
    let mut previous = vec![usize::MAX; values.len()];
    for (i, &v) in values.iter().enumerate() {
        let length = tails.partition_point(|&t| values[t] < v);
        if length > 0 {
            previous[i] = tails[length - 1];
        }
        if length == tails.len() {
            tails.push(i);
        } else {
            tails[length] = i;
        }
    }
    let mut keep = vec![false; values.len()];
    let mut at = tails.last().copied().unwrap_or(usize::MAX);
    while at != usize::MAX {
        keep[at] = true;
        at = previous[at];
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::rng::Rng;
    use humanart_io::json::Number;

    const SAMPLE: &str = "humanart-character 1\n\n[meta]\nid = \"00ff\"\nname = \"Ada \\\"A\\\"\"\n\n[params]\nface.nose.width = 0.25\nidentity.age = -1.5\n\n[sculpt_layer 0a1b]\nopacity = 1.0\nregion = null\nvisible = true\nweights = [1,2.5,{\"a\":[]}]\n";

    #[test]
    fn doc_21_canonical_text_round_trips_byte_identical() {
        let file = parse(SAMPLE.as_bytes(), &TextLimits::DEFAULT).unwrap();
        assert_eq!(file.version, 1);
        assert_eq!(file.sections().len(), 3);
        assert_eq!(file.section("params", None).unwrap().get("identity.age"), Some(&Value::from(Number::Float(-1.5))));
        assert_eq!(write(&file).unwrap(), SAMPLE);
    }

    #[test]
    fn lenient_input_is_rewritten_canonically() {
        let messy = "humanart-character 7\r\n# a comment\r\n[b k-1]\r\nz=1\r\n\r\n a = 2\r\n";
        // A leading space makes ` a = 2` an invalid key.
        assert!(matches!(parse(messy.as_bytes(), &TextLimits::DEFAULT), Err(DocError::Syntax { line: 6, .. })));
        let messy = "humanart-character 7\r\n# a comment\r\n[b k-1]\r\nz=1\r\n\r\na   =   2  \r\n";
        let file = parse(messy.as_bytes(), &TextLimits::DEFAULT).unwrap();
        assert_eq!(write(&file).unwrap(), "humanart-character 7\n\n[b k-1]\na = 2\nz = 1\n");
    }

    #[test]
    fn grammar_errors_name_the_line() {
        let cases: [(&str, usize); 12] = [
            ("", 1),
            ("humanart-character 01\n", 1),
            ("humanart-character\n", 1),
            ("humanart-character 1\nx = 1\n", 2),
            ("humanart-character 1\n[A]\n", 2),
            ("humanart-character 1\n[a b c]\n", 2),
            ("humanart-character 1\n[a\n", 2),
            ("humanart-character 1\n[a]\nk\n", 3),
            ("humanart-character 1\n[a]\nk = \n", 3),
            ("humanart-character 1\n[a]\nk = tru\n", 3),
            ("humanart-character 1\n[a]\n[a]\n", 3),
            ("humanart-character 1\n[a]\nk: 1\n", 3),
        ];
        for (text, line) in cases {
            match parse(text.as_bytes(), &TextLimits::DEFAULT) {
                Err(DocError::Syntax { line: l, .. } | DocError::Value { line: l, .. }) => {
                    assert_eq!(l, line, "{text:?}");
                }
                other => panic!("{text:?}: {other:?}"),
            }
        }
        assert!(matches!(
            parse(b"humanart-character 1\n[a]\nk = 1\nk = 2\n", &TextLimits::DEFAULT),
            Err(DocError::Field { .. })
        ));
        assert!(parse(b"humanart-character 1\n[a]\nk = 1\xff\n", &TextLimits::DEFAULT).is_err());
    }

    #[test]
    fn rob_02_limits_are_enforced() {
        let limits = TextLimits {
            max_sections: 2,
            max_entries: 2,
            max_line_bytes: 32,
            max_value_depth: 2,
            ..TextLimits::DEFAULT
        };
        let many = "humanart-character 1\n[a]\n[b]\n[c]\n";
        assert!(matches!(parse(many.as_bytes(), &limits), Err(DocError::Limit { .. })));
        let entries = "humanart-character 1\n[a]\nx = 1\ny = 1\nz = 1\n";
        assert!(matches!(parse(entries.as_bytes(), &limits), Err(DocError::Limit { .. })));
        let long = format!("humanart-character 1\n[a]\nx = \"{}\"\n", "a".repeat(40));
        assert!(matches!(parse(long.as_bytes(), &limits), Err(DocError::Limit { .. })));
        let deep = "humanart-character 1\n[a]\nx = [[[1]]]\n";
        assert!(matches!(parse(deep.as_bytes(), &limits), Err(DocError::Value { .. })));
        let big = TextLimits { max_bytes: 10, ..TextLimits::DEFAULT };
        assert!(matches!(parse(SAMPLE.as_bytes(), &big), Err(DocError::Limit { .. })));
    }

    #[test]
    fn writing_nan_fails() {
        let mut file = TextFile::new(1);
        let mut section = Section::new("a", None).unwrap();
        section.set("x", Value::from(Number::Float(f64::NAN))).unwrap();
        file.push(section).unwrap();
        assert!(write(&file).is_err());
    }

    #[test]
    fn section_api_keeps_order_and_refuses_bad_names() {
        assert!(Section::new("Bad", None).is_err());
        assert!(Section::new("ok", Some("bad key")).is_err());
        let mut s = Section::new("ok", Some("k")).unwrap();
        assert!(s.set("b", Value::from(true)).unwrap().is_none());
        assert!(s.set("a", Value::from(false)).unwrap().is_none());
        assert_eq!(s.set("b", Value::Null).unwrap(), Some(Value::from(true)));
        assert!(s.set("=", Value::Null).is_err());
        assert_eq!(s.entries()[0].0, "a");
        assert_eq!(s.remove("a"), Some(Value::from(false)));
        assert_eq!(s.len(), 1);
        let mut file = TextFile::new(1);
        file.push(s.clone()).unwrap();
        assert!(file.push(s).is_err());
    }

    fn random_file(rng: &mut Rng) -> TextFile {
        let mut file = TextFile::new(1);
        let kinds = ["meta", "layer", "groom"];
        for _ in 0..rng.index(8) {
            let kind = kinds[rng.index(kinds.len())];
            let key = if kind == "meta" { None } else { Some(format!("k{}", rng.index(6))) };
            if file.find(kind, key.as_deref()).is_some() {
                continue;
            }
            let mut section = Section::new(kind, key.as_deref()).unwrap();
            for _ in 0..rng.index(5) {
                let value = match rng.index(3) {
                    0 => Value::from(rng.index(3) as u64),
                    1 => Value::from(rng.chance(0.5)),
                    _ => Value::from(format!("s{}", rng.index(3))),
                };
                section.set(&format!("e{}", rng.index(5)), value).unwrap();
            }
            let at = rng.index(file.sections().len() + 1);
            file.insert(at, section).unwrap();
        }
        file
    }

    #[test]
    fn diff_apply_and_invert_are_exact_on_random_files() {
        let mut rng = Rng::new(0x0d0c_0007);
        for case in 0..3000 {
            let a = random_file(&mut rng);
            let b = random_file(&mut rng);
            let edits = diff(&a, &b);
            let mut x = a.clone();
            apply(&mut x, &edits).unwrap_or_else(|e| panic!("case {case}: {e}"));
            assert_eq!(x, b, "case {case}");
            apply(&mut x, &invert(&edits)).unwrap();
            assert_eq!(x, a, "case {case}");
            assert!(diff(&a, &a).is_empty());
            let text = write(&b).unwrap();
            assert_eq!(write(&parse(text.as_bytes(), &TextLimits::DEFAULT).unwrap()).unwrap(), text);
        }
    }

    #[test]
    fn apply_detects_conflicts_and_is_atomic() {
        let a = parse(SAMPLE.as_bytes(), &TextLimits::DEFAULT).unwrap();
        let mut b = a.clone();
        b.section_mut("params", None).unwrap().set("identity.age", Value::from(Number::Float(0.5))).unwrap();
        let edits = diff(&a, &b);
        assert_eq!(edits.len(), 1);
        // Someone else changed the same entry: the edit no longer finds its `before`.
        let mut c = a.clone();
        c.section_mut("params", None).unwrap().set("identity.age", Value::from(Number::Float(2.0))).unwrap();
        let mut stale = edits.clone();
        stale.insert(
            0,
            Edit::Set {
                section: SectionRef::new("meta", None),
                key: "x".into(),
                before: None,
                after: Some(Value::Null),
            },
        );
        let snapshot = c.clone();
        assert!(matches!(apply(&mut c, &stale), Err(DocError::Conflict { .. })));
        assert_eq!(c, snapshot, "a failed apply leaves the file untouched");
        let bad = Edit::Remove { index: 0, section: Section::new("zzz", None).unwrap() };
        assert!(apply(&mut c, &[bad]).is_err());
        let noop = Edit::Set { section: SectionRef::new("meta", None), key: "q".into(), before: None, after: None };
        assert!(apply(&mut c, &[noop]).is_err());
    }

    #[test]
    fn longest_increasing_finds_a_maximal_run() {
        let keep = longest_increasing(&[3, 0, 1, 4, 2]);
        assert_eq!(keep.iter().filter(|k| **k).count(), 3);
        assert!(longest_increasing(&[]).is_empty());
    }

    #[test]
    fn rob_01_text_parser_survives_mutation_fuzz() {
        let corpus = vec![SAMPLE.as_bytes().to_vec(), b"humanart-character 1\n[a k]\nx = {\"a\":[1,[2]]}\n".to_vec()];
        humanart_core::fuzz::run("doc-text", &corpus, 4000, 0x7e47, std::time::Duration::from_secs(60), |bytes| {
            if let Ok(file) = parse(bytes, &TextLimits::FUZZ) {
                let text = write(&file).unwrap();
                let again = parse(text.as_bytes(), &TextLimits::FUZZ).unwrap();
                assert_eq!(write(&again).unwrap(), text);
            }
        });
    }
}
