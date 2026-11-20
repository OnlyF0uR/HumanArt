//! Pair kerning: `GPOS` pair adjustment (lookup type 2, formats 1 and 2, also behind extension
//! lookups of type 9) and the `kern` table (format 0) `[UI-03]`.
//!
//! Only the horizontal advance adjustment of the first glyph of a pair is read (`XAdvance` of the
//! first value record), which is what "kerning" means for left-to-right text. Placement offsets, the
//! second glyph's value record, device tables and variation deltas are ignored: the default instance
//! of a variable font is what is measured.

use super::{FORMAT, Font, GlyphId, Table, invalid};
use crate::error::{Error, Result};

/// The most pair-adjustment subtables that are indexed from one `GPOS` table `[ROB-02]`. Real fonts
/// have a few dozen at most.
pub(super) const MAX_PAIR_SUBTABLES: usize = 1 << 16;

/// The most script, feature, lookup and subtable entries [`Gpos::parse`] examines `[ROB-02]`. Every
/// count on the way is 16 bits wide, but nested they multiply (65 535 scripts sharing one language
/// system of 65 535 features), so the walk as a whole has a budget. Real fonts need a few thousand.
pub(super) const MAX_LAYOUT_STEPS: u32 = 1 << 20;

/// Counts the entries the layout walk has examined and stops it at [`MAX_LAYOUT_STEPS`].
struct Budget(u32);

impl Budget {
    /// Accounts for one more entry.
    fn step(&mut self) -> Result<()> {
        self.0 += 1;
        if self.0 > MAX_LAYOUT_STEPS {
            return Err(Error::Limit {
                format: FORMAT,
                what: "GPOS layout entries examined",
                limit: u64::from(MAX_LAYOUT_STEPS),
                got: u64::from(self.0),
            });
        }
        Ok(())
    }
}

const TAG_KERN: [u8; 4] = *b"kern";
const TAG_DFLT: [u8; 4] = *b"DFLT";
const TAG_LATN: [u8; 4] = *b"latn";

/// `GPOS` lookup type: pair adjustment.
const LOOKUP_PAIR: u16 = 2;
/// `GPOS` lookup type: extension (a 32-bit offset to a subtable of another type).
const LOOKUP_EXTENSION: u16 = 9;
/// Value-format bit: the record has an `XAdvance`.
const VALUE_X_ADVANCE: u16 = 0x0004;

/// One kerning adjustment between two glyphs, in font units.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KerningPair {
    /// The first (left) glyph.
    pub left: GlyphId,
    /// The glyph that follows it.
    pub right: GlyphId,
    /// The change of the left glyph's advance; negative moves the pair together. Never 0.
    pub value: i16,
}

/// The pair-adjustment lookups of the feature `kern`, located once when the font is opened.
#[derive(Clone, Debug)]
pub(super) struct Gpos<'a> {
    table: Table<'a>,
    /// The offsets in `table` of every pair-adjustment subtable, lookup after lookup in lookup-list
    /// order (extension subtables already resolved).
    subtables: Vec<u32>,
    /// For each lookup that has pair subtables, the index one past its last entry in `subtables`.
    lookup_ends: Vec<u32>,
}

/// The size in bytes of a value record of the given format: one 16-bit field per set bit.
fn value_size(format: u16) -> usize {
    2 * format.count_ones() as usize
}

/// The `XAdvance` of the value record at `at`, 0 when the format has none; `None` when the record
/// lies outside the table.
fn x_advance(t: Table<'_>, at: usize, format: u16) -> Option<i16> {
    if format & VALUE_X_ADVANCE == 0 {
        return Some(0);
    }
    // XPlacement and YPlacement, when present, come first.
    t.i16(at.checked_add(2 * (format & 0x0003).count_ones() as usize)?)
}

/// The part of `t` that starts at the 16-bit offset stored at `at`; an error when it is outside.
fn at_offset16(t: Table<'_>, at: usize) -> Result<Table<'_>> {
    let offset = usize::from(t.need_u16(at)?);
    t.tail(offset).ok_or_else(|| invalid(t.base + at, "GPOS offset lies outside the table"))
}

/// The coverage index of `glyph` in a Coverage table (formats 1 and 2), by binary search.
fn coverage_index(t: Table<'_>, glyph: u16) -> Option<u16> {
    let count = usize::from(t.u16(2)?);
    let (mut lo, mut hi) = (0, count);
    match t.u16(0)? {
        1 => {
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let here = t.u16(4 + mid * 2)?;
                if here == glyph {
                    return Some(mid as u16);
                }
                if here < glyph {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            None
        }
        2 => {
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let record = 4 + mid * 6;
                let (start, end) = (t.u16(record)?, t.u16(record + 2)?);
                if glyph < start {
                    hi = mid;
                } else if glyph > end {
                    lo = mid + 1;
                } else {
                    return t.u16(record + 4)?.checked_add(glyph - start);
                }
            }
            None
        }
        _ => None,
    }
}

/// The class of `glyph` in a ClassDef table (formats 1 and 2); 0 for a glyph the table does not
/// list, as the format defines, and for a damaged table.
fn class_of(t: Table<'_>, glyph: u16) -> u16 {
    let find = || match t.u16(0)? {
        1 => {
            let index = glyph.checked_sub(t.u16(2)?)?;
            if index >= t.u16(4)? {
                return None;
            }
            t.u16(6 + usize::from(index) * 2)
        }
        2 => {
            let (mut lo, mut hi) = (0, usize::from(t.u16(2)?));
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let record = 4 + mid * 6;
                let (start, end) = (t.u16(record)?, t.u16(record + 2)?);
                if glyph < start {
                    hi = mid;
                } else if glyph > end {
                    lo = mid + 1;
                } else {
                    return t.u16(record + 4);
                }
            }
            None
        }
        _ => None,
    };
    find().unwrap_or(0)
}

/// What a pair subtable holds for one first glyph that it covers.
#[derive(Clone, Copy, Debug)]
enum Seconds<'a> {
    /// Format 1: the PairSet of the first glyph, its records sorted by second glyph.
    Glyphs { set: Table<'a>, count: usize },
    /// Format 2: the row of the first glyph's class in the class matrix.
    Classes { subtable: Table<'a>, row: usize, class_def: Option<Table<'a>>, class_count: u16 },
}

/// A pair subtable that covers a given first glyph.
#[derive(Clone, Copy, Debug)]
struct Covered<'a> {
    /// Which lookup (index into `Gpos::lookup_ends`) the subtable belongs to.
    lookup: usize,
    /// The value format of the first glyph.
    format1: u16,
    /// The size of one record: second glyph id (format 1 only) and both value records.
    record: usize,
    seconds: Seconds<'a>,
}

impl<'a> Covered<'a> {
    /// The subtable at `offset`, if it covers `left`.
    fn find(table: Table<'a>, offset: u32, lookup: usize, left: u16) -> Option<Self> {
        let t = table.tail(offset as usize)?;
        let index = usize::from(coverage_index(t.tail(usize::from(t.u16(2)?))?, left)?);
        let (format1, format2) = (t.u16(4)?, t.u16(6)?);
        let values = value_size(format1) + value_size(format2);
        match t.u16(0)? {
            1 => {
                if index >= usize::from(t.u16(8)?) {
                    return None;
                }
                let set = t.tail(usize::from(t.u16(10 + index * 2)?))?;
                let count = usize::from(set.u16(0)?);
                Some(Self { lookup, format1, record: 2 + values, seconds: Seconds::Glyphs { set, count } })
            }
            2 => {
                // A null offset means "every glyph is class 0".
                let class_def = |at: usize| match t.u16(at)? {
                    0 => Some(None),
                    offset => t.tail(usize::from(offset)).map(Some),
                };
                let class1 = class_def(8)?.map_or(0, |def| class_of(def, left));
                let (count1, count2) = (t.u16(12)?, t.u16(14)?);
                if class1 >= count1 {
                    return None;
                }
                // Checked: the product passes 32 bits for hostile counts, and usize is 32 bits on wasm.
                let row = values.checked_mul(usize::from(class1))?.checked_mul(usize::from(count2))?.checked_add(16)?;
                let seconds = Seconds::Classes { subtable: t, row, class_def: class_def(10)?, class_count: count2 };
                Some(Self { lookup, format1, record: values, seconds })
            }
            _ => None,
        }
    }

    /// The first glyph's advance adjustment when `right` follows it; `None` when the subtable does
    /// not apply to this pair (the next subtable of the lookup is then tried).
    fn value(&self, right: u16) -> Option<i16> {
        match self.seconds {
            Seconds::Glyphs { set, count } => {
                let (mut lo, mut hi) = (0, count);
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    let record = 2 + mid * self.record;
                    let second = set.u16(record)?;
                    if second == right {
                        return x_advance(set, record + 2, self.format1);
                    }
                    if second < right {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                None
            }
            Seconds::Classes { subtable, row, class_def, class_count } => {
                let class2 = class_def.map_or(0, |def| class_of(def, right));
                if class2 >= class_count {
                    return None;
                }
                x_advance(subtable, row.checked_add(self.record * usize::from(class2))?, self.format1)
            }
        }
    }
}

impl<'a> Gpos<'a> {
    /// Locates the pair-adjustment subtables of the feature `kern`.
    ///
    /// The lookups are those of every `kern` feature referenced by the default language system of
    /// the scripts `DFLT` and `latn`; a font that has neither script contributes every `kern`
    /// feature of its feature list. `Ok(None)` for a `GPOS` table of an unknown major version, which
    /// is then ignored. An offset on the way that points outside the table is an error, and so is a
    /// table whose lists are so heavily nested that the walk passes [`MAX_LAYOUT_STEPS`] entries.
    pub(super) fn parse(table: Table<'a>) -> Result<Option<Self>> {
        if table.need_u16(0)? != 1 {
            return Ok(None);
        }
        let features = at_offset16(table, 6)?;
        let lookups = at_offset16(table, 8)?;
        let feature_count = usize::from(features.need_u16(0)?);
        let lookup_count = usize::from(lookups.need_u16(0)?);
        if 2 + feature_count * 6 > features.len() {
            return Err(features.truncated(features.len()));
        }
        if 2 + lookup_count * 2 > lookups.len() {
            return Err(lookups.truncated(lookups.len()));
        }

        let mut budget = Budget(0);
        // Which features the chosen scripts use; without DFLT or latn, all of them.
        let mut feature_used = vec![false; feature_count];
        let mut any_script = false;
        if table.need_u16(4)? != 0 {
            let scripts = at_offset16(table, 4)?;
            for i in 0..usize::from(scripts.need_u16(0)?) {
                budget.step()?;
                let record = 2 + i * 6;
                let tag = scripts.need_u32(record)?.to_be_bytes();
                if tag != TAG_DFLT && tag != TAG_LATN {
                    continue;
                }
                any_script = true;
                let script = at_offset16(scripts, record + 4)?;
                if script.need_u16(0)? == 0 {
                    continue;
                }
                let lang_sys = at_offset16(script, 0)?;
                let required = usize::from(lang_sys.need_u16(2)?);
                if let Some(used) = feature_used.get_mut(required) {
                    *used = true;
                }
                for k in 0..usize::from(lang_sys.need_u16(4)?) {
                    budget.step()?;
                    let feature = usize::from(lang_sys.need_u16(6 + k * 2)?);
                    if let Some(used) = feature_used.get_mut(feature) {
                        *used = true;
                    }
                }
            }
        }

        let mut lookup_used = vec![false; lookup_count];
        for (i, _) in feature_used.iter().enumerate().filter(|&(_, &used)| used || !any_script) {
            let record = 2 + i * 6;
            if features.need_u32(record)?.to_be_bytes() != TAG_KERN {
                continue;
            }
            let feature = at_offset16(features, record + 4)?;
            for k in 0..usize::from(feature.need_u16(2)?) {
                budget.step()?;
                let lookup = usize::from(feature.need_u16(4 + k * 2)?);
                if let Some(used) = lookup_used.get_mut(lookup) {
                    *used = true;
                }
            }
        }

        let mut gpos = Self { table, subtables: Vec::new(), lookup_ends: Vec::new() };
        for (i, _) in lookup_used.iter().enumerate().filter(|&(_, &used)| used) {
            let lookup = at_offset16(lookups, 2 + i * 2)?;
            let kind = lookup.need_u16(0)?;
            if kind != LOOKUP_PAIR && kind != LOOKUP_EXTENSION {
                continue;
            }
            let before = gpos.subtables.len();
            for s in 0..usize::from(lookup.need_u16(4)?) {
                budget.step()?;
                let mut subtable = at_offset16(lookup, 6 + s * 2)?;
                if kind == LOOKUP_EXTENSION {
                    if subtable.need_u16(0)? != 1 || subtable.need_u16(2)? != LOOKUP_PAIR {
                        continue;
                    }
                    let offset = subtable.need_u32(4)? as usize;
                    subtable = subtable
                        .tail(offset)
                        .ok_or_else(|| invalid(subtable.base + 4, "GPOS extension offset lies outside the table"))?;
                }
                // Formats other than 1 and 2 do not exist; a subtable claiming one is skipped.
                let header = match subtable.need_u16(0)? {
                    1 => 10,
                    2 => 16,
                    _ => continue,
                };
                if subtable.len() < header {
                    return Err(subtable.truncated(subtable.len()));
                }
                if gpos.subtables.len() >= MAX_PAIR_SUBTABLES {
                    return Err(Error::Limit {
                        format: FORMAT,
                        what: "GPOS pair subtable count",
                        limit: MAX_PAIR_SUBTABLES as u64,
                        got: gpos.subtables.len() as u64 + 1,
                    });
                }
                // A table is addressed with 32-bit offsets, so this fits.
                gpos.subtables.push((subtable.base - table.base) as u32);
            }
            if gpos.subtables.len() > before {
                gpos.lookup_ends.push(gpos.subtables.len() as u32);
            }
        }
        Ok(Some(gpos))
    }

    /// Whether the feature `kern` has any pair-adjustment subtable.
    pub(super) fn has_pairs(&self) -> bool {
        !self.subtables.is_empty()
    }

    /// The number of lookups with pair subtables, and of pair subtables.
    pub(super) fn counts(&self) -> (usize, usize) {
        (self.lookup_ends.len(), self.subtables.len())
    }

    /// Collects into `out` the subtables that cover `left`, in application order.
    fn covering(&self, left: u16, out: &mut Vec<Covered<'a>>) {
        out.clear();
        let mut first = 0;
        for (lookup, &end) in self.lookup_ends.iter().enumerate() {
            let offsets = self.subtables.get(first..end as usize).unwrap_or(&[]);
            first = end as usize;
            out.extend(offsets.iter().filter_map(|&offset| Covered::find(self.table, offset, lookup, left)));
        }
    }

    /// The sum over the lookups of the first matching subtable's `XAdvance`, given the subtables
    /// that cover the first glyph.
    fn advance_among(covered: &[Covered<'a>], right: u16) -> i32 {
        let mut total = 0_i32;
        let mut applied = None;
        for subtable in covered {
            if applied == Some(subtable.lookup) {
                continue;
            }
            if let Some(value) = subtable.value(right) {
                total = total.saturating_add(i32::from(value));
                applied = Some(subtable.lookup);
            }
        }
        total
    }

    /// The adjustment of the advance of `left` when `right` follows it: every lookup contributes its
    /// first subtable that applies to the pair, as a shaper applies them.
    pub(super) fn pair_advance(&self, left: u16, right: u16) -> i32 {
        let mut total = 0_i32;
        let mut first = 0;
        for (lookup, &end) in self.lookup_ends.iter().enumerate() {
            let offsets = self.subtables.get(first..end as usize).unwrap_or(&[]);
            first = end as usize;
            let value = offsets
                .iter()
                .find_map(|&offset| Covered::find(self.table, offset, lookup, left).and_then(|c| c.value(right)));
            total = total.saturating_add(i32::from(value.unwrap_or(0)));
        }
        total
    }
}

/// The header of one `kern` subtable, in either flavour of the table.
struct KernSubtable {
    /// The offset of the format-specific data.
    data: usize,
    /// The declared length of the whole subtable, header included.
    length: usize,
    /// Whether it is a horizontal, non-cross-stream, kerning (not minimum) subtable of format 0.
    usable: bool,
    /// Whether its value replaces what earlier subtables accumulated.
    replaces: bool,
}

/// Reads the subtable header at `at` of a `kern` table (`apple`: the version 1.0 layout).
fn kern_subtable(kern: Table<'_>, at: usize, apple: bool) -> Option<KernSubtable> {
    if apple {
        let length = kern.u32(at)? as usize;
        let coverage = kern.u16(at.checked_add(4)?)?;
        // Bits 15..13: vertical, cross-stream, variation. The low byte is the format.
        Some(KernSubtable { data: at + 8, length, usable: coverage & 0xE0FF == 0, replaces: false })
    } else {
        let length = usize::from(kern.u16(at.checked_add(2)?)?);
        let coverage = kern.u16(at + 4)?;
        // Bit 0 horizontal, bit 1 minimum values, bit 2 cross-stream, bit 3 override; the high
        // byte is the format.
        Some(KernSubtable {
            data: at + 6,
            length,
            usable: coverage & 0xFF07 == 0x0001,
            replaces: coverage & 0x0008 != 0,
        })
    }
}

/// The value of the pair in a format 0 subtable whose data starts at `at`, by binary search over
/// its sorted pairs.
fn kern_format0_pair(kern: Table<'_>, at: usize, left: u16, right: u16) -> Option<i16> {
    let key = u32::from(left) << 16 | u32::from(right);
    let (mut lo, mut hi) = (0, usize::from(kern.u16(at)?));
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let record = at + 8 + mid * 6;
        let here = kern.u32(record)?;
        if here == key {
            return kern.i16(record + 4);
        }
        if here < key {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    None
}

/// The kerning of a pair from a `kern` table: the sum of its horizontal format 0 subtables (a
/// subtable with the override flag replaces the sum so far). Both the Windows layout (version 0)
/// and Apple's (version 1.0) are read; any other version, format or a damaged table gives 0 for
/// what cannot be read.
pub(super) fn kern_table_pair(kern: Table<'_>, left: u16, right: u16) -> i32 {
    let (count, mut at, apple) = match (kern.u16(0), kern.u16(2), kern.u32(4)) {
        (Some(0), Some(count), _) => (u32::from(count), 4, false),
        (Some(1), Some(0), Some(count)) => (count, 8, true),
        _ => return 0,
    };
    let mut total = 0_i32;
    for _ in 0..count {
        let Some(subtable) = kern_subtable(kern, at, apple) else { break };
        if subtable.usable
            && let Some(value) = kern_format0_pair(kern, subtable.data, left, right)
        {
            total = if subtable.replaces { i32::from(value) } else { total.saturating_add(i32::from(value)) };
        }
        // A length that does not even span the header would not advance: stop there. (Fonts whose
        // single subtable is too long for its 16-bit length field end the scan the same way.)
        if subtable.length <= subtable.data - at {
            break;
        }
        let Some(next) = at.checked_add(subtable.length) else { break };
        at = next;
    }
    total
}

impl Font<'_> {
    /// Calls `visit` for every ordered pair of glyphs from `glyphs` that has a non-zero kerning, in
    /// the order of `glyphs` (by left glyph, then by right glyph). This is what a font baker stores
    /// next to its atlas: `glyphs` is the set of glyphs in the atlas.
    ///
    /// The values are those of [`Font::kerning`]. The cost is one coverage search per left glyph and
    /// pair subtable plus one record search per pair, so the full set of a UI font (a few hundred
    /// glyphs) takes milliseconds. A glyph listed twice is reported twice.
    pub fn for_each_kerning_pair(&self, glyphs: &[GlyphId], mut visit: impl FnMut(KerningPair)) {
        let clamp = |value: i32| value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        match (&self.gpos, self.kern) {
            (Some(gpos), _) if gpos.has_pairs() => {
                let mut covered = Vec::new();
                for &left in glyphs {
                    gpos.covering(left.0, &mut covered);
                    if covered.is_empty() {
                        continue;
                    }
                    for &right in glyphs {
                        let value = clamp(Gpos::advance_among(&covered, right.0));
                        if value != 0 {
                            visit(KerningPair { left, right, value });
                        }
                    }
                }
            }
            (_, Some(kern)) => {
                for &left in glyphs {
                    for &right in glyphs {
                        let value = clamp(kern_table_pair(kern, left.0, right.0));
                        if value != 0 {
                            visit(KerningPair { left, right, value });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Every non-zero kerning among `glyphs`, in the order of [`Font::for_each_kerning_pair`].
    #[must_use]
    pub fn kerning_pairs(&self, glyphs: &[GlyphId]) -> Vec<KerningPair> {
        let mut out = Vec::new();
        self.for_each_kerning_pair(glyphs, |pair| out.push(pair));
        out
    }

    /// How many `GPOS` lookups of the feature `kern` hold pair adjustments, and how many pair
    /// subtables they have in total; `(0, 0)` for a font without `GPOS` kerning.
    #[must_use]
    pub fn gpos_pair_lookup_counts(&self) -> (usize, usize) {
        self.gpos.as_ref().map_or((0, 0), Gpos::counts)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// Big-endian bytes of a list of 16-bit words.
    fn words(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_be_bytes()).collect()
    }

    fn table(data: &[u8]) -> Table<'_> {
        Table { data, base: 0 }
    }

    #[test]
    fn coverage_formats_1_and_2() {
        let one = words(&[1, 4, 3, 7, 20, 300]);
        let t = table(&one);
        assert_eq!(coverage_index(t, 3), Some(0));
        assert_eq!(coverage_index(t, 20), Some(2));
        assert_eq!(coverage_index(t, 300), Some(3));
        assert_eq!(coverage_index(t, 4), None);
        assert_eq!(coverage_index(t, 301), None);
        // Ranges 10..=12 → 0.., 40..=40 → 3, 50..=59 → 4..
        let two = words(&[2, 3, 10, 12, 0, 40, 40, 3, 50, 59, 4]);
        let t = table(&two);
        assert_eq!(coverage_index(t, 10), Some(0));
        assert_eq!(coverage_index(t, 12), Some(2));
        assert_eq!(coverage_index(t, 40), Some(3));
        assert_eq!(coverage_index(t, 55), Some(9));
        assert_eq!(coverage_index(t, 13), None);
        assert_eq!(coverage_index(t, 9), None);
        assert_eq!(coverage_index(t, 60), None);
        // Unknown format, truncated tables and an overflowing index are "not covered".
        assert_eq!(coverage_index(table(&words(&[3, 1, 5])), 5), None);
        assert_eq!(coverage_index(table(&words(&[1, 4, 3])), 20), None);
        assert_eq!(coverage_index(table(&words(&[2, 1, 0, 0xFFFF, 0xFFFF])), 9), None);
        assert_eq!(coverage_index(table(&[]), 0), None);
    }

    #[test]
    fn class_def_formats_1_and_2() {
        let one = words(&[1, 100, 3, 5, 0, 7]);
        let t = table(&one);
        assert_eq!(class_of(t, 100), 5);
        assert_eq!(class_of(t, 101), 0);
        assert_eq!(class_of(t, 102), 7);
        assert_eq!(class_of(t, 99), 0);
        assert_eq!(class_of(t, 103), 0);
        let two = words(&[2, 2, 10, 19, 1, 30, 30, 2]);
        let t = table(&two);
        assert_eq!(class_of(t, 10), 1);
        assert_eq!(class_of(t, 19), 1);
        assert_eq!(class_of(t, 30), 2);
        assert_eq!(class_of(t, 20), 0);
        assert_eq!(class_of(t, 5), 0);
        assert_eq!(class_of(table(&words(&[9, 9])), 1), 0);
        assert_eq!(class_of(table(&words(&[2, 5, 10])), 12), 0);
    }

    #[test]
    fn value_records() {
        assert_eq!(value_size(0), 0);
        assert_eq!(value_size(0x0004), 2);
        assert_eq!(value_size(0x00FF), 16);
        let data = words(&[11, 22, 0xFFF6, 44]);
        let t = table(&data);
        assert_eq!(x_advance(t, 0, 0x0004), Some(11));
        assert_eq!(x_advance(t, 0, 0x0005), Some(22));
        assert_eq!(x_advance(t, 0, 0x0007), Some(-10));
        assert_eq!(x_advance(t, 0, 0x0003), Some(0));
        assert_eq!(x_advance(t, 6, 0x0005), None);
        assert_eq!(x_advance(t, usize::MAX, 0x0007), None);
    }

    /// A GPOS table with one script (`tag`), a `kern` feature (and a `mark` feature that must be
    /// ignored) and three lookups:
    ///
    /// - lookup 0, type 2: a format 1 subtable (glyph 5 followed by 6 → −50, by 9 → +20, with an
    ///   XPlacement in front of the XAdvance) and then a format 2 subtable (glyphs 5 and 7 are class
    ///   1, glyph 6 second class 1 → −30). Glyph 5 is covered by both: the first that applies wins.
    /// - lookup 1, type 9: an extension subtable to a format 1 pair subtable (7 followed by 6 → −7).
    /// - lookup 2, type 1 (single adjustment), referenced only by `mark`.
    #[allow(clippy::trivially_copy_pass_by_ref, reason = "call sites pass byte-string literals")]
    pub(in crate::ttf) fn sample_gpos(tag: &[u8; 4]) -> Vec<u8> {
        // Pair subtable A: format 1, value format 1 = XPlacement | XAdvance, value format 2 = 0.
        let mut a = words(&[1, 0, 0x0005, 0, 1, 12]);
        a.extend(words(&[2, 6, 99, 0xFFCE, 9, 99, 20])); // PairSet at 12: two records of 6 bytes.
        let coverage_a = a.len() as u16;
        a.extend(words(&[1, 1, 5]));
        a[2..4].copy_from_slice(&coverage_a.to_be_bytes());
        // Pair subtable B: format 2, value format 1 = XAdvance, 2 × 2 classes.
        let mut b = words(&[2, 0, 0x0004, 0, 0, 0, 2, 2]);
        b.extend(words(&[0, 0, 0, 0xFFE2])); // [class1][class2]: only (1, 1) = −30.
        let coverage_b = b.len() as u16;
        b.extend(words(&[2, 1, 5, 7, 0])); // glyphs 5..=7
        let class1_b = b.len() as u16;
        b.extend(words(&[2, 2, 5, 5, 1, 7, 7, 1]));
        let class2_b = b.len() as u16;
        b.extend(words(&[1, 6, 1, 1]));
        b[2..4].copy_from_slice(&coverage_b.to_be_bytes());
        b[8..10].copy_from_slice(&class1_b.to_be_bytes());
        b[10..12].copy_from_slice(&class2_b.to_be_bytes());
        // Pair subtable C (behind the extension): format 1, XAdvance only.
        let mut c = words(&[1, 0, 0x0004, 0, 1, 12]);
        c.extend(words(&[1, 6, 0xFFF9]));
        let coverage_c = c.len() as u16;
        c.extend(words(&[1, 1, 7]));
        c[2..4].copy_from_slice(&coverage_c.to_be_bytes());

        // Lookup 0: type 2, two subtables.
        let mut lookup0 = words(&[2, 0, 2, 10, 10 + a.len() as u16]);
        lookup0.extend(&a);
        lookup0.extend(&b);
        // Lookup 1: type 9, one extension subtable followed by its target.
        let mut lookup1 = words(&[9, 0, 1, 8, 1, 2, 0, 8]);
        lookup1.extend(&c);
        // Lookup 2: type 1.
        let lookup2 = words(&[1, 0, 0]);
        let mut lookup_list = words(&[3, 8, 8 + lookup0.len() as u16, 8 + (lookup0.len() + lookup1.len()) as u16]);
        lookup_list.extend(&lookup0);
        lookup_list.extend(&lookup1);
        lookup_list.extend(&lookup2);

        // Feature list: `mark` → lookup 2, `kern` → lookups 1 and 0 (order in the feature is free).
        let mut feature_list = words(&[2]);
        feature_list.extend(b"mark");
        feature_list.extend(words(&[14]));
        feature_list.extend(b"kern");
        feature_list.extend(words(&[20]));
        feature_list.extend(words(&[0, 1, 2]));
        feature_list.extend(words(&[0, 2, 1, 0]));

        // Script list: one script whose default language system uses both features.
        let mut script_list = words(&[1]);
        script_list.extend(tag);
        script_list.extend(words(&[8]));
        script_list.extend(words(&[4, 0])); // Script: default LangSys at 4, no other languages.
        script_list.extend(words(&[0, 0xFFFF, 2, 0, 1])); // LangSys: no required feature, features 0 and 1.

        let script_at = 10_u16;
        let feature_at = script_at + script_list.len() as u16;
        let lookup_at = feature_at + feature_list.len() as u16;
        let mut gpos = words(&[1, 0, script_at, feature_at, lookup_at]);
        gpos.extend(&script_list);
        gpos.extend(&feature_list);
        gpos.extend(&lookup_list);
        gpos
    }

    #[test]
    fn gpos_pair_formats_extension_and_first_match_per_lookup() {
        for tag in [b"DFLT", b"latn", b"cyrl"] {
            let data = sample_gpos(tag);
            let gpos = Gpos::parse(table(&data)).unwrap().unwrap();
            assert!(gpos.has_pairs());
            assert_eq!(gpos.counts(), (2, 3), "{tag:?}");
            // Format 1 wins for glyph 5 where it has the pair...
            assert_eq!(gpos.pair_advance(5, 6), -50);
            assert_eq!(gpos.pair_advance(5, 9), 20);
            // ...and the class subtable applies to its other pairs (class 0 second: value 0).
            assert_eq!(gpos.pair_advance(5, 7), 0);
            // Glyph 7: class pair −30 from lookup 0 plus −7 from the extension lookup.
            assert_eq!(gpos.pair_advance(7, 6), -37);
            // Glyph 6 is covered by the class subtable with class 0.
            assert_eq!(gpos.pair_advance(6, 6), 0);
            assert_eq!(gpos.pair_advance(1, 6), 0);
            assert_eq!(gpos.pair_advance(0xFFFF, 0xFFFF), 0);
            // The two-phase path used for enumeration agrees with the direct one.
            let mut covered = Vec::new();
            for left in 0..12 {
                gpos.covering(left, &mut covered);
                for right in 0..12 {
                    assert_eq!(Gpos::advance_among(&covered, right), gpos.pair_advance(left, right), "{left} {right}");
                }
            }
        }
    }

    #[test]
    fn gpos_script_selection_and_damage() {
        // With a latn script present, a kern feature that only another script uses is not applied.
        let mut data = sample_gpos(b"latn");
        let kern_tag = data.windows(4).position(|w| w == b"kern").unwrap();
        // LangSys feature indices: make it reference feature 0 (`mark`) twice.
        let lang_sys_features = 10 + 2 + 6 + 4 + 6;
        data[lang_sys_features..lang_sys_features + 4].copy_from_slice(&words(&[0, 0]));
        let gpos = Gpos::parse(table(&data)).unwrap().unwrap();
        assert!(!gpos.has_pairs());
        assert_eq!(gpos.pair_advance(5, 6), 0);
        // A required feature index counts as used.
        data[lang_sys_features - 4..lang_sys_features - 2].copy_from_slice(&words(&[1]));
        assert!(Gpos::parse(table(&data)).unwrap().unwrap().has_pairs());
        // Without the kern tag there is nothing.
        let mut data = sample_gpos(b"DFLT");
        data[kern_tag..kern_tag + 4].copy_from_slice(b"liga");
        assert!(!Gpos::parse(table(&data)).unwrap().unwrap().has_pairs());
        // Unknown major version: ignored. Truncations and wild offsets: errors, never panics.
        let data = sample_gpos(b"DFLT");
        let mut other = data.clone();
        other[1] = 2;
        assert!(Gpos::parse(table(&other)).unwrap().is_none());
        for cut in 0..data.len() {
            if let Ok(Some(gpos)) = Gpos::parse(table(&data[..cut])) {
                let _ = gpos.pair_advance(5, 6) + gpos.pair_advance(7, 6);
            }
        }
        assert!(Gpos::parse(table(&data[..9])).is_err());
        let mut wild = data.clone();
        wild[8..10].copy_from_slice(&[0xFF, 0xFF]);
        assert!(matches!(Gpos::parse(table(&wild)), Err(Error::Invalid { .. })));
    }

    /// The sorted `(left, right, value)` pairs of one `kern` subtable.
    pub(in crate::ttf) type KernPairs<'a> = &'a [(u16, u16, i16)];

    /// A version 0 `kern` table with the given subtables: (coverage word, sorted pairs).
    pub(in crate::ttf) fn kern_table(subtables: &[(u16, KernPairs<'_>)]) -> Vec<u8> {
        let mut out = words(&[0, subtables.len() as u16]);
        for (coverage, pairs) in subtables {
            out.extend(words(&[0, 14 + pairs.len() as u16 * 6, *coverage, pairs.len() as u16, 0, 0, 0]));
            for &(left, right, value) in *pairs {
                out.extend(words(&[left, right, value as u16]));
            }
        }
        out
    }

    #[test]
    fn kern_table_format_0() {
        let pairs: &[(u16, u16, i16)] = &[(1, 2, -40), (1, 5, 12), (3, 2, -8), (300, 2, 5)];
        let data = kern_table(&[(0x0001, pairs)]);
        let t = table(&data);
        for &(left, right, value) in pairs {
            assert_eq!(kern_table_pair(t, left, right), i32::from(value));
        }
        assert_eq!(kern_table_pair(t, 2, 1), 0);
        assert_eq!(kern_table_pair(t, 1, 3), 0);
        // Subtables add up; vertical, minimum, cross-stream and other formats are skipped; an
        // override subtable replaces the sum.
        let data = kern_table(&[
            (0x0001, &[(1, 2, -40)]),
            (0x0001, &[(1, 2, -2)]),
            (0x0000, &[(1, 2, 1000)]),
            (0x0003, &[(1, 2, 1000)]),
            (0x0005, &[(1, 2, 1000)]),
            (0x0201, &[(1, 2, 1000)]),
        ]);
        assert_eq!(kern_table_pair(table(&data), 1, 2), -42);
        let data = kern_table(&[(0x0001, &[(1, 2, -40)]), (0x0009, &[(1, 2, 7)]), (0x0001, &[(1, 2, 1)])]);
        assert_eq!(kern_table_pair(table(&data), 1, 2), 8);
        // Damage: truncation anywhere, a zero length (no progress) and unknown versions give 0 or
        // a partial sum, never a panic or a hang.
        for cut in 0..data.len() {
            let _ = kern_table_pair(table(&data[..cut]), 1, 2);
        }
        let mut stuck = data.clone();
        stuck[6..8].copy_from_slice(&[0, 0]);
        assert_eq!(kern_table_pair(table(&stuck), 1, 2), -40);
        assert_eq!(kern_table_pair(table(&words(&[2, 1, 0, 14, 1, 0])), 1, 2), 0);
    }

    #[test]
    fn kern_table_apple_layout() {
        // Version 1.0, one subtable: length u32, coverage (format 0, horizontal), tuple index.
        let mut data = words(&[1, 0, 0, 2]);
        for (coverage, value) in [(0x0000_u16, -25_i16), (0x8000, 99)] {
            data.extend(words(&[0, 8 + 8 + 6, coverage, 0, 1, 0, 0, 0, 4, 9, value as u16]));
        }
        assert_eq!(kern_table_pair(table(&data), 4, 9), -25);
        assert_eq!(kern_table_pair(table(&data), 4, 8), 0);
        for cut in 0..data.len() {
            let _ = kern_table_pair(table(&data[..cut]), 4, 9);
        }
    }
}
