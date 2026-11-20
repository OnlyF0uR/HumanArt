//! The command journal `[DOC-14]`, `[SAVE-01]`: an append-only log of the commands committed since
//! the last full save, for crash recovery and provenance.
//!
//! The journal is a text file `journal/journal.log`: a header line naming the document and the
//! hash of the saved document text it continues from, then one line per record holding a sequence
//! number, a CRC-32 and the record as compact JSON (the command system, WP-08, decides what the JSON
//! holds). The layout is in `docs/format-anthropos.md`.
//!
//! A crash can interrupt an append, leaving a partial last line. Reading treats a damaged *last*
//! record (no line end, bad CRC, bad JSON) as the torn tail: it is reported in
//! [`JournalContents::torn_tail`] and dropped, and [`Journal::repair`] cuts it off before new records
//! are appended. A damaged record followed by intact ones is corruption and an error, because
//! replaying past a hole would rebuild the wrong document.

use humanart_core::hash::{Hash128, crc32_update};
use humanart_core::id::Id128;
use humanart_core::platform::{FileSystem, parent_path};
use humanart_io::json::{self, JsonLimits, Style, Value};

use crate::error::{DocError, Result};

/// The first word of the header line.
pub const HEADER_WORD: &str = "humanart-journal";

/// The journal format version.
pub const VERSION: u32 = 1;

/// The size limits of reading a journal `[ROB-02]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JournalLimits {
    /// Largest journal, in bytes.
    pub max_bytes: u64,
    /// Largest record line, in bytes.
    pub max_record_bytes: u64,
    /// Most records.
    pub max_records: u64,
    /// Deepest nesting inside one record's JSON.
    pub max_depth: u32,
}

impl JournalLimits {
    /// 1 GiB, 64 MiB per record, 2^24 records, 64 levels.
    pub const DEFAULT: Self =
        Self { max_bytes: 1 << 30, max_record_bytes: 64 << 20, max_records: 1 << 24, max_depth: 64 };
    /// Small limits for fuzz tests.
    pub const FUZZ: Self = Self { max_bytes: 1 << 20, max_record_bytes: 64 << 10, max_records: 4096, max_depth: 16 };
}

impl Default for JournalLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// What a journal continues from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JournalHeader {
    /// The document's id.
    pub document: Id128,
    /// The hash of the saved document text the records apply to.
    pub base: Hash128,
}

/// One journal record.
#[derive(Clone, Debug, PartialEq)]
pub struct JournalRecord {
    /// The sequence number; consecutive records differ by exactly 1.
    pub seq: u64,
    /// The record (a serialised command).
    pub payload: Value,
}

/// A journal as read.
#[derive(Clone, Debug, PartialEq)]
pub struct JournalContents {
    /// The header.
    pub header: JournalHeader,
    /// The intact records, in order.
    pub records: Vec<JournalRecord>,
    /// The byte offset of a torn last record that was dropped, if there was one.
    pub torn_tail: Option<usize>,
    /// The length of the intact part (header and intact records), in bytes.
    pub intact_len: usize,
}

/// The header line, with its line end.
pub fn encode_header(header: &JournalHeader) -> String {
    format!("{HEADER_WORD} {VERSION} {} {}\n", header.document, header.base.to_hex().as_str())
}

fn record_crc(seq: &str, json: &str) -> u32 {
    let crc = crc32_update(0, seq.as_bytes());
    let crc = crc32_update(crc, b" ");
    crc32_update(crc, json.as_bytes())
}

/// A record line, with its line end. Fails when the payload holds NaN or an infinity.
pub fn encode_record(record: &JournalRecord) -> Result<String> {
    let mut payload = String::new();
    json::write(&record.payload, Style::COMPACT, &mut payload)?;
    let seq = record.seq.to_string();
    Ok(format!("{seq} {:08x} {payload}\n", record_crc(&seq, &payload)))
}

fn parse_header(line: &str) -> Option<JournalHeader> {
    let mut parts = line.split(' ');
    if parts.next()? != HEADER_WORD || parts.next()? != "1" {
        return None;
    }
    let document = Id128::parse(parts.next()?).ok()?;
    let base = Hash128::parse(parts.next()?).ok()?;
    parts.next().is_none().then_some(JournalHeader { document, base })
}

fn parse_record(line: &str, limits: &JournalLimits) -> Option<JournalRecord> {
    let (seq_text, rest) = line.split_once(' ')?;
    let (crc_text, payload) = rest.split_once(' ')?;
    if seq_text.is_empty() || seq_text.len() > 20 || !seq_text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if seq_text.len() > 1 && seq_text.starts_with('0') {
        return None;
    }
    let seq: u64 = seq_text.parse().ok()?;
    if crc_text.len() != 8 || !crc_text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return None;
    }
    let crc = u32::from_str_radix(crc_text, 16).ok()?;
    if crc != record_crc(seq_text, payload) {
        return None;
    }
    let len = payload.len() as u64;
    let json_limits = JsonLimits {
        max_depth: limits.max_depth,
        max_input_bytes: len,
        max_string_bytes: len,
        max_members: len,
        max_values: len,
    };
    let payload = json::parse_with(payload.as_bytes(), &json_limits).ok()?;
    Some(JournalRecord { seq, payload })
}

/// Reads a journal. See the module documentation for the torn-tail rule.
pub fn read(bytes: &[u8], limits: &JournalLimits) -> Result<JournalContents> {
    if bytes.len() as u64 > limits.max_bytes {
        return Err(DocError::Limit { what: "bytes of journal", limit: limits.max_bytes });
    }
    let header_end =
        bytes.iter().position(|&b| b == b'\n').ok_or(DocError::Journal { line: 1, what: "no header line" })?;
    let header = std::str::from_utf8(&bytes[..header_end])
        .ok()
        .and_then(parse_header)
        .ok_or(DocError::Journal { line: 1, what: "bad header line" })?;
    let mut records: Vec<JournalRecord> = Vec::new();
    let mut at = header_end + 1;
    let mut line_number = 1;
    while at < bytes.len() {
        line_number += 1;
        let rest = &bytes[at..];
        let end = rest.iter().position(|&b| b == b'\n');
        let line_len = end.unwrap_or(rest.len());
        if line_len as u64 > limits.max_record_bytes {
            return Err(DocError::Limit { what: "bytes in one journal record", limit: limits.max_record_bytes });
        }
        let is_last = end.is_none_or(|e| at + e + 1 >= bytes.len());
        let parsed = end.and_then(|e| std::str::from_utf8(&rest[..e]).ok()).and_then(|l| parse_record(l, limits));
        let Some(record) = parsed else {
            if is_last {
                return Ok(JournalContents { header, records, torn_tail: Some(at), intact_len: at });
            }
            return Err(DocError::Journal { line: line_number, what: "damaged record before the end of the journal" });
        };
        if let Some(previous) = records.last()
            && previous.seq.checked_add(1) != Some(record.seq)
        {
            return Err(DocError::Journal { line: line_number, what: "sequence numbers are not consecutive" });
        }
        if records.len() as u64 >= limits.max_records {
            return Err(DocError::Limit { what: "journal records", limit: limits.max_records });
        }
        records.push(record);
        at += line_len + 1;
    }
    Ok(JournalContents { header, records, torn_tail: None, intact_len: at })
}

/// A journal file on a platform file system.
pub struct Journal<'fs> {
    fs: &'fs dyn FileSystem,
    path: String,
}

impl<'fs> Journal<'fs> {
    /// The journal at `path`.
    pub fn new(fs: &'fs dyn FileSystem, path: &str) -> Self {
        Self { fs, path: path.to_owned() }
    }

    /// The file's path.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Starts a new, empty journal after a full save, replacing the old one atomically.
    pub fn start(&self, header: &JournalHeader) -> Result<()> {
        let dir = parent_path(&self.path);
        if !dir.is_empty() {
            self.fs.create_dir_all(dir).map_err(|e| DocError::platform(dir, e))?;
        }
        self.fs
            .write_atomic(&self.path, encode_header(header).as_bytes())
            .map_err(|e| DocError::platform(&self.path, e))
    }

    /// Appends one record `[SAVE-01]`. The caller keeps the sequence numbers consecutive.
    pub fn append(&self, record: &JournalRecord) -> Result<()> {
        let line = encode_record(record)?;
        self.fs.append(&self.path, line.as_bytes()).map_err(|e| DocError::platform(&self.path, e))
    }

    /// Reads the journal, or `None` when there is none.
    pub fn read(&self, limits: &JournalLimits) -> Result<Option<JournalContents>> {
        match self.fs.metadata(&self.path) {
            Ok(meta) if meta.len > limits.max_bytes => {
                return Err(DocError::Limit { what: "bytes of journal", limit: limits.max_bytes });
            }
            Ok(_) => {}
            Err(humanart_core::error::PlatformError::NotFound) => return Ok(None),
            Err(e) => return Err(DocError::platform(&self.path, e)),
        }
        let bytes = self.fs.read(&self.path).map_err(|e| DocError::platform(&self.path, e))?;
        read(&bytes, limits).map(Some)
    }

    /// Cuts a torn tail off (rewriting the intact part atomically) so that new records follow the
    /// last intact one. Returns what was read.
    pub fn repair(&self, limits: &JournalLimits) -> Result<Option<JournalContents>> {
        let Some(contents) = self.read(limits)? else {
            return Ok(None);
        };
        if contents.torn_tail.is_some() {
            let bytes = self.fs.read(&self.path).map_err(|e| DocError::platform(&self.path, e))?;
            let intact = bytes.get(..contents.intact_len).unwrap_or(&bytes);
            self.fs.write_atomic(&self.path, intact).map_err(|e| DocError::platform(&self.path, e))?;
        }
        Ok(Some(contents))
    }

    /// Deletes the journal (nothing to recover).
    pub fn remove(&self) -> Result<()> {
        match self.fs.remove(&self.path) {
            Ok(()) | Err(humanart_core::error::PlatformError::NotFound) => Ok(()),
            Err(e) => Err(DocError::platform(&self.path, e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::hash::hash128;
    use humanart_core::platform::MemoryFs;
    use humanart_io::json::{Number, Object};

    fn header() -> JournalHeader {
        JournalHeader { document: Id128::from_u128(0xabc), base: hash128(b"saved text") }
    }

    fn record(seq: u64) -> JournalRecord {
        let mut o = Object::new();
        o.insert("kind", "set_params");
        o.insert("value", Value::from(Number::Float(seq as f64 * 0.5)));
        o.insert("label", "line\nbreak \"quoted\"");
        JournalRecord { seq, payload: Value::Object(o) }
    }

    fn journal(n: u64) -> String {
        let mut text = encode_header(&header());
        for seq in 10..10 + n {
            text.push_str(&encode_record(&record(seq)).unwrap());
        }
        text
    }

    #[test]
    fn doc_14_records_round_trip() {
        let text = journal(5);
        let contents = read(text.as_bytes(), &JournalLimits::DEFAULT).unwrap();
        assert_eq!(contents.header, header());
        assert_eq!(contents.records, (10..15).map(record).collect::<Vec<_>>());
        assert_eq!(contents.torn_tail, None);
        assert_eq!(contents.intact_len, text.len());
        assert!(read(encode_header(&header()).as_bytes(), &JournalLimits::DEFAULT).unwrap().records.is_empty());
    }

    #[test]
    fn save_01_a_torn_tail_is_dropped_and_reported() {
        let text = journal(3);
        let full = journal(4);
        // Every cut inside the last record leaves three intact records.
        for cut in text.len() + 1..full.len() {
            let contents = read(&full.as_bytes()[..cut], &JournalLimits::DEFAULT).unwrap();
            assert_eq!(contents.records.len(), 3, "cut {cut}");
            assert_eq!(contents.torn_tail, Some(text.len()));
        }
        // A flipped byte in the last record also counts as torn.
        let mut bad = full.clone().into_bytes();
        let last = bad.len() - 3;
        bad[last] ^= 1;
        assert_eq!(read(&bad, &JournalLimits::DEFAULT).unwrap().records.len(), 3);
    }

    #[test]
    fn save_01_damage_before_the_end_is_an_error() {
        let full = journal(4);
        let second_line = full.find('\n').unwrap() + 2;
        let mut bad = full.clone().into_bytes();
        bad[second_line] ^= 1;
        assert!(matches!(read(&bad, &JournalLimits::DEFAULT), Err(DocError::Journal { line: 2, .. })));
        // A gap in the sequence numbers.
        let mut gap = encode_header(&header());
        gap.push_str(&encode_record(&record(1)).unwrap());
        gap.push_str(&encode_record(&record(3)).unwrap());
        assert!(matches!(read(gap.as_bytes(), &JournalLimits::DEFAULT), Err(DocError::Journal { line: 3, .. })));
        assert!(read(b"", &JournalLimits::DEFAULT).is_err());
        assert!(read(b"humanart-journal 2 x y\n", &JournalLimits::DEFAULT).is_err());
        let tiny = JournalLimits { max_records: 2, ..JournalLimits::DEFAULT };
        assert!(matches!(read(full.as_bytes(), &tiny), Err(DocError::Limit { .. })));
    }

    #[test]
    fn save_01_journal_file_starts_appends_repairs_and_removes() {
        let fs = MemoryFs::new();
        let j = Journal::new(&fs, "proj/journal/journal.log");
        assert_eq!(j.read(&JournalLimits::DEFAULT).unwrap(), None);
        j.start(&header()).unwrap();
        j.append(&record(1)).unwrap();
        j.append(&record(2)).unwrap();
        fs.append(j.path(), b"3 0000").unwrap(); // a crash in the middle of an append
        let repaired = j.repair(&JournalLimits::DEFAULT).unwrap().unwrap();
        assert_eq!(repaired.records.len(), 2);
        assert!(repaired.torn_tail.is_some());
        j.append(&record(3)).unwrap();
        let contents = j.read(&JournalLimits::DEFAULT).unwrap().unwrap();
        assert_eq!(contents.records, vec![record(1), record(2), record(3)]);
        assert_eq!(contents.torn_tail, None);
        j.start(&header()).unwrap();
        assert!(j.read(&JournalLimits::DEFAULT).unwrap().unwrap().records.is_empty());
        j.remove().unwrap();
        j.remove().unwrap();
        assert_eq!(j.read(&JournalLimits::DEFAULT).unwrap(), None);
    }

    #[test]
    fn rob_01_journal_reader_survives_mutation_fuzz() {
        let corpus = vec![journal(6).into_bytes()];
        humanart_core::fuzz::run("journal", &corpus, 4000, 0x10f, std::time::Duration::from_secs(60), |bytes| {
            if let Ok(contents) = read(bytes, &JournalLimits::FUZZ) {
                assert!(contents.intact_len <= bytes.len());
                let mut text = encode_header(&contents.header);
                for r in &contents.records {
                    text.push_str(&encode_record(r).unwrap());
                }
                let again = read(text.as_bytes(), &JournalLimits::FUZZ).unwrap();
                assert_eq!(again.records, contents.records);
            }
        });
    }
}
