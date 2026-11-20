//! The content-addressed blob store `[IO-30]`, `[DOC-22]`.
//!
//! A blob is immutable binary content (sculpt deltas, textures, guide curves, meshes, snapshot
//! states) named by its 128-bit content hash (`humanart_core::hash::hash128`, MurmurHash3 x64_128)
//! in lower-case hexadecimal. It lives in one file `blobs/<hash>` holding a 16-byte header and the
//! content, either stored or compressed with the in-house LZ codec (`humanart_io::lz`); the layout
//! is in `docs/format-anthropos.md`. Reading checks the header, the size against [`BlobLimits`]
//! before anything is allocated, the LZ frame's own checksum, and finally that the content hashes to
//! the file's name, so a damaged or swapped file is always reported `[IO-30]` (integrity on load).
//!
//! The empty content hashes to [`Hash128::ZERO`], which the document uses for "no blob": it is never
//! written and always reads as empty. Blobs are written before the document that references them
//! `[SAVE-02]`, each atomically. [`BlobStore::collect_garbage`] deletes what [`reachable`] does not
//! reach from the document, its snapshots and any extra roots (proposals, WP-08).

use std::collections::BTreeSet;

use humanart_core::error::PlatformError;
use humanart_core::hash::{Hash128, hash128};
use humanart_core::platform::{FileKind, FileSystem, join_path};
use humanart_io::{Limits, lz};

use crate::document::Document;
use crate::error::{DocError, Result};
use crate::registry::Registry;
use crate::text::TextLimits;

/// The first four bytes of a blob file.
pub const MAGIC: [u8; 4] = *b"HABL";

/// The blob file version this build reads and writes.
pub const VERSION: u8 = 1;

/// The size of the blob file header.
pub const HEADER_LEN: usize = 16;

/// Flag bit: the payload is an LZ frame.
pub const FLAG_LZ: u8 = 1;

/// Content smaller than this is always stored uncompressed by [`Compression::Auto`].
pub const AUTO_MIN_BYTES: usize = 256;

/// How [`encode`] stores content.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compression {
    /// Stored as is.
    None,
    /// Always LZ-compressed.
    Lz,
    /// LZ-compressed when the content is at least [`AUTO_MIN_BYTES`] and compression saves at least
    /// an eighth; stored otherwise.
    Auto,
}

/// The size limit of reading blobs `[ROB-02]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlobLimits {
    /// Largest content, in bytes.
    pub max_bytes: u64,
}

impl BlobLimits {
    /// 2 GiB of content.
    pub const DEFAULT: Self = Self { max_bytes: 2 << 30 };
    /// 16 MiB, for fuzz tests.
    pub const FUZZ: Self = Self { max_bytes: 16 << 20 };
}

impl Default for BlobLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The blob file for `content` and its hash.
pub fn encode(content: &[u8], compression: Compression) -> (Hash128, Vec<u8>) {
    let hash = hash128(content);
    let compressed = match compression {
        Compression::None => None,
        Compression::Lz => Some(lz::compress(content)),
        Compression::Auto => {
            if content.len() < AUTO_MIN_BYTES {
                None
            } else {
                let frame = lz::compress(content);
                (frame.len() <= content.len() - content.len() / 8).then_some(frame)
            }
        }
    };
    let payload = compressed.as_deref().unwrap_or(content);
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(if compressed.is_some() { FLAG_LZ } else { 0 });
    out.extend_from_slice(&[0, 0]);
    out.extend_from_slice(&(content.len() as u64).to_le_bytes());
    out.extend_from_slice(payload);
    (hash, out)
}

/// Reads a blob file and checks it against `expected`, the hash its name gives.
pub fn decode(file: &[u8], expected: Hash128, limits: &BlobLimits) -> Result<Vec<u8>> {
    let bad = |what| DocError::Blob { hash: expected, what };
    let header = file.get(..HEADER_LEN).ok_or_else(|| bad("the file is shorter than its header"))?;
    if header[..4] != MAGIC {
        return Err(bad("not a blob file (bad magic)"));
    }
    if header[4] != VERSION {
        return Err(DocError::Version { format: "blob", found: u64::from(header[4]), supported: u64::from(VERSION) });
    }
    let flags = header[5];
    if flags & !FLAG_LZ != 0 || header[6] != 0 || header[7] != 0 {
        return Err(bad("unknown flags or reserved bytes set"));
    }
    let mut len_bytes = [0_u8; 8];
    len_bytes.copy_from_slice(&header[8..16]);
    let len = u64::from_le_bytes(len_bytes);
    if len > limits.max_bytes {
        return Err(DocError::Limit { what: "bytes in one blob", limit: limits.max_bytes });
    }
    let payload = &file[HEADER_LEN..];
    let content = if flags & FLAG_LZ != 0 {
        let lz_limits = Limits { max_bytes: len, ..Limits::DEFAULT };
        let content = lz::decompress(payload, &lz_limits)?;
        if content.len() as u64 != len {
            return Err(bad("the decompressed size differs from the header"));
        }
        content
    } else {
        if payload.len() as u64 != len {
            return Err(bad("the stored size differs from the header"));
        }
        payload.to_vec()
    };
    if hash128(&content) != expected {
        return Err(bad("the content does not match its hash"));
    }
    Ok(content)
}

/// What [`BlobStore::collect_garbage`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GcReport {
    /// Blobs kept.
    pub kept: usize,
    /// Blobs deleted.
    pub removed: Vec<Hash128>,
    /// Leftover temporary files deleted (from interrupted writes).
    pub temp_files_removed: usize,
}

/// A blob directory on a platform file system.
pub struct BlobStore<'fs> {
    fs: &'fs dyn FileSystem,
    dir: String,
    limits: BlobLimits,
}

impl<'fs> BlobStore<'fs> {
    /// The store in directory `dir` (created on the first write).
    pub fn new(fs: &'fs dyn FileSystem, dir: &str) -> Self {
        Self { fs, dir: dir.to_owned(), limits: BlobLimits::DEFAULT }
    }

    /// The same store with other read limits.
    #[must_use]
    pub fn with_limits(mut self, limits: BlobLimits) -> Self {
        self.limits = limits;
        self
    }

    /// The store's directory.
    pub fn dir(&self) -> &str {
        &self.dir
    }

    /// The path of a blob's file.
    pub fn path(&self, hash: Hash128) -> String {
        join_path(&self.dir, hash.to_hex().as_str())
    }

    /// Stores `content` and returns its hash. Content already in the store is not written again
    /// (blobs are immutable); empty content is never written and returns [`Hash128::ZERO`].
    pub fn put(&self, content: &[u8], compression: Compression) -> Result<Hash128> {
        if content.is_empty() {
            return Ok(Hash128::ZERO);
        }
        let hash = hash128(content);
        let path = self.path(hash);
        if self.fs.exists(&path) {
            return Ok(hash);
        }
        self.fs.create_dir_all(&self.dir).map_err(|e| DocError::platform(&self.dir, e))?;
        let (_, file) = encode(content, compression);
        self.fs.write_atomic(&path, &file).map_err(|e| DocError::platform(&path, e))?;
        Ok(hash)
    }

    /// The content of a blob, checked against its hash. [`Hash128::ZERO`] reads as empty.
    pub fn get(&self, hash: Hash128) -> Result<Vec<u8>> {
        if hash.is_zero() {
            return Ok(Vec::new());
        }
        let path = self.path(hash);
        let meta = self.fs.metadata(&path).map_err(|e| match e {
            PlatformError::NotFound => DocError::Blob { hash, what: "missing from the blob store" },
            e => DocError::platform(&path, e),
        })?;
        // A blob file is never much larger than its content (header + LZ frame overhead).
        let max_file = self.limits.max_bytes.saturating_add(self.limits.max_bytes / 64).saturating_add(1024);
        if meta.len > max_file {
            return Err(DocError::Limit { what: "bytes in one blob file", limit: max_file });
        }
        let file = self.fs.read(&path).map_err(|e| DocError::platform(&path, e))?;
        decode(&file, hash, &self.limits)
    }

    /// `true` when the blob's file exists ([`Hash128::ZERO`] always exists). Does not read it.
    pub fn contains(&self, hash: Hash128) -> bool {
        hash.is_zero() || self.fs.exists(&self.path(hash))
    }

    /// The hashes of every blob file, sorted. Files whose names are not hashes are ignored; a
    /// missing directory is an empty store.
    pub fn list(&self) -> Result<Vec<Hash128>> {
        let entries = match self.fs.list(&self.dir) {
            Ok(entries) => entries,
            Err(PlatformError::NotFound) => return Ok(Vec::new()),
            Err(e) => return Err(DocError::platform(&self.dir, e)),
        };
        let mut out: Vec<Hash128> = entries
            .iter()
            .filter(|e| e.kind == FileKind::File && e.name.len() == 32)
            .filter_map(|e| Hash128::parse(&e.name).ok())
            .collect();
        out.sort_unstable();
        Ok(out)
    }

    /// Reads every blob and returns the ones that fail their checks.
    pub fn verify_all(&self) -> Result<Vec<(Hash128, DocError)>> {
        let mut failures = Vec::new();
        for hash in self.list()? {
            if let Err(e) = self.get(hash) {
                failures.push((hash, e));
            }
        }
        Ok(failures)
    }

    /// Deletes every blob not in `reachable`, and leftover temporary files of interrupted writes.
    pub fn collect_garbage(&self, reachable: &BTreeSet<Hash128>) -> Result<GcReport> {
        let mut report = GcReport::default();
        for hash in self.list()? {
            if reachable.contains(&hash) {
                report.kept += 1;
            } else {
                let path = self.path(hash);
                self.fs.remove(&path).map_err(|e| DocError::platform(&path, e))?;
                report.removed.push(hash);
            }
        }
        if let Ok(entries) = self.fs.list(&self.dir) {
            for entry in entries.iter().filter(|e| e.kind == FileKind::File && e.name.ends_with(".tmp~")) {
                let path = join_path(&self.dir, &entry.name);
                self.fs.remove(&path).map_err(|e| DocError::platform(&path, e))?;
                report.temp_files_removed += 1;
            }
        }
        Ok(report)
    }
}

/// Every blob reachable from `documents` (the saved document, open proposals ...), from their
/// snapshots, and from `extra` roots `[DOC-22]`. Snapshot states are read from the store and parsed,
/// so a missing or damaged snapshot blob is an error: garbage collection must not run on an
/// incomplete reachability set.
pub fn reachable(
    documents: &[&Document],
    extra: &[Hash128],
    store: &BlobStore<'_>,
    registry: &Registry,
) -> Result<BTreeSet<Hash128>> {
    let mut out: BTreeSet<Hash128> = extra.iter().copied().filter(|h| !h.is_zero()).collect();
    for doc in documents {
        out.extend(doc.blob_refs());
        for snapshot in &doc.snapshots {
            let text = store.get(snapshot.state)?;
            // Snapshot texts hold no snapshots ([`Document::snapshot_text`]), so this is one level.
            let state = Document::parse(&text, registry, &TextLimits::DEFAULT)?;
            out.extend(state.blob_refs());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use humanart_core::platform::MemoryFs;
    use humanart_core::rng::Rng;

    fn content(n: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng::new(seed);
        // Compressible: repeated words with some noise.
        (0..n).map(|i| if rng.chance(0.1) { rng.next_u32() as u8 } else { b"skin layer "[i % 11] }).collect()
    }

    #[test]
    fn io_30_blob_files_round_trip_in_every_mode() {
        for compression in [Compression::None, Compression::Lz, Compression::Auto] {
            for n in [1, 100, 255, 256, 10_000] {
                let data = content(n, n as u64);
                let (hash, file) = encode(&data, compression);
                assert_eq!(hash, hash128(&data));
                assert_eq!(&file[..4], b"HABL");
                assert_eq!(decode(&file, hash, &BlobLimits::DEFAULT).unwrap(), data);
                let lz = file[5] & FLAG_LZ != 0;
                match compression {
                    Compression::None => assert!(!lz),
                    Compression::Lz => assert!(lz),
                    Compression::Auto => assert_eq!(lz, n >= 256, "{n}"),
                }
            }
        }
    }

    #[test]
    fn io_30_damage_is_detected() {
        let data = content(5000, 1);
        for compression in [Compression::None, Compression::Lz] {
            let (hash, file) = encode(&data, compression);
            assert!(decode(&file, hash128(b"other"), &BlobLimits::DEFAULT).is_err(), "wrong name");
            for i in 0..file.len() {
                let mut bad = file.clone();
                bad[i] ^= 0x10;
                assert!(decode(&bad, hash, &BlobLimits::DEFAULT).is_err(), "flipped byte {i} accepted");
            }
            for cut in [0, 8, 15, 16, file.len() - 1] {
                assert!(decode(&file[..cut], hash, &BlobLimits::DEFAULT).is_err());
            }
            let small = BlobLimits { max_bytes: 100 };
            assert!(matches!(decode(&file, hash, &small), Err(DocError::Limit { .. })));
        }
    }

    #[test]
    fn io_30_store_puts_dedupes_lists_and_collects_garbage() {
        let fs = MemoryFs::new();
        let store = BlobStore::new(&fs, "proj/blobs");
        assert_eq!(store.list().unwrap(), []);
        assert_eq!(store.put(b"", Compression::Auto).unwrap(), Hash128::ZERO);
        assert_eq!(store.get(Hash128::ZERO).unwrap(), b"");
        let a = store.put(&content(3000, 2), Compression::Auto).unwrap();
        let b = store.put(b"short blob", Compression::Auto).unwrap();
        assert_eq!(store.put(b"short blob", Compression::Lz).unwrap(), b, "same content, same blob");
        assert!(store.contains(a) && store.contains(b) && !store.contains(hash128(b"absent")));
        let mut both = vec![a, b];
        both.sort_unstable();
        assert_eq!(store.list().unwrap(), both);
        assert_eq!(store.get(b).unwrap(), b"short blob");
        assert!(matches!(store.get(hash128(b"absent")), Err(DocError::Blob { .. })));
        // A damaged file is found by verification.
        let mut file = fs.read(&store.path(a)).unwrap();
        let last = file.len() - 1;
        file[last] ^= 1;
        fs.write(&store.path(a), &file).unwrap();
        let failures = store.verify_all().unwrap();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, a);
        // Garbage collection keeps the reachable blobs and removes interrupted writes.
        fs.write("proj/blobs/0123.tmp~", b"partial").unwrap();
        fs.write("proj/blobs/notes.txt", b"not a blob").unwrap();
        let report = store.collect_garbage(&BTreeSet::from([b])).unwrap();
        assert_eq!(report, GcReport { kept: 1, removed: vec![a], temp_files_removed: 1 });
        assert_eq!(store.list().unwrap(), [b]);
        assert!(fs.exists("proj/blobs/notes.txt"));
    }

    #[test]
    fn doc_22_reachability_follows_documents_snapshots_and_extra_roots() {
        let registry = crate::registry::tests::real_registry();
        let fs = MemoryFs::new();
        let store = BlobStore::new(&fs, "blobs");
        let mut doc = crate::document::tests::sample(&registry);
        // Make the snapshot real: its state references one blob the current document no longer does.
        let old_payload = store.put(b"old sculpt", Compression::None).unwrap();
        let mut old = doc.clone();
        old.sculpt_layers[0].payload = old_payload;
        old.snapshots.clear();
        let state = store.put(old.snapshot_text().unwrap().as_bytes(), Compression::Auto).unwrap();
        doc.snapshots[0].state = state;
        let proposal = hash128(b"proposal blob");
        let set = reachable(&[&doc], &[proposal, Hash128::ZERO], &store, &registry).unwrap();
        assert!(set.contains(&old_payload) && set.contains(&state) && set.contains(&proposal));
        assert!(set.is_superset(&doc.blob_refs()));
        assert!(!set.contains(&Hash128::ZERO));
        // A missing snapshot state is an error, not an empty set.
        doc.snapshots[0].state = hash128(b"lost");
        assert!(reachable(&[&doc], &[], &store, &registry).is_err());
    }

    #[test]
    fn io_30_hash_has_no_collisions_on_near_identical_inputs() {
        // 2^18 inputs that differ in one counter and in length: no 128-bit collision, and none in
        // either 64-bit half (for a good hash the chance of one is about 2^-28).
        let mut full = BTreeSet::new();
        let mut h1 = BTreeSet::new();
        let mut h2 = BTreeSet::new();
        let mut buf = [0_u8; 40];
        for i in 0..(1_u32 << 18) {
            buf[..4].copy_from_slice(&i.to_le_bytes());
            let len = 4 + (i as usize % 37);
            let h = hash128(&buf[..len]);
            assert!(full.insert(h) && h1.insert(h.h1()) && h2.insert(h.h2()), "collision at {i}");
        }
    }

    #[test]
    fn rob_01_blob_decoder_survives_mutation_fuzz() {
        let data = content(4000, 3);
        let corpus = vec![encode(&data, Compression::None).1, encode(&data, Compression::Lz).1];
        let hash = hash128(&data);
        humanart_core::fuzz::run("blob", &corpus, 3000, 0xb10b, std::time::Duration::from_secs(60), move |bytes| {
            if let Ok(out) = decode(bytes, hash, &BlobLimits::FUZZ) {
                assert_eq!(hash128(&out), hash);
            }
        });
    }
}
