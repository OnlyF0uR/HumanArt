//! The project directory `[DOC-20]`, saving `[SAVE-02]` and loading with crash detection
//! `[SAVE-01]`.
//!
//! A project is a directory:
//!
//! ```text
//! <root>/character.anthropos   the document text (canonical form)
//! <root>/blobs/<hash>          content-addressed blobs (crate::blob)
//! <root>/journal/journal.log   commands since the last full save (crate::journal)
//! <root>/renders/              cached renders (owned by the renderer; never read here)
//! <root>/refs/                 reference photos and annotations (Reference::path is relative to it)
//! ```
//!
//! Saving checks the document, checks that every blob it references is already in the store
//! (blobs are written first, `[SAVE-02]`), writes the text atomically and starts a fresh journal
//! whose header names the hash of the text just written. Loading parses (migrating older versions),
//! reports blobs that are missing, and inspects the journal: records in a journal that continues the
//! loaded text mean the editor did not shut down cleanly and the command system (WP-08) replays them
//! `[SAVE-01]`.

use std::collections::BTreeSet;

use humanart_core::error::PlatformError;
use humanart_core::hash::{Hash128, hash128};
use humanart_core::platform::{FileSystem, join_path};

use crate::blob::{BlobStore, GcReport, reachable};
use crate::document::Document;
use crate::error::{DocError, Result};
use crate::journal::{Journal, JournalHeader, JournalLimits, JournalRecord};
use crate::registry::Registry;
use crate::text::{self, TextLimits};

/// The document file's name.
pub const DOCUMENT_FILE: &str = "character.anthropos";
/// The blob directory's name.
pub const BLOBS_DIR: &str = "blobs";
/// The journal directory's name.
pub const JOURNAL_DIR: &str = "journal";
/// The journal file's name inside [`JOURNAL_DIR`].
pub const JOURNAL_FILE: &str = "journal.log";
/// The render cache directory's name.
pub const RENDERS_DIR: &str = "renders";
/// The reference directory's name.
pub const REFS_DIR: &str = "refs";

/// The limits of loading a project.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadLimits {
    /// For the document text.
    pub text: TextLimits,
    /// For the journal.
    pub journal: JournalLimits,
}

/// What the journal says about the last session.
#[derive(Clone, Debug, PartialEq)]
pub enum JournalState {
    /// There is no journal (a project written by another tool, or deleted by hand).
    Missing,
    /// The journal continues the loaded text and holds no records: the last session ended cleanly.
    Clean,
    /// The journal continues the loaded text and holds records: the last session ended without a
    /// save; replay them onto the document `[SAVE-01]`. A torn last record has already been cut off.
    Recover(Vec<JournalRecord>),
    /// The journal belongs to another document or another save; it cannot be replayed.
    Stale(JournalHeader),
}

/// A loaded project.
#[derive(Clone, Debug, PartialEq)]
pub struct Loaded {
    /// The document, migrated to the current format version.
    pub document: Document,
    /// The hash of the document file's bytes (the journal's base).
    pub text_hash: Hash128,
    /// The format version of the file when it was older than the current one.
    pub migrated_from: Option<u32>,
    /// Blobs the document references that are not in the store.
    pub missing_blobs: Vec<Hash128>,
    /// The journal's state.
    pub journal: JournalState,
}

/// A project directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Project {
    root: String,
}

impl Project {
    /// The project at `root` (nothing is checked or created).
    pub fn at(root: &str) -> Self {
        Self { root: root.to_owned() }
    }

    /// The root directory.
    pub fn root(&self) -> &str {
        &self.root
    }

    /// The document file's path.
    pub fn document_path(&self) -> String {
        join_path(&self.root, DOCUMENT_FILE)
    }

    /// The blob directory's path.
    pub fn blobs_dir(&self) -> String {
        join_path(&self.root, BLOBS_DIR)
    }

    /// The journal file's path.
    pub fn journal_path(&self) -> String {
        join_path(&join_path(&self.root, JOURNAL_DIR), JOURNAL_FILE)
    }

    /// The render cache directory's path.
    pub fn renders_dir(&self) -> String {
        join_path(&self.root, RENDERS_DIR)
    }

    /// The reference directory's path.
    pub fn refs_dir(&self) -> String {
        join_path(&self.root, REFS_DIR)
    }

    /// The project's blob store.
    pub fn blobs<'fs>(&self, fs: &'fs dyn FileSystem) -> BlobStore<'fs> {
        BlobStore::new(fs, &self.blobs_dir())
    }

    /// The project's journal.
    pub fn journal<'fs>(&self, fs: &'fs dyn FileSystem) -> Journal<'fs> {
        Journal::new(fs, &self.journal_path())
    }

    /// Creates the directory layout and saves `doc` as the first version. Fails when the directory
    /// already holds a document.
    pub fn create(fs: &dyn FileSystem, root: &str, doc: &Document, registry: &Registry) -> Result<Self> {
        let project = Self::at(root);
        let path = project.document_path();
        if fs.exists(&path) {
            return Err(DocError::platform(&path, PlatformError::AlreadyExists));
        }
        for dir in [project.blobs_dir(), join_path(root, JOURNAL_DIR), project.renders_dir(), project.refs_dir()] {
            fs.create_dir_all(&dir).map_err(|e| DocError::platform(&dir, e))?;
        }
        project.save(fs, doc, registry)?;
        Ok(project)
    }

    /// Saves `doc` `[SAVE-02]` and returns the hash of the written text: checks it, checks that its
    /// blobs are in the store, writes the text atomically, then starts a fresh journal.
    pub fn save(&self, fs: &dyn FileSystem, doc: &Document, registry: &Registry) -> Result<Hash128> {
        doc.check(registry)?;
        let store = self.blobs(fs);
        if let Some(missing) = doc.blob_refs().into_iter().find(|h| !store.contains(*h)) {
            return Err(DocError::Blob { hash: missing, what: "referenced by the document but not in the blob store" });
        }
        let text = doc.write()?;
        let path = self.document_path();
        fs.create_dir_all(&self.root).map_err(|e| DocError::platform(&self.root, e))?;
        fs.write_atomic(&path, text.as_bytes()).map_err(|e| DocError::platform(&path, e))?;
        let hash = hash128(text.as_bytes());
        self.journal(fs).start(&JournalHeader { document: doc.meta.id, base: hash })?;
        Ok(hash)
    }

    /// Loads the project: parses and migrates the document, lists missing blobs and inspects the
    /// journal (cutting off a torn tail).
    pub fn load(&self, fs: &dyn FileSystem, registry: &Registry, limits: &LoadLimits) -> Result<Loaded> {
        let path = self.document_path();
        let meta = fs.metadata(&path).map_err(|e| DocError::platform(&path, e))?;
        if meta.len > limits.text.max_bytes {
            return Err(DocError::Limit { what: "bytes of document text", limit: limits.text.max_bytes });
        }
        let bytes = fs.read(&path).map_err(|e| DocError::platform(&path, e))?;
        let file = text::parse(&bytes, &limits.text)?;
        let version = file.version;
        let document = Document::from_text(&crate::migrate::migrate(file)?, registry)?;
        let text_hash = hash128(&bytes);
        let store = self.blobs(fs);
        let missing_blobs = document.blob_refs().into_iter().filter(|h| !store.contains(*h)).collect();
        let journal = match self.journal(fs).repair(&limits.journal)? {
            None => JournalState::Missing,
            Some(contents) => {
                if contents.header.document != document.meta.id || contents.header.base != text_hash {
                    JournalState::Stale(contents.header)
                } else if contents.records.is_empty() {
                    JournalState::Clean
                } else {
                    JournalState::Recover(contents.records)
                }
            }
        };
        let migrated_from = (version != crate::document::FORMAT_VERSION).then_some(version);
        Ok(Loaded { document, text_hash, migrated_from, missing_blobs, journal })
    }

    /// Deletes blobs that neither `doc` (with its snapshots), nor the `others` documents (open
    /// proposals), nor the `extra` roots reach `[DOC-22]`.
    pub fn collect_garbage(
        &self,
        fs: &dyn FileSystem,
        doc: &Document,
        others: &[&Document],
        extra: &[Hash128],
        registry: &Registry,
    ) -> Result<GcReport> {
        let store = self.blobs(fs);
        let mut documents = vec![doc];
        documents.extend_from_slice(others);
        let keep: BTreeSet<Hash128> = reachable(&documents, extra, &store, registry)?;
        store.collect_garbage(&keep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::Compression;
    use crate::document::tests::sample;
    use crate::registry::ParamValue;
    use crate::registry::tests::real_registry;
    use humanart_core::platform::MemoryFs;
    use humanart_io::json::Value;

    /// Puts every blob the sample references, with made-up content, and points the sample at them.
    fn stored_sample(fs: &MemoryFs, project: &Project, registry: &Registry) -> Document {
        let store = project.blobs(fs);
        let mut doc = sample(registry);
        let payload = store.put(b"sculpt deltas", Compression::Auto).unwrap();
        doc.sculpt_layers[0].payload = payload;
        let mut base = doc.clone();
        base.snapshots.clear();
        doc.snapshots[0].state = store.put(base.snapshot_text().unwrap().as_bytes(), Compression::Auto).unwrap();
        for h in doc.blob_refs() {
            if !store.contains(h) {
                // The sample's other hashes are hashes of known strings; store those strings.
                for s in [
                    &b"pore map"[..],
                    b"scar texture",
                    b"capture set",
                    b"guides",
                    b"density map",
                    b"glasses obj",
                    b"glasses mtl",
                ] {
                    if hash128(s) == h {
                        store.put(s, Compression::None).unwrap();
                    }
                }
            }
        }
        doc
    }

    #[test]
    fn doc_20_project_is_created_saved_and_loaded() {
        let registry = real_registry();
        let fs = MemoryFs::new();
        let project = Project::at("projects/ada");
        let doc = stored_sample(&fs, &project, &registry);
        let created = Project::create(&fs, "projects/ada", &doc, &registry).unwrap();
        assert_eq!(created, project);
        for dir in ["blobs", "journal", "renders", "refs"] {
            assert!(fs.metadata(&format!("projects/ada/{dir}")).unwrap().is_dir(), "{dir}");
        }
        let loaded = project.load(&fs, &registry, &LoadLimits::default()).unwrap();
        assert_eq!(loaded.document, doc);
        assert_eq!(loaded.journal, JournalState::Clean);
        assert!(loaded.missing_blobs.is_empty());
        assert_eq!(loaded.migrated_from, None);
        assert_eq!(loaded.text_hash, hash128(&fs.read("projects/ada/character.anthropos").unwrap()));
        assert!(Project::create(&fs, "projects/ada", &doc, &registry).is_err(), "never overwrites a project");
    }

    #[test]
    fn save_02_blobs_must_be_stored_before_the_document() {
        let registry = real_registry();
        let fs = MemoryFs::new();
        let doc = sample(&registry); // its blobs are not in this store
        assert!(matches!(Project::create(&fs, "p", &doc, &registry), Err(DocError::Blob { .. })));
        assert!(!fs.exists("p/character.anthropos"));
    }

    #[test]
    fn save_01_unclean_shutdown_is_detected_and_journal_replays() {
        let registry = real_registry();
        let fs = MemoryFs::new();
        let project = Project::at("p");
        let mut doc = stored_sample(&fs, &project, &registry);
        Project::create(&fs, "p", &doc, &registry).unwrap();
        let journal = project.journal(&fs);
        let rec = |seq| JournalRecord { seq, payload: Value::from("set identity.age") };
        journal.append(&rec(1)).unwrap();
        journal.append(&rec(2)).unwrap();
        fs.append(&project.journal_path(), b"3 deadbe").unwrap(); // torn by the crash
        let loaded = project.load(&fs, &registry, &LoadLimits::default()).unwrap();
        assert_eq!(loaded.journal, JournalState::Recover(vec![rec(1), rec(2)]));
        // The torn tail is gone, so the next append follows record 2.
        journal.append(&rec(3)).unwrap();
        assert_eq!(journal.read(&JournalLimits::DEFAULT).unwrap().unwrap().records.len(), 3);
        // After the next save the journal is fresh again.
        doc.set_param(&registry, "identity.age", ParamValue::Scalar(1.0), "human.artist").unwrap();
        project.save(&fs, &doc, &registry).unwrap();
        assert_eq!(project.load(&fs, &registry, &LoadLimits::default()).unwrap().journal, JournalState::Clean);
        // A document replaced behind the journal's back makes it stale.
        let mut other = doc.clone();
        other.meta.name = "Other".into();
        fs.write("p/character.anthropos", other.write().unwrap().as_bytes()).unwrap();
        assert!(matches!(
            project.load(&fs, &registry, &LoadLimits::default()).unwrap().journal,
            JournalState::Stale(_)
        ));
        project.journal(&fs).remove().unwrap();
        assert_eq!(project.load(&fs, &registry, &LoadLimits::default()).unwrap().journal, JournalState::Missing);
    }

    #[test]
    fn io_30_missing_blobs_are_reported_and_garbage_is_collected() {
        let registry = real_registry();
        let fs = MemoryFs::new();
        let project = Project::at("p");
        let doc = stored_sample(&fs, &project, &registry);
        Project::create(&fs, "p", &doc, &registry).unwrap();
        let store = project.blobs(&fs);
        let orphan = store.put(b"orphan blob", Compression::None).unwrap();
        let proposal_blob = store.put(b"proposal blob", Compression::None).unwrap();
        let report = project.collect_garbage(&fs, &doc, &[], &[proposal_blob], &registry).unwrap();
        assert_eq!(report.removed, vec![orphan]);
        assert!(store.contains(proposal_blob));
        let lost = doc.sculpt_layers[0].payload;
        fs.remove(&store.path(lost)).unwrap();
        let loaded = project.load(&fs, &registry, &LoadLimits::default()).unwrap();
        assert_eq!(loaded.missing_blobs, vec![lost]);
    }

    #[test]
    fn doc_23_every_fixture_version_loads_through_a_project() {
        let registry = real_registry();
        let fs = MemoryFs::new();
        fs.create_dir_all("old").unwrap();
        for version in 1..=crate::document::FORMAT_VERSION {
            let path = format!("{}/tests/fixtures/document-v{version}.anthropos", env!("CARGO_MANIFEST_DIR"));
            fs.write("old/character.anthropos", &std::fs::read(&path).unwrap()).unwrap();
            let loaded = Project::at("old").load(&fs, &registry, &LoadLimits::default()).unwrap();
            assert_eq!(loaded.migrated_from, (version != crate::document::FORMAT_VERSION).then_some(version));
            assert_eq!(loaded.journal, JournalState::Missing);
            assert_eq!(loaded.document.meta.name, "Ada \"test\" Lovelace");
        }
    }
}
