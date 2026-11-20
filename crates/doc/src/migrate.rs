//! Migrations of the document text from older format versions `[DOC-23]`.
//!
//! A migration is a function that rewrites a [`TextFile`] of version `n` into version `n + 1`;
//! [`MIGRATIONS`] holds them in order, so a file of any older version is brought to
//! [`FORMAT_VERSION`] by running the chain from its version on. Migrations work on the text form,
//! not on the typed document, because the typed document only knows the current version.
//!
//! Format version 1 is the first released version, so the chain is empty today. Every version
//! keeps a fixture in `crates/doc/tests/fixtures/document-v<n>.anthropos`, and the tests load every
//! one of them through [`migrate`]; adding version 2 means adding a migration here and keeping the
//! version 1 fixture, which the existing test then runs through it. [`migrate_with`] runs any chain
//! and is what the tests use to exercise the mechanism itself.

use crate::document::FORMAT_VERSION;
use crate::error::{DocError, Result};
use crate::text::TextFile;

/// One step: rewrites a file of version `n` into version `n + 1` (the caller sets the version).
pub type Migration = fn(&mut TextFile) -> Result<()>;

/// The chain: `MIGRATIONS[i]` turns version `i + 1` into version `i + 2`. Its length is always
/// `FORMAT_VERSION - 1`.
pub const MIGRATIONS: [Migration; (FORMAT_VERSION - 1) as usize] = [];

/// Brings a file of any supported version to [`FORMAT_VERSION`].
pub fn migrate(file: TextFile) -> Result<TextFile> {
    migrate_with(file, &MIGRATIONS, FORMAT_VERSION)
}

/// Brings a file to version `target` with `chain` (`chain[i]` turns version `i + 1` into `i + 2`).
/// Fails on version 0, on versions newer than `target`, on a chain of the wrong length and when a
/// step fails.
pub fn migrate_with(mut file: TextFile, chain: &[Migration], target: u32) -> Result<TextFile> {
    if chain.len() as u64 + 1 != u64::from(target) {
        return Err(DocError::Migration {
            from: 0,
            what: "the migration chain does not reach the target version".into(),
        });
    }
    if file.version == 0 || file.version > target {
        return Err(DocError::Version {
            format: "document",
            found: u64::from(file.version),
            supported: u64::from(target),
        });
    }
    while file.version < target {
        let from = file.version;
        let step = chain[(from - 1) as usize];
        step(&mut file).map_err(|e| DocError::Migration { from, what: e.to_string() })?;
        file.version = from + 1;
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::{Section, TextLimits, parse, write};
    use humanart_io::json::Value;

    /// Version 1 to 2 of a made-up history: the entry `nick` of `[meta]` became `name`.
    fn rename_nick(file: &mut TextFile) -> Result<()> {
        let meta = file.section_mut("meta", None).ok_or_else(|| DocError::field("meta", "", "missing"))?;
        if let Some(v) = meta.remove("nick") {
            meta.set("name", v)?;
        }
        Ok(())
    }

    /// Version 2 to 3: a `[settings]` section appeared with a default.
    fn add_settings(file: &mut TextFile) -> Result<()> {
        let mut s = Section::new("settings", None)?;
        s.set("protect_human_edits", Value::from(true))?;
        file.insert(1, s)
    }

    fn fail(_: &mut TextFile) -> Result<()> {
        Err(DocError::field("x", "", "broken"))
    }

    const CHAIN: [Migration; 2] = [rename_nick, add_settings];

    #[test]
    fn doc_23_chain_runs_from_every_older_version() {
        let v1 = parse(b"humanart-character 1\n\n[meta]\nnick = \"Ada\"\n", &TextLimits::DEFAULT).unwrap();
        let v2 = parse(b"humanart-character 2\n\n[meta]\nname = \"Ada\"\n", &TextLimits::DEFAULT).unwrap();
        let expected = "humanart-character 3\n\n[meta]\nname = \"Ada\"\n\n[settings]\nprotect_human_edits = true\n";
        assert_eq!(write(&migrate_with(v1, &CHAIN, 3).unwrap()).unwrap(), expected);
        assert_eq!(write(&migrate_with(v2, &CHAIN, 3).unwrap()).unwrap(), expected);
        let v3 = parse(expected.as_bytes(), &TextLimits::DEFAULT).unwrap();
        assert_eq!(write(&migrate_with(v3, &CHAIN, 3).unwrap()).unwrap(), expected, "current files pass unchanged");
    }

    #[test]
    fn doc_23_bad_versions_and_failing_steps_are_reported() {
        let file = |v: u32| TextFile::new(v);
        assert!(matches!(migrate_with(file(4), &CHAIN, 3), Err(DocError::Version { found: 4, .. })));
        assert!(matches!(migrate_with(file(0), &CHAIN, 3), Err(DocError::Version { found: 0, .. })));
        assert!(matches!(migrate_with(file(1), &CHAIN, 4), Err(DocError::Migration { .. })));
        assert!(matches!(migrate_with(file(1), &[fail], 2), Err(DocError::Migration { from: 1, .. })));
        assert!(matches!(migrate(file(FORMAT_VERSION + 1)), Err(DocError::Version { .. })));
        assert_eq!(migrate(file(FORMAT_VERSION)).unwrap().version, FORMAT_VERSION);
    }
}
