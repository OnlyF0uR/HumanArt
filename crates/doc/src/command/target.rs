//! What edits touch: the unit of conflict detection for selective undo `[CMD-04]` and proposals
//! `[PROP-02]`, `[PROP-03]`.

use std::fmt;

use crate::text::{Edit, SectionRef};

/// One thing an edit touches: an entry of a section, or (with `key` `None`) a whole section.
/// A parameter's value and its author entry are the same target.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Target {
    /// The section.
    pub section: SectionRef,
    /// The entry, or `None` for the whole section (inserted, removed or moved).
    pub key: Option<String>,
}

impl Target {
    /// The target of one edit.
    pub fn of(edit: &Edit) -> Self {
        match edit {
            Edit::Set { section, key, .. } => {
                let section =
                    if section.kind == "param_authors" { SectionRef::new("params", None) } else { section.clone() };
                Self { section, key: Some(key.clone()) }
            }
            Edit::Insert { section, .. } | Edit::Remove { section, .. } => {
                Self { section: section.reference(), key: None }
            }
        }
    }

    /// `true` when two targets share something: the same section and either the same entry or a
    /// whole-section change.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.section == other.section
            && match (&self.key, &other.key) {
                (Some(a), Some(b)) => a == b,
                _ => true,
            }
    }
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.key {
            Some(key) => write!(f, "{} {key}", self.section),
            None => write!(f, "{}", self.section),
        }
    }
}

/// The targets of `edits`, sorted, without repeats.
pub fn targets(edits: &[Edit]) -> Vec<Target> {
    let mut out: Vec<Target> = edits.iter().map(Target::of).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// The targets of `a` that overlap a target of `b`.
pub fn any_overlap(a: &[Target], b: &[Target]) -> Vec<Target> {
    a.iter().filter(|t| b.iter().any(|u| t.overlaps(u))).cloned().collect()
}
