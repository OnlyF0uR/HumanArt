//! Proposals `[PROP-01]`..`[PROP-04]`: agent work the human reviews before it joins the document.
//!
//! A [`Proposal`] is a branch: a copy of the main document taken when the agent opened it (the
//! `base`), the branch document (`doc`, the base with the proposal's commands applied, which the
//! renderer shows for the A/B toggle) and the commands as history [`Entry`]s (the per-command
//! list). [`Proposal::changes`] and [`Proposal::region_heat`] are the data of the overlay's diff
//! list and per-region heat map; turning region weights into a coloured surface is the renderer's
//! job.
//!
//! Accepting takes the whole proposal or a selection of its commands. [`Proposal::select`] checks
//! dependencies `[PROP-02]`: a selected command that changes a target an earlier, unselected
//! command of the proposal changed builds on it (its `before` values are that command's `after`
//! values, or it edits an object that command created) and is refused with the commands it needs.
//! [`Proposal::conflicts`] compares the selection with what the main document changed since the
//! proposal was opened (`base` against the current main document, so it holds across restarts),
//! per target `[PROP-03]`: conflicts are reported, never resolved silently.
//!
//! Proposals are stored in the project `[PROP-04]` as `proposals/<id>.json`, written atomically
//! after every change, so an open proposal survives a restart or a crash. The file is JSON:
//!
//! ```text
//! {"format": "humanart-proposal", "version": 1, "id": "<id>", "author": "<author id>",
//!  "name": "<name>", "created": <unix ms>, "base": "<document text>", "entries": [<entry>...]}
//! ```
//!
//! with entries in the form of [`Entry::to_json`]. Reading checks the file size and the JSON
//! against [`ProposalLimits`] before anything is built, and re-applies every entry to the base.

use humanart_core::id::Id128;
use humanart_core::platform::{FileKind, FileSystem, join_path};
use humanart_io::json::{self, JsonLimits, Number, Object, Style, Value};

use crate::command::codec::CommandLimits;
use crate::command::{Change, Rejection, Target, any_overlap, apply_edits, describe, targets};
use crate::document::Document;
use crate::error::{DocError, Result};
use crate::history::Entry;
use crate::registry::{ParamValue, Registry};
use crate::text::{self, TextLimits};

/// The `format` member of a proposal file.
pub const FORMAT_WORD: &str = "humanart-proposal";

/// The proposal file format version.
pub const VERSION: u32 = 1;

/// The proposals directory's name inside a project.
pub const PROPOSALS_DIR: &str = "proposals";

/// The extension of proposal files.
pub const EXTENSION: &str = "json";

/// The limits of reading a proposal file `[ROB-01]`, `[ROB-02]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProposalLimits {
    /// Largest file, in bytes.
    pub max_bytes: u64,
    /// Deepest JSON nesting.
    pub max_depth: u32,
    /// Most commands in one proposal.
    pub max_entries: usize,
    /// For the base document text.
    pub text: TextLimits,
    /// For the entries' edits.
    pub commands: CommandLimits,
}

impl ProposalLimits {
    /// 1 GiB, depth 64, 65 536 commands, the default text and command limits.
    pub const DEFAULT: Self = Self {
        max_bytes: 1 << 30,
        max_depth: 64,
        max_entries: 1 << 16,
        text: TextLimits::DEFAULT,
        commands: CommandLimits::DEFAULT,
    };
}

impl Default for ProposalLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// One proposal `[PROP-01]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Proposal {
    /// Its id.
    pub id: Id128,
    /// The agent (or other author) that opened it; only it adds commands.
    pub author: String,
    /// A name for the review panel.
    pub name: String,
    /// When it was opened, Unix milliseconds.
    pub created_ms: u64,
    /// The main document when the proposal was opened.
    pub base: Document,
    /// The branch: `base` with every entry applied.
    pub doc: Document,
    /// The proposal's commands, oldest first.
    pub entries: Vec<Entry>,
}

fn bad(key: &str, what: &str) -> DocError {
    DocError::field("proposal", key, what)
}

impl Proposal {
    /// A new, empty proposal on `base`.
    pub fn new(id: Id128, author: &str, name: &str, created_ms: u64, base: Document) -> Self {
        Self {
            id,
            author: author.to_owned(),
            name: name.to_owned(),
            created_ms,
            doc: base.clone(),
            base,
            entries: Vec::new(),
        }
    }

    /// The ids of the proposal's commands, in order.
    pub fn command_ids(&self) -> Vec<Id128> {
        self.entries.iter().map(|e| e.id).collect()
    }

    /// The indices of the commands `selection` names (every command for `None`), in proposal
    /// order, after the dependency check `[PROP-02]`. An id that is not in the proposal is an
    /// error; a selected command that needs an unselected one is refused.
    pub fn select(&self, selection: Option<&[Id128]>) -> std::result::Result<Vec<usize>, SelectError> {
        let Some(ids) = selection else {
            return Ok((0..self.entries.len()).collect());
        };
        if let Some(missing) = ids.iter().find(|id| !self.entries.iter().any(|e| e.id == **id)) {
            return Err(SelectError::NotInProposal(*missing));
        }
        let chosen: Vec<bool> = self.entries.iter().map(|e| ids.contains(&e.id)).collect();
        let all_targets: Vec<Vec<Target>> = self.entries.iter().map(Entry::targets).collect();
        for (j, entry) in self.entries.iter().enumerate() {
            if !chosen[j] {
                continue;
            }
            let needs: Vec<Id128> = (0..j)
                .filter(|&i| !chosen[i] && !any_overlap(&all_targets[j], &all_targets[i]).is_empty())
                .map(|i| self.entries[i].id)
                .collect();
            if !needs.is_empty() {
                return Err(SelectError::Rejected(Rejection::ProposalDependency { command: entry.id, needs }));
            }
        }
        Ok((0..self.entries.len()).filter(|&i| chosen[i]).collect())
    }

    /// The selected commands that change a target the main document `main` changed since the
    /// proposal was opened, each with the shared targets `[PROP-03]`. Empty when they can merge.
    pub fn conflicts(&self, main: &Document, selected: &[usize]) -> Result<Vec<(Id128, Vec<String>)>> {
        let main_targets = targets(&self.base.diff(main)?);
        let mut out = Vec::new();
        for entry in selected.iter().filter_map(|&i| self.entries.get(i)) {
            let shared = any_overlap(&entry.targets(), &main_targets);
            if !shared.is_empty() {
                out.push((entry.id, shared.iter().map(ToString::to_string).collect()));
            }
        }
        Ok(out)
    }

    /// What the whole proposal changes against its base, for the review list `[PROP-01]`.
    pub fn changes(&self, registry: &Registry) -> Result<Vec<Change>> {
        Ok(describe(&self.base.diff(&self.doc)?, registry))
    }

    /// How much the proposal changes each region, for the heat map `[PROP-01]`: per region the sum
    /// of the changed scalar parameters' moves as a fraction of their range (1 for a changed
    /// parameter of another kind), plus 1 for every sculpt or skin layer masked to the region that
    /// was added, removed or changed. Sorted by region; regions with no change are left out.
    pub fn region_heat(&self, registry: &Registry) -> Result<Vec<(String, f64)>> {
        let mut heat: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
        for param in registry.params() {
            let Some(region) = &param.region else { continue };
            let old = self.base.param(registry, &param.id)?;
            let new = self.doc.param(registry, &param.id)?;
            if old == new {
                continue;
            }
            let weight = match (old, new, param.range) {
                (ParamValue::Scalar(a), ParamValue::Scalar(b), Some(r)) if r.max > r.min => {
                    ((b - a).abs() / (r.max - r.min)).min(1.0)
                }
                _ => 1.0,
            };
            *heat.entry(region.clone()).or_insert(0.0) += weight;
        }
        let mut bump = |region: Option<&String>| {
            if let Some(region) = region {
                *heat.entry(region.clone()).or_insert(0.0) += 1.0;
            }
        };
        let (old_sculpt, new_sculpt) = (&self.base.sculpt_layers, &self.doc.sculpt_layers);
        for layer in new_sculpt.iter().filter(|l| !old_sculpt.contains(l)) {
            bump(layer.region.as_ref());
        }
        for layer in old_sculpt.iter().filter(|l| !new_sculpt.iter().any(|n| n.id == l.id && n.region == l.region)) {
            bump(layer.region.as_ref());
        }
        let (old_skin, new_skin) = (&self.base.skin_layers, &self.doc.skin_layers);
        for layer in new_skin.iter().filter(|l| !old_skin.contains(l)) {
            bump(layer.region.as_ref());
        }
        for layer in old_skin.iter().filter(|l| !new_skin.iter().any(|n| n.id == l.id && n.region == l.region)) {
            bump(layer.region.as_ref());
        }
        Ok(heat.into_iter().collect())
    }

    /// The file form (see the module documentation).
    pub fn to_json(&self) -> Result<Value> {
        let mut o = Object::with_capacity(8);
        o.insert("format", FORMAT_WORD);
        o.insert("version", Value::from(Number::from(u64::from(VERSION))));
        o.insert("id", self.id.to_hex().as_str());
        o.insert("author", self.author.as_str());
        o.insert("name", self.name.as_str());
        o.insert("created", Value::from(Number::from(self.created_ms)));
        o.insert("base", self.base.write()?.as_str());
        o.insert("entries", Value::Array(self.entries.iter().map(Entry::to_json).collect()));
        Ok(Value::Object(o))
    }

    /// The file's bytes.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut out = String::new();
        json::write(&self.to_json()?, Style::COMPACT, &mut out)?;
        out.push('\n');
        Ok(out.into_bytes())
    }

    /// Reads a proposal file: checks size and JSON limits, parses the base document (migrating an
    /// older format version) and re-applies every entry to it.
    pub fn decode(bytes: &[u8], registry: &Registry, limits: &ProposalLimits) -> Result<Self> {
        if bytes.len() as u64 > limits.max_bytes {
            return Err(DocError::Limit { what: "bytes of a proposal file", limit: limits.max_bytes });
        }
        let len = bytes.len() as u64;
        let json_limits = JsonLimits {
            max_depth: limits.max_depth,
            max_input_bytes: len,
            max_string_bytes: len,
            max_members: len,
            max_values: len,
        };
        let value = json::parse_with(bytes, &json_limits)
            .map_err(|source| DocError::Json { file: "proposal".to_owned(), source })?;
        let o = value.as_object().ok_or_else(|| bad("", "expected an object"))?;
        if let Some((k, _)) = o.entries().iter().find(|(k, _)| {
            !matches!(k.as_str(), "format" | "version" | "id" | "author" | "name" | "created" | "base" | "entries")
        }) {
            return Err(bad(k, "unknown member"));
        }
        let text = |key: &str| -> Result<&str> {
            o.get(key).and_then(Value::as_str).ok_or_else(|| bad(key, "expected a string"))
        };
        if text("format")? != FORMAT_WORD {
            return Err(bad("format", "not a proposal file"));
        }
        let version = o.get("version").and_then(Value::as_u64).ok_or_else(|| bad("version", "expected an integer"))?;
        if version != u64::from(VERSION) {
            return Err(bad("version", "unsupported proposal format version"));
        }
        let id = Id128::parse(text("id")?).map_err(|_| bad("id", "expected an id"))?;
        let created_ms =
            o.get("created").and_then(Value::as_u64).ok_or_else(|| bad("created", "expected an unsigned integer"))?;
        let file = text::parse(text("base")?.as_bytes(), &limits.text)?;
        let base = Document::from_text(&crate::migrate::migrate(file)?, registry)?;
        let list = o.get("entries").and_then(Value::as_array).ok_or_else(|| bad("entries", "expected an array"))?;
        if list.len() > limits.max_entries {
            return Err(DocError::Limit { what: "commands in a proposal", limit: limits.max_entries as u64 });
        }
        let mut proposal = Self::new(id, text("author")?, text("name")?, created_ms, base);
        for item in list {
            let entry = Entry::from_json(item, &limits.commands)?;
            apply_edits(&mut proposal.doc, registry, &entry.edits)?;
            proposal.entries.push(entry);
        }
        Ok(proposal)
    }
}

/// Why a selection of proposal commands cannot be accepted.
#[derive(Clone, Debug, PartialEq)]
pub enum SelectError {
    /// The id names no command of the proposal.
    NotInProposal(Id128),
    /// The dependency check refused the selection.
    Rejected(Rejection),
}

/// The proposal files of a project.
pub struct ProposalStore<'fs> {
    fs: &'fs dyn FileSystem,
    dir: String,
}

/// What [`ProposalStore::load_all`] found.
#[derive(Debug, Default)]
pub struct LoadedProposals {
    /// The proposals that read back, sorted by creation time and id.
    pub proposals: Vec<Proposal>,
    /// Files that could not be read, with the reason; they are left on disk untouched.
    pub damaged: Vec<(String, DocError)>,
}

impl<'fs> ProposalStore<'fs> {
    /// The store in directory `dir` (the project's [`PROPOSALS_DIR`]).
    pub fn new(fs: &'fs dyn FileSystem, dir: &str) -> Self {
        Self { fs, dir: dir.to_owned() }
    }

    /// The file path of proposal `id`.
    pub fn path(&self, id: Id128) -> String {
        join_path(&self.dir, &format!("{}.{EXTENSION}", id.to_hex().as_str()))
    }

    /// Writes the proposal atomically `[PROP-04]`.
    pub fn save(&self, proposal: &Proposal) -> Result<()> {
        self.fs.create_dir_all(&self.dir).map_err(|e| DocError::platform(&self.dir, e))?;
        let path = self.path(proposal.id);
        let bytes = proposal.encode()?;
        self.fs.write_atomic(&path, &bytes).map_err(|e| DocError::platform(&path, e))
    }

    /// Deletes the proposal's file (nothing happens when there is none).
    pub fn remove(&self, id: Id128) -> Result<()> {
        let path = self.path(id);
        match self.fs.remove(&path) {
            Ok(()) | Err(humanart_core::error::PlatformError::NotFound) => Ok(()),
            Err(e) => Err(DocError::platform(&path, e)),
        }
    }

    /// Reads every proposal file. A missing directory means no proposals.
    pub fn load_all(&self, registry: &Registry, limits: &ProposalLimits) -> Result<LoadedProposals> {
        let entries = match self.fs.list(&self.dir) {
            Ok(entries) => entries,
            Err(humanart_core::error::PlatformError::NotFound) => return Ok(LoadedProposals::default()),
            Err(e) => return Err(DocError::platform(&self.dir, e)),
        };
        let mut out = LoadedProposals::default();
        for entry in entries {
            let suffix = format!(".{EXTENSION}");
            if entry.kind != FileKind::File || !entry.name.ends_with(&suffix) {
                continue;
            }
            let path = join_path(&self.dir, &entry.name);
            let read = || -> Result<Proposal> {
                let meta = self.fs.metadata(&path).map_err(|e| DocError::platform(&path, e))?;
                if meta.len > limits.max_bytes {
                    return Err(DocError::Limit { what: "bytes of a proposal file", limit: limits.max_bytes });
                }
                let bytes = self.fs.read(&path).map_err(|e| DocError::platform(&path, e))?;
                let proposal = Proposal::decode(&bytes, registry, limits)?;
                if self.path(proposal.id) != path {
                    return Err(bad("id", "does not match the file name"));
                }
                Ok(proposal)
            };
            match read() {
                Ok(p) => out.proposals.push(p),
                Err(e) => out.damaged.push((path, e)),
            }
        }
        out.proposals.sort_by(|a, b| a.created_ms.cmp(&b.created_ms).then(a.id.cmp(&b.id)));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::tests::sample;
    use crate::registry::tests::real_registry;
    use crate::text::{Edit, SectionRef};
    use humanart_core::platform::MemoryFs;

    fn scalar(v: f64) -> Value {
        Value::from(Number::Float(v))
    }

    fn set(doc: &Document, id: &str, v: f64) -> Edit {
        let before = doc.params.get(id).map(crate::registry::value_to_json);
        Edit::Set { section: SectionRef::new("params", None), key: id.into(), before, after: Some(scalar(v)) }
    }

    fn entry(n: u128, edits: Vec<Edit>) -> Entry {
        Entry {
            id: Id128::from_u128(n),
            author: "agent.assistant".into(),
            kind: "set_params".into(),
            label: format!("step {n}"),
            time_ms: 10 + n as u64,
            last_ms: 10 + n as u64,
            edits,
            reverts: None,
            open: None,
            elements: 0,
        }
    }

    /// A proposal with three commands: two on `face.nose.bridge_height` (the second builds on
    /// the first) and one on `face.jaw.width`.
    fn three(reg: &Registry) -> Proposal {
        let mut p = Proposal::new(Id128::from_u128(9), "agent.assistant", "Nose", 5, sample(reg));
        for (n, (id, v)) in
            [("face.nose.bridge_height", 0.5), ("face.nose.bridge_height", 0.6), ("face.jaw.width", 0.2)]
                .iter()
                .enumerate()
        {
            let e = entry(n as u128 + 1, vec![set(&p.doc, id, *v)]);
            apply_edits(&mut p.doc, reg, &e.edits).unwrap();
            p.entries.push(e);
        }
        p
    }

    #[test]
    fn prop02_selection_checks_dependencies() {
        let reg = real_registry();
        let p = three(&reg);
        assert_eq!(p.select(None), Ok(vec![0, 1, 2]));
        assert_eq!(p.select(Some(&[Id128::from_u128(3)])), Ok(vec![2]));
        assert_eq!(p.select(Some(&[Id128::from_u128(1), Id128::from_u128(3)])), Ok(vec![0, 2]));
        assert_eq!(
            p.select(Some(&[Id128::from_u128(2)])),
            Err(SelectError::Rejected(Rejection::ProposalDependency {
                command: Id128::from_u128(2),
                needs: vec![Id128::from_u128(1)]
            }))
        );
        assert_eq!(p.select(Some(&[Id128::from_u128(77)])), Err(SelectError::NotInProposal(Id128::from_u128(77))));
    }

    #[test]
    fn prop03_conflicts_are_found_per_target() {
        let reg = real_registry();
        let p = three(&reg);
        let mut main = p.base.clone();
        assert!(p.conflicts(&main, &[0, 1, 2]).unwrap().is_empty());
        let e = set(&main, "face.jaw.width", -0.3);
        apply_edits(&mut main, &reg, &[e]).unwrap();
        let c = p.conflicts(&main, &[0, 1, 2]).unwrap();
        assert_eq!(c, vec![(Id128::from_u128(3), vec!["params face.jaw.width".to_owned()])]);
        assert!(p.conflicts(&main, &[0, 1]).unwrap().is_empty());
    }

    #[test]
    fn prop01_changes_and_region_heat_describe_the_branch() {
        let reg = real_registry();
        let p = three(&reg);
        assert_eq!(p.changes(&reg).unwrap().len(), 2);
        let heat = p.region_heat(&reg).unwrap();
        assert!(!heat.is_empty());
        assert!(heat.iter().all(|(_, w)| *w > 0.0 && *w <= 2.0));
        // A layer moved to another region heats both regions.
        let mut moved = p.clone();
        moved.doc.sculpt_layers[0].region = Some("chin".into());
        let heat = moved.region_heat(&reg).unwrap();
        assert!(heat.iter().any(|(r, w)| r == "chin" && *w >= 1.0), "{heat:?}");
        let empty = Proposal::new(Id128::from_u128(1), "agent.assistant", "", 0, p.base.clone());
        assert!(empty.region_heat(&reg).unwrap().is_empty());
    }

    #[test]
    fn prop04_proposals_survive_a_restart() {
        let reg = real_registry();
        let fs = MemoryFs::new();
        let store = ProposalStore::new(&fs, "p/proposals");
        let p = three(&reg);
        store.save(&p).unwrap();
        fs.write("p/proposals/readme.txt", b"not a proposal").unwrap();
        fs.write(&store.path(Id128::from_u128(4)), b"{\"format\":").unwrap();
        let loaded = store.load_all(&reg, &ProposalLimits::DEFAULT).unwrap();
        assert_eq!(loaded.proposals, vec![p.clone()]);
        assert_eq!(loaded.damaged.len(), 1);
        store.remove(p.id).unwrap();
        store.remove(p.id).unwrap();
        assert!(ProposalStore::new(&fs, "none").load_all(&reg, &ProposalLimits::DEFAULT).unwrap().proposals.is_empty());
        let small = ProposalLimits { max_bytes: 64, ..ProposalLimits::DEFAULT };
        assert!(matches!(Proposal::decode(&p.encode().unwrap(), &reg, &small), Err(DocError::Limit { .. })));
    }

    #[test]
    fn rob01_proposal_decoding_survives_mutated_input() {
        let reg = real_registry();
        let corpus = vec![three(&reg).encode().unwrap()];
        let limits = ProposalLimits { max_bytes: 1 << 20, max_depth: 16, max_entries: 64, ..ProposalLimits::DEFAULT };
        humanart_core::fuzz::run("proposal", &corpus, 300, 0x9e0, std::time::Duration::from_secs(120), move |bytes| {
            if let Ok(p) = Proposal::decode(bytes, &reg, &limits) {
                assert!(p.entries.len() <= 64);
            }
        });
    }
}
