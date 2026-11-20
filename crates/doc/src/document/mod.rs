//! The character document `[DOC-01]`..`[DOC-14]`: everything that defines one character, as plain
//! values with stable ids.
//!
//! The document is a value: the command system (WP-08) changes it only through
//! [`Document::apply_edits`] with edits computed by [`Document::diff`] (or built by hand), so every
//! change is a value diff on the text form ([`crate::text::Edit`]) and its inverse is exact.
//! Objects in lists (layers, grooms, presets, locks, pins, snapshots, attachments, references)
//! carry random [`Id128`]s that never change; list order is meaningful where the specification says
//! so (sculpt and skin layer stacks, bottom first).
//!
//! What is stored where:
//!
//! - `[DOC-01]` [`Metadata`] and [`Author`]s; the format version is the text header's.
//! - `[DOC-02]` [`Document::params`]: only values that differ from the registry default, with the
//!   author of the last change in [`Document::param_authors`] (for `[LOCK-02]`).
//! - `[DOC-05]`..`[DOC-07]` (procedural skin, eyes, mouth) are registry parameters in the skin, eyes
//!   and mouth groups, so they live in `params`; maps and iris textures are `asset_ref` parameters.
//! - `[DOC-09]` current expression weights are the expression parameters; presets are
//!   [`ExpressionPreset`] and [`PosePreset`], the current preview pose is [`Document::pose`].
//! - `[DOC-14]` the journal lives next to the document ([`crate::journal`]), not inside it.

mod codec;

use std::collections::{BTreeMap, BTreeSet};

use humanart_core::hash::Hash128;
use humanart_core::id::Id128;

use crate::error::{DocError, Result};
use crate::registry::{ParamValue, Registry};
use crate::text::{self, Edit, TextFile, TextLimits};

pub use codec::{SNAPSHOT_EXCLUDED_KINDS, section_kind};

/// The document format version this build writes `[DOC-23]`.
pub const FORMAT_VERSION: u32 = 1;

/// The longest display name (document, layer, preset ...), in bytes.
pub const MAX_NAME_BYTES: usize = 256;

/// `[DOC-01]` Metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Metadata {
    /// The document's random id.
    pub id: Id128,
    /// The character's name.
    pub name: String,
    /// The name of the template the document was made with (`TemplateInfo::name`).
    pub template: String,
    /// The template version (`TemplateInfo::version`).
    pub template_version: u32,
    /// Creation time, Unix milliseconds.
    pub created_ms: u64,
    /// Last modification time, Unix milliseconds.
    pub modified_ms: u64,
}

/// Project-wide policies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    /// `[LOCK-02]`: values and layers last changed by a human are protected from agent commands.
    pub protect_human_edits: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self { protect_human_edits: true }
    }
}

/// Whether an author is a person or an agent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AuthorKind {
    /// A person at the editor.
    Human,
    /// An AI agent or a script acting for one.
    Agent,
}

/// `[DOC-01]` One author. Everything that records an author names one of these by id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Author {
    /// Stable id (a text key: `human.jerry`, `agent.claude`).
    pub id: String,
    /// Human or agent.
    pub kind: AuthorKind,
    /// The name shown in the history.
    pub name: String,
}

/// `[DOC-03]` One sculpt layer: sparse per-vertex deltas added in rest-pose space.
#[derive(Clone, Debug, PartialEq)]
pub struct SculptLayer {
    /// Stable id.
    pub id: Id128,
    /// Display name.
    pub name: String,
    /// The region mask, or `None` for the whole body.
    pub region: Option<String>,
    /// Weight of the layer, 0..1.
    pub opacity: f64,
    /// The delta payload blob ([`Hash128::ZERO`]: an empty layer).
    pub payload: Hash128,
    /// Locked layers reject changes `[LOCK-01]`.
    pub locked: bool,
    /// Hidden layers do not deform the mesh.
    pub visible: bool,
    /// The author who created it.
    pub author: String,
    /// The command that created it, if known.
    pub created_by_command: Option<Id128>,
}

/// `[DOC-04]` The channels a skin layer writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SkinChannel {
    /// Albedo tint (mixed).
    AlbedoTint,
    /// Albedo multiply.
    AlbedoMultiply,
    /// Specular intensity.
    Specular,
    /// Roughness.
    Roughness,
    /// Normal detail.
    NormalDetail,
    /// Height / displacement.
    Height,
    /// Translucency / thickness.
    Translucency,
    /// Subsurface scattering colour.
    SssColor,
    /// Subsurface scattering radius.
    SssRadius,
    /// Freckle mask.
    FreckleMask,
    /// Mole mask.
    MoleMask,
    /// Blemish mask.
    BlemishMask,
    /// Vascular redness.
    Redness,
    /// Cavity.
    Cavity,
}

impl SkinChannel {
    /// Every channel, with its name in files.
    pub const ALL: [(Self, &'static str); 14] = [
        (Self::AlbedoTint, "albedo_tint"),
        (Self::AlbedoMultiply, "albedo_multiply"),
        (Self::Specular, "specular"),
        (Self::Roughness, "roughness"),
        (Self::NormalDetail, "normal_detail"),
        (Self::Height, "height"),
        (Self::Translucency, "translucency"),
        (Self::SssColor, "sss_color"),
        (Self::SssRadius, "sss_radius"),
        (Self::FreckleMask, "freckle_mask"),
        (Self::MoleMask, "mole_mask"),
        (Self::BlemishMask, "blemish_mask"),
        (Self::Redness, "redness"),
        (Self::Cavity, "cavity"),
    ];

    /// The name in files.
    pub fn name(self) -> &'static str {
        Self::ALL.iter().find(|(c, _)| *c == self).map_or("", |(_, n)| n)
    }
}

/// How a skin layer combines with the layers below it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SkinBlend {
    /// Linear interpolation towards the layer by its opacity.
    Mix,
    /// Multiplication.
    Multiply,
    /// Addition.
    Add,
    /// Overlay (multiply below 0.5, screen above).
    Overlay,
}

/// `[DOC-04]` Where a skin layer's content comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum SkinSource {
    /// A procedural generator (`[OTH-02]`: freckles, moles, veins ...) with its parameters and seed.
    Procedural {
        /// The generator's name.
        generator: String,
        /// Its parameters, by name.
        params: BTreeMap<String, f64>,
        /// The random seed.
        seed: u64,
    },
    /// A painted texture blob.
    Painted {
        /// The texture.
        texture: Hash128,
    },
    /// A captured appearance set (section 11.4).
    Captured {
        /// The appearance set blob.
        set: Hash128,
    },
}

/// `[DOC-04]` One skin layer.
#[derive(Clone, Debug, PartialEq)]
pub struct SkinLayer {
    /// Stable id.
    pub id: Id128,
    /// Display name.
    pub name: String,
    /// The channels it writes, sorted, no repeats.
    pub channels: Vec<SkinChannel>,
    /// Blend mode.
    pub blend: SkinBlend,
    /// Opacity, 0..1.
    pub opacity: f64,
    /// Region mask, or `None` for everywhere.
    pub region: Option<String>,
    /// Content.
    pub source: SkinSource,
    /// Locked layers reject changes.
    pub locked: bool,
    /// Hidden layers are skipped by the compositor.
    pub visible: bool,
    /// The author who created it.
    pub author: String,
}

/// `[DOC-08]` What a groom covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GroomKind {
    /// Scalp hair.
    Scalp,
    /// Eyebrows.
    Eyebrows,
    /// Eyelashes.
    Eyelashes,
    /// Beard, moustache, stubble.
    Facial,
    /// Body hair.
    Body,
}

/// The maps a groom may have `[DOC-08]`, `[HAIR-02]`.
pub const GROOM_MAPS: [&str; 7] = ["clump", "color", "curl", "density", "direction", "frizz", "length"];

/// `[DOC-08]` One groom.
#[derive(Clone, Debug, PartialEq)]
pub struct Groom {
    /// Stable id.
    pub id: Id128,
    /// Display name.
    pub name: String,
    /// What it covers.
    pub kind: GroomKind,
    /// The guide curves blob ([`Hash128::ZERO`]: no guides yet).
    pub guides: Hash128,
    /// Generation parameters by name (curl radius, clump spread ...), owned by the hair generator.
    pub params: BTreeMap<String, f64>,
    /// Maps by name (one of [`GROOM_MAPS`]) to texture blobs.
    pub maps: BTreeMap<String, Hash128>,
    /// The random seed of the generator.
    pub seed: u64,
    /// Locked grooms reject changes.
    pub locked: bool,
    /// Hidden grooms are not generated or drawn.
    pub visible: bool,
    /// The author who created it.
    pub author: String,
}

/// `[DOC-09]` A named set of expression weights.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpressionPreset {
    /// Stable id.
    pub id: Id128,
    /// Display name.
    pub name: String,
    /// Expression parameter id to weight; parameters not listed are 0.
    pub weights: BTreeMap<String, f64>,
}

/// `[DOC-09]` A named pose: joint rotations relative to the rest pose.
#[derive(Clone, Debug, PartialEq)]
pub struct PosePreset {
    /// Stable id.
    pub id: Id128,
    /// Display name.
    pub name: String,
    /// Joint name to unit quaternion `(x, y, z, w)`; joints not listed stay at rest.
    pub joints: BTreeMap<String, [f64; 4]>,
}

/// What a lock protects `[LOCK-01]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LockTarget {
    /// Shape, sculpt and skin changes inside a template region.
    Region(String),
    /// One parameter.
    Parameter(String),
}

/// `[DOC-10]` A region or parameter lock. (Layer locks are the layers' `locked` flags.)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lock {
    /// Stable id.
    pub id: Id128,
    /// What it protects.
    pub target: LockTarget,
    /// The author who set it.
    pub owner: String,
}

/// `[DOC-10]` A pinned landmark `[LOCK-03]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Pin {
    /// Stable id.
    pub id: Id128,
    /// The landmark name.
    pub landmark: String,
    /// The rest-pose position it keeps, metres.
    pub position: [f64; 3],
    /// The author who set it.
    pub owner: String,
}

/// `[DOC-11]` A named snapshot: a full document state by content address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    /// Stable id.
    pub id: Id128,
    /// Display name.
    pub name: String,
    /// The blob holding [`Document::snapshot_text`] at the time.
    pub state: Hash128,
    /// When it was taken, Unix milliseconds.
    pub created_ms: u64,
    /// The author who took it.
    pub author: String,
}

/// The file format of an attachment mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeshFormat {
    /// Wavefront OBJ (materials in [`Attachment::materials`] as MTL).
    Obj,
    /// glTF 2.0 binary.
    Glb,
}

/// How an attachment follows the body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Binding {
    /// Rigidly attached to one joint.
    Rigid,
    /// Skinned to the skeleton by its own weights.
    Skinned,
}

/// `[DOC-12]` An imported mesh attached to the skeleton (glasses, earrings). No simulation.
#[derive(Clone, Debug, PartialEq)]
pub struct Attachment {
    /// Stable id.
    pub id: Id128,
    /// Display name.
    pub name: String,
    /// The mesh file blob.
    pub mesh: Hash128,
    /// Its format.
    pub format: MeshFormat,
    /// A material file blob (MTL for OBJ), or [`Hash128::ZERO`].
    pub materials: Hash128,
    /// The joint it is attached to (the root joint of a skinned attachment).
    pub joint: String,
    /// Rigid or skinned.
    pub binding: Binding,
    /// Offset from the joint: translation (metres), rotation (unit quaternion `(x, y, z, w)`),
    /// scale.
    pub translation: [f64; 3],
    /// See `translation`.
    pub rotation: [f64; 4],
    /// See `translation`.
    pub scale: [f64; 3],
    /// Hidden attachments are not drawn.
    pub visible: bool,
    /// The author who attached it.
    pub author: String,
}

/// What a reference is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceKind {
    /// A reference photo.
    Photo,
    /// Landmark annotations of a photo.
    Landmarks,
    /// A colour-calibration target.
    ColorTarget,
    /// Capture lighting.
    Lighting,
}

/// `[DOC-13]` A reference asset, stored under the project's `refs/` directory (not embedded in
/// the character).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reference {
    /// Stable id.
    pub id: Id128,
    /// Display name.
    pub name: String,
    /// What it is.
    pub kind: ReferenceKind,
    /// The path relative to `refs/` (`/`-separated, no `..`).
    pub path: String,
    /// The content hash of the file when it was imported ([`Hash128::ZERO`]: not recorded).
    pub hash: Hash128,
}

/// The character document.
#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    /// `[DOC-01]`.
    pub meta: Metadata,
    /// Project policies.
    pub settings: Settings,
    /// `[DOC-01]` authors, sorted by id.
    pub authors: Vec<Author>,
    /// `[DOC-02]` non-default parameter values by id.
    pub params: BTreeMap<String, ParamValue>,
    /// The author of the last change of each stored parameter value.
    pub param_authors: BTreeMap<String, String>,
    /// `[DOC-09]` the current preview pose: joint name to unit quaternion `(x, y, z, w)`.
    pub pose: BTreeMap<String, [f64; 4]>,
    /// `[DOC-03]` sculpt layers, bottom first.
    pub sculpt_layers: Vec<SculptLayer>,
    /// `[DOC-04]` skin layers, bottom first.
    pub skin_layers: Vec<SkinLayer>,
    /// `[DOC-08]` grooms.
    pub grooms: Vec<Groom>,
    /// `[DOC-09]` expression presets.
    pub expression_presets: Vec<ExpressionPreset>,
    /// `[DOC-09]` pose presets.
    pub pose_presets: Vec<PosePreset>,
    /// `[DOC-10]` region and parameter locks.
    pub locks: Vec<Lock>,
    /// `[DOC-10]` landmark pins.
    pub pins: Vec<Pin>,
    /// `[DOC-11]` snapshots.
    pub snapshots: Vec<Snapshot>,
    /// `[DOC-12]` attachments.
    pub attachments: Vec<Attachment>,
    /// `[DOC-13]` references.
    pub references: Vec<Reference>,
}

impl Document {
    /// A new, empty document made by `author` at `now_ms` for the given template.
    pub fn new(id: Id128, name: &str, template: &str, template_version: u32, now_ms: u64, author: Author) -> Self {
        Self {
            meta: Metadata {
                id,
                name: name.to_owned(),
                template: template.to_owned(),
                template_version,
                created_ms: now_ms,
                modified_ms: now_ms,
            },
            settings: Settings::default(),
            authors: vec![author],
            params: BTreeMap::new(),
            param_authors: BTreeMap::new(),
            pose: BTreeMap::new(),
            sculpt_layers: Vec::new(),
            skin_layers: Vec::new(),
            grooms: Vec::new(),
            expression_presets: Vec::new(),
            pose_presets: Vec::new(),
            locks: Vec::new(),
            pins: Vec::new(),
            snapshots: Vec::new(),
            attachments: Vec::new(),
            references: Vec::new(),
        }
    }

    /// The author with this id.
    pub fn author(&self, id: &str) -> Option<&Author> {
        self.authors.iter().find(|a| a.id == id)
    }

    /// Adds an author (kept sorted by id), or replaces the one with the same id.
    pub fn add_author(&mut self, author: Author) {
        match self.authors.binary_search_by(|a| a.id.cmp(&author.id)) {
            Ok(i) => self.authors[i] = author,
            Err(i) => self.authors.insert(i, author),
        }
    }

    /// The value of a parameter: the stored one or the registry default.
    pub fn param<'a>(&'a self, registry: &'a Registry, id: &str) -> Result<&'a ParamValue> {
        let param = registry.require(id)?;
        Ok(self.params.get(id).unwrap_or(&param.default))
    }

    /// Sets a parameter by `author` and returns the value stored before (`None`: it was at the
    /// default). The value must be valid for the parameter ([`crate::registry::Param::check_value`]);
    /// read-only parameters are refused `[REG-05]`; the default value removes the entry
    /// `[DOC-02]`. Locks and protection are the command system's business, not checked here.
    pub fn set_param(
        &mut self,
        registry: &Registry,
        id: &str,
        value: ParamValue,
        author: &str,
    ) -> Result<Option<ParamValue>> {
        let param = registry.require(id)?;
        if param.read_only {
            return Err(DocError::param_value(id, "read-only measurement: set it through the landmark solver"));
        }
        param.check_value(&value)?;
        if self.author(author).is_none() {
            return Err(DocError::field("param_authors", id, format!("unknown author {author}")));
        }
        if param.is_default(&value) {
            self.param_authors.remove(id);
            Ok(self.params.remove(id))
        } else {
            self.param_authors.insert(id.to_owned(), author.to_owned());
            Ok(self.params.insert(id.to_owned(), value))
        }
    }

    /// Every blob the document references (layer payloads, textures, guides, maps, meshes,
    /// snapshot states, asset parameters), without [`Hash128::ZERO`] `[DOC-22]`.
    pub fn blob_refs(&self) -> BTreeSet<Hash128> {
        let mut out = BTreeSet::new();
        for value in self.params.values() {
            if let ParamValue::AssetRef(h) = value {
                out.insert(*h);
            }
        }
        out.extend(self.sculpt_layers.iter().map(|l| l.payload));
        for layer in &self.skin_layers {
            match &layer.source {
                SkinSource::Painted { texture } => {
                    out.insert(*texture);
                }
                SkinSource::Captured { set } => {
                    out.insert(*set);
                }
                SkinSource::Procedural { .. } => {}
            }
        }
        for groom in &self.grooms {
            out.insert(groom.guides);
            out.extend(groom.maps.values().copied());
        }
        out.extend(self.snapshots.iter().map(|s| s.state));
        for a in &self.attachments {
            out.insert(a.mesh);
            out.insert(a.materials);
        }
        out.remove(&Hash128::ZERO);
        out
    }

    /// Checks the whole document against the registry: names, ranges, author references,
    /// unique ids per list, stored parameters valid, not default and not read-only.
    pub fn check(&self, registry: &Registry) -> Result<()> {
        codec::check(self, registry)
    }

    /// The text form `[DOC-21]`.
    pub fn to_text(&self) -> Result<TextFile> {
        codec::to_text(self)
    }

    /// Reads the text form of the current version, checking everything ([`Document::check`]).
    /// Default parameter values in the text are dropped. Use [`crate::migrate::migrate`] first for
    /// older versions.
    pub fn from_text(file: &TextFile, registry: &Registry) -> Result<Self> {
        codec::from_text(file, registry)
    }

    /// The canonical text of the document.
    pub fn write(&self) -> Result<String> {
        text::write(&self.to_text()?)
    }

    /// Parses document text of any supported version: syntax, migration to the current version,
    /// typed reading and checks.
    pub fn parse(bytes: &[u8], registry: &Registry, limits: &TextLimits) -> Result<Self> {
        let file = crate::migrate::migrate(text::parse(bytes, limits)?)?;
        Self::from_text(&file, registry)
    }

    /// The text a snapshot stores: the canonical text without snapshot sections, so that a
    /// snapshot never contains other snapshots `[DOC-11]`.
    pub fn snapshot_text(&self) -> Result<String> {
        let mut file = self.to_text()?;
        for kind in SNAPSHOT_EXCLUDED_KINDS {
            file.remove_kind(kind);
        }
        text::write(&file)
    }

    /// The edits that turn this document into `other` (see [`crate::text::diff`]).
    pub fn diff(&self, other: &Self) -> Result<Vec<Edit>> {
        Ok(text::diff(&self.to_text()?, &other.to_text()?))
    }

    /// Applies edits atomically: the result must be a valid document in canonical form (so that
    /// the inverse edits restore this document exactly), otherwise nothing changes. Conflicting
    /// edits (a value is not what the edit expects) fail with [`DocError::Conflict`].
    pub fn apply_edits(&mut self, edits: &[Edit], registry: &Registry) -> Result<()> {
        let mut file = self.to_text()?;
        text::apply(&mut file, edits)?;
        let next = Self::from_text(&file, registry)?;
        if next.to_text()? != file {
            return Err(DocError::field(
                "",
                "",
                "the edits leave a non-canonical document (a default value, a misplaced section or an orphan entry)",
            ));
        }
        *self = next;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests;
