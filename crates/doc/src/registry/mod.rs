//! The parameter registry: every user- and agent-controllable value, as data `[REG-01]`..`[REG-05]`.
//!
//! The registry is read from JSON files under `data/template/params/` ([`Registry::load_dir`],
//! [`Registry::from_files`]); nothing about individual parameters is written in Rust `[REG-02]`.
//! The file format is described in `docs/format-anthropos.md` (section "Parameter registry files").
//! Each [`Param`] has the fields `[REG-01]` lists; groups follow `[REG-03]`; [`Registry::describe`]
//! returns the JSON agents read and [`Registry::search`] finds parameters by name and tag `[REG-04]`;
//! derived values are read-only measurements `[REG-05]`.
//!
//! Ids are stable: an id is never renamed or reused. The registry is sorted by id, so lookups are a
//! binary search and every listing comes out in the same order.

mod coverage;
mod describe;
mod parse;

pub use coverage::Coverage;
pub use parse::{RegistryLimits, parse_file};

use humanart_core::hash::Hash128;
use humanart_core::platform::FileSystem;

use crate::error::{DocError, Result};

/// The version of the registry file format this build reads and [`Registry::describe`] writes.
pub const FORMAT_VERSION: u32 = 1;

/// The value of the `format` member of registry files.
pub const FORMAT_NAME: &str = "humanart-params";

/// The value of the `format` member of [`Registry::describe`]'s output.
pub const DESCRIBE_FORMAT_NAME: &str = "humanart-params-description";

/// The longest parameter id, in bytes.
pub const MAX_ID_LEN: usize = 96;

/// The most components of a vector parameter.
pub const MAX_DIMENSION: u8 = 16;

/// The most points of a curve parameter.
pub const MAX_CURVE_POINTS: usize = 64;

/// The top-level groups of `[REG-03]`, in display order. A parameter's group path starts with one.
pub const GROUPS: [&str; 11] = [
    "identity",
    "face",
    "body",
    "expression",
    "skin",
    "eyes",
    "mouth",
    "hair",
    "brows_lashes",
    "facial_hair",
    "lighting_camera",
];

/// The group whose parameters are editor state, not part of the character `[REG-03]`.
pub const EDITOR_GROUP: &str = "lighting_camera";

/// `true` when `id` is a valid parameter id: dot-separated segments of `[a-z0-9_]`, each starting
/// with a letter or digit, at most [`MAX_ID_LEN`] bytes.
pub fn is_param_id(id: &str) -> bool {
    id.len() <= MAX_ID_LEN
        && id.split('.').all(|seg| {
            let b = seg.as_bytes();
            !b.is_empty()
                && b[0].is_ascii_alphanumeric()
                && b.iter().all(|&c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_')
        })
}

/// What kind of value a parameter holds `[REG-01]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParamKind {
    /// One number.
    Scalar,
    /// A fixed number of numbers, each within the parameter's range.
    Vector,
    /// A linear Rec.709 RGB colour, each channel within the range.
    Color,
    /// One of a list of named choices.
    Enum,
    /// On or off.
    Bool,
    /// A piecewise-linear curve: points `[x, y]` with `x` strictly increasing in `[0, 1]` and `y`
    /// within the range.
    Curve,
    /// A reference to a content-addressed blob (a texture, a map), or none.
    AssetRef,
}

impl ParamKind {
    /// Every kind, in the order of the documentation.
    pub const ALL: [Self; 7] =
        [Self::Scalar, Self::Vector, Self::Color, Self::Enum, Self::Bool, Self::Curve, Self::AssetRef];

    /// The name used in files.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Vector => "vector",
            Self::Color => "color",
            Self::Enum => "enum",
            Self::Bool => "bool",
            Self::Curve => "curve",
            Self::AssetRef => "asset_ref",
        }
    }

    /// `true` for the kinds that have a numeric range.
    pub const fn is_numeric(self) -> bool {
        matches!(self, Self::Scalar | Self::Vector | Self::Color | Self::Curve)
    }
}

/// The numeric range of a parameter: hard limits, the plausible ("soft") span and the UI step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range {
    /// The smallest allowed value.
    pub min: f64,
    /// The largest allowed value.
    pub max: f64,
    /// The start of the usual span (sliders open on it; values outside are allowed).
    pub soft_min: f64,
    /// The end of the usual span.
    pub soft_max: f64,
    /// The increment of keyboard and wheel steps.
    pub step: f64,
}

impl Range {
    /// `true` when `v` lies within the hard limits.
    pub fn contains(&self, v: f64) -> bool {
        v >= self.min && v <= self.max
    }

    /// `v` clamped to the hard limits (NaN becomes `min`).
    pub fn clamp(&self, v: f64) -> f64 {
        if v.is_nan() { self.min } else { v.clamp(self.min, self.max) }
    }
}

/// The side of the body a parameter acts on. Left is the character's left (+X).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// The character's left, +X.
    Left,
    /// The character's right, -X.
    Right,
}

impl Side {
    /// The name used in files.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
        }
    }

    /// The template's spelling in region and component names (`brow-L`, `AU12-L`).
    pub const fn letter(self) -> &'static str {
        match self {
            Self::Left => "L",
            Self::Right => "R",
        }
    }

    /// The other side.
    #[must_use]
    pub const fn mirror(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

/// How a parameter relates to its mirror image `[REG-01]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Symmetry {
    /// The parameter acts on both sides or on the midline.
    None,
    /// The parameter acts on one side and has a partner acting on the other; mirrored editing
    /// moves both.
    Mirror {
        /// The partner's id.
        partner: String,
        /// The side this parameter acts on.
        side: Side,
    },
    /// The parameter acts on one side and has no partner.
    Sided {
        /// The side.
        side: Side,
    },
}

impl Symmetry {
    /// The side, when the parameter has one.
    pub const fn side(&self) -> Option<Side> {
        match self {
            Self::None => None,
            Self::Mirror { side, .. } | Self::Sided { side } => Some(*side),
        }
    }
}

/// What a parameter changes `[REG-01]` (`affects`): the renderer and the evaluation snapshot use it
/// to decide what to recompute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Channel {
    /// Rest shape (identity, semantic and asymmetry deltas).
    Shape,
    /// Expression deltas and wrinkle drivers.
    Expression,
    /// Skeleton pose and rig transforms.
    Pose,
    /// Skin appearance (pigment, micro-detail, layers).
    Skin,
    /// Eye appearance and geometry.
    Eyes,
    /// Mouth interior appearance.
    Mouth,
    /// Hair, brows, lashes, facial and body hair.
    Hair,
    /// Editor state that does not change the character (camera, preview lighting, gaze preview).
    View,
}

impl Channel {
    /// Every channel.
    pub const ALL: [Self; 8] =
        [Self::Shape, Self::Expression, Self::Pose, Self::Skin, Self::Eyes, Self::Mouth, Self::Hair, Self::View];

    /// The name used in files.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Shape => "shape",
            Self::Expression => "expression",
            Self::Pose => "pose",
            Self::Skin => "skin",
            Self::Eyes => "eyes",
            Self::Mouth => "mouth",
            Self::Hair => "hair",
            Self::View => "view",
        }
    }
}

/// The axis of a measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Axis {
    /// Left-right.
    X,
    /// Up-down.
    Y,
    /// Front-back.
    Z,
}

impl Axis {
    /// The name used in files.
    pub const fn name(self) -> &'static str {
        match self {
            Self::X => "x",
            Self::Y => "y",
            Self::Z => "z",
        }
    }
}

/// How a parameter reaches the model or the shaders `[REG-01]` (`evaluation`). Targets are names in
/// the template package or in the material parameter blocks, resolved to indices when the registry
/// is bound to a template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Evaluation {
    /// The weight of a delta component of the template (identity global or regional, semantic,
    /// asymmetry): `P_rest += value * delta` `[ID-01]`.
    Component(String),
    /// The weight of an expression target of the template `[EXP-01]`.
    Expression(String),
    /// A rig transform (eye rotation, eyeball scale) applied through the skeleton.
    Rig(String),
    /// A slot of a material parameter block read by the shaders `[SKN-10]`.
    Material(String),
    /// A parameter of the hair groom generator `[HAIR-02]`.
    Groom(String),
    /// Editor state (preview camera and lighting, gaze preview); not exported with the character.
    Editor(String),
    /// A read-only measurement `[REG-05]`: the distance between two landmarks of the deformed rest
    /// shape, along one axis or straight. Setting it goes through the landmark solver `[LM-01]`.
    Measurement {
        /// The two landmark names.
        landmarks: [String; 2],
        /// The axis, or `None` for the straight distance.
        axis: Option<Axis>,
    },
}

impl Evaluation {
    /// The name used in files.
    pub const fn kind_name(&self) -> &'static str {
        match self {
            Self::Component(_) => "component",
            Self::Expression(_) => "expression",
            Self::Rig(_) => "rig",
            Self::Material(_) => "material",
            Self::Groom(_) => "groom",
            Self::Editor(_) => "editor",
            Self::Measurement { .. } => "measurement",
        }
    }

    /// The target name (`None` for a measurement).
    pub fn target(&self) -> Option<&str> {
        match self {
            Self::Component(t)
            | Self::Expression(t)
            | Self::Rig(t)
            | Self::Material(t)
            | Self::Groom(t)
            | Self::Editor(t) => Some(t),
            Self::Measurement { .. } => None,
        }
    }
}

/// A parameter value. Which variant a parameter takes is its [`ParamKind`].
#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    /// A [`ParamKind::Scalar`] value.
    Scalar(f64),
    /// A [`ParamKind::Vector`] value.
    Vector(Vec<f64>),
    /// A [`ParamKind::Color`] value, linear Rec.709.
    Color([f64; 3]),
    /// A [`ParamKind::Enum`] value: one of the choices.
    Enum(String),
    /// A [`ParamKind::Bool`] value.
    Bool(bool),
    /// A [`ParamKind::Curve`] value: `[x, y]` points.
    Curve(Vec<[f64; 2]>),
    /// A [`ParamKind::AssetRef`] value: a blob hash, [`Hash128::ZERO`] for none.
    AssetRef(Hash128),
}

impl ParamValue {
    /// The kind this value belongs to.
    pub const fn kind(&self) -> ParamKind {
        match self {
            Self::Scalar(_) => ParamKind::Scalar,
            Self::Vector(_) => ParamKind::Vector,
            Self::Color(_) => ParamKind::Color,
            Self::Enum(_) => ParamKind::Enum,
            Self::Bool(_) => ParamKind::Bool,
            Self::Curve(_) => ParamKind::Curve,
            Self::AssetRef(_) => ParamKind::AssetRef,
        }
    }

    /// The scalar, when this is one.
    pub const fn as_scalar(&self) -> Option<f64> {
        match self {
            Self::Scalar(v) => Some(*v),
            _ => None,
        }
    }
}

/// One registry entry `[REG-01]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Param {
    /// Stable id, never reused or renamed.
    pub id: String,
    /// The name the UI shows.
    pub display_name: String,
    /// What the parameter does, written so that an agent understands the visual effect.
    pub description: String,
    /// The group path (`face/nose`); the first segment is one of [`GROUPS`].
    pub group: String,
    /// The template region it acts on, if one.
    pub region: Option<String>,
    /// The kind of value.
    pub kind: ParamKind,
    /// The number of components of a vector (3 for a colour, 1 otherwise).
    pub dimension: u8,
    /// The choices of an enum (empty otherwise).
    pub choices: Vec<String>,
    /// The unit, as a short word (`coefficient`, `weight`, `fraction`, `m`, `deg`, `linear_rgb`);
    /// empty for kinds without one.
    pub unit: String,
    /// The numeric range, for scalar, vector, colour and curve parameters.
    pub range: Option<Range>,
    /// The default value. Documents store only values that differ from it `[DOC-02]`.
    pub default: ParamValue,
    /// Mirror relation.
    pub symmetry: Symmetry,
    /// What it changes, sorted, without repeats.
    pub affects: Vec<Channel>,
    /// How it is evaluated.
    pub evaluation: Evaluation,
    /// `true` for derived values: shown as measurements, set only through the solver `[REG-05]`.
    pub read_only: bool,
    /// `true` when new documents start with the parameter locked.
    pub locked_by_default: bool,
    /// The weight of the parameter in the plausibility prior `[ID-04]` (0: not part of it).
    pub plausibility_weight: f64,
    /// The registry version that introduced the parameter.
    pub since_version: u32,
    /// Search tags `[REG-04]`, sorted, without repeats.
    pub tags: Vec<String>,
    /// Why the range or default is provisional (not yet taken from the specification or from
    /// data), or `None` when they are settled.
    pub provisional: Option<String>,
}

impl Param {
    /// `true` for editor state that is not part of the character (`[REG-03]` Lighting/Camera, and
    /// previews such as the gaze direction).
    pub fn editor_only(&self) -> bool {
        self.group.split('/').next() == Some(EDITOR_GROUP) || matches!(self.evaluation, Evaluation::Editor(_))
    }

    /// The side, for one-sided parameters.
    pub const fn side(&self) -> Option<Side> {
        self.symmetry.side()
    }

    /// Checks that `value` is a valid value of this parameter: right kind and dimension, finite,
    /// inside the hard range, one of the choices, curve points ordered.
    pub fn check_value(&self, value: &ParamValue) -> Result<()> {
        let fail = |what: &str| Err(DocError::param_value(&self.id, what));
        if value.kind() != self.kind {
            return fail(&format!("expected a {} value, found a {}", self.kind.name(), value.kind().name()));
        }
        let in_range = |v: f64| self.range.is_none_or(|r| r.contains(v)) && v.is_finite();
        match value {
            ParamValue::Scalar(v) => {
                if !in_range(*v) {
                    return fail("value outside the parameter's range");
                }
            }
            ParamValue::Vector(vs) => {
                if vs.len() != usize::from(self.dimension) {
                    return fail("wrong number of components");
                }
                if !vs.iter().all(|v| in_range(*v)) {
                    return fail("component outside the parameter's range");
                }
            }
            ParamValue::Color(c) => {
                if !c.iter().all(|v| in_range(*v)) {
                    return fail("colour channel outside the parameter's range");
                }
            }
            ParamValue::Enum(choice) => {
                if !self.choices.iter().any(|c| c == choice) {
                    return fail("not one of the parameter's choices");
                }
            }
            ParamValue::Curve(points) => {
                if points.len() < 2 || points.len() > MAX_CURVE_POINTS {
                    return fail("a curve needs 2 to 64 points");
                }
                if !points.iter().all(|[x, y]| x.is_finite() && (0.0..=1.0).contains(x) && in_range(*y)) {
                    return fail("curve point outside [0, 1] x range");
                }
                if !points.windows(2).all(|w| w[0][0] < w[1][0]) {
                    return fail("curve points must have strictly increasing x");
                }
            }
            ParamValue::Bool(_) | ParamValue::AssetRef(_) => {}
        }
        Ok(())
    }

    /// `value` brought into the parameter's range: numbers clamped to the hard limits (NaN to the
    /// minimum), a vector padded or cut to the dimension. Values of the wrong kind, unknown choices
    /// and invalid curves give the default. The result always passes [`Param::check_value`].
    pub fn clamp_value(&self, value: &ParamValue) -> ParamValue {
        let clamp = |v: f64| self.range.map_or(if v.is_finite() { v } else { 0.0 }, |r| r.clamp(v));
        let clamped = match value {
            ParamValue::Scalar(v) => ParamValue::Scalar(clamp(*v)),
            ParamValue::Vector(vs) => {
                let mut out: Vec<f64> = vs.iter().map(|v| clamp(*v)).collect();
                out.resize(usize::from(self.dimension), self.range.map_or(0.0, |r| r.min.max(0.0).min(r.max)));
                ParamValue::Vector(out)
            }
            ParamValue::Color(c) => ParamValue::Color(c.map(clamp)),
            ParamValue::Curve(points) => ParamValue::Curve(points.iter().map(|[x, y]| [*x, clamp(*y)]).collect()),
            other => other.clone(),
        };
        if self.check_value(&clamped).is_ok() { clamped } else { self.default.clone() }
    }

    /// `true` when `value` equals the default (such values are not stored in documents).
    pub fn is_default(&self, value: &ParamValue) -> bool {
        *value == self.default
    }
}

/// The parameter registry: all entries, sorted by id.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Registry {
    params: Vec<Param>,
}

impl Registry {
    /// A registry from already-checked entries. Sorts them, and checks ids, ranges, defaults and
    /// mirror partners as [`Registry::from_files`] does.
    pub fn new(mut params: Vec<Param>) -> Result<Self> {
        params.sort_by(|a, b| a.id.cmp(&b.id));
        let registry = Self { params };
        parse::check_registry(&registry)?;
        Ok(registry)
    }

    /// Parses registry files (name, bytes) into one registry `[REG-02]`. Names are used in error
    /// messages only. Fails on the first invalid file or entry, on duplicate ids across files and on
    /// broken mirror pairs.
    pub fn from_files(files: &[(&str, &[u8])], limits: &RegistryLimits) -> Result<Self> {
        if files.len() as u64 > limits.max_files {
            return Err(DocError::Limit { what: "registry files", limit: limits.max_files });
        }
        let mut params = Vec::new();
        for (name, bytes) in files {
            params.extend(parse_file(name, bytes, limits)?);
            if params.len() as u64 > limits.max_params {
                return Err(DocError::Limit { what: "registry parameters", limit: limits.max_params });
            }
        }
        Self::new(params)
    }

    /// Reads every `*.json` file of directory `dir`, in name order, through the platform file system
    /// `[REG-02]`.
    pub fn load_dir(fs: &dyn FileSystem, dir: &str, limits: &RegistryLimits) -> Result<Self> {
        let entries = fs.list(dir).map_err(|e| DocError::platform(dir, e))?;
        let mut files = Vec::new();
        for entry in entries {
            let is_json =
                humanart_core::platform::extension(&entry.name).is_some_and(|e| e.eq_ignore_ascii_case("json"));
            if !is_json || entry.kind != humanart_core::platform::FileKind::File {
                continue;
            }
            if entry.len > limits.max_file_bytes {
                return Err(DocError::Limit { what: "bytes in one registry file", limit: limits.max_file_bytes });
            }
            let path = humanart_core::platform::join_path(dir, &entry.name);
            let bytes = fs.read(&path).map_err(|e| DocError::platform(&path, e))?;
            files.push((entry.name, bytes));
        }
        let borrowed: Vec<(&str, &[u8])> = files.iter().map(|(n, b)| (n.as_str(), b.as_slice())).collect();
        Self::from_files(&borrowed, limits)
    }

    /// All entries, sorted by id.
    pub fn params(&self) -> &[Param] {
        &self.params
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.params.len()
    }

    /// `true` when the registry has no entries.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// The position of `id` in [`Registry::params`].
    pub fn index_of(&self, id: &str) -> Option<usize> {
        self.params.binary_search_by(|p| p.id.as_str().cmp(id)).ok()
    }

    /// The entry with this id.
    pub fn get(&self, id: &str) -> Option<&Param> {
        self.index_of(id).map(|i| &self.params[i])
    }

    /// The entry with this id, or [`DocError::UnknownParameter`].
    pub fn require(&self, id: &str) -> Result<&Param> {
        self.get(id).ok_or_else(|| DocError::UnknownParameter(id.to_owned()))
    }

    /// The entries whose group is `group` or lies under it (`face` finds `face/nose`).
    pub fn in_group<'a>(&'a self, group: &'a str) -> impl Iterator<Item = &'a Param> + 'a {
        self.params.iter().filter(move |p| {
            p.group == group || (p.group.starts_with(group) && p.group.as_bytes().get(group.len()) == Some(&b'/'))
        })
    }

    /// The entries acting on `region`.
    pub fn in_region<'a>(&'a self, region: &'a str) -> impl Iterator<Item = &'a Param> + 'a {
        self.params.iter().filter(move |p| p.region.as_deref() == Some(region))
    }

    /// The entries carrying `tag`.
    pub fn with_tag<'a>(&'a self, tag: &'a str) -> impl Iterator<Item = &'a Param> + 'a {
        self.params.iter().filter(move |p| p.tags.iter().any(|t| t == tag))
    }

    /// Searches by name and semantic tags `[REG-04]`.
    ///
    /// The query is split at white space; every word must occur (case-insensitively) in the id,
    /// display name, tags, group, region or description. Results are ranked: an exact id first, then
    /// matches in id and display name, then tags, group and region, then description only; ties in
    /// id order.
    pub fn search(&self, query: &str) -> Vec<&Param> {
        let words: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
        if words.is_empty() {
            return Vec::new();
        }
        let mut hits: Vec<(u32, &Param)> = Vec::new();
        for param in &self.params {
            let names = format!("{} {}", param.id, param.display_name.to_lowercase());
            let labels = format!("{} {} {}", param.tags.join(" "), param.group, param.region.as_deref().unwrap_or(""))
                .to_lowercase();
            let mut score = 0;
            let mut all = true;
            for word in &words {
                score += if names.contains(word.as_str()) {
                    4
                } else if labels.contains(word.as_str()) {
                    2
                } else if param.description.to_lowercase().contains(word.as_str()) {
                    1
                } else {
                    all = false;
                    break;
                };
            }
            if all {
                if words.len() == 1 && param.id == words[0] {
                    score += 1000;
                }
                hits.push((score, param));
            }
        }
        hits.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.id.cmp(&b.1.id)));
        hits.into_iter().map(|(_, p)| p).collect()
    }

    /// The JSON description of every entry, for agents `[REG-04]`, `[AC-WP07-2]`. See
    /// [`describe::describe_param`] for the shape of one entry.
    pub fn describe(&self) -> humanart_io::json::Value {
        describe::describe_registry(self)
    }

    /// The JSON description of every entry, as text (pretty, deterministic). This is what the agent
    /// tool `describe_parameters` returns.
    pub fn describe_parameters(&self) -> Result<String> {
        let mut out = String::new();
        humanart_io::json::write(&self.describe(), humanart_io::json::Style::PRETTY_INLINE, &mut out)?;
        out.push('\n');
        Ok(out)
    }

    /// Checks the registry against a template `[REG-02]`: every target names something the template
    /// has, and every identity, semantic and asymmetry component and every expression of the
    /// template has an entry.
    pub fn check_template(&self, template: &humanart_model::Template) -> Coverage {
        coverage::check(self, template)
    }
}

pub use describe::{decode_value, describe_param, param_from_description, value_from_json, value_to_json};

#[cfg(test)]
pub(crate) mod tests;
