//! Reading and checking registry files `[REG-01]`, `[REG-02]`, `[ROB-01]`.

use humanart_io::json::{self, AccessError, JsonLimits, Node, Object, Value};

use super::{
    Axis, Channel, Evaluation, FORMAT_NAME, FORMAT_VERSION, GROUPS, Param, ParamKind, Range, Registry, Side, Symmetry,
    describe::decode_value, is_param_id,
};
use crate::error::{DocError, Result};

/// The size limits of reading registry files `[ROB-02]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegistryLimits {
    /// Most files in one registry.
    pub max_files: u64,
    /// Largest file, in bytes.
    pub max_file_bytes: u64,
    /// Most parameters in the whole registry.
    pub max_params: u64,
}

impl RegistryLimits {
    /// 1024 files of at most 8 MiB, 65 536 parameters.
    pub const DEFAULT: Self = Self { max_files: 1024, max_file_bytes: 8 << 20, max_params: 1 << 16 };
}

impl Default for RegistryLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The members a registry entry may have.
const ENTRY_KEYS: [&str; 26] = [
    "id",
    "display_name",
    "description",
    "group",
    "region",
    "kind",
    "dimension",
    "choices",
    "unit",
    "min",
    "max",
    "soft_min",
    "soft_max",
    "step",
    "default",
    "symmetry",
    "affects",
    "evaluation",
    "read_only",
    "locked_by_default",
    "plausibility_weight",
    "since_version",
    "tags",
    "provisional",
    "editor_only",
    "sides",
];

/// The placeholder that `sides` entries replace by `L` or `R` in region, target and landmark names.
const SIDE_PLACEHOLDER: &str = "{side}";

/// Parses one registry file: `{"format": "humanart-params", "version": 1, "description": ...,
/// "defaults": {...}, "parameters": [...]}`. Members of `defaults` apply to every entry that does
/// not set them; an entry with `"sides": true` becomes two entries `<id>.l` and `<id>.r`. Every
/// entry is checked on its own; cross-entry checks (unique ids, mirror partners) are
/// [`Registry::new`]'s.
pub fn parse_file(name: &str, bytes: &[u8], limits: &RegistryLimits) -> Result<Vec<Param>> {
    if bytes.len() as u64 > limits.max_file_bytes {
        return Err(DocError::Limit { what: "bytes in one registry file", limit: limits.max_file_bytes });
    }
    let json_limits = JsonLimits {
        max_depth: 8,
        max_input_bytes: limits.max_file_bytes,
        max_string_bytes: 1 << 16,
        max_members: limits.max_params.max(64),
        max_values: limits.max_file_bytes,
    };
    let value =
        json::parse_with(bytes, &json_limits).map_err(|source| DocError::Json { file: name.to_owned(), source })?;
    let access = |source: AccessError| DocError::Access { file: name.to_owned(), source };
    let root = Node::root(&value);
    root.deny_unknown(&["format", "version", "description", "defaults", "parameters"]).map_err(access)?;
    let format: &str = root.get("format").map_err(access)?;
    if format != FORMAT_NAME {
        return Err(DocError::registry(name, "the file's `format` is not \"humanart-params\""));
    }
    let version: u64 = root.get("version").map_err(access)?;
    if version != u64::from(FORMAT_VERSION) {
        return Err(DocError::Version {
            format: "registry file",
            found: version,
            supported: u64::from(FORMAT_VERSION),
        });
    }
    let _: Option<&str> = root.get_opt("description").map_err(access)?;
    let empty = Object::new();
    let defaults = match root.opt_field("defaults").map_err(access)? {
        Some(node) => {
            node.value().as_object().ok_or_else(|| DocError::registry(name, "`defaults` must be an object"))?
        }
        None => &empty,
    };
    if defaults.contains_key("id") || defaults.contains_key("sides") {
        return Err(DocError::registry(name, "`defaults` may not set `id` or `sides`"));
    }
    let entries = root.field("parameters").map_err(access)?;
    let mut out = Vec::new();
    for entry in entries.elements().map_err(access)? {
        let object = entry.value().as_object().ok_or_else(|| DocError::registry(name, "an entry is not an object"))?;
        let mut merged = defaults.clone();
        for (key, value) in object {
            merged.insert(key, value.clone());
        }
        let merged = Value::Object(merged);
        let params = parse_entry(&Node::root(&merged), name)?;
        for param in &params {
            check_param(param)?;
        }
        out.extend(params);
        if out.len() as u64 > limits.max_params {
            return Err(DocError::Limit { what: "registry parameters", limit: limits.max_params });
        }
    }
    Ok(out)
}

fn choice<T: Copy>(node: &Node<'_, '_>, key: &str, choices: &[(&str, T)]) -> std::result::Result<T, AccessError> {
    node.field(key)?.choice(choices)
}

const KINDS: [(&str, ParamKind); 7] = [
    ("scalar", ParamKind::Scalar),
    ("vector", ParamKind::Vector),
    ("color", ParamKind::Color),
    ("enum", ParamKind::Enum),
    ("bool", ParamKind::Bool),
    ("curve", ParamKind::Curve),
    ("asset_ref", ParamKind::AssetRef),
];

const SIDES: [(&str, Side); 2] = [("left", Side::Left), ("right", Side::Right)];

const CHANNELS: [(&str, Channel); 8] = [
    ("shape", Channel::Shape),
    ("expression", Channel::Expression),
    ("pose", Channel::Pose),
    ("skin", Channel::Skin),
    ("eyes", Channel::Eyes),
    ("mouth", Channel::Mouth),
    ("hair", Channel::Hair),
    ("view", Channel::View),
];

/// Parses one entry (after `defaults` are merged in) into one parameter, or two with `sides`.
pub(super) fn parse_entry(node: &Node<'_, '_>, file: &str) -> Result<Vec<Param>> {
    let access = |source: AccessError| DocError::Access { file: file.to_owned(), source };
    node.deny_unknown(&ENTRY_KEYS).map_err(access)?;
    let id: String = node.get("id").map_err(access)?;
    let bad = |what: &str| DocError::registry(id.clone(), what);
    let display_name: String = node.get("display_name").map_err(access)?;
    let description: String = node.get("description").map_err(access)?;
    let group: String = node.get("group").map_err(access)?;
    let region: Option<String> = node.get_opt("region").map_err(access)?;
    let kind = choice(node, "kind", &KINDS).map_err(access)?;
    let dimension = match (kind, node.get_opt::<u64>("dimension").map_err(access)?) {
        (ParamKind::Vector, Some(d)) if (2..=u64::from(super::MAX_DIMENSION)).contains(&d) => d as u8,
        (ParamKind::Vector, _) => return Err(bad("a vector needs a `dimension` from 2 to 16")),
        (ParamKind::Color, None | Some(3)) => 3,
        (ParamKind::Color, Some(_)) => return Err(bad("a colour has dimension 3")),
        (_, None | Some(1)) => 1,
        (_, Some(_)) => return Err(bad("only vectors and colours have a dimension")),
    };
    let choices: Vec<String> = node.get_opt("choices").map_err(access)?.unwrap_or_default();
    if kind != ParamKind::Enum && !choices.is_empty() {
        return Err(bad("only enums have choices"));
    }
    let unit: String = node.get_or("unit", String::new()).map_err(access)?;
    let numeric_keys = ["min", "max", "soft_min", "soft_max", "step"];
    let range = if kind.is_numeric() {
        let min: f64 = node.get("min").map_err(access)?;
        let max: f64 = node.get("max").map_err(access)?;
        Some(Range {
            min,
            max,
            soft_min: node.get_or("soft_min", min).map_err(access)?,
            soft_max: node.get_or("soft_max", max).map_err(access)?,
            step: node.get_or("step", (max - min) / 100.0).map_err(access)?,
        })
    } else {
        if numeric_keys.iter().any(|k| node.opt_field(k).is_ok_and(|f| f.is_some())) {
            return Err(bad("only numeric kinds have min, max, soft_min, soft_max and step"));
        }
        None
    };
    let default_node = node.field("default").map_err(access)?;
    let default =
        decode_value(kind, dimension, default_node.value()).ok_or_else(|| bad("`default` does not fit the kind"))?;
    let symmetry = match node.opt_field("symmetry").map_err(access)? {
        None => Symmetry::None,
        Some(s) => {
            s.deny_unknown(&["kind", "partner", "side"]).map_err(access)?;
            match s.get::<&str>("kind").map_err(access)? {
                "none" => {
                    if s.get_opt::<&str>("partner").map_err(access)?.is_some()
                        || s.get_opt::<&str>("side").map_err(access)?.is_some()
                    {
                        return Err(bad("symmetry `none` has no partner or side"));
                    }
                    Symmetry::None
                }
                "mirror" => Symmetry::Mirror {
                    partner: s.get("partner").map_err(access)?,
                    side: choice(&s, "side", &SIDES).map_err(access)?,
                },
                "sided" => {
                    if s.get_opt::<&str>("partner").map_err(access)?.is_some() {
                        return Err(bad("symmetry `sided` has no partner"));
                    }
                    Symmetry::Sided { side: choice(&s, "side", &SIDES).map_err(access)? }
                }
                _ => return Err(bad("symmetry kind must be none, mirror or sided")),
            }
        }
    };
    let mut affects = Vec::new();
    for element in node.field("affects").map_err(access)?.elements().map_err(access)? {
        affects.push(element.choice(&CHANNELS).map_err(access)?);
    }
    affects.sort_unstable();
    affects.dedup();
    let evaluation = {
        let e = node.field("evaluation").map_err(access)?;
        e.deny_unknown(&["kind", "target", "landmarks", "axis"]).map_err(access)?;
        let kind: &str = e.get("kind").map_err(access)?;
        if kind == "measurement" {
            if e.get_opt::<&str>("target").map_err(access)?.is_some() {
                return Err(bad("a measurement has landmarks, not a target"));
            }
            let axis = match e.opt_field("axis").map_err(access)? {
                Some(a) => Some(a.choice(&[("x", Axis::X), ("y", Axis::Y), ("z", Axis::Z)]).map_err(access)?),
                None => None,
            };
            Evaluation::Measurement { landmarks: e.get("landmarks").map_err(access)?, axis }
        } else {
            if e.get_opt::<Value>("landmarks").map_err(access)?.is_some()
                || e.get_opt::<Value>("axis").map_err(access)?.is_some()
            {
                return Err(bad("only measurements have landmarks and an axis"));
            }
            let target: String = e.get("target").map_err(access)?;
            match kind {
                "component" => Evaluation::Component(target),
                "expression" => Evaluation::Expression(target),
                "rig" => Evaluation::Rig(target),
                "material" => Evaluation::Material(target),
                "groom" => Evaluation::Groom(target),
                "editor" => Evaluation::Editor(target),
                _ => return Err(bad("unknown evaluation kind")),
            }
        }
    };
    let since_version: u64 = node.get_or("since_version", 1).map_err(access)?;
    let mut tags: Vec<String> = node.get_or("tags", Vec::new()).map_err(access)?;
    tags.sort();
    tags.dedup();
    let param = Param {
        id: id.clone(),
        display_name,
        description,
        group,
        region,
        kind,
        dimension,
        choices,
        unit,
        range,
        default,
        symmetry,
        affects,
        evaluation,
        read_only: node.get_or("read_only", false).map_err(access)?,
        locked_by_default: node.get_or("locked_by_default", false).map_err(access)?,
        plausibility_weight: node.get_or("plausibility_weight", 0.0).map_err(access)?,
        since_version: u32::try_from(since_version).map_err(|_| bad("`since_version` is too large"))?,
        tags,
        provisional: node.get_opt("provisional").map_err(access)?,
    };
    if let Some(editor_only) = node.get_opt::<bool>("editor_only").map_err(access)?
        && editor_only != param.editor_only()
    {
        return Err(bad("`editor_only` contradicts the group and evaluation"));
    }
    if !node.get_or("sides", false).map_err(access)? {
        return Ok(vec![param]);
    }
    if param.symmetry != Symmetry::None {
        return Err(bad("an entry with `sides` gets its symmetry from them"));
    }
    let make = |side: Side| {
        let fill = |s: &str| s.replace(SIDE_PLACEHOLDER, side.letter());
        let mut p = param.clone();
        p.id = format!("{id}.{}", side.letter().to_ascii_lowercase());
        p.display_name = format!("{} ({})", param.display_name, side.name());
        p.region = param.region.as_deref().map(fill);
        p.symmetry =
            Symmetry::Mirror { partner: format!("{id}.{}", side.mirror().letter().to_ascii_lowercase()), side };
        p.evaluation = match &param.evaluation {
            Evaluation::Component(t) => Evaluation::Component(fill(t)),
            Evaluation::Expression(t) => Evaluation::Expression(fill(t)),
            Evaluation::Rig(t) => Evaluation::Rig(fill(t)),
            Evaluation::Material(t) => Evaluation::Material(fill(t)),
            Evaluation::Groom(t) => Evaluation::Groom(fill(t)),
            Evaluation::Editor(t) => Evaluation::Editor(fill(t)),
            Evaluation::Measurement { landmarks: [a, b], axis } => {
                Evaluation::Measurement { landmarks: [fill(a), fill(b)], axis: *axis }
            }
        };
        p
    };
    Ok(vec![make(Side::Left), make(Side::Right)])
}

fn is_word(text: &str, allow_empty: bool) -> bool {
    (allow_empty || !text.is_empty())
        && text.len() <= 64
        && text.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

fn is_name(text: &str) -> bool {
    humanart_model::names::check_name("registry", text).is_ok()
}

/// Checks one entry on its own `[REG-01]`.
pub(super) fn check_param(p: &Param) -> Result<()> {
    let bad = |what: &str| Err(DocError::registry(p.id.clone(), what));
    if !is_param_id(&p.id) {
        return bad("invalid id: dot-separated segments of a-z, 0-9 and _, at most 96 bytes");
    }
    if p.display_name.trim().is_empty() || p.display_name.len() > 128 {
        return bad("the display name must be 1 to 128 bytes");
    }
    if p.description.trim().len() < 16 {
        return bad("the description must say what the parameter does (at least 16 characters)");
    }
    let mut segments = p.group.split('/');
    if !segments.next().is_some_and(|top| GROUPS.contains(&top)) || !p.group.split('/').all(|s| is_word(s, false)) {
        return bad("the group must be a path under one of the REG-03 groups");
    }
    if p.region.as_deref().is_some_and(|r| !is_name(r)) {
        return bad("invalid region name");
    }
    if !is_word(&p.unit, true) {
        return bad("invalid unit word");
    }
    if p.kind == ParamKind::Enum {
        if p.choices.is_empty() || !p.choices.iter().all(|c| is_word(c, false)) {
            return bad("an enum needs choices of a-z, 0-9, _ and -");
        }
        let mut sorted = p.choices.clone();
        sorted.sort();
        sorted.dedup();
        if sorted.len() != p.choices.len() {
            return bad("repeated choice");
        }
    }
    match (p.kind.is_numeric(), &p.range) {
        (true, Some(r)) => {
            let all = [r.min, r.max, r.soft_min, r.soft_max, r.step];
            if !all.iter().all(|v| v.is_finite()) {
                return bad("range values must be finite");
            }
            if !(r.min < r.max && r.min <= r.soft_min && r.soft_min < r.soft_max && r.soft_max <= r.max) {
                return bad("the range must satisfy min <= soft_min < soft_max <= max and min < max");
            }
            if !(r.step > 0.0 && r.step <= r.max - r.min) {
                return bad("the step must be positive and at most max - min");
            }
        }
        (false, None) => {}
        _ => return bad("numeric kinds have a range, the others none"),
    }
    if !matches!(p.kind, ParamKind::Vector | ParamKind::Color) && p.dimension != 1 {
        return bad("only vectors and colours have a dimension");
    }
    p.check_value(&p.default).map_err(|e| DocError::registry(p.id.clone(), format!("bad default: {e}")))?;
    if let Symmetry::Mirror { partner, .. } = &p.symmetry
        && (!is_param_id(partner) || *partner == p.id)
    {
        return bad("invalid mirror partner");
    }
    if p.affects.is_empty() {
        return bad("`affects` must name at least one channel");
    }
    match &p.evaluation {
        Evaluation::Measurement { landmarks, .. } => {
            if !landmarks.iter().all(|l| is_name(l)) || landmarks[0] == landmarks[1] {
                return bad("a measurement needs two different valid landmark names");
            }
        }
        other => {
            if !other.target().is_some_and(is_name) {
                return bad("invalid evaluation target name");
            }
        }
    }
    if p.read_only != matches!(p.evaluation, Evaluation::Measurement { .. }) {
        return bad("exactly the measurements are read-only (REG-05)");
    }
    if !(p.plausibility_weight.is_finite() && p.plausibility_weight >= 0.0) {
        return bad("the plausibility weight must be finite and not negative");
    }
    if p.since_version == 0 {
        return bad("`since_version` starts at 1");
    }
    if !p.tags.iter().all(|t| is_word(t, false)) {
        return bad("tags are words of a-z, 0-9, _ and -");
    }
    if p.provisional.as_deref().is_some_and(|r| r.trim().is_empty()) {
        return bad("a provisional marker must give the reason");
    }
    Ok(())
}

/// Checks a sorted registry: every entry, unique ids, and mirror partners that name each other,
/// sit on opposite sides and share kind, range, choices and default.
pub(super) fn check_registry(registry: &Registry) -> Result<()> {
    let params = registry.params();
    for pair in params.windows(2) {
        if pair[0].id == pair[1].id {
            return Err(DocError::registry(pair[0].id.clone(), "duplicate id"));
        }
    }
    for p in params {
        check_param(p)?;
        if let Symmetry::Mirror { partner, side } = &p.symmetry {
            let bad = |what: &str| Err(DocError::registry(p.id.clone(), what));
            let Some(other) = registry.get(partner) else {
                return bad("the mirror partner does not exist");
            };
            match &other.symmetry {
                Symmetry::Mirror { partner: back, side: other_side }
                    if *back == p.id && *other_side == side.mirror() => {}
                _ => return bad("the mirror partner does not name this parameter on the other side"),
            }
            if other.kind != p.kind
                || other.range != p.range
                || other.choices != p.choices
                || other.dimension != p.dimension
                || other.default != p.default
            {
                return bad("mirror partners must share kind, range, choices and default");
            }
        }
    }
    Ok(())
}
