//! The document's text form: which sections and keys hold what `[DOC-21]`. The layout is listed in
//! `docs/format-anthropos.md`.

use std::collections::{BTreeMap, BTreeSet};

use humanart_core::hash::Hash128;
use humanart_core::id::Id128;
use humanart_io::json::{Number, Object, Value};

use super::{
    Attachment, Author, AuthorKind, Binding, Document, ExpressionPreset, FORMAT_VERSION, GROOM_MAPS, Groom, GroomKind,
    Lock, LockTarget, MAX_NAME_BYTES, MeshFormat, Metadata, Pin, PosePreset, Reference, ReferenceKind, SculptLayer,
    Settings, SkinBlend, SkinChannel, SkinLayer, SkinSource, Snapshot,
};
use crate::error::{DocError, Result};
use crate::registry::{Registry, value_from_json, value_to_json};
use crate::text::{Section, TextFile, is_key};

/// The section kinds of a document, in the order the canonical form writes them.
pub const KINDS: [&str; 16] = [
    "meta",
    "settings",
    "author",
    "params",
    "param_authors",
    "pose",
    "sculpt_layer",
    "skin_layer",
    "groom",
    "expression_preset",
    "pose_preset",
    "lock",
    "pin",
    "snapshot",
    "attachment",
    "reference",
];

/// The section kinds [`Document::snapshot_text`] leaves out.
pub const SNAPSHOT_EXCLUDED_KINDS: [&str; 1] = ["snapshot"];

/// The section kind of a document object list, for the command system's targets: `sculpt_layer`,
/// `skin_layer`, `groom`, `expression_preset`, `pose_preset`, `lock`, `pin`, `snapshot`,
/// `attachment`, `reference`; `None` for anything else.
pub fn section_kind(kind: &str) -> Option<&'static str> {
    KINDS[6..].iter().copied().find(|k| *k == kind)
}

// --- Writing ---------------------------------------------------------------------------------------

fn num(v: f64) -> Value {
    Value::from(Number::Float(v))
}

fn int(v: u64) -> Value {
    Value::from(Number::from(v))
}

fn hash(h: Hash128) -> Value {
    if h.is_zero() { Value::Null } else { Value::from(h.to_hex().as_str()) }
}

fn id(i: Id128) -> Value {
    Value::from(i.to_hex().as_str())
}

fn opt_str(s: Option<&str>) -> Value {
    s.map_or(Value::Null, Value::from)
}

fn floats(vs: &[f64]) -> Value {
    Value::Array(vs.iter().map(|v| num(*v)).collect())
}

fn number_map(map: &BTreeMap<String, f64>) -> Value {
    let mut o = Object::with_capacity(map.len());
    for (k, v) in map {
        o.insert(k.as_str(), num(*v));
    }
    Value::Object(o)
}

/// A section under construction; keys are static and valid, so `set` cannot fail on them.
struct Builder(Section);

impl Builder {
    fn new(kind: &str, key: Option<&str>) -> Result<Self> {
        Ok(Self(Section::new(kind, key)?))
    }

    fn put(&mut self, key: &str, value: Value) -> Result<&mut Self> {
        self.0.set(key, value)?;
        Ok(self)
    }
}

const fn author_kind_name(k: AuthorKind) -> &'static str {
    match k {
        AuthorKind::Human => "human",
        AuthorKind::Agent => "agent",
    }
}

const BLENDS: [(SkinBlend, &str); 4] = [
    (SkinBlend::Mix, "mix"),
    (SkinBlend::Multiply, "multiply"),
    (SkinBlend::Add, "add"),
    (SkinBlend::Overlay, "overlay"),
];
const GROOM_KINDS: [(GroomKind, &str); 5] = [
    (GroomKind::Scalp, "scalp"),
    (GroomKind::Eyebrows, "eyebrows"),
    (GroomKind::Eyelashes, "eyelashes"),
    (GroomKind::Facial, "facial"),
    (GroomKind::Body, "body"),
];
const FORMATS: [(MeshFormat, &str); 2] = [(MeshFormat::Obj, "obj"), (MeshFormat::Glb, "glb")];
const BINDINGS: [(Binding, &str); 2] = [(Binding::Rigid, "rigid"), (Binding::Skinned, "skinned")];
const REFERENCE_KINDS: [(ReferenceKind, &str); 4] = [
    (ReferenceKind::Photo, "photo"),
    (ReferenceKind::Landmarks, "landmarks"),
    (ReferenceKind::ColorTarget, "color_target"),
    (ReferenceKind::Lighting, "lighting"),
];

fn name_of<T: PartialEq + Copy>(table: &[(T, &'static str)], v: T) -> &'static str {
    table.iter().find(|(t, _)| *t == v).map_or("", |(_, n)| n)
}

fn source_value(source: &SkinSource) -> Value {
    let mut o = Object::with_capacity(4);
    match source {
        SkinSource::Procedural { generator, params, seed } => {
            o.insert("generator", generator.as_str());
            o.insert("kind", "procedural");
            o.insert("params", number_map(params));
            o.insert("seed", int(*seed));
        }
        SkinSource::Painted { texture } => {
            o.insert("kind", "painted");
            o.insert("texture", hash(*texture));
        }
        SkinSource::Captured { set } => {
            o.insert("kind", "captured");
            o.insert("set", hash(*set));
        }
    }
    Value::Object(o)
}

pub(super) fn to_text(doc: &Document) -> Result<TextFile> {
    let mut file = TextFile::new(FORMAT_VERSION);
    let m = &doc.meta;
    let mut meta = Builder::new("meta", None)?;
    meta.put("created", int(m.created_ms))?
        .put("id", id(m.id))?
        .put("modified", int(m.modified_ms))?
        .put("name", Value::from(m.name.as_str()))?
        .put("template", Value::from(m.template.as_str()))?
        .put("template_version", int(u64::from(m.template_version)))?;
    file.push(meta.0)?;
    let mut settings = Builder::new("settings", None)?;
    settings.put("protect_human_edits", Value::from(doc.settings.protect_human_edits))?;
    file.push(settings.0)?;
    for a in &doc.authors {
        let mut s = Builder::new("author", Some(&a.id))?;
        s.put("kind", Value::from(author_kind_name(a.kind)))?.put("name", Value::from(a.name.as_str()))?;
        file.push(s.0)?;
    }
    let mut params = Builder::new("params", None)?;
    for (k, v) in &doc.params {
        params.put(k, value_to_json(v))?;
    }
    file.push(params.0)?;
    let mut param_authors = Builder::new("param_authors", None)?;
    for (k, v) in &doc.param_authors {
        param_authors.put(k, Value::from(v.as_str()))?;
    }
    file.push(param_authors.0)?;
    let mut pose = Builder::new("pose", None)?;
    for (joint, q) in &doc.pose {
        pose.put(joint, floats(q))?;
    }
    file.push(pose.0)?;
    for l in &doc.sculpt_layers {
        let mut s = Builder::new("sculpt_layer", Some(&l.id.to_hex()))?;
        s.put("author", Value::from(l.author.as_str()))?
            .put("blend", Value::from("additive"))?
            .put("created_by_command", l.created_by_command.map_or(Value::Null, id))?
            .put("locked", Value::from(l.locked))?
            .put("name", Value::from(l.name.as_str()))?
            .put("opacity", num(l.opacity))?
            .put("payload", hash(l.payload))?
            .put("region", opt_str(l.region.as_deref()))?
            .put("visible", Value::from(l.visible))?;
        file.push(s.0)?;
    }
    for l in &doc.skin_layers {
        let mut s = Builder::new("skin_layer", Some(&l.id.to_hex()))?;
        s.put("author", Value::from(l.author.as_str()))?
            .put("blend", Value::from(name_of(&BLENDS, l.blend)))?
            .put("channels", Value::Array(l.channels.iter().map(|c| Value::from(c.name())).collect()))?
            .put("locked", Value::from(l.locked))?
            .put("name", Value::from(l.name.as_str()))?
            .put("opacity", num(l.opacity))?
            .put("region", opt_str(l.region.as_deref()))?
            .put("source", source_value(&l.source))?
            .put("visible", Value::from(l.visible))?;
        file.push(s.0)?;
    }
    for g in &doc.grooms {
        let mut maps = Object::with_capacity(g.maps.len());
        for (k, v) in &g.maps {
            maps.insert(k.as_str(), hash(*v));
        }
        let mut s = Builder::new("groom", Some(&g.id.to_hex()))?;
        s.put("author", Value::from(g.author.as_str()))?
            .put("guides", hash(g.guides))?
            .put("kind", Value::from(name_of(&GROOM_KINDS, g.kind)))?
            .put("locked", Value::from(g.locked))?
            .put("maps", Value::Object(maps))?
            .put("name", Value::from(g.name.as_str()))?
            .put("params", number_map(&g.params))?
            .put("seed", int(g.seed))?
            .put("visible", Value::from(g.visible))?;
        file.push(s.0)?;
    }
    for p in &doc.expression_presets {
        let mut s = Builder::new("expression_preset", Some(&p.id.to_hex()))?;
        s.put("name", Value::from(p.name.as_str()))?.put("weights", number_map(&p.weights))?;
        file.push(s.0)?;
    }
    for p in &doc.pose_presets {
        let mut joints = Object::with_capacity(p.joints.len());
        for (k, q) in &p.joints {
            joints.insert(k.as_str(), floats(q));
        }
        let mut s = Builder::new("pose_preset", Some(&p.id.to_hex()))?;
        s.put("joints", Value::Object(joints))?.put("name", Value::from(p.name.as_str()))?;
        file.push(s.0)?;
    }
    for l in &doc.locks {
        let (kind, target) = match &l.target {
            LockTarget::Region(r) => ("region", r),
            LockTarget::Parameter(p) => ("parameter", p),
        };
        let mut s = Builder::new("lock", Some(&l.id.to_hex()))?;
        s.put("kind", Value::from(kind))?
            .put("owner", Value::from(l.owner.as_str()))?
            .put("target", Value::from(target.as_str()))?;
        file.push(s.0)?;
    }
    for p in &doc.pins {
        let mut s = Builder::new("pin", Some(&p.id.to_hex()))?;
        s.put("landmark", Value::from(p.landmark.as_str()))?
            .put("owner", Value::from(p.owner.as_str()))?
            .put("position", floats(&p.position))?;
        file.push(s.0)?;
    }
    for p in &doc.snapshots {
        let mut s = Builder::new("snapshot", Some(&p.id.to_hex()))?;
        s.put("author", Value::from(p.author.as_str()))?
            .put("created", int(p.created_ms))?
            .put("name", Value::from(p.name.as_str()))?
            .put("state", hash(p.state))?;
        file.push(s.0)?;
    }
    for a in &doc.attachments {
        let mut s = Builder::new("attachment", Some(&a.id.to_hex()))?;
        s.put("author", Value::from(a.author.as_str()))?
            .put("binding", Value::from(name_of(&BINDINGS, a.binding)))?
            .put("format", Value::from(name_of(&FORMATS, a.format)))?
            .put("joint", Value::from(a.joint.as_str()))?
            .put("materials", hash(a.materials))?
            .put("mesh", hash(a.mesh))?
            .put("name", Value::from(a.name.as_str()))?
            .put("rotation", floats(&a.rotation))?
            .put("scale", floats(&a.scale))?
            .put("translation", floats(&a.translation))?
            .put("visible", Value::from(a.visible))?;
        file.push(s.0)?;
    }
    for r in &doc.references {
        let mut s = Builder::new("reference", Some(&r.id.to_hex()))?;
        s.put("hash", hash(r.hash))?
            .put("kind", Value::from(name_of(&REFERENCE_KINDS, r.kind)))?
            .put("name", Value::from(r.name.as_str()))?
            .put("path", Value::from(r.path.as_str()))?;
        file.push(s.0)?;
    }
    Ok(file)
}

// --- Reading ---------------------------------------------------------------------------------------

/// Reads the entries of one section, each exactly once, and refuses entries nobody read.
struct Reader<'a> {
    section: &'a Section,
    label: String,
    read: Vec<&'static str>,
}

impl<'a> Reader<'a> {
    fn new(section: &'a Section) -> Self {
        Self { section, label: section.reference().to_string(), read: Vec::new() }
    }

    fn fail(&self, key: &str, what: impl Into<String>) -> DocError {
        DocError::field(self.label.clone(), key, what)
    }

    fn value(&mut self, key: &'static str) -> Result<&'a Value> {
        self.read.push(key);
        self.section.get(key).ok_or_else(|| self.fail(key, "missing"))
    }

    fn string(&mut self, key: &'static str) -> Result<String> {
        let v = self.value(key)?;
        v.as_str().map(str::to_owned).ok_or_else(|| self.fail(key, "expected a string"))
    }

    fn opt_string(&mut self, key: &'static str) -> Result<Option<String>> {
        let v = self.value(key)?;
        if v.is_null() {
            return Ok(None);
        }
        v.as_str().map(|s| Some(s.to_owned())).ok_or_else(|| self.fail(key, "expected a string or null"))
    }

    fn name(&mut self, key: &'static str) -> Result<String> {
        let s = self.string(key)?;
        if s.is_empty() || s.len() > MAX_NAME_BYTES {
            return Err(self.fail(key, "a name must be 1 to 256 bytes"));
        }
        Ok(s)
    }

    fn f64(&mut self, key: &'static str) -> Result<f64> {
        let v = self.value(key)?;
        v.as_f64().ok_or_else(|| self.fail(key, "expected a number"))
    }

    fn u64(&mut self, key: &'static str) -> Result<u64> {
        let v = self.value(key)?;
        v.as_u64().ok_or_else(|| self.fail(key, "expected a non-negative integer"))
    }

    fn bool(&mut self, key: &'static str) -> Result<bool> {
        let v = self.value(key)?;
        v.as_bool().ok_or_else(|| self.fail(key, "expected true or false"))
    }

    fn hash(&mut self, key: &'static str) -> Result<Hash128> {
        let v = self.value(key)?;
        parse_hash(v).ok_or_else(|| self.fail(key, "expected a 32-digit hash or null"))
    }

    fn id(&mut self, key: &'static str) -> Result<Id128> {
        let v = self.value(key)?;
        v.as_str().and_then(|s| Id128::parse(s).ok()).ok_or_else(|| self.fail(key, "expected a 32-digit id"))
    }

    fn floats<const N: usize>(&mut self, key: &'static str) -> Result<[f64; N]> {
        let v = self.value(key)?;
        parse_floats(v).ok_or_else(|| self.fail(key, format!("expected an array of {N} numbers")))
    }

    fn choice<T: Copy>(&mut self, key: &'static str, table: &[(T, &'static str)]) -> Result<T> {
        let s = self.string(key)?;
        table.iter().find(|(_, n)| *n == s).map(|(t, _)| *t).ok_or_else(|| self.fail(key, "not one of the choices"))
    }

    fn finish(self) -> Result<()> {
        for (key, _) in self.section.entries() {
            if !self.read.contains(&key.as_str()) {
                return Err(self.fail(key, "unknown entry"));
            }
        }
        Ok(())
    }
}

fn parse_hash(v: &Value) -> Option<Hash128> {
    if v.is_null() {
        return Some(Hash128::ZERO);
    }
    let h = Hash128::parse(v.as_str()?).ok()?;
    // ZERO is written as null; the 32 zeros are not a second spelling of it.
    (!h.is_zero()).then_some(h)
}

fn parse_floats<const N: usize>(v: &Value) -> Option<[f64; N]> {
    let items = v.as_array()?;
    if items.len() != N {
        return None;
    }
    let mut out = [0.0; N];
    for (o, item) in out.iter_mut().zip(items) {
        *o = item.as_f64()?;
    }
    Some(out)
}

fn number_map_from(v: &Value) -> Option<BTreeMap<String, f64>> {
    let mut out = BTreeMap::new();
    for (k, v) in v.as_object()? {
        if !is_key(k) {
            return None;
        }
        out.insert(k.to_owned(), v.as_f64()?);
    }
    Some(out)
}

fn object_id(section: &Section) -> Result<Id128> {
    section
        .key()
        .and_then(|k| Id128::parse(k).ok())
        .filter(|i| !i.is_nil())
        .ok_or_else(|| DocError::field(section.reference().to_string(), "", "the section key must be a 32-digit id"))
}

fn source_from(v: &Value) -> Option<SkinSource> {
    let o = v.as_object()?;
    let only = |keys: &[&str]| o.len() == keys.len() && keys.iter().all(|k| o.contains_key(k));
    match o.get("kind")?.as_str()? {
        "procedural" if only(&["generator", "kind", "params", "seed"]) => {
            let generator = o.get("generator")?.as_str()?;
            if !is_key(generator) {
                return None;
            }
            Some(SkinSource::Procedural {
                generator: generator.to_owned(),
                params: number_map_from(o.get("params")?)?,
                seed: o.get("seed")?.as_u64()?,
            })
        }
        "painted" if only(&["kind", "texture"]) => {
            Some(SkinSource::Painted { texture: parse_hash(o.get("texture")?)? })
        }
        "captured" if only(&["kind", "set"]) => Some(SkinSource::Captured { set: parse_hash(o.get("set")?)? }),
        _ => None,
    }
}

pub(super) fn from_text(file: &TextFile, registry: &Registry) -> Result<Document> {
    if file.version != FORMAT_VERSION {
        return Err(DocError::Version {
            format: "document",
            found: u64::from(file.version),
            supported: u64::from(FORMAT_VERSION),
        });
    }
    for s in file.sections() {
        if !KINDS.contains(&s.kind()) {
            return Err(DocError::field(s.kind(), "", "unknown section kind"));
        }
        let keyed = !matches!(s.kind(), "meta" | "settings" | "params" | "param_authors" | "pose");
        if keyed != s.key().is_some() {
            return Err(DocError::field(
                s.reference().to_string(),
                "",
                "this kind of section has a key iff it is a list",
            ));
        }
    }
    let meta_section =
        file.section("meta", None).ok_or_else(|| DocError::field("meta", "", "the document has no [meta] section"))?;
    let mut r = Reader::new(meta_section);
    let meta = Metadata {
        created_ms: r.u64("created")?,
        id: r.id("id")?,
        modified_ms: r.u64("modified")?,
        name: r.name("name")?,
        template: r.string("template")?,
        template_version: u32::try_from(r.u64("template_version")?)
            .map_err(|_| DocError::field("meta", "template_version", "too large"))?,
    };
    r.finish()?;
    let settings = match file.section("settings", None) {
        Some(s) => {
            let mut r = Reader::new(s);
            let settings = Settings { protect_human_edits: r.bool("protect_human_edits")? };
            r.finish()?;
            settings
        }
        None => Settings::default(),
    };
    let mut doc = Document::new(
        meta.id,
        &meta.name,
        &meta.template,
        meta.template_version,
        0,
        Author { id: String::new(), kind: AuthorKind::Human, name: String::new() },
    );
    doc.meta = meta;
    doc.settings = settings;
    doc.authors.clear();
    for s in file.sections_of("author") {
        let mut r = Reader::new(s);
        let author = Author {
            id: s.key().unwrap_or_default().to_owned(),
            kind: r.choice("kind", &[(AuthorKind::Human, "human"), (AuthorKind::Agent, "agent")])?,
            name: r.name("name")?,
        };
        r.finish()?;
        doc.add_author(author);
    }
    if let Some(s) = file.section("params", None) {
        for (key, value) in s.entries() {
            let param = registry.require(key)?;
            let v = value_from_json(param, value)?;
            if !param.is_default(&v) {
                doc.params.insert(key.clone(), v);
            }
        }
    }
    if let Some(s) = file.section("param_authors", None) {
        for (key, value) in s.entries() {
            let author = value.as_str().ok_or_else(|| DocError::field("param_authors", key, "expected a string"))?;
            if doc.params.contains_key(key) {
                doc.param_authors.insert(key.clone(), author.to_owned());
            }
        }
    }
    if let Some(s) = file.section("pose", None) {
        for (key, value) in s.entries() {
            let q = parse_floats(value).ok_or_else(|| DocError::field("pose", key, "expected [x, y, z, w]"))?;
            doc.pose.insert(key.clone(), q);
        }
    }
    for s in file.sections_of("sculpt_layer") {
        let mut r = Reader::new(s);
        let layer = SculptLayer {
            id: object_id(s)?,
            author: r.string("author")?,
            created_by_command: {
                let v = r.value("created_by_command")?;
                if v.is_null() {
                    None
                } else {
                    Some(v.as_str().and_then(|t| Id128::parse(t).ok()).ok_or_else(|| {
                        DocError::field(s.reference().to_string(), "created_by_command", "expected an id or null")
                    })?)
                }
            },
            locked: r.bool("locked")?,
            name: r.name("name")?,
            opacity: r.f64("opacity")?,
            payload: r.hash("payload")?,
            region: r.opt_string("region")?,
            visible: r.bool("visible")?,
        };
        if r.string("blend")? != "additive" {
            return Err(DocError::field(s.reference().to_string(), "blend", "sculpt layers blend additively"));
        }
        r.finish()?;
        doc.sculpt_layers.push(layer);
    }
    for s in file.sections_of("skin_layer") {
        let mut r = Reader::new(s);
        let label = s.reference().to_string();
        let channels_value = r.value("channels")?;
        let mut channels = Vec::new();
        for c in
            channels_value.as_array().ok_or_else(|| DocError::field(label.clone(), "channels", "expected an array"))?
        {
            let name = c.as_str().unwrap_or_default();
            let channel = SkinChannel::ALL
                .iter()
                .find(|(_, n)| *n == name)
                .map(|(ch, _)| *ch)
                .ok_or_else(|| DocError::field(label.clone(), "channels", "unknown channel"))?;
            channels.push(channel);
        }
        let source_raw = r.value("source")?;
        let layer = SkinLayer {
            id: object_id(s)?,
            author: r.string("author")?,
            blend: r.choice("blend", &BLENDS)?,
            channels,
            locked: r.bool("locked")?,
            name: r.name("name")?,
            opacity: r.f64("opacity")?,
            region: r.opt_string("region")?,
            source: source_from(source_raw)
                .ok_or_else(|| DocError::field(label.clone(), "source", "invalid source"))?,
            visible: r.bool("visible")?,
        };
        r.finish()?;
        doc.skin_layers.push(layer);
    }
    for s in file.sections_of("groom") {
        let mut r = Reader::new(s);
        let label = s.reference().to_string();
        let maps_value = r.value("maps")?;
        let mut maps = BTreeMap::new();
        for (k, v) in
            maps_value.as_object().ok_or_else(|| DocError::field(label.clone(), "maps", "expected an object"))?
        {
            let h = parse_hash(v).ok_or_else(|| DocError::field(label.clone(), "maps", "expected hashes"))?;
            maps.insert(k.to_owned(), h);
        }
        let params_value = r.value("params")?;
        let groom = Groom {
            id: object_id(s)?,
            author: r.string("author")?,
            guides: r.hash("guides")?,
            kind: r.choice("kind", &GROOM_KINDS)?,
            locked: r.bool("locked")?,
            maps,
            name: r.name("name")?,
            params: number_map_from(params_value)
                .ok_or_else(|| DocError::field(label.clone(), "params", "expected an object of numbers"))?,
            seed: r.u64("seed")?,
            visible: r.bool("visible")?,
        };
        r.finish()?;
        doc.grooms.push(groom);
    }
    for s in file.sections_of("expression_preset") {
        let mut r = Reader::new(s);
        let weights_value = r.value("weights")?;
        let preset = ExpressionPreset {
            id: object_id(s)?,
            name: r.name("name")?,
            weights: number_map_from(weights_value).ok_or_else(|| {
                DocError::field(s.reference().to_string(), "weights", "expected an object of numbers")
            })?,
        };
        r.finish()?;
        doc.expression_presets.push(preset);
    }
    for s in file.sections_of("pose_preset") {
        let mut r = Reader::new(s);
        let label = s.reference().to_string();
        let joints_value = r.value("joints")?;
        let mut joints = BTreeMap::new();
        for (k, v) in
            joints_value.as_object().ok_or_else(|| DocError::field(label.clone(), "joints", "expected an object"))?
        {
            let q = parse_floats(v).ok_or_else(|| DocError::field(label.clone(), "joints", "expected [x, y, z, w]"))?;
            joints.insert(k.to_owned(), q);
        }
        let preset = PosePreset { id: object_id(s)?, name: r.name("name")?, joints };
        r.finish()?;
        doc.pose_presets.push(preset);
    }
    for s in file.sections_of("lock") {
        let mut r = Reader::new(s);
        let kind = r.choice("kind", &[(0_u8, "region"), (1_u8, "parameter")])?;
        let target = r.string("target")?;
        let lock = Lock {
            id: object_id(s)?,
            target: if kind == 0 { LockTarget::Region(target) } else { LockTarget::Parameter(target) },
            owner: r.string("owner")?,
        };
        r.finish()?;
        doc.locks.push(lock);
    }
    for s in file.sections_of("pin") {
        let mut r = Reader::new(s);
        let pin = Pin {
            id: object_id(s)?,
            landmark: r.string("landmark")?,
            owner: r.string("owner")?,
            position: r.floats("position")?,
        };
        r.finish()?;
        doc.pins.push(pin);
    }
    for s in file.sections_of("snapshot") {
        let mut r = Reader::new(s);
        let snapshot = Snapshot {
            id: object_id(s)?,
            author: r.string("author")?,
            created_ms: r.u64("created")?,
            name: r.name("name")?,
            state: r.hash("state")?,
        };
        r.finish()?;
        doc.snapshots.push(snapshot);
    }
    for s in file.sections_of("attachment") {
        let mut r = Reader::new(s);
        let attachment = Attachment {
            id: object_id(s)?,
            author: r.string("author")?,
            binding: r.choice("binding", &BINDINGS)?,
            format: r.choice("format", &FORMATS)?,
            joint: r.string("joint")?,
            materials: r.hash("materials")?,
            mesh: r.hash("mesh")?,
            name: r.name("name")?,
            rotation: r.floats("rotation")?,
            scale: r.floats("scale")?,
            translation: r.floats("translation")?,
            visible: r.bool("visible")?,
        };
        r.finish()?;
        doc.attachments.push(attachment);
    }
    for s in file.sections_of("reference") {
        let mut r = Reader::new(s);
        let reference = Reference {
            id: object_id(s)?,
            hash: r.hash("hash")?,
            kind: r.choice("kind", &REFERENCE_KINDS)?,
            name: r.name("name")?,
            path: r.string("path")?,
        };
        r.finish()?;
        doc.references.push(reference);
    }
    check(&doc, registry)?;
    Ok(doc)
}

// --- Checks ----------------------------------------------------------------------------------------

fn is_name(s: &str) -> bool {
    humanart_model::names::check_name("document", s).is_ok()
}

fn is_unit_quat(q: &[f64; 4]) -> bool {
    let n = q.iter().map(|v| v * v).sum::<f64>();
    q.iter().all(|v| v.is_finite()) && (n - 1.0).abs() <= 1e-6
}

fn unique<T>(items: &[T], id: impl Fn(&T) -> Id128, kind: &str) -> Result<()> {
    let mut seen = BTreeSet::new();
    for item in items {
        let i = id(item);
        if i.is_nil() || !seen.insert(i) {
            return Err(DocError::field(kind, i.to_hex().as_str(), "nil or repeated id"));
        }
    }
    Ok(())
}

fn is_relative_path(p: &str) -> bool {
    !p.is_empty()
        && p.len() <= 512
        && p.split('/').all(|seg| {
            !seg.is_empty()
                && seg != "."
                && seg != ".."
                && seg.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b' '))
        })
}

pub(super) fn check(doc: &Document, registry: &Registry) -> Result<()> {
    let m = &doc.meta;
    if m.id.is_nil() {
        return Err(DocError::field("meta", "id", "the document id must not be nil"));
    }
    if m.name.is_empty() || m.name.len() > MAX_NAME_BYTES {
        return Err(DocError::field("meta", "name", "a name must be 1 to 256 bytes"));
    }
    if !is_name(&m.template) {
        return Err(DocError::field("meta", "template", "invalid template name"));
    }
    if m.modified_ms < m.created_ms {
        return Err(DocError::field("meta", "modified", "modified before created"));
    }
    if doc.authors.is_empty() {
        return Err(DocError::field("author", "", "a document has at least one author"));
    }
    for pair in doc.authors.windows(2) {
        if pair[0].id >= pair[1].id {
            return Err(DocError::field("author", pair[1].id.as_str(), "authors must be sorted and unique"));
        }
    }
    for a in &doc.authors {
        if !is_key(&a.id) || a.name.is_empty() || a.name.len() > MAX_NAME_BYTES {
            return Err(DocError::field("author", a.id.as_str(), "invalid author id or name"));
        }
    }
    let author = |section: &str, key: &str, id: &str| {
        if doc.author(id).is_some() {
            Ok(())
        } else {
            Err(DocError::field(section, key, format!("unknown author {id}")))
        }
    };
    for (id, value) in &doc.params {
        let param = registry.require(id)?;
        if param.read_only {
            return Err(DocError::param_value(id.as_str(), "read-only measurements are not stored"));
        }
        param.check_value(value)?;
        if param.is_default(value) {
            return Err(DocError::param_value(id.as_str(), "default values are not stored"));
        }
    }
    for (id, who) in &doc.param_authors {
        if !doc.params.contains_key(id) {
            return Err(DocError::field("param_authors", id.as_str(), "no stored value for this parameter"));
        }
        author("param_authors", id, who)?;
    }
    for (joint, q) in &doc.pose {
        if !is_name(joint) || !is_unit_quat(q) {
            return Err(DocError::field("pose", joint.as_str(), "needs a joint name and a unit quaternion"));
        }
    }
    let name_ok = |n: &str| !n.is_empty() && n.len() <= MAX_NAME_BYTES;
    let opacity_ok = |o: f64| (0.0..=1.0).contains(&o);
    unique(&doc.sculpt_layers, |l| l.id, "sculpt_layer")?;
    for l in &doc.sculpt_layers {
        let key = l.id.to_hex();
        if !name_ok(&l.name) || !opacity_ok(l.opacity) || l.region.as_deref().is_some_and(|r| !is_name(r)) {
            return Err(DocError::field("sculpt_layer", key.as_str(), "invalid name, opacity or region"));
        }
        author("sculpt_layer", &key, &l.author)?;
    }
    unique(&doc.skin_layers, |l| l.id, "skin_layer")?;
    for l in &doc.skin_layers {
        let key = l.id.to_hex();
        let sorted = l.channels.windows(2).all(|w| w[0] < w[1]);
        if !name_ok(&l.name) || !opacity_ok(l.opacity) || l.region.as_deref().is_some_and(|r| !is_name(r)) || !sorted {
            return Err(DocError::field("skin_layer", key.as_str(), "invalid name, opacity, region or channel set"));
        }
        if let SkinSource::Procedural { generator, params, .. } = &l.source
            && (!is_key(generator) || !params.iter().all(|(k, v)| is_key(k) && v.is_finite()))
        {
            return Err(DocError::field("skin_layer", key.as_str(), "invalid generator or parameters"));
        }
        author("skin_layer", &key, &l.author)?;
    }
    unique(&doc.grooms, |g| g.id, "groom")?;
    for g in &doc.grooms {
        let key = g.id.to_hex();
        let params_ok = g.params.iter().all(|(k, v)| is_key(k) && v.is_finite());
        let maps_ok = g.maps.keys().all(|k| GROOM_MAPS.contains(&k.as_str()));
        if !name_ok(&g.name) || !params_ok || !maps_ok {
            return Err(DocError::field("groom", key.as_str(), "invalid name, parameters or map names"));
        }
        author("groom", &key, &g.author)?;
    }
    unique(&doc.expression_presets, |p| p.id, "expression_preset")?;
    for p in &doc.expression_presets {
        if !name_ok(&p.name) {
            return Err(DocError::field("expression_preset", p.id.to_hex().as_str(), "invalid name"));
        }
        for (id, w) in &p.weights {
            let param = registry.require(id)?;
            if !param.group.starts_with("expression") {
                return Err(DocError::param_value(id.as_str(), "presets hold expression parameters only"));
            }
            param.check_value(&crate::registry::ParamValue::Scalar(*w))?;
        }
    }
    unique(&doc.pose_presets, |p| p.id, "pose_preset")?;
    for p in &doc.pose_presets {
        if !name_ok(&p.name) || !p.joints.iter().all(|(j, q)| is_name(j) && is_unit_quat(q)) {
            return Err(DocError::field("pose_preset", p.id.to_hex().as_str(), "invalid name, joint or quaternion"));
        }
    }
    unique(&doc.locks, |l| l.id, "lock")?;
    for l in &doc.locks {
        let key = l.id.to_hex();
        match &l.target {
            LockTarget::Region(r) if !is_name(r) => {
                return Err(DocError::field("lock", key.as_str(), "invalid region name"));
            }
            LockTarget::Parameter(p) => {
                registry.require(p)?;
            }
            LockTarget::Region(_) => {}
        }
        author("lock", &key, &l.owner)?;
    }
    unique(&doc.pins, |p| p.id, "pin")?;
    for p in &doc.pins {
        let key = p.id.to_hex();
        if !is_name(&p.landmark) || !p.position.iter().all(|v| v.is_finite()) {
            return Err(DocError::field("pin", key.as_str(), "invalid landmark or position"));
        }
        author("pin", &key, &p.owner)?;
    }
    unique(&doc.snapshots, |s| s.id, "snapshot")?;
    for s in &doc.snapshots {
        let key = s.id.to_hex();
        if !name_ok(&s.name) || s.state.is_zero() {
            return Err(DocError::field("snapshot", key.as_str(), "invalid name or missing state"));
        }
        author("snapshot", &key, &s.author)?;
    }
    unique(&doc.attachments, |a| a.id, "attachment")?;
    for a in &doc.attachments {
        let key = a.id.to_hex();
        let finite = a.translation.iter().chain(&a.scale).all(|v| v.is_finite());
        if !name_ok(&a.name) || !is_name(&a.joint) || a.mesh.is_zero() || !finite || !is_unit_quat(&a.rotation) {
            return Err(DocError::field("attachment", key.as_str(), "invalid name, joint, mesh or transform"));
        }
        author("attachment", &key, &a.author)?;
    }
    unique(&doc.references, |r| r.id, "reference")?;
    for r in &doc.references {
        if !name_ok(&r.name) || !is_relative_path(&r.path) {
            return Err(DocError::field("reference", r.id.to_hex().as_str(), "invalid name or path"));
        }
    }
    Ok(())
}
