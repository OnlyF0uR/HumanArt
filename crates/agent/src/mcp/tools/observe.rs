//! Observation tools (section 10.3): read-only views of the character, the registry, the template,
//! renders, events and history.

// Every handler has the table's signature, so some return `Ok` only.
#![allow(clippy::unnecessary_wraps)]

use humanart_doc::document::{AuthorKind, Document, LockTarget};
use humanart_doc::notify::Since;
use humanart_doc::registry::{Evaluation, Param, Registry, describe_param, value_to_json};
use humanart_io::json::{Object, Value};
use humanart_math::Vec3;
use humanart_math::color::lab::rgb_to_lab;
use humanart_math::color::transfer::srgb_to_linear;
use humanart_render::camera::Camera;
use humanart_render::capture::encode_png_rgba8;

use super::super::render::{
    CAMERA_PRESETS, CameraSpec, QUALITY_NAMES, RIG_NAMES, ViewRequest, project, quality_by_name, quality_name,
    rig_by_name, rig_name, view_by_name,
};
use super::super::schema::{Field, Ty};
use super::{Access, PendingRender, RenderRecord, ToolCtx, ToolError, ToolOutput, ToolSpec, args, num, object, uint};
use crate::engine::Model;

const QUERY_FIELDS: [Field; 7] = [
    Field::opt("query", Ty::Str { max: 256 }, "Words to search in ids, names, descriptions and tags."),
    Field::opt("group", Ty::Str { max: 128 }, "Only this group and its subgroups (e.g. face/nose)."),
    Field::opt("region", Ty::Region, "Only parameters of this region."),
    Field::opt("tag", Ty::Str { max: 64 }, "Only parameters with this semantic tag."),
    Field::opt("changed_only", Ty::Bool, "Only parameters whose value differs from the default."),
    Field::opt("limit", Ty::Int { min: 1, max: 500 }, "Most results (default 50)."),
    Field::opt("offset", Ty::Int { min: 0, max: 100_000 }, "Results to skip (paging)."),
];
const IDS_FIELDS: [Field; 1] = [Field::req("ids", Ty::List(&Ty::Param, 256), "Parameter ids.")];
const LAYER_FIELDS: [Field; 1] = [Field::req("id", Ty::Id, "The layer or groom id.")];
const REGION_FIELDS: [Field; 2] = [
    Field::req("region", Ty::Region, "The region (its subregions are included)."),
    Field::opt(
        "render_id",
        Ty::Int { min: 0, max: i64::MAX },
        "Colour statistics from this render (default: the last).",
    ),
];
const PAIR: Ty = Ty::List(&Ty::Landmark, 2);
const RATIO: [Field; 2] =
    [Field::req("numerator", PAIR, "Two landmarks."), Field::req("denominator", PAIR, "Two landmarks.")];
const MEASURE_FIELDS: [Field; 4] = [
    Field::opt("distances", Ty::List(&PAIR, 64), "Landmark pairs: distance in metres."),
    Field::opt("ratios", Ty::List(&Ty::Record(&RATIO), 32), "Distance ratios."),
    Field::opt(
        "angles",
        Ty::List(&Ty::List(&Ty::Landmark, 3), 32),
        "Landmark triples: the angle at the middle one, degrees.",
    ),
    Field::opt("named", Ty::List(&Ty::Param, 64), "Measured registry parameters (identity/measurements)."),
];
const EXPLICIT_CAMERA: [Field; 4] = [
    Field::req("position", Ty::Vec3, "Eye position, metres."),
    Field::req("target", Ty::Vec3, "Point looked at, metres."),
    Field::opt("up", Ty::Vec3, "Up direction (default +Y)."),
    Field::opt("focal_mm", Ty::Num { min: 14.0, max: 300.0 }, "Focal length (default 85)."),
];
const RENDER_FIELDS: [Field; 14] = [
    Field::opt("camera", Ty::Choice(&CAMERA_PRESETS), "Camera preset (default: the set_camera default, else head)."),
    Field::opt("explicit_camera", Ty::Record(&EXPLICIT_CAMERA), "An explicit camera instead of a preset."),
    Field::opt("views", Ty::List(&Ty::Choice(&CAMERA_PRESETS), 24), "A batch of presets (e.g. the review cameras)."),
    Field::opt("turntable", Ty::Int { min: 0, max: 24 }, "Render this many angles evenly around the character."),
    Field::opt("rig", Ty::Choice(&RIG_NAMES), "Light rig (default: the set_lighting_rig default)."),
    Field::opt("quality", Ty::Choice(&QUALITY_NAMES), "Quality tier (default review)."),
    Field::opt("view_mode", Ty::Str { max: 32 }, "beauty (default) or a debug view: albedo, normals, depth ..."),
    Field::opt("wireframe", Ty::Bool, "Draw the wireframe."),
    Field::opt("width", Ty::Int { min: 16, max: 4096 }, "Width in pixels (default 768)."),
    Field::opt("height", Ty::Int { min: 16, max: 4096 }, "Height in pixels (default 768)."),
    Field::opt("frames", Ty::Int { min: 1, max: 256 }, "Frames accumulated before the capture (default 16)."),
    Field::opt("region", Ty::Region, "Crop to this region's projected bounds."),
    Field::opt("inline", Ty::Bool, "Return the PNGs as image content items too (default true)."),
    Field::opt("async", Ty::Bool, "Return at once and send notifications/humanart/render_ready when done."),
];
const EVENT_FIELDS: [Field; 2] = [
    Field::opt("since", Ty::Int { min: 0, max: i64::MAX }, "Events after this sequence number (default 0)."),
    Field::opt("limit", Ty::Int { min: 1, max: 1000 }, "Most events (default 200)."),
];
const HISTORY_FIELDS: [Field; 1] = [Field::opt("limit", Ty::Int { min: 1, max: 500 }, "Most entries (default 20).")];

/// The observation tools.
pub static TOOLS: [ToolSpec; 11] = [
    ToolSpec {
        name: "describe_character",
        title: "Describe the character",
        description: "Summary of the state: non-default parameters, layers, grooms, locks, pins, snapshots, human-protection markers, open proposals and the plausibility score when a scorer is installed.",
        fields: &[],
        access: Access::Read,
        handler: describe_character,
    },
    ToolSpec {
        name: "describe_parameters",
        title: "Search the parameter registry",
        description: "Registry search by words, group, region or semantic tag, with ranges, units, descriptions, current values, authors, locks and protection.",
        fields: &QUERY_FIELDS,
        access: Access::Read,
        handler: describe_parameters,
    },
    ToolSpec {
        name: "get_parameters",
        title: "Get parameters",
        description: "Values and metadata of the given parameter ids.",
        fields: &IDS_FIELDS,
        access: Access::Read,
        handler: get_parameters,
    },
    ToolSpec {
        name: "list_layers",
        title: "List layers",
        description: "Sculpt layers, skin layers and grooms with their metadata (no blobs).",
        fields: &[],
        access: Access::Read,
        handler: list_layers,
    },
    ToolSpec {
        name: "get_layer",
        title: "Get a layer",
        description: "One layer or groom: metadata and statistics (blob size, region vertex count).",
        fields: &LAYER_FIELDS,
        access: Access::Read,
        handler: get_layer,
    },
    ToolSpec {
        name: "list_regions",
        title: "List regions",
        description: "Template regions: name, parent, side, vertex count, lock state.",
        fields: &[],
        access: Access::Read,
        handler: list_regions,
    },
    ToolSpec {
        name: "get_region_stats",
        title: "Region statistics",
        description: "Vertex and triangle counts, area, bounds of the evaluated surface, and colour statistics (linear mean, Lab mean and standard deviation) of the region's camera-facing vertices in a render.",
        fields: &REGION_FIELDS,
        access: Access::Read,
        handler: get_region_stats,
    },
    ToolSpec {
        name: "measure",
        title: "Measure",
        description: "Landmark distances, ratios and angles on the rest shape, and the registry's named measurements.",
        fields: &MEASURE_FIELDS,
        access: Access::Read,
        handler: measure,
    },
    ToolSpec {
        name: "render_view",
        title: "Render a view",
        description: "Renders with the editor's renderer: camera preset or explicit, light rig, resolution, quality tier and view mode; batches of review cameras or turntable angles; region crops. Each image is written as a PNG to the exchange directory and returned as an image content item with its metadata.",
        fields: &RENDER_FIELDS,
        access: Access::Read,
        handler: render_view,
    },
    ToolSpec {
        name: "get_events",
        title: "Events since",
        description: "Notification events after sequence number N (what the human and the agent did); reports a gap when they are no longer kept.",
        fields: &EVENT_FIELDS,
        access: Access::Read,
        handler: get_events,
    },
    ToolSpec {
        name: "get_history",
        title: "History",
        description: "Recent history entries with their authors, newest last, and the undo cursor.",
        fields: &HISTORY_FIELDS,
        access: Access::Read,
        handler: get_history,
    },
];

fn is_human(doc: &Document, author: &str) -> bool {
    doc.author(author).is_some_and(|a| a.kind == AuthorKind::Human)
}

/// Whether the parameter's value is protected from the agent `[LOCK-02]`.
pub(crate) fn param_protected(doc: &Document, id: &str) -> bool {
    doc.settings.protect_human_edits && doc.param_authors.get(id).is_some_and(|a| is_human(doc, a))
}

fn region_locked(doc: &Document, region: &str) -> bool {
    doc.locks.iter().any(|l| matches!(&l.target, LockTarget::Region(r) if r == region))
}

fn param_locked(doc: &Document, p: &Param) -> bool {
    doc.locks.iter().any(|l| matches!(&l.target, LockTarget::Parameter(x) if *x == p.id))
        || p.region.as_deref().is_some_and(|r| region_locked(doc, r))
}

/// A parameter's description with its state in the document.
fn param_state(doc: &Document, reg: &Registry, p: &Param) -> Value {
    let mut v = describe_param(p);
    let value = doc.param(reg, &p.id).map_or(Value::Null, value_to_json);
    if let Value::Object(o) = &mut v {
        o.insert("value", value);
        o.insert("is_default", doc.params.get(&p.id).is_none_or(|x| p.is_default(x)));
        o.insert("author", doc.param_authors.get(&p.id).map_or(Value::Null, |a| Value::from(a.as_str())));
        o.insert("locked", param_locked(doc, p));
        o.insert("protected", param_protected(doc, &p.id));
    }
    v
}

fn layer_common(
    doc: &Document,
    id: humanart_core::id::Id128,
    name: &str,
    author: &str,
    locked: bool,
    visible: bool,
) -> Object {
    let mut o = Object::with_capacity(10);
    o.insert("id", id.to_hex().as_str());
    o.insert("name", name);
    o.insert("author", author);
    o.insert("locked", locked);
    o.insert("visible", visible);
    o.insert("protected", doc.settings.protect_human_edits && is_human(doc, author));
    o
}

fn sculpt_json(doc: &Document, l: &humanart_doc::document::SculptLayer) -> Value {
    let mut o = layer_common(doc, l.id, &l.name, &l.author, l.locked, l.visible);
    o.insert("kind", "sculpt");
    o.insert("region", l.region.as_deref().map_or(Value::Null, Value::from));
    o.insert("opacity", num(l.opacity));
    o.insert("payload", l.payload.to_hex().as_str());
    Value::Object(o)
}

fn skin_json(doc: &Document, l: &humanart_doc::document::SkinLayer) -> Value {
    use humanart_doc::document::SkinSource;
    let mut o = layer_common(doc, l.id, &l.name, &l.author, l.locked, l.visible);
    o.insert("kind", "skin");
    o.insert("region", l.region.as_deref().map_or(Value::Null, Value::from));
    o.insert("opacity", num(l.opacity));
    o.insert("channels", Value::Array(l.channels.iter().map(|c| Value::from(c.name())).collect()));
    let source = match &l.source {
        SkinSource::Procedural { generator, params, seed } => object(vec![
            ("kind", "procedural".into()),
            ("generator", generator.as_str().into()),
            ("params", Value::Object(params.iter().map(|(k, v)| (k.as_str(), num(*v))).collect())),
            ("seed", uint(*seed)),
        ]),
        SkinSource::Painted { texture } => {
            object(vec![("kind", "painted".into()), ("texture", texture.to_hex().as_str().into())])
        }
        SkinSource::Captured { set } => {
            object(vec![("kind", "captured".into()), ("set", set.to_hex().as_str().into())])
        }
    };
    o.insert("source", source);
    Value::Object(o)
}

fn groom_json(doc: &Document, g: &humanart_doc::document::Groom) -> Value {
    let mut o = layer_common(doc, g.id, &g.name, &g.author, g.locked, g.visible);
    o.insert("kind", "groom");
    o.insert("groom_kind", format!("{:?}", g.kind).to_ascii_lowercase());
    o.insert("guides", g.guides.to_hex().as_str());
    o.insert("params", Value::Object(g.params.iter().map(|(k, v)| (k.as_str(), num(*v))).collect()));
    o.insert("seed", uint(g.seed));
    Value::Object(o)
}

fn describe_character(ctx: &mut ToolCtx<'_, '_>, _: &Value) -> Result<ToolOutput, ToolError> {
    let session = ctx.engine.session();
    let doc = session.document();
    let reg = ctx.engine.registry();
    let changed: Vec<Value> = reg
        .params()
        .iter()
        .filter(|p| doc.params.get(&p.id).is_some_and(|v| !p.is_default(v)))
        .map(|p| {
            object(vec![
                ("id", p.id.as_str().into()),
                ("value", doc.param(reg, &p.id).map_or(Value::Null, value_to_json)),
                ("author", doc.param_authors.get(&p.id).map_or(Value::Null, |a| Value::from(a.as_str()))),
                ("protected", param_protected(doc, &p.id).into()),
            ])
        })
        .collect();
    let locks: Vec<Value> = doc
        .locks
        .iter()
        .map(|l| {
            let (kind, target) = match &l.target {
                LockTarget::Region(r) => ("region", r),
                LockTarget::Parameter(p) => ("parameter", p),
            };
            object(vec![("kind", kind.into()), ("target", target.as_str().into()), ("owner", l.owner.as_str().into())])
        })
        .collect();
    let pins: Vec<Value> = doc
        .pins
        .iter()
        .map(|p| {
            object(vec![
                ("landmark", p.landmark.as_str().into()),
                ("position", Value::Array(p.position.iter().map(|x| num(*x)).collect())),
                ("owner", p.owner.as_str().into()),
            ])
        })
        .collect();
    let snapshots: Vec<Value> = doc
        .snapshots
        .iter()
        .map(|s| {
            object(vec![
                ("id", s.id.to_hex().as_str().into()),
                ("name", s.name.as_str().into()),
                ("author", s.author.as_str().into()),
                ("created", uint(s.created_ms)),
            ])
        })
        .collect();
    let proposals: Vec<Value> = session
        .proposals()
        .iter()
        .map(|p| {
            object(vec![
                ("id", p.id.to_hex().as_str().into()),
                ("name", p.name.as_str().into()),
                ("author", p.author.as_str().into()),
                ("commands", uint(p.entries.len() as u64)),
                ("state", if ctx.state.in_review.contains(&p.id) { "awaiting_review" } else { "open" }.into()),
            ])
        })
        .collect();
    let protected = reg.params().iter().filter(|p| param_protected(doc, &p.id)).count();
    let data = object(vec![
        (
            "document",
            object(vec![
                ("id", doc.meta.id.to_hex().as_str().into()),
                ("name", doc.meta.name.as_str().into()),
                ("template", doc.meta.template.as_str().into()),
                ("modified", uint(doc.meta.modified_ms)),
            ]),
        ),
        ("non_default_parameters", Value::Array(changed)),
        ("sculpt_layers", Value::Array(doc.sculpt_layers.iter().map(|l| sculpt_json(doc, l)).collect())),
        ("skin_layers", Value::Array(doc.skin_layers.iter().map(|l| skin_json(doc, l)).collect())),
        ("grooms", Value::Array(doc.grooms.iter().map(|g| groom_json(doc, g)).collect())),
        ("pose_joints", uint(doc.pose.len() as u64)),
        ("locks", Value::Array(locks)),
        ("pins", Value::Array(pins)),
        ("snapshots", Value::Array(snapshots)),
        (
            "protection",
            object(vec![
                ("protect_human_edits", doc.settings.protect_human_edits.into()),
                ("protected_parameters", uint(protected as u64)),
            ]),
        ),
        // No plausibility scorer exists yet (WP-25); outcomes carry scores once one is installed.
        ("plausibility", Value::Null),
        ("proposals", Value::Array(proposals)),
        ("last_event", uint(session.events().last_seq())),
        ("agent", ctx.agent.id.as_str().into()),
    ]);
    Ok(ToolOutput::data(data))
}

fn describe_parameters(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let reg = ctx.engine.registry();
    let doc = ctx.engine.session().document();
    let mut list: Vec<&Param> = match args::str(a, "query") {
        Some(q) if !q.trim().is_empty() => reg.search(q),
        _ => reg.params().iter().collect(),
    };
    if let Some(g) = args::str(a, "group") {
        list.retain(|p| p.group == g || p.group.strip_prefix(g).is_some_and(|r| r.starts_with('/')));
    }
    if let Some(r) = args::str(a, "region") {
        list.retain(|p| p.region.as_deref() == Some(r));
    }
    if let Some(t) = args::str(a, "tag") {
        list.retain(|p| p.tags.iter().any(|x| x == t));
    }
    if args::bool(a, "changed_only", false) {
        list.retain(|p| doc.params.get(&p.id).is_some_and(|v| !p.is_default(v)));
    }
    let total = list.len();
    let offset = args::u64(a, "offset", 0) as usize;
    let limit = args::u64(a, "limit", 50) as usize;
    let items: Vec<Value> = list.iter().skip(offset).take(limit).map(|p| param_state(doc, reg, p)).collect();
    Ok(ToolOutput::data(object(vec![("total", uint(total as u64)), ("parameters", Value::Array(items))])))
}

fn get_parameters(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let reg = ctx.engine.registry();
    let doc = ctx.engine.session().document();
    let ids = a.get("ids").and_then(Value::as_array).unwrap_or(&[]);
    let items: Vec<Value> =
        ids.iter().filter_map(Value::as_str).filter_map(|id| reg.get(id)).map(|p| param_state(doc, reg, p)).collect();
    Ok(ToolOutput::data(object(vec![("parameters", Value::Array(items))])))
}

fn list_layers(ctx: &mut ToolCtx<'_, '_>, _: &Value) -> Result<ToolOutput, ToolError> {
    let doc = ctx.engine.session().document();
    Ok(ToolOutput::data(object(vec![
        ("sculpt_layers", Value::Array(doc.sculpt_layers.iter().map(|l| sculpt_json(doc, l)).collect())),
        ("skin_layers", Value::Array(doc.skin_layers.iter().map(|l| skin_json(doc, l)).collect())),
        ("grooms", Value::Array(doc.grooms.iter().map(|g| groom_json(doc, g)).collect())),
    ])))
}

fn blob_size(ctx: &ToolCtx<'_, '_>, hash: humanart_core::hash::Hash128) -> Value {
    if hash == humanart_core::hash::Hash128::ZERO {
        return uint(0);
    }
    let Some(project) = ctx.engine.session().project() else { return Value::Null };
    let path = project.blobs(ctx.fs).path(hash);
    ctx.fs.metadata(&path).map_or(Value::Null, |m| uint(m.len))
}

fn region_vertices(model: &Model, region: Option<&str>) -> Option<u64> {
    let map = model.template().regions();
    let r = map.find(region?)?;
    Some(map.vertex_regions.iter().filter(|&&v| map.is_within(v, r)).count() as u64)
}

fn get_layer(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let id = args::req_id(a, "id")?;
    let doc = ctx.engine.session().document();
    let (mut v, blob, region) = if let Some(l) = doc.sculpt_layers.iter().find(|l| l.id == id) {
        (sculpt_json(doc, l), l.payload, l.region.clone())
    } else if let Some(l) = doc.skin_layers.iter().find(|l| l.id == id) {
        let blob = match &l.source {
            humanart_doc::document::SkinSource::Painted { texture } => *texture,
            humanart_doc::document::SkinSource::Captured { set } => *set,
            humanart_doc::document::SkinSource::Procedural { .. } => humanart_core::hash::Hash128::ZERO,
        };
        (skin_json(doc, l), blob, l.region.clone())
    } else if let Some(g) = doc.grooms.iter().find(|g| g.id == id) {
        (groom_json(doc, g), g.guides, None)
    } else {
        return Err(ToolError::new("not_found", format!("no layer or groom {id}")));
    };
    let size = blob_size(ctx, blob);
    let verts = region_vertices(ctx.engine.model(), region.as_deref()).map_or(Value::Null, uint);
    if let Value::Object(o) = &mut v {
        o.insert("stats", object(vec![("blob_bytes", size), ("region_vertices", verts)]));
    }
    Ok(ToolOutput::data(v))
}

fn list_regions(ctx: &mut ToolCtx<'_, '_>, _: &Value) -> Result<ToolOutput, ToolError> {
    let model = ctx.engine.model();
    let doc = ctx.engine.session().document();
    let map = model.template().regions();
    let counts = map.vertex_counts();
    let items: Vec<Value> = map
        .regions
        .iter()
        .enumerate()
        .map(|(i, r)| {
            object(vec![
                ("id", uint(i as u64)),
                ("name", r.name.as_str().into()),
                (
                    "parent",
                    r.parent
                        .and_then(|p| map.regions.get(usize::from(p)))
                        .map_or(Value::Null, |p| p.name.as_str().into()),
                ),
                ("side", format!("{:?}", r.side).to_ascii_lowercase().into()),
                ("vertices", uint(u64::from(counts.get(i).copied().unwrap_or(0)))),
                ("locked", region_locked(doc, &r.name).into()),
            ])
        })
        .collect();
    Ok(ToolOutput::data(object(vec![("regions", Value::Array(items))])))
}

fn get_region_stats(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let name = args::str(a, "region").unwrap_or("");
    let model = ctx.engine.model();
    let map = model.template().regions();
    let r = map.find(name).ok_or_else(|| ToolError::new("not_found", format!("no region {name}")))?;
    let inside: Vec<bool> = map.vertex_regions.iter().map(|&v| map.is_within(v, r)).collect();
    let positions = model.positions();
    let normals = model.normals();
    let tris = model.triangles();
    let (mut lo, mut hi, mut sum, mut n) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN), Vec3::ZERO, 0u64);
    for (p, _) in positions.iter().zip(&inside).filter(|(_, i)| **i) {
        lo = lo.min(*p);
        hi = hi.max(*p);
        sum += *p;
        n += 1;
    }
    let (mut area, mut triangles) = (0.0_f64, 0u64);
    for t in tris.as_chunks::<3>().0 {
        let [a0, b0, c0] = t.map(|i| i as usize);
        if inside.get(a0).copied().unwrap_or(false)
            && let (Some(pa), Some(pb), Some(pc)) = (positions.get(a0), positions.get(b0), positions.get(c0))
        {
            area += f64::from((*pb - *pa).cross(*pc - *pa).length()) * 0.5;
            triangles += 1;
        }
    }
    let vec = |v: Vec3| Value::Array(vec![num(f64::from(v.x)), num(f64::from(v.y)), num(f64::from(v.z))]);
    let geometry = object(vec![
        ("vertices", uint(n)),
        ("triangles", uint(triangles)),
        ("area_m2", num(area)),
        ("bounds_min", if n > 0 { vec(lo) } else { Value::Null }),
        ("bounds_max", if n > 0 { vec(hi) } else { Value::Null }),
        ("centroid", if n > 0 { vec(sum * (1.0 / n as f32)) } else { Value::Null }),
    ]);
    let wanted = a.get("render_id").and_then(Value::as_u64);
    let record = match wanted {
        Some(id) => ctx.state.records.iter().find(|r| r.id == id),
        None => ctx.state.records.last(),
    };
    let colour = match record {
        None => Value::Null,
        Some(rec) => {
            let img = &rec.image;
            let cam = img.camera;
            let mut samples: Vec<[f64; 3]> = Vec::new();
            for ((p, nrm), _) in positions.iter().zip(normals).zip(&inside).filter(|(_, i)| **i) {
                if nrm.dot(cam.position - *p) <= 0.0 {
                    continue;
                }
                let Some([x, y]) = project(&cam, *p, img.width, img.height) else { continue };
                if x < 0.0 || y < 0.0 {
                    continue;
                }
                if let Some(px) = img.pixel(x as u32, y as u32) {
                    samples.push(px.map(|c| srgb_to_linear(f64::from(c) / 255.0)));
                }
            }
            colour_stats(rec.id, &samples)
        }
    };
    Ok(ToolOutput::data(object(vec![
        ("region", name.into()),
        ("geometry", geometry),
        ("colour", colour),
        // Not available yet: texel density needs the UV layout metrics of WP-24, the plausibility
        // contribution the identity prior's data (WP-25).
        ("texel_density", Value::Null),
        ("plausibility_contribution", Value::Null),
    ])))
}

/// Linear mean and Lab mean / standard deviation of linear RGB samples. The samples are the
/// region's camera-facing vertices projected into the render (occlusion by other parts is not
/// tested: the renderer has no per-pixel region ids yet).
fn colour_stats(render: u64, samples: &[[f64; 3]]) -> Value {
    if samples.is_empty() {
        return object(vec![("render_id", uint(render)), ("samples", uint(0))]);
    }
    let n = samples.len() as f64;
    let mut mean = [0.0; 3];
    let labs: Vec<[f64; 3]> = samples.iter().map(|s| rgb_to_lab(*s)).collect();
    let mut lab_mean = [0.0; 3];
    for (s, l) in samples.iter().zip(&labs) {
        for k in 0..3 {
            mean[k] += s[k] / n;
            lab_mean[k] += l[k] / n;
        }
    }
    let mut var = [0.0; 3];
    for l in &labs {
        for k in 0..3 {
            var[k] += (l[k] - lab_mean[k]).powi(2) / n;
        }
    }
    let arr = |v: [f64; 3]| Value::Array(v.iter().map(|x| num(*x)).collect());
    object(vec![
        ("render_id", uint(render)),
        ("samples", uint(samples.len() as u64)),
        ("linear_mean", arr(mean)),
        ("lab_mean", arr(lab_mean)),
        ("lab_std", arr(var.map(f64::sqrt))),
    ])
}

/// The rest shape (current shape coefficients, no pose or expression).
pub(crate) fn rest_positions(model: &Model) -> Vec<Vec3> {
    model.template().deformed_positions(model.input().shape)
}

fn landmark_at(model: &Model, rest: &[Vec3], name: &str) -> Result<Vec3, ToolError> {
    model
        .template()
        .landmarks()
        .iter()
        .find(|l| l.name == name)
        .and_then(|l| l.position(rest))
        .ok_or_else(|| ToolError::new("not_found", format!("landmark {name} is not on the surface")))
}

fn names(v: &Value) -> Vec<&str> {
    v.as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default()
}

fn measure(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let model = ctx.engine.model();
    let rest = rest_positions(model);
    let dist = |pair: &[&str]| -> Result<f64, ToolError> {
        match pair {
            [x, y] => Ok(f64::from(landmark_at(model, &rest, x)?.distance(landmark_at(model, &rest, y)?))),
            _ => Err(ToolError::new("invalid_arguments", "a pair names two landmarks")),
        }
    };
    let mut distances = Vec::new();
    for pair in a.get("distances").and_then(Value::as_array).unwrap_or(&[]) {
        let p = names(pair);
        distances.push(object(vec![("landmarks", pair.clone()), ("metres", num(dist(&p)?))]));
    }
    let mut ratios = Vec::new();
    for r in a.get("ratios").and_then(Value::as_array).unwrap_or(&[]) {
        let top = dist(&names(r.get("numerator").unwrap_or(&Value::Null)))?;
        let bottom = dist(&names(r.get("denominator").unwrap_or(&Value::Null)))?;
        let ratio = if bottom > 0.0 { num(top / bottom) } else { Value::Null };
        ratios.push(object(vec![("ratio", ratio), ("numerator_m", num(top)), ("denominator_m", num(bottom))]));
    }
    let mut angles = Vec::new();
    for t in a.get("angles").and_then(Value::as_array).unwrap_or(&[]) {
        let [x, y, z] = names(t)[..] else {
            return Err(ToolError::new("invalid_arguments", "an angle names three landmarks"));
        };
        let (pa, pb, pc) =
            (landmark_at(model, &rest, x)?, landmark_at(model, &rest, y)?, landmark_at(model, &rest, z)?);
        let (u, v) = ((pa - pb).normalize_or(Vec3::X), (pc - pb).normalize_or(Vec3::X));
        angles.push(object(vec![
            ("landmarks", t.clone()),
            ("degrees", num(f64::from(u.dot(v).clamp(-1.0, 1.0).acos().to_degrees()))),
        ]));
    }
    let reg = ctx.engine.registry();
    let mut named = Vec::new();
    for id in names(a.get("named").unwrap_or(&Value::Null)) {
        let Some(p) = reg.get(id) else { continue };
        let Evaluation::Measurement { landmarks, axis } = &p.evaluation else {
            return Err(ToolError::new("invalid_arguments", format!("{id} is not a measurement")));
        };
        let (pa, pb) = (landmark_at(model, &rest, &landmarks[0])?, landmark_at(model, &rest, &landmarks[1])?);
        let d = pb - pa;
        let metres = match axis.map(humanart_doc::registry::Axis::name) {
            Some("x") => d.x.abs(),
            Some("y") => d.y.abs(),
            Some("z") => d.z.abs(),
            _ => d.length(),
        };
        named.push(object(vec![
            ("id", id.into()),
            ("metres", num(f64::from(metres))),
            ("unit_in_registry", p.unit.as_str().into()),
        ]));
    }
    Ok(ToolOutput::data(object(vec![
        ("distances", Value::Array(distances)),
        ("ratios", Value::Array(ratios)),
        ("angles", Value::Array(angles)),
        ("named", Value::Array(named)),
        ("shape", "rest".into()),
    ])))
}

fn render_view(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let requests = view_requests(ctx, a)?;
    if args::bool(a, "async", false) {
        let ids: Vec<u64> = requests
            .iter()
            .map(|_| {
                ctx.state.next_render += 1;
                ctx.state.next_render
            })
            .collect();
        let mut later = a.clone();
        if let Value::Object(o) = &mut later {
            o.remove("async");
        }
        ctx.state.pending.push(PendingRender { ids: ids.clone(), args: later });
        return Ok(ToolOutput::data(object(vec![
            ("state", "pending".into()),
            ("render_ids", Value::Array(ids.into_iter().map(uint).collect())),
        ])));
    }
    let ids: Vec<u64> = requests
        .iter()
        .map(|_| {
            ctx.state.next_render += 1;
            ctx.state.next_render
        })
        .collect();
    run_render(ctx, a, &ids)
}

/// The view requests of `render_view` arguments.
fn view_requests(ctx: &ToolCtx<'_, '_>, a: &Value) -> Result<Vec<(String, ViewRequest)>, ToolError> {
    let bad = |m: &str| ToolError::new("invalid_arguments", m);
    let defaults = ctx.state.view;
    let mut base = ViewRequest {
        rig: args::str(a, "rig").and_then(rig_by_name).unwrap_or(defaults.rig),
        quality: args::str(a, "quality").and_then(quality_by_name).unwrap_or(defaults.quality),
        wireframe: args::bool(a, "wireframe", false),
        width: args::u64(a, "width", 768) as u32,
        height: args::u64(a, "height", 768) as u32,
        frames: args::u64(a, "frames", 16) as u32,
        ..ViewRequest::default()
    };
    if let Some(v) = args::str(a, "view_mode") {
        base.view = view_by_name(v).ok_or_else(|| bad("unknown view_mode"))?;
    }
    let limits = ctx.render_limits;
    if base.width > limits.max_side || base.height > limits.max_side || base.frames > limits.max_frames {
        return Err(bad("the image or frame count exceeds the server's render limits"));
    }
    let (label, camera) = if let Some(e) = a.get("explicit_camera") {
        let p = args::vec3(e, "position").ok_or_else(|| bad("explicit_camera.position"))?;
        let t = args::vec3(e, "target").ok_or_else(|| bad("explicit_camera.target"))?;
        let up = args::vec3(e, "up").unwrap_or([0.0, 1.0, 0.0]);
        let v = |x: [f64; 3]| Vec3::new(x[0] as f32, x[1] as f32, x[2] as f32);
        let cam = Camera {
            position: v(p),
            target: v(t),
            up: v(up),
            focal_mm: args::f64(e, "focal_mm").unwrap_or(85.0) as f32,
            ..Camera::default()
        };
        cam.validate().map_err(|e| ToolError::new("invalid_arguments", e.to_string()))?;
        ("explicit".to_owned(), CameraSpec::Explicit(cam))
    } else {
        match (args::str(a, "camera"), defaults.preset, defaults.camera) {
            (Some(name), _, _) => (name.to_owned(), CameraSpec::preset(name).ok_or_else(|| bad("unknown camera"))?),
            (None, _, Some(cam)) => ("explicit".to_owned(), CameraSpec::Explicit(cam)),
            (None, Some(name), None) => {
                (name.to_owned(), CameraSpec::preset(name).ok_or_else(|| bad("unknown camera"))?)
            }
            (None, None, None) => ("head".to_owned(), CameraSpec::preset("head").ok_or_else(|| bad("unknown camera"))?),
        }
    };
    base.camera = camera;
    let mut out = Vec::new();
    if let Some(views) = a.get("views").and_then(Value::as_array) {
        for name in views.iter().filter_map(Value::as_str) {
            let camera = CameraSpec::preset(name).ok_or_else(|| bad("unknown camera in views"))?;
            out.push((name.to_owned(), ViewRequest { camera, ..base }));
        }
    }
    let turns = args::u64(a, "turntable", 0);
    for i in 0..turns {
        let deg = 360.0 * i as f32 / turns as f32;
        out.push((format!("{label}@{deg:.1}"), ViewRequest { turn_degrees: deg, ..base }));
    }
    if out.is_empty() {
        out.push((label, base));
    }
    if out.len() > limits.max_views {
        return Err(bad("too many views in one request"));
    }
    Ok(out)
}

/// Renders the views of `a` under the reserved `ids`, writes the PNGs and remembers the renders.
pub(crate) fn run_render(ctx: &mut ToolCtx<'_, '_>, a: &Value, ids: &[u64]) -> Result<ToolOutput, ToolError> {
    let requests = view_requests(ctx, a)?;
    let crop = args::str(a, "region");
    let inline = args::bool(a, "inline", true);
    let mut metas = Vec::new();
    let mut images = Vec::new();
    for ((label, req), &id) in requests.iter().zip(ids) {
        let model = ctx.engine.model_mut();
        let mut image = ctx.state.renders.render(model, req).map_err(|e| ToolError::new("render", e.to_string()))?;
        let mut crop_rect = Value::Null;
        if let Some(region) = crop
            && let Some(rect) = region_rect(ctx.engine.model(), region, &image)
        {
            crop_rect = Value::Array(rect.iter().map(|v| uint(u64::from(*v))).collect());
            image = image.crop(rect);
        }
        let png = encode_png_rgba8(image.width, image.height, &image.rgba)
            .map_err(|e| ToolError::new("render", e.to_string()))?;
        let path = ctx
            .sandbox
            .resolve_write(&format!("renders/render-{id:06}.png"))
            .map_err(|e| ToolError::new("sandbox", e.to_string()))?;
        std::fs::write(&path, &png).map_err(|e| ToolError::new("io", e.to_string()))?;
        let cam = image.camera;
        let v3 = |v: Vec3| Value::Array(vec![num(f64::from(v.x)), num(f64::from(v.y)), num(f64::from(v.z))]);
        let meta = object(vec![
            ("id", uint(id)),
            ("uri", format!("humanart://renders/{id}").into()),
            ("file", ctx.sandbox.display(&path).into()),
            ("width", uint(u64::from(image.width))),
            ("height", uint(u64::from(image.height))),
            ("view", label.as_str().into()),
            (
                "camera",
                object(vec![
                    ("position", v3(cam.position)),
                    ("target", v3(cam.target)),
                    ("focal_mm", num(f64::from(cam.focal_mm))),
                ]),
            ),
            ("rig", rig_name(req.rig).into()),
            ("quality", quality_name(req.quality).into()),
            ("view_mode", req.view.name().into()),
            ("frames", uint(u64::from(req.frames))),
            ("crop", crop_rect),
            ("adapter", ctx.state.renders.adapter().into()),
            ("document_event", uint(ctx.engine.session().events().last_seq())),
        ]);
        if inline {
            images.push(png.clone());
        }
        metas.push(meta.clone());
        ctx.state.records.push(RenderRecord { id, file: ctx.sandbox.display(&path), png, image, meta });
        let keep = ctx.render_limits.keep.max(1);
        if ctx.state.records.len() > keep {
            let extra = ctx.state.records.len() - keep;
            ctx.state.records.drain(..extra);
        }
    }
    Ok(ToolOutput { data: object(vec![("renders", Value::Array(metas))]), images })
}

/// The pixel rectangle of a region's projected vertices, padded by 8 %.
fn region_rect(model: &Model, region: &str, image: &super::super::render::Image) -> Option<[u32; 4]> {
    let map = model.template().regions();
    let r = map.find(region)?;
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for (p, &v) in model.positions().iter().zip(&map.vertex_regions) {
        if !map.is_within(v, r) {
            continue;
        }
        if let Some([x, y]) = project(&image.camera, *p, image.width, image.height) {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    if x0 > x1 {
        return None;
    }
    let pad = 0.08 * (x1 - x0).max(y1 - y0) + 2.0;
    let clamp = |v: f32, max: u32| v.clamp(0.0, max as f32) as u32;
    let rect = [
        clamp(x0 - pad, image.width),
        clamp(y0 - pad, image.height),
        clamp(x1 + pad, image.width),
        clamp(y1 + pad, image.height),
    ];
    (rect[2] > rect[0] && rect[3] > rect[1]).then_some(rect)
}

fn get_events(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let since = args::u64(a, "since", 0);
    let limit = args::u64(a, "limit", 200) as usize;
    let log = ctx.engine.session().events();
    let (events, gap) = match log.since(since) {
        Since::Events(list) => (list.iter().take(limit).map(|e| e.to_json()).collect(), false),
        Since::Gap { .. } => (Vec::new(), true),
    };
    Ok(ToolOutput::data(object(vec![
        ("events", Value::Array(events)),
        ("last_seq", uint(log.last_seq())),
        ("gap", gap.into()),
    ])))
}

fn get_history(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let limit = args::u64(a, "limit", 20) as usize;
    let h = ctx.engine.session().history();
    let entries = h.entries();
    let start = entries.len().saturating_sub(limit);
    let items: Vec<Value> = entries[start..]
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let mut v = e.to_json();
            if let Value::Object(o) = &mut v {
                o.insert("applied", (start + i) < h.cursor());
            }
            v
        })
        .collect();
    Ok(ToolOutput::data(object(vec![("entries", Value::Array(items)), ("cursor", uint(h.cursor() as u64))])))
}
