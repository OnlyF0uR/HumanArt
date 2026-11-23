//! Mutation tools (section 10.4): every one produces document commands by the agent (or, for
//! `set_lighting_rig` / `set_camera`, changes editor-only view defaults), and every one takes
//! `proposal_id`. Results are the command outcomes of `[CMD-06]`; a refused command is an error,
//! never a success.

// Every handler has the table's signature, so some return `Ok` only.
#![allow(clippy::unnecessary_wraps)]

use humanart_core::id::Id128;
use humanart_doc::command::{Action, Command, ObjectKind, Outcome, RangePolicy, StrokeKind};
use humanart_doc::document::{Document, LockTarget};
use humanart_doc::expression_drive::ExpressionDrive;
use humanart_doc::history::Scope;
use humanart_doc::registry::{Evaluation, ParamValue, decode_value};
use humanart_io::json::{Object, Value};
use humanart_math::Vec3;
use humanart_model::eval::landmarks::{LandmarkGoal, LandmarkSolver, SolveRequest, SolveSettings};
use humanart_model::expression::presets::Presets;
use humanart_render::camera::Camera;

use super::super::render::{CAMERA_PRESETS, RIG_NAMES, rig_by_name, rig_name};
use super::super::schema::{Field, ParamFilter, Ty};
use super::observe::param_protected;
use super::{
    Access, PROPOSAL, REASON, ToolCtx, ToolError, ToolOutput, ToolSpec, args, ensure_agent, num, object, submit, uint,
};
use crate::engine::Binding;

const POLICY: Field =
    Field::opt("clamp", Ty::Bool, "Clamp values outside a parameter's range instead of refusing them.");

const fn values_fields(filter: ParamFilter) -> [Field; 5] {
    [
        Field::req("values", Ty::ParamValues(filter), "Parameter id to value."),
        Field::opt("relative", Ty::Bool, "Add the numbers to the current values."),
        POLICY,
        REASON,
        PROPOSAL,
    ]
}

const SET_ALL: [Field; 5] = values_fields(ParamFilter::All);
const SET_EYES: [Field; 5] = values_fields(ParamFilter::Groups(&["eyes"]));
const SET_MOUTH: [Field; 5] = values_fields(ParamFilter::Groups(&["mouth"]));
const SET_SKIN: [Field; 5] = values_fields(ParamFilter::Groups(&["skin"]));
const SET_EXPRESSION: [Field; 5] = values_fields(ParamFilter::Groups(&["expression"]));

const APPLY_EXPRESSION: [Field; 4] = [
    Field::opt("preset_id", Ty::Id, "The expression preset's id."),
    Field::opt("preset_name", Ty::Str { max: 128 }, "Or its name."),
    REASON,
    PROPOSAL,
];
const POSE: [Field; 4] = [
    Field::opt("joints", Ty::Map(&Ty::Quat, 256), "Joint name to rotation quaternion (x, y, z, w), normalised here."),
    Field::opt("reset", Ty::List(&Ty::Joint, 256), "Joints put back to rest."),
    REASON,
    PROPOSAL,
];
const MOVE_LANDMARK: [Field; 6] = [
    Field::req("landmark", Ty::Landmark, "The landmark to move."),
    Field::req("position", Ty::Vec3, "Target in the rest pose, metres (or an offset with relative)."),
    Field::opt("relative", Ty::Bool, "position is an offset from the current place."),
    POLICY,
    REASON,
    PROPOSAL,
];
const TARGET: [Field; 2] = [
    Field::req("landmark", Ty::Landmark, "The landmark."),
    Field::req("position", Ty::Vec3, "Rest-pose target, metres."),
];
const TARGETS: [Field; 4] = [
    Field::req("targets", Ty::List(&Ty::Record(&TARGET), 32), "Landmark targets solved together (pins hold)."),
    POLICY,
    REASON,
    PROPOSAL,
];
const LAYER_KINDS: [&str; 3] = ["sculpt", "skin", "groom"];
const ADD_SCULPT: [Field; 6] = [
    Field::req("name", Ty::Str { max: 128 }, "Layer name."),
    Field::opt("region", Ty::Region, "Region mask (none: everywhere)."),
    Field::opt("opacity", Ty::Num { min: 0.0, max: 1.0 }, "Opacity (default 1)."),
    Field::opt("payload", Ty::Id, "A delta blob already in the project's blob store (default: empty)."),
    REASON,
    PROPOSAL,
];
const CHANNELS: [&str; 15] = [
    "albedo_tint",
    "albedo_multiply",
    "specular",
    "roughness",
    "normal_detail",
    "height",
    "translucency",
    "sss_color",
    "sss_radius",
    "freckle_mask",
    "mole_mask",
    "blemish_mask",
    "redness",
    "cavity",
    "oiliness",
];
const ADD_SKIN: [Field; 11] = [
    Field::req("name", Ty::Str { max: 128 }, "Layer name."),
    Field::req("channels", Ty::List(&Ty::Choice(&CHANNELS), 15), "Channels it writes."),
    Field::opt("blend", Ty::Choice(&["mix", "multiply", "add", "overlay"]), "Blend mode (default mix)."),
    Field::opt("opacity", Ty::Num { min: 0.0, max: 1.0 }, "Opacity (default 1)."),
    Field::opt("region", Ty::Region, "Region mask (none: everywhere)."),
    Field::opt("generator", Ty::Str { max: 64 }, "Procedural generator name (freckles, moles, veins ...)."),
    Field::opt("params", Ty::Map(&Ty::Num { min: -1e9, max: 1e9 }, 64), "Generator parameters."),
    Field::opt("seed", Ty::Int { min: 0, max: i64::MAX }, "Generator seed."),
    Field::opt("texture", Ty::Id, "Or: a painted texture blob already in the project's blob store."),
    REASON,
    PROPOSAL,
];
const MODIFY_LAYER: [Field; 5] = [
    Field::req("kind", Ty::Choice(&LAYER_KINDS), "Which list."),
    Field::req("id", Ty::Id, "The layer or groom."),
    Field::req(
        "fields",
        Ty::Fields,
        "Fields to change, in the document's text form (name, opacity, region, visible ...).",
    ),
    REASON,
    PROPOSAL,
];
const DELETE_LAYER: [Field; 4] = [
    Field::req("kind", Ty::Choice(&LAYER_KINDS), "Which list."),
    Field::req("id", Ty::Id, "The layer or groom."),
    REASON,
    PROPOSAL,
];
const REORDER: [Field; 4] = [
    Field::req("kind", Ty::Choice(&LAYER_KINDS), "Which list."),
    Field::req("order", Ty::List(&Ty::Id, 1024), "Every id of the list once, bottom first."),
    REASON,
    PROPOSAL,
];
const LOCK_LAYER: [Field; 5] = [
    Field::req("kind", Ty::Choice(&LAYER_KINDS), "Which list."),
    Field::req("id", Ty::Id, "The layer or groom."),
    Field::req("locked", Ty::Bool, "Lock or unlock."),
    REASON,
    PROPOSAL,
];
const GROOM_CREATE: [Field; 7] = [
    Field::req("name", Ty::Str { max: 128 }, "Groom name."),
    Field::req("kind", Ty::Choice(&["scalp", "eyebrows", "eyelashes", "facial", "body"]), "What it covers."),
    Field::opt("params", Ty::Map(&Ty::Num { min: -1e9, max: 1e9 }, 128), "Groom parameters by name."),
    Field::opt("seed", Ty::Int { min: 0, max: i64::MAX }, "Random seed."),
    Field::opt("guides", Ty::Id, "A guide blob already in the project's blob store (default: none)."),
    REASON,
    PROPOSAL,
];
const GROOM_MODIFY: [Field; 4] = [
    Field::req("id", Ty::Id, "The groom."),
    Field::req("fields", Ty::Fields, "Fields to change (name, params, seed, visible, maps)."),
    REASON,
    PROPOSAL,
];
const GROOM_GUIDES: [Field; 5] = [
    Field::req("id", Ty::Id, "The groom."),
    Field::req("guides", Ty::Id, "The new guide blob (in the project's blob store)."),
    Field::opt("points", Ty::Int { min: 0, max: i64::MAX }, "Guide points touched (reported)."),
    REASON,
    PROPOSAL,
];
const RIG: [Field; 1] = [Field::req("rig", Ty::Choice(&RIG_NAMES), "The rig renders use by default.")];
const EXPLICIT: [Field; 4] = [
    Field::req("position", Ty::Vec3, "Eye position, metres."),
    Field::req("target", Ty::Vec3, "Point looked at."),
    Field::opt("up", Ty::Vec3, "Up (default +Y)."),
    Field::opt("focal_mm", Ty::Num { min: 14.0, max: 300.0 }, "Focal length (default 85)."),
];
const CAMERA: [Field; 2] = [
    Field::opt("camera", Ty::Choice(&CAMERA_PRESETS), "A preset renders use by default."),
    Field::opt("explicit_camera", Ty::Record(&EXPLICIT), "Or an explicit camera."),
];
const REGION: [Field; 3] = [Field::req("region", Ty::Region, "The region."), REASON, PROPOSAL];
const LOCK_PARAM: [Field; 4] = [
    Field::req("id", Ty::Param, "The parameter."),
    Field::opt("locked", Ty::Bool, "Lock (default) or unlock."),
    REASON,
    PROPOSAL,
];
const PIN: [Field; 4] = [
    Field::req("landmark", Ty::Landmark, "The landmark."),
    Field::opt("position", Ty::Vec3, "Where it stays (default: where it is now, rest pose)."),
    REASON,
    PROPOSAL,
];
const UNPIN: [Field; 3] = [Field::req("landmark", Ty::Landmark, "The landmark."), REASON, PROPOSAL];
const SNAPSHOT: [Field; 3] = [Field::req("name", Ty::Str { max: 128 }, "Snapshot name."), REASON, PROPOSAL];
const RESTORE: [Field; 3] = [Field::req("id", Ty::Id, "The snapshot."), REASON, PROPOSAL];
const OPEN_PROPOSAL: [Field; 1] = [Field::req("name", Ty::Str { max: 128 }, "What the proposal is about.")];
const PROPOSAL_ID: [Field; 1] = [Field::req("proposal_id", Ty::Id, "The proposal.")];
const UNDO: [Field; 2] = [
    Field::opt(
        "scope",
        Ty::Choice(&["mine", "any"]),
        "mine (default): the agent's last command; any: the last command.",
    ),
    PROPOSAL,
];
const IMPORT: [Field; 5] = [
    Field::req("file", Ty::Str { max: 1024 }, "Image in the exchange directory (or project:<path>)."),
    Field::req("name", Ty::Str { max: 128 }, "Reference name."),
    Field::opt("kind", Ty::Choice(&["photo", "landmarks", "color_target", "lighting"]), "Kind (default photo)."),
    REASON,
    PROPOSAL,
];

/// The largest reference file `import_reference` reads.
pub const MAX_REFERENCE_BYTES: u64 = 64 << 20;

macro_rules! tool {
    ($name:literal, $title:literal, $access:expr, $fields:expr, $handler:expr, $doc:literal) => {
        ToolSpec { name: $name, title: $title, description: $doc, fields: &$fields, access: $access, handler: $handler }
    };
}

/// The mutation tools.
pub static TOOLS: [ToolSpec; 35] = [
    tool!(
        "set_parameters",
        "Set parameters",
        Access::Mutate,
        SET_ALL,
        set_parameters,
        "Batch set of registry parameters, absolute or relative, with a clamping policy and a reason recorded in provenance."
    ),
    tool!(
        "set_eyes",
        "Set eye parameters",
        Access::Mutate,
        SET_EYES,
        set_parameters,
        "Sets parameters of the eyes groups (iris, sclera, cornea, gaze ...)."
    ),
    tool!(
        "set_mouth",
        "Set mouth parameters",
        Access::Mutate,
        SET_MOUTH,
        set_parameters,
        "Sets parameters of the mouth groups (teeth, gums, tongue, lips)."
    ),
    tool!(
        "set_skin_parameters",
        "Set skin parameters",
        Access::Mutate,
        SET_SKIN,
        set_parameters,
        "Sets parameters of the skin groups (pigment, pores, wrinkles, features ...)."
    ),
    tool!(
        "set_expression_weights",
        "Set expression weights",
        Access::Mutate,
        SET_EXPRESSION,
        set_parameters,
        "Sets expression controls."
    ),
    tool!(
        "apply_expression",
        "Apply an expression preset",
        Access::Mutate,
        APPLY_EXPRESSION,
        apply_expression,
        "Sets every expression control to a preset's weights: one stored in the document, or a built-in one by name (neutral, happiness.high, viseme.PP, ...)."
    ),
    tool!(
        "set_pose",
        "Set the pose",
        Access::Mutate,
        POSE,
        set_pose,
        "Rotates joints of the preview pose or puts them back to rest."
    ),
    tool!(
        "move_landmark",
        "Move a landmark",
        Access::Mutate,
        MOVE_LANDMARK,
        move_landmark,
        "Solves for the smallest identity change that moves the landmark to the target while pinned landmarks stay (LM-01); applied as a parameter change."
    ),
    tool!(
        "set_landmark_targets",
        "Set landmark targets",
        Access::Mutate,
        TARGETS,
        set_landmark_targets,
        "Moves several landmarks at once with the landmark solver."
    ),
    tool!(
        "add_sculpt_layer",
        "Add a sculpt layer",
        Access::Mutate,
        ADD_SCULPT,
        add_sculpt_layer,
        "Adds a sculpt layer, empty or from a stored delta blob. Described operations (smooth, inflate, move) need the sculpt tools of WP-20 and are not available."
    ),
    tool!(
        "add_skin_layer",
        "Add a skin layer",
        Access::Mutate,
        ADD_SKIN,
        add_skin_layer,
        "Adds a skin layer from a procedural generator with parameters, or from a stored painted texture."
    ),
    tool!(
        "modify_layer",
        "Modify a layer",
        Access::Mutate,
        MODIFY_LAYER,
        modify_layer,
        "Changes fields of a sculpt layer, skin layer or groom."
    ),
    tool!(
        "delete_layer",
        "Delete a layer",
        Access::Mutate,
        DELETE_LAYER,
        delete_layer,
        "Deletes a sculpt layer, skin layer or groom."
    ),
    tool!("reorder_layers", "Reorder layers", Access::Mutate, REORDER, reorder_layers, "Reorders a layer list."),
    tool!(
        "lock_layer",
        "Lock a layer",
        Access::Mutate,
        LOCK_LAYER,
        lock_layer,
        "Locks or unlocks a layer or groom (LOCK-01)."
    ),
    tool!(
        "groom_create",
        "Create a groom",
        Access::Mutate,
        GROOM_CREATE,
        groom_create,
        "Creates a groom with parameters and an optional stored guide blob."
    ),
    tool!("groom_modify", "Modify a groom", Access::Mutate, GROOM_MODIFY, groom_modify, "Changes groom fields."),
    tool!(
        "groom_set_guides",
        "Set groom guides",
        Access::Mutate,
        GROOM_GUIDES,
        groom_set_guides,
        "Replaces a groom's guide curves with a stored guide blob (a guide edit command)."
    ),
    tool!(
        "set_lighting_rig",
        "Set the lighting rig",
        Access::Editor,
        RIG,
        set_lighting_rig,
        "The rig renders use by default (editor-only; not character provenance)."
    ),
    tool!(
        "set_camera",
        "Set the camera",
        Access::Editor,
        CAMERA,
        set_camera,
        "The camera renders use by default (editor-only; not character provenance)."
    ),
    tool!(
        "lock_region",
        "Lock a region",
        Access::Mutate,
        REGION,
        lock_region,
        "Locks a region against shape, sculpt and skin changes."
    ),
    tool!("unlock_region", "Unlock a region", Access::Mutate, REGION, unlock_region, "Removes a region lock."),
    tool!(
        "lock_parameter",
        "Lock a parameter",
        Access::Mutate,
        LOCK_PARAM,
        lock_parameter,
        "Locks or unlocks a parameter."
    ),
    tool!(
        "pin_landmark",
        "Pin a landmark",
        Access::Mutate,
        PIN,
        pin_landmark,
        "Pins a landmark at a rest-pose position (LOCK-03)."
    ),
    tool!("unpin_landmark", "Unpin a landmark", Access::Mutate, UNPIN, unpin_landmark, "Removes a landmark pin."),
    tool!(
        "create_snapshot",
        "Create a snapshot",
        Access::Mutate,
        SNAPSHOT,
        create_snapshot,
        "Stores the character state as a named snapshot."
    ),
    tool!(
        "restore_snapshot",
        "Restore a snapshot",
        Access::Mutate,
        RESTORE,
        restore_snapshot,
        "Restores the character state of a snapshot."
    ),
    tool!(
        "open_proposal",
        "Open a proposal",
        Access::Mutate,
        OPEN_PROPOSAL,
        open_proposal,
        "Opens a proposal: a branch where the agent's commands apply for the human to review (PROP-01)."
    ),
    tool!(
        "commit_proposal",
        "Commit a proposal",
        Access::Mutate,
        PROPOSAL_ID,
        commit_proposal,
        "Hands the proposal to the human for review; merges it at once only when the project's policy allows."
    ),
    tool!(
        "discard_proposal",
        "Discard a proposal",
        Access::Mutate,
        PROPOSAL_ID,
        discard_proposal,
        "Discards one of the agent's proposals."
    ),
    tool!(
        "describe_proposal",
        "Describe a proposal",
        Access::Read,
        PROPOSAL_ID,
        describe_proposal,
        "A proposal's commands, changes, per-region change weights, conflicts with the main document and review state."
    ),
    tool!(
        "undo",
        "Undo",
        Access::Mutate,
        UNDO,
        undo,
        "Undoes the agent's last command (or the last command) in the main document or a proposal (CMD-04)."
    ),
    tool!("redo", "Redo", Access::Mutate, UNDO, redo, "Redoes the next command of the main document."),
    tool!(
        "import_reference",
        "Import a reference",
        Access::Mutate,
        IMPORT,
        import_reference,
        "Imports a reference image from the exchange directory into the project's blob store and the document."
    ),
    tool!(
        "get_policy",
        "Agent policy",
        Access::Read,
        [],
        get_policy,
        "What the agent may do without a proposal, and whether its proposals need review."
    ),
];

fn command(ctx: &ToolCtx<'_, '_>, a: &Value, action: Action) -> Command {
    let mut cmd = Command::new(&ctx.agent.id, action);
    if let Some(r) = args::str(a, "reason") {
        r.clone_into(&mut cmd.label);
    }
    cmd
}

fn run(ctx: &mut ToolCtx<'_, '_>, a: &Value, action: Action) -> Result<ToolOutput, ToolError> {
    let proposal = args::id(a, "proposal_id")?;
    let cmd = command(ctx, a, action);
    Ok(ToolOutput::data(submit(ctx, proposal, cmd)?))
}

fn policy(a: &Value) -> RangePolicy {
    if args::bool(a, "clamp", false) { RangePolicy::Clamp } else { RangePolicy::Reject }
}

fn bad(m: impl Into<String>) -> ToolError {
    ToolError::new("invalid_arguments", m)
}

/// The document a command with `proposal` goes to.
fn target_doc<'d>(ctx: &'d ToolCtx<'_, '_>, proposal: Option<Id128>) -> Result<&'d Document, ToolError> {
    match proposal {
        None => Ok(ctx.engine.session().document()),
        Some(p) => ctx
            .engine
            .session()
            .proposal(p)
            .map(|p| &p.doc)
            .ok_or_else(|| ToolError::new("unknown_proposal", format!("no open proposal {p}"))),
    }
}

fn set_parameters(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let reg = ctx.engine.registry();
    let o = a.get("values").and_then(Value::as_object).ok_or_else(|| bad("values"))?;
    let mut values = Vec::with_capacity(o.len());
    for (id, v) in o.entries() {
        let p = reg.get(id).ok_or_else(|| bad(format!("unknown parameter {id}")))?;
        let value = decode_value(p.kind, p.dimension, v).ok_or_else(|| bad(format!("{id}: wrong value shape")))?;
        values.push((id.clone(), value));
    }
    run(ctx, a, Action::SetParams { values, relative: args::bool(a, "relative", false), policy: policy(a) })
}

fn apply_expression(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let proposal = args::id(a, "proposal_id")?;
    let preset = match (args::id(a, "preset_id")?, args::str(a, "preset_name")) {
        (Some(id), _) => id,
        (None, Some(name)) => {
            let stored = target_doc(ctx, proposal)?.expression_presets.iter().find(|p| p.name == name).map(|p| p.id);
            match stored {
                Some(id) => id,
                // No preset of that name in the document: a built-in one (todo 1.8).
                None => return apply_builtin_expression(ctx, a, name),
            }
        }
        (None, None) => return Err(bad("give preset_id or preset_name")),
    };
    run(ctx, a, Action::ApplyExpressionPreset { preset })
}

/// Applies the built-in expression preset `name` (`data/template/expressions/presets.json`:
/// emotions and visemes) as one parameter change, as a stored preset applies: every expression
/// parameter to the preset's weight for its control, the others back to their defaults.
fn apply_builtin_expression(ctx: &mut ToolCtx<'_, '_>, a: &Value, name: &str) -> Result<ToolOutput, ToolError> {
    let presets =
        Presets::builtin().map_err(|e| ToolError::new("internal", format!("the built-in expression presets: {e}")))?;
    let preset =
        presets.find(name).ok_or_else(|| ToolError::new("not_found", format!("no expression preset named {name}")))?;
    let reg = ctx.engine.registry();
    let mut values: Vec<(String, ParamValue)> = reg
        .params()
        .iter()
        .filter(|p| matches!(p.evaluation, Evaluation::Expression(_)) && !p.read_only)
        .map(|p| (p.id.clone(), p.default.clone()))
        .collect();
    for (control, weight) in &preset.controls {
        let p = ExpressionDrive::param_of_control(reg, control).ok_or_else(|| {
            ToolError::new("internal", format!("preset {name}: no parameter drives the control {control}"))
        })?;
        let value = ParamValue::Scalar(f64::from(*weight));
        match values.iter_mut().find(|(id, _)| *id == p.id) {
            Some(slot) => slot.1 = value,
            None => values.push((p.id.clone(), value)),
        }
    }
    run(ctx, a, Action::SetParams { values, relative: false, policy: policy(a) })
}

fn set_pose(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let skeleton = ctx.engine.model().template().skeleton();
    let mut joints: Vec<(String, Option<[f64; 4]>)> = Vec::new();
    if let Some(map) = a.get("joints").and_then(Value::as_object) {
        for (name, q) in map.entries() {
            if skeleton.find(name).is_none() {
                return Err(bad(format!("unknown joint {name}")));
            }
            let q: Vec<f64> = q.as_array().map(|v| v.iter().filter_map(Value::as_f64).collect()).unwrap_or_default();
            let [x, y, z, w] = q[..] else { return Err(bad(format!("{name}: a quaternion has four numbers"))) };
            let len = (x * x + y * y + z * z + w * w).sqrt();
            if !(len > 1e-6 && len.is_finite()) {
                return Err(bad(format!("{name}: the quaternion has no length")));
            }
            joints.push((name.clone(), Some([x / len, y / len, z / len, w / len])));
        }
    }
    for name in a.get("reset").and_then(Value::as_array).unwrap_or(&[]).iter().filter_map(Value::as_str) {
        joints.push((name.to_owned(), None));
    }
    if joints.is_empty() {
        return Err(bad("give joints or reset"));
    }
    run(ctx, a, Action::SetPose { joints })
}

/// Moves landmarks with the landmark solver `[LM-01]` and applies the result as parameters.
fn solve_and_set(
    ctx: &mut ToolCtx<'_, '_>,
    a: &Value,
    goals: &[(String, [f64; 3], bool)],
) -> Result<ToolOutput, ToolError> {
    let proposal = args::id(a, "proposal_id")?;
    let reg = ctx.engine.registry();
    let model = ctx.engine.model();
    let template = model.template();
    let doc = target_doc(ctx, proposal)?;
    let components = template.deltas().len();
    // The shape coefficients of the target document and which components the agent may change:
    // only those a registry parameter drives, unlocked and not protected by the human.
    let mut weights = vec![0.0_f32; components];
    let mut locked = vec![true; components];
    let mut owner: Vec<Option<usize>> = vec![None; components];
    let mut bounds: Vec<Option<(f32, f32)>> = vec![None; components];
    for (i, (p, b)) in reg.params().iter().zip(model.bindings()).enumerate() {
        if let Binding::Shape(c) = *b {
            let c = c as usize;
            if c >= components {
                continue;
            }
            weights[c] = doc.params.get(&p.id).unwrap_or(&p.default).as_scalar().unwrap_or(0.0) as f32;
            let param_lock = doc.locks.iter().any(|l| matches!(&l.target, LockTarget::Parameter(x) if *x == p.id));
            let region_lock = p
                .region
                .as_deref()
                .is_some_and(|r| doc.locks.iter().any(|l| matches!(&l.target, LockTarget::Region(x) if x == r)));
            locked[c] = param_lock || region_lock || param_protected(doc, &p.id);
            owner[c] = Some(i);
            bounds[c] = p.range.map(|r| (r.min as f32, r.max as f32));
        }
    }
    let start = weights.clone();
    let rest = template.deformed_positions(&weights);
    let lm = template.landmarks();
    let index = |name: &str| lm.iter().position(|l| l.name == name);
    let mut targets = Vec::with_capacity(goals.len());
    for (name, pos, relative) in goals {
        let i = index(name).ok_or_else(|| bad(format!("unknown landmark {name}")))?;
        let here =
            lm.get(i).and_then(|l| l.position(&rest)).ok_or_else(|| bad(format!("{name} is not on the surface")))?;
        let p = Vec3::new(pos[0] as f32, pos[1] as f32, pos[2] as f32);
        targets.push(LandmarkGoal { landmark: i as u32, position: if *relative { here + p } else { p } });
    }
    let pins: Vec<LandmarkGoal> = doc
        .pins
        .iter()
        .filter(|p| !goals.iter().any(|g| g.0 == p.landmark))
        .filter_map(|p| {
            let i = index(&p.landmark)?;
            Some(LandmarkGoal {
                landmark: i as u32,
                position: Vec3::new(p.position[0] as f32, p.position[1] as f32, p.position[2] as f32),
            })
        })
        .collect();
    // The solver knows no parameter ranges: an active set keeps the result inside them. A component
    // whose solved value leaves its range is held at the bound and the others are solved again;
    // every round holds at least one more component, so this ends within `components` rounds.
    let mut solver = LandmarkSolver::new(template, SolveSettings::default());
    let mut posed = rest;
    let mut report;
    let mut rounds = 0;
    loop {
        report = solver
            .solve(&posed, &SolveRequest { targets: &targets, pins: &pins, locked: &locked, regularisation: &[] })
            .map_err(|e| ToolError::new("solver", e.to_string()))?;
        let mut held = false;
        for (c, d) in solver.delta().iter().enumerate() {
            let (Some(Some((lo, hi))), false) = (bounds.get(c), locked.get(c).copied().unwrap_or(true)) else {
                continue;
            };
            let v = weights[c] + d;
            if v < *lo || v > *hi {
                weights[c] = v.clamp(*lo, *hi);
                locked[c] = true;
                held = true;
            }
        }
        rounds += 1;
        if !held || rounds > components {
            break;
        }
        posed = template.deformed_positions(&weights);
    }
    let delta = solver.delta();
    let mut values = Vec::new();
    for (c, w0) in start.iter().enumerate() {
        let Some(Some(pi)) = owner.get(c) else { continue };
        let free = !locked.get(c).copied().unwrap_or(true);
        let v = if free { weights[c] + delta.get(c).copied().unwrap_or(0.0) } else { weights[c] };
        if (v - w0).abs() <= 1e-7 {
            continue;
        }
        let Some(p) = reg.params().get(*pi) else { continue };
        values.push((p.id.clone(), ParamValue::Scalar(f64::from(v))));
    }
    if values.is_empty() {
        return Err(ToolError::new(
            "no_effect",
            "no unlocked, unprotected parameter moves these landmarks (the template's shape components may not reach them)",
        ));
    }
    let cmd = command(ctx, a, Action::SetParams { values, relative: false, policy: policy(a) });
    let outcome = submit(ctx, proposal, cmd)?;
    Ok(ToolOutput::data(object(vec![
        ("outcome", outcome),
        (
            "solve",
            object(vec![
                ("max_residual_m", num(f64::from(report.max_residual))),
                ("max_pin_drift_m", num(f64::from(report.max_pin_drift))),
                ("exceeds_model_span", report.exceeds_span.into()),
                ("pins_held", uint(pins.len() as u64)),
            ]),
        ),
    ])))
}

fn move_landmark(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let name = args::str(a, "landmark").ok_or_else(|| bad("landmark"))?.to_owned();
    let pos = args::vec3(a, "position").ok_or_else(|| bad("position"))?;
    solve_and_set(ctx, a, &[(name, pos, args::bool(a, "relative", false))])
}

fn set_landmark_targets(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let mut goals = Vec::new();
    for t in a.get("targets").and_then(Value::as_array).unwrap_or(&[]) {
        let name = args::str(t, "landmark").ok_or_else(|| bad("landmark"))?;
        if goals.iter().any(|(n, _, _): &(String, [f64; 3], bool)| n == name) {
            return Err(bad(format!("{name} is given twice")));
        }
        goals.push((name.to_owned(), args::vec3(t, "position").ok_or_else(|| bad("position"))?, false));
    }
    if goals.is_empty() {
        return Err(bad("give at least one target"));
    }
    solve_and_set(ctx, a, &goals)
}

fn fields(members: Vec<(&str, Value)>) -> Object {
    let mut o = Object::with_capacity(members.len());
    for (k, v) in members {
        o.insert(k, v);
    }
    o
}

fn kind_of(a: &Value) -> Result<ObjectKind, ToolError> {
    match args::str(a, "kind") {
        Some("sculpt") => Ok(ObjectKind::SculptLayer),
        Some("skin") => Ok(ObjectKind::SkinLayer),
        Some("groom") => Ok(ObjectKind::Groom),
        _ => Err(bad("kind is sculpt, skin or groom")),
    }
}

fn opt_str(a: &Value, name: &str) -> Value {
    args::str(a, name).map_or(Value::Null, Value::from)
}

fn add_sculpt_layer(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let payload = args::hash(a, "payload")?.unwrap_or(humanart_core::hash::Hash128::ZERO);
    let f = fields(vec![
        ("name", opt_str(a, "name")),
        ("region", opt_str(a, "region")),
        ("opacity", num(args::f64(a, "opacity").unwrap_or(1.0))),
        (
            "payload",
            if payload == humanart_core::hash::Hash128::ZERO { Value::Null } else { payload.to_hex().as_str().into() },
        ),
    ]);
    run(ctx, a, Action::AddObject { kind: ObjectKind::SculptLayer, id: Id128::NIL, index: None, fields: f })
}

fn add_skin_layer(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let source = match (args::str(a, "generator"), args::hash(a, "texture")?) {
        (Some(g), None) => object(vec![
            ("kind", "procedural".into()),
            ("generator", g.into()),
            ("params", a.get("params").cloned().unwrap_or_else(|| Value::Object(Object::new()))),
            ("seed", uint(args::u64(a, "seed", 0))),
        ]),
        (None, Some(t)) => object(vec![("kind", "painted".into()), ("texture", t.to_hex().as_str().into())]),
        _ => return Err(bad("give either generator (procedural) or texture (painted)")),
    };
    let f = fields(vec![
        ("name", opt_str(a, "name")),
        ("channels", a.get("channels").cloned().unwrap_or(Value::Array(Vec::new()))),
        ("blend", args::str(a, "blend").unwrap_or("mix").into()),
        ("opacity", num(args::f64(a, "opacity").unwrap_or(1.0))),
        ("region", opt_str(a, "region")),
        ("source", source),
    ]);
    run(ctx, a, Action::AddObject { kind: ObjectKind::SkinLayer, id: Id128::NIL, index: None, fields: f })
}

fn modify_layer(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let kind = kind_of(a)?;
    let id = args::req_id(a, "id")?;
    let f = a.get("fields").and_then(Value::as_object).cloned().ok_or_else(|| bad("fields"))?;
    run(ctx, a, Action::ModifyObject { kind, id, fields: f })
}

fn delete_layer(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let (kind, id) = (kind_of(a)?, args::req_id(a, "id")?);
    run(ctx, a, Action::DeleteObject { kind, id })
}

fn reorder_layers(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let kind = kind_of(a)?;
    let mut order = Vec::new();
    for v in a.get("order").and_then(Value::as_array).unwrap_or(&[]) {
        order.push(v.as_str().and_then(|s| Id128::parse(s).ok()).ok_or_else(|| bad("order holds ids"))?);
    }
    run(ctx, a, Action::ReorderObjects { kind, order })
}

fn lock_layer(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let (kind, id) = (kind_of(a)?, args::req_id(a, "id")?);
    run(ctx, a, Action::SetObjectLock { kind, id, locked: args::bool(a, "locked", true) })
}

fn groom_create(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let guides = args::hash(a, "guides")?;
    let f = fields(vec![
        ("name", opt_str(a, "name")),
        ("kind", opt_str(a, "kind")),
        ("params", a.get("params").cloned().unwrap_or_else(|| Value::Object(Object::new()))),
        ("seed", uint(args::u64(a, "seed", 0))),
        ("guides", guides.map_or(Value::Null, |g| g.to_hex().as_str().into())),
    ]);
    run(ctx, a, Action::AddObject { kind: ObjectKind::Groom, id: Id128::NIL, index: None, fields: f })
}

fn groom_modify(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let id = args::req_id(a, "id")?;
    let f = a.get("fields").and_then(Value::as_object).cloned().ok_or_else(|| bad("fields"))?;
    run(ctx, a, Action::ModifyObject { kind: ObjectKind::Groom, id, fields: f })
}

fn groom_set_guides(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let target = args::req_id(a, "id")?;
    let blob = args::hash(a, "guides")?.ok_or_else(|| bad("guides"))?;
    let action = Action::Stroke {
        stroke: StrokeKind::Guides,
        target,
        blob,
        regions: Vec::new(),
        elements: args::u64(a, "points", 0),
    };
    run(ctx, a, action)
}

fn set_lighting_rig(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let rig = args::str(a, "rig").and_then(rig_by_name).ok_or_else(|| bad("unknown rig"))?;
    ctx.state.view.rig = rig;
    Ok(ToolOutput::data(object(vec![("rig", rig_name(rig).into())])))
}

fn set_camera(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    if let Some(e) = a.get("explicit_camera") {
        let v = |x: [f64; 3]| Vec3::new(x[0] as f32, x[1] as f32, x[2] as f32);
        let cam = Camera {
            position: v(args::vec3(e, "position").ok_or_else(|| bad("position"))?),
            target: v(args::vec3(e, "target").ok_or_else(|| bad("target"))?),
            up: v(args::vec3(e, "up").unwrap_or([0.0, 1.0, 0.0])),
            focal_mm: args::f64(e, "focal_mm").unwrap_or(85.0) as f32,
            ..Camera::default()
        };
        cam.validate().map_err(|e| bad(e.to_string()))?;
        ctx.state.view.camera = Some(cam);
        ctx.state.view.preset = None;
        return Ok(ToolOutput::data(object(vec![("camera", "explicit".into())])));
    }
    let name = args::str(a, "camera").ok_or_else(|| bad("give camera or explicit_camera"))?;
    let preset = CAMERA_PRESETS.iter().find(|p| **p == name).ok_or_else(|| bad("unknown camera"))?;
    ctx.state.view.preset = Some(preset);
    ctx.state.view.camera = None;
    Ok(ToolOutput::data(object(vec![("camera", (*preset).into())])))
}

fn lock_region(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let r = args::str(a, "region").ok_or_else(|| bad("region"))?.to_owned();
    run(ctx, a, Action::SetLock { target: LockTarget::Region(r), locked: true })
}

fn unlock_region(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let r = args::str(a, "region").ok_or_else(|| bad("region"))?.to_owned();
    run(ctx, a, Action::SetLock { target: LockTarget::Region(r), locked: false })
}

fn lock_parameter(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let id = args::str(a, "id").ok_or_else(|| bad("id"))?.to_owned();
    run(ctx, a, Action::SetLock { target: LockTarget::Parameter(id), locked: args::bool(a, "locked", true) })
}

fn pin_landmark(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let landmark = args::str(a, "landmark").ok_or_else(|| bad("landmark"))?.to_owned();
    let position = if let Some(p) = args::vec3(a, "position") {
        p
    } else {
        {
            let model = ctx.engine.model();
            let rest = super::observe::rest_positions(model);
            let p = model
                .template()
                .landmarks()
                .iter()
                .find(|l| l.name == landmark)
                .and_then(|l| l.position(&rest))
                .ok_or_else(|| bad("the landmark is not on the surface"))?;
            [f64::from(p.x), f64::from(p.y), f64::from(p.z)]
        }
    };
    run(ctx, a, Action::PinLandmark { landmark, position })
}

fn unpin_landmark(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let landmark = args::str(a, "landmark").ok_or_else(|| bad("landmark"))?.to_owned();
    run(ctx, a, Action::UnpinLandmark { landmark })
}

fn create_snapshot(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let name = args::str(a, "name").ok_or_else(|| bad("name"))?.to_owned();
    run(ctx, a, Action::CreateSnapshot { id: Id128::NIL, name, state: humanart_core::hash::Hash128::ZERO })
}

fn restore_snapshot(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let id = args::req_id(a, "id")?;
    run(ctx, a, Action::RestoreSnapshot { id })
}

/// Raises `proposal_state_changed` for a state the event log does not carry (awaiting review);
/// opened, accepted and discarded come from the session's events.
fn state_note(ctx: &mut ToolCtx<'_, '_>, proposal: Id128, state: &str) {
    ctx.notes.push((
        "notifications/humanart/proposal_state_changed",
        object(vec![("proposal", proposal.to_hex().as_str().into()), ("state", state.into())]),
    ));
}

fn open_proposal(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    ensure_agent(ctx)?;
    let name = args::str(a, "name").unwrap_or("proposal");
    let agent = ctx.agent.id.clone();
    let id = ctx.engine.session_mut().open_proposal(&agent, name)?;
    Ok(ToolOutput::data(object(vec![("proposal_id", id.to_hex().as_str().into()), ("state", "open".into())])))
}

fn commit_proposal(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let id = args::req_id(a, "proposal_id")?;
    let p = ctx
        .engine
        .session()
        .proposal(id)
        .ok_or_else(|| ToolError::new("unknown_proposal", format!("no open proposal {id}")))?;
    if p.author != ctx.agent.id {
        return Err(ToolError::new("not_proposal_author", "only the agent's own proposals can be committed"));
    }
    if ctx.policy.commit_without_review {
        let agent = ctx.agent.id.clone();
        let accepted = ctx.engine.session_mut().accept(id, &agent, None)?;
        ctx.engine.update().map_err(|e| ToolError::new("engine", e.to_string()))?;
        ctx.state.in_review.retain(|x| *x != id);
        return Ok(ToolOutput::data(object(vec![
            ("state", "accepted".into()),
            ("outcome", accepted.outcome.to_json()),
            ("dropped", Value::Array(accepted.dropped.iter().map(|d| Value::from(d.to_hex().as_str())).collect())),
        ])));
    }
    if !ctx.state.in_review.contains(&id) {
        ctx.state.in_review.push(id);
    }
    state_note(ctx, id, "awaiting_review");
    Ok(ToolOutput::data(object(vec![
        ("state", "awaiting_review".into()),
        (
            "message",
            "the human reviews the proposal in the editor; proposal_accepted or proposal_discarded follows".into(),
        ),
    ])))
}

fn discard_proposal(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let id = args::req_id(a, "proposal_id")?;
    let agent = ctx.agent.id.clone();
    ctx.engine.session_mut().discard(id, &agent)?;
    ctx.state.in_review.retain(|x| *x != id);
    Ok(ToolOutput::data(object(vec![("state", "discarded".into())])))
}

fn describe_proposal(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let id = args::req_id(a, "proposal_id")?;
    let reg = ctx.engine.registry();
    let session = ctx.engine.session();
    let p = session.proposal(id).ok_or_else(|| ToolError::new("unknown_proposal", format!("no open proposal {id}")))?;
    let doc_err = |e: humanart_doc::DocError| ToolError::new("invalid", e.to_string());
    let changes = p.changes(reg).map_err(doc_err)?;
    let summary = Outcome {
        command: id,
        author: p.author.clone(),
        kind: "proposal".into(),
        label: p.name.clone(),
        time_ms: p.created_ms,
        proposal: Some(id),
        changes,
        warnings: Vec::new(),
        elements: 0,
        plausibility: None,
        coalesced: false,
    };
    let all: Vec<usize> = (0..p.entries.len()).collect();
    let conflicts = p.conflicts(session.document(), &all).map_err(doc_err)?;
    let heat = p.region_heat(reg).map_err(doc_err)?;
    let commands: Vec<Value> = p
        .entries
        .iter()
        .map(|e| {
            object(vec![
                ("id", e.id.to_hex().as_str().into()),
                ("kind", e.kind.as_str().into()),
                ("label", e.label.as_str().into()),
            ])
        })
        .collect();
    Ok(ToolOutput::data(object(vec![
        ("proposal_id", id.to_hex().as_str().into()),
        ("name", p.name.as_str().into()),
        ("author", p.author.as_str().into()),
        ("state", if ctx.state.in_review.contains(&id) { "awaiting_review" } else { "open" }.into()),
        ("commands", Value::Array(commands)),
        ("changes", summary.to_json().get("changes").cloned().unwrap_or(Value::Null)),
        ("region_heat", Value::Object(heat.iter().map(|(r, w)| (r.as_str(), num(*w))).collect())),
        (
            "conflicts",
            Value::Array(
                conflicts
                    .iter()
                    .map(|(c, t)| {
                        object(vec![
                            ("command", c.to_hex().as_str().into()),
                            ("targets", Value::Array(t.iter().map(Value::from).collect())),
                        ])
                    })
                    .collect(),
            ),
        ),
    ])))
}

fn scope(ctx: &ToolCtx<'_, '_>, a: &Value) -> Scope {
    if args::str(a, "scope") == Some("any") { Scope::Any } else { Scope::Author(ctx.agent.id.clone()) }
}

fn undo(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let agent = ctx.agent.id.clone();
    let outcome = if let Some(p) = args::id(a, "proposal_id")? {
        ctx.engine.session_mut().undo_in(p, &agent)?
    } else {
        {
            if !ctx.policy.direct_edits {
                return Err(ToolError::new("policy", "this project lets the agent edit only inside proposals"));
            }
            let s = scope(ctx, a);
            ctx.engine.session_mut().undo(&agent, &s)?
        }
    };
    ctx.engine.update().map_err(|e| ToolError::new("engine", e.to_string()))?;
    Ok(ToolOutput::data(outcome.to_json()))
}

fn redo(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    if args::id(a, "proposal_id")?.is_some() {
        return Err(bad("proposals have no redo; repeat the command"));
    }
    if !ctx.policy.direct_edits {
        return Err(ToolError::new("policy", "this project lets the agent edit only inside proposals"));
    }
    let agent = ctx.agent.id.clone();
    let s = scope(ctx, a);
    let outcome = ctx.engine.session_mut().redo(&agent, &s)?;
    ctx.engine.update().map_err(|e| ToolError::new("engine", e.to_string()))?;
    Ok(ToolOutput::data(outcome.to_json()))
}

fn import_reference(ctx: &mut ToolCtx<'_, '_>, a: &Value) -> Result<ToolOutput, ToolError> {
    let file = args::str(a, "file").ok_or_else(|| bad("file"))?;
    let path = ctx.sandbox.resolve_read(file).map_err(|e| ToolError::new("sandbox", e.to_string()))?;
    let len = std::fs::metadata(&path).map_err(|e| ToolError::new("io", e.to_string()))?.len();
    if len > MAX_REFERENCE_BYTES {
        return Err(ToolError::new("too_large", format!("references are at most {MAX_REFERENCE_BYTES} bytes")));
    }
    let bytes = std::fs::read(&path).map_err(|e| ToolError::new("io", e.to_string()))?;
    let kind = args::str(a, "kind").unwrap_or("photo");
    if kind == "photo" && !(bytes.starts_with(b"\x89PNG\r\n\x1a\n") || bytes.starts_with(&[0xFF, 0xD8, 0xFF])) {
        return Err(bad("a photo reference is a PNG or JPEG file"));
    }
    // [DOC-13] References live under the project's `refs/` directory, named by their content hash
    // (so importing the same file twice stores it once), not in the character.
    let project = ctx.engine.session().project().ok_or_else(|| ToolError::new("no_project", "no project is open"))?;
    let hash = humanart_core::hash::hash128(&bytes);
    let ext = if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        "png"
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        "jpg"
    } else {
        "bin"
    };
    let name = format!("{}.{ext}", hash.to_hex());
    let dir = project.refs_dir();
    let target = format!("{dir}/{name}");
    if !ctx.fs.exists(&target) {
        ctx.fs.create_dir_all(&dir).map_err(|e| ToolError::new("storage", e.to_string()))?;
        ctx.fs.write_atomic(&target, &bytes).map_err(|e| ToolError::new("storage", e.to_string()))?;
    }
    let f = fields(vec![
        ("name", opt_str(a, "name")),
        ("kind", kind.into()),
        ("path", name.as_str().into()),
        ("hash", hash.to_hex().as_str().into()),
    ]);
    run(ctx, a, Action::AddObject { kind: ObjectKind::Reference, id: Id128::NIL, index: None, fields: f })
}

fn get_policy(ctx: &mut ToolCtx<'_, '_>, _: &Value) -> Result<ToolOutput, ToolError> {
    Ok(ToolOutput::data(object(vec![
        ("direct_edits", ctx.policy.direct_edits.into()),
        ("commit_without_review", ctx.policy.commit_without_review.into()),
        ("agent", ctx.agent.id.as_str().into()),
        ("exchange", ctx.sandbox.display(ctx.sandbox.exchange()).into()),
    ])))
}
