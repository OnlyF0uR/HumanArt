//! Integration tests of the agent server: every tool's schema and negative cases, the scripted
//! edit sessions over stdio and HTTP with a human editing at the same time, the security measures
//! over real sockets, pause / revoke, and headless-versus-editor render parity.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

use humanart_core::id::Id128;
use humanart_core::platform::{ManualClock, MemoryFs};
use humanart_doc::command::{Action, Command, ObjectKind, RangePolicy};
use humanart_doc::document::{Author, AuthorKind, Document, LockTarget};
use humanart_doc::registry::{Param, ParamKind, ParamValue};
use humanart_doc::session::{Session, SessionConfig};
use humanart_io::json::{Object, Value, parse, parse_str};

use super::client::{ClientError, EventStream, HttpClient, StdioClient, with_meta};
use super::jsonrpc::{Id, Incoming, RpcLimits};
use super::security::{Sandbox, Token};
use super::server::{Envelope, Gate, Inbound, Reply, Server, ServerConfig, Transport};
use super::tools::{self, AgentPolicy};
use super::transport::{HttpConfig, HttpServer, spawn_stdio};
use crate::engine::tests::{registry, template};
use crate::engine::{Engine, Model};

const HUMAN: &str = "human.test";
const AGENT: &str = "agent.test";
const LONG: Duration = Duration::from_secs(60);

struct World {
    fs: MemoryFs,
    clock: ManualClock,
}

fn world() -> World {
    let clock = ManualClock::new(1_700_000_000_000);
    clock.set_auto_advance(1_000_000);
    World { fs: MemoryFs::new(), clock }
}

fn engine(w: &World) -> Engine<'_> {
    let reg = registry();
    let me = Author { id: HUMAN.into(), kind: AuthorKind::Human, name: "Test".into() };
    let doc = Document::new(Id128::from_u128(9), "test", "generated", 1, 0, me);
    let session =
        Session::create("proj", doc, reg, &w.fs, &w.clock, SessionConfig::default(), Id128::from_u128(1)).unwrap();
    let mut e = Engine::new(session, reg, Model::new(Arc::new(template().clone()), reg));
    e.update().unwrap();
    e
}

fn exchange(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("humanart-mcp-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// A server for the agent, admitted into the engine's document by the human (as the host does).
fn server(tag: &str, policy: AgentPolicy, e: &mut Engine<'_>) -> Server {
    let mut cfg = ServerConfig::new(AGENT, "Test agent");
    cfg.policy = policy;
    let s = Server::new(cfg, Sandbox::new(&exchange(tag), None).unwrap(), 42);
    s.admit(e, HUMAN).unwrap();
    s.admit(e, HUMAN).unwrap();
    e.update().unwrap();
    s
}

fn json(text: &str) -> Value {
    parse_str(text).unwrap()
}

/// Sends one request straight into the server (as stdio would) and returns (status, message).
fn send(s: &mut Server, e: &mut Engine<'_>, w: &World, method: &str, params: Value) -> (u16, Value) {
    let (tx, rx) = channel();
    let incoming = Incoming::Request { id: Id::Number(1), method: method.into(), params };
    s.handle(
        e,
        &w.fs,
        Inbound::Message(Envelope {
            incoming,
            transport: Transport::Stdio,
            reply: tx,
            deadline: Instant::now() + LONG,
        }),
    );
    match rx.try_recv() {
        Ok(Reply::Message { body, status, .. }) => (status, parse(&body).unwrap()),
        other => panic!("no message for {method}: {other:?}"),
    }
}

/// Calls a tool directly and returns its result object.
fn call(s: &mut Server, e: &mut Engine<'_>, w: &World, name: &str, args: Value) -> Value {
    let mut p = Object::new();
    p.insert("name", name);
    p.insert("arguments", args);
    let (_, m) = send(s, e, w, "tools/call", with_meta(Value::Object(p)));
    m.get("result").cloned().unwrap_or_else(|| panic!("{name}: {m:?}"))
}

fn ok(r: &Value) -> &Value {
    assert_eq!(r.get("isError"), Some(&Value::Bool(false)), "{r:?}");
    r.get("structuredContent").unwrap()
}

fn code(r: &Value) -> String {
    assert_eq!(r.get("isError"), Some(&Value::Bool(true)), "expected an error: {r:?}");
    r.get("structuredContent")
        .and_then(|s| s.get("error"))
        .and_then(|e| e.get("code"))
        .and_then(Value::as_str)
        .unwrap()
        .to_owned()
}

fn human(e: &mut Engine<'_>, action: Action) {
    e.execute(Command::new(HUMAN, action)).unwrap();
    e.update().unwrap();
}

fn scalar(id: &str, v: f64) -> Action {
    Action::SetParams {
        values: vec![(id.to_owned(), ParamValue::Scalar(v))],
        relative: false,
        policy: RangePolicy::Reject,
    }
}

/// Scalar parameters for the tests: two free ones and one of a template region (with its region).
fn params() -> (&'static Param, &'static Param, &'static Param, String) {
    let reg = registry();
    let regions = template().regions();
    let usable = |p: &&Param| p.kind == ParamKind::Scalar && !p.read_only && p.range.is_some();
    let regional = reg
        .params()
        .iter()
        .filter(usable)
        .find(|p| p.region.as_deref().is_some_and(|r| regions.find(r).is_some()))
        .expect("a regional scalar parameter");
    let mut free = reg.params().iter().filter(usable).filter(|p| p.id != regional.id && p.region != regional.region);
    let a = free.next().unwrap();
    let b = free.find(|p| p.region != a.region || p.region.is_none()).unwrap();
    (a, b, regional, regional.region.clone().unwrap())
}

/// A value in the plausible span that is not the default.
fn mid(p: &Param) -> f64 {
    let r = p.range.unwrap();
    let v = f64::midpoint(r.soft_min, r.soft_max);
    if p.is_default(&ParamValue::Scalar(v)) { f64::midpoint(v, r.soft_max) } else { v }
}

fn obj(members: &[(&str, Value)]) -> Value {
    let mut o = Object::new();
    for (k, v) in members {
        o.insert(*k, v.clone());
    }
    Value::Object(o)
}

#[test]
fn ag_02_every_tool_has_a_schema_and_refuses_bad_arguments() {
    let w = world();
    let mut e = engine(&w);
    let mut s = server("schema", AgentPolicy::DEFAULT, &mut e);
    let (_, list) = send(&mut s, &mut e, &w, "tools/list", with_meta(Value::Null));
    let defs = list.get("result").and_then(|r| r.get("tools")).and_then(Value::as_array).unwrap();
    assert_eq!(defs.len(), tools::all().count());
    let mut names: Vec<&str> = defs.iter().map(|d| d.get("name").and_then(Value::as_str).unwrap()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), defs.len(), "tool names are unique");
    for d in defs {
        let name = d.get("name").and_then(Value::as_str).unwrap();
        let schema = d.get("inputSchema").unwrap();
        assert_eq!(schema.get("type").and_then(Value::as_str), Some("object"), "{name}");
        assert_eq!(schema.get("additionalProperties"), Some(&Value::Bool(false)), "{name}");
        assert!(d.get("description").and_then(Value::as_str).is_some_and(|t| t.len() > 10), "{name}");
        // Negative: an unknown member, and (where something is required) nothing at all.
        assert_eq!(
            code(&call(&mut s, &mut e, &w, name, json(r#"{"no_such_member":1}"#))),
            "invalid_arguments",
            "{name}"
        );
        let required = schema.get("required").and_then(Value::as_array).map_or(0, <[Value]>::len);
        if required > 0 {
            assert_eq!(code(&call(&mut s, &mut e, &w, name, json("{}"))), "invalid_arguments", "{name}");
        }
        // Negative: every typed member given a value of the wrong JSON type.
        if let Some(Value::Object(props)) = schema.get("properties") {
            for (member, sub) in props.entries() {
                let wrong = match sub.get("type").and_then(Value::as_str) {
                    Some("string" | "array" | "object" | "boolean") => Value::from(Number::from(7u64)),
                    Some("integer" | "number") => Value::from("seven"),
                    _ => continue,
                };
                let r = call(&mut s, &mut e, &w, name, obj(&[(member.as_str(), wrong)]));
                assert_eq!(code(&r), "invalid_arguments", "{name}.{member}");
            }
        }
    }
    let (status, m) = send(&mut s, &mut e, &w, "tools/call", with_meta(json(r#"{"name":"no_such_tool"}"#)));
    assert_eq!((status, m.get("error").and_then(|e| e.get("code")).and_then(Value::as_i64)), (400, Some(-32602)));
}

use humanart_io::json::Number;

#[test]
fn ac_wp23_1_every_tool_runs_through_the_session() {
    let w = world();
    let mut e = engine(&w);
    let mut s = server("tools", AgentPolicy { direct_edits: true, commit_without_review: false }, &mut e);
    let (a, b, regional, region) = params();
    let called: std::cell::RefCell<Vec<&str>> = std::cell::RefCell::new(Vec::new());
    let run = |s: &mut Server, e: &mut Engine<'_>, name: &'static str, args: Value| -> Value {
        called.borrow_mut().push(name);
        ok(&call(s, e, &w, name, args)).clone()
    };
    let lm = &template().landmarks()[0].name;
    let lm2 = &template().landmarks()[1].name;
    let lm3 = &template().landmarks()[2].name;
    let joint = &template().skeleton().joints[1].name;
    run(&mut s, &mut e, "describe_character", json("{}"));
    let found = run(&mut s, &mut e, "describe_parameters", obj(&[("query", a.display_name.as_str().into())]));
    assert!(found.get("total").and_then(Value::as_u64).unwrap() >= 1);
    run(&mut s, &mut e, "get_parameters", obj(&[("ids", Value::Array(vec![a.id.as_str().into()]))]));
    let mut values = Object::new();
    values.insert(a.id.as_str(), Value::from(Number::from(mid(a))));
    let out =
        run(&mut s, &mut e, "set_parameters", obj(&[("values", Value::Object(values)), ("reason", "test".into())]));
    assert_eq!(out.get("author").and_then(Value::as_str), Some(AGENT));
    assert_eq!(out.get("label").and_then(Value::as_str), Some("test"));
    let group_tools = [
        ("set_eyes", "eyes"),
        ("set_mouth", "mouth"),
        ("set_skin_parameters", "skin"),
        ("set_expression_weights", "expression"),
    ];
    for (tool, group) in group_tools {
        let p = registry()
            .params()
            .iter()
            .find(|p| {
                p.kind == ParamKind::Scalar
                    && !p.read_only
                    && p.range.is_some()
                    && (p.group == group || p.group.starts_with(&format!("{group}/")))
            })
            .unwrap();
        let mut v = Object::new();
        v.insert(p.id.as_str(), Value::from(Number::from(mid(p))));
        run(&mut s, &mut e, tool, obj(&[("values", Value::Object(v))]));
        // The tool refuses parameters of other groups.
        let mut other = Object::new();
        other.insert(regional.id.as_str(), Value::from(Number::from(mid(regional))));
        assert_eq!(
            code(&call(&mut s, &mut e, &w, tool, obj(&[("values", Value::Object(other))]))),
            "invalid_arguments",
            "{tool}"
        );
    }
    let quat = Value::Array([0.0, 0.0, 0.258_819, 0.965_926].iter().map(|x| Value::from(Number::from(*x))).collect());
    let mut joints = Object::new();
    joints.insert(joint.as_str(), quat);
    run(&mut s, &mut e, "set_pose", obj(&[("joints", Value::Object(joints))]));
    // The expression preset has to exist first: the human stores one.
    let preset = Id128::from_u128(77);
    let mut fields = Object::new();
    fields.insert("name", "smile");
    fields.insert("weights", Value::Object(Object::new()));
    human(&mut e, Action::AddObject { kind: ObjectKind::ExpressionPreset, id: preset, index: None, fields });
    run(&mut s, &mut e, "apply_expression", obj(&[("preset_name", "smile".into())]));
    run(&mut s, &mut e, "list_regions", json("{}"));
    let stats = run(&mut s, &mut e, "get_region_stats", obj(&[("region", region.as_str().into())]));
    assert!(stats.get("geometry").and_then(|g| g.get("vertices")).and_then(Value::as_u64).unwrap() > 0);
    let triple = Value::Array(vec![lm.as_str().into(), lm2.as_str().into(), lm3.as_str().into()]);
    let pair = Value::Array(vec![lm.as_str().into(), lm2.as_str().into()]);
    let m = run(
        &mut s,
        &mut e,
        "measure",
        obj(&[("distances", Value::Array(vec![pair.clone()])), ("angles", Value::Array(vec![triple]))]),
    );
    assert!(
        m.get("distances").and_then(|d| d.at(0)).and_then(|d| d.get("metres")).and_then(Value::as_f64).unwrap() > 0.0
    );
    // Landmark solver: a 2 mm move of the first landmark. Whether the template's components reach
    // it decides between a parameter change and an explicit `no_effect`; both are honest results.
    let moved = call(
        &mut s,
        &mut e,
        &w,
        "move_landmark",
        obj(&[("landmark", lm.as_str().into()), ("position", json("[0, 0.002, 0]")), ("relative", true.into())]),
    );
    called.borrow_mut().push("move_landmark");
    if moved.get("isError") == Some(&Value::Bool(true)) {
        assert_eq!(code(&moved), "no_effect");
    } else {
        assert!(ok(&moved).get("solve").is_some());
    }
    let targets = obj(&[(
        "targets",
        Value::Array(vec![obj(&[("landmark", lm2.as_str().into()), ("position", json("[0, 1.5, 0.1]"))])]),
    )]);
    let t = call(&mut s, &mut e, &w, "set_landmark_targets", targets);
    called.borrow_mut().push("set_landmark_targets");
    assert!(t.get("isError") == Some(&Value::Bool(false)) || code(&t) == "no_effect", "{t:?}");
    run(&mut s, &mut e, "pin_landmark", obj(&[("landmark", lm.as_str().into())]));
    run(&mut s, &mut e, "unpin_landmark", obj(&[("landmark", lm.as_str().into())]));
    let sculpt = run(
        &mut s,
        &mut e,
        "add_sculpt_layer",
        obj(&[("name", "agent sculpt".into()), ("region", region.as_str().into())]),
    );
    let sculpt_id = sculpt
        .get("changes")
        .and_then(|c| c.at(0))
        .and_then(|c| c.get("id"))
        .and_then(Value::as_str)
        .unwrap()
        .to_owned();
    run(
        &mut s,
        &mut e,
        "add_skin_layer",
        json(
            r#"{"name":"freckles","channels":["freckle_mask"],"generator":"freckles","params":{"density":0.3},"seed":7}"#,
        ),
    );
    let layers = run(&mut s, &mut e, "list_layers", json("{}"));
    let skin_id = layers
        .get("skin_layers")
        .and_then(|l| l.at(0))
        .and_then(|l| l.get("id"))
        .and_then(Value::as_str)
        .unwrap()
        .to_owned();
    run(&mut s, &mut e, "get_layer", obj(&[("id", sculpt_id.as_str().into())]));
    run(
        &mut s,
        &mut e,
        "modify_layer",
        obj(&[("kind", "skin".into()), ("id", skin_id.as_str().into()), ("fields", json(r#"{"opacity":0.5}"#))]),
    );
    run(&mut s, &mut e, "add_sculpt_layer", json(r#"{"name":"second"}"#));
    let layers = run(&mut s, &mut e, "list_layers", json("{}"));
    let order: Vec<Value> = layers
        .get("sculpt_layers")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .rev()
        .map(|l| l.get("id").cloned().unwrap())
        .collect();
    run(&mut s, &mut e, "reorder_layers", obj(&[("kind", "sculpt".into()), ("order", Value::Array(order))]));
    run(
        &mut s,
        &mut e,
        "lock_layer",
        obj(&[("kind", "sculpt".into()), ("id", sculpt_id.as_str().into()), ("locked", true.into())]),
    );
    assert_eq!(
        code(&call(
            &mut s,
            &mut e,
            &w,
            "delete_layer",
            obj(&[("kind", "sculpt".into()), ("id", sculpt_id.as_str().into())])
        )),
        "object_locked"
    );
    run(
        &mut s,
        &mut e,
        "lock_layer",
        obj(&[("kind", "sculpt".into()), ("id", sculpt_id.as_str().into()), ("locked", false.into())]),
    );
    run(&mut s, &mut e, "delete_layer", obj(&[("kind", "sculpt".into()), ("id", sculpt_id.as_str().into())]));
    let groom = run(
        &mut s,
        &mut e,
        "groom_create",
        json(r#"{"name":"crop","kind":"scalp","params":{"length":0.03},"seed":3}"#),
    );
    let groom_id = groom
        .get("changes")
        .and_then(|c| c.at(0))
        .and_then(|c| c.get("id"))
        .and_then(Value::as_str)
        .unwrap()
        .to_owned();
    run(&mut s, &mut e, "groom_modify", obj(&[("id", groom_id.as_str().into()), ("fields", json(r#"{"seed":4}"#))]));
    // A guide blob must be in the store: put one there as the groom tools would.
    let blob =
        e.session().project().unwrap().blobs(&w.fs).put(b"guides", humanart_doc::blob::Compression::None).unwrap();
    run(
        &mut s,
        &mut e,
        "groom_set_guides",
        obj(&[("id", groom_id.as_str().into()), ("guides", blob.to_hex().as_str().into()), ("points", 12u64.into())]),
    );
    run(&mut s, &mut e, "set_lighting_rig", json(r#"{"rig":"golden_hour"}"#));
    run(&mut s, &mut e, "set_camera", json(r#"{"camera":"profile_left"}"#));
    run(&mut s, &mut e, "lock_region", obj(&[("region", region.as_str().into())]));
    run(&mut s, &mut e, "unlock_region", obj(&[("region", region.as_str().into())]));
    run(&mut s, &mut e, "lock_parameter", obj(&[("id", b.id.as_str().into())]));
    run(&mut s, &mut e, "lock_parameter", obj(&[("id", b.id.as_str().into()), ("locked", false.into())]));
    run(&mut s, &mut e, "create_snapshot", json(r#"{"name":"before"}"#));
    let snap = e.session().document().snapshots[0].id.to_hex();
    run(&mut s, &mut e, "restore_snapshot", obj(&[("id", snap.as_str().into())]));
    run(&mut s, &mut e, "undo", json("{}"));
    run(&mut s, &mut e, "redo", json("{}"));
    std::fs::write(s_exchange("tools").join("ref.png"), b"\x89PNG\r\n\x1a\nrest").unwrap();
    run(&mut s, &mut e, "import_reference", json(r#"{"file":"ref.png","name":"front photo"}"#));
    assert_eq!(
        code(&call(&mut s, &mut e, &w, "import_reference", json(r#"{"file":"../outside.png","name":"x"}"#))),
        "sandbox"
    );
    let p = run(&mut s, &mut e, "open_proposal", json(r#"{"name":"try"}"#));
    let pid = p.get("proposal_id").and_then(Value::as_str).unwrap().to_owned();
    let mut v = Object::new();
    v.insert(b.id.as_str(), Value::from(Number::from(mid(b))));
    run(&mut s, &mut e, "set_parameters", obj(&[("values", Value::Object(v)), ("proposal_id", pid.as_str().into())]));
    let d = run(&mut s, &mut e, "describe_proposal", obj(&[("proposal_id", pid.as_str().into())]));
    assert_eq!(d.get("commands").and_then(Value::as_array).map(<[Value]>::len), Some(1));
    let c = run(&mut s, &mut e, "commit_proposal", obj(&[("proposal_id", pid.as_str().into())]));
    assert_eq!(c.get("state").and_then(Value::as_str), Some("awaiting_review"));
    run(&mut s, &mut e, "discard_proposal", obj(&[("proposal_id", pid.as_str().into())]));
    let ev = run(&mut s, &mut e, "get_events", json(r#"{"since":0}"#));
    assert!(ev.get("events").and_then(Value::as_array).unwrap().len() > 10);
    let h = run(&mut s, &mut e, "get_history", json(r#"{"limit":5}"#));
    assert_eq!(h.get("entries").and_then(Value::as_array).map(<[Value]>::len), Some(5));
    run(&mut s, &mut e, "get_policy", json("{}"));
    let r = call(
        &mut s,
        &mut e,
        &w,
        "render_view",
        json(r#"{"width":48,"height":48,"frames":1,"views":["front","profile_left"]}"#),
    );
    called.borrow_mut().push("render_view");
    let content = r.get("content").and_then(Value::as_array).unwrap();
    assert_eq!(content.iter().filter(|c| c.get("type").and_then(Value::as_str) == Some("image")).count(), 2);
    let renders = ok(&r).get("renders").and_then(Value::as_array).unwrap();
    assert_eq!(renders[1].get("rig").and_then(Value::as_str), Some("golden_hour"), "set_lighting_rig is the default");
    let file = renders[0].get("file").and_then(Value::as_str).unwrap();
    assert!(file.starts_with("exchange:renders/"));
    assert!(s_exchange("tools").join(file.trim_start_matches("exchange:")).is_file());
    // Colour statistics now have a render to read.
    let stats = run(&mut s, &mut e, "get_region_stats", obj(&[("region", region.as_str().into())]));
    assert!(stats.get("colour").is_some_and(|c| !c.is_null()));
    let called = called.borrow();
    let missing: Vec<&str> = tools::all().map(|t| t.name).filter(|n| !called.contains(n)).collect();
    assert!(missing.is_empty(), "tools never called: {missing:?}");
}

fn s_exchange(tag: &str) -> PathBuf {
    std::fs::canonicalize(std::env::temp_dir().join(format!("humanart-mcp-{tag}-{}", std::process::id()))).unwrap()
}

#[test]
fn ac_wp23_1_negative_locks_protection_and_ranges() {
    let w = world();
    let mut e = engine(&w);
    let mut s = server("negative", AgentPolicy::DEFAULT, &mut e);
    let (a, b, regional, region) = params();
    let set = |id: &str, v: Value, extra: &[(&str, Value)]| {
        let mut values = Object::new();
        values.insert(id, v);
        let mut m: Vec<(&str, Value)> = vec![("values", Value::Object(values))];
        m.extend(extra.iter().cloned());
        obj(&m)
    };
    // [SEC-03] Default policy: no direct edits.
    assert_eq!(
        code(&call(&mut s, &mut e, &w, "set_parameters", set(&a.id, Value::from(Number::from(mid(a))), &[]))),
        "policy"
    );
    assert_eq!(code(&call(&mut s, &mut e, &w, "undo", json("{}"))), "policy");
    let p = ok(&call(&mut s, &mut e, &w, "open_proposal", json(r#"{"name":"n"}"#))).clone();
    let pid: Value = p.get("proposal_id").cloned().unwrap();
    let inp = [("proposal_id", pid.clone())];
    // Ranges [CMD-05]: refused by default, clamped with a warning on request.
    let too_big = Value::from(Number::from(a.range.unwrap().max + 1.0));
    assert_eq!(code(&call(&mut s, &mut e, &w, "set_parameters", set(&a.id, too_big.clone(), &inp))), "out_of_range");
    let mut with_clamp = inp.to_vec();
    with_clamp.push(("clamp", true.into()));
    let clamped = call(&mut s, &mut e, &w, "set_parameters", set(&a.id, too_big, &with_clamp));
    assert!(!ok(&clamped).get("warnings").and_then(Value::as_array).unwrap().is_empty());
    // [LOCK-02] Values the human set are protected: on the document they are refused (with the way
    // out in the message); inside a proposal they only raise a `needs_review` warning, since the
    // human decides when accepting.
    human(&mut e, scalar(&b.id, mid(b)));
    let soft_min = Value::from(Number::from(b.range.unwrap().soft_min));
    let later = ok(&call(&mut s, &mut e, &w, "open_proposal", json(r#"{"name":"after the human"}"#)))
        .get("proposal_id")
        .cloned()
        .unwrap();
    let r = ok(&call(&mut s, &mut e, &w, "set_parameters", set(&b.id, soft_min.clone(), &[("proposal_id", later)])))
        .clone();
    let warned = r.get("warnings").and_then(Value::as_array).unwrap();
    assert!(warned.iter().any(|x| x.get("code").and_then(Value::as_str) == Some("needs_review")), "{r:?}");
    s.set_policy(AgentPolicy { direct_edits: true, commit_without_review: false });
    let r = call(&mut s, &mut e, &w, "set_parameters", set(&b.id, soft_min, &[]));
    assert_eq!(code(&r), "human_protected");
    assert!(
        r.get("structuredContent")
            .and_then(|s| s.get("error"))
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
            .unwrap()
            .contains("ask the human")
    );
    // [LOCK-01] Region and parameter locks (they apply on the document and in proposals).
    human(&mut e, Action::SetLock { target: LockTarget::Region(region.clone()), locked: true });
    let r =
        call(&mut s, &mut e, &w, "set_parameters", set(&regional.id, Value::from(Number::from(mid(regional))), &[]));
    assert_eq!(code(&r), "region_locked");
    assert_eq!(
        code(&call(&mut s, &mut e, &w, "unlock_region", obj(&[("region", region.as_str().into())]))),
        "human_protected"
    );
    human(&mut e, Action::SetLock { target: LockTarget::Region(region.clone()), locked: false });
    human(&mut e, Action::SetLock { target: LockTarget::Parameter(regional.id.clone()), locked: true });
    let r =
        call(&mut s, &mut e, &w, "set_parameters", set(&regional.id, Value::from(Number::from(mid(regional))), &[]));
    assert_eq!(code(&r), "parameter_locked");
    // [AC-WP23-4] Layers the human made are protected: modify, delete and unlock are refused.
    let mut fields = Object::new();
    fields.insert("name", "human skin");
    fields.insert("channels", json(r#"["redness"]"#));
    fields.insert("source", json(r#"{"kind":"procedural","generator":"redness","params":{},"seed":1}"#));
    let layer = Id128::from_u128(0xAB);
    human(&mut e, Action::AddObject { kind: ObjectKind::SkinLayer, id: layer, index: None, fields });
    let id = layer.to_hex();
    for (tool, args) in [
        (
            "modify_layer",
            obj(&[("kind", "skin".into()), ("id", id.as_str().into()), ("fields", json(r#"{"opacity":0.1}"#))]),
        ),
        ("delete_layer", obj(&[("kind", "skin".into()), ("id", id.as_str().into())])),
    ] {
        assert_eq!(code(&call(&mut s, &mut e, &w, tool, args)), "human_protected", "{tool}");
    }
    // ... and a human's layer lock cannot be lifted.
    human(&mut e, Action::SetObjectLock { kind: ObjectKind::SkinLayer, id: layer, locked: true });
    let unlock = obj(&[("kind", "skin".into()), ("id", id.as_str().into()), ("locked", false.into())]);
    assert_eq!(code(&call(&mut s, &mut e, &w, "lock_layer", unlock)), "human_protected");
    s.set_policy(AgentPolicy::DEFAULT);
    assert_eq!(e.session().document().skin_layers.len(), 1, "the protected layer is untouched");
    // The landmark solver leaves protected and locked parameters alone or reports no effect.
    let lm = &template().landmarks()[0].name;
    let r = call(
        &mut s,
        &mut e,
        &w,
        "move_landmark",
        obj(&[
            ("landmark", lm.as_str().into()),
            ("position", json("[0,0.001,0]")),
            ("relative", true.into()),
            ("proposal_id", pid.clone()),
        ]),
    );
    if r.get("isError") == Some(&Value::Bool(true)) {
        assert!(["no_effect", "out_of_range"].contains(&code(&r).as_str()), "{r:?}");
    }
    // Unknown proposal, someone else's proposal.
    let bogus = Value::from("0123456789abcdef0123456789abcdef");
    assert_eq!(
        code(&call(
            &mut s,
            &mut e,
            &w,
            "set_parameters",
            set(&a.id, Value::from(Number::from(mid(a))), &[("proposal_id", bogus.clone())])
        )),
        "unknown_proposal"
    );
    assert_eq!(code(&call(&mut s, &mut e, &w, "commit_proposal", obj(&[("proposal_id", bogus)]))), "unknown_proposal");
    let theirs = e.session_mut().open_proposal(HUMAN, "human's").unwrap();
    assert_eq!(
        code(&call(&mut s, &mut e, &w, "commit_proposal", obj(&[("proposal_id", theirs.to_hex().as_str().into())]))),
        "not_proposal_author"
    );
    // Render limits.
    assert_eq!(code(&call(&mut s, &mut e, &w, "render_view", json(r#"{"width":5000}"#))), "invalid_arguments");
    assert_eq!(code(&call(&mut s, &mut e, &w, "render_view", json(r#"{"view_mode":"xray"}"#))), "invalid_arguments");
}

#[test]
fn sec_03_pause_queues_mutations_and_revoke_refuses_everything() {
    let w = world();
    let mut e = engine(&w);
    let mut s = server("gate", AgentPolicy { direct_edits: true, commit_without_review: true }, &mut e);
    let (a, ..) = params();
    s.set_gate(Gate::Paused);
    let (tx, rx) = channel();
    let mut values = Object::new();
    values.insert(a.id.as_str(), Value::from(Number::from(mid(a))));
    let params =
        with_meta(obj(&[("name", "set_parameters".into()), ("arguments", obj(&[("values", Value::Object(values))]))]));
    let incoming = Incoming::Request { id: Id::Number(5), method: "tools/call".into(), params: params.clone() };
    s.handle(
        &mut e,
        &w.fs,
        Inbound::Message(Envelope {
            incoming,
            transport: Transport::Stdio,
            reply: tx,
            deadline: Instant::now() + LONG,
        }),
    );
    assert!(rx.try_recv().is_err(), "queued, not applied");
    assert_eq!(s.queued(), 1);
    assert!(!e.session().document().params.contains_key(&a.id));
    ok(&call(&mut s, &mut e, &w, "describe_character", json("{}")));
    s.set_gate(Gate::Open);
    let (_, inbox) = channel::<Inbound>();
    s.pump(&mut e, &w.fs, &inbox);
    let Ok(Reply::Message { body, .. }) = rx.try_recv() else { panic!("the queued call ran on resume") };
    assert_eq!(parse(&body).unwrap().get("result").and_then(|r| r.get("isError")), Some(&Value::Bool(false)));
    assert!(e.session().document().params.contains_key(&a.id));
    // An expired queued call is answered, not run.
    s.set_gate(Gate::Paused);
    let (tx, rx) = channel();
    let incoming = Incoming::Request { id: Id::Number(6), method: "tools/call".into(), params: params.clone() };
    s.handle(
        &mut e,
        &w.fs,
        Inbound::Message(Envelope { incoming, transport: Transport::Stdio, reply: tx, deadline: Instant::now() }),
    );
    std::thread::sleep(Duration::from_millis(5));
    s.set_gate(Gate::Open);
    s.pump(&mut e, &w.fs, &inbox);
    let Ok(Reply::Message { status, .. }) = rx.try_recv() else { panic!() };
    assert_eq!(status, 504);
    // Revoke: the queue and later requests are refused; discovery still answers.
    s.set_gate(Gate::Paused);
    let (tx, rx) = channel();
    let incoming = Incoming::Request { id: Id::Number(7), method: "tools/call".into(), params };
    s.handle(
        &mut e,
        &w.fs,
        Inbound::Message(Envelope {
            incoming,
            transport: Transport::Stdio,
            reply: tx,
            deadline: Instant::now() + LONG,
        }),
    );
    s.set_gate(Gate::Revoked);
    let Ok(Reply::Message { status, .. }) = rx.try_recv() else { panic!() };
    assert_eq!(status, 403);
    let (status, m) = send(&mut s, &mut e, &w, "tools/call", with_meta(json(r#"{"name":"describe_character"}"#)));
    assert_eq!((status, m.get("error").and_then(|e| e.get("code")).and_then(Value::as_i64)), (403, Some(4001)));
    let (status, _) = send(&mut s, &mut e, &w, "server/discover", with_meta(Value::Null));
    assert_eq!(status, 200);
}

#[test]
fn ag_01_protocol_versions_metadata_and_legacy_handshake() {
    let w = world();
    let mut e = engine(&w);
    let mut s = server("protocol", AgentPolicy::DEFAULT, &mut e);
    let (_, d) = send(&mut s, &mut e, &w, "server/discover", with_meta(Value::Null));
    let r = d.get("result").unwrap();
    assert_eq!(r.get("resultType").and_then(Value::as_str), Some("complete"));
    assert!(r.get("supportedVersions").and_then(Value::as_array).unwrap().contains(&Value::from("2026-07-28")));
    assert!(r.get("_meta").and_then(|m| m.get("io.modelcontextprotocol/serverInfo")).is_some());
    let old = json(
        r#"{"_meta":{"io.modelcontextprotocol/protocolVersion":"1900-01-01","io.modelcontextprotocol/clientCapabilities":{}}}"#,
    );
    let (status, m) = send(&mut s, &mut e, &w, "tools/list", old);
    assert_eq!((status, m.get("error").and_then(|e| e.get("code")).and_then(Value::as_i64)), (400, Some(-32022)));
    assert!(m.get("error").and_then(|e| e.get("data")).and_then(|d| d.get("supported")).is_some());
    let no_caps = json(r#"{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28"}}"#);
    assert_eq!(send(&mut s, &mut e, &w, "tools/list", no_caps).0, 400);
    assert_eq!(send(&mut s, &mut e, &w, "tools/list", Value::Null).0, 400, "no _meta and no handshake");
    assert_eq!(send(&mut s, &mut e, &w, "no/such", with_meta(Value::Null)).0, 404);
    // Resources.
    let (_, l) = send(&mut s, &mut e, &w, "resources/list", with_meta(Value::Null));
    assert!(l.get("result").and_then(|r| r.get("resources")).and_then(Value::as_array).unwrap().len() >= 4);
    for uri in ["humanart://document", "humanart://registry", "humanart://cameras", "humanart://rigs"] {
        let (status, m) = send(&mut s, &mut e, &w, "resources/read", with_meta(obj(&[("uri", uri.into())])));
        assert_eq!(status, 200, "{uri}");
        let text = m
            .get("result")
            .and_then(|r| r.get("contents"))
            .and_then(|c| c.at(0))
            .and_then(|c| c.get("text"))
            .and_then(Value::as_str)
            .unwrap();
        assert!(parse_str(text).is_ok(), "{uri}");
    }
    assert_eq!(
        send(&mut s, &mut e, &w, "resources/read", with_meta(json(r#"{"uri":"humanart://renders/99"}"#))).0,
        400
    );
    // Legacy: initialize on stdio, then requests without _meta.
    let (status, m) = send(
        &mut s,
        &mut e,
        &w,
        "initialize",
        json(r#"{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"old","version":"1"}}"#),
    );
    assert_eq!(status, 200);
    assert_eq!(m.get("result").and_then(|r| r.get("protocolVersion")).and_then(Value::as_str), Some("2025-06-18"));
    assert_eq!(send(&mut s, &mut e, &w, "tools/list", Value::Null).0, 200);
}

// ---- Threaded sessions: the script runs on its own thread, the engine stays on this one. ----

enum HumanAct {
    Set(String, f64),
    Accept(String),
    Pause,
    Resume,
}

type HumanLink = Sender<(HumanAct, Sender<()>)>;

fn act(link: &HumanLink, a: HumanAct) {
    let (tx, rx) = channel();
    link.send((a, tx)).unwrap();
    rx.recv_timeout(LONG).unwrap();
}

/// Pumps the server and applies the human's actions until `script` ends; re-raises its panic.
fn drive(
    s: &mut Server,
    e: &mut Engine<'_>,
    w: &World,
    inbox: &Receiver<Inbound>,
    human: &Receiver<(HumanAct, Sender<()>)>,
    script: std::thread::JoinHandle<()>,
) {
    while !script.is_finished() {
        s.pump(e, &w.fs, inbox);
        while let Ok((a, done)) = human.try_recv() {
            match a {
                HumanAct::Set(id, v) => human_set(e, &id, v),
                HumanAct::Accept(p) => {
                    e.session_mut().accept(Id128::parse(&p).unwrap(), HUMAN, None).unwrap();
                    e.update().unwrap();
                }
                HumanAct::Pause => s.set_gate(Gate::Paused),
                HumanAct::Resume => s.set_gate(Gate::Open),
            }
            s.flush(e);
            let _ = done.send(());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    if let Err(p) = script.join() {
        std::panic::resume_unwind(p);
    }
}

fn human_set(e: &mut Engine<'_>, id: &str, v: f64) {
    human(e, scalar(id, v));
}

/// What the session script needs from a client.
trait Agent {
    fn tool(&mut self, name: &str, args: Value) -> Value;
    fn rpc(&mut self, method: &str, params: Value) -> Result<Value, ClientError>;
    fn note(&mut self, pick: &dyn Fn(&Value) -> bool) -> Option<Value>;
}

struct Http {
    client: HttpClient,
    events: EventStream,
}

impl Agent for Http {
    fn tool(&mut self, name: &str, args: Value) -> Value {
        self.client.call_tool(name, args).unwrap()
    }
    fn rpc(&mut self, method: &str, params: Value) -> Result<Value, ClientError> {
        self.client.request(method, params)
    }
    fn note(&mut self, pick: &dyn Fn(&Value) -> bool) -> Option<Value> {
        self.events.wait_for(LONG, pick)
    }
}

struct Stdio {
    client: StdioClient<std::io::PipeWriter>,
}

impl Agent for Stdio {
    fn tool(&mut self, name: &str, args: Value) -> Value {
        self.client.call_tool(name, args).unwrap()
    }
    fn rpc(&mut self, method: &str, params: Value) -> Result<Value, ClientError> {
        self.client.request(method, params)
    }
    fn note(&mut self, pick: &dyn Fn(&Value) -> bool) -> Option<Value> {
        self.client.wait_notification(LONG, pick)
    }
}

fn is(method: &'static str) -> impl Fn(&Value) -> bool {
    move |v: &Value| v.get("method").and_then(Value::as_str) == Some(method)
}

/// The scripted edit session `[AC-WP23-1]`, `[AC-WP23-4]`.
fn script(agent: &mut dyn Agent, link: &HumanLink, ids: (String, f64, String, f64)) {
    let (a, va, b, vb) = ids;
    let discover = agent.rpc("server/discover", Value::Null).unwrap();
    assert!(discover.get("supportedVersions").is_some());
    let tools = agent.rpc("tools/list", Value::Null).unwrap();
    assert_eq!(tools.get("tools").and_then(Value::as_array).map(<[Value]>::len), Some(tools::all().count()));
    ok(&agent.tool("describe_character", json("{}")));
    let p = ok(&agent.tool("open_proposal", json(r#"{"name":"session"}"#))).clone();
    let pid = p.get("proposal_id").and_then(Value::as_str).unwrap().to_owned();
    let opened = agent.note(&is("notifications/humanart/proposal_state_changed")).unwrap();
    assert_eq!(opened.get("params").and_then(|p| p.get("state")).and_then(Value::as_str), Some("open"));
    let mut v = Object::new();
    v.insert(a.as_str(), Value::from(Number::from(va)));
    let out = agent.tool(
        "set_parameters",
        obj(&[("values", Value::Object(v)), ("proposal_id", pid.as_str().into()), ("reason", "scripted".into())]),
    );
    assert_eq!(ok(&out).get("proposal").and_then(Value::as_str), Some(pid.as_str()));
    // The human edits during the session: the agent sees it.
    act(link, HumanAct::Set(b.clone(), vb));
    let seen = agent
        .note(&|n: &Value| {
            n.get("method").and_then(Value::as_str) == Some("notifications/humanart/event")
                && n.get("params").and_then(|p| p.get("outcome")).and_then(|o| o.get("author")).and_then(Value::as_str)
                    == Some(HUMAN)
        })
        .expect("the human's edit is notified");
    let summary = seen.get("params").and_then(|p| p.get("summary")).and_then(Value::as_str).unwrap();
    assert!(summary.contains(&b), "{summary}");
    // ... and cannot overwrite it silently [LOCK-02]: in a proposal the change is marked for the
    // human's review (the default policy allows no direct edit at all).
    let mut v = Object::new();
    v.insert(b.as_str(), Value::from(Number::from(vb * 0.5)));
    let direct = agent.tool("set_parameters", obj(&[("values", Value::Object(v.clone()))]));
    assert_eq!(code(&direct), "policy");
    let review = ok(&agent.tool("open_proposal", json(r#"{"name":"over the human"}"#)))
        .get("proposal_id")
        .and_then(Value::as_str)
        .unwrap()
        .to_owned();
    let r =
        ok(&agent
            .tool("set_parameters", obj(&[("values", Value::Object(v)), ("proposal_id", review.as_str().into())])))
        .clone();
    let warned = r.get("warnings").and_then(Value::as_array).unwrap();
    assert!(warned.iter().any(|x| x.get("code").and_then(Value::as_str) == Some("needs_review")), "{r:?}");
    ok(&agent.tool("discard_proposal", obj(&[("proposal_id", review.as_str().into())])));
    let r = agent.tool("render_view", json(r#"{"width":32,"height":32,"frames":1,"camera":"front"}"#));
    assert!(
        r.get("content")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .any(|c| c.get("type").and_then(Value::as_str) == Some("image"))
    );
    let pending = ok(&agent.tool("render_view", json(r#"{"width":32,"height":32,"frames":1,"async":true}"#))).clone();
    assert_eq!(pending.get("state").and_then(Value::as_str), Some("pending"));
    let ready = agent.note(&is("notifications/humanart/render_ready")).expect("render_ready");
    assert_eq!(ready.get("params").and_then(|p| p.get("render_ids")), pending.get("render_ids"));
    let c = ok(&agent.tool("commit_proposal", obj(&[("proposal_id", pid.as_str().into())]))).clone();
    assert_eq!(c.get("state").and_then(Value::as_str), Some("awaiting_review"));
    assert!(
        agent
            .note(&|n: &Value| n.get("params").and_then(|p| p.get("state")).and_then(Value::as_str)
                == Some("awaiting_review"))
            .is_some()
    );
    act(link, HumanAct::Accept(pid.clone()));
    let accepted = agent
        .note(&|n: &Value| n.get("params").and_then(|p| p.get("state")).and_then(Value::as_str) == Some("accepted"));
    assert!(accepted.is_some(), "proposal_state_changed accepted");
    assert!(agent.note(&is("notifications/resources/updated")).is_some(), "the document resource changed");
    // Pause: a mutation waits; resume releases it.
    act(link, HumanAct::Pause);
    let r = agent.tool("describe_character", json("{}"));
    assert!(ok(&r).get("proposals").is_some(), "reads work while paused");
    act(link, HumanAct::Resume);
    let events = ok(&agent.tool("get_events", json(r#"{"since":0}"#))).clone();
    let types: Vec<&str> = events
        .get("events")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .filter_map(|e| e.get("type").and_then(Value::as_str))
        .collect();
    assert!(types.contains(&"proposal_accepted") && types.contains(&"command_committed"), "{types:?}");
}

fn script_ids() -> (String, f64, String, f64) {
    let (a, b, ..) = params();
    (a.id.clone(), mid(a), b.id.clone(), mid(b))
}

#[test]
fn ac_wp23_1_scripted_session_over_http() {
    let w = world();
    let mut e = engine(&w);
    let mut s = server("http-session", AgentPolicy::DEFAULT, &mut e);
    let (tx, inbox) = channel();
    let token = Token::from_text("test-token-0123456789abcdef").unwrap();
    let mut http = HttpServer::start(0, token, HttpConfig::DEFAULT, tx).unwrap();
    let addr = http.addr();
    assert!(addr.ip().is_loopback());
    let (link, human_rx) = channel();
    let ids = script_ids();
    let run = std::thread::spawn(move || {
        let mut client = HttpClient::new(addr, "test-token-0123456789abcdef");
        let events = client
            .listen(json(r#"{"com.humanart/events":true,"resourceSubscriptions":["humanart://document"]}"#))
            .unwrap();
        let mut agent = Http { client, events };
        script(&mut agent, &link, ids);
    });
    drive(&mut s, &mut e, &w, &inbox, &human_rx, run);
    s.shutdown();
    http.stop();
}

#[test]
fn ac_wp23_1_scripted_session_over_stdio() {
    let w = world();
    let mut e = engine(&w);
    let mut s = server("stdio-session", AgentPolicy::DEFAULT, &mut e);
    let (server_in, client_out) = std::io::pipe().unwrap();
    let (client_in, server_out) = std::io::pipe().unwrap();
    let (inbox, threads) = spawn_stdio(server_in, server_out, RpcLimits::DEFAULT).unwrap();
    let (link, human_rx) = channel();
    let ids = script_ids();
    let run = std::thread::spawn(move || {
        let mut client = StdioClient::new(client_in, client_out).unwrap();
        // Garbage and oversized-free junk lines get parse errors, the session goes on [ROB-01].
        client.send_line(b"{not json").unwrap();
        let err = client.next_message(LONG).unwrap();
        assert_eq!(err.get("error").and_then(|e| e.get("code")).and_then(Value::as_i64), Some(-32700));
        let listen = super::jsonrpc::request(
            &Id::String("events".into()),
            "subscriptions/listen",
            with_meta(json(
                r#"{"notifications":{"com.humanart/events":true,"resourceSubscriptions":["humanart://document"]}}"#,
            )),
        );
        client.send_line(&super::jsonrpc::encode(&listen)).unwrap();
        let ack = client.wait_notification(LONG, is("notifications/subscriptions/acknowledged")).unwrap();
        assert_eq!(
            ack.get("params")
                .and_then(|p| p.get("_meta"))
                .and_then(|m| m.get("io.modelcontextprotocol/subscriptionId"))
                .and_then(Value::as_str),
            Some("events")
        );
        let mut agent = Stdio { client };
        script(&mut agent, &link, ids);
        // Cancelling the subscription on stdio ends it.
        agent
            .client
            .send_line(br#"{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":"events"}}"#)
            .unwrap();
        drop(agent);
    });
    drive(&mut s, &mut e, &w, &inbox, &human_rx, run);
    s.pump(&mut e, &w.fs, &inbox);
    assert_eq!(s.subscriptions(), 0, "the cancelled subscription is gone");
    drop(inbox);
    drop(s);
    threads.join();
}

#[test]
fn sec_01_http_refuses_tokens_hosts_origins_limits_and_slow_clients() {
    use std::io::Write as _;
    let w = world();
    let mut e = engine(&w);
    let mut s = server("http-sec", AgentPolicy::DEFAULT, &mut e);
    let (tx, inbox) = channel();
    let config = HttpConfig {
        // The deadline is short for the slow-client check but leaves room for a loaded test machine.
        limits: super::http::HttpLimits {
            max_body_bytes: 4096,
            deadline: Duration::from_millis(1500),
            ..super::http::HttpLimits::DEFAULT
        },
        max_connections: 4,
        burst: 40,
        per_second: 1,
        ..HttpConfig::DEFAULT
    };
    let token = "sec-token-0123456789abcdef";
    let mut http = HttpServer::start(0, Token::from_text(token).unwrap(), config, tx).unwrap();
    let addr = http.addr();
    let (_link, human_rx) = channel::<(HumanAct, Sender<()>)>();
    let run = std::thread::spawn(move || {
        let port = addr.port();
        let host = format!("127.0.0.1:{port}");
        let bearer = format!("Bearer {token}");
        let body = br#"{"jsonrpc":"2.0","id":1,"method":"server/discover","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}}}}"#;
        let c = HttpClient::new(addr, token);
        let base = |extra: &[(&'static str, String)]| {
            let mut h: Vec<(&str, String)> = vec![
                ("Host", host.clone()),
                ("Authorization", bearer.clone()),
                ("Content-Type", "application/json".into()),
                ("Accept", "application/json, text/event-stream".into()),
                ("MCP-Protocol-Version", "2026-07-28".into()),
                ("Mcp-Method", "server/discover".into()),
            ];
            for (k, v) in extra {
                h.retain(|(x, _)| !x.eq_ignore_ascii_case(k));
                if !v.is_empty() {
                    h.push((k, v.clone()));
                }
            }
            h
        };
        let status = |extra: &[(&'static str, String)], method: &str, body: &[u8]| {
            let h = base(extra);
            let refs: Vec<(&str, &str)> = h.iter().map(|(k, v)| (*k, v.as_str())).collect();
            c.raw(method, &refs, body).unwrap_or_else(|e| panic!("{method} {extra:?}: {e}")).status
        };
        assert_eq!(status(&[], "POST", body), 200);
        // [SEC-01] token.
        assert_eq!(status(&[("Authorization", String::new())], "POST", body), 401);
        assert_eq!(status(&[("Authorization", "Bearer wrong-token-0123456789ab".into())], "POST", body), 401);
        assert_eq!(status(&[("Authorization", format!("Basic {token}"))], "POST", body), 401);
        // [SEC-01] Host / Origin (DNS rebinding, browsers).
        assert_eq!(status(&[("Host", format!("evil.example:{port}"))], "POST", body), 403);
        assert_eq!(status(&[("Host", "127.0.0.1:1".into())], "POST", body), 403);
        assert_eq!(status(&[("Origin", "http://evil.example".into())], "POST", body), 403);
        assert_eq!(status(&[("Origin", "null".into())], "POST", body), 403);
        assert_eq!(status(&[("Origin", format!("http://localhost:{port}"))], "POST", body), 200);
        // Media types, methods, paths.
        assert_eq!(status(&[("Content-Type", "text/plain".into())], "POST", body), 415);
        assert_eq!(status(&[("Accept", "text/html".into())], "POST", body), 406);
        assert_eq!(status(&[], "GET", b""), 405);
        assert_eq!(status(&[], "PUT", body), 405);
        // Header / body mismatches (2026-07-28 server validation).
        assert_eq!(status(&[("Mcp-Method", "tools/list".into())], "POST", body), 400);
        assert_eq!(status(&[("MCP-Protocol-Version", String::new())], "POST", body), 400);
        // Malformed bodies [ROB-01].
        assert_eq!(status(&[], "POST", b"{nope"), 400);
        assert_eq!(status(&[], "POST", b"[]"), 400);
        // [ROB-02] Oversized body.
        assert_eq!(status(&[], "POST", &vec![b' '; 5000]), 413);
        // [ROB-03] A slow client gets 408 at the deadline.
        let mut slow = std::net::TcpStream::connect(addr).unwrap();
        slow.write_all(b"POST /mcp HTTP/1.1\r\nHost: ").unwrap();
        slow.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let t = Instant::now();
        let (resp, _) = super::http::read_response_head(&mut slow, &super::http::HttpLimits::DEFAULT).unwrap();
        assert_eq!(resp.status, 408);
        assert!(t.elapsed() < Duration::from_secs(5));
        // Legacy session ids that do not exist.
        assert_eq!(status(&[("Mcp-Session-Id", "feedface".into())], "GET", b""), 404);
        // [SEC-01] rate limit: the bucket empties.
        let mut limited = false;
        for _ in 0..60 {
            if status(&[], "POST", body) == 429 {
                limited = true;
                break;
            }
        }
        assert!(limited, "the rate limiter refuses a burst");
    });
    drive(&mut s, &mut e, &w, &inbox, &human_rx, run);
    http.stop();
}

#[test]
fn sec_01_http_caps_connections_and_binds_loopback_only() {
    let (tx, _inbox) = channel();
    let config = HttpConfig { max_connections: 2, ..HttpConfig::DEFAULT };
    let mut http = HttpServer::start(0, Token::from_text("cap-token-0123456789abcdef").unwrap(), config, tx).unwrap();
    assert!(http.addr().ip().is_loopback());
    let idle: Vec<std::net::TcpStream> = (0..2).map(|_| std::net::TcpStream::connect(http.addr()).unwrap()).collect();
    std::thread::sleep(Duration::from_millis(200));
    let mut third = std::net::TcpStream::connect(http.addr()).unwrap();
    third.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let (resp, _) = super::http::read_response_head(&mut third, &super::http::HttpLimits::DEFAULT).unwrap();
    assert_eq!(resp.status, 503);
    drop(idle);
    http.stop();
}

#[test]
fn ac_wp23_3_headless_render_matches_the_editor_viewport() {
    use humanart_gpu::resources::{Budget, Resources};
    let w = world();
    let mut e = engine(&w);
    let mut s = server("parity", AgentPolicy { direct_edits: true, commit_without_review: false }, &mut e);
    // The same edited state for both.
    let (a, ..) = params();
    let mut v = Object::new();
    v.insert(a.id.as_str(), Value::from(Number::from(mid(a))));
    ok(&call(&mut s, &mut e, &w, "set_parameters", obj(&[("values", Value::Object(v))])));
    let (size, frames) = (160u32, 12u32);
    let r = call(
        &mut s,
        &mut e,
        &w,
        "render_view",
        obj(&[
            ("camera", "head".into()),
            ("width", u64::from(size).into()),
            ("height", u64::from(size).into()),
            ("frames", u64::from(frames).into()),
            ("quality", "interactive".into()),
        ]),
    );
    let png = &s.renders().last().unwrap().png;
    assert_eq!(
        ok(&r).get("renders").and_then(|x| x.at(0)).and_then(|x| x.get("width")).and_then(Value::as_u64),
        Some(u64::from(size))
    );
    let headless = humanart_io::png::decode(png, &humanart_io::Limits::default()).unwrap();
    // The editor's path: a viewport viewer framed on the head, default settings.
    let gpu = humanart_gpu::shared_headless().unwrap();
    let mut res = Resources::new(gpu, Budget::UNLIMITED);
    let mut viewer = crate::engine::Viewer::new(gpu, &mut res, e.model(), size, size).unwrap();
    viewer.frame_model(e.model(), crate::engine::Framing::Head);
    for _ in 0..frames {
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("editor") });
        viewer.record(&mut res, &mut enc, e.model_mut(), &[]).unwrap();
        gpu.queue.submit([enc.finish()]);
        viewer.after_submit();
    }
    let (_, _, editor) =
        humanart_render::capture::read_texture(&mut res, viewer.display_texture().unwrap(), 0).unwrap();
    viewer.destroy(&mut res);
    // Compare colour channels only: the PNG may or may not carry alpha.
    let channels = usize::from(headless.channels);
    let pixels: Vec<u8> = headless.as_u8().unwrap().chunks_exact(channels).flat_map(|p| p[..3].to_vec()).collect();
    let editor: Vec<u8> = editor.as_chunks::<4>().0.iter().flat_map(|p| p[..3].to_vec()).collect();
    assert_eq!(pixels.len(), editor.len());
    let diffs: Vec<u8> = pixels.iter().zip(&editor).map(|(x, y)| x.abs_diff(*y)).collect();
    let mean = diffs.iter().map(|d| f64::from(*d)).sum::<f64>() / diffs.len() as f64;
    let max = diffs.iter().copied().max().unwrap_or(0);
    println!("[AC-WP23-3] headless vs editor viewport: mean abs diff {mean:.4}/255, max {max}/255");
    assert!(mean < 0.5, "mean difference {mean}");
    assert!(max <= 8, "max difference {max}");
}
