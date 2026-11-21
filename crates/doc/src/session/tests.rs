use humanart_core::platform::{ManualClock, MemoryFs, parent_path};
use humanart_core::rng::Rng;
use humanart_io::json::{self, Number, Object, Value};
use std::sync::Mutex;

use super::*;
use crate::command::{Action, ObjectKind, RangePolicy};
use crate::document::LockTarget;
use crate::notify::Since;
use crate::registry::ParamValue;
use crate::registry::tests::real_registry;

const HUMAN: &str = "human.artist";
const AGENT: &str = "agent.assistant";
const PARAMS: [&str; 7] = [
    "face.nose.bridge_height",
    "face.nose.bridge_width",
    "face.nose.length",
    "face.nose.tip_projection",
    "face.eyes.spacing",
    "face.eyes.size",
    "face.jaw.width",
];
const REGIONS: [&str; 3] = ["nasal-tip", "chin", "forehead"];
const ROOT: &str = "proj";

fn fresh() -> Document {
    let human = Author { id: HUMAN.into(), kind: AuthorKind::Human, name: "Artist".into() };
    let mut doc = Document::new(Id128::from_u128(0xd0c), "Test", "humanart-template", 3, 1_763_500_000_000, human);
    doc.add_author(Author { id: AGENT.into(), kind: AuthorKind::Agent, name: "Assistant".into() });
    doc
}

fn config() -> SessionConfig {
    SessionConfig { journal_interval_ms: 0, autosave_every: 0, autosave_idle_ms: 0, ..SessionConfig::DEFAULT }
}

fn seed(n: u128) -> Id128 {
    Id128::from_u128(0x5eed_0000_0000 + n)
}

fn set(author: &str, id: &str, v: f64) -> Command {
    Command::new(
        author,
        Action::SetParams {
            values: vec![(id.into(), ParamValue::Scalar(v))],
            relative: false,
            policy: RangePolicy::Reject,
        },
    )
}

fn value(s: &Session<'_>, id: &str) -> f64 {
    s.document().param(s.registry, id).unwrap().as_scalar().unwrap()
}

fn round(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

fn add_layer(n: u64, region: &str) -> Command {
    let mut fields = Object::new();
    fields.insert("name", format!("Layer {n}").as_str());
    fields.insert("region", region);
    Command::new(HUMAN, Action::AddObject { kind: ObjectKind::SculptLayer, id: Id128::NIL, index: None, fields })
}

/// A random command by the human, valid on `doc` (except that a reset of a default value changes
/// nothing).
fn random_command(doc: &Document, rng: &mut Rng, n: u64) -> Command {
    let param = PARAMS[rng.next_u32() as usize % PARAMS.len()];
    let layer = (!doc.sculpt_layers.is_empty())
        .then(|| doc.sculpt_layers[rng.next_u32() as usize % doc.sculpt_layers.len()].id);
    match (rng.next_u32() % 7, layer) {
        (3, _) => add_layer(n, REGIONS[rng.next_u32() as usize % REGIONS.len()]),
        (4, Some(id)) => {
            let mut fields = Object::new();
            fields.insert("opacity", Value::from(Number::Float(round(rng.range_f64(0.0, 1.0)))));
            Command::new(HUMAN, Action::ModifyObject { kind: ObjectKind::SculptLayer, id, fields })
        }
        (5, Some(id)) => Command::new(HUMAN, Action::DeleteObject { kind: ObjectKind::SculptLayer, id }),
        (6, _) => Command::new(HUMAN, Action::ResetParams { ids: vec![param.into()] }),
        _ => set(HUMAN, param, round(rng.range_f64(-2.0, 2.0))),
    }
}

/// One random step: a command, a slider gesture, an undo or a redo. Returns `true` when the
/// step made a new history step (so the model pushes a state). `frame` sees the session after
/// every frame of a gesture.
fn random_step(
    s: &mut Session<'_>,
    rng: &mut Rng,
    n: u64,
    frame: &mut dyn FnMut(&Session<'_>),
) -> Result<Option<bool>, CommandError> {
    match rng.next_u32() % 10 {
        0..=4 => {
            let before = s.document().clone();
            let cmd = random_command(s.document(), rng, n);
            s.execute(cmd)?;
            Ok(Some(s.document() != &before))
        }
        5 => {
            let before = s.document().clone();
            let param = PARAMS[rng.next_u32() as usize % PARAMS.len()];
            for _ in 0..2 + rng.next_u32() % 5 {
                s.execute(set(HUMAN, param, round(rng.range_f64(-2.0, 2.0))).in_gesture())?;
                frame(s);
            }
            s.end_gesture()?;
            Ok(Some(s.document() != &before))
        }
        6..=7 => s.undo(HUMAN, &Scope::Any).map(|_| None),
        _ => s.redo(HUMAN, &Scope::Any).map(|_| None),
    }
}

fn copy_fs(fs: &MemoryFs) -> MemoryFs {
    let copy = MemoryFs::new();
    for path in fs.file_paths() {
        let dir = parent_path(&path);
        if !dir.is_empty() {
            copy.create_dir_all(dir).unwrap();
        }
        copy.write(&path, &fs.read(&path).unwrap()).unwrap();
    }
    copy
}

fn journal_path() -> String {
    Project::at(ROOT).journal_path()
}

#[test]
fn ac_wp08_1_undo_redo_algebra_on_random_sequences_and_after_replay() {
    let reg = real_registry();
    for case in 0..6_u64 {
        let fs = MemoryFs::new();
        let clock = ManualClock::new(1_763_600_000_000);
        let mut s = Session::create(ROOT, fresh(), &reg, &fs, &clock, config(), seed(1)).unwrap();
        let mut rng = Rng::new(0xa1_9e00 + case);
        let mut states = vec![s.document().clone()];
        let mut cursor = 0;
        for n in 0..120 {
            clock.advance_millis(7);
            match random_step(&mut s, &mut rng, n, &mut |_| {}) {
                Ok(Some(true)) => {
                    states.truncate(cursor + 1);
                    states.push(s.document().clone());
                    cursor += 1;
                }
                Ok(Some(false)) | Err(CommandError::Rejected(Rejection::NothingToDo { .. })) => {}
                Ok(None) => {
                    // An undo or redo that succeeded moves along the model.
                    let back = cursor.checked_sub(1).filter(|&c| &states[c] == s.document());
                    let fwd = (cursor + 1 < states.len() && &states[cursor + 1] == s.document()).then_some(cursor + 1);
                    cursor = back.or(fwd).unwrap_or_else(|| panic!("case {case} step {n}: undo/redo left the model"));
                }
                Err(e) => panic!("case {case} step {n}: {e}"),
            }
            assert_eq!(s.document(), &states[cursor], "case {case} step {n}");
            assert_eq!(s.history().cursor(), cursor, "case {case} step {n}");
        }
        // [AC-WP08-1] replay from the journal rebuilds the same document and the same history.
        let (mut r, report) = Session::open(ROOT, &reg, &fs, &clock, config(), seed(2)).unwrap();
        assert!(matches!(report.journal, JournalOutcome::Recovered { .. }), "{:?}", report.journal);
        assert_eq!(r.document(), &states[cursor]);
        assert_eq!(r.history().cursor(), cursor);
        assert_eq!(r.history().entries().len(), states.len() - 1);
        for c in (0..cursor).rev() {
            r.undo(HUMAN, &Scope::Any).unwrap();
            assert_eq!(r.document(), &states[c]);
        }
        for state in &states[1..] {
            r.redo(HUMAN, &Scope::Any).unwrap();
            assert_eq!(r.document(), state);
        }
        assert_eq!(r.redo(HUMAN, &Scope::Any).unwrap_err().code(), "nothing_to_do");
    }
}

#[test]
fn ac_wp08_3_a_crash_at_any_byte_of_the_journal_restores_the_last_consistent_state() {
    let reg = real_registry();
    let fs = MemoryFs::new();
    let clock = ManualClock::new(1_763_600_000_000);
    let mut s = Session::create(ROOT, fresh(), &reg, &fs, &clock, config(), seed(1)).unwrap();
    let journal_len = |fs: &MemoryFs| fs.read(&journal_path()).unwrap().len();
    let mut checkpoints = vec![(journal_len(&fs), s.document().clone())];
    let mut rng = Rng::new(0xc4a5);
    for n in 0..40 {
        clock.advance_millis(5);
        let mut frames = Vec::new();
        let result = random_step(&mut s, &mut rng, n, &mut |s| frames.push((journal_len(&fs), s.document().clone())));
        checkpoints.extend(frames);
        match result {
            Ok(_) | Err(CommandError::Rejected(Rejection::NothingToDo { .. })) => {}
            Err(e) => panic!("step {n}: {e}"),
        }
        checkpoints.push((journal_len(&fs), s.document().clone()));
    }
    let bytes = fs.read(&journal_path()).unwrap();
    // Every record boundary, and three cuts inside every record (a kill during its write).
    let mut cuts: Vec<usize> = Vec::new();
    let mut start = 0;
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'\n' {
            let end = i + 1;
            cuts.extend([start + 1, usize::midpoint(start, end), end - 1, end]);
            start = end;
        }
    }
    cuts.retain(|&c| c > checkpoints[0].0 && c <= bytes.len());
    cuts.dedup();
    assert!(cuts.len() > 100, "{} cuts", cuts.len());
    for &cut in &cuts {
        let copy = copy_fs(&fs);
        copy.write(&journal_path(), &bytes[..cut]).unwrap();
        let expected = &checkpoints.iter().rev().find(|(len, _)| *len <= cut).unwrap().1;
        let (mut r, _) = Session::open(ROOT, &reg, &copy, &clock, config(), seed(3)).unwrap();
        assert_eq!(r.document(), expected, "cut at {cut} of {}", bytes.len());
        // Work continues on the repaired journal and survives the next restart too.
        r.execute(set(HUMAN, "face.jaw.width", 1.25)).unwrap();
        let after = r.document().clone();
        let (again, _) = Session::open(ROOT, &reg, &copy, &clock, config(), seed(4)).unwrap();
        assert_eq!(again.document(), &after, "cut at {cut}");
    }
}

/// A scripted session touching every kind of change; stops at the first error.
fn script(s: &mut Session<'_>, clock: &ManualClock) -> Result<(), CommandError> {
    s.execute(set(HUMAN, "face.jaw.width", 0.5))?;
    s.execute(set(AGENT, "face.nose.length", -0.25))?;
    s.execute(add_layer(1, "chin"))?;
    let p = s.open_proposal(AGENT, "Longer nose")?;
    s.execute_in(p, set(AGENT, "face.nose.length", 0.75))?;
    s.execute_in(p, set(AGENT, "face.eyes.size", 0.5))?;
    clock.advance_millis(10);
    s.undo(HUMAN, &Scope::Any)?;
    s.redo(HUMAN, &Scope::Any)?;
    s.select(HUMAN, Some("chin"), None)?;
    s.accept(p, HUMAN, None)?;
    s.save()?;
    s.execute(set(HUMAN, "face.eyes.spacing", 0.25))?;
    let q = s.open_proposal(AGENT, "Wider")?;
    s.execute_in(q, set(AGENT, "face.nose.bridge_width", 1.0))?;
    s.undo(AGENT, &Scope::Author(AGENT.into()))?;
    s.autosave()?;
    s.execute(set(HUMAN, "face.eyes.spacing", 0.5))?;
    Ok(())
}

#[test]
fn ac_wp08_3_failed_writes_at_every_step_never_leave_an_unrecoverable_state() {
    let reg = real_registry();
    let mut completed = false;
    for budget in 0..400 {
        let fs = MemoryFs::new();
        let clock = ManualClock::new(1_763_600_000_000);
        let mut s = Session::create(ROOT, fresh(), &reg, &fs, &clock, config(), seed(1)).unwrap();
        fs.fail_after(Some(budget));
        let result = script(&mut s, &clock);
        fs.fail_after(None);
        if let Err(e) = &result {
            assert_eq!(e.code(), "storage", "budget {budget}: {e}");
        }
        // The session's state is what the disk says, and work goes on.
        let state = s.document().clone();
        let proposals: Vec<Id128> = s.proposals().iter().map(|p| p.id).collect();
        let (r, _) = Session::open(ROOT, &reg, &fs, &clock, config(), seed(2)).unwrap();
        assert_eq!(r.document(), &state, "budget {budget}");
        assert_eq!(r.proposals().iter().map(|p| p.id).collect::<Vec<_>>(), proposals, "budget {budget}");
        s.execute(set(HUMAN, "face.jaw.width", -1.0)).unwrap();
        let state = s.document().clone();
        let (r, _) = Session::open(ROOT, &reg, &fs, &clock, config(), seed(3)).unwrap();
        assert_eq!(r.document(), &state, "budget {budget} after recovery");
        if result.is_ok() {
            // Every write of the script was a failure point once.
            assert!(budget > 30, "only {budget} writes");
            completed = true;
            break;
        }
    }
    assert!(completed, "the script never ran to the end");
}

#[test]
fn cmd03_a_gesture_is_one_undo_step_and_one_event() {
    let reg = real_registry();
    let fs = MemoryFs::new();
    let clock = ManualClock::new(1_763_600_000_000);
    let mut s = Session::create(ROOT, fresh(), &reg, &fs, &clock, SessionConfig::DEFAULT, seed(1)).unwrap();
    for n in 1..=10 {
        clock.advance_millis(16);
        let o = s.execute(set(HUMAN, "face.jaw.width", f64::from(n) / 10.0).in_gesture()).unwrap();
        assert_eq!(o.coalesced, n > 1);
        // Every frame reaches the document at once.
        assert!((value(&s, "face.jaw.width") - f64::from(n) / 10.0).abs() < 1e-12);
    }
    assert_eq!(s.events().last_seq(), 0, "no event while the slider moves");
    let sealed = s.end_gesture().unwrap().unwrap();
    assert_eq!(s.history().entries().len(), 1);
    assert_eq!(s.events().last_seq(), 1);
    assert!(
        sealed.summary().contains("face.jaw.width") && sealed.summary().contains("(+1.000)"),
        "{}",
        sealed.summary()
    );
    s.undo(HUMAN, &Scope::Any).unwrap();
    assert_eq!(s.document(), &fresh());
    s.redo(HUMAN, &Scope::Any).unwrap();
    // A pause longer than the threshold ends the step; a new drag is a new step.
    s.execute(set(HUMAN, "face.jaw.width", 0.5).in_gesture()).unwrap();
    clock.advance_millis(SessionConfig::DEFAULT.coalesce_pause_ms + 1);
    assert_eq!(s.tick().unwrap(), None);
    s.execute(set(HUMAN, "face.jaw.width", 0.25).in_gesture()).unwrap();
    // Another author or another target does not coalesce.
    s.execute(set(AGENT, "face.eyes.size", 0.3).in_gesture()).unwrap();
    s.execute(set(AGENT, "face.nose.length", 0.3).in_gesture()).unwrap();
    s.end_gesture().unwrap();
    assert_eq!(s.history().entries().len(), 5);
    // The journal batched the frames and replays to the same state.
    let (r, _) = Session::open(ROOT, &reg, &fs, &clock, config(), seed(2)).unwrap();
    assert_eq!(r.document(), s.document());
    assert_eq!(r.history().entries(), s.history().entries());
}

#[test]
fn cmd04_undo_by_author_reverts_or_reports_the_conflict() {
    let reg = real_registry();
    let fs = MemoryFs::new();
    let clock = ManualClock::new(1_763_600_000_000);
    let mut s = Session::new(fresh(), &reg, &fs, &clock, config(), seed(1));
    s.execute(set(HUMAN, "face.jaw.width", 0.5)).unwrap();
    let agent_cmd = s.execute(set(AGENT, "face.nose.length", 1.0)).unwrap().command;
    s.execute(set(HUMAN, "face.eyes.size", 0.25)).unwrap();
    // "Undo the agent's last": no later command touches the nose, so it is reverted on its own.
    let o = s.undo(HUMAN, &Scope::Author(AGENT.into())).unwrap();
    assert_eq!(o.kind, "undo");
    assert!(value(&s, "face.nose.length").abs() < 1e-12);
    assert!((value(&s, "face.eyes.size") - 0.25).abs() < 1e-12);
    assert_eq!(s.history().entries().last().unwrap().reverts, Some(agent_cmd));
    // Nothing of the agent's is left to undo; the revert itself is undoable.
    assert_eq!(s.undo(HUMAN, &Scope::Author(AGENT.into())).unwrap_err().code(), "nothing_to_do");
    s.undo(HUMAN, &Scope::Any).unwrap();
    assert!((value(&s, "face.nose.length") - 1.0).abs() < 1e-12);
    // A later human command on the same target makes it a conflict, reported with both ids.
    let later = s.execute(set(HUMAN, "face.nose.length", 1.5)).unwrap().command;
    match s.undo(HUMAN, &Scope::Author(AGENT.into())) {
        Err(CommandError::Rejected(Rejection::UndoConflict { command, later: l, targets })) => {
            assert_eq!(command, agent_cmd);
            assert_eq!(l, vec![later]);
            assert_eq!(targets, vec!["params face.nose.length".to_owned()]);
        }
        other => panic!("{other:?}"),
    }
    // "Undo my last" for the human is their newest command; redo by scope checks the author.
    s.undo(HUMAN, &Scope::Author(HUMAN.into())).unwrap();
    assert!((value(&s, "face.nose.length") - 1.0).abs() < 1e-12);
    assert_eq!(s.redo(AGENT, &Scope::Author(AGENT.into())).unwrap_err().code(), "not_yours");
    s.redo(HUMAN, &Scope::Author(HUMAN.into())).unwrap();
    assert_eq!(s.undo("nobody", &Scope::Any).unwrap_err().code(), "unknown_author");
}

#[test]
fn ac_wp08_2_human_edit_protection_policy() {
    let reg = real_registry();
    let fs = MemoryFs::new();
    let clock = ManualClock::new(1_763_600_000_000);
    let mut s = Session::new(fresh(), &reg, &fs, &clock, config(), seed(1));
    s.execute(set(HUMAN, "face.jaw.width", 0.5)).unwrap();
    let layer = s.execute(add_layer(1, "chin")).unwrap();
    let layer_id = s.document().sculpt_layers[0].id;
    assert_eq!(layer.changes.len(), 1);
    // [LOCK-02] the agent may not change what the human changed last ...
    let err = s.execute(set(AGENT, "face.jaw.width", 0.7)).unwrap_err();
    assert_eq!(err.code(), "human_protected");
    assert!(err.to_string().contains("ask the human"), "{err}");
    let mut fields = Object::new();
    fields.insert("opacity", Value::from(Number::Float(0.5)));
    let modify = Action::ModifyObject { kind: ObjectKind::SculptLayer, id: layer_id, fields };
    assert_eq!(s.execute(Command::new(AGENT, modify.clone())).unwrap_err().code(), "human_protected");
    // ... nor undo the human's command, nor change the policy.
    assert_eq!(s.undo(AGENT, &Scope::Any).unwrap_err().code(), "human_protected");
    let off = Action::SetSettings { protect_human_edits: false };
    assert_eq!(s.execute(Command::new(AGENT, off.clone())).unwrap_err().code(), "human_only");
    // In a proposal the same change is allowed and flagged for review.
    let p = s.open_proposal(AGENT, "Jaw").unwrap();
    let o = s.execute_in(p, set(AGENT, "face.jaw.width", 0.7)).unwrap();
    assert!(o.warnings.iter().any(|w| w.code == WarningCode::NeedsReview), "{:?}", o.warnings);
    // Locks still apply everywhere.
    s.execute(Command::new(HUMAN, Action::SetLock { target: LockTarget::Region("chin".into()), locked: true }))
        .unwrap();
    assert_eq!(s.execute_in(p, Command::new(AGENT, modify.clone())).unwrap_err().code(), "region_locked");
    s.execute(Command::new(HUMAN, Action::SetLock { target: LockTarget::Region("chin".into()), locked: false }))
        .unwrap();
    // The human switches the policy off for the project: the agent may now edit directly.
    s.execute(Command::new(HUMAN, off)).unwrap();
    s.execute(set(AGENT, "face.jaw.width", 0.7)).unwrap();
    s.execute(Command::new(AGENT, modify)).unwrap();
    assert!((value(&s, "face.jaw.width") - 0.7).abs() < 1e-12);
}

/// `[LOCK-02]` `[PROP-01]` (todo 6): a human edit made on the main document after a proposal was
/// opened is not in the proposal's branch, yet an agent change of the same parameter inside the
/// proposal must still be flagged for review.
#[test]
fn lock02_proposal_sees_human_edits_made_after_it_was_opened() {
    let reg = real_registry();
    let fs = MemoryFs::new();
    let clock = ManualClock::new(1_763_600_000_000);
    let mut s = Session::new(fresh(), &reg, &fs, &clock, config(), seed(1));
    let p = s.open_proposal(AGENT, "Jaw").unwrap();
    s.execute(set(HUMAN, "face.jaw.width", 0.5)).unwrap();
    // The branch still has the old value and no human author for the parameter.
    assert!(!s.proposal(p).unwrap().doc.param_authors.contains_key("face.jaw.width"));
    let review = |o: &Outcome, id: &str| {
        o.warnings.iter().filter(|w| w.code == WarningCode::NeedsReview && w.target == id).count()
    };
    let o = s.execute_in(p, set(AGENT, "face.jaw.width", 0.7)).unwrap();
    assert_eq!(review(&o, "face.jaw.width"), 1, "{:?}", o.warnings);
    assert!(o.warnings.iter().find(|w| w.code == WarningCode::NeedsReview).unwrap().message.contains(HUMAN));
    // A second agent change in the branch: the branch now says the agent changed it last, the
    // main document still says the human did.
    let o = s.execute_in(p, set(AGENT, "face.jaw.width", 0.8)).unwrap();
    assert_eq!(review(&o, "face.jaw.width"), 1, "{:?}", o.warnings);
    // A parameter no human touched raises nothing.
    let o = s.execute_in(p, set(AGENT, "face.nose.length", 0.3)).unwrap();
    assert!(o.warnings.iter().all(|w| w.code != WarningCode::NeedsReview), "{:?}", o.warnings);
    // A human edit before the proposal was opened is warned once, not twice.
    s.execute(set(HUMAN, "face.eyes.size", 0.2)).unwrap();
    let q = s.open_proposal(AGENT, "Eyes").unwrap();
    let o = s.execute_in(q, set(AGENT, "face.eyes.size", 0.4)).unwrap();
    assert_eq!(review(&o, "face.eyes.size"), 1, "{:?}", o.warnings);
}

#[test]
fn ac_wp08_2_proposal_accept_partial_accept_discard_and_conflicts() {
    let reg = real_registry();
    let fs = MemoryFs::new();
    let clock = ManualClock::new(1_763_600_000_000);
    let mut s = Session::create(ROOT, fresh(), &reg, &fs, &clock, config(), seed(1)).unwrap();
    let p = s.open_proposal(AGENT, "Nose and eyes").unwrap();
    let file = format!("{ROOT}/proposals/{}.json", p.to_hex().as_str());
    assert!(fs.exists(&file));
    let c1 = s.execute_in(p, set(AGENT, "face.nose.length", 0.5)).unwrap().command;
    let c2 = s.execute_in(p, set(AGENT, "face.nose.length", 0.75)).unwrap().command;
    let c3 = s.execute_in(p, set(AGENT, "face.eyes.size", 0.5)).unwrap().command;
    // [PROP-01] the branch shows the work; the main document does not have it.
    assert!(
        (s.proposal(p).unwrap().doc.param(&reg, "face.nose.length").unwrap().as_scalar().unwrap() - 0.75).abs() < 1e-12
    );
    assert!(value(&s, "face.nose.length").abs() < 1e-12);
    assert_eq!(s.proposal(p).unwrap().command_ids(), vec![c1, c2, c3]);
    // Only the proposal's author adds to it, only a human accepts.
    assert_eq!(s.execute_in(p, set(HUMAN, "face.nose.length", 0.1)).unwrap_err().code(), "not_proposal_author");
    assert_eq!(s.accept(p, AGENT, None).unwrap_err().code(), "human_only");
    // [PROP-03] the human keeps working; the eyes now conflict.
    s.execute(set(HUMAN, "face.eyes.size", -0.5)).unwrap();
    match s.accept(p, HUMAN, None) {
        Err(CommandError::Rejected(Rejection::ProposalConflict { conflicts })) => {
            assert_eq!(conflicts, vec![(c3, vec!["params face.eyes.size".to_owned()])]);
        }
        other => panic!("{other:?}"),
    }
    // [PROP-02] a selection must include what it builds on.
    match s.accept(p, HUMAN, Some(&[c2])) {
        Err(CommandError::Rejected(Rejection::ProposalDependency { command, needs })) => {
            assert_eq!((command, needs), (c2, vec![c1]));
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(s.accept(p, HUMAN, Some(&[Id128::from_u128(5)])).unwrap_err().code(), "invalid");
    assert!(s.proposal(p).is_some(), "refused accepts change nothing");
    let done = s.accept(p, HUMAN, Some(&[c1, c2])).unwrap();
    assert_eq!((done.accepted.clone(), done.dropped.clone()), (vec![c1, c2], vec![c3]));
    assert!((value(&s, "face.nose.length") - 0.75).abs() < 1e-12);
    assert!((value(&s, "face.eyes.size") + 0.5).abs() < 1e-12);
    // Merged with the agent as author; the proposal is closed and its file gone.
    let authors: Vec<&str> = s.history().applied().iter().map(|e| e.author.as_str()).collect();
    assert_eq!(authors, vec![HUMAN, AGENT, AGENT]);
    assert_eq!(s.document().param_authors.get("face.nose.length").map(String::as_str), Some(AGENT));
    assert!(s.proposal(p).is_none());
    assert!(!fs.exists(&file));
    // The merged commands undo one by one like any other.
    s.undo(HUMAN, &Scope::Any).unwrap();
    assert!((value(&s, "face.nose.length") - 0.5).abs() < 1e-12);
    s.redo(HUMAN, &Scope::Any).unwrap();
    // Discarding: the human, or the proposal's own author.
    let q = s.open_proposal(AGENT, "Discard me").unwrap();
    s.execute_in(q, set(AGENT, "face.jaw.width", 1.0)).unwrap();
    s.undo_in(q, AGENT).unwrap();
    assert!(s.proposal(q).unwrap().entries.is_empty());
    assert_eq!(s.proposal(q).unwrap().doc, s.proposal(q).unwrap().base);
    s.discard(q, HUMAN).unwrap();
    assert!(s.proposal(q).is_none());
    assert_eq!(s.discard(q, HUMAN).unwrap_err().code(), "unknown_proposal");
    // [PROP-04] an open proposal survives a restart.
    let r = s.open_proposal(AGENT, "Survivor").unwrap();
    s.execute_in(r, set(AGENT, "face.jaw.width", 1.0)).unwrap();
    let kept = s.proposal(r).unwrap().clone();
    let (back, report) = Session::open(ROOT, &reg, &fs, &clock, config(), seed(2)).unwrap();
    assert!(report.damaged_proposals.is_empty());
    assert_eq!(back.proposals(), &[kept]);
    assert_eq!(back.document(), s.document());
}

/// A fixed script for the determinism test.
fn event_script(s: &mut Session<'_>, clock: &ManualClock) -> Vec<String> {
    s.execute(set(HUMAN, "face.jaw.width", 0.5)).unwrap();
    for n in 1..=5 {
        clock.advance_millis(16);
        s.execute(set(HUMAN, "face.nose.length", f64::from(n) / 10.0).in_gesture()).unwrap();
    }
    s.select(HUMAN, Some("nasal-tip"), Some("nose.tip")).unwrap();
    s.execute(Command::new(HUMAN, Action::PinLandmark { landmark: "nose.tip".into(), position: [0.0, 1.6, 0.12] }))
        .unwrap();
    s.execute(Command::new(HUMAN, Action::CreateSnapshot { id: Id128::NIL, name: "A".into(), state: Hash128::ZERO }))
        .unwrap();
    let p = s.open_proposal(AGENT, "Eyes").unwrap();
    s.execute_in(p, set(AGENT, "face.eyes.size", 0.5)).unwrap();
    s.accept(p, HUMAN, None).unwrap();
    let q = s.open_proposal(AGENT, "Other").unwrap();
    s.discard(q, HUMAN).unwrap();
    s.undo(HUMAN, &Scope::Any).unwrap();
    s.redo(HUMAN, &Scope::Any).unwrap();
    s.save().unwrap();
    s.events().iter().map(|e| json::to_string(&e.to_json()).unwrap()).collect()
}

#[test]
fn ac_wp08_2_event_sequences_are_deterministic() {
    let reg = real_registry();
    let run = || {
        let fs = MemoryFs::new();
        let clock = ManualClock::new(1_763_600_000_000);
        let mut s = Session::create(ROOT, fresh(), &reg, &fs, &clock, config(), seed(7)).unwrap();
        event_script(&mut s, &clock)
    };
    let first = run();
    assert_eq!(first, run());
    let types: Vec<String> = first
        .iter()
        .map(|e| json::parse_str(e).unwrap().as_object().unwrap().get("type").unwrap().as_str().unwrap().to_owned())
        .collect();
    assert_eq!(
        types,
        [
            "command_committed",
            "command_committed",
            "selection_changed",
            "lock_changed",
            "snapshot_changed",
            "proposal_opened",
            "command_committed",
            "proposal_accepted",
            "proposal_opened",
            "proposal_discarded",
            "command_undone",
            "command_redone",
            "saved"
        ]
    );
    assert!(
        first[1].contains("\"summary\":\"human.artist set_params; face.nose.length ") && first[1].contains("(+0.500)"),
        "{}",
        first[1]
    );
    // [NOTE-02] sequence numbers are consecutive from 1; "since N" catches up.
    for (n, e) in first.iter().enumerate() {
        assert!(e.starts_with(&format!("{{\"seq\":{},", n + 1)), "{e}");
    }
    let fs = MemoryFs::new();
    let clock = ManualClock::new(1_763_600_000_000);
    let small = SessionConfig { event_capacity: 4, ..config() };
    let mut s = Session::create(ROOT, fresh(), &reg, &fs, &clock, small, seed(7)).unwrap();
    event_script(&mut s, &clock);
    match s.events().since(11) {
        Since::Events(events) => assert_eq!(events.iter().map(|e| e.seq).collect::<Vec<_>>(), vec![12, 13]),
        Since::Gap { .. } => panic!("no gap expected"),
    }
    assert_eq!(s.events().since(2), Since::Gap { oldest: 10 });
}

struct JawScore;

impl Plausibility for JawScore {
    fn score(&self, doc: &Document, registry: &Registry) -> f64 {
        1.0 - doc.param(registry, "face.jaw.width").ok().and_then(ParamValue::as_scalar).unwrap_or(0.0).abs() / 3.0
    }
}

#[test]
fn cmd06_outcomes_carry_changes_warnings_and_plausibility() {
    let reg = real_registry();
    let fs = MemoryFs::new();
    let clock = ManualClock::new(1_763_600_000_000);
    let scorer = JawScore;
    let mut s = Session::new(fresh(), &reg, &fs, &clock, config(), seed(1)).with_scorer(&scorer);
    let o = s.execute(set(HUMAN, "face.jaw.width", 1.5).labelled("Wide jaw")).unwrap();
    assert_eq!(o.label, "Wide jaw");
    assert_eq!(o.plausibility, Some([1.0, 0.5]));
    assert!(o.warnings.iter().any(|w| w.code == WarningCode::PlausibilityDrop));
    let json = json::to_string(&o.to_json()).unwrap();
    assert!(json.contains("\"plausibility\":[1,0.5]") || json.contains("\"plausibility\":[1.0,0.5]"), "{json}");
    // [CMD-05] out of range: rejected with the reason, or clamped and said so.
    assert_eq!(s.execute(set(HUMAN, "face.jaw.width", 9.0)).unwrap_err().code(), "out_of_range");
    let clamp = Action::SetParams {
        values: vec![("face.jaw.width".into(), ParamValue::Scalar(9.0))],
        relative: false,
        policy: RangePolicy::Clamp,
    };
    let o = s.execute(Command::new(HUMAN, clamp)).unwrap();
    assert!(o.warnings.iter().any(|w| w.code == WarningCode::Clamped));
    assert!((value(&s, "face.jaw.width") - 3.0).abs() < 1e-12);
    // A command that changes nothing leaves no history step.
    let n = s.history().entries().len();
    s.execute(set(HUMAN, "face.jaw.width", 3.0)).unwrap();
    assert_eq!(s.history().entries().len(), n);
    assert_eq!(s.save().unwrap_err().code(), "invalid", "no project");
}

#[test]
fn save03_tick_autosaves_every_n_changes_and_on_idle_and_rotates() {
    let reg = real_registry();
    let fs = MemoryFs::new();
    let clock = ManualClock::new(1_763_600_000_000);
    let cfg = SessionConfig { autosave_every: 3, autosave_idle_ms: 5000, autosave_keep: 2, ..config() };
    let mut s = Session::create(ROOT, fresh(), &reg, &fs, &clock, cfg, seed(1)).unwrap();
    assert_eq!(s.tick().unwrap(), None, "nothing changed yet");
    for n in 1..=3 {
        s.execute(set(HUMAN, "face.jaw.width", f64::from(n) / 10.0)).unwrap();
        clock.advance_millis(10);
        assert_eq!(s.tick().unwrap().is_some(), n == 3);
    }
    s.execute(set(HUMAN, "face.jaw.width", 0.5)).unwrap();
    assert_eq!(s.tick().unwrap(), None);
    clock.advance_millis(5000);
    let path = s.tick().unwrap().unwrap();
    assert_eq!(
        Document::parse(&fs.read(&path).unwrap(), &reg, &crate::text::TextLimits::DEFAULT).unwrap(),
        *s.document()
    );
    s.execute(set(HUMAN, "face.jaw.width", 0.6)).unwrap();
    s.autosave().unwrap();
    let autosaves = Autosaves::new(&fs, s.project().unwrap()).list().unwrap();
    assert_eq!(autosaves.iter().map(|(n, _)| *n).collect::<Vec<_>>(), vec![2, 3]);
    // Autosaves never touch the user's save: the journal still recovers from it.
    let (r, report) = Session::open(ROOT, &reg, &fs, &clock, config(), seed(2)).unwrap();
    assert!(matches!(report.journal, JournalOutcome::Recovered { records: 5 }), "{:?}", report.journal);
    assert_eq!(r.document(), s.document());
    assert_eq!(r.events().iter().next().map(|e| e.kind.type_name()), Some("recovered"));
}

#[test]
fn save01_a_damaged_journal_is_kept_aside_and_the_save_opens() {
    let reg = real_registry();
    let fs = MemoryFs::new();
    let clock = ManualClock::new(1_763_600_000_000);
    let mut s = Session::create(ROOT, fresh(), &reg, &fs, &clock, config(), seed(1)).unwrap();
    s.execute(set(HUMAN, "face.jaw.width", 0.5)).unwrap();
    s.execute(set(HUMAN, "face.jaw.width", 0.7)).unwrap();
    // Swap the two records' order (both intact): replay cannot apply them.
    let text = String::from_utf8(fs.read(&journal_path()).unwrap()).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let records: Vec<JournalRecord> = crate::journal::read(text.as_bytes(), &JournalLimits::DEFAULT).unwrap().records;
    let mut swapped = format!("{}\n", lines[0]);
    for (n, r) in [&records[1], &records[0]].iter().enumerate() {
        swapped.push_str(&encode_record(&JournalRecord { seq: n as u64 + 1, payload: r.payload.clone() }).unwrap());
    }
    fs.write(&journal_path(), swapped.as_bytes()).unwrap();
    let (mut r, report) = Session::open(ROOT, &reg, &fs, &clock, config(), seed(2)).unwrap();
    match report.journal {
        JournalOutcome::Unreplayable { kept_at, .. } => assert!(fs.exists(&kept_at)),
        other => panic!("{other:?}"),
    }
    assert_eq!(r.document(), &fresh());
    r.execute(set(HUMAN, "face.jaw.width", 0.1)).unwrap();
    let (again, report) = Session::open(ROOT, &reg, &fs, &clock, config(), seed(3)).unwrap();
    assert!(matches!(report.journal, JournalOutcome::Recovered { records: 1 }));
    assert!((value(&again, "face.jaw.width") - 0.1).abs() < 1e-12);
}

// [PERF-R1] The test binary counts allocations so the drag test can assert zero.
#[global_allocator]
static ALLOCATOR: humanart_core::alloc::CountingAllocator = humanart_core::alloc::CountingAllocator;

/// Everything observable of a session: document, history, cursor, events.
fn observed(s: &Session<'_>) -> (Document, Vec<Entry>, usize, Vec<String>) {
    let events = s.events().iter().map(|e| json::to_string(&e.to_json()).unwrap()).collect();
    (s.document().clone(), s.history().entries().to_vec(), s.history().cursor(), events)
}

fn clamped_gesture(param: &str, v: f64) -> Command {
    Command::new(
        HUMAN,
        Action::SetParams {
            values: vec![(param.into(), ParamValue::Scalar(v))],
            relative: false,
            policy: RangePolicy::Clamp,
        },
    )
    .in_gesture()
}

#[test]
fn cmd03_perf_r1_drag_frames_equal_execute_and_fold_in_place() {
    // [CMD-03] [PERF-R1] `drag_scalar` gives the same document, history, events and journal as
    // executing the same frames as gesture commands, over random drags with values that land on
    // the default, outside the range and on the current value, interleaved agent commands, ticks,
    // undo and redo; most steady frames take the in-place path.
    let reg = real_registry();
    let mut folded = 0_u32;
    let mut frames = 0_u32;
    for case in 0..4_u64 {
        let cfg = SessionConfig { journal_interval_ms: 40, ..config() };
        let (fs_a, fs_b) = (MemoryFs::new(), MemoryFs::new());
        let (clock_a, clock_b) = (ManualClock::new(1_763_600_000_000), ManualClock::new(1_763_600_000_000));
        let mut a = Session::create(ROOT, fresh(), &reg, &fs_a, &clock_a, cfg, seed(1)).unwrap();
        let mut b = Session::create(ROOT, fresh(), &reg, &fs_b, &clock_b, cfg, seed(1)).unwrap();
        let mut rng = Rng::new(0xd7a6_0000 + case);
        for step in 0..60 {
            let param = PARAMS[rng.next_u32() as usize % PARAMS.len()];
            match rng.next_u32() % 8 {
                0..=4 => {
                    let mut v = round(rng.range_f64(-1.0, 1.0));
                    for _ in 0..=(rng.next_u32() % 30) {
                        match rng.next_u32() % 10 {
                            0 => v = 0.0,
                            1 => v = 5.0,
                            2 => {}
                            _ => v = round(v + rng.range_f64(-0.1, 0.1)),
                        }
                        let ms = u64::from(rng.next_u32() % 20);
                        clock_a.advance_millis(ms);
                        clock_b.advance_millis(ms);
                        let by_execute = a.execute(clamped_gesture(param, v));
                        let by_drag = b.drag_scalar(HUMAN, param, v);
                        frames += 1;
                        match (&by_execute, &by_drag) {
                            (Ok(_), Ok(None)) => folded += 1,
                            (Ok(x), Ok(Some(y))) => assert_eq!(x, y, "case {case} step {step}"),
                            (Err(x), Err(y)) => assert_eq!(x.to_string(), y.to_string()),
                            _ => panic!("case {case} step {step}: {by_execute:?} vs {by_drag:?}"),
                        }
                        if rng.next_u32().is_multiple_of(4) {
                            assert_eq!(a.tick().is_ok(), b.tick().is_ok());
                        }
                        assert_eq!(observed(&a), observed(&b), "case {case} step {step}");
                        assert_eq!(a.revision(), b.revision());
                    }
                    if !rng.next_u32().is_multiple_of(3) {
                        assert_eq!(a.end_gesture().unwrap(), b.end_gesture().unwrap());
                    }
                }
                5 => {
                    let v = round(rng.range_f64(-1.0, 1.0));
                    assert_eq!(a.execute(set(AGENT, param, v)).is_ok(), b.execute(set(AGENT, param, v)).is_ok());
                }
                6 => assert_eq!(a.undo(HUMAN, &Scope::Any).is_ok(), b.undo(HUMAN, &Scope::Any).is_ok()),
                _ => assert_eq!(a.redo(HUMAN, &Scope::Any).is_ok(), b.redo(HUMAN, &Scope::Any).is_ok()),
            }
            assert_eq!(observed(&a), observed(&b), "case {case} step {step}");
        }
        a.end_gesture().unwrap();
        b.end_gesture().unwrap();
        assert_eq!(fs_a.read(&journal_path()).unwrap(), fs_b.read(&journal_path()).unwrap(), "case {case}: journals");
        // The journal written through the fast path replays to the same state.
        let (r, _) = Session::open(ROOT, &reg, &fs_b, &clock_b, cfg, seed(2)).unwrap();
        assert_eq!(r.document(), b.document());
        assert_eq!(r.history().entries(), b.history().entries());
    }
    println!("{folded} of {frames} drag frames folded in place");
    assert!(folded * 2 > frames, "most frames take the in-place path: {folded} of {frames}");
}

/// Drags `param` for `frames` frames of 16 ms; returns (allocator calls, worst frame, frames
/// that allocated).
fn counted_drag(s: &mut Session<'_>, clock: &ManualClock, param: &str, frames: u32) -> (u64, u64, u32) {
    let mut worst = 0;
    let mut total = 0;
    let mut allocating = 0;
    for i in 0..frames {
        clock.advance_millis(16);
        let v = 0.3 + f64::from(i % 50) * 0.01;
        let mut folded = Err(CommandError::Rejected(Rejection::NothingToDo { what: "not run" }));
        let counts = humanart_core::alloc::count_in(|| folded = s.drag_scalar(HUMAN, param, v));
        assert!(matches!(folded, Ok(None)), "frame {i}: {folded:?}");
        let n = counts.allocations + counts.reallocations;
        worst = worst.max(n);
        total += n;
        allocating += u32::from(n > 0);
    }
    (total, worst, allocating)
}

#[test]
fn perf_r1_steady_drag_frames_do_not_allocate() {
    // [PERF-R1] [CMD-03] once a drag's gesture is open, its frames allocate nothing; with a
    // project, the batched journal frame written every `journal_interval_ms` allocates (it
    // encodes and appends one record), and so does the frame that starts the next batch.
    let reg = real_registry();
    let param = "face.nose.length";
    let fs = MemoryFs::new();
    let clock = ManualClock::new(1_763_600_000_000);
    // Without a project (nothing is journaled).
    let mut s = Session::new(fresh(), &reg, &fs, &clock, config(), seed(1));
    s.drag_scalar(HUMAN, param, 0.2).unwrap();
    let (total, worst, _) = counted_drag(&mut s, &clock, param, 300);
    println!("no project: 300 frames, {total} allocator calls (worst frame {worst})");
    assert_eq!(total, 0, "[PERF-R1] steady drag frames allocate nothing");
    // A project whose batch interval is longer than the drag.
    let cfg = SessionConfig { journal_interval_ms: 60_000, ..config() };
    let mut s = Session::create(ROOT, fresh(), &reg, &fs, &clock, cfg, seed(1)).unwrap();
    s.drag_scalar(HUMAN, param, 0.2).unwrap();
    s.drag_scalar(HUMAN, param, 0.25).unwrap();
    let (total, worst, _) = counted_drag(&mut s, &clock, param, 300);
    println!("project, no flush during the drag: 300 frames, {total} allocator calls (worst frame {worst})");
    assert_eq!(total, 0, "[PERF-R1] steady drag frames allocate nothing");
    // The default batching: the frames that flush or start a batch allocate, no other.
    let fs = MemoryFs::new();
    let interval = SessionConfig::DEFAULT.journal_interval_ms;
    let cfg = SessionConfig { journal_interval_ms: interval, ..config() };
    let mut s = Session::create(ROOT, fresh(), &reg, &fs, &clock, cfg, seed(1)).unwrap();
    s.drag_scalar(HUMAN, param, 0.2).unwrap();
    s.drag_scalar(HUMAN, param, 0.25).unwrap();
    let (total, worst, allocating) = counted_drag(&mut s, &clock, param, 300);
    let batches = u32::try_from(300 * 16 / interval).unwrap();
    println!(
        "project on MemoryFs, {interval} ms batches, 16 ms frames: 300 frames, {allocating} allocate, {total} \
         allocator calls (worst frame {worst}), {batches} batches"
    );
    // What is left is inside `MemoryFs::append` (it normalises the path into a new string and
    // grows the file's vector); the session's own part is measured below.
    assert!(allocating <= batches + 1, "{allocating} frames allocate, {batches} batches");
    assert!(s.end_gesture().unwrap().is_some());

    // The same with a file system whose journal append allocates nothing: after the first batch
    // (which sizes the reused line buffer) the batched journal writes allocate nothing either.
    let fs = MemoryFs::new();
    let quiet = QuietJournalFs { inner: &fs, path: format!("{ROOT}/journal/journal.log"), log: Mutex::new(Vec::new()) };
    let mut s = Session::create(ROOT, fresh(), &reg, &quiet, &clock, cfg, seed(1)).unwrap();
    s.drag_scalar(HUMAN, param, 0.2).unwrap();
    s.drag_scalar(HUMAN, param, 0.25).unwrap();
    let _ = counted_drag(&mut s, &clock, param, 20);
    quiet.log.lock().unwrap().reserve(1 << 20);
    let before = quiet.log.lock().unwrap().len();
    let (total, worst, allocating) = counted_drag(&mut s, &clock, param, 300);
    let written = quiet.log.lock().unwrap().len() - before;
    println!(
        "project, allocation-free append, {interval} ms batches: 300 frames, {total} allocator calls (worst frame \
         {worst}), {written} journal bytes written"
    );
    assert_eq!((total, allocating), (0, 0), "[PERF-R1] batched journal writes allocate nothing");
    assert!(written > 0);
    // The journal written this way replays to the dragged value.
    let last = value(&s, param);
    s.end_gesture().unwrap();
    drop(s);
    fs.append(&quiet.path, &quiet.log.lock().unwrap()).unwrap();
    let (again, _) = Session::open(ROOT, &reg, &fs, &clock, cfg, seed(2)).unwrap();
    assert!((value(&again, param) - last).abs() < 1e-12);
}

/// A file system whose appends to `path` go into a vector the test reserves (allocation-free),
/// everything else to `inner`.
struct QuietJournalFs<'m> {
    inner: &'m MemoryFs,
    path: String,
    log: Mutex<Vec<u8>>,
}

impl FileSystem for QuietJournalFs<'_> {
    fn read(&self, path: &str) -> Result<Vec<u8>, humanart_core::error::PlatformError> {
        self.inner.read(path)
    }
    fn write(&self, path: &str, bytes: &[u8]) -> Result<(), humanart_core::error::PlatformError> {
        self.inner.write(path, bytes)
    }
    fn append(&self, path: &str, bytes: &[u8]) -> Result<(), humanart_core::error::PlatformError> {
        if path == self.path {
            self.log.lock().unwrap().extend_from_slice(bytes);
            Ok(())
        } else {
            self.inner.append(path, bytes)
        }
    }
    fn rename(&self, from: &str, to: &str) -> Result<(), humanart_core::error::PlatformError> {
        self.inner.rename(from, to)
    }
    fn list(&self, dir: &str) -> Result<Vec<humanart_core::platform::DirEntry>, humanart_core::error::PlatformError> {
        self.inner.list(dir)
    }
    fn metadata(&self, path: &str) -> Result<humanart_core::platform::Metadata, humanart_core::error::PlatformError> {
        self.inner.metadata(path)
    }
    fn remove(&self, path: &str) -> Result<(), humanart_core::error::PlatformError> {
        self.inner.remove(path)
    }
    fn create_dir_all(&self, path: &str) -> Result<(), humanart_core::error::PlatformError> {
        self.inner.create_dir_all(path)
    }
}
