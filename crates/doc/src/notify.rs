//! Notification events `[NOTE-01]`, `[NOTE-02]`: what the human (or the agent) did, with sequence
//! numbers so a client can catch up deterministically.
//!
//! Every committed command, undo and redo, proposal state change, selection change, save and
//! recovery emits one [`Event`]. Lock and pin changes and snapshots are commands; their events have
//! the types `lock_changed` and `snapshot_changed` instead of `command_committed` so a client can
//! filter on them. A gesture emits one event when it is sealed (released), not one per frame.
//!
//! The JSON schema ([`Event::to_json`]) is stable: `{"seq": n, "type": t, ...}` with, per type:
//!
//! - `command_committed`, `lock_changed`, `snapshot_changed`, `command_undone`, `command_redone`:
//!   `"outcome"` (the [`Outcome`] JSON) and `"summary"` (one line, e.g. `human.artist set_params;
//!   face.nose.bridge_height 0.35 -> 0.47 (+0.120)`);
//! - `proposal_opened`: `"proposal"`, `"author"`, `"name"`;
//! - `proposal_accepted`: `"proposal"`, `"accepted"` (command ids), `"dropped"` (command ids);
//! - `proposal_discarded`: `"proposal"`;
//! - `selection_changed`: `"author"`, `"region"`, `"landmark"` (null when none);
//! - `saved`: `"hash"` (the saved text's hash); `recovered`: `"records"` (journal records replayed).
//!
//! The log keeps the most recent events up to its capacity; asking for events older than that
//! reports a gap ([`Since::Gap`]) and the client re-reads the document instead.

use std::collections::VecDeque;

use humanart_core::hash::Hash128;
use humanart_core::id::Id128;
use humanart_io::json::{Number, Object, Value};

use crate::command::Outcome;

/// What happened.
#[derive(Clone, Debug, PartialEq)]
pub enum EventKind {
    /// A command was committed (to the main document or a proposal).
    Committed(Outcome),
    /// A command was undone (plain undo; a selective undo is a committed `undo` command).
    Undone(Outcome),
    /// A command was redone.
    Redone(Outcome),
    /// An agent opened a proposal.
    ProposalOpened {
        /// The proposal.
        proposal: Id128,
        /// Its author.
        author: String,
        /// Its name.
        name: String,
    },
    /// The human accepted a proposal, wholly or in part.
    ProposalAccepted {
        /// The proposal.
        proposal: Id128,
        /// The commands merged into the main history.
        accepted: Vec<Id128>,
        /// The commands left out.
        dropped: Vec<Id128>,
    },
    /// The human discarded a proposal.
    ProposalDiscarded {
        /// The proposal.
        proposal: Id128,
    },
    /// The region or landmark someone works on changed.
    SelectionChanged {
        /// Who.
        author: String,
        /// The region, if any.
        region: Option<String>,
        /// The landmark, if any.
        landmark: Option<String>,
    },
    /// The document was saved.
    Saved {
        /// The hash of the saved text.
        hash: Hash128,
    },
    /// Journal records were replayed after an unclean shutdown `[SAVE-01]`.
    Recovered {
        /// How many.
        records: usize,
    },
}

const LOCK_KINDS: [&str; 4] = ["set_lock", "set_object_lock", "pin_landmark", "unpin_landmark"];
const SNAPSHOT_KINDS: [&str; 2] = ["create_snapshot", "restore_snapshot"];

impl EventKind {
    /// The `type` member of the JSON form.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Committed(o) if LOCK_KINDS.contains(&o.kind.as_str()) => "lock_changed",
            Self::Committed(o) if SNAPSHOT_KINDS.contains(&o.kind.as_str()) => "snapshot_changed",
            Self::Committed(_) => "command_committed",
            Self::Undone(_) => "command_undone",
            Self::Redone(_) => "command_redone",
            Self::ProposalOpened { .. } => "proposal_opened",
            Self::ProposalAccepted { .. } => "proposal_accepted",
            Self::ProposalDiscarded { .. } => "proposal_discarded",
            Self::SelectionChanged { .. } => "selection_changed",
            Self::Saved { .. } => "saved",
            Self::Recovered { .. } => "recovered",
        }
    }
}

/// One event.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    /// The sequence number: 1 for the first event of a session, then consecutive.
    pub seq: u64,
    /// What happened.
    pub kind: EventKind,
}

fn ids(list: &[Id128]) -> Value {
    Value::Array(list.iter().map(|i| Value::from(i.to_hex().as_str())).collect())
}

fn opt(s: Option<&str>) -> Value {
    s.map_or(Value::Null, Value::from)
}

impl Event {
    /// The JSON form (see the module documentation for the schema).
    pub fn to_json(&self) -> Value {
        let mut o = Object::with_capacity(5);
        o.insert("seq", Value::from(Number::from(self.seq)));
        o.insert("type", self.kind.type_name());
        match &self.kind {
            EventKind::Committed(outcome) | EventKind::Undone(outcome) | EventKind::Redone(outcome) => {
                o.insert("outcome", outcome.to_json());
                o.insert("summary", outcome.summary());
            }
            EventKind::ProposalOpened { proposal, author, name } => {
                o.insert("proposal", proposal.to_hex().as_str());
                o.insert("author", author.as_str());
                o.insert("name", name.as_str());
            }
            EventKind::ProposalAccepted { proposal, accepted, dropped } => {
                o.insert("proposal", proposal.to_hex().as_str());
                o.insert("accepted", ids(accepted));
                o.insert("dropped", ids(dropped));
            }
            EventKind::ProposalDiscarded { proposal } => {
                o.insert("proposal", proposal.to_hex().as_str());
            }
            EventKind::SelectionChanged { author, region, landmark } => {
                o.insert("author", author.as_str());
                o.insert("region", opt(region.as_deref()));
                o.insert("landmark", opt(landmark.as_deref()));
            }
            EventKind::Saved { hash } => {
                o.insert("hash", hash.to_hex().as_str());
            }
            EventKind::Recovered { records } => {
                o.insert("records", Value::from(Number::from(*records)));
            }
        }
        Value::Object(o)
    }
}

/// The answer to "events since N".
#[derive(Debug, PartialEq)]
pub enum Since<'a> {
    /// The events after N, oldest first (maybe none).
    Events(Vec<&'a Event>),
    /// Events after N are no longer kept: the oldest kept one has this sequence number.
    Gap {
        /// The oldest sequence number still available.
        oldest: u64,
    },
}

/// The event log of a session.
#[derive(Clone, Debug, PartialEq)]
pub struct EventLog {
    events: VecDeque<Event>,
    next: u64,
    capacity: usize,
}

impl EventLog {
    /// An empty log keeping at most `capacity` events (at least 1).
    pub fn new(capacity: usize) -> Self {
        Self { events: VecDeque::new(), next: 1, capacity: capacity.max(1) }
    }

    /// Appends an event and returns its sequence number.
    pub fn push(&mut self, kind: EventKind) -> u64 {
        let seq = self.next;
        self.next += 1;
        if self.events.len() == self.capacity {
            self.events.pop_front();
        }
        self.events.push_back(Event { seq, kind });
        seq
    }

    /// The sequence number of the latest event (0: none yet).
    pub fn last_seq(&self) -> u64 {
        self.next - 1
    }

    /// The events with sequence numbers above `after` `[NOTE-02]`.
    pub fn since(&self, after: u64) -> Since<'_> {
        let oldest = self.events.front().map_or(self.next, |e| e.seq);
        if after.saturating_add(1) < oldest {
            return Since::Gap { oldest };
        }
        Since::Events(self.events.iter().filter(|e| e.seq > after).collect())
    }

    /// Every kept event, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &Event> {
        self.events.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note02_since_returns_later_events_or_a_gap() {
        let mut log = EventLog::new(3);
        assert_eq!(log.since(0), Since::Events(vec![]));
        for n in 0..5 {
            log.push(EventKind::Recovered { records: n });
        }
        assert_eq!(log.last_seq(), 5);
        match log.since(3) {
            Since::Events(events) => assert_eq!(events.iter().map(|e| e.seq).collect::<Vec<_>>(), vec![4, 5]),
            Since::Gap { .. } => panic!("no gap expected"),
        }
        assert_eq!(log.since(2), Since::Events(log.iter().collect()));
        assert_eq!(log.since(1), Since::Gap { oldest: 3 });
        assert_eq!(log.since(9), Since::Events(vec![]));
        let json = log.iter().next().unwrap().to_json();
        assert_eq!(humanart_io::json::to_string(&json).unwrap(), r#"{"seq":3,"type":"recovered","records":2}"#);
    }
}
