//! Agent presence `[APP-04]`: the agent's live status, its permission, and the one-click **pause**
//! and **revoke** controls.
//!
//! Every agent command reaches the document through [`AgentLink::submit`]. While the agent is
//! paused its commands are queued in arrival order without being applied; [`AgentLink::resume`]
//! applies the queue (each command validated as if it arrived then) and returns what happened to
//! each. Revoking drops the queue and refuses every later command until access is granted again.
//! The transport (the JSON-RPC server of WP-23) calls `submit`; the Agent panel shows
//! [`AgentLink::status`], the queue length and the recent commands of the provenance state.

use std::collections::VecDeque;

use humanart_doc::command::{Command, CommandError, Outcome};
use humanart_doc::session::Session;

/// The most commands queued while paused; later ones are refused.
pub const MAX_QUEUED: usize = 1024;

/// What the agent is doing, as it reports it (and as the link sees it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentStatus {
    /// No agent connected.
    Disconnected,
    /// Connected and waiting.
    Idle,
    /// The agent reports it is thinking.
    Thinking,
    /// A command is being applied.
    Applying,
}

impl AgentStatus {
    /// The text the Agent panel shows.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Disconnected => "Not connected",
            Self::Idle => "Idle",
            Self::Thinking => "Thinking",
            Self::Applying => "Applying",
        }
    }
}

/// What became of a submitted command.
#[derive(Debug)]
pub enum Submitted {
    /// Applied to the session.
    Applied(Box<Outcome>),
    /// Queued while the agent is paused (the queue length after it).
    Queued(usize),
    /// Refused: access is revoked or the queue is full.
    Refused(&'static str),
}

/// The agent link (see the module documentation).
#[derive(Debug)]
pub struct AgentLink {
    status: AgentStatus,
    name: Option<String>,
    paused: bool,
    revoked: bool,
    queue: VecDeque<Command>,
}

impl Default for AgentLink {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentLink {
    /// A link with no agent connected.
    pub fn new() -> Self {
        Self { status: AgentStatus::Disconnected, name: None, paused: false, revoked: false, queue: VecDeque::new() }
    }

    /// An agent connected under `name`.
    pub fn connect(&mut self, name: &str) {
        name.clone_into(self.name.get_or_insert_with(String::new));
        self.status = AgentStatus::Idle;
    }

    /// The agent disconnected.
    pub fn disconnect(&mut self) {
        self.status = AgentStatus::Disconnected;
        self.name = None;
    }

    /// The agent reports it is thinking (`true`) or idle again.
    pub fn set_thinking(&mut self, thinking: bool) {
        if self.status != AgentStatus::Disconnected {
            self.status = if thinking { AgentStatus::Thinking } else { AgentStatus::Idle };
        }
    }

    /// The live status.
    pub fn status(&self) -> AgentStatus {
        self.status
    }

    /// The connected agent's name.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Whether the agent is paused.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Whether the agent's access is revoked.
    pub fn is_revoked(&self) -> bool {
        self.revoked
    }

    /// Commands waiting for [`AgentLink::resume`].
    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    /// Hands an agent command to the session, or queues or refuses it (see the module
    /// documentation).
    pub fn submit(&mut self, session: &mut Session<'_>, cmd: Command) -> Result<Submitted, CommandError> {
        if self.revoked {
            return Ok(Submitted::Refused("the agent's access is revoked"));
        }
        if self.paused {
            if self.queue.len() >= MAX_QUEUED {
                return Ok(Submitted::Refused("the paused agent's queue is full"));
            }
            self.queue.push_back(cmd);
            return Ok(Submitted::Queued(self.queue.len()));
        }
        let before = self.status;
        self.status = AgentStatus::Applying;
        let result = session.execute(cmd);
        self.status = if before == AgentStatus::Disconnected { AgentStatus::Idle } else { before };
        result.map(|o| Submitted::Applied(Box::new(o)))
    }

    /// Pauses the agent: its commands queue up without being applied.
    pub fn pause(&mut self) {
        self.paused = true;
    }

    /// Resumes the agent and applies the queued commands in order; returns each one's result.
    pub fn resume(&mut self, session: &mut Session<'_>) -> Vec<Result<Outcome, CommandError>> {
        self.paused = false;
        let mut results = Vec::with_capacity(self.queue.len());
        while let Some(cmd) = self.queue.pop_front() {
            results.push(session.execute(cmd));
        }
        results
    }

    /// Revokes the agent's access: drops the queue and refuses later commands. Returns how many
    /// queued commands were dropped.
    pub fn revoke(&mut self) -> usize {
        self.revoked = true;
        self.paused = false;
        let dropped = self.queue.len();
        self.queue.clear();
        dropped
    }

    /// Grants access again after a revoke.
    pub fn grant(&mut self) {
        self.revoked = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{AGENT, Fixture};

    #[test]
    fn app_04_pause_queues_and_resume_applies_in_order() {
        let f = Fixture::new();
        let mut s = f.session();
        let (a, _) = f.two_scalar_params_with_regions();
        let mut link = AgentLink::new();
        link.connect("test agent");
        assert_eq!(link.status(), AgentStatus::Idle);
        link.set_thinking(true);
        assert_eq!(link.status(), AgentStatus::Thinking);
        link.pause();
        for v in [0.2, 0.3] {
            assert!(matches!(link.submit(&mut s, Fixture::set(AGENT, &a, v)).unwrap(), Submitted::Queued(_)));
        }
        assert_eq!(link.queued(), 2);
        assert!(!s.document().params.contains_key(&a), "[APP-04] pausing queues commands without applying them");
        let results = link.resume(&mut s);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(Result::is_ok));
        assert_eq!(s.document().params.get(&a), Some(&humanart_doc::registry::ParamValue::Scalar(0.3)));
    }

    #[test]
    fn app_04_revoke_drops_the_queue_and_refuses() {
        let f = Fixture::new();
        let mut s = f.session();
        let (a, _) = f.two_scalar_params_with_regions();
        let mut link = AgentLink::new();
        link.connect("test agent");
        link.pause();
        link.submit(&mut s, Fixture::set(AGENT, &a, 0.2)).unwrap();
        assert_eq!(link.revoke(), 1);
        assert!(matches!(link.submit(&mut s, Fixture::set(AGENT, &a, 0.4)).unwrap(), Submitted::Refused(_)));
        assert!(!s.document().params.contains_key(&a));
        link.grant();
        assert!(matches!(link.submit(&mut s, Fixture::set(AGENT, &a, 0.4)).unwrap(), Submitted::Applied(_)));
    }
}
