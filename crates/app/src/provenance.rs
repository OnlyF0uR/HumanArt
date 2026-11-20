//! The provenance UI state `[APP-03]`: who last set each parameter (the badge), which changes
//! flash, and what changed since the human last looked.
//!
//! The badge is read from the document itself (`param_authors`, whose author ids name a human or
//! an agent), so it is right after an undo, a reload or a crash recovery. Flashes and the "changed
//! since I last looked" set come from the session's event log `[NOTE-01]`: [`Provenance::update`]
//! reads the events after the last one it saw. A change committed, undone or redone by an agent
//! flashes its parameters in the Properties panel and its regions in the viewport for the
//! configured duration; human changes do not flash (the human made them). Every change, by anyone,
//! enters the "changed since I last looked" set until [`Provenance::mark_looked`].

use std::collections::{BTreeMap, BTreeSet};

use humanart_doc::command::{Change, Outcome};
use humanart_doc::document::{AuthorKind, Document};
use humanart_doc::notify::{EventKind, Since};
use humanart_doc::registry::Registry;
use humanart_doc::session::Session;

/// Who last set a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Badge {
    /// Nobody: the registry default.
    Default,
    /// A person.
    Human,
    /// An agent.
    Agent,
}

/// The badge of parameter `id` in `doc`.
pub fn badge(doc: &Document, id: &str) -> Badge {
    match doc.param_authors.get(id).and_then(|a| doc.author(a)) {
        None => Badge::Default,
        Some(a) if a.kind == AuthorKind::Agent => Badge::Agent,
        Some(_) => Badge::Human,
    }
}

/// The provenance state (see the module documentation).
#[derive(Clone, Debug, Default)]
pub struct Provenance {
    seen: u64,
    param_flash: BTreeMap<String, f64>,
    region_flash: BTreeMap<String, f64>,
    changed: BTreeSet<String>,
    agent_commands: Vec<(String, String)>,
}

/// Most agent commands remembered for the Agent panel.
pub const RECENT_AGENT_COMMANDS: usize = 32;

impl Provenance {
    /// A state that has seen nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads the session's events after the last one seen; `now` is the UI clock in seconds.
    /// Returns how many events were read. Nothing is allocated when there is no new event.
    pub fn update(&mut self, session: &Session<'_>, registry: &Registry, now: f64) -> usize {
        let last = session.events().last_seq();
        if last == self.seen {
            return 0;
        }
        let doc = session.document();
        let events = match session.events().since(self.seen) {
            Since::Events(list) => list,
            // Older events were dropped: read what is kept.
            Since::Gap { .. } => session.events().iter().collect(),
        };
        let n = events.len();
        for e in events {
            if let EventKind::Committed(o) | EventKind::Undone(o) | EventKind::Redone(o) = &e.kind
                && o.proposal.is_none()
            {
                self.record(doc, registry, o, now);
            }
        }
        self.seen = last;
        n
    }

    fn record(&mut self, doc: &Document, registry: &Registry, o: &Outcome, now: f64) {
        let agent = doc.author(&o.author).is_some_and(|a| a.kind == AuthorKind::Agent);
        if agent {
            if self.agent_commands.len() >= RECENT_AGENT_COMMANDS {
                self.agent_commands.remove(0);
            }
            let label = if o.label.is_empty() { o.kind.clone() } else { o.label.clone() };
            self.agent_commands.push((o.author.clone(), label));
        }
        for c in &o.changes {
            let key = match c {
                Change::Param { id, .. } => id.clone(),
                Change::Pose { joint, .. } => format!("pose.{joint}"),
                _ => continue,
            };
            if agent {
                if let Change::Param { id, .. } = c
                    && let Some(region) = registry.get(id).and_then(|p| p.region.clone())
                {
                    self.region_flash.insert(region, now);
                }
                self.param_flash.insert(key.clone(), now);
            }
            self.changed.insert(key);
        }
    }

    /// The flash intensity of parameter `id` at `now` (1 when it just changed, fading to 0 after
    /// `duration` seconds).
    pub fn flash(&self, id: &str, now: f64, duration: f32) -> f32 {
        self.param_flash.get(id).map_or(0.0, |&t| fade(now - t, duration))
    }

    /// The regions flashing at `now`, with their intensity.
    pub fn region_flashes(&self, now: f64, duration: f32) -> impl Iterator<Item = (&str, f32)> + '_ {
        self.region_flash.iter().map(move |(r, &t)| (r.as_str(), fade(now - t, duration))).filter(|(_, f)| *f > 0.0)
    }

    /// Whether anything is flashing at `now` (the shell keeps drawing frames while it is).
    pub fn animating(&self, now: f64, duration: f32) -> bool {
        self.param_flash.values().any(|&t| fade(now - t, duration) > 0.0)
    }

    /// Whether `id` changed since the last [`Provenance::mark_looked`].
    pub fn changed_since_look(&self, id: &str) -> bool {
        self.changed.contains(id)
    }

    /// How many values changed since the last look.
    pub fn changed_count(&self) -> usize {
        self.changed.len()
    }

    /// The human has looked: clear the "changed since" set.
    pub fn mark_looked(&mut self) {
        self.changed.clear();
    }

    /// The agent's recent commands (author id, label), oldest first.
    pub fn agent_commands(&self) -> &[(String, String)] {
        &self.agent_commands
    }

    /// The last event sequence number read.
    pub fn seen(&self) -> u64 {
        self.seen
    }
}

fn fade(age: f64, duration: f32) -> f32 {
    if duration <= 0.0 || !(0.0..f64::from(duration)).contains(&age) {
        return 0.0;
    }
    #[allow(clippy::cast_possible_truncation)] // A ratio in 0..1.
    let t = (age / f64::from(duration)) as f32;
    1.0 - t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{AGENT, Fixture, HUMAN};

    #[test]
    fn ac_wp21_2_badges_and_flashes_follow_the_author() {
        // [AC-WP21-2] [APP-03] a human change shows a human badge and does not flash; an agent
        // change shows an agent badge, flashes its parameter and region for the configured time.
        let f = Fixture::new();
        let mut s = f.session();
        let (human_id, agent_id) = f.two_scalar_params_with_regions();
        let mut prov = Provenance::new();
        assert_eq!(badge(s.document(), &human_id), Badge::Default);
        s.execute(Fixture::set(HUMAN, &human_id, 0.4)).unwrap();
        prov.update(&s, f.registry, 10.0);
        assert_eq!(badge(s.document(), &human_id), Badge::Human);
        assert_eq!(prov.flash(&human_id, 10.0, 2.0), 0.0, "human changes do not flash");
        assert!(prov.changed_since_look(&human_id));

        s.execute(Fixture::set(AGENT, &agent_id, 0.6)).unwrap();
        prov.update(&s, f.registry, 11.0);
        assert_eq!(badge(s.document(), &agent_id), Badge::Agent);
        assert!((prov.flash(&agent_id, 11.0, 2.0) - 1.0).abs() < 1e-6);
        assert!((prov.flash(&agent_id, 12.0, 2.0) - 0.5).abs() < 1e-6);
        assert_eq!(prov.flash(&agent_id, 13.5, 2.0), 0.0, "the flash ends after its duration");
        let region = f.registry.get(&agent_id).and_then(|p| p.region.clone()).unwrap();
        assert!(prov.region_flashes(11.5, 2.0).any(|(r, _)| r == region), "the viewport highlights the region");
        assert_eq!(prov.agent_commands().len(), 1);

        // The agent overwrites the human's value: the badge turns to agent.
        s.execute(Fixture::set(AGENT, &human_id, 0.1)).unwrap();
        prov.update(&s, f.registry, 20.0);
        assert_eq!(badge(s.document(), &human_id), Badge::Agent);
        prov.mark_looked();
        assert_eq!(prov.changed_count(), 0);
        assert_eq!(prov.update(&s, f.registry, 21.0), 0, "no new event, no work");
    }
}
