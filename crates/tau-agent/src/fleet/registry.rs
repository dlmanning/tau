//! Spec/handle registry with a load-bearing invariant.
//!
//! **Invariant**: a spec exists in `specs` iff its id is in `idle`
//! (stored, awaiting resume) OR `running` (currently executing).
//!
//! In the original `tau-agent`, this invariant was maintained by six
//! free helper functions plus inline mutations in `respec` / `send`,
//! with a single test asserting one path. Easy to drift on a future
//! change. Here it's enforced by *only* exposing methods that
//! preserve it. The maps are private; mutations go through this
//! type's methods.
//!
//! Concurrency: all three maps live behind a single `Mutex` so
//! check-and-mutate operations (e.g. respec's "verify-not-running,
//! then detach from idle") are atomic by construction.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tau_ai::Usage;

use crate::core::handle::AgentHandle;
use crate::fleet::manager::AgentSpec;
use crate::fleet::snapshot::AgentSnapshot;

/// Per-agent metadata stored alongside its handle in idle / running /
/// adopted state. Cloneable so the registry can keep a snapshot in
/// `running` while handing the same snapshot to the lifecycle for the
/// duration of a resume.
#[derive(Clone)]
pub struct AgentEntry {
    pub handle: AgentHandle,
    pub description: String,
    /// Token usage snapshot at the last storage time. Used to compute
    /// per-resume delta in `lifecycle::send`.
    pub usage_at_pause: Usage,
    /// Conversation message count at the last storage time. Used to
    /// compute per-resume tool-call count.
    pub messages_at_pause: usize,
    /// Live cumulative usage, accumulated from `TurnEnd` events seen by
    /// the fleet bus. Distinct from `usage_at_pause` (which only
    /// updates at suspend boundaries) — `usage` is current as of the
    /// last forwarded `TurnEnd`.
    pub usage: Usage,
    /// Cumulative count of `ToolExecutionEnd` events observed for this
    /// agent. Counts every completed tool call, including errored ones,
    /// because the count reflects "tools attempted" rather than "tools
    /// succeeded".
    pub tool_use_count: u32,
    /// Wall-clock timestamp of first transition into a tracked state
    /// (commit_running or adopt). Once set, never overwritten.
    pub started_at: Option<DateTime<Utc>>,
    /// Wall-clock timestamp of the most recent `finish_to_idle`.
    /// Refreshes on every resume → idle cycle.
    pub completed_at: Option<DateTime<Utc>>,
}

impl AgentEntry {
    pub fn new(handle: AgentHandle, description: String) -> Self {
        Self {
            handle,
            description,
            usage_at_pause: Usage::default(),
            messages_at_pause: 0,
            usage: Usage::default(),
            tool_use_count: 0,
            started_at: None,
            completed_at: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Running,
    Idle,
    /// Registered with the manager via `adopt` but not in the
    /// fleet-managed lifecycle: the host built this handle directly and
    /// drives its actor. `spec_for` and `respec` work; `send` does not.
    Adopted,
}

/// Result of `find_agent` — id, description, and current status.
#[derive(Debug, Clone)]
pub struct Located {
    pub agent_id: String,
    pub description: String,
    pub status: Status,
}

struct Inner {
    /// Idle agents in LRU-eviction order. Newest pushed back; oldest
    /// popped front when at capacity.
    idle: VecDeque<(String, AgentEntry)>,
    /// Currently executing agents.
    running: HashMap<String, AgentEntry>,
    /// Externally-built handles registered via `adopt`. The fleet
    /// doesn't manage their actor lifecycle (the host does); they're
    /// here only so `spec_for` and `respec` can find them. Not subject
    /// to LRU eviction.
    adopted: HashMap<String, AgentEntry>,
    /// Specs keyed by agent id. Maintained in lockstep with
    /// `idle ∪ running ∪ adopted`.
    specs: HashMap<String, Arc<AgentSpec>>,
    max_agents: usize,
}

pub struct Registry {
    inner: Mutex<Inner>,
}

impl Registry {
    pub fn new(max_agents: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                idle: VecDeque::new(),
                running: HashMap::new(),
                adopted: HashMap::new(),
                specs: HashMap::new(),
                max_agents,
            }),
        }
    }

    // ─── Spec read ───────────────────────────────────────────────────

    pub fn spec_for(&self, agent_id: &str) -> Option<Arc<AgentSpec>> {
        self.inner.lock().specs.get(agent_id).cloned()
    }

    // ─── Begin-spawn: insert spec; agent is about to run ─────────────

    /// Reserve a spec for a new agent. The caller is responsible for
    /// calling [`Self::commit_running`] or [`Self::abandon`] depending
    /// on whether the spawn succeeded.
    pub fn begin_spawn(&self, agent_id: &str, spec: Arc<AgentSpec>) {
        self.inner.lock().specs.insert(agent_id.to_string(), spec);
    }

    /// Promote a reserved agent into the running set. Stamps
    /// `started_at` on first insertion (preserved across subsequent
    /// idle→running cycles, which go through `take_idle_into_running`
    /// instead and therefore don't reach this method).
    pub fn commit_running(&self, agent_id: &str, mut entry: AgentEntry) {
        if entry.started_at.is_none() {
            entry.started_at = Some(Utc::now());
        }
        self.inner
            .lock()
            .running
            .insert(agent_id.to_string(), entry);
    }

    /// Drop a reserved-but-not-yet-running agent (spawn failure path).
    /// Removes spec; nothing to remove from idle/running.
    pub fn abandon(&self, agent_id: &str) {
        self.inner.lock().specs.remove(agent_id);
    }

    // ─── End-of-run: move running → idle (or drop) ───────────────────

    /// Move an agent from running into idle storage. Evicts the oldest
    /// idle entry (and drops its spec) if at capacity. The caller is
    /// responsible for populating `entry.usage_at_pause` and
    /// `entry.messages_at_pause` before calling — the registry can't
    /// query the handle (async; we're holding the inner lock).
    ///
    /// Bus-bookkeeping fields (`usage`, `tool_use_count`, `started_at`)
    /// are carried over from the running entry if it exists, since the
    /// caller typically passes a freshly-built `AgentEntry` that
    /// doesn't reflect the bus's running updates. `completed_at` is
    /// stamped here.
    pub fn finish_to_idle(&self, agent_id: &str, mut entry: AgentEntry) {
        let mut inner = self.inner.lock();
        if let Some(running) = inner.running.remove(agent_id) {
            // Preserve fields the bus mutated during the run.
            entry.usage = running.usage;
            entry.tool_use_count = running.tool_use_count;
            entry.started_at = running.started_at.or(entry.started_at);
        }
        if entry.started_at.is_none() {
            entry.started_at = Some(Utc::now());
        }
        entry.completed_at = Some(Utc::now());
        if inner.idle.len() >= inner.max_agents {
            if let Some((evicted_id, _)) = inner.idle.pop_front() {
                inner.specs.remove(&evicted_id);
            }
        }
        inner.idle.push_back((agent_id.to_string(), entry));
    }

    /// Drop a running agent without idling it (error path, interactive
    /// remove_interactive, etc.). Drops the spec only if the agent
    /// isn't also in idle (which would only be the case mid-resume).
    ///
    /// **Snapshot behavior**: agents discarded via `drop_running` are
    /// removed from every tracked bucket and therefore do **not**
    /// appear in subsequent [`Self::snapshot`] results. This means
    /// error-terminated runs leave no trace in the fleet snapshot.
    /// A future refactor could add a dedicated "terminated" bucket so
    /// failed agents remain visible (with `completed_at` set) for
    /// post-mortem UI; today, hosts that need that history must rely
    /// on the `SubagentCompleted { outcome: Failed | Aborted, .. }`
    /// event stream instead.
    pub fn drop_running(&self, agent_id: &str) {
        let mut inner = self.inner.lock();
        inner.running.remove(agent_id);
        let in_idle = inner.idle.iter().any(|(id, _)| id == agent_id);
        if !in_idle {
            inner.specs.remove(agent_id);
        }
    }

    // ─── Resume: atomic idle → running transition ────────────────────

    /// Atomically move an entry from idle to running for the duration
    /// of a resume. The entry is cloned: one copy goes into `running`
    /// (so concurrent lookups and respec-blocking see the agent), the
    /// other is returned to the caller for the resume itself.
    ///
    /// Without this atomic step, a concurrent `respec` between the
    /// caller's `take_idle` and `commit_running` would see neither
    /// `idle` nor `running` populated and surface `Err("missing")`
    /// instead of the correct `Err("running")`.
    pub fn take_idle_into_running(&self, agent_id: &str) -> Option<AgentEntry> {
        let mut inner = self.inner.lock();
        let pos = inner.idle.iter().position(|(k, _)| k == agent_id)?;
        let (id, entry) = inner.idle.remove(pos).expect("position was found");
        inner.running.insert(id, entry.clone());
        Some(entry)
    }

    // ─── Adopt: register an externally-built handle ──────────────────

    /// Insert an externally-built handle + spec atomically. The spec is
    /// recorded under `agent_id`; the entry goes into the `adopted`
    /// bucket. Preserves the invariant: spec exists ⟺ id ∈ idle ∪
    /// running ∪ adopted.
    pub fn adopt(&self, agent_id: &str, mut entry: AgentEntry, spec: Arc<AgentSpec>) {
        if entry.started_at.is_none() {
            entry.started_at = Some(Utc::now());
        }
        let mut inner = self.inner.lock();
        inner.adopted.insert(agent_id.to_string(), entry);
        inner.specs.insert(agent_id.to_string(), spec);
    }

    // ─── Respec: atomic verify-and-detach ────────────────────────────

    /// Verify the agent is not running and detach it under a single
    /// lock. Returns the entry on success. If the agent is currently
    /// running, returns `Err("running")`. If not found in idle or
    /// adopted, returns `Err("missing")`. The spec remains in `specs`
    /// so an in-flight `spec_for` call during the surrounding async
    /// work resolves until the caller explicitly drops it via
    /// [`Self::drop_respec_source`].
    pub fn detach_for_respec(&self, agent_id: &str) -> Result<AgentEntry, &'static str> {
        let mut inner = self.inner.lock();
        if inner.running.contains_key(agent_id) {
            return Err("running");
        }
        if let Some(pos) = inner.idle.iter().position(|(k, _)| k == agent_id) {
            return Ok(inner.idle.remove(pos).expect("position was found").1);
        }
        if let Some(entry) = inner.adopted.remove(agent_id) {
            return Ok(entry);
        }
        Err("missing")
    }

    /// On successful respec, drop the old spec.
    pub fn drop_respec_source(&self, agent_id: &str) {
        self.inner.lock().specs.remove(agent_id);
    }

    /// On failed respec, restore the original idle entry. Push to the
    /// back (treats the recovered entry as freshly-used).
    pub fn restore_respec_source(&self, agent_id: &str, entry: AgentEntry) {
        self.inner
            .lock()
            .idle
            .push_back((agent_id.to_string(), entry));
    }

    // ─── Bus bookkeeping ────────────────────────────────────────────

    /// Accumulate a turn's reported `Usage` onto whichever bucket
    /// currently holds the agent. `TurnEnd.usage` is per-turn, so we
    /// add it onto the entry's running total.
    pub fn record_turn_end(&self, agent_id: &str, usage: &Usage) {
        let mut inner = self.inner.lock();
        if let Some(entry) = inner.running.get_mut(agent_id) {
            add_usage(&mut entry.usage, usage);
            return;
        }
        if let Some((_, entry)) = inner.idle.iter_mut().find(|(k, _)| k == agent_id) {
            add_usage(&mut entry.usage, usage);
            return;
        }
        if let Some(entry) = inner.adopted.get_mut(agent_id) {
            add_usage(&mut entry.usage, usage);
        }
    }

    /// Increment the tool-call counter on whichever bucket currently
    /// holds the agent. Called from the bus on `ToolExecutionEnd`.
    pub fn record_tool_use(&self, agent_id: &str) {
        let mut inner = self.inner.lock();
        if let Some(entry) = inner.running.get_mut(agent_id) {
            entry.tool_use_count = entry.tool_use_count.saturating_add(1);
            return;
        }
        if let Some((_, entry)) = inner.idle.iter_mut().find(|(k, _)| k == agent_id) {
            entry.tool_use_count = entry.tool_use_count.saturating_add(1);
            return;
        }
        if let Some(entry) = inner.adopted.get_mut(agent_id) {
            entry.tool_use_count = entry.tool_use_count.saturating_add(1);
        }
    }

    // ─── Snapshot ────────────────────────────────────────────────────

    /// Collect every tracked agent into a `Vec<AgentSnapshot>`. Walks
    /// running, idle, and adopted buckets under a single lock — every
    /// snapshot represents a consistent point in time.
    pub fn snapshot(&self) -> Vec<AgentSnapshot> {
        let inner = self.inner.lock();
        let mut out = Vec::with_capacity(
            inner.running.len() + inner.idle.len() + inner.adopted.len(),
        );
        for (id, entry) in &inner.running {
            out.push(snapshot_of(id, entry, Status::Running));
        }
        for (id, entry) in &inner.idle {
            out.push(snapshot_of(id, entry, Status::Idle));
        }
        for (id, entry) in &inner.adopted {
            out.push(snapshot_of(id, entry, Status::Adopted));
        }
        out
    }

    // ─── Lookup ──────────────────────────────────────────────────────

    /// Locate an agent by id or by case-insensitive description
    /// substring. Running wins over idle wins over adopted; exact-id
    /// matches win over substring matches. First match wins; substring
    /// ambiguity across `HashMap` is unspecified.
    pub fn find(&self, name_or_id: &str) -> Option<Located> {
        let inner = self.inner.lock();
        if let Some(entry) = inner.running.get(name_or_id) {
            return Some(Located {
                agent_id: name_or_id.into(),
                description: entry.description.clone(),
                status: Status::Running,
            });
        }
        let needle = name_or_id.to_lowercase();
        for (id, entry) in &inner.running {
            if entry.description.to_lowercase().contains(&needle) {
                return Some(Located {
                    agent_id: id.clone(),
                    description: entry.description.clone(),
                    status: Status::Running,
                });
            }
        }
        if let Some((id, entry)) = inner.idle.iter().find(|(k, _)| k == name_or_id) {
            return Some(Located {
                agent_id: id.clone(),
                description: entry.description.clone(),
                status: Status::Idle,
            });
        }
        for (id, entry) in &inner.idle {
            if entry.description.to_lowercase().contains(&needle) {
                return Some(Located {
                    agent_id: id.clone(),
                    description: entry.description.clone(),
                    status: Status::Idle,
                });
            }
        }
        if let Some(entry) = inner.adopted.get(name_or_id) {
            return Some(Located {
                agent_id: name_or_id.into(),
                description: entry.description.clone(),
                status: Status::Adopted,
            });
        }
        for (id, entry) in &inner.adopted {
            if entry.description.to_lowercase().contains(&needle) {
                return Some(Located {
                    agent_id: id.clone(),
                    description: entry.description.clone(),
                    status: Status::Adopted,
                });
            }
        }
        None
    }

    /// Clone of a running agent's handle (used by `AgentTool` to look
    /// up the parent for bg-spawn follow-up tracking). Adopted handles
    /// are *not* returned here — the lookup is for fleet-managed
    /// children of the running agent.
    pub fn handle_for(&self, agent_id: &str) -> Option<AgentHandle> {
        self.inner
            .lock()
            .running
            .get(agent_id)
            .map(|e| e.handle.clone())
    }

    /// Clone of any registered handle (running / idle / adopted), for
    /// history-inheriting spawns.
    pub fn handle_any(&self, agent_id: &str) -> Option<AgentHandle> {
        let inner = self.inner.lock();
        if let Some(e) = inner.running.get(agent_id) {
            return Some(e.handle.clone());
        }
        if let Some((_, e)) = inner.idle.iter().find(|(k, _)| k == agent_id) {
            return Some(e.handle.clone());
        }
        inner.adopted.get(agent_id).map(|e| e.handle.clone())
    }
}

fn snapshot_of(id: &str, entry: &AgentEntry, status: Status) -> AgentSnapshot {
    AgentSnapshot {
        agent_id: id.to_string(),
        description: entry.description.clone(),
        status,
        usage: entry.usage.clone(),
        tool_use_count: entry.tool_use_count,
        started_at: entry.started_at,
        completed_at: entry.completed_at,
    }
}

fn add_usage(dst: &mut Usage, src: &Usage) {
    dst.input = dst.input.saturating_add(src.input);
    dst.output = dst.output.saturating_add(src.output);
    dst.cache_read = dst.cache_read.saturating_add(src.cache_read);
    dst.cache_write = dst.cache_write.saturating_add(src.cache_write);
    dst.thinking = dst.thinking.saturating_add(src.thinking);
    dst.cache_creation_1h = dst.cache_creation_1h.saturating_add(src.cache_creation_1h);
    dst.cache_creation_5m = dst.cache_creation_5m.saturating_add(src.cache_creation_5m);
    if dst.service_tier.is_none() {
        dst.service_tier = src.service_tier.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::builder::AgentBuilder;
    use crate::test_utils::*;

    fn dummy_handle() -> AgentHandle {
        let transport = TextTransport::create("x");
        AgentBuilder::new(test_config(), transport).handle()
    }

    #[test]
    fn invariant_after_begin_then_abandon() {
        let r = Registry::new(4);
        let spec = Arc::new(AgentSpec {
            system_prompt: String::new(),
            tools: vec![],
            max_turns: 1,
            allows_worktree: false,
            allowed_subagent_specs: None,
        });
        r.begin_spawn("a1", spec);
        assert!(r.spec_for("a1").is_some());
        r.abandon("a1");
        assert!(r.spec_for("a1").is_none());
    }

    #[test]
    fn invariant_finish_to_idle() {
        let r = Registry::new(4);
        let spec = Arc::new(AgentSpec {
            system_prompt: String::new(),
            tools: vec![],
            max_turns: 1,
            allows_worktree: false,
            allowed_subagent_specs: None,
        });
        r.begin_spawn("a1", spec);
        let entry = AgentEntry::new(dummy_handle(), "first".into());
        r.commit_running("a1", entry);
        let entry = AgentEntry::new(dummy_handle(), "first".into());
        r.finish_to_idle("a1", entry);
        assert!(r.spec_for("a1").is_some(), "spec preserved across run→idle");
        assert!(r.find("a1").is_some(), "agent locatable");
    }

    #[test]
    fn eviction_drops_oldest_spec() {
        let r = Registry::new(2);
        for id in &["a", "b", "c"] {
            let spec = Arc::new(AgentSpec {
                system_prompt: String::new(),
                tools: vec![],
                max_turns: 1,
                allows_worktree: false,
                allowed_subagent_specs: None,
            });
            r.begin_spawn(id, spec);
            let entry = AgentEntry::new(dummy_handle(), id.to_string());
            r.commit_running(id, entry);
            let entry = AgentEntry::new(dummy_handle(), id.to_string());
            r.finish_to_idle(id, entry);
        }
        // `a` should have been evicted; `b` and `c` remain.
        assert!(r.spec_for("a").is_none(), "oldest evicted");
        assert!(r.spec_for("b").is_some());
        assert!(r.spec_for("c").is_some());
    }

    #[test]
    fn detach_for_respec_blocks_when_running() {
        let r = Registry::new(4);
        let spec = Arc::new(AgentSpec {
            system_prompt: String::new(),
            tools: vec![],
            max_turns: 1,
            allows_worktree: false,
            allowed_subagent_specs: None,
        });
        r.begin_spawn("a", spec);
        r.commit_running("a", AgentEntry::new(dummy_handle(), "x".into()));
        assert!(matches!(r.detach_for_respec("a"), Err("running")));
    }

    #[test]
    fn detach_for_respec_succeeds_when_idle() {
        let r = Registry::new(4);
        let spec = Arc::new(AgentSpec {
            system_prompt: String::new(),
            tools: vec![],
            max_turns: 1,
            allows_worktree: false,
            allowed_subagent_specs: None,
        });
        r.begin_spawn("a", spec);
        r.commit_running("a", AgentEntry::new(dummy_handle(), "x".into()));
        r.finish_to_idle("a", AgentEntry::new(dummy_handle(), "x".into()));
        let entry = r.detach_for_respec("a").expect("idle agent detaches");
        assert_eq!(entry.description, "x");
        // Spec still present until drop_respec_source.
        assert!(r.spec_for("a").is_some());
        r.drop_respec_source("a");
        assert!(r.spec_for("a").is_none());
    }
}
