//! Subagent spawning — create independent agent instances for parallel work.
//!
//! The runtime is ignorant of which agent types exist, what tools are
//! available, or what prompts they use. Hosts construct an [`AgentSpec`]
//! (system prompt, tools, max turns, worktree allowance, allowed subagent
//! specs) and pass it to [`AgentManager::spawn`]. Spec names are opaque
//! strings owned by the host; the runtime treats `AgentSpec` as a fixed
//! per-agent input. Changing the spec means spawning a new agent —
//! exposed via [`AgentManager::respec`] / [`AgentHandle::respec`].

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex as ParkingMutex;
use tau_ai::{Content, Message, Model, Usage};
use tokio::sync::{Mutex, broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::approval::{ApprovalPolicy, DefaultApprovalPolicy};
use crate::builder::AgentBuilder;
use crate::config::AgentConfig;
use crate::events::{AgentEvent, SubagentOutcome};
use crate::handle::AgentHandle;
use crate::interaction::InteractionRequest;
use crate::tool::{BoxedTool, send_event};
use crate::transcript::record_transcript;
use crate::transport::Transport;
use crate::worktree::{WorktreeInfo, cleanup_worktree, create_worktree};

/// Immutable per-agent input. The system prompt + tool set + parameters the
/// LLM sees as fixed for the agent's lifetime. To change any of these,
/// spawn a new agent (typically via [`AgentManager::respec`] /
/// [`AgentHandle::respec`]).
#[derive(Clone)]
pub struct AgentSpec {
    /// Agent-type-specific instruction text. The runtime wraps it with
    /// the env + tool-list section before assigning it as the actor's
    /// `system_prompt`. Hosts pass the bare instruction; the runtime adds
    /// the boilerplate via [`crate::prompts::build_subagent_prompt`].
    pub system_prompt: String,
    /// Tools the agent has access to. Hosts filter / construct this list
    /// per spec — including any nested [`AgentTool`](crate) for
    /// recursive spawning.
    ///
    /// **Tool sharing across spawns**: `BoxedTool` is `Arc<dyn Tool>`, so
    /// reusing the same `AgentSpec` for multiple concurrent spawns shares
    /// the same underlying tool objects. Tools that capture per-agent
    /// state via [`Tool::bind_to_agent`](crate::tool::Tool::bind_to_agent)
    /// (e.g. `AgentTool`'s `OnceLock<AgentHandle>`) will **bind to whichever
    /// spawn happens first** and silently mis-route for subsequent ones.
    /// Hosts that spawn the same spec concurrently must construct fresh
    /// tool instances per spawn (typically via a resolver closure that
    /// builds a new `AgentSpec` each call) rather than cloning a shared
    /// `Vec<BoxedTool>`.
    pub tools: Vec<BoxedTool>,
    /// Max turns before the agent loop stops.
    pub max_turns: u32,
    /// Whether this agent may run inside a git worktree if the spawn opts
    /// request `isolation = "worktree"`. Hosts set this to `false` for
    /// read-only specs (Explore, Plan).
    pub allows_worktree: bool,
    /// Spec names this agent is permitted to spawn as subagents. Carried
    /// through to the host's spawn-time resolver / `AgentTool`. The
    /// runtime does not enforce this directly — the AgentTool the host
    /// installs in `tools` reads it.
    pub allowed_subagent_specs: Option<Vec<String>>,
}

/// How a subagent's filesystem state is isolated from the parent's.
///
/// Serializes as a snake_case string so the same enum can be used in tool
/// argument schemas without divergence — the LLM sees `"worktree"`, the
/// runtime gets a typed value.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Isolation {
    /// Run the subagent inside a fresh git worktree on a per-agent
    /// branch. Honored only when the spec sets `allows_worktree = true`;
    /// otherwise silently ignored.
    Worktree,
}

/// Per-spawn options that don't belong on the agent's immutable spec.
#[derive(Default, Clone)]
pub struct SpawnOpts {
    /// Short description for the parent UI / event timeline.
    pub description: String,
    /// Override the spec's default model.
    pub model: Option<Model>,
    /// Working directory for the subagent.
    pub cwd: Option<String>,
    /// Isolation mode for the subagent's filesystem state. `None` means
    /// "share the parent's working tree." Honored only when the spec
    /// allows it (see [`Isolation`]).
    pub isolation: Option<Isolation>,
    /// Seed the new subagent with a stored agent's full message history,
    /// then apply the spawn prompt as a follow-up user message. Useful for
    /// plan → execute handoffs: the executor inherits the planner's
    /// investigation as its own history.
    ///
    /// The named agent must already be stored (i.e., have terminated).
    pub inherit_history_from: Option<String>,
    /// Override the approval policy this subagent (and any descendants
    /// not overriding again) runs under.
    pub approval_policy: Option<Arc<dyn ApprovalPolicy>>,
    /// Host-supplied spec name to stamp on `SubagentStarted` events.
    /// The runtime stores it but does not interpret it.
    pub spec_name: Option<String>,
    /// Seed the new agent with an explicit message vector before its
    /// first turn. Wins over [`Self::inherit_history_from`] when both
    /// are set: this is the lower-level primitive (the host already has
    /// the messages — no lookup needed). Useful for `/branch`-style
    /// flows that fork from an arbitrary index in the parent's history.
    pub seed_messages: Option<Vec<Message>>,
}

/// Result from a completed subagent.
#[derive(Debug)]
pub struct SubagentResult {
    pub agent_id: String,
    pub text: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub tool_use_count: u32,
    pub duration_ms: u64,
    pub worktree_path: Option<String>,
    pub worktree_branch: Option<String>,
    /// Path to the JSONL transcript on disk, when transcript recording
    /// succeeded.
    pub transcript_path: Option<String>,
}

/// Whether an agent is currently executing or stored (idle).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStatus {
    Running,
    Idle,
}

/// Manages subagent lifecycle: spawn, resume, evict.
///
/// **Registry invariant**: a spec exists in `agent_specs` iff its id is
/// either in `agents` (idle storage) or `running_handles` (currently
/// executing). All mutations go through [`Self::insert_spec`],
/// [`Self::drop_spec`], [`Self::track_running`], [`Self::untrack_running`],
/// [`Self::forget`], and [`Self::store`] so the invariant is maintained
/// in one place.
pub struct AgentManager {
    agents: Mutex<VecDeque<(String, ManagedAgent)>>,
    max_agents: usize,
    transport: Arc<dyn Transport>,
    parent_config: AgentConfig,
    parent_event_tx: broadcast::Sender<AgentEvent>,
    /// Per-agent specs, keyed by agent id. Maintained in lockstep with
    /// `agents` ∪ `running_handles` — see the type-level invariant above.
    /// `AgentHandle::with_system_prompt` / `with_tools` look up the
    /// current spec here so callers can change a single field without
    /// rebuilding the whole spec.
    ///
    /// Stored as `Arc<AgentSpec>` because specs carry a `Vec<Arc<dyn Tool>>`
    /// that's expensive to deep-clone. Sharing one Arc across the registry,
    /// concurrent `spec_for` reads, and the run-time clone in spawn paths
    /// keeps the cost to refcount bumps.
    agent_specs: Mutex<HashMap<String, Arc<AgentSpec>>>,
    /// Parent's interaction sender. When set, each subagent gets a wrapping
    /// channel that stamps `agent_id` on outgoing requests and forwards them
    /// to this parent. When `None`, subagents have no interaction channel.
    parent_interaction_tx: Option<mpsc::Sender<InteractionRequest>>,
    /// Default `ApprovalPolicy` for spawned subagents. Inherited from the
    /// parent process so subagents gate the same way the parent does (the
    /// spawn opts can override per-call). Defaults to
    /// [`DefaultApprovalPolicy`] when not configured.
    parent_approval_policy: ParkingMutex<Arc<dyn ApprovalPolicy>>,
    /// Handles for agents that are currently executing (keyed by agent_id).
    running_handles: Mutex<HashMap<String, (AgentHandle, String)>>,
}

struct ManagedAgent {
    handle: AgentHandle,
    description: String,
    usage_at_pause: Usage,
    messages_at_pause: usize,
}

// ---------- Construction & configuration ----------

impl AgentManager {
    pub fn new(
        parent_event_tx: broadcast::Sender<AgentEvent>,
        parent_config: AgentConfig,
        transport: Arc<dyn Transport>,
        max_agents: usize,
    ) -> Self {
        Self {
            agents: Mutex::new(VecDeque::new()),
            max_agents,
            transport,
            parent_config,
            parent_event_tx,
            parent_interaction_tx: None,
            parent_approval_policy: ParkingMutex::new(Arc::new(DefaultApprovalPolicy)),
            running_handles: Mutex::new(HashMap::new()),
            agent_specs: Mutex::new(HashMap::new()),
        }
    }

    /// Forward subagent interaction requests to this parent channel,
    /// stamping `agent_id` along the way. Without this, `submit_plan` and
    /// `ask_user` from a subagent silently fail.
    pub fn with_parent_interaction_sender(
        mut self,
        tx: mpsc::Sender<InteractionRequest>,
    ) -> Self {
        self.parent_interaction_tx = Some(tx);
        self
    }

    /// Default policy spawned subagents run under. Per-spawn override is
    /// available via [`SpawnOpts::approval_policy`].
    pub fn with_parent_approval_policy(mut self, policy: Arc<dyn ApprovalPolicy>) -> Self {
        // We own `self` here, so the lock is uncontended; this is the
        // construction-time twin of `set_default_approval_policy`.
        *self.parent_approval_policy.get_mut() = policy;
        self
    }

    /// Replace the default approval policy at runtime. Affects future
    /// spawns only; in-flight subagents keep the policy they were spawned
    /// with.
    pub fn set_default_approval_policy(&self, policy: Arc<dyn ApprovalPolicy>) {
        *self.parent_approval_policy.lock() = policy;
    }
}

// ---------- Registry helpers (single source of truth for the invariant) ----------

impl AgentManager {
    /// Look up the current spec for an agent the manager knows about
    /// (running or idle). Returns `None` if the id is unknown — typically
    /// because the agent was evicted, never spawned through this manager,
    /// or has been replaced by `respec`. The returned `Arc` shares with
    /// the registry: clone it freely; mutate via [`Arc::unwrap_or_clone`]
    /// or `(*spec).clone()` when you need owned access.
    pub async fn spec_for(&self, agent_id: &str) -> Option<Arc<AgentSpec>> {
        self.agent_specs.lock().await.get(agent_id).cloned()
    }

    async fn insert_spec(&self, agent_id: &str, spec: Arc<AgentSpec>) {
        self.agent_specs.lock().await.insert(agent_id.to_string(), spec);
    }

    async fn drop_spec(&self, agent_id: &str) {
        self.agent_specs.lock().await.remove(agent_id);
    }

    async fn track_running(&self, agent_id: &str, handle: AgentHandle, description: String) {
        self.running_handles
            .lock()
            .await
            .insert(agent_id.to_string(), (handle, description));
    }

    async fn untrack_running(&self, agent_id: &str) {
        self.running_handles.lock().await.remove(agent_id);
    }

    /// Final cleanup for an agent that won't be stored idle. Removes the
    /// running entry; drops the spec if the agent isn't already in idle
    /// storage. Used by error paths and by [`Self::remove_interactive`].
    async fn forget(&self, agent_id: &str) {
        self.untrack_running(agent_id).await;
        let in_storage = self
            .agents
            .lock()
            .await
            .iter()
            .any(|(id, _)| id == agent_id);
        if !in_storage {
            self.drop_spec(agent_id).await;
        }
    }

    /// Move an agent into idle storage, evicting the oldest entry (and its
    /// spec) if at capacity. Spec stays in `agent_specs` while idle.
    async fn store(&self, id: String, handle: AgentHandle, description: String) {
        let mut agents = self.agents.lock().await;
        let evicted_id = if agents.len() >= self.max_agents {
            agents.pop_front().map(|(eid, _)| eid)
        } else {
            None
        };
        if let Some(eid) = evicted_id {
            self.agent_specs.lock().await.remove(&eid);
        }
        let state = handle.state().await.unwrap_or_default();
        let usage = state.total_usage.clone();
        let message_count = state.messages.len();
        agents.push_back((
            id,
            ManagedAgent {
                handle,
                description,
                usage_at_pause: usage,
                messages_at_pause: message_count,
            },
        ));
    }

    /// Register a builder-spawned handle with this manager so it can
    /// `respec` / `with_system_prompt` / `with_tools`. Stamps the
    /// handle's `agent_id` and `manager` `OnceLock`s and records the
    /// spec. Returns the assigned id. No-op (returning the existing id)
    /// if the handle already has one.
    ///
    /// Use this for the host's "root" agent that was built directly via
    /// `AgentBuilder::spawn()` rather than through the manager.
    pub async fn adopt(
        self: &Arc<Self>,
        handle: &AgentHandle,
        spec: impl Into<Arc<AgentSpec>>,
    ) -> String {
        let spec = spec.into();
        if let Some(existing) = handle.agent_id.get() {
            self.insert_spec(existing, spec).await;
            return existing.clone();
        }
        let id = uuid::Uuid::new_v4().to_string();
        let _ = handle.agent_id.set(id.clone());
        let _ = handle.manager.set(Arc::downgrade(self));
        self.insert_spec(&id, spec).await;
        id
    }

    /// Remove an interactive agent from the running handles. Drops the
    /// spec if the agent was never stored (so storage stays bounded).
    pub async fn remove_interactive(&self, agent_id: &str) {
        self.forget(agent_id).await;
    }

    /// Find an agent by name or ID.
    ///
    /// Resolution order: exact-id match in `running_handles`, then a
    /// case-insensitive description-substring match in `running_handles`,
    /// then the same two passes against idle storage. **First match wins**
    /// — an ambiguous needle (e.g. a substring shared by several
    /// descriptions) returns whichever matched first in iteration order,
    /// which for HashMap-backed storage is unspecified. Hosts that need
    /// disambiguation should pass an exact id.
    pub async fn find_agent(&self, name_or_id: &str) -> Option<(String, String, AgentStatus)> {
        {
            let running = self.running_handles.lock().await;
            if let Some((_, desc)) = running.get(name_or_id) {
                return Some((name_or_id.to_string(), desc.clone(), AgentStatus::Running));
            }
            let needle = name_or_id.to_lowercase();
            for (id, (_, desc)) in running.iter() {
                if desc.to_lowercase().contains(&needle) {
                    return Some((id.clone(), desc.clone(), AgentStatus::Running));
                }
            }
        }

        let agents = self.agents.lock().await;
        if let Some((id, e)) = agents.iter().find(|(id, _)| id == name_or_id) {
            return Some((id.clone(), e.description.clone(), AgentStatus::Idle));
        }
        let needle = name_or_id.to_lowercase();
        for (id, e) in agents.iter() {
            if e.description.to_lowercase().contains(&needle) {
                return Some((id.clone(), e.description.clone(), AgentStatus::Idle));
            }
        }
        None
    }

    /// Look up the live `AgentHandle` for a currently-running agent by
    /// id. Returns `None` if the agent isn't running (idle, evicted, or
    /// never spawned). Tools that hold a `Weak<AgentManager>` + agent id
    /// (rather than a strong handle) reach the live handle through this.
    pub async fn handle_for(&self, agent_id: &str) -> Option<AgentHandle> {
        self.running_handles
            .lock()
            .await
            .get(agent_id)
            .map(|(h, _)| h.clone())
    }

    /// Send a message to a currently running agent via steering.
    pub async fn send_to_running(&self, id: &str, message: Message) -> bool {
        let handle = {
            let guard = self.running_handles.lock().await;
            guard.get(id).map(|(h, _)| h.clone())
        };
        if let Some(handle) = handle {
            let _ = handle.steer(message).await;
            true
        } else {
            false
        }
    }

    /// Look up an agent by id and return a clone of its message log.
    /// Used to seed an inheriting subagent's conversation. Checks the
    /// stored-idle queue first, then `running_handles` (so interactive
    /// agents can still hand off their history without being moved into
    /// idle storage first).
    async fn fetch_stored_messages(&self, source_id: &str) -> crate::error::Result<Vec<Message>> {
        let stored = {
            let agents = self.agents.lock().await;
            agents
                .iter()
                .find(|(id, _)| id == source_id)
                .map(|(_, e)| e.handle.clone())
        };
        let handle = if let Some(h) = stored {
            h
        } else {
            let guard = self.running_handles.lock().await;
            match guard.get(source_id).map(|(h, _)| h.clone()) {
                Some(h) => {
                    drop(guard);
                    h
                }
                None => {
                    return Err(crate::error::Error::Other(format!(
                        "inherit_history_from: no agent with id '{source_id}' \
                         (still running, evicted, or never spawned)"
                    )));
                }
            }
        };
        // Propagate fetch failures rather than masking them as "empty
        // history": silently inheriting nothing would look like a clean
        // handoff to callers, which is the worst kind of bug. The actor
        // returning `None` here means it's gone or unresponsive — both
        // cases the caller needs to know about.
        handle.messages().await.ok_or_else(|| {
            crate::error::Error::Other(format!(
                "inherit_history_from: agent '{source_id}' did not return its message log \
                 (actor unresponsive or shutting down)"
            ))
        })
    }

    /// Build a per-subagent interaction sender that stamps `agent_id` and
    /// forwards to this manager's parent channel. Returns `None` when no
    /// parent channel is configured (subagent runs headless).
    fn build_subagent_interaction_sender(
        &self,
        agent_id: &str,
    ) -> Option<mpsc::Sender<InteractionRequest>> {
        let parent_tx = self.parent_interaction_tx.as_ref()?.clone();
        let agent_id = agent_id.to_string();
        let (sub_tx, mut sub_rx) = mpsc::channel::<InteractionRequest>(8);
        tokio::spawn(async move {
            while let Some(mut req) = sub_rx.recv().await {
                if req.agent_id.is_none() {
                    req.agent_id = Some(agent_id.clone());
                }
                if parent_tx.send(req).await.is_err() {
                    break;
                }
            }
        });
        Some(sub_tx)
    }

    fn spawn_event_forwarder(
        &self,
        mut events: broadcast::Receiver<AgentEvent>,
        agent_id: &str,
        description: &str,
    ) -> tokio::task::JoinHandle<()> {
        let tx = self.parent_event_tx.clone();
        let agent_id = agent_id.to_string();
        let desc = description.to_string();
        tokio::spawn(async move {
            loop {
                match events.recv().await {
                    Ok(event) => {
                        let _ = tx.send(AgentEvent::Subagent {
                            agent_id: agent_id.clone(),
                            description: desc.clone(),
                            event: Box::new(event),
                        });
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(
                            agent_id = %agent_id,
                            dropped = n,
                            "subagent event stream lagged; dropped events will not reach the parent"
                        );
                    }
                }
            }
        })
    }
}

// ---------- Spawn / resume / respec ----------

impl AgentManager {
    /// Spawn a foreground subagent. Blocks until completion.
    pub async fn spawn(
        self: &Arc<Self>,
        spec: impl Into<Arc<AgentSpec>>,
        initial_prompt: String,
        opts: SpawnOpts,
        cancel: CancellationToken,
    ) -> crate::error::Result<SubagentResult> {
        let agent_id = uuid::Uuid::new_v4().to_string();
        let description = opts.description.clone();
        // Accept either an owned `AgentSpec` (auto-Arc'd via `From`) or an
        // existing `Arc<AgentSpec>`. The wrap is free for the latter.
        let spec = spec.into();
        // Spec must exist before any handle.respec lookups during the run.
        self.insert_spec(&agent_id, Arc::clone(&spec)).await;
        let result = self
            .run_subagent(&spec, &initial_prompt, &opts, cancel, &agent_id)
            .await;
        self.untrack_running(&agent_id).await;
        match result {
            Ok((subresult, handle)) => {
                self.store(subresult.agent_id.clone(), handle, description)
                    .await;
                Ok(subresult)
            }
            Err(e) => {
                self.drop_spec(&agent_id).await;
                Err(e)
            }
        }
    }

    /// Spawn a background subagent. Returns immediately with the agent_id.
    pub async fn spawn_background(
        self: &Arc<Self>,
        spec: impl Into<Arc<AgentSpec>>,
        initial_prompt: String,
        opts: SpawnOpts,
        parent_handle: AgentHandle,
        parent_cancel: CancellationToken,
    ) -> String {
        let agent_id = uuid::Uuid::new_v4().to_string();
        let description = opts.description.clone();
        let bg_cancel = CancellationToken::new();

        parent_handle.expect_follow_up();
        let spec = spec.into();
        self.insert_spec(&agent_id, Arc::clone(&spec)).await;

        let manager = self.clone();
        let desc = description.clone();
        let aid = agent_id.clone();
        let bg_cancel_inner = bg_cancel.clone();

        tokio::spawn(async move {
            // Forward parent cancellation to bg_cancel without dropping
            // the run-future. `run_subagent` honors bg_cancel and runs
            // its own cleanup (event forwarder abort, untrack_running)
            // before returning.
            let cancel_forwarder = {
                let bg_cancel = bg_cancel.clone();
                tokio::spawn(async move {
                    parent_cancel.cancelled().await;
                    bg_cancel.cancel();
                })
            };

            let result = manager
                .run_subagent(&spec, &initial_prompt, &opts, bg_cancel_inner, &aid)
                .await;
            cancel_forwarder.abort();

            manager.untrack_running(&aid).await;

            match result {
                Ok((subresult, handle)) => {
                    manager
                        .store(subresult.agent_id.clone(), handle, desc.clone())
                        .await;

                    let _ = parent_handle
                        .follow_up(Message::subagent_completed(
                            &subresult.agent_id,
                            &desc,
                            format!(
                                "{}\n[Agent {} | {} in + {} out tokens | {} tool calls | {}ms]",
                                subresult.text,
                                subresult.agent_id,
                                subresult.input_tokens,
                                subresult.output_tokens,
                                subresult.tool_use_count,
                                subresult.duration_ms,
                            ),
                        ))
                        .await;
                }
                Err(e) => {
                    manager.drop_spec(&aid).await;
                    let _ = parent_handle
                        .follow_up(Message::subagent_failed(
                            &aid,
                            &desc,
                            format!("Error: {}", e),
                        ))
                        .await;
                }
            }
        });

        agent_id
    }

    /// Spawn a subagent and return its handle for interactive use.
    /// The caller drives the conversation by sending prompts via the handle.
    /// No event forwarder is set up — the caller subscribes directly.
    pub async fn spawn_interactive(
        self: &Arc<Self>,
        spec: impl Into<Arc<AgentSpec>>,
        opts: SpawnOpts,
    ) -> crate::error::Result<(AgentHandle, String)> {
        let agent_id = uuid::Uuid::new_v4().to_string();
        let spec = spec.into();
        self.insert_spec(&agent_id, Arc::clone(&spec)).await;

        let builder = match self.configure_builder(&spec, &opts, &agent_id, None).await {
            Ok(b) => b,
            Err(e) => {
                self.drop_spec(&agent_id).await;
                return Err(e);
            }
        };
        let handle = builder.spawn();
        self.track_running(&agent_id, handle.clone(), opts.description.clone())
            .await;
        Ok((handle, agent_id))
    }

    /// Transition an idle stored agent to a new spec. Fetches the old
    /// agent's history, spawns a fresh agent under `new_spec` seeded
    /// with that history, and **evicts the original idle entry** — the
    /// old `agent_id` no longer resolves after this call. Use
    /// [`Self::spawn`] / [`SpawnOpts::inherit_history_from`] directly if
    /// you want to fork instead of transition.
    ///
    /// The agent must already be idle. The caller drives the returned
    /// handle and cleans up via [`Self::remove_interactive`] when done,
    /// same as [`Self::spawn_interactive`].
    pub async fn respec(
        self: &Arc<Self>,
        agent_id: &str,
        new_spec: impl Into<Arc<AgentSpec>>,
    ) -> crate::error::Result<AgentHandle> {
        if self.running_handles.lock().await.contains_key(agent_id) {
            return Err(crate::error::Error::Other(format!(
                "respec: agent '{agent_id}' is currently running; abort and await its terminal event before respec"
            )));
        }
        let opts = SpawnOpts {
            description: format!("respec({agent_id})"),
            inherit_history_from: Some(agent_id.to_string()),
            ..Default::default()
        };
        let (handle, _new_id) = self.spawn_interactive(new_spec, opts).await?;
        // Drop the old idle entry — respec is a transition, not a fork.
        // Done after the spawn succeeds so a failed handoff leaves the
        // original recoverable.
        {
            let mut agents = self.agents.lock().await;
            if let Some(pos) = agents.iter().position(|(id, _)| id == agent_id) {
                agents.remove(pos);
            }
        }
        self.drop_spec(agent_id).await;
        Ok(handle)
    }

    /// Send a message to a previously stored agent (resume).
    pub async fn send(
        &self,
        id: &str,
        message: &str,
        parent_cancel: CancellationToken,
    ) -> crate::error::Result<SubagentResult> {
        let started_at = chrono::Utc::now();
        let start = Instant::now();

        let mut entry = {
            let mut agents = self.agents.lock().await;
            let pos = agents.iter().position(|(k, _)| k == id).ok_or_else(|| {
                crate::error::Error::Other(format!(
                    "No agent with ID '{}'. It may have been evicted, never stored, or is currently running.",
                    id
                ))
            })?;
            agents.remove(pos).expect("pos was found via position()").1
        };

        let usage_before = entry.usage_at_pause.clone();
        let messages_at_pause = entry.messages_at_pause;

        send_event(
            &self.parent_event_tx,
            AgentEvent::SubagentResumed {
                agent_id: id.to_string(),
                description: entry.description.clone(),
                prompt: message.to_string(),
                resumed_at: started_at,
            },
        );

        let event_task =
            self.spawn_event_forwarder(entry.handle.subscribe(), id, &entry.description);
        let cancel_bridge = spawn_cancel_bridge(entry.handle.clone(), parent_cancel.clone());

        // Maintain the registry invariant during resume: the agent has
        // been removed from `agents` (idle), so it must show up in
        // `running_handles` until we push it back. Without this,
        // `find_agent` and `send_to_running` lose the agent for the
        // duration of the resume.
        self.track_running(id, entry.handle.clone(), entry.description.clone())
            .await;

        let prompt_result = entry.handle.prompt_and_wait(message).await;
        cancel_bridge.abort();
        event_task.abort();

        let messages = entry.handle.messages().await.unwrap_or_default();
        let current_state = entry.handle.state().await.unwrap_or_default();
        let (delta_input, delta_output) = usage_delta(&usage_before, &current_state.total_usage);
        let tool_use_count = count_tool_uses_since(&messages, messages_at_pause);
        let text = extract_final_text(&messages);

        let transcript_path = record_transcript(id, &messages)
            .await
            .map(|p| p.display().to_string());
        entry.usage_at_pause = current_state.total_usage.clone();
        entry.messages_at_pause = messages.len();
        let description = entry.description.clone();
        // Push back to idle BEFORE untracking from running so the agent
        // is always visible in at least one map. Brief overlap (in both)
        // is harmless: `find_agent` checks `running_handles` first and
        // returns Running, which is still accurate during the post-prompt
        // bookkeeping before idle storage takes over.
        self.agents.lock().await.push_back((id.to_string(), entry));
        self.untrack_running(id).await;

        let completed_at = chrono::Utc::now();
        let duration_ms = start.elapsed().as_millis() as u64;
        let outcome = outcome_from(&prompt_result, &parent_cancel);

        send_event(
            &self.parent_event_tx,
            AgentEvent::SubagentCompleted {
                agent_id: id.to_string(),
                description,
                outcome,
                started_at,
                completed_at,
                duration_ms,
                usage: Usage {
                    input: delta_input,
                    output: delta_output,
                    ..Default::default()
                },
                tool_use_count,
                worktree_path: None,
                worktree_branch: None,
            },
        );

        prompt_result?;

        Ok(SubagentResult {
            agent_id: id.to_string(),
            text,
            input_tokens: delta_input,
            output_tokens: delta_output,
            tool_use_count,
            duration_ms,
            worktree_path: None,
            worktree_branch: None,
            transcript_path,
        })
    }
}

// ---------- Internal: configuration + run mechanics ----------

impl AgentManager {
    /// Build a fully-configured `AgentBuilder` ready to spawn: parent
    /// config tweaked for a subagent, cwd resolved, approval policy +
    /// interaction sender wired, tools added, system prompt assembled,
    /// history seeded, `pre_handle` stamped with id + manager, and every
    /// tool given a chance to capture the handle via `bind_to_agent`.
    ///
    /// `fallback_cwd` is used when `opts.cwd` is `None` — typically the
    /// worktree path for `run_agent_inner`. For `spawn_interactive` pass
    /// `None`.
    async fn configure_builder(
        self: &Arc<Self>,
        spec: &AgentSpec,
        opts: &SpawnOpts,
        agent_id: &str,
        fallback_cwd: Option<&str>,
    ) -> crate::error::Result<AgentBuilder> {
        let cwd = opts
            .cwd
            .clone()
            .or_else(|| fallback_cwd.map(String::from))
            .unwrap_or_else(|| {
                std::env::current_dir()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| ".".into())
            });

        let mut agent_cfg = self.parent_config.clone();
        agent_cfg.system_prompt = None;
        agent_cfg.max_turns = Some(spec.max_turns);
        agent_cfg.max_tokens = None;
        // Disable prompt caching for subagents — short-lived, never read back.
        agent_cfg.cache_scope = None;
        agent_cfg.cache_ttl = None;
        if let Some(ref model) = opts.model {
            agent_cfg.model = model.clone();
        }

        let mut builder = AgentBuilder::new(agent_cfg, self.transport.clone());
        builder.set_cwd(&cwd);

        let policy = opts
            .approval_policy
            .clone()
            .unwrap_or_else(|| self.parent_approval_policy.lock().clone());
        builder.set_approval_policy(policy);

        if let Some(sub_tx) = self.build_subagent_interaction_sender(agent_id) {
            builder.set_interaction_sender(sub_tx);
        }

        for tool in &spec.tools {
            builder.add_tool(tool.clone());
        }

        let tool_names = builder.tool_names();
        let system_prompt =
            crate::prompts::build_subagent_prompt(&spec.system_prompt, &tool_names, &cwd);
        builder.set_system_prompt(system_prompt);

        // Seed history. `seed_messages` (explicit vector from the host)
        // wins over `inherit_history_from` (lookup by id).
        if let Some(seed) = opts.seed_messages.clone() {
            builder.set_messages(seed);
        } else if let Some(ref source_id) = opts.inherit_history_from {
            let source_messages = self.fetch_stored_messages(source_id).await?;
            builder.set_messages(source_messages);
        }

        // Stamp the pre-spawn handle with this agent's id + manager
        // backref, then give every tool a chance to capture it.
        let pre = builder.pre_handle();
        let _ = pre.agent_id.set(agent_id.to_string());
        let _ = pre.manager.set(Arc::downgrade(self));
        for tool in &spec.tools {
            tool.bind_to_agent(&pre);
        }

        Ok(builder)
    }

    async fn run_subagent(
        self: &Arc<Self>,
        spec: &AgentSpec,
        initial_prompt: &str,
        opts: &SpawnOpts,
        cancel: CancellationToken,
        agent_id: &str,
    ) -> crate::error::Result<(SubagentResult, AgentHandle)> {
        let started_at = chrono::Utc::now();
        let start = Instant::now();

        // Emit Started before any setup work that could fail (worktree
        // creation, builder configuration). Hosts otherwise see an
        // unbracketed Err with no record of the spawn attempt.
        send_event(
            &self.parent_event_tx,
            AgentEvent::SubagentStarted {
                agent_id: agent_id.to_string(),
                spec_name: opts.spec_name.clone(),
                description: opts.description.clone(),
                prompt: initial_prompt.to_string(),
                started_at,
            },
        );

        // Helper to emit Completed{Failed} for setup-time failures and
        // return the same Err to the caller. Keeps the bracket complete
        // even when we never produced a handle. `wt_path`/`wt_branch`
        // carry teardown signal in the post-worktree-creation path
        // (configure_builder failed but the worktree was already made),
        // so a non-clean cleanup is still surfaced.
        let emit_setup_failure =
            |e: crate::error::Error,
             wt_path: Option<String>,
             wt_branch: Option<String>| async {
                let completed_at = chrono::Utc::now();
                let duration_ms = start.elapsed().as_millis() as u64;
                send_event(
                    &self.parent_event_tx,
                    AgentEvent::SubagentCompleted {
                        agent_id: agent_id.to_string(),
                        description: opts.description.clone(),
                        outcome: SubagentOutcome::Failed {
                            reason: e.to_string(),
                        },
                        started_at,
                        completed_at,
                        duration_ms,
                        usage: Usage::default(),
                        tool_use_count: 0,
                        worktree_path: wt_path,
                        worktree_branch: wt_branch,
                    },
                );
                e
            };

        let worktree = if spec.allows_worktree && opts.isolation == Some(Isolation::Worktree) {
            match create_worktree(agent_id).await {
                Ok(wt) => Some(wt),
                Err(e) => {
                    // No worktree exists — nothing to tear down.
                    let err = crate::error::Error::Other(format!("Worktree setup failed: {e}"));
                    return Err(emit_setup_failure(err, None, None).await);
                }
            }
        } else {
            None
        };

        let inner = self
            .run_agent_inner(spec, initial_prompt, opts, agent_id, &worktree, cancel.clone())
            .await;
        let (wt_path, wt_branch) = teardown_worktree(&worktree).await;

        let completed_at = chrono::Utc::now();
        let duration_ms = start.elapsed().as_millis() as u64;

        let RunOutcome {
            handle,
            prompt_result,
            mut result,
        } = match inner {
            Ok(o) => o,
            Err(e) => {
                // configure_builder failed (e.g. inherit_history_from
                // pointed at an unknown agent). No handle exists, so
                // nothing to record beyond the Failed outcome — but the
                // worktree was created and torn down above, so its
                // teardown signal rides along on the failure event.
                return Err(emit_setup_failure(e, wt_path, wt_branch).await);
            }
        };

        // Always record the transcript — failed runs are usually the most
        // diagnostically interesting. `messages()` returns `None` only if
        // the actor is gone; for a panicked actor we'll have less context
        // but still emit Completed.
        let messages = handle.messages().await.unwrap_or_default();
        let transcript_path = record_transcript(agent_id, &messages)
            .await
            .map(|p| p.display().to_string());

        let outcome = outcome_from(&prompt_result, &cancel);
        result.transcript_path = transcript_path;
        result.worktree_path = wt_path.clone();
        result.worktree_branch = wt_branch.clone();
        result.duration_ms = duration_ms;

        let usage = Usage {
            input: result.input_tokens,
            output: result.output_tokens,
            ..Default::default()
        };
        let tool_use_count = result.tool_use_count;

        send_event(
            &self.parent_event_tx,
            AgentEvent::SubagentCompleted {
                agent_id: agent_id.to_string(),
                description: opts.description.clone(),
                outcome,
                started_at,
                completed_at,
                duration_ms,
                usage,
                tool_use_count,
                worktree_path: wt_path,
                worktree_branch: wt_branch,
            },
        );

        match prompt_result {
            Ok(()) => Ok((result, handle)),
            Err(e) => Err(e),
        }
    }

    async fn run_agent_inner(
        self: &Arc<Self>,
        spec: &AgentSpec,
        initial_prompt: &str,
        opts: &SpawnOpts,
        agent_id: &str,
        worktree: &Option<WorktreeInfo>,
        cancel: CancellationToken,
    ) -> crate::error::Result<RunOutcome> {
        let wt_cwd = worktree.as_ref().map(|w| w.path.display().to_string());
        let builder = self
            .configure_builder(spec, opts, agent_id, wt_cwd.as_deref())
            .await?;
        let handle = builder.spawn();

        let event_task =
            self.spawn_event_forwarder(handle.subscribe(), agent_id, &opts.description);
        let cancel_bridge = spawn_cancel_bridge(handle.clone(), cancel.clone());

        self.track_running(agent_id, handle.clone(), opts.description.clone())
            .await;

        let prompt_result = handle.prompt_and_wait(initial_prompt).await;
        cancel_bridge.abort();
        event_task.abort();

        // Compute partial metrics regardless of outcome — we want failed
        // runs to surface tokens used, tool calls made, and any final
        // text the model produced before the error.
        let messages = handle.messages().await.unwrap_or_default();
        let text = extract_final_text(&messages);
        let state = handle.state().await.unwrap_or_default();
        let tool_use_count = count_tool_uses_since(&messages, 0);

        let result = SubagentResult {
            agent_id: agent_id.to_string(),
            text,
            input_tokens: state.total_usage.input,
            output_tokens: state.total_usage.output,
            tool_use_count,
            duration_ms: 0,
            worktree_path: None,
            worktree_branch: None,
            transcript_path: None,
        };
        Ok(RunOutcome {
            handle,
            prompt_result,
            result,
        })
    }
}

/// Carries everything needed to finalize a run after `prompt_and_wait`,
/// regardless of whether the prompt succeeded. Letting the handle and
/// metrics survive a prompt error is what makes failure-path transcripts
/// possible.
struct RunOutcome {
    handle: AgentHandle,
    prompt_result: crate::error::Result<()>,
    result: SubagentResult,
}

// ---------- Free helpers ----------

/// Cancel-bridge: forwards a parent cancellation to the subagent's handle.
fn spawn_cancel_bridge(handle: AgentHandle, parent_cancel: CancellationToken) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        parent_cancel.cancelled().await;
        handle.abort();
    })
}

/// Tear down a worktree if present. Returns `(path, branch)` for the
/// `SubagentCompleted` event — `None`/`None` when the tree was cleanly
/// removed (no diff, no untracked); `Some`/`Some` when changes were left
/// behind so the host can surface them.
async fn teardown_worktree(worktree: &Option<WorktreeInfo>) -> (Option<String>, Option<String>) {
    match worktree {
        Some(wt) => match cleanup_worktree(wt).await {
            Ok(true) => (None, None),
            _ => (Some(wt.path.display().to_string()), Some(wt.branch.clone())),
        },
        None => (None, None),
    }
}

/// Map a prompt result + cancellation state to a `SubagentOutcome`.
fn outcome_from(prompt_result: &crate::error::Result<()>, cancel: &CancellationToken) -> SubagentOutcome {
    match prompt_result {
        Ok(()) if cancel.is_cancelled() => SubagentOutcome::Aborted {
            reason: "cancelled by parent".to_string(),
        },
        Ok(()) => SubagentOutcome::Completed,
        Err(e) if cancel.is_cancelled() => SubagentOutcome::Aborted {
            reason: e.to_string(),
        },
        Err(e) => SubagentOutcome::Failed {
            reason: e.to_string(),
        },
    }
}

/// Tokens used since `before`, saturating to avoid wrap on bookkeeping
/// glitches.
fn usage_delta(before: &Usage, after: &Usage) -> (u64, u64) {
    (
        after.input.saturating_sub(before.input),
        after.output.saturating_sub(before.output),
    )
}

/// Count assistant tool calls in `messages[from..]`. `from` is clamped
/// to the slice length so callers don't have to bounds-check.
fn count_tool_uses_since(messages: &[Message], from: usize) -> u32 {
    let start = from.min(messages.len());
    messages[start..]
        .iter()
        .map(|m| match m {
            Message::Assistant { content, .. } => content
                .iter()
                .filter(|c| matches!(c, Content::ToolCall { .. }))
                .count(),
            _ => 0,
        })
        .sum::<usize>() as u32
}

/// Return the text of the literal *last* assistant message. Empty string
/// when the last assistant turn was tool-calls only or there is no
/// assistant message at all. We deliberately don't fall back to earlier
/// turns: returning stale text from before the actual work would mislead
/// the parent into thinking it had the agent's final answer.
fn extract_final_text(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find_map(|m| match m {
            Message::Assistant { content, .. } => Some(
                content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(""),
            ),
            _ => None,
        })
        .unwrap_or_default()
}

// ---------- Tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DequeueMode;
    use async_trait::async_trait;
    use tau_ai::{AssistantMetadata, Content, Message};

    #[test]
    fn test_extract_final_text() {
        let messages = vec![
            Message::user("hello"),
            Message::Assistant {
                content: vec![Content::text("first response")],
                metadata: AssistantMetadata::default(),
            },
            Message::user("follow up"),
            Message::Assistant {
                content: vec![Content::text("second response")],
                metadata: AssistantMetadata::default(),
            },
        ];
        assert_eq!(extract_final_text(&messages), "second response");
    }

    #[test]
    fn test_extract_final_text_empty() {
        let messages: Vec<Message> = vec![];
        assert_eq!(extract_final_text(&messages), "");
    }

    /// Tool-only last assistant turn must NOT fall back to earlier text —
    /// returning stale "I'll start by doing X" instead of "" would mislead
    /// the parent into thinking the agent had a final answer.
    #[test]
    fn test_extract_final_text_tool_only_last_returns_empty() {
        let messages = vec![
            Message::Assistant {
                content: vec![Content::text("plan: I'll grep then edit")],
                metadata: AssistantMetadata::default(),
            },
            Message::user("ok"),
            Message::Assistant {
                content: vec![Content::ToolCall {
                    id: "c1".into(),
                    name: "grep".into(),
                    arguments: serde_json::json!({}),
                }],
                metadata: AssistantMetadata::default(),
            },
        ];
        assert_eq!(extract_final_text(&messages), "");
    }

    #[test]
    fn test_outcome_from_completed() {
        let cancel = CancellationToken::new();
        let res: crate::error::Result<()> = Ok(());
        assert!(matches!(outcome_from(&res, &cancel), SubagentOutcome::Completed));
    }

    #[test]
    fn test_outcome_from_aborted_when_cancelled() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let ok: crate::error::Result<()> = Ok(());
        assert!(matches!(
            outcome_from(&ok, &cancel),
            SubagentOutcome::Aborted { .. }
        ));
        let err: crate::error::Result<()> = Err(crate::error::Error::Other("boom".into()));
        assert!(matches!(
            outcome_from(&err, &cancel),
            SubagentOutcome::Aborted { .. }
        ));
    }

    #[test]
    fn test_outcome_from_failed() {
        let cancel = CancellationToken::new();
        let err: crate::error::Result<()> = Err(crate::error::Error::Other("boom".into()));
        assert!(matches!(
            outcome_from(&err, &cancel),
            SubagentOutcome::Failed { .. }
        ));
    }

    #[test]
    fn test_usage_delta_saturates() {
        let before = Usage {
            input: 10,
            output: 5,
            ..Default::default()
        };
        let after = Usage {
            input: 25,
            output: 4, // less than before — saturates to 0
            ..Default::default()
        };
        assert_eq!(usage_delta(&before, &after), (15, 0));
    }

    #[test]
    fn test_count_tool_uses_since_clamps() {
        let messages = vec![Message::user("hi")];
        // `from` past end of slice: returns 0, doesn't panic.
        assert_eq!(count_tool_uses_since(&messages, 99), 0);
    }

    struct DummyTransport;

    #[async_trait]
    impl crate::transport::Transport for DummyTransport {
        async fn run(
            &self,
            _messages: Vec<tau_ai::Message>,
            _config: &crate::transport::AgentRunConfig,
            _cancel: CancellationToken,
        ) -> tau_ai::Result<crate::transport::AgentEventStream> {
            unimplemented!()
        }
    }

    fn make_test_config() -> AgentConfig {
        AgentConfig {
            system_prompt: None,
            model: tau_ai::Model {
                id: "test".into(),
                name: "test".into(),
                api: tau_ai::Api::AnthropicMessages,
                provider: tau_ai::Provider::Anthropic,
                base_url: "http://localhost".into(),
                reasoning: false,
                input_types: vec![],
                cost: tau_ai::CostInfo::default(),
                context_window: 200000,
                max_tokens: 4096,
                headers: Default::default(),
            },
            reasoning: tau_ai::ReasoningLevel::Off,
            thinking_adaptive: false,
            max_tokens: None,
            max_turns: None,
            compaction: crate::compaction::CompactionConfig::default(),
            steering_mode: DequeueMode::All,
            follow_up_mode: DequeueMode::All,
            cache_scope: None,
            cache_ttl: None,
            system_prompt_boundary: None,
        }
    }

    fn make_test_manager(max_agents: usize) -> Arc<AgentManager> {
        let (tx, _rx) = tokio::sync::broadcast::channel::<AgentEvent>(16);
        let transport: Arc<dyn crate::transport::Transport> = Arc::new(DummyTransport);
        let config = make_test_config();
        Arc::new(AgentManager::new(tx, config, transport, max_agents))
    }

    #[tokio::test]
    async fn test_event_forwarder_wraps_events() {
        let manager = make_test_manager(20);

        let (child_tx, _) = tokio::sync::broadcast::channel::<AgentEvent>(16);
        let mut parent_rx = manager.parent_event_tx.subscribe();

        let task = manager.spawn_event_forwarder(child_tx.subscribe(), "agent-123", "test task");

        child_tx
            .send(AgentEvent::ToolExecutionStart {
                tool_call_id: "c1".into(),
                tool_name: "bash".into(),
                arguments: serde_json::json!({}),
                activity: "Running bash".into(),
            })
            .unwrap();

        drop(child_tx);
        let _ = task.await;

        let mut received = vec![];
        while let Ok(event) = parent_rx.try_recv() {
            if let AgentEvent::Subagent {
                agent_id,
                description,
                ..
            } = event
            {
                assert_eq!(agent_id, "agent-123");
                assert_eq!(description, "test task");
                received.push(true);
            }
        }

        assert!(!received.is_empty(), "should have forwarded events");
    }

    #[tokio::test]
    async fn test_event_forwarder_survives_lagged() {
        let manager = make_test_manager(20);

        let (child_tx, _keep_alive) = tokio::sync::broadcast::channel::<AgentEvent>(2);
        let mut parent_rx = manager.parent_event_tx.subscribe();

        let forwarder_rx = child_tx.subscribe();

        for i in 0..5 {
            child_tx
                .send(AgentEvent::ToolExecutionStart {
                    tool_call_id: format!("c{i}"),
                    tool_name: "bash".into(),
                    arguments: serde_json::json!({}),
                    activity: format!("Running bash {i}"),
                })
                .unwrap();
        }

        let task = manager.spawn_event_forwarder(forwarder_rx, "agent-lag", "lag test");

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        child_tx
            .send(AgentEvent::ToolExecutionStart {
                tool_call_id: "c-after-lag".into(),
                tool_name: "bash".into(),
                arguments: serde_json::json!({}),
                activity: "after lag".into(),
            })
            .unwrap();

        drop(child_tx);
        let _ = task.await;

        let mut saw_post_lag_event = false;
        while let Ok(event) = parent_rx.try_recv() {
            if let AgentEvent::Subagent { event, .. } = event {
                if let AgentEvent::ToolExecutionStart { tool_call_id, .. } = *event {
                    if tool_call_id == "c-after-lag" {
                        saw_post_lag_event = true;
                    }
                }
            }
        }

        assert!(
            saw_post_lag_event,
            "forwarder should keep delivering events after Lagged"
        );
    }

    /// Registry invariant: spec exists iff id is in `agents` ∪ `running_handles`.
    #[tokio::test]
    async fn test_registry_invariant_after_lifecycle_ops() {
        let manager = make_test_manager(20);

        // insert_spec + track_running → both visible
        let id = "test-1";
        let spec = AgentSpec {
            system_prompt: "x".into(),
            tools: vec![],
            max_turns: 1,
            allows_worktree: false,
            allowed_subagent_specs: None,
        };
        manager.insert_spec(id, Arc::new(spec)).await;
        assert!(manager.spec_for(id).await.is_some());

        // forget when not stored → spec removed
        manager.forget(id).await;
        assert!(manager.spec_for(id).await.is_none());
    }
}
