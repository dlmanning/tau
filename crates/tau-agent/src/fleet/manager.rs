//! `AgentManager` — composition root for the fleet.
//!
//! Holds the registry, the default-approval policy, and the parent's
//! event + interaction channels. Lifecycle methods (`spawn`, `send`,
//! `respec`, `adopt`, …) are thin wrappers that build a
//! [`LifecycleCtx`] and dispatch to free functions in
//! [`crate::fleet::lifecycle`]. The single-responsibility split lives
//! at the module level, not the type level — `AgentManager` is the
//! one type a host imports.

use std::sync::Arc;

use parking_lot::Mutex as ParkingMutex;
use serde::{Deserialize, Serialize};
use tau_ai::{Message, Model};
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::core::approval::{ApprovalPolicy, DefaultPolicy};
use crate::core::config::AgentConfig;
use crate::core::handle::AgentHandle;
use crate::core::interaction::InteractionRequest;
use crate::core::tool::BoxedTool;
use crate::core::transport::Transport;
use crate::fleet::lifecycle::{self, LifecycleCtx};
pub use crate::fleet::registry::Status as AgentStatus;
use crate::fleet::registry::{Located, Registry};
use crate::fleet::result::SubagentResult;
use crate::types::error::Result;
use crate::types::events::AgentEvent;

/// Immutable per-agent input. To change any field, spawn a new agent
/// (typically via [`AgentManager::respec`]).
///
/// `Clone` shares the underlying tool `Arc`s. Tools are stateless
/// w.r.t. the agent — they receive identity per call via
/// [`ExecutionContext::agent_id`](crate::core::tool::ExecutionContext::agent_id),
/// not by capturing it on construction — so sharing tool instances
/// across multiple agents is safe and expected.
#[derive(Clone)]
pub struct AgentSpec {
    pub system_prompt: String,
    pub tools: Vec<BoxedTool>,
    pub max_turns: u32,
    /// Per-spec opt-in for [`Isolation::Worktree`]. Ignored unless the
    /// spawn opts request worktree isolation.
    pub allows_worktree: bool,
    /// Spec names this agent may spawn as subagents. Carried through
    /// to the host's spawn-time resolver / `AgentTool`. The runtime
    /// does not enforce this directly.
    pub allowed_subagent_specs: Option<Vec<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Isolation {
    /// Run inside a fresh git worktree on a per-agent branch. Honored
    /// only when the spec sets `allows_worktree = true`.
    Worktree,
}

#[derive(Default, Clone)]
pub struct SpawnOpts {
    pub description: String,
    pub model: Option<Model>,
    pub cwd: Option<String>,
    pub isolation: Option<Isolation>,
    /// Inherit another stored agent's full message history, then
    /// apply the spawn prompt as a follow-up. The named agent must
    /// already be stored or running.
    pub inherit_history_from: Option<String>,
    pub approval_policy: Option<Arc<dyn ApprovalPolicy>>,
    pub spec_name: Option<String>,
    /// Lower-level: explicit message vector. Wins over
    /// `inherit_history_from` when both are set.
    pub seed_messages: Option<Vec<Message>>,
    /// Depth to stamp on the spawned agent's
    /// [`ExecutionContext::subagent_depth`](crate::ExecutionContext::subagent_depth).
    /// Callers that originate spawns from inside a tool should set this
    /// to `ctx.subagent_depth + 1`; root spawns leave it at `0`.
    pub subagent_depth: u32,
}

pub struct AgentManager {
    registry: Arc<Registry>,
    transport: Arc<dyn Transport>,
    parent_config: AgentConfig,
    parent_event_tx: broadcast::Sender<AgentEvent>,
    parent_interaction_tx: Option<mpsc::Sender<InteractionRequest>>,
    /// Default approval policy for spawned subagents. Swapped at
    /// runtime via [`Self::set_default_approval_policy`]; in-flight
    /// subagents keep the policy they were spawned with.
    default_approval: ParkingMutex<Arc<dyn ApprovalPolicy>>,
    /// Per-subagent interaction-router channel capacity. See
    /// [`crate::fleet::bus::DEFAULT_INTERACTION_ROUTER_CAPACITY`].
    interaction_router_capacity: usize,
}

impl AgentManager {
    pub fn new(
        parent_event_tx: broadcast::Sender<AgentEvent>,
        parent_config: AgentConfig,
        transport: Arc<dyn Transport>,
        max_agents: usize,
    ) -> Self {
        Self {
            registry: Arc::new(Registry::new(max_agents)),
            transport,
            parent_config,
            parent_event_tx,
            parent_interaction_tx: None,
            default_approval: ParkingMutex::new(Arc::new(DefaultPolicy)),
            interaction_router_capacity: crate::fleet::bus::DEFAULT_INTERACTION_ROUTER_CAPACITY,
        }
    }

    pub fn with_parent_interaction_sender(mut self, tx: mpsc::Sender<InteractionRequest>) -> Self {
        self.parent_interaction_tx = Some(tx);
        self
    }

    pub fn with_default_approval_policy(self, policy: Arc<dyn ApprovalPolicy>) -> Self {
        *self.default_approval.lock() = policy;
        self
    }

    /// Override the per-subagent interaction-router channel capacity.
    /// Increase this when subagents are expected to emit bursts of
    /// concurrent interaction requests (e.g. many gated tool calls per
    /// turn).
    pub fn with_interaction_router_capacity(mut self, capacity: usize) -> Self {
        self.interaction_router_capacity = capacity;
        self
    }

    pub fn set_default_approval_policy(&self, policy: Arc<dyn ApprovalPolicy>) {
        *self.default_approval.lock() = policy;
    }

    fn ctx(&self) -> LifecycleCtx {
        LifecycleCtx {
            registry: Arc::clone(&self.registry),
            transport: Arc::clone(&self.transport),
            parent_config: self.parent_config.clone(),
            parent_event_tx: self.parent_event_tx.clone(),
            parent_interaction_tx: self.parent_interaction_tx.clone(),
            default_approval: self.default_approval.lock().clone(),
            interaction_router_capacity: self.interaction_router_capacity,
        }
    }

    // ─── Spec lookup ─────────────────────────────────────────────────

    pub fn spec_for(&self, agent_id: &str) -> Option<Arc<AgentSpec>> {
        self.registry.spec_for(agent_id)
    }

    // ─── Lifecycle ───────────────────────────────────────────────────

    pub async fn spawn(
        &self,
        spec: impl Into<Arc<AgentSpec>>,
        initial_prompt: String,
        opts: SpawnOpts,
        cancel: CancellationToken,
    ) -> Result<SubagentResult> {
        lifecycle::spawn(&self.ctx(), spec, initial_prompt, opts, cancel).await
    }

    pub async fn spawn_interactive(
        &self,
        spec: impl Into<Arc<AgentSpec>>,
        opts: SpawnOpts,
    ) -> Result<(AgentHandle, String)> {
        lifecycle::spawn_interactive(&self.ctx(), spec, opts).await
    }

    /// Clean up an interactive agent the caller is done driving. Drops
    /// it from the running bucket and drops its spec if it isn't also
    /// held in idle storage.
    pub fn remove_interactive(&self, agent_id: &str) {
        self.registry.drop_running(agent_id);
    }

    /// Spawn a subagent that runs in the background. Returns
    /// immediately with the agent's id. On completion (success or
    /// failure), a `FollowUp` message is posted to `parent_handle`.
    pub async fn spawn_background(
        &self,
        spec: impl Into<Arc<AgentSpec>>,
        initial_prompt: String,
        opts: SpawnOpts,
        parent_handle: AgentHandle,
        parent_cancel: CancellationToken,
    ) -> String {
        lifecycle::spawn_background(
            &self.ctx(),
            spec,
            initial_prompt,
            opts,
            parent_handle,
            parent_cancel,
        )
        .await
    }

    pub async fn send(
        &self,
        id: &str,
        message: &str,
        cancel: CancellationToken,
    ) -> Result<SubagentResult> {
        lifecycle::send(&self.ctx(), id, message, cancel).await
    }

    pub async fn respec(
        &self,
        agent_id: &str,
        new_spec: impl Into<Arc<AgentSpec>>,
    ) -> Result<AgentHandle> {
        lifecycle::respec(&self.ctx(), agent_id, new_spec).await
    }

    /// Register an externally-built handle so `spec_for` / `respec`
    /// work for it. Returns the agent's id (a freshly-minted UUID if
    /// the handle didn't already have one). The handle is recorded in
    /// the registry's `adopted` bucket; the fleet does not manage its
    /// actor lifecycle, but the spec is queryable and a respec is
    /// available.
    pub fn adopt(
        &self,
        handle: &AgentHandle,
        description: impl Into<String>,
        spec: impl Into<Arc<AgentSpec>>,
    ) -> String {
        lifecycle::adopt(&self.registry, handle, description, spec)
    }

    // ─── Lookups ─────────────────────────────────────────────────────

    /// Resolve an id-or-description to an agent. See
    /// [`Registry::find`] for the resolution order.
    pub fn find_agent(&self, name_or_id: &str) -> Option<Located> {
        self.registry.find(name_or_id)
    }

    /// Clone of a running agent's handle. Used by `AgentTool` to look
    /// up the parent for background-spawn follow-up tracking.
    pub fn handle_for(&self, agent_id: &str) -> Option<AgentHandle> {
        self.registry.handle_for(agent_id)
    }

    // ─── Steering shortcut for "find by id, then steer" ──────────────

    pub async fn send_to_running(&self, id: &str, message: Message) -> bool {
        let Some(handle) = self.registry.handle_for(id) else {
            return false;
        };
        let _ = handle.steer(message).await;
        true
    }
}
