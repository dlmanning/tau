//! AgentBuilder — setup phase, consumed by spawn().

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicU32};

use futures::FutureExt;
use tokio::sync::{Notify, broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::actor::run_actor;
use crate::approval::{ApprovalPolicy, DefaultApprovalPolicy};
use crate::config::AgentConfig;
use crate::events::AgentEvent;
use crate::handle::AgentHandle;
use crate::state::AgentState;
use crate::tool::BoxedTool;
use crate::transport::Transport;

use parking_lot::Mutex as ParkingMutex;
use tau_ai::{Message, ServerTool};

/// Type alias for the transform context callback.
pub(crate) type TransformContextFn = dyn Fn(Vec<Message>) -> Vec<Message> + Send + Sync;

/// Default capacity of the urgent (Steer/FollowUp) command channel. Sized
/// for fan-in from many background subagents and bursty user steering.
pub const DEFAULT_URGENT_CAPACITY: usize = 256;

/// Default capacity of the normal command channel (config setters, queries,
/// prompt requests). Smaller because volume is naturally bounded.
pub const DEFAULT_NORMAL_CAPACITY: usize = 64;

/// Setup phase. Configure the agent, then call spawn() to start the actor.
///
/// Shared primitives (command channel, cancel token, etc.) are created eagerly
/// so that `pre_handle()` can return a working `AgentHandle` before `spawn()`.
/// This resolves the circular dependency where the AgentTool factory needs a
/// handle to the subagent before the subagent is spawned.
pub struct AgentBuilder {
    config: AgentConfig,
    transport: Arc<dyn Transport>,
    tools: Vec<BoxedTool>,
    server_tools: Vec<ServerTool>,
    interaction_tx: Option<mpsc::Sender<crate::interaction::InteractionRequest>>,
    approval_policy: Arc<dyn ApprovalPolicy>,
    cwd: Option<PathBuf>,
    transform_context: Option<Arc<TransformContextFn>>,
    initial_messages: Vec<Message>,
    previous_summary: Option<String>,

    // Pre-created shared primitives (created in new(), used by pre_handle() and spawn())
    event_tx: broadcast::Sender<AgentEvent>,
    urgent_tx: mpsc::Sender<crate::command::Command>,
    urgent_rx: Option<mpsc::Receiver<crate::command::Command>>,
    normal_tx: mpsc::Sender<crate::command::Command>,
    normal_rx: Option<mpsc::Receiver<crate::command::Command>>,
    cancel: Arc<ParkingMutex<CancellationToken>>,
    is_running: Arc<AtomicBool>,
    pending_follow_ups: Arc<AtomicU32>,
    /// Written by the actor's catch_unwind wrapper on panic.
    shutdown_reason: Arc<ParkingMutex<Option<String>>>,
    /// Notified after `shutdown_reason` is written.
    shutdown_signaled: Arc<Notify>,
    /// Optional agent id slot, shared with all `AgentHandle` clones. The
    /// `AgentManager` (or a host calling `manager.adopt`) fills this in
    /// when this handle is registered.
    agent_id: Arc<OnceLock<String>>,
    /// Optional manager weak reference, shared with all `AgentHandle`
    /// clones. Filled in alongside `agent_id`.
    manager: Arc<OnceLock<std::sync::Weak<crate::manager::AgentManager>>>,
}

impl AgentBuilder {
    pub fn new(config: AgentConfig, transport: Arc<dyn Transport>) -> Self {
        Self::with_channel_capacities(
            config,
            transport,
            DEFAULT_URGENT_CAPACITY,
            DEFAULT_NORMAL_CAPACITY,
        )
    }

    /// Build a new `AgentBuilder` with custom command channel capacities.
    /// Use this when expected steering or follow-up volume exceeds the
    /// defaults — for example, ambient agents that fan in many background
    /// subagent completions.
    pub fn with_channel_capacities(
        config: AgentConfig,
        transport: Arc<dyn Transport>,
        urgent_capacity: usize,
        normal_capacity: usize,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        let (urgent_tx, urgent_rx) = mpsc::channel(urgent_capacity);
        let (normal_tx, normal_rx) = mpsc::channel(normal_capacity);
        let cancel = Arc::new(ParkingMutex::new(CancellationToken::new()));
        let is_running = Arc::new(AtomicBool::new(false));
        let pending_follow_ups = Arc::new(AtomicU32::new(0));
        let shutdown_reason = Arc::new(ParkingMutex::new(None));
        let shutdown_signaled = Arc::new(Notify::new());
        let agent_id = Arc::new(OnceLock::new());
        let manager = Arc::new(OnceLock::new());

        Self {
            config,
            transport,
            tools: vec![],
            server_tools: vec![],
            interaction_tx: None,
            approval_policy: Arc::new(DefaultApprovalPolicy),
            cwd: None,
            transform_context: None,
            initial_messages: vec![],
            previous_summary: None,
            event_tx,
            urgent_tx,
            urgent_rx: Some(urgent_rx),
            normal_tx,
            normal_rx: Some(normal_rx),
            cancel,
            is_running,
            pending_follow_ups,
            shutdown_reason,
            shutdown_signaled,
            agent_id,
            manager,
        }
    }

    // === Builder methods ===

    pub fn add_tool(&mut self, tool: Arc<dyn crate::tool::Tool>) -> &mut Self {
        self.tools.push(tool);
        self
    }

    pub fn set_tools(&mut self, tools: Vec<BoxedTool>) -> &mut Self {
        self.tools = tools;
        self
    }

    pub fn add_server_tool(&mut self, tool: ServerTool) -> &mut Self {
        self.server_tools.push(tool);
        self
    }

    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) -> &mut Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    pub fn set_interaction_sender(
        &mut self,
        tx: mpsc::Sender<crate::interaction::InteractionRequest>,
    ) -> &mut Self {
        self.interaction_tx = Some(tx);
        self
    }

    pub fn set_approval_policy(&mut self, policy: Arc<dyn ApprovalPolicy>) -> &mut Self {
        self.approval_policy = policy;
        self
    }

    pub fn set_cwd(&mut self, cwd: impl Into<PathBuf>) -> &mut Self {
        self.cwd = Some(cwd.into());
        self
    }

    pub fn set_messages(&mut self, messages: Vec<Message>) -> &mut Self {
        self.initial_messages = messages;
        self
    }

    pub fn set_previous_summary(&mut self, summary: Option<String>) -> &mut Self {
        self.previous_summary = summary;
        self
    }

    pub fn set_transform_context(&mut self, f: Arc<TransformContextFn>) -> &mut Self {
        self.transform_context = Some(f);
        self
    }

    // === Read access (needed before spawn for AgentManager setup) ===

    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    pub fn tools(&self) -> &[BoxedTool] {
        &self.tools
    }

    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Get the event sender. Available before spawn.
    pub fn event_sender(&self) -> broadcast::Sender<AgentEvent> {
        self.event_tx.clone()
    }

    /// Get a working `AgentHandle` before calling `spawn()`.
    ///
    /// This handle is connected to the same command channel and shared primitives
    /// that `spawn()` will use. It can be given to the AgentTool factory to resolve
    /// the circular dependency (factory needs handle → handle needs spawn → spawn
    /// needs tools including agent tool).
    ///
    /// The handle is fully functional — commands sent through it will be received
    /// by the actor once `spawn()` starts it.
    pub fn pre_handle(&self) -> AgentHandle {
        AgentHandle {
            urgent_tx: self.urgent_tx.clone(),
            normal_tx: self.normal_tx.clone(),
            event_tx: self.event_tx.clone(),
            cancel: self.cancel.clone(),
            is_running: self.is_running.clone(),
            pending_follow_ups: self.pending_follow_ups.clone(),
            shutdown_reason: self.shutdown_reason.clone(),
            shutdown_signaled: self.shutdown_signaled.clone(),
            agent_id: self.agent_id.clone(),
            manager: self.manager.clone(),
        }
    }

    /// Consume builder, spawn actor task, return handle.
    ///
    /// The returned handle is identical to one from `pre_handle()` — same channels,
    /// same shared primitives.
    pub fn spawn(mut self) -> AgentHandle {
        let urgent_rx = self
            .urgent_rx
            .take()
            .expect("spawn() called twice on the same builder");
        let normal_rx = self
            .normal_rx
            .take()
            .expect("spawn() called twice on the same builder");

        // Build conversation from initial state
        let conversation = crate::conversation::Conversation {
            messages: self.initial_messages,
            previous_summary: self.previous_summary,
            ..Default::default()
        };

        // Build schema cache for tool argument validation
        let schema_cache: HashMap<String, Arc<jsonschema::Validator>> = self
            .tools
            .iter()
            .filter_map(|tool| {
                let schema = tool.parameters_schema();
                let validator = jsonschema::validator_for(&schema).ok()?;
                Some((tool.name().to_string(), Arc::new(validator)))
            })
            .collect();

        // Create the actor state
        let state = AgentState {
            config: self.config,
            conversation,
            tools: self.tools,
            transport: self.transport,
            event_tx: self.event_tx.clone(),
            server_tools: self.server_tools,
            schema_cache,
            cwd: self.cwd,
            file_access: Arc::new(ParkingMutex::new(crate::tool::FileAccessTracker::default())),
            interaction_tx: self.interaction_tx,
            approval_policy: self.approval_policy,
            transform_context: self.transform_context,
            steering_queue: Vec::new(),
            follow_up_queue: Vec::new(),
            pending_follow_ups: self.pending_follow_ups.clone(),
            is_running: self.is_running.clone(),
            cancel: self.cancel.clone(),
        };

        // Wrap the actor future in catch_unwind so we record the panic
        // reason from *inside* the actor task, before tokio drops the spawn.
        // This closes the race where a handle's send returns `Closed` (from
        // receivers dropping during unwind) before the supervisor would have
        // had a chance to write `shutdown_reason`. Handles call
        // `shutdown_signaled.notified()` to bound a small wait for that
        // write to land.
        let event_tx_for_supervisor = self.event_tx.clone();
        let shutdown_reason = self.shutdown_reason.clone();
        let shutdown_signaled = self.shutdown_signaled.clone();
        tokio::spawn(async move {
            let actor_future = AssertUnwindSafe(run_actor(state, urgent_rx, normal_rx));
            match actor_future.catch_unwind().await {
                Ok(()) => {
                    // Clean shutdown — leave shutdown_reason as None.
                    // Still signal so any concurrent dead_actor_error wakes
                    // (it will fall through to the `Other` fallback).
                    shutdown_signaled.notify_waiters();
                }
                Err(payload) => {
                    let reason = payload
                        .downcast_ref::<&'static str>()
                        .map(|s| (*s).to_string())
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "<non-string panic payload>".to_string());
                    tracing::error!(reason = %reason, "agent actor task panicked");
                    *shutdown_reason.lock() = Some(reason.clone());
                    shutdown_signaled.notify_waiters();
                    // Best-effort: broadcast an Error event for any subscribed
                    // consumers. Ignored if there are no subscribers.
                    let _ = event_tx_for_supervisor.send(AgentEvent::Error {
                        message: format!("Actor panicked: {reason}"),
                    });
                }
            }
        });

        AgentHandle {
            urgent_tx: self.urgent_tx,
            normal_tx: self.normal_tx,
            event_tx: self.event_tx,
            cancel: self.cancel,
            is_running: self.is_running,
            pending_follow_ups: self.pending_follow_ups,
            shutdown_reason: self.shutdown_reason,
            shutdown_signaled: self.shutdown_signaled,
            agent_id: self.agent_id,
            manager: self.manager,
        }
    }
}
