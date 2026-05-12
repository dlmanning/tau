//! `AgentBuilder` — configure and spawn an agent.
//!
//! Shared primitives (channels, atomics) are created eagerly so
//! [`Self::handle`] can return a working `AgentHandle` *before*
//! [`Self::spawn`] consumes the builder. Used by:
//!
//! - The fleet manager, which stamps the agent's id on the handle's
//!   shared cell before consuming the builder.
//! - Hosts that want to subscribe to events before the actor starts.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::Arc;

use futures::FutureExt;
use parking_lot::Mutex as ParkingMutex;
use tau_ai::{Message, ServerTool};
use tokio::sync::{broadcast, mpsc};

use crate::core::approval::{ApprovalPolicy, DefaultPolicy};
use crate::core::command::Command;
use crate::core::config::AgentConfig;
use crate::core::handle::AgentHandle;
use crate::core::interaction::InteractionRequest;
use crate::core::state::{Conv, Frame, Shared, State, TransformContextFn};
use crate::core::tool::{BoxedTool, FileAccessTracker};
use crate::core::transport::Transport;
use crate::types::conversation::Conversation;
use crate::types::events::AgentEvent;

/// Default capacity of the urgent (Steer / FollowUp) channel. Sized
/// for fan-in from many background subagents and bursty user steering.
pub const DEFAULT_URGENT_CAPACITY: usize = 256;

/// Default capacity of the normal channel (config setters, queries,
/// prompt requests). Smaller because volume is naturally bounded.
pub const DEFAULT_NORMAL_CAPACITY: usize = 64;

pub struct AgentBuilder {
    config: AgentConfig,
    transport: Arc<dyn Transport>,
    tools: Vec<BoxedTool>,
    server_tools: Vec<ServerTool>,
    interaction_tx: Option<mpsc::Sender<InteractionRequest>>,
    approval_policy: Arc<dyn ApprovalPolicy>,
    cwd: Option<PathBuf>,
    transform_context: Option<Arc<TransformContextFn>>,
    initial_messages: Vec<Message>,
    previous_summary: Option<String>,
    subagent_depth: u32,

    // Pre-created shared primitives.
    event_tx: broadcast::Sender<AgentEvent>,
    urgent_tx: mpsc::Sender<Command>,
    urgent_rx: Option<mpsc::Receiver<Command>>,
    normal_tx: mpsc::Sender<Command>,
    normal_rx: Option<mpsc::Receiver<Command>>,
    shared: Shared,
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

    pub fn with_channel_capacities(
        config: AgentConfig,
        transport: Arc<dyn Transport>,
        urgent_capacity: usize,
        normal_capacity: usize,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        let (urgent_tx, urgent_rx) = mpsc::channel(urgent_capacity);
        let (normal_tx, normal_rx) = mpsc::channel(normal_capacity);
        let shared = Shared::default();

        Self {
            config,
            transport,
            tools: vec![],
            server_tools: vec![],
            interaction_tx: None,
            approval_policy: Arc::new(DefaultPolicy),
            cwd: None,
            transform_context: None,
            initial_messages: vec![],
            previous_summary: None,
            subagent_depth: 0,
            event_tx,
            urgent_tx,
            urgent_rx: Some(urgent_rx),
            normal_tx,
            normal_rx: Some(normal_rx),
            shared,
        }
    }

    // ─── Configuration ───────────────────────────────────────────────

    pub fn add_tool(&mut self, tool: BoxedTool) -> &mut Self {
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

    pub fn set_interaction_sender(&mut self, tx: mpsc::Sender<InteractionRequest>) -> &mut Self {
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

    /// Set this agent's depth in the subagent spawn tree. `0` for the
    /// host's root; the fleet's spawn paths increment by one for each
    /// descendant. Tools read this back via
    /// [`ExecutionContext::subagent_depth`](crate::core::tool::ExecutionContext::subagent_depth).
    pub fn set_subagent_depth(&mut self, depth: u32) -> &mut Self {
        self.subagent_depth = depth;
        self
    }

    pub fn set_transform_context(&mut self, f: Arc<TransformContextFn>) -> &mut Self {
        self.transform_context = Some(f);
        self
    }

    // ─── Read access (for fleet setup) ───────────────────────────────

    pub fn config(&self) -> &AgentConfig {
        &self.config
    }
    pub fn tools(&self) -> &[BoxedTool] {
        &self.tools
    }
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }
    pub fn event_sender(&self) -> broadcast::Sender<AgentEvent> {
        self.event_tx.clone()
    }

    /// Get a working `AgentHandle` before calling [`Self::spawn`].
    ///
    /// The returned handle shares all primitives with the eventual
    /// post-spawn handle, including the shared `agent_id` cell that
    /// `State.shared.agent_id` reads to populate
    /// [`ExecutionContext::agent_id`](crate::core::tool::ExecutionContext::agent_id).
    /// Used by:
    ///
    /// - The fleet manager (to stamp `agent_id` before consuming the
    ///   builder).
    /// - Hosts that want to subscribe to events before the actor task
    ///   starts.
    ///
    /// Commands sent through this handle are queued and processed
    /// once `spawn()` starts the actor.
    pub fn handle(&self) -> AgentHandle {
        AgentHandle {
            urgent_tx: self.urgent_tx.clone(),
            normal_tx: self.normal_tx.clone(),
            event_tx: self.event_tx.clone(),
            shared: self.shared.clone(),
        }
    }

    /// Consume the builder, spawn the actor task, return the handle.
    ///
    /// The returned handle is interchangeable with one from
    /// [`Self::handle`] — same channels, same shared atomics.
    pub fn spawn(mut self) -> AgentHandle {
        let urgent_rx = self.urgent_rx.take().expect("spawn() consumes self");
        let normal_rx = self.normal_rx.take().expect("spawn() consumes self");

        let conversation = Conversation {
            messages: self.initial_messages,
            previous_summary: self.previous_summary,
            ..Default::default()
        };

        let schema_cache: HashMap<String, (Arc<jsonschema::Validator>, Arc<serde_json::Value>)> =
            self.tools
                .iter()
                .filter_map(|tool| {
                    let schema = tool.parameters_schema();
                    match jsonschema::validator_for(&schema) {
                        Ok(v) => Some((
                            tool.name().to_string(),
                            (Arc::new(v), Arc::new(schema)),
                        )),
                        Err(e) => {
                            tracing::warn!(
                                tool = tool.name(),
                                error = %e,
                                "Tool schema failed to compile; arguments will not be validated"
                            );
                            None
                        }
                    }
                })
                .collect();

        let frame = Frame {
            config: self.config,
            tools: self.tools,
            server_tools: self.server_tools,
            schema_cache,
            transport: self.transport,
            event_tx: self.event_tx.clone(),
            interaction_tx: self.interaction_tx,
            approval_policy: self.approval_policy,
            transform_context: self.transform_context,
            file_access: Arc::new(ParkingMutex::new(FileAccessTracker::default())),
            subagent_depth: self.subagent_depth,
        };
        let conv = Conv {
            conversation,
            steering_queue: Vec::new(),
            follow_up_queue: Vec::new(),
            cwd: self.cwd,
        };
        let state = State {
            frame,
            conv,
            shared: self.shared.clone(),
        };

        // Wrap the actor future in `catch_unwind` so we record the
        // panic reason from *inside* the actor task before tokio drops
        // the spawn. Closes the race where a handle's send returns
        // `Closed` (from receivers dropping during unwind) before any
        // supervisor would have written `shutdown_reason`.
        let event_tx_for_supervisor = self.event_tx.clone();
        let shutdown_reason = self.shared.shutdown_reason.clone();
        let shutdown_signaled = self.shared.shutdown_signaled.clone();
        tokio::spawn(async move {
            let actor_future =
                AssertUnwindSafe(crate::core::actor::run_actor(state, urgent_rx, normal_rx));
            match actor_future.catch_unwind().await {
                Ok(()) => {
                    // Clean shutdown — leave shutdown_reason as None.
                    // Still notify so any concurrent `dead_actor_error`
                    // wakes and falls through to the fallback.
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
            shared: self.shared,
        }
    }
}
