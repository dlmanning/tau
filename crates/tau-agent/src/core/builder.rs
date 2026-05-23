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
use std::time::Duration;

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

/// History to load into an agent at construction.
///
/// Used by both [`AgentBuilder::seed`] and
/// [`SpawnOpts::seed`](crate::fleet::manager::SpawnOpts::seed). The
/// `Inherit` variant is only meaningful inside `SpawnOpts` — the fleet
/// resolves it against the registry at spawn time. Setting `Inherit`
/// on [`AgentBuilder`] is a no-op (the builder has no registry to
/// look up against) and emits a `tracing::warn!`.
#[derive(Debug, Clone, Default)]
pub enum AgentSeed {
    /// Start fresh — no messages, no prior summary.
    #[default]
    Empty,
    /// Restore from explicit history, optionally with a compaction tail.
    /// `previous_summary` seeds [`Conversation::previous_summary`](crate::types::conversation::Conversation::previous_summary)
    /// so the next compaction can thread continuity.
    Messages {
        messages: Vec<Message>,
        previous_summary: Option<String>,
    },
    /// Clone another tracked agent's current message history at spawn
    /// time. The named id must be in
    /// [`AgentManager`](crate::fleet::AgentManager)'s registry (idle,
    /// running, or adopted) when the fleet resolves this variant.
    Inherit { agent_id: String },
}

pub struct AgentBuilder {
    config: AgentConfig,
    transport: Arc<dyn Transport>,
    tools: Vec<BoxedTool>,
    server_tools: Vec<ServerTool>,
    interaction_tx: Option<mpsc::Sender<InteractionRequest>>,
    interaction_timeout: Option<Duration>,
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

    /// Test-only: when set, the spawned actor task panics before
    /// signalling readiness, exercising `spawn() -> Err(ActorPanic)`.
    #[cfg(any(test, feature = "test-utils"))]
    panic_at_startup: bool,
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
            interaction_timeout: None,
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
            #[cfg(any(test, feature = "test-utils"))]
            panic_at_startup: false,
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

    /// Set a deadline on every `InteractionRequest` the runtime emits
    /// (today, that's the `tool.confirm` approval gate). If the host
    /// hasn't responded within `timeout`, the actor synthesizes
    /// [`InteractionResponse::Rejected`](crate::core::interaction::InteractionResponse::Rejected)
    /// with an `"interaction timed out"` reason, rejects the tool call,
    /// and continues the prompt — closing the "host never replies"
    /// hang.
    ///
    /// The same value is exposed to tools via
    /// [`ExecutionContext::interaction_timeout`](crate::core::tool::ExecutionContext::interaction_timeout)
    /// so tool-initiated interactions can apply the same deadline.
    /// Default (no call) is unbounded wait — preserves historical
    /// behavior.
    pub fn set_interaction_timeout(&mut self, timeout: Duration) -> &mut Self {
        self.interaction_timeout = Some(timeout);
        self
    }

    /// Clear any previously-set interaction timeout (revert to
    /// unbounded waits).
    pub fn clear_interaction_timeout(&mut self) -> &mut Self {
        self.interaction_timeout = None;
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

    /// Seed the agent's conversation history. See [`AgentSeed`] for the
    /// variants. Replaces any prior seed; pass [`AgentSeed::Empty`] to
    /// clear back to a fresh agent.
    pub fn seed(&mut self, seed: AgentSeed) -> &mut Self {
        match seed {
            AgentSeed::Empty => {
                self.initial_messages = vec![];
                self.previous_summary = None;
            }
            AgentSeed::Messages {
                messages,
                previous_summary,
            } => {
                self.initial_messages = messages;
                self.previous_summary = previous_summary;
            }
            AgentSeed::Inherit { agent_id } => {
                tracing::warn!(
                    agent_id = %agent_id,
                    "AgentSeed::Inherit ignored on AgentBuilder (no registry to look up); use SpawnOpts::seed via AgentManager instead"
                );
                self.initial_messages = vec![];
                self.previous_summary = None;
            }
        }
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

    /// Subscribe to the agent's event stream before [`Self::spawn`]
    /// starts the actor. Mirrors [`broadcast::Sender::subscribe`].
    ///
    /// Subscribing before `spawn` is the only way to guarantee
    /// receipt of `AgentStart` — receivers created after the actor
    /// task is running may miss it depending on scheduling.
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.event_tx.subscribe()
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

    /// Test-only: configure the actor to panic at startup before
    /// signalling readiness. Used by `spawn().await -> Err(ActorPanic)`
    /// tests. The panic is injected by wrapping the actor future in
    /// `spawn()` itself; the actor task and `Frame` see no test hook.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn set_panic_at_startup(&mut self, panic: bool) -> &mut Self {
        self.panic_at_startup = panic;
        self
    }

    /// Consume the builder, spawn the actor task, and wait for the
    /// actor to signal readiness before returning the handle.
    ///
    /// Returns `Err(Error::ActorPanic)` if the actor task panicked
    /// before signalling — i.e. before reaching `Idle` for the first
    /// time. The actor itself currently has no fallible async setup,
    /// but the contract leaves room for future startup checks
    /// (transport pre-warm, tool init) to surface their errors here
    /// rather than via a side-channel.
    ///
    /// The returned handle is interchangeable with one from
    /// [`Self::handle`] — same channels, same shared atomics.
    pub async fn spawn(mut self) -> crate::types::error::Result<AgentHandle> {
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
            interaction_timeout: self.interaction_timeout,
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

        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();

        // Wrap the actor future in `catch_unwind` so we record the
        // panic reason from *inside* the actor task before tokio drops
        // the spawn. Closes the race where a handle's send returns
        // `Closed` (from receivers dropping during unwind) before any
        // supervisor would have written `panic_reason`.
        let event_tx_for_supervisor = self.event_tx.clone();
        let panic_reason = self.shared.panic_reason.clone();
        let shutdown_signaled = self.shared.shutdown_signaled.clone();
        let is_terminated = self.shared.is_terminated.clone();
        #[cfg(any(test, feature = "test-utils"))]
        let panic_at_startup = self.panic_at_startup;
        tokio::spawn(async move {
            // Both the test-fixture panic and the actor's runtime
            // panic must be inside the same `catch_unwind` so the
            // supervisor records `panic_reason` and `spawn().await`
            // surfaces `Err(ActorPanic)` either way.
            let task = AssertUnwindSafe(async move {
                #[cfg(any(test, feature = "test-utils"))]
                if panic_at_startup {
                    panic!("tau-agent test fixture: panic_at_startup");
                }
                crate::core::actor::run_actor(state, urgent_rx, normal_rx, ready_tx).await;
            });
            match task.catch_unwind().await {
                Ok(()) => {
                    // Clean shutdown — leave panic_reason as None.
                }
                Err(payload) => {
                    let reason = payload
                        .downcast_ref::<&'static str>()
                        .map(|s| (*s).to_string())
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "<non-string panic payload>".to_string());
                    tracing::error!(reason = %reason, "agent actor task panicked");
                    *panic_reason.lock() = Some(reason.clone());
                    let _ = event_tx_for_supervisor.send(AgentEvent::Error {
                        message: format!("Actor panicked: {reason}"),
                    });
                }
            }
            // Always: mark terminated and notify waiters, on both
            // clean exit and panic. `spawn().await` listens on this
            // notification to disambiguate "RecvError from dropped
            // ready_tx" vs "still waiting for setup."
            is_terminated.store(true, std::sync::atomic::Ordering::Release);
            shutdown_signaled.notify_waiters();
        });

        // Wait for the actor to signal readiness. If the actor panics
        // before signalling, `ready_tx` is dropped and `ready_rx`
        // returns `RecvError`. Wait briefly for the supervisor to
        // record the panic payload so we can return a meaningful
        // `Error::ActorPanic` rather than a generic "channel closed".
        match ready_rx.await {
            Ok(Ok(())) => Ok(AgentHandle {
                urgent_tx: self.urgent_tx,
                normal_tx: self.normal_tx,
                event_tx: self.event_tx,
                shared: self.shared,
            }),
            Ok(Err(setup_err)) => Err(setup_err),
            Err(_recv_err) => {
                // `ready_tx` was dropped without a signal — the actor
                // panicked during startup. The supervisor records
                // `panic_reason` and *then* notifies; on a multi-threaded
                // runtime that write can land after we observe RecvError,
                // so register for the notification before reading (mirrors
                // `AgentHandle::dead_actor_error`) and wait briefly so we
                // surface the real reason instead of the generic fallback.
                let notified = self.shared.shutdown_signaled.notified();
                if self.shared.panic_reason.lock().is_none() {
                    let _ = tokio::time::timeout(Duration::from_millis(100), notified).await;
                }
                let reason = self
                    .shared
                    .panic_reason
                    .lock()
                    .clone()
                    .unwrap_or_else(|| "actor died during startup".to_string());
                Err(crate::types::error::Error::ActorPanic(reason))
            }
        }
    }
}
