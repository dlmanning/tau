//! AgentBuilder — setup phase, consumed by spawn().

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32};

use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::actor::run_actor;
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
}

impl AgentBuilder {
    pub fn new(config: AgentConfig, transport: Arc<dyn Transport>) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        let (urgent_tx, urgent_rx) = mpsc::channel(64);
        let (normal_tx, normal_rx) = mpsc::channel(32);
        let cancel = Arc::new(ParkingMutex::new(CancellationToken::new()));
        let is_running = Arc::new(AtomicBool::new(false));
        let pending_follow_ups = Arc::new(AtomicU32::new(0));

        Self {
            config,
            transport,
            tools: vec![],
            server_tools: vec![],
            interaction_tx: None,
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
            transform_context: self.transform_context,
            steering_queue: Vec::new(),
            follow_up_queue: Vec::new(),
            pending_follow_ups: self.pending_follow_ups.clone(),
            is_running: self.is_running.clone(),
            cancel: self.cancel.clone(),
        };

        // Spawn the actor task
        tokio::spawn(run_actor(state, urgent_rx, normal_rx));

        AgentHandle {
            urgent_tx: self.urgent_tx,
            normal_tx: self.normal_tx,
            event_tx: self.event_tx,
            cancel: self.cancel,
            is_running: self.is_running,
            pending_follow_ups: self.pending_follow_ups,
        }
    }
}
