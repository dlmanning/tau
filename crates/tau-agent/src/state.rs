//! Core agent state types.
//!
//! `AgentState` is the single mutable state object owned exclusively by the
//! actor task. `ToolCall` is a parsed tool invocation from a model response.
//! Both are used by `actor.rs` (I/O loop) and `logic.rs` (decision methods).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32};

use parking_lot::Mutex;
use tau_ai::Message;
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::approval::ApprovalPolicy;
use crate::builder::TransformContextFn;
use crate::config::AgentConfig;
use crate::conversation::Conversation;
use crate::events::AgentEvent;
use crate::tool::{BoxedTool, FileAccessTracker};
use crate::transport::Transport;

/// A single tool call extracted from the model's response.
#[derive(Debug)]
pub(crate) struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

/// A conversation mutation queued for application after the current prompt
/// finishes. Issued mid-prompt, these would otherwise wipe the assistant
/// message that contained the in-flight `tool_use` blocks, leaving orphan
/// `tool_result` messages that providers reject.
#[derive(Debug)]
pub(crate) enum ConversationOp {
    Clear,
    Set(Vec<Message>),
    SetPreviousSummary(Option<String>),
}

/// All mutable state the agent needs. Owned exclusively by the actor task.
pub(crate) struct AgentState {
    pub config: AgentConfig,
    pub conversation: Conversation,
    pub tools: Vec<BoxedTool>,
    pub transport: Arc<dyn Transport>,
    pub event_tx: broadcast::Sender<AgentEvent>,
    pub server_tools: Vec<tau_ai::ServerTool>,
    pub schema_cache: HashMap<String, Arc<jsonschema::Validator>>,
    pub cwd: Option<PathBuf>,
    pub file_access: Arc<Mutex<FileAccessTracker>>,
    pub interaction_tx: Option<mpsc::Sender<crate::interaction::InteractionRequest>>,
    pub approval_policy: Arc<dyn ApprovalPolicy>,
    pub transform_context: Option<Arc<TransformContextFn>>,
    pub steering_queue: Vec<Message>,
    pub follow_up_queue: Vec<Message>,
    /// Conversation mutations issued mid-prompt that must wait until the
    /// current prompt finishes (drained in the actor's `Done` arm). Idle-time
    /// mutations bypass this queue and apply immediately.
    pub pending_conversation_ops: Vec<ConversationOp>,
    /// Shared with AgentHandle. External code (AgentManager) increments via handle.expect_follow_up().
    pub pending_follow_ups: Arc<AtomicU32>,
    /// Shared with AgentHandle.
    pub is_running: Arc<AtomicBool>,
    /// Shared with all AgentHandle clones. The actor replaces the inner token at
    /// each prompt start so handle.abort() always targets the active prompt.
    pub cancel: Arc<Mutex<CancellationToken>>,
}
