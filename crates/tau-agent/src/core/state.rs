//! Agent state, split into three structs by mutation discipline.
//!
//! The original `tau-agent` had a single `AgentState` god-struct
//! holding ~17 fields with three different ownership stories (immutable
//! wiring, mutable per-turn data, atomics shared with the handle). The
//! v2 split makes "what's actually mutable per turn?" answerable from
//! the type signature of every transition function:
//!
//! - [`Frame`] — agent wiring (tools, transport, policies, config).
//!   Borrowed `&Frame` by transition methods. Mutated *only* by the
//!   actor's command handler in response to `SetModel` /
//!   `SetReasoning` / `SetCompactionConfig` / `SetApprovalPolicy`;
//!   never by transition functions themselves.
//! - [`Conv`] — mutable per-turn state (conversation, queues, cwd).
//!   Borrowed `&mut Conv` by `apply_*` transitions.
//! - [`Shared`] — atomics shared with [`AgentHandle`]. The actor
//!   writes a few of these (e.g. `is_running`); tools read from them
//!   via [`ExecutionContext`].
//!
//! All three live on the actor task. `Shared` is just `Arc`s — it
//! holds no senders, so even though the state is owned by the actor,
//! the actor's exit condition (all senders dropped) remains reachable
//! once external handles are gone. (`AgentHandle` itself does hold the
//! `mpsc::Sender`s — that's the whole point of the handle. The
//! invariant is that the actor's `State` never does, so the actor
//! doesn't keep itself alive by side effect.)

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;
use tau_ai::Message;
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::core::approval::ApprovalPolicy;
use crate::core::config::AgentConfig;
use crate::core::interaction::InteractionRequest;
use crate::core::tool::{BoxedTool, FileAccessTracker};
use crate::core::transport::Transport;
use crate::types::conversation::Conversation;
use crate::types::events::AgentEvent;

/// Closure type used to transform the message history sent to the LLM
/// (e.g. to inject system reminders). Applied per-turn after building
/// context but before calling the transport.
pub type TransformContextFn = dyn Fn(Vec<Message>) -> Vec<Message> + Send + Sync;

/// A parsed tool invocation extracted from an assistant message.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

// ─── Frame: per-agent wiring ──────────────────────────────────────────

/// Per-agent wiring: tools, transport, schema cache, policies, the
/// event sender, and the agent's `AgentConfig`. Treated as read-only
/// by transition functions (their signatures take `&Frame`), with two
/// caveats:
///
/// 1. **`handle_busy_command` mutates `Frame` fields directly** while
///    servicing `SetModel` / `SetReasoning` / `SetCompactionConfig` /
///    `SetApprovalPolicy`. Changes take effect for the next turn.
/// 2. **Interior-mutable Arcs reachable from `Frame` may be mutated
///    elsewhere.** `file_access`'s `Arc<Mutex<...>>` is written by
///    tool tasks via `ExecutionContext`. That's not a mutation
///    through `&Frame` itself; the contract here is about *which
///    fields the actor reassigns*, not about every byte reachable
///    through the type.
pub struct Frame {
    pub config: AgentConfig,
    pub tools: Vec<BoxedTool>,
    pub server_tools: Vec<tau_ai::ServerTool>,
    /// Compiled validator + the original schema. The schema is kept so
    /// that validation errors can echo the expected shape back to the
    /// LLM, helping it self-correct on the next call.
    pub schema_cache: HashMap<String, (Arc<jsonschema::Validator>, Arc<serde_json::Value>)>,
    pub transport: Arc<dyn Transport>,
    pub event_tx: broadcast::Sender<AgentEvent>,
    pub interaction_tx: Option<mpsc::Sender<InteractionRequest>>,
    pub approval_policy: Arc<dyn ApprovalPolicy>,
    pub transform_context: Option<Arc<TransformContextFn>>,
    /// File-access tracker. Logically "shared mutable" but it's owned
    /// by the actor and reached by tools per-call via
    /// [`ExecutionContext::file_access`]; not part of the per-prompt
    /// `Conv` mutation discipline.
    pub file_access: Arc<Mutex<FileAccessTracker>>,
    /// Depth of this agent in the subagent spawn tree. `0` for the
    /// host's root agent; the manager's spawn paths increment by one
    /// for each descendant. Surfaced to tools via
    /// [`ExecutionContext::subagent_depth`].
    pub subagent_depth: u32,
}

// ─── Conv: mutable per-turn state ─────────────────────────────────────

/// Genuinely mutable per-turn state. Every `apply_*` transition takes
/// `&mut Conv`; pure decision functions take `&Conv`.
pub struct Conv {
    pub conversation: Conversation,
    pub steering_queue: Vec<Message>,
    pub follow_up_queue: Vec<Message>,
    pub cwd: Option<PathBuf>,
}

// ─── Shared: atomics shared with the handle ──────────────────────────

/// Atomics and shared cells visible to both the actor and any
/// `AgentHandle` clones. Cloning a [`Shared`] is just refcount bumps;
/// it never holds Senders, so cloning it (or storing it on the actor's
/// state) does not extend the actor's lifetime.
#[derive(Clone)]
pub struct Shared {
    pub is_running: Arc<AtomicBool>,
    pub pending_follow_ups: Arc<AtomicU32>,
    /// The actor swaps the inner token at each prompt start under this
    /// mutex; `handle.abort()` cancels whichever token is current.
    pub cancel: Arc<Mutex<CancellationToken>>,
    /// Stamped by the manager when the agent is registered (`spawn` /
    /// `adopt`). Surfaced to tools via
    /// [`ExecutionContext::agent_id`](crate::core::tool::ExecutionContext::agent_id).
    pub agent_id: Arc<OnceLock<String>>,
    /// Recorded by the actor's `catch_unwind` wrapper if the actor
    /// task panics. `None` while the actor is alive or after a clean
    /// shutdown.
    pub shutdown_reason: Arc<Mutex<Option<String>>>,
    /// Notified after `shutdown_reason` is written.
    pub shutdown_signaled: Arc<tokio::sync::Notify>,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            is_running: Arc::new(AtomicBool::new(false)),
            pending_follow_ups: Arc::new(AtomicU32::new(0)),
            cancel: Arc::new(Mutex::new(CancellationToken::new())),
            agent_id: Arc::new(OnceLock::new()),
            shutdown_reason: Arc::new(Mutex::new(None)),
            shutdown_signaled: Arc::new(tokio::sync::Notify::new()),
        }
    }
}

// ─── State: the actor's owned bundle ─────────────────────────────────

/// Bundle owned exclusively by the actor task. Three structs, three
/// mutation stories. Transition functions take borrows into this
/// bundle so the type system enforces what's mutable per call.
pub struct State {
    pub frame: Frame,
    pub conv: Conv,
    pub shared: Shared,
}
