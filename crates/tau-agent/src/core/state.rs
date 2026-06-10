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
//! - [`Shared`] — atomics shared with [`AgentHandle`](crate::AgentHandle).
//!   The actor writes a few of these (e.g. `prompt_in_flight`); tools
//!   read from them via [`ExecutionContext`](crate::ExecutionContext).
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
use std::time::Duration;

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
    /// Maximum time the actor will wait on a pending
    /// [`InteractionRequest`] response before synthesizing
    /// [`InteractionResponse::Rejected`](crate::core::interaction::InteractionResponse::Rejected)
    /// and unblocking the tool batch. `None` (the default) means wait
    /// indefinitely — historical behavior, which lets a host that
    /// never replies hang the agent.
    ///
    /// Also surfaced to tools via
    /// [`ExecutionContext::interaction_timeout`](crate::core::tool::ExecutionContext::interaction_timeout)
    /// so tool-initiated interactions can self-apply the same deadline.
    pub interaction_timeout: Option<Duration>,
    pub approval_policy: Arc<dyn ApprovalPolicy>,
    pub transform_context: Option<Arc<TransformContextFn>>,
    /// File-access tracker. Logically "shared mutable" but it's owned
    /// by the actor and reached by tools per-call via
    /// [`ExecutionContext::file_access`](crate::ExecutionContext::file_access);
    /// not part of the per-prompt
    /// `Conv` mutation discipline.
    pub file_access: Arc<Mutex<FileAccessTracker>>,
    /// Depth of this agent in the subagent spawn tree. `0` for the
    /// host's root agent; the manager's spawn paths increment by one
    /// for each descendant. Surfaced to tools via
    /// [`ExecutionContext::subagent_depth`](crate::ExecutionContext::subagent_depth).
    pub subagent_depth: u32,
}

// ─── Conv: mutable per-turn state ─────────────────────────────────────

/// Genuinely mutable per-turn state. Every `apply_*` transition takes
/// `&mut Conv`; pure decision functions take `&Conv`.
///
/// Discipline: `Conv` is mutated **only** through
/// [`transitions`](crate::core::transitions) `apply_*` functions — the
/// actor never writes fields directly. This keeps every state mutation
/// nameable, testable, and greppable (`&mut Conv` outside
/// `transitions.rs` is a review flag).
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
    /// `true` while a prompt is in flight (between `AgentStart` and
    /// `AgentEnd`). Combined with `is_terminated` by
    /// [`AgentHandle::health`](crate::core::handle::AgentHandle::health)
    /// to compute `Running` / `Idle` / `Dead`.
    pub prompt_in_flight: Arc<AtomicBool>,
    /// `true` after the actor task has exited (clean shutdown OR
    /// panic). Set by the supervisor wrapper in
    /// [`AgentBuilder::spawn`](crate::core::builder::AgentBuilder::spawn).
    /// Once set it never clears.
    pub is_terminated: Arc<AtomicBool>,
    pub pending_follow_ups: Arc<AtomicU32>,
    /// The actor swaps the inner token at each prompt start under this
    /// mutex; `handle.abort()` cancels whichever token is current.
    pub cancel: Arc<Mutex<CancellationToken>>,
    /// Set by `handle.interrupt()` to request a graceful stop. Checked
    /// at the top of each new turn, after the current tool batch has
    /// completed and before the next LLM call. Distinct from `cancel`
    /// (which hard-cancels in-flight work). Reset to `false` at the
    /// start of every new prompt.
    pub interrupt_requested: Arc<AtomicBool>,
    /// Stamped by the manager when the agent is registered (`spawn` /
    /// `adopt`). Surfaced to tools via
    /// [`ExecutionContext::agent_id`](crate::core::tool::ExecutionContext::agent_id).
    pub agent_id: Arc<OnceLock<String>>,
    /// Recorded by the actor's `catch_unwind` wrapper if the actor
    /// task panics. `None` while the actor is alive or after a clean
    /// shutdown.
    pub panic_reason: Arc<Mutex<Option<String>>>,
    /// Notified after `is_terminated` is set and `panic_reason` is
    /// written. Used by `spawn()` to wait briefly for the supervisor
    /// to record a panic payload when the readiness oneshot drops.
    pub shutdown_signaled: Arc<tokio::sync::Notify>,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            prompt_in_flight: Arc::new(AtomicBool::new(false)),
            is_terminated: Arc::new(AtomicBool::new(false)),
            pending_follow_ups: Arc::new(AtomicU32::new(0)),
            cancel: Arc::new(Mutex::new(CancellationToken::new())),
            interrupt_requested: Arc::new(AtomicBool::new(false)),
            agent_id: Arc::new(OnceLock::new()),
            panic_reason: Arc::new(Mutex::new(None)),
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
