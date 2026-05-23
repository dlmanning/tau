//! Error types for the agent runtime.
//!
//! `Error` separates conditions callers may want to branch on
//! (`AgentNotFound`, `AgentBusy`, `Busy`, `ChannelFull`, …) from the
//! catch-all `Other(String)` that wraps situations that don't yet
//! have a structured shape. Avoid adding new string-encoded conditions
//! — prefer introducing a variant.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    /// An error from the AI provider layer.
    #[error(transparent)]
    Ai(#[from] tau_ai::Error),

    /// Compaction failed (string-based — the underlying summarization
    /// LLM call may have failed for any of several reasons).
    #[error("Compaction error: {0}")]
    Compaction(String),

    /// Agent is already processing a prompt.
    #[error("Agent is busy processing another prompt")]
    Busy,

    /// The actor task panicked. The string carries the panic payload
    /// (or a best-effort description if it was not a string).
    #[error("Actor panicked: {0}")]
    ActorPanic(String),

    /// Non-blocking `try_*` send rejected because the command channel
    /// was full. Distinct from `ActorPanic` / `Other` (channel closed):
    /// callers can retry after a brief delay or use the awaiting variant.
    #[error("Command channel `{channel}` is full")]
    ChannelFull { channel: &'static str },

    /// A fleet operation referenced an agent id that isn't tracked
    /// (or isn't in the bucket the operation requires — `respec`
    /// needs an idle agent, `send` walks every bucket). The id may
    /// have been evicted under LRU pressure, never spawned, or
    /// already detached by a concurrent `respec`.
    #[error("agent '{id}' not found in the fleet registry")]
    AgentNotFound { id: String },

    /// A fleet operation that requires an idle agent encountered one
    /// that's currently executing a prompt. Recover by aborting or
    /// interrupting the agent and waiting for `AgentEnd` before
    /// retrying the operation.
    #[error("agent '{id}' is currently running; abort or interrupt before this operation")]
    AgentBusy { id: String },

    /// `respec` failed to spawn the agent under its new spec. The
    /// previous spec has been restored — the agent is still in the
    /// registry under its prior configuration, so the caller can
    /// continue using it. Inspect `source` for the underlying cause.
    #[error("respec for agent '{id}' rolled back to previous spec")]
    RespecRolledBack {
        id: String,
        #[source]
        source: Box<Error>,
    },

    /// The runtime failed to set up a per-agent git worktree before
    /// the agent could start. Typically a filesystem or git error.
    #[error("worktree setup failed: {reason}")]
    WorktreeSetupFailed { reason: String },

    /// Unstructured error. Reserved for situations that don't yet
    /// have a dedicated variant — channel-closed-after-actor-death,
    /// internal invariant violations, etc. New error conditions
    /// should grow their own variant rather than landing here.
    #[error("{0}")]
    Other(String),
}

impl Error {
    pub fn is_context_overflow(&self) -> bool {
        match self {
            Error::Ai(e) => e.is_context_overflow(),
            _ => false,
        }
    }
}
