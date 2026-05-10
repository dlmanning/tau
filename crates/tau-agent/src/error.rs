//! Error types for tau-agent

use thiserror::Error;

/// Result type alias using tau-agent Error
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during agent operations
#[derive(Error, Debug)]
pub enum Error {
    /// An error from the AI provider layer
    #[error(transparent)]
    Ai(#[from] tau_ai::Error),

    /// An error during compaction (string-based for flexibility)
    #[error("Compaction error: {0}")]
    Compaction(String),

    /// The agent is already processing a prompt
    #[error("Agent is busy processing another prompt")]
    Busy,

    /// The actor task panicked. The string carries the panic payload (or a
    /// best-effort description if it was not a string).
    #[error("Actor panicked: {0}")]
    ActorPanic(String),

    /// A non-blocking `try_X` send was rejected because the command channel
    /// was full. Distinct from `ActorPanic` / `Other` (channel closed):
    /// callers can retry a `ChannelFull` after a brief delay or fall back
    /// to the awaiting `X.await` variant.
    #[error("Command channel `{channel}` is full")]
    ChannelFull { channel: &'static str },

    /// The handle is not associated with an `AgentManager`, so spec-
    /// changing operations (`respec`, `with_system_prompt`, `with_tools`)
    /// can't dispatch. Builder-spawned root handles are unmanaged unless
    /// the host explicitly registers them with a manager.
    #[error("handle is not associated with an AgentManager; cannot respec")]
    Unmanaged,

    /// A generic agent error
    #[error("{0}")]
    Other(String),
}

impl Error {
    /// Check if this error indicates a context overflow
    pub fn is_context_overflow(&self) -> bool {
        match self {
            Error::Ai(e) => e.is_context_overflow(),
            _ => false,
        }
    }
}
