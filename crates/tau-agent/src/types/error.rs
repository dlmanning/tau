//! Error types for the agent runtime.

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
