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
