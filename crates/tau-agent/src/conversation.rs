//! Conversation state: messages, usage, streaming status, and compaction history.

use tau_ai::{Message, Usage};

/// Conversation state: messages, usage, streaming, and compaction history.
#[derive(Default)]
pub struct Conversation {
    /// Conversation messages
    pub messages: Vec<Message>,
    /// Whether currently streaming
    pub is_streaming: bool,
    /// Current streaming message (partial)
    pub stream_message: Option<Message>,
    /// Total usage across all turns
    pub total_usage: Usage,
    /// Last error
    pub error: Option<String>,
    /// Previous compaction summary (for iterative compaction)
    pub previous_summary: Option<String>,
}

/// Backward-compatible alias.
pub type AgentState = Conversation;
