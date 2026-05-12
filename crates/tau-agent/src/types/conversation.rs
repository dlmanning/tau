//! Conversation state.

use tau_ai::{Message, Usage};

/// The mutable conversational record. Owned exclusively by the actor
/// task via [`crate::core::state::Conv`].
#[derive(Default, Clone)]
pub struct Conversation {
    pub messages: Vec<Message>,
    pub is_streaming: bool,
    pub total_usage: Usage,
    pub error: Option<String>,
    pub previous_summary: Option<String>,
}
