//! Conversation state.

use tau_ai::{Message, Usage};

/// The mutable conversational record. Owned exclusively by the actor
/// task via [`crate::core::state::Conv`].
///
/// "Is a prompt in flight?" is no longer represented here — query
/// [`AgentHandle::health`](crate::core::handle::AgentHandle::health)
/// instead, which distinguishes `Running` from `Idle`.
#[derive(Default, Clone)]
#[non_exhaustive]
pub struct Conversation {
    pub messages: Vec<Message>,
    pub total_usage: Usage,
    pub error: Option<String>,
    pub previous_summary: Option<String>,
}
