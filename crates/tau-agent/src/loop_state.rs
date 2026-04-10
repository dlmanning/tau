//! State machine for the agent loop.
//!
//! Each variant of [`LoopState`] represents a distinct phase.
//! [`Agent::step`] advances from one state to the next.

use tau_ai::Message;

/// A single tool call extracted from the model's response.
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

/// State machine driving the agent loop.
pub enum LoopState {
    /// Start of a new turn: check cancellation and turn limits.
    StartTurn {
        messages: Vec<Message>,
        first_user_message: Option<Message>,
    },

    /// Call the model and process the response stream.
    CallModel {
        messages: Vec<Message>,
        first_user_message: Option<Message>,
    },

    /// Execute tool calls returned by the model.
    ExecuteTools {
        tool_calls: Vec<ToolCall>,
        first_user_message: Option<Message>,
    },

    /// Check follow-up queue after a turn with no tool calls.
    DrainFollowUps,

    /// Loop is finished.
    Done(crate::error::Result<()>),
}
