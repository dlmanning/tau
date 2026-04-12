//! Agent event types

use serde::{Deserialize, Serialize};
use tau_ai::{Message, Usage};

/// Events emitted during agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Agent started processing
    AgentStart,

    /// A new turn started
    TurnStart { turn_number: u32 },

    /// Message streaming started
    MessageStart { message: Message },

    /// Message content updated during streaming
    MessageUpdate { message: Message },

    /// Message completed
    MessageEnd { message: Message },

    /// Tool execution started
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
        /// Human-readable activity description (e.g. "Reading main.rs")
        activity: String,
    },

    /// Tool execution progress update (emitted by tools during execution)
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        content: String,
    },

    /// Tool execution completed
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: String,
        is_error: bool,
    },

    /// A turn completed
    TurnEnd {
        turn_number: u32,
        message: Message,
        usage: Usage,
    },

    /// Agent finished processing
    AgentEnd {
        total_turns: u32,
        total_usage: Usage,
    },

    /// Context compaction started
    CompactionStart {
        reason: crate::compaction::CompactionReason,
    },

    /// Context compaction completed
    CompactionEnd {
        tokens_before: u64,
        tokens_after: u64,
    },

    /// Error occurred
    Error { message: String },

    /// Event from a subagent, wrapped with identity.
    Subagent {
        agent_id: String,
        description: String,
        event: Box<AgentEvent>,
    },
}

impl AgentEvent {
    /// Check if this is a terminal event.
    /// A `Subagent` event is never terminal for the parent even if the inner event is.
    pub fn is_terminal(&self) -> bool {
        matches!(self, AgentEvent::AgentEnd { .. } | AgentEvent::Error { .. })
    }
}
