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
        tokens_before: u32,
        tokens_after: u32,
    },

    /// Error occurred
    Error { message: String },
}

impl AgentEvent {
    /// Check if this is a terminal event
    pub fn is_terminal(&self) -> bool {
        matches!(self, AgentEvent::AgentEnd { .. } | AgentEvent::Error { .. })
    }
}
