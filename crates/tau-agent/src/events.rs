//! Agent event types

use chrono::{DateTime, Utc};
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

    /// Approval gate decided how to handle a tool call. Emitted before the
    /// tool runs (or, for rejected calls, in lieu of running it).
    ToolApprovalResolved {
        tool_call_id: String,
        tool_name: String,
        outcome: crate::approval::ToolApprovalOutcome,
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

    /// A plan step started executing. Emitted by the `step_started` tool;
    /// hosts pair with `PlanStepCompleted` by `step_id` to compute duration
    /// and current-step state.
    PlanStepStarted {
        step_id: String,
        activity: Option<String>,
        started_at: DateTime<Utc>,
    },

    /// A plan step completed. The `summary` is whatever short note the
    /// model wants to surface in the running view.
    PlanStepCompleted {
        step_id: String,
        summary: Option<String>,
        completed_at: DateTime<Utc>,
    },

    /// All plan steps completed. The agent may still emit a final summary
    /// turn after this; the actual lifecycle terminator is `AgentEnd`. This
    /// event is a UI milestone, not a stop signal — hosts should keep
    /// listening for `AgentEnd` to know when the actor is idle.
    PlanCompleted {
        summary: String,
        completed_at: DateTime<Utc>,
    },
}

impl AgentEvent {
    /// Check if this is a terminal event.
    /// A `Subagent` event is never terminal for the parent even if the inner event is.
    pub fn is_terminal(&self) -> bool {
        matches!(self, AgentEvent::AgentEnd { .. } | AgentEvent::Error { .. })
    }
}
