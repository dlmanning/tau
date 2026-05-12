//! Broadcast events and their auxiliary payloads.
//!
//! Events flow on a `tokio::sync::broadcast` channel. Subscribers are
//! the host UI, the fleet's event-forwarding bus (which wraps child
//! events as [`AgentEvent::Subagent`]), and any test collectors.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tau_ai::{Message, Usage};

/// Visual classification for a streamed console line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleLevel {
    Muted,
    Normal,
    Warning,
    Success,
    Danger,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsoleLine {
    pub content: String,
    pub level: ConsoleLevel,
}

impl ConsoleLine {
    pub fn new(content: impl Into<String>, level: ConsoleLevel) -> Self {
        Self {
            content: content.into(),
            level,
        }
    }

    pub fn normal(content: impl Into<String>) -> Self {
        Self::new(content, ConsoleLevel::Normal)
    }
}

/// Why compaction ran. Reported on `CompactionStart`. Lives in `types`
/// because it's an event payload — the *policy* that decides when to
/// compact is in `core::compaction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionReason {
    Threshold,
    Overflow,
    Manual,
}

/// Outcome of a single tool's approval, emitted on `ToolApprovalResolved`.
/// Lives in `types` because it's an event payload — the *policy* that
/// produces the underlying decision is in `core::approval`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolApprovalOutcome {
    AutoApproved,
    Approved,
    Rejected { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    AgentStart,
    TurnStart {
        turn_number: u32,
    },

    MessageStart {
        message: Message,
    },
    MessageUpdate {
        message: Message,
    },
    MessageEnd {
        message: Message,
    },

    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
        activity: String,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        lines: Vec<ConsoleLine>,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: String,
        is_error: bool,
    },
    ToolApprovalResolved {
        tool_call_id: String,
        tool_name: String,
        outcome: ToolApprovalOutcome,
    },

    TurnEnd {
        turn_number: u32,
        message: Message,
        usage: Usage,
    },
    AgentEnd {
        total_turns: u32,
        total_usage: Usage,
    },

    CompactionStart {
        reason: CompactionReason,
    },
    CompactionEnd {
        tokens_before: u64,
        tokens_after: u64,
    },

    Error {
        message: String,
    },

    /// Event from a subagent, wrapped with identity by the fleet bus.
    Subagent {
        agent_id: String,
        description: String,
        event: Box<AgentEvent>,
    },

    /// File mutation reported by a tool. Hosts feed these into a diff
    /// overlay. `before = None` means new file (Add); `after = None`
    /// means removed (Delete). Binary files intentionally not reported.
    FileChanged {
        path: PathBuf,
        #[serde(skip_serializing_if = "Option::is_none")]
        before: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        after: Option<String>,
        tool_call_id: String,
    },

    /// Parent dispatched a subagent spawn (parent-timeline event).
    SubagentStarted {
        agent_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        spec_name: Option<String>,
        description: String,
        prompt: String,
        started_at: DateTime<Utc>,
    },
    /// Previously-paused subagent reactivated (parent-timeline event).
    SubagentResumed {
        agent_id: String,
        description: String,
        prompt: String,
        resumed_at: DateTime<Utc>,
    },
    /// Subagent terminated (parent-timeline event).
    SubagentCompleted {
        agent_id: String,
        description: String,
        outcome: SubagentOutcome,
        started_at: DateTime<Utc>,
        completed_at: DateTime<Utc>,
        duration_ms: u64,
        usage: Usage,
        tool_use_count: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        worktree_path: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        worktree_branch: Option<String>,
    },
    /// Subagent self-labels its outcome (child-timeline event).
    SubagentReport {
        #[serde(skip_serializing_if = "Option::is_none")]
        tag: Option<String>,
        summary: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SubagentOutcome {
    Completed,
    Aborted { reason: String },
    Failed { reason: String },
}

impl AgentEvent {
    /// Whether this event terminates the agent's own timeline. A
    /// `Subagent { event }` is never terminal for the parent even if
    /// the inner event is.
    pub fn is_terminal(&self) -> bool {
        matches!(self, AgentEvent::AgentEnd { .. } | AgentEvent::Error { .. })
    }
}
