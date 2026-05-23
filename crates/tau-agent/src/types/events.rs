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
#[non_exhaustive]
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

/// Events emitted on a single agent's broadcast channel.
///
/// Only this agent's own activity appears here — no subagent events.
/// For fleet-level events (which agent started, which one finished,
/// forwarded child events) subscribe to
/// [`AgentManager`](crate::fleet::AgentManager)'s
/// [`FleetEvent`] channel.
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
        /// `true` when the prompt ended because `handle.interrupt()`
        /// was observed at the top of a turn (graceful stop). `false`
        /// for normal completion, errors, or `handle.abort()`.
        #[serde(default)]
        interrupted: bool,
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

    /// Tool-emitted self-label. The fleet bus translates this into
    /// [`FleetEvent::AgentReport`] when forwarding, stamping the
    /// emitting agent's id; consumers subscribed directly to a single
    /// agent's channel see it here without an id.
    AgentReport {
        #[serde(skip_serializing_if = "Option::is_none")]
        tag: Option<String>,
        summary: String,
    },
}

/// Events emitted on [`AgentManager`](crate::fleet::AgentManager)'s
/// broadcast channel.
///
/// Three kinds:
///
/// - **Lifecycle** (`AgentStarted` / `AgentResumed` / `AgentCompleted`)
///   — emitted by the manager itself when an agent crosses a
///   lifecycle boundary.
/// - **Self-reports** (`AgentReport`) — translated from
///   [`AgentEvent::AgentReport`] when the fleet bus forwards a child's
///   event. The originating agent's id is stamped on the variant.
/// - **Forwarded** (`Forwarded`) — every other [`AgentEvent`] a tracked
///   agent emits, stamped with `agent_id` and `description`.
///
/// Nesting is structurally impossible: `Forwarded::event` is an
/// [`AgentEvent`], not a `FleetEvent`. A grandchild's events arrive on
/// the same manager channel with the grandchild's `agent_id` — they do
/// not re-wrap.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FleetEvent {
    /// Parent dispatched an agent spawn.
    AgentStarted {
        agent_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        spec_name: Option<String>,
        description: String,
        prompt: String,
        started_at: DateTime<Utc>,
    },
    /// Previously-paused agent reactivated.
    AgentResumed {
        agent_id: String,
        description: String,
        prompt: String,
        resumed_at: DateTime<Utc>,
    },
    /// Agent terminated (success, abort, or failure).
    AgentCompleted {
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
    /// Tool inside the named agent emitted a self-report.
    AgentReport {
        agent_id: String,
        description: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        tag: Option<String>,
        summary: String,
    },
    /// Forwarded child event, stamped with the originating agent's id.
    Forwarded {
        agent_id: String,
        description: String,
        event: AgentEvent,
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
    /// Whether this event terminates the agent's own timeline.
    pub fn is_terminal(&self) -> bool {
        matches!(self, AgentEvent::AgentEnd { .. } | AgentEvent::Error { .. })
    }
}
