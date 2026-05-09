//! Agent event types

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tau_ai::{Message, Usage};

/// Visual classification for a streamed console line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsoleLevel {
    /// Faded — blank lines, headers, "$ command" prompts.
    Muted,
    /// Default — informational output.
    Normal,
    /// Caution — `warning:`, `Running …`, soft errors.
    Warning,
    /// Positive — `test … ok`, `✓`, success markers.
    Success,
    /// Failure — `error:`, panics, `FAILED`.
    Danger,
}

/// One line of streamed tool output with a host-facing classification.
/// `content` may contain ANSI escapes; the level is the tool's call on what
/// the line means semantically.
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

    /// Tool execution progress update (emitted by tools during execution).
    /// Each `ConsoleLine` carries a classification the host can style
    /// without re-parsing the content. Tools that don't classify just
    /// emit `ConsoleLevel::Normal`.
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        lines: Vec<ConsoleLine>,
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

    /// A conversation mutation (clear / set messages / set summary) was
    /// issued mid-prompt. The runtime buffered it and will apply it after
    /// the current prompt finishes, before broadcasting `AgentEnd`. Hosts
    /// can surface "queued — applies when this turn ends" if they want.
    ///
    /// **Delivery is best-effort.** Buffered ops live in the actor's
    /// in-memory state and are dropped if the actor terminates (panic,
    /// cancellation, or all handles released) before reaching the next
    /// `Done` phase. Hosts that need the mutation to survive a crash
    /// should persist it themselves before issuing the command.
    ConversationOpDeferred { kind: DeferredOpKind },

    /// Event from a subagent, wrapped with identity.
    Subagent {
        agent_id: String,
        description: String,
        event: Box<AgentEvent>,
    },

    /// A single line of streamed tool output, classified for UI styling.
    /// ANSI escape codes are preserved in `content`; web hosts strip on
    /// display, terminal hosts pass through.
    ///
    /// A file-mutating tool reports a before/after snapshot. Hosts feed
    /// these into a diff overlay (e.g. `tau_tools::diff::SessionDiffOverlay`)
    /// to render the cumulative session diff. `before = None` means the file
    /// did not exist (Add); `after = None` means it was removed (Delete).
    /// Binary files are intentionally not reported.
    FileChanged {
        path: PathBuf,
        #[serde(skip_serializing_if = "Option::is_none")]
        before: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        after: Option<String>,
        tool_call_id: String,
    },

    /// The parent dispatched a subagent spawn. Emitted synchronously by the
    /// parent's manager *before the child actor task exists*; bracketed by
    /// `SubagentCompleted` with the same `agent_id`.
    ///
    /// This event lives on the **parent's timeline**: it records an action
    /// the parent took, analogous to how `ToolExecutionStart` records a
    /// parent-side tool dispatch. It carries metadata only the parent
    /// knows — `agent_type`, the request `prompt`, and the parent-side
    /// `started_at` — none of which the child can supply.
    ///
    /// The child's own intrinsic lifecycle (`AgentStart` → … →
    /// `AgentEnd`) arrives separately on the **child's timeline**,
    /// forwarded onto the parent's stream as
    /// `Subagent { agent_id, event: AgentStart }`, etc. The two are not
    /// duplicates — they describe different things. A host rendering the
    /// parent's actions watches for `SubagentStarted`; a host rendering
    /// child progress (tool calls, turns, completion) watches the
    /// forwarded child events. Both fire exactly once per fresh spawn,
    /// with `SubagentStarted` first.
    SubagentStarted {
        agent_id: String,
        agent_type: String,
        description: String,
        prompt: String,
        started_at: DateTime<Utc>,
    },

    /// A previously-paused subagent was reactivated for a follow-up turn
    /// (e.g. `agent({ to: "<id>", prompt: "..." })`). Bracketed by a fresh
    /// `SubagentCompleted` with the same `agent_id`.
    SubagentResumed {
        agent_id: String,
        description: String,
        prompt: String,
        resumed_at: DateTime<Utc>,
    },

    /// A subagent terminated (success or abort). Pairs with the most
    /// recent `SubagentStarted`/`SubagentResumed` for the same `agent_id`.
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

    /// A subagent self-labels its outcome before terminating. Emitted by
    /// the `subagent_report` tool on the subagent's own stream; reaches
    /// the parent host wrapped as `Subagent { event: SubagentReport { … } }`.
    /// Hosts correlate with the eventual `SubagentCompleted` by `agent_id`
    /// and use the `tag` for product-specific badges (e.g. `"passed"`,
    /// `"failed"`, `"approve"`, `"changes"`).
    SubagentReport {
        #[serde(skip_serializing_if = "Option::is_none")]
        tag: Option<String>,
        summary: String,
    },
}

/// Kind of deferred conversation mutation, surfaced to hosts via
/// [`AgentEvent::ConversationOpDeferred`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeferredOpKind {
    Clear,
    SetMessages,
    SetPreviousSummary,
}

/// How a subagent terminated.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SubagentOutcome {
    /// Reached `AgentEnd` cleanly (success or model-reported failure).
    Completed,
    /// Cancelled by the parent or its own cancellation token.
    Aborted { reason: String },
    /// Hit a non-cancellation error (transport, validation, etc.) before
    /// reaching `AgentEnd`. Hosts may render this differently from `Aborted`
    /// (which represents user/parent intent) since `Failed` indicates an
    /// infrastructure problem.
    Failed { reason: String },
}

impl AgentEvent {
    /// Check if this is a terminal event.
    /// A `Subagent` event is never terminal for the parent even if the inner event is.
    pub fn is_terminal(&self) -> bool {
        matches!(self, AgentEvent::AgentEnd { .. } | AgentEvent::Error { .. })
    }
}
