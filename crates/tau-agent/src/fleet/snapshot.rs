//! `FleetSnapshot` / `AgentSnapshot` — synchronous point-in-time view
//! of every agent the manager knows about (running, idle, adopted).
//!
//! Snapshot values are accumulated by the fleet bus as child events
//! flow through: `TurnEnd` adds to `usage`, `ToolExecutionEnd`
//! increments `tool_use_count`. `started_at` is stamped on first
//! `commit_running` / `adopt`; `completed_at` on `finish_to_idle`.
//! See [`crate::fleet::registry`] for the mutation methods that keep
//! these fields current.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tau_ai::Usage;

use crate::fleet::manager::AgentStatus;

/// One agent's worth of snapshot data. Cheap to clone.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AgentSnapshot {
    pub agent_id: String,
    pub description: String,
    pub status: AgentStatus,
    pub usage: Usage,
    /// Cumulative count of `ToolExecutionEnd` events observed for this
    /// agent across its entire lifetime. Note this is computed via an
    /// independent path from [`crate::SubagentResult::tool_use_count`]:
    /// the snapshot counter is incremented as events flow through the
    /// fleet bus, while `SubagentResult.tool_use_count` is derived by
    /// scanning the final message log for `Content::ToolCall` blocks.
    /// They should usually agree, but can drift if e.g. a tool errors
    /// before emitting `ToolExecutionEnd`.
    pub tool_use_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<DateTime<Utc>>,
    /// Wall-clock timestamp of the most recent `finish_to_idle`.
    /// Refreshes on every resume → idle cycle, so always reflects the
    /// last completed turn rather than the agent's original creation.
    ///
    /// **Not set for error-terminated agents**: agents removed from
    /// the registry via [`crate::fleet::registry::Registry::drop_running`]
    /// (failed spawns, `remove_interactive`, etc.) are dropped entirely
    /// and do not appear in snapshots. Hosts that need to observe
    /// failed runs should consume `SubagentCompleted` events directly.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
}

/// All agents the manager currently tracks. Ordering is unspecified —
/// hosts should sort by `started_at` or `agent_id` if a stable order is
/// needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FleetSnapshot {
    pub agents: Vec<AgentSnapshot>,
}
