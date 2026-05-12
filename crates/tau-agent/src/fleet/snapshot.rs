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
pub struct AgentSnapshot {
    pub agent_id: String,
    pub description: String,
    pub status: AgentStatus,
    pub usage: Usage,
    pub tool_use_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
}

/// All agents the manager currently tracks. Ordering is unspecified —
/// hosts should sort by `started_at` or `agent_id` if a stable order is
/// needed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetSnapshot {
    pub agents: Vec<AgentSnapshot>,
}
