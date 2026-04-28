use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::source::SourceId;

pub type DraftId = String;

/// A queued source-action awaiting user approval.
///
/// `tool_name` + `arguments` form a deferred tool call. When the user
/// calls `approve_draft`, the desk looks up the named tool in its
/// registry and dispatches it with `arguments` (bypassing the runtime's
/// `ApprovalPolicy` — the user already approved at the draft level).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Draft {
    pub id: DraftId,
    pub source_id: Option<SourceId>,
    pub tool_name: String,
    pub arguments: Value,
    pub rationale: Option<String>,
    pub status: DraftStatus,
    pub created_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub outcome: Option<ActionOutcome>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DraftStatus {
    Pending,
    Approved,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionOutcome {
    pub success: bool,
    pub summary: String,
    pub payload: Value,
    pub at: DateTime<Utc>,
}
