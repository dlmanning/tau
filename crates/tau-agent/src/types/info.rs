//! Snapshot-query result types.
//!
//! These are data shapes returned by `AgentHandle` query methods
//! (`context_stats()`, `list_tools()`, etc.). Kept separate from
//! `events.rs` because they are *query results*, not event-stream
//! payloads.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::core::tool::ToolCategory;

/// Snapshot of the agent's context-window usage at the moment the
/// query is processed by the actor.
///
/// `used` is a cheap char/4 heuristic over the current conversation
/// (the same estimate driving overflow detection). `limit` is the
/// model's advertised context window. `remaining` is `limit - used`
/// saturated at zero.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ContextStats {
    pub used: u64,
    pub remaining: u64,
    pub limit: u64,
    pub updated_at: DateTime<Utc>,
}

/// Snapshot of a single tool registered on an agent at query time.
///
/// `currently_allowed` reflects the *active* approval policy: it is
/// `true` only when the policy classifies a no-argument invocation as
/// [`crate::core::approval::ApprovalDecision::Auto`] — i.e. the tool
/// would dispatch without prompting the user. `Gate` and `Reject(_)`
/// both map to `false`.
///
/// `default_allowed` is policy-independent and derives purely from the
/// tool's own [`crate::core::approval::ToolRisk`] under the built-in
/// `DefaultPolicy`: `Safe` and `Local` are allowed by default,
/// `Elevated` is not.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub category: ToolCategory,
    pub default_allowed: bool,
    pub currently_allowed: bool,
}
