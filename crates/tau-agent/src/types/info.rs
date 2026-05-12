//! Snapshot-query result types.
//!
//! These are data shapes returned by `AgentHandle` query methods
//! (`list_tools()`, future `context_stats()`, etc.). Kept separate from
//! `events.rs` because they are *query results*, not event-stream
//! payloads.

use serde::{Deserialize, Serialize};

use crate::core::tool::ToolCategory;

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
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub category: ToolCategory,
    pub default_allowed: bool,
    pub currently_allowed: bool,
}
