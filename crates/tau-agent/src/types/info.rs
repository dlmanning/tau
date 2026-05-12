//! Snapshot types returned by `AgentHandle` queries.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Snapshot of the agent's context-window usage at the moment the
/// query is processed by the actor.
///
/// `used` is a cheap char/4 heuristic over the current conversation
/// (the same estimate driving overflow detection). `limit` is the
/// model's advertised context window. `remaining` is `limit - used`
/// saturated at zero.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextStats {
    pub used: u64,
    pub remaining: u64,
    pub limit: u64,
    pub updated_at: DateTime<Utc>,
}
