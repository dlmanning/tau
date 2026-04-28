use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::draft::DraftId;
use crate::source::SourceId;

pub type ActivityId = String;

/// Chronological log of agent and user actions. Drives the right-side
/// activity feed in the UI; also feeds the Now-zone projection (entries
/// with `suggest_session.is_some()` become Suggestion chips).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityEntry {
    pub id: ActivityId,
    pub seq: u64,
    pub at: DateTime<Utc>,
    pub text: String,
    pub kind: Option<ActivityKind>,
    pub suggest_session: Option<SessionSeed>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ActivityKind {
    AgentMessage,
    ScanStarted {
        task: String,
    },
    ScanCompleted {
        task: String,
        mutations: u32,
    },
    DraftCreated {
        draft_id: DraftId,
    },
    DraftApproved {
        draft_id: DraftId,
    },
    DraftRejected {
        draft_id: DraftId,
    },
    /// Agent attempted `upsert_card` on a tombstoned `external_ref`. Used
    /// by the Now-zone projection to inform the user about a blocked
    /// re-emergence so they can `undismiss` if desired.
    TombstoneHit {
        external_ref: String,
        original_summary: String,
    },
    SourceError {
        source_id: SourceId,
        message: String,
    },
}

/// Carried by activity entries that propose a session handoff. The
/// Now-zone projection surfaces these as Suggestion chips.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSeed {
    pub title: String,
    pub project: Option<PathBuf>,
    pub branch: Option<String>,
    pub kickoff: String,
    /// Reference back to the source that prompted this suggestion
    /// ("jira:PLT-312", "github:org/repo:pr/4821"). Used by
    /// `mute_suggestion` and by Now-zone dedup.
    pub seed_from: Option<String>,
}

/// Read view over the activity log. Storage-backed; this trait is held
/// by `DeskAgent` and exposed via `desk.activity()`.
pub trait ActivityFeed: Send + Sync {
    fn recent(&self, limit: usize) -> Vec<ActivityEntry>;
    fn since(&self, seq: u64) -> Vec<ActivityEntry>;
    fn get(&self, id: &str) -> Option<ActivityEntry>;
}
