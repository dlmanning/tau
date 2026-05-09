use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::activity::SessionSeed;

/// The "Now" UI zone — projected at read time from `tau-session` state
/// and recent activity. Not stored. Not a pile.
///
/// PickUp = the most recently hibernated coding session, derived from
/// `SessionManager::list().filter(Hibernated).max_by(last_activity)`.
///
/// Suggestions = recent `ActivityEntry`s whose `suggest_session.is_some()`,
/// deduped by `seed_from`, filtered by `SuggestionMutes`, capped at N.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NowZone {
    pub pickup: Option<PickUpView>,
    pub suggestions: Vec<SuggestionView>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PickUpView {
    pub session_id: String,
    pub title: String,
    pub project: Option<PathBuf>,
    pub branch: Option<String>,
    pub paused_at: DateTime<Utc>,
    /// Derived from gap #4 file-change overlay. `None` if no diff.
    pub diff_summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestionView {
    pub activity_id: String,
    pub seed: SessionSeed,
    pub at: DateTime<Utc>,
}
