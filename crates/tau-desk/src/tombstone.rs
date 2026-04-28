use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Records that the user has dismissed an `external_ref` and that the
/// agent must not re-create cards for it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DismissalRecord {
    pub external_ref: String,
    pub dismissed_at: DateTime<Utc>,
    pub reason: Option<String>,
}
