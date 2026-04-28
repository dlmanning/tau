use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Singleton — the desk has one brief at a time. The agent regenerates
/// it via `update_brief` (typically as the last step of a morning scan).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Brief {
    pub greeting: String,
    pub summary: String,
    pub stats: Vec<BriefStat>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BriefStat {
    pub label: String,
    pub value: String,
    pub delta: Option<String>,
}
