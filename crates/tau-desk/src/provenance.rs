use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::card::CardPile;
use crate::source::SourceId;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "by", rename_all = "snake_case")]
pub enum Provenance {
    User,
    Agent {
        /// Identifies *which* agent: `"chat"`, `"morning_scan"`,
        /// `"webhook:gh"`, etc. `None` is allowed for legacy/uncategorized
        /// agent activity but new code should always set it.
        agent_id: Option<String>,
    },
    Source {
        source_id: SourceId,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardEvent {
    pub at: DateTime<Utc>,
    pub by: Provenance,
    pub kind: CardEventKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum CardEventKind {
    Created,
    Updated,
    /// `agent_take` was patched. Distinct from `Updated` so the history
    /// can be filtered/coalesced for chatty take-revisers.
    TakeUpdated,
    Moved {
        from: CardPile,
        to: CardPile,
    },
    Retired {
        reason: Option<String>,
    },
    AttachmentAdded {
        kind: String,
    },
    Pinned,
    Unpinned,
}
