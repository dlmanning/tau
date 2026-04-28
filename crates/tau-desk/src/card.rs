use std::collections::VecDeque;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::draft::DraftId;
use crate::provenance::{CardEvent, Provenance};

pub type CardId = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CardPile {
    NeedsYou,
    Drafts,
    Watching,
    Done,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardData {
    pub id: CardId,

    pub pile: CardPile,

    /// Stable platform reference for dedup across runs and across slightly
    /// different agent-picked ids. Conventionally a URL or platform-native
    /// id ("https://jira/browse/PLT-312", "github:org/repo:pr/4821").
    pub external_ref: Option<String>,

    pub body: CardBody,

    /// Editorial commentary the agent attaches to the card. Updated
    /// independently of `body` via `update_take`. Agent-only.
    pub agent_take: Option<AgentTake>,

    /// Cross-source synthesis (`kind: "slack-thread"`, `"linked-pr"`,
    /// etc.) and user-authored prose (`kind: "user-note"`). Lives on
    /// `CardData` rather than per-body so any card type can carry
    /// attachments. Agent's `upsert_card` should preserve existing
    /// attachments unless deliberately replacing them.
    pub attachments: Vec<Attachment>,

    pub metadata: serde_json::Value,

    pub pinned: bool,

    pub created_at: DateTime<Utc>,
    pub last_modified: DateTime<Utc>,
    pub last_modified_by: Provenance,
    pub last_modified_reason: Option<String>,

    /// Bounded ring buffer (default cap 32). Older events fall off; hosts
    /// that need a full audit log subscribe to `DeskEvent` and persist it
    /// themselves.
    pub history: VecDeque<CardEvent>,
}

/// Agent-authored prose attached to a card. Two slots:
/// - `ask`: imperative — "why this is on your plate" ("review — she's
///   blocked on a deploy window").
/// - `note`: editorial — "what the agent thinks" ("I skimmed it; one
///   risky change in refund_test.rs worth flagging").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTake {
    pub ask: Option<String>,
    pub note: Option<String>,
    pub updated_at: DateTime<Utc>,
}

/// A reference attached to a card. Two flavors share this shape:
///
/// - **Cross-source synthesis** (agent): `kind: "slack-thread"`,
///   `"linked-pr"`, `"ci-run"` etc. Carries `url` + a one-line `summary`.
/// - **User notes** (user): `kind: "user-note"`. `url: None`,
///   `summary` carries the note body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    pub kind: String,
    pub url: Option<String>,
    pub summary: String,
}

/// PickUp and Suggestion are deliberately absent — both are render-time
/// projections (see `now::NowZone`), not stored card kinds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CardBody {
    Pr {
        url: String,
        title: String,
        repo: String,
        author: String,
        ci: Option<String>,
    },
    Jira {
        url: String,
        title: String,
        project: String,
        status: Option<String>,
    },
    Thread {
        platform: String,
        url: Option<String>,
        snippet: String,
    },
    Watch {
        title: String,
        description: String,
        status: Option<String>,
    },
    Note {
        body: String,
    },
    Draft {
        draft_id: DraftId,
        summary: String,
    },
    Other {
        body_kind: String,
        body: serde_json::Value,
    },
}

impl CardBody {
    /// Source-backed bodies are agent-write-only via `upsert_card`. `Note`
    /// is user-write-only via the Note CRUD verbs. Used by the verb
    /// matrix to enforce `Error::WrongAuthor`.
    pub fn is_user_owned(&self) -> bool {
        matches!(self, CardBody::Note { .. })
    }
}
