use tau_agent::AgentEvent;

use crate::activity::ActivityEntry;
use crate::brief::Brief;
use crate::card::{CardData, CardId, CardPile};
use crate::draft::{ActionOutcome, Draft, DraftId};
use crate::source::SourceId;
use crate::tombstone::DismissalRecord;

/// Broadcast events emitted by `DeskAgent`. Hosts subscribe via
/// `desk.subscribe()` to drive UI re-renders.
#[derive(Debug, Clone)]
pub enum DeskEvent {
    BriefUpdated {
        brief: Brief,
    },
    CardUpserted {
        card: CardData,
    },
    CardMoved {
        id: CardId,
        from: CardPile,
        to: CardPile,
    },
    CardRetired {
        id: CardId,
        reason: Option<String>,
    },
    CardTakeUpdated {
        id: CardId,
    },
    CardAttachmentAdded {
        id: CardId,
        kind: String,
    },
    CardPinned {
        id: CardId,
        pinned: bool,
    },
    CardDismissed {
        record: DismissalRecord,
    },
    CardUndismissed {
        external_ref: String,
    },

    DraftCreated {
        draft: Draft,
    },
    DraftApproved {
        draft_id: DraftId,
        outcome: ActionOutcome,
    },
    DraftRejected {
        draft_id: DraftId,
        reason: Option<String>,
    },

    ActivityAppended {
        entry: ActivityEntry,
    },

    SuggestionMuted {
        seed_from: String,
    },
    SuggestionUnmuted {
        seed_from: String,
    },

    ScanStarted {
        task: String,
    },
    ScanCompleted {
        task: String,
    },
    ScanFailed {
        task: String,
        message: String,
    },

    SourceError {
        source_id: SourceId,
        message: String,
    },

    /// Raw event from one of the desk's underlying agents (chat or per-task).
    /// Forwarded for hosts that want to render live agent reasoning (debug
    /// overlays, transcript views). Most UIs ignore this and rely on
    /// `ActivityAppended` for user-facing updates.
    AgentEvent {
        agent_id: String,
        event: AgentEvent,
    },
}
