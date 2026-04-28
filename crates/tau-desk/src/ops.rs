//! Shared mutation primitives used by both `DeskAgent` user-side
//! methods and agent-side tools. Keeps the read-modify-write +
//! history-append pattern in one place.

use chrono::Utc;

use crate::Result;
use crate::card::{CardData, CardId};
use crate::error::Error;
use crate::provenance::{CardEvent, CardEventKind, Provenance};
use crate::storage::DeskStorage;

/// Read a card, apply the closure, stamp `last_modified*`, append a
/// history event (ring-buffer trimmed to `history_cap`), and write
/// back. Returns the stored card.
pub(crate) async fn mutate_card_with_history<F>(
    storage: &dyn DeskStorage,
    id: &CardId,
    by: Provenance,
    reason: Option<String>,
    event_kind: CardEventKind,
    history_cap: usize,
    mutate: F,
) -> Result<CardData>
where
    F: FnOnce(&mut CardData),
{
    let mut card = storage
        .read_card(id)
        .await?
        .ok_or_else(|| Error::NotFound(id.clone()))?;

    mutate(&mut card);

    let now = Utc::now();
    card.last_modified = now;
    card.last_modified_by = by.clone();
    card.last_modified_reason = reason;

    card.history.push_back(CardEvent {
        at: now,
        by,
        kind: event_kind,
    });
    while card.history.len() > history_cap {
        card.history.pop_front();
    }

    storage.upsert_card(&card).await?;
    Ok(card)
}
