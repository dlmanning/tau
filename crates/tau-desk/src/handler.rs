//! Mechanical handlers — deterministic responses to source signals
//! that don't need to invoke an agent.
//!
//! Editorial flows (deciding which pile a card should be in, drafting
//! commentary) go through agent prompts and cost tokens. Some flows
//! are too cheap and too frequent for that — CI status updates on a
//! Watch card, "PR merged → move to Done." For those, register a
//! [`MechanicalHandler`] that mutates state deterministically with
//! `Provenance::Source { source_id }`.
//!
//! When [`DeskAgent::ingest_signal`](crate::DeskAgent::ingest_signal)
//! receives a [`ChangeNotice`](crate::ChangeNotice), it tries each
//! registered handler in registration order. The first whose
//! [`handles`](MechanicalHandler::handles) returns `true` runs
//! exclusively — the notice does *not* also fall through to OnSignal
//! tasks. Handlers that don't match leave the notice for the
//! merged-watch fallback path.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::broadcast;

use crate::Result;
use crate::card::{CardData, CardId};
use crate::event::DeskEvent;
use crate::ops;
use crate::provenance::{CardEventKind, Provenance};
use crate::source::{ChangeNotice, SourceId};
use crate::storage::DeskStorage;

#[async_trait]
pub trait MechanicalHandler: Send + Sync {
    /// Identifier for logs/debug. Default `"anon"`.
    fn id(&self) -> &str {
        "anon"
    }

    /// Predicate: does this handler claim the given notice?
    fn handles(&self, notice: &ChangeNotice) -> bool;

    /// Execute the deterministic mutation. Errors propagate back to
    /// [`DeskAgent::ingest_signal`](crate::DeskAgent::ingest_signal)'s
    /// caller — the host can decide whether to retry / log.
    async fn apply(&self, notice: ChangeNotice, ctx: &HandlerContext) -> Result<()>;
}

/// State exposed to a [`MechanicalHandler`] during `apply`. Narrower
/// than `Arc<DeskAgent>` to avoid coupling handler crates to the full
/// desk surface, and to keep the dependency direction one-way.
pub struct HandlerContext {
    pub storage: Arc<dyn DeskStorage>,
    pub events: broadcast::Sender<DeskEvent>,
    pub source_id: SourceId,
    pub history_cap: usize,
}

impl HandlerContext {
    /// `Provenance::Source { source_id }` for this handler's signal.
    pub fn provenance(&self) -> Provenance {
        Provenance::Source {
            source_id: self.source_id.clone(),
        }
    }

    /// Read-modify-write a card with `Provenance::Source` stamping
    /// and history-event append. Convenience wrapper around the same
    /// `ops::mutate_card_with_history` that DeskAgent and tools use.
    pub async fn mutate_card<F>(
        &self,
        id: &CardId,
        kind: CardEventKind,
        reason: Option<String>,
        mutate: F,
    ) -> Result<CardData>
    where
        F: FnOnce(&mut CardData) + Send,
    {
        ops::mutate_card_with_history(
            &*self.storage,
            id,
            self.provenance(),
            reason,
            kind,
            self.history_cap,
            mutate,
        )
        .await
    }
}
