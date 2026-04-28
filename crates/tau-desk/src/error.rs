use chrono::{DateTime, Utc};
use thiserror::Error;

use crate::card::{CardId, CardPile};
use crate::draft::DraftId;
use crate::provenance::Provenance;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("card not found: {0}")]
    NotFound(CardId),

    #[error(
        "external_ref `{external_ref}` is tombstoned (dismissed at {dismissed_at}{})",
        reason.as_ref().map(|r| format!(": {r}")).unwrap_or_default()
    )]
    Tombstoned {
        external_ref: String,
        dismissed_at: DateTime<Utc>,
        reason: Option<String>,
    },

    #[error("`{0:?}` is a managed pile; cards cannot be moved into or out of it")]
    ManagedPile(CardPile),

    #[error("operation requires {expected:?}, but caller is {actual:?}")]
    WrongAuthor {
        expected: AuthorClass,
        actual: AuthorClass,
    },

    #[error("draft not found: {0}")]
    DraftNotFound(DraftId),

    #[error("draft `{0}` already resolved")]
    DraftAlreadyResolved(DraftId),

    #[error("source `{0}` is not registered")]
    UnknownSource(String),

    #[error("tool `{0}` is not in the registry; cannot dispatch")]
    UnknownTool(String),

    #[error(
        "card `{card_id}` was last modified by user {hours_ago}h ago; \
         pass `override = true` with a justification if the agent must override"
    )]
    UserOverrideRecent {
        card_id: CardId,
        hours_ago: u32,
        by: Provenance,
    },

    #[error("storage: {0}")]
    Storage(#[source] anyhow::Error),

    #[error("source error: {0}")]
    Source(#[source] anyhow::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthorClass {
    User,
    Agent,
    Source,
}
