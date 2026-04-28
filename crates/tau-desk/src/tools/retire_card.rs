use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::broadcast;

use tau_agent::tool::{ExecutionContext, Tool, ToolResult};

use crate::card::CardPile;
use crate::error::Error;
use crate::event::DeskEvent;
use crate::ops;
use crate::provenance::{CardEventKind, Provenance};
use crate::storage::DeskStorage;

/// Agent or user. Move a card to `Done` with a reason. Soft — a future
/// `upsert_card` with the same `external_ref` restores the card to the
/// upserter's chosen pile.
pub struct RetireCardTool {
    storage: Arc<dyn DeskStorage>,
    events: broadcast::Sender<DeskEvent>,
    agent_id: Option<String>,
    history_cap: usize,
}

impl RetireCardTool {
    pub fn new(
        storage: Arc<dyn DeskStorage>,
        events: broadcast::Sender<DeskEvent>,
        agent_id: Option<String>,
        history_cap: usize,
    ) -> Self {
        Self {
            storage,
            events,
            agent_id,
            history_cap,
        }
    }
}

#[async_trait]
impl Tool for RetireCardTool {
    fn name(&self) -> &str {
        "desk_retire_card"
    }

    fn description(&self) -> &str {
        "Retire a card to Done with a reason. Soft — if the underlying \
         work re-emerges (e.g., a closed Jira ticket gets reopened), a \
         future upsert restores the card. Use this when the card no longer \
         needs the user's attention but the source still exists."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "card_id": { "type": "string" },
                "reason":  { "type": ["string", "null"] }
            },
            "required": ["card_id"]
        })
    }

    async fn execute(&self, arguments: Value, _ctx: ExecutionContext) -> ToolResult {
        let id = match arguments.get("card_id").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => return ToolResult::error("`card_id` is required"),
        };
        let reason = arguments
            .get("reason")
            .and_then(|v| v.as_str())
            .map(String::from);

        // Check pile up front to reject Drafts.
        let current = match self.storage.read_card(&id).await {
            Ok(Some(c)) => c,
            Ok(None) => return ToolResult::error(format!("card `{id}` not found")),
            Err(e) => return ToolResult::error(format!("storage: {e}")),
        };
        if matches!(current.pile, CardPile::Drafts) {
            return ToolResult::error("cannot retire a Draft directly; reject_draft instead");
        }

        let by = Provenance::Agent {
            agent_id: self.agent_id.clone(),
        };

        let event_reason = reason.clone();
        let result = ops::mutate_card_with_history(
            &*self.storage,
            &id,
            by,
            reason.clone(),
            CardEventKind::Retired { reason: event_reason },
            self.history_cap,
            |c| c.pile = CardPile::Done,
        )
        .await;

        match result {
            Ok(_) => {
                let _ = self.events.send(DeskEvent::CardRetired {
                    id: id.clone(),
                    reason,
                });
                ToolResult::text(format!("retired `{id}`"))
            }
            Err(Error::NotFound(_)) => ToolResult::error(format!("card `{id}` not found")),
            Err(e) => ToolResult::error(format!("storage: {e}")),
        }
    }
}
