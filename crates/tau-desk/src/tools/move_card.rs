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

/// Agent or user (verb-matrix). Change a card's pile. Rejects moves
/// into or out of `Drafts` (`Error::ManagedPile`).
pub struct MoveCardTool {
    storage: Arc<dyn DeskStorage>,
    events: broadcast::Sender<DeskEvent>,
    agent_id: Option<String>,
    history_cap: usize,
}

impl MoveCardTool {
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
impl Tool for MoveCardTool {
    fn name(&self) -> &str {
        "desk_move_card"
    }

    fn description(&self) -> &str {
        "Move a card to a different pile. Cannot move into or out of the \
         Drafts pile (drafts are queued actions, not just cards). \
         If the card was last modified by the user recently, the system prompt \
         instructs you to respect their intent unless you have a strong reason."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "card_id": { "type": "string" },
                "to":      { "type": "string", "enum": ["needs_you", "watching", "done"] },
                "reason":  { "type": ["string", "null"] }
            },
            "required": ["card_id", "to"]
        })
    }

    async fn execute(&self, arguments: Value, _ctx: ExecutionContext) -> ToolResult {
        let id = match arguments.get("card_id").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => return ToolResult::error("`card_id` is required"),
        };
        let to = match arguments.get("to").and_then(|v| v.as_str()) {
            Some("needs_you") => CardPile::NeedsYou,
            Some("watching") => CardPile::Watching,
            Some("done") => CardPile::Done,
            Some("drafts") => {
                return ToolResult::error("cannot move into the Drafts pile");
            }
            Some(other) => return ToolResult::error(format!("unknown pile `{other}`")),
            None => return ToolResult::error("`to` is required"),
        };
        let reason = arguments
            .get("reason")
            .and_then(|v| v.as_str())
            .map(String::from);

        // Check current pile up front so we can surface ManagedPile and
        // populate the Moved event with the correct `from`.
        let current = match self.storage.read_card(&id).await {
            Ok(Some(c)) => c,
            Ok(None) => return ToolResult::error(format!("card `{id}` not found")),
            Err(e) => return ToolResult::error(format!("storage: {e}")),
        };
        if matches!(current.pile, CardPile::Drafts) {
            return ToolResult::error("cannot move out of the Drafts pile");
        }
        let from = current.pile;
        if from == to {
            return ToolResult::text(format!("card `{id}` already in `{to:?}`"));
        }

        let by = Provenance::Agent {
            agent_id: self.agent_id.clone(),
        };

        let result = ops::mutate_card_with_history(
            &*self.storage,
            &id,
            by,
            reason,
            CardEventKind::Moved { from, to },
            self.history_cap,
            |c| c.pile = to,
        )
        .await;

        match result {
            Ok(_) => {
                let _ = self.events.send(DeskEvent::CardMoved {
                    id: id.clone(),
                    from,
                    to,
                });
                ToolResult::text(format!("moved `{id}` → {to:?}"))
            }
            Err(Error::NotFound(_)) => ToolResult::error(format!("card `{id}` not found")),
            Err(e) => ToolResult::error(format!("storage: {e}")),
        }
    }
}
