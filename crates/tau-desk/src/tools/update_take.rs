use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};
use tokio::sync::broadcast;

use tau_agent::{ExecutionContext, Tool, ToolResult};

use crate::card::AgentTake;
use crate::error::Error;
use crate::event::DeskEvent;
use crate::ops;
use crate::provenance::{CardEventKind, Provenance};
use crate::storage::DeskStorage;

/// Agent-only. Patch `agent_take.{ask, note}` on a card without
/// rewriting `body`. Cheap revisions for chatty take-revisers; emits
/// `CardEventKind::TakeUpdated` to the card's history.
pub struct UpdateTakeTool {
    storage: Arc<dyn DeskStorage>,
    events: broadcast::Sender<DeskEvent>,
    agent_id: Option<String>,
    history_cap: usize,
}

impl UpdateTakeTool {
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
impl Tool for UpdateTakeTool {
    fn name(&self) -> &str {
        "desk_update_take"
    }

    fn description(&self) -> &str {
        "Update the agent's editorial take on a card. `ask` is the imperative \
         (\"why this is on your plate\"); `note` is the editorial commentary. \
         Use this to revise commentary without re-deriving body from source."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "card_id": { "type": "string" },
                "ask":     { "type": ["string", "null"] },
                "note":    { "type": ["string", "null"] }
            },
            "required": ["card_id"]
        })
    }

    async fn execute(&self, arguments: Value, _ctx: ExecutionContext) -> ToolResult {
        let id = match arguments.get("card_id").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => return ToolResult::error("`card_id` is required"),
        };

        let ask = arguments
            .get("ask")
            .and_then(|v| v.as_str())
            .map(String::from);
        let note = arguments
            .get("note")
            .and_then(|v| v.as_str())
            .map(String::from);
        if ask.is_none() && note.is_none() {
            return ToolResult::error("provide at least one of `ask` or `note`");
        }

        let by = Provenance::Agent {
            agent_id: self.agent_id.clone(),
        };

        let result = ops::mutate_card_with_history(
            &*self.storage,
            &id,
            by,
            None,
            CardEventKind::TakeUpdated,
            self.history_cap,
            move |c| {
                c.agent_take = Some(AgentTake {
                    ask,
                    note,
                    updated_at: Utc::now(),
                });
            },
        )
        .await;

        match result {
            Ok(_) => {
                let _ = self
                    .events
                    .send(DeskEvent::CardTakeUpdated { id: id.clone() });
                ToolResult::text(format!("updated take on `{id}`"))
            }
            Err(Error::NotFound(_)) => ToolResult::error(format!("card `{id}` not found")),
            Err(e) => ToolResult::error(format!("storage: {e}")),
        }
    }
}
