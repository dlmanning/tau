use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::broadcast;

use tau_agent::tool::{ExecutionContext, Tool, ToolResult};

use crate::card::Attachment;
use crate::error::Error;
use crate::event::DeskEvent;
use crate::ops;
use crate::provenance::{CardEventKind, Provenance};
use crate::storage::DeskStorage;

/// Agent-only. Add a structured `Attachment { kind, url, summary }` to
/// a card's top-level `attachments`. Used for cross-source synthesis —
/// e.g., a Slack thread that references a Jira ticket attaches to the
/// Jira card rather than spawning a duplicate Thread card.
///
/// Convention: `kind` describes the source-typed shape
/// (`"slack-thread"`, `"linked-pr"`, `"ci-run"`). User-authored
/// attachments use `kind: "user-note"` and arrive via
/// `desk.user_attach_note`.
pub struct AttachToCardTool {
    storage: Arc<dyn DeskStorage>,
    events: broadcast::Sender<DeskEvent>,
    agent_id: Option<String>,
    history_cap: usize,
}

impl AttachToCardTool {
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
impl Tool for AttachToCardTool {
    fn name(&self) -> &str {
        "desk_attach_to_card"
    }

    fn description(&self) -> &str {
        "Attach a cross-source reference to an existing card. Use when \
         you find related context in a different source (e.g., a Slack \
         thread about a Jira ticket) rather than creating a duplicate card."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "card_id": { "type": "string" },
                "kind":    { "type": "string" },
                "url":     { "type": ["string", "null"] },
                "summary": { "type": "string" }
            },
            "required": ["card_id", "kind", "summary"]
        })
    }

    async fn execute(&self, arguments: Value, _ctx: ExecutionContext) -> ToolResult {
        let id = match arguments.get("card_id").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => return ToolResult::error("`card_id` is required"),
        };
        let kind = match arguments.get("kind").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => return ToolResult::error("`kind` is required"),
        };
        if kind == "user-note" {
            return ToolResult::error(
                "kind `user-note` is reserved for user-authored attachments; \
                 the agent should pick a source-typed kind \
                 (`slack-thread`, `linked-pr`, `ci-run`, ...)",
            );
        }
        let url = arguments
            .get("url")
            .and_then(|v| v.as_str())
            .map(String::from);
        let summary = match arguments.get("summary").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => return ToolResult::error("`summary` is required"),
        };

        let kind_for_event = kind.clone();
        let attachment = Attachment {
            kind: kind.clone(),
            url,
            summary,
        };

        let by = Provenance::Agent {
            agent_id: self.agent_id.clone(),
        };

        let result = ops::mutate_card_with_history(
            &*self.storage,
            &id,
            by,
            None,
            CardEventKind::AttachmentAdded { kind: kind.clone() },
            self.history_cap,
            move |c| c.attachments.push(attachment),
        )
        .await;

        match result {
            Ok(_) => {
                let _ = self.events.send(DeskEvent::CardAttachmentAdded {
                    id: id.clone(),
                    kind: kind_for_event,
                });
                ToolResult::text(format!("attached to `{id}`"))
            }
            Err(Error::NotFound(_)) => ToolResult::error(format!("card `{id}` not found")),
            Err(e) => ToolResult::error(format!("storage: {e}")),
        }
    }
}
