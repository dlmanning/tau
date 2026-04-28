use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};
use tokio::sync::broadcast;

use tau_agent::tool::{ExecutionContext, Tool, ToolResult};

use crate::card::{AgentTake, CardBody, CardData, CardPile};
use crate::error::Error;
use crate::event::DeskEvent;
use crate::provenance::{CardEvent, CardEventKind, Provenance};
use crate::storage::DeskStorage;

/// Agent-only. Insert or refresh a source-backed card. Dedup by
/// `external_ref`. Body must be one of the source-derived `CardBody`
/// variants (not `Note`). Returns `Err(Tombstoned)` if the
/// `external_ref` is dismissed.
///
/// Preserves existing `agent_take`, `attachments`, `pinned`, and
/// `created_at` on update — the agent's `body` refresh shouldn't erase
/// user-added context (notes, source synthesis, pin state).
pub struct UpsertCardTool {
    storage: Arc<dyn DeskStorage>,
    events: broadcast::Sender<DeskEvent>,
    agent_id: Option<String>,
    history_cap: usize,
}

impl UpsertCardTool {
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
impl Tool for UpsertCardTool {
    fn name(&self) -> &str {
        "desk_upsert_card"
    }

    fn description(&self) -> &str {
        "Create or refresh a source-backed card on the desk. \
         Dedup by external_ref. Returns an error if the ref is tombstoned \
         (the user has dismissed it); fall back to add_activity in that case. \
         Existing agent_take, attachments, pin state, and created_at are \
         preserved on update."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id":           { "type": "string" },
                "external_ref": { "type": ["string", "null"] },
                "pile":         { "type": "string", "enum": ["needs_you", "watching"] },
                "body":         { "type": "object" },
                "agent_take": {
                    "type": ["object", "null"],
                    "properties": {
                        "ask":  { "type": ["string", "null"] },
                        "note": { "type": ["string", "null"] }
                    }
                },
                "metadata": { "type": ["object", "null"] },
                "reason":   { "type": ["string", "null"] }
            },
            "required": ["id", "pile", "body"]
        })
    }

    async fn execute(&self, arguments: Value, _ctx: ExecutionContext) -> ToolResult {
        let id = match arguments.get("id").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => return ToolResult::error("`id` is required"),
        };

        let pile = match arguments.get("pile").and_then(|v| v.as_str()) {
            Some("needs_you") => CardPile::NeedsYou,
            Some("watching") => CardPile::Watching,
            Some(other) => {
                return ToolResult::error(format!(
                    "cannot upsert into pile `{other}`; only `needs_you` and `watching` are allowed"
                ));
            }
            None => CardPile::NeedsYou,
        };

        let body: CardBody = match arguments.get("body") {
            Some(v) => match serde_json::from_value(v.clone()) {
                Ok(b) => b,
                Err(e) => return ToolResult::error(format!("invalid body: {e}")),
            },
            None => return ToolResult::error("`body` is required"),
        };

        if body.is_user_owned() {
            return ToolResult::error(
                "cannot upsert a Note via this tool — Notes are user-only \
                 (see desk.user_create_note)",
            );
        }

        let external_ref = arguments
            .get("external_ref")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from);

        let metadata = arguments
            .get("metadata")
            .filter(|v| !v.is_null())
            .cloned()
            .unwrap_or(json!({}));
        let reason = arguments
            .get("reason")
            .and_then(|v| v.as_str())
            .map(String::from);

        let agent_take = arguments
            .get("agent_take")
            .filter(|v| !v.is_null())
            .and_then(|v| {
                let ask = v
                    .get("ask")
                    .and_then(|x| x.as_str())
                    .map(String::from);
                let note = v
                    .get("note")
                    .and_then(|x| x.as_str())
                    .map(String::from);
                if ask.is_none() && note.is_none() {
                    None
                } else {
                    Some(AgentTake {
                        ask,
                        note,
                        updated_at: Utc::now(),
                    })
                }
            });

        // Look up existing card — by id first, then by external_ref.
        let existing = match self.storage.read_card(&id).await {
            Ok(Some(c)) => Some(c),
            Ok(None) => {
                if let Some(r) = &external_ref {
                    self.storage.read_card_by_ref(r).await.unwrap_or(None)
                } else {
                    None
                }
            }
            Err(e) => return ToolResult::error(format!("storage: {e}")),
        };

        let now = Utc::now();
        let by = Provenance::Agent {
            agent_id: self.agent_id.clone(),
        };

        let card = match existing {
            Some(prev) => {
                // Preserve sacred fields; overwrite agent-managed ones.
                let take = agent_take.or(prev.agent_take.clone());
                let mut history = prev.history.clone();
                history.push_back(CardEvent {
                    at: now,
                    by: by.clone(),
                    kind: CardEventKind::Updated,
                });
                while history.len() > self.history_cap {
                    history.pop_front();
                }
                CardData {
                    id: prev.id.clone(),
                    pile,
                    external_ref,
                    body,
                    agent_take: take,
                    attachments: prev.attachments.clone(),
                    metadata,
                    pinned: prev.pinned,
                    created_at: prev.created_at,
                    last_modified: now,
                    last_modified_by: by,
                    last_modified_reason: reason,
                    history,
                }
            }
            None => CardData {
                id,
                pile,
                external_ref,
                body,
                agent_take,
                attachments: vec![],
                metadata,
                pinned: false,
                created_at: now,
                last_modified: now,
                last_modified_by: by.clone(),
                last_modified_reason: reason,
                history: VecDeque::from([CardEvent {
                    at: now,
                    by,
                    kind: CardEventKind::Created,
                }]),
            },
        };

        match self.storage.upsert_card(&card).await {
            Ok(outcome) => {
                let card_id = card.id.clone();
                let _ = self.events.send(DeskEvent::CardUpserted { card });
                ToolResult::text(format!("upserted `{card_id}` ({outcome:?})"))
            }
            Err(Error::Tombstoned {
                external_ref,
                dismissed_at,
                reason,
            }) => ToolResult::error(format!(
                "external_ref `{external_ref}` is tombstoned (dismissed at \
                 {dismissed_at}{}); fall back to `add_activity`",
                reason
                    .map(|r| format!(": {r}"))
                    .unwrap_or_default()
            )),
            Err(e) => ToolResult::error(format!("upsert failed: {e}")),
        }
    }
}
