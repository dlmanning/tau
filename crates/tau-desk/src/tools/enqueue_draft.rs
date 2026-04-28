use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};
use tokio::sync::broadcast;
use uuid::Uuid;

use tau_agent::tool::{ExecutionContext, Tool, ToolResult};

use crate::card::{CardBody, CardData, CardPile};
use crate::draft::{Draft, DraftStatus};
use crate::event::DeskEvent;
use crate::provenance::{CardEvent, CardEventKind, Provenance};
use crate::storage::DeskStorage;

/// Agent-only. Queue a tool call for the user to approve later.
///
/// On approval, the desk looks up `tool_name` in the registry and
/// dispatches it with `arguments` (bypassing the runtime's
/// `ApprovalPolicy` — the user already approved at the draft level).
///
/// Side effects: writes a `Draft` row, then upserts a `CardData` with
/// `body: CardBody::Draft { draft_id, summary }` into the `Drafts`
/// pile. The card is what users see; the draft row carries the
/// dispatchable payload.
pub struct EnqueueDraftTool {
    storage: Arc<dyn DeskStorage>,
    events: broadcast::Sender<DeskEvent>,
    agent_id: Option<String>,
}

impl EnqueueDraftTool {
    pub fn new(
        storage: Arc<dyn DeskStorage>,
        events: broadcast::Sender<DeskEvent>,
        agent_id: Option<String>,
    ) -> Self {
        Self {
            storage,
            events,
            agent_id,
        }
    }
}

#[async_trait]
impl Tool for EnqueueDraftTool {
    fn name(&self) -> &str {
        "desk_enqueue_draft"
    }

    fn description(&self) -> &str {
        "Queue a tool call for the user to approve later. Use this for \
         comments, replies, posts — anything you shouldn't send unsupervised \
         but don't need to await. Provide a rationale: the user will see it \
         when deciding whether to approve. `summary` is the short label \
         shown on the draft card."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Name of an existing tool in the registry to dispatch on approval."
                },
                "arguments": { "type": "object" },
                "rationale": { "type": ["string", "null"] },
                "summary":   { "type": ["string", "null"] }
            },
            "required": ["tool_name", "arguments"]
        })
    }

    async fn execute(&self, arguments: Value, _ctx: ExecutionContext) -> ToolResult {
        let tool_name = match arguments.get("tool_name").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => return ToolResult::error("`tool_name` is required"),
        };
        let args = arguments
            .get("arguments")
            .filter(|v| !v.is_null())
            .cloned()
            .unwrap_or(json!({}));
        let rationale = arguments
            .get("rationale")
            .and_then(|v| v.as_str())
            .map(String::from);
        let summary = arguments
            .get("summary")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| format!("Pending: {tool_name}"));

        // Best-effort source attribution: tools follow the
        // `<source>_<verb>` convention.
        let source_id = tool_name
            .split_once('_')
            .map(|(prefix, _)| prefix.to_string());

        let now = Utc::now();
        let by = Provenance::Agent {
            agent_id: self.agent_id.clone(),
        };
        let draft_id = format!("draft:{}", Uuid::new_v4());

        let draft = Draft {
            id: draft_id.clone(),
            source_id,
            tool_name,
            arguments: args,
            rationale,
            status: DraftStatus::Pending,
            created_at: now,
            resolved_at: None,
            outcome: None,
        };

        if let Err(e) = self.storage.write_draft(&draft).await {
            return ToolResult::error(format!("write draft: {e}"));
        }

        let card_id = format!("card-{draft_id}");
        let card = CardData {
            id: card_id.clone(),
            pile: CardPile::Drafts,
            external_ref: None,
            body: CardBody::Draft {
                draft_id: draft_id.clone(),
                summary,
            },
            agent_take: None,
            attachments: vec![],
            metadata: json!({}),
            pinned: false,
            created_at: now,
            last_modified: now,
            last_modified_by: by.clone(),
            last_modified_reason: None,
            history: VecDeque::from([CardEvent {
                at: now,
                by,
                kind: CardEventKind::Created,
            }]),
        };
        if let Err(e) = self.storage.upsert_card(&card).await {
            return ToolResult::error(format!("upsert draft card: {e}"));
        }

        let _ = self.events.send(DeskEvent::DraftCreated {
            draft: draft.clone(),
        });
        let _ = self.events.send(DeskEvent::CardUpserted { card });

        ToolResult::text(format!("queued draft `{draft_id}`"))
    }
}
