use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};
use tokio::sync::broadcast;
use uuid::Uuid;

use tau_agent::tool::{ExecutionContext, Tool, ToolResult};

use crate::activity::{ActivityEntry, ActivityKind, SessionSeed};
use crate::event::DeskEvent;
use crate::storage::DeskStorage;

/// Agent-only. Append an entry to the activity feed.
///
/// Entries with `suggest_session.is_some()` become Suggestion chips in
/// the Now-zone projection (until muted by the user via
/// `mute_suggestion(seed_from)`). Entries with `kind: TombstoneHit` are
/// the canonical fallback when `upsert_card` fails on a dismissed
/// `external_ref`.
pub struct AddActivityTool {
    storage: Arc<dyn DeskStorage>,
    events: broadcast::Sender<DeskEvent>,
}

impl AddActivityTool {
    pub fn new(storage: Arc<dyn DeskStorage>, events: broadcast::Sender<DeskEvent>) -> Self {
        Self { storage, events }
    }
}

#[async_trait]
impl Tool for AddActivityTool {
    fn name(&self) -> &str {
        "desk_add_activity"
    }

    fn description(&self) -> &str {
        "Append an entry to the activity feed. Optional `suggest_session` \
         turns the entry into a Now-zone Suggestion chip the user can \
         click to start a coding session."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "text":            { "type": "string" },
                "kind":            { "type": ["object", "null"] },
                "suggest_session": {
                    "type": ["object", "null"],
                    "properties": {
                        "title":     { "type": "string" },
                        "project":   { "type": ["string", "null"] },
                        "branch":    { "type": ["string", "null"] },
                        "kickoff":   { "type": "string" },
                        "seed_from": { "type": ["string", "null"] }
                    },
                    "required": ["title", "kickoff"]
                }
            },
            "required": ["text"]
        })
    }

    async fn execute(&self, arguments: Value, _ctx: ExecutionContext) -> ToolResult {
        let text = match arguments.get("text").and_then(|v| v.as_str()) {
            Some(t) if !t.is_empty() => t.to_string(),
            _ => return ToolResult::error("`text` is required"),
        };

        let kind: Option<ActivityKind> = arguments
            .get("kind")
            .filter(|v| !v.is_null())
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let suggest_session: Option<SessionSeed> = arguments
            .get("suggest_session")
            .filter(|v| !v.is_null())
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let entry = ActivityEntry {
            id: format!("act:{}", Uuid::new_v4()),
            seq: 0, // assigned by storage
            at: Utc::now(),
            text,
            kind,
            suggest_session,
        };

        match self.storage.append_activity(&entry).await {
            Ok(seq) => {
                let mut emitted = entry.clone();
                emitted.seq = seq;
                let _ = self.events.send(DeskEvent::ActivityAppended { entry: emitted });
                ToolResult::text(format!("recorded activity {}", entry.id))
            }
            Err(e) => ToolResult::error(format!("storage error: {e}")),
        }
    }
}
