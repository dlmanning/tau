use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{Value, json};
use tokio::sync::broadcast;

use tau_agent::tool::{ExecutionContext, Tool, ToolResult};

use crate::brief::{Brief, BriefStat};
use crate::event::DeskEvent;
use crate::storage::DeskStorage;

/// Agent-only. Replace the singleton `Brief`. Typically called as the
/// last step of a morning scan, with a 2–3 sentence summary and a
/// fresh `stats` array.
pub struct UpdateBriefTool {
    storage: Arc<dyn DeskStorage>,
    events: broadcast::Sender<DeskEvent>,
}

impl UpdateBriefTool {
    pub fn new(storage: Arc<dyn DeskStorage>, events: broadcast::Sender<DeskEvent>) -> Self {
        Self { storage, events }
    }
}

#[async_trait]
impl Tool for UpdateBriefTool {
    fn name(&self) -> &str {
        "desk_update_brief"
    }

    fn description(&self) -> &str {
        "Replace the desk's morning brief: greeting + a 2-3 sentence \
         summary + a stats array of (label, value, optional delta). \
         Typically the final step of the morning scan."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "greeting": { "type": "string" },
                "summary":  { "type": "string" },
                "stats": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": { "type": "string" },
                            "value": { "type": "string" },
                            "delta": { "type": ["string", "null"] }
                        },
                        "required": ["label", "value"]
                    }
                }
            },
            "required": ["greeting", "summary", "stats"]
        })
    }

    async fn execute(&self, arguments: Value, _ctx: ExecutionContext) -> ToolResult {
        let greeting = arguments
            .get("greeting")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let summary = arguments
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let stats: Vec<BriefStat> = arguments
            .get("stats")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let brief = Brief {
            greeting,
            summary,
            stats,
            updated_at: Utc::now(),
        };

        match self.storage.write_brief(&brief).await {
            Ok(_) => {
                let _ = self.events.send(DeskEvent::BriefUpdated {
                    brief: brief.clone(),
                });
                ToolResult::text("updated brief")
            }
            Err(e) => ToolResult::error(format!("storage: {e}")),
        }
    }
}
