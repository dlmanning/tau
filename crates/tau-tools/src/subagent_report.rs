//! `subagent_report` — a subagent self-labels its outcome before terminating.
//!
//! Emits [`AgentEvent::SubagentReport`] on the subagent's own stream. The
//! parent's host receives it wrapped as `Subagent { event: SubagentReport }`
//! and correlates with the eventual `SubagentCompleted` by `agent_id`.
//!
//! The `tag` is intentionally free-form — different host products want
//! different vocabularies (`"passed"`/`"failed"`, `"approve"`/`"changes"`,
//! `"shipped"`/`"reverted"`). Use whatever your UI renders.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;

use tau_agent::AgentEvent;
use tau_agent::ToolRisk;
use tau_agent::{Concurrency, ExecutionContext, Tool, ToolResult};

use crate::cached_schema;

#[derive(Deserialize, JsonSchema)]
struct SubagentReportArgs {
    /// Optional short label the host UI will render as a badge (free-form,
    /// product-specific vocabulary). Examples: "passed", "failed", "approve",
    /// "changes", "shipped".
    #[serde(default)]
    tag: Option<String>,
    /// One- or two-sentence summary of what the subagent concluded. The
    /// host shows this in the collapsed subagent block.
    summary: String,
}

pub struct SubagentReportTool;

impl Default for SubagentReportTool {
    fn default() -> Self {
        Self
    }
}

impl SubagentReportTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for SubagentReportTool {
    fn name(&self) -> &str {
        "subagent_report"
    }

    fn description(&self) -> &str {
        "Self-label your final outcome before stopping. Pass an optional `tag` \
         (a short product-specific badge like \"passed\" or \"changes\") and a \
         brief `summary` of what you concluded. Call this once near the end \
         of your work; the parent's UI uses it to render your block."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(SubagentReportArgs)
    }

    fn concurrency(&self) -> Concurrency {
        Concurrency::Sequential
    }

    fn risk(&self, _arguments: &serde_json::Value) -> ToolRisk {
        ToolRisk::Safe
    }

    fn activity_description(&self, _arguments: &serde_json::Value) -> String {
        "Reporting outcome".to_string()
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let args: SubagentReportArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };
        ctx.progress.emit(AgentEvent::SubagentReport {
            tag: args.tag,
            summary: args.summary,
        });
        ToolResult::text("reported")
    }
}
