//! Plan submission: the [`Plan`] payload, the [`SubmitPlanTool`] that
//! emits it, and helpers for the planner-subagent flow.
//!
//! Three roles, all belonging to the planner→executor pattern:
//!
//! - **Payload types** — `Plan` / `PlanStep` / `PlanFile` / `PlanFlag`
//!   define the payload shape of the `plan.submit` typed interaction.
//!   `tau-agent` does not depend on these types; it routes the JSON
//!   payload opaquely.
//!
//! - **`SubmitPlanTool`** — the tool the model calls to submit a plan.
//!   Sends a `Typed { schema_id: "plan.submit", payload }` interaction
//!   request, awaits an `Approved { payload }` (carrying the edited body
//!   or `None` to accept the original) / `Rejected { reason }` /
//!   `Cancelled` reply, and returns the (possibly-edited) approved plan
//!   as a JSON tool result.
//!
//! - **Planner-subagent helpers** — `build_context_summary`,
//!   `extract_final_text`, and `build_plan_prompt` are used by the
//!   `agent` tool when spawning a Plan subagent and recovering the
//!   resulting plan body.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tau_agent::{Concurrency, ExecutionContext, Tool, ToolResult};
use tau_agent::{InteractionKind, InteractionRequest, InteractionResponse};
use tau_ai::{Content, Message};

// ---------------------------------------------------------------------------
// Plan payload types — schema for the `plan.submit` typed interaction.
// ---------------------------------------------------------------------------

/// The structured plan body submitted by the agent.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Plan {
    /// Ordered execution steps.
    pub items: Vec<PlanStep>,
    /// Files this plan will touch.
    #[serde(default)]
    pub files: Vec<PlanFile>,
    /// Concerns the planner wants to flag for the user before approval.
    #[serde(default)]
    pub flags: Vec<PlanFlag>,
}

/// A single step in the plan.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PlanStep {
    /// Stable identifier the host can reference.
    pub id: String,
    /// Short title for the step.
    pub title: String,
    /// Longer description of what the step does.
    pub description: String,
    /// File paths this step touches, for UI chips.
    #[serde(default)]
    pub touches: Vec<String>,
}

/// A file affected by the plan, with op + line counts for diff previews.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PlanFile {
    pub op: PlanFileOp,
    pub path: String,
    #[serde(default)]
    pub adds: u32,
    #[serde(default)]
    pub dels: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PlanFileOp {
    Add,
    Modify,
    Delete,
}

/// A pre-approval concern: incompatibility, missing context, irreversibility.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PlanFlag {
    pub severity: PlanFlagSeverity,
    pub title: String,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PlanFlagSeverity {
    Info,
    Warning,
    Danger,
}

// ---------------------------------------------------------------------------
// SubmitPlanTool — the tool that emits the `plan.submit` interaction.
// ---------------------------------------------------------------------------

pub struct SubmitPlanTool;

impl Default for SubmitPlanTool {
    fn default() -> Self {
        Self
    }
}

impl SubmitPlanTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for SubmitPlanTool {
    fn name(&self) -> &str {
        "submit_plan"
    }

    fn description(&self) -> &str {
        "Submit a structured implementation plan for the user to review, edit, \
         and approve.\n\n\
         Required argument shape (not a freeform string — a JSON object):\n\
         {\n  \
         \"items\": [\n    \
         {\"id\": \"s1\", \"title\": \"short imperative\", \"description\": \"what + why\", \"touches\": [\"path/to/file.rs\"]},\n    \
         {\"id\": \"s2\", ...}\n  \
         ],\n  \
         \"files\": [\n    \
         {\"op\": \"add|modify|delete\", \"path\": \"src/foo.rs\", \"adds\": 12, \"dels\": 3}\n  \
         ],\n  \
         \"flags\": [\n    \
         {\"severity\": \"info|warning|danger\", \"title\": \"…\", \"description\": \"…\"}\n  \
         ]\n}\n\n\
         `items` is required. `files` and `flags` are optional but encouraged. \
         After approval the tool returns the approved plan body. If rejected, \
         revise the plan based on the feedback and call again. Once approved, \
         do not call this tool again — output a brief final summary and stop."
    }

    fn concurrency(&self) -> Concurrency {
        Concurrency::Sequential
    }

    fn parameters_schema(&self) -> serde_json::Value {
        crate::cached_schema!(Plan)
    }

    fn activity_description(&self, _arguments: &serde_json::Value) -> String {
        "Submitting plan for review".to_string()
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let plan: Plan = match serde_json::from_value(arguments) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid plan: {}", e)),
        };

        let interaction_tx = match ctx.interaction {
            Some(ref tx) => tx.clone(),
            None => return ToolResult::error("No interactive session available for plan review"),
        };

        let payload = match serde_json::to_value(&plan) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format!("Failed to serialize plan: {e}")),
        };

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        let request = InteractionRequest {
            agent_id: None,
            kind: InteractionKind::Typed {
                schema_id: "plan.submit".to_string(),
                payload,
            },
            response_tx,
        };

        if interaction_tx.send(request).await.is_err() {
            return ToolResult::error("Interaction channel closed");
        }

        match response_rx.await {
            Ok(InteractionResponse::Approved { payload }) => {
                let approved_plan = match payload {
                    Some(value) => match serde_json::from_value::<Plan>(value) {
                        Ok(edited) => edited,
                        Err(e) => {
                            return ToolResult::error(format!(
                                "Edited plan failed to deserialize: {e}"
                            ));
                        }
                    },
                    None => plan,
                };
                match serde_json::to_string_pretty(&approved_plan) {
                    Ok(json) => ToolResult::text(format!(
                        "Plan accepted for user review:\n{json}\n\nThe user has the plan; \
                         they will choose when (or whether) to execute it. Output a brief \
                         acknowledgement and stop. Do not call submit_plan again."
                    )),
                    Err(e) => ToolResult::error(format!("Failed to serialize approved plan: {e}")),
                }
            }
            Ok(InteractionResponse::Rejected { reason }) => {
                ToolResult::error(format!("Plan rejected: {reason}"))
            }
            Ok(InteractionResponse::Cancelled) => ToolResult::error("Plan review cancelled"),
            Ok(InteractionResponse::Answer(_)) => {
                ToolResult::error("Unexpected response to plan submission")
            }
            Err(_) => ToolResult::error("Interaction channel closed"),
        }
    }
}

// ---------------------------------------------------------------------------
// Plan-mode utilities — context summary for Plan subagent injection.
// ---------------------------------------------------------------------------

/// Maximum number of recent messages to include in the context summary.
const MAX_SUMMARY_MESSAGES: usize = 20;

/// Build a lightweight text summary of recent conversation for injection
/// into a Plan subagent. Includes user/assistant text, skips tool
/// calls and results for brevity.
pub fn build_context_summary(messages: &[Message], previous_summary: Option<&str>) -> String {
    let mut parts: Vec<String> = Vec::new();

    if let Some(summary) = previous_summary {
        parts.push(format!("Earlier conversation summary:\n{}", summary));
    }

    let start = messages.len().saturating_sub(MAX_SUMMARY_MESSAGES);
    for msg in &messages[start..] {
        match msg {
            Message::User { content, .. } => {
                let text = extract_text(content);
                if !text.is_empty() {
                    parts.push(format!("User: {}", text));
                }
            }
            Message::Assistant { content, .. } => {
                let text = extract_text(content);
                if !text.is_empty() {
                    parts.push(format!("Assistant: {}", text));
                }
            }
            // Skip tool results and system injections
            _ => {}
        }
    }

    parts.join("\n\n")
}

fn extract_text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| c.as_text())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extract the last non-empty assistant text from a message list.
/// Used to recover the plan from a Plan subagent's transcript.
pub fn extract_final_text(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find_map(|m| {
            if let Message::Assistant { content, .. } = m {
                let text: String = content
                    .iter()
                    .filter_map(|c| c.as_text())
                    .collect::<Vec<_>>()
                    .join("");
                if text.is_empty() { None } else { Some(text) }
            } else {
                None
            }
        })
        .unwrap_or_default()
}

/// Format the full prompt for a Plan subagent, combining context and task description.
pub fn build_plan_prompt(context_summary: &str, description: &str) -> String {
    if context_summary.is_empty() {
        format!(
            "Create an implementation plan for the following task:\n\n{}",
            description
        )
    } else {
        format!(
            "Here is context from the conversation so far:\n\n\
             <context>\n{}\n</context>\n\n\
             Create an implementation plan for the following task:\n\n{}",
            context_summary, description
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_round_trips_via_json() {
        let plan = Plan {
            items: vec![PlanStep {
                id: "s1".into(),
                title: "Add module".into(),
                description: "Create new module".into(),
                touches: vec!["src/foo.rs".into()],
            }],
            files: vec![PlanFile {
                op: PlanFileOp::Add,
                path: "src/foo.rs".into(),
                adds: 10,
                dels: 0,
            }],
            flags: vec![PlanFlag {
                severity: PlanFlagSeverity::Warning,
                title: "Migration".into(),
                description: "Requires DB migration".into(),
            }],
        };

        let json = serde_json::to_value(&plan).unwrap();
        let back: Plan = serde_json::from_value(json).unwrap();
        assert_eq!(back.items.len(), 1);
        assert_eq!(back.files[0].op, PlanFileOp::Add);
        assert_eq!(back.flags[0].severity, PlanFlagSeverity::Warning);
    }

    #[test]
    fn plan_schema_lists_required_fields() {
        let schema = schemars::schema_for!(Plan);
        let json = serde_json::to_value(&schema).unwrap();
        let required = json
            .get("required")
            .and_then(|r| r.as_array())
            .cloned()
            .unwrap_or_default();
        let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"items"));
    }
}
