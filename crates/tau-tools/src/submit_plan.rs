//! `submit_plan` tool — round-trip a structured implementation plan through
//! the host UI for review/approval.
//!
//! The tool's `parameters_schema` is the JSON Schema for [`Plan`]. Its
//! `execute` body sends a `Typed { schema_id: "plan.submit", payload }`
//! interaction request whose payload is the serialized [`Plan`], awaits the
//! response, and returns the (possibly-edited) approved plan as a JSON tool
//! result. On rejection, returns an error result the model can revise
//! against.

use async_trait::async_trait;

use tau_agent::Plan;
use tau_agent::interaction::{
    InteractionKind, InteractionRequest, InteractionResponse,
};
use tau_agent::tool::{Concurrency, ExecutionContext, Tool, ToolResult};

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
         and approve. After approval the tool returns the approved plan body. \
         If rejected, revise the plan based on the feedback and call again. \
         Once approved, do not call this tool again — output a brief final \
         summary and stop."
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
                    Ok(json) => ToolResult::text(format!("Plan approved:\n{json}")),
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
