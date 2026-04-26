//! Plan-step boundary tools — `step_started`, `step_completed`,
//! `plan_complete`. These are dumb event emitters used by the executing
//! agent to mark progress through an approved plan. Hosts pair the
//! events by `step_id` to render a Pending → Running → Done timeline.
//!
//! No actor state is involved; the runtime stays a faithful pipe.
//! Out-of-order or duplicate sequences are host-rendering issues, not
//! runtime errors.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;

use tau_agent::approval::ToolRisk;
use tau_agent::events::AgentEvent;
use tau_agent::tool::{Concurrency, ExecutionContext, Tool, ToolResult};

use crate::cached_schema;

#[derive(Deserialize, JsonSchema)]
struct StepStartedArgs {
    /// Stable identifier for this step (matches the plan's step.id).
    step_id: String,
    /// Optional one-line activity label shown in the running view's
    /// sub-label (e.g. "running tests").
    #[serde(default)]
    activity: Option<String>,
}

pub struct StepStartedTool;

impl Default for StepStartedTool {
    fn default() -> Self {
        Self
    }
}

impl StepStartedTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for StepStartedTool {
    fn name(&self) -> &str {
        "step_started"
    }

    fn description(&self) -> &str {
        "Mark the start of a plan step. Pass `step_id` matching the approved \
         plan and an optional `activity` label. Call this once per step before \
         doing the step's actual work."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(StepStartedArgs)
    }

    fn concurrency(&self) -> Concurrency {
        Concurrency::Sequential
    }

    fn risk(&self, _arguments: &serde_json::Value) -> ToolRisk {
        ToolRisk::Safe
    }

    fn activity_description(&self, arguments: &serde_json::Value) -> String {
        let id = arguments.get("step_id").and_then(|v| v.as_str()).unwrap_or("?");
        if let Some(activity) = arguments.get("activity").and_then(|v| v.as_str()) {
            format!("Step {id}: {activity}")
        } else {
            format!("Starting step {id}")
        }
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let args: StepStartedArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };
        ctx.progress.emit(AgentEvent::PlanStepStarted {
            step_id: args.step_id.clone(),
            activity: args.activity,
            started_at: chrono::Utc::now(),
        });
        ToolResult::text(format!("step started: {}", args.step_id))
    }
}

#[derive(Deserialize, JsonSchema)]
struct StepCompletedArgs {
    /// The step_id passed to the matching `step_started` call.
    step_id: String,
    /// Optional one-line summary of what the step did.
    #[serde(default)]
    summary: Option<String>,
}

pub struct StepCompletedTool;

impl Default for StepCompletedTool {
    fn default() -> Self {
        Self
    }
}

impl StepCompletedTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for StepCompletedTool {
    fn name(&self) -> &str {
        "step_completed"
    }

    fn description(&self) -> &str {
        "Mark the end of a plan step. Pass the same `step_id` as the matching \
         `step_started` call and an optional one-line `summary`. Call this \
         once per step after the step's work is done."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(StepCompletedArgs)
    }

    fn concurrency(&self) -> Concurrency {
        Concurrency::Sequential
    }

    fn risk(&self, _arguments: &serde_json::Value) -> ToolRisk {
        ToolRisk::Safe
    }

    fn activity_description(&self, arguments: &serde_json::Value) -> String {
        let id = arguments.get("step_id").and_then(|v| v.as_str()).unwrap_or("?");
        format!("Completing step {id}")
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let args: StepCompletedArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };
        ctx.progress.emit(AgentEvent::PlanStepCompleted {
            step_id: args.step_id.clone(),
            summary: args.summary,
            completed_at: chrono::Utc::now(),
        });
        ToolResult::text(format!("step completed: {}", args.step_id))
    }
}

#[derive(Deserialize, JsonSchema)]
struct PlanCompleteArgs {
    /// Brief summary of the overall outcome shown in the running-to-done
    /// transition.
    summary: String,
}

pub struct PlanCompleteTool;

impl Default for PlanCompleteTool {
    fn default() -> Self {
        Self
    }
}

impl PlanCompleteTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for PlanCompleteTool {
    fn name(&self) -> &str {
        "plan_complete"
    }

    fn description(&self) -> &str {
        "Mark the entire plan as complete with a brief overall summary. Call \
         once after the final step has finished."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(PlanCompleteArgs)
    }

    fn concurrency(&self) -> Concurrency {
        Concurrency::Sequential
    }

    fn risk(&self, _arguments: &serde_json::Value) -> ToolRisk {
        ToolRisk::Safe
    }

    fn activity_description(&self, _arguments: &serde_json::Value) -> String {
        "Marking plan complete".to_string()
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let args: PlanCompleteArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };
        ctx.progress.emit(AgentEvent::PlanCompleted {
            summary: args.summary,
            completed_at: chrono::Utc::now(),
        });
        ToolResult::text("plan complete")
    }
}
