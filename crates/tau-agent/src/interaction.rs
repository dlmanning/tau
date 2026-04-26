//! Protocol types for tool–UI interaction.
//!
//! Tools that need user input send an [`InteractionRequest`] through the
//! channel on [`ExecutionContext`](crate::tool::ExecutionContext) and await
//! the [`InteractionResponse`] on the embedded oneshot. The UI layer
//! (TUI, CLI, tests) handles the other end.

use serde_json::Value;
use tokio::sync::oneshot;

use crate::approval::ToolRisk;
use crate::plan::Plan;

/// A request from a tool (or the runtime) to the UI layer.
pub struct InteractionRequest {
    /// Identity of the agent that originated the request. `None` for the root
    /// agent; `Some(agent_id)` for a subagent. Hosts that render a tree of
    /// agents use this to attribute the prompt to the right node.
    pub agent_id: Option<String>,
    pub kind: InteractionKind,
    pub response_tx: oneshot::Sender<InteractionResponse>,
}

/// What the tool (or runtime) is asking for.
pub enum InteractionKind {
    /// Present a question with a set of options.
    AskQuestion {
        question: String,
        options: Vec<QuestionOption>,
    },
    /// Confirm execution of a tool call before it runs. Emitted by the
    /// approval gate, not by tools themselves.
    ConfirmTool {
        tool_call_id: String,
        tool_name: String,
        arguments: Value,
        activity: String,
        risk: ToolRisk,
    },
    /// Submit a structured plan for review. The host renders, optionally
    /// edits, and replies with `PlanApproved { plan }` (carrying the edited
    /// body) or `Rejected { reason }`.
    SubmitPlan { plan: Plan },
}

/// A single option in a question.
pub struct QuestionOption {
    pub label: String,
    pub description: String,
}

/// The UI layer's answer.
pub enum InteractionResponse {
    /// User picked an option (the label string). Reply to `AskQuestion`.
    Answer(String),
    /// User cancelled the interaction.
    Cancelled,
    /// User approved a `ConfirmTool` request.
    Approved,
    /// User rejected a `ConfirmTool` or `SubmitPlan` request, with a reason
    /// that will be surfaced to the model as the tool's error result.
    Rejected { reason: String },
    /// User approved a `SubmitPlan` request. The body is the (possibly-edited)
    /// plan; `submit_plan` returns it to the model as the tool result.
    PlanApproved { plan: Plan },
}
