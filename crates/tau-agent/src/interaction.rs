//! Protocol types for tool–UI interaction.
//!
//! Tools that need user input send an [`InteractionRequest`] through the
//! channel on [`ExecutionContext`](crate::tool::ExecutionContext) and await
//! the [`InteractionResponse`] on the embedded oneshot. The UI layer
//! (TUI, CLI, tests) handles the other end.
//!
//! There are two shapes of interaction:
//!
//! - [`InteractionKind::AskQuestion`] — a generic question with a list of
//!   options. The host returns [`InteractionResponse::Answer`] with the
//!   chosen label, or [`InteractionResponse::Cancelled`].
//! - [`InteractionKind::Typed`] — a schema-keyed structured payload. The host
//!   maintains a renderer table keyed on `schema_id`; tools and hosts agree
//!   on the JSON shape of `payload` per schema. The host replies with
//!   [`InteractionResponse::Approved`] (optionally carrying an edited
//!   payload), [`InteractionResponse::Rejected`], or
//!   [`InteractionResponse::Cancelled`].
//!
//! Two `schema_id` values are defined in this crate:
//!
//! - `"tool.confirm"` — emitted by the approval gate before running a Gated
//!   tool. Payload: `{ tool_call_id, tool_name, arguments, activity, risk }`.
//! - `"plan.submit"` — emitted by the `submit_plan` tool. Payload: a `Plan`
//!   (see [`crate::plan`]).

use serde_json::Value;
use tokio::sync::oneshot;

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
    /// Present a question with a set of options. Replied with
    /// [`InteractionResponse::Answer`] or [`InteractionResponse::Cancelled`].
    AskQuestion {
        question: String,
        options: Vec<QuestionOption>,
    },
    /// A schema-keyed structured interaction. Hosts dispatch on `schema_id`
    /// to a per-schema renderer. The reply is one of
    /// [`InteractionResponse::Approved`], [`InteractionResponse::Rejected`],
    /// or [`InteractionResponse::Cancelled`].
    Typed {
        schema_id: String,
        payload: Value,
    },
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
    /// User approved a `Typed` request. `payload` is `Some(value)` if the
    /// host edited the body before approving (the tool deserializes it as
    /// the schema dictates); `None` if the original payload should be used.
    Approved { payload: Option<Value> },
    /// User rejected a `Typed` request, with a reason that will be surfaced
    /// to the model as the tool's error result.
    Rejected { reason: String },
    /// User cancelled the interaction.
    Cancelled,
}
