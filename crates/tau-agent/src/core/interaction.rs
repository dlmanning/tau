//! Tool → host UI round-trip protocol.
//!
//! Tools that need user input send an [`InteractionRequest`] through
//! [`ExecutionContext::interaction`](crate::core::tool::ExecutionContext)
//! and await the [`InteractionResponse`] on the embedded oneshot. The
//! host UI (TUI, CLI, web, tests) handles the other end.
//!
//! Two shapes:
//!
//! - [`InteractionKind::AskQuestion`] — generic question with a list of
//!   options. Reply with [`InteractionResponse::Answer`] or
//!   [`InteractionResponse::Cancelled`].
//! - [`InteractionKind::Typed`] — schema-keyed structured payload. Hosts
//!   maintain a renderer table keyed on `schema_id`; tools and hosts
//!   agree on the JSON shape per schema. Reply with
//!   [`InteractionResponse::Approved`] (optionally edited),
//!   [`InteractionResponse::Rejected`], or [`InteractionResponse::Cancelled`].
//!
//! The runtime defines exactly one `schema_id`:
//!
//! - `"tool.confirm"` — emitted by the approval gate before running a
//!   `Gate`d tool. Payload: `{ tool_call_id, tool_name, arguments,
//!   activity, risk }`.
//!
//! Any other `schema_id` is host- or tool-defined; the runtime treats
//! the payload as opaque JSON.

use serde_json::Value;
use tokio::sync::oneshot;

pub struct InteractionRequest {
    /// Identity of the agent that originated the request. `None` for
    /// the root agent; `Some(id)` for a subagent (the fleet bus stamps
    /// this on outgoing requests). Hosts that render a tree of agents
    /// use this to attribute the prompt.
    pub agent_id: Option<String>,
    pub kind: InteractionKind,
    pub response_tx: oneshot::Sender<InteractionResponse>,
}

pub enum InteractionKind {
    AskQuestion {
        question: String,
        options: Vec<QuestionOption>,
    },
    Typed {
        schema_id: String,
        payload: Value,
    },
}

pub struct QuestionOption {
    pub label: String,
    pub description: String,
}

pub enum InteractionResponse {
    /// Reply to `AskQuestion`. Carries the chosen label.
    Answer(String),
    /// Reply to `Typed`. `payload = Some(value)` if the host edited the
    /// body before approving; `None` to use the original.
    Approved {
        payload: Option<Value>,
    },
    Rejected {
        reason: String,
    },
    Cancelled,
}
