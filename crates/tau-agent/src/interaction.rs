//! Protocol types for tool–UI interaction.
//!
//! Tools that need user input send an [`InteractionRequest`] through the
//! channel on [`ExecutionContext`](crate::tool::ExecutionContext) and await
//! the [`InteractionResponse`] on the embedded oneshot. The UI layer
//! (TUI, CLI, tests) handles the other end.

use tokio::sync::oneshot;

/// A request from a tool to the UI layer.
pub struct InteractionRequest {
    pub kind: InteractionKind,
    pub response_tx: oneshot::Sender<InteractionResponse>,
}

/// What the tool is asking for.
pub enum InteractionKind {
    /// Present a question with a set of options.
    AskQuestion {
        question: String,
        options: Vec<QuestionOption>,
    },
}

/// A single option in a question.
pub struct QuestionOption {
    pub label: String,
    pub description: String,
}

/// The UI layer's answer.
pub enum InteractionResponse {
    /// User picked an option (the label string).
    Answer(String),
    /// User cancelled the interaction.
    Cancelled,
}
