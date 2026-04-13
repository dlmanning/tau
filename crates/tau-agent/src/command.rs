//! Internal command protocol between AgentHandle and the actor task.

use tau_ai::{Content, Message, Model, ReasoningLevel};
use tokio::sync::oneshot;

use crate::compaction::{CompactionConfig, CompactionReason};
use crate::config::AgentConfig;
use crate::conversation::Conversation;

/// Result of a prompt or compaction operation.
pub struct PromptResult {
    pub result: Result<(), crate::error::Error>,
}

/// Commands sent from AgentHandle to the actor task.
pub(crate) enum Command {
    // === Prompt lifecycle ===
    Prompt {
        content: Vec<Content>,
        reply: oneshot::Sender<PromptResult>,
    },
    Steer(Message),
    FollowUp(Message),

    // === Config mutations (fire-and-forget) ===
    SetModel(Model),
    SetReasoning(ReasoningLevel),
    SetSystemPrompt(String),
    SetCompactionConfig(CompactionConfig),

    // === Conversation mutations (fire-and-forget) ===
    ClearMessages,
    SetMessages(Vec<Message>),
    SetPreviousSummary(Option<String>),

    // === Queries (request-reply via oneshot) ===
    GetConfig(oneshot::Sender<AgentConfig>),
    GetMessages(oneshot::Sender<Vec<Message>>),
    GetState(oneshot::Sender<Conversation>),

    // === Manual compaction ===
    Compact {
        reason: CompactionReason,
        reply: oneshot::Sender<PromptResult>,
    },
}
