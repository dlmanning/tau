//! Internal command protocol between AgentHandle and the actor task.

use std::sync::Arc;

use tau_ai::{Content, Message, Model, ReasoningLevel};
use tokio::sync::oneshot;

use crate::approval::ApprovalPolicy;
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
    SetApprovalPolicy(Arc<dyn ApprovalPolicy>),

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

impl Command {
    /// Whether this command needs priority processing during streaming/tool execution.
    /// Urgent commands bypass any queued normal commands via a separate channel.
    pub(crate) fn is_urgent(&self) -> bool {
        matches!(self, Command::Steer(_) | Command::FollowUp(_))
    }
}
