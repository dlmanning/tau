//! Internal command protocol between [`AgentHandle`] and the actor.
//!
//! Commands split into two channels:
//!
//! - **urgent** — `Steer`, `FollowUp`. Processed with priority during
//!   streaming and tool execution. The biased `select!` in the actor's
//!   busy phases drains urgent first.
//! - **normal** — everything else (prompts, queries, config setters,
//!   manual compaction).
//!
//! [`Command::is_urgent`] picks the channel.

use std::sync::Arc;

use tau_ai::{Content, Message, Model, ReasoningLevel};
use tokio::sync::oneshot;

use crate::core::approval::ApprovalPolicy;
use crate::core::compaction::CompactionConfig;
use crate::core::config::AgentConfig;
use crate::types::conversation::Conversation;
use crate::types::events::CompactionReason;
use crate::types::info::ToolInfo;

/// Result of a `Prompt` or `Compact` operation, returned via the
/// embedded oneshot.
pub struct PromptResult {
    pub result: Result<(), crate::types::error::Error>,
}

pub enum Command {
    // Prompt lifecycle
    Prompt {
        content: Vec<Content>,
        reply: oneshot::Sender<PromptResult>,
    },
    Steer(Message),
    FollowUp(Message),

    // Config (fire-and-forget)
    SetModel(Model),
    SetReasoning(ReasoningLevel),
    SetCompactionConfig(CompactionConfig),
    SetApprovalPolicy(Arc<dyn ApprovalPolicy>),

    // Queries (request-reply via oneshot)
    GetConfig(oneshot::Sender<AgentConfig>),
    GetMessages(oneshot::Sender<Vec<Message>>),
    GetState(oneshot::Sender<Conversation>),
    ListTools(oneshot::Sender<Vec<ToolInfo>>),

    // Manual compaction
    Compact {
        reason: CompactionReason,
        reply: oneshot::Sender<PromptResult>,
    },
}

impl Command {
    /// Whether this command needs priority processing during streaming
    /// or tool execution.
    pub fn is_urgent(&self) -> bool {
        matches!(self, Command::Steer(_) | Command::FollowUp(_))
    }
}
