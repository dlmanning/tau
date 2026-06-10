//! Internal command protocol between [`AgentHandle`](crate::AgentHandle) and the actor.
//!
//! # Channel contract
//!
//! Commands split into two bounded mpsc channels. This is the single
//! statement of the contract; the actor's `select!`s implement it.
//!
//! - **urgent** — `Steer`, `FollowUp` only. Capacity
//!   [`DEFAULT_URGENT_CAPACITY`](crate::core::builder::DEFAULT_URGENT_CAPACITY)
//!   (256), sized for bursts of background-subagent completions.
//! - **normal** — everything else: prompts, queries (`GetConfig`,
//!   `GetMessages`, …), config setters, manual compaction. Capacity
//!   [`DEFAULT_NORMAL_CAPACITY`](crate::core::builder::DEFAULT_NORMAL_CAPACITY)
//!   (64).
//!
//! Both capacities are overridable via
//! `AgentBuilder::with_channel_capacities`. [`Command::is_urgent`]
//! picks the channel; senders never choose directly.
//!
//! **Priority**: every actor phase that can receive commands (idle,
//! streaming, approval gates, tool execution, drain/waiting) uses a
//! `biased` `select!` that polls urgent before normal — and, in the
//! cancellable tool/drain phases, prompt-cancellation before both.
//! Consequences callers should know:
//!
//! - Steering and follow-ups are never queued behind queries or
//!   config changes.
//! - Under a *sustained* urgent burst, normal commands starve: a
//!   `GetMessages` issued mid-turn waits until the urgent channel is
//!   momentarily empty. This is accepted because steering is
//!   human-rate and completion bursts are short-lived; if a host ever
//!   observes query latency here, the planned remedy is serving
//!   read-only queries from shared state (watch snapshots) instead of
//!   the command channel — not reordering the priority.
//! - The handle's non-async setters use `try_send`; a full channel
//!   surfaces as [`Error::ChannelFull`](crate::Error) rather than
//!   blocking. The actor never drops commands it has received.

use std::sync::Arc;

use tau_ai::{Content, Message, Model, ReasoningLevel};
use tokio::sync::oneshot;

use crate::core::approval::ApprovalPolicy;
use crate::core::compaction::CompactionConfig;
use crate::core::config::AgentConfig;
use crate::types::conversation::Conversation;
use crate::types::info::{ContextStats, ToolInfo};

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
    GetContextStats(oneshot::Sender<ContextStats>),
    ListTools(oneshot::Sender<Vec<ToolInfo>>),

    // Manual compaction. The actor emits `CompactionStart` with
    // `CompactionReason::Manual` — this entry point is manual by
    // construction.
    Compact {
        custom_instructions: Option<String>,
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
