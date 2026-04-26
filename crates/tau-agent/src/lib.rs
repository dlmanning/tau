//! tau-agent: Actor-based agent runtime with tool execution
//!
//! This crate provides the agent loop using an actor-based architecture
//! where the agent runs as a background task and consumers interact
//! via channels (AgentHandle).

pub(crate) mod actor;
pub mod approval;
pub mod builder;
pub(crate) mod command;
pub mod compaction;
pub mod config;
pub mod context;
pub mod conversation;
pub mod error;
pub mod events;
pub mod handle;
pub mod interaction;
pub(crate) mod logic;
pub mod manager;
pub(crate) mod overflow;
pub mod plan;
pub mod prompts;
pub(crate) mod state;
pub mod stream;
pub mod tool;
pub(crate) mod tool_executor;
pub mod transcript;
pub mod transport;
pub(crate) mod worktree;

pub use approval::{
    ApprovalDecision, ApprovalPolicy, AutoAcceptAllPolicy, DefaultApprovalPolicy, RulePolicy,
    ToolApprovalOutcome, ToolRisk, ToolRule,
};
pub use builder::AgentBuilder;
pub use command::PromptResult;
pub use compaction::{CompactionConfig, CompactionReason};
pub use config::{AgentConfig, DequeueMode};
pub use conversation::Conversation;
pub use error::Error;
pub use events::AgentEvent;
pub use handle::AgentHandle;
pub use interaction::{InteractionKind, InteractionRequest, InteractionResponse, QuestionOption};
pub use plan::{Plan, PlanFile, PlanFileOp, PlanFlag, PlanFlagSeverity, PlanStep};
pub use tool::{
    BoxedTool, Concurrency, ExecutionContext, FileAccessTracker, ProgressSender, Tool, ToolResult,
};
pub use transport::Transport;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;
