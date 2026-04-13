//! tau-agent: Actor-based agent runtime with tool execution
//!
//! This crate provides the agent loop using an actor-based architecture
//! where the agent runs as a background task and consumers interact
//! via channels (AgentHandle).

pub(crate) mod actor;
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
pub mod manager;
pub(crate) mod overflow;
pub mod prompts;
pub mod stream;
pub mod tool;
pub(crate) mod tool_executor;
pub mod transcript;
pub mod transport;
pub(crate) mod worktree;

pub use compaction::{CompactionConfig, CompactionReason};
pub use command::PromptResult;
pub use config::{AgentConfig, DequeueMode};
pub use conversation::Conversation;
pub use error::Error;
pub use events::AgentEvent;
pub use handle::AgentHandle;
pub use interaction::{InteractionKind, InteractionRequest, InteractionResponse, QuestionOption};
pub use tool::{
    BoxedTool, Concurrency, ExecutionContext, FileAccessTracker, ProgressSender, Tool, ToolResult,
};
pub use builder::AgentBuilder;
pub use transport::Transport;


