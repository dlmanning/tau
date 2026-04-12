//! tau-agent: Agent runtime with tool execution
//!
//! This crate provides the agent loop that handles multi-turn conversations
//! with LLMs, including tool execution and state management.

pub mod agent;
pub mod agent_manager;
pub mod compaction;
pub mod context;
pub mod conversation;
pub mod error;
pub mod events;
pub mod handle;
pub mod interaction;
pub mod loop_state;
pub(crate) mod overflow;
pub mod prompts;
pub mod stream;
pub mod tool;
pub(crate) mod tool_executor;
pub mod transcript;
pub mod transport;
pub(crate) mod worktree;

pub use agent::{Agent, AgentConfig, DequeueMode};
pub use compaction::{CompactionConfig, CompactionReason};
pub use conversation::{AgentState, Conversation};
pub use error::Error;
pub use events::AgentEvent;
pub use handle::AgentHandle;
pub use interaction::{InteractionKind, InteractionRequest, InteractionResponse, QuestionOption};
pub use tool::{
    Concurrency, ExecutionContext, FileAccessTracker, ProgressSender, Tool, ToolResult,
};
pub use transport::Transport;
