//! tau-agent: Agent runtime with tool execution
//!
//! This crate provides the agent loop that handles multi-turn conversations
//! with LLMs, including tool execution and state management.

pub mod agent;
pub mod compaction;
pub mod conversation;
pub mod error;
pub mod events;
pub mod handle;
pub mod prompts;
pub mod tool;
pub mod transport;

pub use agent::{Agent, AgentConfig, DequeueMode};
pub use compaction::{CompactionConfig, CompactionReason};
pub use conversation::{AgentState, Conversation};
pub use error::Error;
pub use events::AgentEvent;
pub use handle::AgentHandle;
pub use tool::{ProgressSender, Tool, ToolResult};
pub use transport::Transport;
