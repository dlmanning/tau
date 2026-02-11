//! tau-agent: Agent runtime with tool execution
//!
//! This crate provides the agent loop that handles multi-turn conversations
//! with LLMs, including tool execution and state management.

pub mod agent;
pub mod events;
pub mod tool;
pub mod transport;

pub use agent::{Agent, AgentConfig};
pub use events::AgentEvent;
pub use tool::{Tool, ToolResult};
pub use transport::Transport;
