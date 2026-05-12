//! Agent configuration.
//!
//! Held inside [`Frame`](crate::core::state::Frame) (immutable per-prompt).
//! Mutations from [`AgentHandle`](crate::core::handle::AgentHandle) update
//! it between prompts via the actor's `Idle` phase.

use tau_ai::{Model, ReasoningLevel};

use crate::core::compaction::CompactionConfig;

/// Drain mode for the steering / follow-up queues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequeueMode {
    /// Drain everything queued in one batch.
    All,
    /// Pull one message and leave the rest queued.
    OneAtATime,
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub system_prompt: Option<String>,
    pub model: Model,
    pub reasoning: ReasoningLevel,
    pub thinking_adaptive: bool,
    pub max_tokens: Option<u32>,
    /// `None` = unlimited. The actor still flushes a final summary turn
    /// when the limit is hit on a tool-call boundary.
    pub max_turns: Option<u32>,
    pub compaction: CompactionConfig,
    pub steering_mode: DequeueMode,
    pub follow_up_mode: DequeueMode,
    pub cache_scope: Option<String>,
    pub cache_ttl: Option<String>,
    pub system_prompt_boundary: Option<String>,
}
