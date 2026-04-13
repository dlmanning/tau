//! Agent configuration types.
//!
//! Extracted from the old `agent.rs` into a dedicated module.

use tau_ai::{Model, ReasoningLevel};

use crate::compaction::CompactionConfig;

/// Controls how messages are drained from a queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequeueMode {
    /// Drain all pending messages at once.
    All,
    /// Drain one message at a time.
    OneAtATime,
}

/// Agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// System prompt
    pub system_prompt: Option<String>,
    /// Model to use
    pub model: Model,
    /// Reasoning/thinking level
    pub reasoning: ReasoningLevel,
    /// Use adaptive thinking (model decides when to think)
    pub thinking_adaptive: bool,
    /// Maximum tokens per response
    pub max_tokens: Option<u32>,
    /// Maximum number of turns before the agent loop stops.
    /// None means unlimited (default for the main agent).
    pub max_turns: Option<u32>,
    /// Context compaction configuration
    pub compaction: CompactionConfig,
    /// How to drain the steering queue
    pub steering_mode: DequeueMode,
    /// How to drain the follow-up queue
    pub follow_up_mode: DequeueMode,
    /// Cache scope for prompt caching ("global" or "org")
    pub cache_scope: Option<String>,
    /// Cache TTL (e.g. "1h")
    pub cache_ttl: Option<String>,
    /// Dynamic boundary marker for system prompt splitting
    pub system_prompt_boundary: Option<String>,
}
