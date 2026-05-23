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

/// Per-agent configuration. Construct with [`AgentConfig::builder`] and
/// inspect via the accessor methods. Fields are `pub(crate)` so the
/// actor/transition layer can read/write them directly; downstream
/// callers go through the typed surface.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub(crate) system_prompt: Option<String>,
    pub(crate) model: Model,
    pub(crate) reasoning: ReasoningLevel,
    pub(crate) thinking_adaptive: bool,
    pub(crate) max_tokens: Option<u32>,
    /// `None` = unlimited. The actor still flushes a final summary turn
    /// when the limit is hit on a tool-call boundary.
    pub(crate) max_turns: Option<u32>,
    pub(crate) compaction: CompactionConfig,
    pub(crate) steering_mode: DequeueMode,
    pub(crate) follow_up_mode: DequeueMode,
    pub(crate) cache_scope: Option<String>,
    pub(crate) cache_ttl: Option<String>,
    pub(crate) system_prompt_boundary: Option<String>,
}

impl AgentConfig {
    /// Start a builder seeded with the given model. Every other field
    /// is set to its default — see [`AgentConfigBuilder`].
    pub fn builder(model: Model) -> AgentConfigBuilder {
        AgentConfigBuilder::new(model)
    }

    /// Convert this config back into a builder for further tweaks.
    /// Useful when adjusting one or two fields on top of a defaulted
    /// or shared config.
    pub fn into_builder(self) -> AgentConfigBuilder {
        AgentConfigBuilder { inner: self }
    }

    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }
    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn reasoning(&self) -> ReasoningLevel {
        self.reasoning
    }
    pub fn thinking_adaptive(&self) -> bool {
        self.thinking_adaptive
    }
    pub fn max_tokens(&self) -> Option<u32> {
        self.max_tokens
    }
    pub fn max_turns(&self) -> Option<u32> {
        self.max_turns
    }
    pub fn compaction(&self) -> &CompactionConfig {
        &self.compaction
    }
    pub fn steering_mode(&self) -> DequeueMode {
        self.steering_mode
    }
    pub fn follow_up_mode(&self) -> DequeueMode {
        self.follow_up_mode
    }
    pub fn cache_scope(&self) -> Option<&str> {
        self.cache_scope.as_deref()
    }
    pub fn cache_ttl(&self) -> Option<&str> {
        self.cache_ttl.as_deref()
    }
    pub fn system_prompt_boundary(&self) -> Option<&str> {
        self.system_prompt_boundary.as_deref()
    }
}

/// Builder for [`AgentConfig`]. Construction requires a [`Model`]; all
/// other fields default to sensible values. Anthropic-specific fields
/// (`thinking_adaptive`, `cache_scope`, `cache_ttl`,
/// `system_prompt_boundary`) are ignored by other providers.
#[derive(Debug, Clone)]
pub struct AgentConfigBuilder {
    inner: AgentConfig,
}

impl AgentConfigBuilder {
    pub fn new(model: Model) -> Self {
        Self {
            inner: AgentConfig {
                system_prompt: None,
                model,
                reasoning: ReasoningLevel::default(),
                thinking_adaptive: false,
                max_tokens: None,
                max_turns: None,
                compaction: CompactionConfig::default(),
                steering_mode: DequeueMode::All,
                follow_up_mode: DequeueMode::All,
                cache_scope: None,
                cache_ttl: None,
                system_prompt_boundary: None,
            },
        }
    }

    pub fn system_prompt(mut self, s: impl Into<String>) -> Self {
        self.inner.system_prompt = Some(s.into());
        self
    }

    pub fn model(mut self, model: Model) -> Self {
        self.inner.model = model;
        self
    }

    pub fn reasoning(mut self, level: ReasoningLevel) -> Self {
        self.inner.reasoning = level;
        self
    }

    /// Anthropic-only: let the model pick its own thinking budget per
    /// call rather than using the fixed budget from `reasoning`.
    pub fn thinking_adaptive(mut self, enabled: bool) -> Self {
        self.inner.thinking_adaptive = enabled;
        self
    }

    pub fn max_tokens(mut self, n: u32) -> Self {
        self.inner.max_tokens = Some(n);
        self
    }

    pub fn max_turns(mut self, n: u32) -> Self {
        self.inner.max_turns = Some(n);
        self
    }

    pub fn compaction(mut self, cfg: CompactionConfig) -> Self {
        self.inner.compaction = cfg;
        self
    }

    pub fn steering_mode(mut self, mode: DequeueMode) -> Self {
        self.inner.steering_mode = mode;
        self
    }

    pub fn follow_up_mode(mut self, mode: DequeueMode) -> Self {
        self.inner.follow_up_mode = mode;
        self
    }

    /// Anthropic-only prompt-cache scope (`"global"` / `"org"`).
    pub fn cache_scope(mut self, scope: impl Into<String>) -> Self {
        self.inner.cache_scope = Some(scope.into());
        self
    }

    /// Anthropic-only prompt-cache TTL (e.g. `"5m"`, `"1h"`).
    pub fn cache_ttl(mut self, ttl: impl Into<String>) -> Self {
        self.inner.cache_ttl = Some(ttl.into());
        self
    }

    /// Anthropic-only split marker for prompt caching. See
    /// [`AgentConfig::system_prompt_boundary`] for placement rules.
    pub fn system_prompt_boundary(mut self, marker: impl Into<String>) -> Self {
        self.inner.system_prompt_boundary = Some(marker.into());
        self
    }

    pub fn build(self) -> AgentConfig {
        self.inner
    }
}
