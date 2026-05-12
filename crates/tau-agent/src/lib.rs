//! tau-agent — clean-slate rewrite of `tau-agent`.
//!
//! Architecture (strict layering, top to bottom):
//!
//! ```text
//! fleet/         — multi-agent management (registry, lifecycle, bus)
//!   ↓ depends on
//! core/          — single-agent actor: handle, state, transitions, I/O
//!   ↓ depends on
//! types/         — leaf types (events, errors, conversation, …)
//! ```
//!
//! `core` knows nothing about subagents. `fleet` composes `core` agents.
//! Code in `types` has no agent-runtime dependencies.

pub mod core;
pub mod fleet;
pub mod types;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

// ─── Common re-exports ───────────────────────────────────────────────

pub use crate::core::approval::{
    ApprovalDecision, ApprovalPolicy, AutoAcceptAll, DefaultPolicy, RulePolicy, ToolRisk, ToolRule,
};
pub use crate::core::builder::AgentBuilder;
pub use crate::core::compaction::{CompactionConfig, CompactionReason};
pub use crate::core::config::{AgentConfig, DequeueMode};
pub use crate::core::handle::AgentHandle;
pub use crate::core::interaction::{
    InteractionKind, InteractionRequest, InteractionResponse, QuestionOption,
};
pub use crate::core::tool::{
    BoxedTool, Concurrency, ExecutionContext, FileAccessTracker, ProgressSender, Tool,
    ToolCategory, ToolResult,
};
pub use crate::core::transport::{AgentEventStream, AgentRunConfig, ProviderTransport, Transport};

pub use crate::types::conversation::Conversation;
pub use crate::types::error::{Error, Result};
pub use crate::types::events::{
    AgentEvent, ConsoleLevel, ConsoleLine, SubagentOutcome, ToolApprovalOutcome,
};
pub use crate::types::info::{ContextStats, ToolInfo};

pub use crate::fleet::manager::{AgentManager, AgentSpec, AgentStatus, Isolation, SpawnOpts};
pub use crate::fleet::registry::Located;
pub use crate::fleet::result::SubagentResult;
pub use crate::fleet::snapshot::{AgentSnapshot, FleetSnapshot};
