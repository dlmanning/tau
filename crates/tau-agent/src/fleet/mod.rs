//! Multi-agent management on top of `core` agents.
//!
//! The manager is a thin composition over three responsibilities:
//!
//! - [`registry`] — owns the maps; the spec/idle/running invariant is
//!   the only set of public methods, not a doc comment to maintain.
//! - [`lifecycle`] — spawn / send (resume) / respec / adopt operations.
//!   Side effects (worktrees, transcripts) live with the operations
//!   that produce them.
//! - [`bus`] — child→parent event forwarding and interaction-channel
//!   routing.
//!
//! [`AgentManager`](manager::AgentManager) holds the three.

pub mod bus;
pub mod lifecycle;
pub mod manager;
pub mod registry;
pub mod result;
pub mod snapshot;
pub mod transcript;
pub mod worktree;

use tau_ai::{Content, InjectionSource, Message};

/// Constructors for the subagent-lifecycle [`Message::SystemInjection`]
/// messages a background agent's completion posts to its parent.
///
/// Defined here rather than on `Message` in tau-ai because subagent
/// completion is a fleet concept: tau-ai owns only the data shape
/// ([`InjectionSource`]), the fleet owns the semantics (these pair
/// with `expect_follow_up()` — see
/// `transitions::is_subagent_completion`).
pub trait SubagentMessageExt {
    /// System-injection message for a subagent that completed
    /// successfully.
    fn subagent_completed(
        agent_id: impl Into<String>,
        description: impl Into<String>,
        text: impl Into<String>,
    ) -> Message;

    /// System-injection message for a subagent that failed.
    fn subagent_failed(
        agent_id: impl Into<String>,
        description: impl Into<String>,
        error: impl Into<String>,
    ) -> Message;
}

impl SubagentMessageExt for Message {
    fn subagent_completed(
        agent_id: impl Into<String>,
        description: impl Into<String>,
        text: impl Into<String>,
    ) -> Message {
        Message::SystemInjection {
            content: vec![Content::text(text)],
            source: InjectionSource::SubagentCompleted {
                agent_id: agent_id.into(),
                description: description.into(),
            },
        }
    }

    fn subagent_failed(
        agent_id: impl Into<String>,
        description: impl Into<String>,
        error: impl Into<String>,
    ) -> Message {
        Message::SystemInjection {
            content: vec![Content::text(error)],
            source: InjectionSource::SubagentFailed {
                agent_id: agent_id.into(),
                description: description.into(),
            },
        }
    }
}
