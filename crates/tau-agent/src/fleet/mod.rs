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
pub mod transcript;
pub mod worktree;
