//! Top-level session manager for `tau-agent`.
//!
//! Where `tau_agent::manager::AgentManager` runs *subagents* under one root
//! agent, this crate manages the *roots* themselves: multiple long-lived
//! conversations across restarts, each with its own metadata, persistence,
//! and lifecycle (Idle/Running/Hibernated/Closed).
//!
//! # Architecture
//!
//! - [`SessionManager`] owns the active set + the storage backend.
//! - [`SessionStorage`] is the pluggable persistence trait. The default
//!   [`storage::FsStorage`] writes one directory per session under a
//!   user-chosen root.
//! - [`SessionInfo`] is cheap metadata (used for the sidebar list).
//!   [`SessionSnapshot`] is the full restorable state (info + messages +
//!   compaction summary + opaque host UI state).
//! - On `create`/`activate`, the manager spawns a background persister
//!   that subscribes to the agent's event stream and writes incrementally
//!   (debounced) so a crash mid-session loses at most a few seconds of
//!   activity.
//!
//! Subagents are *not* persisted — hibernation aborts in-flight subagents.
//! On activation the parent's conversation contains their textual results
//! (gap #2's transcript and `inherit_history_from`).

pub mod info;
pub mod manager;
pub mod snapshot;
pub mod storage;

pub use info::{ProjectInfo, SessionId, SessionInfo, SessionStatus};
pub use manager::{
    ActiveSession, NewSessionRequest, SessionManager, SessionManagerEvent,
};
pub use snapshot::SessionSnapshot;
pub use storage::{FsStorage, SessionStorage};

/// Result alias for tau-session APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by [`SessionManager`] and storage backends.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("session not found: {0}")]
    NotFound(SessionId),
    #[error("session already exists: {0}")]
    AlreadyExists(SessionId),
    #[error("session is currently running and cannot be modified: {0}")]
    Running(SessionId),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("agent error: {0}")]
    Agent(#[from] tau_agent::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("{0}")]
    Other(String),
}
