//! Single-agent runtime.
//!
//! Knows nothing about subagents or registries. Holds the actor pattern,
//! the channel-based handle, the state split (Frame / Conv / Shared),
//! sync transitions, and the I/O subsystems (transport, compaction,
//! interaction).

pub mod actor;
pub mod approval;
pub mod builder;
pub mod command;
pub mod compaction;
pub mod config;
pub mod handle;
pub mod interaction;
pub mod overflow;
pub mod state;
pub mod stream;
pub mod tool;
pub mod transitions;
pub mod transport;
