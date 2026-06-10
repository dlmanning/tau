//! Agent liveness states.
//!
//! Returned by [`AgentHandle::health`](crate::core::handle::AgentHandle::health).
//! Collapses the older `is_running` + `shutdown_reason` pair into a
//! single enum that's harder to misuse: callers can no longer ask
//! "is it alive?" and "is it running a prompt?" with two separate
//! atomics that could disagree across a phase boundary.

/// Snapshot of an agent's lifecycle state.
///
/// `Running` and `Idle` both mean "actor task is alive." `Dead` means
/// the actor task has terminated; the handle is permanently inert
/// and further commands will fail.
#[derive(Debug, Clone)]
pub enum AgentHealth {
    /// Actor is alive and currently processing a prompt (between
    /// `AgentStart` and `AgentEnd`).
    Running,
    /// Actor is alive and waiting for work. A new `prompt()` will
    /// be accepted.
    Idle,
    /// Actor task has terminated. `reason` is `Some(payload)` on
    /// panic, `None` on clean shutdown (all senders dropped).
    ///
    /// Not `PartialEq`-compared: panic payload strings carry stack
    /// addresses and timestamps and are not meaningful for equality.
    /// Use `matches!(health, AgentHealth::Dead { .. })` to test for it.
    Dead { reason: Option<String> },
}
