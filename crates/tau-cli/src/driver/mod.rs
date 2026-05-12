//! Session driver: the unified control loop that runs the agent and
//! routes events through a pluggable [`Frontend`] (stdout, TUI, …).
//!
//! Architecture:
//!
//! ```text
//!  ┌──────────────┐                 ┌────────────┐
//!  │  Session     │ ──events──▶     │  Frontend  │
//!  │  (loop +     │ ◀──input──      │  (stdout,  │
//!  │  state)      │ ◀interactions─  │   TUI)     │
//!  └──────────────┘                 └────────────┘
//! ```
//!
//! `Session` owns the agent handle, the manager, the spec resolver, and
//! the legacy persistence wrapper. It exposes typed methods
//! (`submit_prompt`, `enter_plan_mode`, …); the frontend never touches
//! `AgentHandle` directly.

mod frontend;
mod session;

pub use frontend::{Frontend, FrontendAction, SessionStart, UserInput};
pub use session::{Session, SessionConfig};
