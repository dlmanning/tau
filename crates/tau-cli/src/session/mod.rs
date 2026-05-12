//! Session persistence and branching.
//!
//! Layout:
//!
//! - [`store`]: file I/O, JSONL serialization, the `SessionManager`
//!   handle plus `SessionInfo` for listing.
//! - [`branch`]: free function that creates a new session pre-seeded
//!   with a prefix of an existing conversation.
//! - [`cli`]: rendering for `tau sessions ls`.

pub mod branch;
mod cli;
pub mod store;

pub(crate) use cli::list_sessions_cli;
pub use store::SessionManager;
