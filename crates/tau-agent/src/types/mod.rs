//! Leaf types with no agent-runtime dependencies.
//!
//! Everything here is data: events, errors, conversation state, the
//! command protocol between handle and actor. Modules in `core` and
//! `fleet` import from here, never the other way around.

pub mod conversation;
pub mod error;
pub mod events;
pub mod health;
pub mod info;
