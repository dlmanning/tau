//! Conversation history as a content-addressed graph, modeled after
//! git's object database.
//!
//! An agent's history is "what gets sent to the model on the next
//! API call." We organize that as a git-shaped repository: blobs
//! (opaque content), trees (named directories of blobs and subtrees),
//! and commits (tree pointers with parents). Branches are refs into
//! the commit DAG; each agent has its own branch.
//!
//! ```text
//!                                  blob: assistant message
//!                                     ▲
//!                          tree: {tools/, messages/, …}
//!                                     ▲
//!     ┌── commit (turn 1) ── commit (turn 2) ── commit (turn 3) ──┐
//!     │                                                            │
//!  branch_a ─────────────────────────────────────── tip ───────────┘
//!     │
//!     └── fork ─── commit (turn 2') ── tip(branch_b)
//! ```
//!
//! The agent runtime reads through the [`History`] trait, which
//! exposes the conversation as `messages` + `system_prompt` +
//! `tools` + `previous_summary` — exactly the four pieces that
//! shape the next API request. Hosts and the fleet work with
//! [`Branch`] directly, using its git-flavored API (`commit`,
//! `merge`, `fork`, `tip`) plus [`Repository`] for object access.
//!
//! # Cache alignment
//!
//! Two branches that share a commit prefix share the entire prompt
//! prefix the API will see — system prompt, tools, and the messages
//! up to the divergence point. That's the same property that makes
//! prompt caching work. The graph structure and the API cache
//! structure are the same structure.
//!
//! Changing tools means a new commit whose `/tools` subtree hash
//! differs from the parent's. The model's API request reads `tools`
//! from the tip commit; the previous request used a different
//! subtree hash, so the cache invalidates from tools onward. The
//! graph isn't describing the cache cost — it *is* the cache cost.
//!
//! # Status
//!
//! Pre-1.0, in-process only. Hashing is SHA-256 of `serde_json`
//! bytes, stable within a process but **not canonical across
//! versions / platforms** — two semantically-equal messages may
//! hash differently if their JSON serialization differs in key
//! order or whitespace, or if struct fields are reordered (since
//! `serde_json` emits fields in struct declaration order). Don't
//! use the hashes as cross-process identifiers yet.

mod branch;
mod history;
mod objects;
mod repository;

pub use repository::Repository;
pub use branch::{Branch, MessagesOp, TreePatch};
pub use history::{History, HistoryError};
pub use objects::{
    Blob, Commit, ObjectHash, ObjectKind, StoreError, ToolDef, Tree, TreeEntry,
};
