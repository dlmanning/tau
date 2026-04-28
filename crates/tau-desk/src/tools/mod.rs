//! Desk-state tools: agent-callable `Tool` implementations that mutate
//! desk state via shared references. Same `Tool` trait as any other tool;
//! no actor interception, no special dispatch path.
//!
//! All mutations stamp `Provenance::Agent { agent_id }` derived from the
//! caller — chat agent's calls become `agent_id: Some("chat")`,
//! per-task agents stamp their task name (`agent_id: Some("morning_scan")`).
//!
//! Each tool takes a shared handle to desk state at construction
//! (typically `Arc<DeskAgent>` or a narrower facet) so it can both write
//! to the store and emit `DeskEvent`s on the broadcast channel.

pub mod add_activity;
pub mod attach_to_card;
pub mod enqueue_draft;
pub mod move_card;
pub mod retire_card;
pub mod update_brief;
pub mod update_take;
pub mod upsert_card;
