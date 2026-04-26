//! Full restorable session state.
//!
//! Written to disk on hibernate (and debounced during runtime) so the
//! session can be reactivated with the same conversation, compaction
//! continuity, and host UI state.
//!
//! The agent's restorable state is intentionally just messages +
//! previous compaction summary — both already settable on `AgentHandle`.
//! Plan, step history, and session-diff overlays aren't agent state;
//! they live in messages or host-side overlays the host rebuilds.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tau_ai::Message;

use crate::info::SessionInfo;

/// Bump when the snapshot shape changes incompatibly. Storage backends
/// should refuse to read snapshots with a higher version than they know.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub info: SessionInfo,
    pub messages: Vec<Message>,
    /// Compaction continuity: the summary the agent had when it was
    /// last hibernated, restored via `AgentHandle::set_previous_summary`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_summary: Option<String>,
    /// Host-opaque payload (composer text, scroll position, expanded
    /// blocks, etc.). The manager round-trips it but never inspects it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ui_state: Option<Value>,
    pub schema_version: u32,
}

impl SessionSnapshot {
    pub fn new(info: SessionInfo) -> Self {
        Self {
            info,
            messages: Vec::new(),
            previous_summary: None,
            ui_state: None,
            schema_version: CURRENT_SCHEMA_VERSION,
        }
    }
}
