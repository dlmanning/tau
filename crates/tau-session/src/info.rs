//! Lightweight session metadata used for the sidebar list.
//!
//! Cheap to read (one JSON file per session) so the host can render the
//! sidebar without loading full conversation histories. Updated
//! incrementally by the manager's persister as the agent runs.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tau_ai::Usage;

/// Stable identifier for a session. UUID v4.
pub type SessionId = String;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// Persisted on disk, no in-memory handle.
    Hibernated,
    /// In-memory handle exists, agent is not currently mid-prompt.
    Idle,
    /// In-memory handle exists, agent is mid-prompt.
    Running,
    /// Soft-deleted; survives in storage until evicted.
    Closed,
}

/// Project context for the session — what repo + branch was active when
/// it started. Hosts use this for the sidebar caption and to detect when
/// a session was created against a branch the user has since switched away
/// from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectInfo {
    pub name: String,
    pub path: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
}

impl ProjectInfo {
    /// Best-effort construction from a working directory: name = basename,
    /// branch + commit detected via git when available.
    pub fn from_path(path: PathBuf) -> Self {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let (branch, commit) = git_head(&path);
        Self {
            name,
            path,
            branch,
            commit,
        }
    }
}

fn git_head(path: &std::path::Path) -> (Option<String>, Option<String>) {
    let branch = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(path)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(path)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    (branch, commit)
}

/// Per-session metadata. Fits easily in a sidebar list; the full
/// conversation lives separately in `messages.jsonl` / `snapshot.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub id: SessionId,
    pub title: String,
    pub project: ProjectInfo,
    pub original_request: String,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub status: SessionStatus,
    pub message_count: usize,
    pub total_usage: Usage,
}

impl SessionInfo {
    pub fn new(
        id: SessionId,
        title: String,
        project: ProjectInfo,
        original_request: String,
    ) -> Self {
        let now = Utc::now();
        Self {
            id,
            title,
            project,
            original_request,
            created_at: now,
            last_activity: now,
            status: SessionStatus::Idle,
            message_count: 0,
            total_usage: Usage::default(),
        }
    }
}
