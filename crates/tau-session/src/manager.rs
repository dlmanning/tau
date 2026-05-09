//! [`SessionManager`] — owns the active set of sessions plus a storage
//! backend and a per-session persister task.
//!
//! Lifecycle:
//!
//! 1. `create` — new id, write `info.json`, build an `AgentBuilder` with
//!    the host's tools/transport, spawn the agent, wire the persister.
//! 2. `activate` — read snapshot, build agent with `set_messages` +
//!    `set_previous_summary`, spawn, wire persister.
//! 3. `hibernate` — abort the agent, flush a snapshot, drop the handle.
//! 4. `close` — soft-delete: status flips to `Closed`, info kept.
//! 5. `evict_idle` — hibernate idle sessions older than a threshold.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use serde_json::Value;
use tau_agent::config::AgentConfig;
use tau_agent::events::AgentEvent;
use tau_agent::handle::AgentHandle;
use tau_agent::tool::BoxedTool;
use tau_agent::transport::Transport;
use tau_ai::Message;
use tokio::sync::{Mutex, broadcast};

use crate::info::{ProjectInfo, SessionId, SessionInfo, SessionStatus};
use crate::snapshot::SessionSnapshot;
use crate::storage::SessionStorage;
use crate::{Error, Result};

/// Closure that runs against the `AgentBuilder` after the manager has
/// applied its own setup (cwd, tools, seed messages) but before
/// `spawn()`. Use this to wire builder methods the request struct
/// doesn't expose directly: interaction sender, approval policy,
/// transform context, etc.
pub type CustomizeBuilder =
    Box<dyn FnOnce(&mut tau_agent::builder::AgentBuilder) + Send>;

/// Request to create a fresh session.
pub struct NewSessionRequest {
    /// `None` means derive a placeholder from `initial_prompt` (or the
    /// first user prompt sent later).
    pub title: Option<String>,
    pub project_path: PathBuf,
    /// Override the default `AgentConfig` (model, reasoning, etc.). The
    /// manager fills in tools + transport from its own configuration.
    pub config: AgentConfig,
    /// Tools the new agent gets. Cloned into the builder.
    pub tools: Vec<BoxedTool>,
    /// Transport the new agent uses.
    pub transport: Arc<dyn Transport>,
    /// Optional pre-existing message history (e.g. branched from another
    /// session).
    pub seed_messages: Vec<Message>,
    /// Optional compaction continuity for branched/seeded sessions.
    pub previous_summary: Option<String>,
    /// If set, send this prompt immediately after spawning. The session's
    /// `original_request` is populated from this.
    pub initial_prompt: Option<String>,
    /// Escape hatch for builder methods not exposed above (interaction
    /// sender for submit_plan/ask_user, approval policy override,
    /// transform_context, system_prompt overrides). The closure runs
    /// after the manager applies its own setup, just before `spawn()`.
    pub customize: Option<CustomizeBuilder>,
}

/// Returned from `create` / `activate` — the live handle plus the snapshot
/// the host needs to render the first frame.
pub struct ActiveSession {
    pub id: SessionId,
    pub handle: AgentHandle,
    pub snapshot: SessionSnapshot,
}

/// Lifecycle events broadcast to subscribers.
#[derive(Debug, Clone)]
pub enum SessionManagerEvent {
    Created { id: SessionId, info: Box<SessionInfo> },
    Activated { id: SessionId },
    Hibernated { id: SessionId },
    Closed { id: SessionId },
    Deleted { id: SessionId },
    InfoUpdated { id: SessionId, info: Box<SessionInfo> },
}

/// In-memory active session record.
struct ActiveRecord {
    handle: AgentHandle,
    /// Cancels the persister task when the session is hibernated/closed.
    persister_cancel: tokio_util::sync::CancellationToken,
    /// Joinable handle for the persister task; hibernate awaits it after
    /// cancelling so storage I/O serializes cleanly.
    persister_task: tokio::task::JoinHandle<()>,
}

pub struct SessionManager {
    storage: Arc<dyn SessionStorage>,
    active: Mutex<HashMap<SessionId, ActiveRecord>>,
    events_tx: broadcast::Sender<SessionManagerEvent>,
}

impl SessionManager {
    pub fn new(storage: Arc<dyn SessionStorage>) -> Self {
        let (events_tx, _events_rx) = broadcast::channel(64);
        Self {
            storage,
            active: Mutex::new(HashMap::new()),
            events_tx,
        }
    }

    /// Subscribe to lifecycle events.
    pub fn subscribe(&self) -> broadcast::Receiver<SessionManagerEvent> {
        self.events_tx.subscribe()
    }

    /// List every session that has metadata on disk, sorted by
    /// `last_activity` descending (most recent first).
    pub async fn list(&self) -> Result<Vec<SessionInfo>> {
        let ids = self.storage.list().await?;
        let mut infos = Vec::with_capacity(ids.len());
        for id in ids {
            match self.storage.read_info(&id).await {
                Ok(info) => infos.push(info),
                Err(e) => tracing::warn!("skipping unreadable session {id}: {e}"),
            }
        }
        infos.sort_by(|a, b| b.last_activity.cmp(&a.last_activity));
        Ok(infos)
    }

    /// Create a fresh session. Spawns the agent, wires the persister,
    /// returns the live handle + first-frame snapshot.
    pub async fn create(self: &Arc<Self>, req: NewSessionRequest) -> Result<ActiveSession> {
        let id = uuid::Uuid::new_v4().to_string();
        let project = ProjectInfo::from_path(req.project_path.clone());
        let original = req.initial_prompt.clone().unwrap_or_default();
        let title = req.title.unwrap_or_else(|| derive_title(&original));
        let info = SessionInfo::new(id.clone(), title, project, original);
        self.storage.write_info(&info).await?;

        // Build the agent.
        let mut builder =
            tau_agent::builder::AgentBuilder::new(req.config, req.transport);
        builder.set_cwd(&req.project_path);
        for tool in &req.tools {
            builder.add_tool(tool.clone());
        }
        if !req.seed_messages.is_empty() {
            builder.set_messages(req.seed_messages);
        }
        if req.previous_summary.is_some() {
            builder.set_previous_summary(req.previous_summary);
        }
        if let Some(customize) = req.customize {
            customize(&mut builder);
        }
        let handle = builder.spawn();

        // Wire persister.
        let persister_cancel = tokio_util::sync::CancellationToken::new();
        let persister_task = spawn_persister(
            self.storage.clone(),
            self.events_tx.clone(),
            id.clone(),
            handle.clone(),
            handle.subscribe(),
            persister_cancel.clone(),
        );

        // Optional initial prompt — fire and forget. The returned receiver
        // is intentionally dropped; callers that want completion should
        // use `handle.prompt_and_wait` themselves on the returned handle.
        if let Some(ref prompt) = req.initial_prompt {
            let _rx = handle.prompt(prompt).await?;
        }

        let snapshot = SessionSnapshot::new(info.clone());
        self.active.lock().await.insert(
            id.clone(),
            ActiveRecord {
                handle: handle.clone(),
                persister_cancel,
                persister_task,
            },
        );

        let _ = self.events_tx.send(SessionManagerEvent::Created {
            id: id.clone(),
            info: Box::new(info),
        });

        Ok(ActiveSession {
            id,
            handle,
            snapshot,
        })
    }

    /// Reactivate a previously-hibernated session. Reads snapshot, rebuilds
    /// the agent with the same messages + compaction summary.
    pub async fn activate(
        self: &Arc<Self>,
        id: &SessionId,
        config: AgentConfig,
        tools: Vec<BoxedTool>,
        transport: Arc<dyn Transport>,
    ) -> Result<ActiveSession> {
        // Already active? Return the live handle.
        {
            let active = self.active.lock().await;
            if let Some(rec) = active.get(id) {
                let info = self.storage.read_info(id).await?;
                let snapshot = SessionSnapshot {
                    info: info.clone(),
                    messages: self.storage.read_messages(id).await.unwrap_or_default(),
                    previous_summary: None,
                    ui_state: self.storage.read_ui_state(id).await.unwrap_or(None),
                    schema_version: crate::snapshot::CURRENT_SCHEMA_VERSION,
                };
                return Ok(ActiveSession {
                    id: id.clone(),
                    handle: rec.handle.clone(),
                    snapshot,
                });
            }
        }

        // Always prefer the live JSONL log for messages — snapshot.json is
        // only refreshed at hibernate, so after a crash mid-session it can
        // be older than the JSONL. The snapshot is consulted only for
        // compaction continuity (`previous_summary`) and as a fallback if
        // the JSONL is missing.
        let info = self.storage.read_info(id).await?;
        let stored_snapshot = self.storage.read_snapshot(id).await.ok();
        let jsonl_messages = self.storage.read_messages(id).await.unwrap_or_default();
        let messages = if !jsonl_messages.is_empty() {
            jsonl_messages
        } else {
            stored_snapshot
                .as_ref()
                .map(|s| s.messages.clone())
                .unwrap_or_default()
        };
        let previous_summary = stored_snapshot
            .as_ref()
            .and_then(|s| s.previous_summary.clone());
        let ui_state = self.storage.read_ui_state(id).await.unwrap_or(None);
        let snapshot = SessionSnapshot {
            info,
            messages,
            previous_summary,
            ui_state,
            schema_version: crate::snapshot::CURRENT_SCHEMA_VERSION,
        };

        let mut builder = tau_agent::builder::AgentBuilder::new(config, transport);
        builder.set_cwd(&snapshot.info.project.path);
        for tool in &tools {
            builder.add_tool(tool.clone());
        }
        if !snapshot.messages.is_empty() {
            builder.set_messages(snapshot.messages.clone());
        }
        if let Some(ref s) = snapshot.previous_summary {
            builder.set_previous_summary(Some(s.clone()));
        }
        let handle = builder.spawn();

        let persister_cancel = tokio_util::sync::CancellationToken::new();
        let persister_task = spawn_persister(
            self.storage.clone(),
            self.events_tx.clone(),
            id.clone(),
            handle.clone(),
            handle.subscribe(),
            persister_cancel.clone(),
        );

        // Update info.status -> Idle and persist.
        let mut info = snapshot.info.clone();
        info.status = SessionStatus::Idle;
        info.last_activity = Utc::now();
        self.storage.write_info(&info).await?;

        self.active.lock().await.insert(
            id.clone(),
            ActiveRecord {
                handle: handle.clone(),
                persister_cancel,
                persister_task,
            },
        );

        let _ = self.events_tx.send(SessionManagerEvent::Activated { id: id.clone() });

        Ok(ActiveSession {
            id: id.clone(),
            handle,
            snapshot: SessionSnapshot { info, ..snapshot },
        })
    }

    /// Live handle for an active session.
    pub async fn handle(&self, id: &SessionId) -> Option<AgentHandle> {
        self.active.lock().await.get(id).map(|r| r.handle.clone())
    }

    /// Abort the agent, flush a final snapshot, drop the handle.
    pub async fn hibernate(&self, id: &SessionId) -> Result<()> {
        let rec = self
            .active
            .lock()
            .await
            .remove(id)
            .ok_or_else(|| Error::NotFound(id.clone()))?;
        // Stop the persister BEFORE we touch storage so it can't interleave
        // writes with our snapshot. Cancel + await: the persister might be
        // mid-event when we cancel, so we wait for it to actually exit
        // (the broadcast channel close will also unblock it if needed).
        rec.persister_cancel.cancel();
        let _ = rec.persister_task.await;
        rec.handle.abort();
        let messages = rec.handle.messages().await.unwrap_or_default();
        let state = rec.handle.state().await;
        let mut info = self.storage.read_info(id).await?;
        info.status = SessionStatus::Hibernated;
        info.last_activity = Utc::now();
        info.message_count = messages.len();
        if let Some(ref s) = state {
            info.total_usage = s.total_usage.clone();
        }
        let snapshot = SessionSnapshot {
            info: info.clone(),
            messages,
            previous_summary: state.and_then(|s| s.previous_summary),
            ui_state: self.storage.read_ui_state(id).await.unwrap_or(None),
            schema_version: crate::snapshot::CURRENT_SCHEMA_VERSION,
        };
        self.storage.write_snapshot(&snapshot).await?;
        self.storage.write_info(&info).await?;
        let _ = self.events_tx.send(SessionManagerEvent::Hibernated { id: id.clone() });
        Ok(())
    }

    /// Soft-delete: status -> `Closed`, hibernate if active. Storage is
    /// retained until [`SessionManager::delete`] is called explicitly.
    pub async fn close(&self, id: &SessionId) -> Result<()> {
        if self.active.lock().await.contains_key(id) {
            self.hibernate(id).await?;
        }
        let mut info = self.storage.read_info(id).await?;
        info.status = SessionStatus::Closed;
        info.last_activity = Utc::now();
        self.storage.write_info(&info).await?;
        let _ = self.events_tx.send(SessionManagerEvent::Closed { id: id.clone() });
        Ok(())
    }

    /// Permanently remove the session's storage.
    pub async fn delete(&self, id: &SessionId) -> Result<()> {
        if self.active.lock().await.contains_key(id) {
            return Err(Error::Running(id.clone()));
        }
        self.storage.delete(id).await?;
        let _ = self.events_tx.send(SessionManagerEvent::Deleted { id: id.clone() });
        Ok(())
    }

    /// Persist host UI state (composer text, scroll, etc.).
    pub async fn save_ui_state(&self, id: &SessionId, ui: Value) -> Result<()> {
        self.storage.write_ui_state(id, &ui).await
    }

    /// Hibernate every active session whose `last_activity` is older than
    /// `older_than`. Returns the number hibernated.
    pub async fn evict_idle(&self, older_than: Duration) -> Result<u32> {
        let cutoff = Utc::now() - chrono::Duration::from_std(older_than).unwrap_or_default();
        let ids: Vec<SessionId> = self.active.lock().await.keys().cloned().collect();
        let mut count = 0u32;
        for id in ids {
            if let Ok(info) = self.storage.read_info(&id).await {
                if info.last_activity < cutoff && self.hibernate(&id).await.is_ok() {
                    count += 1;
                }
            }
        }
        Ok(count)
    }
}

/// Background task that mirrors the agent's events into incremental disk
/// updates. Cancelled when the session is hibernated/closed. Returns the
/// `JoinHandle` so callers can await actual termination — important
/// because hibernate's storage I/O must not race with the persister's.
fn spawn_persister(
    storage: Arc<dyn SessionStorage>,
    events_tx: broadcast::Sender<SessionManagerEvent>,
    id: SessionId,
    handle: AgentHandle,
    mut events: broadcast::Receiver<AgentEvent>,
    cancel: tokio_util::sync::CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => break,
                event = events.recv() => {
                    let Ok(event) = event else { break };
                    if let Err(e) =
                        handle_event(storage.as_ref(), &events_tx, &id, &handle, &event).await
                    {
                        tracing::warn!(session_id = %id, "persister error: {e}");
                    }
                }
            }
        }
    })
}

async fn handle_event(
    storage: &dyn SessionStorage,
    events_tx: &broadcast::Sender<SessionManagerEvent>,
    id: &SessionId,
    handle: &AgentHandle,
    event: &AgentEvent,
) -> Result<()> {
    match event {
        AgentEvent::MessageEnd { .. } => {
            // MessageEnd fires only for assistant messages, but we want
            // the JSONL log to be the full conversation. Rebuild from the
            // agent's authoritative state — cheap for typical sessions,
            // and lets a follow-up optimize incrementally without changing
            // the on-disk shape.
            let messages = handle.messages().await.unwrap_or_default();
            rewrite_messages(storage, id, &messages).await?;
            let mut info = storage.read_info(id).await?;
            info.message_count = messages.len();
            info.last_activity = Utc::now();
            storage.write_info(&info).await?;
            let _ = events_tx.send(SessionManagerEvent::InfoUpdated {
                id: id.clone(),
                info: Box::new(info),
            });
        }
        AgentEvent::TurnEnd { usage, .. } => {
            let mut info = storage.read_info(id).await?;
            info.total_usage.input += usage.input;
            info.total_usage.output += usage.output;
            info.total_usage.cache_read += usage.cache_read;
            info.total_usage.cache_write += usage.cache_write;
            info.total_usage.thinking += usage.thinking;
            info.last_activity = Utc::now();
            storage.write_info(&info).await?;
            let _ = events_tx.send(SessionManagerEvent::InfoUpdated {
                id: id.clone(),
                info: Box::new(info),
            });
        }
        AgentEvent::AgentStart => {
            let mut info = storage.read_info(id).await?;
            info.status = SessionStatus::Running;
            storage.write_info(&info).await?;
        }
        AgentEvent::AgentEnd { .. } => {
            let mut info = storage.read_info(id).await?;
            info.status = SessionStatus::Idle;
            storage.write_info(&info).await?;
        }
        _ => {}
    }
    Ok(())
}

/// Replace the entire JSONL log with `messages`. Used by the persister
/// because user messages don't fire events — the simplest correct option
/// is to rewrite the log from the agent's authoritative state.
///
/// Atomic: either the whole new log is visible on disk or the old one is.
/// A crash mid-write never leaves messages.jsonl in a partial state.
async fn rewrite_messages(
    storage: &dyn SessionStorage,
    id: &SessionId,
    messages: &[Message],
) -> Result<()> {
    storage.write_messages(id, messages).await
}

/// Default title from a prompt: first ~60 chars of the first non-empty
/// line, with trailing whitespace trimmed.
fn derive_title(prompt: &str) -> String {
    let first = prompt.lines().find(|l| !l.trim().is_empty()).unwrap_or("untitled");
    let mut t: String = first.chars().take(60).collect();
    if first.chars().count() > 60 {
        t.push('…');
    }
    t.trim().to_string()
}
