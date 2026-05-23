use std::collections::{HashMap, VecDeque};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use tokio::sync::{broadcast, oneshot};
use tokio::task::JoinHandle;
use uuid::Uuid;

use tau_agent::PromptResult;
use tau_agent::{AgentBuilder, AgentConfig, ApprovalPolicy, FileAccessTracker, Transport};
use tau_agent::{BoxedTool, ExecutionContext, ProgressSender};
use tau_session::{ActiveSession, NewSessionRequest, SessionId, SessionManager, SessionStatus};
use tokio_util::sync::CancellationToken;

use crate::Result;
use crate::activity::{ActivityEntry, ActivityFeed, SessionSeed};
use crate::brief::Brief;
use crate::card::{CardBody, CardData, CardId, CardPile};
use crate::draft::{ActionOutcome, Draft, DraftId};
use crate::error::Error;
use crate::event::DeskEvent;
use crate::now::{NowZone, PickUpView, SuggestionView};
use crate::provenance::{CardEvent, CardEventKind, Provenance};
use crate::scheduler::{Concurrency, HydrationSpec, PromptSpec, ScheduledTask, TaskName, Trigger};
use crate::source::{ChangeNotice, Source, SourceRegistry};
use crate::storage::{CardFilter, DeskStorage};
use crate::tombstone::DismissalRecord;

/// Default cap on per-card history ring buffer.
const DEFAULT_HISTORY_CAP: usize = 32;

/// Default broadcast channel capacity for `DeskEvent`s. Lossy under
/// extreme bursts; consumers that lag past this drop and resync via
/// re-reads.
const EVENT_CHANNEL_CAP: usize = 512;

/// Title used to identify the desk's chat session in `SessionManager`'s
/// listing. Lets the desk recover its long-lived chat across restarts
/// without needing a dedicated persistence channel.
const CHAT_SESSION_TITLE: &str = "<tau-desk chat>";

/// Construction-time configuration for [`DeskAgent`].
pub struct DeskConfig {
    pub transport: Arc<dyn Transport>,
    pub storage: Arc<dyn DeskStorage>,
    pub sessions: Arc<SessionManager>,
    pub approval: Arc<dyn ApprovalPolicy>,
    pub sources: SourceRegistry,
    pub tasks: Vec<ScheduledTask>,
    /// Template for spawning per-task agents (model, reasoning, etc.).
    /// `system_prompt` is overridden per task at fire time.
    pub agent_config: AgentConfig,
    /// Concurrency cap for in-flight per-task agents. Default: 4.
    pub max_concurrent_tasks: usize,
    /// Where the desk's own scratch state lives (cache, debug archives).
    pub data_dir: PathBuf,
    /// Per-card history ring-buffer cap. Default: 32.
    pub history_cap: usize,
}

impl DeskConfig {
    pub fn new(
        transport: Arc<dyn Transport>,
        storage: Arc<dyn DeskStorage>,
        sessions: Arc<SessionManager>,
        approval: Arc<dyn ApprovalPolicy>,
        agent_config: AgentConfig,
        data_dir: PathBuf,
    ) -> Self {
        Self {
            transport,
            storage,
            sessions,
            approval,
            sources: SourceRegistry::new(),
            tasks: Vec::new(),
            agent_config,
            max_concurrent_tasks: 4,
            data_dir,
            history_cap: DEFAULT_HISTORY_CAP,
        }
    }
}

/// The ambient agent layer.
pub struct DeskAgent {
    storage: Arc<dyn DeskStorage>,
    sessions: Arc<SessionManager>,
    events: broadcast::Sender<DeskEvent>,
    history_cap: usize,
    transport: Arc<dyn Transport>,
    approval: Arc<dyn ApprovalPolicy>,
    sources: RwLock<SourceRegistry>,
    agent_config: AgentConfig,
    tasks: RwLock<Vec<ScheduledTask>>,
    data_dir: PathBuf,

    // ----- Scheduler runtime state -----
    /// Background task handles for trigger loops. Aborted on `shutdown`.
    loop_handles: Mutex<Vec<JoinHandle<()>>>,
    /// Per-task in-flight cancel tokens. Presence in the map = task is
    /// running. v1 supports a single in-flight run per task name (Skip
    /// semantics); Parallel deferred until needed.
    in_flight: Mutex<HashMap<TaskName, tokio_util::sync::CancellationToken>>,

    /// Mechanical handlers checked first by `ingest_signal`. First
    /// `handles()` match wins — that handler runs and the notice is
    /// not forwarded to OnSignal tasks. Registration order = match
    /// order.
    handlers: RwLock<Vec<Arc<dyn crate::handler::MechanicalHandler>>>,

    // ----- Chat agent (long-lived `tau-session`) -----
    /// Cached id for the desk's chat session. Discovered on `new` by
    /// matching `CHAT_SESSION_TITLE`; populated lazily on first `ask`
    /// if no prior chat exists.
    chat_session_id: RwLock<Option<SessionId>>,
    /// Serializes chat session create/activate to avoid two concurrent
    /// `ask`s spawning duplicate sessions.
    chat_create_lock: tokio::sync::Mutex<()>,

    #[allow(dead_code)]
    max_concurrent_tasks: usize,
}

impl DeskAgent {
    pub async fn new(config: DeskConfig) -> Result<Self> {
        let (events, _) = broadcast::channel(EVENT_CHANNEL_CAP);

        // Recover an existing chat session by title, if any. Errors here
        // are non-fatal — we'll lazily create on first `ask`.
        let chat_session_id = match config.sessions.list().await {
            Ok(infos) => infos
                .into_iter()
                .find(|s| s.title == CHAT_SESSION_TITLE)
                .map(|s| s.id),
            Err(_) => None,
        };

        Ok(Self {
            storage: config.storage,
            sessions: config.sessions,
            events,
            history_cap: config.history_cap,
            transport: config.transport,
            approval: config.approval,
            sources: RwLock::new(config.sources),
            agent_config: config.agent_config,
            tasks: RwLock::new(config.tasks),
            max_concurrent_tasks: config.max_concurrent_tasks,
            data_dir: config.data_dir,
            loop_handles: Mutex::new(Vec::new()),
            in_flight: Mutex::new(HashMap::new()),
            handlers: RwLock::new(Vec::new()),
            chat_session_id: RwLock::new(chat_session_id),
            chat_create_lock: tokio::sync::Mutex::new(()),
        })
    }

    /// Spawn a per-task background loop for every enabled task. Cron
    /// expressions are parsed eagerly; an invalid expression aborts
    /// startup. `Manual` triggers create no loop — they fire only via
    /// [`trigger_scan`](Self::trigger_scan).
    pub async fn start(self: &Arc<Self>) -> Result<()> {
        let tasks = self.tasks.read().clone();
        let mut handles: Vec<JoinHandle<()>> = Vec::new();

        for task in tasks {
            if !task.enabled {
                continue;
            }
            let me = Arc::clone(self);
            let handle = match &task.trigger {
                Trigger::Cron(expr) => {
                    let schedule = cron::Schedule::from_str(expr).map_err(|e| {
                        Error::Other(anyhow::anyhow!(
                            "task `{}`: invalid cron expression `{expr}`: {e}",
                            task.name
                        ))
                    })?;
                    tokio::spawn(async move { me.cron_loop(task, schedule).await })
                }
                Trigger::Interval(d) => {
                    let d = *d;
                    tokio::spawn(async move { me.interval_loop(task, d).await })
                }
                Trigger::OnSignal(source_id) => {
                    let source_id = source_id.clone();
                    let rx = self.sources.read().merged_watch();
                    tokio::spawn(async move { me.signal_loop(task, source_id, rx).await })
                }
                Trigger::Manual => continue,
            };
            handles.push(handle);
        }

        self.loop_handles.lock().extend(handles);
        Ok(())
    }

    /// Abort all background loops, cancel any in-flight per-task
    /// agents, and hibernate the chat session if active. Idempotent:
    /// safe to call multiple times.
    pub async fn shutdown(&self) -> Result<()> {
        let handles: Vec<_> = std::mem::take(&mut *self.loop_handles.lock());
        for h in &handles {
            h.abort();
        }
        let tokens: Vec<_> = self.in_flight.lock().drain().map(|(_, t)| t).collect();
        for t in tokens {
            t.cancel();
        }

        // Hibernate the chat session if it's currently active. Errors
        // are non-fatal here — shutdown should always succeed.
        let chat_id = self.chat_session_id.read().clone();
        if let Some(id) = chat_id {
            if self.sessions.handle(&id).await.is_some() {
                let _ = self.sessions.hibernate(&id).await;
            }
        }
        Ok(())
    }

    pub fn subscribe(&self) -> broadcast::Receiver<DeskEvent> {
        self.events.subscribe()
    }

    // ============================================================================
    // Reads
    // ============================================================================

    pub async fn brief(&self) -> Result<Option<Brief>> {
        self.storage.read_brief().await
    }

    pub async fn cards(&self, pile: Option<CardPile>) -> Result<Vec<CardData>> {
        self.storage
            .list_cards(CardFilter {
                pile,
                ..Default::default()
            })
            .await
    }

    pub async fn drafts(&self) -> Result<Vec<Draft>> {
        self.storage.list_drafts(None).await
    }

    pub fn activity(&self) -> Arc<dyn ActivityFeed> {
        Arc::new(StorageActivityFeed {
            storage: self.storage.clone(),
        })
    }

    pub async fn list_tombstones(&self) -> Result<Vec<DismissalRecord>> {
        self.storage.list_tombstones().await
    }

    /// Read-time projection over `tau-session` (paused sessions) and the
    /// activity feed (entries with `suggest_session.is_some()`). Filters
    /// by suggestion mutes. Not stored.
    pub async fn now_zone(&self) -> Result<NowZone> {
        let pickup = self.derive_pickup().await?;
        let suggestions = self.derive_suggestions(8).await?;
        Ok(NowZone {
            pickup,
            suggestions,
        })
    }

    async fn derive_pickup(&self) -> Result<Option<PickUpView>> {
        let sessions = self
            .sessions
            .list()
            .await
            .map_err(|e| Error::Other(anyhow::anyhow!("sessions.list: {e}")))?;

        let latest = sessions
            .into_iter()
            .filter(|s| s.status == SessionStatus::Hibernated)
            .max_by_key(|s| s.last_activity);

        Ok(latest.map(|s| PickUpView {
            session_id: s.id,
            title: s.title,
            project: Some(s.project.path),
            branch: s.project.branch,
            paused_at: s.last_activity,
            // diff_summary requires walking the session's message log
            // for `FileChanged` events (gap #4 overlay). Deferred.
            diff_summary: None,
        }))
    }

    async fn derive_suggestions(&self, limit: usize) -> Result<Vec<SuggestionView>> {
        let mutes: std::collections::HashSet<String> =
            self.storage.list_mutes().await?.into_iter().collect();

        // Walk recent activity, dedupe by `seed_from` (most recent wins).
        let mut seen_refs: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut out: Vec<SuggestionView> = Vec::new();

        for entry in self.storage.list_activity(limit * 8).await? {
            let Some(seed) = entry.suggest_session.clone() else {
                continue;
            };

            // Mute filter: a present `seed_from` ref that's muted skips.
            if let Some(ref_) = &seed.seed_from {
                if mutes.contains(ref_) {
                    continue;
                }
                if !seen_refs.insert(ref_.clone()) {
                    continue;
                }
            }

            out.push(SuggestionView {
                activity_id: entry.id,
                seed,
                at: entry.at,
            });

            if out.len() >= limit {
                break;
            }
        }

        Ok(out)
    }

    // ============================================================================
    // User mutations — stamp `Provenance::User`
    // ============================================================================

    pub async fn user_move_card(&self, id: &CardId, to: CardPile) -> Result<()> {
        if matches!(to, CardPile::Drafts) {
            return Err(Error::ManagedPile(to));
        }
        let from = self
            .storage
            .read_card(id)
            .await?
            .ok_or_else(|| Error::NotFound(id.clone()))?
            .pile;
        if matches!(from, CardPile::Drafts) {
            return Err(Error::ManagedPile(from));
        }

        self.mutate_card(
            id,
            Provenance::User,
            None,
            CardEventKind::Moved { from, to },
            |c| c.pile = to,
        )
        .await?;

        let _ = self.events.send(DeskEvent::CardMoved {
            id: id.clone(),
            from,
            to,
        });
        Ok(())
    }

    pub async fn user_retire_card(&self, id: &CardId, reason: Option<String>) -> Result<()> {
        let from = self
            .storage
            .read_card(id)
            .await?
            .ok_or_else(|| Error::NotFound(id.clone()))?
            .pile;
        if matches!(from, CardPile::Drafts) {
            return Err(Error::ManagedPile(from));
        }

        self.mutate_card(
            id,
            Provenance::User,
            reason.clone(),
            CardEventKind::Retired {
                reason: reason.clone(),
            },
            |c| c.pile = CardPile::Done,
        )
        .await?;

        let _ = self.events.send(DeskEvent::CardRetired {
            id: id.clone(),
            reason,
        });
        Ok(())
    }

    pub async fn user_dismiss_card(&self, id: &CardId, reason: Option<String>) -> Result<()> {
        let card = self
            .storage
            .read_card(id)
            .await?
            .ok_or_else(|| Error::NotFound(id.clone()))?;

        // Tombstone first (keyed by external_ref, if any).
        if let Some(ref_) = &card.external_ref {
            self.storage.add_tombstone(ref_, reason.clone()).await?;
            let _ = self.events.send(DeskEvent::CardDismissed {
                record: DismissalRecord {
                    external_ref: ref_.clone(),
                    dismissed_at: Utc::now(),
                    reason: reason.clone(),
                },
            });
        }

        // Then delete the card itself.
        self.storage.delete_card(id).await?;
        Ok(())
    }

    pub async fn user_undismiss(&self, external_ref: &str) -> Result<()> {
        let removed = self.storage.remove_tombstone(external_ref).await?;
        if removed {
            let _ = self.events.send(DeskEvent::CardUndismissed {
                external_ref: external_ref.to_string(),
            });
        }
        Ok(())
    }

    pub async fn user_pin_card(&self, id: &CardId, pinned: bool) -> Result<()> {
        let event_kind = if pinned {
            CardEventKind::Pinned
        } else {
            CardEventKind::Unpinned
        };
        self.mutate_card(id, Provenance::User, None, event_kind, |c| {
            c.pinned = pinned
        })
        .await?;

        let _ = self.events.send(DeskEvent::CardPinned {
            id: id.clone(),
            pinned,
        });
        Ok(())
    }

    pub async fn user_attach_note(&self, id: &CardId, note: String) -> Result<()> {
        let kind = "user-note".to_string();
        self.mutate_card(
            id,
            Provenance::User,
            None,
            CardEventKind::AttachmentAdded { kind: kind.clone() },
            move |c| {
                c.attachments.push(crate::card::Attachment {
                    kind: "user-note".into(),
                    url: None,
                    summary: note,
                });
            },
        )
        .await?;

        let _ = self.events.send(DeskEvent::CardAttachmentAdded {
            id: id.clone(),
            kind,
        });
        Ok(())
    }

    pub async fn user_create_note(&self, body: String, pile: CardPile) -> Result<CardId> {
        if matches!(pile, CardPile::Drafts) {
            return Err(Error::ManagedPile(pile));
        }
        let id = format!("note:{}", Uuid::new_v4());
        let now = Utc::now();
        let card = CardData {
            id: id.clone(),
            pile,
            external_ref: None,
            body: CardBody::Note { body },
            agent_take: None,
            attachments: vec![],
            metadata: serde_json::json!({}),
            pinned: false,
            created_at: now,
            last_modified: now,
            last_modified_by: Provenance::User,
            last_modified_reason: None,
            history: VecDeque::from([CardEvent {
                at: now,
                by: Provenance::User,
                kind: CardEventKind::Created,
            }]),
        };
        self.storage.upsert_card(&card).await?;
        let _ = self.events.send(DeskEvent::CardUpserted { card });
        Ok(id)
    }

    pub async fn user_edit_note(&self, id: &CardId, body: String) -> Result<()> {
        let existing = self
            .storage
            .read_card(id)
            .await?
            .ok_or_else(|| Error::NotFound(id.clone()))?;
        if !existing.body.is_user_owned() {
            return Err(Error::WrongAuthor {
                expected: crate::error::AuthorClass::User,
                actual: crate::error::AuthorClass::User,
            });
        }
        self.mutate_card(id, Provenance::User, None, CardEventKind::Updated, |c| {
            c.body = CardBody::Note { body }
        })
        .await?;
        Ok(())
    }

    pub async fn user_delete_note(&self, id: &CardId) -> Result<()> {
        let existing = self
            .storage
            .read_card(id)
            .await?
            .ok_or_else(|| Error::NotFound(id.clone()))?;
        if !existing.body.is_user_owned() {
            return Err(Error::WrongAuthor {
                expected: crate::error::AuthorClass::User,
                actual: crate::error::AuthorClass::Agent,
            });
        }
        self.storage.delete_card(id).await?;
        Ok(())
    }

    pub async fn user_mute_suggestion(&self, seed_from: &str) -> Result<()> {
        self.storage.add_mute(seed_from).await?;
        let _ = self.events.send(DeskEvent::SuggestionMuted {
            seed_from: seed_from.to_string(),
        });
        Ok(())
    }

    pub async fn user_unmute_suggestion(&self, seed_from: &str) -> Result<()> {
        if self.storage.remove_mute(seed_from).await? {
            let _ = self.events.send(DeskEvent::SuggestionUnmuted {
                seed_from: seed_from.to_string(),
            });
        }
        Ok(())
    }

    // ============================================================================
    // Drafts
    // ============================================================================

    /// Look up the named tool, dispatch with stored arguments, capture
    /// the result. Bypasses the runtime's `ApprovalPolicy` — the user
    /// already approved at the draft level.
    ///
    /// Drafts target source-write tools (`gh_post_pr_comment`, etc.); we
    /// look up by name in the source registry. The associated draft card
    /// (`card-<draft_id>`) moves from `Drafts` to `Done` with the
    /// outcome's summary. This is the only path that's allowed to move
    /// a card out of `Drafts` (the verb-matrix `move_card` rejects it).
    pub async fn approve_draft(&self, id: &DraftId) -> Result<ActionOutcome> {
        let mut draft = self
            .storage
            .read_draft(id)
            .await?
            .ok_or_else(|| Error::DraftNotFound(id.clone()))?;
        if !matches!(draft.status, crate::draft::DraftStatus::Pending) {
            return Err(Error::DraftAlreadyResolved(id.clone()));
        }

        // Look up the tool. Source registry only — desk-state tools
        // aren't dispatchable through drafts (they're agent-state
        // mutations, not externalizable actions).
        let tool = self
            .sources
            .read()
            .all_tools()
            .into_iter()
            .find(|t| t.name() == draft.tool_name)
            .ok_or_else(|| Error::UnknownTool(draft.tool_name.clone()))?;

        // Synthetic ExecutionContext. No interaction channel — if a
        // dispatched tool needs UI input, the model shouldn't have
        // queued it as a deferred draft.
        let (progress_tx, _progress_rx) =
            tokio::sync::broadcast::channel::<tau_agent::AgentEvent>(64);
        let ctx = ExecutionContext {
            cwd: self.data_dir.clone(),
            cancel: CancellationToken::new(),
            progress: ProgressSender::new(
                progress_tx,
                format!("draft:{}", draft.id),
                draft.tool_name.clone(),
            ),
            interaction: None,
            interaction_timeout: None,
            file_access: Arc::new(parking_lot::Mutex::new(FileAccessTracker::default())),
            agent_id: None,
            subagent_depth: 0,
        };

        // Dispatch. `is_error` on ToolResult signals a logical failure;
        // the ActionOutcome carries that signal forward to the user.
        let result = tool.execute(draft.arguments.clone(), ctx).await;

        let outcome = ActionOutcome {
            success: !result.is_error,
            summary: result.text_content(),
            payload: serde_json::to_value(&result).unwrap_or(serde_json::Value::Null),
            at: Utc::now(),
        };

        // Persist resolution.
        draft.status = crate::draft::DraftStatus::Approved;
        draft.resolved_at = Some(outcome.at);
        draft.outcome = Some(outcome.clone());
        self.storage.write_draft(&draft).await?;

        // Move the draft card (`card-<draft_id>`) from `Drafts` to
        // `Done`. Bypasses `move_card`'s `ManagedPile` guard — this is
        // the sanctioned exit path for the Drafts pile.
        let card_id = format!("card-{}", draft.id);
        if let Some(mut card) = self.storage.read_card(&card_id).await? {
            let now = Utc::now();
            let from = card.pile;
            card.pile = CardPile::Done;
            card.last_modified = now;
            card.last_modified_by = Provenance::User;
            card.last_modified_reason = Some(if outcome.success {
                "draft approved".into()
            } else {
                format!("draft approved (dispatch failed: {})", outcome.summary)
            });
            card.history.push_back(CardEvent {
                at: now,
                by: Provenance::User,
                kind: CardEventKind::Moved {
                    from,
                    to: CardPile::Done,
                },
            });
            while card.history.len() > self.history_cap {
                card.history.pop_front();
            }
            self.storage.upsert_card(&card).await?;
        }

        let _ = self.events.send(DeskEvent::DraftApproved {
            draft_id: draft.id.clone(),
            outcome: outcome.clone(),
        });

        Ok(outcome)
    }

    pub async fn reject_draft(&self, id: &DraftId, reason: Option<String>) -> Result<()> {
        let mut draft = self
            .storage
            .read_draft(id)
            .await?
            .ok_or_else(|| Error::DraftNotFound(id.clone()))?;
        if !matches!(draft.status, crate::draft::DraftStatus::Pending) {
            return Err(Error::DraftAlreadyResolved(id.clone()));
        }
        draft.status = crate::draft::DraftStatus::Rejected;
        draft.resolved_at = Some(Utc::now());
        self.storage.write_draft(&draft).await?;

        let _ = self.events.send(DeskEvent::DraftRejected {
            draft_id: id.clone(),
            reason,
        });
        Ok(())
    }

    // ============================================================================
    // Free-form chat
    // ============================================================================

    /// Send a prompt to the desk's long-lived chat agent. The chat
    /// session is created lazily on first call (or recovered by title
    /// across desk restarts). Returns a oneshot receiver for the
    /// completion result; live token stream available via the agent's
    /// own subscribe channel through `chat_handle()`.
    ///
    /// Conversation accumulates across calls — same session, same
    /// `AgentHandle`. Mutations the chat agent performs stamp
    /// `Provenance::Agent { agent_id: Some("chat") }`.
    pub async fn ask(&self, prompt: String) -> Result<oneshot::Receiver<PromptResult>> {
        let handle = self.chat_handle().await?;
        handle
            .prompt(&prompt)
            .await
            .map_err(|e| Error::Other(anyhow::anyhow!("chat agent: {e}")))
    }

    /// Get a live `AgentHandle` for the chat session. Creates the
    /// session if none exists, activates a hibernated one if found.
    /// Serialized via `chat_create_lock` to avoid duplicate sessions.
    async fn chat_handle(&self) -> Result<tau_agent::AgentHandle> {
        let _create_guard = self.chat_create_lock.lock().await;

        // Cached id → live handle (already-active session)?
        let cached_id = self.chat_session_id.read().clone();
        if let Some(id) = cached_id.clone() {
            if let Some(handle) = self.sessions.handle(&id).await {
                return Ok(handle);
            }
            // Session exists but is hibernated — activate it with the
            // current tool set (sources may have been added since the
            // last activation).
            let tools = self.chat_tools();
            let active = self
                .sessions
                .activate(
                    &id,
                    self.agent_config.clone(),
                    tools,
                    self.transport.clone(),
                )
                .await
                .map_err(|e| Error::Other(anyhow::anyhow!("activate chat session: {e}")))?;
            return Ok(active.handle);
        }

        // No prior chat — create a fresh session.
        let tools = self.chat_tools();
        let approval = self.approval.clone();
        let req = NewSessionRequest {
            title: Some(CHAT_SESSION_TITLE.into()),
            project_path: self.data_dir.clone(),
            config: self.agent_config.clone(),
            tools,
            transport: self.transport.clone(),
            seed_messages: vec![],
            previous_summary: None,
            initial_prompt: None,
            customize: Some(Box::new(move |builder| {
                builder.set_approval_policy(approval);
                builder.set_system_prompt(
                    "You are tau, the user's ambient desk assistant. \
                     Help them surface, summarize, or act on work items \
                     using the desk-state and source tools available. \
                     Mutations you make to the desk are stamped with \
                     agent_id `chat`.",
                );
            })),
        };

        let active = self
            .sessions
            .create(req)
            .await
            .map_err(|e| Error::Other(anyhow::anyhow!("create chat session: {e}")))?;
        *self.chat_session_id.write() = Some(active.id.clone());
        Ok(active.handle)
    }

    fn chat_tools(&self) -> Vec<BoxedTool> {
        let mut tools = self.desk_state_tools(Some("chat".into()));
        tools.extend(self.sources.read().all_tools());
        tools
    }

    /// Returns the current chat session id, if any. Useful for tests
    /// and for hosts that want to subscribe directly to the chat
    /// agent's events through the session manager.
    pub fn chat_session_id(&self) -> Option<SessionId> {
        self.chat_session_id.read().clone()
    }

    /// Reference to the underlying [`SessionManager`]. Hosts use this
    /// to subscribe to `SessionManagerEvent`s, list sessions for a
    /// sidebar UI, or operate on a specific session id (the chat
    /// session id is exposed via [`chat_session_id`](Self::chat_session_id);
    /// other session ids come from manual `start_session` /
    /// `resume_session` calls).
    pub fn sessions(&self) -> &Arc<SessionManager> {
        &self.sessions
    }

    // ============================================================================
    // Working-mode handoff to `tau-session`
    // ============================================================================

    pub async fn start_session(&self, _seed: SessionSeed) -> Result<ActiveSession> {
        todo!("session creation from seed")
    }

    pub async fn resume_session(&self, _id: &SessionId) -> Result<ActiveSession> {
        todo!("session activation")
    }

    // ============================================================================
    // Triggers
    // ============================================================================

    /// Manually fire a registered task by name. Honors the task's
    /// `Concurrency` policy (Skip / Coalesce return `Ok(())` without
    /// running if a previous fire is still in flight; Parallel always
    /// runs). Awaits completion of the per-task agent.
    pub async fn trigger_scan(&self, name: &TaskName) -> Result<()> {
        let task = self
            .tasks
            .read()
            .iter()
            .find(|t| t.name == *name)
            .cloned()
            .ok_or_else(|| {
                Error::Other(anyhow::anyhow!("no task registered with name `{name}`"))
            })?;
        self.fire_task(&task).await
    }

    /// Apply a task's concurrency policy, hydrate its prompt, and
    /// dispatch through `run_task_once`. Tracks a cancel token in
    /// `in_flight` for the duration of the run.
    async fn fire_task(&self, task: &ScheduledTask) -> Result<()> {
        // Concurrency check (v1: Skip and Coalesce both gate; Parallel
        // bypasses tracking entirely).
        match &task.concurrency {
            Concurrency::Skip | Concurrency::Coalesce { .. } => {
                if self.in_flight.lock().contains_key(&task.name) {
                    tracing::warn!(
                        task = %task.name,
                        "skipping fire — previous run still in flight"
                    );
                    return Ok(());
                }
            }
            Concurrency::Parallel => {}
        }

        let prompt = self.hydrate_prompt(&task.prompt).await?;
        let cancel = tokio_util::sync::CancellationToken::new();

        // Track in_flight only for non-Parallel modes.
        if !matches!(task.concurrency, Concurrency::Parallel) {
            self.in_flight
                .lock()
                .insert(task.name.clone(), cancel.clone());
        }

        let outcome = self.run_task_with_cancel(&task.name, prompt, cancel).await;

        if !matches!(task.concurrency, Concurrency::Parallel) {
            self.in_flight.lock().remove(&task.name);
        }
        outcome
    }

    /// Spawn a fresh per-task root agent, run a single prompt to
    /// completion, drop the handle. Each call is independent — no
    /// conversation continuity across tasks; state carries via the
    /// desk store. `agent_id` (= the task name) stamps
    /// `Provenance::Agent` on every mutation.
    pub async fn run_task_once(&self, name: &TaskName, prompt: String) -> Result<()> {
        let cancel = tokio_util::sync::CancellationToken::new();
        self.run_task_with_cancel(name, prompt, cancel).await
    }

    async fn run_task_with_cancel(
        &self,
        name: &TaskName,
        prompt: String,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<()> {
        let _ = self
            .events
            .send(DeskEvent::ScanStarted { task: name.clone() });

        let mut tools = self.desk_state_tools(Some(name.clone()));
        tools.extend(self.sources.read().all_tools());

        let mut builder = AgentBuilder::new(self.agent_config.clone(), self.transport.clone());
        builder.set_tools(tools);
        builder.set_approval_policy(self.approval.clone());

        let handle = builder
            .spawn()
            .await
            .map_err(|e| Error::Other(anyhow::anyhow!("task `{name}` spawn failed: {e}")))?;
        let prompt_fut = handle.prompt_and_wait(&prompt);

        let outcome = tokio::select! {
            r = prompt_fut => r.map_err(|e| Error::Other(anyhow::anyhow!("task `{name}` failed: {e}"))),
            _ = cancel.cancelled() => {
                handle.abort();
                Err(Error::Other(anyhow::anyhow!("task `{name}` cancelled")))
            }
        };

        match &outcome {
            Ok(_) => {
                let _ = self
                    .events
                    .send(DeskEvent::ScanCompleted { task: name.clone() });
            }
            Err(e) => {
                let _ = self.events.send(DeskEvent::ScanFailed {
                    task: name.clone(),
                    message: e.to_string(),
                });
            }
        }
        outcome?;
        Ok(())
    }

    /// Construct desk-state tool instances bound to a particular caller
    /// (`agent_id`). Each per-task spawn — and the chat agent — gets its
    /// own set, so mutations stamp the right `Provenance::Agent { agent_id }`.
    fn desk_state_tools(&self, agent_id: Option<String>) -> Vec<BoxedTool> {
        use crate::tools::{
            add_activity::AddActivityTool, attach_to_card::AttachToCardTool,
            enqueue_draft::EnqueueDraftTool, move_card::MoveCardTool, retire_card::RetireCardTool,
            update_brief::UpdateBriefTool, update_take::UpdateTakeTool,
            upsert_card::UpsertCardTool,
        };

        let storage = self.storage.clone();
        let events = self.events.clone();
        let cap = self.history_cap;

        vec![
            Arc::new(UpsertCardTool::new(
                storage.clone(),
                events.clone(),
                agent_id.clone(),
                cap,
            )),
            Arc::new(UpdateTakeTool::new(
                storage.clone(),
                events.clone(),
                agent_id.clone(),
                cap,
            )),
            Arc::new(AttachToCardTool::new(
                storage.clone(),
                events.clone(),
                agent_id.clone(),
                cap,
            )),
            Arc::new(MoveCardTool::new(
                storage.clone(),
                events.clone(),
                agent_id.clone(),
                cap,
            )),
            Arc::new(RetireCardTool::new(
                storage.clone(),
                events.clone(),
                agent_id.clone(),
                cap,
            )),
            Arc::new(EnqueueDraftTool::new(
                storage.clone(),
                events.clone(),
                agent_id,
            )),
            Arc::new(AddActivityTool::new(storage.clone(), events.clone())),
            Arc::new(UpdateBriefTool::new(storage, events)),
        ]
    }

    /// Cancel the in-flight run (if any) for the named task. The
    /// `tau-agent` actor receives the cancellation; the
    /// `run_task_with_cancel` future returns an `Error::Other`
    /// describing the cancel. No-op if nothing is running.
    pub async fn cancel_task(&self, name: &TaskName) -> Result<()> {
        if let Some(token) = self.in_flight.lock().get(name).cloned() {
            token.cancel();
        }
        Ok(())
    }

    // ============================================================================
    // Scheduler loops
    // ============================================================================

    async fn cron_loop(self: Arc<Self>, task: ScheduledTask, schedule: cron::Schedule) {
        loop {
            let Some(next) = schedule.upcoming(Utc).next() else {
                tracing::warn!(task = %task.name, "cron schedule yielded no upcoming time; loop exiting");
                return;
            };
            let wait = next
                .signed_duration_since(Utc::now())
                .to_std()
                .unwrap_or(Duration::ZERO);
            tokio::time::sleep(wait).await;
            if let Err(e) = self.fire_task(&task).await {
                tracing::warn!(task = %task.name, error = %e, "cron fire failed");
            }
        }
    }

    async fn interval_loop(self: Arc<Self>, task: ScheduledTask, period: Duration) {
        let mut tick = tokio::time::interval(period);
        // First tick fires immediately; skip it so the first fire
        // happens after one full period elapses.
        tick.tick().await;
        loop {
            tick.tick().await;
            if let Err(e) = self.fire_task(&task).await {
                tracing::warn!(task = %task.name, error = %e, "interval fire failed");
            }
        }
    }

    async fn signal_loop(
        self: Arc<Self>,
        task: ScheduledTask,
        source_id: crate::source::SourceId,
        mut rx: broadcast::Receiver<ChangeNotice>,
    ) {
        loop {
            match rx.recv().await {
                Ok(notice) if notice.source == source_id => {
                    if let Err(e) = self.fire_task(&task).await {
                        tracing::warn!(task = %task.name, error = %e, "signal fire failed");
                    }
                }
                Ok(_) => continue, // notice from a different source
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(task = %task.name, dropped = n, "signal stream lagged");
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => return,
            }
        }
    }

    // ============================================================================
    // Prompt hydration
    // ============================================================================

    async fn hydrate_prompt(&self, spec: &PromptSpec) -> Result<String> {
        match spec {
            PromptSpec::Plain(s) => Ok(s.clone()),
            PromptSpec::Hydrated { template, include } => {
                let state = self.render_state(include).await?;
                Ok(template.replace("{{state}}", &state))
            }
        }
    }

    /// Render desk state into a structured text block suitable for
    /// substitution into a hydrated prompt template via `{{state}}`.
    async fn render_state(&self, include: &HydrationSpec) -> Result<String> {
        let mut buf = String::new();

        for pile in &include.cards_in {
            let cards = self
                .storage
                .list_cards(CardFilter {
                    pile: Some(*pile),
                    ..Default::default()
                })
                .await?;
            let _ = writeln!(buf, "Cards in {pile:?} ({}):", cards.len());
            for c in cards {
                let _ = writeln!(
                    buf,
                    "- id={} body={}",
                    c.id,
                    serde_json::to_string(&c.body).unwrap_or_default()
                );
                if include.show_provenance {
                    let _ = writeln!(
                        buf,
                        "  last_modified_by={:?} at {} ({})",
                        c.last_modified_by,
                        c.last_modified.to_rfc3339(),
                        rel_time(c.last_modified)
                    );
                }
                if let Some(take) = &c.agent_take {
                    if let Some(ask) = &take.ask {
                        let _ = writeln!(buf, "  ask: {ask}");
                    }
                    if let Some(note) = &take.note {
                        let _ = writeln!(buf, "  note: {note}");
                    }
                }
                if !c.attachments.is_empty() {
                    let _ = writeln!(
                        buf,
                        "  attachments: {}",
                        serde_json::to_string(&c.attachments).unwrap_or_default()
                    );
                }
            }
            buf.push('\n');
        }

        if include.drafts {
            let drafts = self
                .storage
                .list_drafts(Some(crate::draft::DraftStatus::Pending))
                .await?;
            let _ = writeln!(buf, "Pending drafts ({}):", drafts.len());
            for d in drafts {
                let rationale = d.rationale.as_deref().unwrap_or("(no rationale)");
                let _ = writeln!(buf, "- {} via {} — {rationale}", d.id, d.tool_name);
            }
            buf.push('\n');
        }

        if include.activity_recent > 0 {
            let entries = self.storage.list_activity(include.activity_recent).await?;
            let _ = writeln!(
                buf,
                "Recent activity ({} most recent, newest first):",
                entries.len()
            );
            for e in entries {
                let _ = writeln!(
                    buf,
                    "- {} ({}): {}",
                    rel_time(e.at),
                    e.at.to_rfc3339(),
                    e.text
                );
            }
            buf.push('\n');
        }

        if include.notes {
            let all = self.storage.list_cards(CardFilter::default()).await?;
            let notes: Vec<_> = all
                .iter()
                .filter(|c| matches!(c.body, CardBody::Note { .. }))
                .collect();
            let _ = writeln!(buf, "Notes ({}):", notes.len());
            for c in notes {
                if let CardBody::Note { body } = &c.body {
                    let _ = writeln!(buf, "- {body}");
                }
            }
            buf.push('\n');
        }

        if include.brief {
            if let Some(brief) = self.storage.read_brief().await? {
                let _ = writeln!(buf, "Current brief:");
                let _ = writeln!(buf, "  greeting: {}", brief.greeting);
                let _ = writeln!(buf, "  summary: {}", brief.summary);
                buf.push('\n');
            }
        }

        // Tombstones — let the agent see what it shouldn't try to recreate.
        let tombs = self.storage.list_tombstones().await?;
        if !tombs.is_empty() {
            let _ = writeln!(buf, "Active tombstones ({}):", tombs.len());
            for t in tombs.iter().take(20) {
                let _ = writeln!(
                    buf,
                    "- {} dismissed at {} ({})",
                    t.external_ref,
                    t.dismissed_at.to_rfc3339(),
                    t.reason.as_deref().unwrap_or("no reason")
                );
            }
            buf.push('\n');
        }

        Ok(buf)
    }

    /// Webhook ingestion. Tries registered mechanical handlers first
    /// (registration-order, first-match wins). If none claim the
    /// notice, broadcasts it on the merged watch stream so any
    /// `Trigger::OnSignal(matching_source)` task fires.
    ///
    /// Hosts call this from their HTTP receiver after performing
    /// source-specific signature verification. The desk doesn't know
    /// or care about HTTP framing.
    pub async fn ingest_signal(&self, notice: ChangeNotice) -> Result<()> {
        // First-match handler dispatch. Cloning to release the lock
        // before any await; handlers themselves shouldn't need the
        // lock and we don't want them blocking other ingestions.
        let handlers: Vec<Arc<dyn crate::handler::MechanicalHandler>> =
            self.handlers.read().clone();
        for h in &handlers {
            if h.handles(&notice) {
                let ctx = crate::handler::HandlerContext {
                    storage: self.storage.clone(),
                    events: self.events.clone(),
                    source_id: notice.source.clone(),
                    history_cap: self.history_cap,
                };
                tracing::debug!(
                    handler = h.id(),
                    source = %notice.source,
                    "mechanical handler claiming notice"
                );
                return h.apply(notice, &ctx).await;
            }
        }

        // No handler claimed it — fall through to OnSignal tasks via
        // the merged watch stream.
        self.sources.read().publish(notice);
        Ok(())
    }

    /// Register a mechanical handler. Order of registration determines
    /// match priority. Call before `start()` for handlers that should
    /// race with OnSignal tasks at signal time.
    pub fn register_handler(&self, handler: Arc<dyn crate::handler::MechanicalHandler>) {
        self.handlers.write().push(handler);
    }

    // ============================================================================
    // Source registration (pre-`start`)
    // ============================================================================

    pub fn register_source(&self, source: Arc<dyn Source>) -> Result<()> {
        self.sources.write().register(source)
    }

    // ============================================================================
    // Internal helpers
    // ============================================================================

    async fn mutate_card<F>(
        &self,
        id: &CardId,
        by: Provenance,
        reason: Option<String>,
        event_kind: CardEventKind,
        mutate: F,
    ) -> Result<CardData>
    where
        F: FnOnce(&mut CardData),
    {
        crate::ops::mutate_card_with_history(
            &*self.storage,
            id,
            by,
            reason,
            event_kind,
            self.history_cap,
            mutate,
        )
        .await
    }
}

/// `ActivityFeed` impl backed by storage. Calls are synchronous on a
/// `tokio::runtime::Handle::current()` block; suitable for UIs polling
/// from sync contexts. Async callers should hit storage directly.
struct StorageActivityFeed {
    storage: Arc<dyn DeskStorage>,
}

impl ActivityFeed for StorageActivityFeed {
    fn recent(&self, limit: usize) -> Vec<ActivityEntry> {
        let s = self.storage.clone();
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async move { s.list_activity(limit).await })
                .unwrap_or_default()
        })
    }

    fn since(&self, seq: u64) -> Vec<ActivityEntry> {
        let s = self.storage.clone();
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async move { s.activity_since(seq).await })
                .unwrap_or_default()
        })
    }

    fn get(&self, id: &str) -> Option<ActivityEntry> {
        let s = self.storage.clone();
        let id = id.to_string();
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async move { s.read_activity(&id).await })
                .ok()
                .flatten()
        })
    }
}

/// Render a past `DateTime` as a short relative string ("3m ago",
/// "2h ago"). For hydration prompts; readability over precision.
fn rel_time(at: chrono::DateTime<Utc>) -> String {
    let delta = Utc::now().signed_duration_since(at);
    let secs = delta.num_seconds().max(0);
    if secs < 60 {
        format!("{secs}s ago")
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86_400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86_400)
    }
}

// AuthorClass conversion for error paths involving Provenance.
impl From<Provenance> for crate::error::AuthorClass {
    fn from(p: Provenance) -> Self {
        match p {
            Provenance::User => crate::error::AuthorClass::User,
            Provenance::Agent { .. } => crate::error::AuthorClass::Agent,
            Provenance::Source { .. } => crate::error::AuthorClass::Source,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    use tau_agent::test_utils::{MockTransport, make_test_config};
    use tau_agent::{ApprovalPolicy, DefaultPolicy};
    use tau_session::{FsStorage, SessionManager};

    use crate::activity::ActivityEntry;
    use crate::card::{CardBody, CardData, CardPile};
    use crate::storage::MemDeskStorage;

    fn pr_card(id: &str) -> CardData {
        let now = Utc::now();
        CardData {
            id: id.into(),
            pile: CardPile::NeedsYou,
            external_ref: Some(format!("github:pr/{id}")),
            body: CardBody::Pr {
                url: "https://example.test/pr".into(),
                title: "Test".into(),
                repo: "x/y".into(),
                author: "alice".into(),
                ci: None,
            },
            agent_take: None,
            attachments: vec![],
            metadata: serde_json::json!({}),
            pinned: false,
            created_at: now,
            last_modified: now,
            last_modified_by: Provenance::Agent {
                agent_id: Some("morning_scan".into()),
            },
            last_modified_reason: None,
            history: VecDeque::from([CardEvent {
                at: now,
                by: Provenance::Agent {
                    agent_id: Some("morning_scan".into()),
                },
                kind: CardEventKind::Created,
            }]),
        }
    }

    async fn make_desk() -> (DeskAgent, tempfile::TempDir) {
        make_desk_with_transport(Arc::new(MockTransport::new())).await
    }

    async fn make_desk_with_transport(
        transport: Arc<dyn tau_agent::Transport>,
    ) -> (DeskAgent, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let session_storage = Arc::new(FsStorage::new(tmp.path().to_path_buf()));
        let sessions = Arc::new(SessionManager::new(session_storage));
        let approval: Arc<dyn ApprovalPolicy> = Arc::new(DefaultPolicy);
        let storage: Arc<dyn DeskStorage> = Arc::new(MemDeskStorage::new());

        let cfg = DeskConfig::new(
            transport,
            storage,
            sessions,
            approval,
            make_test_config(),
            tmp.path().to_path_buf(),
        );
        let desk = DeskAgent::new(cfg).await.unwrap();
        (desk, tmp)
    }

    #[tokio::test]
    async fn user_move_card_updates_pile_and_history() {
        let (desk, _tmp) = make_desk().await;
        desk.storage.upsert_card(&pr_card("a")).await.unwrap();

        desk.user_move_card(&"a".into(), CardPile::Watching)
            .await
            .unwrap();

        let stored = desk.storage.read_card(&"a".into()).await.unwrap().unwrap();
        assert_eq!(stored.pile, CardPile::Watching);
        assert_eq!(stored.last_modified_by, Provenance::User);

        let last = stored.history.back().unwrap();
        assert!(matches!(
            last.kind,
            CardEventKind::Moved {
                from: CardPile::NeedsYou,
                to: CardPile::Watching
            }
        ));
        assert_eq!(last.by, Provenance::User);
    }

    #[tokio::test]
    async fn user_move_to_drafts_rejected() {
        let (desk, _tmp) = make_desk().await;
        desk.storage.upsert_card(&pr_card("a")).await.unwrap();

        let res = desk.user_move_card(&"a".into(), CardPile::Drafts).await;
        assert!(matches!(res, Err(Error::ManagedPile(CardPile::Drafts))));
    }

    #[tokio::test]
    async fn user_retire_card_moves_to_done() {
        let (desk, _tmp) = make_desk().await;
        desk.storage.upsert_card(&pr_card("a")).await.unwrap();

        desk.user_retire_card(&"a".into(), Some("done with it".into()))
            .await
            .unwrap();

        let stored = desk.storage.read_card(&"a".into()).await.unwrap().unwrap();
        assert_eq!(stored.pile, CardPile::Done);
        assert_eq!(stored.last_modified_reason.as_deref(), Some("done with it"));
    }

    #[tokio::test]
    async fn user_dismiss_card_tombstones_and_deletes() {
        let (desk, _tmp) = make_desk().await;
        let card = pr_card("a");
        let ext_ref = card.external_ref.clone().unwrap();
        desk.storage.upsert_card(&card).await.unwrap();

        desk.user_dismiss_card(&"a".into(), Some("nope".into()))
            .await
            .unwrap();

        // Card gone.
        assert!(desk.storage.read_card(&"a".into()).await.unwrap().is_none());

        // Tombstone in place.
        let tomb = desk
            .storage
            .read_tombstone(&ext_ref)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(tomb.reason.as_deref(), Some("nope"));

        // Re-upserting blocked.
        let res = desk.storage.upsert_card(&pr_card("a")).await;
        assert!(matches!(res, Err(Error::Tombstoned { .. })));
    }

    #[tokio::test]
    async fn user_undismiss_clears_tombstone() {
        let (desk, _tmp) = make_desk().await;
        desk.storage.upsert_card(&pr_card("a")).await.unwrap();
        desk.user_dismiss_card(&"a".into(), None).await.unwrap();

        let ext_ref = "github:pr/a";
        desk.user_undismiss(ext_ref).await.unwrap();

        // Now upserting again succeeds.
        desk.storage.upsert_card(&pr_card("a")).await.unwrap();
        assert!(desk.storage.read_card(&"a".into()).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn user_pin_round_trip() {
        let (desk, _tmp) = make_desk().await;
        desk.storage.upsert_card(&pr_card("a")).await.unwrap();

        desk.user_pin_card(&"a".into(), true).await.unwrap();
        assert!(
            desk.storage
                .read_card(&"a".into())
                .await
                .unwrap()
                .unwrap()
                .pinned
        );

        desk.user_pin_card(&"a".into(), false).await.unwrap();
        assert!(
            !desk
                .storage
                .read_card(&"a".into())
                .await
                .unwrap()
                .unwrap()
                .pinned
        );

        let stored = desk.storage.read_card(&"a".into()).await.unwrap().unwrap();
        let recent: Vec<_> = stored
            .history
            .iter()
            .rev()
            .take(2)
            .map(|e| std::mem::discriminant(&e.kind))
            .collect();
        assert_eq!(recent[0], std::mem::discriminant(&CardEventKind::Unpinned));
        assert_eq!(recent[1], std::mem::discriminant(&CardEventKind::Pinned));
    }

    #[tokio::test]
    async fn user_create_edit_delete_note() {
        let (desk, _tmp) = make_desk().await;

        let id = desk
            .user_create_note("don't ping David before 10am".into(), CardPile::Watching)
            .await
            .unwrap();

        let stored = desk.storage.read_card(&id).await.unwrap().unwrap();
        assert!(matches!(stored.body, CardBody::Note { .. }));
        assert_eq!(stored.last_modified_by, Provenance::User);

        desk.user_edit_note(&id, "moved to 11am".into())
            .await
            .unwrap();
        let edited = desk.storage.read_card(&id).await.unwrap().unwrap();
        match &edited.body {
            CardBody::Note { body } => assert_eq!(body, "moved to 11am"),
            _ => panic!("expected Note"),
        }

        desk.user_delete_note(&id).await.unwrap();
        assert!(desk.storage.read_card(&id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn user_edit_note_rejects_non_note() {
        let (desk, _tmp) = make_desk().await;
        desk.storage.upsert_card(&pr_card("a")).await.unwrap();

        let res = desk.user_edit_note(&"a".into(), "hi".into()).await;
        assert!(matches!(res, Err(Error::WrongAuthor { .. })));
    }

    #[tokio::test]
    async fn user_create_note_rejects_drafts_pile() {
        let (desk, _tmp) = make_desk().await;
        let res = desk.user_create_note("x".into(), CardPile::Drafts).await;
        assert!(matches!(res, Err(Error::ManagedPile(CardPile::Drafts))));
    }

    #[tokio::test]
    async fn user_attach_note_adds_attachment_and_history() {
        let (desk, _tmp) = make_desk().await;
        desk.storage.upsert_card(&pr_card("a")).await.unwrap();

        desk.user_attach_note(&"a".into(), "ping Maya re KubeCon".into())
            .await
            .unwrap();

        let stored = desk.storage.read_card(&"a".into()).await.unwrap().unwrap();
        assert_eq!(stored.attachments.len(), 1);
        let att = &stored.attachments[0];
        assert_eq!(att.kind, "user-note");
        assert_eq!(att.summary, "ping Maya re KubeCon");
        assert!(att.url.is_none());

        let last = stored.history.back().unwrap();
        assert!(matches!(
            &last.kind,
            CardEventKind::AttachmentAdded { kind } if kind == "user-note"
        ));
        assert_eq!(last.by, Provenance::User);
    }

    #[tokio::test]
    async fn user_attach_note_works_on_note_cards() {
        let (desk, _tmp) = make_desk().await;
        let id = desk
            .user_create_note("standup at 10:30".into(), CardPile::Watching)
            .await
            .unwrap();

        // Notes can also receive attachments — the field is on CardData,
        // not CardBody::Pr/Jira specifically.
        desk.user_attach_note(&id, "moved to 11 next week".into())
            .await
            .unwrap();

        let stored = desk.storage.read_card(&id).await.unwrap().unwrap();
        assert_eq!(stored.attachments.len(), 1);
    }

    #[tokio::test]
    async fn user_attach_note_missing_card_errors() {
        let (desk, _tmp) = make_desk().await;
        let res = desk.user_attach_note(&"missing".into(), "x".into()).await;
        assert!(matches!(res, Err(Error::NotFound(_))));
    }

    #[tokio::test]
    async fn mute_lifecycle() {
        let (desk, _tmp) = make_desk().await;
        desk.user_mute_suggestion("jira:PLT-312").await.unwrap();
        assert_eq!(
            desk.storage.list_mutes().await.unwrap(),
            vec!["jira:PLT-312"]
        );

        desk.user_unmute_suggestion("jira:PLT-312").await.unwrap();
        assert!(desk.storage.list_mutes().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn history_ring_buffer_caps() {
        let (desk, _tmp) = make_desk().await;
        let mut card = pr_card("a");
        card.history.clear();
        desk.storage.upsert_card(&card).await.unwrap();

        // Push more events than the cap; oldest should fall off.
        for i in 0..(DEFAULT_HISTORY_CAP + 5) {
            let to = if i % 2 == 0 {
                CardPile::Watching
            } else {
                CardPile::NeedsYou
            };
            desk.user_move_card(&"a".into(), to).await.unwrap();
        }
        let stored = desk.storage.read_card(&"a".into()).await.unwrap().unwrap();
        assert_eq!(stored.history.len(), DEFAULT_HISTORY_CAP);
    }

    #[tokio::test]
    async fn now_zone_derives_suggestions_from_activity() {
        let (desk, _tmp) = make_desk().await;

        // Two activity entries with seed_from refs, one without.
        for (id, seed_from) in [
            ("a1", Some("jira:PLT-312")),
            ("a2", None),
            ("a3", Some("jira:PLT-313")),
        ] {
            let entry = ActivityEntry {
                id: id.into(),
                seq: 0,
                at: Utc::now(),
                text: "x".into(),
                kind: None,
                suggest_session: Some(SessionSeed {
                    title: "spike".into(),
                    project: None,
                    branch: None,
                    kickoff: "k".into(),
                    seed_from: seed_from.map(String::from),
                }),
            };
            desk.storage.append_activity(&entry).await.unwrap();
        }

        let zone = desk.now_zone().await.unwrap();
        assert_eq!(zone.suggestions.len(), 3);

        // Mute one.
        desk.user_mute_suggestion("jira:PLT-312").await.unwrap();
        let zone = desk.now_zone().await.unwrap();
        assert_eq!(zone.suggestions.len(), 2);
        assert!(
            !zone
                .suggestions
                .iter()
                .any(|s| s.seed.seed_from.as_deref() == Some("jira:PLT-312"))
        );
    }

    #[tokio::test]
    async fn now_zone_pickup_none_when_no_hibernated_sessions() {
        let (desk, _tmp) = make_desk().await;
        let zone = desk.now_zone().await.unwrap();
        assert!(zone.pickup.is_none());
    }

    #[tokio::test]
    async fn subscribe_receives_events() {
        let (desk, _tmp) = make_desk().await;
        let mut rx = desk.subscribe();

        desk.storage.upsert_card(&pr_card("a")).await.unwrap();
        desk.user_move_card(&"a".into(), CardPile::Watching)
            .await
            .unwrap();

        // First non-trivial event we sent.
        let event = tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv())
            .await
            .expect("event timeout")
            .expect("recv");
        assert!(matches!(event, DeskEvent::CardMoved { .. }));
    }

    // ============================================================
    // Draft dispatch
    // ============================================================

    /// A canned "fake gh_post_pr_comment" tool we register on the desk's
    /// source registry to validate draft dispatch end-to-end. Records
    /// every call it sees so the test can assert it ran with the right
    /// arguments.
    struct FakePostPrCommentTool {
        calls: Arc<parking_lot::Mutex<Vec<serde_json::Value>>>,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl tau_agent::Tool for FakePostPrCommentTool {
        fn name(&self) -> &str {
            "gh_post_pr_comment"
        }
        fn description(&self) -> &str {
            "Post a comment on a GitHub PR (test stub)."
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(
            &self,
            arguments: serde_json::Value,
            _ctx: tau_agent::ExecutionContext,
        ) -> tau_agent::ToolResult {
            self.calls.lock().push(arguments);
            if self.fail {
                tau_agent::ToolResult::error("network down")
            } else {
                tau_agent::ToolResult::text("comment posted: gh.example/c/42")
            }
        }
    }

    struct FakeGhSource {
        tool: Arc<FakePostPrCommentTool>,
    }

    #[async_trait::async_trait]
    impl Source for FakeGhSource {
        fn id(&self) -> &str {
            "gh"
        }
        fn tools(&self) -> Vec<tau_agent::BoxedTool> {
            vec![self.tool.clone()]
        }
    }

    async fn make_desk_with_fake_gh(
        fail: bool,
    ) -> (DeskAgent, Arc<FakePostPrCommentTool>, tempfile::TempDir) {
        let (desk, tmp) = make_desk().await;
        let tool = Arc::new(FakePostPrCommentTool {
            calls: Arc::new(parking_lot::Mutex::new(Vec::new())),
            fail,
        });
        desk.register_source(Arc::new(FakeGhSource { tool: tool.clone() }))
            .unwrap();
        (desk, tool, tmp)
    }

    /// Helper: enqueue a draft directly through storage so the
    /// approve-side tests don't have to go through the agent loop.
    async fn enqueue_test_draft(
        desk: &DeskAgent,
        tool_name: &str,
        args: serde_json::Value,
    ) -> String {
        let draft_id = format!("draft:test-{}", Uuid::new_v4());
        let now = Utc::now();
        let by = Provenance::Agent {
            agent_id: Some("test".into()),
        };
        let draft = crate::draft::Draft {
            id: draft_id.clone(),
            source_id: tool_name.split_once('_').map(|(p, _)| p.to_string()),
            tool_name: tool_name.into(),
            arguments: args,
            rationale: None,
            status: crate::draft::DraftStatus::Pending,
            created_at: now,
            resolved_at: None,
            outcome: None,
        };
        desk.storage.write_draft(&draft).await.unwrap();

        let card_id = format!("card-{draft_id}");
        let card = CardData {
            id: card_id,
            pile: CardPile::Drafts,
            external_ref: None,
            body: CardBody::Draft {
                draft_id: draft_id.clone(),
                summary: "test draft".into(),
            },
            agent_take: None,
            attachments: vec![],
            metadata: serde_json::json!({}),
            pinned: false,
            created_at: now,
            last_modified: now,
            last_modified_by: by.clone(),
            last_modified_reason: None,
            history: VecDeque::from([CardEvent {
                at: now,
                by,
                kind: CardEventKind::Created,
            }]),
        };
        desk.storage.upsert_card(&card).await.unwrap();
        draft_id
    }

    #[tokio::test]
    async fn approve_draft_dispatches_tool_and_moves_card_to_done() {
        let (desk, fake, _tmp) = make_desk_with_fake_gh(false).await;
        let mut events = desk.subscribe();

        let args = serde_json::json!({ "pr": 4821, "body": "lgtm" });
        let draft_id = enqueue_test_draft(&desk, "gh_post_pr_comment", args.clone()).await;

        let outcome = desk.approve_draft(&draft_id).await.unwrap();
        assert!(outcome.success);
        assert!(outcome.summary.contains("comment posted"));

        // Tool was invoked once with the stored arguments.
        let calls = fake.calls.lock().clone();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0], args);

        // Draft row is Approved with outcome captured.
        let stored = desk.storage.read_draft(&draft_id).await.unwrap().unwrap();
        assert_eq!(stored.status, crate::draft::DraftStatus::Approved);
        assert!(stored.resolved_at.is_some());
        assert!(stored.outcome.is_some());

        // Draft card moved Drafts → Done.
        let card = desk
            .storage
            .read_card(&format!("card-{draft_id}"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(card.pile, CardPile::Done);
        let last = card.history.back().unwrap();
        assert!(matches!(
            last.kind,
            CardEventKind::Moved {
                from: CardPile::Drafts,
                to: CardPile::Done
            }
        ));

        // DraftApproved event fired.
        let mut saw = false;
        for _ in 0..10 {
            match tokio::time::timeout(std::time::Duration::from_millis(50), events.recv()).await {
                Ok(Ok(DeskEvent::DraftApproved {
                    draft_id: id,
                    outcome,
                })) => {
                    assert_eq!(id, draft_id);
                    assert!(outcome.success);
                    saw = true;
                    break;
                }
                Ok(Ok(_)) => continue,
                Ok(Err(_)) | Err(_) => break,
            }
        }
        assert!(saw, "expected DraftApproved event");
    }

    #[tokio::test]
    async fn approve_draft_records_dispatch_failure_in_outcome() {
        let (desk, _fake, _tmp) = make_desk_with_fake_gh(true).await;

        let draft_id = enqueue_test_draft(
            &desk,
            "gh_post_pr_comment",
            serde_json::json!({ "pr": 4821, "body": "lgtm" }),
        )
        .await;

        let outcome = desk.approve_draft(&draft_id).await.unwrap();
        assert!(!outcome.success, "tool error → outcome.success = false");
        assert!(outcome.summary.contains("network down"));

        // Draft is still marked Approved (we *did* dispatch — just failed).
        // The user's decision to approve is final; the failure is informational.
        let stored = desk.storage.read_draft(&draft_id).await.unwrap().unwrap();
        assert_eq!(stored.status, crate::draft::DraftStatus::Approved);
        assert!(!stored.outcome.unwrap().success);

        // Card still moves to Done; reason notes the failure.
        let card = desk
            .storage
            .read_card(&format!("card-{draft_id}"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(card.pile, CardPile::Done);
        assert!(
            card.last_modified_reason
                .unwrap()
                .contains("dispatch failed")
        );
    }

    #[tokio::test]
    async fn approve_draft_rejects_unknown_tool() {
        let (desk, _tmp) = make_desk().await;
        let draft_id = enqueue_test_draft(&desk, "nonexistent_tool", serde_json::json!({})).await;

        let res = desk.approve_draft(&draft_id).await;
        assert!(matches!(res, Err(Error::UnknownTool(_))));

        // Draft and card unchanged on lookup failure.
        let stored = desk.storage.read_draft(&draft_id).await.unwrap().unwrap();
        assert_eq!(stored.status, crate::draft::DraftStatus::Pending);
        let card = desk
            .storage
            .read_card(&format!("card-{draft_id}"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(card.pile, CardPile::Drafts);
    }

    #[tokio::test]
    async fn approve_draft_rejects_already_resolved() {
        let (desk, _fake, _tmp) = make_desk_with_fake_gh(false).await;
        let draft_id = enqueue_test_draft(
            &desk,
            "gh_post_pr_comment",
            serde_json::json!({ "pr": 1, "body": "ok" }),
        )
        .await;

        // First approval succeeds.
        desk.approve_draft(&draft_id).await.unwrap();

        // Second one rejects.
        let res = desk.approve_draft(&draft_id).await;
        assert!(matches!(res, Err(Error::DraftAlreadyResolved(_))));
    }

    /// End-to-end slice: agent calls `desk_upsert_card` with a PR
    /// body, then `desk_update_take` to revise commentary, then
    /// `desk_update_brief`. All three should hit storage and emit
    /// the expected events with `Provenance::Agent { agent_id }`.
    #[tokio::test]
    async fn run_task_once_dispatches_card_mutations_via_agent() {
        let body = serde_json::json!({
            "kind": "pr",
            "url":   "https://github.com/x/y/pull/4821",
            "title": "Switch payments SDK",
            "repo":  "x/y",
            "author": "priya.s",
            "ci":    "passing"
        });

        let transport = MockTransport::new()
            .with_tool_call_response(
                "desk_upsert_card",
                "u1",
                serde_json::json!({
                    "id": "github:pr/4821",
                    "external_ref": "github:pr/4821",
                    "pile": "needs_you",
                    "body": body,
                }),
            )
            .with_tool_call_response(
                "desk_update_take",
                "u2",
                serde_json::json!({
                    "card_id": "github:pr/4821",
                    "ask":  "review — Priya is blocked on a deploy window",
                    "note": "one risky change in refund_test.rs worth flagging"
                }),
            )
            .with_tool_call_response(
                "desk_update_brief",
                "u3",
                serde_json::json!({
                    "greeting": "Good morning, Alex.",
                    "summary":  "Quiet overnight. One PR needs your eyes.",
                    "stats": [
                        { "label": "Review queue", "value": "1", "delta": null }
                    ]
                }),
            )
            .with_text_response("morning scan complete");

        let (desk, _tmp) = make_desk_with_transport(Arc::new(transport)).await;
        desk.run_task_once(&"morning_scan".to_string(), "Run the morning scan".into())
            .await
            .expect("task should complete");

        let card = desk
            .storage
            .read_card(&"github:pr/4821".into())
            .await
            .unwrap()
            .expect("card upserted");
        assert_eq!(card.pile, CardPile::NeedsYou);
        assert_eq!(
            card.last_modified_by,
            Provenance::Agent {
                agent_id: Some("morning_scan".into()),
            }
        );
        let take = card.agent_take.expect("take present");
        assert_eq!(
            take.ask.as_deref(),
            Some("review — Priya is blocked on a deploy window")
        );
        assert!(take.note.as_deref().unwrap().contains("refund_test"));

        // History: Created → Updated (from the body re-upsert if any) → TakeUpdated.
        let last = card.history.back().unwrap();
        assert!(matches!(last.kind, CardEventKind::TakeUpdated));

        let brief = desk.storage.read_brief().await.unwrap().expect("brief");
        assert_eq!(brief.greeting, "Good morning, Alex.");
        assert_eq!(brief.stats.len(), 1);
    }

    /// Agent enqueues a draft → desk persists the draft row and
    /// publishes a card with `body: CardBody::Draft` into the Drafts pile.
    #[tokio::test]
    async fn run_task_once_dispatches_enqueue_draft() {
        let transport = MockTransport::new()
            .with_tool_call_response(
                "desk_enqueue_draft",
                "d1",
                serde_json::json!({
                    "tool_name": "gh_post_pr_comment",
                    "arguments": { "pr": 4821, "body": "lgtm" },
                    "rationale": "Priya is waiting; you've approved similar PRs.",
                    "summary":   "Comment on PR #4821"
                }),
            )
            .with_text_response("done");

        let (desk, _tmp) = make_desk_with_transport(Arc::new(transport)).await;
        desk.run_task_once(&"morning_scan".to_string(), "Draft a comment".into())
            .await
            .unwrap();

        let drafts = desk.storage.list_drafts(None).await.unwrap();
        assert_eq!(drafts.len(), 1);
        let d = &drafts[0];
        assert_eq!(d.tool_name, "gh_post_pr_comment");
        assert_eq!(d.source_id.as_deref(), Some("gh"));
        assert_eq!(d.status, crate::draft::DraftStatus::Pending);

        // A draft card lands in the Drafts pile.
        let cards = desk
            .storage
            .list_cards(crate::storage::CardFilter {
                pile: Some(CardPile::Drafts),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(cards.len(), 1);
        assert!(matches!(cards[0].body, CardBody::Draft { .. }));
    }

    /// Tombstone is enforced inside the tool path — agent that tries
    /// to upsert a dismissed external_ref gets an error result it can
    /// react to (typically falls back to add_activity per its prompt).
    #[tokio::test]
    async fn run_task_once_upsert_blocked_by_tombstone() {
        let body = serde_json::json!({
            "kind": "pr", "url": "https://x/pr/1", "title": "T",
            "repo": "x/y", "author": "a", "ci": null
        });
        let transport = MockTransport::new()
            .with_tool_call_response(
                "desk_upsert_card",
                "u1",
                serde_json::json!({
                    "id": "github:pr/1",
                    "external_ref": "github:pr/1",
                    "pile": "needs_you",
                    "body": body
                }),
            )
            .with_text_response("ok, falling back");

        let (desk, _tmp) = make_desk_with_transport(Arc::new(transport)).await;
        desk.storage
            .add_tombstone("github:pr/1", Some("user dismissed".into()))
            .await
            .unwrap();

        desk.run_task_once(&"morning_scan".to_string(), "Run".into())
            .await
            .unwrap();

        // Card was not created.
        assert!(
            desk.storage
                .read_card(&"github:pr/1".into())
                .await
                .unwrap()
                .is_none()
        );
    }

    // ============================================================
    // Signal ingestion + mechanical handlers
    // ============================================================

    /// A handler that flips a card's pile to Done when it sees a
    /// "merged" notice. Records every call for inspection.
    struct CiMergedHandler {
        calls: Arc<parking_lot::Mutex<Vec<ChangeNotice>>>,
    }

    #[async_trait::async_trait]
    impl crate::handler::MechanicalHandler for CiMergedHandler {
        fn id(&self) -> &str {
            "ci_merged"
        }
        fn handles(&self, notice: &ChangeNotice) -> bool {
            notice.source == "gh"
                && notice.context.get("event").and_then(|v| v.as_str()) == Some("pr_merged")
        }
        async fn apply(
            &self,
            notice: ChangeNotice,
            ctx: &crate::handler::HandlerContext,
        ) -> Result<()> {
            self.calls.lock().push(notice.clone());
            let card_id = notice
                .context
                .get("card_id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Other(anyhow::anyhow!("notice missing card_id")))?
                .to_string();
            ctx.mutate_card(
                &card_id,
                CardEventKind::Moved {
                    from: CardPile::NeedsYou,
                    to: CardPile::Done,
                },
                Some("PR merged".into()),
                |c| c.pile = CardPile::Done,
            )
            .await?;
            Ok(())
        }
    }

    fn pr_card_local(id: &str) -> CardData {
        let now = Utc::now();
        CardData {
            id: id.into(),
            pile: CardPile::NeedsYou,
            external_ref: Some(format!("github:pr/{id}")),
            body: CardBody::Pr {
                url: "https://x".into(),
                title: "T".into(),
                repo: "x/y".into(),
                author: "a".into(),
                ci: None,
            },
            agent_take: None,
            attachments: vec![],
            metadata: serde_json::json!({}),
            pinned: false,
            created_at: now,
            last_modified: now,
            last_modified_by: Provenance::Agent {
                agent_id: Some("morning_scan".into()),
            },
            last_modified_reason: None,
            history: VecDeque::from([CardEvent {
                at: now,
                by: Provenance::Agent {
                    agent_id: Some("morning_scan".into()),
                },
                kind: CardEventKind::Created,
            }]),
        }
    }

    #[tokio::test]
    async fn ingest_signal_claimed_by_mechanical_handler() {
        let (desk, _tmp) = make_desk().await;
        desk.storage
            .upsert_card(&pr_card_local("4821"))
            .await
            .unwrap();

        let calls = Arc::new(parking_lot::Mutex::new(Vec::new()));
        desk.register_handler(Arc::new(CiMergedHandler {
            calls: calls.clone(),
        }));

        desk.ingest_signal(ChangeNotice {
            source: "gh".into(),
            summary: "PR #4821 merged".into(),
            context: serde_json::json!({
                "event": "pr_merged",
                "card_id": "4821"
            }),
        })
        .await
        .unwrap();

        // Handler ran exactly once.
        assert_eq!(calls.lock().len(), 1);

        // Card moved to Done with Source provenance.
        let card = desk
            .storage
            .read_card(&"4821".into())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(card.pile, CardPile::Done);
        assert_eq!(
            card.last_modified_by,
            Provenance::Source {
                source_id: "gh".into()
            }
        );
        let last = card.history.back().unwrap();
        assert!(matches!(
            last.kind,
            CardEventKind::Moved {
                to: CardPile::Done,
                ..
            }
        ));
        assert!(matches!(last.by, Provenance::Source { .. }));
    }

    #[tokio::test]
    async fn ingest_signal_falls_through_when_no_handler_matches() {
        // No handler registered. The notice should hit the merged
        // watch channel; an OnSignal task can subscribe and fire.
        // Here we directly subscribe to confirm the notice broadcast.
        let (desk, _tmp) = make_desk().await;
        let mut rx = desk.sources.read().merged_watch();

        let notice = ChangeNotice {
            source: "jira".into(),
            summary: "issue updated".into(),
            context: serde_json::json!({}),
        };
        desk.ingest_signal(notice.clone()).await.unwrap();

        let received = tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv())
            .await
            .expect("notice should broadcast")
            .expect("recv");
        assert_eq!(received.source, "jira");
        assert_eq!(received.summary, "issue updated");
    }

    #[tokio::test]
    async fn handler_match_suppresses_onsignal_task() {
        // A registered handler claims a notice; an OnSignal task for
        // the same source should NOT fire because the notice never
        // reaches the merged stream.
        let calls: Arc<parking_lot::Mutex<Vec<ChangeNotice>>> =
            Arc::new(parking_lot::Mutex::new(Vec::new()));
        let task = ScheduledTask {
            id: "should_not_fire".into(),
            name: "should_not_fire".into(),
            trigger: Trigger::OnSignal("ci".into()),
            concurrency: Concurrency::Skip,
            // Empty MockTransport — would panic if dispatched.
            prompt: PromptSpec::Plain("never runs".into()),
            enabled: true,
        };

        // Build desk with the OnSignal task and an empty MockTransport.
        let tmp = tempfile::tempdir().unwrap();
        let session_storage = Arc::new(FsStorage::new(tmp.path().to_path_buf()));
        let sessions = Arc::new(SessionManager::new(session_storage));
        let approval: Arc<dyn ApprovalPolicy> = Arc::new(DefaultPolicy);
        let storage: Arc<dyn DeskStorage> = Arc::new(MemDeskStorage::new());
        let mut cfg = DeskConfig::new(
            Arc::new(MockTransport::new()),
            storage.clone(),
            sessions,
            approval,
            make_test_config(),
            tmp.path().to_path_buf(),
        );
        cfg.tasks = vec![task];
        let desk = Arc::new(DeskAgent::new(cfg).await.unwrap());

        // Register a handler that claims any "ci" notice.
        struct ClaimsCi {
            calls: Arc<parking_lot::Mutex<u32>>,
        }
        #[async_trait::async_trait]
        impl crate::handler::MechanicalHandler for ClaimsCi {
            fn handles(&self, notice: &ChangeNotice) -> bool {
                notice.source == "ci"
            }
            async fn apply(
                &self,
                _notice: ChangeNotice,
                _ctx: &crate::handler::HandlerContext,
            ) -> Result<()> {
                *self.calls.lock() += 1;
                Ok(())
            }
        }
        let claim_count = Arc::new(parking_lot::Mutex::new(0u32));
        desk.register_handler(Arc::new(ClaimsCi {
            calls: claim_count.clone(),
        }));

        desk.start().await.unwrap();

        desk.ingest_signal(ChangeNotice {
            source: "ci".into(),
            summary: "build green".into(),
            context: serde_json::json!({}),
        })
        .await
        .unwrap();

        // Give the merged-watch consumer a chance to (not) fire.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        assert_eq!(*claim_count.lock(), 1, "handler should have run");
        // No activity — task didn't fire (which would have panicked
        // on the empty MockTransport anyway).
        assert!(storage.list_activity(10).await.unwrap().is_empty());
        let _ = calls; // suppress unused warning

        desk.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn first_handler_wins() {
        // Register two handlers that both claim the same notice.
        // Only the first should run.
        let (desk, _tmp) = make_desk().await;

        struct CountingHandler {
            label: &'static str,
            calls: Arc<parking_lot::Mutex<u32>>,
        }
        #[async_trait::async_trait]
        impl crate::handler::MechanicalHandler for CountingHandler {
            fn id(&self) -> &str {
                self.label
            }
            fn handles(&self, _notice: &ChangeNotice) -> bool {
                true
            }
            async fn apply(
                &self,
                _: ChangeNotice,
                _: &crate::handler::HandlerContext,
            ) -> Result<()> {
                *self.calls.lock() += 1;
                Ok(())
            }
        }

        let first = Arc::new(parking_lot::Mutex::new(0u32));
        let second = Arc::new(parking_lot::Mutex::new(0u32));
        desk.register_handler(Arc::new(CountingHandler {
            label: "first",
            calls: first.clone(),
        }));
        desk.register_handler(Arc::new(CountingHandler {
            label: "second",
            calls: second.clone(),
        }));

        desk.ingest_signal(ChangeNotice {
            source: "anything".into(),
            summary: "x".into(),
            context: serde_json::json!({}),
        })
        .await
        .unwrap();

        assert_eq!(*first.lock(), 1);
        assert_eq!(*second.lock(), 0, "later handlers don't run");
    }

    #[tokio::test]
    async fn handler_error_propagates() {
        let (desk, _tmp) = make_desk().await;

        struct FailingHandler;
        #[async_trait::async_trait]
        impl crate::handler::MechanicalHandler for FailingHandler {
            fn handles(&self, _: &ChangeNotice) -> bool {
                true
            }
            async fn apply(
                &self,
                _: ChangeNotice,
                _: &crate::handler::HandlerContext,
            ) -> Result<()> {
                Err(Error::Other(anyhow::anyhow!("simulated handler failure")))
            }
        }
        desk.register_handler(Arc::new(FailingHandler));

        let res = desk
            .ingest_signal(ChangeNotice {
                source: "x".into(),
                summary: "y".into(),
                context: serde_json::json!({}),
            })
            .await;
        assert!(res.is_err());
    }

    // ============================================================
    // Chat agent (long-lived tau-session)
    // ============================================================

    /// Helper: build a desk pointing at a specific tempdir so multiple
    /// desks in a test can share session storage (validates restart
    /// recovery).
    async fn make_desk_at(
        tmp: &tempfile::TempDir,
        transport: Arc<dyn tau_agent::Transport>,
    ) -> Arc<DeskAgent> {
        let session_storage = Arc::new(FsStorage::new(tmp.path().to_path_buf()));
        let sessions = Arc::new(SessionManager::new(session_storage));
        let approval: Arc<dyn ApprovalPolicy> = Arc::new(DefaultPolicy);
        let storage: Arc<dyn DeskStorage> = Arc::new(MemDeskStorage::new());

        let cfg = DeskConfig::new(
            transport,
            storage,
            sessions,
            approval,
            make_test_config(),
            tmp.path().to_path_buf(),
        );
        Arc::new(DeskAgent::new(cfg).await.unwrap())
    }

    #[tokio::test]
    async fn ask_creates_chat_session_lazily() {
        let transport = Arc::new(MockTransport::new().with_text_response("hello"));
        let tmp = tempfile::tempdir().unwrap();
        let desk = make_desk_at(&tmp, transport).await;

        // No chat session before first ask.
        assert!(desk.chat_session_id().is_none());

        let rx = desk.ask("hi tau".into()).await.unwrap();
        let result = rx.await.unwrap();
        assert!(result.result.is_ok());

        // Chat session exists now.
        let id = desk.chat_session_id().expect("chat session created");

        // Listed under the marker title.
        let infos = desk.sessions.list().await.unwrap();
        let chat_info = infos.iter().find(|s| s.id == id).unwrap();
        assert_eq!(chat_info.title, "<tau-desk chat>");
    }

    #[tokio::test]
    async fn ask_multi_turn_reuses_same_session() {
        let transport = Arc::new(
            MockTransport::new()
                .with_text_response("first")
                .with_text_response("second"),
        );
        let tmp = tempfile::tempdir().unwrap();
        let desk = make_desk_at(&tmp, transport).await;

        desk.ask("turn one".into()).await.unwrap().await.unwrap();
        let id1 = desk.chat_session_id().unwrap();

        desk.ask("turn two".into()).await.unwrap().await.unwrap();
        let id2 = desk.chat_session_id().unwrap();

        assert_eq!(id1, id2, "multi-turn ask should reuse the same session");

        // Conversation accumulates: 2 user prompts → ≥4 messages
        // (user/assistant pairs).
        let handle = desk.sessions.handle(&id1).await.unwrap();
        let messages = handle.messages().await.unwrap();
        assert!(
            messages.len() >= 4,
            "expected accumulated conversation, got {} messages",
            messages.len()
        );
    }

    #[tokio::test]
    async fn shutdown_hibernates_chat_session() {
        let transport = Arc::new(MockTransport::new().with_text_response("hi"));
        let tmp = tempfile::tempdir().unwrap();
        let desk = make_desk_at(&tmp, transport).await;

        desk.ask("hi".into()).await.unwrap().await.unwrap();
        let id = desk.chat_session_id().unwrap();

        // Active before shutdown.
        assert!(desk.sessions.handle(&id).await.is_some());

        desk.shutdown().await.unwrap();

        // Hibernated after shutdown.
        assert!(desk.sessions.handle(&id).await.is_none());
        let info = desk
            .sessions
            .list()
            .await
            .unwrap()
            .into_iter()
            .find(|s| s.id == id)
            .unwrap();
        assert_eq!(info.status, SessionStatus::Hibernated);
    }

    /// Restart preservation: shut down a desk, build a fresh `DeskAgent`
    /// pointing at the same `tau-session` storage, ask again, verify
    /// the chat session was recovered and conversation history persists.
    #[tokio::test]
    async fn chat_session_recovered_across_desk_restart() {
        let tmp = tempfile::tempdir().unwrap();

        // First desk: create chat, accumulate one turn, shutdown.
        let id_before = {
            let transport = Arc::new(MockTransport::new().with_text_response("first"));
            let desk = make_desk_at(&tmp, transport).await;
            desk.ask("hi".into()).await.unwrap().await.unwrap();
            let id = desk.chat_session_id().unwrap();
            desk.shutdown().await.unwrap();
            id
        };

        // Second desk: same tempdir, fresh transport, fresh DeskAgent.
        // The chat session should be discovered by title and reused.
        let transport = Arc::new(MockTransport::new().with_text_response("second"));
        let desk2 = make_desk_at(&tmp, transport).await;

        // Recovered before any ask.
        let id_after = desk2.chat_session_id().expect("chat session recovered");
        assert_eq!(id_after, id_before);

        // Ask again — should activate the hibernated session and
        // append to the existing conversation.
        desk2.ask("again".into()).await.unwrap().await.unwrap();

        let handle = desk2.sessions.handle(&id_after).await.unwrap();
        let messages = handle.messages().await.unwrap();
        assert!(
            messages.len() >= 4,
            "expected continuation of prior conversation; got {} messages",
            messages.len()
        );
    }

    #[tokio::test]
    async fn chat_agent_can_use_desk_state_tools() {
        // Chat agent calls desk_add_activity. Validates the
        // `agent_id: "chat"` provenance contract end-to-end.
        let transport = Arc::new(
            MockTransport::new()
                .with_tool_call_response(
                    "desk_add_activity",
                    "chat-call-1",
                    serde_json::json!({ "text": "user asked about PRs" }),
                )
                .with_text_response("logged it"),
        );
        let tmp = tempfile::tempdir().unwrap();
        let desk = make_desk_at(&tmp, transport).await;

        desk.ask("log that I asked about PRs".into())
            .await
            .unwrap()
            .await
            .unwrap();

        let activity = desk.storage.list_activity(10).await.unwrap();
        assert_eq!(activity.len(), 1);
        assert_eq!(activity[0].text, "user asked about PRs");
    }

    // ============================================================
    // Scheduler runtime
    // ============================================================

    fn task_plain(name: &str, prompt: &str, concurrency: Concurrency) -> ScheduledTask {
        ScheduledTask {
            id: name.to_string(),
            name: name.to_string(),
            trigger: Trigger::Manual,
            concurrency,
            prompt: PromptSpec::Plain(prompt.to_string()),
            enabled: true,
        }
    }

    async fn make_desk_with_tasks(
        transport: Arc<dyn tau_agent::Transport>,
        tasks: Vec<ScheduledTask>,
    ) -> (Arc<DeskAgent>, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let session_storage = Arc::new(FsStorage::new(tmp.path().to_path_buf()));
        let sessions = Arc::new(SessionManager::new(session_storage));
        let approval: Arc<dyn ApprovalPolicy> = Arc::new(DefaultPolicy);
        let storage: Arc<dyn DeskStorage> = Arc::new(MemDeskStorage::new());

        let mut cfg = DeskConfig::new(
            transport,
            storage,
            sessions,
            approval,
            make_test_config(),
            tmp.path().to_path_buf(),
        );
        cfg.tasks = tasks;
        let desk = Arc::new(DeskAgent::new(cfg).await.unwrap());
        (desk, tmp)
    }

    #[tokio::test]
    async fn trigger_scan_runs_registered_task() {
        let transport = Arc::new(
            MockTransport::new()
                .with_tool_call_response(
                    "desk_add_activity",
                    "t1",
                    serde_json::json!({ "text": "scan ran" }),
                )
                .with_text_response("done"),
        );
        let task = task_plain("morning_scan", "Run", Concurrency::Skip);
        let (desk, _tmp) = make_desk_with_tasks(transport, vec![task]).await;

        desk.trigger_scan(&"morning_scan".to_string())
            .await
            .unwrap();

        let activity = desk.storage.list_activity(10).await.unwrap();
        assert_eq!(activity.len(), 1);
        assert_eq!(activity[0].text, "scan ran");
    }

    #[tokio::test]
    async fn trigger_scan_unknown_task_errors() {
        let (desk, _tmp) = make_desk_with_tasks(Arc::new(MockTransport::new()), vec![]).await;
        let res = desk.trigger_scan(&"nonexistent".to_string()).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn skip_policy_skips_when_in_flight() {
        // Pre-mark a fake in-flight token; second trigger should skip.
        let transport = Arc::new(MockTransport::new()); // no responses queued — would panic if dispatched
        let task = task_plain("blocked", "Run", Concurrency::Skip);
        let (desk, _tmp) = make_desk_with_tasks(transport, vec![task]).await;

        desk.in_flight.lock().insert(
            "blocked".to_string(),
            tokio_util::sync::CancellationToken::new(),
        );

        // No agent dispatch; transport is never called → no panic.
        desk.trigger_scan(&"blocked".to_string()).await.unwrap();

        // Still in flight (we left the token there).
        assert!(desk.in_flight.lock().contains_key("blocked"));
    }

    #[tokio::test]
    async fn signal_trigger_fires_on_matching_notice() {
        let (source_tx, _) = broadcast::channel::<ChangeNotice>(8);

        struct WatchOnly {
            tx: broadcast::Sender<ChangeNotice>,
        }
        #[async_trait::async_trait]
        impl Source for WatchOnly {
            fn id(&self) -> &str {
                "gh"
            }
            fn tools(&self) -> Vec<tau_agent::BoxedTool> {
                vec![]
            }
            fn watch(&self) -> Option<broadcast::Receiver<ChangeNotice>> {
                Some(self.tx.subscribe())
            }
        }

        let transport = Arc::new(
            MockTransport::new()
                .with_tool_call_response(
                    "desk_add_activity",
                    "t1",
                    serde_json::json!({ "text": "signal received" }),
                )
                .with_text_response("done"),
        );

        let task = ScheduledTask {
            id: "webhook_gh".into(),
            name: "webhook_gh".into(),
            trigger: Trigger::OnSignal("gh".into()),
            concurrency: Concurrency::Parallel,
            prompt: PromptSpec::Plain("Handle signal".into()),
            enabled: true,
        };
        let tmp = tempfile::tempdir().unwrap();
        let session_storage = Arc::new(FsStorage::new(tmp.path().to_path_buf()));
        let sessions = Arc::new(SessionManager::new(session_storage));
        let approval: Arc<dyn ApprovalPolicy> = Arc::new(DefaultPolicy);
        let storage: Arc<dyn DeskStorage> = Arc::new(MemDeskStorage::new());

        let mut cfg = DeskConfig::new(
            transport,
            storage.clone(),
            sessions,
            approval,
            make_test_config(),
            tmp.path().to_path_buf(),
        );
        cfg.sources
            .register(Arc::new(WatchOnly {
                tx: source_tx.clone(),
            }))
            .unwrap();
        cfg.tasks = vec![task];
        let desk = Arc::new(DeskAgent::new(cfg).await.unwrap());

        desk.start().await.unwrap();

        // Push a matching notice.
        source_tx
            .send(ChangeNotice {
                source: "gh".into(),
                summary: "PR comment".into(),
                context: serde_json::json!({}),
            })
            .unwrap();

        // Wait for the signal loop + fire to land. Real time, short.
        for _ in 0..40 {
            tokio::time::sleep(Duration::from_millis(25)).await;
            if !storage.list_activity(10).await.unwrap().is_empty() {
                break;
            }
        }
        let activity = storage.list_activity(10).await.unwrap();
        assert_eq!(activity.len(), 1);
        assert_eq!(activity[0].text, "signal received");

        desk.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn shutdown_aborts_loops() {
        let task = ScheduledTask {
            id: "ticker".into(),
            name: "ticker".into(),
            trigger: Trigger::Interval(Duration::from_secs(3600)), // very long; never fires in test window
            concurrency: Concurrency::Skip,
            prompt: PromptSpec::Plain("tick".into()),
            enabled: true,
        };
        let (desk, _tmp) = make_desk_with_tasks(Arc::new(MockTransport::new()), vec![task]).await;

        desk.start().await.unwrap();
        assert_eq!(desk.loop_handles.lock().len(), 1);

        desk.shutdown().await.unwrap();

        // Loop handles drained.
        assert!(desk.loop_handles.lock().is_empty());

        // Subsequent calls are idempotent.
        desk.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn start_with_invalid_cron_errors() {
        let task = ScheduledTask {
            id: "bogus".into(),
            name: "bogus".into(),
            trigger: Trigger::Cron("not a real cron".into()),
            concurrency: Concurrency::Skip,
            prompt: PromptSpec::Plain("x".into()),
            enabled: true,
        };
        let (desk, _tmp) = make_desk_with_tasks(Arc::new(MockTransport::new()), vec![task]).await;

        let res = desk.start().await;
        assert!(res.is_err(), "invalid cron should error");
    }

    #[tokio::test]
    async fn cancel_task_unblocks_in_flight_run() {
        // Test that cancel_task triggers the cancel token. We don't run
        // a real agent here — manually insert a token, call cancel, and
        // verify it fired.
        let (desk, _tmp) = make_desk_with_tasks(Arc::new(MockTransport::new()), vec![]).await;

        let token = tokio_util::sync::CancellationToken::new();
        desk.in_flight
            .lock()
            .insert("running".into(), token.clone());

        assert!(!token.is_cancelled());
        desk.cancel_task(&"running".to_string()).await.unwrap();
        assert!(token.is_cancelled());
    }

    #[tokio::test]
    async fn hydrate_prompt_substitutes_state() {
        let (desk, _tmp) = make_desk_with_tasks(Arc::new(MockTransport::new()), vec![]).await;

        // Seed some state.
        desk.user_create_note("Don't ping David before 10".into(), CardPile::Watching)
            .await
            .unwrap();
        desk.storage
            .add_tombstone("jira:OLD-1", Some("dismissed".into()))
            .await
            .unwrap();

        let spec = PromptSpec::Hydrated {
            template: "Pre.\n\n{{state}}\nPost.".into(),
            include: HydrationSpec {
                cards_in: vec![CardPile::Watching],
                drafts: false,
                activity_recent: 0,
                notes: true,
                brief: false,
                show_provenance: true,
            },
        };
        let out = desk.hydrate_prompt(&spec).await.unwrap();

        assert!(out.starts_with("Pre.\n"));
        assert!(out.ends_with("\nPost."));
        assert!(out.contains("Cards in Watching"));
        assert!(out.contains("last_modified_by=User"));
        assert!(out.contains("Notes"));
        assert!(out.contains("Don't ping David"));
        assert!(out.contains("Active tombstones"));
        assert!(out.contains("jira:OLD-1"));
    }

    /// End-to-end slice: spawn a per-task agent, let it call
    /// `desk_add_activity` via the standard tool dispatch path, verify
    /// the activity log got the entry. Validates: tool registration on
    /// `AgentBuilder`, tool execute body wiring, the per-task spawn +
    /// drop pattern, and `DeskEvent::ActivityAppended` emission.
    #[tokio::test]
    async fn run_task_once_dispatches_add_activity_via_agent() {
        let transport = MockTransport::new()
            .with_tool_call_response(
                "desk_add_activity",
                "call-1",
                serde_json::json!({ "text": "morning scan complete" }),
            )
            .with_text_response("done");

        let (desk, _tmp) = make_desk_with_transport(Arc::new(transport)).await;

        // Subscribe before running so we don't race the broadcast.
        let mut rx = desk.subscribe();

        desk.run_task_once(&"morning_scan".to_string(), "Run the morning scan".into())
            .await
            .expect("task should complete");

        let activity = desk.storage.list_activity(10).await.unwrap();
        assert_eq!(activity.len(), 1);
        assert_eq!(activity[0].text, "morning scan complete");
        assert!(activity[0].seq > 0);

        // ScanStarted, ActivityAppended, ScanCompleted should all fire.
        let mut saw_started = false;
        let mut saw_appended = false;
        let mut saw_completed = false;
        for _ in 0..10 {
            match tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv()).await {
                Ok(Ok(DeskEvent::ScanStarted { task })) if task == "morning_scan" => {
                    saw_started = true;
                }
                Ok(Ok(DeskEvent::ActivityAppended { entry })) => {
                    assert_eq!(entry.text, "morning scan complete");
                    saw_appended = true;
                }
                Ok(Ok(DeskEvent::ScanCompleted { task })) if task == "morning_scan" => {
                    saw_completed = true;
                }
                Ok(Ok(_)) => {}
                Ok(Err(_)) | Err(_) => break,
            }
        }
        assert!(saw_started, "missing ScanStarted");
        assert!(saw_appended, "missing ActivityAppended");
        assert!(saw_completed, "missing ScanCompleted");
    }
}
