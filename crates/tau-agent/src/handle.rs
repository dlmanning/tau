//! Channel-based handle for interacting with a running agent.
//!
//! `AgentHandle` is `Clone + Send + Sync`. All methods take `&self`.
//! Fire-and-forget methods use `try_send`. Abort cancels the shared
//! `CancellationToken` inside `Arc<Mutex<>>`.

use std::sync::{Arc, OnceLock, Weak};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::{Notify, broadcast, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::approval::ApprovalPolicy;
use crate::command::{Command, PromptResult};
use crate::compaction::{CompactionConfig, CompactionReason};
use crate::config::AgentConfig;
use crate::conversation::Conversation;
use crate::events::AgentEvent;
use crate::manager::{AgentManager, AgentSpec};

/// Cloneable handle to a running agent. All methods take `&self`.
/// This is the only way external code interacts with the agent.
///
/// Commands are split across two channels:
/// - **urgent** (Steer, FollowUp) — processed with priority during streaming/tools
/// - **normal** (everything else) — queries, config mutations, prompts
#[derive(Clone)]
pub struct AgentHandle {
    /// Priority channel for Steer/FollowUp commands.
    pub(crate) urgent_tx: mpsc::Sender<Command>,
    /// Normal channel for everything else.
    pub(crate) normal_tx: mpsc::Sender<Command>,
    pub(crate) event_tx: broadcast::Sender<AgentEvent>,
    /// Shared with actor. The actor replaces the inner token at each prompt start.
    /// `abort()` cancels the current token. `Arc<Mutex<>>` ensures the handle
    /// always cancels whichever token the actor is currently using.
    pub(crate) cancel: Arc<Mutex<CancellationToken>>,
    pub(crate) is_running: Arc<AtomicBool>,
    pub(crate) pending_follow_ups: Arc<AtomicU32>,
    /// Set by the actor's catch_unwind wrapper if the actor task panics.
    /// `None` while the actor is alive or after a clean shutdown.
    pub(crate) shutdown_reason: Arc<Mutex<Option<String>>>,
    /// Notified by the actor's catch_unwind wrapper after writing
    /// `shutdown_reason`. Used by `dead_actor_error` to bound a small wait
    /// closing the race between channel-close (visible to the handle as
    /// `SendError`) and the panic reason being recorded.
    pub(crate) shutdown_signaled: Arc<Notify>,
    /// Manager-assigned id, set when the handle was produced by the
    /// `AgentManager` (or registered with one). `None` for builder-spawned
    /// root agents that never met a manager — those can't `respec`.
    pub(crate) agent_id: Arc<OnceLock<String>>,
    /// Weak reference back to the `AgentManager` that owns this agent.
    /// `respec` / `with_system_prompt` / `with_tools` need this to look up
    /// the current spec and dispatch a fresh-agent spawn. `None` for
    /// unmanaged handles.
    pub(crate) manager: Arc<OnceLock<Weak<AgentManager>>>,
}

impl AgentHandle {
    /// Send a prompt. Returns a oneshot receiver for the completion result.
    pub async fn prompt(
        &self,
        input: &str,
    ) -> crate::error::Result<oneshot::Receiver<PromptResult>> {
        let content = vec![tau_ai::Content::text(input)];
        self.prompt_with_content(content).await
    }

    /// Send a prompt with structured content blocks.
    pub async fn prompt_with_content(
        &self,
        content: Vec<tau_ai::Content>,
    ) -> crate::error::Result<oneshot::Receiver<PromptResult>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self
            .normal_tx
            .send(Command::Prompt {
                content,
                reply: reply_tx,
            })
            .await
            .is_err()
        {
            return Err(self.dead_actor_error("Agent task has shut down").await);
        }
        Ok(reply_rx)
    }

    /// Convenience: send prompt and block until completion.
    pub async fn prompt_and_wait(&self, input: &str) -> crate::error::Result<()> {
        let rx = self.prompt(input).await?;
        match rx.await {
            Ok(result) => result.result,
            Err(_) => Err(self
                .dead_actor_error("Agent task dropped without responding")
                .await),
        }
    }

    /// Async build of the dead-actor error. Briefly waits (≤100 ms) for the
    /// catch_unwind wrapper to record a panic reason, since the channel
    /// closes during unwind *before* the reason is written. Without this
    /// wait, the first post-panic send would race and surface `Error::Other`
    /// instead of `Error::ActorPanic`.
    async fn dead_actor_error(&self, fallback: &str) -> crate::error::Error {
        // Create the Notified future *before* reading the reason so we can't
        // miss a notification fired between the read and the await.
        let notified = self.shutdown_signaled.notified();
        if let Some(reason) = self.shutdown_reason() {
            return crate::error::Error::ActorPanic(reason);
        }
        let _ = tokio::time::timeout(Duration::from_millis(100), notified).await;
        match self.shutdown_reason() {
            Some(reason) => crate::error::Error::ActorPanic(reason),
            None => crate::error::Error::Other(fallback.into()),
        }
    }

    /// Sync best-effort version for `try_X` paths that cannot await. May
    /// return `Error::Other` if the panic reason hasn't been recorded yet,
    /// even when one is in flight. Callers wanting the precise reason
    /// should use the async send variants.
    fn dead_actor_error_sync(&self, fallback: &str) -> crate::error::Error {
        if let Some(reason) = self.shutdown_reason() {
            crate::error::Error::ActorPanic(reason)
        } else {
            crate::error::Error::Other(fallback.into())
        }
    }

    /// Reason the actor task is no longer alive, if known. `Some(reason)`
    /// indicates a panic; `None` means either the actor is alive or shut down
    /// cleanly (e.g. all handles dropped).
    pub fn shutdown_reason(&self) -> Option<String> {
        self.shutdown_reason.lock().clone()
    }

    /// Subscribe to the event stream.
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.event_tx.subscribe()
    }

    /// Get the event sender (for AgentManager event forwarding).
    pub fn event_sender(&self) -> broadcast::Sender<AgentEvent> {
        self.event_tx.clone()
    }

    // === Mutations ===
    //
    // Each mutation comes in two flavors:
    //   * `try_<op>(...)` — sync, non-blocking. Returns
    //     `Err(Error::ChannelFull { channel })` if the channel is full
    //     (caller may retry) or `Err(Error::ActorPanic(_) | Error::Other(_))`
    //     if the actor is dead. Use from sync contexts (UI event handlers)
    //     where you cannot await.
    //   * `<op>(...).await` — async, awaits channel space. Returns
    //     `Err(Error::ActorPanic(...))` or `Err(Error::Other(...))` only if
    //     the actor is dead. Use from async contexts; preferred for
    //     correctness-critical commands (Steer, FollowUp, config setters).

    /// Pick the right channel for `cmd`.
    fn channel_for(&self, cmd: &Command) -> &mpsc::Sender<Command> {
        if cmd.is_urgent() {
            &self.urgent_tx
        } else {
            &self.normal_tx
        }
    }

    /// Sync, non-blocking send. Returns `Error::ChannelFull` (retryable) or
    /// `Error::ActorPanic`/`Error::Other` (terminal) so callers can
    /// distinguish backpressure from a dead actor.
    fn try_send_command(&self, cmd: Command) -> crate::error::Result<()> {
        let label: &'static str = if cmd.is_urgent() { "urgent" } else { "normal" };
        let tx = self.channel_for(&cmd);
        match tx.try_send(cmd) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                tracing::warn!("Command channel `{label}` is full; command dropped");
                Err(crate::error::Error::ChannelFull { channel: label })
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Err(self.dead_actor_error_sync("Agent task has shut down"))
            }
        }
    }

    /// Async send that awaits channel space. Only fails if the actor is dead.
    async fn send_command(&self, cmd: Command) -> crate::error::Result<()> {
        let tx = self.channel_for(&cmd);
        if tx.send(cmd).await.is_err() {
            return Err(self.dead_actor_error("Agent task has shut down").await);
        }
        Ok(())
    }

    // --- Steer / FollowUp (urgent channel) ---

    pub fn try_steer(&self, message: tau_ai::Message) -> crate::error::Result<()> {
        self.try_send_command(Command::Steer(message))
    }

    pub async fn steer(&self, message: tau_ai::Message) -> crate::error::Result<()> {
        self.send_command(Command::Steer(message)).await
    }

    pub fn try_follow_up(&self, message: tau_ai::Message) -> crate::error::Result<()> {
        self.try_send_command(Command::FollowUp(message))
    }

    pub async fn follow_up(&self, message: tau_ai::Message) -> crate::error::Result<()> {
        self.send_command(Command::FollowUp(message)).await
    }

    // --- Config mutations (normal channel) ---

    pub fn try_set_model(&self, model: tau_ai::Model) -> crate::error::Result<()> {
        self.try_send_command(Command::SetModel(model))
    }

    pub async fn set_model(&self, model: tau_ai::Model) -> crate::error::Result<()> {
        self.send_command(Command::SetModel(model)).await
    }

    pub fn try_set_reasoning(&self, level: tau_ai::ReasoningLevel) -> crate::error::Result<()> {
        self.try_send_command(Command::SetReasoning(level))
    }

    pub async fn set_reasoning(
        &self,
        level: tau_ai::ReasoningLevel,
    ) -> crate::error::Result<()> {
        self.send_command(Command::SetReasoning(level)).await
    }

    pub fn try_set_compaction_config(&self, config: CompactionConfig) -> crate::error::Result<()> {
        self.try_send_command(Command::SetCompactionConfig(config))
    }

    pub async fn set_compaction_config(
        &self,
        config: CompactionConfig,
    ) -> crate::error::Result<()> {
        self.send_command(Command::SetCompactionConfig(config))
            .await
    }

    pub fn try_set_approval_policy(
        &self,
        policy: Arc<dyn ApprovalPolicy>,
    ) -> crate::error::Result<()> {
        self.try_send_command(Command::SetApprovalPolicy(policy))
    }

    pub async fn set_approval_policy(
        &self,
        policy: Arc<dyn ApprovalPolicy>,
    ) -> crate::error::Result<()> {
        self.send_command(Command::SetApprovalPolicy(policy)).await
    }

    // === Abort ===

    /// Abort the current operation. Cancels the current token inside the
    /// shared `Arc<Mutex<>>`. The actor replaces the token at each prompt start,
    /// so this always targets the active prompt.
    pub fn abort(&self) {
        self.cancel.lock().cancel();
    }

    /// Get a clone of the shared cancel token container.
    pub fn cancel_token(&self) -> Arc<Mutex<CancellationToken>> {
        self.cancel.clone()
    }

    // === Background follow-up tracking ===

    pub fn expect_follow_up(&self) {
        self.pending_follow_ups.fetch_add(1, Ordering::Release);
    }

    pub fn consume_follow_up(&self) {
        let _ = self
            .pending_follow_ups
            .fetch_update(Ordering::Release, Ordering::Acquire, |n| {
                if n > 0 { Some(n - 1) } else { None }
            });
    }

    pub fn has_pending_follow_ups(&self) -> bool {
        self.pending_follow_ups.load(Ordering::Acquire) > 0
    }

    // === Request-reply queries ===

    pub async fn config(&self) -> Option<AgentConfig> {
        let (tx, rx) = oneshot::channel();
        self.normal_tx.send(Command::GetConfig(tx)).await.ok()?;
        rx.await.ok()
    }

    pub async fn messages(&self) -> Option<Vec<tau_ai::Message>> {
        let (tx, rx) = oneshot::channel();
        self.normal_tx.send(Command::GetMessages(tx)).await.ok()?;
        rx.await.ok()
    }

    pub async fn state(&self) -> Option<Conversation> {
        let (tx, rx) = oneshot::channel();
        self.normal_tx.send(Command::GetState(tx)).await.ok()?;
        rx.await.ok()
    }

    // === Compaction ===

    pub async fn compact(
        &self,
        reason: CompactionReason,
    ) -> crate::error::Result<oneshot::Receiver<PromptResult>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self
            .normal_tx
            .send(Command::Compact {
                reason,
                reply: reply_tx,
            })
            .await
            .is_err()
        {
            return Err(self.dead_actor_error("Agent task has shut down").await);
        }
        Ok(reply_rx)
    }

    /// Whether the agent is currently executing a prompt.
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Manager-assigned agent id, when this handle came from (or was
    /// registered with) an `AgentManager`. `None` for builder-spawned
    /// root agents.
    pub fn agent_id(&self) -> Option<&str> {
        self.agent_id.get().map(String::as_str)
    }

    /// Clone of the inner `Arc<OnceLock<String>>` carrying this agent's
    /// id. Tools that need to refer to their owning agent without holding
    /// a full `AgentHandle` (which forms a tools→handle→channel→actor
    /// cycle preventing the actor from exiting on eviction) capture this
    /// shared cell and call `.get()` at use time. The cell is populated
    /// when the manager stamps an id (via spawn or `adopt`), so capturing
    /// it pre-spawn from `pre_handle()` and reading it at execute time
    /// works.
    pub fn agent_id_arc(&self) -> Arc<OnceLock<String>> {
        self.agent_id.clone()
    }

    // === Respec ===
    //
    // Spec changes are new agents, full stop. The runtime treats every
    // `AgentSpec` as immutable for the agent's lifetime. To change one,
    // call `respec` (or one of the `with_*` convenience wrappers): the
    // manager spawns a fresh agent inheriting this agent's history under
    // the new spec and returns the new handle. The host then drives the
    // returned handle.

    /// Continue this conversation under a new spec. Requires the agent
    /// to be idle and to have been produced by an `AgentManager`. The
    /// returned handle is for the new agent; the old idle entry is
    /// dropped (see [`AgentManager::respec`]).
    pub async fn respec(
        self,
        new_spec: impl Into<std::sync::Arc<AgentSpec>>,
    ) -> crate::error::Result<AgentHandle> {
        let (mgr, id) = self.manager_and_id()?;
        mgr.respec(&id, new_spec).await
    }

    /// Convenience: respec, changing only the system prompt.
    pub async fn with_system_prompt(
        self,
        prompt: String,
    ) -> crate::error::Result<AgentHandle> {
        let (mgr, id) = self.manager_and_id()?;
        let spec = mgr.spec_for(&id).await.ok_or_else(|| {
            crate::error::Error::Other(format!(
                "with_system_prompt: no spec recorded for agent '{id}'"
            ))
        })?;
        // Reuse the buffer when we hold the only reference; otherwise
        // pay one deep clone. The registry holds a clone, so this almost
        // always clones — but it's the *only* clone in the path.
        let mut spec = Arc::unwrap_or_clone(spec);
        spec.system_prompt = prompt;
        mgr.respec(&id, spec).await
    }

    /// Convenience: respec, changing only the tool set.
    pub async fn with_tools(
        self,
        tools: Vec<crate::tool::BoxedTool>,
    ) -> crate::error::Result<AgentHandle> {
        let (mgr, id) = self.manager_and_id()?;
        let spec = mgr.spec_for(&id).await.ok_or_else(|| {
            crate::error::Error::Other(format!(
                "with_tools: no spec recorded for agent '{id}'"
            ))
        })?;
        let mut spec = Arc::unwrap_or_clone(spec);
        spec.tools = tools;
        mgr.respec(&id, spec).await
    }

    fn manager_and_id(&self) -> crate::error::Result<(Arc<AgentManager>, String)> {
        let id = self
            .agent_id
            .get()
            .cloned()
            .ok_or_else(|| crate::error::Error::Unmanaged)?;
        let mgr = self
            .manager
            .get()
            .and_then(Weak::upgrade)
            .ok_or_else(|| crate::error::Error::Unmanaged)?;
        Ok((mgr, id))
    }
}
