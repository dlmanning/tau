//! `AgentHandle` — channel-based interaction surface for an agent.
//!
//! `Clone + Send + Sync`. All methods take `&self`. Cloning is cheap
//! (refcount bumps) and clones are interchangeable.
//!
//! Each mutation comes in two flavors:
//!
//! - `try_<op>(...)` — sync, non-blocking. Returns
//!   `Err(Error::ChannelFull)` if the channel is full (caller may
//!   retry) or `Err(Error::ActorPanic | Error::Other)` if the actor
//!   is dead. Use from sync contexts (UI event handlers).
//! - `<op>(...).await` — async, awaits channel space. Only fails if
//!   the actor is dead. Preferred for correctness-critical commands.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::core::approval::ApprovalPolicy;
use crate::core::command::{Command, PromptResult};
use crate::core::compaction::CompactionConfig;
use crate::core::config::AgentConfig;
use crate::core::state::Shared;
use crate::types::conversation::Conversation;
use crate::types::error::{Error, Result};
use crate::types::events::{AgentEvent, CompactionReason};
use crate::types::info::ToolInfo;

/// Channel-based handle into a running agent. **Pure-core**: this type
/// does not know about the fleet. Spec transitions (`respec` /
/// `respec_with`) are methods on `fleet::AgentManager`; callers pair
/// a handle with its manager when they need them.
#[derive(Clone)]
pub struct AgentHandle {
    pub(crate) urgent_tx: mpsc::Sender<Command>,
    pub(crate) normal_tx: mpsc::Sender<Command>,
    pub(crate) event_tx: broadcast::Sender<AgentEvent>,
    pub(crate) shared: Shared,
}

impl AgentHandle {
    // ─── Prompts ─────────────────────────────────────────────────────

    pub async fn prompt(&self, input: &str) -> Result<oneshot::Receiver<PromptResult>> {
        let content = vec![tau_ai::Content::text(input)];
        self.prompt_with_content(content).await
    }

    pub async fn prompt_with_content(
        &self,
        content: Vec<tau_ai::Content>,
    ) -> Result<oneshot::Receiver<PromptResult>> {
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
    pub async fn prompt_and_wait(&self, input: &str) -> Result<()> {
        let rx = self.prompt(input).await?;
        match rx.await {
            Ok(r) => r.result,
            Err(_) => Err(self
                .dead_actor_error("Agent task dropped without responding")
                .await),
        }
    }

    // ─── Steer / follow-up (urgent) ──────────────────────────────────

    pub fn try_steer(&self, message: tau_ai::Message) -> Result<()> {
        self.try_send(Command::Steer(message))
    }
    pub async fn steer(&self, message: tau_ai::Message) -> Result<()> {
        self.send(Command::Steer(message)).await
    }
    pub fn try_follow_up(&self, message: tau_ai::Message) -> Result<()> {
        self.try_send(Command::FollowUp(message))
    }
    pub async fn follow_up(&self, message: tau_ai::Message) -> Result<()> {
        self.send(Command::FollowUp(message)).await
    }

    // ─── Config setters ──────────────────────────────────────────────

    pub fn try_set_model(&self, m: tau_ai::Model) -> Result<()> {
        self.try_send(Command::SetModel(m))
    }
    pub async fn set_model(&self, m: tau_ai::Model) -> Result<()> {
        self.send(Command::SetModel(m)).await
    }
    pub fn try_set_reasoning(&self, l: tau_ai::ReasoningLevel) -> Result<()> {
        self.try_send(Command::SetReasoning(l))
    }
    pub async fn set_reasoning(&self, l: tau_ai::ReasoningLevel) -> Result<()> {
        self.send(Command::SetReasoning(l)).await
    }
    pub fn try_set_compaction_config(&self, c: CompactionConfig) -> Result<()> {
        self.try_send(Command::SetCompactionConfig(c))
    }
    pub async fn set_compaction_config(&self, c: CompactionConfig) -> Result<()> {
        self.send(Command::SetCompactionConfig(c)).await
    }
    pub fn try_set_approval_policy(&self, p: Arc<dyn ApprovalPolicy>) -> Result<()> {
        self.try_send(Command::SetApprovalPolicy(p))
    }
    pub async fn set_approval_policy(&self, p: Arc<dyn ApprovalPolicy>) -> Result<()> {
        self.send(Command::SetApprovalPolicy(p)).await
    }

    // ─── Queries (request-reply) ─────────────────────────────────────

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

    /// Snapshot of every tool currently registered on the agent, with
    /// its category and current approval status under the active
    /// policy. Returns `None` if the actor is dead.
    pub async fn list_tools(&self) -> Option<Vec<ToolInfo>> {
        let (tx, rx) = oneshot::channel();
        self.normal_tx.send(Command::ListTools(tx)).await.ok()?;
        rx.await.ok()
    }

    pub async fn compact(
        &self,
        reason: CompactionReason,
    ) -> Result<oneshot::Receiver<PromptResult>> {
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

    // ─── Subscribe / read ────────────────────────────────────────────

    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.event_tx.subscribe()
    }

    pub fn event_sender(&self) -> broadcast::Sender<AgentEvent> {
        self.event_tx.clone()
    }

    pub fn is_running(&self) -> bool {
        self.shared.is_running.load(Ordering::Acquire)
    }

    /// Manager-assigned id when known. `None` for unmanaged handles.
    pub fn agent_id(&self) -> Option<&str> {
        self.shared.agent_id.get().map(String::as_str)
    }

    /// Set the agent's id. Called by the fleet at adoption / spawn
    /// time. The cell is `OnceLock`; a second set with a different
    /// value silently fails — the first writer wins.
    pub fn set_agent_id(&self, id: String) -> std::result::Result<(), String> {
        self.shared.agent_id.set(id)
    }

    pub fn shutdown_reason(&self) -> Option<String> {
        self.shared.shutdown_reason.lock().clone()
    }

    // ─── Abort ───────────────────────────────────────────────────────

    /// Cancel the current operation. The actor swaps the inner token
    /// at each prompt start so this always targets the active prompt.
    pub fn abort(&self) {
        self.shared.cancel.lock().cancel();
    }

    pub fn cancel_token(&self) -> Arc<Mutex<CancellationToken>> {
        self.shared.cancel.clone()
    }

    // ─── Background follow-up tracking (called by the fleet) ─────────

    pub fn expect_follow_up(&self) {
        self.shared
            .pending_follow_ups
            .fetch_add(1, Ordering::Release);
    }

    pub fn consume_follow_up(&self) {
        let _ = self.shared.pending_follow_ups.fetch_update(
            Ordering::Release,
            Ordering::Acquire,
            |n| if n > 0 { Some(n - 1) } else { None },
        );
    }

    pub fn has_pending_follow_ups(&self) -> bool {
        self.shared.pending_follow_ups.load(Ordering::Acquire) > 0
    }

    // ─── Internal: channel routing ───────────────────────────────────

    fn channel_for(&self, cmd: &Command) -> &mpsc::Sender<Command> {
        if cmd.is_urgent() {
            &self.urgent_tx
        } else {
            &self.normal_tx
        }
    }

    fn try_send(&self, cmd: Command) -> Result<()> {
        let label: &'static str = if cmd.is_urgent() { "urgent" } else { "normal" };
        let tx = self.channel_for(&cmd);
        match tx.try_send(cmd) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                tracing::warn!("Command channel `{label}` is full; command dropped");
                Err(Error::ChannelFull { channel: label })
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Err(self.dead_actor_error_sync("Agent task has shut down"))
            }
        }
    }

    async fn send(&self, cmd: Command) -> Result<()> {
        let tx = self.channel_for(&cmd);
        if tx.send(cmd).await.is_err() {
            return Err(self.dead_actor_error("Agent task has shut down").await);
        }
        Ok(())
    }

    /// Async version: briefly waits for the catch_unwind wrapper to
    /// record a panic reason. Without this, a post-panic send races
    /// the channel-close vs. the reason-write and surfaces
    /// `Error::Other` instead of `Error::ActorPanic`.
    async fn dead_actor_error(&self, fallback: &str) -> Error {
        let notified = self.shared.shutdown_signaled.notified();
        if let Some(reason) = self.shutdown_reason() {
            return Error::ActorPanic(reason);
        }
        let _ = tokio::time::timeout(Duration::from_millis(100), notified).await;
        match self.shutdown_reason() {
            Some(reason) => Error::ActorPanic(reason),
            None => Error::Other(fallback.into()),
        }
    }

    /// Sync best-effort. May return `Error::Other` if the panic reason
    /// hasn't been recorded yet, even when one is in flight. Async
    /// callers should prefer `dead_actor_error`.
    fn dead_actor_error_sync(&self, fallback: &str) -> Error {
        if let Some(reason) = self.shutdown_reason() {
            Error::ActorPanic(reason)
        } else {
            Error::Other(fallback.into())
        }
    }
}
