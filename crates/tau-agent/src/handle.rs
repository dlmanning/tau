//! Channel-based handle for interacting with a running agent.
//!
//! `AgentHandle` is `Clone + Send + Sync`. All methods take `&self`.
//! Fire-and-forget methods use `try_send`. Abort cancels the shared
//! `CancellationToken` inside `Arc<Mutex<>>`.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::command::{Command, PromptResult};
use crate::compaction::{CompactionConfig, CompactionReason};
use crate::config::AgentConfig;
use crate::conversation::Conversation;
use crate::events::AgentEvent;

/// Cloneable handle to a running agent. All methods take `&self`.
/// This is the only way external code interacts with the agent.
#[derive(Clone)]
pub struct AgentHandle {
    pub(crate) cmd_tx: mpsc::Sender<Command>,
    pub(crate) event_tx: broadcast::Sender<AgentEvent>,
    /// Shared with actor. The actor replaces the inner token at each prompt start.
    /// `abort()` cancels the current token. `Arc<Mutex<>>` ensures the handle
    /// always cancels whichever token the actor is currently using.
    pub(crate) cancel: Arc<Mutex<CancellationToken>>,
    pub(crate) is_running: Arc<AtomicBool>,
    pub(crate) pending_follow_ups: Arc<AtomicU32>,
}

impl AgentHandle {
    /// Send a prompt. Returns a oneshot receiver for the completion result.
    pub async fn prompt(&self, input: &str) -> crate::error::Result<oneshot::Receiver<PromptResult>> {
        let content = vec![tau_ai::Content::text(input)];
        self.prompt_with_content(content).await
    }

    /// Send a prompt with structured content blocks.
    pub async fn prompt_with_content(
        &self,
        content: Vec<tau_ai::Content>,
    ) -> crate::error::Result<oneshot::Receiver<PromptResult>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.cmd_tx
            .send(Command::Prompt {
                content,
                reply: reply_tx,
            })
            .await
            .map_err(|_| crate::error::Error::Other("Agent task has shut down".into()))?;
        Ok(reply_rx)
    }

    /// Convenience: send prompt and block until completion.
    pub async fn prompt_and_wait(&self, input: &str) -> crate::error::Result<()> {
        let rx = self.prompt(input).await?;
        match rx.await {
            Ok(result) => result.result,
            Err(_) => Err(crate::error::Error::Other(
                "Agent task dropped without responding".into(),
            )),
        }
    }

    /// Subscribe to the event stream.
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.event_tx.subscribe()
    }

    /// Get the event sender (for AgentManager event forwarding).
    pub fn event_sender(&self) -> broadcast::Sender<AgentEvent> {
        self.event_tx.clone()
    }

    // === Fire-and-forget mutations ===

    pub fn steer(&self, message: tau_ai::Message) {
        let _ = self.cmd_tx.try_send(Command::Steer(message));
    }

    pub fn follow_up(&self, message: tau_ai::Message) {
        let _ = self.cmd_tx.try_send(Command::FollowUp(message));
    }

    pub fn set_model(&self, model: tau_ai::Model) {
        let _ = self.cmd_tx.try_send(Command::SetModel(model));
    }

    pub fn set_reasoning(&self, level: tau_ai::ReasoningLevel) {
        let _ = self.cmd_tx.try_send(Command::SetReasoning(level));
    }

    pub fn set_system_prompt(&self, prompt: String) {
        let _ = self.cmd_tx.try_send(Command::SetSystemPrompt(prompt));
    }

    pub fn set_compaction_config(&self, config: CompactionConfig) {
        let _ = self.cmd_tx.try_send(Command::SetCompactionConfig(config));
    }

    pub fn clear_messages(&self) {
        let _ = self.cmd_tx.try_send(Command::ClearMessages);
    }

    pub fn set_messages(&self, messages: Vec<tau_ai::Message>) {
        let _ = self.cmd_tx.try_send(Command::SetMessages(messages));
    }

    pub fn set_previous_summary(&self, summary: Option<String>) {
        let _ = self.cmd_tx.try_send(Command::SetPreviousSummary(summary));
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
        self.cmd_tx.send(Command::GetConfig(tx)).await.ok()?;
        rx.await.ok()
    }

    pub async fn messages(&self) -> Option<Vec<tau_ai::Message>> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx.send(Command::GetMessages(tx)).await.ok()?;
        rx.await.ok()
    }

    pub async fn state(&self) -> Option<Conversation> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx.send(Command::GetState(tx)).await.ok()?;
        rx.await.ok()
    }

    // === Compaction ===

    pub async fn compact(
        &self,
        reason: CompactionReason,
    ) -> crate::error::Result<oneshot::Receiver<PromptResult>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.cmd_tx
            .send(Command::Compact {
                reason,
                reply: reply_tx,
            })
            .await
            .map_err(|_| crate::error::Error::Other("Agent task has shut down".into()))?;
        Ok(reply_rx)
    }

    /// Whether the agent is currently executing a prompt.
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }
}
