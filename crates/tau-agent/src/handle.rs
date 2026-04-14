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
        self.normal_tx
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

    /// Route a command to the correct channel based on urgency.
    fn send_command(&self, cmd: Command) {
        let tx = if cmd.is_urgent() { &self.urgent_tx } else { &self.normal_tx };
        let _ = tx.try_send(cmd);
    }

    pub fn steer(&self, message: tau_ai::Message) {
        self.send_command(Command::Steer(message));
    }

    pub fn follow_up(&self, message: tau_ai::Message) {
        self.send_command(Command::FollowUp(message));
    }

    pub fn set_model(&self, model: tau_ai::Model) {
        self.send_command(Command::SetModel(model));
    }

    pub fn set_reasoning(&self, level: tau_ai::ReasoningLevel) {
        self.send_command(Command::SetReasoning(level));
    }

    pub fn set_system_prompt(&self, prompt: String) {
        self.send_command(Command::SetSystemPrompt(prompt));
    }

    pub fn set_compaction_config(&self, config: CompactionConfig) {
        self.send_command(Command::SetCompactionConfig(config));
    }

    pub fn clear_messages(&self) {
        self.send_command(Command::ClearMessages);
    }

    pub fn set_messages(&self, messages: Vec<tau_ai::Message>) {
        self.send_command(Command::SetMessages(messages));
    }

    pub fn set_previous_summary(&self, summary: Option<String>) {
        self.send_command(Command::SetPreviousSummary(summary));
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
        self.normal_tx
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
