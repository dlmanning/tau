//! A cloneable handle for poking the agent from external code.

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU32, Ordering},
};

use parking_lot::Mutex;
use tau_ai::Message;
use tokio_util::sync::CancellationToken;

/// A cloneable handle for poking the agent from external code.
///
/// All fields are `Arc`-wrapped, so cloning is cheap.
#[derive(Clone)]
pub struct AgentHandle {
    pub(crate) cancel: Arc<Mutex<CancellationToken>>,
    pub(crate) steering_queue: Arc<Mutex<Vec<Message>>>,
    pub(crate) follow_up_queue: Arc<Mutex<Vec<Message>>>,
    pub(crate) follow_up_notify: Arc<tokio::sync::Notify>,
    pub(crate) idle_notify: Arc<tokio::sync::Notify>,
    pub(crate) is_running: Arc<AtomicBool>,
    /// Number of background agents that have been spawned but haven't
    /// posted their follow-up yet.
    pub(crate) pending_follow_ups: Arc<AtomicU32>,
}

impl AgentHandle {
    pub(crate) fn new() -> Self {
        Self {
            cancel: Arc::new(Mutex::new(CancellationToken::new())),
            steering_queue: Arc::new(Mutex::new(Vec::new())),
            follow_up_queue: Arc::new(Mutex::new(Vec::new())),
            follow_up_notify: Arc::new(tokio::sync::Notify::new()),
            idle_notify: Arc::new(tokio::sync::Notify::new()),
            is_running: Arc::new(AtomicBool::new(false)),
            pending_follow_ups: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Abort the current operation.
    pub fn abort(&self) {
        self.cancel.lock().cancel();
    }

    /// Get the cancellation token (for external callers that need direct access).
    pub fn cancel_token(&self) -> Arc<Mutex<CancellationToken>> {
        Arc::clone(&self.cancel)
    }

    /// Maximum number of messages in each queue.
    const MAX_QUEUE_SIZE: usize = 100;

    /// Enqueue a steering message that interrupts after the current tool completes.
    pub fn steer(&self, message: Message) {
        let mut q = self.steering_queue.lock();
        if q.len() >= Self::MAX_QUEUE_SIZE {
            tracing::warn!(
                "Steering queue full ({} messages), dropping oldest",
                Self::MAX_QUEUE_SIZE
            );
            q.remove(0);
        }
        q.push(message);
    }

    /// Enqueue a follow-up message consumed after the loop finishes.
    pub fn follow_up(&self, message: Message) {
        let mut q = self.follow_up_queue.lock();
        if q.len() >= Self::MAX_QUEUE_SIZE {
            tracing::warn!(
                "Follow-up queue full ({} messages), dropping oldest",
                Self::MAX_QUEUE_SIZE
            );
            q.remove(0);
        }
        q.push(message);
        drop(q);
        self.follow_up_notify.notify_one();
    }

    /// Record that a background agent was spawned and its follow-up is expected.
    pub fn expect_follow_up(&self) {
        self.pending_follow_ups.fetch_add(1, Ordering::Release);
    }

    /// Record that an expected follow-up was consumed.
    /// Only decrements if the counter is positive (follow-ups posted
    /// without a prior `expect_follow_up` are ignored).
    pub fn consume_follow_up(&self) {
        let _ = self
            .pending_follow_ups
            .fetch_update(Ordering::Release, Ordering::Acquire, |n| {
                if n > 0 { Some(n - 1) } else { None }
            });
    }

    /// Whether there are background agents that haven't posted their follow-up yet.
    pub fn has_pending_follow_ups(&self) -> bool {
        self.pending_follow_ups.load(Ordering::Acquire) > 0
    }

    /// Wait until a follow-up message is posted.
    pub async fn wait_for_follow_up(&self) {
        self.follow_up_notify.notified().await;
    }

    /// Wait until the agent loop becomes idle (finishes running).
    ///
    /// The ordering here is intentional and not a TOCTOU race: `notified()` registers
    /// the waiter *before* checking `is_running`, so if the agent finishes between
    /// registration and the check, the notification is already captured by the future.
    pub async fn wait_for_idle(&self) {
        let notified = self.idle_notify.notified();
        if !self.is_running.load(Ordering::Acquire) {
            return;
        }
        notified.await;
    }

    /// Wait until the agent loop becomes idle, with a timeout.
    /// Returns `true` if idle was reached, `false` on timeout.
    pub async fn wait_for_idle_timeout(&self, timeout: std::time::Duration) -> bool {
        tokio::time::timeout(timeout, self.wait_for_idle())
            .await
            .is_ok()
    }

    /// Whether the agent loop is currently running.
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }
}
