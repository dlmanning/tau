//! A cloneable handle for poking the agent from external code.

use parking_lot::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
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
    pub(crate) idle_notify: Arc<tokio::sync::Notify>,
    pub(crate) is_running: Arc<AtomicBool>,
}

impl AgentHandle {
    pub(crate) fn new() -> Self {
        Self {
            cancel: Arc::new(Mutex::new(CancellationToken::new())),
            steering_queue: Arc::new(Mutex::new(Vec::new())),
            follow_up_queue: Arc::new(Mutex::new(Vec::new())),
            idle_notify: Arc::new(tokio::sync::Notify::new()),
            is_running: Arc::new(AtomicBool::new(false)),
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
            tracing::warn!("Steering queue full ({} messages), dropping oldest", Self::MAX_QUEUE_SIZE);
            q.remove(0);
        }
        q.push(message);
    }

    /// Enqueue a follow-up message consumed after the loop finishes.
    pub fn follow_up(&self, message: Message) {
        let mut q = self.follow_up_queue.lock();
        if q.len() >= Self::MAX_QUEUE_SIZE {
            tracing::warn!("Follow-up queue full ({} messages), dropping oldest", Self::MAX_QUEUE_SIZE);
            q.remove(0);
        }
        q.push(message);
    }

    /// Wait until the agent loop becomes idle (finishes running).
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
        if !self.is_running.load(Ordering::Acquire) {
            return true;
        }
        tokio::time::timeout(timeout, self.wait_for_idle())
            .await
            .is_ok()
    }

    /// Whether the agent loop is currently running.
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }
}
