//! Fleet event bus: child→parent event forwarding and child→parent
//! interaction routing.
//!
//! Subagents broadcast events on their own channel. The bus spawns a
//! forwarder task per spawn that wraps each event as
//! `AgentEvent::Subagent { agent_id, description, event }` and posts
//! it to the parent's broadcast.
//!
//! Subagents that emit `InteractionRequest`s (e.g. via `AskUserQuestion`
//! tools) send them on a per-spawn `mpsc` channel. The bus drains
//! that channel, stamps `agent_id`, and forwards to the parent's
//! interaction channel — so root-level UI sees a flat stream of
//! requests tagged with their originating agent.

use std::sync::Arc;

use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::core::interaction::InteractionRequest;
use crate::fleet::registry::Registry;
use crate::types::events::AgentEvent;

/// Default capacity of the per-subagent interaction router's mpsc
/// channel. Reached when a subagent issues many concurrent gated tool
/// calls before the host UI drains them. Override via
/// [`AgentManager::with_interaction_router_capacity`] when the host
/// expects bursts.
pub const DEFAULT_INTERACTION_ROUTER_CAPACITY: usize = 64;

/// Spawn a task that forwards `child_rx` events to `parent_tx`, wrapped
/// as `AgentEvent::Subagent`. Aborts on `Closed`; logs and continues
/// on `Lagged`.
///
/// Shutdown protocol: callers should signal `shutdown` and then `await`
/// the returned `JoinHandle` rather than `abort()`-ing it. On shutdown,
/// the forwarder enters a synchronous drain loop (`try_recv`) and
/// flushes every event already buffered in the broadcast receiver
/// before exiting. This closes a race where the actor emits a final
/// `TurnEnd` / `ToolExecutionEnd` (e.g. the closing turn of
/// `prompt_and_wait`) milliseconds before the lifecycle proceeds to
/// teardown: an immediate `abort()` would drop those buffered events
/// and the registry's `usage` / `tool_use_count` would systematically
/// under-count the final turn.
///
/// Before wrapping, the forwarder inspects each event for fleet
/// bookkeeping:
///   - `TurnEnd { usage, .. }` is accumulated onto the agent's
///     registry entry via [`Registry::record_turn_end`].
///   - `ToolExecutionEnd { .. }` increments the per-agent tool counter
///     via [`Registry::record_tool_use`]. Both errored and successful
///     tool calls are counted (the tool *was* invoked). See
///     [`Registry::record_tool_use`] for the semantic relationship
///     with `SubagentResult.tool_use_count`.
///
/// The registry handle is optional so headless test paths can skip
/// bookkeeping; in normal fleet flows it is always present.
pub fn spawn_event_forwarder(
    mut child_rx: broadcast::Receiver<AgentEvent>,
    parent_tx: broadcast::Sender<AgentEvent>,
    agent_id: String,
    description: String,
    registry: Option<Arc<Registry>>,
    shutdown: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // Helper: apply registry bookkeeping + forward upstream.
        let forward = |event: AgentEvent| {
            if let Some(reg) = registry.as_ref() {
                match &event {
                    AgentEvent::TurnEnd { usage, .. } => {
                        reg.record_turn_end(&agent_id, usage);
                    }
                    AgentEvent::ToolExecutionEnd { .. } => {
                        reg.record_tool_use(&agent_id);
                    }
                    _ => {}
                }
            }
            let _ = parent_tx.send(AgentEvent::Subagent {
                agent_id: agent_id.clone(),
                description: description.clone(),
                event: Box::new(event),
            });
        };

        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => break,
                res = child_rx.recv() => match res {
                    Ok(event) => forward(event),
                    Err(broadcast::error::RecvError::Closed) => break,
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(
                            agent_id = %agent_id,
                            dropped = n,
                            "subagent event stream lagged; dropped events will not reach the parent"
                        );
                    }
                }
            }
        }

        // Drain phase: synchronously pull any events already buffered
        // in this receiver's queue. This is what closes the race —
        // events emitted between the actor's last `send` and our
        // shutdown signal are still sitting in the receiver and would
        // be lost on a bare `abort()`.
        loop {
            match child_rx.try_recv() {
                Ok(event) => forward(event),
                Err(broadcast::error::TryRecvError::Lagged(n)) => {
                    tracing::warn!(
                        agent_id = %agent_id,
                        dropped = n,
                        "subagent event stream lagged during drain; dropped events will not reach the parent"
                    );
                }
                Err(_) => break, // Empty or Closed → done.
            }
        }
    })
}

/// Build a per-subagent interaction sender that stamps `agent_id` on
/// outgoing requests and forwards to the parent. Returns `None` when
/// there is no parent interaction channel (headless subagent).
///
/// The returned sender is wired into the subagent's `Frame`. Each
/// time the subagent's actor or a tool sends an `InteractionRequest`,
/// the spawned router pulls it, stamps `agent_id` (if not already
/// set), and forwards.
pub fn spawn_interaction_router(
    parent_tx: Option<mpsc::Sender<InteractionRequest>>,
    agent_id: String,
    capacity: usize,
) -> Option<mpsc::Sender<InteractionRequest>> {
    let parent_tx = parent_tx?;
    let (sub_tx, mut sub_rx) = mpsc::channel::<InteractionRequest>(capacity);
    tokio::spawn(async move {
        while let Some(mut req) = sub_rx.recv().await {
            if req.agent_id.is_none() {
                req.agent_id = Some(agent_id.clone());
            }
            if parent_tx.send(req).await.is_err() {
                break;
            }
        }
    });
    Some(sub_tx)
}
