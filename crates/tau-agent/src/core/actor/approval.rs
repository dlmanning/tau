//! Approval sub-machine: classifying tool calls against the policy,
//! awaiting pending approval gates, and finalizing the batch.

use std::collections::{HashMap, HashSet};

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use tau_ai::Message;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::core::approval::{ApprovalDecision, ToolRisk};
use crate::core::command::Command;
use crate::core::interaction::{InteractionKind, InteractionRequest, InteractionResponse};
use crate::core::state::{Frame, State, ToolCall};
use crate::core::tool::{ToolResult, send_event};
use crate::types::events::{AgentEvent, ToolApprovalOutcome};

use super::executing::spawn_group;
use super::{
    GateFuture, Phase, ToolPhase, Turn, TurnSub, finish_cancelled_batch, handle_busy_command,
};

#[allow(clippy::too_many_arguments)]
pub(super) async fn step_approval(
    tool_calls: Vec<ToolCall>,
    groups: Vec<Vec<usize>>,
    mut pre_results: HashMap<usize, (String, String, ToolResult)>,
    mut dispatch: HashSet<usize>,
    mut pending_gates: FuturesUnordered<GateFuture>,
    first_user_message: Option<Message>,
    state: &mut State,
    urgent_rx: &mut mpsc::Receiver<Command>,
    normal_rx: &mut mpsc::Receiver<Command>,
    prompt_cancel: &CancellationToken,
) -> Phase {
    loop {
        if pending_gates.is_empty() {
            return finalize_approval(
                state,
                tool_calls,
                groups,
                pre_results,
                &dispatch,
                first_user_message,
                prompt_cancel,
            );
        }
        tokio::select! {
            biased;
            _ = prompt_cancel.cancelled() => {
                return finish_cancelled_batch(state, &tool_calls, &mut pre_results);
            }
            Some(cmd) = urgent_rx.recv() => handle_busy_command(state, cmd),
            Some(cmd) = normal_rx.recv() => handle_busy_command(state, cmd),
            Some((idx, id, name, resp)) = pending_gates.next() => {
                apply_gate_response(state, idx, id, name, resp, &tool_calls, &mut pre_results, &mut dispatch);
            }
        }
    }
}

// ─── Approval gates ──────────────────────────────────────────────────

fn synth_rejection(tc: &ToolCall, reason: &str) -> (String, String, ToolResult) {
    (
        tc.id.clone(),
        tc.name.clone(),
        ToolResult::error(format!("Tool call rejected: {reason}")),
    )
}

/// Classify each tool call against the policy. Sends a `Typed
/// {schema_id: "tool.confirm"}` interaction request for each `Gate`
/// decision via `try_send` (saturated channel ⇒ synthetic rejection).
/// Returns the initial `ToolPhase::AwaitingApproval` — caller wraps
/// in a `Turn` so `first_user_message` lives on the outer struct.
pub(super) fn classify_and_enter_approval(
    state: &State,
    tool_calls: Vec<ToolCall>,
    groups: Vec<Vec<usize>>,
) -> ToolPhase {
    let mut pre_results: HashMap<usize, (String, String, ToolResult)> = HashMap::new();
    let mut dispatch: HashSet<usize> = HashSet::new();
    let pending_gates: FuturesUnordered<GateFuture> = FuturesUnordered::new();

    for (idx, tc) in tool_calls.iter().enumerate() {
        let tool = state
            .frame
            .tools
            .iter()
            .find(|t| t.name() == tc.name)
            .cloned();
        let risk = tool
            .as_ref()
            .map(|t| t.risk(&tc.args))
            .unwrap_or(ToolRisk::Local);
        let activity = tool
            .as_ref()
            .map(|t| t.activity_description(&tc.args))
            .unwrap_or_else(|| format!("Running {}", tc.name));

        match state
            .frame
            .approval_policy
            .classify(&tc.name, &tc.args, risk)
        {
            ApprovalDecision::Auto => {
                emit_resolved(
                    &state.frame,
                    &tc.id,
                    &tc.name,
                    ToolApprovalOutcome::AutoApproved,
                );
                dispatch.insert(idx);
            }
            ApprovalDecision::Reject(reason) => {
                reject_tool_call(&state.frame, tc, idx, &mut pre_results, &reason);
            }
            ApprovalDecision::Gate => {
                let Some(ref interaction_tx) = state.frame.interaction_tx else {
                    let reason = GateFailure::NoChannel.reason();
                    reject_tool_call(&state.frame, tc, idx, &mut pre_results, &reason);
                    continue;
                };
                let (response_tx, response_rx) = oneshot::channel();
                let request = InteractionRequest {
                    agent_id: None,
                    kind: InteractionKind::Typed {
                        schema_id: "tool.confirm".into(),
                        payload: serde_json::json!({
                            "tool_call_id": tc.id.clone(),
                            "tool_name": tc.name.clone(),
                            "arguments": tc.args.clone(),
                            "activity": activity,
                            "risk": risk,
                        }),
                    },
                    response_tx,
                };

                let interaction_tx = interaction_tx.clone();
                let id = tc.id.clone();
                let name = tc.name.clone();
                let timeout = state.frame.interaction_timeout;
                pending_gates.push(Box::pin(async move {
                    let resp = run_gate(interaction_tx, request, response_rx, timeout).await;
                    (idx, id, name, resp)
                }));
            }
        }
    }

    ToolPhase::AwaitingApproval {
        tool_calls,
        groups,
        pre_results,
        dispatch,
        pending_gates,
    }
}

/// Why a gate could not be presented to or resolved by the host.
/// Produces the rejection reason surfaced in
/// [`ToolApprovalOutcome::Rejected`] so hosts can tell these apart
/// from policy or user rejections.
enum GateFailure {
    /// No `interaction_tx` was configured on the agent.
    NoChannel,
    /// The host dropped its receiver end of the interaction channel
    /// (or its response sender).
    ChannelClosed,
    /// The configured round-trip timeout elapsed.
    Timeout(std::time::Duration),
}

impl GateFailure {
    fn reason(&self) -> String {
        match self {
            GateFailure::NoChannel => "no interaction channel".to_string(),
            GateFailure::ChannelClosed => "interaction channel closed".to_string(),
            GateFailure::Timeout(dur) => format!("interaction timed out after {dur:?}"),
        }
    }
}

/// Reject a tool call before dispatch: emit the observable
/// [`ToolApprovalOutcome::Rejected`] and record a synthetic error
/// result so the batch's committed `tool_use` blocks stay answered.
fn reject_tool_call(
    frame: &Frame,
    tc: &ToolCall,
    idx: usize,
    pre_results: &mut HashMap<usize, (String, String, ToolResult)>,
    reason: &str,
) {
    emit_resolved(
        frame,
        &tc.id,
        &tc.name,
        ToolApprovalOutcome::Rejected {
            reason: reason.to_string(),
        },
    );
    pre_results.insert(idx, synth_rejection(tc, reason));
}

/// Send the gate request and await the host's response.
///
/// The send waits for channel capacity — a momentarily full interaction
/// channel applies backpressure instead of silently rejecting the tool
/// (the actor's select loop in `step_approval` stays responsive to
/// commands and abort while gates are pending). The timeout, when
/// configured, bounds the whole round-trip: capacity wait plus host
/// think time.
///
/// Failures resolve to `Ok(Rejected { reason })` with a
/// [`GateFailure`]-attributed reason so the host can distinguish
/// "channel closed" from "timed out" from a user rejection. The
/// `response_rx` is dropped on timeout; a late host reply fails
/// silently, matching the existing "host replied after the actor moved
/// on" semantics.
async fn run_gate(
    interaction_tx: mpsc::Sender<InteractionRequest>,
    request: InteractionRequest,
    response_rx: oneshot::Receiver<InteractionResponse>,
    timeout: Option<std::time::Duration>,
) -> std::result::Result<InteractionResponse, oneshot::error::RecvError> {
    let round_trip = async move {
        if interaction_tx.send(request).await.is_err() {
            return Ok(InteractionResponse::Rejected {
                reason: GateFailure::ChannelClosed.reason(),
            });
        }
        response_rx.await
    };
    match timeout {
        Some(dur) => match tokio::time::timeout(dur, round_trip).await {
            Ok(r) => r,
            Err(_elapsed) => {
                tracing::warn!(
                    timeout_ms = dur.as_millis() as u64,
                    "interaction request timed out; rejecting tool call"
                );
                Ok(InteractionResponse::Rejected {
                    reason: GateFailure::Timeout(dur).reason(),
                })
            }
        },
        None => round_trip.await,
    }
}

fn emit_resolved(frame: &Frame, id: &str, name: &str, outcome: ToolApprovalOutcome) {
    send_event(
        &frame.event_tx,
        AgentEvent::ToolApprovalResolved {
            tool_call_id: id.into(),
            tool_name: name.into(),
            outcome,
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn apply_gate_response(
    state: &State,
    idx: usize,
    id: String,
    name: String,
    resp: std::result::Result<InteractionResponse, oneshot::error::RecvError>,
    tool_calls: &[ToolCall],
    pre_results: &mut HashMap<usize, (String, String, ToolResult)>,
    dispatch: &mut HashSet<usize>,
) {
    let outcome = match resp {
        Ok(InteractionResponse::Approved { .. }) => Ok(()),
        Ok(InteractionResponse::Rejected { reason }) => Err(reason),
        Ok(InteractionResponse::Cancelled) => Err("cancelled".into()),
        Ok(InteractionResponse::Answer(_)) => Err("unexpected response to tool.confirm".into()),
        Err(_) => Err(GateFailure::ChannelClosed.reason()),
    };
    match outcome {
        Ok(()) => {
            emit_resolved(&state.frame, &id, &name, ToolApprovalOutcome::Approved);
            dispatch.insert(idx);
        }
        Err(reason) => {
            reject_tool_call(&state.frame, &tool_calls[idx], idx, pre_results, &reason);
        }
    }
}

fn finalize_approval(
    state: &State,
    tool_calls: Vec<ToolCall>,
    groups: Vec<Vec<usize>>,
    pre_results: HashMap<usize, (String, String, ToolResult)>,
    dispatch: &HashSet<usize>,
    first_user_message: Option<Message>,
    cancel: &CancellationToken,
) -> Phase {
    let mut filtered: Vec<Vec<usize>> = groups
        .into_iter()
        .map(|g| g.into_iter().filter(|i| dispatch.contains(i)).collect())
        .filter(|g: &Vec<usize>| !g.is_empty())
        .collect();

    if filtered.is_empty() {
        return Phase::Turn(Turn {
            first_user_message,
            sub: TurnSub::Tool(ToolPhase::Applying {
                tool_calls,
                results_map: pre_results,
            }),
        });
    }

    let first = filtered.remove(0);
    let join_set = spawn_group(state, &tool_calls, &first, cancel);
    Phase::Turn(Turn {
        first_user_message,
        sub: TurnSub::Tool(ToolPhase::Executing {
            join_set,
            remaining_groups: filtered,
            all_tool_calls: tool_calls,
            results_map: pre_results,
        }),
    })
}
