//! Actor task and nested phase machine.
//!
//! The actor task owns [`State`] exclusively. The outer loop dispatches
//! on [`Phase`] — a four-variant enum (`Idle`, `Turn`, `Compaction`,
//! `Done`). Each non-trivial outer variant has a sub-machine:
//!
//! - [`Turn`] / [`TurnSub`] — the body of a prompt: preparing, awaiting
//!   the model, processing the response, running tools, draining
//!   follow-up queues.
//! - [`ToolPhase`] — sub-states inside `TurnSub::Tool`: awaiting
//!   approval gates, executing tool tasks, applying results.
//! - [`DrainPhase`] — sub-states inside `TurnSub::Drain`: checking the
//!   steering/follow-up queues, blocking on background follow-ups.
//! - [`CompactionTrigger`] — the two ways compaction enters as a
//!   top-level phase (manual `Command::Compact`, or overflow recovery
//!   with a turn to resume after). Proactive (threshold-based)
//!   compaction is run inline in `step_processing` without going
//!   through `Phase::Compaction`.
//!
//! Pure decisions live in [`crate::core::transitions`]; all async I/O
//! lives here.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use tau_ai::Message;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::core::approval::{ApprovalDecision, ToolRisk};
use crate::core::command::{Command, PromptResult};
use crate::core::interaction::{InteractionKind, InteractionRequest, InteractionResponse};
use crate::core::state::{Frame, State, ToolCall};
use crate::core::stream::{StreamOutcome, StreamReducer};
use crate::core::tool::{ExecutionContext, ProgressSender, ToolResult, send_event};
use crate::core::transitions as t;
use crate::core::transport::AgentEventStream;
use crate::types::events::{AgentEvent, CompactionReason, ToolApprovalOutcome};

/// Future yielded by a pending approval gate.
type GateFuture = BoxFuture<
    'static,
    (
        usize,
        String,
        String,
        std::result::Result<InteractionResponse, oneshot::error::RecvError>,
    ),
>;

// ─── Phase: outer state machine ─────────────────────────────────────

enum Phase {
    /// No prompt active; block on commands.
    Idle,

    /// A prompt is in progress; the [`Turn`] sub-machine handles the
    /// body (prepare → model → response → tools → drain).
    Turn(Turn),

    /// Compaction runs as its own top-level phase, distinct from
    /// `Turn`, because it has two distinct shapes (manual-from-idle,
    /// overflow-recovery-from-turn) and its own success/failure
    /// transitions.
    Compaction(CompactionTrigger),

    /// Prompt finished (success or error). The outer loop emits
    /// `AgentEnd`, replies to the prompt sender, and transitions back
    /// to `Idle`.
    Done(Result<(), crate::types::error::Error>),
}

/// In-progress prompt body. `first_user_message` lives here, on the
/// `Turn` itself, so it doesn't have to be threaded through every
/// sub-variant.
struct Turn {
    first_user_message: Option<Message>,
    sub: TurnSub,
}

enum TurnSub {
    /// About to call the LLM.
    Prepare { pending: Vec<Message> },
    /// Stream is open; observing events.
    AwaitingModel {
        stream: AgentEventStream,
        pending: Vec<Message>,
    },
    /// Stream finished; deciding what to do next.
    Processing {
        outcome: Box<StreamOutcome>,
        pending: Vec<Message>,
    },
    /// Inside a tool batch (approval / execute / apply).
    Tool(ToolPhase),
    /// After a no-tool turn: check steering / follow-up queues.
    Drain(DrainPhase),
}

enum ToolPhase {
    /// Classifying tool calls; awaiting any pending approval gates.
    AwaitingApproval {
        tool_calls: Vec<ToolCall>,
        groups: Vec<Vec<usize>>,
        pre_results: HashMap<usize, (String, String, ToolResult)>,
        dispatch: HashSet<usize>,
        pending_gates: FuturesUnordered<GateFuture>,
    },
    /// Tool tasks running on a `JoinSet`.
    Executing {
        join_set: tokio::task::JoinSet<(usize, String, String, ToolResult)>,
        remaining_groups: Vec<Vec<usize>>,
        all_tool_calls: Vec<ToolCall>,
        results_map: HashMap<usize, (String, String, ToolResult)>,
    },
    /// All tool results collected; commit and prepare the next turn.
    Applying {
        tool_calls: Vec<ToolCall>,
        results_map: HashMap<usize, (String, String, ToolResult)>,
    },
}

enum DrainPhase {
    /// Synchronous check of the steering / follow-up queues.
    CheckQueues,
    /// All queues drained but background subagents are in flight;
    /// block on the urgent channel waiting for a follow-up to arrive.
    WaitingForBackground,
}

enum CompactionTrigger {
    /// Manual `Command::Compact` issued from idle. Replies on the
    /// embedded oneshot when compaction finishes.
    Manual {
        reply: oneshot::Sender<PromptResult>,
    },
    /// Overflow recovery mid-prompt. After compaction succeeds, the
    /// prompt resumes with `resume_pending` as the next turn's
    /// pending messages.
    Overflow {
        resume_pending: (Vec<Message>, Option<Message>),
    },
}

// ─── Entry point ────────────────────────────────────────────────────

pub(crate) async fn run_actor(
    mut state: State,
    mut urgent_rx: mpsc::Receiver<Command>,
    mut normal_rx: mpsc::Receiver<Command>,
) {
    let mut phase = Phase::Idle;
    let mut prompt_reply: Option<oneshot::Sender<PromptResult>> = None;
    let mut turn_number: u32 = 0;
    let mut prompt_cancel = CancellationToken::new();

    loop {
        phase = match phase {
            Phase::Idle => {
                tokio::select! {
                    biased;
                    Some(cmd) = urgent_rx.recv() => {
                        handle_idle_command(&mut state, cmd, &mut prompt_reply, &mut turn_number, &mut prompt_cancel)
                    }
                    Some(cmd) = normal_rx.recv() => {
                        handle_idle_command(&mut state, cmd, &mut prompt_reply, &mut turn_number, &mut prompt_cancel)
                    }
                    else => break,
                }
            }
            Phase::Turn(turn) => {
                step_turn(
                    turn,
                    &mut state,
                    &mut urgent_rx,
                    &mut normal_rx,
                    &prompt_cancel,
                    &mut turn_number,
                )
                .await
            }
            Phase::Compaction(trigger) => {
                step_compaction(trigger, &mut state, &prompt_cancel, &mut turn_number).await
            }
            Phase::Done(result) => {
                emit_end_and_idle(&mut state, result, &mut prompt_reply, &mut turn_number)
            }
        };
    }
}

fn emit_end_and_idle(
    state: &mut State,
    result: Result<(), crate::types::error::Error>,
    prompt_reply: &mut Option<oneshot::Sender<PromptResult>>,
    turn_number: &mut u32,
) -> Phase {
    state.shared.is_running.store(false, Ordering::Release);
    state.conv.conversation.is_streaming = false;
    // `interrupted` is true when a graceful interrupt was observed at
    // the top of a turn. We swap the flag atomically so the reset is
    // observable even if a stale `interrupt()` raced the prompt end.
    let interrupted = state
        .shared
        .interrupt_requested
        .swap(false, Ordering::AcqRel);
    send_event(
        &state.frame.event_tx,
        AgentEvent::AgentEnd {
            total_turns: *turn_number,
            total_usage: state.conv.conversation.total_usage.clone(),
            interrupted,
        },
    );
    if let Some(reply) = prompt_reply.take() {
        let _ = reply.send(PromptResult { result });
    }
    *turn_number = 0;
    Phase::Idle
}

// ─── Turn sub-machine ───────────────────────────────────────────────

async fn step_turn(
    turn: Turn,
    state: &mut State,
    urgent_rx: &mut mpsc::Receiver<Command>,
    normal_rx: &mut mpsc::Receiver<Command>,
    prompt_cancel: &CancellationToken,
    turn_number: &mut u32,
) -> Phase {
    let Turn {
        first_user_message,
        sub,
    } = turn;
    match sub {
        TurnSub::Prepare { pending } => {
            step_prepare(
                pending,
                first_user_message,
                state,
                prompt_cancel,
                turn_number,
            )
            .await
        }
        TurnSub::AwaitingModel { stream, pending } => {
            step_awaiting_model(
                stream,
                pending,
                first_user_message,
                state,
                urgent_rx,
                normal_rx,
            )
            .await
        }
        TurnSub::Processing { outcome, pending } => {
            step_processing(*outcome, pending, first_user_message, state, prompt_cancel).await
        }
        TurnSub::Tool(tp) => {
            step_tool(
                tp,
                first_user_message,
                state,
                urgent_rx,
                normal_rx,
                prompt_cancel,
            )
            .await
        }
        TurnSub::Drain(dp) => step_drain(dp, state, urgent_rx, normal_rx, prompt_cancel).await,
    }
}

// ─── Prepare ────────────────────────────────────────────────────────

async fn step_prepare(
    pending: Vec<Message>,
    first_user_message: Option<Message>,
    state: &mut State,
    cancel: &CancellationToken,
    turn_number: &mut u32,
) -> Phase {
    if cancel.is_cancelled() {
        return Phase::Done(Ok(()));
    }
    // Graceful interrupt: check between turns, after any prior tool
    // batch has completed and before requesting the next LLM turn.
    // Mid-stream and mid-tool work are NOT interrupted here — that is
    // the responsibility of `abort()`.
    if state.shared.interrupt_requested.load(Ordering::Acquire) {
        return Phase::Done(Ok(()));
    }
    if let Some(max) = state.frame.config.max_turns {
        if *turn_number >= max {
            run_final_summary(state, &pending, *turn_number, cancel).await;
            return Phase::Done(Ok(()));
        }
    }

    *turn_number += 1;
    let context = t::build_context(&state.frame, &state.conv, &pending);
    let run_config = t::build_run_config(&state.frame, *turn_number);

    match state
        .frame
        .transport
        .run(context, &run_config, cancel.clone())
        .await
    {
        Ok(stream) => Phase::Turn(Turn {
            first_user_message,
            sub: TurnSub::AwaitingModel { stream, pending },
        }),
        Err(e) => {
            let error_msg = e.to_string();
            let overflow =
                e.is_context_overflow() || crate::core::overflow::is_context_overflow(&error_msg);
            if overflow && state.frame.config.compaction.enabled {
                t::apply_pending(&mut state.conv, &pending);
                Phase::Compaction(CompactionTrigger::Overflow {
                    resume_pending: (
                        first_user_message.iter().cloned().collect(),
                        first_user_message,
                    ),
                })
            } else {
                state.conv.conversation.error = Some(error_msg.clone());
                send_event(
                    &state.frame.event_tx,
                    AgentEvent::Error { message: error_msg },
                );
                Phase::Done(Err(crate::types::error::Error::Ai(e)))
            }
        }
    }
}

// ─── Awaiting model ─────────────────────────────────────────────────

async fn step_awaiting_model(
    mut stream: AgentEventStream,
    pending: Vec<Message>,
    first_user_message: Option<Message>,
    state: &mut State,
    urgent_rx: &mut mpsc::Receiver<Command>,
    normal_rx: &mut mpsc::Receiver<Command>,
) -> Phase {
    let mut reducer = StreamReducer::default();
    loop {
        tokio::select! {
            biased;
            Some(cmd) = urgent_rx.recv() => handle_busy_command(state, cmd),
            Some(cmd) = normal_rx.recv() => handle_busy_command(state, cmd),
            event = stream.next() => match event {
                Some(e) => {
                    send_event(&state.frame.event_tx, e.clone());
                    reducer.observe(&e);
                }
                None => break Phase::Turn(Turn {
                    first_user_message,
                    sub: TurnSub::Processing {
                        outcome: Box::new(reducer.finalize()),
                        pending,
                    },
                }),
            }
        }
    }
}

// ─── Processing ─────────────────────────────────────────────────────

async fn step_processing(
    outcome: StreamOutcome,
    pending: Vec<Message>,
    first_user_message: Option<Message>,
    state: &mut State,
    cancel: &CancellationToken,
) -> Phase {
    let decision = t::decide_response_action(&state.frame, &outcome, first_user_message.clone());

    let action = match decision.action {
        t::ResponseAction::Compact {
            reason: _,
            resume_pending,
        } => {
            // Overflow → compaction: force-commit pending so it
            // survives compaction and is in history when the prompt
            // resumes.
            t::apply_partial_on_error(
                &mut state.conv,
                &outcome,
                &pending,
                /* force_pending */ true,
            );
            return Phase::Compaction(CompactionTrigger::Overflow {
                resume_pending: resume_pending.unwrap_or_else(|| (Vec::new(), None)),
            });
        }
        t::ResponseAction::Error(e) => {
            t::apply_partial_on_error(
                &mut state.conv,
                &outcome,
                &pending,
                /* force_pending */ false,
            );
            let msg = e.to_string();
            state.conv.conversation.error = Some(msg.clone());
            send_event(&state.frame.event_tx, AgentEvent::Error { message: msg });
            return Phase::Done(Err(e));
        }
        action => action,
    };

    // Success path: commit pending + assistant message + usage.
    t::apply_response(&mut state.conv, outcome, &pending);

    // Proactive (threshold-based) compaction runs inline.
    if decision.needs_proactive_compaction {
        run_proactive_compaction(state, cancel).await;
    }

    match action {
        t::ResponseAction::RunTools {
            tool_calls,
            groups,
            first_user_message,
        } => {
            let tp = classify_and_enter_approval(state, tool_calls, groups);
            Phase::Turn(Turn {
                first_user_message,
                sub: TurnSub::Tool(tp),
            })
        }
        t::ResponseAction::Done => Phase::Turn(Turn {
            first_user_message,
            sub: TurnSub::Drain(DrainPhase::CheckQueues),
        }),
        t::ResponseAction::Compact { .. } | t::ResponseAction::Error(_) => unreachable!(),
    }
}

// ─── Tool sub-machine ───────────────────────────────────────────────

async fn step_tool(
    tool_phase: ToolPhase,
    first_user_message: Option<Message>,
    state: &mut State,
    urgent_rx: &mut mpsc::Receiver<Command>,
    normal_rx: &mut mpsc::Receiver<Command>,
    prompt_cancel: &CancellationToken,
) -> Phase {
    match tool_phase {
        ToolPhase::AwaitingApproval {
            tool_calls,
            groups,
            pre_results,
            dispatch,
            pending_gates,
        } => {
            step_approval(
                tool_calls,
                groups,
                pre_results,
                dispatch,
                pending_gates,
                first_user_message,
                state,
                urgent_rx,
                normal_rx,
                prompt_cancel,
            )
            .await
        }
        ToolPhase::Executing {
            join_set,
            remaining_groups,
            all_tool_calls,
            results_map,
        } => {
            step_executing(
                join_set,
                remaining_groups,
                all_tool_calls,
                results_map,
                first_user_message,
                state,
                urgent_rx,
                normal_rx,
                prompt_cancel,
            )
            .await
        }
        ToolPhase::Applying {
            tool_calls,
            mut results_map,
        } => {
            t::apply_tool_results(&mut state.conv, &tool_calls, &mut results_map);
            Phase::Turn(Turn {
                first_user_message,
                sub: TurnSub::Prepare { pending: vec![] },
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn step_approval(
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
            _ = prompt_cancel.cancelled() => return Phase::Done(Ok(())),
            Some(cmd) = urgent_rx.recv() => handle_busy_command(state, cmd),
            Some(cmd) = normal_rx.recv() => handle_busy_command(state, cmd),
            Some((idx, id, name, resp)) = pending_gates.next() => {
                apply_gate_response(state, idx, id, name, resp, &tool_calls, &mut pre_results, &mut dispatch);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn step_executing(
    mut join_set: tokio::task::JoinSet<(usize, String, String, ToolResult)>,
    remaining_groups: Vec<Vec<usize>>,
    all_tool_calls: Vec<ToolCall>,
    mut results_map: HashMap<usize, (String, String, ToolResult)>,
    first_user_message: Option<Message>,
    state: &mut State,
    urgent_rx: &mut mpsc::Receiver<Command>,
    normal_rx: &mut mpsc::Receiver<Command>,
    prompt_cancel: &CancellationToken,
) -> Phase {
    loop {
        tokio::select! {
            biased;
            _ = prompt_cancel.cancelled() => {
                // JoinSet aborts spawned tasks on drop.
                return Phase::Done(Ok(()));
            }
            Some(cmd) = urgent_rx.recv() => handle_busy_command(state, cmd),
            Some(cmd) = normal_rx.recv() => handle_busy_command(state, cmd),
            result = join_set.join_next() => match result {
                Some(Ok((idx, id, name, tool_result))) => {
                    results_map.insert(idx, (id, name, tool_result));
                }
                Some(Err(join_err)) => {
                    tracing::error!("Tool task panicked: {}", join_err);
                }
                None => {
                    let action = t::decide_batch_complete(
                        &state.frame, &state.conv, &remaining_groups, &all_tool_calls,
                    );
                    return match action {
                        t::BatchCompleteAction::Redirect { skipped_indices } => {
                            // Inject skipped synth-results, commit, drain, redirect.
                            for (idx, id, name) in &skipped_indices {
                                let skip = ToolResult::error("Skipped due to steering message");
                                results_map.insert(*idx, (id.clone(), name.clone(), skip));
                                send_event(&state.frame.event_tx, AgentEvent::ToolExecutionStart {
                                    tool_call_id: id.clone(),
                                    tool_name: name.clone(),
                                    arguments: serde_json::Value::Null,
                                    activity: "Skipped".into(),
                                });
                                send_event(&state.frame.event_tx, AgentEvent::ToolExecutionEnd {
                                    tool_call_id: id.clone(),
                                    tool_name: name.clone(),
                                    result: "Skipped due to steering message".into(),
                                    is_error: true,
                                });
                            }
                            t::apply_tool_results(&mut state.conv, &all_tool_calls, &mut results_map);
                            // Redirect path is steering-driven; FollowUps drains
                            // happen only at the post-turn DrainFollowUps phase.
                            let drained = t::apply_drain_queues(&state.frame, &mut state.conv);
                            Phase::Turn(Turn {
                                first_user_message: None,
                                sub: TurnSub::Prepare { pending: drained.messages },
                            })
                        }
                        t::BatchCompleteAction::AllGroupsDone => Phase::Turn(Turn {
                            first_user_message,
                            sub: TurnSub::Tool(ToolPhase::Applying {
                                tool_calls: all_tool_calls,
                                results_map,
                            }),
                        }),
                        t::BatchCompleteAction::NextGroup(next_group) => {
                            let mut remaining = remaining_groups;
                            remaining.remove(0);
                            let new_join = spawn_group(state, &all_tool_calls, &next_group, prompt_cancel);
                            Phase::Turn(Turn {
                                first_user_message,
                                sub: TurnSub::Tool(ToolPhase::Executing {
                                    join_set: new_join,
                                    remaining_groups: remaining,
                                    all_tool_calls,
                                    results_map,
                                }),
                            })
                        }
                    };
                }
            }
        }
    }
}

// ─── Drain sub-machine ──────────────────────────────────────────────

async fn step_drain(
    dp: DrainPhase,
    state: &mut State,
    urgent_rx: &mut mpsc::Receiver<Command>,
    normal_rx: &mut mpsc::Receiver<Command>,
    prompt_cancel: &CancellationToken,
) -> Phase {
    match dp {
        DrainPhase::CheckQueues => {
            let drained = t::apply_drain_queues(&state.frame, &mut state.conv);
            match drained.source {
                t::DrainedFrom::Steering => Phase::Turn(Turn {
                    first_user_message: None,
                    sub: TurnSub::Prepare {
                        pending: drained.messages,
                    },
                }),
                t::DrainedFrom::FollowUps => {
                    let count = drained.messages.len() as u32;
                    let _ = state.shared.pending_follow_ups.fetch_update(
                        Ordering::Release,
                        Ordering::Acquire,
                        |n| Some(n.saturating_sub(count)),
                    );
                    Phase::Turn(Turn {
                        first_user_message: None,
                        sub: TurnSub::Prepare {
                            pending: drained.messages,
                        },
                    })
                }
                t::DrainedFrom::Nothing => {
                    if state.shared.pending_follow_ups.load(Ordering::Acquire) > 0 {
                        Phase::Turn(Turn {
                            first_user_message: None,
                            sub: TurnSub::Drain(DrainPhase::WaitingForBackground),
                        })
                    } else {
                        Phase::Done(Ok(()))
                    }
                }
            }
        }
        DrainPhase::WaitingForBackground => loop {
            tokio::select! {
                biased;
                _ = prompt_cancel.cancelled() => break Phase::Done(Ok(())),
                Some(cmd) = urgent_rx.recv() => match cmd {
                    Command::FollowUp(msg) => {
                        state.conv.follow_up_queue.push(msg);
                        break Phase::Turn(Turn {
                            first_user_message: None,
                            sub: TurnSub::Drain(DrainPhase::CheckQueues),
                        });
                    }
                    other => handle_busy_command(state, other),
                },
                Some(cmd) = normal_rx.recv() => handle_busy_command(state, cmd),
            }
        },
    }
}

// ─── Compaction ─────────────────────────────────────────────────────

async fn step_compaction(
    trigger: CompactionTrigger,
    state: &mut State,
    cancel: &CancellationToken,
    turn_number: &mut u32,
) -> Phase {
    let reason = match &trigger {
        CompactionTrigger::Manual { .. } => CompactionReason::Manual,
        CompactionTrigger::Overflow { .. } => CompactionReason::Overflow,
    };
    send_event(
        &state.frame.event_tx,
        AgentEvent::CompactionStart { reason },
    );

    let result = crate::core::compaction::compact(
        &state.conv.conversation.messages,
        &state.frame.config.compaction,
        &state.frame.config,
        &state.frame.transport,
        state.conv.conversation.previous_summary.as_deref(),
        cancel,
    )
    .await;

    match result {
        Ok(cr) => {
            let tokens_after = crate::core::compaction::estimate_total_tokens(
                &state.conv.conversation.messages[cr.first_kept_index..],
            );
            send_event(
                &state.frame.event_tx,
                AgentEvent::CompactionEnd {
                    tokens_before: cr.tokens_before,
                    tokens_after,
                },
            );
            crate::core::compaction::apply_compaction_result(
                &mut state.conv.conversation.messages,
                &mut state.conv.conversation.previous_summary,
                cr,
            );

            match trigger {
                CompactionTrigger::Manual { reply } => {
                    let _ = reply.send(PromptResult { result: Ok(()) });
                    Phase::Idle
                }
                CompactionTrigger::Overflow {
                    resume_pending: (pending, first_user_message),
                } => {
                    // Overflow recovery — reset turns and retry.
                    *turn_number = 0;
                    Phase::Turn(Turn {
                        first_user_message,
                        sub: TurnSub::Prepare { pending },
                    })
                }
            }
        }
        Err(e) => {
            send_event(
                &state.frame.event_tx,
                AgentEvent::Error {
                    message: format!("Compaction failed: {e}"),
                },
            );
            match trigger {
                CompactionTrigger::Manual { reply } => {
                    let _ = reply.send(PromptResult {
                        result: Err(crate::types::error::Error::Compaction(e)),
                    });
                    Phase::Idle
                }
                CompactionTrigger::Overflow { .. } => {
                    Phase::Done(Err(crate::types::error::Error::Compaction(e)))
                }
            }
        }
    }
}

/// Best-effort proactive compaction run inline after a successful
/// turn whose usage signaled the threshold. Failures are logged and
/// the loop continues — the conversation isn't lost, it just stays
/// larger than ideal until the next opportunity.
async fn run_proactive_compaction(state: &mut State, cancel: &CancellationToken) {
    send_event(
        &state.frame.event_tx,
        AgentEvent::CompactionStart {
            reason: CompactionReason::Threshold,
        },
    );
    let result = crate::core::compaction::compact(
        &state.conv.conversation.messages,
        &state.frame.config.compaction,
        &state.frame.config,
        &state.frame.transport,
        state.conv.conversation.previous_summary.as_deref(),
        cancel,
    )
    .await;
    match result {
        Ok(cr) => {
            let tokens_after = crate::core::compaction::estimate_total_tokens(
                &state.conv.conversation.messages[cr.first_kept_index..],
            );
            send_event(
                &state.frame.event_tx,
                AgentEvent::CompactionEnd {
                    tokens_before: cr.tokens_before,
                    tokens_after,
                },
            );
            crate::core::compaction::apply_compaction_result(
                &mut state.conv.conversation.messages,
                &mut state.conv.conversation.previous_summary,
                cr,
            );
        }
        Err(e) => tracing::warn!("Proactive compaction failed: {e}"),
    }
}

// ─── Command handling ────────────────────────────────────────────────

fn handle_idle_command(
    state: &mut State,
    cmd: Command,
    prompt_reply: &mut Option<oneshot::Sender<PromptResult>>,
    turn_number: &mut u32,
    prompt_cancel: &mut CancellationToken,
) -> Phase {
    match cmd {
        Command::Prompt { content, reply } => {
            *prompt_reply = Some(reply);
            state.shared.is_running.store(true, Ordering::Release);
            // Clear any stale graceful-interrupt request so it does
            // not latch across prompts.
            state
                .shared
                .interrupt_requested
                .store(false, Ordering::Release);
            // Swap in a fresh cancellation token. Holding the lock
            // across the swap prevents `handle.abort()` from cancelling
            // the wrong token.
            let fresh = CancellationToken::new();
            {
                let mut guard = state.shared.cancel.lock();
                *prompt_cancel = fresh.clone();
                *guard = fresh;
            }
            *turn_number = 0;
            state.conv.conversation.is_streaming = true;
            state.conv.conversation.error = None;
            send_event(&state.frame.event_tx, AgentEvent::AgentStart);

            let user_message = Message::User {
                content: content.clone(),
                timestamp: chrono::Utc::now().timestamp_millis(),
            };
            Phase::Turn(Turn {
                first_user_message: Some(user_message.clone()),
                sub: TurnSub::Prepare {
                    pending: vec![user_message],
                },
            })
        }
        Command::Compact { reason: _, reply } => {
            // The `reason` field on Command::Compact is informational
            // for the host; the actor emits `CompactionStart` with
            // `Manual` since this entry point is manual by
            // construction.
            Phase::Compaction(CompactionTrigger::Manual { reply })
        }
        other => {
            handle_busy_command(state, other);
            Phase::Idle
        }
    }
}

fn handle_busy_command(state: &mut State, cmd: Command) {
    match cmd {
        Command::GetConfig(reply) => {
            let _ = reply.send(state.frame.config.clone());
        }
        Command::GetMessages(reply) => {
            let _ = reply.send(state.conv.conversation.messages.clone());
        }
        Command::GetState(reply) => {
            let _ = reply.send(state.conv.conversation.clone());
        }
        Command::SetModel(m) => state.frame.config.model = m,
        Command::SetReasoning(l) => state.frame.config.reasoning = l,
        Command::SetCompactionConfig(c) => state.frame.config.compaction = c,
        Command::SetApprovalPolicy(p) => state.frame.approval_policy = p,
        Command::Steer(msg) => state.conv.steering_queue.push(msg),
        Command::FollowUp(msg) => state.conv.follow_up_queue.push(msg),
        // Reject concurrent prompts / manual compactions.
        Command::Prompt { reply, .. } => {
            let _ = reply.send(PromptResult {
                result: Err(crate::types::error::Error::Busy),
            });
        }
        Command::Compact { reply, .. } => {
            let _ = reply.send(PromptResult {
                result: Err(crate::types::error::Error::Busy),
            });
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
fn classify_and_enter_approval(
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
                emit_resolved(
                    &state.frame,
                    &tc.id,
                    &tc.name,
                    ToolApprovalOutcome::Rejected {
                        reason: reason.clone(),
                    },
                );
                pre_results.insert(idx, synth_rejection(tc, &reason));
            }
            ApprovalDecision::Gate => {
                let Some(ref interaction_tx) = state.frame.interaction_tx else {
                    let reason = "no interaction channel".to_string();
                    emit_resolved(
                        &state.frame,
                        &tc.id,
                        &tc.name,
                        ToolApprovalOutcome::Rejected {
                            reason: reason.clone(),
                        },
                    );
                    pre_results.insert(idx, synth_rejection(tc, &reason));
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

                match interaction_tx.try_send(request) {
                    Ok(()) => {
                        let id = tc.id.clone();
                        let name = tc.name.clone();
                        pending_gates.push(Box::pin(async move {
                            let resp = response_rx.await;
                            (idx, id, name, resp)
                        }));
                    }
                    Err(_) => {
                        let reason = "interaction channel saturated or closed".to_string();
                        emit_resolved(
                            &state.frame,
                            &tc.id,
                            &tc.name,
                            ToolApprovalOutcome::Rejected {
                                reason: reason.clone(),
                            },
                        );
                        pre_results.insert(idx, synth_rejection(tc, &reason));
                    }
                }
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
        Err(_) => Err("interaction channel closed".into()),
    };
    match outcome {
        Ok(()) => {
            emit_resolved(&state.frame, &id, &name, ToolApprovalOutcome::Approved);
            dispatch.insert(idx);
        }
        Err(reason) => {
            emit_resolved(
                &state.frame,
                &id,
                &name,
                ToolApprovalOutcome::Rejected {
                    reason: reason.clone(),
                },
            );
            pre_results.insert(idx, synth_rejection(&tool_calls[idx], &reason));
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

// ─── Tool execution ─────────────────────────────────────────────────

fn spawn_group(
    state: &State,
    tool_calls: &[ToolCall],
    group: &[usize],
    cancel: &CancellationToken,
) -> tokio::task::JoinSet<(usize, String, String, ToolResult)> {
    let mut join_set = tokio::task::JoinSet::new();
    let cwd = state
        .conv
        .cwd
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    let agent_id = state.shared.agent_id.get().cloned();

    for &idx in group {
        let tc = &tool_calls[idx];
        let tool = state
            .frame
            .tools
            .iter()
            .find(|t| t.name() == tc.name)
            .cloned();
        let validator_and_schema = state.frame.schema_cache.get(&tc.name).cloned();
        let event_tx = state.frame.event_tx.clone();
        let cancel = cancel.clone();
        let file_access = state.frame.file_access.clone();
        let interaction_tx = state.frame.interaction_tx.clone();
        let cwd = cwd.clone();
        let agent_id = agent_id.clone();
        let subagent_depth = state.frame.subagent_depth;

        let id = tc.id.clone();
        let name = tc.name.clone();
        let args = tc.args.clone();

        let progress = ProgressSender::new(event_tx.clone(), &id, &name);
        let ctx = ExecutionContext {
            cwd,
            cancel,
            progress,
            interaction: interaction_tx,
            file_access,
            agent_id,
            subagent_depth,
        };

        join_set.spawn(async move {
            let result = run_single_tool(
                tool,
                id.clone(),
                name.clone(),
                args,
                validator_and_schema,
                event_tx,
                ctx,
            )
            .await;
            (idx, id, name, result)
        });
    }

    join_set
}

async fn run_single_tool(
    tool: Option<crate::core::tool::BoxedTool>,
    id: String,
    name: String,
    args: serde_json::Value,
    validator_and_schema: Option<(
        std::sync::Arc<jsonschema::Validator>,
        std::sync::Arc<serde_json::Value>,
    )>,
    event_tx: tokio::sync::broadcast::Sender<AgentEvent>,
    ctx: ExecutionContext,
) -> ToolResult {
    let activity = tool
        .as_ref()
        .map(|t| t.activity_description(&args))
        .unwrap_or_else(|| format!("Running {}", name));
    send_event(
        &event_tx,
        AgentEvent::ToolExecutionStart {
            tool_call_id: id.clone(),
            tool_name: name.clone(),
            arguments: args.clone(),
            activity,
        },
    );

    let result = if let Some(tool) = tool {
        let validation_error =
            validator_and_schema.and_then(|(v, schema)| validate_with(&args, &v, &schema));
        if let Some(err) = validation_error {
            ToolResult::error(err)
        } else {
            tool.execute(args, ctx).await
        }
    } else {
        ToolResult::error(format!("Tool not found: {}", name))
    };

    send_event(
        &event_tx,
        AgentEvent::ToolExecutionEnd {
            tool_call_id: id,
            tool_name: name,
            result: result.text_content(),
            is_error: result.is_error,
        },
    );
    result
}

fn validate_with(
    args: &serde_json::Value,
    validator: &jsonschema::Validator,
    schema: &serde_json::Value,
) -> Option<String> {
    let errs: Vec<String> = validator
        .iter_errors(args)
        .map(|e| {
            let p = e.instance_path().to_string();
            if p.is_empty() {
                e.to_string()
            } else {
                format!("{p}: {e}")
            }
        })
        .collect();
    if errs.is_empty() {
        return None;
    }
    // Include the schema so the LLM can self-correct on the next call.
    // Terse JSON keeps the token cost low; the model already sees this
    // schema in the tool definition but echoing it on failure has been
    // observed to break wrong-shape loops materially faster.
    let schema_str = serde_json::to_string(schema).unwrap_or_else(|_| "<unavailable>".into());
    Some(format!(
        "Tool argument validation failed:\n{}\nExpected schema: {}",
        errs.join("\n"),
        schema_str
    ))
}

// ─── Final summary on max-turns ─────────────────────────────────────

async fn run_final_summary(
    state: &mut State,
    pending: &[Message],
    turn_number: u32,
    cancel: &CancellationToken,
) {
    let last_has_tool_calls = state
        .conv
        .conversation
        .messages
        .last()
        .is_some_and(|m| !m.tool_calls().is_empty());
    if !last_has_tool_calls || pending.is_empty() {
        return;
    }
    let max = state.frame.config.max_turns.unwrap_or(0);
    tracing::info!("Agent reached max turns ({}), running final summary", max);

    t::apply_pending(&mut state.conv, pending);
    let summary_prompt = vec![Message::user(format!(
        "[System: You have reached the maximum of {} turns. Summarize your findings so far. Do not call any tools.]",
        max
    ))];

    let context = t::build_context(&state.frame, &state.conv, &summary_prompt);
    let mut final_config = t::build_run_config(&state.frame, turn_number);
    final_config.tools.clear();

    if let Ok(mut stream) = state
        .frame
        .transport
        .run(context, &final_config, cancel.clone())
        .await
    {
        let mut reducer = StreamReducer::default();
        while let Some(event) = stream.next().await {
            send_event(&state.frame.event_tx, event.clone());
            reducer.observe(&event);
        }
        let outcome = reducer.finalize();
        t::apply_usage(&mut state.conv, &outcome.usage);
        if let Some(msg) = outcome.assistant_message {
            state.conv.conversation.messages.push(msg);
        }
    }
}
