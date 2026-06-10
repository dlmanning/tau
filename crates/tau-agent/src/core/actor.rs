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

mod approval;
mod drain;
mod executing;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use tau_ai::Message;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use serde_json::Value as JsonValue;

use crate::core::approval::{ApprovalDecision, ToolRisk};
use crate::core::command::{Command, PromptResult};
use crate::core::compaction::estimate_total_tokens;
use crate::core::interaction::InteractionResponse;
use crate::core::state::{State, ToolCall};
use crate::core::stream::{StreamOutcome, StreamReducer};
use crate::core::tool::{ToolResult, send_event};
use crate::core::transitions as t;
use crate::core::transport::AgentEventStream;
use crate::types::events::{AgentEvent, CompactionReason};
use crate::types::info::{ContextStats, ToolInfo};

use approval::{classify_and_enter_approval, step_approval};
use drain::step_drain;
use executing::step_executing;

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
    /// embedded oneshot when compaction finishes. `custom_instructions`,
    /// when present, is appended to the summarization prompt as a
    /// `## User instructions` section.
    Manual {
        custom_instructions: Option<String>,
        reply: oneshot::Sender<PromptResult>,
    },
    /// Overflow recovery mid-prompt. Nothing is committed to history
    /// until compaction succeeds — if it fails, the prompt dies with
    /// the in-flight messages un-committed (re-presentable) instead of
    /// baked into a history that just proved too large.
    Overflow {
        /// `(pending, first_user_message)` for the resumed turn: the
        /// un-committed messages are re-presented against the
        /// compacted history.
        resume_pending: (Vec<Message>, Option<Message>),
        /// Messages committed to history immediately after a
        /// successful compaction, *before* the resume — e.g. a
        /// meaningful partial assistant message (together with the
        /// pending that precedes it) whose position must be preserved.
        commit_on_success: Vec<Message>,
    },
}

// ─── Entry point ────────────────────────────────────────────────────

pub(crate) async fn run_actor(
    mut state: State,
    mut urgent_rx: mpsc::Receiver<Command>,
    mut normal_rx: mpsc::Receiver<Command>,
    ready_tx: oneshot::Sender<crate::types::error::Result<()>>,
) {
    let mut phase = Phase::Idle;
    let mut prompt_reply: Option<oneshot::Sender<PromptResult>> = None;
    let mut turn_number: u32 = 0;
    let mut prompt_cancel = CancellationToken::new();

    // Any fallible async setup happens here, before we signal
    // readiness. Today there is none — the actor is ready as soon as
    // it enters the loop — but the channel exists so future startup
    // checks (transport pre-warm, tool init) can surface failures to
    // `AgentBuilder::spawn` without a side-channel.

    // Drop on RecvError is fine — the caller may have given up.
    let _ = ready_tx.send(Ok(()));

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
    state.shared.prompt_in_flight.store(false, Ordering::Release);
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
                // `pending` stays un-committed across compaction (see
                // `CompactionTrigger::Overflow`) and is re-presented by
                // the resumed turn — committing it first would bake it
                // into history even if compaction fails, and duplicate
                // it in the model's context when compaction succeeds.
                Phase::Compaction(CompactionTrigger::Overflow {
                    resume_pending: (pending, first_user_message),
                    commit_on_success: vec![],
                })
            } else {
                t::apply_error(&mut state.conv, &error_msg);
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
        t::ResponseAction::Compact => {
            // Defer all commits until compaction succeeds (see
            // `CompactionTrigger::Overflow`). A meaningful partial must
            // keep its position after the pending that precedes it, so
            // when one exists both go through `commit_on_success` and
            // the resume re-presents nothing; otherwise pending is
            // simply re-presented by the resumed turn.
            let (commit_on_success, resume_messages) = match t::meaningful_partial(&outcome) {
                Some(partial) => {
                    let mut commit = pending.clone();
                    commit.push(partial);
                    (commit, Vec::new())
                }
                None => (Vec::new(), pending.clone()),
            };
            return Phase::Compaction(CompactionTrigger::Overflow {
                resume_pending: (resume_messages, first_user_message),
                commit_on_success,
            });
        }
        t::ResponseAction::Error(e) => {
            t::apply_partial_on_error(&mut state.conv, &outcome, &pending);
            let msg = e.to_string();
            t::apply_error(&mut state.conv, &msg);
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
        t::ResponseAction::Compact | t::ResponseAction::Error(_) => unreachable!(),
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

/// Close out a tool batch on prompt cancellation. The assistant message
/// (committed in `step_processing`) carries `tool_use` blocks; every
/// tool-phase exit must answer them or the next prompt sends malformed
/// history to the provider (each `tool_use` must have a `tool_result`).
/// Calls with a result in `results` keep it (resolved/rejected gates,
/// completed tools); the rest — approved-but-unrun, still-pending gates,
/// aborted tools — get synthetic "cancelled" errors via
/// `collect_ordered_results`.
fn finish_cancelled_batch(
    state: &mut State,
    tool_calls: &[ToolCall],
    results: &mut HashMap<usize, (String, String, ToolResult)>,
) -> Phase {
    t::apply_tool_results(&mut state.conv, tool_calls, results);
    Phase::Done(Ok(()))
}

// ─── Compaction ─────────────────────────────────────────────────────

/// Run one compaction pass: emit `CompactionStart`, summarize via
/// [`compaction::compact`](crate::core::compaction::compact), and on
/// success emit `CompactionEnd` and commit the result to the
/// conversation. On failure nothing is committed and no event beyond
/// `CompactionStart` is emitted — what the failure *means* is the
/// caller's decision: fatal for forced compaction (`step_compaction`,
/// overflow/manual), logged-and-ignored for the proactive threshold
/// pass (`run_proactive_compaction`).
async fn run_compaction_pass(
    state: &mut State,
    reason: CompactionReason,
    custom_instructions: Option<&str>,
    cancel: &CancellationToken,
) -> Result<(), String> {
    send_event(
        &state.frame.event_tx,
        AgentEvent::CompactionStart { reason },
    );
    let cr = crate::core::compaction::compact(
        &state.conv.conversation.messages,
        &state.frame.config.compaction,
        &state.frame.config,
        &state.frame.transport,
        state.conv.conversation.previous_summary.as_deref(),
        custom_instructions,
        cancel,
    )
    .await?;
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
    Ok(())
}

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
    let custom_instructions: Option<&str> = match &trigger {
        CompactionTrigger::Manual {
            custom_instructions,
            ..
        } => custom_instructions.as_deref(),
        CompactionTrigger::Overflow { .. } => None,
    };

    match run_compaction_pass(state, reason, custom_instructions, cancel).await {
        Ok(()) => match trigger {
            CompactionTrigger::Manual { reply, .. } => {
                let _ = reply.send(PromptResult { result: Ok(()) });
                Phase::Idle
            }
            CompactionTrigger::Overflow {
                resume_pending: (pending, first_user_message),
                commit_on_success,
            } => {
                // Commit what the overflowing turn had produced
                // (e.g. a meaningful partial) now that the
                // compacted history has room, then reset turns
                // and retry.
                t::apply_pending(&mut state.conv, &commit_on_success);
                *turn_number = 0;
                Phase::Turn(Turn {
                    first_user_message,
                    sub: TurnSub::Prepare { pending },
                })
            }
        },
        Err(e) => {
            send_event(
                &state.frame.event_tx,
                AgentEvent::Error {
                    message: format!("Compaction failed: {e}"),
                },
            );
            match trigger {
                CompactionTrigger::Manual { reply, .. } => {
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
    if let Err(e) = run_compaction_pass(state, CompactionReason::Threshold, None, cancel).await {
        tracing::warn!("Proactive compaction failed: {e}");
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
            state.shared.prompt_in_flight.store(true, Ordering::Release);
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
            t::apply_clear_error(&mut state.conv);
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
        Command::Compact {
            custom_instructions,
            reply,
        } => Phase::Compaction(CompactionTrigger::Manual {
            custom_instructions,
            reply,
        }),
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
        Command::GetContextStats(reply) => {
            let used = estimate_total_tokens(&state.conv.conversation.messages);
            let limit = u64::from(state.frame.config.model.context_window);
            let remaining = limit.saturating_sub(used);
            let _ = reply.send(ContextStats {
                used,
                remaining,
                limit,
                updated_at: chrono::Utc::now(),
            });
        }
        Command::ListTools(reply) => {
            // `currently_allowed` = the policy would auto-dispatch a
            // no-argument call. `Gate` and `Reject(_)` both surface as
            // not auto-allowed.
            let infos: Vec<ToolInfo> = state
                .frame
                .tools
                .iter()
                .map(|tool| {
                    let risk = tool.risk(&JsonValue::Null);
                    let default_allowed = matches!(risk, ToolRisk::Safe | ToolRisk::Local);
                    let currently_allowed = matches!(
                        state.frame.approval_policy.classify(
                            tool.name(),
                            &JsonValue::Null,
                            risk,
                        ),
                        ApprovalDecision::Auto
                    );
                    ToolInfo {
                        name: tool.name().to_string(),
                        description: tool.description().to_string(),
                        category: tool.category(),
                        default_allowed,
                        currently_allowed,
                    }
                })
                .collect();
            let _ = reply.send(infos);
        }
        Command::SetModel(m) => state.frame.config.model = m,
        Command::SetReasoning(l) => state.frame.config.reasoning = l,
        Command::SetCompactionConfig(c) => state.frame.config.compaction = c,
        Command::SetApprovalPolicy(p) => state.frame.approval_policy = p,
        Command::Steer(msg) => t::apply_enqueue_steering(&mut state.conv, msg),
        Command::FollowUp(msg) => t::apply_enqueue_follow_up(&mut state.conv, msg),
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
            t::apply_final_summary(&mut state.conv, msg);
        }
    }
}
