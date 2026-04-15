//! Actor task and stepped state machine.
//!
//! The actor task owns `AgentState` exclusively and processes commands
//! from the channel. During async operations (LLM calls, tool execution),
//! it `select!`s on the command channel to handle queries and abort.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use futures::StreamExt;
use tau_ai::Message;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::command::{Command, PromptResult};
use crate::compaction::{self, CompactionReason};
use crate::events::AgentEvent;
use crate::logic::{self, FollowUpAction, ResponseAction};
use crate::overflow::is_context_overflow;
use crate::state::{AgentState, ToolCall};
use crate::stream::{StreamOutcome, StreamReducer};
use crate::tool::{ExecutionContext, ProgressSender, ToolResult, send_event};
use crate::tool_executor::run_single_tool;
use crate::transport::AgentEventStream;

/// Phases of a single turn in the agent loop.
enum StepPhase {
    /// No prompt active. Block on command channel.
    Idle,

    /// About to call the LLM.
    /// `pending` = messages to send (not yet committed to conversation).
    PrepareTurn {
        pending: Vec<Message>,
        first_user_message: Option<Message>,
    },

    /// Waiting for LLM response stream.
    AwaitingModel {
        stream: AgentEventStream,
        first_user_message: Option<Message>,
        pending: Vec<Message>,
    },

    /// LLM responded. Process tool calls or finish.
    ProcessResponse {
        outcome: Box<StreamOutcome>,
        first_user_message: Option<Message>,
        pending: Vec<Message>,
    },

    /// Executing tool calls concurrently.
    AwaitingTools {
        join_set: tokio::task::JoinSet<(usize, String, String, ToolResult)>,
        remaining_groups: Vec<Vec<usize>>,
        all_tool_calls: Vec<ToolCall>,
        results_map: HashMap<usize, (String, String, ToolResult)>,
        first_user_message: Option<Message>,
    },

    /// Tools done. Apply results, prepare next turn.
    ApplyToolResults {
        tool_calls: Vec<ToolCall>,
        results_map: HashMap<usize, (String, String, ToolResult)>,
        first_user_message: Option<Message>,
    },

    /// Turn complete, no tool calls. Check steering/follow-up queues (synchronous).
    DrainFollowUps,

    /// Waiting for background agents to post follow-ups. Handled inline in the
    /// main loop with select! on cmd_rx (like AwaitingModel/AwaitingTools).
    WaitingForFollowUps,

    /// Run compaction (async phase that the idle or in-flight actor can enter).
    RunCompaction {
        reason: CompactionReason,
        reply: Option<tokio::sync::oneshot::Sender<PromptResult>>,
        /// If true, retry the current turn after compaction.
        resume_after: Option<(Vec<Message>, Option<Message>)>,
    },

    /// Prompt finished (success or error).
    Done(Result<(), crate::error::Error>),
}

// ─── Actor entry point ──────────────────────────────────────────────

pub(crate) async fn run_actor(
    mut state: AgentState,
    mut urgent_rx: mpsc::Receiver<Command>,
    mut normal_rx: mpsc::Receiver<Command>,
) {
    let mut phase = StepPhase::Idle;
    let mut prompt_reply: Option<tokio::sync::oneshot::Sender<PromptResult>> = None;
    let mut turn_number: u32 = 0;
    // Per-prompt cancellation token. Cloned from state.cancel at prompt start.
    // handle.abort() cancels the token inside state.cancel, which is the same
    // object the actor cloned from.
    let mut prompt_cancel = CancellationToken::new();

    loop {
        phase = match phase {
            StepPhase::Idle => {
                // In Idle, drain both channels. Urgent first via biased select.
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

            StepPhase::PrepareTurn {
                pending,
                first_user_message,
            } => {
                if prompt_cancel.is_cancelled() {
                    StepPhase::Done(Ok(()))
                } else if let Some(max) = state.config.max_turns {
                    if turn_number >= max {
                        run_final_summary(&mut state, &pending, &prompt_cancel).await;
                        StepPhase::Done(Ok(()))
                    } else {
                        prepare_turn(
                            &mut state,
                            pending,
                            first_user_message,
                            &mut turn_number,
                            &prompt_cancel,
                        )
                        .await
                    }
                } else {
                    prepare_turn(
                        &mut state,
                        pending,
                        first_user_message,
                        &mut turn_number,
                        &prompt_cancel,
                    )
                    .await
                }
            }

            StepPhase::AwaitingModel {
                mut stream,
                first_user_message,
                pending,
            } => {
                let mut reducer = StreamReducer::default();
                loop {
                    tokio::select! {
                        biased;
                        Some(cmd) = urgent_rx.recv() => {
                            handle_busy_command(&mut state, cmd);
                        }
                        Some(cmd) = normal_rx.recv() => {
                            handle_busy_command(&mut state, cmd);
                        }
                        event = stream.next() => {
                            match event {
                                Some(e) => {
                                    send_event(&state.event_tx, e.clone());
                                    reducer.observe(&e);
                                }
                                None => {
                                    break StepPhase::ProcessResponse {
                                        outcome: Box::new(reducer.finalize()),
                                        first_user_message,
                                        pending,
                                    };
                                }
                            }
                        }
                    }
                }
            }

            StepPhase::ProcessResponse {
                outcome,
                first_user_message,
                pending,
            } => {
                let decision = state.process_response(*outcome, pending, first_user_message);
                if decision.needs_proactive_compaction {
                    run_proactive_compaction(&mut state, &prompt_cancel).await;
                }
                match decision.action {
                    ResponseAction::RunTools {
                        tool_calls,
                        groups,
                        first_user_message,
                    } => {
                        let mut remaining = groups;
                        let first = remaining.remove(0);
                        let join_set = spawn_group(&state, &tool_calls, &first, &prompt_cancel);
                        StepPhase::AwaitingTools {
                            join_set,
                            remaining_groups: remaining,
                            all_tool_calls: tool_calls,
                            results_map: HashMap::new(),
                            first_user_message,
                        }
                    }
                    ResponseAction::Compact {
                        reason,
                        resume_pending,
                    } => StepPhase::RunCompaction {
                        reason,
                        reply: None,
                        resume_after: resume_pending,
                    },
                    ResponseAction::Done => StepPhase::DrainFollowUps,
                    ResponseAction::Error(e) => {
                        send_event(
                            &state.event_tx,
                            AgentEvent::Error {
                                message: e.to_string(),
                            },
                        );
                        StepPhase::Done(Err(e))
                    }
                }
            }

            StepPhase::AwaitingTools {
                mut join_set,
                remaining_groups,
                all_tool_calls,
                mut results_map,
                first_user_message,
            } => {
                loop {
                    tokio::select! {
                        biased;
                        _ = prompt_cancel.cancelled() => {
                            // JoinSet aborts all spawned tasks on drop.
                            break StepPhase::Done(Ok(()));
                        }
                        Some(cmd) = urgent_rx.recv() => {
                            handle_busy_command(&mut state, cmd);
                        }
                        Some(cmd) = normal_rx.recv() => {
                            handle_busy_command(&mut state, cmd);
                        }
                        result = join_set.join_next() => {
                            match result {
                                Some(Ok((idx, id, name, tool_result))) => {
                                    results_map.insert(idx, (id, name, tool_result));
                                }
                                Some(Err(join_err)) => {
                                    tracing::error!("Tool task panicked: {}", join_err);
                                }
                                None => {
                                    // Current batch done — delegate decision to logic.
                                    match state.handle_batch_complete(
                                        &remaining_groups,
                                        &all_tool_calls,
                                        &mut results_map,
                                    ) {
                                        logic::BatchCompleteAction::Redirect {
                                            steering,
                                            skipped_indices,
                                            ..
                                        } => {
                                            // Emit skip events for the skipped tools (I/O)
                                            for (_, id, name) in &skipped_indices {
                                                send_event(
                                                    &state.event_tx,
                                                    AgentEvent::ToolExecutionStart {
                                                        tool_call_id: id.clone(),
                                                        tool_name: name.clone(),
                                                        arguments: serde_json::Value::Null,
                                                        activity: "Skipped".to_string(),
                                                    },
                                                );
                                                send_event(
                                                    &state.event_tx,
                                                    AgentEvent::ToolExecutionEnd {
                                                        tool_call_id: id.clone(),
                                                        tool_name: name.clone(),
                                                        result: "Skipped due to steering message".to_string(),
                                                        is_error: true,
                                                    },
                                                );
                                            }
                                            break StepPhase::PrepareTurn {
                                                pending: steering,
                                                first_user_message: None,
                                            };
                                        }
                                        logic::BatchCompleteAction::AllGroupsDone => {
                                            break StepPhase::ApplyToolResults {
                                                tool_calls: all_tool_calls,
                                                results_map,
                                                first_user_message,
                                            };
                                        }
                                        logic::BatchCompleteAction::NextGroup(next_group) => {
                                            let mut remaining = remaining_groups;
                                            remaining.remove(0);
                                            let new_join_set = spawn_group(
                                                &state,
                                                &all_tool_calls,
                                                &next_group,
                                                &prompt_cancel,
                                            );
                                            break StepPhase::AwaitingTools {
                                                join_set: new_join_set,
                                                remaining_groups: remaining,
                                                all_tool_calls,
                                                results_map,
                                                first_user_message,
                                            };
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            StepPhase::ApplyToolResults {
                tool_calls,
                mut results_map,
                first_user_message,
            } => {
                let tool_results = logic::collect_ordered_results(&tool_calls, &mut results_map);
                state
                    .conversation
                    .messages
                    .extend(tool_results.iter().cloned());

                // Tool results are now committed to conversation.messages.
                // Next turn's pending is empty — build_context reads from conversation.
                StepPhase::PrepareTurn {
                    pending: vec![],
                    first_user_message,
                }
            }

            StepPhase::DrainFollowUps => match state.drain_follow_ups() {
                FollowUpAction::Continue(msgs) => StepPhase::PrepareTurn {
                    pending: msgs,
                    first_user_message: None,
                },
                FollowUpAction::WaitForFollowUps => StepPhase::WaitingForFollowUps,
                FollowUpAction::Done => StepPhase::Done(Ok(())),
            },

            StepPhase::WaitingForFollowUps => {
                // Block waiting for FollowUp commands from background agents.
                // Urgent channel gets priority (FollowUp is urgent).
                loop {
                    tokio::select! {
                        biased;
                        _ = prompt_cancel.cancelled() => {
                            break StepPhase::Done(Ok(()));
                        }
                        Some(cmd) = urgent_rx.recv() => {
                            match cmd {
                                Command::FollowUp(msg) => {
                                    state.follow_up_queue.push(msg);
                                    break StepPhase::DrainFollowUps;
                                }
                                other => handle_busy_command(&mut state, other),
                            }
                        }
                        Some(cmd) = normal_rx.recv() => {
                            handle_busy_command(&mut state, cmd);
                        }
                    }
                }
            }

            StepPhase::RunCompaction {
                reason,
                reply,
                resume_after,
            } => {
                run_compaction_phase(
                    &mut state,
                    reason,
                    reply,
                    resume_after,
                    &prompt_cancel,
                    &mut turn_number,
                )
                .await
            }

            StepPhase::Done(result) => {
                state.is_running.store(false, Ordering::Release);
                state.conversation.is_streaming = false;
                send_event(
                    &state.event_tx,
                    AgentEvent::AgentEnd {
                        total_turns: turn_number,
                        total_usage: state.conversation.total_usage.clone(),
                    },
                );
                if let Some(reply) = prompt_reply.take() {
                    let _ = reply.send(PromptResult { result });
                }
                turn_number = 0;
                StepPhase::Idle
            }
        };
    }
}

// ─── Command handling ───────────────────────────────────────────────

fn handle_idle_command(
    state: &mut AgentState,
    cmd: Command,
    prompt_reply: &mut Option<tokio::sync::oneshot::Sender<PromptResult>>,
    turn_number: &mut u32,
    prompt_cancel: &mut CancellationToken,
) -> StepPhase {
    match cmd {
        Command::Prompt { content, reply } => {
            *prompt_reply = Some(reply);
            state.is_running.store(true, Ordering::Release);
            // Replace the token inside the shared mutex with a fresh one.
            // handle.abort() cancels this same token via Arc<Mutex<>>.
            let fresh = CancellationToken::new();
            *prompt_cancel = fresh.clone();
            *state.cancel.lock() = fresh;
            *turn_number = 0;
            state.conversation.is_streaming = true;
            state.conversation.error = None;
            send_event(&state.event_tx, AgentEvent::AgentStart);

            let user_message = Message::User {
                content: content.clone(),
                timestamp: chrono::Utc::now().timestamp_millis(),
            };

            StepPhase::PrepareTurn {
                pending: vec![user_message.clone()],
                first_user_message: Some(user_message),
            }
        }

        Command::Compact { reason, reply } => {
            // Idle — run compaction inline in the actor loop
            StepPhase::RunCompaction {
                reason,
                reply: Some(reply),
                resume_after: None,
            }
        }

        other => {
            handle_busy_command(state, other);
            StepPhase::Idle
        }
    }
}

fn handle_busy_command(state: &mut AgentState, cmd: Command) {
    match cmd {
        // Queries — reply immediately
        Command::GetConfig(reply) => {
            let _ = reply.send(state.config.clone());
        }
        Command::GetMessages(reply) => {
            let _ = reply.send(state.conversation.messages.clone());
        }
        Command::GetState(reply) => {
            let _ = reply.send(state.conversation.clone());
        }

        // Config mutations
        Command::SetModel(m) => state.config.model = m,
        Command::SetReasoning(l) => state.config.reasoning = l,
        Command::SetSystemPrompt(s) => state.config.system_prompt = Some(s),
        Command::SetCompactionConfig(c) => state.config.compaction = c,

        // Steer / follow-up
        Command::Steer(msg) => state.steering_queue.push(msg),
        Command::FollowUp(msg) => {
            state.follow_up_queue.push(msg);
        }

        // Reject concurrent prompts
        Command::Prompt { reply, .. } => {
            let _ = reply.send(PromptResult {
                result: Err(crate::error::Error::Busy),
            });
        }

        // Conversation mutations
        Command::ClearMessages => {
            state.conversation.messages.clear();
            state.conversation.total_usage = Default::default();
            state.conversation.previous_summary = None;
            state.file_access.lock().clear();
        }
        Command::SetMessages(msgs) => {
            state
                .file_access
                .lock()
                .rebuild_from_messages(&msgs, &state.cwd);
            state.conversation.messages = msgs;
        }
        Command::SetPreviousSummary(s) => state.conversation.previous_summary = s,

        Command::Compact { reply, .. } => {
            let _ = reply.send(PromptResult {
                result: Err(crate::error::Error::Busy),
            });
        }
    }
}

// ─── Turn preparation ───────────────────────────────────────────────

async fn prepare_turn(
    state: &mut AgentState,
    pending: Vec<Message>,
    first_user_message: Option<Message>,
    turn_number: &mut u32,
    cancel: &CancellationToken,
) -> StepPhase {
    *turn_number += 1;

    let context = state.build_context(&pending);
    let run_config = state.build_run_config();

    match state
        .transport
        .run(context, &run_config, cancel.clone())
        .await
    {
        Ok(stream) => StepPhase::AwaitingModel {
            stream,
            first_user_message,
            pending,
        },
        Err(e) => {
            let error_msg = e.to_string();
            let overflow = e.is_context_overflow() || is_context_overflow(&error_msg);
            if overflow && state.config.compaction.enabled {
                // Commit pending so compaction sees them
                logic::flush_pending(&mut state.conversation.messages, &pending);
                StepPhase::RunCompaction {
                    reason: CompactionReason::Overflow,
                    reply: None,
                    resume_after: Some((
                        first_user_message.iter().cloned().collect(),
                        first_user_message,
                    )),
                }
            } else {
                state.conversation.error = Some(error_msg.clone());
                send_event(
                    &state.event_tx,
                    AgentEvent::Error {
                        message: error_msg.clone(),
                    },
                );
                StepPhase::Done(Err(crate::error::Error::Ai(e)))
            }
        }
    }
}

// ─── Tool execution ─────────────────────────────────────────────────

/// Spawn tool tasks for a single group. Returns (idx, id, name, result) tuples
/// to preserve original ordering.
fn spawn_group(
    state: &AgentState,
    tool_calls: &[ToolCall],
    group: &[usize],
    cancel: &CancellationToken,
) -> tokio::task::JoinSet<(usize, String, String, ToolResult)> {
    let mut join_set = tokio::task::JoinSet::new();

    let cwd = state
        .cwd
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

    for &idx in group {
        let tc = &tool_calls[idx];
        let tool = state.tools.iter().find(|t| t.name() == tc.name).cloned();
        let validator = state.schema_cache.get(&tc.name).cloned();
        let event_tx = state.event_tx.clone();
        let cancel = cancel.clone();
        let file_access = state.file_access.clone();
        let interaction_tx = state.interaction_tx.clone();
        let cwd = cwd.clone();

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
        };

        join_set.spawn(async move {
            let result = run_single_tool(
                tool,
                id.clone(),
                name.clone(),
                args,
                validator,
                event_tx,
                ctx,
            )
            .await;
            (idx, id, name, result)
        });
    }

    join_set
}

// (skip_remaining_groups logic moved to AgentState::handle_batch_complete in logic.rs)

// ─── Proactive compaction (I/O) ────────────────────────────────────

/// Run proactive compaction inline. Called when `ResponseDecision::needs_proactive_compaction`
/// is true, before executing tool calls or finishing the turn.
async fn run_proactive_compaction(state: &mut AgentState, cancel: &CancellationToken) {
    send_event(
        &state.event_tx,
        AgentEvent::CompactionStart {
            reason: CompactionReason::Threshold,
        },
    );
    let result = compaction::compact(
        &state.conversation.messages,
        &state.config.compaction,
        &state.config,
        &state.transport,
        state.conversation.previous_summary.as_deref(),
        cancel,
    )
    .await;
    match result {
        Ok(cr) => {
            let tokens_after = compaction::estimate_total_tokens(
                &state.conversation.messages[cr.first_kept_index..],
            );
            send_event(
                &state.event_tx,
                AgentEvent::CompactionEnd {
                    tokens_before: cr.tokens_before,
                    tokens_after,
                },
            );
            state.apply_compaction_result(cr);
        }
        Err(e) => {
            tracing::warn!("Proactive compaction failed: {}", e);
        }
    }
}

// ─── Compaction ─────────────────────────────────────────────────────

async fn run_compaction_phase(
    state: &mut AgentState,
    reason: CompactionReason,
    reply: Option<tokio::sync::oneshot::Sender<PromptResult>>,
    resume_after: Option<(Vec<Message>, Option<Message>)>,
    cancel: &CancellationToken,
    turn_number: &mut u32,
) -> StepPhase {
    send_event(&state.event_tx, AgentEvent::CompactionStart { reason });

    let result = compaction::compact(
        &state.conversation.messages,
        &state.config.compaction,
        &state.config,
        &state.transport,
        state.conversation.previous_summary.as_deref(),
        cancel,
    )
    .await;

    match result {
        Ok(cr) => {
            let tokens_after = compaction::estimate_total_tokens(
                &state.conversation.messages[cr.first_kept_index..],
            );
            send_event(
                &state.event_tx,
                AgentEvent::CompactionEnd {
                    tokens_before: cr.tokens_before,
                    tokens_after,
                },
            );
            state.apply_compaction_result(cr);

            if let Some(r) = reply {
                let _ = r.send(PromptResult { result: Ok(()) });
                // Idle compaction — return to idle
                StepPhase::Idle
            } else if let Some((pending, first_user_message)) = resume_after {
                // Overflow recovery — reset turns and retry
                *turn_number = 0;
                StepPhase::PrepareTurn {
                    pending,
                    first_user_message,
                }
            } else {
                // Proactive compaction mid-prompt — continue to DrainFollowUps
                // (tool calls will be picked up on next turn if present)
                StepPhase::DrainFollowUps
            }
        }
        Err(e) => {
            send_event(
                &state.event_tx,
                AgentEvent::Error {
                    message: format!("Compaction failed: {}", e),
                },
            );
            if let Some(r) = reply {
                let _ = r.send(PromptResult {
                    result: Err(crate::error::Error::Compaction(e)),
                });
                StepPhase::Idle
            } else {
                StepPhase::Done(Err(crate::error::Error::Compaction(e)))
            }
        }
    }
}

/// Run a final summary turn with tools disabled when max_turns is reached.
async fn run_final_summary(
    state: &mut AgentState,
    pending: &[Message],
    cancel: &CancellationToken,
) {
    let last_has_tool_calls = state
        .conversation
        .messages
        .last()
        .is_some_and(|m| !m.tool_calls().is_empty());

    if !last_has_tool_calls || pending.is_empty() {
        return;
    }

    let max = state.config.max_turns.unwrap_or(0);
    tracing::info!("Agent reached max turns ({}), running final summary", max);

    // Flush pending tool results
    logic::flush_pending(&mut state.conversation.messages, pending);

    let summary_prompt = vec![Message::user(format!(
        "[System: You have reached the maximum of {} turns. \
         Summarize your findings so far. Do not call any tools.]",
        max
    ))];

    let context = state.build_context(&summary_prompt);
    let mut final_config = state.build_run_config();
    final_config.tools.clear();

    if let Ok(mut stream) = state
        .transport
        .run(context, &final_config, cancel.clone())
        .await
    {
        let mut reducer = StreamReducer::default();
        while let Some(event) = stream.next().await {
            send_event(&state.event_tx, event.clone());
            reducer.observe(&event);
        }
        let outcome = reducer.finalize();
        state.accumulate_usage(&outcome.usage);
        if let Some(msg) = outcome.assistant_message {
            state.conversation.messages.push(msg);
        }
    }
}
