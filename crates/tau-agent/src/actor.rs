//! Actor task and stepped state machine.
//!
//! The actor task owns `AgentState` exclusively and processes commands
//! from the channel. During async operations (LLM calls, tool execution),
//! it `select!`s on the command channel to handle queries and abort.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use futures::StreamExt;
use parking_lot::Mutex;
use tau_ai::{Content, Message};
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use crate::builder::TransformContextFn;
use crate::command::{Command, PromptResult};
use crate::compaction::{self, CompactionReason};
use crate::config::{AgentConfig, DequeueMode};
use crate::conversation::Conversation;
use crate::events::AgentEvent;
use crate::overflow::is_context_overflow;
use crate::stream::{StreamOutcome, StreamReducer};
use crate::tool::{
    BoxedTool, Concurrency, ExecutionContext, FileAccessTracker, ProgressSender, ToolResult,
    send_event, to_api_tool,
};
use crate::tool_executor::{has_meaningful_content, run_single_tool};
use crate::transport::{AgentEventStream, AgentRunConfig, Transport};

/// A single tool call extracted from the model's response.
pub(crate) struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

/// All mutable state the agent needs. Owned exclusively by the actor task.
pub(crate) struct AgentState {
    pub config: AgentConfig,
    pub conversation: Conversation,
    pub tools: Vec<BoxedTool>,
    pub transport: Arc<dyn Transport>,
    pub event_tx: broadcast::Sender<AgentEvent>,
    pub server_tools: Vec<tau_ai::ServerTool>,
    pub schema_cache: HashMap<String, Arc<jsonschema::Validator>>,
    pub cwd: Option<PathBuf>,
    pub file_access: Arc<Mutex<FileAccessTracker>>,
    pub interaction_tx: Option<mpsc::Sender<crate::interaction::InteractionRequest>>,
    pub transform_context: Option<Arc<TransformContextFn>>,
    pub steering_queue: Vec<Message>,
    pub follow_up_queue: Vec<Message>,
    /// Shared with AgentHandle. External code (AgentManager) increments via handle.expect_follow_up().
    pub pending_follow_ups: Arc<AtomicU32>,
    // Shared with AgentHandle
    pub is_running: Arc<AtomicBool>,
    /// Shared with all AgentHandle clones. The actor replaces the inner token at
    /// each prompt start so handle.abort() always targets the active prompt.
    pub cancel: Arc<Mutex<CancellationToken>>,
}

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

pub(crate) async fn run_actor(mut state: AgentState, mut cmd_rx: mpsc::Receiver<Command>) {
    let mut phase = StepPhase::Idle;
    let mut prompt_reply: Option<tokio::sync::oneshot::Sender<PromptResult>> = None;
    let mut turn_number: u32 = 0;
    // Per-prompt cancellation token. Cloned from state.cancel at prompt start.
    // handle.abort() cancels the token inside state.cancel, which is the same
    // object the actor cloned from.
    let mut prompt_cancel = CancellationToken::new();

    loop {
        phase = match phase {
            StepPhase::Idle => match cmd_rx.recv().await {
                Some(cmd) => {
                    handle_idle_command(
                        &mut state,
                        cmd,
                        &mut prompt_reply,
                        &mut turn_number,
                        &mut prompt_cancel,
                    )
                }
                None => break,
            },

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
                        prepare_turn(&mut state, pending, first_user_message, &mut turn_number, &prompt_cancel).await
                    }
                } else {
                    prepare_turn(&mut state, pending, first_user_message, &mut turn_number, &prompt_cancel).await
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
                        Some(cmd) = cmd_rx.recv() => {
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
            } => process_response(&mut state, *outcome, first_user_message, pending, &prompt_cancel).await,

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
                        Some(cmd) = cmd_rx.recv() => {
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
                                    // Current batch done. Check steering before next group.
                                    if !state.steering_queue.is_empty() {
                                        // Steering arrived — skip remaining groups
                                        let steering = drain_queue(
                                            &mut state.steering_queue,
                                            state.config.steering_mode,
                                        );
                                        skip_remaining_groups(
                                            &state.event_tx,
                                            &remaining_groups,
                                            &all_tool_calls,
                                            &mut results_map,
                                        );
                                        // Collect final ordered results + steering
                                        let mut tool_results = collect_ordered_results(
                                            &all_tool_calls,
                                            results_map,
                                        );
                                        tool_results.extend(steering);
                                        state.conversation.messages.extend(tool_results.iter().cloned());
                                        break StepPhase::PrepareTurn {
                                            pending: tool_results,
                                            first_user_message: None,
                                        };
                                    }

                                    if remaining_groups.is_empty() {
                                        break StepPhase::ApplyToolResults {
                                            tool_calls: all_tool_calls,
                                            results_map,
                                            first_user_message,
                                        };
                                    } else {
                                        // Spawn next group
                                        let mut remaining = remaining_groups;
                                        let next_group = remaining.remove(0);
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

            StepPhase::ApplyToolResults {
                tool_calls,
                results_map,
                first_user_message,
            } => {
                let tool_results = collect_ordered_results(&tool_calls, results_map);
                state.conversation.messages.extend(tool_results.iter().cloned());

                // Tool results are now committed to conversation.messages.
                // Next turn's pending is empty — build_context reads from conversation.
                StepPhase::PrepareTurn {
                    pending: vec![],
                    first_user_message,
                }
            }

            StepPhase::DrainFollowUps => {
                drain_follow_ups(&mut state)
            }

            StepPhase::WaitingForFollowUps => {
                // Block on cmd_rx waiting for FollowUp commands from background agents.
                // Also handle queries and cancellation.
                loop {
                    tokio::select! {
                        biased;
                        _ = prompt_cancel.cancelled() => {
                            break StepPhase::Done(Ok(()));
                        }
                        cmd = cmd_rx.recv() => {
                            match cmd {
                                Some(Command::FollowUp(msg)) => {
                                    state.follow_up_queue.push(msg);
                                    // Re-check via DrainFollowUps (it will drain and start a turn)
                                    break StepPhase::DrainFollowUps;
                                }
                                Some(other) => {
                                    handle_busy_command(&mut state, other);
                                }
                                None => break StepPhase::Done(Ok(())),
                            }
                        }
                    }
                }
            }

            StepPhase::RunCompaction {
                reason,
                reply,
                resume_after,
            } => {
                run_compaction_phase(&mut state, reason, reply, resume_after, &prompt_cancel, &mut turn_number).await
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
            state.file_access.lock().rebuild_from_messages(&msgs, &state.cwd);
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

    // Build context: conversation.messages + pending (not yet committed)
    let context = build_context(state, &pending);
    let run_config = build_run_config(state);

    match state.transport.run(context, &run_config, cancel.clone()).await {
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
                flush_pending(&mut state.conversation.messages, &pending);
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
                    AgentEvent::Error { message: error_msg.clone() },
                );
                StepPhase::Done(Err(crate::error::Error::Ai(e)))
            }
        }
    }
}

/// Build context: conversation.messages + pending, with optional transform.
fn build_context(state: &AgentState, pending: &[Message]) -> Vec<Message> {
    let mut context: Vec<Message> = state
        .conversation
        .messages
        .iter()
        .cloned()
        .chain(pending.iter().cloned())
        .collect();

    if let Some(ref transform) = state.transform_context {
        context = transform(context);
    }
    context
}

fn build_run_config(state: &AgentState) -> AgentRunConfig {
    AgentRunConfig {
        system_prompt: state.config.system_prompt.clone(),
        tools: state.tools.iter().map(|t| to_api_tool(t.as_ref())).collect(),
        server_tools: state.server_tools.clone(),
        model: state.config.model.clone(),
        reasoning: Some(state.config.reasoning),
        thinking_adaptive: state.config.thinking_adaptive,
        max_tokens: state.config.max_tokens,
        temperature: None,
        cache_scope: state.config.cache_scope.clone(),
        cache_ttl: state.config.cache_ttl.clone(),
        system_prompt_boundary: state.config.system_prompt_boundary.clone(),
    }
}

/// Commit pending messages into the conversation.
fn flush_pending(messages: &mut Vec<Message>, pending: &[Message]) {
    for m in pending {
        messages.push(m.clone());
    }
}

// ─── Response processing ────────────────────────────────────────────

async fn process_response(
    state: &mut AgentState,
    outcome: StreamOutcome,
    first_user_message: Option<Message>,
    pending: Vec<Message>,
    cancel: &CancellationToken,
) -> StepPhase {
    // Handle error
    if let Some(ref error_msg) = outcome.error {
        // Save partial message if present
        if let Some(ref partial) = outcome.partial_message {
            if has_meaningful_content(partial) {
                flush_pending(&mut state.conversation.messages, &pending);
                state.conversation.messages.push(partial.clone());
            }
        }

        // Try overflow recovery
        let overflow = is_context_overflow(error_msg);
        if overflow && state.config.compaction.enabled {
            flush_pending(&mut state.conversation.messages, &pending);
            return StepPhase::RunCompaction {
                reason: CompactionReason::Overflow,
                reply: None,
                resume_after: Some((
                    first_user_message.iter().cloned().collect(),
                    first_user_message,
                )),
            };
        }

        state.conversation.error = Some(error_msg.clone());
        return StepPhase::Done(Err(crate::error::Error::Other(error_msg.clone())));
    }

    // Update usage
    state.conversation.total_usage.input += outcome.usage.input;
    state.conversation.total_usage.output += outcome.usage.output;
    state.conversation.total_usage.cache_read += outcome.usage.cache_read;
    state.conversation.total_usage.cache_write += outcome.usage.cache_write;
    state.conversation.total_usage.thinking += outcome.usage.thinking;
    state.conversation.total_usage.cache_creation_1h += outcome.usage.cache_creation_1h;
    state.conversation.total_usage.cache_creation_5m += outcome.usage.cache_creation_5m;

    if let Some(ref assistant_msg) = outcome.assistant_message {
        // Always commit pending + assistant message (original never gates on has_meaningful_content here)
        flush_pending(&mut state.conversation.messages, &pending);
        state.conversation.messages.push(assistant_msg.clone());

        // Proactive compaction check using real token counts from the API.
        // Run inline so tool calls are not lost.
        let used = outcome.usage.input + outcome.usage.cache_read;
        let limit = (state.config.model.context_window as u64)
            .saturating_sub(state.config.compaction.reserve_tokens);
        if state.config.compaction.enabled && used > limit {
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
                    apply_compaction_result(state, cr);
                }
                Err(e) => {
                    tracing::warn!("Proactive compaction failed: {}", e);
                }
            }
        }

        // Extract tool calls
        let tool_calls = extract_tool_calls(assistant_msg);

        if tool_calls.is_empty() {
            StepPhase::DrainFollowUps
        } else {
            // Build groups and spawn first batch
            let groups = build_tool_groups(&state.tools, &tool_calls);
            if groups.is_empty() {
                StepPhase::DrainFollowUps
            } else {
                let mut remaining = groups;
                let first_group = remaining.remove(0);
                let join_set = spawn_group(state, &tool_calls, &first_group, cancel);
                StepPhase::AwaitingTools {
                    join_set,
                    remaining_groups: remaining,
                    all_tool_calls: tool_calls,
                    results_map: HashMap::new(),
                    first_user_message,
                }
            }
        }
    } else {
        // No assistant message at all
        StepPhase::Done(Ok(()))
    }
}

fn extract_tool_calls(msg: &Message) -> Vec<ToolCall> {
    let content = match msg {
        Message::Assistant { content, .. } => content,
        _ => return vec![],
    };

    content
        .iter()
        .filter_map(|c| match c {
            Content::ToolCall { id, name, arguments } => Some(ToolCall {
                id: id.clone(),
                name: name.clone(),
                args: arguments.clone(),
            }),
            _ => None,
        })
        .collect()
}

// ─── Tool execution ─────────────────────────────────────────────────

/// Build groups of tool calls: consecutive Parallel tools form a group, Sequential is singleton.
fn build_tool_groups(tools: &[BoxedTool], tool_calls: &[ToolCall]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = vec![];
    let mut current_group: Vec<usize> = vec![];
    let mut current_is_parallel = false;

    for (idx, tc) in tool_calls.iter().enumerate() {
        let is_parallel = tools
            .iter()
            .find(|t| t.name() == tc.name.as_str())
            .map(|t| t.concurrency() == Concurrency::Parallel)
            .unwrap_or(false);

        if idx == 0 {
            current_is_parallel = is_parallel;
            current_group.push(idx);
        } else if is_parallel && current_is_parallel {
            current_group.push(idx);
        } else {
            groups.push(std::mem::take(&mut current_group));
            current_group.push(idx);
            current_is_parallel = is_parallel;
        }
    }
    if !current_group.is_empty() {
        groups.push(current_group);
    }
    groups
}

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
            let result = run_single_tool(tool, id.clone(), name.clone(), args, validator, event_tx, ctx).await;
            (idx, id, name, result)
        });
    }

    join_set
}

/// Skip remaining tool groups by emitting start/end events with error results.
fn skip_remaining_groups(
    event_tx: &broadcast::Sender<AgentEvent>,
    remaining_groups: &[Vec<usize>],
    tool_calls: &[ToolCall],
    results_map: &mut HashMap<usize, (String, String, ToolResult)>,
) {
    for group in remaining_groups {
        for &idx in group {
            let tc = &tool_calls[idx];
            send_event(
                event_tx,
                AgentEvent::ToolExecutionStart {
                    tool_call_id: tc.id.clone(),
                    tool_name: tc.name.clone(),
                    arguments: serde_json::Value::Null,
                    activity: "Skipped".to_string(),
                },
            );
            let skip_result = ToolResult::error("Skipped due to steering message");
            send_event(
                event_tx,
                AgentEvent::ToolExecutionEnd {
                    tool_call_id: tc.id.clone(),
                    tool_name: tc.name.clone(),
                    result: skip_result.text_content(),
                    is_error: skip_result.is_error,
                },
            );
            results_map.insert(idx, (tc.id.clone(), tc.name.clone(), skip_result));
        }
    }
}

/// Collect tool results in original request order, producing ToolResult messages.
fn collect_ordered_results(
    tool_calls: &[ToolCall],
    mut results_map: HashMap<usize, (String, String, ToolResult)>,
) -> Vec<Message> {
    let mut messages = Vec::new();
    for (idx, tc) in tool_calls.iter().enumerate() {
        let (id, name, result) = results_map.remove(&idx).unwrap_or_else(|| {
            (
                tc.id.clone(),
                tc.name.clone(),
                ToolResult::error("Task failed (panicked or cancelled)"),
            )
        });
        messages.push(Message::ToolResult {
            tool_call_id: id,
            tool_name: name,
            content: result.content,
            is_error: result.is_error,
            timestamp: chrono::Utc::now().timestamp_millis(),
        });
    }
    messages
}

// ─── Follow-up draining ─────────────────────────────────────────────

fn drain_follow_ups(state: &mut AgentState) -> StepPhase {
    // Check steering queue first (user messages injected during execution)
    let steering = drain_queue(&mut state.steering_queue, state.config.steering_mode);
    if !steering.is_empty() {
        return StepPhase::PrepareTurn {
            pending: steering,
            first_user_message: None,
        };
    }

    // Check follow-up queue
    let follow_ups = drain_queue(&mut state.follow_up_queue, state.config.follow_up_mode);
    if !follow_ups.is_empty() {
        let count = follow_ups.len() as u32;
        let _ = state.pending_follow_ups.fetch_update(
            Ordering::Release,
            Ordering::Acquire,
            |n| Some(n.saturating_sub(count)),
        );
        return StepPhase::PrepareTurn {
            pending: follow_ups,
            first_user_message: None,
        };
    }

    // If background agents are still pending, wait for their follow-ups.
    // WaitingForFollowUps is handled in the main loop with select! on cmd_rx.
    if state.pending_follow_ups.load(Ordering::Acquire) > 0 {
        return StepPhase::WaitingForFollowUps;
    }

    StepPhase::Done(Ok(()))
}

fn drain_queue(queue: &mut Vec<Message>, mode: DequeueMode) -> Vec<Message> {
    match mode {
        DequeueMode::All => std::mem::take(queue),
        DequeueMode::OneAtATime => {
            if queue.is_empty() {
                vec![]
            } else {
                vec![queue.remove(0)]
            }
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
    send_event(
        &state.event_tx,
        AgentEvent::CompactionStart { reason },
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
            apply_compaction_result(state, cr);

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

fn apply_compaction_result(state: &mut AgentState, result: compaction::CompactionResult) {
    state.conversation.previous_summary = Some(result.summary.clone());
    let kept = state.conversation.messages.split_off(result.first_kept_index);
    state.conversation.messages = vec![Message::user(format!(
        "<context-summary>\n{}\n</context-summary>\n\nThe conversation was compacted. Continue from where we left off.",
        result.summary
    ))];
    state.conversation.messages.extend(kept);
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
    flush_pending(&mut state.conversation.messages, pending);

    let summary_prompt = vec![Message::user(format!(
        "[System: You have reached the maximum of {} turns. \
         Summarize your findings so far. Do not call any tools.]",
        max
    ))];

    let context = build_context(state, &summary_prompt);
    let mut final_config = build_run_config(state);
    final_config.tools.clear();

    if let Ok(mut stream) = state.transport.run(context, &final_config, cancel.clone()).await {
        let mut reducer = StreamReducer::default();
        while let Some(event) = stream.next().await {
            send_event(&state.event_tx, event.clone());
            reducer.observe(&event);
        }
        let outcome = reducer.finalize();
        state.conversation.total_usage.input += outcome.usage.input;
        state.conversation.total_usage.output += outcome.usage.output;
        if let Some(msg) = outcome.assistant_message {
            state.conversation.messages.push(msg);
        }
    }
}
