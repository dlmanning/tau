//! Executing sub-machine: tool tasks running on a `JoinSet`, plus the
//! single-tool execution and argument-validation helpers.

use std::collections::HashMap;

use tau_ai::Message;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::core::command::Command;
use crate::core::state::{State, ToolCall};
use crate::core::tool::{ExecutionContext, ProgressSender, ToolResult, send_event};
use crate::core::transitions as t;
use crate::types::events::AgentEvent;

use super::{Phase, ToolPhase, Turn, TurnSub, finish_cancelled_batch, handle_busy_command};

#[allow(clippy::too_many_arguments)]
pub(super) async fn step_executing(
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
                return finish_cancelled_batch(state, &all_tool_calls, &mut results_map);
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

// ─── Tool execution ─────────────────────────────────────────────────

pub(super) fn spawn_group(
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
            interaction_timeout: state.frame.interaction_timeout,
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
