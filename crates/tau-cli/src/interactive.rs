//! Simple stdin/stdout interactive REPL mode

use std::io::{self, Write};
use std::sync::Arc;

use tau_agent::manager::AgentManager;
use tau_agent::{AgentEvent, AgentHandle};

use crate::{cli::get_available_models, commands, run_command::handle_interaction_stdin, session};

/// Session + cumulative-usage tracker for delta computation. The pair travels
/// together because the session log records per-turn usage deltas.
struct SaveCtx<'a> {
    session: &'a mut session::SessionManager,
    prev_usage: &'a mut tau_ai::Usage,
}

/// Send a prompt to `handle` and stream the resulting events to stdout.
/// When `save` is `Some`, new messages and the turn's usage delta are appended
/// to the session. Plan-mode prompts pass `None` so plan-agent traffic does not
/// pollute the main session.
async fn run_prompt_and_stream(
    handle: &AgentHandle,
    input: &str,
    save: Option<SaveCtx<'_>>,
) -> anyhow::Result<()> {
    let mut receiver = handle.subscribe();
    let model_for_cost = handle
        .config()
        .await
        .ok_or_else(|| anyhow::anyhow!("Agent shut down"))?
        .model
        .clone();

    let is_tty = std::io::IsTerminal::is_terminal(&io::stdout());
    let event_handle = tokio::spawn(async move {
        let mut last_text_len = 0;
        while let Ok(event) = receiver.recv().await {
            match event {
                AgentEvent::MessageUpdate { message } => {
                    let text = message.text();
                    let text_chars: Vec<char> = text.chars().collect();
                    if text_chars.len() > last_text_len {
                        let new_text: String = text_chars[last_text_len..].iter().collect();
                        print!("{}", new_text);
                        io::stdout().flush().ok();
                        last_text_len = text_chars.len();
                    }
                }
                AgentEvent::MessageEnd { .. } => {
                    println!();
                    last_text_len = 0;
                }
                AgentEvent::ToolExecutionStart { tool_name, .. } => {
                    print!("\n[{}...", tool_name);
                    io::stdout().flush().ok();
                }
                AgentEvent::ToolExecutionUpdate { content, .. } => {
                    print!(" {}", content);
                    io::stdout().flush().ok();
                }
                AgentEvent::ToolExecutionEnd {
                    tool_name: _,
                    result,
                    is_error,
                    ..
                } => {
                    if is_error {
                        println!(" error]");
                        let preview = crate::utils::truncate_chars(&result, 80);
                        println!("  {}", preview.replace('\n', " "));
                    } else {
                        let char_count = result.chars().count();
                        let first_line = result.lines().next().unwrap_or("");

                        if char_count <= 60 && !result.contains('\n') {
                            println!(" {}]", result);
                        } else if first_line.chars().count() <= 50 {
                            println!(" {}...]", first_line);
                        } else {
                            let preview: String = first_line.chars().take(40).collect();
                            println!(" {}...]", preview);
                        }
                    }
                }
                AgentEvent::AgentEnd { total_usage, .. } => {
                    let cost = total_usage.calculate_cost(&model_for_cost);
                    if is_tty {
                        println!(
                            "[{} in, {} out | ${:.4}]",
                            total_usage.input, total_usage.output, cost.total
                        );
                    }
                    break;
                }
                AgentEvent::CompactionStart { reason } => {
                    println!(
                        "[Compacting context ({})]",
                        crate::utils::compaction_reason_str(reason)
                    );
                }
                AgentEvent::CompactionEnd {
                    tokens_before,
                    tokens_after,
                } => {
                    println!("[Compacted: ~{} -> ~{} tokens]", tokens_before, tokens_after);
                }
                AgentEvent::Error { message } => {
                    eprintln!("\nError: {}", message);
                }
                _ => {}
            }
        }
    });

    let msgs_before = handle.messages().await.map(|m| m.len()).unwrap_or(0);

    if let Err(e) = handle.prompt_and_wait(input).await {
        eprintln!("Error: {}", e);
    }

    if let Some(save) = save {
        let all_msgs = handle.messages().await.unwrap_or_default();
        for msg in all_msgs.iter().skip(msgs_before) {
            let _ = save.session.append_message(msg);
        }
        let state = handle
            .state()
            .await
            .ok_or_else(|| anyhow::anyhow!("Agent shut down"))?;
        let turn_usage = tau_ai::Usage {
            input: state.total_usage.input.saturating_sub(save.prev_usage.input),
            output: state
                .total_usage
                .output
                .saturating_sub(save.prev_usage.output),
            cache_read: state
                .total_usage
                .cache_read
                .saturating_sub(save.prev_usage.cache_read),
            cache_write: state
                .total_usage
                .cache_write
                .saturating_sub(save.prev_usage.cache_write),
            thinking: state
                .total_usage
                .thinking
                .saturating_sub(save.prev_usage.thinking),
            ..Default::default()
        };
        let _ = save.session.append_usage(&turn_usage);
        *save.prev_usage = state.total_usage.clone();
    }

    if tokio::time::timeout(std::time::Duration::from_secs(2), event_handle)
        .await
        .is_err()
    {
        tracing::debug!("Event handler did not finish in time");
    }

    println!();
    Ok(())
}

pub(crate) async fn run_interactive(
    handle: &AgentHandle,
    mut session: Option<session::SessionManager>,
    interaction_rx: tokio::sync::mpsc::Receiver<tau_agent::InteractionRequest>,
    manager: Arc<AgentManager>,
) -> anyhow::Result<()> {
    let available_models = get_available_models();
    let _interaction_handle = tokio::spawn(handle_interaction_stdin(interaction_rx));

    if std::io::IsTerminal::is_terminal(&std::io::stderr()) {
        let config = handle
            .config()
            .await
            .ok_or_else(|| anyhow::anyhow!("Agent shut down"))?;
        let model_id = &config.model.id;
        let model_short = model_id.split('/').next_back().unwrap_or(model_id);
        if let Some(ref s) = session {
            eprintln!("tau ({}) session: {}", model_short, &s.id()[..8]);
        } else {
            eprintln!("tau ({})", model_short);
        }
        eprintln!();
    }

    let mut prev_usage = tau_ai::Usage::default();
    let mut active_agent: Option<(AgentHandle, String)> = None; // (handle, description)

    loop {
        if active_agent.is_some() {
            print!("plan> ");
        } else {
            print!("> ");
        }
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        if input.starts_with('/') {
            let effective = active_agent.as_ref().map(|a| &a.0).unwrap_or(handle);
            let config = effective
                .config()
                .await
                .ok_or_else(|| anyhow::anyhow!("Agent shut down"))?;
            let messages = effective
                .messages()
                .await
                .ok_or_else(|| anyhow::anyhow!("Agent shut down"))?;
            let state = effective
                .state()
                .await
                .ok_or_else(|| anyhow::anyhow!("Agent shut down"))?;
            let ctx = commands::CommandContext {
                args: "",
                config: &config,
                messages: &messages,
                usage: &state.total_usage,
                available_models: &available_models,
                has_active_agent: active_agent.is_some(),
            };
            if let Some(result) = commands::execute_command(input, &ctx) {
                match result {
                    commands::CommandResult::Clear => {
                        handle.clear_messages();
                        println!("Cleared conversation.");
                    }
                    commands::CommandResult::Exit => {
                        break;
                    }
                    commands::CommandResult::Message(msg) => {
                        println!("{}", msg);
                    }
                    commands::CommandResult::ChangeModel(new_model) => {
                        println!(
                            "Switched to: {} ({})",
                            new_model.id,
                            new_model.provider.name()
                        );
                        handle.set_model(new_model);
                    }
                    commands::CommandResult::ChangeReasoning(level) => {
                        println!("Reasoning level set to: {:?}", level);
                        handle.set_reasoning(level);
                    }
                    commands::CommandResult::Unknown(cmd) => {
                        println!("Unknown command: /{}", cmd);
                        println!("Type /help for available commands.");
                    }
                    commands::CommandResult::OpenModelSelector => {
                        println!(
                            "{}",
                            commands::ModelCommand::list_models_text(
                                &config.model,
                                &available_models
                            )
                        );
                    }
                    commands::CommandResult::OpenBranchSelector => {
                        if messages.is_empty() {
                            println!("No messages to branch from.");
                        } else {
                            println!("Messages in conversation:");
                            for (i, msg) in messages.iter().enumerate() {
                                let role = match msg {
                                    tau_ai::Message::User { .. } => "user",
                                    tau_ai::Message::Assistant { .. } => "assistant",
                                    tau_ai::Message::ToolResult { .. } => "tool",
                                    tau_ai::Message::SystemInjection { .. } => "system",
                                };
                                let text = msg.text();
                                let preview: String = text.chars().take(60).collect();
                                let preview = preview.replace('\n', " ");
                                println!("  {}: [{}] {}", i, role, preview);
                            }
                            println!("\nUse /branch <index> to create a branch from that message.");
                        }
                    }
                    commands::CommandResult::Compact => {
                        println!("Compacting context...");
                        match handle.compact(tau_agent::CompactionReason::Manual).await {
                            Ok(rx) => match rx.await {
                                Ok(r) if r.result.is_ok() => {
                                    let msg_count =
                                        handle.messages().await.map(|m| m.len()).unwrap_or(0);
                                    println!(
                                        "Context compacted. {} messages remaining.",
                                        msg_count
                                    );
                                }
                                _ => println!("Compaction failed."),
                            },
                            Err(e) => println!("Compaction failed: {}", e),
                        }
                    }
                    commands::CommandResult::PlanStart(description) => {
                        println!("Entering plan mode...");
                        let main_messages = handle.messages().await.unwrap_or_default();
                        let main_state = handle.state().await;
                        let prev_summary = main_state
                            .as_ref()
                            .and_then(|s| s.previous_summary.as_deref());
                        let context =
                            tau_tools::plan::build_context_summary(&main_messages, prev_summary);
                        let prompt = tau_tools::plan::build_plan_prompt(&context, &description);

                        let request = tau_agent::manager::SpawnRequest {
                            agent_type: tau_agent::manager::AgentType::Plan,
                            prompt: String::new(),
                            description: format!("Planning: {}", description),
                            model: None,
                            cwd: None,
                            isolation: None,
                            depth: 0,
                            inherit_history_from: None,
                        };

                        match manager.spawn_interactive(request).await {
                            Ok((plan_handle, agent_id)) => {
                                println!(
                                    "Plan mode active. Use /plan approve or /plan exit.\n"
                                );
                                active_agent = Some((plan_handle, agent_id));
                                let effective =
                                    active_agent.as_ref().map(|a| &a.0).unwrap_or(handle);
                                if let Err(e) =
                                    run_prompt_and_stream(effective, &prompt, None).await
                                {
                                    eprintln!("Plan agent error: {}", e);
                                }
                            }
                            Err(e) => {
                                println!("Failed to start plan mode: {}", e);
                            }
                        }
                    }
                    commands::CommandResult::PlanApprove => {
                        if let Some((agent_handle, _)) = active_agent.as_ref() {
                            let plan_text = agent_handle
                                .messages()
                                .await
                                .map(|m| tau_tools::plan::extract_final_text(&m))
                                .unwrap_or_default();
                            if plan_text.trim().is_empty() {
                                println!(
                                    "Plan agent has no plan to approve yet. Wait for it to respond, or use /plan exit to discard."
                                );
                            } else {
                                let (_, agent_id) = active_agent.take().unwrap();
                                manager.remove_interactive(&agent_id).await;
                                handle.steer(tau_ai::Message::user(format!(
                                    "Approved plan:\n\n{}\n\nProceed with implementation.",
                                    plan_text
                                )));
                                println!("Plan approved. Returned to main agent.");
                            }
                        } else {
                            println!("Not in plan mode.");
                        }
                    }
                    commands::CommandResult::PlanExit => {
                        if let Some((_, agent_id)) = active_agent.take() {
                            manager.remove_interactive(&agent_id).await;
                            println!("Exited plan mode.");
                        } else {
                            println!("Not in plan mode.");
                        }
                    }
                    commands::CommandResult::BranchFrom(branch_index) => {
                        match session::SessionManager::branch_from(
                            &messages,
                            branch_index,
                            &config.model.id,
                        ) {
                            Ok(new_session) => {
                                let msg_count = branch_index.map(|i| i + 1).unwrap_or(0);
                                println!(
                                    "Created branch: {} ({} messages)",
                                    new_session.id(),
                                    msg_count
                                );
                                if let Some(idx) = branch_index {
                                    let truncated: Vec<_> =
                                        messages.iter().take(idx + 1).cloned().collect();
                                    handle.set_messages(truncated);
                                } else {
                                    handle.clear_messages();
                                }
                                session = Some(new_session);
                                println!("Continue from this point with a fresh context.");
                            }
                            Err(e) => {
                                println!("Failed to create branch: {}", e);
                            }
                        }
                    }
                }
                println!();
                continue;
            }
        }

        println!();

        // Plan-agent traffic stays out of the main session log.
        let in_plan_mode = active_agent.is_some();
        let effective = active_agent.as_ref().map(|a| &a.0).unwrap_or(handle);
        let save = if in_plan_mode {
            None
        } else {
            session.as_mut().map(|s| SaveCtx {
                session: s,
                prev_usage: &mut prev_usage,
            })
        };
        run_prompt_and_stream(effective, input, save).await?;
    }

    Ok(())
}
