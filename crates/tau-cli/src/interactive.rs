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
        loop {
            let event = match receiver.recv().await {
                Ok(event) => event,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(dropped = n, "interactive event stream lagged");
                    continue;
                }
            };
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
                AgentEvent::ToolExecutionUpdate { lines, .. } => {
                    for line in lines {
                        print!(" {}", line.content);
                    }
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
    handle: AgentHandle,
    mut session: Option<session::SessionManager>,
    interaction_rx: tokio::sync::mpsc::Receiver<tau_agent::InteractionRequest>,
    manager: Arc<AgentManager>,
    spec_resolver: tau_tools::SpecResolver,
) -> anyhow::Result<()> {
    let mut handle = handle;
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
            let effective = active_agent.as_ref().map(|a| &a.0).unwrap_or(&handle);
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
                        // /clear: spawn a fresh agent with the same spec
                        // and no inherited history; switch the active
                        // handle. The old handle is dropped.
                        let old_id = handle.agent_id().map(str::to_string);
                        let spec = match old_id.as_deref() {
                            Some(id) => manager.spec_for(id).await,
                            None => None,
                        };
                        match spec {
                            Some(spec) => {
                                handle.abort();
                                let opts = tau_agent::SpawnOpts {
                                    description: "main".into(),
                                    ..Default::default()
                                };
                                match manager.spawn_interactive(spec, opts).await {
                                    Ok((new_handle, _new_id)) => {
                                        if let Some(id) = old_id {
                                            manager.remove_interactive(&id).await;
                                        }
                                        handle = new_handle;
                                        prev_usage = tau_ai::Usage::default();
                                        println!("Cleared conversation.");
                                    }
                                    Err(e) => println!("/clear failed: {e}"),
                                }
                            }
                            None => println!(
                                "/clear unavailable: parent agent has no recorded spec"
                            ),
                        }
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
                        let _ = handle.set_model(new_model).await;
                    }
                    commands::CommandResult::ChangeReasoning(level) => {
                        println!("Reasoning level set to: {:?}", level);
                        let _ = handle.set_reasoning(level).await;
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

                        let plan_spec = match spec_resolver("plan", 0) {
                            Some(s) => s,
                            None => {
                                println!("Plan mode unavailable: 'plan' spec not registered.");
                                continue;
                            }
                        };
                        let opts = tau_agent::SpawnOpts {
                            description: format!("Planning: {}", description),
                            ..Default::default()
                        };

                        match manager.spawn_interactive(plan_spec, opts).await {
                            Ok((plan_handle, agent_id)) => {
                                println!(
                                    "Plan mode active. Use /plan approve or /plan exit.\n"
                                );
                                active_agent = Some((plan_handle, agent_id));
                                let effective =
                                    active_agent.as_ref().map(|a| &a.0).unwrap_or(&handle);
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
                                let (_, plan_agent_id) = active_agent.take().unwrap();
                                // Don't `remove_interactive` — the executor
                                // spawn below reads the plan agent's history
                                // by id from running_handles.
                                let executor_spec = match spec_resolver(
                                    "general-purpose:executor",
                                    0,
                                ) {
                                    Some(s) => s,
                                    None => {
                                        println!(
                                            "Executor unavailable: 'general-purpose:executor' spec not registered."
                                        );
                                        manager.remove_interactive(&plan_agent_id).await;
                                        continue;
                                    }
                                };
                                println!("Plan approved. Executing...");
                                let cancel = tokio_util::sync::CancellationToken::new();
                                let opts = tau_agent::SpawnOpts {
                                    description: "Executor: approved plan".into(),
                                    inherit_history_from: Some(plan_agent_id.clone()),
                                    spec_name: Some("general-purpose:executor".into()),
                                    ..Default::default()
                                };
                                match manager
                                    .spawn(
                                        executor_spec,
                                        "Execute the approved plan.".to_string(),
                                        opts,
                                        cancel,
                                    )
                                    .await
                                {
                                    Ok(result) => println!("\n{}\n", result.text),
                                    Err(e) => println!("Executor failed: {e}"),
                                }
                                manager.remove_interactive(&plan_agent_id).await;
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
                                // Truncate the in-process conversation by
                                // respawning the parent agent with a
                                // seed_messages slice. The old handle is
                                // dropped.
                                let old_id = handle.agent_id().map(str::to_string);
                                let spec = match old_id.as_deref() {
                                    Some(id) => manager.spec_for(id).await,
                                    None => None,
                                };
                                let truncated: Vec<_> = match branch_index {
                                    Some(idx) => messages.iter().take(idx + 1).cloned().collect(),
                                    None => Vec::new(),
                                };
                                match spec {
                                    Some(spec) => {
                                        handle.abort();
                                        let opts = tau_agent::SpawnOpts {
                                            description: "main".into(),
                                            seed_messages: Some(truncated),
                                            ..Default::default()
                                        };
                                        match manager.spawn_interactive(spec, opts).await {
                                            Ok((new_handle, _)) => {
                                                if let Some(id) = old_id {
                                                    manager.remove_interactive(&id).await;
                                                }
                                                handle = new_handle;
                                                prev_usage = tau_ai::Usage::default();
                                                session = Some(new_session);
                                                println!(
                                                    "Continuing from branch point ({} message(s)).",
                                                    msg_count
                                                );
                                            }
                                            Err(e) => println!("/branch failed: {e}"),
                                        }
                                    }
                                    None => {
                                        println!(
                                            "/branch unavailable: parent agent has no recorded spec; restart with --resume {} instead.",
                                            new_session.id()
                                        );
                                        session = Some(new_session);
                                    }
                                }
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
        let effective = active_agent.as_ref().map(|a| &a.0).unwrap_or(&handle);
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
