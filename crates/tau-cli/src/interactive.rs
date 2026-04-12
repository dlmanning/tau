//! Simple stdin/stdout interactive REPL mode

use std::io::{self, Write};

use tau_agent::{Agent, AgentEvent};
use tau_ai::{Model, ReasoningLevel};

use crate::{cli::get_available_models, commands, run_command::handle_interaction_stdin, session};

pub(crate) async fn run_interactive(
    agent: &mut Agent,
    model: &mut Model,
    reasoning: &mut ReasoningLevel,
    mut session: Option<session::SessionManager>,
    interaction_rx: tokio::sync::mpsc::Receiver<tau_agent::InteractionRequest>,
) -> anyhow::Result<()> {
    let available_models = get_available_models();
    let _interaction_handle = tokio::spawn(handle_interaction_stdin(interaction_rx));

    if std::io::IsTerminal::is_terminal(&std::io::stderr()) {
        let model_short = model.id.split('/').next_back().unwrap_or(&model.id);
        if let Some(ref s) = session {
            eprintln!("tau ({}) session: {}", model_short, &s.id()[..8]);
        } else {
            eprintln!("tau ({})", model_short);
        }
        eprintln!();
    }

    loop {
        print!("> ");
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
            if let Some(result) =
                commands::execute_command(input, agent, model, *reasoning, &available_models)
            {
                match result {
                    commands::CommandResult::Clear => {
                        agent.clear_messages();
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
                        *model = new_model.clone();
                        agent.set_model(new_model);
                    }
                    commands::CommandResult::ChangeReasoning(level) => {
                        println!("Reasoning level set to: {:?}", level);
                        *reasoning = level;
                        agent.set_reasoning(level);
                    }
                    commands::CommandResult::Unknown(cmd) => {
                        println!("Unknown command: /{}", cmd);
                        println!("Type /help for available commands.");
                    }
                    commands::CommandResult::OpenModelSelector => {
                        println!(
                            "{}",
                            commands::ModelCommand::list_models_text(model, &available_models)
                        );
                    }
                    commands::CommandResult::OpenBranchSelector => {
                        let messages = agent.messages();
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
                        match agent
                            .run_compaction(tau_agent::CompactionReason::Manual)
                            .await
                        {
                            Ok(()) => {
                                println!(
                                    "Context compacted. {} messages remaining.",
                                    agent.messages().len()
                                );
                            }
                            Err(e) => {
                                println!("Compaction failed: {}", e);
                            }
                        }
                    }
                    commands::CommandResult::BranchFrom(branch_index) => {
                        match session::SessionManager::branch_from(
                            agent.messages(),
                            branch_index,
                            &model.id,
                        ) {
                            Ok(new_session) => {
                                let msg_count = branch_index.map(|i| i + 1).unwrap_or(0);
                                println!(
                                    "Created branch: {} ({} messages)",
                                    new_session.id(),
                                    msg_count
                                );
                                if let Some(idx) = branch_index {
                                    let messages: Vec<_> =
                                        agent.messages().iter().take(idx + 1).cloned().collect();
                                    agent.set_messages(messages);
                                } else {
                                    agent.clear_messages();
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

        let mut receiver = agent.subscribe();
        let model_for_cost = model.clone();

        let is_tty = std::io::IsTerminal::is_terminal(&io::stdout());
        let handle = tokio::spawn(async move {
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
                        println!(
                            "[Compacted: ~{} -> ~{} tokens]",
                            tokens_before, tokens_after
                        );
                    }
                    AgentEvent::Error { message } => {
                        eprintln!("\nError: {}", message);
                    }
                    _ => {}
                }
            }
        });

        // Track message count before prompt to save new messages after
        let msgs_before = agent.messages().len();

        if let Err(e) = agent.prompt(input).await {
            eprintln!("Error: {}", e);
        }

        if let Some(ref mut s) = session {
            let all_msgs = agent.messages();
            for msg in all_msgs.iter().skip(msgs_before) {
                let _ = s.append_message(msg);
            }
            let _ = s.append_usage(&agent.state().total_usage);
        }

        match tokio::time::timeout(std::time::Duration::from_secs(2), handle).await {
            Ok(_) => {}
            Err(_) => tracing::debug!("Event handler did not finish in time"),
        }

        println!();
    }

    Ok(())
}
