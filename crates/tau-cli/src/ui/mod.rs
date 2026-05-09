//! TUI implementation for tau
//!
//! ## Screen layout
//!
//! The interface is divided into four horizontal strips, top to bottom:
//!
//! | Area | Height | Renderer | Contents |
//! |------|--------|----------|----------|
//! | **Header** | 1 | `render_header` | τ glyph (rainbow when processing, green when idle), cwd in `{ }` brackets, clock (MM/DD/YYYY HH:MM:SS AM) |
//! | **Conversation** | flex | `render_conversation` | Message thread — user (▶), assistant (◀), tools (⚙), agents (◇), system (●), steer (▷). Bottom border shows status (Ready/Thinking/Cancelling). |
//! | **Status line** | 1 | `render_status_line` | Model name, thinking level, token counts, cache stats, cost |
//! | **Input** | 3 | `InputBox` widget | Text entry with placeholder |
//!
//! The header style is inspired by the HP 48GX calculator status area.

mod agents;
mod constants;
mod events;
mod input;
mod render;
mod state;
mod theme;
mod types;
mod widgets;

use state::TuiState;
use types::PendingInteraction;
pub use types::UiMessage;

use std::sync::Arc;

use crossterm::event::EventStream;
use futures::StreamExt;
use tau_agent::manager::AgentManager;
use tau_agent::{AgentEvent, AgentHandle};
use tau_ai::Model;
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::mpsc;
use widgets::{SelectorState, message_list::ChatMessage};

/// A non-main agent that the user is currently interacting with.
struct ActiveAgent {
    handle: AgentHandle,
    agent_id: String,
    #[allow(dead_code)] // rendered via TuiState.plan_indicator (Phase 2.1)
    description: String,
}

/// Apply a branch operation: create a branch session and truncate conversation state.
async fn apply_branch(state: &mut TuiState, handle: &AgentHandle, branch_index: Option<usize>) {
    let Some(config) = handle.config().await else {
        return;
    };
    let Some(messages) = handle.messages().await else {
        return;
    };
    match crate::session::SessionManager::branch_from(&messages, branch_index, &config.model.id) {
        Ok(new_session) => {
            let msg_count = branch_index.map(|i| i + 1).unwrap_or(0);
            state.show_system_message(&format!(
                "Created branch: {} ({} messages)\nContinue from this point with a fresh context.",
                new_session.id(),
                msg_count
            ));
            if let Some(idx) = branch_index {
                let truncated: Vec<_> = messages.iter().take(idx + 1).cloned().collect();
                let _ = handle.set_messages(truncated).await;
                state.messages.truncate(idx + 1);
            } else {
                let _ = handle.clear_messages().await;
                state.messages.clear();
            }
            state.reset_stats();
        }
        Err(e) => {
            state.show_system_message(&format!("Failed to create branch: {}", e));
        }
    }
}

/// Dispatch a UI message received from the input handler.
/// Returns `false` if the TUI should exit.
async fn dispatch_ui_message(
    msg: UiMessage,
    state: &mut TuiState,
    handle: &AgentHandle,
    available_models: &[Model],
    pending_prompt: &mut Option<String>,
    active_agent: &mut Option<ActiveAgent>,
    manager: &Arc<AgentManager>,
) -> bool {
    use crate::commands::{CommandContext, CommandResult, execute_command};

    match msg {
        UiMessage::Submit(content) => {
            *pending_prompt = Some(content);
        }
        UiMessage::Command(cmd) => {
            let effective = active_agent.as_ref().map(|a| &a.handle).unwrap_or(handle);
            let Some(config) = effective.config().await else {
                return true;
            };
            let Some(messages) = effective.messages().await else {
                return true;
            };
            let Some(conv_state) = effective.state().await else {
                return true;
            };
            let ctx = CommandContext {
                args: "",
                config: &config,
                messages: &messages,
                usage: &conv_state.total_usage,
                available_models,
                has_active_agent: active_agent.is_some(),
            };
            if let Some(result) = execute_command(&cmd, &ctx) {
                match result {
                    CommandResult::Message(msg) => {
                        state.show_system_message(&msg);
                    }
                    CommandResult::Clear => {
                        let _ = handle.clear_messages().await;
                        state.messages.clear();
                        state.reset_stats();
                        state.status = "Cleared".to_string();
                    }
                    CommandResult::ChangeModel(new_model) => {
                        state.show_system_message(&format!("Switched to: {}", new_model.id));
                        let _ = handle.set_model(new_model).await;
                    }
                    CommandResult::ChangeReasoning(level) => {
                        state.show_system_message(&format!("Reasoning: {:?}", level));
                        let _ = handle.set_reasoning(level).await;
                    }
                    CommandResult::Exit => return false,
                    CommandResult::Unknown(cmd) => {
                        state.show_system_message(&format!(
                            "Unknown command: /{}\nType /help for available commands.",
                            cmd
                        ));
                    }
                    CommandResult::OpenModelSelector => {
                        state.model_selector.show();
                    }
                    CommandResult::OpenBranchSelector => {
                        state.open_branch_selector();
                    }
                    CommandResult::Compact => {
                        state.show_system_message("Compacting context...");
                        match handle.compact(tau_agent::CompactionReason::Manual).await {
                            Ok(rx) => match rx.await {
                                Ok(r) if r.result.is_ok() => {
                                    let msg_count =
                                        handle.messages().await.map(|m| m.len()).unwrap_or(0);
                                    state.show_system_message(&format!(
                                        "Context compacted. {} messages remaining.",
                                        msg_count,
                                    ));
                                }
                                _ => {
                                    state.show_system_message("Compaction failed.");
                                }
                            },
                            Err(e) => {
                                state.show_system_message(&format!("Compaction failed: {}", e));
                            }
                        }
                    }
                    CommandResult::BranchFrom(branch_index) => {
                        apply_branch(state, handle, branch_index).await;
                    }
                    CommandResult::PlanStart(description) => {
                        state.show_system_message("Entering plan mode...");

                        // Build context from main agent conversation
                        let main_messages = handle.messages().await.unwrap_or_default();
                        let main_state = handle.state().await;
                        let prev_summary = main_state
                            .as_ref()
                            .and_then(|s| s.previous_summary.as_deref());
                        let context =
                            tau_tools::plan::build_context_summary(&main_messages, prev_summary);
                        let prompt =
                            tau_tools::plan::build_plan_prompt(&context, &description);

                        // Spawn Plan subagent
                        let request = tau_agent::manager::SpawnRequest {
                            agent_type: tau_agent::manager::AgentType::Plan,
                            prompt: String::new(), // sent via handle.prompt() below
                            description: format!("Planning: {}", description),
                            model: None,
                            cwd: None,
                            isolation: None,
                            depth: 0,
                            inherit_history_from: None,
                            approval_policy: None,
                        };

                        match manager.spawn_interactive(request).await {
                            Ok((plan_handle, agent_id)) => {
                                let desc = format!("Planning: {}", description);
                                *active_agent = Some(ActiveAgent {
                                    handle: plan_handle,
                                    agent_id,
                                    description: desc.clone(),
                                });
                                state.show_system_message(&format!(
                                    "Plan mode active: {}\nUse /plan approve when done, or /plan exit to cancel.",
                                    description
                                ));
                                // Send context + task as the first prompt
                                *pending_prompt = Some(prompt);
                            }
                            Err(e) => {
                                state.show_system_message(&format!(
                                    "Failed to start plan mode: {}",
                                    e
                                ));
                            }
                        }
                    }
                    CommandResult::PlanApprove => {
                        if let Some(agent) = active_agent.as_ref() {
                            let plan_text = agent
                                .handle
                                .messages()
                                .await
                                .map(|m| tau_tools::plan::extract_final_text(&m))
                                .unwrap_or_default();
                            if plan_text.trim().is_empty() {
                                state.show_system_message(
                                    "Plan agent has no plan to approve yet. Wait for it to respond, or use /plan exit to discard.",
                                );
                            } else {
                                let agent = active_agent.take().unwrap();
                                manager.remove_interactive(&agent.agent_id).await;
                                let _ = handle
                                    .steer(tau_ai::Message::user(format!(
                                        "Approved plan:\n\n{}\n\nProceed with implementation.",
                                        plan_text
                                    )))
                                    .await;
                                state.show_system_message(
                                    "Plan approved. Returned to main agent.",
                                );
                            }
                        }
                    }
                    CommandResult::PlanExit => {
                        if let Some(agent) = active_agent.take() {
                            manager.remove_interactive(&agent.agent_id).await;
                            state.show_system_message("Exited plan mode.");
                        }
                    }
                }
            }
        }
        UiMessage::ChangeModel(index) => {
            if let Some(new_model) = available_models.get(index) {
                state.show_system_message(&format!("Switched to: {}", new_model.id));
                let _ = handle.set_model(new_model.clone()).await;
            }
        }
        UiMessage::Clear => {
            let _ = handle.clear_messages().await;
            state.messages.clear();
            state.reset_stats();
            state.status = "Cleared".to_string();
        }
        UiMessage::Abort => {
            if let Some(ref agent) = *active_agent {
                agent.handle.abort();
            } else {
                handle.abort();
            }
        }
        UiMessage::Branch(branch_index) => {
            apply_branch(state, handle, branch_index).await;
        }
        UiMessage::Quit => return false,
    }
    true
}

/// Run the TUI application
pub async fn run_tui(
    handle: &AgentHandle,
    available_models: &[Model],
    mut interaction_rx: tokio::sync::mpsc::Receiver<tau_agent::InteractionRequest>,
    manager: Arc<AgentManager>,
) -> anyhow::Result<()> {
    use std::io;

    use crossterm::{
        event::{
            DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        },
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    };
    use ratatui::{Terminal, backend::CrosstermBackend};

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        EnableBracketedPaste
    )?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let (ui_tx, mut ui_rx) = mpsc::channel::<UiMessage>(constants::UI_CHANNEL_CAPACITY);

    let config = handle
        .config()
        .await
        .ok_or_else(|| anyhow::anyhow!("Agent shut down"))?;
    let mut state = TuiState::new(&config, available_models.to_vec(), ui_tx);

    let mut agent_rx = handle.subscribe();
    let mut event_stream = EventStream::new();

    let mut tick_interval = tokio::time::interval(std::time::Duration::from_millis(
        constants::TICK_INTERVAL_MS,
    ));

    let mut pending_prompt: Option<String> = None;
    let mut active_agent: Option<ActiveAgent> = None;

    let result = 'outer: loop {
        // Determine which handle receives prompts and events
        let effective_handle = active_agent
            .as_ref()
            .map(|a| &a.handle)
            .unwrap_or(handle);

        if let Some(content) = pending_prompt.take() {
            // Re-subscribe to the effective handle's events for this prompt
            agent_rx = effective_handle.subscribe();

            state.is_processing = true;
            state.status = if active_agent.is_some() {
                "Planning...".to_string()
            } else {
                "Thinking...".to_string()
            };
            state.messages.push(ChatMessage::assistant_streaming(""));
            state.scroll_to_bottom();

            if let Some(cfg) = effective_handle.config().await {
                state.sync_from_config(&cfg);
            }

            let prompt_rx = match effective_handle.prompt(&content).await {
                Ok(rx) => rx,
                Err(e) => {
                    state.handle_agent_event(AgentEvent::Error {
                        message: e.to_string(),
                    });
                    state.is_processing = false;
                    continue;
                }
            };
            let mut prompt_rx = prompt_rx;

            loop {
                terminal.draw(|frame| state.render(frame))?;
                let area_width = terminal.size()?.width;

                tokio::select! {
                    biased;

                    result = &mut prompt_rx => {
                        match result {
                            Ok(r) => {
                                if let Err(e) = r.result {
                                    state.handle_agent_event(AgentEvent::Error { message: e.to_string() });
                                }
                            }
                            Err(_) => {
                                state.handle_agent_event(AgentEvent::Error { message: "Agent task dropped".into() });
                            }
                        }
                        state.pending_interaction = None;
                        break;
                    }

                    event = agent_rx.recv() => {
                        match event {
                            Ok(ev) => state.handle_agent_event(ev),
                            Err(RecvError::Lagged(n)) => {
                                tracing::warn!(lagged = n, "TUI event subscriber lagged; {n} agent event(s) dropped");
                            }
                            Err(RecvError::Closed) => break,
                        }
                    }

                    event = event_stream.next() => {
                        match event {
                            Some(Ok(ev)) => {
                                if !state.handle_event_while_processing(ev, area_width, effective_handle) {
                                    break 'outer Ok(());
                                }
                            }
                            Some(Err(_)) | None => {
                                break 'outer Ok(());
                            }
                        }
                    }

                    request = interaction_rx.recv() => {
                        if let Some(request) = request {
                            use tau_agent::interaction::{InteractionKind, InteractionResponse};
                            match request.kind {
                                InteractionKind::AskQuestion { question, options } => {
                                    state.status = "Waiting for your choice...".to_string();
                                    state.pending_interaction = Some(PendingInteraction {
                                        question,
                                        options,
                                        response_tx: request.response_tx,
                                        selector: SelectorState::default(),
                                    });
                                }
                                InteractionKind::Typed { schema_id, payload } => match schema_id.as_str() {
                                    "tool.confirm" => {
                                        // Should not fire: main.rs installs AutoAcceptAllPolicy
                                        // for the TUI until a confirm UI is built.
                                        let tool_name = payload
                                            .get("tool_name")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("?");
                                        state.status =
                                            format!("Unexpected tool.confirm for {tool_name}");
                                        let _ = request.response_tx.send(InteractionResponse::Rejected {
                                            reason: "TUI confirm not implemented".into(),
                                        });
                                    }
                                    "plan.submit" => {
                                        let step_count = payload
                                            .get("items")
                                            .and_then(|v| v.as_array())
                                            .map(|a| a.len())
                                            .unwrap_or(0);
                                        state.status = format!(
                                            "Plan submitted ({step_count} step(s)) — auto-approving (TUI plan UI not implemented)"
                                        );
                                        let _ = request
                                            .response_tx
                                            .send(InteractionResponse::Approved { payload: None });
                                    }
                                    other => {
                                        state.status = format!("Unknown typed interaction: {other}");
                                        let _ = request.response_tx.send(InteractionResponse::Rejected {
                                            reason: format!("Unknown schema: {other}"),
                                        });
                                    }
                                },
                            }
                        }
                    }

                    _ = tick_interval.tick() => {
                        state.git_branch.poll();
                        state.git_branch.maybe_refresh();
                    }
                }
            }

            while let Ok(agent_event) = agent_rx.try_recv() {
                state.handle_agent_event(agent_event);
            }

            terminal.draw(|frame| state.render(frame))?;

            continue;
        }

        if let Some(cfg) = effective_handle.config().await {
            state.sync_from_config(&cfg);
        }
        terminal.draw(|frame| state.render(frame))?;

        let area_width = terminal.size()?.width;

        tokio::select! {
            biased;

            event = agent_rx.recv() => {
                match event {
                    Ok(ev) => state.handle_agent_event(ev),
                    Err(RecvError::Lagged(n)) => {
                        tracing::warn!(lagged = n, "TUI event subscriber lagged; {n} agent event(s) dropped");
                    }
                    Err(RecvError::Closed) => break Ok(()),
                }
            }

            event = event_stream.next() => {
                match event {
                    Some(Ok(ev)) => {
                        if !state.handle_event_while_idle(ev, area_width).await {
                            break Ok(());
                        }
                    }
                    Some(Err(e)) => {
                        break Err(anyhow::anyhow!("Event error: {}", e));
                    }
                    None => {
                        break Ok(());
                    }
                }
            }

            _ = tick_interval.tick() => {
                state.git_branch.poll();
                state.git_branch.maybe_refresh();
            }

            msg = ui_rx.recv() => {
                match msg {
                    Some(msg) => {
                        let prev_id = active_agent.as_ref().map(|a| a.agent_id.clone());
                        if !dispatch_ui_message(
                            msg, &mut state, handle,
                            available_models, &mut pending_prompt,
                            &mut active_agent, &manager,
                        ).await {
                            break Ok(());
                        }
                        let new_id = active_agent.as_ref().map(|a| a.agent_id.clone());
                        if prev_id != new_id {
                            let new_effective = active_agent
                                .as_ref()
                                .map(|a| &a.handle)
                                .unwrap_or(handle);
                            agent_rx = new_effective.subscribe();
                        }
                    }
                    None => break Ok(()),
                }
            }
        }
    };

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture,
        DisableBracketedPaste
    )?;
    terminal.show_cursor()?;

    result
}
