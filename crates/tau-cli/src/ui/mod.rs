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

use crossterm::event::EventStream;
use futures::StreamExt;
use tau_agent::{AgentEvent, AgentHandle};
use tau_ai::Model;
use widgets::{SelectorState, message_list::ChatMessage};
use tokio::sync::mpsc;

/// Apply a branch operation: create a branch session and truncate conversation state.
async fn apply_branch(
    state: &mut TuiState,
    handle: &AgentHandle,
    branch_index: Option<usize>,
) {
    let Some(config) = handle.config().await else { return };
    let Some(messages) = handle.messages().await else { return };
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
                handle.set_messages(truncated);
                state.messages.truncate(idx + 1);
            } else {
                handle.clear_messages();
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
) -> bool {
    use crate::commands::{CommandContext, CommandResult, execute_command};

    match msg {
        UiMessage::Submit(content) => {
            *pending_prompt = Some(content);
        }
        UiMessage::Command(cmd) => {
            let Some(config) = handle.config().await else { return true };
            let Some(messages) = handle.messages().await else { return true };
            let Some(conv_state) = handle.state().await else { return true };
            let ctx = CommandContext {
                args: "",
                config: &config,
                messages: &messages,
                usage: &conv_state.total_usage,
                available_models,
            };
            if let Some(result) = execute_command(&cmd, &ctx) {
                match result {
                    CommandResult::Message(msg) => {
                        state.show_system_message(&msg);
                    }
                    CommandResult::Clear => {
                        handle.clear_messages();
                        state.messages.clear();
                        state.reset_stats();
                        state.status = "Cleared".to_string();
                    }
                    CommandResult::ChangeModel(new_model) => {
                        state.show_system_message(&format!("Switched to: {}", new_model.id));
                        handle.set_model(new_model);
                    }
                    CommandResult::ChangeReasoning(level) => {
                        state.show_system_message(&format!("Reasoning: {:?}", level));
                        handle.set_reasoning(level);
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
                        match handle
                            .compact(tau_agent::CompactionReason::Manual)
                            .await
                        {
                            Ok(rx) => match rx.await {
                                Ok(r) if r.result.is_ok() => {
                                    let msg_count = handle.messages().await.map(|m| m.len()).unwrap_or(0);
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
                }
            }
        }
        UiMessage::ChangeModel(index) => {
            if let Some(new_model) = available_models.get(index) {
                state.show_system_message(&format!("Switched to: {}", new_model.id));
                handle.set_model(new_model.clone());
            }
        }
        UiMessage::Clear => {
            handle.clear_messages();
            state.messages.clear();
            state.reset_stats();
            state.status = "Cleared".to_string();
        }
        UiMessage::Abort => {
            handle.abort();
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

    let config = handle.config().await.ok_or_else(|| anyhow::anyhow!("Agent shut down"))?;
    let mut state = TuiState::new(
        &config,
        available_models.to_vec(),
        ui_tx,
    );

    let mut agent_rx = handle.subscribe();
    let mut event_stream = EventStream::new();

    let mut tick_interval = tokio::time::interval(std::time::Duration::from_millis(
        constants::TICK_INTERVAL_MS,
    ));

    let mut pending_prompt: Option<String> = None;

    let result = 'outer: loop {
        if let Some(content) = pending_prompt.take() {
            state.is_processing = true;
            state.status = "Thinking...".to_string();
            state.messages.push(ChatMessage::assistant_streaming(""));
            state.scroll_to_bottom();

            if let Some(cfg) = handle.config().await {
                state.sync_from_config(&cfg);
            }

            // Start the prompt — returns a receiver for the completion result.
            // This is the key difference from the old &mut Agent pattern:
            // the handle is &self, so we can keep using it freely.
            let prompt_rx = match handle.prompt(&content).await {
                Ok(rx) => rx,
                Err(e) => {
                    state.handle_agent_event(AgentEvent::Error { message: e.to_string() });
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
                        if let Ok(agent_event) = event {
                            state.handle_agent_event(agent_event);
                        }
                    }

                    event = event_stream.next() => {
                        match event {
                            Some(Ok(ev)) => {
                                if !state.handle_event_while_processing(ev, area_width, handle) {
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
                            use tau_agent::interaction::InteractionKind;
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

        if let Some(cfg) = handle.config().await {
            state.sync_from_config(&cfg);
        }
        terminal.draw(|frame| state.render(frame))?;

        let area_width = terminal.size()?.width;

        tokio::select! {
            biased;

            event = agent_rx.recv() => {
                if let Ok(agent_event) = event {
                    state.handle_agent_event(agent_event);
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
                        if !dispatch_ui_message(
                            msg, &mut state, handle,
                            available_models, &mut pending_prompt,
                        ).await {
                            break Ok(());
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
