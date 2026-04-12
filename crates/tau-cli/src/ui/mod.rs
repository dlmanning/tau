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
mod render;
mod state;
mod types;

pub use types::UiMessage;
use state::TuiState;
use types::PendingInteraction;

use crossterm::event::EventStream;
use futures::StreamExt;
use tau_agent::{Agent, AgentEvent};
use tau_ai::Model;
use tau_tui::widgets::{SelectorState, message_list::ChatMessage};
use tokio::sync::mpsc;

/// Apply a branch operation: create a branch session and truncate conversation state.
fn apply_branch(
    state: &mut TuiState,
    agent: &mut Agent,
    model_id: &str,
    branch_index: Option<usize>,
) {
    match crate::session::SessionManager::branch_from(
        agent.messages(),
        branch_index,
        model_id,
    ) {
        Ok(new_session) => {
            let msg_count = branch_index.map(|i| i + 1).unwrap_or(0);
            state.show_system_message(&format!(
                "Created branch: {} ({} messages)\nContinue from this point with a fresh context.",
                new_session.id(),
                msg_count
            ));
            if let Some(idx) = branch_index {
                let messages: Vec<_> = agent.messages().iter().take(idx + 1).cloned().collect();
                agent.set_messages(messages);
                state.messages.truncate(idx + 1);
            } else {
                agent.clear_messages();
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
    agent: &mut Agent,
    model: &mut Model,
    reasoning: &mut tau_ai::ReasoningLevel,
    available_models: &[Model],
    pending_prompt: &mut Option<String>,
) -> bool {
    use crate::commands::{CommandResult, execute_command};

    match msg {
        UiMessage::Submit(content) => {
            *pending_prompt = Some(content);
        }
        UiMessage::Command(cmd) => {
            if let Some(result) = execute_command(&cmd, agent, model, *reasoning, available_models) {
                match result {
                    CommandResult::Message(msg) => {
                        state.show_system_message(&msg);
                    }
                    CommandResult::Clear => {
                        agent.clear_messages();
                        state.messages.clear();
                        state.reset_stats();
                        state.status = "Cleared".to_string();
                    }
                    CommandResult::ChangeModel(new_model) => {
                        state.show_system_message(&format!("Switched to: {}", new_model.id));
                        *model = new_model.clone();
                        state.set_model(new_model.clone());
                        agent.set_model(new_model);
                    }
                    CommandResult::ChangeReasoning(level) => {
                        state.show_system_message(&format!("Reasoning: {:?}", level));
                        *reasoning = level;
                        state.reasoning = level;
                        agent.set_reasoning(level);
                    }
                    CommandResult::Exit => return false,
                    CommandResult::Unknown(cmd) => {
                        state.show_system_message(&format!(
                            "Unknown command: /{}\nType /help for available commands.", cmd
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
                        match agent.run_compaction(tau_agent::CompactionReason::Manual).await {
                            Ok(()) => {
                                state.show_system_message(&format!(
                                    "Context compacted. {} messages remaining.",
                                    agent.messages().len()
                                ));
                            }
                            Err(e) => {
                                state.show_system_message(&format!("Compaction failed: {}", e));
                            }
                        }
                    }
                    CommandResult::BranchFrom(branch_index) => {
                        apply_branch(state, agent, &model.id, branch_index);
                    }
                }
            }
        }
        UiMessage::ChangeModel(index) => {
            if let Some(new_model) = available_models.get(index) {
                state.show_system_message(&format!("Switched to: {}", new_model.id));
                *model = new_model.clone();
                state.set_model(new_model.clone());
                agent.set_model(new_model.clone());
            }
        }
        UiMessage::Clear => {
            agent.clear_messages();
            state.messages.clear();
            state.reset_stats();
            state.status = "Cleared".to_string();
        }
        UiMessage::Abort => {
            agent.abort();
        }
        UiMessage::Branch(branch_index) => {
            apply_branch(state, agent, &model.id, branch_index);
        }
        UiMessage::Quit => return false,
    }
    true
}

/// Run the TUI application
pub async fn run_tui(
    agent: &mut Agent,
    model: &mut Model,
    reasoning: &mut tau_ai::ReasoningLevel,
    available_models: &[Model],
    mut interaction_rx: tokio::sync::mpsc::Receiver<tau_agent::InteractionRequest>,
) -> anyhow::Result<()> {
    use std::io;

    use crossterm::{
        execute,
        event::{EnableBracketedPaste, DisableBracketedPaste, EnableMouseCapture, DisableMouseCapture},
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    };
    use ratatui::{Terminal, backend::CrosstermBackend};

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture, EnableBracketedPaste)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let (ui_tx, mut ui_rx) = mpsc::channel::<UiMessage>(constants::UI_CHANNEL_CAPACITY);

    let mut state = TuiState::new(
        model.clone(),
        *reasoning,
        agent.config().thinking_adaptive,
        available_models.to_vec(),
        ui_tx,
    );

    let mut agent_rx = agent.subscribe();
    let mut event_stream = EventStream::new();

    // Tick interval for animations (80ms for smooth spinner)
    let mut tick_interval = tokio::time::interval(std::time::Duration::from_millis(constants::TICK_INTERVAL_MS));

    // Pending prompt content - we'll process this at the start of the next loop iteration
    // This is stored as a String so it lives long enough for the future
    let mut pending_prompt: Option<String> = None;

    // Captures Ok/Err from various break points in the nested event loops.
    let result = 'outer: loop {
        // If there's a pending prompt, start processing it
        // We create the future here where `content` is still in scope
        if let Some(content) = pending_prompt.take() {
            state.is_processing = true;
            state.status = "Thinking...".to_string();
            state.messages.push(ChatMessage::assistant_streaming(""));
            state.scroll_to_bottom();

            let agent_handle = agent.handle();
            let mut prompt_future = std::pin::pin!(agent.prompt(&content));

            loop {
                terminal.draw(|frame| state.render(frame))?;
                let area_width = terminal.size()?.width;

                tokio::select! {
                    biased;

                    result = &mut prompt_future => {
                        if let Err(e) = result {
                            state.handle_agent_event(AgentEvent::Error { message: e.to_string() });
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
                                if !state.handle_event_while_processing(ev, area_width, &agent_handle) {
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

            continue; // Continue outer loop after prompt completes
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
                            msg, &mut state, agent, model, reasoning,
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
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture, DisableBracketedPaste)?;
    terminal.show_cursor()?;

    result
}
