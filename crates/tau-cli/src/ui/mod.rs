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
mod state;
mod types;

pub use types::UiMessage;
use state::TuiState;
use types::{PendingInteraction, rainbow_tau_style};

use crossterm::event::EventStream;
use futures::StreamExt;
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
};
use tau_agent::{Agent, AgentEvent};
use tau_ai::Model;
use tau_tui::widgets::{
    MessageList, OwnedSelector, OwnedSelectorItem, Selector, SelectorItem,
    SelectorState, message_list::ChatMessage,
};
use tokio::sync::mpsc;

use crate::utils::format_tokens;

impl TuiState {
    /// Render the UI
    pub fn render(&mut self, frame: &mut Frame) {
        let size = frame.area();

        // Layout: header (1), conversation (flex), status line (1), input (3)
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Header: τ glyph, cwd, clock
                Constraint::Min(1),    // Conversation: message thread
                Constraint::Length(1), // Status line: model, thinking, tokens, cost
                Constraint::Length(3), // Input: text entry
            ])
            .split(size);

        self.render_header(frame, chunks[0]);
        self.render_conversation(frame, chunks[1]);
        self.render_status_line(frame, chunks[2]);
        self.input
            .render(chunks[3], frame.buffer_mut(), &self.theme);

        if self.pending_interaction.is_some() {
            self.render_question_selector(frame, size);
        }

        if self.model_selector.visible {
            self.render_model_selector(frame, size);
        }

        if self.branch_selector.visible {
            self.render_branch_selector(frame, size);
        }
    }

    /// Render the question selector popup
    fn render_question_selector(&self, frame: &mut Frame, area: Rect) {
        let Some(pi) = self.pending_interaction.as_ref() else {
            return;
        };
        let items: Vec<OwnedSelectorItem> = pi
            .options
            .iter()
            .map(|opt| OwnedSelectorItem {
                label: opt.label.clone(),
                description: Some(opt.description.clone()),
                is_current: false,
            })
            .collect();

        let selector = OwnedSelector::new(&pi.question, items, &self.theme)
            .with_selected(pi.selector.selected);

        selector.render_centered(area, frame.buffer_mut());
    }

    /// Render the model selector popup
    fn render_model_selector(&self, frame: &mut Frame, area: Rect) {
        let items: Vec<SelectorItem> = self
            .available_models
            .iter()
            .map(|m| {
                SelectorItem {
                    label: &m.name,
                    description: Some(m.provider.name()),
                    is_current: m.id == self.model.id,
                }
            })
            .collect();

        let selector = Selector::new("Select Model", items, &self.theme)
            .with_selected(self.model_selector.selected);

        selector.render_centered(area, frame.buffer_mut());
    }

    /// Render the branch selector popup
    fn render_branch_selector(&self, frame: &mut Frame, area: Rect) {
        let items: Vec<OwnedSelectorItem> = self
            .messages
            .iter()
            .enumerate()
            .map(|(i, msg)| {
                let preview = crate::utils::truncate_chars(&msg.content, constants::BRANCH_PREVIEW_CHARS);
                let preview = preview.replace('\n', " ");
                OwnedSelectorItem {
                    label: format!("{}: [{}] {}", i, msg.role, preview),
                    description: None,
                    is_current: false,
                }
            })
            .collect();

        let selector = OwnedSelector::new("Branch from message", items, &self.theme)
            .with_selected(self.branch_selector.selected);

        selector.render_centered(area, frame.buffer_mut());
    }

    fn render_conversation(&mut self, frame: &mut Frame, area: Rect) {
        let status_style = if self.is_processing {
            Style::default().fg(Color::Yellow)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(self.theme.border_style())
            .title_bottom(Line::from(vec![
                Span::raw(" "),
                Span::styled(&self.status, status_style),
                Span::raw(" "),
            ]));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        if inner.height == 0 || self.messages.is_empty() {
            let model_name = &self.model.name;
            let welcome = Paragraph::new(vec![
                Line::from(""),
                Line::from(vec![
                    Span::styled(
                        "  τ ",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        "tau",
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        " - AI coding assistant",
                        Style::default().fg(Color::DarkGray),
                    ),
                ]),
                Line::from(""),
                Line::from(Span::styled(
                    format!("  Model: {}", model_name),
                    Style::default().fg(Color::DarkGray),
                )),
                Line::from(""),
                Line::from(""),
                Line::from(Span::styled(
                    "  Keybindings",
                    Style::default().fg(Color::Yellow),
                )),
                Line::from(""),
                Line::from(vec![
                    Span::styled("    Enter     ", Style::default().fg(Color::Cyan)),
                    Span::styled("Send message", Style::default().fg(Color::White)),
                ]),
                Line::from(vec![
                    Span::styled("    Ctrl+K    ", Style::default().fg(Color::Cyan)),
                    Span::styled("Select model", Style::default().fg(Color::White)),
                ]),
                Line::from(vec![
                    Span::styled("    Ctrl+L    ", Style::default().fg(Color::Cyan)),
                    Span::styled("Clear conversation", Style::default().fg(Color::White)),
                ]),
                Line::from(vec![
                    Span::styled("    Ctrl+C    ", Style::default().fg(Color::Cyan)),
                    Span::styled("Abort / Quit", Style::default().fg(Color::White)),
                ]),
                Line::from(vec![
                    Span::styled("    PgUp/Dn   ", Style::default().fg(Color::Cyan)),
                    Span::styled("Scroll history", Style::default().fg(Color::White)),
                ]),
                Line::from(""),
                Line::from(""),
                Line::from(Span::styled(
                    "  Type a message to get started...",
                    Style::default().fg(Color::DarkGray),
                )),
            ]);
            frame.render_widget(welcome, inner);
            return;
        }

        // Calculate scroll
        let content_height = tau_tui::widgets::message_list::calculate_message_height(
            &self.messages,
            inner.width as usize,
            &self.theme,
        );

        let max_scroll = content_height.saturating_sub(inner.height as usize);
        if self.follow_bottom {
            self.scroll = max_scroll;
        } else {
            self.scroll = self.scroll.min(max_scroll);
            // Re-pin if user scrolled to the bottom
            if self.scroll >= max_scroll {
                self.follow_bottom = true;
            }
        }

        let message_list = MessageList::new(&self.messages, &self.theme).scroll(self.scroll);
        frame.render_widget(message_list, inner);

        if content_height > inner.height as usize {
            let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("↑"))
                .end_symbol(Some("↓"))
                .track_symbol(Some("│"))
                .thumb_symbol("█");

            let mut scrollbar_state = ScrollbarState::new(content_height)
                .position(self.scroll)
                .viewport_content_length(inner.height as usize);

            frame.render_stateful_widget(scrollbar, inner, &mut scrollbar_state);
        }
    }

    fn render_header(&mut self, frame: &mut Frame, area: Rect) {
        let cwd = std::env::current_dir()
            .ok()
            .and_then(|p| {
                if let Some(home) = dirs::home_dir() {
                    if let Ok(rest) = p.strip_prefix(&home) {
                        return Some(format!("~/{}", rest.display()));
                    }
                }
                Some(p.display().to_string())
            })
            .unwrap_or_default();

        self.git_branch.poll();
        self.git_branch.maybe_refresh();

        let info_content = match &self.git_branch.branch {
            Some(b) => format!("{{ {} · {} }}", cwd, b),
            None => format!("{{ {} }}", cwd),
        };

        // τ glyph — rainbow cycle when processing, dim green when idle
        let tau_style = if self.is_processing {
            rainbow_tau_style()
        } else {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        };

        // Clock: MM/DD/YYYY HH:MM:SS AM
        let now = chrono::Local::now();
        let right_content = now.format("%m/%d/%Y %I:%M:%S%p").to_string();

        let left_width = 2 + info_content.chars().count();
        let right_width = right_content.chars().count();
        let available = area.width as usize;

        let dim = Style::default().fg(Color::DarkGray);

        let line = if left_width + right_width + 2 <= available {
            let spacing = available - left_width - right_width;
            Line::from(vec![
                Span::styled("τ ", tau_style),
                Span::styled(&info_content, dim),
                Span::raw(" ".repeat(spacing)),
                Span::styled(&right_content, dim),
            ])
        } else {
            Line::from(vec![
                Span::styled("τ ", tau_style),
                Span::styled(&info_content, dim),
            ])
        };

        frame.render_widget(Paragraph::new(line), area);
    }

    fn render_status_line(&self, frame: &mut Frame, area: Rect) {
        let dim = Style::default().fg(Color::DarkGray);
        let mut parts: Vec<Span> = Vec::new();

        // Model name
        parts.push(Span::styled(&self.model.name, dim));

        // Thinking level
        let thinking_str = match self.reasoning {
            tau_ai::ReasoningLevel::Off => None,
            level => {
                let name = match level {
                    tau_ai::ReasoningLevel::Minimal => "min",
                    tau_ai::ReasoningLevel::Low => "low",
                    tau_ai::ReasoningLevel::Medium => "med",
                    tau_ai::ReasoningLevel::High => "high",
                    tau_ai::ReasoningLevel::Off => unreachable!(),
                };
                if self.thinking_adaptive {
                    Some(format!("think:{}/a", name))
                } else {
                    Some(format!("think:{}", name))
                }
            }
        };
        if let Some(t) = thinking_str {
            parts.push(Span::styled(" · ", dim));
            parts.push(Span::styled(t, dim));
        }

        // Token stats
        if self.usage.input_tokens > 0 || self.usage.output_tokens > 0 {
            parts.push(Span::styled(" · ", dim));
            parts.push(Span::styled(
                format!("{} in, {} out", format_tokens(self.usage.input_tokens), format_tokens(self.usage.output_tokens)),
                dim,
            ));

            if self.usage.cache_read > 0 || self.usage.cache_write > 0 {
                parts.push(Span::styled(" · ", dim));
                parts.push(Span::styled(
                    format!("cache: {}r {}w", format_tokens(self.usage.cache_read), format_tokens(self.usage.cache_write)),
                    dim,
                ));
            }

            if self.usage.cost > 0.0 {
                parts.push(Span::styled(" · ", dim));
                parts.push(Span::styled(format!("${:.4}", self.usage.cost), dim));
            }
        }

        frame.render_widget(Paragraph::new(Line::from(parts)), area);
    }
}

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

                    _ = tick_interval.tick() => {}
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

            _ = tick_interval.tick() => {}

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
