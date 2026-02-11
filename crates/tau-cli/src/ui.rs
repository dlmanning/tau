//! TUI implementation for tau

use tokio::sync::mpsc;

use crossterm::event::{Event, EventStream, MouseEventKind};
use futures::StreamExt;
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
};
use std::time::Instant;
use tau_agent::{Agent, AgentEvent};
use tau_ai::Model;
use tau_tui::{
    Theme,
    input::Action,
    widgets::{
        InputBox, MessageList, OwnedSelector, OwnedSelectorItem, Selector, SelectorItem,
        SelectorState, Spinner, message_list::ChatMessage,
    },
};

/// Messages sent from UI to agent handler
#[derive(Debug)]
pub enum UiMessage {
    /// User submitted input
    Submit(String),
    /// User requested quit
    Quit,
    /// User requested clear
    Clear,
    /// User requested abort of current operation
    Abort,
    /// Slash command
    Command(String),
    /// Change model (index into available_models)
    ChangeModel(usize),
    /// Create branch from message index (None = empty branch)
    Branch(Option<usize>),
}

/// TUI application state
pub struct TuiState {
    /// Chat messages
    messages: Vec<ChatMessage>,
    /// Input box
    input: InputBox,
    /// Current scroll position
    scroll: usize,
    /// Whether agent is currently processing
    is_processing: bool,
    /// Current status message
    status: String,
    /// Theme
    theme: Theme,
    /// Total tokens used
    total_input_tokens: u32,
    total_output_tokens: u32,
    /// Model for cost calculation
    model: Model,
    /// Available models for selection
    available_models: Vec<Model>,
    /// Total cost
    total_cost: f64,
    /// Channel to send messages to agent handler
    ui_tx: mpsc::Sender<UiMessage>,
    /// Spinner start time for animation
    spinner_start: Instant,
    /// Model selector state
    model_selector: SelectorState,
    /// Branch selector state
    branch_selector: SelectorState,
}

impl TuiState {
    pub fn new(model: Model, available_models: Vec<Model>, ui_tx: mpsc::Sender<UiMessage>) -> Self {
        let mut input = InputBox::new().with_placeholder("Type a message...");
        input.set_focused(true);

        // Find the current model's index in available models
        let current_index = available_models
            .iter()
            .position(|m| m.id == model.id)
            .unwrap_or(0);

        let model_selector = SelectorState {
            selected: current_index,
            ..Default::default()
        };

        Self {
            messages: vec![],
            input,
            scroll: 0,
            is_processing: false,
            status: "Ready".to_string(),
            theme: Theme::dark(),
            total_input_tokens: 0,
            total_output_tokens: 0,
            model,
            available_models,
            total_cost: 0.0,
            ui_tx,
            spinner_start: Instant::now(),
            model_selector,
            branch_selector: SelectorState::default(),
        }
    }

    /// Open the branch selector popup
    pub fn open_branch_selector(&mut self) {
        if !self.messages.is_empty() {
            self.branch_selector.selected = self.messages.len().saturating_sub(1);
            self.branch_selector.show();
        }
    }

    /// Handle agent events
    pub fn handle_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::AgentStart => {
                // Already handled when Submit is received - this is just a confirmation
                self.is_processing = true;
            }
            AgentEvent::MessageUpdate { message } => {
                let text = message.text();
                // Update the streaming message
                if let Some(last) = self.messages.last_mut() {
                    if last.is_streaming {
                        last.content = text;
                        self.scroll_to_bottom();
                        return;
                    }
                }
                self.messages.push(ChatMessage::assistant_streaming(text));
                self.scroll_to_bottom();
            }
            AgentEvent::MessageEnd { message } => {
                // Replace streaming message with final
                if let Some(last) = self.messages.last_mut() {
                    if last.is_streaming {
                        last.content = message.text();
                        last.is_streaming = false;
                        return;
                    }
                }
                self.messages.push(ChatMessage::assistant(message.text()));
                self.scroll_to_bottom();
            }
            AgentEvent::ToolExecutionStart { tool_name, .. } => {
                self.status = format!("Running {}...", tool_name);
            }
            AgentEvent::ToolExecutionEnd {
                tool_name,
                result,
                is_error,
                ..
            } => {
                // Use chars for proper Unicode handling
                let result_chars: Vec<char> = result.chars().collect();
                let preview = if result_chars.len() > 200 {
                    let truncated: String = result_chars[..200].iter().collect();
                    format!("{}...", truncated)
                } else {
                    result
                };
                self.messages
                    .push(ChatMessage::tool(&tool_name, preview, is_error));
                self.scroll_to_bottom();
            }
            AgentEvent::TurnEnd { usage, .. } => {
                self.total_input_tokens += usage.input;
                self.total_output_tokens += usage.output;
                let cost = usage.calculate_cost(&self.model);
                self.total_cost += cost.total;
            }
            AgentEvent::AgentEnd { .. } => {
                self.is_processing = false;
                self.status = format!(
                    "Ready | {} in, {} out | ${:.4}",
                    self.total_input_tokens, self.total_output_tokens, self.total_cost
                );
            }
            AgentEvent::Error { message } => {
                self.is_processing = false;
                self.status = format!("Error: {}", message);
                self.messages.push(ChatMessage {
                    role: "system".to_string(),
                    content: format!("Error: {}", message),
                    is_error: true,
                    is_streaming: false,
                });
            }
            // Ignore turn/message start events (we handle updates/ends)
            AgentEvent::TurnStart { .. } | AgentEvent::MessageStart { .. } => {}
        }
    }

    fn scroll_to_bottom(&mut self) {
        // Will be calculated during render based on content height
        self.scroll = usize::MAX;
    }

    /// Show a system message
    pub fn show_system_message(&mut self, content: &str) {
        self.messages.push(ChatMessage::system(content));
        self.scroll_to_bottom();
    }

    /// Update the model
    pub fn set_model(&mut self, model: Model) {
        self.model = model;
    }

    /// Handle keyboard action
    pub async fn handle_action(&mut self, action: Action, width: u16) -> bool {
        // Handle branch selector if visible
        if self.branch_selector.visible {
            match action {
                Action::Up => {
                    self.branch_selector.up(self.messages.len());
                    return true;
                }
                Action::Down => {
                    self.branch_selector.down(self.messages.len());
                    return true;
                }
                Action::Submit => {
                    // Create branch from selected message
                    let selected = self.branch_selector.selected;
                    self.branch_selector.hide();
                    let _ = self.ui_tx.send(UiMessage::Branch(Some(selected))).await;
                    return true;
                }
                Action::Escape => {
                    // Close without branching
                    self.branch_selector.hide();
                    return true;
                }
                _ => {
                    // Ignore other actions while selector is open
                    return true;
                }
            }
        }

        // Handle model selector if visible
        if self.model_selector.visible {
            match action {
                Action::Up => {
                    self.model_selector.up(self.available_models.len());
                    return true;
                }
                Action::Down => {
                    self.model_selector.down(self.available_models.len());
                    return true;
                }
                Action::Submit => {
                    // Select the model and close
                    let selected = self.model_selector.selected;
                    self.model_selector.hide();
                    let _ = self.ui_tx.send(UiMessage::ChangeModel(selected)).await;
                    return true;
                }
                Action::Escape | Action::ModelSelect => {
                    // Close without selecting
                    self.model_selector.hide();
                    return true;
                }
                _ => {
                    // Ignore other actions while selector is open
                    return true;
                }
            }
        }

        match action {
            Action::Submit => {
                let content = self.input.content().to_string();
                if !content.is_empty() && !self.is_processing {
                    self.input.clear();

                    if content.starts_with('/') {
                        // Handle slash command
                        let _ = self.ui_tx.send(UiMessage::Command(content)).await;
                    } else {
                        // Regular message
                        self.messages.push(ChatMessage::user(&content));
                        self.scroll_to_bottom();
                        let _ = self.ui_tx.send(UiMessage::Submit(content)).await;
                    }
                }
                true
            }
            Action::Quit => {
                let _ = self.ui_tx.send(UiMessage::Quit).await;
                false
            }
            Action::Interrupt => {
                if self.is_processing {
                    // Cancel current operation
                    let _ = self.ui_tx.send(UiMessage::Abort).await;
                    self.status = "Cancelling...".to_string();
                    true
                } else {
                    let _ = self.ui_tx.send(UiMessage::Quit).await;
                    false
                }
            }
            Action::Escape => {
                if self.is_processing {
                    // Cancel current operation
                    let _ = self.ui_tx.send(UiMessage::Abort).await;
                    self.status = "Cancelling...".to_string();
                    true
                } else {
                    let _ = self.ui_tx.send(UiMessage::Quit).await;
                    false
                }
            }
            Action::PageUp => {
                self.scroll = self.scroll.saturating_sub(10);
                true
            }
            Action::PageDown => {
                self.scroll = self.scroll.saturating_add(10);
                true
            }
            Action::Clear => {
                let _ = self.ui_tx.send(UiMessage::Clear).await;
                self.messages.clear();
                self.total_input_tokens = 0;
                self.total_output_tokens = 0;
                self.total_cost = 0.0;
                self.status = "Ready".to_string();
                true
            }
            Action::ModelSelect => {
                // Open model selector (only when not processing)
                if !self.is_processing {
                    self.model_selector.show();
                }
                true
            }
            _ => {
                self.input.handle_action(&action, width);
                true
            }
        }
    }

    /// Render the UI
    pub fn render(&mut self, frame: &mut Frame) {
        let size = frame.area();

        // Layout: messages (flex), status bar (1), input (3)
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(1),    // Messages
                Constraint::Length(1), // Status
                Constraint::Length(3), // Input
            ])
            .split(size);

        // Render messages
        self.render_messages(frame, chunks[0]);

        // Render status bar
        self.render_status(frame, chunks[1]);

        // Render input
        self.input
            .render(chunks[2], frame.buffer_mut(), &self.theme);

        // Render model selector popup if visible
        if self.model_selector.visible {
            self.render_model_selector(frame, size);
        }

        // Render branch selector popup if visible
        if self.branch_selector.visible {
            self.render_branch_selector(frame, size);
        }
    }

    /// Render the model selector popup
    fn render_model_selector(&self, frame: &mut Frame, area: Rect) {
        let items: Vec<SelectorItem> = self
            .available_models
            .iter()
            .map(|m| {
                let label = m.id.split('/').next_back().unwrap_or(&m.id);
                SelectorItem {
                    label,
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
                // Truncate content for display
                let content_chars: Vec<char> = msg.content.chars().collect();
                let preview = if content_chars.len() > 50 {
                    let truncated: String = content_chars[..50].iter().collect();
                    format!("{}...", truncated)
                } else {
                    msg.content.clone()
                };
                // Replace newlines with spaces for single-line display
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

    fn render_messages(&mut self, frame: &mut Frame, area: Rect) {
        let model_name = self
            .model
            .id
            .split('/')
            .next_back()
            .unwrap_or(&self.model.id);
        let title = format!(" tau │ {} ", model_name);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(self.theme.border_style())
            .title(title);

        let inner = block.inner(area);
        frame.render_widget(block, area);

        if inner.height == 0 || self.messages.is_empty() {
            // Show welcome screen with ASCII art and help
            let model_name = self
                .model
                .id
                .split('/')
                .next_back()
                .unwrap_or(&self.model.id);
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
        );

        if self.scroll == usize::MAX {
            // Auto-scroll to bottom
            self.scroll = content_height.saturating_sub(inner.height as usize);
        } else {
            // Clamp scroll
            self.scroll = self
                .scroll
                .min(content_height.saturating_sub(inner.height as usize));
        }

        let message_list = MessageList::new(&self.messages, &self.theme).scroll(self.scroll);
        frame.render_widget(message_list, inner);

        // Render scrollbar if content overflows
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

    fn render_status(&self, frame: &mut Frame, area: Rect) {
        if self.is_processing {
            // Use animated spinner during processing
            let spinner =
                Spinner::new(&self.status, &self.theme).with_start_time(self.spinner_start);
            frame.render_widget(spinner, area);
        } else {
            // Show status with keybindings help on the right
            let model_name = self
                .model
                .id
                .split('/')
                .next_back()
                .unwrap_or(&self.model.id);
            let left_content = format!("{} │ {}", model_name, self.status);
            let right_content = "Ctrl+K: model │ Ctrl+L: clear │ Ctrl+C: quit";

            let left_width = left_content.chars().count();
            let right_width = right_content.chars().count();
            let available = area.width as usize;

            // Build the line with spacing
            let line = if left_width + right_width + 2 <= available {
                let spacing = available - left_width - right_width;
                Line::from(vec![
                    Span::styled(&left_content, self.theme.dim_style()),
                    Span::raw(" ".repeat(spacing)),
                    Span::styled(right_content, Style::default().fg(Color::DarkGray)),
                ])
            } else {
                Line::from(Span::styled(&left_content, self.theme.dim_style()))
            };

            let status = Paragraph::new(line);
            frame.render_widget(status, area);
        }
    }
}

/// Run the TUI application
pub async fn run_tui(
    agent: &mut Agent,
    model: &mut Model,
    reasoning: &mut tau_ai::ReasoningLevel,
    available_models: &[Model],
) -> anyhow::Result<()> {
    use crate::commands::{CommandResult, execute_command};
    use crossterm::{
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    };
    use ratatui::{Terminal, backend::CrosstermBackend};
    use std::io;

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create channels
    let (ui_tx, mut ui_rx) = mpsc::channel::<UiMessage>(32);

    // Create state
    let mut state = TuiState::new(model.clone(), available_models.to_vec(), ui_tx);

    // Subscribe to agent events
    let mut agent_rx = agent.subscribe();

    // Event stream
    let mut event_stream = EventStream::new();

    // Tick interval for animations (80ms for smooth spinner)
    let mut tick_interval = tokio::time::interval(std::time::Duration::from_millis(80));

    // Pending prompt content - we'll process this at the start of the next loop iteration
    // This is stored as a String so it lives long enough for the future
    let mut pending_prompt: Option<String> = None;

    let result = loop {
        // If there's a pending prompt, start processing it
        // We create the future here where `content` is still in scope
        if let Some(content) = pending_prompt.take() {
            // Show thinking indicator
            state.is_processing = true;
            state.spinner_start = Instant::now();
            state.status = "Thinking...".to_string();
            state.messages.push(ChatMessage::assistant_streaming(""));
            state.scroll_to_bottom();

            // Get cancel handle before creating the future (so we can cancel without borrowing agent)
            let cancel_handle = agent.cancel_handle();

            // Create the prompt future
            let mut prompt_future = std::pin::pin!(agent.prompt(&content));

            // Poll it alongside other events until completion
            loop {
                // Render each iteration to show spinner animation
                terminal.draw(|frame| state.render(frame))?;
                let area_width = terminal.size()?.width;

                tokio::select! {
                    biased;

                    // Poll the prompt future
                    result = &mut prompt_future => {
                        if let Err(e) = result {
                            state.handle_agent_event(AgentEvent::Error { message: e });
                        }
                        break; // Exit inner loop, prompt is done
                    }

                    // Handle agent events (highest priority for responsiveness)
                    event = agent_rx.recv() => {
                        if let Ok(agent_event) = event {
                            state.handle_agent_event(agent_event);
                        }
                    }

                    // Handle terminal events - input works during processing!
                    event = event_stream.next() => {
                        match event {
                            Some(Ok(Event::Key(key))) => {
                                let action = tau_tui::input::key_to_action(key);
                                // During processing, only handle interrupt/quit differently
                                match action {
                                    Action::Interrupt | Action::Escape => {
                                        // Cancel using the handle (doesn't need to borrow agent)
                                        cancel_handle.lock().cancel();
                                        state.status = "Cancelling...".to_string();
                                    }
                                    Action::Quit => {
                                        // Cleanup and exit
                                        disable_raw_mode()?;
                                        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
                                        terminal.show_cursor()?;
                                        return Ok(());
                                    }
                                    _ => {
                                        // Still allow typing during processing
                                        state.input.handle_action(&action, area_width);
                                    }
                                }
                            }
                            Some(Ok(Event::Paste(text))) => {
                                state.input.handle_action(&Action::Paste(text), area_width);
                            }
                            Some(Ok(Event::Mouse(mouse))) => {
                                match mouse.kind {
                                    MouseEventKind::ScrollUp => {
                                        state.scroll = state.scroll.saturating_sub(3);
                                    }
                                    MouseEventKind::ScrollDown => {
                                        state.scroll = state.scroll.saturating_add(3);
                                    }
                                    _ => {}
                                }
                            }
                            Some(Ok(Event::Resize(_, _))) => {}
                            Some(Err(_)) | None => {
                                // Exit on error or stream end
                                disable_raw_mode()?;
                                execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
                                terminal.show_cursor()?;
                                return Ok(());
                            }
                            _ => {}
                        }
                    }

                    // Tick for animations
                    _ = tick_interval.tick() => {}
                }
            }

            // Drain any remaining agent events after prompt completes
            while let Ok(agent_event) = agent_rx.try_recv() {
                state.handle_agent_event(agent_event);
            }

            // Render final state before continuing
            terminal.draw(|frame| state.render(frame))?;

            continue; // Continue outer loop after prompt completes
        }

        // Render
        terminal.draw(|frame| state.render(frame))?;

        let area_width = terminal.size()?.width;

        tokio::select! {
            biased;

            // Handle agent events first (highest priority for responsiveness)
            event = agent_rx.recv() => {
                if let Ok(agent_event) = event {
                    state.handle_agent_event(agent_event);
                }
            }

            // Handle terminal events (keyboard input)
            event = event_stream.next() => {
                match event {
                    Some(Ok(Event::Key(key))) => {
                        let action = tau_tui::input::key_to_action(key);
                        if !state.handle_action(action, area_width).await {
                            break Ok(());
                        }
                    }
                    Some(Ok(Event::Paste(text))) => {
                        state.handle_action(Action::Paste(text), area_width).await;
                    }
                    Some(Ok(Event::Mouse(mouse))) => {
                        match mouse.kind {
                            MouseEventKind::ScrollUp => {
                                state.scroll = state.scroll.saturating_sub(3);
                            }
                            MouseEventKind::ScrollDown => {
                                state.scroll = state.scroll.saturating_add(3);
                            }
                            _ => {}
                        }
                    }
                    Some(Ok(Event::Resize(_, _))) => {}
                    Some(Err(e)) => {
                        break Err(anyhow::anyhow!("Event error: {}", e));
                    }
                    None => {
                        break Ok(());
                    }
                    _ => {}
                }
            }

            // Tick for animations (spinner updates)
            _ = tick_interval.tick() => {}

            // Handle UI messages (submit, quit, clear, abort, command)
            msg = ui_rx.recv() => {
                match msg {
                    Some(UiMessage::Submit(content)) => {
                        // Queue the prompt - will be processed at start of next loop iteration
                        pending_prompt = Some(content);
                    }
                    Some(UiMessage::Command(cmd)) => {
                        if let Some(result) = execute_command(&cmd, agent, model, *reasoning, available_models) {
                            match result {
                                CommandResult::Message(msg) => {
                                    state.show_system_message(&msg);
                                }
                                CommandResult::Clear => {
                                    agent.clear_messages();
                                    state.messages.clear();
                                    state.total_input_tokens = 0;
                                    state.total_output_tokens = 0;
                                    state.total_cost = 0.0;
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
                                    agent.set_reasoning(level);
                                }
                                CommandResult::Exit => {
                                    break Ok(());
                                }
                                CommandResult::Unknown(cmd) => {
                                    state.show_system_message(&format!("Unknown command: /{}\nType /help for available commands.", cmd));
                                }
                                CommandResult::OpenModelSelector => {
                                    // Open the model selector popup
                                    state.model_selector.show();
                                }
                                CommandResult::OpenBranchSelector => {
                                    // Open the branch selector popup
                                    state.open_branch_selector();
                                }
                                CommandResult::BranchFrom(branch_index) => {
                                    // Create branch directly (CLI mode or with index)
                                    match crate::session::SessionManager::branch_from(
                                        agent.messages(),
                                        branch_index,
                                        &model.id,
                                    ) {
                                        Ok(new_session) => {
                                            let msg_count = branch_index.map(|i| i + 1).unwrap_or(0);
                                            state.show_system_message(&format!(
                                                "Created branch session: {} ({} messages)",
                                                new_session.id(),
                                                msg_count
                                            ));
                                            // Truncate agent messages to branch point
                                            if let Some(idx) = branch_index {
                                                let messages: Vec<_> = agent.messages().iter().take(idx + 1).cloned().collect();
                                                agent.set_messages(messages);
                                                // Truncate UI messages too
                                                state.messages.truncate(idx + 1);
                                            } else {
                                                agent.clear_messages();
                                                state.messages.clear();
                                            }
                                            state.total_input_tokens = 0;
                                            state.total_output_tokens = 0;
                                            state.total_cost = 0.0;
                                        }
                                        Err(e) => {
                                            state.show_system_message(&format!("Failed to create branch: {}", e));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Some(UiMessage::ChangeModel(index)) => {
                        if let Some(new_model) = available_models.get(index) {
                            state.show_system_message(&format!("Switched to: {}", new_model.id));
                            *model = new_model.clone();
                            state.set_model(new_model.clone());
                            agent.set_model(new_model.clone());
                        }
                    }
                    Some(UiMessage::Clear) => {
                        agent.clear_messages();
                        state.messages.clear();
                        state.total_input_tokens = 0;
                        state.total_output_tokens = 0;
                        state.total_cost = 0.0;
                        state.status = "Cleared".to_string();
                    }
                    Some(UiMessage::Abort) => {
                        agent.abort();
                    }
                    Some(UiMessage::Branch(branch_index)) => {
                        // Create branch from selected message
                        match crate::session::SessionManager::branch_from(
                            agent.messages(),
                            branch_index,
                            &model.id,
                        ) {
                            Ok(new_session) => {
                                let msg_count = branch_index.map(|i| i + 1).unwrap_or(0);
                                state.show_system_message(&format!(
                                    "Created branch: {} ({} messages)\nContinue from this point with a fresh context.",
                                    new_session.id(),
                                    msg_count
                                ));
                                // Truncate agent messages to branch point
                                if let Some(idx) = branch_index {
                                    let messages: Vec<_> = agent.messages().iter().take(idx + 1).cloned().collect();
                                    agent.set_messages(messages);
                                    // Truncate UI messages too
                                    state.messages.truncate(idx + 1);
                                } else {
                                    agent.clear_messages();
                                    state.messages.clear();
                                }
                                state.total_input_tokens = 0;
                                state.total_output_tokens = 0;
                                state.total_cost = 0.0;
                            }
                            Err(e) => {
                                state.show_system_message(&format!("Failed to create branch: {}", e));
                            }
                        }
                    }
                    Some(UiMessage::Quit) | None => {
                        break Ok(());
                    }
                }
            }
        }
    };

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}
