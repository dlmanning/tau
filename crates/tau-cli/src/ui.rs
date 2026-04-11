//! TUI implementation for tau

use std::time::Instant;

use crossterm::event::{Event, EventStream, MouseEventKind};
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
use tau_tui::{
    Theme,
    input::Action,
    widgets::{
        InputBox, MessageList, OwnedSelector, OwnedSelectorItem, Selector, SelectorItem,
        SelectorState, Spinner, message_list::ChatMessage,
    },
};
use tokio::sync::mpsc;

/// Pending interaction request waiting for user input in the TUI.
struct PendingInteraction {
    question: String,
    options: Vec<tau_agent::QuestionOption>,
    response_tx: tokio::sync::oneshot::Sender<tau_agent::InteractionResponse>,
    selector: SelectorState,
}

/// Per-agent progress tracking for richer subagent display.
struct AgentProgress {
    tool_count: u32,
    input_tokens: u32,
    output_tokens: u32,
    activity: String,
    finished: bool,
}

impl AgentProgress {
    fn new() -> Self {
        Self {
            tool_count: 0,
            input_tokens: 0,
            output_tokens: 0,
            activity: "starting...".to_string(),
            finished: false,
        }
    }

}

/// Build a human-readable description of what a tool is doing.
fn describe_tool_activity(tool_name: &str, arguments: &serde_json::Value) -> String {
    let short_path = || {
        arguments
            .get("path")
            .or_else(|| arguments.get("file_path"))
            .and_then(|v| v.as_str())
            .and_then(|p| p.rsplit('/').next())
            .unwrap_or("file")
    };

    match tool_name {
        "read" => format!("Reading {}", short_path()),
        "write" => format!("Writing {}", short_path()),
        "edit" => format!("Editing {}", short_path()),
        "bash" => {
            let cmd = arguments
                .get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("command");
            let short: String = cmd.chars().take(30).collect();
            format!("Running {}", short)
        }
        "grep" => {
            let pattern = arguments
                .get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or("...");
            format!("Searching for \"{}\"", pattern)
        }
        "glob" => {
            let pattern = arguments
                .get("pattern")
                .and_then(|v| v.as_str())
                .unwrap_or("...");
            format!("Finding {}", pattern)
        }
        "list" => "Listing directory".to_string(),
        "lsp" => "Querying language server".to_string(),
        "agent" => "Spawning agent".to_string(),
        other => format!("Running {}", other),
    }
}

/// Format a token count compactly (e.g. 1234 → "1.2k", 56 → "56")
fn format_tokens(n: u32) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

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
    /// Per-agent progress state keyed by agent_id.
    agent_progress: std::collections::HashMap<String, AgentProgress>,
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
    total_cache_read: u32,
    total_cache_write: u32,
    /// Model for cost calculation
    model: Model,
    /// Current reasoning level
    reasoning: tau_ai::ReasoningLevel,
    /// Whether adaptive thinking is enabled
    thinking_adaptive: bool,
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
    /// Pending interaction request (question waiting for user to pick an option)
    pending_interaction: Option<PendingInteraction>,
}

impl TuiState {
    pub fn new(
        model: Model,
        reasoning: tau_ai::ReasoningLevel,
        thinking_adaptive: bool,
        available_models: Vec<Model>,
        ui_tx: mpsc::Sender<UiMessage>,
    ) -> Self {
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
            agent_progress: std::collections::HashMap::new(),
            input,
            scroll: 0,
            is_processing: false,
            status: "Ready".to_string(),
            theme: Theme::dark(),
            total_input_tokens: 0,
            total_output_tokens: 0,
            total_cache_read: 0,
            total_cache_write: 0,
            model,
            reasoning,
            thinking_adaptive,
            available_models,
            total_cost: 0.0,
            ui_tx,
            spinner_start: Instant::now(),
            model_selector,
            branch_selector: SelectorState::default(),
            pending_interaction: None,
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
                self.is_processing = true;
            }
            AgentEvent::MessageUpdate { message } => {
                let text = message.text();
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
            AgentEvent::ToolExecutionStart {
                tool_name,
                ..
            } => {
                self.status = format!("Running {}...", tool_name);
            }
            AgentEvent::ToolExecutionUpdate {
                tool_name,
                content,
                ..
            } => {
                self.status = format!("{}: {}", tool_name, content);
            }
            AgentEvent::ToolExecutionEnd {
                tool_name,
                result,
                is_error,
                ..
            } => {
                let preview = crate::utils::truncate_chars(&result, 200);
                self.messages
                    .push(ChatMessage::tool(&tool_name, preview, is_error));
                self.scroll_to_bottom();
            }
            AgentEvent::TurnEnd { usage, .. } => {
                self.total_input_tokens += usage.input;
                self.total_output_tokens += usage.output;
                self.total_cache_read += usage.cache_read;
                self.total_cache_write += usage.cache_write;
                let cost = usage.calculate_cost(&self.model);
                self.total_cost += cost.total;
            }
            AgentEvent::AgentEnd { .. } => {
                self.is_processing = false;
                let cache_str = if self.total_cache_read > 0 || self.total_cache_write > 0 {
                    format!(
                        " | cache: {}r {}w",
                        format_tokens(self.total_cache_read),
                        format_tokens(self.total_cache_write),
                    )
                } else {
                    String::new()
                };
                self.status = format!(
                    "Ready | {} in, {} out{} | ${:.4}",
                    self.total_input_tokens, self.total_output_tokens, cache_str, self.total_cost
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
                    id: None,
                });
            }
            AgentEvent::CompactionStart { reason } => {
                self.status = format!(
                    "Compacting context ({})...",
                    crate::utils::compaction_reason_str(reason)
                );
            }
            AgentEvent::CompactionEnd {
                tokens_before,
                tokens_after,
            } => {
                self.messages.push(ChatMessage::system(format!(
                    "Context compacted: ~{} -> ~{} tokens",
                    tokens_before, tokens_after
                )));
                self.scroll_to_bottom();
            }
            // Subagent events — track in status bar, show summary on completion
            AgentEvent::Subagent {
                agent_id,
                description,
                event,
            } => {
                match *event {
                    AgentEvent::AgentStart => {
                        let progress = AgentProgress::new();
                        self.agent_progress.insert(agent_id, progress);
                        self.status = format!("Agent: {}...", description);
                    }
                    AgentEvent::ToolExecutionStart {
                        ref tool_name,
                        ref arguments,
                        ..
                    } => {
                        if let Some(progress) = self.agent_progress.get_mut(&agent_id) {
                            progress.tool_count += 1;
                            progress.activity =
                                describe_tool_activity(tool_name, arguments);
                        }
                        self.status = format!("Agent: {} [{}]", description, tool_name);
                    }
                    AgentEvent::TurnEnd { ref usage, .. } => {
                        if let Some(progress) = self.agent_progress.get_mut(&agent_id) {
                            progress.input_tokens += usage.input;
                            progress.output_tokens += usage.output;
                        }
                        self.total_input_tokens += usage.input;
                        self.total_output_tokens += usage.output;
                        self.total_cache_read += usage.cache_read;
                        self.total_cache_write += usage.cache_write;
                        let cost = usage.calculate_cost(&self.model);
                        self.total_cost += cost.total;
                    }
                    AgentEvent::AgentEnd { .. } => {
                        if let Some(progress) = self.agent_progress.get_mut(&agent_id) {
                            progress.finished = true;
                            let tokens =
                                format_tokens(progress.input_tokens + progress.output_tokens);
                            let msg = ChatMessage {
                                role: format!("agent:{}", description),
                                content: format!(
                                    "{} tools · {} tokens",
                                    progress.tool_count, tokens
                                ),
                                is_error: false,
                                is_streaming: false,
                                id: Some(agent_id),
                            };
                            self.messages.push(msg);
                            self.scroll_to_bottom();
                        }
                        self.status = "Ready".to_string();
                    }
                    AgentEvent::Error { ref message } => {
                        if let Some(progress) = self.agent_progress.get_mut(&agent_id) {
                            progress.finished = true;
                        }
                        let msg = ChatMessage {
                            role: format!("agent:{}", description),
                            content: message.clone(),
                            is_error: true,
                            is_streaming: false,
                            id: Some(agent_id),
                        };
                        self.messages.push(msg);
                        self.scroll_to_bottom();
                        self.status = "Ready".to_string();
                    }
                    _ => {}
                }
            }
            // Ignore turn/message start events (we handle updates/ends)
            AgentEvent::TurnStart { .. } | AgentEvent::MessageStart { .. } => {}
        }
    }

    /// Handle mouse scroll events.
    fn handle_mouse_scroll(&mut self, kind: MouseEventKind) {
        match kind {
            MouseEventKind::ScrollUp => {
                self.scroll = self.scroll.saturating_sub(3);
            }
            MouseEventKind::ScrollDown => {
                self.scroll = self.scroll.saturating_add(3);
            }
            _ => {}
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
                    let selected = self.branch_selector.selected;
                    self.branch_selector.hide();
                    let _ = self.ui_tx.send(UiMessage::Branch(Some(selected))).await;
                    return true;
                }
                Action::Escape => {
                    self.branch_selector.hide();
                    return true;
                }
                _ => {
                    return true;
                }
            }
        }

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
                    let selected = self.model_selector.selected;
                    self.model_selector.hide();
                    let _ = self.ui_tx.send(UiMessage::ChangeModel(selected)).await;
                    return true;
                }
                Action::Escape | Action::ModelSelect => {
                    self.model_selector.hide();
                    return true;
                }
                _ => {
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
                        let _ = self.ui_tx.send(UiMessage::Command(content)).await;
                    } else {
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
                self.agent_progress.clear();
                self.total_input_tokens = 0;
                self.total_output_tokens = 0;
                self.total_cache_read = 0;
                self.total_cache_write = 0;
                self.total_cost = 0.0;
                self.status = "Ready".to_string();
                true
            }
            Action::ModelSelect => {
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

        // Layout: status bar (1), messages (flex), input (3)
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Status
                Constraint::Min(1),    // Messages
                Constraint::Length(3), // Input
            ])
            .split(size);

        self.render_status(frame, chunks[0]);
        self.render_messages(frame, chunks[1]);
        self.input
            .render(chunks[2], frame.buffer_mut(), &self.theme);

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
        let pi = self.pending_interaction.as_ref().unwrap();
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
                let preview = crate::utils::truncate_chars(&msg.content, 50);
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
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(self.theme.border_style());

        let inner = block.inner(area);
        frame.render_widget(block, area);

        if inner.height == 0 || self.messages.is_empty() {
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
            &self.theme,
        );

        if self.scroll == usize::MAX {
            self.scroll = content_height.saturating_sub(inner.height as usize);
        } else {
            self.scroll = self
                .scroll
                .min(content_height.saturating_sub(inner.height as usize));
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

    fn render_status(&self, frame: &mut Frame, area: Rect) {
        if self.is_processing {
            let spinner =
                Spinner::new(&self.status, &self.theme).with_start_time(self.spinner_start);
            frame.render_widget(spinner, area);
        } else {
            let model_name = self
                .model
                .id
                .split('/')
                .next_back()
                .unwrap_or(&self.model.id);
            let reasoning_str = match self.reasoning {
                tau_ai::ReasoningLevel::Off => "",
                tau_ai::ReasoningLevel::Minimal => " │ thinking: minimal",
                tau_ai::ReasoningLevel::Low => " │ thinking: low",
                tau_ai::ReasoningLevel::Medium => " │ thinking: medium",
                tau_ai::ReasoningLevel::High => " │ thinking: high",
            };
            let adaptive_str = if self.thinking_adaptive && self.reasoning != tau_ai::ReasoningLevel::Off {
                " (adaptive)"
            } else {
                ""
            };
            let left_content = format!(
                "{}{}{} │ {}",
                model_name, reasoning_str, adaptive_str, self.status
            );
            let right_content = "Ctrl+K: model │ Ctrl+L: clear │ Ctrl+C: quit";

            let left_width = left_content.chars().count();
            let right_width = right_content.chars().count();
            let available = area.width as usize;

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
    mut interaction_rx: tokio::sync::mpsc::Receiver<tau_agent::InteractionRequest>,
) -> anyhow::Result<()> {
    use std::io;

    use crossterm::{
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    };
    use ratatui::{Terminal, backend::CrosstermBackend};

    use crate::commands::{CommandResult, execute_command};

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let (ui_tx, mut ui_rx) = mpsc::channel::<UiMessage>(32);

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
    let mut tick_interval = tokio::time::interval(std::time::Duration::from_millis(80));

    // Pending prompt content - we'll process this at the start of the next loop iteration
    // This is stored as a String so it lives long enough for the future
    let mut pending_prompt: Option<String> = None;

    let result = loop {
        // If there's a pending prompt, start processing it
        // We create the future here where `content` is still in scope
        if let Some(content) = pending_prompt.take() {
            state.is_processing = true;
            state.spinner_start = Instant::now();
            state.status = "Thinking...".to_string();
            state.messages.push(ChatMessage::assistant_streaming(""));
            state.scroll_to_bottom();

            // Get handles before creating the future (so we can use them without borrowing agent)
            let cancel_handle = agent.cancel_handle();
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
                        // Drop any pending interaction (tool future is gone)
                        state.pending_interaction = None;
                        break; // Exit inner loop, prompt is done
                    }

                    event = agent_rx.recv() => {
                        if let Ok(agent_event) = event {
                            state.handle_agent_event(agent_event);
                        }
                    }

                    event = event_stream.next() => {
                        match event {
                            Some(Ok(Event::Key(key))) if state.pending_interaction.is_some() => {
                                let action = tau_tui::input::key_to_action(key);
                                match action {
                                    Action::Up => {
                                        let pi = state.pending_interaction.as_mut().unwrap();
                                        pi.selector.up(pi.options.len());
                                    }
                                    Action::Down => {
                                        let pi = state.pending_interaction.as_mut().unwrap();
                                        pi.selector.down(pi.options.len());
                                    }
                                    Action::Submit => {
                                        let pi = state.pending_interaction.take().unwrap();
                                        let label = pi.options[pi.selector.selected].label.clone();
                                        let _ = pi.response_tx.send(
                                            tau_agent::InteractionResponse::Answer(label),
                                        );
                                        state.status = "Thinking...".to_string();
                                    }
                                    Action::Escape | Action::Interrupt => {
                                        let pi = state.pending_interaction.take().unwrap();
                                        let _ = pi.response_tx.send(
                                            tau_agent::InteractionResponse::Cancelled,
                                        );
                                        state.status = "Thinking...".to_string();
                                    }
                                    _ => {} // consume all other input while modal is open
                                }
                            }
                            Some(Ok(Event::Key(key))) => {
                                let action = tau_tui::input::key_to_action(key);
                                match action {
                                    Action::Interrupt | Action::Escape => {
                                        cancel_handle.lock().cancel();
                                        state.status = "Cancelling...".to_string();
                                    }
                                    Action::Quit => {
                                        disable_raw_mode()?;
                                        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
                                        terminal.show_cursor()?;
                                        return Ok(());
                                    }
                                    Action::Submit => {
                                        let content = state.input.content().to_string();
                                        if !content.is_empty() {
                                            state.input.clear();
                                            // Use "steer" role to visually distinguish from
                                            // normal prompts (▷ dim italic vs ▶ bold accent)
                                            state.messages.push(ChatMessage {
                                                role: "steer".to_string(),
                                                content: content.clone(),
                                                is_error: false,
                                                is_streaming: false,
                                                id: None,
                                            });
                                            state.scroll_to_bottom();
                                            agent_handle.steer(tau_ai::Message::user(&content));
                                        }
                                    }
                                    _ => {
                                        state.input.handle_action(&action, area_width);
                                    }
                                }
                            }
                            Some(Ok(Event::Paste(text))) => {
                                state.input.handle_action(&Action::Paste(text), area_width);
                            }
                            Some(Ok(Event::Mouse(mouse))) => {
                                state.handle_mouse_scroll(mouse.kind);
                            }
                            Some(Ok(Event::Resize(_, _))) => {}
                            Some(Err(_)) | None => {
                                disable_raw_mode()?;
                                execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
                                terminal.show_cursor()?;
                                return Ok(());
                            }
                            _ => {}
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
                        state.handle_mouse_scroll(mouse.kind);
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

            _ = tick_interval.tick() => {}

            msg = ui_rx.recv() => {
                match msg {
                    Some(UiMessage::Submit(content)) => {
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
                                    state.total_cache_read = 0;
                                    state.total_cache_write = 0;
                                    state.total_cost = 0.0;
                                    state.agent_progress.clear();
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
                                CommandResult::Exit => {
                                    break Ok(());
                                }
                                CommandResult::Unknown(cmd) => {
                                    state.show_system_message(&format!("Unknown command: /{}\nType /help for available commands.", cmd));
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
                                            if let Some(idx) = branch_index {
                                                let messages: Vec<_> = agent.messages().iter().take(idx + 1).cloned().collect();
                                                agent.set_messages(messages);
                                                state.messages.truncate(idx + 1);
                                            } else {
                                                agent.clear_messages();
                                                state.messages.clear();
                                            }
                                            state.total_input_tokens = 0;
                                            state.total_output_tokens = 0;
                                            state.total_cache_read = 0;
                                            state.total_cache_write = 0;
                                            state.total_cost = 0.0;
                                            state.agent_progress.clear();
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
                        state.total_cache_read = 0;
                        state.total_cache_write = 0;
                        state.total_cost = 0.0;
                        state.agent_progress.clear();
                        state.status = "Cleared".to_string();
                    }
                    Some(UiMessage::Abort) => {
                        agent.abort();
                    }
                    Some(UiMessage::Branch(branch_index)) => {
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
                                if let Some(idx) = branch_index {
                                    let messages: Vec<_> = agent.messages().iter().take(idx + 1).cloned().collect();
                                    agent.set_messages(messages);
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

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}
