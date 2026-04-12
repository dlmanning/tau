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

use std::time::Instant;

use crossterm::event::{Event, EventStream, MouseEventKind};
use futures::{FutureExt, StreamExt};
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
        SelectorState, message_list::ChatMessage,
    },
};
use tokio::sync::mpsc;

use crate::utils::format_tokens;

/// Pending interaction request waiting for user input in the TUI.
struct PendingInteraction {
    question: String,
    options: Vec<tau_agent::QuestionOption>,
    response_tx: tokio::sync::oneshot::Sender<tau_agent::InteractionResponse>,
    selector: SelectorState,
}

/// Per-agent progress tracking for richer subagent display.
struct AgentProgress {
    description: String,
    tool_count: u32,
    input_tokens: u64,
    output_tokens: u64,
    activity: String,
    finished: bool,
}

impl AgentProgress {
    fn new(description: String) -> Self {
        Self {
            description,
            tool_count: 0,
            input_tokens: 0,
            output_tokens: 0,
            activity: "starting...".to_string(),
            finished: false,
        }
    }
}

/// Build a tree diagram of active/completed subagents.
fn build_agent_tree(
    agent_order: &[String],
    agent_progress: &std::collections::HashMap<String, AgentProgress>,
) -> String {
    let agents: Vec<&AgentProgress> = agent_order
        .iter()
        .filter_map(|id| agent_progress.get(id))
        .collect();
    let total = agents.len();
    let mut lines = Vec::new();

    for (i, agent) in agents.iter().enumerate() {
        // Single agent: no tree chrome. Multiple: use box drawing.
        let (branch, cont) = if total == 1 {
            ("◇", " ")
        } else {
            let is_last = i == total - 1;
            if is_last { ("└─", "  ") } else { ("├─", "│ ") }
        };

        if agent.finished {
            let tokens = format_tokens(agent.input_tokens + agent.output_tokens);
            let indicator = if agent.activity.starts_with("error:") { "✗" } else { "✓" };
            lines.push(format!(
                "{} {} {} ({} tools · {} tokens)",
                branch, indicator, agent.description, agent.tool_count, tokens
            ));
        } else {
            lines.push(format!("{} ◇ {}", branch, agent.description));
            lines.push(format!("{}   ⚙ {}", cont, agent.activity));
        }
    }

    lines.join("\n")
}

/// Slow rainbow color shift for the τ glyph when agent is working.
/// Smoothly interpolates through the spectrum over ~4 seconds.
fn rainbow_tau_style() -> Style {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    // Full cycle every ~4 seconds (4096ms)
    let phase = (ms % 4096) as f64 / 4096.0;
    // HSV to RGB with S=1, V=1 — hue rotates through 0..360
    let hue = phase * 360.0;
    let c = 1.0_f64;
    let x = 1.0 - ((hue / 60.0) % 2.0 - 1.0).abs();
    let (r, g, b) = match hue as u32 {
        0..60 => (c, x, 0.0),
        60..120 => (x, c, 0.0),
        120..180 => (0.0, c, x),
        180..240 => (0.0, x, c),
        240..300 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Style::default()
        .fg(Color::Rgb(
            (r * 255.0) as u8,
            (g * 255.0) as u8,
            (b * 255.0) as u8,
        ))
        .add_modifier(Modifier::BOLD)
}

/// Get the current git branch name, or None if not in a git repo.
fn get_git_branch() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .stderr(std::process::Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
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
    /// Insertion order of agent IDs for tree rendering.
    agent_order: Vec<String>,
    /// Input box
    input: InputBox,
    /// Current scroll position
    scroll: usize,
    /// Whether to auto-follow new content at the bottom
    follow_bottom: bool,
    /// Whether agent is currently processing
    is_processing: bool,
    /// Cached git branch name (refreshed periodically via background task)
    git_branch: Option<String>,
    /// When the git branch was last checked
    git_branch_checked: Instant,
    /// Background task for refreshing git branch
    git_branch_task: Option<tokio::task::JoinHandle<Option<String>>>,
    /// Current status message
    status: String,
    /// Theme
    theme: Theme,
    /// Total tokens used
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_cache_read: u64,
    total_cache_write: u64,
    /// Model for cost calculation
    model: Model,
    /// Current reasoning level
    reasoning: tau_ai::ReasoningLevel,
    /// Whether adaptive thinking is enabled (fixed per session; reasoning level
    /// can change at runtime but adaptive mode cannot be toggled).
    thinking_adaptive: bool,
    /// Available models for selection
    available_models: Vec<Model>,
    /// Total cost
    total_cost: f64,
    /// Channel to send messages to agent handler
    ui_tx: mpsc::Sender<UiMessage>,
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
            agent_order: Vec::new(),
            input,
            scroll: 0,
            follow_bottom: true,
            is_processing: false,
            git_branch: get_git_branch(),
            git_branch_checked: Instant::now(),
            git_branch_task: None,
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
                tool_call_id,
                tool_name,
                activity,
                ..
            } => {
                self.messages.push(ChatMessage {
                    role: format!("tool:{}", tool_name),
                    content: activity,
                    is_error: false,
                    is_streaming: true,
                    id: Some(tool_call_id),
                });
                self.scroll_to_bottom();
            }
            AgentEvent::ToolExecutionUpdate {
                tool_call_id,
                content,
                ..
            } => {
                if let Some(msg) = self.messages.iter_mut().rev()
                    .find(|m| m.id.as_deref() == Some(&tool_call_id))
                {
                    msg.content = content;
                }
            }
            AgentEvent::ToolExecutionEnd {
                tool_call_id,
                tool_name,
                result,
                is_error,
                ..
            } => {
                let preview = crate::utils::truncate_chars(&result, 200);
                if let Some(msg) = self.messages.iter_mut().rev()
                    .find(|m| m.id.as_deref() == Some(&tool_call_id))
                {
                    msg.content = preview.to_string();
                    msg.is_streaming = false;
                    msg.is_error = is_error;
                } else {
                    self.messages.push(ChatMessage::tool(&tool_name, preview, is_error));
                }
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
                self.status = "Ready".to_string();
                // Clean up orphaned subagent progress if parent ends before children
                if !self.agent_progress.is_empty() {
                    if let Some(msg) = self.messages.iter_mut().rev()
                        .find(|m| m.role == "agents" && m.is_streaming)
                    {
                        msg.is_streaming = false;
                    }
                    self.agent_progress.clear();
                    self.agent_order.clear();
                }
            }
            AgentEvent::Error { message } => {
                self.is_processing = false;
                self.status = "Ready".to_string();
                if !self.agent_progress.is_empty() {
                    if let Some(msg) = self.messages.iter_mut().rev()
                        .find(|m| m.role == "agents")
                    {
                        msg.is_streaming = false;
                        msg.is_error = true;
                    }
                    self.agent_progress.clear();
                    self.agent_order.clear();
                }
                self.messages.push(ChatMessage {
                    role: "system".to_string(),
                    content: format!("Error: {}", message),
                    is_error: true,
                    is_streaming: false,
                    id: None,
                });
            }
            AgentEvent::CompactionStart { .. } => {}
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
            // Subagent events — render as tree in conversation
            AgentEvent::Subagent {
                agent_id,
                description,
                event,
            } => {
                match *event {
                    AgentEvent::AgentStart => {
                        let progress = AgentProgress::new(description);
                        self.agent_progress.insert(agent_id.clone(), progress);
                        self.agent_order.push(agent_id);
                        self.update_agent_tree();
                    }
                    AgentEvent::ToolExecutionStart {
                        ref activity,
                        ..
                    } => {
                        if let Some(progress) = self.agent_progress.get_mut(&agent_id) {
                            progress.tool_count += 1;
                            progress.activity = activity.clone();
                        }
                        self.update_agent_tree();
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
                        }
                        self.update_agent_tree();
                        // If all agents done, finalize the tree message
                        if self.agent_progress.values().all(|p| p.finished) {
                            if let Some(msg) = self.messages.iter_mut().rev()
                                .find(|m| m.role == "agents")
                            {
                                msg.is_streaming = false;
                            }
                            self.agent_progress.clear();
                            self.agent_order.clear();
                        }
                    }
                    AgentEvent::Error { ref message } => {
                        if let Some(progress) = self.agent_progress.get_mut(&agent_id) {
                            progress.finished = true;
                            progress.activity = format!("error: {}", message);
                        }
                        self.update_agent_tree();
                        if self.agent_progress.values().all(|p| p.finished) {
                            if let Some(msg) = self.messages.iter_mut().rev()
                                .find(|m| m.role == "agents")
                            {
                                msg.is_streaming = false;
                                msg.is_error = true;
                            }
                            self.agent_progress.clear();
                            self.agent_order.clear();
                        }
                    }
                    _ => {}
                }
            }
            // Ignore turn/message start events (we handle updates/ends)
            AgentEvent::TurnStart { .. } | AgentEvent::MessageStart { .. } => {}
        }
    }

    /// Update or insert the agent tree message in the conversation.
    fn update_agent_tree(&mut self) {
        let tree = build_agent_tree(&self.agent_order, &self.agent_progress);
        if let Some(msg) = self.messages.iter_mut().rev()
            .find(|m| m.role == "agents" && m.is_streaming)
        {
            msg.content = tree;
        } else {
            self.messages.push(ChatMessage {
                role: "agents".to_string(),
                content: tree,
                is_error: false,
                is_streaming: true,
                id: None,
            });
        }
        self.scroll_to_bottom();
    }

    /// Handle mouse scroll events.
    fn handle_mouse_scroll(&mut self, kind: MouseEventKind) {
        match kind {
            MouseEventKind::ScrollUp => {
                self.scroll = self.scroll.saturating_sub(3);
                self.follow_bottom = false;
            }
            MouseEventKind::ScrollDown => {
                self.scroll = self.scroll.saturating_add(3);
            }
            _ => {}
        }
    }

    /// Send a UI message, logging a warning if the channel is closed.
    async fn send_ui(&self, msg: UiMessage) {
        if self.ui_tx.send(msg).await.is_err() {
            tracing::warn!("UI message channel closed");
        }
    }

    fn scroll_to_bottom(&mut self) {
        self.follow_bottom = true;
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

    /// Reset token/cost counters and agent progress
    pub fn reset_stats(&mut self) {
        self.total_input_tokens = 0;
        self.total_output_tokens = 0;
        self.total_cache_read = 0;
        self.total_cache_write = 0;
        self.total_cost = 0.0;
        self.agent_progress.clear();
        self.agent_order.clear();
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
                    self.send_ui(UiMessage::Branch(Some(selected))).await;
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
                    self.send_ui(UiMessage::ChangeModel(selected)).await;
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
                        self.send_ui(UiMessage::Command(content)).await;
                    } else {
                        self.messages.push(ChatMessage::user(&content));
                        self.scroll_to_bottom();
                        self.send_ui(UiMessage::Submit(content)).await;
                    }
                }
                true
            }
            Action::Quit => {
                self.send_ui(UiMessage::Quit).await;
                false
            }
            Action::Interrupt | Action::Escape => {
                if self.is_processing {
                    self.send_ui(UiMessage::Abort).await;
                    self.status = "Cancelling...".to_string();
                    true
                } else {
                    self.send_ui(UiMessage::Quit).await;
                    false
                }
            }
            Action::PageUp => {
                self.scroll = self.scroll.saturating_sub(10);
                self.follow_bottom = false;
                true
            }
            Action::PageDown => {
                self.scroll = self.scroll.saturating_add(10);
                // Re-pin resolved in render if we've reached the bottom
                true
            }
            Action::Clear => {
                self.send_ui(UiMessage::Clear).await;
                self.messages.clear();
                self.reset_stats();
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

    /// Handle a terminal event while a prompt is executing.
    /// Returns `false` if the TUI should exit immediately.
    pub fn handle_event_while_processing(
        &mut self,
        event: Event,
        area_width: u16,
        agent_handle: &tau_agent::AgentHandle,
    ) -> bool {
        match event {
            Event::Key(key) if self.pending_interaction.is_some() => {
                let action = tau_tui::input::key_to_action(key);
                match action {
                    Action::Up => {
                        if let Some(pi) = self.pending_interaction.as_mut() {
                            pi.selector.up(pi.options.len());
                        }
                    }
                    Action::Down => {
                        if let Some(pi) = self.pending_interaction.as_mut() {
                            pi.selector.down(pi.options.len());
                        }
                    }
                    Action::Submit => {
                        if let Some(pi) = self.pending_interaction.take() {
                            let label = pi.options[pi.selector.selected].label.clone();
                            // Oneshot: Err only if receiver dropped, which is fine
                            let _ = pi.response_tx.send(
                                tau_agent::InteractionResponse::Answer(label),
                            );
                            self.status = "Thinking...".to_string();
                        }
                    }
                    Action::Escape | Action::Interrupt => {
                        if let Some(pi) = self.pending_interaction.take() {
                            let _ = pi.response_tx.send(
                                tau_agent::InteractionResponse::Cancelled,
                            );
                            self.status = "Thinking...".to_string();
                        }
                    }
                    _ => {} // consume all other input while modal is open
                }
                true
            }
            Event::Key(key) => {
                let action = tau_tui::input::key_to_action(key);
                match action {
                    Action::Interrupt | Action::Escape => {
                        agent_handle.abort();
                        self.status = "Cancelling...".to_string();
                    }
                    Action::Quit => return false,
                    Action::Submit => {
                        let content = self.input.content().to_string();
                        if !content.is_empty() {
                            self.input.clear();
                            self.messages.push(ChatMessage {
                                role: "steer".to_string(),
                                content: content.clone(),
                                is_error: false,
                                is_streaming: false,
                                id: None,
                            });
                            self.scroll_to_bottom();
                            agent_handle.steer(tau_ai::Message::user(&content));
                        }
                    }
                    _ => {
                        self.input.handle_action(&action, area_width);
                    }
                }
                true
            }
            Event::Paste(text) => {
                self.input.handle_action(&Action::Paste(text), area_width);
                true
            }
            Event::Mouse(mouse) => {
                self.handle_mouse_scroll(mouse.kind);
                true
            }
            Event::Resize(_, _) => true,
            _ => true,
        }
    }

    /// Handle a terminal event while idle (no prompt executing).
    /// Returns `false` if the TUI should exit.
    pub async fn handle_event_while_idle(
        &mut self,
        event: Event,
        area_width: u16,
    ) -> bool {
        match event {
            Event::Key(key) => {
                let action = tau_tui::input::key_to_action(key);
                self.handle_action(action, area_width).await
            }
            Event::Paste(text) => {
                self.handle_action(Action::Paste(text), area_width).await
            }
            Event::Mouse(mouse) => {
                self.handle_mouse_scroll(mouse.kind);
                true
            }
            Event::Resize(_, _) => true,
            _ => true,
        }
    }

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

        // Collect result from a previously spawned git branch refresh.
        // unwrap()s are safe: is_some_and guards the take(), is_finished() guarantees now_or_never() returns Some.
        if self.git_branch_task.as_ref().is_some_and(|t| t.is_finished()) {
            if let Ok(branch) = self.git_branch_task.take().unwrap().now_or_never().unwrap() {
                self.git_branch = branch;
            }
        }

        // Spawn a background refresh every 5 seconds
        if self.git_branch_checked.elapsed() > std::time::Duration::from_secs(5)
            && self.git_branch_task.is_none()
        {
            self.git_branch_checked = Instant::now();
            self.git_branch_task = Some(tokio::task::spawn_blocking(get_git_branch));
        }

        let info_content = match &self.git_branch {
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
        if self.total_input_tokens > 0 || self.total_output_tokens > 0 {
            parts.push(Span::styled(" · ", dim));
            parts.push(Span::styled(
                format!("{} in, {} out", format_tokens(self.total_input_tokens), format_tokens(self.total_output_tokens)),
                dim,
            ));

            if self.total_cache_read > 0 || self.total_cache_write > 0 {
                parts.push(Span::styled(" · ", dim));
                parts.push(Span::styled(
                    format!("cache: {}r {}w", format_tokens(self.total_cache_read), format_tokens(self.total_cache_write)),
                    dim,
                ));
            }

            if self.total_cost > 0.0 {
                parts.push(Span::styled(" · ", dim));
                parts.push(Span::styled(format!("${:.4}", self.total_cost), dim));
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
