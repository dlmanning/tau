use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
};
use super::widgets::{MessageList, OwnedSelector, OwnedSelectorItem, Selector, SelectorItem};

use super::constants;
use super::state::TuiState;
use super::types::rainbow_tau_style;
use crate::utils::format_tokens;

impl TuiState {
    /// Render the UI. Calls clamp_scroll for scroll state, then dispatches
    /// to &self sub-render methods.
    pub fn render(&mut self, frame: &mut Frame) {
        let size = frame.area();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Header
                Constraint::Min(1),    // Conversation
                Constraint::Length(1), // Status line
                Constraint::Length(3), // Input
            ])
            .split(size);

        // Compute conversation inner area for scroll clamping
        let conv_block = Block::default().borders(Borders::ALL);
        let conv_inner = conv_block.inner(chunks[1]);
        self.clamp_scroll(conv_inner);

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

    fn render_model_selector(&self, frame: &mut Frame, area: Rect) {
        let items: Vec<SelectorItem> = self
            .available_models
            .iter()
            .map(|m| SelectorItem {
                label: &m.name,
                description: Some(m.provider.name()),
                is_current: m.id == self.model.id,
            })
            .collect();

        let selector = Selector::new("Select Model", items, &self.theme)
            .with_selected(self.model_selector.selected);

        selector.render_centered(area, frame.buffer_mut());
    }

    fn render_branch_selector(&self, frame: &mut Frame, area: Rect) {
        let items: Vec<OwnedSelectorItem> = self
            .messages
            .iter()
            .enumerate()
            .map(|(i, msg)| {
                let preview =
                    crate::utils::truncate_chars(&msg.content, constants::BRANCH_PREVIEW_CHARS);
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

    fn render_conversation(&self, frame: &mut Frame, area: Rect) {
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

        // Scroll was already clamped by render() via clamp_scroll()
        let message_list = MessageList::new(&self.messages, &self.theme).scroll(self.scroll);
        frame.render_widget(message_list, inner);

        let content_height = super::widgets::message_list::calculate_message_height(
            &self.messages,
            inner.width as usize,
            &self.theme,
        );
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

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let cwd = std::env::current_dir()
            .ok()
            .map(|p| {
                if let Some(home) = dirs::home_dir() {
                    if let Ok(rest) = p.strip_prefix(&home) {
                        return format!("~/{}", rest.display());
                    }
                }
                p.display().to_string()
            })
            .unwrap_or_default();

        let info_content = match &self.git_branch.branch {
            Some(b) => format!("{{ {} · {} }}", cwd, b),
            None => format!("{{ {} }}", cwd),
        };

        let tau_style = if self.is_processing {
            rainbow_tau_style()
        } else {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        };

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

        parts.push(Span::styled(&self.model.name, dim));

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

        if self.usage.input_tokens > 0 || self.usage.output_tokens > 0 {
            parts.push(Span::styled(" · ", dim));
            parts.push(Span::styled(
                format!(
                    "{} in, {} out",
                    format_tokens(self.usage.input_tokens),
                    format_tokens(self.usage.output_tokens)
                ),
                dim,
            ));

            if self.usage.cache_read > 0 || self.usage.cache_write > 0 {
                parts.push(Span::styled(" · ", dim));
                parts.push(Span::styled(
                    format!(
                        "cache: {}r {}w",
                        format_tokens(self.usage.cache_read),
                        format_tokens(self.usage.cache_write)
                    ),
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
