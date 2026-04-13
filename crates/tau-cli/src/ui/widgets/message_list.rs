//! Message list widget for displaying chat messages

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget},
};
use textwrap;

use super::{super::theme::Theme, markdown::render_markdown};

/// Count the visual width of a Line (sum of span character widths).
fn line_width(line: &Line) -> usize {
    line.spans.iter().map(|s| s.content.chars().count()).sum()
}

/// How many visual rows a Line occupies at the given terminal width.
fn visual_line_count(line: &Line, width: usize) -> usize {
    if width == 0 {
        return 1;
    }
    let w = line_width(line);
    if w == 0 { 1 } else { w.div_ceil(width) }
}

/// Split a Line that exceeds `width` into multiple Lines.
/// Breaks at span boundaries and within spans at character positions.
fn wrap_line_into<'a>(line: Line<'a>, width: usize, out: &mut Vec<Line<'a>>) {
    if width == 0 || line_width(&line) <= width {
        out.push(line);
        return;
    }

    let mut current_spans: Vec<Span<'a>> = Vec::new();
    let mut current_width: usize = 0;

    for span in line.spans {
        let span_text: &str = &span.content;
        let style = span.style;

        if span_text.is_empty() {
            current_spans.push(span);
            continue;
        }

        let mut remaining = span_text.to_string();
        while !remaining.is_empty() {
            let available = width.saturating_sub(current_width);
            if available == 0 {
                out.push(Line::from(std::mem::take(&mut current_spans)));
                current_width = 0;
                continue;
            }

            let char_count = remaining.chars().count();
            if char_count <= available {
                current_width += char_count;
                current_spans.push(Span::styled(remaining, style));
                break;
            }

            // Split at `available` characters
            let split_at: usize = remaining
                .char_indices()
                .nth(available)
                .map(|(i, _)| i)
                .unwrap_or(remaining.len());
            let (head, tail) = remaining.split_at(split_at);
            current_spans.push(Span::styled(head.to_string(), style));
            out.push(Line::from(std::mem::take(&mut current_spans)));
            current_width = 0;
            remaining = tail.to_string();
        }
    }

    if !current_spans.is_empty() {
        out.push(Line::from(current_spans));
    }
}

/// Get a tick counter for animations (~80ms per frame).
fn animation_tick() -> usize {
    (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        / 80) as usize
}

/// A single message in the chat
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Role: "user", "assistant", "tool", "system", "agent"
    pub role: String,
    /// Message content
    pub content: String,
    /// Whether this is an error message
    pub is_error: bool,
    /// Whether this is currently streaming
    pub is_streaming: bool,
    /// Optional ID for updating specific messages (e.g. active subagents)
    pub id: Option<String>,
}

impl ChatMessage {
    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            is_error: false,
            is_streaming: false,
            id: None,
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            is_error: false,
            is_streaming: false,
            id: None,
        }
    }

    /// Create a streaming assistant message
    pub fn assistant_streaming(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            is_error: false,
            is_streaming: true,
            id: None,
        }
    }

    /// Create a tool message
    pub fn tool(name: &str, content: impl Into<String>, is_error: bool) -> Self {
        Self {
            role: format!("tool:{}", name),
            content: content.into(),
            is_error,
            is_streaming: false,
            id: None,
        }
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            is_error: false,
            is_streaming: false,
            id: None,
        }
    }
}

/// Widget for displaying a list of chat messages
pub struct MessageList<'a> {
    messages: &'a [ChatMessage],
    theme: &'a Theme,
    scroll: usize,
}

impl<'a> MessageList<'a> {
    /// Create a new message list
    pub fn new(messages: &'a [ChatMessage], theme: &'a Theme) -> Self {
        Self {
            messages,
            theme,
            scroll: 0,
        }
    }

    /// Set scroll offset
    pub fn scroll(mut self, scroll: usize) -> Self {
        self.scroll = scroll;
        self
    }

    fn render_message(&self, msg: &ChatMessage, width: usize) -> Vec<Line<'static>> {
        let mut lines = Vec::new();

        // Prefix arrow and content width (arrow + space = 2 chars)
        let content_width = width.saturating_sub(2);

        match msg.role.as_str() {
            "user" => {
                let style = self.theme.accent_bold();
                let wrapped = textwrap::wrap(&msg.content, content_width);
                for (i, line) in wrapped.iter().enumerate() {
                    let prefix = if i == 0 { "▶ " } else { "  " };
                    lines.push(Line::from(Span::styled(
                        format!("{}{}", prefix, line),
                        style,
                    )));
                }
            }
            "steer" => {
                let style = self.theme.dim_style().add_modifier(Modifier::ITALIC);
                let wrapped = textwrap::wrap(&msg.content, content_width);
                for (i, line) in wrapped.iter().enumerate() {
                    let prefix = if i == 0 { "▷ " } else { "  " };
                    lines.push(Line::from(Span::styled(
                        format!("{}{}", prefix, line),
                        style,
                    )));
                }
            }
            "assistant" => {
                if msg.content.is_empty() && msg.is_streaming {
                    // No content yet — τ animation in status bar is enough
                } else {
                    let arrow_style = self.theme.success_style().add_modifier(Modifier::BOLD);
                    let md_lines = render_markdown(&msg.content, self.theme, content_width);
                    for (i, line) in md_lines.into_iter().enumerate() {
                        let mut spans = Vec::new();
                        if i == 0 {
                            spans.push(Span::styled("◀ ", arrow_style));
                        } else {
                            spans.push(Span::raw("  "));
                        }
                        spans.extend(
                            line.spans
                                .into_iter()
                                .map(|s| Span::styled(s.content.into_owned(), s.style)),
                        );
                        lines.push(Line::from(spans));
                    }
                }
            }
            r if r.starts_with("agent:") => {
                if msg.is_streaming {
                    let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
                    let frame_idx = animation_tick() % frames.len();
                    let display = if msg.content.is_empty() {
                        format!("◇ {} working...", frames[frame_idx])
                    } else {
                        format!("◇ {} {}", frames[frame_idx], msg.content)
                    };
                    lines.push(Line::from(Span::styled(
                        display,
                        Style::default().fg(Color::Cyan),
                    )));
                } else {
                    let (indicator, style) = if msg.is_error {
                        ("✗", self.theme.error_style())
                    } else {
                        ("✓", self.theme.success_style())
                    };
                    lines.push(Line::from(Span::styled(
                        format!("◇ {} {}", indicator, msg.content),
                        style,
                    )));
                }
            }
            "agents" => {
                // Pre-formatted tree diagram of parallel agents
                let branch_style = Style::default().fg(Color::Cyan);
                let activity_style = Style::default().fg(Color::DarkGray);
                for text_line in msg.content.lines() {
                    // Activity lines (⚙) get dim style, structure lines get cyan
                    let style = if text_line.contains('⚙') {
                        activity_style
                    } else {
                        branch_style
                    };
                    lines.push(Line::from(Span::styled(text_line.to_string(), style)));
                }
            }
            "system" => {
                let wrapped = textwrap::wrap(&msg.content, content_width);
                for (i, line) in wrapped.iter().enumerate() {
                    let prefix = if i == 0 { "● " } else { "  " };
                    lines.push(Line::from(Span::styled(
                        format!("{}{}", prefix, line),
                        self.theme.dim_style(),
                    )));
                }
            }
            r if r.starts_with("tool:") => {
                let style = if msg.is_error {
                    self.theme.error_style()
                } else {
                    Style::default().fg(Color::DarkGray)
                };
                let wrapped = textwrap::wrap(&msg.content, content_width);
                for (i, line) in wrapped.iter().enumerate() {
                    let prefix = if i == 0 { "⚙ " } else { "  " };
                    lines.push(Line::from(Span::styled(
                        format!("{}{}", prefix, line),
                        style,
                    )));
                }
            }
            _ => {
                let style = if msg.is_error {
                    self.theme.error_style()
                } else {
                    self.theme.base_style()
                };
                let wrapped = textwrap::wrap(&msg.content, content_width);
                for line in wrapped.iter() {
                    lines.push(Line::from(Span::styled(format!("  {}", line), style)));
                }
            }
        }

        lines.push(Line::from(""));

        lines
    }
}

impl Widget for MessageList<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default().borders(Borders::NONE);

        let inner = block.inner(area);
        block.render(area, buf);

        if inner.width == 0 || inner.height == 0 {
            return;
        }

        let width = inner.width as usize;
        let mut all_lines: Vec<Line> = Vec::new();

        for msg in self.messages {
            for line in self.render_message(msg, width) {
                wrap_line_into(line, width, &mut all_lines);
            }
        }

        let visible_lines: Vec<Line> = all_lines
            .into_iter()
            .skip(self.scroll)
            .take(inner.height as usize)
            .collect();

        let paragraph = Paragraph::new(visible_lines);
        paragraph.render(inner, buf);
    }
}

/// Calculate total height of messages.
/// Uses the same render path as `MessageList::render_message` to stay in sync.
pub fn calculate_message_height(messages: &[ChatMessage], width: usize, theme: &Theme) -> usize {
    let list = MessageList::new(messages, theme);
    let mut total = 0;
    for msg in messages {
        for line in list.render_message(msg, width) {
            total += visual_line_count(&line, width);
        }
    }
    total
}
