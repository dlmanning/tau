//! Message list widget for displaying chat messages

use crate::theme::Theme;
use crate::widgets::markdown::render_markdown;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget, Wrap},
};
use textwrap;

/// A single message in the chat
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Role: "user", "assistant", "tool", "system"
    pub role: String,
    /// Message content
    pub content: String,
    /// Whether this is an error message
    pub is_error: bool,
    /// Whether this is currently streaming
    pub is_streaming: bool,
}

impl ChatMessage {
    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            is_error: false,
            is_streaming: false,
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            is_error: false,
            is_streaming: false,
        }
    }

    /// Create a streaming assistant message
    pub fn assistant_streaming(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            is_error: false,
            is_streaming: true,
        }
    }

    /// Create a tool message
    pub fn tool(name: &str, content: impl Into<String>, is_error: bool) -> Self {
        Self {
            role: format!("tool:{}", name),
            content: content.into(),
            is_error,
            is_streaming: false,
        }
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            is_error: false,
            is_streaming: false,
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

        // Role header with visual distinction
        let (role_text, role_style, prefix) = match msg.role.as_str() {
            "user" => ("You", self.theme.accent_bold(), "▶ "),
            "assistant" => (
                "Assistant",
                self.theme.success_style().add_modifier(Modifier::BOLD),
                "◀ ",
            ),
            "system" => ("System", self.theme.dim_style(), "● "),
            r if r.starts_with("tool:") => {
                let tool_name = &r[5..];
                let style = if msg.is_error {
                    self.theme.error_style()
                } else {
                    Style::default().fg(Color::Magenta)
                };
                (tool_name, style, "⚙ ")
            }
            _ => ("Unknown", self.theme.dim_style(), "  "),
        };

        let header = if msg.is_streaming {
            format!("{}{} ▌", prefix, role_text)
        } else {
            format!("{}{}", prefix, role_text)
        };

        lines.push(Line::from(Span::styled(header, role_style)));

        // Content - use markdown for assistant messages, plain text for others
        let content_width = width.saturating_sub(2);

        if msg.role == "assistant" && !msg.is_error {
            if msg.content.is_empty() && msg.is_streaming {
                // Show animated thinking indicator for empty streaming message
                // Use time-based frame selection for animation
                let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
                let frame_idx = (std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
                    / 80) as usize
                    % frames.len();
                lines.push(Line::from(Span::styled(
                    format!("  {} thinking...", frames[frame_idx]),
                    Style::default().fg(Color::Yellow),
                )));
            } else {
                // Render markdown for assistant messages
                let md_lines = render_markdown(&msg.content, self.theme, content_width);
                for line in md_lines {
                    // Indent the line
                    let mut indented_spans = vec![Span::raw("  ")];
                    indented_spans.extend(
                        line.spans
                            .into_iter()
                            .map(|s| Span::styled(s.content.into_owned(), s.style)),
                    );
                    lines.push(Line::from(indented_spans));
                }
            }
        } else {
            // Plain text with wrapping for other messages
            let content_style = if msg.is_error {
                self.theme.error_style()
            } else if msg.role.starts_with("tool:") {
                Style::default().fg(Color::DarkGray)
            } else {
                self.theme.base_style()
            };

            let wrapped = textwrap::wrap(&msg.content, content_width);
            for line in wrapped {
                lines.push(Line::from(Span::styled(
                    format!("  {}", line),
                    content_style,
                )));
            }
        }

        // Empty line between messages
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

        // Render all messages into lines
        let width = inner.width as usize;
        let mut all_lines: Vec<Line> = Vec::new();

        for msg in self.messages {
            all_lines.extend(self.render_message(msg, width));
        }

        // Apply scroll and take visible lines
        let visible_lines: Vec<Line> = all_lines
            .into_iter()
            .skip(self.scroll)
            .take(inner.height as usize)
            .collect();

        let paragraph = Paragraph::new(visible_lines).wrap(Wrap { trim: false });
        paragraph.render(inner, buf);
    }
}

/// Calculate total height of messages
pub fn calculate_message_height(messages: &[ChatMessage], width: usize) -> usize {
    let mut total = 0;
    let theme = Theme::dark(); // Use default theme for calculation
    let content_width = width.saturating_sub(2);
    
    for msg in messages {
        // Role header
        total += 1;
        
        // Content lines - must match actual rendering logic
        if msg.role == "assistant" && !msg.is_error {
            if msg.content.is_empty() && msg.is_streaming {
                // Thinking indicator
                total += 1;
            } else {
                // Render markdown to count actual lines
                let md_lines = render_markdown(&msg.content, &theme, content_width);
                total += md_lines.len();
            }
        } else {
            // Plain text with wrapping
            let wrapped = textwrap::wrap(&msg.content, content_width);
            total += wrapped.len();
        }
        
        // Separator
        total += 1;
    }
    total
}
