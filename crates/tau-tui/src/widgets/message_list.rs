//! Message list widget for displaying chat messages

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Widget, Wrap},
};
use textwrap;

use crate::{theme::Theme, widgets::markdown::render_markdown};

/// Get a tick counter for animations (~80ms per frame).
fn animation_tick() -> usize {
    (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        / 80) as usize
}

/// Color-cycling style for the streaming τ glyph.
/// Cycles: green → cyan → blue → cyan → green
fn streaming_tau_style() -> Style {
    const COLORS: [Color; 6] = [
        Color::Green,
        Color::Cyan,
        Color::Blue,
        Color::Cyan,
        Color::Green,
        Color::LightGreen,
    ];
    let idx = animation_tick() % COLORS.len();
    Style::default()
        .fg(COLORS[idx])
        .add_modifier(Modifier::BOLD)
}

/// Orbiting braille dots around τ — returns (left_char, right_char).
/// The dot pattern rotates: left, top-left, top, top-right, right, etc.
fn orbiting_braille(tick: usize) -> (char, char) {
    // 8 positions around τ, represented as (left, right) braille pairs
    const FRAMES: [(char, char); 8] = [
        ('⠂', ' '), // left
        ('⠁', ' '), // upper-left
        ('⠈', '⠁'), // top (split across both sides)
        (' ', '⠈'), // upper-right
        (' ', '⠐'), // right
        (' ', '⠠'), // lower-right
        ('⠠', '⠄'), // bottom (split across both sides)
        ('⠄', ' '), // lower-left
    ];
    FRAMES[tick % FRAMES.len()]
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

        let (role_text, role_style, prefix) = match msg.role.as_str() {
            "user" => ("You", self.theme.accent_bold(), "▶ "),
            "assistant" => {
                if msg.is_streaming {
                    let tau_style = streaming_tau_style();
                    ("τ", tau_style, "")
                } else {
                    (
                        "τ",
                        self.theme.success_style().add_modifier(Modifier::BOLD),
                        "",
                    )
                }
            }
            r if r.starts_with("agent:") => {
                let desc = &r[6..];
                let style = if msg.is_error {
                    self.theme.error_style()
                } else {
                    Style::default().fg(Color::Cyan)
                };
                (desc, style, "◇ ")
            }
            "steer" => ("You", self.theme.dim_style().add_modifier(Modifier::ITALIC), "▷ "),
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

        if msg.role == "assistant" {
            // Custom rendering for τ header with animations
            let spans = if msg.is_streaming {
                let tick = animation_tick();
                let orbit = orbiting_braille(tick);
                vec![
                    Span::styled(orbit.0.to_string(), Style::default().fg(Color::DarkGray)),
                    Span::styled("τ", role_style),
                    Span::styled(orbit.1.to_string(), Style::default().fg(Color::DarkGray)),
                ]
            } else {
                vec![Span::styled("τ", role_style)]
            };
            lines.push(Line::from(spans));
        } else {
            let header = format!("{}{}", prefix, role_text);
            lines.push(Line::from(Span::styled(header, role_style)));
        }

        let content_width = width.saturating_sub(2);

        if msg.role == "assistant" && !msg.is_error {
            if msg.content.is_empty() && msg.is_streaming {
                // No content yet — the orbiting dots on the header are enough
            } else {
                let md_lines = render_markdown(&msg.content, self.theme, content_width);
                for line in md_lines {
                    let mut indented_spans = vec![Span::raw("  ")];
                    indented_spans.extend(
                        line.spans
                            .into_iter()
                            .map(|s| Span::styled(s.content.into_owned(), s.style)),
                    );
                    lines.push(Line::from(indented_spans));
                }
            }
        } else if msg.role.starts_with("agent:") && msg.is_streaming {
            let frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let frame_idx = (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
                / 80) as usize
                % frames.len();
            let display = if msg.content.is_empty() {
                format!("  {} working...", frames[frame_idx])
            } else {
                format!("  {} {}", frames[frame_idx], msg.content)
            };
            lines.push(Line::from(Span::styled(
                display,
                Style::default().fg(Color::Cyan),
            )));
        } else if msg.role.starts_with("agent:") {
            // Finished agent — show ✓ or ✗ prefix with stats
            let (indicator, style) = if msg.is_error {
                ("✗", self.theme.error_style())
            } else {
                ("✓", self.theme.success_style())
            };
            lines.push(Line::from(Span::styled(
                format!("  {} {}", indicator, msg.content),
                style,
            )));
        } else {
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
            all_lines.extend(self.render_message(msg, width));
        }

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
pub fn calculate_message_height(messages: &[ChatMessage], width: usize, theme: &Theme) -> usize {
    let mut total = 0;
    let content_width = width.saturating_sub(2);

    for msg in messages {
        total += 1;

        if msg.role == "assistant" && !msg.is_error {
            if !(msg.content.is_empty() && msg.is_streaming) {
                let md_lines = render_markdown(&msg.content, theme, content_width);
                total += md_lines.len();
            }
        } else {
            let wrapped = textwrap::wrap(&msg.content, content_width);
            total += wrapped.len();
        }

        total += 1;
    }
    total
}
