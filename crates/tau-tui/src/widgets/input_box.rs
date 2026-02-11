//! Text input widget

use crate::input::Action;
use crate::theme::Theme;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Style,
    widgets::{Block, Borders, Paragraph, Widget},
};
use unicode_width::UnicodeWidthStr;

/// Single-line text input widget
#[derive(Debug, Default)]
pub struct InputBox {
    /// Current input text
    content: String,
    /// Cursor position (character index, not byte index)
    cursor: usize,
    /// Horizontal scroll offset (in display width)
    scroll: usize,
    /// Placeholder text
    placeholder: String,
    /// Whether the input is focused
    focused: bool,
}

impl InputBox {
    /// Create a new input box
    pub fn new() -> Self {
        Self::default()
    }

    /// Set placeholder text
    pub fn with_placeholder(mut self, placeholder: impl Into<String>) -> Self {
        self.placeholder = placeholder.into();
        self
    }

    /// Set focus state
    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }

    /// Get the current content
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Set the content
    pub fn set_content(&mut self, content: impl Into<String>) {
        self.content = content.into();
        self.cursor = self.content.chars().count();
        self.update_scroll(80); // Default width
    }

    /// Clear the content
    pub fn clear(&mut self) {
        self.content.clear();
        self.cursor = 0;
        self.scroll = 0;
    }

    /// Get the byte offset for the current cursor position
    fn cursor_byte_offset(&self) -> usize {
        self.content
            .char_indices()
            .nth(self.cursor)
            .map(|(i, _)| i)
            .unwrap_or(self.content.len())
    }

    /// Get the display width of text before the cursor
    fn cursor_display_width(&self) -> usize {
        self.content
            .chars()
            .take(self.cursor)
            .map(|c| c.to_string().width())
            .sum()
    }

    /// Handle an input action
    pub fn handle_action(&mut self, action: &Action, width: u16) -> bool {
        let char_count = self.content.chars().count();

        match action {
            Action::Char(c) => {
                self.insert_char(*c);
                self.update_scroll(width as usize);
                true
            }
            Action::Backspace => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    let byte_offset = self.cursor_byte_offset();
                    // Find the next char boundary after this position
                    let next_boundary = self.content[byte_offset..]
                        .char_indices()
                        .nth(1)
                        .map(|(i, _)| byte_offset + i)
                        .unwrap_or(self.content.len());
                    self.content.drain(byte_offset..next_boundary);
                    self.update_scroll(width as usize);
                    true
                } else {
                    false
                }
            }
            Action::Delete => {
                if self.cursor < char_count {
                    let byte_offset = self.cursor_byte_offset();
                    // Find the next char boundary after this position
                    let next_boundary = self.content[byte_offset..]
                        .char_indices()
                        .nth(1)
                        .map(|(i, _)| byte_offset + i)
                        .unwrap_or(self.content.len());
                    self.content.drain(byte_offset..next_boundary);
                    true
                } else {
                    false
                }
            }
            Action::Left => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.update_scroll(width as usize);
                    true
                } else {
                    false
                }
            }
            Action::Right => {
                if self.cursor < char_count {
                    self.cursor += 1;
                    self.update_scroll(width as usize);
                    true
                } else {
                    false
                }
            }
            Action::Home => {
                self.cursor = 0;
                self.update_scroll(width as usize);
                true
            }
            Action::End => {
                self.cursor = char_count;
                self.update_scroll(width as usize);
                true
            }
            Action::ClearLine => {
                self.clear();
                true
            }
            Action::DeleteWord => {
                // Delete word before cursor
                let mut new_cursor = self.cursor;
                let chars: Vec<char> = self.content.chars().collect();

                // Skip trailing spaces
                while new_cursor > 0 && chars.get(new_cursor - 1) == Some(&' ') {
                    new_cursor -= 1;
                }
                // Skip word characters
                while new_cursor > 0 && chars.get(new_cursor - 1) != Some(&' ') {
                    new_cursor -= 1;
                }

                // Calculate byte offsets for the range to delete
                let start_byte = self
                    .content
                    .char_indices()
                    .nth(new_cursor)
                    .map(|(i, _)| i)
                    .unwrap_or(self.content.len());
                let end_byte = self.cursor_byte_offset();

                self.content.drain(start_byte..end_byte);
                self.cursor = new_cursor;
                self.update_scroll(width as usize);
                true
            }
            Action::Paste(text) => {
                for c in text.chars() {
                    // Convert newlines to spaces for single-line input
                    if c == '\n' || c == '\r' {
                        // Avoid double spaces from \r\n
                        if !self.content.ends_with(' ') && self.cursor > 0 {
                            self.insert_char(' ');
                        }
                    } else {
                        self.insert_char(c);
                    }
                }
                self.update_scroll(width as usize);
                true
            }
            _ => false,
        }
    }

    fn insert_char(&mut self, c: char) {
        let byte_offset = self.cursor_byte_offset();
        self.content.insert(byte_offset, c);
        self.cursor += 1;
    }

    fn update_scroll(&mut self, width: usize) {
        let visible_width = width.saturating_sub(4); // Account for borders/padding
        let cursor_pos = self.cursor_display_width();

        if cursor_pos < self.scroll {
            self.scroll = cursor_pos;
        } else if cursor_pos >= self.scroll + visible_width {
            self.scroll = cursor_pos - visible_width + 1;
        }
    }

    /// Render the input box
    pub fn render(&self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(if self.focused {
                theme.accent_style()
            } else {
                theme.border_style()
            });

        let inner = block.inner(area);
        block.render(area, buf);

        // Render content or placeholder
        let display_text = if self.content.is_empty() {
            self.placeholder.clone()
        } else {
            // Apply scroll
            let visible_width = inner.width as usize;
            let chars: Vec<char> = self.content.chars().collect();
            let mut start_idx = 0;
            let mut current_width = 0;

            // Find start position based on scroll
            for (i, c) in chars.iter().enumerate() {
                if current_width >= self.scroll {
                    start_idx = i;
                    break;
                }
                current_width += c.to_string().width();
            }

            // Collect visible characters
            let mut visible = String::new();
            current_width = 0;
            for c in chars.iter().skip(start_idx) {
                let char_width = c.to_string().width();
                if current_width + char_width > visible_width {
                    break;
                }
                visible.push(*c);
                current_width += char_width;
            }
            visible
        };

        let style = if self.content.is_empty() {
            theme.dim_style()
        } else {
            theme.base_style()
        };

        let paragraph = Paragraph::new(display_text).style(style);
        paragraph.render(inner, buf);

        // Render cursor if focused
        if self.focused && inner.width > 0 {
            let cursor_x = self.cursor_display_width().saturating_sub(self.scroll);
            if cursor_x < inner.width as usize {
                let x = inner.x + cursor_x as u16;
                let y = inner.y;
                if let Some(cell) = buf.cell_mut((x, y)) {
                    cell.set_style(Style::default().bg(theme.accent));
                }
            }
        }
    }
}
