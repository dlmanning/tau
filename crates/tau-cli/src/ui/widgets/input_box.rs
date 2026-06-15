//! Text input widget

use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::Style,
    widgets::{Block, Borders, Paragraph, Widget},
};
use unicode_width::UnicodeWidthStr;

use super::super::{input::Action, theme::Theme};

/// Text input widget. Renders as a single line (embedded newlines show
/// as `⏎`), but the content — and therefore the submitted prompt —
/// preserves newlines from pastes and Shift/Alt+Enter.
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
    /// Previously committed entries, oldest first.
    history: Vec<String>,
    /// `Some(i)` while browsing `history[i]` with Up/Down.
    history_pos: Option<usize>,
    /// The in-progress draft stashed while browsing history.
    draft: String,
}

/// Display width of one char as the input box renders it: newlines
/// render as the `⏎` glyph (width 1).
fn char_display_width(c: char) -> usize {
    if c == '\n' {
        1
    } else {
        c.to_string().width()
    }
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
    #[allow(dead_code)]
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
            .map(char_display_width)
            .sum()
    }

    /// Commit the current content: push it to history (deduplicating
    /// consecutive repeats), reset history browsing, clear the box,
    /// and return the text.
    pub fn commit(&mut self) -> String {
        let text = self.content.clone();
        if !text.trim().is_empty() && self.history.last() != Some(&text) {
            self.history.push(text.clone());
        }
        self.history_pos = None;
        self.draft.clear();
        self.clear();
        text
    }

    /// Load a history entry (or the stashed draft for `None`) into the
    /// box with the cursor at the end.
    fn load_history(&mut self, pos: Option<usize>, width: usize) {
        self.history_pos = pos;
        self.content = match pos {
            Some(i) => self.history[i].clone(),
            None => std::mem::take(&mut self.draft),
        };
        self.cursor = self.content.chars().count();
        self.update_scroll(width);
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
                let mut new_cursor = self.cursor;
                let chars: Vec<char> = self.content.chars().collect();

                while new_cursor > 0 && chars.get(new_cursor - 1) == Some(&' ') {
                    new_cursor -= 1;
                }
                while new_cursor > 0 && chars.get(new_cursor - 1) != Some(&' ') {
                    new_cursor -= 1;
                }

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
                // Preserve newlines (normalized to `\n`) — pasted code
                // must survive round-trip into the prompt.
                let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
                for c in normalized.chars() {
                    self.insert_char(c);
                }
                self.update_scroll(width as usize);
                true
            }
            Action::Up => {
                let next = match self.history_pos {
                    None if !self.history.is_empty() => {
                        self.draft = self.content.clone();
                        Some(self.history.len() - 1)
                    }
                    Some(i) if i > 0 => Some(i - 1),
                    other => return other.is_some(), // top of history: consume; no history: pass
                };
                self.load_history(next, width as usize);
                true
            }
            Action::Down => match self.history_pos {
                Some(i) => {
                    let next = if i + 1 < self.history.len() {
                        Some(i + 1)
                    } else {
                        None // back to the stashed draft
                    };
                    self.load_history(next, width as usize);
                    true
                }
                None => false,
            },
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

        let display_text = if self.content.is_empty() {
            self.placeholder.clone()
        } else {
            let visible_width = inner.width as usize;
            let chars: Vec<char> = self.content.chars().collect();
            let mut start_idx = 0;
            let mut current_width = 0;

            for (i, c) in chars.iter().enumerate() {
                if current_width >= self.scroll {
                    start_idx = i;
                    break;
                }
                current_width += char_display_width(*c);
            }

            let mut visible = String::new();
            current_width = 0;
            for c in chars.iter().skip(start_idx) {
                let char_width = char_display_width(*c);
                if current_width + char_width > visible_width {
                    break;
                }
                // Newlines are real data but render as a marker in the
                // single-line view.
                visible.push(if *c == '\n' { '⏎' } else { *c });
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Pasted multiline text must keep its newlines in the content
    /// (the old behavior flattened them to spaces, corrupting code).
    #[test]
    fn paste_preserves_newlines() {
        let mut input = InputBox::new();
        input.handle_action(&Action::Paste("fn main() {\r\n    body\r\n}".into()), 80);
        assert_eq!(input.content(), "fn main() {\n    body\n}");
    }

    /// Up recalls prior entries newest-first; Down returns through
    /// them and finally restores the unsent draft.
    #[test]
    fn history_round_trip_preserves_draft() {
        let mut input = InputBox::new();
        input.handle_action(&Action::Paste("first".into()), 80);
        input.commit();
        input.handle_action(&Action::Paste("second".into()), 80);
        input.commit();

        input.handle_action(&Action::Paste("draft".into()), 80);
        input.handle_action(&Action::Up, 80);
        assert_eq!(input.content(), "second");
        input.handle_action(&Action::Up, 80);
        assert_eq!(input.content(), "first");
        // Top of history: Up is consumed but content stays.
        input.handle_action(&Action::Up, 80);
        assert_eq!(input.content(), "first");
        input.handle_action(&Action::Down, 80);
        assert_eq!(input.content(), "second");
        input.handle_action(&Action::Down, 80);
        assert_eq!(input.content(), "draft");
    }

    /// Consecutive identical commits don't duplicate history entries,
    /// and blank commits are not recorded.
    #[test]
    fn history_dedupes_and_skips_blank() {
        let mut input = InputBox::new();
        input.handle_action(&Action::Paste("same".into()), 80);
        input.commit();
        input.handle_action(&Action::Paste("same".into()), 80);
        input.commit();
        input.handle_action(&Action::Paste("   ".into()), 80);
        input.commit();
        assert_eq!(input.history.len(), 1);
    }

    /// With no history, Up is not consumed (falls through to the
    /// caller); with history browsing inactive, Down is not consumed.
    #[test]
    fn unconsumed_arrows_fall_through() {
        let mut input = InputBox::new();
        assert!(!input.handle_action(&Action::Up, 80));
        assert!(!input.handle_action(&Action::Down, 80));
    }
}
