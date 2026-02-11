//! Input handling

use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};

/// Processed input action
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Regular character input
    Char(char),
    /// Enter/submit
    Submit,
    /// Backspace
    Backspace,
    /// Delete
    Delete,
    /// Move cursor left
    Left,
    /// Move cursor right
    Right,
    /// Move cursor up
    Up,
    /// Move cursor down
    Down,
    /// Move to start of line
    Home,
    /// Move to end of line
    End,
    /// Page up
    PageUp,
    /// Page down
    PageDown,
    /// Tab
    Tab,
    /// Shift+Tab
    BackTab,
    /// Escape
    Escape,
    /// Ctrl+C (interrupt)
    Interrupt,
    /// Ctrl+D (EOF)
    Eof,
    /// Ctrl+L (clear screen)
    Clear,
    /// Ctrl+U (clear line)
    ClearLine,
    /// Ctrl+W (delete word)
    DeleteWord,
    /// Ctrl+A (select all / start of line)
    SelectAll,
    /// Paste (from clipboard or bracketed paste)
    Paste(String),
    /// Copy selection
    Copy,
    /// Cut selection
    Cut,
    /// Undo
    Undo,
    /// Redo
    Redo,
    /// Quit application
    Quit,
    /// Open model selector
    ModelSelect,
    /// Unknown/unhandled
    Unknown,
}

/// Convert a crossterm key event to an action
pub fn key_to_action(event: KeyEvent) -> Action {
    let KeyEvent {
        code, modifiers, ..
    } = event;

    // Handle Ctrl combinations first
    if modifiers.contains(KeyModifiers::CONTROL) {
        return match code {
            KeyCode::Char('c') => Action::Interrupt,
            KeyCode::Char('d') => Action::Eof,
            KeyCode::Char('l') => Action::Clear,
            KeyCode::Char('u') => Action::ClearLine,
            KeyCode::Char('w') => Action::DeleteWord,
            KeyCode::Char('a') => Action::SelectAll,
            KeyCode::Char('z') => Action::Undo,
            KeyCode::Char('y') => Action::Redo,
            KeyCode::Char('q') => Action::Quit,
            KeyCode::Char('k') => Action::ModelSelect,
            _ => Action::Unknown,
        };
    }

    // Handle Alt combinations
    if modifiers.contains(KeyModifiers::ALT) {
        return Action::Unknown;
    }

    // Regular keys
    match code {
        KeyCode::Char(c) => Action::Char(c),
        KeyCode::Enter => Action::Submit,
        KeyCode::Backspace => Action::Backspace,
        KeyCode::Delete => Action::Delete,
        KeyCode::Left => Action::Left,
        KeyCode::Right => Action::Right,
        KeyCode::Up => Action::Up,
        KeyCode::Down => Action::Down,
        KeyCode::Home => Action::Home,
        KeyCode::End => Action::End,
        KeyCode::PageUp => Action::PageUp,
        KeyCode::PageDown => Action::PageDown,
        KeyCode::Tab => {
            if modifiers.contains(KeyModifiers::SHIFT) {
                Action::BackTab
            } else {
                Action::Tab
            }
        }
        KeyCode::BackTab => Action::BackTab,
        KeyCode::Esc => Action::Escape,
        _ => Action::Unknown,
    }
}

/// Convert a crossterm event to an action
pub fn event_to_action(event: Event) -> Option<Action> {
    match event {
        Event::Key(key_event) => Some(key_to_action(key_event)),
        Event::Paste(text) => Some(Action::Paste(text)),
        _ => None,
    }
}
