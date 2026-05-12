use super::{input::Action, widgets::message_list::ChatMessage};
use crossterm::event::{Event, MouseEventKind};

use super::constants;
use super::state::TuiState;
use super::types::UiMessage;

impl TuiState {
    /// Handle mouse scroll events.
    fn handle_mouse_scroll(&mut self, kind: MouseEventKind) {
        match kind {
            MouseEventKind::ScrollUp => {
                self.scroll = self.scroll.saturating_sub(constants::SCROLL_LINES_MOUSE);
                self.follow_bottom = false;
            }
            MouseEventKind::ScrollDown => {
                self.scroll = self.scroll.saturating_add(constants::SCROLL_LINES_MOUSE);
            }
            _ => {}
        }
    }

    /// Handle keyboard action.
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
                self.scroll = self.scroll.saturating_sub(constants::SCROLL_LINES_PAGE);
                self.follow_bottom = false;
                true
            }
            Action::PageDown => {
                self.scroll = self.scroll.saturating_add(constants::SCROLL_LINES_PAGE);
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

    /// Handle a terminal event while idle (no prompt executing).
    /// Returns `false` if the TUI should exit.
    pub async fn handle_event_while_idle(&mut self, event: Event, area_width: u16) -> bool {
        match event {
            Event::Key(key) => {
                let action = super::input::key_to_action(key);
                self.handle_action(action, area_width).await
            }
            Event::Paste(text) => self.handle_action(Action::Paste(text), area_width).await,
            Event::Mouse(mouse) => {
                self.handle_mouse_scroll(mouse.kind);
                true
            }
            Event::Resize(_, _) => true,
            _ => true,
        }
    }
}
