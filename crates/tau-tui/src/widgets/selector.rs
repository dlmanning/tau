//! Selector popup widget for choosing from a list of options

use crate::Theme;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, HighlightSpacing, List, ListItem, ListState, Widget},
};

/// A popup selector for choosing from a list of options
pub struct Selector<'a> {
    title: &'a str,
    items: Vec<SelectorItem<'a>>,
    selected: usize,
    theme: &'a Theme,
}

/// An item in the selector (borrowed version)
pub struct SelectorItem<'a> {
    /// Display label
    pub label: &'a str,
    /// Optional description
    pub description: Option<&'a str>,
    /// Whether this item is currently active
    pub is_current: bool,
}

/// An item in the selector (owned version for dynamic content)
pub struct OwnedSelectorItem {
    /// Display label
    pub label: String,
    /// Optional description
    pub description: Option<String>,
    /// Whether this item is currently active
    pub is_current: bool,
}

/// A popup selector with owned items (for dynamic content)
pub struct OwnedSelector<'a> {
    title: String,
    items: Vec<OwnedSelectorItem>,
    selected: usize,
    theme: &'a Theme,
}

impl<'a> OwnedSelector<'a> {
    /// Create a new owned selector
    pub fn new(title: impl Into<String>, items: Vec<OwnedSelectorItem>, theme: &'a Theme) -> Self {
        let selected = items.iter().position(|item| item.is_current).unwrap_or(0);
        Self {
            title: title.into(),
            items,
            selected,
            theme,
        }
    }

    /// Set the selected index
    pub fn with_selected(mut self, index: usize) -> Self {
        self.selected = index.min(self.items.len().saturating_sub(1));
        self
    }

    /// Calculate the ideal size for the popup
    fn calculate_size(&self) -> (u16, u16) {
        let mut max_width = self.title.len() + 4;

        for item in &self.items {
            let item_width = item.label.len() + 6;
            max_width = max_width.max(item_width);

            if let Some(ref desc) = item.description {
                max_width = max_width.max(desc.len() + 8);
            }
        }

        let height = self.items.len() as u16 + 2;
        let width = (max_width as u16).clamp(20, 80);

        (width, height.min(20))
    }

    /// Render the selector centered in the given area
    pub fn render_centered(&self, area: Rect, buf: &mut Buffer) {
        let (width, height) = self.calculate_size();

        let x = area.x + (area.width.saturating_sub(width)) / 2;
        let y = area.y + (area.height.saturating_sub(height)) / 2;

        let popup_area = Rect::new(x, y, width.min(area.width), height.min(area.height));

        Clear.render(popup_area, buf);

        let items: Vec<ListItem> = self
            .items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let is_selected = i == self.selected;
                let prefix = if item.is_current { "● " } else { "  " };

                let style = if is_selected {
                    Style::default()
                        .bg(self.theme.accent)
                        .fg(self.theme.bg)
                        .add_modifier(Modifier::BOLD)
                } else if item.is_current {
                    self.theme.accent_style()
                } else {
                    self.theme.base_style()
                };

                let content = format!("{}{}", prefix, item.label);
                ListItem::new(Line::from(Span::styled(content, style)))
            })
            .collect();

        let block = Block::default()
            .title(format!(" {} ", self.title))
            .title_style(self.theme.accent_bold())
            .borders(Borders::ALL)
            .border_style(self.theme.accent_style());

        let list = List::new(items)
            .block(block)
            .highlight_spacing(HighlightSpacing::Always);

        let mut state = ListState::default();
        state.select(Some(self.selected));

        ratatui::widgets::StatefulWidget::render(list, popup_area, buf, &mut state);
    }
}

impl<'a> Selector<'a> {
    /// Create a new selector
    pub fn new(title: &'a str, items: Vec<SelectorItem<'a>>, theme: &'a Theme) -> Self {
        // Find the currently selected item (the one marked as current)
        let selected = items.iter().position(|item| item.is_current).unwrap_or(0);

        Self {
            title,
            items,
            selected,
            theme,
        }
    }

    /// Set the selected index
    pub fn with_selected(mut self, index: usize) -> Self {
        self.selected = index.min(self.items.len().saturating_sub(1));
        self
    }

    /// Get the selected index
    pub fn selected(&self) -> usize {
        self.selected
    }

    /// Move selection up
    pub fn up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        } else {
            // Wrap to bottom
            self.selected = self.items.len().saturating_sub(1);
        }
    }

    /// Move selection down
    pub fn down(&mut self) {
        if self.selected < self.items.len().saturating_sub(1) {
            self.selected += 1;
        } else {
            // Wrap to top
            self.selected = 0;
        }
    }

    /// Calculate the ideal size for the popup
    fn calculate_size(&self) -> (u16, u16) {
        let mut max_width = self.title.len() + 4; // Title + borders + padding

        for item in &self.items {
            let item_width = item.label.len() + 6; // Prefix + padding
            max_width = max_width.max(item_width);

            if let Some(desc) = item.description {
                max_width = max_width.max(desc.len() + 8);
            }
        }

        let height = self.items.len() as u16 + 2; // Items + borders
        let width = (max_width as u16).clamp(20, 60);

        (width, height.min(20))
    }

    /// Render the selector centered in the given area
    pub fn render_centered(&self, area: Rect, buf: &mut Buffer) {
        let (width, height) = self.calculate_size();

        // Center the popup
        let x = area.x + (area.width.saturating_sub(width)) / 2;
        let y = area.y + (area.height.saturating_sub(height)) / 2;

        let popup_area = Rect::new(x, y, width.min(area.width), height.min(area.height));

        // Clear the area behind the popup
        Clear.render(popup_area, buf);

        // Create list items
        let items: Vec<ListItem> = self
            .items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let is_selected = i == self.selected;

                let prefix = if item.is_current { "● " } else { "  " };

                let style = if is_selected {
                    Style::default()
                        .bg(self.theme.accent)
                        .fg(self.theme.bg)
                        .add_modifier(Modifier::BOLD)
                } else if item.is_current {
                    self.theme.accent_style()
                } else {
                    self.theme.base_style()
                };

                let content = format!("{}{}", prefix, item.label);
                ListItem::new(Line::from(Span::styled(content, style)))
            })
            .collect();

        let block = Block::default()
            .title(format!(" {} ", self.title))
            .title_style(self.theme.accent_bold())
            .borders(Borders::ALL)
            .border_style(self.theme.accent_style());

        let list = List::new(items)
            .block(block)
            .highlight_spacing(HighlightSpacing::Always);

        // We need to render with state for the selection highlight
        let mut state = ListState::default();
        state.select(Some(self.selected));

        // Render the list
        ratatui::widgets::StatefulWidget::render(list, popup_area, buf, &mut state);
    }
}

/// State for the selector popup
#[derive(Default)]
pub struct SelectorState {
    /// Currently selected index
    pub selected: usize,
    /// Whether the selector is visible
    pub visible: bool,
}

impl SelectorState {
    /// Show the selector
    pub fn show(&mut self) {
        self.visible = true;
    }

    /// Hide the selector
    pub fn hide(&mut self) {
        self.visible = false;
    }

    /// Toggle visibility
    pub fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    /// Move selection up
    pub fn up(&mut self, item_count: usize) {
        if item_count == 0 {
            return;
        }
        if self.selected > 0 {
            self.selected -= 1;
        } else {
            self.selected = item_count - 1;
        }
    }

    /// Move selection down
    pub fn down(&mut self, item_count: usize) {
        if item_count == 0 {
            return;
        }
        if self.selected < item_count - 1 {
            self.selected += 1;
        } else {
            self.selected = 0;
        }
    }
}
