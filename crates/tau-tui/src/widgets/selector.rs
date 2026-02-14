//! Selector popup widget for choosing from a list of options

use crate::Theme;
use ratatui::{
    buffer::Buffer,
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, HighlightSpacing, List, ListItem, ListState, Widget},
};

/// Maximum width for selector popups
const MAX_POPUP_WIDTH: u16 = 80;

// --- Shared rendering helpers ---

/// Compute the popup size given a title, item labels/descriptions, and max width.
fn compute_popup_size(
    title_len: usize,
    items: impl Iterator<Item = (usize, Option<usize>)>, // (label_len, desc_len)
    count: usize,
) -> (u16, u16) {
    let mut max_width = title_len + 4;
    for (label_len, desc_len) in items {
        max_width = max_width.max(label_len + 6);
        if let Some(d) = desc_len {
            max_width = max_width.max(d + 8);
        }
    }
    let height = count as u16 + 2;
    let width = (max_width as u16).clamp(20, MAX_POPUP_WIDTH);
    (width, height.min(20))
}

/// Render a generic selector popup centered in `area`.
fn render_selector_popup(
    title: &str,
    items: Vec<ListItem<'_>>,
    selected: usize,
    popup_size: (u16, u16),
    theme: &Theme,
    area: Rect,
    buf: &mut Buffer,
) {
    let (width, height) = popup_size;
    let x = area.x + (area.width.saturating_sub(width)) / 2;
    let y = area.y + (area.height.saturating_sub(height)) / 2;
    let popup_area = Rect::new(x, y, width.min(area.width), height.min(area.height));

    Clear.render(popup_area, buf);

    let block = Block::default()
        .title(format!(" {} ", title))
        .title_style(theme.accent_bold())
        .borders(Borders::ALL)
        .border_style(theme.accent_style());

    let list = List::new(items)
        .block(block)
        .highlight_spacing(HighlightSpacing::Always);

    let mut state = ListState::default();
    state.select(Some(selected));

    ratatui::widgets::StatefulWidget::render(list, popup_area, buf, &mut state);
}

/// Build a styled `ListItem` for a selector entry.
fn build_list_item<'a>(
    label: &str,
    is_current: bool,
    is_selected: bool,
    theme: &Theme,
) -> ListItem<'a> {
    let prefix = if is_current { "● " } else { "  " };
    let style = if is_selected {
        Style::default()
            .bg(theme.accent)
            .fg(theme.bg)
            .add_modifier(Modifier::BOLD)
    } else if is_current {
        theme.accent_style()
    } else {
        theme.base_style()
    };
    let content = format!("{}{}", prefix, label);
    ListItem::new(Line::from(Span::styled(content, style)))
}

// --- Borrowed selector ---

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

impl<'a> Selector<'a> {
    /// Create a new selector
    pub fn new(title: &'a str, items: Vec<SelectorItem<'a>>, theme: &'a Theme) -> Self {
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
            self.selected = self.items.len().saturating_sub(1);
        }
    }

    /// Move selection down
    pub fn down(&mut self) {
        if self.selected < self.items.len().saturating_sub(1) {
            self.selected += 1;
        } else {
            self.selected = 0;
        }
    }

    /// Render the selector centered in the given area
    pub fn render_centered(&self, area: Rect, buf: &mut Buffer) {
        let size = compute_popup_size(
            self.title.len(),
            self.items
                .iter()
                .map(|item| (item.label.len(), item.description.map(|d| d.len()))),
            self.items.len(),
        );

        let list_items: Vec<ListItem> = self
            .items
            .iter()
            .enumerate()
            .map(|(i, item)| build_list_item(item.label, item.is_current, i == self.selected, self.theme))
            .collect();

        render_selector_popup(self.title, list_items, self.selected, size, self.theme, area, buf);
    }
}

// --- Owned selector ---

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

    /// Render the selector centered in the given area
    pub fn render_centered(&self, area: Rect, buf: &mut Buffer) {
        let size = compute_popup_size(
            self.title.len(),
            self.items
                .iter()
                .map(|item| (item.label.len(), item.description.as_ref().map(|d| d.len()))),
            self.items.len(),
        );

        let list_items: Vec<ListItem> = self
            .items
            .iter()
            .enumerate()
            .map(|(i, item)| build_list_item(&item.label, item.is_current, i == self.selected, self.theme))
            .collect();

        render_selector_popup(&self.title, list_items, self.selected, size, self.theme, area, buf);
    }
}

// --- Selector state ---

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
