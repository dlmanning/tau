//! Color theme support

use ratatui::style::{Color, Modifier, Style};

/// Color theme for the UI
#[derive(Debug, Clone)]
pub struct Theme {
    /// Background color
    pub bg: Color,
    /// Primary text color
    pub fg: Color,
    /// Dimmed/secondary text
    pub dim: Color,
    /// Accent color (highlights, prompts)
    pub accent: Color,
    /// Error color
    pub error: Color,
    /// Success color
    pub success: Color,
    /// Warning color
    pub warning: Color,
    /// Border color
    pub border: Color,
    /// Selection/highlight background
    pub selection_bg: Color,
    /// Code/preformatted text color
    pub code: Color,
    /// Link color
    pub link: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

impl Theme {
    /// Dark theme (default)
    pub fn dark() -> Self {
        Self {
            bg: Color::Reset,
            fg: Color::White,
            dim: Color::DarkGray,
            accent: Color::Cyan,
            error: Color::Red,
            success: Color::Green,
            warning: Color::Yellow,
            border: Color::DarkGray,
            selection_bg: Color::DarkGray,
            code: Color::Magenta,
            link: Color::Blue,
        }
    }

    /// Light theme
    pub fn light() -> Self {
        Self {
            bg: Color::White,
            fg: Color::Black,
            dim: Color::Gray,
            accent: Color::Blue,
            error: Color::Red,
            success: Color::Green,
            warning: Color::Rgb(180, 120, 0),
            border: Color::Gray,
            selection_bg: Color::LightBlue,
            code: Color::Magenta,
            link: Color::Blue,
        }
    }

    /// Get base style
    pub fn base_style(&self) -> Style {
        Style::default().fg(self.fg).bg(self.bg)
    }

    /// Get dimmed style
    pub fn dim_style(&self) -> Style {
        Style::default().fg(self.dim)
    }

    /// Get accent style
    pub fn accent_style(&self) -> Style {
        Style::default().fg(self.accent)
    }

    /// Get bold accent style
    pub fn accent_bold(&self) -> Style {
        Style::default()
            .fg(self.accent)
            .add_modifier(Modifier::BOLD)
    }

    /// Get error style
    pub fn error_style(&self) -> Style {
        Style::default().fg(self.error)
    }

    /// Get success style
    pub fn success_style(&self) -> Style {
        Style::default().fg(self.success)
    }

    /// Get code/preformatted style
    pub fn code_style(&self) -> Style {
        Style::default().fg(self.code)
    }

    /// Get border style
    pub fn border_style(&self) -> Style {
        Style::default().fg(self.border)
    }
}
