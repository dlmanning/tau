//! Animated spinner widget

use crate::theme::Theme;
use ratatui::{buffer::Buffer, layout::Rect, text::Span, widgets::Widget};
use std::time::{Duration, Instant};

/// Spinner animation frames
const SPINNER_FRAMES: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Animated spinner widget
pub struct Spinner<'a> {
    label: &'a str,
    theme: &'a Theme,
    start_time: Instant,
}

impl<'a> Spinner<'a> {
    /// Create a new spinner
    pub fn new(label: &'a str, theme: &'a Theme) -> Self {
        Self {
            label,
            theme,
            start_time: Instant::now(),
        }
    }

    /// Create with a specific start time (for consistent animation)
    pub fn with_start_time(mut self, start: Instant) -> Self {
        self.start_time = start;
        self
    }

    /// Get the current frame based on elapsed time
    fn current_frame(&self) -> &'static str {
        let elapsed = self.start_time.elapsed();
        let frame_duration = Duration::from_millis(80);
        let frame_index = (elapsed.as_millis() / frame_duration.as_millis()) as usize;
        SPINNER_FRAMES[frame_index % SPINNER_FRAMES.len()]
    }
}

impl Widget for Spinner<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width < 3 {
            return;
        }

        let frame = self.current_frame();
        let text = format!("{} {}", frame, self.label);

        let span = Span::styled(&text, self.theme.accent_style());
        buf.set_span(area.x, area.y, &span, area.width);
    }
}

/// Simple loading indicator (non-animated)
pub struct LoadingIndicator<'a> {
    label: &'a str,
    theme: &'a Theme,
}

impl<'a> LoadingIndicator<'a> {
    /// Create a new loading indicator
    pub fn new(label: &'a str, theme: &'a Theme) -> Self {
        Self { label, theme }
    }
}

impl Widget for LoadingIndicator<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let text = format!("⋯ {}", self.label);
        let span = Span::styled(&text, self.theme.dim_style());
        buf.set_span(area.x, area.y, &span, area.width);
    }
}
