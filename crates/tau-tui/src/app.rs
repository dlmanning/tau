//! Main application framework

use crate::input::{Action, event_to_action};
use crate::theme::Theme;
use crossterm::{
    event::{
        self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};
use std::io::{self, Stdout};
use std::time::Duration;

/// Application state trait
pub trait AppState {
    /// Handle an input action, return true to continue, false to quit
    fn handle_action(&mut self, action: Action) -> bool;

    /// Render the UI
    fn render(&self, frame: &mut ratatui::Frame);

    /// Called on each tick (for animations)
    fn tick(&mut self) {}
}

/// Main application runner
pub struct App {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    theme: Theme,
    tick_rate: Duration,
}

impl App {
    /// Create a new application
    pub fn new() -> io::Result<Self> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(
            stdout,
            EnterAlternateScreen,
            EnableMouseCapture,
            EnableBracketedPaste
        )?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        Ok(Self {
            terminal,
            theme: Theme::default(),
            tick_rate: Duration::from_millis(100),
        })
    }

    /// Set the color theme
    pub fn with_theme(mut self, theme: Theme) -> Self {
        self.theme = theme;
        self
    }

    /// Set the tick rate for animations
    pub fn with_tick_rate(mut self, rate: Duration) -> Self {
        self.tick_rate = rate;
        self
    }

    /// Get the theme
    pub fn theme(&self) -> &Theme {
        &self.theme
    }

    /// Run the application with the given state
    pub fn run<S: AppState>(&mut self, state: &mut S) -> io::Result<()> {
        loop {
            // Render
            self.terminal.draw(|frame| {
                state.render(frame);
            })?;

            // Handle events
            if event::poll(self.tick_rate)? {
                let evt = event::read()?;
                if let Event::Key(_) | Event::Paste(_) = &evt {
                    if let Some(action) = event_to_action(evt) {
                        if !state.handle_action(action) {
                            return Ok(());
                        }
                    }
                }
            }

            // Tick for animations
            state.tick();
        }
    }

    /// Run with async event handling
    pub async fn run_async<S, F, Fut>(
        &mut self,
        state: &mut S,
        mut event_handler: F,
    ) -> io::Result<()>
    where
        S: AppState,
        F: FnMut(&mut S, Action) -> Fut,
        Fut: std::future::Future<Output = bool>,
    {
        use crossterm::event::EventStream;
        use futures::StreamExt;

        let mut event_stream = EventStream::new();

        loop {
            // Render
            self.terminal.draw(|frame| {
                state.render(frame);
            })?;

            // Handle events with timeout for ticks
            let event = tokio::time::timeout(self.tick_rate, event_stream.next()).await;

            match event {
                Ok(Some(Ok(evt))) => {
                    if let Some(action) = event_to_action(evt) {
                        if !event_handler(state, action).await {
                            return Ok(());
                        }
                    }
                }
                Ok(Some(Err(e))) => {
                    return Err(e);
                }
                Ok(None) => {
                    // Stream ended
                    return Ok(());
                }
                Err(_) => {
                    // Timeout - tick
                    state.tick();
                }
            }
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // Restore terminal
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture,
            DisableBracketedPaste
        );
        let _ = self.terminal.show_cursor();
    }
}
