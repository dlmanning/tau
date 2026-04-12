/// Tick interval for TUI animations (spinner, rainbow glyph).
pub const TICK_INTERVAL_MS: u64 = 80;

/// Lines scrolled per mouse wheel tick.
pub const SCROLL_LINES_MOUSE: usize = 3;

/// Lines scrolled per PageUp/PageDown.
pub const SCROLL_LINES_PAGE: usize = 10;

/// Maximum characters shown for tool result previews.
pub const TOOL_RESULT_PREVIEW_CHARS: usize = 200;

/// Maximum characters shown for branch selector message previews.
pub const BRANCH_PREVIEW_CHARS: usize = 50;

/// Interval between background git branch refresh polls (seconds).
pub const GIT_BRANCH_REFRESH_SECS: u64 = 5;

/// Channel capacity for UiMessage sender.
pub const UI_CHANNEL_CAPACITY: usize = 32;

/// Full rainbow hue cycle duration in milliseconds.
pub const RAINBOW_CYCLE_MS: u128 = 4096;
