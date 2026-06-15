use std::time::Instant;

use super::widgets::SelectorState;
use futures::FutureExt;
use ratatui::style::{Color, Modifier, Style};

use super::constants;

/// Pending interaction request waiting for user input in the TUI.
pub(super) struct PendingInteraction {
    pub question: String,
    pub options: Vec<tau_agent::QuestionOption>,
    pub response_tx: tokio::sync::oneshot::Sender<tau_agent::InteractionResponse>,
    pub selector: SelectorState,
}

/// Two-stage plan-review state: first the user sees the plan and picks
/// an action (Approve / Execute now / Reject); on Reject we flip to
/// `EnteringReason` and route the next input-box submit to the
/// rejection reason.
#[derive(Debug)]
pub(super) enum PlanModalMode {
    Reviewing,
    EnteringReason,
}

/// Pending plan submission awaiting user review. Mirrors
/// [`PendingInteraction`]; rendered as a full-screen overlay by
/// `render_plan_modal`.
pub(super) struct PendingPlan {
    pub plan: tau_tools::Plan,
    pub response_tx: tokio::sync::oneshot::Sender<tau_agent::InteractionResponse>,
    pub mode: PlanModalMode,
    /// Vertical scroll position (line offset) in the plan body.
    pub scroll: u16,
}

/// A `tool.confirm` gate awaiting the user's decision. Rendered as a
/// centered overlay by `render_approval_modal`. Gates can arrive
/// concurrently (parallel tool batches, subagents), so `TuiState`
/// queues them and shows one at a time.
pub(super) struct PendingApproval {
    pub tool_name: String,
    /// Human-readable activity string from the tool (what it's about
    /// to do).
    pub activity: String,
    /// Risk classification as reported by the gate payload.
    pub risk: String,
    /// Pretty-printed tool arguments.
    pub args: String,
    pub response_tx: tokio::sync::oneshot::Sender<tau_agent::InteractionResponse>,
    /// Vertical scroll position in the args body.
    pub scroll: u16,
}

/// Messages sent from UI to agent handler
#[derive(Debug)]
pub enum UiMessage {
    /// User submitted input
    Submit(String),
    /// User requested quit
    Quit,
    /// User requested clear
    Clear,
    /// User requested abort of current operation
    Abort,
    /// Slash command
    Command(String),
    /// Change model (index into available_models)
    ChangeModel(usize),
    /// Create branch from message index (None = empty branch)
    Branch(Option<usize>),
}

/// Cached git branch name with background refresh.
pub(super) struct GitBranchState {
    /// Current branch name, or None if not in a git repo.
    pub branch: Option<String>,
    /// When the branch was last refreshed.
    last_checked: Instant,
    /// Background task for the next refresh.
    task: Option<tokio::task::JoinHandle<Option<String>>>,
}

impl GitBranchState {
    pub fn new() -> Self {
        Self {
            branch: get_git_branch(),
            last_checked: Instant::now(),
            task: None,
        }
    }

    /// Collect the result from a previously spawned background refresh.
    pub fn poll(&mut self) {
        // unwrap()s are safe: is_some_and guards the take(),
        // is_finished() guarantees now_or_never() returns Some.
        if self.task.as_ref().is_some_and(|t| t.is_finished()) {
            if let Ok(branch) = self.task.take().unwrap().now_or_never().unwrap() {
                self.branch = branch;
            }
        }
    }

    /// Spawn a background refresh if enough time has elapsed since the last check.
    pub fn maybe_refresh(&mut self) {
        if self.last_checked.elapsed()
            > std::time::Duration::from_secs(constants::GIT_BRANCH_REFRESH_SECS)
            && self.task.is_none()
        {
            self.last_checked = Instant::now();
            self.task = Some(tokio::task::spawn_blocking(get_git_branch));
        }
    }
}

/// Get the current git branch name, or None if not in a git repo.
fn get_git_branch() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .stderr(std::process::Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Slow rainbow color shift for the τ glyph when agent is working.
/// Smoothly interpolates through the spectrum over ~4 seconds.
pub(super) fn rainbow_tau_style() -> Style {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    // Full hue rotation cycle
    let phase = (ms % constants::RAINBOW_CYCLE_MS) as f64 / constants::RAINBOW_CYCLE_MS as f64;
    // HSV to RGB with S=1, V=1 — hue rotates through 0..360
    let hue = phase * 360.0;
    let c = 1.0_f64;
    let x = 1.0 - ((hue / 60.0) % 2.0 - 1.0).abs();
    let (r, g, b) = match hue as u32 {
        0..60 => (c, x, 0.0),
        60..120 => (x, c, 0.0),
        120..180 => (0.0, c, x),
        180..240 => (0.0, x, c),
        240..300 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Style::default()
        .fg(Color::Rgb(
            (r * 255.0) as u8,
            (g * 255.0) as u8,
            (b * 255.0) as u8,
        ))
        .add_modifier(Modifier::BOLD)
}
