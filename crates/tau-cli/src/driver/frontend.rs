//! The [`Frontend`] trait — what stdout / TUI must implement.

use async_trait::async_trait;
use tau_agent::{AgentEvent, InteractionRequest};
use tau_ai::{Model, Usage};

/// Snapshot of context passed once at session start so frontends can
/// render a banner.
pub struct SessionStart<'a> {
    pub model: &'a Model,
    /// Short session id (first 8 chars) when resuming or persisting.
    pub session_id: Option<&'a str>,
}

/// Side-channel signals that the frontend produces independently of
/// the normal input stream. Polled by the [`Session`](super::Session)
/// after each command / prompt cycle.
#[derive(Debug, Clone)]
pub enum FrontendAction {
    /// User chose "Execute now" in the plan-review modal — Session
    /// should immediately invoke `/plan approve` after the planner
    /// finishes its turn, skipping the manual command.
    ExecutePlanNow,
}

/// User-supplied input that drives the session forward.
pub enum UserInput {
    /// A normal prompt to send to the agent.
    Prompt(String),
    /// A slash command, leading `/` included (e.g. `/help`, `/model gpt-4`).
    Command(String),
    /// User asked to abort the current turn (Ctrl-C, etc.).
    Abort,
    /// User asked to inject a steering message mid-prompt.
    Steer(String),
    /// User asked to exit.
    Quit,
}

#[async_trait]
pub trait Frontend: Send {
    /// Called once before the main loop starts. Default: no-op.
    async fn on_session_start(&mut self, _info: SessionStart<'_>) {}

    /// Called once on exit.
    async fn on_session_end(&mut self) {}

    /// Block until the user gives the session something to do. Return
    /// `None` to end the session (EOF, window closed, etc.).
    async fn next_input(&mut self) -> Option<UserInput>;

    /// Render one agent event during an in-flight turn.
    async fn render_event(&mut self, event: AgentEvent);

    /// Called after each turn completes, with the run's total usage so
    /// the frontend can show a cost summary.
    async fn render_turn_end(&mut self, _total_usage: &Usage, _model: &Model) {}

    /// Show a system message (host-side, not from the agent — e.g.
    /// "Compaction failed").
    async fn show_system(&mut self, text: &str);

    /// Show an error message.
    async fn show_error(&mut self, text: &str);

    /// Handle a runtime interaction request from a tool (plan submit,
    /// ask question, tool confirm). Should resolve `req.response_tx`
    /// when the user answers.
    async fn handle_interaction(&mut self, req: InteractionRequest);

    /// Reset the frontend's local display state after a conversation
    /// reset (e.g. `/clear`). Default: no-op (stdout simply prints the
    /// confirmation system message).
    async fn reset_view(&mut self) {}

    /// Notify the frontend that the agent's runtime config changed
    /// (typically after `/model` or `/thinking`). TUI frontends should
    /// refresh their status line. Default: no-op.
    async fn on_config_change(&mut self, _config: &tau_agent::AgentConfig) {}

    /// Open a branch-selector overlay over the given messages. Return
    /// `true` if the frontend handled the selection (and will emit a
    /// `/branch <i>` command via its own input channel when the user
    /// picks), `false` if the caller should fall back to a text list.
    /// Default: `false`.
    async fn open_branch_selector(&mut self, _messages: &[tau_ai::Message]) -> bool {
        false
    }

    /// Drain any pending side-channel action. Polled by the
    /// [`Session`](super::Session) after each command / prompt cycle.
    /// Default: no action.
    fn take_action(&mut self) -> Option<FrontendAction> {
        None
    }

    /// Whether this frontend can render `tool.confirm` approval prompts.
    /// When `false`, the [`Session`](super::Session) installs
    /// `AutoAcceptAll` so elevated tools don't deadlock waiting for
    /// approval that can't be rendered.
    fn can_render_approval(&self) -> bool {
        false
    }

    /// Optional periodic tick during in-flight prompts. The
    /// [`Session`](super::Session) calls this in its event-pump
    /// `select!`, giving the frontend a chance to draw frames and
    /// process its own input. Return `Some(UserInput::Abort)` to
    /// interrupt the current turn. Default: never resolves (no ticks).
    async fn tick(&mut self) -> Option<UserInput> {
        std::future::pending().await
    }
}
