//! TUI frontend implementation. Owns the terminal, the `TuiState`,
//! and the crossterm event stream; translates TUI events into
//! [`UserInput`] and renders [`AgentEvent`]s into [`TuiState`] for the
//! existing render path.

use std::io::Stdout;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event,
    EventStream,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures::StreamExt;
use ratatui::{Terminal, backend::CrosstermBackend};
use tau_agent::{
    AgentEvent, InteractionKind, InteractionRequest, InteractionResponse,
};
use tau_ai::{Model, Usage};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::driver::{Frontend, FrontendAction, SessionStart, UserInput};

use super::constants;
use super::input::{Action, key_to_action};
use super::state::TuiState;
use super::types::{PendingInteraction, PendingPlan, PlanModalMode, UiMessage};
use super::widgets::{SelectorState, message_list::ChatMessage};

/// Minimum gap between redraws — ~60fps. Coalesces rapid `render_event`
/// calls during streaming so we don't redraw on every char.
const MIN_REDRAW_GAP: Duration = Duration::from_millis(16);

pub struct TuiFrontend {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    state: TuiState,
    /// UI messages emitted by the state's action handler (e.g.
    /// `Action::Submit` → `UiMessage::Submit(content)`).
    ui_rx: mpsc::Receiver<UiMessage>,
    /// Crossterm input events forwarded from a dedicated background
    /// task. Buffering input here means keys aren't lost when the
    /// `Session`'s event pump is busy with agent events.
    crossterm_rx: mpsc::Receiver<Event>,
    /// Keeps the crossterm pump alive for the frontend's lifetime.
    _crossterm_task: JoinHandle<()>,
    tick_interval: tokio::time::Interval,
    available_models: Vec<Model>,
    /// Last time we issued a draw. Used by [`maybe_redraw`] to coalesce
    /// rapid redraws during streaming.
    last_redraw: Instant,
    /// Side-channel action drained by `Session` via `take_action`. Set
    /// when the user picks "Execute now" in the plan-review modal so
    /// the Session knows to auto-fire `/plan approve` after the
    /// planner's turn ends.
    pending_action: Option<FrontendAction>,
}

impl TuiFrontend {
    /// Construct the frontend. Enables raw mode + alternate screen.
    /// The returned frontend must be passed to
    /// [`crate::driver::Session::drive`] which will eventually call
    /// `on_session_end` to restore the terminal.
    pub async fn new(
        config: &tau_agent::AgentConfig,
        available_models: Vec<Model>,
    ) -> anyhow::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(
            stdout,
            EnterAlternateScreen,
            EnableMouseCapture,
            EnableBracketedPaste
        )?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        // Restore the terminal before the default panic output runs, so
        // the message is readable and the shell isn't left in raw mode
        // on the alternate screen (`on_session_end` never runs when we
        // unwind). The hook is process-global and stays installed after
        // a clean exit — re-emitting the restore sequences on a normal
        // screen is harmless.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let _ = disable_raw_mode();
            let _ = execute!(
                std::io::stdout(),
                LeaveAlternateScreen,
                DisableMouseCapture,
                DisableBracketedPaste
            );
            prev_hook(info);
        }));

        let (ui_tx, ui_rx) = mpsc::channel::<UiMessage>(constants::UI_CHANNEL_CAPACITY);
        let state = TuiState::new(config, available_models.clone(), ui_tx);

        // Background task: pumps crossterm events into an mpsc so we
        // don't lose keypresses while the `Session` is busy.
        let (crossterm_tx, crossterm_rx) = mpsc::channel::<Event>(256);
        let crossterm_task = tokio::spawn(async move {
            let mut event_stream = EventStream::new();
            while let Some(Ok(ev)) = event_stream.next().await {
                if crossterm_tx.send(ev).await.is_err() {
                    break;
                }
            }
        });

        Ok(Self {
            terminal,
            state,
            ui_rx,
            crossterm_rx,
            _crossterm_task: crossterm_task,
            tick_interval: tokio::time::interval(std::time::Duration::from_millis(
                constants::TICK_INTERVAL_MS,
            )),
            available_models,
            last_redraw: Instant::now() - MIN_REDRAW_GAP,
            pending_action: None,
        })
    }

    /// Draw the current state. Errors are logged and swallowed — TUI
    /// rendering failures shouldn't tear down the session.
    fn redraw(&mut self) {
        if let Err(e) = self.terminal.draw(|f| self.state.render(f)) {
            tracing::warn!(error = %e, "TUI redraw failed");
        }
        self.last_redraw = Instant::now();
    }

    /// Redraw only if the previous draw was more than [`MIN_REDRAW_GAP`]
    /// ago. Used during streaming where `render_event` may fire many
    /// times per second.
    fn maybe_redraw(&mut self) {
        if self.last_redraw.elapsed() >= MIN_REDRAW_GAP {
            self.redraw();
        }
    }

    /// Translate a `UiMessage` into a `UserInput`. Some messages need
    /// translation (e.g. `ChangeModel(idx)` becomes a `/model <id>`
    /// command); others map directly.
    fn ui_message_to_user_input(&self, msg: UiMessage) -> Option<UserInput> {
        match msg {
            UiMessage::Submit(s) => Some(UserInput::Prompt(s)),
            UiMessage::Command(c) => Some(UserInput::Command(c)),
            UiMessage::Quit => Some(UserInput::Quit),
            UiMessage::Abort => Some(UserInput::Abort),
            UiMessage::Clear => Some(UserInput::Command("/clear".into())),
            UiMessage::ChangeModel(idx) => self
                .available_models
                .get(idx)
                .map(|m| UserInput::Command(format!("/model {}", m.id))),
            UiMessage::Branch(idx) => Some(UserInput::Command(match idx {
                Some(i) => format!("/branch {}", i),
                None => "/branch".into(),
            })),
        }
    }

    /// Plan-modal key handler. Intercepts key events when a
    /// `PendingPlan` is active. Two modes:
    /// * `Reviewing` — A/E/R/Esc + scroll keys.
    /// * `EnteringReason` — input box captures the rejection reason
    ///   text. `Action::Submit` finalizes; `Action::Escape` returns to
    ///   review.
    ///
    /// Returns `true` if the key was consumed by the modal.
    fn handle_plan_modal_key(&mut self, key: crossterm::event::KeyEvent, area_width: u16) -> bool {
        let Some(mode) = self.state.pending_plan.as_ref().map(|p| match p.mode {
            PlanModalMode::Reviewing => 0,
            PlanModalMode::EnteringReason => 1,
        }) else {
            return false;
        };

        let action = key_to_action(key);
        if mode == 0 {
            // Reviewing.
            match action {
                Action::Char('a') | Action::Char('A') => {
                    if let Some(pp) = self.state.pending_plan.take() {
                        let _ = pp.response_tx.send(InteractionResponse::Approved { payload: None });
                    }
                }
                Action::Char('e') | Action::Char('E') => {
                    if let Some(pp) = self.state.pending_plan.take() {
                        let _ = pp.response_tx.send(InteractionResponse::Approved { payload: None });
                        self.pending_action = Some(FrontendAction::ExecutePlanNow);
                    }
                }
                Action::Char('r') | Action::Char('R') => {
                    if let Some(pp) = self.state.pending_plan.as_mut() {
                        pp.mode = PlanModalMode::EnteringReason;
                        self.state.input.clear();
                    }
                }
                Action::Escape | Action::Interrupt => {
                    if let Some(pp) = self.state.pending_plan.take() {
                        let _ = pp.response_tx.send(InteractionResponse::Cancelled);
                    }
                }
                Action::Up => {
                    if let Some(pp) = self.state.pending_plan.as_mut() {
                        pp.scroll = pp.scroll.saturating_sub(1);
                    }
                }
                Action::Down => {
                    if let Some(pp) = self.state.pending_plan.as_mut() {
                        pp.scroll = pp.scroll.saturating_add(1);
                    }
                }
                Action::PageUp => {
                    if let Some(pp) = self.state.pending_plan.as_mut() {
                        pp.scroll = pp.scroll.saturating_sub(10);
                    }
                }
                Action::PageDown => {
                    if let Some(pp) = self.state.pending_plan.as_mut() {
                        pp.scroll = pp.scroll.saturating_add(10);
                    }
                }
                _ => {}
            }
            true
        } else {
            // EnteringReason: the input box captures text. Submit fires
            // Rejected; Escape returns to Reviewing.
            match action {
                Action::Submit => {
                    let reason = self.state.input.content().to_string();
                    let reason = if reason.trim().is_empty() {
                        "User rejected the plan.".to_string()
                    } else {
                        reason
                    };
                    self.state.input.clear();
                    if let Some(pp) = self.state.pending_plan.take() {
                        let _ = pp.response_tx.send(InteractionResponse::Rejected { reason });
                    }
                }
                Action::Escape | Action::Interrupt => {
                    if let Some(pp) = self.state.pending_plan.as_mut() {
                        pp.mode = PlanModalMode::Reviewing;
                    }
                    self.state.input.clear();
                }
                _ => {
                    self.state.input.handle_action(&action, area_width);
                }
            }
            true
        }
    }

    /// Handle a crossterm event while a prompt is executing. Returns
    /// `Some` for events the Session needs to react to (Abort, Steer,
    /// Quit). Other events mutate `self.state` directly.
    fn handle_processing_event(&mut self, event: Event, area_width: u16) -> Option<UserInput> {
        // Plan modal short-circuits all other handling.
        if let Event::Key(key) = event
            && self.state.pending_plan.is_some()
            && self.handle_plan_modal_key(key, area_width)
        {
            return None;
        }
        match event {
            Event::Key(key) if self.state.pending_interaction.is_some() => {
                let action = key_to_action(key);
                match action {
                    Action::Up => {
                        if let Some(pi) = self.state.pending_interaction.as_mut() {
                            pi.selector.up(pi.options.len());
                        }
                    }
                    Action::Down => {
                        if let Some(pi) = self.state.pending_interaction.as_mut() {
                            pi.selector.down(pi.options.len());
                        }
                    }
                    Action::Submit => {
                        if let Some(pi) = self.state.pending_interaction.take() {
                            let label = pi.options[pi.selector.selected].label.clone();
                            let _ = pi.response_tx.send(InteractionResponse::Answer(label));
                            self.state.status = "Thinking...".to_string();
                        }
                    }
                    Action::Escape | Action::Interrupt => {
                        if let Some(pi) = self.state.pending_interaction.take() {
                            let _ = pi.response_tx.send(InteractionResponse::Cancelled);
                            self.state.status = "Thinking...".to_string();
                        }
                    }
                    _ => {}
                }
                None
            }
            Event::Key(key) => {
                let action = key_to_action(key);
                match action {
                    Action::Interrupt | Action::Escape => {
                        self.state.status = "Cancelling...".to_string();
                        Some(UserInput::Abort)
                    }
                    Action::Quit => Some(UserInput::Quit),
                    Action::Submit => {
                        let content = self.state.input.content().to_string();
                        if content.is_empty() {
                            return None;
                        }
                        self.state.input.clear();
                        self.state.messages.push(ChatMessage {
                            role: "steer".to_string(),
                            content: content.clone(),
                            is_error: false,
                            is_streaming: false,
                            id: None,
                        });
                        self.state.scroll_to_bottom();
                        Some(UserInput::Steer(content))
                    }
                    _ => {
                        self.state.input.handle_action(&action, area_width);
                        None
                    }
                }
            }
            Event::Paste(text) => {
                self.state.input.handle_action(&Action::Paste(text), area_width);
                None
            }
            Event::Mouse(_) | Event::Resize(_, _) => None,
            _ => None,
        }
    }
}

#[async_trait]
impl Frontend for TuiFrontend {
    async fn on_session_start(&mut self, _info: SessionStart<'_>) {
        self.redraw();
    }

    async fn on_session_end(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture,
            DisableBracketedPaste
        );
        let _ = self.terminal.show_cursor();
    }

    async fn next_input(&mut self) -> Option<UserInput> {
        // Idle state — Session is awaiting our input. Drive the redraw
        // and crossterm loops until a UiMessage emerges.
        loop {
            self.redraw();
            let area_width = self.terminal.size().map(|s| s.width).unwrap_or(80);

            tokio::select! {
                msg = self.ui_rx.recv() => {
                    match msg {
                        Some(m) => {
                            if let Some(input) = self.ui_message_to_user_input(m) {
                                return Some(input);
                            }
                        }
                        None => return None,
                    }
                }
                event = self.crossterm_rx.recv() => {
                    match event {
                        Some(ev) => {
                            // handle_event_while_idle returns false on
                            // Quit; in that case the action handler has
                            // already pushed UiMessage::Quit to ui_tx,
                            // which we'll see on the next iteration.
                            let _ = self.state.handle_event_while_idle(ev, area_width).await;
                        }
                        None => return None,
                    }
                }
                _ = self.tick_interval.tick() => {
                    self.state.git_branch.poll();
                    self.state.git_branch.maybe_refresh();
                }
            }
        }
    }

    async fn render_event(&mut self, event: AgentEvent) {
        // Translate into state mutation via the existing handler.
        // is_processing is owned by Session conceptually; the state's
        // is_processing flag controls the spinner / status row.
        self.state.is_processing = true;
        self.state.handle_agent_event(event);
        self.maybe_redraw();
    }

    async fn render_fleet_event(&mut self, event: tau_agent::FleetEvent) {
        self.state.handle_fleet_event(event);
        self.maybe_redraw();
    }

    async fn render_turn_end(&mut self, _total_usage: &Usage, _model: &Model) {
        self.state.is_processing = false;
        self.state.status = "Ready".to_string();
        self.redraw();
    }

    async fn show_system(&mut self, text: &str) {
        self.state.show_system_message(text);
        self.redraw();
    }

    async fn show_error(&mut self, text: &str) {
        self.state.show_system_message(&format!("Error: {}", text));
        self.redraw();
    }

    async fn handle_interaction(&mut self, req: InteractionRequest) {
        match req.kind {
            InteractionKind::AskQuestion { question, options } => {
                self.state.status = "Waiting for your choice...".to_string();
                self.state.pending_interaction = Some(PendingInteraction {
                    question,
                    options,
                    response_tx: req.response_tx,
                    selector: SelectorState::default(),
                });
                self.redraw();
            }
            InteractionKind::Typed { schema_id, payload } => match schema_id.as_str() {
                "tool.confirm" => {
                    // Should not fire: TuiFrontend reports
                    // can_render_approval() == false, so Session
                    // installs AutoAcceptAll.
                    let tool_name = payload
                        .get("tool_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?");
                    self.state.status = format!("Unexpected tool.confirm for {tool_name}");
                    let _ = req.response_tx.send(InteractionResponse::Rejected {
                        reason: "TUI confirm not implemented".into(),
                    });
                    self.redraw();
                }
                "plan.submit" => {
                    match serde_json::from_value::<tau_tools::Plan>(payload) {
                        Ok(plan) => {
                            let step_count = plan.items.len();
                            self.state.status = format!(
                                "Plan submitted ({step_count} step(s)) — awaiting your review"
                            );
                            self.state.pending_plan = Some(PendingPlan {
                                plan,
                                response_tx: req.response_tx,
                                mode: PlanModalMode::Reviewing,
                                scroll: 0,
                            });
                            self.redraw();
                        }
                        Err(e) => {
                            self.state.status =
                                format!("Plan submission failed to parse: {e}");
                            let _ = req.response_tx.send(InteractionResponse::Rejected {
                                reason: format!("Plan failed to deserialize: {e}"),
                            });
                            self.redraw();
                        }
                    }
                }
                other => {
                    self.state.status = format!("Unknown typed interaction: {other}");
                    let _ = req.response_tx.send(InteractionResponse::Rejected {
                        reason: format!("Unknown schema: {other}"),
                    });
                    self.redraw();
                }
            },
        }
    }

    fn can_render_approval(&self) -> bool {
        false
    }

    fn take_action(&mut self) -> Option<FrontendAction> {
        self.pending_action.take()
    }

    async fn on_config_change(&mut self, config: &tau_agent::AgentConfig) {
        self.state.sync_from_config(config);
        self.redraw();
    }

    async fn open_branch_selector(&mut self, _messages: &[tau_ai::Message]) -> bool {
        self.state.open_branch_selector();
        self.redraw();
        true
    }

    async fn tick(&mut self) -> Option<UserInput> {
        // During an in-flight turn, drive one frame and check for
        // crossterm input. Returns `Some` when the Session needs to act
        // (Abort, Steer, Quit) and `None` on every regular tick.
        let area_width = self.terminal.size().map(|s| s.width).unwrap_or(80);

        // Non-blocking drain: process all buffered key events first.
        // This is the load-bearing change for keyboard responsiveness
        // during agent streaming — keys never get stuck waiting for the
        // outer select to round-robin back to us.
        while let Ok(ev) = self.crossterm_rx.try_recv() {
            if let Some(input) = self.handle_processing_event(ev, area_width) {
                self.maybe_redraw();
                return Some(input);
            }
        }
        // Also drain any pending ui_rx messages (Quit/Abort from the
        // idle-mode action handler) so we respect Ctrl-Q / Ctrl-C
        // during a turn.
        while let Ok(msg) = self.ui_rx.try_recv() {
            match msg {
                UiMessage::Quit => return Some(UserInput::Quit),
                UiMessage::Abort => return Some(UserInput::Abort),
                _ => {} // idle-only
            }
        }

        self.maybe_redraw();

        // Bound the tick to at most one tick_interval period. Tokio's
        // `select!` cancels other branches when one wins; if a key
        // arrives during this wait, we return immediately.
        tokio::select! {
            _ = self.tick_interval.tick() => {
                self.state.git_branch.poll();
                self.state.git_branch.maybe_refresh();
                None
            }
            event = self.crossterm_rx.recv() => {
                match event {
                    Some(ev) => self.handle_processing_event(ev, area_width),
                    None => None,
                }
            }
            msg = self.ui_rx.recv() => {
                match msg {
                    Some(UiMessage::Quit) => Some(UserInput::Quit),
                    Some(UiMessage::Abort) => Some(UserInput::Abort),
                    Some(_) | None => None,
                }
            }
        }
    }

    async fn reset_view(&mut self) {
        self.state.messages.clear();
        self.state.reset_stats();
        self.state.scroll = 0;
        self.state.follow_bottom = true;
        self.state.status = "Ready".to_string();
        self.redraw();
    }
}
