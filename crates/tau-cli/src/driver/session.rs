//! The [`Session`] state machine + main driver loop.

use std::sync::Arc;

use tau_agent::{
    AgentEvent, AgentHandle, AgentManager, AutoAcceptAll, FleetEvent,
    InteractionRequest, SpawnOpts,
};
use tau_ai::{Message, Model, Usage};
use tokio::sync::broadcast;
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::mpsc;

use crate::commands;
use crate::session::SessionManager;

use super::frontend::{Frontend, FrontendAction, SessionStart, UserInput};

/// Live host-side state. The agent's own state lives in the actor.
enum State {
    Idle,
    InPlanMode {
        plan_handle: AgentHandle,
        plan_agent_id: String,
        /// `true` once the planner has called `submit_plan` and the
        /// user has accepted it for review. Free-form prompts now
        /// route to the main agent again; only `/plan approve` and
        /// `/plan exit` still consult the plan_handle.
        plan_ready: bool,
    },
}

/// Static configuration handed to [`Session::new`].
pub struct SessionConfig {
    pub handle: AgentHandle,
    pub manager: Arc<AgentManager>,
    pub spec_resolver: tau_tools::SpecResolver,
    pub interaction_rx: mpsc::Receiver<InteractionRequest>,
    pub available_models: Vec<Model>,
    /// Optional persistence sink for the main conversation. Plan-mode
    /// traffic does not write here.
    pub persistence: Option<SessionManager>,
}

pub struct Session {
    handle: AgentHandle,
    manager: Arc<AgentManager>,
    spec_resolver: tau_tools::SpecResolver,
    interaction_rx: mpsc::Receiver<InteractionRequest>,
    /// Long-lived fleet-event subscription. Subscribed once at session
    /// construction (not per-prompt) so subagent lifecycle/forwarded
    /// events that fire while the user is idle — notably a background
    /// agent finishing between turns — still reach the frontend. Drained
    /// by both the idle loop and `submit_prompt`.
    fleet_events: broadcast::Receiver<FleetEvent>,
    available_models: Vec<Model>,
    persistence: Option<SessionManager>,
    prev_usage: Usage,
    state: State,
    /// Set by a command (typically `/quit`) to exit the driver loop on
    /// the next iteration.
    exit_requested: bool,
    /// Whether the user has already been warned about a session-log
    /// write failure (warn once, log every failure).
    warned_persistence: bool,
    /// Whether any prompt in this session failed with an agent error.
    /// One-shot mode turns this into a non-zero exit code.
    had_agent_error: bool,
    /// Tools the user chose "always allow" for in the approval modal.
    /// Applied as a [`SessionAllowlistPolicy`] on the root agent and
    /// the manager (so subagents inherit it).
    always_allowed_tools: std::collections::HashSet<String>,
}

/// [`DefaultPolicy`] plus a session allowlist: tools the user approved
/// with "always allow" skip the gate; everything else keeps the
/// default risk-based gating.
struct SessionAllowlistPolicy {
    allowed: std::collections::HashSet<String>,
}

impl tau_agent::ApprovalPolicy for SessionAllowlistPolicy {
    fn classify(
        &self,
        tool: &str,
        arguments: &serde_json::Value,
        risk: tau_agent::ToolRisk,
    ) -> tau_agent::ApprovalDecision {
        if self.allowed.contains(tool) {
            return tau_agent::ApprovalDecision::Auto;
        }
        tau_agent::ApprovalPolicy::classify(&tau_agent::DefaultPolicy, tool, arguments, risk)
    }
}

impl Session {
    pub fn new(cfg: SessionConfig) -> Self {
        // Subscribe before `cfg.manager` is moved into the struct, and
        // before any prompt, so no fleet event is missed from session
        // start onward.
        let fleet_events = cfg.manager.subscribe();
        Self {
            handle: cfg.handle,
            manager: cfg.manager,
            spec_resolver: cfg.spec_resolver,
            interaction_rx: cfg.interaction_rx,
            fleet_events,
            available_models: cfg.available_models,
            persistence: cfg.persistence,
            prev_usage: Usage::default(),
            state: State::Idle,
            exit_requested: false,
            warned_persistence: false,
            had_agent_error: false,
            always_allowed_tools: std::collections::HashSet::new(),
        }
    }

    /// Whether any prompt failed with an agent error during this
    /// session. `tau run` maps this to exit code 1.
    pub fn had_agent_error(&self) -> bool {
        self.had_agent_error
    }

    // ─── Accessors for commands ───────────────────────────────────────

    /// Current model list the host knows about. Used by `/model`.
    pub(crate) fn available_models(&self) -> &[Model] {
        &self.available_models
    }

    /// Whether the session is currently driving a plan subagent.
    pub(crate) fn is_plan_mode(&self) -> bool {
        matches!(self.state, State::InPlanMode { .. })
    }

    /// Snapshot of the effective handle's config. `None` if the agent
    /// has shut down.
    pub(crate) async fn current_config(&self) -> Option<tau_agent::AgentConfig> {
        self.effective_handle().config().await
    }

    /// Snapshot of the effective handle's messages.
    pub(crate) async fn current_messages(&self) -> Vec<Message> {
        self.effective_handle().messages().await.unwrap_or_default()
    }

    /// Snapshot of the effective handle's total usage.
    pub(crate) async fn current_usage(&self) -> Usage {
        self.effective_handle()
            .state()
            .await
            .map(|s| s.total_usage)
            .unwrap_or_default()
    }

    /// Request the driver loop to exit after the current command.
    pub(crate) fn request_exit(&mut self) {
        self.exit_requested = true;
    }

    // ─── Mutating ops invoked by commands ─────────────────────────────

    pub(crate) async fn change_model(
        &mut self,
        new_model: tau_ai::Model,
        frontend: &mut dyn Frontend,
    ) {
        frontend
            .show_system(&format!(
                "Switched to: {} ({})",
                new_model.id,
                new_model.provider.name()
            ))
            .await;
        let handle = self.effective_handle().clone();
        let _ = handle.set_model(new_model).await;
        if let Some(cfg) = handle.config().await {
            frontend.on_config_change(&cfg).await;
        }
    }

    pub(crate) async fn change_reasoning(
        &mut self,
        level: tau_ai::ReasoningLevel,
        frontend: &mut dyn Frontend,
    ) {
        frontend
            .show_system(&format!("Reasoning level set to: {:?}", level))
            .await;
        let handle = self.effective_handle().clone();
        let _ = handle.set_reasoning(level).await;
        if let Some(cfg) = handle.config().await {
            frontend.on_config_change(&cfg).await;
        }
    }

    /// Drive the session against a frontend until the user quits or
    /// EOF is reached.
    pub async fn drive(&mut self, frontend: &mut dyn Frontend) -> anyhow::Result<()> {
        // If the frontend can't render approval prompts, swap the
        // policy on both the root agent and the manager so subagents
        // inherit it.
        if !frontend.can_render_approval() {
            let auto = Arc::new(AutoAcceptAll);
            self.handle.set_approval_policy(auto.clone()).await?;
            self.manager.set_default_approval_policy(auto);
        }

        // Banner.
        let config = self
            .handle
            .config()
            .await
            .ok_or_else(|| anyhow::anyhow!("Agent shut down"))?;
        let session_id = self.persistence.as_ref().map(|s| &s.id()[..8]);
        frontend
            .on_session_start(SessionStart {
                model: config.model(),
                session_id,
            })
            .await;

        loop {
            // Idle: wait for either the next user input or a tool
            // interaction. Interactions can fire even between turns if
            // a background subagent runs.
            // In idle, the frontend's `next_input` is expected to
            // drive its own tick / redraw loop (the TUI does this; the
            // stdout frontend just blocks on stdin). Interaction
            // requests that fire between turns are dispatched as a
            // race.
            let input = tokio::select! {
                input = frontend.next_input() => input,
                Some(req) = self.interaction_rx.recv() => {
                    frontend.handle_interaction(req).await;
                    continue;
                }
                fleet_ev = self.fleet_events.recv() => {
                    // Background subagents can emit lifecycle/forwarded
                    // events while we're sitting idle between prompts;
                    // render them so the agent tree stays current.
                    match fleet_ev {
                        Ok(event) => frontend.render_fleet_event(event).await,
                        Err(RecvError::Closed) => {}
                        Err(RecvError::Lagged(n)) => {
                            tracing::warn!(dropped = n, "fleet event stream lagged (idle)");
                        }
                    }
                    continue;
                }
            };

            let Some(input) = input else { break };
            match input {
                UserInput::Quit => break,
                UserInput::Abort => {
                    self.effective_handle().abort();
                }
                UserInput::Steer(_) => {
                    // Steer arrives only during a turn (from tick); ignore if it leaks
                    // to idle.
                }
                UserInput::Prompt(text) => {
                    self.submit_prompt(&text, frontend).await?;
                    self.drain_frontend_action(frontend).await?;
                }
                UserInput::Command(line) => {
                    self.handle_command(&line, frontend).await;
                    if self.exit_requested {
                        break;
                    }
                    self.drain_frontend_action(frontend).await?;
                }
            }
        }

        frontend.on_session_end().await;
        Ok(())
    }

    /// The handle currently receiving prompts: the plan agent while
    /// it's actively drafting, otherwise the main agent.
    fn effective_handle(&self) -> &AgentHandle {
        match &self.state {
            State::Idle => &self.handle,
            // Once a plan is accepted for review, follow-up prompts
            // shouldn't reawaken the planner — they go to the main
            // agent. The plan_handle is still reachable via
            // `/plan approve` / `/plan exit`.
            State::InPlanMode { plan_handle, plan_ready, .. } => {
                if *plan_ready {
                    &self.handle
                } else {
                    plan_handle
                }
            }
        }
    }

    /// Send `prompt` to the effective handle and pump events through
    /// `frontend` until the turn ends. When in main-agent mode, also
    /// appends the new messages + usage delta to the persistence sink.
    async fn submit_prompt(
        &mut self,
        prompt: &str,
        frontend: &mut dyn Frontend,
    ) -> anyhow::Result<()> {
        let handle = self.effective_handle().clone();
        let config = handle
            .config()
            .await
            .ok_or_else(|| anyhow::anyhow!("Agent shut down"))?;
        let model = config.model().clone();

        // Subscribe to the root agent's own stream per-prompt (it only
        // emits during its own turn). Fleet events use the session-long
        // `self.fleet_events` so nothing emitted before this prompt — or
        // by a still-running background agent — is missed.
        let mut events = handle.subscribe();
        let msgs_before = handle.messages().await.map(|m| m.len()).unwrap_or(0);
        // Compaction sentinel: if `previous_summary` changes across the
        // prompt, in-memory history was rewritten and the append-only
        // message diff below would be invalid.
        let summary_before = handle.state().await.and_then(|s| s.previous_summary);

        // Fire-and-await the prompt. The actor will start producing
        // events; we drain them in this same task via select!.
        let prompt_owned = prompt.to_string();
        let handle_for_task = handle.clone();
        let mut prompt_task =
            tokio::spawn(async move { handle_for_task.prompt_and_wait(&prompt_owned).await });

        // Cumulative-text tracker for MessageEnd: not needed — frontend
        // handles its own delta state via render_event.
        let mut total_usage = Usage::default();
        // Captured prompt outcome. The loop can exit via the AgentEnd
        // event *before* `prompt_task` is polled to completion, so the
        // result is also collected after the loop — otherwise a prompt
        // error is silently dropped (and `tau run` exits 0 on failure).
        let mut prompt_res: Option<Result<(), tau_agent::Error>> = None;
        loop {
            // No `biased;`: fair polling so frontend.tick() (and the
            // input it carries) doesn't starve when agent events stream
            // continuously.
            tokio::select! {
                ev = events.recv() => match ev {
                    Ok(AgentEvent::AgentEnd { total_usage: u, interrupted, .. }) => {
                        total_usage = u.clone();
                        frontend.render_event(AgentEvent::AgentEnd {
                            total_usage: u,
                            total_turns: 0,
                            interrupted,
                        }).await;
                        break;
                    }
                    Ok(event) => frontend.render_event(event).await,
                    Err(RecvError::Closed) => break,
                    Err(RecvError::Lagged(n)) => {
                        tracing::warn!(dropped = n, "session event stream lagged");
                    }
                },
                fleet_ev = self.fleet_events.recv() => match fleet_ev {
                    Ok(event) => frontend.render_fleet_event(event).await,
                    Err(RecvError::Closed) => {}
                    Err(RecvError::Lagged(n)) => {
                        tracing::warn!(dropped = n, "fleet event stream lagged");
                    }
                },
                Some(req) = self.interaction_rx.recv() => {
                    frontend.handle_interaction(req).await;
                }
                tick_input = frontend.tick() => {
                    // tick may return None (just a frame); only act on
                    // input signals.
                    match tick_input {
                        Some(UserInput::Abort) => handle.abort(),
                        Some(UserInput::Steer(msg)) => {
                            let _ = handle.try_steer(tau_ai::Message::user(&msg));
                        }
                        Some(UserInput::Quit) => {
                            handle.abort();
                            break;
                        }
                        _ => {}
                    }
                }
                res = &mut prompt_task => {
                    // Prompt completed but we may still have events
                    // queued. Drain them non-blockingly.
                    prompt_res = Some(res.unwrap_or(Ok(())));
                    while let Ok(event) = events.try_recv() {
                        let is_end = matches!(event, AgentEvent::AgentEnd { .. });
                        if let AgentEvent::AgentEnd { total_usage: u, .. } = &event {
                            total_usage = u.clone();
                        }
                        frontend.render_event(event).await;
                        if is_end { break; }
                    }
                    while let Ok(event) = self.fleet_events.try_recv() {
                        frontend.render_fleet_event(event).await;
                    }
                    break;
                }
            }
        }

        // Collect the prompt outcome if the loop exited via the
        // AgentEnd event (or Quit) before polling the task. The actor
        // has already finished (or been aborted), so this resolves
        // promptly; the timeout is a backstop against a wedged actor.
        let res = match prompt_res {
            Some(r) => r,
            None => tokio::time::timeout(std::time::Duration::from_secs(2), &mut prompt_task)
                .await
                .map(|join| join.unwrap_or(Ok(())))
                .unwrap_or_else(|_elapsed| {
                    prompt_task.abort();
                    Ok(())
                }),
        };
        if let Err(e) = res {
            self.had_agent_error = true;
            frontend.show_error(&format!("{}", e)).await;
        }

        frontend.render_turn_end(&total_usage, &model).await;

        // Persist new messages + usage delta when running against the
        // main agent (not plan mode).
        let mut persist_err: Option<std::io::Error> = None;
        if matches!(self.state, State::Idle)
            && let Some(persist) = self.persistence.as_mut()
        {
            let all_msgs = self.handle.messages().await.unwrap_or_default();
            let state = self.handle.state().await;
            let summary_after = state.as_ref().and_then(|s| s.previous_summary.clone());
            if summary_after != summary_before
                && let Some(summary) = &summary_after
            {
                // A compaction rewrote in-memory history this prompt;
                // the index-based diff below would log the wrong
                // messages. Snapshot instead: compaction marker + the
                // kept tail (everything after the synthetic summary
                // head).
                if let Err(e) =
                    persist.append_compaction_snapshot(summary, all_msgs.get(1..).unwrap_or(&[]))
                {
                    persist_err.get_or_insert(e);
                }
            } else {
                for msg in all_msgs.iter().skip(msgs_before) {
                    if let Err(e) = persist.append_message(msg) {
                        persist_err.get_or_insert(e);
                    }
                }
            }
            if let Some(s) = state {
                let delta = Usage {
                    input: s.total_usage.input.saturating_sub(self.prev_usage.input),
                    output: s.total_usage.output.saturating_sub(self.prev_usage.output),
                    cache_read: s
                        .total_usage
                        .cache_read
                        .saturating_sub(self.prev_usage.cache_read),
                    cache_write: s
                        .total_usage
                        .cache_write
                        .saturating_sub(self.prev_usage.cache_write),
                    thinking: s
                        .total_usage
                        .thinking
                        .saturating_sub(self.prev_usage.thinking),
                    ..Default::default()
                };
                if let Err(e) = persist.append_usage(&delta) {
                    persist_err.get_or_insert(e);
                }
                self.prev_usage = s.total_usage.clone();
            }
        }
        if let Some(e) = persist_err {
            self.report_persist_error(frontend, &e).await;
        }

        Ok(())
    }

    /// Report a session-log write failure: every failure is traced,
    /// but the user is warned only once per session to avoid spamming
    /// the frontend on a persistently failing disk.
    async fn report_persist_error(&mut self, frontend: &mut dyn Frontend, e: &std::io::Error) {
        tracing::error!("session log write failed: {e}");
        if !self.warned_persistence {
            self.warned_persistence = true;
            frontend
                .show_error(&format!(
                    "Session log write failed: {e}. History may be incomplete if this \
                     session is resumed; further failures will only be logged."
                ))
                .await;
        }
    }

    /// Drain any side-channel action the frontend produced during the
    /// turn (e.g. "Execute now" from the plan-review modal) and act on
    /// it. Called after every command / prompt cycle.
    async fn drain_frontend_action(
        &mut self,
        frontend: &mut dyn Frontend,
    ) -> anyhow::Result<()> {
        while let Some(action) = frontend.take_action() {
            match action {
                FrontendAction::ExecutePlanNow => {
                    if matches!(self.state, State::InPlanMode { .. }) {
                        self.approve_plan(frontend).await?;
                    } else {
                        frontend
                            .show_system("Execute now is only available within /plan mode.")
                            .await;
                    }
                }
                FrontendAction::AlwaysAllowTool(tool_name) => {
                    self.always_allowed_tools.insert(tool_name.clone());
                    let policy = Arc::new(SessionAllowlistPolicy {
                        allowed: self.always_allowed_tools.clone(),
                    });
                    if self.handle.set_approval_policy(policy.clone()).await.is_ok() {
                        self.manager.set_default_approval_policy(policy);
                        frontend
                            .show_system(&format!(
                                "Auto-approving `{tool_name}` for the rest of this session."
                            ))
                            .await;
                    }
                }
            }
        }
        Ok(())
    }

    /// Parse the leading slash command and dispatch.
    async fn handle_command(&mut self, line: &str, frontend: &mut dyn Frontend) {
        let line = line.trim();
        let line = line.strip_prefix('/').unwrap_or(line);
        let (cmd_name, args) = match line.split_once(' ') {
            Some((n, a)) => (n.to_ascii_lowercase(), a.trim()),
            None => (line.to_ascii_lowercase(), ""),
        };
        let commands = commands::all_commands();
        let matched = commands
            .iter()
            .find(|c| c.name() == cmd_name || c.aliases().contains(&cmd_name.as_str()));
        match matched {
            Some(cmd) => cmd.execute(args, self, frontend).await,
            None => {
                frontend
                    .show_system(&format!(
                        "Unknown command: /{}\nType /help for available commands.",
                        cmd_name
                    ))
                    .await;
            }
        }
    }

    pub(crate) async fn clear(&mut self, frontend: &mut dyn Frontend) {
        let old_id = self.handle.agent_id().map(str::to_string);
        let Some(spec) = old_id.as_deref().and_then(|id| self.manager.spec_for(id)) else {
            frontend
                .show_error("/clear unavailable: parent agent has no recorded spec")
                .await;
            return;
        };
        self.handle.abort();
        let opts = SpawnOpts {
            description: "main".into(),
            ..Default::default()
        };
        match self.manager.spawn_interactive(spec, opts).await {
            Ok((new_handle, _)) => {
                if let Some(id) = old_id {
                    self.manager.remove_interactive(&id);
                }
                self.handle = new_handle;
                self.prev_usage = Usage::default();
                frontend.reset_view().await;
                frontend.show_system("Cleared conversation.").await;
            }
            Err(e) => frontend.show_error(&format!("/clear failed: {e}")).await,
        }
    }

    pub(crate) async fn compact(&mut self, frontend: &mut dyn Frontend) {
        frontend.show_system("Compacting context...").await;
        let handle = self.effective_handle().clone();
        match handle.compact(None).await {
            Ok(rx) => match rx.await {
                Ok(r) if r.result.is_ok() => {
                    let msgs = handle.messages().await.unwrap_or_default();
                    // Mirror the compaction in the session log so a
                    // restore doesn't replay pre-compaction history.
                    let mut persist_err: Option<std::io::Error> = None;
                    if matches!(self.state, State::Idle)
                        && let Some(persist) = self.persistence.as_mut()
                        && let Some(summary) =
                            handle.state().await.and_then(|s| s.previous_summary)
                    {
                        if let Err(e) = persist
                            .append_compaction_snapshot(&summary, msgs.get(1..).unwrap_or(&[]))
                        {
                            persist_err = Some(e);
                        }
                    }
                    if let Some(e) = persist_err {
                        self.report_persist_error(frontend, &e).await;
                    }
                    frontend
                        .show_system(&format!(
                            "Context compacted. {} messages remaining.",
                            msgs.len()
                        ))
                        .await;
                }
                _ => frontend.show_error("Compaction failed.").await,
            },
            Err(e) => {
                frontend
                    .show_error(&format!("Compaction failed: {}", e))
                    .await
            }
        }
    }

    pub(crate) async fn enter_plan_mode(
        &mut self,
        description: String,
        frontend: &mut dyn Frontend,
    ) -> anyhow::Result<()> {
        let main_messages = self.handle.messages().await.unwrap_or_default();
        let main_state = self.handle.state().await;
        let prev_summary = main_state
            .as_ref()
            .and_then(|s| s.previous_summary.as_deref());
        let context = tau_tools::plan::build_context_summary(&main_messages, prev_summary);
        let prompt = tau_tools::plan::build_plan_prompt(&context, &description);

        let plan_spec = match (self.spec_resolver)("plan", 0) {
            Some(s) => s,
            None => {
                frontend
                    .show_error("Plan mode unavailable: 'plan' spec not registered.")
                    .await;
                return Ok(());
            }
        };
        let opts = SpawnOpts {
            description: format!("Planning: {}", description),
            ..Default::default()
        };

        match self.manager.spawn_interactive(plan_spec, opts).await {
            Ok((plan_handle, agent_id)) => {
                self.state = State::InPlanMode {
                    plan_handle,
                    plan_agent_id: agent_id,
                    plan_ready: false,
                };
                frontend
                    .show_system(&format!(
                        "Plan mode active: {}\nThe planner will draft a plan; you'll review it in a modal before anything executes.",
                        description
                    ))
                    .await;
                // Run the planning prompt immediately against the plan
                // handle. The Frontend reads the events as usual.
                self.submit_prompt(&prompt, frontend).await?;

                // If the planner successfully completed submit_plan,
                // surface the next-step hint and mark the plan ready so
                // free-form follow-up prompts route to the main agent
                // instead of reawakening the planner.
                if self.planner_finished_with_plan().await {
                    if let State::InPlanMode { plan_ready, .. } = &mut self.state {
                        *plan_ready = true;
                    }
                    frontend
                        .show_system(
                            "Planner finished. Use `/plan approve` to execute, `/plan exit` to discard, or just keep chatting — follow-ups now go to the main agent.",
                        )
                        .await;
                }
            }
            Err(e) => {
                frontend
                    .show_error(&format!("Failed to start plan mode: {}", e))
                    .await;
            }
        }
        Ok(())
    }

    /// Inspect the plan handle's messages for a successful `submit_plan`
    /// tool call. Used after `enter_plan_mode`'s submit_prompt returns
    /// to decide whether to flip `plan_ready`.
    async fn planner_finished_with_plan(&self) -> bool {
        let State::InPlanMode { plan_handle, .. } = &self.state else {
            return false;
        };
        let messages = plan_handle.messages().await.unwrap_or_default();
        // Walk messages in reverse looking for a tool_result for submit_plan
        // that isn't an error.
        messages.iter().rev().any(|msg| {
            matches!(msg, tau_ai::Message::ToolResult { content, is_error, .. }
                if !is_error
                    && content.iter().any(|c| matches!(c, tau_ai::Content::Text { text } if text.starts_with("Plan accepted for user review"))))
        })
    }

    pub(crate) async fn approve_plan(&mut self, frontend: &mut dyn Frontend) -> anyhow::Result<()> {
        let State::InPlanMode {
            plan_handle,
            plan_agent_id,
            plan_ready,
        } = std::mem::replace(&mut self.state, State::Idle)
        else {
            frontend.show_system("Not in plan mode.").await;
            return Ok(());
        };

        let plan_text = plan_handle
            .messages()
            .await
            .map(|m| tau_tools::plan::extract_final_text(&m))
            .unwrap_or_default();
        if plan_text.trim().is_empty() {
            frontend
                .show_system(
                    "Plan agent has no plan to approve yet. Wait for it to respond, or use /plan exit to discard.",
                )
                .await;
            // Restore plan-mode state.
            self.state = State::InPlanMode {
                plan_handle,
                plan_agent_id,
                plan_ready,
            };
            return Ok(());
        }

        let executor_spec = match (self.spec_resolver)("general-purpose:executor", 0) {
            Some(s) => s,
            None => {
                frontend
                    .show_error("Executor unavailable: 'general-purpose:executor' spec not registered.")
                    .await;
                self.manager.remove_interactive(&plan_agent_id);
                return Ok(());
            }
        };
        frontend
            .show_system("Plan approved. Executing inherited plan...")
            .await;

        let cancel = tokio_util::sync::CancellationToken::new();
        let opts = SpawnOpts {
            description: "Executor: approved plan".into(),
            seed: tau_agent::AgentSeed::Inherit {
                agent_id: plan_agent_id.clone(),
            },
            spec_name: Some("general-purpose:executor".into()),
            ..Default::default()
        };
        match self
            .manager
            .spawn(
                executor_spec,
                "Execute the approved plan.".to_string(),
                opts,
                cancel,
            )
            .await
        {
            Ok(result) => {
                frontend
                    .show_system(&format!("Executor finished: {}", result.text))
                    .await;
            }
            Err(e) => {
                frontend
                    .show_error(&format!("Executor failed: {e}"))
                    .await;
            }
        }
        self.manager.remove_interactive(&plan_agent_id);
        Ok(())
    }

    pub(crate) async fn exit_plan_mode(&mut self, frontend: &mut dyn Frontend) {
        let State::InPlanMode { plan_agent_id, .. } =
            std::mem::replace(&mut self.state, State::Idle)
        else {
            frontend.show_system("Not in plan mode.").await;
            return;
        };
        self.manager.remove_interactive(&plan_agent_id);
        frontend.show_system("Exited plan mode.").await;
    }

    pub(crate) async fn branch_from(
        &mut self,
        index: Option<usize>,
        frontend: &mut dyn Frontend,
    ) {
        let messages = self.handle.messages().await.unwrap_or_default();
        let messages = messages.as_slice();
        let model_id = match self.handle.config().await {
            Some(c) => c.model().id.clone(),
            None => {
                frontend.show_error("Agent shut down.").await;
                return;
            }
        };
        let new_session = match crate::session::branch::branch_from(messages, index, &model_id) {
            Ok(s) => s,
            Err(e) => {
                frontend
                    .show_error(&format!("Failed to create branch: {}", e))
                    .await;
                return;
            }
        };
        let msg_count = index.map(|i| i + 1).unwrap_or(0);
        frontend
            .show_system(&format!(
                "Created branch: {} ({} messages)",
                new_session.id(),
                msg_count
            ))
            .await;

        // Truncate the in-process conversation by respawning with an explicit seed.
        let old_id = self.handle.agent_id().map(str::to_string);
        let spec = old_id.as_deref().and_then(|id| self.manager.spec_for(id));
        let truncated: Vec<Message> = match index {
            Some(idx) => messages.iter().take(idx + 1).cloned().collect(),
            None => Vec::new(),
        };
        let Some(spec) = spec else {
            frontend
                .show_system(&format!(
                    "/branch unavailable in-process: parent agent has no recorded spec; restart with --resume {} instead.",
                    new_session.id()
                ))
                .await;
            self.persistence = Some(new_session);
            return;
        };
        self.handle.abort();
        let opts = SpawnOpts {
            description: "main".into(),
            seed: tau_agent::AgentSeed::Messages {
                messages: truncated,
                previous_summary: None,
            },
            ..Default::default()
        };
        match self.manager.spawn_interactive(spec, opts).await {
            Ok((new_handle, _)) => {
                if let Some(id) = old_id {
                    self.manager.remove_interactive(&id);
                }
                self.handle = new_handle;
                self.prev_usage = Usage::default();
                self.persistence = Some(new_session);
                frontend
                    .show_system(&format!(
                        "Continuing from branch point ({} message(s)).",
                        msg_count
                    ))
                    .await;
            }
            Err(e) => frontend.show_error(&format!("/branch failed: {e}")).await,
        }
    }
}

