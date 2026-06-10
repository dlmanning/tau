use std::collections::HashMap;

use super::widgets::message_list::ChatMessage;
use tau_agent::{AgentEvent, FleetEvent, SubagentOutcome};

use super::constants;
use super::state::{AgentProgress, TuiState};
use crate::utils::format_tokens;

/// Build a tree diagram of active/completed subagents.
fn build_agent_tree(
    agent_order: &[String],
    agent_progress: &HashMap<String, AgentProgress>,
) -> String {
    let agents: Vec<&AgentProgress> = agent_order
        .iter()
        .filter_map(|id| agent_progress.get(id))
        .collect();
    let total = agents.len();
    let mut lines = Vec::new();

    for (i, agent) in agents.iter().enumerate() {
        let (branch, cont) = if total == 1 {
            ("◇", " ")
        } else {
            let is_last = i == total - 1;
            if is_last {
                ("└─", "  ")
            } else {
                ("├─", "│ ")
            }
        };

        if agent.finished {
            let tokens = format_tokens(agent.input_tokens + agent.output_tokens);
            let indicator = if agent.activity.starts_with("error:") {
                "✗"
            } else {
                "✓"
            };
            lines.push(format!(
                "{} {} {} ({} tools · {} tokens)",
                branch, indicator, agent.description, agent.tool_count, tokens
            ));
        } else {
            lines.push(format!("{} ◇ {}", branch, agent.description));
            lines.push(format!("{}   ⚙ {}", cont, agent.activity));
        }
    }

    lines.join("\n")
}

impl TuiState {
    /// Handle agent events.
    pub fn handle_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::AgentStart => {
                self.is_processing = true;
            }
            AgentEvent::MessageUpdate { message } => {
                let text = message.text();
                if let Some(last) = self.messages.last_mut() {
                    if last.is_streaming {
                        last.content = text;
                        self.scroll_to_bottom();
                        return;
                    }
                }
                self.messages.push(ChatMessage::assistant_streaming(text));
                self.scroll_to_bottom();
            }
            AgentEvent::MessageEnd { message } => {
                if let Some(last) = self.messages.last_mut() {
                    if last.is_streaming {
                        last.content = message.text();
                        last.is_streaming = false;
                        return;
                    }
                }
                self.messages.push(ChatMessage::assistant(message.text()));
                self.scroll_to_bottom();
            }
            AgentEvent::ToolExecutionStart {
                tool_call_id,
                tool_name,
                activity,
                ..
            } => {
                self.messages.push(ChatMessage {
                    role: format!("tool:{}", tool_name),
                    content: activity,
                    is_error: false,
                    is_streaming: true,
                    id: Some(tool_call_id),
                });
                self.scroll_to_bottom();
            }
            AgentEvent::ToolExecutionUpdate {
                tool_call_id,
                lines,
                ..
            } => {
                if let Some(msg) = self
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|m| m.id.as_deref() == Some(&tool_call_id))
                {
                    // Show only the latest line as the in-progress activity
                    // until the TUI grows a proper styled console block.
                    if let Some(last) = lines.last() {
                        msg.content = last.content.clone();
                    }
                }
            }
            AgentEvent::ToolExecutionEnd {
                tool_call_id,
                tool_name,
                result,
                is_error,
                ..
            } => {
                let preview =
                    crate::utils::truncate_chars(&result, constants::TOOL_RESULT_PREVIEW_CHARS);
                if let Some(msg) = self
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|m| m.id.as_deref() == Some(&tool_call_id))
                {
                    msg.content = preview.to_string();
                    msg.is_streaming = false;
                    msg.is_error = is_error;
                } else {
                    self.messages
                        .push(ChatMessage::tool(&tool_name, preview, is_error));
                }
                self.scroll_to_bottom();
            }
            AgentEvent::TurnEnd { usage, .. } => {
                self.usage.accumulate(&usage, &self.model);
            }
            AgentEvent::AgentEnd { .. } => {
                self.is_processing = false;
                self.status = "Ready".to_string();
                self.finalize_agent_progress(false);
            }
            AgentEvent::Error { message } => {
                self.is_processing = false;
                self.status = "Ready".to_string();
                self.finalize_agent_progress(true);
                self.messages.push(ChatMessage {
                    role: "system".to_string(),
                    content: format!("Error: {}", message),
                    is_error: true,
                    is_streaming: false,
                    id: None,
                });
            }
            AgentEvent::CompactionStart { .. } => {}
            AgentEvent::CompactionEnd {
                tokens_before,
                tokens_after,
            } => {
                self.messages.push(ChatMessage::system(format!(
                    "Context compacted: ~{} -> ~{} tokens",
                    tokens_before, tokens_after
                )));
                self.scroll_to_bottom();
            }
            AgentEvent::TurnStart { .. }
            | AgentEvent::MessageStart { .. }
            | AgentEvent::ToolApprovalResolved { .. }
            | AgentEvent::FileChanged { .. }
            | AgentEvent::AgentReport { .. } => {}
        }
    }

    /// Handle fleet events (per-subagent lifecycle and forwarded child events).
    pub fn handle_fleet_event(&mut self, event: FleetEvent) {
        match event {
            FleetEvent::AgentStarted {
                agent_id,
                description,
                ..
            } => {
                let progress = AgentProgress::new(description);
                self.agent_progress.insert(agent_id.clone(), progress);
                self.agent_order.push(agent_id);
                self.update_agent_tree();
            }
            FleetEvent::AgentResumed { agent_id, description, .. } => {
                self.agent_progress
                    .entry(agent_id.clone())
                    .or_insert_with(|| {
                        self.agent_order.push(agent_id);
                        AgentProgress::new(description)
                    });
                self.update_agent_tree();
            }
            FleetEvent::AgentCompleted {
                agent_id, outcome, ..
            } => {
                // A subagent that fails at setup (e.g. worktree creation)
                // or is aborted emits AgentCompleted with a Failed/Aborted
                // outcome and no forwarded AgentEvent::Error, so the error
                // indicator must come from the outcome here — otherwise a
                // failed agent renders with a success ✓.
                let error_reason = match &outcome {
                    SubagentOutcome::Failed { reason } | SubagentOutcome::Aborted { reason } => {
                        Some(reason.clone())
                    }
                    SubagentOutcome::Completed => None,
                };
                if let Some(progress) = self.agent_progress.get_mut(&agent_id) {
                    progress.finished = true;
                    if let Some(reason) = &error_reason {
                        progress.activity = format!("error: {}", reason);
                    }
                }
                self.update_agent_tree();
                if self.agent_progress.values().all(|p| p.finished) {
                    let any_error = self
                        .agent_progress
                        .values()
                        .any(|p| p.activity.starts_with("error:"));
                    self.finalize_agent_progress(any_error);
                }
            }
            FleetEvent::AgentReport { .. } => {}
            FleetEvent::Forwarded {
                agent_id,
                event: inner,
                ..
            } => match inner {
                AgentEvent::ToolExecutionStart { activity, .. } => {
                    if let Some(progress) = self.agent_progress.get_mut(&agent_id) {
                        progress.tool_count += 1;
                        progress.activity = activity;
                    }
                    self.update_agent_tree();
                }
                AgentEvent::TurnEnd { usage, .. } => {
                    if let Some(progress) = self.agent_progress.get_mut(&agent_id) {
                        progress.input_tokens += usage.input;
                        progress.output_tokens += usage.output;
                    }
                    self.usage.accumulate(&usage, &self.model);
                }
                AgentEvent::Error { message } => {
                    if let Some(progress) = self.agent_progress.get_mut(&agent_id) {
                        progress.finished = true;
                        progress.activity = format!("error: {}", message);
                    }
                    self.update_agent_tree();
                    if self.agent_progress.values().all(|p| p.finished) {
                        self.finalize_agent_progress(true);
                    }
                }
                _ => {}
            },
        }
    }

    /// Finalize the agent tree message and clear tracking state — but
    /// only once every tracked agent has finished.
    ///
    /// A background subagent (spawned with `run_in_background: true`)
    /// outlives the prompt that launched it, so its `AgentCompleted`
    /// can arrive after the root agent's `AgentEnd`. Clearing the map
    /// here unconditionally would drop that completion — `get_mut`
    /// would miss it, so a failed bg agent would render as a silent
    /// success and `any_error` would be computed over the wrong set.
    /// We therefore defer the freeze+clear while any agent is still
    /// running; the deferred call fires from the `AgentCompleted` /
    /// forwarded-`Error` handlers when the last one finishes.
    fn finalize_agent_progress(&mut self, is_error: bool) {
        if self.agent_progress.is_empty() {
            return;
        }
        // Record the error state on the tree as soon as it's known, even
        // if a still-running bg agent means we can't freeze yet.
        if is_error {
            if let Some(msg) = self.messages.iter_mut().rev().find(|m| m.role == "agents") {
                msg.is_error = true;
            }
        }
        // Defer until nothing is left running so no completion is dropped
        // and finished agents stay visible in the tree.
        if !self.agent_progress.values().all(|p| p.finished) {
            return;
        }
        if let Some(msg) = self.messages.iter_mut().rev().find(|m| m.role == "agents") {
            msg.is_streaming = false;
        }
        self.agent_progress.clear();
        self.agent_order.clear();
    }

    /// Update or insert the agent tree message in the conversation.
    fn update_agent_tree(&mut self) {
        let tree = build_agent_tree(&self.agent_order, &self.agent_progress);
        if let Some(msg) = self
            .messages
            .iter_mut()
            .rev()
            .find(|m| m.role == "agents" && m.is_streaming)
        {
            msg.content = tree;
        } else {
            self.messages.push(ChatMessage {
                role: "agents".to_string(),
                content: tree,
                is_error: false,
                is_streaming: true,
                id: None,
            });
        }
        self.scroll_to_bottom();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tau_agent::test_utils::{make_test_config, make_test_model};

    fn test_state() -> TuiState {
        let (tx, _rx) = tokio::sync::mpsc::channel(64);
        TuiState::new(&make_test_config(), vec![make_test_model()], tx)
    }

    fn started(agent_id: &str) -> FleetEvent {
        FleetEvent::AgentStarted {
            agent_id: agent_id.into(),
            spec_name: None,
            description: format!("agent {agent_id}"),
            prompt: "go".into(),
            started_at: chrono::Utc::now(),
        }
    }

    fn completed(agent_id: &str, outcome: SubagentOutcome) -> FleetEvent {
        FleetEvent::AgentCompleted {
            agent_id: agent_id.into(),
            description: format!("agent {agent_id}"),
            outcome,
            started_at: chrono::Utc::now(),
            completed_at: chrono::Utc::now(),
            duration_ms: 0,
            usage: Default::default(),
            tool_use_count: 0,
            worktree_path: None,
            worktree_branch: None,
        }
    }

    fn agent_end() -> AgentEvent {
        AgentEvent::AgentEnd {
            total_turns: 1,
            total_usage: Default::default(),
            interrupted: false,
        }
    }

    /// A background subagent that completes *after* the root agent's
    /// `AgentEnd` must not be dropped: its failure has to surface rather
    /// than render as a silent success.
    #[test]
    fn background_failure_after_agent_end_is_not_lost() {
        let mut s = test_state();

        // A background subagent starts, then the root prompt ends while
        // it is still running.
        s.handle_fleet_event(started("bg"));
        s.handle_agent_event(agent_end());

        // It must still be tracked — finalize must not have cleared it.
        assert!(
            s.agent_progress.contains_key("bg"),
            "a still-running bg agent must survive the root AgentEnd"
        );

        // The bg agent later fails. This must be recorded, not dropped.
        s.handle_fleet_event(completed("bg", SubagentOutcome::Failed { reason: "boom".into() }));

        assert!(
            s.agent_progress.is_empty(),
            "tree finalizes once the last agent finishes"
        );
        let tree = s
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "agents")
            .expect("agents tree message");
        assert!(tree.is_error, "a failed bg agent must mark the tree as error");
        assert!(!tree.is_streaming, "tree frozen after the last agent finished");
        assert!(
            tree.content.contains('✗'),
            "tree should show the failure indicator, got: {:?}",
            tree.content
        );
    }

    /// Regression guard: the common foreground-only flow still finalizes
    /// (freeze + clear) once every subagent has completed.
    #[test]
    fn foreground_agents_finalize_when_all_complete() {
        let mut s = test_state();
        s.handle_fleet_event(started("a"));
        s.handle_fleet_event(started("b"));

        s.handle_fleet_event(completed("a", SubagentOutcome::Completed));
        assert!(
            !s.agent_progress.is_empty(),
            "must not finalize until every agent has finished"
        );

        s.handle_fleet_event(completed("b", SubagentOutcome::Completed));
        assert!(
            s.agent_progress.is_empty(),
            "finalized once all agents completed"
        );
        let tree = s
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "agents")
            .expect("agents tree message");
        assert!(!tree.is_error);
        assert!(!tree.is_streaming);
    }
}
