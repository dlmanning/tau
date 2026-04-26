use std::collections::HashMap;

use super::widgets::message_list::ChatMessage;
use tau_agent::AgentEvent;

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
                content,
                ..
            } => {
                if let Some(msg) = self
                    .messages
                    .iter_mut()
                    .rev()
                    .find(|m| m.id.as_deref() == Some(&tool_call_id))
                {
                    msg.content = content;
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
            AgentEvent::Subagent {
                agent_id,
                description,
                event,
            } => match *event {
                AgentEvent::AgentStart => {
                    let progress = AgentProgress::new(description);
                    self.agent_progress.insert(agent_id.clone(), progress);
                    self.agent_order.push(agent_id);
                    self.update_agent_tree();
                }
                AgentEvent::ToolExecutionStart { ref activity, .. } => {
                    if let Some(progress) = self.agent_progress.get_mut(&agent_id) {
                        progress.tool_count += 1;
                        progress.activity = activity.clone();
                    }
                    self.update_agent_tree();
                }
                AgentEvent::TurnEnd { ref usage, .. } => {
                    if let Some(progress) = self.agent_progress.get_mut(&agent_id) {
                        progress.input_tokens += usage.input;
                        progress.output_tokens += usage.output;
                    }
                    self.usage.accumulate(usage, &self.model);
                }
                AgentEvent::AgentEnd { .. } => {
                    if let Some(progress) = self.agent_progress.get_mut(&agent_id) {
                        progress.finished = true;
                    }
                    self.update_agent_tree();
                    if self.agent_progress.values().all(|p| p.finished) {
                        self.finalize_agent_progress(false);
                    }
                }
                AgentEvent::Error { ref message } => {
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
            AgentEvent::TurnStart { .. }
            | AgentEvent::MessageStart { .. }
            | AgentEvent::ToolApprovalResolved { .. }
            | AgentEvent::PlanStepStarted { .. }
            | AgentEvent::PlanStepCompleted { .. }
            | AgentEvent::PlanCompleted { .. } => {}
        }
    }

    /// Finalize the agent tree message and clear tracking state.
    fn finalize_agent_progress(&mut self, is_error: bool) {
        if !self.agent_progress.is_empty() {
            if let Some(msg) = self.messages.iter_mut().rev().find(|m| m.role == "agents") {
                msg.is_streaming = false;
                if is_error {
                    msg.is_error = true;
                }
            }
            self.agent_progress.clear();
            self.agent_order.clear();
        }
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
