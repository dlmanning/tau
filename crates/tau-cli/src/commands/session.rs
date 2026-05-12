//! /session command - show session info and stats

use async_trait::async_trait;

use super::Command;
use crate::driver::{Frontend, Session};
use crate::utils::format_tokens as format_number;

pub struct SessionCommand;

#[async_trait]
impl Command for SessionCommand {
    fn name(&self) -> &str {
        "session"
    }
    fn aliases(&self) -> &[&str] {
        &["s"]
    }
    fn description(&self) -> &str {
        "Show session info and token usage"
    }
    async fn execute(&self, _args: &str, session: &mut Session, frontend: &mut dyn Frontend) {
        let Some(config) = session.current_config().await else {
            frontend.show_error("Agent shut down.").await;
            return;
        };
        let messages = session.current_messages().await;
        let usage = session.current_usage().await;
        let model = &config.model;

        let mut output = String::from("Session Info\n");
        output.push_str(&"-".repeat(40));
        output.push('\n');

        output.push_str(&format!(
            "Model:      {} ({})\n",
            model.id,
            model.provider.name()
        ));
        output.push_str(&format!("Reasoning:  {:?}\n", config.reasoning));
        output.push('\n');

        let user_msgs = messages
            .iter()
            .filter(|m| matches!(m, tau_ai::Message::User { .. }))
            .count();
        let assistant_msgs = messages
            .iter()
            .filter(|m| matches!(m, tau_ai::Message::Assistant { .. }))
            .count();
        let tool_results = messages
            .iter()
            .filter(|m| matches!(m, tau_ai::Message::ToolResult { .. }))
            .count();

        output.push_str(&format!("Messages:   {} total\n", messages.len()));
        output.push_str(&format!(
            "            {} user, {} assistant, {} tool results\n",
            user_msgs, assistant_msgs, tool_results
        ));
        output.push('\n');

        output.push_str("Token Usage:\n");
        output.push_str(&format!(
            "  Input:       {:>8}\n",
            format_number(usage.input)
        ));
        output.push_str(&format!(
            "  Output:      {:>8}\n",
            format_number(usage.output)
        ));
        if usage.cache_read > 0 {
            output.push_str(&format!(
                "  Cache read:  {:>8}\n",
                format_number(usage.cache_read)
            ));
        }
        if usage.cache_write > 0 {
            output.push_str(&format!(
                "  Cache write: {:>8}\n",
                format_number(usage.cache_write)
            ));
        }
        output.push('\n');

        let cost = usage.calculate_cost(model);
        output.push_str(&format!("Estimated cost: ${:.4}\n", cost.total));

        frontend.show_system(&output).await;
    }
}
