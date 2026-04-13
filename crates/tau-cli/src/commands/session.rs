//! /session command - show session info and stats

use super::{Command, CommandContext, CommandResult};
use crate::utils::format_tokens as format_number;

pub struct SessionCommand;

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
    fn execute(&self, ctx: &CommandContext) -> CommandResult {
        let usage = ctx.usage;
        let model = &ctx.config.model;

        let mut output = String::from("Session Info\n");
        output.push_str(&"-".repeat(40));
        output.push('\n');

        output.push_str(&format!(
            "Model:      {} ({})\n",
            model.id,
            model.provider.name()
        ));
        output.push_str(&format!("Reasoning:  {:?}\n", ctx.config.reasoning));
        output.push('\n');

        let user_msgs = ctx.messages
            .iter()
            .filter(|m| matches!(m, tau_ai::Message::User { .. }))
            .count();
        let assistant_msgs = ctx.messages
            .iter()
            .filter(|m| matches!(m, tau_ai::Message::Assistant { .. }))
            .count();
        let tool_results = ctx.messages
            .iter()
            .filter(|m| matches!(m, tau_ai::Message::ToolResult { .. }))
            .count();

        output.push_str(&format!("Messages:   {} total\n", ctx.messages.len()));
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

        let total_tokens = usage.input + usage.output;
        let context_pct = (total_tokens as f64 / model.context_window as f64) * 100.0;
        output.push_str(&format!(
            "Context usage:  ~{:.1}% of {}k window\n",
            context_pct,
            model.context_window / 1000
        ));

        CommandResult::Message(output)
    }
}
