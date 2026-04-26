//! /plan command — enter plan mode via Plan subagent

use super::{Command, CommandContext, CommandResult};

pub struct PlanCommand;

impl Command for PlanCommand {
    fn name(&self) -> &str {
        "plan"
    }

    fn aliases(&self) -> &[&str] {
        &["p"]
    }

    fn description(&self) -> &str {
        "Enter plan mode (/plan <description>), approve (/plan approve), or exit (/plan exit)"
    }

    fn execute(&self, ctx: &CommandContext) -> CommandResult {
        let args = ctx.args.trim();

        if !ctx.has_active_agent {
            if args.is_empty() {
                return CommandResult::Message(
                    "Usage: /plan <description> — enter plan mode to explore and design an approach"
                        .to_string(),
                );
            }
            return CommandResult::PlanStart(args.to_string());
        }

        match args {
            "" => CommandResult::Message(
                "In plan mode. Use /plan approve to approve, or /plan exit to cancel."
                    .to_string(),
            ),
            "approve" | "ok" | "yes" => CommandResult::PlanApprove,
            "exit" | "cancel" | "quit" => CommandResult::PlanExit,
            _ => CommandResult::Message(
                "Already in plan mode. Use /plan approve or /plan exit first.".to_string(),
            ),
        }
    }
}
