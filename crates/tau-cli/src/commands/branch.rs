//! /branch command - create a new branch from a conversation point

use super::{Command, CommandContext, CommandResult};

pub struct BranchCommand;

impl Command for BranchCommand {
    fn name(&self) -> &str {
        "branch"
    }
    fn aliases(&self) -> &[&str] {
        &["b"]
    }
    fn description(&self) -> &str {
        "Branch conversation from a message (/branch [index])"
    }
    fn execute(&self, ctx: &CommandContext) -> CommandResult {
        let message_count = ctx.messages.len();

        if ctx.args.is_empty() {
            if message_count == 0 {
                return CommandResult::Message(
                    "No messages to branch from. Start a conversation first.".to_string(),
                );
            }
            CommandResult::OpenBranchSelector
        } else {
            match ctx.args.parse::<usize>() {
                Ok(index) => {
                    if index >= message_count {
                        CommandResult::Message(format!(
                            "Invalid index {}. Valid range: 0-{}",
                            index,
                            message_count.saturating_sub(1)
                        ))
                    } else {
                        CommandResult::BranchFrom(Some(index))
                    }
                }
                Err(_) => CommandResult::Message(format!(
                    "Invalid index '{}'. Use a number (0-{}) or no argument to open selector.",
                    ctx.args,
                    message_count.saturating_sub(1)
                )),
            }
        }
    }
}
