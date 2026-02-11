//! /branch command - create a new branch from a conversation point

use super::CommandResult;
use tau_agent::Agent;

pub struct BranchCommand;

impl BranchCommand {
    /// Execute /branch command
    /// - No args: open branch selector (TUI) or show message count (CLI)
    /// - With index: branch from that message index
    pub fn execute(args: &str, agent: &Agent) -> CommandResult {
        let message_count = agent.messages().len();

        if args.is_empty() {
            if message_count == 0 {
                return CommandResult::Message(
                    "No messages to branch from. Start a conversation first.".to_string(),
                );
            }
            // Open branch selector in TUI mode
            CommandResult::OpenBranchSelector
        } else {
            // Parse the index
            match args.parse::<usize>() {
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
                    args,
                    message_count.saturating_sub(1)
                )),
            }
        }
    }
}
