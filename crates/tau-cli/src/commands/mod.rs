//! Slash commands for interactive mode

mod branch;
mod model;
mod session;
mod thinking;

pub use branch::BranchCommand;
pub use model::ModelCommand;
pub use session::SessionCommand;
pub use thinking::ThinkingCommand;

use tau_agent::Agent;
use tau_ai::{Model, ReasoningLevel};

/// Result of executing a slash command
pub enum CommandResult {
    /// Clear the conversation
    Clear,
    /// Change the model
    ChangeModel(Model),
    /// Change the reasoning level
    ChangeReasoning(ReasoningLevel),
    /// Show a message to the user (not sent to agent)
    Message(String),
    /// Exit the application
    Exit,
    /// Unknown command
    Unknown(String),
    /// Open model selector (TUI only)
    OpenModelSelector,
    /// Open branch selector (TUI only) - lets user pick a message to branch from
    OpenBranchSelector,
    /// Create branch from specific message index
    BranchFrom(Option<usize>),
}

/// Parse and execute a slash command
pub fn execute_command(
    input: &str,
    agent: &Agent,
    current_model: &Model,
    current_reasoning: ReasoningLevel,
    available_models: &[Model],
) -> Option<CommandResult> {
    let input = input.trim();

    if !input.starts_with('/') {
        return None;
    }

    let parts: Vec<&str> = input[1..].splitn(2, ' ').collect();
    let command = parts[0].to_lowercase();
    let args = parts.get(1).map(|s| s.trim()).unwrap_or("");

    Some(match command.as_str() {
        "help" | "h" | "?" => CommandResult::Message(help_message()),

        "clear" | "c" => CommandResult::Clear,

        "quit" | "exit" | "q" => CommandResult::Exit,

        "model" | "m" => ModelCommand::execute(args, current_model, available_models),

        "thinking" | "t" => ThinkingCommand::execute(args, current_reasoning),

        "session" | "s" => SessionCommand::execute(agent, current_model, current_reasoning),

        "branch" | "b" => BranchCommand::execute(args, agent),

        _ => CommandResult::Unknown(command),
    })
}

fn help_message() -> String {
    r#"Available commands:
  /help, /h, /?        Show this help message
  /model, /m [name]    List models or switch to a model
  /thinking, /t [lvl]  Show or set reasoning level (off/minimal/low/medium/high)
  /session, /s         Show session info and token usage
  /branch, /b [index]  Branch conversation from a message (opens selector if no index)
  /clear, /c           Clear conversation history
  /quit, /exit, /q     Exit tau

Examples:
  /model               List available models
  /model sonnet        Switch to first model matching "sonnet"
  /thinking medium     Set reasoning to medium
  /branch              Open message selector to branch from
  /branch 3            Branch from message at index 3
  /clear               Start fresh conversation"#
        .to_string()
}
