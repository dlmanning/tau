//! Slash commands for interactive mode

mod branch;
mod model;
mod plan;
mod session;
mod thinking;

pub use model::ModelCommand;
use tau_agent::AgentConfig;
use tau_ai::{Message, Model, ReasoningLevel};

/// Context passed to every command — uses pre-fetched snapshots from the handle.
pub struct CommandContext<'a> {
    pub args: &'a str,
    pub config: &'a AgentConfig,
    pub messages: &'a [Message],
    pub usage: &'a tau_ai::Usage,
    pub available_models: &'a [Model],
    /// Whether a non-main agent is currently active (e.g. plan mode).
    pub has_active_agent: bool,
}

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
    /// Trigger manual context compaction
    Compact,
    /// Enter plan mode — spawn a Plan subagent with the given description
    PlanStart(String),
    /// Approve the plan and return to the main agent
    PlanApprove,
    /// Exit plan mode without approving
    PlanExit,
}

/// Trait for slash commands
pub trait Command {
    /// Primary name (e.g. "model")
    fn name(&self) -> &str;

    /// Aliases (e.g. &["m"]). Default: none.
    fn aliases(&self) -> &[&str] {
        &[]
    }

    /// One-line description for /help
    fn description(&self) -> &str;

    /// Execute the command
    fn execute(&self, ctx: &CommandContext) -> CommandResult;
}

fn all_commands() -> Vec<Box<dyn Command>> {
    vec![
        Box::new(HelpCommand),
        Box::new(ClearCommand),
        Box::new(ExitCommand),
        Box::new(model::ModelCommand),
        Box::new(thinking::ThinkingCommand),
        Box::new(session::SessionCommand),
        Box::new(branch::BranchCommand),
        Box::new(plan::PlanCommand),
        Box::new(CompactCommand),
    ]
}

/// Parse and execute a slash command
pub fn execute_command(input: &str, ctx: &CommandContext) -> Option<CommandResult> {
    let input = input.trim();

    if !input.starts_with('/') {
        return None;
    }

    let parts: Vec<&str> = input[1..].splitn(2, ' ').collect();
    let cmd_name = parts[0].to_lowercase();
    let args = parts.get(1).map(|s| s.trim()).unwrap_or("");

    let ctx = CommandContext { args, ..*ctx };

    let commands = all_commands();
    let matched = commands
        .iter()
        .find(|c| c.name() == cmd_name || c.aliases().contains(&cmd_name.as_str()));

    Some(match matched {
        Some(cmd) => cmd.execute(&ctx),
        None => CommandResult::Unknown(cmd_name),
    })
}

// --- Simple commands inlined here ---

struct HelpCommand;

impl Command for HelpCommand {
    fn name(&self) -> &str {
        "help"
    }
    fn aliases(&self) -> &[&str] {
        &["h", "?"]
    }
    fn description(&self) -> &str {
        "Show available commands"
    }
    fn execute(&self, _ctx: &CommandContext) -> CommandResult {
        let commands = all_commands();
        let mut output = String::from("Available commands:\n");
        for cmd in &commands {
            let aliases = cmd.aliases();
            let names = if aliases.is_empty() {
                format!("/{}", cmd.name())
            } else {
                let all: Vec<String> = std::iter::once(cmd.name())
                    .chain(aliases.iter().copied())
                    .map(|n| format!("/{}", n))
                    .collect();
                all.join(", ")
            };
            output.push_str(&format!("  {:<24} {}\n", names, cmd.description()));
        }
        CommandResult::Message(output)
    }
}

struct ClearCommand;

impl Command for ClearCommand {
    fn name(&self) -> &str {
        "clear"
    }
    fn aliases(&self) -> &[&str] {
        &["c"]
    }
    fn description(&self) -> &str {
        "Clear conversation history"
    }
    fn execute(&self, _ctx: &CommandContext) -> CommandResult {
        CommandResult::Clear
    }
}

struct ExitCommand;

impl Command for ExitCommand {
    fn name(&self) -> &str {
        "quit"
    }
    fn aliases(&self) -> &[&str] {
        &["exit", "q"]
    }
    fn description(&self) -> &str {
        "Exit tau"
    }
    fn execute(&self, _ctx: &CommandContext) -> CommandResult {
        CommandResult::Exit
    }
}

struct CompactCommand;

impl Command for CompactCommand {
    fn name(&self) -> &str {
        "compact"
    }
    fn description(&self) -> &str {
        "Compact context by summarizing old messages"
    }
    fn execute(&self, _ctx: &CommandContext) -> CommandResult {
        CommandResult::Compact
    }
}
