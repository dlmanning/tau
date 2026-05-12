//! Slash commands.
//!
//! Each command implements the [`Command`] trait and acts directly on
//! the [`Session`] (state-changing operations) and the
//! [`Frontend`](crate::driver::Frontend) (output). The driver
//! ([`Session::handle_command`](crate::driver::Session)) looks up a
//! command by name/alias and calls `execute`.

mod branch;
mod model;
mod plan;
mod session;
mod thinking;

use async_trait::async_trait;

use crate::driver::{Frontend, Session};

/// A slash command.
#[async_trait]
pub trait Command: Send + Sync {
    fn name(&self) -> &str;
    fn aliases(&self) -> &[&str] {
        &[]
    }
    fn description(&self) -> &str;
    async fn execute(&self, args: &str, session: &mut Session, frontend: &mut dyn Frontend);
}

/// Registry of all built-in commands. Cheap to call: returns a fresh
/// `Vec<Box<dyn Command>>` per invocation. Commands are stateless.
pub fn all_commands() -> Vec<Box<dyn Command>> {
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

// ─── Simple commands inlined ─────────────────────────────────────────

struct HelpCommand;

#[async_trait]
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
    async fn execute(&self, _args: &str, _session: &mut Session, frontend: &mut dyn Frontend) {
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
        frontend.show_system(&output).await;
    }
}

struct ClearCommand;

#[async_trait]
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
    async fn execute(&self, _args: &str, session: &mut Session, frontend: &mut dyn Frontend) {
        session.clear(frontend).await;
    }
}

struct ExitCommand;

#[async_trait]
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
    async fn execute(&self, _args: &str, session: &mut Session, _frontend: &mut dyn Frontend) {
        session.request_exit();
    }
}

struct CompactCommand;

#[async_trait]
impl Command for CompactCommand {
    fn name(&self) -> &str {
        "compact"
    }
    fn description(&self) -> &str {
        "Compact context by summarizing old messages"
    }
    async fn execute(&self, _args: &str, session: &mut Session, frontend: &mut dyn Frontend) {
        session.compact(frontend).await;
    }
}
