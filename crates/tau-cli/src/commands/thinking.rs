//! /thinking command - show and set reasoning level

use tau_ai::ReasoningLevel;

use super::{Command, CommandContext, CommandResult};

pub struct ThinkingCommand;

impl Command for ThinkingCommand {
    fn name(&self) -> &str {
        "thinking"
    }
    fn aliases(&self) -> &[&str] {
        &["t"]
    }
    fn description(&self) -> &str {
        "Show or set reasoning level (/thinking <off|minimal|low|medium|high>)"
    }
    fn execute(&self, ctx: &CommandContext) -> CommandResult {
        if ctx.args.is_empty() {
            CommandResult::Message(show_levels(ctx.config.reasoning))
        } else {
            match parse_level(ctx.args) {
                Some(level) => CommandResult::ChangeReasoning(level),
                None => CommandResult::Message(format!(
                    "Unknown reasoning level: '{}'\nValid levels: off, minimal, low, medium, high",
                    ctx.args
                )),
            }
        }
    }
}

fn show_levels(current: ReasoningLevel) -> String {
    let levels = [
        (ReasoningLevel::Off, "off", "No extended thinking"),
        (ReasoningLevel::Minimal, "minimal", "Brief reasoning"),
        (ReasoningLevel::Low, "low", "Light reasoning"),
        (ReasoningLevel::Medium, "medium", "Moderate reasoning"),
        (ReasoningLevel::High, "high", "Deep reasoning"),
    ];

    let mut output = String::from("Reasoning levels:\n\n");

    for (level, name, desc) in levels {
        let marker = if level == current { " *" } else { "" };
        output.push_str(&format!("  {:<10} {}{}\n", name, desc, marker));
    }

    output.push_str("\nSet with: /thinking <level>");
    output
}

fn parse_level(s: &str) -> Option<ReasoningLevel> {
    match s.to_lowercase().as_str() {
        "off" | "none" | "0" => Some(ReasoningLevel::Off),
        "minimal" | "min" | "1" => Some(ReasoningLevel::Minimal),
        "low" | "2" => Some(ReasoningLevel::Low),
        "medium" | "med" | "3" => Some(ReasoningLevel::Medium),
        "high" | "4" => Some(ReasoningLevel::High),
        _ => None,
    }
}
