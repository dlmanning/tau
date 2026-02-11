//! /thinking command - show and set reasoning level

use super::CommandResult;
use tau_ai::ReasoningLevel;

pub struct ThinkingCommand;

impl ThinkingCommand {
    pub fn execute(args: &str, current: ReasoningLevel) -> CommandResult {
        if args.is_empty() {
            // Show current level and options
            CommandResult::Message(show_levels(current))
        } else {
            // Try to set level
            match parse_level(args) {
                Some(level) => CommandResult::ChangeReasoning(level),
                None => CommandResult::Message(format!(
                    "Unknown reasoning level: '{}'\nValid levels: off, minimal, low, medium, high",
                    args
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

#[allow(dead_code)]
fn level_name(level: ReasoningLevel) -> &'static str {
    match level {
        ReasoningLevel::Off => "off",
        ReasoningLevel::Minimal => "minimal",
        ReasoningLevel::Low => "low",
        ReasoningLevel::Medium => "medium",
        ReasoningLevel::High => "high",
    }
}
