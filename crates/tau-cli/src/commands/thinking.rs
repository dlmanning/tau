//! /thinking command - show and set reasoning level

use async_trait::async_trait;
use tau_ai::ReasoningLevel;

use super::Command;
use crate::driver::{Frontend, Session};

pub struct ThinkingCommand;

#[async_trait]
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
    async fn execute(&self, args: &str, session: &mut Session, frontend: &mut dyn Frontend) {
        if args.is_empty() {
            let current = session
                .current_config()
                .await
                .map(|c| c.reasoning)
                .unwrap_or(ReasoningLevel::Off);
            frontend.show_system(&show_levels(current)).await;
            return;
        }
        match parse_level(args) {
            Some(level) => session.change_reasoning(level, frontend).await,
            None => {
                frontend
                    .show_system(&format!(
                        "Unknown reasoning level: '{}'\nValid levels: off, minimal, low, medium, high",
                        args
                    ))
                    .await
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
