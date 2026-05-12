//! /plan command — enter plan mode via Plan subagent

use async_trait::async_trait;

use super::Command;
use crate::driver::{Frontend, Session};

pub struct PlanCommand;

#[async_trait]
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

    async fn execute(&self, args: &str, session: &mut Session, frontend: &mut dyn Frontend) {
        let args = args.trim();

        if !session.is_plan_mode() {
            if args.is_empty() {
                frontend
                    .show_system(
                        "Usage: /plan <description> — enter plan mode to explore and design an approach",
                    )
                    .await;
                return;
            }
            if let Err(e) = session.enter_plan_mode(args.to_string(), frontend).await {
                frontend
                    .show_error(&format!("Plan mode failed: {}", e))
                    .await;
            }
            return;
        }

        match args {
            "" => {
                frontend
                    .show_system(
                        "In plan mode. Use /plan approve to approve, or /plan exit to cancel.",
                    )
                    .await
            }
            "approve" | "ok" | "yes" => {
                if let Err(e) = session.approve_plan(frontend).await {
                    frontend
                        .show_error(&format!("Plan approval failed: {}", e))
                        .await;
                }
            }
            "exit" | "cancel" | "quit" => session.exit_plan_mode(frontend).await,
            _ => {
                frontend
                    .show_system(
                        "Already in plan mode. Use /plan approve or /plan exit first.",
                    )
                    .await
            }
        }
    }
}
