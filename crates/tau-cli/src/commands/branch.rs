//! /branch command - create a new branch from a conversation point

use async_trait::async_trait;

use super::Command;
use crate::driver::{Frontend, Session};

pub struct BranchCommand;

#[async_trait]
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
    async fn execute(&self, args: &str, session: &mut Session, frontend: &mut dyn Frontend) {
        let messages = session.current_messages().await;
        let message_count = messages.len();

        if args.is_empty() {
            if message_count == 0 {
                frontend
                    .show_system("No messages to branch from. Start a conversation first.")
                    .await;
                return;
            }
            // Prefer the frontend's native picker (TUI popup); fall
            // back to a text list if the frontend can't render one.
            if frontend.open_branch_selector(&messages).await {
                return;
            }
            let mut out = String::from("Messages in conversation:\n");
            for (i, msg) in messages.iter().enumerate() {
                let role = match msg {
                    tau_ai::Message::User { .. } => "user",
                    tau_ai::Message::Assistant { .. } => "assistant",
                    tau_ai::Message::ToolResult { .. } => "tool",
                    tau_ai::Message::SystemInjection { .. } => "system",
                };
                let text = msg.text();
                let preview: String =
                    text.chars().take(60).collect::<String>().replace('\n', " ");
                out.push_str(&format!("  {}: [{}] {}\n", i, role, preview));
            }
            out.push_str("\nUse /branch <index> to create a branch from that message.");
            frontend.show_system(&out).await;
            return;
        }

        match args.parse::<usize>() {
            Ok(index) if index < message_count => session.branch_from(Some(index), frontend).await,
            Ok(index) => {
                frontend
                    .show_system(&format!(
                        "Invalid index {}. Valid range: 0-{}",
                        index,
                        message_count.saturating_sub(1)
                    ))
                    .await
            }
            Err(_) => {
                frontend
                    .show_system(&format!(
                        "Invalid index '{}'. Use a number (0-{}) or no argument to list messages.",
                        args,
                        message_count.saturating_sub(1)
                    ))
                    .await
            }
        }
    }
}
