//! Plan mode utilities — context summary for Plan subagent injection.

use tau_ai::{Content, Message};

/// Maximum number of recent messages to include in the context summary.
const MAX_SUMMARY_MESSAGES: usize = 20;

/// Build a lightweight text summary of recent conversation for injection
/// into a Plan subagent. Includes user/assistant text, skips tool
/// calls and results for brevity.
pub fn build_context_summary(messages: &[Message], previous_summary: Option<&str>) -> String {
    let mut parts: Vec<String> = Vec::new();

    if let Some(summary) = previous_summary {
        parts.push(format!("Earlier conversation summary:\n{}", summary));
    }

    let start = messages.len().saturating_sub(MAX_SUMMARY_MESSAGES);
    for msg in &messages[start..] {
        match msg {
            Message::User { content, .. } => {
                let text = extract_text(content);
                if !text.is_empty() {
                    parts.push(format!("User: {}", text));
                }
            }
            Message::Assistant { content, .. } => {
                let text = extract_text(content);
                if !text.is_empty() {
                    parts.push(format!("Assistant: {}", text));
                }
            }
            // Skip tool results and system injections
            _ => {}
        }
    }

    parts.join("\n\n")
}

fn extract_text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| c.as_text())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extract the last non-empty assistant text from a message list.
/// Used to recover the plan from a Plan subagent's transcript.
pub fn extract_final_text(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find_map(|m| {
            if let Message::Assistant { content, .. } = m {
                let text: String = content
                    .iter()
                    .filter_map(|c| c.as_text())
                    .collect::<Vec<_>>()
                    .join("");
                if text.is_empty() { None } else { Some(text) }
            } else {
                None
            }
        })
        .unwrap_or_default()
}

/// Format the full prompt for a Plan subagent, combining context and task description.
pub fn build_plan_prompt(context_summary: &str, description: &str) -> String {
    if context_summary.is_empty() {
        format!("Create an implementation plan for the following task:\n\n{}", description)
    } else {
        format!(
            "Here is context from the conversation so far:\n\n\
             <context>\n{}\n</context>\n\n\
             Create an implementation plan for the following task:\n\n{}",
            context_summary, description
        )
    }
}
