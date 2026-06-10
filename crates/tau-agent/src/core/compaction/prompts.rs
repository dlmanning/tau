//! Prompt templates and prompt assembly for summarization.
//!
//! Everything here is pure string work: serializing messages into the
//! `<conversation>` transcript format and splicing it (plus file lists,
//! a previous summary, and optional custom instructions) into the
//! hardcoded templates.

use tau_ai::{Content, Message};

pub(super) const SUMMARIZATION_SYSTEM_PROMPT: &str = "\
You are a specialized summarization model. Your task is to create a comprehensive \
yet concise summary of a coding conversation. This summary will replace the original \
messages in the conversation context, so it must capture all essential information \
needed to continue the conversation effectively.";

const SUMMARIZATION_PROMPT: &str = "\
Please provide a detailed summary of this conversation so far. The summary should:

1. **Goal**: What is the user's primary objective?
2. **Progress**: What has been accomplished so far? List specific changes made.
3. **Key Decisions**: What important technical decisions were made and why?
4. **Next Steps**: What was the user about to do or ask about next?
5. **Critical Context**: Any important constraints, preferences, or context that would be lost.
6. **Files Read**: {read_files}
7. **Files Modified**: {modified_files}

Format your response as a structured summary using the headers above. Be thorough but concise. \
Focus on information that would be needed to continue the conversation seamlessly.

<conversation>
{conversation}
</conversation>";

const UPDATE_SUMMARIZATION_PROMPT: &str = "\
Below is an existing summary of an earlier portion of this conversation, followed by \
new messages that occurred after that summary. Please create an updated, comprehensive \
summary that integrates both.

<previous-summary>
{previous_summary}
</previous-summary>

Please provide an updated summary that incorporates the new messages below. The summary should:

1. **Goal**: What is the user's primary objective? (update if it has evolved)
2. **Progress**: What has been accomplished so far? Include both previous and new progress.
3. **Key Decisions**: What important technical decisions were made and why?
4. **Next Steps**: What was about to happen next?
5. **Critical Context**: Any important constraints, preferences, or context.
6. **Files Read**: {read_files}
7. **Files Modified**: {modified_files}

<new-messages>
{conversation}
</new-messages>";

const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = "\
The following is the beginning of a conversation turn that was split during context compaction. \
Please provide a very brief summary of what was happening in this partial turn, focusing on \
what the assistant was doing and what tool calls were made.

<partial-turn>
{conversation}
</partial-turn>";

/// Assemble the main summarization prompt: the initial template, or the
/// update template when a previous summary exists. `custom_instructions`,
/// when present and non-empty after trimming, is appended as a
/// `## User instructions` section.
pub(super) fn build_main_prompt(
    conversation_text: &str,
    previous_summary: Option<&str>,
    read_files: &[String],
    modified_files: &[String],
    custom_instructions: Option<&str>,
) -> String {
    let read_files_str = if read_files.is_empty() {
        "(none)".to_string()
    } else {
        read_files.join(", ")
    };
    let modified_files_str = if modified_files.is_empty() {
        "(none)".to_string()
    } else {
        modified_files.join(", ")
    };

    let mut prompt = if let Some(prev) = previous_summary {
        UPDATE_SUMMARIZATION_PROMPT
            .replace("{previous_summary}", prev)
            .replace("{conversation}", conversation_text)
            .replace("{read_files}", &read_files_str)
            .replace("{modified_files}", &modified_files_str)
    } else {
        SUMMARIZATION_PROMPT
            .replace("{conversation}", conversation_text)
            .replace("{read_files}", &read_files_str)
            .replace("{modified_files}", &modified_files_str)
    };

    if let Some(instructions) = custom_instructions {
        let trimmed = instructions.trim();
        if !trimmed.is_empty() {
            prompt.push_str("\n\n## User instructions\n\n");
            prompt.push_str(trimmed);
        }
    }

    prompt
}

/// Assemble the sub-summary prompt for the prefix of a split turn.
/// Intentionally never carries custom instructions.
pub(super) fn build_turn_prefix_prompt(turn_prefix_text: &str) -> String {
    TURN_PREFIX_SUMMARIZATION_PROMPT.replace("{conversation}", turn_prefix_text)
}

pub(super) fn serialize_messages_for_summary(messages: &[Message]) -> String {
    let mut out = String::new();
    for msg in messages {
        match msg {
            Message::User { content, .. } => {
                let text = content_to_text(content);
                if text.is_empty() {
                    continue;
                }
                // Skip the synthetic `<context-summary>` user message
                // prepended by a previous compaction pass. The
                // UPDATE_SUMMARIZATION_PROMPT already embeds the
                // summary text in its own `<previous-summary>`
                // section; including the echo here would double the
                // prompt's token cost and confuse the model into
                // re-summarizing its own prior summary.
                if text.starts_with("<context-summary>") {
                    continue;
                }
                out.push_str("[User]: ");
                out.push_str(&text);
                out.push('\n');
            }
            Message::Assistant { content, .. } => {
                let mut thinking_parts = Vec::new();
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();
                for c in content {
                    match c {
                        Content::Thinking { thinking, .. } => {
                            thinking_parts.push(thinking.as_str())
                        }
                        Content::Text { text } => text_parts.push(text.as_str()),
                        Content::ToolCall {
                            name, arguments, ..
                        } => {
                            tool_calls.push(format!("{name}({})", format_tool_args(arguments)));
                        }
                        _ => {}
                    }
                }
                if !thinking_parts.is_empty() {
                    out.push_str("[Assistant thinking]: ");
                    out.push_str(&thinking_parts.join(" "));
                    out.push('\n');
                }
                if !text_parts.is_empty() {
                    out.push_str("[Assistant]: ");
                    out.push_str(&text_parts.join(""));
                    out.push('\n');
                }
                if !tool_calls.is_empty() {
                    out.push_str("[Assistant tool calls]: ");
                    out.push_str(&tool_calls.join("; "));
                    out.push('\n');
                }
            }
            Message::ToolResult {
                tool_name,
                content,
                is_error,
                ..
            } => {
                let text = content_to_text(content);
                let label = if *is_error {
                    format!("[Tool error ({tool_name})]: ")
                } else {
                    format!("[Tool result ({tool_name})]: ")
                };
                out.push_str(&label);
                if text.len() > 2000 {
                    out.push_str(&text[..2000]);
                    out.push_str("...(truncated)");
                } else {
                    out.push_str(&text);
                }
                out.push('\n');
            }
            Message::SystemInjection { content, .. } => {
                let text = content_to_text(content);
                if !text.is_empty() {
                    out.push_str("[System]: ");
                    out.push_str(&text);
                    out.push('\n');
                }
            }
        }
    }
    out
}

fn content_to_text(content: &[Content]) -> String {
    content
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            Content::Image { .. } => Some("[image]"),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn format_tool_args(args: &serde_json::Value) -> String {
    match args {
        serde_json::Value::Object(map) => map
            .iter()
            .map(|(k, v)| {
                let val = match v {
                    serde_json::Value::String(s) => {
                        if s.len() > 100 {
                            format!("\"{}...\"", &s[..100])
                        } else {
                            format!("\"{s}\"")
                        }
                    }
                    other => {
                        let s = other.to_string();
                        if s.len() > 100 {
                            format!("{}...", &s[..100])
                        } else {
                            s
                        }
                    }
                };
                format!("{k}={val}")
            })
            .collect::<Vec<_>>()
            .join(", "),
        _ => args.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tau_ai::{AssistantMetadata, Content, Message};

    fn user(text: &str) -> Message {
        Message::User {
            content: vec![Content::text(text)],
            timestamp: 0,
        }
    }
    fn assistant(text: &str) -> Message {
        Message::Assistant {
            content: vec![Content::text(text)],
            metadata: AssistantMetadata::default(),
        }
    }

    #[test]
    fn serialize_messages_smoke() {
        let messages = vec![user("Hello"), assistant("Hi there!")];
        let text = serialize_messages_for_summary(&messages);
        assert!(text.contains("[User]: Hello"));
        assert!(text.contains("[Assistant]: Hi there!"));
    }

    /// A previous compaction's `<context-summary>` user message must
    /// not be re-embedded in the next summarization prompt. The
    /// UPDATE template already injects it as `<previous-summary>`;
    /// including the echo here would double-charge tokens and confuse
    /// the model.
    #[test]
    fn serialize_skips_context_summary_user_message() {
        let messages = vec![
            user(
                "<context-summary>\nOld stuff happened.\n</context-summary>\n\nThe conversation was compacted. Continue from where we left off.",
            ),
            user("a new thing"),
            assistant("response"),
        ];
        let text = serialize_messages_for_summary(&messages);
        assert!(
            !text.contains("<context-summary>"),
            "context-summary echo excluded: {text}"
        );
        assert!(text.contains("[User]: a new thing"));
        assert!(text.contains("[Assistant]: response"));
    }
}
