//! Context compaction for long conversations
//!
//! When conversations grow too large for the model's context window,
//! this module summarizes old messages and replaces them with a compact summary.

use std::sync::Arc;

use tau_ai::{Content, Message};
use tokio_util::sync::CancellationToken;

use crate::config::AgentConfig;
use crate::transport::{AgentRunConfig, Transport};

/// Configuration for context compaction
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Whether compaction is enabled
    pub enabled: bool,
    /// Reserve this many tokens below context_window to trigger proactive compaction
    pub reserve_tokens: u64,
    /// Keep at least this many tokens of recent messages when compacting
    pub keep_recent_tokens: u64,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 16384,
            keep_recent_tokens: 20000,
        }
    }
}

/// Result of a compaction operation
pub struct CompactionResult {
    /// The generated summary text
    pub summary: String,
    /// Index of first message kept (not summarized)
    pub first_kept_index: usize,
    /// Estimated tokens before compaction
    pub tokens_before: u64,
    /// Files that were read during the summarized portion
    pub read_files: Vec<String>,
    /// Files that were modified during the summarized portion
    pub modified_files: Vec<String>,
}

/// Reason for compaction
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionReason {
    /// Context approaching window limit
    Threshold,
    /// Context overflow error from LLM
    Overflow,
    /// Manual /compact command
    Manual,
}

/// Result of finding a cut point in the message list
struct CutPointResult {
    first_kept_index: usize,
    turn_start_index: Option<usize>,
    is_split_turn: bool,
}

/// Estimate token count for a single message (chars/4 heuristic)
pub fn estimate_tokens(message: &Message) -> u64 {
    let char_count: usize = match message {
        Message::User { content, .. }
        | Message::Assistant { content, .. }
        | Message::ToolResult { content, .. }
        | Message::SystemInjection { content, .. } => content_char_count(content),
    };
    (char_count / 4) as u64
}

/// Estimate total tokens for a slice of messages
pub fn estimate_total_tokens(messages: &[Message]) -> u64 {
    messages.iter().map(estimate_tokens).sum()
}

fn content_char_count(content: &[Content]) -> usize {
    content
        .iter()
        .map(|c| match c {
            Content::Text { text } => text.len(),
            Content::Thinking { thinking, .. } => thinking.len(),
            Content::ToolCall {
                name, arguments, ..
            } => name.len() + serde_json::to_string(arguments).unwrap_or_default().len(),
            Content::Image { .. } => 4800,
            Content::RedactedThinking { data } => data.len(),
            Content::ServerToolUse { name, input, .. } => {
                name.len() + serde_json::to_string(input).unwrap_or_default().len()
            }
            Content::ServerToolResult { content, .. } => {
                serde_json::to_string(content).unwrap_or_default().len()
            }
        })
        .sum()
}

/// Serialize messages to plain text for the summarization prompt.
fn serialize_messages_for_summary(messages: &[Message]) -> String {
    let mut out = String::new();

    for msg in messages {
        match msg {
            Message::User { content, .. } => {
                let text = content_to_text(content);
                if !text.is_empty() {
                    out.push_str("[User]: ");
                    out.push_str(&text);
                    out.push('\n');
                }
            }
            Message::Assistant { content, .. } => {
                let mut thinking_parts = Vec::new();
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();

                for c in content {
                    match c {
                        Content::Thinking { thinking, .. } => {
                            thinking_parts.push(thinking.as_str());
                        }
                        Content::Text { text } => {
                            text_parts.push(text.as_str());
                        }
                        Content::ToolCall {
                            name, arguments, ..
                        } => {
                            let args_str = format_tool_args(arguments);
                            tool_calls.push(format!("{}({})", name, args_str));
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
                    format!("[Tool error ({})]: ", tool_name)
                } else {
                    format!("[Tool result ({})]: ", tool_name)
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
                            format!("\"{}\"", s)
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
                format!("{}={}", k, val)
            })
            .collect::<Vec<_>>()
            .join(", "),
        _ => args.to_string(),
    }
}

/// Tool names that perform read-only file operations.
const READ_TOOLS: &[&str] = &["read", "glob", "grep", "list"];
/// Tool names that perform file modifications.
const WRITE_TOOLS: &[&str] = &["write", "edit"];

/// Extract file paths from tool calls in messages
fn extract_file_operations(messages: &[Message]) -> (Vec<String>, Vec<String>) {
    let mut read_files = Vec::new();
    let mut modified_files = Vec::new();

    for msg in messages {
        if let Message::Assistant { content, .. } = msg {
            for c in content {
                if let Content::ToolCall {
                    name, arguments, ..
                } = c
                {
                    let name_str = name.as_str();
                    if READ_TOOLS.contains(&name_str) {
                        if let Some(path) = arguments.get("path").and_then(|v| v.as_str()) {
                            if !read_files.contains(&path.to_string()) {
                                read_files.push(path.to_string());
                            }
                        }
                    } else if WRITE_TOOLS.contains(&name_str) {
                        if let Some(path) = arguments.get("path").and_then(|v| v.as_str()) {
                            if !modified_files.contains(&path.to_string()) {
                                modified_files.push(path.to_string());
                            }
                        }
                        if let Some(path) = arguments.get("file_path").and_then(|v| v.as_str()) {
                            if !modified_files.contains(&path.to_string()) {
                                modified_files.push(path.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    (read_files, modified_files)
}

/// Find where to cut messages for compaction.
fn find_cut_point(
    messages: &[Message],
    keep_recent_tokens: u64,
    cancel: &CancellationToken,
) -> Option<CutPointResult> {
    if messages.len() < 2 {
        return None;
    }

    let mut accumulated: u64 = 0;
    let mut cut_index = messages.len();

    for i in (0..messages.len()).rev() {
        if cancel.is_cancelled() {
            return None;
        }
        accumulated += estimate_tokens(&messages[i]);
        if accumulated >= keep_recent_tokens {
            cut_index = i + 1;
            break;
        }
    }

    if cut_index <= 1 {
        return None;
    }

    if cut_index >= messages.len() {
        cut_index = messages.len().saturating_sub(2);
        if cut_index <= 1 {
            return None;
        }
    }

    let mut first_kept = cut_index;
    while first_kept < messages.len() {
        if cancel.is_cancelled() {
            return None;
        }
        match &messages[first_kept] {
            Message::User { .. } | Message::SystemInjection { .. } => break,
            Message::Assistant { .. } => break,
            Message::ToolResult { .. } => {
                first_kept += 1;
            }
        }
    }

    if first_kept >= messages.len() {
        return None;
    }

    let is_split_turn = matches!(&messages[first_kept], Message::Assistant { .. })
        && first_kept > 0
        && has_tool_calls_with_results(messages, first_kept);

    let turn_start_index = if is_split_turn {
        Some(find_turn_start(messages, first_kept))
    } else {
        None
    };

    Some(CutPointResult {
        first_kept_index: first_kept,
        turn_start_index,
        is_split_turn,
    })
}

fn has_tool_calls_with_results(messages: &[Message], idx: usize) -> bool {
    if let Message::Assistant { content, .. } = &messages[idx] {
        let has_tool_calls = content
            .iter()
            .any(|c| matches!(c, Content::ToolCall { .. }));
        if has_tool_calls && idx + 1 < messages.len() {
            return matches!(&messages[idx + 1], Message::ToolResult { .. });
        }
    }
    false
}

fn find_turn_start(messages: &[Message], from: usize) -> usize {
    let mut idx = from;
    while idx > 0 {
        match &messages[idx - 1] {
            Message::ToolResult { .. } => idx -= 1,
            Message::Assistant { .. } => {
                idx -= 1;
                continue;
            }
            _ => break,
        }
    }
    idx
}

const SUMMARIZATION_SYSTEM_PROMPT: &str = "\
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

/// Run compaction on the given messages.
pub async fn compact(
    messages: &[Message],
    config: &CompactionConfig,
    agent_config: &AgentConfig,
    transport: &Arc<dyn Transport>,
    previous_summary: Option<&str>,
    cancel: &CancellationToken,
) -> Result<CompactionResult, String> {
    let tokens_before = estimate_total_tokens(messages);

    if cancel.is_cancelled() {
        return Err("Compaction cancelled".to_string());
    }
    let cut = find_cut_point(messages, config.keep_recent_tokens, cancel).ok_or_else(|| {
        if cancel.is_cancelled() {
            "Compaction cancelled".to_string()
        } else {
            "Not enough messages to compact".to_string()
        }
    })?;

    let messages_to_summarize = &messages[..cut.first_kept_index];

    let (read_files, modified_files) = extract_file_operations(messages_to_summarize);
    let conversation_text = serialize_messages_for_summary(messages_to_summarize);

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

    let prompt = if let Some(prev_summary) = previous_summary {
        UPDATE_SUMMARIZATION_PROMPT
            .replace("{previous_summary}", prev_summary)
            .replace("{conversation}", &conversation_text)
            .replace("{read_files}", &read_files_str)
            .replace("{modified_files}", &modified_files_str)
    } else {
        SUMMARIZATION_PROMPT
            .replace("{conversation}", &conversation_text)
            .replace("{read_files}", &read_files_str)
            .replace("{modified_files}", &modified_files_str)
    };

    let mut full_summary = String::new();

    if cut.is_split_turn {
        if let Some(turn_start) = cut.turn_start_index {
            let turn_prefix_messages = &messages[turn_start..cut.first_kept_index];
            let turn_prefix_text = serialize_messages_for_summary(turn_prefix_messages);
            let turn_prompt =
                TURN_PREFIX_SUMMARIZATION_PROMPT.replace("{conversation}", &turn_prefix_text);

            let turn_summary =
                call_summarization_llm(&turn_prompt, agent_config, transport, cancel).await?;
            full_summary.push_str("## Split Turn Context\n");
            full_summary.push_str(&turn_summary);
            full_summary.push_str("\n\n");
        }
    }

    if cancel.is_cancelled() {
        return Err("Compaction cancelled".to_string());
    }
    let main_summary = call_summarization_llm(&prompt, agent_config, transport, cancel).await?;
    full_summary.push_str(&main_summary);

    Ok(CompactionResult {
        summary: full_summary,
        first_kept_index: cut.first_kept_index,
        tokens_before,
        read_files,
        modified_files,
    })
}

/// Make an LLM call for summarization using the same transport infrastructure
async fn call_summarization_llm(
    prompt: &str,
    agent_config: &AgentConfig,
    transport: &Arc<dyn Transport>,
    cancel: &CancellationToken,
) -> Result<String, String> {
    use futures::StreamExt;

    let run_config = AgentRunConfig {
        system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_string()),
        tools: vec![],
        server_tools: vec![],
        model: agent_config.model.clone(),
        reasoning: None,
        thinking_adaptive: false,
        max_tokens: Some(4096),
        temperature: None,
        cache_scope: None,
        cache_ttl: None,
        system_prompt_boundary: None,
    };

    let user_message = Message::user(prompt);

    let mut event_stream = transport
        .run(vec![user_message], &run_config, cancel.clone())
        .await
        .map_err(|e| format!("Compaction LLM call failed: {}", e))?;

    let mut result_text = String::new();

    while let Some(event) = event_stream.next().await {
        match event {
            crate::events::AgentEvent::MessageEnd { message } => {
                result_text = message.text();
            }
            crate::events::AgentEvent::Error { message } => {
                return Err(format!("Compaction LLM error: {}", message));
            }
            _ => {}
        }
    }

    if result_text.is_empty() {
        return Err("Compaction LLM returned empty response".to_string());
    }

    Ok(result_text)
}

#[cfg(test)]
mod tests {
    use tau_ai::{AssistantMetadata, Content, Message};

    use super::*;

    fn user_msg(text: &str) -> Message {
        Message::User {
            content: vec![Content::text(text)],
            timestamp: 0,
        }
    }

    fn assistant_msg(text: &str) -> Message {
        Message::Assistant {
            content: vec![Content::text(text)],
            metadata: AssistantMetadata::default(),
        }
    }

    #[test]
    fn test_estimate_tokens_text() {
        let msg = user_msg("Hello world!"); // 12 chars -> 3 tokens
        assert_eq!(estimate_tokens(&msg), 3);
    }

    #[test]
    fn test_estimate_total_tokens() {
        let messages = vec![
            user_msg(&"x".repeat(400)),      // 100 tokens
            assistant_msg(&"y".repeat(800)), // 200 tokens
        ];
        assert_eq!(estimate_total_tokens(&messages), 300);
    }

    #[test]
    fn test_find_cut_point_not_enough_messages() {
        let messages = vec![user_msg("hi")];
        let cancel = CancellationToken::new();
        assert!(find_cut_point(&messages, 100, &cancel).is_none());
    }

    #[test]
    fn test_serialize_messages() {
        let messages = vec![user_msg("Hello"), assistant_msg("Hi there!")];
        let text = serialize_messages_for_summary(&messages);
        assert!(text.contains("[User]: Hello"));
        assert!(text.contains("[Assistant]: Hi there!"));
    }

    fn assistant_tool_call(name: &str) -> Message {
        Message::Assistant {
            content: vec![Content::tool_call("id", name, serde_json::json!({}))],
            metadata: AssistantMetadata::default(),
        }
    }

    fn tool_result(tool_call_id: &str) -> Message {
        Message::ToolResult {
            tool_call_id: tool_call_id.to_string(),
            tool_name: "test".to_string(),
            content: vec![Content::text("result")],
            is_error: false,
            timestamp: 0,
        }
    }

    // ─── find_turn_start ──────────────────────────────────────────

    #[test]
    fn find_turn_start_walks_back_through_multi_step_execution() {
        // A single agent execution: user prompt → assistant(tool) → result → assistant(tool)
        // find_turn_start should walk back to the first assistant message (index 1),
        // since all intermediate assistant/tool_result pairs are part of one execution.
        let messages = vec![
            user_msg("do something"),         // 0
            assistant_tool_call("read"),       // 1 — first step
            tool_result("id"),                 // 2
            assistant_tool_call("write"),      // 3 — second step (from=3)
            tool_result("id"),                 // 4
        ];
        assert_eq!(find_turn_start(&messages, 3), 1);
    }

    #[test]
    fn find_turn_start_stops_at_user_boundary() {
        // Two separate user turns. find_turn_start from the second assistant
        // should stop at the user message boundary (index 3), not walk into turn 1.
        let messages = vec![
            user_msg("do X"),                 // 0
            assistant_tool_call("read"),       // 1 — turn 1
            tool_result("id"),                 // 2
            user_msg("now do Y"),             // 3 — new user turn
            assistant_tool_call("write"),      // 4 — turn 2 start
            tool_result("id"),                 // 5
            assistant_tool_call("edit"),       // 6 — from=6
            tool_result("id"),                 // 7
        ];
        assert_eq!(find_turn_start(&messages, 6), 4);
    }

    #[test]
    fn find_turn_start_at_beginning_of_conversation() {
        // from points to the very first assistant message — should return 0.
        let messages = vec![
            assistant_tool_call("read"),       // 0
            tool_result("id"),                 // 1
            assistant_tool_call("write"),      // 2 — from=2
            tool_result("id"),                 // 3
        ];
        assert_eq!(find_turn_start(&messages, 2), 0);
    }

    #[test]
    fn find_turn_start_single_step() {
        // Only one assistant+tool_result before from — should return the assistant index.
        let messages = vec![
            user_msg("hello"),                // 0
            assistant_tool_call("read"),       // 1 — from=1
            tool_result("id"),                 // 2
        ];
        assert_eq!(find_turn_start(&messages, 1), 1);
    }
}
