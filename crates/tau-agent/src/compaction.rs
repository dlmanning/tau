//! Context compaction for long conversations
//!
//! When conversations grow too large for the model's context window,
//! this module summarizes old messages and replaces them with a compact summary.

use std::sync::Arc;

use tau_ai::{Content, Message};

use crate::transport::{AgentRunConfig, Transport};

/// Configuration for context compaction
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Whether compaction is enabled
    pub enabled: bool,
    /// Reserve this many tokens below context_window to trigger proactive compaction
    pub reserve_tokens: u32,
    /// Keep at least this many tokens of recent messages when compacting
    pub keep_recent_tokens: u32,
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
    pub tokens_before: u32,
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
    /// Index of the first message to keep
    first_kept_index: usize,
    /// If the cut falls mid-turn, this is where the turn starts
    turn_start_index: Option<usize>,
    /// Whether we split a turn (assistant + tool results)
    is_split_turn: bool,
}

// --- Token Estimation ---

/// Estimate token count for a single message (chars/4 heuristic)
pub fn estimate_tokens(message: &Message) -> u32 {
    let char_count: usize = match message {
        Message::User { content, .. } => content_char_count(content),
        Message::Assistant { content, .. } => content_char_count(content),
        Message::ToolResult { content, .. } => content_char_count(content),
    };
    (char_count / 4) as u32
}

/// Estimate total tokens for a slice of messages
pub fn estimate_total_tokens(messages: &[Message]) -> u32 {
    messages.iter().map(|m| estimate_tokens(m)).sum()
}

fn content_char_count(content: &[Content]) -> usize {
    content
        .iter()
        .map(|c| match c {
            Content::Text { text } => text.len(),
            Content::Thinking { thinking } => thinking.len(),
            Content::ToolCall {
                name, arguments, ..
            } => name.len() + serde_json::to_string(arguments).unwrap_or_default().len(),
            Content::Image { .. } => 4800, // ~1200 tokens * 4 chars/token
        })
        .sum()
}

// --- Message Serialization ---

/// Serialize messages to plain text for the summarization prompt.
/// Uses a human-readable format to prevent the LLM from trying to "continue" the conversation.
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
                // Separate thinking from text and tool calls
                let mut thinking_parts = Vec::new();
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();

                for c in content {
                    match c {
                        Content::Thinking { thinking } => {
                            thinking_parts.push(thinking.as_str());
                        }
                        Content::Text { text } => {
                            text_parts.push(text.as_str());
                        }
                        Content::ToolCall {
                            name, arguments, ..
                        } => {
                            let args_str =
                                format_tool_args(arguments);
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
                // Truncate very long tool results
                if text.len() > 2000 {
                    out.push_str(&text[..2000]);
                    out.push_str("...(truncated)");
                } else {
                    out.push_str(&text);
                }
                out.push('\n');
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

// --- File Operation Tracking ---

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
                        // Also check file_path for edit tool
                        if let Some(path) =
                            arguments.get("file_path").and_then(|v| v.as_str())
                        {
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

// --- Cut Point Algorithm ---

/// Find where to cut messages for compaction.
/// Walks backwards from the end, keeping at least `keep_recent_tokens` tokens.
/// Never cuts at a ToolResult — finds the nearest User or Assistant boundary.
fn find_cut_point(messages: &[Message], keep_recent_tokens: u32) -> Option<CutPointResult> {
    if messages.len() < 2 {
        return None;
    }

    // Walk backwards accumulating tokens
    let mut accumulated: u32 = 0;
    let mut cut_index = messages.len();

    for i in (0..messages.len()).rev() {
        accumulated += estimate_tokens(&messages[i]);
        if accumulated >= keep_recent_tokens {
            cut_index = i + 1; // Keep from i+1 onwards
            break;
        }
    }

    // If we'd keep everything, no compaction needed
    if cut_index <= 1 {
        return None;
    }

    // If cut_index is at or past the end, nothing to compact
    if cut_index >= messages.len() {
        // All messages fit in keep_recent_tokens — but we were called, so force a cut
        // Keep at least the last 2 messages
        cut_index = messages.len().saturating_sub(2);
        if cut_index <= 1 {
            return None;
        }
    }

    // Find a valid cut point: never cut at a ToolResult
    // Walk forward from cut_index to find a User or start-of-turn boundary
    let mut first_kept = cut_index;
    while first_kept < messages.len() {
        match &messages[first_kept] {
            Message::User { .. } => break,
            Message::Assistant { .. } => break,
            Message::ToolResult { .. } => {
                // Can't start here — tool results must follow their assistant message
                first_kept += 1;
            }
        }
    }

    if first_kept >= messages.len() {
        return None;
    }

    // Check if we split an assistant turn (assistant message with tool calls has results after it)
    let is_split_turn = matches!(&messages[first_kept], Message::Assistant { .. })
        && first_kept > 0
        && has_tool_calls_with_results(&messages, first_kept);

    let turn_start_index = if is_split_turn {
        // Find the assistant message that starts this turn
        Some(find_turn_start(&messages, first_kept))
    } else {
        None
    };

    Some(CutPointResult {
        first_kept_index: first_kept,
        turn_start_index,
        is_split_turn,
    })
}

/// Check if an assistant message at `idx` has tool call results that follow it
fn has_tool_calls_with_results(messages: &[Message], idx: usize) -> bool {
    if let Message::Assistant { content, .. } = &messages[idx] {
        let has_tool_calls = content.iter().any(|c| matches!(c, Content::ToolCall { .. }));
        if has_tool_calls && idx + 1 < messages.len() {
            return matches!(&messages[idx + 1], Message::ToolResult { .. });
        }
    }
    false
}

/// Walk backwards to find where a turn starts (the first message in the turn group)
fn find_turn_start(messages: &[Message], from: usize) -> usize {
    let mut idx = from;
    while idx > 0 {
        match &messages[idx - 1] {
            Message::ToolResult { .. } => idx -= 1,
            Message::Assistant { .. } => {
                idx -= 1;
                // Check if this assistant message also has tool results before it
                continue;
            }
            _ => break,
        }
    }
    idx
}

// --- Summarization Prompts ---

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

// --- Main Compaction Function ---

/// Run compaction on the given messages.
///
/// This generates a summary of older messages by calling the LLM, and returns
/// the summary along with information about what was compacted.
pub async fn compact(
    messages: &[Message],
    config: &CompactionConfig,
    agent_config: &crate::agent::AgentConfig,
    transport: &Arc<dyn Transport>,
    previous_summary: Option<&str>,
) -> Result<CompactionResult, String> {
    let tokens_before = estimate_total_tokens(messages);

    // Find the cut point
    let cut = find_cut_point(messages, config.keep_recent_tokens)
        .ok_or_else(|| "Not enough messages to compact".to_string())?;

    let messages_to_summarize = &messages[..cut.first_kept_index];

    // Extract file operations
    let (read_files, modified_files) = extract_file_operations(messages_to_summarize);

    // Serialize messages to text
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

    // Build the summarization prompt
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

    // Generate turn prefix summary if we split a turn
    let mut full_summary = String::new();

    if cut.is_split_turn {
        if let Some(turn_start) = cut.turn_start_index {
            let turn_prefix_messages = &messages[turn_start..cut.first_kept_index];
            let turn_prefix_text = serialize_messages_for_summary(turn_prefix_messages);
            let turn_prompt =
                TURN_PREFIX_SUMMARIZATION_PROMPT.replace("{conversation}", &turn_prefix_text);

            let turn_summary = call_summarization_llm(
                &turn_prompt,
                agent_config,
                transport,
            )
            .await?;
            full_summary.push_str("## Split Turn Context\n");
            full_summary.push_str(&turn_summary);
            full_summary.push_str("\n\n");
        }
    }

    // Generate the main summary
    let main_summary = call_summarization_llm(
        &prompt,
        agent_config,
        transport,
    )
    .await?;
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
    agent_config: &crate::agent::AgentConfig,
    transport: &Arc<dyn Transport>,
) -> Result<String, String> {
    use futures::StreamExt;

    let run_config = AgentRunConfig {
        system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_string()),
        tools: vec![],
        model: agent_config.model.clone(),
        reasoning: None, // No reasoning for summarization
        max_tokens: Some(4096),
        temperature: None,
    };

    let user_message = Message::user(prompt);
    let cancel = tokio_util::sync::CancellationToken::new();

    let mut event_stream = transport
        .run(vec![], user_message, &run_config, cancel)
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
    use super::*;
    use tau_ai::{AssistantMetadata, Content, Message};

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

    fn assistant_with_tool_call(text: &str, tool_name: &str, args: serde_json::Value) -> Message {
        Message::Assistant {
            content: vec![
                Content::text(text),
                Content::tool_call("call_1", tool_name, args),
            ],
            metadata: AssistantMetadata::default(),
        }
    }

    fn tool_result_msg(name: &str, text: &str) -> Message {
        Message::ToolResult {
            tool_call_id: "call_1".to_string(),
            tool_name: name.to_string(),
            content: vec![Content::text(text)],
            is_error: false,
            timestamp: 0,
        }
    }

    #[test]
    fn test_estimate_tokens_text() {
        let msg = user_msg("Hello world!"); // 12 chars -> 3 tokens
        assert_eq!(estimate_tokens(&msg), 3);
    }

    #[test]
    fn test_estimate_tokens_image() {
        let msg = Message::User {
            content: vec![Content::image("base64data", "image/png")],
            timestamp: 0,
        };
        // Image is flat 4800 chars -> 1200 tokens
        assert_eq!(estimate_tokens(&msg), 1200);
    }

    #[test]
    fn test_estimate_total_tokens() {
        let messages = vec![
            user_msg(&"x".repeat(400)),     // 100 tokens
            assistant_msg(&"y".repeat(800)), // 200 tokens
        ];
        assert_eq!(estimate_total_tokens(&messages), 300);
    }

    #[test]
    fn test_find_cut_point_not_enough_messages() {
        let messages = vec![user_msg("hi")];
        assert!(find_cut_point(&messages, 100).is_none());
    }

    #[test]
    fn test_find_cut_point_basic() {
        // Create messages with known token sizes
        let messages = vec![
            user_msg(&"a".repeat(400)),      // 100 tokens
            assistant_msg(&"b".repeat(400)),  // 100 tokens
            user_msg(&"c".repeat(400)),       // 100 tokens
            assistant_msg(&"d".repeat(400)),  // 100 tokens
        ];
        // keep_recent_tokens=150 -> should keep last ~2 messages
        let cut = find_cut_point(&messages, 150).unwrap();
        assert!(cut.first_kept_index >= 2);
    }

    #[test]
    fn test_find_cut_point_skips_tool_result() {
        let messages = vec![
            user_msg(&"a".repeat(400)),
            assistant_with_tool_call("let me read", "read", serde_json::json!({"path": "/foo"})),
            tool_result_msg("read", &"content".repeat(100)),
            user_msg(&"b".repeat(400)),
            assistant_msg(&"c".repeat(400)),
        ];
        // Should never have first_kept_index pointing at a ToolResult
        let cut = find_cut_point(&messages, 200);
        if let Some(cut) = cut {
            assert!(!matches!(&messages[cut.first_kept_index], Message::ToolResult { .. }));
        }
    }

    #[test]
    fn test_serialize_messages() {
        let messages = vec![
            user_msg("Hello"),
            assistant_msg("Hi there!"),
        ];
        let text = serialize_messages_for_summary(&messages);
        assert!(text.contains("[User]: Hello"));
        assert!(text.contains("[Assistant]: Hi there!"));
    }

    #[test]
    fn test_serialize_tool_calls() {
        let messages = vec![assistant_with_tool_call(
            "Let me read that",
            "read",
            serde_json::json!({"path": "/tmp/test.rs"}),
        )];
        let text = serialize_messages_for_summary(&messages);
        assert!(text.contains("[Assistant]: Let me read that"));
        assert!(text.contains("[Assistant tool calls]: read("));
        assert!(text.contains("/tmp/test.rs"));
    }

    #[test]
    fn test_extract_file_operations() {
        let messages = vec![
            assistant_with_tool_call("", "read", serde_json::json!({"path": "/foo.rs"})),
            tool_result_msg("read", "contents"),
            assistant_with_tool_call("", "edit", serde_json::json!({"file_path": "/bar.rs", "old_string": "a", "new_string": "b"})),
            tool_result_msg("edit", "ok"),
        ];
        let (read, modified) = extract_file_operations(&messages);
        assert!(read.contains(&"/foo.rs".to_string()));
        assert!(modified.contains(&"/bar.rs".to_string()));
    }
}
