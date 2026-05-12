//! Context compaction.
//!
//! When a conversation grows past the model's context window (or hits
//! a `keep_recent_tokens` reserve), the actor pauses to summarize the
//! oldest messages and replaces them with a `<context-summary>`
//! block. Split-turn detection ensures a partially-summarized turn
//! gets its prefix described separately so the kept assistant
//! message reads coherently.

use std::sync::Arc;

use futures::StreamExt;
use tau_ai::{Content, Message};
use tokio_util::sync::CancellationToken;

use crate::core::config::AgentConfig;
use crate::core::transport::{AgentRunConfig, Transport};
use crate::types::events::AgentEvent;

// `CompactionReason` is the event payload — re-exported for ergonomic
// imports from this module.
pub use crate::types::events::CompactionReason;

#[derive(Debug, Clone)]
pub struct CompactionConfig {
    pub enabled: bool,
    /// Trigger proactive compaction when `(input + cache_read)` is
    /// within this many tokens of the model's `context_window`.
    pub reserve_tokens: u64,
    /// Lower bound on how many tokens of recent messages survive a
    /// compaction pass — the cut-point search walks back until it has
    /// accumulated at least this much, then continues to the next
    /// message boundary.
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

pub struct CompactionResult {
    /// Summary text. Wrapped in `<context-summary>` markers and
    /// prepended to the kept messages by [`apply_compaction_result`].
    pub summary: String,
    /// Index of the first message to keep (everything before it gets
    /// summarized).
    pub first_kept_index: usize,
    /// Total estimated tokens before compaction (for the
    /// `CompactionEnd` event).
    pub tokens_before: u64,
    pub read_files: Vec<String>,
    pub modified_files: Vec<String>,
}

// ─── Token estimation (char/4 heuristic) ─────────────────────────────

pub fn estimate_tokens(message: &Message) -> u64 {
    let char_count: usize = match message {
        Message::User { content, .. }
        | Message::Assistant { content, .. }
        | Message::ToolResult { content, .. }
        | Message::SystemInjection { content, .. } => content_char_count(content),
    };
    (char_count / 4) as u64
}

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

// ─── Cut-point finding ───────────────────────────────────────────────

struct CutPointResult {
    first_kept_index: usize,
    turn_start_index: Option<usize>,
    is_split_turn: bool,
}

/// Walk backward through messages, accumulating tokens, to find a
/// boundary where the kept-suffix exceeds `keep_recent_tokens`. Then
/// advance past any leading `ToolResult` messages so the kept slice
/// starts at a user or assistant boundary.
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

    // If the rev-loop never crossed the threshold, fall back to
    // "keep the last two messages" — better than failing entirely.
    if cut_index >= messages.len() {
        cut_index = messages.len().saturating_sub(2);
    }

    // Bail if we'd only summarize one message: catches both
    // "loop broke at i=0" and "fallback on too-small history."
    if cut_index <= 1 {
        return None;
    }

    // Don't start the kept slice on a tool result without its
    // preceding assistant call.
    let mut first_kept = cut_index;
    while first_kept < messages.len() {
        if cancel.is_cancelled() {
            return None;
        }
        match &messages[first_kept] {
            Message::User { .. } | Message::SystemInjection { .. } => break,
            Message::Assistant { .. } => break,
            Message::ToolResult { .. } => first_kept += 1,
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
    let Message::Assistant { content, .. } = &messages[idx] else {
        return false;
    };
    let has_tool_calls = content
        .iter()
        .any(|c| matches!(c, Content::ToolCall { .. }));
    has_tool_calls
        && idx + 1 < messages.len()
        && matches!(&messages[idx + 1], Message::ToolResult { .. })
}

/// Walk backward from `from` past `ToolResult` + `Assistant` pairs
/// until we hit a user / system-injection boundary or the start.
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

// ─── Serialization for the summarization prompt ──────────────────────

fn serialize_messages_for_summary(messages: &[Message]) -> String {
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

// ─── File-operation extraction for the summary metadata ──────────────

const READ_TOOLS: &[&str] = &["read", "glob", "grep", "list"];
const WRITE_TOOLS: &[&str] = &["write", "edit"];

fn extract_file_operations(messages: &[Message]) -> (Vec<String>, Vec<String>) {
    let mut read_files = Vec::new();
    let mut modified_files = Vec::new();
    for msg in messages {
        let Message::Assistant { content, .. } = msg else {
            continue;
        };
        for c in content {
            let Content::ToolCall {
                name, arguments, ..
            } = c
            else {
                continue;
            };
            let n = name.as_str();
            if READ_TOOLS.contains(&n) {
                if let Some(p) = arguments.get("path").and_then(|v| v.as_str()) {
                    if !read_files.contains(&p.to_string()) {
                        read_files.push(p.into());
                    }
                }
            } else if WRITE_TOOLS.contains(&n) {
                for key in ["path", "file_path"] {
                    if let Some(p) = arguments.get(key).and_then(|v| v.as_str()) {
                        if !modified_files.contains(&p.to_string()) {
                            modified_files.push(p.into());
                        }
                    }
                }
            }
        }
    }
    (read_files, modified_files)
}

// ─── Prompts ─────────────────────────────────────────────────────────

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

// ─── Entry point ─────────────────────────────────────────────────────

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
        return Err("Compaction cancelled".into());
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

    let prompt = if let Some(prev) = previous_summary {
        UPDATE_SUMMARIZATION_PROMPT
            .replace("{previous_summary}", prev)
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
            let turn_prefix = &messages[turn_start..cut.first_kept_index];
            let turn_prefix_text = serialize_messages_for_summary(turn_prefix);
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
        return Err("Compaction cancelled".into());
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

async fn call_summarization_llm(
    prompt: &str,
    agent_config: &AgentConfig,
    transport: &Arc<dyn Transport>,
    cancel: &CancellationToken,
) -> Result<String, String> {
    let run_config = AgentRunConfig {
        system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.into()),
        tools: vec![],
        server_tools: vec![],
        model: agent_config.model.clone(),
        reasoning: None,
        thinking_adaptive: false,
        max_tokens: Some(4096),
        temperature: None,
        // Summarization is a one-shot call, not part of any turn loop.
        turn_number: 0,
        cache_scope: None,
        cache_ttl: None,
        system_prompt_boundary: None,
    };

    let user_message = Message::user(prompt);
    let mut stream = transport
        .run(vec![user_message], &run_config, cancel.clone())
        .await
        .map_err(|e| format!("Compaction LLM call failed: {e}"))?;

    let mut result_text = String::new();
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::MessageEnd { message } => result_text = message.text(),
            AgentEvent::Error { message } => {
                return Err(format!("Compaction LLM error: {message}"));
            }
            _ => {}
        }
    }

    if result_text.is_empty() {
        return Err("Compaction LLM returned empty response".into());
    }
    Ok(result_text)
}

/// Apply a successful compaction result to a conversation: splice off
/// the summarized prefix, prepend a `<context-summary>` user message,
/// keep the suffix.
pub fn apply_compaction_result(
    messages: &mut Vec<Message>,
    previous_summary: &mut Option<String>,
    result: CompactionResult,
) {
    *previous_summary = Some(result.summary.clone());
    let kept = messages.split_off(result.first_kept_index);
    *messages = vec![Message::user(format!(
        "<context-summary>\n{}\n</context-summary>\n\nThe conversation was compacted. Continue from where we left off.",
        result.summary
    ))];
    messages.extend(kept);
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
    fn assistant_tool(name: &str) -> Message {
        Message::Assistant {
            content: vec![Content::tool_call("id", name, serde_json::json!({}))],
            metadata: AssistantMetadata::default(),
        }
    }
    fn tool_result(tool_call_id: &str) -> Message {
        Message::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: "test".into(),
            content: vec![Content::text("result")],
            is_error: false,
            timestamp: 0,
        }
    }

    #[test]
    fn estimate_tokens_char_quarter() {
        // 12 chars / 4 = 3 tokens
        assert_eq!(estimate_tokens(&user("Hello world!")), 3);
    }

    #[test]
    fn cut_point_too_few_messages_returns_none() {
        let messages = vec![user("hi")];
        let cancel = CancellationToken::new();
        assert!(find_cut_point(&messages, 100, &cancel).is_none());
    }

    #[test]
    fn find_turn_start_walks_back_multi_step() {
        // user → assistant(tool) → result → assistant(tool) → result
        // from = 3 (second assistant) should find turn start at 1.
        let messages = vec![
            user("do"),
            assistant_tool("read"),
            tool_result("id"),
            assistant_tool("write"),
            tool_result("id"),
        ];
        assert_eq!(find_turn_start(&messages, 3), 1);
    }

    #[test]
    fn find_turn_start_stops_at_user_boundary() {
        let messages = vec![
            user("X"),
            assistant_tool("read"),
            tool_result("id"),
            user("Y"),
            assistant_tool("write"),
            tool_result("id"),
            assistant_tool("edit"),
            tool_result("id"),
        ];
        assert_eq!(find_turn_start(&messages, 6), 4);
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

    #[test]
    fn apply_compaction_replaces_prefix() {
        let mut messages = vec![
            user("old 1"),
            assistant("old 2"),
            user("recent 1"),
            assistant("recent 2"),
        ];
        let mut prev = None;
        apply_compaction_result(
            &mut messages,
            &mut prev,
            CompactionResult {
                summary: "Summary of old conversation".into(),
                first_kept_index: 2,
                tokens_before: 1000,
                read_files: vec![],
                modified_files: vec![],
            },
        );
        // summary + 2 recent
        assert_eq!(messages.len(), 3);
        assert!(messages[0].text().contains("context-summary"));
        assert_eq!(messages[1].text(), "recent 1");
        assert_eq!(messages[2].text(), "recent 2");
        assert_eq!(prev.as_deref(), Some("Summary of old conversation"));
    }
}
