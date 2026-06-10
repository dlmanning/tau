//! Cut-point (split-turn) detection.
//!
//! Pure logic — no transport. Walks the conversation backward to find
//! the boundary between "summarize this prefix" and "keep this suffix",
//! then classifies whether the boundary lands mid-turn.

use tau_ai::{Content, Message};
use tokio_util::sync::CancellationToken;

use super::estimate_tokens;

pub(super) struct CutPointResult {
    pub(super) first_kept_index: usize,
    pub(super) turn_start_index: Option<usize>,
    pub(super) is_split_turn: bool,
}

/// Walk backward through messages, accumulating tokens, to find a
/// boundary where the kept-suffix exceeds `keep_recent_tokens`. Then
/// advance past any leading `ToolResult` messages so the kept slice
/// starts at a user or assistant boundary.
pub(super) fn find_cut_point(
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
pub(super) fn find_turn_start(messages: &[Message], from: usize) -> usize {
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
}
