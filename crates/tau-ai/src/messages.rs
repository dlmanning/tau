//! Message normalization and repair utilities.
//!
//! Ensures tool_use/tool_result pairing is valid before sending to the API.

use std::collections::{HashMap, HashSet};

use crate::types::{Content, Message};

const SYNTHETIC_TOOL_RESULT_PLACEHOLDER: &str =
    "Tool execution was interrupted. No result available.";

/// Ensure every tool_use has a matching tool_result and vice versa.
///
/// This prevents API errors caused by:
/// - Orphaned tool_results (no matching tool_use) — removed
/// - Orphaned tool_uses (no matching tool_result) — synthetic error result injected
/// - Duplicate tool_results (same tool_use_id twice) — duplicates removed
pub fn ensure_tool_result_pairing(messages: &mut Vec<Message>) {
    // Pass 1: Collect all tool_use IDs from assistant messages, with their names
    let mut tool_use_info: HashMap<String, String> = HashMap::new(); // id -> name
    for msg in messages.iter() {
        if let Message::Assistant { content, .. } = msg {
            for c in content {
                if let Content::ToolCall { id, name, .. } = c {
                    tool_use_info.insert(id.clone(), name.clone());
                }
            }
        }
    }

    // Pass 2: Remove orphaned and duplicate tool_results
    let mut seen_result_ids: HashSet<String> = HashSet::new();
    messages.retain(|msg| {
        if let Message::ToolResult { tool_call_id, .. } = msg {
            if !tool_use_info.contains_key(tool_call_id) {
                return false; // orphaned — no matching tool_use
            }
            if !seen_result_ids.insert(tool_call_id.clone()) {
                return false; // duplicate
            }
        }
        true
    });

    // Pass 3: Find tool_uses missing results and inject synthetics.
    // Walk messages and after each assistant turn's last tool_result,
    // insert synthetics for any missing IDs from that turn.
    let mut result_ids: HashSet<String> = HashSet::new();
    for msg in messages.iter() {
        if let Message::ToolResult { tool_call_id, .. } = msg {
            result_ids.insert(tool_call_id.clone());
        }
    }

    let mut i = 0;
    while i < messages.len() {
        if let Message::Assistant { content, .. } = &messages[i] {
            let turn_tool_ids: Vec<(String, String)> = content
                .iter()
                .filter_map(|c| {
                    if let Content::ToolCall { id, name, .. } = c {
                        Some((id.clone(), name.clone()))
                    } else {
                        None
                    }
                })
                .collect();

            if !turn_tool_ids.is_empty() {
                // Find the insertion point: after the last consecutive tool_result
                let mut insert_at = i + 1;
                while insert_at < messages.len() {
                    if matches!(&messages[insert_at], Message::ToolResult { .. }) {
                        insert_at += 1;
                    } else {
                        break;
                    }
                }

                // Insert synthetics for missing tool_use IDs
                for (id, name) in &turn_tool_ids {
                    if !result_ids.contains(id) {
                        messages.insert(
                            insert_at,
                            Message::tool_result(
                                id,
                                name,
                                vec![Content::text(SYNTHETIC_TOOL_RESULT_PLACEHOLDER)],
                                true,
                            ),
                        );
                        result_ids.insert(id.clone());
                        insert_at += 1;
                    }
                }
            }
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AssistantMetadata, Content, Message};

    fn user_msg(text: &str) -> Message {
        Message::user(text)
    }

    fn assistant_with_tools(tools: &[(&str, &str)]) -> Message {
        Message::Assistant {
            content: tools
                .iter()
                .map(|(id, name)| Content::ToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments: serde_json::json!({}),
                })
                .collect(),
            metadata: AssistantMetadata::default(),
        }
    }

    fn assistant_with_tool(id: &str, name: &str) -> Message {
        assistant_with_tools(&[(id, name)])
    }

    fn tool_result(id: &str, name: &str) -> Message {
        Message::tool_result(id, name, vec![Content::text("result")], false)
    }

    #[test]
    fn test_valid_pairing_unchanged() {
        let mut msgs = vec![
            user_msg("hello"),
            assistant_with_tool("c1", "bash"),
            tool_result("c1", "bash"),
        ];
        ensure_tool_result_pairing(&mut msgs);
        assert_eq!(msgs.len(), 3);
    }

    #[test]
    fn test_orphaned_tool_result_removed() {
        let mut msgs = vec![
            user_msg("hello"),
            tool_result("c_nonexistent", "bash"),
            assistant_with_tool("c1", "bash"),
            tool_result("c1", "bash"),
        ];
        ensure_tool_result_pairing(&mut msgs);
        assert_eq!(msgs.len(), 3);
        assert!(matches!(msgs[0], Message::User { .. }));
    }

    #[test]
    fn test_orphaned_tool_use_gets_synthetic_result() {
        let mut msgs = vec![
            user_msg("hello"),
            assistant_with_tool("c1", "bash"),
            // Missing tool_result for c1
        ];
        ensure_tool_result_pairing(&mut msgs);
        assert_eq!(msgs.len(), 3);
        if let Message::ToolResult {
            tool_call_id,
            is_error,
            ..
        } = &msgs[2]
        {
            assert_eq!(tool_call_id, "c1");
            assert!(is_error);
        } else {
            panic!("Expected synthetic tool result");
        }
    }

    #[test]
    fn test_duplicate_tool_result_removed() {
        let mut msgs = vec![
            user_msg("hello"),
            assistant_with_tool("c1", "bash"),
            tool_result("c1", "bash"),
            tool_result("c1", "bash"), // duplicate
        ];
        ensure_tool_result_pairing(&mut msgs);
        assert_eq!(msgs.len(), 3);
    }

    #[test]
    fn test_multiple_tool_calls_all_paired() {
        let mut msgs = vec![
            user_msg("hello"),
            assistant_with_tool("c1", "bash"),
            tool_result("c1", "bash"),
            assistant_with_tool("c2", "read"),
            // Missing tool_result for c2
        ];
        ensure_tool_result_pairing(&mut msgs);
        assert_eq!(msgs.len(), 5);
        assert!(matches!(msgs[4], Message::ToolResult { .. }));
    }

    #[test]
    fn test_multi_tool_assistant_partial_results() {
        // Assistant calls c1 and c2, but only c1 has a result
        let mut msgs = vec![
            user_msg("hello"),
            assistant_with_tools(&[("c1", "bash"), ("c2", "read")]),
            tool_result("c1", "bash"),
            // c2 is missing
            user_msg("thanks"),
        ];
        ensure_tool_result_pairing(&mut msgs);
        assert_eq!(msgs.len(), 5);
        // Synthetic for c2 should be after c1's result, before "thanks"
        assert!(matches!(&msgs[1], Message::Assistant { .. }));
        assert!(matches!(&msgs[2], Message::ToolResult { .. })); // c1 real
        if let Message::ToolResult {
            tool_call_id,
            is_error,
            ..
        } = &msgs[3]
        {
            assert_eq!(tool_call_id, "c2");
            assert!(is_error); // c2 synthetic
        } else {
            panic!("Expected synthetic tool result for c2 at index 3");
        }
        assert!(matches!(&msgs[4], Message::User { .. })); // "thanks"
    }

    #[test]
    fn test_multi_tool_both_missing() {
        let mut msgs = vec![
            user_msg("hello"),
            assistant_with_tools(&[("c1", "bash"), ("c2", "read")]),
            // Both missing
        ];
        ensure_tool_result_pairing(&mut msgs);
        assert_eq!(msgs.len(), 4);
        assert!(matches!(&msgs[2], Message::ToolResult { .. }));
        assert!(matches!(&msgs[3], Message::ToolResult { .. }));
    }
}
