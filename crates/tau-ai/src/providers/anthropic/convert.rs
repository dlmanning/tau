//! Message and tool conversion for the Anthropic API

use serde::Serialize;

use super::request::{AnthropicMessage, AnthropicTool, SystemBlock};
use super::CacheScope;
use crate::messages::ensure_tool_result_pairing;
use crate::types::{Content, Message, StopReason, Tool};

/// Cache control configuration for prompt caching
#[derive(Debug, Clone, Serialize)]
pub(super) struct CacheControl {
    #[serde(rename = "type")]
    pub control_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<CacheScope>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

/// Build a CacheControl with the given scope and TTL options
pub(super) fn make_cache_control(scope: &Option<CacheScope>, ttl: &Option<String>) -> CacheControl {
    CacheControl {
        control_type: "ephemeral".to_string(),
        scope: scope.clone(),
        ttl: ttl.clone(),
    }
}

/// Split a system prompt at a dynamic boundary marker.
///
/// When a boundary is found and the caller opted into global caching (`cache_scope: Global`),
/// the static content before the boundary gets global scope and the dynamic content after
/// gets no caching. Without global scope, the static part uses the caller's scope.
///
/// Returns at least one block as long as the prompt is non-empty.
pub(super) fn split_system_prompt(
    prompt: &str,
    boundary: Option<&str>,
    scope: &Option<CacheScope>,
    ttl: &Option<String>,
) -> Vec<SystemBlock> {
    if let Some(marker) = boundary {
        if let Some(pos) = prompt.find(marker) {
            let static_part = prompt[..pos].trim();
            let dynamic_part = prompt[pos + marker.len()..].trim();
            let mut blocks = vec![];
            if !static_part.is_empty() {
                blocks.push(SystemBlock {
                    block_type: "text".to_string(),
                    text: static_part.to_string(),
                    cache_control: Some(make_cache_control(scope, ttl)),
                });
            }
            if !dynamic_part.is_empty() {
                blocks.push(SystemBlock {
                    block_type: "text".to_string(),
                    text: dynamic_part.to_string(),
                    cache_control: None,
                });
            }
            if !blocks.is_empty() {
                return blocks;
            }
        }
    }
    // No boundary, not found, or both parts empty — single block with configured scope
    vec![SystemBlock {
        block_type: "text".to_string(),
        text: prompt.to_string(),
        cache_control: Some(make_cache_control(scope, ttl)),
    }]
}

pub(super) fn convert_messages(
    messages: &[Message],
    cache_breakpoint_budget: usize,
    cache_scope: &Option<CacheScope>,
    cache_ttl: &Option<String>,
) -> Vec<AnthropicMessage> {
    // Repair tool_use/tool_result pairing before conversion
    let mut messages = messages.to_vec();
    ensure_tool_result_pairing(&mut messages);

    let mut result = vec![];

    for message in &messages {
        match message {
            Message::User { content, .. } => {
                let blocks: Vec<serde_json::Value> = content
                    .iter()
                    .map(|c| match c {
                        Content::Text { text } => {
                            serde_json::json!({ "type": "text", "text": text })
                        }
                        Content::Image { data, mime_type } => {
                            serde_json::json!({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": data
                                }
                            })
                        }
                        _ => serde_json::json!({ "type": "text", "text": "" }),
                    })
                    .collect();

                result.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: serde_json::Value::Array(blocks),
                });
            }
            Message::Assistant { content, .. } => {
                let blocks: Vec<serde_json::Value> = content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text } => {
                            Some(serde_json::json!({ "type": "text", "text": text }))
                        }
                        Content::Thinking {
                            thinking,
                            signature,
                            ..
                        } => {
                            let mut block =
                                serde_json::json!({ "type": "thinking", "thinking": thinking });
                            if let Some(sig) = signature {
                                block["signature"] = serde_json::json!(sig);
                            }
                            Some(block)
                        }
                        Content::ToolCall {
                            id,
                            name,
                            arguments,
                        } => Some(serde_json::json!({
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": arguments
                        })),
                        Content::RedactedThinking { data } => {
                            Some(serde_json::json!({ "type": "redacted_thinking", "data": data }))
                        }
                        Content::ServerToolUse { id, name, input } => {
                            Some(serde_json::json!({ "type": "server_tool_use", "id": id, "name": name, "input": input }))
                        }
                        Content::Image { .. } | Content::ServerToolResult { .. } => None,
                    })
                    .collect();

                if !blocks.is_empty() {
                    result.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: serde_json::Value::Array(blocks),
                    });
                }
            }
            Message::ToolResult {
                tool_call_id,
                content,
                is_error,
                ..
            } => {
                let text_content: String = content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let tool_result = serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": text_content,
                    "is_error": is_error
                });

                result.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: serde_json::Value::Array(vec![tool_result]),
                });
            }
            Message::SystemInjection { content, source } => {
                // Convert to user message with source context prefix
                let prefix = match source {
                    crate::types::InjectionSource::SubagentCompleted { description, .. } => {
                        format!("[Subagent \"{}\" completed]\n", description)
                    }
                    crate::types::InjectionSource::SubagentFailed { description, .. } => {
                        format!("[Subagent \"{}\" failed]\n", description)
                    }
                };
                let text: String = content
                    .iter()
                    .filter_map(|c| c.as_text())
                    .collect::<Vec<_>>()
                    .join("\n");
                let block = serde_json::json!({ "type": "text", "text": format!("{}{}", prefix, text) });
                result.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: serde_json::Value::Array(vec![block]),
                });
            }
        }
    }

    // Consolidate consecutive messages with the same role.
    let mut consolidated: Vec<AnthropicMessage> = Vec::with_capacity(result.len());
    for msg in result {
        if let Some(last) = consolidated.last_mut() {
            if last.role == msg.role {
                if let serde_json::Value::Array(new_blocks) = msg.content {
                    last.content.as_array_mut().unwrap().extend(new_blocks);
                }
                continue;
            }
        }
        consolidated.push(msg);
    }

    // Add cache breakpoints to recent messages.
    if cache_breakpoint_budget > 0 {
        let total = consolidated.len();
        let cache_zone_start = total.saturating_sub(cache_breakpoint_budget);
        for msg in &mut consolidated[cache_zone_start..] {
            if let serde_json::Value::Array(ref mut blocks) = msg.content {
                if let Some(last_idx) = blocks
                    .iter()
                    .rposition(|b| b.get("type").and_then(|t| t.as_str()) != Some("thinking"))
                {
                    let mut cc = serde_json::json!({"type": "ephemeral"});
                    if let Some(scope) = cache_scope {
                        cc["scope"] = serde_json::to_value(scope).unwrap();
                    }
                    if let Some(ttl) = cache_ttl {
                        cc["ttl"] = serde_json::json!(ttl);
                    }
                    blocks[last_idx]["cache_control"] = cc;
                }
            }
        }
    }

    consolidated
}

pub(super) fn convert_tools(
    tools: &[Tool],
    cache_last: bool,
    cache_scope: &Option<CacheScope>,
    cache_ttl: &Option<String>,
) -> Vec<AnthropicTool> {
    let len = tools.len();
    tools
        .iter()
        .enumerate()
        .map(|(i, tool)| {
            let input_schema = if tool.parameters.is_object() {
                let mut schema = tool.parameters.clone();
                if let Some(obj) = schema.as_object_mut() {
                    obj.entry("type").or_insert(serde_json::json!("object"));
                }
                schema
            } else {
                serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                })
            };

            let cache_control = if cache_last && i == len - 1 {
                Some(make_cache_control(cache_scope, cache_ttl))
            } else {
                None
            };

            AnthropicTool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema,
                cache_control,
                strict: None,
                defer_loading: None,
                eager_input_streaming: None,
                tool_type: None,
            }
        })
        .collect()
}

pub(super) fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" => StopReason::Stop,
        "max_tokens" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        "stop_sequence" => StopReason::Stop,
        _ => StopReason::Stop,
    }
}
