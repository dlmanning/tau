//! Anthropic SSE streaming event consumption

use async_stream::stream;
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde::Deserialize;

use super::convert::map_stop_reason;
use crate::{
    stream::MessageEvent,
    types::{Api, Content, Message, Model, StopReason, Usage},
};

// ============================================================================
// Internal content block tracking
// ============================================================================

#[derive(Debug, Default)]
pub(super) enum ContentBlock {
    #[default]
    Empty,
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
    RedactedThinking {
        data: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments_json: String,
    },
    ServerToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

// ============================================================================
// SSE event deserialization types
// ============================================================================

#[derive(Debug, Deserialize)]
pub(super) struct MessageStartEvent {
    pub message: MessageInfo,
}

#[derive(Debug, Deserialize)]
pub(super) struct MessageInfo {
    pub usage: UsageInfo,
}

#[derive(Debug, Deserialize)]
pub(super) struct UsageInfo {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cache_read_input_tokens: Option<u32>,
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    pub thinking_output_tokens: Option<u32>,
    #[serde(default)]
    pub cache_creation: Option<CacheCreationUsage>,
    #[serde(default)]
    pub service_tier: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct CacheCreationUsage {
    pub ephemeral_1h_input_tokens: u32,
    pub ephemeral_5m_input_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub(super) struct ContentBlockStartEvent {
    pub index: u32,
    pub content_block: ContentBlockInfo,
}

#[derive(Debug, Deserialize)]
pub(super) struct ContentBlockInfo {
    #[serde(rename = "type")]
    pub block_type: String,
    pub id: Option<String>,
    pub name: Option<String>,
    pub data: Option<String>,
    pub input: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ContentBlockDeltaEvent {
    pub index: u32,
    pub delta: DeltaInfo,
}

#[derive(Debug, Deserialize)]
pub(super) struct DeltaInfo {
    #[serde(rename = "type")]
    pub delta_type: String,
    pub text: Option<String>,
    pub thinking: Option<String>,
    pub partial_json: Option<String>,
    pub signature: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ContentBlockStopEvent {
    pub index: u32,
    #[serde(default)]
    pub content_block: Option<ContentBlockStopInfo>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ContentBlockStopInfo {
    pub signature: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct MessageDeltaEvent {
    pub delta: MessageDelta,
    pub usage: UsageInfo,
}

#[derive(Debug, Deserialize)]
pub(super) struct MessageDelta {
    pub stop_reason: Option<String>,
    #[allow(dead_code)]
    pub stop_sequence: Option<String>,
    #[allow(dead_code)]
    pub stop_details: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ErrorEvent {
    pub error: ApiError,
}

#[derive(Debug, Deserialize)]
pub(super) struct ApiError {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    pub error_type: String,
    pub message: String,
}

// ============================================================================
// Stream creation
// ============================================================================

/// Create the event stream from SSE events
pub(super) fn create_stream(
    mut event_source: EventSource,
    model: Model,
) -> impl futures::Stream<Item = MessageEvent> {
    stream! {
        let mut usage = Usage::default();
        let mut stop_reason = StopReason::Stop;
        let mut content_blocks: Vec<ContentBlock> = vec![];
        let mut error_message: Option<String> = None;

        // Yield start event
        yield MessageEvent::Start {
            message: Message::assistant_empty(),
        };

        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => {}
                Ok(Event::Message(message)) => {
                    if message.event == "message_start" {
                        if let Ok(data) = serde_json::from_str::<MessageStartEvent>(&message.data) {
                            usage.input = data.message.usage.input_tokens;
                            usage.output = data.message.usage.output_tokens;
                            usage.cache_read = data.message.usage.cache_read_input_tokens.unwrap_or(0);
                            usage.cache_write = data.message.usage.cache_creation_input_tokens.unwrap_or(0);
                            usage.thinking = data.message.usage.thinking_output_tokens.unwrap_or(0);
                            if let Some(ref cc) = data.message.usage.cache_creation {
                                usage.cache_creation_1h = cc.ephemeral_1h_input_tokens;
                                usage.cache_creation_5m = cc.ephemeral_5m_input_tokens;
                            }
                            usage.service_tier = data.message.usage.service_tier.clone();
                        }
                    } else if message.event == "content_block_start" {
                        if let Ok(data) = serde_json::from_str::<ContentBlockStartEvent>(&message.data) {
                            let index = data.index as usize;
                            while content_blocks.len() <= index {
                                content_blocks.push(ContentBlock::default());
                            }

                            match data.content_block.block_type.as_str() {
                                "text" => {
                                    content_blocks[index] = ContentBlock::Text {
                                        text: String::new(),
                                    };
                                    yield MessageEvent::TextStart { content_index: index };
                                }
                                "thinking" => {
                                    content_blocks[index] = ContentBlock::Thinking {
                                        thinking: String::new(),
                                        signature: None,
                                    };
                                    yield MessageEvent::ThinkingStart { content_index: index };
                                }
                                "tool_use" => {
                                    let id = data.content_block.id.unwrap_or_default();
                                    let name = data.content_block.name.unwrap_or_default();
                                    content_blocks[index] = ContentBlock::ToolCall {
                                        id: id.clone(),
                                        name: name.clone(),
                                        arguments_json: String::new(),
                                    };
                                    yield MessageEvent::ToolCallStart {
                                        content_index: index,
                                        id,
                                        name,
                                    };
                                }
                                "redacted_thinking" => {
                                    content_blocks[index] = ContentBlock::RedactedThinking {
                                        data: data.content_block.data.unwrap_or_default(),
                                    };
                                }
                                "server_tool_use" => {
                                    let id = data.content_block.id.unwrap_or_default();
                                    let name = data.content_block.name.unwrap_or_default();
                                    let input = data.content_block.input.unwrap_or(serde_json::Value::Null);
                                    content_blocks[index] = ContentBlock::ServerToolUse { id, name, input };
                                }
                                _ => {}
                            }
                        }
                    } else if message.event == "content_block_delta" {
                        if let Ok(data) = serde_json::from_str::<ContentBlockDeltaEvent>(&message.data) {
                            let index = data.index as usize;
                            if index < content_blocks.len() {
                                match data.delta.delta_type.as_str() {
                                    "text_delta" => {
                                        if let ContentBlock::Text { ref mut text } = content_blocks[index] {
                                            let delta = data.delta.text.unwrap_or_default();
                                            text.push_str(&delta);
                                            yield MessageEvent::TextDelta {
                                                content_index: index,
                                                delta,
                                            };
                                        }
                                    }
                                    "thinking_delta" => {
                                        if let ContentBlock::Thinking { ref mut thinking, .. } = content_blocks[index] {
                                            let delta = data.delta.thinking.unwrap_or_default();
                                            thinking.push_str(&delta);
                                            yield MessageEvent::ThinkingDelta {
                                                content_index: index,
                                                delta,
                                            };
                                        }
                                    }
                                    "input_json_delta" => {
                                        if let ContentBlock::ToolCall { ref mut arguments_json, .. } = content_blocks[index] {
                                            let delta = data.delta.partial_json.unwrap_or_default();
                                            arguments_json.push_str(&delta);
                                            yield MessageEvent::ToolCallDelta {
                                                content_index: index,
                                                delta,
                                            };
                                        }
                                    }
                                    "signature_delta" => {
                                        if let ContentBlock::Thinking { ref mut signature, .. } = content_blocks[index] {
                                            let sig = data.delta.signature.unwrap_or_default();
                                            match signature {
                                                Some(s) => s.push_str(&sig),
                                                None => *signature = Some(sig),
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    } else if message.event == "content_block_stop" {
                        if let Ok(data) = serde_json::from_str::<ContentBlockStopEvent>(&message.data) {
                            let index = data.index as usize;
                            if index < content_blocks.len() {
                                // Capture thinking signature from content_block_stop event
                                let stop_signature = data.content_block
                                    .as_ref()
                                    .and_then(|cb| cb.signature.clone());
                                if let (Some(sig), ContentBlock::Thinking { signature, .. }) =
                                    (stop_signature, &mut content_blocks[index])
                                {
                                    *signature = Some(sig);
                                }

                                match &content_blocks[index] {
                                    ContentBlock::Text { text } => {
                                        yield MessageEvent::TextEnd {
                                            content_index: index,
                                            text: text.clone(),
                                        };
                                    }
                                    ContentBlock::Thinking { thinking, signature } => {
                                        yield MessageEvent::ThinkingEnd {
                                            content_index: index,
                                            thinking: thinking.clone(),
                                            signature: signature.clone(),
                                        };
                                    }
                                    ContentBlock::ToolCall { id, name, arguments_json } => {
                                        let arguments = serde_json::from_str(arguments_json)
                                            .unwrap_or(serde_json::Value::Null);
                                        yield MessageEvent::ToolCallEnd {
                                            content_index: index,
                                            id: id.clone(),
                                            name: name.clone(),
                                            arguments,
                                        };
                                    }
                                    _ => {}
                                }
                            }
                        }
                    } else if message.event == "message_delta" {
                        if let Ok(data) = serde_json::from_str::<MessageDeltaEvent>(&message.data) {
                            if let Some(reason) = data.delta.stop_reason {
                                stop_reason = map_stop_reason(&reason);
                            }
                            usage.input = data.usage.input_tokens;
                            usage.output = data.usage.output_tokens;
                            usage.cache_read = data.usage.cache_read_input_tokens.unwrap_or(0);
                            usage.cache_write = data.usage.cache_creation_input_tokens.unwrap_or(0);
                            usage.thinking = data.usage.thinking_output_tokens.unwrap_or(0);
                            if let Some(ref cc) = data.usage.cache_creation {
                                usage.cache_creation_1h = cc.ephemeral_1h_input_tokens;
                                usage.cache_creation_5m = cc.ephemeral_5m_input_tokens;
                            }
                            usage.service_tier = data.usage.service_tier.clone();
                        }
                    } else if message.event == "message_stop" {
                        break;
                    } else if message.event == "error" {
                        if let Ok(data) = serde_json::from_str::<ErrorEvent>(&message.data) {
                            error_message = Some(data.error.message);
                            stop_reason = StopReason::Error;
                        }
                        break;
                    }
                }
                Err(e) => {
                    error_message = Some(e.to_string());
                    stop_reason = StopReason::Error;
                    break;
                }
            }
        }

        // Build final message
        let content: Vec<Content> = content_blocks
            .into_iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(Content::Text { text }),
                ContentBlock::Thinking { thinking, signature } => Some(Content::Thinking { thinking, signature }),
                ContentBlock::RedactedThinking { data } => Some(Content::RedactedThinking { data }),
                ContentBlock::ToolCall { id, name, arguments_json } => {
                    let arguments = serde_json::from_str(&arguments_json)
                        .unwrap_or(serde_json::Value::Null);
                    Some(Content::ToolCall { id, name, arguments })
                }
                ContentBlock::ServerToolUse { id, name, input } => {
                    Some(Content::ServerToolUse { id, name, input })
                }
                ContentBlock::Empty => None,
            })
            .collect();

        let final_message = Message::Assistant {
            content,
            metadata: crate::types::AssistantMetadata {
                api: Some(Api::AnthropicMessages),
                provider: Some(model.provider),
                model: Some(model.id.clone()),
                usage: usage.clone(),
                stop_reason: Some(stop_reason),
                error_message: error_message.clone(),
                timestamp: chrono::Utc::now().timestamp_millis(),
            },
        };

        if let Some(error_msg) = error_message {
            yield MessageEvent::Error { message: error_msg };
        } else {
            yield MessageEvent::Done {
                message: final_message,
                stop_reason,
                usage,
            };
        }
    }
}
