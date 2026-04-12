//! Anthropic SSE streaming event consumption

use async_stream::stream;
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde::Deserialize;

use super::convert::map_stop_reason;
use crate::{
    stream::{MessageEvent, StreamAccumulator},
    types::{Api, Model},
};

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
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    pub thinking_output_tokens: Option<u64>,
    #[serde(default)]
    pub cache_creation: Option<CacheCreationUsage>,
    #[serde(default)]
    pub service_tier: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct CacheCreationUsage {
    pub ephemeral_1h_input_tokens: u64,
    pub ephemeral_5m_input_tokens: u64,
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
    /// For server tool result blocks (e.g. web_search_tool_result)
    pub tool_use_id: Option<String>,
    /// Content of server tool results (search results array, etc.)
    pub content: Option<serde_json::Value>,
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

/// Create the event stream from SSE events
pub(super) fn create_stream(
    mut event_source: EventSource,
    model: Model,
) -> impl futures::Stream<Item = MessageEvent> {
    stream! {
        let (mut acc, start) = StreamAccumulator::new(
            Api::AnthropicMessages,
            model.provider,
            model.id.clone(),
        );
        yield start;

        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => {}
                Ok(Event::Message(message)) => {
                    if message.event == "message_start" {
                        if let Ok(data) = serde_json::from_str::<MessageStartEvent>(&message.data) {
                            apply_usage(acc.usage_mut(), &data.message.usage);
                        }
                    } else if message.event == "content_block_start" {
                        if let Ok(data) = serde_json::from_str::<ContentBlockStartEvent>(&message.data) {
                            let index = data.index as usize;
                            match data.content_block.block_type.as_str() {
                                "text" => {
                                    for ev in acc.text_start(index) { yield ev; }
                                }
                                "thinking" => {
                                    for ev in acc.thinking_start(index) { yield ev; }
                                }
                                "tool_use" => {
                                    let id = data.content_block.id.unwrap_or_default();
                                    let name = data.content_block.name.unwrap_or_default();
                                    for ev in acc.tool_call_start(index, &id, &name) { yield ev; }
                                }
                                "redacted_thinking" => {
                                    acc.add_redacted_thinking(
                                        index,
                                        data.content_block.data.unwrap_or_default(),
                                    );
                                }
                                "server_tool_use" => {
                                    for ev in acc.add_server_tool_use(
                                        index,
                                        data.content_block.id.unwrap_or_default(),
                                        data.content_block.name.unwrap_or_default(),
                                        data.content_block.input.unwrap_or(serde_json::Value::Null),
                                    ) { yield ev; }
                                }
                                "web_search_tool_result" | "server_tool_result" => {
                                    // Server tool result blocks arrive fully formed (no deltas)
                                    let tool_use_id = data.content_block.tool_use_id.unwrap_or_default();
                                    let content_val = data.content_block.content.unwrap_or(serde_json::Value::Null);
                                    for ev in acc.add_server_tool_result(
                                        index,
                                        tool_use_id,
                                        content_val,
                                        data.content_block.block_type,
                                    ) { yield ev; }
                                }
                                _ => {}
                            }
                        }
                    } else if message.event == "content_block_delta" {
                        if let Ok(data) = serde_json::from_str::<ContentBlockDeltaEvent>(&message.data) {
                            let index = data.index as usize;
                            match data.delta.delta_type.as_str() {
                                "text_delta" => {
                                    let delta = data.delta.text.unwrap_or_default();
                                    for ev in acc.text_delta(index, &delta) { yield ev; }
                                }
                                "thinking_delta" => {
                                    let delta = data.delta.thinking.unwrap_or_default();
                                    for ev in acc.thinking_delta(index, &delta) { yield ev; }
                                }
                                "input_json_delta" => {
                                    let delta = data.delta.partial_json.unwrap_or_default();
                                    for ev in acc.tool_call_delta(index, &delta) { yield ev; }
                                }
                                "signature_delta" => {
                                    let sig = data.delta.signature.unwrap_or_default();
                                    acc.thinking_signature_delta(index, &sig);
                                }
                                _ => {}
                            }
                        }
                    } else if message.event == "content_block_stop" {
                        if let Ok(data) = serde_json::from_str::<ContentBlockStopEvent>(&message.data) {
                            let index = data.index as usize;
                            let override_sig = data.content_block
                                .as_ref()
                                .and_then(|cb| cb.signature.clone());
                            for ev in acc.end_block(index, override_sig) { yield ev; }
                        }
                    } else if message.event == "message_delta" {
                        if let Ok(data) = serde_json::from_str::<MessageDeltaEvent>(&message.data) {
                            if let Some(reason) = data.delta.stop_reason {
                                acc.set_stop_reason(map_stop_reason(&reason));
                            }
                            apply_usage(acc.usage_mut(), &data.usage);
                        }
                    } else if message.event == "message_stop" {
                        break;
                    } else if message.event == "error" {
                        if let Ok(data) = serde_json::from_str::<ErrorEvent>(&message.data) {
                            acc.set_error(data.error.message);
                        }
                        break;
                    }
                }
                Err(e) => {
                    acc.set_error(e.to_string());
                    break;
                }
            }
        }

        for ev in acc.finish() { yield ev; }
    }
}

/// Apply Anthropic usage info to the accumulator's usage struct.
fn apply_usage(usage: &mut crate::types::Usage, info: &UsageInfo) {
    usage.input = info.input_tokens;
    usage.output = info.output_tokens;
    usage.cache_read = info.cache_read_input_tokens.unwrap_or(0);
    usage.cache_write = info.cache_creation_input_tokens.unwrap_or(0);
    usage.thinking = info.thinking_output_tokens.unwrap_or(0);
    if let Some(ref cc) = info.cache_creation {
        usage.cache_creation_1h = cc.ephemeral_1h_input_tokens;
        usage.cache_creation_5m = cc.ephemeral_5m_input_tokens;
    }
    usage.service_tier = info.service_tier.clone();
}
