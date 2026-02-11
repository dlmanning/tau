//! Anthropic Claude API provider

use crate::{
    error::{Error, Result},
    stream::{MessageEvent, MessageEventStream},
    types::{Api, Content, Context, Message, Model, StopReason, StreamOptions, Tool, Usage},
};
use async_stream::stream;
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};

/// Anthropic-specific streaming options
#[derive(Debug, Clone, Default)]
pub struct AnthropicOptions {
    /// Base streaming options
    pub base: StreamOptions,
    /// Enable extended thinking
    pub thinking_enabled: bool,
    /// Budget for thinking tokens
    pub thinking_budget_tokens: Option<u32>,
    /// Tool choice strategy
    pub tool_choice: Option<ToolChoice>,
}

/// Tool choice strategy
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    Any,
    None,
    Tool { name: String },
}

/// Anthropic API client
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider with an API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
        }
    }

    /// Create from environment variable
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| Error::InvalidApiKey)?;
        Ok(Self::new(api_key))
    }

    /// Stream a response from Claude
    pub async fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&AnthropicOptions>,
    ) -> Result<MessageEventStream> {
        let default_options = AnthropicOptions::default();
        let opts = options.unwrap_or(&default_options);

        let request = self.build_request(model, context, opts)?;
        let url = format!("{}/v1/messages", model.base_url);

        tracing::debug!("Anthropic API URL: {}", url);

        let is_oauth = self.api_key.contains("sk-ant-oat");
        let mut headers = reqwest::header::HeaderMap::new();

        if is_oauth {
            headers.insert(
                "Authorization",
                format!("Bearer {}", self.api_key).parse().unwrap(),
            );
            headers.insert(
                "anthropic-beta",
                "oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14"
                    .parse()
                    .unwrap(),
            );
            headers.insert(
                "anthropic-dangerous-direct-browser-access",
                "true".parse().unwrap(),
            );
        } else {
            headers.insert("x-api-key", self.api_key.parse().unwrap());
            headers.insert(
                "anthropic-beta",
                "fine-grained-tool-streaming-2025-05-14".parse().unwrap(),
            );
        }
        headers.insert("accept", "application/json".parse().unwrap());
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.insert("anthropic-version", "2023-06-01".parse().unwrap());

        // SDK identification headers (required for OAuth)
        headers.insert("User-Agent", "Anthropic/JS 0.52.0".parse().unwrap());
        headers.insert("X-Stainless-Lang", "js".parse().unwrap());
        headers.insert("X-Stainless-Package-Version", "0.52.0".parse().unwrap());
        headers.insert("X-Stainless-OS", "MacOS".parse().unwrap());
        headers.insert("X-Stainless-Arch", "arm64".parse().unwrap());
        headers.insert("X-Stainless-Runtime", "node".parse().unwrap());
        headers.insert("X-Stainless-Runtime-Version", "v22.0.0".parse().unwrap());
        headers.insert("X-Stainless-Retry-Count", "0".parse().unwrap());

        // Add model-specific headers
        for (key, value) in &model.headers {
            if let (Ok(name), Ok(val)) = (
                key.parse::<reqwest::header::HeaderName>(),
                value.parse::<reqwest::header::HeaderValue>(),
            ) {
                headers.insert(name, val);
            }
        }

        let request_builder = self
            .client
            .post(&url)
            .headers(headers.clone())
            .json(&request);

        let event_source = EventSource::new(request_builder)
            .map_err(|e| Error::Sse(format!("Failed to create event source: {}", e)))?;

        Ok(Box::pin(create_stream(event_source, model.clone())))
    }

    fn build_request(
        &self,
        model: &Model,
        context: &Context,
        options: &AnthropicOptions,
    ) -> Result<AnthropicRequest> {
        let messages = convert_messages(&context.messages);
        let tools = if context.tools.is_empty() {
            None
        } else {
            Some(convert_tools(&context.tools))
        };

        let max_tokens = options.base.max_tokens.unwrap_or(model.max_tokens / 3);

        let is_oauth = self.api_key.contains("sk-ant-oat");

        let mut request = AnthropicRequest {
            model: model.id.clone(),
            messages,
            max_tokens,
            stream: true,
            system: None,
            temperature: options.base.temperature,
            tools,
            tool_choice: options.tool_choice.clone(),
            thinking: None,
        };

        // Set system prompt - OAuth tokens MUST include Claude Code identity
        if is_oauth {
            let mut system_blocks = vec![SystemBlock {
                block_type: "text".to_string(),
                text: "You are Claude Code, Anthropic's official CLI for Claude.".to_string(),
                cache_control: Some(CacheControl {
                    control_type: "ephemeral".to_string(),
                }),
            }];
            if let Some(ref system_prompt) = context.system_prompt {
                system_blocks.push(SystemBlock {
                    block_type: "text".to_string(),
                    text: system_prompt.clone(),
                    cache_control: Some(CacheControl {
                        control_type: "ephemeral".to_string(),
                    }),
                });
            }
            request.system = Some(system_blocks);
        } else if let Some(ref system_prompt) = context.system_prompt {
            request.system = Some(vec![SystemBlock {
                block_type: "text".to_string(),
                text: system_prompt.clone(),
                cache_control: Some(CacheControl {
                    control_type: "ephemeral".to_string(),
                }),
            }]);
        }

        // Enable thinking if requested
        if options.thinking_enabled && model.reasoning {
            request.thinking = Some(ThinkingConfig {
                thinking_type: "enabled".to_string(),
                budget_tokens: options.thinking_budget_tokens.unwrap_or(1024),
            });
        }

        Ok(request)
    }
}

/// Create the event stream from SSE events
fn create_stream(
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
                                        if let ContentBlock::Thinking { ref mut thinking } = content_blocks[index] {
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
                                    _ => {}
                                }
                            }
                        }
                    } else if message.event == "content_block_stop" {
                        if let Ok(data) = serde_json::from_str::<ContentBlockStopEvent>(&message.data) {
                            let index = data.index as usize;
                            if index < content_blocks.len() {
                                match &content_blocks[index] {
                                    ContentBlock::Text { text } => {
                                        yield MessageEvent::TextEnd {
                                            content_index: index,
                                            text: text.clone(),
                                        };
                                    }
                                    ContentBlock::Thinking { thinking } => {
                                        yield MessageEvent::ThinkingEnd {
                                            content_index: index,
                                            thinking: thinking.clone(),
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
                        }
                    } else if message.event == "message_stop" {
                        // Stream complete
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
                ContentBlock::Thinking { thinking } => Some(Content::Thinking { thinking }),
                ContentBlock::ToolCall { id, name, arguments_json } => {
                    let arguments = serde_json::from_str(&arguments_json)
                        .unwrap_or(serde_json::Value::Null);
                    Some(Content::ToolCall { id, name, arguments })
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

// ============================================================================
// Internal types for content block tracking
// ============================================================================

#[derive(Debug, Default)]
enum ContentBlock {
    #[default]
    Empty,
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments_json: String,
    },
}

// ============================================================================
// Request types
// ============================================================================

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    max_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<SystemBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<ThinkingConfig>,
}

#[derive(Debug, Serialize)]
struct SystemBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Debug, Serialize)]
struct CacheControl {
    #[serde(rename = "type")]
    control_type: String,
}

#[derive(Debug, Serialize)]
struct ThinkingConfig {
    #[serde(rename = "type")]
    thinking_type: String,
    budget_tokens: u32,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

// ============================================================================
// Response event types
// ============================================================================

#[derive(Debug, Deserialize)]
struct MessageStartEvent {
    message: MessageInfo,
}

#[derive(Debug, Deserialize)]
struct MessageInfo {
    usage: UsageInfo,
}

#[derive(Debug, Deserialize)]
struct UsageInfo {
    input_tokens: u32,
    output_tokens: u32,
    cache_read_input_tokens: Option<u32>,
    cache_creation_input_tokens: Option<u32>,
    /// Extended thinking tokens (Claude reasoning)
    #[serde(default)]
    thinking_output_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ContentBlockStartEvent {
    index: u32,
    content_block: ContentBlockInfo,
}

#[derive(Debug, Deserialize)]
struct ContentBlockInfo {
    #[serde(rename = "type")]
    block_type: String,
    id: Option<String>,
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContentBlockDeltaEvent {
    index: u32,
    delta: DeltaInfo,
}

#[derive(Debug, Deserialize)]
struct DeltaInfo {
    #[serde(rename = "type")]
    delta_type: String,
    text: Option<String>,
    thinking: Option<String>,
    partial_json: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContentBlockStopEvent {
    index: u32,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaEvent {
    delta: MessageDelta,
    usage: UsageInfo,
}

#[derive(Debug, Deserialize)]
struct MessageDelta {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ErrorEvent {
    error: ApiError,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    error_type: String,
    message: String,
}

// ============================================================================
// Conversion functions
// ============================================================================

fn convert_messages(messages: &[Message]) -> Vec<AnthropicMessage> {
    let mut result = vec![];

    for message in messages {
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
                        Content::Thinking { thinking } => {
                            // Convert thinking to text block for compatibility
                            Some(serde_json::json!({
                                "type": "text",
                                "text": format!("<thinking>\n{}\n</thinking>", thinking)
                            }))
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
                        Content::Image { .. } => None,
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
        }
    }

    result
}

fn convert_tools(tools: &[Tool]) -> Vec<AnthropicTool> {
    tools
        .iter()
        .map(|tool| {
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

            AnthropicTool {
                name: tool.name.clone(),
                description: tool.description.clone(),
                input_schema,
            }
        })
        .collect()
}

fn map_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" => StopReason::Stop,
        "max_tokens" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        "stop_sequence" => StopReason::Stop,
        _ => StopReason::Stop,
    }
}

// ============================================================================
// Convenience function
// ============================================================================

/// Stream a response from Anthropic Claude
pub async fn stream_anthropic(
    model: &Model,
    context: &Context,
    options: Option<&AnthropicOptions>,
) -> Result<MessageEventStream> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| Error::InvalidApiKey)?;
    let provider = AnthropicProvider::new(api_key);
    provider.stream(model, context, options).await
}
