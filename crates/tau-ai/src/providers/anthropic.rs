//! Anthropic Claude API provider

use async_stream::stream;
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    messages::ensure_tool_result_pairing,
    stream::{MessageEvent, MessageEventStream},
    types::{Api, Content, Context, Message, Model, StopReason, StreamOptions, Tool, Usage},
};

/// Cache scope for prompt caching
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheScope {
    /// Global scope — shared across all users/orgs (1P only)
    Global,
    /// Org scope — shared within an organization
    Org,
}

/// Anthropic-specific streaming options
#[derive(Debug, Clone, Default)]
pub struct AnthropicOptions {
    /// Base streaming options
    pub base: StreamOptions,
    /// Enable extended thinking
    pub thinking_enabled: bool,
    /// Use adaptive thinking (model decides when to think)
    pub thinking_adaptive: bool,
    /// Budget for thinking tokens (used when not adaptive)
    pub thinking_budget_tokens: Option<u32>,
    /// Tool choice strategy
    pub tool_choice: Option<ToolChoice>,
    /// Cache scope for prompt caching breakpoints
    pub cache_scope: Option<CacheScope>,
    /// Cache TTL (e.g. "1h", "5m")
    pub cache_ttl: Option<String>,
    /// Dynamic boundary marker for system prompt splitting
    pub system_prompt_boundary: Option<String>,
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

/// Build a CacheControl with the given scope and TTL options
fn make_cache_control(scope: &Option<CacheScope>, ttl: &Option<String>) -> CacheControl {
    CacheControl {
        control_type: "ephemeral".to_string(),
        scope: scope.clone(),
        ttl: ttl.clone(),
    }
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

        // Build beta headers based on features in use
        let mut betas = vec!["fine-grained-tool-streaming-2025-05-14"];
        if opts.thinking_enabled {
            betas.push("interleaved-thinking-2025-05-14");
        }
        if matches!(opts.cache_scope, Some(CacheScope::Global)) {
            betas.push("prompt-caching-scope-2026-01-05");
        }

        if is_oauth {
            betas.insert(0, "oauth-2025-04-20");
            headers.insert(
                "Authorization",
                format!("Bearer {}", self.api_key).parse().unwrap(),
            );
            headers.insert(
                "anthropic-dangerous-direct-browser-access",
                "true".parse().unwrap(),
            );
        } else {
            headers.insert("x-api-key", self.api_key.parse().unwrap());
        }
        headers.insert("anthropic-beta", betas.join(",").parse().unwrap());
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
        let is_oauth = self.api_key.contains("sk-ant-oat");
        let has_tools = !context.tools.is_empty();

        // Build system blocks first so we can count cache breakpoints accurately.
        let cache = || make_cache_control(&options.cache_scope, &options.cache_ttl);
        let system_blocks: Option<Vec<SystemBlock>> = if is_oauth {
            let mut blocks = vec![SystemBlock {
                block_type: "text".to_string(),
                text: "You are Claude Code, Anthropic's official CLI for Claude.".to_string(),
                cache_control: Some(cache()),
            }];
            if let Some(ref system_prompt) = context.system_prompt {
                blocks.extend(split_system_prompt(
                    system_prompt,
                    options.system_prompt_boundary.as_deref(),
                    &options.cache_scope,
                    &options.cache_ttl,
                ));
            }
            Some(blocks)
        } else {
            context.system_prompt.as_ref().map(|sp| {
                split_system_prompt(
                    sp,
                    options.system_prompt_boundary.as_deref(),
                    &options.cache_scope,
                    &options.cache_ttl,
                )
            })
        };

        // Count actual cache_control breakpoints in system blocks
        let system_cache_blocks = system_blocks
            .as_ref()
            .map(|blocks| blocks.iter().filter(|b| b.cache_control.is_some()).count())
            .unwrap_or(0);
        let tool_cache_blocks: usize = if has_tools { 1 } else { 0 };

        // Anthropic allows max 4 cache_control breakpoints total per request.
        let message_cache_budget = 4_usize.saturating_sub(system_cache_blocks + tool_cache_blocks);
        let messages = convert_messages(
            &context.messages,
            message_cache_budget,
            &options.cache_scope,
            &options.cache_ttl,
        );
        let tools = if has_tools {
            Some(convert_tools(
                &context.tools,
                true,
                &options.cache_scope,
                &options.cache_ttl,
            ))
        } else {
            None
        };

        let max_tokens = options.base.max_tokens.unwrap_or(model.max_tokens / 3);

        let mut request = AnthropicRequest {
            model: model.id.clone(),
            messages,
            max_tokens,
            stream: true,
            system: system_blocks,
            temperature: options.base.temperature,
            tools,
            tool_choice: options.tool_choice.clone(),
            thinking: None,
        };

        // Enable thinking if requested
        if options.thinking_enabled && model.reasoning {
            request.thinking = Some(if options.thinking_adaptive {
                ThinkingConfig::Adaptive {
                    thinking_type: "adaptive".to_string(),
                }
            } else {
                ThinkingConfig::Enabled {
                    thinking_type: "enabled".to_string(),
                    budget_tokens: options.thinking_budget_tokens.unwrap_or(1024),
                }
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
                ContentBlock::Thinking { thinking, signature } => Some(Content::Thinking { thinking, signature }),
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
        signature: Option<String>,
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

#[derive(Debug, Clone, Serialize)]
struct CacheControl {
    #[serde(rename = "type")]
    control_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    scope: Option<CacheScope>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ThinkingConfig {
    Adaptive {
        #[serde(rename = "type")]
        thinking_type: String,
    },
    Enabled {
        #[serde(rename = "type")]
        thinking_type: String,
        budget_tokens: u32,
    },
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
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
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
    signature: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContentBlockStopEvent {
    index: u32,
    /// Signature for thinking blocks (sent by Anthropic API on content_block_stop)
    #[serde(default)]
    content_block: Option<ContentBlockStopInfo>,
}

#[derive(Debug, Deserialize)]
struct ContentBlockStopInfo {
    /// Signature for thinking blocks
    signature: Option<String>,
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

/// Split a system prompt at a dynamic boundary marker.
///
/// When a boundary is found and the caller opted into global caching (`cache_scope: Global`),
/// the static content before the boundary gets global scope and the dynamic content after
/// gets no caching. Without global scope, the static part uses the caller's scope.
///
/// Returns at least one block as long as the prompt is non-empty.
fn split_system_prompt(
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
                // Static part gets the caller's scope (Global if they opted in, Org otherwise)
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
            // Fix issue 2: if boundary splits produced empty parts, fall through
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

fn convert_messages(
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

    // Consolidate consecutive messages with the same role.
    // This merges e.g. multiple tool_result user messages into one.
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

    // Add cache breakpoints to recent messages, respecting the budget.
    // Anthropic allows max 4 cache_control blocks total per request;
    // the budget accounts for system prompt blocks already used.
    if cache_breakpoint_budget > 0 {
        let total = consolidated.len();
        let cache_zone_start = total.saturating_sub(cache_breakpoint_budget);
        for msg in &mut consolidated[cache_zone_start..] {
            if let serde_json::Value::Array(ref mut blocks) = msg.content {
                // Find the last non-thinking block to add cache_control to
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

fn convert_tools(
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

            // Place cache_control on the last tool definition.
            // This creates a cache prefix covering system prompt + all tools.
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

#[cfg(test)]
mod tests {
    use super::*;

    // Issue 8: Verify CacheScope serialization
    #[test]
    fn test_cache_scope_serialization() {
        assert_eq!(
            serde_json::to_value(CacheScope::Global).unwrap(),
            serde_json::json!("global")
        );
        assert_eq!(
            serde_json::to_value(CacheScope::Org).unwrap(),
            serde_json::json!("org")
        );
    }

    #[test]
    fn test_cache_control_serialization_minimal() {
        let cc = CacheControl {
            control_type: "ephemeral".to_string(),
            scope: None,
            ttl: None,
        };
        let json = serde_json::to_value(&cc).unwrap();
        assert_eq!(json, serde_json::json!({"type": "ephemeral"}));
        assert!(json.get("scope").is_none());
        assert!(json.get("ttl").is_none());
    }

    #[test]
    fn test_cache_control_serialization_full() {
        let cc = CacheControl {
            control_type: "ephemeral".to_string(),
            scope: Some(CacheScope::Global),
            ttl: Some("1h".to_string()),
        };
        let json = serde_json::to_value(&cc).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"type": "ephemeral", "scope": "global", "ttl": "1h"})
        );
    }

    // Issue 9: Verify ThinkingConfig serialization for both variants
    #[test]
    fn test_thinking_config_adaptive_serialization() {
        let config = ThinkingConfig::Adaptive {
            thinking_type: "adaptive".to_string(),
        };
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(json, serde_json::json!({"type": "adaptive"}));
        assert!(json.get("budget_tokens").is_none());
    }

    #[test]
    fn test_thinking_config_enabled_serialization() {
        let config = ThinkingConfig::Enabled {
            thinking_type: "enabled".to_string(),
            budget_tokens: 4096,
        };
        let json = serde_json::to_value(&config).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"type": "enabled", "budget_tokens": 4096})
        );
    }

    // Issue 7: Verify split_system_prompt behavior
    #[test]
    fn test_split_system_prompt_no_boundary() {
        let blocks = split_system_prompt("Hello world", None, &None, &None);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, "Hello world");
        assert!(blocks[0].cache_control.is_some());
    }

    #[test]
    fn test_split_system_prompt_boundary_not_found() {
        let blocks = split_system_prompt(
            "Hello world",
            Some("<!-- BOUNDARY -->"),
            &Some(CacheScope::Org),
            &None,
        );
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, "Hello world");
    }

    #[test]
    fn test_split_system_prompt_boundary_splits() {
        let prompt = "Static part<!-- BOUNDARY -->Dynamic part";
        let blocks = split_system_prompt(
            prompt,
            Some("<!-- BOUNDARY -->"),
            &Some(CacheScope::Global),
            &Some("1h".to_string()),
        );
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].text, "Static part");
        assert!(blocks[0].cache_control.is_some());
        let cc = blocks[0].cache_control.as_ref().unwrap();
        assert!(matches!(cc.scope, Some(CacheScope::Global)));
        assert_eq!(cc.ttl.as_deref(), Some("1h"));
        assert_eq!(blocks[1].text, "Dynamic part");
        assert!(blocks[1].cache_control.is_none());
    }

    #[test]
    fn test_split_system_prompt_respects_caller_scope() {
        // Issue 1 fix: static part should use caller's scope, not hardcode Global
        let prompt = "Static<!-- B -->Dynamic";
        let blocks = split_system_prompt(
            prompt,
            Some("<!-- B -->"),
            &Some(CacheScope::Org),
            &None,
        );
        assert_eq!(blocks.len(), 2);
        let cc = blocks[0].cache_control.as_ref().unwrap();
        assert!(matches!(cc.scope, Some(CacheScope::Org)));
    }

    #[test]
    fn test_split_system_prompt_boundary_at_edges() {
        // Issue 2 fix: boundary at start — empty static part, should still produce a block
        let prompt = "<!-- B -->Dynamic only";
        let blocks =
            split_system_prompt(prompt, Some("<!-- B -->"), &Some(CacheScope::Global), &None);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, "Dynamic only");
        assert!(blocks[0].cache_control.is_none());

        // boundary at end — empty dynamic part
        let prompt = "Static only<!-- B -->";
        let blocks =
            split_system_prompt(prompt, Some("<!-- B -->"), &Some(CacheScope::Global), &None);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, "Static only");
        assert!(blocks[0].cache_control.is_some());
    }

    #[test]
    fn test_split_system_prompt_boundary_is_entire_prompt() {
        // Issue 2 fix: if prompt IS the boundary, both parts empty → fallback to single block
        let prompt = "<!-- B -->";
        let blocks =
            split_system_prompt(prompt, Some("<!-- B -->"), &Some(CacheScope::Global), &None);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, "<!-- B -->");
        assert!(blocks[0].cache_control.is_some());
    }
}
