//! OpenAI Chat Completions API provider

use async_stream::stream;
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    stream::{MessageEvent, MessageEventStream},
    types::{AssistantMetadata, Content, Context, Message, Model, StopReason, Usage},
};

/// OpenAI API client
pub struct OpenAIProvider {
    client: reqwest::Client,
    api_key: String,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider with an API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
        }
    }

    /// Create from environment variable
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| Error::InvalidApiKey)?;
        Ok(Self::new(api_key))
    }

    /// List available models from OpenAI
    pub async fn list_models(&self) -> Result<Vec<OpenAIModelInfo>> {
        let url = "https://api.openai.com/v1/models";

        let response = self
            .client
            .get(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(Error::api("model_list_error", text));
        }

        let list: OpenAIModelList = response.json().await?;

        // Filter to chat models only
        let chat_models: Vec<_> = list
            .data
            .into_iter()
            .filter(|m| is_chat_model(&m.id))
            .collect();

        Ok(chat_models)
    }

    /// Stream a response from OpenAI
    pub async fn stream(&self, model: &Model, context: &Context) -> Result<MessageEventStream> {
        let request = self.build_request(model, context)?;
        let url = format!("{}/chat/completions", model.base_url);

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "Authorization",
            format!("Bearer {}", self.api_key).parse().unwrap(),
        );
        headers.insert("content-type", "application/json".parse().unwrap());

        // Add model-specific headers
        for (key, value) in &model.headers {
            if let (Ok(name), Ok(val)) = (
                key.parse::<reqwest::header::HeaderName>(),
                value.parse::<reqwest::header::HeaderValue>(),
            ) {
                headers.insert(name, val);
            }
        }

        let request_builder = self.client.post(&url).headers(headers).json(&request);

        let event_source = EventSource::new(request_builder)
            .map_err(|e| Error::Sse(format!("Failed to create event source: {}", e)))?;

        Ok(Box::pin(create_stream(event_source, model.clone())))
    }

    fn build_request(&self, model: &Model, context: &Context) -> Result<OpenAIRequest> {
        let mut messages = Vec::new();

        // Add system prompt as first message
        if let Some(ref system_prompt) = context.system_prompt {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(MessageContent::Text(system_prompt.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Convert messages
        for msg in &context.messages {
            messages.extend(convert_message(msg));
        }

        // Convert tools
        let tools = if context.tools.is_empty() {
            None
        } else {
            Some(
                context
                    .tools
                    .iter()
                    .map(|t| OpenAITool {
                        tool_type: "function".to_string(),
                        function: OpenAIFunction {
                            name: t.name.clone(),
                            description: Some(t.description.clone()),
                            parameters: Some(t.parameters.clone()),
                        },
                    })
                    .collect(),
            )
        };

        let has_tools = tools.is_some();
        Ok(OpenAIRequest {
            model: model.id.clone(),
            messages,
            stream: true,
            max_tokens: Some(model.max_tokens / 3),
            temperature: None,
            tools,
            tool_choice: if has_tools {
                Some(serde_json::json!("auto"))
            } else {
                None
            },
        })
    }
}

/// Filter function to identify chat-capable models
fn is_chat_model(id: &str) -> bool {
    // Include GPT-4 and GPT-3.5 turbo models
    if id.starts_with("gpt-4") || id.starts_with("gpt-3.5-turbo") {
        // Exclude instruct, embedding, and other non-chat variants
        !id.contains("instruct") && !id.contains("embedding") && !id.contains("vision")
    } else if id.starts_with("o1") || id.starts_with("o3") {
        // Include reasoning models
        true
    } else {
        false
    }
}

/// Model info returned from OpenAI API
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIModelInfo {
    pub id: String,
    pub created: i64,
    pub owned_by: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelList {
    data: Vec<OpenAIModelInfo>,
}

fn convert_message(msg: &Message) -> Vec<OpenAIMessage> {
    match msg {
        Message::User { content, .. } => {
            let text = content
                .iter()
                .filter_map(|c| match c {
                    Content::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");

            vec![OpenAIMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Text(text)),
                tool_calls: None,
                tool_call_id: None,
            }]
        }
        Message::Assistant { content, .. } => {
            let mut text_parts = Vec::new();
            let mut tool_calls = Vec::new();

            for c in content {
                match c {
                    Content::Text { text } => text_parts.push(text.clone()),
                    Content::ToolCall {
                        id,
                        name,
                        arguments,
                    } => {
                        tool_calls.push(OpenAIToolCall {
                            id: id.clone(),
                            call_type: "function".to_string(),
                            function: OpenAIFunctionCall {
                                name: name.clone(),
                                arguments: serde_json::to_string(arguments).unwrap_or_default(),
                            },
                        });
                    }
                    _ => {}
                }
            }

            let content = if text_parts.is_empty() {
                None
            } else {
                Some(MessageContent::Text(text_parts.join("")))
            };

            vec![OpenAIMessage {
                role: "assistant".to_string(),
                content,
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                tool_call_id: None,
            }]
        }
        Message::ToolResult {
            tool_call_id,
            content,
            ..
        } => {
            let text = content
                .iter()
                .filter_map(|c| match c {
                    Content::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");

            vec![OpenAIMessage {
                role: "tool".to_string(),
                content: Some(MessageContent::Text(text)),
                tool_calls: None,
                tool_call_id: Some(tool_call_id.clone()),
            }]
        }
    }
}

fn create_stream(
    mut event_source: EventSource,
    model: Model,
) -> impl futures::Stream<Item = MessageEvent> {
    stream! {
        let mut accumulated_text = String::new();
        let mut tool_calls: Vec<(String, String, String)> = Vec::new(); // (id, name, args)
        let mut current_tool_index: Option<usize> = None;
        let mut finish_reason: Option<String> = None;
        let mut usage = Usage::default();

        // Emit start event
        let start_message = Message::Assistant {
            content: vec![],
            metadata: AssistantMetadata {
                model: Some(model.id.clone()),
                ..Default::default()
            },
        };
        yield MessageEvent::Start { message: start_message };

        while let Some(event) = event_source.next().await {
            match event {
                Ok(Event::Open) => {}
                Ok(Event::Message(msg)) => {
                    if msg.data == "[DONE]" {
                        break;
                    }

                    let chunk: std::result::Result<StreamChunk, _> = serde_json::from_str(&msg.data);
                    match chunk {
                        Ok(chunk) => {
                            for choice in &chunk.choices {
                                // Handle text delta
                                if let Some(ref content) = choice.delta.content {
                                    accumulated_text.push_str(content);
                                    yield MessageEvent::TextDelta {
                                        content_index: 0,
                                        delta: content.clone(),
                                    };
                                }

                                // Handle tool calls
                                if let Some(ref tcs) = choice.delta.tool_calls {
                                    for tc in tcs {
                                        let idx = tc.index as usize;

                                        // Ensure we have space for this tool call
                                        while tool_calls.len() <= idx {
                                            tool_calls.push((String::new(), String::new(), String::new()));
                                        }

                                        // Update tool call data
                                        if let Some(ref id) = tc.id {
                                            tool_calls[idx].0 = id.clone();
                                        }
                                        if let Some(ref function) = tc.function {
                                            if let Some(ref name) = function.name {
                                                tool_calls[idx].1 = name.clone();
                                            }
                                            if let Some(ref args) = function.arguments {
                                                tool_calls[idx].2.push_str(args);
                                            }
                                        }

                                        // Track current tool for delta events
                                        if current_tool_index != Some(idx) {
                                            current_tool_index = Some(idx);
                                        }

                                        // Emit tool call start when we have the name
                                        if let Some(ref function) = tc.function {
                                            if function.name.is_some() && !tool_calls[idx].1.is_empty() {
                                                yield MessageEvent::ToolCallStart {
                                                    content_index: idx,
                                                    id: tool_calls[idx].0.clone(),
                                                    name: tool_calls[idx].1.clone(),
                                                };
                                            }
                                            // Emit tool call delta for arguments
                                            if let Some(ref args) = function.arguments {
                                                yield MessageEvent::ToolCallDelta {
                                                    content_index: idx,
                                                    delta: args.clone(),
                                                };
                                            }
                                        }
                                    }
                                }

                                // Capture finish reason
                                if let Some(ref reason) = choice.finish_reason {
                                    finish_reason = Some(reason.clone());
                                }
                            }

                            // Handle usage in final chunk
                            if let Some(ref stream_usage) = chunk.usage {
                                usage.input = stream_usage.prompt_tokens;
                                usage.output = stream_usage.completion_tokens;
                            }
                        }
                        Err(e) => {
                            yield MessageEvent::Error {
                                message: format!("Failed to parse chunk: {}", e),
                            };
                            return;
                        }
                    }
                }
                Err(e) => {
                    yield MessageEvent::Error {
                        message: format!("SSE error: {}", e),
                    };
                    return;
                }
            }
        }

        // Build final content
        let mut content = Vec::new();

        if !accumulated_text.is_empty() {
            content.push(Content::Text {
                text: accumulated_text,
            });
        }

        for (id, name, args) in tool_calls {
            if !id.is_empty() && !name.is_empty() {
                let arguments = serde_json::from_str(&args).unwrap_or(serde_json::json!({}));
                content.push(Content::ToolCall {
                    id,
                    name,
                    arguments,
                });
            }
        }

        let stop_reason = match finish_reason.as_deref() {
            Some("stop") => Some(StopReason::Stop),
            Some("length") => Some(StopReason::Length),
            Some("tool_calls") => Some(StopReason::ToolUse),
            _ => None,
        };

        let final_message = Message::Assistant {
            content,
            metadata: AssistantMetadata {
                api: Some(crate::Api::OpenAICompletions),
                provider: Some(crate::Provider::OpenAI),
                model: Some(model.id.clone()),
                stop_reason,
                timestamp: chrono::Utc::now().timestamp_millis(),
                ..Default::default()
            },
        };

        yield MessageEvent::Done {
            message: final_message,
            stop_reason: stop_reason.unwrap_or(StopReason::Stop),
            usage,
        };
    }
}

// Request/Response types

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
}

#[derive(Debug, Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize)]
struct OpenAIFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAIFunctionCall,
}

#[derive(Debug, Serialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

// Streaming response types

#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<StreamUsage>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct StreamToolCall {
    index: i32,
    id: Option<String>,
    function: Option<StreamFunction>,
}

#[derive(Debug, Deserialize)]
struct StreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}
