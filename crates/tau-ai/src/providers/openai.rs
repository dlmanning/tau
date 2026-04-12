//! OpenAI Chat Completions API provider

use async_stream::stream;
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    messages::ensure_tool_result_pairing,
    stream::{MessageEvent, MessageEventStream, StreamAccumulator},
    types::{Content, Context, Message, Model, StopReason},
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
            format!("Bearer {}", self.api_key)
                .parse()
                .map_err(|_| Error::InvalidConfig("invalid API key for header".into()))?,
        );
        headers.insert("content-type", super::APPLICATION_JSON);

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

        if let Some(ref system_prompt) = context.system_prompt {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(MessageContent::Text(system_prompt.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        let mut context_messages = context.messages.clone();
        ensure_tool_result_pairing(&mut context_messages);

        for msg in &context_messages {
            messages.extend(convert_message(msg));
        }

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
            stream_options: Some(serde_json::json!({"include_usage": true})),
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
        Message::SystemInjection { content, source } => {
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
            vec![OpenAIMessage {
                role: "user".to_string(),
                content: Some(MessageContent::Text(format!("{}{}", prefix, text))),
                tool_calls: None,
                tool_call_id: None,
            }]
        }
    }
}

fn create_stream(
    mut event_source: EventSource,
    model: Model,
) -> impl futures::Stream<Item = MessageEvent> {
    stream! {
        let (mut acc, start) = StreamAccumulator::new(
            crate::Api::OpenAICompletions,
            crate::Provider::OpenAI,
            model.id.clone(),
        );
        yield start;

        while let Some(event) = event_source.next().await {
            match event {
                Ok(Event::Open) => {}
                Ok(Event::Message(msg)) => {
                    if msg.data == "[DONE]" {
                        break;
                    }

                    match serde_json::from_str::<StreamChunk>(&msg.data) {
                        Ok(chunk) => {
                            for choice in &chunk.choices {
                                if let Some(ref content) = choice.delta.content {
                                    for ev in acc.text_delta(0, content) { yield ev; }
                                }

                                if let Some(ref tcs) = choice.delta.tool_calls {
                                    for tc in tcs {
                                        let idx = tc.index as usize;
                                        if let Some(ref function) = tc.function {
                                            if let Some(ref name) = function.name {
                                                let id = tc.id.as_deref().unwrap_or("");
                                                for ev in acc.tool_call_start(idx, id, name) { yield ev; }
                                            }
                                            if let Some(ref args) = function.arguments {
                                                for ev in acc.tool_call_delta(idx, args) { yield ev; }
                                            }
                                        }
                                    }
                                }

                                if let Some(ref reason) = choice.finish_reason {
                                    acc.set_stop_reason(match reason.as_str() {
                                        "stop" => StopReason::Stop,
                                        "length" => StopReason::Length,
                                        "tool_calls" => StopReason::ToolUse,
                                        _ => StopReason::Stop,
                                    });
                                }
                            }

                            if let Some(ref stream_usage) = chunk.usage {
                                let u = acc.usage_mut();
                                u.input = stream_usage.prompt_tokens;
                                u.output = stream_usage.completion_tokens;
                            }
                        }
                        Err(e) => {
                            yield StreamAccumulator::error_event(format!("Failed to parse chunk: {}", e));
                            return;
                        }
                    }
                }
                Err(e) => {
                    yield StreamAccumulator::error_event(format!("SSE error: {}", e));
                    return;
                }
            }
        }

        for ev in acc.finish() { yield ev; }
    }
}

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
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<serde_json::Value>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Content, Message};

    #[test]
    fn test_convert_user_text_message() {
        let msg = Message::user("Hello");
        let result = convert_message(&msg);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        match &result[0].content {
            Some(MessageContent::Text(t)) => assert_eq!(t, "Hello"),
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn test_convert_assistant_text_message() {
        let msg = Message::Assistant {
            content: vec![Content::text("Hi")],
            metadata: Default::default(),
        };
        let result = convert_message(&msg);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "assistant");
        match &result[0].content {
            Some(MessageContent::Text(t)) => assert_eq!(t, "Hi"),
            other => panic!("expected Text, got {:?}", other),
        }
    }

    #[test]
    fn test_convert_assistant_tool_call() {
        let msg = Message::Assistant {
            content: vec![Content::ToolCall {
                id: "call_1".to_string(),
                name: "bash".to_string(),
                arguments: serde_json::json!({"command": "ls"}),
            }],
            metadata: Default::default(),
        };
        let result = convert_message(&msg);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "assistant");
        let tc = result[0].tool_calls.as_ref().unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].id, "call_1");
        assert_eq!(tc[0].function.name, "bash");
    }

    #[test]
    fn test_convert_tool_result() {
        let msg = Message::tool_result("call_1", "bash", vec![Content::text("output")], false);
        let result = convert_message(&msg);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "tool");
        assert_eq!(result[0].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn test_convert_assistant_mixed_text_and_tools() {
        let msg = Message::Assistant {
            content: vec![
                Content::text("Let me check"),
                Content::ToolCall {
                    id: "call_1".to_string(),
                    name: "read".to_string(),
                    arguments: serde_json::json!({"path": "foo.rs"}),
                },
            ],
            metadata: Default::default(),
        };
        let result = convert_message(&msg);
        assert_eq!(result.len(), 1);
        // Should have both text content and tool calls
        assert!(result[0].content.is_some());
        assert!(result[0].tool_calls.is_some());
    }

    #[test]
    fn test_request_includes_stream_options() {
        let provider = OpenAIProvider::new("test-key");
        let model = Model {
            id: "gpt-4".to_string(),
            name: "GPT-4".to_string(),
            api: crate::Api::OpenAICompletions,
            provider: crate::Provider::OpenAI,
            base_url: "https://api.openai.com/v1".to_string(),
            reasoning: false,
            input_types: vec![],
            cost: Default::default(),
            context_window: 128000,
            max_tokens: 8192,
            headers: Default::default(),
        };
        let context = Context {
            system_prompt: None,
            messages: vec![],
            tools: vec![],
            server_tools: vec![],
        };
        let request = provider.build_request(&model, &context).unwrap();
        assert!(request.stream);
        assert!(request.stream_options.is_some());
        let opts = request.stream_options.unwrap();
        assert_eq!(opts["include_usage"], true);
    }
}
