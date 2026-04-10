//! Google Generative AI (Gemini) API provider

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

/// Google Generative AI client
pub struct GoogleProvider {
    client: reqwest::Client,
    api_key: String,
}

impl GoogleProvider {
    /// Create a new Google provider with an API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
        }
    }

    /// Create from environment variable
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("GOOGLE_API_KEY")
            .or_else(|_| std::env::var("GEMINI_API_KEY"))
            .map_err(|_| Error::InvalidApiKey)?;
        Ok(Self::new(api_key))
    }

    /// List available models from Google
    pub async fn list_models(&self) -> Result<Vec<GoogleModelInfo>> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models?key={}",
            self.api_key
        );

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(Error::api("model_list_error", text));
        }

        let list: GoogleModelList = response.json().await?;

        let chat_models: Vec<_> = list
            .models
            .into_iter()
            .filter(|m| {
                m.supported_generation_methods
                    .iter()
                    .any(|method| method == "generateContent")
            })
            .collect();

        Ok(chat_models)
    }

    /// Stream a response from Gemini
    pub async fn stream(&self, model: &Model, context: &Context) -> Result<MessageEventStream> {
        let request = self.build_request(model, context)?;
        let url = format!(
            "{}/models/{}:streamGenerateContent?alt=sse&key={}",
            model.base_url, model.id, self.api_key
        );

        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());

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

    fn build_request(&self, model: &Model, context: &Context) -> Result<GeminiRequest> {
        let mut contents = Vec::new();

        let mut context_messages = context.messages.clone();
        ensure_tool_result_pairing(&mut context_messages);

        for msg in &context_messages {
            if let Some(content) = convert_message(msg) {
                contents.push(content);
            }
        }

        let system_instruction = context.system_prompt.as_ref().map(|prompt| GeminiContent {
            role: None,
            parts: vec![GeminiPart::Text {
                text: prompt.clone(),
            }],
        });

        let tools = if context.tools.is_empty() {
            None
        } else {
            let function_declarations: Vec<GeminiFunctionDeclaration> = context
                .tools
                .iter()
                .map(|t| GeminiFunctionDeclaration {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: Some(t.parameters.clone()),
                })
                .collect();
            Some(vec![GeminiTool {
                function_declarations,
            }])
        };

        Ok(GeminiRequest {
            contents,
            system_instruction,
            tools,
            generation_config: Some(GeminiGenerationConfig {
                max_output_tokens: Some(model.max_tokens / 3),
                temperature: None,
                top_p: None,
                top_k: None,
            }),
        })
    }
}

fn convert_message(msg: &Message) -> Option<GeminiContent> {
    match msg {
        Message::User { content, .. } => {
            let parts: Vec<GeminiPart> = content
                .iter()
                .filter_map(|c| match c {
                    Content::Text { text } => Some(GeminiPart::Text { text: text.clone() }),
                    Content::Image { mime_type, data } => Some(GeminiPart::InlineData {
                        inline_data: GeminiInlineData {
                            mime_type: mime_type.clone(),
                            data: data.clone(),
                        },
                    }),
                    _ => None,
                })
                .collect();

            if parts.is_empty() {
                None
            } else {
                Some(GeminiContent {
                    role: Some("user".to_string()),
                    parts,
                })
            }
        }
        Message::Assistant { content, .. } => {
            let mut parts = Vec::new();

            for c in content {
                match c {
                    Content::Text { text } => {
                        parts.push(GeminiPart::Text { text: text.clone() });
                    }
                    Content::ToolCall {
                        id,
                        name,
                        arguments,
                    } => {
                        parts.push(GeminiPart::FunctionCall {
                            function_call: GeminiFunctionCall {
                                name: name.clone(),
                                args: arguments.clone(),
                            },
                        });
                        // Store ID in metadata (Gemini doesn't use IDs)
                        let _ = id;
                    }
                    _ => {}
                }
            }

            if parts.is_empty() {
                None
            } else {
                Some(GeminiContent {
                    role: Some("model".to_string()),
                    parts,
                })
            }
        }
        Message::ToolResult {
            tool_call_id,
            tool_name,
            content,
            ..
        } => {
            let response_text = content
                .iter()
                .filter_map(|c| match c {
                    Content::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");

            // Gemini uses function response format
            let _ = tool_call_id; // Gemini doesn't use IDs
            Some(GeminiContent {
                role: Some("function".to_string()),
                parts: vec![GeminiPart::FunctionResponse {
                    function_response: GeminiFunctionResponse {
                        name: tool_name.clone(),
                        response: serde_json::json!({ "result": response_text }),
                    },
                }],
            })
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
            Some(GeminiContent {
                role: Some("user".to_string()),
                parts: vec![GeminiPart::Text {
                    text: format!("{}{}", prefix, text),
                }],
            })
        }
    }
}

fn create_stream(
    mut event_source: EventSource,
    model: Model,
) -> impl futures::Stream<Item = MessageEvent> {
    stream! {
        let (mut acc, start) = StreamAccumulator::new(
            crate::Api::GoogleGenerativeAI,
            crate::Provider::Google,
            model.id.clone(),
        );
        yield start;

        let mut tool_count = 0usize;

        while let Some(event) = event_source.next().await {
            match event {
                Ok(Event::Open) => {}
                Ok(Event::Message(msg)) => {
                    if msg.data.is_empty() || msg.data == "[DONE]" {
                        continue;
                    }

                    match serde_json::from_str::<GeminiStreamResponse>(&msg.data) {
                        Ok(response) => {
                            for candidate in &response.candidates {
                                if let Some(ref content) = candidate.content {
                                    for part in &content.parts {
                                        match part {
                                            GeminiResponsePart::Text { text } => {
                                                for ev in acc.text_delta(0, text) { yield ev; }
                                            }
                                            GeminiResponsePart::FunctionCall { function_call } => {
                                                let idx = tool_count;
                                                tool_count += 1;
                                                let id = format!("call_{}", idx);
                                                for ev in acc.tool_call_start(idx, &id, &function_call.name) { yield ev; }
                                                let args_str = serde_json::to_string(&function_call.args).unwrap_or_default();
                                                for ev in acc.tool_call_delta(idx, &args_str) { yield ev; }
                                            }
                                        }
                                    }
                                }

                                if let Some(ref reason) = candidate.finish_reason {
                                    acc.set_stop_reason(match reason.as_str() {
                                        "STOP" => StopReason::Stop,
                                        "MAX_TOKENS" => StopReason::Length,
                                        "SAFETY" | "RECITATION" => StopReason::Stop,
                                        _ => StopReason::Stop,
                                    });
                                }
                            }

                            if let Some(ref usage) = response.usage_metadata {
                                let u = acc.usage_mut();
                                u.input = usage.prompt_token_count.unwrap_or(0);
                                u.output = usage.candidates_token_count.unwrap_or(0);
                            }
                        }
                        Err(e) => {
                            if let Ok(error_response) = serde_json::from_str::<GeminiErrorResponse>(&msg.data) {
                                yield StreamAccumulator::error_event(error_response.error.message);
                                return;
                            }
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
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum GeminiPart {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GeminiInlineData,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResponse,
    },
}

#[derive(Debug, Serialize)]
struct GeminiInlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct GeminiFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct GeminiFunctionResponse {
    name: String,
    response: serde_json::Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiTool {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<i32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiStreamResponse {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: Option<GeminiResponseContent>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponseContent {
    parts: Vec<GeminiResponsePart>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum GeminiResponsePart {
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiResponseFunctionCall,
    },
}

#[derive(Debug, Deserialize)]
struct GeminiResponseFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: Option<u32>,
    candidates_token_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct GeminiErrorResponse {
    error: GeminiError,
}

#[derive(Debug, Deserialize)]
struct GeminiError {
    message: String,
}

/// Model info returned from Google API
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoogleModelInfo {
    /// Model name (e.g., "models/gemini-1.5-pro")
    pub name: String,
    /// Display name
    pub display_name: String,
    /// Description
    #[serde(default)]
    pub description: String,
    /// Supported generation methods
    #[serde(default)]
    pub supported_generation_methods: Vec<String>,
    /// Input token limit
    #[serde(default)]
    pub input_token_limit: Option<u32>,
    /// Output token limit
    #[serde(default)]
    pub output_token_limit: Option<u32>,
}

impl GoogleModelInfo {
    /// Get the model ID (without "models/" prefix)
    pub fn id(&self) -> &str {
        self.name.strip_prefix("models/").unwrap_or(&self.name)
    }
}

#[derive(Debug, Deserialize)]
struct GoogleModelList {
    models: Vec<GoogleModelInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Content, Message};

    #[test]
    fn test_convert_user_text_message() {
        let msg = Message::user("Hello");
        let result = convert_message(&msg).unwrap();
        assert_eq!(result.role, Some("user".to_string()));
        assert_eq!(result.parts.len(), 1);
        let json = serde_json::to_value(&result.parts[0]).unwrap();
        assert_eq!(json["text"], "Hello");
    }

    #[test]
    fn test_convert_user_image_message() {
        let msg = Message::User {
            content: vec![Content::Image {
                mime_type: "image/png".to_string(),
                data: "base64data".to_string(),
            }],
            timestamp: 0,
        };
        let result = convert_message(&msg).unwrap();
        assert_eq!(result.parts.len(), 1);
        let json = serde_json::to_value(&result.parts[0]).unwrap();
        // Verify correct nested structure: { "inlineData": { "mimeType": "...", "data": "..." } }
        assert_eq!(json["inlineData"]["mimeType"], "image/png");
        assert_eq!(json["inlineData"]["data"], "base64data");
    }

    #[test]
    fn test_convert_assistant_text_message() {
        let msg = Message::Assistant {
            content: vec![Content::text("Hi there")],
            metadata: Default::default(),
        };
        let result = convert_message(&msg).unwrap();
        assert_eq!(result.role, Some("model".to_string()));
        let json = serde_json::to_value(&result.parts[0]).unwrap();
        assert_eq!(json["text"], "Hi there");
    }

    #[test]
    fn test_convert_assistant_tool_call() {
        let msg = Message::Assistant {
            content: vec![Content::ToolCall {
                id: "call_1".to_string(),
                name: "read".to_string(),
                arguments: serde_json::json!({"path": "/foo.rs"}),
            }],
            metadata: Default::default(),
        };
        let result = convert_message(&msg).unwrap();
        let json = serde_json::to_value(&result.parts[0]).unwrap();
        assert_eq!(json["functionCall"]["name"], "read");
        assert_eq!(json["functionCall"]["args"]["path"], "/foo.rs");
    }

    #[test]
    fn test_convert_tool_result() {
        let msg = Message::tool_result(
            "call_1",
            "read",
            vec![Content::text("file contents")],
            false,
        );
        let result = convert_message(&msg).unwrap();
        assert_eq!(result.role, Some("function".to_string()));
        let json = serde_json::to_value(&result.parts[0]).unwrap();
        assert_eq!(json["functionResponse"]["name"], "read");
    }

    #[test]
    fn test_inline_data_serialization_format() {
        let part = GeminiPart::InlineData {
            inline_data: GeminiInlineData {
                mime_type: "image/jpeg".to_string(),
                data: "abc123".to_string(),
            },
        };
        let json = serde_json::to_value(&part).unwrap();
        // Must be nested: {"inlineData": {"mimeType": "image/jpeg", "data": "abc123"}}
        assert!(json.get("inlineData").is_some(), "missing inlineData key");
        assert_eq!(json["inlineData"]["mimeType"], "image/jpeg");
        assert_eq!(json["inlineData"]["data"], "abc123");
        // Must NOT have flat fields
        assert!(json.get("mime_type").is_none());
    }
}
