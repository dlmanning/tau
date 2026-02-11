//! Google Generative AI (Gemini) API provider

use crate::{
    error::{Error, Result},
    stream::{MessageEvent, MessageEventStream},
    types::{AssistantMetadata, Content, Context, Message, Model, StopReason, Usage},
};
use async_stream::stream;
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use serde::{Deserialize, Serialize};

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

        // Filter to generative models that support generateContent
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

    fn build_request(&self, model: &Model, context: &Context) -> Result<GeminiRequest> {
        let mut contents = Vec::new();

        // Convert messages
        for msg in &context.messages {
            if let Some(content) = convert_message(msg) {
                contents.push(content);
            }
        }

        // System instruction (if present)
        let system_instruction = context.system_prompt.as_ref().map(|prompt| GeminiContent {
            role: None,
            parts: vec![GeminiPart::Text {
                text: prompt.clone(),
            }],
        });

        // Convert tools
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
                        mime_type: mime_type.clone(),
                        data: data.clone(),
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
    }
}

fn create_stream(
    mut event_source: EventSource,
    model: Model,
) -> impl futures::Stream<Item = MessageEvent> {
    stream! {
        let mut accumulated_text = String::new();
        let mut tool_calls: Vec<(String, String, serde_json::Value)> = Vec::new(); // (id, name, args)
        let mut finish_reason: Option<String> = None;
        let mut total_input_tokens = 0u32;
        let mut total_output_tokens = 0u32;

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
                    if msg.data.is_empty() || msg.data == "[DONE]" {
                        continue;
                    }

                    let chunk: std::result::Result<GeminiStreamResponse, _> = serde_json::from_str(&msg.data);
                    match chunk {
                        Ok(response) => {
                            for candidate in &response.candidates {
                                if let Some(ref content) = candidate.content {
                                    for part in &content.parts {
                                        match part {
                                            GeminiResponsePart::Text { text } => {
                                                accumulated_text.push_str(text);
                                                yield MessageEvent::TextDelta {
                                                    content_index: 0,
                                                    delta: text.clone(),
                                                };
                                            }
                                            GeminiResponsePart::FunctionCall { function_call } => {
                                                // Generate a unique ID since Gemini doesn't provide one
                                                let id = format!("call_{}", tool_calls.len());
                                                let name = function_call.name.clone();
                                                let args = function_call.args.clone();

                                                yield MessageEvent::ToolCallStart {
                                                    content_index: tool_calls.len(),
                                                    id: id.clone(),
                                                    name: name.clone(),
                                                };

                                                let args_str = serde_json::to_string(&args).unwrap_or_default();
                                                yield MessageEvent::ToolCallDelta {
                                                    content_index: tool_calls.len(),
                                                    delta: args_str,
                                                };

                                                tool_calls.push((id, name, args));
                                            }
                                        }
                                    }
                                }

                                // Capture finish reason
                                if let Some(ref reason) = candidate.finish_reason {
                                    finish_reason = Some(reason.clone());
                                }
                            }

                            // Handle usage metadata
                            if let Some(ref usage) = response.usage_metadata {
                                total_input_tokens = usage.prompt_token_count.unwrap_or(0);
                                total_output_tokens = usage.candidates_token_count.unwrap_or(0);
                            }
                        }
                        Err(e) => {
                            // Try to parse as error response
                            if let Ok(error_response) = serde_json::from_str::<GeminiErrorResponse>(&msg.data) {
                                yield MessageEvent::Error {
                                    message: error_response.error.message,
                                };
                                return;
                            }
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
            content.push(Content::ToolCall {
                id,
                name,
                arguments: args,
            });
        }

        let stop_reason = match finish_reason.as_deref() {
            Some("STOP") => Some(StopReason::Stop),
            Some("MAX_TOKENS") => Some(StopReason::Length),
            Some("SAFETY") => Some(StopReason::Stop),
            Some("RECITATION") => Some(StopReason::Stop),
            _ => None,
        };

        let final_message = Message::Assistant {
            content,
            metadata: AssistantMetadata {
                api: Some(crate::Api::GoogleGenerativeAI),
                provider: Some(crate::Provider::Google),
                model: Some(model.id.clone()),
                stop_reason,
                timestamp: chrono::Utc::now().timestamp_millis(),
                ..Default::default()
            },
        };

        yield MessageEvent::Done {
            message: final_message,
            stop_reason: stop_reason.unwrap_or(StopReason::Stop),
            usage: Usage {
                input: total_input_tokens,
                output: total_output_tokens,
                ..Default::default()
            },
        };
    }
}

// Request types

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
        mime_type: String,
        data: String,
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

// Response types

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

// Model listing types

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
