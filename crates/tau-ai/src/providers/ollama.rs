//! Ollama native API provider
//!
//! Uses Ollama's `/api/chat` endpoint with NDJSON streaming instead of the
//! OpenAI compatibility layer, giving access to native features like model
//! management, thinking/reasoning, structured outputs, and full parameter control.

use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    messages::ensure_tool_result_pairing,
    stream::{MessageEvent, MessageEventStream, StreamAccumulator},
    types::{Content, Context, Message, Model, ReasoningLevel, StopReason, StreamOptions},
};

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// Ollama API client
pub struct OllamaProvider {
    client: reqwest::Client,
    base_url: String,
}

impl OllamaProvider {
    /// Create a new Ollama provider with a custom base URL
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
        }
    }

    /// Check if Ollama is running
    pub async fn is_running(&self) -> bool {
        self.client
            .get(&self.base_url)
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
            .is_ok()
    }

    /// List locally installed models (`GET /api/tags`)
    pub async fn list_models(&self) -> Result<Vec<OllamaModelInfo>> {
        let url = format!("{}/api/tags", self.base_url);
        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(Error::api("ollama_error", text));
        }

        let list: OllamaModelList = response.json().await?;
        Ok(list.models)
    }

    /// List currently loaded/running models (`GET /api/ps`)
    pub async fn list_running(&self) -> Result<Vec<OllamaRunningModel>> {
        let url = format!("{}/api/ps", self.base_url);
        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(Error::api("ollama_error", text));
        }

        let list: OllamaRunningList = response.json().await?;
        Ok(list.models)
    }

    /// Get details for a specific model (`POST /api/show`)
    pub async fn show_model(&self, name: &str) -> Result<OllamaModelDetail> {
        let url = format!("{}/api/show", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&serde_json::json!({"model": name}))
            .send()
            .await?;

        if !response.status().is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(Error::api("ollama_error", text));
        }

        Ok(response.json().await?)
    }

    /// Stream a response using Ollama's native chat API (`POST /api/chat`)
    pub async fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&OllamaOptions>,
    ) -> Result<MessageEventStream> {
        let request = self.build_request(model, context, options);
        let url = format!("{}/api/chat", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();

            if status.as_u16() == 404 || text.contains("not found") {
                return Err(Error::ModelNotFound(format!(
                    "Model '{}' not found. Run `ollama pull {}` to download it.",
                    model.id, model.id
                )));
            }
            return Err(Error::api("ollama_error", text));
        }

        Ok(Box::pin(create_stream(response, model.clone())))
    }

    fn build_request(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&OllamaOptions>,
    ) -> OllamaRequest {
        let mut messages = Vec::new();

        if let Some(ref system_prompt) = context.system_prompt {
            messages.push(OllamaMessage {
                role: "system".to_string(),
                content: system_prompt.clone(),
                images: None,
                tool_calls: None,
                thinking: None,
                tool_name: None,
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
                    .map(|t| OllamaTool {
                        tool_type: "function".to_string(),
                        function: OllamaFunctionDef {
                            name: t.name.clone(),
                            description: t.description.clone(),
                            parameters: t.parameters.clone(),
                        },
                    })
                    .collect(),
            )
        };

        // Map reasoning level → think field (only if model supports it)
        let think = if model.reasoning {
            options.and_then(|o| match o.reasoning.unwrap_or_default() {
                ReasoningLevel::Off => None,
                ReasoningLevel::Minimal | ReasoningLevel::Low => Some(serde_json::json!("low")),
                ReasoningLevel::Medium => Some(serde_json::json!("medium")),
                ReasoningLevel::High => Some(serde_json::json!("high")),
            })
        } else {
            None
        };

        // Build native options map
        let mut opts_map = serde_json::Map::new();
        if let Some(opts) = options {
            insert_opt(&mut opts_map, "temperature", opts.base.temperature);
            insert_opt(&mut opts_map, "num_predict", opts.base.max_tokens);
            if !opts.base.stop_sequences.is_empty() {
                opts_map.insert("stop".into(), serde_json::json!(opts.base.stop_sequences));
            }
            // Runner options
            insert_opt(&mut opts_map, "num_ctx", opts.num_ctx);
            insert_opt(&mut opts_map, "num_gpu", opts.num_gpu);
            insert_opt(&mut opts_map, "num_batch", opts.num_batch);
            insert_opt(&mut opts_map, "num_thread", opts.num_thread);
            // Sampling options
            insert_opt(&mut opts_map, "seed", opts.seed);
            insert_opt(&mut opts_map, "top_k", opts.top_k);
            insert_opt(&mut opts_map, "top_p", opts.top_p);
            insert_opt(&mut opts_map, "min_p", opts.min_p);
            insert_opt(&mut opts_map, "repeat_penalty", opts.repeat_penalty);
            insert_opt(&mut opts_map, "presence_penalty", opts.presence_penalty);
            insert_opt(&mut opts_map, "frequency_penalty", opts.frequency_penalty);
            insert_opt(&mut opts_map, "repeat_last_n", opts.repeat_last_n);
        }

        let keep_alive = options.and_then(|o| o.keep_alive.clone());
        let format = options.and_then(|o| o.format.clone());
        let truncate = options.and_then(|o| o.truncate);
        let shift = options.and_then(|o| o.shift);

        OllamaRequest {
            model: model.id.clone(),
            messages,
            stream: true,
            tools,
            options: if opts_map.is_empty() {
                None
            } else {
                Some(serde_json::Value::Object(opts_map))
            },
            keep_alive,
            think,
            format,
            truncate,
            shift,
        }
    }
}

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new("http://localhost:11434")
    }
}

/// Helper: insert a value into a serde_json::Map if it's Some.
fn insert_opt<V: Into<serde_json::Value>>(
    map: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: Option<V>,
) {
    if let Some(v) = value {
        map.insert(key.into(), v.into());
    }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Ollama-specific streaming options
#[derive(Debug, Clone, Default)]
pub struct OllamaOptions {
    /// Base streaming options (max_tokens, temperature, stop_sequences)
    pub base: StreamOptions,

    // -- Runner options --
    /// Context window size for this request
    pub num_ctx: Option<u32>,
    /// Number of GPU layers (-1 = all)
    pub num_gpu: Option<i32>,
    /// Batch size for prompt evaluation
    pub num_batch: Option<u32>,
    /// Number of CPU threads
    pub num_thread: Option<u32>,

    // -- Sampling options --
    /// Random seed (0 = random)
    pub seed: Option<i64>,
    /// Top-K sampling
    pub top_k: Option<u32>,
    /// Nucleus sampling threshold
    pub top_p: Option<f32>,
    /// Minimum probability relative to most likely token
    pub min_p: Option<f32>,
    /// Repetition penalty
    pub repeat_penalty: Option<f32>,
    /// Presence penalty
    pub presence_penalty: Option<f32>,
    /// Frequency penalty
    pub frequency_penalty: Option<f32>,
    /// Lookback window for repeat penalty (0=disabled, -1=num_ctx)
    pub repeat_last_n: Option<i32>,

    // -- Lifecycle --
    /// Keep model loaded after request (e.g. "5m", "1h", "-1" for forever)
    pub keep_alive: Option<String>,

    // -- Features --
    /// Thinking/reasoning level
    pub reasoning: Option<ReasoningLevel>,
    /// Structured output format: `"json"` for JSON mode, or a JSON Schema object
    pub format: Option<serde_json::Value>,
    /// Truncate history if prompt exceeds context length
    pub truncate: Option<bool>,
    /// Shift (drop old context) on overflow instead of erroring
    pub shift: Option<bool>,
}

// ---------------------------------------------------------------------------
// Message conversion
// ---------------------------------------------------------------------------

fn convert_message(msg: &Message) -> Vec<OllamaMessage> {
    match msg {
        Message::User { content, .. } => {
            let mut text_parts = Vec::new();
            let mut images = Vec::new();

            for c in content {
                match c {
                    Content::Text { text } => text_parts.push(text.clone()),
                    Content::Image { data, .. } => images.push(data.clone()),
                    _ => {}
                }
            }

            vec![OllamaMessage {
                role: "user".to_string(),
                content: text_parts.join(""),
                images: if images.is_empty() {
                    None
                } else {
                    Some(images)
                },
                tool_calls: None,
                thinking: None,
                tool_name: None,
                tool_call_id: None,
            }]
        }
        Message::Assistant { content, .. } => {
            let mut text_parts = Vec::new();
            let mut thinking_parts = Vec::new();
            let mut tool_calls = Vec::new();

            for c in content {
                match c {
                    Content::Text { text } => text_parts.push(text.clone()),
                    Content::Thinking { thinking, .. } => thinking_parts.push(thinking.clone()),
                    Content::ToolCall {
                        id,
                        name,
                        arguments,
                    } => {
                        tool_calls.push(OllamaToolCall {
                            id: Some(id.clone()),
                            function: OllamaFunctionCall {
                                name: name.clone(),
                                arguments: arguments.clone(),
                            },
                        });
                    }
                    _ => {}
                }
            }

            vec![OllamaMessage {
                role: "assistant".to_string(),
                content: text_parts.join(""),
                images: None,
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                thinking: if thinking_parts.is_empty() {
                    None
                } else {
                    Some(thinking_parts.join(""))
                },
                tool_name: None,
                tool_call_id: None,
            }]
        }
        Message::ToolResult {
            tool_call_id,
            tool_name,
            content,
            ..
        } => {
            let text = content
                .iter()
                .filter_map(|c| c.as_text())
                .collect::<Vec<_>>()
                .join("");

            vec![OllamaMessage {
                role: "tool".to_string(),
                content: text,
                images: None,
                tool_calls: None,
                thinking: None,
                tool_name: Some(tool_name.clone()),
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

            vec![OllamaMessage {
                role: "user".to_string(),
                content: format!("{}{}", prefix, text),
                images: None,
                tool_calls: None,
                thinking: None,
                tool_name: None,
                tool_call_id: None,
            }]
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

fn create_stream(
    response: reqwest::Response,
    model: Model,
) -> impl futures::Stream<Item = MessageEvent> {
    stream! {
        let (mut acc, start) = StreamAccumulator::new(
            model.api,
            model.provider,
            model.id.clone(),
        );
        yield start;

        let mut next_index = 0usize;
        let mut text_index: Option<usize> = None;
        let mut thinking_index: Option<usize> = None;
        let mut has_tool_calls = false;
        let mut byte_stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(bytes) => bytes,
                Err(e) => {
                    yield StreamAccumulator::error_event(format!("Stream error: {}", e));
                    return;
                }
            };

            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete NDJSON lines
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim().to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                match serde_json::from_str::<OllamaChatResponse>(&line) {
                    Ok(resp) => {
                        // Thinking content
                        if let Some(ref thinking) = resp.message.thinking {
                            if !thinking.is_empty() {
                                let idx = match thinking_index {
                                    Some(i) => i,
                                    None => {
                                        let i = next_index;
                                        next_index += 1;
                                        thinking_index = Some(i);
                                        for ev in acc.thinking_start(i) { yield ev; }
                                        i
                                    }
                                };
                                for ev in acc.thinking_delta(idx, thinking) { yield ev; }
                            }
                        }

                        // Text content
                        if !resp.message.content.is_empty() {
                            let idx = *text_index.get_or_insert_with(|| {
                                let i = next_index;
                                next_index += 1;
                                i
                            });
                            for ev in acc.text_delta(idx, &resp.message.content) {
                                yield ev;
                            }
                        }

                        // Tool calls (arrive fully formed, not streamed)
                        if let Some(ref tool_calls) = resp.message.tool_calls {
                            has_tool_calls = true;
                            for tc in tool_calls {
                                let idx = next_index;
                                next_index += 1;
                                let id = tc.id.clone().unwrap_or_else(|| format!("call_{}", idx));

                                for ev in acc.tool_call_start(idx, &id, &tc.function.name) {
                                    yield ev;
                                }
                                let args_str = serde_json::to_string(&tc.function.arguments)
                                    .unwrap_or_default();
                                for ev in acc.tool_call_delta(idx, &args_str) {
                                    yield ev;
                                }
                                for ev in acc.tool_call_end(idx) {
                                    yield ev;
                                }
                            }
                        }

                        // Final chunk with usage stats and timing
                        if resp.done {
                            let u = acc.usage_mut();
                            u.input = resp.prompt_eval_count.unwrap_or(0);
                            u.output = resp.eval_count.unwrap_or(0);

                            // Log timing via tracing (provider-specific, not in shared types)
                            let timing = OllamaTiming::from_response(&resp);
                            tracing::info!(
                                total_ms = timing.total_ms(),
                                load_ms = timing.load_ms(),
                                prompt_eval_ms = timing.prompt_eval_ms(),
                                eval_ms = timing.eval_ms(),
                                tokens_per_second = format!("{:.1}", timing.tokens_per_second(
                                    resp.eval_count.unwrap_or(0)
                                )),
                                "Ollama generation complete"
                            );

                            let stop = if has_tool_calls {
                                StopReason::ToolUse
                            } else if let Some(ref reason) = resp.done_reason {
                                match reason.as_str() {
                                    "stop" => StopReason::Stop,
                                    "length" => StopReason::Length,
                                    _ => StopReason::Stop,
                                }
                            } else {
                                StopReason::Stop
                            };
                            acc.set_stop_reason(stop);
                            break;
                        }
                    }
                    Err(e) => {
                        if let Ok(err_resp) = serde_json::from_str::<OllamaErrorResponse>(&line) {
                            yield StreamAccumulator::error_event(err_resp.error);
                        } else {
                            yield StreamAccumulator::error_event(
                                format!("Failed to parse Ollama response: {}", e)
                            );
                        }
                        return;
                    }
                }
            }
        }

        for ev in acc.finish() { yield ev; }
    }
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OllamaTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncate: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shift: Option<bool>,
}

#[derive(Debug, Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct OllamaTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OllamaFunctionDef,
}

#[derive(Debug, Serialize)]
struct OllamaFunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    function: OllamaFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaFunctionCall {
    name: String,
    arguments: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: OllamaResponseMessage,
    done: bool,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
    // Timing (nanoseconds, only on done=true)
    #[serde(default)]
    total_duration: Option<u64>,
    #[serde(default)]
    load_duration: Option<u64>,
    #[serde(default)]
    prompt_eval_duration: Option<u64>,
    #[serde(default)]
    eval_duration: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct OllamaResponseMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OllamaErrorResponse {
    error: String,
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// Performance timing from an Ollama response (nanoseconds)
#[derive(Debug, Clone, Default)]
pub struct OllamaTiming {
    pub total_ns: u64,
    pub load_ns: u64,
    pub prompt_eval_ns: u64,
    pub eval_ns: u64,
}

impl OllamaTiming {
    fn from_response(resp: &OllamaChatResponse) -> Self {
        Self {
            total_ns: resp.total_duration.unwrap_or(0),
            load_ns: resp.load_duration.unwrap_or(0),
            prompt_eval_ns: resp.prompt_eval_duration.unwrap_or(0),
            eval_ns: resp.eval_duration.unwrap_or(0),
        }
    }

    /// Total time in milliseconds
    pub fn total_ms(&self) -> u64 {
        self.total_ns / 1_000_000
    }

    /// Model load time in milliseconds
    pub fn load_ms(&self) -> u64 {
        self.load_ns / 1_000_000
    }

    /// Prompt evaluation time in milliseconds
    pub fn prompt_eval_ms(&self) -> u64 {
        self.prompt_eval_ns / 1_000_000
    }

    /// Generation time in milliseconds
    pub fn eval_ms(&self) -> u64 {
        self.eval_ns / 1_000_000
    }

    /// Tokens per second during generation
    pub fn tokens_per_second(&self, eval_count: u64) -> f64 {
        if self.eval_ns == 0 {
            return 0.0;
        }
        eval_count as f64 / (self.eval_ns as f64 / 1_000_000_000.0)
    }
}

// ---------------------------------------------------------------------------
// Model info types
// ---------------------------------------------------------------------------

/// Information about a locally installed Ollama model
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelInfo {
    /// Model name (e.g. "llama3.2:latest")
    pub name: String,
    /// Model size in bytes
    #[serde(default)]
    pub size: u64,
    /// Model digest
    #[serde(default)]
    pub digest: String,
    /// Model details
    #[serde(default)]
    pub details: Option<OllamaModelDetails>,
}

/// Details about a model's architecture
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelDetails {
    /// Model family (e.g. "llama")
    #[serde(default)]
    pub family: String,
    /// Parameter size (e.g. "3.2B")
    #[serde(default)]
    pub parameter_size: String,
    /// Quantization level (e.g. "Q4_K_M")
    #[serde(default)]
    pub quantization_level: String,
}

/// Detailed model information from `/api/show`
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelDetail {
    /// Model template
    #[serde(default)]
    pub template: String,
    /// Model parameters
    #[serde(default)]
    pub parameters: String,
    /// Model details
    #[serde(default)]
    pub details: Option<OllamaModelDetails>,
    /// Model capabilities (e.g. ["completion", "vision", "tools"])
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Raw model metadata
    #[serde(default)]
    pub model_info: Option<serde_json::Value>,
}

/// Information about a currently loaded/running model from `/api/ps`
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaRunningModel {
    /// Model name
    pub name: String,
    /// Total model size in bytes
    #[serde(default)]
    pub size: u64,
    /// VRAM used in bytes
    #[serde(default)]
    pub size_vram: u64,
    /// Active context length
    #[serde(default)]
    pub context_length: Option<u64>,
    /// When the model will be unloaded
    #[serde(default)]
    pub expires_at: Option<String>,
    /// Model details
    #[serde(default)]
    pub details: Option<OllamaModelDetails>,
}

#[derive(Debug, Deserialize)]
struct OllamaModelList {
    models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Deserialize)]
struct OllamaRunningList {
    models: Vec<OllamaRunningModel>,
}

impl OllamaModelInfo {
    /// Get the model ID (name without tag if it's ":latest")
    pub fn id(&self) -> &str {
        self.name.strip_suffix(":latest").unwrap_or(&self.name)
    }

    /// Human-readable size (e.g. "3.2 GB")
    pub fn size_display(&self) -> String {
        const GB: u64 = 1_000_000_000;
        const MB: u64 = 1_000_000;
        if self.size >= GB {
            format!("{:.1} GB", self.size as f64 / GB as f64)
        } else {
            format!("{:.0} MB", self.size as f64 / MB as f64)
        }
    }
}

impl OllamaModelDetail {
    /// Check if this model supports a specific capability
    pub fn has_capability(&self, cap: &str) -> bool {
        self.capabilities.iter().any(|c| c == cap)
    }

    /// Whether this model supports vision/images
    pub fn supports_vision(&self) -> bool {
        self.has_capability("vision")
    }

    /// Whether this model supports tool/function calling
    pub fn supports_tools(&self) -> bool {
        self.has_capability("tools")
    }

    /// Extract the context length from model_info.
    ///
    /// Ollama stores this as `{arch}.context_length` in the model_info blob
    /// (e.g. `"llama.context_length": 131072`).
    pub fn context_length(&self) -> Option<u32> {
        let info = self.model_info.as_ref()?.as_object()?;
        // Look for any key ending in ".context_length"
        for (key, value) in info {
            if key.ends_with(".context_length") {
                return value.as_u64().and_then(|v| u32::try_from(v).ok());
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AssistantMetadata, Content, Message};

    fn test_model() -> Model {
        Model {
            id: "llama3.2".to_string(),
            name: "Llama 3.2".to_string(),
            api: crate::Api::Ollama,
            provider: crate::Provider::Ollama,
            base_url: "http://localhost:11434".to_string(),
            reasoning: false,
            input_types: vec![],
            cost: Default::default(),
            context_window: 128000,
            max_tokens: 8192,
            headers: Default::default(),
        }
    }

    // -- Message conversion --

    #[test]
    fn test_convert_user_text_message() {
        let msg = Message::user("Hello");
        let result = convert_message(&msg);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
        assert_eq!(result[0].content, "Hello");
        assert!(result[0].images.is_none());
    }

    #[test]
    fn test_convert_user_image_message() {
        let msg = Message::User {
            content: vec![
                Content::text("What's in this image?"),
                Content::Image {
                    mime_type: "image/png".to_string(),
                    data: "base64data".to_string(),
                },
            ],
            timestamp: 0,
        };
        let result = convert_message(&msg);
        assert_eq!(result[0].content, "What's in this image?");
        let images = result[0].images.as_ref().unwrap();
        assert_eq!(images, &["base64data"]);
    }

    #[test]
    fn test_convert_assistant_text_message() {
        let msg = Message::Assistant {
            content: vec![Content::text("Hi there")],
            metadata: AssistantMetadata::default(),
        };
        let result = convert_message(&msg);
        assert_eq!(result[0].role, "assistant");
        assert_eq!(result[0].content, "Hi there");
        assert!(result[0].thinking.is_none());
    }

    #[test]
    fn test_convert_assistant_with_thinking() {
        let msg = Message::Assistant {
            content: vec![
                Content::thinking("Let me reason about this..."),
                Content::text("The answer is 42."),
            ],
            metadata: AssistantMetadata::default(),
        };
        let result = convert_message(&msg);
        assert_eq!(result[0].content, "The answer is 42.");
        assert_eq!(
            result[0].thinking.as_deref(),
            Some("Let me reason about this...")
        );
    }

    #[test]
    fn test_convert_assistant_tool_call() {
        let msg = Message::Assistant {
            content: vec![Content::ToolCall {
                id: "call_1".to_string(),
                name: "bash".to_string(),
                arguments: serde_json::json!({"command": "ls"}),
            }],
            metadata: AssistantMetadata::default(),
        };
        let result = convert_message(&msg);
        let tool_calls = result[0].tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls[0].id.as_deref(), Some("call_1"));
        assert_eq!(tool_calls[0].function.name, "bash");
    }

    #[test]
    fn test_convert_tool_result_uses_separate_fields() {
        let msg = Message::tool_result("call_1", "bash", vec![Content::text("output")], false);
        let result = convert_message(&msg);
        assert_eq!(result[0].role, "tool");
        assert_eq!(result[0].content, "output"); // plain text, not JSON
        assert_eq!(result[0].tool_name.as_deref(), Some("bash"));
        assert_eq!(result[0].tool_call_id.as_deref(), Some("call_1"));
    }

    // -- Request building --

    #[test]
    fn test_build_request_with_options() {
        let provider = OllamaProvider::default();
        let model = test_model();
        let context = Context {
            system_prompt: Some("You are helpful.".to_string()),
            messages: vec![Message::user("Hello")],
            tools: vec![],
            server_tools: vec![],
        };
        let options = OllamaOptions {
            base: StreamOptions {
                max_tokens: Some(2048),
                temperature: Some(0.7),
                stop_sequences: vec!["STOP".to_string()],
                ..Default::default()
            },
            num_ctx: Some(4096),
            keep_alive: Some("10m".to_string()),
            seed: Some(42),
            top_k: Some(50),
            top_p: Some(0.95),
            ..Default::default()
        };
        let request = provider.build_request(&model, &context, Some(&options));
        assert_eq!(request.model, "llama3.2");
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.keep_alive, Some("10m".to_string()));

        let opts = request.options.unwrap();
        assert_eq!(opts["num_predict"], 2048);
        assert_eq!(opts["num_ctx"], 4096);
        assert_eq!(opts["seed"], 42);
        assert_eq!(opts["top_k"], 50);
        assert_eq!(opts["stop"], serde_json::json!(["STOP"]));
    }

    #[test]
    fn test_build_request_with_thinking() {
        let provider = OllamaProvider::default();
        let mut model = test_model();
        model.reasoning = true;
        let context = Context::default();

        let options = OllamaOptions {
            reasoning: Some(ReasoningLevel::High),
            ..Default::default()
        };
        let request = provider.build_request(&model, &context, Some(&options));
        assert_eq!(request.think, Some(serde_json::json!("high")));

        let options_med = OllamaOptions {
            reasoning: Some(ReasoningLevel::Medium),
            ..Default::default()
        };
        let request_med = provider.build_request(&model, &context, Some(&options_med));
        assert_eq!(request_med.think, Some(serde_json::json!("medium")));

        let options_off = OllamaOptions {
            reasoning: Some(ReasoningLevel::Off),
            ..Default::default()
        };
        let request_off = provider.build_request(&model, &context, Some(&options_off));
        assert!(request_off.think.is_none());

        // Non-reasoning model ignores reasoning option
        let non_reasoning_model = test_model(); // reasoning: false
        let request_nr = provider.build_request(&non_reasoning_model, &context, Some(&options));
        assert!(request_nr.think.is_none());
    }

    #[test]
    fn test_build_request_with_format() {
        let provider = OllamaProvider::default();
        let model = test_model();
        let context = Context::default();

        // JSON mode
        let options = OllamaOptions {
            format: Some(serde_json::json!("json")),
            ..Default::default()
        };
        let request = provider.build_request(&model, &context, Some(&options));
        assert_eq!(request.format, Some(serde_json::json!("json")));

        // JSON Schema
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        });
        let options_schema = OllamaOptions {
            format: Some(schema.clone()),
            ..Default::default()
        };
        let request_schema = provider.build_request(&model, &context, Some(&options_schema));
        assert_eq!(request_schema.format, Some(schema));
    }

    #[test]
    fn test_build_request_truncate_shift() {
        let provider = OllamaProvider::default();
        let model = test_model();
        let context = Context::default();

        let options = OllamaOptions {
            truncate: Some(true),
            shift: Some(true),
            ..Default::default()
        };
        let request = provider.build_request(&model, &context, Some(&options));
        assert_eq!(request.truncate, Some(true));
        assert_eq!(request.shift, Some(true));
    }

    #[test]
    fn test_build_request_with_tools() {
        let provider = OllamaProvider::default();
        let model = test_model();
        let context = Context {
            system_prompt: None,
            messages: vec![],
            tools: vec![crate::Tool::new(
                "bash",
                "Run a command",
                serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
            )],
            server_tools: vec![],
        };
        let request = provider.build_request(&model, &context, None);
        let tools = request.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "bash");
    }

    // -- Response parsing --

    #[test]
    fn test_parse_chat_response() {
        let json = r#"{"model":"llama3.2","message":{"role":"assistant","content":"Hello!"},"done":false}"#;
        let resp: OllamaChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message.content, "Hello!");
        assert!(!resp.done);
        assert!(resp.message.thinking.is_none());
    }

    #[test]
    fn test_parse_chat_response_with_thinking() {
        let json = r#"{"model":"qwq","message":{"role":"assistant","content":"","thinking":"Let me think..."},"done":false}"#;
        let resp: OllamaChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message.thinking.as_deref(), Some("Let me think..."));
    }

    #[test]
    fn test_parse_chat_response_done_with_timing() {
        let json = r#"{"model":"llama3.2","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":46,"eval_count":113,"total_duration":5589157167,"load_duration":3013701500,"prompt_eval_duration":1160282000,"eval_duration":1325948000}"#;
        let resp: OllamaChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.done);
        assert_eq!(resp.prompt_eval_count, Some(46));
        assert_eq!(resp.eval_count, Some(113));

        let timing = OllamaTiming::from_response(&resp);
        assert_eq!(timing.total_ms(), 5589);
        assert_eq!(timing.load_ms(), 3013);
        assert!(timing.tokens_per_second(113) > 80.0);
    }

    #[test]
    fn test_parse_chat_response_with_tool_calls() {
        let json = r#"{"model":"llama3.2","message":{"role":"assistant","content":"","tool_calls":[{"id":"tc_1","function":{"name":"get_weather","arguments":{"city":"Paris"}}}]},"done":false}"#;
        let resp: OllamaChatResponse = serde_json::from_str(json).unwrap();
        let tool_calls = resp.message.tool_calls.unwrap();
        assert_eq!(tool_calls[0].id.as_deref(), Some("tc_1"));
        assert_eq!(tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_parse_tool_call_without_id() {
        let json = r#"{"model":"llama3.2","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"bash","arguments":{"cmd":"ls"}}}]},"done":false}"#;
        let resp: OllamaChatResponse = serde_json::from_str(json).unwrap();
        let tool_calls = resp.message.tool_calls.unwrap();
        assert!(tool_calls[0].id.is_none());
    }

    #[test]
    fn test_parse_error_response() {
        let json = r#"{"error":"model 'nonexistent' not found"}"#;
        let resp: OllamaErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.error, "model 'nonexistent' not found");
    }

    // -- Model info --

    #[test]
    fn test_model_info_id_strips_latest() {
        let info = OllamaModelInfo {
            name: "llama3.2:latest".to_string(),
            size: 0,
            digest: String::new(),
            details: None,
        };
        assert_eq!(info.id(), "llama3.2");
    }

    #[test]
    fn test_model_info_size_display() {
        let info = OllamaModelInfo {
            name: "test".to_string(),
            size: 3_200_000_000,
            digest: String::new(),
            details: None,
        };
        assert_eq!(info.size_display(), "3.2 GB");
    }

    #[test]
    fn test_model_detail_capabilities() {
        let json =
            r#"{"template":"","parameters":"","capabilities":["completion","vision","tools"]}"#;
        let detail: OllamaModelDetail = serde_json::from_str(json).unwrap();
        assert!(detail.supports_vision());
        assert!(detail.supports_tools());
        assert!(detail.has_capability("completion"));
        assert!(!detail.has_capability("embedding"));
    }

    #[test]
    fn test_model_detail_context_length() {
        let json = r#"{"template":"","parameters":"","capabilities":["completion"],"model_info":{"general.architecture":"qwen3","qwen3.context_length":262144}}"#;
        let detail: OllamaModelDetail = serde_json::from_str(json).unwrap();
        assert_eq!(detail.context_length(), Some(262144));

        // No model_info → None
        let json2 = r#"{"template":"","parameters":"","capabilities":[]}"#;
        let detail2: OllamaModelDetail = serde_json::from_str(json2).unwrap();
        assert_eq!(detail2.context_length(), None);
    }

    #[test]
    fn test_parse_running_model() {
        let json = r#"{"name":"llama3.2:latest","size":3200000000,"size_vram":2800000000,"context_length":4096,"expires_at":"2025-01-01T00:05:00Z","details":{"family":"llama","parameter_size":"3.2B","quantization_level":"Q4_K_M"}}"#;
        let model: OllamaRunningModel = serde_json::from_str(json).unwrap();
        assert_eq!(model.name, "llama3.2:latest");
        assert_eq!(model.size_vram, 2800000000);
        assert_eq!(model.context_length, Some(4096));
        assert_eq!(model.details.unwrap().parameter_size, "3.2B");
    }

    #[test]
    fn test_timing_tokens_per_second() {
        let timing = OllamaTiming {
            total_ns: 5_000_000_000,
            load_ns: 0,
            prompt_eval_ns: 0,
            eval_ns: 2_000_000_000, // 2 seconds
        };
        // 100 tokens in 2 seconds = 50 tok/s
        assert!((timing.tokens_per_second(100) - 50.0).abs() < 0.1);
    }
}
