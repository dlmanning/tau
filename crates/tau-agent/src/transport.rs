//! Transport abstraction for running agents

use std::{pin::Pin, time::Duration};

use async_stream::stream;
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;
use tau_ai::{
    Api, AssistantMetadata, Context, Message, Model, Provider, ReasoningLevel, Result, ServerTool,
    Tool as AiTool, Usage,
    stream::{MessageBuilder, MessageEvent},
};
use tokio::time;
use tokio_stream::Stream;

use crate::events::AgentEvent;

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Multiplier for exponential backoff
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 2.0,
        }
    }
}

/// Seconds of stream inactivity before logging a stall warning
const STREAM_STALL_WARN_SECS: u64 = 30;
/// Seconds of stream inactivity before aborting the stream
const STREAM_IDLE_TIMEOUT_SECS: u64 = 90;

/// Longer timeouts for local model providers where time-to-first-token can be slow
const LOCAL_STREAM_STALL_WARN_SECS: u64 = 120;
const LOCAL_STREAM_IDLE_TIMEOUT_SECS: u64 = 600;

fn is_local_provider(provider: Provider) -> bool {
    matches!(provider, Provider::Ollama | Provider::Custom)
}

impl RetryConfig {
    /// Calculate delay for a given attempt (0-indexed)
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let delay_secs =
            self.initial_delay.as_secs_f64() * self.backoff_multiplier.powi(attempt as i32);
        Duration::from_secs_f64(delay_secs.min(self.max_delay.as_secs_f64()))
    }
}

/// Helper to create provider and get stream using trait-based dispatch
async fn create_provider_and_stream(
    model: &Model,
    context: &Context,
    config: &AgentRunConfig,
    api_key: Option<&str>,
) -> Result<tau_ai::stream::MessageEventStream> {
    match model.api {
        Api::AnthropicMessages => {
            use tau_ai::providers::anthropic::{AnthropicOptions, AnthropicProvider, CacheScope};

            let provider = if let Some(key) = api_key {
                AnthropicProvider::new(key.to_string())
            } else {
                AnthropicProvider::from_env()?
            };

            let reasoning = config.reasoning.unwrap_or_default();
            let thinking_enabled = reasoning != ReasoningLevel::Off;

            let budget = match reasoning {
                ReasoningLevel::Off => None,
                ReasoningLevel::Minimal => Some(1024),
                ReasoningLevel::Low => Some(4096),
                ReasoningLevel::Medium => Some(10000),
                ReasoningLevel::High => Some(32000),
            };

            let cache_scope = config.cache_scope.as_deref().and_then(|s| match s {
                "global" => Some(CacheScope::Global),
                "org" => Some(CacheScope::Org),
                _ => None,
            });

            let options = AnthropicOptions {
                base: tau_ai::StreamOptions {
                    max_tokens: config.max_tokens,
                    temperature: config.temperature,
                    reasoning: config.reasoning,
                    stop_sequences: vec![],
                },
                thinking_enabled,
                thinking_adaptive: config.thinking_adaptive,
                thinking_budget_tokens: budget,
                thinking_display: None,
                tool_choice: None,
                cache_scope,
                cache_ttl: config.cache_ttl.clone(),
                system_prompt_boundary: config.system_prompt_boundary.clone(),
                metadata: None,
                service_tier: None,
                effort: None,
                output_format: None,
                container: None,
                auto_cache_control: false,
                inference_geo: None,
            };

            provider.stream(model, context, Some(&options)).await
        }
        Api::OpenAICompletions | Api::OpenAIResponses => {
            let provider = if let Some(key) = api_key {
                tau_ai::providers::openai::OpenAIProvider::new(key.to_string())
            } else if model.provider.api_key_env_var().is_some() {
                tau_ai::providers::openai::OpenAIProvider::from_env()?
            } else {
                tau_ai::providers::openai::OpenAIProvider::without_key()
            };
            let options = tau_ai::StreamOptions {
                max_tokens: config.max_tokens,
                temperature: config.temperature,
                ..Default::default()
            };
            provider.stream(model, context, Some(&options)).await
        }
        Api::GoogleGenerativeAI => {
            let provider = if let Some(key) = api_key {
                tau_ai::providers::google::GoogleProvider::new(key.to_string())
            } else {
                tau_ai::providers::google::GoogleProvider::from_env()?
            };
            let options = tau_ai::StreamOptions {
                max_tokens: config.max_tokens,
                temperature: config.temperature,
                ..Default::default()
            };
            provider.stream(model, context, Some(&options)).await
        }
        Api::Ollama => {
            let provider =
                tau_ai::providers::ollama::OllamaProvider::new(&model.base_url);
            let options = tau_ai::providers::ollama::OllamaOptions {
                base: tau_ai::StreamOptions {
                    max_tokens: config.max_tokens,
                    temperature: config.temperature,
                    ..Default::default()
                },
                reasoning: config.reasoning,
                ..Default::default()
            };
            provider.stream(model, context, Some(&options)).await
        }
    }
}

/// Format a server tool result for display.
fn format_server_tool_result(api_type: &str, content: &Value) -> String {
    if api_type.contains("web_search") {
        if let Some(results) = content.as_array() {
            let entries: Vec<String> = results
                .iter()
                .filter_map(|r| {
                    let title = r.get("title").and_then(|t| t.as_str())?;
                    let url = r.get("url").and_then(|u| u.as_str())?;
                    Some(format!("- {} ({})", title, url))
                })
                .collect();
            if entries.is_empty() {
                return "No results found".to_string();
            }
            return format!("{} results:\n{}", entries.len(), entries.join("\n"));
        }
        if let Some(error_code) = content.get("error_code").and_then(|e| e.as_str()) {
            return format!("Search error: {}", error_code);
        }
    }
    let json = serde_json::to_string_pretty(content).unwrap_or_default();
    if json.len() > 500 {
        format!("{}...", &json[..500])
    } else {
        json
    }
}

/// Check if an error is retryable
fn is_retryable_error(error: &str) -> bool {
    if error.contains("429") || error.contains("rate limit") || error.contains("Rate limit") {
        return true;
    }
    if error.contains("timeout") || error.contains("Timeout") {
        return true;
    }
    if error.contains("connection") || error.contains("Connection") {
        return true;
    }
    if error.contains("500")
        || error.contains("502")
        || error.contains("503")
        || error.contains("504")
    {
        return true;
    }
    if error.contains("overloaded") || error.contains("Overloaded") {
        return true;
    }
    false
}

/// Configuration for an agent run
#[derive(Debug, Clone)]
pub struct AgentRunConfig {
    /// System prompt
    pub system_prompt: Option<String>,
    /// Available tools (as API definitions)
    pub tools: Vec<AiTool>,
    /// Server-controlled tools (e.g. web search)
    pub server_tools: Vec<ServerTool>,
    /// Model to use
    pub model: Model,
    /// Reasoning/thinking level
    pub reasoning: Option<ReasoningLevel>,
    /// Use adaptive thinking (model decides when to think)
    pub thinking_adaptive: bool,
    /// Maximum tokens per response
    pub max_tokens: Option<u32>,
    /// Temperature
    pub temperature: Option<f32>,
    /// Cache scope for prompt caching
    pub cache_scope: Option<String>,
    /// Cache TTL (e.g. "1h")
    pub cache_ttl: Option<String>,
    /// Dynamic boundary marker for system prompt splitting
    pub system_prompt_boundary: Option<String>,
}

/// A stream of agent events
pub type AgentEventStream = Pin<Box<dyn Stream<Item = AgentEvent> + Send>>;

/// Transport for running agent interactions
#[async_trait]
pub trait Transport: Send + Sync {
    /// Run an agent turn, streaming events
    async fn run(
        &self,
        messages: Vec<Message>,
        config: &AgentRunConfig,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<AgentEventStream>;
}

/// Direct provider transport - calls LLM APIs directly
pub struct ProviderTransport {
    api_key: Option<String>,
    retry_config: RetryConfig,
}

impl ProviderTransport {
    /// Create a new provider transport
    pub fn new() -> Self {
        Self {
            api_key: None,
            retry_config: RetryConfig::default(),
        }
    }

    /// Create with a specific API key
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self {
            api_key: Some(api_key.into()),
            retry_config: RetryConfig::default(),
        }
    }

    /// Set retry configuration
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }
}

impl Default for ProviderTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
#[allow(unused_assignments)]
impl Transport for ProviderTransport {
    async fn run(
        &self,
        messages: Vec<Message>,
        config: &AgentRunConfig,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<AgentEventStream> {
        let context = Context {
            system_prompt: config.system_prompt.clone(),
            messages,
            tools: config.tools.clone(),
            server_tools: config.server_tools.clone(),
        };

        let model = config.model.clone();
        let run_config = config.clone();
        let api_key = self.api_key.clone();
        let retry_config = self.retry_config.clone();

        let event_stream: AgentEventStream = Box::pin(stream! {
            yield AgentEvent::TurnStart { turn_number: 1 };

            let mut attempt = 0u32;
            let message_stream;

            loop {
                if cancel.is_cancelled() {
                    yield AgentEvent::Error { message: "Cancelled".to_string() };
                    return;
                }

                match create_provider_and_stream(&model, &context, &run_config, api_key.as_deref()).await {
                    Ok(s) => {
                        message_stream = s;
                        break;
                    }
                    Err(e) => {
                        if e.is_context_overflow() {
                            yield AgentEvent::Error { message: e.to_string() };
                            return;
                        }

                        let error_msg = e.to_string();
                        let retryable = e.is_retryable() || is_retryable_error(&error_msg);

                        if attempt < retry_config.max_retries && retryable {
                            let delay = retry_config.delay_for_attempt(attempt);
                            tracing::warn!(
                                "Request failed (attempt {}/{}): {}. Retrying in {:?}...",
                                attempt + 1,
                                retry_config.max_retries + 1,
                                error_msg,
                                delay
                            );
                            attempt += 1;
                            time::sleep(delay).await;
                            continue;
                        }

                        yield AgentEvent::Error { message: error_msg };
                        return;
                    }
                }
            }

            let mut message_stream = message_stream;

            let mut builder = MessageBuilder::new();
            let mut final_message = None;
            let mut final_usage = Usage::default();
            let mut stall_warned = false;
            let mut last_event_at = time::Instant::now();

            let local = is_local_provider(model.provider);
            let stall_warn_secs = if local { LOCAL_STREAM_STALL_WARN_SECS } else { STREAM_STALL_WARN_SECS };
            let idle_timeout_secs = if local { LOCAL_STREAM_IDLE_TIMEOUT_SECS } else { STREAM_IDLE_TIMEOUT_SECS };

            loop {
                let stall_remaining = Duration::from_secs(stall_warn_secs)
                    .saturating_sub(last_event_at.elapsed());
                let idle_remaining = Duration::from_secs(idle_timeout_secs)
                    .saturating_sub(last_event_at.elapsed());

                if idle_remaining.is_zero() {
                    tracing::error!("Stream idle for {}s, aborting", idle_timeout_secs);
                    yield AgentEvent::Error {
                        message: format!("Stream timed out after {}s of inactivity", idle_timeout_secs),
                    };
                    return;
                }

                if stall_remaining.is_zero() && !stall_warned {
                    tracing::warn!("Stream stalled for {}s, waiting up to {}s before aborting",
                        stall_warn_secs, idle_timeout_secs);
                    stall_warned = true;
                }

                let event = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => {
                        yield AgentEvent::Error { message: "Cancelled".to_string() };
                        return;
                    }
                    result = time::timeout(idle_remaining, message_stream.next()) => {
                        match result {
                            Ok(event) => event,
                            Err(_) => {
                                tracing::error!("Stream idle for {}s, aborting", idle_timeout_secs);
                                yield AgentEvent::Error {
                                    message: format!("Stream timed out after {}s of inactivity", idle_timeout_secs),
                                };
                                return;
                            }
                        }
                    }
                };

                let Some(event) = event else { break };

                last_event_at = time::Instant::now();
                stall_warned = false;

                builder.process_event(&event);

                match &event {
                    MessageEvent::Start { message } => {
                        yield AgentEvent::MessageStart { message: message.clone() };
                    }
                    MessageEvent::TextDelta { .. }
                    | MessageEvent::ThinkingDelta { .. }
                    | MessageEvent::ToolCallDelta { .. } => {
                        let partial = Message::Assistant {
                            content: builder.current_content(),
                            metadata: AssistantMetadata::default(),
                        };
                        yield AgentEvent::MessageUpdate { message: partial };
                    }
                    MessageEvent::ServerToolStart { id, name, input, .. } => {
                        yield AgentEvent::ToolExecutionStart {
                            tool_call_id: id.clone(),
                            tool_name: name.clone(),
                            arguments: input.clone(),
                            activity: format!("Running {}", name),
                        };
                    }
                    MessageEvent::ServerToolEnd { tool_use_id, api_type, content, .. } => {
                        let result = format_server_tool_result(api_type, content);
                        yield AgentEvent::ToolExecutionEnd {
                            tool_call_id: tool_use_id.clone(),
                            tool_name: api_type.clone(),
                            result,
                            is_error: false,
                        };
                    }
                    MessageEvent::Done { message, usage, .. } => {
                        final_message = Some(message.clone());
                        final_usage = usage.clone();
                        yield AgentEvent::MessageEnd { message: message.clone() };
                    }
                    MessageEvent::Error { message } => {
                        yield AgentEvent::Error { message: message.clone() };
                        return;
                    }
                    _ => {}
                }
            }

            if let Some(msg) = final_message {
                yield AgentEvent::TurnEnd {
                    turn_number: 1,
                    message: msg,
                    usage: final_usage,
                };
            }
        });

        Ok(event_stream)
    }
}
