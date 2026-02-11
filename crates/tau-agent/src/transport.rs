//! Transport abstraction for running agents

use std::{pin::Pin, time::Duration};

use async_stream::stream;
use async_trait::async_trait;
use futures::StreamExt;
use tau_ai::{Context, Model, Result, stream::MessageBuilder};
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
    api_key: Option<&str>,
) -> Result<Result<tau_ai::stream::MessageEventStream>> {
    match model.api {
        tau_ai::Api::AnthropicMessages => {
            let provider = if let Some(key) = api_key {
                tau_ai::providers::anthropic::AnthropicProvider::new(key.to_string())
            } else {
                tau_ai::providers::anthropic::AnthropicProvider::from_env()?
            };
            Ok(provider.stream(model, context, None).await)
        }
        tau_ai::Api::OpenAICompletions => {
            let provider = if let Some(key) = api_key {
                tau_ai::providers::openai::OpenAIProvider::new(key.to_string())
            } else {
                tau_ai::providers::openai::OpenAIProvider::from_env()?
            };
            Ok(provider.stream(model, context).await)
        }
        tau_ai::Api::GoogleGenerativeAI => {
            let provider = if let Some(key) = api_key {
                tau_ai::providers::google::GoogleProvider::new(key.to_string())
            } else {
                tau_ai::providers::google::GoogleProvider::from_env()?
            };
            Ok(provider.stream(model, context).await)
        }
        _ => Err(tau_ai::Error::UnsupportedProvider(format!(
            "{:?}",
            model.api
        ))),
    }
}

/// Check if an error is retryable
fn is_retryable_error(error: &str) -> bool {
    // Rate limit errors
    if error.contains("429") || error.contains("rate limit") || error.contains("Rate limit") {
        return true;
    }
    // Transient network errors
    if error.contains("timeout") || error.contains("Timeout") {
        return true;
    }
    if error.contains("connection") || error.contains("Connection") {
        return true;
    }
    // Server errors (5xx)
    if error.contains("500")
        || error.contains("502")
        || error.contains("503")
        || error.contains("504")
    {
        return true;
    }
    // Overloaded
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
    pub tools: Vec<tau_ai::Tool>,
    /// Model to use
    pub model: Model,
    /// Reasoning/thinking level
    pub reasoning: Option<tau_ai::ReasoningLevel>,
    /// Maximum tokens per response
    pub max_tokens: Option<u32>,
    /// Temperature
    pub temperature: Option<f32>,
}

/// A stream of agent events
pub type AgentEventStream = Pin<Box<dyn Stream<Item = AgentEvent> + Send>>;

/// Transport for running agent interactions
#[async_trait]
pub trait Transport: Send + Sync {
    /// Run an agent turn, streaming events
    async fn run(
        &self,
        messages: Vec<tau_ai::Message>,
        user_message: tau_ai::Message,
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
impl Transport for ProviderTransport {
    async fn run(
        &self,
        messages: Vec<tau_ai::Message>,
        user_message: tau_ai::Message,
        config: &AgentRunConfig,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<AgentEventStream> {
        let mut context = Context {
            system_prompt: config.system_prompt.clone(),
            messages,
            tools: config.tools.clone(),
        };
        context.push(user_message);

        // Get the appropriate provider and stream
        let model = config.model.clone();
        let api_key = self.api_key.clone();
        let retry_config = self.retry_config.clone();

        let event_stream: AgentEventStream = Box::pin(stream! {
            yield AgentEvent::TurnStart { turn_number: 1 };

            // Retry loop
            let mut attempt = 0u32;
            let message_stream;

            loop {
                if cancel.is_cancelled() {
                    yield AgentEvent::Error { message: "Cancelled".to_string() };
                    return;
                }

                // Create provider based on model API
                let stream_result = create_provider_and_stream(&model, &context, api_key.as_deref()).await;

                // Handle provider creation errors
                let stream_result = match stream_result {
                    Ok(result) => result,
                    Err(e) => {
                        yield AgentEvent::Error { message: e.to_string() };
                        return;
                    }
                };

                match stream_result {
                    Ok(s) => {
                        message_stream = s;
                        break;
                    }
                    Err(e) => {
                        let error_msg = e.to_string();

                        // Check if we should retry
                        if attempt < retry_config.max_retries && is_retryable_error(&error_msg) {
                            let delay = retry_config.delay_for_attempt(attempt);
                            tracing::warn!(
                                "Request failed (attempt {}/{}): {}. Retrying in {:?}...",
                                attempt + 1,
                                retry_config.max_retries + 1,
                                error_msg,
                                delay
                            );
                            attempt += 1;
                            tokio::time::sleep(delay).await;
                            continue;
                        }

                        // Non-retryable or max retries exceeded
                        yield AgentEvent::Error { message: error_msg };
                        return;
                    }
                }
            }

            let mut message_stream = message_stream;

            let mut builder = MessageBuilder::new();
            let mut final_message = None;
            let mut final_usage = tau_ai::Usage::default();

            while let Some(event) = message_stream.next().await {
                if cancel.is_cancelled() {
                    yield AgentEvent::Error { message: "Cancelled".to_string() };
                    return;
                }

                builder.process_event(&event);

                match &event {
                    tau_ai::stream::MessageEvent::Start { message } => {
                        yield AgentEvent::MessageStart { message: message.clone() };
                    }
                    tau_ai::stream::MessageEvent::TextDelta { .. }
                    | tau_ai::stream::MessageEvent::ThinkingDelta { .. }
                    | tau_ai::stream::MessageEvent::ToolCallDelta { .. } => {
                        // Build current partial message
                        let partial = tau_ai::Message::Assistant {
                            content: builder.current_content(),
                            metadata: tau_ai::AssistantMetadata::default(),
                        };
                        yield AgentEvent::MessageUpdate { message: partial };
                    }
                    tau_ai::stream::MessageEvent::Done { message, usage, .. } => {
                        final_message = Some(message.clone());
                        final_usage = usage.clone();
                        yield AgentEvent::MessageEnd { message: message.clone() };
                    }
                    tau_ai::stream::MessageEvent::Error { message } => {
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
                // Note: AgentEnd is sent by the Agent, not the transport
            }
        });

        Ok(event_stream)
    }
}
