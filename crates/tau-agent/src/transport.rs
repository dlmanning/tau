//! Transport abstraction for running agents

use std::{pin::Pin, sync::LazyLock, time::Duration};

use regex::Regex;

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
) -> Result<tau_ai::stream::MessageEventStream> {
    match model.api {
        tau_ai::Api::AnthropicMessages => {
            let provider = if let Some(key) = api_key {
                tau_ai::providers::anthropic::AnthropicProvider::new(key.to_string())
            } else {
                tau_ai::providers::anthropic::AnthropicProvider::from_env()?
            };
            provider.stream(model, context, None).await
        }
        tau_ai::Api::OpenAICompletions | tau_ai::Api::OpenAIResponses => {
            let provider = if let Some(key) = api_key {
                tau_ai::providers::openai::OpenAIProvider::new(key.to_string())
            } else {
                tau_ai::providers::openai::OpenAIProvider::from_env()?
            };
            provider.stream(model, context).await
        }
        tau_ai::Api::GoogleGenerativeAI => {
            let provider = if let Some(key) = api_key {
                tau_ai::providers::google::GoogleProvider::new(key.to_string())
            } else {
                tau_ai::providers::google::GoogleProvider::from_env()?
            };
            provider.stream(model, context).await
        }
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

/// Compiled regex patterns for detecting context overflow errors across providers.
/// Covers: Anthropic, Bedrock, OpenAI, Google, xAI, Groq, OpenRouter, Copilot,
/// llama.cpp, LM Studio, MiniMax, Kimi.
static OVERFLOW_PATTERNS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        // Generic / multi-provider patterns
        r"(?i)context.?length.?exceed",
        r"(?i)maximum.?context.?length",
        r"(?i)context.?window.?(exceed|full|limit)",
        r"(?i)too.?many.?tokens",
        r"(?i)prompt.?is.?too.?long",
        r"(?i)input.?too.?long",
        r"(?i)token.?limit.?(exceed|reach)",
        r"(?i)content.?too.?large",
        // Anthropic / Bedrock
        r"(?i)prompt.?too.?long",
        r"(?i)request.?too.?large",
        r"(?i)messages?.?too.?long",
        // OpenAI
        r"(?i)maximum.?number.?of.?tokens",
        r"(?i)reduce.?the.?length",
        r"(?i)context_length_exceeded",
        // "max_tokens" only when followed by overflow language (not config errors)
        r"(?i)max_tokens.*(exceed|limit|too|overflow)",
        // Google (Gemini)
        r"(?i)exceeds?.+token.?limit",
        r"(?i)input.?token.?limit",
        // xAI / Groq / OpenRouter
        r"(?i)context.?overflow",
        r"(?i)sequence.?too.?long",
        // llama.cpp / LM Studio
        r"(?i)context.?size.?exceed",
        r"(?i)n_ctx",
        r"(?i)slot.?context.?overflow",
        // MiniMax / Kimi
        r"(?i)total.?tokens?.?exceed",
        r"(?i)max_prompt_tokens",
        // HTTP status-based (embedded in error strings)
        r"\b413\b",
    ]
    .iter()
    .filter_map(|p| Regex::new(p).ok())
    .collect()
});

/// Regex for HTTP 400 status codes in error strings (e.g. "400 Bad Request", "HTTP 400", "status: 400").
/// Requires word boundary to avoid matching port numbers or IDs containing "400".
static HTTP_400_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(?:status|http|error)[:\s]*400\b|\b400\s+bad\s+request").unwrap());

/// Check if an error indicates a context overflow / too many tokens
pub fn is_context_overflow(error: &str) -> bool {
    // HTTP 400 with token-related keywords — requires structured 400 reference
    if HTTP_400_PATTERN.is_match(error) {
        let lower = error.to_lowercase();
        if lower.contains("token") || lower.contains("context") || lower.contains("length") {
            return true;
        }
    }

    OVERFLOW_PATTERNS.iter().any(|re| re.is_match(error))
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

                match create_provider_and_stream(&model, &context, api_key.as_deref()).await {
                    Ok(s) => {
                        message_stream = s;
                        break;
                    }
                    Err(e) => {
                        // Context overflow is never retryable
                        if e.is_context_overflow() {
                            yield AgentEvent::Error { message: e.to_string() };
                            return;
                        }

                        // Check retryability: typed check + string fallback for wrapped errors
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

#[cfg(test)]
mod tests {
    use super::*;

    // -- Item 1: Overflow pattern tests --

    #[test]
    fn test_overflow_anthropic_prompt_too_long() {
        assert!(is_context_overflow("prompt is too long"));
        assert!(is_context_overflow("Prompt is too long for this model"));
    }

    #[test]
    fn test_overflow_anthropic_request_too_large() {
        assert!(is_context_overflow("request too large"));
    }

    #[test]
    fn test_overflow_anthropic_messages_too_long() {
        assert!(is_context_overflow("messages too long"));
    }

    #[test]
    fn test_overflow_openai_context_length_exceeded() {
        assert!(is_context_overflow(
            "This model's maximum context length is 128000 tokens. context_length_exceeded"
        ));
    }

    #[test]
    fn test_overflow_openai_reduce_length() {
        assert!(is_context_overflow(
            "Please reduce the length of the messages"
        ));
    }

    #[test]
    fn test_overflow_openai_max_tokens_with_exceeds() {
        assert!(is_context_overflow(
            "max_tokens exceeds the model limit"
        ));
    }

    #[test]
    fn test_no_overflow_max_tokens_config_error() {
        // "max_tokens" alone (e.g. config validation) should NOT match
        assert!(!is_context_overflow("max_tokens parameter must be positive"));
        assert!(!is_context_overflow("invalid value for max_tokens"));
    }

    #[test]
    fn test_overflow_google_token_limit() {
        assert!(is_context_overflow(
            "Request exceeds the token limit for this model"
        ));
    }

    #[test]
    fn test_overflow_google_input_token_limit() {
        assert!(is_context_overflow("Input token limit exceeded"));
    }

    #[test]
    fn test_overflow_generic_too_many_tokens() {
        assert!(is_context_overflow("too many tokens in the request"));
    }

    #[test]
    fn test_overflow_generic_context_window() {
        assert!(is_context_overflow("context window exceeded"));
        assert!(is_context_overflow("context window full"));
        assert!(is_context_overflow("exceeds context window limit"));
    }

    #[test]
    fn test_overflow_generic_token_limit() {
        assert!(is_context_overflow("token limit exceeded"));
        assert!(is_context_overflow("token limit reached"));
    }

    #[test]
    fn test_overflow_llama_cpp_n_ctx() {
        assert!(is_context_overflow("n_ctx exceeded, cannot process"));
    }

    #[test]
    fn test_overflow_llama_cpp_slot_overflow() {
        assert!(is_context_overflow("slot context overflow"));
    }

    #[test]
    fn test_overflow_lm_studio_context_size() {
        assert!(is_context_overflow("context size exceeded"));
    }

    #[test]
    fn test_overflow_groq_sequence_too_long() {
        assert!(is_context_overflow("sequence too long for model"));
    }

    #[test]
    fn test_overflow_minimax_total_tokens() {
        assert!(is_context_overflow("total tokens exceed the limit"));
    }

    #[test]
    fn test_overflow_http_413() {
        assert!(is_context_overflow("HTTP 413 Payload Too Large"));
    }

    #[test]
    fn test_overflow_http_400_with_token() {
        assert!(is_context_overflow("HTTP 400: token count exceeds limit"));
        assert!(is_context_overflow("status: 400 - too many tokens in context"));
        assert!(is_context_overflow("error 400: context length exceeded"));
    }

    #[test]
    fn test_overflow_http_400_bad_request_with_context() {
        assert!(is_context_overflow("400 Bad Request: context too large"));
    }

    #[test]
    fn test_no_overflow_normal_errors() {
        assert!(!is_context_overflow("401 Unauthorized"));
        assert!(!is_context_overflow("rate limit exceeded"));
        assert!(!is_context_overflow("internal server error 500"));
        assert!(!is_context_overflow("connection timeout"));
        assert!(!is_context_overflow("invalid API key"));
    }

    #[test]
    fn test_no_overflow_http_400_without_keywords() {
        // 400 alone without token/context/length keywords should NOT match
        assert!(!is_context_overflow("400 Bad Request: invalid field"));
    }

    #[test]
    fn test_no_overflow_400_in_unrelated_context() {
        // "400" appearing as port or ID should NOT match
        assert!(!is_context_overflow("connected to port 14001 with token auth"));
        assert!(!is_context_overflow("processed 400 items in context manager"));
    }
}
