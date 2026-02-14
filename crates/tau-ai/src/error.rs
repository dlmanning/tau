//! Error types for tau-ai

use thiserror::Error;

/// Result type alias using tau-ai Error
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur when interacting with LLM providers
#[derive(Error, Debug)]
pub enum Error {
    /// HTTP request failed
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization/deserialization failed
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// API returned an error response
    #[error("API error: {message} (type: {error_type})")]
    Api { error_type: String, message: String },

    /// Rate limit exceeded
    #[error("Rate limited: retry after {retry_after:?} seconds")]
    RateLimited { retry_after: Option<u64> },

    /// Authentication failed
    #[error("Authentication failed: {0}")]
    Auth(String),

    /// Invalid API key
    #[error("Invalid or missing API key")]
    InvalidApiKey,

    /// Stream was aborted
    #[error("Request aborted")]
    Aborted,

    /// Server-sent events error
    #[error("SSE error: {0}")]
    Sse(String),

    /// Unexpected response format
    #[error("Unexpected response: {0}")]
    UnexpectedResponse(String),

    /// Model not found
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Provider not supported
    #[error("Provider not supported: {0}")]
    UnsupportedProvider(String),

    /// Tool execution failed
    #[error("Tool execution failed: {0}")]
    ToolExecution(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Context overflow / too many tokens
    #[error("Context overflow: {0}")]
    ContextOverflow(String),
}

impl Error {
    /// Create an API error from type and message
    pub fn api(error_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Api {
            error_type: error_type.into(),
            message: message.into(),
        }
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::Http(_) | Error::RateLimited { .. } | Error::Sse(_) => true,
            Error::Api {
                error_type,
                message,
            } => {
                let et = error_type.to_lowercase();
                let msg = message.to_lowercase();
                // Rate limit / overload patterns in API errors
                et.contains("rate_limit")
                    || et.contains("overloaded")
                    || msg.contains("rate limit")
                    || msg.contains("overloaded")
                    || msg.contains("too many requests")
                    || msg.contains("529")
            }
            _ => false,
        }
    }

    /// Check if this error indicates a context overflow / too many tokens
    pub fn is_context_overflow(&self) -> bool {
        match self {
            Error::ContextOverflow(_) => true,
            Error::Api { message, .. } => {
                let msg = message.to_lowercase();
                msg.contains("too many tokens")
                    || msg.contains("context length")
                    || msg.contains("context window")
                    || msg.contains("token limit")
                    || msg.contains("prompt is too long")
                    || msg.contains("prompt too long")
                    || msg.contains("request too large")
                    || msg.contains("messages too long")
                    || msg.contains("reduce the length")
                    || msg.contains("context_length_exceeded")
                    || msg.contains("content too large")
                    || msg.contains("input too long")
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- is_retryable on Api variant ---

    #[test]
    fn test_retryable_typed_variants() {
        assert!(Error::RateLimited { retry_after: Some(5) }.is_retryable());
        assert!(Error::Sse("connection reset".into()).is_retryable());
    }

    #[test]
    fn test_retryable_api_rate_limit_error_type() {
        let e = Error::api("rate_limit_error", "You have exceeded the rate limit");
        assert!(e.is_retryable());
    }

    #[test]
    fn test_retryable_api_overloaded_error_type() {
        let e = Error::api("overloaded_error", "The server is overloaded");
        assert!(e.is_retryable());
    }

    #[test]
    fn test_retryable_api_rate_limit_message() {
        let e = Error::api("error", "Rate limit exceeded, please retry");
        assert!(e.is_retryable());
    }

    #[test]
    fn test_retryable_api_overloaded_message() {
        let e = Error::api("server_error", "API is overloaded right now");
        assert!(e.is_retryable());
    }

    #[test]
    fn test_retryable_api_too_many_requests() {
        let e = Error::api("error", "Too many requests");
        assert!(e.is_retryable());
    }

    #[test]
    fn test_not_retryable_api_auth() {
        let e = Error::api("authentication_error", "Invalid API key");
        assert!(!e.is_retryable());
    }

    #[test]
    fn test_not_retryable_non_api() {
        assert!(!Error::InvalidApiKey.is_retryable());
        assert!(!Error::Aborted.is_retryable());
        assert!(!Error::ContextOverflow("too big".into()).is_retryable());
    }

    // --- is_context_overflow on Api variant ---

    #[test]
    fn test_overflow_typed_variant() {
        assert!(Error::ContextOverflow("too big".into()).is_context_overflow());
    }

    #[test]
    fn test_overflow_api_too_many_tokens() {
        let e = Error::api("invalid_request_error", "Too many tokens in the request");
        assert!(e.is_context_overflow());
    }

    #[test]
    fn test_overflow_api_context_length_exceeded() {
        let e = Error::api(
            "invalid_request_error",
            "This model's maximum context length is 200000 tokens. context_length_exceeded",
        );
        assert!(e.is_context_overflow());
    }

    #[test]
    fn test_overflow_api_prompt_too_long() {
        let e = Error::api("invalid_request_error", "Prompt is too long for this model");
        assert!(e.is_context_overflow());
    }

    #[test]
    fn test_overflow_api_request_too_large() {
        let e = Error::api("invalid_request_error", "Request too large");
        assert!(e.is_context_overflow());
    }

    #[test]
    fn test_overflow_api_reduce_length() {
        let e = Error::api(
            "invalid_request_error",
            "Please reduce the length of the messages",
        );
        assert!(e.is_context_overflow());
    }

    #[test]
    fn test_overflow_api_input_too_long() {
        let e = Error::api("invalid_request_error", "Input too long for model");
        assert!(e.is_context_overflow());
    }

    #[test]
    fn test_not_overflow_api_normal_error() {
        let e = Error::api("authentication_error", "Invalid API key");
        assert!(!e.is_context_overflow());
    }

    #[test]
    fn test_not_overflow_non_api() {
        assert!(!Error::InvalidApiKey.is_context_overflow());
        assert!(!Error::Aborted.is_context_overflow());
        assert!(!Error::RateLimited { retry_after: None }.is_context_overflow());
    }
}
