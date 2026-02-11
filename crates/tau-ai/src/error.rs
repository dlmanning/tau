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
        matches!(
            self,
            Error::Http(_) | Error::RateLimited { .. } | Error::Sse(_)
        )
    }
}
