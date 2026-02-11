//! LLM Provider implementations

pub mod anthropic;
pub mod google;
pub mod openai;

use crate::{Context, Error, MessageEventStream, Model, Result, StreamOptions};
use async_trait::async_trait;

/// Trait for LLM providers
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Stream a response from the LLM
    async fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> Result<MessageEventStream>;
}

/// Get an API key from environment or provided value
pub fn get_api_key(provided: Option<&str>, env_var: &str) -> Result<String> {
    if let Some(key) = provided {
        return Ok(key.to_string());
    }

    std::env::var(env_var).map_err(|_| Error::InvalidApiKey)
}
