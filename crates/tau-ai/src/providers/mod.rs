//! LLM Provider implementations

pub mod anthropic;
pub mod google;
pub mod ollama;
pub mod openai;

use crate::{Error, Result};
use reqwest::header::HeaderValue;

/// Common header values shared across providers
const APPLICATION_JSON: HeaderValue = HeaderValue::from_static("application/json");

/// Get an API key from environment or provided value
pub fn get_api_key(provided: Option<&str>, env_var: &str) -> Result<String> {
    if let Some(key) = provided {
        return Ok(key.to_string());
    }

    std::env::var(env_var).map_err(|_| Error::InvalidApiKey)
}
