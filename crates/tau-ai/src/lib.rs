//! tau-ai: Unified LLM provider abstraction layer
//!
//! This crate provides a common interface for interacting with various LLM providers
//! including Anthropic, OpenAI, and Google.

pub mod error;
pub mod models;
mod models_generated;
pub mod providers;
pub mod stream;
pub mod types;

pub use error::{Error, Result};
pub use stream::MessageEventStream;
pub use types::*;
