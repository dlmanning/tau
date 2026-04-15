//! Core types for LLM interactions

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Supported API types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Api {
    AnthropicMessages,
    OpenAICompletions,
    OpenAIResponses,
    GoogleGenerativeAI,
    Ollama,
}

/// Known LLM providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Anthropic,
    OpenAI,
    Google,
    Groq,
    Cerebras,
    XAI,
    OpenRouter,
    Ollama,
    Custom,
}

impl Provider {
    /// Get a human-readable name for this provider
    pub fn name(&self) -> &'static str {
        match self {
            Provider::Anthropic => "Anthropic",
            Provider::OpenAI => "OpenAI",
            Provider::Google => "Google",
            Provider::Groq => "Groq",
            Provider::Cerebras => "Cerebras",
            Provider::XAI => "xAI",
            Provider::OpenRouter => "OpenRouter",
            Provider::Ollama => "Ollama",
            Provider::Custom => "Custom",
        }
    }

    /// Get the environment variable name for this provider's API key
    pub fn api_key_env_var(&self) -> Option<&'static str> {
        match self {
            Provider::Anthropic => Some("ANTHROPIC_API_KEY"),
            Provider::OpenAI => Some("OPENAI_API_KEY"),
            Provider::Google => Some("GOOGLE_API_KEY"),
            Provider::Groq => Some("GROQ_API_KEY"),
            Provider::Cerebras => Some("CEREBRAS_API_KEY"),
            Provider::XAI => Some("XAI_API_KEY"),
            Provider::OpenRouter => Some("OPENROUTER_API_KEY"),
            Provider::Ollama => None,
            Provider::Custom => None,
        }
    }

    /// Parse a provider from a case-insensitive string identifier.
    pub fn from_id(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "anthropic" => Provider::Anthropic,
            "openai" => Provider::OpenAI,
            "google" => Provider::Google,
            "groq" => Provider::Groq,
            "cerebras" => Provider::Cerebras,
            "xai" => Provider::XAI,
            "openrouter" => Provider::OpenRouter,
            "ollama" => Provider::Ollama,
            _ => Provider::Custom,
        }
    }

    /// Default API type for this provider.
    pub fn default_api(&self) -> Api {
        match self {
            Provider::Anthropic => Api::AnthropicMessages,
            Provider::OpenAI => Api::OpenAIResponses,
            Provider::Google => Api::GoogleGenerativeAI,
            Provider::Ollama => Api::Ollama,
            _ => Api::OpenAICompletions,
        }
    }

    /// Default base URL for this provider's API.
    pub fn default_base_url(&self) -> &'static str {
        match self {
            Provider::Anthropic => "https://api.anthropic.com",
            Provider::OpenAI => "https://api.openai.com/v1",
            Provider::Google => "https://generativelanguage.googleapis.com/v1beta",
            Provider::Groq => "https://api.groq.com/openai/v1",
            Provider::Cerebras => "https://api.cerebras.ai/v1",
            Provider::XAI => "https://api.x.ai/v1",
            Provider::OpenRouter => "https://openrouter.ai/api/v1",
            Provider::Ollama => "http://localhost:11434",
            Provider::Custom => "",
        }
    }
}

/// Cost information for a model (per million tokens)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostInfo {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    /// Thinking/reasoning tokens cost (for extended thinking models)
    #[serde(default)]
    pub thinking: f64,
}

/// Model definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// Model identifier (e.g., "claude-sonnet-4-5-20250514")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// API type to use
    pub api: Api,
    /// Provider
    pub provider: Provider,
    /// Base URL for API calls
    pub base_url: String,
    /// Whether the model supports reasoning/thinking
    pub reasoning: bool,
    /// Supported input types
    pub input_types: Vec<InputType>,
    /// Cost per million tokens
    pub cost: CostInfo,
    /// Context window size in tokens
    pub context_window: u32,
    /// Maximum output tokens
    pub max_tokens: u32,
    /// Additional headers for API calls
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

/// Supported input types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InputType {
    Text,
    Image,
}

/// Token usage information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    /// Thinking/reasoning tokens (Claude extended thinking)
    pub thinking: u64,
    /// Granular cache creation breakdown by TTL tier
    #[serde(default)]
    pub cache_creation_1h: u64,
    /// Granular cache creation breakdown by TTL tier
    #[serde(default)]
    pub cache_creation_5m: u64,
    /// Service tier used for this request ("standard", "priority", "batch")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

impl Usage {
    /// Calculate cost for this usage given a model
    pub fn calculate_cost(&self, model: &Model) -> CostBreakdown {
        // input_tokens from the API includes cache hits/writes — subtract them
        // to get the uncached portion charged at the full input rate.
        let uncached_input = self
            .input
            .saturating_sub(self.cache_read)
            .saturating_sub(self.cache_write);
        let input = (uncached_input as f64 / 1_000_000.0) * model.cost.input;
        let output = (self.output as f64 / 1_000_000.0) * model.cost.output;
        let cache_read = (self.cache_read as f64 / 1_000_000.0) * model.cost.cache_read;
        let cache_write = (self.cache_write as f64 / 1_000_000.0) * model.cost.cache_write;
        let thinking = (self.thinking as f64 / 1_000_000.0) * model.cost.thinking;

        CostBreakdown {
            input,
            output,
            cache_read,
            cache_write,
            thinking,
            total: input + output + cache_read + cache_write + thinking,
        }
    }
}

/// Cost breakdown in dollars
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub thinking: f64,
    pub total: f64,
}

/// Reason why generation stopped
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Natural end of response
    Stop,
    /// Maximum tokens reached
    Length,
    /// Tool use requested
    ToolUse,
    /// Error occurred
    Error,
    /// Request was aborted
    Aborted,
}

/// Content types in messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    /// Text content
    Text { text: String },
    /// Image content (base64 encoded)
    Image { data: String, mime_type: String },
    /// Thinking/reasoning content
    Thinking {
        thinking: String,
        /// Anthropic thinking block signature for cache efficiency
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Tool call request
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    /// Redacted thinking block (content hidden, signature preserved)
    RedactedThinking { data: String },
    /// Server-initiated tool use (web search, code execution, etc.)
    ServerToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Server tool result (web search results, etc.)
    ServerToolResult {
        tool_use_id: String,
        content: serde_json::Value,
        /// Original API block type for round-tripping (e.g. "web_search_tool_result")
        #[serde(default)]
        api_type: String,
    },
}

impl Content {
    /// Create text content
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create image content from base64 data
    pub fn image(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self::Image {
            data: data.into(),
            mime_type: mime_type.into(),
        }
    }

    /// Create thinking content
    pub fn thinking(thinking: impl Into<String>) -> Self {
        Self::Thinking {
            thinking: thinking.into(),
            signature: None,
        }
    }

    /// Create thinking content with a signature
    pub fn thinking_with_signature(
        thinking: impl Into<String>,
        signature: impl Into<String>,
    ) -> Self {
        Self::Thinking {
            thinking: thinking.into(),
            signature: Some(signature.into()),
        }
    }

    /// Create a tool call
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    /// Get text if this is text content
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Check if this is a tool call
    pub fn is_tool_call(&self) -> bool {
        matches!(self, Self::ToolCall { .. })
    }
}

/// Source of a system-injected message (not from user or model).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InjectionSource {
    /// A subagent completed successfully.
    SubagentCompleted {
        agent_id: String,
        description: String,
    },
    /// A subagent failed.
    SubagentFailed {
        agent_id: String,
        description: String,
    },
}

/// Message roles
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    /// User message
    User {
        content: Vec<Content>,
        #[serde(default)]
        timestamp: i64,
    },
    /// Assistant response
    Assistant {
        content: Vec<Content>,
        #[serde(flatten)]
        metadata: AssistantMetadata,
    },
    /// Tool result
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        content: Vec<Content>,
        #[serde(default)]
        is_error: bool,
        #[serde(default)]
        timestamp: i64,
    },
    /// System-injected message (e.g. subagent completion notification).
    /// Not from the user or the model. Converted to a user-role message
    /// before being sent to LLM APIs.
    SystemInjection {
        content: Vec<Content>,
        source: InjectionSource,
    },
}

/// Metadata for assistant messages
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AssistantMetadata {
    pub api: Option<Api>,
    pub provider: Option<Provider>,
    pub model: Option<String>,
    #[serde(default)]
    pub usage: Usage,
    pub stop_reason: Option<StopReason>,
    pub error_message: Option<String>,
    #[serde(default)]
    pub timestamp: i64,
}

impl Message {
    /// Create a user message with text content
    pub fn user(text: impl Into<String>) -> Self {
        Self::User {
            content: vec![Content::text(text)],
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Create a user message with multiple content blocks
    pub fn user_with_content(content: Vec<Content>) -> Self {
        Self::User {
            content,
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Create an empty assistant message
    pub fn assistant_empty() -> Self {
        Self::Assistant {
            content: vec![],
            metadata: AssistantMetadata {
                timestamp: chrono::Utc::now().timestamp_millis(),
                ..Default::default()
            },
        }
    }

    /// Create a tool result message
    pub fn tool_result(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: Vec<Content>,
        is_error: bool,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            content,
            is_error,
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Create a system injection message for subagent completion.
    pub fn subagent_completed(
        agent_id: impl Into<String>,
        description: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self::SystemInjection {
            content: vec![Content::text(text)],
            source: InjectionSource::SubagentCompleted {
                agent_id: agent_id.into(),
                description: description.into(),
            },
        }
    }

    /// Create a system injection message for subagent failure.
    pub fn subagent_failed(
        agent_id: impl Into<String>,
        description: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self::SystemInjection {
            content: vec![Content::text(error)],
            source: InjectionSource::SubagentFailed {
                agent_id: agent_id.into(),
                description: description.into(),
            },
        }
    }

    /// Get the role as a string
    pub fn role(&self) -> &'static str {
        match self {
            Self::User { .. } => "user",
            Self::Assistant { .. } => "assistant",
            Self::ToolResult { .. } => "tool_result",
            Self::SystemInjection { .. } => "system_injection",
        }
    }

    /// Get the content blocks
    pub fn content(&self) -> &[Content] {
        match self {
            Self::User { content, .. }
            | Self::Assistant { content, .. }
            | Self::ToolResult { content, .. }
            | Self::SystemInjection { content, .. } => content,
        }
    }

    /// Extract all tool calls from an assistant message
    pub fn tool_calls(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        match self {
            Self::Assistant { content, .. } => content
                .iter()
                .filter_map(|c| match c {
                    Content::ToolCall {
                        id,
                        name,
                        arguments,
                    } => Some((id.as_str(), name.as_str(), arguments)),
                    _ => None,
                })
                .collect(),
            Self::User { .. } | Self::ToolResult { .. } | Self::SystemInjection { .. } => vec![],
        }
    }

    /// Get combined text content
    pub fn text(&self) -> String {
        self.content()
            .iter()
            .filter_map(|c| c.as_text())
            .collect::<Vec<_>>()
            .join("")
    }
}

/// Server-controlled tool definitions (handled by the API, not the client)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerTool {
    /// Anthropic web search tool
    #[serde(rename = "web_search_20250305")]
    WebSearch {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_uses: Option<u8>,
        #[serde(skip_serializing_if = "Option::is_none")]
        allowed_domains: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        blocked_domains: Option<Vec<String>>,
    },
}

/// Tool definition for function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool name (used in API calls)
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// JSON Schema for parameters
    pub parameters: serde_json::Value,
}

impl Tool {
    /// Create a new tool definition
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// Context for an LLM request
#[derive(Debug, Clone, Default)]
pub struct Context {
    /// System prompt
    pub system_prompt: Option<String>,
    /// Conversation messages
    pub messages: Vec<Message>,
    /// Available tools
    pub tools: Vec<Tool>,
    /// Server-controlled tools (e.g. web search)
    #[allow(dead_code)]
    pub server_tools: Vec<ServerTool>,
}

impl Context {
    /// Create a new context with a system prompt
    pub fn with_system(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: Some(system_prompt.into()),
            messages: vec![],
            tools: vec![],
            server_tools: vec![],
        }
    }

    /// Add a message to the context
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Add a tool to the context
    pub fn add_tool(&mut self, tool: Tool) {
        self.tools.push(tool);
    }
}

/// Options for streaming requests
#[derive(Debug, Clone, Default)]
pub struct StreamOptions {
    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,
    /// Temperature (0.0 - 2.0)
    pub temperature: Option<f32>,
    /// Enable reasoning/thinking mode
    pub reasoning: Option<ReasoningLevel>,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
}

/// Reasoning/thinking level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ReasoningLevel {
    #[default]
    Off,
    Minimal,
    Low,
    Medium,
    High,
}
