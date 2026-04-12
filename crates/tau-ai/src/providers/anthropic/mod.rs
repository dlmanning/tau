//! Anthropic Claude API provider

mod convert;
mod request;
mod streaming;
#[cfg(test)]
mod tests;

use reqwest::header::HeaderValue;
use reqwest_eventsource::EventSource;
use serde::Serialize;

use crate::{
    error::{Error, Result},
    stream::MessageEventStream,
    types::{Context, Model, StreamOptions},
};

use convert::{convert_messages, convert_tools, make_cache_control, split_system_prompt};
use request::{AnthropicRequest, OutputConfig, SystemBlock, ThinkingConfig};
use streaming::create_stream;

/// Cache scope for prompt caching
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheScope {
    /// Global scope — shared across all users/orgs (1P only)
    Global,
    /// Org scope — shared within an organization
    Org,
}

/// Anthropic-specific streaming options
#[derive(Debug, Clone, Default)]
pub struct AnthropicOptions {
    /// Base streaming options
    pub base: StreamOptions,
    /// Enable extended thinking
    pub thinking_enabled: bool,
    /// Use adaptive thinking (model decides when to think)
    pub thinking_adaptive: bool,
    /// Budget for thinking tokens (used when not adaptive)
    pub thinking_budget_tokens: Option<u32>,
    /// Thinking display mode ("summarized" or "omitted")
    pub thinking_display: Option<String>,
    /// Tool choice strategy
    pub tool_choice: Option<ToolChoice>,
    /// Cache scope for prompt caching breakpoints
    pub cache_scope: Option<CacheScope>,
    /// Cache TTL (e.g. "1h", "5m")
    pub cache_ttl: Option<String>,
    /// Dynamic boundary marker for system prompt splitting
    pub system_prompt_boundary: Option<String>,
    /// Request metadata (e.g. `{"user_id": "..."}`)
    pub metadata: Option<serde_json::Value>,
    /// Service tier ("auto" or "standard_only")
    pub service_tier: Option<String>,
    /// Effort level ("low", "medium", "high", "max")
    pub effort: Option<String>,
    /// Structured output format (JSON schema)
    pub output_format: Option<serde_json::Value>,
    /// Container ID for code execution sandboxing
    pub container: Option<String>,
    /// Top-level cache control — auto-applies to last cacheable block
    pub auto_cache_control: bool,
    /// Geographic region for inference processing
    pub inference_geo: Option<String>,
}

/// Tool choice strategy
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    None,
    Tool {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
}

impl ToolChoice {
    pub fn auto() -> Self {
        Self::Auto {
            disable_parallel_tool_use: None,
        }
    }
    pub fn any() -> Self {
        Self::Any {
            disable_parallel_tool_use: None,
        }
    }
    pub fn tool(name: impl Into<String>) -> Self {
        Self::Tool {
            name: name.into(),
            disable_parallel_tool_use: None,
        }
    }
}

const SPOOFED_SDK_VERSION: &str = "0.85.0";
const SPOOFED_LANG: HeaderValue = HeaderValue::from_static("js");
const SPOOFED_RUNTIME: HeaderValue = HeaderValue::from_static("node");
const SPOOFED_RUNTIME_VERSION: HeaderValue = HeaderValue::from_static("v22.12.0");
const ANTHROPIC_VERSION: HeaderValue = HeaderValue::from_static("2023-06-01");

/// Apply Stainless SDK identification headers.
///
/// These are required for OAuth tokens and expected by the Anthropic API for
/// analytics. We spoof the JS SDK identity because there is no official Rust SDK
/// and OAuth validation may check against known SDK versions.
fn apply_stainless_headers(headers: &mut reqwest::header::HeaderMap) {
    let os = match std::env::consts::OS {
        "macos" => "MacOS",
        "linux" => "Linux",
        "windows" => "Windows",
        "freebsd" => "FreeBSD",
        "openbsd" => "OpenBSD",
        _ => "Unknown",
    };
    let arch = match std::env::consts::ARCH {
        "x86" => "x32",
        "x86_64" => "x64",
        "arm" => "arm",
        "aarch64" => "arm64",
        other => other,
    };

    headers.insert(
        "User-Agent",
        format!("Anthropic/JS {}", SPOOFED_SDK_VERSION)
            .parse()
            .expect("valid User-Agent header"),
    );
    headers.insert("X-Stainless-Lang", SPOOFED_LANG);
    headers.insert(
        "X-Stainless-Package-Version",
        HeaderValue::from_static(SPOOFED_SDK_VERSION),
    );
    headers.insert("X-Stainless-OS", HeaderValue::from_static(os));
    headers.insert("X-Stainless-Arch", HeaderValue::from_static(arch));
    headers.insert("X-Stainless-Runtime", SPOOFED_RUNTIME);
    headers.insert("X-Stainless-Runtime-Version", SPOOFED_RUNTIME_VERSION);
    headers.insert("X-Stainless-Retry-Count", HeaderValue::from_static("0"));
}

/// Anthropic API client
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider with an API key
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
        }
    }

    /// Create from environment variable
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| Error::InvalidApiKey)?;
        Ok(Self::new(api_key))
    }

    /// Stream a response from Claude
    pub async fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&AnthropicOptions>,
    ) -> Result<MessageEventStream> {
        let default_options = AnthropicOptions::default();
        let opts = options.unwrap_or(&default_options);

        let request = self.build_request(model, context, opts)?;
        let url = format!("{}/v1/messages", model.base_url);

        tracing::debug!("Anthropic API URL: {}", url);

        let is_oauth = self.api_key.contains("sk-ant-oat");
        let mut headers = reqwest::header::HeaderMap::new();

        // Build beta headers based on features in use
        let mut betas = vec![
            "fine-grained-tool-streaming-2025-05-14",
            "token-efficient-tools-2026-03-28",
        ];
        if opts.thinking_enabled {
            betas.push("interleaved-thinking-2025-05-14");
        }
        if matches!(opts.cache_scope, Some(CacheScope::Global)) {
            betas.push("prompt-caching-scope-2026-01-05");
        }
        if !context.server_tools.is_empty() {
            betas.push("web-search-2025-03-05");
        }

        if is_oauth {
            betas.insert(0, "oauth-2025-04-20");
            headers.insert(
                "Authorization",
                format!("Bearer {}", self.api_key)
                    .parse()
                    .map_err(|_| Error::InvalidConfig("invalid API key for header".into()))?,
            );
            headers.insert(
                "anthropic-dangerous-direct-browser-access",
                HeaderValue::from_static("true"),
            );
        } else {
            headers.insert(
                "x-api-key",
                self.api_key
                    .parse()
                    .map_err(|_| Error::InvalidConfig("invalid API key for header".into()))?,
            );
        }
        headers.insert(
            "anthropic-beta",
            betas
                .join(",")
                .parse()
                .map_err(|_| Error::InvalidConfig("invalid beta header".into()))?,
        );
        headers.insert("accept", super::APPLICATION_JSON);
        headers.insert("content-type", super::APPLICATION_JSON);
        headers.insert("anthropic-version", ANTHROPIC_VERSION);

        apply_stainless_headers(&mut headers);

        for (key, value) in &model.headers {
            if let (Ok(name), Ok(val)) = (
                key.parse::<reqwest::header::HeaderName>(),
                value.parse::<reqwest::header::HeaderValue>(),
            ) {
                headers.insert(name, val);
            }
        }

        let request_builder = self
            .client
            .post(&url)
            .headers(headers.clone())
            .json(&request);

        let event_source = EventSource::new(request_builder)
            .map_err(|e| Error::Sse(format!("Failed to create event source: {}", e)))?;

        Ok(Box::pin(create_stream(event_source, model.clone())))
    }

    fn build_request(
        &self,
        model: &Model,
        context: &Context,
        options: &AnthropicOptions,
    ) -> Result<AnthropicRequest> {
        let is_oauth = self.api_key.contains("sk-ant-oat");
        let has_tools = !context.tools.is_empty();

        // Build system blocks first so we can count cache breakpoints accurately.
        let cache = || make_cache_control(&options.cache_scope, &options.cache_ttl);
        let system_blocks: Option<Vec<SystemBlock>> = if is_oauth {
            let mut blocks = vec![SystemBlock {
                block_type: "text".to_string(),
                text: "You are Claude Code, Anthropic's official CLI for Claude.".to_string(),
                cache_control: Some(cache()),
            }];
            if let Some(ref system_prompt) = context.system_prompt {
                blocks.extend(split_system_prompt(
                    system_prompt,
                    options.system_prompt_boundary.as_deref(),
                    &options.cache_scope,
                    &options.cache_ttl,
                ));
            }
            Some(blocks)
        } else {
            context.system_prompt.as_ref().map(|sp| {
                split_system_prompt(
                    sp,
                    options.system_prompt_boundary.as_deref(),
                    &options.cache_scope,
                    &options.cache_ttl,
                )
            })
        };

        // Count actual cache_control breakpoints in system blocks
        let system_cache_blocks = system_blocks
            .as_ref()
            .map(|blocks| blocks.iter().filter(|b| b.cache_control.is_some()).count())
            .unwrap_or(0);
        let tool_cache_blocks: usize = if has_tools { 1 } else { 0 };

        // Anthropic allows max 4 cache_control breakpoints total per request.
        let message_cache_budget = 4_usize.saturating_sub(system_cache_blocks + tool_cache_blocks);
        let messages = convert_messages(
            &context.messages,
            message_cache_budget,
            &options.cache_scope,
            &options.cache_ttl,
        );
        let has_server_tools = !context.server_tools.is_empty();
        let tools = if has_tools || has_server_tools {
            let mut all_tools: Vec<serde_json::Value> = vec![];

            // Add client tools
            if has_tools {
                let client_tools = convert_tools(
                    &context.tools,
                    !has_server_tools, // cache_last only if no server tools follow
                    &options.cache_scope,
                    &options.cache_ttl,
                );
                for tool in client_tools {
                    all_tools.push(serde_json::to_value(&tool).unwrap_or_default());
                }
            }

            // Add server tools (web search, etc.)
            for server_tool in &context.server_tools {
                if let Ok(val) = serde_json::to_value(server_tool) {
                    all_tools.push(val);
                }
            }

            Some(serde_json::Value::Array(all_tools))
        } else {
            None
        };

        let max_tokens = options.base.max_tokens.unwrap_or(model.max_tokens / 3);

        // Build output_config if effort or format is specified
        let output_config = if options.effort.is_some() || options.output_format.is_some() {
            Some(OutputConfig {
                effort: options.effort.clone(),
                format: options.output_format.clone(),
            })
        } else {
            None
        };

        let mut request = AnthropicRequest {
            model: model.id.clone(),
            messages,
            max_tokens,
            stream: true,
            system: system_blocks,
            temperature: options.base.temperature,
            tools,
            tool_choice: options.tool_choice.clone(),
            thinking: None,
            stop_sequences: options.base.stop_sequences.clone(),
            metadata: options.metadata.clone(),
            service_tier: options.service_tier.clone(),
            output_config,
            container: options.container.clone(),
            cache_control: if options.auto_cache_control {
                Some(make_cache_control(&options.cache_scope, &options.cache_ttl))
            } else {
                None
            },
            inference_geo: options.inference_geo.clone(),
        };

        if options.thinking_enabled && model.reasoning {
            let display = options.thinking_display.clone();
            request.thinking = Some(if options.thinking_adaptive {
                ThinkingConfig::Adaptive {
                    thinking_type: "adaptive".to_string(),
                    display,
                }
            } else {
                ThinkingConfig::Enabled {
                    thinking_type: "enabled".to_string(),
                    budget_tokens: options.thinking_budget_tokens.unwrap_or(1024),
                    display,
                }
            });
        }

        Ok(request)
    }
}

/// Stream a response from Anthropic Claude
pub async fn stream_anthropic(
    model: &Model,
    context: &Context,
    options: Option<&AnthropicOptions>,
) -> Result<MessageEventStream> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| Error::InvalidApiKey)?;
    let provider = AnthropicProvider::new(api_key);
    provider.stream(model, context, options).await
}
