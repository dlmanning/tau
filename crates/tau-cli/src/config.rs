//! Configuration file support

use std::{fs, path::PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use tau_agent::{AgentConfig, CompactionConfig, DequeueMode};
use tau_ai::{Model, ReasoningLevel};

/// Configuration for tau
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Default model to use
    pub model: Option<String>,
    /// Default provider
    pub provider: Option<String>,
    /// Default reasoning level
    pub reasoning_level: Option<String>,
    /// Use adaptive thinking (model decides when to think)
    pub thinking_adaptive: Option<bool>,
    /// Whether to use TUI mode by default
    pub tui: Option<bool>,
    /// API keys (alternative to environment variables)
    #[serde(default)]
    pub api_keys: ApiKeys,
    /// Compaction settings
    #[serde(default)]
    pub compaction: Option<CompactionSettings>,
    /// Cache settings
    #[serde(default)]
    pub cache: Option<CacheSettings>,
    /// Enable Anthropic-internal prompt additions (stricter verification,
    /// comment philosophy, faithful reporting, richer communication style)
    pub acolyte_mode: Option<bool>,
    /// MCP server definitions keyed by server name (BTreeMap so the
    /// derived tool list — and the provider prompt cache — is stable).
    #[serde(default)]
    pub mcp_servers: std::collections::BTreeMap<String, McpServerConfig>,
}

/// One MCP server in `[mcp_servers.<name>]`. Exactly one of `command`
/// (stdio) or `url` (streamable HTTP) must be set.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct McpServerConfig {
    // stdio transport
    pub command: Option<String>,
    pub args: Vec<String>,
    /// Extra environment for the child; values support `${VAR}`
    /// expansion at connect time.
    pub env: std::collections::BTreeMap<String, String>,
    // http transport
    pub url: Option<String>,
    /// Only `Authorization` is supported in v1; values support `${VAR}`.
    pub headers: std::collections::BTreeMap<String, String>,
    // common
    pub enabled: Option<bool>,
    /// Per tool-call timeout in seconds (default 60).
    pub timeout_secs: Option<u64>,
    /// "untrusted" (default): tools are approval-gated unless the
    /// server annotates them read-only. "trusted": all of this
    /// server's tools auto-approve. Annotations are server-asserted —
    /// only mark servers you run yourself as trusted.
    pub trust: Option<String>,
    /// Remote tool names to expose (omit = all).
    pub include_tools: Option<Vec<String>>,
    /// Remote tool names to hide (applied after include_tools).
    pub exclude_tools: Vec<String>,
}

/// Settings for prompt caching
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheSettings {
    /// Cache scope: "global" or "org"
    pub scope: Option<String>,
    /// Cache TTL: "1h" or "5m"
    pub ttl: Option<String>,
    /// Dynamic boundary marker for system prompt splitting
    pub prompt_boundary: Option<String>,
}

/// Settings for context compaction
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CompactionSettings {
    /// Whether compaction is enabled (default: true)
    pub enabled: Option<bool>,
    /// Reserve this many tokens below context window to trigger proactive compaction
    pub reserve_tokens: Option<u64>,
    /// Keep at least this many tokens of recent messages when compacting
    pub keep_recent_tokens: Option<u64>,
}

/// API key configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiKeys {
    pub anthropic: Option<String>,
    pub openai: Option<String>,
    pub google: Option<String>,
}

impl Config {
    /// Get the config directory
    pub fn config_dir() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("tau")
    }

    /// Get the config file path
    pub fn config_path() -> PathBuf {
        if let Ok(path) = std::env::var("TAU_CONFIG_PATH") {
            return PathBuf::from(path);
        }
        Self::config_dir().join("config.toml")
    }

    /// Load config from file
    pub fn load() -> anyhow::Result<Self> {
        let path = Self::config_path();
        if !path.exists() {
            return Ok(Self::default());
        }

        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read config file {}", path.display()))?;

        let config: Self = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    /// Validate config values
    fn validate(&self) -> anyhow::Result<()> {
        if let Some(ref level) = self.reasoning_level {
            match level.as_str() {
                "off" | "minimal" | "low" | "medium" | "high" => {}
                _ => anyhow::bail!(
                    "Invalid reasoning_level '{}' in config. Valid values: off, minimal, low, medium, high",
                    level
                ),
            }
        }
        for (name, server) in &self.mcp_servers {
            let name_ok = !name.is_empty()
                && name.len() <= 32
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
            if !name_ok {
                anyhow::bail!(
                    "Invalid MCP server name '{}': use 1-32 chars of [a-zA-Z0-9_-]",
                    name
                );
            }
            match (&server.command, &server.url) {
                (Some(_), Some(_)) => anyhow::bail!(
                    "MCP server '{}': set either `command` (stdio) or `url` (http), not both",
                    name
                ),
                (None, None) => anyhow::bail!(
                    "MCP server '{}': set `command` (stdio) or `url` (http)",
                    name
                ),
                _ => {}
            }
            if server.url.is_none() && !server.headers.is_empty() {
                anyhow::bail!("MCP server '{}': `headers` requires `url`", name);
            }
            if server.command.is_none() && (!server.args.is_empty() || !server.env.is_empty()) {
                anyhow::bail!("MCP server '{}': `args`/`env` require `command`", name);
            }
            if let Some(key) = server.headers.keys().find(|k| !k.eq_ignore_ascii_case("authorization")) {
                anyhow::bail!(
                    "MCP server '{}': unsupported header '{}' (only Authorization is supported in v1)",
                    name,
                    key
                );
            }
            if let Some(ref trust) = server.trust
                && trust != "trusted"
                && trust != "untrusted"
            {
                anyhow::bail!(
                    "MCP server '{}': invalid trust '{}'. Valid values: trusted, untrusted",
                    name,
                    trust
                );
            }
            if server.timeout_secs == Some(0) {
                anyhow::bail!("MCP server '{}': timeout_secs must be >= 1", name);
            }
        }
        if let Some(ref cache) = self.cache {
            if let Some(ref scope) = cache.scope {
                match scope.as_str() {
                    "global" | "org" => {}
                    _ => anyhow::bail!(
                        "Invalid cache.scope '{}' in config. Valid values: global, org",
                        scope
                    ),
                }
            }
            if let Some(ref ttl) = cache.ttl {
                match ttl.as_str() {
                    "1h" | "5m" => {}
                    _ => anyhow::bail!(
                        "Invalid cache.ttl '{}' in config. Valid values: 1h, 5m",
                        ttl
                    ),
                }
            }
        }
        Ok(())
    }

    /// Save config to file
    pub fn save(&self) -> std::io::Result<()> {
        let path = Self::config_path();
        let dir = path.parent().expect("config path has parent directory");
        fs::create_dir_all(dir)?;

        let content = toml::to_string_pretty(self).map_err(std::io::Error::other)?;
        fs::write(&path, content)?;

        // Restrict permissions since the file may contain API keys
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
        }

        Ok(())
    }

    /// Create a default config file if it doesn't exist
    pub fn init() -> std::io::Result<PathBuf> {
        let path = Self::config_path();
        if path.exists() {
            return Ok(path);
        }

        let default_config = Config {
            model: Some("claude-sonnet-4-5-20250929".to_string()),
            provider: Some("anthropic".to_string()),
            reasoning_level: Some("off".to_string()),
            thinking_adaptive: None,
            tui: Some(true),
            api_keys: ApiKeys::default(),
            compaction: None,
            cache: None,
            acolyte_mode: None,
            mcp_servers: Default::default(),
        };

        default_config.save()?;
        Ok(path)
    }

    /// Convert configured MCP servers into transport-agnostic specs
    /// for `tau_tools::mcp::McpManager`, dropping `enabled = false`
    /// entries. `${VAR}` values stay unexpanded — the manager expands
    /// at connect time so a missing variable fails only that server.
    pub fn mcp_specs(&self) -> Vec<tau_tools::mcp::McpServerSpec> {
        use tau_tools::mcp::{McpServerSpec, McpTransportSpec};
        self.mcp_servers
            .iter()
            .filter(|(_, s)| s.enabled != Some(false))
            .map(|(name, s)| {
                let transport = if let Some(url) = &s.url {
                    McpTransportSpec::Http {
                        url: url.clone(),
                        auth_header: s
                            .headers
                            .iter()
                            .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
                            .map(|(_, v)| v.clone()),
                    }
                } else {
                    McpTransportSpec::Stdio {
                        command: s.command.clone().unwrap_or_default(),
                        args: s.args.clone(),
                        env: s.env.clone(),
                    }
                };
                McpServerSpec {
                    name: name.clone(),
                    transport,
                    call_timeout: std::time::Duration::from_secs(s.timeout_secs.unwrap_or(60)),
                    trust: if s.trust.as_deref() == Some("trusted") {
                        tau_tools::mcp::McpTrust::Trusted
                    } else {
                        tau_tools::mcp::McpTrust::Untrusted
                    },
                    include_tools: s.include_tools.clone(),
                    exclude_tools: s.exclude_tools.clone(),
                }
            })
            .collect()
    }

    /// Get API key for a provider, checking config then env (sync version)
    pub fn get_api_key(&self, provider: &str) -> Option<String> {
        let from_config = match provider {
            "anthropic" => self.api_keys.anthropic.clone(),
            "openai" => self.api_keys.openai.clone(),
            "google" => self.api_keys.google.clone(),
            _ => None,
        };

        if from_config.is_some() {
            tracing::warn!(
                "Using API key from config file. Consider using environment variables instead for better security."
            );
            return from_config;
        }

        let env_var = match provider {
            "anthropic" => "ANTHROPIC_API_KEY",
            "openai" => "OPENAI_API_KEY",
            "google" => "GOOGLE_API_KEY",
            "groq" => "GROQ_API_KEY",
            "cerebras" => "CEREBRAS_API_KEY",
            "xai" => "XAI_API_KEY",
            "openrouter" => "OPENROUTER_API_KEY",
            _ => return None,
        };

        std::env::var(env_var).ok()
    }

    /// Build the runtime [`AgentConfig`] from this config plus CLI-supplied
    /// `model` and `reasoning` (which override any config values).
    pub fn to_agent_config(&self, model: Model, reasoning: ReasoningLevel) -> AgentConfig {
        // Start from the runtime default (fraction-scaled across model
        // sizes) and only override what the user explicitly pinned in
        // their TOML. Setting `reserve_tokens` or `keep_recent_tokens`
        // switches that knob to an absolute count — a sensible
        // interpretation of "the user wrote a number, they meant that
        // number." Leaving them unset means the fraction default
        // applies and the values scale with the model's context window.
        let compaction = match self.compaction.as_ref() {
            Some(c) => {
                let mut cfg = CompactionConfig::default();
                if let Some(enabled) = c.enabled {
                    cfg.enabled = enabled;
                }
                if let Some(n) = c.reserve_tokens {
                    cfg.reserve = tau_agent::CompactionThreshold::Tokens(n);
                }
                if let Some(n) = c.keep_recent_tokens {
                    cfg.keep_recent = tau_agent::CompactionThreshold::Tokens(n);
                }
                cfg
            }
            None => CompactionConfig::default(),
        };
        let mut builder = AgentConfig::builder(model)
            .reasoning(reasoning)
            .thinking_adaptive(self.thinking_adaptive.unwrap_or(false))
            .max_turns(200)
            .compaction(compaction)
            .steering_mode(DequeueMode::All)
            .follow_up_mode(DequeueMode::All);
        if let Some(cache) = &self.cache {
            if let Some(scope) = &cache.scope {
                builder = builder.cache_scope(scope.clone());
            }
            if let Some(ttl) = &cache.ttl {
                builder = builder.cache_ttl(ttl.clone());
            }
            if let Some(boundary) = &cache.prompt_boundary {
                builder = builder.system_prompt_boundary(boundary.clone());
            }
        }
        builder.build()
    }

    /// Get API key for a provider, checking OAuth first, then config, then env
    pub async fn get_api_key_with_oauth(&self, provider: &str) -> Option<String> {
        if provider == "anthropic" {
            if let Some(token) =
                crate::oauth::get_oauth_token(crate::oauth::OAuthProvider::Anthropic).await
            {
                return Some(token);
            }

            if let Ok(token) = std::env::var("ANTHROPIC_OAUTH_TOKEN") {
                return Some(token);
            }
        }

        self.get_api_key(provider)
    }
}

/// Generate example config content
pub fn example_config() -> &'static str {
    r#"# tau configuration file
# Place at ~/.config/tau/config.toml (Linux/Mac) or %APPDATA%\tau\config.toml (Windows)

# Default model to use
model = "claude-sonnet-4-5-20250929"

# Default provider (anthropic, openai, google)
provider = "anthropic"

# Default reasoning level (off, minimal, low, medium, high)
reasoning_level = "off"

# Use adaptive thinking — model decides when and how much to think
# thinking_adaptive = true

# Whether to use TUI mode by default (true by default)
# Set to false for simple stdin/stdout mode
tui = true

# MCP servers (optional). Tools appear to the agent as mcp__<server>__<tool>.
# Values support ${VAR} expansion from the environment at connect time.
# [mcp_servers.linear]
# command = "npx"
# args = ["-y", "@linear/mcp-server"]
# env = { LINEAR_API_KEY = "${LINEAR_API_KEY}" }
# trust = "untrusted"   # "trusted" auto-approves this server's tools
# timeout_secs = 60     # per tool call
# # include_tools = ["create_issue"]
# # exclude_tools = ["delete_issue"]
#
# [mcp_servers.docs]
# url = "https://example.com/mcp"
# headers = { Authorization = "Bearer ${DOCS_TOKEN}" }

# API keys (optional - can also use environment variables)
# It's recommended to use environment variables instead for security
[api_keys]
# anthropic = "sk-ant-..."
# openai = "sk-..."
# google = "..."

# Context compaction settings (optional). By default these scale with
# the model's context window (8% reserve / 10% kept-recent), so a
# 200K-context model gets ~16K headroom and a 32K-context model gets
# ~2.5K — small models won't reserve half their window. Pin absolute
# counts here only if you need explicit control across all models.
# [compaction]
# enabled = true
# reserve_tokens = 16384
# keep_recent_tokens = 20000

# Prompt caching settings (optional)
# [cache]
# scope = "org"           # "global" (1P only) or "org"
# ttl = "1h"              # "1h" or "5m" (default: 5m)
# prompt_boundary = "<!-- DYNAMIC_BOUNDARY -->"
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(toml_str: &str) -> anyhow::Result<Config> {
        let config: Config = toml::from_str(toml_str)?;
        config.validate()?;
        Ok(config)
    }

    #[test]
    fn mcp_stdio_server_parses_and_converts() {
        let cfg = parse(
            r#"
[mcp_servers.linear]
command = "npx"
args = ["-y", "@linear/mcp-server"]
env = { KEY = "${KEY}" }
trust = "trusted"
timeout_secs = 30
"#,
        )
        .unwrap();
        let specs = cfg.mcp_specs();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "linear");
        assert_eq!(specs[0].trust, tau_tools::mcp::McpTrust::Trusted);
        assert_eq!(specs[0].call_timeout.as_secs(), 30);
        assert!(matches!(
            &specs[0].transport,
            tau_tools::mcp::McpTransportSpec::Stdio { command, .. } if command == "npx"
        ));
    }

    #[test]
    fn mcp_http_server_maps_authorization_header() {
        let cfg = parse(
            r#"
[mcp_servers.docs]
url = "https://example.com/mcp"
headers = { Authorization = "Bearer x" }
"#,
        )
        .unwrap();
        let specs = cfg.mcp_specs();
        assert!(matches!(
            &specs[0].transport,
            tau_tools::mcp::McpTransportSpec::Http { auth_header: Some(h), .. } if h == "Bearer x"
        ));
    }

    #[test]
    fn mcp_disabled_server_is_skipped() {
        let cfg = parse("[mcp_servers.off]\ncommand = \"x\"\nenabled = false\n").unwrap();
        assert!(cfg.mcp_specs().is_empty());
    }

    #[test]
    fn mcp_validation_rejects_bad_configs() {
        // Both transports set.
        assert!(parse("[mcp_servers.a]\ncommand = \"x\"\nurl = \"http://y\"\n").is_err());
        // Neither transport set.
        assert!(parse("[mcp_servers.a]\ntrust = \"trusted\"\n").is_err());
        // Bad name.
        assert!(parse("[mcp_servers.\"has space\"]\ncommand = \"x\"\n").is_err());
        // Unsupported header.
        assert!(
            parse("[mcp_servers.a]\nurl = \"http://y\"\nheaders = { X-Custom = \"v\" }\n").is_err()
        );
        // Bad trust value.
        assert!(parse("[mcp_servers.a]\ncommand = \"x\"\ntrust = \"sorta\"\n").is_err());
        // Zero timeout.
        assert!(parse("[mcp_servers.a]\ncommand = \"x\"\ntimeout_secs = 0\n").is_err());
        // headers without url / env without command.
        assert!(parse("[mcp_servers.a]\ncommand = \"x\"\nheaders = { Authorization = \"y\" }\n").is_err());
        assert!(parse("[mcp_servers.a]\nurl = \"http://y\"\nenv = { K = \"v\" }\n").is_err());
    }
}
