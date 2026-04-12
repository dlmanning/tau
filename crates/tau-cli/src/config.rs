//! Configuration file support

use std::{fs, path::PathBuf};

use serde::{Deserialize, Serialize};

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
    /// Custom system prompt file path
    pub system_prompt_file: Option<String>,
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
    pub reserve_tokens: Option<u32>,
    /// Keep at least this many tokens of recent messages when compacting
    pub keep_recent_tokens: Option<u32>,
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
    pub fn load() -> Self {
        let path = Self::config_path();
        if !path.exists() {
            return Self::default();
        }

        match fs::read_to_string(&path) {
            Ok(content) => match toml::from_str(&content) {
                Ok(config) => config,
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to parse config file {}: {}",
                        path.display(),
                        e
                    );
                    Self::default()
                }
            },
            Err(e) => {
                eprintln!(
                    "Warning: Failed to read config file {}: {}",
                    path.display(),
                    e
                );
                Self::default()
            }
        }
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
            system_prompt_file: None,
            api_keys: ApiKeys::default(),
            compaction: None,
            cache: None,
            acolyte_mode: None,
        };

        default_config.save()?;
        Ok(path)
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

# Custom system prompt file (optional)
# system_prompt_file = "~/.config/tau/system_prompt.txt"

# API keys (optional - can also use environment variables)
# It's recommended to use environment variables instead for security
[api_keys]
# anthropic = "sk-ant-..."
# openai = "sk-..."
# google = "..."

# Context compaction settings (optional)
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
