//! Configuration file support

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

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
    /// Whether to use TUI mode by default
    pub tui: Option<bool>,
    /// Custom system prompt file path
    pub system_prompt_file: Option<String>,
    /// API keys (alternative to environment variables)
    #[serde(default)]
    pub api_keys: ApiKeys,
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
        // Check for TAU_CONFIG_PATH env var first
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
                    eprintln!("Warning: Failed to parse config file: {}", e);
                    Self::default()
                }
            },
            Err(e) => {
                eprintln!("Warning: Failed to read config file: {}", e);
                Self::default()
            }
        }
    }

    /// Save config to file
    pub fn save(&self) -> std::io::Result<()> {
        let path = Self::config_path();
        let dir = path.parent().unwrap();
        fs::create_dir_all(dir)?;

        let content = toml::to_string_pretty(self).map_err(std::io::Error::other)?;
        fs::write(path, content)
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
            tui: Some(true),
            system_prompt_file: None,
            api_keys: ApiKeys::default(),
        };

        default_config.save()?;
        Ok(path)
    }

    /// Get API key for a provider, checking config then env (sync version)
    pub fn get_api_key(&self, provider: &str) -> Option<String> {
        // Check config first
        let from_config = match provider {
            "anthropic" => self.api_keys.anthropic.clone(),
            "openai" => self.api_keys.openai.clone(),
            "google" => self.api_keys.google.clone(),
            _ => None,
        };

        if from_config.is_some() {
            return from_config;
        }

        // Fall back to env var
        let env_var = match provider {
            "anthropic" => "ANTHROPIC_API_KEY",
            "openai" => "OPENAI_API_KEY",
            "google" => "GOOGLE_API_KEY",
            _ => return None,
        };

        std::env::var(env_var).ok()
    }

    /// Get API key for a provider, checking OAuth first, then config, then env
    pub async fn get_api_key_with_oauth(&self, provider: &str) -> Option<String> {
        // For Anthropic, check OAuth first
        if provider == "anthropic" {
            // Check OAuth storage (auto-refresh if needed)
            if let Some(token) =
                crate::oauth::get_oauth_token(crate::oauth::OAuthProvider::Anthropic).await
            {
                return Some(token);
            }

            // Check ANTHROPIC_OAUTH_TOKEN env var (manual OAuth token)
            if let Ok(token) = std::env::var("ANTHROPIC_OAUTH_TOKEN") {
                return Some(token);
            }
        }

        // Fall back to regular API key lookup
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
"#
}
