//! CLI argument parsing and model resolution.
//!
//! Subcommand structure:
//!
//! ```text
//! tau                       # interactive (TUI) — default
//! tau run <prompt>          # one-shot non-interactive
//! tau auth login <provider>
//! tau auth logout <provider>
//! tau auth status
//! tau sessions ls
//! tau sessions resume <id>
//! tau config init
//! ```
//!
//! Common runtime flags (`--model`, `--provider`, `--reasoning`,
//! `--working-dir`, `--no-tui`, `--verbose`) sit on the top level and
//! apply to the implicit-default and `run` commands.

use clap::{Parser, Subcommand};
use tau_ai::{CostInfo, InputType, Model, Provider, ReasoningLevel};

/// tau - AI-powered coding agent
#[derive(Parser, Debug)]
#[command(name = "tau")]
#[command(author, version, about, long_about = None)]
pub(crate) struct Args {
    /// Model to use (default: claude-sonnet-4-5-20250929)
    #[arg(short, long, global = true)]
    pub model: Option<String>,

    /// Provider (anthropic, openai, google)
    #[arg(short, long, global = true)]
    pub provider: Option<String>,

    /// Enable reasoning/thinking mode
    #[arg(short, long, global = true)]
    pub reasoning: bool,

    /// Reasoning level (off, minimal, low, medium, high)
    #[arg(long, global = true)]
    pub reasoning_level: Option<String>,

    /// Working directory
    #[arg(short, long, global = true)]
    pub working_dir: Option<String>,

    /// Verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Disable TUI mode (use simple stdin/stdout)
    #[arg(long, global = true)]
    pub no_tui: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Command {
    /// Run a single prompt non-interactively and exit.
    Run {
        /// The prompt to send to the agent.
        prompt: String,
    },
    /// OAuth login / logout / status.
    #[command(subcommand)]
    Auth(AuthCmd),
    /// List or resume saved sessions.
    #[command(subcommand)]
    Sessions(SessionsCmd),
    /// Manage the configuration file.
    #[command(subcommand)]
    Config(ConfigCmd),
}

#[derive(Subcommand, Debug)]
pub(crate) enum AuthCmd {
    /// Log in to an OAuth provider (currently: anthropic).
    Login {
        /// Provider id (e.g. `anthropic`).
        provider: String,
    },
    /// Log out from an OAuth provider.
    Logout {
        /// Provider id (e.g. `anthropic`).
        provider: String,
    },
    /// Show login status across all known providers.
    Status,
}

#[derive(Subcommand, Debug)]
pub(crate) enum SessionsCmd {
    /// List saved sessions.
    Ls,
    /// Resume a saved session by id.
    Resume {
        /// The session id (or short prefix).
        id: String,
    },
}

#[derive(Subcommand, Debug)]
pub(crate) enum ConfigCmd {
    /// Create a default config file if one doesn't exist.
    Init,
}

pub(crate) fn parse_reasoning_level(s: &str) -> anyhow::Result<ReasoningLevel> {
    match s.to_lowercase().as_str() {
        "off" => Ok(ReasoningLevel::Off),
        "minimal" => Ok(ReasoningLevel::Minimal),
        "low" => Ok(ReasoningLevel::Low),
        "medium" => Ok(ReasoningLevel::Medium),
        "high" => Ok(ReasoningLevel::High),
        _ => anyhow::bail!(
            "Invalid reasoning level '{}'. Valid options: off, minimal, low, medium, high",
            s
        ),
    }
}

pub(crate) async fn get_model(provider: &str, model_id: &str) -> Model {
    if let Some(model) = tau_ai::models::get_model_by_id(model_id) {
        return model;
    }

    // Fallback: construct a default model for unknown/custom model IDs
    let provider_enum = Provider::from_id(provider);

    let mut model = Model {
        id: model_id.to_string(),
        name: model_id.to_string(),
        api: provider_enum.default_api(),
        provider: provider_enum,
        base_url: provider_enum.default_base_url().to_string(),
        reasoning: false,
        input_types: vec![InputType::Text],
        cost: CostInfo::default(),
        context_window: 128000,
        max_tokens: 8192,
        headers: Default::default(),
    };

    // Auto-detect capabilities for Ollama models
    if provider_enum == Provider::Ollama {
        let ollama = tau_ai::providers::ollama::OllamaProvider::new(model.base_url.clone());
        if let Ok(detail) = ollama.show_model(model_id).await {
            if detail.supports_vision() {
                model.input_types.push(InputType::Image);
            }
            model.reasoning = detail.has_capability("thinking");
            if let Some(ctx) = detail.context_length() {
                model.context_window = ctx;
                // Default max output to 1/3 of context, capped at 16K
                model.max_tokens = (ctx / 3).min(16384);
            }
        }
    }

    model
}

/// Get list of commonly available models
pub(crate) fn get_available_models() -> Vec<Model> {
    tau_ai::models::get_all_models()
}
