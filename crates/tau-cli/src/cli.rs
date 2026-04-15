//! CLI argument parsing and model resolution

use clap::Parser;
use tau_ai::{CostInfo, InputType, Model, Provider, ReasoningLevel};

/// tau - AI-powered coding agent
#[derive(Parser, Debug)]
#[command(name = "tau")]
#[command(author, version, about, long_about = None)]
pub(crate) struct Args {
    /// Model to use (default: claude-sonnet-4-5-20250929)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Provider (anthropic, openai, google)
    #[arg(short, long)]
    pub provider: Option<String>,

    /// Enable reasoning/thinking mode
    #[arg(short, long)]
    pub reasoning: bool,

    /// Reasoning level (off, minimal, low, medium, high)
    #[arg(long)]
    pub reasoning_level: Option<String>,

    /// Run in non-interactive mode with a single prompt
    #[arg(short = 'c', long)]
    pub command: Option<String>,

    /// Working directory
    #[arg(short, long)]
    pub working_dir: Option<String>,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Disable TUI mode (use simple stdin/stdout)
    #[arg(long)]
    pub no_tui: bool,

    /// Resume a previous session by ID
    #[arg(long)]
    pub resume: Option<String>,

    /// List saved sessions
    #[arg(long)]
    pub sessions: bool,

    /// Initialize config file
    #[arg(long)]
    pub init_config: bool,

    /// Login to an OAuth provider (anthropic)
    #[arg(long)]
    pub login: Option<String>,

    /// Logout from an OAuth provider (anthropic)
    #[arg(long)]
    pub logout: Option<String>,

    /// List OAuth login status
    #[arg(long)]
    pub auth_status: bool,
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
