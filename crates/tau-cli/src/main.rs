//! tau - AI-powered coding agent CLI

mod auth;
mod cli;
mod commands;
mod config;
mod interactive;
mod lsp;
mod oauth;
mod run_command;
mod session;
mod ui;
mod utils;

use std::sync::Arc;

use clap::Parser;
use tau_agent::AgentConfig;
use tau_ai::ReasoningLevel;

use cli::{Args, get_available_models, get_model, parse_reasoning_level};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("tau=debug")
            .init();
    }

    if args.init_config {
        match config::Config::init() {
            Ok(path) => {
                println!("Config file created at: {}", path.display());
                println!("\nExample config:\n{}", config::example_config());
            }
            Err(e) => {
                eprintln!("Error creating config: {}", e);
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    if args.sessions {
        return session::list_sessions_cli();
    }

    if let Some(provider_id) = args.login {
        return auth::handle_oauth_login(&provider_id).await;
    }

    if let Some(provider_id) = args.logout {
        return auth::handle_oauth_logout(&provider_id);
    }

    if args.auth_status {
        return auth::show_auth_status();
    }

    let cfg = config::Config::load()?;

    if let Some(ref dir) = args.working_dir {
        std::env::set_current_dir(dir)?;
    }

    // Merge config with CLI args (CLI takes precedence)
    let provider = args
        .provider
        .or(cfg.provider.clone())
        .unwrap_or_else(|| "anthropic".to_string());

    let model_id = args
        .model
        .or(cfg.model.clone())
        .unwrap_or_else(|| "claude-sonnet-4-5-20250929".to_string());

    let model = get_model(&provider, &model_id).await;

    let reasoning = if args.reasoning {
        ReasoningLevel::Medium
    } else if let Some(ref level) = args.reasoning_level {
        parse_reasoning_level(level)?
    } else {
        cfg.reasoning_level
            .as_ref()
            .map(|s| parse_reasoning_level(s))
            .transpose()?
            .unwrap_or(ReasoningLevel::Off)
    };

    let use_tui = !args.no_tui && cfg.tui.unwrap_or(true);

    // Check for API key (OAuth, config, or env)
    // Providers like Ollama don't require an API key
    let api_key: Option<String> = match cfg.get_api_key_with_oauth(&provider).await {
        Some(key) => Some(key),
        None => {
            if model.provider.api_key_env_var().is_some() {
                let api_key_var = model
                    .provider
                    .api_key_env_var()
                    .unwrap_or("ANTHROPIC_API_KEY");
                eprintln!("Error: No authentication found for {}", provider);
                eprintln!();
                if provider == "anthropic" {
                    eprintln!("Options:");
                    eprintln!("  1. Login with Claude Pro/Max: tau --login anthropic");
                    eprintln!("  2. Set API key: export {}=your-key", api_key_var);
                    eprintln!("  3. Add to config: tau --init-config");
                } else {
                    eprintln!("Set your API key with: export {}=your-key", api_key_var);
                    eprintln!("Or add it to config file: tau --init-config");
                }
                std::process::exit(1);
            }
            None
        }
    };

    let transport = Arc::new(if let Some(key) = api_key {
        tau_agent::transport::ProviderTransport::with_api_key(key)
    } else {
        tau_agent::transport::ProviderTransport::new()
    });

    let compaction = if let Some(ref compaction_settings) = cfg.compaction {
        tau_agent::CompactionConfig {
            enabled: compaction_settings.enabled.unwrap_or(true),
            reserve_tokens: compaction_settings.reserve_tokens.unwrap_or(16384),
            keep_recent_tokens: compaction_settings.keep_recent_tokens.unwrap_or(20000),
        }
    } else {
        tau_agent::CompactionConfig::default()
    };

    let thinking_adaptive = cfg.thinking_adaptive.unwrap_or(false);
    let cache_scope = cfg.cache.as_ref().and_then(|c| c.scope.clone());
    let cache_ttl = cfg.cache.as_ref().and_then(|c| c.ttl.clone());
    let system_prompt_boundary = cfg.cache.as_ref().and_then(|c| c.prompt_boundary.clone());

    // Create agent builder
    let agent_config = AgentConfig {
        system_prompt: None,
        model: model.clone(),
        reasoning,
        thinking_adaptive,
        max_tokens: None,
        max_turns: Some(200),
        compaction,
        steering_mode: tau_agent::DequeueMode::All,
        follow_up_mode: tau_agent::DequeueMode::All,
        cache_scope,
        cache_ttl,
        system_prompt_boundary,
    };
    let mut builder = tau_agent::AgentBuilder::new(agent_config, transport.clone());

    // Set up interaction channel for tools that need user input
    let (interaction_tx, interaction_rx) =
        tokio::sync::mpsc::channel::<tau_agent::InteractionRequest>(8);
    builder.set_interaction_sender(interaction_tx.clone());

    builder.add_tool(Arc::new(tau_tools::AskTool::new()));
    builder.add_tool(Arc::new(tau_tools::BashTool::new()));
    builder.add_tool(Arc::new(tau_tools::ReadTool::new()));
    builder.add_tool(Arc::new(tau_tools::WriteTool::new()));
    builder.add_tool(Arc::new(tau_tools::EditTool::new()));
    builder.add_tool(Arc::new(tau_tools::GlobTool::new()));
    builder.add_tool(Arc::new(tau_tools::GrepTool::new()));
    builder.add_tool(Arc::new(tau_tools::ListTool::new()));
    builder.add_tool(Arc::new(tau_tools::WebFetchTool::new()));
    builder.add_tool(Arc::new(tau_tools::SubmitPlanTool::new()));
    builder.add_tool(Arc::new(tau_tools::StepStartedTool::new()));
    builder.add_tool(Arc::new(tau_tools::StepCompletedTool::new()));
    builder.add_tool(Arc::new(tau_tools::PlanCompleteTool::new()));

    let lsp_manager = Arc::new(lsp::LspManager::new(std::env::current_dir()?).await);
    if lsp_manager.is_available() {
        builder.add_tool(Arc::new(lsp::LspTool::new(lsp_manager.clone())));
    }

    // Add agent tool (subagent spawning)
    let parent_tools: Vec<Arc<dyn tau_agent::tool::Tool>> = builder.tools().to_vec();
    let agent_handle = builder.pre_handle();
    let manager = Arc::new(
        tau_agent::manager::AgentManager::new(
            builder.event_sender(),
            parent_tools,
            builder.config().clone(),
            transport.clone(),
            20,
        )
        .with_parent_interaction_sender(interaction_tx),
    );

    // Create factory that makes AgentTools referencing this manager.
    // Plan parents get a tool restricted to spawning read-only subagents
    // so the read-only invariant the Plan prompt advertises is actually enforced.
    let mgr_for_factory = manager.clone();
    manager.set_agent_tool_factory(Arc::new(move |depth, handle, parent_type| {
        let mut tool = tau_tools::AgentTool::new(mgr_for_factory.clone(), depth).with_handle(handle);
        if matches!(parent_type, tau_agent::manager::AgentType::Plan) {
            tool = tool.with_allowed_types(vec![
                tau_agent::manager::AgentType::Explore,
                tau_agent::manager::AgentType::Plan,
            ]);
        }
        Arc::new(tool)
    }));

    let mgr_for_send = manager.clone();
    let mgr_for_commands = manager.clone();
    builder.add_tool(Arc::new(
        tau_tools::AgentTool::new(manager, 0).with_handle(agent_handle),
    ));
    builder.add_tool(Arc::new(tau_tools::SendMessageTool::new(mgr_for_send)));

    // Enable web search for Anthropic models
    if model.provider == tau_ai::Provider::Anthropic {
        builder.add_server_tool(tau_ai::ServerTool::WebSearch {
            name: "web_search".to_string(),
            max_uses: Some(8),
            allowed_domains: None,
            blocked_domains: None,
        });
    }

    // Build dynamic system prompt based on registered tools
    let tool_names = builder.tool_names();
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let acolyte_mode = cfg.acolyte_mode.unwrap_or(false);
    let prompt_opts = tau_agent::prompts::PromptOptions {
        tool_names: &tool_names,
        cwd: &cwd,
        acolyte_mode,
    };
    builder.set_system_prompt(tau_agent::prompts::build_system_prompt(&prompt_opts));

    let mut resumed_session: Option<session::SessionManager> = None;
    if let Some(ref session_id) = args.resume {
        match session::SessionManager::load(session_id) {
            Ok((session_mgr, messages, previous_summary)) => {
                println!(
                    "Resuming session {} ({} messages{})",
                    session_id,
                    messages.len(),
                    if previous_summary.is_some() {
                        ", with compacted context"
                    } else {
                        ""
                    }
                );
                builder.set_messages(messages);
                builder.set_previous_summary(previous_summary);
                resumed_session = Some(session_mgr);
            }
            Err(e) => {
                eprintln!("Error loading session: {}", e);
                std::process::exit(1);
            }
        }
    }

    // Spawn the agent actor — from here on we use the handle
    let handle = builder.spawn();

    // Non-interactive mode
    let result = if let Some(command) = args.command {
        run_command::run_command(&handle, &command, interaction_rx).await
    } else if use_tui {
        // TODO(approval-ui): until the TUI renders ConfirmTool prompts,
        // auto-accept all elevated calls so bash etc. don't get rejected.
        handle.set_approval_policy(Arc::new(tau_agent::AutoAcceptAllPolicy));
        // TUI mode
        let available_models = get_available_models();
        ui::run_tui(&handle, &available_models, interaction_rx, mgr_for_commands.clone()).await
    } else {
        // Interactive mode (simple stdin/stdout)
        let session = resumed_session.or_else(|| session::SessionManager::new(&model.id).ok());
        interactive::run_interactive(&handle, session, interaction_rx, mgr_for_commands).await
    };

    lsp_manager.shutdown_all().await;
    result
}
