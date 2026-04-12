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
mod tools;
mod ui;
mod utils;

use std::sync::Arc;

use clap::Parser;
use tau_agent::{Agent, AgentConfig};
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

    let model = get_model(&provider, &model_id);

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
    let api_key = match cfg.get_api_key_with_oauth(&provider).await {
        Some(key) => key,
        None => {
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
    };

    let transport = Arc::new(tau_agent::transport::ProviderTransport::with_api_key(
        api_key,
    ));

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

    // Create agent with initial config (no system prompt yet)
    let config = AgentConfig {
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
    let mut agent = Agent::new(config, transport.clone());

    // Set up interaction channel for tools that need user input (e.g. ask_user)
    let (interaction_tx, interaction_rx) =
        tokio::sync::mpsc::channel::<tau_agent::InteractionRequest>(8);
    agent.set_interaction_sender(interaction_tx);

    agent.add_tool(Arc::new(tools::AskTool::new()));
    agent.add_tool(Arc::new(tools::BashTool::new()));
    agent.add_tool(Arc::new(tools::ReadTool::new()));
    agent.add_tool(Arc::new(tools::WriteTool::new()));
    agent.add_tool(Arc::new(tools::EditTool::new()));
    agent.add_tool(Arc::new(tools::GlobTool::new()));
    agent.add_tool(Arc::new(tools::GrepTool::new()));
    agent.add_tool(Arc::new(tools::ListTool::new()));
    agent.add_tool(Arc::new(tools::WebFetchTool::new()));

    let lsp_manager = Arc::new(lsp::LspManager::new(std::env::current_dir()?).await);
    if lsp_manager.is_available() {
        agent.add_tool(Arc::new(tools::LspTool::new(lsp_manager.clone())));
    }

    // Add agent tool (subagent spawning)
    let parent_tools: Vec<Arc<dyn tau_agent::tool::Tool>> = agent.tools().to_vec();
    let agent_handle = agent.handle();
    let manager = Arc::new(tau_agent::agent_manager::AgentManager::new(
        agent.event_sender(),
        parent_tools,
        agent.config().clone(),
        transport.clone(),
        20,
    ));

    // Create factory that makes AgentTools referencing this manager.
    // The handle parameter ensures background sub-subagents report to the
    // correct parent rather than always routing to the root agent.
    let mgr_for_factory = manager.clone();
    manager.set_agent_tool_factory(Arc::new(move |depth, handle| {
        let tool = tools::AgentTool::new(mgr_for_factory.clone(), depth).with_handle(handle);
        Arc::new(tool)
    }));

    let mgr_for_send = manager.clone();
    agent.add_tool(Arc::new(
        tools::AgentTool::new(manager, 0).with_handle(agent_handle),
    ));
    agent.add_tool(Arc::new(tools::SendMessageTool::new(mgr_for_send)));

    // Enable web search for Anthropic models
    if model.provider == tau_ai::Provider::Anthropic {
        agent.add_server_tool(tau_ai::ServerTool::WebSearch {
            name: "web_search".to_string(),
            max_uses: Some(8),
            allowed_domains: None,
            blocked_domains: None,
        });
    }

    // Build dynamic system prompt based on registered tools
    let tool_names = agent.tool_names();
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let acolyte_mode = cfg.acolyte_mode.unwrap_or(false);
    let prompt_opts = tau_agent::prompts::PromptOptions {
        tool_names: &tool_names,
        cwd: &cwd,
        acolyte_mode,
    };
    agent.set_system_prompt(tau_agent::prompts::build_system_prompt(&prompt_opts));

    if let Some(ref session_id) = args.resume {
        match session::SessionManager::load(session_id) {
            Ok((_session, messages, previous_summary)) => {
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
                agent.set_messages(messages);
                agent.set_previous_summary(previous_summary);
            }
            Err(e) => {
                eprintln!("Error loading session: {}", e);
                std::process::exit(1);
            }
        }
    }

    // Non-interactive mode
    let result = if let Some(command) = args.command {
        run_command::run_command(&mut agent, &command, &model, interaction_rx).await
    } else if use_tui {
        // TUI mode
        let mut model = model;
        let mut reasoning = reasoning;
        let available_models = get_available_models();
        ui::run_tui(
            &mut agent,
            &mut model,
            &mut reasoning,
            &available_models,
            interaction_rx,
        )
        .await
    } else {
        // Interactive mode (simple stdin/stdout)
        let session = session::SessionManager::new(&model.id).ok();
        let mut model = model;
        let mut reasoning = reasoning;
        interactive::run_interactive(
            &mut agent,
            &mut model,
            &mut reasoning,
            session,
            interaction_rx,
        )
        .await
    };

    lsp_manager.shutdown_all().await;
    result
}
