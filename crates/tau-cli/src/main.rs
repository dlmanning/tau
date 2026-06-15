//! tau - AI-powered coding agent CLI

mod auth;
mod cli;
mod commands;
mod config;
mod context;
mod driver;
mod frontends;
mod oauth;
mod prompts;
mod session;
mod subagents;
mod ui;
mod utils;

use std::sync::Arc;

use clap::Parser;
use tau_ai::ReasoningLevel;

use cli::{
    Args, AuthCmd, Command, ConfigCmd, McpCmd, ModelsCmd, SessionsCmd, get_available_models,
    get_model, parse_reasoning_level,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if args.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("tau=debug")
            .init();
    }

    // Subcommands that don't need an agent.
    let mut resume_id: Option<String> = None;
    let mut run_prompt: Option<String> = None;
    let mut mcp_cmd: Option<McpCmd> = None;
    match args.command {
        Some(Command::Config(ConfigCmd::Init)) => {
            return match config::Config::init() {
                Ok(path) => {
                    println!("Config file created at: {}", path.display());
                    println!("\nExample config:\n{}", config::example_config());
                    Ok(())
                }
                Err(e) => {
                    eprintln!("Error creating config: {}", e);
                    std::process::exit(1);
                }
            };
        }
        Some(Command::Auth(AuthCmd::Login { provider })) => {
            return auth::handle_oauth_login(&provider).await;
        }
        Some(Command::Auth(AuthCmd::Logout { provider })) => {
            return auth::handle_oauth_logout(&provider);
        }
        Some(Command::Auth(AuthCmd::Status)) => {
            return auth::show_auth_status();
        }
        Some(Command::Sessions(SessionsCmd::Ls)) => {
            return session::list_sessions_cli();
        }
        Some(Command::Sessions(SessionsCmd::Resume { id })) => {
            // Resolve prefix → full id now, before the auth gate, so a
            // bad session id fails with a session error.
            resume_id = Some(session::SessionManager::resolve_id(&id)?);
        }
        Some(Command::Run { prompt }) => {
            run_prompt = Some(prompt);
        }
        Some(Command::Models(ModelsCmd::List)) => {
            cli::print_models_list();
            return Ok(());
        }
        Some(Command::Mcp(cmd)) => {
            // Needs the config, which loads below.
            mcp_cmd = Some(cmd);
        }
        None => {}
    }

    let cfg = config::Config::load()?;

    if let Some(McpCmd::List) = mcp_cmd {
        return commands::mcp::list(&cfg).await;
    }

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

    let model = get_model(&provider, &model_id).await?;

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
                    eprintln!("  1. Login with Claude Pro/Max: tau auth login anthropic");
                    eprintln!("  2. Set API key: export {}=your-key", api_key_var);
                    eprintln!("  3. Add to config: tau config init");
                } else {
                    eprintln!("Set your API key with: export {}=your-key", api_key_var);
                    eprintln!("Or add it to config file: tau config init");
                }
                std::process::exit(1);
            }
            None
        }
    };

    let transport = Arc::new(if let Some(key) = api_key {
        tau_agent::ProviderTransport::with_api_key(key)
    } else {
        tau_agent::ProviderTransport::new()
    });

    let agent_config = cfg.to_agent_config(model.clone(), reasoning);
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
    builder.add_tool(Arc::new(tau_tools::SubagentReportTool::new()));

    // LSP discovery and MCP server connections are independent
    // startup I/O — run them concurrently.
    let (lsp_manager, mcp_manager) = tokio::join!(
        tau_tools::lsp::LspManager::new(std::env::current_dir()?),
        tau_tools::mcp::McpManager::connect_all(cfg.mcp_specs()),
    );
    let lsp_manager = Arc::new(lsp_manager);
    if lsp_manager.is_available() {
        builder.add_tool(Arc::new(tau_tools::lsp::LspTool::new(lsp_manager.clone())));
    }
    let mcp_manager = Arc::new(mcp_manager);
    for (name, err) in mcp_manager.failures() {
        eprintln!("warning: MCP server '{name}' unavailable: {err}");
    }
    // Added before `build_resolver` so AllExceptAgent subagent specs
    // inherit MCP tools; curated whitelists stay MCP-free.
    for tool in mcp_manager.tools().await {
        builder.add_tool(tool);
    }

    let manager = Arc::new(
        tau_agent::AgentManager::new(builder.config().clone(), transport.clone(), 20)
            .with_parent_interaction_sender(interaction_tx),
    );

    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let resolver = subagents::build_resolver(manager.clone(), builder.tools(), &cwd);
    let resolver_for_host = resolver.clone();

    let agent_tool = tau_tools::AgentTool::new(manager.clone())
        .with_spec_resolver(resolver)
        .with_worktree_specs(subagents::worktree_specs());
    builder.add_tool(Arc::new(agent_tool));
    builder.add_tool(Arc::new(tau_tools::SendMessageTool::new(manager.clone())));

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
    let acolyte_mode = cfg.acolyte_mode.unwrap_or(false);
    let parent_system_prompt = {
        let tool_names = builder.tool_names();
        let prompt_opts = crate::prompts::PromptOptions {
            tool_names: &tool_names,
            cwd: &cwd,
            acolyte_mode,
        };
        crate::prompts::build_system_prompt(&prompt_opts)
    };
    builder.set_system_prompt(parent_system_prompt.clone());

    let mut resumed_session: Option<session::SessionManager> = None;
    if let Some(ref session_id) = resume_id {
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
                builder.seed(tau_agent::AgentSeed::Messages {
                    messages,
                    previous_summary,
                });
                resumed_session = Some(session_mgr);
            }
            Err(e) => {
                eprintln!("Error loading session: {}", e);
                std::process::exit(1);
            }
        }
    }

    // Snapshot pieces needed to build the "root" spec before consuming the builder.
    let root_tools: Vec<tau_agent::BoxedTool> = builder.tools().to_vec();
    let root_spec = tau_agent::AgentSpec {
        system_prompt: parent_system_prompt,
        tools: root_tools,
        max_turns: 200,
    };

    // Spawn the agent actor — from here on we use the handle
    let handle = builder.spawn().await?;
    // Register the root with the manager so handle.respec works.
    let _root_id = manager.adopt(&handle, "root", root_spec);

    // Dispatch to the appropriate frontend. Session owns the loop;
    // the frontend handles I/O.
    let available_models = get_available_models();
    let is_one_shot = run_prompt.is_some();
    let persistence = if is_one_shot {
        None
    } else {
        resumed_session.or_else(|| session::SessionManager::new(&model.id).ok())
    };
    let mut sess = driver::Session::new(driver::SessionConfig {
        handle: handle.clone(),
        manager: manager.clone(),
        spec_resolver: resolver_for_host,
        interaction_rx,
        available_models: available_models.clone(),
        persistence,
    });
    let result = if use_tui && !is_one_shot {
        let agent_config = handle
            .config()
            .await
            .ok_or_else(|| anyhow::anyhow!("Agent shut down"))?;
        let theme = ui::Theme::detect(cfg.theme.as_deref());
        let mut frontend = ui::TuiFrontend::new(&agent_config, available_models, theme).await?;
        sess.drive(&mut frontend).await
    } else {
        let mut frontend = match run_prompt {
            Some(prompt) => frontends::stdout::StdoutFrontend::one_shot(prompt, args.quiet),
            None => frontends::stdout::StdoutFrontend::repl(),
        };
        sess.drive(&mut frontend).await
    };

    mcp_manager.shutdown_all().await;
    lsp_manager.shutdown_all().await;
    result?;
    // Scripting contract: `tau run` exits non-zero when the prompt
    // failed, so pipelines can detect failure.
    if is_one_shot && sess.had_agent_error() {
        std::process::exit(1);
    }
    Ok(())
}

