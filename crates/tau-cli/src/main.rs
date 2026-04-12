//! tau - AI-powered coding agent CLI

mod commands;
mod config;
mod lsp;
mod oauth;
mod session;
mod tools;
mod ui;
mod utils;

use std::sync::Arc;

use clap::Parser;
use tau_agent::{Agent, AgentConfig, AgentEvent};
use tau_ai::{CostInfo, InputType, Model, Provider, ReasoningLevel};

/// tau - AI-powered coding agent
#[derive(Parser, Debug)]
#[command(name = "tau")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model to use (default: claude-sonnet-4-5-20250929)
    #[arg(short, long)]
    model: Option<String>,

    /// Provider (anthropic, openai, google)
    #[arg(short, long)]
    provider: Option<String>,

    /// Enable reasoning/thinking mode
    #[arg(short, long)]
    reasoning: bool,

    /// Reasoning level (off, minimal, low, medium, high)
    #[arg(long)]
    reasoning_level: Option<String>,

    /// Run in non-interactive mode with a single prompt
    #[arg(short = 'c', long)]
    command: Option<String>,

    /// Working directory
    #[arg(short, long)]
    working_dir: Option<String>,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Disable TUI mode (use simple stdin/stdout)
    #[arg(long)]
    no_tui: bool,

    /// Resume a previous session by ID
    #[arg(long)]
    resume: Option<String>,

    /// List saved sessions
    #[arg(long)]
    sessions: bool,

    /// Initialize config file
    #[arg(long)]
    init_config: bool,

    /// Login to an OAuth provider (anthropic)
    #[arg(long)]
    login: Option<String>,

    /// Logout from an OAuth provider (anthropic)
    #[arg(long)]
    logout: Option<String>,

    /// List OAuth login status
    #[arg(long)]
    auth_status: bool,
}

fn parse_reasoning_level(s: &str) -> ReasoningLevel {
    match s.to_lowercase().as_str() {
        "minimal" => ReasoningLevel::Minimal,
        "low" => ReasoningLevel::Low,
        "medium" => ReasoningLevel::Medium,
        "high" => ReasoningLevel::High,
        _ => ReasoningLevel::Off,
    }
}

fn get_model(provider: &str, model_id: &str) -> Model {
    if let Some(model) = tau_ai::models::get_model_by_id(model_id) {
        return model;
    }

    // Fallback: construct a default model for unknown/custom model IDs
    let provider_enum = Provider::from_id(provider);

    Model {
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
    }
}

/// Get list of commonly available models
fn get_available_models() -> Vec<Model> {
    tau_ai::models::get_all_models()
}

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
        return list_sessions();
    }

    if let Some(provider_id) = args.login {
        return handle_oauth_login(&provider_id).await;
    }

    if let Some(provider_id) = args.logout {
        return handle_oauth_logout(&provider_id);
    }

    if args.auth_status {
        return show_auth_status();
    }

    let cfg = config::Config::load();

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
        parse_reasoning_level(level)
    } else {
        cfg.reasoning_level
            .as_ref()
            .map(|s| parse_reasoning_level(s))
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
        agent.add_tool(Arc::new(tools::LspTool::new(lsp_manager)));
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
        let tool = tools::AgentTool::new(mgr_for_factory.clone(), depth)
            .with_handle(handle);
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
    if let Some(command) = args.command {
        return run_command(&mut agent, &command, &model, interaction_rx).await;
    }

    // TUI mode
    if use_tui {
        let mut model = model;
        let mut reasoning = reasoning;
        let available_models = get_available_models();
        return ui::run_tui(
            &mut agent,
            &mut model,
            &mut reasoning,
            &available_models,
            interaction_rx,
        )
        .await;
    }

    // Interactive mode (simple stdin/stdout)
    // Create a new session for auto-save
    let session = session::SessionManager::new(&model.id).ok();
    let mut model = model;
    let mut reasoning = reasoning;
    run_interactive(&mut agent, &mut model, &mut reasoning, session, interaction_rx).await
}

async fn run_command(
    agent: &mut Agent,
    command: &str,
    model: &Model,
    interaction_rx: tokio::sync::mpsc::Receiver<tau_agent::InteractionRequest>,
) -> anyhow::Result<()> {
    println!("tau> {}", command);
    println!();

    let mut receiver = agent.subscribe();
    let model_for_cost = model.clone();

    let handle = tokio::spawn(async move {
        while let Ok(event) = receiver.recv().await {
            match event {
                AgentEvent::MessageUpdate { message } => {
                    let text = message.text();
                    if !text.is_empty() {
                        print!("\r{}", text);
                    }
                }
                AgentEvent::MessageEnd { message } => {
                    println!("\r{}", message.text());
                }
                AgentEvent::ToolExecutionStart { tool_name, .. } => {
                    println!("\n[Running {}...]", tool_name);
                }
                AgentEvent::ToolExecutionUpdate {
                    tool_name, content, ..
                } => {
                    println!("[{}: {}]", tool_name, content);
                }
                AgentEvent::ToolExecutionEnd {
                    tool_name,
                    result,
                    is_error,
                    ..
                } => {
                    if is_error {
                        println!("[{} failed: {}]", tool_name, result);
                    } else {
                        let preview = crate::utils::truncate_chars(&result, 200);
                        println!("[{}: {}]", tool_name, preview);
                    }
                }
                AgentEvent::CompactionStart { reason } => {
                    println!(
                        "[Compacting context ({})]",
                        crate::utils::compaction_reason_str(reason)
                    );
                }
                AgentEvent::CompactionEnd {
                    tokens_before,
                    tokens_after,
                } => {
                    println!(
                        "[Compacted: ~{} -> ~{} tokens]",
                        tokens_before, tokens_after
                    );
                }
                AgentEvent::Error { message } => {
                    eprintln!("Error: {}", message);
                }
                AgentEvent::AgentEnd { total_usage, .. } => {
                    let cost = total_usage.calculate_cost(&model_for_cost);
                    println!(
                        "\n[Tokens: {} in, {} out | Cost: ${:.4}]",
                        total_usage.input, total_usage.output, cost.total
                    );
                    break;
                }
                _ => {}
            }
        }
    });

    let interaction_handle = tokio::spawn(handle_interaction_stdin(interaction_rx));

    agent.prompt(command).await?;

    // Wait for event handler to finish (it breaks on AgentEnd)
    match tokio::time::timeout(std::time::Duration::from_secs(2), handle).await {
        Ok(_) => {}
        Err(_) => tracing::debug!("Event handler did not finish in time"),
    }
    interaction_handle.abort();

    Ok(())
}

/// Handle interaction requests by printing to stdout and reading from stdin.
/// Used in non-TUI modes (command mode and simple interactive mode).
async fn handle_interaction_stdin(
    mut rx: tokio::sync::mpsc::Receiver<tau_agent::InteractionRequest>,
) {
    use tau_agent::interaction::{InteractionKind, InteractionResponse};

    while let Some(request) = rx.recv().await {
        match request.kind {
            InteractionKind::AskQuestion { question, options } => {
                println!("\n{}", question);
                for (i, opt) in options.iter().enumerate() {
                    println!("  {}) {} — {}", i + 1, opt.label, opt.description);
                }
                print!("Enter choice (1-{}): ", options.len());
                std::io::Write::flush(&mut std::io::stdout()).ok();

                let num_options = options.len();
                let line = tokio::task::spawn_blocking(|| {
                    let mut input = String::new();
                    std::io::stdin().read_line(&mut input).ok();
                    input
                })
                .await
                .unwrap_or_default();

                let response = match line.trim().parse::<usize>() {
                    Ok(n) if n >= 1 && n <= num_options => {
                        InteractionResponse::Answer(options[n - 1].label.clone())
                    }
                    _ => InteractionResponse::Cancelled,
                };

                let _ = request.response_tx.send(response);
            }
        }
    }
}

async fn run_interactive(
    agent: &mut Agent,
    model: &mut Model,
    reasoning: &mut ReasoningLevel,
    mut session: Option<session::SessionManager>,
    interaction_rx: tokio::sync::mpsc::Receiver<tau_agent::InteractionRequest>,
) -> anyhow::Result<()> {
    use std::io::{self, Write};

    let available_models = get_available_models();
    let _interaction_handle = tokio::spawn(handle_interaction_stdin(interaction_rx));

    if std::io::IsTerminal::is_terminal(&std::io::stderr()) {
        let model_short = model.id.split('/').next_back().unwrap_or(&model.id);
        if let Some(ref s) = session {
            eprintln!("tau ({}) session: {}", model_short, &s.id()[..8]);
        } else {
            eprintln!("tau ({})", model_short);
        }
        eprintln!();
    }

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        if input.starts_with('/') {
            if let Some(result) =
                commands::execute_command(input, agent, model, *reasoning, &available_models)
            {
                match result {
                    commands::CommandResult::Clear => {
                        agent.clear_messages();
                        println!("Cleared conversation.");
                    }
                    commands::CommandResult::Exit => {
                        break;
                    }
                    commands::CommandResult::Message(msg) => {
                        println!("{}", msg);
                    }
                    commands::CommandResult::ChangeModel(new_model) => {
                        println!(
                            "Switched to: {} ({})",
                            new_model.id,
                            new_model.provider.name()
                        );
                        *model = new_model.clone();
                        agent.set_model(new_model);
                    }
                    commands::CommandResult::ChangeReasoning(level) => {
                        println!("Reasoning level set to: {:?}", level);
                        *reasoning = level;
                        agent.set_reasoning(level);
                    }
                    commands::CommandResult::Unknown(cmd) => {
                        println!("Unknown command: /{}", cmd);
                        println!("Type /help for available commands.");
                    }
                    commands::CommandResult::OpenModelSelector => {
                        println!(
                            "{}",
                            commands::ModelCommand::list_models_text(model, &available_models)
                        );
                    }
                    commands::CommandResult::OpenBranchSelector => {
                        let messages = agent.messages();
                        if messages.is_empty() {
                            println!("No messages to branch from.");
                        } else {
                            println!("Messages in conversation:");
                            for (i, msg) in messages.iter().enumerate() {
                                let role = match msg {
                                    tau_ai::Message::User { .. } => "user",
                                    tau_ai::Message::Assistant { .. } => "assistant",
                                    tau_ai::Message::ToolResult { .. } => "tool",
                                    tau_ai::Message::SystemInjection { .. } => "system",
                                };
                                let text = msg.text();
                                let preview: String = text.chars().take(60).collect();
                                let preview = preview.replace('\n', " ");
                                println!("  {}: [{}] {}", i, role, preview);
                            }
                            println!("\nUse /branch <index> to create a branch from that message.");
                        }
                    }
                    commands::CommandResult::Compact => {
                        println!("Compacting context...");
                        match agent
                            .run_compaction(tau_agent::CompactionReason::Manual)
                            .await
                        {
                            Ok(()) => {
                                println!(
                                    "Context compacted. {} messages remaining.",
                                    agent.messages().len()
                                );
                            }
                            Err(e) => {
                                println!("Compaction failed: {}", e);
                            }
                        }
                    }
                    commands::CommandResult::BranchFrom(branch_index) => {
                        match session::SessionManager::branch_from(
                            agent.messages(),
                            branch_index,
                            &model.id,
                        ) {
                            Ok(new_session) => {
                                let msg_count = branch_index.map(|i| i + 1).unwrap_or(0);
                                println!(
                                    "Created branch: {} ({} messages)",
                                    new_session.id(),
                                    msg_count
                                );
                                if let Some(idx) = branch_index {
                                    let messages: Vec<_> =
                                        agent.messages().iter().take(idx + 1).cloned().collect();
                                    agent.set_messages(messages);
                                } else {
                                    agent.clear_messages();
                                }
                                session = Some(new_session);
                                println!("Continue from this point with a fresh context.");
                            }
                            Err(e) => {
                                println!("Failed to create branch: {}", e);
                            }
                        }
                    }
                }
                println!();
                continue;
            }
        }

        println!();

        let mut receiver = agent.subscribe();
        let model_for_cost = model.clone();

        let is_tty = std::io::IsTerminal::is_terminal(&io::stdout());
        let handle = tokio::spawn(async move {
            let mut last_text_len = 0;
            while let Ok(event) = receiver.recv().await {
                match event {
                    AgentEvent::MessageUpdate { message } => {
                        let text = message.text();
                        // Use chars().count() for proper Unicode handling
                        let text_chars: Vec<char> = text.chars().collect();
                        if text_chars.len() > last_text_len {
                            let new_text: String = text_chars[last_text_len..].iter().collect();
                            print!("{}", new_text);
                            io::stdout().flush().ok();
                            last_text_len = text_chars.len();
                        }
                    }
                    AgentEvent::MessageEnd { .. } => {
                        println!();
                        last_text_len = 0;
                    }
                    AgentEvent::ToolExecutionStart { tool_name, .. } => {
                        print!("\n[{}...", tool_name);
                        io::stdout().flush().ok();
                    }
                    AgentEvent::ToolExecutionUpdate { content, .. } => {
                        print!(" {}", content);
                        io::stdout().flush().ok();
                    }
                    AgentEvent::ToolExecutionEnd {
                        tool_name: _,
                        result,
                        is_error,
                        ..
                    } => {
                        if is_error {
                            println!(" error]");
                            let preview = crate::utils::truncate_chars(&result, 80);
                            println!("  {}", preview.replace('\n', " "));
                        } else {
                            // Compact success output
                            let char_count = result.chars().count();
                            let first_line = result.lines().next().unwrap_or("");

                            if char_count <= 60 && !result.contains('\n') {
                                // Short single-line result: show inline
                                println!(" {}]", result);
                            } else if first_line.chars().count() <= 50 {
                                // Multi-line but short first line: show preview
                                println!(" {}...]", first_line);
                            } else {
                                // Long content: just close the bracket
                                let preview: String = first_line.chars().take(40).collect();
                                println!(" {}...]", preview);
                            }
                        }
                    }
                    AgentEvent::AgentEnd { total_usage, .. } => {
                        let cost = total_usage.calculate_cost(&model_for_cost);
                        // Print stats to stderr so they don't interfere with piped output
                        if is_tty {
                            println!(
                                "[{} in, {} out | ${:.4}]",
                                total_usage.input, total_usage.output, cost.total
                            );
                        }
                        break;
                    }
                    AgentEvent::CompactionStart { reason } => {
                        println!(
                            "[Compacting context ({})]",
                            crate::utils::compaction_reason_str(reason)
                        );
                    }
                    AgentEvent::CompactionEnd {
                        tokens_before,
                        tokens_after,
                    } => {
                        println!(
                            "[Compacted: ~{} -> ~{} tokens]",
                            tokens_before, tokens_after
                        );
                    }
                    AgentEvent::Error { message } => {
                        eprintln!("\nError: {}", message);
                    }
                    _ => {}
                }
            }
        });

        // Track message count before prompt to save new messages after
        let msgs_before = agent.messages().len();

        if let Err(e) = agent.prompt(input).await {
            eprintln!("Error: {}", e);
        }

        if let Some(ref mut s) = session {
            let all_msgs = agent.messages();
            for msg in all_msgs.iter().skip(msgs_before) {
                let _ = s.append_message(msg);
            }
            let _ = s.append_usage(&agent.state().total_usage);
        }

        match tokio::time::timeout(std::time::Duration::from_secs(2), handle).await {
            Ok(_) => {}
            Err(_) => tracing::debug!("Event handler did not finish in time"),
        }

        println!();
    }

    Ok(())
}

fn list_sessions() -> anyhow::Result<()> {
    match session::SessionManager::list_sessions() {
        Ok(sessions) => {
            if sessions.is_empty() {
                println!("No saved sessions found.");
                println!(
                    "Sessions are stored in: {}",
                    session::SessionManager::sessions_dir().display()
                );
            } else {
                println!("Saved sessions:\n");
                println!("{:<38} {:<20} {:<8} Working Dir", "ID", "Created", "Msgs");
                println!("{}", "-".repeat(90));
                for s in sessions {
                    println!(
                        "{:<38} {:<20} {:<8} {}",
                        s.id,
                        s.created_at_display(),
                        s.message_count,
                        s.working_dir
                    );
                }
                println!("\nResume with: tau --resume <session-id>");
            }
        }
        Err(e) => {
            eprintln!("Error listing sessions: {}", e);
        }
    }
    Ok(())
}


async fn handle_oauth_login(provider_id: &str) -> anyhow::Result<()> {
    let provider = match oauth::OAuthProvider::from_id(provider_id) {
        Some(p) => p,
        None => {
            eprintln!("Unknown OAuth provider: {}", provider_id);
            eprintln!("Available providers: anthropic");
            std::process::exit(1);
        }
    };

    println!("Logging in to {}...", provider.name());
    println!();

    match oauth::login(
        provider,
        |url| {
            println!("Opening browser to authorize...");
            println!();
            println!("If the browser doesn't open, visit this URL:");
            println!("  {}", url);
            println!();

            #[cfg(target_os = "macos")]
            let _ = std::process::Command::new("open").arg(&url).spawn();
            #[cfg(target_os = "linux")]
            let _ = std::process::Command::new("xdg-open").arg(&url).spawn();
            #[cfg(target_os = "windows")]
            let _ = std::process::Command::new("cmd")
                .args(["/C", "start", &url])
                .spawn();
        },
        || async {
            println!("After authorizing, paste the code below (format: code#state):");
            print!("> ");
            use std::io::Write;
            std::io::stdout().flush().ok();

            let mut input = String::new();
            std::io::stdin().read_line(&mut input).ok();
            input.trim().to_string()
        },
    )
    .await
    {
        Ok(()) => {
            println!();
            println!("Successfully logged in to {}!", provider.name());
            println!("Credentials saved to ~/.config/tau/oauth.json");
        }
        Err(e) => {
            eprintln!();
            eprintln!("Login failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

fn handle_oauth_logout(provider_id: &str) -> anyhow::Result<()> {
    let provider = match oauth::OAuthProvider::from_id(provider_id) {
        Some(p) => p,
        None => {
            eprintln!("Unknown OAuth provider: {}", provider_id);
            eprintln!("Available providers: anthropic");
            std::process::exit(1);
        }
    };

    match oauth::logout(provider) {
        Ok(()) => {
            println!("Successfully logged out of {}", provider.name());
        }
        Err(e) => {
            eprintln!("Logout failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

fn show_auth_status() -> anyhow::Result<()> {
    println!("OAuth Authentication Status");
    println!("{}", "-".repeat(40));

    for provider in oauth::OAuthProvider::available() {
        let status = if let Some(creds) = oauth::load_oauth_credentials(provider.id()) {
            let expires = chrono::DateTime::from_timestamp_millis(creds.expires)
                .map(|dt| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
                .unwrap_or_else(|| "unknown".to_string());

            if chrono::Utc::now().timestamp_millis() >= creds.expires {
                "Logged in (token expired, will refresh on next use)".to_string()
            } else {
                format!("Logged in (expires: {})", expires)
            }
        } else {
            "Not logged in".to_string()
        };

        println!("{:<25} {}", provider.name(), status);
    }

    println!();
    println!("Login with: tau --login <provider>");
    println!("Logout with: tau --logout <provider>");

    Ok(())
}
