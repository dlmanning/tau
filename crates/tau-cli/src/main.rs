//! tau - AI-powered coding agent CLI

mod commands;
mod config;
mod context;
mod oauth;
mod session;
mod tools;
mod ui;
mod utils;

use clap::Parser;
use std::sync::Arc;
use tau_agent::{Agent, AgentConfig, AgentEvent};
use tau_ai::{Api, CostInfo, InputType, Model, Provider, ReasoningLevel};

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

fn parse_provider(s: &str) -> Provider {
    match s.to_lowercase().as_str() {
        "anthropic" => Provider::Anthropic,
        "openai" => Provider::OpenAI,
        "google" => Provider::Google,
        "groq" => Provider::Groq,
        "cerebras" => Provider::Cerebras,
        "xai" => Provider::XAI,
        "openrouter" => Provider::OpenRouter,
        "ollama" => Provider::Ollama,
        _ => Provider::Custom,
    }
}

fn get_model(provider: &str, model_id: &str) -> Model {
    // Try registry lookup first
    if let Some(model) = tau_ai::models::get_model_by_id(model_id) {
        return model;
    }

    // Fallback: construct a default model for unknown/custom model IDs
    let provider_enum = parse_provider(provider);

    let (api, base_url) = match provider {
        "anthropic" => (
            Api::AnthropicMessages,
            "https://api.anthropic.com".to_string(),
        ),
        "openai" => (
            Api::OpenAIResponses,
            "https://api.openai.com/v1".to_string(),
        ),
        "google" => (
            Api::GoogleGenerativeAI,
            "https://generativelanguage.googleapis.com/v1beta".to_string(),
        ),
        "groq" => (
            Api::OpenAICompletions,
            "https://api.groq.com/openai/v1".to_string(),
        ),
        "cerebras" => (
            Api::OpenAICompletions,
            "https://api.cerebras.ai/v1".to_string(),
        ),
        "xai" => (
            Api::OpenAICompletions,
            "https://api.x.ai/v1".to_string(),
        ),
        "openrouter" => (
            Api::OpenAICompletions,
            "https://openrouter.ai/api/v1".to_string(),
        ),
        "ollama" => (
            Api::OpenAICompletions,
            "http://localhost:11434/v1".to_string(),
        ),
        _ => (
            Api::OpenAICompletions,
            String::new(),
        ),
    };

    Model {
        id: model_id.to_string(),
        name: model_id.to_string(),
        api,
        provider: provider_enum,
        base_url,
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

    // Setup tracing
    if args.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("tau=debug")
            .init();
    }

    // Initialize config and exit
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

    // List sessions and exit
    if args.sessions {
        return list_sessions();
    }

    // Handle OAuth login
    if let Some(provider_id) = args.login {
        return handle_oauth_login(&provider_id).await;
    }

    // Handle OAuth logout
    if let Some(provider_id) = args.logout {
        return handle_oauth_logout(&provider_id);
    }

    // Show auth status
    if args.auth_status {
        return show_auth_status();
    }

    // Load config file
    let cfg = config::Config::load();

    // Change working directory if specified
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
    if cfg.get_api_key_with_oauth(&provider).await.is_none() {
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

    // Get API key (OAuth token or regular API key)
    let api_key = cfg
        .get_api_key_with_oauth(&provider)
        .await
        .expect("API key check passed but key not found");

    let transport = Arc::new(tau_agent::transport::ProviderTransport::with_api_key(
        api_key,
    ));

    // Build compaction config from settings
    let compaction = if let Some(ref compaction_settings) = cfg.compaction {
        tau_agent::CompactionConfig {
            enabled: compaction_settings.enabled.unwrap_or(true),
            reserve_tokens: compaction_settings.reserve_tokens.unwrap_or(16384),
            keep_recent_tokens: compaction_settings.keep_recent_tokens.unwrap_or(20000),
        }
    } else {
        tau_agent::CompactionConfig::default()
    };

    // Create agent with initial config (no system prompt yet)
    let config = AgentConfig {
        system_prompt: None,
        model: model.clone(),
        reasoning,
        max_tokens: None,
        compaction,
        steering_mode: tau_agent::DequeueMode::All,
        follow_up_mode: tau_agent::DequeueMode::All,
    };
    let mut agent = Agent::new(config, transport);

    // Add tools
    agent.add_tool(Arc::new(tools::BashTool::new()));
    agent.add_tool(Arc::new(tools::ReadTool::new()));
    agent.add_tool(Arc::new(tools::WriteTool::new()));
    agent.add_tool(Arc::new(tools::EditTool::new()));
    agent.add_tool(Arc::new(tools::GlobTool::new()));
    agent.add_tool(Arc::new(tools::GrepTool::new()));
    agent.add_tool(Arc::new(tools::ListTool::new()));

    // Build dynamic system prompt based on registered tools
    let tool_names = agent.tool_names();
    agent.set_system_prompt(build_system_prompt(&tool_names));

    // Resume session if specified
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
        return run_command(&mut agent, &command, &model).await;
    }

    // TUI mode
    if use_tui {
        let mut model = model;
        let mut reasoning = reasoning;
        let available_models = get_available_models();
        return ui::run_tui(&mut agent, &mut model, &mut reasoning, &available_models).await;
    }

    // Interactive mode (simple stdin/stdout)
    // Create a new session for auto-save
    let mut session = session::SessionManager::new(&model.id).ok();
    let mut model = model;
    let mut reasoning = reasoning;
    run_interactive(&mut agent, &mut model, &mut reasoning, session.as_mut()).await
}

async fn run_command(agent: &mut Agent, command: &str, model: &Model) -> anyhow::Result<()> {
    println!("tau> {}", command);
    println!();

    let mut receiver = agent.subscribe();
    let model_for_cost = model.clone();

    // Spawn event handler
    let handle = tokio::spawn(async move {
        while let Ok(event) = receiver.recv().await {
            match event {
                AgentEvent::MessageUpdate { message } => {
                    // Print streaming text
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
                    println!("[Compacting context ({})]", crate::utils::compaction_reason_str(reason));
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
                }
                _ => {}
            }
        }
    });

    agent.prompt(command).await?;

    // Wait a bit for final events
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    handle.abort();

    Ok(())
}

async fn run_interactive(
    agent: &mut Agent,
    model: &mut Model,
    reasoning: &mut ReasoningLevel,
    mut session: Option<&mut session::SessionManager>,
) -> anyhow::Result<()> {
    use std::io::{self, Write};

    let available_models = get_available_models();

    // Show minimal startup info (only if TTY)
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
            // EOF
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle slash commands
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
                        // In CLI mode, just list the models
                        println!(
                            "{}",
                            commands::ModelCommand::list_models_text(model, &available_models)
                        );
                    }
                    commands::CommandResult::OpenBranchSelector => {
                        // In CLI mode, show message list and ask for index
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
                                // Truncate agent messages to branch point
                                if let Some(idx) = branch_index {
                                    let messages: Vec<_> =
                                        agent.messages().iter().take(idx + 1).cloned().collect();
                                    agent.set_messages(messages);
                                } else {
                                    agent.clear_messages();
                                }
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

        // Spawn event handler
        // Check if stdout is a TTY for cursor handling
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
                    AgentEvent::ToolExecutionUpdate {
                        content, ..
                    } => {
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
                    }
                    AgentEvent::CompactionStart { reason } => {
                        println!("[Compacting context ({})]", crate::utils::compaction_reason_str(reason));
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

        // Save user message to session before prompting
        if let Some(ref mut s) = session {
            let user_msg = tau_ai::Message::user(input);
            let _ = s.append_message(&user_msg);
        }

        if let Err(e) = agent.prompt(input).await {
            eprintln!("Error: {}", e);
        }

        // Save assistant response to session
        if let Some(ref mut s) = session {
            // Get the last message (should be assistant response)
            if let Some(last_msg) = agent.messages().last() {
                let _ = s.append_message(last_msg);
            }
            // Save usage
            let _ = s.append_usage(&agent.state().total_usage);
        }

        // Wait for events to finish
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        handle.abort();

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

/// Build dynamic system prompt based on available tools
fn build_system_prompt(tool_names: &[&str]) -> String {
    let has_bash = tool_names.contains(&"bash");
    let has_read = tool_names.contains(&"read");
    let has_write = tool_names.contains(&"write");
    let has_edit = tool_names.contains(&"edit");
    let has_glob = tool_names.contains(&"glob");
    let has_grep = tool_names.contains(&"grep");
    let has_list = tool_names.contains(&"list");

    let can_modify = has_write || has_edit || has_bash;

    let mut prompt = String::from("You are tau, an AI-powered coding assistant.\n\n");

    // Mode notice if read-only
    if !can_modify {
        prompt.push_str("NOTE: You are in READ-ONLY mode. You can explore and analyze code but cannot make changes.\n\n");
    }

    // Tool descriptions - only list tools that are actually available
    if !tool_names.is_empty() {
        prompt.push_str("Tools:\n");
        if has_bash {
            prompt.push_str("- bash: Execute shell commands\n");
        }
        if has_read {
            prompt.push_str("- read: Read file contents\n");
        }
        if has_write {
            prompt.push_str("- write: Write content to a file\n");
        }
        if has_edit {
            prompt.push_str("- edit: Make text replacements in files\n");
        }
        if has_glob {
            prompt.push_str("- glob: Find files by pattern\n");
        }
        if has_grep {
            prompt.push_str("- grep: Search file contents\n");
        }
        if has_list {
            prompt.push_str("- list: List directory contents\n");
        }
        prompt.push('\n');
    }

    // Conditional guidelines based on available tools
    prompt.push_str("Guidelines:\n");
    prompt.push_str("- Be concise and helpful\n");

    if has_read && (has_edit || has_write) {
        prompt.push_str("- Always read files before making edits\n");
    }

    if has_edit && has_write {
        prompt.push_str("- Use edit for small changes, write for new files\n");
    }

    if has_glob || has_grep {
        prompt.push_str("- Use glob/grep to explore before making changes\n");
    }

    if has_bash {
        prompt.push_str("- Warn before destructive commands\n");
    }

    // Add current working directory
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());
    prompt.push_str(&format!("\nWorking directory: {}", cwd));

    // Load hierarchical context files (AGENTS.md, CLAUDE.md)
    if let Some(context_content) = context::load_context() {
        format!(
            "{}\n\n---\n\n# Project Context\n\n{}",
            prompt, context_content
        )
    } else {
        prompt
    }
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

            // Try to open browser
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
