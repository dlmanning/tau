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
    builder.add_tool(Arc::new(tau_tools::SubagentReportTool::new()));

    let lsp_manager = Arc::new(lsp::LspManager::new(std::env::current_dir()?).await);
    if lsp_manager.is_available() {
        builder.add_tool(Arc::new(lsp::LspTool::new(lsp_manager.clone())));
    }

    // Add agent tool (subagent spawning).
    //
    // The runtime is ignorant of which spec names exist. We build a host-
    // owned spec map once, then install a SpecResolver on the parent's
    // AgentTool. The resolver:
    //   1. Looks up the base spec by name in the precomputed map.
    //   2. If the spec allows nested subagents AND `depth + 1 < MAX_DEPTH`,
    //      appends an `AgentTool` configured with the bumped depth and a
    //      recursive ref to the same resolver. This is the host's
    //      depth-limit enforcement; the runtime knows nothing about depth.
    let agent_handle = builder.pre_handle();
    let manager = Arc::new(
        tau_agent::manager::AgentManager::new(
            builder.event_sender(),
            builder.config().clone(),
            transport.clone(),
            20,
        )
        .with_parent_interaction_sender(interaction_tx),
    );

    // Precompute the base spec map once.
    let base_specs = build_base_specs(builder.tools());

    // Recursive resolver: closure captures a Weak ref to itself via
    // OnceLock, so the AgentTool it appends per spawn can resolve
    // again when the LLM nests further.
    const MAX_DEPTH: u32 = 3;
    use std::sync::OnceLock as StdOnceLock;
    use std::sync::Weak as StdWeak;
    let resolver_self: Arc<StdOnceLock<StdWeak<dyn Fn(&str, u32) -> Option<tau_agent::AgentSpec>
        + Send + Sync>>> = Arc::new(StdOnceLock::new());
    let resolver_self_for_closure = resolver_self.clone();
    let mgr_for_resolver = manager.clone();
    let base_specs_arc = Arc::new(base_specs);
    let base_specs_for_closure = base_specs_arc.clone();
    let resolver: tau_tools::SpecResolver = Arc::new(move |name: &str, depth: u32| {
        let mut spec = base_specs_for_closure.get(name).cloned()?;
        if let Some(ref allowed) = spec.allowed_subagent_specs {
            if depth + 1 < MAX_DEPTH {
                let recursive: tau_tools::SpecResolver = resolver_self_for_closure
                    .get()
                    .and_then(StdWeak::upgrade)
                    .expect("resolver self-ref not yet set");
                let nested = tau_tools::AgentTool::new(
                    mgr_for_resolver.clone(),
                    depth + 1,
                )
                .with_spec_resolver(recursive)
                .with_allowed_specs(allowed.clone());
                spec.tools.push(Arc::new(nested));
            }
        }
        Some(spec)
    });
    let _ = resolver_self.set(Arc::downgrade(&resolver));

    let mgr_for_send = manager.clone();
    let mgr_for_commands = manager.clone();
    let resolver_for_host = resolver.clone();
    let agent_tool = tau_tools::AgentTool::new(manager.clone(), 0)
        .with_handle(agent_handle)
        .with_spec_resolver(resolver);
    builder.add_tool(Arc::new(agent_tool));
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
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let acolyte_mode = cfg.acolyte_mode.unwrap_or(false);
    let parent_system_prompt = {
        let tool_names = builder.tool_names();
        let prompt_opts = tau_agent::prompts::PromptOptions {
            tool_names: &tool_names,
            cwd: &cwd,
            acolyte_mode,
        };
        tau_agent::prompts::build_system_prompt(&prompt_opts)
    };
    builder.set_system_prompt(parent_system_prompt.clone());

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

    // Snapshot pieces needed to build the "root" spec before consuming the builder.
    let root_tools: Vec<tau_agent::BoxedTool> = builder.tools().to_vec();
    let root_spec = tau_agent::AgentSpec {
        system_prompt: parent_system_prompt,
        tools: root_tools,
        max_turns: 200,
        allows_worktree: true,
        allowed_subagent_specs: Some(vec![
            "general-purpose".into(),
            "explore".into(),
            "plan".into(),
        ]),
    };

    // Spawn the agent actor — from here on we use the handle
    let handle = builder.spawn();
    // Register the root with the manager so handle.respec works.
    let _root_id = manager.adopt(&handle, root_spec).await;

    // Non-interactive mode
    let result = if let Some(command) = args.command {
        run_command::run_command(&handle, &command, interaction_rx).await
    } else if use_tui {
        // TODO(approval-ui): until the TUI renders `tool.confirm` prompts,
        // auto-accept all elevated calls so bash etc. don't get rejected.
        // Same override is applied to the subagent manager so spawned
        // subagents inherit the same policy.
        let auto = Arc::new(tau_agent::AutoAcceptAllPolicy);
        handle.set_approval_policy(auto.clone()).await?;
        mgr_for_commands.set_default_approval_policy(auto);
        // TUI mode
        let available_models = get_available_models();
        ui::run_tui(
            handle.clone(),
            &available_models,
            interaction_rx,
            mgr_for_commands.clone(),
            resolver_for_host,
        )
        .await
    } else {
        // Interactive mode (simple stdin/stdout)
        let session = resumed_session.or_else(|| session::SessionManager::new(&model.id).ok());
        interactive::run_interactive(
            handle.clone(),
            session,
            interaction_rx,
            mgr_for_commands,
            resolver_for_host,
        )
        .await
    };

    lsp_manager.shutdown_all().await;
    result
}

/// Build the host's base spec map: `general-purpose` / `explore` / `plan`
/// (and a `general-purpose:executor` variant for plan execution). Each
/// spec carries the bare tool list (no nested AgentTool); the resolver
/// appends one with the right depth + recursion guard at use time.
fn build_base_specs(
    all_tools: &[tau_agent::BoxedTool],
) -> std::collections::HashMap<String, tau_agent::AgentSpec> {
    let read_only_names = ["read", "glob", "grep", "list", "lsp"];
    let read_only_tools: Vec<tau_agent::BoxedTool> = all_tools
        .iter()
        .filter(|t| read_only_names.contains(&t.name()))
        .cloned()
        .collect();
    let report_tool = all_tools
        .iter()
        .find(|t| t.name() == "subagent_report")
        .cloned();
    let submit_plan_tool = all_tools
        .iter()
        .find(|t| t.name() == "submit_plan")
        .cloned();
    let general_purpose_tools: Vec<tau_agent::BoxedTool> = all_tools
        .iter()
        .filter(|t| t.name() != "agent")
        .cloned()
        .collect();

    let general_prompt = include_str!("prompts/agent_general.md");
    let explore_prompt = include_str!("prompts/agent_explore.md");
    let plan_prompt = include_str!("prompts/agent_plan.md");
    let executor_suffix = include_str!("prompts/agent_executor.md");

    let mut explore_tools = read_only_tools.clone();
    if let Some(ref t) = report_tool {
        explore_tools.push(t.clone());
    }
    let mut plan_tools = read_only_tools.clone();
    if let Some(ref t) = report_tool {
        plan_tools.push(t.clone());
    }
    if let Some(ref t) = submit_plan_tool {
        plan_tools.push(t.clone());
    }

    let mut map = std::collections::HashMap::new();
    map.insert(
        "general-purpose".into(),
        tau_agent::AgentSpec {
            system_prompt: general_prompt.to_string(),
            tools: general_purpose_tools.clone(),
            max_turns: 200,
            allows_worktree: true,
            allowed_subagent_specs: Some(vec![
                "general-purpose".into(),
                "explore".into(),
                "plan".into(),
            ]),
        },
    );
    map.insert(
        "explore".into(),
        tau_agent::AgentSpec {
            system_prompt: explore_prompt.to_string(),
            tools: explore_tools,
            max_turns: 200,
            allows_worktree: false,
            allowed_subagent_specs: None,
        },
    );
    map.insert(
        "plan".into(),
        tau_agent::AgentSpec {
            system_prompt: plan_prompt.to_string(),
            tools: plan_tools,
            max_turns: 200,
            allows_worktree: false,
            allowed_subagent_specs: Some(vec!["explore".into(), "plan".into()]),
        },
    );
    // Executor variant: same tool set as general-purpose, but the prompt
    // ends with the executor instructions. Spawned by the host on /plan
    // approve with `inherit_history_from = <plan_agent_id>`.
    map.insert(
        "general-purpose:executor".into(),
        tau_agent::AgentSpec {
            system_prompt: format!("{general_prompt}\n\n{executor_suffix}"),
            tools: general_purpose_tools,
            max_turns: 200,
            allows_worktree: true,
            allowed_subagent_specs: Some(vec![
                "general-purpose".into(),
                "explore".into(),
                "plan".into(),
            ]),
        },
    );
    map
}

#[cfg(test)]
mod spec_tests {
    use super::build_base_specs;

    #[test]
    fn executor_spec_carries_executor_suffix() {
        // Build with empty tool list; we only care about the prompt.
        let map = build_base_specs(&[]);
        let exec = map
            .get("general-purpose:executor")
            .expect("executor spec registered");
        assert!(
            exec.system_prompt.contains("Plan Executor Mode"),
            "executor spec carries the executor prompt suffix"
        );
        let general = map
            .get("general-purpose")
            .expect("general-purpose registered");
        assert!(
            !general.system_prompt.contains("Plan Executor Mode"),
            "non-executor spec does NOT carry the suffix"
        );
    }
}
