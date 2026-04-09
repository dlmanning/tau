//! Subagent spawning — create independent agent instances for parallel work.

use std::path::PathBuf;
use std::sync::Arc;

use tau_ai::{Content, Message, Model};

use crate::agent::{Agent, AgentConfig};
use crate::tool::BoxedTool;
use crate::transport::Transport;

/// Maximum recursion depth for nested agent spawning.
pub const MAX_AGENT_DEPTH: u32 = 3;

/// Agent type determines tool set and system prompt.
#[derive(Debug, Clone)]
pub enum AgentType {
    GeneralPurpose,
    Explore,
    Plan,
}

impl AgentType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "general-purpose" => Some(Self::GeneralPurpose),
            "Explore" => Some(Self::Explore),
            "Plan" => Some(Self::Plan),
            _ => None,
        }
    }

    pub fn is_read_only(&self) -> bool {
        matches!(self, Self::Explore | Self::Plan)
    }
}

/// Configuration for spawning a subagent.
pub struct SubagentConfig {
    pub agent_type: AgentType,
    pub prompt: String,
    pub description: String,
    pub model: Option<Model>,
    pub cwd: Option<String>,
    pub isolation: Option<String>,
    pub depth: u32,
    pub transport: Arc<dyn Transport>,
    pub all_tools: Vec<BoxedTool>,
    pub parent_config: AgentConfig,
    /// Factory for creating depth-limited Agent tools for recursive subagents.
    /// Called with (depth) → tool. If None, general-purpose subagents won't
    /// be able to spawn their own subagents.
    pub agent_tool_factory: Option<AgentToolFactory>,
}

/// Factory function to create a depth-limited Agent tool.
pub type AgentToolFactory = Arc<dyn Fn(u32) -> BoxedTool + Send + Sync>;

/// Result from a completed subagent.
pub struct SubagentResult {
    pub agent_id: String,
    pub text: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub tool_use_count: u32,
    pub duration_ms: u64,
    pub worktree_path: Option<String>,
    pub worktree_branch: Option<String>,
}

/// Run a subagent to completion and return its result.
pub async fn run_subagent(config: SubagentConfig) -> crate::error::Result<SubagentResult> {
    let start = std::time::Instant::now();
    let agent_id = uuid::Uuid::new_v4().to_string();

    // Worktree setup
    let worktree = if config.isolation.as_deref() == Some("worktree") {
        Some(
            create_worktree(&agent_id)
                .await
                .map_err(|e| crate::error::Error::Other(format!("Worktree setup failed: {}", e)))?,
        )
    } else {
        None
    };

    // Determine CWD
    let cwd = config
        .cwd
        .clone()
        .or_else(|| worktree.as_ref().map(|w| w.path.display().to_string()))
        .unwrap_or_else(|| {
            std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| ".".into())
        });

    // Build tool set
    let tools = build_tool_set(
        &config.agent_type,
        &config.all_tools,
        config.depth,
        &config.agent_tool_factory,
    );

    // Create agent
    let mut agent_cfg = config.parent_config.clone();
    agent_cfg.system_prompt = None;
    if let Some(ref model) = config.model {
        agent_cfg.model = model.clone();
    }

    let mut agent = Agent::new(agent_cfg, config.transport.clone());
    for tool in tools {
        agent.add_tool(tool);
    }

    // Build system prompt
    let tool_names = agent.tool_names();
    let prompt_opts = crate::prompts::PromptOptions {
        tool_names: &tool_names,
        cwd: &cwd,
        acolyte_mode: false,
    };
    let system_prompt = format!(
        "{}\n\n{}",
        crate::prompts::build_system_prompt(&prompt_opts),
        agent_type_suffix(&config.agent_type),
    );
    agent.set_system_prompt(system_prompt);

    // Run
    agent
        .prompt(&config.prompt)
        .await?;

    // Collect result
    let text = extract_final_text(agent.messages());
    let state = agent.state();

    let tool_use_count = agent
        .messages()
        .iter()
        .map(|m| match m {
            Message::Assistant { content, .. } => content
                .iter()
                .filter(|c| matches!(c, Content::ToolCall { .. }))
                .count(),
            _ => 0,
        })
        .sum::<usize>() as u32;

    // Cleanup worktree
    let (wt_path, wt_branch) = if let Some(wt) = worktree {
        match cleanup_worktree(&wt).await {
            Ok(true) => (None, None),
            _ => (
                Some(wt.path.display().to_string()),
                Some(wt.branch.clone()),
            ),
        }
    } else {
        (None, None)
    };

    Ok(SubagentResult {
        agent_id,
        text,
        input_tokens: state.total_usage.input,
        output_tokens: state.total_usage.output,
        tool_use_count,
        duration_ms: start.elapsed().as_millis() as u64,
        worktree_path: wt_path,
        worktree_branch: wt_branch,
    })
}

// ============================================================================
// Tool filtering
// ============================================================================

fn build_tool_set(
    agent_type: &AgentType,
    all_tools: &[BoxedTool],
    depth: u32,
    agent_tool_factory: &Option<AgentToolFactory>,
) -> Vec<BoxedTool> {
    match agent_type {
        AgentType::Explore | AgentType::Plan => {
            let read_only = ["read", "glob", "grep", "list", "lsp"];
            all_tools
                .iter()
                .filter(|t| read_only.contains(&t.name()))
                .cloned()
                .collect()
        }
        AgentType::GeneralPurpose => {
            let mut tools: Vec<BoxedTool> = all_tools
                .iter()
                .filter(|t| t.name() != "agent")
                .cloned()
                .collect();

            // Add depth-limited agent tool if under max depth and factory available
            if depth + 1 < MAX_AGENT_DEPTH {
                if let Some(factory) = agent_tool_factory {
                    tools.push(factory(depth + 1));
                }
            }

            tools
        }
    }
}

// ============================================================================
// System prompt suffixes
// ============================================================================

fn agent_type_suffix(agent_type: &AgentType) -> &'static str {
    match agent_type {
        AgentType::GeneralPurpose => {
            "You are a subagent. Complete the task fully — don't gold-plate, but don't \
             leave it half-done. When done, respond with a concise report covering what \
             was done and any key findings."
        }
        AgentType::Explore => {
            "You are a fast exploration agent. Use read-only tools to search the codebase \
             and answer questions. Be thorough but concise. Report what you found."
        }
        AgentType::Plan => {
            "You are a planning agent. Design implementation strategies. Identify critical \
             files, consider trade-offs, and present a concrete step-by-step plan."
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn extract_final_text(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find_map(|m| match m {
            Message::Assistant { content, .. } => {
                let text: String = content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                if text.is_empty() {
                    None
                } else {
                    Some(text)
                }
            }
            _ => None,
        })
        .unwrap_or_default()
}

// ============================================================================
// Worktree management
// ============================================================================

struct WorktreeInfo {
    path: PathBuf,
    branch: String,
    #[allow(dead_code)]
    head_commit: String,
    #[allow(dead_code)]
    git_root: PathBuf,
}

async fn create_worktree(agent_id: &str) -> Result<WorktreeInfo, String> {
    let git_root_output = tokio::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .await
        .map_err(|e| format!("git rev-parse failed: {}", e))?;

    if !git_root_output.status.success() {
        return Err("Not in a git repository".into());
    }

    let git_root = PathBuf::from(String::from_utf8_lossy(&git_root_output.stdout).trim());
    let slug = &agent_id[..8.min(agent_id.len())];
    let branch = format!("worktree-agent-{}", slug);
    let path = git_root.join(format!(".tau-worktrees/agent-{}", slug));

    // Ensure parent directory exists
    tokio::fs::create_dir_all(path.parent().unwrap())
        .await
        .map_err(|e| format!("Failed to create worktree directory: {}", e))?;

    let output = tokio::process::Command::new("git")
        .args([
            "worktree",
            "add",
            &path.display().to_string(),
            "-b",
            &branch,
        ])
        .output()
        .await
        .map_err(|e| format!("git worktree add failed: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "git worktree add failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let head_output = tokio::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .await
        .map_err(|e| format!("git rev-parse HEAD failed: {}", e))?;

    let head_commit = String::from_utf8_lossy(&head_output.stdout)
        .trim()
        .to_string();

    Ok(WorktreeInfo {
        path,
        branch,
        head_commit,
        git_root,
    })
}

async fn cleanup_worktree(info: &WorktreeInfo) -> Result<bool, String> {
    // Check for changes
    let diff = tokio::process::Command::new("git")
        .args([
            "-C",
            &info.path.display().to_string(),
            "diff",
            "--quiet",
            &info.head_commit,
        ])
        .status()
        .await
        .map_err(|e| format!("git diff failed: {}", e))?;

    if diff.success() {
        // No changes — clean up
        let _ = tokio::process::Command::new("git")
            .args(["worktree", "remove", &info.path.display().to_string()])
            .output()
            .await;
        let _ = tokio::process::Command::new("git")
            .args(["branch", "-D", &info.branch])
            .output()
            .await;
        Ok(true) // removed
    } else {
        Ok(false) // kept — has changes
    }
}
