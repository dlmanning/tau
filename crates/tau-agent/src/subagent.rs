//! Subagent spawning — create independent agent instances for parallel work.
//!
//! ## Known limitations
//!
//! - The subagent's bash tool runs with the process's actual CWD, not the
//!   worktree path. The system prompt tells the model the correct CWD, but
//!   processes spawned by bash will inherit the parent's CWD. A future fix
//!   would inject a CWD override into the bash tool.
//!
//! - Background agent notification uses XML-like tags (`<agent-completed>`)
//!   that the parent model interprets as text, not structured data.

use std::path::PathBuf;
use std::sync::Arc;

use tau_ai::{Content, Message, Model};
use tokio_util::sync::CancellationToken;

use crate::agent::{Agent, AgentConfig};
use crate::tool::BoxedTool;
use crate::transport::Transport;

/// Maximum recursion depth for nested agent spawning.
pub const MAX_AGENT_DEPTH: u32 = 3;

/// Maximum turns a subagent can execute before being stopped.
pub const MAX_SUBAGENT_TURNS: u32 = 30;

/// Agent type determines tool set and system prompt.
#[derive(Debug, Clone)]
pub enum AgentType {
    GeneralPurpose,
    Explore,
    Plan,
}

impl AgentType {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "general-purpose" => Some(Self::GeneralPurpose),
            "Explore" => Some(Self::Explore),
            "Plan" => Some(Self::Plan),
            _ => None,
        }
    }

    /// Maximum turns before logging a warning.
    fn max_turns(&self) -> u32 {
        match self {
            Self::GeneralPurpose => MAX_SUBAGENT_TURNS,
            Self::Explore => 10,
            Self::Plan => 15,
        }
    }
}

impl std::fmt::Display for AgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GeneralPurpose => write!(f, "general-purpose"),
            Self::Explore => write!(f, "Explore"),
            Self::Plan => write!(f, "Plan"),
        }
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
    pub agent_tool_factory: Option<AgentToolFactory>,
    /// Cancellation token — checked between turns.
    pub cancel: CancellationToken,
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
/// Worktree cleanup runs even if the agent fails.
pub async fn run_subagent(config: SubagentConfig) -> crate::error::Result<SubagentResult> {
    let start = std::time::Instant::now();
    let agent_id = uuid::Uuid::new_v4().to_string();

    // Worktree setup
    let worktree = if config.isolation.as_deref() == Some("worktree") {
        match create_worktree(&agent_id).await {
            Ok(wt) => Some(wt),
            Err(e) => {
                return Err(crate::error::Error::Other(format!(
                    "Worktree setup failed: {}",
                    e
                )));
            }
        }
    } else {
        None
    };

    // Run the agent, capturing the result (success or failure)
    let agent_result = run_agent_inner(&config, &agent_id, &worktree).await;

    // Always cleanup worktree, even on failure
    let (wt_path, wt_branch) = if let Some(wt) = &worktree {
        match cleanup_worktree(wt).await {
            Ok(true) => (None, None),
            _ => (
                Some(wt.path.display().to_string()),
                Some(wt.branch.clone()),
            ),
        }
    } else {
        (None, None)
    };

    // Return result with worktree info
    let mut result = agent_result?;
    result.worktree_path = wt_path;
    result.worktree_branch = wt_branch;
    result.duration_ms = start.elapsed().as_millis() as u64;
    Ok(result)
}

/// Inner agent execution — separated so worktree cleanup can run regardless.
async fn run_agent_inner(
    config: &SubagentConfig,
    agent_id: &str,
    worktree: &Option<WorktreeInfo>,
) -> crate::error::Result<SubagentResult> {
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

    // Run the agent
    agent.prompt(&config.prompt).await?;

    // Log if turn count was high
    let max_turns = config.agent_type.max_turns();
    let assistant_count = agent
        .messages()
        .iter()
        .filter(|m| matches!(m, Message::Assistant { .. }))
        .count() as u32;
    if assistant_count > max_turns {
        tracing::warn!(
            "Subagent {} ran {} turns (limit {})",
            agent_id,
            assistant_count,
            max_turns
        );
    }

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

    Ok(SubagentResult {
        agent_id: agent_id.to_string(),
        text,
        input_tokens: state.total_usage.input,
        output_tokens: state.total_usage.output,
        tool_use_count,
        duration_ms: 0, // filled in by caller
        worktree_path: None,
        worktree_branch: None,
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
    head_commit: String,
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
    // Use full UUID to avoid branch name collisions
    let branch = format!("worktree-agent-{}", agent_id);
    let path = git_root.join(format!(".tau-worktrees/agent-{}", agent_id));

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
    })
}

async fn cleanup_worktree(info: &WorktreeInfo) -> Result<bool, String> {
    let path_str = info.path.display().to_string();

    // Check for tracked changes
    let diff = tokio::process::Command::new("git")
        .args(["-C", &path_str, "diff", "--quiet", &info.head_commit])
        .status()
        .await
        .map_err(|e| format!("git diff failed: {}", e))?;

    // Check for untracked files
    let untracked = tokio::process::Command::new("git")
        .args([
            "-C",
            &path_str,
            "ls-files",
            "--others",
            "--exclude-standard",
        ])
        .output()
        .await
        .map_err(|e| format!("git ls-files failed: {}", e))?;

    let has_untracked = !String::from_utf8_lossy(&untracked.stdout)
        .trim()
        .is_empty();

    if diff.success() && !has_untracked {
        // No changes — clean up
        let _ = tokio::process::Command::new("git")
            .args(["worktree", "remove", &path_str])
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
