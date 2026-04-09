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

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use tau_ai::{Content, Message, Model};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
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

/// Callback for receiving progress updates from a running subagent.
pub type ProgressCallback = Arc<dyn Fn(&str) + Send + Sync>;

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
    /// Optional progress callback — receives tool execution updates.
    pub on_progress: Option<ProgressCallback>,
}

/// Factory function to create a depth-limited Agent tool.
pub type AgentToolFactory = Arc<dyn Fn(u32) -> BoxedTool + Send + Sync>;

/// Result from a completed subagent.
#[derive(Debug)]
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

// ============================================================================
// Agent registry — keeps completed agents alive for resumption
// ============================================================================

/// Maximum number of agents to keep in the registry before evicting oldest.
#[allow(dead_code)] // used inside async methods
const MAX_REGISTRY_SIZE: usize = 20;

/// Registry of completed agents that can be resumed with new messages.
pub struct AgentRegistry {
    agents: Mutex<HashMap<String, AgentEntry>>,
}

struct AgentEntry {
    agent: Agent,
    description: String,
}

impl AgentRegistry {
    pub fn new() -> Self {
        Self {
            agents: Mutex::new(HashMap::new()),
        }
    }

    /// Store a completed agent for later resumption.
    /// Evicts an arbitrary entry if the registry is at capacity.
    pub async fn store(&self, id: String, agent: Agent, description: String) {
        let mut agents = self.agents.lock().await;
        if agents.len() >= MAX_REGISTRY_SIZE {
            if let Some(evict_id) = agents.keys().next().cloned() {
                agents.remove(&evict_id);
            }
        }
        agents.insert(id, AgentEntry { agent, description });
    }

    /// Resume a previously stored agent with a new message.
    /// Temporarily removes the agent from the registry while it runs
    /// to avoid holding the lock across API calls.
    pub async fn resume(
        &self,
        id: &str,
        message: &str,
        on_progress: Option<&ProgressCallback>,
    ) -> crate::error::Result<SubagentResult> {
        let start = std::time::Instant::now();

        // Take the agent out so we don't hold the lock during prompt()
        let mut entry = {
            let mut agents = self.agents.lock().await;
            agents.remove(id).ok_or_else(|| {
                crate::error::Error::Other(format!(
                    "No agent with ID '{}'. It may have been evicted, never stored, or is currently running.",
                    id
                ))
            })?
        };

        // Forward progress events if callback provided
        let progress_task = if let Some(cb) = on_progress {
            let mut events = entry.agent.subscribe();
            let cb = cb.clone();
            Some(tokio::spawn(async move {
                while let Ok(event) = events.recv().await {
                    match &event {
                        crate::events::AgentEvent::ToolExecutionStart { tool_name, .. } => {
                            cb(&format!("[{}...]", tool_name));
                        }
                        crate::events::AgentEvent::ToolExecutionEnd { tool_name, is_error, .. } => {
                            if *is_error {
                                cb(&format!("[{} error]", tool_name));
                            } else {
                                cb(&format!("[{} done]", tool_name));
                            }
                        }
                        _ => {}
                    }
                }
            }))
        } else {
            None
        };

        let prompt_result = entry.agent.prompt(message).await;

        if let Some(task) = progress_task {
            task.abort();
        }

        // Collect data before re-inserting (avoids borrow-after-move)
        let text = extract_final_text(entry.agent.messages());
        let input_tokens = entry.agent.state().total_usage.input;
        let output_tokens = entry.agent.state().total_usage.output;
        let tool_use_count = entry
            .agent
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

        // Record transcript
        record_transcript(id, entry.agent.messages()).await;

        // Put it back regardless of success/failure
        self.agents.lock().await.insert(id.to_string(), entry);

        // Propagate error after re-inserting
        prompt_result?;

        Ok(SubagentResult {
            agent_id: id.to_string(),
            text,
            input_tokens,
            output_tokens,
            tool_use_count,
            duration_ms: start.elapsed().as_millis() as u64,
            worktree_path: None,
            worktree_branch: None,
        })
    }

    /// List all stored agents (id, description).
    pub async fn list(&self) -> Vec<(String, String)> {
        self.agents
            .lock()
            .await
            .iter()
            .map(|(id, e)| (id.clone(), e.description.clone()))
            .collect()
    }

    /// Number of stored agents.
    pub async fn len(&self) -> usize {
        self.agents.lock().await.len()
    }
}

impl Default for AgentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Transcript recording
// ============================================================================

/// Record a subagent's conversation to disk for debugging.
/// Writes JSONL to `~/.local/share/tau/agent-transcripts/{agent_id}.jsonl`.
/// Failures are logged and silently ignored.
pub async fn record_transcript(agent_id: &str, messages: &[Message]) {
    let dir = match dirs::data_dir() {
        Some(d) => d.join("tau/agent-transcripts"),
        None => return,
    };

    if tokio::fs::create_dir_all(&dir).await.is_err() {
        return;
    }

    let path = dir.join(format!("{}.jsonl", agent_id));
    let mut file = match tokio::fs::File::create(&path).await {
        Ok(f) => f,
        Err(e) => {
            tracing::debug!("Failed to create transcript file: {}", e);
            return;
        }
    };

    for msg in messages {
        if let Ok(json) = serde_json::to_string(msg) {
            let _ = file.write_all(format!("{}\n", json).as_bytes()).await;
        }
    }
}

/// Run a subagent to completion and return its result plus the agent
/// (for optional storage in a registry). Worktree cleanup runs even if the agent fails.
/// Transcripts are recorded to `~/.local/share/tau/agent-transcripts/`.
pub async fn run_subagent(config: SubagentConfig) -> crate::error::Result<(SubagentResult, Agent)> {
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

    let (mut result, agent) = agent_result?;

    // Record transcript
    record_transcript(&agent_id, agent.messages()).await;

    result.worktree_path = wt_path;
    result.worktree_branch = wt_branch;
    result.duration_ms = start.elapsed().as_millis() as u64;
    Ok((result, agent))
}

/// Inner agent execution — separated so worktree cleanup can run regardless.
async fn run_agent_inner(
    config: &SubagentConfig,
    agent_id: &str,
    worktree: &Option<WorktreeInfo>,
) -> crate::error::Result<(SubagentResult, Agent)> {
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

    // Forward progress events if callback provided
    let progress_task = if let Some(ref on_progress) = config.on_progress {
        let mut events = agent.subscribe();
        let cb = on_progress.clone();
        Some(tokio::spawn(async move {
            while let Ok(event) = events.recv().await {
                match &event {
                    crate::events::AgentEvent::ToolExecutionStart {
                        tool_name, ..
                    } => {
                        cb(&format!("[{}...]", tool_name));
                    }
                    crate::events::AgentEvent::ToolExecutionEnd {
                        tool_name,
                        is_error,
                        ..
                    } => {
                        if *is_error {
                            cb(&format!("[{} error]", tool_name));
                        } else {
                            cb(&format!("[{} done]", tool_name));
                        }
                    }
                    _ => {}
                }
            }
        }))
    } else {
        None
    };

    // Run the agent
    let result = agent.prompt(&config.prompt).await;

    // Abort progress forwarder before returning
    if let Some(task) = progress_task {
        task.abort();
    }

    result?;

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

    let result = SubagentResult {
        agent_id: agent_id.to_string(),
        text,
        input_tokens: state.total_usage.input,
        output_tokens: state.total_usage.output,
        tool_use_count,
        duration_ms: 0, // filled in by caller
        worktree_path: None,
        worktree_branch: None,
    };
    Ok((result, agent))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;
    use async_trait::async_trait;
    use tau_ai::{AssistantMetadata, Content, Message};

    // Minimal mock tool for testing tool filtering
    struct MockTool {
        tool_name: &'static str,
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            self.tool_name
        }
        fn description(&self) -> &str {
            ""
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        async fn execute(
            &self,
            _id: &str,
            _args: serde_json::Value,
            _cancel: CancellationToken,
        ) -> crate::tool::ToolResult {
            crate::tool::ToolResult::text("")
        }
    }

    fn mock_tools() -> Vec<BoxedTool> {
        vec![
            Arc::new(MockTool { tool_name: "bash" }),
            Arc::new(MockTool { tool_name: "read" }),
            Arc::new(MockTool { tool_name: "write" }),
            Arc::new(MockTool { tool_name: "edit" }),
            Arc::new(MockTool { tool_name: "glob" }),
            Arc::new(MockTool { tool_name: "grep" }),
            Arc::new(MockTool { tool_name: "list" }),
            Arc::new(MockTool { tool_name: "lsp" }),
            Arc::new(MockTool { tool_name: "agent" }),
        ]
    }

    // ── AgentType ──

    #[test]
    fn test_parse_valid_types() {
        assert!(matches!(AgentType::parse("general-purpose"), Some(AgentType::GeneralPurpose)));
        assert!(matches!(AgentType::parse("Explore"), Some(AgentType::Explore)));
        assert!(matches!(AgentType::parse("Plan"), Some(AgentType::Plan)));
    }

    #[test]
    fn test_parse_invalid_type() {
        assert!(AgentType::parse("unknown").is_none());
        assert!(AgentType::parse("explore").is_none()); // case-sensitive
        assert!(AgentType::parse("").is_none());
    }

    #[test]
    fn test_display() {
        assert_eq!(AgentType::GeneralPurpose.to_string(), "general-purpose");
        assert_eq!(AgentType::Explore.to_string(), "Explore");
        assert_eq!(AgentType::Plan.to_string(), "Plan");
    }

    #[test]
    fn test_max_turns() {
        assert_eq!(AgentType::GeneralPurpose.max_turns(), 30);
        assert_eq!(AgentType::Explore.max_turns(), 10);
        assert_eq!(AgentType::Plan.max_turns(), 15);
    }

    // ── Tool filtering ──

    #[test]
    fn test_explore_gets_read_only_tools() {
        let all = mock_tools();
        let filtered = build_tool_set(&AgentType::Explore, &all, 0, &None);
        let names: Vec<&str> = filtered.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"read"));
        assert!(names.contains(&"glob"));
        assert!(names.contains(&"grep"));
        assert!(names.contains(&"list"));
        assert!(names.contains(&"lsp"));
        assert!(!names.contains(&"bash"));
        assert!(!names.contains(&"write"));
        assert!(!names.contains(&"edit"));
        assert!(!names.contains(&"agent"));
    }

    #[test]
    fn test_plan_gets_read_only_tools() {
        let all = mock_tools();
        let filtered = build_tool_set(&AgentType::Plan, &all, 0, &None);
        let names: Vec<&str> = filtered.iter().map(|t| t.name()).collect();
        assert_eq!(names.len(), 5);
        assert!(!names.contains(&"bash"));
        assert!(!names.contains(&"agent"));
    }

    #[test]
    fn test_general_purpose_gets_all_except_agent() {
        let all = mock_tools();
        let filtered = build_tool_set(&AgentType::GeneralPurpose, &all, 0, &None);
        let names: Vec<&str> = filtered.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"bash"));
        assert!(names.contains(&"read"));
        assert!(names.contains(&"write"));
        assert!(!names.contains(&"agent")); // removed, no factory
    }

    #[test]
    fn test_general_purpose_gets_agent_tool_from_factory() {
        let all = mock_tools();
        let factory: AgentToolFactory = Arc::new(|_depth| {
            Arc::new(MockTool { tool_name: "agent" })
        });
        let filtered = build_tool_set(&AgentType::GeneralPurpose, &all, 0, &Some(factory));
        let names: Vec<&str> = filtered.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"agent")); // added by factory
    }

    #[test]
    fn test_depth_limit_excludes_agent_tool() {
        let all = mock_tools();
        let factory: AgentToolFactory = Arc::new(|_depth| {
            Arc::new(MockTool { tool_name: "agent" })
        });
        // At MAX_AGENT_DEPTH - 1, the next level would be MAX_AGENT_DEPTH → excluded
        let filtered = build_tool_set(
            &AgentType::GeneralPurpose,
            &all,
            MAX_AGENT_DEPTH - 1,
            &Some(factory),
        );
        let names: Vec<&str> = filtered.iter().map(|t| t.name()).collect();
        assert!(!names.contains(&"agent")); // depth exceeded
    }

    // ── extract_final_text ──

    #[test]
    fn test_extract_final_text_from_assistant() {
        let messages = vec![
            Message::user("hello"),
            Message::Assistant {
                content: vec![Content::text("first response")],
                metadata: AssistantMetadata::default(),
            },
            Message::user("follow up"),
            Message::Assistant {
                content: vec![Content::text("second response")],
                metadata: AssistantMetadata::default(),
            },
        ];
        assert_eq!(extract_final_text(&messages), "second response");
    }

    #[test]
    fn test_extract_final_text_skips_tool_calls() {
        let messages = vec![
            Message::Assistant {
                content: vec![
                    Content::text("here is my answer"),
                    Content::tool_call("c1", "bash", serde_json::json!({})),
                ],
                metadata: AssistantMetadata::default(),
            },
        ];
        assert_eq!(extract_final_text(&messages), "here is my answer");
    }

    #[test]
    fn test_extract_final_text_empty_conversation() {
        let messages: Vec<Message> = vec![];
        assert_eq!(extract_final_text(&messages), "");
    }

    #[test]
    fn test_extract_final_text_no_assistant() {
        let messages = vec![Message::user("hello")];
        assert_eq!(extract_final_text(&messages), "");
    }

    // ── agent_type_suffix ──

    #[test]
    fn test_suffixes_are_nonempty() {
        assert!(!agent_type_suffix(&AgentType::GeneralPurpose).is_empty());
        assert!(!agent_type_suffix(&AgentType::Explore).is_empty());
        assert!(!agent_type_suffix(&AgentType::Plan).is_empty());
    }

    #[test]
    fn test_suffixes_are_distinct() {
        let gp = agent_type_suffix(&AgentType::GeneralPurpose);
        let ex = agent_type_suffix(&AgentType::Explore);
        let pl = agent_type_suffix(&AgentType::Plan);
        assert_ne!(gp, ex);
        assert_ne!(gp, pl);
        assert_ne!(ex, pl);
    }

    // ── Progress callback ──

    #[test]
    fn test_progress_callback_type() {
        let received = Arc::new(std::sync::Mutex::new(Vec::new()));
        let received_clone = received.clone();
        let cb: ProgressCallback = Arc::new(move |msg: &str| {
            received_clone.lock().unwrap().push(msg.to_string());
        });

        cb("[bash...]");
        cb("[bash done]");
        cb("[read...]");

        let msgs = received.lock().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0], "[bash...]");
        assert_eq!(msgs[1], "[bash done]");
        assert_eq!(msgs[2], "[read...]");
    }

    #[tokio::test]
    async fn test_progress_forwarding_from_agent_events() {
        // Simulate the progress forwarding logic from run_agent_inner:
        // subscribe to an event channel, emit tool events, verify callback is called
        use crate::events::AgentEvent;

        let (tx, _rx) = tokio::sync::broadcast::channel::<AgentEvent>(16);

        let received = Arc::new(std::sync::Mutex::new(Vec::new()));
        let received_clone = received.clone();
        let cb: ProgressCallback = Arc::new(move |msg: &str| {
            received_clone.lock().unwrap().push(msg.to_string());
        });

        // Subscribe and spawn forwarder (same logic as run_agent_inner)
        let mut events = tx.subscribe();
        let cb_clone = cb.clone();
        let task = tokio::spawn(async move {
            while let Ok(event) = events.recv().await {
                match &event {
                    AgentEvent::ToolExecutionStart { tool_name, .. } => {
                        cb_clone(&format!("[{}...]", tool_name));
                    }
                    AgentEvent::ToolExecutionEnd { tool_name, is_error, .. } => {
                        if *is_error {
                            cb_clone(&format!("[{} error]", tool_name));
                        } else {
                            cb_clone(&format!("[{} done]", tool_name));
                        }
                    }
                    _ => {}
                }
            }
        });

        // Emit events
        tx.send(AgentEvent::ToolExecutionStart {
            tool_call_id: "c1".into(),
            tool_name: "bash".into(),
            arguments: serde_json::json!({}),
        }).unwrap();
        tx.send(AgentEvent::ToolExecutionEnd {
            tool_call_id: "c1".into(),
            tool_name: "bash".into(),
            result: "ok".into(),
            is_error: false,
        }).unwrap();
        tx.send(AgentEvent::ToolExecutionStart {
            tool_call_id: "c2".into(),
            tool_name: "read".into(),
            arguments: serde_json::json!({}),
        }).unwrap();
        tx.send(AgentEvent::ToolExecutionEnd {
            tool_call_id: "c2".into(),
            tool_name: "read".into(),
            result: "failed".into(),
            is_error: true,
        }).unwrap();

        // Drop sender to close channel, which ends the forwarder
        drop(tx);
        let _ = task.await;

        let msgs = received.lock().unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0], "[bash...]");
        assert_eq!(msgs[1], "[bash done]");
        assert_eq!(msgs[2], "[read...]");
        assert_eq!(msgs[3], "[read error]");
    }

    // ── AgentRegistry ──

    // Minimal transport that never gets called (agent is stored, not run)
    struct DummyTransport;

    #[async_trait]
    impl crate::transport::Transport for DummyTransport {
        async fn run(
            &self,
            _messages: Vec<tau_ai::Message>,
            _config: &crate::transport::AgentRunConfig,
            _cancel: CancellationToken,
        ) -> tau_ai::Result<crate::transport::AgentEventStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn test_registry_store_and_list() {
        use crate::compaction::CompactionConfig;

        let registry = AgentRegistry::new();
        assert_eq!(registry.len().await, 0);

        let transport: Arc<dyn crate::transport::Transport> = Arc::new(DummyTransport);
        let config = AgentConfig {
            system_prompt: None,
            model: tau_ai::Model {
                id: "test".into(),
                name: "test".into(),
                api: tau_ai::Api::AnthropicMessages,
                provider: tau_ai::Provider::Anthropic,
                base_url: "http://localhost".into(),
                reasoning: false,
                input_types: vec![],
                cost: tau_ai::CostInfo::default(),
                context_window: 200000,
                max_tokens: 4096,
                headers: Default::default(),
            },
            reasoning: tau_ai::ReasoningLevel::Off,
            thinking_adaptive: false,
            max_tokens: None,
            compaction: CompactionConfig::default(),
            steering_mode: crate::agent::DequeueMode::All,
            follow_up_mode: crate::agent::DequeueMode::All,
            cache_scope: None,
            cache_ttl: None,
            system_prompt_boundary: None,
        };
        let agent = Agent::new(config, transport);

        registry.store("abc123".into(), agent, "test agent".into()).await;
        assert_eq!(registry.len().await, 1);

        let list = registry.list().await;
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].0, "abc123");
        assert_eq!(list[0].1, "test agent");
    }

    #[tokio::test]
    async fn test_registry_resume_nonexistent() {
        let registry = AgentRegistry::new();
        let result = registry.resume("nonexistent", "hello", None).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("No agent with ID"), "got: {}", err);
    }

    #[tokio::test]
    async fn test_registry_eviction() {
        use crate::compaction::CompactionConfig;

        let registry = AgentRegistry::new();

        let make_agent = || {
            let transport: Arc<dyn crate::transport::Transport> = Arc::new(DummyTransport);
            let config = AgentConfig {
                system_prompt: None,
                model: tau_ai::Model {
                    id: "test".into(),
                    name: "test".into(),
                    api: tau_ai::Api::AnthropicMessages,
                    provider: tau_ai::Provider::Anthropic,
                    base_url: "http://localhost".into(),
                    reasoning: false,
                    input_types: vec![],
                    cost: tau_ai::CostInfo::default(),
                    context_window: 200000,
                    max_tokens: 4096,
                    headers: Default::default(),
                },
                reasoning: tau_ai::ReasoningLevel::Off,
                thinking_adaptive: false,
                max_tokens: None,
                compaction: CompactionConfig::default(),
                steering_mode: crate::agent::DequeueMode::All,
                follow_up_mode: crate::agent::DequeueMode::All,
                cache_scope: None,
                cache_ttl: None,
                system_prompt_boundary: None,
            };
            Agent::new(config, transport)
        };

        // Fill to capacity
        for i in 0..MAX_REGISTRY_SIZE {
            registry
                .store(format!("agent-{}", i), make_agent(), format!("desc-{}", i))
                .await;
        }
        assert_eq!(registry.len().await, MAX_REGISTRY_SIZE);

        // One more should evict
        registry
            .store("agent-overflow".into(), make_agent(), "overflow".into())
            .await;
        assert_eq!(registry.len().await, MAX_REGISTRY_SIZE);

        // The new one should be present
        let list = registry.list().await;
        assert!(list.iter().any(|(id, _)| id == "agent-overflow"));
    }
}
