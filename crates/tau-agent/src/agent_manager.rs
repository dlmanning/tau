//! Subagent spawning — create independent agent instances for parallel work.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use tau_ai::{Content, Message, Model, Usage};
use tokio::io::AsyncWriteExt;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::agent::{Agent, AgentConfig};
use crate::events::AgentEvent;
use crate::handle::AgentHandle;
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

    /// Whether this agent type only gets read-only tools.
    pub fn is_read_only(&self) -> bool {
        matches!(self, Self::Explore | Self::Plan)
    }

    /// Maximum turns before the agent loop stops.
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

/// Factory function to create a depth-limited Agent tool.
/// Receives the depth and the spawning agent's handle so background
/// sub-subagents report to the correct parent.
type AgentToolFactory = Arc<dyn Fn(u32, AgentHandle) -> BoxedTool + Send + Sync>;

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

/// Lightweight request struct for spawning a subagent via AgentManager.
pub struct SpawnRequest {
    pub agent_type: AgentType,
    pub prompt: String,
    pub description: String,
    pub model: Option<Model>,
    pub cwd: Option<String>,
    pub isolation: Option<String>,
    pub depth: u32,
}

/// Whether an agent is currently executing or stored (idle).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStatus {
    Running,
    Idle,
}

/// Manages subagent lifecycle: spawn, resume, evict.
pub struct AgentManager {
    agents: tokio::sync::Mutex<VecDeque<(String, ManagedAgent)>>,
    max_agents: usize,
    transport: Arc<dyn Transport>,
    tools: Vec<BoxedTool>,
    parent_config: AgentConfig,
    /// Event sender for the parent agent. Used to forward subagent events
    /// wrapped in `AgentEvent::Subagent`.
    parent_event_tx: broadcast::Sender<AgentEvent>,
    agent_tool_factory: parking_lot::Mutex<Option<AgentToolFactory>>,
    /// Handles for agents that are currently executing (keyed by agent_id).
    /// Value is (handle, description). Inserted before prompt(), removed after.
    running_handles: tokio::sync::Mutex<HashMap<String, (AgentHandle, String)>>,
}

struct ManagedAgent {
    agent: Agent,
    description: String,
    usage_at_pause: Usage,
    messages_at_pause: usize,
}

impl AgentManager {
    pub fn new(
        parent_event_tx: broadcast::Sender<AgentEvent>,
        tools: Vec<BoxedTool>,
        parent_config: AgentConfig,
        transport: Arc<dyn Transport>,
        max_agents: usize,
    ) -> Self {
        Self {
            agents: tokio::sync::Mutex::new(VecDeque::new()),
            max_agents,
            transport,
            tools,
            parent_config,
            parent_event_tx,
            agent_tool_factory: parking_lot::Mutex::new(None),
            running_handles: tokio::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Set the factory for creating recursive agent tools.
    /// Must be called after construction (breaks circular Arc dependency).
    pub fn set_agent_tool_factory(
        &self,
        factory: Arc<dyn Fn(u32, AgentHandle) -> BoxedTool + Send + Sync>,
    ) {
        *self.agent_tool_factory.lock() = Some(factory);
    }

    /// Spawn a foreground subagent. Blocks until completion.
    /// Stores the agent for later resumption.
    /// Events are forwarded as `AgentEvent::Subagent` on the parent's event channel.
    pub async fn spawn(
        self: &Arc<Self>,
        request: SpawnRequest,
        cancel: CancellationToken,
    ) -> crate::error::Result<SubagentResult> {
        let agent_id = uuid::Uuid::new_v4().to_string();
        let description = request.description.clone();
        let result = self.run_subagent(&request, cancel, &agent_id).await;
        // Always clean up running handle (run_agent_inner inserts before prompt,
        // removes after, but if the future is dropped mid-execution it leaks).
        self.running_handles.lock().await.remove(&agent_id);
        let (result, agent) = result?;
        self.store(result.agent_id.clone(), agent, description).await;
        Ok(result)
    }

    /// Spawn a background subagent. Returns immediately with the agent_id.
    /// Events are forwarded as `AgentEvent::Subagent` on the parent's event channel.
    /// On completion, posts a SystemInjection message to parent_handle.follow_up.
    pub async fn spawn_background(
        self: &Arc<Self>,
        request: SpawnRequest,
        parent_handle: AgentHandle,
        parent_cancel: CancellationToken,
    ) -> String {
        let agent_id = uuid::Uuid::new_v4().to_string();
        let description = request.description.clone();
        let bg_cancel = CancellationToken::new();

        parent_handle.expect_follow_up();

        let manager = self.clone();
        let desc = description.clone();
        let aid = agent_id.clone();
        let bg_cancel_inner = bg_cancel.clone();

        tokio::spawn(async move {
            let result = tokio::select! {
                r = manager.run_subagent(&request, bg_cancel_inner, &aid) => r,
                _ = parent_cancel.cancelled() => {
                    bg_cancel.cancel();
                    Err(crate::error::Error::Other("Cancelled by parent".into()))
                }
            };

            // Clean up running handle (may have been inserted by run_agent_inner
            // but not removed if the future was dropped by the select above).
            manager.running_handles.lock().await.remove(&aid);

            match result {
                Ok((subresult, agent)) => {
                    manager
                        .store(subresult.agent_id.clone(), agent, desc.clone())
                        .await;

                    parent_handle.follow_up(Message::subagent_completed(
                        &subresult.agent_id,
                        &desc,
                        format!(
                            "{}\n[Agent {} | {} in + {} out tokens | {} tool calls | {}ms]",
                            subresult.text,
                            subresult.agent_id,
                            subresult.input_tokens,
                            subresult.output_tokens,
                            subresult.tool_use_count,
                            subresult.duration_ms,
                        ),
                    ));
                }
                Err(e) => {
                    parent_handle.follow_up(Message::subagent_failed(
                        &aid,
                        &desc,
                        format!("Error: {}", e),
                    ));
                }
            }

        });

        agent_id
    }

    /// Send a message to a previously stored agent (resume).
    /// Uses delta usage tracking and wires cancel bridge.
    /// Events are forwarded as `AgentEvent::Subagent` on the parent's event channel.
    pub async fn send(
        &self,
        id: &str,
        message: &str,
        parent_cancel: CancellationToken,
    ) -> crate::error::Result<SubagentResult> {
        let start = std::time::Instant::now();

        let mut entry = {
            let mut agents = self.agents.lock().await;
            let pos = agents.iter().position(|(k, _)| k == id).ok_or_else(|| {
                crate::error::Error::Other(format!(
                    "No agent with ID '{}'. It may have been evicted, never stored, or is currently running.",
                    id
                ))
            })?;
            agents.remove(pos).unwrap().1
        };

        // Use the stored snapshot (not current total_usage) for correct delta
        let usage_before = entry.usage_at_pause.clone();

        let event_task = self.spawn_event_forwarder(
            entry.agent.subscribe(),
            id,
            &entry.description,
        );

        let agent_handle = entry.agent.handle();
        let bridge = tokio::spawn({
            let parent_cancel = parent_cancel.clone();
            async move {
                parent_cancel.cancelled().await;
                agent_handle.abort();
            }
        });

        let prompt_result = entry.agent.prompt(message).await;
        bridge.abort();
        event_task.abort();

        let current_usage = entry.agent.state().total_usage.clone();
        let delta_input = current_usage.input.saturating_sub(usage_before.input);
        let delta_output = current_usage.output.saturating_sub(usage_before.output);

        let msg_start = entry.messages_at_pause.min(entry.agent.messages().len());
        let tool_use_count = entry.agent.messages()[msg_start..]
            .iter()
            .map(|m| match m {
                Message::Assistant { content, .. } => content
                    .iter()
                    .filter(|c| matches!(c, Content::ToolCall { .. }))
                    .count(),
                _ => 0,
            })
            .sum::<usize>() as u32;

        let text = extract_final_text(entry.agent.messages());

        record_transcript(id, entry.agent.messages()).await;
        entry.usage_at_pause = current_usage;
        entry.messages_at_pause = entry.agent.messages().len();
        self.agents
            .lock()
            .await
            .push_back((id.to_string(), entry));

        prompt_result?;

        Ok(SubagentResult {
            agent_id: id.to_string(),
            text,
            input_tokens: delta_input,
            output_tokens: delta_output,
            tool_use_count,
            duration_ms: start.elapsed().as_millis() as u64,
            worktree_path: None,
            worktree_branch: None,
        })
    }

    #[cfg(test)]
    async fn list(&self) -> Vec<(String, String)> {
        self.agents
            .lock()
            .await
            .iter()
            .map(|(id, e)| (id.clone(), e.description.clone()))
            .collect()
    }

    /// Find an agent by name or ID. Checks running agents first, then stored.
    /// Name matching is case-insensitive substring on the description.
    pub async fn find_agent(&self, name_or_id: &str) -> Option<(String, String, AgentStatus)> {
        // Check running agents (exact ID, then fuzzy description)
        {
            let running = self.running_handles.lock().await;
            if let Some((_, desc)) = running.get(name_or_id) {
                return Some((name_or_id.to_string(), desc.clone(), AgentStatus::Running));
            }
            let needle = name_or_id.to_lowercase();
            for (id, (_, desc)) in running.iter() {
                if desc.to_lowercase().contains(&needle) {
                    return Some((id.clone(), desc.clone(), AgentStatus::Running));
                }
            }
        } // running lock dropped before acquiring agents lock

        // Check stored agents (exact ID, then fuzzy description)
        let agents = self.agents.lock().await;
        if let Some((id, e)) = agents.iter().find(|(id, _)| id == name_or_id) {
            return Some((id.clone(), e.description.clone(), AgentStatus::Idle));
        }
        let needle = name_or_id.to_lowercase();
        for (id, e) in agents.iter() {
            if e.description.to_lowercase().contains(&needle) {
                return Some((id.clone(), e.description.clone(), AgentStatus::Idle));
            }
        }
        None
    }

    /// Send a message to a currently running agent via its steering queue.
    /// Returns `true` if the agent was found and the message was delivered.
    pub async fn send_to_running(&self, id: &str, message: Message) -> bool {
        if let Some((handle, _)) = self.running_handles.lock().await.get(id) {
            handle.steer(message);
            true
        } else {
            false
        }
    }

    #[cfg(test)]
    async fn len(&self) -> usize {
        self.agents.lock().await.len()
    }

    /// Spawn a task that forwards events from a subagent's broadcast channel
    /// to the parent's event channel, wrapped in `AgentEvent::Subagent`.
    fn spawn_event_forwarder(
        &self,
        mut events: broadcast::Receiver<AgentEvent>,
        agent_id: &str,
        description: &str,
    ) -> tokio::task::JoinHandle<()> {
        let tx = self.parent_event_tx.clone();
        let agent_id = agent_id.to_string();
        let desc = description.to_string();
        tokio::spawn(async move {
            while let Ok(event) = events.recv().await {
                let _ = tx.send(AgentEvent::Subagent {
                    agent_id: agent_id.clone(),
                    description: desc.clone(),
                    event: Box::new(event),
                });
            }
        })
    }

    /// Run a subagent to completion and return its result plus the agent
    /// (for optional storage in a registry). Worktree cleanup runs even if the agent fails.
    /// Transcripts are recorded to `~/.local/share/tau/agent-transcripts/`.
    async fn run_subagent(
        &self,
        req: &SpawnRequest,
        cancel: CancellationToken,
        agent_id: &str,
    ) -> crate::error::Result<(SubagentResult, Agent)> {
        let start = std::time::Instant::now();

        let worktree = if req.isolation.as_deref() == Some("worktree") {
            match create_worktree(agent_id).await {
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

        let agent_result = self.run_agent_inner(req, agent_id, &worktree, cancel).await;
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

        record_transcript(agent_id, agent.messages()).await;
        result.worktree_path = wt_path;
        result.worktree_branch = wt_branch;
        result.duration_ms = start.elapsed().as_millis() as u64;
        Ok((result, agent))
    }

    /// Inner agent execution — separated so worktree cleanup can run regardless.
    async fn run_agent_inner(
        &self,
        req: &SpawnRequest,
        agent_id: &str,
        worktree: &Option<WorktreeInfo>,
        cancel: CancellationToken,
    ) -> crate::error::Result<(SubagentResult, Agent)> {
        let cwd = req
            .cwd
            .clone()
            .or_else(|| worktree.as_ref().map(|w| w.path.display().to_string()))
            .unwrap_or_else(|| {
                std::env::current_dir()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| ".".into())
            });

        let mut agent_cfg = self.parent_config.clone();
        agent_cfg.system_prompt = None;
        agent_cfg.max_turns = Some(req.agent_type.max_turns());
        if let Some(ref model) = req.model {
            agent_cfg.model = model.clone();
        }

        let mut agent = Agent::new(agent_cfg, self.transport.clone());
        agent.set_cwd(&cwd);

        let agent_handle = agent.handle();

        let factory = self.agent_tool_factory.lock().clone();
        let tools = build_tool_set(
            &req.agent_type,
            &self.tools,
            req.depth,
            &factory,
            &agent_handle,
        );
        for tool in tools {
            agent.add_tool(tool);
        }

        let tool_names = agent.tool_names();
        let prompt_opts = crate::prompts::PromptOptions {
            tool_names: &tool_names,
            cwd: &cwd,
            acolyte_mode: false,
        };
        let system_prompt = format!(
            "{}\n\n{}",
            crate::prompts::build_system_prompt(&prompt_opts),
            agent_type_suffix(&req.agent_type),
        );
        agent.set_system_prompt(system_prompt);

        let event_task = self.spawn_event_forwarder(
            agent.subscribe(),
            agent_id,
            &req.description,
        );

        // Wire the parent's cancel token to the subagent so that
        // cancelling the parent also cancels this subagent.
        let cancel_handle = agent_handle.clone();
        let parent_cancel = cancel.clone();
        let cancel_bridge = tokio::spawn(async move {
            parent_cancel.cancelled().await;
            cancel_handle.abort();
        });

        // Track this agent as running so SendMessage can reach it.
        // Use a struct guard to ensure cleanup even if the future is dropped.
        self.running_handles.lock().await.insert(
            agent_id.to_string(),
            (agent_handle, req.description.clone()),
        );

        let prompt_result = agent.prompt(&req.prompt).await;
        cancel_bridge.abort();
        event_task.abort();

        // Always remove from running handles, regardless of success/failure.
        self.running_handles.lock().await.remove(agent_id);

        prompt_result?;

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

    async fn store(&self, id: String, agent: Agent, description: String) {
        let mut agents = self.agents.lock().await;
        if agents.len() >= self.max_agents {
            agents.pop_front();
        }
        let usage = agent.state().total_usage.clone();
        let message_count = agent.messages().len();
        agents.push_back((
            id,
            ManagedAgent {
                agent,
                description,
                usage_at_pause: usage,
                messages_at_pause: message_count,
            },
        ));
    }
}

/// Record a subagent's conversation to disk for debugging.
/// Writes JSONL to `~/.local/share/tau/agent-transcripts/{agent_id}.jsonl`.
/// Overwrites any previous transcript for this agent (e.g. after resumption,
/// the new snapshot includes the full conversation).
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

fn build_tool_set(
    agent_type: &AgentType,
    all_tools: &[BoxedTool],
    depth: u32,
    agent_tool_factory: &Option<AgentToolFactory>,
    handle: &AgentHandle,
) -> Vec<BoxedTool> {
    let mut tools: Vec<BoxedTool> = match agent_type {
        AgentType::Explore | AgentType::Plan => {
            let read_only = ["read", "glob", "grep", "list", "lsp"];
            all_tools
                .iter()
                .filter(|t| read_only.contains(&t.name()))
                .cloned()
                .collect()
        }
        AgentType::GeneralPurpose => all_tools
            .iter()
            .filter(|t| t.name() != "agent")
            .cloned()
            .collect(),
    };

    if matches!(agent_type, AgentType::GeneralPurpose) && depth + 1 < MAX_AGENT_DEPTH {
        if let Some(factory) = agent_tool_factory {
            tools.push(factory(depth + 1, handle.clone()));
        }
    }

    tools
}

const AGENT_GENERAL_PROMPT: &str = include_str!("prompts/agent_general.md");
const AGENT_EXPLORE_PROMPT: &str = include_str!("prompts/agent_explore.md");
const AGENT_PLAN_PROMPT: &str = include_str!("prompts/agent_plan.md");

fn agent_type_suffix(agent_type: &AgentType) -> &'static str {
    match agent_type {
        AgentType::GeneralPurpose => AGENT_GENERAL_PROMPT,
        AgentType::Explore => AGENT_EXPLORE_PROMPT,
        AgentType::Plan => AGENT_PLAN_PROMPT,
    }
}

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

    let diff = tokio::process::Command::new("git")
        .args(["-C", &path_str, "diff", "--quiet", &info.head_commit])
        .status()
        .await
        .map_err(|e| format!("git diff failed: {}", e))?;

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

    /// Local alias for the (now-private) factory type, used in tests.
    type TestAgentToolFactory = Arc<dyn Fn(u32, AgentHandle) -> BoxedTool + Send + Sync>;

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
            _args: serde_json::Value,
            _ctx: crate::tool::ExecutionContext,
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

    #[test]
    fn test_parse_valid_types() {
        assert!(matches!(
            AgentType::parse("general-purpose"),
            Some(AgentType::GeneralPurpose)
        ));
        assert!(matches!(
            AgentType::parse("Explore"),
            Some(AgentType::Explore)
        ));
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

    fn test_handle() -> AgentHandle {
        AgentHandle::new()
    }

    #[test]
    fn test_explore_gets_read_only_tools() {
        let all = mock_tools();
        let handle = test_handle();
        let filtered = build_tool_set(&AgentType::Explore, &all, 0, &None, &handle);
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
        let handle = test_handle();
        let filtered = build_tool_set(&AgentType::Plan, &all, 0, &None, &handle);
        let names: Vec<&str> = filtered.iter().map(|t| t.name()).collect();
        assert_eq!(names.len(), 5);
        assert!(!names.contains(&"bash"));
        assert!(!names.contains(&"agent"));
    }

    #[test]
    fn test_general_purpose_gets_all_except_agent() {
        let all = mock_tools();
        let handle = test_handle();
        let filtered = build_tool_set(&AgentType::GeneralPurpose, &all, 0, &None, &handle);
        let names: Vec<&str> = filtered.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"bash"));
        assert!(names.contains(&"read"));
        assert!(names.contains(&"write"));
        assert!(!names.contains(&"agent")); // removed, no factory
    }

    #[test]
    fn test_general_purpose_gets_agent_tool_from_factory() {
        let all = mock_tools();
        let handle = test_handle();
        let factory: TestAgentToolFactory = Arc::new(|_depth, _handle| Arc::new(MockTool { tool_name: "agent" }));
        let filtered = build_tool_set(&AgentType::GeneralPurpose, &all, 0, &Some(factory), &handle);
        let names: Vec<&str> = filtered.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"agent")); // added by factory
    }

    #[test]
    fn test_depth_limit_excludes_agent_tool() {
        let all = mock_tools();
        let handle = test_handle();
        let factory: TestAgentToolFactory = Arc::new(|_depth, _handle| Arc::new(MockTool { tool_name: "agent" }));
        // At MAX_AGENT_DEPTH - 1, the next level would be MAX_AGENT_DEPTH -> excluded
        let filtered = build_tool_set(
            &AgentType::GeneralPurpose,
            &all,
            MAX_AGENT_DEPTH - 1,
            &Some(factory),
            &handle,
        );
        let names: Vec<&str> = filtered.iter().map(|t| t.name()).collect();
        assert!(!names.contains(&"agent")); // depth exceeded
    }

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
        let messages = vec![Message::Assistant {
            content: vec![
                Content::text("here is my answer"),
                Content::tool_call("c1", "bash", serde_json::json!({})),
            ],
            metadata: AssistantMetadata::default(),
        }];
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

    #[tokio::test]
    async fn test_event_forwarder_wraps_events() {
        use crate::events::AgentEvent;

        let manager = make_test_manager(20);

        // Create a child broadcast channel (simulates a subagent's event_tx)
        let (child_tx, _) = tokio::sync::broadcast::channel::<AgentEvent>(16);

        // Subscribe to the parent channel that the manager forwards to
        let mut parent_rx = manager.parent_event_tx.subscribe();

        // Spawn the forwarder
        let task = manager.spawn_event_forwarder(
            child_tx.subscribe(),
            "agent-123",
            "test task",
        );

        // Emit events on the child channel
        child_tx
            .send(AgentEvent::ToolExecutionStart {
                tool_call_id: "c1".into(),
                tool_name: "bash".into(),
                arguments: serde_json::json!({}),
            })
            .unwrap();
        child_tx
            .send(AgentEvent::ToolExecutionEnd {
                tool_call_id: "c1".into(),
                tool_name: "bash".into(),
                result: "ok".into(),
                is_error: false,
            })
            .unwrap();
        child_tx
            .send(AgentEvent::TurnEnd {
                turn_number: 1,
                message: tau_ai::Message::assistant_empty(),
                usage: tau_ai::Usage::default(),
            })
            .unwrap();

        drop(child_tx);
        let _ = task.await;

        // Verify events arrived wrapped in Subagent
        let mut received = vec![];
        while let Ok(event) = parent_rx.try_recv() {
            if let AgentEvent::Subagent {
                agent_id,
                description,
                event,
            } = event
            {
                assert_eq!(agent_id, "agent-123");
                assert_eq!(description, "test task");
                let label = match *event {
                    AgentEvent::ToolExecutionStart { ref tool_name, .. } => {
                        format!("[{}...]", tool_name)
                    }
                    AgentEvent::ToolExecutionEnd {
                        ref tool_name,
                        is_error,
                        ..
                    } => {
                        if is_error {
                            format!("[{} error]", tool_name)
                        } else {
                            format!("[{} done]", tool_name)
                        }
                    }
                    AgentEvent::TurnEnd { turn_number, .. } => {
                        format!("[turn {}]", turn_number)
                    }
                    _ => continue,
                };
                received.push(label);
            }
        }

        assert_eq!(received.len(), 3);
        assert_eq!(received[0], "[bash...]");
        assert_eq!(received[1], "[bash done]");
        assert_eq!(received[2], "[turn 1]");
    }

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

    fn make_test_config() -> AgentConfig {
        use crate::compaction::CompactionConfig;
        AgentConfig {
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
            max_turns: None,
            compaction: CompactionConfig::default(),
            steering_mode: crate::agent::DequeueMode::All,
            follow_up_mode: crate::agent::DequeueMode::All,
            cache_scope: None,
            cache_ttl: None,
            system_prompt_boundary: None,
        }
    }

    fn make_test_manager(max_agents: usize) -> Arc<AgentManager> {
        let (tx, _rx) = tokio::sync::broadcast::channel::<AgentEvent>(16);
        let transport: Arc<dyn crate::transport::Transport> = Arc::new(DummyTransport);
        let config = make_test_config();
        Arc::new(AgentManager::new(tx, vec![], config, transport, max_agents))
    }

    #[tokio::test]
    async fn test_manager_store_and_list() {
        let manager = make_test_manager(20);
        assert_eq!(manager.len().await, 0);

        let transport: Arc<dyn crate::transport::Transport> = Arc::new(DummyTransport);
        let config = make_test_config();
        let agent = Agent::new(config, transport);

        manager
            .store("abc123".into(), agent, "test agent".into())
            .await;
        assert_eq!(manager.len().await, 1);

        let list = manager.list().await;
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].0, "abc123");
        assert_eq!(list[0].1, "test agent");
    }

    #[tokio::test]
    async fn test_manager_send_nonexistent() {
        let manager = make_test_manager(20);
        let result = manager
            .send("nonexistent", "hello", CancellationToken::new())
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("No agent with ID"), "got: {}", err);
    }

    #[tokio::test]
    async fn test_manager_eviction() {
        let max = 20usize;
        let manager = make_test_manager(max);

        let make_agent = || {
            let transport: Arc<dyn crate::transport::Transport> = Arc::new(DummyTransport);
            let config = make_test_config();
            Agent::new(config, transport)
        };

        for i in 0..max {
            manager
                .store(format!("agent-{}", i), make_agent(), format!("desc-{}", i))
                .await;
        }
        assert_eq!(manager.len().await, max);

        manager
            .store("agent-overflow".into(), make_agent(), "overflow".into())
            .await;
        assert_eq!(manager.len().await, max);

        // The oldest (agent-0) should have been evicted, and the new one should be present
        let list = manager.list().await;
        assert!(
            !list.iter().any(|(id, _)| id == "agent-0"),
            "Expected oldest agent (agent-0) to be evicted"
        );
        assert!(list.iter().any(|(id, _)| id == "agent-overflow"));
    }
}
