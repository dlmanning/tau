//! Subagent spawning — create independent agent instances for parallel work.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex as ParkingMutex;
use tau_ai::{Content, Message, Model, Usage};
use tokio::sync::{Mutex, broadcast};
use tokio_util::sync::CancellationToken;

use crate::builder::AgentBuilder;
use crate::config::AgentConfig;
use crate::events::AgentEvent;
use crate::handle::AgentHandle;
use crate::tool::BoxedTool;
use crate::transcript::record_transcript;
use crate::transport::Transport;
use crate::worktree::{WorktreeInfo, cleanup_worktree, create_worktree};

/// Maximum recursion depth for nested agent spawning.
pub const MAX_AGENT_DEPTH: u32 = 3;

/// Maximum turns a subagent can execute before being stopped.
pub const MAX_SUBAGENT_TURNS: u32 = 200;

/// Agent type determines tool set and system prompt.
#[derive(Debug, Clone)]
pub enum AgentType {
    GeneralPurpose,
    Explore,
    Plan,
}

impl AgentType {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "general-purpose" => Some(Self::GeneralPurpose),
            "explore" => Some(Self::Explore),
            "plan" => Some(Self::Plan),
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
            Self::Explore => MAX_SUBAGENT_TURNS,
            Self::Plan => MAX_SUBAGENT_TURNS,
        }
    }
}

impl std::fmt::Display for AgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GeneralPurpose => write!(f, "general-purpose"),
            Self::Explore => write!(f, "explore"),
            Self::Plan => write!(f, "plan"),
        }
    }
}

/// Factory function to create a depth-limited Agent tool.
type AgentToolFactory = Arc<dyn Fn(u32, AgentHandle) -> BoxedTool + Send + Sync>;

/// Result from a completed subagent.
#[derive(Debug)]
pub struct SubagentResult {
    pub agent_id: String,
    pub text: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
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
    agents: Mutex<VecDeque<(String, ManagedAgent)>>,
    max_agents: usize,
    transport: Arc<dyn Transport>,
    tools: Vec<BoxedTool>,
    parent_config: AgentConfig,
    parent_event_tx: broadcast::Sender<AgentEvent>,
    agent_tool_factory: ParkingMutex<Option<AgentToolFactory>>,
    /// Handles for agents that are currently executing (keyed by agent_id).
    running_handles: Mutex<HashMap<String, (AgentHandle, String)>>,
}

struct ManagedAgent {
    handle: AgentHandle,
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
            agents: Mutex::new(VecDeque::new()),
            max_agents,
            transport,
            tools,
            parent_config,
            parent_event_tx,
            agent_tool_factory: ParkingMutex::new(None),
            running_handles: Mutex::new(HashMap::new()),
        }
    }

    /// Set the factory for creating recursive agent tools.
    pub fn set_agent_tool_factory(
        &self,
        factory: Arc<dyn Fn(u32, AgentHandle) -> BoxedTool + Send + Sync>,
    ) {
        *self.agent_tool_factory.lock() = Some(factory);
    }

    /// Spawn a foreground subagent. Blocks until completion.
    pub async fn spawn(
        self: &Arc<Self>,
        request: SpawnRequest,
        cancel: CancellationToken,
    ) -> crate::error::Result<SubagentResult> {
        let agent_id = uuid::Uuid::new_v4().to_string();
        let description = request.description.clone();
        let result = self.run_subagent(&request, cancel, &agent_id).await;
        self.running_handles.lock().await.remove(&agent_id);
        let (result, handle) = result?;
        self.store(result.agent_id.clone(), handle, description)
            .await;
        Ok(result)
    }

    /// Spawn a background subagent. Returns immediately with the agent_id.
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

            manager.running_handles.lock().await.remove(&aid);

            match result {
                Ok((subresult, handle)) => {
                    manager
                        .store(subresult.agent_id.clone(), handle, desc.clone())
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
    pub async fn send(
        &self,
        id: &str,
        message: &str,
        parent_cancel: CancellationToken,
    ) -> crate::error::Result<SubagentResult> {
        let start = Instant::now();

        let mut entry = {
            let mut agents = self.agents.lock().await;
            let pos = agents.iter().position(|(k, _)| k == id).ok_or_else(|| {
                crate::error::Error::Other(format!(
                    "No agent with ID '{}'. It may have been evicted, never stored, or is currently running.",
                    id
                ))
            })?;
            agents.remove(pos).expect("pos was found via position()").1
        };

        let usage_before = entry.usage_at_pause.clone();

        let event_task =
            self.spawn_event_forwarder(entry.handle.subscribe(), id, &entry.description);

        // Wire parent cancel to subagent abort
        let agent_handle = entry.handle.clone();
        let bridge = tokio::spawn({
            let parent_cancel = parent_cancel.clone();
            async move {
                parent_cancel.cancelled().await;
                agent_handle.abort();
            }
        });

        let prompt_result = entry.handle.prompt_and_wait(message).await;
        bridge.abort();
        event_task.abort();

        let current_state = entry.handle.state().await.unwrap_or_default();
        let current_usage = current_state.total_usage.clone();
        let delta_input = current_usage.input.saturating_sub(usage_before.input);
        let delta_output = current_usage.output.saturating_sub(usage_before.output);

        let messages = entry.handle.messages().await.unwrap_or_default();
        let msg_start = entry.messages_at_pause.min(messages.len());
        let tool_use_count = messages[msg_start..]
            .iter()
            .map(|m| match m {
                Message::Assistant { content, .. } => content
                    .iter()
                    .filter(|c| matches!(c, Content::ToolCall { .. }))
                    .count(),
                _ => 0,
            })
            .sum::<usize>() as u32;

        let text = extract_final_text(&messages);

        record_transcript(id, &messages).await;
        entry.usage_at_pause = current_usage;
        entry.messages_at_pause = messages.len();
        self.agents.lock().await.push_back((id.to_string(), entry));

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

    /// Find an agent by name or ID.
    pub async fn find_agent(&self, name_or_id: &str) -> Option<(String, String, AgentStatus)> {
        // Check running agents first
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
        }

        // Check stored agents
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

    /// Send a message to a currently running agent via steering.
    pub async fn send_to_running(&self, id: &str, message: Message) -> bool {
        if let Some((handle, _)) = self.running_handles.lock().await.get(id) {
            handle.steer(message);
            true
        } else {
            false
        }
    }

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

    async fn run_subagent(
        &self,
        req: &SpawnRequest,
        cancel: CancellationToken,
        agent_id: &str,
    ) -> crate::error::Result<(SubagentResult, AgentHandle)> {
        let start = Instant::now();

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
                _ => (Some(wt.path.display().to_string()), Some(wt.branch.clone())),
            }
        } else {
            (None, None)
        };

        let (mut result, handle) = agent_result?;

        let messages = handle.messages().await.unwrap_or_default();
        record_transcript(agent_id, &messages).await;
        result.worktree_path = wt_path;
        result.worktree_branch = wt_branch;
        result.duration_ms = start.elapsed().as_millis() as u64;
        Ok((result, handle))
    }

    async fn run_agent_inner(
        &self,
        req: &SpawnRequest,
        agent_id: &str,
        worktree: &Option<WorktreeInfo>,
        cancel: CancellationToken,
    ) -> crate::error::Result<(SubagentResult, AgentHandle)> {
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
        agent_cfg.max_tokens = None;
        // Disable prompt caching for subagents — they're short-lived so
        // cache writes are wasted (the cache is never read back).
        agent_cfg.cache_scope = None;
        agent_cfg.cache_ttl = None;
        if let Some(ref model) = req.model {
            agent_cfg.model = model.clone();
        }

        let mut builder = AgentBuilder::new(agent_cfg, self.transport.clone());
        builder.set_cwd(&cwd);

        let factory = self.agent_tool_factory.lock().clone();
        // pre_handle() gives us a working handle connected to the same channel
        // that spawn() will use — resolves the circular dependency.
        let agent_handle = builder.pre_handle();

        let tools = build_tool_set(
            &req.agent_type,
            &self.tools,
            req.depth,
            &factory,
            &agent_handle,
        );
        for tool in &tools {
            builder.add_tool(tool.clone());
        }

        let tool_names = builder.tool_names();
        let system_prompt = crate::prompts::build_subagent_prompt(
            agent_type_suffix(&req.agent_type),
            &tool_names,
            &cwd,
        );
        builder.set_system_prompt(system_prompt);

        let handle = builder.spawn();

        let event_task = self.spawn_event_forwarder(handle.subscribe(), agent_id, &req.description);

        // Wire cancel bridge
        let cancel_handle = handle.clone();
        let parent_cancel = cancel.clone();
        let cancel_bridge = tokio::spawn(async move {
            parent_cancel.cancelled().await;
            cancel_handle.abort();
        });

        // Track as running
        self.running_handles.lock().await.insert(
            agent_id.to_string(),
            (handle.clone(), req.description.clone()),
        );

        let prompt_result = handle.prompt_and_wait(&req.prompt).await;
        cancel_bridge.abort();
        event_task.abort();

        self.running_handles.lock().await.remove(agent_id);

        prompt_result?;

        let messages = handle.messages().await.unwrap_or_default();
        let text = extract_final_text(&messages);
        let state = handle.state().await.unwrap_or_default();

        let tool_use_count = messages
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
            duration_ms: 0,
            worktree_path: None,
            worktree_branch: None,
        };
        Ok((result, handle))
    }

    async fn store(&self, id: String, handle: AgentHandle, description: String) {
        let mut agents = self.agents.lock().await;
        if agents.len() >= self.max_agents {
            agents.pop_front();
        }
        let state = handle.state().await.unwrap_or_default();
        let usage = state.total_usage.clone();
        let message_count = state.messages.len();
        agents.push_back((
            id,
            ManagedAgent {
                handle,
                description,
                usage_at_pause: usage,
                messages_at_pause: message_count,
            },
        ));
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
                if text.is_empty() { None } else { Some(text) }
            }
            _ => None,
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DequeueMode;
    use crate::tool::Tool;
    use async_trait::async_trait;
    use tau_ai::{AssistantMetadata, Content, Message};

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
            AgentType::parse("explore"),
            Some(AgentType::Explore)
        ));
        assert!(matches!(AgentType::parse("plan"), Some(AgentType::Plan)));
    }

    #[test]
    fn test_parse_case_insensitive() {
        assert!(matches!(
            AgentType::parse("Explore"),
            Some(AgentType::Explore)
        ));
        assert!(matches!(AgentType::parse("Plan"), Some(AgentType::Plan)));
        assert!(matches!(
            AgentType::parse("EXPLORE"),
            Some(AgentType::Explore)
        ));
        assert!(matches!(
            AgentType::parse("General-Purpose"),
            Some(AgentType::GeneralPurpose)
        ));
    }

    #[test]
    fn test_parse_invalid_type() {
        assert!(AgentType::parse("unknown").is_none());
        assert!(AgentType::parse("").is_none());
    }

    #[test]
    fn test_display() {
        assert_eq!(AgentType::GeneralPurpose.to_string(), "general-purpose");
        assert_eq!(AgentType::Explore.to_string(), "explore");
        assert_eq!(AgentType::Plan.to_string(), "plan");
    }

    #[test]
    fn test_max_turns() {
        assert_eq!(AgentType::GeneralPurpose.max_turns(), 200);
        assert_eq!(AgentType::Explore.max_turns(), 200);
        assert_eq!(AgentType::Plan.max_turns(), 200);
    }

    fn test_handle() -> AgentHandle {
        let config = make_test_config();
        let transport: Arc<dyn crate::transport::Transport> = Arc::new(DummyTransport);
        let builder = crate::builder::AgentBuilder::new(config, transport);
        builder.pre_handle()
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
    fn test_general_purpose_gets_all_except_agent() {
        let all = mock_tools();
        let handle = test_handle();
        let filtered = build_tool_set(&AgentType::GeneralPurpose, &all, 0, &None, &handle);
        let names: Vec<&str> = filtered.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"bash"));
        assert!(names.contains(&"read"));
        assert!(names.contains(&"write"));
        assert!(!names.contains(&"agent"));
    }

    #[test]
    fn test_general_purpose_gets_agent_tool_from_factory() {
        let all = mock_tools();
        let handle = test_handle();
        let factory: AgentToolFactory =
            Arc::new(|_depth, _handle| Arc::new(MockTool { tool_name: "agent" }));
        let filtered = build_tool_set(&AgentType::GeneralPurpose, &all, 0, &Some(factory), &handle);
        let names: Vec<&str> = filtered.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"agent"), "factory should add agent tool");
    }

    #[test]
    fn test_depth_limit_excludes_agent_tool() {
        let all = mock_tools();
        let handle = test_handle();
        let factory: AgentToolFactory =
            Arc::new(|_depth, _handle| Arc::new(MockTool { tool_name: "agent" }));
        let filtered = build_tool_set(
            &AgentType::GeneralPurpose,
            &all,
            MAX_AGENT_DEPTH - 1,
            &Some(factory),
            &handle,
        );
        let names: Vec<&str> = filtered.iter().map(|t| t.name()).collect();
        assert!(!names.contains(&"agent"), "depth exceeded — no agent tool");
    }

    #[test]
    fn test_extract_final_text() {
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
    fn test_extract_final_text_empty() {
        let messages: Vec<Message> = vec![];
        assert_eq!(extract_final_text(&messages), "");
    }

    #[test]
    fn test_suffixes_are_nonempty() {
        assert!(!agent_type_suffix(&AgentType::GeneralPurpose).is_empty());
        assert!(!agent_type_suffix(&AgentType::Explore).is_empty());
        assert!(!agent_type_suffix(&AgentType::Plan).is_empty());
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
            compaction: crate::compaction::CompactionConfig::default(),
            steering_mode: DequeueMode::All,
            follow_up_mode: DequeueMode::All,
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
    async fn test_event_forwarder_wraps_events() {
        let manager = make_test_manager(20);

        let (child_tx, _) = tokio::sync::broadcast::channel::<AgentEvent>(16);
        let mut parent_rx = manager.parent_event_tx.subscribe();

        let task = manager.spawn_event_forwarder(child_tx.subscribe(), "agent-123", "test task");

        child_tx
            .send(AgentEvent::ToolExecutionStart {
                tool_call_id: "c1".into(),
                tool_name: "bash".into(),
                arguments: serde_json::json!({}),
                activity: "Running bash".into(),
            })
            .unwrap();

        drop(child_tx);
        let _ = task.await;

        let mut received = vec![];
        while let Ok(event) = parent_rx.try_recv() {
            if let AgentEvent::Subagent {
                agent_id,
                description,
                ..
            } = event
            {
                assert_eq!(agent_id, "agent-123");
                assert_eq!(description, "test task");
                received.push(true);
            }
        }

        assert!(!received.is_empty(), "should have forwarded events");
    }
}
