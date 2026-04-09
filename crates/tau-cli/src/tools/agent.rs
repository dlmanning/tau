//! Agent tool — spawn subagents for parallel work

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tau_agent::agent::AgentConfig;
use tau_agent::handle::AgentHandle;
use tau_agent::subagent::{
    AgentRegistry, AgentToolFactory, AgentType, ProgressCallback, SubagentConfig, run_subagent,
};
use tau_agent::tool::{BoxedTool, ProgressSender, Tool, ToolResult};
use tau_agent::transport::Transport;
use tokio_util::sync::CancellationToken;

/// Tool for spawning independent subagents.
pub struct AgentTool {
    transport: Arc<dyn Transport>,
    tools: Vec<BoxedTool>,
    parent_config: AgentConfig,
    depth: u32,
    agent_handle: Option<AgentHandle>,
    registry: Arc<AgentRegistry>,
}

impl AgentTool {
    pub fn new(
        transport: Arc<dyn Transport>,
        tools: Vec<BoxedTool>,
        parent_config: AgentConfig,
        depth: u32,
        registry: Arc<AgentRegistry>,
    ) -> Self {
        Self {
            transport,
            tools,
            parent_config,
            depth,
            agent_handle: None,
            registry,
        }
    }

    pub fn with_handle(mut self, handle: AgentHandle) -> Self {
        self.agent_handle = Some(handle);
        self
    }

    fn make_factory(&self) -> AgentToolFactory {
        let transport = self.transport.clone();
        let tools = self.tools.clone();
        let config = self.parent_config.clone();
        let handle = self.agent_handle.clone();
        let registry = self.registry.clone();
        Arc::new(move |depth| {
            let mut tool = AgentTool::new(
                transport.clone(),
                tools.clone(),
                config.clone(),
                depth,
                registry.clone(),
            );
            tool.agent_handle = handle.clone();
            Arc::new(tool)
        })
    }
}

#[async_trait]
impl Tool for AgentTool {
    fn name(&self) -> &str {
        "agent"
    }

    fn description(&self) -> &str {
        "Spawn a subagent to handle a task independently, or send a message to a \
         previously spawned agent. The subagent makes its own API calls and has its \
         own tool set. Use for parallel work, codebase exploration, or isolating \
         changes in a git worktree."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short (3-5 word) description of the task"
                },
                "prompt": {
                    "type": "string",
                    "description": "Detailed task instructions for the subagent"
                },
                "to": {
                    "type": "string",
                    "description": "Resume a previous agent by ID. Use with prompt to send a follow-up message."
                },
                "subagent_type": {
                    "type": "string",
                    "enum": ["general-purpose", "Explore", "Plan"],
                    "description": "Type of agent. Explore/Plan are read-only."
                },
                "model": {
                    "type": "string",
                    "description": "Override model for this subagent"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the subagent"
                },
                "isolation": {
                    "type": "string",
                    "enum": ["worktree"],
                    "description": "Run in an isolated git worktree"
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Run in background and return immediately"
                }
            },
            "required": ["prompt"]
        })
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        arguments: serde_json::Value,
        cancel: CancellationToken,
    ) -> ToolResult {
        let (tx, _rx) = tokio::sync::broadcast::channel(1);
        let progress = ProgressSender::new(tx, tool_call_id, self.name());
        self.execute_with_progress(tool_call_id, arguments, cancel, progress)
            .await
    }

    async fn execute_with_progress(
        &self,
        _tool_call_id: &str,
        arguments: serde_json::Value,
        cancel: CancellationToken,
        progress: ProgressSender,
    ) -> ToolResult {
        let prompt = match arguments.get("prompt").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolResult::error("Missing 'prompt'"),
        };

        // Resume existing agent
        if let Some(agent_id) = arguments.get("to").and_then(|v| v.as_str()) {
            let progress_cb: ProgressCallback = Arc::new(move |msg: &str| {
                progress.send(msg);
            });
            return match self.registry.resume(agent_id, &prompt, Some(&progress_cb)).await {
                Ok(result) => ToolResult::text(format_result(&result)),
                Err(e) => ToolResult::error(format!("Resume failed: {}", e)),
            };
        }

        // New agent spawn
        let description = arguments
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("subagent")
            .to_string();

        let agent_type = arguments
            .get("subagent_type")
            .and_then(|v| v.as_str())
            .and_then(AgentType::parse)
            .unwrap_or(AgentType::GeneralPurpose);

        let model = arguments
            .get("model")
            .and_then(|v| v.as_str())
            .and_then(|id| tau_ai::models::get_model_by_id(id));

        let cwd = arguments
            .get("cwd")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let isolation = arguments
            .get("isolation")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let run_in_background = arguments
            .get("run_in_background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if run_in_background {
            let bg_cancel = CancellationToken::new();
            let config = SubagentConfig {
                agent_type,
                prompt,
                description: description.clone(),
                model,
                cwd,
                isolation,
                depth: self.depth,
                transport: self.transport.clone(),
                all_tools: self.tools.clone(),
                parent_config: self.parent_config.clone(),
                agent_tool_factory: Some(self.make_factory()),
                cancel: bg_cancel.clone(),
                on_progress: None,
            };

            let handle = self.agent_handle.clone();
            let desc = description.clone();
            let registry = self.registry.clone();
            let parent_cancel = cancel.clone();

            tokio::spawn(async move {
                let result = tokio::select! {
                    r = run_subagent(config) => r,
                    _ = parent_cancel.cancelled() => {
                        Err(tau_agent::error::Error::Other("Cancelled by parent".into()))
                    }
                };

                match result {
                    Ok((subresult, agent)) => {
                        if let Some(handle) = handle {
                            handle.follow_up(tau_ai::Message::user(format!(
                                "<agent-completed description=\"{}\">\n{}\n\
                                 [Agent {} | {} in + {} out tokens | {} tool calls | {}ms]\n\
                                 </agent-completed>",
                                desc, subresult.text, subresult.agent_id,
                                subresult.input_tokens, subresult.output_tokens,
                                subresult.tool_use_count, subresult.duration_ms,
                            )));
                        }
                        registry.store(subresult.agent_id, agent, desc).await;
                    }
                    Err(e) => {
                        if let Some(handle) = handle {
                            handle.follow_up(tau_ai::Message::user(format!(
                                "<agent-failed description=\"{}\">\nError: {}\n</agent-failed>",
                                desc, e,
                            )));
                        }
                    }
                }
            });

            return ToolResult::text(format!("Agent launched in background: {}", description));
        }

        // Foreground execution
        let progress_cb: ProgressCallback = Arc::new(move |msg: &str| {
            progress.send(msg);
        });
        let config = SubagentConfig {
            agent_type,
            prompt,
            description: description.clone(),
            model,
            cwd,
            isolation,
            depth: self.depth,
            transport: self.transport.clone(),
            all_tools: self.tools.clone(),
            parent_config: self.parent_config.clone(),
            agent_tool_factory: Some(self.make_factory()),
            cancel,
            on_progress: Some(progress_cb),
        };

        match run_subagent(config).await {
            Ok((result, agent)) => {
                // Store for potential resumption
                let agent_id = result.agent_id.clone();
                let desc = description.clone();
                self.registry.store(agent_id, agent, desc).await;

                ToolResult::text(format_result(&result))
            }
            Err(e) => ToolResult::error(format!("Agent failed: {}", e)),
        }
    }
}

fn format_result(result: &tau_agent::subagent::SubagentResult) -> String {
    let mut output = result.text.clone();
    let mut meta = format!(
        "\n[Agent {} | {} in + {} out tokens | {} tool calls | {}ms",
        result.agent_id, result.input_tokens, result.output_tokens,
        result.tool_use_count, result.duration_ms,
    );
    if let Some(ref p) = result.worktree_path {
        meta.push_str(&format!(" | worktree: {}", p));
    }
    if let Some(ref b) = result.worktree_branch {
        meta.push_str(&format!(" | branch: {}", b));
    }
    meta.push(']');
    output.push_str(&meta);
    output
}
