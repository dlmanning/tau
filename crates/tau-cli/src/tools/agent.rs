//! Agent tool — spawn subagents for parallel work

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tau_agent::agent::AgentConfig;
use tau_agent::handle::AgentHandle;
use tau_agent::subagent::{AgentToolFactory, AgentType, SubagentConfig, run_subagent};
use tau_agent::tool::{BoxedTool, Tool, ToolResult};
use tau_agent::transport::Transport;
use tokio_util::sync::CancellationToken;

/// Tool for spawning independent subagents.
pub struct AgentTool {
    transport: Arc<dyn Transport>,
    tools: Vec<BoxedTool>,
    parent_config: AgentConfig,
    depth: u32,
    agent_handle: Option<AgentHandle>,
}

impl AgentTool {
    pub fn new(
        transport: Arc<dyn Transport>,
        tools: Vec<BoxedTool>,
        parent_config: AgentConfig,
        depth: u32,
    ) -> Self {
        Self {
            transport,
            tools,
            parent_config,
            depth,
            agent_handle: None,
        }
    }

    pub fn with_handle(mut self, handle: AgentHandle) -> Self {
        self.agent_handle = Some(handle);
        self
    }

    /// Build the factory function for recursive subagent creation.
    /// Propagates the agent handle so nested background agents can notify.
    fn make_factory(&self) -> AgentToolFactory {
        let transport = self.transport.clone();
        let tools = self.tools.clone();
        let config = self.parent_config.clone();
        let handle = self.agent_handle.clone();
        Arc::new(move |depth| {
            let mut tool =
                AgentTool::new(transport.clone(), tools.clone(), config.clone(), depth);
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
        "Spawn a subagent to handle a task independently. The subagent makes its own \
         API calls and has its own tool set. Use for parallel work, codebase exploration, \
         or isolating changes in a git worktree."
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
            "required": ["description", "prompt"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        arguments: serde_json::Value,
        cancel: CancellationToken,
    ) -> ToolResult {
        let description = match arguments.get("description").and_then(|v| v.as_str()) {
            Some(d) => d.to_string(),
            None => return ToolResult::error("Missing 'description'"),
        };
        let prompt = match arguments.get("prompt").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolResult::error("Missing 'prompt'"),
        };

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
            };

            let handle = self.agent_handle.clone();
            let desc = description.clone();

            // Store cancel token so parent abort can propagate
            let parent_cancel = cancel.clone();
            tokio::spawn(async move {
                let result = tokio::select! {
                    r = run_subagent(config) => r,
                    _ = parent_cancel.cancelled() => {
                        Err(tau_agent::error::Error::Other("Cancelled by parent".into()))
                    }
                };

                if let Some(handle) = handle {
                    let msg = format_notification(&desc, &result);
                    handle.follow_up(tau_ai::Message::user(msg));
                }
            });

            return ToolResult::text(format!("Agent launched in background: {}", description));
        }

        // Foreground execution
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
        };

        match run_subagent(config).await {
            Ok(result) => {
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
                ToolResult::text(output)
            }
            Err(e) => ToolResult::error(format!("Agent failed: {}", e)),
        }
    }
}

fn format_notification(description: &str, result: &tau_agent::error::Result<tau_agent::subagent::SubagentResult>) -> String {
    match result {
        Ok(r) => format!(
            "<agent-completed description=\"{}\">\n{}\n\
             [Agent {} | {} in + {} out tokens | {} tool calls | {}ms]\n\
             </agent-completed>",
            description, r.text, r.agent_id,
            r.input_tokens, r.output_tokens,
            r.tool_use_count, r.duration_ms,
        ),
        Err(e) => format!(
            "<agent-failed description=\"{}\">\nError: {}\n</agent-failed>",
            description, e,
        ),
    }
}

