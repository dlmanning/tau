//! Agent tool — spawn subagents for parallel work

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tau_agent::handle::AgentHandle;
use tau_agent::agent_manager::{AgentManager, AgentType, SpawnRequest};
use tau_agent::tool::{Concurrency, ExecutionContext, Tool, ToolResult};

/// Tool for spawning independent subagents.
pub struct AgentTool {
    manager: Arc<AgentManager>,
    depth: u32,
    agent_handle: Option<AgentHandle>,
}

impl AgentTool {
    pub fn new(manager: Arc<AgentManager>, depth: u32) -> Self {
        Self {
            manager,
            depth,
            agent_handle: None,
        }
    }

    pub fn with_handle(mut self, handle: AgentHandle) -> Self {
        self.agent_handle = Some(handle);
        self
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

    fn concurrency(&self) -> Concurrency {
        Concurrency::Parallel
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
            "required": ["description", "prompt"]
        })
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let prompt = match arguments.get("prompt").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => return ToolResult::error("Missing 'prompt'"),
        };

        if let Some(agent_id) = arguments.get("to").and_then(|v| v.as_str()) {
            return match self.manager.send(agent_id, &prompt, ctx.cancel).await {
                Ok(result) => ToolResult::text(format_result(&result)),
                Err(e) => ToolResult::error(format!("Resume failed: {}", e)),
            };
        }

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
            .and_then(tau_ai::models::get_model_by_id);

        let cwd = arguments
            .get("cwd")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let isolation = arguments
            .get("isolation")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let run_in_background = arguments
            .get("run_in_background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let request = SpawnRequest {
            agent_type,
            prompt,
            description: description.clone(),
            model,
            cwd,
            isolation,
            depth: self.depth,
        };

        if run_in_background {
            if let Some(ref handle) = self.agent_handle {
                let agent_id = self
                    .manager
                    .spawn_background(request, handle.clone(), ctx.cancel)
                    .await;
                ToolResult::text(format!(
                    "Agent launched in background ({}): {}",
                    agent_id, description
                ))
            } else {
                ToolResult::error("Cannot run background agent: no parent handle")
            }
        } else {
            match self.manager.spawn(request, ctx.cancel).await {
                Ok(result) => ToolResult::text(format_result(&result)),
                Err(e) => ToolResult::error(format!("Agent failed: {}", e)),
            }
        }
    }
}

fn format_result(result: &tau_agent::agent_manager::SubagentResult) -> String {
    let mut output = result.text.clone();
    let mut meta = format!(
        "\n[Agent {} | {} in + {} out tokens | {} tool calls | {}ms",
        result.agent_id, result.input_tokens, result.output_tokens, result.tool_use_count,
        result.duration_ms,
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
