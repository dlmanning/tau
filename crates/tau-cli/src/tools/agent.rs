//! Agent tool — spawn subagents for parallel work

use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tau_agent::handle::AgentHandle;
use tau_agent::agent_manager::{AgentManager, AgentType, SpawnRequest};
use tau_agent::tool::{ExecutionContext, Tool, ToolResult};

#[derive(Deserialize, JsonSchema)]
struct AgentArgs {
    /// Short (3-5 word) description of the task
    description: String,
    /// Detailed task instructions for the subagent
    prompt: String,
    /// Resume a previous agent by ID. Use with prompt to send a follow-up message.
    to: Option<String>,
    /// Type of agent. Explore/Plan are read-only.
    #[schemars(extend("enum" = ["general-purpose", "Explore", "Plan"]))]
    subagent_type: Option<String>,
    /// Override model for this subagent
    model: Option<String>,
    /// Working directory for the subagent
    cwd: Option<String>,
    /// Run in an isolated git worktree
    #[schemars(extend("enum" = ["worktree"]))]
    isolation: Option<String>,
    /// Run in background and return immediately
    run_in_background: Option<bool>,
}

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

    fn activity_description(&self, _arguments: &serde_json::Value) -> String {
        "Spawning agent".to_string()
    }

    fn description(&self) -> &str {
        "Spawn a subagent to handle a task independently, or send a message to a \
         previously spawned agent. The subagent makes its own API calls and has its \
         own tool set. Use for parallel work, codebase exploration, or isolating \
         changes in a git worktree."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(AgentArgs)
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let args: AgentArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };

        if let Some(ref agent_id) = args.to {
            return match self.manager.send(agent_id, &args.prompt, ctx.cancel).await {
                Ok(result) => ToolResult::text(format_result(&result)),
                Err(e) => ToolResult::error(format!("Resume failed: {}", e)),
            };
        }

        let agent_type = args
            .subagent_type
            .as_deref()
            .and_then(AgentType::parse)
            .unwrap_or(AgentType::GeneralPurpose);

        let model = args
            .model
            .as_deref()
            .and_then(tau_ai::models::get_model_by_id);

        let run_in_background = args.run_in_background.unwrap_or(false);

        let request = SpawnRequest {
            agent_type,
            prompt: args.prompt,
            description: args.description.clone(),
            model,
            cwd: args.cwd,
            isolation: args.isolation,
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
                    agent_id, args.description
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
