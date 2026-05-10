//! Agent tool — spawn subagents for parallel work
use crate::cached_schema;

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tau_agent::approval::ApprovalPolicy;
use tau_agent::handle::AgentHandle;
use tau_agent::manager::{AgentManager, AgentSpec, Isolation, SpawnOpts};
use tau_agent::tool::{ExecutionContext, Tool, ToolResult};

/// Resolves a host-defined spec name (and the depth at which it's being
/// resolved) to a fully-constructed [`AgentSpec`]. Hosts pass this to
/// [`AgentTool::with_spec_resolver`] so the runtime stays ignorant of
/// which spec names are valid — the LLM learns the names from the system
/// prompt; the resolver validates them. The `depth` argument lets hosts
/// gate recursive spawning (e.g. don't include another `AgentTool` in
/// the returned spec when `depth + 1` would exceed the host's limit).
pub type SpecResolver = Arc<dyn Fn(&str, u32) -> Option<AgentSpec> + Send + Sync>;

#[derive(Deserialize, JsonSchema)]
struct AgentArgs {
    /// Short (3-5 word) description of the task
    description: String,
    /// Detailed task instructions for the subagent
    prompt: String,
    /// Resume a previous agent by ID. Use with prompt to send a follow-up message.
    to: Option<String>,
    /// Type of agent. Host-defined; see the system prompt for the valid set.
    subagent_type: String,
    /// Override model for this subagent
    model: Option<String>,
    /// Working directory for the subagent
    cwd: Option<String>,
    /// Run in an isolated git worktree
    isolation: Option<Isolation>,
    /// Run in background and return immediately
    run_in_background: Option<bool>,
    /// Seed the new subagent with another stored agent's full message
    /// history, then send `prompt` as a follow-up user message. Use this
    /// for plan → execute handoffs: pass the plan subagent's id here so
    /// the executor inherits the planner's investigation and the approved
    /// plan as its own conversation history.
    inherit_history_from: Option<String>,
}

/// Tool for spawning independent subagents.
pub struct AgentTool {
    manager: Arc<AgentManager>,
    /// Depth this tool sits at in the recursive spawn tree. `0` for a
    /// root agent's own AgentTool; the host's resolver bumps it by 1
    /// when constructing each child's spec.
    depth: u32,
    /// Handle of the agent that *owns* this tool, captured via
    /// [`Tool::bind_to_agent`] when the runtime spawns its agent.
    /// `with_handle` pre-populates it for the root agent that's built
    /// directly via `AgentBuilder` (and therefore bypasses
    /// `bind_to_agent`).
    ///
    /// `Arc<OnceLock<...>>` so the value can be filled in *after*
    /// construction (the new agent's handle doesn't exist yet when the
    /// resolver builds the spec) while still being shared across all
    /// clones of this tool Arc.
    agent_handle: Arc<OnceLock<AgentHandle>>,
    /// If set, only these spec names are allowed. None means no restriction.
    allowed_specs: Option<Vec<String>>,
    /// Resolves a `subagent_type` string to a fully-built [`AgentSpec`].
    /// `None` means this tool can't actually spawn — `execute()` will
    /// error. Hosts must install one for the tool to be functional.
    spec_resolver: Option<SpecResolver>,
    /// Effective approval policy for the agent that *owns* this tool. When
    /// this agent spawns a descendant, the spawn opts inherit this policy
    /// so "applies at that level and below" holds. `None` lets the manager
    /// use its own default.
    inherited_policy: Option<Arc<dyn ApprovalPolicy>>,
}

impl AgentTool {
    pub fn new(manager: Arc<AgentManager>, depth: u32) -> Self {
        Self {
            manager,
            depth,
            agent_handle: Arc::new(OnceLock::new()),
            allowed_specs: None,
            spec_resolver: None,
            inherited_policy: None,
        }
    }

    /// Pre-bind to a specific handle. Use for the root agent — built
    /// outside the manager's spawn flow, so it bypasses
    /// [`Tool::bind_to_agent`]. For subagent tools assembled by a
    /// resolver, leave this unset and let the manager's spawn flow bind
    /// it via the trait hook.
    pub fn with_handle(self, handle: AgentHandle) -> Self {
        let _ = self.agent_handle.set(handle);
        self
    }

    pub fn with_allowed_specs(mut self, names: Vec<String>) -> Self {
        self.allowed_specs = Some(names);
        self
    }

    pub fn with_spec_resolver(mut self, resolver: SpecResolver) -> Self {
        self.spec_resolver = Some(resolver);
        self
    }

    /// Set the effective approval policy this tool's owner is running
    /// under. Descendants spawned via this tool inherit it.
    pub fn with_inherited_policy(mut self, policy: Arc<dyn ApprovalPolicy>) -> Self {
        self.inherited_policy = Some(policy);
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
        "Spawn a subagent (or follow up with a previously spawned one). This is \
         also the only mechanism for executing an approved plan: there is no \
         separate plan-execution tool. Plans are executed by spawning an \
         executor subagent that inherits the planner's history.\n\n\
         To execute an approved plan: take the `agent_id` of the planner \
         subagent that produced it, spawn an executor subagent with \
         `inherit_history_from: <plan_agent_id>` and a prompt such as \
         \"execute the approved plan.\" The executor sees the planner's \
         investigation, the approved plan, and the user's intent as its own \
         conversation history.\n\n\
         Use cases beyond plan execution: parallel work that benefits from \
         isolated contexts, codebase exploration that would otherwise pollute \
         the parent's context, or running edits in a git worktree."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(AgentArgs)
    }

    fn bind_to_agent(&self, handle: &AgentHandle) {
        // First-write-wins; ignore if already pre-bound via `with_handle`
        // (the root case) or already bound for this agent.
        let _ = self.agent_handle.set(handle.clone());
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

        if let Some(ref allowed) = self.allowed_specs
            && !allowed.contains(&args.subagent_type)
        {
            return ToolResult::error(format!(
                "subagent_type '{}' not allowed here. Allowed: {}.",
                args.subagent_type,
                allowed.join(", ")
            ));
        }

        let resolver = match self.spec_resolver.as_ref() {
            Some(r) => r,
            None => {
                return ToolResult::error(
                    "AgentTool has no spec resolver installed; cannot spawn subagents",
                );
            }
        };
        let spec = match resolver(&args.subagent_type, self.depth) {
            Some(s) => s,
            None => {
                return ToolResult::error(format!(
                    "Unknown subagent_type '{}' (or recursion depth limit reached)",
                    args.subagent_type
                ));
            }
        };

        let model = args
            .model
            .as_deref()
            .and_then(tau_ai::models::get_model_by_id);

        let run_in_background = args.run_in_background.unwrap_or(false);

        let opts = SpawnOpts {
            description: args.description.clone(),
            model,
            cwd: args.cwd,
            isolation: args.isolation,
            inherit_history_from: args.inherit_history_from,
            // Propagate the parent's effective policy so a per-spawn
            // override at a higher level reaches deeper subagents.
            approval_policy: self.inherited_policy.clone(),
            spec_name: Some(args.subagent_type.clone()),
            seed_messages: None,
        };

        if run_in_background {
            if let Some(handle) = self.agent_handle.get() {
                let agent_id = self
                    .manager
                    .spawn_background(spec, args.prompt, opts, handle.clone(), ctx.cancel)
                    .await;
                ToolResult::text(format!(
                    "Agent launched in background ({}): {}",
                    agent_id, args.description
                ))
            } else {
                ToolResult::error("Cannot run background agent: no parent handle")
            }
        } else {
            match self.manager.spawn(spec, args.prompt, opts, ctx.cancel).await {
                Ok(result) => ToolResult::text(format_result(&result)),
                Err(e) => ToolResult::error(format!("Agent failed: {}", e)),
            }
        }
    }
}

fn format_result(result: &tau_agent::manager::SubagentResult) -> String {
    let mut output = result.text.clone();
    let mut meta = format!(
        "\n[Agent {} | {} in + {} out tokens | {} tool calls | {}ms",
        result.agent_id,
        result.input_tokens,
        result.output_tokens,
        result.tool_use_count,
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
