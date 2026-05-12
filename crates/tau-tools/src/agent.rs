//! Agent tool — spawn subagents for parallel work
use crate::cached_schema;

use std::sync::{Arc, Weak};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tau_agent::ApprovalPolicy;
use tau_agent::{AgentManager, AgentSpec, Isolation, SpawnOpts};
use tau_agent::{ExecutionContext, Tool, ToolResult};

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
///
/// AgentTool holds [`Weak<AgentManager>`] rather than a strong reference
/// to break the cycle `manager → agent_specs → AgentSpec → tools →
/// AgentTool → manager`. The owning agent's id is **not** captured on
/// this struct — it's read from [`ExecutionContext::agent_id`] at
/// invocation time, which means the tool carries no per-agent state and
/// can be safely shared across spawns without binding hooks.
pub struct AgentTool {
    manager: Weak<AgentManager>,
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
    /// Whether the owner is allowed to spawn with `run_in_background:
    /// true`. Planners should not — background spawns are executor
    /// territory. Defaults to `true`.
    allow_background: bool,
    /// Whether the owner is allowed to use `inherit_history_from`.
    /// Planners should not — inheriting another agent's history is an
    /// executor-handoff capability. Defaults to `true`.
    allow_inherit_history: bool,
}

impl AgentTool {
    /// Construct an AgentTool bound to a manager. Depth tracking lives
    /// in [`ExecutionContext::subagent_depth`] — the same instance
    /// works at every level of the spawn tree.
    pub fn new(manager: Arc<AgentManager>) -> Self {
        Self {
            manager: Arc::downgrade(&manager),
            allowed_specs: None,
            spec_resolver: None,
            inherited_policy: None,
            allow_background: true,
            allow_inherit_history: true,
        }
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

    /// Restrict the capabilities this tool exposes. Used for spec
    /// variants that shouldn't have full agent-spawning power — e.g.
    /// planners that should explore + draft, not background-launch
    /// long-running execution.
    pub fn with_restrictions(
        mut self,
        allow_background: bool,
        allow_inherit_history: bool,
    ) -> Self {
        self.allow_background = allow_background;
        self.allow_inherit_history = allow_inherit_history;
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

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        // Upgrade once at the top — if the manager has been dropped,
        // there's no spawning to do regardless of which branch we'd take.
        let manager = match self.manager.upgrade() {
            Some(m) => m,
            None => {
                return ToolResult::error(
                    "AgentTool: parent AgentManager has been dropped; cannot spawn",
                );
            }
        };

        let args: AgentArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };

        if let Some(ref agent_id) = args.to {
            return match manager.send(agent_id, &args.prompt, ctx.cancel).await {
                Ok(result) => ToolResult::text(format_result(&result)),
                Err(e) => ToolResult::error(format!("Resume failed: {}", e)),
            };
        }

        // Per-tool capability gates. The host configures these via
        // `with_restrictions`; LLM-supplied args that violate them are
        // rejected with a clear error so the model can either drop the
        // field or call from a different context.
        if !self.allow_background && args.run_in_background.unwrap_or(false) {
            return ToolResult::error(
                "run_in_background is not available from this agent context. \
                 Background spawning is an executor capability; drop the field \
                 or omit it.",
            );
        }
        if !self.allow_inherit_history && args.inherit_history_from.is_some() {
            return ToolResult::error(
                "inherit_history_from is not available from this agent context. \
                 History inheritance is an executor handoff used by `/plan approve`; \
                 drop the field or omit it.",
            );
        }

        // Canonicalize subagent_type so the LLM can pass "Explore" /
        // "explore" / "EXPLORE" interchangeably. Spec names in
        // `allowed_specs` and the resolver's table are stored in lower
        // case.
        let subagent_type = args.subagent_type.to_ascii_lowercase();

        if let Some(ref allowed) = self.allowed_specs
            && !allowed.iter().any(|s| s.eq_ignore_ascii_case(&subagent_type))
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
        // Depth is read from the runtime-supplied context: a fresh
        // root agent runs tools at depth 0; each spawned child sees
        // its parent's depth + 1 (set by the manager in SpawnOpts).
        let depth = ctx.subagent_depth;
        let spec = match resolver(&subagent_type, depth) {
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
            // Stamp the child's depth — one beyond ours.
            subagent_depth: depth + 1,
        };

        if run_in_background {
            // spawn_background needs a live parent handle to post the
            // FollowUp on completion. Look it up via running_handles —
            // the parent must be running, since this tool is firing
            // mid-prompt on its behalf.
            let parent_id = match ctx.agent_id.as_deref() {
                Some(id) => id.to_string(),
                None => {
                    return ToolResult::error(
                        "Cannot run background agent: tool was invoked outside an agent context (no agent_id on ExecutionContext)",
                    );
                }
            };
            let parent_handle = match manager.handle_for(&parent_id) {
                Some(h) => h,
                None => {
                    return ToolResult::error(format!(
                        "Cannot run background agent: parent agent '{}' not found in running handles \
                         (evicted or not yet registered)",
                        parent_id
                    ));
                }
            };
            let agent_id = manager
                .spawn_background(spec, args.prompt, opts, parent_handle, ctx.cancel)
                .await;
            ToolResult::text(format!(
                "Agent launched in background ({}): {}",
                agent_id, args.description
            ))
        } else {
            match manager.spawn(spec, args.prompt, opts, ctx.cancel).await {
                Ok(result) => ToolResult::text(format_result(&result)),
                Err(e) => ToolResult::error(format!("Agent failed: {}", e)),
            }
        }
    }
}

fn format_result(result: &tau_agent::SubagentResult) -> String {
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

#[cfg(test)]
mod tests {
    use super::*;
    use tau_agent::AgentEvent;
    use tau_agent::AgentManager;
    use tau_agent::test_utils::{MockTransport, make_test_config};
    use tokio::sync::broadcast;

    fn make_manager() -> Arc<AgentManager> {
        let (tx, _) = broadcast::channel::<AgentEvent>(16);
        let transport = Arc::new(MockTransport::new()) as Arc<dyn tau_agent::Transport>;
        Arc::new(AgentManager::new(tx, make_test_config(), transport, 4))
    }

    /// Regression: an AgentTool inside an AgentSpec must not pin the
    /// manager. The cycle `manager → agent_specs → AgentSpec → tools →
    /// AgentTool → manager` was a strong cycle when AgentTool held
    /// `Arc<AgentManager>`; switching to `Weak<AgentManager>` breaks it,
    /// so dropping the host's manager Arc actually frees the manager.
    #[test]
    fn agent_tool_holds_weak_manager() {
        let manager = make_manager();
        let tool = AgentTool::new(Arc::clone(&manager));
        let weak_manager = Arc::downgrade(&manager);
        // Drop the host's strong reference. The tool still references the
        // manager, but only via Weak — so the strong count is now 0 and
        // the manager is freed.
        drop(manager);
        assert!(
            weak_manager.upgrade().is_none(),
            "AgentTool holds a strong reference to AgentManager — registry cycle leaks"
        );
        // Tool itself is still usable as a value (would error on
        // execute, but doesn't crash on construction or drop).
        drop(tool);
    }
}
