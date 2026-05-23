//! Host-side subagent spec table and resolver.
//!
//! The runtime is ignorant of which spec names exist. This module owns:
//!   * The declarative table of supported subagent specs.
//!   * The recursive `SpecResolver` that materializes a spec by name +
//!     depth, attaching a nested `AgentTool` (with the right depth and
//!     allowed-spawn list) when the spec permits further subagents.
//!
//! `main.rs` just calls [`build_resolver`].

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, Weak};

use tau_agent::{AgentManager, AgentSpec, BoxedTool};

/// Maximum recursive spawn depth. `depth == 0` is the root agent's
/// `AgentTool`; each nested spawn increments by one until this limit is
/// reached, at which point the resolver omits the nested `AgentTool` so
/// the LLM can't spawn deeper.
const MAX_DEPTH: u32 = 3;

/// Which tools a spec carries.
enum ToolSet {
    /// Everything in the host's tool list except `agent` (the host's
    /// recursive `AgentTool` is added separately by the resolver).
    AllExceptAgent,
    /// Pick tools whose names appear in this whitelist.
    Whitelist(&'static [&'static str]),
}

struct SubagentDef {
    name: &'static str,
    /// Bare instruction text. The resolver wraps it with the env / tool
    /// section via [`crate::prompts::build_subagent_prompt`].
    prompt: &'static str,
    /// Appended to `prompt` before wrapping. Used by the executor
    /// variant to layer "plan execution" instructions on top of the
    /// general-purpose prompt.
    prompt_suffix: Option<&'static str>,
    tools: ToolSet,
    /// Extra tool names to include beyond what `tools` selects. Used to
    /// add `subagent_report` / `submit_plan` to whitelist-based specs.
    extras: &'static [&'static str],
    /// Whether this spec may run in a git worktree. The runtime no longer
    /// gates worktree creation per-spec; this host-side flag is enforced
    /// by the `AgentTool`'s worktree allowlist (see [`build_resolver`]).
    /// Read-only/planning specs set this `false` so the model can't put
    /// them on an isolated branch.
    allows_worktree: bool,
    /// Spec names this agent is allowed to spawn. `None` means it cannot
    /// spawn subagents at all. Host-side allowlist — the runtime does
    /// not enforce it; the resolver uses it to install nested
    /// `AgentTool`s with the right allowlist.
    can_spawn: Option<&'static [&'static str]>,
}

const SPECS: &[SubagentDef] = &[
    SubagentDef {
        name: "general-purpose",
        prompt: include_str!("prompts/agent_general.md"),
        prompt_suffix: None,
        tools: ToolSet::AllExceptAgent,
        extras: &[],
        allows_worktree: true,
        can_spawn: Some(&["general-purpose", "explore", "plan"]),
    },
    SubagentDef {
        name: "explore",
        prompt: include_str!("prompts/agent_explore.md"),
        prompt_suffix: None,
        tools: ToolSet::Whitelist(&["read", "glob", "grep", "list", "lsp"]),
        extras: &["subagent_report"],
        allows_worktree: false,
        can_spawn: None,
    },
    SubagentDef {
        name: "plan",
        prompt: include_str!("prompts/agent_plan.md"),
        prompt_suffix: None,
        tools: ToolSet::Whitelist(&["read", "glob", "grep", "list", "lsp"]),
        extras: &["subagent_report", "submit_plan"],
        allows_worktree: false,
        can_spawn: Some(&["explore", "plan"]),
    },
    SubagentDef {
        name: "general-purpose:executor",
        prompt: include_str!("prompts/agent_general.md"),
        prompt_suffix: Some(include_str!("prompts/agent_executor.md")),
        tools: ToolSet::AllExceptAgent,
        extras: &[],
        allows_worktree: true,
        can_spawn: Some(&["general-purpose", "explore", "plan"]),
    },
];

/// Canonical (lowercase) names of specs that may run in a git worktree.
/// Passed to every [`AgentTool`](tau_tools::AgentTool) so a spawn that
/// requests `isolation: worktree` for an ineligible spec is rejected.
pub fn worktree_specs() -> Vec<String> {
    SPECS
        .iter()
        .filter(|def| def.allows_worktree)
        .map(|def| def.name.to_string())
        .collect()
}

/// A materialized spec plus its host-side allowlist for further spawns.
struct ResolvedSpec {
    spec: AgentSpec,
    can_spawn: Option<Vec<String>>,
}

/// Build the spec resolver. Captures `manager` (weakly via the closure
/// it returns to the caller) and the precomputed spec map; isolates the
/// `Arc<OnceLock<Weak<...>>>` self-reference trick needed for the
/// resolver to attach itself to nested `AgentTool` instances.
pub fn build_resolver(
    manager: Arc<AgentManager>,
    all_tools: &[BoxedTool],
    cwd: &str,
) -> tau_tools::SpecResolver {
    let base_specs = Arc::new(materialize_specs(all_tools, cwd));

    type SpecResolverFn = dyn Fn(&str, u32) -> Option<AgentSpec> + Send + Sync;
    let resolver_self: Arc<OnceLock<Weak<SpecResolverFn>>> = Arc::new(OnceLock::new());
    let resolver_self_for_closure = resolver_self.clone();

    let resolver: tau_tools::SpecResolver = Arc::new(move |name: &str, depth: u32| {
        // Spec names are canonicalized lowercase. The `AgentTool`
        // already lowercases user input before calling us; this guard
        // covers direct callers (`Session::enter_plan_mode`, etc.) and
        // future entry points.
        let key = name.to_ascii_lowercase();
        let resolved = base_specs.get(key.as_str())?;
        let mut spec = resolved.spec.clone();
        if let Some(ref allowed) = resolved.can_spawn
            && depth + 1 < MAX_DEPTH
        {
            let recursive: tau_tools::SpecResolver = resolver_self_for_closure
                .get()
                .and_then(Weak::upgrade)
                .expect("resolver self-ref not yet set");
            // Planners can spawn explore/plan subagents but they
            // shouldn't fire-and-forget background spawns or take over
            // another agent's history — those are executor moves.
            let is_planner = key == "plan";
            // Depth no longer baked into the tool — the runtime
            // supplies it via `ExecutionContext::subagent_depth`. We
            // still consult the closure's `depth` argument above as
            // the host-side recursion guard.
            let mut nested = tau_tools::AgentTool::new(manager.clone())
                .with_spec_resolver(recursive)
                .with_allowed_specs(allowed.clone())
                .with_worktree_specs(worktree_specs());
            if is_planner {
                nested = nested.with_restrictions(false, false);
            }
            spec.tools.push(Arc::new(nested));
        }
        Some(spec)
    });
    let _ = resolver_self.set(Arc::downgrade(&resolver));
    resolver
}

fn materialize_specs(all_tools: &[BoxedTool], cwd: &str) -> HashMap<String, ResolvedSpec> {
    SPECS
        .iter()
        .map(|def| (def.name.to_string(), materialize(def, all_tools, cwd)))
        .collect()
}

fn materialize(def: &SubagentDef, all_tools: &[BoxedTool], cwd: &str) -> ResolvedSpec {
    let tools = pick_tools(&def.tools, def.extras, all_tools);
    let bare_prompt = match def.prompt_suffix {
        Some(suffix) => format!("{}\n\n{}", def.prompt, suffix),
        None => def.prompt.to_string(),
    };
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
    let system_prompt = crate::prompts::build_subagent_prompt(&bare_prompt, &tool_names, cwd);

    ResolvedSpec {
        spec: AgentSpec {
            system_prompt,
            tools,
            max_turns: 200,
        },
        can_spawn: def.can_spawn.map(|s| s.iter().map(|n| n.to_string()).collect()),
    }
}

fn pick_tools(set: &ToolSet, extras: &[&str], all_tools: &[BoxedTool]) -> Vec<BoxedTool> {
    let mut picked: Vec<BoxedTool> = match set {
        ToolSet::AllExceptAgent => all_tools
            .iter()
            .filter(|t| t.name() != "agent")
            .cloned()
            .collect(),
        ToolSet::Whitelist(names) => all_tools
            .iter()
            .filter(|t| names.contains(&t.name()))
            .cloned()
            .collect(),
    };
    for extra in extras {
        if let Some(t) = all_tools.iter().find(|t| t.name() == *extra) {
            picked.push(t.clone());
        }
    }
    picked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cwd() -> String {
        ".".to_string()
    }

    #[test]
    fn executor_spec_carries_executor_suffix() {
        let map = materialize_specs(&[], &cwd());
        let exec = map
            .get("general-purpose:executor")
            .expect("executor spec registered");
        assert!(
            exec.spec.system_prompt.contains("Plan Executor Mode"),
            "executor spec carries the executor prompt suffix"
        );
        let general = map
            .get("general-purpose")
            .expect("general-purpose registered");
        assert!(
            !general.spec.system_prompt.contains("Plan Executor Mode"),
            "non-executor spec does NOT carry the suffix"
        );
    }

    #[test]
    fn plan_can_spawn_explore_plan() {
        let map = materialize_specs(&[], &cwd());
        let plan = map.get("plan").unwrap();
        let allowed = plan.can_spawn.as_ref().unwrap();
        assert_eq!(allowed, &vec!["explore".to_string(), "plan".to_string()]);
    }

    #[test]
    fn worktree_specs_excludes_read_only_agents() {
        let wt = worktree_specs();
        assert!(
            wt.contains(&"general-purpose".to_string()),
            "general-purpose must be worktree-eligible"
        );
        assert!(
            wt.contains(&"general-purpose:executor".to_string()),
            "executor must be worktree-eligible"
        );
        assert!(
            !wt.contains(&"explore".to_string()),
            "explore must not be worktree-eligible"
        );
        assert!(
            !wt.contains(&"plan".to_string()),
            "plan must not be worktree-eligible"
        );
    }

    #[test]
    fn explore_spec_cannot_spawn() {
        let map = materialize_specs(&[], &cwd());
        let explore = map.get("explore").unwrap();
        assert!(explore.can_spawn.is_none());
    }

    #[test]
    fn resolver_is_case_insensitive() {
        // Build a manager just to satisfy the resolver signature; we
        // don't actually spawn anything.
        use std::sync::Arc;
        use tau_agent::AgentManager;
        use tau_agent::test_utils::{MockTransport, make_test_config};
        let transport = Arc::new(MockTransport::new()) as Arc<dyn tau_agent::Transport>;
        let manager = Arc::new(AgentManager::new(make_test_config(), transport, 4));
        let resolver = build_resolver(manager, &[], &cwd());

        // Each of these should resolve to the same spec.
        assert!(resolver("plan", 0).is_some(), "lowercase");
        assert!(resolver("Plan", 0).is_some(), "title case");
        assert!(resolver("PLAN", 0).is_some(), "upper case");
        assert!(
            resolver("GENERAL-PURPOSE:EXECUTOR", 0).is_some(),
            "compound name, upper"
        );
        assert!(resolver("nonsense", 0).is_none(), "unknown still rejected");
    }
}
