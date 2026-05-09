//! Tool-call approval policy.
//!
//! Before a tool runs, the runtime asks an [`ApprovalPolicy`] to classify the
//! call into one of three buckets:
//!
//! - [`ApprovalDecision::Auto`] — dispatch immediately.
//! - [`ApprovalDecision::Gate`] — send a `Typed { schema_id: "tool.confirm" }`
//!   interaction request and wait for the host to approve or reject.
//! - [`ApprovalDecision::Reject`] — synthesize an error result; never run.
//!
//! Tools self-report their inherent risk via [`crate::tool::Tool::risk`].
//! Policies combine that with the tool name and arguments to make a call-site
//! decision.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// How risky a tool invocation is. Tools self-report via `Tool::risk`; policies
/// combine this with name/arguments to decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolRisk {
    /// No side effects beyond reading; safe to run automatically.
    Safe,
    /// Local mutations (file edits, etc.) the user normally allows.
    Local,
    /// Side effects that should default to user confirmation (shell, network
    /// posts, draft sends).
    Elevated,
}

/// Per-call decision returned by an `ApprovalPolicy`.
#[derive(Debug, Clone)]
pub enum ApprovalDecision {
    /// Run the tool without asking.
    Auto,
    /// Ask the host to confirm via `Typed { schema_id: "tool.confirm" }`.
    Gate,
    /// Refuse the call. The reason is surfaced to the model as the tool's
    /// error result.
    Reject(String),
}

/// Decides what to do with each pending tool call.
pub trait ApprovalPolicy: Send + Sync {
    fn classify(&self, tool: &str, arguments: &Value, risk: ToolRisk) -> ApprovalDecision;
}

/// Default policy: gate `Elevated`, auto-approve everything else.
pub struct DefaultApprovalPolicy;

impl ApprovalPolicy for DefaultApprovalPolicy {
    fn classify(&self, _tool: &str, _arguments: &Value, risk: ToolRisk) -> ApprovalDecision {
        match risk {
            ToolRisk::Safe | ToolRisk::Local => ApprovalDecision::Auto,
            ToolRisk::Elevated => ApprovalDecision::Gate,
        }
    }
}

/// Auto-approve every tool call regardless of risk. Used for headless contexts
/// (CI, scripts) and the user-initiated "auto-accept" toggle.
pub struct AutoAcceptAllPolicy;

impl ApprovalPolicy for AutoAcceptAllPolicy {
    fn classify(&self, _tool: &str, _arguments: &Value, _risk: ToolRisk) -> ApprovalDecision {
        ApprovalDecision::Auto
    }
}

/// A simple substring rule on `tool:arg-substring`. Matches when
/// `tool` equals the rule's tool name and any of `arg_substrings` appears in
/// the JSON-serialized arguments.
#[derive(Debug, Clone)]
pub struct ToolRule {
    pub tool: String,
    /// Substrings to match against the JSON-serialized arguments. Empty list
    /// matches any arguments for the named tool.
    pub arg_substrings: Vec<String>,
}

impl ToolRule {
    pub fn any(tool: impl Into<String>) -> Self {
        Self {
            tool: tool.into(),
            arg_substrings: vec![],
        }
    }

    pub fn matches(&self, tool: &str, arguments: &Value) -> bool {
        if self.tool != tool {
            return false;
        }
        if self.arg_substrings.is_empty() {
            return true;
        }
        let serialized = serde_json::to_string(arguments).unwrap_or_default();
        self.arg_substrings
            .iter()
            .any(|needle| serialized.contains(needle))
    }
}

/// Allow/deny lists with a fallback policy for everything else.
pub struct RulePolicy {
    pub allow: Vec<ToolRule>,
    pub deny: Vec<ToolRule>,
    pub fallback: Arc<dyn ApprovalPolicy>,
}

impl RulePolicy {
    pub fn new(fallback: Arc<dyn ApprovalPolicy>) -> Self {
        Self {
            allow: vec![],
            deny: vec![],
            fallback,
        }
    }

    pub fn allow(mut self, rule: ToolRule) -> Self {
        self.allow.push(rule);
        self
    }

    pub fn deny(mut self, rule: ToolRule) -> Self {
        self.deny.push(rule);
        self
    }
}

impl ApprovalPolicy for RulePolicy {
    fn classify(&self, tool: &str, arguments: &Value, risk: ToolRisk) -> ApprovalDecision {
        if self.deny.iter().any(|r| r.matches(tool, arguments)) {
            return ApprovalDecision::Reject(format!("denied by policy: {tool}"));
        }
        if self.allow.iter().any(|r| r.matches(tool, arguments)) {
            return ApprovalDecision::Auto;
        }
        self.fallback.classify(tool, arguments, risk)
    }
}

/// Outcome of a single tool's approval, emitted on the event channel as
/// [`crate::events::AgentEvent::ToolApprovalResolved`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolApprovalOutcome {
    /// Approved without asking the user.
    AutoApproved,
    /// Approved by the user.
    Approved,
    /// Rejected (by policy or by the user).
    Rejected { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_gates_elevated() {
        let p = DefaultApprovalPolicy;
        assert!(matches!(
            p.classify("bash", &Value::Null, ToolRisk::Elevated),
            ApprovalDecision::Gate
        ));
        assert!(matches!(
            p.classify("read", &Value::Null, ToolRisk::Local),
            ApprovalDecision::Auto
        ));
        assert!(matches!(
            p.classify("read", &Value::Null, ToolRisk::Safe),
            ApprovalDecision::Auto
        ));
    }

    #[test]
    fn auto_accept_passes_everything() {
        let p = AutoAcceptAllPolicy;
        assert!(matches!(
            p.classify("bash", &Value::Null, ToolRisk::Elevated),
            ApprovalDecision::Auto
        ));
    }

    #[test]
    fn rule_policy_allow_overrides_fallback_gate() {
        let p = RulePolicy::new(Arc::new(DefaultApprovalPolicy)).allow(ToolRule {
            tool: "bash".into(),
            arg_substrings: vec!["git status".into()],
        });
        let args = serde_json::json!({"command": "git status"});
        assert!(matches!(
            p.classify("bash", &args, ToolRisk::Elevated),
            ApprovalDecision::Auto
        ));
        // Different command still gates
        let args2 = serde_json::json!({"command": "rm -rf /"});
        assert!(matches!(
            p.classify("bash", &args2, ToolRisk::Elevated),
            ApprovalDecision::Gate
        ));
    }

    #[test]
    fn rule_policy_deny_takes_precedence() {
        let p = RulePolicy::new(Arc::new(AutoAcceptAllPolicy)).deny(ToolRule {
            tool: "bash".into(),
            arg_substrings: vec!["rm -rf".into()],
        });
        let args = serde_json::json!({"command": "rm -rf /"});
        assert!(matches!(
            p.classify("bash", &args, ToolRisk::Elevated),
            ApprovalDecision::Reject(_)
        ));
    }

    #[test]
    fn tool_rule_any_matches_anything_for_tool() {
        let rule = ToolRule::any("bash");
        assert!(rule.matches("bash", &Value::Null));
        assert!(rule.matches("bash", &serde_json::json!({"command": "x"})));
        assert!(!rule.matches("read", &Value::Null));
    }
}
