//! Tool-call approval policy.
//!
//! Before a tool runs, the actor asks the configured [`ApprovalPolicy`]
//! to classify the call. The decision is one of:
//!
//! - [`ApprovalDecision::Auto`] — dispatch immediately.
//! - [`ApprovalDecision::Gate`] — emit a `Typed { schema_id: "tool.confirm" }`
//!   interaction request and wait for the host UI to approve or reject.
//! - [`ApprovalDecision::Reject`] — synthesize an error tool result without
//!   running the tool.
//!
//! Tools self-report their inherent risk via [`Tool::risk`](crate::core::tool::Tool::risk);
//! policies combine that with the tool name and arguments to make the call.
//!
//! After the gate resolves, the actor emits [`ToolApprovalOutcome`] on the
//! event channel — that's the *observable* outcome (which lives in
//! [`crate::types::events`] because it's purely an event payload). The
//! distinction is deliberate: `ApprovalDecision` is the policy's input to
//! the actor; `ToolApprovalOutcome` is the actor's report to the world.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use crate::types::events::ToolApprovalOutcome;

/// How risky a tool invocation is. Tools self-report via
/// [`Tool::risk`](crate::core::tool::Tool::risk); policies combine this
/// with name and arguments to decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolRisk {
    /// Read-only; safe to auto-run.
    Safe,
    /// Local mutation (file edits, etc.) the user normally allows.
    Local,
    /// Side effects that should default to user confirmation (shell,
    /// network posts, sending drafts).
    Elevated,
}

#[derive(Debug, Clone)]
pub enum ApprovalDecision {
    Auto,
    Gate,
    Reject(String),
}

pub trait ApprovalPolicy: Send + Sync {
    fn classify(&self, tool: &str, arguments: &Value, risk: ToolRisk) -> ApprovalDecision;
}

/// Default: gate `Elevated`, auto-approve everything else.
pub struct DefaultPolicy;

impl ApprovalPolicy for DefaultPolicy {
    fn classify(&self, _tool: &str, _arguments: &Value, risk: ToolRisk) -> ApprovalDecision {
        match risk {
            ToolRisk::Safe | ToolRisk::Local => ApprovalDecision::Auto,
            ToolRisk::Elevated => ApprovalDecision::Gate,
        }
    }
}

/// Auto-approve everything regardless of risk. Used for headless
/// (CI / scripts) and the user-initiated "auto-accept" toggle.
pub struct AutoAcceptAll;

impl ApprovalPolicy for AutoAcceptAll {
    fn classify(&self, _tool: &str, _arguments: &Value, _risk: ToolRisk) -> ApprovalDecision {
        ApprovalDecision::Auto
    }
}

/// Substring rule on `tool:arg-substring`. Matches when `tool` equals
/// the rule's tool name AND any of `arg_substrings` appears in the
/// JSON-serialized arguments. Empty `arg_substrings` matches any
/// arguments for the named tool.
#[derive(Debug, Clone)]
pub struct ToolRule {
    pub tool: String,
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
        self.arg_substrings.iter().any(|n| serialized.contains(n))
    }
}

/// Allow / deny lists with a fallback policy. Deny wins over allow;
/// allow wins over fallback.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_gates_elevated() {
        let p = DefaultPolicy;
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
        let p = AutoAcceptAll;
        assert!(matches!(
            p.classify("bash", &Value::Null, ToolRisk::Elevated),
            ApprovalDecision::Auto
        ));
    }

    #[test]
    fn deny_beats_allow_beats_fallback() {
        let p = RulePolicy::new(Arc::new(AutoAcceptAll)).deny(ToolRule {
            tool: "bash".into(),
            arg_substrings: vec!["rm -rf".into()],
        });
        let dangerous = serde_json::json!({"command": "rm -rf /"});
        assert!(matches!(
            p.classify("bash", &dangerous, ToolRisk::Elevated),
            ApprovalDecision::Reject(_)
        ));
    }

    #[test]
    fn rule_any_matches_only_named_tool() {
        let r = ToolRule::any("bash");
        assert!(r.matches("bash", &Value::Null));
        assert!(r.matches("bash", &serde_json::json!({"command": "x"})));
        assert!(!r.matches("read", &Value::Null));
    }
}
