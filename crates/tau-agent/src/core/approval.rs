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
//! After the gate resolves, the actor emits
//! [`ToolApprovalOutcome`](crate::ToolApprovalOutcome) on the
//! event channel — that's the *observable* outcome (which lives in
//! [`crate::types::events`] because it's purely an event payload). The
//! distinction is deliberate: `ApprovalDecision` is the policy's input to
//! the actor; `ToolApprovalOutcome` is the actor's report to the world.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

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

/// How to compare a needle against a string-typed leaf in a tool's
/// arguments. Runs of ASCII whitespace (spaces, tabs, newlines) collapse
/// to a single space before comparison, closing the historical footgun
/// where `"rm  -rf"` (two spaces) bypassed a rule written for `"rm -rf"`.
/// Matching is case-sensitive.
///
/// Whitespace at the ends is handled per variant:
/// - [`Contains`](Self::Contains) keeps a needle's leading/trailing space
///   significant, so `" rm "` matches `"foo rm bar"` but not
///   `"transform"`. An empty or whitespace-only `Contains` needle matches
///   **nothing** (it would otherwise match every value).
/// - [`Equals`](Self::Equals) trims both ends, so surrounding padding is
///   insignificant.
#[derive(Debug, Clone)]
pub enum ArgPattern {
    /// Substring match: the (normalized) needle appears anywhere in
    /// the (normalized) value.
    Contains(String),
    /// Exact string equality after normalization.
    Equals(String),
}

impl ArgPattern {
    fn matches_value(&self, value: &str) -> bool {
        match self {
            ArgPattern::Contains(needle) => {
                let needle = collapse_ws(needle);
                // An empty or whitespace-only needle carries no
                // constraint; treat it as "never matches" rather than
                // "matches everything" — the latter would silently
                // auto-approve (or block) every invocation of the tool.
                !needle.is_empty() && collapse_ws(value).contains(&needle)
            }
            // Exact match modulo whitespace: collapse internal runs, then
            // trim the ends so surrounding padding is insignificant.
            ArgPattern::Equals(needle) => {
                collapse_ws(value).trim() == collapse_ws(needle).trim()
            }
        }
    }
}

/// One argument-side check. A [`ToolRule`] matches when **any** of its
/// `ArgMatch`es match (OR), preserving the old `arg_substrings` mental
/// model. Path scoping is per-match.
#[derive(Debug, Clone)]
pub struct ArgMatch {
    /// JSON path scoping the match. `None` searches every string-typed
    /// leaf in the arguments tree. `Some("command")` looks at
    /// `arguments.command`; `Some("options.shell")` walks nested
    /// objects; array indices use `.N` (e.g. `"argv.0"`). Path
    /// components are exact — no globbing.
    ///
    /// If the path resolves to a non-string value (object, array,
    /// number, null) the match fails. To match a value buried inside
    /// an object/array, leave `path: None` so the walker visits every
    /// string leaf.
    pub path: Option<String>,
    pub pattern: ArgPattern,
}

impl ArgMatch {
    pub fn matches(&self, arguments: &Value) -> bool {
        match &self.path {
            Some(path) => match resolve_path(arguments, path) {
                Some(Value::String(s)) => self.pattern.matches_value(s),
                _ => false,
            },
            None => any_string_leaf(arguments, &mut |leaf| self.pattern.matches_value(leaf)),
        }
    }
}

/// Argument-keyed rule for a single tool. **Compared against typed
/// JSON values, not the serialized JSON string** — earlier versions of
/// this type matched substrings against `serde_json::to_string(args)`,
/// which made whitespace, key order, and JSON escaping part of the
/// security boundary. Don't do that.
///
/// A rule matches when:
///   1. `tool` equals the call's tool name, AND
///   2. either `matches` is empty (any invocation matches), OR any
///      single [`ArgMatch`] in `matches` matches the arguments (OR
///      semantics — combine via separate rules for AND).
#[derive(Debug, Clone)]
pub struct ToolRule {
    pub tool: String,
    pub matches: Vec<ArgMatch>,
}

impl ToolRule {
    /// Match any invocation of the named tool, regardless of arguments.
    pub fn any(tool: impl Into<String>) -> Self {
        Self {
            tool: tool.into(),
            matches: vec![],
        }
    }

    /// Convenience: substring match against any string-typed argument
    /// leaf, with whitespace normalization. Equivalent to the legacy
    /// `arg_substrings: vec![needle]` shape, minus the brittle
    /// serialize-then-substring behavior.
    pub fn contains(tool: impl Into<String>, needle: impl Into<String>) -> Self {
        Self {
            tool: tool.into(),
            matches: vec![ArgMatch {
                path: None,
                pattern: ArgPattern::Contains(needle.into()),
            }],
        }
    }

    /// Convenience: substring match scoped to a specific JSON path
    /// (e.g. `"command"`). See [`ArgMatch::path`] for the path syntax.
    pub fn contains_at(
        tool: impl Into<String>,
        path: impl Into<String>,
        needle: impl Into<String>,
    ) -> Self {
        Self {
            tool: tool.into(),
            matches: vec![ArgMatch {
                path: Some(path.into()),
                pattern: ArgPattern::Contains(needle.into()),
            }],
        }
    }

    /// Convenience: exact match (after normalization) at a JSON path.
    pub fn equals_at(
        tool: impl Into<String>,
        path: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            tool: tool.into(),
            matches: vec![ArgMatch {
                path: Some(path.into()),
                pattern: ArgPattern::Equals(value.into()),
            }],
        }
    }

    pub fn matches(&self, tool: &str, arguments: &Value) -> bool {
        if self.tool != tool {
            return false;
        }
        if self.matches.is_empty() {
            return true;
        }
        self.matches.iter().any(|m| m.matches(arguments))
    }
}

/// Collapse runs of ASCII whitespace to a single space, preserving a
/// single leading/trailing space when it borders content. Unlike a full
/// trim, this keeps a `Contains` needle's intentional boundary spaces
/// (e.g. `" rm "`) significant, so the needle isn't silently broadened
/// into matching `"transform"`. A string with no non-whitespace content
/// collapses to the empty string.
///
/// Applied symmetrically to needle and value so `"rm  -rf"` (two spaces)
/// and `"rm -rf"` compare equal.
fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending_space = false;
    for ch in s.chars() {
        if ch.is_ascii_whitespace() {
            pending_space = true;
        } else {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(ch);
        }
    }
    // A trailing whitespace run becomes a single space only when there's
    // content before it; a pure-whitespace input stays empty.
    if pending_space && !out.is_empty() {
        out.push(' ');
    }
    out
}

/// Walk a JSON tree, invoking `f` on every string-typed leaf until one
/// returns `true`. Returns whether any leaf matched.
fn any_string_leaf<F>(value: &Value, f: &mut F) -> bool
where
    F: FnMut(&str) -> bool,
{
    match value {
        Value::String(s) => f(s),
        Value::Array(items) => items.iter().any(|v| any_string_leaf(v, f)),
        Value::Object(map) => map.values().any(|v| any_string_leaf(v, f)),
        _ => false,
    }
}

/// Resolve a dot-separated path against a JSON value. Path components
/// that parse as `usize` index arrays; everything else looks up object
/// keys. Returns `None` if any component is missing.
fn resolve_path<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = value;
    for component in path.split('.') {
        current = match current {
            Value::Object(map) => map.get(component)?,
            Value::Array(items) => {
                let idx: usize = component.parse().ok()?;
                items.get(idx)?
            }
            _ => return None,
        };
    }
    Some(current)
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
        let p =
            RulePolicy::new(Arc::new(AutoAcceptAll)).deny(ToolRule::contains("bash", "rm -rf"));
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

    #[test]
    fn whitespace_normalization_closes_double_space_bypass() {
        // The historical bug: `arg_substrings: ["rm -rf"]` did not
        // match `"rm  -rf"` (two spaces) in the serialized JSON. After
        // the typed-walk + whitespace-normalize fix, both compare
        // equal.
        let rule = ToolRule::contains("bash", "rm -rf");
        let one_space = serde_json::json!({"command": "rm -rf /tmp/foo"});
        let two_spaces = serde_json::json!({"command": "rm  -rf /tmp/foo"});
        let tab_separated = serde_json::json!({"command": "rm\t-rf /tmp/foo"});
        let newline_separated = serde_json::json!({"command": "rm\n-rf /tmp/foo"});
        assert!(rule.matches("bash", &one_space));
        assert!(rule.matches("bash", &two_spaces), "double space bypass");
        assert!(rule.matches("bash", &tab_separated), "tab bypass");
        assert!(
            rule.matches("bash", &newline_separated),
            "newline bypass"
        );
    }

    #[test]
    fn match_walks_string_leaves_not_serialized_json() {
        // The needle "git status" must match against the value
        // `"git status"`, not against the substring of the serialized
        // JSON `{"command":"git status"}`. The two were equivalent
        // before (substring of the serialization) but are distinct
        // now — a needle of `command":"git` no longer matches.
        let rule = ToolRule::contains("bash", "git status");
        let call = serde_json::json!({"command": "git status"});
        assert!(rule.matches("bash", &call));

        let fake = ToolRule::contains("bash", r#"command":"git"#);
        assert!(
            !fake.matches("bash", &call),
            "needle that only matched the serialized JSON must no longer match"
        );
    }

    #[test]
    fn path_scoping_restricts_match() {
        // A rule scoped to `command` only matches if the string at
        // that path matches — content elsewhere in the args is not
        // considered.
        let rule = ToolRule::contains_at("bash", "command", "rm -rf");
        let in_command = serde_json::json!({"command": "rm -rf /", "label": "safe"});
        let elsewhere = serde_json::json!({"command": "ls", "label": "rm -rf /"});
        assert!(rule.matches("bash", &in_command));
        assert!(
            !rule.matches("bash", &elsewhere),
            "path-scoped rule must not match string in other fields"
        );
    }

    #[test]
    fn path_resolves_nested_objects_and_arrays() {
        let rule = ToolRule::contains_at("exec", "options.argv.0", "rm");
        let call = serde_json::json!({"options": {"argv": ["rm", "-rf", "/tmp"]}});
        assert!(rule.matches("exec", &call));

        let other = serde_json::json!({"options": {"argv": ["ls", "-la"]}});
        assert!(!rule.matches("exec", &other));
    }

    #[test]
    fn equals_at_normalizes_whitespace() {
        let rule = ToolRule::equals_at("bash", "command", "rm -rf /");
        let exact = serde_json::json!({"command": "rm -rf /"});
        let padded = serde_json::json!({"command": "  rm  -rf  /  "});
        assert!(rule.matches("bash", &exact));
        assert!(rule.matches("bash", &padded));
    }

    #[test]
    fn empty_contains_needle_matches_nothing() {
        // A degenerate empty needle must not collapse a rule into
        // "matches everything" — that would auto-approve (allow) or
        // block (deny) every invocation of the tool.
        let empty = ToolRule::contains("bash", "");
        assert!(!empty.matches("bash", &serde_json::json!({"command": "rm -rf /"})));
        assert!(!empty.matches("bash", &serde_json::json!({"command": ""})));

        // Whitespace-only is equally degenerate after collapsing.
        let ws_only = ToolRule::contains_at("bash", "command", "  \t\n ");
        assert!(!ws_only.matches("bash", &serde_json::json!({"command": "anything at all"})));
    }

    #[test]
    fn contains_needle_boundary_spaces_are_significant() {
        // Surrounding spaces mean "the word rm", not a bare substring.
        // Normalization must preserve that boundary rather than trimming
        // it into a broad substring match.
        let rule = ToolRule::contains_at("bash", "command", " rm ");
        assert!(
            rule.matches("bash", &serde_json::json!({"command": "foo rm bar"})),
            "space-delimited word should match"
        );
        assert!(
            !rule.matches("bash", &serde_json::json!({"command": "transform x"})),
            "boundary spaces must not be trimmed into a substring match"
        );
        assert!(
            !rule.matches("bash", &serde_json::json!({"command": "perform y"})),
            "boundary spaces must not be trimmed into a substring match"
        );
    }

    #[test]
    fn multiple_matchers_or_semantics() {
        // Multiple ArgMatch entries in one rule: any match is enough.
        let rule = ToolRule {
            tool: "bash".into(),
            matches: vec![
                ArgMatch {
                    path: Some("command".into()),
                    pattern: ArgPattern::Contains("rm -rf".into()),
                },
                ArgMatch {
                    path: Some("command".into()),
                    pattern: ArgPattern::Contains("dd if=".into()),
                },
            ],
        };
        let rm = serde_json::json!({"command": "rm -rf /"});
        let dd = serde_json::json!({"command": "dd if=/dev/zero of=/dev/sda"});
        let safe = serde_json::json!({"command": "ls"});
        assert!(rule.matches("bash", &rm));
        assert!(rule.matches("bash", &dd));
        assert!(!rule.matches("bash", &safe));
    }
}
