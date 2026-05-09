//! Typed implementation plan submitted by the agent and reviewed by the host.
//!
//! Producers (the `submit_plan` tool) build a [`Plan`] and round-trip it
//! through the [`InteractionRequest`](crate::interaction::InteractionRequest)
//! channel as a `Typed { schema_id: "plan.submit", payload }` request whose
//! payload is the serialized `Plan`. The host renders, optionally edits, and
//! replies with
//! [`InteractionResponse::Approved`](crate::interaction::InteractionResponse::Approved)
//! — `payload: Some(value)` carrying the edited body, or `None` to accept
//! the original — or [`InteractionResponse::Rejected`] with a reason the
//! model can revise against.
//!
//! [`InteractionResponse::Rejected`]: crate::interaction::InteractionResponse::Rejected

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The structured plan body submitted by the agent.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Plan {
    /// Ordered execution steps.
    pub items: Vec<PlanStep>,
    /// Files this plan will touch.
    #[serde(default)]
    pub files: Vec<PlanFile>,
    /// Concerns the planner wants to flag for the user before approval.
    #[serde(default)]
    pub flags: Vec<PlanFlag>,
}

/// A single step in the plan.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PlanStep {
    /// Stable identifier the host can reference (e.g. for step events later).
    pub id: String,
    /// Short title for the step.
    pub title: String,
    /// Longer description of what the step does.
    pub description: String,
    /// File paths this step touches, for UI chips.
    #[serde(default)]
    pub touches: Vec<String>,
}

/// A file affected by the plan, with op + line counts for diff previews.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PlanFile {
    pub op: PlanFileOp,
    pub path: String,
    #[serde(default)]
    pub adds: u32,
    #[serde(default)]
    pub dels: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PlanFileOp {
    Add,
    Modify,
    Delete,
}

/// A pre-approval concern: incompatibility, missing context, irreversibility.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct PlanFlag {
    pub severity: PlanFlagSeverity,
    pub title: String,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PlanFlagSeverity {
    Info,
    Warning,
    Danger,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_round_trips_via_json() {
        let plan = Plan {
            items: vec![PlanStep {
                id: "s1".into(),
                title: "Add module".into(),
                description: "Create new module".into(),
                touches: vec!["src/foo.rs".into()],
            }],
            files: vec![PlanFile {
                op: PlanFileOp::Add,
                path: "src/foo.rs".into(),
                adds: 10,
                dels: 0,
            }],
            flags: vec![PlanFlag {
                severity: PlanFlagSeverity::Warning,
                title: "Migration".into(),
                description: "Requires DB migration".into(),
            }],
        };

        let json = serde_json::to_value(&plan).unwrap();
        let back: Plan = serde_json::from_value(json).unwrap();
        assert_eq!(back.items.len(), 1);
        assert_eq!(back.files[0].op, PlanFileOp::Add);
        assert_eq!(back.flags[0].severity, PlanFlagSeverity::Warning);
    }

    #[test]
    fn plan_schema_lists_required_fields() {
        let schema = schemars::schema_for!(Plan);
        let json = serde_json::to_value(&schema).unwrap();
        // Top-level Plan should require items (files/flags default to []).
        let required = json
            .get("required")
            .and_then(|r| r.as_array())
            .cloned()
            .unwrap_or_default();
        let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"items"));
    }
}
