//! Tests for `AgentHandle::list_tools()`.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tau_agent::{AutoAcceptAll, DefaultPolicy, ToolRisk};
use tau_agent::{Concurrency, ExecutionContext, Tool, ToolCategory, ToolResult};
use tau_agent::test_utils::*;
use tau_agent::*;

/// Read-category, Safe-risk tool.
struct ReadyTool;

#[async_trait]
impl Tool for ReadyTool {
    fn name(&self) -> &str {
        "ready"
    }
    fn description(&self) -> &str {
        "a safe, read-only test tool"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({"type": "object", "properties": {}})
    }
    fn risk(&self, _arguments: &Value) -> ToolRisk {
        ToolRisk::Safe
    }
    fn category(&self) -> ToolCategory {
        ToolCategory::Read
    }
    async fn execute(&self, _args: Value, _ctx: ExecutionContext) -> ToolResult {
        ToolResult::text("ok")
    }
}

/// Execute-category, Elevated-risk tool.
struct DangerTool;

#[async_trait]
impl Tool for DangerTool {
    fn name(&self) -> &str {
        "danger"
    }
    fn description(&self) -> &str {
        "elevated test tool"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({"type": "object", "properties": {}})
    }
    fn concurrency(&self) -> Concurrency {
        Concurrency::Sequential
    }
    fn risk(&self, _arguments: &Value) -> ToolRisk {
        ToolRisk::Elevated
    }
    fn category(&self) -> ToolCategory {
        ToolCategory::Execute
    }
    async fn execute(&self, _args: Value, _ctx: ExecutionContext) -> ToolResult {
        ToolResult::text("ran")
    }
}

async fn spawn_with_policy(policy: Arc<dyn ApprovalPolicy>) -> AgentHandle {
    let transport = MockTransport::new();
    let mut builder = AgentBuilder::new(test_config(), Arc::new(transport));
    builder.add_tool(Arc::new(ReadyTool));
    builder.add_tool(Arc::new(DangerTool));
    // EchoTool falls back to `category()` default (Other) and `risk()`
    // default (Local) — exercises the trait defaults.
    builder.add_tool(Arc::new(EchoTool));
    builder.set_approval_policy(policy);
    let handle = builder.handle();
    builder.spawn().await.unwrap();
    handle
}

#[tokio::test]
async fn list_tools_reports_categories_and_descriptions() {
    let handle = spawn_with_policy(Arc::new(DefaultPolicy)).await;
    let infos = handle.list_tools().await.expect("list_tools");

    assert_eq!(infos.len(), 3);

    let ready = infos.iter().find(|i| i.name == "ready").unwrap();
    assert_eq!(ready.category, ToolCategory::Read);
    assert_eq!(ready.description, "a safe, read-only test tool");

    let danger = infos.iter().find(|i| i.name == "danger").unwrap();
    assert_eq!(danger.category, ToolCategory::Execute);

    let echo = infos.iter().find(|i| i.name == "echo").unwrap();
    // EchoTool doesn't override category(); should be the default.
    assert_eq!(echo.category, ToolCategory::Other);
}

#[tokio::test]
async fn default_policy_marks_elevated_as_not_currently_allowed() {
    let handle = spawn_with_policy(Arc::new(DefaultPolicy)).await;
    let infos = handle.list_tools().await.unwrap();

    let ready = infos.iter().find(|i| i.name == "ready").unwrap();
    assert!(
        ready.default_allowed,
        "Safe risk: default_allowed should be true"
    );
    assert!(
        ready.currently_allowed,
        "DefaultPolicy auto-approves Safe risk"
    );

    let echo = infos.iter().find(|i| i.name == "echo").unwrap();
    assert!(
        echo.default_allowed,
        "Local risk (Tool::risk default): default_allowed should be true"
    );
    assert!(
        echo.currently_allowed,
        "DefaultPolicy auto-approves Local risk"
    );

    let danger = infos.iter().find(|i| i.name == "danger").unwrap();
    assert!(
        !danger.default_allowed,
        "Elevated risk: default_allowed should be false"
    );
    assert!(
        !danger.currently_allowed,
        "DefaultPolicy gates Elevated risk → currently_allowed should be false"
    );
}

#[tokio::test]
async fn auto_accept_marks_everything_currently_allowed() {
    let handle = spawn_with_policy(Arc::new(AutoAcceptAll)).await;
    let infos = handle.list_tools().await.unwrap();

    for info in &infos {
        assert!(
            info.currently_allowed,
            "AutoAcceptAll: every tool ({}) should be currently_allowed",
            info.name
        );
    }

    // default_allowed is policy-independent.
    let danger = infos.iter().find(|i| i.name == "danger").unwrap();
    assert!(
        !danger.default_allowed,
        "default_allowed reflects intrinsic risk, not the active policy"
    );
}
