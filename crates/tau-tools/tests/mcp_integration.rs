//! End-to-end MCP tests against an in-process fixture server connected
//! over `tokio::io::duplex` (the rmcp SDK's own test pattern — no
//! child processes or network).

#![cfg(feature = "mcp")]

use std::time::Duration;

use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ClientCapabilities, ClientInfo, Content, Implementation, ServerCapabilities,
    ServerInfo,
};
use rmcp::{ServerHandler, ServiceExt, schemars, tool, tool_handler, tool_router};
use tau_agent::ToolRisk;
use tau_agent::test_utils::make_execution_context;
use tau_tools::mcp::{McpManager, McpServerSpec, McpTransportSpec, McpTrust};

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct EchoArgs {
    message: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct SleepArgs {
    ms: u64,
}

#[derive(Clone)]
struct Fixture {
    // Read by the #[tool_handler]-generated ServerHandler methods.
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl Fixture {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Echo a message back")]
    fn echo(&self, Parameters(EchoArgs { message }): Parameters<EchoArgs>) -> String {
        format!("echo: {message}")
    }

    #[tool(description = "A read-only peek", annotations(read_only_hint = true))]
    fn peek(&self) -> String {
        "peeked".to_string()
    }

    #[tool(description = "Always fails")]
    fn fail(&self) -> CallToolResult {
        CallToolResult::error(vec![Content::text("deliberate failure")])
    }

    #[tool(description = "Sleep for the given milliseconds")]
    async fn sleep_ms(&self, Parameters(SleepArgs { ms }): Parameters<SleepArgs>) -> String {
        tokio::time::sleep(Duration::from_millis(ms)).await;
        "slept".to_string()
    }
}

#[tool_handler]
impl ServerHandler for Fixture {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
    }
}

fn spec(name: &str, trust: McpTrust, timeout: Duration) -> McpServerSpec {
    McpServerSpec {
        name: name.into(),
        // Transport is unused by from_running_services; any value works.
        transport: McpTransportSpec::Http {
            url: "http://unused/mcp".into(),
            auth_header: None,
        },
        call_timeout: timeout,
        trust,
        include_tools: None,
        exclude_tools: vec![],
    }
}

/// Serve the fixture over a duplex pipe and return a connected manager.
async fn manager_with(trust: McpTrust, timeout: Duration) -> McpManager {
    let (server_io, client_io) = tokio::io::duplex(4096);
    tokio::spawn(async move {
        let service = Fixture::new().serve(server_io).await.unwrap();
        let _ = service.waiting().await;
    });
    let client_info = ClientInfo::new(
        ClientCapabilities::default(),
        Implementation::new("tau-test", "0.0.0"),
    );
    let running = client_info.serve(client_io).await.unwrap();
    McpManager::from_running_services(vec![(spec("fixture", trust, timeout), running)])
        .await
        .unwrap()
}

fn find<'a>(
    tools: &'a [tau_agent::BoxedTool],
    name: &str,
) -> &'a tau_agent::BoxedTool {
    tools
        .iter()
        .find(|t| t.name() == name)
        .unwrap_or_else(|| panic!("tool {name} not found"))
}

#[tokio::test]
async fn tools_are_namespaced_and_risk_classified() {
    let manager = manager_with(McpTrust::Untrusted, Duration::from_secs(5)).await;
    let tools = manager.tools().await;
    assert_eq!(tools.len(), 4);

    let echo = find(&tools, "mcp__fixture__echo");
    assert_eq!(echo.risk(&serde_json::json!({})), ToolRisk::Elevated);
    assert_eq!(echo.label(), "fixture:echo");
    let schema = echo.parameters_schema();
    assert_eq!(schema["type"], "object");
    assert!(schema["properties"]["message"].is_object());

    // read_only_hint downgrades to Local even on an untrusted server.
    let peek = find(&tools, "mcp__fixture__peek");
    assert_eq!(peek.risk(&serde_json::json!({})), ToolRisk::Local);

    manager.shutdown_all().await;
}

#[tokio::test]
async fn call_round_trips_and_errors_map() {
    let manager = manager_with(McpTrust::Trusted, Duration::from_secs(5)).await;
    let tools = manager.tools().await;

    let echo = find(&tools, "mcp__fixture__echo");
    let result = echo
        .execute(serde_json::json!({"message": "hi"}), make_execution_context())
        .await;
    assert!(!result.is_error);
    let text = match &result.content[0] {
        tau_ai::Content::Text { text } => text.clone(),
        other => panic!("expected text, got {other:?}"),
    };
    assert!(text.contains("echo: hi"), "got: {text}");

    let fail = find(&tools, "mcp__fixture__fail");
    let result = fail
        .execute(serde_json::json!({}), make_execution_context())
        .await;
    assert!(result.is_error, "server is_error must map through");

    manager.shutdown_all().await;
}

#[tokio::test]
async fn slow_calls_time_out() {
    let manager = manager_with(McpTrust::Trusted, Duration::from_millis(50)).await;
    let tools = manager.tools().await;
    let sleep = find(&tools, "mcp__fixture__sleep_ms");
    let result = sleep
        .execute(serde_json::json!({"ms": 5000}), make_execution_context())
        .await;
    assert!(result.is_error);
    let text = match &result.content[0] {
        tau_ai::Content::Text { text } => text.clone(),
        _ => panic!("expected text"),
    };
    assert!(text.contains("timed out"), "got: {text}");
    manager.shutdown_all().await;
}

#[tokio::test]
async fn cancellation_aborts_the_call() {
    let manager = manager_with(McpTrust::Trusted, Duration::from_secs(30)).await;
    let tools = manager.tools().await;
    let sleep = find(&tools, "mcp__fixture__sleep_ms").clone();

    let ctx = make_execution_context();
    let cancel = ctx.cancel.clone();
    let call = tokio::spawn(async move {
        sleep
            .execute(serde_json::json!({"ms": 30000}), ctx)
            .await
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    cancel.cancel();
    let result = tokio::time::timeout(Duration::from_secs(2), call)
        .await
        .expect("cancellation must resolve the call promptly")
        .unwrap();
    assert!(result.is_error);
    manager.shutdown_all().await;
}
