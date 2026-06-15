//! [`Tool`] implementation bridging one remote MCP tool to tau.

use async_trait::async_trait;
use rmcp::RoleClient;
use rmcp::model::CallToolRequestParams;
use rmcp::service::Peer;
use serde_json::Value;
use tau_agent::{Concurrency, ExecutionContext, Tool, ToolCategory, ToolResult, ToolRisk};

use super::content;

pub(crate) struct McpTool {
    /// Provider-facing name: `mcp__<server>__<tool>` (sanitized).
    pub name: String,
    /// Human-readable `server:tool` for UI labels.
    pub label: String,
    pub description: String,
    /// Normalized input schema (see [`super::schema`]).
    pub schema: Value,
    /// Original remote tool name, sent in `tools/call`.
    pub remote_name: String,
    pub server_name: String,
    pub peer: Peer<RoleClient>,
    pub timeout: std::time::Duration,
    pub risk: ToolRisk,
    pub category: ToolCategory,
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        self.schema.clone()
    }

    fn concurrency(&self) -> Concurrency {
        // rmcp multiplexes concurrent requests over one connection.
        Concurrency::Parallel
    }

    fn activity_description(&self, _arguments: &Value) -> String {
        format!(
            "Calling MCP tool {} on {}",
            self.remote_name, self.server_name
        )
    }

    fn risk(&self, _arguments: &Value) -> ToolRisk {
        self.risk
    }

    fn category(&self) -> ToolCategory {
        self.category
    }

    async fn execute(&self, arguments: Value, ctx: ExecutionContext) -> ToolResult {
        let params = CallToolRequestParams::new(self.remote_name.clone())
            .with_arguments(arguments.as_object().cloned().unwrap_or_default());
        tokio::select! {
            _ = ctx.cancel.cancelled() => {
                ToolResult::error("MCP tool call cancelled")
            }
            res = tokio::time::timeout(self.timeout, self.peer.call_tool(params)) => {
                match res {
                    Err(_elapsed) => ToolResult::error(format!(
                        "MCP tool '{}' on '{}' timed out after {}s",
                        self.remote_name,
                        self.server_name,
                        self.timeout.as_secs()
                    )),
                    Ok(Err(e)) => ToolResult::error(format!(
                        "MCP tool '{}' on '{}' failed: {e}",
                        self.remote_name, self.server_name
                    )),
                    Ok(Ok(result)) => content::map_call_tool_result(result),
                }
            }
        }
    }
}
