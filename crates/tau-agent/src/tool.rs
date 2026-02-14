//! Tool trait and execution

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tau_ai::Content;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::events::AgentEvent;

/// Result of a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Content to return to the LLM
    pub content: Vec<Content>,
    /// Whether the execution resulted in an error
    pub is_error: bool,
    /// Optional structured details (for UI rendering)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl ToolResult {
    /// Create a successful text result
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![Content::text(text)],
            is_error: false,
            details: None,
        }
    }

    /// Create an error result
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![Content::text(message)],
            is_error: true,
            details: None,
        }
    }

    /// Create a result with multiple content blocks
    pub fn with_content(content: Vec<Content>) -> Self {
        Self {
            content,
            is_error: false,
            details: None,
        }
    }

    /// Add details to the result
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }

    /// Get the text content as a single string
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| c.as_text())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A sender for tool progress updates during execution.
///
/// Tools can use this to emit `ToolExecutionUpdate` events while running.
#[derive(Clone)]
pub struct ProgressSender {
    tx: broadcast::Sender<AgentEvent>,
    tool_call_id: String,
    tool_name: String,
}

impl ProgressSender {
    /// Create a new progress sender for a specific tool invocation.
    pub fn new(
        tx: broadcast::Sender<AgentEvent>,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> Self {
        Self {
            tx,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
        }
    }

    /// Send a progress update.
    pub fn send(&self, content: impl Into<String>) {
        let _ = self.tx.send(AgentEvent::ToolExecutionUpdate {
            tool_call_id: self.tool_call_id.clone(),
            tool_name: self.tool_name.clone(),
            content: content.into(),
        });
    }
}

/// Trait for executable tools
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (used in API calls)
    fn name(&self) -> &str;

    /// Human-readable label for UI
    fn label(&self) -> &str {
        self.name()
    }

    /// Tool description for the LLM
    fn description(&self) -> &str;

    /// JSON Schema for parameters
    fn parameters_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given arguments
    async fn execute(
        &self,
        tool_call_id: &str,
        arguments: serde_json::Value,
        cancel: CancellationToken,
    ) -> ToolResult;

    /// Execute the tool with progress reporting support.
    ///
    /// Default implementation ignores the progress sender and delegates to `execute()`.
    /// Tools can override this to emit progress updates during execution.
    async fn execute_with_progress(
        &self,
        tool_call_id: &str,
        arguments: serde_json::Value,
        cancel: CancellationToken,
        _progress: ProgressSender,
    ) -> ToolResult {
        self.execute(tool_call_id, arguments, cancel).await
    }
}

/// Type alias for a boxed tool
pub type BoxedTool = Arc<dyn Tool>;

/// Convert a Tool to a tau_ai::Tool for API calls
pub fn to_api_tool(tool: &dyn Tool) -> tau_ai::Tool {
    tau_ai::Tool {
        name: tool.name().to_string(),
        description: tool.description().to_string(),
        parameters: tool.parameters_schema(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple test tool that echoes its arguments.
    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echoes input"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string" }
                }
            })
        }
        async fn execute(
            &self,
            _tool_call_id: &str,
            arguments: serde_json::Value,
            _cancel: CancellationToken,
        ) -> ToolResult {
            let text = arguments
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("(empty)");
            ToolResult::text(text)
        }
    }

    #[tokio::test]
    async fn test_execute_with_progress_default_delegates() {
        let tool = EchoTool;
        let (tx, _rx) = broadcast::channel(16);
        let progress = ProgressSender::new(tx, "call_1", "echo");
        let cancel = CancellationToken::new();
        let args = serde_json::json!({"text": "hello"});

        let result = tool
            .execute_with_progress("call_1", args, cancel, progress)
            .await;

        assert!(!result.is_error);
        assert_eq!(result.text_content(), "hello");
    }

    #[tokio::test]
    async fn test_progress_sender_emits_events() {
        let (tx, mut rx) = broadcast::channel(16);
        let sender = ProgressSender::new(tx, "call_42", "bash");

        sender.send("50% complete");
        sender.send("done");

        let event1 = rx.recv().await.unwrap();
        let event2 = rx.recv().await.unwrap();

        match event1 {
            AgentEvent::ToolExecutionUpdate {
                tool_call_id,
                tool_name,
                content,
            } => {
                assert_eq!(tool_call_id, "call_42");
                assert_eq!(tool_name, "bash");
                assert_eq!(content, "50% complete");
            }
            other => panic!("expected ToolExecutionUpdate, got {:?}", other),
        }

        match event2 {
            AgentEvent::ToolExecutionUpdate { content, .. } => {
                assert_eq!(content, "done");
            }
            other => panic!("expected ToolExecutionUpdate, got {:?}", other),
        }
    }

    #[test]
    fn test_tool_result_text() {
        let r = ToolResult::text("ok");
        assert!(!r.is_error);
        assert_eq!(r.text_content(), "ok");
    }

    #[test]
    fn test_tool_result_error() {
        let r = ToolResult::error("bad");
        assert!(r.is_error);
        assert_eq!(r.text_content(), "bad");
    }

    #[test]
    fn test_to_api_tool() {
        let tool = EchoTool;
        let api_tool = to_api_tool(&tool);
        assert_eq!(api_tool.name, "echo");
        assert_eq!(api_tool.description, "Echoes input");
    }
}
