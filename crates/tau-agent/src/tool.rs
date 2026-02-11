//! Tool trait and execution

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tau_ai::Content;
use tokio_util::sync::CancellationToken;

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
