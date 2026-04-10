//! File writing tool

use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::json;
use tau_agent::tool::{ExecutionContext, Tool, ToolResult};
use tokio::fs;

/// Tool for writing file contents
pub struct WriteTool;

impl WriteTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for WriteTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write (relative or absolute)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        ctx: ExecutionContext,
    ) -> ToolResult {
        let path_str = match arguments.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::error("Missing 'path' argument"),
        };

        let content = match arguments.get("content").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return ToolResult::error("Missing 'content' argument"),
        };

        let path = if let Some(stripped) = path_str.strip_prefix("~/") {
            if let Some(home) = dirs::home_dir() {
                home.join(stripped)
            } else {
                PathBuf::from(path_str)
            }
        } else if path_str == "~" {
            return ToolResult::error("Cannot write to home directory itself");
        } else {
            super::resolve_path(path_str, &ctx.cwd)
        };

        if ctx.cancel.is_cancelled() {
            return ToolResult::error("Operation cancelled");
        }

        if let Some(parent) = path.parent() {
            if !parent.exists() {
                if let Err(e) = fs::create_dir_all(parent).await {
                    return ToolResult::error(format!("Failed to create directory: {}", e));
                }
            }
        }

        match fs::write(&path, content).await {
            Ok(()) => ToolResult::text(format!(
                "Successfully wrote {} bytes to {}",
                content.len(),
                path_str
            )),
            Err(e) => ToolResult::error(format!("Failed to write file: {}", e)),
        }
    }
}
