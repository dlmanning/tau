//! File writing tool

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tau_agent::tool::{Concurrency, ExecutionContext, Tool, ToolResult};
use tokio::fs;

#[derive(Deserialize, JsonSchema)]
struct WriteArgs {
    /// Path to the file to write (relative or absolute)
    path: String,
    /// Content to write to the file
    content: String,
}

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

    fn concurrency(&self) -> Concurrency {
        Concurrency::Sequential
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(WriteArgs)
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        ctx: ExecutionContext,
    ) -> ToolResult {
        let args: WriteArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };

        if args.path == "~" {
            return ToolResult::error("Cannot write to home directory itself");
        }
        let path = ctx.resolve_path(&args.path);

        if ctx.cancel.is_cancelled() {
            return ToolResult::error("Operation cancelled");
        }

        if let Err(e) = ctx.require_read(&path) {
            return ToolResult::error(e);
        }

        if let Some(parent) = path.parent() {
            if !parent.exists() {
                if let Err(e) = fs::create_dir_all(parent).await {
                    return ToolResult::error(format!("Failed to create directory: {}", e));
                }
            }
        }

        match fs::write(&path, &args.content).await {
            Ok(()) => ToolResult::text(format!(
                "Successfully wrote {} bytes to {}",
                args.content.len(),
                args.path
            )),
            Err(e) => ToolResult::error(format!("Failed to write file: {}", e)),
        }
    }
}
