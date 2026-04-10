//! Glob file pattern matching tool

use std::path::PathBuf;

use async_trait::async_trait;
use glob::glob;
use serde_json::json;
use tau_agent::tool::{ExecutionContext, Tool, ToolResult};

/// Tool for finding files matching a glob pattern
pub struct GlobTool;

impl GlobTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GlobTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern. Supports patterns like '**/*.rs', 'src/*.ts', etc."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to match (e.g., '**/*.rs', 'src/**/*.ts')"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the pattern (optional, defaults to current directory)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (optional, defaults to 100)"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        ctx: ExecutionContext,
    ) -> ToolResult {
        let pattern = match arguments.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::error("Missing 'pattern' argument"),
        };

        let cwd = arguments
            .get("cwd")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .unwrap_or_else(|| ctx.cwd.clone());
        let cwd = Some(cwd);

        let limit = arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;

        let full_pattern = match &cwd {
            Some(dir) => dir.join(pattern).to_string_lossy().to_string(),
            None => pattern.to_string(),
        };

        let entries = match glob(&full_pattern) {
            Ok(paths) => paths,
            Err(e) => return ToolResult::error(format!("Invalid glob pattern: {}", e)),
        };

        let mut results = Vec::new();
        for entry in entries {
            if ctx.cancel.is_cancelled() {
                return ToolResult::error("Glob cancelled");
            }

            match entry {
                Ok(path) => {
                    results.push(path.display().to_string());
                    if results.len() >= limit {
                        break;
                    }
                }
                Err(e) => {
                    tracing::debug!("Glob entry error: {}", e);
                }
            }
        }

        if results.is_empty() {
            return ToolResult::text("No files matched the pattern");
        }

        let truncated = results.len() >= limit;
        let mut output = results.join("\n");

        if truncated {
            output.push_str(&format!("\n\n(showing first {} results)", limit));
        }

        ToolResult::text(output)
    }
}
