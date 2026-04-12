//! Glob file pattern matching tool

use std::path::PathBuf;

use async_trait::async_trait;
use glob::glob;
use schemars::JsonSchema;
use serde::Deserialize;
use tau_agent::tool::{ExecutionContext, Tool, ToolResult};

#[derive(Deserialize, JsonSchema)]
struct GlobArgs {
    /// The glob pattern to match (e.g., '**/*.rs', 'src/**/*.ts')
    pattern: String,
    /// Working directory for the pattern (optional, defaults to current directory)
    cwd: Option<String>,
    /// Maximum number of results to return (optional, defaults to 100)
    limit: Option<u64>,
}

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

    fn activity_description(&self, arguments: &serde_json::Value) -> String {
        let pattern = arguments
            .get("pattern")
            .and_then(|v| v.as_str())
            .unwrap_or("...");
        format!("Finding {}", pattern)
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern. Supports patterns like '**/*.rs', 'src/*.ts', etc."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(GlobArgs)
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let args: GlobArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };

        let cwd = args
            .cwd
            .map(PathBuf::from)
            .unwrap_or_else(|| ctx.cwd.clone());

        let limit = args.limit.unwrap_or(100) as usize;

        let full_pattern = cwd.join(&args.pattern).to_string_lossy().to_string();

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
