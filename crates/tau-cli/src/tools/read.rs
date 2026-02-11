//! File reading tool

use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;
use tau_agent::tool::{Tool, ToolResult};
use tokio::fs;
use tokio_util::sync::CancellationToken;

const MAX_LINES: usize = 2000;
const MAX_LINE_LENGTH: usize = 2000;

/// Tool for reading file contents
pub struct ReadTool;

impl ReadTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReadTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Supports text files. For large files, use offset and limit parameters."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative or absolute)"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        arguments: serde_json::Value,
        cancel: CancellationToken,
    ) -> ToolResult {
        let path_str = match arguments.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::error("Missing 'path' argument"),
        };

        // Expand ~ to home directory
        let path = if let Some(stripped) = path_str.strip_prefix("~/") {
            if let Some(home) = dirs::home_dir() {
                home.join(stripped)
            } else {
                PathBuf::from(path_str)
            }
        } else if path_str == "~" {
            dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
        } else {
            PathBuf::from(path_str)
        };

        // Check for cancellation
        if cancel.is_cancelled() {
            return ToolResult::error("Operation cancelled");
        }

        // Read the file
        let content = match fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("Failed to read file: {}", e)),
        };

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        // Parse offset and limit
        let offset = arguments
            .get("offset")
            .and_then(|v| v.as_u64())
            .map(|o| (o as usize).saturating_sub(1)) // 1-indexed to 0-indexed
            .unwrap_or(0);

        let limit = arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|l| l as usize)
            .unwrap_or(MAX_LINES);

        // Check bounds
        if offset >= total_lines {
            return ToolResult::error(format!(
                "Offset {} is beyond end of file ({} lines total)",
                offset + 1,
                total_lines
            ));
        }

        let end = (offset + limit).min(total_lines);
        let selected_lines = &lines[offset..end];

        // Truncate long lines
        let mut had_truncated = false;
        let formatted: Vec<String> = selected_lines
            .iter()
            .map(|line| {
                if line.len() > MAX_LINE_LENGTH {
                    had_truncated = true;
                    line[..MAX_LINE_LENGTH].to_string()
                } else {
                    line.to_string()
                }
            })
            .collect();

        let mut output = formatted.join("\n");

        // Add notices
        let mut notices = Vec::new();
        if had_truncated {
            notices.push(format!(
                "Some lines were truncated to {} characters",
                MAX_LINE_LENGTH
            ));
        }
        if end < total_lines {
            let remaining = total_lines - end;
            notices.push(format!(
                "{} more lines not shown. Use offset={} to continue reading",
                remaining,
                end + 1
            ));
        }

        if !notices.is_empty() {
            output.push_str(&format!("\n\n... ({})", notices.join(". ")));
        }

        ToolResult::text(output)
    }
}
