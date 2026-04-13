//! File reading tool
use crate::cached_schema;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tau_agent::tool::{ExecutionContext, Tool, ToolResult};
use tokio::fs;

const MAX_LINES: usize = 2000;
const MAX_LINE_LENGTH: usize = 2000;

#[derive(Deserialize, JsonSchema)]
struct ReadArgs {
    /// Path to the file to read (relative or absolute)
    path: String,
    /// Line number to start reading from (1-indexed)
    offset: Option<u64>,
    /// Maximum number of lines to read
    limit: Option<u64>,
}

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

    fn activity_description(&self, arguments: &serde_json::Value) -> String {
        let name = crate::short_filename(arguments);
        format!("Reading {}", name)
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Output uses cat -n format with line numbers. For large files, use offset and limit parameters."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(ReadArgs)
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let args: ReadArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };

        let path = ctx.resolve_path(&args.path);

        if ctx.cancel.is_cancelled() {
            return ToolResult::error("Operation cancelled");
        }

        let content = match fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("Failed to read file: {}", e)),
        };

        ctx.mark_read(&path);

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let offset = args
            .offset
            .map(|o| (o as usize).saturating_sub(1)) // 1-indexed to 0-indexed
            .unwrap_or(0);

        let limit = args.limit.map(|l| l as usize).unwrap_or(MAX_LINES);

        if offset >= total_lines {
            return ToolResult::error(format!(
                "Offset {} is beyond end of file ({} lines total)",
                offset + 1,
                total_lines
            ));
        }

        let end = (offset + limit).min(total_lines);
        let selected_lines = &lines[offset..end];

        // Format with line numbers (cat -n style) and truncate long lines
        let mut had_truncated = false;
        let num_width = total_lines.max(1).to_string().len().max(6);
        let formatted: Vec<String> = selected_lines
            .iter()
            .enumerate()
            .map(|(i, line)| {
                let line_num = offset + i + 1; // 1-indexed
                let truncated = line.chars().count() > MAX_LINE_LENGTH;
                if truncated {
                    had_truncated = true;
                }
                let content: String = if truncated {
                    line.chars().take(MAX_LINE_LENGTH).collect()
                } else {
                    line.to_string()
                };
                format!("{:>width$}\t{}", line_num, content, width = num_width)
            })
            .collect();

        let mut output = formatted.join("\n");

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
