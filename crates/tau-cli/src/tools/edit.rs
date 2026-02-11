//! File editing tool

use async_trait::async_trait;
use serde_json::json;
use similar::{ChangeTag, TextDiff};
use std::path::PathBuf;
use tau_agent::tool::{Tool, ToolResult};
use tokio::fs;
use tokio_util::sync::CancellationToken;

/// Tool for editing files with find/replace
pub struct EditTool;

impl EditTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EditTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing exact text. The old_text must match exactly (including whitespace). Use this for precise, surgical edits."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit (relative or absolute)"
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find and replace (must match exactly)"
                },
                "new_text": {
                    "type": "string",
                    "description": "New text to replace the old text with"
                }
            },
            "required": ["path", "old_text", "new_text"]
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

        let old_text = match arguments.get("old_text").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => return ToolResult::error("Missing 'old_text' argument"),
        };

        let new_text = match arguments.get("new_text").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => return ToolResult::error("Missing 'new_text' argument"),
        };

        // Expand ~ to home directory
        let path = if let Some(rest) = path_str.strip_prefix("~/") {
            if let Some(home) = dirs::home_dir() {
                home.join(rest)
            } else {
                PathBuf::from(path_str)
            }
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

        // Check if old text exists
        if !content.contains(old_text) {
            return ToolResult::error(format!(
                "Could not find the exact text in {}. The old text must match exactly including all whitespace and newlines.",
                path_str
            ));
        }

        // Count occurrences
        let occurrences = content.matches(old_text).count();
        if occurrences > 1 {
            return ToolResult::error(format!(
                "Found {} occurrences of the text in {}. The text must be unique. Please provide more context to make it unique.",
                occurrences, path_str
            ));
        }

        // Perform replacement
        let new_content = content.replacen(old_text, new_text, 1);

        // Check that something changed
        if content == new_content {
            return ToolResult::error(format!(
                "No changes made to {}. The replacement produced identical content.",
                path_str
            ));
        }

        // Generate diff for output
        let diff = generate_diff(&content, &new_content);

        // Check for cancellation before writing
        if cancel.is_cancelled() {
            return ToolResult::error("Operation cancelled");
        }

        // Write the file
        match fs::write(&path, &new_content).await {
            Ok(()) => {
                let result = format!(
                    "Successfully replaced text in {}. Changed {} characters to {} characters.\n\nDiff:\n{}",
                    path_str,
                    old_text.len(),
                    new_text.len(),
                    diff
                );
                ToolResult::text(result).with_details(json!({ "diff": diff }))
            }
            Err(e) => ToolResult::error(format!("Failed to write file: {}", e)),
        }
    }
}

/// Generate a unified diff string
fn generate_diff(old: &str, new: &str) -> String {
    let diff = TextDiff::from_lines(old, new);
    let mut output = Vec::new();

    for change in diff.iter_all_changes() {
        let sign = match change.tag() {
            ChangeTag::Delete => "-",
            ChangeTag::Insert => "+",
            ChangeTag::Equal => " ",
        };
        output.push(format!("{}{}", sign, change));
    }

    // Limit output to avoid huge diffs
    if output.len() > 50 {
        output.truncate(50);
        output.push("... (diff truncated)".to_string());
    }

    output.join("")
}
