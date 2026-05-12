//! File editing tool
use crate::cached_schema;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use similar::{ChangeTag, TextDiff};
use tau_agent::AgentEvent;
use tau_agent::{Concurrency, ExecutionContext, Tool, ToolCategory, ToolResult};
use tokio::fs;

#[derive(Deserialize, JsonSchema)]
struct EditArgs {
    /// Path to the file to edit (relative or absolute)
    path: String,
    /// Exact text to find and replace (must match exactly)
    old_text: String,
    /// New text to replace the old text with
    new_text: String,
    /// Replace all occurrences instead of requiring a unique match (default: false)
    replace_all: Option<bool>,
}

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

    fn activity_description(&self, arguments: &serde_json::Value) -> String {
        let name = crate::short_filename(arguments);
        format!("Editing {}", name)
    }

    fn description(&self) -> &str {
        "Edit a file by replacing exact text. The old_text must match exactly (including whitespace). Use this for precise, surgical edits."
    }

    fn concurrency(&self) -> Concurrency {
        Concurrency::Sequential
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(EditArgs)
    }

    fn category(&self) -> ToolCategory {
        ToolCategory::Edit
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let args: EditArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };

        let path = ctx.resolve_path(&args.path);

        if ctx.cancel.is_cancelled() {
            return ToolResult::error("Operation cancelled");
        }

        if let Err(e) = ctx.require_read(&path) {
            return ToolResult::error(e);
        }

        let content = match fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("Failed to read file: {}", e)),
        };

        let replace_all = args.replace_all.unwrap_or(false);

        if !content.contains(&args.old_text) {
            return ToolResult::error(format!(
                "Could not find the exact text in {}. The old text must match exactly including all whitespace and newlines.",
                args.path
            ));
        }

        let occurrences = content.matches(&args.old_text).count();

        let new_content = if replace_all {
            content.replace(&args.old_text, &args.new_text)
        } else if occurrences > 1 {
            return ToolResult::error(format!(
                "Found {} occurrences of the text in {}. The text must be unique. Please provide more context to make it unique, or set replace_all to true.",
                occurrences, args.path
            ));
        } else {
            content.replacen(&args.old_text, &args.new_text, 1)
        };

        if content == new_content {
            return ToolResult::error(format!(
                "No changes made to {}. The replacement produced identical content.",
                args.path
            ));
        }

        let diff = generate_diff(&content, &new_content);

        if ctx.cancel.is_cancelled() {
            return ToolResult::error("Operation cancelled");
        }

        match fs::write(&path, &new_content).await {
            Ok(()) => {
                ctx.progress.emit(AgentEvent::FileChanged {
                    path: path.clone(),
                    before: Some(content.clone()),
                    after: Some(new_content.clone()),
                    tool_call_id: ctx.progress.tool_call_id().to_string(),
                });
                let result = if replace_all && occurrences > 1 {
                    format!(
                        "Successfully replaced {} occurrences in {}.\n\nDiff:\n{}",
                        occurrences, args.path, diff
                    )
                } else {
                    format!(
                        "Successfully replaced text in {}. Changed {} characters to {} characters.\n\nDiff:\n{}",
                        args.path,
                        args.old_text.len(),
                        args.new_text.len(),
                        diff
                    )
                };
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
