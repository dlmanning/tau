//! List directory tool
use crate::cached_schema;

use std::{
    fs,
    path::{Path, PathBuf},
};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tau_agent::{ExecutionContext, Tool, ToolResult};
use tokio_util::sync::CancellationToken;

#[derive(Deserialize, JsonSchema)]
struct ListArgs {
    /// Directory path to list (defaults to current directory)
    path: Option<String>,
    /// Whether to list recursively (default: false)
    recursive: Option<bool>,
    /// Whether to show hidden files (default: false)
    show_hidden: Option<bool>,
    /// Maximum number of entries to return (default: 100)
    limit: Option<u64>,
}

/// Tool for listing directory contents
pub struct ListTool;

impl ListTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ListTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ListTool {
    fn name(&self) -> &str {
        "list"
    }

    fn activity_description(&self, _arguments: &serde_json::Value) -> String {
        "Listing directory".to_string()
    }

    fn description(&self) -> &str {
        "List contents of a directory with file metadata."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(ListArgs)
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
        let args: ListArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };

        let path = args
            .path
            .as_deref()
            .map(|p| ctx.resolve_path(p))
            .unwrap_or_else(|| ctx.cwd.clone());

        let recursive = args.recursive.unwrap_or(false);
        let show_hidden = args.show_hidden.unwrap_or(false);
        let limit = args.limit.unwrap_or(100) as usize;

        if !path.exists() {
            return ToolResult::error(format!("Path does not exist: {}", path.display()));
        }

        if !path.is_dir() {
            return ToolResult::error(format!("Path is not a directory: {}", path.display()));
        }

        let mut entries = Vec::new();

        if recursive {
            collect_recursive(&path, &path, show_hidden, &ctx.cancel, &mut entries, limit);
        } else {
            collect_flat(&path, show_hidden, &ctx.cancel, &mut entries, limit);
        }

        if ctx.cancel.is_cancelled() {
            return ToolResult::error("List cancelled");
        }

        if entries.is_empty() {
            return ToolResult::text("(empty directory)");
        }

        let truncated = entries.len() >= limit;
        let mut output = entries.join("\n");

        if truncated {
            output.push_str(&format!("\n\n(showing first {} entries)", limit));
        }

        ToolResult::text(output)
    }
}

fn collect_flat(
    path: &PathBuf,
    show_hidden: bool,
    cancel: &CancellationToken,
    entries: &mut Vec<String>,
    limit: usize,
) {
    let read_dir = match fs::read_dir(path) {
        Ok(d) => d,
        Err(e) => {
            entries.push(format!("Error reading directory: {}", e));
            return;
        }
    };

    let mut items: Vec<_> = read_dir.flatten().collect();
    items.sort_by_key(|e| e.path());

    for entry in items {
        if cancel.is_cancelled() || entries.len() >= limit {
            break;
        }

        let name = entry.file_name().to_string_lossy().to_string();

        if !show_hidden && name.starts_with('.') {
            continue;
        }

        let metadata = entry.metadata();
        let entry_str = format_entry(&name, &entry.path(), metadata.ok().as_ref());
        entries.push(entry_str);
    }
}

fn collect_recursive(
    base: &PathBuf,
    path: &PathBuf,
    show_hidden: bool,
    cancel: &CancellationToken,
    entries: &mut Vec<String>,
    limit: usize,
) {
    if cancel.is_cancelled() || entries.len() >= limit {
        return;
    }

    let read_dir = match fs::read_dir(path) {
        Ok(d) => d,
        Err(_) => return,
    };

    let mut items: Vec<_> = read_dir.flatten().collect();
    items.sort_by_key(|e| e.path());

    for entry in items {
        if cancel.is_cancelled() || entries.len() >= limit {
            break;
        }

        let name = entry.file_name().to_string_lossy().to_string();

        if !show_hidden && name.starts_with('.') {
            continue;
        }
        if name == "node_modules" || name == "target" || name == ".git" {
            continue;
        }

        let full_path = entry.path();
        let relative = full_path.strip_prefix(base).unwrap_or(&full_path);
        let metadata = entry.metadata();

        let entry_str = format_entry(
            &relative.to_string_lossy(),
            &full_path,
            metadata.ok().as_ref(),
        );
        entries.push(entry_str);

        if full_path.is_dir() {
            collect_recursive(base, &full_path, show_hidden, cancel, entries, limit);
        }
    }
}

fn format_entry(name: &str, path: &Path, metadata: Option<&fs::Metadata>) -> String {
    let type_indicator = if path.is_dir() { "/" } else { "" };

    match metadata {
        Some(m) => {
            let size = if path.is_file() {
                format_size(m.len())
            } else {
                "-".to_string()
            };
            format!("{}{}\t{}", name, type_indicator, size)
        }
        None => format!("{}{}", name, type_indicator),
    }
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1}G", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1}M", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1}K", bytes as f64 / KB as f64)
    } else {
        format!("{}B", bytes)
    }
}
