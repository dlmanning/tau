//! List directory tool

use async_trait::async_trait;
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use tau_agent::tool::{Tool, ToolResult};
use tokio_util::sync::CancellationToken;

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

    fn description(&self) -> &str {
        "List contents of a directory with file metadata."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (defaults to current directory)"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to list recursively (default: false)"
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Whether to show hidden files (default: false)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of entries to return (default: 100)"
                }
            },
            "required": []
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        arguments: serde_json::Value,
        cancel: CancellationToken,
    ) -> ToolResult {
        let path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));

        let recursive = arguments
            .get("recursive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let show_hidden = arguments
            .get("show_hidden")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let limit = arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(100) as usize;

        if !path.exists() {
            return ToolResult::error(format!("Path does not exist: {}", path.display()));
        }

        if !path.is_dir() {
            return ToolResult::error(format!("Path is not a directory: {}", path.display()));
        }

        let mut entries = Vec::new();

        if recursive {
            collect_recursive(&path, &path, show_hidden, &cancel, &mut entries, limit);
        } else {
            collect_flat(&path, show_hidden, &cancel, &mut entries, limit);
        }

        if cancel.is_cancelled() {
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

        // Skip hidden files unless requested
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

        // Skip hidden files and common large directories
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

        // Recurse into directories
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
