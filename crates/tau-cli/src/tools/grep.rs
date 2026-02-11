//! Grep content search tool

use async_trait::async_trait;
use serde_json::json;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use tau_agent::tool::{Tool, ToolResult};
use tokio_util::sync::CancellationToken;

/// Maximum matches to return by default
const DEFAULT_LIMIT: usize = 50;
/// Maximum length of a matching line before truncation
const MAX_LINE_LENGTH: usize = 500;

/// Tool for searching file contents with regex
pub struct GrepTool;

impl GrepTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GrepTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for a pattern in files. Returns matching lines with file paths and line numbers."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (defaults to current directory)"
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.rs', '**/*.ts')"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Whether to ignore case (default: false)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (default: 50)"
                },
                "context": {
                    "type": "integer",
                    "description": "Number of context lines before and after match (default: 0)"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        arguments: serde_json::Value,
        cancel: CancellationToken,
    ) -> ToolResult {
        let pattern_str = match arguments.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::error("Missing 'pattern' argument"),
        };

        let case_insensitive = arguments
            .get("case_insensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let regex_pattern = if case_insensitive {
            format!("(?i){}", pattern_str)
        } else {
            pattern_str.to_string()
        };

        let regex = match regex::Regex::new(&regex_pattern) {
            Ok(r) => r,
            Err(e) => return ToolResult::error(format!("Invalid regex pattern: {}", e)),
        };

        let path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));

        let glob_pattern = arguments.get("glob").and_then(|v| v.as_str());

        let limit = arguments
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_LIMIT as u64) as usize;

        let context_lines = arguments
            .get("context")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // Collect files to search
        let files = collect_files(&path, glob_pattern);

        let mut matches = Vec::new();
        let mut total_matches = 0;

        for file_path in files {
            if cancel.is_cancelled() {
                return ToolResult::error("Search cancelled");
            }

            if let Ok(file_matches) = search_file(&file_path, &regex, context_lines) {
                for m in file_matches {
                    matches.push(m);
                    total_matches += 1;
                    if matches.len() >= limit {
                        break;
                    }
                }
            }

            if matches.len() >= limit {
                break;
            }
        }

        if matches.is_empty() {
            return ToolResult::text("No matches found");
        }

        let mut output = matches.join("\n");

        if total_matches >= limit {
            output.push_str(&format!("\n\n(showing first {} matches)", limit));
        }

        ToolResult::text(output)
    }
}

fn collect_files(path: &Path, glob_pattern: Option<&str>) -> Vec<PathBuf> {
    let mut files = Vec::new();

    if path.is_file() {
        files.push(path.to_path_buf());
        return files;
    }

    // Build glob pattern
    let pattern = match glob_pattern {
        Some(g) => path.join(g).to_string_lossy().to_string(),
        None => path.join("**/*").to_string_lossy().to_string(),
    };

    if let Ok(entries) = glob::glob(&pattern) {
        for entry in entries.flatten() {
            if entry.is_file() {
                // Skip binary files and hidden directories
                let path_str = entry.to_string_lossy();
                if !path_str.contains("/.git/")
                    && !path_str.contains("/node_modules/")
                    && !path_str.contains("/target/")
                {
                    files.push(entry);
                }
            }
        }
    }

    files
}

/// Truncate a line if it exceeds MAX_LINE_LENGTH
fn truncate_line(line: &str) -> String {
    if line.len() > MAX_LINE_LENGTH {
        format!("{}...", &line[..MAX_LINE_LENGTH])
    } else {
        line.to_string()
    }
}

fn search_file(
    path: &PathBuf,
    regex: &regex::Regex,
    context_lines: usize,
) -> std::io::Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;
    let mut matches = Vec::new();

    for (line_num, line) in lines.iter().enumerate() {
        if regex.is_match(line) {
            let display_path = path.display();

            if context_lines > 0 {
                // Add context before
                let start = line_num.saturating_sub(context_lines);
                for (i, line_content) in lines.iter().enumerate().take(line_num).skip(start) {
                    matches.push(format!(
                        "{}:{}: {}",
                        display_path,
                        i + 1,
                        truncate_line(line_content)
                    ));
                }

                // Add the matching line
                matches.push(format!(
                    "{}:{}> {}",
                    display_path,
                    line_num + 1,
                    truncate_line(line)
                ));

                // Add context after
                let end = (line_num + context_lines + 1).min(lines.len());
                for (i, line_content) in lines.iter().enumerate().take(end).skip(line_num + 1) {
                    matches.push(format!(
                        "{}:{}: {}",
                        display_path,
                        i + 1,
                        truncate_line(line_content)
                    ));
                }

                matches.push(String::new()); // Separator between matches
            } else {
                matches.push(format!(
                    "{}:{}: {}",
                    display_path,
                    line_num + 1,
                    truncate_line(line)
                ));
            }
        }
    }

    Ok(matches)
}
