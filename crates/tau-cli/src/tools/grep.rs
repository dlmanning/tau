//! Grep content search tool

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tau_agent::tool::{ExecutionContext, Tool, ToolResult};

/// Maximum matches to return by default
const DEFAULT_LIMIT: usize = 50;
/// Maximum length of a matching line before truncation
const MAX_LINE_LENGTH: usize = 500;

#[derive(Deserialize, JsonSchema)]
struct GrepArgs {
    /// The regex pattern to search for
    pattern: String,
    /// File or directory to search in (defaults to current directory)
    path: Option<String>,
    /// Glob pattern to filter files (e.g., '*.rs', '**/*.ts')
    glob: Option<String>,
    /// Whether to ignore case (default: false)
    case_insensitive: Option<bool>,
    /// Maximum number of matches to return (default: 50)
    limit: Option<u64>,
    /// Number of context lines before and after match (default: 0)
    context: Option<u64>,
}

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
        cached_schema!(GrepArgs)
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        ctx: ExecutionContext,
    ) -> ToolResult {
        let args: GrepArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };

        let case_insensitive = args.case_insensitive.unwrap_or(false);

        let regex_pattern = if case_insensitive {
            format!("(?i){}", args.pattern)
        } else {
            args.pattern.clone()
        };

        let regex = match regex::Regex::new(&regex_pattern) {
            Ok(r) => r,
            Err(e) => return ToolResult::error(format!("Invalid regex pattern: {}", e)),
        };

        let path = args
            .path
            .as_deref()
            .map(|p| ctx.resolve_path(p))
            .unwrap_or_else(|| ctx.cwd.clone());

        let glob_pattern = args.glob.as_deref();

        let limit = args.limit.unwrap_or(DEFAULT_LIMIT as u64) as usize;

        let context_lines = args.context.unwrap_or(0) as usize;

        let files = collect_files(&path, glob_pattern);

        let mut matches = Vec::new();
        let mut total_matches = 0;

        for file_path in files {
            if ctx.cancel.is_cancelled() {
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

    let pattern = match glob_pattern {
        Some(g) => path.join(g).to_string_lossy().to_string(),
        None => path.join("**/*").to_string_lossy().to_string(),
    };

    if let Ok(entries) = glob::glob(&pattern) {
        for entry in entries.flatten() {
            if entry.is_file() {
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
                let start = line_num.saturating_sub(context_lines);
                for (i, line_content) in lines.iter().enumerate().take(line_num).skip(start) {
                    matches.push(format!(
                        "{}:{}: {}",
                        display_path,
                        i + 1,
                        truncate_line(line_content)
                    ));
                }

                matches.push(format!(
                    "{}:{}> {}",
                    display_path,
                    line_num + 1,
                    truncate_line(line)
                ));

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
