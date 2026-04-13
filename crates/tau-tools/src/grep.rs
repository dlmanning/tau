//! Grep content search tool
use crate::cached_schema;

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tau_agent::tool::{ExecutionContext, Tool, ToolResult};

/// Maximum file size to search (10 MB) — skip larger files to avoid OOM
const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024;

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

    fn activity_description(&self, arguments: &serde_json::Value) -> String {
        let pattern = arguments
            .get("pattern")
            .and_then(|v| v.as_str())
            .unwrap_or("...");
        format!("Searching for \"{}\"", pattern)
    }

    fn description(&self) -> &str {
        "Search for a pattern in files. Returns matching lines with file paths and line numbers."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(GrepArgs)
    }

    async fn execute(&self, arguments: serde_json::Value, ctx: ExecutionContext) -> ToolResult {
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
    if path.is_file() {
        return vec![path.to_path_buf()];
    }

    let mut builder = ignore::WalkBuilder::new(path);
    builder
        .hidden(true) // skip hidden files by default
        .git_ignore(true) // respect .gitignore
        .git_exclude(true) // respect .git/info/exclude
        .git_global(true); // respect global gitignore

    // Apply glob filter if provided — use overrides so path globs like
    // "src/**/*.ts" work, not just filename patterns.
    if let Some(g) = glob_pattern {
        let mut overrides = ignore::overrides::OverrideBuilder::new(path);
        // Whitelist only files matching the glob; everything else is excluded.
        if overrides.add(g).is_ok() {
            if let Ok(ov) = overrides.build() {
                builder.overrides(ov);
            }
        }
    }

    let mut files = Vec::new();
    for entry in builder.build().flatten() {
        if entry.file_type().is_some_and(|ft| ft.is_file()) {
            files.push(entry.into_path());
        }
    }
    files
}

/// Truncate a line if it exceeds MAX_LINE_LENGTH (Unicode-safe)
fn truncate_line(line: &str) -> String {
    crate::truncate_chars(line, MAX_LINE_LENGTH)
}

/// Check if a file appears to be binary by reading the first 8KB and looking
/// for null bytes, which don't appear in text files.
fn is_binary(path: &Path) -> std::io::Result<bool> {
    use std::io::Read;
    let mut buf = [0u8; 8192];
    let mut file = File::open(path)?;
    let n = file.read(&mut buf)?;
    Ok(buf[..n].contains(&0))
}

fn search_file(
    path: &PathBuf,
    regex: &regex::Regex,
    context_lines: usize,
) -> std::io::Result<Vec<String>> {
    // Skip files that are too large to avoid OOM
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > MAX_FILE_SIZE {
        return Ok(Vec::new());
    }

    // Skip binary files
    if is_binary(path)? {
        return Ok(Vec::new());
    }

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
