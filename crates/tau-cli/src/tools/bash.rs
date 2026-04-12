//! Bash command execution tool

use std::{collections::VecDeque, process::Stdio};

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::Deserialize;
use tau_agent::tool::{Concurrency, ExecutionContext, Tool, ToolResult};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
};

/// Maximum output size in bytes before truncation
const MAX_OUTPUT_SIZE: usize = 30_000; // 30KB
/// Maximum number of lines before truncation
const MAX_OUTPUT_LINES: usize = 500;

#[derive(Deserialize, JsonSchema)]
struct BashArgs {
    /// The bash command to execute
    command: String,
    /// Timeout in seconds (optional)
    timeout: Option<u64>,
}

/// Tool for executing bash commands
pub struct BashTool;

impl BashTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BashTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Collects output lines with head+tail truncation.
/// Keeps the first half and last half of lines within size/line limits.
struct OutputCollector {
    /// Lines kept from the beginning of output
    head: Vec<String>,
    /// Rolling buffer of recent lines (the tail)
    tail: VecDeque<String>,
    /// Total bytes in head
    head_bytes: usize,
    /// Total bytes in tail
    tail_bytes: usize,
    /// Whether we've exceeded head capacity and started collecting tail
    head_full: bool,
    /// Total number of lines seen (for truncation notice)
    total_lines: usize,
    /// Max bytes for head portion
    max_head_bytes: usize,
    /// Max lines for head portion
    max_head_lines: usize,
    /// Max bytes for tail portion
    max_tail_bytes: usize,
    /// Max lines for tail portion
    max_tail_lines: usize,
}

impl OutputCollector {
    fn new() -> Self {
        let half_bytes = MAX_OUTPUT_SIZE / 2;
        let half_lines = MAX_OUTPUT_LINES / 2;
        Self {
            head: Vec::new(),
            tail: VecDeque::new(),
            head_bytes: 0,
            tail_bytes: 0,
            head_full: false,
            total_lines: 0,
            max_head_bytes: half_bytes,
            max_head_lines: half_lines,
            max_tail_bytes: half_bytes,
            max_tail_lines: half_lines,
        }
    }

    fn push_line(&mut self, line: String) {
        self.total_lines += 1;
        let line_len = line.len() + 1; // +1 for newline

        if !self.head_full {
            if self.head.len() < self.max_head_lines
                && self.head_bytes + line_len <= self.max_head_bytes
            {
                self.head_bytes += line_len;
                self.head.push(line);
                return;
            }
            self.head_full = true;
        }

        self.tail_bytes += line_len;
        self.tail.push_back(line);
        while self.tail.len() > self.max_tail_lines
            || (self.tail_bytes > self.max_tail_bytes && self.tail.len() > 1)
        {
            if let Some(evicted) = self.tail.pop_front() {
                self.tail_bytes -= evicted.len() + 1;
            }
        }
    }

    fn into_string(self) -> String {
        if !self.head_full {
            return self.head.join("\n");
        }

        let head_text = self.head.join("\n");
        let tail_count = self.tail.len();
        let tail_text: String = self.tail.into_iter().collect::<Vec<_>>().join("\n");
        let omitted = self.total_lines - self.head.len() - tail_count;
        format!(
            "{}\n\n... [{} lines truncated] ...\n\n{}",
            head_text, omitted, tail_text
        )
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn activity_description(&self, arguments: &serde_json::Value) -> String {
        let cmd = arguments
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("command");
        let short: String = cmd.chars().take(30).collect();
        format!("Running {}", short)
    }

    fn description(&self) -> &str {
        "Execute a bash command in the current working directory. Returns stdout and stderr."
    }

    fn concurrency(&self) -> Concurrency {
        Concurrency::Sequential
    }

    fn parameters_schema(&self) -> serde_json::Value {
        cached_schema!(BashArgs)
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        ctx: ExecutionContext,
    ) -> ToolResult {
        let args: BashArgs = match serde_json::from_value(arguments) {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Invalid arguments: {}", e)),
        };

        let command = &args.command;
        let timeout_secs = args.timeout.unwrap_or(120);

        let (shell, shell_arg) = if cfg!(target_os = "windows") {
            ("cmd", "/C")
        } else {
            ("sh", "-c")
        };

        let mut cmd = Command::new(shell);
        cmd.arg(shell_arg)
            .arg(command)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(&ctx.cwd);
        let mut child = match cmd.spawn()
        {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("Failed to spawn command: {}", e)),
        };

        let stdout = child.stdout.take().expect("stdout is piped");
        let stderr = child.stderr.take().expect("stderr is piped");

        let mut stdout_reader = BufReader::new(stdout).lines();
        let mut stderr_reader = BufReader::new(stderr).lines();

        let mut stdout_collector = OutputCollector::new();
        let mut stderr_collector = OutputCollector::new();
        let mut stdout_done = false;
        let mut stderr_done = false;

        let timeout = tokio::time::Duration::from_secs(timeout_secs);
        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            tokio::select! {
                _ = ctx.cancel.cancelled() => {
                    let _ = child.kill().await;
                    return ToolResult::error("Command cancelled");
                }
                _ = tokio::time::sleep_until(deadline) => {
                    let _ = child.kill().await;
                    let result = format!(
                        "{}\n{}\n\nCommand timed out after {} seconds",
                        stdout_collector.into_string(),
                        stderr_collector.into_string(),
                        timeout_secs
                    );
                    return ToolResult::error(result);
                }
                line = stdout_reader.next_line(), if !stdout_done => {
                    match line {
                        Ok(Some(l)) => {
                            stdout_collector.push_line(l);
                        }
                        Ok(None) => { stdout_done = true; }
                        Err(e) => {
                            stderr_collector.push_line(format!("Stdout read error: {}", e));
                            stdout_done = true;
                        }
                    }
                }
                line = stderr_reader.next_line(), if !stderr_done => {
                    match line {
                        Ok(Some(l)) => {
                            stderr_collector.push_line(l);
                        }
                        Ok(None) => { stderr_done = true; }
                        Err(e) => {
                            stderr_collector.push_line(format!("Stderr read error: {}", e));
                            stderr_done = true;
                        }
                    }
                }
                status = child.wait() => {
                    match status {
                        Ok(exit_status) => {
                            // Drain remaining pipe data. Use a bounded select so
                            // inherited fds from grandchild processes can't hang us.
                            let drain_timeout = tokio::time::Duration::from_secs(5);
                            tokio::select! {
                                _ = async {
                                    while let Ok(Some(l)) = stdout_reader.next_line().await {
                                        stdout_collector.push_line(l);
                                    }
                                    while let Ok(Some(l)) = stderr_reader.next_line().await {
                                        stderr_collector.push_line(l);
                                    }
                                } => {}
                                _ = ctx.cancel.cancelled() => {}
                                _ = tokio::time::sleep(drain_timeout) => {}
                            }

                            let mut result = stdout_collector.into_string();

                            let err_output = stderr_collector.into_string();
                            if !err_output.is_empty() {
                                if !result.is_empty() {
                                    result.push('\n');
                                }
                                result.push_str(&err_output);
                            }

                            if result.is_empty() {
                                result = "(no output)".to_string();
                            }

                            if exit_status.success() {
                                return ToolResult::text(result);
                            } else {
                                let code = exit_status.code().unwrap_or(-1);
                                return ToolResult::error(format!(
                                    "{}\n\nCommand exited with code {}",
                                    result, code
                                ));
                            }
                        }
                        Err(e) => {
                            return ToolResult::error(format!("Failed to wait for command: {}", e));
                        }
                    }
                }
            }
        }
    }
}
