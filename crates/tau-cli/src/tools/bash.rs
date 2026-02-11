//! Bash command execution tool

use async_trait::async_trait;
use serde_json::json;
use std::process::Stdio;
use tau_agent::tool::{Tool, ToolResult};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

/// Maximum output size in bytes before truncation
const MAX_OUTPUT_SIZE: usize = 100_000; // 100KB
/// Maximum number of lines before truncation
const MAX_OUTPUT_LINES: usize = 1000;

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

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command in the current working directory. Returns stdout and stderr."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (optional)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        arguments: serde_json::Value,
        cancel: CancellationToken,
    ) -> ToolResult {
        let command = match arguments.get("command").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return ToolResult::error("Missing 'command' argument"),
        };

        let timeout_secs = arguments
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(120);

        // Determine shell
        let (shell, shell_arg) = if cfg!(target_os = "windows") {
            ("cmd", "/C")
        } else {
            ("sh", "-c")
        };

        let mut child = match Command::new(shell)
            .arg(shell_arg)
            .arg(command)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("Failed to spawn command: {}", e)),
        };

        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();

        let mut stdout_reader = BufReader::new(stdout).lines();
        let mut stderr_reader = BufReader::new(stderr).lines();

        let mut output = String::new();
        let mut error_output = String::new();
        let mut stdout_lines = 0usize;
        let mut stderr_lines = 0usize;
        let mut stdout_truncated = false;
        let mut stderr_truncated = false;

        let timeout = tokio::time::Duration::from_secs(timeout_secs);
        let deadline = tokio::time::Instant::now() + timeout;

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    let _ = child.kill().await;
                    return ToolResult::error("Command cancelled");
                }
                _ = tokio::time::sleep_until(deadline) => {
                    let _ = child.kill().await;
                    let result = format!(
                        "{}\n{}\n\nCommand timed out after {} seconds",
                        output, error_output, timeout_secs
                    );
                    return ToolResult::error(result);
                }
                line = stdout_reader.next_line() => {
                    match line {
                        Ok(Some(l)) => {
                            // Check truncation limits
                            if stdout_truncated {
                                continue; // Skip remaining lines
                            }
                            if stdout_lines >= MAX_OUTPUT_LINES || output.len() + l.len() > MAX_OUTPUT_SIZE {
                                stdout_truncated = true;
                                continue;
                            }
                            if !output.is_empty() {
                                output.push('\n');
                            }
                            output.push_str(&l);
                            stdout_lines += 1;
                        }
                        Ok(None) => {}
                        Err(e) => {
                            error_output.push_str(&format!("\nStdout read error: {}", e));
                        }
                    }
                }
                line = stderr_reader.next_line() => {
                    match line {
                        Ok(Some(l)) => {
                            // Check truncation limits
                            if stderr_truncated {
                                continue; // Skip remaining lines
                            }
                            if stderr_lines >= MAX_OUTPUT_LINES || error_output.len() + l.len() > MAX_OUTPUT_SIZE {
                                stderr_truncated = true;
                                continue;
                            }
                            if !error_output.is_empty() {
                                error_output.push('\n');
                            }
                            error_output.push_str(&l);
                            stderr_lines += 1;
                        }
                        Ok(None) => {}
                        Err(e) => {
                            error_output.push_str(&format!("\nStderr read error: {}", e));
                        }
                    }
                }
                status = child.wait() => {
                    match status {
                        Ok(exit_status) => {
                            let mut result = output;

                            // Add truncation notice for stdout
                            if stdout_truncated {
                                result.push_str(&format!(
                                    "\n\n... (stdout truncated at {} lines / {}KB)",
                                    stdout_lines,
                                    MAX_OUTPUT_SIZE / 1024
                                ));
                            }

                            if !error_output.is_empty() {
                                if !result.is_empty() {
                                    result.push('\n');
                                }
                                result.push_str(&error_output);

                                // Add truncation notice for stderr
                                if stderr_truncated {
                                    result.push_str(&format!(
                                        "\n\n... (stderr truncated at {} lines / {}KB)",
                                        stderr_lines,
                                        MAX_OUTPUT_SIZE / 1024
                                    ));
                                }
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
