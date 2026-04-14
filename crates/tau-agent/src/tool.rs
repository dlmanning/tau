//! Tool trait and execution

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tau_ai::{Content, Message};
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;

use crate::events::AgentEvent;

/// Send an event on a broadcast channel, ignoring errors (no receivers).
pub(crate) fn send_event(tx: &broadcast::Sender<AgentEvent>, event: AgentEvent) {
    let _ = tx.send(event);
}

/// Tracks which files have been read so write/edit tools can enforce
/// a read-before-write policy. Shared via `Arc<Mutex<...>>` on `ExecutionContext`.
#[derive(Default)]
pub struct FileAccessTracker {
    read_files: HashSet<PathBuf>,
}

impl FileAccessTracker {
    /// Record that a file has been successfully read.
    pub fn mark_read(&mut self, path: impl Into<PathBuf>) {
        self.read_files.insert(path.into());
    }

    /// Check that a file has been read before writing. Returns `Ok(())` if the
    /// file doesn't exist yet (new file) or has been read. Returns an error
    /// message if the file exists but hasn't been read.
    pub fn require_read(&self, path: &Path) -> Result<(), String> {
        if path.exists() && !self.read_files.contains(path) {
            Err("You must read this file before editing it. Use the read tool first.".to_string())
        } else {
            Ok(())
        }
    }

    /// Clear all tracked reads.
    pub fn clear(&mut self) {
        self.read_files.clear();
    }

    /// Rebuild from conversation history (for session restore).
    pub fn rebuild_from_messages(&mut self, messages: &[Message], cwd: &Option<PathBuf>) {
        self.read_files.clear();
        for msg in messages {
            if let Message::Assistant { content, .. } = msg {
                for c in content {
                    if let Content::ToolCall {
                        name,
                        arguments,
                        id,
                    } = c
                    {
                        if name == "read" {
                            let has_success = messages.iter().any(|m| {
                                matches!(m, Message::ToolResult { tool_call_id, is_error, .. }
                                    if tool_call_id == id && !is_error)
                            });
                            if has_success {
                                if let Some(path) = resolve_tool_path(arguments, cwd) {
                                    self.read_files.insert(path);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Resolve a file path from tool arguments, handling `~/`, relative paths,
/// and canonicalization.
fn resolve_tool_path(args: &Value, cwd: &Option<PathBuf>) -> Option<PathBuf> {
    let path_str = args.get("path").and_then(|v| v.as_str())?;
    let path = if let Some(rest) = path_str.strip_prefix("~/") {
        dirs::home_dir()
            .map(|h| h.join(rest))
            .unwrap_or_else(|| PathBuf::from(path_str))
    } else if path_str == "~" {
        dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
    } else {
        let p = PathBuf::from(path_str);
        if p.is_absolute() {
            p
        } else {
            let base = cwd
                .clone()
                .or_else(|| std::env::current_dir().ok())
                .unwrap_or_default();
            base.join(p)
        }
    };
    Some(std::fs::canonicalize(&path).unwrap_or(path))
}

/// Result of a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Content to return to the LLM
    pub content: Vec<Content>,
    /// Whether the execution resulted in an error
    pub is_error: bool,
    /// Optional structured details (for UI rendering)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

impl ToolResult {
    /// Create a successful text result
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![Content::text(text)],
            is_error: false,
            details: None,
        }
    }

    /// Create an error result
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![Content::text(message)],
            is_error: true,
            details: None,
        }
    }

    /// Create a result with multiple content blocks
    pub fn with_content(content: Vec<Content>) -> Self {
        Self {
            content,
            is_error: false,
            details: None,
        }
    }

    /// Add details to the result
    pub fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }

    /// Get the text content as a single string
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| c.as_text())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A sender for tool progress updates during execution.
///
/// Tools can use this to emit `ToolExecutionUpdate` events while running.
#[derive(Clone)]
pub struct ProgressSender {
    tx: broadcast::Sender<AgentEvent>,
    tool_call_id: String,
    tool_name: String,
}

impl ProgressSender {
    /// Create a new progress sender for a specific tool invocation.
    pub fn new(
        tx: broadcast::Sender<AgentEvent>,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> Self {
        Self {
            tx,
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
        }
    }

    /// Send a progress update.
    pub fn send(&self, content: impl Into<String>) {
        send_event(
            &self.tx,
            AgentEvent::ToolExecutionUpdate {
                tool_call_id: self.tool_call_id.clone(),
                tool_name: self.tool_name.clone(),
                content: content.into(),
            },
        );
    }

    /// Emit a raw event on the parent's event channel.
    /// Used by the agent tool to forward subagent events (e.g. TurnEnd for usage).
    pub fn emit(&self, event: AgentEvent) {
        send_event(&self.tx, event);
    }
}

/// Whether a tool can run concurrently with others.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Concurrency {
    /// Must run alone (default for most tools).
    Sequential,
    /// Safe to run concurrently with other Parallel tools.
    Parallel,
}

/// Context passed to every tool execution.
pub struct ExecutionContext {
    /// Working directory for this execution. Tools resolve relative paths against this.
    pub cwd: PathBuf,
    /// Cancellation token — tools should check this periodically for long operations.
    pub cancel: CancellationToken,
    /// Progress sender — tools can emit updates during execution.
    pub progress: ProgressSender,
    /// Channel for tools that need user input (e.g. AskUserQuestion).
    /// `None` in non-interactive contexts or subagents.
    pub interaction: Option<tokio::sync::mpsc::Sender<crate::interaction::InteractionRequest>>,
    /// File access tracker for read-before-write policy.
    pub file_access: Arc<Mutex<FileAccessTracker>>,
}

impl ExecutionContext {
    /// Resolve a path string against the working directory.
    /// Handles `~/`, relative paths, and absolute passthrough.
    pub fn resolve_path(&self, path_str: &str) -> PathBuf {
        if let Some(rest) = path_str.strip_prefix("~/") {
            dirs::home_dir()
                .map(|h| h.join(rest))
                .unwrap_or_else(|| PathBuf::from(path_str))
        } else if path_str == "~" {
            dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
        } else {
            let p = PathBuf::from(path_str);
            if p.is_absolute() { p } else { self.cwd.join(p) }
        }
    }

    /// Mark a path as read in the file access tracker.
    pub fn mark_read(&self, path: &Path) {
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.file_access.lock().mark_read(canonical);
    }

    /// Check read-before-write policy. Returns error if file exists but hasn't been read.
    pub fn require_read(&self, path: &Path) -> Result<(), String> {
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.file_access.lock().require_read(&canonical)
    }
}

/// Trait for executable tools
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (used in API calls)
    fn name(&self) -> &str;

    /// Human-readable label for UI
    fn label(&self) -> &str {
        self.name()
    }

    /// Tool description for the LLM
    fn description(&self) -> &str;

    /// JSON Schema for parameters
    fn parameters_schema(&self) -> Value;

    /// Whether this tool can run concurrently with other Parallel tools.
    /// Defaults to `Parallel`. Override to `Sequential` for tools that
    /// mutate files, require UI exclusivity, or have side effects.
    fn concurrency(&self) -> Concurrency {
        Concurrency::Parallel
    }

    /// Short, human-readable description of what this invocation is doing,
    /// shown in the TUI while the tool executes (e.g. "Reading main.rs").
    /// Default: "Running {name}".
    fn activity_description(&self, _arguments: &Value) -> String {
        format!("Running {}", self.name())
    }

    /// Execute the tool with the given arguments and execution context.
    async fn execute(&self, arguments: Value, ctx: ExecutionContext) -> ToolResult;
}

/// Type alias for a boxed tool
pub type BoxedTool = Arc<dyn Tool>;

/// Convert a Tool to a tau_ai::Tool for API calls
pub fn to_api_tool(tool: &dyn Tool) -> tau_ai::Tool {
    tau_ai::Tool {
        name: tool.name().to_string(),
        description: tool.description().to_string(),
        parameters: tool.parameters_schema(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "Echoes input"
        }
        fn parameters_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string" }
                }
            })
        }
        async fn execute(&self, arguments: Value, _ctx: ExecutionContext) -> ToolResult {
            let text = arguments
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("(empty)");
            ToolResult::text(text)
        }
    }

    #[tokio::test]
    async fn test_execute_with_context() {
        let tool = EchoTool;
        let (tx, _rx) = broadcast::channel(16);
        let progress = ProgressSender::new(tx, "call_1", "echo");
        let cancel = CancellationToken::new();
        let args = serde_json::json!({"text": "hello"});

        let ctx = ExecutionContext {
            cwd: PathBuf::from("/tmp"),
            cancel,
            progress,
            interaction: None,
            file_access: Arc::new(Mutex::new(FileAccessTracker::default())),
        };

        let result = tool.execute(args, ctx).await;

        assert!(!result.is_error);
        assert_eq!(result.text_content(), "hello");
    }

    #[tokio::test]
    async fn test_progress_sender_emits_events() {
        let (tx, mut rx) = broadcast::channel(16);
        let sender = ProgressSender::new(tx, "call_42", "bash");

        sender.send("50% complete");
        sender.send("done");

        let event1 = rx.recv().await.unwrap();
        let event2 = rx.recv().await.unwrap();

        match event1 {
            AgentEvent::ToolExecutionUpdate {
                tool_call_id,
                tool_name,
                content,
            } => {
                assert_eq!(tool_call_id, "call_42");
                assert_eq!(tool_name, "bash");
                assert_eq!(content, "50% complete");
            }
            other => panic!("expected ToolExecutionUpdate, got {:?}", other),
        }

        match event2 {
            AgentEvent::ToolExecutionUpdate { content, .. } => {
                assert_eq!(content, "done");
            }
            other => panic!("expected ToolExecutionUpdate, got {:?}", other),
        }
    }

    #[test]
    fn test_tool_result_text() {
        let r = ToolResult::text("ok");
        assert!(!r.is_error);
        assert_eq!(r.text_content(), "ok");
    }

    #[test]
    fn test_tool_result_error() {
        let r = ToolResult::error("bad");
        assert!(r.is_error);
        assert_eq!(r.text_content(), "bad");
    }

    #[test]
    fn test_to_api_tool() {
        let tool = EchoTool;
        let api_tool = to_api_tool(&tool);
        assert_eq!(api_tool.name, "echo");
        assert_eq!(api_tool.description, "Echoes input");
    }

    #[test]
    fn test_file_access_new_file_allowed() {
        let tracker = FileAccessTracker::default();
        let result = tracker.require_read(Path::new("/nonexistent/path/to/file.txt"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_file_access_read_then_write() {
        let mut tracker = FileAccessTracker::default();
        let path = PathBuf::from("/tmp/test-file-access-tracker-agent2");
        std::fs::write(&path, "test").ok();
        let result = tracker.require_read(&path);
        assert!(result.is_err());
        tracker.mark_read(path.clone());
        let result = tracker.require_read(&path);
        assert!(result.is_ok());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_file_access_clear() {
        let mut tracker = FileAccessTracker::default();
        tracker.mark_read("/some/path");
        tracker.clear();
        assert!(tracker.read_files.is_empty());
    }

    fn make_ctx(cwd: &str) -> ExecutionContext {
        let (tx, _rx) = broadcast::channel(16);
        ExecutionContext {
            cwd: PathBuf::from(cwd),
            cancel: CancellationToken::new(),
            progress: ProgressSender::new(tx, "test", "test"),
            interaction: None,
            file_access: Arc::new(Mutex::new(FileAccessTracker::default())),
        }
    }

    #[test]
    fn test_resolve_path_absolute() {
        let ctx = make_ctx("/work");
        assert_eq!(
            ctx.resolve_path("/usr/bin/ls"),
            PathBuf::from("/usr/bin/ls")
        );
    }

    #[test]
    fn test_resolve_path_relative() {
        let ctx = make_ctx("/work");
        assert_eq!(
            ctx.resolve_path("src/main.rs"),
            PathBuf::from("/work/src/main.rs")
        );
    }

    #[test]
    fn test_resolve_path_tilde_prefix() {
        let ctx = make_ctx("/work");
        let resolved = ctx.resolve_path("~/Documents/file.txt");
        let home = dirs::home_dir().expect("home dir exists in test");
        assert_eq!(resolved, home.join("Documents/file.txt"));
    }

    #[test]
    fn test_resolve_path_tilde_alone() {
        let ctx = make_ctx("/work");
        let resolved = ctx.resolve_path("~");
        let home = dirs::home_dir().expect("home dir exists in test");
        assert_eq!(resolved, home);
    }
}
