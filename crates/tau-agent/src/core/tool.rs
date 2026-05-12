//! `Tool` trait, [`ExecutionContext`], [`ToolResult`].
//!
//! Tools are stateless w.r.t. the agent. Identity (the owning agent's
//! id), per-call cancellation, the progress event sender, and the
//! interaction channel all flow through [`ExecutionContext`] each call.
//! There is **no** post-construction binding hook — anything a tool
//! needs to know about its caller arrives via this struct.

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

use crate::core::approval::ToolRisk;
use crate::types::events::{AgentEvent, ConsoleLevel, ConsoleLine};

/// Helper: send an event on a broadcast channel; ignore "no subscribers".
pub(crate) fn send_event(tx: &broadcast::Sender<AgentEvent>, event: AgentEvent) {
    let _ = tx.send(event);
}

/// Tracks which files have been read so write/edit tools can enforce a
/// read-before-write policy. Shared via `Arc<Mutex<...>>` on
/// [`ExecutionContext`].
#[derive(Default)]
pub struct FileAccessTracker {
    read_files: HashSet<PathBuf>,
}

impl FileAccessTracker {
    pub fn mark_read(&mut self, path: impl Into<PathBuf>) {
        self.read_files.insert(path.into());
    }

    /// Returns `Ok(())` if the file doesn't exist (new file) or has
    /// been read. Returns an error message otherwise.
    pub fn require_read(&self, path: &Path) -> Result<(), String> {
        if path.exists() && !self.read_files.contains(path) {
            Err("You must read this file before editing it. Use the read tool first.".into())
        } else {
            Ok(())
        }
    }

    pub fn clear(&mut self) {
        self.read_files.clear();
    }

    /// Rebuild the tracker from a conversation history (for session
    /// restore). A successful `read` tool call marks its `path` as read.
    pub fn rebuild_from_messages(&mut self, messages: &[Message], cwd: &Option<PathBuf>) {
        self.read_files.clear();
        for msg in messages {
            let Message::Assistant { content, .. } = msg else {
                continue;
            };
            for c in content {
                let Content::ToolCall {
                    name,
                    arguments,
                    id,
                } = c
                else {
                    continue;
                };
                if name != "read" {
                    continue;
                }
                let succeeded = messages.iter().any(|m| {
                    matches!(m, Message::ToolResult { tool_call_id, is_error, .. }
                        if tool_call_id == id && !is_error)
                });
                if !succeeded {
                    continue;
                }
                if let Some(path) = resolve_tool_path(arguments, cwd) {
                    self.read_files.insert(path);
                }
            }
        }
    }
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: Vec<Content>,
    pub is_error: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

impl ToolResult {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![Content::text(text)],
            is_error: false,
            details: None,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![Content::text(message)],
            is_error: true,
            details: None,
        }
    }

    pub fn with_content(content: Vec<Content>) -> Self {
        Self {
            content,
            is_error: false,
            details: None,
        }
    }

    pub fn with_details(mut self, details: Value) -> Self {
        self.details = Some(details);
        self
    }

    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| c.as_text())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Per-call progress sender. Tools emit `ToolExecutionUpdate` events
/// while running.
#[derive(Clone)]
pub struct ProgressSender {
    tx: broadcast::Sender<AgentEvent>,
    tool_call_id: String,
    tool_name: String,
}

impl ProgressSender {
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

    pub fn tool_call_id(&self) -> &str {
        &self.tool_call_id
    }
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    pub fn send(&self, content: impl Into<String>) {
        self.send_at(content, ConsoleLevel::Normal);
    }

    pub fn send_at(&self, content: impl Into<String>, level: ConsoleLevel) {
        self.send_lines(vec![ConsoleLine::new(content, level)]);
    }

    pub fn send_lines(&self, lines: Vec<ConsoleLine>) {
        if lines.is_empty() {
            return;
        }
        send_event(
            &self.tx,
            AgentEvent::ToolExecutionUpdate {
                tool_call_id: self.tool_call_id.clone(),
                tool_name: self.tool_name.clone(),
                lines,
            },
        );
    }

    /// Emit a raw event (e.g. for forwarding subagent events from
    /// inside a tool that ran a child agent).
    pub fn emit(&self, event: AgentEvent) {
        send_event(&self.tx, event);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Concurrency {
    Sequential,
    Parallel,
}

/// Per-execution context. **No `AgentHandle` here** — tools that need
/// identity read [`Self::agent_id`]; tools that need to drive other
/// agents (e.g. recursive `AgentTool`) reach for the manager via
/// whatever weak reference they hold.
pub struct ExecutionContext {
    pub cwd: PathBuf,
    pub cancel: CancellationToken,
    pub progress: ProgressSender,
    pub interaction:
        Option<tokio::sync::mpsc::Sender<crate::core::interaction::InteractionRequest>>,
    pub file_access: Arc<Mutex<FileAccessTracker>>,
    /// Identity of the agent running this tool, when it has one.
    /// `None` for builder-spawned root agents that haven't been adopted
    /// by an `AgentManager`.
    pub agent_id: Option<String>,
    /// Depth in the subagent spawn tree: `0` for the host's root agent,
    /// incremented by one on each spawn. Recursive tools (e.g.
    /// `AgentTool`) read this to enforce host-side recursion limits
    /// without needing per-depth instances.
    pub subagent_depth: u32,
}

impl ExecutionContext {
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

    pub fn mark_read(&self, path: &Path) {
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.file_access.lock().mark_read(canonical);
    }

    pub fn require_read(&self, path: &Path) -> Result<(), String> {
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.file_access.lock().require_read(&canonical)
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;

    fn label(&self) -> &str {
        self.name()
    }

    fn description(&self) -> &str;

    fn parameters_schema(&self) -> Value;

    fn concurrency(&self) -> Concurrency {
        Concurrency::Parallel
    }

    fn activity_description(&self, _arguments: &Value) -> String {
        format!("Running {}", self.name())
    }

    fn risk(&self, _arguments: &Value) -> ToolRisk {
        ToolRisk::Local
    }

    async fn execute(&self, arguments: Value, ctx: ExecutionContext) -> ToolResult;
}

pub type BoxedTool = Arc<dyn Tool>;

pub fn to_api_tool(tool: &dyn Tool) -> tau_ai::Tool {
    tau_ai::Tool {
        name: tool.name().to_string(),
        description: tool.description().to_string(),
        parameters: tool.parameters_schema(),
    }
}
