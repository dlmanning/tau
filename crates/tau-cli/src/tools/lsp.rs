//! LSP code intelligence tool

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tau_agent::tool::{Tool, ToolResult};
use tokio_util::sync::CancellationToken;

use crate::lsp::LspManager;

/// Tool for querying language servers (go-to-definition, find-references, hover, etc.)
pub struct LspTool {
    manager: Arc<LspManager>,
}

impl LspTool {
    pub fn new(manager: Arc<LspManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for LspTool {
    fn name(&self) -> &str {
        "lsp"
    }

    fn description(&self) -> &str {
        "Query language servers for code intelligence. Supports go-to-definition, \
         find-references, hover (type info), and document symbols. Requires a language \
         server to be installed (e.g. rust-analyzer, typescript-language-server, pyright, gopls)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["goToDefinition", "findReferences", "hover", "documentSymbol"],
                    "description": "The LSP operation to perform"
                },
                "filePath": {
                    "type": "string",
                    "description": "Absolute path to the file"
                },
                "line": {
                    "type": "integer",
                    "description": "Line number (1-based). Required for all operations except documentSymbol."
                },
                "character": {
                    "type": "integer",
                    "description": "Character offset on the line (1-based). Required for all operations except documentSymbol."
                }
            },
            "required": ["operation", "filePath"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: &str,
        arguments: serde_json::Value,
        _cancel: CancellationToken,
    ) -> ToolResult {
        let operation = match arguments.get("operation").and_then(|v| v.as_str()) {
            Some(op) => op,
            None => return ToolResult::error("Missing 'operation' argument"),
        };

        let file_path = match arguments.get("filePath").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolResult::error("Missing 'filePath' argument"),
        };

        let path = PathBuf::from(file_path);
        if !path.is_absolute() {
            return ToolResult::error("filePath must be an absolute path");
        }
        if !path.exists() {
            return ToolResult::error(format!("File not found: {}", file_path));
        }

        let line = arguments
            .get("line")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);
        let character = arguments
            .get("character")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let result = match operation {
            "goToDefinition" => {
                let (line, character) = match (line, character) {
                    (Some(l), Some(c)) => (l, c),
                    _ => return ToolResult::error("goToDefinition requires 'line' and 'character'"),
                };
                self.manager.go_to_definition(&path, line, character).await
            }
            "findReferences" => {
                let (line, character) = match (line, character) {
                    (Some(l), Some(c)) => (l, c),
                    _ => return ToolResult::error("findReferences requires 'line' and 'character'"),
                };
                self.manager.find_references(&path, line, character).await
            }
            "hover" => {
                let (line, character) = match (line, character) {
                    (Some(l), Some(c)) => (l, c),
                    _ => return ToolResult::error("hover requires 'line' and 'character'"),
                };
                self.manager.hover(&path, line, character).await
            }
            "documentSymbol" => self.manager.document_symbol(&path).await,
            _ => {
                return ToolResult::error(format!(
                    "Unknown operation '{}'. Valid: goToDefinition, findReferences, hover, documentSymbol",
                    operation
                ))
            }
        };

        match result {
            Ok(text) => ToolResult::text(text),
            Err(e) => ToolResult::error(format!("LSP error: {}", e)),
        }
    }
}
